# fileName: extractor.py
"""
Video extraction adapter for Clip Tool v4.

The extractor keeps the pipeline contract intact:
- input: clips_plan from selector
- output: job_dir/clips/{clip_name}_raw.mp4
- next stage: captioner.py can continue using the same naming convention

All crop/template/speaker-render logic lives in pipeline.speaker_tracking, which
is now the non-GUI Clip Tool v4 engine plus a small pipeline adapter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import config
import threading
from pipeline import speaker_tracking

MIN_VALID_CLIP_BYTES = 4096


def _log(logger: Any, level: str, message: str) -> None:
    try:
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(message)
        elif callable(logger):
            logger(message)
        else:
            print(message)
    except Exception:
        try:
            print(message)
        except Exception:
            pass


def _read_meta_video_path(job_dir: Path) -> Path:
    meta_path = job_dir / "meta.json"
    video_filename = "input_video.mp4"
    meta = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            video_filename = meta.get("video_filename", video_filename)
        except Exception:
            pass

    resolved_path = job_dir / video_filename
    
    # Try resolving via title from metadata if the default doesn't exist
    if not resolved_path.exists() and "title" in meta:
        import re
        clean_title = re.sub(r'[<>:"/\\|?*]', '', meta["title"])[:50].strip()
        if clean_title:
            title_filename = f"{clean_title}.mp4"
            title_path = job_dir / title_filename
            if title_path.exists():
                resolved_path = title_path
                video_filename = title_filename

    # If the file still doesn't exist, search the directory for any .mp4 file
    if not resolved_path.exists():
        try:
            for fn in os.listdir(job_dir):
                if fn.endswith(".mp4") and not fn.startswith("_") and not fn.endswith("_raw.mp4") and not fn.endswith("_captioned.mp4"):
                    candidate_path = job_dir / fn
                    if candidate_path.is_file():
                        resolved_path = candidate_path
                        break
        except Exception:
            pass

    return resolved_path



def _normalise_resume_marker(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_resume_start(clips_plan: List[Dict[str, Any]], marker: Any, logger: Any) -> int:
    marker = _normalise_resume_marker(marker)
    if not marker:
        return 0

    if marker.isdigit():
        idx = max(0, int(marker) - 1)
        if idx >= len(clips_plan):
            _log(logger, "warning", f"Extraction resume marker {marker!r} is beyond the clips plan; nothing new to extract")
        else:
            _log(logger, "info", f"Extraction resume marker {marker!r}: starting at clip #{idx + 1}")
        return idx

    marker_l = marker.lower()
    for idx, clip in enumerate(clips_plan):
        if str(clip.get("clip_name", "")).lower() == marker_l:
            _log(logger, "info", f"Extraction resume marker {marker!r}: starting at {clip.get('clip_name')}")
            return idx

    _log(logger, "warning", f"Extraction resume marker {marker!r} did not match any clip; using normal resume-by-existing-output")
    return 0


def _valid_clip(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > MIN_VALID_CLIP_BYTES
    except Exception:
        return False


def _clip_segments(clip: Dict[str, Any]) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    for seg in clip.get("segments", []) or []:
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            continue
        if end > start:
            segments.append((start, end))
    return segments


def _setting(settings: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in settings and settings[name] not in (None, ""):
            return settings[name]
    return default


def _manual_templates_path(job_dir: Path, settings: Dict[str, Any]) -> Optional[str]:
    configured = _setting(settings, "manual_templates_path", "v4_manual_templates_path", default=None)
    candidates = [
        configured,
        getattr(config, "V4_MANUAL_TEMPLATES_PATH", ""),
        job_dir / "manual_templates.json",
        job_dir / "templates" / "manual_templates.json",
        job_dir / "files" / "manual_templates.json",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and Path(text).exists():
            return text
    return None


def _delete_manual_templates(job_dir: Path, logger: Any) -> None:
    """Remove any job-local manual_templates.json so the authoring tool
    re-snaps fresh. Only called when the user explicitly forces a re-snap
    (force_template_resnap); normal runs and resumes reuse existing templates.
    Externally-configured template paths are left untouched."""
    for rel in (
        "manual_templates.json",
        "templates/manual_templates.json",
        "files/manual_templates.json",
    ):
        p = job_dir / rel
        try:
            if p.exists():
                p.unlink()
                _log(logger, "info", f"Cleared stale manual templates for re-snap: {p}")
        except OSError as exc:
            _log(logger, "warning", f"Could not delete stale manual templates {p}: {exc}")


def _bool_setting(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _template_tool_path(job_dir: Path, settings: Dict[str, Any]) -> Optional[Path]:
    configured = _setting(settings, "template_tool_path", "v4_template_tool_path", default=None)
    candidates = [
        configured,
        getattr(config, "V4_TEMPLATE_TOOL_PATH", ""),
        Path(__file__).with_name("template_tool.py"),
        job_dir / "files" / "template_tool.py",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and Path(text).exists():
            return Path(text)
    return None


def _maybe_launch_template_tool(
    job_dir: Path,
    video_path: Path,
    settings: Dict[str, Any],
    logger: Any,
) -> Optional[str]:
    auto_launch = _bool_setting(
        _setting(
            settings,
            "template_tool_auto_launch",
            "v4_template_tool_auto_launch",
            default=getattr(config, "V4_TEMPLATE_TOOL_AUTO_LAUNCH", True),
        ),
        True,
    )
    if not auto_launch:
        return None

    tool_path = _template_tool_path(job_dir, settings)
    if tool_path is None:
        _log(logger, "warning", "Clip Tool v4 template authoring tool not found; auto discovery will be used")
        return None

    output_path = job_dir / "manual_templates.json"
    err_log = output_path.with_name("template_tool_error.log")
    try:
        if err_log.exists():
            err_log.unlink()
    except OSError:
        pass
    _log(logger, "info", f"Clip Tool v4 launching template authoring tool: {tool_path}")
    _log(logger, "info", f"Clip Tool v4 manual templates will be saved to: {output_path}")

    # Force UTF-8 in the child so a non-ASCII filename/progress glyph can't crash
    # it with a charmap error, and give it its OWN console+window on Windows so
    # the cv2 CV screen reliably appears even when the pipeline was started
    # without an interactive console (a detached child silently exits 1).
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0

    try:
        result = subprocess.run(
            [sys.executable, str(tool_path), "--video", str(video_path), "--output", str(output_path)],
            cwd=str(tool_path.parent),
            stdin=None,
            env=env,
            creationflags=creationflags,
        )
    except Exception as exc:
        _log(logger, "warning", f"Clip Tool v4 template tool launch failed: {exc}; auto discovery will be used")
        return None

    if result.returncode != 0:
        detail = ""
        try:
            if err_log.exists():
                detail = err_log.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        if detail:
            _log(
                logger, "warning",
                f"Clip Tool v4 manual CV screen failed (exit {result.returncode}) — auto "
                f"discovery will be used. Crash detail:\n{detail}"
            )
        else:
            _log(
                logger, "warning",
                f"Clip Tool v4 manual CV screen exited {result.returncode} with no crash "
                f"log — likely a GUI/display problem (e.g. headless OpenCV, or no desktop "
                f"session). Auto discovery will be used."
            )
        return None

    if output_path.exists():
        return str(output_path)
    _log(
        logger, "warning",
        "Clip Tool v4 template tool exited cleanly but wrote no templates "
        "(closed without snapping?); auto discovery will be used"
    )
    return None


def _diarization_path(job_dir: Path, settings: Dict[str, Any]) -> Optional[str]:
    configured = _setting(settings, "diarization_path", "v4_diarization_path", default=None)
    candidates = [
        configured,
        getattr(config, "V4_DIARIZATION_PATH", ""),
        job_dir / "diarization.json",
        job_dir / "diarization" / "diarization.json",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and Path(text).exists():
            return text
    return None


def _render_segment(
    video_path: Path,
    job_dir: Path,
    out_path: Path,
    start: float,
    end: float,
    settings: Dict[str, Any],
    logger: Any,
    manual_templates_path: Optional[str],
    diarization_path: Optional[str],
    gpu_lock: Optional[threading.Semaphore] = None,
) -> bool:
    speaker_filter = _setting(
        settings,
        "v4_speaker_filter",
        "speaker_filter",
        default=getattr(config, "V4_SPEAKER_FILTER", "continuous"),
    )
    clear_template_cache = bool(_setting(
        settings,
        "v4_clear_template_cache_each_run",
        default=getattr(config, "V4_CLEAR_TEMPLATE_CACHE_EACH_RUN", True),
    ))
    diar_py = _setting(settings, "v4_diar_python", default=getattr(config, "V4_DIAR_PYTHON", None))
    ffmpeg_bin = _setting(settings, "ffmpeg_path", default=getattr(config, "FFMPEG_PATH", None))

    try:
        src_w, src_h, _ = speaker_tracking.get_video_info(str(video_path))
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"Invalid video dimensions: {src_w}x{src_h}")
        
        aspect = settings.get("aspect_ratio", config.DEFAULT_ASPECT_RATIO)
        if aspect == "9:16":
            if src_h > src_w:
                # Parent is already a vertical video
                target_res = (src_w, src_h)
            else:
                # Parent is landscape; vertical crop matches native source height
                target_w = int(src_h * 9 / 16)
                # Ensure width is divisible by 2 for encoders
                target_w = (target_w // 2) * 2
                target_res = (target_w, src_h)
        elif aspect == "16:9":
            if src_w > src_h:
                # Parent is already a landscape video
                target_res = (src_w, src_h)
            else:
                # Parent is vertical; landscape crop matches native source width
                target_h = int(src_w * 9 / 16)
                # Ensure height is divisible by 2 for encoders
                target_h = (target_h // 2) * 2
                target_res = (src_w, target_h)
        else:
            # Fallback for other aspect ratios is parent resolution
            target_res = (src_w, src_h)
    except Exception:
        aspect = settings.get("aspect_ratio", config.DEFAULT_ASPECT_RATIO)
        if aspect == "9:16":
            target_res = getattr(config, "VERTICAL_RESOLUTION", (1080, 1920))
        else:
            target_res = getattr(config, "HORIZONTAL_RESOLUTION", (1920, 1080))

    return speaker_tracking.render_pipeline_clip_v4(
        video_path=str(video_path),
        start_sec=start,
        end_sec=end,
        out_path=str(out_path),
        pipeline_job_dir=job_dir,
        logger=logger,
        speaker_filter=str(speaker_filter or "continuous"),
        manual_templates_path=manual_templates_path,
        diarization_path=diarization_path,
        ffmpeg_bin=ffmpeg_bin,
        diar_py=diar_py,
        target_resolution=target_res,
        clear_template_cache=clear_template_cache,
        gpu_lock=gpu_lock,
    )


def _concat_files(parts: List[Path], out_path: Path, logger: Any) -> bool:
    if not parts:
        return False

    # Crossfade the AI-segment joins when enabled. Each join overlaps by
    # XFADE_FRAMES/fps, so the rendered clip is shorter than the sum of its parts;
    # caption localisation compensates via the rendered-duration manifest. The
    # stream-copy path below remains the robust fallback if the xfade encode fails.
    if len(parts) > 1 and bool(getattr(config, "CLIP_JOIN_XFADE", True)):
        try:
            ffmpeg_bin = str(getattr(config, "FFMPEG_PATH", "ffmpeg"))
            try:
                fps = float(speaker_tracking.get_video_info(str(parts[0]))[2])
            except Exception:
                fps = 0.0
            ok, err = speaker_tracking._ffmpeg_xfade_concat(
                [str(p) for p in parts], str(out_path), ffmpeg_bin, fps=max(1.0, fps or 30.0)
            )
            if ok and _valid_clip(out_path):
                return True
            _log(logger, "warning", f"xfade concat failed for {out_path.name} ({(err or 'invalid output')[-500:]}); falling back to stream-copy concat")
        except Exception as exc:
            _log(logger, "warning", f"xfade concat error for {out_path.name}: {exc}; falling back to stream-copy concat")

    concat_list = out_path.with_name(out_path.stem + "_concat.txt")
    try:
        with concat_list.open("w", encoding="utf-8") as f:
            for part in parts:
                safe = str(part.resolve()).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = [
            str(getattr(config, "FFMPEG_PATH", "ffmpeg")),
            "-y",
            "-nostdin",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            _log(logger, "error", f"FFmpeg concat failed for {out_path.name}: {result.stderr[-2000:]}")
            return False
        return _valid_clip(out_path)
    finally:
        try:
            concat_list.unlink(missing_ok=True)
        except Exception:
            pass


def _cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception:
            pass


def _sweep_clip_workdirs(clips_dir: Path, logger: Any) -> None:
    """Delete every per-render working dir (``*_job``) and temp part leftover
    (``_temp_*``) in the clips folder so it ends up holding only final outputs.
    Called once extraction finishes; safe because these are regenerated fresh
    each run (scene-cut caches included)."""
    removed = 0
    try:
        targets = list(clips_dir.glob("*_job")) + list(clips_dir.glob("_temp_*"))
    except Exception:
        return
    for p in targets:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    if removed:
        _log(logger, "info", f"Cleaned {removed} temp/work item(s) from clips/")


def _probe_duration(path: Path) -> float:
    """Container duration in seconds, or 0.0 if it can't be read."""
    try:
        ffmpeg = str(getattr(config, "FFMPEG_PATH", "ffmpeg"))
        base = os.path.dirname(ffmpeg)
        probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        ffprobe = os.path.join(base, probe_name) if base else probe_name
        res = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return float((res.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _write_render_manifest(
    out_path: Path,
    clip_name: str,
    segments: List[Tuple[float, float]],
    rendered_durs: List[float],
    logger: Any,
) -> None:
    """Record the actual playback start-offset of each AI-segment in the final
    clip, so caption localisation can compensate for crossfade overlap instead of
    assuming output_duration == sum(source segment durations). Best effort: any
    failure leaves captioning on the additive (source-duration) fallback path."""
    try:
        if not rendered_durs or len(rendered_durs) != len(segments):
            return
        xfade_join = 0.0
        if len(rendered_durs) > 1 and bool(getattr(config, "CLIP_JOIN_XFADE", True)):
            try:
                fps = float(speaker_tracking.get_video_info(str(out_path))[2]) or 30.0
            except Exception:
                fps = 30.0
            xfade_join = max(1, int(getattr(config, "XFADE_FRAMES", 5))) / max(1.0, fps)
        offsets: List[float] = []
        acc = 0.0
        n = len(rendered_durs)
        for i, d in enumerate(rendered_durs):
            offsets.append(round(acc, 4))
            acc += max(0.0, float(d))
            if i < n - 1:
                acc -= xfade_join
        manifest = {
            "clip_name": clip_name,
            "xfade_join": round(xfade_join, 4),
            "segment_source": [[round(float(s), 4), round(float(e), 4)] for (s, e) in segments],
            "segment_rendered": [round(float(d), 4) for d in rendered_durs],
            "segment_offsets": offsets,
            "total_rendered": round(max(0.0, acc), 4),
        }
        mpath = out_path.with_name(out_path.stem + ".render.json")
        with mpath.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
    except Exception as exc:
        _log(logger, "warning", f"render manifest write failed for {clip_name}: {exc}")


def _extract_single_clip(
    video_path: Path,
    job_dir: Path,
    clip_name: str,
    segments: List[Tuple[float, float]],
    out_path: Path,
    settings: Dict[str, Any],
    logger: Any,
    manual_templates_path: Optional[str],
    diarization_path: Optional[str],
    gpu_lock: Optional[threading.Semaphore] = None,
) -> bool:
    if len(segments) == 1:
        start, end = segments[0]
        ok = _render_segment(
            video_path,
            job_dir,
            out_path,
            start,
            end,
            settings,
            logger,
            manual_templates_path,
            diarization_path,
            gpu_lock=gpu_lock,
        ) and _valid_clip(out_path)
        if ok:
            _write_render_manifest(out_path, clip_name, segments, [_probe_duration(out_path)], logger)
        return ok

    parts: List[Path] = []
    rendered_durs: List[float] = []
    for idx, (start, end) in enumerate(segments, start=1):
        part_path = out_path.with_name(f"_temp_{clip_name}_part_{idx:03d}.mp4")
        _log(logger, "info", f"  -> rendering concat part {idx}/{len(segments)} for {clip_name}: {start:.2f}-{end:.2f}s")
        ok = _render_segment(
            video_path,
            job_dir,
            part_path,
            start,
            end,
            settings,
            logger,
            manual_templates_path,
            diarization_path,
            gpu_lock=gpu_lock,
        )
        if not ok or not _valid_clip(part_path):
            _log(logger, "error", f"  -> {clip_name}: concat part {idx} failed")
            _cleanup_paths(parts + [part_path])
            return False
        parts.append(part_path)
        rendered_durs.append(_probe_duration(part_path))

    ok = _concat_files(parts, out_path, logger)
    if ok:
        _write_render_manifest(out_path, clip_name, segments, rendered_durs, logger)
    _cleanup_paths(parts)
    return ok


def _purge_stale_clip_job_dirs(clips_dir: Path, clip_name: str, logger: Any) -> None:
    """Delete cached per-clip job dirs whose scene_cuts / classification cache
    may be stale from a prior extraction of the same clip_name at a DIFFERENT
    time range. Without this, speaker_tracking re-uses old scene_cuts.json and
    the new segment is rendered against the wrong cuts → zero-face trim wipes
    the segment → "concat part 1 failed".
    """
    candidates = [
        clips_dir / f"{clip_name}_raw_job",
    ]
    candidates.extend(clips_dir.glob(f"_temp_{clip_name}_part_*_job"))
    for d in candidates:
        if d.exists() and d.is_dir():
            try:
                shutil.rmtree(d, ignore_errors=True)
                _log(logger, "info", f"  -> purged stale job dir: {d.name}")
            except Exception as exc:
                _log(logger, "warning", f"  -> could not purge {d.name}: {exc}")


def extract_clips(job_dir, clips_plan, settings, logger):
    """Extract raw clips from clips_plan using the Clip Tool v4 render logic."""
    job_path = Path(job_dir)
    settings = settings or {}
    clips_plan = list(clips_plan or [])
    clips_dir = job_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_path = _read_meta_video_path(job_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    for _c in clips_plan:
        _cn = str(_c.get("clip_name", "")).strip()
        if _cn and not _valid_clip(clips_dir / f"{_cn}_raw.mp4"):
            _purge_stale_clip_job_dirs(clips_dir, _cn, logger)

    # Reuse saved framing templates whenever manual_templates.json exists —
    # resumes must never wipe them and relaunch the authoring tool. The tool
    # only runs when no templates file is found (delete manual_templates.json
    # in the job folder to force a re-snap). force_template_resnap (set by the
    # ending adjuster's rerun_templates option) performs that deletion.
    if _bool_setting(settings.get("force_template_resnap"), False):
        _delete_manual_templates(job_path, logger)
    manual_templates = _manual_templates_path(job_path, settings)
    if manual_templates:
        _log(logger, "info", f"Reusing existing manual templates: {manual_templates}")
    else:
        manual_templates = _maybe_launch_template_tool(job_path, video_path, settings, logger)
    if not manual_templates:
        # Tool disabled / closed without snapping / failed → last-resort fall back
        # to an externally-configured template file if one is set.
        manual_templates = _manual_templates_path(job_path, settings)
    diarization = _diarization_path(job_path, settings)
    if manual_templates:
        _log(logger, "info", f"Clip Tool v4 manual templates: {manual_templates}")
    else:
        _log(logger, "info", "Clip Tool v4 manual templates: none supplied; auto discovery will rebuild per clip")
    if diarization:
        _log(logger, "info", f"Clip Tool v4 diarization source: {diarization}")
    else:
        _log(logger, "info", "Clip Tool v4 diarization source: none supplied; v4 fallback rules apply")

    resume_marker = (
        _setting(settings, "extract_resume_from_clip", "resume_from_clip", "resume_start_clip", default=None)
        or getattr(config, "EXTRACTION_RESUME_FROM_CLIP", "")
    )
    resume_start = _resolve_resume_start(clips_plan, resume_marker, logger)

    extracted: List[str] = []
    if resume_start > 0:
        for skipped in clips_plan[:resume_start]:
            skipped_name = str(skipped.get("clip_name", "")).strip()
            if not skipped_name:
                continue
            skipped_path = clips_dir / f"{skipped_name}_raw.mp4"
            if _valid_clip(skipped_path):
                extracted.append(str(skipped_path))
            else:
                _log(logger, "warning", f"Resume skipped {skipped_name}, but {skipped_path.name} is not present/valid")

    workers = int(settings.get("extraction_workers", getattr(config, "EXTRACTION_WORKERS", 2)))

    if workers <= 1:
        for plan_idx, clip in list(enumerate(clips_plan))[resume_start:]:
            clip_name = str(clip.get("clip_name", f"clip_{plan_idx + 1:02d}")).strip()
            out_path = clips_dir / f"{clip_name}_raw.mp4"

            if _valid_clip(out_path):
                size_mb = out_path.stat().st_size / (1024 * 1024)
                _log(logger, "info", f"  -> {out_path.name} already exists ({size_mb:.1f} MB); skipping")
                extracted.append(str(out_path))
                continue

            segments = _clip_segments(clip)
            if not segments:
                _log(logger, "error", f"  -> {clip_name}: no valid segments in clips_plan entry")
                continue

            _log(logger, "info", f"Extracting {clip_name}: {clip.get('title', '')} ({len(segments)} segment(s))")
            ok = _extract_single_clip(
                video_path=video_path,
                job_dir=job_path,
                clip_name=clip_name,
                segments=segments,
                out_path=out_path,
                settings=settings,
                logger=logger,
                manual_templates_path=manual_templates,
                diarization_path=diarization,
            )

            if ok and _valid_clip(out_path):
                extracted.append(str(out_path))
                _log(logger, "info", f"  -> extracted {out_path.name}")
            else:
                _log(logger, "error", f"  -> {clip_name}: Clip Tool v4 extraction failed")

        # CRITICAL: the sequential branch must return the list too — without
        # this the runner does len(None) and the whole job dies *after* every
        # clip was already rendered.
        _log(logger, "info", f"Sequential extraction complete: {len(extracted)}/{len(clips_plan)} raw clips")
        _sweep_clip_workdirs(clips_dir, logger)
        return extracted
    else:
        from pipeline.parallel_extractor import extract_clips_parallel
        result = extract_clips_parallel(job_dir, clips_plan, settings, logger)
        _sweep_clip_workdirs(clips_dir, logger)
        return result


def cleanup_gpu():
    """Release tracking backend GPU memory between jobs."""
    speaker_tracking.cleanup_gpu_resources()

