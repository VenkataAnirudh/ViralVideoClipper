"""
Captioner (v9)
==============
Burns animated word-by-word captions into clips using FFmpeg drawtext.

Changes vs v8.1:
  â€¢ Font size formula simplified: style.font_size IS the target px at 1920h.
    No FONT_SIZE_BOOST, no x2 multiplier. 210 in config = 210px on screen.
  â€¢ BORDER_RATIO 0.09 -> 0.038 (8px at 210px font), MIN_BORDER_PX 9 -> 8.
  â€¢ Line height 1.22x -> 0.80x â€” tight TikTok-style packing.
  â€¢ Line 2 now renders in highlight_color (#FFD100) â€” yellow word accent.
  â€¢ Shadow added: 35% opacity black, 4px vertical drop (resolution-scaled).
  â€¢ position_pct default 62 -> 58 â€” block bottom at ~78% from top = 22% above bottom.

Carried forward from v8.1:
  â€¢ Single drawtext per word (no multi-copy bulk smear).
  â€¢ expansion=none and fix_bounds=1 on every entry.
  â€¢ %% escaping for percent signs in transcript text.
  â€¢ _enc_args() always attempts h264_nvenc first; _ffmpeg() retries libx264.
  â€¢ Uniform pair sizing: min(fs1, fs2) so both words match.
  â€¢ Raw file logging per clip.

Music: Randomly selects from the music/ folder if no user music is provided.
"""


import glob
import json
import os
import random
import shutil
import subprocess
import logging
import re
import threading
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
from pipeline.downloader import download_audio_track
from pipeline.gpu_scheduler import ffmpeg_worker_count

# A dict mapping job_dir -> Thread for background music downloading/analyzing
_bg_music_threads = {}


def pre_process_music_async(job_dir: str, settings: dict, logger) -> None:
    """
    Start downloading and analyzing background music in a separate thread.
    This runs concurrently with video transcription, AI analysis, etc.
    """
    music_url = settings.get("music_url") or ""
    music_path = settings.get("music_path") or ""

    if not music_url and (not music_path or music_path == "none"):
        return

    def worker():
        try:
            m_path = music_path
            if music_url:
                # 1. Download music
                m_path = download_audio_track(job_dir, music_url, logger)
                # Update settings in-place so it's visible to the main thread
                settings["music_path"] = m_path
            if m_path and os.path.exists(m_path):
                # 2. Pre-analyze loudness curve (which populates the cache)
                _analyze_music_volume_curve(m_path, logger)
        except Exception as e:
            logger.warning(f"Background music pre-processing failed: {e}")

    t = threading.Thread(
        target=worker, name=f"bg-music-{os.path.basename(job_dir)}", daemon=True)
    _bg_music_threads[job_dir] = t
    t.start()
    if music_url:
        logger.info(
            f"Started background music pre-processing (download & loudness analysis) for: {music_url}")
    else:
        logger.info(
            f"Started background music pre-processing (loudness analysis) for: {os.path.basename(music_path)}")


def wait_for_music_pre_process(job_dir: str, logger) -> None:
    """
    Wait for background music downloading/analysis thread to finish if running.
    """
    t = _bg_music_threads.get(job_dir)
    if t and t.is_alive():
        logger.info(
            "Waiting for background music download and ebur128 loudness profiling to complete...")
        t.join()
        logger.info("Background music download/profiling completed.")


def rename_music_job_dir(old_job_dir: str, new_job_dir: str) -> None:
    """
    Update the job directory key in the thread tracker if renamed.
    """
    if old_job_dir in _bg_music_threads:
        _bg_music_threads[new_job_dir] = _bg_music_threads.pop(old_job_dir)


def _probe_resolution(video_path: str, default_w: int, default_h: int, logger) -> Tuple[int, int]:
    """Get clip width and height via ffprobe, falling back to default_w and default_h."""
    cmd = [
        config.FFPROBE_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        parts = r.stdout.strip().split('x')
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            if w > 0 and h > 0:
                return w, h
    except Exception as exc:
        logger.warning(f"ffprobe resolution probe failed for {video_path}: {exc}")
    return default_w, default_h


# â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â•
# Public entry point
# â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â•

def burn_captions(job_dir, clips_plan, transcript, settings, logger):
    """Burn captions into all extracted clips (or a subset via reburn_clips)."""
    clips_dir = os.path.join(job_dir, "clips")
    style_name = settings.get("caption_style", config.DEFAULT_CAPTION_STYLE)
    style = config.CAPTION_STYLES.get(style_name)
    if not style:
        style_name = list(config.CAPTION_STYLES.keys())[0]
        style = config.CAPTION_STYLES[style_name]

    overrides = {
        "font_color":         settings.get("caption_font_color"),
        "highlight_color":    settings.get("caption_highlight_color"),
        "word_case":          settings.get("caption_word_case"),
        "line_spacing_ratio": settings.get("caption_line_spacing"),
        "entrance_anim":      settings.get("caption_entrance_anim"),
        "font_size":          settings.get("caption_font_size"),
    }
    if any(value is not None for value in overrides.values()):
        style = dict(style)
        for key, value in overrides.items():
            if value is None or value == "style":
                continue
            if key == "line_spacing_ratio":
                style[key] = _safe_float(
                    value, style.get(key, TWO_LINE_GAP_RATIO))
            elif key == "font_size":
                style[key] = _safe_int(value, style.get(key, 215))
            else:
                style[key] = value

    aspect = settings.get("aspect_ratio", config.DEFAULT_ASPECT_RATIO)
    if aspect == "9:16":
        vid_w, vid_h = config.VERTICAL_RESOLUTION
    else:
        vid_w, vid_h = config.HORIZONTAL_RESOLUTION

    variant_suffix = _caption_variant_suffix(style_name, settings)
    logger.info(
        f"Captioning: style={style['name']}, res={vid_w}x{vid_h}, suffix='{variant_suffix}'")

    # ── Wait for background music download/profiling ──────────────────────────────
    wait_for_music_pre_process(job_dir, logger)

    # â”€â”€ Optional per-clip filter (for re-burn API) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reburn_clips = settings.get("reburn_clips", "all")
    if reburn_clips and reburn_clips != "all":
        selected_names = {c.strip() for c in reburn_clips.split(",")}
        clips_to_process = [
            c for c in clips_plan if c["clip_name"] in selected_names]
        logger.info(
            f"Re-burn mode: processing {len(clips_to_process)} selected clip(s): {selected_names}")
    else:
        clips_to_process = clips_plan

    # ── Pre-flight: confirm at least one _raw.mp4 / intro_*.mp4 exists ────────────
    # When resuming from the caption stage the extract stage is skipped, so the
    # raw clips MUST already be on disk. If none are present, fail loud rather
    # than silently logging "Captioned 0/N clips" and reporting success.
    available_sources = 0
    missing_names: list[str] = []
    for clip in clips_to_process:
        name = clip.get("clip_name", "")
        raw_path = os.path.join(clips_dir, f"{name}_raw.mp4")
        intro_path = os.path.join(clips_dir, "hooks", f"intro_{name}.mp4")
        if os.path.exists(raw_path) or os.path.exists(intro_path):
            available_sources += 1
        else:
            missing_names.append(name)
    if clips_to_process and available_sources == 0:
        sample = ", ".join(missing_names[:5])
        more = "" if len(missing_names) <= 5 else f" (+{len(missing_names) - 5} more)"
        raise RuntimeError(
            f"Captioning aborted: no source clips found in {clips_dir}. "
            f"Expected {len(clips_to_process)} _raw.mp4 file(s); missing: {sample}{more}. "
            "If you reached caption stage via resume, re-run from 'extract' to render the raw clips first."
        )
    if missing_names:
        logger.warning(
            f"Captioning: {len(missing_names)} of {len(clips_to_process)} clips are missing "
            f"_raw.mp4 sources and will be skipped: {', '.join(missing_names[:5])}"
            f"{'' if len(missing_names) <= 5 else f' (+{len(missing_names) - 5} more)'}"
        )

    # ── Verify font exists ────────────────────────────────────────────────────────
    font_path_raw = style.get("font", "C:/Windows/Fonts/impact.ttf")
    if not os.path.isabs(font_path_raw):
        # Resolve relative to project root
        font_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), font_path_raw))
    else:
        font_path = font_path_raw

    if not os.path.exists(font_path):
        filename = os.path.basename(font_path)
        # Try to download font dynamically from Google Fonts GitHub
        from pipeline.font_downloader import download_font_from_google
        download_font_from_google(filename, logger)

    if not os.path.exists(font_path):
        logger.warning(f"Font not found: {font_path}, falling back to Impact")
        font_path = "C:/Windows/Fonts/impact.ttf"
        if not os.path.exists(font_path):
            logger.error(
                "Impact font not found either â€” captions will use FFmpeg default")
            font_path = ""

    # ── Background music setup ────────────────────────────────────────────────────
    music_path = settings.get("music_path") or ""
    music_url = settings.get("music_url") or ""

    if not music_path and music_url:
        try:
            music_path = download_audio_track(job_dir, music_url, logger)
            settings["music_path"] = music_path
        except Exception as exc:
            logger.warning(f"Background music URL download failed: {exc}")

    if music_path == "none":
        music_path = ""
        music_volume = 0.0
    else:
        # No random fallback: music plays ONLY when a link, upload or kept
        # background_music file explicitly supplies it. Empty = silent.
        music_volume = max(0.0, min(1.0, _safe_float(
            settings.get("music_volume"), config.BACKGROUND_MUSIC_DEFAULT_VOLUME
        )))
    music_enabled = bool(music_path and os.path.exists(
        music_path) and music_volume > 0)

    # User-provided FIXED music offset (settings key `music_start_time`, plumbed
    # from the UI "Music Start Offset (sec)"). When set, the SAME offset is used
    # for EVERY clip and the ebur128 per-clip crescendo auto-pick is skipped.
    fixed_music_offset = None
    if music_enabled:
        _mst = settings.get("music_start_time")
        if _mst is not None and str(_mst).strip() != "":
            try:
                fixed_music_offset = max(0.0, float(_mst))
            except (TypeError, ValueError):
                logger.warning(
                    f"Ignoring invalid music_start_time={_mst!r}; "
                    f"using automatic per-clip offset")
                fixed_music_offset = None

    if music_enabled:
        logger.info(
            f"Background music: {os.path.basename(music_path)} @ {music_volume * 100:.0f}% volume")
        global _used_music_offsets
        _used_music_offsets = set()
        if fixed_music_offset is not None:
            logger.info(
                f"Using FIXED user music offset for ALL clips: "
                f"{fixed_music_offset:.1f}s (skipping ebur128 auto-pick)")
        else:
            # Pre-analyze music loudness curve once
            _analyze_music_volume_curve(music_path, logger)
    elif music_path:
        logger.warning(
            f"Background music file not found or muted: {music_path}")

    def _music_offset_for(clip_dur):
        """Same fixed offset for every clip when the user supplied one;
        otherwise the ebur128 per-clip crescendo auto-pick."""
        if fixed_music_offset is not None:
            return fixed_music_offset
        return _pick_music_offset(music_path, clip_dur, logger)

    # ── Per-clip processing ───────────────────────────────────────────────
    captioned = []
    workers = ffmpeg_worker_count(
        "Captioning",
        getattr(config, "CAPTION_MAX_PARALLEL_JOBS", 1),
        logger,
    )

    def _render_clip(clip):
        try:
            raw_clip_path = os.path.join(clips_dir, f"{clip['clip_name']}_raw.mp4")
            intro_video_path = os.path.join(clips_dir, "hooks", f"intro_{clip['clip_name']}.mp4")

            clip_suffix = _unique_caption_suffix(
                clips_dir,
                clip["clip_name"],
                variant_suffix,
                settings.get("preserve_caption_variants"),
            )
            cap_path = os.path.join(
                clips_dir, f"{clip['clip_name']}_captioned{clip_suffix}.mp4")
            hooked_cap_path = os.path.join(
                clips_dir, f"{clip['clip_name']}_captioned{clip_suffix}_hooked.mp4")
            srt_path = os.path.join(clips_dir, f"{clip['clip_name']}.srt")

            hook_mode = str(settings.get("tts_hook_mode", "")).strip().lower()
            hook_enabled = bool(settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED))
            allow_hook_merge = hook_enabled and hook_mode not in {"save_only", "remove"}

            if not allow_hook_merge:
                import glob
                stale_patterns = [
                    os.path.join(clips_dir, f"{clip['clip_name']}_captioned*_hooked.mp4"),
                    os.path.join(clips_dir, f"{clip['clip_name']}_hooked.mp4")
                ]
                for pattern in stale_patterns:
                    for p in glob.glob(pattern):
                        try:
                            os.remove(p)
                            logger.info(f"Cleaned up stale hooked output: {p}")
                        except Exception as e:
                            logger.warning(f"Could not remove stale hooked output {p}: {e}")

            # Determine actual resolution
            probe_src = raw_clip_path if os.path.exists(raw_clip_path) else (intro_video_path if os.path.exists(intro_video_path) else "")
            if not probe_src:
                logger.warning(f"No source clips found for {clip['clip_name']}!")
                return
            clip_w, clip_h = _probe_resolution(probe_src, vid_w, vid_h, logger)

            # Build/remap local words
            render_offsets = _load_render_offsets(clips_dir, clip["clip_name"])
            if "edited_words" in clip:
                local_words = _localize_words(clip["edited_words"], clip, render_offsets)
            else:
                local_words = _get_clip_local_words(clip, transcript, render_offsets)

            if settings.get("emoji_captions") and local_words:
                local_words = _add_emojis_to_words(local_words)

            hook_title = clip.get("hook_title", "")

            # --- Subtitle Change Detection ---
            subtitles_changed = False
            new_srt_content = _get_plan_srt_content(local_words, style)

            if os.path.exists(srt_path):
                try:
                    with open(srt_path, "r", encoding="utf-8") as f:
                        existing_srt_content = f.read()
                    if _normalize_srt(existing_srt_content) != _normalize_srt(new_srt_content):
                        subtitles_changed = True
                except Exception as e:
                    logger.warning(f"Failed to read existing SRT for {clip['clip_name']}: {e}")
                    subtitles_changed = True
            else:
                subtitles_changed = True

            if subtitles_changed:
                logger.info(f"Subtitles changed for {clip['clip_name']}. Invalidating old captioned files.")
                for f_path in [cap_path, hooked_cap_path, srt_path]:
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                            logger.info(f"Removed outdated file: {f_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove outdated file {f_path}: {e}")

            # Always write the clean style-independent source-of-truth SRT file
            if subtitles_changed or not os.path.exists(srt_path):
                _generate_srt(local_words, srt_path, style, logger)

            # Helper to burn captions on a single video target with custom tts_shift
            def _burn_single_variant(src_video, dest_video, tts_shift):
                # Shift timestamps
                shifted_words = []
                if local_words:
                    for w in local_words:
                        shifted_words.append({
                            "word": w["word"],
                            "start": w["start"] + tts_shift,
                            "end": w["end"] + tts_shift
                        })

                # Write unique filter and srt files
                var_base = os.path.basename(dest_video).replace(".mp4", "")
                var_filter_path = os.path.join(clips_dir, f"{var_base}_filter.txt")
                var_srt_path = os.path.join(clips_dir, f"{var_base}.srt")

                if shifted_words:
                    if getattr(config, "CAPTION_SAVE_SRT", False):
                        _generate_srt(shifted_words, var_srt_path, style, logger)
                    _generate_drawtext_filter(
                        shifted_words, var_filter_path, style, clip_w, clip_h, font_path, logger)
                else:
                    with open(var_filter_path, "w", encoding="utf-8") as f:
                        f.write("")

                if hook_title:
                    hook_fs = int(style.get("font_size", 210) * 0.45 * (clip_h / 1920.0))
                    if hook_fs < 24:
                        hook_fs = 24
                    hook_color = style.get("highlight_color", "#FFD100")
                    border_color = style.get("outline_color", "black")
                    border_w = max(MIN_BORDER_PX, int(hook_fs * BORDER_RATIO))
                    y_expr = "h*0.15"

                    clean_title = _strip_caption_punctuation(hook_title)
                    # Hook title is ALWAYS all-caps, regardless of caption word_case —
                    # a mixed-case edited hook phrase must never render in smalls.
                    h_title = clean_title.upper()

                    hook_filter = _build_hook_title_drawtext_entry(
                        h_title, font_path, hook_fs, hook_color, border_color, border_w, y_expr, start_time=tts_shift
                    )

                    # Append to existing drawtext filter script
                    existing = ""
                    if os.path.exists(var_filter_path):
                        with open(var_filter_path, "r", encoding="utf-8") as f:
                            existing = f.read().strip()
                    if existing and existing != "null":
                        new_filter = existing + "," + hook_filter
                    else:
                        new_filter = hook_filter

                    with open(var_filter_path, "w", encoding="utf-8") as f:
                        f.write(new_filter)

                # Check for music
                if music_enabled:
                    clip_dur = max(0.1, _probe_dur(src_video, logger))
                    music_offset = _music_offset_for(clip_dur)
                    try:
                        _burn_with_drawtext_and_music(
                            src_video, var_filter_path, music_path, dest_video,
                            music_volume, music_offset, logger, tts_shift
                        )
                    except Exception as mix_err:
                        logger.warning(
                            f"Caption+music render failed for variant {dest_video}: "
                            f"{mix_err} — retrying without music"
                        )
                        _burn_with_drawtext(src_video, var_filter_path, dest_video, logger)
                else:
                    _burn_with_drawtext(src_video, var_filter_path, dest_video, logger)

                if os.path.exists(dest_video):
                    sz = os.path.getsize(dest_video) / (1024 * 1024)
                    if not getattr(config, "CAPTION_KEEP_FILTER_FILES", False):
                        _cleanup_caption_artifacts(var_filter_path, var_srt_path, logger)
                    logger.info(f"  ✓ {os.path.basename(dest_video)}  ({sz:.1f} MB)")
                    return True
                else:
                    logger.error(f"  ✗ Caption burn produced no output for variant {dest_video}")
                    return False

            rendered_any = False

            # Case A: If there are no words and no hook title
            if not local_words and not hook_title:
                logger.warning(f"No words or hook title for {clip['clip_name']}, copying raw/hooked")
                # 1. Mix raw
                if os.path.exists(raw_clip_path):
                    if os.path.exists(cap_path):
                        logger.info(f"Skipping mix raw for {clip['clip_name']} (already exists)")
                        captioned.append(cap_path)
                        rendered_any = True
                    else:
                        clip_dur_for_mix = max(0.1, _probe_dur(raw_clip_path, logger)) if music_enabled else 0
                        m_offset = _music_offset_for(clip_dur_for_mix) if music_enabled else 0.0
                        _copy_or_mix(raw_clip_path, None, cap_path, music_path, music_volume, music_enabled, m_offset, logger, 0.0)
                        captioned.append(cap_path)
                        rendered_any = True
                
                # 2. Mix hooked (by prepending hook intro)
                if allow_hook_merge and os.path.exists(intro_video_path):
                    if os.path.exists(hooked_cap_path):
                        logger.info(f"Skipping mix hooked for {clip['clip_name']} (already exists)")
                        captioned.append(hooked_cap_path)
                        rendered_any = True
                    else:
                        # Ensure cap_path exists
                        if not os.path.exists(cap_path):
                            clip_dur_for_mix = max(0.1, _probe_dur(raw_clip_path, logger)) if music_enabled else 0
                            m_offset = _music_offset_for(clip_dur_for_mix) if music_enabled else 0.0
                            _copy_or_mix(raw_clip_path, None, cap_path, music_path, music_volume, music_enabled, m_offset, logger, 0.0)
                        
                        if os.path.exists(cap_path):
                            logger.info(f"Generating captioned hooked clip (no text) for {clip['clip_name']} by prepending hook intro...")
                            if _concat_videos(intro_video_path, cap_path, hooked_cap_path, logger):
                                captioned.append(hooked_cap_path)
                                rendered_any = True
                return

            # Case B: Render standard captioned (no hook)
            if os.path.exists(raw_clip_path):
                if os.path.exists(cap_path):
                    logger.info(f"Skipping captioned raw clip for {clip['clip_name']} (already exists)")
                    captioned.append(cap_path)
                    rendered_any = True
                else:
                    logger.info(f"Rendering captioned raw clip for {clip['clip_name']}...")
                    if _burn_single_variant(raw_clip_path, cap_path, 0.0):
                        captioned.append(cap_path)
                        rendered_any = True

            # Case C: Render hooked captioned (by prepending hook intro to captioned standard clip)
            if allow_hook_merge and os.path.exists(intro_video_path):
                if os.path.exists(hooked_cap_path):
                    logger.info(f"Skipping captioned hooked clip for {clip['clip_name']} (already exists)")
                    captioned.append(hooked_cap_path)
                    rendered_any = True
                else:
                    logger.info(f"Generating captioned hooked clip for {clip['clip_name']} by prepending hook intro...")
                    # Ensure standard captioned video is rendered first
                    if not os.path.exists(cap_path):
                        logger.info(f"Rendering standard captioned clip first for {clip['clip_name']}...")
                        _burn_single_variant(raw_clip_path, cap_path, 0.0)
                    
                    if os.path.exists(cap_path):
                        if _concat_videos(intro_video_path, cap_path, hooked_cap_path, logger):
                            captioned.append(hooked_cap_path)
                            rendered_any = True
                        else:
                            logger.error(f"Failed to prepend hook intro for {clip['clip_name']}")
                    else:
                        logger.error(f"Cannot prepend hook intro because standard captioned video failed to render.")

            if not rendered_any:
                logger.error(f"  ✗ Caption burn produced no output for {clip['clip_name']}")

        except Exception as exc:
            logger.error(
                f"Captioning failed for {clip['clip_name']}: {exc}", exc_info=True)

    if workers <= 1 or len(clips_to_process) <= 1:
        for clip in clips_to_process:
            _render_clip(clip)
    else:
        logger.info(f"Captioning with {workers} parallel FFmpeg worker(s)")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_clip, clip)
                       for clip in clips_to_process]
            for future in as_completed(futures):
                future.result()

    logger.info(f"Captioned {len(captioned)}/{len(clips_to_process)} clips")
    return captioned


def _caption_variant_suffix(style_name, settings):
    """
    Always embed the font/style name in the output filename.
    e.g. clip_01_captioned_impact.mp4, clip_01_captioned_bebas_neue.mp4

    An explicit caption_output_suffix from settings still wins (used by the
    reburn API), otherwise we always fall back to the style name â€” never "".
    This guarantees different fonts never overwrite each other, and re-running
    the same font cleanly replaces the previous render of that font.
    """
    explicit = str(settings.get("caption_output_suffix", "") or "").strip()
    if explicit:
        return "_" + _slug(explicit).lstrip("_")
    return "_" + _slug(style_name)


def _unique_caption_suffix(clips_dir, clip_name, suffix, preserve_variants):
    """
    Font name is now always embedded in the suffix.
    Same font  → overwrite (simple replace, no counter needed).
    Different font → different suffix → different file (no conflict).
    """
    return suffix


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return value.strip("_") or "caption"


# ══════════════════════════════════════════════════════════════════════════════
# Smart music offset — pick a random *loud/crescendo* portion per clip
# ══════════════════════════════════════════════════════════════════════════════

# Cache: {(abs_path, mtime) -> {"duration": float, "candidates": [dict, ...]}}
_music_analysis_cache: dict = {}
_used_music_offsets: set = set()

# Tuning constants
_INTRO_SKIP_S = 10.0  # skip first N seconds of the track (often a quiet intro)
_OUTRO_SKIP_S = 10.0  # skip last  N seconds of the track (often a fade-out)


def _analyze_music_volume_curve(music_path: str, logger) -> dict:
    """Run FFmpeg ebur128 once per music file and cache the result.
    Scans for volume envelope rises (crescendos) and sustained plateaus.
    """
    abs_path = os.path.abspath(music_path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        return {"duration": 0.0, "candidates": []}

    cache_key = (abs_path, mtime)
    if cache_key in _music_analysis_cache:
        return _music_analysis_cache[cache_key]

    # Persistent disk caching setup
    music_dir = os.path.dirname(abs_path)
    cache_dir = os.path.join(music_dir, ".loudness")
    cache_file = os.path.join(cache_dir, os.path.basename(abs_path) + ".loudness.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data.get("mtime") == mtime:
                result = {
                    "duration": cached_data["duration"],
                    "candidates": cached_data["candidates"]
                }
                _music_analysis_cache[cache_key] = result
                logger.info(f"Loaded cached loudness analysis from disk for {os.path.basename(music_path)}")
                return result
        except Exception as exc:
            logger.warning(f"Failed to read disk loudness cache for {os.path.basename(music_path)}: {exc}")

    music_dur = _probe_dur(music_path, logger)
    if music_dur <= 0:
        result = {"duration": 0.0, "candidates": []}
        _music_analysis_cache[cache_key] = result
        return result

    # Run FFmpeg ebur128
    cmd = [
        config.FFMPEG_PATH,
        "-i", music_path,
        "-af", "ebur128",
        "-f", "null", "-",
    ]
    logger.info(
        f"Analyzing music loudness (ebur128): {os.path.basename(music_path)} ({music_dur:.0f}s)")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=90,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        output = (proc.stderr or "") + "\n" + (proc.stdout or "")
    except Exception as exc:
        logger.warning(f"ebur128 loudness analysis failed: {exc}")
        result = {
            "duration": music_dur,
            "candidates": [{"start": 0.0, "end": music_dur, "type": "sustain", "score": 10.0}]
        }
        _music_analysis_cache[cache_key] = result
        return result

    # Parse output to build volume curve
    volume_curve = []
    for line in output.splitlines():
        if "t:" in line and "M:" in line:
            m = re.search(r"t:\s*([\d\.]+).*?\bM:\s*([-\d\.]+|-\s*inf)", line)
            if m:
                try:
                    t_sec = float(m.group(1))
                    val_str = m.group(2).replace(" ", "")
                    m_db = -120.0 if "inf" in val_str else float(val_str)
                    volume_curve.append((t_sec, m_db))
                except (ValueError, IndexError):
                    pass

    if not volume_curve:
        logger.warning("No volume samples parsed from ebur128 output")
        result = {
            "duration": music_dur,
            "candidates": [{"start": 0.0, "end": music_dur, "type": "sustain", "score": 10.0}]
        }
        _music_analysis_cache[cache_key] = result
        return result

    # Scan for candidates
    # We want rise/sustain windows of length 8s to 25s
    safe_start = _INTRO_SKIP_S
    safe_end = max(0.0, music_dur - _OUTRO_SKIP_S)
    if safe_end <= safe_start + 8.0:
        safe_start = 0.0
        safe_end = music_dur

    candidates = []
    durations = [8.0, 12.0, 16.0, 20.0, 24.0]

    def get_avg_vol(t_start, t_end):
        vals = [val for t, val in volume_curve if t_start <= t <= t_end]
        if not vals:
            return -120.0
        return sum(vals) / len(vals)

    def get_max_vol(t_start, t_end):
        vals = [val for t, val in volume_curve if t_start <= t <= t_end]
        if not vals:
            return -120.0
        return max(vals)

    t_step = 1.0
    t_curr = safe_start
    while t_curr < safe_end - 8.0:
        for dur in durations:
            t_end = t_curr + dur
            if t_end > safe_end:
                continue

            split_start_end = t_curr + dur * 0.2
            split_end_start = t_curr + dur * 0.8

            avg_start = get_avg_vol(t_curr, split_start_end)
            avg_end = get_avg_vol(split_end_start, t_end)
            peak_vol = get_max_vol(t_curr, t_end)

            if peak_vol < -40.0:
                continue

            # Is it a rise? ending volume is louder by at least 3dB/LUFS
            if avg_end > avg_start + 3.0 and peak_vol > -25.0:
                rise_amt = avg_end - avg_start
                score = rise_amt * dur * (1.0 + (peak_vol + 20.0) / 100.0)
                candidates.append({
                    "start": t_curr,
                    "end": t_end,
                    "type": "rise",
                    "score": round(score, 2),
                    "peak_vol": peak_vol,
                    "avg_vol": get_avg_vol(t_curr, t_end)
                })
            else:
                # Is it a sustain?
                avg_all = get_avg_vol(t_curr, t_end)
                if avg_all > -22.0:
                    score = (100.0 + avg_all) * (dur / 10.0)
                    candidates.append({
                        "start": t_curr,
                        "end": t_end,
                        "type": "sustain",
                        "score": round(score, 2),
                        "peak_vol": peak_vol,
                        "avg_vol": avg_all
                    })

        t_curr += t_step

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Filter overlapping candidates
    unique_candidates = []
    for cand in candidates:
        overlap = False
        for kept in unique_candidates:
            o_start = max(cand["start"], kept["start"])
            o_end = min(cand["end"], kept["end"])
            if o_end > o_start:
                o_dur = o_end - o_start
                min_dur = min(cand["end"] - cand["start"],
                              kept["end"] - kept["start"])
                if o_dur / min_dur > 0.5:
                    overlap = True
                    break
        if not overlap:
            unique_candidates.append(cand)

    logger.info(
        f"ebur128 Analysis: found {len(unique_candidates)} unique candidates "
        f"(rises: {len([c for c in unique_candidates if c['type'] == 'rise'])}), "
        f"duration={music_dur:.1f}s"
    )

    result = {"duration": music_dur, "candidates": unique_candidates}
    _music_analysis_cache[cache_key] = result

    # Save to disk cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "mtime": mtime,
                "duration": music_dur,
                "candidates": unique_candidates
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved loudness analysis cache to disk at {cache_file}")
    except Exception as exc:
        logger.warning(f"Failed to save disk loudness cache for {os.path.basename(music_path)}: {exc}")

    return result


def _pick_music_offset(music_path: str, clip_duration: float, logger) -> float:
    """Pick a unique music portion per clip, preferring energy-rise (crescendos).
    Aligns the crescendo peak to min(12.0, clip_duration * 0.4) seconds into the clip.
    """
    global _used_music_offsets
    if not music_path or clip_duration <= 0:
        return 0.0

    analysis = _analyze_music_volume_curve(music_path, logger)
    music_dur = analysis.get("duration", 0.0)
    candidates = analysis.get("candidates", [])

    if music_dur < clip_duration * 2 or not candidates:
        max_offset = max(0.0, music_dur - clip_duration)
        return random.uniform(0.0, max_offset) if max_offset > 0.0 else 0.0

    # Filter out candidates that have already been used in this job
    unused_candidates = [
        c for c in candidates
        if (round(c["start"], 1), round(c["end"], 1)) not in _used_music_offsets
    ]

    # If all candidates have been used, reset and use all candidates
    if not unused_candidates:
        unused_candidates = candidates

    # Prefer 'rise' type over 'sustain'
    rise_candidates = [c for c in unused_candidates if c["type"] == "rise"]
    sustain_candidates = [
        c for c in unused_candidates if c["type"] == "sustain"]

    if rise_candidates:
        chosen = random.choice(rise_candidates)
    elif sustain_candidates:
        chosen = random.choice(sustain_candidates)
    else:
        chosen = random.choice(candidates)

    # Mark as used
    _used_music_offsets.add(
        (round(chosen["start"], 1), round(chosen["end"], 1)))

    if chosen["type"] == "rise":
        target_clip_offset = min(12.0, clip_duration * 0.4)
        offset = chosen["end"] - target_clip_offset
    else:
        max_start = max(chosen["start"], chosen["end"] - clip_duration)
        offset = random.uniform(
            chosen["start"], max(chosen["start"], max_start))

    offset = max(0.0, min(offset, music_dur - clip_duration))
    logger.info(
        f"Smart music offset: {offset:.1f}s (Type: {chosen['type']}, "
        f"Window: {chosen['start']:.1f}s - {chosen['end']:.1f}s, "
        f"Score: {chosen['score']:.1f})"
    )
    return offset

# ══════════════════════════════════════════════════════════════════════════════
# Drawtext subtitle generation — CHUNKER
# ══════════════════════════════════════════════════════════════════════════════


# if a single word exceeds this, it gets its own line
MAX_CHARS_PER_LINE = getattr(config, "MAX_CHARS_PER_LINE", 12)
MIN_REVEAL_GAP = getattr(config, "MIN_REVEAL_GAP", 0.16)
SAFE_MARGIN_PCT = getattr(config, "SAFE_MARGIN_PCT",
                          0.10)   # 10% safe zone each side
LONG_WORD_MARGIN_PCT = getattr(config, "LONG_WORD_MARGIN_PCT", 0.13)
LONG_WORD_WIDTH_SAFETY = getattr(config, "LONG_WORD_WIDTH_SAFETY", 1.16)
TWO_LINE_TOP_OFFSET_RATIO = getattr(config, "TWO_LINE_TOP_OFFSET_RATIO", 0.18)
TWO_LINE_GAP_RATIO = getattr(config, "TWO_LINE_GAP_RATIO", 1.02)
# 1. Match period, comma, colon NOT sandwiched between digits
# 2. Match anything that isn't a word char, space, or one of our protected symbols ($, %, +, &, #, @, /, ', ’)
# NB: hyphens/dashes are intentionally NOT protected here — _DASH_RE below turns
# every dash variant into a space first, so "anti-aging" reads as "ANTI AGING"
# and a stray dash is never burnt into the caption.
CAPTION_PUNCT_RE = re.compile(
    r"(?:(?<!\d)[.,:]|[.,:](?!\d))|[^\w\s$%+&#@/'’.,:*]",
    re.UNICODE
)

# All dash/hyphen variants → space: ASCII hyphen-minus, Unicode hyphen (U+2010),
# non-breaking hyphen (U+2011), figure dash (U+2012), en dash (U+2013), em dash
# (U+2014), horizontal bar (U+2015), and minus sign (U+2212).
_DASH_RE = re.compile(r"[-‐‑‒–—―−]")

# ── Border: spec calls for 8px on a 210px font (≈3.8%) ────────────────────────
# 8px ÷ 210px — scales cleanly with font size
BORDER_RATIO = getattr(config, "BORDER_RATIO", 0.038)
# floor at 8px regardless of font size
MIN_BORDER_PX = getattr(config, "MIN_BORDER_PX", 8)


def _strip_caption_punctuation(text: str) -> str:
    """Remove sentence punctuation AND all dashes/hyphens (dashes become a space
    so compound words split cleanly); preserve numbers, currency, %, etc."""
    text = _DASH_RE.sub(" ", str(text or ""))
    cleaned = CAPTION_PUNCT_RE.sub("", text).replace("_", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _apply_word_case(text: str, word_case: str) -> str:
    text = _strip_caption_punctuation(text)
    if word_case == "lower":
        return text.lower()
    if word_case == "title":
        return text.title()
    if word_case == "natural":
        return text
    return text.upper()


def chunk_words(word_list, word_case="upper"):
    """
    Returns list of (line1, line1_start, line2, line2_start, end_time) tuples.

    line1_start / line2_start are the timestamps at which each word should
    appear.  line2 / line2_start may be empty/None for solo-word chunks.

    Word-by-word reveal:
      • line1 becomes visible at line1_start
      • line2 becomes visible at line2_start (which is >= line1_start)
      • Both remain on screen until end_time (the end of the last word in chunk)
    """
    # Sort by start time first — Whisper occasionally emits out-of-order word
    # timestamps (especially around silence, music, or overlapping speech).
    # Chunking unsorted words produces invalid start/end ranges that crash FFmpeg.
    word_list = sorted(word_list, key=lambda w: float(w.get("start", 0.0)))

    chunks = []
    i = 0
    while i < len(word_list):
        w1_dict = word_list[i]
        w1 = _apply_word_case(w1_dict.get("word", "").strip(), word_case)
        if not w1:
            i += 1
            continue

        w2_dict = word_list[i + 1] if i + 1 < len(word_list) else None
        w2 = _apply_word_case(w2_dict.get(
            "word", "").strip(), word_case) if w2_dict else ""

        line1_start = float(w1_dict.get("start", 0.0))

        # w1 is too long on its own — show solo
        if len(w1) > MAX_CHARS_PER_LINE:
            end_time = max(
                float(w1_dict.get("end", line1_start + 0.3)), line1_start + 0.2)
            chunks.append((w1, line1_start, "", None, end_time))
            i += 1

        # Pair together would be too long — show w1 solo
        elif w2 and len(w1 + " " + w2) > MAX_CHARS_PER_LINE * 1.5:
            end_time = max(
                float(w1_dict.get("end", line1_start + 0.3)), line1_start + 0.2)
            chunks.append((w1, line1_start, "", None, end_time))
            i += 1

        else:
            # Pair both words: line2 appears at its own word timestamp
            if w2_dict:
                line2_start = float(w2_dict.get("start", float(
                    w1_dict.get("end", line1_start + 0.2))))
                line2_start = max(line2_start, line1_start + MIN_REVEAL_GAP)
                end_time = float(w2_dict.get("end", line1_start + 0.5))
                end_time = max(end_time, line2_start + 0.12)
            else:
                line2_start = None
                end_time = float(w1_dict.get("end", line1_start + 0.3))

            end_time = max(end_time, line1_start + 0.2)
            chunks.append(
                (w1, line1_start, w2 if w2 else "", line2_start, end_time))
            i += 2 if w2 else 1

    return chunks


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Per-word dynamic font size
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _estimated_text_width(text, font_size, letter_spacing=0, width_factor=1.0):
    """
    Estimate rendered text width.

    Ratios are calibrated for condensed fonts (Impact, Bebas Neue, Oswald).
    These fonts are ~55-65% as wide as regular-width fonts at the same point
    size, so using regular-font ratios (0.78 / 0.58) was over-estimating widths
    and forcing unnecessary font-size reductions on words that actually fit fine.
    """
    width = 0.0
    for ch in text:
        if ch in "MW@#%&":
            ratio = 0.55   # was 0.78 ── condensed M/W are much narrower
        elif ch in "ilI1!|":
            ratio = 0.27   # was 0.34 ── narrow chars, condensed font narrows further
        elif ch.isspace():
            ratio = 0.28   # was 0.32
        else:
            ratio = 0.42   # was 0.58 ── core fix: condensed fonts are ~28% narrower
        width += ratio * font_size
    if len(text) > 1:
        width += max(0, len(text) - 1) * letter_spacing
    if width_factor is not None:
        width *= width_factor
    return width


def _calc_font_size_for_word(word, base_font_size, vid_w, margin_pct=SAFE_MARGIN_PCT,
                             letter_spacing=0, width_factor=None, border_w=0):
    """
    Return an appropriate font size for this word so it stays within
    safe horizontal bounds (leaving margin_pct on each side).

    border_w: the drawtext border thickness in pixels. This is subtracted from
    the safe width on each side so thick outlines never clip at the frame edge.

    If the word fits at base_font_size, returns base_font_size unchanged.
    Otherwise scales down proportionally.  Minimum 36px so text is never invisible.
    """
    if not word:
        return base_font_size

    if len(word) > MAX_CHARS_PER_LINE:
        margin_pct = max(margin_pct, LONG_WORD_MARGIN_PCT)

    # Subtract border_w from each side (border extends outside the text_w box)
    safe_w = vid_w * (1.0 - margin_pct * 2) - border_w * 2
    # floor: never go below 50% of frame width
    safe_w = max(safe_w, vid_w * 0.5)
    needed_w = _estimated_text_width(
        word, base_font_size, letter_spacing, width_factor)
    if len(word) > MAX_CHARS_PER_LINE:
        needed_w *= LONG_WORD_WIDTH_SAFETY

    if needed_w <= safe_w:
        return base_font_size

    reduced = int(base_font_size * (safe_w / max(1.0, needed_w)))
    return max(reduced, 36)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Drawtext filter builder
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _build_drawtext_entry(text, font_path, font_size, font_color, border_color,
                          border_width, y_expr, start, end,
                          shadow_color="", shadow_x=0, shadow_y=0, fade_in=False):
    """Build a single drawtext filter entry.

    Escaping rules for FFmpeg drawtext inside a filter_script file:
    - Text: escape backslash, then single-quote, then colon, then semicolons,
      then percent-sign (must be %% or FFmpeg expands it as a strftime token)
    - Font path: use forward slashes, escape colons with backslash
    - expansion=none: disables all FFmpeg text-expansion so % in words is safe
    - fix_bounds=1: prevents text from being clipped at the frame boundary
    - shadowcolor / shadowx / shadowy: hard directional drop shadow (no blur in drawtext)
    """
    # Subtract 5ms from end to prevent inclusive-boundary overlap between consecutive chunks.
    # Keep it at least 5ms after start so it renders on at least one frame.
    adjusted_end = max(start + 0.005, end - 0.005)

    # Enforce start < adjusted_end to prevent FFmpeg between(t, start, end) crashes
    if adjusted_end <= start:
        adjusted_end = start + 0.033

    safe_text = (
        _strip_caption_punctuation(text)
        .replace("\\", "\\\\")
        .replace("'", "'\\\\\\''")
        .replace(":", "\\:")
        .replace(";", "\\;")
    )

    safe_font = font_path.replace("\\", "/").replace(":", "\\:")
    safe_font_escaped = safe_font.replace("'", "'\\\\\\''")

    shadow_str = (
        f":shadowcolor={shadow_color}:shadowx={shadow_x}:shadowy={shadow_y}"
        if shadow_color else ""
    )

    alpha_str = ""
    if fade_in:
        fade_dur = 0.08
        alpha_str = f":alpha='min(1,(t-{start:.3f})/{fade_dur:.2f})'"

    if font_path:
        return (
            f"drawtext="
            f"fontfile='{safe_font_escaped}'"
            f":text='{safe_text}'"
            f":fontsize={font_size}"
            f":fontcolor={font_color}"
            f":bordercolor={border_color}"
            f":borderw={border_width}"
            f":x=(w-text_w)/2"
            f":y={y_expr}"
            f":expansion=none"
            f":fix_bounds=1"
            f"{shadow_str}"
            f"{alpha_str}"
            f":enable='between(t,{start:.3f},{adjusted_end:.3f})'"
        )
    else:
        return (
            f"drawtext="
            f"text='{safe_text}'"
            f":fontsize={font_size}"
            f":fontcolor={font_color}"
            f":bordercolor={border_color}"
            f":borderw={border_width}"
            f":x=(w-text_w)/2"
            f":y={y_expr}"
            f":expansion=none"
            f":fix_bounds=1"
            f"{shadow_str}"
            f"{alpha_str}"
            f":enable='between(t,{start:.3f},{adjusted_end:.3f})'"
        )


def _generate_drawtext_filter(words, filter_path, style, vid_w, vid_h,
                              font_path_override, logger):
    """
    Generate an FFmpeg filter script for word-by-word drawtext captions.

    Key design decisions:
    â”€ Font size: style.font_size IS the target px at 1920h â€” no multiplier.
      Scales linearly to actual vid_h. 210 in config = 210px on 1920-tall frame.
    â”€ Per-word scaling: long words are scaled DOWN only â€” base is never exceeded.
    â”€ UNIFORM PAIR SIZE: both words in a 2-word chunk use min(fs1, fs2).
    â”€ Line height 80%: tight TikTok-style packing (line_gap = 0.80 Ã— font_size).
    â”€ Line 1 â†’ font_color (#F2F0EA).  Line 2 â†’ highlight_color (#FFD100).
    â”€ Shadow: 35% opacity black, 4px vertical drop (FFmpeg drawtext approx).
    â”€ Single drawtext per word â€” no multi-copy bulk technique.
    """
    font_path = font_path_override

    # â”€â”€ Font size: style value = actual px at 1920h, scaled to vid_h â”€â”€â”€â”€â”€â”€â”€â”€
    base_font_size = int(style.get("font_size", 210) * (vid_h / 1920.0))

    font_color = style.get("font_color",      "#F2F0EA")
    highlight_color = style.get("highlight_color", "#FFD100")
    border_color = style.get("outline_color",   "black")
    safe_margin_pct = style.get("safe_margin_pct", SAFE_MARGIN_PCT)
    word_case = style.get("word_case", "upper")
    line_spc_ratio = _safe_float(
        style.get("line_spacing_ratio"), TWO_LINE_GAP_RATIO)
    active_word_clr = style.get("active_word_color", "")
    entrance_anim = style.get("entrance_anim", "none")
    slide_px = int(28 * (vid_h / 1920.0))
    letter_spc = style.get("letter_spacing",  -2)   # width estimation only
    # 58% ── block bottom ──78% = 22% above bottom
    position_pct = style.get("position_pct",    58)
    width_fact = style.get("width_factor",    1.0)

    # â”€â”€ Shadow: per-style override or default black soft shadow â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    style_shadow_color = style.get("shadow_color", "")
    if style_shadow_color:
        shadow_color = style_shadow_color
        shadow_x = max(0, int(style.get("shadow_x", 0) * (vid_h / 1920.0)))
        shadow_y = max(0, int(style.get("shadow_y", 0) * (vid_h / 1920.0)))
    else:
        # Default: 35% opacity black, no X offset, 4px Y drop
        shadow_color = "0x000000@0.35"
        shadow_x = 0
        shadow_y = max(1, int(4 * (vid_h / 1920.0)))

    chunks = chunk_words(words, word_case)

    # Pre-compute an estimated border width at base_font_size for use in the
    # font-size calculation. The actual per-chunk border_w is recomputed below.
    border_w_est = max(MIN_BORDER_PX, int(base_font_size * BORDER_RATIO))

    # ── Fix chunk overlap: clamp each chunk's end to the next chunk's start ──
    # Without this, the previous top-line lingers for 1-2 frames while the next
    # chunk's top-line is already appearing, creating a brief double-flash.
    for ci in range(len(chunks) - 1):
        line1, line1_start, line2, line2_start, end_time = chunks[ci]
        next_start = chunks[ci + 1][1]  # next chunk's line1_start
        if end_time > next_start:
            # Clamp to ~1 frame before the next chunk starts (33ms at 30fps).
            # Always guarantee new_end > line1_start by at least 50ms.
            new_end = max(next_start - 0.033, line1_start + 0.05)

            # CRITICAL: if line2_start >= new_end, the FFmpeg expression
            # between(t, line2_start, new_end) would have start > end — an
            # invalid argument that aborts rendering with "Error: Invalid argument".
            # Demote to single-line so line2 is simply not drawn.
            if line2 and line2_start is not None and line2_start >= new_end - 0.05:
                chunks[ci] = (line1, line1_start, "", None, new_end)
            else:
                chunks[ci] = (line1, line1_start, line2, line2_start, new_end)
    filters = []

    for (line1, line1_start, line2, line2_start, end) in chunks:

        # â”€â”€ Per-word font sizes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        fs1_raw = _calc_font_size_for_word(
            line1, base_font_size, vid_w,
            margin_pct=safe_margin_pct, letter_spacing=letter_spc,
            width_factor=width_fact, border_w=border_w_est
        )

        if line2 and line2_start is not None:
            fs2_raw = _calc_font_size_for_word(
                line2, base_font_size, vid_w,
                margin_pct=safe_margin_pct, letter_spacing=letter_spc,
                width_factor=width_fact, border_w=border_w_est
            )
            # UNIFORM PAIR SIZE: both lines use same font size so neither
            # looks disproportionately large vs its partner.
            pair_fs = min(fs1_raw, fs2_raw)
        else:
            pair_fs = fs1_raw
            fs2_raw = pair_fs

        fs1 = fs2 = pair_fs

        # â”€â”€ Geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        border_w = max(MIN_BORDER_PX, int(pair_fs * BORDER_RATIO))
        has_pair = bool(line2 and line2_start is not None)
        pos_ratio = position_pct / 100.0

        # line_gap is the top-to-top distance between the two lines.
        # For a single-line chunk line_gap is unused for y-math.
        line_gap = int(pair_fs * line_spc_ratio) if has_pair else 0

        # ── Bottom-anchored vertical positioning ─────────────────────────────
        # pos_ratio (position_pct/100) is the BOTTOM EDGE of the caption block.
        # This means regardless of whether we are in single-line or pair mode,
        # the lowest pixel of the last caption line is always at h*pos_ratio.
        # No more jitter when transitioning between 1-word and 2-word chunks.
        #
        # In FFmpeg drawtext, y= is the TOP of the glyph bounding box.
        #   Single:  top_of_text  = h*pos_ratio - pair_fs
        #   Pair y2: top_of_line2 = h*pos_ratio - pair_fs        (bottom anchored)
        #   Pair y1: top_of_line1 = h*pos_ratio - pair_fs - line_gap  (above y2)
        if has_pair:
            y1_static = f"h*{pos_ratio:.4f}-{pair_fs + line_gap}"
            y2_static = f"h*{pos_ratio:.4f}-{pair_fs}"
        else:
            y1_static = f"h*{pos_ratio:.4f}-{pair_fs}"
            y2_static = None

        def _smooth_y(y_base_expr, word_start):
            return (
                f"({y_base_expr})"
                f"-if(lt(t-{word_start:.3f}\\,0.167)"
                f"\\,(1.0-(t-{word_start:.3f})/0.167)*{slide_px}"
                f"\\,0)"
            )

        if entrance_anim == "smooth":
            y1_anim = _smooth_y(y1_static, line1_start)
            y2_anim = _smooth_y(y2_static, line2_start) if has_pair else None
        else:
            y1_anim = y1_static
            y2_anim = y2_static

        if active_word_clr and has_pair:
            filters.append(_build_drawtext_entry(
                line1, font_path, fs1, active_word_clr, border_color,
                border_w, y1_anim, line1_start, line2_start - 0.01,
                shadow_color=shadow_color, shadow_x=shadow_x, shadow_y=shadow_y,
                fade_in=True
            ))
            filters.append(_build_drawtext_entry(
                line1, font_path, fs1, font_color, border_color,
                border_w, y1_static, line2_start, end,
                shadow_color=shadow_color, shadow_x=shadow_x, shadow_y=shadow_y,
                fade_in=False
            ))
            filters.append(_build_drawtext_entry(
                line2, font_path, fs2, active_word_clr, border_color,
                border_w, y2_anim, line2_start, end,
                shadow_color=shadow_color, shadow_x=shadow_x, shadow_y=shadow_y,
                fade_in=True
            ))
        elif active_word_clr:
            filters.append(_build_drawtext_entry(
                line1, font_path, fs1, active_word_clr, border_color,
                border_w, y1_anim, line1_start, end,
                shadow_color=shadow_color, shadow_x=shadow_x, shadow_y=shadow_y,
                fade_in=True
            ))
        else:
            filters.append(_build_drawtext_entry(
                line1, font_path, fs1, font_color, border_color,
                border_w, y1_anim, line1_start, end,
                shadow_color=shadow_color, shadow_x=shadow_x, shadow_y=shadow_y,
                fade_in=True
            ))
            if has_pair:
                filters.append(_build_drawtext_entry(
                    line2, font_path, fs2, highlight_color, border_color,
                    border_w, y2_anim, line2_start, end,
                    shadow_color=shadow_color, shadow_x=shadow_x, shadow_y=shadow_y,
                    fade_in=True
                ))

        if entrance_anim == "flash":
            flash_clr = active_word_clr or "#FFFFFF"
            flash_dur = 0.066
            filters.append(_build_drawtext_entry(
                line1, font_path, fs1, flash_clr, border_color,
                border_w, y1_static, line1_start, line1_start + flash_dur,
                shadow_color="", shadow_x=0, shadow_y=0,
            ))
            if has_pair:
                filters.append(_build_drawtext_entry(
                    line2, font_path, fs2, flash_clr, border_color,
                    border_w, y2_static, line2_start, line2_start + flash_dur,
                    shadow_color="", shadow_x=0, shadow_y=0,
                ))

    filter_str = ",".join(filters) if filters else "null"

    with open(filter_path, "w", encoding="utf-8") as f:
        f.write(filter_str)

    logger.info(
        f"Drawtext filter: {len(chunks)} chunks, base_font={base_font_size}px "
        f"[{style.get('name')}] uniform-pair sizing active"
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SRT (compatibility output)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _get_plan_srt_content(words, style) -> str:
    chunks = chunk_words(words, style.get("word_case", "upper"))
    lines = []
    for idx, (line1, line1_start, line2, line2_start, end_sec) in enumerate(chunks, 1):
        start_ts = _srt_ts(line1_start)
        end_ts = _srt_ts(end_sec)
        text = f"{line1} {line2}".strip() if line2 else line1
        lines += [str(idx), f"{start_ts} --> {end_ts}", text, ""]
    return "\n".join(lines)


def _normalize_srt(content: str) -> str:
    lines = [line.strip() for line in content.replace("\r\n", "\n").split("\n")]
    normalized_parts = []
    for line in lines:
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                t1 = parts[0].strip()
                t2 = parts[1].strip()
                def norm_ts(ts):
                    if "," in ts:
                        base, ms = ts.split(",")
                        try:
                            # Round ms to nearest 10ms (truncate the last digit to handle 1ms drift)
                            val = int(ms)
                            rounded_val = int(round(val / 10.0))
                            return f"{base},{rounded_val:02d}"
                        except ValueError:
                            return ts
                    return ts
                line = f"{norm_ts(t1)} --> {norm_ts(t2)}"
        else:
            # Subtitle text line: compare case-insensitively, ignore punctuation and spaces
            line = re.sub(r"[^\w\s*]", "", line.lower()).strip()
            line = re.sub(r"\s+", " ", line)
        normalized_parts.append(line)
    return "\n".join(normalized_parts)



def _concat_videos(video_1: str, video_2: str, output_path: str, logger) -> bool:
    """Concatenate video_1 and video_2 into output_path using filter_complex concat."""
    use_nvenc = getattr(config, "VIDEO_ENCODER", "libx264") == "h264_nvenc"
    ffmpeg_path = config.GPU_FFMPEG_PATH if use_nvenc else config.FFMPEG_PATH

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", video_1,
        "-i", video_2,
        "-filter_complex", "[0:v][0:a][1:v][1:a] concat=n=2:v=1:a=1 [v][a]",
        "-map", "[v]",
        "-map", "[a]"
    ]

    if use_nvenc:
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", config.NVENC_PRESET,
            "-b:v", "0",
            "-cq", str(getattr(config, "NVENC_CQ", 23)),
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
        ]
        
    cmd += [
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=True
        )
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
    except Exception as exc:
        stderr_msg = exc.stderr.decode() if hasattr(exc, "stderr") and exc.stderr else str(exc)
        logger.error(f"FFmpeg concatenation failed: {stderr_msg}")
        return False


def _generate_srt(words, srt_path, style, logger):
    content = _get_plan_srt_content(words, style)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug(f"SRT written: {len(words)} words")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Word extraction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _load_render_offsets(clips_dir, clip_name):
    """Read the render manifest written by the extractor and return a list of
    (source_start, source_end, playback_offset) tuples. The playback offset is
    where each AI-segment actually begins in the rendered clip, which already
    accounts for crossfade overlap between segments. Returns None when no
    manifest exists so callers transparently fall back to additive timing."""
    try:
        mpath = os.path.join(clips_dir, f"{clip_name}_raw.render.json")
        if not os.path.exists(mpath):
            return None
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        sources = data.get("segment_source") or []
        offsets = data.get("segment_offsets") or []
        if not sources or len(sources) != len(offsets):
            return None
        out = []
        for (pair, off) in zip(sources, offsets):
            try:
                out.append((float(pair[0]), float(pair[1]), float(off)))
            except (TypeError, ValueError, IndexError):
                return None
        return out
    except Exception:
        return None


def _match_render_offset(render_offsets, seg_start, seg_end, tol=0.05):
    """Find the playback offset for a source [start, end] segment by value match
    (robust to index drift when zero-length segments are filtered out). Returns
    None when nothing matches within tolerance."""
    if not render_offsets:
        return None
    for (s, e, off) in render_offsets:
        if abs(s - seg_start) <= tol and abs(e - seg_end) <= tol:
            return off
    return None


def _get_clip_local_words(clip, transcript, render_offsets=None):
    """
    Return words that overlap with any segment of this clip,
    remapped to clip-local time (0 = start of first segment).
    When render_offsets is supplied, each segment is anchored at its true
    rendered playback offset (crossfade-compensated) instead of the additive
    sum of source durations.
    """
    result = []
    local_offset = 0.0

    for seg in clip["segments"]:
        seg_start = _safe_float(seg.get("start"), 0.0)
        seg_end = _safe_float(seg.get("end"),   seg_start)
        seg_duration = max(0.0, seg_end - seg_start)

        seg_base = local_offset
        matched = _match_render_offset(render_offsets, seg_start, seg_end)
        if matched is not None:
            seg_base = matched

        for word in transcript.get("words", []):
            word_start = _safe_float(word.get("start"), 0.0)
            word_end = _safe_float(word.get("end"),   word_start)

            if word_end <= seg_start or word_start >= seg_end:
                continue

            clip_start = max(word_start, seg_start) - seg_start
            clip_end = min(word_end,   seg_end) - seg_start
            clip_end = max(clip_end, clip_start + 0.08)

            result.append({
                "word":  str(word.get("word", "")).strip(),
                "start": round(seg_base + clip_start, 3),
                "end":   round(seg_base + clip_end,   3),
            })

        local_offset += seg_duration

    # ── Deduplicate overlapping words ────────────────────────────────────────
    # When a word's timestamp straddles two adjacent segments it can appear
    # twice with nearly identical timing. Remove exact-duplicate (word+start)
    # entries and sort by start time so chunking always sees a clean timeline.
    seen_keys: set = set()
    deduped = []
    for w in sorted(result, key=lambda x: x["start"]):
        key = (w["word"].lower(), round(w["start"], 2))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(w)
    result = deduped

    return result


def _localize_words(words, clip, render_offsets=None):
    if not words:
        return []

    is_absolute = False
    timebase = clip.get("edited_words_timebase")
    if timebase == "absolute":
        is_absolute = True
    else:
        first_seg_start = clip["segments"][0]["start"] if clip.get("segments") else 0.0
        if first_seg_start > 2.0:
            first_w_start = words[0].get("start")
            if first_w_start is not None:
                if abs(first_w_start - first_seg_start) < abs(first_w_start - 0.0):
                    is_absolute = True

    if not is_absolute:
        local_words = []
        for w in words:
            local_words.append({
                "word": str(w.get("word", "")).strip(),
                "start": round(_safe_float(w.get("start"), 0.0), 3),
                "end": round(_safe_float(w.get("end"), 0.0), 3),
            })
        return local_words

    local_words = []
    local_offset = 0.0
    for seg in clip.get("segments", []):
        seg_start = _safe_float(seg.get("start"), 0.0)
        seg_end = _safe_float(seg.get("end"), seg_start)
        seg_duration = max(0.0, seg_end - seg_start)

        seg_base = local_offset
        matched = _match_render_offset(render_offsets, seg_start, seg_end)
        if matched is not None:
            seg_base = matched

        for w in words:
            w_start = _safe_float(w.get("start"), 0.0)
            w_end = _safe_float(w.get("end"), w_start)

            if w_end <= seg_start or w_start >= seg_end:
                continue

            clip_start = max(w_start, seg_start) - seg_start
            clip_end = min(w_end, seg_end) - seg_start
            clip_end = max(clip_end, clip_start + 0.08)

            local_words.append({
                "word": str(w.get("word", "")).strip(),
                "start": round(seg_base + clip_start, 3),
                "end": round(seg_base + clip_end, 3),
            })

        local_offset += seg_duration

    seen_keys = set()
    deduped = []
    for w in sorted(local_words, key=lambda x: x["start"]):
        key = (w["word"].lower(), round(w["start"], 2))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(w)
    return deduped



# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FFmpeg burn helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _burn_with_drawtext(raw_path, filter_path, cap_path, logger):
    """Burn drawtext subtitles into video."""
    with open(filter_path, "r", encoding="utf-8") as fh:
        chain = fh.read().strip()

    cx_path = filter_path.replace("_filter.txt", "_cx.txt")
    with open(cx_path, "w", encoding="utf-8") as fh:
        fh.write(f"[0:v]{chain}[vout]")

    cmd = [
        config.FFMPEG_PATH,
        "-i", raw_path,
        "-filter_complex_script", cx_path,
        "-map", "[vout]", "-map", "0:a?",
        "-c:a", "copy"
    ]
    cmd += _enc_args()
    cmd += ["-y", cap_path]
    _ffmpeg(cmd, logger)


def _burn_with_drawtext_and_music(raw_path, filter_path, music_path, cap_path,
                                  vol, music_offset, logger, tts_hook_duration=0.0):
    """Burn captions + mix background music in one pass.

    music_offset: seconds into the music file to start reading from.
    Uses -ss (input seek) so FFmpeg doesn't decode the skipped portion.
    """
    dur = max(0.1, _probe_dur(raw_path, logger))

    with open(filter_path, "r", encoding="utf-8") as f:
        drawtext_filters = f.read()

    has_whoosh = False
    whoosh_path = getattr(config, "TRANSITION_SFX", "")
    if tts_hook_duration > 0.05 and whoosh_path and os.path.exists(whoosh_path):
        has_whoosh = True

    fc = f"[0:v]{drawtext_filters}[v];"

    if tts_hook_duration > 0.05:
        delay_ms = int(tts_hook_duration * 1000)
        fc += f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume={vol:.3f},afade=t=out:st={max(0.0, dur - 0.35):.3f}:d=0.35,adelay={delay_ms}|{delay_ms}[music];"
    else:
        fc += f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume={vol:.3f},afade=t=out:st={max(0.0, dur - 0.35):.3f}:d=0.35[music];"

    fc += "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[base];"

    if has_whoosh:
        # slightly overlap whoosh before transition
        delay_ms = int(max(0.0, tts_hook_duration - 0.2) * 1000)
        fc += f"[2:a]volume=0.8,adelay={delay_ms}|{delay_ms}[whoosh];"
        fc += "[base][music][whoosh]amix=inputs=3:duration=first:dropout_transition=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    else:
        fc += "[base][music]amix=inputs=2:duration=first:dropout_transition=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]"

    combined_filter_path = filter_path.replace("_filter.txt", "_combined.txt")
    with open(combined_filter_path, "w", encoding="utf-8") as f:
        f.write(fc)

    # -ss before -i seeks efficiently (no decode of skipped portion)
    music_input = []
    if music_offset > 0.5:
        music_input += ["-ss", f"{music_offset:.3f}"]
    music_input += ["-stream_loop", "-1", "-t", f"{dur:.3f}", "-i", music_path]
    
    extra_inputs = []
    if has_whoosh:
        extra_inputs += ["-i", whoosh_path]

    cmd = [
        config.FFMPEG_PATH,
        "-i", raw_path,
    ] + music_input + extra_inputs + [
        "-filter_complex_script", combined_filter_path,
        "-map", "[v]", "-map", "[a]",
    ]
    cmd += _enc_args()
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-y", cap_path]
    _ffmpeg(cmd, logger)


def _mix_background_music(video_path, music_path, output_path, vol, music_offset, logger, tts_hook_duration=0.0):
    """Mix music under clip audio without any caption burn."""
    dur = max(0.1, _probe_dur(video_path, logger))

    # -ss before -i seeks efficiently
    music_input = []
    if music_offset > 0.5:
        music_input += ["-ss", f"{music_offset:.3f}"]
    music_input += ["-stream_loop", "-1", "-t", f"{dur:.3f}", "-i", music_path]

    has_whoosh = False
    whoosh_path = getattr(config, "TRANSITION_SFX", "")
    if tts_hook_duration > 0.05 and whoosh_path and os.path.exists(whoosh_path):
        has_whoosh = True

    fc = ""
    if tts_hook_duration > 0.05:
        delay_ms = int(tts_hook_duration * 1000)
        fc += f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume={vol:.3f},afade=t=out:st={max(0.0, dur - 0.35):.3f}:d=0.35,adelay={delay_ms}|{delay_ms}[music];"
    else:
        fc += f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume={vol:.3f},afade=t=out:st={max(0.0, dur - 0.35):.3f}:d=0.35[music];"
        
    fc += "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a0];"

    if has_whoosh:
        delay_ms = int(max(0.0, tts_hook_duration - 0.2) * 1000)
        fc += f"[2:a]volume=0.8,adelay={delay_ms}|{delay_ms}[whoosh];"
        fc += "[a0][music][whoosh]amix=inputs=3:duration=first:dropout_transition=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    else:
        fc += "[a0][music]amix=inputs=2:duration=first:dropout_transition=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]"

    extra_inputs = []
    if has_whoosh:
        extra_inputs += ["-i", whoosh_path]

    cmd = [
        config.FFMPEG_PATH,
        "-i", video_path,
    ] + music_input + extra_inputs + [
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-y", output_path,
    ]
    _ffmpeg(cmd, logger)


def _copy_or_mix(raw_path, filter_path, cap_path, music_path, vol,
                 music_enabled, music_offset, logger, tts_hook_duration=0.0):
    """Copy or music-mix when there are no words to caption."""
    if music_enabled:
        try:
            _mix_background_music(raw_path, music_path,
                                  cap_path, vol, music_offset, logger, tts_hook_duration)
            return
        except Exception as exc:
            logger.warning(f"Music mix fallback failed: {exc}")
    import shutil as _sh
    _sh.copy2(raw_path, cap_path)


def _cleanup_caption_artifacts(filter_path, srt_path, logger):
    """Remove caption render scratch files after a successful burn."""
    paths = [
        filter_path,
        filter_path.replace("_filter.txt", "_combined.txt"),
        filter_path.replace("_filter.txt", "_cx.txt"),
    ]
    if not getattr(config, "CAPTION_SAVE_SRT", False):
        paths.append(srt_path)

    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.debug(f"Could not remove caption artifact {path}: {exc}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Utilities
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _srt_ts(seconds: float) -> str:
    """Seconds -> SRT timestamp HH:MM:SS,mmm"""
    seconds = round(seconds, 3)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
        if s >= 60:
            s -= 60
            m += 1
            if m >= 60:
                m -= 60
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"



def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _enc_args() -> list:
    """
    Always attempt NVENC (GPU) encoding first.
    If NVENC is unavailable on this machine, _ffmpeg() catches the failure
    and automatically retries with libx264 (CPU) - no extra configuration needed.
    """
    nvenc_cq = getattr(config, "NVENC_CQ", 23)
    return ["-c:v", "h264_nvenc", "-preset", config.NVENC_PRESET, "-b:v", "0", "-cq", str(nvenc_cq)]


def _probe_dur(video_path: str, logger) -> float:
    """Get clip duration via ffprobe."""
    cmd = [
        config.FFPROBE_PATH, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return float(r.stdout.strip())
    except Exception as exc:
        logger.warning(f"ffprobe duration failed: {exc}")
        return 0.0


def _ffmpeg(cmd: list, logger) -> None:
    """
    Run FFmpeg. On NVENC failure automatically retries with libx264 (silent fallback).
    Logs whether GPU or CPU was used so you can verify in the console.
    """
    if config.LOG_FFMPEG_COMMANDS:
        logger.debug("FFmpeg: " + " ".join(cmd))

    r = subprocess.run(
        cmd, capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if r.returncode == 0:
        if "nvenc" in " ".join(cmd).lower():
            logger.info("GPU (NVENC) encode: OK")
        return

    # NVENC failed â€” retry transparently with libx264
    if "nvenc" in " ".join(cmd).lower():
        logger.warning("NVENC unavailable â€” falling back to libx264 (CPU)")
        new_cmd = []
        skip_next = False
        for i, arg in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if arg == "h264_nvenc":
                new_cmd.append("libx264")
            elif arg in ("-b:v", "-cq") and i + 1 < len(cmd):
                # Drop NVENC-only bitrate/CQ flags; CRF added below
                skip_next = True
                continue
            elif arg == "-preset" and i + 1 < len(cmd) and cmd[i + 1] == config.NVENC_PRESET:
                new_cmd += ["-preset", config.X264_PRESET]
                skip_next = True
            else:
                new_cmd.append(arg)
        # Ensure we use standard FFmpeg executable for CPU fallback
        if new_cmd:
            new_cmd[0] = config.FFMPEG_PATH
        # Insert CRF after the codec flag
        try:
            ix = new_cmd.index("libx264")
            new_cmd[ix + 1:ix + 1] = ["-crf", str(config.X264_CRF)]
        except ValueError:
            pass
        r = subprocess.run(
            new_cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if r.returncode == 0:
            return
        logger.error(f"libx264 fallback also failed: {r.stderr[:2000]}")
        raise RuntimeError(f"FFmpeg caption burn failed: {r.stderr[:2000]}")

    logger.error(f"FFmpeg error: {r.stderr[:2000]}")
    raise RuntimeError(f"FFmpeg caption burn failed: {r.stderr[:2000]}")


def _add_emojis_to_words(words):
    import re
    emoji_map = {
        "money": "💰",
        "cash": "💰",
        "dollar": "💵",
        "fire": "🔥",
        "growth": "🚀",
        "rocket": "🚀",
        "heart": "❤️",
        "love": "❤️",
        "win": "🏆",
        "winner": "🏆",
        "lose": "❌",
        "success": "✅",
        "cool": "😎",
        "brain": "🧠",
        "idea": "💡",
        "light": "💡",
        "time": "⏱️"
    }
    emoji_words = []
    for w in words:
        word_text = w.get("word", "")
        # Remove common punctuation for matching
        clean = re.sub(r"[^\w]", "", word_text.lower())
        emoji = emoji_map.get(clean)
        if emoji:
            # Check if emoji is already in the word to avoid duplicates
            if emoji not in word_text:
                word_text = f"{word_text} {emoji}"
        emoji_words.append({
            "word": word_text,
            "start": w.get("start", 0.0),
            "end": w.get("end", 0.0)
        })
    return emoji_words


def _build_hook_title_drawtext_entry(text, font_path, font_size, font_color, border_color, border_width, y_expr, start_time=0.0, end_time=3.0):
    # Escape without stripping punctuation
    safe_text = (
        text.replace("\\", "\\\\")
        .replace("'", "'\\\\\\''")
        .replace(":", "\\:")
        .replace(";", "\\;")
    )
    safe_font = font_path.replace("\\", "/").replace(":", "\\:")
    safe_font_escaped = safe_font.replace("'", "'\\\\\\''")

    font_file_part = f"fontfile='{safe_font_escaped}':" if font_path else ""

    # Render with translucent box
    return (
        f"drawtext="
        f"{font_file_part}"
        f"text='{safe_text}':"
        f"fontsize={font_size}:"
        f"fontcolor={font_color}:"
        f"bordercolor={border_color}:"
        f"borderw={border_width}:"
        f"box=1:boxcolor=black@0.55:boxborderw=15:"
        f"x=(w-text_w)/2:"
        f"y={y_expr}:"
        f"expansion=none:"
        f"fix_bounds=1:"
        f"enable='between(t,{start_time:.3f},{start_time+end_time:.3f})'"
    )
