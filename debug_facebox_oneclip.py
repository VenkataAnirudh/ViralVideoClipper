"""Render ONE raw clip from a job's clips_plan.json with the face-detection
bounding box burned into the video (FACE_DEBUG_BOX overlay) plus the normal
segment/join crossfade transitions.

Debug utility — the output is written as <clip>_raw_facebox.mp4 so it can
never be mistaken for (or block) a production raw clip.

Usage:
    python debug_facebox_oneclip.py --job "outputs\\<job folder>"
    python debug_facebox_oneclip.py --job "outputs\\<job folder>" --clip clip_03
    python debug_facebox_oneclip.py --job "outputs\\<job folder>" --dry-run

Behavior:
  * Picks a RANDOM multi-segment clip by default (>=2 segments, so the join
    crossfade is actually visible). --clip overrides.
  * Reuses the job's manual_templates.json if present; otherwise the template
    snap tool pops up (snap templates, close it, and the render continues).
  * Renders through the exact production path (_extract_single_clip), so the
    scene-cut xfade (CLIP_SEGMENT_XFADE) and join xfade (CLIP_JOIN_XFADE) are
    applied identically to a real run.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Must be set BEFORE config is imported anywhere.
os.environ["FACE_DEBUG_BOX"] = os.environ.get("FACE_DEBUG_BOX", "true")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from pipeline.extractor import (  # noqa: E402
    _clip_segments,
    _diarization_path,
    _extract_single_clip,
    _manual_templates_path,
    _maybe_launch_template_tool,
    _read_meta_video_path,
    _valid_clip,
)


def log(msg: str) -> None:
    print(f"[facebox] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="Job output folder (contains clips_plan.json)")
    ap.add_argument("--clip", default="", help="clip_name to render; default = random multi-segment clip")
    ap.add_argument("--dry-run", action="store_true", help="Only show which clip would be rendered")
    args = ap.parse_args()

    job_dir = Path(args.job).resolve()
    plan_path = job_dir / "clips_plan.json"
    if not plan_path.exists():
        log(f"ERROR: no clips_plan.json in {job_dir}")
        return 1

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    clips = plan if isinstance(plan, list) else (plan.get("clips") or [])
    if not clips:
        log("ERROR: clips plan is empty")
        return 1

    if args.clip:
        chosen = next((c for c in clips if str(c.get("clip_name", "")).strip() == args.clip), None)
        if chosen is None:
            log(f"ERROR: clip '{args.clip}' not found in plan ({len(clips)} clips)")
            return 1
    else:
        multi = [c for c in clips if len(_clip_segments(c)) >= 2]
        pool = multi or clips
        chosen = random.choice(pool)
        if not multi:
            log("note: no multi-segment clips in plan; join transition won't appear")

    clip_name = str(chosen.get("clip_name", "")).strip() or "clip_unknown"
    segments = _clip_segments(chosen)
    if not segments:
        log(f"ERROR: {clip_name} has no valid segments")
        return 1

    total = sum(e - s for s, e in segments)
    log(f"selected {clip_name}: {len(segments)} segment(s), {total:.1f}s source total")
    for i, (s, e) in enumerate(segments, 1):
        log(f"  seg {i}: {s:.2f} -> {e:.2f}  ({e - s:.2f}s)")
    log(f"FACE_DEBUG_BOX={getattr(config, 'FACE_DEBUG_BOX', False)}  "
        f"CLIP_SEGMENT_XFADE={getattr(config, 'CLIP_SEGMENT_XFADE', True)}  "
        f"CLIP_JOIN_XFADE={getattr(config, 'CLIP_JOIN_XFADE', True)}  "
        f"XFADE_FRAMES={getattr(config, 'XFADE_FRAMES', 5)}")

    if args.dry_run:
        log("dry run only — nothing rendered")
        return 0

    video_path = _read_meta_video_path(job_dir)
    if not video_path.exists():
        log(f"ERROR: source video not found: {video_path}")
        return 1
    log(f"source video: {video_path.name}")

    settings = {}
    sp = job_dir / "settings.json"
    if sp.exists():
        try:
            settings = json.loads(sp.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"warning: could not read settings.json ({exc}); using defaults")

    clips_dir = job_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    out_path = clips_dir / f"{clip_name}_raw_facebox.mp4"
    if out_path.exists():
        out_path.unlink()

    # Reuse saved framing templates when they exist; otherwise pop the snap
    # tool once (delete manual_templates.json in the job folder to re-snap).
    manual_templates = _manual_templates_path(job_dir, settings)
    if manual_templates:
        log(f"reusing manual templates: {manual_templates}")
    else:
        log("no manual templates found — launching the template snap tool "
            "(snap your speakers, then close it)")
        manual_templates = _maybe_launch_template_tool(job_dir, video_path, settings, log)
        if manual_templates:
            log(f"templates saved: {manual_templates}")
        else:
            log("no templates snapped; continuing with auto discovery")

    diarization = _diarization_path(job_dir, settings)
    log(f"diarization file: {diarization or '(none — computed per-clip as usual)'}")

    t0 = time.time()
    ok = _extract_single_clip(
        video_path=video_path,
        job_dir=job_dir,
        clip_name=clip_name,
        segments=segments,
        out_path=out_path,
        settings=settings,
        logger=log,
        manual_templates_path=manual_templates,
        diarization_path=diarization,
        gpu_lock=None,
    )
    elapsed = time.time() - t0

    if ok and _valid_clip(out_path):
        size_mb = out_path.stat().st_size / (1024 * 1024)
        log(f"DONE in {elapsed:.1f}s -> {out_path}  ({size_mb:.1f} MB)")
        return 0
    log(f"FAILED after {elapsed:.1f}s — see messages above (partial output removed)")
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
