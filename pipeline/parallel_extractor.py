# fileName: pipeline/parallel_extractor.py
"""
Parallel video extractor wrapper for Clip Tool v4.
Coordinates concurrent clip extraction threads and synchronizes GPU tracking resources.

Resource allocation model:
──────────────────────────────────────────────────────────────────────
  gpu_tracking_lock = Semaphore(1)   — exactly 1 GPU tracking at a time
  ThreadPoolExecutor(workers=3)      — 3 concurrent clip pipelines

  Worker 1: [GPU TRACK clip_01] ──► [FFmpeg RENDER clip_01]
  Worker 2:     wait on GPU…   [GPU TRACK clip_02] ──► [FFmpeg RENDER clip_02]
  Worker 3:          wait…          wait…   [GPU TRACK clip_03] ──► [RENDER]

  After ramp-up, the pipeline stabilises:
  ─ GPU is always busy tracking the next clip (no idle time)
  ─ FFmpeg renders overlap on CPU/NVENC while GPU tracks
  ─ Face backend (buffalo_l ~950 MB) stays in VRAM as a singleton
    → loaded ONCE, reused across all clips (saves ~55s × N reloads)
──────────────────────────────────────────────────────────────────────
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

import config
from pipeline import extractor
from pipeline import speaker_tracking

# Only one clip can use the GPU face detection/tracking engine at a time.
# The Semaphore(1) serialises GPU access so that:
#   - clip N tracks on GPU while clips N-1, N-2 render on CPU/NVENC concurrently
#   - the singleton face backend is never accessed from two threads at once
gpu_tracking_lock = threading.Semaphore(1)


def _process_one_staggered(
    plan_idx: int,
    clip: Dict[str, Any],
    video_path: Path,
    clips_dir: Path,
    job_path: Path,
    settings: Dict[str, Any],
    logger: Any,
    manual_templates: Any,
    diarization: Any
) -> tuple[int, str | None, bool]:
    """Extract one clip with staggered GPU locking.

    The gpu_tracking_lock is passed down to speaker_tracking.process_clip():
      acquire lock → face detect/track → release lock → FFmpeg render (no lock)
    This ensures the GPU is freed ASAP for the next clip's tracking phase.
    """
    clip_name = str(clip.get("clip_name", f"clip_{plan_idx + 1:02d}")).strip()
    out_path = clips_dir / f"{clip_name}_raw.mp4"

    if extractor._valid_clip(out_path):
        size_mb = out_path.stat().st_size / (1024 * 1024)
        extractor._log(logger, "info", f"  -> {out_path.name} already exists ({size_mb:.1f} MB); skipping")
        return plan_idx, str(out_path), True

    segments = extractor._clip_segments(clip)
    if not segments:
        extractor._log(logger, "error", f"  -> {clip_name}: no valid segments in clips_plan entry")
        return plan_idx, None, False

    extractor._log(logger, "info",
                   f"Extracting {clip_name} (Parallel Staggered): "
                   f"{clip.get('title', '')} ({len(segments)} segment(s))")

    # Pass the gpu_tracking_lock so speaker_tracking serialises GPU access
    ok = extractor._extract_single_clip(
        video_path=video_path,
        job_dir=job_path,
        clip_name=clip_name,
        segments=segments,
        out_path=out_path,
        settings=settings,
        logger=logger,
        manual_templates_path=manual_templates,
        diarization_path=diarization,
        gpu_lock=gpu_tracking_lock,
    )

    if ok and extractor._valid_clip(out_path):
        extractor._log(logger, "info", f"  -> extracted {out_path.name}")
        return plan_idx, str(out_path), True
    else:
        extractor._log(logger, "error", f"  -> {clip_name}: Clip Tool v4 extraction failed")
        return plan_idx, None, False


def extract_clips_parallel(
    job_dir: str,
    clips_plan: List[Dict[str, Any]],
    settings: Dict[str, Any],
    logger: Any,
) -> List[str]:
    """Extract raw clips in parallel using ThreadPoolExecutor with staggered GPU locking.

    Guarantees:
      1. Exactly 1 CUDA tracking task runs at a time (gpu_tracking_lock)
      2. Multiple FFmpeg renders run concurrently (no lock needed)
      3. Face backend (buffalo_l) is loaded ONCE and reused across all clips
      4. Errors are reported in real-time as each clip completes
      5. GPU resources are cleaned up after all clips finish
    """
    job_path = Path(job_dir)
    settings = settings or {}
    clips_plan = list(clips_plan or [])
    clips_dir = job_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_path = extractor._read_meta_video_path(job_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    manual_templates = extractor._manual_templates_path(job_path, settings)
    if not manual_templates:
        manual_templates = extractor._maybe_launch_template_tool(job_path, video_path, settings, logger)
    diarization = extractor._diarization_path(job_path, settings)

    resume_marker = (
        extractor._setting(settings, "extract_resume_from_clip", "resume_from_clip", "resume_start_clip", default=None)
        or getattr(config, "EXTRACTION_RESUME_FROM_CLIP", "")
    )
    resume_start = extractor._resolve_resume_start(clips_plan, resume_marker, logger)

    extracted: List[str] = []
    if resume_start > 0:
        for skipped in clips_plan[:resume_start]:
            skipped_name = str(skipped.get("clip_name", "")).strip()
            if not skipped_name:
                continue
            skipped_path = clips_dir / f"{skipped_name}_raw.mp4"
            if extractor._valid_clip(skipped_path):
                extracted.append(str(skipped_path))
            else:
                extractor._log(logger, "warning",
                               f"Resume skipped {skipped_name}, but {skipped_path.name} is not present/valid")

    # Worker count: 3 workers gives good GPU/CPU overlap with Semaphore(1)
    workers = int(settings.get("extraction_workers", getattr(config, "EXTRACTION_WORKERS", 3)))
    if workers < 1:
        workers = 3

    clips_to_extract = list(enumerate(clips_plan))[resume_start:]
    total_clips = len(clips_to_extract)

    extractor._log(logger, "info",
                   f"Starting staggered parallel extraction: {total_clips} clips, "
                   f"{workers} workers, GPU lock=Semaphore(1)")

    t0_all = time.time()
    results: Dict[int, str] = {}
    completed_count = 0
    failed_count = 0

    # as_completed INSIDE the with-block: results are collected as clips finish,
    # not after all clips are done.  This gives real-time progress and error reporting.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for plan_idx, clip in clips_to_extract:
            fut = executor.submit(
                _process_one_staggered,
                plan_idx,
                clip,
                video_path,
                clips_dir,
                job_path,
                settings,
                logger,
                manual_templates,
                diarization,
            )
            futures[fut] = plan_idx

        for fut in as_completed(futures):
            p_idx = futures[fut]
            clip_name = str(clips_plan[p_idx].get("clip_name", f"clip_{p_idx + 1:02d}")).strip()
            try:
                _, path, success = fut.result()
                if success and path:
                    results[p_idx] = path
                    completed_count += 1
                    elapsed = time.time() - t0_all
                    extractor._log(logger, "info",
                                   f"  [{completed_count}/{total_clips}] {clip_name} done "
                                   f"({elapsed:.0f}s elapsed)")
                else:
                    failed_count += 1
                    extractor._log(logger, "error",
                                   f"  [{completed_count + failed_count}/{total_clips}] "
                                   f"{clip_name} FAILED")
            except Exception as e:
                failed_count += 1
                extractor._log(logger, "error",
                               f"  -> {clip_name}: unhandled parallel worker error: {e}")

    # Reconstruct list in plan order
    for plan_idx in range(resume_start, len(clips_plan)):
        if plan_idx in results:
            extracted.append(results[plan_idx])

    elapsed_total = time.time() - t0_all

    # Clean up GPU resources now that ALL clips are done.
    # The singleton face backend can be released from VRAM.
    try:
        speaker_tracking.cleanup_gpu_resources()
        extractor._log(logger, "info", "[GPU] Singleton face backend released from VRAM")
    except Exception as e:
        extractor._log(logger, "warning", f"[GPU] cleanup_gpu_resources error (non-fatal): {e}")

    extractor._log(logger, "info",
                   f"Parallel extraction complete: {len(extracted)}/{len(clips_plan)} raw clips "
                   f"({elapsed_total:.0f}s total, {failed_count} failed)")
    return extracted
