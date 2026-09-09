"""Small GPU-aware worker-count helper for FFmpeg stages."""

import os
import subprocess

import config


def ffmpeg_worker_count(stage_name: str, requested_max: int, logger) -> int:
    """Return a safe FFmpeg worker count for the current GPU memory state."""
    requested_max = max(1, int(requested_max or 1))
    if requested_max <= 1:
        logger.info(f"{stage_name}: GPU parallelism disabled (workers=1)")
        return 1

    free_mb = _query_free_vram_mb()
    if free_mb is None:
        logger.info(f"{stage_name}: nvidia-smi unavailable; using one FFmpeg worker")
        return 1

    reserve = max(0, int(getattr(config, "GPU_MEMORY_RESERVE_MB", 768)))
    per_job = max(256, int(getattr(config, "GPU_MEMORY_PER_FFMPEG_JOB_MB", 1400)))
    usable = max(0, free_mb - reserve)
    workers = max(1, min(requested_max, usable // per_job))
    logger.info(
        f"{stage_name}: free_vram={free_mb}MB reserve={reserve}MB "
        f"per_job={per_job}MB workers={workers}/{requested_max}"
    )
    return workers


def _query_free_vram_mb() -> int | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return max(values) if values else None
