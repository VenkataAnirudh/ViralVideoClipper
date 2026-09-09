"""
diarizer.py — STUB

pyannote/speaker-diarization-3.1 has been removed.
Speaker tracking uses pure CV scoring via InsightFace (see speaker_tracking.py).

This stub exists so no import statements elsewhere need changing.
"""


def run_diarization(video_path: str, logger) -> list:
    """Stub — always returns empty list. CV scoring handles all cases."""
    logger.info("  diarizer: pyannote removed; using CV-only speaker tracking")
    return []
