#!/usr/bin/env python3
"""
PodcastClipper v3
==================
Template-first, diarization-aware, face-anchored 9:16 reframer for podcast clips.

Core design:
- TransNetV2 scene detection
- nearby-frame template discovery (pre/post cut)
- pHash-based template clustering
- InsightFace / MediaPipe / Haar fallback face detection
- face embedding speaker linking
- shirt-color fallback
- sparse 250 ms crop planning
- FFmpeg render with piecewise-linear crop paths
- tkinter GUI

Defaults are wired for the project layout used in the existing workflow:
- main env:  D:\Coding\Video Clipper\venv
- diarization env:  D:\Coding\Video Clipper\venv1
- ffmpeg:   D:\Coding\Video Clipper\ffmpeg-gpu\ffmpeg.exe
"""

from __future__ import annotations

import bisect
import dataclasses
from dataclasses import dataclass, asdict, field
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

try:
    import config  # type: ignore
except Exception:  # speaker_tracking may be imported standalone
    config = None  # type: ignore

try:
    from PIL import Image
    import imagehash
    HAS_IMAGEHASH = True
except Exception:
    HAS_IMAGEHASH = False

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    HAS_CUPY = False

try:
    import av
    HAS_AV = True
except Exception:
    av = None  # type: ignore
    HAS_AV = False


# ======================================================================
# Config
# ======================================================================

APP_NAME = "PodcastClipper v3"
DEFAULT_BASE_DIR = Path(r"D:\Coding\Video Clipper")
DEFAULT_MAIN_PY = DEFAULT_BASE_DIR / "venv" / "Scripts" / "python.exe"
DEFAULT_DIAR_PY = DEFAULT_BASE_DIR / "venv1" / "Scripts" / "python.exe"
DEFAULT_FFMPEG = DEFAULT_BASE_DIR / "ffmpeg-bin" / "ffmpeg.exe"

# Scene detection
TRANSNET_THRESHOLD = 0.25
TRANSNET_MIN_FRAMES = 10
PYSCENE_THRESHOLD = 1.2
SCDET_PRIMARY_THRESH = 2.5
SCDET_SECONDARY_THRESH = 1.5
SCDET_GAP_MIN = 2.0
CUT_DEDUP_GAP = 0.35

# Template discovery
PRE_CUT_SAMPLES = [0.30, 0.15, 0.05]
POST_CUT_SAMPLES = [0.05, 0.15, 0.30, 0.50, 0.80]
PHASH_THRESHOLD = 12
MAX_TEMPLATES = 16

# Detection / tracking
FACE_MIN_SCORE = 0.45
TRACK_SAMPLE_INTERVAL = 0.25
STABLE_START_OFFSET_SEC = 0.10
POST_CUT_BOOST_DURATION = 0.4
POST_CUT_BOOST_FRAMES = 3
LOOKAHEAD = 0.10
EMA_ALPHA = 0.35
MAX_PAN_PER_SEC = 900.0
SNAP_JUMP_THRESH = 70.0
NOSE_ROOM_FACTOR = 0.08
MAX_NOSE_FRAC = 0.15
HEAD_ROOM_FRAC = 0.45

# Crop / render
TARGET_W = 2160
TARGET_H = 3840
ANALYSIS_MAX_SIDE = 960
ANALYSIS_FRAME_STEP = 4
MIN_CLIP_BYTES = 4096
PROGRESS_LOG_INTERVAL = 1.5

# Temporal/template classification
TEMPLATE_WINDOW_RADIUS = 2.0
TEMPLATE_STAGE_B_TOP_K = 3
HIDDEN_CUT_CONFIDENCE_DROP = 0.35
WIDE_SHOT_FACE_FRACTION = 0.40
WIDE_SHOT_CENTER_BIAS = 0.55
EXPORT_USE_NVENC = True

# Fallback thresholds
FACE_CONF_FALLBACK = 0.55
# Wide-shot: max normalized distance (in slot-diagonals) a detection may sit
# from the active slot center and still be accepted as that speaker. Beyond
# this, fall back to the manual slot anchor instead of a wrong person.
WIDE_SLOT_MATCH_MAX = 1.25
ZERO_FACE_BOUNDARY_TRIM_MAX_FRAMES = 3
ZERO_FACE_BOUNDARY_TRIM_MAX_SEC = 0.12
EMBED_SIM_THRESHOLD = 0.45
SHIRT_SIM_THRESHOLD = 0.40
SLOT_MATCH_MARGIN = 0.05

# Cache / storage
TEMPLATE_DIR_NAME = "templates"
REGISTRY_FILE = "template_registry.json"
SCENE_FILE = "scene_cuts.json"
DIAR_FILE = "diarization.json"
DIAR_DIR_NAME = "diarization"
KEYFRAME_FILE = "crop_keyframes.json"
TRACK_FILE = "tracks.json"
LOG_FILE = "pipeline.log"
DEBUG_LOG_FILE = "debug.txt"
SEGMENT_ANALYSIS_FILE = "segment_analysis.txt"
TRIM_MANIFEST_FILE = "trim_manifest.json"
TIMELINE_MAP_FILE = "timeline_map.json"
CLASSIFICATION_CACHE_DIR_NAME = "classification_cache"
CLASSIFICATION_CACHE_VERSION = 2
TEMPLATE_MANIFEST_FILE = "template_manifest.json"
VERBOSE_STAGE_LOGGING = True


def _write_segment_analysis(
    job_dir: Path,
    plan: "SegmentPlan",
    template_display_name: str,
    faces_detected: int,
    total_ticks: int,
    face_detect_rate: float,
) -> None:
    """
    FIX 10 â€” Automated segment analysis log.
    Appends a standardized data block to segment_analysis.txt after each segment
    completes its planning and rendering cycle.
    """
    out_path = job_dir / SEGMENT_ANALYSIS_FILE
    ensure_dir(job_dir)
    start_ms = int(plan.start * 1000)
    end_ms = int(plan.end * 1000)
    dur_ms = end_ms - start_ms
    block = (
        f"\n{'='*70}\n"
        f"SEGMENT   : #{plan.index + 1:04d}\n"
        f"TIMELINE  : {start_ms} ms -> {end_ms} ms  (dur={dur_ms} ms)\n"
        f"SPEAKER   : {plan.speaker}\n"
        f"TEMPLATE  : id={plan.template_id}  name={template_display_name}\n"
        f"SLOT      : {plan.target_slot or 'N/A'}\n"
        f"INITIAL_CROP: x={plan.crop_expr_x}  y={plan.crop_expr_y}  "
        f"w={plan.crop_w}  h={plan.crop_h}\n"
        f"FACE_TRACK: {faces_detected}/{total_ticks} ticks "
        f"({face_detect_rate:.1f}%)  keyframes={len(plan.keyframes)}\n"
        f"SNAPS     : {sum(1 for k in plan.keyframes if k.is_snap)}\n"
        f"{'='*70}\n"
    )
    try:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        pass

# ======================================================================
# Data classes
# ======================================================================

@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str

@dataclass
class FaceObs:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cx: float
    cy: float
    area: float
    yaw: float = 0.0
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    shirt_hist: Optional[np.ndarray] = None
    shirt_rgb: Optional[Tuple[int, int, int]] = None
    torso_conf: float = 0.0

@dataclass
class FaceSlot:
    slot_name: str
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    center_norm: Tuple[float, float]
    bbox_norm: Tuple[float, float, float, float]
    confidence: float
    embedding: Optional[List[float]] = None
    shirt_hist: Optional[List[float]] = None
    shirt_rgb: Optional[Tuple[int, int, int]] = None
    speaker_label: Optional[str] = None
    slot_order: int = 0

@dataclass
class TemplateRecord:
    template_id: int
    phash: List[int]
    segment_indices: List[int]
    type: str
    canonical_frame: str
    canonical_timestamp: float
    n_faces: int
    face_slots: List[FaceSlot] = field(default_factory=list)
    crop_anchor: Dict[str, float] = field(default_factory=dict)
    speaker_votes: Dict[str, int] = field(default_factory=dict)
    speaker_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    shirt_signatures: Dict[str, List[float]] = field(default_factory=dict)
    slot_geometry: Dict[str, Dict[str, float]] = field(default_factory=dict)
    display_name: str = ""

@dataclass
class KeyframePoint:
    t_rel: float
    crop_x: float
    crop_y: float
    is_snap: bool = False
    confidence: float = 1.0
    # Source-coordinate face bbox (x1, y1, x2, y2) of the focus face at this
    # tick; None when the subject was lost/held. Only consumed by the
    # FACE_DEBUG_BOX overlay — has no effect on cropping.
    face_box: Optional[Tuple[float, float, float, float]] = None

@dataclass
class SegmentPlan:
    index: int
    start: float
    end: float
    template_id: int
    speaker: str
    crop_w: int
    crop_h: int
    keyframes: List[KeyframePoint] = field(default_factory=list)
    target_slot: Optional[str] = None
    crop_expr_x: str = ""
    crop_expr_y: str = ""

@dataclass
class ClassifiedSegment:
    index: int
    start: float
    end: float
    template_id: int
    confidence: float = 0.0
    hidden: bool = False
    scores: Dict[int, float] = field(default_factory=dict)

@dataclass
class SegmentTrimInfo:
    index: int
    original_start: float
    original_end: float
    render_start: float
    render_end: float
    start_trim: float
    end_trim: float
    output_start: float
    output_end: float


# ======================================================================
# Small helpers
# ======================================================================

def is_windows() -> bool:
    return platform.system().lower().startswith("win")

def quote_path_for_ffmpeg(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")

def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_remove(path: Path | str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass

def file_stem_no_ext(p: str) -> str:
    return Path(p).stem

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def normalize_vec(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < eps:
        return vec.astype(np.float32, copy=True)
    return (vec / n).astype(np.float32)

def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    aa = normalize_vec(np.asarray(a, dtype=np.float32))
    bb = normalize_vec(np.asarray(b, dtype=np.float32))
    return float(np.dot(aa, bb))

def bhattacharyya_hist(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    if h1 is None or h2 is None:
        return 1.0
    a = np.asarray(h1, dtype=np.float32).ravel()
    b = np.asarray(h2, dtype=np.float32).ravel()
    if a.size != b.size:
        return 1.0
    a = a / (a.sum() + 1e-9)
    b = b / (b.sum() + 1e-9)
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))

def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def parse_time(s: str) -> float:
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0



# ======================================================================
# FFmpeg expression helpers
# ======================================================================

def _escape_ffmpeg_expr(expr: str) -> str:
    """Escape characters that break FFmpeg filtergraph parsing."""
    if expr is None:
        return ""
    return (
        str(expr)
        .replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .strip()
    )


def _center_crop_expr() -> Tuple[str, str]:
    return "(in_w-out_w)/2", "0"


def _registry_cuts_match(expected: Sequence[float], actual: Sequence[float], tol: float = 0.25) -> bool:
    if len(expected) != len(actual):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(expected, actual))

# ======================================================================
# Logging
# ======================================================================

class Logger:
    def __init__(self, log_fn=None, file_path: Optional[Path] = None, extra_file_path: Optional[Path] = None):
        self.log_fn = log_fn
        self.file_path = file_path
        self.extra_file_path = extra_file_path
        self._lock = threading.Lock()

    def _write_file(self, path: Optional[Path], text: str) -> None:
        if path is None:
            return
        try:
            ensure_dir(path.parent)
            with path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass

    def __call__(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {msg}"
        with self._lock:
            self._write_file(self.file_path, text)
            self._write_file(self.extra_file_path, text)
            if self.log_fn is not None:
                try:
                    self.log_fn(text)
                except Exception:
                    pass
            else:
                print(text)

# ======================================================================
# JSON helpers

# ======================================================================

def write_json(path: Path | str, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_env_file(path: Path | str, override: bool = False) -> int:
    """
    Small .env loader for this standalone tool. It intentionally avoids adding
    a dependency and only supports simple KEY=VALUE lines.
    """
    env_path = Path(path)
    if not env_path.exists():
        return 0
    loaded = 0
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded += 1
    except Exception:
        return loaded
    return loaded

def _get_hf_token() -> str:
    return (
        os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )

def _ensure_hf_token_env(logger: Optional["Logger"] = None) -> str:
    _load_env_file(DEFAULT_BASE_DIR / ".env", override=False)
    _load_env_file(Path.cwd() / ".env", override=False)
    token = _get_hf_token()
    if token and not os.environ.get("HUGGINGFACE_TOKEN", "").strip():
        os.environ["HUGGINGFACE_TOKEN"] = token
        if logger:
            logger("[DIAR] using HF_TOKEN as HUGGINGFACE_TOKEN")
    return token

# ======================================================================
# Video decode
# ======================================================================

def get_video_info(path: str) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return width, height, fps

def extract_frame_cv2(path: str, ts_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts_sec) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()

def extract_frame_av(path: str, ts_sec: float) -> Optional[np.ndarray]:
    if not HAS_AV:
        return None
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target_ts = int(max(0.0, ts_sec) / float(stream.time_base))
        container.seek(target_ts, stream=stream, backward=True)
        for frame in container.decode(video=0):
            frame_t = float(frame.pts * float(stream.time_base))
            if frame_t >= max(0.0, ts_sec) - 0.12:
                img = frame.to_ndarray(format="bgr24")
                container.close()
                return img
        container.close()
    except Exception:
        return None
    return None

def extract_frame(path: str, ts_sec: float) -> Optional[np.ndarray]:
    frame = extract_frame_av(path, ts_sec)
    if frame is not None:
        return frame
    return extract_frame_cv2(path, ts_sec)

def sample_frame_times(start: float, end: float, stride: float) -> List[float]:
    if end <= start:
        return [start]
    times = [start]
    t = start + stride
    while t < end - 1e-6:
        times.append(t)
        t += stride
    if abs(times[-1] - end) > 1e-6:
        times.append(end)
    return times

def sharpness_score(frame: Optional[np.ndarray]) -> float:
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def frame_mean_absdiff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

# ======================================================================
# Diarization
# ======================================================================



def _refine_cut_boundary(
    video_path: str,
    coarse_t: float,
    search_radius: float = 0.3,
    fps: float = 29.97,
    logger: Optional[Logger] = None,
) -> float:
    if coarse_t <= 0.0:
        return 0.0

    fps = max(1.0, float(fps))
    frame_step = 1.0 / fps
    lo = max(0.0, coarse_t - search_radius)
    hi = coarse_t + search_radius

    scan_times: List[float] = []
    t = lo
    while t <= hi + 1e-6:
        scan_times.append(round(t, 6))
        t += frame_step

    if len(scan_times) < 2:
        return round(coarse_t, 4)

    best_t = round(coarse_t, 4)
    best_score = -1.0
    scores_dict: Dict[float, float] = {}

    if not HAS_AV:
        return round(coarse_t, 4)

    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        seek_ts = int(lo / float(stream.time_base))
        container.seek(seek_ts, stream=stream, backward=True)

        prev_frame = None
        prev_t = None
        for frame in container.decode(video=0):
            cur_t = float(frame.pts * float(stream.time_base))
            if cur_t < lo - 0.02:
                continue
            if cur_t > hi + 0.02:
                break

            cur_frame = frame.to_ndarray(format="bgr24")
            if prev_frame is not None and prev_t is not None:
                mad = frame_mean_absdiff(prev_frame, cur_frame)

                prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
                cur_hsv = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2HSV)
                prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [24, 32], [0, 180, 0, 256])
                cur_hist = cv2.calcHist([cur_hsv], [0, 1], None, [24, 32], [0, 180, 0, 256])
                cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(cur_hist, cur_hist, 0, 1, cv2.NORM_MINMAX)
                hist_diff = float(cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_BHATTACHARYYA))

                score = mad * (0.5 + 0.5 * hist_diff)
                scores_dict[cur_t] = score
                if score > best_score:
                    best_score = score
                    best_t = round(cur_t, 4)

            prev_frame = cur_frame
            prev_t = cur_t
    except Exception:
        return round(coarse_t, 4)
    finally:
        if container is not None:
            container.close()

    if logger and best_score >= 0.0:
        neighborhood_msgs = []
        best_idx = -1
        for idx, t_val in enumerate(scan_times):
            if abs(t_val - best_t) < 0.5 * frame_step:
                best_idx = idx
                break
        
        if best_idx != -1:
            for offset in range(-2, 3):
                k = best_idx + offset
                if 1 <= k < len(scan_times):
                    t_val = scan_times[k]
                    score_val = scores_dict.get(t_val, -1.0)
                    marker = " [CUT]" if offset == 0 else ""
                    neighborhood_msgs.append(f"F{offset:+d}({t_val:.3f}s): {score_val:.2f}{marker}")
        
        neighborhood_str = " | ".join(neighborhood_msgs)
        logger(
            f"[CUT] refined {coarse_t:.3f} -> {best_t:.3f} score={best_score:.2f} | Neighbors: {neighborhood_str}"
        )
    return best_t
def load_diarization(path: str) -> List[DiarSegment]:
    data = read_json(path)
    out: List[DiarSegment] = []
    if isinstance(data, list):
        for seg in data:
            try:
                start = float(seg.get("start", seg.get("begin", seg.get("t_start", 0.0))))
                end = float(seg.get("end", seg.get("t_end", 0.0)))
                speaker = str(seg.get("speaker", seg.get("label", "UNKNOWN")))
                if end > start:
                    out.append(DiarSegment(start=start, end=end, speaker=speaker))
            except Exception:
                continue
    elif isinstance(data, dict):
        for spk, intervals in data.items():
            if not isinstance(intervals, list):
                continue
            for iv in intervals:
                if isinstance(iv, (list, tuple)) and len(iv) >= 2:
                    try:
                        start = float(iv[0]); end = float(iv[1])
                        if end > start:
                            out.append(DiarSegment(start=start, end=end, speaker=str(spk)))
                    except Exception:
                        continue
    out.sort(key=lambda s: s.start)
    return out

def active_speaker_at(t: float, timeline: Sequence[DiarSegment]) -> str:
    if not timeline:
        return "UNKNOWN"
    starts = [s.start for s in timeline]
    idx = bisect.bisect_right(starts, t) - 1
    if idx < 0:
        return timeline[0].speaker
    if idx >= len(timeline):
        return timeline[-1].speaker
    seg = timeline[idx]
    if seg.start <= t < seg.end:
        return seg.speaker
    nearest = min(timeline, key=lambda s: min(abs(s.start - t), abs(s.end - t)))
    return nearest.speaker

def speaker_totals(timeline: Sequence[DiarSegment]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in timeline:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (seg.end - seg.start)
    return totals

def _clip_diarization_cache_paths(job_dir: Path, video_path: str, start_sec: float, end_sec: float) -> Tuple[Path, Path, Path, Path]:
    diar_dir = job_dir / DIAR_DIR_NAME
    ensure_dir(diar_dir)
    key_src = f"{Path(video_path).resolve()}|{start_sec:.3f}|{end_sec:.3f}"
    key = hashlib.sha1(key_src.encode("utf-8", errors="ignore")).hexdigest()[:10]
    stem = f"clip_{int(round(start_sec * 1000)):013d}_{int(round(end_sec * 1000)):013d}_{key}"
    return (
        diar_dir / f"{stem}.json",
        diar_dir / f"{stem}.meta.json",
        diar_dir / f"{stem}.raw.json",
        diar_dir / f"{stem}.wav",
    )

def _slice_diarization_to_range(timeline: Sequence[DiarSegment], start_sec: float, end_sec: float) -> List[DiarSegment]:
    out: List[DiarSegment] = []
    for seg in timeline:
        s = max(float(start_sec), float(seg.start))
        e = min(float(end_sec), float(seg.end))
        if e > s:
            out.append(DiarSegment(start=s, end=e, speaker=str(seg.speaker)))
    out.sort(key=lambda x: x.start)
    return out

def _load_or_create_clip_diarization(
    video_path: str,
    start_sec: float,
    end_sec: float,
    job_dir: Path,
    diar_path: Optional[str],
    ffmpeg_bin: str,
    diar_py: str,
    logger: Logger,
) -> List[DiarSegment]:
    """
    Load or create a clip-scoped diarization timeline. Runtime timestamps are
    normalized to source-video time so the rest of the pipeline can query by
    absolute timestamps.
    """
    diar_json, meta_json, raw_json, wav_path = _clip_diarization_cache_paths(job_dir, video_path, start_sec, end_sec)
    if diar_json.exists():
        cached = load_diarization(str(diar_json))
        if cached:
            logger(f"[DIAR] clip cache hit -> {diar_json}")
            return cached

    logger(f"[DIAR] clip cache miss -> {diar_json}")
    token = _ensure_hf_token_env(logger)
    helper_script = DEFAULT_BASE_DIR / "pipeline" / "diarize_helper.py"
    if not helper_script.exists():
        helper_script = Path(__file__).resolve().parent / "pipeline" / "diarize_helper.py"

    if token and Path(diar_py).exists() and helper_script.exists():
        if _extract_audio_mono_range(video_path, str(wav_path), ffmpeg_bin, start_sec, end_sec):
            cmd = [
                diar_py, str(helper_script),
                "--audio", str(wav_path),
                "--token", token,
                "--output", str(raw_json),
            ]
            try:
                res = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                    creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
                )
                if res.returncode == 0:
                    relative = load_diarization(str(raw_json))
                    absolute = [
                        DiarSegment(
                            start=max(start_sec, start_sec + seg.start),
                            end=min(end_sec, start_sec + seg.end),
                            speaker=seg.speaker,
                        )
                        for seg in relative
                        if min(end_sec, start_sec + seg.end) > max(start_sec, start_sec + seg.start)
                    ]
                    if absolute:
                        write_json(diar_json, [asdict(x) for x in absolute])
                        write_json(meta_json, {
                            "source": "clip_diarization",
                            "video_path": str(Path(video_path).resolve()),
                            "clip_start": float(start_sec),
                            "clip_end": float(end_sec),
                            "timestamp_mode": "source_absolute",
                        })
                        logger(f"[DIAR] clip diarization saved -> {diar_json}")
                        return absolute
                else:
                    logger(f"[DIAR] clip diarization failed: {res.stderr[-400:]}")
            except Exception as exc:
                logger(f"[DIAR] clip diarization exception: {exc}")
        else:
            logger("[DIAR] clip audio extraction failed")
    else:
        if not token:
            logger("[DIAR] HF_TOKEN/HUGGINGFACE_TOKEN missing; clip diarization inference skipped")
        elif not Path(diar_py).exists():
            logger(f"[DIAR] diarization python not found: {diar_py}")
        elif not helper_script.exists():
            logger(f"[DIAR] diarization helper not found: {helper_script}")

    # Compatibility escape hatch: if an older caller still passes a full-video
    # JSON, slice it into this clip cache instead of using the global file
    # directly. New automated runs pass an empty diar_path and skip this path.
    legacy_path = str(diar_path or "").strip()
    if legacy_path and Path(legacy_path).exists():
        try:
            sliced = _slice_diarization_to_range(load_diarization(legacy_path), start_sec, end_sec)
            if sliced:
                write_json(diar_json, [asdict(x) for x in sliced])
                write_json(meta_json, {
                    "source": "legacy_json_sliced",
                    "legacy_path": legacy_path,
                    "clip_start": float(start_sec),
                    "clip_end": float(end_sec),
                    "timestamp_mode": "source_absolute",
                })
                logger(f"[DIAR] legacy JSON sliced into clip cache -> {diar_json}")
                return sliced
        except Exception as exc:
            logger(f"[DIAR] legacy diarization slice failed: {exc}")

    return []


# ======================================================================
# Scene detection
# ======================================================================

def _run_ffmpeg_scdet(video_path: str, start_sec: float, end_sec: float, ffmpeg_bin: str, thresh: float) -> List[float]:
    dur = end_sec - start_sec
    if dur < 0.5:
        return []
    cmd = [
        ffmpeg_bin, "-nostdin",
        "-ss", f"{start_sec:.3f}", "-t", f"{dur:.3f}", "-i", video_path,
        "-vf", f"scale=960:-2,scdet=threshold={thresh}:sc_pass=1",
        "-an", "-f", "null", "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=max(60, int(dur * 3)),
            creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
        )
        cuts: List[float] = []
        for line in res.stderr.splitlines():
            m = re.search(r"lavfi\.scd\.time:\s*([\d.]+)", line)
            if m:
                t = start_sec + float(m.group(1))
                if start_sec < t < end_sec:
                    cuts.append(round(t, 4))
        return cuts
    except Exception:
        return []

def _detect_scdet_twopass(video_path: str, start_sec: float, end_sec: float, ffmpeg_bin: str) -> List[float]:
    cuts_a = _run_ffmpeg_scdet(video_path, start_sec, end_sec, ffmpeg_bin, SCDET_PRIMARY_THRESH)
    boundaries = sorted(set([start_sec] + cuts_a + [end_sec]))
    cuts_b: List[float] = []
    for i in range(len(boundaries) - 1):
        gs, ge = boundaries[i], boundaries[i + 1]
        if ge - gs >= SCDET_GAP_MIN:
            cuts_b.extend(_run_ffmpeg_scdet(video_path, gs, ge, ffmpeg_bin, SCDET_SECONDARY_THRESH))
    all_cuts = sorted(set(cuts_a + cuts_b))
    out: List[float] = []
    for c in all_cuts:
        if not out or c - out[-1] > CUT_DEDUP_GAP:
            out.append(c)
    return out

def _detect_pyscenedetect(video_path: str, start_sec: float, end_sec: float, logger: Optional[Logger] = None) -> List[float]:
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector, AdaptiveDetector
    except Exception as e:
        if logger:
            logger(f"[CUT] PySceneDetect import failed: {e}")
        return []
    try:
        if logger:
            logger(f"[CUT] Starting PySceneDetect: range {start_sec:.2f}s - {end_sec:.2f}s, threshold={PYSCENE_THRESHOLD}")
        video = open_video(video_path, framerate=None)
        width, height = video.frame_size
        downscale = max(1, int(round(height / 240.0))) if height > 240 else 1
        sm = SceneManager()
        sm.auto_downscale = False
        sm.downscale = downscale
        sm.add_detector(ContentDetector(threshold=PYSCENE_THRESHOLD * 10))
        sm.add_detector(AdaptiveDetector(adaptive_threshold=PYSCENE_THRESHOLD))
        video.seek(start_sec)
        sm.detect_scenes(video, end_time=end_sec, show_progress=False)
        cuts: List[float] = []
        last = start_sec
        for scene in sm.get_scene_list():
            t = scene[0].get_seconds()
            if start_sec < t < end_sec and (t - last) > CUT_DEDUP_GAP:
                cuts.append(round(t, 4))
                last = t
        if logger:
            logger(f"[CUT] PySceneDetect found {len(cuts)} cuts")
        return cuts
    except Exception as e:
        if logger:
            logger(f"[CUT] PySceneDetect run failed: {e}")
        return []

def _detect_transnetv2(video_path: str, start_sec: float, end_sec: float, ffmpeg_bin: str, logger: Optional[Logger]) -> Optional[List[float]]:
    try:
        import torch
        try:
            from transnetv2_pytorch import TransNetV2
        except Exception:
            try:
                from transnetv2 import TransNetV2
            except Exception:
                return None
    except Exception:
        return None

    duration = end_sec - start_sec
    if duration <= 0:
        return []

    frames_rgb: List[np.ndarray] = []
    frame_times: List[float] = []
    try:
        if not HAS_AV:
            return None
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        seek_ts = int(start_sec / float(stream.time_base))
        container.seek(seek_ts, stream=stream, backward=True)
        for frame in container.decode(video=0):
            t = float(frame.pts * float(stream.time_base))
            if t < start_sec - 0.05:
                continue
            if t > end_sec + 0.05:
                break
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, (48, 27), interpolation=cv2.INTER_LINEAR)
            frames_rgb.append(img)
            frame_times.append(t)
        container.close()
    except Exception as e:
        if logger:
            logger(f"[CUT] TransNetV2 decode failed: {e}")
        return None

    if len(frames_rgb) < TRANSNET_MIN_FRAMES:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = TransNetV2().to(device)
        model.eval()
    except Exception as e:
        if logger:
            logger(f"[CUT] TransNetV2 init failed: {e}")
        return None

    try:
        with torch.no_grad():
            inp = torch.from_numpy(np.asarray(frames_rgb, dtype=np.uint8)).unsqueeze(0).to(device)
            result = model(inp)
            probs = result[0] if isinstance(result, (tuple, list)) else result
            probs = probs.squeeze(0).detach().cpu().numpy()
    except Exception as e:
        if logger:
            logger(f"[CUT] TransNetV2 inference failed: {e}")
        return None
    finally:
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    cuts: List[float] = []
    last = start_sec
    for i, p in enumerate(probs):
        if float(p) >= TRANSNET_THRESHOLD and i < len(frame_times):
            t = frame_times[i]
            if start_sec < t < end_sec and (t - last) > CUT_DEDUP_GAP:
                cuts.append(round(t, 4))
                last = t
    if logger:
        logger(f"[CUT] TransNetV2 found {len(cuts)} cuts")
    return cuts

def _pyscene_subprocess_worker(video_path_w, start_w, end_w, threshold_w, dedup_gap_w, result_queue):
    """Subprocess target for PySceneDetect (CPU-only, GIL-free)."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector, AdaptiveDetector
        video = open_video(video_path_w, framerate=None)
        width, height = video.frame_size
        downscale = max(1, int(round(height / 240.0))) if height > 240 else 1
        sm = SceneManager()
        sm.auto_downscale = False
        sm.downscale = downscale
        sm.add_detector(ContentDetector(threshold=threshold_w * 10))
        sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold_w))
        video.seek(start_w)
        sm.detect_scenes(video, end_time=end_w, show_progress=False)
        cuts = []
        last = start_w
        for scene in sm.get_scene_list():
            t = scene[0].get_seconds()
            if start_w < t < end_w and (t - last) > dedup_gap_w:
                cuts.append(round(t, 4))
                last = t
        result_queue.put(cuts)
    except Exception:
        result_queue.put([])

def detect_scenes(
    video_path: str,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: str,
    logger: Optional[Logger] = None,
    fps: float = 29.97,
) -> List[float]:
    t1: List[float] = []
    t2: List[float] = []
    t3: List[float] = []

    # TransNetV2 (GPU) runs in main process; PySceneDetect (CPU) in subprocess
    # to bypass the GIL and achieve true parallelism.
    import multiprocessing as _mp_mod
    _mp_q = _mp_mod.Queue()
    _mp_proc = _mp_mod.Process(
        target=_pyscene_subprocess_worker,
        args=(video_path, start_sec, end_sec, PYSCENE_THRESHOLD, CUT_DEDUP_GAP, _mp_q),
        daemon=True,
    )
    _mp_proc.start()

    # TransNetV2 runs in main process (CUDA context)
    t1_result = _detect_transnetv2(video_path, start_sec, end_sec, ffmpeg_bin, logger)
    t1 = t1_result if t1_result is not None else []

    # Collect PySceneDetect from subprocess
    try:
        t2 = _mp_q.get(timeout=600)
        if not isinstance(t2, list):
            t2 = []
    except Exception:
        t2 = []
    finally:
        _mp_proc.join()
    if logger:
        logger(f"[CUT] PySceneDetect (subprocess) found {len(t2)} cuts")

    if t1_result is None and len(t2) == 0:
        t3 = _detect_scdet_twopass(video_path, start_sec, end_sec, ffmpeg_bin)

    merged = sorted(set(t1 + t2 + t3))
    coarse: List[float] = []
    for c in merged:
        if not coarse or c - coarse[-1] > CUT_DEDUP_GAP:
            coarse.append(c)

    refined: List[float] = [float(start_sec)]
    for c in coarse:
        # Narrowed search radius: Â±0.3 s instead of Â±1.0 s.
        # PySceneDetect/TransNetV2 cuts are already accurate to <0.1 s,
        # so the wide window was burning ~3Ã— more decode time for no benefit.
        refined_t = _refine_cut_boundary(video_path, c, search_radius=0.3, fps=fps, logger=logger)
        if refined_t - refined[-1] > CUT_DEDUP_GAP:
            refined.append(refined_t)
        elif logger:
            logger(
                f"[CUT] merged refined cut {refined_t:.3f} into previous boundary "
                f"(gap={refined_t - refined[-1]:.3f}s)"
            )

    if refined[-1] != float(end_sec):
        refined.append(float(end_sec))
    return refined

def _phash(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    if HAS_IMAGEHASH:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(pil)
        arr = h.hash.astype(np.uint8).flatten()
        return arr, "imagehash"
    small = cv2.resize(frame, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)
    chunk = dct[:8, :8].flatten()
    bits = (chunk > chunk.mean()).astype(np.uint8)
    return bits, "dct"

def _phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a).astype(np.uint8).ravel()
    b = np.asarray(b).astype(np.uint8).ravel()
    if a.size != b.size:
        return 64
    return int(np.sum(a != b))

def _medoid_hash(frames: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    valid = [f for f in frames if f is not None]
    if not valid:
        return None, None
    if len(valid) == 1:
        h, _ = _phash(valid[0])
        return h, valid[0]
    hashes = [_phash(f)[0] for f in valid]
    best_idx = 0
    best_total = float("inf")
    for i in range(len(valid)):
        total = 0
        for j in range(len(valid)):
            if i == j:
                continue
            total += _phash_distance(hashes[i], hashes[j])
        if total < best_total:
            best_total = total
            best_idx = i
    return hashes[best_idx], valid[best_idx]

# ======================================================================
# Face backends
# ======================================================================

class BaseFaceBackend:
    name = "base"
    supports_embedding = False

    def detect(self, bgr: np.ndarray) -> List[FaceObs]:
        raise NotImplementedError

    def close(self) -> None:
        return

def _calc_yaw_from_landmarks(landmarks: Optional[np.ndarray], bbox: Tuple[float, float, float, float]) -> float:
    if landmarks is None:
        return 0.0
    try:
        pts = np.asarray(landmarks, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts.squeeze(0)
        if pts.shape[0] < 2:
            return 0.0
        # crude yaw proxy: left/right eye x offset from face center
        lx = float(np.min(pts[:, 0]))
        rx = float(np.max(pts[:, 0]))
        cx = (bbox[0] + bbox[2]) / 2.0
        width = max(1e-6, bbox[2] - bbox[0])
        center_proxy = (lx + rx) / 2.0
        yaw = ((center_proxy - cx) / (width / 2.0)) * 30.0
        return float(clamp(yaw, -45.0, 45.0))
    except Exception:
        return 0.0

def _torso_roi_from_bbox(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    fw = max(1.0, x2 - x1)
    fh = max(1.0, y2 - y1)
    rx1 = int(clamp(x1 - 0.20 * fw, 0, w - 1))
    rx2 = int(clamp(x2 + 0.20 * fw, 0, w - 1))
    ry1 = int(clamp(y2 + 0.20 * fh, 0, h - 1))
    ry2 = int(clamp(y2 + 1.50 * fh, 0, h - 1))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return frame[ry1:ry2, rx1:rx2]

def _shirt_signature(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int]]]:
    roi = _torso_roi_from_bbox(frame, bbox)
    if roi is None or roi.size == 0:
        return None, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    dom = np.mean(roi.reshape(-1, 3), axis=0)
    rgb = tuple(int(x) for x in dom[::-1])
    return hist.flatten().astype(np.float32), rgb

class InsightFaceBackend(BaseFaceBackend):
    name = "insightface"
    supports_embedding = True

    def __init__(self, use_cuda: bool = True, model_name: str = "buffalo_l", det_size: Tuple[int, int] = (640, 640), logger: Optional[Logger] = None):
        self.logger = logger
        self.app = None
        try:
            from insightface.app import FaceAnalysis
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            if use_cuda:
                providers = [(
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": 1024 * 1024 * 1024,
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "DEFAULT",
                        "do_copy_in_default_stream": True,
                    },
                ), "CPUExecutionProvider"]
                ctx_id = 0
            allowed = ["detection", "recognition", "landmark_2d_106"]
            self.app = FaceAnalysis(name=model_name, providers=providers, allowed_modules=allowed)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=FACE_MIN_SCORE)
            if self.logger:
                self.logger(f"[FACE] InsightFace {model_name} ready ({'CUDA' if use_cuda else 'CPU'})")
        except Exception as e:
            if self.logger:
                self.logger(f"[FACE] InsightFace init failed: {e}")
            self.app = None

    def detect(self, bgr: np.ndarray) -> List[FaceObs]:
        if self.app is None or bgr is None:
            return []
        try:
            faces = self.app.get(bgr)
            out: List[FaceObs] = []
            for f in faces:
                bbox = np.asarray(f.bbox, dtype=np.float32).ravel()
                if bbox.size < 4:
                    continue
                x1, y1, x2, y2 = map(float, bbox[:4])
                conf = float(getattr(f, "det_score", 0.0))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                area = max(0.0, (x2 - x1) * (y2 - y1))
                embedding = None
                if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                    embedding = np.asarray(f.normed_embedding, dtype=np.float32)
                landmarks = None
                if hasattr(f, "landmark_2d_106") and f.landmark_2d_106 is not None:
                    landmarks = np.asarray(f.landmark_2d_106, dtype=np.float32)
                yaw = 0.0
                try:
                    if hasattr(f, "pose") and f.pose is not None:
                        yaw = float(f.pose[1])
                except Exception:
                    yaw = 0.0
                if yaw == 0.0:
                    yaw = _calc_yaw_from_landmarks(landmarks, (x1, y1, x2, y2))
                shirt_hist, shirt_rgb = _shirt_signature(bgr, (x1, y1, x2, y2))
                out.append(FaceObs(x1, y1, x2, y2, conf, cx, cy, area, yaw=yaw, embedding=embedding, landmarks=landmarks, shirt_hist=shirt_hist, shirt_rgb=shirt_rgb, torso_conf=float(0.6 if shirt_hist is not None else 0.0)))
            return sorted(out, key=lambda o: o.area, reverse=True)
        except Exception:
            return []

    def close(self) -> None:
        try:
            self.app = None
        except Exception:
            pass


class RTDetrFaceBackend(BaseFaceBackend):
    """RT-DETR-L (Ultralytics) person detector, used as a face-detection proxy.

    Why a PERSON detector for face tracking? RT-DETR is pose-invariant —
    it finds people at any head angle (front, profile, back-of-head),
    eliminating the side-profile blind spot of face detectors like SCRFD.
    For the pipeline's needs (centering a 9:16 crop on the active speaker)
    the head region is derived as the top ~30% of each person box. This is
    accurate enough for face-centered cropping while gaining:
      • Pose invariance (no side-face / back-of-head misses)
      • Fewer dropped frames in the per-tick tracking loop
      • Compatibility with manual template slots: in wide shots, the active
        slot's bbox_norm is the IDENTITY ground truth — we pick the
        RT-DETR detection whose head center is nearest to the slot center.
        Diarization still drives which slot is active. No embeddings needed.

    Weights: `pipeline/models/weights/rtdetr-l.pt` (Ultralytics format).
    """
    name = "rtdetr"
    supports_embedding = False

    # Top fraction of the person box treated as the head region. Empirically,
    # a seated podcast frame has shoulders ~25–30% from the top of a tight
    # person box; 0.30 reliably contains the face even on partial torsos.
    HEAD_FRACTION = 0.30

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.30,
        imgsz: int = 640,
        min_area_frac: float = 0.0,
        pose_refiner: Optional["PoseHeadRefiner"] = None,
        logger: Optional["Logger"] = None,
    ):
        self.logger = logger
        self.weights_path = str(weights_path)
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.min_area_frac = float(min_area_frac)
        self.pose_refiner = pose_refiner
        self.model = None
        try:
            from ultralytics import RTDETR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install it into the project venv: "
                "venv/Scripts/pip install ultralytics"
            ) from exc
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"RT-DETR weights not found at {self.weights_path}. "
                "Drop the rtdetr-l.pt file into pipeline/models/weights/."
            )
        if logger:
            logger(f"[RTDETR] Loading weights: {self.weights_path} (device={self.device})")
        self.model = RTDETR(self.weights_path)
        try:
            self.model.to(self.device)
        except Exception:
            # Ultralytics handles device internally on first predict if .to() isn't supported.
            pass
        if logger:
            logger(
                f"[RTDETR] Backend ready. supports_embedding=False — wide-shot identity "
                f"comes from template slot proximity + diarization. "
                f"conf>={self.conf_threshold:.2f} min_area_frac={self.min_area_frac:.3f} "
                f"pose_head={'on' if self.pose_refiner is not None else 'off'}"
            )

    def _result_to_faces(self, res0, frame_w: int, frame_h: int) -> List[FaceObs]:
        """Convert one Ultralytics result into person-head FaceObs.

        Applies the person-class filter and the min-area gate (which rejects
        tiny background persons such as a face on a tablet/phone screen). The
        head center starts as the top-of-box estimate; the pose refiner (if any)
        overrides it with the true keypoint-derived head center later.
        """
        obs: List[FaceObs] = []
        boxes = getattr(res0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return obs
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
        except Exception as exc:
            if self.logger:
                self.logger(f"[RTDETR] could not read boxes: {exc}")
            return obs

        frame_area = max(1.0, float(frame_w) * float(frame_h))
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
            if int(cls_id) != 0:  # only persons
                continue
            person_w = max(1.0, float(x2) - float(x1))
            person_h = max(1.0, float(y2) - float(y1))
            # Min-area gate: drop tiny background persons (tablet/phone screens,
            # people far behind the camera) even if they clear the conf threshold.
            if self.min_area_frac > 0.0 and (person_w * person_h) / frame_area < self.min_area_frac:
                continue
            # Head region = top fraction of the person box (fallback center).
            hx1 = float(x1)
            hx2 = float(x2)
            hy1 = float(y1)
            hy2 = float(y1) + person_h * self.HEAD_FRACTION
            head_w = max(1.0, hx2 - hx1)
            head_h = max(1.0, hy2 - hy1)
            cx = (hx1 + hx2) / 2.0
            cy = (hy1 + hy2) / 2.0
            area = head_w * head_h
            obs.append(FaceObs(
                x1=hx1, y1=hy1, x2=hx2, y2=hy2,
                conf=float(conf),
                cx=float(cx), cy=float(cy),
                area=float(area),
                yaw=0.0,
                embedding=None,
                landmarks=None,
            ))
        return obs

    def detect(self, bgr: np.ndarray) -> List[FaceObs]:
        if self.model is None or bgr is None or bgr.size == 0:
            return []
        try:
            # Ultralytics accepts BGR ndarray directly. verbose=False silences per-call prints.
            results = self.model.predict(
                bgr,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            if self.logger:
                self.logger(f"[RTDETR] predict failed: {exc}")
            return []
        if not results:
            return []
        h, w = bgr.shape[:2]
        obs = self._result_to_faces(results[0], w, h)
        if obs and self.pose_refiner is not None:
            try:
                _apply_pose_head_centers(obs, self.pose_refiner.head_centers(bgr), float(w), float(h))
            except Exception as exc:
                if self.logger:
                    self.logger(f"[POSE] refine failed: {exc}")
        return obs

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[FaceObs]]:
        """Batched detection over many frames at once (GPU-saturating pass).

        Returns one FaceObs list per input frame, in the same order. None/empty
        frames yield empty lists. On any batch error, falls back to per-frame
        detection so a single failure never drops a whole segment.
        """
        out: List[List[FaceObs]] = [[] for _ in frames]
        if self.model is None or not frames:
            return out
        idx_valid = [i for i, f in enumerate(frames) if f is not None and getattr(f, "size", 0) > 0]
        if not idx_valid:
            return out
        valid_frames = [frames[i] for i in idx_valid]
        try:
            results = self.model.predict(
                valid_frames,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            if self.logger:
                self.logger(f"[RTDETR] batch predict failed ({exc}); per-frame fallback")
            for i in idx_valid:
                out[i] = self.detect(frames[i])
            return out

        pose_cands: List[List[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]]] = [
            [] for _ in valid_frames
        ]
        if self.pose_refiner is not None:
            try:
                pose_cands = self.pose_refiner.head_centers_batch(valid_frames)
            except Exception as exc:
                if self.logger:
                    self.logger(f"[POSE] batch refine failed: {exc}")
                pose_cands = [[] for _ in valid_frames]

        for k, res0 in enumerate(results):
            fr = valid_frames[k]
            h, w = fr.shape[:2]
            obs = self._result_to_faces(res0, w, h)
            if obs and k < len(pose_cands) and pose_cands[k]:
                _apply_pose_head_centers(obs, pose_cands[k], float(w), float(h))
            out[idx_valid[k]] = obs
        return out

    def close(self) -> None:
        try:
            self.model = None
            if self.pose_refiner is not None:
                self.pose_refiner.close()
            self.pose_refiner = None
        except Exception:
            pass


class MediaPipeFaceBackend(BaseFaceBackend):
    name = "mediapipe"
    supports_embedding = False

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.detector = None
        self.mesh = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_MIN_SCORE)
            self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=8, refine_landmarks=True, min_detection_confidence=FACE_MIN_SCORE)
            if self.logger:
                self.logger("[FACE] MediaPipe face detector ready")
        except Exception as e:
            if self.logger:
                self.logger(f"[FACE] MediaPipe init failed: {e}")
            self.detector = None
            self.mesh = None

    def detect(self, bgr: np.ndarray) -> List[FaceObs]:
        if self.detector is None or bgr is None:
            return []
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            det = self.detector.process(rgb)
            out: List[FaceObs] = []
            if det.detections:
                h, w = bgr.shape[:2]
                for d in det.detections:
                    bbox = d.location_data.relative_bounding_box
                    x1 = float(clamp(bbox.xmin * w, 0, w - 1))
                    y1 = float(clamp(bbox.ymin * h, 0, h - 1))
                    x2 = float(clamp((bbox.xmin + bbox.width) * w, 0, w - 1))
                    y2 = float(clamp((bbox.ymin + bbox.height) * h, 0, h - 1))
                    conf = float(d.score[0] if d.score else 0.0)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    area = max(0.0, (x2 - x1) * (y2 - y1))
                    shirt_hist, shirt_rgb = _shirt_signature(bgr, (x1, y1, x2, y2))
                    out.append(FaceObs(x1, y1, x2, y2, conf, cx, cy, area, shirt_hist=shirt_hist, shirt_rgb=shirt_rgb, torso_conf=float(0.6 if shirt_hist is not None else 0.0)))
            # Landmarks only if mesh available; can refine yawn/center.
            return sorted(out, key=lambda o: o.area, reverse=True)
        except Exception:
            return []

    def close(self) -> None:
        self.detector = None
        self.mesh = None

class HaarFaceBackend(BaseFaceBackend):
    name = "haar"
    supports_embedding = False

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if self.logger:
            self.logger("[FACE] Haar fallback ready")

    def detect(self, bgr: np.ndarray) -> List[FaceObs]:
        if bgr is None:
            return []
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            out: List[FaceObs] = []
            for (x, y, w, h) in faces:
                x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                area = float(w * h)
                shirt_hist, shirt_rgb = _shirt_signature(bgr, (x1, y1, x2, y2))
                out.append(FaceObs(x1, y1, x2, y2, 0.5, cx, cy, area, shirt_hist=shirt_hist, shirt_rgb=shirt_rgb, torso_conf=float(0.4 if shirt_hist is not None else 0.0)))
            return sorted(out, key=lambda o: o.area, reverse=True)
        except Exception:
            return []

def _apply_pose_head_centers(
    faces: List[FaceObs],
    candidates: Sequence[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]],
    frame_w: float,
    frame_h: float,
) -> None:
    """Override each person-box head center with the nearest pose-derived head
    keypoint center that falls within the person's horizontal span.

    Each candidate is (x, y, score, head_box) in the same pixel space as
    `faces`. The top-of-box fallback center is kept whenever no keypoints match
    (e.g. the speaker's back is to the camera). Greedy
    nearest-by-horizontal-distance assignment prevents two boxes from claiming
    the same head.

    When the candidate carries a keypoint-derived head_box, it replaces the
    coarse person-box top-fraction bbox: area ranking, torso/shirt sampling
    (anchored below y2) and the debug overlay then see real head geometry
    instead of a shoulder-wide slab anchored at the person-box top.
    """
    if not candidates or not faces:
        return
    used = [False] * len(candidates)
    for f in faces:
        fcx = (f.x1 + f.x2) / 2.0
        h_head = max(1.0, f.y2 - f.y1)
        best_j = -1
        best_d = 1e18
        for j, (kx, ky, _kconf, _kbox) in enumerate(candidates):
            if used[j]:
                continue
            if kx < f.x1 - 2.0 or kx > f.x2 + 2.0:
                continue
            if ky < f.y1 - h_head or ky > f.y2 + h_head:
                continue
            d = abs(kx - fcx)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            kx, ky, _kconf, kbox = candidates[best_j]
            used[best_j] = True
            f.cx = float(kx)
            f.cy = float(clamp(ky, f.y1, f.y2))
            if kbox is not None:
                bx1 = clamp(float(kbox[0]), 0.0, max(0.0, frame_w - 2.0))
                by1 = clamp(float(kbox[1]), 0.0, max(0.0, frame_h - 2.0))
                bx2 = clamp(float(kbox[2]), bx1 + 2.0, float(frame_w))
                by2 = clamp(float(kbox[3]), by1 + 2.0, float(frame_h))
                f.x1, f.y1, f.x2, f.y2 = bx1, by1, bx2, by2
                f.area = (bx2 - bx1) * (by2 - by1)


class PoseHeadRefiner:
    """yolo11n-pose head-center estimator.

    RT-DETR gives only a person box; its horizontal center drifts as the speaker
    gestures. This refiner returns a pose-invariant head center from the COCO
    head keypoints (nose, eyes, ears), so the 9:16 crop locks onto the true head.
    Pure refinement — it never adds or removes subjects; RT-DETR remains the
    detector and identity ground truth.
    """

    # COCO keypoint indices that define the head.
    _HEAD_KPTS = (0, 1, 2, 3, 4)  # nose, left_eye, right_eye, left_ear, right_ear
    _KPT_CONF = 0.30

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        logger: Optional["Logger"] = None,
    ):
        self.logger = logger
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.model = None
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed (needed for yolo11n-pose head refiner)."
            ) from exc
        if not Path(str(weights_path)).exists():
            raise FileNotFoundError(
                f"Pose weights not found at {weights_path}. "
                "Drop yolo11n-pose.pt into pipeline/models/weights/."
            )
        self.model = YOLO(str(weights_path))
        try:
            self.model.to(self.device)
        except Exception:
            pass
        if logger:
            logger(f"[POSE] yolo11n-pose head refiner ready (device={self.device})")

    def _result_to_centers(
        self, res0
    ) -> List[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]]:
        """Per detected person: (cx, cy, weight, head_box or None).

        head_box is a keypoint-derived (x1, y1, x2, y2) estimate of the actual
        head extent — ear-to-ear span scaled up for hair/margin. None when the
        keypoint spread is too degenerate to size a box (single keypoint)."""
        out: List[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]] = []
        kpts = getattr(res0, "keypoints", None)
        if kpts is None:
            return out
        try:
            xy = kpts.xy.cpu().numpy()  # (N, K, 2) in pixels of the input frame
            conf_t = getattr(kpts, "conf", None)
            kconf = conf_t.cpu().numpy() if conf_t is not None else None  # (N, K)
        except Exception:
            return out
        if xy.ndim != 3 or xy.shape[0] == 0:
            return out
        n_kpts = xy.shape[1]
        for i in range(xy.shape[0]):
            sx = sy = sw = 0.0
            pts: List[Tuple[int, float, float]] = []  # (kpt_idx, px, py)
            for k in self._HEAD_KPTS:
                if k >= n_kpts:
                    continue
                kc = float(kconf[i, k]) if kconf is not None else 1.0
                if kc < self._KPT_CONF:
                    continue
                px, py = float(xy[i, k, 0]), float(xy[i, k, 1])
                if px <= 0.0 and py <= 0.0:
                    continue
                pts.append((k, px, py))
                sx += px * kc
                sy += py * kc
                sw += kc
            if sw <= 0.0:
                continue
            cx, cy = sx / sw, sy / sw
            box: Optional[Tuple[float, float, float, float]] = None
            if len(pts) >= 2:
                xs = [p[1] for p in pts]
                span = max(xs) - min(xs)
                if span >= 4.0:
                    # Ear-to-ear span covers ~3/4 of the full head width (hair
                    # included); eye/nose-only spans (profile / occlusion) are
                    # much narrower, so scale them up more aggressively.
                    have_both_ears = {3, 4}.issubset({p[0] for p in pts})
                    factor = 1.35 if have_both_ears else 2.2
                    head_w = max(24.0, span * factor)
                    head_h = head_w * 1.35
                    box = (
                        cx - head_w / 2.0, cy - head_h * 0.50,
                        cx + head_w / 2.0, cy + head_h * 0.50,
                    )
            out.append((cx, cy, sw, box))
        return out

    def head_centers(
        self, bgr: np.ndarray
    ) -> List[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]]:
        if self.model is None or bgr is None or getattr(bgr, "size", 0) == 0:
            return []
        try:
            results = self.model.predict(
                bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
            )
        except Exception as exc:
            if self.logger:
                self.logger(f"[POSE] predict failed: {exc}")
            return []
        if not results:
            return []
        return self._result_to_centers(results[0])

    def head_centers_batch(
        self, frames: Sequence[np.ndarray]
    ) -> List[List[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]]]:
        out: List[List[Tuple[float, float, float, Optional[Tuple[float, float, float, float]]]]] = [
            [] for _ in frames
        ]
        if self.model is None or not frames:
            return out
        try:
            results = self.model.predict(
                list(frames), imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
            )
        except Exception as exc:
            if self.logger:
                self.logger(f"[POSE] batch predict failed: {exc}")
            return out
        for k, res0 in enumerate(results):
            if k < len(out):
                out[k] = self._result_to_centers(res0)
        return out

    def close(self) -> None:
        try:
            self.model = None
        except Exception:
            pass


def create_face_backend(logger: Optional[Logger] = None) -> BaseFaceBackend:
    """Pick a detector based on config.FACE_BACKEND (default 'rtdetr').

    Order honoured:
      • 'rtdetr'      → RT-DETR-L person detector (pose-invariant; default).
      • 'insightface' → InsightFace buffalo_l (face + embedding).
      • auto fallbacks: MediaPipe → Haar.
    """
    preferred = str(_pipeline_config_value("FACE_BACKEND", "rtdetr") or "rtdetr").lower().strip()

    # ── RT-DETR-L ──────────────────────────────────────────────────────────
    if preferred == "rtdetr":
        try:
            weights = _pipeline_config_value(
                "RTDETR_WEIGHTS_PATH",
                str(Path(__file__).with_name("models") / "weights" / "rtdetr-l.pt"),
            )
            device = "cuda" if _pipeline_config_value("DEVICE", "cuda") == "cuda" else "cpu"
            conf = float(_pipeline_config_value("RTDETR_CONF_THRESHOLD", 0.70))
            imgsz = int(_pipeline_config_value("RTDETR_IMGSZ", 640))
            min_area = float(_pipeline_config_value("RTDETR_MIN_AREA_FRAC", 0.0))
            pose_refiner = None
            if bool(_pipeline_config_value("HEAD_POSE_ENABLED", True)):
                try:
                    pose_weights = _pipeline_config_value(
                        "HEAD_POSE_WEIGHTS_PATH",
                        str(Path(__file__).with_name("models") / "weights" / "yolo11n-pose.pt"),
                    )
                    pose_refiner = PoseHeadRefiner(
                        weights_path=pose_weights,
                        device=device,
                        imgsz=imgsz,
                        logger=logger,
                    )
                except Exception as pexc:
                    if logger:
                        logger(f"[POSE] disabled (init failed: {pexc}); using top-of-box head center")
                    pose_refiner = None
            return RTDetrFaceBackend(
                weights_path=weights,
                device=device,
                conf_threshold=conf,
                imgsz=imgsz,
                min_area_frac=min_area,
                pose_refiner=pose_refiner,
                logger=logger,
            )
        except Exception as exc:
            if logger:
                logger(f"[FACE] RT-DETR init failed ({exc}); falling back to InsightFace")

    # ── InsightFace ────────────────────────────────────────────────────────
    if preferred in ("rtdetr", "insightface"):
        try:
            model_name = _pipeline_config_value("INSIGHTFACE_MODEL", "buffalo_l")
            det_size = _pipeline_config_value("TRACKING_GPU_DET_SIZE", (640, 640))
            return InsightFaceBackend(use_cuda=True, model_name=model_name, det_size=det_size, logger=logger)
        except Exception:
            pass

    try:
        return MediaPipeFaceBackend(logger=logger)
    except Exception:
        pass
    return HaarFaceBackend(logger=logger)


# ======================================================================
# Template discovery
# ======================================================================

def _canonical_crop_dims(src_w: int, src_h: int) -> Tuple[int, int]:
    crop_w = min(int(src_h * 9 / 16), src_w)
    crop_h = src_h
    return crop_w, crop_h

def _compute_crop_from_face(face: FaceObs, src_w: int, src_h: int) -> Tuple[float, float]:
    crop_w, crop_h = _canonical_crop_dims(src_w, src_h)
    face_cx = face.cx
    face_cy = face.cy
    yaw = float(face.yaw or 0.0)
    nose_raw = -(yaw / 45.0) * (crop_w * NOSE_ROOM_FACTOR)
    nose_off = clamp(nose_raw, -crop_w * MAX_NOSE_FRAC, crop_w * MAX_NOSE_FRAC)
    crop_x = clamp(face_cx - crop_w / 2.0 + nose_off, 0.0, max(0.0, src_w - crop_w))
    crop_y = clamp(face_cy - crop_h * HEAD_ROOM_FRAC, 0.0, max(0.0, src_h - crop_h))
    return crop_x, crop_y

def _pick_sharpest(frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    return max(valid, key=sharpness_score)

def _slot_name_for_index(idx: int, n: int) -> str:
    if n <= 1:
        return "single"
    if n == 2:
        return "left" if idx == 0 else "right"
    if n == 3:
        return ["left", "center", "right"][idx]
    if n == 4:
        return ["left", "left_center", "right_center", "right"][idx]
    names = ["left", "left_center", "center", "right_center", "right"]
    if idx < len(names):
        return names[idx]
    return f"slot_{idx}"

def _cluster_faces_by_sorted_index(observations_by_frame: List[List[FaceObs]], max_slots: int = 4) -> List[List[FaceObs]]:
    slots: List[List[FaceObs]] = [[] for _ in range(max_slots)]
    counts: List[int] = []
    for obs_list in observations_by_frame:
        obs_list = sorted(obs_list, key=lambda o: o.cx)
        if not obs_list:
            continue
        counts.append(len(obs_list))
        for idx, obs in enumerate(obs_list[:max_slots]):
            slots[idx].append(obs)
    # trim empty trailing slots
    while slots and not slots[-1]:
        slots.pop()
    return slots

def _aggregate_slot(slot_obs: List[FaceObs], frame_w: int, frame_h: int, slot_name: str, slot_order: int) -> Optional[FaceSlot]:
    valid = [o for o in slot_obs if o is not None]
    if not valid:
        return None
    cx = float(np.median([o.cx for o in valid]))
    cy = float(np.median([o.cy for o in valid]))
    x1 = float(np.median([o.x1 for o in valid]))
    y1 = float(np.median([o.y1 for o in valid]))
    x2 = float(np.median([o.x2 for o in valid]))
    y2 = float(np.median([o.y2 for o in valid]))
    conf = float(np.mean([o.conf for o in valid]))
    embeddings = [o.embedding for o in valid if o.embedding is not None]
    emb = None
    if embeddings:
        emb = normalize_vec(np.mean(np.stack(embeddings, axis=0), axis=0))
    shirt_hists = [o.shirt_hist for o in valid if o.shirt_hist is not None]
    shirt_hist = None
    if shirt_hists:
        shirt_hist = np.mean(np.stack(shirt_hists, axis=0), axis=0)
        shirt_hist = shirt_hist / (shirt_hist.sum() + 1e-9)
    shirt_rgbs = [o.shirt_rgb for o in valid if o.shirt_rgb is not None]
    shirt_rgb = None
    if shirt_rgbs:
        shirt_rgb = tuple(int(round(sum(v[i] for v in shirt_rgbs) / len(shirt_rgbs))) for i in range(3))
    return FaceSlot(
        slot_name=slot_name,
        center=(cx, cy),
        bbox=(x1, y1, x2, y2),
        center_norm=(cx / max(1, frame_w), cy / max(1, frame_h)),
        bbox_norm=(x1 / max(1, frame_w), y1 / max(1, frame_h), x2 / max(1, frame_w), y2 / max(1, frame_h)),
        confidence=conf,
        embedding=emb.tolist() if emb is not None else None,
        shirt_hist=shirt_hist.tolist() if shirt_hist is not None else None,
        shirt_rgb=shirt_rgb,
        speaker_label=None,
        slot_order=slot_order,
    )


def _build_template_geometry(frame: np.ndarray, frame_obs_list: List[List[FaceObs]]) -> Tuple[List[FaceSlot], Dict[str, float], Dict[str, List[float]], Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    h, w = frame.shape[:2]
    slots = _cluster_faces_by_sorted_index(frame_obs_list, max_slots=4)
    face_slots: List[FaceSlot] = []
    speaker_embeddings: Dict[str, List[float]] = {}
    shirt_signatures: Dict[str, List[float]] = {}
    slot_geometry: Dict[str, Dict[str, float]] = {}

    for idx, slot_obs in enumerate(slots):
        slot_name = _slot_name_for_index(idx, len(slots))
        agg = _aggregate_slot(slot_obs, w, h, slot_name, idx)
        if agg is None:
            continue
        face_slots.append(agg)
        slot_geometry[agg.slot_name] = {
            "center_x": float(agg.center[0]),
            "center_y": float(agg.center[1]),
            "x1": float(agg.bbox[0]),
            "y1": float(agg.bbox[1]),
            "x2": float(agg.bbox[2]),
            "y2": float(agg.bbox[3]),
            "order": float(agg.slot_order),
        }

    if face_slots:
        dom = max(face_slots, key=lambda s: s.confidence * (s.bbox_norm[2] - s.bbox_norm[0]) * (s.bbox_norm[3] - s.bbox_norm[1]))
        crop_x, crop_y = _compute_crop_from_face(
            FaceObs(
                x1=dom.bbox[0], y1=dom.bbox[1], x2=dom.bbox[2], y2=dom.bbox[3],
                conf=dom.confidence, cx=dom.center[0], cy=dom.center[1],
                area=(dom.bbox[2]-dom.bbox[0])*(dom.bbox[3]-dom.bbox[1]),
                yaw=0.0,
            ),
            w, h
        )
    else:
        crop_x = max(0.0, (w - _canonical_crop_dims(w, h)[0]) / 2.0)
        crop_y = 0.0

    crop_anchor = {
        "x": float(crop_x),
        "y": float(crop_y),
        "w": float(_canonical_crop_dims(w, h)[0]),
        "h": float(_canonical_crop_dims(w, h)[1]),
    }

    for s in face_slots:
        if s.embedding is not None:
            speaker_embeddings[s.slot_name] = s.embedding
        if s.shirt_hist is not None:
            shirt_signatures[s.slot_name] = s.shirt_hist

    return face_slots, crop_anchor, speaker_embeddings, shirt_signatures, slot_geometry
def _serialize_template(t: TemplateRecord) -> Dict[str, Any]:
    d = asdict(t)
    # normalize paths/arrays
    for slot in d.get("face_slots", []):
        if isinstance(slot.get("center"), tuple):
            slot["center"] = list(slot["center"])
        if isinstance(slot.get("bbox"), tuple):
            slot["bbox"] = list(slot["bbox"])
        if isinstance(slot.get("center_norm"), tuple):
            slot["center_norm"] = list(slot["center_norm"])
        if isinstance(slot.get("bbox_norm"), tuple):
            slot["bbox_norm"] = list(slot["bbox_norm"])
        if isinstance(slot.get("shirt_rgb"), tuple):
            slot["shirt_rgb"] = list(slot["shirt_rgb"])
    return d



def _template_image_paths(job_dir: Path, template_id: int, display_name: str = "") -> Tuple[Path, Path]:
    base = job_dir / TEMPLATE_DIR_NAME
    stem = display_name if display_name else f"template_{template_id:02d}"
    raw_path = base / f"{stem}.jpg"
    overlay_path = base / f"{stem}_overlay.jpg"
    return raw_path, overlay_path


def _build_template_display_name(template: "TemplateRecord") -> str:
    """Build a human-readable filename stem for a template image."""
    slots_with_labels = [s for s in template.face_slots if s.speaker_label]
    is_wide = template.type.upper() == "WIDE" or template.n_faces >= 2
    if is_wide:
        if slots_with_labels:
            labels = "_".join(s.speaker_label.replace("SPEAKER_", "spk") for s in
                              sorted(slots_with_labels, key=lambda s: s.center_norm[0]))
        else:
            labels = f"{template.n_faces}face"
        return f"wide_{labels}_{template.template_id:02d}"
    elif slots_with_labels:
        spk = slots_with_labels[0].speaker_label.replace("SPEAKER_", "spk")
        return f"{spk}_{template.template_id:02d}"
    else:
        return f"scene_{template.template_id:02d}"


def _draw_template_overlay(frame: np.ndarray, template: TemplateRecord) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    for slot in template.face_slots:
        x1, y1, x2, y2 = slot.bbox
        p1 = (int(round(clamp(x1, 0, w - 1))), int(round(clamp(y1, 0, h - 1))))
        p2 = (int(round(clamp(x2, 0, w - 1))), int(round(clamp(y2, 0, h - 1))))
        cv2.rectangle(overlay, p1, p2, (0, 255, 0), 3)
        label_bits = [slot.slot_name]
        if slot.speaker_label:
            label_bits.append(slot.speaker_label)
        label_bits.append(f"{slot.confidence:.2f}")
        label = " | ".join(label_bits)
        tx = p1[0]
        ty = max(22, p1[1] - 8)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    dname = template.display_name or f"template_{template.template_id:02d}"
    header = f"{dname} | {template.type} | faces={template.n_faces}"
    cv2.putText(overlay, header, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _write_template_manifest(job_dir: Path, templates: Sequence[TemplateRecord], disabled_ids: Sequence[int]) -> None:
    manifest = {
        "templates": [
            {
                "template_id": int(t.template_id),
                "display_name": t.display_name or f"template_{t.template_id:02d}",
                "raw_image": str((_template_image_paths(job_dir, t.template_id, t.display_name)[0]).relative_to(job_dir)),
                "overlay_image": str((_template_image_paths(job_dir, t.template_id, t.display_name)[1]).relative_to(job_dir)),
                "type": t.type,
                "n_faces": int(t.n_faces),
                "slot_names": [s.slot_name for s in t.face_slots],
                "speaker_votes": dict(t.speaker_votes),
                "speaker_labels": [s.speaker_label for s in t.face_slots if s.speaker_label],
            }
            for t in templates
        ],
        "disabled_template_ids": [int(x) for x in sorted(set(int(v) for v in disabled_ids))],
        "instructions": "Delete an image file and rerun to prune that template from the pipeline.",
    }
    write_json(job_dir / TEMPLATE_DIR_NAME / TEMPLATE_MANIFEST_FILE, manifest)


def _ensure_template_images(
    video_path: str,
    job_dir: Path,
    templates: List[TemplateRecord],
    logger: Optional[Logger] = None,
    registry: Optional[Dict[str, Any]] = None,
    force_overwrite: bool = True,
) -> Tuple[List[TemplateRecord], List[int]]:
    """
    Export per-template images for human review.

    When force_overwrite=True (default), existing images are always regenerated so the
    display_name and overlay labels stay up-to-date with the current speaker assignments.
    Missing images (relative to manifest) are treated as manual pruning on reruns only
    when force_overwrite=False.
    """
    ensure_dir(job_dir / TEMPLATE_DIR_NAME)
    manifest_path = job_dir / TEMPLATE_DIR_NAME / TEMPLATE_MANIFEST_FILE
    manifest_exists = manifest_path.exists() and not force_overwrite

    # Write a README on first run
    readme = job_dir / TEMPLATE_DIR_NAME / "README.txt"
    if not readme.exists():
        try:
            readme.write_text(
                "TEMPLATE IMAGES\n"
                "===============\n"
                "Each template is one distinct camera layout detected in the video.\n"
                "  <name>.jpg         raw frame\n"
                "  <name>_overlay.jpg  frame with face boxes and speaker labels\n\n"
                "TO PRUNE A TEMPLATE:\n"
                "  Delete the <name>.jpg file, then rerun â€” the pipeline will skip it.\n\n"
                "Naming convention:\n"
                "  spk<N>_XX.jpg       single-speaker close-up\n"
                "  wide_spkA_spkB_XX.jpg  two-person wide shot\n"
                "  scene_XX.jpg        no face detected\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    registry_disabled: set[int] = set()
    if registry:
        raw_disabled = registry.get("disabled_template_ids", [])
        if isinstance(raw_disabled, list):
            for x in raw_disabled:
                try:
                    registry_disabled.add(int(x))
                except Exception:
                    continue

    active: List[TemplateRecord] = []
    pruned_missing: List[int] = []
    exported = 0

    for template in sorted(templates, key=lambda t: t.template_id):
        if template.template_id in registry_disabled:
            if logger:
                logger(f"[TEMPLATE] id={template.template_id} DISABLED by registry â€” skipping")
            continue

        raw_path, overlay_path = _template_image_paths(job_dir, template.template_id, template.display_name)

        # On rerun without force, treat missing image as user-pruned
        if manifest_exists and not raw_path.exists() and not force_overwrite:
            pruned_missing.append(template.template_id)
            if logger:
                logger(f"[TEMPLATE] id={template.template_id} PRUNED â€” image deleted by user")
            continue

        # Load the source frame
        frame = None
        src = Path(template.canonical_frame) if template.canonical_frame else None
        if src is not None and src.exists() and str(src) != str(raw_path):
            frame = cv2.imread(str(src))
        if frame is None:
            ts = float(template.canonical_timestamp)
            frame = extract_frame(video_path, ts)
            if logger and frame is None:
                logger(f"[TEMPLATE] id={template.template_id} WARNING: could not extract frame at t={ts:.2f}s")
        if frame is None:
            if logger:
                logger(f"[TEMPLATE] id={template.template_id} SKIPPED â€” no frame available")
            continue

        try:
            ok_raw, buf_raw = cv2.imencode(".jpg", frame)
            if ok_raw:
                raw_path.write_bytes(buf_raw.tobytes())
            overlay = _draw_template_overlay(frame, template)
            ok_ov, buf_ov = cv2.imencode(".jpg", overlay)
            if ok_ov:
                overlay_path.write_bytes(buf_ov.tobytes())
            if ok_raw:
                template.canonical_frame = str(raw_path)
                exported += 1
                active.append(template)
                dname = template.display_name or f"template_{template.template_id:02d}"
                if logger:
                    logger(f"[TEMPLATE] EXPORTED id={template.template_id} name={dname} -> {raw_path.name}")
            else:
                if logger:
                    logger(f"[TEMPLATE] id={template.template_id} ENCODE FAILED")
        except Exception as e:
            if logger:
                logger(f"[TEMPLATE] id={template.template_id} EXPORT FAILED: {e}")

    disabled_ids = sorted(set(registry_disabled) | set(pruned_missing))
    _write_template_manifest(job_dir, active, disabled_ids)

    if logger:
        logger(f"[TEMPLATE] Export complete â€” active={len(active)} exported={exported} pruned={len(pruned_missing)} disabled={len(registry_disabled)}")
        logger(f"[TEMPLATE] Images saved to: {job_dir / TEMPLATE_DIR_NAME}")
    return active, disabled_ids


def _template_slot_by_name(template: TemplateRecord, slot_name: Optional[str]) -> Optional[FaceSlot]:
    if not slot_name:
        return None
    for slot in template.face_slots:
        if slot.slot_name == slot_name:
            return slot
    return None


def _template_slot_to_face(template: TemplateRecord, slot_name: Optional[str], source_w: int, source_h: int) -> Optional[FaceObs]:
    slot = _template_slot_by_name(template, slot_name)
    if slot is None:
        return None
    x1, y1, x2, y2 = slot.bbox
    cx, cy = slot.center
    area = max(0.0, (x2 - x1) * (y2 - y1))
    return FaceObs(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        conf=float(slot.confidence),
        cx=float(cx),
        cy=float(cy),
        area=float(area),
        embedding=np.asarray(slot.embedding, dtype=np.float32) if slot.embedding is not None else None,
        shirt_hist=np.asarray(slot.shirt_hist, dtype=np.float32) if slot.shirt_hist is not None else None,
        shirt_rgb=slot.shirt_rgb,
    )


def _assign_template_slot_labels(
    templates: Sequence[TemplateRecord],
    speaker_db: Dict[str, List[float]],
    manual_template_ids: Optional[Set[int]] = None,
    logger: Optional[Logger] = None,
) -> None:
    """
    Attach best-effort speaker labels to face slots so wide-shot routing can keep a
    stable left/right speaker map across cuts.

    manual_template_ids: when provided, slots in those templates that already carry a
    speaker_label (set from the manual JSON) are preserved as-is and not overwritten by
    the embedding comparison. This prevents the common failure where both templates are
    authored at timestamps where only one speaker is visible, making all embeddings
    identical and causing the comparison to always pick the same speaker.
    """
    if not templates or not speaker_db:
        return
    speaker_vecs = {spk: np.asarray(vec, dtype=np.float32) for spk, vec in speaker_db.items() if vec}
    for template in templates:
        is_manual = manual_template_ids is not None and template.template_id in manual_template_ids
        for slot in template.face_slots:
            # For manually authored templates, trust the speaker_label set in the JSON.
            if is_manual and slot.speaker_label:
                if logger and VERBOSE_STAGE_LOGGING:
                    logger(
                        f"[SPEAKER] template {template.template_id} slot {slot.slot_name}"
                        f" -> {slot.speaker_label} (manual label preserved)"
                    )
                continue
            best_spk = None
            best_score = -1e9
            slot_vec = np.asarray(slot.embedding, dtype=np.float32) if slot.embedding is not None else None
            if slot_vec is None:
                continue
            for spk, proto in speaker_vecs.items():
                score = cosine_sim(slot_vec, proto)
                if score > best_score:
                    best_score = score
                    best_spk = spk
            if best_spk is not None and best_score >= EMBED_SIM_THRESHOLD:
                slot.speaker_label = best_spk
                if logger and VERBOSE_STAGE_LOGGING:
                    logger(f"[SPEAKER] template {template.template_id} slot {slot.slot_name} -> {best_spk} ({best_score:.3f})")
    # Set human-readable display names now that speaker labels are assigned
    for template in templates:
        template.display_name = _build_template_display_name(template)
        if logger and VERBOSE_STAGE_LOGGING:
            logger(f"[TEMPLATE] id={template.template_id} display_name={template.display_name}")
def discover_templates(
    video_path: str,
    cuts: List[float],
    face_backend: BaseFaceBackend,
    job_dir: Path,
    logger: Logger,
) -> Tuple[List[TemplateRecord], Dict[int, int], Dict[int, List[np.ndarray]]]:
    ensure_dir(job_dir / TEMPLATE_DIR_NAME)
    templates: List[TemplateRecord] = []
    scene_to_template: Dict[int, int] = {}
    template_frames: Dict[int, List[np.ndarray]] = {}
    template_scene_frames: Dict[int, List[np.ndarray]] = {}
    template_samples: Dict[int, List[List[FaceObs]]] = {}
    template_segment_indices: Dict[int, List[int]] = {}

    src_w, src_h, fps = get_video_info(video_path)
    logger(f"[TEMPLATE] source {src_w}x{src_h} @ {fps:.2f} fps")

    def scene_sample_times(i: int) -> List[float]:
        seg_start = cuts[i]
        seg_end = cuts[i + 1]
        frames: List[float] = []
        # FPS-aware sampling: 2 pre-cut frames, 2 post-cut frames
        one_frame = 1.0 / max(1.0, fps)
        pre_offsets = [1 * one_frame, 4 * one_frame]       # 1 and 4 frames before cut
        post_offsets = [2 * one_frame, 6 * one_frame]      # 2 and 6 frames after cut
        for dt in pre_offsets:
            t = seg_start - dt
            if t >= 0:
                frames.append(t)
        for dt in post_offsets:
            t = seg_start + dt
            if t < seg_end:
                frames.append(t)
        if not frames:
            frames = [seg_start + min(0.3, max(0.0, (seg_end - seg_start) * 0.25))]
        return frames

    # First pass: cluster scenes by pHash
    reps: List[Dict[str, Any]] = []
    for i in range(len(cuts) - 1):
        times = scene_sample_times(i)
        frames = [extract_frame(video_path, t) for t in times]
        valid = [f for f in frames if f is not None]
        if not valid:
            scene_to_template[i] = 0
            continue
        ph, canonical_candidate = _medoid_hash(valid)
        if ph is None or canonical_candidate is None:
            scene_to_template[i] = 0
            continue
        canonical = _pick_sharpest([f for f in frames if f is not None])
        if canonical is None:
            canonical = canonical_candidate
        matched = None
        for rep in reps:
            if _phash_distance(ph, rep["phash"]) < PHASH_THRESHOLD:
                matched = rep["template_id"]
                rep["scene_indices"].append(i)
                rep["frames"].append(canonical)
                break
        if matched is None:
            matched = len(reps)
            reps.append({
                "template_id": matched,
                "phash": ph,
                "scene_indices": [i],
                "frames": [canonical],
            })
        scene_to_template[i] = matched

    if not reps:
        fallback = extract_frame(video_path, cuts[0])
        if fallback is None:
            fallback = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        reps.append({"template_id": 0, "phash": np.zeros(64, dtype=np.uint8), "scene_indices": [0], "frames": [fallback]})

    # For each rep/template, collect more frames from all scenes in it
    for rep in reps:
        t_id = rep["template_id"]
        frames_for_template: List[np.ndarray] = []
        observations: List[List[FaceObs]] = []
        for scene_idx in rep["scene_indices"]:
            for t in scene_sample_times(scene_idx):
                frame = extract_frame(video_path, t)
                if frame is None:
                    continue
                frames_for_template.append(frame)
                obs = face_backend.detect(frame)
                observations.append(obs)
        template_frames[t_id] = frames_for_template

        canonical = _pick_sharpest(frames_for_template)
        if canonical is None:
            canonical = rep["frames"][0]
        cpath = job_dir / TEMPLATE_DIR_NAME / f"template_{t_id:02d}.jpg"
        cv2.imwrite(str(cpath), canonical)

        face_slots, crop_anchor, speaker_embeddings, shirt_signatures, slot_geometry = _build_template_geometry(canonical, observations)
        template_type = "SINGLE" if len(face_slots) <= 1 else ("WIDE" if len(face_slots) == 2 else "MULTI")

        tmpl = TemplateRecord(
            template_id=t_id,
            phash=[int(x) for x in np.asarray(rep["phash"], dtype=np.uint8).ravel().tolist()],
            segment_indices=rep["scene_indices"],
            type=template_type,
            canonical_frame=str(cpath),
            canonical_timestamp=float(cuts[rep["scene_indices"][0]] if rep["scene_indices"] else 0.0),
            n_faces=len(face_slots),
            face_slots=face_slots,
            crop_anchor=crop_anchor,
            speaker_votes={},
            speaker_embeddings=speaker_embeddings,
            shirt_signatures=shirt_signatures,
            slot_geometry=slot_geometry,
        )
        templates.append(tmpl)
        template_scene_frames[t_id] = frames_for_template

    # Second pass: window-based refinement against the provisional templates.
    try:
        refined_map = _refine_scene_template_map(
            video_path=video_path,
            cuts=cuts,
            templates=templates,
            face_backend=face_backend,
            speaker_db={},
            source_w=src_w,
            source_h=src_h,
            logger=logger,
        )
        if refined_map:
            scene_to_template.update(refined_map)
    except Exception as e:
        logger(f"[TEMPLATE] refinement skipped: {e}")

    templates = _template_prune_duplicates(templates, logger=logger)
    return templates, scene_to_template, template_scene_frames


def _deserialize_face_slot(data: Dict[str, Any]) -> FaceSlot:
    center = data.get("center", (0.0, 0.0))
    bbox = data.get("bbox", (0.0, 0.0, 0.0, 0.0))
    center_norm = data.get("center_norm", (0.0, 0.0))
    bbox_norm = data.get("bbox_norm", (0.0, 0.0, 0.0, 0.0))
    shirt_rgb = data.get("shirt_rgb")
    if isinstance(shirt_rgb, list):
        shirt_rgb = tuple(int(v) for v in shirt_rgb)
    return FaceSlot(
        slot_name=str(data.get("slot_name", "slot_0")),
        center=(float(center[0]), float(center[1])),
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        center_norm=(float(center_norm[0]), float(center_norm[1])),
        bbox_norm=(float(bbox_norm[0]), float(bbox_norm[1]), float(bbox_norm[2]), float(bbox_norm[3])),
        confidence=float(data.get("confidence", 0.0)),
        embedding=data.get("embedding"),
        shirt_hist=data.get("shirt_hist"),
        shirt_rgb=shirt_rgb,
        speaker_label=data.get("speaker_label"),
        slot_order=int(data.get("slot_order", 0)),
    )



def _deserialize_template_record(data: Dict[str, Any]) -> TemplateRecord:
    face_slots = [_deserialize_face_slot(fs) for fs in data.get("face_slots", []) if isinstance(fs, dict)]
    crop_anchor = {k: float(v) for k, v in dict(data.get("crop_anchor", {})).items() if isinstance(v, (int, float, str))}
    speaker_votes = {str(k): int(v) for k, v in dict(data.get("speaker_votes", {})).items()}
    speaker_embeddings = {str(k): list(map(float, v)) for k, v in dict(data.get("speaker_embeddings", {})).items() if isinstance(v, (list, tuple))}
    shirt_signatures = {str(k): list(map(float, v)) for k, v in dict(data.get("shirt_signatures", {})).items() if isinstance(v, (list, tuple))}
    slot_geometry_raw = dict(data.get("slot_geometry", {}))
    slot_geometry: Dict[str, Dict[str, float]] = {}
    if isinstance(slot_geometry_raw, dict):
        for k, v in slot_geometry_raw.items():
            if isinstance(v, dict):
                slot_geometry[str(k)] = {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float, str))}
    phash = [int(x) for x in data.get("phash", [])]
    return TemplateRecord(
        template_id=int(data.get("template_id", 0)),
        phash=phash,
        segment_indices=[int(x) for x in data.get("segment_indices", [])],
        type=str(data.get("type", "SINGLE")),
        canonical_frame=str(data.get("canonical_frame", "")),
        canonical_timestamp=float(data.get("canonical_timestamp", 0.0)),
        n_faces=int(data.get("n_faces", len(face_slots))),
        face_slots=face_slots,
        crop_anchor=crop_anchor,
        speaker_votes=speaker_votes,
        speaker_embeddings=speaker_embeddings,
        shirt_signatures=shirt_signatures,
        slot_geometry=slot_geometry,
        display_name=str(data.get("display_name", "")),
    )

def deserialize_template_registry(data: Dict[str, Any]) -> Tuple[List[TemplateRecord], Dict[int, int], Dict[str, List[float]], List[float]]:
    templates = [_deserialize_template_record(t) for t in data.get("templates", []) if isinstance(t, dict)]
    scene_to_template_raw = data.get("scene_to_template", {})
    scene_to_template: Dict[int, int] = {}
    if isinstance(scene_to_template_raw, dict):
        for k, v in scene_to_template_raw.items():
            try:
                scene_to_template[int(k)] = int(v)
            except Exception:
                continue
    else:
        for t in templates:
            for idx in t.segment_indices:
                scene_to_template[int(idx)] = int(t.template_id)
    speaker_db_raw = data.get("speaker_embedding_db", {})
    speaker_db: Dict[str, List[float]] = {}
    if isinstance(speaker_db_raw, dict):
        for k, v in speaker_db_raw.items():
            if isinstance(v, (list, tuple)):
                try:
                    speaker_db[str(k)] = [float(x) for x in v]
                except Exception:
                    continue
    cuts: List[float] = []
    for x in data.get("cuts", []):
        try:
            cuts.append(float(x))
        except Exception:
            continue
    return templates, scene_to_template, speaker_db, cuts



def save_template_registry(job_dir: Path, templates: List[TemplateRecord], speaker_db: Dict[str, List[float]], cuts: List[float], scene_to_template: Dict[int, int], manual_templates: Optional[List[TemplateRecord]] = None) -> None:
    ensure_dir(job_dir / TEMPLATE_DIR_NAME)
    reg_path = job_dir / TEMPLATE_DIR_NAME / REGISTRY_FILE

    preserved: Dict[str, Any] = {}
    if reg_path.exists():
        try:
            existing = read_json(reg_path)
            if isinstance(existing, dict):
                for key in ("manual_templates", "disabled_template_ids"):
                    if key in existing:
                        preserved[key] = existing[key]
        except Exception:
            pass

    data: Dict[str, Any] = {
        "n_segments": max(0, len(cuts) - 1),
        "cuts": [float(x) for x in cuts],
        "scene_to_template": {int(k): int(v) for k, v in scene_to_template.items()},
        "templates": [_serialize_template(t) for t in templates],
        "speaker_embedding_db": {k: list(map(float, v)) for k, v in speaker_db.items()},
    }
    if manual_templates is not None:
        data["manual_templates"] = [_serialize_template(t) for t in manual_templates]
    data.update(preserved)

    tmp_path = reg_path.with_suffix(reg_path.suffix + ".tmp")
    write_json(tmp_path, data)
    os.replace(tmp_path, reg_path)




def _force_clear_template_cache(job_dir: Path, logger: Optional[Logger] = None) -> None:
    """Delete template registry and all existing template images so we always rebuild fresh."""
    reg = job_dir / TEMPLATE_DIR_NAME / REGISTRY_FILE
    manifest = job_dir / TEMPLATE_DIR_NAME / TEMPLATE_MANIFEST_FILE
    for p in (reg, manifest):
        if p.exists():
            try:
                p.unlink()
                if logger:
                    logger(f"[CACHE] Deleted old cache file: {p.name}")
            except Exception as e:
                if logger:
                    logger(f"[CACHE] Could not delete {p.name}: {e}")
    # Also wipe old image files so stale names dont accumulate
    tdir = job_dir / TEMPLATE_DIR_NAME
    if tdir.exists():
        for img in list(tdir.glob("*.jpg")):
            try:
                img.unlink()
            except Exception:
                pass


def _export_raw_template_images(
    video_path: str,
    job_dir: Path,
    templates: List[TemplateRecord],
    logger: Optional[Logger] = None,
) -> None:
    """
    Export raw (no-overlay) canonical frames for every template immediately after discovery.
    This runs BEFORE speaker analysis so the user can inspect/delete duplicates right away.
    """
    ensure_dir(job_dir / TEMPLATE_DIR_NAME)
    for template in sorted(templates, key=lambda t: t.template_id):
        # Use a temporary stem before display_name is known
        stem = f"template_{template.template_id:02d}_raw"
        raw_path = job_dir / TEMPLATE_DIR_NAME / f"{stem}.jpg"
        # Try canonical_frame path first, then extract from video
        frame = None
        src = Path(template.canonical_frame) if template.canonical_frame else None
        if src is not None and src.exists():
            frame = cv2.imread(str(src))
        if frame is None:
            ts = float(template.canonical_timestamp)
            frame = extract_frame(video_path, ts)
        if frame is None:
            if logger:
                logger(f"[TEMPLATE] id={template.template_id} WARNING: could not extract raw frame at t={template.canonical_timestamp:.2f}s")
            continue
        try:
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                raw_path.write_bytes(buf.tobytes())
            if logger:
                status = "OK" if ok2 else "ENCODE_FAILED"
                logger(f"[TEMPLATE] Early export id={template.template_id} type={template.type} faces={template.n_faces} [{status}] -> {raw_path.name}")
        except Exception as e:
            if logger:
                logger(f"[TEMPLATE] id={template.template_id} early export exception: {e}")



def load_template_registry(job_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the saved template registry JSON from *job_dir/templates/template_registry.json*.

    Returns the parsed dict if the file exists and is valid JSON, otherwise returns None so
    the caller falls back to a full template-discovery pass.
    """
    reg_path = job_dir / TEMPLATE_DIR_NAME / REGISTRY_FILE
    if not reg_path.exists():
        return None
    try:
        data = read_json(reg_path)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None
def _remap_scene_templates_to_active(
    scene_to_template: Dict[int, int],
    all_templates: Sequence[TemplateRecord],
    active_templates: Sequence[TemplateRecord],
    pruned_ids: Sequence[int],
    logger: Optional[Logger] = None,
) -> Dict[int, int]:
    """
    When a user manually prunes template images, remap scenes that pointed to the removed
    templates onto the closest remaining template by pHash.
    """
    active_by_id = {t.template_id: t for t in active_templates}
    if not active_by_id:
        return scene_to_template

    old_by_id = {t.template_id: t for t in all_templates}
    active_list = list(active_templates)
    remapped: Dict[int, int] = {}

    for scene_idx, tmpl_id in scene_to_template.items():
        if tmpl_id in active_by_id:
            remapped[scene_idx] = tmpl_id
            continue

        old = old_by_id.get(tmpl_id)
        if old is not None:
            old_hash = np.asarray(old.phash, dtype=np.uint8)
            best = min(
                active_list,
                key=lambda t: _phash_distance(old_hash, np.asarray(t.phash, dtype=np.uint8)),
            )
            best_id = best.template_id
        else:
            best_id = active_list[0].template_id

        remapped[scene_idx] = best_id
        if logger:
            logger(f"[TEMPLATE] remapped scene {scene_idx} from pruned template {tmpl_id} -> {best_id}")

    if logger and pruned_ids:
        logger(f"[TEMPLATE] pruned template ids: {sorted(set(int(x) for x in pruned_ids))}")
    return remapped

    data: Dict[str, Any] = {
        "n_segments": max(0, len(cuts) - 1),
        "cuts": [float(x) for x in cuts],
        "scene_to_template": {int(k): int(v) for k, v in scene_to_template.items()},
        "templates": [_serialize_template(t) for t in templates],
        "speaker_embedding_db": {k: list(map(float, v)) for k, v in speaker_db.items()},
    }
    data.update(preserved)
    write_json(reg_path, data)


# ======================================================================
# Speaker prototype DB
# ======================================================================

def build_speaker_embedding_db(
    templates: List[TemplateRecord],
    scene_to_template: Dict[int, int],
    diar_timeline: Sequence[DiarSegment],
    video_path: str,
    face_backend: BaseFaceBackend,
    cuts: List[float],
    logger: Logger,
) -> Dict[str, List[float]]:
    """
    Build a best-effort speaker embedding DB from close-up and stable wide-shot observations.
    The rule is intentionally conservative: only inject an embedding when one face slot is
    clearly dominant, or the template is a single-face template.
    """
    db: Dict[str, List[np.ndarray]] = {}

    template_by_id = {t.template_id: t for t in templates}
    for scene_idx, tmpl_id in scene_to_template.items():
        tmpl = template_by_id.get(tmpl_id)
        if tmpl is None or not tmpl.face_slots:
            continue

        ranked = sorted(
            tmpl.face_slots,
            key=lambda s: (s.confidence * max(1e-6, (s.bbox_norm[2] - s.bbox_norm[0]) * (s.bbox_norm[3] - s.bbox_norm[1]))),
            reverse=True,
        )
        if not ranked:
            continue

        use_slot: Optional[FaceSlot] = None
        if len(ranked) == 1 or tmpl.type == "SINGLE":
            use_slot = ranked[0]
        elif len(ranked) >= 2:
            gap = ranked[0].confidence - ranked[1].confidence
            if gap >= 0.15:
                use_slot = ranked[0]

        if use_slot is None or use_slot.embedding is None:
            continue

        mid = (cuts[scene_idx] + cuts[scene_idx + 1]) / 2.0
        spk = active_speaker_at(mid, diar_timeline)
        if spk == "UNKNOWN":
            continue
        db.setdefault(spk, []).append(np.asarray(use_slot.embedding, dtype=np.float32))

    out: Dict[str, List[float]] = {}
    for spk, vecs in db.items():
        if not vecs:
            continue
        proto = normalize_vec(np.mean(np.stack(vecs, axis=0), axis=0))
        out[spk] = proto.tolist()
        logger(f"[SPEAKER-DB] {spk}: built from {len(vecs)} observations")
    return out


def assign_template_votes(
    templates: List[TemplateRecord],
    scene_to_template: Dict[int, int],
    diar_timeline: Sequence[DiarSegment],
    cuts: List[float],
    logger: Logger,
) -> None:
    """
    Fill per-template speaker votes using diarization midpoints.
    """
    template_by_id = {t.template_id: t for t in templates}
    for scene_idx, tmpl_id in scene_to_template.items():
        tmpl = template_by_id.get(tmpl_id)
        if tmpl is None:
            continue
        mid = (cuts[scene_idx] + cuts[scene_idx + 1]) / 2.0
        spk = active_speaker_at(mid, diar_timeline)
        tmpl.speaker_votes[spk] = tmpl.speaker_votes.get(spk, 0) + 1

    for t in templates:
        if t.speaker_votes:
            logger(f"[TEMPLATE] #{t.template_id} votes: {t.speaker_votes}")

def _scene_index_for_time(cuts: Sequence[float], t: float) -> Optional[int]:
    for j in range(len(cuts) - 1):
        if cuts[j] <= t < cuts[j + 1]:
            return j
    if len(cuts) >= 2 and abs(t - cuts[-1]) < 1e-3:
        return len(cuts) - 2
    return None

def _template_primary_speaker_label(template: TemplateRecord) -> Optional[str]:
    labels = [s.speaker_label for s in template.face_slots if s.speaker_label]
    if len(labels) == 1:
        return str(labels[0])
    if len(template.face_slots) == 1 and template.face_slots[0].speaker_label:
        return str(template.face_slots[0].speaker_label)
    return None

def _classify_segments_with_windows(
    job_dir: Optional[Path],
    video_path: str,
    segs: Sequence[DiarSegment],
    templates: Sequence[TemplateRecord],
    scene_to_template: Dict[int, int],
    speaker_db: Dict[str, List[float]],
    diar_timeline: Sequence[DiarSegment],
    cuts: Sequence[float],
    face_backend: BaseFaceBackend,
    source_w: int,
    source_h: int,
    logger: Logger,
) -> List[ClassifiedSegment]:
    classified: List[ClassifiedSegment] = []
    if not templates:
        return classified
    default_template_id = int(templates[0].template_id)
    for i, seg in enumerate(segs):
        mid = (seg.start + seg.end) / 2.0
        scene_idx = _scene_index_for_time(cuts, mid)
        template_id = scene_to_template.get(scene_idx, default_template_id)
        cls_id, cls_conf, hidden_cut, scores = _classify_template_window_cached(
            job_dir=job_dir,
            cache_label=f"plan_seg_{i + 1:04d}",
            video_path=video_path,
            cut_time=mid,
            templates=templates,
            face_backend=face_backend,
            speaker_db=speaker_db,
            source_w=source_w,
            source_h=source_h,
            start_bound=seg.start,
            end_bound=seg.end,
            logger=logger,
            active_speaker_hint="UNKNOWN",
        )
        if cls_id is not None and (hidden_cut or cls_conf >= 0.28):
            template_id = int(cls_id)
        if logger and VERBOSE_STAGE_LOGGING:
            top_scores = ", ".join(
                f"#{tid}:{score:.3f}"
                for tid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            )
            logger(
                f"[PLAN] seg#{i + 1} classified template={template_id} "
                f"conf={cls_conf:.3f} hidden={hidden_cut} scores=[{top_scores}]"
            )
        classified.append(ClassifiedSegment(
            index=i,
            start=float(seg.start),
            end=float(seg.end),
            template_id=int(template_id),
            confidence=float(cls_conf),
            hidden=bool(hidden_cut),
            scores={int(k): float(v) for k, v in scores.items()},
        ))
    return classified

def _build_diarization_speaker_map(
    templates: Sequence[TemplateRecord],
    classified: Sequence[ClassifiedSegment],
    diar_timeline: Sequence[DiarSegment],
    logger: Logger,
) -> Dict[str, str]:
    """
    Map diarization speaker labels onto template speaker labels. Solo classified
    segments are the visual truth; wide shots consume the mapped labels.
    """
    template_by_id = {t.template_id: t for t in templates}
    matrix: Dict[str, Dict[str, float]] = {}
    for item in classified:
        tmpl = template_by_id.get(item.template_id)
        if tmpl is None or _template_is_wide(tmpl):
            continue
        visual_spk = _template_primary_speaker_label(tmpl)
        if not visual_spk:
            continue
        diar_spk = active_speaker_at((item.start + item.end) / 2.0, diar_timeline)
        if not diar_spk or diar_spk == "UNKNOWN":
            continue
        dur = max(0.05, item.end - item.start)
        weight = dur * max(0.25, item.confidence)
        matrix.setdefault(diar_spk, {})
        matrix[diar_spk][visual_spk] = matrix[diar_spk].get(visual_spk, 0.0) + weight

    pairs: List[Tuple[float, str, str]] = []
    for diar_spk, visual_scores in matrix.items():
        for visual_spk, score in visual_scores.items():
            pairs.append((float(score), diar_spk, visual_spk))
    pairs.sort(reverse=True)

    mapping: Dict[str, str] = {}
    used_visual: Set[str] = set()
    for score, diar_spk, visual_spk in pairs:
        if diar_spk in mapping or visual_spk in used_visual:
            continue
        mapping[diar_spk] = visual_spk
        used_visual.add(visual_spk)
        logger(f"[DIAR-MAP] {diar_spk} -> {visual_spk} score={score:.3f}")

    diar_speakers = sorted({s.speaker for s in diar_timeline if s.speaker and s.speaker != "UNKNOWN"})
    visual_speakers = sorted({s.speaker_label for t in templates for s in t.face_slots if s.speaker_label})
    if len(diar_speakers) == 2 and len(visual_speakers) == 2 and len(mapping) == 1:
        missing_diar = [s for s in diar_speakers if s not in mapping]
        missing_visual = [s for s in visual_speakers if s not in used_visual]
        if len(missing_diar) == 1 and len(missing_visual) == 1:
            mapping[missing_diar[0]] = str(missing_visual[0])
            logger(f"[DIAR-MAP] {missing_diar[0]} -> {missing_visual[0]} inferred")

    if not mapping:
        logger("[DIAR-MAP] no speaker mapping built; wide shots will use diar labels as-is")
    return mapping

def _apply_speaker_map_to_timeline(timeline: Sequence[DiarSegment], mapping: Dict[str, str]) -> List[DiarSegment]:
    if not mapping:
        return [DiarSegment(start=s.start, end=s.end, speaker=s.speaker) for s in timeline]
    return [
        DiarSegment(start=s.start, end=s.end, speaker=mapping.get(s.speaker, s.speaker))
        for s in timeline
    ]

def assign_template_votes_from_classified_segments(
    templates: Sequence[TemplateRecord],
    classified: Sequence[ClassifiedSegment],
    diar_timeline: Sequence[DiarSegment],
    logger: Logger,
) -> None:
    template_by_id = {t.template_id: t for t in templates}
    for t in templates:
        t.speaker_votes.clear()
    for item in classified:
        tmpl = template_by_id.get(item.template_id)
        if tmpl is None:
            continue
        spk = active_speaker_at((item.start + item.end) / 2.0, diar_timeline)
        if spk and spk != "UNKNOWN":
            tmpl.speaker_votes[spk] = tmpl.speaker_votes.get(spk, 0) + 1
    for t in templates:
        if t.speaker_votes:
            logger(f"[TEMPLATE] #{t.template_id} votes: {t.speaker_votes}")

def _face_count_at_time(video_path: str, t: float, face_backend: BaseFaceBackend) -> Optional[int]:
    frame = extract_frame(video_path, t)
    if frame is None:
        return None
    try:
        return len(face_backend.detect(frame))
    except Exception:
        return None

def _trim_zero_face_boundaries(
    video_path: str,
    segs: Sequence[DiarSegment],
    fps: float,
    face_backend: BaseFaceBackend,
    job_dir: Path,
    logger: Logger,
) -> Tuple[List[DiarSegment], List[SegmentTrimInfo]]:
    fps_local = max(1.0, float(fps))
    frame_dur = 1.0 / fps_local
    max_frames = max(1, min(
        ZERO_FACE_BOUNDARY_TRIM_MAX_FRAMES,
        int(round(ZERO_FACE_BOUNDARY_TRIM_MAX_SEC * fps_local)),
    ))
    trimmed: List[DiarSegment] = []
    manifest: List[SegmentTrimInfo] = []
    output_cursor = 0.0

    for idx, seg in enumerate(segs):
        original_start = float(seg.start)
        original_end = float(seg.end)
        start_frames = 0
        end_frames = 0

        for off in range(max_frames):
            t = original_start + (off * frame_dur)
            if t >= original_end:
                break
            count = _face_count_at_time(video_path, t, face_backend)
            if count == 0:
                start_frames += 1
            else:
                break

        for off in range(max_frames):
            t = original_end - ((off + 1) * frame_dur)
            if t <= original_start:
                break
            count = _face_count_at_time(video_path, t, face_backend)
            if count == 0:
                end_frames += 1
            else:
                break

        render_start = original_start + (start_frames * frame_dur)
        render_end = original_end - (end_frames * frame_dur)
        if render_end - render_start <= 0.15:
            render_start = original_start
            render_end = original_end
            start_frames = 0
            end_frames = 0

        start_trim = max(0.0, render_start - original_start)
        end_trim = max(0.0, original_end - render_end)
        if start_trim > 0.0 or end_trim > 0.0:
            logger(
                f"[TRIM] seg#{idx + 1} zero-face boundary trim "
                f"start={start_trim:.4f}s end={end_trim:.4f}s "
                f"frames=({start_frames},{end_frames})"
            )

        out_start = output_cursor
        out_end = out_start + max(0.0, render_end - render_start)
        output_cursor = out_end
        trimmed.append(DiarSegment(start=render_start, end=render_end, speaker=seg.speaker))
        manifest.append(SegmentTrimInfo(
            index=idx,
            original_start=original_start,
            original_end=original_end,
            render_start=render_start,
            render_end=render_end,
            start_trim=start_trim,
            end_trim=end_trim,
            output_start=out_start,
            output_end=out_end,
        ))

    try:
        data = [asdict(x) for x in manifest]
        write_json(job_dir / TRIM_MANIFEST_FILE, data)
        write_json(job_dir / TIMELINE_MAP_FILE, {
            "timestamp_mode": "source_absolute_to_output_relative",
            "segments": data,
        })
        logger(f"[TRIM] timeline map saved -> {job_dir / TIMELINE_MAP_FILE}")
    except Exception:
        pass
    return trimmed, manifest




# ======================================================================
# Windowed template classification, hidden-cut detection, and debug exports
# ======================================================================

def _analysis_frame(frame: Optional[np.ndarray], max_side: int = ANALYSIS_MAX_SIDE) -> Optional[np.ndarray]:
    """Downscale frames for analysis so template matching stays lightweight."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _analysis_frame_at(video_path: str, ts_sec: float) -> Optional[np.ndarray]:
    frame = extract_frame(video_path, ts_sec)
    return _analysis_frame(frame)


def _frame_exposure_score(frame: Optional[np.ndarray]) -> float:
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    # Prefer mid-range, non-clipped exposure.
    return float(1.0 - min(1.0, abs(mean - 128.0) / 128.0) * 0.65 - min(1.0, abs(std - 60.0) / 60.0) * 0.35)


def _frame_quality_score(frame: Optional[np.ndarray], face_count: int) -> float:
    if frame is None:
        return 0.0
    sharp = sharpness_score(frame)
    exp = _frame_exposure_score(frame)
    fc = clamp(face_count / 2.0, 0.0, 1.0)
    sharp_norm = clamp(sharp / 600.0, 0.0, 1.0)
    return float(0.45 * sharp_norm + 0.35 * exp + 0.20 * fc)


def _structured_window_times(center: float, start: float, end: float) -> List[float]:
    if end <= start:
        return []
    dur = float(end - start)
    return sorted({
        round(start + (dur * 0.25), 4),
        round(start + (dur * 0.50), 4),
        round(start + (dur * 0.75), 4),
    })


def _template_is_wide(template: TemplateRecord, source_w: Optional[int] = None, source_h: Optional[int] = None) -> bool:
    if template.type.upper() == "WIDE":
        return True
    if template.n_faces >= 2:
        # Very wide layouts usually keep faces apart with modest face area.
        slots = sorted(template.face_slots, key=lambda s: s.center[0])
        if len(slots) >= 2:
            spread = abs(slots[-1].center_norm[0] - slots[0].center_norm[0])
            if spread >= WIDE_SHOT_FACE_FRACTION:
                return True
    if source_w and source_h:
        crop_w = template.crop_anchor.get("w", 0.0)
        if crop_w and float(crop_w) / max(1.0, float(source_w)) <= 0.60 and template.n_faces >= 2:
            return True
    return False


def _template_anchor_center(template: TemplateRecord, source_w: Optional[int] = None, source_h: Optional[int] = None) -> Tuple[float, float]:
    if template.crop_anchor:
        x = float(template.crop_anchor.get("x", 0.0))
        y = float(template.crop_anchor.get("y", 0.0))
        w = float(template.crop_anchor.get("w", source_w or 1.0))
        h = float(template.crop_anchor.get("h", source_h or 1.0))
        return x + w / 2.0, y + h / 2.0
    if template.face_slots:
        xs = [s.center[0] for s in template.face_slots]
        ys = [s.center[1] for s in template.face_slots]
        return float(np.median(xs)), float(np.median(ys))
    if source_w and source_h:
        return source_w / 2.0, source_h / 2.0
    return 0.0, 0.0


def _template_geometry_distance(template: TemplateRecord, detected: List[FaceObs], frame_w: int, frame_h: int) -> float:
    if not template.face_slots or not detected:
        return 1.0
    templ = sorted(template.face_slots, key=lambda s: s.center[0])
    obs = sorted(detected, key=lambda o: o.cx)
    n = min(len(templ), len(obs))
    if n == 0:
        return 1.0
    dists = []
    for i in range(n):
        tx, ty = templ[i].center_norm
        ox = obs[i].cx / max(1.0, float(frame_w))
        oy = obs[i].cy / max(1.0, float(frame_h))
        dists.append(math.hypot(tx - ox, ty - oy))
    return float(min(1.0, np.mean(dists) * 2.5))


def _template_face_count_distance(template: TemplateRecord, detected: List[FaceObs]) -> float:
    if not detected and template.n_faces <= 0:
        return 0.0
    exp = max(1, int(template.n_faces or len(template.face_slots) or 1))
    return float(min(1.0, abs(len(detected) - exp) / max(1, exp)))


def _template_phash_distance_frame(template: TemplateRecord, frame: np.ndarray) -> float:
    if frame is None:
        return 1.0
    try:
        h, _ = _phash(frame)
        tmpl = np.asarray(template.phash, dtype=np.uint8).ravel()
        if tmpl.size != h.size:
            return 1.0
        return float(_phash_distance(h, tmpl) / max(1, tmpl.size))
    except Exception:
        return 1.0


def _template_stage_a_score(template: TemplateRecord, frame: np.ndarray, detected: List[FaceObs]) -> float:
    if frame is None:
        return 0.0
    ph_dist = _template_phash_distance_frame(template, frame)
    geom_dist = _template_geometry_distance(template, detected, frame.shape[1], frame.shape[0])
    count_dist = _template_face_count_distance(template, detected)
    blur = clamp(sharpness_score(frame) / 800.0, 0.0, 1.0)
    exposure = _frame_exposure_score(frame)
    stability = 1.0 - min(1.0, 0.5 * geom_dist + 0.5 * count_dist)
    quality = _frame_quality_score(frame, len(detected))
    score = (
        0.30 * (1.0 - ph_dist)
        + 0.20 * (1.0 - geom_dist)
        + 0.15 * (1.0 - count_dist)
        + 0.15 * blur
        + 0.10 * exposure
        + 0.10 * stability
    )
    return float(clamp(score * (0.5 + 0.5 * quality), 0.0, 1.0))


def _template_stage_b_score(
    template: TemplateRecord,
    detected: List[FaceObs],
    speaker_db: Dict[str, List[float]],
    active_speaker: str,
) -> float:
    if not detected:
        return 0.0
    embedding_bonus = 0.0
    shirt_bonus = 0.0
    if active_speaker in speaker_db:
        proto = np.asarray(speaker_db[active_speaker], dtype=np.float32)
        sims = [cosine_sim(face.embedding, proto) for face in detected if face.embedding is not None]
        if sims:
            embedding_bonus = max(sims)
    slot_embeds = [np.asarray(s.embedding, dtype=np.float32) for s in template.face_slots if s.embedding is not None]
    if slot_embeds:
        per_face = []
        for face in detected:
            if face.embedding is None:
                continue
            per_face.append(max((cosine_sim(face.embedding, emb) for emb in slot_embeds), default=-1.0))
        if per_face:
            embedding_bonus = max(embedding_bonus, max(per_face))
    slot_shirts = [np.asarray(s.shirt_hist, dtype=np.float32) for s in template.face_slots if s.shirt_hist is not None]
    if slot_shirts:
        per_face = []
        for face in detected:
            if face.shirt_hist is None:
                continue
            per_face.append(max((1.0 - bhattacharyya_hist(face.shirt_hist, sh) for sh in slot_shirts), default=0.0))
        if per_face:
            shirt_bonus = max(per_face)
    slot_match = 0.0
    if template.face_slots:
        slot_match = 1.0 - _template_geometry_distance(template, detected, int(max(o.cx for o in detected) + 1), int(max(o.cy for o in detected) + 1))
    return float(clamp(0.50 * max(0.0, embedding_bonus) + 0.25 * max(0.0, shirt_bonus) + 0.25 * max(0.0, slot_match), 0.0, 1.0))


def _classify_template_window(
    video_path: str,
    cut_time: float,
    templates: Sequence[TemplateRecord],
    face_backend: BaseFaceBackend,
    speaker_db: Dict[str, List[float]],
    source_w: int,
    source_h: int,
    window_radius: float = TEMPLATE_WINDOW_RADIUS,
    start_bound: float = 0.0,
    end_bound: Optional[float] = None,
    logger: Optional[Logger] = None,
    active_speaker_hint: Optional[str] = None,
) -> Tuple[Optional[int], float, bool, Dict[int, float]]:
    """Return template_id, confidence, hidden-cut flag, and per-template scores."""
    if end_bound is None:
        end_bound = float("inf")
    times = _structured_window_times(cut_time, max(start_bound, cut_time - window_radius), min(end_bound, cut_time + window_radius))
    if not times:
        return None, 0.0, True, {}

    template_by_id = {t.template_id: t for t in templates}
    per_template: Dict[int, float] = {t.template_id: 0.0 for t in templates}
    votes: Dict[int, float] = {t.template_id: 0.0 for t in templates}
    stability_scores: Dict[int, List[float]] = {t.template_id: [] for t in templates}

    sampled: List[Tuple[float, Optional[np.ndarray], List[FaceObs]]] = []
    for t in times:
        frame = _analysis_frame_at(video_path, t)
        detected = face_backend.detect(frame) if frame is not None else []
        sampled.append((t, frame, detected))

    face_counts = [len(d) for _, _, d in sampled]
    face_counts_med = float(np.median(face_counts)) if face_counts else 0.0

    if logger and VERBOSE_STAGE_LOGGING:
        logger(f"[TEMPLATE] window@{cut_time:.3f} samples={len(times)} range={times[0]:.3f}->{times[-1]:.3f} faces_med={face_counts_med:.2f}")

    for tmpl in templates:
        sid = tmpl.template_id
        local_best = 0.0
        local_votes = 0.0
        for t, frame, detected in sampled:
            if frame is None:
                continue
            stage_a = _template_stage_a_score(tmpl, frame, detected)
            if stage_a <= 0.0:
                continue
            weight = 1.0 - min(1.0, abs(t - cut_time) / max(1e-6, window_radius))
            active = active_speaker_hint or "UNKNOWN"
            stage_b = _template_stage_b_score(tmpl, detected, speaker_db, active)
            combined = 0.72 * stage_a + 0.28 * stage_b
            votes[sid] += combined * max(0.15, weight)
            stability_scores[sid].append(combined)
            per_template[sid] = max(per_template[sid], combined)
            local_best = max(local_best, combined)
            local_votes += combined * max(0.15, weight)

        if logger and VERBOSE_STAGE_LOGGING and local_votes > 0:
            logger(f"[TEMPLATE] candidate {sid}: vote={local_votes:.3f} best={local_best:.3f}")

    if not votes:
        return None, 0.0, True, {}

    ranked = sorted(votes.items(), key=lambda kv: (kv[1], per_template.get(kv[0], 0.0)), reverse=True)
    best_id, best_vote = ranked[0]
    second_vote = ranked[1][1] if len(ranked) > 1 else 0.0
    confid = float(clamp(best_vote / max(1e-6, sum(votes.values())), 0.0, 1.0))
    stability = float(np.mean(stability_scores[best_id])) if stability_scores[best_id] else 0.0

    face_ratio = float(sum(1 for c in face_counts if c > 0) / max(1, len(face_counts)))
    low_face_card = face_ratio < 0.25 and face_counts_med <= 0.5
    hidden_cut = (best_vote - second_vote) < HIDDEN_CUT_CONFIDENCE_DROP or confid < 0.28 or stability < 0.25 or low_face_card

    if face_counts_med >= 1.0:
        confid = float(clamp(confid + 0.10, 0.0, 1.0))
    if logger:
        top = ", ".join(f"#{tid}:{votes[tid]:.3f}" for tid, _ in ranked[:3])
        logger(f"[TEMPLATE] cut@{cut_time:.3f} -> template {best_id} conf={confid:.3f} hidden={hidden_cut} top=[{top}]")
    return best_id, confid, hidden_cut, per_template


def _classification_cache_dir(job_dir: Path) -> Path:
    out = job_dir / CLASSIFICATION_CACHE_DIR_NAME
    ensure_dir(out)
    return out

def _classification_cache_signature(
    video_path: str,
    templates: Sequence[TemplateRecord],
    speaker_db: Dict[str, List[float]],
    cut_time: float,
    start_bound: float,
    end_bound: Optional[float],
    source_w: int,
    source_h: int,
    active_speaker_hint: Optional[str],
) -> str:
    try:
        vp = Path(video_path).resolve()
        stat = vp.stat()
        video_sig = {
            "path": str(vp),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except Exception:
        video_sig = {"path": str(video_path)}

    template_sig = [
        {
            "id": int(t.template_id),
            "name": t.display_name,
            "timestamp": round(float(t.canonical_timestamp), 4),
            "faces": int(t.n_faces),
            "slots": [
                {
                    "name": s.slot_name,
                    "speaker": s.speaker_label,
                    "bbox": [round(float(v), 5) for v in s.bbox_norm],
                }
                for s in t.face_slots
            ],
        }
        for t in templates
    ]
    speaker_sig = hashlib.sha1(
        json.dumps(speaker_db, sort_keys=True, default=str).encode("utf-8", errors="ignore")
    ).hexdigest()
    payload = {
        "version": CLASSIFICATION_CACHE_VERSION,
        "video": video_sig,
        "templates": template_sig,
        "speaker_db": speaker_sig,
        "cut_time": round(float(cut_time), 4),
        "start_bound": round(float(start_bound), 4),
        "end_bound": None if end_bound is None else round(float(end_bound), 4),
        "source_w": int(source_w),
        "source_h": int(source_h),
        "active_speaker_hint": active_speaker_hint or "UNKNOWN",
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8", errors="ignore")
    ).hexdigest()

def _classify_template_window_cached(
    *,
    job_dir: Optional[Path],
    cache_label: str,
    video_path: str,
    cut_time: float,
    templates: Sequence[TemplateRecord],
    face_backend: BaseFaceBackend,
    speaker_db: Dict[str, List[float]],
    source_w: int,
    source_h: int,
    start_bound: float = 0.0,
    end_bound: Optional[float] = None,
    logger: Optional[Logger] = None,
    active_speaker_hint: Optional[str] = None,
) -> Tuple[Optional[int], float, bool, Dict[int, float]]:
    cache_path: Optional[Path] = None
    if job_dir is not None:
        key = _classification_cache_signature(
            video_path=video_path,
            templates=templates,
            speaker_db=speaker_db,
            cut_time=cut_time,
            start_bound=start_bound,
            end_bound=end_bound,
            source_w=source_w,
            source_h=source_h,
            active_speaker_hint=active_speaker_hint,
        )
        cache_path = _classification_cache_dir(job_dir) / f"{key}.json"
        if cache_path.exists():
            try:
                data = read_json(cache_path)
                result = data.get("result", {})
                scores = {int(k): float(v) for k, v in result.get("scores", {}).items()}
                template_id = result.get("template_id")
                out_id = None if template_id is None else int(template_id)
                if logger and VERBOSE_STAGE_LOGGING:
                    logger(
                        f"[CLASSIFY-CACHE] hit {cache_label} cut={cut_time:.3f} "
                        f"template={out_id} conf={float(result.get('confidence', 0.0)):.3f}"
                    )
                return (
                    out_id,
                    float(result.get("confidence", 0.0)),
                    bool(result.get("hidden", True)),
                    scores,
                )
            except Exception:
                safe_remove(cache_path)

    result = _classify_template_window(
        video_path=video_path,
        cut_time=cut_time,
        templates=templates,
        face_backend=face_backend,
        speaker_db=speaker_db,
        source_w=source_w,
        source_h=source_h,
        start_bound=start_bound,
        end_bound=end_bound,
        logger=logger,
        active_speaker_hint=active_speaker_hint,
    )
    if cache_path is not None:
        template_id, confidence, hidden_cut, scores = result
        try:
            write_json(cache_path, {
                "label": cache_label,
                "cut_time": float(cut_time),
                "start_bound": float(start_bound),
                "end_bound": None if end_bound is None else float(end_bound),
                "template_ids": [int(t.template_id) for t in templates],
                "result": {
                    "template_id": None if template_id is None else int(template_id),
                    "confidence": float(confidence),
                    "hidden": bool(hidden_cut),
                    "scores": {str(k): float(v) for k, v in scores.items()},
                },
            })
            if logger and VERBOSE_STAGE_LOGGING:
                logger(f"[CLASSIFY-CACHE] saved {cache_label} -> {cache_path.name}")
        except Exception:
            pass
    return result


def _refine_scene_template_map(
    video_path: str,
    cuts: List[float],
    templates: List[TemplateRecord],
    face_backend: BaseFaceBackend,
    speaker_db: Dict[str, List[float]],
    source_w: int,
    source_h: int,
    logger: Logger,
) -> Dict[int, int]:
    refined: Dict[int, int] = {}
    if len(cuts) < 2 or not templates:
        return refined
    for i in range(len(cuts) - 1):
        mid = (cuts[i] + cuts[i + 1]) / 2.0
        template_id, confidence, hidden_cut, _ = _classify_template_window(
            video_path=video_path,
            cut_time=mid,
            templates=templates,
            face_backend=face_backend,
            speaker_db=speaker_db,
            source_w=source_w,
            source_h=source_h,
            start_bound=cuts[i],
            end_bound=cuts[i + 1],
            logger=logger,
        )
        if template_id is not None:
            refined[i] = template_id
        if hidden_cut and logger:
            logger(f"[CUT] possible hidden cut in scene {i} at {mid:.3f} (conf={confidence:.3f})")
    return refined


def _detect_hidden_cut_points(
    job_dir: Optional[Path],
    video_path: str,
    seg_start: float,
    seg_end: float,
    template: TemplateRecord,
    face_backend: BaseFaceBackend,
    speaker_db: Dict[str, List[float]],
    diar_timeline: Sequence[DiarSegment],
    source_w: int,
    source_h: int,
    logger: Logger,
) -> List[float]:
    """Split segments where template continuity breaks strongly."""
    if seg_end - seg_start < 4.0:
        return []

    probes = _structured_window_times((seg_start + seg_end) / 2.0, seg_start + 0.25, seg_end - 0.25)
    if len(probes) < 4:
        return []

    template_id, conf, hidden_cut, scores = _classify_template_window_cached(
        job_dir=job_dir,
        cache_label=f"hidden_probe_t{template.template_id}",
        video_path=video_path,
        cut_time=(seg_start + seg_end) / 2.0,
        templates=[template],
        face_backend=face_backend,
        speaker_db=speaker_db,
        source_w=source_w,
        source_h=source_h,
        start_bound=seg_start,
        end_bound=seg_end,
        logger=None,
        active_speaker_hint=active_speaker_at((seg_start + seg_end) / 2.0, diar_timeline),
    )
    if logger and VERBOSE_STAGE_LOGGING:
        logger(
            f"[CUT] hidden-cut probe seg={seg_start:.3f}->{seg_end:.3f} "
            f"template={template.template_id} conf={conf:.3f} hidden={hidden_cut}"
        )
    if not hidden_cut and conf >= 0.35:
        return []

    candidates: List[float] = []
    prev_tag: Optional[int] = None
    prev_t: Optional[float] = None

    for t in probes:
        frame = _analysis_frame_at(video_path, t)
        if frame is None:
            continue
        detected = face_backend.detect(frame)
        stage_a = _template_stage_a_score(template, frame, detected)
        stage_b = _template_stage_b_score(template, detected, speaker_db, active_speaker_at(t, diar_timeline))
        continuity = 0.7 * stage_a + 0.3 * stage_b
        tag = 1 if continuity >= 0.30 else 0

        if prev_tag is not None and tag != prev_tag and prev_t is not None:
            coarse = (prev_t + t) / 2.0
            refined = _refine_cut_boundary(video_path, coarse, search_radius=min(0.75, (t - prev_t)), logger=logger)
            if seg_start < refined < seg_end:
                candidates.append(refined)
                if logger and VERBOSE_STAGE_LOGGING:
                    logger(f"[CUT] probe t={t:.3f} continuity={continuity:.3f} tag={tag} refined={refined:.3f}")
        elif logger and VERBOSE_STAGE_LOGGING:
            logger(f"[CUT] probe t={t:.3f} continuity={continuity:.3f} tag={tag}")

        prev_tag = tag
        prev_t = t

    return sorted(set(round(c, 4) for c in candidates if seg_start < c < seg_end))

def _template_prune_duplicates(templates: List[TemplateRecord], logger: Optional[Logger] = None) -> List[TemplateRecord]:
    """Drop near-identical templates while preserving manual/canonical entries first."""
    if not templates:
        return templates
    kept: List[TemplateRecord] = []
    for t in sorted(templates, key=lambda x: (0 if x.canonical_frame else 1, x.template_id)):
        duplicate = False
        for k in kept:
            if _phash_distance(np.asarray(t.phash, dtype=np.uint8), np.asarray(k.phash, dtype=np.uint8)) <= 4:
                if abs(t.crop_anchor.get("x", 0.0) - k.crop_anchor.get("x", 0.0)) <= 24.0 and abs(t.crop_anchor.get("y", 0.0) - k.crop_anchor.get("y", 0.0)) <= 24.0:
                    duplicate = True
                    if logger:
                        logger(f"[TEMPLATE] pruning redundant template {t.template_id} (near {k.template_id})")
                    break
        if not duplicate:
            kept.append(t)
    # re-number only if ids collide; preserve existing ids for backward compatibility
    return kept


def _apply_template_registry_controls(
    templates: List[TemplateRecord],
    registry: Optional[Dict[str, Any]],
    logger: Optional[Logger] = None,
) -> List[TemplateRecord]:
    if not registry:
        return templates
    disabled_ids = set()
    raw_disabled = registry.get("disabled_template_ids", [])
    if isinstance(raw_disabled, list):
        for x in raw_disabled:
            try:
                disabled_ids.add(int(x))
            except Exception:
                continue
    manual = registry.get("manual_templates", [])
    merged: List[TemplateRecord] = []
    if isinstance(manual, list):
        for entry in manual:
            if not isinstance(entry, dict):
                continue
            try:
                merged.append(_deserialize_template_record(entry))
            except Exception:
                continue
    for t in templates:
        if t.template_id in disabled_ids:
            if logger:
                logger(f"[TEMPLATE] disabled template {t.template_id} skipped by registry")
            continue
        merged.append(t)
    return _template_prune_duplicates(merged, logger=logger)


def _export_debug_artifacts(
    job_dir: Path,
    templates: Sequence[TemplateRecord],
    plans: Sequence[SegmentPlan],
    speaker_db: Dict[str, List[float]],
    cuts: Sequence[float],
) -> None:
    ensure_dir(job_dir / "debug")
    try:
        write_json(job_dir / "debug" / "templates_debug.json", [_serialize_template(t) for t in templates])
        write_json(job_dir / "debug" / "crop_anchors.json", {str(t.template_id): t.crop_anchor for t in templates})
        write_json(job_dir / "debug" / "speaker_assignments.json", {p.index: {"template_id": p.template_id, "speaker": p.speaker, "target_slot": p.target_slot} for p in plans})
        write_json(job_dir / "debug" / "cut_confidence.json", {"cuts": [float(c) for c in cuts]})
        if speaker_db:
            write_json(job_dir / "debug" / "speaker_db.json", speaker_db)
    except Exception:
        pass



def _build_sendcmd_file(keyframes: List["KeyframePoint"], out_path: str, fps: float = 25.0) -> Optional[str]:
    """
    Build an FFmpeg sendcmd file with dense linear interpolation and rolling average smoothing.

    Zero-face frames (confidence ≤ 0.15) are excluded so boundary artefacts don't
    inject wrong positions into the animated path.

    Returns the file path on success, None if there are not enough valid keyframes
    to justify animation (static crop will be used instead).
    """
    valid_kf = sorted((k for k in keyframes if k.confidence > 0.15), key=lambda k: k.t_rel)
    if len(valid_kf) < 2:
        return None
    try:
        sample_step = 1.0 / max(1.0, float(fps))
        t_max = float(valid_kf[-1].t_rel)

        sample_times: List[float] = []
        t = 0.0
        while t < t_max - 1e-6:
            sample_times.append(round(t, 4))
            t += sample_step
        sample_times.append(round(t_max, 4))
        sample_times = sorted(set(sample_times))

        # Linear interpolation
        x_coords = []
        y_coords = []
        kf_idx = 0
        for t in sample_times:
            while kf_idx < len(valid_kf) - 1 and valid_kf[kf_idx + 1].t_rel < t - 1e-6:
                kf_idx += 1
            prev = valid_kf[kf_idx]
            if kf_idx >= len(valid_kf) - 1:
                x_coords.append(float(prev.crop_x))
                y_coords.append(float(prev.crop_y))
            else:
                cur = valid_kf[kf_idx + 1]
                t0 = float(prev.t_rel)
                t1 = float(cur.t_rel)
                if t1 <= t0 + 1e-6:
                    x_coords.append(float(cur.crop_x))
                    y_coords.append(float(cur.crop_y))
                else:
                    u = (t - t0) / (t1 - t0)
                    u = max(0.0, min(1.0, u))
                    x_coords.append(float(prev.crop_x + (cur.crop_x - prev.crop_x) * u))
                    y_coords.append(float(prev.crop_y + (cur.crop_y - prev.crop_y) * u))

        # Smooth with a rolling average filter to round off corners
        window = 7
        smoothed_x = []
        smoothed_y = []
        n = len(sample_times)
        half = window // 2
        for i in range(n):
            i_min = max(0, i - half)
            i_max = min(n - 1, i + half)
            smoothed_x.append(sum(x_coords[i_min:i_max + 1]) / (i_max - i_min + 1))
            smoothed_y.append(sum(y_coords[i_min:i_max + 1]) / (i_max - i_min + 1))

        lines: List[str] = []
        for t, x, y in zip(sample_times, smoothed_x, smoothed_y):
            lines.append(f"{t:.4f} crop x {x:.3f};")
            lines.append(f"{t:.4f} crop y {y:.3f};")

        # FACE_DEBUG_BOX: animate a drawbox (inserted before the crop by
        # _segment_filter) to the raw detected face bbox at each tick. Raw,
        # unsmoothed values on purpose — the box shows detector output, not the
        # EMA camera path. sendcmd holds the last value between commands, so
        # face-less (held) ticks keep the previous box, mirroring the crop hold.
        if bool(_pipeline_config_value("FACE_DEBUG_BOX", False)):
            for k in sorted(keyframes, key=lambda kk: kk.t_rel):
                fb = getattr(k, "face_box", None)
                if not fb:
                    continue
                bx, by = float(fb[0]), float(fb[1])
                bw = max(2.0, float(fb[2]) - float(fb[0]))
                bh = max(2.0, float(fb[3]) - float(fb[1]))
                t_cmd = max(0.0, float(k.t_rel))
                lines.append(f"{t_cmd:.4f} drawbox x {bx:.1f};")
                lines.append(f"{t_cmd:.4f} drawbox y {by:.1f};")
                lines.append(f"{t_cmd:.4f} drawbox w {bw:.1f};")
                lines.append(f"{t_cmd:.4f} drawbox h {bh:.1f};")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out_path
    except Exception:
        return None


def _segment_filter(
    crop_w: int,
    crop_h: int,
    expr_x: str,
    expr_y: str,
    center_fallback: bool = False,
    uncropped_fallback: bool = False,
    sendcmd_path: Optional[str] = None,
    debug_box: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """
    Build the -vf filter string for one segment render.

    Strategy: CPU decode (always) â†’ CPU crop/scale â†’ h264_nvenc encode.
    This is the correct and universally reliable pipeline for GTX 1650 Ti:
      - NVENC accepts plain YUV frames from CPU with zero extra steps.
      - hwupload_cuda / scale_cuda are NOT used â€” they require the decoder
        to already be in CUDA memory, which is NOT the case for CPU decode,
        causing "no packets" failures on short segments.

    When sendcmd_path is provided the crop x/y are animated per-keyframe via
    FFmpeg's sendcmd filter, giving a live tracking camera rather than a
    frozen static crop.  The expr_x/expr_y values are used as the initial
    crop position before the first sendcmd fires (at t=0 of the segment).
    """
    if uncropped_fallback:
        return (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2"
        )
    if center_fallback:
        cx, cy = _center_crop_expr()
        crop_part = f"crop={crop_w}:{crop_h}:{cx}:{cy}"
        return f"setpts=PTS-STARTPTS,{crop_part},scale={TARGET_W}:{TARGET_H}:flags=lanczos"

    # FACE_DEBUG_BOX overlay: drawn on the SOURCE frame before the crop so the
    # box sits exactly where the detector saw the face; sendcmd (when present)
    # re-targets its x/y/w/h per keyframe. debug_box seeds the pre-first-command
    # position, mirroring how expr_x/expr_y seed the crop.
    box_part = ""
    if debug_box is not None:
        bx, by, bw, bh = debug_box
        box_part = (
            f"drawbox=x={bx:.1f}:y={by:.1f}:w={bw:.1f}:h={bh:.1f}"
            f":color=lime@0.9:t=6,"
        )

    if sendcmd_path:
        # Animated crop: sendcmd updates x/y at each keyframe timestamp.
        # Forward slashes for Windows; escape FFmpeg filter metacharacters.
        # The drive colon in C:/... must be escaped or FFmpeg parses it as
        # another sendcmd option, causing "Invalid argument" and static fallback.
        safe = sendcmd_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        ex = _escape_ffmpeg_expr(expr_x)
        ey = _escape_ffmpeg_expr(expr_y)
        crop_part = f"crop={crop_w}:{crop_h}:{ex}:{ey}"
        return f"setpts=PTS-STARTPTS,sendcmd=f='{safe}',{box_part}{crop_part},scale={TARGET_W}:{TARGET_H}:flags=lanczos"

    # Static fallback
    ex = _escape_ffmpeg_expr(expr_x)
    ey = _escape_ffmpeg_expr(expr_y)
    crop_part = f"crop={crop_w}:{crop_h}:{ex}:{ey}"
    return f"setpts=PTS-STARTPTS,{box_part}{crop_part},scale={TARGET_W}:{TARGET_H}:flags=lanczos"


def _run_ffmpeg_cmd(
    cmd: List[str],
    logger: Logger,
    out_path: str,
) -> Tuple[bool, str]:
    """Execute one FFmpeg command, stream its output, return (success, last_err)."""
    progress_re = re.compile(
        r"frame=\s*(\d+)\s+fps=\s*([\d.]+).*?time=\s*([\d:.]+).*?speed=\s*([\d.]+)x"
    )
    last_log = [0.0]
    stderr_buf: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
        )
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break
            if line:
                stderr_buf.append(line)
                m = progress_re.search(line)
                if m:
                    now = time.monotonic()
                    if now - last_log[0] >= PROGRESS_LOG_INTERVAL:
                        logger(
                            f"[ENCODE] frame={m.group(1)} fps={m.group(2)} "
                            f"time={m.group(3)} speed={m.group(4)}x"
                        )
                        last_log[0] = now
        ret = proc.wait()
        if ret == 0 and os.path.exists(out_path) and os.path.getsize(out_path) >= MIN_CLIP_BYTES:
            return True, ""
        return False, "".join(stderr_buf[-30:]) or f"ffmpeg exit {ret}"
    except FileNotFoundError:
        return False, f"FFmpeg not found: {cmd[0]}"
    except Exception as exc:
        return False, str(exc)


def _ffmpeg_render_segment(
    video_path: str,
    plan: SegmentPlan,
    out_path: str,
    ffmpeg_bin: str,
    logger: Logger,
    accurate_seek: bool = False,
    center_fallback: bool = False,
    uncropped_fallback: bool = False,
) -> Tuple[bool, str]:
    """
    Render one segment to out_path.

    Attempt order (each tried in sequence until one succeeds):
      1. Animated crop (sendcmd) + fast seek   â€” live tracking per keyframe
      2. Static crop  (fast seek)              â€” if sendcmd not available / fails
      3. Center crop  (fast seek)              â€” if #1 and #2 fail
      4. Animated crop (accurate seek)         â€” slow seek retry
      5. Static crop  (accurate seek)
      6. Center crop  (accurate seek)
      7. Uncropped/padded (accurate seek)      â€” last resort

    All attempts use CPU decode â†’ CPU crop/scale â†’ h264_nvenc (NVENC on GTX 1650 Ti).
    """
    seg_dur = f"{(plan.end - plan.start - 0.020):.3f}"
    hwaccel = _pipeline_config_value("FFMPEG_HWACCEL", "cuda")
    if hwaccel:
        logger(f"[RENDER] segment#{plan.index + 1}: using GPU hardware decoding ({hwaccel}) for rendering")
    else:
        logger(f"[RENDER] segment#{plan.index + 1}: using CPU software decoding for rendering")

    # Build the sendcmd file for animated crop (attempt 1 / 4).
    # Route to the OS temp directory so apostrophes or other special characters
    # in the video/output path never break FFmpeg's filter-string parser.
    _scmd_name = f"clip_scmd_{plan.index:04d}_{abs(hash(out_path)) % 1_000_000:06d}.sendcmd"
    sendcmd_path = str(Path(tempfile.gettempdir()) / _scmd_name)
    _, _, fps = get_video_info(video_path)
    actual_sendcmd = _build_sendcmd_file(plan.keyframes, sendcmd_path, fps=fps)

    # FACE_DEBUG_BOX: seed drawbox with the first detected face bbox so the box
    # is correct before the first sendcmd command fires. Segments with zero
    # face keyframes render without an overlay.
    debug_box: Optional[Tuple[float, float, float, float]] = None
    if bool(_pipeline_config_value("FACE_DEBUG_BOX", False)):
        for _k in sorted(plan.keyframes, key=lambda kk: kk.t_rel):
            _fb = getattr(_k, "face_box", None)
            if _fb:
                debug_box = (
                    float(_fb[0]), float(_fb[1]),
                    max(2.0, float(_fb[2]) - float(_fb[0])),
                    max(2.0, float(_fb[3]) - float(_fb[1])),
                )
                break

    def _build_cmd(use_accurate: bool, fallback: str) -> List[str]:
        """fallback: 'animated' | 'crop' | 'center' | 'uncropped'"""
        use_scmd = actual_sendcmd is not None and fallback == "animated"
        vf = _segment_filter(
            plan.crop_w, plan.crop_h,
            plan.crop_expr_x, plan.crop_expr_y,
            center_fallback=(fallback == "center"),
            uncropped_fallback=(fallback == "uncropped"),
            sendcmd_path=actual_sendcmd if use_scmd else None,
            debug_box=debug_box if fallback in ("animated", "crop") else None,
        )
        if use_accurate:
            seek_args_pre: List[str] = []
            seek_args_post: List[str] = ["-ss", f"{plan.start:.3f}"]
        else:
            seek_args_pre = ["-ss", f"{plan.start:.3f}"]
            seek_args_post = []

        hw_args = ["-hwaccel", hwaccel] if hwaccel else []

        return (
            [ffmpeg_bin, "-y", "-nostdin"]
            + hw_args
            + seek_args_pre
            + ["-i", video_path]
            + seek_args_post
            + [
                "-t", seg_dur,
                "-vf", vf,
                "-pix_fmt", "yuv420p",
                "-c:v", "h264_nvenc",
                "-preset", str(_pipeline_config_value("NVENC_PRESET", "p4")),
                "-cq", "23",
                "-rc", "vbr",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                out_path,
            ]
        )

    try:
        # --- Attempt 1: fast seek, animated crop (sendcmd) ---
        if actual_sendcmd is not None:
            cmd = _build_cmd(use_accurate=False, fallback="animated")
            logger(f"[FFMPEG] {cmd}")
            ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
            if ok:
                logger(f"[FFMPEG] seg#{plan.index + 1}: animated-crop fast-seek OK")
                return True, ""
            safe_remove(out_path)
            logger(f"[RETRY] seg#{plan.index + 1}: animated-crop fast-seek failed, trying accurate animated :: {err[-240:]}")

            # --- Attempt 2: accurate seek, animated crop ---
            cmd = _build_cmd(use_accurate=True, fallback="animated")
            logger(f"[FFMPEG] {cmd}")
            ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
            if ok:
                logger(f"[FFMPEG] seg#{plan.index + 1}: animated-crop accurate-seek OK")
                return True, ""
            safe_remove(out_path)
            logger(
                f"[ERROR] seg#{plan.index + 1}: animated crop failed; refusing static tracked fallback "
                f"because it would freeze the camera :: {err[-240:]}"
            )
            return False, err

        # --- Attempt 2: fast seek, static crop (only when no animation exists) ---
        cmd = _build_cmd(use_accurate=False, fallback="crop")
        logger(f"[FFMPEG] {cmd}")
        ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
        if ok:
            return True, ""
        safe_remove(out_path)

        # --- Attempt 3: fast seek, center crop ---
        logger(f"[RETRY] seg#{plan.index + 1}: fast-seek center-crop :: {err[-180:]}")
        cmd = _build_cmd(use_accurate=False, fallback="center")
        logger(f"[FFMPEG] {cmd}")
        ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
        if ok:
            return True, ""
        safe_remove(out_path)

        # --- Attempt 5: accurate seek, static crop ---
        logger(f"[RETRY] seg#{plan.index + 1}: accurate-seek static-crop :: {err[-180:]}")
        cmd = _build_cmd(use_accurate=True, fallback="crop")
        logger(f"[FFMPEG] {cmd}")
        ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
        if ok:
            return True, ""
        safe_remove(out_path)

        # --- Attempt 6: accurate seek, center crop ---
        logger(f"[RETRY] seg#{plan.index + 1}: accurate-seek center-crop :: {err[-180:]}")
        cmd = _build_cmd(use_accurate=True, fallback="center")
        logger(f"[FFMPEG] {cmd}")
        ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
        if ok:
            return True, ""
        safe_remove(out_path)

        # --- Attempt 7: accurate seek, uncropped/padded â€” last resort ---
        logger(f"[RETRY] seg#{plan.index + 1}: accurate-seek uncropped-pad :: {err[-180:]}")
        cmd = _build_cmd(use_accurate=True, fallback="uncropped")
        logger(f"[FFMPEG] {cmd}")
        ok, err = _run_ffmpeg_cmd(cmd, logger, out_path)
        if ok:
            return True, ""
        safe_remove(out_path)

        return False, err

    finally:
        # Always clean up the sendcmd temp file
        safe_remove(sendcmd_path)

def _ffmpeg_hardcut_concat(
    clips: Sequence[str],
    out_path: str,
    ffmpeg_bin: str,
    fps: float,
) -> Tuple[bool, str]:
    """Overlap-free concat of a clip's rendered segments.

    Unlike xfade, this preserves additive duration: output length ==
    sum(segment lengths). That keeps burned captions in sync — the caption
    timeline is computed as a cumulative sum of segment durations
    (_get_clip_local_words), and xfade's per-join overlap (5/fps) shifted every
    subsequent word earlier on multi-segment clips. Re-encodes through the
    concat filter so segments with differing params still join cleanly.
    """
    if not clips:
        return False, "No clips to concat"
    if len(clips) == 1:
        try:
            shutil.copy2(clips[0], out_path)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    cmd = [ffmpeg_bin, "-y", "-nostdin"]
    for c in clips:
        cmd += ["-i", c]
    n = len(clips)
    pairs = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    filter_complex = f"{pairs}concat=n={n}:v=1:a=1[v][a]"
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc",
        "-preset", str(_pipeline_config_value("NVENC_PRESET", "p4")),
        "-cq", "23",
        "-rc", "vbr",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
    )
    return (res.returncode == 0, "" if res.returncode == 0 else res.stderr)


def _ffmpeg_xfade_concat(
    clips: Sequence[str],
    out_path: str,
    ffmpeg_bin: str,
    fps: float,
) -> Tuple[bool, str]:
    if not clips:
        return False, "No clips to concat"
    if len(clips) == 1:
        try:
            shutil.copy2(clips[0], out_path)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _probe_duration(path: str) -> float:
        ffprobe_bin = str(Path(ffmpeg_bin).with_name("ffprobe.exe" if is_windows() else "ffprobe"))
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
        )
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or "ffprobe failed")
        return float(res.stdout.strip())

    durations = [_probe_duration(c) for c in clips]
    _xfade_frames = max(1, int(getattr(config, "XFADE_FRAMES", 5)))
    xfade_dur = _xfade_frames / max(1.0, float(fps))

    cmd = [ffmpeg_bin, "-y", "-nostdin"]
    for c in clips:
        cmd += ["-i", c]

    filter_parts: List[str] = []
    cumulative = durations[0]
    for idx in range(1, len(clips)):
        offset = max(0.0, cumulative - xfade_dur)
        prev_v = f"v{idx - 1}" if idx > 1 else "0:v"
        prev_a = f"a{idx - 1}" if idx > 1 else "0:a"
        filter_parts.append(
            f"[{prev_v}][{idx}:v]xfade=transition=fade:duration={xfade_dur:.6f}:offset={offset:.6f}[v{idx}]"
        )
        filter_parts.append(
            f"[{prev_a}][{idx}:a]acrossfade=d={xfade_dur:.6f}[a{idx}]"
        )
        cumulative += durations[idx] - xfade_dur

    last_v = f"v{len(clips) - 1}"
    last_a = f"a{len(clips) - 1}"
    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{last_v}]",
        "-map", f"[{last_a}]",
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc",
        "-preset", str(_pipeline_config_value("NVENC_PRESET", "p4")),
        "-cq", "23",
        "-rc", "vbr",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
    )
    return (res.returncode == 0, "" if res.returncode == 0 else res.stderr)

# ======================================================================
# Diarization subprocess wrapper
# ======================================================================

def _extract_audio_mono(video_path: str, wav_path: str, ffmpeg_bin: str) -> bool:
    cmd = [
        ffmpeg_bin, "-y", "-nostdin",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", wav_path,
    ]
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
        )
        return res.returncode == 0 and os.path.exists(wav_path)
    except Exception:
        return False

def _extract_audio_mono_range(video_path: str, wav_path: str, ffmpeg_bin: str, start_sec: float, end_sec: float) -> bool:
    dur = max(0.0, float(end_sec) - float(start_sec))
    if dur <= 0.05:
        return False
    cmd = [
        ffmpeg_bin, "-y", "-nostdin",
        "-ss", f"{float(start_sec):.3f}",
        "-t", f"{dur:.3f}",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", wav_path,
    ]
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
        )
        return res.returncode == 0 and os.path.exists(wav_path)
    except Exception:
        return False

def _run_diarization_subprocess(video_path: str, job_dir: Path, ffmpeg_bin: str, diar_py: str, logger: Logger) -> List[DiarSegment]:
    diar_json = job_dir / DIAR_FILE
    if diar_json.exists():
        try:
            cached = load_diarization(str(diar_json))
            if cached:
                logger(f"[DIAR] cache hit -> {diar_json}")
                return cached
        except Exception:
            pass

    ensure_dir(job_dir)
    wav_path = job_dir / "audio_mono.wav"
    if not _extract_audio_mono(video_path, str(wav_path), ffmpeg_bin):
        logger("[DIAR] audio extraction failed")
        return []

    # If venv1 is missing, try local import fallback.
    if not Path(diar_py).exists():
        logger(f"[DIAR] diarization python not found: {diar_py}")
        return []

    helper_script = DEFAULT_BASE_DIR / "pipeline" / "diarize_helper.py"
    if not helper_script.exists():
        # allow same-folder helper if present
        helper_script = Path(__file__).resolve().parent / "pipeline" / "diarize_helper.py"

    out_path = job_dir / "diarization_raw.json"
    token = _ensure_hf_token_env(logger)
    if not token:
        logger("[DIAR] HF_TOKEN/HUGGINGFACE_TOKEN is missing; diarization disabled")
        return []
    cmd = [
        diar_py, str(helper_script),
        "--audio", str(wav_path),
        "--token", token,
        "--output", str(out_path),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=7200,
            creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0,
        )
        if res.returncode != 0:
            logger(f"[DIAR] failed: {res.stderr[-400:]}")
            return []
        timeline = load_diarization(str(out_path))
        if timeline:
            write_json(diar_json, [asdict(x) for x in timeline])
            logger(f"[DIAR] saved -> {diar_json}")
            return timeline
    except Exception as e:
        logger(f"[DIAR] exception: {e}")
    return []



def _serialize_keyframes(keyframes: Sequence[KeyframePoint]) -> List[Dict[str, Any]]:
    """Persist crop tracking samples in a JSON-friendly form for debugging."""
    out: List[Dict[str, Any]] = []
    for k in keyframes:
        if not isinstance(k, KeyframePoint):
            continue
        out.append({
            "t_rel": float(k.t_rel),
            "crop_x": float(k.crop_x),
            "crop_y": float(k.crop_y),
            "is_snap": bool(k.is_snap),
            "confidence": float(k.confidence),
            "face_box": [float(v) for v in k.face_box] if getattr(k, "face_box", None) else None,
        })
    return out


def _apply_stable_start_anchor(
    keyframes: List[KeyframePoint],
    seg_index: int,
    logger: Optional[Logger],
    offset_sec: float = STABLE_START_OFFSET_SEC,
) -> Optional[Tuple[float, float, float]]:
    """
    Prevent boundary-frame flashes by using the first confident crop at/after a
    small offset as the rendered t=0 crop.
    """
    if not keyframes:
        return None
    valid = sorted((k for k in keyframes if k.confidence > 0.15), key=lambda k: k.t_rel)
    if not valid:
        return None

    threshold = max(0.0, float(offset_sec))
    stable = next((k for k in valid if k.t_rel >= threshold - 1e-4), valid[0])
    stable_t = max(0.0, float(stable.t_rel))
    stable_x = float(stable.crop_x)
    stable_y = float(stable.crop_y)
    stable_conf = float(stable.confidence)

    has_t0 = False
    for k in keyframes:
        if k.t_rel <= stable_t + 1e-4:
            k.crop_x = stable_x
            k.crop_y = stable_y
            k.is_snap = False
        if abs(k.t_rel) <= 1e-4:
            has_t0 = True
            k.t_rel = 0.0

    if not has_t0:
        keyframes.append(KeyframePoint(
            t_rel=0.0,
            crop_x=stable_x,
            crop_y=stable_y,
            is_snap=False,
            confidence=stable_conf,
        ))
    keyframes.sort(key=lambda k: k.t_rel)

    if logger and VERBOSE_STAGE_LOGGING and stable_t > 1e-4:
        logger(
            f"[START] seg#{seg_index + 1} stable initial crop from +{stable_t:.3f}s "
            f"-> ({stable_x:.1f},{stable_y:.1f}) conf={stable_conf:.3f}"
        )
    return stable_t, stable_x, stable_y


def _select_focus_face(
    detected: Sequence[FaceObs],
    template: TemplateRecord,
    speaker_db: Dict[str, List[float]],
    active_speaker: str,
) -> Optional[FaceObs]:
    """Pick the face that drives the wide-shot crop for this sample.

    Identity routing depends on what the detector provides:
      • Embedding-capable backends (InsightFace): use active-speaker /
        template-slot embedding similarity as the primary signal.
      • Position-only backends (RT-DETR): the active template slot's manual
        bbox is the IDENTITY GROUND TRUTH. Pick the detection whose center
        is nearest to that slot's center — diarization decides which slot is
        active, position decides which person is the slot.

    Both paths keep confidence + size + shirt cues as secondary tie-breakers.
    """
    valid = [f for f in detected if f is not None]
    if not valid:
        return None

    # ── Position-based pick (RT-DETR / any embeddingless backend) ───────────
    embeddings_available = any(f.embedding is not None for f in valid)
    if not embeddings_available and template.face_slots and active_speaker:
        active_slot = next(
            (s for s in template.face_slots if s.speaker_label == active_speaker),
            None,
        )
        if active_slot is not None:
            sx, sy = float(active_slot.center[0]), float(active_slot.center[1])
            slot_w = max(1.0, float(active_slot.bbox[2] - active_slot.bbox[0]))
            slot_h = max(1.0, float(active_slot.bbox[3] - active_slot.bbox[1]))
            slot_diag = (slot_w * slot_w + slot_h * slot_h) ** 0.5
            best_face = None
            best_score = -1e9
            best_norm = 1e18
            for face in valid:
                dx = float(face.cx) - sx
                dy = float(face.cy) - sy
                pixel_dist = (dx * dx + dy * dy) ** 0.5
                # Normalize by slot diagonal so we're robust to source res.
                norm_dist = pixel_dist / max(1.0, slot_diag)
                # Strongly prefer proximity to the slot; light tie-breakers
                # on confidence and area.
                score = -norm_dist * 2.0 + 0.5 * float(face.conf) + 0.0001 * float(face.area)
                if score > best_score:
                    best_score = score
                    best_face = face
                    best_norm = norm_dist
            # Distance sanity cap for wide shots with >2 people: if even the
            # nearest person is far outside the slot window, trust the manual
            # slot anchor (the identity ground truth) rather than locking onto
            # an unrelated guest. Diarization picked the slot; this just refuses
            # a bad position match.
            if best_face is not None and best_norm > WIDE_SLOT_MATCH_MAX:
                return None
            return best_face

    # ── Embedding-based pick (InsightFace, etc.) ────────────────────────────
    active_proto = None
    if active_speaker and active_speaker in speaker_db:
        try:
            active_proto = np.asarray(speaker_db[active_speaker], dtype=np.float32)
        except Exception:
            active_proto = None

    template_embeds = [np.asarray(s.embedding, dtype=np.float32) for s in template.face_slots if s.embedding is not None]
    template_shirts = [np.asarray(s.shirt_hist, dtype=np.float32) for s in template.face_slots if s.shirt_hist is not None]

    best_face = None
    best_score = -1e9
    for face in valid:
        score = float(face.conf) * 0.65 + clamp(face.area / max(1.0, float(face.area + 1.0)), 0.0, 1.0) * 0.35

        if active_proto is not None and face.embedding is not None:
            try:
                score += 1.00 * cosine_sim(face.embedding, active_proto)
            except Exception:
                pass

        if template_embeds and face.embedding is not None:
            try:
                score += 0.25 * max((cosine_sim(face.embedding, emb) for emb in template_embeds), default=0.0)
            except Exception:
                pass

        if template_shirts and face.shirt_hist is not None:
            try:
                score += 0.15 * max((1.0 - bhattacharyya_hist(face.shirt_hist, sh) for sh in template_shirts), default=0.0)
            except Exception:
                pass

        if active_speaker and template.face_slots and face.embedding is not None:
            label_bonus = 0.0
            for slot in template.face_slots:
                if slot.speaker_label == active_speaker and slot.embedding is not None:
                    try:
                        label_bonus = max(label_bonus, 0.65 * cosine_sim(face.embedding, np.asarray(slot.embedding, dtype=np.float32)))
                    except Exception:
                        continue
            score += label_bonus

        if score > best_score:
            best_score = score
            best_face = face

    return best_face

def _select_solo_focus_face(
    detected: Sequence[FaceObs],
    source_w: Optional[int] = None,
    source_h: Optional[int] = None,
) -> Optional[FaceObs]:
    """Pick the dominant subject for a solo shot — pure face-driven, no template.

    Selection is centrality + head size: prefer the largest, most central head
    among confident detections, with confidence as a light tie-breaker. Unlike
    the old "exactly one face or nothing" rule, this NEVER returns None when any
    confident face exists, so the crop locks onto the subject from the first
    detected frame and stays there even when RT-DETR over-detects a background
    person. The min-area gate in detection already drops tiny far-away persons.
    """
    valid = [f for f in detected if f is not None and f.conf >= FACE_MIN_SCORE]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    if source_w and source_h:
        fcx, fcy = float(source_w) / 2.0, float(source_h) / 2.0
        diag = max(1.0, (float(source_w) ** 2 + float(source_h) ** 2) ** 0.5)
    else:
        xs = [f.cx for f in valid]
        ys = [f.cy for f in valid]
        fcx, fcy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        diag = max(1.0, ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5)

    max_area = max((f.area for f in valid), default=1.0) or 1.0
    best = None
    best_score = -1e18
    for f in valid:
        dx = f.cx - fcx
        dy = f.cy - fcy
        dist = ((dx * dx + dy * dy) ** 0.5) / diag
        size = f.area / max_area
        # Centrality dominates, head size next, confidence as a light nudge.
        score = (-dist * 1.5) + (size * 1.0) + (float(f.conf) * 0.25)
        if score > best_score:
            best_score = score
            best = f
    return best



def _choose_target_slot(
    template: TemplateRecord,
    detected: Sequence[FaceObs],
    speaker_db: Dict[str, List[float]],
    active_speaker: str,
) -> Optional[str]:
    """
    Pick a stable slot name for diagnostics and multi-person continuity.
    """
    if not template.face_slots:
        return None

    slots = sorted(template.face_slots, key=lambda s: s.center_norm[0])
    if len(slots) == 1:
        return slots[0].slot_name

    active_proto = None
    if active_speaker and active_speaker in speaker_db:
        try:
            active_proto = np.asarray(speaker_db[active_speaker], dtype=np.float32)
        except Exception:
            active_proto = None

    wide = _template_is_wide(template)
    best_slot = None
    best_score = -1e9

    for slot in slots:
        score = float(slot.confidence)

        if active_speaker and slot.speaker_label == active_speaker:
            score += 2.0
        elif active_speaker and slot.speaker_label and slot.speaker_label != active_speaker:
            score -= 0.2

        # No center pull for wide shots. Use the pre-calculated geometry and labels only.
        if wide and template.slot_geometry:
            geom = template.slot_geometry.get(slot.slot_name)
            if geom:
                span = abs(float(geom.get("x2", 0.0)) - float(geom.get("x1", 0.0)))
                score += 0.01 * span

        if active_proto is not None and slot.embedding is not None:
            try:
                score += 0.90 * cosine_sim(np.asarray(slot.embedding, dtype=np.float32), active_proto)
            except Exception:
                pass

        if detected and slot.embedding is not None:
            try:
                slot_emb = np.asarray(slot.embedding, dtype=np.float32)
                face_scores = [cosine_sim(face.embedding, slot_emb) for face in detected if face.embedding is not None]
                if face_scores:
                    score += 0.25 * max(face_scores)
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_slot = slot.slot_name

    return best_slot



def plan_segment(
    video_path: str,
    seg_index: int,
    seg_start: float,
    seg_end: float,
    template: TemplateRecord,
    speaker_db: Dict[str, List[float]],
    diar_timeline: Sequence[DiarSegment],
    face_backend: BaseFaceBackend,
    source_w: int,
    source_h: int,
    logger: Logger,
) -> SegmentPlan:
    """
    Build a stable crop plan for one output segment.
    Wide shots: target_slot is re-evaluated per 0.25 s tick from diarization.
    Close-ups: warm-started from a pre-segment frame to avoid cold-start flash.
    """
    _t0_plan = time.time()
    crop_w, crop_h = _canonical_crop_dims(source_w, source_h)
    wide = _template_is_wide(template, source_w, source_h)
    mid = (seg_start + seg_end) / 2.0

    diar_mid_speaker = active_speaker_at(mid, diar_timeline)
    if wide:
        active_speaker = diar_mid_speaker
        if active_speaker == "UNKNOWN" and template.speaker_votes:
            try:
                active_speaker = max(template.speaker_votes.items(), key=lambda kv: kv[1])[0]
            except Exception:
                active_speaker = "UNKNOWN"
        route_mode = "wide-diar-slot"
    else:
        active_speaker = _template_primary_speaker_label(template) or "SOLO_FACE"
        route_mode = "solo-face"
        if logger and VERBOSE_STAGE_LOGGING and diar_mid_speaker != "UNKNOWN":
            logger(
                f"[SPEAKER-ROUTE] seg#{seg_index + 1} solo shot ignores diar_mid={diar_mid_speaker}; "
                f"tracking=InsightFace-only"
            )

    # For solo shots, target_slot is diagnostic only. For wide shots, the slot
    # is re-evaluated per tick from mapped diarization.
    target_slot = _choose_target_slot(template, [], speaker_db, active_speaker)

    base_x = float(template.crop_anchor.get("x", max(0.0, (source_w - crop_w) / 2.0)))
    base_y = float(template.crop_anchor.get("y", 0.0))

    if logger and VERBOSE_STAGE_LOGGING:
        logger(
            f"[PLAN] seg#{seg_index + 1} {seg_start:.3f}->{seg_end:.3f} template={template.template_id} "
            f"mode={route_mode} speaker={active_speaker} wide={wide} "
            f"target_slot={target_slot} base=({base_x:.1f},{base_y:.1f})"
        )

    # ------------------------------------------------------------------ #
    # FIX 3  â€” Â±2 s chunk cache: decode the segment once into RAM;       #
    #          no more random per-frame disk seeks.                        #
    # FIX 4  â€” Bounded queue (maxsize=1): CPU reader blocks until GPU     #
    #          finishes its tracking pass and pops the slot.              #
    # FIX 6  â€” ROI tracking: InsightFace receives only the slot window    #
    #          cropped to slot dimensions â€” blind to the other speaker.   #
    #          No downscale, full pixel quality for side profiles.        #
    # FIX 7  â€” Cadence lock: tick every 7 frames (~0.25 s at 30 fps).    #
    #          Wide-shot init override: first check at Frame 4 (~0.1 s). #
    # FIX 8  â€” Template seed: median_crop removed. Template pixel coords  #
    #          seed motion only; detected faces drive the live crop.      #
    # PRIMARY â€” [TRACK] runs exclusively on the GTX 1650 Ti CUDA via     #
    #           InsightFace CUDA provider (buffalo_l, no CPU fallback).  #
    # ------------------------------------------------------------------ #

    import queue as _queue_mod
    import threading as _threading_mod

    fps_local = max(1.0, get_video_info(video_path)[2])
    original_cut_time = float(seg_start)

    # ---- 4K decode-downscale for analysis ----
    # The person detector + pose run on a downscaled copy (longest side capped at
    # RTDETR_ANALYSIS_MAX_SIDE); detections are scaled back to source pixels. This
    # cuts the dominant 4K detection cost without changing any downstream geometry.
    _analysis_max = int(getattr(config, "RTDETR_ANALYSIS_MAX_SIDE", 0) or 0)
    _src_long = max(int(source_w), int(source_h))
    _analysis_scale = 1.0
    if _analysis_max > 0 and _src_long > _analysis_max:
        _analysis_scale = _analysis_max / float(_src_long)
    _analysis_inv = (1.0 / _analysis_scale) if _analysis_scale > 0 else 1.0

    def _scale_face(f: FaceObs, s: float) -> FaceObs:
        if s == 1.0:
            return f
        lm = f.landmarks
        if lm is not None:
            try:
                lm = np.asarray(lm, dtype=np.float32) * s
            except Exception:
                lm = f.landmarks
        return FaceObs(
            x1=f.x1 * s, y1=f.y1 * s, x2=f.x2 * s, y2=f.y2 * s,
            conf=f.conf, cx=f.cx * s, cy=f.cy * s, area=f.area * s * s,
            yaw=f.yaw, embedding=f.embedding, landmarks=lm,
            shirt_hist=f.shirt_hist, shirt_rgb=f.shirt_rgb, torso_conf=f.torso_conf,
        )

    def _downscale(img: np.ndarray) -> np.ndarray:
        nh = max(1, int(round(img.shape[0] * _analysis_scale)))
        nw = max(1, int(round(img.shape[1] * _analysis_scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _detect_scaled(img: np.ndarray) -> List[FaceObs]:
        if _analysis_scale >= 1.0:
            return face_backend.detect(img)
        return [_scale_face(f, _analysis_inv) for f in face_backend.detect(_downscale(img))]

    def _detect_batch_scaled(imgs: List[np.ndarray]) -> List[List[FaceObs]]:
        if not imgs:
            return []
        use = [_downscale(im) for im in imgs] if _analysis_scale < 1.0 else imgs
        if hasattr(face_backend, "detect_batch"):
            try:
                raw = face_backend.detect_batch(use)
            except Exception as e:
                logger(f"[DETECT] batch failed ({e}); per-frame fallback")
                raw = [face_backend.detect(im) for im in use]
        else:
            raw = [face_backend.detect(im) for im in use]
        if _analysis_scale < 1.0:
            return [[_scale_face(f, _analysis_inv) for f in faces] for faces in raw]
        return raw

    # Ensure cadence-locked tracking happens exactly every TRACK_SAMPLE_INTERVAL after the surgical offset period
    _frames_per_tick = max(1, round(TRACK_SAMPLE_INTERVAL * fps_local))
    frame_interval = _frames_per_tick / fps_local

    tick_offsets = (-3, -2, -1, 0, 1, 2, 3, 4, 6, 9)
    tick_times_raw: List[float] = [
        round(original_cut_time + (off / fps_local), 4)
        for off in tick_offsets
        if seg_start <= original_cut_time + (off / fps_local) <= seg_end
    ]
    
    # Fill in the rest of the segment with the standard 0.25s cadence
    t_cur = tick_times_raw[-1] + frame_interval if tick_times_raw else seg_start
    while t_cur <= seg_end + 1e-6:
        tick_times_raw.append(round(t_cur, 4))
        t_cur += frame_interval
        
    last_val = round(seg_end, 4)
    if not tick_times_raw or abs(tick_times_raw[-1] - last_val) > 1e-4:
        tick_times_raw.append(last_val)
    tick_times = sorted(set(tick_times_raw))
    total_ticks = len(tick_times)

    # ---- FIX 3: one-shot linear RAM cache for the full segment ----
    _frame_cache: Dict[float, Optional[np.ndarray]] = {}
    # Decode-progress signals so the solo GPU pre-pass below can consume tick
    # frames WHILE this linear pass is still decoding (frames arrive in
    # presentation order, so progress is monotone).
    _decode_done = _threading_mod.Event()
    _decode_progress = [-1.0e9]   # newest decoded source-time in seconds

    def _load_segment_into_ram() -> None:
        if not tick_times:
            _decode_done.set()
            return
        t_lo = max(0.0, tick_times[0] - 0.10)
        t_hi = tick_times[-1] + 0.10
        half_fi = 1.0 / fps_local

        if not HAS_AV:
            logger("[DECODE] CPU PyAV fallback requested but PyAV is not installed")
            _decode_done.set()
            return

        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            seek_ts = int(t_lo / float(stream.time_base))
            container.seek(seek_ts, stream=stream, backward=True)
            for av_frame in container.decode(video=0):
                ft = float(av_frame.pts * float(stream.time_base))
                _decode_progress[0] = ft
                if ft < t_lo - 0.02:
                    continue
                if ft > t_hi + 0.02:
                    break
                frame = av_frame.to_ndarray(format="bgr24")
                for rq_t in tick_times:
                    if rq_t not in _frame_cache and abs(rq_t - ft) <= half_fi:
                        _frame_cache[rq_t] = frame
                        break
            logger(f"[DECODE] segment#{seg_index + 1}: CPU PyAV decode successful ({len(_frame_cache)}/{len(tick_times)} frames cached)")
        except Exception as e:
            logger(f"[DECODE] CPU PyAV decode failed: {e}")
        finally:
            _decode_done.set()
            if container is not None:
                container.close()

    # ---- Two-phase solo detection: saturate the GPU up front ----
    # For close-ups the detector does not depend on tracking state, so every tick
    # frame is batch-detected here (phase 1, GPU-bound) and the tracking loop below
    # (phase 2, CPU-bound) just reads the result. No GPU idling between ticks.
    # Wide shots stay per-tick because detection is ROI-gated by the live slot.
    #
    # The linear PyAV decode runs on a thread and batches are dispatched to the
    # GPU as soon as their tick frames are cached (decode is time-ordered and
    # tick_times is sorted), so CPU decode and GPU inference overlap instead of
    # running back-to-back. Same frames, same batch composition, same results.
    det_map: Dict[float, List[FaceObs]] = {}
    if not wide:
        _dec_th = _threading_mod.Thread(target=_load_segment_into_ram, daemon=True)
        _dec_th.start()
        _batch_n = max(1, int(getattr(config, "RTDETR_BATCH_SIZE", 16)))
        _half_fi = 1.0 / fps_local
        _buf_ts: List[float] = []
        _buf_imgs: List[np.ndarray] = []

        def _flush_det_batch() -> None:
            if not _buf_ts:
                return
            for _tt, _faces in zip(_buf_ts, _detect_batch_scaled(_buf_imgs)):
                det_map[_tt] = _faces
            _buf_ts.clear()
            _buf_imgs.clear()

        for tt in tick_times:
            # Wait until the decoder has passed this tick's match window (or
            # finished); a tick the decoder skipped stays None, as before.
            while (tt not in _frame_cache
                   and not _decode_done.is_set()
                   and _decode_progress[0] < tt + _half_fi + 0.05):
                time.sleep(0.004)
            img = _frame_cache.get(tt)
            if img is None:
                continue
            _buf_ts.append(tt)
            _buf_imgs.append(img)
            if len(_buf_ts) >= _batch_n:
                _flush_det_batch()
        _flush_det_batch()
        _dec_th.join()
        if logger and VERBOSE_STAGE_LOGGING:
            logger(f"[DETECT] seg#{seg_index + 1}: solo batch pre-pass {len(det_map)}/{len(tick_times)} ticks (scale={_analysis_scale:.3f}, decode-overlapped)")
    else:
        _load_segment_into_ram()   # wide: single linear pass into RAM (per-tick ROI detection follows)

    def _get_cached_frame(t: float) -> Optional[np.ndarray]:
        return _frame_cache.get(t)

    # ---- FIX 4: bounded queue gate â€” maxsize=1 ----
    _frame_q: _queue_mod.Queue = _queue_mod.Queue(maxsize=1)
    _SENTINEL = object()

    def _producer() -> None:
        for rq_t in tick_times:
            _frame_q.put((rq_t, _get_cached_frame(rq_t)), block=True)
        _frame_q.put((_SENTINEL, None), block=True)

    _prod_thread = _threading_mod.Thread(target=_producer, daemon=True)

    # â”€â”€ Template-seeded initialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Seed last_x/last_y from the template's own geometry rather than from
    # a warmup frame in the *previous* segment.  The old warmup approach
    # inherited the wrong speaker's face position and caused a visible pan
    # (or snap-flash) at the very first tick of every new segment.
    #
    # Close-up  â†’ base_x/base_y  (crop_anchor computed from slot centre)
    # Wide shot â†’ slot geometry anchor for the committed speaker's slot
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if wide and target_slot:
        _af = _template_slot_to_face(template, target_slot, source_w, source_h)
        if _af is not None:
            _ax, _ay = _compute_crop_from_face(_af, source_w, source_h)
        else:
            _ax, _ay = base_x, base_y
    else:
        _ax, _ay = base_x, base_y
    last_x = clamp(_ax, 0.0, max(0.0, source_w - crop_w))
    last_y = clamp(_ay, 0.0, max(0.0, source_h - crop_h))
    last_t = seg_start

    _prod_thread.start()

    keyframes: List[KeyframePoint] = []
    prev_wide_speaker = active_speaker
    # FIX 2: strict 1.0 s dwell gate (sub-1 s interruptions are ignored)
    WIDE_SPEAKER_DWELL_MIN = 1.0
    _wide_committed_speaker: str = active_speaker
    _wide_committed_slot: Optional[str] = target_slot
    _wide_candidate_speaker: Optional[str] = None
    _wide_candidate_since: float = seg_start
    
    _has_locked_on = False
    # Indices of keyframes appended while HOLDING (no subject / missing frame).
    # Back-filled to the reacquired position once tracking resumes.
    _pending_hold_idx: List[int] = []

    tick_i = 0
    try:
        while True:
            item_t, frame = _frame_q.get(block=True)   # GPU unblocks producer here
            if item_t is _SENTINEL:
                break
            t = float(item_t)
            tick_i += 1

            if frame is None:
                kf_index = len(keyframes)
                keyframes.append(KeyframePoint(
                    t_rel=t - seg_start, crop_x=last_x, crop_y=last_y,
                    is_snap=False, confidence=0.0))
                _pending_hold_idx.append(kf_index)
                if logger and VERBOSE_STAGE_LOGGING:
                    logger(f"[TRACK] seg#{seg_index + 1} [{tick_i}/{total_ticks}] t={t:.3f} frame=MISSING -> hold ({last_x:.1f},{last_y:.1f})")
                continue

            if wide:
                # Route before detection: mapped diarization selects the template
                # slot, then InsightFace searches only near that slot.
                tick_speaker_pre = active_speaker_at(t, diar_timeline)
                if tick_speaker_pre == "UNKNOWN":
                    tick_speaker_pre = _wide_committed_speaker
                if tick_speaker_pre != _wide_committed_speaker:
                    if tick_speaker_pre != _wide_candidate_speaker:
                        _wide_candidate_speaker = tick_speaker_pre
                        _wide_candidate_since = t
                    if (t - _wide_candidate_since) >= WIDE_SPEAKER_DWELL_MIN:
                        new_slot_pre = _choose_target_slot(template, [], speaker_db, tick_speaker_pre)
                        if new_slot_pre != _wide_committed_slot:
                            if logger and VERBOSE_STAGE_LOGGING:
                                logger(
                                    f"[WIDE] seg#{seg_index + 1} t={t:.3f} switch "
                                    f"{_wide_committed_speaker}->{tick_speaker_pre} "
                                    f"slot {_wide_committed_slot}->{new_slot_pre} "
                                    f"(dwell={t - _wide_candidate_since:.2f}s)"
                                )
                            _wide_committed_slot = new_slot_pre
                        _wide_committed_speaker = tick_speaker_pre
                        _wide_candidate_speaker = None
                else:
                    _wide_candidate_speaker = None
                target_slot = _wide_committed_slot

            # ---- FIX 6: ROI-restricted GPU detection (InsightFace, full resolution) ----
            # The frame is pre-cropped to the active slot window before sending to
            # InsightFace, making it blind to the opposite speaker.
            if wide and target_slot:
                slot_obj = _template_slot_by_name(template, target_slot)
                if slot_obj is not None:
                    padding = 0.08   # 8 % margin so tight faces are not clipped
                    sh, sw = frame.shape[:2]
                    bw = max(1.0, slot_obj.bbox[2] - slot_obj.bbox[0])
                    bh = max(1.0, slot_obj.bbox[3] - slot_obj.bbox[1])
                    roi_x1 = int(clamp(slot_obj.bbox[0] - padding * bw, 0, sw - 1))
                    roi_y1 = int(clamp(slot_obj.bbox[1] - padding * bh, 0, sh - 1))
                    roi_x2 = int(clamp(slot_obj.bbox[2] + padding * bw, 0, sw))
                    roi_y2 = int(clamp(slot_obj.bbox[3] + padding * bh, 0, sh))
                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                        roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                        # ROI detection on the analysis-scaled crop; coords scale
                        # back to crop pixels, then the +roi offset maps to source.
                        detected_roi = _detect_scaled(roi_crop)
                        detected: List[FaceObs] = [
                            FaceObs(
                                x1=f.x1 + roi_x1, y1=f.y1 + roi_y1,
                                x2=f.x2 + roi_x1, y2=f.y2 + roi_y1,
                                conf=f.conf,
                                cx=f.cx + roi_x1, cy=f.cy + roi_y1,
                                area=f.area,
                                yaw=f.yaw, embedding=f.embedding,
                                landmarks=f.landmarks,
                                shirt_hist=f.shirt_hist, shirt_rgb=f.shirt_rgb,
                                torso_conf=f.torso_conf,
                            )
                            for f in detected_roi
                        ]
                    else:
                        detected = _detect_scaled(frame)
                else:
                    detected = _detect_scaled(frame)
            elif wide:
                detected = _detect_scaled(frame)
            else:
                # Solo: read the GPU-saturated batch pre-pass result.
                detected = det_map.get(t, [])

            # Speaker dwell gate (FIX 2)
            if wide:
                tick_speaker = active_speaker_at(t, diar_timeline)
                if tick_speaker == "UNKNOWN":
                    tick_speaker = _wide_committed_speaker
                if tick_speaker != _wide_committed_speaker:
                    if tick_speaker != _wide_candidate_speaker:
                        _wide_candidate_speaker = tick_speaker
                        _wide_candidate_since = t
                    if (t - _wide_candidate_since) >= WIDE_SPEAKER_DWELL_MIN:
                        new_slot = _choose_target_slot(template, detected, speaker_db, tick_speaker)
                        if new_slot != _wide_committed_slot:
                            if logger and VERBOSE_STAGE_LOGGING:
                                logger(f"[WIDE] seg#{seg_index + 1} t={t:.3f} switch {_wide_committed_speaker}->{tick_speaker} slot {_wide_committed_slot}->{new_slot} (dwell={t - _wide_candidate_since:.2f}s)")
                            _wide_committed_slot = new_slot
                        _wide_committed_speaker = tick_speaker
                        _wide_candidate_speaker = None
                else:
                    _wide_candidate_speaker = None
                target_slot = _wide_committed_slot
                prev_wide_speaker = _wide_committed_speaker
                focus = _select_focus_face(detected, template, speaker_db, _wide_committed_speaker)
            else:
                focus = _select_solo_focus_face(detected, source_w, source_h)

            if wide and logger and VERBOSE_STAGE_LOGGING:
                logger(
                    f"[WIDE-ROI] seg#{seg_index + 1} t={t:.3f} "
                    f"speaker={_wide_committed_speaker} slot={target_slot or 'N/A'} "
                    f"hit={'yes' if focus is not None else 'no'} faces={len(detected)}"
                )

            held = False
            if focus is not None:
                desired_x, desired_y = _compute_crop_from_face(focus, source_w, source_h)
                confidence = float(clamp(focus.conf, 0.0, 1.0))
                if not wide and target_slot is None:
                    target_slot = _choose_target_slot(template, detected, speaker_db, active_speaker)
            else:
                desired_x, desired_y = base_x, base_y
                confidence = 0.15

            if wide:
                # Wide → committed slot anchor is the immutable origin; a confident
                # face only nudges it. The EMA controller below smooths every move,
                # so slot switches read as a deliberate pan rather than a teleport.
                slot_face = _template_slot_to_face(template, target_slot, source_w, source_h)
                if slot_face is not None:
                    slot_x, slot_y = _compute_crop_from_face(slot_face, source_w, source_h)
                    if focus is not None and confidence >= FACE_CONF_FALLBACK:
                        ref_x, ref_y = _compute_crop_from_face(focus, source_w, source_h)
                        desired_x = (0.95 * slot_x) + (0.05 * ref_x)
                        desired_y = (0.95 * slot_y) + (0.05 * ref_y)
                    else:
                        desired_x, desired_y = slot_x, slot_y
                else:
                    desired_x, desired_y = base_x, base_y
            else:
                # Solo is face-driven. When the subject is momentarily lost we HOLD
                # the last good crop (freeze) instead of drifting to the template
                # anchor; the frozen frames are back-filled on reacquisition.
                if focus is None:
                    held = True
                    desired_x, desired_y = last_x, last_y
                    confidence = 0.0

            # ─── Motion controller: EMA at all times, lock from frame 1, hold-on-miss ───
            if held:
                # Subject lost: freeze on the last good crop.
                new_x, new_y = last_x, last_y
            elif not _has_locked_on:
                # First real target of the cut: lock instantly to it.
                new_x, new_y = desired_x, desired_y
            else:
                # Locked: smooth EMA toward the desired crop on every frame.
                new_x = (EMA_ALPHA * desired_x) + ((1.0 - EMA_ALPHA) * last_x)
                new_y = (EMA_ALPHA * desired_y) + ((1.0 - EMA_ALPHA) * last_y)
            is_snap = False

            new_x = clamp(new_x, 0.0, max(0.0, source_w - crop_w))
            new_y = clamp(new_y, 0.0, max(0.0, source_h - crop_h))

            if not held and not _has_locked_on:
                # Time machine: rewrite every earlier placeholder/held keyframe to
                # this first locked position so the camera never sat at the wrong
                # template anchor.
                _has_locked_on = True
                for k in keyframes:
                    k.crop_x = float(new_x)
                    k.crop_y = float(new_y)
                _pending_hold_idx.clear()
            elif not held and _pending_hold_idx:
                # Reacquired after a hold gap: back-fill the frozen frames to the
                # resumed position so there is no freeze-then-jump artifact.
                for ki in _pending_hold_idx:
                    if 0 <= ki < len(keyframes):
                        keyframes[ki].crop_x = float(new_x)
                        keyframes[ki].crop_y = float(new_y)
                _pending_hold_idx.clear()

            kf_index = len(keyframes)
            keyframes.append(KeyframePoint(
                t_rel=round(t - seg_start, 4),
                crop_x=float(new_x),
                crop_y=float(new_y),
                is_snap=bool(is_snap),
                confidence=float(clamp(confidence, 0.0, 1.0)),
                face_box=(
                    (float(focus.x1), float(focus.y1), float(focus.x2), float(focus.y2))
                    if focus is not None else None
                ),
            ))
            if held:
                _pending_hold_idx.append(kf_index)

            route_speaker = _wide_committed_speaker if wide else active_speaker
            if logger and VERBOSE_STAGE_LOGGING:
                logger(
                    f"[TRACK] seg#{seg_index + 1} [{tick_i}/{total_ticks}] t={t:.3f} "
                    f"faces={len(detected)} focus={'yes' if focus is not None else 'no'} "
                    f"mode={route_mode} speaker={route_speaker} slot={target_slot or 'N/A'} held={held} "
                    f"crop=({new_x:.1f},{new_y:.1f}) desired=({desired_x:.1f},{desired_y:.1f}) conf={confidence:.3f} GPU=yes"
                )
            if not held:
                last_x, last_y, last_t = new_x, new_y, t

    finally:
        try:
            while True:
                _frame_q.get_nowait()
        except Exception:
            pass
        _prod_thread.join(timeout=10.0)

        # FIX 3: explicit flush â€” free the RAM buffer before the next segment loads
        _frame_cache.clear()
        del _frame_cache

    # ------------------------------------------------------------------ #
    # DERIVE crop_expr_x / crop_expr_y                                   #
    #                                                                     #
    # Priority:                                                           #
    #   1. Median of high-confidence tracked keyframes                   #
    #      (the EMA-smoothed positions from actual face detections).     #
    #      Used when â‰¥ 30 % of ticks had faces â€” i.e. tracking worked.  #
    #                                                                     #
    #   2. Slot-specific pixel anchor (wide shots only).                 #
    #      The committed target_slot's screen-centre, computed once      #
    #      from the manually authored bbox_norm â€” not the union centre.  #
    #                                                                     #
    #   3. base_x / base_y â€” the template's union crop anchor.           #
    #      Last resort when no faces found and no slot geometry.         #
    # ------------------------------------------------------------------ #
    # â”€â”€ Crop position derivation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # crop_expr_x / crop_expr_y serve two roles:
    #   1. Initial value in the FFmpeg crop filter before sendcmd kicks in.
    #   2. Static fallback if sendcmd rendering is unavailable.
    #
    # Zero-face frames (confidence â‰¤ 0.15) are excluded â€“ they are cut-
    # boundary artefacts that would distort the starting position.
    #
    # Priority:
    #   1. First valid keyframe position (first frame where a face was found).
    #   2. Slot-geometry anchor (wide shots only).
    #   3. Template union anchor (last resort).
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _apply_stable_start_anchor(keyframes, seg_index, logger)
    valid_kf = [k for k in keyframes if k.confidence > 0.15]

    if keyframes:
        overall_conf = float(np.clip(np.mean([k.confidence for k in keyframes]), 0.0, 1.0))
        snaps = sum(1 for k in keyframes if k.is_snap)
        faces_detected = len(valid_kf)
    else:
        overall_conf = 0.0
        snaps = 0
        faces_detected = 0

    total_kf = max(1, len(keyframes))

    if valid_kf:
        # Use the first valid keyframe as the initial / fallback crop position.
        # This is where the camera starts and where sendcmd begins from.
        crop_x = float(valid_kf[0].crop_x)
        crop_y = float(valid_kf[0].crop_y)
    elif wide and target_slot:
        slot_face = _template_slot_to_face(template, target_slot, source_w, source_h)
        if slot_face is not None:
            crop_x, crop_y = _compute_crop_from_face(slot_face, source_w, source_h)
        else:
            crop_x, crop_y = base_x, base_y
    else:
        crop_x, crop_y = base_x, base_y

    crop_x = clamp(crop_x, 0.0, max(0.0, source_w - crop_w))
    crop_y = clamp(crop_y, 0.0, max(0.0, source_h - crop_h))

    crop_expr_x = f"{crop_x:.3f}"
    crop_expr_y = f"{crop_y:.3f}"

    elapsed = time.time() - _t0_plan
    logger(
        f"[PLAN] seg#{seg_index + 1} DONE in {elapsed:.1f}s | "
        f"frames={total_ticks} faces_detected={faces_detected}/{total_ticks} "
        f"({100*faces_detected//max(1,total_ticks)}%) snaps={snaps} | "
        f"initial_crop=({crop_x:.1f},{crop_y:.1f}) conf={overall_conf:.3f} wide={wide}"
    )

    return SegmentPlan(
        index=seg_index,
        start=float(seg_start),
        end=float(seg_end),
        template_id=int(template.template_id),
        speaker=str(active_speaker if active_speaker else "UNKNOWN"),
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        keyframes=keyframes,
        target_slot=target_slot,
        crop_expr_x=crop_expr_x,
        crop_expr_y=crop_expr_y,
    )


# ======================================================================
# Manual template support
# ======================================================================

def _bbox_norm_to_abs(bbox_norm: Sequence[float], src_w: int, src_h: int) -> Tuple[float, float, float, float]:
    x1 = clamp(float(bbox_norm[0]) * src_w, 0.0, max(0.0, src_w - 1))
    y1 = clamp(float(bbox_norm[1]) * src_h, 0.0, max(0.0, src_h - 1))
    x2 = clamp(float(bbox_norm[2]) * src_w, 0.0, max(0.0, src_w - 1))
    y2 = clamp(float(bbox_norm[3]) * src_h, 0.0, max(0.0, src_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return float(x1), float(y1), float(x2), float(y2)


def _abs_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _manual_crop_anchor_from_slots(face_slots: Sequence[FaceSlot], src_w: int, src_h: int) -> Dict[str, float]:
    crop_w, crop_h = _canonical_crop_dims(src_w, src_h)
    if not face_slots:
        return {
            "x": float(max(0.0, (src_w - crop_w) / 2.0)),
            "y": 0.0,
            "w": float(crop_w),
            "h": float(crop_h),
        }

    if len(face_slots) == 1:
        bbox = face_slots[0].bbox
        cx, cy = _abs_bbox_center(bbox)
        x = clamp(cx - crop_w / 2.0, 0.0, max(0.0, src_w - crop_w))
        y = clamp(cy - crop_h * HEAD_ROOM_FRAC, 0.0, max(0.0, src_h - crop_h))
        return {"x": float(x), "y": float(y), "w": float(crop_w), "h": float(crop_h)}

    xs1 = [s.bbox[0] for s in face_slots]
    ys1 = [s.bbox[1] for s in face_slots]
    xs2 = [s.bbox[2] for s in face_slots]
    ys2 = [s.bbox[3] for s in face_slots]
    union_cx = (min(xs1) + max(xs2)) / 2.0
    union_cy = (min(ys1) + max(ys2)) / 2.0
    x = clamp(union_cx - crop_w / 2.0, 0.0, max(0.0, src_w - crop_w))
    y = clamp(union_cy - crop_h * 0.28, 0.0, max(0.0, src_h - crop_h))
    return {"x": float(x), "y": float(y), "w": float(crop_w), "h": float(crop_h)}


def _manual_template_candidates(job_dir: Path, manual_templates_path: Optional[str] = None) -> List[Path]:
    candidates: List[Path] = []
    if manual_templates_path:
        candidates.append(Path(manual_templates_path))
    candidates.extend([
        job_dir / "manual_templates.json",
        job_dir / TEMPLATE_DIR_NAME / "manual_templates.json",
    ])
    seen: set[str] = set()
    unique: List[Path] = []
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _manual_template_display_name(entry: Dict[str, Any], template_id: int) -> str:
    name = str(entry.get("display_name", "")).strip()
    if name:
        return name
    raw_type = str(entry.get("type", "SINGLE")).strip().upper()
    if raw_type == "WIDE":
        return f"wide_manual_{template_id:02d}"
    return f"manual_{template_id:02d}"


def _manual_slot_from_entry(
    slot_entry: Dict[str, Any],
    src_w: int,
    src_h: int,
    slot_order: int,
) -> Optional[FaceSlot]:
    bbox_norm = slot_entry.get("bbox_norm")
    if not isinstance(bbox_norm, (list, tuple)) or len(bbox_norm) < 4:
        return None

    bbox = _bbox_norm_to_abs(bbox_norm, src_w, src_h)
    cx, cy = _abs_bbox_center(bbox)
    label = str(slot_entry.get("speaker_label", "")).strip() or None
    slot_name = str(slot_entry.get("slot_name", "")).strip() or _slot_name_for_index(slot_order, 1)
    confidence = float(slot_entry.get("confidence", 1.0))
    shirt_rgb = slot_entry.get("shirt_rgb")
    if isinstance(shirt_rgb, (list, tuple)) and len(shirt_rgb) >= 3:
        try:
            shirt_rgb = (int(shirt_rgb[0]), int(shirt_rgb[1]), int(shirt_rgb[2]))
        except Exception:
            shirt_rgb = None
    else:
        shirt_rgb = None

    return FaceSlot(
        slot_name=slot_name,
        center=(float(cx), float(cy)),
        bbox=bbox,
        center_norm=(float(cx / max(1, src_w)), float(cy / max(1, src_h))),
        bbox_norm=(float(bbox[0] / max(1, src_w)), float(bbox[1] / max(1, src_h)), float(bbox[2] / max(1, src_w)), float(bbox[3] / max(1, src_h))),
        confidence=confidence,
        embedding=None,
        shirt_hist=None,
        shirt_rgb=shirt_rgb,
        speaker_label=label,
        slot_order=slot_order,
    )


def load_manual_templates(
    job_dir: Path,
    video_path: str,
    src_w: int,
    src_h: int,
    logger: Optional[Logger] = None,
    manual_templates_path: Optional[str] = None,
) -> Optional[List[TemplateRecord]]:
    """
    Load hand-authored templates from manual_templates.json.

    Supported search order:
      1. explicit manual_templates_path
      2. job_dir/manual_templates.json
      3. job_dir/templates/manual_templates.json
    """
    candidates = _manual_template_candidates(job_dir, manual_templates_path)
    src_json: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            src_json = candidate
            break
    if src_json is None:
        return None

    try:
        data = read_json(src_json)
    except Exception as exc:
        if logger:
            logger(f"[TEMPLATE] failed to read manual templates: {exc}")
        return None

    if not isinstance(data, list):
        if logger:
            logger(f"[TEMPLATE] manual template file must contain a list: {src_json}")
        return None

    ensure_dir(job_dir / TEMPLATE_DIR_NAME)
    templates: List[TemplateRecord] = []

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        try:
            template_id = int(entry.get("template_id", idx))
        except Exception:
            template_id = idx
        try:
            timestamp = float(entry.get("timestamp", 0.0))
        except Exception:
            timestamp = 0.0

        display_name = _manual_template_display_name(entry, template_id)
        template_type = str(entry.get("type", "SINGLE")).strip().upper()
        frame = extract_frame(video_path, timestamp)
        if frame is None:
            if logger:
                logger(f"[TEMPLATE] manual template {template_id} skipped, no frame at {timestamp:.3f}s")
            continue

        slots_raw = entry.get("slots", [])
        face_slots: List[FaceSlot] = []
        if isinstance(slots_raw, list):
            for sidx, slot_entry in enumerate(slots_raw):
                if not isinstance(slot_entry, dict):
                    continue
                slot = _manual_slot_from_entry(slot_entry, src_w, src_h, sidx)
                if slot is not None:
                    face_slots.append(slot)

        if not face_slots:
            if logger:
                logger(f"[TEMPLATE] manual template {template_id} skipped, no valid slots")
            continue

        raw_path = job_dir / TEMPLATE_DIR_NAME / f"{display_name}.jpg"
        overlay_path = job_dir / TEMPLATE_DIR_NAME / f"{display_name}_overlay.jpg"
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                raw_path.write_bytes(buf.tobytes())
            template_for_overlay = TemplateRecord(
                template_id=template_id,
                phash=[],
                segment_indices=[],
                type=template_type,
                canonical_frame=str(raw_path),
                canonical_timestamp=timestamp,
                n_faces=len(face_slots),
                face_slots=face_slots,
                crop_anchor=_manual_crop_anchor_from_slots(face_slots, src_w, src_h),
                speaker_votes={},
                speaker_embeddings={},
                shirt_signatures={},
                slot_geometry={
                    s.slot_name: {
                        "center_x": float(s.center[0]),
                        "center_y": float(s.center[1]),
                        "x1": float(s.bbox[0]),
                        "y1": float(s.bbox[1]),
                        "x2": float(s.bbox[2]),
                        "y2": float(s.bbox[3]),
                        "order": float(s.slot_order),
                    }
                    for s in face_slots
                },
                display_name=display_name,
            )
            overlay = _draw_template_overlay(frame, template_for_overlay)
            ok_ov, buf_ov = cv2.imencode(".jpg", overlay)
            if ok_ov:
                overlay_path.write_bytes(buf_ov.tobytes())
            phash, _ = _phash(frame)
            template_for_overlay.phash = phash.astype(np.uint8).tolist()
            templates.append(template_for_overlay)
            if logger:
                logger(f"[TEMPLATE] manual template loaded id={template_id} name={display_name} slots={len(face_slots)}")
        except Exception as exc:
            if logger:
                logger(f"[TEMPLATE] manual template {template_id} export failed: {exc}")

    templates.sort(key=lambda t: t.template_id)
    if not templates:
        if logger:
            logger(f"[TEMPLATE] no usable manual templates found in {src_json}")
        return None

    return templates


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (area_a + area_b - inter))


def _match_face_to_slot(
    detected: Sequence[FaceObs],
    slot: FaceSlot,
) -> Optional[FaceObs]:
    best_face: Optional[FaceObs] = None
    best_score = -1.0
    for face in detected:
        face_bbox = (face.x1, face.y1, face.x2, face.y2)
        score = _bbox_iou(face_bbox, slot.bbox)
        if score > best_score:
            best_score = score
            best_face = face
    return best_face if best_score > 0.0 else None


def _classify_cuts_against_manual_templates(
    video_path: str,
    cuts: List[float],
    templates: Sequence[TemplateRecord],
    logger: Optional[Logger] = None,
) -> Dict[int, int]:
    scene_to_template: Dict[int, int] = {}
    if not templates:
        return scene_to_template

    default_id = int(templates[0].template_id)
    for scene_idx in range(max(0, len(cuts) - 1)):
        scene_to_template[scene_idx] = default_id
    return scene_to_template


def _build_speaker_db_from_manual_templates(
    templates: Sequence[TemplateRecord],
    video_path: str,
    face_backend: BaseFaceBackend,
    logger: Optional[Logger] = None,
) -> Dict[str, List[float]]:
    """
    Build speaker embedding DB from manually-authored templates.

    If the canonical timestamp doesn't yield a usable face detection (e.g. the
    face is sideways or partially off-screen at that exact moment), we try up to
    six nearby offsets before giving up on that template.  This makes the DB
    far more robust for manual templates where the timestamp was chosen for
    visual clarity, not necessarily for frontal-face quality.
    """
    db: Dict[str, List[np.ndarray]] = {}

    for tmpl in templates:
        # Try canonical timestamp, then Â±0.5 s, Â±1.0 s, Â±1.5 s, +2.0 s offsets
        offsets = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0, -1.5]
        matched_this_tmpl = False

        for offset in offsets:
            ts = max(0.0, tmpl.canonical_timestamp + offset)
            frame = extract_frame(video_path, ts)
            if frame is None:
                continue
            detected = face_backend.detect(frame)
            if not detected:
                continue

            for slot in tmpl.face_slots:
                if not slot.speaker_label:
                    continue
                # Skip slots whose speaker we already have a good embedding for
                if slot.speaker_label in db and len(db[slot.speaker_label]) >= 3:
                    continue
                matched = _match_face_to_slot(detected, slot)
                if matched is None:
                    continue
                if matched.embedding is not None:
                    emb = normalize_vec(np.asarray(matched.embedding, dtype=np.float32))
                    slot.embedding = emb.tolist()
                    db.setdefault(slot.speaker_label, []).append(emb)
                    matched_this_tmpl = True
                    if logger:
                        logger(
                            f"[SPEAKER-DB] {slot.speaker_label} <- "
                            f"template {tmpl.template_id}:{slot.slot_name} @{ts:.2f}s"
                        )
            if matched_this_tmpl:
                break   # found at least one embedding for this template

    out: Dict[str, List[float]] = {}
    for spk, vecs in db.items():
        if not vecs:
            continue
        proto = normalize_vec(np.mean(np.stack(vecs, axis=0), axis=0))
        out[spk] = proto.tolist()
        if logger:
            logger(f"[SPEAKER-DB] manual speaker {spk}: {len(vecs)} embedding(s)")
    return out

# ======================================================================
# Main pipeline

# ======================================================================
# Main pipeline
# ======================================================================
def save_scene_cuts(job_dir: Path, cuts: List[float]) -> None:
    write_json(job_dir / SCENE_FILE, [float(c) for c in cuts])

def process_clip(
    video_path: str,
    diar_path: str,
    start_sec: float,
    end_sec: float,
    speaker_filter: Optional[str],
    out_path: str,
    ffmpeg_bin: str,
    diar_py: str,
    logger: Logger,
    manual_templates_path: Optional[str] = None,
    gpu_lock: Optional[threading.Semaphore] = None,
) -> bool:
    if gpu_lock is not None:
        gpu_lock.acquire()
    _t0_pipeline = time.time()
    ensure_dir(Path(out_path).parent)
    job_dir = Path(out_path).with_suffix("").with_name(Path(out_path).stem + "_job")
    ensure_dir(job_dir)
    logger.file_path = job_dir / LOG_FILE
    logger.extra_file_path = job_dir / DEBUG_LOG_FILE

    logger(f"[PROCESS] job_dir={job_dir}")
    logger(f"[PROCESS] template_dir={job_dir / TEMPLATE_DIR_NAME}")
    logger(f"[PROCESS] classification_cache={job_dir / CLASSIFICATION_CACHE_DIR_NAME}")
    src_w, src_h, fps = get_video_info(video_path)
    logger(f"Video: {src_w}x{src_h} @ {fps:.2f} fps")

    dynamic_target_h = int(src_h)
    dynamic_target_w = max(2, int(math.floor((src_h * 9 / 16) / 2.0) * 2))
    _configure_pipeline_resolution((dynamic_target_w, dynamic_target_h))

    # ── Diarization head-start ──────────────────────────────────────────────
    # Manual templates are a cheap read and reveal has_wide BEFORE scene-cut
    # detection, so the diarization subprocess (the longest serial stage of a
    # clip, often 1-2 min) can run concurrently with scene cuts, template
    # image export and cut classification. Identical inputs and outputs — only
    # the schedule changes. Auto-discovery clips keep the serial path because
    # has_wide is unknown until discovery has run.
    manual_templates = load_manual_templates(
        job_dir=job_dir,
        video_path=video_path,
        src_w=src_w,
        src_h=src_h,
        logger=logger,
        manual_templates_path=manual_templates_path,
    )
    _diar_thread: Optional[threading.Thread] = None
    _diar_box: Dict[str, List[DiarSegment]] = {}
    if manual_templates and any(_template_is_wide(t, src_w, src_h) for t in manual_templates):
        def _diar_worker() -> None:
            try:
                _diar_box["timeline"] = _load_or_create_clip_diarization(
                    video_path=video_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    job_dir=job_dir,
                    diar_path=diar_path,
                    ffmpeg_bin=ffmpeg_bin,
                    diar_py=diar_py,
                    logger=logger,
                )
            except Exception as _dexc:
                logger(f"[DIAR] background diarization failed: {_dexc}")

        _diar_thread = threading.Thread(target=_diar_worker, daemon=True, name="clip-diar")
        _diar_thread.start()
        logger("[DIAR] started in background (overlaps scene cuts + template prep)")

    cache_file = job_dir / SCENE_FILE
    if cache_file.exists():
        logger(f"[PROCESS] Loading cached scene cuts from {cache_file}")
        cuts = read_json(cache_file)
    else:
        cuts = detect_scenes(video_path, start_sec, end_sec, ffmpeg_bin, logger, fps=fps)
        logger(f"[PROCESS] scene cuts detected: {len(cuts)}")
        save_scene_cuts(job_dir, cuts)

    logger("-- Creating/reusing singleton face backend ---------")
    global _PIPELINE_FACE_BACKEND
    if _PIPELINE_FACE_BACKEND is not None:
        face_backend = _PIPELINE_FACE_BACKEND
        logger("[FACE] Reusing cached face backend (singleton)")
    else:
        face_backend = create_face_backend(logger)
        _PIPELINE_FACE_BACKEND = face_backend
        logger("[FACE] Face backend created and cached as singleton")

    try:
        logger("â”€â”€ Loading or discovering templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        # manual_templates was loaded before scene-cut detection so the
        # diarization subprocess could get a head start (see above).

        templates: List[TemplateRecord] = []
        scene_to_template: Dict[int, int] = {}
        speaker_db: Dict[str, List[float]] = {}

        if manual_templates:
            logger(f"[TEMPLATE] Loaded {len(manual_templates)} manual templates â€” skipping auto discovery")
            templates = list(manual_templates)
            templates, pruned_ids = _ensure_template_images(
                video_path,
                job_dir,
                templates,
                logger,
                registry=None,
                force_overwrite=False,
            )
            if pruned_ids:
                logger(f"[TEMPLATE] {len(pruned_ids)} manual template(s) pruned by deleted image(s)")
            if not templates:
                logger("No manual templates available after pruning.")
                return False
            scene_to_template = _classify_cuts_against_manual_templates(video_path, cuts, templates, logger)
        else:
            registry = load_template_registry(job_dir)
            cached_cuts: List[float] = []
            cache_ok = False

            if registry is not None:
                try:
                    templates, scene_to_template, speaker_db, cached_cuts = deserialize_template_registry(registry)
                    cache_ok = bool(templates) and _registry_cuts_match(cached_cuts, cuts)
                    if cache_ok:
                        logger(f"[CACHE] Loaded template registry with {len(templates)} templates")
                        templates = _apply_template_registry_controls(templates, registry, logger)
                    else:
                        logger("[CACHE] Registry present but cuts/templates do not match; rebuilding")
                except Exception as e:
                    logger(f"[CACHE] Registry load failed, rebuilding: {e}")
                    cache_ok = False

            if not cache_ok:
                logger("[CACHE] Clearing stale template cache before fresh discovery...")
                _force_clear_template_cache(job_dir, logger)
                logger("[DISCOVER] Starting template discovery...")
                _t0_discover = time.time()
                templates, scene_to_template, _ = discover_templates(video_path, cuts, face_backend, job_dir, logger)
                logger(f"[TIMER] discover_templates done in {time.time()-_t0_discover:.1f}s â€” {len(templates)} template(s) found")
                if not templates:
                    logger("No templates discovered.")
                    return False
                templates = _template_prune_duplicates(_apply_template_registry_controls(templates, registry, logger), logger=logger)

                logger(f"[TEMPLATE] *** Exporting {len(templates)} raw template frames immediately for review ***")
                _t0_img = time.time()
                _export_raw_template_images(video_path, job_dir, templates, logger)
                logger(f"[TIMER] raw image export done in {time.time()-_t0_img:.1f}s")

        # --- 2. Now we have templates. Check for wide templates ---
        has_wide = any(_template_is_wide(t, src_w, src_h) for t in templates)

        # --- 3. Conditionally load diarization ---
        if has_wide:
            logger("â”€â”€ Loading diarization â”•â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
            if _diar_thread is not None:
                _t0_djoin = time.time()
                _diar_thread.join()
                logger(f"[DIAR] background diarization ready (blocked {time.time() - _t0_djoin:.1f}s at join)")
                diar_timeline = _diar_box.get("timeline") or []
            else:
                diar_timeline = _load_or_create_clip_diarization(
                    video_path=video_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    job_dir=job_dir,
                    diar_path=diar_path,
                    ffmpeg_bin=ffmpeg_bin,
                    diar_py=diar_py,
                    logger=logger,
                )
            if not diar_timeline:
                logger("WARN: diarization timeline empty or unreadable")
            else:
                totals = speaker_totals(diar_timeline)
                for spk, sec in sorted(totals.items(), key=lambda kv: kv[0]):
                    logger(f"  {spk}: {fmt_time(sec)} total speech")
        else:
            logger("[DIAR] skipped â€” no wide templates in this clip")
            diar_timeline = []

        # --- 4. Build segments (segs) ---
        if speaker_filter == "continuous" or not has_wide:
            if not has_wide:
                logger("[PROCESS] No wide templates found. Forcing continuous segment mode to bypass diarization.")
            else:
                logger("[PROCESS] Continuous mode active: using scene cuts as the base timeline.")
            segs = [DiarSegment(start=cuts[j], end=cuts[j + 1], speaker="UNKNOWN") for j in range(len(cuts) - 1)]
        elif speaker_filter:
            segs = [s for s in diar_timeline if s.speaker == speaker_filter and s.end > start_sec and s.start < end_sec]
        else:
            segs = [s for s in diar_timeline if s.end > start_sec and s.start < end_sec]

        if speaker_filter != "continuous" and has_wide:
            for s in segs:
                s.start = max(s.start, start_sec)
                s.end = min(s.end, end_sec)
            segs = [s for s in segs if (s.end - s.start) > 0.15]

        if not segs:
            logger("No segments to process in the requested range.")
            return False

        logger(f"[PROCESS] segments queued: {len(segs)}")

        # --- 5. Finish template setup (stage 2: speaker DB and overlays) ---
        if manual_templates:
            speaker_db = _build_speaker_db_from_manual_templates(templates, video_path, face_backend, logger)
            _manual_tpl_ids: Set[int] = {t.template_id for t in manual_templates}
            _assign_template_slot_labels(templates, speaker_db, _manual_tpl_ids, logger)
            save_template_registry(job_dir, templates, speaker_db, cuts, scene_to_template, manual_templates=templates)
        else:
            if not cache_ok:
                _t0_spk = time.time()
                speaker_db = build_speaker_embedding_db(templates, scene_to_template, diar_timeline, video_path, face_backend, cuts, logger)
                logger(f"[TIMER] speaker embedding DB built in {time.time()-_t0_spk:.1f}s â€” speakers: {list(speaker_db.keys())}")
                _assign_template_slot_labels(templates, speaker_db, logger=logger)

                logger("[TEMPLATE] Re-exporting overlays with speaker labels applied...")
                all_templates = list(templates)
                templates, pruned_ids = _ensure_template_images(video_path, job_dir, templates, logger, registry, force_overwrite=True)
                if pruned_ids:
                    scene_to_template = _remap_scene_templates_to_active(scene_to_template, all_templates, templates, pruned_ids, logger)
                save_template_registry(job_dir, templates, speaker_db, cuts, scene_to_template)
            else:
                all_templates = list(templates)
                templates, pruned_ids = _ensure_template_images(video_path, job_dir, templates, logger, registry, force_overwrite=False)
                if pruned_ids:
                    scene_to_template = _remap_scene_templates_to_active(scene_to_template, all_templates, templates, pruned_ids, logger)
                if not speaker_db:
                    speaker_db = build_speaker_embedding_db(templates, scene_to_template, diar_timeline, video_path, face_backend, cuts, logger)
                _assign_template_slot_labels(templates, speaker_db, logger=logger)
                save_template_registry(job_dir, templates, speaker_db, cuts, scene_to_template)

        logger(f"[PROCESS] templates active={len(templates)} speaker_db={list(speaker_db.keys())}")
        template_by_id = {t.template_id: t for t in templates}

        refined_segs: List[DiarSegment] = []
        for seg in segs:
            mid = (seg.start + seg.end) / 2.0
            scene_idx = _scene_index_for_time(cuts, mid)
            template_id = scene_to_template.get(scene_idx, templates[0].template_id if templates else 0)
            cls_id, cls_conf, cls_hidden, _scores = _classify_template_window_cached(
                job_dir=job_dir,
                cache_label=f"hidden_route_{len(refined_segs) + 1:04d}",
                video_path=video_path,
                cut_time=mid,
                templates=templates,
                face_backend=face_backend,
                speaker_db=speaker_db,
                source_w=src_w,
                source_h=src_h,
                start_bound=seg.start,
                end_bound=seg.end,
                logger=logger,
                active_speaker_hint="UNKNOWN",
            )
            if cls_id is not None and (cls_hidden or cls_conf >= 0.28):
                template_id = int(cls_id)
            template = template_by_id.get(template_id, templates[0])
            hidden_points = _detect_hidden_cut_points(
                job_dir=job_dir,
                video_path=video_path,
                seg_start=seg.start,
                seg_end=seg.end,
                template=template,
                face_backend=face_backend,
                speaker_db=speaker_db,
                diar_timeline=diar_timeline,
                source_w=src_w,
                source_h=src_h,
                logger=logger,
            )
            if not hidden_points:
                refined_segs.append(seg)
                continue

            prev = seg.start
            for pt in hidden_points:
                if pt - prev >= 0.20:
                    refined_segs.append(DiarSegment(start=prev, end=pt, speaker=seg.speaker))
                prev = pt
            if seg.end - prev >= 0.20:
                refined_segs.append(DiarSegment(start=prev, end=seg.end, speaker=seg.speaker))
        if refined_segs:
            logger(f"[CUT] hidden-cut refinement split {len(segs)} -> {len(refined_segs)} segments")
            segs = refined_segs

        if segs:
            pad_start = 3.0 / fps
            pad_end = 2.0 / fps
            segs = [
                DiarSegment(
                    start=max(start_sec, s.start - pad_start),
                    end=min(end_sec, s.end + pad_end),
                    speaker=s.speaker,
                )
                for s in segs
            ]

        segs, trim_manifest = _trim_zero_face_boundaries(
            video_path=video_path,
            segs=segs,
            fps=fps,
            face_backend=face_backend,
            job_dir=job_dir,
            logger=logger,
        )
        segs = [s for s in segs if (s.end - s.start) > 0.15]
        if not segs:
            logger("No segments left after zero-face boundary trimming.")
            return False

        logger("â”€â”€ Planning segments â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        classified_segments = _classify_segments_with_windows(
            job_dir=job_dir,
            video_path=video_path,
            segs=segs,
            templates=templates,
            scene_to_template=scene_to_template,
            speaker_db=speaker_db,
            diar_timeline=diar_timeline,
            cuts=cuts,
            face_backend=face_backend,
            source_w=src_w,
            source_h=src_h,
            logger=logger,
        )
        if has_wide:
            speaker_map = _build_diarization_speaker_map(templates, classified_segments, diar_timeline, logger)
            routed_diar_timeline = _apply_speaker_map_to_timeline(diar_timeline, speaker_map)
        else:
            speaker_map = {}
            routed_diar_timeline = []
        assign_template_votes_from_classified_segments(templates, classified_segments, routed_diar_timeline, logger)
        try:
            write_json(job_dir / DIAR_DIR_NAME / "speaker_map.json", speaker_map)
            save_template_registry(job_dir, templates, speaker_db, cuts, scene_to_template, manual_templates=templates if manual_templates else None)
        except Exception:
            pass

        plans: List[SegmentPlan] = []
        for i, (seg, cls) in enumerate(zip(segs, classified_segments)):
            template = template_by_id.get(cls.template_id, templates[0])
            plan = plan_segment(
                video_path=video_path,
                seg_index=i,
                seg_start=seg.start,
                seg_end=seg.end,
                template=template,
                speaker_db=speaker_db,
                diar_timeline=routed_diar_timeline,
                face_backend=face_backend,
                source_w=src_w,
                source_h=src_h,
                logger=logger,
            )
            # FIX 10: write structured segment analysis log entry at planning time
            _kf_total = len(plan.keyframes)
            _kf_faces = sum(1 for k in plan.keyframes if k.confidence > 0.15)
            _kf_rate = (100.0 * _kf_faces / max(1, _kf_total))
            _tmpl_dname = template.display_name or f"template_{plan.template_id:02d}"
            _write_segment_analysis(job_dir, plan, _tmpl_dname, _kf_faces, _kf_total, _kf_rate)
            plans.append(plan)

        write_json(job_dir / KEYFRAME_FILE, [
            {
                "index": p.index,
                "start": p.start,
                "end": p.end,
                "template_id": p.template_id,
                "speaker": p.speaker,
                "crop_w": p.crop_w,
                "crop_h": p.crop_h,
                "target_slot": p.target_slot,
                "crop_expr_x": p.crop_expr_x,
                "crop_expr_y": p.crop_expr_y,
                "keyframes": _serialize_keyframes(p.keyframes),
            }
            for p in plans
        ])
        _export_debug_artifacts(job_dir, templates, plans, speaker_db, cuts)

        # Release GPU lock before heavy FFmpeg render starts.
        # Face backend stays in VRAM as singleton for next clip to reuse.
        if gpu_lock is not None:
            try:
                gpu_lock.release()
            except Exception:
                pass
            gpu_lock = None  # Prevent double release in finally block

        # FIX 5 â€” Rolling-delete: each segment temp file is deleted immediately
        # after concat succeeds, so peak disk usage = one segment at a time.
        _t0_render = time.time()
        logger(f"[RENDER] Starting render of {len(plans)} segments...")
        tmp_dir = job_dir / "segments"
        ensure_dir(tmp_dir)
        clips: List[str] = []
        render_ok_count = 0
        render_fail_count = 0
        for p in plans:
            _t0_seg = time.time()
            seg_dur = p.end - p.start
            dname = ""
            tmpl_obj = next((t for t in templates if t.template_id == p.template_id), None)
            if tmpl_obj:
                dname = tmpl_obj.display_name or f"template_{p.template_id:02d}"
            logger(
                f"[RENDER] seg#{p.index + 1}/{len(plans)} dur={seg_dur:.1f}s "
                f"template={dname}({p.template_id}) speaker={p.speaker} "
                f"slot={p.target_slot} initial_crop=({p.crop_expr_x},{p.crop_expr_y})"
            )
            tmp = tmp_dir / f"seg_{p.index:04d}.mp4"
            # _ffmpeg_render_segment handles all fallback attempts internally.
            ok, err = _ffmpeg_render_segment(video_path, p, str(tmp), ffmpeg_bin, logger)

            seg_elapsed = time.time() - _t0_seg
            if ok and tmp.exists() and tmp.stat().st_size >= MIN_CLIP_BYTES:
                clips.append(str(tmp))
                sz_mb = tmp.stat().st_size / (1024 * 1024)
                logger(f"  [OK] seg#{p.index+1}/{len(plans)} rendered in {seg_elapsed:.1f}s | {sz_mb:.1f} MB")
                render_ok_count += 1
            else:
                logger(f"  [FAIL] seg#{p.index+1}/{len(plans)} in {seg_elapsed:.1f}s :: {err[-220:]}")
                render_fail_count += 1

        render_elapsed = time.time() - _t0_render
        logger(f"[TIMER] Render phase done in {render_elapsed:.1f}s | OK={render_ok_count} FAIL={render_fail_count}/{len(plans)}")

        if not clips:
            logger("No segments rendered.")
            return False

        # Crossfade scene-cut joins by default (CLIP_SEGMENT_XFADE). Each join
        # overlaps by XFADE_FRAMES/fps, so the rendered clip is shorter than the
        # sum of its segments; captions compensate via the rendered-duration
        # manifest (see _write_render_duration_manifest / captioner localisation).
        if bool(_pipeline_config_value("CLIP_SEGMENT_XFADE", True)):
            ok, err = _ffmpeg_xfade_concat(clips, out_path, ffmpeg_bin, fps=fps)
        else:
            ok, err = _ffmpeg_hardcut_concat(clips, out_path, ffmpeg_bin, fps=fps)
        # FIX 5: Delete segment temp files immediately after concat, freeing disk space.
        for cl in clips:
            safe_remove(cl)
        try:
            tmp_dir.rmdir()   # remove the segments dir if now empty
        except Exception:
            pass
        if ok:
            out_sz = Path(out_path).stat().st_size / (1024*1024) if Path(out_path).exists() else 0
            logger(f"[DONE] Output ready -> {out_path}  ({out_sz:.1f} MB)")
            logger(f"[DONE] Segment temp files deleted (FIX5 bypass)")
            return True
        logger(f"Concat failed: {err[-500:]}")
        return False
    finally:
        if gpu_lock is not None:
            try:
                gpu_lock.release()
            except Exception:
                pass
        # Face backend intentionally NOT closed here — singleton stays in VRAM
        # for the next clip to reuse.  cleanup_gpu_resources() should only be
        # called once at the very end of the full extraction pipeline.


# ======================================================================
# GUI
# ======================================================================
# GUI
# ======================================================================
# ======================================================================
# Pipeline adapter
# ======================================================================
# The code above is the non-GUI Clip Tool v4 engine.  The functions below are
# intentionally thin: they adapt the existing pipeline call shape to the v4
# process_clip() entry point without changing the v4 planning/render logic.

_PIPELINE_FACE_BACKEND = None


def _pipeline_config_value(name: str, default: Any = None) -> Any:
    try:
        import config as _pipeline_config  # type: ignore
        return getattr(_pipeline_config, name, default)
    except Exception:
        return default


def _pipeline_log_adapter(logger: Any):
    def _log(text: str) -> None:
        try:
            if hasattr(logger, "info"):
                logger.info(text)
            elif callable(logger):
                logger(text)
            else:
                print(text)
        except Exception:
            try:
                print(text)
            except Exception:
                pass
    return _log


def _first_existing_path(*candidates: Any) -> str:
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and Path(text).exists():
            return text
    return ""


def _pipeline_manual_templates_path(job_dir: str | Path, explicit: Optional[str] = None) -> Optional[str]:
    job = Path(job_dir)
    path = _first_existing_path(
        explicit,
        _pipeline_config_value("V4_MANUAL_TEMPLATES_PATH", ""),
        job / "manual_templates.json",
        job / "templates" / "manual_templates.json",
        job / "files" / "manual_templates.json",
    )
    return path or None


def _pipeline_diarization_path(job_dir: str | Path, explicit: Optional[str] = None) -> str:
    job = Path(job_dir)
    return _first_existing_path(
        explicit,
        _pipeline_config_value("V4_DIARIZATION_PATH", ""),
        job / DIAR_FILE,
        job / DIAR_DIR_NAME / DIAR_FILE,
    )


def _configure_pipeline_resolution(target_resolution: Optional[Tuple[int, int]] = None) -> None:
    global TARGET_W, TARGET_H
    if target_resolution is None:
        target_resolution = _pipeline_config_value("VERTICAL_RESOLUTION", (TARGET_W, TARGET_H))
    try:
        width, height = target_resolution
        TARGET_W = int(width)
        TARGET_H = int(height)
    except Exception:
        pass


def _clear_pipeline_clip_cache(out_path: str | Path, logger: Optional[Logger] = None) -> None:
    job_dir = Path(out_path).with_suffix("").with_name(Path(out_path).stem + "_job")
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
            if logger:
                logger(f"[CACHE] Cleared per-clip v4 job cache: {job_dir}")
        except Exception as e:
            if logger:
                logger(f"[CACHE] Could not clear per-clip v4 job cache {job_dir}: {e}")


def render_pipeline_clip_v4(
    video_path: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
    pipeline_job_dir: str | Path,
    logger: Any = None,
    speaker_filter: Optional[str] = None,
    manual_templates_path: Optional[str] = None,
    diarization_path: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    diar_py: Optional[str] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    clear_template_cache: Optional[bool] = None,
    gpu_lock: Optional[threading.Semaphore] = None,
) -> bool:
    """Render one clips_plan segment with the Clip Tool v4 engine."""
    _configure_pipeline_resolution(target_resolution)
    v4_logger = Logger(log_fn=_pipeline_log_adapter(logger) if logger is not None else None)

    if speaker_filter is None:
        speaker_filter = str(_pipeline_config_value("V4_SPEAKER_FILTER", "continuous") or "continuous")
    if ffmpeg_bin is None:
        ffmpeg_bin = str(_pipeline_config_value("FFMPEG_PATH", DEFAULT_FFMPEG) or DEFAULT_FFMPEG)
    if diar_py is None:
        diar_py = str(_pipeline_config_value("V4_DIAR_PYTHON", DEFAULT_DIAR_PY) or DEFAULT_DIAR_PY)
    manual_templates_path = _pipeline_manual_templates_path(pipeline_job_dir, manual_templates_path)
    diarization_path = _pipeline_diarization_path(pipeline_job_dir, diarization_path)

    if clear_template_cache is None:
        clear_template_cache = bool(_pipeline_config_value("V4_CLEAR_TEMPLATE_CACHE_EACH_RUN", True))
    if clear_template_cache:
        _clear_pipeline_clip_cache(out_path, v4_logger)

    return process_clip(
        video_path=str(video_path),
        diar_path=str(diarization_path or ""),
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        speaker_filter=speaker_filter,
        out_path=str(out_path),
        ffmpeg_bin=str(ffmpeg_bin),
        diar_py=str(diar_py),
        logger=v4_logger,
        manual_templates_path=manual_templates_path,
        gpu_lock=gpu_lock,
    )


def _get_face_detector(logger: Any = None):
    """Compatibility hook used by other pipeline modules such as tts_hook.py."""
    global _PIPELINE_FACE_BACKEND
    if _PIPELINE_FACE_BACKEND is None:
        _PIPELINE_FACE_BACKEND = create_face_backend(
            Logger(log_fn=_pipeline_log_adapter(logger) if logger is not None else None)
        )
    return _PIPELINE_FACE_BACKEND


def _detect_faces(bgr: np.ndarray, detector: Any = None) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning dicts from the v4 FaceObs objects."""
    backend = detector or _get_face_detector()
    obs = backend.detect(bgr) if hasattr(backend, "detect") else []
    faces: List[Dict[str, Any]] = []
    for face in obs:
        if hasattr(face, "bbox"):
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
        else:
            x1, y1, x2, y2 = float(face.x1), float(face.y1), float(face.x2), float(face.y2)
        faces.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": float(face.conf),
            "cx": float(face.cx),
            "area": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
            "yaw": float(face.yaw or 0.0),
            "embedding": face.embedding,
        })
    return faces


def compute_podcast_crop(*args: Any, **kwargs: Any) -> str:
    """
    Legacy compatibility only.  The extractor now calls render_pipeline_clip_v4(),
    which produces the actual v4 keyframes and render.  This fallback returns a
    centered crop expression for any older caller that still imports it.
    """
    source_w = float(kwargs.get("src_w") or kwargs.get("source_w") or (args[0] if args else 1920))
    target_w = float(kwargs.get("out_w") or _pipeline_config_value("VERTICAL_RESOLUTION", (1080, 1920))[0])
    target_h = float(kwargs.get("out_h") or _pipeline_config_value("VERTICAL_RESOLUTION", (1080, 1920))[1])
    source_h = float(kwargs.get("src_h") or kwargs.get("source_h") or (args[1] if len(args) > 1 else 1080))
    crop_w = min(source_w, source_h * (target_w / max(1.0, target_h)))
    return f"{max(0.0, (source_w - crop_w) / 2.0):.1f}"


def cleanup_gpu_resources() -> None:
    try:
        global _PIPELINE_FACE_BACKEND
        _PIPELINE_FACE_BACKEND = None
    except Exception:
        pass
    try:
        if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if HAS_CUPY and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

