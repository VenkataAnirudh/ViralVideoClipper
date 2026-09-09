"""
Branding Engine  —  reusable post-captioning finisher
=====================================================
Consumes the FINISHED captioned clips (clips/clip_NN_captioned*.mp4) and applies
client-specific branding in ONE ffmpeg pass per clip:

  • Logo overlay (top-right, sized as % of video WIDTH — resolution independent)
  • Hook banner in the first N seconds (full hook_phrase, auto-wrapped, animated)
  • Background-music engine (source / assignment / segment modes)
  • Speech-aware ducking (sidechain), loudness normalization (-14 LUFS), fades
  • Color grade, subtle hook punch-in, retention progress bar, @handle watermark
  • End CTA freeze-frame, cover/thumbnail export, per-clip upload-kit sidecar
  • Fast dry-run (first N seconds only) to preview placement

Every feature is independently toggleable via the config dict so a single bad
filter can be switched off without code changes. All builders are defined ONCE
and called per clip (reusable across clips, jobs and clients).

Nothing here mutates the source clips; outputs land in clips/branded/.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Project imports (best-effort; the tool still works if they fail) ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config as _cfg  # type: ignore
    FFMPEG = getattr(_cfg, "FFMPEG_PATH", "ffmpeg")
    FFPROBE = getattr(_cfg, "FFPROBE_PATH", "ffprobe")
    GPU_FFMPEG = getattr(_cfg, "GPU_FFMPEG_PATH", FFMPEG)
except Exception:  # pragma: no cover
    _cfg = None
    FFMPEG = "ffmpeg"
    FFPROBE = "ffprobe"
    GPU_FFMPEG = "ffmpeg"

try:
    from pipeline.captioner import _analyze_music_volume_curve, _pick_music_offset  # type: ignore
except Exception:  # pragma: no cover
    _analyze_music_volume_curve = None
    _pick_music_offset = None

try:
    from pipeline.downloader import download_audio_track  # type: ignore
except Exception:  # pragma: no cover
    download_audio_track = None

try:
    import cv2  # face-aware banner placement
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont  # rounded-border banner PNG
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


# ─────────────────────────────────────────────────────────────────────────────
# Default configuration (the UI writes branding_config.json over these defaults)
# ─────────────────────────────────────────────────────────────────────────────
def default_config() -> Dict[str, Any]:
    return {
        "encoder": "x264",              # "x264" (safe) | "nvenc" (GTX 1650)
        "input_variant": "captioned",  # which *_captioned*.mp4 flavour to brand
        "only_clips": [],               # [] = all; else e.g. [1,2,3] or ["clip_01"] — render just these
        "output_tag": "branded",        # output name = clip_NN_<tag>.mp4 (use "bckg" etc. to keep variants)
        "render_selected_only": True,   # UI: Run renders only the gallery selection (guarded when empty)

        "logo": {
            "enabled": False,          # opt-in: never burned without selection
            "path": "",                # absolute or project-relative png (with alpha); empty = auto-detect branding/logo.png
            "scale_pct": 24.0,         # % of video WIDTH  (approved trial size)
            "margin_pct": 1.0,         # % of video WIDTH from the corner — tiny gap; safe zones do NOT push the logo
            "opacity": 1.0,            # 0..1
            "fade_in": 0.0,            # seconds (0 = hard burn-in, no fade)
            "position": "top_right",   # top_right | top_left | bottom_right | bottom_left
        },

        "banner": {
            "enabled": False,          # opt-in (default OFF)
            "text_source": "hook_phrase",   # hook_phrase | youtube_title (per-clip override wins)
            "font": "assets/fonts/BarlowCondensed-Black.ttf",
            "color": "#FFC83D",             # fallback text colour when auto_color is off
            "auto_color": True,             # per-clip: sample bg -> light font on dark bg / dark on light
            "color_dark_bg": "#F2ECE0",     # soft cream (used when the banner bg is dark)
            "color_light_bg": "#1A1A1A",    # near-black (used when the banner bg is light)
            "border_color": "#FFF8C8",      # pale yellow border (separate from text)
            "border": True,                 # draw the rounded border box (on/off)
            "border_thickness_pct": 0.40,   # % of height  — thin line
            "corner_radius_pct": 2.4,       # % of height
            "pad_pct": 2.2,                 # inner padding, % of height (even on all sides)
            "font_pct": 8.594,              # MAX font, % of height — auto-fit shrinks to fit (165px @1920h, same as subtitle default)
            "font_min_pct": 3.2,            # floor font, % of height
            "line_spacing": 1.07,           # line-to-line step as a multiple of fs (tight)
            "max_width_pct": 86,            # legacy wrap cap, % of width
            "side_margin_pct": 6.0,         # min clear space each side, % of width
            "bottom_forbidden_pct": 20.0,   # box may NOT enter the bottom N% of the frame
            "duration": 3.0,                # seconds shown from clip start
            "fade": 0.3,                    # in/out seconds
            "placement": "subtitle",        # auto | subtitle | top | mid | bottom
            "uppercase": True,              # ALL CAPS hook
            "clean_hook": True,             # first `duration`s from RAW clip (no burnt subs)
            "sub_top_pct": 48.0,            # subtitle zone top
            "sub_bottom_pct": 65.0,         # subtitle zone bottom
        },

        "music": {
            "enabled": True,
            "source": "folder",        # folder | single | youtube
            "folder": "music/viral",   # curated viral library (frequently used)
            "single_path": "",
            "youtube_url": "",
            "assignment": "random",    # same | random | roundrobin
            "no_repeat_consecutive": True,
            "segment_mode": "fixed",   # fixed | loudness  (multipart kept for old configs only)
            "fixed_offset": 0.0,       # play from the start of the track by default
            "multipart_count": 3,      # legacy — no longer exposed in the UI
            "volume": 0.15,            # 0..1  (15% — frequently used level)
            "fade": 0.6,               # seconds in AND out (fixed; not exposed in the UI)
            "duck": False,             # sidechain duck under speech (default OFF)
            "duck_threshold": 0.05,
            "duck_ratio": 8.0,
            "loudnorm": True,          # normalise final mix to -14 LUFS (kept on)
            "end_ramp_volume": 0.5,    # music swells to this across the end card
        },

        # Extra upgrades — all toggleable
        "color_grade": {"enabled": True, "contrast": 1.08, "brightness": 0.05,
                         "saturation": 1.08, "gamma": 1.06, "vibrance": 0.18,
                         "clarity": 0.30, "sharpen": 0.8},
        "punch_in":    {"enabled": False, "amount": 1.06, "duration": 2.5},   # extra: off
        "progress_bar":{"enabled": False, "height_pct": 1.1, "color": "white@0.85"},  # extra: off
        "watermark":   {"enabled": False, "text": "@yourhandle",
                         "font": "assets/fonts/Montserrat-ExtraBold.ttf",
                         "font_pct": 3.2, "color": "white@0.75", "position": "bottom_left"},
        "end_card":    {"enabled": True, "duration": 3.0, "text": "FOLLOW FOR WISDOM",
                         "font": "assets/fonts/Anton-Regular.ttf", "font_pct": 9.0,
                         "color": "white", "one_word_per_line": True},
        "thumbnail":   {"enabled": False, "at": "hook"},  # extra: off (opt-in)
        "upload_kit":  {"enabled": False},                # extra: off (opt-in)

        # TTS hook generation (same voice/bass/tail settings as the main
        # pipeline's config.py). When enabled, clips missing a *_hooked
        # variant get their intro generated + prepended before branding.
        "tts_hook":    {"enabled": False, "regenerate": False},
        "safe_zones":  {"enabled": True, "platform": "reels"},  # reels|tiktok|shorts|none
        "dry_run":     {"enabled": False, "seconds": 5.0},

        # Per-clip overrides keyed by clip_name; any field above can be overridden
        # plus: {"skip": true, "music_path": "...", "music_volume": 0.1,
        #        "music_offset": 12, "banner": false, "logo": false,
        #        "trim_end_s": 41.633}   # frame-accurate end cut (UI slider)
        "clips": {},
    }


def _deep_merge(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    cfg = default_config()
    p = Path(path)
    if p.exists():
        try:
            cfg = _deep_merge(cfg, json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────
def probe_duration(path: str) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
        return float((r.stdout or "0").strip())
    except Exception:
        return 0.0


def probe_dims(path: str) -> Tuple[int, int]:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
        w, h = (r.stdout or "0x0").strip().split("x")[:2]
        return int(w), int(h)
    except Exception:
        return 1080, 1920


def _abs(path: str) -> str:
    """Resolve a path relative to the project root if not absolute."""
    if not path:
        return ""
    p = Path(path)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


def _ff_path(p: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument."""
    return str(p).replace("\\", "/").replace(":", "\\:")


def _esc_text(s: str) -> str:
    """Escape literal text for drawtext text='...'."""
    s = str(s).replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
    s = s.replace("%", "\\%")
    return s


def _encoder_args(encoder: str) -> Tuple[str, List[str]]:
    """Return (ffmpeg_binary, video-encode args)."""
    if str(encoder).lower() == "nvenc":
        return GPU_FFMPEG, ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                            "-cq", "20", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return FFMPEG, ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p"]


# ─────────────────────────────────────────────────────────────────────────────
# Safe zones — keep overlays clear of platform UI (right action rail, bottom)
# ─────────────────────────────────────────────────────────────────────────────
def safe_insets(platform: str, w: int, h: int) -> Dict[str, int]:
    """Return pixel insets {top,bottom,left,right} to avoid platform chrome."""
    presets = {
        "reels":  (0.06, 0.18, 0.04, 0.06),
        "tiktok": (0.06, 0.20, 0.04, 0.14),
        "shorts": (0.06, 0.16, 0.04, 0.06),
        "none":   (0.0, 0.0, 0.0, 0.0),
    }
    t, b, l, r = presets.get(str(platform).lower(), presets["reels"])
    return {"top": int(h * t), "bottom": int(h * b), "left": int(w * l), "right": int(w * r)}


# ─────────────────────────────────────────────────────────────────────────────
# Text wrapping for the banner (drawtext does not auto-wrap)
# ─────────────────────────────────────────────────────────────────────────────
def wrap_text(text: str, max_chars: int) -> str:
    words = str(text or "").split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines) if lines else ""


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO filter builders — each returns a list of simple-filter strings
# ─────────────────────────────────────────────────────────────────────────────
def f_punch_in(cfg: Dict, w: int, h: int) -> List[str]:
    if not cfg.get("enabled"):
        return []
    amt = float(cfg.get("amount", 1.06))
    dur = float(cfg.get("duration", 2.5))
    # zoom = amt -> 1.0 across the first `dur`s, then hold at 1.0
    z = f"if(lt(t,{dur}),{amt}-({amt}-1)*t/{dur},1)"
    cw = f"trunc(iw/({z})/2)*2"
    ch = f"trunc(ih/({z})/2)*2"
    return [f"crop=w='{cw}':h='{ch}':x='(iw-ow)/2':y='(ih-oh)/2'", f"scale={w}:{h}"]


def f_color_grade(cfg: Dict) -> List[str]:
    """Versatile, product-flattering grade: gentle contrast + skin-safe vibrance
    for pop, then large-radius 'clarity' (midtone local contrast) and a fine
    sharpen pass to lift perceived sharpness/clarity — without colour casts or
    halos. Every knob is configurable so it stays neutral on any footage."""
    if not cfg.get("enabled"):
        return []
    out = [f"eq=contrast={cfg.get('contrast', 1.08)}"
           f":brightness={cfg.get('brightness', 0.05)}"
           f":saturation={cfg.get('saturation', 1.08)}"
           f":gamma={cfg.get('gamma', 1.06)}"]
    vib = float(cfg.get("vibrance", 0.0) or 0.0)
    if vib > 0:
        # boosts muted colours more than already-saturated ones -> natural pop
        out.append(f"vibrance=intensity={vib}")
    clarity = float(cfg.get("clarity", 0.0) or 0.0)
    if clarity > 0:
        # large-radius luma unsharp == 'clarity' (midtone local contrast)
        out.append(f"unsharp=13:13:{clarity}:5:5:0.0")
    sharpen = float(cfg.get("sharpen", 0.0) or 0.0)
    if sharpen > 0:
        # fine-radius luma unsharp == edge sharpness
        out.append(f"unsharp=5:5:{sharpen}:5:5:0.0")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HOOK BANNER — rendered as a transparent PNG (golden text + rounded golden
# border, NO fill — like the user's mock) then overlaid, face-aware, for N sec.
# ffmpeg can't draw rounded corners, so Pillow builds the chip.
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_pixels(draw, text: str, font, max_w: int) -> List[str]:
    lines, cur = [], ""
    for word in str(text).split():
        trial = (cur + " " + word).strip()
        if not cur or draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [str(text)]


def render_banner_png(text: str, out_png: str, w: int, h: int, cfg: Dict,
                      text_color: Optional[str] = None,
                      max_box_h: Optional[int] = None,
                      ) -> Optional[Tuple[str, int, int]]:
    """Render the hook chip to a transparent PNG. Returns (path, box_w, box_h).

    The font is auto-MAXIMISED: it grows to the largest size (capped by
    ``font_pct``) whose wrapped text still fits the side-safe width AND, when
    ``max_box_h`` is given, the allowed vertical band. Padding is even on every
    side; the rounded border is optional via ``border``. ``text_color`` (from
    the per-clip auto-contrast pick) overrides the static ``color``.
    """
    if Image is None or not text:
        return None
    text = text.upper()   # always all-caps by design

    text_color = text_color or cfg.get("color", "#FFC83D")
    border_on = bool(cfg.get("border", True))
    border_color = cfg.get("border_color", "#FFF8C8")
    font_path = _abs(cfg.get("font", "assets/fonts/BarlowCondensed-Black.ttf"))

    pad = int(h * float(cfg.get("pad_pct", 2.2)) / 100.0)
    border = int(max(2, h * float(cfg.get("border_thickness_pct", 0.40)) / 100.0)) if border_on else 0
    radius = int(h * float(cfg.get("corner_radius_pct", 2.4)) / 100.0)

    # Side-safe usable width (honours side_margin_pct AND the legacy max_width_pct).
    side_margin = float(cfg.get("side_margin_pct", 6.0))
    usable_w = int(w * (1.0 - 2.0 * side_margin / 100.0))
    legacy_w = int(w * float(cfg.get("max_width_pct", 86)) / 100.0)
    max_text_w = max(40, min(usable_w, legacy_w) - 2 * (pad + border))
    inner_h_limit = max(1, int(max_box_h) - 2 * (pad + border)) if max_box_h else None

    fs_hi = max(20, int(h * float(cfg.get("font_pct", 8.5)) / 100.0))
    fs_lo = max(14, int(h * float(cfg.get("font_min_pct", 3.2)) / 100.0))

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    line_spacing = float(cfg.get("line_spacing", 1.07))

    def _layout(fs: int):
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            font = ImageFont.load_default()
        lines = _wrap_pixels(probe, text, font, max_text_w)
        asc, desc = font.getmetrics()
        line_h = asc + desc
        advance = max(1, int(fs * line_spacing))   # tight step, independent of the font box
        tw = int(max(probe.textlength(l, font=font) for l in lines))
        th = advance * (len(lines) - 1) + line_h   # enclose the last line's full box
        return font, lines, line_h, advance, tw, th

    # Grow the font as large as fits both the width and (if given) the band.
    chosen = None
    fs = fs_hi
    while fs >= fs_lo:
        cand = _layout(fs)
        _f, _l, _lh, _g, tw, th = cand
        if tw <= max_text_w and (inner_h_limit is None or th <= inner_h_limit):
            chosen = cand
            break
        fs -= 2
    if chosen is None:
        chosen = _layout(fs_lo)   # floor; may overflow a hair but never blank
    font, lines, line_h, advance, text_w, text_h = chosen

    box_w = text_w + 2 * (pad + border)
    box_h = text_h + 2 * (pad + border)
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    if border > 0:
        off = border / 2.0
        dr.rounded_rectangle([off, off, box_w - off - 1, box_h - off - 1],
                             radius=radius, outline=border_color, width=border)

    # Even top/bottom padding: advance by a CONSISTENT line_h+gap (matches text_h),
    # so the block is symmetric in the box (no drift from per-line bboxes).
    cx = box_w // 2
    y = border + pad
    for l in lines:
        dr.text((cx + 2, y + 2), l, font=font, fill=(0, 0, 0, 160), anchor='mt')
        dr.text((cx, y), l, font=font, fill=text_color, anchor='mt')
        y += advance

    try:
        img.save(out_png)
        return out_png, box_w, box_h
    except Exception:
        return None


def detect_face_box(video_path: str, secs: float = 1.2
                    ) -> Optional[Tuple[int, int, int, int]]:
    """Union face box across the first `secs` (Haar cascade). None if no cv2/face."""
    if cv2 is None:
        return None
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = max(1, int(fps * secs))
        step = max(1, int(fps / 5))  # ~5 samples per second
        boxes = []
        for i in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            if i % step:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            fh = fr.shape[0]
            found = cascade.detectMultiScale(
                gray, 1.2, 5, minSize=(int(fh * 0.06), int(fh * 0.06)))
            for (x, y, bw, bh) in found:
                boxes.append((int(x), int(y), int(bw), int(bh)))
        cap.release()
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        return (x0, y0, x1 - x0, y1 - y0)
    except Exception:
        return None


def pick_contrast_color(video_path: str, band_top: int, band_bot: int, cfg: Dict,
                        secs: float = 1.2) -> str:
    """Sample the banner background band luminance over the first `secs` and pick
    a font colour for contrast: dark bg -> light font, light bg -> dark font.
    Falls back to the light tone (or the static colour if auto_color is off)."""
    light = cfg.get("color_dark_bg", "#F2ECE0")   # used on a DARK background
    dark = cfg.get("color_light_bg", "#1A1A1A")   # used on a LIGHT background
    if not bool(cfg.get("auto_color", True)):
        return cfg.get("color", "#FFC83D")
    if cv2 is None:
        return light
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = max(1, int(fps * secs))
        step = max(1, int(fps / 4))
        vals = []
        for i in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            if i % step:
                continue
            hh = fr.shape[0]
            y0 = max(0, min(hh - 1, band_top))
            y1 = max(y0 + 1, min(hh, band_bot))
            crop = fr[y0:y1, :, :]
            if crop.size == 0:
                continue
            # BGR -> perceived luma (Rec.601)
            luma = (0.114 * crop[:, :, 0].mean()
                    + 0.587 * crop[:, :, 1].mean()
                    + 0.299 * crop[:, :, 2].mean())
            vals.append(float(luma))
        cap.release()
        if not vals:
            return light
        return dark if (sum(vals) / len(vals)) >= 140.0 else light
    except Exception:
        return light


def pick_banner_y(h: int, box_h: int, face_box, placement: str,
                  sub_top: int, sub_bottom: int, insets: Dict[str, int],
                  bottom_limit: Optional[int] = None) -> int:
    """Choose a vertical position that avoids the face and the subtitle band.
    `bottom_limit` (if given) is an absolute y the box bottom must not cross —
    used to keep the banner out of the forbidden bottom band."""
    margin = int(h * 0.012)
    safe_top = insets.get("top", 0) + margin
    safe_bottom = h - insets.get("bottom", 0) - margin
    if bottom_limit is not None:
        safe_bottom = min(safe_bottom, int(bottom_limit))
    f_top = face_box[1] if face_box else int(h * 0.20)
    f_bot = (face_box[1] + face_box[3]) if face_box else int(h * 0.46)

    if placement == "subtitle":
        center = (sub_top + sub_bottom) / 2.0
        return int(max(safe_top, min(center - box_h / 2.0, safe_bottom - box_h)))
    if placement == "top":
        return max(safe_top, int(h * 0.07))
    if placement == "bottom":
        return min(safe_bottom - box_h, int(h * 0.84))
    if placement == "mid":
        center = (f_bot + sub_top) / 2
        return int(max(safe_top, min(center - box_h / 2, sub_top - box_h)))

    # auto: largest gap that fits, preferring top -> mid -> bottom on ties
    gaps = [
        ("top", safe_top, min(f_top, sub_top)),
        ("mid", f_bot, sub_top),
        ("bottom", sub_bottom, safe_bottom),
    ]
    best = None
    for _name, a, b in gaps:
        room = b - a
        if room >= box_h + margin and (best is None or room > best[0] + 1):
            best = (room, a, b)
    if best is None:
        return max(safe_top, int(h * 0.06))
    _room, a, b = best
    return int(a + (b - a - box_h) / 2)


def f_progress(cfg: Dict, total: float, w: int, h: int) -> List[str]:
    if not cfg.get("enabled") or total <= 0:
        return []
    bh = max(2, int(h * float(cfg.get("height_pct", 1.1)) / 100.0))
    color = cfg.get("color", "white@0.85")
    return [f"drawbox=x=0:y=ih-{bh}:w='iw*min(t/{total},1)':h={bh}:color={color}:t=fill"]


def f_watermark(cfg: Dict, w: int, h: int, insets: Dict[str, int]) -> List[str]:
    if not cfg.get("enabled") or not cfg.get("text"):
        return []
    fs = max(12, int(h * float(cfg.get("font_pct", 3.2)) / 100.0))
    font = _ff_path(_abs(cfg.get("font", "assets/fonts/Montserrat-ExtraBold.ttf")))
    pos = str(cfg.get("position", "bottom_left"))
    m = max(insets.get("left", 0), int(w * 0.04))
    by = h - insets.get("bottom", 0) - fs - 8
    x = f"{m}" if "left" in pos else f"w-text_w-{m}"
    y = f"{max(insets.get('top',0)+4, 8)}" if "top" in pos else f"{by}"
    return [f"drawtext=fontfile='{font}':text='{_esc_text(cfg['text'])}'"
            f":fontcolor={cfg.get('color','white@0.75')}:fontsize={fs}:x={x}:y={y}"]


def _append_end_card_graph(fc_parts: List[str], cur: str, cfg: Dict,
                           total: float, w: int, h: int) -> str:
    """Freeze the last frame and fade it to black, then composite a GLOWING CTA:
    words stacked one-per-line, a soft gblur halo behind the crisp text, and a
    translucent->opaque alpha so the text is see-through while the frame is visible
    but fully opaque on the black. Appends the sub-graph to `fc_parts` and returns
    the new video label.

    The transparent text canvas is derived by splitting the (faded) base stream
    and zeroing its alpha, so it inherits the source's exact fps/timebase — a
    standalone `color` source at a mismatched rate deadlocks the overlay. Font size
    is static on purpose: an animated `fontsize` expression crashes libx264 partway
    through the encode."""
    if not cfg.get("enabled"):
        fc_parts.append(f"{cur}null[ecout]")
        return "[ecout]"

    dur = float(cfg.get("duration", 2.0))
    fs = max(20, int(h * float(cfg.get("font_pct", 9.0)) / 100.0))
    font = _ff_path(_abs(cfg.get("font", "assets/fonts/Anton-Regular.ttf")))
    color = cfg.get("color", "white")
    raw = str(cfg.get("text", "FOLLOW FOR WISDOM")).upper()
    words = raw.split() if cfg.get("one_word_per_line", True) else [raw]

    T, D = f"{total:.3f}", f"{dur:.3f}"

    # 1) freeze + fade the frozen frame to black; keep a 2nd copy to build the
    #    transparent text layer from (inherits fps/timebase -> no overlay mismatch).
    fc_parts.append(f"{cur}tpad=stop_mode=clone:stop_duration={D},"
                    f"fade=t=out:st={T}:d={D}[ecfrm]")
    fc_parts.append("[ecfrm]split[ecbase][ecblank]")

    # 2) transparent canvas from the base copy + stacked words (translucent->opaque)
    alpha = f"min(1,0.55+0.45*(t-{T})/{D})"
    line_gap = int(fs * 0.30)
    n = max(1, len(words))
    block_h = n * fs + (n - 1) * line_gap
    y0 = (h - block_h) // 2
    layer = "[ecblank]format=rgba,colorchannelmixer=aa=0"
    for i, wtxt in enumerate(words):
        yy = y0 + i * (fs + line_gap)
        txt = _esc_text(wtxt)
        layer += (f",drawtext=fontfile='{font}':text='{txt}':fontcolor={color}"
                  f":fontsize={fs}:x=(w-text_w)/2:y={yy}"
                  f":alpha='{alpha}':enable='gte(t,{T})'")
    fc_parts.append(layer + "[ectxt]")

    # 3) soft glow: blur a copy at half-res (cheaper + softer), lift its brightness,
    #    and place it BEHIND the crisp text.
    sigma = max(3, int(fs * 0.06))
    fc_parts.append("[ectxt]split[ec1][ec2]")
    fc_parts.append(f"[ec2]scale=trunc(iw/2):trunc(ih/2),gblur=sigma={sigma}:steps=2,"
                    f"eq=brightness=0.06,scale={w}:{h}[ecglow]")
    fc_parts.append("[ecglow][ec1]overlay[ectg]")

    # 4) composite the glowing text onto the faded frame during the end card
    fc_parts.append(f"[ecbase][ectg]overlay=enable='gte(t,{T})'[ecout]")
    return "[ecout]"


def _trim_logo(path: str, tmp_dir: str) -> str:
    """Crop away transparent padding so the real logo fills the scaled box and
    sits flush in the corner. Falls back to the original path on any issue."""
    if Image is None or not tmp_dir:
        return path
    try:
        im = Image.open(path).convert("RGBA")
        bbox = im.split()[-1].getbbox()  # bounding box of non-transparent pixels
        if bbox and bbox != (0, 0, im.width, im.height):
            out = os.path.join(tmp_dir, "_logo_trimmed.png")
            im.crop(bbox).save(out)
            return out
    except Exception:
        pass
    return path


# Path to the logo.png bundled inside the branding/ folder (sits next to engine.py)
_BRANDING_DIR = Path(__file__).resolve().parent
_DEFAULT_LOGO = _BRANDING_DIR / "logo.png"


def logo_overlay(cfg: Dict, w: int, h: int, insets: Dict[str, int], in_index: int,
                 tmp_dir: str = "") -> Tuple[List[str], str, str]:
    """Return (input_args, prep_filter_with_labels, overlay_xy). Empty if disabled.

    If no logo path is configured the function automatically falls back to the
    ``logo.png`` file that lives in the same directory as this engine module
    (i.e. ``branding/logo.png``).  This makes the logo a zero-config burn-in
    that is always applied as long as ``logo.enabled`` is True.
    """
    if not cfg.get("enabled"):
        return [], "", ""
    path = _abs(cfg.get("path", ""))
    # ── Auto-fallback: use branding/logo.png when no path is configured ───────
    if not path or not os.path.exists(path):
        if _DEFAULT_LOGO.exists():
            path = str(_DEFAULT_LOGO)
        else:
            return [], "", ""
    path = _trim_logo(path, tmp_dir)
    lw = max(16, int(w * float(cfg.get("scale_pct", 7.0)) / 100.0))
    margin = int(w * float(cfg.get("margin_pct", 1.0)) / 100.0)
    op = max(0.0, min(1.0, float(cfg.get("opacity", 1.0))))
    fade = float(cfg.get("fade_in", 0.0) or 0.0)
    pos = str(cfg.get("position", "top_right"))
    # Logo is pinned to its corner with just the margin — platform safe-zone
    # insets are intentionally NOT added (they pushed the logo ~8% inward,
    # which read as a bug). Banner/watermark still respect safe zones.
    mx = margin
    my = margin
    x = f"W-w-{mx}" if "right" in pos else f"{mx}"
    y = f"{my}" if "top" in pos else f"H-h-{my}"
    chain = [f"scale={lw}:-1", "format=rgba"]
    if op < 1.0:
        chain.append(f"colorchannelmixer=aa={op}")
    if fade > 0:
        chain.append(f"fade=in:st=0:d={fade}:alpha=1")
    prep = f"[{in_index}:v]" + ",".join(chain) + "[logo]"
    return ["-i", path], prep, f"{x}:{y}"


# ─────────────────────────────────────────────────────────────────────────────
# MUSIC engine — track selection + segment offset(s) + bed pre-render
# ─────────────────────────────────────────────────────────────────────────────
def list_folder_tracks(folder: str) -> List[str]:
    d = Path(_abs(folder))
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)


class MusicPicker:
    """Resolves which track each clip gets, honouring assignment mode + no-repeat."""

    def __init__(self, cfg: Dict, job_dir: str, logger):
        self.cfg = cfg
        self.job_dir = job_dir
        self.logger = logger
        self._idx = 0
        self._last = None
        self._single: Optional[str] = None
        self.tracks: List[str] = []
        src = cfg.get("source", "folder")
        if src == "folder":
            self.tracks = list_folder_tracks(cfg.get("folder", "assets/audio"))
        elif src == "single":
            self._single = _abs(cfg.get("single_path", "")) or None
        elif src == "upload":
            self._single = _abs(cfg.get("single_path", "")) or None
        elif src == "youtube" and download_audio_track:
            try:
                self._single = download_audio_track(job_dir, cfg.get("youtube_url", ""), logger)
            except Exception as e:
                logger and logger.warning(f"YouTube music download failed: {e}")
                self._single = None

    def for_clip(self, clip_name: str, per_clip: Dict) -> Optional[str]:
        if per_clip.get("music_path"):
            return _abs(per_clip["music_path"])
        if self._single:
            return self._single
        if not self.tracks:
            return None
        mode = self.cfg.get("assignment", "random")
        if mode == "roundrobin":
            t = self.tracks[self._idx % len(self.tracks)]
            self._idx += 1
        elif mode == "same":
            t = self.tracks[0]
        else:  # random
            pool = self.tracks
            if self.cfg.get("no_repeat_consecutive") and len(pool) > 1 and self._last:
                pool = [x for x in pool if x != self._last] or pool
            t = random.choice(pool)
        self._last = t
        return t


def _loud_windows(track: str, clip_dur: float, count: int, logger) -> List[float]:
    """Return up to `count` distinct start-offsets at energetic windows."""
    offs: List[float] = []
    if _analyze_music_volume_curve:
        try:
            a = _analyze_music_volume_curve(track, logger)
            for c in a.get("candidates", [])[:count]:
                offs.append(max(0.0, float(c.get("start", 0.0))))
        except Exception:
            pass
    if not offs:
        td = probe_duration(track)
        step = max(0.0, td - clip_dur)
        offs = [random.uniform(0, step) if step > 0 else 0.0]
    return offs[:count]


def build_music_bed(track: str, clip_dur: float, mcfg: Dict, per_clip: Dict,
                    out_path: str, logger) -> Optional[str]:
    """Render a clip-length music bed honouring segment mode + fades + volume."""
    if not track or not os.path.exists(track) or clip_dur <= 0:
        return None
    vol = float(per_clip.get("music_volume", mcfg.get("volume", 0.12)))
    fade = float(mcfg.get("fade", 0.6))
    seg_mode = mcfg.get("segment_mode", "loudness")
    ff = FFMPEG

    af_tail = (f"volume={vol},afade=t=in:st=0:d={fade},"
               f"afade=t=out:st={max(0.0, clip_dur - fade):.3f}:d={fade},"
               f"atrim=0:{clip_dur:.3f},asetpts=N/SR/TB")

    try:
        if seg_mode == "multipart":
            n = max(2, int(mcfg.get("multipart_count", 3)))
            offs = _loud_windows(track, clip_dur, n, logger)
            part = clip_dur / len(offs)
            cmd = [ff, "-y"]
            for o in offs:
                cmd += ["-ss", f"{o:.3f}", "-t", f"{part + max(fade,0.4):.3f}", "-i", track]
            labels = "".join(f"[{i}:a]" for i in range(len(offs)))
            xfade = f"{labels}concat=n={len(offs)}:v=0:a=1[cat]"
            filt = f"{xfade};[cat]{af_tail}[out]"
            cmd += ["-filter_complex", filt, "-map", "[out]",
                    "-c:a", "aac", "-b:a", "192k", out_path]
        else:
            if seg_mode == "fixed":
                off = float(per_clip.get("music_offset", mcfg.get("fixed_offset", 0.0)))
            else:  # loudness
                if _pick_music_offset:
                    try:
                        off = float(_pick_music_offset(track, clip_dur, logger))
                    except Exception:
                        off = 0.0
                else:
                    off = float(per_clip.get("music_offset", 0.0))
            cmd = [ff, "-y", "-stream_loop", "-1", "-ss", f"{off:.3f}", "-t",
                   f"{clip_dur:.3f}", "-i", track,
                   "-af", af_tail, "-c:a", "aac", "-b:a", "192k", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           creationflags=_NO_WINDOW)
        if r.returncode == 0 and os.path.exists(out_path):
            return out_path
        logger and logger.warning(f"Music bed build failed: {(r.stderr or '')[-300:]}")
    except Exception as e:
        logger and logger.warning(f"Music bed exception: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip render
# ─────────────────────────────────────────────────────────────────────────────
def brand_one_clip(in_path: str, out_path: str, clip: Dict, cfg: Dict,
                   logger, tmp_dir: str, raw_path: str = "") -> bool:
    name = clip.get("clip_name", Path(in_path).stem)
    per = (cfg.get("clips", {}) or {}).get(name, {}) or {}
    if per.get("skip"):
        logger and logger.info(f"[{name}] skipped by config")
        return False

    w, h = probe_dims(in_path)
    total = probe_duration(in_path)
    if total <= 0:
        logger and logger.warning(f"[{name}] could not probe duration; skipping")
        return False

    # Frame-accurate per-clip end cut (clips.{name}.trim_end_s from the UI
    # slider). Applied as an input read limit; the re-encode makes the cut
    # exact, and everything downstream (banner window, end card, progress
    # bar, music bed) budgets against the trimmed length.
    in0_args: List[str] = []
    trim_end = float(per.get("trim_end_s", 0) or 0)
    if 3.0 <= trim_end < total - 0.01:
        total = trim_end
        in0_args = ["-t", f"{trim_end:.3f}"]
        logger and logger.info(f"[{name}] end-trim to {trim_end:.3f}s")
    elif trim_end > 0:
        # A sub-3s branded clip is never intentional — this is the slider
        # foot-gun (a stray drag once cut a 44s clip to 1s). Ignore it loudly.
        logger and logger.warning(
            f"[{name}] ignoring trim_end_s={trim_end:g} (must be >= 3s and "
            f"shorter than the clip); rendering full length")

    dry = cfg.get("dry_run", {})
    if dry.get("enabled"):
        total = min(total, float(dry.get("seconds", 5.0)))

    insets = safe_insets(cfg.get("safe_zones", {}).get("platform", "reels"), w, h) \
        if cfg.get("safe_zones", {}).get("enabled", True) else {"top": 0, "bottom": 0, "left": 0, "right": 0}

    # ── Banner text + clean-hook decision (decided before input wiring) ───────
    banner_cfg = dict(cfg.get("banner", {}))
    if not per.get("banner", True):
        banner_cfg["enabled"] = False
    # A *_hooked input already opens with the TTS hook intro (text + voice);
    # the banner chip would duplicate it and clean_hook would chop the intro
    # off by splicing raw frames over it — force both off for hooked inputs.
    if Path(in_path).stem.endswith("_hooked"):
        banner_cfg["enabled"] = False
    bsrc = banner_cfg.get("text_source", "hook_phrase")
    banner_text = (per.get("banner_text")
                   or (clip.get("youtube_title") if bsrc == "youtube_title" else None)
                   or clip.get("hook_phrase") or clip.get("title") or "")
    banner_on = bool(banner_cfg.get("enabled") and banner_text)
    bdur = float(banner_cfg.get("duration", 3.0))
    hook_dur = min(bdur, max(0.1, total - 0.05))
    # Clean hook: the first hook_dur seconds come from the RAW (caption-free) clip
    # so the banner can sit in the SUBTITLE slot without clashing with burnt
    # captions; the rest of the clip plays the captioned video (same frames).
    clean_hook = bool(banner_on and banner_cfg.get("clean_hook", True)
                      and raw_path and os.path.exists(raw_path))

    # ── Video base chain (punch-in + grade) ──────────────────────────────────
    vchain: List[str] = []
    vchain += f_punch_in(cfg.get("punch_in", {}), w, h)
    vchain += f_color_grade(cfg.get("color_grade", {}))

    # ── Inputs (0 = captioned/audio/body; 1 = raw hook source when clean) ─────
    inputs: List[str] = in0_args + ["-i", in_path]
    next_idx = 1
    fc_parts: List[str] = []
    if clean_hook:
        inputs += ["-i", raw_path]
        next_idx = 2
        fc_parts.append(f"[1:v]trim=0:{hook_dur:.3f},setpts=PTS-STARTPTS[hk]")
        fc_parts.append(f"[0:v]trim=start={hook_dur:.3f},setpts=PTS-STARTPTS[bd]")
        fc_parts.append("[hk][bd]concat=n=2:v=1:a=0[vcat]")
        cur = "[vcat]"
    else:
        cur = "[0:v]"
    if vchain:
        fc_parts.append(cur + ",".join(vchain) + "[vbase]")
        cur = "[vbase]"

    # ── Logo overlay (opt-in: default OFF unless explicitly enabled) ─────────
    if per.get("logo", cfg.get("logo", {}).get("enabled", False)):
        l_args, l_prep, l_xy = logo_overlay(cfg.get("logo", {}), w, h, insets,
                                            in_index=next_idx, tmp_dir=tmp_dir)
        if l_args:
            inputs += l_args
            next_idx += 1
            fc_parts.append(l_prep)
            fc_parts.append(f"{cur}[logo]overlay={l_xy}[vlogo]")
            cur = "[vlogo]"

    # ── Hook banner (golden rounded-border PNG; subtitle / face-aware) ────────
    if banner_on:
        png = os.path.join(tmp_dir, f"{name}_banner.png")
        sub_top = int(h * float(banner_cfg.get("sub_top_pct", 48.0)) / 100.0)
        sub_bot = int(h * float(banner_cfg.get("sub_bottom_pct", 65.0)) / 100.0)
        bottom_forbidden = float(banner_cfg.get("bottom_forbidden_pct", 20.0))
        bottom_limit = int(h * (1.0 - bottom_forbidden / 100.0))
        band_h = max(1, bottom_limit - sub_top)   # box must fit subtitle-top .. forbidden line
        # per-clip auto-contrast text colour sampled from the banner band
        b_color = pick_contrast_color(in_path, sub_top, bottom_limit, banner_cfg)
        rendered = render_banner_png(banner_text, png, w, h, banner_cfg,
                                     text_color=b_color, max_box_h=band_h)
        if rendered:
            _png, _bw, box_h = rendered
            # Blank/None from the UI must NOT fall through to the auto gap-finder
            # (which prefers the top and collides with the face/logo). Default to
            # the subtitle slot, and keep 'auto' in the subtitle slot during the
            # clean-hook window.
            placement = (banner_cfg.get("placement") or "subtitle").strip().lower()
            if placement not in ("subtitle", "top", "mid", "bottom", "auto"):
                placement = "subtitle"
            if clean_hook and placement == "auto":
                placement = "subtitle"   # caption-free window → use the subtitle slot
            need_face = placement in ("auto", "mid")
            face = detect_face_box(in_path) if need_face else None
            by = pick_banner_y(h, box_h, face, placement, sub_top, sub_bot, insets,
                               bottom_limit=bottom_limit)
            bfade = float(banner_cfg.get("fade", 0.3))
            inputs += ["-loop", "1", "-t", f"{hook_dur + 0.3:.2f}", "-i", png]
            bidx = next_idx
            next_idx += 1
            bchain = ["format=rgba"]
            if bfade > 0:
                bchain.append(f"fade=in:st=0:d={bfade}:alpha=1")
                bchain.append(f"fade=out:st={max(0.0, hook_dur - bfade):.2f}:d={bfade}:alpha=1")
            fc_parts.append(f"[{bidx}:v]" + ",".join(bchain) + "[ban]")
            fc_parts.append(f"{cur}[ban]overlay=x=(W-w)/2:y={by}"
                            f":enable='between(t,0,{hook_dur:.3f})'[vban]")
            cur = "[vban]"
            logger and logger.info(
                f"[{name}] banner @ y={by} placement={placement} clean_hook={clean_hook}")

    # ── Post: watermark + progress (inline), then the glowing end card ────────
    inline: List[str] = []
    inline += f_watermark(cfg.get("watermark", {}), w, h, insets)
    inline += f_progress(cfg.get("progress_bar", {}), total, w, h)
    ec_cfg = cfg.get("end_card", {})
    if ec_cfg.get("enabled"):
        if inline:
            fc_parts.append(cur + ",".join(inline) + "[vpost]")
            cur = "[vpost]"
        cur = _append_end_card_graph(fc_parts, cur, ec_cfg, total, w, h)
        fc_parts.append(cur + "null[vout]")
    elif inline:
        fc_parts.append(cur + ",".join(inline) + "[vout]")
    else:
        fc_parts.append(cur + "null[vout]")

    # ── AUDIO chain ──────────────────────────────────────────────────────────
    mcfg = cfg.get("music", {})
    music_enabled = per.get("music", mcfg.get("enabled", True)) and not (
        str(per.get("music_path", "")).lower() == "none")
    end_dur = float(cfg.get("end_card", {}).get("duration", 0.0)) if cfg.get("end_card", {}).get("enabled") else 0.0
    base_vol = float(per.get("music_volume", mcfg.get("volume", 0.10)))
    ramp_to = float(mcfg.get("end_ramp_volume", 0.25))

    bed_path = None
    if music_enabled:
        picker: MusicPicker = cfg["_picker"]
        track = picker.for_clip(name, per)
        if track:
            # Bed must cover the end card too, so music keeps playing over it.
            bed_path = build_music_bed(track, total + end_dur, mcfg, per,
                                       os.path.join(tmp_dir, f"{name}_bed.m4a"), logger)

    # Speech is only `total` long — pad a silent tail so the music alone carries
    # the outro (no more apad-silence killing the music during the end card).
    if end_dur > 0:
        fc_parts.append(f"[0:a]apad=pad_dur={end_dur:.3f}[sp]")
        speech = "[sp]"
    else:
        speech = "[0:a]"

    if bed_path:
        inputs += ["-i", bed_path]
        midx = next_idx
        next_idx += 1
        music_lbl = f"[{midx}:a]"
        if end_dur > 0:
            # The bed is already at base_vol, so ramp a RELATIVE gain ×1 -> ×(ramp_to/base)
            # across the end card, i.e. music swells base_vol -> ramp_to.
            mult_end = ramp_to / max(base_vol, 1e-3)
            volexpr = (f"if(lt(t,{total:.3f}),1,"
                       f"1+({mult_end:.4f}-1)*min(1,(t-{total:.3f})/{end_dur:.3f}))")
            fc_parts.append(f"{music_lbl}volume='{volexpr}':eval=frame[mvol]")
            music_lbl = "[mvol]"
        if mcfg.get("duck", False):
            fc_parts.append(f"{speech}asplit=2[spa][spb]")
            fc_parts.append(
                f"{music_lbl}[spb]sidechaincompress="
                f"threshold={mcfg.get('duck_threshold',0.05)}:ratio={mcfg.get('duck_ratio',8)}"
                f":attack=20:release=300[duck]")
            fc_parts.append("[spa][duck]amix=inputs=2:duration=first:dropout_transition=0[amix]")
        else:
            fc_parts.append(f"{speech}{music_lbl}amix=inputs=2:duration=first:dropout_transition=0[amix]")
        atail = "[amix]"
    else:
        atail = speech

    afilters = []
    if mcfg.get("loudnorm", True):
        afilters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    if afilters:
        fc_parts.append(atail + ",".join(afilters) + "[aout]")
    elif atail != "[aout]":
        fc_parts.append(atail + "anull[aout]")

    ffbin, venc = _encoder_args(cfg.get("encoder", "x264"))
    cmd = [ffbin, "-y"] + inputs + [
        "-filter_complex", ";".join(fc_parts),
        "-map", "[vout]", "-map", "[aout]",
    ] + venc + ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    if dry.get("enabled"):
        cmd += ["-t", f"{total:.3f}"]
    cmd.append(out_path)

    logger and logger.info(f"[{name}] rendering -> {os.path.basename(out_path)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           creationflags=_NO_WINDOW)
        if r.returncode == 0 and os.path.exists(out_path):
            return True
        logger and logger.error(f"[{name}] ffmpeg failed: {(r.stderr or '')[-600:]}")
    except Exception as e:
        logger and logger.error(f"[{name}] render exception: {e}")
    return False


def export_thumbnail(in_path: str, out_path: str, when: Any, total: float, logger) -> bool:
    if when == "middle":
        t = total / 2
    elif when == "hook":
        t = min(1.5, total * 0.1)
    else:
        try:
            t = float(when)
        except (TypeError, ValueError):
            t = 1.0
    try:
        r = subprocess.run([FFMPEG, "-y", "-ss", f"{t:.3f}", "-i", in_path,
                            "-frames:v", "1", "-q:v", "2", out_path],
                           capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW)
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception as e:
        logger and logger.warning(f"thumbnail failed: {e}")
        return False


def write_upload_kit(clip: Dict, out_txt: str) -> None:
    lines = [
        clip.get("youtube_title") or clip.get("title", ""),
        "",
        clip.get("description_text") or clip.get("description", ""),
        "",
        clip.get("description_hashtags", ""),
        "",
        "TAGS:",
        clip.get("youtube_tags", ""),
    ]
    Path(out_txt).write_text("\n".join(str(x) for x in lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Clip discovery + job runner
# ─────────────────────────────────────────────────────────────────────────────
def find_captioned(clips_dir: str, clip_name: str, variant: str) -> Optional[str]:
    """Pick the captioned file for a clip. variant: captioned | hooked."""
    d = Path(clips_dir)
    want_hooked = variant == "hooked"
    cands = sorted(d.glob(f"{clip_name}_captioned*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    hooked = [p for p in cands if p.stem.endswith("_hooked")]
    plain = [p for p in cands if not p.stem.endswith("_hooked")]
    chosen = (hooked or plain) if want_hooked else (plain or hooked)
    return str(chosen[0]) if chosen else None


def load_clips_plan(job_dir: str) -> List[Dict]:
    p = Path(job_dir) / "clips_plan.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


# stems that are pipeline artefacts / our own outputs — never brandable inputs
_SKIP_STEM_TOKENS = ("_branded", "_proxy", "_raw", "_filter", "_drytest")


def discover_targets(job_dir: str, variant: str = "captioned") -> List[Dict]:
    """Unified clip discovery used by BOTH the UI listing and run_job, so what
    you see is exactly what renders. Three tiers:

      1. clips_plan.json entries that have a captioned file (full metadata);
      2. captioned files on disk the plan doesn't know about (hand-stitched
         clips like a manual clip_06);
      3. external mode — when the folder has no captioned clips at all, every
         .mp4 at the job root or in clips/ becomes a target (any folder of
         shorts under outputs/ can be branded; no plan required).

    Every returned entry has clip_name + source_path.
    """
    job = Path(job_dir)
    clips_dir = job / "clips"
    out: List[Dict] = []
    used: set = set()

    for c in load_clips_plan(job_dir):
        name = c.get("clip_name")
        if not name or name in used:
            continue
        src = find_captioned(str(clips_dir), name, variant)
        if src:
            entry = dict(c)
            entry["source_path"] = src
            out.append(entry)
            used.add(name)

    if clips_dir.is_dir():
        for f in sorted(clips_dir.glob("*_captioned*.mp4")):
            name = f.stem.split("_captioned")[0]
            if name in used:
                continue
            src = find_captioned(str(clips_dir), name, variant)
            if src:
                out.append({"clip_name": name, "source_path": src})
                used.add(name)

    if out:
        return out

    # Recursive: any mp4 anywhere inside the folder is brandable ("every sub
    # folder inside outputs"). Nested files get a folder-qualified clip_name
    # so same-named files in different subfolders can't collide.
    for f in sorted(job.rglob("*.mp4")):
        rel_parts = f.relative_to(job).parts
        if "branded" in rel_parts[:-1]:      # never re-ingest our own outputs
            continue
        stem = f.stem
        if any(t in stem for t in _SKIP_STEM_TOKENS):
            continue
        name = "__".join(rel_parts[:-1] + (stem,)) if len(rel_parts) > 1 else stem
        if name in used:
            continue
        out.append({"clip_name": name, "source_path": str(f)})
        used.add(name)
    return out


def _concat_intro(intro: str, body: str, out_path: str, logger) -> bool:
    """Prepend the TTS intro to a captioned clip — same concat the main
    pipeline's captioner performs when it builds *_hooked variants."""
    cmd = [FFMPEG, "-y", "-i", intro, "-i", body,
           "-filter_complex", "[0:v][0:a][1:v][1:a] concat=n=2:v=1:a=1 [v][a]",
           "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k", out_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                           creationflags=_NO_WINDOW)
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception as e:
        logger and logger.error(f"hook concat failed: {e}")
        return False


def ensure_tts_hooked(job_dir: str, clip: Dict, captioned_path: str,
                      regenerate: bool, logger) -> Optional[str]:
    """Branding-side TTS hook generation: guarantee a *_hooked variant exists
    for this clip. Missing intros are generated with the main pipeline's
    tts_hook module using the same voice / bass / tail settings (config.py),
    so branding can do the TTS part standalone. Returns the hooked path, or
    None (caller falls back to the un-hooked input)."""
    name = clip.get("clip_name", "")
    cap = Path(captioned_path)
    if cap.stem.endswith("_hooked"):
        return str(cap)
    hooked_path = cap.with_name(cap.stem + "_hooked.mp4")
    if hooked_path.exists() and not regenerate:
        return str(hooked_path)

    try:
        import config as main_config
        from pipeline.tts_hook import (generate_hook_audio, build_hook_intro,
                                       _sanitize_hook_phrase)
    except Exception as e:
        logger and logger.warning(f"[{name}] TTS hook modules unavailable: {e}")
        return None

    clips_dir = Path(job_dir) / "clips"
    hooks_dir = clips_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    wav = hooks_dir / f"hook_{name}.wav"
    intro = hooks_dir / f"intro_{name}.mp4"
    raw = clips_dir / f"{name}_raw.mp4"
    frame_src = raw if raw.exists() else cap

    hook_phrase = _sanitize_hook_phrase(
        clip.get("hook_phrase") or clip.get("title") or "check this out")

    if regenerate or not (wav.exists() and intro.exists()):
        import wave as _wave
        if regenerate or not wav.exists():
            res = generate_hook_audio(
                text=hook_phrase, output_path=str(wav),
                voice_id=main_config.TTS_HOOK_VOICE_ID,
                sample_rate=main_config.TTS_HOOK_SAMPLE_RATE,
                speed=main_config.TTS_HOOK_SPEED,
                api_key=getattr(main_config, "SMALLEST_API_KEY", ""),
                logger=logger)
            if not res.get("success"):
                logger and logger.warning(f"[{name}] TTS synthesis failed; skipping hook")
                return None
        with _wave.open(str(wav), "rb") as wf:
            speech_s = wf.getnframes() / float(wf.getframerate())
        duration_s = speech_s + float(getattr(main_config, "TTS_HOOK_TAIL_S", 0.2))
        settings = {
            "tts_hook_font_size_pct": main_config.TTS_HOOK_FONT_SIZE_PCT,
            "tts_hook_text_placement": main_config.TTS_HOOK_TEXT_PLACEMENT,
            "tts_hook_text_color": main_config.TTS_HOOK_TEXT_COLOR,
            "tts_hook_text_color_revealed": main_config.TTS_HOOK_TEXT_COLOR_REVEALED,
            "tts_hook_sample_rate": main_config.TTS_HOOK_SAMPLE_RATE,
        }
        ok = build_hook_intro(
            raw_clip_path=str(frame_src), hook_phrase=hook_phrase,
            tts_wav_path=str(wav), output_path=str(intro),
            duration_s=duration_s, settings=settings, logger=logger,
            speech_duration_s=speech_s)
        if not ok or not intro.exists():
            logger and logger.warning(f"[{name}] hook intro render failed")
            return None

    if hooked_path.exists():
        try:
            hooked_path.unlink()
        except OSError:
            pass
    if _concat_intro(str(intro), str(cap), str(hooked_path), logger):
        logger and logger.info(f"[{name}] TTS hook prepended -> {hooked_path.name}")
        return str(hooked_path)
    return None


def _normalize_only(only) -> set:
    """Accept [1,2,3] / ['1','2'] / ['clip_01'] / '1,2,3' -> {'clip_01',...}."""
    out: set = set()
    if not only:
        return out
    if isinstance(only, str):
        only = [p for p in re.split(r"[,\s]+", only) if p]
    for e in only:
        s = str(e).strip()
        if not s:
            continue
        if s.lower().startswith("clip_"):
            out.add(s)
        elif s.isdigit():
            out.add(f"clip_{int(s):02d}")
        else:
            out.add(s)
    return out


def _clean_tag(tag) -> str:
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag or "branded").strip()).strip("_")
    return t or "branded"


def run_job(job_dir: str, cfg: Dict, logger,
            progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict:
    job_dir = str(job_dir)
    clips_dir = os.path.join(job_dir, "clips")
    out_dir = os.path.join(clips_dir, "branded")
    tmp_dir = os.path.join(out_dir, ".tmp")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    cfg = dict(cfg)
    cfg["_picker"] = MusicPicker(cfg.get("music", {}), job_dir, logger)
    variant = cfg.get("input_variant", "captioned")
    tag = _clean_tag(cfg.get("output_tag", "branded"))

    # Optional subset: render only the requested clips. Accepts ints (1),
    # numeric strings ("1") or names ("clip_01"); empty = render everything.
    only = _normalize_only(cfg.get("only_clips", []))

    done, failed, results = 0, 0, []
    discovered = discover_targets(job_dir, variant)
    targets = [c for c in discovered if not only or c["clip_name"] in only]
    n = len(targets)
    if not discovered:
        logger and logger.warning(f"No brandable clips found in {job_dir}")
    if only:
        logger and logger.info(f"Rendering subset: {n} clip(s) -> {sorted(c['clip_name'] for c in targets)}")
    for i, clip in enumerate(targets, 1):
        name = clip["clip_name"]
        src = clip.get("source_path") or find_captioned(clips_dir, name, variant)
        if not src:
            logger and logger.warning(f"[{name}] no source file found; skipping")
            failed += 1
            progress_cb and progress_cb(i, n, f"{name}: no source file")
            continue
        if cfg.get("tts_hook", {}).get("enabled"):
            progress_cb and progress_cb(i, n, f"{name}: TTS hook")
            hooked = ensure_tts_hooked(job_dir, clip, src,
                                       bool(cfg.get("tts_hook", {}).get("regenerate")), logger)
            if hooked:
                src = hooked
        out_path = os.path.join(out_dir, f"{name}_{tag}.mp4")
        raw = os.path.join(clips_dir, f"{name}_raw.mp4")
        ok = brand_one_clip(src, out_path, clip, cfg, logger, tmp_dir,
                            raw_path=(raw if os.path.exists(raw) else ""))
        if ok:
            done += 1
            results.append(out_path)
            total = probe_duration(out_path)
            if cfg.get("thumbnail", {}).get("enabled", False):
                export_thumbnail(out_path, os.path.join(out_dir, f"{name}_{tag}_cover.jpg"),
                                 cfg.get("thumbnail", {}).get("at", "hook"), total, logger)
            if cfg.get("upload_kit", {}).get("enabled", False):
                write_upload_kit(clip, os.path.join(out_dir, f"{name}_{tag}.txt"))
        else:
            failed += 1
        progress_cb and progress_cb(i, n, f"{name}: {'ok' if ok else 'failed'}")

    # best-effort tmp cleanup
    try:
        for f in Path(tmp_dir).glob("*"):
            f.unlink()
        Path(tmp_dir).rmdir()
    except Exception:
        pass

    summary = {"total": n, "done": done, "failed": failed, "out_dir": out_dir, "outputs": results}
    logger and logger.info(f"Branding complete: {done}/{n} ok, {failed} failed -> {out_dir}")
    return summary


# ── tiny stdout logger for CLI use ───────────────────────────────────────────
class _PrintLogger:
    def info(self, m): print(f"[INFO] {m}", flush=True)
    def warning(self, m): print(f"[WARN] {m}", flush=True)
    def error(self, m): print(f"[ERR ] {m}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Branding finisher (CLI)")
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--config", default="")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else default_config()
    run_job(args.job_dir, cfg, _PrintLogger(),
            progress_cb=lambda i, n, m: print(f"  [{i}/{n}] {m}", flush=True))
