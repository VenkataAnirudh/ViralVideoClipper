#!/usr/bin/env python3
"""
Template Authoring Tool  -  proxy-edition (AV1/H264/HEVC safe)
===============================================================
* Generates a 960p MJPEG proxy via CPU (works for ANY codec: AV1, H264, HEVC)
* MJPEG = every frame is I-frame => instant lag-free seek/step
* Background prefetch thread for smooth playback
* All snap prompts via cv2.waitKey - zero blocking input() calls
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DISPLAY_MAX_W   = 1280
DISPLAY_MAX_H   = 720
PROXY_W         = 960        # proxy width (height auto from aspect ratio)
PROXY_QUALITY   = 5          # MJPEG q: 2=best 31=worst; 4-6 is good
PREFETCH_FRAMES = 10
PLAY_WAIT_MS    = 16         # ~60 fps during playback
SCRUB_STEPS     = 1000
# Default: proxy only the FIRST 10 MINUTES. Podcast camera angles are
# established early, so 10 min is enough to snap every template while keeping the
# proxy small and fast to build. Set TEMPLATE_PROXY_MAX_SEC=0 for the FULL video,
# or any positive number of seconds to change the cap.
PROXY_MAX_SEC   = int(os.getenv("TEMPLATE_PROXY_MAX_SEC", "600"))

FFMPEG_CANDIDATES = [
    r"D:\Coding\Video Clipper\ffmpeg-gpu\ffmpeg.exe",
    r"D:\Coding\Video Clipper\ffmpeg-bin\ffmpeg.exe",
    "ffmpeg",
]


def _find_ffmpeg() -> str:
    for p in FFMPEG_CANDIDATES:
        if os.path.isfile(p):
            return p
    return "ffmpeg"


# ---------------------------------------------------------------------------
# Proxy generation  (CPU-only decode — works for AV1, H264, HEVC, VP9, etc.)
# ---------------------------------------------------------------------------

def _proxy_path(src: Path) -> Path:
    return src.with_name(src.stem + "_proxy.avi")


def _duration_sec(src: Path, ffmpeg: str) -> float:
    """Quick ffprobe to get video duration."""
    # Swap only the basename: a full-string replace also corrupts the directory
    # (ffmpeg-gpu\ffmpeg.exe -> ffprobe-gpu\ffprobe.exe, which doesn't exist),
    # making every probe return 0 and forcing a proxy rebuild on every launch.
    name = Path(ffmpeg).name
    if "ffmpeg" in name:
        cand = Path(ffmpeg).with_name(name.replace("ffmpeg", "ffprobe"))
        ff = str(cand) if cand.is_file() else "ffprobe"
    else:
        ff = "ffprobe"
    try:
        r = subprocess.run(
            [ff, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def build_proxy(src: Path, ffmpeg: str) -> Optional[Path]:
    """
    Transcode src to a low-res MJPEG AVI proxy.
    - CPU decode: reliable for AV1, H264, HEVC, VP9, etc.
    - -threads 0: uses all CPU cores
    - MJPEG: every frame is a keyframe -> O(1) seeking in OpenCV
    Returns proxy path, or None on failure.
    """
    dst = _proxy_path(src)
    duration = _duration_sec(src, ffmpeg)
    # PROXY_MAX_SEC <= 0 means "full video". Otherwise cap.
    if PROXY_MAX_SEC and PROXY_MAX_SEC > 0:
        target_dur = min(duration, PROXY_MAX_SEC) if duration > 0 else PROXY_MAX_SEC
    else:
        target_dur = duration  # full length (0 if probe failed → no -t cap)

    # Reuse ONLY a complete, seekable proxy. A truncated / un-finalized MJPEG AVI
    # (a previous build was interrupted or capped) has no AVI index, so OpenCV
    # reports frame_count=0 AND POS_FRAMES seeking silently fails — that is the
    # frozen "f=1/1, can't step through frames" screen the user hit. Validate by
    # DURATION (not file size), and rebuild if it's short/broken.
    if dst.exists() and dst.stat().st_size > 4096:
        proxy_dur = _duration_sec(dst, ffmpeg)
        if proxy_dur > 0 and (target_dur <= 0 or proxy_dur >= target_dur * 0.9):
            print(f"[proxy] Reusing existing proxy: {dst.name} ({proxy_dur:.0f}s)")
            return dst
        print(f"[proxy] Existing proxy incomplete/unseekable "
              f"(proxy={proxy_dur:.0f}s vs expected ~{target_dur:.0f}s) — rebuilding.")
        try:
            dst.unlink()
        except OSError:
            pass
    dur_str  = f"{int(duration//60)}m{int(duration%60)}s" if duration else "unknown length"
    print(f"[proxy] Building 960p MJPEG proxy for {src.name} ({dur_str})")
    print("[proxy] Using all CPU cores — please wait (one-time operation)...")

    cmd = [ffmpeg, "-y", "-i", str(src)]
    if target_dur and target_dur > 0:
        cmd += ["-t", str(int(target_dur))]   # cap only when a limit is set
    cmd += [
        "-vf", f"scale={PROXY_W}:-2",   # scale to 960 wide, keep aspect
        "-c:v", "mjpeg",
        "-q:v", str(PROXY_QUALITY),
        "-an",                           # no audio needed
        "-threads", "0",                 # all CPU cores
        str(dst),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        BAR_W  = 28
        t0     = time.time()
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").strip()
            if "time=" in line:
                m = re.search(r"time=(\d+:\d+:\d+\.?\d*)", line)
                if m:
                    parts   = m.group(1).split(":")
                    done    = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                    done    = min(done, target_dur)
                    frac    = done / max(target_dur, 1)
                    filled  = int(BAR_W * frac)
                    bar     = chr(0x2588)*filled + chr(0x2591)*(BAR_W - filled)
                    elapsed = int(time.time() - t0)
                    speed   = done / max(elapsed, 0.001)
                    eta     = int((target_dur - done) / max(speed, 0.001))
                    mm_d, ss_d = divmod(int(done), 60)
                    mm_t, ss_t = divmod(int(target_dur), 60)
                    print(f"  |{bar}| {frac*100:5.1f}%  "
                          f"{mm_d:02d}:{ss_d:02d}/{mm_t:02d}:{ss_t:02d}  "
                          f"elapsed={elapsed}s  eta~{eta}s    ",
                          end="\r")
            elif line.startswith("[") and "error" in line.lower():
                print(f"\n  [warn] {line}")

        proc.wait()
        print()   # newline after \r progress

        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 4096:
            sz = dst.stat().st_size / 1e6
            print(f"[proxy] Done -> {dst.name}  ({sz:.0f} MB)")
            return dst

        print(f"[proxy] FFmpeg exited {proc.returncode} — proxy failed")

    except Exception as e:
        print(f"[proxy] Exception: {e}")

    # Cleanup broken partial file
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass

    print("[proxy] Falling back to original video (navigation may be slow)")
    return None


# ---------------------------------------------------------------------------
# Frame prefetch thread
# ---------------------------------------------------------------------------

class PrefetchReader:
    _STOP = object()

    def __init__(self, cap: cv2.VideoCapture, buf: int = PREFETCH_FRAMES):
        self._cap  = cap
        self._q: queue.Queue = queue.Queue(maxsize=buf)
        self._ev   = threading.Event()
        self._lock = threading.Lock()
        self._th   = threading.Thread(target=self._work, daemon=True)
        self._th.start()

    def _work(self):
        while not self._ev.is_set():
            with self._lock:
                ok, frame = self._cap.read()
            if not ok:
                try:
                    self._q.put(self._STOP, timeout=0.1)
                except queue.Full:
                    pass
                break
            try:
                self._q.put((ok, frame), timeout=0.05)
            except queue.Full:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Any]:
        try:
            item = self._q.get(timeout=0.04)
            if item is self._STOP:
                return False, None
            return item
        except queue.Empty:
            return False, None

    def seek(self, idx: int):
        self._ev.set()
        self._th.join(timeout=1.0)
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self._ev.clear()
        self._th = threading.Thread(target=self._work, daemon=True)
        self._th.start()

    def release(self):
        self._ev.set()
        self._th.join(timeout=1.0)
        self._cap.release()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_bbox(box: Tuple[int, int, int, int], w: int, h: int) -> List[float]:
    x, y, bw, bh = box
    return [
        clamp(x / max(w, 1), 0.0, 1.0),
        clamp(y / max(h, 1), 0.0, 1.0),
        clamp((x + bw) / max(w, 1), 0.0, 1.0),
        clamp((y + bh) / max(h, 1), 0.0, 1.0),
    ]


def scale_frame(frame, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = frame.shape[:2]
    s = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if s < 1.0:
        return cv2.resize(frame, (int(w * s), int(h * s)), cv2.INTER_AREA), s
    return frame.copy(), 1.0


def draw_banner(img, text: str, color=(100, 255, 100)):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, h - 44), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def make_disp_name(tid: int, slots: List[Dict]) -> str:
    lbls = [str(s.get("speaker_label") or "").replace("SPEAKER_", "spk")
            for s in slots if s.get("speaker_label")]
    if len(lbls) >= 2:
        return f"wide_{'_'.join(lbls)}_{tid:02d}"
    if len(lbls) == 1:
        return f"{lbls[0]}_{tid:02d}"
    return f"manual_{tid:02d}"


# ---------------------------------------------------------------------------
# Snap interaction  (all cv2.waitKey - no input() blocking)
# ---------------------------------------------------------------------------

SNAP_WIN = "Snap  (Enter=accept | Esc=done)"


def collect_snap(disp_frame) -> List[Tuple[Tuple, Optional[str]]]:
    accepted: List[Tuple[Tuple, Optional[str]]] = []
    work = disp_frame.copy()
    cv2.namedWindow(SNAP_WIN, cv2.WINDOW_AUTOSIZE)

    while True:
        guide = work.copy()
        draw_banner(guide, "Draw face box  |  Enter=accept   Esc=done", (255, 255, 80))
        roi = cv2.selectROI(SNAP_WIN, guide, fromCenter=False, showCrosshair=True)
        x, y, bw, bh = map(int, roi)
        if bw <= 0 or bh <= 0:
            break

        # Ask for speaker label
        li = work.copy()
        cv2.rectangle(li, (x, y), (x + bw, y + bh), (0, 200, 255), 3)
        draw_banner(li, "Label:  0=SPEAKER_00   1=SPEAKER_01   Enter=skip", (80, 220, 255))
        cv2.imshow(SNAP_WIN, li)
        cv2.waitKey(1)

        label: Optional[str] = None
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord('0'):       label = "SPEAKER_00"; break
            elif k == ord('1'):     label = "SPEAKER_01"; break
            elif k in (13, 10, 27): break

        accepted.append(((x, y, bw, bh), label))
        cv2.rectangle(work, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        if label:
            cv2.putText(work, label, (x + 4, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Another box?
        mi = work.copy()
        draw_banner(mi, "Another box?  Y=yes   N / Enter=done", (100, 255, 100))
        cv2.imshow(SNAP_WIN, mi)
        cv2.waitKey(1)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('y'), ord('Y')):
                break
            elif k in (ord('n'), ord('N'), 13, 10, 27):
                cv2.destroyWindow(SNAP_WIN)
                return accepted

    cv2.destroyWindow(SNAP_WIN)
    return accepted


def make_entry(tid: int, ts: float, disp_frame,
               bl: List[Tuple[Tuple, Optional[str]]]) -> Dict[str, Any]:
    dh, dw = disp_frame.shape[:2]
    slots = []
    for idx, (box, lbl) in enumerate(bl):
        slot = ("single" if len(bl) == 1
                else "left" if idx == 0 else "right" if idx == 1 else f"slot_{idx}")
        slots.append({"slot_name": slot, "speaker_label": lbl,
                      "bbox_norm": norm_bbox(box, dw, dh), "confidence": 1.0})
    return {
        "template_id":  tid,
        "timestamp":    round(float(ts), 4),
        "type":         "WIDE" if len(slots) >= 2 else "SINGLE",
        "display_name": make_disp_name(tid, slots),
        "slots":        slots,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src      = Path(args.video)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_ffmpeg()
    print(f"[ffmpeg] {ffmpeg}")

    # Build proxy (runs synchronously with terminal progress)
    proxy    = build_proxy(src, ffmpeg)
    nav_path = proxy if proxy else src
    print(f"[nav] Opening: {nav_path.name}")

    raw_cap = cv2.VideoCapture(str(nav_path))
    if not raw_cap.isOpened():
        print(f"[ERROR] Cannot open: {nav_path}")
        return 1

    fps   = raw_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 1:
        # MJPEG/AVI sometimes won't report a frame count even when seekable.
        # Derive it from the probed duration so the scrubber + Left/Right/A/D
        # stepping have a real range instead of collapsing to one frame ("f=1/1").
        _dur = _duration_sec(nav_path, ffmpeg) or _duration_sec(src, ffmpeg)
        if _dur > 0 and fps > 0:
            total = int(_dur * fps)
    total = max(1, total)
    nw    = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nh    = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ds    = min(DISPLAY_MAX_W / max(nw, 1), DISPLAY_MAX_H / max(nh, 1), 1.0)
    dw    = int(nw * ds)
    dh    = int(nh * ds)

    print(f"[nav] {nw}x{nh} @ {fps:.1f} fps  display {dw}x{dh}  frames={total}")
    print("Space=play/pause | Left/Right=step | A/D=1s | S=snap | Q=save+quit")

    reader    = PrefetchReader(raw_cap, PREFETCH_FRAMES)
    frame_idx = 0
    playing   = False
    templates: List[Dict[str, Any]] = []
    cur_disp  = None
    cur_vis   = None

    def _save_templates(tpls: List[Dict[str, Any]]) -> None:
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(tpls, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [warn] could not save templates: {e}")

    WIN = "Template Tool  (S=snap | Q=save+quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh + 30)

    tb_val  = [0]
    tb_seek = [False]
    tb_prog = [False]   # True while the CODE moves the trackbar (not the user)

    def _on_scrub(v):
        # cv2.setTrackbarPos fires this callback too. The slider has only
        # SCRUB_STEPS(=1000) positions for ~15-18k proxy frames, so a
        # programmatic sync echoes back a value quantized up to a step-width
        # (~18 frames) EARLIER — which used to be treated as a user drag and
        # re-seeked backward after every click. Ignore our own echoes.
        if tb_prog[0]:
            return
        tb_val[0] = v
        tb_seek[0] = True

    cv2.createTrackbar("Position", WIN, 0, SCRUB_STEPS, _on_scrub)

    def _sync_trackbar():
        tb_prog[0] = True
        try:
            cv2.setTrackbarPos("Position", WIN,
                               int(frame_idx / max(total - 1, 1) * SCRUB_STEPS))
        except cv2.error:
            pass   # window closed via [X]; the main loop exits gracefully below
        finally:
            tb_prog[0] = False

    def seek(idx: int):
        nonlocal frame_idx, cur_disp, cur_vis, playing
        playing   = False
        frame_idx = max(0, min(idx, total - 1))
        reader.seek(frame_idx)
        cur_disp = cur_vis = None
        _sync_trackbar()

    seek(0)

    while True:
        # Scrubber drag
        if tb_seek[0]:
            tb_seek[0] = False
            target = int(tb_val[0] / SCRUB_STEPS * (total - 1))
            if abs(target - frame_idx) > 1:
                seek(target)

        # Fetch frame
        if playing or cur_disp is None:
            ok, raw = reader.read()
            if ok and raw is not None:
                cur_disp, _ = scale_frame(raw)
                frame_idx  += 1
                if playing:
                    _sync_trackbar()
            elif playing:
                playing = False

        # Draw HUD
        if cur_disp is not None:
            ts = frame_idx / max(fps, 1.0)
            mm, ss = divmod(int(ts), 60)
            vis = cur_disp.copy()
            h0  = vis.shape[0]
            cv2.putText(vis,
                        f"{'PLAY' if playing else 'PAUSE'}  {mm:02d}:{ss:02d}  "
                        f"f={frame_idx}/{total}  templates={len(templates)}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis,
                        "Space=play/pause | S=snap | Q=quit | A/D=1s | Arrows=step | scrubber",
                        (8, h0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (170, 170, 170), 1, cv2.LINE_AA)
            cur_vis = vis
            cv2.imshow(WIN, vis)

        key = cv2.waitKeyEx(PLAY_WAIT_MS if playing else 30)

        # Closed with the [X] button? Exit gracefully and KEEP what was snapped
        # (previously this crashed in seek() -> setTrackbarPos on a NULL window
        # and every snapped template was lost).
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                print("[nav] Window closed — finishing with "
                      f"{len(templates)} snapped template(s).")
                break
        except cv2.error:
            print("[nav] Window gone — finishing with "
                  f"{len(templates)} snapped template(s).")
            break

        if key in (ord('q'), ord('Q')):
            break
        elif key == 32:                          # Space - play/pause
            playing = not playing
        elif key in (ord('s'), ord('S')):        # Snap
            playing = False
            snap_ts = frame_idx / max(fps, 1.0)
            smm, sss = divmod(int(snap_ts), 60)
            bl = collect_snap(cur_disp)
            if bl:
                entry = make_entry(len(templates), snap_ts, cur_disp, bl)
                templates.append(entry)
                _save_templates(templates)   # persist immediately — crash-proof
                lbls = [l for _, l in bl if l]
                print(f"  [+] Template #{len(templates)-1}  "
                      f"t={smm:02d}:{sss:02d}  "
                      f"boxes={len(bl)}  labels={lbls or 'none'}")
            else:
                print("  [~] Snap cancelled.")
            if cur_vis is not None:
                cv2.imshow(WIN, cur_vis)
        elif key in (81, 2424832):               # Left arrow
            seek(frame_idx - 1)
        elif key in (83, 2555904):               # Right arrow
            seek(frame_idx + 1)
        elif key in (ord('a'), ord('A')):
            seek(frame_idx - int(round(fps)))
        elif key in (ord('d'), ord('D')):
            seek(frame_idx + int(round(fps)))

    reader.release()
    cv2.destroyAllWindows()

    if templates:
        _save_templates(templates)
        print(f"\n[OK] Saved {len(templates)} template(s) -> {out_path}")
    else:
        # Don't write an empty file — the launcher treats a missing output as
        # "closed without snapping" and falls back to auto discovery.
        print("\n[~] No templates snapped — nothing written.")
    return 0


if __name__ == "__main__":
    # Force UTF-8 console so non-ASCII filenames / progress glyphs (em-dash, curly
    # quotes) can't crash the tool with a charmap UnicodeEncodeError when it is
    # spawned by the pipeline with a non-UTF-8 stdout pipe (Windows default cp1252).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 — capture ANY crash for diagnosis
        import traceback
        import datetime
        # Write the crash to a log next to --output so the launcher and user can
        # see WHY the CV screen failed, instead of a silent fallback to auto.
        try:
            _ap = argparse.ArgumentParser()
            _ap.add_argument("--output")
            _known, _ = _ap.parse_known_args()
            _errp = (
                Path(_known.output).with_name("template_tool_error.log")
                if _known.output else Path("template_tool_error.log")
            )
            _errp.parent.mkdir(parents=True, exist_ok=True)
            with _errp.open("w", encoding="utf-8") as _f:
                _f.write(f"[{datetime.datetime.now()}] template_tool.py crashed:\n")
                _f.write("".join(
                    traceback.format_exception(type(_exc), _exc, _exc.__traceback__)
                ))
        except Exception:
            pass
        traceback.print_exc()
        raise SystemExit(1)
