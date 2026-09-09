import os
import subprocess
import logging
from typing import List, Tuple
import config

logger = logging.getLogger("trimmer")


def _probe_fps(video_path: str) -> float:
    """Return the source frame rate, or 30.0 if it can't be determined."""
    try:
        ffmpeg = str(getattr(config, "FFMPEG_PATH", "ffmpeg"))
        base = os.path.dirname(ffmpeg)
        probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        ffprobe = os.path.join(base, probe_name) if base else probe_name
        res = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        val = (res.stdout or "").strip()
        if "/" in val:
            num, den = val.split("/", 1)
            fps = float(num) / float(den) if float(den) else 0.0
        else:
            fps = float(val) if val else 0.0
        return fps if fps > 0 else 30.0
    except Exception:
        return 30.0


def extract_preview_snippet(video_path: str, out_path: str, start: float, duration: float) -> bool:
    """Cut a short, all-keyframe, browser-seekable snippet for the frame-accurate
    ending adjuster. Small + fast (480p, ultrafast); `-g 1` makes every frame an
    I-frame so <video>.currentTime seeks land on the exact frame."""
    ffmpeg = str(getattr(config, "FFMPEG_PATH", "ffmpeg"))
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{max(0.0, start):.3f}",
        "-i", video_path,
        "-t", f"{max(0.1, duration):.3f}",
        "-vf", "scale=-2:480",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-g", "1", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        ok = res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
        if not ok:
            logger.warning(f"Preview snippet failed (rc={res.returncode}): {(res.stderr or b'')[-400:]}")
        return ok
    except Exception:
        logger.exception("Preview snippet extraction failed")
        return False


def trim_video(video_path: str, keep_segments: List[Tuple[float, float]], transition: str = "sharp") -> str:
    """
    Takes an MP4 video and an array of [start, end] keep segments in seconds.
    Uses FFmpeg to cut and concatenate the segments, re-encoding with hardware acceleration.
    transition can be "sharp" or "crossfade".
    Returns the absolute path to the trimmed video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not keep_segments:
        raise ValueError("No keep_segments provided for trimming.")

    # Sort segments by start time
    keep_segments = sorted(keep_segments, key=lambda x: x[0])

    base_dir = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    name, ext = os.path.splitext(filename)
    
    # Generate unique output filename
    out_filename = f"{name}_trimmed{ext}"
    out_path = os.path.join(base_dir, out_filename)
    
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass

    filter_lines = []
    video_labels = []
    audio_labels = []
    
    n_segments = len(keep_segments)
    
    cmd_inputs = []
    for _ in range(n_segments):
        cmd_inputs.extend(["-i", video_path])

    for i, (start, end) in enumerate(keep_segments):
        v_in = f"[{i}:v]"
        a_in = f"[{i}:a]"
        v_lbl = f"[v{i}]"
        a_lbl = f"[a{i}]"
        
        # trim the video stream and reset presentation timestamps
        v_filter = f"{v_in}trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS{v_lbl}"
        # trim the audio stream and reset presentation timestamps
        a_filter = f"{a_in}atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS{a_lbl}"
        
        filter_lines.append(v_filter)
        filter_lines.append(a_filter)
        video_labels.append(v_lbl)
        audio_labels.append(a_lbl)

    # Concat or XFade filter
    if n_segments == 1:
        # Just map the single segment
        pass # The map args below need to be updated to map [v0] and [a0]
    elif transition == "crossfade":
        _xfade_frames = max(1, int(getattr(config, "XFADE_FRAMES", 5)))
        crossfade_duration = _xfade_frames / max(1.0, _probe_fps(video_path))
        current_v = "[v0]"
        current_a = "[a0]"
        current_dur = keep_segments[0][1] - keep_segments[0][0]
        
        for i in range(1, n_segments):
            seg_dur = keep_segments[i][1] - keep_segments[i][0]
            next_v = f"[v{i}]"
            next_a = f"[a{i}]"
            out_v = f"[v_xf{i}]" if i < n_segments - 1 else "[v]"
            out_a = f"[a_xf{i}]" if i < n_segments - 1 else "[a]"
            
            # Ensure offset is valid (cannot be negative)
            offset = max(0.01, current_dur - crossfade_duration)
            
            filter_lines.append(f"{current_v}{next_v}xfade=transition=fade:duration={crossfade_duration}:offset={offset:.3f}{out_v}")
            filter_lines.append(f"{current_a}{next_a}acrossfade=d={crossfade_duration}{out_a}")
            
            current_v = out_v
            current_a = out_a
            current_dur = current_dur + seg_dur - crossfade_duration
    else:
        # Default Sharp Cut (Concat)
        concat_input = "".join([f"{v}{a}" for v, a in zip(video_labels, audio_labels)])
        concat_filter = f"{concat_input}concat=n={n_segments}:v=1:a=1[v][a]"
        filter_lines.append(concat_filter)
    
    filter_complex = ";".join(filter_lines)

    # Output mappings
    map_v = "[v0]" if n_segments == 1 else "[v]"
    map_a = "[a0]" if n_segments == 1 else "[a]"

    # Build FFmpeg command
    cmd = [
        config.FFMPEG_PATH,
        "-y"
    ]
    cmd.extend(cmd_inputs)
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", map_v,
        "-map", map_a,
    ])

    # Check hardware encoding availability
    use_gpu = getattr(config, "VIDEO_ENCODER", "libx264") == "h264_nvenc"
    if use_gpu:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", str(getattr(config, "NVENC_PRESET", "p4")),
            "-cq", str(getattr(config, "NVENC_CQ", 23)),
            "-rc", "vbr",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(getattr(config, "X264_CRF", 20)),
        ])

    cmd.extend([
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ])

    logger.info(f"Running FFmpeg Trim Command: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        logger.info(f"Successfully trimmed video: {out_path}")
        return out_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg trim failed. Exit code: {e.returncode}\n{e.stderr}")
        raise RuntimeError(f"FFmpeg failed to trim video: {e.stderr}")
