# fileName: pipeline/transcode_helper.py
"""
Utility module for checking downloaded video codecs and automatically transcoding
unsupported formats (like AV1) to H.264 to enable GPU decoding (NVDEC) in subsequent stages.
"""

import os
import subprocess
import time
import shutil
import logging
from pathlib import Path
import config

def get_ffmpeg_paths(logger=None) -> tuple[str, str]:
    """Find local or system ffmpeg and ffprobe path."""
    project_root = Path(__file__).parent.parent.resolve()
    local_ffmpeg_gpu = project_root / "ffmpeg-gpu" / "ffmpeg.exe"
    local_ffmpeg = project_root / "ffmpeg-bin" / "ffmpeg.exe"
    local_ffprobe = project_root / "ffmpeg-bin" / "ffprobe.exe"

    if local_ffmpeg_gpu.exists():
        return str(local_ffmpeg_gpu), str(local_ffprobe)
    if local_ffmpeg.exists():
        return str(local_ffmpeg), str(local_ffprobe)
        
    if os.path.exists(config.FFMPEG_PATH):
        ffprobe = config.FFPROBE_PATH
        if not os.path.exists(ffprobe):
            ffprobe = str(Path(config.FFMPEG_PATH).parent / "ffprobe.exe")
        return config.FFMPEG_PATH, ffprobe
        
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path:
        return ffmpeg_path, (ffprobe_path or str(Path(ffmpeg_path).parent / "ffprobe.exe"))
        
    raise RuntimeError("FFmpeg/FFprobe binaries not found")


def ensure_h264_source(video_path: str, logger: logging.Logger, progress_callback=None) -> str:
    """
    Check if the video is AV1 (or another codec not supported by hardware decode).
    If it is, transcode it to H.264 using GPU (NVENC) encoding to unlock fast NVDEC decoding.
    Returns the path to the H.264 video.
    """
    if not getattr(config, "AUTO_TRANSCODE_TO_H264", True):
        logger.info("[TRANSCODE] Auto-transcode to H.264 is disabled in config.py")
        return video_path

    if not os.path.exists(video_path):
        logger.warning(f"[TRANSCODE] Video file not found for transcoding check: {video_path}")
        return video_path

    try:
        ffmpeg_bin, ffprobe_bin = get_ffmpeg_paths(logger)
    except Exception as e:
        logger.warning(f"[TRANSCODE] Skipping transcode check - FFmpeg binaries not resolved: {e}")
        return video_path

    # Step 1: Detect video codec of the source video
    probe_cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        res = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        codec = res.stdout.strip().lower()
        logger.info(f"[TRANSCODE] Video file '{os.path.basename(video_path)}' is in '{codec}' format")
    except Exception as e:
        logger.warning(f"[TRANSCODE] Could not probe video codec: {e}")
        return video_path

    # Step 2: Decide whether to transcode
    # GTX 1650 lacks physical AV1 hardware decoder blocks, so AV1 is the primary bottleneck.
    # Other legacy/uncommon codecs should also be transcoded to ensure hardware acceleration works.
    supported_codecs = {"h264"}
    if codec in supported_codecs:
        logger.info(f"[TRANSCODE] '{codec}' matches standard H.264. Skipping transcode.")
        return video_path

    # Step 3: Trigger transcoding of AV1 -> H.264 using GPU (NVENC)
    logger.info(f"[TRANSCODE] 🚨 ALERT: Video codec is '{codec}' (unsupported by GPU hardware decoding).")
    logger.info("[TRANSCODE] Starting automatic transcode to H.264 (CPU decode, GPU encode) to unlock NVDEC...")
    
    # Get original duration to calculate percentage
    dur_cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        dur_res = subprocess.run(
            dur_cmd, capture_output=True, text=True, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        duration = float(dur_res.stdout.strip() or 0.0)
    except Exception:
        duration = 0.0

    t_start = time.time()
    temp_transcoded = video_path + ".transcoded.mp4"
    
    # We use h264_nvenc for super fast GPU encoding.
    # We copy the audio stream (-c:a copy) to preserve it exactly and avoid re-encoding audio.
    transcode_cmd = [
        ffmpeg_bin, "-y", "-nostdin",
        "-threads", "12",
        "-i", video_path,
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-cq", str(getattr(config, "NVENC_CQ", 23)),
        "-b:v", "0",
        "-rc", "vbr",
        "-c:a", "copy",
        temp_transcoded
    ]
    
    try:
        logger.info(f"[TRANSCODE] Running transcode: {' '.join(transcode_cmd)}")
        
        import re
        import sys
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            transcode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # FFmpeg outputs log info to stderr, which is combined here
            text=True,
            bufsize=1,
            creationflags=creationflags
        )
        
        ffmpeg_time_re = re.compile(r'time=(\d+):(\d+):(\d+\.\d+)')
        last_reported_pct = -1
        full_output = []
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if not line:
                continue
            full_output.append(line)
            
            # Parse ffmpeg time
            m = ffmpeg_time_re.search(line)
            if m and duration > 0:
                h, m_min, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
                total_secs = h * 3600 + m_min * 60 + s
                pct = int((total_secs / duration) * 100)
                pct = max(0, min(100, pct))
                
                if pct != last_reported_pct:
                    last_reported_pct = pct
                    sys.stdout.write(f"\r[TRANSCODE] Progress: {pct}%")
                    sys.stdout.flush()
                    if pct % 10 == 0:
                        logger.info(f"[TRANSCODE] Progress: {pct}%")
                    if progress_callback:
                        try:
                            progress_callback(pct, "transcode")
                        except Exception:
                            pass
                            
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, transcode_cmd, output="".join(full_output))
        
        # Verify transcoded output exists and is valid
        if os.path.exists(temp_transcoded) and os.path.getsize(temp_transcoded) > 4096:
            # Check duration of transcoded file to ensure completeness
            dur_cmd = [
                ffprobe_bin, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                temp_transcoded
            ]
            dur_res = subprocess.run(
                dur_cmd, capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            transcoded_dur = float(dur_res.stdout.strip() or 0.0)
            
            elapsed = time.time() - t_start
            size_mb = os.path.getsize(temp_transcoded) / (1024 * 1024)
            logger.info(f"[TRANSCODE] ✓ Transcoding completed in {elapsed:.1f}s | Size: {size_mb:.1f} MB | Duration: {transcoded_dur:.1f}s")
            
            # Atomic replacement: rename/replace source file with transcoded file
            backup_path = video_path + ".orig"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            shutil.move(video_path, backup_path)
            shutil.move(temp_transcoded, video_path)
            
            logger.info(f"[TRANSCODE] ✓ Successfully replaced original {codec} video with H.264 version.")
            
            # Clean up the backup to avoid duplicating large files
            try:
                os.remove(backup_path)
                logger.info("[TRANSCODE] ✓ Cleaned up backup file.")
            except OSError as e:
                logger.warning(f"[TRANSCODE] Could not remove backup file: {e}")
                
            return video_path
        else:
            raise RuntimeError("Transcoded output file is missing or too small")
            
    except Exception as e:
        logger.error(f"[TRANSCODE] ❌ Transcoding failed: {e}")
        if os.path.exists(temp_transcoded):
            try:
                os.remove(temp_transcoded)
            except OSError:
                pass
        logger.warning("[TRANSCODE] ⚠ Falling back to original source video. Processing will be slow!")
        return video_path
