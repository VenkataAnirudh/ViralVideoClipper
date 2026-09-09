"""
Test script to verify AV1 download -> H.264 transcode workflow
"""
import sys
import os
import subprocess
import time
from pathlib import Path

test_url = "https://www.youtube.com/watch?v=Q2xs6xiW9hM"
test_dir = Path("test_transcode_output")
test_dir.mkdir(exist_ok=True)

print("=" * 80)
print("AV1 -> H.264 TRANSCODE WORKFLOW TEST")
print("=" * 80)

# Step 1: Download AV1 video
print("\n[STEP 1] Downloading AV1 4K video...")
download_cmd = [
    sys.executable, "-m", "yt_dlp",
    "-f", "bestvideo[height<=2160]/bestvideo",
    "-o", str(test_dir / "test_video.mp4"),
    test_url
]
t_start = time.time()
result = subprocess.run(download_cmd, capture_output=True, text=True)
download_time = time.time() - t_start

if result.returncode != 0:
    print(f"ERROR: Download failed - {result.stderr}")
    sys.exit(1)

print(f"[OK] Download completed in {download_time:.1f}s")

# Step 2: Check codec
print("\n[STEP 2] Checking video codec...")
video_path = test_dir / "test_video.mp4"
probe_cmd = [
    "ffmpeg-bin/ffprobe.exe", "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,width,height",
    "-of", "default=noprint_wrappers=1",
    str(video_path)
]
result = subprocess.run(probe_cmd, capture_output=True, text=True)
print(result.stdout)

# Step 3: Test transcode with GPU acceleration
print("\n[STEP 3] Testing GPU-accelerated transcode...")
output_path = test_dir / "test_transcoded.mp4"

transcode_cmd = [
    "ffmpeg-gpu/ffmpeg.exe", "-y", "-nostdin",
    "-hwaccel", "cuda",
    "-hwaccel_output_format", "cuda",
    "-i", str(video_path),
    "-c:v", "h264_nvenc",
    "-preset", "p1",
    "-cq", "23",
    "-b:v", "0",
    "-rc", "vbr",
    "-c:a", "copy",
    str(output_path)
]

print(f"Command: {' '.join(transcode_cmd)}")
print("\nTranscoding...")
t_start = time.time()
result = subprocess.run(transcode_cmd, capture_output=True, text=True)
transcode_time = time.time() - t_start

if result.returncode != 0:
    print(f"\n❌ TRANSCODE FAILED:")
    print(result.stderr)
    print("\nTrying without GPU acceleration...")
    
    # Fallback: CPU transcode
    transcode_cmd_cpu = [
        "ffmpeg-bin/ffmpeg.exe", "-y", "-nostdin",
        "-i", str(video_path),
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-cq", "23",
        "-b:v", "0",
        "-rc", "vbr",
        "-c:a", "copy",
        str(output_path)
    ]
    
    t_start = time.time()
    result = subprocess.run(transcode_cmd_cpu, capture_output=True, text=True)
    transcode_time = time.time() - t_start
    
    if result.returncode != 0:
        print(f"❌ CPU TRANSCODE ALSO FAILED:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"[OK] CPU transcode completed in {transcode_time:.1f}s (without GPU decode)")
else:
    print(f"[OK] GPU transcode completed in {transcode_time:.1f}s")

# Step 4: Verify output
print("\n[STEP 4] Verifying transcoded output...")
probe_cmd = [
    "ffmpeg-bin/ffprobe.exe", "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,width,height",
    "-of", "default=noprint_wrappers=1",
    str(output_path)
]
result = subprocess.run(probe_cmd, capture_output=True, text=True)
print(result.stdout)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Download time:   {download_time:.1f}s")
print(f"Transcode time:  {transcode_time:.1f}s")
print(f"Total time:      {download_time + transcode_time:.1f}s")
print(f"\nOriginal size:   {video_path.stat().st_size / (1024*1024):.1f} MB")
print(f"Transcoded size: {output_path.stat().st_size / (1024*1024):.1f} MB")

# Calculate FPS (video duration / transcode time)
dur_cmd = [
    "ffmpeg-bin/ffprobe.exe", "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    str(video_path)
]
result = subprocess.run(dur_cmd, capture_output=True, text=True)
duration = float(result.stdout.strip() or 0.0)
if duration > 0:
    speed = duration / transcode_time
    print(f"\nTranscode speed: {speed:.2f}x realtime")
    if speed < 1.0:
        print("[WARNING] Transcode is slower than realtime!")
    elif speed < 2.0:
        print("[WARNING] Transcode is slow - may need optimization")
    else:
        print("[OK] Transcode speed is good")

print("\n[OK] Test completed. Files saved in:", test_dir.absolute())
