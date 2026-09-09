"""
Diagnostic script to identify YouTube download quality issues
"""
import subprocess
import sys
import json
from pathlib import Path

test_url = "https://www.youtube.com/watch?v=Q2xs6xiW9hM"

def check_available_formats():
    """Check what formats are available for the test video"""
    print("=" * 80)
    print("STEP 1: Checking available formats")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-json",
        test_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    info = json.loads(result.stdout)
    
    print(f"Video title: {info.get('title')}")
    print(f"Duration: {info.get('duration')} seconds")
    print(f"Available formats: {len(info.get('formats', []))}")
    
    # Find best video quality available
    formats = info.get('formats', [])
    video_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('height')]
    video_formats.sort(key=lambda x: x.get('height', 0), reverse=True)
    
    print("\nTop 5 video formats by resolution:")
    for i, fmt in enumerate(video_formats[:5]):
        print(f"  {i+1}. {fmt.get('format_id'):4s} | {fmt.get('ext'):5s} | "
              f"{fmt.get('width')}x{fmt.get('height')} | "
              f"{fmt.get('vcodec', 'unknown'):15s} | "
              f"{fmt.get('tbr', 0):.0f}k")
    
    # Check audio formats
    audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
    audio_formats.sort(key=lambda x: x.get('abr', 0), reverse=True)
    
    print("\nTop 3 audio formats:")
    for i, fmt in enumerate(audio_formats[:3]):
        print(f"  {i+1}. {fmt.get('format_id'):4s} | {fmt.get('ext'):5s} | "
              f"{fmt.get('acodec', 'unknown'):15s} | "
              f"{fmt.get('abr', 0):.0f}k")
    
    return info

def check_current_download_command():
    """Show what the current code would download"""
    print("\n" + "=" * 80)
    print("STEP 2: Current download logic analysis")
    print("=" * 80)
    
    print("\nCurrent video selection: bestvideo[height<=2160]/bestvideo")
    print("Current audio selection: bestaudio/best")
    print("\nWhat this means:")
    print("  - Video: Downloads best quality up to 4K (2160p)")
    print("  - Audio: Downloads best audio quality available")
    print("  - Merges them with FFmpeg (stream copy)")
    
    # Simulate what would be selected
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=2160]/bestvideo",
        "--get-format",
        test_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    video_format = result.stdout.strip() if result.returncode == 0 else "ERROR"
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--get-format",
        test_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    audio_format = result.stdout.strip() if result.returncode == 0 else "ERROR"
    
    print(f"\nSelected video format: {video_format}")
    print(f"Selected audio format: {audio_format}")

def check_codec_compatibility():
    """Check if video codecs need transcoding"""
    print("\n" + "=" * 80)
    print("STEP 3: Codec compatibility check")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=2160]/bestvideo",
        "--print", "%(vcodec)s",
        test_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    vcodec = result.stdout.strip() if result.returncode == 0 else "unknown"
    
    print(f"Video codec: {vcodec}")
    
    if vcodec.startswith("vp9"):
        print("⚠️  VP9 codec detected - will be transcoded to H.264")
        print("    Reason: VP9 not supported by GPU hardware decoder (NVDEC)")
    elif vcodec.startswith("av01"):
        print("⚠️  AV1 codec detected - will be transcoded to H.264")
        print("    Reason: GTX 1650 lacks AV1 hardware decoder")
    elif vcodec.startswith("avc1") or vcodec.startswith("h264"):
        print("✓  H.264 codec - no transcoding needed!")
    else:
        print(f"⚠️  {vcodec} codec may need transcoding")

def check_transcode_settings():
    """Check transcode_helper configuration"""
    print("\n" + "=" * 80)
    print("STEP 4: Transcode settings")
    print("=" * 80)
    
    import config
    
    print(f"Auto-transcode enabled: {config.AUTO_TRANSCODE_TO_H264}")
    print(f"NVENC preset: {config.NVENC_PRESET}")
    print(f"NVENC CQ: {config.NVENC_CQ}")
    print(f"FFmpeg path: {config.FFMPEG_PATH}")
    print(f"GPU FFmpeg path: {config.GPU_FFMPEG_PATH}")

def identify_root_causes():
    """Summarize the root causes"""
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print("\n🔍 FINDINGS:")
    print("\n1. FORMAT SELECTION:")
    print("   Current: bestvideo[height<=2160]/bestvideo + bestaudio/best")
    print("   Issue: This often selects VP9 or AV1 codecs (better compression)")
    print("   Impact: These require CPU decode + GPU encode transcoding")
    
    print("\n2. TRANSCODING PIPELINE:")
    print("   - AV1/VP9 videos are detected after download")
    print("   - ensure_h264_source() transcodes them to H.264")
    print("   - Uses h264_nvenc (GPU encoding) but CPU decoding")
    print("   - This is SLOW for 4K content")
    
    print("\n3. WORKFLOW INEFFICIENCY:")
    print("   - Downloads high-quality VP9/AV1 (smaller file)")
    print("   - Then transcodes to H.264 (larger file)")
    print("   - Two-step process wastes time and disk space")
    
    print("\n💡 SOLUTIONS:")
    print("\n   Option A: Prefer H.264 at download time (RECOMMENDED)")
    print("   Change format to: bestvideo[vcodec^=avc1][height<=2160]/bestvideo[height<=2160]")
    print("   Benefit: No transcoding needed, direct to H.264")
    print("   Tradeoff: Slightly larger download size")
    
    print("\n   Option B: Download lower resolution VP9")
    print("   Change format to: bestvideo[height<=1080]/bestvideo")
    print("   Benefit: Smaller file, faster transcode")
    print("   Tradeoff: Lower quality (but still good for vertical crops)")
    
    print("\n   Option C: GPU-accelerated download & transcode")
    print("   Use NVDEC for decode + NVENC for encode")
    print("   Benefit: Much faster transcoding")
    print("   Tradeoff: Requires ffmpeg built with CUDA support")
    
    print("\n   Option D: Smart format selection")
    print("   Check available formats, prefer H.264 if available, fallback to VP9")
    print("   Benefit: Best quality without unnecessary transcoding")
    print("   Tradeoff: More complex logic")

if __name__ == "__main__":
    info = check_available_formats()
    check_current_download_command()
    check_codec_compatibility()
    check_transcode_settings()
    identify_root_causes()
