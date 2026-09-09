# ====================================================================
# pipeline/downloader.py — Integrated YouTube Video Downloader
# ====================================================================
# Mirrors the root test_download.py module exactly
# ====================================================================

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import config


def _extract_youtube_id(url: str) -> str | None:
    """Extract 11-character YouTube video ID from any YouTube URL format.
    
    Handles: desktop, mobile, music, shorts, embeds, live, playlist items, unlisted.
    """
    url = url.strip()
    if not url:
        return None
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
    except Exception:
        hostname = ""
        parsed = None

    video_id = None

    if parsed and hostname:
        if hostname == "youtu.be" or hostname.endswith(".youtu.be"):
            # Path is /VIDEO_ID
            path_parts = [p for p in parsed.path.split('/') if p]
            if path_parts:
                candidate = path_parts[0]
                if len(candidate) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', candidate):
                    video_id = candidate
        elif "youtube.com" in hostname:
            # First try query param 'v'
            queries = parse_qs(parsed.query)
            if 'v' in queries and queries['v']:
                candidate = queries['v'][0]
                if len(candidate) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', candidate):
                    video_id = candidate
            
            # If not in query, check path segments (e.g., shorts, embed, v, live, clip)
            if not video_id:
                path_parts = [p for p in parsed.path.split('/') if p]
                # Look for segments after shorts/embed/v/live/clip
                for i, part in enumerate(path_parts):
                    if part in ("shorts", "embed", "v", "live", "clip") and i + 1 < len(path_parts):
                        candidate = path_parts[i+1]
                        if len(candidate) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', candidate):
                            video_id = candidate
                            break
                # Fallback: check if any segment is 11 chars
                if not video_id:
                    for part in path_parts:
                        if len(part) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', part):
                            video_id = part
                            break

    # Robust regex fallbacks in case URL is malformed or custom
    if not video_id:
        m = re.search(r'(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})', url, re.IGNORECASE)
        if m:
            video_id = m.group(1)

    if not video_id:
        m = re.search(r'[?&]v=([A-Za-z0-9_-]{11})', url)
        if m:
            video_id = m.group(1)

    if not video_id:
        m = re.search(r'/(?:shorts|embed|v|live|clip)/([A-Za-z0-9_-]{11})', url, re.IGNORECASE)
        if m:
            video_id = m.group(1)

    return video_id


def _normalise_youtube_url(url: str) -> str:
    """Convert any YouTube URL variant to https://www.youtube.com/watch?v=VIDEO_ID.

    Handles: youtu.be, youtube.com/watch, /shorts/, /embed/, /v/, /live/
    Strips: list, index, si, t, pp, and all other query params.
    """
    video_id = _extract_youtube_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    # Not a recognised YouTube URL — return as-is (yt-dlp supports other sites too)
    return url


def ensure_ytdlp(logger=None):
    """Ensure yt-dlp is installed"""
    try:
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True,
            check=True
        )
        if logger:
            logger.debug("✓ yt-dlp found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        if logger:
            logger.info("Installing yt-dlp...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "yt-dlp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if logger:
            logger.info("✓ yt-dlp installed")


def check_ffmpeg(logger=None):
    """Check if FFmpeg is available and return its path"""
    local_ffmpeg = Path(__file__).parent.parent / "ffmpeg-gpu" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        if logger:
            logger.debug(f"✓ FFmpeg found (local): {local_ffmpeg}")
        return str(local_ffmpeg)
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise SystemExit("FFmpeg is required but not found")
    if logger:
        logger.debug(f"✓ FFmpeg found: {ffmpeg_path}")
    return ffmpeg_path


def download_audio_track(job_dir: str, url: str, logger: logging.Logger) -> str:
    """Download a YouTube video's audio as MP3 for background music."""
    import datetime
    url = _normalise_youtube_url(url)
    logger.info(f"Downloading background audio: {url}")
    ffmpeg = check_ffmpeg(logger)
    ensure_ytdlp(logger)

    video_id = _extract_youtube_id(url) or "custom_track"

    os.makedirs(config.MUSIC_DIR, exist_ok=True)

    index_path = os.path.join(config.MUSIC_DIR, "_index.json")
    index = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load music index: {e}")

    if url in index:
        info = index[url]
        final_path = os.path.join(config.MUSIC_DIR, info["filename"])
        if os.path.exists(final_path):
            logger.info(f"Background music already in index: {final_path} (Title: {info.get('title')})")
            return os.path.abspath(final_path)

    final = os.path.join(config.MUSIC_DIR, f"{video_id}.mp3")
    if os.path.exists(final):
        logger.info(f"Background music already in library: {final}")
        if url not in index:
            try:
                title_cmd = [sys.executable, "-m", "yt_dlp", "--print", "%(title)s", url]
                title = subprocess.check_output(title_cmd, stderr=subprocess.DEVNULL, text=True).strip()
            except Exception:
                title = f"YouTube Track ({video_id})"
            index[url] = {
                "filename": f"{video_id}.mp3",
                "title": title,
                "video_id": video_id,
                "downloaded_at": datetime.datetime.now().isoformat()
            }
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        return os.path.abspath(final)

    tmp = os.path.join(config.MUSIC_DIR, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    out_tmpl = os.path.join(tmp, f"{video_id}_temp.%(ext)s")
    out_mp3 = os.path.join(tmp, f"{video_id}_temp.mp3")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings",
        "--extractor-args", "youtube:player_client=web,android",
        "--ffmpeg-location", ffmpeg,
        "-f", "bestaudio/best",
        "-x", "--audio-format", "mp3", "--audio-quality", "320K",
        "-o", out_tmpl,
        url,
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    cookies = None
    for candidate in [project_root / "cookies.txt", Path.cwd() / "cookies.txt"]:
        if candidate.is_file():
            cookies = str(candidate)
            break
    if cookies:
        cmd += ["--cookies", cookies]

    _run_cmd_with_progress(cmd, logger, "download")

    if not os.path.exists(out_mp3):
        raise RuntimeError("Background music download produced no MP3 file")

    if os.path.exists(final):
        os.remove(final)
    shutil.move(out_mp3, final)

    try:
        if os.path.exists(out_mp3):
            os.remove(out_mp3)
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    try:
        title_cmd = [sys.executable, "-m", "yt_dlp", "--print", "%(title)s", url]
        if cookies:
            title_cmd += ["--cookies", cookies]
        title = subprocess.check_output(title_cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        title = f"YouTube Track ({video_id})"

    index[url] = {
        "filename": f"{video_id}.mp3",
        "title": title,
        "video_id": video_id,
        "downloaded_at": datetime.datetime.now().isoformat()
    }
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save music index: {e}")

    logger.info(f"Background music saved to library: {final} (Title: {title})")
    return os.path.abspath(final)


def _run_cmd_with_progress(
    cmd: list[str],
    logger: logging.Logger,
    phase: str,
    progress_callback=None,
    duration: float = None
) -> str:
    """
    Runs a subprocess (yt-dlp or ffmpeg) and parses percentage progress in real-time.
    Prints carriage-return progress to CMD and invokes progress_callback if provided.
    """
    logger.debug(f"Running command: {' '.join(str(c) for c in cmd)}")
    
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=creationflags
    )
    
    full_output = []
    last_reported_pct = -1
    
    ytdlp_pct_re = re.compile(r'(\d+(?:\.\d+)?)%')
    ffmpeg_time_re = re.compile(r'time=(\d+):(\d+):(\d+\.\d+)')
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if not line:
            continue
            
        full_output.append(line)
        
        pct = None
        if phase == "download":
            m = ytdlp_pct_re.search(line)
            if m:
                pct = int(float(m.group(1)))
        elif phase == "transcode" and duration:
            m = ffmpeg_time_re.search(line)
            if m:
                h, m_min, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
                total_secs = h * 3600 + m_min * 60 + s
                pct = int((total_secs / duration) * 100)
                pct = max(0, min(100, pct))
                
        if pct is not None and pct != last_reported_pct:
            last_reported_pct = pct
            sys.stdout.write(f"\r[{phase.upper()}] Progress: {pct}%")
            sys.stdout.flush()
            
            if pct % 10 == 0:
                logger.info(f"[{phase.upper()}] Progress: {pct}%")
                
            if progress_callback:
                try:
                    progress_callback(pct, phase)
                except Exception:
                    pass
                    
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    returncode = process.wait()
    if returncode != 0:
        output_str = "".join(full_output)
        err_summary = output_str[-1000:].strip()
        raise subprocess.CalledProcessError(
            returncode, cmd, output=output_str, stderr=err_summary
        )
    return "".join(full_output)


def _merge_audio_args(temp_audio: Path, ffprobe_path: str) -> list[str]:
    """AAC audio is stream-copied as-is; anything else (e.g. the 160k Opus
    that bestaudio now selects) is encoded once to 256k AAC so the MP4 stays
    universally readable downstream."""
    try:
        res = subprocess.run(
            [
                ffprobe_path, "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(temp_audio),
            ],
            capture_output=True, text=True, timeout=30,
        )
        codec = (res.stdout or "").strip().lower()
    except Exception:
        codec = ""
    if codec == "aac":
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "256k"]


def download_video(job_dir: str, url: str, logger: logging.Logger, progress_callback=None) -> dict:
    """
    Pipeline entry point for downloading a video in the best quality (up to 4K).
    Returns the metadata dictionary containing the video_filename.
    """
    url = _normalise_youtube_url(url)
    logger.info(f"Starting download from: {url}")
    ensure_ytdlp(logger)
    ffmpeg_path = check_ffmpeg(logger)

    # 1. Fetch metadata
    logger.info("Fetching video metadata...")
    meta_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-single-json", "--skip-download", "--no-playlist",
        url,
    ]
    result = subprocess.run(meta_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = (result.stderr or "").strip()[-500:]
        logger.error(f"yt-dlp metadata fetch failed: {err_msg}")
        raise RuntimeError(f"yt-dlp metadata failed for {url}: {err_msg}")
    info = json.loads(result.stdout)

    meta = {
        "title":       info.get("title",       "Unknown"),
        "description": info.get("description", ""),
        "channel":     info.get("channel",     info.get("uploader", "Unknown")),
        "duration":    info.get("duration",    0),
        "url":         url,
        "thumbnail":   info.get("thumbnail",   ""),
        "upload_date": info.get("upload_date", ""),
        "view_count":  info.get("view_count",  0),
    }

    clean_title = re.sub(r'[<>:"/\\|?*]', '', meta["title"])[:50].strip() or "video"
    video_filename = f"{clean_title}.mp4"
    meta["video_filename"] = video_filename

    output_path = Path(job_dir) / video_filename
    logger.info(f"Video will be saved as: {video_filename}")

    temp_dir = Path(job_dir) / "temp_download"
    temp_dir.mkdir(exist_ok=True)

    temp_video = temp_dir / "video.mp4"
    temp_audio = temp_dir / "audio.m4a"

    merge_success = False

    def make_unified_callback(start_weight, end_weight):
        def unified_cb(pct, phase):
            unified_pct = int(start_weight + (pct / 100.0) * (end_weight - start_weight))
            if progress_callback:
                try:
                    progress_callback(unified_pct, f"{phase} ({unified_pct}%)")
                except Exception:
                    pass
        return unified_cb

    try:
        # Download best video up to 4K (2160p)
        logger.info("  → Downloading best video stream (up to 4K)...")
        if progress_callback:
            progress_callback(0, "downloading video (0%)")
            
        video_cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[height<=2160]/bestvideo",
            "--no-playlist",
            "--force-overwrites",
            "-o", str(temp_video),
            url
        ]
        _run_cmd_with_progress(
            video_cmd, logger, "download",
            progress_callback=make_unified_callback(0, 60)
        )
        logger.info("  ✓ Video downloaded")

        # Download best audio
        logger.info("  → Downloading audio stream...")
        if progress_callback:
            progress_callback(60, "downloading audio (60%)")
            
        audio_cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestaudio/best",
            "--no-playlist",
            "--force-overwrites",
            "-o", str(temp_audio),
            url
        ]
        _run_cmd_with_progress(
            audio_cmd, logger, "download",
            progress_callback=make_unified_callback(60, 75)
        )
        logger.info("  ✓ Audio downloaded")

        # Merge using FFmpeg stream copy (FAST - no re-encoding)
        logger.info("  → Merging streams (stream copy)...")
        if progress_callback:
            progress_callback(77, "merging streams (77%)")

        input_size_mb = (temp_video.stat().st_size + temp_audio.stat().st_size) / (1024 ** 2)
        merge_timeout = max(300, int(input_size_mb / 3) + 120)

        ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe.exe"))

        merge_cmd = [
            ffmpeg_path, "-y", "-nostdin",
            "-i", str(temp_video),
            "-i", str(temp_audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            *_merge_audio_args(temp_audio, ffprobe_path),
            str(output_path),
        ]

        subprocess.run(
            merge_cmd,
            check=True,
            capture_output=True,
            timeout=merge_timeout,
            text=True
        )

        # Verify output with ffprobe
        logger.info("  → Validating merged output...")
        probe = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if probe.returncode != 0:
            raise RuntimeError(f"Merged output is invalid: {probe.stderr.strip()}")

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024**2)
            logger.info(f"  ✓ Merged successfully: {size_mb:.1f} MB")
            merge_success = True
        else:
            raise Exception("Merge failed - output file not created")

    except subprocess.TimeoutExpired:
        logger.error("  ❌ Merge timed out")
        raise
    except Exception as e:
        logger.error(f"  ❌ Error downloading: {e}")
        raise
    finally:
        if merge_success:
            if temp_video.exists():
                temp_video.unlink()
            if temp_audio.exists():
                temp_audio.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass
        else:
            logger.warning(f"Preserving temp download files for retry/debug: {temp_dir}")

    # Ensure video codec is supported (transcode AV1/VP9 to H.264 if needed)
    try:
        from pipeline.transcode_helper import ensure_h264_source
        if progress_callback:
            progress_callback(77, "checking codec (77%)")
        ensure_h264_source(
            str(output_path), logger,
            progress_callback=make_unified_callback(77, 100)
        )
    except Exception as e:
        logger.warning(f"Failed to check/transcode source video format: {e}")

    if progress_callback:
        progress_callback(100, "download complete")

    # Save meta
    meta_path = Path(job_dir) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def download_audio_only(job_dir: str, url: str, logger: logging.Logger, progress_callback=None) -> dict:
    """
    Downloads the audio stream only and transcodes it immediately to audio.wav.
    Also fetches and saves metadata.
    """
    url = _normalise_youtube_url(url)
    logger.info(f"[AUDIO-DL] Starting audio download from: {url}")
    ensure_ytdlp(logger)
    ffmpeg_path = check_ffmpeg(logger)

    # 1. Fetch metadata
    logger.info("[AUDIO-DL] Fetching video metadata...")
    meta_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-single-json", "--skip-download", "--no-playlist",
        url,
    ]
    result = subprocess.run(meta_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err_msg = (result.stderr or "").strip()[-500:]
        logger.error(f"yt-dlp metadata fetch failed: {err_msg}")
        raise RuntimeError(f"yt-dlp metadata failed for {url}: {err_msg}")
    info = json.loads(result.stdout)

    meta = {
        "title":       info.get("title",       "Unknown"),
        "description": info.get("description", ""),
        "channel":     info.get("channel",     info.get("uploader", "Unknown")),
        "duration":    info.get("duration",    0),
        "url":         url,
        "thumbnail":   info.get("thumbnail",   ""),
        "upload_date": info.get("upload_date", ""),
        "view_count":  info.get("view_count",  0),
    }

    clean_title = re.sub(r'[<>:"/\\|?*]', '', meta["title"])[:50].strip() or "video"
    video_filename = f"{clean_title}.mp4"
    meta["video_filename"] = video_filename

    # Save meta.json immediately
    meta_path = Path(job_dir) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Download best audio
    temp_dir = Path(job_dir) / "temp_download"
    temp_dir.mkdir(exist_ok=True)
    temp_audio = temp_dir / "audio.m4a"
    audio_wav_path = Path(job_dir) / "audio.wav"

    logger.info("[AUDIO-DL] Downloading audio stream...")
    if progress_callback:
        progress_callback(0, "downloading audio stream")

    audio_cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "--force-overwrites",
        "-o", str(temp_audio),
        url
    ]
    
    # Audio-only download mapped 0-80%
    def audio_pct_callback(pct, phase):
        if progress_callback:
            progress_callback(int(pct * 0.8), "downloading audio stream")
            
    _run_cmd_with_progress(audio_cmd, logger, "download", progress_callback=audio_pct_callback)
    logger.info("[AUDIO-DL] Audio downloaded successfully.")

    # Transcode to WAV (16kHz PCM, mono)
    logger.info("[AUDIO-DL] Transcoding audio to WAV...")
    if progress_callback:
        progress_callback(85, "transcoding audio to WAV")

    wav_cmd = [
        ffmpeg_path, "-y", "-nostdin",
        "-i", str(temp_audio),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_wav_path)
    ]
    subprocess.run(wav_cmd, check=True, capture_output=True)
    logger.info(f"[AUDIO-DL] WAV audio ready at: {audio_wav_path}")

    if progress_callback:
        progress_callback(100, "audio download complete")

    # Clean up temp audio stream file
    if temp_audio.exists():
        temp_audio.unlink()
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return meta


def download_video_stream_only(
    job_dir: str,
    url: str,
    meta: dict,
    logger: logging.Logger,
    progress_callback=None
) -> None:
    """
    Downloads the high quality 4K video stream only and merges it with the audio stream,
    saving the result as the video_filename specified in meta.
    Finally, automatically runs ensure_h264_source to transcode it via GPU NVENC.
    """
    url = _normalise_youtube_url(url)
    logger.info(f"[VIDEO-DL] Starting background 4K video download from: {url}")
    ensure_ytdlp(logger)
    ffmpeg_path = check_ffmpeg(logger)
    
    video_filename = meta.get("video_filename", "video.mp4")
    output_path = Path(job_dir) / video_filename

    # Background thread MUST not share temp_download/ with the foreground audio
    # download — same path raced and broke the bg thread on Win.
    temp_dir = Path(job_dir) / "temp_download_bg"
    temp_dir.mkdir(exist_ok=True)

    temp_video = temp_dir / "video.mp4"
    temp_audio = temp_dir / "audio.m4a"

    merge_success = False

    def make_unified_bg_callback(start_weight, end_weight):
        def unified_cb(pct, phase):
            unified_pct = int(start_weight + (pct / 100.0) * (end_weight - start_weight))
            if progress_callback:
                try:
                    progress_callback(unified_pct, phase)
                except Exception:
                    pass
        return unified_cb
    
    try:
        # Download best video up to 4K (2160p)
        logger.info("[VIDEO-DL] Downloading best video stream (up to 4K)...")
        if progress_callback:
            progress_callback(0, "download")
            
        video_cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[height<=2160]/bestvideo",
            "--no-playlist",
            "--force-overwrites",
            "-o", str(temp_video),
            url
        ]
        _run_cmd_with_progress(
            video_cmd, logger, "download",
            progress_callback=make_unified_bg_callback(0, 60)
        )
        logger.info("[VIDEO-DL] Video stream downloaded.")
        
        # Download best audio stream (needed for final merged file)
        logger.info("[VIDEO-DL] Downloading audio stream for merge...")
        if progress_callback:
            progress_callback(60, "download")
            
        audio_cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestaudio/best",
            "--no-playlist",
            "--force-overwrites",
            "-o", str(temp_audio),
            url
        ]
        _run_cmd_with_progress(
            audio_cmd, logger, "download",
            progress_callback=make_unified_bg_callback(60, 70)
        )
        logger.info("[VIDEO-DL] Audio stream downloaded.")
        
        # Merge streams
        logger.info("[VIDEO-DL] Merging video and audio streams...")
        if progress_callback:
            progress_callback(72, "merge")
            
        input_size_mb = (temp_video.stat().st_size + temp_audio.stat().st_size) / (1024 ** 2)
        merge_timeout = max(600, int(input_size_mb / 2) + 120)

        ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe.exe"))
        
        merge_cmd = [
            ffmpeg_path, "-y", "-nostdin",
            "-i", str(temp_video),
            "-i", str(temp_audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            *_merge_audio_args(temp_audio, ffprobe_path),
            str(output_path),
        ]
        
        subprocess.run(
            merge_cmd,
            check=True,
            capture_output=True,
            timeout=merge_timeout,
            text=True
        )
        logger.info(f"[VIDEO-DL] Merged output generated: {output_path}")
        merge_success = True
        
    except Exception as e:
        logger.error(f"[VIDEO-DL] Error during background video download/merge: {e}")
        raise
    finally:
        # Clean up temp files
        if merge_success:
            if temp_video.exists():
                temp_video.unlink()
            if temp_audio.exists():
                temp_audio.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass
        else:
            logger.warning(f"[VIDEO-DL] Preserving temp folder on failure: {temp_dir}")
            
    # Auto-transcode to GPU H.264 if needed
    try:
        from pipeline.transcode_helper import ensure_h264_source
        logger.info("[VIDEO-DL] Checking codec and encoding via GPU NVENC to H.264...")
        if progress_callback:
            progress_callback(75, "transcode")
        ensure_h264_source(
            str(output_path), logger,
            progress_callback=make_unified_bg_callback(75, 100)
        )
        logger.info("[VIDEO-DL] GPU transcode check/execution done.")
    except Exception as e:
        logger.warning(f"[VIDEO-DL] Failed to run ensure_h264_source: {e}")
        
    if progress_callback:
        progress_callback(100, "done")
