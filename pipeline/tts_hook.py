# fileName: pipeline/tts_hook.py
"""
TTS Hook Overlay Module
=======================
Handles Smallest.ai TTS voice generation, face-aware kinetic text positioning,
and seamless prepending of the hook intro segment to extracted video clips.
"""

import os
import time
import wave
import subprocess
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from pipeline.speaker_tracking import _get_face_detector, _detect_faces


def generate_hook_audio(
    text: str,
    output_path: str,
    voice_id: str = "jessica",
    sample_rate: int = 44100,
    speed: float = 1.0,
    api_key: str = "",
    logger = None,
) -> dict:
    """
    Synthesize hook phrase using Smallest AI's Lightning v3.1 endpoint.
    Returns dict with success status and duration of the generated audio.
    """
    if not api_key:
        if logger:
            logger.warning("Smallest AI API key is missing. Skipping speech generation.")
        return {"success": False, "duration_s": 2.0}

    endpoint_url = "https://api.smallest.ai/waves/v1/lightning-v3.1/get_speech"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text.lower(),
        "voice_id": voice_id,
        "sample_rate": sample_rate,
        "speed": speed,
        "output_format": "wav"
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if logger:
                logger.info(f"Generating speech via Smallest AI for voice '{voice_id}' (attempt {attempt}/{max_retries})...")
            
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                # Read duration from WAV
                try:
                    with wave.open(output_path, "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                    if logger:
                        logger.info(f"✓ Speech generated successfully on attempt {attempt}: {output_path} ({duration:.2f}s)")
                    return {"success": True, "duration_s": duration}
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to parse WAV duration: {e}. Defaulting to 2.0s.")
                    return {"success": True, "duration_s": 2.0}
            else:
                if logger:
                    logger.warning(f"Smallest AI API attempt {attempt} failed ({response.status_code}): {response.text}")
        except Exception as exc:
            if logger:
                logger.warning(f"Smallest AI API attempt {attempt} threw exception: {exc}")
        
        if attempt < max_retries:
            time.sleep(2)
            
    if logger:
        logger.error(f"All {max_retries} attempts to call Smallest AI TTS API failed.")
    return {"success": False, "duration_s": 2.0}


def _sanitize_hook_phrase(raw_hook: str) -> str:
    import re
    cleaned = re.sub(r"[^\w\s]", "", str(raw_hook or "")).replace("_", "")
    cleaned = cleaned.replace("’", "").replace("‘", "").replace("”", "").replace("“", "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def build_hook_intro(
    raw_clip_path: str,
    hook_phrase: str,
    tts_wav_path: str | None,
    output_path: str,
    duration_s: float,
    settings: dict,
    logger,
    speech_duration_s: float | None = None,
) -> bool:
    hook_phrase = _sanitize_hook_phrase(hook_phrase).upper()
    """
    Generate the hook intro video segment frame-by-frame:
    - B&W desaturation + Gaussian blur background
    - Ken Burns slow zoom (1.0 -> zoom_factor)
    - Face-aware vertical text alignment (no overlay on speaker face)
    - Percentage-based responsive font sizing
    - Word-by-word pop-in animation (active word in vibrant yellow, revealed in light yellow)
    Muxes the TTS WAV (or silence if TTS failed) and saves to output_path.
    """
    # 1. Open the source clip to get properties and the first frame
    cap = cv2.VideoCapture(raw_clip_path)
    if not cap.isOpened():
        logger.error(f"Could not open raw clip: {raw_clip_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        logger.error(f"Failed to read first frame from: {raw_clip_path}")
        return False

    # 2. Get configurations
    font_size_pct = float(settings.get("tts_hook_font_size_pct", 5.5))
    zoom_factor = float(settings.get("tts_hook_zoom_factor", 0.06))
    blur_strength = int(settings.get("tts_hook_bg_blur", 12))
    line_height_ratio = float(settings.get("tts_hook_line_height_ratio", 1.02))
    stroke_width = int(settings.get("tts_hook_stroke_width", 3))
    
    active_color = settings.get("tts_hook_text_color", "#FFD600")
    revealed_color = settings.get("tts_hook_text_color_revealed", "#FFFFB0")
    placement = settings.get("tts_hook_text_placement", "auto")

    # 3. Font size derived from tts_hook_font_size_pct (% of clip height), so the
    #    configured TTS_HOOK_FONT_SIZE_PCT actually takes effect instead of a
    #    hard-coded 120px. Clamped to a sane range.
    font_size = int(height * (font_size_pct / 100.0))
    font_size = max(40, min(font_size, int(height * 0.30)))
    font_path = getattr(config, "TTS_HOOK_FONT", "assets/fonts/BarlowCondensed-Black.ttf")
    if not os.path.exists(font_path):
        # Fallback search inside project
        font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", "BarlowCondensed-Black.ttf")

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.warning(f"Could not load Barlow font from {font_path}: {e}. Falling back to default font.")
        font = ImageFont.load_default()

    # 4. Wrap text and prevent overflow
    max_w = int(width * 0.85)
    words = hook_phrase.strip().split()
    if not words:
        words = ["Listen", "To", "This"]

    def wrap_text(words_list, current_font, max_width):
        lines_list = []
        current_line = []
        for w in words_list:
            test_line = " ".join(current_line + [w])
            try:
                line_w = current_font.getlength(test_line)
            except AttributeError:
                # Pillow fallback for older versions
                line_w = current_font.getbbox(test_line)[2]
            
            if line_w <= max_width:
                current_line.append(w)
            else:
                if current_line:
                    lines_list.append(" ".join(current_line))
                    current_line = [w]
                else:
                    lines_list.append(w)
                    current_line = []
        if current_line:
            lines_list.append(" ".join(current_line))
        return lines_list

    lines = wrap_text(words, font, max_w)
    
    # Scale down font size dynamically if text takes more than 2 lines
    current_font_size = font_size
    while len(lines) > 4 and current_font_size > int(height * 0.03):
        current_font_size = int(current_font_size * 0.85)
        try:
            font = ImageFont.truetype(font_path, current_font_size)
        except Exception:
            break
        lines = wrap_text(words, font, max_w)

    # 5. Speaker's face detection to position text
    face_y1, face_y2 = None, None
    detector = _get_face_detector(logger)
    if detector is not None:
        try:
            faces = _detect_faces(first_frame, detector)
            if faces:
                # Pick the largest face
                best_f = max(faces, key=lambda f: f.get("area", 0.0))
                face_y1 = best_f.get("y1")
                face_y2 = best_f.get("y2")
                logger.info(f"Speaker face detected at Y range: {face_y1:.1f} - {face_y2:.1f}")
        except Exception as e:
            logger.warning(f"Face detection on first frame failed: {e}")

    # Determine visual text block positioning
    line_height = int(current_font_size * line_height_ratio)
    total_text_h = len(lines) * line_height

    if face_y1 is not None and face_y2 is not None:
        if placement == "auto":
            # If the face is in the top-middle part, place text below face.
            # Otherwise, place text above face (centered in the top region).
            if face_y1 > height * 0.35:
                # Place above
                target_center_y = face_y1 / 2.0
            else:
                # Place below
                target_center_y = (face_y2 + height * 0.70) / 2.0
        elif placement == "above_face":
            target_center_y = face_y1 / 2.0
        elif placement == "below_face":
            target_center_y = (face_y2 + height * 0.70) / 2.0
        elif placement == "top":
            target_center_y = height * 0.25
        elif placement == "bottom":
            target_center_y = height * 0.65
        else:
            target_center_y = height * 0.35
    else:
        # No face detected fallbacks
        if placement == "top":
            target_center_y = height * 0.25
        elif placement == "bottom":
            target_center_y = height * 0.65
        else:
            target_center_y = height * 0.35

    start_y = int(target_center_y - total_text_h / 2.0)
    # Clamp to ensure text is fully within screen limits
    start_y = max(int(height * 0.05), min(start_y, int(height * 0.85 - total_text_h)))
    logger.info(f"Positioning hook text block Y start: {start_y}px (height={height}px, style={placement})")

    total_frames = max(1, int(duration_s * fps))
    
    # 7. Apply background operations once and save to temp image
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    bw_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ksize = blur_strength * 2 + 1
    blurred = cv2.GaussianBlur(bw_frame, (ksize, ksize), 0)
    bg_img_path = output_path + ".bg.jpg"
    
    # Use imencode to handle unicode paths on Windows correctly
    is_success, im_buf_arr = cv2.imencode(".jpg", blurred)
    if is_success:
        im_buf_arr.tofile(bg_img_path)
    else:
        logger.error(f"Failed to encode background image: {bg_img_path}")
        return False
    
    # 8. Construct FFmpeg filter complex
    ffmpeg_path = config.GPU_FFMPEG_PATH if getattr(config, "VIDEO_ENCODER", "libx264") == "h264_nvenc" else config.FFMPEG_PATH
    
    zoom_expr = f"1.0+({zoom_factor}*in/{total_frames})"
    fontfile_escaped = font_path.replace("\\", "/").replace(":", "\\:")
    active_color_str = active_color.lstrip("#")
    revealed_color_str = revealed_color.lstrip("#")
    
    total_words = sum(len(line.split()) for line in lines)
    # Word pop-ins pace against the SPEECH length, not the padded intro length,
    # so the text stays in sync with the voice through the silent tail.
    speech_dur = speech_duration_s if speech_duration_s and speech_duration_s > 0 else duration_s
    word_dur = speech_dur / max(1, total_words)
    word_index = 0
    
    drawtexts = []
    current_y = start_y
    for line in lines:
        line_words = line.split()
        try:
            line_w = font.getlength(line)
            space_w = font.getlength(" ")
        except AttributeError:
            line_w = font.getbbox(line)[2]
            space_w = font.getbbox(" ")[2]
            
        start_x = (width - line_w) / 2
        current_x = start_x
        
        for w in line_words:
            safe_w = w.replace("'", "\\'").replace(":", "\\:")
            try:
                word_w = font.getlength(w)
            except AttributeError:
                word_w = font.getbbox(w)[2]
                
            t_start = word_index * word_dur
            t_next = (word_index + 1) * word_dur
            
            # Active pop-in (Bright Yellow)
            drawtexts.append(
                f"drawtext=fontfile='{fontfile_escaped}':text='{safe_w}':fontsize={current_font_size}:"
                f"fontcolor=0x{active_color_str}:bordercolor=black:borderw={stroke_width}:"
                f"shadowcolor=black@0.6:shadowx=3:shadowy=3:x={current_x}:y={current_y}:"
                f"enable='between(t,{t_start:.3f},{t_next:.3f})'"
            )
            
            # Revealed state (Faint Yellow)
            drawtexts.append(
                f"drawtext=fontfile='{fontfile_escaped}':text='{safe_w}':fontsize={current_font_size}:"
                f"fontcolor=0x{revealed_color_str}:bordercolor=black:borderw={stroke_width}:"
                f"shadowcolor=black@0.6:shadowx=3:shadowy=3:x={current_x}:y={current_y}:"
                f"enable='gte(t,{t_next:.3f})'"
            )
            
            current_x += word_w + space_w
            word_index += 1
            
        current_y += line_height
        
    v_filter = f"[0:v]zoompan=z='{zoom_expr}':d={total_frames}:s={width}x{height}:fps={fps}[bg]"
    vout_map = "[bg]"
    if drawtexts:
        v_filter += f";[bg]{','.join(drawtexts)}[vout]"
        vout_map = "[vout]"
        
    filter_parts = [v_filter]
    
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", bg_img_path
    ]
    
    # Audio inputs
    sample_rate = int(settings.get("tts_hook_sample_rate", 44100))
    tts_music_path = getattr(config, "TTS_HOOK_MUSIC", "")
    has_music = os.path.exists(tts_music_path) if tts_music_path else False

    # Voice polish: low-shelf bass lift + gentle compression, capped by a
    # limiter — the "breaky" crackle on loud hooks is digital clipping (not a
    # sample-rate problem) and the limiter makes the boost clip-proof. apad
    # keeps audio flowing through the post-speech tail so the hook->clip
    # concat never sees a gap in the audio stream.
    bass_gain = float(settings.get(
        "tts_hook_bass_gain_db", getattr(config, "TTS_HOOK_BASS_GAIN_DB", 6.0)))
    voice_chain = (
        f"highpass=f=55,bass=g={bass_gain:.1f}:f=110,"
        f"acompressor=threshold=0.125:ratio=2.5:attack=8:release=140,"
        f"alimiter=limit=0.891,apad"
    )

    audio_map = ""
    if tts_wav_path and os.path.exists(tts_wav_path):
        cmd += ["-i", tts_wav_path]
        if has_music:
            cmd += ["-i", tts_music_path]
            fade_st = max(0.0, duration_s - 0.5)
            a_filter = (
                f"[1:a]{voice_chain}[voice];"
                f"[2:a]volume=0.10,afade=t=out:st={fade_st:.3f}:d=0.5[bgm];"
                f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                f"alimiter=limit=0.891[aout]"
            )
            filter_parts.append(a_filter)
            audio_map = "[aout]"
        else:
            filter_parts.append(f"[1:a]{voice_chain}[aout]")
            audio_map = "[aout]"
    else:
        cmd += [
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}"
        ]
        if has_music:
            cmd += ["-i", tts_music_path]
            fade_st = max(0.0, duration_s - 0.5)
            a_filter = f"[2:a]volume=0.10,afade=t=out:st={fade_st:.3f}:d=0.5[aout]"
            filter_parts.append(a_filter)
            audio_map = "[aout]"
        else:
            audio_map = "1:a"
            
    filter_complex_str = ";".join(filter_parts)
    filter_script_path = output_path + ".filter.txt"
    try:
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(filter_complex_str)
    except Exception as e:
        logger.error(f"Failed to write FFmpeg filter script: {e}")
        return False

    cmd += ["-filter_complex_script", filter_script_path]
    cmd += ["-map", vout_map]
    if audio_map:
        cmd += ["-map", audio_map]
        
    cmd += [
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    
    try:
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            if proc.returncode != 0:
                logger.error(f"FFmpeg returned error {proc.returncode}. Stderr: {proc.stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg for hook intro: {e}")
            return False
    finally:
        # Cleanup temp image and filter script
        if os.path.exists(bg_img_path):
            try:
                os.remove(bg_img_path)
            except:
                pass
        if os.path.exists(filter_script_path):
            try:
                os.remove(filter_script_path)
            except:
                pass
            
    return True


def apply_hooks_to_clips(
    job_dir: str,
    clips_plan: list[dict],
    settings: dict,
    logger,
) -> None:
    """
    For each clip in clips_plan:
    1. Check if TTS Hook is enabled.
    2. Generate Smallest AI speech synthesized audio.
    3. Generate the blurred desaturated zooming visual intro segment.
    4. Concat hook intro segment at the start of the raw clip video.
    """
    enabled = settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED)
    if not enabled:
        logger.info("TTS Hook Overlay feature is disabled. Removing any existing hooked clips to revert to raw clips.")
        for clip in clips_plan:
            clip["tts_hook_duration"] = 0.0
            hooked_path = os.path.join(job_dir, "clips", f"{clip['clip_name']}_hooked.mp4")
            wav_path = os.path.join(job_dir, "clips", "hooks", f"hook_{clip['clip_name']}.wav")
            intro_video_path = os.path.join(job_dir, "clips", "hooks", f"intro_{clip['clip_name']}.mp4")
            
            import glob
            stale_files = [hooked_path, wav_path, intro_video_path]
            stale_patterns = [
                os.path.join(job_dir, "clips", f"{clip['clip_name']}_captioned*_hooked.mp4")
            ]
            for pattern in stale_patterns:
                stale_files.extend(glob.glob(pattern))
                
            for path in stale_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info(f"Removed stale hooked output/asset: {path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {path}: {e}")
        # Save updated plan
        import json
        plan_path = os.path.join(job_dir, "clips_plan.json")
        try:
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(clips_plan, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return

    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    
    # Subfolder for intermediate hook segments
    hooks_dir = os.path.join(clips_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)
    
    api_key = settings.get("smallest_key") or getattr(config, "SMALLEST_API_KEY", "")
    voice_id = settings.get("tts_hook_voice_id", config.TTS_HOOK_VOICE_ID)
    sample_rate = int(settings.get("tts_hook_sample_rate", config.TTS_HOOK_SAMPLE_RATE))
    speed = float(settings.get("tts_hook_speed", config.TTS_HOOK_SPEED))
    tail_s = max(0.0, float(settings.get(
        "tts_hook_tail_s", getattr(config, "TTS_HOOK_TAIL_S", 0.2))))

    for clip in clips_plan:
        clip_name = clip["clip_name"]
        raw_hook = (clip.get("hook_phrase") or clip.get("title") or "Check This Out")
        hook_phrase = _sanitize_hook_phrase(raw_hook)
        
        logger.info(f"--- Processing TTS Hook for {clip_name}: '{hook_phrase}' ---")
        
        # Paths
        wav_path = os.path.join(hooks_dir, f"hook_{clip_name}.wav")
        intro_video_path = os.path.join(hooks_dir, f"intro_{clip_name}.mp4")
        raw_clip_path = os.path.join(clips_dir, f"{clip_name}_raw.mp4")
        temp_final_path = os.path.join(clips_dir, f"{clip_name}_final_hooked.mp4")
        hooked_clip_path = os.path.join(clips_dir, f"{clip_name}_hooked.mp4")

        clip["tts_hook_duration"] = 0.0

        target_video_exists = os.path.exists(intro_video_path)

        if target_video_exists and os.path.exists(wav_path):
            try:
                # Use the intro video's REAL length as the shift (older hooks
                # were rendered without the post-speech tail, newer ones with
                # it — the shift must match the file on disk either way).
                duration_s = 0.0
                icap = cv2.VideoCapture(intro_video_path)
                if icap.isOpened():
                    ifps = icap.get(cv2.CAP_PROP_FPS) or 0.0
                    ifrm = icap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                    if ifps > 0 and ifrm > 0:
                        duration_s = ifrm / ifps
                icap.release()
                if duration_s <= 0:
                    import wave
                    with wave.open(wav_path, "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                    duration_s = frames / float(rate)
                clip["tts_hook_duration"] = duration_s
                
                # Delete any legacy hooked clip to avoid redundant files
                if os.path.exists(hooked_clip_path):
                    try:
                        os.remove(hooked_clip_path)
                    except Exception:
                        pass
                            
                logger.info(f"Skipping TTS hook for {clip_name} (already exists)")
                continue
            except Exception as e:
                logger.warning(f"Hook exists but failed to read wav duration: {e}. Regenerating...")

        if not os.path.exists(raw_clip_path):
            logger.warning(f"Raw clip video file not found at: {raw_clip_path}. Skipping hook.")
            continue

        # Step A: Synthesize Speech
        tts_res = generate_hook_audio(
            text=hook_phrase,
            output_path=wav_path,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            api_key=api_key,
            logger=logger
        )
        
        speech_s = tts_res["duration_s"]
        # Tail keeps the intro rolling briefly after the voice stops so the
        # speech never feels cut off at the transition into the clip.
        duration_s = speech_s + tail_s
        clip["tts_hook_duration"] = duration_s

        # If API key is empty or TTS failed, wav_path won't exist. Pass None to build_hook_intro
        tts_wav = wav_path if tts_res.get("success") and os.path.exists(wav_path) else None

        # Step B: Render visual intro segment
        logger.info(f"Rendering intro visual segment ({duration_s:.2f}s incl. {tail_s:.2f}s tail) for {clip_name}...")
        success = build_hook_intro(
            raw_clip_path=raw_clip_path,
            hook_phrase=hook_phrase,
            tts_wav_path=tts_wav,
            output_path=intro_video_path,
            duration_s=duration_s,
            settings=settings,
            logger=logger,
            speech_duration_s=speech_s,
        )

        if not success or not os.path.exists(intro_video_path):
            logger.error(f"Failed to generate intro segment for {clip_name}. Hook skipped.")
            continue

        # Delete any legacy hooked clip to avoid redundant files
        if os.path.exists(hooked_clip_path):
            try:
                os.remove(hooked_clip_path)
            except Exception:
                pass
            
        # We preserve the generated visual/audio hook segments inside clips/hooks/ 
        # as requested, so we comment out the cleanup code below:
        # if os.path.exists(intro_video_path):
        #     try:
        #         os.remove(intro_video_path)
        #     except Exception:
        #         pass

    # Save the updated clips_plan.json to disk
    import json
    plan_path = os.path.join(job_dir, "clips_plan.json")
    try:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(clips_plan, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully serialized clips_plan.json with tts_hook_duration.")
    except Exception as e:
        logger.error(f"Failed to save clips_plan.json: {e}")
