"""
Video Transcriber (v2)
======================
Uses faster-whisper (CTranslate2) for GPU-accelerated transcription.
Supports multi-method fallback: Local GPU -> OpenRouter API -> Smallest.ai -> CPU.
Produces an enriched transcript JSON with word timestamps and speaker labels.
"""

import base64
import gc
import json
import logging
import os
import subprocess

import config


def transcribe_video(
    job_dir: str,
    meta: dict,
    settings: dict,
    logger: logging.Logger,
) -> dict:
    """
    Transcribe the downloaded video with word-level timestamps.
    Uses a fallback chain of transcription methods.

    Args:
        job_dir: Path to job output directory
        meta: Video metadata dict
        settings: Per-job settings dict
        logger: Job logger instance

    Returns:
        dict with keys: segments, words, language, duration
    """
    video_filename = meta.get("video_filename", "input_video.mp4")
    video_path = os.path.join(job_dir, video_filename)

    # Resolve dynamically if the video file does not exist
    if not os.path.exists(video_path):
        import re
        if "title" in meta:
            clean_title = re.sub(r'[<>:"/\\|?*]', '', meta["title"])[:50].strip()
            if clean_title:
                title_filename = f"{clean_title}.mp4"
                title_path = os.path.join(job_dir, title_filename)
                if os.path.exists(title_path):
                    video_path = title_path
                    video_filename = title_filename

    if not os.path.exists(video_path):
        try:
            for fn in os.listdir(job_dir):
                if fn.endswith(".mp4") and not fn.startswith("_") and not fn.endswith("_raw.mp4") and not fn.endswith("_captioned.mp4"):
                    candidate_path = os.path.join(job_dir, fn)
                    if os.path.isfile(candidate_path):
                        video_path = candidate_path
                        video_filename = fn
                        break
        except Exception:
            pass

    audio_path = os.path.join(job_dir, "audio.wav")

    if os.path.exists(audio_path):
        logger.info("Audio file already exists, skipping extraction")
    else:
        logger.info("Extracting audio from video...")
        _extract_audio(video_path, audio_path, logger)


    method = settings.get("transcription_method", config.TRANSCRIPTION_METHOD)
    if method == "auto":
        order = list(config.TRANSCRIPTION_FALLBACK_ORDER)
        logger.info(f"Transcription fallback order: {' -> '.join(order)}")
    else:
        order = [method]
        logger.info(f"Transcription method locked to: {method}")

    smallest_key = settings.get("smallest_key") or config.SMALLEST_API_KEY

    media_duration = meta.get("duration", 0) or _probe_audio_duration(audio_path, logger)

    dispatch = {
        "local": lambda: _transcribe_local(audio_path, "cuda", logger),
        "cpu": lambda: _transcribe_local(audio_path, "cpu", logger),
        "smallest": lambda: _transcribe_smallest(audio_path, smallest_key, media_duration, logger),
    }

    result = None
    used_method = None
    for current_method in order:
        fn = dispatch.get(current_method)
        if not fn:
            logger.warning(f"Unknown transcription method: {current_method}, skipping")
            continue

        if current_method == "smallest" and not smallest_key:
            logger.debug("Skipping Smallest.ai (no API key)")
            continue

        try:
            logger.info(f"Trying transcription method: {current_method}")
            result = fn()
            if result and result.get("segments"):
                used_method = current_method
                logger.info(f"Transcription succeeded with method: {current_method}")
                break
            logger.warning(f"Method {current_method} returned empty result, trying next...")
        except Exception as exc:
            logger.warning(f"Method {current_method} failed: {exc}, trying next...")

    if not result or not result.get("segments"):
        raise RuntimeError("All transcription methods failed")

    transcript = _build_transcript(result, meta, used_method, logger)

    # Restore sentence punctuation from pauses when the STT dropped it (e.g.
    # Smallest.ai). Self-guards on terminator density, so healthy transcripts
    # are left untouched. Failure here must never block transcription.
    if getattr(config, "PUNCT_RESTORE_ENABLED", True):
        try:
            restore_punctuation(
                transcript,
                logger,
                sentence_gap_s=getattr(config, "PUNCT_SENTENCE_GAP_S", 0.6),
                clause_gap_s=getattr(config, "PUNCT_CLAUSE_GAP_S", 0.32),
                force_max_words=getattr(config, "PUNCT_FORCE_MAX_WORDS", 28),
                min_density=getattr(config, "PUNCT_RESTORE_MIN_DENSITY", 0.025),
            )
        except Exception as exc:
            logger.warning(f"Punctuation restoration failed: {exc}; using raw transcript")

    transcript_path = os.path.join(job_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as handle:
        json.dump(transcript, handle, indent=2, ensure_ascii=False)
    logger.info(
        f"Transcript saved: {transcript['word_count']} words, "
        f"{transcript['segment_count']} segments"
    )

    if not config.KEEP_INTERMEDIATE_FILES:
        try:
            os.remove(audio_path)
            logger.debug("Removed intermediate audio file")
        except OSError:
            pass

    return transcript


def _transcribe_local(audio_path: str, device: str, logger: logging.Logger) -> dict:
    """Transcribe using faster-whisper with CTranslate2 backend."""
    from faster_whisper import WhisperModel

    requested_device = device
    compute_type = config.WHISPER_COMPUTE_TYPE if device == "cuda" else "float32"

    logger.info(
        f"Loading faster-whisper model: {config.WHISPER_MODEL} "
        f"on {device} ({compute_type})"
    )
    try:
        model = WhisperModel(
            config.WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        if requested_device == "cuda":
            raise RuntimeError(
                "Local faster-whisper could not initialize on the GPU. "
                "This local path is GPU-only, so switch to Auto/OpenRouter/Smallest "
                "if you want a non-GPU fallback."
            ) from exc
        raise

    logger.info("Model loaded, starting transcription...")

    transcribe_opts = {
        "beam_size": config.WHISPER_BEAM_SIZE,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if config.WHISPER_LANGUAGE:
        transcribe_opts["language"] = config.WHISPER_LANGUAGE

    segments_gen, info = model.transcribe(audio_path, **transcribe_opts)
    logger.info(
        f"Detected language: {info.language} "
        f"(prob={info.language_probability:.2f})"
    )

    segments = []
    word_count = 0
    for seg in segments_gen:
        seg_data = {
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": [],
        }
        for word in seg.words or []:
            seg_data["words"].append(
                {
                    "word": word.word.strip(),
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                }
            )
            word_count += 1
        segments.append(seg_data)

    logger.info(f"Transcription complete: {len(segments)} segments, {word_count} words")

    del model
    gc.collect()
    if requested_device == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    logger.debug("Model unloaded, GPU memory freed")

    return {
        "segments": segments,
        "language": info.language,
        "duration": info.duration,
    }


def find_best_split_time(audio_path: str, target_time: float, search_window: float = 10.0) -> float:
    """Finds the quietest millisecond (lowest RMS energy) in a search window around target_time."""
    try:
        import soundfile as sf
        import numpy as np
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            total_samples = len(f)
            start_sec = max(0.0, target_time - search_window / 2.0)
            end_sec = min(total_samples / sr, target_time + search_window / 2.0)
            
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            
            if start_idx >= total_samples:
                return total_samples / sr
                
            f.seek(start_idx)
            frames_to_read = end_idx - start_idx
            if frames_to_read <= 0:
                return target_time
                
            data = f.read(frames_to_read)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                
            window_size = int(0.1 * sr)  # 100ms
            hop_size = int(0.01 * sr)    # 10ms hop
            
            best_energy = float('inf')
            best_idx = 0
            
            for i in range(0, len(data) - window_size, hop_size):
                window = data[i : i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                if energy < best_energy:
                    best_energy = energy
                    best_idx = i
                    
            best_split_sec = start_sec + (best_idx + window_size / 2.0) / sr
            return round(best_split_sec, 3)
    except Exception:
        return target_time


def _transcribe_smallest_single_chunk(
    audio_path: str,
    api_key: str,
    media_duration: float,
    logger: logging.Logger,
) -> dict:
    """Transcribe a single audio chunk using Smallest.ai."""
    import requests
    import json

    logger.info(f"Sending audio chunk of {media_duration:.1f}s to Smallest.ai Pulse STT...")

    params = {
        "language": config.SMALLEST_LANGUAGE,
        "word_timestamps": "true"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "audio/wav",
    }

    with open(audio_path, "rb") as handle:
        audio_bytes = handle.read()

    response = requests.post(
        config.SMALLEST_API_URL,
        params=params,
        headers=headers,
        data=audio_bytes,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    text = (
        data.get("transcription", "")
        or data.get("text", "")
        or data.get("transcript", "")
        or data.get("data", {}).get("text", "")
        or data.get("result", "")
        or data.get("output", "")
    )

    if not text:
        logger.error(f"Smallest.ai response body: {json.dumps(data)[:500]}")
        raise RuntimeError(
            "Smallest.ai returned an empty transcript. "
            f"Response keys: {list(data.keys())}"
        )

    words_data = data.get("words", []) or data.get("data", {}).get("words", [])
    if words_data:
        segments = []
        for i in range(0, len(words_data), 20):
            chunk = words_data[i:i+20]
            start = float(chunk[0].get("start", 0))
            end = float(chunk[-1].get("end", start + 0.5))
            chunk_text = " ".join(str(w.get("word", "")) for w in chunk)
            
            clean_words = []
            for w in chunk:
                clean_words.append({
                    "word": str(w.get("word", "")),
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0))
                })

            segments.append({
                "id": len(segments),
                "start": start,
                "end": end,
                "text": chunk_text,
                "words": clean_words
            })
    else:
        logger.warning("Smallest.ai did not return 'words' array, falling back to pseudo-timestamps")
        segments = _text_to_pseudo_segments(text, media_duration)

    return {
        "segments": segments,
        "language": config.SMALLEST_LANGUAGE,
        "duration": media_duration,
    }


def _transcribe_smallest(
    audio_path: str,
    api_key: str,
    media_duration: float,
    logger: logging.Logger,
) -> dict:
    """Transcribe using Smallest.ai, applying sliding window chunking if audio is long."""
    duration = media_duration or _probe_audio_duration(audio_path, logger)
    chunk_size = 1200.0  # 20 minutes

    # If duration is 20 minutes or less, transcribe as a single chunk
    if duration <= chunk_size:
        return _transcribe_smallest_single_chunk(audio_path, api_key, duration, logger)

    # Determine chunk intervals
    splits = [0.0]
    cursor = chunk_size
    while cursor < duration - 60.0:  # If less than a minute remains, don't split
        best_split = find_best_split_time(audio_path, cursor, search_window=10.0)
        splits.append(best_split)
        cursor = best_split + chunk_size
    splits.append(duration)

    logger.info(f"Splitting audio into {len(splits) - 1} chunk(s) near 20-minute boundaries: {splits}")

    all_segments = []
    total_words_transcribed = 0
    current_seg_id = 0
    base_dir = os.path.dirname(audio_path)

    for idx in range(len(splits) - 1):
        start = splits[idx]
        end = splits[idx + 1]
        chunk_dur = end - start
        logger.info(f"Processing chunk {idx + 1}/{len(splits) - 1}: {start:.3f}s -> {end:.3f}s (dur={chunk_dur:.3f}s)")

        chunk_path = os.path.join(base_dir, f"audio_chunk_{idx}.wav")

        # Extract audio chunk using FFmpeg (fast stream copy)
        cmd = [
            config.FFMPEG_PATH,
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", audio_path,
            "-acodec", "copy",
            "-y",
            chunk_path,
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                check=True,
            )
        except Exception as exc:
            logger.error(f"FFmpeg chunk extraction failed: {exc}")
            raise

        # Transcribe chunk
        try:
            chunk_result = _transcribe_smallest_single_chunk(
                chunk_path, api_key, chunk_dur, logger
            )
        except Exception as exc:
            logger.error(f"Failed to transcribe chunk {idx + 1}: {exc}")
            raise
        finally:
            # Clean up chunk file
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

        # Shift timestamps and merge segments/words
        chunk_segs = chunk_result.get("segments", [])
        for seg in chunk_segs:
            seg["start"] = round(seg["start"] + start, 3)
            seg["end"] = round(seg["end"] + start, 3)
            seg["id"] = current_seg_id
            current_seg_id += 1

            for w in seg.get("words", []):
                w["start"] = round(w["start"] + start, 3)
                w["end"] = round(w["end"] + start, 3)
                w["segment_id"] = seg["id"]
                total_words_transcribed += 1

            all_segments.append(seg)

    logger.info(f"Chunked transcription merged: {len(all_segments)} segments, {total_words_transcribed} words")
    return {
        "segments": all_segments,
        "language": config.SMALLEST_LANGUAGE,
        "duration": duration,
    }


def _text_to_pseudo_segments(text: str, media_duration: float = 0) -> list:
    """Convert plain text to estimated timed segments when an API lacks timestamps."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    clean_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    total_words = sum(len(sentence.split()) for sentence in clean_sentences)
    if total_words <= 0:
        return []

    duration = float(media_duration or 0)
    if duration <= 0:
        duration = max(total_words * 0.35, len(clean_sentences) * 2.0)

    segments = []
    cursor = 0.0
    for idx, sentence in enumerate(clean_sentences):
        sentence_words = sentence.split()
        word_count = len(sentence_words)
        if word_count <= 0:
            continue

        remaining_words = sum(len(s.split()) for s in clean_sentences[idx:])
        remaining_duration = max(0.1, duration - cursor)
        sentence_duration = remaining_duration * (word_count / max(1, remaining_words))
        sentence_start = cursor
        sentence_end = min(duration, sentence_start + max(sentence_duration, word_count * 0.18))
        word_duration = max(0.12, (sentence_end - sentence_start) / word_count)

        words = [
            {
                "word": word,
                "start": round(sentence_start + (word_idx * word_duration), 3),
                "end": round(min(sentence_end, sentence_start + ((word_idx + 1) * word_duration)), 3),
            }
            for word_idx, word in enumerate(sentence_words)
        ]
        segments.append(
            {
                "id": idx,
                "start": round(sentence_start, 3),
                "end": round(sentence_end, 3),
                "text": sentence,
                "words": words,
            }
        )
        cursor = sentence_end
    return segments


def restore_punctuation(
    transcript: dict,
    logger: logging.Logger,
    *,
    sentence_gap_s: float = 0.6,
    clause_gap_s: float = 0.32,
    force_max_words: int = 28,
    min_density: float = 0.025,
) -> int:
    """Restore sentence/clause punctuation from inter-word pauses.

    Engine-independent and dependency-free: some STT engines (Smallest.ai) drop
    sentence terminators, leaving long unpunctuated runs that starve both the AI
    boundary picker and the matcher's sentence-snap. This walks the word stream
    in time order and appends a terminator after a word when the gap to the next
    word looks like a sentence boundary, a comma on a smaller clause pause.

    Safe by construction:
      • Skips entirely when the transcript's terminator density is already healthy
        (>= ``min_density`` terminators/word), so good transcripts are untouched.
      • Never rewrites a word that already ends on ``.``/``!``/``?`` — so the
        already-punctuated spans of a mixed transcript are preserved.
      • The matcher's ``clean_word()`` strips punctuation before phrase matching,
        so added terminators never break word-exact matching; only the snap and
        ``has_sentence_end`` read the raw ``word`` and will now find real ends.

    Returns the number of sentence terminators added.
    """
    segments = transcript.get("segments") or []
    words = []
    for seg in segments:
        for w in seg.get("words") or []:
            words.append(w)
    words.sort(key=lambda w: float(w.get("start", 0.0) or 0.0))
    n = len(words)
    if n < 2:
        return 0

    def _ends_term(raw: str) -> bool:
        return str(raw or "").rstrip()[-1:] in (".", "!", "?")

    term_now = sum(1 for w in words if _ends_term(w.get("word", "")))
    density = term_now / float(n)
    if density >= min_density:
        logger.info(
            f"Punctuation restoration skipped: terminator density {density:.3f}/word "
            f"already >= {min_density}"
        )
        return 0

    added = 0
    since = 0  # words since the last terminator
    for i in range(n):
        w = words[i]
        raw = str(w.get("word", ""))
        if _ends_term(raw):
            since = 0
            continue
        since += 1
        if i == n - 1:
            # Close the transcript on a terminator.
            w["word"] = raw.rstrip() + "."
            added += 1
            break
        nxt = words[i + 1]
        try:
            gap = float(nxt.get("start", 0.0)) - float(w.get("end", 0.0))
        except (TypeError, ValueError):
            gap = 0.0
        # After a long unpunctuated run, accept a smaller pause so we don't emit
        # a mega-sentence the snap can't break.
        effective_gap = sentence_gap_s if since < force_max_words else clause_gap_s
        if gap >= effective_gap:
            w["word"] = raw.rstrip() + "."
            added += 1
            since = 0
        elif gap >= clause_gap_s and raw.rstrip()[-1:] not in (".", "!", "?", ","):
            w["word"] = raw.rstrip() + ","

    # Rebuild segment text so the AI sees the restored punctuation.
    for seg in segments:
        toks = [str(w.get("word", "")).strip() for w in (seg.get("words") or [])]
        toks = [t for t in toks if t]
        if toks:
            seg["text"] = " ".join(toks)

    logger.info(
        f"Punctuation restoration: added {added} sentence terminator(s) "
        f"(was {term_now} in {n} words, density {density:.3f}/word)"
    )
    return added


def _build_transcript(result: dict, meta: dict, method: str, logger: logging.Logger) -> dict:
    """Build enriched transcript from raw transcription result."""
    segments = result.get("segments", [])
    all_words = []

    for seg in segments:
        seg.setdefault("speaker", None)
        for word in seg.get("words", []):
            word["segment_id"] = seg.get("id", 0)
            word["speaker"] = None
            all_words.append(word)

    logger.info("Diarization skipped (pyannote removed; using CV-only speaker tracking)")

    return {
        "language": result.get("language", "unknown"),
        "duration": meta.get("duration", 0) or result.get("duration", 0),
        "segments": segments,
        "words": all_words,
        "word_count": len(all_words),
        "segment_count": len(segments),
        "has_speakers": False,
        "transcription_method": method,
    }


def _probe_audio_duration(audio_path: str, logger: logging.Logger) -> float:
    """Return audio duration when API transcript timing has to be estimated."""
    cmd = [
        config.FFPROBE_PATH,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        logger.warning(f"Could not probe audio duration: {exc}")
        return 0.0


def _extract_audio(video_path: str, audio_path: str, logger: logging.Logger):
    """Extract audio from video as WAV using FFmpeg."""
    cmd = [
        config.FFMPEG_PATH,
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path,
    ]

    if config.LOG_FFMPEG_COMMANDS:
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    if result.returncode != 0:
        logger.error(f"FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"Audio extraction failed: {result.stderr[:500]}")

    logger.debug(f"Audio extracted to {audio_path}")
