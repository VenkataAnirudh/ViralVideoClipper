"""
Pipeline Runner (Orchestrator) v2
==================================
Runs the complete pipeline in sequence, reporting progress at each step.
Includes blog post generation and resume detection for partially completed jobs.
"""

import json
import os
import re
import shutil
import time
import threading
import traceback
import logging
from pathlib import Path
import config
from pipeline.logger import create_debug_file_handler, create_job_logger
from test_download import download_video
from pipeline.transcriber import transcribe_video
from pipeline.analyzer import analyze_transcript, analyze_transcript_external
from pipeline.clip_selector import select_clips
from pipeline.local_clips_generator import back_annotate_candidate_txt
from pipeline.extractor import extract_clips
from pipeline.captioner import burn_captions, _get_clip_local_words, _load_render_offsets, pre_process_music_async, rename_music_job_dir
from pipeline.copywriter import generate_copy
from pipeline.blogger import generate_blog_post
from pipeline.stage_state import inputs_changed, record_stage, validate_artifacts


# Pipeline stages with weights for progress bar
# Weights adjusted for faster-whisper (transcription is now much faster)
STAGES = [
    ("downloading",   "Downloading video",       6),
    ("transcribing",  "Transcribing audio",     10),
    ("analyzing",     "AI analysis",            12),
    ("selecting",     "Selecting clips",         4),
    ("extracting",    "Face-aware extraction",  34),
    ("captioning",    "Burning captions",       22),
    ("copywriting",   "Generating copy",         4),
    ("blogging",      "Writing blog post",       2),
    ("packaging",     "Packaging results",       6),
]


def _redact_settings(settings):
    redacted = dict(settings or {})
    for key in list(redacted):
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "***" if redacted.get(key) else ""
    return redacted


def _stage_start(logger, stage):
    logger.info(f"[STAGE] {stage}: start")
    return time.time()


def _stage_done(logger, stage, started_at, job_dir=None, artifacts=None, extra=""):
    elapsed = time.time() - started_at
    suffix = f" | {extra}" if extra else ""
    logger.info(f"[STAGE] {stage}: done in {elapsed:.1f}s{suffix}")
    if job_dir and artifacts:
        for rel in artifacts:
            path = os.path.join(job_dir, rel)
            if os.path.exists(path):
                try:
                    size = os.path.getsize(path)
                    logger.info(f"[ARTIFACT] {rel}: {size} bytes")
                except OSError:
                    logger.info(f"[ARTIFACT] {rel}: present")
            else:
                logger.info(f"[ARTIFACT] {rel}: missing")


def _load_json_artifact(job_dir, filename, logger, label, required=True):
    ok, reason, _ = validate_artifacts(job_dir, [filename], json_files=[filename])
    if not ok:
        logger.warning(f"[RESUME] {label}: invalid ({reason})")
        if required:
            raise RuntimeError(f"{filename} invalid or missing: {reason}")
        return None
    path = os.path.join(job_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = len(data) if isinstance(data, (list, dict)) else "?"
    logger.info(f"[RESUME] {label}: loaded {filename} ({count} item(s))")
    return data


def _back_annotate_candidate_files(job_dir, clips_plan, logger):
    """After clips_plan.json exists, write resolved start/end into
    eligible_candidates.txt and refined_candidates.txt so those files become a
    truthful audit trail instead of carrying pre-resolution hallucinations.
    Pairing is by candidate_id (the AI candidate's stable id), never by index.
    """
    cid_to_span = {}
    for clip in clips_plan or []:
        cid = clip.get("candidate_id") or clip.get("source_candidate_id")
        if not cid:
            continue
        segs = clip.get("segments") or []
        if not segs:
            continue
        try:
            start = float(segs[0]["start"])
            end = float(segs[-1]["end"])
        except (KeyError, TypeError, ValueError):
            continue
        cid_to_span[str(cid)] = (start, end)

    if not cid_to_span:
        logger.info("[BACK-ANNOTATE] No candidate_id -> span pairs to write; skipped.")
        return

    for name in ("refined_candidates.txt", "eligible_candidates.txt"):
        back_annotate_candidate_txt(Path(job_dir) / name, cid_to_span, logger)


def _refinement_metadata_present(clips_plan) -> bool:
    """True when boundary refinement already populated per-clip metadata for at
    least half the clips (so the separate copywriter stage can be skipped)."""
    clips = clips_plan or []
    if not clips:
        return False
    have = sum(
        1 for c in clips
        if str(c.get("description_text") or "").strip()
        and (c.get("hashtags") or c.get("description_hashtags"))
    )
    return have >= max(1, len(clips) // 2)


_ANALYSIS_SETTINGS_KEYS = [
    "clip_count", "get_all_clips", "min_duration", "max_duration",
    "ai_provider", "ai_model", "analysis_window_seconds", "analysis_overlap_seconds",
    "skip_refinement", "skip_judge", "skip_compression", "skip_duplication",
    "refinement_batch_size", "compression_batch_size", "enable_reasoning",
]


def _clear_analysis_context(job_dir, logger, settings=None):
    """Remove all analysis artifacts EXCEPT api_cache/ and source files.

    On every analysis restart, this wipes all generated artifacts (analysis.json,
    clips_plan.json, eligible/refined candidates, fallback reports, bad-response
    logs, etc.) so the pipeline rebuilds cleanly from cached API responses.

    PRESERVED (never deleted):
      • api_cache/          — cached per-window discovery and per-batch refinement
                              API responses.  Keeping them means the pipeline
                              resumes from whatever windows are already cached;
                              only uncached windows/batches trigger new API calls.
      • meta.json           — video metadata (needed by every stage)
      • transcript.json     — transcription output
      • video / audio files — source media
      • clips/              — extracted clip videos (downstream stage)
      • analysis_settings.json — fingerprint for the current settings

    Everything else is cheap to regenerate from the cached API responses,
    so deleting it guarantees a fresh state without wasting API quota.
    """
    from pipeline.stage_state import settings_fingerprint

    # Entries to always keep (case-sensitive basenames or directory names).
    PRESERVE = {
        "api_cache",
        "meta.json",
        "transcript.json",
        "clips",
        "analysis_settings.json",
    }
    # Keep video / audio files by extension.
    MEDIA_EXTENSIONS = {
        ".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".ts",
        ".wav", ".mp3", ".aac", ".m4a", ".ogg", ".flac", ".opus",
        ".jpg", ".jpeg", ".png", ".webp",
    }

    removed = 0
    for entry in os.listdir(job_dir):
        if entry in PRESERVE:
            continue
        entry_path = os.path.join(job_dir, entry)
        # Preserve media files (video, audio, thumbnail) regardless of name.
        if os.path.isfile(entry_path):
            _, ext = os.path.splitext(entry.lower())
            if ext in MEDIA_EXTENSIONS:
                continue
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)
            else:
                os.remove(entry_path)
            removed += 1
        except OSError as exc:
            logger.warning(f"Could not remove {entry}: {exc}")

    # Persist the current settings fingerprint so future runs can detect changes.
    fp_path = os.path.join(job_dir, "analysis_settings.json")
    try:
        current_fp = settings_fingerprint(settings or {}, _ANALYSIS_SETTINGS_KEYS)
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": current_fp}, f)
    except OSError:
        pass

    logger.info(
        f"Cleared analysis context: {removed} file(s)/dir(s) removed.  "
        f"api_cache/ preserved for resume."
    )



class PipelineCancelled(Exception):
    """Raised at a stage boundary when the user cancels the job."""


def run_pipeline(job_id: str, url: str, settings: dict, progress_callback, rename_callback=None, cancel_check=None):
    """
    Execute the full video processing pipeline.

    Args:
        job_id: Unique job identifier
        url: YouTube video URL
        settings: User settings dict
        progress_callback: Function(stage, percent, message) to report progress
    """
    job_dir = os.path.join(config.OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, "clips"), exist_ok=True)

    logger = create_job_logger(job_id, job_dir)

    def _ck(stage_label=""):
        """Abort at a stage boundary if the job was cancelled."""
        if cancel_check and cancel_check():
            logger.info(f"[CANCEL] Job cancelled by user (at {stage_label or 'stage boundary'})")
            raise PipelineCancelled("Cancelled by user")

    logger.info("=" * 60)
    logger.info(f"PIPELINE START — Job: {job_id}")
    logger.info(f"URL: {url}")
    logger.info(f"Settings: {json.dumps(_redact_settings(settings), indent=2)}")
    logger.info("=" * 60)

    # Start background music pre-processing early in pipeline
    try:
        pre_process_music_async(job_dir, settings, logger)
    except Exception as e:
        logger.warning(f"Failed to start async music pre-processing: {e}")

    start_time = time.time()
    cumulative_weight = 0
    total_weight = sum(s[2] for s in STAGES)

    current_stage = "downloading"
    current_pct = 0
    current_detail = ""
    bg_download_pct = 0
    bg_phase = "starting"

    def update_progress(stage_idx, detail=""):
        nonlocal current_stage, current_pct, current_detail
        stage_key, stage_name, weight = STAGES[stage_idx]
        current_stage = stage_key
        current_detail = detail
        nonlocal cumulative_weight
        current_pct = int((cumulative_weight / total_weight) * 100)
        
        # Build dual status message if bg download thread is active
        msg = f"{stage_name}... {detail}"
        if video_thread and video_thread.is_alive():
            msg += f" (Bg Video {bg_phase.capitalize()}: {bg_download_pct}%)"
            
        logger.debug(f"Progress {stage_key} {current_pct}%: {detail}")
        progress_callback(stage_key, current_pct, msg)

    def bg_progress_callback(pct, phase):
        nonlocal bg_download_pct, bg_phase
        bg_download_pct = pct
        bg_phase = phase
        
        # Find active stage name
        stage_name = "Processing"
        for stage_key, s_name, _ in STAGES:
            if stage_key == current_stage:
                stage_name = s_name
                break
                
        msg = f"{stage_name}... {current_detail} (Bg Video {phase.capitalize()}: {pct}%)"
        progress_callback(current_stage, current_pct, msg)

    video_thread = None
    video_download_error = None

    try:
        # ── Stage 1: Download ─────────────────────────────────────
        update_progress(0, "Starting download")
        stage_t = _stage_start(logger, "download")

        # Resume detection: skip download if video already exists
        has_video = any(f.endswith(".mp4") for f in os.listdir(job_dir)) if os.path.exists(job_dir) else False
        if has_video and os.path.exists(os.path.join(job_dir, "meta.json")):
            logger.info("Video file already exists, skipping download")
            # Load existing metadata if available
            meta_path = os.path.join(job_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {"title": "Resumed Video", "channel": "", "duration": 0}
        else:
            # 1. Download metadata & audio stream first (synchronous, extremely fast)
            from test_download import download_audio_only
            def audio_progress(pct, msg_detail):
                progress_callback("downloading", 0, f"Downloading video... audio {pct}%")
            meta = download_audio_only(job_dir, url, logger, progress_callback=audio_progress)

        if rename_callback and meta.get("title"):
            import re
            safe_title = re.sub(r'[<>:"/\\|?*]', '', meta.get("title", ""))[:50].strip()
            if safe_title:
                highest_n = 0
                if os.path.exists(config.OUTPUT_DIR):
                    for folder in os.listdir(config.OUTPUT_DIR):
                        if os.path.isdir(os.path.join(config.OUTPUT_DIR, folder)):
                            match = re.match(r'^Video (\d+) - ', folder)
                            if match:
                                n = int(match.group(1))
                                if n > highest_n:
                                    highest_n = n
                next_n = highest_n + 1
                new_job_dir = os.path.join(config.OUTPUT_DIR, f"Video {next_n} - {safe_title}")
                
                if new_job_dir != job_dir:
                    try:
                        handlers = logger.handlers[:]
                        for h in handlers:
                            if isinstance(h, logging.FileHandler):
                                h.close()
                                logger.removeHandler(h)
                                
                        old_job_dir = job_dir
                        os.rename(job_dir, new_job_dir)
                        job_dir = new_job_dir
                        try:
                            rename_music_job_dir(old_job_dir, job_dir)
                        except Exception as e:
                            logger.warning(f"Failed to rename music job dir tracker: {e}")
                        rename_callback(job_dir)
                        
                        log_file = os.path.join(job_dir, "debug.log")
                        formatter = logging.Formatter(
                            fmt="%(asctime)s │ %(levelname)-8s │ %(module)-15s │ %(message)s",
                            datefmt="%H:%M:%S"
                        )
                        file_handler = create_debug_file_handler(log_file, formatter)
                        logger.addHandler(file_handler)
                        
                        with open(os.path.join(job_dir, ".job_id"), "w", encoding="utf-8") as f:
                            f.write(job_id)
                            
                        logger.info(f"Successfully renamed job directory to {os.path.basename(job_dir)}")
                    except Exception as e:
                        try:
                            log_file = os.path.join(job_dir, "debug.log")
                            formatter = logging.Formatter(
                                fmt="%(asctime)s │ %(levelname)-8s │ %(module)-15s │ %(message)s",
                                datefmt="%H:%M:%S"
                            )
                            file_handler = create_debug_file_handler(log_file, formatter)
                            logger.addHandler(file_handler)
                        except Exception:
                            pass
                        logger.warning(f"Failed to rename job directory: {e}")

        _stage_done(logger, "download", stage_t, job_dir, ["meta.json"], extra=meta.get("title", ""))
        record_stage(
            job_dir,
            "download",
            "done",
            outputs=["meta.json"],
            counts={"has_video": bool(has_video), "duration": meta.get("duration", 0)},
            settings=settings,
        )
        cumulative_weight += STAGES[0][2]

        # Now, if we downloaded audio, start the background video download thread
        # using the final job_dir (after potential renaming).
        if not has_video:
            from test_download import download_video_stream_only
            def bg_video_download_task():
                nonlocal video_download_error
                try:
                    logger.info("[BG-DOWNLOAD] Background video download & transcode started...")
                    download_video_stream_only(job_dir, url, meta, logger, progress_callback=bg_progress_callback)
                    logger.info("[BG-DOWNLOAD] Background video download & transcode completed successfully.")
                except Exception as e:
                    video_download_error = e
                    logger.error(f"[BG-DOWNLOAD] Background video download failed: {e}")

            video_thread = threading.Thread(
                target=bg_video_download_task,
                name="BG-VideoDownload",
                daemon=True
            )
            video_thread.start()
            logger.info("[BG-DOWNLOAD] Thread started. Transcription and AI analysis will proceed concurrently.")

        # BUG 8 FIX: Explicitly fire a progress update at 8% immediately after
        # download completes. Without this explicit callback the UI stays at 0%
        # until transcription starts, so if transcription crashes the user sees
        # no confirmation that the download actually worked.
        progress_callback("downloading", int((cumulative_weight / total_weight) * 100),
                          f"Download complete: {meta.get('title', 'video')[:60]}")

        # ── Stage 2: Transcribe ───────────────────────────────────
        _ck("before transcribe")
        method = settings.get("transcription_method", config.TRANSCRIPTION_METHOD)
        update_progress(1, f"Loading model ({method})")
        stage_t = _stage_start(logger, "transcribe")
        transcript = transcribe_video(job_dir, meta, settings, logger)
        _stage_done(logger, "transcribe", stage_t, job_dir, ["transcript.json"], extra=f"words={transcript.get('word_count', '?')}")
        record_stage(
            job_dir,
            "transcribe",
            "done",
            inputs=["meta.json"],
            outputs=["transcript.json"],
            counts={"words": transcript.get("word_count", 0)},
            settings=settings,
            settings_keys=["transcription_method", "whisper_model", "whisper_device"],
        )
        cumulative_weight += STAGES[1][2]
        update_progress(1, f"{transcript['word_count']} words ({transcript.get('transcription_method', 'local')})")

        # ── Stage 3: AI Analysis (API route OR manual external-LLM route) ──
        _ck("before analysis")
        analysis_mode = str(settings.get("analysis_mode",
                            getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"))).strip().lower()
        stage_t = _stage_start(logger, "analyze")
        if analysis_mode == "manual":
            from pipeline import external_llm
            if not external_llm.has_external_llm_input(job_dir):
                # First pass: emit the paste-ready windows + reference the system
                # prompt, then PAUSE so the user can run them through their own LLM.
                external_llm.prepare_external_llm_inputs(job_dir, meta, transcript, settings, logger)
                logger.info("Manual analysis route: paused — awaiting external-LLM output.")
                progress_callback(
                    "paused_external_llm", cumulative_weight,
                    "PAUSED: run the transcript windows through your LLM, paste the output, then resume.",
                )
                return {}
            logger.info("Manual analysis route: ingesting pasted external-LLM output.")
            update_progress(2, "Ingesting manual analysis")
            analysis = analyze_transcript_external(job_dir, meta, transcript, settings, logger)
            # Manual route: your own LLM already chose the boundaries (verbatim
            # words → local word-mapping in select_clips). Validation stays
            # suppressed. Stitch pass: the UI "skip AI stitching" toggle wins
            # when ON; otherwise auto — allowed only when an OpenAI key exists
            # (clip_selector routes it to the standalone 2-pass external
            # stitcher, never the API route's). Assigned both ways so a stale
            # flag from a prior run can't leak in.
            _user_skip_stitch = str(settings.get("skip_stitch_pass", "")).strip().lower() in {"1", "true", "yes", "on"}
            _openai_key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
            settings["skip_stitch_pass"] = _user_skip_stitch or not bool(_openai_key)
            settings["skip_validation"] = True
        else:
            update_progress(2, f"Sending to {settings.get('ai_provider', 'AI')}")
            _clear_analysis_context(job_dir, logger, settings)
            analysis = analyze_transcript(job_dir, meta, transcript, settings, logger)
        _stage_done(
            logger,
            "analyze",
            stage_t,
            job_dir,
            ["analysis.json", "analysis_transcript.txt", "eligible_candidates.txt", "refined_candidates.txt"],
            extra=f"candidates={len(analysis.get('candidates', []))}",
        )
        record_stage(
            job_dir,
            "analyze",
            "done",
            inputs=["transcript.json"],
            outputs=["analysis.json", "analysis_transcript.txt", "eligible_candidates.txt", "refined_candidates.txt"],
            counts={"candidates": len(analysis.get("candidates", []))},
            settings=settings,
            settings_keys=["ai_provider", "ai_model", "clip_count", "min_duration", "max_duration"],
        )
        cumulative_weight += STAGES[2][2]
        n_candidates = len(analysis.get("candidates", []))
        update_progress(2, f"Found {n_candidates} candidates")

        # ── Stage 4: Clip Selection ───────────────────────────────
        update_progress(3, "Snapping boundaries")
        stage_t = _stage_start(logger, "select")
        try:
            clips_plan = select_clips(job_dir, transcript, analysis, settings, logger)
        except Exception as select_err:
            logger.warning(f"AI clip selection failed: {select_err}. Triggering local fallback alignment...")
            from pipeline.local_clips_generator import run_local_alignment
            success, msg = run_local_alignment(job_dir, logger=logger)
            if success:
                logger.info("Local fallback alignment successfully created clips_plan.json.")
                with open(os.path.join(job_dir, "clips_plan.json"), "r", encoding="utf-8") as f:
                    clips_plan = json.load(f)
            else:
                raise RuntimeError(f"AI clip selection failed, and local fallback alignment failed: {msg}") from select_err

        _stage_done(logger, "select", stage_t, job_dir, ["clips_plan.json", "clip_info.txt"], extra=f"clips={len(clips_plan)}")
        record_stage(
            job_dir,
            "select",
            "done",
            inputs=["analysis.json", "transcript.json"],
            outputs=["clips_plan.json", "clip_info.txt"],
            counts={"clips": len(clips_plan)},
            settings=settings,
            settings_keys=["clip_count", "min_duration", "max_duration"],
        )
        # local_clips_generator inside select_clips already pinned word-exact
        # boundaries via candidate_id. Now back-annotate the resolved spans
        # into the .txt audit files so they stop carrying pre-resolution
        # hallucinations. No more legacy clip_alignment / index pairing.
        try:
            _back_annotate_candidate_files(job_dir, clips_plan, logger)
        except Exception as bx:
            logger.warning(f"Back-annotation of candidate .txt files failed (non-fatal): {bx}")
        cumulative_weight += STAGES[3][2]
        update_progress(3, f"{len(clips_plan)} clips selected")

        # Wait for background video download/transcode if running
        if video_thread is not None:
            if video_thread.is_alive():
                logger.info("[runner] AI analysis and clip selection complete, but background video download is still running. Waiting for it to finish...")
                progress_callback("selecting", 50, "Waiting for video download/transcode to complete...")
                video_thread.join()
            if video_download_error is not None:
                # Old behavior: raise RuntimeError — committed pipeline-suicide
                # right before extraction, destroying every clip job and
                # corrupting open caption writes. New behavior: log + attempt
                # one synchronous retry on a clean temp dir, then raise only if
                # the retry also fails. Either way, the AI analysis on disk is
                # preserved so the user can resume from 'extract' later.
                logger.error(f"[runner] Background video download/transcode failed: {video_download_error}")
                logger.warning("[runner] Attempting synchronous recovery download before extraction...")
                progress_callback("selecting", 55, "Retrying video download synchronously...")
                try:
                    from test_download import download_video_stream_only
                    download_video_stream_only(job_dir, url, meta, logger)
                    logger.info("[runner] Synchronous recovery download succeeded — proceeding to extraction.")
                except Exception as retry_err:
                    logger.error(f"[runner] Recovery download also failed: {retry_err}")
                    raise RuntimeError(
                        f"Source video unavailable for extraction. "
                        f"Original bg error: {video_download_error}. "
                        f"Retry error: {retry_err}. "
                        f"AI analysis is preserved in this job folder; "
                        f"resume from 'extract' once the video is available."
                    )
            else:
                logger.info("[runner] Background video download and transcode complete. Proceeding to extraction.")

        # ── Stage 5: Extraction ───────────────────────────────────
        _ck("before extraction")
        update_progress(4, "Face-aware crop + NVENC encode")
        stage_t = _stage_start(logger, "extract")
        extracted = extract_clips(job_dir, clips_plan, settings, logger)
        _stage_done(logger, "extract", stage_t, job_dir, ["clips"], extra=f"raw_clips={len(extracted)}")
        record_stage(
            job_dir,
            "extract",
            "done",
            inputs=["clips_plan.json", "manual_templates.json", "templates/manual_templates.json", "files/manual_templates.json"],
            outputs=["clips"],
            counts={"raw_clips": len(extracted)},
            settings=settings,
            settings_keys=["v4_speaker_filter", "v4_clear_template_cache_each_run", "manual_templates_path"],
        )
        cumulative_weight += STAGES[4][2]
        update_progress(4, f"{len(extracted)} clips extracted")

        # ── Stage 5b: TTS Hook Overlay ────────────────────────────
        if settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED):
            from pipeline.tts_hook import apply_hooks_to_clips
            logger.info("Applying TTS Hook Overlays to extracted clips...")
            stage_t = _stage_start(logger, "tts_hook")
            apply_hooks_to_clips(job_dir, clips_plan, settings, logger)
            _stage_done(logger, "tts_hook", stage_t, job_dir, ["clips"])
            record_stage(job_dir, "tts_hook", "done", inputs=["clips_plan.json", "clips"], outputs=["clips"], settings=settings)
        else:
            logger.info("[STAGE] tts_hook: skipped (disabled)")

        # Pause before captioning if requested
        if settings.get("pause_before_captioning"):
            logger.info("Pausing pipeline before captioning for user review as requested.")
            progress_callback("review", cumulative_weight, "PAUSED: Review and edit subtitles in the dashboard before captioning.")
            return {}

        # ── Stage 6: Captioning ───────────────────────────────────
        _ck("before captioning")
        update_progress(5, "Burning captions")
        stage_t = _stage_start(logger, "caption")
        captioned = burn_captions(job_dir, clips_plan, transcript, settings, logger)
        _stage_done(logger, "caption", stage_t, job_dir, ["clips"], extra=f"captioned={len(captioned)}")
        record_stage(
            job_dir,
            "caption",
            "done",
            inputs=["clips_plan.json", "transcript.json", "clips"],
            outputs=["clips"],
            counts={"captioned": len(captioned)},
            settings=settings,
        )
        cumulative_weight += STAGES[5][2]
        update_progress(5, f"{len(captioned)} clips captioned")

        # ── Stage 7: Copywriting (fallback-only) ──────────────────
        # Boundary refinement now owns YouTube metadata (title, description_text,
        # 30 description_hashtags, tags, cliffhanger hook). Only run the separate
        # copywriter when refinement did NOT produce metadata for most clips.
        update_progress(6, "Writing social captions")
        if _refinement_metadata_present(clips_plan):
            logger.info("[STAGE] copywriting: skipped — refinement already produced per-clip metadata")
            try:
                from pipeline.copywriter import write_youtube_packages
                write_youtube_packages(job_dir, clips_plan, logger)
            except Exception as pkg_err:
                logger.warning(f"YouTube package export failed (non-fatal): {pkg_err}")
        else:
            stage_t = _stage_start(logger, "copywriting")
            copy_data = generate_copy(job_dir, clips_plan, meta, transcript, settings, logger)
            _stage_done(logger, "copywriting", stage_t, job_dir, ["social_copy.json"])
            record_stage(job_dir, "copywriting", "done", inputs=["clips_plan.json", "transcript.json"], outputs=["social_copy.json"], settings=settings)
        cumulative_weight += STAGES[6][2]
        update_progress(6, "Copy generated")

        # ── Stage 8: Blog Post ────────────────────────────────────
        update_progress(7, "Writing blog post")
        stage_t = _stage_start(logger, "blogging")
        blog_result = generate_blog_post(
            job_dir, meta, transcript, analysis, clips_plan, settings, logger
        )
        _stage_done(logger, "blogging", stage_t, job_dir, ["blog_post.md", "blog_post.txt"], extra=f"words={blog_result.get('word_count', 0) if blog_result else 0}")
        record_stage(job_dir, "blogging", "done", inputs=["clips_plan.json", "analysis.json"], outputs=["blog_post.md", "blog_post.txt"], counts={"words": blog_result.get("word_count", 0) if blog_result else 0}, settings=settings)
        cumulative_weight += STAGES[7][2]
        blog_words = blog_result.get("word_count", 0) if blog_result else 0
        update_progress(7, f"Blog post: {blog_words} words")

        # ── Stage 9: Packaging ────────────────────────────────────
        update_progress(8, "Building results")
        stage_t = _stage_start(logger, "packaging")
        results = _package_results(job_dir, clips_plan, meta, analysis, blog_result, logger, settings)
        _stage_done(logger, "packaging", stage_t, job_dir, ["results.json"])
        record_stage(job_dir, "packaging", "done", inputs=["clips_plan.json", "social_copy.json"], outputs=["results.json"], counts={"clips": len(results.get("clips", []))}, settings=settings)
        cumulative_weight += STAGES[8][2]

        elapsed = time.time() - start_time
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        progress_callback("done", 100, f"Complete! {len(clips_plan)} clips in {elapsed:.0f}s")

        return results

    except PipelineCancelled as ce:
        logger.info(f"PIPELINE CANCELLED: {ce}")
        progress_callback("error", -1, "Cancelled by user")
        return {}
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(traceback.format_exc())
        progress_callback("error", -1, str(e))
        raise


def _missing_raw_clips(job_dir: str, clips_plan: list) -> list:
    """Return the list of clip_name values whose _raw.mp4 is missing on disk.

    Used during late-stage resume (tts_hook/caption/copy) to auto-rewind to the
    extract stage when the user wiped clips/ or never extracted for this job.
    """
    clips_dir = os.path.join(job_dir, "clips")
    missing = []
    for clip in clips_plan or []:
        name = clip.get("clip_name", "")
        if not name:
            continue
        if not os.path.exists(os.path.join(clips_dir, f"{name}_raw.mp4")):
            missing.append(name)
    return missing


def run_single_stage(job_dir: str, stage: str, settings: dict, progress_callback):
    """Run exactly ONE pipeline stage on an existing job folder, then stop.

    Stages: download_video | download_audio | music_audio | transcribe |
            analyze | extract | tts_hook | caption

    Unlike run_pipeline_from, nothing downstream is executed — surgical
    re-runs (re-fetch a stream, refresh the transcript, rebuild the plan,
    re-extract raws, re-burn captions) without touching the rest of the job.
    """
    logger = create_job_logger("single", job_dir)
    logger.info("=" * 60)
    logger.info(f"SINGLE STAGE '{stage}' — {job_dir}")
    logger.info("=" * 60)

    meta = {}
    meta_path = os.path.join(job_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    def _done(msg):
        logger.info(f"[SINGLE] {stage}: {msg}")
        progress_callback("done", 100, msg)
        return {}

    def _dl_cb(label):
        return lambda p, m: progress_callback(label, max(5, min(99, int(p or 0))), str(m))

    if stage == "download_video":
        url = str(settings.get("video_url") or meta.get("url") or "").strip()
        if not url:
            raise RuntimeError("No video URL available: meta.json has no 'url'")
        from test_download import download_video
        progress_callback("downloading", 5, "Re-downloading best-quality video...")
        meta = download_video(job_dir, url, logger, progress_callback=_dl_cb("downloading"))
        return _done(f"Video re-downloaded: {meta.get('video_filename', '?')}")

    if stage == "download_audio":
        url = str(settings.get("video_url") or meta.get("url") or "").strip()
        if not url:
            raise RuntimeError("No video URL available: meta.json has no 'url'")
        from test_download import download_audio_only
        progress_callback("downloading", 5, "Downloading audio stream (audio.wav)...")
        download_audio_only(job_dir, url, logger, progress_callback=_dl_cb("downloading"))
        return _done("audio.wav rebuilt from the source's best audio stream")

    if stage == "music_audio":
        music_url = str(settings.get("music_url") or "").strip()
        if not music_url:
            raise RuntimeError("Music download needs a YouTube link — set Background Music Mode to 'YouTube Link' and paste one")
        from pipeline.downloader import download_audio_track
        progress_callback("downloading", 10, "Downloading background music...")
        path = download_audio_track(job_dir, music_url, logger)
        try:
            from pipeline.captioner import _analyze_music_volume_curve
            progress_callback("downloading", 70, "Profiling music loudness (ebur128)...")
            _analyze_music_volume_curve(path, logger)
        except Exception as exc:
            logger.warning(f"Loudness profiling failed: {exc}")
        return _done(f"Music ready in library: {os.path.basename(path)}")

    if stage == "transcribe":
        progress_callback("transcribing", 10, "Transcribing audio...")
        transcript = transcribe_video(job_dir, meta, settings, logger)
        record_stage(job_dir, "transcribe", "done", inputs=["meta.json"], outputs=["transcript.json"],
                     counts={"words": transcript.get("word_count", 0)}, settings=settings)
        # Refresh the analysis-transcript view + paste-ready external-LLM windows
        try:
            from pipeline import external_llm
            progress_callback("transcribing", 90, "Building analysis windows...")
            external_llm.prepare_external_llm_inputs(job_dir, meta, transcript, settings, logger)
        except Exception as exc:
            logger.warning(f"Analysis-window build failed: {exc}")
        return _done(f"{transcript.get('word_count', '?')} words transcribed; analysis_transcript.txt refreshed")

    if stage == "analyze":
        transcript = _load_json_artifact(job_dir, "transcript.json", logger, "transcript")
        analysis_mode = str(settings.get("analysis_mode",
                            getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"))).strip().lower()
        if analysis_mode == "manual":
            from pipeline import external_llm
            if not external_llm.has_external_llm_input(job_dir):
                external_llm.prepare_external_llm_inputs(job_dir, meta, transcript, settings, logger)
                progress_callback("paused_external_llm", 30,
                                  "PAUSED: run the windows through your LLM, paste the output, then run this stage again.")
                return {}
            progress_callback("analyzing", 30, "Ingesting manual analysis...")
            analysis = analyze_transcript_external(job_dir, meta, transcript, settings, logger)
            _user_skip_stitch = str(settings.get("skip_stitch_pass", "")).strip().lower() in {"1", "true", "yes", "on"}
            _openai_key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
            settings["skip_stitch_pass"] = _user_skip_stitch or not bool(_openai_key)
            settings["skip_validation"] = True
        else:
            progress_callback("analyzing", 30, "AI analysis...")
            analysis = analyze_transcript(job_dir, meta, transcript, settings, logger)
        progress_callback("selecting", 70, "Selecting clips...")
        clips_plan = select_clips(job_dir, transcript, analysis, settings, logger)
        return _done(f"clips_plan.json built: {len(clips_plan)} clip(s) — no extraction run")

    if stage == "extract":
        clips_plan = _load_json_artifact(job_dir, "clips_plan.json", logger, "clips plan")
        progress_callback("extracting", 10, "Extracting raw clips...")
        extracted = extract_clips(job_dir, clips_plan, settings, logger)
        return _done(f"{len(extracted)} raw clip(s) extracted into clips/")

    if stage == "tts_hook":
        clips_plan = _load_json_artifact(job_dir, "clips_plan.json", logger, "clips plan")
        from pipeline.tts_hook import apply_hooks_to_clips
        hook_settings = dict(settings)
        # Explicit single-stage request overrides the global toggle
        hook_settings["tts_hook_enabled"] = True
        hook_settings["tts_hook_mode"] = "add"
        progress_callback("extracting", 10, "Generating TTS hook overlays...")
        apply_hooks_to_clips(job_dir, clips_plan, hook_settings, logger)
        return _done("TTS hook intros generated (existing complete hooks skipped)")

    if stage == "caption":
        clips_plan = _load_json_artifact(job_dir, "clips_plan.json", logger, "clips plan")
        transcript = _load_json_artifact(job_dir, "transcript.json", logger, "transcript")
        progress_callback("captioning", 10, "Burning captions...")
        captioned = burn_captions(job_dir, clips_plan, transcript, settings, logger)
        return _done(f"{len(captioned)} clip(s) captioned")

    raise RuntimeError(f"Unknown single stage: {stage}")


def run_pipeline_from(job_dir: str, resume_from: str, settings: dict, progress_callback):
    """
    Resume the pipeline from a specific stage using existing artifacts.

    Args:
        job_dir: Path to an existing job output directory
        resume_from: Stage to resume from — one of:
            'transcribe' — re-run from transcription onward
            'analyze'    — re-run from AI analysis onward
            'extract'    — re-run from clip extraction onward
            'caption'    — re-run only the captioning stage
        settings: User settings dict
        progress_callback: Function(stage, percent, message)
    """
    logger = create_job_logger("resume", job_dir)
    logger.info("=" * 60)
    logger.info(f"PIPELINE RESUME from '{resume_from}' — {job_dir}")
    logger.info("=" * 60)

    # Start background music pre-processing early in resume pipeline
    try:
        pre_process_music_async(job_dir, settings, logger)
    except Exception as e:
        logger.warning(f"Failed to start async music pre-processing: {e}")

    start_time = time.time()

    # ── Load existing meta.json ──────────────────────────────────────────
    meta_path = os.path.join(job_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"meta.json not found in {job_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    logger.info(f"Loaded meta: {meta.get('title', 'Unknown')}")

    # ── Find the video file ──────────────────────────────────────────────
    video_file = None
    for fn in os.listdir(job_dir):
        if fn.endswith(".mp4") and not fn.startswith("_"):
            video_file = fn
            break
    if not video_file:
        raise RuntimeError(f"No .mp4 video file found in {job_dir}")
    logger.info(f"Video file: {video_file}")

    # Ensure video is H.264 (transcode AV1 if needed on resume)
    video_path = os.path.join(job_dir, video_file)
    try:
        from pipeline.transcode_helper import ensure_h264_source
        ensure_h264_source(video_path, logger)
    except Exception as e:
        logger.warning(f"Failed to check/transcode source video format on resume: {e}")

    os.makedirs(os.path.join(job_dir, "clips"), exist_ok=True)

    try:
        transcript = None
        analysis = None
        clips_plan = None

        # ── Stage: Transcribe ────────────────────────────────────────────
        if resume_from in ("transcribe",):
            progress_callback("transcribing", 10, "Transcribing audio...")
            stage_t = _stage_start(logger, "transcribe")
            transcript = transcribe_video(job_dir, meta, settings, logger)
            _stage_done(logger, "transcribe", stage_t, job_dir, ["transcript.json"], extra=f"words={transcript.get('word_count', '?')}")
            record_stage(job_dir, "transcribe", "done", inputs=["meta.json"], outputs=["transcript.json"], counts={"words": transcript.get("word_count", 0)}, settings=settings)
            progress_callback("transcribing", 20, f"{transcript['word_count']} words transcribed")
        else:
            # Load existing transcript
            transcript = _load_json_artifact(job_dir, "transcript.json", logger, "transcript")
            logger.info(f"Loaded existing transcript: {transcript.get('word_count', '?')} words")

        # ── Stage: Analyze (API route OR manual external-LLM route) ──────
        if resume_from in ("transcribe", "analyze"):
            analysis_mode = str(settings.get("analysis_mode",
                                getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"))).strip().lower()
            if analysis_mode == "manual":
                from pipeline import external_llm
                if not external_llm.has_external_llm_input(job_dir):
                    external_llm.prepare_external_llm_inputs(job_dir, meta, transcript, settings, logger)
                    logger.info("Manual analysis route (resume): paused — awaiting external-LLM output.")
                    progress_callback(
                        "paused_external_llm", 30,
                        "PAUSED: run the transcript windows through your LLM, paste the output, then resume.",
                    )
                    return {}
                progress_callback("analyzing", 30, "Ingesting manual analysis...")
                analysis = analyze_transcript_external(job_dir, meta, transcript, settings, logger)
                # Manual route (resume): validation stays suppressed. The UI
                # "skip AI stitching" toggle wins when ON; otherwise the stitch
                # pass is allowed when an OpenAI key exists (standalone 2-pass
                # external stitcher in clip_selector).
                _user_skip_stitch = str(settings.get("skip_stitch_pass", "")).strip().lower() in {"1", "true", "yes", "on"}
                _openai_key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
                settings["skip_stitch_pass"] = _user_skip_stitch or not bool(_openai_key)
                settings["skip_validation"] = True
            else:
                progress_callback("analyzing", 30, "AI analysis...")
                _clear_analysis_context(job_dir, logger, settings)
                analysis = analyze_transcript(job_dir, meta, transcript, settings, logger)
            progress_callback("analyzing", 45, f"{len(analysis.get('candidates', []))} candidates found")

            progress_callback("selecting", 48, "Selecting clips...")
            try:
                clips_plan = select_clips(job_dir, transcript, analysis, settings, logger)
            except Exception as select_err:
                logger.warning(f"AI clip selection failed during resume: {select_err}. Triggering local fallback alignment...")
                from pipeline.local_clips_generator import run_local_alignment
                success, msg = run_local_alignment(job_dir, logger=logger)
                if success:
                    logger.info("Local fallback alignment successfully created clips_plan.json.")
                    with open(os.path.join(job_dir, "clips_plan.json"), "r", encoding="utf-8") as f:
                        clips_plan = json.load(f)
                else:
                    raise RuntimeError(f"AI clip selection failed during resume, and local fallback alignment failed: {msg}") from select_err
            progress_callback("selecting", 50, f"{len(clips_plan)} clips selected")
        else:
            # Load existing analysis and clips_plan
            an_path = os.path.join(job_dir, "analysis.json")
            cp_path = os.path.join(job_dir, "clips_plan.json")
            if os.path.exists(an_path):
                with open(an_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
            else:
                analysis = {"summary": "", "chapters": [], "candidates": []}
            if not os.path.exists(cp_path):
                raise RuntimeError(f"clips_plan.json not found — cannot resume from '{resume_from}'")
            with open(cp_path, "r", encoding="utf-8") as f:
                clips_plan = json.load(f)
            logger.info(f"Loaded existing clips_plan: {len(clips_plan)} clips")

        # ── Stage: Extract ───────────────────────────────────────────────
        # Auto-rewind: if resuming from tts_hook/caption/copy but the raw clip
        # videos are gone (user wiped clips/, or extract never ran for this
        # job), run extract first so downstream stages have their inputs.
        force_extract = False
        if resume_from in ("tts_hook", "caption", "copy"):
            missing = _missing_raw_clips(job_dir, clips_plan)
            if missing:
                force_extract = True
                sample = ", ".join(missing[:5])
                more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                logger.warning(
                    f"[RESUME] {len(missing)}/{len(clips_plan)} raw clip(s) missing on disk "
                    f"({sample}{more}). Auto-rewinding to 'extract' before continuing."
                )
        if resume_from in ("transcribe", "analyze", "extract") or force_extract:
            progress_callback("extracting", 35, "Face-aware crop + NVENC encode...")
            extracted = extract_clips(job_dir, clips_plan, settings, logger)
            progress_callback("extracting", 68, f"{len(extracted)} clips extracted")

        # Pause before TTS Hook and captioning if requested
        if settings.get("pause_before_captioning") and resume_from not in ("tts_hook", "caption", "copy"):
            logger.info("Pausing resumed pipeline before TTS hook and captioning for user review as requested.")
            progress_callback("review", 69, "PAUSED: Review and edit subtitles/hooks in the dashboard before continuing.")
            return {}

        # ── Stage: TTS Hook Overlay (runs if enabled/requested) ───
        tts_mode = settings.get("tts_hook_mode")
        if not tts_mode:
            # Fallback for when tts_hook_mode is not explicitly passed (e.g. from
            # retry or legacy calls). "keep", NOT "remove": a resume that says
            # nothing about hooks must never delete existing hook files.
            if settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED):
                tts_mode = "add"
            else:
                tts_mode = "keep"
        elif tts_mode == "keep" and resume_from == "tts_hook":
            tts_mode = "add"

        # The UI TTS-hook toggle is authoritative: when OFF, hooks are never
        # generated or rewritten regardless of mode — only an explicit
        # "remove" may touch existing hook files.
        if tts_mode in ("add", "save_only") and not settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED):
            logger.info(f"[RESUME] tts_hook: skipped (toggle off; mode '{tts_mode}' ignored, existing hook files untouched)")
        elif tts_mode in ("add", "save_only"):
            from pipeline.tts_hook import apply_hooks_to_clips
            progress_callback("extracting", 69, "Generating TTS hook overlays...")
            hook_settings = dict(settings)
            hook_settings["tts_hook_enabled"] = True
            hook_settings["tts_hook_mode"] = tts_mode
            apply_hooks_to_clips(job_dir, clips_plan, hook_settings, logger)
            if tts_mode == "add":
                settings["tts_hook_enabled"] = True
            else:
                settings["tts_hook_enabled"] = False
            settings["tts_hook_mode"] = tts_mode
        elif tts_mode == "remove":
            from pipeline.tts_hook import apply_hooks_to_clips
            progress_callback("extracting", 69, "Removing TTS hook overlays...")
            hook_settings = dict(settings)
            hook_settings["tts_hook_enabled"] = False
            hook_settings["tts_hook_mode"] = tts_mode
            apply_hooks_to_clips(job_dir, clips_plan, hook_settings, logger)
            settings["tts_hook_enabled"] = False
            settings["tts_hook_mode"] = tts_mode
        else:
            logger.info("[RESUME] tts_hook: skipped (mode='keep')")


        # ── Stage: Caption (always runs on resume) ───────────────────────
        if resume_from != "copy":
            progress_callback("captioning", 70, "Burning captions...")
            captioned = burn_captions(job_dir, clips_plan, transcript, settings, logger)
            progress_callback("captioning", 90, f"{len(captioned)} clips captioned")

        # ── Stage: Copywriting (fallback-only) ────────────────────────────
        # The YouTube package files just FORMAT metadata already on the clips
        # (refinement / external-LLM) — no API call — so they must be written even
        # on a caption-only / reburn resume (skip_copywriting). Otherwise the
        # clips folder is left with the old 4-block fallback format instead of the
        # ready-to-paste TITLE/DESCRIPTION/HASHTAGS/TAGS package.
        if _refinement_metadata_present(clips_plan):
            if settings.get("skip_copywriting"):
                logger.info("skip_copywriting set — still writing YouTube package files from existing metadata")
            else:
                logger.info("Skipping social copy — refinement metadata already present on clips")
            try:
                from pipeline.copywriter import write_youtube_packages
                write_youtube_packages(job_dir, clips_plan, logger)
            except Exception as pkg_err:
                logger.warning(f"YouTube package export failed (non-fatal): {pkg_err}")
        elif settings.get("skip_copywriting"):
            logger.info("Skipping social copy generation for caption-only resume")
        else:
            progress_callback("copywriting", 93, "Generating social copy...")
            copy_data = generate_copy(job_dir, clips_plan, meta, transcript, settings, logger)
            progress_callback("copywriting", 96, "Copy generated")

        # ── Package ──────────────────────────────────────────────────────
        blog_result = {}
        if settings.get("blog_post_enabled", config.BLOG_POST_ENABLED) and resume_from != "caption":
            progress_callback("blogging", 96, "Writing blog post...")
            blog_result = generate_blog_post(
                job_dir, meta, transcript, analysis, clips_plan, settings, logger
            )
            blog_words = blog_result.get("word_count", 0) if blog_result else 0
            progress_callback("blogging", 97, f"Blog post: {blog_words} words")

        progress_callback("packaging", 98, "Packaging results...")
        results = _package_results(job_dir, clips_plan, meta, analysis, blog_result, logger, settings)

        elapsed = time.time() - start_time
        logger.info(f"PIPELINE RESUME COMPLETE in {elapsed:.1f}s")
        progress_callback("done", 100, f"Resume complete! {len(clips_plan)} clips in {elapsed:.0f}s")
        return results

    except Exception as e:
        logger.error(f"PIPELINE RESUME FAILED: {e}")
        logger.error(traceback.format_exc())
        progress_callback("error", -1, str(e))
        raise


def _package_results(job_dir, clips_plan, meta, analysis, blog_result, logger, settings=None):
    """Package final results into results.json."""
    clips_dir = os.path.join(job_dir, "clips")
    existing_results_path = os.path.join(job_dir, "results.json")
    existing_by_name = {}
    old_results = {}
    if os.path.exists(existing_results_path):
        try:
            with open(existing_results_path, "r", encoding="utf-8") as f:
                old_results = json.load(f)
            existing_by_name = {
                c.get("clip_name"): c
                for c in old_results.get("clips", [])
                if c.get("clip_name")
            }
        except Exception:
            existing_by_name = {}

    social_copy = {}
    social_copy_path = os.path.join(job_dir, "social_copy.json")
    if os.path.exists(social_copy_path):
        try:
            with open(social_copy_path, "r", encoding="utf-8") as f:
                social_copy = json.load(f)
        except Exception:
            pass

    transcript = {}
    transcript_path = os.path.join(job_dir, "transcript.json")
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception:
            pass

    summary_val = analysis.get("summary", "")
    if not summary_val and old_results.get("summary"):
        summary_val = old_results.get("summary")

    chapters_val = analysis.get("chapters", [])
    if not chapters_val and old_results.get("chapters"):
        chapters_val = old_results.get("chapters")

    results = {
        "video_title": meta.get("title", ""),
        "video_channel": meta.get("channel", ""),
        "video_duration": meta.get("duration", 0),
        "summary": summary_val,
        "chapters": chapters_val,
        "clips": [],
        "blog_post": {},
    }

    # Add blog post info
    if blog_result:
        results["blog_post"] = {
            "md_file": os.path.basename(blog_result.get("md_path", "")),
            "txt_file": os.path.basename(blog_result.get("txt_path", "")),
            "word_count": blog_result.get("word_count", 0),
        }
    elif old_results.get("blog_post"):
        results["blog_post"] = old_results.get("blog_post")

    for clip in clips_plan:
        name = clip["clip_name"]
        raw_path = os.path.join(clips_dir, f"{name}_raw.mp4")
        cap_path = os.path.join(clips_dir, f"{name}_captioned.mp4")
        ass_path = os.path.join(clips_dir, f"{name}.ass")
        txt_path = os.path.join(clips_dir, f"{name}_caption.txt")
        old_clip = existing_by_name.get(name, {})
        caption_variants = _caption_variants(clips_dir, name)
        primary_caption = _primary_caption_file(
            clips_dir, name, caption_variants, settings, cap_path
        )
        srt_file = _primary_srt_file(clips_dir, name, settings)

        words = clip.get("edited_words")
        if words is None:
            words = _get_clip_local_words(clip, transcript, _load_render_offsets(clips_dir, name))

        # Helper to fall back to old values when copywriting is skipped / values are empty/drafts
        def get_fallback(field, default_val):
            val = clip.get(field)
            old_val = old_clip.get(field)

            # Map field to social_copy keys
            sc_field = field
            if field == "social_title":
                sc_field = "title"
            elif field == "social_description":
                sc_field = "description"
            elif field == "social_caption":
                sc_field = "caption"
            sc_val = social_copy.get(name, {}).get(sc_field)

            # If skip_copywriting is True, strongly prefer old/social_copy values if they exist
            if settings and settings.get("skip_copywriting"):
                if old_val not in (None, "", []):
                    return old_val
                if sc_val not in (None, "", []):
                    return sc_val
            # If the new value is None, empty string, or empty list, fall back
            if val is None or val == "" or (isinstance(val, list) and not val):
                if old_val not in (None, "", []):
                    return old_val
                if sc_val not in (None, "", []):
                    return sc_val
            # Special check for hashtags: if the new list looks like draft tags or is very small, fall back
            if field == "hashtags" and isinstance(val, list) and len(val) > 0:
                if any(x.lower() in ("#hashtag", "#tag", "#draft", "#temp") for x in val) or len(val) < 5:
                    if old_val not in (None, "", []):
                        return old_val
                    if sc_val not in (None, "", []):
                        return sc_val
            return val if val is not None else default_val

        resolved_title = get_fallback("title", "")
        resolved_reason = get_fallback("reason", "")
        resolved_social_title = get_fallback("social_title", "")
        resolved_social_desc = get_fallback("social_description", "")
        resolved_social_caption = get_fallback("social_caption", "")
        resolved_hashtags = get_fallback("hashtags", [])
        resolved_hook_title = get_fallback("hook_title", "")

        # Write _caption.txt if it doesn't exist or is empty
        if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(resolved_social_title or "")
                    f.write("\n\n")
                    f.write(resolved_social_desc or "")
                    f.write("\n\n")
                    f.write(resolved_social_caption or "")
                    f.write("\n\n")
                    f.write(" ".join(resolved_hashtags or []))
            except Exception as write_err:
                logger.warning(f"Could not auto-write missing caption file {txt_path}: {write_err}")

        intro_video_path = os.path.join(clips_dir, "hooks", f"intro_{name}.mp4")
        clip_result = {
            "clip_name": name,
            "title": resolved_title,
            "duration": clip.get("total_duration", 0),
            "hook_score": clip.get("hook_score", 0),
            "flow_score": clip.get("flow_score", 0),
            "virality_score": clip.get("virality_score", 0),
            "reason": resolved_reason,
            "social_title": resolved_social_title,
            "social_description": resolved_social_desc,
            "social_caption": resolved_social_caption,
            "hashtags": resolved_hashtags,
            "has_raw": os.path.exists(raw_path),
            "has_hooked": os.path.exists(intro_video_path),
            "has_captioned": bool(caption_variants),
            "has_srt": bool(srt_file),
            "has_ass": os.path.exists(ass_path),
            "has_caption_txt": os.path.exists(txt_path),
            "caption_failed": not os.path.exists(raw_path) or (os.path.exists(raw_path) and not caption_variants),
            "caption_variants": caption_variants,
            "words": words,
            "hook_title": resolved_hook_title,
            "files": {
                "raw": f"{name}_raw.mp4",
                "hooked": f"hooks/intro_{name}.mp4" if os.path.exists(intro_video_path) else None,
                "captioned": primary_caption,
                "srt": srt_file,
                "ass": f"{name}.ass",
                "caption": f"{name}_caption.txt",
            },
        }
        results["clips"].append(clip_result)

    # Save results
    results_path = os.path.join(job_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results packaged: {len(results['clips'])} clips")
    return results


def _caption_variants(clips_dir, clip_name):
    variants = []
    if not os.path.isdir(clips_dir):
        return variants
    prefix = f"{clip_name}_captioned"
    for filename in sorted(os.listdir(clips_dir)):
        if not filename.startswith(prefix) or not filename.endswith(".mp4"):
            continue
        suffix = filename[len(prefix):-4].strip("_")
        variants.append({
            "file": filename,
            "label": suffix.replace("_", " ").title() if suffix else "Default",
            "suffix": suffix,
        })
    return variants


def _primary_caption_file(clips_dir, clip_name, caption_variants, settings, legacy_cap_path):
    """Prefer the active caption style variant (hooked first), then newest caption output."""
    if not caption_variants:
        return f"{clip_name}_captioned.mp4"

    wanted_suffix = _caption_suffix_from_settings(settings or {})
    if wanted_suffix:
        # Prioritize hooked variant of the wanted style suffix first
        wanted_hooked = wanted_suffix + "_hooked"
        for variant in caption_variants:
            if variant.get("suffix") == wanted_hooked:
                return variant["file"]
        # Fallback to unhooked variant of the wanted style suffix
        for variant in caption_variants:
            if variant.get("suffix") == wanted_suffix:
                return variant["file"]

    # Also try to find any hooked file among the variants if no style suffix matched
    for variant in caption_variants:
        if variant.get("suffix") and variant.get("suffix").endswith("_hooked"):
            return variant["file"]

    newest = max(
        caption_variants,
        key=lambda v: os.path.getmtime(os.path.join(clips_dir, v["file"])),
    )
    if newest:
        return newest["file"]

    if os.path.exists(legacy_cap_path):
        return f"{clip_name}_captioned.mp4"
    return caption_variants[0]["file"]


def _caption_suffix_from_settings(settings):
    explicit = str(settings.get("caption_output_suffix", "") or "").strip()
    style_name = str(settings.get("caption_style", "") or "").strip()
    value = explicit or style_name
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_")
    return suffix


def _primary_srt_file(clips_dir, clip_name, settings):
    filename = f"{clip_name}.srt"
    if os.path.exists(os.path.join(clips_dir, filename)):
        return filename
    return ""
