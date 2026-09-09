"""
Video Clipper — Flask Application
==================================
Central server: handles all API routes, serves the web UI, manages jobs.

KEY CHANGES vs previous version:
  • Every job stores its actual output_dir (folder is renamed to
    "Video N - Title" after the download stage knows the video title).
  • _find_job_dir(job_id) looks up the correct folder from JOBS or, after
    a server restart, scans output folders for the hidden .job_id marker file.
  • All API file-serving endpoints use _find_job_dir — they don't assume the
    folder name == job_id any more.
  • Integrated async background music downloading & loudness profiling from captioner.py.
  • Added support for emoji_captions mapping via configuration and settings pipeline.
  • NEW: /api/reburn  — re-burn captions on all or selected clips from a
    completed job, with optional style/music overrides.  Uses the existing
    run_pipeline_from("caption") machinery so no duplicate logic.
"""

import json
import os
import sys
import threading
import time
import uuid
import logging
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import config
from pipeline.runner import run_pipeline, run_pipeline_from, run_single_stage
from pipeline.analyzer import test_api_key
from pipeline.logger import get_log_contents
from pipeline.captioner import pre_process_music_async, rename_music_job_dir

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger("video_clipper")
# Silence redundant Werkzeug HTTP request logging while keeping pipeline logs
logging.getLogger("werkzeug").setLevel(logging.WARNING)

JOBS = {}
JOBS_LOCK = threading.Lock()

ACTIVE_URLS = {}
ACTIVE_URLS_LOCK = threading.Lock()

ALLOWED_MUSIC_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}

_CAPTION_OVERRIDE_KEYS = [
    "caption_font_color",
    "caption_highlight_color",
    "caption_word_case",
    "caption_line_spacing",
    "caption_entrance_anim",
    "caption_font_size",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_provider_name(provider: str) -> str:
    """Map legacy/aliased provider strings to canonical ones so a job saved
    under an old provider (featherless/openrouter) resolves the right key."""
    p = str(provider or "").strip().lower()
    aliases = {
        "anthropic": "claude", "google": "gemini", "chatgpt": "openai",
        "nim": "nvidia", "nvidia_nim": "nvidia",
        "featherless": "nvidia", "featherless.ai": "nvidia",
        "featherless_ai": "nvidia", "openrouter": "nvidia",
    }
    return aliases.get(p, p)


def _get_provider_api_key(provider: str) -> str:
    provider = _normalize_provider_name(provider)
    if provider == "claude":
        return config.ANTHROPIC_API_KEY
    if provider == "gemini":
        return config.GOOGLE_API_KEY
    if provider == "nvidia":
        return config.NVIDIA_API_KEY
    if provider == "openai":
        return config.OPENAI_API
    return ""


def _clean_secret(value) -> str:
    return str(value or "").strip()


# Distinctive key prefixes per provider. Used to reject a stale key inherited
# from a prior job's settings.json that belonged to a *different* provider
# (e.g. an OpenAI sk-proj-… key stored on a job now set to provider=nvidia →
# it would otherwise be sent as the NVIDIA bearer and 401).
_PROVIDER_KEY_PREFIXES = {
    "nvidia": ("nvapi-",),
    "claude": ("sk-ant-",),
    "gemini": ("AIza",),
    "openai": ("sk-",),
}


def _resolve_ai_api_key(provider: str, api_key: str = "") -> str:
    """Resolve the key for the selected provider.

    Prefers the per-request key, but if that key clearly belongs to a different
    provider (wrong prefix) we discard it and use the configured env key —
    otherwise a stale cross-provider key from settings.json causes auth (401)
    failures on resume/retry.
    """
    provider = _normalize_provider_name(provider)
    explicit_key = _clean_secret(api_key)
    env_key = _get_provider_api_key(provider)
    if explicit_key:
        prefixes = _PROVIDER_KEY_PREFIXES.get(provider)
        if prefixes and not explicit_key.startswith(prefixes) and env_key:
            return env_key  # supplied key is for another provider → use env
        return explicit_key
    return env_key


def _get_request_data():
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        return request.form
    return request.get_json(silent=True) or {}


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    parsed = _to_int(value, default)
    return max(minimum, min(maximum, parsed))


def _load_job_settings(job_dir: str) -> dict:
    path = os.path.join(job_dir, "settings.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_job_settings(job_dir: str, settings: dict) -> None:
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "settings.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _apply_caption_overrides(settings: dict, data, original_settings: dict | None = None) -> None:
    """Copy caption UI override keys into a pipeline settings dict."""
    original_settings = original_settings or {}
    for key in _CAPTION_OVERRIDE_KEYS:
        sentinel = object()
        value = data.get(key, sentinel)
        if value is sentinel:
            if key in original_settings:
                settings[key] = original_settings[key]
            continue
        if value in (None, "", "style"):
            continue
        if key == "caption_line_spacing":
            settings[key] = _to_float(
                value,
                original_settings.get(
                    key,
                    config.CAPTION_COMMON_STYLE.get(
                        "line_spacing_ratio", 0.96),
                ),
            )
        elif key == "caption_font_size":
            settings[key] = _clamp_int(
                value,
                original_settings.get(
                    key,
                    config.CAPTION_COMMON_STYLE.get("font_size", 165),
                ),
                100,
                350,
            )
        else:
            settings[key] = value


def _save_music_upload(job_dir: str, field_name: str = "music_file") -> str:
    upload = request.files.get(field_name)
    if not upload or not upload.filename:
        return ""
    filename = secure_filename(upload.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_MUSIC_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_MUSIC_EXTENSIONS))
        raise ValueError(f"Unsupported music file type. Allowed: {allowed}")
    os.makedirs(job_dir, exist_ok=True)
    music_path = os.path.join(job_dir, f"background_music{ext}")
    upload.save(music_path)
    return os.path.abspath(music_path)


def _parse_clip_selection(raw: str) -> str:
    """
    Normalise clip selection input from the UI.

      'all'  (any case)  → 'all'
      '1,5,7'            → 'clip_01,clip_05,clip_07'
      'clip_01,clip_03'  → 'clip_01,clip_03'  (unchanged)
      ''  /  None        → 'all'
    """
    val = (raw or "all").strip()
    if not val or val.lower() == "all":
        return "all"
    parts = [p.strip() for p in val.split(",") if p.strip()]
    expanded = []
    for part in parts:
        if part.isdigit():
            expanded.append(f"clip_{int(part):02d}")
        else:
            expanded.append(part)
    return ",".join(expanded)


def _find_job_dir(job_id: str) -> str:
    """
    Return the actual output folder for a job_id.

    Priority:
      1. JOBS dict (fastest — live jobs)
      2. Scan OUTPUT_DIR for a folder containing a .job_id marker file
         (handles server restarts — the marker is written by runner.py)
      3. Legacy fallback: output_dir == job_id (old naming scheme)
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job:
        return job.get("output_dir", os.path.join(config.OUTPUT_DIR, job_id))

    if os.path.exists(config.OUTPUT_DIR):
        for folder_name in os.listdir(config.OUTPUT_DIR):
            folder_path = os.path.join(config.OUTPUT_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
            marker = os.path.join(folder_path, ".job_id")
            if os.path.isfile(marker):
                try:
                    with open(marker, "r", encoding="utf-8") as f:
                        if job_id in f.read().split():
                            return folder_path
                except OSError:
                    pass

    return os.path.join(config.OUTPUT_DIR, job_id)


def _safe_under(base_dir: str, *user_parts: str) -> str | None:
    """Join user-supplied path parts under base_dir and confirm the resolved
    path stays inside base_dir. Returns the absolute path, or None on traversal
    attempts (e.g. '..%2f..%2f.env'). Guards the file-serving routes —
    FLASK_HOST=0.0.0.0 means these are reachable on the LAN.
    """
    base_abs = os.path.abspath(base_dir)
    # Collapse any stray separators/encoded traversal in each part.
    candidate = os.path.abspath(os.path.join(base_abs, *user_parts))
    try:
        if os.path.commonpath([base_abs, candidate]) != base_abs:
            return None
    except ValueError:
        # Different drive / invalid path → reject.
        return None
    return candidate


def _make_progress_callback(job_id: str):
    """Factory: returns a progress_callback that updates JOBS[job_id]."""
    def progress_callback(stage, percent, message):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["stage"] = stage
                JOBS[job_id]["progress"] = percent
                JOBS[job_id]["message"] = message
                JOBS[job_id]["updated_at"] = time.time()
                if stage == "done":
                    JOBS[job_id]["status"] = "done"
                elif stage == "error":
                    JOBS[job_id]["status"] = "error"
                elif stage == "review":
                    JOBS[job_id]["status"] = "paused_for_review"
                elif stage == "paused_external_llm":
                    JOBS[job_id]["status"] = "paused_for_external_llm"
                else:
                    JOBS[job_id]["status"] = "processing"
                JOBS[job_id]["events"].append({
                    "stage":     stage,
                    "progress":  percent,
                    "message":   message,
                    "timestamp": time.time(),
                })
    return progress_callback


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "claude_models":          config.CLAUDE_MODELS,
        "gemini_models":          config.GEMINI_MODELS,
        "nvidia_models":          config.NVIDIA_MODELS,
        "openai_models":          config.OPENAI_MODELS,
        "default_provider":       config.DEFAULT_AI_PROVIDER,
        "default_claude_model":   config.DEFAULT_CLAUDE_MODEL,
        "default_gemini_model":   config.DEFAULT_GEMINI_MODEL,
        "default_nvidia_model":   config.DEFAULT_NVIDIA_MODEL,
        "default_openai_model":   config.DEFAULT_OPENAI_MODEL,
        "nvidia_test_model":      config.NVIDIA_TEST_MODEL,
        "nvidia_base_url":        config.NVIDIA_BASE_URL,
        "ai_cross_provider_fallback": config.AI_CROSS_PROVIDER_FALLBACK,
        "default_analysis_mode":  getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"),
        "default_clip_count":     config.DEFAULT_CLIP_COUNT,
        "default_min_duration":   config.DEFAULT_MIN_DURATION,
        "default_max_duration":   config.DEFAULT_MAX_DURATION,
        "default_aspect_ratio":   config.DEFAULT_ASPECT_RATIO,
        "default_caption_style":  config.DEFAULT_CAPTION_STYLE,
        "default_music_volume":   config.BACKGROUND_MUSIC_DEFAULT_VOLUME,
        "default_skip_refinement": False,
        "default_refinement_batch_size": getattr(config, "REFINEMENT_BATCH_SIZE", 5),
        "default_skip_compression": False,
        "default_compression_batch_size": 10,
        "default_emoji_captions":  config.DEFAULT_EMOJI_CAPTIONS,
        "caption_styles":         {k: v["name"] for k, v in config.CAPTION_STYLES.items()},
        "has_anthropic_key":      bool(config.ANTHROPIC_API_KEY),
        "has_google_key":         bool(config.GOOGLE_API_KEY),
        "has_nvidia_key":         bool(config.NVIDIA_API_KEY),
        "has_openai_key":         bool(config.OPENAI_API),
        "has_smallest_key":       bool(config.SMALLEST_API_KEY),
        "tts_hook_enabled":       config.TTS_HOOK_ENABLED,
        "tts_hook_voice_id":      config.TTS_HOOK_VOICE_ID,
        "tts_hook_sample_rate":   config.TTS_HOOK_SAMPLE_RATE,
        "tts_hook_speed":         config.TTS_HOOK_SPEED,
        "tts_hook_font_size_pct": config.TTS_HOOK_FONT_SIZE_PCT,
        "tts_hook_text_placement": config.TTS_HOOK_TEXT_PLACEMENT,
        "tts_hook_text_color":    config.TTS_HOOK_TEXT_COLOR,
        "tts_hook_text_color_revealed": config.TTS_HOOK_TEXT_COLOR_REVEALED,
        "transcription_method":   config.TRANSCRIPTION_METHOD,
        "transcription_fallback_order": config.TRANSCRIPTION_FALLBACK_ORDER,
        "blog_post_enabled":      config.BLOG_POST_ENABLED,
        "pause_before_captioning": True,
        "runtime_source_root":    os.path.abspath(os.getcwd()),
        "runtime_python":         sys.executable,
        "runtime_analyzer_file":  os.path.abspath(os.path.join("pipeline", "analyzer.py")),
    })


@app.route("/api/test-api", methods=["POST"])
@app.route("/api/test-key", methods=["POST"])
def api_test_key():
    data = _get_request_data()
    provider = data.get("provider", config.DEFAULT_AI_PROVIDER)
    api_key = data.get("api_key", "")
    model = data.get("model", "")
    if not api_key:
        api_key = _get_provider_api_key(provider)
    if not api_key:
        return jsonify({"success": False, "message": "No API key provided"})
    return jsonify(test_api_key(provider, api_key, model))


@app.route("/api/download-direct", methods=["POST"])
def api_download_direct():
    data = _get_request_data()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"success": False, "message": "No URL provided"})

    job_id = f"DL_{int(time.time())}_{uuid.uuid4().hex[:4]}"
    job_dir = os.path.join(config.OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id":           job_id,
            "url":          url,
            "status":       "starting",
            "stage":        "downloading",
            "progress":     0,
            "message":      "Initializing download...",
            "created_at":   time.time(),
            "updated_at":   time.time(),
            "events":       [],
            "output_dir":   job_dir,
            "display_name": f"Download: {url[:30]}...",
        }

    def run_direct_download():
        import logging
        download_logger = logging.getLogger(f"dl_direct_{job_id}")
        
        def progress_cb(pct, message):
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = pct
                    JOBS[job_id]["message"] = message
                    JOBS[job_id]["updated_at"] = time.time()
                    JOBS[job_id]["events"].append({
                        "stage":     "downloading",
                        "progress":  pct,
                        "message":   message,
                        "timestamp": time.time(),
                    })
        
        try:
            from test_download import download_video
            meta = download_video(job_dir, url, download_logger, progress_callback=progress_cb)
            video_filename = meta.get("video_filename")
            if not video_filename:
                raise RuntimeError("Failed to determine filename")

            import re
            safe_title = re.sub(r'[<>:"/\\|?*]', '', meta.get("title", ""))[:50].strip()
            if not safe_title:
                raise RuntimeError("Failed to get video title")

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
            os.rename(job_dir, new_job_dir)
            
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "done"
                    JOBS[job_id]["progress"] = 100
                    JOBS[job_id]["message"] = "Download complete"
                    JOBS[job_id]["n"] = next_n
                    JOBS[job_id]["safe_title"] = safe_title
                    JOBS[job_id]["video_filename"] = video_filename
                    JOBS[job_id]["updated_at"] = time.time()
                    JOBS[job_id]["events"].append({
                        "stage":     "done",
                        "progress":  100,
                        "message":   "Download complete",
                        "timestamp": time.time(),
                    })
        except Exception as e:
            download_logger.error(f"Direct download failed: {e}")
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["message"] = str(e)
                    JOBS[job_id]["updated_at"] = time.time()
                    JOBS[job_id]["events"].append({
                        "stage":     "error",
                        "progress":  0,
                        "message":   str(e),
                        "timestamp": time.time(),
                    })

    threading.Thread(target=run_direct_download, daemon=True).start()

    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/download-file-direct/<n>/<title>/<filename>")
def api_download_file_direct(n, title, filename):
    folder_name = f"Video {n} - {os.path.basename(title)}"
    folder_path = os.path.join(config.OUTPUT_DIR, folder_name)
    file_path = _safe_under(folder_path, os.path.basename(filename))
    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Resume pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/resume", methods=["POST"])
def api_resume():
    """Resume pipeline from a specific stage on an existing job folder."""
    data = _get_request_data()
    job_folder = data.get("job_folder", "").strip()
    resume_from = data.get("resume_from", "caption").strip()

    if not job_folder:
        return jsonify({"error": "No job folder path provided"}), 400

    if os.path.isabs(job_folder):
        job_dir = job_folder
    else:
        resolved = _find_job_dir(job_folder)
        job_dir = resolved if os.path.isdir(resolved) else os.path.join(config.OUTPUT_DIR, job_folder)

    if not os.path.isdir(job_dir):
        return jsonify({"error": f"Folder not found: {job_dir}"}), 404

    valid_stages = ("transcribe", "analyze", "extract", "tts_hook", "caption", "copy")
    if resume_from not in valid_stages:
        return jsonify({"error": f"Invalid resume_from. Must be one of: {valid_stages}"}), 400

    # Single-stage mode: run exactly one stage and stop (overrides resume_from)
    single_stage = str(data.get("single_stage") or "").strip().lower()
    valid_single = ("download_video", "download_audio", "music_audio",
                    "transcribe", "analyze", "extract", "tts_hook", "caption")
    if single_stage and single_stage not in valid_single:
        return jsonify({"error": f"Invalid single_stage. Must be one of: {valid_single}"}), 400

    # ── Manual analysis route: persist any pasted external-LLM output ─────────
    # When the user submits the dashboard's "Manual analysis" box, we save it to
    # <job>/external_llm.txt and clear stale analysis/clips so the manual route
    # re-ingests the fresh paste instead of loading an old analysis.json.
    external_text = data.get("external_llm_text")
    if external_text is not None and str(external_text).strip():
        try:
            from pipeline import external_llm as _ext
            _ext.save_external_llm_input(job_dir, str(external_text))
            for stale in ("analysis.json", "clips_plan.json"):
                stale_path = os.path.join(job_dir, stale)
                if os.path.exists(stale_path):
                    try:
                        os.remove(stale_path)
                    except OSError:
                        pass
        except Exception as exc:
            return jsonify({"error": f"Failed to save manual analysis input: {exc}"}), 400

    orig_settings = _load_job_settings(job_dir)
    if not orig_settings:
        with JOBS_LOCK:
            for job in JOBS.values():
                if os.path.abspath(job.get("output_dir", "")) == os.path.abspath(job_dir):
                    orig_settings = job.get("settings", {})
                    break

    def get_val(key, default=None):
        val = data.get(key)
        if val is not None and val != "":
            return val
        return orig_settings.get(key, default)

    def get_bool(key, default=False):
        val = data.get(key)
        if val is not None and val != "":
            return _to_bool(val, default)
        return _to_bool(orig_settings.get(key), default)

    def get_int(key, default=0):
        val = data.get(key)
        if val is not None and val != "":
            return _to_int(val, default)
        return _to_int(orig_settings.get(key), default)

    def get_float(key, default=0.0):
        val = data.get(key)
        if val is not None and val != "":
            return _to_float(val, default)
        return _to_float(orig_settings.get(key), default)

    job_id = f"resume_{int(time.time())}_{uuid.uuid4().hex[:4]}"
    smallest_key = _clean_secret(data.get("smallest_key")) or orig_settings.get("smallest_key", "")
    
    get_all_clips = get_bool("get_all_clips", True)
    if get_all_clips:
        clip_count = 999
    else:
        clip_count = _clamp_int(
            data.get("clip_count") if data.get("clip_count") is not None and data.get("clip_count") != "" else orig_settings.get("clip_count"),
            config.DEFAULT_CLIP_COUNT, 1, 50
        )

    # background music overrides logic
    music_mode = data.get("music_mode", "keep")
    music_path = ""
    music_url = ""
    music_volume = 0.0

    if music_mode == "remove":
        music_path = "none"
        music_url = ""
        music_volume = 0.0
    elif music_mode == "link":
        music_url = data.get("music_url", "").strip()
        music_path = ""
        music_volume = max(0.0, min(1.0, _to_float(data.get("music_volume"), config.BACKGROUND_MUSIC_DEFAULT_VOLUME)))
    elif music_mode == "upload":
        try:
            music_path = _save_music_upload(job_dir, "resume_music_file")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        music_url = ""
        music_volume = max(0.0, min(1.0, _to_float(data.get("music_volume"), config.BACKGROUND_MUSIC_DEFAULT_VOLUME)))
    else: # "keep"
        # Look for existing background_music file in the job directory
        existing_music = ""
        if os.path.isdir(job_dir):
            for fn in os.listdir(job_dir):
                if fn.startswith("background_music") and os.path.splitext(fn)[1].lower() in ALLOWED_MUSIC_EXTENSIONS:
                    existing_music = os.path.abspath(os.path.join(job_dir, fn))
                    break
        if existing_music:
            music_path = existing_music
            music_url = ""
        else:
            music_path = orig_settings.get("music_path", "")
            music_url = orig_settings.get("music_url", "")
            
        req_vol = data.get("music_volume")
        if req_vol is not None and req_vol != "":
            music_volume = max(0.0, min(1.0, _to_float(req_vol, config.BACKGROUND_MUSIC_DEFAULT_VOLUME)))
        else:
            music_volume = _to_float(orig_settings.get("music_volume"), config.BACKGROUND_MUSIC_DEFAULT_VOLUME)

    music_start_time = data.get("music_start_time")
    if music_start_time is None or music_start_time == "":
        music_start_time = orig_settings.get("music_start_time")

    settings = {
        "ai_provider":          get_val("ai_provider",     config.DEFAULT_AI_PROVIDER),
        "ai_model":             get_val("ai_model",         ""),
        "api_key":              get_val("api_key",           ""),
        "analysis_mode":        str(get_val("analysis_mode", getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"))).strip().lower(),
        "smallest_key":         smallest_key,
        "clip_count":           clip_count,
        "get_all_clips":        get_all_clips,
        "min_duration":         get_int("min_duration",   config.DEFAULT_MIN_DURATION),
        "max_duration":         get_int("max_duration",   config.DEFAULT_MAX_DURATION),
        "caption_style":        get_val("caption_style",    config.DEFAULT_CAPTION_STYLE),
        # Always present so no burn can silently fall back to the style's
        # built-in size (_apply_caption_overrides re-clamps when the UI sends it)
        "caption_font_size":    get_int("caption_font_size", config.CAPTION_COMMON_STYLE.get("font_size", 165)),
        "aspect_ratio":         get_val("aspect_ratio",     config.DEFAULT_ASPECT_RATIO),
        "transcription_method": get_val("transcription_method", config.TRANSCRIPTION_METHOD),
        "music_path":           music_path,
        "music_url":            music_url,
        "music_volume":         music_volume,
        "music_start_time":     music_start_time,
        "preserve_caption_variants": get_bool("preserve_caption_variants", True),
        "caption_output_suffix": get_val("caption_output_suffix", get_val("caption_style", config.DEFAULT_CAPTION_STYLE)),
        "skip_copywriting":     get_bool("skip_copywriting", False) or resume_from == "caption",
        "enable_reasoning":     get_bool("enable_reasoning", False),
        "skip_refinement":      get_bool("skip_refinement", False),
        "skip_stitch_pass":     get_bool("skip_stitch_pass", False),
        "refinement_batch_size": get_int("refinement_batch_size", getattr(config, "REFINEMENT_BATCH_SIZE", 5)),
        "skip_compression":     get_bool("skip_compression", False),
        "compression_batch_size": get_int("compression_batch_size", 10),
        "skip_judge":           get_bool("skip_judge", getattr(config, "SKIP_JUDGE", True)),
        "skip_validation":      get_bool("skip_validation", False),
        "skip_duplication":     get_bool("skip_duplication", False),
        "allow_overlaps":       get_bool("allow_overlaps", True),
        "skip_duration_floor":  get_bool("skip_duration_floor", False),
        "enable_nano_tiebreak": get_bool("enable_nano_tiebreak", False),
        "analysis_window_seconds": get_float("analysis_window_seconds", config.AI_ANALYSIS_WINDOW_SECONDS),
        "analysis_overlap_seconds": get_float("analysis_overlap_seconds", config.AI_ANALYSIS_OVERLAP_SECONDS),
        "blog_post_enabled":    get_bool("blog_post_enabled", config.BLOG_POST_ENABLED),
        "emoji_captions":       get_bool("emoji_captions",  config.DEFAULT_EMOJI_CAPTIONS),
        "tts_hook_mode":        get_val("tts_hook_mode", "keep").strip().lower(),
        "tts_hook_enabled":     get_bool("tts_hook_enabled", config.TTS_HOOK_ENABLED),
        "tts_hook_voice_id":    get_val("tts_hook_voice_id", config.TTS_HOOK_VOICE_ID),
        "tts_hook_sample_rate": get_int("tts_hook_sample_rate", config.TTS_HOOK_SAMPLE_RATE),
        "tts_hook_speed":       get_float("tts_hook_speed", config.TTS_HOOK_SPEED),
        "tts_hook_font_size_pct": get_float("tts_hook_font_size_pct", config.TTS_HOOK_FONT_SIZE_PCT),
        "tts_hook_text_placement": get_val("tts_hook_text_placement", config.TTS_HOOK_TEXT_PLACEMENT),
        "tts_hook_text_color":    get_val("tts_hook_text_color", config.TTS_HOOK_TEXT_COLOR),
        "tts_hook_text_color_revealed": get_val("tts_hook_text_color_revealed", config.TTS_HOOK_TEXT_COLOR_REVEALED),
        "pause_before_captioning": get_bool("pause_before_captioning", True),
        "reburn_clips": _parse_clip_selection(data.get("selected_clips", "all")),
        # Ending-adjuster: when the review panel nudged a clip's end, its raw
        # was invalidated and auto-rewind re-extracts it — reuse the existing
        # framing templates so the template GUI doesn't relaunch over a nudge.
        "reuse_manual_templates": _to_bool(data.get("reuse_manual_templates"), False),
    }
    settings["api_key"] = _resolve_ai_api_key(
        settings["ai_provider"], settings["api_key"]
    )
    _apply_caption_overrides(settings, data)
    _save_job_settings(job_dir, settings)

    try:
        with open(os.path.join(job_dir, ".job_id"), "a", encoding="utf-8") as f:
            f.write(f"\n{job_id}")
    except Exception:
        pass

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id":           job_id,
            "url":          f"resume://{os.path.basename(job_dir)}",
            "status":       "starting",
            "stage":        "",
            "progress":     0,
            "message":      (f"Single stage: {single_stage}..." if single_stage
                             else f"Resuming from {resume_from}..."),
            "created_at":   time.time(),
            "updated_at":   time.time(),
            "settings":     settings,
            "events":       [],
            "output_dir":   job_dir,
            "display_name": os.path.basename(job_dir),
        }

    pre_process_music_async(job_dir, settings, logger)
    progress_callback = _make_progress_callback(job_id)

    def run_resume():
        try:
            if single_stage:
                run_single_stage(job_dir, single_stage,
                                 settings, progress_callback)
            else:
                run_pipeline_from(job_dir, resume_from,
                                  settings, progress_callback)
        except Exception as exc:
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["message"] = str(exc)
                    JOBS[job_id]["updated_at"] = time.time()
                    JOBS[job_id]["events"].append({
                        "stage": "error", "progress": -1,
                        "message": str(exc), "timestamp": time.time(),
                    })

    threading.Thread(target=run_resume, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started",
                    "resume_from": resume_from, "single_stage": single_stage or None})


@app.route("/api/external-llm/<job_id>", methods=["GET"])
def api_external_llm(job_id):
    """Return the paste-ready windows + the fixed system prompt for the manual
    analysis route (used by the 'paused_for_external_llm' panel)."""
    job_dir = _find_job_dir(job_id)
    if not os.path.isdir(job_dir):
        return jsonify({"error": f"Job not found: {job_id}"}), 404
    from pipeline import external_llm as _ext
    return jsonify({
        "job_id":             job_id,
        "job_folder":         os.path.basename(job_dir),
        "system_prompt":      _ext.read_system_prompt(),
        "system_prompt_path": getattr(config, "EXTERNAL_LLM_SYSTEM_PROMPT_FILE", "pipeline/prompts.txt"),
        "windows_text":       _ext.read_windows_text(job_dir),
        "has_input":          _ext.has_external_llm_input(job_dir),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Re-burn captions
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/reburn", methods=["POST"])
def api_reburn():
    """
    Re-burn captions on an existing job's clips with new style/music settings.

    Body fields:
      job_id          — job ID (used to locate the job folder)
      selected_clips  — "all"  OR  comma-separated clip names e.g. "clip_01,clip_03"
      caption_style   — optional style override
      aspect_ratio    — optional aspect ratio override
      music_volume    — optional 0-1 float
      music_url       — optional YouTube URL for background music
    """
    data = _get_request_data()
    job_id_src = data.get("job_id", "").strip()
    selected = data.get("selected_clips", "all").strip() or "all"

    if not job_id_src:
        return jsonify({"error": "job_id is required"}), 400

    job_dir = _find_job_dir(job_id_src)
    if not os.path.isdir(job_dir):
        return jsonify({"error": f"Job folder not found: {job_dir}"}), 404

    # Load original settings from the stored job (fall back to config defaults)
    with JOBS_LOCK:
        orig_settings = JOBS.get(job_id_src, {}).get("settings", {})

    settings = {
        # Inherit originals, then let caller override style/music
        "ai_provider":   orig_settings.get("ai_provider",  config.DEFAULT_AI_PROVIDER),
        "ai_model":      orig_settings.get("ai_model",     ""),
        "api_key":       orig_settings.get("api_key",      ""),
        "clip_count":    orig_settings.get("clip_count",   config.DEFAULT_CLIP_COUNT),
        "get_all_clips": orig_settings.get("get_all_clips", True),
        "min_duration":  orig_settings.get("min_duration", config.DEFAULT_MIN_DURATION),
        "max_duration":  orig_settings.get("max_duration", config.DEFAULT_MAX_DURATION),
        "aspect_ratio":  data.get("aspect_ratio", orig_settings.get("aspect_ratio", config.DEFAULT_ASPECT_RATIO)),
        "transcription_method": orig_settings.get("transcription_method", config.TRANSCRIPTION_METHOD),
        # Overrideable by caller
        "caption_style": data.get("caption_style", orig_settings.get("caption_style", config.DEFAULT_CAPTION_STYLE)),
        "music_path":    "",
        "music_url":     data.get("music_url", orig_settings.get("music_url", "")).strip(),
        "music_volume":  max(0.0, min(1.0, _to_float(
            data.get("music_volume"),
            orig_settings.get(
                "music_volume", config.BACKGROUND_MUSIC_DEFAULT_VOLUME)
        ))),
        "music_start_time": data.get("music_start_time", orig_settings.get("music_start_time")),
        "emoji_captions":   _to_bool(data.get("emoji_captions"), orig_settings.get("emoji_captions", config.DEFAULT_EMOJI_CAPTIONS)),
        # Hooks: a caption re-burn must never generate or delete hook files
        "tts_hook_mode": "keep",
        "tts_hook_enabled": orig_settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED),
        # Clip filter — passed through to captioner.burn_captions()
        "reburn_clips":  _parse_clip_selection(selected),
        "preserve_caption_variants": True,
        "caption_output_suffix": data.get("caption_output_suffix", data.get("caption_style", config.DEFAULT_CAPTION_STYLE)),
        "skip_copywriting": True,
    }
    _apply_caption_overrides(settings, data, orig_settings)

    new_job_id = f"reburn_{int(time.time())}_{uuid.uuid4().hex[:4]}"

    try:
        with open(os.path.join(job_dir, ".job_id"), "a", encoding="utf-8") as f:
            f.write(f"\n{new_job_id}")
    except Exception:
        pass

    with JOBS_LOCK:
        JOBS[new_job_id] = {
            "id":           new_job_id,
            "url":          f"reburn://{os.path.basename(job_dir)}",
            "status":       "starting",
            "stage":        "",
            "progress":     0,
            "message":      f"Re-burning captions ({selected})…",
            "created_at":   time.time(),
            "updated_at":   time.time(),
            "settings":     settings,
            "events":       [],
            "output_dir":   job_dir,
            "display_name": os.path.basename(job_dir),
            # Link back so the frontend can reload the original job's results
            "source_job_id": job_id_src,
        }

    progress_callback = _make_progress_callback(new_job_id)

    def run_reburn():
        try:
            run_pipeline_from(job_dir, "caption", settings, progress_callback)
        except Exception as exc:
            with JOBS_LOCK:
                if new_job_id in JOBS:
                    JOBS[new_job_id]["status"] = "error"
                    JOBS[new_job_id]["message"] = str(exc)
                    JOBS[new_job_id]["updated_at"] = time.time()
                    JOBS[new_job_id]["events"].append({
                        "stage": "error", "progress": -1,
                        "message": str(exc), "timestamp": time.time(),
                    })

    threading.Thread(target=run_reburn, daemon=True).start()
    return jsonify({
        "job_id":       new_job_id,
        "status":       "started",
        "source_job_id": job_id_src,
    })


@app.route("/api/retry", methods=["POST"])
def api_retry():
    """Retry a job from a selected stage without manually pasting the folder."""
    data = _get_request_data()
    source_job_id = data.get("job_id", "").strip()
    resume_from = data.get("resume_from", "copy").strip()
    valid_stages = ("transcribe", "analyze", "extract", "tts_hook", "caption", "copy")
    if resume_from not in valid_stages:
        return jsonify({"error": f"Invalid resume_from. Must be one of: {valid_stages}"}), 400
    if not source_job_id:
        return jsonify({"error": "job_id is required"}), 400

    job_dir = _find_job_dir(source_job_id)
    if not os.path.isdir(job_dir):
        return jsonify({"error": f"Job folder not found: {job_dir}"}), 404

    with JOBS_LOCK:
        orig_settings = JOBS.get(source_job_id, {}).get("settings", {})
    if not orig_settings:
        orig_settings = _load_job_settings(job_dir)

    caption_style = data.get("caption_style", orig_settings.get("caption_style", config.DEFAULT_CAPTION_STYLE))
    smallest_key = _clean_secret(data.get("smallest_key")) or orig_settings.get("smallest_key", "")
    api_key = _clean_secret(data.get("api_key")) or orig_settings.get("api_key", "")
    clip_count = _clamp_int(orig_settings.get("clip_count"), config.DEFAULT_CLIP_COUNT, 1, 50)
    settings = {
        "ai_provider": data.get("ai_provider", orig_settings.get("ai_provider", config.DEFAULT_AI_PROVIDER)),
        "ai_model": data.get("ai_model", orig_settings.get("ai_model", "")),
        "api_key": api_key,
        "analysis_mode": str(data.get("analysis_mode", orig_settings.get("analysis_mode", getattr(config, "DEFAULT_ANALYSIS_MODE", "manual")))).strip().lower(),
        "smallest_key": smallest_key,
        "clip_count": clip_count,
        "get_all_clips": _to_bool(data.get("get_all_clips"), orig_settings.get("get_all_clips", True)),
        "min_duration": orig_settings.get("min_duration", config.DEFAULT_MIN_DURATION),
        "max_duration": orig_settings.get("max_duration", config.DEFAULT_MAX_DURATION),
        "caption_style": caption_style,
        "aspect_ratio": data.get("aspect_ratio", orig_settings.get("aspect_ratio", config.DEFAULT_ASPECT_RATIO)),
        "transcription_method": data.get("transcription_method", orig_settings.get("transcription_method", config.TRANSCRIPTION_METHOD)),
        "music_path": "",
        "music_url": data.get("music_url", orig_settings.get("music_url", "")).strip(),
        "music_volume": max(0.0, min(1.0, _to_float(
            data.get("music_volume"),
            orig_settings.get("music_volume", config.BACKGROUND_MUSIC_DEFAULT_VOLUME)
        ))),
        "preserve_caption_variants": True,
        "caption_output_suffix": data.get("caption_output_suffix", caption_style),
        "skip_copywriting": resume_from == "caption",
        "blog_post_enabled": _to_bool(
            data.get("blog_post_enabled"),
            orig_settings.get("blog_post_enabled", config.BLOG_POST_ENABLED),
        ),
        "emoji_captions": _to_bool(data.get("emoji_captions"), orig_settings.get("emoji_captions", config.DEFAULT_EMOJI_CAPTIONS)),
        "tts_hook_mode": data.get("tts_hook_mode", "keep").strip().lower(),
        "tts_hook_enabled": _to_bool(data.get("tts_hook_enabled"), orig_settings.get("tts_hook_enabled", config.TTS_HOOK_ENABLED)),
        "tts_hook_voice_id": data.get("tts_hook_voice_id") or orig_settings.get("tts_hook_voice_id", config.TTS_HOOK_VOICE_ID),
        "tts_hook_sample_rate": _to_int(data.get("tts_hook_sample_rate"), orig_settings.get("tts_hook_sample_rate", config.TTS_HOOK_SAMPLE_RATE)),
        "tts_hook_speed": _to_float(data.get("tts_hook_speed"), orig_settings.get("tts_hook_speed", config.TTS_HOOK_SPEED)),
        "tts_hook_font_size_pct": _to_float(data.get("tts_hook_font_size_pct"), orig_settings.get("tts_hook_font_size_pct", config.TTS_HOOK_FONT_SIZE_PCT)),
        "tts_hook_text_placement": data.get("tts_hook_text_placement") or orig_settings.get("tts_hook_text_placement", config.TTS_HOOK_TEXT_PLACEMENT),
        "tts_hook_text_color": data.get("tts_hook_text_color") or orig_settings.get("tts_hook_text_color", config.TTS_HOOK_TEXT_COLOR),
        "tts_hook_text_color_revealed": data.get("tts_hook_text_color_revealed") or orig_settings.get("tts_hook_text_color_revealed", config.TTS_HOOK_TEXT_COLOR_REVEALED),
    }
    settings["api_key"] = _resolve_ai_api_key(
        settings["ai_provider"], settings["api_key"]
    )
    _apply_caption_overrides(settings, data, orig_settings)
    _save_job_settings(job_dir, settings)

    retry_job_id = f"retry_{int(time.time())}_{uuid.uuid4().hex[:4]}"
    
    try:
        with open(os.path.join(job_dir, ".job_id"), "a", encoding="utf-8") as f:
            f.write(f"\n{retry_job_id}")
    except Exception:
        pass

    with JOBS_LOCK:
        JOBS[retry_job_id] = {
            "id": retry_job_id,
            "url": f"retry://{os.path.basename(job_dir)}",
            "status": "starting",
            "stage": "",
            "progress": 0,
            "message": f"Retrying from {resume_from}...",
            "created_at": time.time(),
            "updated_at": time.time(),
            "settings": settings,
            "events": [],
            "output_dir": job_dir,
            "display_name": os.path.basename(job_dir),
            "source_job_id": source_job_id,
        }

    pre_process_music_async(job_dir, settings, logger)
    progress_callback = _make_progress_callback(retry_job_id)

    def run_retry():
        try:
            run_pipeline_from(job_dir, resume_from,
                              settings, progress_callback)
        except Exception as exc:
            with JOBS_LOCK:
                if retry_job_id in JOBS:
                    JOBS[retry_job_id]["status"] = "error"
                    JOBS[retry_job_id]["message"] = str(exc)
                    JOBS[retry_job_id]["updated_at"] = time.time()
                    JOBS[retry_job_id]["events"].append({
                        "stage": "error",
                        "progress": -1,
                        "message": str(exc),
                        "timestamp": time.time(),
                    })

    threading.Thread(target=run_retry, daemon=True).start()
    return jsonify({"job_id": retry_job_id, "status": "started", "source_job_id": source_job_id})


@app.route("/api/process", methods=["POST"])
def api_process():
    data = _get_request_data()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    with ACTIVE_URLS_LOCK:
        if url in ACTIVE_URLS:
            existing_id = ACTIVE_URLS[url]
            with JOBS_LOCK:
                existing_job = JOBS.get(existing_id)
                if existing_job and existing_job["status"] in ("starting", "processing"):
                    if existing_job.get("progress", 0) > 20:
                        return jsonify({
                            "error": f"URL already processing (Job: {existing_id}). Wait for it to complete.",
                            "existing_job_id": existing_id,
                        }), 409
                    del ACTIVE_URLS[url]

    job_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job_dir = os.path.join(config.OUTPUT_DIR, job_id)

    try:
        music_path = _save_music_upload(job_dir)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    smallest_key = _clean_secret(data.get("smallest_key",   ""))
    clip_count = _clamp_int(data.get("clip_count"),
                            config.DEFAULT_CLIP_COUNT, 1, 50)
    # ── "Get All Clips" mode: bypass clip_count limit ─────────────────────
    get_all_clips = str(data.get("get_all_clips", "true")).lower() in ("true", "1", "yes")
    if get_all_clips:
        clip_count = 999

    settings = {
        "ai_provider":          data.get("ai_provider",    config.DEFAULT_AI_PROVIDER),
        "ai_model":             data.get("ai_model",        ""),
        "api_key":              data.get("api_key",          ""),
        "analysis_mode":        str(data.get("analysis_mode", getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"))).strip().lower(),
        "smallest_key":         smallest_key,
        "clip_count":           clip_count,
        "get_all_clips":        get_all_clips,
        "min_duration":         _to_int(data.get("min_duration"),  config.DEFAULT_MIN_DURATION),
        "max_duration":         _to_int(data.get("max_duration"),  config.DEFAULT_MAX_DURATION),
        "caption_style":        data.get("caption_style",   config.DEFAULT_CAPTION_STYLE),
        "caption_font_size":    _to_int(data.get("caption_font_size"), config.CAPTION_COMMON_STYLE.get("font_size", 165)),
        "aspect_ratio":         data.get("aspect_ratio",    config.DEFAULT_ASPECT_RATIO),
        "transcription_method": data.get("transcription_method", config.TRANSCRIPTION_METHOD),
        "music_path":           music_path,
        "music_url":            data.get("music_url",       "").strip(),
        "music_volume":         max(0.0, min(1.0, _to_float(
            data.get("music_volume"), config.BACKGROUND_MUSIC_DEFAULT_VOLUME
        ))),
        "music_start_time":     data.get("music_start_time"),
        "skip_copywriting":     str(data.get("skip_copywriting", "")).lower() == "true",
        "enable_reasoning":     str(data.get("enable_reasoning", "false")).lower() == "true",
        "skip_refinement":      str(data.get("skip_refinement", "false")).lower() == "true",
        "skip_stitch_pass":     str(data.get("skip_stitch_pass", "false")).lower() == "true",
        "refinement_batch_size": _to_int(data.get("refinement_batch_size"), getattr(config, "REFINEMENT_BATCH_SIZE", 5)),
        "skip_compression":     str(data.get("skip_compression", "false")).lower() == "true",
        "compression_batch_size": _to_int(data.get("compression_batch_size"), 10),
        "skip_judge":           str(data.get("skip_judge", str(getattr(config, "SKIP_JUDGE", True)).lower())).lower() == "true",
        "skip_validation":      str(data.get("skip_validation", "false")).lower() == "true",
        "skip_duplication":     str(data.get("skip_duplication", "false")).lower() == "true",
        "allow_overlaps":       _to_bool(data.get("allow_overlaps"), True),
        "skip_duration_floor":  _to_bool(data.get("skip_duration_floor"), False),
        "enable_nano_tiebreak": _to_bool(data.get("enable_nano_tiebreak"), False),
        "analysis_window_seconds": _to_float(data.get("analysis_window_seconds"), config.AI_ANALYSIS_WINDOW_SECONDS),
        "analysis_overlap_seconds": _to_float(data.get("analysis_overlap_seconds"), config.AI_ANALYSIS_OVERLAP_SECONDS),
        "blog_post_enabled":    _to_bool(data.get("blog_post_enabled"), config.BLOG_POST_ENABLED),
        "emoji_captions":       _to_bool(data.get("emoji_captions"), config.DEFAULT_EMOJI_CAPTIONS),
        "tts_hook_mode":        data.get("tts_hook_mode", "keep").strip().lower(),
        "tts_hook_enabled":     _to_bool(data.get("tts_hook_enabled"), config.TTS_HOOK_ENABLED),
        "tts_hook_voice_id":    data.get("tts_hook_voice_id", config.TTS_HOOK_VOICE_ID),
        "tts_hook_sample_rate": _to_int(data.get("tts_hook_sample_rate"), config.TTS_HOOK_SAMPLE_RATE),
        "tts_hook_speed":       _to_float(data.get("tts_hook_speed"), config.TTS_HOOK_SPEED),
        "tts_hook_font_size_pct": _to_float(data.get("tts_hook_font_size_pct"), config.TTS_HOOK_FONT_SIZE_PCT),
        "tts_hook_text_placement": data.get("tts_hook_text_placement", config.TTS_HOOK_TEXT_PLACEMENT),
        "tts_hook_text_color":    data.get("tts_hook_text_color", config.TTS_HOOK_TEXT_COLOR),
        "tts_hook_text_color_revealed": data.get("tts_hook_text_color_revealed", config.TTS_HOOK_TEXT_COLOR_REVEALED),
        "pause_before_captioning": _to_bool(data.get("pause_before_captioning"), True),
    }

    settings["api_key"] = _resolve_ai_api_key(
        settings["ai_provider"], settings["api_key"]
    )
    _apply_caption_overrides(settings, data)
    _save_job_settings(job_dir, settings)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id":           job_id,
            "url":          url,
            "status":       "starting",
            "stage":        "",
            "progress":     0,
            "message":      "Initializing...",
            "created_at":   time.time(),
            "updated_at":   time.time(),
            "settings":     settings,
            "events":       [],
            "output_dir":   job_dir,
            "display_name": job_id,
        }

    with ACTIVE_URLS_LOCK:
        ACTIVE_URLS[url] = job_id

    pre_process_music_async(job_dir, settings, logger)
    threading.Thread(target=_run_job, args=(
        job_id, url, settings), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started"})


# ═══════════════════════════════════════════════════════════════════════════════
# Progress / results / download
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    def event_stream():
        last_event_count = 0
        last_heartbeat_at = time.time()
        while True:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            events = job.get("events", [])
            new_events = events[last_event_count:]
            for evt in new_events:
                yield f"data: {json.dumps(evt)}\n\n"
            last_event_count = len(events)

            now = time.time()
            if now - last_heartbeat_at >= 10:
                heartbeat = {
                    "stage":         job["stage"],
                    "progress":      job["progress"],
                    "message":       job["message"],
                    "status":        job["status"],
                    "heartbeat":     True,
                    "stale_seconds": int(now - job.get("updated_at", job["created_at"])),
                }
                yield f"data: {json.dumps(heartbeat)}\n\n"
                last_heartbeat_at = now

            if job["status"] in ("done", "error"):
                final_data = {
                    "stage":    job["stage"],
                    "progress": job["progress"],
                    "message":  job["message"],
                    "status":   job["status"]
                }
                if "n" in job:
                    final_data["n"] = job["n"]
                if "safe_title" in job:
                    final_data["safe_title"] = job["safe_title"]
                if "video_filename" in job:
                    final_data["video_filename"] = job["video_filename"]
                yield f"data: {json.dumps(final_data)}\n\n"
                break

            time.sleep(1)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/jobs")
def api_jobs():
    with JOBS_LOCK:
        return jsonify([
            {
                "id":           jid,
                "url":          job["url"],
                "status":       job["status"],
                "progress":     job["progress"],
                "stage":        job["stage"],
                "message":      job["message"],
                "created_at":   job["created_at"],
                "updated_at":   job.get("updated_at", job["created_at"]),
                "display_name": job.get("display_name", jid),
                "stale_seconds": int(time.time() - job.get("updated_at", job["created_at"])),
            }
            for jid, job in JOBS.items()
        ])


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({
            "id":              job["id"],
            "url":             job["url"],
            "status":          job["status"],
            "stage":           job["stage"],
            "progress":        job["progress"],
            "message":         job["message"],
            "created_at":      job["created_at"],
            "updated_at":      job.get("updated_at", job["created_at"]),
            "display_name":    job.get("display_name", job_id),
            "stale_seconds":   int(time.time() - job.get("updated_at", job["created_at"])),
            "source_job_id":   job.get("source_job_id"),   # set on reburn jobs
            "settings":        job.get("settings", {}),
        })


@app.route("/api/results/<job_id>")
def api_results(job_id):
    job_dir = _find_job_dir(job_id)
    results_path = os.path.join(job_dir, "results.json")
    if not os.path.exists(results_path):
        return jsonify({"error": "Results not found"}), 404
    with open(results_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/clips-plan/<job_id>", methods=["GET", "POST"])
def api_clips_plan(job_id):
    from pathlib import Path
    job_dir = _find_job_dir(job_id)
    plan_path = os.path.join(job_dir, "clips_plan.json")
    
    if request.method == "GET":
        if not os.path.exists(plan_path):
            return jsonify({"error": "clips_plan.json not found"}), 404
        with open(plan_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
            
    elif request.method == "POST":
        data = request.get_json() or {}
        edited_clips = data.get("clips")
        if not isinstance(edited_clips, list):
            return jsonify({"error": "Invalid payload format. Expected list of clips."}), 400
            
        if not os.path.exists(plan_path):
            return jsonify({"error": "Original clips_plan.json not found"}), 404
            
        with open(plan_path, "r", encoding="utf-8") as f:
            clips_plan = json.load(f)
            
        # Load transcript.json for original word alignment
        transcript_path = os.path.join(job_dir, "transcript.json")
        transcript = {}
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
                
        from pipeline.local_clips_generator import flatten_transcript_words
        original_words = flatten_transcript_words(transcript)
        
        # Build lookup for original words using difflib for robust spelling edits
        def align_words(new_text, original_words_subset):
            import difflib
            new_word_tokens = new_text.strip().split()
            if not new_word_tokens:
                return []

            orig_tokens = [w.text for w in original_words_subset]
            matcher = difflib.SequenceMatcher(None, orig_tokens, new_word_tokens)
            
            aligned = [None] * len(new_word_tokens)
            
            clip_start = original_words_subset[0].start if original_words_subset else 0.0
            clip_end = original_words_subset[-1].end if original_words_subset else 10.0
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    for offset in range(i2 - i1):
                        orig_w = original_words_subset[i1 + offset]
                        aligned[j1 + offset] = {
                            "word": new_word_tokens[j1 + offset],
                            "start": orig_w.start,
                            "end": orig_w.end
                        }
                elif tag == 'replace':
                    if (i2 - i1) == (j2 - j1):
                        for offset in range(i2 - i1):
                            orig_w = original_words_subset[i1 + offset]
                            aligned[j1 + offset] = {
                                "word": new_word_tokens[j1 + offset],
                                "start": orig_w.start,
                                "end": orig_w.end
                            }
                    else:
                        segment_start = original_words_subset[i1].start
                        segment_end = original_words_subset[i2 - 1].end
                        n_new = j2 - j1
                        step = (segment_end - segment_start) / n_new
                        for offset in range(n_new):
                            aligned[j1 + offset] = {
                                "word": new_word_tokens[j1 + offset],
                                "start": round(segment_start + offset * step, 3),
                                "end": round(segment_start + (offset + 1) * step, 3)
                            }
                elif tag == 'insert':
                    for offset in range(j2 - j1):
                        aligned[j1 + offset] = {
                            "word": new_word_tokens[j1 + offset],
                            "start": None,
                            "end": None
                        }

            n = len(aligned)
            for i in range(n):
                if aligned[i] is None or aligned[i]["start"] is None:
                    prev_time = clip_start
                    for j in range(i - 1, -1, -1):
                        if aligned[j] and aligned[j]["end"] is not None:
                            prev_time = aligned[j]["end"]
                            break
                    next_time = clip_end
                    for j in range(i + 1, n):
                        if aligned[j] and aligned[j]["start"] is not None:
                            next_time = aligned[j]["start"]
                            break
                    
                    missing_count = 0
                    for j in range(i, n):
                        if aligned[j] is None or aligned[j]["start"] is None:
                            missing_count += 1
                        else:
                            break
                    
                    step = (next_time - prev_time) / (missing_count + 1)
                    for k in range(missing_count):
                        idx = i + k
                        aligned[idx] = {
                            "word": new_word_tokens[idx],
                            "start": round(prev_time + k * step, 3),
                            "end": round(prev_time + (k + 1) * step, 3)
                        }
            return aligned

        # Update each clip in the plan
        for clip in clips_plan:
            clip_name = clip.get("clip_name")
            edited = next((c for c in edited_clips if c.get("clip_name") == clip_name), None)
            if not edited:
                continue
                
            clip["title"] = edited.get("title", clip.get("title", ""))
            clip["hook_phrase"] = edited.get("hook_phrase", clip.get("hook_phrase", ""))
            
            new_text = edited.get("transcript_text", "").strip()
            if new_text and new_text != clip.get("transcript_text", "").strip():
                clip["transcript_text"] = new_text
                start_time = clip["segments"][0]["start"] if clip.get("segments") else 0.0
                end_time = clip["segments"][0]["end"] if clip.get("segments") else 10.0
                words_subset = [w for w in original_words if w.end >= start_time and w.start <= end_time]
                clip["edited_words"] = align_words(new_text, words_subset)
                clip["edited_words_timebase"] = "absolute"
                
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(clips_plan, f, indent=2, ensure_ascii=False)
            
        from pipeline.local_clips_generator import _write_clip_info
        _write_clip_info(Path(job_dir), clips_plan)
        
        return jsonify({"success": True, "message": "Clips plan updated successfully."})


@app.route("/api/download/<job_id>/<clip_name>")
def api_download(job_id, clip_name):
    job_dir = _find_job_dir(job_id)
    file_path = _safe_under(os.path.join(job_dir, "clips"), os.path.basename(clip_name))
    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


@app.route("/api/preview/<job_id>/<clip_name>")
def api_preview(job_id, clip_name):
    job_dir = _find_job_dir(job_id)
    file_path = _safe_under(os.path.join(job_dir, "clips"), os.path.basename(clip_name))
    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, mimetype="video/mp4")


# ── Frame-accurate clip-ending adjuster ──────────────────────────────────────

_CLIP_NAME_PATTERN = r"[A-Za-z0-9_\-]+"


def _read_source_duration(job_dir: str) -> float:
    """Source video duration in seconds from meta.json (0.0 if unknown)."""
    meta_path = os.path.join(job_dir, "meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("duration") or 0.0)
    except Exception:
        return 0.0


def _invalidate_clip_artifacts(job_dir: str, clip_name: str) -> list:
    """Delete ONE clip's render artifacts so the next extract/caption pass
    rebuilds it: raw + render manifest, per-clip cache dirs, captioned outputs
    (all variants + filter scripts), SRT, ending previews. Hook intro files are
    start-side and stay. Other clips' files are never touched."""
    import glob as _glob
    import shutil as _shutil
    clips_dir = os.path.join(job_dir, "clips")
    removed = []
    for pattern in (
        f"{clip_name}_raw.mp4",
        f"{clip_name}_raw.render.json",
        f"{clip_name}.srt",
        f"{clip_name}_captioned*",
        f"_temp_{clip_name}_part_*",
        f"{clip_name}_endpreview_*.mp4",
    ):
        for path in _glob.glob(os.path.join(clips_dir, pattern)):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed.append(os.path.basename(path))
                except OSError:
                    pass
    for dpattern in (f"{clip_name}_raw_job", f"_temp_{clip_name}_part_*_job"):
        for path in _glob.glob(os.path.join(clips_dir, dpattern)):
            if os.path.isdir(path):
                _shutil.rmtree(path, ignore_errors=True)
                removed.append(os.path.basename(path) + "/")
    return removed


@app.route("/api/ending-preview/<job_id>/<clip_name>")
def api_ending_preview(job_id, clip_name):
    """Prepare (and cache) a short all-keyframe snippet of the SOURCE video
    around a clip's current END, so the dashboard's ending adjuster can scrub
    frame-by-frame both before AND beyond the current boundary."""
    import re as _re
    from pathlib import Path as _Path
    if not _re.fullmatch(_CLIP_NAME_PATTERN, clip_name or ""):
        return jsonify({"error": "Invalid clip name"}), 400

    job_dir = _find_job_dir(job_id)
    plan_path = os.path.join(job_dir, "clips_plan.json")
    if not os.path.exists(plan_path):
        return jsonify({"error": "clips_plan.json not found"}), 404
    with open(plan_path, "r", encoding="utf-8") as f:
        clips_plan = json.load(f)
    clip = next((c for c in clips_plan if c.get("clip_name") == clip_name), None)
    if not clip or not clip.get("segments"):
        return jsonify({"error": f"Clip not found in plan: {clip_name}"}), 404

    last_seg = clip["segments"][-1]
    current_end = float(last_seg.get("end", 0.0))
    last_start = float(last_seg.get("start", 0.0))

    from pipeline.extractor import _read_meta_video_path, _probe_duration
    from pipeline.trimmer import _probe_fps, extract_preview_snippet

    video_path = _read_meta_video_path(_Path(job_dir))
    if not video_path.exists():
        return jsonify({"error": f"Source video not found: {video_path.name}"}), 404

    source_duration = _read_source_duration(job_dir)
    if source_duration <= 0:
        source_duration = _probe_duration(video_path)

    pre = max(2.0, min(30.0, _to_float(request.args.get("pre"), 8.0)))
    post = max(2.0, min(30.0, _to_float(request.args.get("post"), 6.0)))
    snippet_start = max(0.0, current_end - pre)
    snippet_end = current_end + post
    if source_duration > 0:
        snippet_end = min(source_duration, snippet_end)
    if snippet_end <= snippet_start:
        return jsonify({"error": "Empty preview window"}), 400

    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    fname = f"{clip_name}_endpreview_{int(round(current_end * 1000))}.mp4"
    out_path = os.path.join(clips_dir, fname)

    # Drop snippets built for previous end values of this clip.
    import glob as _glob
    for stale in _glob.glob(os.path.join(clips_dir, f"{clip_name}_endpreview_*.mp4")):
        if os.path.basename(stale) != fname:
            try:
                os.remove(stale)
            except OSError:
                pass

    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        if not extract_preview_snippet(str(video_path), out_path, snippet_start, snippet_end - snippet_start):
            return jsonify({"error": "Failed to build preview snippet (see server log)"}), 500

    # Saved framing templates: when present the regen reuses them (no GUI, no
    # redundant re-snap) unless the user explicitly asks to re-run the tool.
    has_templates = any(
        os.path.exists(os.path.join(job_dir, rel)) for rel in (
            "manual_templates.json",
            os.path.join("templates", "manual_templates.json"),
            os.path.join("files", "manual_templates.json"),
        )
    )

    return jsonify({
        "url": f"/api/preview/{job_id}/{fname}",
        "fps": _probe_fps(str(video_path)),
        "snippet_start": round(snippet_start, 3),
        "snippet_end": round(snippet_end, 3),
        "current_end": round(current_end, 3),
        "min_end": round(last_start + 1.0, 3),
        "max_end": round(source_duration, 3) if source_duration > 0 else None,
        "segment_count": len(clip["segments"]),
        "total_duration": clip.get("total_duration"),
        "has_templates": has_templates,
    })


@app.route("/api/adjust-ending", methods=["POST"])
def api_adjust_ending():
    """Persist a frame-accurate new END time for one clip and (optionally)
    spawn a single-clip regeneration job: re-extract + re-caption ONLY the
    adjusted clip(s), reusing the original framing templates, caption style and
    music. The separate branding app is untouched — regenerated clips flow into
    it as usual.

    Body: {job_id, clip_name, new_end (source-video seconds), regenerate (bool),
           regen_clips (optional comma list — regenerate several pending clips at once)}
    """
    import re as _re
    from pathlib import Path as _Path

    data = _get_request_data()
    job_id = str(data.get("job_id", "")).strip()
    clip_name = str(data.get("clip_name", "")).strip()
    new_end = _to_float(data.get("new_end"), -1.0)
    regenerate = _to_bool(data.get("regenerate"), False)
    # Default False = reuse saved templates when they exist (no redundant
    # re-snap). True = user explicitly wants the template tool to run again.
    rerun_templates = _to_bool(data.get("rerun_templates"), False)

    if not job_id or not _re.fullmatch(_CLIP_NAME_PATTERN, clip_name or ""):
        return jsonify({"error": "job_id and a valid clip_name are required"}), 400
    if new_end <= 0:
        return jsonify({"error": "new_end must be a positive number of seconds"}), 400

    job_dir = _find_job_dir(job_id)
    plan_path = os.path.join(job_dir, "clips_plan.json")
    if not os.path.exists(plan_path):
        return jsonify({"error": "clips_plan.json not found"}), 404
    with open(plan_path, "r", encoding="utf-8") as f:
        clips_plan = json.load(f)
    clip = next((c for c in clips_plan if c.get("clip_name") == clip_name), None)
    if not clip or not clip.get("segments"):
        return jsonify({"error": f"Clip not found in plan: {clip_name}"}), 404

    last_seg = clip["segments"][-1]
    last_start = float(last_seg.get("start", 0.0))
    old_end = float(last_seg.get("end", 0.0))
    min_end = last_start + 1.0
    source_duration = _read_source_duration(job_dir)

    if new_end < min_end:
        return jsonify({"error": f"new_end {new_end:.3f}s is too early — the last segment must keep at least 1s (min {min_end:.3f}s)"}), 400
    if source_duration > 0 and new_end > source_duration:
        return jsonify({"error": f"new_end {new_end:.3f}s is past the source video end ({source_duration:.3f}s)"}), 400

    changed = abs(new_end - old_end) >= 0.0005
    removed = []
    if changed:
        last_seg["end"] = round(new_end, 3)
        last_seg["duration"] = round(new_end - last_start, 3)
        clip["total_duration"] = round(sum(
            max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
            for s in clip["segments"]
        ), 3)
        clip["end_adjusted"] = True
        # edited_words were aligned to the OLD clip window; dropping them makes
        # captions fall back to slicing transcript.json, which is correct for
        # the new boundary.
        clip.pop("edited_words", None)
        clip.pop("edited_words_timebase", None)

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(clips_plan, f, indent=2, ensure_ascii=False)
        from pipeline.local_clips_generator import _write_clip_info
        _write_clip_info(_Path(job_dir), clips_plan)

        removed = _invalidate_clip_artifacts(job_dir, clip_name)

    response = {
        "success": True,
        "clip_name": clip_name,
        "old_end": round(old_end, 3),
        "new_end": round(float(last_seg.get("end", 0.0)), 3),
        "total_duration": clip.get("total_duration"),
        "changed": changed,
        "invalidated": removed,
    }
    if not regenerate:
        return jsonify(response)

    # ── Spawn the regeneration job (mirrors the /api/reburn pattern, but from
    # the extract stage so the new boundary is re-rendered) ──────────────────
    regen_list = _parse_clip_selection(str(data.get("regen_clips") or clip_name))

    with JOBS_LOCK:
        orig_settings = dict(JOBS.get(job_id, {}).get("settings", {}) or {})
    if not orig_settings:
        orig_settings = _load_job_settings(job_dir)

    settings = dict(orig_settings)
    settings.update({
        "reburn_clips": regen_list,           # captioner: only these clips
        "skip_copywriting": True,             # metadata already exists
        "pause_before_captioning": False,     # regen must run straight through
        "tts_hook_mode": "keep",              # reuse existing hook assets
        # reuse saved framing templates (no GUI) unless the user asked to
        # re-run the template tool; with no saved file the tool runs anyway.
        "reuse_manual_templates": not rerun_templates,
        "force_template_resnap": rerun_templates,
    })

    new_job_id = f"regen_{int(time.time())}_{uuid.uuid4().hex[:4]}"
    try:
        with open(os.path.join(job_dir, ".job_id"), "a", encoding="utf-8") as f:
            f.write(f"\n{new_job_id}")
    except Exception:
        pass

    with JOBS_LOCK:
        JOBS[new_job_id] = {
            "id":           new_job_id,
            "url":          f"regen://{os.path.basename(job_dir)}",
            "status":       "starting",
            "stage":        "",
            "progress":     0,
            "message":      f"Regenerating {regen_list} with adjusted ending…",
            "created_at":   time.time(),
            "updated_at":   time.time(),
            "settings":     settings,
            "events":       [],
            "output_dir":   job_dir,
            "display_name": os.path.basename(job_dir),
            "source_job_id": job_id,
        }

    progress_callback = _make_progress_callback(new_job_id)

    def run_regen():
        try:
            run_pipeline_from(job_dir, "extract", settings, progress_callback)
        except Exception as exc:
            with JOBS_LOCK:
                if new_job_id in JOBS:
                    JOBS[new_job_id]["status"] = "error"
                    JOBS[new_job_id]["message"] = str(exc)
                    JOBS[new_job_id]["updated_at"] = time.time()
                    JOBS[new_job_id]["events"].append({
                        "stage": "error", "progress": -1,
                        "message": str(exc), "timestamp": time.time(),
                    })

    threading.Thread(target=run_regen, daemon=True).start()
    response["regen_job_id"] = new_job_id
    return jsonify(response)


@app.route("/api/blog/<job_id>")
def api_blog(job_id):
    job_dir = _find_job_dir(job_id)
    fmt = request.args.get("format", "md")
    if fmt == "txt":
        blog_path = os.path.join(job_dir, "blog_post.txt")
    else:
        blog_path = None
        if os.path.exists(job_dir):
            for f in os.listdir(job_dir):
                if f.endswith(".md") and f != "README.md":
                    blog_path = os.path.join(job_dir, f)
                    break
        if not blog_path:
            blog_path = os.path.join(job_dir, "blog_post.txt")
    if not os.path.exists(blog_path):
        return jsonify({"error": "Blog post not found"}), 404
    return send_file(blog_path, as_attachment=True)


@app.route("/api/logs/<job_id>")
def api_logs(job_id):
    job_dir = _find_job_dir(job_id)
    lines = int(request.args.get("lines", 100))
    log_text = get_log_contents(job_dir, lines)
    return jsonify({"log": log_text})


# ═══════════════════════════════════════════════════════════════════════════════
# Cancel job
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    """Set the cancel flag on a running job. The pipeline thread checks this."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        job["cancel"] = True
        job["status"] = "error"
        job["message"] = "Cancelled by user"
        job["updated_at"] = time.time()
    return jsonify({"status": "cancelled", "job_id": job_id})


# ═══════════════════════════════════════════════════════════════════════════════
# Trimming Clips
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/trim-clip", methods=["POST"])
def api_trim_clip():
    data = _get_request_data()
    job_id = data.get("job_id", "").strip()
    filename = data.get("filename", "").strip()
    keep_segments = data.get("keep_segments", [])
    transition = data.get("transition", "sharp")
    
    if not job_id or not filename or not keep_segments:
        return jsonify({"error": "Missing required fields: job_id, filename, or keep_segments"}), 400
        
    job_dir = _find_job_dir(job_id)
    if not os.path.isdir(job_dir):
        return jsonify({"error": f"Job folder not found: {job_dir}"}), 404
        
    clips_dir = os.path.join(job_dir, "clips")
    video_path = os.path.join(clips_dir, filename)
    if not os.path.exists(video_path):
        # Fallback to job_dir if clips_dir doesn't exist
        video_path = os.path.join(job_dir, filename)
        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {filename}"}), 404
            
    try:
        from pipeline.trimmer import trim_video
        out_path = trim_video(video_path, keep_segments, transition=transition)
        return jsonify({"success": True, "trimmed_filename": os.path.basename(out_path)})
    except Exception as e:
        logger.error(f"Trim error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Background job runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_job(job_id: str, url: str, settings: dict):
    """Run the pipeline in a background thread."""

    progress_callback = _make_progress_callback(job_id)

    def rename_callback(new_output_dir: str):
        folder_name = os.path.basename(new_output_dir)
        with JOBS_LOCK:
            if job_id in JOBS:
                old_output_dir = JOBS[job_id].get("output_dir")
                JOBS[job_id]["output_dir"] = new_output_dir
                JOBS[job_id]["display_name"] = folder_name
                JOBS[job_id]["updated_at"] = time.time()
                if old_output_dir:
                    rename_music_job_dir(old_output_dir, new_output_dir)

    try:
        def _cancelled():
            with JOBS_LOCK:
                return bool(JOBS.get(job_id, {}).get("cancel"))
        run_pipeline(job_id, url, settings, progress_callback, rename_callback, cancel_check=_cancelled)
    except Exception as exc:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["message"] = str(exc)
                JOBS[job_id]["updated_at"] = time.time()
                JOBS[job_id]["events"].append({
                    "stage":     "error",
                    "progress": -1,
                    "message":   str(exc),
                    "timestamp": time.time(),
                })
    finally:
        with ACTIVE_URLS_LOCK:
            if url in ACTIVE_URLS and ACTIVE_URLS[url] == job_id:
                del ACTIVE_URLS[url]


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # ── Signal and Console Keyboard Handlers for Instant Force-Exit ──
    def setup_force_exit_handlers():
        import signal
        import subprocess
        
        def force_exit_handler(sig, frame):
            sys.stderr.write("\n[!] Force exit requested. Terminating all processes immediately...\n")
            sys.stderr.flush()
            try:
                if os.name == "nt":
                    # Forcefully kill the entire process tree (python + all child processes like ffmpeg)
                    subprocess.Popen(
                        f"taskkill /F /T /PID {os.getpid()}",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    )
                    time.sleep(0.4)
            except Exception:
                pass
            os._exit(0)

        # Register standard console signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, force_exit_handler)
            except Exception:
                pass

        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, force_exit_handler)
            except Exception:
                pass

        # Windows-specific non-blocking keyboard listener thread
        if os.name == "nt":
            try:
                import msvcrt
                
                def win_keyboard_listener():
                    time.sleep(1.0)
                    sys.stderr.write("[*] Key Mapping Active: Press 'q' or 'Esc' in this console to stop the server instantly.\n")
                    sys.stderr.flush()
                    while True:
                        try:
                            if msvcrt.kbhit():
                                ch = msvcrt.getch()
                                # ESC is b'\x1b', 'q' is b'q', 'Q' is b'Q'
                                if ch in (b'\x1b', b'q', b'Q'):
                                    sys.stderr.write(f"\n[!] Key press {ch} detected. Triggering force exit...\n")
                                    sys.stderr.flush()
                                    force_exit_handler(None, None)
                        except Exception:
                            pass
                        time.sleep(0.1)

                t = threading.Thread(target=win_keyboard_listener, daemon=True)
                t.start()
            except ImportError:
                pass

    setup_force_exit_handlers()

    print(
        f"\n{'='*50}\n"
        f"  Video Clipper -> http://localhost:{config.FLASK_PORT}\n"
        f"  Source: {os.path.abspath(os.getcwd())}\n"
        f"  Python: {sys.executable}\n"
        f"  Analyzer: {os.path.abspath(os.path.join('pipeline', 'analyzer.py'))}\n"
        f"{'='*50}\n"
    )
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
    )

