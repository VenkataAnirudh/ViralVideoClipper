"""
API Provider — LLM transport layer
==================================
Everything related to *making* AI calls lives here, split out of analyzer.py so
that module holds only the analysis workflow.

Responsibilities:
  • Provider clients (NVIDIA NIM, OpenAI, Claude, Gemini) and streaming readers.
  • Provider/model fallback plan + provider-name normalization.
  • Per-provider RPM limiting (NVIDIA's ~40 rpm ceiling; others run sequentially).
  • Error classification (auth / rate-limit) and retry-sleep policy.
  • API-response caching and failure logging.
  • JSON-mode parsing (_extract_json).
  • The task runners — including the best-of runner: each NVIDIA work-unit fires
    NVIDIA_RACE_COUNT concurrent calls, waits for ALL of them, and keeps the BEST
    valid response (highest caller-supplied `score`, e.g. most candidates). Domain
    validation AND scoring are injected by the caller via callbacks, so this module
    never imports analyzer (no circular import).

analyzer.py re-exports the public names below so existing callers that do
`from pipeline.analyzer import call_llm` (etc.) keep working unchanged.
"""

import json
import os
import re
import time
import logging
import threading
import collections
import datetime
import config
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════════════════
# CONCURRENCY PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

class _NoOpLock:
    """No-op context manager. Real concurrency control lives in the racing
    runner + RPM limiter. Set AI_FORCE_SERIAL=true to restore one-call-at-a-time."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def acquire(self, *a, **kw): return True
    def release(self): pass


if os.getenv("AI_FORCE_SERIAL", "false").lower() == "true":
    AI_ANALYSIS_CALL_LOCK = threading.Lock()
else:
    AI_ANALYSIS_CALL_LOCK = _NoOpLock()


class _RpmLimiter:
    """Rolling-60s request-per-minute limiter. Every gated call acquires a slot
    before firing; acquire() blocks until the rolling window has room."""

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self._times: "collections.deque[float]" = collections.deque()
        self._lock = threading.Lock()

    def set_rpm(self, rpm: int) -> None:
        self.rpm = max(1, int(rpm))

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.02
            time.sleep(min(max(wait, 0.01), 5.0))


# NVIDIA target rpm, clamped to the hard ceiling so we never exceed the provider.
_NVIDIA_RPM = min(
    int(getattr(config, "NVIDIA_RPM", 35)),
    int(getattr(config, "NVIDIA_RPM_CEILING", 40)),
)
_NVIDIA_RATE = _RpmLimiter(_NVIDIA_RPM)

# Per-provider limiter registry. NVIDIA is the only throttled provider today;
# others run sequentially as fallbacks, so they get a generous limiter that
# effectively never blocks. Add real per-provider rpms here when needed.
_LIMITERS: "dict[str, _RpmLimiter]" = {"nvidia": _NVIDIA_RATE}


def _limiter_for(provider: str) -> _RpmLimiter:
    prov = (provider or "").lower()
    lim = _LIMITERS.get(prov)
    if lim is None:
        lim = _RpmLimiter(int(getattr(config, f"{prov.upper()}_RPM", 10_000)))
        _LIMITERS[prov] = lim
    return lim


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION & RETRY POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def _is_openai_reasoning_model(model: str) -> bool:
    """OpenAI reasoning families that require `max_completion_tokens` (not
    `max_tokens`) and reject a non-default `temperature`: the o-series (o1/o3/o4…)
    AND the GPT-5 line (gpt-5, gpt-5-nano/mini, gpt-5.x…). Treating a gpt-5 model
    as a plain chat model sends `temperature`+`max_tokens` and the call 400s."""
    m = (model or "").lower().strip()
    return m.startswith("o") or m.startswith("gpt-5")


def _is_model_access_error(exc: Exception) -> bool:
    """A 403/404 scoped to ONE model — the project/key lacks access to it, or the
    model id is unknown/retired. This is NOT a credential failure: the key is
    valid, it just can't use this specific model. Must advance to the next model
    in the fallback chain instead of aborting the whole plan.

    Example (OpenAI): HTTP 403 {"code": "model_not_found", "message": "Project ...
    does not have access to model `gpt-4.1`"}."""
    s = str(exc).lower()
    return (
        "model_not_found" in s
        or "does not have access to model" in s
        or "no access to model" in s
        or "the model" in s and "does not exist" in s
    )


def _is_nvidia_auth_error(exc: Exception) -> bool:
    """True credential failures (bad/missing key, account forbidden) are fatal —
    never retry them (a bad key would otherwise spin forever under immediate-retry).

    A 403 scoped to a single model (project lacks access to THAT model) is NOT a
    credential failure — it must fall through to the next model, so it is excluded
    here. (Despite the name, this gates the fatal-abort path for every provider.)"""
    if _is_model_access_error(exc):
        return False
    s = str(exc).lower()
    return (
        "http 401" in s or "http 403" in s
        or "unauthorized" in s or "authentication failed" in s
        or "forbidden" in s
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "http 429" in s or "too many requests" in s or "rate limit" in s


def _retry_sleep_seconds(prov: str, exc: Exception | None = None) -> float:
    """NVIDIA retries immediately (NVIDIA_RETRY_SLEEP, default 0); other providers
    use AI_RETRY_SLEEP_SECONDS. EXCEPTION: HTTP 429 always backs off
    (NVIDIA_RATE_LIMIT_BACKOFF) since immediate retry worsens rate limiting."""
    if exc is not None and _is_rate_limit_error(exc):
        return float(getattr(config, "NVIDIA_RATE_LIMIT_BACKOFF", 6))
    if prov == "nvidia":
        return float(getattr(config, "NVIDIA_RETRY_SLEEP", 0.0))
    return float(getattr(config, "AI_RETRY_SLEEP_SECONDS", 10))


# ═══════════════════════════════════════════════════════════════════════════════
# API CACHING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_cached_api_response(job_dir: str, cache_key: str, logger) -> dict | str | None:
    """Read a cached API response (JSON or text) from the job's api_cache directory."""
    cache_dir = os.path.join(job_dir, "api_cache")
    if not os.path.exists(cache_dir):
        return None

    json_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                logger.info(f"Using cached API response for {cache_key} (JSON)")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cached JSON for {cache_key}: {e}")

    txt_path = os.path.join(cache_dir, f"{cache_key}.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                logger.info(f"Using cached API response for {cache_key} (text)")
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read cached text for {cache_key}: {e}")

    return None


def save_cached_api_response(job_dir: str, cache_key: str, data, logger) -> None:
    """Save API response (JSON dict or raw text string) to the job's api_cache directory."""
    cache_dir = os.path.join(job_dir, "api_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if isinstance(data, (dict, list)):
        json_path = os.path.join(cache_dir, f"{cache_key}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cached API response for {cache_key} (JSON)")
        except Exception as e:
            logger.warning(f"Failed to save cached JSON for {cache_key}: {e}")
    else:
        txt_path = os.path.join(cache_dir, f"{cache_key}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(str(data))
            logger.info(f"Saved cached API response for {cache_key} (text)")
        except Exception as e:
            logger.warning(f"Failed to save cached text for {cache_key}: {e}")


def delete_cached_api_response(job_dir: str, cache_key: str, logger) -> None:
    """Delete a cached API response from the job's api_cache directory."""
    cache_dir = os.path.join(job_dir, "api_cache")
    json_path = os.path.join(cache_dir, f"{cache_key}.json")
    txt_path = os.path.join(cache_dir, f"{cache_key}.txt")
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
            logger.info(f"Deleted cached JSON for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to delete cached JSON for {cache_key}: {e}")
    if os.path.exists(txt_path):
        try:
            os.remove(txt_path)
            logger.info(f"Deleted cached text for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to delete cached text for {cache_key}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _save_bad_ai_response(job_dir, provider, attempt, raw_response, logger) -> None:
    """Persist malformed model output so parser failures are debuggable.

    Dumps go into job_dir/logs/ rather than the job root — Qwen truncation
    bursts produced 80+ analysis_bad_response_*.txt files that polluted the
    user-facing job directory.
    """
    if not raw_response:
        return
    safe_provider = re.sub(r"[^A-Za-z0-9_-]+", "_", provider or "ai")
    logs_dir = os.path.join(job_dir, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except OSError as exc:
        logger.debug(f"Could not create logs/ subdir for bad AI response: {exc}")
        logs_dir = job_dir  # fall back to root rather than losing the dump
    path = os.path.join(logs_dir, f"analysis_bad_response_{safe_provider}_{attempt}.txt")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(raw_response)
        logger.debug(f"Saved malformed AI response for debugging: {path}")
    except OSError as exc:
        logger.debug(f"Could not save malformed AI response: {exc}")


def _save_ai_failure(job_dir, task_name, provider, model, attempt, error_text, raw_response, logger) -> None:
    safe_task = re.sub(r"[^A-Za-z0-9_-]+", "_", task_name or "analysis")
    path = os.path.join(job_dir, "analysis_failures.jsonl")
    record = {
        "task": task_name,
        "provider": provider,
        "model": model,
        "attempt": attempt,
        "error": re.sub(r"\s+", " ", str(error_text or "")).strip()[:1200],
        "response_preview": re.sub(r"\s+", " ", str(raw_response or "")).strip()[:1200],
        "timestamp": time.time(),
    }
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.debug(f"Could not save AI failure details for {safe_task}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER NORMALIZATION & FALLBACK PLAN
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_provider(provider: str) -> str:
    provider = str(provider or config.DEFAULT_AI_PROVIDER).strip().lower()
    aliases = {
        "anthropic": "claude",
        "google": "gemini",
        "openai": "openai",
        "chatgpt": "openai",
        "nim": "nvidia",
        "nvidia_nim": "nvidia",
        # Legacy providers removed — route old saved jobs to NVIDIA so resumes
        # of pre-existing job folders don't crash on an unknown provider.
        "featherless": "nvidia",
        "featherless.ai": "nvidia",
        "featherless_ai": "nvidia",
        "openrouter": "nvidia",
    }
    return aliases.get(provider, provider)


def _nvidia_keys_in_priority() -> list[str]:
    """NVIDIA keys in configured priority order (KEY_2 first by default — it holds
    mistral access; the bare key 403s on it). De-duplicated, empties dropped, with
    any configured-but-unlisted key appended as a safety net."""
    order = getattr(config, "NVIDIA_KEY_ORDER", None) or [
        "NVIDIA_API_KEY_2", "NVIDIA_API_KEY_1", "NVIDIA_API_KEY",
    ]
    keys: list[str] = []
    for name in order:
        k = (getattr(config, name, "") or "").strip()
        if k and k not in keys:
            keys.append(k)
    for name in ("NVIDIA_API_KEY_2", "NVIDIA_API_KEY_1", "NVIDIA_API_KEY"):
        k = (getattr(config, name, "") or "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def _analysis_provider_plan(provider: str, model: str, api_key: str, task_type: str = "complex") -> list[tuple[str, str, str]]:
    """Return the strict AI provider and model fallback chain for analysis."""
    provider = _normalize_provider(provider)

    if provider == "openai":
        key = api_key or config.OPENAI_API
        if task_type == "discovery":
            model_order = list(getattr(config, "OPENAI_DISCOVERY_FALLBACK_MODELS", []))
        else:
            model_order = list(getattr(config, "OPENAI_ANALYSIS_FALLBACK_MODELS", []))
        if not model_order:
            model_order = list(config.OPENAI_MODELS)
    elif provider == "nvidia":
        key = api_key or (_nvidia_keys_in_priority()[0] if _nvidia_keys_in_priority() else config.NVIDIA_API_KEY)
        if task_type == "discovery":
            model_order = list(getattr(config, "NVIDIA_DISCOVERY_FALLBACK_MODELS", []))
        else:
            model_order = list(getattr(config, "NVIDIA_ANALYSIS_FALLBACK_MODELS", []))
        if not model_order:
            model_order = list(config.NVIDIA_MODELS)
    else:
        if provider == "claude":
            key = api_key or config.ANTHROPIC_API_KEY
        elif provider == "gemini":
            key = api_key or config.GOOGLE_API_KEY
        else:
            key = api_key
        model_order = []

    # Prepend user-selected model if provided
    if model:
        cleaned_model = model.strip()
        if cleaned_model in model_order:
            model_order.remove(cleaned_model)
        model_order = [cleaned_model] + model_order

    plan = [(provider, fallback_model, key) for fallback_model in model_order]

    # NVIDIA multi-key failover: after the primary key has burned all its
    # retries on a given model, retry the SAME model with each fallback key
    # in order before degrading to the next model. This preserves model
    # quality when a key just hit its daily quota (common 403 cause).
    if provider == "nvidia":
        # Rotate the SAME model across the remaining keys (in priority order)
        # before degrading to the next model — so a 429/403 on one key fails over
        # to a working key on the same (preferred) model first.
        fallback_keys = [k for k in _nvidia_keys_in_priority() if k and k != key]
        if fallback_keys:
            interleaved = []
            for entry in plan:
                interleaved.append(entry)
                for fk in fallback_keys:
                    interleaved.append(("nvidia", entry[1], fk))
            plan = interleaved

    if not plan and model:
        plan = [(provider, model.strip(), key)]

    if getattr(config, "AI_CROSS_PROVIDER_FALLBACK", True):
        if config.NVIDIA_API_KEY:
            for nv_model in config.NVIDIA_MODELS:
                if ("nvidia", nv_model, config.NVIDIA_API_KEY) not in plan:
                    plan.append(("nvidia", nv_model, config.NVIDIA_API_KEY))
        if config.OPENAI_API:
            for oa_model in config.OPENAI_MODELS:
                if ("openai", oa_model, config.OPENAI_API) not in plan:
                    plan.append(("openai", oa_model, config.OPENAI_API))
        if config.GOOGLE_API_KEY:
            for gem_model in config.GEMINI_MODELS:
                if ("gemini", gem_model, config.GOOGLE_API_KEY) not in plan:
                    plan.append(("gemini", gem_model, config.GOOGLE_API_KEY))
        if config.ANTHROPIC_API_KEY:
            for cl_model in config.CLAUDE_MODELS:
                if ("claude", cl_model, config.ANTHROPIC_API_KEY) not in plan:
                    plan.append(("claude", cl_model, config.ANTHROPIC_API_KEY))

    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _call_claude(prompt: str, model: str, api_key: str, logger: logging.Logger,
                 temperature: float | None = None) -> str:
    """Call Claude API and return the response text."""
    import anthropic

    if not api_key:
        api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("No Anthropic API key provided")
    if not model:
        model = config.DEFAULT_CLAUDE_MODEL

    # Claude's temperature ceiling is 1.0 — clamp so a hot creative temp (e.g. 1.3)
    # passed for copy generation doesn't get rejected.
    temp = config.AI_TEMPERATURE if temperature is None else float(temperature)
    temp = max(0.0, min(1.0, temp))

    client = anthropic.Anthropic(api_key=api_key)
    logger.debug(f"Calling Claude: model={model}, temperature={temp}")
    response = client.messages.create(
        model=model,
        max_tokens=int(getattr(config, "AI_MAX_TOKENS_ANALYSIS", 65536)),
        temperature=temp,
        system="Return valid JSON only. Do not include markdown, commentary, or text outside the JSON object.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    logger.debug(f"Claude response: {len(text)} chars, tokens: {response.usage}")
    return text


def _call_gemini(prompt: str, model: str, api_key: str, logger: logging.Logger,
                 temperature: float | None = None) -> str:
    """Call Gemini API and return the response text.

    Thinking models (gemini-2.5+) embed chain-of-thought in a 'thought_process'
    field that can consume thousands of tokens and truncate the candidates JSON.
    Fix: thinking_budget=0 disables visible thinking output.
    """
    from google import genai

    if not api_key:
        api_key = config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("No Google API key provided")
    if not model:
        model = config.DEFAULT_GEMINI_MODEL

    client = genai.Client(api_key=api_key)
    logger.debug(f"Calling Gemini: model={model}, max_output_tokens=unlimited")

    config_kwargs = {
        "temperature": config.AI_TEMPERATURE if temperature is None else float(temperature),
        "response_mime_type": "application/json",
    }
    try:
        config_kwargs["thinking_config"] = genai.types.ThinkingConfig(thinking_budget=0)
        generation_config = genai.types.GenerateContentConfig(**config_kwargs)
        logger.debug("Gemini thinking output disabled (thinking_budget=0)")
    except (TypeError, AttributeError):
        config_kwargs.pop("thinking_config", None)
        try:
            generation_config = genai.types.GenerateContentConfig(**config_kwargs)
        except (TypeError, ValueError):
            config_kwargs.pop("response_mime_type", None)
            generation_config = genai.types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=generation_config,
    )
    text = response.text or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty analysis response")
    logger.debug(f"Gemini response: {len(text)} chars")
    return text


def _call_nvidia(
    prompt: str,
    model: str,
    api_key: str,
    logger: logging.Logger,
    enable_reasoning: bool = False,
    clip_count: int = 10,
    response_format: bool = True,
    temperature: float | None = None,
) -> str:
    """Call NVIDIA NIM (integrate.api.nvidia.com, OpenAI-compatible) and return text.

    Per-family reasoning parameters are kept strictly separate:
      • Nemotron-3 (ultra/super): chat_template_kwargs.enable_thinking + reasoning_budget
      • DeepSeek-V4 (pro/flash):   chat_template_kwargs.{enable_thinking, thinking}
        DeepSeek NIM hangs if chat_template_kwargs is omitted, so it is always sent.
    Responses are streamed; reasoning_content deltas are discarded — only the final
    JSON content is collected. No read timeout by design (connect timeout still
    applies): a slow window is allowed to run as long as it needs so every clip
    surfaces. The best-of runner waits for all racers, so none are left dangling.
    """
    import requests

    if not api_key:
        api_key = config.NVIDIA_API_KEY
    if not api_key:
        raise ValueError("No NVIDIA API key provided")
    if not model:
        model = config.DEFAULT_NVIDIA_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Connection": "close",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return valid JSON only. Do not include markdown, commentary, or text outside the JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(
            temperature if temperature is not None
            else getattr(config, "NVIDIA_TEMPERATURE", 0.7)
        ),
        "top_p": float(getattr(config, "NVIDIA_TOP_P", 0.95)),
        "stream": True,
    }
    if not bool(getattr(config, "NVIDIA_NO_MAX_TOKENS", True)):
        payload["max_tokens"] = int(getattr(config, "AI_MAX_TOKENS_ANALYSIS", 65536))
    if response_format:
        payload["response_format"] = {"type": "json_object"}

    # ── Per-family reasoning controls (no mix-ups) ──────────────────────────
    # Effort = "low" is the anti-clog default — at higher levels the streaming
    # reader has observed 180k–200k chunks on a single call as the model spun
    # in an endless thinking loop. Override via NVIDIA_REASONING_EFFORT.
    reasoning_effort = str(getattr(config, "NVIDIA_REASONING_EFFORT", "low")).lower()
    model_l = str(model).lower()
    if model_l.startswith(getattr(config, "NVIDIA_NEMOTRON_PREFIX", "nvidia/nemotron")):
        payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_reasoning)}
        if enable_reasoning:
            # Both a hard token budget AND the string-effort knob — Nemotron
            # accepts reasoning_effort too, and combining them gives the
            # tightest reasoning cap.
            payload["reasoning_budget"] = int(getattr(config, "NVIDIA_REASONING_BUDGET", 4096))
            payload["chat_template_kwargs"]["reasoning_effort"] = reasoning_effort
    elif model_l.startswith(getattr(config, "NVIDIA_DEEPSEEK_PREFIX", "deepseek-ai/")):
        payload["chat_template_kwargs"] = {
            "enable_thinking": bool(enable_reasoning),
            "thinking": bool(enable_reasoning),
        }
        if enable_reasoning:
            payload["chat_template_kwargs"]["reasoning_effort"] = reasoning_effort

    logger.debug(f"Calling NVIDIA: model={model}, reasoning={bool(enable_reasoning)}")

    connect_to = float(getattr(config, "NVIDIA_CONNECT_TIMEOUT", 15))
    read_to = None if bool(getattr(config, "NVIDIA_NO_READ_TIMEOUT", True)) else config.AI_ANALYSIS_TIMEOUT
    timeout = (connect_to, read_to)

    _limiter_for("nvidia").acquire()
    session = requests.Session()
    response = None
    try:
        response = session.post(
            config.NVIDIA_CHAT_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
            stream=True,
        )

        if response.status_code >= 400 and _looks_like_response_format_rejection(response.text):
            logger.warning("NVIDIA model rejected response_format; retrying without JSON mode")
            response.close()
            payload.pop("response_format", None)
            response = session.post(
                config.NVIDIA_CHAT_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            )

        if response.status_code >= 400:
            raise RuntimeError(_http_error_message("NVIDIA analysis", response))

        text = _read_streaming_chat_response(response, "NVIDIA", logger)
        if not text.strip():
            raise RuntimeError("NVIDIA returned an empty analysis response")
        logger.debug(f"NVIDIA response: {len(text)} chars")
        return text
    finally:
        if response is not None:
            response.close()
        session.close()


def _call_openai(
    prompt: str,
    model: str,
    api_key: str,
    logger: logging.Logger,
    enable_reasoning: bool = False,
    clip_count: int = 10,
    response_format: bool = True,
    temperature: float | None = None,
) -> str:
    """Call OpenAI Chat Completions API and return the response text."""
    import requests

    if not api_key:
        api_key = config.OPENAI_API
    if not api_key:
        raise ValueError("No OpenAI API key provided")
    if not model:
        model = config.DEFAULT_OPENAI_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    is_reasoning_model = _is_openai_reasoning_model(model)
    sys_role = "developer" if is_reasoning_model else "system"
    sys_content = "Return valid JSON only. Do not include markdown, commentary, or text outside the JSON object."

    payload = {
        "model": model,
        "messages": [
            {"role": sys_role, "content": sys_content},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    if is_reasoning_model:
        payload["max_completion_tokens"] = int(getattr(config, "AI_MAX_TOKENS_ANALYSIS", 65536))
        if enable_reasoning:
            payload["reasoning_effort"] = str(getattr(config, "OPENAI_REASONING_EFFORT", "medium")).lower()
    else:
        payload["temperature"] = config.AI_TEMPERATURE if temperature is None else float(temperature)
        payload["max_tokens"] = int(getattr(config, "AI_MAX_TOKENS_ANALYSIS", 65536))

    if response_format:
        if model != "o1-mini":
            payload["response_format"] = {"type": "json_object"}

    logger.debug(f"Calling OpenAI: model={model}, url={config.OPENAI_CHAT_API_URL}")

    _limiter_for("openai").acquire()
    session = requests.Session()
    response = None
    try:
        response = session.post(
            config.OPENAI_CHAT_API_URL,
            headers=headers,
            json=payload,
            timeout=config.AI_ANALYSIS_TIMEOUT,
        )
        if response.status_code >= 400:
            raise RuntimeError(_http_error_message("OpenAI analysis", response))

        text = _read_chat_response(response, "OpenAI", logger)
        if not text.strip():
            raise RuntimeError("OpenAI returned an empty analysis response")
        logger.debug(f"OpenAI response: {len(text)} chars")
        return text
    finally:
        if response is not None:
            response.close()
        session.close()


def _dispatch_call(
    prov: str, mdl: str, key: str, prompt: str, logger: logging.Logger,
    *, enable_reasoning: bool, clip_count: int, response_format: bool,
    temperature: float | None = None,
) -> str:
    """Route one call to the right provider client (under the serial-mode lock)."""
    if prov == "nvidia":
        with AI_ANALYSIS_CALL_LOCK:
            return _call_nvidia(prompt, mdl, key, logger,
                                enable_reasoning=enable_reasoning,
                                clip_count=clip_count, response_format=response_format,
                                temperature=temperature)
    if prov == "openai":
        with AI_ANALYSIS_CALL_LOCK:
            return _call_openai(prompt, mdl, key, logger,
                                enable_reasoning=enable_reasoning,
                                clip_count=clip_count, response_format=response_format,
                                temperature=temperature)
    if prov == "claude":
        with AI_ANALYSIS_CALL_LOCK:
            return _call_claude(prompt, mdl, key, logger, temperature=temperature)
    if prov == "gemini":
        with AI_ANALYSIS_CALL_LOCK:
            return _call_gemini(prompt, mdl, key, logger, temperature=temperature)
    raise ValueError(f"Unsupported analysis provider: {prov}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE READERS & JSON EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _read_chat_response(response, provider_label: str, logger: logging.Logger) -> str:
    try:
        data = response.json()
    except ValueError:
        return _read_streaming_chat_response(response, provider_label, logger)

    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        if isinstance(text, list):
            parts = []
            for item in text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            text = "".join(parts)
        if str(text).strip():
            return str(text)

    output = data.get("output") if isinstance(data, dict) else None
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        if "".join(parts).strip():
            return "".join(parts)

    logger.debug(f"{provider_label} returned chat JSON without message content")
    return ""


def _read_streaming_chat_response(response, provider_label: str, logger: logging.Logger) -> str:
    full_text = []
    non_sse_lines = []
    chunk_count = 0
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8").strip()
        if line_str == "data: [DONE]":
            continue
        if not line_str.startswith("data: "):
            non_sse_lines.append(line_str)
            continue
        try:
            data = json.loads(line_str[6:])
            # NVIDIA/OpenAI streams emit chunks with EMPTY choices (usage/keepalive
            # /final). Guard against choices==[] which would otherwise raise
            # IndexError and, under retry-forever, hang the whole run.
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            content = delta.get("content") or ""
            if content:
                full_text.append(content)
            chunk_count += 1
            if chunk_count % 50 == 0:
                logger.debug(f"... receiving {provider_label} AI response (chunk {chunk_count}) ...")
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            continue
    if full_text:
        return "".join(full_text)
    if non_sse_lines:
        raw = "\n".join(non_sse_lines)
        try:
            data = json.loads(raw)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            logger.debug(f"{provider_label} returned non-SSE data that was not chat JSON")
    return ""


def _looks_like_response_format_rejection(body: str) -> bool:
    body_l = (body or "").lower()
    return (
        "response_format" in body_l
        or "json mode" in body_l
        or "structured output" in body_l
        or "structured_outputs" in body_l
    )


def _looks_like_reasoning_rejection(body: str) -> bool:
    body_l = (body or "").lower()
    return "reasoning" in body_l and any(
        phrase in body_l
        for phrase in ("unsupported", "not supported", "invalid", "unrecognized", "unknown")
    )


def _http_error_message(label: str, response) -> str:
    body = ""
    try:
        body = response.text or ""
    except Exception:
        body = ""
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > 900:
        body = body[:900] + "..."
    return f"{label} HTTP {response.status_code}: {body or response.reason}"


def _extract_json(text: str, logger: logging.Logger) -> dict:
    """Extract and parse a JSON object from AI response text. Strips Gemini's
    'thought_process'/'thinking' keys that are not part of the expected schema."""
    text = (text or "").strip()
    if not text:
        raise ValueError("AI returned an empty response")

    def _strip_thought(d: dict) -> dict:
        d.pop("thought_process", None)
        d.pop("thinking", None)
        return d

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _strip_thought(parsed)
        raise ValueError(f"AI returned JSON {type(parsed).__name__}, expected object")
    except json.JSONDecodeError:
        pass

    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return _strip_thought(parsed)
            except (json.JSONDecodeError, IndexError, TypeError):
                continue

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return _strip_thought(parsed)

    preview = re.sub(r"\s+", " ", text[:700]).strip()
    logger.error(f"Could not extract JSON object from AI response: {preview}...")
    raise ValueError(f"AI response was not valid JSON. Preview: {preview[:300]}")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON TASK RUNNERS  (validate is injected by the caller — no analyzer import)
# ═══════════════════════════════════════════════════════════════════════════════
#
# A "work-unit" is one (task_name, prompt). For NVIDIA-primary plans every unit
# fires NVIDIA_RACE_COUNT concurrent attempts, WAITS FOR ALL of them, and keeps the
# BEST valid response (highest `score`); failed attempts are ignored, and if every
# attempt fails the whole set is re-fired. For non-NVIDIA primary plans the unit
# runs a single sequential retry loop.
#
# `validate(parsed_dict) -> dict` must raise on an invalid/unusable response.
# `score(parsed_dict) -> number` ranks the successes for the best-of pick (optional).

def _attempt_plan_once(
    job_dir, task_name, prompt, plan, logger, *,
    validate, enable_reasoning, clip_count, temperature=None, worker_label="", done=None,
) -> dict | None:
    """Walk the provider plan once (each model AI_MAX_RETRIES times). Return the
    validated dict on first success, None if aborted because `done` was set, and
    raise if the whole plan was exhausted without a usable response."""
    last_error = None
    for plan_idx, (prov, mdl, key) in enumerate(plan):
        if done is not None and done.is_set():
            return None
        # Are there later plan entries that use a DIFFERENT key for this
        # provider? If so, an auth error on the current key isn't fatal —
        # we want the secondary-key failover entries to get their turn.
        has_alternate_key = any(
            p2 == prov and k2 and k2 != key
            for (p2, _m2, k2) in plan[plan_idx + 1:]
        )
        use_reasoning = bool(enable_reasoning)
        use_json_mode = True
        for attempt in range(1, int(config.AI_MAX_RETRIES) + 1):
            if done is not None and done.is_set():
                return None
            raw_response = ""
            try:
                logger.info(
                    f"[{task_name}{worker_label}] API call starting: provider={prov}, "
                    f"model={mdl}, attempt={attempt}/{config.AI_MAX_RETRIES}"
                )
                raw_response = _dispatch_call(
                    prov, mdl, key, prompt, logger,
                    enable_reasoning=use_reasoning, clip_count=clip_count,
                    response_format=use_json_mode, temperature=temperature,
                )
                parsed = _extract_json(raw_response, logger)
                parsed = validate(parsed)
                if done is not None and done.is_set():
                    # Another racer already won; don't bother caching ours.
                    return None
                logger.info(f"[{task_name}{worker_label}] succeeded with {prov} ({mdl})")
                return parsed
            except Exception as exc:
                last_error = exc
                err_text = str(exc)
                logger.error(
                    f"[{task_name}{worker_label}] failed with {prov} ({mdl}) "
                    f"attempt {attempt}/{config.AI_MAX_RETRIES}: {err_text}"
                )
                _save_ai_failure(job_dir, task_name, prov, mdl, attempt, err_text, raw_response, logger)
                if raw_response:
                    _save_bad_ai_response(job_dir, f"{task_name}_{prov}_{mdl}", attempt, raw_response, logger)

                if _is_nvidia_auth_error(exc):
                    if has_alternate_key:
                        logger.warning(
                            f"[{task_name}{worker_label}] {prov} auth failed on this key — "
                            f"failing over to next key in the plan"
                        )
                        break
                    key_var = {
                        "nvidia": "NVIDIA_API_KEY", "openai": "OPENAI_API",
                        "claude": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY",
                    }.get(prov, "the API key")
                    raise RuntimeError(
                        f"{prov} authentication failed ({mdl}). Check {key_var} in .env. ({err_text})"
                    )

                # A 403/404 scoped to this one model (no access / unknown id) is
                # deterministic — don't burn the remaining retries on it, just move
                # on to the next model in the fallback chain.
                if _is_model_access_error(exc):
                    logger.warning(
                        f"[{task_name}{worker_label}] {prov} model '{mdl}' not accessible "
                        f"to this key; skipping to next model in fallback chain"
                    )
                    break

                err_lower = err_text.lower()
                if _looks_like_response_format_rejection(err_lower):
                    use_json_mode = False
                if _looks_like_reasoning_rejection(err_lower):
                    use_reasoning = False

                if attempt < int(config.AI_MAX_RETRIES):
                    sleep_seconds = _retry_sleep_seconds(prov, exc)
                    if sleep_seconds:
                        logger.info(f"[{task_name}{worker_label}] waiting {sleep_seconds}s before retry")
                        time.sleep(sleep_seconds)
        logger.warning(f"[{task_name}{worker_label}] exhausted retries for {prov} ({mdl}); trying next model")

    raise RuntimeError(f"AI task '{task_name}' failed after all retries and model fallbacks: {last_error}")


def _run_sequential_json_task(
    job_dir, task_name, prompt, plan, logger, *,
    validate, enable_reasoning, clip_count, temperature=None,
) -> dict:
    """Single-threaded retry loop. Retries the whole plan forever under
    AI_RETRY_FOREVER (NVIDIA: NVIDIA_RETRY_FOREVER); non-NVIDIA primaries keep a
    cycle cap so a deterministic failure surfaces instead of hanging."""
    primary = plan[0][0] if plan else ""
    retry_forever = bool(getattr(config, "AI_RETRY_FOREVER", True))
    if primary == "nvidia" and bool(getattr(config, "NVIDIA_RETRY_FOREVER", True)):
        retry_forever = True
        max_cycles = 0
    else:
        max_cycles = int(getattr(config, "AI_MAX_RETRY_CYCLES", 25))

    cycle = 0
    while True:
        cycle += 1
        try:
            parsed = _attempt_plan_once(
                job_dir, task_name, prompt, plan, logger,
                validate=validate, enable_reasoning=enable_reasoning,
                clip_count=clip_count, temperature=temperature,
            )
            if parsed is not None:
                return parsed
        except Exception as exc:
            if not retry_forever:
                raise
            if _is_nvidia_auth_error(exc):
                raise
            if max_cycles and cycle >= max_cycles:
                logger.error(f"[{task_name}] giving up after {cycle} cycles: {exc}")
                raise
            sleep_seconds = _retry_sleep_seconds(primary, exc)
            logger.warning(
                f"[{task_name}] cycle {cycle} exhausted whole provider plan: {exc}. "
                f"Retrying in {sleep_seconds}s (cycle {cycle}/{max_cycles or '∞'})"
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)


def _run_raced_json_task(
    job_dir, task_name, prompt, plan, logger, *,
    validate, enable_reasoning, clip_count, race_count, temperature=None,
    score=None,
) -> dict:
    """Fire `race_count` attempts concurrently, WAIT FOR ALL, return the BEST
    validated response — not the fastest.

    Each racer makes ONE bounded pass over the provider plan (each model tried
    AI_MAX_RETRIES times). Some racers may fail; the best response among the
    successes wins, ranked by `score` (default: most candidates). If EVERY racer
    fails, the whole set is re-fired (retry-forever; NVIDIA: only 401/403 is
    fatal and aborts immediately).

    STRICTLY NO TIMEOUTS: a window is allowed to run as long as it needs so that
    every clip is surfaced. We join all racers before returning, which also means
    no abandoned racer is left streaming into a later pipeline stage.
    """
    if score is None:
        score = lambda _d: 0  # no preference → any success is acceptable

    retry_forever = bool(getattr(config, "AI_RETRY_FOREVER", True))
    nvidia_forever = bool(getattr(config, "NVIDIA_RETRY_FOREVER", True))
    primary = plan[0][0] if plan else ""
    if primary == "nvidia" and nvidia_forever:
        max_cycles = 0
    else:
        max_cycles = int(getattr(config, "AI_MAX_RETRY_CYCLES", 25))

    cycle = 0
    while True:
        cycle += 1
        results: list[dict] = []
        results_lock = threading.Lock()
        fatal: dict = {}

        def worker(idx: int):
            label = f"#race{idx + 1}"
            try:
                parsed = _attempt_plan_once(
                    job_dir, task_name, prompt, plan, logger,
                    validate=validate, enable_reasoning=enable_reasoning,
                    clip_count=clip_count, temperature=temperature,
                    worker_label=label,
                )
                if parsed is not None:
                    with results_lock:
                        results.append(parsed)
            except Exception as exc:
                if _is_nvidia_auth_error(exc):
                    with results_lock:
                        fatal.setdefault("error", exc)
                else:
                    logger.warning(f"[{task_name}{label}] attempt failed (ignored, best-of): {exc}")

        threads = [
            threading.Thread(target=worker, args=(i,), name=f"{task_name}-race{i + 1}", daemon=True)
            for i in range(int(race_count))
        ]
        logger.info(
            f"[{task_name}] firing {race_count} concurrent attempt(s); "
            f"BEST of the successes wins (waiting for all — no timeout)"
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()  # NO TIMEOUT — wait for every racer to finish

        if fatal:
            raise fatal["error"]

        if results:
            best = max(results, key=score)
            logger.info(
                f"[{task_name}] best of {len(results)}/{race_count} success(es) "
                f"chosen (score={score(best)})"
            )
            return best

        # No racer produced a valid response this cycle.
        if not retry_forever:
            raise RuntimeError(
                f"AI task '{task_name}' produced no valid response across {race_count} racers"
            )
        if max_cycles and cycle >= max_cycles:
            raise RuntimeError(
                f"AI task '{task_name}' failed after {cycle} cycle(s) of {race_count} racers"
            )
        sleep_seconds = _retry_sleep_seconds(primary)
        logger.warning(
            f"[{task_name}] all {race_count} racers failed (cycle {cycle}); "
            f"re-firing in {sleep_seconds}s"
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)


def run_json_task(
    job_dir, task_name, prompt, plan, logger, *,
    validate, enable_reasoning, clip_count, race_count: int | None = None,
    temperature: float | None = None, score=None,
) -> dict:
    """Run one work-unit. Checks cache first, then for NVIDIA fires `race_count`
    concurrent attempts and keeps the BEST (highest `score`) of the successes;
    other providers run a single sequential retry loop. Caches the chosen result.
    """
    cached = get_cached_api_response(job_dir, task_name, logger)
    if cached is not None and isinstance(cached, dict):
        try:
            validated = validate(cached)
            logger.info(f"[{task_name}] successfully loaded from cache")
            return validated
        except Exception as e:
            logger.warning(f"Cached data for {task_name} failed validation: {e}. Re-running API task.")

    primary = plan[0][0] if plan else ""
    rc = int(race_count if race_count is not None else getattr(config, "NVIDIA_RACE_COUNT", 5))

    if primary == "nvidia" and rc > 1:
        result = _run_raced_json_task(
            job_dir, task_name, prompt, plan, logger,
            validate=validate, enable_reasoning=enable_reasoning,
            clip_count=clip_count, race_count=rc, temperature=temperature,
            score=score,
        )
    else:
        result = _run_sequential_json_task(
            job_dir, task_name, prompt, plan, logger,
            validate=validate, enable_reasoning=enable_reasoning,
            clip_count=clip_count, temperature=temperature,
        )
    save_cached_api_response(job_dir, task_name, result, logger)
    return result


def run_json_tasks_concurrent(
    job_dir, tasks, plan, logger, *,
    validate, enable_reasoning, clip_count, label: str = "task",
    race_count: int | None = None, max_units: int | None = None,
    temperature: float | None = None, score=None,
) -> list[dict]:
    """Fan a list of (task_name, prompt) work-units across a thread pool. Each
    NVIDIA unit internally fires `race_count` calls and keeps the BEST (highest
    `score`) of the successes, so peak in-flight calls = units_in_parallel ×
    race_count. Results are returned in the same order as `tasks` ({} for any
    unit that raised — only possible when retry-forever off)."""
    if not tasks:
        return []
    units = int(max_units or getattr(config, "AI_MAX_PARALLEL_UNITS", 7))
    units = max(1, min(units, len(tasks)))
    rc = int(race_count if race_count is not None else getattr(config, "NVIDIA_RACE_COUNT", 5))
    logger.info(
        f"[{label}] dispatching {len(tasks)} work-unit(s) "
        f"({units} in parallel x {rc} racers = up to {units * rc} in-flight call(s))"
    )

    results: list[dict] = [{}] * len(tasks)
    with ThreadPoolExecutor(max_workers=units, thread_name_prefix=f"ai-{label}") as pool:
        future_to_idx = {
            pool.submit(
                run_json_task,
                job_dir, name, prompt, plan, logger,
                validate=validate, enable_reasoning=enable_reasoning,
                clip_count=clip_count, race_count=rc, temperature=temperature,
                score=score,
            ): idx
            for idx, (name, prompt) in enumerate(tasks)
        }
        completed = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result() or {}
            except Exception as exc:
                logger.error(f"[{label}] unit #{idx + 1} raised after retries: {exc}")
                results[idx] = {}
            completed += 1
            if completed % max(1, len(tasks) // 4) == 0 or completed == len(tasks):
                logger.info(f"[{label}] {completed}/{len(tasks)} done")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC SINGLE-SHOT LLM CALL (used by copywriter / blogger / clip_selector)
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, settings: dict, logger: logging.Logger,
             temperature: float | None = None) -> str:
    """Call the LLM with a prompt and return the raw response string. Walks the
    configured provider/model plan with AI_MAX_RETRIES per model. `temperature`
    (or settings['temperature']) lets callers run hot for creative copy or cold
    for precision; None falls back to the provider default."""
    provider = _normalize_provider(settings.get("ai_provider", config.DEFAULT_AI_PROVIDER))
    model = settings.get("ai_model", "")
    api_key = settings.get("api_key", "")
    if temperature is None:
        temperature = settings.get("temperature")

    providers_to_try = _analysis_provider_plan(provider, model, api_key, task_type="complex")

    last_error = None
    for prov, mdl, key in providers_to_try:
        for attempt in range(1, int(config.AI_MAX_RETRIES) + 1):
            try:
                logger.info(f"[call_llm] Calling: provider={prov}, model={mdl}, attempt={attempt}/{config.AI_MAX_RETRIES}")
                response_text = _dispatch_call(
                    prov, mdl, key, prompt, logger,
                    enable_reasoning=settings.get("enable_reasoning", False),
                    clip_count=10, response_format=False, temperature=temperature,
                )
                if response_text and response_text.strip():
                    return response_text
            except Exception as e:
                last_error = e
                logger.warning(f"[call_llm] attempt {attempt} failed: {e}")
                if attempt < int(config.AI_MAX_RETRIES):
                    time.sleep(_retry_sleep_seconds(prov, e))

    raise RuntimeError(f"call_llm failed after all retries/fallbacks: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTIVITY TEST (UI "test key" button)
# ═══════════════════════════════════════════════════════════════════════════════

def test_api_key(provider: str, api_key: str, model: str = "") -> dict:
    """Test an API key with a simple prompt. Returns dict(success, message, model)."""
    test_prompt = "OK"
    provider = _normalize_provider(provider)

    try:
        if provider == "claude":
            import anthropic
            if not model:
                model = config.DEFAULT_CLAUDE_MODEL
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model, max_tokens=1,
                messages=[{"role": "user", "content": test_prompt}],
            )
            return {"success": True, "message": f"Connected to Claude ({model})",
                    "model": model, "response": response.content[0].text}

        elif provider == "gemini":
            from google import genai
            if not model:
                model = config.DEFAULT_GEMINI_MODEL
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model, contents=test_prompt,
                config=genai.types.GenerateContentConfig(max_output_tokens=1, temperature=0),
            )
            return {"success": True, "message": f"Connected to Gemini ({model})",
                    "model": model, "response": response.text}

        elif provider == "nvidia":
            import requests
            if not model:
                model = config.NVIDIA_TEST_MODEL
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 16,
                "temperature": 0,
            }
            model_l = str(model).lower()
            if model_l.startswith(config.NVIDIA_DEEPSEEK_PREFIX):
                payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
            elif model_l.startswith(config.NVIDIA_NEMOTRON_PREFIX):
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            response = requests.post(config.NVIDIA_CHAT_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code >= 400:
                raise RuntimeError(_http_error_message("NVIDIA key test", response))
            data = response.json()
            return {"success": True, "message": f"Connected to NVIDIA ({model})", "model": model,
                    "response": data.get("choices", [{}])[0].get("message", {}).get("content", "")}

        elif provider == "openai":
            import requests
            if not model:
                model = config.DEFAULT_OPENAI_MODEL
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Connection": "close"}
            is_reasoning_model = _is_openai_reasoning_model(model)
            payload = {"model": model, "messages": [{"role": "user", "content": test_prompt}], "stream": False}
            if is_reasoning_model:
                payload["max_completion_tokens"] = 1024
            else:
                payload["max_tokens"] = 2
                payload["temperature"] = 0
            response = requests.post(config.OPENAI_CHAT_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code >= 400:
                raise RuntimeError(_http_error_message("OpenAI key test", response))
            data = response.json()
            return {"success": True, "message": f"Connected to OpenAI ({model})", "model": model,
                    "response": data["choices"][0]["message"]["content"]}

        else:
            return {"success": False, "message": f"Unknown provider: {provider}"}

    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}
