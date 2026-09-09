"""
Video Clipper — Central Configuration
======================================
Every tunable setting lives here.  Environment variables from .env override
these defaults where noted.  UI settings override both for the current job.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

FFMPEG_PATH = os.getenv(
    "FFMPEG_PATH",
    os.path.join(BASE_DIR, "ffmpeg-bin", "ffmpeg.exe"),
)
FFPROBE_PATH = os.getenv(
    "FFPROBE_PATH",
    os.path.join(BASE_DIR, "ffmpeg-bin", "ffprobe.exe"),
)

# ═══════════════════════════════════════════════════════════════════════════════
# GPU-ENABLED FFMPEG (for NVENC extraction)
# ═══════════════════════════════════════════════════════════════════════════════

GPU_FFMPEG_PATH = os.getenv(
    "GPU_FFMPEG_PATH",
    os.path.join(BASE_DIR, "ffmpeg-bin", "ffmpeg.exe"),
)
GPU_FFPROBE_PATH = os.getenv(
    "GPU_FFPROBE_PATH",
    os.path.join(BASE_DIR, "ffmpeg-bin", "ffprobe.exe"),
)

# Force single parallel extraction when using GPU to avoid context conflicts
GPU_FORCE_SINGLE_WORKER = True

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# ═══════════════════════════════════════════════════════════════════════════════
# AI PROVIDER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY",    "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
# Optional secondary/tertiary keys — when the primary key's per-model retries
# all fail auth or quota (common: daily-limit 403s), the provider plan
# automatically retries the SAME model with each fallback key in order before
# degrading to the next model. Leave empty to disable a given tier.
NVIDIA_API_KEY_1 = os.getenv("NVIDIA_API_KEY_1", "")
NVIDIA_API_KEY_2 = os.getenv("NVIDIA_API_KEY_2", "")
# Key priority for the NVIDIA analysis ladder. KEY_2 is first by default because
# it currently holds mistral access — the bare NVIDIA_API_KEY 403s ("auth failed")
# on mistral-large-3 (a fixed entitlement issue, not a throttle), while 429s on
# the others are temporary rate limits that clear on failover. Override via env
# as a comma-separated list of these variable names.
NVIDIA_KEY_ORDER = [
    n.strip() for n in os.getenv(
        "NVIDIA_KEY_ORDER", "NVIDIA_API_KEY_2,NVIDIA_API_KEY_1,NVIDIA_API_KEY"
    ).split(",") if n.strip()
]
OPENAI_API = os.getenv("OPENAI_API", "")

DEFAULT_AI_PROVIDER = "nvidia"

# ── Analysis route: API vs Manual (external LLM) ──────────────────────────────
# Two ways to run the clip-finding analysis:
#   "api"    — call the configured AI provider (NVIDIA/OpenAI/Gemini/Claude) over
#              the network, exactly as before.
#   "manual" — the pipeline PAUSES right after transcription and emits the
#              time-coded transcript windows (EXTERNAL_LLM_WINDOWS_FILE) plus the
#              fixed system prompt (pipeline/prompts.txt). You run those by hand
#              in any chat LLM, paste the per-window JSON back into the dashboard,
#              and the pipeline ingests it from <job>/EXTERNAL_LLM_INPUT_FILE with
#              NO API key and NO network call. The manual output already carries
#              boundaries + full metadata, so discovery/refine/judge are skipped;
#              local_clips_generator still pins word-exact times from the verbatim
#              start_words/end_words, so the matcher runs exactly as in API mode.
# Manual is the DEFAULT per project preference (set DEFAULT_ANALYSIS_MODE=api to flip).
DEFAULT_ANALYSIS_MODE = os.getenv("DEFAULT_ANALYSIS_MODE", "manual").strip().lower()
# Per-job file the manual route READS the pasted LLM output from.
EXTERNAL_LLM_INPUT_FILE = "external_llm.txt"
# Per-job file the manual route WRITES the paste-ready, time-coded windows to.
EXTERNAL_LLM_WINDOWS_FILE = "external_llm_windows.txt"
# Project-relative path to the fixed system prompt the user pastes into their LLM.
EXTERNAL_LLM_SYSTEM_PROMPT_FILE = os.path.join("pipeline", "prompts.txt")
# Two manual-route clips are treated as duplicates (the lower-scored one dropped)
# when their stitched transcript text is at least this similar (difflib ratio, 0..1).
EXTERNAL_LLM_SIMILARITY_THRESHOLD = float(
    os.getenv("EXTERNAL_LLM_SIMILARITY_THRESHOLD", "0.90")
)

# ── Claude ──────────────────────────────────────────────────────────────────
CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-haiku-4-20250414",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ── Gemini ──────────────────────────────────────────────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# ── NVIDIA NIM (integrate.api.nvidia.com — OpenAI-compatible) ────────────────
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_CHAT_API_URL = f"{NVIDIA_BASE_URL.rstrip('/')}/chat/completions"
# FULL working-model list for this NVIDIA key, ordered STRONGEST → weakest. Every
# entry returned valid JSON with real content in a live full-catalog benchmark
# (37 models, reasoning ON); ordering is by capability (param scale × reasoning ×
# output richness), so the dropdown shows the strongest at the top (= default).
# This list also IS the runtime fallback ladder: each model is tried
# AI_MAX_RETRIES times, then the next, looping forever until a valid response.
#
# EXCLUDED entirely (benchmarked as stalls / 404 / empty output, never configure):
#   stalls/timeout : nemotron-3-ultra-550b, deepseek-v4-pro, qwen3-next-80b,
#                    glm-5.1, minimax-m2.7
#   404 not hosted : nemotron-4-340b, llama-3.1-nemotron-70b, llama-3.1-nemotron-51b,
#                    dbrx-instruct, phi-3.5-moe, jamba-1.5-large, mistral-large-2,
#                    palmyra-creative-122b
#   broken output  : minimax-m3 (0ch), step-3.5-flash (0ch), kimi-k2.6 (truncated)
NVIDIA_MODELS = [
    "mistralai/mistral-large-3-675b-instruct-2512",   # 675B — biggest, richest (181ch), ~16s
    "qwen/qwen3.5-397b-a17b",                          # 397B MoE reasoning, ~47s
    "nvidia/nemotron-3-super-120b-a12b",               # 120B reasoning, richest reasoner (147ch), ~14s
    "mistralai/mistral-medium-3.5-128b",               # 128B, ~14s
    "qwen/qwen3.5-122b-a10b",                          # 122B reasoning, ~10s
    "openai/gpt-oss-120b",                             # 120B reasoning, fast ~9s, never stalls
    "mistralai/mistral-nemotron",                      # reasoning, rich (178ch), ~16s
    "stockmark/stockmark-2-100b-instruct",             # 100B, very fast ~4s
    "abacusai/dracarys-llama-3.1-70b-instruct",        # 70B, ~23s
    "meta/llama-3.3-70b-instruct",                     # 70B, ~25s
    "meta/llama-3.1-70b-instruct",                     # 70B, ~23s
    "nvidia/llama-3.3-nemotron-super-49b-v1",          # 49B reasoning, FASTEST reliable ~7s
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",        # 49B reasoning, ~24s
    "deepseek-ai/deepseek-v4-flash",                   # reasoning DeepSeek, ~23s
    "meta/llama-4-maverick-17b-128e-instruct",         # 17B×128E MoE, ~19s
    "nvidia/nvidia-nemotron-nano-9b-v2",               # 9B reasoning, ~17s
    "stepfun-ai/step-3.7-flash",                       # works but slow ~88s (last resort)
]
DEFAULT_NVIDIA_MODEL = NVIDIA_MODELS[0]
# Fast, never-stalls model for the UI "Test key" button (not the heavy 675B).
NVIDIA_TEST_MODEL = "openai/gpt-oss-120b"

# Runtime fallback ladder = the full strongest→weakest list above. Applies to
# discovery AND refinement; the user-selected model is prepended at call time.
_NVIDIA_PREFERENCE = list(NVIDIA_MODELS)
NVIDIA_DISCOVERY_FALLBACK_MODELS = list(_NVIDIA_PREFERENCE)
NVIDIA_ANALYSIS_FALLBACK_MODELS = list(_NVIDIA_PREFERENCE)

# ── NVIDIA sampling + reasoning controls (configurable) ──────────────────────
# Nemotron recommends temperature 1.0 / top_p 0.95. Reasoning is gated by the
# existing UI "enable_reasoning" toggle. Per-family payload shaping lives in
# analyzer._call_nvidia so DeepSeek (chat_template_kwargs.thinking) and Nemotron
# (extra_body reasoning_budget) parameters never get mixed up.
NVIDIA_TEMPERATURE = float(os.getenv("NVIDIA_TEMPERATURE", "0.7"))  # fallback when a stage temp isn't set
# ── Per-stage sampling temperature ───────────────────────────────────────────
# Analysis stages favor accuracy (low temp). Note: reasoning is ON for these
# stages and reasoning models tend to repeat/degenerate below ~0.4, so the floor
# is kept there. Creative copy stages (copywriter/blogger) run HOT for punchy,
# varied YouTube output. Clip validation is a precision check, so it stays low.
NVIDIA_TEMP_DISCOVERY = float(os.getenv("NVIDIA_TEMP_DISCOVERY", "0.7"))
NVIDIA_TEMP_JUDGE = float(os.getenv("NVIDIA_TEMP_JUDGE", "0.4"))
NVIDIA_TEMP_REFINE = float(os.getenv("NVIDIA_TEMP_REFINE", "0.4"))        # boundaries = precision
NVIDIA_TEMP_COMPRESSION = float(os.getenv("NVIDIA_TEMP_COMPRESSION", "0.4"))
NVIDIA_TEMP_VALIDATION = float(os.getenv("NVIDIA_TEMP_VALIDATION", "0.4"))  # clip_selector AI validation
AI_TEMP_CREATIVE = float(os.getenv("AI_TEMP_CREATIVE", "1.3"))            # copywriter + blogger
NVIDIA_TOP_P = float(os.getenv("NVIDIA_TOP_P", "0.95"))
NVIDIA_REASONING_BUDGET = int(os.getenv("NVIDIA_REASONING_BUDGET", "4096"))
# Effort level for providers that accept a string knob (DeepSeek / OpenAI o-series).
# "low" is the anti-clog default — at "medium"/"high" the streaming reader
# observed 180k–200k chunks on a single call as the model spun in an endless
# thinking loop. "low" keeps reasoning capped tight enough that the JSON
# actually emerges in a bounded number of chunks.
NVIDIA_REASONING_EFFORT = os.getenv("NVIDIA_REASONING_EFFORT", "low")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
# Model-family prefixes used to select the correct reasoning parameter shape.
NVIDIA_NEMOTRON_PREFIX = "nvidia/nemotron"
NVIDIA_DEEPSEEK_PREFIX = "deepseek-ai/"

# ── NVIDIA execution ─────────────────────────────────────────────────────────
# Discovery and refinement RACE: every work-unit (one discovery window, or one
# refinement batch) fires NVIDIA_RACE_COUNT identical concurrent calls and the
# FIRST valid response wins; the other racers are abandoned. This is a
# reliability hedge — NVIDIA calls fail often, so firing 5 in parallel and
# taking whichever returns first dramatically cuts wall-clock time to a good
# answer. (This intentionally re-introduces racing that was previously removed;
# the reason now is failure-resilience, not latency-vs-richness.)
# No read timeout (connect timeout still applies so dead sockets fail fast); a
# single stalled socket is covered by the other racers.
NVIDIA_NO_READ_TIMEOUT = os.getenv("NVIDIA_NO_READ_TIMEOUT", "true").lower() == "true"
NVIDIA_CONNECT_TIMEOUT = float(os.getenv("NVIDIA_CONNECT_TIMEOUT", "15"))
# Number of identical concurrent calls fired per work-unit (first valid wins).
# Default kept low; per-phase overrides below decide the actual count per phase.
NVIDIA_RACE_COUNT = int(os.getenv("NVIDIA_RACE_COUNT", "3"))
# Per-phase race counts. Only discovery races by default; refinement + judge
# fire a single API call each and rely on the model-ladder fallback for retry.
NVIDIA_DISCOVERY_RACE_COUNT = int(os.getenv("NVIDIA_DISCOVERY_RACE_COUNT", "3"))
NVIDIA_REFINE_RACE_COUNT = int(os.getenv("NVIDIA_REFINE_RACE_COUNT", "1"))
NVIDIA_JUDGE_RACE_COUNT = int(os.getenv("NVIDIA_JUDGE_RACE_COUNT", "1"))
# Max work-units processed in parallel. units × race_count = peak in-flight
# calls. 7 × 3 = 21, staying well under the ~40 rpm NVIDIA ceiling.
AI_MAX_PARALLEL_UNITS = int(os.getenv("AI_MAX_PARALLEL_UNITS", "7"))
# Clips per boundary-refinement batch. The prompt is per-candidate, so default
# is 1 — every clip gets its own focused refinement call, no batch contention.
# Raise via UI/env (e.g. 3-5) only if you've verified your model handles batch
# refinement well and you want to trade quality for fewer API calls.
REFINEMENT_BATCH_SIZE = int(os.getenv("REFINEMENT_BATCH_SIZE", "1"))
# Retry delay between failed NVIDIA attempts (0 = immediate retry).
NVIDIA_RETRY_SLEEP = float(os.getenv("NVIDIA_RETRY_SLEEP", "0"))
# On HTTP 429 (rate limit), immediate retry only makes it worse — back off this
# many seconds instead (overrides NVIDIA_RETRY_SLEEP for rate-limit errors).
NVIDIA_RATE_LIMIT_BACKOFF = float(os.getenv("NVIDIA_RATE_LIMIT_BACKOFF", "6"))

# ── NVIDIA request-per-minute (RPM) limiter ──────────────────────────────────
# NVIDIA's hard ceiling is ~40 rpm; target ~30 to run at max throughput while
# staying safely under. A true rolling-60s limiter gates every NVIDIA call (used
# for both racing discovery and per-clip refinement). Failed clips refire
# immediately, but the limiter spaces starts so we don't trip 429s.
NVIDIA_RPM = int(os.getenv("NVIDIA_RPM", "35"))
NVIDIA_RPM_CEILING = int(os.getenv("NVIDIA_RPM_CEILING", "40"))  # never exceed
# Reasoning is ON everywhere feasible (best-quality output is the priority).
# The 20k-chunk Nemotron blow-up seen before is bounded by NVIDIA_REASONING_BUDGET
# (capped thinking tokens), so discovery reasoning is now safe to enable.
NVIDIA_DISCOVERY_REASONING = os.getenv("NVIDIA_DISCOVERY_REASONING", "true").lower() == "true"
NVIDIA_REFINE_REASONING = os.getenv("NVIDIA_REFINE_REASONING", "true").lower() == "true"
# NVIDIA NIM enforces a TINY default ceiling (~128-512 tokens) when max_tokens
# is omitted, which truncated every Qwen JSON response in production. Send
# max_tokens=AI_MAX_TOKENS_ANALYSIS (65536) so responses can complete in full.
NVIDIA_NO_MAX_TOKENS = os.getenv("NVIDIA_NO_MAX_TOKENS", "false").lower() == "true"
# Retry NVIDIA clips forever until they yield (the user wants no retry cap).
NVIDIA_RETRY_FOREVER = os.getenv("NVIDIA_RETRY_FOREVER", "true").lower() == "true"

# Safety cap for NON-NVIDIA providers so a persistent failure can't hang the run
# even under AI_RETRY_FOREVER. 0 = unlimited. (NVIDIA ignores this per above.)
AI_MAX_RETRY_CYCLES = int(os.getenv("AI_MAX_RETRY_CYCLES", "25"))
# Legacy concurrency cap — retained but the RPM limiter is now the primary
# throttle. Kept generous so it doesn't fight the rate limiter.
NVIDIA_MAX_CONCURRENT = int(os.getenv("NVIDIA_MAX_CONCURRENT", "40"))

# ── OpenAI ──────────────────────────────────────────────────────────────────
OPENAI_CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
# This OpenAI project key is currently provisioned for gpt-5-nano ONLY (verified
# live against /v1/models and per-model chat probes — every other id, incl. the
# whole gpt-4.x/o-series and larger gpt-5 tiers, returns HTTP 403 model_not_found).
# gpt-5-nano is a GPT-5 reasoning model: api_provider treats `gpt-5*` like the
# o-series (max_completion_tokens, no custom temperature). If the project is later
# granted more models, just add their ids here and the UI dropdown updates via
# /api/config — no frontend edit needed.
OPENAI_MODELS = [
    "gpt-5-nano",
]
DEFAULT_OPENAI_MODEL = "gpt-5-nano"

# Fallback chain for discovery
OPENAI_DISCOVERY_FALLBACK_MODELS = [
    "gpt-5-nano",
]

# Fallback chain for complex reasoning tasks
OPENAI_ANALYSIS_FALLBACK_MODELS = [
    "gpt-5-nano",
]

# ── Shared AI settings ──────────────────────────────────────────────────────
AI_MAX_RETRIES = 3
# Concurrency: NVIDIA NIM has generous rate limits — fire many calls in parallel.
# Discovery windows and per-clip refinement use this pool size. Failed tasks
# retry forever (cycling through the model fallback chain) — the user asked us
# to not worry about API call count and re-send failures immediately.
AI_CONCURRENT_WORKERS = int(os.getenv("AI_CONCURRENT_WORKERS", "8"))
AI_RETRY_FOREVER = os.getenv("AI_RETRY_FOREVER", "true").lower() == "true"
AI_TEMPERATURE = 0.9
# Output token caps. Set generous so neither discovery (long candidate lists)
# nor refinement (full YouTube metadata + segment anchors) gets truncated. The
# user prefers paying for tokens over receiving a sentence-clipped JSON.
AI_MAX_TOKENS_ANALYSIS = 65536
AI_MAX_TOKENS_COPY = 8192
AI_RETRY_SLEEP_SECONDS = 10
AI_CROSS_PROVIDER_FALLBACK = True

# ── OpenAI "almighty" fallback ──────────────────────────────────────────────
# NVIDIA stays primary everywhere. OpenAI is reserved as the decider for the
# genuinely tough cases NVIDIA can't crack: a refinement batch that comes back
# EMPTY (failed), and clips that are STILL below-par after NVIDIA's verification
# re-call (scrutiny / tie-break). It never runs on the normal happy path, and a
# per-job cap bounds the spend.
OPENAI_ALMIGHTY_FALLBACK = os.getenv("OPENAI_ALMIGHTY_FALLBACK", "true").lower() == "true"
OPENAI_ALMIGHTY_MAX_CALLS_PER_JOB = int(os.getenv("OPENAI_ALMIGHTY_MAX_CALLS_PER_JOB", "30"))
# Prompt-input char budgets. Lifted so long videos don't get force-compacted
# at the cost of completeness — quality over latency.
AI_ANALYSIS_PROMPT_CHAR_BUDGET = int(
    os.getenv("AI_ANALYSIS_PROMPT_CHAR_BUDGET", "250000"))
AI_ANALYSIS_COMPACT_CHAR_BUDGET = int(
    os.getenv("AI_ANALYSIS_COMPACT_CHAR_BUDGET", "150000"))
# Discovery-pass compact-scan window cap. No quota during identification — set
# generous so the model sees the widest viable set of windows the prompt budget
# allows. Lift this if very long videos still feel under-covered.
AI_ANALYSIS_MAX_WINDOWS = int(os.getenv("AI_ANALYSIS_MAX_WINDOWS", "64"))
# Refinement nearby_context caps — wider so the model has room to move
# boundaries cleanly and detect newly-spotted intra-topic filler.
REFINE_CONTEXT_LINE_CAP = int(os.getenv("REFINE_CONTEXT_LINE_CAP", "100"))
REFINE_CONTEXT_WORD_BUDGET = int(os.getenv("REFINE_CONTEXT_WORD_BUDGET", "2000"))

# ── Local end-of-clip sentence-terminator snap ───────────────────────────────
# After the AI's end_words is resolved to a word index, the local matcher tries
# to land the cut on a sentence terminator (".", "!", "?"). Three knobs:
#   LOOKAHEAD: scan FORWARD this many seconds for the next terminator. Was
#     5.0 — too short for natural sentence completion (clips ended mid-word on
#     "you", "Divorced", etc.). 30s now matches the refinement prompt's
#     "extend up to 50-60s if it finishes the thought" budget while leaving
#     a safety margin.
#   SILENCE_BAIL: if the next word starts more than this many seconds after
#     the previous word's end, stop scanning (we're crossing dead air, not
#     extending a thought). 1.0s was too aggressive — natural inter-sentence
#     pauses can run 1.5-2.0s. 2.5s lets the snap span normal pauses.
#   LOOKBACK: if the forward scan finds NO terminator inside LOOKAHEAD, fall
#     back to scanning BACKWARD this many seconds for the most recent
#     terminator. This handles the over-extension case where the AI's
#     end_words includes the start of the next sentence — instead of shipping
#     a mid-word cut, retreat to the prior sentence end.
END_SENTENCE_LOOKAHEAD_S = float(os.getenv("END_SENTENCE_LOOKAHEAD_S", "30.0"))
END_SENTENCE_LOOKBACK_S = float(os.getenv("END_SENTENCE_LOOKBACK_S", "15.0"))
END_SENTENCE_SILENCE_BAIL_S = float(os.getenv("END_SENTENCE_SILENCE_BAIL_S", "2.5"))
AI_COPY_HASHTAG_MIN = int(os.getenv("AI_COPY_HASHTAG_MIN", "12"))
AI_COPY_HASHTAG_MAX = int(os.getenv("AI_COPY_HASHTAG_MAX", "15"))
AI_ANALYSIS_TIMEOUT = int(os.getenv("AI_ANALYSIS_TIMEOUT", "180"))
MAX_JUDGE_CANDIDATES = int(os.getenv("MAX_JUDGE_CANDIDATES", "64"))

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

TRANSCRIPTION_METHOD = os.getenv("TRANSCRIPTION_METHOD", "smallest")
TRANSCRIPTION_FALLBACK_ORDER = ["local", "smallest", "cpu"]

WHISPER_MODEL = os.getenv("WHISPER_MODEL",        "large-v3-turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE",     None)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE",       "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = 5

# ── Punctuation restoration ─────────────────────────────────────────────────
# Some STT engines (notably Smallest.ai) return word timestamps with little or no
# sentence punctuation. Without terminators, both the AI boundary picker and the
# matcher's sentence-snap have nothing to land on, so clips end mid-thought. This
# pass restores sentence/clause punctuation from inter-word PAUSES (engine- and
# dependency-free, no API). It runs only when the transcript's terminator density
# is below PUNCT_RESTORE_MIN_DENSITY, and never rewrites a word that already ends
# on a terminator — so already-punctuated transcripts are left untouched.
PUNCT_RESTORE_ENABLED = os.getenv("PUNCT_RESTORE_ENABLED", "true").lower() == "true"
# Restore only if existing terminators/word are below this (1 per ~40 words).
PUNCT_RESTORE_MIN_DENSITY = float(os.getenv("PUNCT_RESTORE_MIN_DENSITY", "0.025"))
# Inter-word gap (s) that marks a sentence end / a clause break.
PUNCT_SENTENCE_GAP_S = float(os.getenv("PUNCT_SENTENCE_GAP_S", "0.6"))
PUNCT_CLAUSE_GAP_S = float(os.getenv("PUNCT_CLAUSE_GAP_S", "0.32"))
# If a run goes this many words without a terminator, drop to the clause gap so
# we break the next reasonable pause instead of producing a mega-sentence.
PUNCT_FORCE_MAX_WORDS = int(os.getenv("PUNCT_FORCE_MAX_WORDS", "28"))

# ── Clip end-snap (matcher) ─────────────────────────────────────────────────
# After the AI's end_words is matched to a word-exact time, the matcher snaps to
# the next COMPLETE break — a pause/terminator landing on a content word, never
# on a fragment ("...for your", "...a less"). The forward window is generous
# (~12s) so it can finish the phrase and deliver a complete, learnable point
# (length is flexible). It still bails on a > SILENCE_BAIL gap, so it never
# crosses into a new topic. A pause >= END_SENTENCE_PAUSE_S counts as a break so
# unpunctuated transcripts still snap to a natural breath.
END_SENTENCE_LOOKAHEAD_S = float(os.getenv("END_SENTENCE_LOOKAHEAD_S", "12.0"))
END_SENTENCE_LOOKBACK_S = float(os.getenv("END_SENTENCE_LOOKBACK_S", "6.0"))
END_SENTENCE_SILENCE_BAIL_S = float(os.getenv("END_SENTENCE_SILENCE_BAIL_S", "2.5"))
END_SENTENCE_PAUSE_S = float(os.getenv("END_SENTENCE_PAUSE_S", "0.45"))

# Near-duplicate dedup: two clips are duplicates if they share more than this
# fraction of the SHORTER clip's time span — the lower-scored one is dropped.
# 0.5 (half the shorter clip shared) catches partial near-dupes (measured: the
# tennis trio sat at 56-63%) while keeping genuinely-distinct adjacent clips
# (which measured <=44% overlap). Was 0.85, which let all near-dupes ship.
CLIP_OVERLAP_DEDUP_RATIO = float(os.getenv("CLIP_OVERLAP_DEDUP_RATIO", "0.5"))

# ── Smallest.ai (cloud STT) ─────────────────────────────────────────────────
SMALLEST_API_KEY = os.getenv("SMALLEST_API_KEY", "")
SMALLEST_API_URL = "https://api.smallest.ai/waves/v1/pulse/get_text"
SMALLEST_LANGUAGE = os.getenv("SMALLEST_LANGUAGE", "en")

# ── TTS Hook Overlay (Smallest.ai Lightning v3.1) ─────────────────────────
TTS_HOOK_ENABLED = False                                # Default off
TTS_HOOK_VOICE_ID = os.getenv("TTS_HOOK_VOICE_ID", "robert")
TTS_HOOK_SAMPLE_RATE = int(os.getenv("TTS_HOOK_SAMPLE_RATE", "44100"))
TTS_HOOK_SPEED = 0.75
# Silence kept after the voice finishes so the speech never feels cut off
TTS_HOOK_TAIL_S = 0.2
# Low-shelf bass lift (dB) applied to the hook voice; a limiter downstream
# prevents the boost from clipping
TTS_HOOK_BASS_GAIN_DB = 9.0
TTS_HOOK_FONT = "assets/fonts/BarlowCondensed-Black.ttf"
TTS_HOOK_FONT_SIZE_PCT = float(
    os.getenv("TTS_HOOK_FONT_SIZE_PCT", "17.0"))  # % of height
TTS_HOOK_MUSIC = "assets/audio/tts_music.mp3"
TRANSITION_SFX = "assets/audio/whoosh.mp3"
# Active word — vibrant yellow
TTS_HOOK_TEXT_COLOR = "#FFD600"
# Revealed words — light yellow
TTS_HOOK_TEXT_COLOR_REVEALED = "#FFFFB0"
TTS_HOOK_STROKE_WIDTH = 3                                # px black outline
TTS_HOOK_BG_BLUR = 12                                    # boxblur strength
TTS_HOOK_ZOOM_FACTOR = 0.06                              # Slow zoom 1.0 → 1.06
# auto, above_face, below_face, top, bottom
TTS_HOOK_TEXT_PLACEMENT = "auto"
# Line spacing multiplier
TTS_HOOK_LINE_HEIGHT_RATIO = 1.2


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP EXTRACTION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CLIP_COUNT = 10
DEFAULT_MIN_DURATION = 30
DEFAULT_MAX_DURATION = 90
DEFAULT_ASPECT_RATIO = "9:16"

AI_ANALYSIS_WINDOW_SECONDS = 900.0
AI_ANALYSIS_OVERLAP_SECONDS = 180.0

DEFAULT_REMOVE_SILENCE = False
DEFAULT_EMOJI_CAPTIONS = False

# Optional extraction resume marker. Leave empty for normal resume-by-existing raw clips.
# Accepts a 1-based index ("6") or a clip name ("clip_06").
EXTRACTION_RESUME_FROM_CLIP = os.getenv("EXTRACTION_RESUME_FROM_CLIP", "").strip()
EXTRACTION_WORKERS = 3

# Clip Tool v4 pipeline adapter.  Defaults preserve clips_plan ranges directly and
# force fresh template context for every task/clip unless explicitly overridden.
V4_SPEAKER_FILTER = os.getenv("V4_SPEAKER_FILTER", "continuous")
V4_CLEAR_TEMPLATE_CACHE_EACH_RUN = os.getenv("V4_CLEAR_TEMPLATE_CACHE_EACH_RUN", "false").lower() == "true"
V4_MANUAL_TEMPLATES_PATH = os.getenv("V4_MANUAL_TEMPLATES_PATH", "").strip()
V4_TEMPLATE_TOOL_AUTO_LAUNCH = os.getenv("V4_TEMPLATE_TOOL_AUTO_LAUNCH", "true").lower() == "true"
V4_TEMPLATE_TOOL_PATH = os.getenv("V4_TEMPLATE_TOOL_PATH", "").strip()
V4_DIARIZATION_PATH = os.getenv("V4_DIARIZATION_PATH", "").strip()
V4_DIAR_PYTHON = os.getenv(
    "V4_DIAR_PYTHON",
    os.path.join(BASE_DIR, "venv1", "Scripts", "python.exe"),
)

VERTICAL_RESOLUTION = (1080, 1920)
HORIZONTAL_RESOLUTION = (1920, 1080)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED INSIGHTFACE FACE TRACKING  — Professional Revamp
# ═══════════════════════════════════════════════════════════════════════════════

# ── Master Switch ────────────────────────────────────────────────────────────
FACE_TRACKING_ENABLED = True

# ── Face Backend Selection ───────────────────────────────────────────────────
# "rtdetr"      → RT-DETR-L person detection (Ultralytics). DEFAULT.
#                 Pose-invariant — no side-profile / back-of-head misses.
#                 Identity in wide shots comes from template slot proximity
#                 (the manual bbox is the ground truth) + diarization. No
#                 face embeddings. Faster than InsightFace per call.
# "insightface" → InsightFace buffalo_l (SCRFD + ArcFace).
#                 Legacy path; needed only if you want embedding-based
#                 wide-shot identity instead of position+diarization.
FACE_BACKEND = os.getenv("FACE_BACKEND", "rtdetr").lower().strip()

# RT-DETR knobs (used only when FACE_BACKEND="rtdetr")
RTDETR_WEIGHTS_PATH = os.getenv(
    "RTDETR_WEIGHTS_PATH",
    os.path.join(BASE_DIR, "pipeline", "models", "weights", "rtdetr-l.pt"),
)
RTDETR_CONF_THRESHOLD = float(os.getenv("RTDETR_CONF_THRESHOLD", "0.70"))
RTDETR_IMGSZ = int(os.getenv("RTDETR_IMGSZ", "640"))
# Reject tiny background persons (e.g. a face on a tablet/phone screen far from
# the camera). A detection is dropped if its person box area is below this
# fraction of the (downscaled) analysis frame area.
RTDETR_MIN_AREA_FRAC = float(os.getenv("RTDETR_MIN_AREA_FRAC", "0.015"))
# GPU batch size for the two-phase batched detection pass. 16 fits comfortably
# in 4 GB VRAM at imgsz=640 (RT-DETR ~1.5 GB peak).
RTDETR_BATCH_SIZE = int(os.getenv("RTDETR_BATCH_SIZE", "16"))
# Longest side (px) the detection frames are downscaled to before detection.
# Input is 4K; RT-DETR letterboxes to imgsz internally anyway, so feeding it a
# ~960 px frame slashes decode RAM + transfer cost with no accuracy loss.
# Detection coordinates are scaled back to source resolution for cropping.
RTDETR_ANALYSIS_MAX_SIDE = int(os.getenv("RTDETR_ANALYSIS_MAX_SIDE", "960"))

# ── Head-pose refinement (true head center) ──────────────────────────────────
# RT-DETR gives only person boxes; using the box center drifts horizontally as
# the speaker gestures. YOLO-pose (yolo11n-pose, ~6 MB) gives a pose-invariant
# head center from nose/eye/ear keypoints. Falls back to the top-of-box estimate
# when no keypoints are found (e.g. back of head). Runs in the 4 GB budget.
HEAD_POSE_ENABLED = os.getenv("HEAD_POSE_ENABLED", "true").lower() == "true"
HEAD_POSE_WEIGHTS_PATH = os.getenv(
    "HEAD_POSE_WEIGHTS_PATH",
    os.path.join(BASE_DIR, "pipeline", "models", "weights", "yolo11n-pose.pt"),
)
POSE_BATCH_SIZE = int(os.getenv("POSE_BATCH_SIZE", "32"))

# Clip Tool tracking is intentionally rebuilt per clip and not persisted.
TRACKING_TEMPLATE_CACHE_ENABLED = False
TRACKING_TEMPLATE_FRAME_OFFSET = float(os.getenv("TRACKING_TEMPLATE_FRAME_OFFSET", "0.3"))
TRACKING_SCENE_SAMPLE_RATE = float(os.getenv("TRACKING_SCENE_SAMPLE_RATE", "4.0"))
TRACKING_SCENE_CORRELATION_THRESHOLD = float(os.getenv("TRACKING_SCENE_CORRELATION_THRESHOLD", "0.40"))
TRACKING_SCENE_MIN_GAP = float(os.getenv("TRACKING_SCENE_MIN_GAP", "0.4"))
TRACKING_PHASH_THRESHOLD = int(os.getenv("TRACKING_PHASH_THRESHOLD", "12"))
TRACKING_FACE_CASCADE_MIN_SIZE = (40, 40)

# Global execution device parsing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPU target configuration for InsightFace execution context
INSIGHTFACE_GPU_DEVICE_ID = 0

# ── InsightFace Model ─────────────────────────────────────────────────────────
# buffalo_l: 4 models, ~950 MB VRAM, ~55 s load on 1650 Ti  ← current (too heavy)
# buffalo_s: 2 models (det + rec only), ~220 MB VRAM, ~6 s load ← use this
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

# Drop detection resolution: 480×480 is fully sufficient for 960-px-wide frames.
# Saves ~180 MB VRAM vs 640×640; no meaningful accuracy loss for podcast faces.
TRACKING_GPU_DET_SIZE = (640, 640)

# Match the VRAM cap to the lighter model so the scheduler knows actual headroom.
# 512 MB is enough for buffalo_s; keeps 3.5 GB free for NVENC + other work.
INSIGHTFACE_VRAM_LIMIT = int(
    os.getenv("INSIGHTFACE_VRAM_LIMIT", str(512 * 1024 * 1024)))  # 512 MB

# ── Scene Detection Backend ───────────────────────────────────────────────────
# "transnetv2"   — neural network, ~95 % recall, uses ~350 MB VRAM, fastest
# "pyscenedetect"— CPU-only, ~88 % recall, no extra VRAM, fast
# "scdet"        — legacy FFmpeg two-pass, ~80 % recall (original behaviour)
# Pipeline auto-falls-back down the list if a backend is not installed.
SCENE_DETECT_BACKEND = os.getenv("SCENE_DETECT_BACKEND", "transnetv2")

# TransNetV2 — cut probability threshold (0.0–1.0). Lower = more sensitive.
TRANSNETV2_THRESHOLD = float(os.getenv("TRANSNETV2_THRESHOLD", "0.25"))

# PySceneDetect — adaptive detector sensitivity (lower = more cuts found)
PYSCENEDETECT_THRESHOLD = float(os.getenv("PYSCENEDETECT_THRESHOLD", "2.5"))

# Legacy scdet thresholds (used only when both modern backends are unavailable)
TRACKING_SCDET_PRIMARY = float(os.getenv("TRACKING_SCDET_PRIMARY",  "2.5"))
TRACKING_SCDET_SECONDARY = float(os.getenv("TRACKING_SCDET_SECONDARY", "1.5"))
TRACKING_SCDET_GAP_MIN = float(os.getenv("TRACKING_SCDET_GAP_MIN",  "2.0"))
TRACKING_SCDET_DEDUP_GAP = float(os.getenv("TRACKING_SCDET_DEDUP_GAP", "0.12"))

# ── Face Quality Filters ──────────────────────────────────────────────────────
# Minimum InsightFace detection confidence to accept a face.
TRACKING_MIN_FACE_SCORE = float(os.getenv("TRACKING_MIN_FACE_SCORE", "0.45"))
# Minimum face bounding-box area as a fraction of the decoded frame area.
# Rejects tiny background faces. 0.012 ≈ face must be at least 1.2% of frame.
TRACKING_MIN_FACE_AREA_FRAC = float(
    os.getenv("TRACKING_MIN_FACE_AREA_FRAC", "0.012"))

# ── Podcast Tracking — Template Discovery ────────────────────────────────────
# Hamming distance threshold (0–64): hashes within this distance are the same template.
TEMPLATE_PHASH_THRESHOLD = int(os.getenv("TEMPLATE_PHASH_THRESHOLD", "12"))
# Hard cap on distinct templates discovered per video.
MAX_TEMPLATES = int(os.getenv("MAX_TEMPLATES", "8"))
# Seconds after a cut to sample the representative frame (avoids transition blur).
TEMPLATE_FRAME_OFFSET = float(os.getenv("TEMPLATE_FRAME_OFFSET", "0.5"))

# ── Podcast Tracking — Face Detection Sampling Rates ─────────────────────────
# Seconds between InsightFace calls inside a single-speaker close-up segment.
SINGLE_SAMPLE_INTERVAL = float(os.getenv("SINGLE_SAMPLE_INTERVAL", "3.0"))  # was 1.5
# Seconds between InsightFace calls inside a wide (two-speaker) segment.
WIDE_SAMPLE_INTERVAL = float(os.getenv("WIDE_SAMPLE_INTERVAL", "0.5"))
# Pixels (source resolution) of crop-x drift before injecting a correction keyframe.
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "120.0"))
# Fraction of crop_w to offset horizontally for face direction (nose room).
NOSE_ROOM_FACTOR = float(os.getenv("NOSE_ROOM_FACTOR", "0.08"))

# ── Podcast Tracking — Face Identity ─────────────────────────────────────────
# Cosine similarity threshold (L2-normalised embeddings) for same-person match.
EMBEDDING_MATCH_THRESHOLD = float(
    os.getenv("EMBEDDING_MATCH_THRESHOLD", "0.45"))

# ── Podcast Tracking — Frame Difference Sweep (Layer 3 cut detection) ────────
# Mean absolute pixel difference (0–255 scale on 64×64 thumbnails) to flag as a cut.
FRAME_DIFF_THRESHOLD = float(os.getenv("FRAME_DIFF_THRESHOLD", "30.0"))

# ── Podcast Tracking — Cut Deduplication ─────────────────────────────────────
# Minimum seconds between two accepted cuts across all detection layers.
DEDUP_GAP = float(os.getenv("DEDUP_GAP", "0.4"))

# ── Diarization ───────────────────────────────────────────────────────────────
# HuggingFace access token — required for pyannote/speaker-diarization-3.1.
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN", ""))
# Seconds to look ahead at visual cut boundaries when resolving the active speaker.
DIARIZATION_LOOKAHEAD = float(os.getenv("DIARIZATION_LOOKAHEAD", "0.1"))


# Padding: fraction of crop_w kept between face edge and frame border.
TRACKING_FACE_PADDING = float(os.getenv("TRACKING_FACE_PADDING",     "0.10"))

# ── Frame Extraction ──────────────────────────────────────────────────────────
TRACKING_FFMPEG_FRAME_EXTRACT = True
# Decode width for face detection frames. 960 px gives buffalo_l good input quality.
TRACKING_FRAME_EXTRACT_WIDTH = int(
    os.getenv("TRACKING_FRAME_EXTRACT_WIDTH", "960"))

# ONNX Runtime execution engine context definitions
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else [
    "CPUExecutionProvider"]

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

VIDEO_ENCODER = "h264_nvenc"
VIDEO_ENCODER_FALLBACK = "libx264"

# Crossfade transition length, in FRAMES. The duration in seconds is always
# derived dynamically as XFADE_FRAMES / fps (so 5 frames is 0.166 s @30fps and
# 0.200 s @25fps — never a hardcoded second value). Used at BOTH join levels.
XFADE_FRAMES = int(os.getenv("XFADE_FRAMES", "5"))

# Crossfade the scene-cut sub-segments WITHIN a single rendered clip segment.
CLIP_SEGMENT_XFADE = os.getenv("CLIP_SEGMENT_XFADE", "true").lower() == "true"

# Crossfade the joins BETWEEN a clip's AI-selected segments (extractor concat).
CLIP_JOIN_XFADE = os.getenv("CLIP_JOIN_XFADE", "true").lower() == "true"

# Debug overlay: burn the tracked face-detection bounding box into rendered
# clips. Drawn on the source frame BEFORE the crop (via sendcmd-animated
# drawbox), so the box shows exactly what the tracker locked onto relative to
# the moving camera. Never enable for production renders.
FACE_DEBUG_BOX = os.getenv("FACE_DEBUG_BOX", "false").lower() == "true"

# Hardware decoding / transcoding config
AUTO_TRANSCODE_TO_H264 = os.getenv("AUTO_TRANSCODE_TO_H264", "true").lower() == "true"
FFMPEG_HWACCEL = os.getenv("FFMPEG_HWACCEL", "cuda")

NVENC_PRESET = "p4"
NVENC_CQ = 23
X264_PRESET = "medium"
X264_CRF = 20

# GPU scheduler — per-job VRAM budget
GPU_MEMORY_RESERVE_MB = int(os.getenv("GPU_MEMORY_RESERVE_MB",        "0"))
GPU_MEMORY_PER_FFMPEG_JOB_MB = int(
    os.getenv("GPU_MEMORY_PER_FFMPEG_JOB_MB", "650"))
EXTRACTION_MAX_PARALLEL_JOBS = int(
    os.getenv("EXTRACTION_MAX_PARALLEL_JOBS",  "3"))
CAPTION_MAX_PARALLEL_JOBS = int(
    os.getenv("CAPTION_MAX_PARALLEL_JOBS",     "3"))

AUDIO_LOUDNESS_TARGET = -14
OUTPUT_VIDEO_FORMAT = "mp4"

# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND MUSIC
# ═══════════════════════════════════════════════════════════════════════════════

MUSIC_DIR = os.path.join(os.path.dirname(__file__), "music")
BACKGROUND_MUSIC_DEFAULT_VOLUME = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# CAPTION SETTINGS & RENDERER CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CAPTION_STYLE = "barlow_black"
CAPTION_BG_ENABLED = True
CAPTION_LOWERCASE = False
CAPTION_MAX_WORDS_ON_SCREEN = 3
CAPTION_USE_ASS = True
CAPTION_SAVE_SRT = False
CAPTION_KEEP_FILTER_FILES = False

MAX_CHARS_PER_LINE = int(os.getenv("MAX_CHARS_PER_LINE", "12"))
MIN_REVEAL_GAP = float(os.getenv("MIN_REVEAL_GAP", "0.16"))
SAFE_MARGIN_PCT = float(os.getenv("SAFE_MARGIN_PCT", "0.10"))
LONG_WORD_MARGIN_PCT = float(os.getenv("LONG_WORD_MARGIN_PCT", "0.13"))
LONG_WORD_WIDTH_SAFETY = float(os.getenv("LONG_WORD_WIDTH_SAFETY", "1.16"))
TWO_LINE_TOP_OFFSET_RATIO = float(
    os.getenv("TWO_LINE_TOP_OFFSET_RATIO", "0.18"))
TWO_LINE_GAP_RATIO = float(os.getenv("TWO_LINE_GAP_RATIO", "1.02"))
BORDER_RATIO = float(os.getenv("BORDER_RATIO", "0.038"))
MIN_BORDER_PX = int(os.getenv("MIN_BORDER_PX", "8"))

CAPTION_COMMON_STYLE = {
    "font_size":          165,   # px at 1920h — keep in sync with the UI default
    "max_words":          2,
    "font_color":         "#F5F0E0",
    "outline_color":      "#2B2100",
    "outline_width":      10,
    "highlight_color":    "#FFD100",
    "shadow_color":       "0x2B2100@0.55",
    "shadow_x":           0,
    "shadow_y":           6,
    "letter_spacing": -3,
    # default block bottom at ~78% from top = 22% above bottom
    "position_pct":       70,
    "safe_margin_pct":    0.10,
    "line_spacing_ratio": 0.92,   # Tight TikTok-style packing layout
    "active_word_color":  "",
    "entrance_anim":      "none",
    "word_case":          "upper",
}

# ── Typography Style Matrix Mapping ──────────────────────────────────────────
CAPTION_STYLES = {
    "anton_punch": {
        **CAPTION_COMMON_STYLE,
        "name": "Anton Punch",
        "font": "assets/fonts/Anton-Regular.ttf",
        "width_factor": 1.55,
    },
    "barlow_black": {
        **CAPTION_COMMON_STYLE,
        "name": "Barlow Black",
        "font": "assets/fonts/BarlowCondensed-Black.ttf",
        "width_factor": 1.05,
    },
    "urbanist_xbold": {
        **CAPTION_COMMON_STYLE,
        "name": "Urbanist ExtraBold",
        "font": "assets/fonts/Urbanist-ExtraBold.ttf",
        "width_factor": 1.85,
    },
    "lexend_xbold": {
        **CAPTION_COMMON_STYLE,
        "name": "Lexend ExtraBold",
        "font": "assets/fonts/Lexend-ExtraBold.ttf",
        "width_factor": 1.90,
    },
    "oswald_bold": {
        **CAPTION_COMMON_STYLE,
        "name": "Oswald Bold",
        "font": "assets/fonts/Oswald-Bold.ttf",
        "width_factor": 1.0,
    },
    "impact": {
        **CAPTION_COMMON_STYLE,
        "name": "Impact",
        "font": "C:/Windows/Fonts/impact.ttf",
        "width_factor": 1.0,
    },
    "bebas_neue": {
        **CAPTION_COMMON_STYLE,
        "name": "Bebas Neue",
        "font": "assets/fonts/BebasNeue-Regular.ttf",
        "width_factor": 1.0,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# BLOG POST SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Controlled toggles matching application runtime properties
BLOG_POST_ENABLED = False
BLOG_POST_TONE = "professional"
BLOG_POST_MAX_WORDS = 1500
AI_MAX_TOKENS_BLOG = 4096

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK SERVER
# ═══════════════════════════════════════════════════════════════════════════════

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FFMPEG_COMMANDS = True
KEEP_INTERMEDIATE_FILES = True
LOG_FSYNC_EACH_RECORD = os.getenv(
    "LOG_FSYNC_EACH_RECORD", "true").lower() == "true"

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_VIDEO_DURATION = 0
YTDLP_RETRIES = 3
YTDLP_COOKIES_BROWSER = None

# AI Analysis Stage Overrides
# Judge pass is OFF by default — ranking matters less than discovery, the
# in-tree _build_judge_prompt currently NameErrors anyway (lives only in unused
# pipeline/patch.py), and disabling it preserves more candidates so "in-between"
# gems aren't dropped before clips_plan.
SKIP_JUDGE = True
SKIP_VALIDATION = False
SKIP_DUPLICATION = False
SKIP_API_CACHE_DURING_EXTRACTION = os.getenv("SKIP_API_CACHE_DURING_EXTRACTION", "true").lower() == "true"

# Clip Tool v4 tracking cadence settings
V4_POST_CUT_BOOST_DURATION = float(os.getenv("V4_POST_CUT_BOOST_DURATION", "0.4"))
V4_POST_CUT_BOOST_FRAMES = int(os.getenv("V4_POST_CUT_BOOST_FRAMES", "3"))

# External LLM Generic Filler Hashtags
GENERIC_FILLER_HASHTAGS = "#viral #fyp #shorts #trending #foryou #foryoupage #explore #reels #youtubeshorts #viralvideo #motivation #mindset #trendingshorts #subscribe #podcast #podcastclips #viralshorts #shortsfeed #reelsviral #trend #explorepage #contentcreator #inspiration #selfimprovement #growth #wellness #health #healthtips #lifehacks #knowledge #learnontiktok #educational #facts #science #biohacking #longevity #wellbeing #mentalhealth #dailymotivation #success"
