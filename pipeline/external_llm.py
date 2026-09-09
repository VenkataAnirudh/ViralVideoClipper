"""
Manual / external-LLM analysis route.

This is the second analysis route (the first being the live-API route in
analyzer.analyze_transcript). Instead of calling an AI provider, the pipeline:

  1. Pauses after transcription and emits, per job:
       • external_llm_windows.txt  — the time-coded transcript windows, already
         split + labelled, ready to paste one-at-a-time into any chat LLM.
       • (pipeline/prompts.txt is the fixed system prompt the user pastes once.)
  2. Lets the user run those by hand and paste every window's JSON back into the
     dashboard, which saves it to <job>/external_llm.txt.
  3. Ingests that file here: tolerant-parse → assign candidate_ids → drop
     near-duplicates from the overlapping windows (keep the higher-scored) →
     hand the candidates to the SAME validate/filter/match path the API route
     uses, so local_clips_generator still pins word-exact times.

The fused manual prompt already produces boundaries + full YouTube metadata, so
the discovery / refinement / judge API passes are skipped entirely.

This module is intentionally dependency-light (stdlib only) and imports analyzer
lazily to avoid an import cycle (analyzer imports this module lazily in turn).
"""

from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher

import config


# ── Paths / presence ─────────────────────────────────────────────────────────

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def external_llm_input_path(job_dir: str) -> str:
    """Per-job file the user's pasted LLM output is saved to."""
    return os.path.join(
        job_dir, getattr(config, "EXTERNAL_LLM_INPUT_FILE", "external_llm.txt")
    )


def external_llm_windows_path(job_dir: str) -> str:
    """Per-job file the paste-ready, time-coded windows are written to."""
    return os.path.join(
        job_dir, getattr(config, "EXTERNAL_LLM_WINDOWS_FILE", "external_llm_windows.txt")
    )


def has_external_llm_input(job_dir: str) -> bool:
    """True when the user has supplied a non-empty external_llm.txt for this job."""
    path = external_llm_input_path(job_dir)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return bool(handle.read().strip())
    except OSError:
        return False


def read_external_llm_input(job_dir: str) -> str:
    path = external_llm_input_path(job_dir)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def save_external_llm_input(job_dir: str, text: str) -> str:
    """Persist the user's pasted output to <job>/external_llm.txt."""
    os.makedirs(job_dir, exist_ok=True)
    path = external_llm_input_path(job_dir)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text or "")
    return path


def read_system_prompt() -> str:
    """Return the fixed system prompt text (pipeline/prompts.txt)."""
    rel = getattr(config, "EXTERNAL_LLM_SYSTEM_PROMPT_FILE", os.path.join("pipeline", "prompts.txt"))
    path = rel if os.path.isabs(rel) else os.path.join(_project_root(), rel)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def read_windows_text(job_dir: str) -> str:
    path = external_llm_windows_path(job_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


# ── Prepare the paste-ready windows (called at the pause point) ───────────────

def prepare_external_llm_inputs(job_dir, meta, transcript, settings, logger) -> tuple[str, int]:
    """Build the time-coded windows file the user pastes into their LLM.

    Returns (windows_path, window_count). Windowing matches the API route exactly
    (same window/overlap seconds), so the manual output lines up with how the
    automated discovery pass would have chunked the transcript.
    """
    from pipeline import analyzer  # lazy: avoid import cycle

    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    if not segments:
        raise ValueError("Cannot prepare external-LLM windows from an empty transcript")

    window_sec = float(settings.get("analysis_window_seconds")
                       or getattr(config, "AI_ANALYSIS_WINDOW_SECONDS", 900.0))
    overlap_sec = float(settings.get("analysis_overlap_seconds")
                        or getattr(config, "AI_ANALYSIS_OVERLAP_SECONDS", 180.0))
    if window_sec < 60.0:
        window_sec = 60.0
    if overlap_sec >= window_sec:
        overlap_sec = window_sec * 0.2
    if overlap_sec < 0.0:
        overlap_sec = 0.0

    chunks = analyzer._build_transcript_windows(segments, window_sec, overlap_sec)
    # Keep the AI-style transcript view in sync for parity with the API route.
    try:
        analyzer._save_analysis_transcript(job_dir, chunks, logger)
    except Exception:
        pass

    min_dur = int(settings.get("min_duration", getattr(config, "DEFAULT_MIN_DURATION", 30)) or 30)
    max_dur = int(settings.get("max_duration", getattr(config, "DEFAULT_MAX_DURATION", 90)) or 90)

    lines: list[str] = []
    lines.append("# MANUAL ANALYSIS — paste pipeline/prompts.txt as your LLM's SYSTEM prompt,")
    lines.append("# then send each '=== AI ANALYSIS WINDOW n ===' block below as a separate user message.")
    lines.append("# Collect every window's JSON output and paste them ALL into the dashboard's")
    lines.append("# 'Manual analysis' box, then resume. A Python matcher resolves exact clip times")
    lines.append("# from your verbatim start_words/end_words, so you do not compute any clock times.")
    lines.append("")
    lines.append("VIDEO METADATA")
    lines.append(f"Title       : {meta.get('title', 'Unknown')}")
    lines.append(f"Channel     : {meta.get('channel', 'Unknown')}")
    lines.append(f"Duration    : {meta.get('duration', 0)} seconds")
    lines.append(f"Description : {str(meta.get('description', 'N/A'))[:500]}")
    lines.append(f"Target band : ~{min_dur}-{max_dur}s per clip (rough guide, not a cap)")
    lines.append(f"Windows     : {len(chunks)}")
    lines.append("")
    for idx, start, end, chunk in chunks:
        lines.append(
            f"=== AI ANALYSIS WINDOW {idx + 1} of {len(chunks)}: "
            f"{analyzer._fmt_time(start)} -> {analyzer._fmt_time(end)} ==="
        )
        lines.append(analyzer._format_transcript(chunk))
        lines.append("")

    path = external_llm_windows_path(job_dir)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logger.info(f"[external-llm] wrote {len(chunks)} paste-ready window(s) -> {path}")
    return path, len(chunks)


# ── Tolerant JSON extraction ─────────────────────────────────────────────────

_OPENERS = "{["
_CLOSERS = "}]"
_PAIRS = {"}": "{", "]": "["}

# Smart/curly quotes and invisible characters that browsers + chat UIs inject
# when the user copies JSON out. These break json.loads AND desync the balanced
# walker's string tracking (which keys off the ASCII " ), even when the object's
# structure is otherwise fine. Indentation / blank lines are already JSON-legal
# and are deliberately left untouched.
_SMART_DOUBLE = "“”„‟″«»"   # “ ” „ ‟ ″ « »
_SMART_SINGLE = "‘’‚‛′`´"   # ‘ ’ ‚ ‛ ′ ` ´
_TRANSLATION = {
    **{ord(ch): '"' for ch in _SMART_DOUBLE},
    **{ord(ch): "'" for ch in _SMART_SINGLE},
    0x00A0: " ",    # non-breaking space
    0x202F: " ",    # narrow no-break space
    0x2007: " ",    # figure space
    0x200B: None,   # zero-width space
    0x200C: None,   # zero-width non-joiner
    0x200D: None,   # zero-width joiner
    0xFEFF: None,   # BOM / zero-width no-break space
}


def _presanitize(text: str) -> str:
    """Normalise the curly-quote / invisible-character corruption that copy-paste
    from chat UIs introduces, so both the balanced walker and json.loads see
    clean ASCII delimiters. Structure-preserving: indentation, blank lines, and
    the text content itself are untouched apart from these substitutions."""
    if not text:
        return ""
    return text.translate(_TRANSLATION)


def _match_balanced(text: str, start: int) -> int | None:
    """Return the index of the close that balances the opener at `start`,
    respecting JSON string literals + escapes. None if unbalanced."""
    stack: list[str] = []
    in_str = False
    esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in _OPENERS:
            stack.append(c)
        elif c in _CLOSERS:
            if not stack or stack[-1] != _PAIRS[c]:
                return None
            stack.pop()
            if not stack:
                return j
    return None


def _strip_json_comments(s: str) -> str:
    """Remove ``//`` line comments and ``/* */`` block comments that sit OUTSIDE
    string literals. String-aware (respects ``\\`` escapes) so a ``//`` inside a
    value — e.g. an ``https://`` URL — is preserved. Handles both whole-line and
    inline (trailing) comments, which LLMs add against the schema's hints."""
    out: list[str] = []
    i, n = 0, len(s)
    in_str = False
    esc = False
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "/":
            i += 2
            while i < n and s[i] != "\n":     # drop to end of line, keep the \n
                i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _loads_tolerant(snippet: str):
    """json.loads with progressive repair for the common manual-paste sins.

    Tier 0/1: strict, then ``strict=False`` — the latter accepts literal control
              characters (raw newlines / tabs) inside string values, which LLMs
              routinely leave in multi-line transcript fields.
    Tier 2:   strip ``//`` and ``/* */`` comments (whole-line AND inline, the
              schema's hints) then trailing commas, and retry lenient.
    (Curly quotes + invisible characters are already fixed upstream by
    ``_presanitize`` before the balanced walker even runs.)
    """
    for strict in (True, False):
        try:
            return json.loads(snippet, strict=strict)
        except Exception:
            pass
    repaired = _strip_json_comments(snippet)                      # // and /* */ (string-aware)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)            # trailing commas
    try:
        return json.loads(repaired, strict=False)
    except Exception:
        return None


def extract_json_objects(text: str) -> list:
    """Walk the pasted blob and pull every balanced top-level JSON value.

    Non-JSON text between objects (window headers, the VIDEO METADATA preamble,
    stray prose, ``` fences) is skipped naturally. Handles: many concatenated
    `{...}` window objects, a single top-level `[...]` array, or fenced blocks.
    """
    objects = []
    text = _presanitize(text or "")
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in _OPENERS:
            end = _match_balanced(text, i)
            if end is not None:
                parsed = _loads_tolerant(text[i:end + 1])
                if parsed is not None:
                    objects.append(parsed)
                    i = end + 1
                    continue
        i += 1
    return objects


def _clips_from_obj(obj) -> list[dict]:
    """Pull the clip dicts out of one parsed JSON value."""
    if isinstance(obj, list):
        clips: list[dict] = []
        for item in obj:
            clips.extend(_clips_from_obj(item))
        return clips
    if isinstance(obj, dict):
        for key in ("clips", "candidates", "clip_candidates", "selected_clips"):
            if isinstance(obj.get(key), list):
                return [c for c in obj[key] if isinstance(c, dict)]
        # A bare clip object (no wrapper)?
        if obj.get("segments") or obj.get("start_words") or obj.get("youtube_title"):
            return [obj]
    return []


# ── Normalisation + similarity dedup ─────────────────────────────────────────

_SCORE_KEYS = (
    "hook_score", "flow_score", "virality_score",
    "meaning_score", "completeness_score", "boundary_score",
)


def _score_of(cand: dict) -> float:
    total = 0.0
    for key in _SCORE_KEYS:
        value = cand.get(key)
        try:
            total += float(value)
        except (TypeError, ValueError):
            match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
            total += float(match.group(0)) if match else 5.0
    return total


def _normalize_clip_to_candidate(clip: dict, window: int, local_idx: int) -> dict:
    """Map a manual `clips[]` entry onto the candidate schema the rest of the
    pipeline expects. Keeps every field the manual prompt emits; the heavy
    normalisation (scores, hashtags string→list, segment frame→time synthesis)
    is done later by analyzer._validate_analysis."""
    cand = dict(clip)
    cand["candidate_id"] = f"w{window:02d}_c{local_idx:02d}"
    cand["source_window"] = window
    cand["source"] = "external_llm"
    yt = str(cand.get("youtube_title") or "").strip()
    takeaway = str(cand.get("takeaway") or "").strip()
    if not str(cand.get("working_title") or "").strip():
        cand["working_title"] = (yt or takeaway or f"Clip {window}-{local_idx}")[:120]
    if not str(cand.get("title") or "").strip():
        cand["title"] = cand["working_title"]
    return cand


def _clip_text_key(cand: dict) -> str:
    """Normalised stitched-text key used for similarity dedup."""
    text = str(cand.get("clip_transcript") or "").strip()
    if not text:
        parts: list[str] = []
        for seg in cand.get("segments") or []:
            if isinstance(seg, dict):
                parts.append(str(seg.get("start_words") or ""))
                parts.append(str(seg.get("end_words") or ""))
        text = " ".join(parts)
    if not text:
        text = str(cand.get("takeaway") or cand.get("youtube_title") or "")
    key = re.sub(r"[^a-z0-9 ]", "", text.lower())
    key = re.sub(r"\s+", " ", key).strip()
    return key[:600]


def dedupe_by_similarity(candidates: list[dict], threshold: float, logger: logging.Logger) -> list[dict]:
    """Drop a clip only when it is >= `threshold` similar to a higher-scored one.

    Overlapping windows re-surface the same moment with restarted ranks; this is
    the 'ranks redundancy' fix — sort by combined score desc and keep the best
    representative of each near-duplicate group.
    """
    ranked = sorted(candidates, key=_score_of, reverse=True)
    kept: list[dict] = []
    kept_keys: list[str] = []
    dropped = 0
    for cand in ranked:
        key = _clip_text_key(cand)
        is_dup = False
        if key:
            for existing_key in kept_keys:
                if not existing_key:
                    continue
                # Cheap length gate before the O(n*m) ratio.
                shorter, longer = sorted((len(key), len(existing_key)))
                if longer and shorter / longer < threshold:
                    continue
                if SequenceMatcher(None, key, existing_key).ratio() >= threshold:
                    is_dup = True
                    break
        if is_dup:
            dropped += 1
            continue
        kept.append(cand)
        kept_keys.append(key)
    if dropped:
        logger.info(
            f"[external-llm] dropped {dropped} near-duplicate clip(s) "
            f"(>= {threshold:.0%} similar); kept the higher-scored one"
        )
    return kept


def aggregate_external_candidates(
    text: str,
    logger: logging.Logger,
    similarity_threshold: float | None = None,
) -> tuple[list[dict], list[str]]:
    """Parse + normalise + dedupe the pasted output.

    Returns (candidates, window_summaries). `candidates` carry candidate_id and
    the manual metadata/segments; they are NOT yet validated — feed them through
    analyzer._validate_analysis next.
    """
    objects = extract_json_objects(text)
    candidates: list[dict] = []
    window_summaries: list[str] = []
    window = 0
    for obj in objects:
        clips = _clips_from_obj(obj)
        if isinstance(obj, dict) and obj.get("window_summary"):
            window_summaries.append(str(obj.get("window_summary")).strip())
        if not clips:
            continue
        window += 1
        for local_idx, clip in enumerate(clips, 1):
            candidates.append(_normalize_clip_to_candidate(clip, window, local_idx))

    logger.info(
        f"[external-llm] parsed {len(objects)} JSON value(s) -> "
        f"{window} window(s), {len(candidates)} raw clip(s)"
    )
    if not candidates:
        return [], window_summaries

    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else float(getattr(config, "EXTERNAL_LLM_SIMILARITY_THRESHOLD", 0.90))
    )
    deduped = dedupe_by_similarity(candidates, threshold, logger)
    return deduped, window_summaries
