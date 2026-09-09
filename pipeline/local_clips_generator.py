"""Local clips generator — the SOLE word-exact matcher for the pipeline.

Two responsibilities:
  1. Given AI candidates that describe each segment by (start_words, start_frame,
     end_words, end_frame), pin word-exact start/end times against the
     word-level transcript.json. Supports both:
       • NEW schema (preferred): per-segment {start_words, start_frame,
         end_words, end_frame}. start_frame / end_frame are
         "[MM:SS.ss -> MM:SS.ss]" line-range hints; matching widens by ±N
         transcript lines when the verbatim phrase isn't found inside the cited
         frame (e.g. the phrase spans a segment break).
       • LEGACY schema: a single "content" field encoding the same anchors as
         'Start: "..." [Timeframe: [...]] | End: "..." [Timeframe: [...]]'.
         Old saved jobs and old prompts still emit this; kept working for
         resume-compat.

  2. Local fallback when the AI clip selection stage fails — parse
     refined_candidates.txt / eligible_candidates.txt directly into a
     clips_plan.json.

Identity is keyed on candidate_id; we NEVER pair AI candidates to clip plans by
list index. The previous behaviour (clip_alignment.py) silently mis-paired
after dedupe / overlap-removal / score-resort and caused timestamps to drift.
clip_alignment.py is now retired; its two app.py helpers
(flatten_transcript_words, _write_clip_info) live here.
"""
from __future__ import annotations

import bisect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Time + token helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_time_to_seconds(t_str: str) -> float:
    """Convert MM:SS.cc or HH:MM:SS.cc timestamp strings into total seconds."""
    s = str(t_str or "").strip()
    if not s:
        return 0.0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, TypeError):
        return 0.0


def clean_word(word: str) -> str:
    """Normalize a token for phrase matching."""
    return re.sub(r"[^\w]", "", str(word or "").lower().strip())


def has_sentence_end(word_str: str) -> bool:
    return bool(re.search(r"[.!?]", str(word_str or "")))


def extract_tokens(text: str) -> List[str]:
    return [clean_word(tok) for tok in re.findall(r"\S+", str(text or "")) if clean_word(tok)]


def boundary_match_phrase(phrase: str, side: str) -> str:
    """If a stored phrase contains '...' or '…' (ellipsis), keep the start of the
    start-side fragment or the end of the end-side fragment for matching."""
    parts = [p.strip() for p in re.split(r"\.\.\.|…", str(phrase or "")) if p.strip()]
    if not parts:
        return str(phrase or "").strip()
    if len(parts) == 1:
        return parts[0]
    return parts[0] if side == "start" else parts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Word stream + WordToken (migrated from clip_alignment for app.py editor)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordToken:
    text: str
    norm: str
    start: float
    end: float
    index: int


def _iter_word_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "word" in obj and ("start" in obj or "start_time" in obj) and ("end" in obj or "end_time" in obj):
            yield obj
        for value in obj.values():
            yield from _iter_word_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_word_dicts(value)


def flatten_transcript_words(transcript: Any) -> List[WordToken]:
    """Stable word stream for app.py /api/clips-plan editor — same shape the
    legacy clip_alignment.flatten_transcript_words returned."""
    words: List[WordToken] = []
    for raw in _iter_word_dicts(transcript):
        text = str(raw.get("word") or raw.get("text") or "").strip()
        norm = clean_word(text)
        if not norm:
            continue
        try:
            start = float(raw.get("start", raw.get("start_time")))
            end = float(raw.get("end", raw.get("end_time", start)))
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        words.append(WordToken(text=text, norm=norm, start=start, end=end, index=len(words)))
    return words


def flatten_transcript_json(transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Word stream used by the local matcher — dict shape kept for back-compat
    with the older parts of this module."""
    all_words: List[Dict[str, Any]] = []
    for seg in transcript_data.get("segments", []) or []:
        for w in seg.get("words", []) or []:
            token = str(w.get("word", "")).strip()
            all_words.append({
                "word": token,
                "clean": clean_word(token),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })
    all_words.sort(key=lambda x: x["start"])
    return all_words


def _nearest_index_by_time(words: Sequence[Dict[str, Any]], target_time: float) -> int:
    if not words:
        return 0
    starts = [w["start"] for w in words]
    idx = bisect.bisect_left(starts, target_time)
    if idx <= 0:
        return 0
    if idx >= len(words):
        return len(words) - 1
    prev_idx = idx - 1
    return idx if abs(words[idx]["start"] - target_time) < abs(words[prev_idx]["start"] - target_time) else prev_idx


# ─────────────────────────────────────────────────────────────────────────────
# Phrase → word-exact time matcher
# ─────────────────────────────────────────────────────────────────────────────

def find_best_phrase_match(
    words: Sequence[Dict[str, Any]],
    phrase: str,
    approx_time: Optional[float] = None,
    search_seconds: float = 120.0,
    min_ratio: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """Sliding-window best match for a phrase inside a word stream.

    Tolerant by design: the AI sometimes paraphrases by 1-2 tokens, so
    min_ratio defaults to 0.7. The window also widens its scan by
    ±search_seconds around any time hint."""
    phrase_tokens = extract_tokens(phrase)
    if not phrase_tokens or not words:
        return None

    window_size = len(phrase_tokens)
    search_start_idx = 0
    search_end_idx = len(words)

    if approx_time is not None:
        search_start_idx = max(0, _nearest_index_by_time(words, approx_time - search_seconds))
        search_end_idx = min(len(words), _nearest_index_by_time(words, approx_time + search_seconds) + window_size)

    best_match = None
    best_ratio = 0.0
    for i in range(search_start_idx, max(search_start_idx + 1, search_end_idx - window_size + 1)):
        window = words[i: i + window_size]
        if len(window) < window_size:
            break
        matches = sum(1 for w, pt in zip(window, phrase_tokens) if w["clean"] == pt)
        ratio = matches / window_size
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = {
                "start_idx": i,
                "end_idx": i + window_size - 1,
                "start_time": window[0]["start"],
                "end_time": window[-1]["end"],
                "ratio": ratio,
            }
            if ratio == 1.0:
                break

    return best_match if best_ratio >= min_ratio else None


# ─────────────────────────────────────────────────────────────────────────────
# Frame parsing  [MM:SS.ss -> MM:SS.ss]  (optionally bracketed)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_frame(text: str) -> Optional[Tuple[float, float]]:
    """Parse '[MM:SS.ss -> MM:SS.ss]' or 'MM:SS.ss -> MM:SS.ss' to (start, end)."""
    if not text:
        return None
    cleaned = re.sub(r"[\[\]]", "", str(text)).strip()
    if "->" not in cleaned:
        return None
    a, b = cleaned.split("->", 1)
    try:
        return parse_time_to_seconds(a), parse_time_to_seconds(b)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NEW schema parser:  per-segment {start_words, start_frame, end_words, end_frame}
# ─────────────────────────────────────────────────────────────────────────────

def parse_candidate_segments(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull anchor data from a candidate using the new schema first, falling
    back to the legacy 'content' field. Returns list of:
        {start_phrase, end_phrase, start_time_hint, end_time_hint,
         is_legacy_range}

    Legacy {"start": x, "end": y} segments (produced by analyzer's stitch
    extension when the AI returned fewer segments than the input count) are
    now accepted: they get a `is_legacy_range=True` flag and exact start/end
    times in the hints. The matcher uses those times directly without
    phrase-matching.
    """
    out: List[Dict[str, Any]] = []

    raw_segments = cand.get("segments") or []
    for s in raw_segments:
        if not isinstance(s, dict):
            continue
        # NEW schema fields take precedence.
        if "start_words" in s or "end_words" in s or "start_frame" in s or "end_frame" in s:
            start_phrase = str(s.get("start_words") or "").strip()
            end_phrase = str(s.get("end_words") or "").strip()
            sf = _parse_frame(s.get("start_frame", ""))
            ef = _parse_frame(s.get("end_frame", ""))
            if not start_phrase and not end_phrase:
                continue
            out.append({
                "start_phrase": start_phrase,
                "end_phrase": end_phrase or start_phrase,
                "start_time_hint": sf[0] if sf else None,
                "end_time_hint": ef[1] if ef else (sf[1] if sf else None),
                "is_legacy_range": False,
            })
            continue
        # LEGACY-RANGE schema: stitch-extension segments carry only numeric
        # start/end. Accept them so the analyzer's duration extensions are
        # not silently dropped by downstream parsing.
        legacy_start = s.get("start")
        legacy_end = s.get("end")
        if legacy_start is not None and legacy_end is not None:
            try:
                ls = float(legacy_start)
                le = float(legacy_end)
            except (TypeError, ValueError):
                continue
            if le > ls:
                out.append({
                    "start_phrase": "",
                    "end_phrase": "",
                    "start_time_hint": ls,
                    "end_time_hint": le,
                    "is_legacy_range": True,
                })

    if out:
        return out

    # Legacy fallback: parse the single 'content' string.
    content_str = str(cand.get("content") or "").strip()
    if content_str:
        return parse_content_string(content_str)

    return out


def parse_content_string(content_str: str) -> List[Dict[str, Any]]:
    """Legacy single-line schema:
        Start: "..." [Timeframe: [MM:SS.ss -> MM:SS.ss]] | End: "..." [Timeframe: [MM:SS.ss -> MM:SS.ss]]
        Stitched: same, joined by '||'.
    """
    segments: List[Dict[str, Any]] = []
    parts = str(content_str or "").split("||")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(r"^\[Seg \d+\]\s*", "", part)
        match = re.search(
            r'Start:\s*"(?P<start>[^"]+)"\s*(?:\[Timeframe:\s*\[(?P<start_window>[^\]]+)\]\])?\s*\|\s*End:\s*"(?P<end>[^"]+)"\s*(?:\[Timeframe:\s*\[(?P<end_window>[^\]]+)\]\])?',
            part,
            re.IGNORECASE,
        )
        if not match:
            continue
        seg = {
            "start_phrase": match.group("start").strip(),
            "end_phrase": match.group("end").strip(),
            "start_time_hint": None,
            "end_time_hint": None,
        }
        sw = match.group("start_window")
        if sw and "->" in sw:
            try:
                seg["start_time_hint"] = parse_time_to_seconds(sw.split("->")[0].strip())
            except (ValueError, TypeError):
                pass
        ew = match.group("end_window")
        if ew and "->" in ew:
            try:
                seg["end_time_hint"] = parse_time_to_seconds(ew.split("->")[1].strip())
            except (ValueError, TypeError):
                pass
        segments.append(seg)
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Boundary location with ±N-line / time-window nearby fallback
# ─────────────────────────────────────────────────────────────────────────────

# How far to widen if the phrase isn't found inside the cited frame.
# We widen in time (seconds) — that's effectively ±N transcript lines because
# segments are sub-second to a few seconds long.
_NEARBY_TIME_WIDTHS = (15.0, 45.0, 120.0)  # progressive widening

# Leading-silence trim threshold. If the matched start_words window has an
# inter-word gap larger than this (the speaker thought/uh'd/paused in dead
# air), the resolved start is moved forward past the gap.
_LEADING_SILENCE_GAP_S = 0.8

# Sentence-terminator snap windows. Defaults overridable via config.py / env.
# After end_words is matched, the snap tries to land on a clean ".!?" boundary
# in two passes: forward first, then backward as a fallback when forward fails.
# See config.END_SENTENCE_{LOOKAHEAD,LOOKBACK,SILENCE_BAIL}_S for tuning notes.
try:
    import config as _cfg
    # Lookahead is deliberately SHORT: the AI now picks end_words ON the payoff,
    # so the snap only completes the current clause — it must never sail seconds
    # into the next thought/topic chasing a period (that buried payoffs and
    # shipped cliffhangers). A pause counts as a soft boundary so unpunctuated
    # transcripts still snap cleanly.
    _END_SENTENCE_LOOKAHEAD_S = float(getattr(_cfg, "END_SENTENCE_LOOKAHEAD_S", 4.0))
    _END_SENTENCE_LOOKBACK_S = float(getattr(_cfg, "END_SENTENCE_LOOKBACK_S", 6.0))
    _END_SENTENCE_SILENCE_BAIL_S = float(getattr(_cfg, "END_SENTENCE_SILENCE_BAIL_S", 2.5))
    _END_SENTENCE_PAUSE_S = float(getattr(_cfg, "END_SENTENCE_PAUSE_S", 0.45))
except Exception:
    _END_SENTENCE_LOOKAHEAD_S = 4.0
    _END_SENTENCE_LOOKBACK_S = 6.0
    _END_SENTENCE_SILENCE_BAIL_S = 2.5
    _END_SENTENCE_PAUSE_S = 0.45

# A clip must NOT OPEN on one of these — leading discourse-filler / connectors
# that aren't the hook. After matching the start, we walk forward off a short
# leading run of these to the first real word. (Articles/prepositions are left
# out — trimming "the/of" can cut into content; the AI handles those semantic
# jumps. This only kills obvious filler like "yeah um", "and", "so".)
_LEADING_FILLER_WORDS = {
    "and", "but", "or", "so", "yeah", "um", "uh", "well", "okay", "right",
    "anyway", "basically", "like", "now",
}

# A clip must NOT end on one of these — they signal the thought/phrase isn't
# finished (connectors, articles, possessives, prepositions, auxiliaries,
# intensifiers, determiners, filler, subject contractions). The snap does TWO
# things with this set: (1) it refuses to STOP on one (it keeps extending
# forward to complete the phrase), and (2) as a last resort it trims a trailing
# run of them. The goal is a clip that ends on a COMPLETE, learnable point.
_INCOMPLETE_END_WORDS = {
    # connectors
    "and", "but", "or", "nor", "yet", "so", "because", "since", "although", "though",
    "while", "when", "which", "who", "whom", "that", "as", "than", "if",
    # articles / possessives / determiners
    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their", "whose",
    "this", "these", "those",
    # prepositions
    "to", "of", "for", "with", "in", "into", "on", "at", "from", "by", "about", "like",
    # auxiliaries
    "is", "was", "are", "were", "be", "been", "am", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "may", "might", "must",
    # intensifiers / quantifiers (can't end a thought)
    "very", "really", "such", "more", "less", "most", "least", "quite", "rather", "even",
    # filler
    "um", "uh", "yeah", "well", "just", "kind", "sort", "okay", "right", "anyway", "basically", "now",
    # subject contractions / bare pronoun
    "i", "im", "ill", "id", "ive", "its", "thats", "youre", "youll", "theyre", "theyll",
    "hes", "shes", "weve",
}
# Backwards-compatible alias (older references).
_DANGLING_END_WORDS = _INCOMPLETE_END_WORDS


def _trim_leading_silence(
    start_match: Dict[str, Any],
    all_words: Sequence[Dict[str, Any]],
) -> float:
    """If the matched start window has dead air between its first words,
    advance the resolved start to the word AFTER the first big gap.

    Example seen on Video 32 / clip_01:
      AI start_words: "percent? I took it like 44 to like 60"
      raw transcript: "percent?" ends 477.56s, then 3.23s of silence, then "I" at 480.79s
    Without this, the clip head shows "percent" + 3 s of dead air ("Ummmmm").
    With this, exact_start moves to 480.79 s (start of "I").
    """
    s_idx = int(start_match.get("start_idx", 0))
    e_idx = int(start_match.get("end_idx", s_idx))
    if e_idx <= s_idx or e_idx >= len(all_words):
        return float(start_match["start_time"])

    new_start = float(start_match["start_time"])
    for i in range(s_idx, min(e_idx, len(all_words) - 1)):
        cur = all_words[i]
        nxt = all_words[i + 1]
        try:
            gap = float(nxt["start"]) - float(cur["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if gap > _LEADING_SILENCE_GAP_S:
            new_start = float(nxt["start"])
            break
    return new_start


def _trim_leading_filler(
    start_match: Dict[str, Any],
    all_words: Sequence[Dict[str, Any]],
    max_steps: int = 4,
) -> float:
    """Advance the start past a short leading run of discourse-filler/connector
    words (yeah, um, and, so, ...) so the clip opens on the hook, not on filler.
    Capped at max_steps so it never eats into the real content."""
    s_idx = int(start_match.get("start_idx", 0))
    new_start = float(start_match.get("start_time", 0.0))
    k = s_idx
    steps = 0
    while k < len(all_words) - 1 and steps < max_steps:
        if clean_word(all_words[k].get("word", "")) in _LEADING_FILLER_WORDS:
            k += 1
            steps += 1
            try:
                new_start = float(all_words[k]["start"])
            except (KeyError, TypeError, ValueError):
                break
        else:
            break
    return new_start


def _snap_end_to_sentence(
    end_match: Dict[str, Any],
    all_words: Sequence[Dict[str, Any]],
    lookahead_s: float = _END_SENTENCE_LOOKAHEAD_S,
    lookback_s: float = _END_SENTENCE_LOOKBACK_S,
    silence_bail_s: float = _END_SENTENCE_SILENCE_BAIL_S,
    pause_s: float = _END_SENTENCE_PAUSE_S,
) -> float:
    """Land the cut on the nearest clean boundary to the AI's chosen end, and
    never end on a dangling connector/filler word.

    The AI now selects end_words ON the payoff (not "near a period"), so the
    matcher's job is only to tidy the cut to the closest natural break — NOT to
    go hunting for a far-off sentence terminator. A "boundary" is a word that
    ends on ``.``/``!``/``?`` OR is followed by a real pause (>= ``pause_s``),
    so unpunctuated transcripts still snap to a natural breath.

    Strategy — minimal movement, forward-first, both windows short:
      Pass 0: end_words already ends on a terminator → keep that word.
      Pass 1 (forward, <= lookahead_s ~4s): complete the current clause by
              landing on the next boundary. Bails on silence > silence_bail_s
              so it never crosses dead air into a new topic.
      Pass 2 (backward, <= lookback_s ~6s): only if nothing forward — retreat
              to the most recent boundary so we never ship mid-word.
      Fallback: the AI's exact end.

    Every return path runs through ``_finalize``, which walks back off any
    trailing run of dangling words (and / but / the / to / um / ...) so the clip
    never closes on an obviously-incomplete word.
    """
    base_end = float(end_match["end_time"])
    e_idx = int(end_match.get("end_idx", -1))
    if e_idx < 0 or e_idx >= len(all_words):
        return base_end

    def _incomplete(j: int) -> bool:
        return clean_word(all_words[j].get("word", "")) in _INCOMPLETE_END_WORDS

    def _finalize(idx: int) -> float:
        """Last-resort trim: walk back off a trailing run of incomplete words."""
        k = idx
        steps = 0
        while k > 0 and steps < 8 and _incomplete(k):
            k -= 1
            steps += 1
        try:
            return float(all_words[k]["end"])
        except (KeyError, TypeError, ValueError):
            return base_end

    def _is_boundary(j: int) -> bool:
        """Word j sits at a natural break: a terminator, or a real pause after it."""
        tok = str(all_words[j].get("word", ""))
        if tok and tok.rstrip()[-1:] in {".", "!", "?"}:
            return True
        if j + 1 < len(all_words):
            try:
                gap = float(all_words[j + 1]["start"]) - float(all_words[j]["end"])
                if gap >= pause_s:
                    return True
            except (KeyError, TypeError, ValueError):
                return False
        return False

    def _is_complete_end(j: int) -> bool:
        """A GOOD place to end: a natural break that lands on a content word —
        not on an article/preposition/aux/filler fragment ('...for your')."""
        return _is_boundary(j) and not _incomplete(j)

    # Pass 0 — the AI's end already lands on a CONTENT word (not a fragment):
    # trust it and just trim any trailing filler. We do NOT extend here — on a
    # pause-less transcript a forward hunt can drift; the AI now picks complete
    # ends, so only FRAGMENT ends below need fixing.
    if not _incomplete(e_idx):
        return _finalize(e_idx)

    # Pass 1 — forward: AI ended on a FRAGMENT — EXTEND to the next COMPLETE end so the phrase finishes
    # (length is flexible — a complete, learnable point beats a short fragment).
    # Bails on a big silence gap so it never crosses into a new topic.
    deadline = base_end + lookahead_s
    for j in range(e_idx + 1, len(all_words)):
        w = all_words[j]
        try:
            w_start = float(w["start"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            prev_end = float(all_words[j - 1]["end"])
        except (KeyError, TypeError, ValueError):
            prev_end = w_start
        if w_start - prev_end > silence_bail_s:
            break  # dead air ahead — refuse to extend through it
        if w_start > deadline:
            break
        if _is_complete_end(j):
            return _finalize(j)

    # Pass 2 — backward: retreat to the most recent COMPLETE end.
    earliest = base_end - lookback_s
    for j in range(e_idx, -1, -1):
        w = all_words[j]
        try:
            w_end = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if w_end < earliest:
            break
        if _is_complete_end(j):
            return _finalize(j)

    # Nothing clean nearby — keep the AI's end but trim any trailing fragment.
    return _finalize(e_idx)


def locate_exact_boundaries(
    start_phrase: str,
    end_phrase: str,
    all_words: Sequence[Dict[str, Any]],
    start_time_hint: Optional[float] = None,
    end_time_hint: Optional[float] = None,
    is_legacy_range: bool = False,
) -> Optional[Dict[str, Any]]:
    """Find the absolute exact start and end times for a segment.

    Strategy: try matching inside a tight window around the frame hints first
    (so we lock onto the right occurrence even when a phrase repeats across the
    video). If that fails, progressively widen the search — this is the
    ±N-line nearby fallback the user asked for.

    LEGACY-RANGE path: when `is_legacy_range=True`, both hints are treated as
    exact start/end times (no phrase matching). This handles the stitch
    extensions produced by analyzer when refinement returned fewer segments
    than the input count.
    """
    if is_legacy_range:
        if start_time_hint is None or end_time_hint is None:
            return None
        try:
            es = float(start_time_hint)
            ee = float(end_time_hint)
        except (TypeError, ValueError):
            return None
        if ee <= es:
            return None
        return {
            "exact_start": es,
            "exact_end": ee,
            "start_match": None,
            "end_match": None,
            "is_legacy_range": True,
        }

    start_frag = boundary_match_phrase(start_phrase, "start")
    end_frag = boundary_match_phrase(end_phrase, "end")

    start_match = None
    for width in _NEARBY_TIME_WIDTHS:
        start_match = find_best_phrase_match(
            all_words, start_frag, approx_time=start_time_hint, search_seconds=width
        )
        if start_match:
            break
    if not start_match:
        # last resort: full-stream scan
        start_match = find_best_phrase_match(all_words, start_frag, approx_time=None)
    if not start_match:
        return None

    # End must come AFTER start. Use end_time_hint when given; otherwise probe
    # from a generous offset past the start.
    expected_end = end_time_hint if end_time_hint else (start_match["start_time"] + 30.0)
    end_match = None
    for width in _NEARBY_TIME_WIDTHS:
        end_match = find_best_phrase_match(
            all_words, end_frag, approx_time=expected_end, search_seconds=width
        )
        if end_match and end_match["end_time"] > start_match["start_time"]:
            break
    if not end_match:
        end_match = find_best_phrase_match(all_words, end_frag, approx_time=None)
    if not end_match or end_match["end_time"] <= start_match["start_time"]:
        return None

    # Trim leading dead air AND leading discourse-filler inside the matched start
    # window — open on the hook, not on "yeah um" / "and" / dead air. Take the
    # later of the two trims (whichever pushes the start further into real
    # content), but never past the matched end.
    trimmed_start = max(
        _trim_leading_silence(start_match, all_words),
        _trim_leading_filler(start_match, all_words),
    )
    if trimmed_start >= float(end_match["end_time"]):
        trimmed_start = _trim_leading_silence(start_match, all_words)
    # Snap end forward to the next sentence terminator (≤ lookahead).
    snapped_end = _snap_end_to_sentence(end_match, all_words)
    if snapped_end <= trimmed_start:
        snapped_end = float(end_match["end_time"])

    return {
        "exact_start": trimmed_start,
        "exact_end": snapped_end,
        "start_match": start_match,
        "end_match": end_match,
        "is_legacy_range": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The pipeline entry point — pin word-exact boundaries on a candidate list
# ─────────────────────────────────────────────────────────────────────────────

def map_exact_boundaries(
    candidates: List[Dict[str, Any]],
    transcript_data: Dict[str, Any],
    logger,
) -> List[Dict[str, Any]]:
    """Resolve each candidate's segments to word-exact times.

    Pairing is by candidate_id (preserved end-to-end). If a candidate's anchors
    cannot be located, the candidate is kept as-is so downstream code never
    silently loses clips.
    """
    all_words = flatten_transcript_json(transcript_data)
    mapped: List[Dict[str, Any]] = []
    skipped = 0

    for index, cand in enumerate(candidates, 1):
        cid = cand.get("candidate_id") or f"cand_{index:03d}"
        parsed_segments = parse_candidate_segments(cand)
        if not parsed_segments:
            logger.warning(f"Candidate {cid} has no parseable segment anchors; keeping unresolved.")
            mapped.append(cand)
            continue

        exact_segments: List[Dict[str, float]] = []
        ok = True
        for seg_info in parsed_segments:
            boundaries = locate_exact_boundaries(
                seg_info["start_phrase"],
                seg_info["end_phrase"],
                all_words,
                start_time_hint=seg_info.get("start_time_hint"),
                end_time_hint=seg_info.get("end_time_hint"),
                is_legacy_range=seg_info.get("is_legacy_range", False),
            )
            if not boundaries:
                logger.warning(
                    f"Candidate {cid}: could not locate boundaries for "
                    f"start='{seg_info['start_phrase'][:40]}...' / "
                    f"end='{seg_info['end_phrase'][:40]}...'"
                )
                ok = False
                break
            es = round(boundaries["exact_start"], 3)
            ee = round(boundaries["exact_end"], 3)
            exact_segments.append({
                "start": es,
                "end": ee,
                "duration": round(ee - es, 3),
            })

        if ok and exact_segments:
            cand["segments"] = exact_segments
            cand["start_time"] = exact_segments[0]["start"]
            cand["end_time"] = exact_segments[-1]["end"]
            cand["total_duration"] = round(
                sum(s["end"] - s["start"] for s in exact_segments), 3
            )
            mapped.append(cand)
            logger.info(
                f"Resolved {cid}: {len(exact_segments)} segment(s), "
                f"{cand['total_duration']:.1f}s total"
            )
        else:
            skipped += 1
            mapped.append(cand)

    logger.info(
        f"Local boundary mapping: resolved={len(mapped) - skipped}, "
        f"unresolved/kept-as-is={skipped}"
    )
    return mapped


# ─────────────────────────────────────────────────────────────────────────────
# clip_info.txt writer (migrated from clip_alignment for app.py editor path)
# ─────────────────────────────────────────────────────────────────────────────

def _current_clip_span(clip: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    starts: List[float] = []
    ends: List[float] = []
    for seg in clip.get("segments", []) or []:
        try:
            starts.append(float(seg.get("start")))
            ends.append(float(seg.get("end")))
        except (TypeError, ValueError):
            continue
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _write_clip_info(job_dir: Path, clips: Sequence[Dict[str, Any]]) -> None:
    """Same contract the legacy clip_alignment._write_clip_info had."""
    lines: List[str] = []
    for i, clip in enumerate(clips, start=1):
        start, end = _current_clip_span(clip)
        lines.append(f"Clip {i}: {clip.get('clip_name', f'clip_{i:02d}')}")
        lines.append(f"Title: {clip.get('title', '')}")
        if start is not None and end is not None:
            lines.append(f"Time: {start:.3f} -> {end:.3f} ({end - start:.3f}s)")
        if clip.get("transcript_text"):
            lines.append(f"Transcript: {clip.get('transcript_text')}")
        lines.append("")
    Path(job_dir, "clip_info.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Back-annotate resolved start/end times into eligible/refined .txt files
# ─────────────────────────────────────────────────────────────────────────────

def back_annotate_candidate_txt(
    txt_path: Path,
    candidate_id_to_span: Dict[str, Tuple[float, float]],
    logger,
) -> None:
    """After clips_plan.json is generated, walk the existing eligible/refined
    .txt and append the resolved [start, end] under each candidate. Pre-clips,
    these files only describe segments by phrases + frames; this writes the
    truthful resolved timestamps back so the .txt stays a real audit trail.
    """
    if not txt_path.exists() or not candidate_id_to_span:
        return
    try:
        text = txt_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not read {txt_path.name} for back-annotation: {exc}")
        return

    def _annotate_block(block: str) -> str:
        m = re.search(r"\[(\d+)\]\s+(\S+)", block)
        cid = m.group(2) if m else None
        if not cid or cid not in candidate_id_to_span:
            return block
        start, end = candidate_id_to_span[cid]
        # Skip if already annotated.
        if "resolved_clip_span" in block:
            return re.sub(
                r"resolved_clip_span\s*:.*",
                f"resolved_clip_span: {start:.3f}s -> {end:.3f}s ({end-start:.1f}s)",
                block,
            )
        # Insert at end of the block, before the separator line if present.
        return block.rstrip() + f"\nresolved_clip_span: {start:.3f}s -> {end:.3f}s ({end-start:.1f}s)\n"

    # Split on the dashed separator the writers use; re-join after annotating.
    parts = re.split(r"(\n-{10,}\n)", text)
    parts = [_annotate_block(p) if not p.startswith("\n---") else p for p in parts]
    try:
        txt_path.write_text("".join(parts), encoding="utf-8")
        logger.info(f"Back-annotated resolved spans into {txt_path.name}")
    except OSError as exc:
        logger.warning(f"Could not write back-annotated {txt_path.name}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# .txt parsers — used by the local fallback path when clip selection AI fails
# ─────────────────────────────────────────────────────────────────────────────

def parse_candidates_txt(file_content: str) -> List[Dict[str, Any]]:
    """Parse the eligible/refined .txt format produced by analyzer._write_*."""
    header_matches = list(re.finditer(r"(?m)^\[(\d+)\]\s*(.+)$", file_content))
    candidates: List[Dict[str, Any]] = []

    for idx, header_match in enumerate(header_matches):
        block_start = header_match.start()
        block_end = header_matches[idx + 1].start() if idx + 1 < len(header_matches) else len(file_content)
        block_text = file_content[block_start:block_end].strip()
        lines = [ln.rstrip() for ln in block_text.splitlines() if ln.strip()]
        if not lines:
            continue
        first_line = lines[0]
        # Header looks like: "[001] w01_c02"  →  candidate_id is the part after the bracket.
        m_hdr = re.match(r"\[\d+\]\s*(\S+)", first_line)
        cand_id = m_hdr.group(1) if m_hdr else f"cand_{idx + 1:03d}"

        info: Dict[str, Any] = {
            "candidate_id": cand_id,
            "title": "",
            "reason": "",
            "caption": "",
            "content": "",
            "hook_text": "",
            "hook_phrase": "",
            "takeaway": "",
            "youtube_title": "",
            "description_text": "",
            "description_hashtags": "",
            "youtube_tags": "",
            "hashtags": [],
            "hook_score": 5.0,
            "flow_score": 5.0,
            "virality_score": 5.0,
            "meaning_score": 5.0,
            "completeness_score": 5.0,
            "boundary_score": 5.0,
            "master_score": 0.0,
            "segments": [],
        }

        # Per-segment NEW-schema accumulator: we look for "SEG N start_words"/etc.
        current_seg: Dict[str, Any] = {}

        for line in lines[1:]:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()

            if key in ("title",):
                info["title"] = val
            elif key in ("reason",):
                info["reason"] = val
            elif key in ("caption",):
                info["caption"] = val
            elif key in ("content",):
                info["content"] = val
            elif key in ("hook_text",):
                info["hook_text"] = val
            elif key in ("hook_phrase",):
                info["hook_phrase"] = val
            elif key in ("takeaway",):
                info["takeaway"] = val
            elif key in ("youtube_title",):
                info["youtube_title"] = val
            elif key in ("description_text", "description"):
                info["description_text"] = val
            elif key in ("description_hashtags",):
                info["description_hashtags"] = val
            elif key in ("youtube_tags",):
                info["youtube_tags"] = val
            elif key in ("hashtags",):
                tags = [t.strip() for t in val.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split() if t.strip()]
                info["hashtags"] = tags
            elif key == "scores":
                for sk, sv in re.findall(r"(\w+)=([\d.]+)", val):
                    sk_lower = sk.lower()
                    if sk_lower == "hook":
                        info["hook_score"] = float(sv)
                    elif sk_lower == "flow":
                        info["flow_score"] = float(sv)
                    elif sk_lower == "viral":
                        info["virality_score"] = float(sv)
                    elif sk_lower == "meaning":
                        info["meaning_score"] = float(sv)
                    elif sk_lower == "completeness":
                        info["completeness_score"] = float(sv)
                    elif sk_lower == "boundary":
                        info["boundary_score"] = float(sv)
                    elif sk_lower == "master":
                        info["master_score"] = float(sv)
            # NEW per-segment lines  "seg_1_start_words" / "seg_1_start_frame" / etc.
            else:
                m_seg = re.match(r"seg[_\s]*(\d+)[_\s]*(start_words|start_frame|end_words|end_frame)", key)
                if m_seg:
                    seg_idx = int(m_seg.group(1))
                    field = m_seg.group(2)
                    # ensure list slot exists
                    while len(info["segments"]) < seg_idx:
                        info["segments"].append({})
                    info["segments"][seg_idx - 1][field] = val

        candidates.append(info)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Local fallback alignment — kept as a safety net for AI selection failure
# ─────────────────────────────────────────────────────────────────────────────

def run_local_alignment(job_dir, logger) -> Tuple[bool, str]:
    """Local fallback when AI clip selection fails: read refined/eligible .txt,
    map to clips_plan.json. Same contract pipeline.runner.run_pipeline expects.
    """
    try:
        job_path = Path(job_dir)

        candidates_file = None
        for name in ["refined_candidates.txt", "eligible_candidates.txt"]:
            p = job_path / name
            if p.exists() and p.stat().st_size > 0:
                candidates_file = p
                break

        if not candidates_file:
            for p in sorted(job_path.glob("*.txt")):
                try:
                    sample = p.read_text(encoding="utf-8", errors="ignore")
                    if re.search(r"(?m)^\[\d+\]\s+", sample):
                        candidates_file = p
                        break
                except OSError:
                    continue

        if not candidates_file:
            return False, "No candidate text file found."

        logger.info(f"Parsing candidates from {candidates_file.name}...")
        try:
            file_content = candidates_file.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return False, f"Failed to read candidates file: {e}"

        candidates = parse_candidates_txt(file_content)
        if not candidates:
            return False, f"No candidates parsed from {candidates_file.name}."

        transcript_path = job_path / "transcript.json"
        if not transcript_path.exists():
            return False, "transcript.json missing."
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        mapped = map_exact_boundaries(candidates, transcript_data, logger)

        from pipeline.clip_selector import _build_plan, _remove_overlaps
        transcript_segments = transcript_data.get("segments", [])
        plans = []
        for i, cand in enumerate(mapped):
            try:
                plan = _build_plan(cand, transcript_segments, i, logger)
                if plan:
                    plans.append(plan)
            except Exception as e:
                logger.warning(f"Skipping fallback candidate {i+1}: {e}")
        if not plans:
            return False, "No usable clip plans after fallback boundary mapping."

        plans = _remove_overlaps(plans, logger)
        with open(job_path / "clips_plan.json", "w", encoding="utf-8") as f:
            json.dump(plans, f, indent=2, ensure_ascii=False)
        logger.info(f"Fallback clips_plan.json saved with {len(plans)} clips.")
        return True, f"Aligned {len(plans)} clips locally."

    except Exception as e:
        import traceback
        logger.error(f"Error in run_local_alignment: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)
