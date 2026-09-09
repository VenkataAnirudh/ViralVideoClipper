"""
2-pass AI stitch pass for the manual / external-LLM analysis route.

The API route already has a clip-level stitch pass (analyzer.run_stitch_pass)
wired into clip_selector. The manual route used to suppress it entirely
(runner hardcoded skip_stitch_pass=True) because the user's own LLM chose the
boundaries and no provider may be called implicitly. This module is the manual
route's opt-in replacement: it runs ONLY when analysis_mode == "manual" AND an
OpenAI key is configured, and it never touches the API route's code path.

Two passes, both via api_provider.run_json_task (responses cached per
task_name inside the job dir, so a crash/resume never re-bills or re-runs a
completed call):

  Pass 1 — identification: every under-min_duration clip plus its time-order
      neighbours goes to the model, which returns merge GROUPS of candidate
      ids. Segments are never edited — only concatenated chronologically.

  Pass 2 — metadata fusion: for each approved group, the members' metadata is
      sent (topic hashtags only — generic filler is stripped first to save
      tokens) and the model returns ONE cohesive merged identity (title,
      description, hashtags, tags, hook, takeaway). The generic filler
      hashtags are re-appended locally from config.GENERIC_FILLER_HASHTAGS.

Failure policy — the pipeline must never break because of this pass:
  * no OpenAI key / no short clips / Pass 1 fails → candidates returned as-is
  * Pass 2 fails for a group → the merge still happens, metadata inherited
    from the strongest member (same behaviour as the API route's stitcher)
  * unresolved candidates (word anchors that never mapped to times) are passed
    through untouched and never considered for stitching
"""

import hashlib
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline import api_provider


# ── Small helpers ────────────────────────────────────────────────────────────

def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_resolved(cand: dict) -> bool:
    """True when every segment carries numeric start/end times (end > start).
    Candidates map_exact_boundaries could not resolve keep word-anchor segments
    and must never be stitched (their 'duration' would read as 0)."""
    segments = cand.get("segments") or []
    if not segments:
        return False
    for seg in segments:
        if not isinstance(seg, dict):
            return False
        try:
            if float(seg.get("end")) <= float(seg.get("start")):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _dur(cand: dict) -> float:
    return sum(
        max(0.0, _f(seg.get("end")) - _f(seg.get("start")))
        for seg in cand.get("segments", []) or []
    )


def _start_of(cand: dict) -> float:
    segs = cand.get("segments") or []
    if segs and isinstance(segs[0], dict) and segs[0].get("start") is not None:
        start = _f(segs[0].get("start"), default=-1.0)
        if start >= 0:
            return start
    return _f(cand.get("start_time"), default=0.0)


def _filler_set() -> set:
    return {t.lower() for t in str(getattr(config, "GENERIC_FILLER_HASHTAGS", "")).split()}


def _strip_filler_hashtags(tags) -> str:
    """Topic-only hashtags for prompts: drop every generic filler tag (they are
    re-appended locally after Pass 2, so sending them just burns tokens)."""
    if isinstance(tags, list):
        tokens = [str(t).strip() for t in tags]
    else:
        tokens = str(tags or "").split()
    filler = _filler_set()
    return " ".join(t for t in tokens if t and t.lower() not in filler)


def _append_filler_hashtags(topic_tags) -> str:
    """Topic tags first, then the generic filler block, deduped case-insensitively."""
    if isinstance(topic_tags, list):
        tokens = [str(t).strip() for t in topic_tags]
    else:
        tokens = str(topic_tags or "").split()
    tokens += str(getattr(config, "GENERIC_FILLER_HASHTAGS", "")).split()
    seen: set = set()
    out: list = []
    for tok in tokens:
        if not tok:
            continue
        if not tok.startswith("#"):
            tok = "#" + tok
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return " ".join(out)


def _openai_plan(settings: dict) -> list:
    """OpenAI-only provider plan, same key-resolution pattern as the analyzer's
    almighty fallback (settings override first, then config)."""
    key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
    if not key:
        return []
    models = list(
        getattr(config, "OPENAI_ANALYSIS_FALLBACK_MODELS", None)
        or getattr(config, "OPENAI_MODELS", ["gpt-5-nano"])
    )
    return [("openai", m, key) for m in models]


def _leader_of(members: list) -> dict:
    """Strongest member — its metadata is the fallback identity for the merge."""
    return max(
        members,
        key=lambda c: (
            _f(c.get("master_score")),
            _f(c.get("virality_score")),
            _f(c.get("hook_score")),
        ),
    )


# ── Pass 1: identification ───────────────────────────────────────────────────

def _clip_brief(cand: dict, label: str, gap_sec=None) -> str:
    if not cand:
        return f"{label}: (none)"
    excerpt = str(
        cand.get("clip_transcript") or cand.get("takeaway") or cand.get("description_text") or ""
    )[:300]
    gap_str = "" if gap_sec is None else f" (gap from previous clip: {gap_sec:.1f}s)"
    return (
        f"{label}: id={cand.get('candidate_id', '?')}, duration={_dur(cand):.1f}s{gap_str}\n"
        f"  title: {str(cand.get('youtube_title') or cand.get('working_title') or '')[:120]}\n"
        f"  takeaway: {str(cand.get('takeaway') or '')[:200]}\n"
        f"  transcript excerpt: {excerpt}"
    )


def _build_identify_prompt(
    short_clips: list, neighbor_map: dict, min_duration: int, max_duration: int
) -> str:
    def _gap(a, b):
        if not a or not b:
            return None
        a_end = _f((a.get("segments") or [{}])[-1].get("end"))
        b_start = _f((b.get("segments") or [{}])[0].get("start"))
        return max(0.0, b_start - a_end)

    blocks = []
    for idx, sc in enumerate(short_clips, 1):
        sc_id = sc.get("candidate_id", f"short_{idx:02d}")
        prev_c = neighbor_map.get(sc_id, {}).get("prev")
        next_c = neighbor_map.get(sc_id, {}).get("next")
        blocks.append("\n".join([
            f"════ SHORT CLIP {idx} (id={sc_id}, only {_dur(sc):.1f}s — needs >= {min_duration}s) ════",
            _clip_brief(sc, "SHORT CLIP"),
            _clip_brief(prev_c, "NEIGHBOR BEFORE", gap_sec=_gap(prev_c, sc)),
            _clip_brief(next_c, "NEIGHBOR AFTER", gap_sec=_gap(sc, next_c)),
        ]))
    body = "\n\n".join(blocks)

    return f"""══════════════════════════════════════════════════
EXTERNAL-ROUTE STITCH — PASS 1 (IDENTIFICATION)
══════════════════════════════════════════════════
You are a clip stitching judge. You are given SHORT clips (each below the {min_duration}s minimum) from a manually-analyzed podcast video, plus each one's immediate time-order neighbours (the clip BEFORE and AFTER it).

Your only job: decide which clip ids to combine so under-length clips become viable. You DO NOT rewrite boundaries. You DO NOT touch segments. You only output GROUPS of clip ids to merge.

Rules:
  • Each merge group is an ordered list of clip ids combined into ONE final clip (segments concatenated chronologically).
  • Topic match is NICE-TO-HAVE, not required — loosely related ideas can combine if it gives a more complete watch and stays under {max_duration}s total.
  • Prefer pairs (short + one neighbour). Triples only when both neighbours flow naturally.
  • Aim for combined durations of at least {min_duration}s; NEVER exceed {max_duration}s.
  • If a short clip stands alone meaningfully OR its neighbours are clearly unrelated and combining would feel jarring, list it under "skipped".
  • NEVER include the same clip id in more than one merge group.
  • Do NOT invent ids — only use ids that appear above, verbatim.
  • Ids inside a merge group MUST be in chronological (time) order.

INPUT — under-length clips and their neighbours:

{body}

══════════════════════════════════════════════════
OUTPUT SCHEMA  (return ONLY this JSON, no markdown/commentary)
══════════════════════════════════════════════════
{{
  "summary": "One sentence describing what you stitched and why",
  "merges": [
    {{"merge_ids": ["<id_a>", "<id_b>"], "reason": "Why these belong together"}}
  ],
  "skipped": [
    {{"id": "<short_clip_id>", "reason": "Why this stays as-is"}}
  ]
}}

HARD RULES:
  • Only output JSON. Double-quoted keys. No trailing commas.
  • Each merge_ids list MUST contain >= 2 ids, in chronological order.
  • Every short-clip id from the input MUST appear in EITHER "merges" OR "skipped" — never both, never missing.
"""


def _identify_validate(parsed: dict, logger: logging.Logger) -> dict:
    """Non-raising cleaner: garbage in → empty merges out (clips stay as-is)."""
    if not isinstance(parsed, dict):
        return {"summary": "", "merges": [], "skipped": []}
    clean_merges: list = []
    seen_ids: set = set()
    for m in parsed.get("merges") or []:
        if not isinstance(m, dict):
            continue
        ids = [str(x).strip() for x in (m.get("merge_ids") or []) if str(x).strip()]
        if len(ids) < 2:
            continue
        if any(i in seen_ids for i in ids):
            logger.warning(f"  [ExtStitch] dropping merge with already-used id(s): {ids}")
            continue
        seen_ids.update(ids)
        clean_merges.append({"merge_ids": ids, "reason": str(m.get("reason", ""))[:240]})
    clean_skipped = [
        {"id": str(s.get("id", "")).strip(), "reason": str(s.get("reason", ""))[:240]}
        for s in (parsed.get("skipped") or [])
        if isinstance(s, dict) and s.get("id")
    ]
    return {
        "summary": str(parsed.get("summary", ""))[:400],
        "merges": clean_merges,
        "skipped": clean_skipped,
    }


# ── Pass 2: metadata fusion ──────────────────────────────────────────────────

_FUSION_FIELDS = (
    "youtube_title", "description_text", "description_hashtags",
    "youtube_tags", "hook_phrase", "takeaway",
)


def _build_fusion_prompt(members: list, reason: str) -> str:
    blocks = []
    for idx, m in enumerate(members, 1):
        blocks.append("\n".join([
            f"──── CLIP {idx} (id={m.get('candidate_id', '?')}, {_dur(m):.1f}s) ────",
            f"youtube_title: {str(m.get('youtube_title') or m.get('working_title') or '')[:150]}",
            f"takeaway: {str(m.get('takeaway') or '')[:300]}",
            f"hook_phrase: {str(m.get('hook_phrase') or '')[:150]}",
            f"description_text: {str(m.get('description_text') or m.get('description') or '')[:600]}",
            f"topic_hashtags: {_strip_filler_hashtags(m.get('description_hashtags'))[:400]}",
            f"youtube_tags: {str(m.get('youtube_tags') or '')[:400]}",
            f"transcript: {str(m.get('clip_transcript') or '')[:1500]}",
        ]))
    body = "\n\n".join(blocks)

    return f"""══════════════════════════════════════════════════
EXTERNAL-ROUTE STITCH — PASS 2 (METADATA FUSION)
══════════════════════════════════════════════════
The clips below are being merged into ONE clip, played back-to-back in the order shown (they were combined because: {reason or 'they flow together and fix an under-length clip'}).

Write ONE cohesive identity for the combined clip. It must read as a single piece of content — NOT "part 1 + part 2". Keep the same style, tone and language as the inputs.

Field rules:
  • youtube_title: one scroll-stopping title covering the combined idea (same style as inputs, max ~100 chars).
  • description_text: 2-4 sentences covering the full combined arc. No hashtags inside.
  • description_hashtags: 10-15 TOPIC hashtags only, space-separated, all lowercase, each starting with #. NO generic filler tags (#viral #fyp #shorts #trending etc.) — those are appended automatically later.
  • youtube_tags: comma-separated search tags for the combined clip (merge + dedupe the inputs, drop what no longer fits).
  • hook_phrase: the single strongest opening hook (usually from the FIRST clip — that is what plays first).
  • takeaway: one sentence with the combined payoff.

INPUT CLIPS (chronological playback order):

{body}

══════════════════════════════════════════════════
OUTPUT SCHEMA  (return ONLY this JSON, no markdown/commentary)
══════════════════════════════════════════════════
{{
  "merged_metadata": {{
    "youtube_title": "...",
    "description_text": "...",
    "description_hashtags": "#topic1 #topic2 ...",
    "youtube_tags": "tag one, tag two, ...",
    "hook_phrase": "...",
    "takeaway": "..."
  }}
}}
"""


def _fusion_validate(parsed: dict) -> dict:
    """Raising validator: a malformed response triggers the provider retry loop
    (bounded for OpenAI primaries), and the caller falls back to leader
    metadata if it ultimately fails."""
    if not isinstance(parsed, dict):
        raise ValueError("fusion response is not a JSON object")
    meta = parsed.get("merged_metadata")
    if not isinstance(meta, dict):
        raise ValueError("missing merged_metadata object")
    clean: dict = {}
    for field in _FUSION_FIELDS:
        value = meta.get(field, "")
        if isinstance(value, list):
            value = " ".join(str(v).strip() for v in value if str(v).strip())
        clean[field] = str(value or "").strip()
    if not clean["youtube_title"]:
        raise ValueError("merged_metadata.youtube_title is empty")
    return {"merged_metadata": clean}


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_external_2pass_stitch(
    job_dir: str,
    candidates: list,
    min_duration: int,
    max_duration: int,
    settings: dict,
    logger: logging.Logger,
) -> list:
    """2-pass stitch for external-LLM candidates. Returns the final candidate
    list: merged clips minted, consumed originals removed, everything else
    untouched, chronologically sorted. On ANY failure the input list is
    returned unchanged (or partially merged with leader metadata)."""
    settings = settings or {}
    if not candidates:
        return candidates

    plan = _openai_plan(settings)
    if not plan:
        logger.info("[ExtStitch] no OpenAI key configured — keeping clips as-is.")
        return candidates

    resolved = [c for c in candidates if _is_resolved(c)]
    unresolved_count = len(candidates) - len(resolved)
    if unresolved_count:
        logger.info(
            f"[ExtStitch] {unresolved_count} candidate(s) without resolved times "
            f"are excluded from stitching (passed through untouched)."
        )

    by_start = sorted(resolved, key=_start_of)
    short_clips = [c for c in by_start if _dur(c) < float(min_duration)]
    if not short_clips:
        logger.info("[ExtStitch] no under-length clips — nothing to stitch.")
        return candidates

    pos = {id(c): i for i, c in enumerate(by_start)}
    neighbor_map: dict = {}
    for sc in short_clips:
        i = pos[id(sc)]
        neighbor_map[sc.get("candidate_id", f"short_{i:02d}")] = {
            "prev": by_start[i - 1] if i - 1 >= 0 else None,
            "next": by_start[i + 1] if i + 1 < len(by_start) else None,
        }

    logger.info(
        f"[ExtStitch] Pass 1: {len(short_clips)} under-length clip(s) of "
        f"{len(resolved)} resolved — asking OpenAI which to merge."
    )

    # Cache key folds in durations + the short-clip set, so re-running with
    # different min/max duration (or changed clips) never reuses a stale plan.
    fingerprint = hashlib.md5(
        (f"d{int(min_duration)}-{int(max_duration)}|" + "|".join(
            f"{c.get('candidate_id', '')}:{_dur(c):.1f}" for c in short_clips
        )).encode()
    ).hexdigest()[:10]

    try:
        identified = api_provider.run_json_task(
            job_dir,
            f"external_stitch_identify_{fingerprint}",
            _build_identify_prompt(short_clips, neighbor_map, min_duration, max_duration),
            plan,
            logger,
            validate=lambda p: _identify_validate(p, logger),
            enable_reasoning=False,
            clip_count=len(short_clips),
            race_count=1,
        )
    except Exception as e:
        logger.warning(f"[ExtStitch] Pass 1 failed ({e}); keeping clips as-is.")
        return candidates

    merges = (identified or {}).get("merges") or []
    for s in (identified or {}).get("skipped") or []:
        logger.info(f"  [ExtStitch] skipped {s.get('id')}: {s.get('reason', '')[:120]}")
    if not merges:
        logger.info("[ExtStitch] Pass 1 proposed no merges; keeping clips as-is.")
        return candidates

    by_id = {c.get("candidate_id"): c for c in resolved if c.get("candidate_id")}
    consumed: set = set()
    merged_new: list = []

    for m in merges:
        ids = m.get("merge_ids") or []
        members = [by_id[i] for i in ids if i in by_id and i not in consumed]
        if len(members) < 2:
            logger.warning(f"  [ExtStitch] skipping merge — fewer than 2 known/unused members: {ids}")
            continue
        members.sort(key=_start_of)
        member_ids = [c.get("candidate_id") for c in members]

        all_segs = []
        for member in members:
            all_segs.extend(member.get("segments") or [])
        all_segs.sort(key=lambda s: _f(s.get("start")))
        total = sum(max(0.0, _f(s.get("end")) - _f(s.get("start"))) for s in all_segs)
        if total > float(max_duration) + 30.0:
            logger.warning(
                f"  [ExtStitch] skipping merge {member_ids} — combined {total:.1f}s "
                f"exceeds max_duration {max_duration}s + 30s slack"
            )
            continue

        leader = _leader_of(members)
        merged_clip = dict(leader)
        merged_clip["segments"] = all_segs
        merged_clip["candidate_id"] = "merged_" + "+".join(member_ids)
        merged_clip["stitched_from"] = list(member_ids)
        merged_clip["stitch_reason"] = str(m.get("reason", ""))[:240]
        merged_clip["stitched_by"] = "external_2pass"
        merged_clip["start_time"] = _f(all_segs[0].get("start"))
        merged_clip["end_time"] = _f(all_segs[-1].get("end"))
        merged_clip["total_duration"] = round(total, 3)
        transcript_join = " ".join(
            str(c.get("clip_transcript") or "").strip()
            for c in members if str(c.get("clip_transcript") or "").strip()
        )
        if transcript_join:
            merged_clip["clip_transcript"] = transcript_join

        # Pass 2 — fused identity. Cached per member-id set; on failure the
        # leader's metadata (already copied above) is kept.
        try:
            fused = api_provider.run_json_task(
                job_dir,
                "external_stitch_meta_" + "+".join(member_ids),
                _build_fusion_prompt(members, str(m.get("reason", ""))),
                plan,
                logger,
                validate=_fusion_validate,
                enable_reasoning=False,
                clip_count=1,
                race_count=1,
            )
            meta = fused["merged_metadata"]
            merged_clip["youtube_title"] = meta["youtube_title"][:150]
            merged_clip["working_title"] = meta["youtube_title"][:120]
            merged_clip["title"] = meta["youtube_title"][:120]
            if meta["description_text"]:
                merged_clip["description_text"] = meta["description_text"]
            merged_clip["description_hashtags"] = _append_filler_hashtags(meta["description_hashtags"])
            if meta["youtube_tags"]:
                merged_clip["youtube_tags"] = meta["youtube_tags"]
            if meta["hook_phrase"]:
                merged_clip["hook_phrase"] = meta["hook_phrase"]
            if meta["takeaway"]:
                merged_clip["takeaway"] = meta["takeaway"]
            logger.info(
                f"  [ExtStitch] Pass 2 fused metadata for {merged_clip['candidate_id']}: "
                f"\"{meta['youtube_title'][:80]}\""
            )
        except Exception as e:
            logger.warning(
                f"  [ExtStitch] Pass 2 failed for {member_ids} ({e}); "
                f"merging with leader metadata instead."
            )

        merged_new.append(merged_clip)
        consumed.update(i for i in member_ids if i)
        logger.info(
            f"  [ExtStitch] merged {member_ids} -> {merged_clip['candidate_id']} "
            f"({total:.1f}s, {len(all_segs)} segments)"
        )

    if not merged_new:
        return candidates

    final = [c for c in candidates if c.get("candidate_id") not in consumed]
    final.extend(merged_new)
    final.sort(key=_start_of)
    logger.info(
        f"[ExtStitch] complete: {len(candidates)} candidates -> {len(final)} "
        f"(merged {len(consumed)} into {len(merged_new)})"
    )
    return final
