"""
Transcript Analyzer
====================
Runs strict sequential NVIDIA NIM analysis over transcript windows.
Returns a chapter map, video summary, and ranked viral clip candidates.
"""

import json
import os
import re
import time
import logging
import threading
import config
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ANALYSIS_WINDOW_SECONDS = 900.0
ANALYSIS_OVERLAP_SECONDS = 180.0
MAX_JUDGE_CANDIDATES = 64
# ── Transport layer lives in pipeline/api_provider.py ────────────────────────
# Every "make an AI call" concern (provider clients, RPM limiting, retry/race
# runners, response caching, JSON extraction, key testing) was split out into
# api_provider. The names are imported back here so this module's workflow code
# AND external callers that do `from pipeline.analyzer import <name>` keep
# working unchanged. analyzer.py now holds only the analysis workflow.
from pipeline import api_provider
from pipeline.api_provider import (
    AI_ANALYSIS_CALL_LOCK,
    _NVIDIA_RATE,
    _is_nvidia_auth_error,
    _is_rate_limit_error,
    _retry_sleep_seconds,
    get_cached_api_response,
    save_cached_api_response,
    delete_cached_api_response,
    _save_bad_ai_response,
    _save_ai_failure,
    _normalize_provider,
    _analysis_provider_plan,
    _call_nvidia,
    _call_openai,
    _call_claude,
    _call_gemini,
    _read_chat_response,
    _read_streaming_chat_response,
    _http_error_message,
    _looks_like_response_format_rejection,
    _looks_like_reasoning_rejection,
    _extract_json,
    call_llm,
    test_api_key,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI "ALMIGHTY" FALLBACK  (NVIDIA-primary; OpenAI only for the tough cases)
# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI is reserved as the decider for cases NVIDIA can't crack: a refinement
# batch that comes back EMPTY, and clips STILL below-par after NVIDIA's
# verification re-call. It never runs on the happy path, and a per-job cap bounds
# spend. The counter is keyed by job_dir and guarded by a lock (refinement runs
# concurrently across batches).
_OPENAI_ALMIGHTY_LOCK = threading.Lock()
_OPENAI_ALMIGHTY_USED: dict[str, int] = {}


def _openai_almighty_enabled(settings: dict) -> bool:
    override = settings.get("openai_almighty_fallback")
    if override is not None:
        enabled = bool(override)
    else:
        enabled = bool(getattr(config, "OPENAI_ALMIGHTY_FALLBACK", True))
    key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
    return enabled and bool(key)


def _openai_almighty_take(job_dir: str, settings: dict, logger: logging.Logger) -> bool:
    """Reserve one OpenAI 'almighty' call from this job's budget (thread-safe).
    Returns False when disabled, no key, or the per-job cap is exhausted."""
    if not _openai_almighty_enabled(settings):
        return False
    cap = int(
        settings.get("openai_almighty_max_calls")
        or getattr(config, "OPENAI_ALMIGHTY_MAX_CALLS_PER_JOB", 30)
    )
    with _OPENAI_ALMIGHTY_LOCK:
        used = _OPENAI_ALMIGHTY_USED.get(job_dir, 0)
        if used >= cap:
            logger.info(
                f"OpenAI almighty fallback: per-job cap ({cap}) reached — "
                f"keeping NVIDIA result"
            )
            return False
        _OPENAI_ALMIGHTY_USED[job_dir] = used + 1
        return True


def _openai_only_plan(settings: dict) -> list:
    """An OpenAI-only provider plan (no NVIDIA), for the almighty escalations."""
    key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
    if not key:
        return []
    models = list(
        getattr(config, "OPENAI_ANALYSIS_FALLBACK_MODELS", None)
        or getattr(config, "OPENAI_MODELS", ["gpt-5-nano"])
    )
    return [("openai", m, key) for m in models]


def _openai_refine_batch(
    job_dir, meta, batch, transcript, min_duration, max_duration, settings, logger,
) -> list | None:
    """Last-resort OpenAI refinement for a batch NVIDIA returned EMPTY for.
    Budget-gated; returns the refined candidate list, or None on no-budget/fail."""
    if not _openai_almighty_take(job_dir, settings, logger):
        return None
    prompt = _build_boundary_refinement_prompt(
        meta, batch, transcript, len(batch), min_duration, max_duration
    )
    cid0 = (batch[0].get("candidate_id") if batch else "batch") or "batch"
    try:
        parsed = _run_analysis_json_task(
            job_dir, f"openai_refine_{cid0}", prompt,
            _openai_only_plan(settings), logger,
            enable_reasoning=False, expected_count=len(batch), clip_count=len(batch),
            temperature=getattr(config, "NVIDIA_TEMP_REFINE", 0.4), race_count=1,
        )
        cands = (parsed.get("candidates") if isinstance(parsed, dict) else None) or []
        if cands:
            logger.info(
                f"  [Refinement batch] OpenAI almighty recovered {len(cands)} "
                f"candidate(s) NVIDIA returned empty for"
            )
            return cands
        return None
    except Exception as exc:
        logger.warning(f"  [Refinement batch] OpenAI almighty failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK TRACKING & GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_fallback_status(job_dir: str, stage: str, status: str, details: str) -> None:
    """Write fallback reports to analysis_fallback_status.txt in a structured format."""
    path = os.path.join(job_dir, "analysis_fallback_status.txt")
    
    # Read existing content if it exists
    existing = ""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
        except Exception:
            pass
            
    # Parse or initialize sections
    sections = {}
    if existing:
        current_stage = None
        current_lines = []
        for line in existing.splitlines():
            match = re.match(r"^\[STAGE:\s*(.*?)\]", line)
            if match:
                if current_stage:
                    sections[current_stage] = "\n".join(current_lines).strip()
                current_stage = match.group(1).strip()
                current_lines = []
            elif line.startswith("===") or line.startswith("Timestamp:") or line.startswith("Job Directory:"):
                continue
            else:
                if current_stage is not None:
                    current_lines.append(line)
        if current_stage:
            sections[current_stage] = "\n".join(current_lines).strip()
            
    # Update current stage
    sections[stage.upper()] = f"Status: {status}\nDetails: {details}\nUpdated: {datetime.datetime.now().isoformat()}"
    
    # Write back
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("              PIPELINE FALLBACK STATUS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Job Directory: {job_dir}\n\n")
            
            stages = ["DISCOVERY", "JUDGE", "COMPRESSION", "BOUNDARY REFINE", "VALIDATE CLIPS", "COPYWRITER", "BLOGGER"]
            for stg in stages:
                f.write(f"[STAGE: {stg}]\n")
                if stg in sections:
                    f.write(sections[stg] + "\n\n")
                else:
                    f.write("Status: PENDING\nDetails: Stage not reached yet.\n\n")
            f.write("=" * 60 + "\n")
    except Exception as e:
        print(f"Failed to write fallback status: {e}")

def _generate_local_fallback_candidates(segments: list, min_duration: int, max_duration: int, count: int) -> list[dict]:
    """Generate default clip candidates locally from transcript segments if AI Discovery fails completely."""
    total_duration = float(segments[-1].get("end", 0.0))
    chunk_dur = max(float(min_duration), min(float(max_duration), 60.0))
    candidates = []
    
    # Divide the video into chunks
    current_start = 0.0
    c_idx = 1
    while current_start < total_duration and len(candidates) < count:
        current_end = min(total_duration, current_start + chunk_dur)
        if current_end - current_start < 10.0: # Too short for a clip
            break
            
        # Find segments in this range
        clip_segments = [
            s for s in segments
            if float(s.get("end", 0.0)) > current_start and float(s.get("start", 0.0)) < current_end
        ]
        
        if clip_segments:
            candidates.append({
                "candidate_id": f"w01_c{c_idx:02d}",
                "title": f"Key Segment Part {c_idx}",
                "hook_score": 6,
                "flow_score": 6,
                "virality_score": 6,
                "completeness_score": 6,
                "boundary_score": 6,
                "meaning_score": 6,
                "reason": "Local rule-based fallback candidate",
                "hook_text": clip_segments[0].get("text", "")[:100],
                "hook_phrase": clip_segments[0].get("text", "")[:40] if clip_segments[0].get("text") else "Listen To This",
                "caption": f"Part {c_idx} of the video highlights",
                "description": f"A local fallback candidate slice covering {current_start:.1f}s to {current_end:.1f}s.",
                "hashtags": ["#keymoment", "#video", "#clip", "#fyp"],
                "start_time": current_start,
                "end_time": current_end,
                "segments": [{"start": float(s["start"]), "end": float(s["end"])} for s in clip_segments]
            })
            c_idx += 1
            
        current_start += chunk_dur * 0.8 # Overlap by 20%
        
    return candidates

def _run_batched_boundary_refinement(
    job_dir: str,
    meta: dict,
    candidates: list,
    transcript: dict,
    clip_count: int,
    min_duration: int,
    max_duration: int,
    discovery_providers: list,
    logger: logging.Logger,
    enable_reasoning: bool,
    skip_duplication: bool,
    settings: dict,
) -> tuple[list, str]:
    """Run boundary refinement — N candidates per batch (default 5).

    Candidates are split into batches of REFINEMENT_BATCH_SIZE (settings or
    config; UI-controlled). Each batch is one work-unit fired through the
    concurrent runner — by default no racing (refine_race_count=1), with full
    model-ladder fallback per task. Batches run AI_MAX_PARALLEL_UNITS at a
    time.

    The prompt now handles N candidates per call (each output candidate
    preserves its input candidate_id verbatim, with segments[].length >= its
    own input segment count). Raise the batch size for stronger models, lower
    it for weaker ones.

    No clip is silently dropped: if a batch comes back empty, or an individual
    candidate isn't returned (by candidate_id), the original (unrefined)
    candidate is kept via the per-id fallback below.
    """
    segments = transcript.get("segments", []) or []

    if not candidates:
        return [], ""

    batch_size = int(
        settings.get("refinement_batch_size")
        or getattr(config, "REFINEMENT_BATCH_SIZE", 5)
        or 5
    )
    batch_size = max(1, batch_size)
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]

    logger.info(
        f"Boundary refinement: {len(candidates)} candidate(s) in {len(batches)} "
        f"batch(es) of up to {batch_size} — each batch raced (best of successes wins)"
    )

    tasks: list[tuple[str, str]] = []
    for b_idx, batch in enumerate(batches, 1):
        prompt = _build_boundary_refinement_prompt(
            meta, batch, transcript, len(batch), min_duration, max_duration
        )
        cache_key = f"boundary_refine_batch_{b_idx:02d}of{len(batches):02d}"
        tasks.append((cache_key, prompt))

    # No racing for refinement by default — each task makes a single API call.
    # Tasks still run concurrently via the AI_MAX_PARALLEL_UNITS thread pool
    # (7 tasks in parallel), with 3-retry + model fallback per task.
    parsed_per_batch = _run_analysis_json_tasks_concurrent(
        job_dir, tasks, discovery_providers, logger,
        enable_reasoning=enable_reasoning,
        expected_count_per_task=999,
        clip_count=batch_size,
        label="refinement",
        temperature=getattr(config, "NVIDIA_TEMP_REFINE", 0.4),
        race_count=int(settings.get("refine_race_count")
                       or getattr(config, "NVIDIA_REFINE_RACE_COUNT", 1)),
    )

    all_refined: list[dict] = []
    summaries: list[str] = []

    for batch, refined in zip(batches, parsed_per_batch):
        refined_list = (refined.get("candidates") if isinstance(refined, dict) else None) or []
        if not refined_list:
            # NVIDIA returned empty for this batch — escalate to the OpenAI
            # almighty before giving up (budget-gated). Only if that also fails
            # do we keep the originals unrefined.
            oa = _openai_refine_batch(
                job_dir, meta, batch, transcript, min_duration, max_duration, settings, logger
            )
            if oa:
                refined_list = oa
            else:
                logger.warning(
                    f"  [Refinement batch] empty response, keeping "
                    f"{len(batch)} original (unrefined) candidate(s)"
                )
                all_refined.extend(batch)
                continue

        # Single-clip batch: the model may omit candidate_id — back-fill it.
        if len(batch) == 1:
            for r in refined_list:
                r.setdefault("candidate_id", batch[0].get("candidate_id"))

        filt = _filter_and_score_candidates(
            _dedupe_candidates(refined_list, logger, skip=skip_duplication),
            segments, min_duration, max_duration, logger,
            require_metadata=False,
        )
        refined_ids = {c.get("candidate_id") for c in filt}
        all_refined.extend(filt)
        # Re-add any original candidate the model dropped or that got filtered out.
        for orig in batch:
            if orig.get("candidate_id") not in refined_ids:
                all_refined.append(orig)

        if isinstance(refined, dict) and refined.get("summary"):
            summaries.append(refined["summary"])

    # ── Conditional verification pass ────────────────────────────────────────
    # For candidates whose refinement looks below-par (under-segmented or much
    # shorter than min_duration), fire ONE focused re-call per candidate with a
    # tighten-end / fill-missing-segment prompt. Single call, no race, no
    # batching. Most candidates skip this entirely.
    skip_verification = bool(settings.get("skip_refinement_verification", False))
    if not skip_verification:
        all_refined = _conditional_refinement_verification(
            all_refined,
            original_candidates=candidates,
            meta=meta,
            transcript=transcript,
            providers=discovery_providers,
            job_dir=job_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            settings=settings,
            logger=logger,
            enable_reasoning=enable_reasoning,
        )

    merged_summary = " | ".join(s for s in summaries if s)
    logger.info(
        f"Boundary refinement complete: {len(all_refined)} refined candidate(s) "
        f"from {len(candidates)} input"
    )
    return all_refined, merged_summary


def _estimate_refined_duration_seconds(cand: dict) -> float:
    """Best-effort duration estimate from a refined candidate's segments.

    Refined output carries `start_frame` / `end_frame` as "[MM:SS.ss -> MM:SS.ss]"
    strings. Build (start, end) pairs across all segments, then coalesce
    touching pairs and sum spans — same logic as `_candidate_duration` so a
    contiguous run isn't under-counted by float-rounding drift.
    """
    pairs: list[tuple[float, float]] = []
    for seg in cand.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        sf = _parse_frame_string(seg.get("start_frame", ""))
        ef = _parse_frame_string(seg.get("end_frame", ""))
        if sf and ef:
            seg_s = float(sf[0])
            seg_e = float(ef[1])
            if seg_e > seg_s:
                pairs.append((seg_s, seg_e))
                continue
        try:
            ns = float(seg.get("start", 0.0))
            ne = float(seg.get("end", ns))
            if ne > ns:
                pairs.append((ns, ne))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return 0.0
    CONTIGUOUS_TOLERANCE = 0.5
    pairs.sort(key=lambda p: p[0])
    cur_start, cur_end = pairs[0]
    total = 0.0
    for p_start, p_end in pairs[1:]:
        if p_start <= cur_end + CONTIGUOUS_TOLERANCE:
            cur_end = max(cur_end, p_end)
        else:
            total += max(0.0, cur_end - cur_start)
            cur_start, cur_end = p_start, p_end
    total += max(0.0, cur_end - cur_start)
    return total


def _needs_refinement_verification(
    refined: dict,
    original: dict,
    min_duration: int,
) -> tuple[bool, str]:
    """Return (needs_verification, reason)."""
    refined_segs = refined.get("segments", []) or []
    orig_segs = original.get("segments", []) or []
    if len(refined_segs) < len(orig_segs):
        return True, f"output_seg_count={len(refined_segs)} < input_seg_count={len(orig_segs)}"
    if not refined_segs:
        return True, "refined has no segments"
    est_dur = _estimate_refined_duration_seconds(refined)
    floor = 0.7 * float(min_duration)
    if est_dur < floor:
        return True, f"estimated_duration={est_dur:.1f}s < 0.7×min_duration={floor:.1f}s"
    # Premature-start scrutiny: the clip opens on a connector/filler word OR a
    # vague/soft opener that references something unstated, instead of the hook.
    # (We check word/phrase sets, NOT lowercase — the whole transcript is
    # lowercased, so islower() would flag everything.)
    start_raw = str(refined_segs[0].get("start_words", "")).strip()
    start_low = re.sub(r"^[^A-Za-z0-9]+", "", start_raw).lower()
    first_words = start_low.split()
    if first_words and first_words[0] in _START_FILLER_CONNECTORS:
        return True, f"start opens on filler/connector '{first_words[0]}'"
    for prefix in _SOFT_OPENER_PREFIXES:
        if start_low.startswith(prefix):
            return True, f"start opens on vague/soft phrase '{prefix}...'"
    return False, ""


# Words that should never OPEN a clip — they signal a mid-thought / non-hook
# start. Used to route premature-start clips into the verification/OpenAI pass.
_START_FILLER_CONNECTORS = {
    "and", "but", "or", "so", "because", "which", "then", "also",
    "therefore", "however", "um", "uh", "yeah", "of", "like",
    "well", "okay", "right", "anyway", "basically",
}

# Vague/soft opener phrases that reference something unstated — a weak hook even
# though the first word isn't a bare connector.
_SOFT_OPENER_PREFIXES = (
    "kind of", "sort of", "what that", "the reason i", "the reason that",
    "it was like", "this thing", "that's the thing", "very difficult to argue",
    "difficult to argue", "as i said", "like i said", "you know what",
)


def _build_refinement_verification_prompt(
    meta: dict,
    refined_cand: dict,
    original_cand: dict,
    transcript: dict,
    min_duration: int,
    max_duration: int,
    reason: str,
) -> str:
    """Focused 'fix this one clip' prompt — single candidate, single call."""
    segments = transcript.get("segments", []) or []
    pack = _candidate_pack_for_prompt(
        [refined_cand], include_context=True, transcript_segments=segments
    )
    cid = refined_cand.get("candidate_id") or original_cand.get("candidate_id", "?")
    cid_takeaway = (
        refined_cand.get("takeaway")
        or original_cand.get("takeaway")
        or "(infer the single concrete idea this clip delivers from its transcript)"
    )
    input_seg_count = len(original_cand.get("segments", []) or [])
    return f"""══════════════════════════════════════════════════
FOCUSED BOUNDARY VERIFICATION — ONE CLIP
══════════════════════════════════════════════════
A previous refinement pass returned this clip in a below-par state.
Reason flagged: {reason}

Your ONLY job: re-tighten the boundaries on THIS ONE clip ({cid}). Return exactly ONE candidate in candidates[] preserving candidate_id="{cid}" verbatim. Same JSON contract as before (start_words / start_frame / end_words / end_frame per segment).

THREE targets (all three must hit):
  1. START — opens on the HOOK (curiosity gap / bold claim / stakes), not on filler/connectors AND not on a vague opener that references something unstated (e.g. "kind of like this thing", "what that even means", "the reason I say that is", "very difficult to argue against that"). Push the start to the first words that state the actual point. Opening mid-sentence is fine if those words ARE the hook.
  2. MEANING LOCK — the clip delivers exactly its takeaway: "{cid_takeaway}". Start where that idea is set up, end where it pays off clearly; do not widen into an adjacent idea or the next topic.
  3. END — land where the payoff is CLEAR and complete. Extend the end as far as the SAME idea needs to land clearly — duration is a rough guide, not a cap, so going over to deliver a complete payoff is correct. Only pull BACK if the end has crossed into the NEXT thought / a new example / a topic change. End on a content word, never on a dangling "and / but / so / the / to / um".

Required (this verification pass exists because the first pass missed one of these):
  • segments[] count >= {input_seg_count}. NEVER fewer. NEVER merged.
  • If the first pass cut the payoff short, EXTEND (even past the rough band) until this idea lands clearly. If it crossed into the NEXT idea, pull the end back.
  • If a segment was dropped, restore it from the original.
  • Multi-speaker transcript: do NOT treat speaker changes as boundary signals.

Duration is a ROUGH guide (~30–{max_duration}s), NOT a cap: never truncate the meaning to fit a number, and never pad an already-complete clip.

══════════════════════════════════════════════════
VIDEO METADATA
══════════════════════════════════════════════════
Title       : {meta.get('title', 'Unknown')}
Channel     : {meta.get('channel', 'Unknown')}
Duration    : {meta.get('duration', 0)} seconds

══════════════════════════════════════════════════
THE CLIP TO RE-VERIFY
══════════════════════════════════════════════════
{pack}

══════════════════════════════════════════════════
OUTPUT SCHEMA — exactly one candidate
══════════════════════════════════════════════════
{{
  "summary": "One sentence — what you tightened or restored",
  "candidates": [
    {{
      "candidate_id": "{cid}",
      "takeaway": "One sentence — the concrete idea this clip delivers",
      "youtube_title": "<keep the original or improve it; under 78 chars>",
      "description_text": "<keep original or refine; 2-4 sentences>",
      "description_hashtags": "<keep original; 30-40 hashtags, space-separated>",
      "youtube_tags": "<keep original; ~500 chars>",
      "hook_phrase": "<keep original; lowercase, 5-12 words>",
      "hook_score": 9, "flow_score": 9, "virality_score": 9,
      "meaning_score": 9, "completeness_score": 9, "boundary_score": 9,
      "arc_pattern": "<keep original>",
      "segments": [
        {{
          "start_words": "verbatim first ≤10 words",
          "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words":   "verbatim last ≤10 words",
          "end_frame":   "[MM:SS.ss -> MM:SS.ss]"
        }}
      ]
    }}
  ]
}}

Return ONLY the JSON object. No markdown, no commentary."""


def _conditional_refinement_verification(
    refined_candidates: list[dict],
    original_candidates: list[dict],
    meta: dict,
    transcript: dict,
    providers: list,
    job_dir: str,
    min_duration: int,
    max_duration: int,
    settings: dict,
    logger,
    enable_reasoning: bool,
) -> list[dict]:
    """For each refined candidate that looks below-par, fire ONE re-call.

    No racing, no batching — single API call per failing candidate, model
    ladder fallback per task. Most candidates skip this entirely.
    """
    orig_by_id = {c.get("candidate_id"): c for c in original_candidates}

    needing_verify: list[tuple[int, dict, dict, str]] = []
    for idx, ref in enumerate(refined_candidates):
        cid = ref.get("candidate_id")
        orig = orig_by_id.get(cid) or ref
        ok_skip, reason = _needs_refinement_verification(ref, orig, min_duration)
        if ok_skip:
            needing_verify.append((idx, ref, orig, reason))

    if not needing_verify:
        return refined_candidates

    logger.info(
        f"Boundary verification: {len(needing_verify)}/{len(refined_candidates)} "
        f"candidate(s) flagged for focused re-call (no racing, single API call each)"
    )

    tasks: list[tuple[str, str]] = []
    for idx, ref, orig, reason in needing_verify:
        prompt = _build_refinement_verification_prompt(
            meta, ref, orig, transcript, min_duration, max_duration, reason
        )
        cache_key = f"boundary_verify_{ref.get('candidate_id', f'idx{idx:02d}')}"
        tasks.append((cache_key, prompt))

    verified_per_task = _run_analysis_json_tasks_concurrent(
        job_dir, tasks, providers, logger,
        enable_reasoning=enable_reasoning,
        expected_count_per_task=1,
        clip_count=1,
        label="refine_verify",
        temperature=getattr(config, "NVIDIA_TEMP_REFINE", 0.4),
        race_count=1,
    )

    out = list(refined_candidates)
    nano_tiebreak_pool: list[tuple[int, dict, dict, dict, str]] = []
    for (idx, original_ref, original_cand, reason), parsed in zip(needing_verify, verified_per_task):
        cands = (parsed.get("candidates") if isinstance(parsed, dict) else None) or []
        if not cands:
            logger.warning(
                f"  [Verify {original_ref.get('candidate_id')}] empty response — "
                f"keeping pre-verify candidate as-is"
            )
            continue
        new_ref = cands[0]
        new_ref.setdefault("candidate_id", original_ref.get("candidate_id"))
        # Only accept the verification if it actually meets the targets it was
        # called to fix. Otherwise keep the pre-verify version (don't make
        # things worse).
        still_bad, new_reason = _needs_refinement_verification(
            new_ref, original_cand, min_duration
        )
        if still_bad:
            logger.info(
                f"  [Verify {original_ref.get('candidate_id')}] still below par "
                f"after re-call ({new_reason}) — keeping pre-verify version"
            )
            # Optional nano tie-break opts in on the clips that are STILL below
            # par after the NVIDIA re-call.
            nano_tiebreak_pool.append((idx, original_ref, new_ref, original_cand, new_reason))
            continue
        out[idx] = new_ref
        logger.info(
            f"  [Verify {original_ref.get('candidate_id')}] tightened ({reason})"
        )

    # ── OpenAI "almighty" tie-break on the clips NVIDIA couldn't crack ───────
    # Fires on clips STILL below-par after NVIDIA's verify re-call (the tough /
    # scrutiny ones). On by default via OPENAI_ALMIGHTY_FALLBACK; budget-capped
    # per job inside the pass. `enable_nano_tiebreak` stays as an explicit
    # override.
    if nano_tiebreak_pool and (
        _openai_almighty_enabled(settings) or bool(settings.get("enable_nano_tiebreak", False))
    ):
        try:
            out = _nano_tiebreak_pass(
                out, nano_tiebreak_pool, meta, transcript,
                min_duration, max_duration, settings, logger, job_dir=job_dir,
            )
        except Exception as exc:
            logger.warning(f"OpenAI tie-break pass skipped due to error: {exc}")

    return out


def _nano_tiebreak_pass(
    out: list[dict],
    nano_pool: list[tuple[int, dict, dict, dict, str]],
    meta: dict,
    transcript: dict,
    min_duration: int,
    max_duration: int,
    settings: dict,
    logger,
    job_dir: str = "",
) -> list[dict]:
    """OpenAI 'almighty' decider for clips that remain below-par after NVIDIA
    verification. For each, GPT-5 nano picks between the pre-verify and
    post-verify boundary sets (or proposes a corrected third). One OpenAI call
    per clip, budget-capped per job via _openai_almighty_take.
    """
    from pipeline.api_provider import _call_openai
    openai_key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
    if not openai_key:
        logger.info("OpenAI tie-break: no OpenAI key configured — skipping")
        return out
    nano_model = (
        getattr(config, "OPENAI_MODELS", ["gpt-5-nano"])[:1] or ["gpt-5-nano"]
    )[0]
    segments = transcript.get("segments", []) or []

    for idx, pre_verify, post_verify, original_cand, reason in nano_pool:
        cid = pre_verify.get("candidate_id") or original_cand.get("candidate_id", "?")
        if not _openai_almighty_take(job_dir, settings, logger):
            break  # per-job OpenAI budget exhausted — stop escalating
        cid_takeaway = (
            pre_verify.get("takeaway")
            or original_cand.get("takeaway")
            or "(infer the single concrete idea this clip delivers)"
        )
        pre_pack = _candidate_pack_for_prompt(
            [pre_verify], include_context=True, transcript_segments=segments
        )
        post_pack = _candidate_pack_for_prompt(
            [post_verify], include_context=False, transcript_segments=segments
        )
        prompt = f"""You are a precision video-clip boundary judge. Two refinement passes produced two different boundary sets for the SAME source clip. Both are below-par for reason: "{reason}".

This clip's takeaway (MEANING LOCK — keep exactly this idea): "{cid_takeaway}"

Pick the better of the two — OR propose a corrected third set if you can clearly improve on both. Output ONE candidate.

THREE targets:
  1. START — opens on the hook (curiosity gap / bold claim / stakes). Strip leading filler/connectors ("and", "so", "um", "yeah", "well"). Opening mid-sentence is fine if those words ARE the hook.
  2. MEANING LOCK — the clip delivers exactly the takeaway above: start where that idea is set up, end where it pays off; do NOT widen into an adjacent idea or the next topic.
  3. END — land ON the payoff, even mid-sentence (captions are uppercased, no trailing period needed). NEVER extend past the payoff into the next thought/example/topic to reach punctuation — that buries the payoff and ships a cliffhanger. If a set overshoots past the payoff, pull the end back to it.

Multi-speaker transcript: do NOT use speaker changes as boundary signals.
Soft duration: 30–60s preferred, ~{max_duration}s ceiling. A clean 29s clip that lands its payoff is fine; do not pad.

══════════════════════════════════════════════════
CANDIDATE A — pre-verify boundaries (with nearby_context)
══════════════════════════════════════════════════
{pre_pack}

══════════════════════════════════════════════════
CANDIDATE B — post-verify boundaries
══════════════════════════════════════════════════
{post_pack}

══════════════════════════════════════════════════
OUTPUT SCHEMA — return exactly one candidate
══════════════════════════════════════════════════
{{
  "winner": "A" | "B" | "C",
  "reason": "One short sentence on why",
  "candidates": [
    {{
      "candidate_id": "{cid}",
      "segments": [
        {{
          "start_words": "verbatim first ≤10 words",
          "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words":   "verbatim last ≤10 words",
          "end_frame":   "[MM:SS.ss -> MM:SS.ss]"
        }}
      ]
    }}
  ]
}}

Return ONLY the JSON. No markdown."""
        try:
            raw = _call_openai(
                prompt,
                model=nano_model,
                api_key=openai_key,
                logger=logger,
                enable_reasoning=False,
                clip_count=1,
                response_format=True,
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning(f"  [Nano tie-break {cid}] OpenAI call failed: {exc}")
            continue

        try:
            parsed = json.loads(raw.strip())
        except Exception:
            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else None
            except Exception:
                parsed = None
        if not isinstance(parsed, dict):
            logger.warning(f"  [Nano tie-break {cid}] could not parse response")
            continue

        winner = str(parsed.get("winner", "")).upper().strip()
        nano_cands = parsed.get("candidates") or []
        if winner == "A":
            # Keep pre-verify (which is what `out[idx]` already holds).
            logger.info(f"  [Nano tie-break {cid}] picked A (pre-verify) — {parsed.get('reason', '')}")
            continue
        if winner == "B" and nano_cands:
            # Use post-verify segments but preserve pre-verify metadata.
            merged = dict(pre_verify)
            merged["segments"] = post_verify.get("segments") or merged.get("segments")
            out[idx] = merged
            logger.info(f"  [Nano tie-break {cid}] picked B (post-verify) — {parsed.get('reason', '')}")
            continue
        if winner == "C" and nano_cands:
            new_segs = nano_cands[0].get("segments") or []
            if new_segs:
                merged = dict(pre_verify)
                merged["segments"] = new_segs
                merged.setdefault("candidate_id", cid)
                out[idx] = merged
                logger.info(f"  [Nano tie-break {cid}] picked C (nano-proposed) — {parsed.get('reason', '')}")
                continue
        logger.info(f"  [Nano tie-break {cid}] inconclusive — keeping pre-verify")
    return out



def analyze_transcript(
    job_dir: str,
    meta: dict,
    transcript: dict,
    settings: dict,
    logger: logging.Logger,
) -> dict:
    # Reset this job's OpenAI "almighty" fallback budget (handles same-process resumes).
    with _OPENAI_ALMIGHTY_LOCK:
        _OPENAI_ALMIGHTY_USED[job_dir] = 0
    analysis_path = os.path.join(job_dir, "analysis.json")
    if os.path.exists(analysis_path):
        logger.info("Found existing analysis.json. Loading it to avoid overwriting or calling API.")
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing analysis.json: {e}. Re-running analysis.")

    provider = _normalize_provider(settings.get("ai_provider", config.DEFAULT_AI_PROVIDER))
    model = settings.get("ai_model", "")
    api_key = settings.get("api_key", "")
    enable_reasoning = settings.get("enable_reasoning", False)
    # Per-stage reasoning for NVIDIA: discovery is a listing task (no thinking —
    # reasoning made Nemotron stream ~20k chunks); refinement benefits from it.
    if provider == "nvidia":
        discovery_reasoning = bool(getattr(config, "NVIDIA_DISCOVERY_REASONING", False))
        refine_reasoning = bool(getattr(config, "NVIDIA_REFINE_REASONING", True))
    else:
        discovery_reasoning = enable_reasoning
        refine_reasoning = enable_reasoning
    clip_count = settings.get("clip_count", config.DEFAULT_CLIP_COUNT)
    min_duration = settings.get("min_duration", config.DEFAULT_MIN_DURATION)
    max_duration = settings.get("max_duration", config.DEFAULT_MAX_DURATION)
    clip_count = max(1, int(clip_count or config.DEFAULT_CLIP_COUNT))
    # Default True — keep every discovered candidate by default. Callers that
    # genuinely want a top-N cap must pass get_all_clips=False explicitly.
    # Previously defaulted False, which silently capped to 10 on any code
    # path that didn't pass the key (scripts, tests, direct callers).
    get_all_clips = bool(settings.get("get_all_clips", True))
    if get_all_clips:
        # Override clip_count to a ceiling high enough to pass everything through.
        clip_count = 999
        logger.info("'Get All Clips' mode active — all discovered candidates will be kept.")
    # Default: skip judge (config.SKIP_JUDGE=True). UI can override per-job.
    skip_judge = settings.get("skip_judge", bool(getattr(config, "SKIP_JUDGE", True)))
    skip_duplication = settings.get("skip_duplication", False)

    segments = transcript.get("segments", [])
    if not segments:
        raise ValueError("Cannot analyze empty transcript")

    total_duration = float(segments[-1].get("end", 0.0))
    window_sec = float(settings.get("analysis_window_seconds") or getattr(config, "AI_ANALYSIS_WINDOW_SECONDS", 900.0))
    overlap_sec = float(settings.get("analysis_overlap_seconds") or getattr(config, "AI_ANALYSIS_OVERLAP_SECONDS", 180.0))
    if window_sec < 60.0:
        window_sec = 60.0
    if overlap_sec >= window_sec:
        overlap_sec = window_sec * 0.2
    if overlap_sec < 0.0:
        overlap_sec = 0.0

    is_chunked = total_duration > window_sec
    chunks = _build_transcript_windows(segments, window_sec, overlap_sec)
    _save_analysis_transcript(job_dir, chunks, logger)

    discovery_providers = _analysis_provider_plan(provider, model, api_key, task_type="discovery")
    complex_providers = _analysis_provider_plan(provider, model, api_key, task_type="complex")
    
    logger.info(
        "AI discovery provider order: "
        + " -> ".join(f"{prov}({mdl or 'default'})" for prov, mdl, _ in discovery_providers)
    )
    logger.info(
        f"AI analysis runs CONCURRENTLY (workers={getattr(config, 'AI_CONCURRENT_WORKERS', 8)}, "
        f"retry_forever={bool(getattr(config, 'AI_RETRY_FOREVER', True))}): "
        f"{len(chunks)} discovery window(s) + per-clip refinement"
    )

    all_candidates = []
    all_chapters = []
    all_summaries = []

    discovered = []

    try:
        # ── Build one prompt per discovery window, fire concurrently ──────────
        window_tasks: list[tuple[str, str]] = []
        window_meta: list[tuple[int, float, float]] = []   # (c_idx, w_start, w_end)
        for c_idx, window_start, window_end, c_transcript in chunks:
            formatted_transcript, transcript_note = _format_transcript_for_analysis(
                c_transcript, min_duration, max_duration, logger
            )
            prompt = _build_analysis_prompt(
                meta,
                formatted_transcript,
                min_duration,
                max_duration,
                transcript_note,
            )
            task_name = f"discover_window_{c_idx + 1}"
            window_tasks.append((task_name, prompt))
            window_meta.append((c_idx, window_start, window_end))

        # Each discovery window is a best-of work-unit: NVIDIA_RACE_COUNT calls fire
        # concurrently, we WAIT FOR ALL of them, and keep the response with the most
        # candidates — so a fast but sparse answer can never win over a richer one.
        # Windows run AI_MAX_PARALLEL_UNITS at a time through the model ladder
        # (Nemotron Ultra -> Super -> DeepSeek V4 Pro -> Flash). Reasoning follows
        # NVIDIA_DISCOVERY_REASONING (ON, budget-capped) for best-quality output.
        parsed_windows = _run_analysis_json_tasks_concurrent(
            job_dir, window_tasks, discovery_providers, logger,
            enable_reasoning=discovery_reasoning,
            expected_count_per_task=999,
            clip_count=999,
            label="discovery",
            temperature=getattr(config, "NVIDIA_TEMP_DISCOVERY", 0.5),
            race_count=int(settings.get("discovery_race_count")
                           or getattr(config, "NVIDIA_DISCOVERY_RACE_COUNT", 3)),
        )

        for (c_idx, window_start, window_end), parsed in zip(window_meta, parsed_windows):
            window_candidates = parsed.get("candidates", []) if parsed else []
            for local_idx, cand in enumerate(window_candidates, 1):
                cand["source_window"] = c_idx + 1
                cand["source_window_start"] = round(window_start, 3)
                cand["source_window_end"] = round(window_end, 3)
                cand["candidate_id"] = f"w{c_idx + 1:02d}_c{local_idx:02d}"
            all_candidates.extend(window_candidates)
            all_chapters.extend(parsed.get("chapters", []) or [])
            if parsed.get("summary"):
                all_summaries.append(parsed.get("summary"))
            logger.info(
                f"[Discovery window {c_idx + 1}] {_fmt_time(window_start)} -> "
                f"{_fmt_time(window_end)}: {len(window_candidates)} raw candidate(s)"
            )

        discovered = _filter_and_score_candidates(
            _dedupe_candidates(all_candidates, logger, skip=skip_duplication),
            segments,
            min_duration,
            max_duration,
            logger,
        )
        if not discovered:
            raise RuntimeError("AI discovery produced 0 usable candidates")
            
        write_fallback_status(job_dir, "DISCOVERY", "SUCCESS", f"Found {len(discovered)} candidates via AI discovery.")
    except Exception as exc:
        logger.error(f"AI Discovery failed: {exc}")
        write_fallback_status(
            job_dir, 
            "DISCOVERY", 
            "FAILED", 
            f"AI Discovery failed with: {exc}"
        )
        raise exc



    # ── Export all discovered candidates immediately after Discovery pass ─────
    _write_eligible_candidates(job_dir, discovered, segments, logger)

    # ── Boundary Refinement (run on all discovered candidates) ────────────────
    refinement_pool = list(discovered)
    final_candidates = []
    refined_summary = ""

    try:
        if settings.get("skip_refinement", False):
            logger.info("Skipping AI Boundary Refinement pass based on settings/config.")
            final_candidates = refinement_pool
            refined_summary = "Skipped boundary refinement pass."
            write_fallback_status(job_dir, "BOUNDARY REFINE", "SKIPPED", "AI Boundary Refinement pass skipped based on configuration.")
        else:
            final_candidates, refined_summary = _run_batched_boundary_refinement(
                job_dir              = job_dir,
                meta                 = meta,
                candidates           = refinement_pool,
                transcript           = transcript,
                clip_count           = 999,
                min_duration         = min_duration,
                max_duration         = max_duration,
                discovery_providers  = discovery_providers,
                logger               = logger,
                enable_reasoning     = refine_reasoning,
                skip_duplication     = skip_duplication,
                settings             = settings,
            )
            write_fallback_status(
                job_dir, "BOUNDARY REFINE", "SUCCESS",
                f"Refined boundaries for {len(final_candidates)} candidates."
            )
    except Exception as exc:
        logger.warning(f"AI Boundary Refinement failed: {exc}. Using raw discovery candidates.")
        final_candidates = refinement_pool
        write_fallback_status(
            job_dir, "BOUNDARY REFINE", "FALLBACK TRIGGERED",
            f"Refinement failed: {exc}. Used unrefined candidates."
        )

    # ── Export refined candidates immediately after refinement ────────────────
    _write_refined_candidates(job_dir, final_candidates, segments, logger)

    # ── Judge Pass (run on refined candidates) ────────────────────────────────
    judged_candidates = []
    judge_summary = ""

    if skip_judge:
        logger.info("Skipping AI Judge pass based on settings/config.")
        judged_candidates = final_candidates
        judge_summary = "Skipped judge pass."
        write_fallback_status(job_dir, "JUDGE", "SKIPPED", "AI Judge pass skipped based on configuration.")
    else:
        try:
            judge_prompt = _build_judge_prompt(
                meta, final_candidates, 999, min_duration, max_duration
            )
            # Judge is a single API call with full model-ladder fallback — no racing.
            judged = _run_analysis_json_task(
                job_dir,
                "judge_rank",
                judge_prompt,
                complex_providers,
                logger,
                enable_reasoning=refine_reasoning,
                expected_count=len(final_candidates),
                clip_count=len(final_candidates),
                temperature=getattr(config, "NVIDIA_TEMP_JUDGE", 0.4),
                race_count=1,
            )
            judged_candidates = _filter_and_score_candidates(
                _dedupe_candidates(judged.get("candidates", []), logger, skip=skip_duplication),
                segments,
                min_duration,
                max_duration,
                logger,
            )
            if not judged_candidates:
                raise RuntimeError("AI Judge pass produced 0 usable candidates.")
            judge_summary = judged.get("summary") or ""
            write_fallback_status(job_dir, "JUDGE", "SUCCESS", f"Ranked candidates via AI Judge.")
        except Exception as exc:
            logger.warning(f"AI Judge pass failed: {exc}. Using refined candidates directly.")
            judged_candidates = final_candidates
            write_fallback_status(
                job_dir,
                "JUDGE",
                "FALLBACK TRIGGERED",
                f"AI Judge failed with: {exc}. Used refined candidates directly."
            )

    # ── Compression Pass (tighten clips that still exceed max_duration in batches) 
    skip_compression = settings.get("skip_compression", False)
    compression_source = list(judged_candidates or final_candidates)
    compressed_candidates = []
    compression_events = 0

    if skip_compression:
        logger.info("Skipping AI Compression Pass based on settings/config.")
        write_fallback_status(job_dir, "COMPRESSION", "SKIPPED", "AI Compression Pass skipped based on configuration.")
        judged_candidates = compression_source
    else:
        try:
            eligible_for_compression = []
            non_eligible = []
            for cand in compression_source:
                cand_duration = _candidate_duration(cand)
                if cand_duration > float(max_duration):
                    eligible_for_compression.append(cand)
                else:
                    non_eligible.append(cand)

            if eligible_for_compression:
                logger.info(
                    f"[Compression] Found {len(eligible_for_compression)} overlong candidate(s) (> {max_duration}s) "
                    "eligible for compression. Running batched compression pass."
                )
                
                compression_batch_size = int(settings.get("compression_batch_size", 10))
                batches = [eligible_for_compression[i:i + compression_batch_size] for i in range(0, len(eligible_for_compression), compression_batch_size)]
                total_batches = len(batches)
                
                compressed_map = {}
                
                for b_idx, batch in enumerate(batches, 1):
                    logger.info(
                        f"  [Compression batch {b_idx}/{total_batches}] "
                        f"Compressing {len(batch)} candidate(s)..."
                    )
                    comp_prompt = _build_batched_compression_prompt(
                        meta, batch, transcript, min_duration, max_duration
                    )
                    
                    cache_key = f"compress_batch_{b_idx}of{total_batches}"
                    
                    try:
                        comp_result = _run_batched_compression_json_task(
                            job_dir,
                            cache_key,
                            comp_prompt,
                            discovery_providers,
                            logger,
                            enable_reasoning=refine_reasoning,
                            temperature=getattr(config, "NVIDIA_TEMP_COMPRESSION", 0.4),
                        )
                        
                        returned_cands = comp_result.get("candidates") or []
                        for ret_cand in returned_cands:
                            c_id = ret_cand.get("candidate_id")
                            if c_id:
                                compressed_map[c_id] = ret_cand
                    except Exception as batch_exc:
                        logger.warning(
                            f"  [Compression batch {b_idx}/{total_batches}] failed: {batch_exc}. "
                            "Retrying by splitting into sub-batches of size 5..."
                        )
                        sub_batches = [batch[i:i + 5] for i in range(0, len(batch), 5)]
                        for sb_idx, sub_batch in enumerate(sub_batches, 1):
                            sub_cache_key = f"compress_batch_{b_idx}of{total_batches}_sub_{sb_idx}"
                            sub_prompt = _build_batched_compression_prompt(
                                meta, sub_batch, transcript, min_duration, max_duration
                            )
                            try:
                                sub_result = _run_batched_compression_json_task(
                                    job_dir,
                                    sub_cache_key,
                                    sub_prompt,
                                    discovery_providers,
                                    logger,
                                    enable_reasoning=enable_reasoning,
                                    temperature=getattr(config, "NVIDIA_TEMP_COMPRESSION", 0.4),
                                )
                                for ret_cand in sub_result.get("candidates", []):
                                    c_id = ret_cand.get("candidate_id")
                                    if c_id:
                                        compressed_map[c_id] = ret_cand
                            except Exception as sub_exc:
                                logger.warning(
                                    f"    [Sub-batch {sb_idx}/{len(sub_batches)}] failed: {sub_exc}. "
                                    "Will fall back to original uncompressed clips for this sub-batch."
                                )

                if len(compressed_map) < len(eligible_for_compression):
                    logger.warning(
                        f"[Compression] model returned {len(compressed_map)} compressed result(s) "
                        f"for {len(eligible_for_compression)} overlong clip(s); the rest keep their "
                        "original (uncompressed) segments — no clip is dropped."
                    )

                for cand in compression_source:
                    c_id = cand.get("candidate_id")
                    if c_id in compressed_map:
                        comp_data = compressed_map[c_id]
                        new_segments = comp_data.get("compressed_segments") or []
                        if new_segments:
                            compressed_cand = {**cand, "segments": new_segments}
                            compressed_cand["start_time"] = _parse_time_to_seconds(new_segments[0]["start"])
                            compressed_cand["end_time"] = _parse_time_to_seconds(new_segments[-1]["end"])
                            compressed_cand["compression_note"] = comp_data.get("compression_note", "")
                            compression_events += 1
                            logger.info(
                                f"[Compression] '{cand.get('title', '?')}' compressed from "
                                f"{_candidate_duration(cand):.1f}s to {_candidate_duration(compressed_cand):.1f}s"
                            )
                            compressed_candidates.append(compressed_cand)
                            continue
                    
                    compressed_candidates.append(cand)
            else:
                logger.info("[Compression] No candidates exceeded max_duration. Skipping compression pass.")
                compressed_candidates = compression_source

            if compressed_candidates:
                judged_candidates = _filter_and_score_candidates(
                    _dedupe_candidates(compressed_candidates, logger, skip=skip_duplication),
                    segments,
                    min_duration,
                    max_duration,
                    logger,
                )

            if compression_events:
                write_fallback_status(
                    job_dir,
                    "COMPRESSION",
                    "SUCCESS",
                    f"Compressed {compression_events} candidate(s) that exceeded max_duration."
                )
            else:
                write_fallback_status(
                    job_dir,
                    "COMPRESSION",
                    "SKIPPED",
                    "No candidates were compressed."
                )
        except Exception as exc:
            logger.warning(f"Compression pass failed: {exc}. Keeping judged/refined candidates as-is.")
            write_fallback_status(
                job_dir,
                "COMPRESSION",
                "FALLBACK TRIGGERED",
                f"Compression failed with: {exc}. Kept prior candidates unchanged."
            )
            judged_candidates = compression_source

    # Populate default/fallback metadata fields if any candidate is missing them (e.g. if refinement was skipped)
    for idx, cand in enumerate(judged_candidates):
        cand.setdefault("rank", idx + 1)
        cand.setdefault("title", cand.get("title") or f"Clip {idx + 1}")
        cand.setdefault("hook_text", cand.get("hook_text") or cand.get("title") or "")
        cand.setdefault("hook_phrase", cand.get("hook_phrase") or cand.get("hook_text") or "")
        cand.setdefault("caption", cand.get("caption") or f"Key Takeaway Part {idx + 1}")
        cand.setdefault("description", cand.get("description") or f"Highlights from the video: {cand.get('title')}.")
        cand.setdefault("hashtags", cand.get("hashtags") or ["#keymoment", "#video", "#clip", "#fyp"])

    final_analysis = {
        "summary": refined_summary or judge_summary or (
            "Merged sequential window analysis" if is_chunked else all_summaries[0] if all_summaries else ""
        ),
        "chapters": all_chapters,
        "candidates": judged_candidates,
        "analysis_workflow": {
            "provider": _normalize_provider(provider),
            "models": [model_name for _, model_name, _ in complex_providers],
            "window_seconds": window_sec,
            "overlap_seconds": overlap_sec,
            "passes": ["discover", "boundary_refine", "judge_rank", "compress_overlong"],
            "local_fallback": False,
            "concurrency": f"best-of (x{getattr(config, 'NVIDIA_RACE_COUNT', 5)} per unit, "
                           f"{getattr(config, 'AI_MAX_PARALLEL_UNITS', 7)} units in parallel)",
        },
    }

    analysis_path = os.path.join(job_dir, "analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Analysis complete: {len(judged_candidates)} candidate(s) written to analysis.json."
    )
    return final_analysis


def analyze_transcript_external(
    job_dir: str,
    meta: dict,
    transcript: dict,
    settings: dict,
    logger: logging.Logger,
) -> dict:
    """Manual / external-LLM analysis route (no network call).

    Ingests the user's pasted, per-window JSON (saved to <job>/external_llm.txt by
    the dashboard), aggregates + dedupes it, and routes it through the SAME
    validate → filter → audit-write path the API route uses. The fused manual
    prompt already produced boundaries + full metadata, so the discovery /
    refinement / judge passes are skipped; local_clips_generator still pins
    word-exact times from the verbatim start_words/end_words downstream, so clip
    cutting is identical to the API route. Returns the same analysis dict shape as
    analyze_transcript so select_clips / runner are unchanged.
    """
    from pipeline import external_llm  # lazy: avoid import cycle

    analysis_path = os.path.join(job_dir, "analysis.json")
    if os.path.exists(analysis_path):
        logger.info("Found existing analysis.json. Loading it to avoid overwriting.")
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing analysis.json: {e}. Re-ingesting manual output.")

    segments = transcript.get("segments", [])
    if not segments:
        raise ValueError("Cannot analyze empty transcript")

    min_duration = settings.get("min_duration", config.DEFAULT_MIN_DURATION)
    max_duration = settings.get("max_duration", config.DEFAULT_MAX_DURATION)
    skip_duplication = bool(settings.get("skip_duplication", False))

    if not external_llm.has_external_llm_input(job_dir):
        write_fallback_status(
            job_dir, "EXTERNAL DISCOVERY", "FAILED",
            "Manual analysis selected but external_llm.txt is missing/empty.",
        )
        raise RuntimeError(
            "Manual analysis selected but no pasted LLM output found "
            f"({external_llm.external_llm_input_path(job_dir)} is missing or empty)."
        )

    raw_text = external_llm.read_external_llm_input(job_dir)
    # threshold >= 1.0 effectively disables dedup (a ratio can never reach it >1).
    threshold = 2.0 if skip_duplication else float(
        getattr(config, "EXTERNAL_LLM_SIMILARITY_THRESHOLD", 0.90)
    )
    candidates, window_summaries = external_llm.aggregate_external_candidates(
        raw_text, logger, similarity_threshold=threshold
    )
    if not candidates:
        write_fallback_status(
            job_dir, "EXTERNAL DISCOVERY", "FAILED",
            "Manual analysis produced 0 usable clips. Check the pasted JSON.",
        )
        raise RuntimeError(
            "Manual analysis produced 0 usable candidates from external_llm.txt "
            "(no valid JSON clips found — re-paste the per-window output)."
        )

    # Normalise via the SAME validator the API route uses (aliases clips→candidates,
    # accepts the {start_words,start_frame,end_words,end_frame} segment schema,
    # synthesizes provisional times from frame hints, cleans scores/hashtags).
    validated = _validate_analysis(
        {"summary": " ".join(window_summaries)[:500], "chapters": [], "candidates": candidates},
        999, logger,
    )
    discovered = _filter_and_score_candidates(
        validated.get("candidates", []), segments, min_duration, max_duration, logger,
    )
    if not discovered:
        write_fallback_status(
            job_dir, "EXTERNAL DISCOVERY", "FAILED",
            "All manual clips were filtered out (bad/duplicate boundaries).",
        )
        raise RuntimeError("Manual analysis candidates were all filtered out")

    write_fallback_status(
        job_dir, "EXTERNAL DISCOVERY", "SUCCESS",
        f"Ingested {len(discovered)} clip(s) from manual LLM output.",
    )

    # Audit files — same as the API route, so back-annotation + matcher work.
    _write_eligible_candidates(job_dir, discovered, segments, logger)
    _write_refined_candidates(job_dir, discovered, segments, logger)

    # Backfill any metadata the manual output omitted (mirrors analyze_transcript).
    for idx, cand in enumerate(discovered):
        cand.setdefault("rank", idx + 1)
        cand.setdefault("title", cand.get("title") or cand.get("working_title") or f"Clip {idx + 1}")
        cand.setdefault("hook_text", cand.get("hook_text") or cand.get("title") or "")
        cand.setdefault("hook_phrase", cand.get("hook_phrase") or cand.get("hook_text") or "")
        cand.setdefault("caption", cand.get("caption") or f"Key Takeaway Part {idx + 1}")
        cand.setdefault("description", cand.get("description") or cand.get("description_text") or f"Highlights: {cand.get('title')}.")
        cand.setdefault("hashtags", cand.get("hashtags") or ["#keymoment", "#video", "#clip", "#fyp"])

    final_analysis = {
        "summary": (" ".join(window_summaries).strip()
                    or f"Manual external-LLM analysis: {len(discovered)} clip(s)."),
        "chapters": [],
        "candidates": discovered,
        "analysis_workflow": {
            "provider": "external_llm",
            "models": ["manual"],
            "window_seconds": float(settings.get("analysis_window_seconds")
                                    or getattr(config, "AI_ANALYSIS_WINDOW_SECONDS", 900.0)),
            "overlap_seconds": float(settings.get("analysis_overlap_seconds")
                                     or getattr(config, "AI_ANALYSIS_OVERLAP_SECONDS", 180.0)),
            "passes": ["external_ingest"],
            "local_fallback": False,
            "similarity_dedup_threshold": (None if skip_duplication else
                                           float(getattr(config, "EXTERNAL_LLM_SIMILARITY_THRESHOLD", 0.90))),
        },
    }
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Manual analysis complete: {len(discovered)} candidate(s) written to analysis.json."
    )
    return final_analysis


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _score_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else default


def _build_transcript_windows(segments: list, window_sec: float, overlap_sec: float) -> list[tuple[int, float, float, dict]]:
    if not segments:
        return []
    total_duration = float(segments[-1].get("end", 0.0))
    if total_duration <= window_sec:
        return [(0, 0.0, total_duration, {"segments": segments})]

    windows = []
    step = window_sec - overlap_sec
    current_start = 0.0
    idx = 0
    while current_start < total_duration:
        window_end = min(total_duration, current_start + window_sec)
        chunk_segs = [
            seg for seg in segments
            if float(seg.get("end", 0.0)) > current_start
            and float(seg.get("start", 0.0)) < window_end
        ]
        if chunk_segs:
            windows.append((idx, current_start, window_end, {"segments": chunk_segs}))
        if window_end >= total_duration:
            break
        current_start += step
        idx += 1
    return windows


def _save_analysis_transcript(job_dir: str, chunks: list[tuple[int, float, float, dict]], logger: logging.Logger) -> None:
    path = os.path.join(job_dir, "analysis_transcript.txt")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for idx, start, end, chunk in chunks:
                handle.write(
                    f"\n\n=== AI ANALYSIS WINDOW {idx + 1}: "
                    f"{_fmt_time(start)} -> {_fmt_time(end)} ===\n"
                )
                handle.write(_format_transcript(chunk))
                handle.write("\n")
        logger.info(f"AI-only transcript view saved: {path}")
    except OSError as exc:
        logger.warning(f"Could not save AI-only transcript view: {exc}")


def _extract_anchors_and_timeframes(cand: dict, segments: list) -> str:
    """
    Extracts start anchor (first 5 words of hook_text),
    end anchor (last 5 words of transcript_evidence),
    and their enclosing segment timeframes.
    """
    start_time = _parse_time_to_seconds(cand.get("start_time") or cand.get("segments", [{}])[0].get("start", 0.0))
    end_time = _parse_time_to_seconds(cand.get("end_time") or cand.get("segments", [{}])[-1].get("end", 0.0))
    
    # 1. Start anchor (first 5 words of hook_text)
    hook = cand.get("hook_text", "")
    hook_words = [w for w in hook.split() if w]
    start_anchor = " ".join(hook_words[:5]) if hook_words else ""
    
    # 2. End anchor (last 5 words of transcript_evidence)
    evidence = cand.get("transcript_evidence", "")
    if not evidence and cand.get("reason"):
        evidence = cand.get("reason", "")
    evidence_words = [w for w in evidence.split() if w]
    end_anchor = " ".join(evidence_words[-5:]) if evidence_words else ""
    
    # 3. Find timeframes for start_time and end_time
    start_tf = "[00:00.00 -> 00:00.00]"
    end_tf = "[00:00.00 -> 00:00.00]"
    
    def fmt_t(s):
        m = int(s // 60)
        sec = s % 60
        return f"{m:02d}:{sec:05.2f}"
        
    for seg in segments:
        s_s = _parse_time_to_seconds(seg.get("start", 0.0))
        s_e = _parse_time_to_seconds(seg.get("end", 0.0))
        if s_s <= start_time <= s_e or (s_s - 0.5 <= start_time <= s_s + 0.5):
            start_tf = f"[{fmt_t(s_s)} -> {fmt_t(s_e)}]"
        if s_s <= end_time <= s_e or (s_s - 0.5 <= end_time <= s_s + 0.5):
            end_tf = f"[{fmt_t(s_s)} -> {fmt_t(s_e)}]"
            
    full_text = cand.get("transcript_evidence", "")
    if not full_text:
        full_text = _candidate_transcript_text(cand, segments)
        
    return f'Start: "{start_anchor}" [Timeframe: {start_tf}] | End: "{end_anchor}" [Timeframe: {end_tf}] | Text: "{full_text}"'


def _candidate_segment_lines(cand: dict) -> list[str]:
    """Per-segment phrase + frame lines for the .txt export. Keys match what
    local_clips_generator.parse_candidates_txt picks up (seg_1_start_words,
    seg_1_start_frame, seg_1_end_words, seg_1_end_frame, ...)."""
    out: list[str] = []
    segs = cand.get("segments") or []
    for s_idx, s in enumerate(segs, 1):
        if not isinstance(s, dict):
            continue
        sw = str(s.get("start_words") or "").strip()
        sf = str(s.get("start_frame") or "").strip()
        ew = str(s.get("end_words") or "").strip()
        ef = str(s.get("end_frame") or "").strip()
        if sw or sf:
            out.append(f"seg_{s_idx}_start_words: {sw}")
            out.append(f"seg_{s_idx}_start_frame: {sf}")
        if ew or ef:
            out.append(f"seg_{s_idx}_end_words  : {ew}")
            out.append(f"seg_{s_idx}_end_frame  : {ef}")
        # If segments contain only legacy {start:MM:SS.ss, end:MM:SS.ss} fields,
        # surface them as a hint too.
        if not (sw or sf or ew or ef):
            legacy_start = s.get("start")
            legacy_end = s.get("end")
            if legacy_start or legacy_end:
                out.append(f"seg_{s_idx}_legacy_range: {legacy_start} -> {legacy_end}")
    return out


def _write_eligible_candidates(
    job_dir: str,
    candidates: list,
    segments: list,
    logger: logging.Logger,
) -> None:
    """Post-discovery audit file. LEAN schema — no YouTube fields, no resolved
    start/end. Per-segment phrase + frame anchors only. Pipeline back-annotates
    `resolved_clip_span` after clips_plan.json is generated.
    """
    path = os.path.join(job_dir, "eligible_candidates.txt")
    try:
        lines = [
            "=" * 70,
            "ELIGIBLE CANDIDATES (Post-Discovery)",
            "Schema: phrase + frame anchors only. Resolved spans are back-annotated",
            "after clips_plan.json is built (look for 'resolved_clip_span').",
            f"Total: {len(candidates)}",
            "",
        ]
        for i, cand in enumerate(candidates, 1):
            lines += [
                f"[{i:03d}] {cand.get('candidate_id', f'candidate_{i:03d}')}",
                f"working_title: {cand.get('working_title', cand.get('title', ''))}",
                f"takeaway     : {cand.get('takeaway', '')}",
                f"segments     : {len(cand.get('segments', []))}",
                f"scores       : hook={cand.get('hook_score', 0)} flow={cand.get('flow_score', 0)} viral={cand.get('virality_score', 0)} meaning={cand.get('meaning_score', 0)} master={cand.get('master_score', 0.0)}",
                f"arc_pattern  : {cand.get('arc_pattern', '')}",
                f"reason       : {cand.get('reason', '')}",
                f"content      : {cand.get('content', '')}",
            ]
            lines += _candidate_segment_lines(cand)
            lines += [
                "",
                "-" * 70,
                "",
            ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Eligible candidates export saved: {path}")
    except Exception as exc:
        logger.warning(f"Could not write eligible_candidates.txt: {exc}")


def _write_refined_candidates(
    job_dir: str,
    candidates: list,
    segments: list,
    logger: logging.Logger,
) -> None:
    """Post-refinement audit file. RICH schema — phrase + frame anchors PLUS
    YouTube metadata generated per-clip in refinement. No resolved start/end
    yet; back-annotated after clips_plan.json exists.
    """
    path = os.path.join(job_dir, "refined_candidates.txt")
    try:
        lines = [
            "=" * 70,
            "REFINED CANDIDATES (Post-Refinement)",
            "Schema: phrase + frame anchors + per-clip YouTube metadata.",
            "Resolved spans are back-annotated after clips_plan.json is built.",
            f"Total: {len(candidates)}",
            "",
        ]
        for i, cand in enumerate(candidates, 1):
            lines += [
                f"[{i:03d}] {cand.get('candidate_id', f'candidate_{i:03d}')}",
                f"youtube_title       : {cand.get('youtube_title', '')}",
                f"takeaway            : {cand.get('takeaway', '')}",
                f"hook_phrase         : {cand.get('hook_phrase', '')}",
                f"description_text    : {cand.get('description_text', '')}",
                f"description_hashtags: {cand.get('description_hashtags', '')}",
                f"youtube_tags        : {cand.get('youtube_tags', '')}",
                f"segments            : {len(cand.get('segments', []))}",
                f"scores              : hook={cand.get('hook_score', 0)} flow={cand.get('flow_score', 0)} viral={cand.get('virality_score', 0)} meaning={cand.get('meaning_score', 0)} master={cand.get('master_score', 0.0)}",
                f"arc_pattern         : {cand.get('arc_pattern', '')}",
                f"reason              : {cand.get('reason', '')}",
                f"content             : {cand.get('content', '')}",
            ]
            lines += _candidate_segment_lines(cand)
            lines += [
                "",
                "-" * 70,
                "",
            ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Refined candidates export saved: {path}")
    except Exception as exc:
        logger.warning(f"Could not write refined_candidates.txt: {exc}")


def _score_analysis_response(parsed: dict | None) -> float:
    """Quality-weighted scorer for race-of-N tiebreaking.

    Replaces the legacy count-based scorer that rewarded padding. A response is
    scored as the sum of per-candidate quality points; candidates without valid
    segments (start_words + end_words) contribute zero — so ghost rows added to
    pad the list don't help. Multi-segment stitching and rich metadata both
    earn extra weight, which aligns the race winner with the prompt's "honest
    count + complete output" intent.

    Works the same way for both discovery (variable count, want richness) and
    refinement (count=1, want every metadata field filled).
    """
    if not isinstance(parsed, dict):
        return 0.0
    cands = parsed.get("candidates") or []
    if not isinstance(cands, list) or not cands:
        return 0.0
    total = 0.0
    for c in cands:
        if not isinstance(c, dict):
            continue
        segs = c.get("segments") or []
        if not isinstance(segs, list) or not segs:
            continue
        valid_segs = sum(
            1 for s in segs
            if isinstance(s, dict)
            and str(s.get("start_words") or "").strip()
            and str(s.get("end_words") or "").strip()
        )
        if valid_segs == 0:
            continue  # ghost candidate — no anchors the matcher can resolve
        pts = 1.0
        pts += min(valid_segs - 1, 5) * 0.3   # multi-segment stitch bonus, capped
        if str(c.get("takeaway") or "").strip():            pts += 0.15
        if str(c.get("youtube_title") or "").strip():       pts += 0.15
        if str(c.get("hook_phrase") or "").strip():         pts += 0.15
        if str(c.get("description_text") or "").strip():    pts += 0.10
        if str(c.get("description_hashtags") or "").strip(): pts += 0.10
        if str(c.get("youtube_tags") or "").strip():        pts += 0.10
        if str(c.get("reason") or "").strip():              pts += 0.05
        total += pts
    return total


def _run_analysis_json_task(
    job_dir: str,
    task_name: str,
    prompt: str,
    providers_to_try: list[tuple[str, str, str]],
    logger: logging.Logger,
    enable_reasoning: bool,
    expected_count: int,
    clip_count: int,
    temperature: float | None = None,
    race_count: int | None = None,
) -> dict:
    """Thin wrapper over api_provider.run_json_task. Injects the analysis
    validator and a QUALITY-weighted scorer (not count); the provider layer owns
    cache, racing, retry-forever, and model fallback.

    race_count controls how many concurrent calls fire per work-unit:
      • None  — use the global NVIDIA_RACE_COUNT default (discovery)
      • 1     — sequential only, no racing (refinement)
      • 2+    — small concurrent race (judge prefers 2, degrades to 1)
    """
    return api_provider.run_json_task(
        job_dir, task_name, prompt, providers_to_try, logger,
        validate=lambda parsed: _validate_analysis(parsed, expected_count, logger),
        enable_reasoning=enable_reasoning,
        clip_count=clip_count,
        temperature=temperature,
        score=_score_analysis_response,
        race_count=race_count,
    )


def _run_analysis_json_tasks_concurrent(
    job_dir: str,
    tasks: list[tuple[str, str]],
    providers_to_try: list[tuple[str, str, str]],
    logger: logging.Logger,
    enable_reasoning: bool,
    expected_count_per_task: int,
    clip_count: int,
    label: str = "task",
    max_workers: int | None = None,
    temperature: float | None = None,
    race_count: int | None = None,
) -> list[dict]:
    """Thin wrapper over api_provider.run_json_tasks_concurrent with the analysis
    validator and a QUALITY-weighted scorer injected.

    race_count controls per-task concurrency:
      • None  — use NVIDIA_RACE_COUNT default (discovery: best-of-N racing)
      • 1     — no racing; each task makes a single sequential call with
                3-retry + model fallback (refinement)
    Tasks still run in parallel via AI_MAX_PARALLEL_UNITS thread pool."""
    return api_provider.run_json_tasks_concurrent(
        job_dir, tasks, providers_to_try, logger,
        validate=lambda parsed: _validate_analysis(parsed, expected_count_per_task, logger),
        enable_reasoning=enable_reasoning,
        clip_count=clip_count,
        label=label,
        max_units=max_workers,
        temperature=temperature,
        score=_score_analysis_response,
        race_count=race_count,
    )


def _validate_batched_compression_result(result: dict, logger: logging.Logger) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Compression task did not return a JSON object")

    if "candidates" not in result:
        for alias in ("compressed_candidates", "compressed", "clips"):
            if isinstance(result.get(alias), list):
                result["candidates"] = result[alias]
                break
        else:
            raise ValueError("Compression task missing candidates list")

    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("Compression field is not a list of candidates")

    clean_candidates = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        
        c_id = cand.get("candidate_id")
        if not c_id:
            continue
            
        segments = cand.get("compressed_segments", [])
        if not segments:
            for alias in ("segments", "compressed", "clip_segments"):
                if isinstance(cand.get(alias), list):
                    segments = cand[alias]
                    break
                    
        clean_segments = []
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                seg_start = _parse_time_to_seconds(segment.get("start"))
                seg_end = _parse_time_to_seconds(segment.get("end"))
                if seg_end > seg_start:
                    clean_segments.append({"start": seg_start, "end": seg_end})
                    
        cand["compressed_segments"] = clean_segments
        cand["compression_note"] = str(cand.get("compression_note") or "").strip()
        clean_candidates.append(cand)

    result["candidates"] = clean_candidates
    return result


def _run_batched_compression_json_task(
    job_dir: str,
    task_name: str,
    prompt: str,
    providers_to_try: list[tuple[str, str, str]],
    logger: logging.Logger,
    enable_reasoning: bool,
    temperature: float | None = None,
) -> dict:
    """Thin wrapper over api_provider.run_json_task with the batched-compression
    validator injected (NVIDIA races; best of successes wins)."""
    return api_provider.run_json_task(
        job_dir, task_name, prompt, providers_to_try, logger,
        validate=lambda parsed: _validate_batched_compression_result(parsed, logger),
        enable_reasoning=enable_reasoning,
        clip_count=1,
        temperature=temperature,
    )



def _candidate_duration(candidate: dict) -> float:
    """Physical runtime of a candidate's rendered output.

    Naive sum-of-segment-durations is wrong two ways:
      • Touching segments (the two pieces of a continuous span split into
        two dicts by stitch_extend or by refinement) accumulate float-rounding
        error and report 29.8s for a true 30.0s span — false rejection.
      • Overlapping segments (trailing-silence from refinement) double-count
        and report 32s for a true 30s span — false acceptance.

    A clip stitched across REAL gaps (e.g. 3 separate timeline spans glued
    into one video) genuinely IS gap_a + gap_b + gap_c long, so spans across
    real gaps still sum. We coalesce only segments that touch within
    CONTIGUOUS_TOLERANCE and then sum the coalesced spans.
    """
    segs = candidate.get("segments", []) or []
    if not segs:
        return 0.0
    CONTIGUOUS_TOLERANCE = 0.5
    sorted_segs = sorted(
        segs, key=lambda s: float(s.get("start", 0.0))
    )
    cur_start = float(sorted_segs[0].get("start", 0.0))
    cur_end = float(sorted_segs[0].get("end", cur_start))
    duration = 0.0
    for seg in sorted_segs[1:]:
        s_start = float(seg.get("start", 0.0))
        s_end = float(seg.get("end", s_start))
        if s_start <= cur_end + CONTIGUOUS_TOLERANCE:
            cur_end = max(cur_end, s_end)
        else:
            duration += max(0.0, cur_end - cur_start)
            cur_start = s_start
            cur_end = s_end
    duration += max(0.0, cur_end - cur_start)
    return duration


def _candidate_transcript_text(candidate: dict, segments: list, limit: int = 1800) -> str:
    parts = []
    if not segments:
        return ""

    for clip_segment in candidate.get("segments", []) or []:
        raw_start = float(clip_segment.get("start", 0.0))
        raw_end = float(clip_segment.get("end", raw_start))

        # Snap start/end to closest transcript segment boundaries (up to 3.0s drift tolerance)
        seg_start = raw_start
        seg_end = raw_end
        min_start_diff = 3.0
        min_end_diff = 3.0

        for transcript_seg in segments:
            start = float(transcript_seg.get("start", 0.0))
            end = float(transcript_seg.get("end", start))

            diff_start = abs(start - raw_start)
            if diff_start < min_start_diff:
                min_start_diff = diff_start
                seg_start = start

            diff_end = abs(end - raw_end)
            if diff_end < min_end_diff:
                min_end_diff = diff_end
                seg_end = end

        for transcript_seg in segments:
            start = float(transcript_seg.get("start", 0.0))
            end = float(transcript_seg.get("end", start))
            if end <= seg_start or start >= seg_end:
                continue
            text = str(transcript_seg.get("text", "")).strip()
            if text:
                parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:limit]


def _candidate_quality_score(candidate: dict) -> float:
    ai_score = (
        _score_float(candidate.get("hook_score"), 0)
        + _score_float(candidate.get("flow_score"), 0)
        + _score_float(candidate.get("virality_score"), 0)
    )
    completeness = _score_float(candidate.get("completeness_score"), 0)
    boundary = _score_float(candidate.get("boundary_score"), 0)
    meaning = _score_float(candidate.get("meaning_score"), 0)
    return ai_score + completeness + boundary + meaning


def _stitch_extend_candidate(candidate: dict, segments: list, min_duration: float, logger: logging.Logger):
    """Stitch/extend a candidate's segments using adjacent segments from the transcript list to meet min_duration."""
    if not candidate.get("segments") or not segments:
        return

    duration = _candidate_duration(candidate)
    if duration >= min_duration:
        return

    # Find the matching indices of first and last segments in the main transcript list
    first_idx = -1
    last_idx = -1
    first_start = _parse_time_to_seconds(candidate["segments"][0]["start"])
    last_end = _parse_time_to_seconds(candidate["segments"][-1]["end"])

    for idx, ts in enumerate(segments):
        ts_start = _parse_time_to_seconds(ts.get("start", 0.0))
        ts_end = _parse_time_to_seconds(ts.get("end", 0.0))
        if first_idx == -1 and abs(ts_start - first_start) < 0.5:
            first_idx = idx
        if abs(ts_end - last_end) < 0.5:
            last_idx = idx

    if first_idx == -1 or last_idx == -1:
        # If we couldn't match exactly, search for closest overlap
        best_first_dist = 9999.0
        best_last_dist = 9999.0
        for idx, ts in enumerate(segments):
            ts_start = _parse_time_to_seconds(ts.get("start", 0.0))
            ts_end = _parse_time_to_seconds(ts.get("end", 0.0))
            if abs(ts_start - first_start) < best_first_dist:
                best_first_dist = abs(ts_start - first_start)
                first_idx = idx
            if abs(ts_end - last_end) < best_last_dist:
                best_last_dist = abs(ts_end - last_end)
                last_idx = idx

    if first_idx == -1 or last_idx == -1:
        return

    logger.info(f"Stitching candidate clip: current duration is {duration:.1f}s, target min is {min_duration}s. Indexes in transcript: {first_idx} to {last_idx}")

    # When the adjacent transcript line touches the existing first/last
    # segment's boundary (within CONTIGUOUS_TOLERANCE), EXTEND that segment's
    # start/end in place. Only append/insert a NEW segment dict when there is
    # a real time gap. The old behavior — always appending a fresh dict —
    # transformed a continuous 33s clip into 5 separate one-liners, which the
    # refinement AI was then forced to emit as five 5-second jump-cuts because
    # its system prompt forbids merging input segments.
    CONTIGUOUS_TOLERANCE = 0.5  # adjacent transcript lines usually have ms-level gaps

    # Step 1: sequentially extend (or append, if a real gap exists) forward.
    curr_idx = last_idx + 1
    while duration < min_duration and curr_idx < len(segments):
        ts = segments[curr_idx]
        ts_start = _parse_time_to_seconds(ts.get("start", 0.0))
        ts_end = _parse_time_to_seconds(ts.get("end", 0.0))
        if ts_end > ts_start:
            last_seg = candidate["segments"][-1]
            last_end = float(last_seg.get("end", 0.0))
            if ts_start <= last_end + CONTIGUOUS_TOLERANCE:
                last_seg["end"] = ts_end
            else:
                candidate["segments"].append({"start": ts_start, "end": ts_end})
            duration = _candidate_duration(candidate)
        curr_idx += 1

    # Step 2: if still too short, extend (or insert) backward.
    curr_idx = first_idx - 1
    while duration < min_duration and curr_idx >= 0:
        ts = segments[curr_idx]
        ts_start = _parse_time_to_seconds(ts.get("start", 0.0))
        ts_end = _parse_time_to_seconds(ts.get("end", 0.0))
        if ts_end > ts_start:
            first_seg = candidate["segments"][0]
            first_start = float(first_seg.get("start", 0.0))
            if ts_end >= first_start - CONTIGUOUS_TOLERANCE:
                first_seg["start"] = ts_start
            else:
                candidate["segments"].insert(0, {"start": ts_start, "end": ts_end})
            duration = _candidate_duration(candidate)
        curr_idx -= 1

    # Update start_time and end_time to reflect the new stitched boundaries
    candidate["start_time"] = _parse_time_to_seconds(candidate["segments"][0]["start"])
    candidate["end_time"] = _parse_time_to_seconds(candidate["segments"][-1]["end"])
    logger.info(f"Stitched candidate clip: new duration is {duration:.1f}s, start: {candidate['start_time']:.1f}, end: {candidate['end_time']:.1f}")


def _repair_metadata(candidate: dict, evidence: str):
    """Locally repair missing metadata (title, caption, description, hashtags) in candidate."""
    title = str(candidate.get("title") or "").strip()
    caption = str(candidate.get("caption") or "").strip()
    description = str(candidate.get("description") or "").strip()
    hashtags = candidate.get("hashtags") or []

    if not isinstance(hashtags, list):
        hashtags = str(hashtags).split()

    cleaned_tags = []
    for tag in hashtags:
        ct = _clean_hashtag(tag)
        if ct:
            cleaned_tags.append(ct)
    cleaned_tags = _dedupe_preserve_order(cleaned_tags)

    default_tags = ["#viral", "#trending", "#fyp", "#shorts", "#foryou", "#growth", "#inspiration", "#content", "#clip", "#bestvideo"]
    if len(cleaned_tags) < 10:
        for tag in default_tags:
            if tag not in cleaned_tags:
                cleaned_tags.append(tag)
                if len(cleaned_tags) >= 10:
                    break
    candidate["hashtags"] = cleaned_tags[:15]

    if not title or title.lower() in ("untitled clip", "clip", "") or title.startswith("Clip "):
        evidence_words = evidence.split()
        if len(evidence_words) >= 3:
            candidate["title"] = " ".join(evidence_words[:5]).strip(".,!?") + "..."
        else:
            candidate["title"] = "Valuable Insight"

    if not caption or caption == "":
        candidate["caption"] = evidence[:150].strip() + "..." if len(evidence) > 150 else evidence.strip()

    if not description or description == "":
        candidate["description"] = f"An interesting highlight: {evidence[:400].strip()}..."


def _filter_and_score_candidates(
    candidates: list[dict],
    segments: list,
    min_duration: int,
    max_duration: int,
    logger: logging.Logger,
    require_metadata: bool = False,
) -> list[dict]:
    clean = []
    for idx, candidate in enumerate(candidates, 1):
        try:
            # Enforce non-empty segments
            if "segments" not in candidate or not candidate["segments"]:
                start_val = candidate.get("start_time")
                end_val = candidate.get("end_time")
                if start_val is not None and end_val is not None:
                    candidate["segments"] = [{"start": float(start_val), "end": float(end_val)}]
                else:
                    candidate["segments"] = []

            # Clean and stitch segments if needed.
            # IMPORTANT: preserve the phrase anchors (start_words/start_frame/
            # end_words/end_frame) — the local matcher needs them to resolve
            # word-exact boundaries. Dropping them (the old behavior) forced the
            # matcher to fall back to whole-segment frame times.
            if candidate.get("segments"):
                clean_segs = []
                for s in candidate["segments"]:
                    s_start = float(s.get("start", 0.0))
                    s_end = float(s.get("end", 0.0))
                    if s_end > s_start:
                        seg = {"start": s_start, "end": s_end}
                        for anchor in ("start_words", "start_frame", "end_words", "end_frame"):
                            if s.get(anchor):
                                seg[anchor] = s[anchor]
                        clean_segs.append(seg)
                if clean_segs:
                    candidate["segments"] = clean_segs
                    _stitch_extend_candidate(candidate, segments, float(min_duration), logger)

            duration = _candidate_duration(candidate)
            if duration <= 0:
                logger.warning(f"Skipping candidate {idx}: zero duration")
                continue
            if duration < 5.0:
                # Kept on purpose (recall-biased) — log accurately, don't say "Skipping".
                logger.info(f"Keeping short candidate {idx} ({duration:.1f}s) — below 5s but retained")

            # Enforce a generous upper skip limit so we don't end up with multi-minute clips,
            # but allow complete clips that are up to 90 seconds over the max duration
            generous_max = float(max_duration) + 90.0
            if duration > generous_max:
                logger.warning(f"Skipping candidate {idx}: excessively long ({duration:.1f}s, limit={generous_max}s)")
                continue

            evidence = _candidate_transcript_text(candidate, segments)
            if len(evidence) < 60:
                # Kept (recall-biased); evidence is thin but the clip may still be valid.
                logger.info(f"Candidate {idx}: thin transcript evidence ({len(evidence)} chars) — retained")

            # Auto-repair metadata locally on the fly
            _repair_metadata(candidate, evidence)

            if require_metadata and not _has_required_metadata(candidate):
                logger.warning(f"Skipping candidate {idx}: missing final title/caption/description/hashtags")
                continue

            # Calculate soft duration penalty
            duration_penalty = 0.0
            min_dur = float(min_duration)
            max_dur = float(max_duration)
            
            if duration < min_dur:
                # 0.15 points penalty per second under min_duration
                duration_penalty += (min_dur - duration) * 0.15
            elif duration > max_dur:
                # 0.05 points penalty per second over max_duration (soft penalty for first 60s)
                over_seconds = duration - max_dur
                if over_seconds <= 60.0:
                    duration_penalty += over_seconds * 0.05
                else:
                    # 3.0 points for first 60s, then 0.15 points per second after that
                    duration_penalty += 3.0 + (over_seconds - 60.0) * 0.15

            candidate["total_duration"] = round(duration, 3)
            candidate["transcript_evidence"] = evidence
            candidate["content"] = _candidate_transcript_text(candidate, segments, limit=20000)

            raw_score = _candidate_quality_score(candidate)
            candidate["master_score"] = round(raw_score - duration_penalty, 3)
            clean.append(candidate)
        except (TypeError, ValueError) as exc:
            logger.warning(f"Skipping candidate {idx}: {exc}")

    clean.sort(key=lambda item: item.get("master_score", 0), reverse=True)
    return clean


def _dedupe_candidates(candidates: list[dict], logger: logging.Logger, skip: bool = False) -> list[dict]:
    if skip:
        logger.info("Skipping cross-window candidate deduplication based on settings.")
        return candidates
    ranked = sorted(candidates, key=lambda item: item.get("master_score", _candidate_quality_score(item)), reverse=True)
    unique = []
    for candidate in ranked:
        start = _parse_time_to_seconds(candidate.get("start_time", 0.0))
        end = _parse_time_to_seconds(candidate.get("end_time", start))
        duration = max(0.1, end - start)
        duplicate = False
        for existing in unique:
            ex_start = _parse_time_to_seconds(existing.get("start_time", 0.0))
            ex_end = _parse_time_to_seconds(existing.get("end_time", ex_start))
            overlap = max(0.0, min(end, ex_end) - max(start, ex_start))
            ex_duration = max(0.1, ex_end - ex_start)
            min_dur_pair = min(duration, ex_duration)
            if overlap / min_dur_pair > getattr(config, "CLIP_OVERLAP_DEDUP_RATIO", 0.6):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    dropped = len(candidates) - len(unique)
    if dropped:
        logger.info(f"Cross-window stitching removed {dropped} duplicate/overlapping candidate(s)")
    return unique


def _has_required_metadata(candidate: dict) -> bool:
    title = str(candidate.get("title") or "").strip()
    caption = str(candidate.get("caption") or "").strip()
    description = str(candidate.get("description") or "").strip()
    hashtags = candidate.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = str(hashtags).split()
    return bool(title and caption and description and len(hashtags) >= 10)


def _clean_hashtag(tag) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_#]", "", str(tag or "").strip())
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("#") else f"#{cleaned}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _format_transcript(transcript: dict) -> str:
    """Format transcript segments into readable text with timestamps."""
    lines = []
    for seg in transcript["segments"]:
        start = _fmt_time(seg["start"])
        end = _fmt_time(seg["end"])
        speaker = seg.get("speaker", "")
        speaker_prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"[{start} -> {end}] {speaker_prefix}{seg['text']}")
    return "\n".join(lines)


def _format_transcript_for_analysis(
    transcript: dict,
    min_duration: int,
    max_duration: int,
    logger: logging.Logger,
) -> tuple[str, str]:
    """Return full transcript or a compact, whole-video candidate scan.

    Discovery has no clip-count quota — we always want the widest viable set of
    windows. We size the compact scan generously so the model sees as much of
    the video as the prompt budget allows.
    """
    full = _format_transcript(transcript)
    prompt_budget = max(30000, int(getattr(config, "AI_ANALYSIS_PROMPT_CHAR_BUDGET", 90000)))
    transcript_budget = max(12000, prompt_budget - 22000)
    if len(full) <= transcript_budget:
        return full, (
            "The full transcript is included below. Use it to find the best complete, "
            "self-contained reel moments across the whole video."
        )

    compact_budget = max(12000, int(getattr(config, "AI_ANALYSIS_COMPACT_CHAR_BUDGET", 42000)))
    compact = _format_compact_analysis_transcript(
        transcript, min_duration, max_duration, compact_budget, logger
    )
    logger.warning(
        f"Transcript is {len(full)} chars; using compact candidate scan "
        f"({len(compact)} chars) to avoid provider token/context limits"
    )
    return compact, (
        "The transcript below is a compact candidate scan built from the whole video. "
        "Each window preserves exact timestamped segment boundaries. Pick only from "
        "these shown boundaries and prefer clips whose excerpt contains a complete "
        "setup, insight, and payoff."
    )


def _format_compact_analysis_transcript(
    transcript: dict,
    min_duration: int,
    max_duration: int,
    budget: int,
    logger: logging.Logger,
) -> str:
    segments = transcript.get("segments", []) or []
    if not segments:
        return ""

    desired = min(float(max_duration), max(float(min_duration), 55.0))
    # No clip-count quota during discovery — surface as many viable windows as
    # the prompt budget allows so the model can identify every clip in band.
    max_windows = int(getattr(config, "AI_ANALYSIS_MAX_WINDOWS", 64))
    windows = _rank_analysis_windows(segments, desired, max_windows)
    blocks = []
    used = 0

    for rank, (start_idx, end_idx, score) in enumerate(windows, 1):
        block_lines = [
            f"[WINDOW {rank:02d} score={score:.1f} "
            f"{_fmt_time(segments[start_idx]['start'])} -> {_fmt_time(segments[end_idx]['end'])}]"
        ]
        for seg in segments[start_idx:end_idx + 1]:
            start = _fmt_time(seg["start"])
            end = _fmt_time(seg["end"])
            speaker = seg.get("speaker", "")
            speaker_prefix = f"[{speaker}] " if speaker else ""
            block_lines.append(f"[{start} -> {end}] {speaker_prefix}{seg['text']}")
        block = "\n".join(block_lines)
        if used and used + len(block) + 2 > budget:
            break
        blocks.append(block)
        used += len(block) + 2

    logger.info(f"Compact transcript scan selected {len(blocks)}/{len(windows)} windows")
    return "\n\n".join(blocks)


def _rank_analysis_windows(segments: list, desired_duration: float, max_windows: int) -> list[tuple[int, int, float]]:
    scored = []
    for idx, seg in enumerate(segments):
        score = _score_segment_for_reel(seg)
        if score <= -2:
            continue
        scored.append((score, idx))

    scored.sort(reverse=True)
    selected: list[tuple[int, int, float]] = []

    def add_anchor(anchor_idx: int, score: float) -> None:
        if len(selected) >= max_windows:
            return
        start_idx = _walk_to_clean_start_index(anchor_idx, segments)
        end_idx = _walk_to_clean_end_index(start_idx, segments, desired_duration)
        start = float(segments[start_idx].get("start", 0.0))
        end = float(segments[end_idx].get("end", start))
        for existing_start_idx, existing_end_idx, _ in selected:
            ex_start = float(segments[existing_start_idx].get("start", 0.0))
            ex_end = float(segments[existing_end_idx].get("end", ex_start))
            overlap = max(0.0, min(end, ex_end) - max(start, ex_start))
            if overlap > desired_duration * 0.45:
                return
        selected.append((start_idx, end_idx, score))

    for score, idx in scored:
        add_anchor(idx, score)
        if len(selected) >= max_windows:
            break

    # Add coverage anchors so quieter but meaningful later sections are not lost.
    if len(selected) < max_windows:
        step = max(1, len(segments) // max_windows)
        for idx in range(0, len(segments), step):
            add_anchor(idx, _score_segment_for_reel(segments[idx]) - 1)
            if len(selected) >= max_windows:
                break

    selected.sort(key=lambda item: float(segments[item[0]].get("start", 0.0)))
    return selected


def _score_segment_for_reel(seg: dict) -> float:
    text = str(seg.get("text", "")).strip()
    lower = text.lower()
    score = 0.0
    if 45 <= len(text) <= 260:
        score += 1.5
    if "?" in text:
        score += 1.2
    if re.search(r"\b\d+[%x]?\b", lower):
        score += 0.8
    for phrase in (
        "most people", "nobody", "secret", "mistake", "wrong", "stop ",
        "why ", "how ", "because", "realized", "changed", "important",
        "the problem", "the truth", "what happens", "you need", "if you",
        "never", "always", "actually", "counterintuitive",
    ):
        if phrase in lower:
            score += 0.9
    if re.match(r"^\s*(so|um|uh|anyway|basically|right|now|okay|welcome|today)\b", lower):
        score -= 2.0
    if any(phrase in lower for phrase in ("subscribe", "like and subscribe", "welcome back", "in this video")):
        score -= 2.5
    if any(phrase in lower for phrase in ("look at this", "as you can see", "this graph", "on screen")):
        score -= 1.5
    return score


def _walk_to_clean_start_index(idx: int, segments: list) -> int:
    steps = 0
    while idx > 0 and steps < 8:
        text = str(segments[idx].get("text", "")).strip()
        prev = str(segments[idx - 1].get("text", "")).strip()
        if not _looks_like_continuation_text(text) and _looks_like_sentence_end_text(prev):
            break
        idx -= 1
        steps += 1
    return idx


def _walk_to_clean_end_index(start_idx: int, segments: list, desired_duration: float) -> int:
    start = float(segments[start_idx].get("start", 0.0))
    idx = start_idx
    while idx + 1 < len(segments):
        end = float(segments[idx].get("end", start))
        if end - start >= desired_duration and _looks_like_sentence_end_text(str(segments[idx].get("text", ""))):
            break
        if end - start >= desired_duration + 8:
            break
        idx += 1
    return idx


def _looks_like_continuation_text(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    first = re.sub(r"^[^A-Za-z0-9]+", "", text.split()[0]).lower()
    return first in {
        "and", "but", "or", "so", "because", "which", "that", "who",
        "then", "also", "therefore", "however", "including", "like",
        "of", "in", "on", "at", "to", "for", "with",
    } or text[0].islower()


def _looks_like_sentence_end_text(text: str) -> bool:
    return bool(str(text).strip().endswith((".", "!", "?")))


def _fmt_time(seconds: float) -> str:
    """Format seconds to MM:SS.s"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def _parse_time_to_seconds(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    t_str = str(val).strip()
    if not t_str:
        return 0.0
    # Remove any brackets
    t_str = re.sub(r'[\[\]]', '', t_str)
    # Check if format is like MM:SS.cc or HH:MM:SS.cc
    parts = t_str.split(':')
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            # Handle float seconds with a possible "s" suffix
            clean_val = re.sub(r'[^\d.-]', '', t_str)
            return float(clean_val) if clean_val else 0.0
    except (ValueError, TypeError):
        try:
            return float(t_str)
        except ValueError:
            return 0.0



def _build_analysis_prompt(
    meta: dict, transcript_text: str,
    min_duration: int, max_duration: int, transcript_note: str = "",
) -> str:
    """Discovery prompt — lean schema, exhaustive identification, multi-segment friendly.

    Contract: identify EVERY viable clip between min_duration and max_duration
    in this window. No quota, no minimum target, no density floor — just an
    honest exhaustive scan. The downstream pipeline applies any clip_count
    selection AFTER refinement, so over-padding here corrupts the pool and
    under-counting silently loses good clips.

    Output deliberately omits YouTube metadata and the resolved scalar
    start_time/end_time. Boundaries are described by per-segment phrases +
    transcript-line timeframes; final word-exact timestamps are pinned later by
    the local matcher against the word-level transcript. YouTube metadata
    (catchy title, description text, 30 hashtags, ~500-char tags, cliffhanger
    hook_phrase) is generated per-clip in the refinement stage where the
    selected content + video metadata are exactly the right inputs.
    """
    return f"""You are a world-class short-form viral video strategist. Working from this transcript window, identify EVERY viable {min_duration}–{max_duration}-second clip that could plausibly perform on YouTube Shorts / TikTok / Reels. This is an exhaustive identification pass — there is no quota, no minimum, no maximum. Return the honest count of viable clips this window actually contains: that may be zero, one, three, or twenty. Refinement will tighten boundaries downstream.

Return ONLY one valid JSON object. No markdown, no commentary, no reasoning, no preamble, no postamble.

══════════════════════════════════════════════════
VIDEO METADATA
══════════════════════════════════════════════════
Title       : {meta.get('title', 'Unknown')}
Channel     : {meta.get('channel', 'Unknown')}
Duration    : {meta.get('duration', 0)} seconds
Description : {meta.get('description', 'N/A')[:500]}

══════════════════════════════════════════════════
TRANSCRIPT WINDOW
══════════════════════════════════════════════════
Each line uses the format [MM:SS.ss -> MM:SS.ss] text. These ranges are the ONLY valid boundary cites — copy them verbatim into the output.
{transcript_note}
{transcript_text}

══════════════════════════════════════════════════
WHAT MAKES A CLIP "VIRAL" (use this rubric)
══════════════════════════════════════════════════
Every clip needs all three:
  1. HOOKY START — first ~2 seconds promise a curiosity gap, contradiction, stakes, shock, vivid scene, or bold claim. NOT a warm-up, NOT filler ("so, today..."), and NOT a vague opener that points at something unstated ("kind of like this thing", "what that even means", "the reason I say that is", "very difficult to argue against that"). The opening words must state the actual point.
  2. ONE CONCRETE, LEARNABLE IDEA — exactly ONE memorable point the viewer walks away having LEARNED: a counterintuitive fact, an actionable method, a personal reveal, an opinion that triggers debate. It must be COMPLETE (setup → point), never a half-finished fragment. No muddled breadth. If a passage has no self-contained lesson, skip it.
  3. PROPER ENDING — lands ON the payoff: a punchline, reveal, callback, the final item of a promised sequence, or a sharp closing claim. End the moment the payoff lands (mid-sentence is fine — captions are uppercased). NEVER trail PAST the payoff into setup for the next idea, a new example, or the next topic, and never end on "but / and / so / because" dangling.

Prefer clips that:
  • Stand alone with zero prior context.
  • Are audio-only intelligible (don't depend on "look at this graph").
  • Contain numbers, surprising claims, lists, or "most people get this wrong"-style framings.
  • Have an emotional or contrarian spike — laughter, shock, anger, revelation.

DISQUALIFY only if it is:
  ✗ Self-promotion / sponsor read / "like and subscribe" / channel-trailer / "in this video we'll".
  ✗ Pure filler with no idea, claim, story, or emotion.
  ✗ Logistical chatter (audio check, scheduling, "let's get started").

══════════════════════════════════════════════════
CHAPTERS — A SCAN MAP, NOT A CUTTING GUIDE (READ THIS BEFORE EMITTING)
══════════════════════════════════════════════════
You will emit a "chapters" array that catalogues the distinct topics in this window. Chapters are a REFERENCE so you scan the whole window and miss nothing — they are NOT the clips themselves, and a chapter's start/end is NOT a clip's start/end.

    ► A chapter only POINTS to where a clip might live. The actual clip is found INSIDE the chapter and cut to its own boundaries.

For each chapter, hunt for the moment(s) that satisfy the viral rubric (hooky start + one concrete idea/meaning/controversy + clean end). Then cut the clip to THAT moment — find its real opening sentence, its real payoff, and its real closing line inside the chapter, and stitch out intra-topic filler. Do NOT copy a chapter's start_frame/end_frame as a candidate's boundaries, and do NOT export a whole chapter verbatim as a clip.

How many candidates a chapter yields (there is no quota):
  • A rich chapter with several standalone insights → several candidates.
  • A chapter with one clean arc → one candidate, tightened to its real start/end (not the chapter span).
  • A chapter that is all setup / filler / digression with no self-contained arc → ZERO candidates. Do not force one.
  • NEVER merge content from DIFFERENT chapters into a single candidate. Cross-chapter content = different candidates. Stitching is INTRA-chapter only (see SEGMENTS & STITCHING below).

══════════════════════════════════════════════════
CANDIDATES LIVE INSIDE CHAPTERS, NEVER ON CHAPTER EDGES
══════════════════════════════════════════════════
Chapters are HEADERS for your own planning. They are NOT clips.
A chapter's start_frame and end_frame are NEVER a candidate's boundaries.
For every chapter, you must HUNT INSIDE it and find the actual viral moment(s) with their OWN boundaries.

ANTI-PATTERNS (do the OPPOSITE):
  ✗ COLLAPSING — welding several chapters into one mega-clip. This destroys the other clips.
      CORRECT: One chapter → multiple separate candidates, each with its own boundaries.
  ✗ DUMPING — emitting a chapter's full raw span as the clip with no inner tightening. This gives weak warm-up starts and trailing mid-thought ends.
      CORRECT: Find the tight hook→idea→payoff arc INSIDE the chapter, then cut it to its own start_words and end_words.

CORRECT:
  Walk each chapter → locate the self-contained viral moment(s) inside it → cut each to its own proper
  start_words / end_words (mid-line is normal) → stitch out intra-topic filler. Some chapters yield several
  clips, some yield exactly one, some yield none. Completeness of coverage matters; padding to match the
  chapter count does not.

══════════════════════════════════════════════════
SEGMENTS & STITCHING — MULTI-SEGMENT IS THE DEFAULT
══════════════════════════════════════════════════
A PERFECT clip has ZERO filler in the middle. Speakers naturally digress — they hedge ("now, this is just correlation, not causation, but..."), restate themselves, tell a tangential story, hesitate, repeat the question, or wander into a related-but-distinct topic before circling back to the payoff. Your job is to CUT those digressions out by splitting the clip into multiple segments and stitching only the on-arc parts.

Treat multi-segment stitching as the EXPECTED behavior for anything longer than ~20s. Single-segment clips are only correct when the speaker delivers the entire arc cleanly with no digression. In a typical 10-15 minute talk, most worthwhile clips are 2-, 3-, or 4-segment stitches — not single ranges.

Hard rules for stitching:
  • Segment count is EVIDENCE-BASED, not fixed. A clip with zero filler gets exactly 1 segment. A clip with one digression gets exactly 2 segments. A clip with two digressions gets 3 segments. Count the number of on-topic pieces and emit that many segments.
  • NEVER default to 1 segment for simplicity. If you look at a candidate and think "I'll just make it one segment to avoid work," you are DUMPING. Only 1 segment is correct when the speaker delivers the entire arc with zero filler.
  • Stitching is INTRA-CHAPTER ONLY. Every segment of a stitched clip must belong to the SAME chapter / SAME topic / SAME single arc. If two segments cover different topics, they are TWO candidates — never one stitched candidate.
  • There is NO upper cap on segment count. Use 1, 2, 3, 4, 5, 6, or more — whatever the SAME-TOPIC clip genuinely needs to be fluff-free.
  • TRIGGER TO SPLIT (INTO MORE SEGMENTS, NOT INTO MORE CANDIDATES): if keeping the clip as a single segment would force you to include more than ~5 seconds of off-topic filler / hedging / restatement / tangential story WITHIN the same topic, you MUST split and stitch the on-topic parts.
  • Each segment must itself begin and end on a complete sub-thought — never mid-sentence and never on "and / but / so / because / the / a / to ...". The stitch points must be clean on both sides.
  • Stitching must preserve meaning: the joined segments must read as one coherent idea progressing forward. Stitch parts of the SAME arc that filler interrupted — never weld together different ideas.
  • ALWAYS prefer a tight 3-segment stitch (intra-topic) over a 1-segment clip padded with 30s of meandering. Compactness with completeness is the goal.

── WORKED STITCHING EXAMPLE ──
A speaker delivers a hook, then digresses about methodology for ~15s, then returns to the concrete payoff. The transcript looks roughly like:

  [00:39.06 -> 00:44.83] the very best performing students study three or four hours per day
  [00:44.83 -> 00:54.70] now, you might say they're the best because they study, or they study because they're the best — correlation vs causation, that whole debate
  [00:54.70 -> 01:08.47] but that's not the point. anyway, getting back to it, what they do is they schedule time, eliminate distractions, study alone
  [02:21.60 -> 02:42.21] they study three or four hours per day, broken into two or three sessions, five days a week — that's the pattern

WRONG (single segment, includes 15s of correlation digression):
  one segment from 00:39.06 to 02:42.21  ← bloated, viewer disengages mid-clip

RIGHT (multi-segment, digression cut out):
  segment 1: start_words "the very best performing students study three or four hours per day"
             end_words "study three or four hours per day"
  segment 2: start_words "what they do is they schedule time, eliminate distractions, study alone"
             end_words "schedule time, eliminate distractions, study alone"
  segment 3: start_words "they study three or four hours per day, broken into"
             end_words "five days a week — that's the pattern"

The stitched result is ~35s of pure signal: hook + method + concrete schedule.

══════════════════════════════════════════════════
JSON OUTPUT EXAMPLE — MULTI-SEGMENT CLIP (READ THIS)
══════════════════════════════════════════════════
Below is a real transcript block. The WRONG output copies the whole block as one segment. The RIGHT output splits it into 2 segments, cutting the filler out.

TRANSCRIPT BLOCK:
  [03:12.50 -> 03:18.20] The single biggest mistake people make with their morning routine
  [03:18.20 -> 03:29.00] is they check their phone first. Now, I used to do this too, and I thought it was harmless, but the research shows it spikes cortisol
  [03:29.00 -> 03:45.00] and actually, before we get into the science, let me tell you a quick story about my client Sarah
  [03:45.00 -> 04:02.00] and the science is really clear. When you check your phone within 10 minutes of waking, you are essentially letting 1,000 people decide your mood
  [04:02.00 -> 04:11.00] before you even decide it yourself. So the fix is simple. Put your phone in another room for the first hour.

WRONG OUTPUT (DUMPING — one segment copying the whole block):
  segments: [
    {{"start_words": "The single biggest mistake people make with their morning routine", "start_frame": "[03:12.50 -> 03:18.20]", "end_words": "another room for the first hour", "end_frame": "[04:02.00 -> 04:11.00]"}}
  ]
  estimated_duration_seconds: 119
  ← WRONG: too long, includes the story about Sarah + filler. The boundaries match the block edges, not the actual clip.

RIGHT OUTPUT (2 segments, filler cut out):
  segments: [
    {{"start_words": "The single biggest mistake people make with their morning routine", "start_frame": "[03:12.50 -> 03:18.20]", "end_words": "spikes cortisol", "end_frame": "[03:18.20 -> 03:29.00]"}},
    {{"start_words": "When you check your phone within 10 minutes of waking", "start_frame": "[03:45.00 -> 04:02.00]", "end_words": "before you even decide it yourself", "end_frame": "[04:02.00 -> 04:11.00]"}}
  ]
  estimated_duration_seconds: 43
  ← RIGHT: The story about Sarah (03:29.00 -> 03:45.00) is CUT OUT. Segment 1 ends mid-line at "spikes cortisol". Segment 2 starts mid-line at "When you check your phone". The clip is tight and fluff-free.

══════════════════════════════════════════════════
DURATION — ESTIMATE AS YOU GO
══════════════════════════════════════════════════
The transcript is time-coded, so you can estimate clip duration yourself. For EACH segment compute roughly:

    segment_seconds ≈ (start-value of end_frame) − (start-value of start_frame)

For a segment with start_frame `[00:39.06 -> 00:44.83]` and end_frame `[02:36.53 -> 02:42.21]`:
    segment_seconds ≈ 156.53 − 39.06  ≈  117  seconds

Total clip duration ≈ SUM of all segment_seconds (gaps between segments are CUT, so they don't count).

Emit an "estimated_duration_seconds" integer field per candidate carrying this rough sum. It is a SELF-CHECK to keep you honest about length — it is NOT a strict gate.

Length guidance (the band is a ROUGH guide, NOT a hard cap):
  • Rough target: each clip total {min_duration}–{max_duration}s.
  • If the natural complete thought is shorter than {min_duration}s, EXTEND through the next related sentence(s) to reach the floor — do not submit under-length clips that will need programmatic stitching afterward.
  • Running OVER {max_duration}s is fine and PREFERRED when the extra time makes the SAME idea land more clearly/completely. Never truncate mid-thought to hit a number; never pad with unrelated filler; never extend into a DIFFERENT idea or topic just to add length.

══════════════════════════════════════════════════
HOW TO CITE BOUNDARIES (CRITICAL — read carefully)
══════════════════════════════════════════════════
A transcript LINE usually contains several sentences, and the line boundaries do
NOT line up with sentence boundaries. You must pick the words where the CLIP'S
THOUGHT actually begins and ends — these are almost always IN THE MIDDLE of a
line, not the line's first/last words.

For EACH segment of the clip, give:
  • "start_words" — the first 6–10 words of the clip's OPENING SENTENCE, copied
    verbatim. This is where a viewer should hear the clip begin. It may start
    mid-line. Pick the start of a complete, standalone sentence/thought — never
    a fragment, connector, or the tail of a previous idea.
  • "start_frame" — the [MM:SS.ss -> MM:SS.ss] of the transcript line that
    CONTAINS those start words, copied verbatim.
  • "end_words"   — the last 6–10 words of the clip's PAYOFF, copied verbatim.
    This is where the clip should stop — the moment the payoff lands. It may end
    mid-line, and ending mid-sentence ON the payoff is fine (captions are
    uppercased, so a trailing period is not needed). A sentence terminator is
    nice when it coincides with the payoff, but do NOT extend past the payoff
    just to reach one — trailing into the next thought/example/topic buries the
    payoff and ships a cliffhanger. Never end on a dangling "but/and/so/because/
    the/a/to/when/while/that…". Only extend end_words forward when the payoff
    itself finishes one short clause later.
  • "end_frame"   — the [MM:SS.ss -> MM:SS.ss] of the line containing end_words.

A Python matcher finds the EXACT word-level time of start_words and end_words in
the full transcript, so the clip begins and ends precisely on those words — NOT
at the line boundaries. That is why picking the right mid-line phrases matters.
Do NOT output scalar start_time / end_time — they are resolved later.

── WORKED EXAMPLE ──
Lines:
  [04:50.77 -> 04:58.14] attention. Are a limited but renewable resource in the human brain. The longer you're awake, the more is the buildup
  [04:58.14 -> 05:02.94] of a molecule called adenosine in your brain and body. It makes you sleepy, it makes it harder to focus.
  [05:02.94 -> 05:08.23] When you sleep, adenosine levels are pushed down again, you're able to focus again, you feel more alert. You can
To clip the adenosine idea, the thought STARTS mid-line at "The longer you're awake"
and ENDS mid-line at "you feel more alert":
  ✓ start_words: "The longer you're awake, the more is the buildup"   start_frame: "[04:50.77 -> 04:58.14]"
  ✓ end_words:   "pushed down again, you're able to focus again, you feel more alert"   end_frame: "[05:02.94 -> 05:08.23]"
  ✗ WRONG start_words: "attention. Are a limited but renewable" (line start — cuts mid-thought)
  ✗ WRONG end_words:   "you feel more alert. You can" (line end — trails into the next thought)

══════════════════════════════════════════════════
OUTPUT SCHEMA  (LEAN — youtube fields come later in refinement)
══════════════════════════════════════════════════
{{
  "summary": "One-sentence summary of this window",
  "chapters": [
    {{"start_frame": "[MM:SS.ss -> MM:SS.ss]", "end_frame": "[MM:SS.ss -> MM:SS.ss]", "topic": "Title", "summary": "Short"}}
  ],
  "candidates": [
    {{
      "rank": 1,
      "working_title": "Plain, truthful, descriptive title — refinement will catchify",
      "takeaway": "One sentence stating the concrete idea, meaning, or controversy this clip delivers",
      "hook_score": 8,
      "flow_score": 8,
      "virality_score": 8,
      "meaning_score": 8,
      "completeness_score": 8,
      "boundary_score": 8,
      "arc_pattern": "hook_idea_payoff | struggle_discovery_win | controversy_defense | quote_drop | custom",
      "reason": "One sentence — why this is viral material",
      "estimated_duration_seconds": 47,
      "segments": [
        {{
          "start_words": "exact first up to ten words of this segment verbatim",
          "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words":   "exact last up to ten words of this segment verbatim",
          "end_frame":   "[MM:SS.ss -> MM:SS.ss]"
        }}
      ]
    }},
    {{
      "rank": 2,
      "working_title": "Second clip — multi-segment example with mid-filler cut out",
      "takeaway": "One sentence on this clip's distinct idea",
      "hook_score": 9, "flow_score": 8, "virality_score": 9,
      "meaning_score": 8, "completeness_score": 9, "boundary_score": 8,
      "arc_pattern": "hook_idea_payoff",
      "reason": "One sentence — why this is viral material",
      "estimated_duration_seconds": 38,
      "segments": [
        {{
          "start_words": "verbatim hook phrase from the transcript",
          "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words":   "verbatim closing of the hook portion",
          "end_frame":   "[MM:SS.ss -> MM:SS.ss]"
        }},
        {{
          "start_words": "verbatim resumption after the filler is cut",
          "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words":   "verbatim payoff line",
          "end_frame":   "[MM:SS.ss -> MM:SS.ss]"
        }}
      ]
    }}
    // The two candidates above are FORMAT EXAMPLES ONLY (one single-segment, one multi-segment). They do NOT prescribe a count. Emit one candidate per viable clip in this window — the honest count, whether that is 0, 2, 7, 15, or more.
  ]
}}

══════════════════════════════════════════════════
HARD RULES
══════════════════════════════════════════════════
  • Return only JSON. Double-quoted keys/strings. No trailing commas. No comments.
  • CHAPTERS ARE A COVERAGE CHECK, NOT A QUOTA: use the chapters array to confirm you scanned the whole window and skipped nothing. A chapter yields a candidate ONLY if it actually contains a self-contained hooky-start + concrete-idea + clean-end moment; chapters that are pure setup / filler / digression yield none. Never pad to match the chapter count, and never emit a chapter's raw span as a clip just to have one — always cut the clip to its real boundaries inside the chapter.
  • IDENTIFY EVERY VIABLE CLIP — no minimum, no maximum, no density target. Walk the transcript from the first line to the last and emit one candidate for every distinct {min_duration}–{max_duration}s arc that satisfies hooky-start + one-concrete-idea + clean-end. If the window honestly contains zero viable clips, return zero. If it contains twenty, return twenty. The number is whatever the window contains — never invent, never pad, never withhold.
  • NEVER condense multiple distinct chapters / topics into one stitched candidate. Stitching is INTRA-CHAPTER ONLY. Cross-chapter content = separate candidates.
  • Each candidate must be SELF-CONTAINED and FLUFF-FREE. If filler interrupts the arc, split into multiple segments (within the same chapter) and cut the filler.
  • Multi-segment stitching is the EXPECTED behavior for clips longer than ~20s. Single-segment clips are correct only when the speaker delivers the entire arc cleanly with zero digression. When in doubt, split.
  • Emit "estimated_duration_seconds" per candidate (sum of per-segment seconds, using start_frame/end_frame math). Use it as a self-check, not a strict gate — a complete clip beats one chopped to hit a number.
  • start_words / end_words must be VERBATIM transcript substrings — the local matcher checks them. Paraphrased phrases break the pipeline.
  • All timeframes must be MM:SS.ss strings copied verbatim from the transcript lines above.
  • DO NOT output scalar start_time / end_time, hook_phrase, youtube_title, hashtags, description — those belong to the refinement stage.
  • CANDIDATES ARE THE PRIMARY OUTPUT. Chapters are secondary coverage checks. Do not spend more effort on chapters than on candidates.

══════════════════════════════════════════════════
PRE-FLIGHT CHECK (answer before returning)
══════════════════════════════════════════════════
1. Did I return ONLY candidates whose boundaries fall inside chapters (not on chapter edges)?
2. Did any candidate copy a chapter's start_frame or end_frame as its own boundary? If yes, fix it — find the real clip inside.
3. Did I emit the CORRECT number of segments per candidate? (1 segment = zero filler; 2+ segments = filler was cut out.)
4. Are there any chapters I scanned but forgot to emit candidates from? If yes, re-scan them.
5. Did I use start_words and end_words that are VERBATIM from the transcript? (Not paraphrased.)
"""


def _build_judge_prompt(
    meta: dict,
    candidates: list[dict],
    clip_count: int,
    min_duration: int,
    max_duration: int,
) -> str:
    """Judge/ranking prompt. OFF by default (config.SKIP_JUDGE=True) but fully
    functional when re-enabled. Ranks the candidate pool and returns the
    strongest `clip_count`, preserving candidate_id and the segment anchor
    schema so the local matcher still resolves boundaries afterwards.
    """
    candidate_text = _candidate_pack_for_prompt(candidates, include_context=False)
    return f"""You are the final editorial judge for short-form viral clips. From the candidate pool below, select and rank only the strongest. Fewer excellent clips beat more mediocre ones — never pad.

Return ONLY one valid JSON object. No markdown, no commentary, no reasoning.

VIDEO METADATA
Title  : {meta.get('title', 'Unknown')}
Channel: {meta.get('channel', 'Unknown')}

SELECTION CRITERIA (a winning clip satisfies all):
  - Hooky opening that lands in the first ~2 seconds.
  - Exactly one concrete idea, meaning, or controversy.
  - A clean, satisfying ending (payoff / reveal / final list item) — never mid-thought.
  - Understandable with zero prior context; audio-only intelligible.
  - Not self-promotion / sponsor / logistical filler.
  - Does not substantially overlap a higher-ranked pick.

CANDIDATE POOL
{candidate_text}

OUTPUT SCHEMA — return the best up to {clip_count} candidates, ranked best-first.
Preserve each candidate's "candidate_id" and its "segments" array UNCHANGED
(same start_words / start_frame / end_words / end_frame anchors).
{{
  "summary": "One sentence about the selected set",
  "candidates": [
    {{
      "candidate_id": "preserve verbatim",
      "rank": 1,
      "working_title": "keep or lightly improve",
      "takeaway": "one sentence",
      "hook_score": 9, "flow_score": 9, "virality_score": 9,
      "meaning_score": 9, "completeness_score": 9, "boundary_score": 9,
      "reason": "one sentence why it ranks here",
      "arc_pattern": "keep",
      "segments": [
        {{"start_words": "verbatim", "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words": "verbatim", "end_frame": "[MM:SS.ss -> MM:SS.ss]"}}
      ]
    }}
  ]
}}

HARD RULES:
  • Return at most {clip_count} candidates, ranked best-first. Do not pad.
  • Preserve candidate_id and the segment anchors verbatim — downstream local
    matching depends on them.
  • Do NOT output scalar start_time / end_time. Return only valid JSON.
"""


def _build_batched_compression_prompt(
    meta: dict,
    candidates: list[dict],
    transcript: dict,
    min_duration: int,
    max_duration: int,
) -> str:
    """Build a focused compression prompt for a batch of overlong clips."""
    segments = transcript.get("segments", []) or []
    
    def _ts_to_sec(ts) -> float:
        try:
            if isinstance(ts, (int, float)):
                return float(ts)
            mm, ss = str(ts).split(":")
            return int(mm) * 60 + float(ss)
        except Exception:
            return 0.0

    candidates_sections = []
    for cand in candidates:
        cand_segments = cand.get("segments", []) or []
        if not cand_segments:
            continue
        
        overall_start = _ts_to_sec(cand_segments[0].get("start", "0:00.00")) - 10.0
        overall_end = _ts_to_sec(cand_segments[-1].get("end", "0:00.00")) + 10.0

        slice_lines = []
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            if seg_end >= overall_start and seg_start <= overall_end:
                slice_lines.append(
                    f"[{_fmt_time(seg_start)} -> {_fmt_time(seg_end)}] {str(seg.get('text', '')).strip()}"
                )

        transcript_slice = "\n".join(slice_lines)
        arc_pattern = cand.get("arc_pattern", "unknown")
        arc_roles = cand.get("arc_roles", [])
        total_dur = _candidate_duration(cand)
        
        candidates_sections.append(f"""---
CANDIDATE:
Candidate ID     : {cand.get('candidate_id', '?')}
Title            : {cand.get('title', '?')}
Current duration : {total_dur:.1f}s
Target max       : {max_duration}s
Minimum floor    : {min_duration}s
Arc pattern      : {arc_pattern}
Arc roles        : {arc_roles}

TRANSCRIPT SLICE:
Each line uses [MM:SS.ss -> MM:SS.ss] text.
{transcript_slice}
""")

    all_candidates_text = "\n".join(candidates_sections)

    return f"""You are a precision clip compressor. The following clips are too long and must be tightened without losing meaning.

Return ONLY one valid JSON object. No markdown, no commentary, no reasoning.

VIDEO METADATA
Title  : {meta.get('title', 'Unknown')}
Channel: {meta.get('channel', 'Unknown')}

CLIPS TO COMPRESS:
{all_candidates_text}

TASK:
For each candidate clip, select the smallest chronological set of transcript lines that preserves the complete core argument, hook, and payoff.
Keep the strongest hook sentence, the main explanation, and the true close.
Skip filler, repetition, tangents, false starts, and transition language.
Do not break numbered sequences.
Use at most 15 segments per clip.
The output must still feel like one complete clip.

OUTPUT SCHEMA:
{{
  "candidates": [
    {{
      "candidate_id": "candidate_id_from_above",
      "compression_note": "One sentence explaining what was removed and why the core remains intact",
      "compressed_segments": [
        {{"start": "MM:SS.ss", "end": "MM:SS.ss"}}
      ]
    }}
  ]
}}

Rules:
1. "start" and "end" timestamps in "compressed_segments" MUST EXACTLY match segment start/end timestamps from the clip's transcript slice above. Do not invent new times.
2. The list of segments must be strictly chronological.
3. Every candidate in the request must have a corresponding entry in the "candidates" response list.
"""


def _build_boundary_refinement_prompt(
    meta: dict,
    candidates: list[dict],
    transcript: dict,
    clip_count: int,
    min_duration: int,
    max_duration: int,
) -> str:
    """Refinement prompt — handles N candidates per call (batched).

    Contract:
      • Input: N candidates (1 <= N <= REFINEMENT_BATCH_SIZE).
      • Output: candidates[] MUST contain exactly N entries, one per input,
        preserving each candidate_id verbatim.
      • ZERO rejection — every input must be returned. Fix what you can.
      • PRIMARY job is per-segment boundary tightening (start_words / end_words
        verbatim, phrase-anchored). Metadata is generated in the same pass so
        the call is efficient, but boundaries are the focus.
      • Per-candidate segment-count rule: each output candidate's segments[]
        length >= its input segments count. NEVER merge or drop segments.
        Stitching of under-length CLIPS is a separate downstream pass, not
        this one.
    """
    segments = transcript.get("segments", []) or []
    n = max(1, len(candidates))
    candidate_text = _candidate_pack_for_prompt(candidates, include_context=True, transcript_segments=segments)
    seg_count_summary = ", ".join(
        f"{c.get('candidate_id', f'cand_{i+1:02d}')}={len(c.get('segments', []) or [])}seg"
        for i, c in enumerate(candidates)
    )
    return f"""══════════════════════════════════════════════════
YOUR JOB — READ THIS FIRST
══════════════════════════════════════════════════
You are a precision clip refiner working on {n} clip(s) IN ONE CALL. For each clip, lock in THREE boundary targets — START, MEANING, END — then generate the metadata.

  • TARGET 1 — START (the hook): the clip MUST open on a strong line that pulls a viewer in — a curiosity gap, bold claim, stakes, contradiction, or vivid scene. Strip leading filler/connectors ("um", "yeah", "so", "and", "well", "now") AND vague/soft openers that say nothing on their own. Opening mid-sentence is FINE if those first words ARE the hook.
      ✗ SOFT/VAGUE openers to AVOID (they reference something unstated): "kind of like this thing...", "what that even means...", "the reason I say that is...", "very difficult to argue against that...", "it was like...", "that's the thing...", "so anyway...". Push the start forward to the first words that state the actual point.
      ✓ HOOK openers (start here): "I'm not a natural anything", "failure to me is the norm", "the longest ever study on human happiness found...", "I'll be happy when I get that promotion — we all do this".
  • TARGET 2 — MEANING LOCK (a complete, learnable point): each clip carries a "takeaway" — that ONE idea is the clip, and the viewer must finish it having LEARNED something whole, not a fragment. Keep the boundaries on exactly that idea: start where it is set up, end where it fully pays off / the lesson lands. Do NOT broaden into an adjacent idea, the next topic, or the interviewer's pivot to a new question. Numbered lists keep all items; a promise in the opening must be answered. A longer clip that delivers the COMPLETE point is better than a short one that cuts the lesson off — length is flexible. If the lesson isn't fully inside, you have NOT refined the clip.
  • TARGET 3 — END (finish the thought, on a complete phrase): the clip MUST end where the takeaway's payoff fully LANDS — on a COMPLETE sentence/clause, never mid-phrase on a fragment ("...so it's great for your", "...getting back to a less", "...okay last question if"). Extend the end as far as needed to finish the sentence and let the lesson land clearly — the duration band is a ROUGH guide, NOT a cap, so running longer to deliver a complete point is CORRECT and beats a clipped one. The ONE hard rule: stay on the SAME idea — never drift into the speaker's NEXT thought, a new example, or a topic change (that, not length, is the failure). If the end sits inside the NEXT idea, pull back to where THIS idea finishes. End on a content word — never on "and / but / the / to / your / a / for / um" or any half-finished phrase.

ALWAYS return exactly {n} candidate(s) in candidates[] — one for each input, in the same order, preserving each candidate_id verbatim. No rejections, no empty arrays. If boundaries are already perfect, return them unchanged and just add the metadata.

Return ONLY one valid JSON object. No markdown, no commentary, no reasoning, no preamble, no postamble.

SPEAKER-INDEPENDENCE NOTE (important):
The transcript may contain multiple speakers interleaved (interviewer + guest, host + caller, etc.). DO NOT treat speaker changes as boundary signals. The boundary you pick is about MEANING, not about who's talking. If the payoff comes from speaker B replying to speaker A's question, the clip naturally spans both — that is correct.

══════════════════════════════════════════════════
VIDEO METADATA
══════════════════════════════════════════════════
Title       : {meta.get('title', 'Unknown')}
Channel     : {meta.get('channel', 'Unknown')}
Duration    : {meta.get('duration', 0)} seconds
Description : {meta.get('description', 'N/A')[:500]}

══════════════════════════════════════════════════
CANDIDATES TO REFINE  ({n} clip(s); per-clip input segment counts: {seg_count_summary})
══════════════════════════════════════════════════
{candidate_text}

══════════════════════════════════════════════════
JOB A — REFINE BOUNDARIES (per segment, phrase-anchored)
══════════════════════════════════════════════════
Per-candidate segment-count rule (applies INDEPENDENTLY to each output clip):
  • Each output candidate's segments[] count >= that same candidate's INPUT segments count.
  • Equal is correct when the discovery stitch is already filler-free.
  • MORE is correct only when you spot fresh intra-topic filler that discovery missed inside one of THAT clip's segments — split it into two/three and cut the filler.
  • NEVER drop a segment, NEVER merge two segments into one, NEVER collapse a multi-segment stitch into a single range, NEVER reorder segments.
  • Stitching is intra-topic only — every output segment must still belong to the same topical arc as that clip's input.
  • Do NOT attempt to merge two separate input CANDIDATES — that's a different downstream pass. Treat each candidate independently.

Transcript lines hold several sentences; line edges are NOT sentence edges. Pick the words where the clip's THOUGHT actually begins and ends — usually MID-LINE.

For each output segment of each candidate provide:
  • "start_words" — first 6–10 words of THIS segment's opening sentence, copied verbatim from the transcript. Must hit TARGET 1 (the hook). A complete standalone sentence; may begin mid-line; never a fragment, never a connector, never the tail of a previous idea, never filler like "um/yeah/right/so".
  • "start_frame" — [MM:SS.ss -> MM:SS.ss] of the transcript line that contains start_words, copied verbatim.
  • "end_words"   — last 6–10 words of THIS segment's closing sentence, verbatim. Must hit TARGET 3 (clean close). Ends on a complete sentence terminator (./?/!); never trail on "but/and/so/because/the/a/to…".
  • "end_frame"   — [MM:SS.ss -> MM:SS.ss] of the line containing end_words.

A Python matcher pins these phrases to word-exact times. Phrases MUST be verbatim transcript substrings — paraphrasing breaks the matcher.
  ✗ WRONG: copying the line's first/last words when the sentence actually starts/ends mid-line (produces abrupt, half-baked clips).
  ✗ WRONG: ending on the literal last word that appeared in the candidate's input window when the very next sentence in the post-context actually completes the payoff. ALWAYS read 1–2 sentences past the input window; extend if they finish the thought.

Per-segment tightening checklist (apply to EACH segment of EACH candidate individually):
  • START — Move earlier only to recover a hook that begins 1–2 lines before; move later to drop filler/warm-up that delays the hook. Anchor on the strongest hook line available — opening mid-sentence is fine when those words ARE the hook.
  • MEANING — Verify the takeaway's idea is intact from start to end. If a promise / list / setup-punchline arc gets cut short, extend just far enough for it to land. Never leave an unanswered setup — and never widen past the takeaway into a second idea.
  • END — Land where the takeaway's payoff is CLEAR and complete. Extend forward as far as the SAME idea needs to land clearly (duration is a rough guide, not a cap — a longer clip that fully delivers the point beats a clipped one). Pull the end BACK only if it has crossed into the NEXT idea / a new example / a topic change. Never end on a dangling "and / but / so / the / to / because / um"; never chase a period past the payoff.
  • Keep numbered sequences intact — if N items are promised, ALL N must be inside the clip across whichever segment(s) they live in.
  • Never open a segment with: and, but, or, so, yet, nor, because, since, although, though, which, that, who, however, therefore, thus, hence, consequently, moreover, furthermore, additionally, also, including, such as, for example, for instance, in other words, as well, namely, specifically.

DURATION (ROUGH guide — completeness ALWAYS wins, the band is NOT a cap):
  • Rough band: each clip ≈ 30–{max_duration}s (sum of per-segment seconds — gaps between segments are CUT and don't count). Going OVER to land a clear, complete payoff is correct — never truncate the meaning to fit a number.
  • A clean 29s clip with a complete idea is KEPT. A 100s clip is KEPT if every second serves the SAME idea and it pays off clearly.
  • If a clip is naturally short (under 30s) even after honest tightening, RETURN IT AS-IS — a separate stitching pass will combine it with a neighbor. Do NOT pad to hit duration. Do NOT extend into unrelated material.
  • If a clip is over {max_duration}s but still telling one coherent story, KEEP IT. Do NOT chop a sentence to hit the ceiling. Do NOT truncate mid-thought to land in the band.
  • Never sacrifice MEANING (Target 2) or a clean END (Target 3) to chase a duration number.

══════════════════════════════════════════════════
JOB B — GENERATE METADATA  (cliffhanger hook + full YouTube fields, PER CLIP)
══════════════════════════════════════════════════
youtube_title:
  Under 78 chars. **CLICKBAITY but accurate** — lead with curiosity gap, contradiction, stakes, shock, or a specific surprising number. NEVER promise something the clip does not actually deliver. May include ONE emoji + ONE hashtag (optional, at the end).

  Pick ONE hook pattern: NEGATIVE FRAMING / THE ONE THING / SPECIFIC SHOCKING NUMBER / CURIOSITY-GAP QUESTION / CONTRADICTION-MYTH-BUST / PERSONAL STAKE / NEGATIVE-RESULT WARNING.

  GOOD: "Top students never study for 3 hours straight — here's why 🧠 #studytips"
  GOOD: "The one habit that wrecked his focus for 5 years"
  BAD : "Andrew Huberman discusses study habits"   ← descriptive, no hook
  BAD : "An interesting clip about learning"        ← LLM-sounding fluff

description_text:
  Plain English. 2–4 sentences. State WHAT is in the clip, the concrete idea or controversy, and the payoff. No LLM fluff, no "Check out".

description_hashtags:
  Aim for 30–40 hashtags. Ideal split: 20 topical + 20 viral. Output as ONE space-separated string (NOT a JSON array).

youtube_tags:
  Comma-separated tag string, aim for ~500 characters total: first ~250 chars topical, last ~250 chars viral/discovery.

hook_phrase:
  THIS IS THE LITERAL STRING SENT TO TEXT-TO-SPEECH. Cliffhanger style — pose a curiosity gap that the clip itself answers. Lowercase, no emoji/hashtag/quotes/markdown. 5–12 words. Spoken-ready.
  Good: "watch how this neuroscientist actually kills stress"
  Bad:  "check this out" / a description of the clip.

takeaway:
  One sentence — the single concrete idea/meaning/controversy this clip delivers.

══════════════════════════════════════════════════
OUTPUT SCHEMA  ({n} candidate object(s) in candidates[], one per input clip, IN ORDER)
══════════════════════════════════════════════════
{{
  "summary": "One sentence summarising what changed across the refined set",
  "candidates": [
    {{
      "candidate_id": "<copy this clip's input candidate_id verbatim>",
      "rank": 1,
      "takeaway": "One sentence — the concrete idea / meaning / controversy",
      "youtube_title": "Clickbaity-but-accurate title under 78 chars using one hook pattern (1 emoji + 1 hashtag optional)",
      "description_text": "Plain English description, 2-4 sentences, no LLM artifacts",
      "description_hashtags": "#topic1 #topic2 ... #viral1 #viral2 ...  ← 30–40 hashtags, space-separated",
      "youtube_tags": "first ~250 chars topical, then ~250 chars viral — ~500 chars total",
      "hook_phrase": "lowercase cliffhanger spoken-ready line for TTS",
      "hook_score": 9,
      "flow_score": 9,
      "virality_score": 9,
      "meaning_score": 9,
      "completeness_score": 9,
      "boundary_score": 9,
      "reason": "One sentence — why this clip lands",
      "arc_pattern": "hook_idea_payoff | struggle_discovery_win | controversy_defense | quote_drop | custom",
      "segments": [
        // >= this clip's INPUT segments count, in chronological order.
        {{
          "start_words": "verbatim first ≤10 words of segment 1",
          "start_frame": "[MM:SS.ss -> MM:SS.ss]",
          "end_words":   "verbatim last ≤10 words of segment 1",
          "end_frame":   "[MM:SS.ss -> MM:SS.ss]"
        }}
        // ...repeat for every segment of this clip
      ]
    }}
    // ...REPEAT the candidate object for every input clip ({n} total), in the same order as input
  ]
}}

══════════════════════════════════════════════════
HARD RULES
══════════════════════════════════════════════════
  • candidates[] MUST contain exactly {n} entry(ies) — one per input clip, in the same order.
  • Each output candidate MUST preserve its input candidate_id verbatim. Wrong/missing/swapped IDs corrupt the pipeline.
  • NEVER return an empty candidates array. NEVER drop a clip. NEVER merge two input clips into one.
  • Each candidate's segments[] count >= that candidate's INPUT segments count. NEVER fewer. NEVER merged. NEVER reordered.
  • start_words / end_words must be VERBATIM transcript substrings.
  • All timeframes must be MM:SS.ss strings copied verbatim from the nearby context above.
  • description_hashtags: 30–40 hashtags, space-separated. NOT a JSON array.
  • youtube_tags: ~500 chars, comma-separated.
  • youtube_title must use a hook pattern from the menu — clickbaity but accurate.
  • Do NOT output scalar start_time / end_time anywhere.
  • Return only JSON. Double-quoted keys/strings. No trailing commas. No comments.
"""


def _build_stitch_prompt(
    short_clips: list[dict],
    neighbor_map: dict[str, dict],
    min_duration: int,
    max_duration: int,
) -> str:
    """Build the clip-level stitch prompt.

    short_clips : list of resolved candidates whose total duration < min_duration
    neighbor_map: candidate_id -> {"prev": <candidate or None>, "next": <candidate or None>}
    """
    def _pack(c: dict, label: str, gap_sec: float | None = None) -> str:
        if not c:
            return f"{label}: (none)"
        dur = sum(max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
                  for s in c.get("segments", []) or [])
        first = (c.get("segments") or [{}])[0]
        last = (c.get("segments") or [{}])[-1]
        text = (c.get("transcript_evidence") or c.get("description") or c.get("title") or "")[:240]
        gap_str = "" if gap_sec is None else f" (gap from prev clip end: {gap_sec:.1f}s)"
        return (
            f"{label}: id={c.get('candidate_id','?')}, total_duration={dur:.1f}s{gap_str}\n"
            f"  start_words: {first.get('start_words','')}\n"
            f"  end_words:   {last.get('end_words','')}\n"
            f"  excerpt: {text}"
        )

    blocks: list[str] = []
    for idx, sc in enumerate(short_clips, 1):
        sc_id = sc.get("candidate_id", f"short_{idx:02d}")
        sc_dur = sum(max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
                     for s in sc.get("segments", []) or [])
        prev_c = neighbor_map.get(sc_id, {}).get("prev")
        next_c = neighbor_map.get(sc_id, {}).get("next")

        def _gap(a, b):
            if not a or not b:
                return None
            a_end = float((a.get("segments") or [{}])[-1].get("end", 0.0))
            b_start = float((b.get("segments") or [{}])[0].get("start", 0.0))
            return max(0.0, b_start - a_end)

        block = [
            f"════ SHORT CLIP {idx} (id={sc_id}, only {sc_dur:.1f}s — needs >= {min_duration}s) ════",
            _pack(sc, "SHORT CLIP", gap_sec=None),
            _pack(prev_c, "NEIGHBOR BEFORE", gap_sec=_gap(prev_c, sc)),
            _pack(next_c, "NEIGHBOR AFTER",  gap_sec=_gap(sc, next_c)),
        ]
        blocks.append("\n".join(block))

    body = "\n\n".join(blocks)

    return f"""══════════════════════════════════════════════════
CLIP-LEVEL STITCH PASS
══════════════════════════════════════════════════
You are a clip stitching judge. You are given several SHORT clips (each below the {min_duration}s minimum). For each one, you also see its immediate time-order neighbours (the candidate clip BEFORE it and AFTER it).

Your only job: decide which clips to combine so that under-length clips become viable. You DO NOT rewrite boundaries. You DO NOT touch the segments themselves. You only output GROUPS of clip ids to merge.

Rules of the game:
  • Each "merge group" is an ordered list of clip ids that should be combined into ONE final clip (their segments concatenated chronologically).
  • Topic match is NICE-TO-HAVE, not required — even loosely related ideas can be combined if it gives the viewer a more complete watch and stays under {max_duration}s total.
  • Prefer pairs (short + one neighbour). Triples (prev + short + next) only when both neighbours flow naturally with the short.
  • If the short clip CAN stand alone meaningfully OR its neighbours are clearly unrelated AND combining them would feel jarring, leave it alone (list it under "skipped").
  • NEVER include the same clip id in more than one merge group.
  • NEVER combine clips whose total duration would exceed {max_duration}s.
  • Do NOT invent new ids — only use ids that appear in the input below.
  • The order of ids inside a merge group determines chronological playback order — keep them in time order (BEFORE neighbour, then SHORT, then AFTER neighbour).

INPUT — under-length clips and their neighbours:

{body}

══════════════════════════════════════════════════
OUTPUT SCHEMA  (return ONLY this JSON, no markdown/commentary)
══════════════════════════════════════════════════
{{
  "summary": "One sentence describing what you stitched and why",
  "merges": [
    {{
      "merge_ids": ["<id_a>", "<id_b>"],
      "reason": "Why these belong together (topic flow, payoff, etc.)"
    }}
    // ...one entry per merge group
  ],
  "skipped": [
    {{
      "id": "<short_clip_id>",
      "reason": "Why this short clip should stay as-is (no good neighbour)"
    }}
  ]
}}

HARD RULES:
  • Only output JSON. Double-quoted keys. No trailing commas.
  • Each merge_ids list MUST contain >= 2 ids, in chronological order.
  • Every short-clip id from the input MUST appear in EITHER "merges" OR "skipped" — never both, never missing.
  • Use ids verbatim.
"""


def _stitch_validate(parsed: dict, expected_count: int, logger: logging.Logger) -> dict:
    """Lightweight validator for the stitch pass response."""
    if not isinstance(parsed, dict):
        return {"summary": "", "merges": [], "skipped": []}
    merges = parsed.get("merges") or []
    skipped = parsed.get("skipped") or []
    clean_merges: list[dict] = []
    seen_ids: set[str] = set()
    for m in merges:
        if not isinstance(m, dict):
            continue
        ids = m.get("merge_ids") or []
        ids = [str(x).strip() for x in ids if str(x).strip()]
        if len(ids) < 2:
            continue
        if any(i in seen_ids for i in ids):
            logger.warning(f"  [Stitch] dropping merge with already-used id(s): {ids}")
            continue
        seen_ids.update(ids)
        clean_merges.append({"merge_ids": ids, "reason": str(m.get("reason", ""))[:240]})
    clean_skipped = [
        {"id": str(s.get("id", "")).strip(), "reason": str(s.get("reason", ""))[:240]}
        for s in skipped if isinstance(s, dict) and s.get("id")
    ]
    return {
        "summary": str(parsed.get("summary", ""))[:400],
        "merges": clean_merges,
        "skipped": clean_skipped,
    }


def run_stitch_pass(
    job_dir: str,
    candidates: list[dict],
    transcript_segments: list,
    min_duration: int,
    max_duration: int,
    providers_to_try: list[tuple[str, str, str]],
    logger: logging.Logger,
    settings: dict | None = None,
    enable_reasoning: bool = False,
) -> list[dict]:
    """One API-driven pass that merges adjacent under-length clips.

    Input: candidates with resolved start/end times on every segment.
    Returns: a new candidate list with merge groups combined into single
    candidates (their segments concatenated). Candidates not involved in any
    merge are returned untouched.
    """
    settings = settings or {}
    if not candidates:
        return candidates

    def _dur(c: dict) -> float:
        return sum(max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
                   for s in c.get("segments", []) or [])

    # Sort by chronological start so "neighbour" means time-adjacent, not list-adjacent.
    by_start = sorted(
        candidates,
        key=lambda c: float((c.get("segments") or [{"start": 0}])[0].get("start", 0.0)),
    )
    short_clips = [c for c in by_start if _dur(c) < float(min_duration)]
    if not short_clips:
        logger.info("Stitch pass: no under-length clips, skipping.")
        return candidates

    pos = {id(c): i for i, c in enumerate(by_start)}
    neighbor_map: dict[str, dict] = {}
    for sc in short_clips:
        i = pos[id(sc)]
        prev_c = by_start[i - 1] if i - 1 >= 0 else None
        next_c = by_start[i + 1] if i + 1 < len(by_start) else None
        neighbor_map[sc.get("candidate_id", f"short_{i:02d}")] = {"prev": prev_c, "next": next_c}

    logger.info(
        f"Stitch pass: {len(short_clips)} under-length clip(s) — calling AI to decide merges"
    )

    prompt = _build_stitch_prompt(short_clips, neighbor_map, min_duration, max_duration)

    try:
        parsed = api_provider.run_json_task(
            job_dir,
            "clip_stitch_pass",
            prompt,
            providers_to_try,
            logger,
            validate=lambda p: _stitch_validate(p, len(short_clips), logger),
            enable_reasoning=enable_reasoning,
            clip_count=len(short_clips),
            temperature=getattr(config, "NVIDIA_TEMP_REFINE", 0.4),
            score=lambda p: len((p or {}).get("merges", []) or []),
            race_count=1,
        )
    except Exception as e:
        logger.warning(f"Stitch pass failed ({e}); keeping clips as-is.")
        return candidates

    merges = (parsed or {}).get("merges") or []
    if not merges:
        logger.info("Stitch pass: model produced no merges; keeping clips as-is.")
        return candidates

    by_id = {c.get("candidate_id"): c for c in candidates if c.get("candidate_id")}
    consumed: set[str] = set()
    merged_new: list[dict] = []

    for m in merges:
        ids = m.get("merge_ids") or []
        members = [by_id[i] for i in ids if i in by_id and i not in consumed]
        if len(members) < 2:
            logger.warning(f"  [Stitch] skipping merge — fewer than 2 known/unused members: {ids}")
            continue
        # Concatenate segments in chronological order.
        all_segs = []
        for member in members:
            all_segs.extend(member.get("segments") or [])
        all_segs.sort(key=lambda s: float(s.get("start", 0.0)))
        total = sum(max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0))) for s in all_segs)
        if total > float(max_duration) + 30.0:
            logger.warning(
                f"  [Stitch] skipping merge {ids} — combined {total:.1f}s exceeds max_duration "
                f"{max_duration}s + 30s slack"
            )
            continue
        # Inherit metadata from the strongest member (master_score, then virality).
        leader = max(
            members,
            key=lambda c: (
                float(c.get("master_score", 0.0) or 0.0),
                float(c.get("virality_score", 0.0) or 0.0),
            ),
        )
        merged_clip = dict(leader)
        merged_clip["segments"] = all_segs
        merged_clip["candidate_id"] = "merged_" + "+".join(m.get("merge_ids") or [])
        merged_clip["stitched_from"] = list(m.get("merge_ids") or [])
        merged_clip["stitch_reason"] = str(m.get("reason", ""))[:240]
        merged_clip["start_time"] = float(all_segs[0].get("start", 0.0))
        merged_clip["end_time"] = float(all_segs[-1].get("end", 0.0))
        merged_new.append(merged_clip)
        consumed.update(c.get("candidate_id") for c in members if c.get("candidate_id"))
        logger.info(
            f"  [Stitch] merged {ids} → {merged_clip['candidate_id']} "
            f"({total:.1f}s, {len(all_segs)} segments)"
        )

    if not merged_new:
        return candidates

    final: list[dict] = [c for c in candidates if c.get("candidate_id") not in consumed]
    final.extend(merged_new)
    # Re-sort chronologically.
    final.sort(key=lambda c: float((c.get("segments") or [{"start": 0}])[0].get("start", 0.0)))
    logger.info(
        f"Stitch pass complete: {len(candidates)} candidates → {len(final)} "
        f"(merged {len(consumed)} into {len(merged_new)})"
    )
    return final


def _candidate_pack_for_prompt(
    candidates: list[dict],
    include_context: bool,
    transcript_segments: list | None = None,
) -> str:
    blocks = []
    for idx, cand in enumerate(candidates, 1):
        segments_text = ", ".join(
            f"{float(seg.get('start', 0.0)):.2f}-{float(seg.get('end', 0.0)):.2f}s"
            for seg in cand.get("segments", []) or []
        )
        block = [
            f"[CANDIDATE {idx}]",
            f"id: {cand.get('candidate_id', f'candidate_{idx:02d}')}",
            f"title: {cand.get('title', '') or cand.get('working_title', '')}",
            f"takeaway (MEANING LOCK — keep exactly this idea): {cand.get('takeaway', '') or '(none — infer the single idea from the evidence)'}",
            f"segments: {segments_text}",
            f"scores: hook={cand.get('hook_score', '')}, flow={cand.get('flow_score', '')}, viral={cand.get('virality_score', '')}, meaning={cand.get('meaning_score', '')}, master={cand.get('master_score', '')}",
            f"reason: {cand.get('reason', '')}",
            f"hook_text: {cand.get('hook_text', '')}",
            f"hook_phrase: {cand.get('hook_phrase', '')}",
            f"caption: {cand.get('caption', '')}",
            f"description: {cand.get('description', '')}",
            f"hashtags: {' '.join(cand.get('hashtags', []) if isinstance(cand.get('hashtags'), list) else str(cand.get('hashtags', '')).split())}",
            f"transcript_evidence: {cand.get('transcript_evidence', '')[:1200]}",
        ]
        if include_context and transcript_segments:
            block.append("nearby_context:")
            block.append(_candidate_nearby_context(cand, transcript_segments))
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _candidate_nearby_context(
    candidate: dict,
    transcript_segments: list,
    pad_pre_seconds: float = 15.0,
    pad_post_seconds: float = 40.0,
) -> str:
    """Build the refinement-time "nearby_context" block for a candidate.

    Refinement only sees this context window (not the full transcript), so it
    must be wide enough that the model can move start_words/end_words to a
    cleaner sentence boundary or split a segment to cut newly-spotted filler.

    Slightly asymmetric: a small (~15s) lookback so the model can recover a
    hook that begins a line or two early, and a ~40s lookahead so it has room to
    EXTEND the end far enough for the SAME idea to land clearly (duration is a
    rough guide, not a cap). The window is bounded, not unlimited: the end must
    stay on the same idea and pay off — it must not drift into the next thought
    or topic just to reach a sentence terminator (that buries the payoff and
    ships a cliffhanger).

    Multi-segment candidates (e.g. a clip stitched from 3 time-spans) used to
    print the transcript window THREE times — once per sub-segment — and a
    given transcript line ended up labeled SELECTED in one print and CONTEXT
    in the others, badly confusing the refinement AI. This builds the window
    ONCE across overall_start → overall_end, and a transcript line is marked
    SELECTED if it falls within ANY sub-segment (with floating-point tolerance
    so programmatically-extended boundaries at 477.25s correctly include the
    transcript line at 477.245s).
    """
    BOUNDARY_TOLERANCE = 0.1  # 100ms slack for float-rounding drift

    clip_segments = candidate.get("segments", []) or []
    if not clip_segments or not transcript_segments:
        return ""

    sub_segments = []
    for cs in clip_segments:
        cs_start = float(cs.get("start", 0.0))
        cs_end = float(cs.get("end", cs_start))
        if cs_end > cs_start:
            sub_segments.append((cs_start, cs_end))
    if not sub_segments:
        return ""

    overall_start = min(s for s, _ in sub_segments)
    overall_end = max(e for _, e in sub_segments)
    context_start = max(0.0, overall_start - pad_pre_seconds)
    context_end = overall_end + pad_post_seconds

    lines = []
    for transcript_seg in transcript_segments:
        start = float(transcript_seg.get("start", 0.0))
        end = float(transcript_seg.get("end", start))
        if end < context_start or start > context_end:
            continue
        is_selected = any(
            start >= seg_s - BOUNDARY_TOLERANCE and end <= seg_e + BOUNDARY_TOLERANCE
            for seg_s, seg_e in sub_segments
        )
        marker = "SELECTED" if is_selected else "CONTEXT"
        text = re.sub(r"\s+", " ", str(transcript_seg.get("text", "")).strip())
        if text:
            lines.append(f"[{marker} {start:.2f}s ({_fmt_time(start)}) -> {end:.2f}s ({_fmt_time(end)})] {text}")
    # Caps: line count + word budget. Both lifted so the model sees enough
    # surrounding context to refine boundaries and detect intra-topic filler.
    CONTEXT_LINE_CAP = int(getattr(config, "REFINE_CONTEXT_LINE_CAP", 100))
    CONTEXT_WORD_BUDGET = int(getattr(config, "REFINE_CONTEXT_WORD_BUDGET", 2000))
    trimmed: list[str] = []
    word_count = 0
    for line in lines[:CONTEXT_LINE_CAP]:
        line_words = len(line.split())
        if word_count + line_words > CONTEXT_WORD_BUDGET:
            break
        trimmed.append(line)
        word_count += line_words
    return "\n".join(trimmed)





def _parse_frame_string(frame_text) -> tuple[float, float] | None:
    """Parse '[MM:SS.ss -> MM:SS.ss]' (or unbracketed) to (start_sec, end_sec)."""
    if not frame_text:
        return None
    cleaned = re.sub(r"[\[\]]", "", str(frame_text)).strip()
    if "->" not in cleaned:
        return None
    a, b = cleaned.split("->", 1)
    try:
        return _parse_time_to_seconds(a.strip()), _parse_time_to_seconds(b.strip())
    except (ValueError, TypeError):
        return None


def _validate_analysis(analysis: dict, expected_count: int, logger: logging.Logger) -> dict:
    """Validate and clean up the analysis data.

    Tolerant of BOTH segment schemas:
      • NEW (preferred): {start_words, start_frame, end_words, end_frame}
        — anchor fields preserved verbatim; provisional numeric start/end
        synthesized from frame hints so downstream scoring keeps working.
        Final word-exact times are pinned later by local_clips_generator.
      • LEGACY: {start: "MM:SS.ss", end: "MM:SS.ss"} — parsed in place.
    """
    if "summary" not in analysis:
        analysis["summary"] = ""
    if "chapters" not in analysis:
        analysis["chapters"] = []
    if "candidates" not in analysis:
        for alias in ("clip_candidates", "clips", "selected_clips"):
            if isinstance(analysis.get(alias), list):
                analysis["candidates"] = analysis[alias]
                break
        else:
            analysis["candidates"] = []

    valid_candidates = []
    for i, cand in enumerate(analysis["candidates"]):
        try:
            cand.setdefault("rank", i + 1)
            # title aliases: working_title (discovery) or title (legacy/refinement)
            title_src = cand.get("title") or cand.get("working_title") or f"Clip {i + 1}"
            cand["title"] = str(title_src).strip()[:120]
            cand.setdefault("working_title", cand["title"])
            cand.setdefault("hook_score", 5)
            cand.setdefault("flow_score", 5)
            cand.setdefault("virality_score", 5)
            cand.setdefault("meaning_score", 5)
            cand.setdefault("completeness_score", cand.get("flow_score", 5))
            cand.setdefault("boundary_score", cand.get("flow_score", 5))
            cand["hook_score"] = max(0, min(10, _score_float(cand.get("hook_score"), 5)))
            cand["flow_score"] = max(0, min(10, _score_float(cand.get("flow_score"), 5)))
            cand["virality_score"] = max(0, min(10, _score_float(cand.get("virality_score"), 5)))
            cand["meaning_score"] = max(0, min(10, _score_float(cand.get("meaning_score"), 5)))
            cand["completeness_score"] = max(0, min(10, _score_float(cand.get("completeness_score"), 5)))
            cand["boundary_score"] = max(0, min(10, _score_float(cand.get("boundary_score"), 5)))
            cand["reason"] = str(cand.get("reason") or "").strip()[:280]
            cand["takeaway"] = str(cand.get("takeaway") or "").strip()[:280]
            # Refinement fields — caps aligned to the refinement prompt's asks:
            #   hook_phrase    : prompt asks 5-12 words. 12 words ≈ 90 chars; cap at 140 as safety net.
            #   youtube_title  : prompt asks ≤78 chars (+ optional emoji + hashtag). Cap at 110 backstop.
            #   description    : prompt asks 2-4 sentences (≈ 300-500 chars). Cap at 700 backstop.
            #   tags           : prompt asks ~500 chars. Cap at 600 backstop.
            #   hashtags string: 30-40 hashtags, ~25 chars each, ≈ 1000 chars. Cap at 1500 backstop.
            cand["hook_phrase"] = str(cand.get("hook_phrase") or "").strip()[:140]
            cand["youtube_title"] = str(cand.get("youtube_title") or "").strip()[:110]
            cand["description_text"] = str(cand.get("description_text") or cand.get("description") or "").strip()[:700]
            # description_hashtags is a single space-separated string per the new prompt.
            dh = cand.get("description_hashtags")
            if isinstance(dh, list):
                dh = " ".join(str(x).strip() for x in dh if str(x).strip())
            cand["description_hashtags"] = str(dh or "").strip()[:1500]
            cand["youtube_tags"] = str(cand.get("youtube_tags") or "").strip()[:600]
            # Legacy/compat fields kept (downstream still reads some of them).
            cand["hook_text"] = str(cand.get("hook_text") or "").strip()[:220]
            cand["caption"] = str(cand.get("caption") or "").strip()[:180]
            cand["description"] = cand["description_text"]
            hashtags = cand.get("hashtags") or []
            if not isinstance(hashtags, list):
                hashtags = str(hashtags).split()
            # Backfill legacy `hashtags` list from `description_hashtags` string
            # so downstream code (clip_selector, captioner) keeps working when
            # refinement only emits the string form.
            if not hashtags and cand.get("description_hashtags"):
                hashtags = str(cand["description_hashtags"]).split()
            cand["hashtags"] = _dedupe_preserve_order([
                _clean_hashtag(tag) for tag in hashtags if _clean_hashtag(tag)
            ])[:40]

            # ── Segments ────────────────────────────────────────────────────
            raw_segments = cand.get("segments") or []
            clean_segments: list[dict] = []
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                # Preserve any anchor fields the local matcher will use.
                preserved = {
                    k: seg[k] for k in ("start_words", "start_frame", "end_words", "end_frame")
                    if k in seg
                }
                # Resolve provisional numeric times.
                seg_start = None
                seg_end = None
                if "start" in seg or "end" in seg:
                    try:
                        seg_start = _parse_time_to_seconds(seg.get("start"))
                        seg_end = _parse_time_to_seconds(seg.get("end"))
                    except (ValueError, TypeError):
                        seg_start = seg_end = None
                if (seg_start is None or seg_end is None or seg_end <= seg_start) and preserved:
                    sf = _parse_frame_string(preserved.get("start_frame"))
                    ef = _parse_frame_string(preserved.get("end_frame"))
                    if sf:
                        seg_start = sf[0]
                    if ef:
                        seg_end = ef[1]
                    elif sf:
                        seg_end = sf[1]
                if seg_start is None or seg_end is None or seg_end <= seg_start:
                    continue
                out = {
                    "start": float(seg_start),
                    "end": float(seg_end),
                    "duration": float(seg_end) - float(seg_start),
                }
                out.update(preserved)
                clean_segments.append(out)

            # Fallback: candidate-level start/end_time if no usable segments.
            if not clean_segments:
                st = cand.get("start_time")
                et = cand.get("end_time")
                if st is not None and et is not None:
                    try:
                        s = _parse_time_to_seconds(st)
                        e = _parse_time_to_seconds(et)
                        if e > s:
                            clean_segments = [{"start": s, "end": e}]
                    except (ValueError, TypeError):
                        pass

            if not clean_segments:
                logger.warning(f"Skipping candidate {i+1}: no usable segments/anchors")
                continue

            cand["segments"] = clean_segments
            cand["start_time"] = clean_segments[0]["start"]
            cand["end_time"] = clean_segments[-1]["end"]
            valid_candidates.append(cand)

        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping candidate {i+1}: {e}")

    analysis["candidates"] = valid_candidates[:expected_count]
    logger.debug(f"Validated {len(valid_candidates)} candidates")
    return analysis


# test_api_key now lives in pipeline/api_provider.py and is imported at the top
# of this module (re-exported so app.py's `from pipeline.analyzer import
# test_api_key` keeps working).