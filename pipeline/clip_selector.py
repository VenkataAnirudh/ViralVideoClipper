"""
Clip Selector
==============
Snaps boundaries to sentences, enforces duration, removes overlaps.
"""

import json
import os
import logging
import re
import config


# ---------------------------------------------------------------------------
# Words that, when they appear as the FIRST word of a clip, signal that the
# cut landed mid-sentence.  The snap functions walk backward until they find
# a segment whose text starts with something that is NOT one of these.
# ---------------------------------------------------------------------------
_CONTINUATION_STARTERS = frozenset({
    # coordinating conjunctions
    "and", "but", "or", "so", "yet", "nor", "for",
    # subordinating / relative
    "because", "since", "although", "though", "even", "while", "whereas",
    "if", "unless", "until", "when", "whenever", "where", "wherever",
    "which", "that", "who", "whom", "whose", "after", "before",
    # adverbial connectors
    "however", "therefore", "thus", "hence", "consequently",
    "moreover", "furthermore", "additionally", "also", "besides",
    "meanwhile", "nevertheless", "nonetheless", "otherwise", "instead",
    "then", "next", "finally", "lastly",
    # enumerative
    "namely", "specifically", "particularly", "especially", "including",
    "such", "like", "as", "with", "without", "by", "from",
    "of", "in", "on", "at", "to",
    # discourse markers
    "plus", "except", "rather", "indeed", "certainly", "actually",
    "basically", "essentially", "ultimately",
    # ── NEW: spoken-language weak openers ──────────────────────
    "yeah", "yes", "hey", "well", "now", "so", "right", "okay", "ok",
    "um", "uh", "hmm", "ah", "oh", "look", "listen", "alright",
    "anyway", "you", "know", "i", "mean", "just", "literally",
})

# Maximum number of segments to walk backward when searching for a clean
# sentence start.  Keeps runtime bounded on very long transcripts.
_MAX_SENTENCE_WALK = 8

# Sentence-ending punctuation characters.
_SENTENCE_END_CHARS = frozenset(".!?")

# When extending an end snap for a cleaner sentence boundary, do not extend
# by more than this many seconds past the originally requested end time.
_MAX_END_EXTENSION_S = 6.0
_MAX_START_EXTENSION_S = 8.0

_FILLER_STARTERS = frozenset({
    "um", "uh", "hmm", "ah", "oh",
    "anyway", "basically", "right", "okay", "ok",
    "well", "like", "you", "know",
})

_INTRO_PHRASES = (
    "welcome back",
    "welcome to",
    "in this video",
    "today we're going",
    "today we are going",
    "like and subscribe",
    "subscribe",
    "hit the bell",
    "sponsor",
)


import sys
import os
from pathlib import Path

# Add pipeline directory to path so we can import local_clips_generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_clips_generator import map_exact_boundaries



def select_clips(job_dir, transcript, analysis, settings, logger):
    path = os.path.join(job_dir, "clips_plan.json")
    if os.path.exists(path):
        logger.info("Found existing clips_plan.json. Loading it directly to respect custom clips.")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing clips_plan.json: {e}. Re-running selection.")

    candidates = analysis.get("candidates", [])
    max_clips = settings.get("clip_count", config.DEFAULT_CLIP_COUNT)

    logger.info(f"Refining {len(candidates)} candidates (target: {max_clips} clips)")
    
    # NEW LOGIC: Use exact word-level boundaries from the AI's content field
    logger.info("Mapping exact word boundaries using local_clips_generator...")
    mapped_candidates = map_exact_boundaries(candidates, transcript, logger)
    transcript_segments = transcript.get("segments", [])

    # Clip-level stitch pass — local pre-filter finds under-length clips, then
    # ONE AI call decides which to merge with neighbours so each viable clip
    # meets min_duration. Skipped silently if no short clips exist or if the
    # AI providers aren't available.
    min_duration = int(settings.get("min_duration", config.DEFAULT_MIN_DURATION))
    max_duration = int(settings.get("max_duration", config.DEFAULT_MAX_DURATION))
    # `skip_duration_floor` (default False) — run the AI stitch pass so any
    # under-length clip that survived discovery+refinement gets merged with a
    # neighbour into a viable-length single clip. Defaulting True caused the
    # stitch pass to be dead code in production: short fragments were rendered
    # as isolated jump-cut clips with no safety net. Set True only when you
    # explicitly want a permissive run that keeps every under-length fragment.
    skip_duration_floor = bool(settings.get("skip_duration_floor", False))
    analysis_mode = str(settings.get("analysis_mode",
                        getattr(config, "DEFAULT_ANALYSIS_MODE", "manual"))).strip().lower()
    if analysis_mode == "manual":
        # External-LLM route → standalone 2-pass stitcher (identification +
        # metadata fusion, OpenAI-only; pipeline/external_stitcher.py). Gated
        # by skip_stitch_pass alone — the runner clears that flag only when an
        # OpenAI key exists, so no key still means zero provider calls. Runs
        # even under soft duration limits: external LLMs routinely emit
        # 12-27s fragments, and Pass 1 may still "skip" any short clip that
        # stands alone well — soft limits are honored at the AI level, not by
        # silently disabling the pass.
        if not settings.get("skip_stitch_pass", False):
            try:
                from pipeline.external_stitcher import run_external_2pass_stitch
                mapped_candidates = run_external_2pass_stitch(
                    job_dir=job_dir,
                    candidates=mapped_candidates,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    settings=settings,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(f"External stitch pass skipped due to error: {e}")
        else:
            logger.info("External route: stitch pass off (no OpenAI key — fully API-free run).")
    elif not settings.get("skip_stitch_pass", False) and not skip_duration_floor:
        try:
            from pipeline.analyzer import run_stitch_pass
            from pipeline.api_provider import _analysis_provider_plan, _normalize_provider
            provider = _normalize_provider(settings.get("ai_provider", config.DEFAULT_AI_PROVIDER))
            stitch_providers = _analysis_provider_plan(
                provider,
                settings.get("ai_model", ""),
                settings.get("api_key", ""),
                task_type="complex",
            )
            if stitch_providers:
                mapped_candidates = run_stitch_pass(
                    job_dir=job_dir,
                    candidates=mapped_candidates,
                    transcript_segments=transcript_segments,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    providers_to_try=stitch_providers,
                    logger=logger,
                    settings=settings,
                    enable_reasoning=False,
                )
        except Exception as e:
            logger.warning(f"Stitch pass skipped due to error: {e}")
    elif skip_duration_floor:
        logger.info("Skipping clip-level stitch pass (skip_duration_floor=True; soft duration limits).")

    plans = []
    for i, cand in enumerate(mapped_candidates):
        try:
            plan = _build_plan(cand, transcript_segments, i, logger)
            if plan:
                plans.append(plan)
        except Exception as e:
            logger.warning(f"Skipping candidate {i+1}: {e}")

    # `allow_overlaps` (default False) drops near-duplicate clips via the
    # span-based _remove_overlaps (CLIP_OVERLAP_DEDUP_RATIO of the shorter clip).
    # The ratio is lenient enough (0.6) that two clips making DIFFERENT points
    # from the same region still survive — only heavy overlaps are collapsed.
    # Set allow_overlaps=True to keep every overlapping/subset clip.
    allow_overlaps = bool(settings.get("allow_overlaps", False))
    if settings.get("skip_duplication", False) or allow_overlaps:
        logger.info(
            "Skipping overlap removal (allow_overlaps=%s, skip_duplication=%s).",
            allow_overlaps, settings.get("skip_duplication", False),
        )
    else:
        plans = _remove_overlaps(plans, logger)
        
    plans.sort(key=lambda p: p.get("master_score", 0.0), reverse=True)

    if not plans:
        raise RuntimeError("AI analysis produced no usable clip plans after boundary validation.")

    validation_pool_size = max(max_clips + 5, int(max_clips * 1.5))
    validation_pool = plans[:validation_pool_size]

    for idx, plan in enumerate(validation_pool):
        plan["clip_index"] = idx + 1
        plan["clip_name"] = f"clip_{idx + 1:02d}"

    import hashlib as _hashlib
    # Fold min/max duration into the fingerprint so re-running with different
    # duration settings does not reuse a validation result computed under the
    # old durations.
    _min_d = settings.get("min_duration", config.DEFAULT_MIN_DURATION)
    _max_d = settings.get("max_duration", config.DEFAULT_MAX_DURATION)
    _pool_fingerprint = _hashlib.md5(
        (f"d{_min_d}-{_max_d}|" + "|".join(
            f"{p.get('clip_name','')}:{p.get('segments', [{}])[0].get('start', 0):.2f}"
            f"-{p.get('segments', [{}])[-1].get('end', 0):.2f}"
            for p in validation_pool
        )).encode()
    ).hexdigest()[:12]
    validation_cache_key = f"validate_clips_{validation_pool_size}_{_pool_fingerprint}"

    validated = validate_clips_with_ai(
        validation_pool, settings, logger,
        job_dir=job_dir,
        transcript_segments=transcript_segments,
        cache_key_override=validation_cache_key,
    )

    final_plans = validated[:max_clips]
    if len(final_plans) < max_clips:
        already_used = {id(p) for p in final_plans}
        for p in validated[max_clips:]:
            if id(p) not in already_used:
                final_plans.append(p)
            if len(final_plans) >= max_clips:
                break

    if len(final_plans) < max_clips:
        logger.warning(
            f"AI produced only {len(final_plans)} usable clip(s) after validation "
            f"(target: {max_clips}). No local fallback clips will be generated."
        )

    plans = final_plans

    for idx, plan in enumerate(plans):
        plan["clip_index"] = idx + 1
        plan["clip_name"] = f"clip_{idx + 1:02d}"

    path = os.path.join(job_dir, "clips_plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)

    _save_clip_info_txt(job_dir, plans, logger)

    logger.info(f"Final clip plan: {len(plans)} clips saved")
    return plans


def _save_clip_info_txt(job_dir, plans, logger):
    """Save a human-readable clip_info.txt with timestamps, durations, and titles."""
    info_path = os.path.join(job_dir, "clip_info.txt")
    lines = [
        "=" * 60,
        "  CLIP INFO — Timestamps & Durations",
        "=" * 60,
        "",
    ]
    for plan in plans:
        lines.append(f"  {plan['clip_name']}  |  {plan.get('title', 'Untitled')}")
        lines.append(f"  Duration: {plan['total_duration']:.1f}s  |  Segments: {len(plan['segments'])}")
        for j, seg in enumerate(plan["segments"]):
            s_start = seg["start"]
            s_end   = seg["end"]
            s_dur   = seg.get("duration", s_end - s_start)
            lines.append(f"    Segment {j+1}: {s_start:.3f}s → {s_end:.3f}s  ({s_dur:.1f}s)")
        lines.append(f"  Hook: {plan.get('hook_text', 'N/A')[:80]}")
        lines.append(f"  Hook Phrase: {plan.get('hook_phrase', 'N/A')[:80]}")
        lines.append(f"  Scores: hook={plan.get('hook_score',0)} flow={plan.get('flow_score',0)} viral={plan.get('virality_score',0)}")
        lines.append("-" * 60)
    lines.append("")
    lines.append(f"Total clips: {len(plans)}")
    lines.append(f"Total duration: {sum(p['total_duration'] for p in plans):.1f}s")

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.debug(f"Clip info saved: {info_path}")



def _fix_segment_overlaps(segments):
    """
    Ensure stitched segments do not overlap each other internally.
    If they overlap, merge them into a single continuous segment.
    """
    if not segments:
        return []
    
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    fixed = [sorted_segs[0]]
    
    for current in sorted_segs[1:]:
        prev = fixed[-1]
        if current["start"] < prev["end"]:
            if current["end"] <= prev["end"]:
                continue
            else:
                prev["end"] = current["end"]
                prev["duration"] = round(prev["end"] - prev["start"], 3)
        else:
            fixed.append(current)
            
    return fixed

def _build_plan(
candidate, segments, index, logger):
    """Construct the final plan dictionary from the exactly mapped candidate."""
    clip_segs = candidate.get("segments", [])
    if not clip_segs:
        return None

    # Internal overlaps fix (e.g. if the AI accidentally stitched two overlapping segments)
    snapped = _fix_segment_overlaps(clip_segs)
    if not snapped:
        return None
    # Normalize: every segment carries a "duration" (unresolved/kept-as-is
    # candidates come straight from analysis and lack it — the source of the
    # 'duration' KeyError that broke the validate/export stage).
    for s in snapped:
        try:
            s["duration"] = round(float(s.get("duration", s["end"] - s["start"])), 3)
        except (KeyError, TypeError, ValueError):
            s["duration"] = 0.0

    total = sum(x.get("duration", x["end"] - x["start"]) for x in snapped)
    clip_text = _get_text(snapped, segments)

    # ── New-schema refinement metadata (with legacy fallbacks) ──────────────
    youtube_title = candidate.get("youtube_title") or candidate.get("working_title") or candidate.get("title", f"Clip {index + 1}")
    description_text = (
        candidate.get("description_text")
        or candidate.get("youtube_description")
        or candidate.get("description", "")
    )
    takeaway = candidate.get("takeaway", "")
    youtube_tags = candidate.get("youtube_tags", "")

    # description_hashtags is a space-separated string of (ideally) 30 tags in
    # the new schema. Accept a list too. Fall back to legacy hashtag fields.
    dh = candidate.get("description_hashtags")
    if isinstance(dh, str):
        hashtags = [t for t in dh.split() if t.strip()]
    elif isinstance(dh, list):
        hashtags = [str(t).strip() for t in dh if str(t).strip()]
    else:
        hashtags = []
    if not hashtags:
        legacy = candidate.get("youtube_hashtags") or candidate.get("hashtags") or []
        hashtags = legacy if isinstance(legacy, list) else str(legacy).split()

    hook_phrase = candidate.get("hook_phrase", "")
    # social_caption: prefer takeaway, then legacy caption
    social_caption = takeaway or candidate.get("caption", "")

    return {
        "clip_index": index + 1,
        "clip_name": f"clip_{index + 1:02d}",
        "candidate_id": candidate.get("candidate_id", ""),
        "title": youtube_title,
        "segments": snapped,
        "total_duration": round(total, 3),
        "is_concat": len(snapped) > 1,
        "hook_score": float(candidate.get("hook_score", 5)),
        "flow_score": float(candidate.get("flow_score", 5)),
        "virality_score": float(candidate.get("virality_score", 5)),
        "meaning_score": float(candidate.get("meaning_score", 5)),
        "master_score": float(candidate.get("master_score", 0.0)),
        "reason": candidate.get("reason", ""),
        "takeaway": takeaway,
        "hook_text": candidate.get("hook_text", ""),
        "hook_phrase": hook_phrase,
        "caption": candidate.get("caption", ""),
        "description": description_text,
        # New canonical metadata fields (refinement-owned):
        "description_text": description_text,
        "description_hashtags": candidate.get("description_hashtags", ""),
        "youtube_title": youtube_title,
        "youtube_description": description_text,
        "youtube_tags": youtube_tags,
        "youtube_hashtags": hashtags,
        "hashtags": hashtags,
        # Defaults so results.json is populated even when copywriter is skipped
        # (refinement owns metadata; copywriter is fallback-only now):
        "social_title": youtube_title,
        "social_description": description_text,
        "social_caption": social_caption,
        "completeness_score": float(candidate.get("completeness_score", 0)),
        "boundary_score": float(candidate.get("boundary_score", 0)),
        "transcript_text": clip_text,
    }



def _remove_overlaps(plans, logger):
    """Drop near-duplicate clips by TOTAL span overlap (not per-segment).

    Two clips are duplicates when the seconds they share — summed across all
    segment pairs — exceed CLIP_OVERLAP_DEDUP_RATIO of the SHORTER clip's total
    duration. Highest master_score wins. Span-based catches partial overlaps
    (e.g. three clips drawn from the same passage) that a per-segment 0.85 check
    let through.
    """
    ratio_thr = getattr(config, "CLIP_OVERLAP_DEDUP_RATIO", 0.6)

    def _dur(p):
        return sum(s.get("duration", s["end"] - s["start"]) for s in p["segments"]) or 1.0

    plans.sort(key=lambda p: p.get("master_score", 0.0), reverse=True)
    kept = []
    for plan in plans:
        p_dur = _dur(plan)
        drop = False
        for ex in kept:
            shared = 0.0
            for sa in plan["segments"]:
                for sb in ex["segments"]:
                    o = min(sa["end"], sb["end"]) - max(sa["start"], sb["start"])
                    if o > 0:
                        shared += o
            shorter = min(p_dur, _dur(ex))
            if shorter > 0 and shared / shorter > ratio_thr:
                drop = True
                break
        if not drop:
            kept.append(plan)
    removed = len(plans) - len(kept)
    if removed:
        logger.info(f"Removed {removed} near-duplicate clip(s) (>{ratio_thr:.0%} span overlap)")
    return kept


def _get_text(clip_segments, transcript_segments):
    """Return the WORD-EXACT text inside the clip's segments.

    Previously this concatenated the FULL text of every transcript segment that
    merely overlapped the clip, so transcript_text bled words from before the
    start / after the end of the actual cut. That padding made QA over-report
    bad boundaries (a clean word-exact clip looked like it started/ended on
    filler). Now it keeps only the words whose midpoint falls inside each clip
    segment's [start, end]; segments without word timing fall back to full text.
    """
    TOL = 0.05
    texts = []
    for cs in clip_segments:
        cs_start = float(cs.get("start", 0.0))
        cs_end = float(cs.get("end", cs_start))
        words = []
        for ts in transcript_segments:
            if float(ts.get("end", 0.0)) <= cs_start or float(ts.get("start", 0.0)) >= cs_end:
                continue
            ts_words = ts.get("words") or []
            if ts_words:
                for w in ts_words:
                    try:
                        ws = float(w.get("start", 0.0))
                        we = float(w.get("end", ws))
                    except (TypeError, ValueError):
                        continue
                    if cs_start - TOL <= (ws + we) / 2.0 <= cs_end + TOL:
                        tok = str(w.get("word", "")).strip()
                        if tok:
                            words.append(tok)
            else:
                tok = str(ts.get("text", "")).strip()
                if tok:
                    words.append(tok)
        if words:
            texts.append(" ".join(words))
    return " ".join(texts).strip()


def validate_clips_with_ai(plans, settings, logger, job_dir=None,
                            transcript_segments=None, cache_key_override=None):
    """
    Performs a distinct post-snap AI review of refined clip text and boundaries
    using a unified LLM prompt.
    """
    from pipeline.analyzer import call_llm, get_cached_api_response, save_cached_api_response, write_fallback_status

    if settings.get("skip_validation", False):
        logger.info("Skipping AI validation pass based on settings.")
        if job_dir:
            write_fallback_status(job_dir, "VALIDATE CLIPS", "SKIPPED", "AI validation pass skipped based on configuration.")
        return plans

    if not plans:
        return plans

    # Make transcript segments available to the adjustment block
    _transcript_segments_ref = transcript_segments or []

    logger.info(f"Running sequential AI validation for {len(plans)} clip candidate(s)...")

    _cache_key = cache_key_override or "validate_clips"

    # 1. Check Cache (skip if extraction-phase cache is disabled)
    cached_result = None
    if job_dir and not getattr(config, "SKIP_API_CACHE_DURING_EXTRACTION", True):
        try:
            cached_result = get_cached_api_response(job_dir, _cache_key, logger)
        except Exception as e:
            logger.warning(f"Error checking validation cache: {e}")

    # Cache invalidation check: verify it covers all current plans
    if cached_result and isinstance(cached_result, dict):
        cached_clip_names = {v.get("clip_name") for v in cached_result.get("validation", [])}
        current_clip_names = {p["clip_name"] for p in plans}
        if not current_clip_names.issubset(cached_clip_names):
            logger.info("Cached validation does not cover current pool — re-running validation.")
            cached_result = None

    result = None
    if cached_result and isinstance(cached_result, dict):
        result = cached_result
        if job_dir:
            write_fallback_status(job_dir, "VALIDATE CLIPS", "SUCCESS", f"Loaded clip validation from API cache ({_cache_key}).")
    else:
        # Build the clips representation for the prompt
        clips_info = []
        for plan in plans:
            clips_info.append({
                "clip_name": plan["clip_name"],
                "title": plan["title"],
                "duration": plan["total_duration"],
                "transcript_text": plan["transcript_text"],
                "segments": plan["segments"]
            })

        prompt = f"""You are an expert video editor and social media viral content analyst.
We have segmented a video transcript into refined clip candidates. However, automated boundary snapping can sometimes cut off mid-thought, mid-sentence, or miss vital opening context, or result in incoherent or generic content.

Your job is to perform a sequential AI verification/validation on these clip candidates.

Here are the clip candidates:
{json.dumps(clips_info, indent=2, ensure_ascii=False)}

For each clip, you must make one of three decisions:
1. "keep" - The clip boundary is perfect, it starts and ends cleanly, represents a cohesive thought, and has good hooks/context.
2. "adjust" - The boundary cuts off slightly too early/late or starts mid-sentence. You can suggest a time adjustment to shift the start or end time.
   - You must format your adjustment EXACTLY as one of these: "start -X.Y", "start +X.Y", "end -X.Y", "end +X.Y" (where X.Y is the offset in seconds, e.g. "start -2.0" to start 2 seconds earlier, or "end +1.5" to end 1.5 seconds later).
   - Only make adjustments of up to +/- 5.0 seconds. Keep them precise.
3. "discard" - ONLY discard the clip if it does not have a proper hook score AND lacks meaningful/informative content together. If it has a decent hook OR represents a meaningful snippet, it MUST be retained. Be extremely conservative with discards; retain clips as much as possible.

You MUST respond with a valid JSON object matching the following JSON schema:
{{
  "validation": [
    {{
      "clip_name": "clip_01",
      "decision": "keep",
      "suggested_adjustment": "",
      "reasoning": "Starts with a clear hooks and ends with a complete thought."
    }},
    {{
      "clip_name": "clip_02",
      "decision": "adjust",
      "suggested_adjustment": "start -2.5",
      "reasoning": "Needs 2.5 seconds of preceding context to capture the full question being asked."
    }}
  ]
}}

Ensure that you return ONLY the JSON block. Do not include any conversational preamble or markdown code blocks outside of the JSON itself."""

        try:
            response_text = call_llm(prompt, settings, logger, temperature=getattr(config, "NVIDIA_TEMP_VALIDATION", 0.4))
            logger.debug(f"AI Validation raw response: {response_text}")

            # Clean response_text of markdown wrappers
            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            # Extract JSON if there is extra text
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)

            result = json.loads(clean_text)
            
            # Save to Cache (skip if extraction-phase cache is disabled)
            if job_dir and not getattr(config, "SKIP_API_CACHE_DURING_EXTRACTION", True):
                save_cached_api_response(job_dir, _cache_key, result, logger)
            if job_dir:
                write_fallback_status(job_dir, "VALIDATE CLIPS", "SUCCESS", f"Validated {len(plans)} clips successfully via AI ({_cache_key}).")
        except Exception as e:
            logger.error(f"Error during AI clip validation call: {e}. Falling back to keeping all plans.")
            if job_dir:
                write_fallback_status(
                    job_dir,
                    "VALIDATE CLIPS",
                    "WARNING / FALLBACK",
                    f"AI clip validation failed: {str(e)}. Kept all {len(plans)} plans by default."
                )
            import traceback
            logger.debug(traceback.format_exc())
            # Upstream filters (_remove_overlaps gated by allow_overlaps /
            # skip_duplication, and the soft duration floor gated by
            # skip_duration_floor) have already been applied to `plans`. So
            # returning plans here does not bypass anything — it just means
            # no AI-decided keep/adjust/discard is applied. This matches the
            # "soft preferences, allow overlaps" policy.
            return plans

    # Process result
    if not result:
        return plans

    try:
        validations = result.get("validation", [])
        # Map by clip_name
        val_map = {v["clip_name"]: v for v in validations}

        final_plans = []
        for plan in plans:
            cname = plan["clip_name"]
            val = val_map.get(cname)
            if not val:
                logger.info(f"Clip {cname}: No validation decision received, keeping by default.")
                final_plans.append(plan)
                continue

            decision = val.get("decision", "keep").lower()
            reasoning = val.get("reasoning", "")
            adjustment = val.get("suggested_adjustment", "").strip()

            if decision == "discard":
                hook_val = float(plan.get("hook_score", 0.0))
                meaning_val = float(plan.get("meaning_score", 0.0))
                if hook_val < 7.0 and meaning_val < 6.0:
                    logger.info(f"Discarding {cname} based on AI validation. Reasoning: {reasoning} (Hook: {hook_val}, Meaning: {meaning_val})")
                    continue
                else:
                    logger.info(f"Overriding discard decision for {cname} because hook_score ({hook_val:.1f}) >= 7.0 or meaning_score ({meaning_val:.1f}) >= 6.0. Reasoning: {reasoning}")
                    decision = "keep"

            if decision == "adjust" and adjustment:
                # Parse adjustment
                match = re.search(r'(start|end)\s*([+-]?\d*(?:\.\d+)?)', adjustment.lower())
                if match:
                     field = match.group(1)
                     try:
                         delta = float(match.group(2))
                         # Need transcript segments for re-snapping
                         # They are available in the outer scope via the `plans` list
                         segs = plan["segments"]
                         if field == "start" and segs:
                             old_start = segs[0]["start"]
                             raw_start = max(0.0, old_start + delta)
                             # Re-snap to nearest word boundary
                             try:
                                 new_start = _snap_to_closest_word(raw_start, _transcript_segments_ref, side="start")
                                 # Clamp: never snap so far back that clip shrinks below 0.5s
                                 if segs[0]["end"] - new_start < 0.5:
                                     new_start = raw_start
                             except Exception:
                                 new_start = raw_start
                             segs[0]["start"] = round(new_start, 3)
                             segs[0]["duration"] = round(segs[0]["end"] - segs[0]["start"], 3)
                             logger.info(
                                 f"Adjusted {cname} start: {old_start:.2f}s → {new_start:.2f}s "
                                 f"(AI delta: {delta:+.2f}s). Reason: {reasoning}"
                             )
                         elif field == "end" and segs:
                             old_end = segs[-1]["end"]
                             raw_end = max(segs[-1]["start"] + 0.1, old_end + delta)
                             # Re-snap to nearest word/sentence boundary
                             try:
                                 new_end = _snap_to_closest_word(raw_end, _transcript_segments_ref, side="end")
                                 if new_end - segs[-1]["start"] < 0.5:
                                     new_end = raw_end
                             except Exception:
                                     new_end = raw_end
                             segs[-1]["end"] = round(new_end, 3)
                             segs[-1]["duration"] = round(segs[-1]["end"] - segs[-1]["start"], 3)
                             logger.info(
                                 f"Adjusted {cname} end: {old_end:.2f}s → {new_end:.2f}s "
                                 f"(AI delta: {delta:+.2f}s). Reason: {reasoning}"
                             )

                         plan["total_duration"] = round(sum(s["duration"] for s in segs), 3)
                         plan["transcript_text"] = _get_text(plan["segments"], _transcript_segments_ref)
                     except ValueError:
                         logger.warning(f"Could not parse delta in adjustment '{adjustment}' for {cname}")
                else:
                     logger.warning(f"Invalid adjustment format '{adjustment}' for {cname}")

            # Re-check intra-clip overlaps after any AI boundary adjustment
            plan["segments"] = _fix_segment_overlaps(plan["segments"])
            if not plan["segments"]:
                logger.info(
                    f"Discarding {cname}: all segments became degenerate "
                    "after AI boundary adjustment."
                )
                continue
            plan["total_duration"] = round(
                sum(s["duration"] for s in plan["segments"]), 3
            )
            plan["is_concat"] = len(plan["segments"]) > 1

            final_plans.append(plan)

        return final_plans

    except Exception as e:
        logger.error(f"Error during parsing AI clip validation results: {e}. Skipping validation adjustments.")
        if job_dir:
            write_fallback_status(
                job_dir,
                "VALIDATE CLIPS",
                "WARNING / FALLBACK",
                f"Error parsing validation results: {str(e)}. Kept all original plans."
            )
        return plans


# ───────────────────────────────────────────────────────────────────────────
# New selection helpers
# ───────────────────────────────────────────────────────────────────────────

def _snap_to_closest_word(target: float, segments: list, side: str = "start") -> float:
    """
    Lightweight snap: find the single closest word boundary without
    any sentence-walking. Used for AI-suggested adjustment offsets so
    that the snap never drifts more than one word away from the intent.
    side='start' → snap to nearest word start
    side='end'   → snap to nearest word end
    """
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))
    if not all_words:
        return target
    key = "start" if side == "start" else "end"
    best = min(all_words, key=lambda w: abs(float(w[key]) - target))
    return float(best[key])


def _forward_trim_weak_start(snapped: list, segments: list, min_dur: float) -> list:
    """
    If the first word of the clip is a weak opener, walk forward word-by-word
    to find the first strong start, provided the resulting clip remains
    >= min_dur * 0.8.
    """
    if not snapped:
        return snapped
    first_seg = snapped[0]
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))
    if not all_words:
        return snapped

    # Find the first word at or after the clip start
    clip_start = float(first_seg["start"])
    start_idx = None
    for i, w in enumerate(all_words):
        if float(w["start"]) >= clip_start - 0.05:
            start_idx = i
            break

    if start_idx is None:
        return snapped

    total_dur = sum(s["duration"] for s in snapped)
    floor_dur = min_dur * 0.8

    idx = start_idx
    while idx < len(all_words) - 1:
        word_text = re.sub(r"^[^A-Za-z0-9]+", "", str(all_words[idx].get("word", ""))).lower()
        if word_text not in _CONTINUATION_STARTERS and word_text not in _FILLER_STARTERS:
            break
        candidate_start = float(all_words[idx + 1]["start"])
        new_dur = total_dur - (candidate_start - clip_start)
        if new_dur < floor_dur:
            break  # Would make clip too short
        clip_start = candidate_start
        idx += 1

    if clip_start != float(first_seg["start"]):
        snapped = [dict(s) for s in snapped]  # copy before mutating
        snapped[0]["start"] = round(clip_start, 3)
        snapped[0]["duration"] = round(snapped[0]["end"] - snapped[0]["start"], 3)
        snapped = [s for s in snapped if s["duration"] > 0.05]
    return snapped


_CLAUSE_END_CHARS = frozenset(".,;:—–-")

def _best_clause_end_before(start: float, target_end: float, transcript_segments: list) -> float:
    """
    Fallback: find the last clause-level punctuation boundary before target_end.
    Returns target_end if nothing is found (clean word boundary is better than mid-word).
    """
    best = None
    for seg in transcript_segments:
        words = seg.get("words", [])
        for w in words:
            w_end = float(w.get("end", 0.0))
            if w_end <= start:
                continue
            if w_end > target_end:
                return best if best is not None else target_end
            word_text = str(w.get("word", "")).strip()
            if word_text and word_text[-1] in _CLAUSE_END_CHARS:
                best = w_end
    return best if best is not None else target_end
