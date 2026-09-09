"""
Copywriter
==========
Generates per-clip social titles, short descriptions, captions, and hashtags in
one batch API call. Each clip block is grounded in that clip's own transcript
window so the model does not mix ideas across clips.
"""

import json
import os
import re
import config


GENERIC_TAGS = {
    "#viral", "#trending", "#mustwatch", "#clips", "#reels", "#shorts",
    "#fyp", "#foryou", "#foryoupage", "#explore", "#explorepage",
}


def generate_copy(job_dir, clips_plan, meta, transcript, settings, logger):
    """Generate social media copy for all clips in one batch API call."""
    from pipeline.analyzer import get_cached_api_response, save_cached_api_response, write_fallback_status

    provider = settings.get("ai_provider", config.DEFAULT_AI_PROVIDER)
    model = settings.get("ai_model", "")
    api_key = settings.get("api_key", "")

    logger.info(f"Generating social copy for {len(clips_plan)} clips using {provider}")

    # Check cache first (skip if extraction-phase cache is disabled)
    cached_data = None
    if not getattr(config, "SKIP_API_CACHE_DURING_EXTRACTION", True):
        cached_data = get_cached_api_response(job_dir, "copywriter_response", logger)
    if cached_data and isinstance(cached_data, dict):
        logger.info("Loaded copywriter response from cache.")
        copy_data = cached_data
        write_fallback_status(job_dir, "COPYWRITER", "SUCCESS", "Loaded social copy from API cache.")
    else:
        prompt = _build_copy_prompt(clips_plan, meta, transcript)
        try:
            from pipeline.analyzer import call_llm
            _creative_temp = getattr(config, "AI_TEMP_CREATIVE", 1.3)
            if provider == "claude":
                raw = _call_claude(prompt, model, api_key, logger, temperature=_creative_temp)
            elif provider == "gemini":
                raw = _call_gemini(prompt, model, api_key, logger, temperature=_creative_temp)
            else:
                # nvidia / openai / legacy providers route through the shared,
                # provider-aware analysis caller (handles fallbacks + NVIDIA).
                raw = call_llm(prompt, settings, logger, temperature=_creative_temp)

            logger.debug(f"Raw copywriter response (first 800 chars):\n{raw[:800]}")
            copy_data = _parse_copy_response(raw, clips_plan, logger)

            # Save to cache (skip if extraction-phase cache is disabled)
            if not getattr(config, "SKIP_API_CACHE_DURING_EXTRACTION", True):
                save_cached_api_response(job_dir, "copywriter_response", copy_data, logger)
            write_fallback_status(job_dir, "COPYWRITER", "SUCCESS", "Generated social copy successfully via AI.")
        except Exception as e:
            logger.error(f"Copy generation failed with primary provider {provider}: {e}")
            try:
                logger.info("Falling back to general call_llm for copy generation...")
                from pipeline.analyzer import call_llm
                raw = call_llm(prompt, settings, logger, temperature=getattr(config, "AI_TEMP_CREATIVE", 1.3))
                logger.debug(f"Raw copywriter response from fallback call_llm (first 800 chars):\n{raw[:800]}")
                copy_data = _parse_copy_response(raw, clips_plan, logger)

                # Save to cache (skip if extraction-phase cache is disabled)
                if not getattr(config, "SKIP_API_CACHE_DURING_EXTRACTION", True):
                    save_cached_api_response(job_dir, "copywriter_response", copy_data, logger)
                write_fallback_status(
                    job_dir, 
                    "COPYWRITER", 
                    "SUCCESS / FALLBACK", 
                    f"Generated social copy via fallback LLM. (Primary failed: {str(e)})"
                )
            except Exception as fallback_err:
                logger.error(f"Copy generation fallback call_llm failed: {fallback_err}")
                copy_data = _generate_fallback_copy(clips_plan, logger)
                write_fallback_status(
                    job_dir, 
                    "COPYWRITER", 
                    "WARNING / FALLBACK", 
                    f"AI copy generation failed (Primary: {str(e)}, Fallback: {str(fallback_err)}). Used local grounded generator."
                )

    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    normalized_copy = {}
    for clip in clips_plan:
        name = clip["clip_name"]
        caption_file = os.path.join(clips_dir, f"{name}_caption.txt")
        caption_data = _normalize_copy_item(copy_data.get(name, {}), clip)

        title_text = caption_data["title"]
        description = caption_data["description"]
        caption_text = caption_data["caption"]
        hashtags = caption_data["hashtags"]

        if _looks_generic(caption_text, hashtags, clip):
            logger.warning(f"Clip {name}: AI copy looked generic; using grounded fallback")
            caption_data = _fallback_item(clip)
            title_text = caption_data["title"]
            description = caption_data["description"]
            caption_text = caption_data["caption"]
            hashtags = caption_data["hashtags"]

        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(title_text)
            f.write("\n\n")
            f.write(description)
            f.write("\n\n")
            f.write(caption_text)
            f.write("\n\n")
            f.write(" ".join(hashtags))

        clip["social_title"] = title_text
        clip["social_description"] = description
        clip["social_caption"] = caption_text
        clip["hashtags"] = hashtags
        normalized_copy[name] = caption_data

    with open(os.path.join(job_dir, "social_copy.json"), "w", encoding="utf-8") as f:
        json.dump(normalized_copy, f, indent=2, ensure_ascii=False)
    logger.info("Social copy generation complete")
    return normalized_copy


def _hashtags_to_lower_string(value) -> str:
    """Normalise hashtags (list or space-separated string) into a single
    lowercase, space-separated string — every token guaranteed to start with #."""
    if isinstance(value, list):
        tokens = [str(v) for v in value]
    else:
        tokens = str(value or "").split()
    out = []
    seen = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        tok = "#" + tok.lstrip("#")
        tok = tok.lower()
        if tok == "#":
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return " ".join(out)


def write_youtube_packages(job_dir, clips_plan, logger):
    """Write a ready-to-copy-paste YouTube package file per clip.

    Used when boundary-refinement / external-LLM metadata is already present (the
    fallback copywriter pass is skipped in that case, so without this the clips
    folder would have no per-clip caption file). Pulls youtube_title /
    description_text / description_hashtags / youtube_tags straight off each clip
    and writes ``<clip>_caption.txt`` in the clips folder, clearly sectioned so
    the title, the description (+ all hashtags), and the tags can each be pasted
    straight into the matching YouTube field.

    Returns the number of package files written.
    """
    clips_dir = os.path.join(job_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    written = 0
    for clip in clips_plan or []:
        name = clip.get("clip_name")
        if not name:
            continue
        title = str(clip.get("youtube_title") or clip.get("title") or "").strip()
        description = str(clip.get("description_text") or clip.get("description") or "").strip()
        hashtags = _hashtags_to_lower_string(
            clip.get("description_hashtags") or clip.get("hashtags") or ""
        )
        tags = str(clip.get("youtube_tags") or "").strip()
        hook = str(clip.get("hook_phrase") or "").strip()

        lines = [
            "=== TITLE ===",
            title,
            "",
            "=== DESCRIPTION ===",
            description,
        ]
        if hashtags:
            lines += ["", hashtags]
        lines += ["", "=== TAGS (comma-separated) ===", tags]
        if hook:
            lines += ["", "=== TTS HOOK ===", hook]

        caption_file = os.path.join(clips_dir, f"{name}_caption.txt")
        try:
            with open(caption_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).rstrip() + "\n")
            written += 1
        except OSError as exc:
            logger.warning(f"Could not write YouTube package for {name}: {exc}")
            continue

        # Surface normalised values on the clip for packaging / UI consumers.
        clip["social_title"] = title
        clip["social_description"] = description
        clip["youtube_title"] = title
        clip["description_text"] = description
        clip["description_hashtags"] = hashtags
        clip["youtube_tags"] = tags

    logger.info(f"Wrote {written} ready-to-paste YouTube package file(s) to clips/")
    return written


def _build_copy_prompt(clips_plan, meta, transcript):
    video_title = meta.get("title", "Unknown Video")
    video_channel = meta.get("channel", "Unknown Channel")

    clip_blocks = []
    for clip in clips_plan:
        name = clip["clip_name"]
        title = clip.get("title", name)
        hook = clip.get("hook_text", "")[:160]
        reason = clip.get("reason", "")[:260]
        excerpt = _clip_transcript_excerpt(clip, transcript, limit=1600) or clip.get("transcript_text", "")[:1600]
        segments = ", ".join(
            f"{float(s.get('start', 0)):.1f}-{float(s.get('end', 0)):.1f}s"
            for s in clip.get("segments", [])
        )
        clip_blocks.append(
            f"[CLIP_START:{name}]\n"
            f"Title: {title}\n"
            f"Timestamp windows: {segments}\n"
            f"Hook text: {hook}\n"
            f"Why it was selected: {reason}\n"
            f"AI analysis caption draft: {clip.get('caption', '')}\n"
            f"AI analysis description draft: {clip.get('description', '')}\n"
            f"AI analysis hashtags draft: {' '.join(clip.get('hashtags', []) if isinstance(clip.get('hashtags'), list) else [])}\n"
            f"Exact transcript for this clip only:\n{excerpt}\n"
            f"[CLIP_END:{name}]"
        )

    key_list = ", ".join(f'"{c["clip_name"]}"' for c in clips_plan)
    all_clips = "\n\n".join(clip_blocks)

    return f"""You are an expert short-form social media copywriter.

SOURCE VIDEO
Title: "{video_title}"
Channel: {video_channel}

TASK
For each clip below, write a specific hook title, short description, social
caption, and 12-15 hashtags. Use only the transcript inside that clip block. Do
not borrow facts from another clip. Internally identify the main reveal,
emotion, and viewer payoff before writing, but do not show reasoning.
Every clip must receive all four outputs; never leave title, description,
caption, or hashtags empty.

CLIPS
{all_clips}

RULES
- title: under 78 characters, hooky but truthful, specific to the transcript.
- description: 2-3 useful sentences. Explain the clip's idea, stakes, and payoff.
- caption: under 180 characters, punchy, no filler, no "Check this out".
- hashtags: 12-15 total, all relevant; include #fyp once. Blend topic tags,
  niche tags, platform tags, and 2-3 tasteful viral discovery tags.
- Avoid generic-only hashtag sets such as #viral #fyp #trending.
- Ground every description in the selected transcript and explain why the
  snippet matters as a standalone reel.
- Do not invent people, claims, products, or numbers that are not in the clip.

OUTPUT
Return ONLY a valid JSON object with exactly these keys: {key_list}

Each value must use this schema:
{{
  "title": "<string under 78 chars>",
  "description": "<2-3 sentence string>",
  "caption": "<string under 180 chars>",
  "hashtags": ["#specific1", "#specific2", "#specific3", "...12 to 15 total", "#fyp"]
}}"""


def _clip_transcript_excerpt(clip, transcript, limit=900):
    pieces = []
    for seg in clip.get("segments", []):
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start))
        for tseg in transcript.get("segments", []):
            start = float(tseg.get("start", 0))
            end = float(tseg.get("end", start))
            if end <= seg_start or start >= seg_end:
                continue
            text = str(tseg.get("text", "")).strip()
            if text:
                pieces.append(text)
    text = " ".join(pieces)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _normalize_copy_item(item, clip):
    if not isinstance(item, dict):
        item = {}
    fallback = _fallback_item(clip)
    title = str(item.get("title") or fallback["title"]).strip()[:90]
    description = str(item.get("description") or fallback["description"]).strip()
    caption = str(item.get("caption") or fallback["caption"]).strip()[:180]
    hashtags = item.get("hashtags") or fallback["hashtags"]
    if not isinstance(hashtags, list):
        hashtags = str(hashtags).split()
    hashtags = [_clean_hashtag(tag) for tag in hashtags if _clean_hashtag(tag)]
    if "#fyp" not in {tag.lower() for tag in hashtags}:
        hashtags.append("#fyp")
    max_tags = max(12, int(getattr(config, "AI_COPY_HASHTAG_MAX", 15)))
    min_tags = max(10, min(max_tags, int(getattr(config, "AI_COPY_HASHTAG_MIN", 12))))
    hashtags = _dedupe_preserve_order(hashtags)[:max_tags]
    filler_pool = fallback["hashtags"] + [
        "#insight", "#learning", "#shorts", "#reels", "#fyp",
        "#creator", "#mindset", "#storytelling", "#viralclips", "#content",
    ]
    for tag in filler_pool:
        if len(hashtags) >= min_tags:
            break
        if tag not in hashtags:
            hashtags.append(tag)
    return {
        "title": title or fallback["title"],
        "description": description or fallback["description"],
        "caption": caption or fallback["caption"],
        "hashtags": hashtags[:max_tags],
    }


def _clean_hashtag(tag):
    tag = re.sub(r"[^A-Za-z0-9_#]", "", str(tag).strip())
    if not tag:
        return ""
    if not tag.startswith("#"):
        tag = "#" + tag
    return tag


def _dedupe_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _looks_generic(caption, hashtags, clip):
    if not caption or caption.strip().lower() == str(clip.get("title", "")).strip().lower():
        return True
    tags = {tag.lower() for tag in hashtags}
    return len(tags - GENERIC_TAGS) < 4


def _fallback_item(clip):
    ai_hashtags = clip.get("hashtags") if isinstance(clip.get("hashtags"), list) else []
    if clip.get("caption") and clip.get("description") and len(ai_hashtags) >= 10:
        return {
            "title": str(clip.get("title") or "Key Moment").strip()[:90],
            "description": str(clip.get("description")).strip(),
            "caption": str(clip.get("caption")).strip()[:180],
            "hashtags": ai_hashtags[: int(getattr(config, "AI_COPY_HASHTAG_MAX", 15))],
        }

    title = str(clip.get("title") or "Key Moment").strip()[:70]
    reason = str(clip.get("reason") or "").strip()
    transcript_text = str(clip.get("transcript_text") or "").strip()
    description_source = reason or transcript_text
    description = (
        description_source[:360]
        if description_source
        else f"This clip highlights a specific, standalone moment from {title}."
    )
    caption = title[:179] if title else "A sharp takeaway from this clip"
    keywords = re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", f"{title} {reason} {transcript_text}")
    stopwords = {
        "this", "that", "with", "from", "your", "about", "there", "their",
        "would", "could", "should", "because", "really", "people", "thing",
        "things", "video", "clip", "just", "like", "what", "when", "where",
    }
    topic_tags = []
    for word in keywords:
        if word.lower() in stopwords:
            continue
        tag = "#" + word[:24].lower()
        if tag not in topic_tags:
            topic_tags.append(tag)
        if len(topic_tags) == 9:
            break
    hashtags = topic_tags + [
        "#insight", "#learning", "#shorts", "#reels", "#fyp",
        "#viralclips", "#creator", "#storytelling",
    ]
    hashtags = _dedupe_preserve_order(hashtags)
    min_tags = max(10, int(getattr(config, "AI_COPY_HASHTAG_MIN", 12)))
    while len(hashtags) < min_tags:
        hashtags.insert(-1, "#content")
    return {
        "title": title,
        "description": description,
        "caption": caption,
        "hashtags": hashtags[: int(getattr(config, "AI_COPY_HASHTAG_MAX", 15))],
    }


def _call_claude(prompt, model, api_key, logger):
    import anthropic
    if not api_key:
        api_key = config.ANTHROPIC_API_KEY
    if not model:
        model = config.DEFAULT_CLAUDE_MODEL
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=config.AI_MAX_TOKENS_COPY,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_gemini(prompt, model, api_key, logger):
    from google import genai
    if not api_key:
        api_key = config.GOOGLE_API_KEY
    if not model:
        model = config.DEFAULT_GEMINI_MODEL
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=config.AI_MAX_TOKENS_COPY,
        ),
    )
    return response.text


def _parse_copy_response(raw, clips_plan, logger):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n\s*```", raw)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                data = None
        else:
            data = None
        if data is None:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None
    if isinstance(data, dict):
        return data
    logger.warning("Could not parse copy response; using fallback captions")
    return _generate_fallback_copy(clips_plan, logger)


def _generate_fallback_copy(clips_plan, logger):
    logger.info("Using fallback captions (AI response unparseable)")
    return {clip["clip_name"]: _fallback_item(clip) for clip in clips_plan}
