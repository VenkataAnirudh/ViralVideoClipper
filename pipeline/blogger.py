"""
Blog Post Generator
====================
Generates structured blog posts from video transcript, AI analysis, and clip data.
Outputs both Markdown (.md) and plain text (.txt) formats.
"""

import json
import os
import re
import logging
import config


def generate_blog_post(
    job_dir: str,
    meta: dict,
    transcript: dict,
    analysis: dict,
    clips_plan: list,
    settings: dict,
    logger: logging.Logger,
) -> dict:
    """
    Generate a blog post from the video content.

    Args:
        job_dir: Path to job output directory
        meta: Video metadata
        transcript: Enriched transcript
        analysis: AI analysis with candidates
        clips_plan: Selected clips
        settings: User settings
        logger: Job logger

    Returns:
        dict with keys: md_path, txt_path, title, word_count
    """
    if not settings.get("blog_post_enabled", config.BLOG_POST_ENABLED):
        logger.info("Blog post generation disabled for this job")
        return {}

    from pipeline.analyzer import get_cached_api_response, save_cached_api_response, write_fallback_status

    provider = settings.get("ai_provider", config.DEFAULT_AI_PROVIDER)
    model = settings.get("ai_model", "")
    api_key = settings.get("api_key", "")

    # Check cache first
    blog_text = get_cached_api_response(job_dir, "blog_text", logger)
    if blog_text and isinstance(blog_text, str):
        logger.info("Loaded blog post from cache.")
        write_fallback_status(job_dir, "BLOGGER", "SUCCESS", "Loaded blog post from API cache.")
    else:
        # Build the blog generation prompt
        prompt = _build_blog_prompt(meta, transcript, analysis, clips_plan)
        logger.info(f"Generating blog post with {provider}...")

        blog_text = None
        last_error = None

        # Try primary provider, then configured fallbacks.
        providers = [(provider, model, api_key)]
        if config.AI_CROSS_PROVIDER_FALLBACK:
            if provider != "nvidia" and config.NVIDIA_API_KEY:
                providers.append(("nvidia", config.DEFAULT_NVIDIA_MODEL, config.NVIDIA_API_KEY))
            if provider != "claude" and config.ANTHROPIC_API_KEY:
                providers.append(("claude", config.DEFAULT_CLAUDE_MODEL, config.ANTHROPIC_API_KEY))
            if provider != "gemini" and config.GOOGLE_API_KEY:
                providers.append(("gemini", config.DEFAULT_GEMINI_MODEL, config.GOOGLE_API_KEY))

        for prov, mdl, key in providers:
            try:
                _creative_temp = getattr(config, "AI_TEMP_CREATIVE", 1.3)
                if prov == "claude":
                    blog_text = _call_claude(prompt, mdl, key, logger, temperature=_creative_temp)
                elif prov == "gemini":
                    blog_text = _call_gemini(prompt, mdl, key, logger, temperature=_creative_temp)
                else:
                    # nvidia / openai / legacy → shared provider-aware caller
                    from pipeline.analyzer import call_llm
                    blog_text = call_llm(prompt, settings, logger, temperature=_creative_temp)
                if blog_text:
                    logger.info(f"Blog post generated with {prov}")
                    write_fallback_status(
                        job_dir, 
                        "BLOGGER", 
                        "SUCCESS", 
                        f"Blog post generated successfully via AI using {prov}."
                    )
                    break
            except Exception as e:
                last_error = e
                logger.warning(f"Blog generation failed with {prov}: {e}")

        if not blog_text:
            logger.error(f"Blog generation failed entirely: {last_error}")
            # Generate a minimal fallback blog post
            blog_text = _generate_fallback_blog(meta, analysis, clips_plan)
            write_fallback_status(
                job_dir, 
                "BLOGGER", 
                "WARNING / FALLBACK", 
                f"Blog generation failed entirely: {str(last_error)}. Generated local fallback blog post."
            )

        # Save to cache
        save_cached_api_response(job_dir, "blog_text", blog_text, logger)

    # Clean up the text
    blog_text = blog_text.strip()

    # Save as Markdown
    title_slug = _sanitize_filename(meta.get("title", "blog_post"))
    md_path = os.path.join(job_dir, f"{title_slug}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(blog_text)
    logger.info(f"Blog post (MD) saved: {md_path}")

    # Save as plain text (strip Markdown formatting for Notepad)
    txt_path = os.path.join(job_dir, "blog_post.txt")
    plain_text = _md_to_plain_text(blog_text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(plain_text)
    logger.info(f"Blog post (TXT) saved: {txt_path}")

    word_count = len(blog_text.split())
    logger.info(f"Blog post: {word_count} words")

    return {
        "md_path": md_path,
        "txt_path": txt_path,
        "title": meta.get("title", ""),
        "word_count": word_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════════

def _build_blog_prompt(meta, transcript, analysis, clips_plan):
    """Build the blog generation prompt."""

    # Build clips summary
    clips_summary = ""
    for i, clip in enumerate(clips_plan, 1):
        clips_summary += f"\n{i}. **{clip.get('title', f'Clip {i}')}** "
        clips_summary += f"({_fmt_time(clip['segments'][0]['start'])} - {_fmt_time(clip['segments'][-1]['end'])})\n"
        clips_summary += f"   Virality: {clip.get('virality_score', 'N/A')}/10 | "
        clips_summary += f"Reason: {clip.get('reason', 'N/A')}\n"

    # Build transcript excerpt (first 3000 chars)
    transcript_text = ""
    for seg in transcript.get("segments", [])[:50]:
        transcript_text += f"[{_fmt_time(seg['start'])}] {seg['text']}\n"
    if len(transcript_text) > 3000:
        transcript_text = transcript_text[:3000] + "\n[...transcript continues...]"

    tone = config.BLOG_POST_TONE
    max_words = config.BLOG_POST_MAX_WORDS

    return f"""You are an expert content writer. Write a compelling, SEO-optimized blog post based on this video content.

## Video Information
- **Title**: {meta.get('title', 'Unknown')}
- **Channel**: {meta.get('channel', 'Unknown')}
- **Duration**: {meta.get('duration', 0)} seconds

## Video Summary
{analysis.get('summary', 'No summary available.')}

## Key Clips Identified
{clips_summary}

## Transcript Excerpt
{transcript_text}

## Writing Instructions

Write a {tone} blog post of approximately {max_words} words. Follow this EXACT structure:

1. **Title**: A compelling, SEO-friendly headline (NOT the video title verbatim)
2. **Meta Description**: One sentence (150-160 chars) for SEO
3. **Tags**: 5 relevant keywords/tags, comma-separated
4. **Introduction**: 2-3 sentences that hook the reader immediately. Reference the video topic.
5. **Key Takeaways**: 4-6 bullet points summarizing the most important insights
6. **Main Content**: 3-5 sections with H2 headings, each 200-300 words expanding on a key topic. Reference timestamps from the video where relevant (format: [Timestamp: MM:SS]).
7. **Conclusion**: 2-3 sentences wrapping up with a call-to-action

## Format Rules
- Output as clean Markdown
- Use ## for section headings, ### for sub-sections
- Use bullet points and numbered lists
- Write in a {tone} but engaging tone
- NO em dashes (use commas or periods instead)
- NO "powered by AI" or similar disclaimers
- DO reference specific timestamps from the video
- Make it genuinely valuable and readable, not a bullet dump

Return ONLY the blog post content in Markdown format.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# AI Providers
# ═══════════════════════════════════════════════════════════════════════════════

def _call_claude(prompt, model, api_key, logger):
    """Call Claude for blog generation."""
    import anthropic

    if not api_key:
        api_key = config.ANTHROPIC_API_KEY
    if not model:
        model = config.DEFAULT_CLAUDE_MODEL

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=config.AI_MAX_TOKENS_BLOG,
        temperature=0.6,  # Slightly more creative for blog writing
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_gemini(prompt, model, api_key, logger):
    """Call Gemini for blog generation."""
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
            temperature=0.6,
            max_output_tokens=config.AI_MAX_TOKENS_BLOG,
        ),
    )
    return response.text


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_fallback_blog(meta, analysis, clips_plan):
    """Generate a minimal blog post without AI."""
    title = meta.get("title", "Video Summary")
    summary = analysis.get("summary", "")

    sections = [f"# {title}\n"]
    if summary:
        sections.append(f"{summary}\n")

    sections.append("## Key Moments\n")
    for i, clip in enumerate(clips_plan, 1):
        sections.append(
            f"{i}. **{clip.get('title', f'Clip {i}')}** "
            f"[Timestamp: {_fmt_time(clip['segments'][0]['start'])}]\n"
            f"   {clip.get('reason', '')}\n"
        )

    sections.append("\n## Watch the Full Video\n")
    sections.append(f"Check out the full video by {meta.get('channel', 'the creator')} for more insights.\n")

    return "\n".join(sections)


def _sanitize_filename(name):
    """Sanitize a string for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = name[:100]  # Max length
    return name or "blog_post"


def _fmt_time(seconds):
    """Format seconds to MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _md_to_plain_text(md_text):
    """Convert Markdown to plain text for Notepad compatibility."""
    text = md_text
    # Remove markdown headings but keep text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove inline code
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Convert bullet points
    text = re.sub(r'^\s*[-*]\s+', '  - ', text, flags=re.MULTILINE)
    # Clean up extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
