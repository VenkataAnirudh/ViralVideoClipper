# External LLM Stitch Pass — Implementation Plan (Standalone 2-Pass Approach)

## Goal
Implement a 2-pass AI stitching process specifically for the external LLM route. 
The process must:
1. **Pass 1 (Identification):** Identify short clips and intelligently propose merges while maintaining awareness of surrounding clips.
2. **Pass 2 (Execution/Stitching):** Actually perform the stitches, generating merged metadata (titles, descriptions, hashtags) for the combined clips.
3. Be isolated in a **new standalone file** (`pipeline/external_stitcher.py`) to guarantee zero disruption to the existing API route.
4. Integrate cleanly into the pipeline and output a perfectly formatted `clips_plan.json`.

## Proposed Architecture

We will create a new module `pipeline/external_stitcher.py` and hook it into the `clip_selector.py` flow. 

### 1. New Module: `pipeline/external_stitcher.py`

This file will contain the core 2-pass logic.

#### Pass 1: The Identification Pass
- **Input:** The full list of normalized external candidates.
- **Logic:** 
  - Scan for clips below the `min_duration`.
  - Build a prompt containing the short clips AND their neighboring clips (to provide context and flow intention).
  - Call `gpt-5-nano` (via `api_provider.run_json_task`).
  - **Prompt Goal:** Ask the AI to identify *which* clips should be merged to solve the duration issue, returning a list of `merge_groups` (e.g., `["w01_c01", "w01_c02"]`).

#### Pass 2: The Metadata Generation Pass
- **Input:** The `merge_groups` identified in Pass 1, plus the full data for those specific clips.
- **Logic:**
  - For each approved merge group, build a prompt containing the full metadata of the clips to be merged (titles, descriptions, hashtags, segments).
  - Call `gpt-5-nano`.
  - **Prompt Goal:** Ask the AI to generate a cohesive, merged identity for the new combined clip.
  - **Output Schema:**
    ```json
    {
      "merged_metadata": {
        "youtube_title": "...",
        "description_text": "...",
        "description_hashtags": "merged topic hashtags only",
        "youtube_tags": "merged comma-separated tags",
        "hook_phrase": "...",
        "takeaway": "..."
      }
    }
    ```
- **Local Application:**
  - Concatenate the `segments` of the merged clips in chronological order.
  - Apply the AI-generated `merged_metadata`.
  - Re-append the generic filler hashtags (e.g., `#viral`, `#fyp`) locally to save AI tokens.
  - Generate a new `candidate_id` (e.g., `merged_w01_c01+w01_c02`).

#### Integration Function: `run_external_2pass_stitch()`
- A public function that orchestrates Pass 1 and Pass 2.
- Takes the candidates, filters out the consumed original clips, appends the newly minted merged clips, sorts them chronologically, and returns the final candidate list.

### 2. Integration Points (Minimal touch to existing code)

#### [MODIFY] `pipeline/runner.py`
- We must stop the pipeline from aggressively skipping stitching for the external route *if* the user has an OpenAI key and wants to stitch.
- Update lines 461 and 778 to conditionally allow stitching:
  ```python
  # Instead of hardcoding skip_stitch_pass = True
  openai_key = settings.get("openai_api_key") or getattr(config, "OPENAI_API", "")
  if not openai_key:
      settings["skip_stitch_pass"] = True
  settings["skip_validation"] = True # Keep validation skipped for external
  ```

#### [MODIFY] `pipeline/clip_selector.py`
- Around line 118, where `run_stitch_pass` is currently called.
- Add logic to branch to our new standalone module if it's the external route:
  ```python
  analysis_mode = str(settings.get("analysis_mode", "manual")).strip().lower()
  if analysis_mode == "manual":
      # Use the new 2-pass standalone stitcher
      from pipeline.external_stitcher import run_external_2pass_stitch
      mapped_candidates = run_external_2pass_stitch(
          job_dir=job_dir,
          candidates=mapped_candidates,
          min_duration=min_duration,
          max_duration=max_duration,
          settings=settings,
          logger=logger
      )
  else:
      # Use the existing 1-pass API route stitcher
      from pipeline.analyzer import run_stitch_pass
      # ... existing logic ...
  ```

#### [MODIFY] `config.py`
- Add `GENERIC_FILLER_HASHTAGS = "#viral #fyp #shorts #trending #foryou #foryoupage #explore #reels #youtubeshorts #viralvideo #motivation #mindset #trendingshorts #subscribe #podcast #podcastclips #viralshorts #shortsfeed #reelsviral #trend #explorepage #contentcreator #inspiration #selfimprovement #growth #wellness #health #healthtips #lifehacks #knowledge #learnontiktok #educational #facts #science #biohacking #longevity #wellbeing #mentalhealth #dailymotivation #success"`
- This allows us to strip these out before sending to the LLM (saving tokens) and cleanly append them locally after Pass 2.

## Verification Plan

1. Create `pipeline/external_stitcher.py` and implement the 2-pass logic.
2. Modify `runner.py`, `clip_selector.py`, and `config.py` as outlined.
3. Run the pipeline on the `Video 35` output directory.
4. Verify that the new `external_stitcher.py` accurately identifies the 14 short clips, successfully merges logical pairs, generates fused metadata, and produces a valid, finalized `clips_plan.json` without breaking any existing functionality.
