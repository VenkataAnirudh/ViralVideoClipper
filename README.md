# Video Clipper

Video Clipper is a local Flask app that turns long-form videos into short, captioned clips. It downloads a source video, transcribes it, asks an AI model to identify high-potential moments, snaps clip boundaries to cleaner sentence cuts, extracts face-aware vertical clips, burns animated captions, mixes optional background music, and generates social copy.

The project is built for a Windows GPU workstation and is designed to keep all generated videos, transcripts, logs, API keys, and local binaries out of Git.

## Features

- Browser dashboard served by Flask.
- YouTube/video download through `yt-dlp`.
- Local transcription with `faster-whisper`, with optional cloud fallbacks.
- AI clip analysis through Featherless, Claude, Gemini, or OpenRouter.
- Sentence-aware clip selection and overlap removal.
- GPU-oriented extraction with FFmpeg/NVENC where available.
- Face-aware/speaker-aware crop logic.
- Word-timed animated captions and optional background music mixing.
- Social copy generation per clip.
- Resume, retry, reburn, preview, download, and log endpoints.

## Project Layout

```text
app.py                  Flask server and API routes
config.py               Central defaults, provider settings, paths, captions, GPU knobs
requirements.txt        Python package dependencies, excluding CUDA PyTorch
install.bat             One-time Windows setup for venv, GPU packages, fonts
run.bat                 Starts the local app on http://localhost:5000
download_fonts.py       Downloads caption fonts into assets/fonts
test_download.py        Downloader module currently used by the runner

pipeline/
  runner.py             Main pipeline orchestrator
  transcriber.py        Local/cloud transcription
  analyzer.py           AI transcript analysis and fallback candidate generation
  clip_selector.py      Timestamp cleanup and clip plan creation
  extractor.py          FFmpeg extraction and crop logic
  speaker_tracking.py   Face/speaker sampling and crop expression generation
  captioner.py          Captions and background music burn-in
  copywriter.py         Per-clip social copy
  blogger.py            Optional blog post generation
  logger.py             Per-job debug logging

templates/              Flask HTML template
static/                 Dashboard JavaScript and CSS
assets/fonts/           Runtime-downloaded caption fonts
music/                  Optional local background music pool
outputs/                Generated job output, ignored by Git
```

## Requirements

- Windows.
- Python 3.11 or newer.
- NVIDIA GPU recommended for the intended fast path.
- CUDA-compatible PyTorch stack installed by `install.bat`.
- FFmpeg/FFprobe available either in `ffmpeg-bin`, `ffmpeg-gpu`, or on `PATH`.
- At least one AI provider API key for best results.

## Setup

1. Create your local environment file:

   ```bat
   copy .env.example .env
   ```

2. Edit `.env` and add the keys you want to use. Do not commit `.env`.

3. Run the one-time installer:

   ```bat
   install.bat
   ```

4. Start the app:

   ```bat
   run.bat
   ```

5. Open the dashboard:

   ```text
   http://localhost:5000
   ```

## Environment Variables

Common variables:

```text
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
FEATHERLESS_API_KEY=
SMALLEST_API_KEY=
HF_TOKEN=
DIARIZATION_ENABLED=true
OUTPUT_DIR=outputs
FFMPEG_PATH=ffmpeg-bin\ffmpeg.exe
FFPROBE_PATH=ffmpeg-bin\ffprobe.exe
GPU_FFMPEG_PATH=ffmpeg-gpu\ffmpeg.exe
GPU_FFPROBE_PATH=ffmpeg-gpu\ffprobe.exe
```

Provider keys can also be supplied from the dashboard for a single run. `.env` is the better local workflow.

## Pipeline

1. `runner.py` creates a job directory and progress reporter.
2. `test_download.py` downloads the source video and writes `meta.json`.
3. `transcriber.py` extracts audio and writes `transcript.json`.
4. `analyzer.py` asks the configured AI provider for clip candidates and writes `analysis.json`.
5. `clip_selector.py` cleans boundaries and writes `clips_plan.json`.
6. `extractor.py` renders raw clips into `clips/`.
7. `captioner.py` burns animated captions and optional music.
8. `copywriter.py` writes `social_copy.json`.
9. `runner.py` packages the final `results.json` consumed by the dashboard.

## What To Commit

Commit the application source and setup files:

```text
README.md
.gitignore
.env.example
app.py
config.py
requirements.txt
install.bat
run.bat
download_fonts.py
test_download.py
architecture.md
pipeline/
templates/
static/
assets/fonts/.gitkeep
music/.gitkeep
```

Keep these out of Git:

```text
.env
cookies.txt
venv/
.uv-cache/
.feather_agent/
__pycache__/
outputs/
scratch/
ffmpeg-bin/ except required LFS-tracked executables
ffmpeg-gpu/ except required LFS-tracked executables
music/*.mp3
assets/fonts/*.ttf
Architecture.txt
*.mp4, *.wav, *.log, *.part
```

Reasons:

- `.env` and `cookies.txt` contain credentials.
- `outputs/` contains generated videos, transcripts, logs, and downloaded source media.
- `venv/`, `.uv-cache/`, and `__pycache__/` are machine-local build/runtime artifacts.
- FFmpeg executables are larger than normal GitHub file limits, so they must be tracked with Git LFS if committed.
- `music/*.mp3` and output videos are large media files and may have licensing constraints.
- `scratch/` contains temporary experiments.
- `Architecture.txt` appears to duplicate `architecture.md`; keep the Markdown version.

## Private GitHub Push

After reviewing `.env` and confirming no secrets are staged:

```bat
git init
git lfs install
git add README.md .gitignore .gitattributes .env.example app.py config.py requirements.txt install.bat run.bat download_fonts.py test_download.py architecture.md pipeline templates static assets/fonts/.gitkeep music/.gitkeep ffmpeg-bin/ffmpeg.exe ffmpeg-bin/ffprobe.exe ffmpeg-bin/ffplay.exe ffmpeg-gpu/ffmpeg.exe ffmpeg-gpu/ffprobe.exe ffmpeg-gpu/ffplay.exe
git status
git commit -m "Initial private video clipper commit"
```

Then create a private GitHub repository and push:

```bat
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/video-clipper.git
git push -u origin main
```

If you use GitHub CLI:

```bat
gh repo create video-clipper --private --source=. --remote=origin --push
```

## Safety Check Before Pushing

Run these before the first push:

```bat
git status --short
git diff --cached --name-only
```

The staged list should not include `.env`, `cookies.txt`, `outputs/`, `venv/`, generated videos, logs, local music files, or the extracted FFmpeg docs/presets folder. The only FFmpeg files staged should be the six `.exe` files tracked through Git LFS.
