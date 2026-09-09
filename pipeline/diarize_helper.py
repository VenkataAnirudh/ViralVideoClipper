# fileName: pipeline/diarize_helper.py
"""
Standalone Diarization Helper Script
=====================================
Runs pyannote.audio speaker diarization in an isolated virtual environment.

Confirmed working with pyannote.audio 4.0.4:
  - pipeline() returns DiarizeOutput
  - DiarizeOutput.speaker_diarization is an Annotation object
  - Iterate with: for turn, speaker in output.speaker_diarization
  - Audio loaded via soundfile to bypass torchcodec/FFmpeg DLL issues on Windows
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run pyannote speaker diarization in isolation")
    parser.add_argument("--audio",  required=True,
                        help="Path to input audio WAV file (16 kHz mono)")
    parser.add_argument("--token",  required=True,
                        help="Hugging Face user access token")
    parser.add_argument("--output", required=True,
                        help="Path to save output diarization JSON")
    args = parser.parse_args()

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        print(f"Failed to import dependencies: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        import soundfile as sf
    except ImportError:
        print(
            "soundfile is not installed in venv1.\n"
            "Run: venv1\\Scripts\\pip install soundfile",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── Load waveform with soundfile (bypasses torchcodec DLL on Windows) ─────
    try:
        data, sample_rate = sf.read(
            args.audio, dtype="float32", always_2d=True)
        # data shape: (frames, channels) -> pyannote wants (channels, frames)
        waveform = torch.from_numpy(data.T)   # shape: (1, N) for mono
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as e:
        print(
            f"Failed to read audio file with soundfile: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Load pipeline ──────────────────────────────────────────────────────────
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=args.token,
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
    except Exception as e:
        import traceback
        print(f"Failed to load pyannote pipeline:\n{traceback.format_exc()}",
              file=sys.stderr)
        sys.exit(1)

    # ── Run diarization ────────────────────────────────────────────────────────
    try:
        output = pipeline(audio_input)

        # pyannote.audio 4.x: output is DiarizeOutput
        # output.speaker_diarization is an Annotation
        # iteration yields (Segment, speaker_label) 2-tuples
        annotation = output.speaker_diarization

        timeline = []
        for turn, speaker in annotation:
            timeline.append({
                "t_start": float(turn.start),
                "t_end":   float(turn.end),
                "speaker": str(speaker),
            })

        if not timeline:
            raise RuntimeError(
                "Diarization returned 0 segments — check your HF token and "
                "that you have accepted the model terms at "
                "https://hf.co/pyannote/speaker-diarization-3.1"
            )

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)

        # ASCII-only print to avoid CP1252 UnicodeEncodeError on Windows cmd
        print(f"[OK] Diarization completed - {len(timeline)} segments")
        sys.exit(0)

    except Exception:
        import traceback
        print(f"Error during diarization inference:\n{traceback.format_exc()}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
