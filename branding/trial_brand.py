"""
Render the full duration of clip_12 with:
  • Hook banner: "TONY STARK SAYS TO HUMBLE YOURSELF ON PURPOSE" (ALL CAPS, black border)
  • Logo: branding/logo.png, top-right, 24% width
  • Music: TIMELESS.mp3 from 107s at 8% volume (same as final clip 2 settings)
  • Output: outputs/Video 33.../clips/clip_12_branded.mp4
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent          # branding/
ROOT        = HERE.parent                              # project root
FFMPEG      = str(ROOT / "ffmpeg-bin" / "ffmpeg.exe")
FFPROBE     = str(ROOT / "ffmpeg-bin" / "ffprobe.exe")

CAPTIONED   = ROOT / "outputs" / "Video 33 - Robert Downey Jr. On Living With Intention, Discip" / "clips" / "clip_12_captioned_barlow_black.mp4"
RAW         = ROOT / "outputs" / "Video 33 - Robert Downey Jr. On Living With Intention, Discip" / "clips" / "clip_12_raw.mp4"
LOGO        = HERE / "logo.png"
MUSIC       = ROOT / "music" / "TIMELESS.mp3"
OUTPUT      = ROOT / "outputs" / "Video 33 - Robert Downey Jr. On Living With Intention, Discip" / "clips" / "clip_12_branded.mp4"
FONT        = ROOT / "assets" / "fonts" / "BarlowCondensed-Black.ttf"

# ── Settings ──────────────────────────────────────────────────────────────────
HOOK_TEXT   = "TONY STARK SAYS TO HUMBLE YOURSELF ON PURPOSE"
HOOK_DUR    = 3.0           # banner on screen for 3 s
LOGO_PCT    = 24.0          # % of video width (4× original)
MARGIN_PCT  = 2.0           # corner margin % of width
MUSIC_SS    = 107.0         # start offset in TIMELESS.mp3
MUSIC_VOL   = 0.08          # 8 %

_NO_WIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

def probe_dims(path):
    r = subprocess.run(
        [FFPROBE,"-v","error","-select_streams","v:0",
         "-show_entries","stream=width,height","-of","csv=p=0:s=x",str(path)],
        capture_output=True, text=True, timeout=15, creationflags=_NO_WIN)
    w, h = (r.stdout or "1080x1920").strip().split("x")[:2]
    return int(w), int(h)

def probe_duration(path):
    r = subprocess.run(
        [FFPROBE,"-v","error","-show_entries","format=duration",
         "-of","default=nw=1:nk=1",str(path)],
        capture_output=True, text=True, timeout=15, creationflags=_NO_WIN)
    return float((r.stdout or "0").strip())

def esc(s):
    return s.replace("'", "\\'").replace(":", "\\:").replace(",","\\,")

def ff_path(p):
    return str(p).replace("\\","/").replace(":","\\:")

def main():
    print(f"  Input : {CAPTIONED.name}")
    print(f"  Output: {OUTPUT}")
    print()

    w, h = probe_dims(CAPTIONED)
    full_dur = probe_duration(CAPTIONED)
    print(f"  Video  : {w}×{h} (duration: {full_dur:.2f} s)")

    lw      = max(16, int(w * LOGO_PCT / 100))
    margin  = int(w * MARGIN_PCT / 100)

    with tempfile.TemporaryDirectory() as tmp:
        # ── Render the hook banner PNG ─────────────────────────────────────────
        banner_png = Path(tmp) / "banner.png"
        banner_ok  = False
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_size   = max(28, int(h * 5.5 / 100))
            try:
                pil_font = ImageFont.truetype(str(FONT), font_size)
            except Exception:
                pil_font = ImageFont.load_default()

            words   = HOOK_TEXT.split()
            lines   = []
            current = []
            tmp_img = Image.new("RGBA", (w, h))
            draw    = ImageDraw.Draw(tmp_img)
            for word in words:
                test = " ".join(current + [word])
                bb   = draw.textbbox((0,0), test, font=pil_font)
                if bb[2] - bb[0] > w * 0.82 and current:
                    lines.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                lines.append(" ".join(current))

            test_img  = Image.new("RGBA", (1, 1))
            test_draw = ImageDraw.Draw(test_img)
            bbs = [test_draw.textbbox((0,0), l, font=pil_font) for l in lines]
            max_w = max(b[2]-b[0] for b in bbs)
            total_h_text = sum(b[3]-b[1] for b in bbs) + (len(lines)-1) * int(font_size * 0.25)

            pad_x = int(w * 0.038)
            pad_y = int(h * 0.022)
            box_w = max_w + pad_x * 2
            box_h = total_h_text + pad_y * 2
            radius = int(h * 0.024)

            canvas = Image.new("RGBA", (w, box_h + pad_y*2), (0,0,0,0))
            draw   = ImageDraw.Draw(canvas)

            # border-only rounded rect — very pale yellow, semi-transparent
            border_col = (255, 248, 200, 200)
            bthick     = max(3, int(h * 0.004))
            x0 = (w - box_w) // 2
            x1 = x0 + box_w
            y0 = pad_y
            y1 = y0 + box_h
            draw.rounded_rectangle((x0, y0, x1, y1), radius=radius,
                                    outline=border_col, width=bthick)

            # text centred inside with black border (stroke)
            ty = y0 + pad_y
            line_gap = int(font_size * 0.22)
            stroke_w = max(2, int(font_size * 0.06))  # clean black border
            for line in lines:
                cx = w // 2
                # golden yellow text with black stroke
                draw.text((cx, ty), line, font=pil_font,
                          fill=(255, 200, 61, 255), anchor='mt',
                          stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
                bb2 = draw.textbbox((cx, ty), line, font=pil_font, anchor='mt')
                ty += (bb2[3] - bb2[1]) + line_gap

            canvas.save(str(banner_png))
            banner_ok  = True
            banner_h   = canvas.height
            print(f"  Banner : rendered {box_w}×{canvas.height}px via PIL")
        except Exception as e:
            print(f"  Banner : PIL unavailable ({e}), will use ffmpeg drawtext fallback")
            banner_h = int(h * 0.12)

        # ── Build the ffmpeg command ──────────────────────────────────────────
        fc       = []
        inputs   = [
            "-ss", "0", "-t", str(full_dur), "-i", str(CAPTIONED),
            "-ss", "0", "-t", str(HOOK_DUR),  "-i", str(RAW),
            "-loop","1", "-i", str(LOGO),
        ]
        next_idx = 3

        if banner_ok:
            inputs += ["-loop","1", "-t", f"{HOOK_DUR+0.3:.2f}", "-i", str(banner_png)]
            banner_idx = next_idx
            next_idx += 1

        # music: start at MUSIC_SS, take full_dur seconds
        inputs += ["-ss", str(MUSIC_SS), "-t", str(full_dur), "-i", str(MUSIC)]
        music_idx = next_idx

        # ── clean hook splice: first HOOK_DUR from raw, rest from captioned ──
        fc.append(f"[1:v]trim=0:{HOOK_DUR:.3f},setpts=PTS-STARTPTS[hk]")
        fc.append(f"[0:v]trim=start={HOOK_DUR:.3f},setpts=PTS-STARTPTS[bd]")
        fc.append(f"[hk][bd]concat=n=2:v=1:a=0[vcat]")
        cur = "[vcat]"

        # ── logo overlay ──────────────────────────────────────────────────────
        logo_w  = max(16, int(w * LOGO_PCT / 100))
        logo_mg = int(w * MARGIN_PCT / 100)
        fc.append(f"[2:v]scale={logo_w}:-1,format=rgba[logo]")
        fc.append(f"{cur}[logo]overlay=x=W-w-{logo_mg}:y={logo_mg}[vlogo]")
        cur = "[vlogo]"

        # ── banner overlay ────────────────────────────────────────────────────
        if banner_ok:
            sub_top = int(h * 0.48)
            sub_bot = int(h * 0.65)
            by      = int((sub_top + sub_bot) / 2 - banner_h / 2)
            by      = max(sub_top, min(by, sub_bot - banner_h))
            fc.append(f"[{banner_idx}:v]format=rgba,"
                      f"fade=in:st=0:d=0.25:alpha=1,"
                      f"fade=out:st={max(0, HOOK_DUR-0.25):.2f}:d=0.25:alpha=1[ban]")
            fc.append(f"{cur}[ban]overlay=x=(W-w)/2:y={by}"
                      f":enable='between(t,0,{HOOK_DUR:.3f})'[vban]")
            cur = "[vban]"
        else:
            # drawtext fallback
            font_safe = ff_path(FONT)
            txt_safe  = esc(HOOK_TEXT)
            fs        = max(28, int(h * 0.055))
            by        = int(h * 0.10)
            drawtext  = (f"drawtext=fontfile='{font_safe}':text='{txt_safe}'"
                         f":fontcolor=white:fontsize={fs}:x=(w-text_w)/2:y={by}"
                         f":enable='between(t,0,{HOOK_DUR:.3f})'")
            fc.append(f"{cur},{drawtext}[vtxt]")
            cur = "[vtxt]"

        # ── audio: original speech + music bed ───────────────────────────────
        fc.append(f"[0:a]asetpts=PTS-STARTPTS[aspeech]")
        fc.append(f"[{music_idx}:a]volume={MUSIC_VOL}[amusic]")
        fc.append(f"[aspeech][amusic]amix=inputs=2:duration=shortest[amix]")

        # ── assemble cmd ──────────────────────────────────────────────────────
        fc_str = ";".join(fc)
        cmd = [
            FFMPEG, "-y",
            *inputs,
            "-filter_complex", fc_str,
            "-map", f"{cur}",
            "-map", "[amix]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(full_dur),
            str(OUTPUT),
        ]

        print(f"\n  Running ffmpeg …\n")
        print("  " + " ".join(f'"{a}"' if " " in str(a) else str(a) for a in cmd))
        print()

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=_NO_WIN, timeout=400,
        )
        if result.returncode == 0 and OUTPUT.exists():
            size_mb = OUTPUT.stat().st_size / 1_048_576
            print(f"\n  ✅  Done!  →  {OUTPUT}  ({size_mb:.1f} MB)")
        else:
            print(f"\n  ❌  ffmpeg failed (rc={result.returncode})")
            print(result.stderr[-3000:])
            sys.exit(1)

if __name__ == "__main__":
    main()
