# ====================================================================
# download_viral_music.py — Top viral/trending YouTube music grabber
# ====================================================================
# Downloads the top N recent trending/viral music tracks from YouTube
# into music/viral/ as 320kbps MP3s, for use as shorts background music.
#
# Completely separate from test_download.py (the main video downloader).
# Re-runs top up with NEW tracks only (.archive.txt remembers past IDs).
#
# Usage:  python download_viral_music.py [count]     (default 10)
# ====================================================================

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIRAL_DIR = ROOT / "music" / "viral"
ARCHIVE = VIRAL_DIR / ".archive.txt"

# Podcast-clip-apt background music: instrumental / ambient / cinematic /
# lo-fi — the styles that sit under spoken word without fighting it.
# Global audience: regional-language results are filtered out below.
SEARCH_QUERIES = [
    "ytsearch30:no copyright background music podcast chill instrumental",
    "ytsearch30:viral ambient instrumental edit audio",
    "ytsearch30:motivational cinematic instrumental viral shorts",
    "ytsearch30:epic cinematic instrumental trending edit",
    "ytsearch30:lofi chill instrumental beat trending",
]

MIN_DUR_S, MAX_DUR_S = 60, 480   # real tracks only: 1-8 min (kills hour-mixes)
MIN_VIEWS = 300_000              # viral means viral — cuts long-tail regional uploads

# Global-audience filters: drop titles in non-Latin scripts (Devanagari, Arabic,
# CJK, Korean, Tamil/Telugu/Bengali/Gurmukhi...) or tagged with regional markers.
_NON_LATIN_RE = None  # compiled lazily in _is_global_title
_BLOCKED_WORDS = {
    "hindi", "punjabi", "bollywood", "marathi", "bhojpuri", "tamil", "telugu",
    "kannada", "gujarati", "haryanvi", "desi", "mashup", "jukebox",
    "nepali", "nepal", "pakistani", "flute", "shehnai", "sitar", "tabla",
    "carnatic", "children", "kids", "nursery",
}


def _is_global_title(title: str) -> bool:
    global _NON_LATIN_RE
    import re as _re
    if _NON_LATIN_RE is None:
        _NON_LATIN_RE = _re.compile(
            r"[؀-ۿऀ-ॿঀ-৿਀-੿଀-୿"
            r"஀-௿ఀ-౿ಀ-೿぀-ヿ一-鿿가-힯]"
        )
    if _NON_LATIN_RE.search(title):
        return False
    low = title.lower()
    return not any(w in low for w in _BLOCKED_WORDS)


def _dedupe_key(title: str) -> str:
    """Same-song re-uploads collapse to one entry (first 4 significant words)."""
    import re as _re
    words = _re.findall(r"[a-z0-9]+", title.lower())
    return " ".join(words[:4])


def _ffmpeg() -> str:
    for cand in (ROOT / "ffmpeg-bin" / "ffmpeg.exe", ROOT / "ffmpeg-gpu" / "ffmpeg.exe"):
        if cand.exists():
            return str(cand)
    return "ffmpeg"


def _cookies() -> str | None:
    for cand in (ROOT / "cookies.txt", Path.cwd() / "cookies.txt"):
        if cand.is_file():
            return str(cand)
    return None


def _safe(text: str) -> str:
    """Console-safe text for Windows code pages."""
    return str(text).encode("ascii", "replace").decode()


def _flat_entries(source: str) -> list[dict]:
    cmd = [sys.executable, "-m", "yt_dlp", "-J", "--flat-playlist", "--no-warnings", source]
    ck = _cookies()
    if ck:
        cmd += ["--cookies", ck]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=180)
        if res.returncode != 0:
            return []
        data = json.loads(res.stdout)
    except Exception:
        return []
    return [e for e in (data.get("entries") or []) if e]


def _archived_ids() -> set[str]:
    ids = set()
    if ARCHIVE.exists():
        for line in ARCHIVE.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) == 2:
                ids.add(parts[1])
    return ids


def pick_candidates(count: int) -> list[dict]:
    seen = _archived_ids()

    # Gather across all queries, then rank globally by view count.
    pool: dict[str, dict] = {}
    for query in SEARCH_QUERIES:
        for e in _flat_entries(query):
            vid = e.get("id") or ""
            title = e.get("title") or vid
            dur = e.get("duration") or 0
            if not vid or vid in seen or vid in pool:
                continue
            if dur and not (MIN_DUR_S <= dur <= MAX_DUR_S):
                continue
            if (e.get("live_status") or "") in ("is_live", "is_upcoming"):
                continue
            if not _is_global_title(title):
                continue
            views = e.get("view_count") or 0
            if views < MIN_VIEWS:
                continue
            pool[vid] = {"id": vid, "title": title, "views": views}

    ranked = sorted(pool.values(), key=lambda p: p["views"], reverse=True)

    picks: list[dict] = []
    dedupe_seen: set[str] = set()
    for p in ranked:
        key = _dedupe_key(p["title"])
        if key in dedupe_seen:
            continue
        dedupe_seen.add(key)
        picks.append(p)
        if len(picks) >= count:
            break
    return picks


def download_track(video_id: str, ffmpeg: str) -> tuple[bool, str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_tmpl = str(VIRAL_DIR / "%(title).80s [%(id)s].%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings", "--no-playlist",
        "--windows-filenames",
        "--ffmpeg-location", ffmpeg,
        "-f", "bestaudio/best",
        "-x", "--audio-format", "mp3", "--audio-quality", "320K",
        "--download-archive", str(ARCHIVE),
        "-o", out_tmpl,
        url,
    ]
    ck = _cookies()
    if ck:
        cmd += ["--cookies", ck]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    tail = ((res.stdout or "") + (res.stderr or ""))[-300:].strip()
    return res.returncode == 0, tail


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    VIRAL_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    print(f"Fetching top {count} recent viral music candidates from YouTube...")
    picks = pick_candidates(count)
    if not picks:
        print("No candidates found (feed + search both empty) — check network/yt-dlp.")
        return 1

    ok_count = 0
    for i, p in enumerate(picks, 1):
        print(f"[{i}/{len(picks)}] {_safe(p['title'])}")
        ok, tail = download_track(p["id"], ffmpeg)
        if ok:
            ok_count += 1
        else:
            print(f"    FAILED: {_safe(tail)}")

    print(f"\nDone: {ok_count}/{len(picks)} tracks saved to {VIRAL_DIR} (320kbps MP3)")
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
