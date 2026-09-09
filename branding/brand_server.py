"""
Branding Server  —  local Flask UI for the branding finisher
============================================================
Run via run_brand.bat (or `python brand_server.py`). Opens a browser UI that:
  • lists jobs under outputs/ that have captioned clips,
  • previews each captioned clip in-browser,
  • lets you configure logo / hook banner / music engine + all upgrades,
  • uploads a logo image and a music file (or uses a folder / YouTube link),
  • runs the branding pass (engine.run_job) in the background with live progress.

Outputs land in outputs/<job>/clips/branded/. Nothing else is touched.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import webbrowser
from pathlib import Path

from flask import (Flask, jsonify, request, send_file, send_from_directory,
                   render_template, abort)

import engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"
PORT = int(os.getenv("BRANDING_PORT", "5056"))

app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("brand")

# job_name -> {running, i, n, msg, summary}
PROGRESS: dict = {}


# ── path safety ──────────────────────────────────────────────────────────────
def _job_dir(job: str) -> Path:
    p = (OUTPUTS / job).resolve()
    if OUTPUTS.resolve() not in p.parents and p != OUTPUTS.resolve():
        abort(400, "bad job")
    if not p.is_dir():
        abort(404, "job not found")
    return p


def _safe_file(job: str, rel: str) -> Path:
    base = _job_dir(job)
    p = (base / rel).resolve()
    if base not in p.parents and p != base:
        abort(400, "bad path")
    if not p.is_file():
        abort(404, "file not found")
    return p


# ── pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", port=PORT)


# ── jobs / clips ─────────────────────────────────────────────────────────────
@app.route("/api/jobs")
def api_jobs():
    """Every folder under outputs/ with brandable video — pipeline jobs
    (captioned clips) AND external folders of plain mp4s (shorts dropped in
    by hand). Uses the same discovery as the renderer, so a listed job always
    has renderable targets."""
    jobs = []
    if OUTPUTS.is_dir():
        for d in sorted(OUTPUTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            try:
                targets = engine.discover_targets(str(d))
            except Exception:
                log.exception(f"discover failed for {d.name}")
                continue
            if not targets:
                continue
            clips = d / "clips"
            n_cap = len(list(clips.glob("*_captioned*.mp4"))) if clips.is_dir() else 0
            jobs.append({"name": d.name, "captioned": len(targets),
                         "external": n_cap == 0,
                         "has_plan": (d / "clips_plan.json").exists()})
    return jsonify(jobs)


@app.route("/api/clips")
def api_clips():
    """Same discovery as engine.run_job — pipeline clips, plan-less captioned
    clips, and external mp4 folders all list (and therefore render)."""
    job = request.args.get("job", "")
    jd = _job_dir(job)
    variant = request.args.get("variant", "captioned")
    out = []
    for c in engine.discover_targets(str(jd), variant):
        rel = os.path.relpath(c["source_path"], jd).replace("\\", "/")
        out.append({
            "clip_name": c["clip_name"],
            "file": rel,
            "hook_phrase": c.get("hook_phrase", ""),
            "title": c.get("youtube_title") or c.get("title", ""),
            "duration": round(float(c.get("total_duration", 0) or 0), 1),
        })
    return jsonify(out)


@app.route("/api/file")
def api_file():
    job = request.args.get("job", "")
    rel = request.args.get("rel", "")
    p = _safe_file(job, rel)
    # conditional=True enables HTTP range requests so <video> can seek.
    return send_file(str(p), conditional=True, mimetype="video/mp4")


@app.route("/api/probe")
def api_probe():
    """fps + duration of one clip file — powers the frame-accurate trim UI."""
    import subprocess
    job = request.args.get("job", "")
    rel = request.args.get("rel", "")
    p = _safe_file(job, rel)
    fps, dur = 30.0, 0.0
    try:
        r = subprocess.run(
            [engine.FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,duration", "-of", "json", str(p)],
            capture_output=True, text=True, timeout=30)
        st = (json.loads(r.stdout).get("streams") or [{}])[0]
        num, den = (st.get("r_frame_rate") or "30/1").split("/")
        fps = float(num) / max(float(den), 1.0)
        dur = float(st.get("duration") or 0)
    except Exception:
        pass
    return jsonify({"fps": round(fps, 3), "duration": round(dur, 3)})


@app.route("/api/fonts")
def api_fonts():
    fonts = []
    if FONTS_DIR.is_dir():
        for f in sorted(FONTS_DIR.glob("*.ttf")):
            fonts.append({"name": f.stem, "path": f"assets/fonts/{f.name}"})
    return jsonify(fonts)


@app.route("/api/music-tracks")
def api_music_tracks():
    folder = request.args.get("folder", "assets/audio")
    tracks = engine.list_folder_tracks(folder)
    return jsonify([{"name": Path(t).name, "path": t} for t in tracks])


# ── uploads ──────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def api_upload():
    job = request.args.get("job", "")
    kind = request.args.get("kind", "asset")
    jd = _job_dir(job)
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, "no file")
    assets = jd / "branding_assets"
    assets.mkdir(exist_ok=True)
    safe = "".join(ch for ch in f.filename if ch.isalnum() or ch in "._- ")
    dest = assets / f"{kind}_{safe}"
    f.save(str(dest))
    rel = os.path.relpath(dest, PROJECT_ROOT).replace("\\", "/")
    return jsonify({"path": str(dest), "rel": rel})


# ── config ───────────────────────────────────────────────────────────────────
def _merge_ui_config(cfg_path: Path, data: dict) -> dict:
    """defaults <- saved job config <- UI payload.

    Merging into the SAVED config (not factory defaults) means keys the UI
    doesn't expose (grade brightness/gamma, banner auto-color, ...) survive a
    Save/Run instead of being silently reset. `clips` and `only_clips` are
    replaced wholesale: the UI owns the full override set, so a deep-merge
    would resurrect per-clip overrides the user just cleared (e.g. trim_end_s).
    """
    merged = engine._deep_merge(engine.load_config(str(cfg_path)), data)
    if "clips" in data:
        merged["clips"] = data["clips"]
    if "only_clips" in data:
        merged["only_clips"] = data["only_clips"]
    merged.pop("_picker", None)
    return merged


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    job = request.args.get("job", "")
    jd = _job_dir(job)
    cfg_path = jd / "branding_config.json"
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        merged = _merge_ui_config(cfg_path, data)
        cfg_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        return jsonify({"ok": True})
    return jsonify(engine.load_config(str(cfg_path)))


# ── run ──────────────────────────────────────────────────────────────────────
def _run_thread(job: str, cfg: dict):
    jd = str(_job_dir(job))
    PROGRESS[job] = {"running": True, "i": 0, "n": 0, "msg": "starting", "summary": None}

    def cb(i, n, msg):
        PROGRESS[job].update({"i": i, "n": n, "msg": msg})

    try:
        summary = engine.run_job(jd, cfg, log, progress_cb=cb)
        PROGRESS[job].update({"running": False, "msg": "done", "summary": summary})
    except Exception as e:
        log.exception("branding run failed")
        PROGRESS[job].update({"running": False, "msg": f"error: {e}", "summary": None})


@app.route("/api/run", methods=["POST"])
def api_run():
    job = request.args.get("job", "")
    jd = _job_dir(job)
    data = request.get_json(force=True) or {}
    cfg = _merge_ui_config(jd / "branding_config.json", data)
    # persist what we run
    (jd / "branding_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    if PROGRESS.get(job, {}).get("running"):
        return jsonify({"ok": False, "error": "already running"}), 409
    threading.Thread(target=_run_thread, args=(job, cfg), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    job = request.args.get("job", "")
    return jsonify(PROGRESS.get(job, {"running": False, "i": 0, "n": 0, "msg": "idle", "summary": None}))


def _open_browser():
    time.sleep(1.0)
    try:
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
    except Exception:
        pass


if __name__ == "__main__":
    print(f"\n  Branding tool  →  http://127.0.0.1:{PORT}/\n", flush=True)
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
