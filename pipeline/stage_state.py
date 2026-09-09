"""Lightweight stage state tracking for pipeline resume decisions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


STATE_FILE = "stage_state.json"


def _job_path(job_dir: str | Path) -> Path:
    return Path(job_dir)


def _rel(job_dir: Path, path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        return str(p.relative_to(job_dir)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _sha1_file(path: Path, max_bytes: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except Exception:
        return None


def settings_fingerprint(settings: Optional[Dict[str, Any]], keys: Optional[Iterable[str]] = None) -> str:
    data = dict(settings or {})
    if keys is not None:
        data = {k: data.get(k) for k in sorted(keys)}
    else:
        data = {
            k: v for k, v in data.items()
            if "key" not in k.lower() and "token" not in k.lower() and "secret" not in k.lower()
        }
    payload = json.dumps(data, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def artifact_info(job_dir: str | Path, paths: Iterable[str | Path]) -> Dict[str, Dict[str, Any]]:
    base = _job_path(job_dir)
    out: Dict[str, Dict[str, Any]] = {}
    for item in paths:
        rel = _rel(base, item)
        path = Path(item) if Path(item).is_absolute() else base / rel
        info: Dict[str, Any] = {
            "exists": path.exists(),
            "size": 0,
            "mtime": None,
            "json_valid": None,
            "count": None,
            "sha1": None,
        }
        if path.exists():
            try:
                stat = path.stat()
                info["size"] = int(stat.st_size)
                info["mtime"] = float(stat.st_mtime)
                info["sha1"] = _sha1_file(path)
            except Exception:
                pass
            if path.suffix.lower() == ".json":
                try:
                    with path.open("r", encoding="utf-8") as f:
                        parsed = json.load(f)
                    info["json_valid"] = True
                    if isinstance(parsed, (list, dict)):
                        info["count"] = len(parsed)
                except Exception as exc:
                    info["json_valid"] = False
                    info["error"] = str(exc)
        out[rel] = info
    return out


def read_stage_state(job_dir: str | Path) -> Dict[str, Any]:
    path = _job_path(job_dir) / STATE_FILE
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_stage_state(job_dir: str | Path, state: Dict[str, Any]) -> None:
    path = _job_path(job_dir) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def record_stage(
    job_dir: str | Path,
    stage: str,
    status: str,
    *,
    inputs: Optional[Iterable[str | Path]] = None,
    outputs: Optional[Iterable[str | Path]] = None,
    counts: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    settings_keys: Optional[Iterable[str]] = None,
    reason: str = "",
) -> Dict[str, Any]:
    base = _job_path(job_dir)
    state = read_stage_state(base)
    entry = {
        "stage": stage,
        "status": status,
        "reason": reason,
        "updated_at": time.time(),
        "settings_fingerprint": settings_fingerprint(settings, settings_keys),
        "inputs": artifact_info(base, inputs or []),
        "outputs": artifact_info(base, outputs or []),
        "counts": counts or {},
    }
    state[stage] = entry
    write_stage_state(base, state)
    return entry


def validate_artifacts(
    job_dir: str | Path,
    required: Iterable[str | Path],
    *,
    json_files: Optional[Iterable[str | Path]] = None,
) -> tuple[bool, str, Dict[str, Dict[str, Any]]]:
    base = _job_path(job_dir)
    required_rels = [_rel(base, p) for p in required]
    json_rels = {_rel(base, p) for p in (json_files or [])}
    info = artifact_info(base, required_rels)
    for rel in required_rels:
        item = info.get(rel, {})
        if not item.get("exists"):
            return False, f"missing {rel}", info
        if int(item.get("size") or 0) <= 0:
            return False, f"empty {rel}", info
        if rel in json_rels and item.get("json_valid") is not True:
            return False, f"invalid_json {rel}", info
    return True, "valid", info


def inputs_changed(
    job_dir: str | Path,
    stage: str,
    inputs: Iterable[str | Path],
    *,
    settings: Optional[Dict[str, Any]] = None,
    settings_keys: Optional[Iterable[str]] = None,
) -> tuple[bool, str]:
    base = _job_path(job_dir)
    state = read_stage_state(base)
    previous = state.get(stage)
    if not previous:
        return True, "no previous stage state"
    current_inputs = artifact_info(base, inputs)
    if previous.get("inputs") != current_inputs:
        return True, "input artifacts changed"
    current_fp = settings_fingerprint(settings, settings_keys)
    if previous.get("settings_fingerprint") != current_fp:
        return True, "settings fingerprint changed"
    return False, "unchanged"
