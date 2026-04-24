"""Run manifest helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def write_manifest(path: Path, payload: dict) -> None:
    """Write manifest atomically via temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_stage_manifest(
    *,
    run_id: str,
    stage: str,
    project_root: Path,
    config_snapshot: dict,
    stage_outputs: dict,
    status: str = "completed",
    input_file_count: int | None = None,
    input_content_hash: str | None = None,
    python_version: str | None = None,
    birdnet_version: str | None = None,
) -> dict:
    now = utc_now_iso()
    return {
        "run_id": run_id,
        "stage": stage,
        "started_at_utc": now,
        "completed_at_utc": now,
        "project_roots": [str(project_root)],
        "config_snapshot": config_snapshot,
        "input_file_count": input_file_count,
        "input_content_hash": input_content_hash,
        "stage_outputs": stage_outputs,
        "package_version": "0.1.0",
        "python_version": python_version,
        "birdnet_version": birdnet_version,
        "status": status,
    }
