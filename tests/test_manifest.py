import json
from pathlib import Path

from turkey_audio_detection.manifest import write_manifest


def test_write_manifest_atomic(tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    payload = {"run_id": "run_1", "status": "ok"}
    write_manifest(out, payload)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "run_1"
