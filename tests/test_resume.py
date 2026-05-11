"""Tests for resume/idempotency behavior per plan verification item 7."""

from pathlib import Path
from datetime import timezone, datetime

import numpy as np
import pandas as pd
import soundfile as sf

from turkey_audio_detection.config import ClipConfig
from turkey_audio_detection.layout import RunLayout
from turkey_audio_detection.stages import stage_extract_clips


def _make_wav(path: Path, duration_s: float = 6.0, sr: int = 16000) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    sf.write(str(path), data, sr)


def _make_detections(layout: RunLayout, wav_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "detection_id": "det_aaa",
                "project_root": str(layout.project_root),
                "aru_id": "ARU_01",
                "audio_path": str(wav_path),
                "start_time_s": 1.0,
                "end_time_s": 2.0,
                "species_code": "Meleagris gallopavo",
                "species_common_name": "Wild Turkey",
                "confidence": 0.9,
                "birdnet_model_version": "birdnetlib",
                "source_filename": wav_path.name,
                "source_row_index": 0,
            }
        ]
    )
    df.to_csv(layout.birdnet_dir / "detections_normalized.csv", index=False)
    return df


def test_extract_clips_no_duplicate_queue_rows_on_rerun(tmp_path: Path) -> None:
    """Running stage_extract_clips twice on the same detections must not produce duplicate queue rows."""
    layout = RunLayout.from_project_root(tmp_path, "run_20260424T010101Z")
    layout.ensure_dirs()

    aru_dir = tmp_path / "data" / "ARU_01"
    aru_dir.mkdir(parents=True, exist_ok=True)
    wav = aru_dir / "2MA09358_20260310_050001.wav"
    _make_wav(wav)
    _make_detections(layout, wav)

    q1 = stage_extract_clips(layout, ClipConfig())
    q2 = stage_extract_clips(layout, ClipConfig())

    # queue CSV is replace-on-rerun for same run_id
    final_queue = pd.read_csv(layout.queue_dir / "review_queue.csv")
    assert len(final_queue) == 1, f"Expected 1 queue row, got {len(final_queue)}"
    assert final_queue["item_id"].nunique() == 1


def test_label_append_only_latest_wins_on_duplicate(tmp_path: Path) -> None:
    """Appending a second label for the same item_id must preserve both rows raw
    but latest-wins must resolve to the most recent label."""
    from turkey_audio_detection.app import _append_label_row, _latest_by_item

    project_root = tmp_path

    base_row = {
        "item_id": "itm_x",
        "detection_id": "det_x",
        "reviewer_id": "reviewer_1",
        "reviewer_name": "Tester",
        "label": "Tom",
        "label_timestamp_utc": "2026-04-24T00:00:00+00:00",
        "session_id": "s1",
        "app_version": "0.1.0",
    }
    updated_row = {**base_row, "label": "Hen", "label_timestamp_utc": "2026-04-24T00:05:00+00:00"}

    _append_label_row(project_root, base_row)
    _append_label_row(project_root, updated_row)

    labels_path = project_root / "data" / "_outputs" / "review" / "labels" / "reviewer_1.csv"
    all_labels = pd.read_csv(labels_path)
    assert len(all_labels) == 2, "Raw label history must preserve both rows"

    resolved = _latest_by_item(all_labels)
    assert len(resolved) == 1
    assert resolved.iloc[0]["label"] == "Hen", "Latest-wins must resolve to the most recent label"
