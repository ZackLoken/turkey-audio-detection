"""Tests for resume/idempotency behavior per plan verification item 7."""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from turkey_audio_detection.config import BirdNetConfig, ClipConfig
from turkey_audio_detection.layout import RunLayout
from turkey_audio_detection.stages import stage_extract_clips, stage_run_birdnet


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

    stage_extract_clips(layout, ClipConfig())
    stage_extract_clips(layout, ClipConfig())

    # queue CSV is replace-on-rerun for same run_id
    final_queue = pd.read_csv(layout.queue_dir / "review_queue.csv")
    assert len(final_queue) == 1, f"Expected 1 queue row, got {len(final_queue)}"
    assert final_queue["item_id"].nunique() == 1


class _FakeAnalyzer:
    def __init__(self, *args, **kwargs) -> None:
        pass


def _make_fake_recording_cls(detections_by_path: dict, crash_on_path: str):
    class _FakeRecording:
        def __init__(self, analyzer, audio_path, **kwargs) -> None:
            self.audio_path = audio_path
            self.detections: list[dict] = []

        def analyze(self) -> None:
            if self.audio_path == crash_on_path:
                raise KeyboardInterrupt("simulated crash")
            self.detections = detections_by_path.get(self.audio_path, [])

    return _FakeRecording


def test_run_birdnet_resumes_after_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash mid-run must not force reprocessing files already checkpointed."""
    import birdnetlib
    import birdnetlib.analyzer

    layout = RunLayout.from_project_root(tmp_path, "run_20260424T010101Z")
    layout.ensure_dirs()

    paths = [str(tmp_path / f"data/ARU_01/file_{i}.wav") for i in range(4)]
    index_df = pd.DataFrame(
        {
            "aru_id": ["ARU_01"] * 4,
            "device_id": ["2MA09358"] * 4,
            "date": ["2026-03-10"] * 4,
            "time": ["05:00:00"] * 4,
            "datetime": ["2026-03-10T05:00:00"] * 4,
            "filepath": paths,
        }
    )
    index_df.to_csv(layout.index_dir / "file_index.csv", index=False)

    detections_by_path = {
        paths[i]: [
            {
                "start_time": 1.0,
                "end_time": 2.0,
                "common_name": "Wild Turkey",
                "scientific_name": "Meleagris gallopavo",
                "confidence": 0.9,
            }
        ]
        for i in range(4)
    }

    monkeypatch.setattr(birdnetlib.analyzer, "Analyzer", _FakeAnalyzer)
    monkeypatch.setattr(birdnetlib, "Recording", _make_fake_recording_cls(detections_by_path, crash_on_path=paths[2]))

    with pytest.raises(KeyboardInterrupt):
        stage_run_birdnet(layout, BirdNetConfig())

    progress_after_crash = pd.read_csv(layout.birdnet_dir / "birdnet_progress.csv")
    assert set(progress_after_crash["filepath"]) == {paths[0], paths[1]}
    detections_after_crash = pd.read_csv(layout.birdnet_dir / "detections_normalized.csv")
    assert set(detections_after_crash["audio_path"]) == {paths[0], paths[1]}

    # Resume: the fake Recording no longer crashes on any file.
    monkeypatch.setattr(birdnetlib, "Recording", _make_fake_recording_cls(detections_by_path, crash_on_path="never"))
    out_df = stage_run_birdnet(layout, BirdNetConfig())

    assert set(out_df["audio_path"]) == set(paths)
    assert len(out_df) == 4


def test_label_append_only_latest_wins_on_duplicate(tmp_path: Path) -> None:
    """Appending a second snapshot for the same item_id must preserve both raw rows but
    latest-by-timestamp must resolve to the most recent snapshot's presence flags."""
    from turkey_audio_detection.app import _append_label_row, _latest_by_item

    project_root = tmp_path

    base_row = {
        "item_id": "itm_x",
        "detection_id": "det_x",
        "reviewer_id": "reviewer_1",
        "reviewer_name": "reviewer_1",
        "regions_json": json.dumps(
            [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}],
            separators=(",", ":"),
        ),
        "other_birds_present": 1,
        "unsure": 0,
        "tom_present": 1,
        "hen_present": 0,
        "label_timestamp_utc": "2026-04-24T00:00:00+00:00",
        "session_id": "s1",
    }
    # Reviewer changes their mind: removes the Tom region, marks unsure.
    updated_row = {
        **base_row,
        "regions_json": json.dumps([], separators=(",", ":")),
        "tom_present": 0,
        "unsure": 1,
        "label_timestamp_utc": "2026-04-24T00:05:00+00:00",
    }

    _append_label_row(project_root, base_row)
    _append_label_row(project_root, updated_row)

    labels_path = project_root / "data" / "_outputs" / "review" / "labels" / "reviewer_1.csv"
    all_labels = pd.read_csv(labels_path)
    assert len(all_labels) == 2, "Raw label history must preserve both rows"

    resolved = _latest_by_item(all_labels)
    assert len(resolved) == 1
    assert int(resolved.iloc[0]["tom_present"]) == 0
    assert int(resolved.iloc[0]["unsure"]) == 1
