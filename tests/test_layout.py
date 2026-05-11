from pathlib import Path

import pytest

from turkey_audio_detection.layout import RunLayout, validate_project_layout


def test_validate_project_layout_requires_aru_dirs(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        validate_project_layout(tmp_path)


def test_run_layout_ensures_expected_directories(tmp_path: Path) -> None:
    aru_dir = tmp_path / "data" / "ARU_01"
    aru_dir.mkdir(parents=True, exist_ok=True)

    layout = RunLayout.from_project_root(tmp_path, "run_20260424T010101Z")
    layout.ensure_dirs()

    assert layout.index_dir.exists()
    assert layout.birdnet_dir.exists()
    assert layout.queue_dir.exists()
    assert layout.clips_dir.exists()
    assert layout.manifests_dir.exists()
    assert layout.review_labels_dir.exists()
    assert layout.review_adjudication_dir.exists()
