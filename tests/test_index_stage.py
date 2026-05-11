from pathlib import Path

from turkey_audio_detection.config import IndexConfig
from turkey_audio_detection.layout import RunLayout
from turkey_audio_detection.stages import stage_index_data


def test_stage_index_data_writes_index_and_quarantine(tmp_path: Path) -> None:
    aru = tmp_path / "data" / "ARU_01"
    aru.mkdir(parents=True, exist_ok=True)

    valid = aru / "2MA09358_20260310_050001.wav"
    invalid = aru / "bad_name.wav"
    predeploy = aru / "2MA09358_20250210_050001.wav"
    valid.write_bytes(b"")
    invalid.write_bytes(b"")
    predeploy.write_bytes(b"")

    layout = RunLayout.from_project_root(tmp_path, "run_20260424T010101Z")
    layout.ensure_dirs()

    index_df, quarantine_df = stage_index_data(layout, IndexConfig())

    assert len(index_df) == 1
    assert len(quarantine_df) == 2
    assert (layout.index_dir / "file_index.csv").exists()
    assert (layout.index_dir / "quarantine_filenames.csv").exists()
