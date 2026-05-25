from pathlib import Path

import json
import pandas as pd

from turkey_audio_detection.adjudication import adjudicate_to_csv


def _row(item_id: str, detection_id: str, reviewer_id: str, regions, ts: str,
         tom: int, hen: int, other: int = 0, unsure: int = 0) -> dict:
    return {
        "item_id": item_id,
        "detection_id": detection_id,
        "reviewer_id": reviewer_id,
        "reviewer_name": reviewer_id,
        "regions_json": json.dumps(regions, separators=(",", ":")),
        "other_birds_present": other,
        "unsure": unsure,
        "tom_present": tom,
        "hen_present": hen,
        "label_timestamp_utc": ts,
        "session_id": f"sess_{reviewer_id}",
    }


def test_adjudication_outputs_files(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Reviewer A: i1=Tom-only, i2=Hen-only
    r1 = pd.DataFrame(
        [
            _row("i1", "d1", "r1",
                 [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}],
                 "2026-04-24T00:00:00+00:00", tom=1, hen=0),
            _row("i2", "d2", "r1",
                 [{"start_s": 0.2, "end_s": 0.9, "freq_min_hz": 600, "freq_max_hz": 2500, "label": "Hen"}],
                 "2026-04-24T00:01:00+00:00", tom=0, hen=1),
        ]
    )
    # Reviewer B: i1=Tom-only (agrees), i2=Tom-only (disagrees on both attributes)
    r2 = pd.DataFrame(
        [
            _row("i1", "d1", "r2",
                 [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}],
                 "2026-04-24T00:00:00+00:00", tom=1, hen=0),
            _row("i2", "d2", "r2",
                 [{"start_s": 1.0, "end_s": 2.0, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}],
                 "2026-04-24T00:01:00+00:00", tom=1, hen=0),
        ]
    )

    r1.to_csv(labels_dir / "r1.csv", index=False)
    r2.to_csv(labels_dir / "r2.csv", index=False)

    kappa_out = tmp_path / "kappa.csv"
    disagreements_out = tmp_path / "disagreements.csv"
    kappa_df, disagreements_df = adjudicate_to_csv(labels_dir, kappa_out, disagreements_out)

    assert kappa_out.exists()
    assert disagreements_out.exists()

    # One reviewer pair × two attributes = 2 kappa rows.
    assert len(kappa_df) == 2
    assert set(kappa_df["attribute"]) == {"tom_present", "hen_present"}
    assert (kappa_df["n_items"] == 2).all()

    # Both attributes mismatch on i2 → 2 disagreement rows, both with item_id == "i2".
    assert len(disagreements_df) == 2
    assert set(disagreements_df["attribute"]) == {"tom_present", "hen_present"}
    assert (disagreements_df["item_id"] == "i2").all()


def test_adjudication_excludes_unsure_by_default(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Both reviewers mark i1 as unsure; without include_unsure it must drop out of agreement stats.
    r1 = pd.DataFrame(
        [_row("i1", "d1", "r1", [], "2026-04-24T00:00:00+00:00", tom=0, hen=0, unsure=1)]
    )
    r2 = pd.DataFrame(
        [_row("i1", "d1", "r2", [], "2026-04-24T00:00:00+00:00", tom=0, hen=0, unsure=1)]
    )
    r1.to_csv(labels_dir / "r1.csv", index=False)
    r2.to_csv(labels_dir / "r2.csv", index=False)

    kappa_df, disagreements_df = adjudicate_to_csv(
        labels_dir, tmp_path / "kappa.csv", tmp_path / "disagreements.csv"
    )
    # With every row dropped by the unsure filter, every attribute reports n_items=0.
    assert (kappa_df["n_items"] == 0).all()
    assert len(disagreements_df) == 0
