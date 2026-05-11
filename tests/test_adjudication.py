from pathlib import Path

import pandas as pd

from turkey_audio_detection.adjudication import adjudicate_to_csv


def test_adjudication_outputs_files(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    r1 = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "detection_id": "d1",
                "reviewer_id": "r1",
                "reviewer_name": "A",
                "label": "Tom",
                "label_timestamp_utc": "2026-04-24T00:00:00+00:00",
                "session_id": "s1",
                "app_version": "0.1.0",
            },
            {
                "item_id": "i2",
                "detection_id": "d2",
                "reviewer_id": "r1",
                "reviewer_name": "A",
                "label": "Hen",
                "label_timestamp_utc": "2026-04-24T00:01:00+00:00",
                "session_id": "s1",
                "app_version": "0.1.0",
            },
        ]
    )
    r2 = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "detection_id": "d1",
                "reviewer_id": "r2",
                "reviewer_name": "B",
                "label": "Tom",
                "label_timestamp_utc": "2026-04-24T00:00:00+00:00",
                "session_id": "s2",
                "app_version": "0.1.0",
            },
            {
                "item_id": "i2",
                "detection_id": "d2",
                "reviewer_id": "r2",
                "reviewer_name": "B",
                "label": "Background",
                "label_timestamp_utc": "2026-04-24T00:01:00+00:00",
                "session_id": "s2",
                "app_version": "0.1.0",
            },
        ]
    )

    r1.to_csv(labels_dir / "r1.csv", index=False)
    r2.to_csv(labels_dir / "r2.csv", index=False)

    kappa_out = tmp_path / "kappa.csv"
    disagreements_out = tmp_path / "disagreements.csv"
    kappa_df, disagreements_df = adjudicate_to_csv(labels_dir, kappa_out, disagreements_out)

    assert len(kappa_df) == 1
    assert kappa_out.exists()
    assert disagreements_out.exists()
    assert len(disagreements_df) == 1
