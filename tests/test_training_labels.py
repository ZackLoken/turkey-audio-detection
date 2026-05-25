"""Tests for the multi-reviewer label aggregator."""

import json
from pathlib import Path

import pandas as pd
import pytest

from turkey_audio_detection.training_labels import _majority_vote, aggregate_reviewers


def _label_row(item_id: str, reviewer_id: str, regions, ts: str, tom: int, hen: int, other: int = 0, unsure: int = 0) -> dict:
    return {
        "item_id": item_id,
        "detection_id": "det_" + item_id,
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


def test_majority_vote_basic() -> None:
    assert _majority_vote(pd.Series([1, 1, 0])) == (1, True)
    assert _majority_vote(pd.Series([0, 0, 1])) == (0, True)


def test_majority_vote_tie_returns_zero_no_consensus() -> None:
    assert _majority_vote(pd.Series([1, 0])) == (0, False)
    assert _majority_vote(pd.Series([1, 1, 0, 0])) == (0, False)


def test_majority_vote_empty() -> None:
    assert _majority_vote(pd.Series(dtype=int)) == (0, False)


def test_aggregate_empty(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_reviewers(labels_dir)
    assert agg.empty
    assert "consensus" in agg.columns


def test_aggregate_single_reviewer_passes_through(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            _label_row("i1", "r1", [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}],
                       "2026-05-01T00:00:00+00:00", tom=1, hen=0, other=1),
        ]
    ).to_csv(labels_dir / "r1.csv", index=False)

    agg = aggregate_reviewers(labels_dir)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["tom_present"] == 1
    assert row["hen_present"] == 0
    assert row["other_birds_present"] == 1
    assert bool(row["consensus"]) is True
    assert row["n_reviewers"] == 1


def test_aggregate_majority_two_of_three(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    base_region = [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}]
    pd.DataFrame([_label_row("i1", "r1", base_region, "2026-05-01T00:00:00+00:00", tom=1, hen=0)]).to_csv(
        labels_dir / "r1.csv", index=False
    )
    pd.DataFrame([_label_row("i1", "r2", base_region, "2026-05-01T00:00:00+00:00", tom=1, hen=0)]).to_csv(
        labels_dir / "r2.csv", index=False
    )
    pd.DataFrame([_label_row("i1", "r3", [], "2026-05-01T00:00:00+00:00", tom=0, hen=0)]).to_csv(
        labels_dir / "r3.csv", index=False
    )

    agg = aggregate_reviewers(labels_dir)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert int(row["tom_present"]) == 1
    assert int(row["n_reviewers"]) == 3
    assert bool(row["consensus"]) is True
    # regions_json must come from one of the two reviewers who saw a Tom call.
    regions = json.loads(row["regions_json"])
    assert any(r["label"] == "Tom" for r in regions)


def test_aggregate_tie_marks_non_consensus(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    region_tom = [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}]
    pd.DataFrame([_label_row("i1", "r1", region_tom, "2026-05-01T00:00:00+00:00", tom=1, hen=0)]).to_csv(
        labels_dir / "r1.csv", index=False
    )
    pd.DataFrame([_label_row("i1", "r2", [], "2026-05-01T00:00:00+00:00", tom=0, hen=0)]).to_csv(
        labels_dir / "r2.csv", index=False
    )

    agg = aggregate_reviewers(labels_dir)
    assert len(agg) == 1
    assert bool(agg.iloc[0]["consensus"]) is False


def test_aggregate_latest_wins_per_reviewer(tmp_path: Path) -> None:
    """If a single reviewer has two snapshots for the same item, the latest one wins
    before the majority vote runs."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    old = _label_row("i1", "r1", [], "2026-05-01T00:00:00+00:00", tom=0, hen=0)
    new = _label_row(
        "i1", "r1",
        [{"start_s": 0.5, "end_s": 1.5, "freq_min_hz": 250, "freq_max_hz": 1500, "label": "Tom"}],
        "2026-05-01T01:00:00+00:00", tom=1, hen=0,
    )
    pd.DataFrame([old, new]).to_csv(labels_dir / "r1.csv", index=False)

    agg = aggregate_reviewers(labels_dir)
    assert int(agg.iloc[0]["tom_present"]) == 1


def test_aggregate_unsure_majority_kills_consensus(tmp_path: Path) -> None:
    """If most reviewers were unsure on a clip, the clip cannot be consensus even if
    the present-attributes agree."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_label_row("i1", "r1", [], "2026-05-01T00:00:00+00:00", tom=0, hen=0, unsure=1)]).to_csv(
        labels_dir / "r1.csv", index=False
    )
    pd.DataFrame([_label_row("i1", "r2", [], "2026-05-01T00:00:00+00:00", tom=0, hen=0, unsure=1)]).to_csv(
        labels_dir / "r2.csv", index=False
    )

    agg = aggregate_reviewers(labels_dir)
    assert bool(agg.iloc[0]["consensus"]) is False
