"""Multi-reviewer label aggregation + training-table construction.

Reads per-reviewer label CSVs produced by the v0.2.0 review app, resolves the latest
snapshot per (reviewer_id, item_id), takes a majority vote across reviewers per
attribute (tom_present, hen_present), and joins the result against a run's review
queue to emit a per-clip training table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from turkey_audio_detection.adjudication import _latest_labels, _load_label_files
from turkey_audio_detection.layout import RunLayout


PRESENCE_ATTRIBUTES = ("tom_present", "hen_present")


def _majority_vote(values: pd.Series) -> tuple[int, bool]:
    """Return (majority_value, is_consensus). Ties resolve to 0 with consensus=False."""
    ints = values.fillna(0).astype(int)
    n = len(ints)
    if n == 0:
        return 0, False
    pos = int((ints == 1).sum())
    neg = n - pos
    if pos > neg:
        return 1, True
    if neg > pos:
        return 0, True
    return 0, False  # tie


def aggregate_reviewers(labels_dir: Path) -> pd.DataFrame:
    """Aggregate per-reviewer latest snapshots into one row per item_id.

    Output columns:
      item_id, n_reviewers, tom_present, hen_present, other_birds_present,
      consensus, regions_json
    The `regions_json` column is taken from the highest-`tom_present + hen_present`
    snapshot (i.e., the most permissive reviewer's regions); ties broken by the
    most recent label_timestamp_utc. This preserves the region geometry for SED
    training without having to merge region polygons across disagreeing reviewers.
    """
    raw = _load_label_files(labels_dir)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "n_reviewers",
                "tom_present",
                "hen_present",
                "other_birds_present",
                "consensus",
                "regions_json",
            ]
        )

    latest = _latest_labels(raw)

    rows: list[dict] = []
    for item_id, group in latest.groupby("item_id"):
        n = int(len(group))
        tom, tom_cons = _majority_vote(group["tom_present"])
        hen, hen_cons = _majority_vote(group["hen_present"])
        other, _ = _majority_vote(group["other_birds_present"])
        # any_unsure votes also factor: if more than half of reviewers were unsure,
        # mark the clip non-consensus regardless of attribute agreement.
        unsure_majority = int((group["unsure"].fillna(0).astype(int) == 1).sum() * 2) > n
        consensus = tom_cons and hen_cons and not unsure_majority

        # Pick the "most informative" snapshot for regions: prefer rows whose
        # (tom_present + hen_present) sum matches the aggregated booleans, then
        # most-recent timestamp. This avoids picking a region list from a reviewer
        # who disagreed about presence.
        target_sum = int(tom + hen)
        candidates = group[
            (group["tom_present"].fillna(0).astype(int) + group["hen_present"].fillna(0).astype(int))
            == target_sum
        ]
        if candidates.empty:
            candidates = group
        candidates = candidates.sort_values("label_timestamp_utc")
        regions_json = str(candidates.iloc[-1].get("regions_json", "[]")) or "[]"

        rows.append(
            {
                "item_id": str(item_id),
                "n_reviewers": n,
                "tom_present": int(tom),
                "hen_present": int(hen),
                "other_birds_present": int(other),
                "consensus": bool(consensus),
                "regions_json": regions_json,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "item_id",
            "n_reviewers",
            "tom_present",
            "hen_present",
            "other_birds_present",
            "consensus",
            "regions_json",
        ],
    )


def build_training_table(
    project_root: Path,
    run_ids: list[str],
    include_non_consensus: bool = False,
) -> pd.DataFrame:
    """Join one or more runs' review queues with the aggregated labels.

    Returns a DataFrame with columns suitable for the dataset loader:
      item_id, clip_path, source_audio_path, aru_id, recording_datetime,
      tom_present, hen_present, other_birds_present, consensus, n_reviewers,
      regions_json
    """
    if not run_ids:
        raise ValueError("build_training_table requires at least one run_id")

    queue_frames: list[pd.DataFrame] = []
    for run_id in run_ids:
        layout = RunLayout.from_project_root(project_root, run_id)
        queue_path = layout.queue_dir / "review_queue.csv"
        if not queue_path.exists():
            raise FileNotFoundError(f"Missing review queue for {run_id}: {queue_path}")
        q = pd.read_csv(queue_path)
        q["__run_id"] = run_id
        queue_frames.append(q)
    queue_df = pd.concat(queue_frames, ignore_index=True)

    # Labels live under data/_outputs/review/labels/ — shared across runs.
    labels_dir = RunLayout.from_project_root(project_root, run_ids[0]).review_labels_dir
    agg = aggregate_reviewers(labels_dir)

    merged = queue_df.merge(agg, on="item_id", how="inner")
    if not include_non_consensus:
        merged = merged[merged["consensus"].astype(bool)].copy()

    keep_cols = [
        "item_id",
        "clip_path",
        "source_audio_path",
        "aru_id",
        "recording_datetime",
        "tom_present",
        "hen_present",
        "other_birds_present",
        "consensus",
        "n_reviewers",
        "regions_json",
    ]
    available = [c for c in keep_cols if c in merged.columns]
    return merged[available].reset_index(drop=True)
