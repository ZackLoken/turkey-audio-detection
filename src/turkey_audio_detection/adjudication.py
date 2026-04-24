"""Inter-rater agreement utilities."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _load_label_files(labels_dir: Path) -> pd.DataFrame:
    files = sorted(labels_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "reviewer_id" not in df.columns:
            reviewer_id = path.stem
            df["reviewer_id"] = reviewer_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _latest_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "label_timestamp_utc" in df.columns:
        df = df.sort_values("label_timestamp_utc")
    return df.drop_duplicates(subset=["reviewer_id", "item_id"], keep="last")


def compute_pairwise_kappa(labels_df: pd.DataFrame, include_skip: bool = False) -> pd.DataFrame:
    if labels_df.empty:
        return pd.DataFrame(columns=["reviewer_a", "reviewer_b", "n_items", "cohen_kappa"])

    df = _latest_labels(labels_df)
    if not include_skip:
        df = df[df["label"] != "Skip"].copy()

    reviewers = sorted(df["reviewer_id"].dropna().unique().tolist())
    rows: list[dict] = []

    for a, b in combinations(reviewers, 2):
        a_raw = df[df["reviewer_id"] == a][["item_id", "label"]].copy()
        b_raw = df[df["reviewer_id"] == b][["item_id", "label"]].copy()
        a_df = pd.DataFrame({"item_id": a_raw["item_id"], "label_a": a_raw["label"]})
        b_df = pd.DataFrame({"item_id": b_raw["item_id"], "label_b": b_raw["label"]})
        merged = a_df.merge(b_df, on="item_id", how="inner")
        if merged.empty:
            rows.append({"reviewer_a": a, "reviewer_b": b, "n_items": 0, "cohen_kappa": None})
            continue
        kappa = float(cohen_kappa_score(merged["label_a"], merged["label_b"]))
        rows.append({"reviewer_a": a, "reviewer_b": b, "n_items": int(len(merged)), "cohen_kappa": kappa})

    return pd.DataFrame(rows, columns=["reviewer_a", "reviewer_b", "n_items", "cohen_kappa"])


def compute_disagreements(labels_df: pd.DataFrame, include_skip: bool = False) -> pd.DataFrame:
    if labels_df.empty:
        return pd.DataFrame(columns=["item_id", "reviewer_a", "label_a", "reviewer_b", "label_b"])

    df = _latest_labels(labels_df)
    if not include_skip:
        df = df[df["label"] != "Skip"].copy()

    rows: list[dict] = []
    reviewers = sorted(df["reviewer_id"].dropna().unique().tolist())
    for a, b in combinations(reviewers, 2):
        a_raw = df[df["reviewer_id"] == a][["item_id", "label"]].copy()
        b_raw = df[df["reviewer_id"] == b][["item_id", "label"]].copy()
        a_df = pd.DataFrame({"item_id": a_raw["item_id"], "label_a": a_raw["label"]})
        b_df = pd.DataFrame({"item_id": b_raw["item_id"], "label_b": b_raw["label"]})
        merged = a_df.merge(b_df, on="item_id", how="inner")
        mismatches = merged[merged["label_a"] != merged["label_b"]]
        for _, row in mismatches.iterrows():
            rows.append(
                {
                    "item_id": row["item_id"],
                    "reviewer_a": a,
                    "label_a": row["label_a"],
                    "reviewer_b": b,
                    "label_b": row["label_b"],
                }
            )

    return pd.DataFrame(rows, columns=["item_id", "reviewer_a", "label_a", "reviewer_b", "label_b"])


def adjudicate_to_csv(
    labels_dir: Path,
    kappa_out: Path,
    disagreements_out: Path,
    include_skip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_df = _load_label_files(labels_dir)
    kappa_df = compute_pairwise_kappa(labels_df, include_skip=include_skip)
    disagreements_df = compute_disagreements(labels_df, include_skip=include_skip)

    kappa_out.parent.mkdir(parents=True, exist_ok=True)
    disagreements_out.parent.mkdir(parents=True, exist_ok=True)
    kappa_df.to_csv(kappa_out, index=False)
    disagreements_df.to_csv(disagreements_out, index=False)
    return kappa_df, disagreements_df
