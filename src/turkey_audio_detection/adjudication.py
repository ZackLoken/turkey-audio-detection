"""Inter-rater agreement utilities."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.metrics import cohen_kappa_score


PRESENCE_ATTRIBUTES = ("tom_present", "hen_present")


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """Build an empty DataFrame with named columns. Wrapping the list in an
    Index keeps the call argument-type-clean (pandas-stubs typed `columns` as
    `Axes | None`, not `list[str]`)."""
    return pd.DataFrame(columns=pd.Index(columns))


def _load_label_files(labels_dir: Path) -> pd.DataFrame:
    files = sorted(labels_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if "reviewer_id" not in df.columns:
            df["reviewer_id"] = path.stem
        frames.append(df)
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def _latest_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "label_timestamp_utc" in df.columns:
        df = df.sort_values("label_timestamp_utc")
    return df.drop_duplicates(subset=["reviewer_id", "item_id"], keep="last")


def _per_attribute_view(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Return a copy of df with a synthetic `label` column derived from one boolean attribute."""
    out = df.copy()
    out["label"] = out[attribute].fillna(0).astype(int).astype(str)
    return out


def _filter_unsure(df: pd.DataFrame, include_unsure: bool) -> pd.DataFrame:
    if include_unsure or "unsure" not in df.columns:
        return df
    return cast(pd.DataFrame, df[df["unsure"].fillna(0).astype(int) == 0].copy())


def _reviewer_sub(df: pd.DataFrame, reviewer: str, label_alias: str) -> pd.DataFrame:
    """Return (item_id, label_<alias>) rows for one reviewer.

    Using `.loc[mask, cols]` instead of chained `df[mask][cols]` keeps pandas-stubs
    narrowing happy — chained indexing's return type widens to ndarray-ish.
    """
    sub = df.loc[df["reviewer_id"] == reviewer, ["item_id", "label"]]
    return sub.rename(columns={"label": f"label_{label_alias}"})


def compute_pairwise_kappa(labels_df: pd.DataFrame, include_unsure: bool = False) -> pd.DataFrame:
    columns = ["attribute", "reviewer_a", "reviewer_b", "n_items", "cohen_kappa"]
    if labels_df.empty:
        return _empty_frame(columns)

    base = _filter_unsure(_latest_labels(labels_df), include_unsure)
    rows: list[dict] = []

    for attribute in PRESENCE_ATTRIBUTES:
        if attribute not in base.columns:
            continue
        df = _per_attribute_view(base, attribute)
        reviewers = sorted(df["reviewer_id"].dropna().unique().tolist())

        for a, b in combinations(reviewers, 2):
            merged = _reviewer_sub(df, a, "a").merge(
                _reviewer_sub(df, b, "b"), on="item_id", how="inner"
            )
            if merged.empty:
                rows.append(
                    {
                        "attribute": attribute,
                        "reviewer_a": a,
                        "reviewer_b": b,
                        "n_items": 0,
                        "cohen_kappa": None,
                    }
                )
                continue
            kappa = float(cohen_kappa_score(merged["label_a"], merged["label_b"]))
            rows.append(
                {
                    "attribute": attribute,
                    "reviewer_a": a,
                    "reviewer_b": b,
                    "n_items": int(len(merged)),
                    "cohen_kappa": kappa,
                }
            )

    return pd.DataFrame(rows, columns=pd.Index(columns))


def compute_disagreements(labels_df: pd.DataFrame, include_unsure: bool = False) -> pd.DataFrame:
    columns = ["attribute", "item_id", "reviewer_a", "label_a", "reviewer_b", "label_b"]
    if labels_df.empty:
        return _empty_frame(columns)

    base = _filter_unsure(_latest_labels(labels_df), include_unsure)
    rows: list[dict] = []

    for attribute in PRESENCE_ATTRIBUTES:
        if attribute not in base.columns:
            continue
        df = _per_attribute_view(base, attribute)
        reviewers = sorted(df["reviewer_id"].dropna().unique().tolist())

        for a, b in combinations(reviewers, 2):
            merged = _reviewer_sub(df, a, "a").merge(
                _reviewer_sub(df, b, "b"), on="item_id", how="inner"
            )
            mismatches = merged[merged["label_a"] != merged["label_b"]]
            for _, row in mismatches.iterrows():
                rows.append(
                    {
                        "attribute": attribute,
                        "item_id": row["item_id"],
                        "reviewer_a": a,
                        "label_a": row["label_a"],
                        "reviewer_b": b,
                        "label_b": row["label_b"],
                    }
                )

    return pd.DataFrame(rows, columns=pd.Index(columns))


def adjudicate_to_csv(
    labels_dir: Path,
    kappa_out: Path,
    disagreements_out: Path,
    include_unsure: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_df = _load_label_files(labels_dir)
    kappa_df = compute_pairwise_kappa(labels_df, include_unsure=include_unsure)
    disagreements_df = compute_disagreements(labels_df, include_unsure=include_unsure)

    kappa_out.parent.mkdir(parents=True, exist_ok=True)
    disagreements_out.parent.mkdir(parents=True, exist_ok=True)
    kappa_df.to_csv(kappa_out, index=False)
    disagreements_df.to_csv(disagreements_out, index=False)
    return kappa_df, disagreements_df
