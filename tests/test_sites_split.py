"""Tests for ARU->site mapping and site/year-grouped splits."""

from __future__ import annotations

import pandas as pd

from turkey_audio_detection.config import SedTrainConfig
from turkey_audio_detection.sed_training import site_split
from turkey_audio_detection.sites import attach_site, load_site_map


def _table(arus, year="2026"):
    return pd.DataFrame(
        [
            {
                "item_id": f"it{i}",
                "aru_id": a,
                "recording_datetime": f"{year}-04-01 06:00:00",
                "tom_present": 0,
                "hen_present": 0,
            }
            for i, a in enumerate(arus)
        ]
    )


def test_load_site_map(tmp_path) -> None:
    p = tmp_path / "site_map.csv"
    pd.DataFrame({"aru_id": ["ARU_01", "ARU_02"], "site_id": ["S1", "S1"]}).to_csv(p, index=False)
    assert load_site_map(p) == {"ARU_01": "S1", "ARU_02": "S1"}


def test_load_site_map_missing_returns_empty(tmp_path) -> None:
    assert load_site_map(tmp_path / "nope.csv") == {}


def test_attach_site_fallback_to_aru() -> None:
    out = attach_site(_table(["ARU_01", "ARU_99"]), {"ARU_01": "S1"})
    assert out.loc[out.aru_id == "ARU_01", "site_id"].iloc[0] == "S1"
    assert out.loc[out.aru_id == "ARU_99", "site_id"].iloc[0] == "ARU_99"


def test_site_split_no_site_in_two_splits() -> None:
    arus = ["ARU_01", "ARU_02", "ARU_03", "ARU_04", "ARU_05", "ARU_06"]
    smap = {"ARU_01": "S1", "ARU_02": "S1", "ARU_03": "S2", "ARU_04": "S2", "ARU_05": "S3", "ARU_06": "S3"}
    table = attach_site(_table(arus), smap)
    cfg = SedTrainConfig(val_fraction=0.34, test_fraction=0.34, seed=0)
    split = site_split(table, cfg)
    # every site lands in exactly one split
    assert (split.groupby("site_id")["split"].nunique() == 1).all()
    assert set(split["split"]).issubset({"train", "val", "test"})
