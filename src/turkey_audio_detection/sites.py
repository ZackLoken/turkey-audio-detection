"""ARU -> site mapping for site-grouped train/val/test splits.

Multiple ARUs can share a physical site, so splits must group by site (not ARU)
to avoid leakage. The owner populates `data/site_map.csv` (columns: aru_id,site_id).
Unmapped ARUs fall back to site_id == aru_id (each ARU its own site) with a warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def load_site_map(path: str | Path) -> dict[str, str]:
    """Read site_map.csv -> {aru_id: site_id}. Missing file -> empty map."""
    p = Path(path)
    if not p.exists():
        warnings.warn(f"site_map not found at {p}; falling back to site_id == aru_id.")
        return {}
    df = pd.read_csv(p, dtype=str)
    if "aru_id" not in df.columns or "site_id" not in df.columns:
        raise ValueError(f"{p} must have columns 'aru_id' and 'site_id'")
    return {
        str(a).strip(): str(s).strip()
        for a, s in zip(df["aru_id"], df["site_id"])
        if str(a).strip() and str(s).strip()
    }


def attach_site(table: pd.DataFrame, site_map: dict[str, str]) -> pd.DataFrame:
    """Add a `site_id` column from the map; unmapped ARUs fall back to their aru_id."""
    out = table.copy()
    aru = out["aru_id"].astype(str) if "aru_id" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["site_id"] = aru.map(lambda a: site_map.get(a, a))
    unmapped = sorted(set(aru) - set(site_map)) if site_map else sorted(set(aru))
    if unmapped:
        warnings.warn(
            f"{len(unmapped)} ARU(s) not in site_map; using aru_id as site_id: {unmapped[:5]}"
        )
    return out
