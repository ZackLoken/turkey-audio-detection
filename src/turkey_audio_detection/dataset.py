"""Shared label taxonomy + region-JSON parsing for the SED pipeline.

Classes are sex only (Tom/Hen). Region geometry and mel/target construction live
in `sed_data.py`; this module holds the small pieces reused across stages.
"""

from __future__ import annotations

import json

import numpy as np

CLASS_INDEX = {"Tom": 0, "Hen": 1}
N_CLASSES = len(CLASS_INDEX)


def parse_regions(regions_json: object) -> list[dict]:
    """Parse a regions_json cell into a list of region dicts (tolerant of NaN/blank)."""
    if regions_json is None:
        return []
    if isinstance(regions_json, float) and np.isnan(regions_json):
        return []
    text = str(regions_json).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [r for r in parsed if isinstance(r, dict)]
