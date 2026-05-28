"""Event-level + segment-level evaluation for frame SED.

Event metric: per-class precision/recall/F1 with greedy time-axis IoU matching of
predicted vs reviewer-box events (1D analogue of yolo-annotator's compute_matches).
Segment metric: threshold-free-ish F1 over fixed time bins. Reported across an IoU
sweep so the operating point isn't hidden.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from turkey_audio_detection.dataset import CLASS_INDEX, N_CLASSES, parse_regions
from turkey_audio_detection.sed_data import LogMelExtractor, SedMelParams, load_waveform, normalize_log_mel
from turkey_audio_detection.sed_inference import frames_to_events

_INDEX_TO_CLASS = {v: k for k, v in CLASS_INDEX.items()}
Event = tuple[float, float]


def time_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def match_events(gt: list[Event], pred: list[Event], iou_threshold: float = 0.3) -> dict:
    """Greedy descending-IoU matching -> {tp, fp, fn}. One GT matches one prediction."""
    candidates: list[tuple[float, int, int]] = []
    for gi, (g0, g1) in enumerate(gt):
        for pi, (p0, p1) in enumerate(pred):
            iou = time_iou(g0, g1, p0, p1)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    matched_g: set[int] = set()
    matched_p: set[int] = set()
    for _iou, gi, pi in candidates:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
    tp = len(matched_g)
    return {"tp": tp, "fp": len(pred) - len(matched_p), "fn": len(gt) - tp}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def segment_f1(gt: list[Event], pred: list[Event], duration_s: float, seg_s: float = 1.0) -> float:
    """F1 over fixed time bins: a bin is positive if any event overlaps it."""
    n = max(1, int(math.ceil(duration_s / seg_s)))
    gt_seg = np.zeros(n, dtype=bool)
    pred_seg = np.zeros(n, dtype=bool)
    for arr, events in ((gt_seg, gt), (pred_seg, pred)):
        for s, e in events:
            i0 = max(0, int(s // seg_s))
            i1 = min(n, int(math.ceil(e / seg_s)))
            arr[i0:i1] = True
    tp = int(np.sum(gt_seg & pred_seg))
    fp = int(np.sum(~gt_seg & pred_seg))
    fn = int(np.sum(gt_seg & ~pred_seg))
    return prf(tp, fp, fn)[2]


def regions_to_events(regions: list[dict], label: str) -> list[Event]:
    """Reviewer boxes of a given class -> [(start_s, end_s)] (time-only)."""
    out: list[Event] = []
    for r in regions:
        if str(r.get("label", "")) != label:
            continue
        s = float(r.get("start_s", 0.0))
        e = float(r.get("end_s", 0.0))
        if e > s:
            out.append((s, e))
    return out


def evaluate_table(
    model,
    table: pd.DataFrame,
    payload: dict,
    device: torch.device,
    mel: SedMelParams = SedMelParams(),
    iou_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5),
    seg_s: float = 1.0,
    thresholds: dict[str, float] | None = None,
    min_event_duration_s: float = 0.1,
    merge_gap_s: float = 0.2,
    clip_duration_s: float = 3.0,
) -> dict:
    """Run the model over a labeled table; return event metrics (IoU sweep) + segment F1."""
    extractor = LogMelExtractor(mel)
    hop_s = (mel.hop_length * int(payload.get("time_downsample", 8))) / mel.sample_rate
    thr = thresholds or payload.get("thresholds", {}) or {}

    # accumulate per-class events for event metrics, and segment tallies
    counts = {iou: {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(N_CLASSES)} for iou in iou_thresholds}
    seg_scores = {c: [] for c in range(N_CLASSES)}

    model.eval()
    for _, row in table.iterrows():
        y = load_waveform(str(row["clip_path"]), clip_duration_s, mel.sample_rate)
        log_mel = normalize_log_mel(extractor(torch.from_numpy(y)).numpy(), mel)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(log_mel)[None].to(device)))[0].cpu().numpy()
        regions = parse_regions(row.get("regions_json", ""))
        for c in range(N_CLASSES):
            label = _INDEX_TO_CLASS[c]
            gt = regions_to_events(regions, label)
            pred = [(e["start_s"], e["end_s"]) for e in frames_to_events(probs[c], float(thr.get(label, 0.5)), min_event_duration_s, merge_gap_s, hop_s)]
            for iou in iou_thresholds:
                m = match_events(gt, pred, iou)
                for k in ("tp", "fp", "fn"):
                    counts[iou][c][k] += m[k]
            seg_scores[c].append(segment_f1(gt, pred, clip_duration_s, seg_s))

    result: dict = {"event": {}, "segment_f1": {}}
    for iou in iou_thresholds:
        result["event"][iou] = {}
        for c in range(N_CLASSES):
            p, r, f = prf(**counts[iou][c])
            result["event"][iou][_INDEX_TO_CLASS[c]] = {"precision": p, "recall": r, "f1": f}
    for c in range(N_CLASSES):
        vals = seg_scores[c]
        result["segment_f1"][_INDEX_TO_CLASS[c]] = float(np.mean(vals)) if vals else 0.0
    return result


def evaluation_to_rows(result: dict) -> pd.DataFrame:
    """Flatten the nested metrics dict into tidy rows for CSV."""
    rows: list[dict] = []
    for iou, by_class in result.get("event", {}).items():
        for cls, m in by_class.items():
            rows.append({"metric": "event", "iou_threshold": iou, "class": cls, **m})
    for cls, f1 in result.get("segment_f1", {}).items():
        rows.append({"metric": "segment_f1", "iou_threshold": None, "class": cls, "f1": f1})
    return pd.DataFrame(rows)
