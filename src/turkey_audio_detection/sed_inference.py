"""Whole-recording frame-level SED inference.

The trained model runs directly over each recording with overlapping sliding
windows. BirdNET is NOT used here — it is only a labeling aid for proposing review
candidates; gating inference on it would cap the detector at BirdNET's recall.

Per-window per-frame probabilities are averaged across overlaps onto a recording
timeline, thresholded per class, grouped into events (start_s, end_s, sex, score),
and aggregated to presence + counts per site/day.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf  # noqa: F401  (kept for parity with other stages / future use)
import torch

from turkey_audio_detection.config import SedInferConfig
from turkey_audio_detection.dataset import CLASS_INDEX, N_CLASSES
from turkey_audio_detection.ids import _digest
from turkey_audio_detection.layout import inference_dir, model_dir
from turkey_audio_detection.manifest import build_stage_manifest, write_manifest
from turkey_audio_detection.models.frame_sed import FrameSed
from turkey_audio_detection.sed_data import LogMelExtractor, SedMelParams, normalize_log_mel
from turkey_audio_detection.sites import attach_site, load_site_map

_INDEX_TO_CLASS = {v: k for k, v in CLASS_INDEX.items()}


def make_event_id(audio_path: str, start_s: float, end_s: float, label: str) -> str:
    return "evt_" + _digest(
        [Path(audio_path).as_posix().lower(), f"{start_s:.3f}", f"{end_s:.3f}", label.strip().lower()]
    )


def _infer_aru_id(audio_path: Path) -> str:
    for parent in audio_path.parents:
        if parent.name.startswith("ARU_"):
            return parent.name
    return ""


def load_sed_model(checkpoint_path: Path, device: torch.device) -> tuple[FrameSed, dict]:
    """Rebuild FrameSed from a training checkpoint and load weights."""
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = payload["config"]
    model = FrameSed(
        n_classes=N_CLASSES,
        n_stages=int(cfg.get("n_stages", 2)),
        temporal=str(cfg.get("temporal", "bigru")),
        hidden_size=int(cfg.get("hidden_size", 256)),
        n_layers=int(cfg.get("n_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        pretrained=False,  # weights come from the checkpoint, not the hub
        config_dict=payload.get("backbone_config"),  # rebuild the exact trained architecture
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def frames_to_events(
    prob: np.ndarray,
    threshold: float,
    min_duration_s: float,
    merge_gap_s: float,
    hop_s: float,
    offset_s: float = 0.0,
) -> list[dict]:
    """Group a 1-D per-frame probability vector into events.

    Threshold -> runs of consecutive positive frames -> merge runs within
    `merge_gap_s` -> drop runs shorter than `min_duration_s`. Score = mean prob.
    """
    above = prob >= threshold
    idx = np.where(above)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    runs = [(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, splits + 1)]

    gap_frames = merge_gap_s / hop_s if hop_s > 0 else 0.0
    merged: list[list[int]] = []
    for t0, t1 in runs:
        if merged and (t0 - merged[-1][1]) <= gap_frames:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1])

    events: list[dict] = []
    for t0, t1 in merged:
        start_s = offset_s + t0 * hop_s
        end_s = offset_s + t1 * hop_s
        if (end_s - start_s) < min_duration_s:
            continue
        events.append({"start_s": round(start_s, 4), "end_s": round(end_s, 4), "score": round(float(prob[t0:t1].mean()), 4)})
    return events


def stitch_windows(window_results: list[tuple[int, np.ndarray]], total_frames: int) -> np.ndarray:
    """Average overlapping per-frame probs onto a (N_CLASSES, total_frames) timeline.

    window_results: list of (frame_offset, probs[N_CLASSES, t_window]).
    """
    acc = np.zeros((N_CLASSES, total_frames), dtype=np.float32)
    cnt = np.zeros((N_CLASSES, total_frames), dtype=np.float32)
    for offset, probs in window_results:
        t = probs.shape[1]
        f1 = min(total_frames, offset + t)
        if f1 <= offset:
            continue
        acc[:, offset:f1] += probs[:, : f1 - offset]
        cnt[:, offset:f1] += 1.0
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def _window_starts(n_samples: int, win_n: int, stride_n: int) -> list[int]:
    """Sample start indices tiling the whole signal, with a final flush-right window."""
    if n_samples <= win_n:
        return [0]
    starts = list(range(0, n_samples - win_n + 1, stride_n))
    if starts[-1] + win_n < n_samples:
        starts.append(n_samples - win_n)
    return starts


def infer_recording(
    audio_path: Path,
    model: FrameSed,
    mel: SedMelParams,
    extractor: LogMelExtractor,
    payload: dict,
    cfg: SedInferConfig,
    device: torch.device,
    inference_id: str,
) -> pd.DataFrame:
    """Slide the model over a full recording and return its events."""
    hop_s = (mel.hop_length * int(payload.get("time_downsample", 8))) / mel.sample_rate
    thresholds = cfg.thresholds or payload.get("thresholds", {}) or {}
    win_n = int(round(cfg.window_duration_s * mel.sample_rate))
    stride_n = max(1, int(round(cfg.window_stride_s * mel.sample_rate)))

    try:
        y, _ = librosa.load(str(audio_path), sr=mel.sample_rate, mono=True)
    except Exception:
        return pd.DataFrame()
    if y.size == 0:
        return pd.DataFrame()

    total_frames = max(1, int(round((y.size / mel.sample_rate) / hop_s)))
    starts = _window_starts(y.size, win_n, stride_n)

    mels: list[np.ndarray] = []
    offsets: list[int] = []
    for s in starts:
        seg = y[s : s + win_n]
        if seg.size < win_n:
            seg = np.concatenate([seg, np.zeros(win_n - seg.size, dtype=seg.dtype)])
        log_mel = normalize_log_mel(extractor(torch.from_numpy(seg.astype(np.float32))).numpy(), mel)
        mels.append(log_mel)
        offsets.append(int(round((s / mel.sample_rate) / hop_s)))

    window_results: list[tuple[int, np.ndarray]] = []
    bs = max(1, cfg.batch_size)
    for i in range(0, len(mels), bs):
        batch = np.stack(mels[i : i + bs])
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(batch).to(device))).cpu().numpy()  # (b, C, T')
        for j in range(probs.shape[0]):
            window_results.append((offsets[i + j], probs[j]))

    timeline = stitch_windows(window_results, total_frames)
    aru_id = _infer_aru_id(audio_path)
    rows: list[dict] = []
    for c in range(N_CLASSES):
        label = _INDEX_TO_CLASS[c]
        thr = float(thresholds.get(label, 0.5))
        for ev in frames_to_events(timeline[c], thr, cfg.min_event_duration_s, cfg.merge_gap_s, hop_s):
            rows.append({
                "event_id": make_event_id(str(audio_path), ev["start_s"], ev["end_s"], label),
                "source_audio_path": str(audio_path),
                "aru_id": aru_id,
                "start_time_s": ev["start_s"],
                "end_time_s": ev["end_s"],
                "sex": label,
                "score": ev["score"],
                "model_id": cfg.model_id,
                "inference_id": inference_id,
            })
    return pd.DataFrame(rows)


def _date_from_path(path: str) -> str:
    import re

    m = re.search(r"_(\d{8})_\d{6}\.wav$", str(path), re.IGNORECASE)
    if not m:
        return ""
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def aggregate_counts(events: pd.DataFrame, site_map: dict[str, str]) -> pd.DataFrame:
    """Events -> presence + counts per (site, date, sex)."""
    if events.empty:
        return pd.DataFrame(columns=pd.Index(["site_id", "date", "sex", "n_events", "present"]))
    df = attach_site(events, site_map)
    df["date"] = pd.to_datetime(df["source_audio_path"].map(_date_from_path), errors="coerce").dt.date.astype("string")
    grouped = df.groupby(["site_id", "date", "sex"]).size().reset_index(name="n_events")
    grouped["present"] = 1
    return grouped


def infer_sed(cfg: SedInferConfig, project_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = model_dir(Path(project_root), cfg.model_id) / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model, payload = load_sed_model(checkpoint, device)
    mel = SedMelParams()
    extractor = LogMelExtractor(mel)

    audio_files = sorted(Path(project_root).glob(cfg.audio_glob))
    if not audio_files:
        raise RuntimeError(f"No audio files matched glob: {cfg.audio_glob} under {project_root}")

    out_dir = inference_dir(Path(project_root), cfg.inference_id)
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[pd.DataFrame] = []
    summary: list[dict] = []
    for audio_path in audio_files:
        ev = infer_recording(audio_path, model, mel, extractor, payload, cfg, device, cfg.inference_id)
        if not ev.empty:
            ev.to_csv(events_dir / (audio_path.stem + ".csv"), index=False)
            all_events.append(ev)
        summary.append({"audio_path": str(audio_path), "n_events": int(len(ev))})

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    aggregate_counts(events_df, load_site_map(Path(project_root) / cfg.site_map_path)).to_csv(
        out_dir / "aggregate_counts.csv", index=False
    )
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

    manifest = build_stage_manifest(
        run_id=cfg.inference_id, stage="classify", project_root=Path(project_root),
        config_snapshot={"sed_infer": cfg.model_dump(mode="json")},
        stage_outputs={"events_dir": str(events_dir), "summary_csv": str(out_dir / "summary.csv")},
        status="completed", input_file_count=len(audio_files),
    )
    write_manifest(out_dir / "inference_manifest.json", manifest)
    return {
        "inference_id": cfg.inference_id,
        "n_files": len(audio_files),
        "n_events_total": int(sum(s["n_events"] for s in summary)),
    }
