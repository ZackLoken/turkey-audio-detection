"""Sliding-window inference for region-level SED.

For each WAV file:
1. Load + resample to PANNs SR (32 kHz).
2. Split into overlapping windows of `window_duration_s` with `window_stride_s` stride.
3. Run the model on each window; aggregate per-pixel probabilities into a
   full-file 2D probability map per class (mel-bin × frame).
4. Threshold → connected components → bounding boxes → events.
5. Filter by `min_event_duration_s`, merge events that adjoin within `merge_gap_s`.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import ndimage

from turkey_audio_detection import __version__ as _PACKAGE_VERSION
from turkey_audio_detection.config import InferConfig
from turkey_audio_detection.dataset import MelParams, parse_regions  # noqa: F401
from turkey_audio_detection.dataset import (
    CLASS_INDEX,
    N_CLASSES,
)
from turkey_audio_detection.ids import _digest
from turkey_audio_detection.layout import inference_dir, model_dir
from turkey_audio_detection.manifest import build_stage_manifest, write_manifest
from turkey_audio_detection.models import CnnSed


INDEX_TO_CLASS = {v: k for k, v in CLASS_INDEX.items()}


def make_event_id(audio_path: str, start_s: float, end_s: float, label: str) -> str:
    return "evt_" + _digest(
        [
            Path(audio_path).as_posix().lower(),
            f"{start_s:.3f}",
            f"{end_s:.3f}",
            label.strip().lower(),
        ]
    )


def _mel_band_centers(p: MelParams) -> np.ndarray:
    return librosa.mel_frequencies(n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax)


def _build_log_mel(y: np.ndarray, p: MelParams) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=p.sample_rate, n_fft=p.n_fft, hop_length=p.hop_length,
        n_mels=p.n_mels, fmin=p.fmin, fmax=p.fmax, power=2.0,
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)


def _window_indices(n_samples: int, window_n: int, stride_n: int) -> list[int]:
    if n_samples <= window_n:
        return [0]
    starts = list(range(0, max(1, n_samples - window_n + 1), stride_n))
    if starts[-1] + window_n < n_samples:
        starts.append(max(0, n_samples - window_n))
    return starts


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> CnnSed:
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = CnnSed(n_classes=N_CLASSES, pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _probmap_for_file(
    audio_path: Path,
    model: CnnSed,
    cfg: InferConfig,
    mel_params: MelParams,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    """Return (prob_map (C, n_mels, total_frames), total_frames, duration_s)."""
    y, sr = librosa.load(str(audio_path), sr=mel_params.sample_rate, mono=True)
    if y.size == 0:
        return np.zeros((N_CLASSES, mel_params.n_mels, 0), dtype=np.float32), 0, 0.0
    duration_s = float(len(y) / mel_params.sample_rate)

    window_n = int(round(cfg.window_duration_s * mel_params.sample_rate))
    stride_n = int(round(cfg.window_stride_s * mel_params.sample_rate))
    starts = _window_indices(len(y), window_n, stride_n)

    total_frames = int(round(len(y) / mel_params.hop_length))
    prob_max = np.zeros((N_CLASSES, mel_params.n_mels, total_frames), dtype=np.float32)

    batch_chunks: list[tuple[int, np.ndarray]] = []
    for s in starts:
        seg = y[s : s + window_n]
        if seg.size < window_n:
            seg = np.concatenate([seg, np.zeros(window_n - seg.size, dtype=seg.dtype)])
        batch_chunks.append((s, _build_log_mel(seg, mel_params)))

    bs = max(1, cfg.batch_size)
    for i in range(0, len(batch_chunks), bs):
        chunk = batch_chunks[i : i + bs]
        mels = np.stack([c[1] for c in chunk])
        tensor = torch.from_numpy(mels).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        for (start_sample, _), p in zip(chunk, probs):
            f0 = int(round(start_sample / mel_params.hop_length))
            n_frames_window = p.shape[-1]
            f1 = min(total_frames, f0 + n_frames_window)
            if f1 <= f0:
                continue
            slice_w = p[:, :, : (f1 - f0)]
            np.maximum(prob_max[:, :, f0:f1], slice_w, out=prob_max[:, :, f0:f1])

    return prob_max, total_frames, duration_s


def _components_to_events(
    prob_class: np.ndarray,
    class_label: str,
    audio_path: Path,
    aru_id: str,
    cfg: InferConfig,
    mel_params: MelParams,
    model_id: str,
    inference_id: str,
) -> list[dict]:
    """Per-class postprocessing: threshold → connected components → events."""
    mask = prob_class > cfg.score_threshold
    if not mask.any():
        return []
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []

    band_centers = _mel_band_centers(mel_params)
    events: list[dict] = []
    for comp_id in range(1, n + 1):
        ys, xs = np.where(labeled == comp_id)
        if xs.size == 0:
            continue
        t0 = int(xs.min())
        t1 = int(xs.max()) + 1
        m0 = int(ys.min())
        m1 = int(ys.max())
        start_s = float(t0 * mel_params.hop_length / mel_params.sample_rate)
        end_s = float(t1 * mel_params.hop_length / mel_params.sample_rate)
        if end_s - start_s < cfg.min_event_duration_s:
            continue
        freq_min_hz = float(band_centers[m0])
        freq_max_hz = float(band_centers[min(m1, len(band_centers) - 1)])
        if freq_max_hz <= freq_min_hz:
            freq_max_hz = freq_min_hz + 1.0
        score = float(prob_class[m0 : m1 + 1, t0:t1].mean())

        events.append(
            {
                "event_id": make_event_id(str(audio_path), start_s, end_s, class_label),
                "source_audio_path": str(audio_path),
                "aru_id": aru_id,
                "start_time_s": round(start_s, 4),
                "end_time_s": round(end_s, 4),
                "freq_min_hz": round(freq_min_hz, 2),
                "freq_max_hz": round(freq_max_hz, 2),
                "label": class_label,
                "score": round(score, 4),
                "model_id": model_id,
                "model_version": _PACKAGE_VERSION,
                "inference_id": inference_id,
            }
        )

    return events


def _merge_adjacent(events: list[dict], merge_gap_s: float) -> list[dict]:
    """Merge same-class events whose time bounds adjoin within `merge_gap_s`."""
    if not events:
        return events
    out: list[dict] = []
    by_label: dict[str, list[dict]] = {}
    for e in events:
        by_label.setdefault(e["label"], []).append(e)
    for label, lst in by_label.items():
        lst.sort(key=lambda e: e["start_time_s"])
        cur = dict(lst[0])
        for e in lst[1:]:
            if e["start_time_s"] - cur["end_time_s"] <= merge_gap_s:
                cur["end_time_s"] = max(cur["end_time_s"], e["end_time_s"])
                cur["freq_min_hz"] = min(cur["freq_min_hz"], e["freq_min_hz"])
                cur["freq_max_hz"] = max(cur["freq_max_hz"], e["freq_max_hz"])
                cur["score"] = max(cur["score"], e["score"])
                cur["event_id"] = make_event_id(
                    cur["source_audio_path"], cur["start_time_s"], cur["end_time_s"], label
                )
            else:
                out.append(cur)
                cur = dict(e)
        out.append(cur)
    return out


def infer_file(
    audio_path: Path,
    model: CnnSed,
    cfg: InferConfig,
    mel_params: MelParams,
    device: torch.device,
    aru_id: str,
    model_id: str,
    inference_id: str,
) -> pd.DataFrame:
    prob, _n_frames, _dur = _probmap_for_file(audio_path, model, cfg, mel_params, device)
    if prob.size == 0:
        return pd.DataFrame()
    events: list[dict] = []
    for c in range(N_CLASSES):
        class_label = INDEX_TO_CLASS[c]
        events.extend(
            _components_to_events(
                prob[c], class_label, audio_path, aru_id, cfg, mel_params, model_id, inference_id
            )
        )
    events = _merge_adjacent(events, cfg.merge_gap_s)
    if not events:
        return pd.DataFrame(
            columns=[
                "event_id", "source_audio_path", "aru_id",
                "start_time_s", "end_time_s",
                "freq_min_hz", "freq_max_hz",
                "label", "score", "model_id", "model_version", "inference_id",
            ]
        )
    return pd.DataFrame(events)


def _infer_aru_id(audio_path: Path) -> str:
    """Walk up the directory tree to find the ARU_* folder name."""
    for parent in audio_path.parents:
        if parent.name.startswith("ARU_"):
            return parent.name
    return ""


def infer(cfg: InferConfig, project_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = model_dir(project_root, cfg.model_id)
    checkpoint = model_root / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model = load_model_from_checkpoint(checkpoint, device)

    audio_files = sorted(project_root.glob(cfg.audio_glob))
    if not audio_files:
        raise RuntimeError(f"No audio files matched glob: {cfg.audio_glob} under {project_root}")

    out_dir = inference_dir(project_root, cfg.inference_id)
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    mel = MelParams()
    summary: list[dict] = []
    for audio_path in audio_files:
        aru_id = _infer_aru_id(audio_path)
        try:
            events_df = infer_file(
                audio_path, model, cfg, mel, device, aru_id,
                model_id=cfg.model_id, inference_id=cfg.inference_id,
            )
        except Exception as exc:
            summary.append({"audio_path": str(audio_path), "n_events": 0, "error": str(exc)})
            continue
        out_csv = events_dir / (audio_path.stem + ".csv")
        events_df.to_csv(out_csv, index=False)
        summary.append({"audio_path": str(audio_path), "n_events": int(len(events_df)), "error": ""})

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

    manifest = build_stage_manifest(
        run_id=cfg.inference_id,
        stage="classify",
        project_root=project_root,
        config_snapshot={"infer": cfg.model_dump(mode="json")},
        stage_outputs={
            "events_dir": str(events_dir),
            "summary_csv": str(out_dir / "summary.csv"),
            "checkpoint": str(checkpoint),
        },
        status="completed",
        input_file_count=len(audio_files),
    )
    write_manifest(out_dir / "inference_manifest.json", manifest)

    return {
        "inference_id": cfg.inference_id,
        "n_files": len(audio_files),
        "n_events_total": int(sum(s["n_events"] for s in summary)),
    }
