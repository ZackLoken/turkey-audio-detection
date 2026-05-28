"""No-code inference app (reach goal): run a trained SED model on audio.

Provides a Streamlit UI for non-coders (select a trained model, drop in a WAV, get
per-call events + per-sex counts) plus a testable `analyze_recording_file` wrapper.
Streamlit and birdnetlib are imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from turkey_audio_detection.config import SedInferConfig
from turkey_audio_detection.layout import model_dir
from turkey_audio_detection.sed_data import LogMelExtractor, SedMelParams
from turkey_audio_detection.sed_inference import aggregate_counts, infer_recording, load_sed_model
from turkey_audio_detection.sites import load_site_map


def analyze_recording_file(
    model_id: str,
    audio_path: str | Path,
    windows: pd.DataFrame,
    project_root: str | Path,
    cfg: SedInferConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a trained model over one recording's candidate windows -> (events, counts)."""
    cfg = cfg or SedInferConfig(model_id=model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_sed_model(model_dir(Path(project_root), model_id) / "checkpoint.pt", device)
    mel = SedMelParams()
    extractor = LogMelExtractor(mel)
    events = infer_recording(
        Path(audio_path), windows, model, mel, extractor, payload, cfg, device, cfg.inference_id or "app"
    )
    counts = aggregate_counts(events, load_site_map(Path(project_root) / cfg.site_map_path))
    return events, counts


def birdnet_windows(audio_path: str | Path, min_confidence: float = 0.1, species: str = "Wild Turkey") -> pd.DataFrame:
    """Run BirdNET on a file and return its candidate windows (start/end seconds)."""
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer

    rec = Recording(Analyzer(), str(audio_path), min_conf=min_confidence)
    rec.analyze()
    rows = [
        {"start_time_s": float(d.get("start_time", 0.0)), "end_time_s": float(d.get("end_time", 0.0))}
        for d in rec.detections
        if species.lower() in str(d.get("common_name", "")).lower()
    ]
    return pd.DataFrame(rows, columns=pd.Index(["start_time_s", "end_time_s"]))


def _list_models(project_root: Path) -> list[str]:
    from turkey_audio_detection.layout import models_root

    root = models_root(project_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "checkpoint.pt").exists())


def main() -> None:  # pragma: no cover - Streamlit UI
    import tempfile

    import streamlit as st

    st.set_page_config(page_title="Turkey SED inference", layout="wide")
    st.title("Turkey call detector")
    project_root = Path(st.text_input("Project root", value=str(Path.cwd())))

    models = _list_models(project_root)
    if not models:
        st.warning("No trained models found under data/_outputs/models.")
        return
    model_id = st.selectbox("Model", models)
    uploaded = st.file_uploader("Audio file (WAV)", type=["wav"])
    min_conf = st.slider("BirdNET candidate confidence", 0.0, 1.0, 0.1, 0.05)

    if uploaded is not None and st.button("Analyze"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            audio_path = tmp.name
        with st.spinner("Finding candidate calls with BirdNET..."):
            windows = birdnet_windows(audio_path, min_confidence=min_conf)
        if windows.empty:
            st.info("No Wild Turkey candidates found.")
            return
        with st.spinner("Classifying calls..."):
            events, counts = analyze_recording_file(model_id, audio_path, windows, project_root)
        st.subheader(f"{len(events)} call events")
        st.dataframe(events)
        st.download_button("Download events CSV", events.to_csv(index=False), "events.csv", "text/csv")
        st.subheader("Counts per site/day/sex")
        st.dataframe(counts)


if __name__ == "__main__":  # pragma: no cover
    main()
