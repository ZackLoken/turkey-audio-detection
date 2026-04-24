"""Streamlit review application for queue labeling."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from turkey_audio_detection.schemas import VALID_LABELS


def _default_project_root() -> Path:
    return Path.cwd()


def _runs_dir(project_root: Path) -> Path:
    return project_root / "data" / "_outputs" / "runs"


def _labels_dir(project_root: Path) -> Path:
    return project_root / "data" / "_outputs" / "review" / "labels"


def _queue_path(project_root: Path, run_id: str) -> Path:
    return _runs_dir(project_root) / run_id / "queue" / "review_queue.csv"


def _load_queue(project_root: Path, run_id: str) -> pd.DataFrame:
    path = _queue_path(project_root, run_id)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "queue_order" in df.columns:
        df = df.sort_values("queue_order").reset_index(drop=True)
    return df


def _labels_path(project_root: Path, reviewer_id: str) -> Path:
    out = _labels_dir(project_root)
    out.mkdir(parents=True, exist_ok=True)
    safe_reviewer = reviewer_id.strip().replace(" ", "_")
    return out / f"{safe_reviewer}.csv"


def _load_existing_labels(project_root: Path, reviewer_id: str) -> pd.DataFrame:
    path = _labels_path(project_root, reviewer_id)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _append_label_row(project_root: Path, row: dict) -> None:
    path = _labels_path(project_root, row["reviewer_id"])
    write_header = not path.exists()
    columns = ["item_id", "detection_id", "reviewer_id", "reviewer_name",
               "label", "label_timestamp_utc", "session_id", "app_version"]
    with path.open("a", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


def _latest_by_item(labels_df: pd.DataFrame) -> pd.DataFrame:
    if labels_df.empty:
        return labels_df
    if "label_timestamp_utc" in labels_df.columns:
        labels_df = labels_df.sort_values("label_timestamp_utc")
    return labels_df.drop_duplicates(subset=["item_id"], keep="last")


def _current_queue_index(queue_df: pd.DataFrame, labels_df: pd.DataFrame) -> int:
    if queue_df.empty:
        return 0
    if labels_df.empty or "item_id" not in labels_df.columns:
        return 0

    latest = _latest_by_item(labels_df)
    done_ids = set(latest["item_id"].astype(str).tolist())
    for idx, row in queue_df.iterrows():
        if str(row["item_id"]) not in done_ids:
            return idx
    return len(queue_df) - 1


def _render_spectrogram(audio_path: Path) -> None:
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        st.warning(f"Unable to load spectrogram: {exc}")
        return

    if y.size == 0:
        st.warning("Audio file was empty.")
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    db = librosa.power_to_db(melspec, ref=max(1e-6, melspec.max()))
    librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _launch_via_streamlit() -> None:
    """Re-launch this module via `streamlit run` when called as a console script."""
    import subprocess
    import sys

    script = Path(__file__).resolve()
    args = ["streamlit", "run", str(script), "--server.headless", "false"] + sys.argv[1:]
    raise SystemExit(subprocess.call(args))


def main() -> None:
    """Entry point: re-launches via streamlit when called as a script, runs app otherwise."""
    try:
        # If streamlit context is already active (i.e. we were invoked by streamlit run),
        # _get_script_run_ctx will return a non-None value.
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            _launch_via_streamlit()
    except ImportError:
        _launch_via_streamlit()

    st.set_page_config(page_title="Turkey Review", layout="wide")
    st.title("Turkey Audio Review")

    with st.sidebar:
        st.header("Session")
        project_root_str = st.text_input("Project root", value=str(_default_project_root()))
        project_root = Path(project_root_str)

        run_candidates: list[str] = []
        runs_path = _runs_dir(project_root)
        if runs_path.exists():
            run_candidates = sorted([p.name for p in runs_path.iterdir() if p.is_dir()], reverse=True)

        if run_candidates:
            selected_run = st.selectbox("Run ID", options=run_candidates)
        else:
            selected_run = st.text_input("Run ID", value="")

        reviewer_id = st.text_input("Reviewer ID", value=st.session_state.get("reviewer_id", ""))
        reviewer_name = st.text_input("Reviewer Name (optional)", value=st.session_state.get("reviewer_name", ""))

        if st.button("Start / Refresh Session", type="primary"):
            st.session_state["reviewer_id"] = reviewer_id.strip()
            st.session_state["reviewer_name"] = reviewer_name.strip()
            st.session_state["run_id"] = selected_run.strip()
            st.session_state["session_id"] = st.session_state.get("session_id") or str(uuid.uuid4())
            st.rerun()

    reviewer_id = st.session_state.get("reviewer_id", "").strip()
    run_id = st.session_state.get("run_id", "").strip()
    reviewer_name = st.session_state.get("reviewer_name", "").strip()

    if not reviewer_id:
        st.info("Enter reviewer identity in the sidebar and click Start / Refresh Session.")
        return
    if not run_id:
        st.info("Select or enter a run ID in the sidebar and click Start / Refresh Session.")
        return

    queue_df = _load_queue(project_root, run_id)
    if queue_df.empty:
        st.warning("Queue file not found or empty for selected run.")
        st.caption(f"Expected: {_queue_path(project_root, run_id)}")
        return

    labels_df = _load_existing_labels(project_root, reviewer_id)
    if "cursor" not in st.session_state:
        st.session_state["cursor"] = _current_queue_index(queue_df, labels_df)

    total = len(queue_df)
    cursor = int(st.session_state.get("cursor", 0))
    cursor = max(0, min(cursor, total - 1))
    row = queue_df.iloc[cursor]

    st.subheader(f"Item {cursor + 1} of {total}")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"Item ID: {row.get('item_id', '')}")
        st.write(f"Detection ID: {row.get('detection_id', '')}")
        st.write(f"ARU: {row.get('aru_id', '')}")
        st.write(f"Source: {row.get('source_audio_path', '')}")
    with c2:
        st.metric("Labeled (latest)", len(_latest_by_item(labels_df)) if not labels_df.empty else 0)

    clip_path = Path(str(row.get("clip_path", "")))
    if clip_path.exists():
        st.audio(str(clip_path), format="audio/wav")
        _render_spectrogram(clip_path)
    else:
        st.error(f"Clip not found: {clip_path}")

    st.markdown("### Label")
    label_cols = st.columns(4)
    labels = ["Tom", "Hen", "Background", "Skip"]

    for i, label in enumerate(labels):
        if label_cols[i].button(label, use_container_width=True):
            if label not in VALID_LABELS:
                st.error(f"Invalid label: {label}")
                return
            out_row = {
                "item_id": str(row.get("item_id", "")),
                "detection_id": str(row.get("detection_id", "")),
                "reviewer_id": reviewer_id,
                "reviewer_name": reviewer_name,
                "label": label,
                "label_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "session_id": st.session_state.get("session_id") or str(uuid.uuid4()),
                "app_version": "0.1.0",
            }
            _append_label_row(project_root, out_row)
            st.session_state["cursor"] = min(cursor + 1, total - 1)
            st.rerun()

    nav1, nav2, nav3 = st.columns(3)
    if nav1.button("Previous", use_container_width=True):
        st.session_state["cursor"] = max(0, cursor - 1)
        st.rerun()
    if nav2.button("Next", use_container_width=True):
        st.session_state["cursor"] = min(total - 1, cursor + 1)
        st.rerun()
    if nav3.button("Jump to first unlabeled", use_container_width=True):
        labels_df = _load_existing_labels(project_root, reviewer_id)
        st.session_state["cursor"] = _current_queue_index(queue_df, labels_df)
        st.rerun()


if __name__ == "__main__":
    main()
