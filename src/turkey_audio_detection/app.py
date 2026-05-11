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
    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    elif "queue_order" in df.columns:
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


_BG = "#0e1117"  # Streamlit dark background


@st.cache_data(show_spinner=False)
def _spectrogram_b64(audio_path_str: str) -> str | None:
    """Compute mel spectrogram and return base64-encoded PNG. Cached by path."""
    import io as _io
    import base64 as _b64
    try:
        y, sr = librosa.load(audio_path_str, sr=None, mono=True)
    except Exception:
        return None
    if y.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 2), facecolor=_BG)
    ax.set_facecolor(_BG)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    db = librosa.power_to_db(melspec, ref=max(1e-6, melspec.max()))
    librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    plt.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return _b64.b64encode(buf.getvalue()).decode()


def _render_spectrogram(audio_path: Path) -> None:
    b64 = _spectrogram_b64(str(audio_path))
    if b64 is None:
        st.warning("Unable to load spectrogram.")
        return
    # Embed as data URI — bypasses Streamlit's ephemeral media file storage.
    st.html(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;display:block;background:{_BG}">'
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Remove vertical scroll */
        html, body { overflow: hidden !important; }
        section[data-testid="stAppViewContainer"] > div:first-child { overflow: hidden !important; }
        section[data-testid="stMain"] > div:first-child { overflow: hidden !important; }

        /* Shrink default Streamlit vertical padding so content fits */
        div[data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important;
            padding-bottom: 0.5rem !important;
        }

        /* Centered page title and detection counter */
        h1, h3 { text-align: center; }

        /* Centered sidebar section header */
        section[data-testid="stSidebar"] h2 { text-align: center; }

        /* Centered Enter button in sidebar */
        section[data-testid="stSidebar"] div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }

        /* Label button colors — identified by being in the only 4-column row.
           Nav buttons use 3 columns so they are not affected.
           Colors are intentionally muted/desaturated for comfortable repeated use. */
        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"]:nth-child(4)
        ) > div[data-testid="stColumn"] button {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            border: none !important;
        }
        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"]:nth-child(4)
        ) > div[data-testid="stColumn"]:nth-child(1) button {
            background-color: #b8922e !important;  /* muted gold */
            color: #f5f0e8 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"]:nth-child(4)
        ) > div[data-testid="stColumn"]:nth-child(2) button {
            background-color: #a84830 !important;  /* muted terracotta */
            color: #f5f0ee !important;
        }
        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"]:nth-child(4)
        ) > div[data-testid="stColumn"]:nth-child(3) button {
            background-color: #52286b !important;  /* muted plum */
            color: #f0ecf5 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(
            > div[data-testid="stColumn"]:nth-child(4)
        ) > div[data-testid="stColumn"]:nth-child(4) button {
            background-color: #5e5e5e !important;  /* dark gray */
            color: #f0f0f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _launch_via_streamlit() -> None:
    """Re-launch this module via `streamlit run` when called as a console script."""
    import os
    import subprocess

    env = os.environ.copy()
    env["_TURKEY_STREAMLIT_CHILD"] = "1"
    script = Path(__file__).resolve()
    args = ["streamlit", "run", str(script), "--server.headless", "false"]
    raise SystemExit(subprocess.call(args, env=env))


def main() -> None:
    """Entry point: re-launches via streamlit when called as a console script."""
    import os
    # _TURKEY_STREAMLIT_CHILD is set by _launch_via_streamlit() so Streamlit
    # inherits it. If it's absent we're a plain Python process and must re-launch.
    if not os.environ.get("_TURKEY_STREAMLIT_CHILD"):
        _launch_via_streamlit()

    st.set_page_config(page_title="Turkey Call Labeler", layout="wide")
    _inject_css()
    st.markdown("<h1>Turkey Call Labeler</h1>", unsafe_allow_html=True)

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

        reviewer_id = st.text_input("Reviewer Name", value=st.session_state.get("reviewer_id", ""))

        if st.button("Enter", type="primary"):
            st.session_state["reviewer_id"] = reviewer_id.strip()
            st.session_state["run_id"] = selected_run.strip()
            st.session_state["session_id"] = st.session_state.get("session_id") or str(uuid.uuid4())
            st.rerun()

    reviewer_id = st.session_state.get("reviewer_id", "").strip()
    run_id = st.session_state.get("run_id", "").strip()

    if not reviewer_id:
        st.info("Enter your name in the sidebar and click Enter.")
        return
    if not run_id:
        st.info("Select a run ID in the sidebar and click Enter.")
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

    st.subheader(f"Detection {cursor + 1} of {total}")

    aru_display = str(row.get("aru_id", "—")).replace("ARU_", "")
    raw_dt = str(row.get("recording_datetime", ""))
    date_str, time_str = "—", "—"
    if raw_dt and raw_dt not in ("", "nan"):
        try:
            _dt = datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
            date_str = _dt.strftime("%m-%d-%Y")
            time_str = _dt.strftime("%I:%M %p").lower()
            if time_str.startswith("0"):
                time_str = time_str[1:]
        except ValueError:
            date_str = raw_dt
    conf_str = "—"
    try:
        _conf_f = float(row.get("confidence", None))
        if not pd.isna(_conf_f):
            conf_str = f"{_conf_f:.1%}"
    except (TypeError, ValueError):
        pass
    _item_id = str(row.get("item_id", ""))
    _existing_label = "No Label"
    if not labels_df.empty and "item_id" in labels_df.columns:
        _latest = _latest_by_item(labels_df)
        _match = _latest[_latest["item_id"].astype(str) == _item_id]
        if not _match.empty:
            _existing_label = str(_match.iloc[0]["label"])
    st.html(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'width:100%;font-size:1.1rem;margin:0.15rem 0;color:inherit">'
        f'<span style="text-align:left">ARU: {aru_display}</span>'
        f'<span>Date: {date_str}</span>'
        f'<span>Time: {time_str}</span>'
        f'<span>BirdNET Confidence: {conf_str}</span>'
        f'<span style="text-align:right">Label: {_existing_label}</span>'
        f'</div>'
    )

    clip_path = Path(str(row.get("clip_path", "")))
    if clip_path.exists():
        import base64
        _audio_b64 = base64.b64encode(clip_path.read_bytes()).decode()
        st.html(
            f'<body style="margin:0;padding:0;background:{_BG}">'
            f'<audio id="clip" controls '
            f'style="width:100%;color-scheme:dark">'
            f'<source src="data:audio/wav;base64,{_audio_b64}" type="audio/wav">'
            f'</audio>'
            f'<script>'
            f'var a=document.getElementById("clip");'
            f'a.load();'
            f'a.play().catch(function(){{}});'
            f'</script>'
            f'</body>',
            unsafe_allow_javascript=True,
        )

        # Label buttons ABOVE the spectrogram
        label_cols = st.columns(4)
        labels_map = [
            ("Tom",        "label-btn-tom"),
            ("Hen",        "label-btn-hen"),
            ("Background", "label-btn-background"),
            ("Skip",       "label-btn-skip"),
        ]
        for i, (label, _css_class) in enumerate(labels_map):
            with label_cols[i]:
                if st.button(label, width='stretch', key=f"label_{label}"):
                    out_row = {
                        "item_id": str(row.get("item_id", "")),
                        "detection_id": str(row.get("detection_id", "")),
                        "reviewer_id": reviewer_id,
                        "reviewer_name": reviewer_id,
                        "label": label,
                        "label_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "session_id": st.session_state.get("session_id") or str(uuid.uuid4()),
                        "app_version": "0.1.0",
                    }
                    _append_label_row(project_root, out_row)
                    st.session_state["cursor"] = min(cursor + 1, total - 1)
                    st.rerun()

        _render_spectrogram(clip_path)
    else:
        st.error(f"Clip not found: {clip_path}")

    nav1, nav2, nav3 = st.columns(3)
    if nav1.button("Previous", width='stretch'):
        st.session_state["cursor"] = max(0, cursor - 1)
        st.rerun()
    if nav2.button("Next", width='stretch'):
        st.session_state["cursor"] = min(total - 1, cursor + 1)
        st.rerun()
    if nav3.button("Jump to first unlabeled", width='stretch'):
        labels_df = _load_existing_labels(project_root, reviewer_id)
        st.session_state["cursor"] = _current_queue_index(queue_df, labels_df)
        st.rerun()


if __name__ == "__main__":
    main()
