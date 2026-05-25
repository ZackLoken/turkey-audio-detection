"""Streamlit review application for queue labeling."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from turkey_audio_detection.schemas import REGION_LABELS


CANVAS_FMIN_HZ = 200.0
CANVAS_FMAX_HZ = 6000.0
CANVAS_SR = 48000
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 320
N_MELS_CANVAS = 128
N_FFT_CANVAS = 2048
HOP_LENGTH_CANVAS = 512
DEFAULT_CLIP_DURATION_S = 3.0

STROKE_COLOR_TOM = "#b8922e"  # gold
STROKE_COLOR_HEN = "#a84830"  # terracotta
FILL_COLOR = "rgba(255, 255, 255, 0.15)"


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


LABEL_COLUMNS = [
    "item_id",
    "detection_id",
    "reviewer_id",
    "reviewer_name",
    "regions_json",
    "other_birds_present",
    "unsure",
    "tom_present",
    "hen_present",
    "label_timestamp_utc",
    "session_id",
]


def _append_label_row(project_root: Path, row: dict) -> None:
    path = _labels_path(project_root, row["reviewer_id"])
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LABEL_COLUMNS})


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


@st.cache_data(show_spinner=False)
def _spectrogram_for_canvas(audio_path_str: str, width_px: int, height_px: int) -> Image.Image | None:
    """Render an axis-free mel spectrogram pinned to the canvas band, sized for st_canvas."""
    try:
        y, sr = librosa.load(audio_path_str, sr=CANVAS_SR, mono=True)
    except Exception:
        return None
    if y.size == 0:
        return None
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT_CANVAS,
        hop_length=HOP_LENGTH_CANVAS,
        n_mels=N_MELS_CANVAS,
        fmin=CANVAS_FMIN_HZ,
        fmax=CANVAS_FMAX_HZ,
    )
    db = librosa.power_to_db(melspec, ref=max(1e-6, float(melspec.max())))

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=_BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(_BG)
    librosa.display.specshow(
        db,
        sr=sr,
        hop_length=HOP_LENGTH_CANVAS,
        x_axis=None,
        y_axis=None,
        ax=ax,
        fmin=CANVAS_FMIN_HZ,
        fmax=CANVAS_FMAX_HZ,
    )
    ax.set_axis_off()
    import io as _io
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0, facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((width_px, height_px))


def pixel_y_to_hz(py: float, canvas_h: int, fmin_hz: float, fmax_hz: float) -> float:
    """Convert a canvas pixel-y (0 at top = high frequency) into Hz on the mel scale."""
    mel_min = librosa.hz_to_mel(fmin_hz)
    mel_max = librosa.hz_to_mel(fmax_hz)
    frac = max(0.0, min(1.0, 1.0 - (py / canvas_h)))
    mel = mel_min + frac * (mel_max - mel_min)
    return float(librosa.mel_to_hz(mel))


def hz_to_pixel_y(hz: float, canvas_h: int, fmin_hz: float, fmax_hz: float) -> float:
    """Inverse of pixel_y_to_hz — used to rebuild canvas rectangles from saved regions."""
    mel_min = librosa.hz_to_mel(fmin_hz)
    mel_max = librosa.hz_to_mel(fmax_hz)
    mel_target = librosa.hz_to_mel(max(fmin_hz, min(fmax_hz, hz)))
    frac = (mel_target - mel_min) / (mel_max - mel_min) if mel_max > mel_min else 0.0
    return float((1.0 - frac) * canvas_h)


def rect_to_region(
    obj: dict,
    canvas_w: int,
    canvas_h: int,
    clip_duration_s: float,
    fmin_hz: float,
    fmax_hz: float,
    snap_freq: bool,
) -> dict | None:
    """Convert one drawable-canvas rect into a region dict, or None if degenerate."""
    if obj.get("type") != "rect":
        return None
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
    height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
    if width <= 0 or height <= 0:
        return None

    start_s = max(0.0, min(clip_duration_s, (left / canvas_w) * clip_duration_s))
    end_s = max(0.0, min(clip_duration_s, ((left + width) / canvas_w) * clip_duration_s))
    if end_s <= start_s:
        return None

    if snap_freq:
        freq_min_hz = float(fmin_hz)
        freq_max_hz = float(fmax_hz)
    else:
        freq_min_hz = pixel_y_to_hz(top + height, canvas_h, fmin_hz, fmax_hz)
        freq_max_hz = pixel_y_to_hz(top, canvas_h, fmin_hz, fmax_hz)
    if freq_max_hz <= freq_min_hz:
        return None

    stroke = str(obj.get("stroke", "")).lower()
    if stroke == STROKE_COLOR_TOM.lower():
        label = "Tom"
    elif stroke == STROKE_COLOR_HEN.lower():
        label = "Hen"
    else:
        # Unknown stroke — fall back to Tom (better than dropping a real annotation).
        label = "Tom"

    return {
        "start_s": round(float(start_s), 4),
        "end_s": round(float(end_s), 4),
        "freq_min_hz": round(float(freq_min_hz), 2),
        "freq_max_hz": round(float(freq_max_hz), 2),
        "label": label,
    }


def region_to_rect(
    region: dict,
    canvas_w: int,
    canvas_h: int,
    clip_duration_s: float,
    fmin_hz: float,
    fmax_hz: float,
) -> dict:
    """Build a fabric.js rect object suitable for st_canvas initial_drawing."""
    left = (region["start_s"] / clip_duration_s) * canvas_w
    width = ((region["end_s"] - region["start_s"]) / clip_duration_s) * canvas_w
    top = hz_to_pixel_y(region["freq_max_hz"], canvas_h, fmin_hz, fmax_hz)
    bottom = hz_to_pixel_y(region["freq_min_hz"], canvas_h, fmin_hz, fmax_hz)
    height = max(1.0, bottom - top)
    stroke = STROKE_COLOR_TOM if region.get("label") == "Tom" else STROKE_COLOR_HEN
    return {
        "type": "rect",
        "left": float(left),
        "top": float(top),
        "width": float(width),
        "height": float(height),
        "scaleX": 1.0,
        "scaleY": 1.0,
        "angle": 0,
        "fill": FILL_COLOR,
        "stroke": stroke,
        "strokeWidth": 2,
        "strokeUniform": True,
    }


def regions_to_initial_drawing(
    regions: list[dict],
    canvas_w: int,
    canvas_h: int,
    clip_duration_s: float,
    fmin_hz: float,
    fmax_hz: float,
) -> dict:
    return {
        "version": "5.3.0",
        "objects": [
            region_to_rect(r, canvas_w, canvas_h, clip_duration_s, fmin_hz, fmax_hz)
            for r in regions
        ],
    }


def derive_presence(regions: list[dict]) -> tuple[int, int]:
    """Return (tom_present, hen_present) booleans derived from a region list."""
    tom = int(any(r.get("label") == "Tom" for r in regions))
    hen = int(any(r.get("label") == "Hen" for r in regions))
    return tom, hen


def _parse_regions(value) -> list[dict]:
    """Parse a regions_json cell from CSV, tolerating empty/NaN values."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [r for r in parsed if isinstance(r, dict)]


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

    # Latest snapshot for this item (if any) — drives both the header summary and the canvas pre-population.
    _latest_row = None
    if not labels_df.empty and "item_id" in labels_df.columns:
        _latest = _latest_by_item(labels_df)
        _match = _latest[_latest["item_id"].astype(str) == _item_id]
        if not _match.empty:
            _latest_row = _match.iloc[0]

    existing_regions: list[dict] = (
        _parse_regions(_latest_row.get("regions_json", "")) if _latest_row is not None else []
    )

    def _format_summary(latest_row, regions: list[dict]) -> str:
        if latest_row is None:
            return "No Label"
        tom_n = sum(1 for r in regions if r.get("label") == "Tom")
        hen_n = sum(1 for r in regions if r.get("label") == "Hen")
        parts: list[str] = []
        if tom_n:
            parts.append(f"{tom_n}×Tom")
        if hen_n:
            parts.append(f"{hen_n}×Hen")
        if not parts:
            parts.append("No turkey")
        if int(latest_row.get("other_birds_present", 0) or 0):
            parts.append("+other birds")
        if int(latest_row.get("unsure", 0) or 0):
            parts.append("(unsure)")
        return " ".join(parts)

    _existing_label = _format_summary(_latest_row, existing_regions)

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
    if not clip_path.exists():
        st.error(f"Clip not found: {clip_path}")
        return

    clip_duration_s = float(row.get("clip_end_s", 0.0)) - float(row.get("clip_start_s", 0.0))
    if clip_duration_s <= 0:
        clip_duration_s = DEFAULT_CLIP_DURATION_S

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

    # Full-band reference spectrogram (small strip).
    _render_spectrogram(clip_path)

    # Per-clip widget state defaults — seed once per item_id from the latest snapshot.
    init_key = f"_init_{_item_id}"
    if not st.session_state.get(init_key):
        st.session_state[f"active_label_{_item_id}"] = "Tom"
        st.session_state[f"snap_freq_{_item_id}"] = False
        st.session_state[f"other_birds_{_item_id}"] = bool(
            int(_latest_row.get("other_birds_present", 0) or 0) if _latest_row is not None else 0
        )
        st.session_state[f"unsure_{_item_id}"] = bool(
            int(_latest_row.get("unsure", 0) or 0) if _latest_row is not None else 0
        )
        st.session_state[f"canvas_nonce_{_item_id}"] = 0
        st.session_state[init_key] = True

    ctrl_cols = st.columns([2, 2, 2, 2])
    with ctrl_cols[0]:
        active_label = st.radio(
            "Active label",
            options=["Tom", "Hen"],
            horizontal=True,
            key=f"active_label_{_item_id}",
        )
    with ctrl_cols[1]:
        snap_freq = st.checkbox("Snap to full frequency band", key=f"snap_freq_{_item_id}")
    with ctrl_cols[2]:
        other_birds_present = st.checkbox("Other birds present", key=f"other_birds_{_item_id}")
    with ctrl_cols[3]:
        unsure = st.checkbox("Unsure", key=f"unsure_{_item_id}")

    stroke_color = STROKE_COLOR_TOM if active_label == "Tom" else STROKE_COLOR_HEN

    spec_pil = _spectrogram_for_canvas(str(clip_path), CANVAS_WIDTH, CANVAS_HEIGHT)
    if spec_pil is None:
        st.warning("Unable to render canvas spectrogram.")
        return

    initial_drawing = regions_to_initial_drawing(
        existing_regions,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        clip_duration_s,
        CANVAS_FMIN_HZ,
        CANVAS_FMAX_HZ,
    ) if existing_regions else None

    nonce = st.session_state.get(f"canvas_nonce_{_item_id}", 0)
    canvas_result = st_canvas(
        fill_color=FILL_COLOR,
        stroke_width=2,
        stroke_color=stroke_color,
        background_image=spec_pil,
        update_streamlit=True,
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="rect",
        initial_drawing=initial_drawing,
        key=f"canvas_{_item_id}_{nonce}",
    )

    st.caption(
        f"Canvas band: {int(CANVAS_FMIN_HZ)}–{int(CANVAS_FMAX_HZ)} Hz   ·   "
        f"clip duration: {clip_duration_s:.2f} s   ·   "
        f"draw rectangles around each Tom / Hen call (switch label before drawing)."
    )

    action_cols = st.columns([2, 1, 1, 1])
    save_clicked = action_cols[0].button("Save & Next", type="primary", width="stretch")
    prev_clicked = action_cols[1].button("Previous", width="stretch")
    discard_clicked = action_cols[2].button("Reset canvas", width="stretch")
    jump_clicked = action_cols[3].button("Jump to first unlabeled", width="stretch")

    if save_clicked:
        raw_objects = []
        if canvas_result is not None and canvas_result.json_data is not None:
            raw_objects = canvas_result.json_data.get("objects", []) or []
        regions: list[dict] = []
        for obj in raw_objects:
            region = rect_to_region(
                obj,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                clip_duration_s,
                CANVAS_FMIN_HZ,
                CANVAS_FMAX_HZ,
                snap_freq,
            )
            if region is not None:
                regions.append(region)

        tom_present, hen_present = derive_presence(regions)
        out_row = {
            "item_id": _item_id,
            "detection_id": str(row.get("detection_id", "")),
            "reviewer_id": reviewer_id,
            "reviewer_name": reviewer_id,
            "regions_json": json.dumps(regions, separators=(",", ":")),
            "other_birds_present": int(bool(other_birds_present)),
            "unsure": int(bool(unsure)),
            "tom_present": tom_present,
            "hen_present": hen_present,
            "label_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": st.session_state.get("session_id") or str(uuid.uuid4()),
        }
        _append_label_row(project_root, out_row)
        st.session_state["cursor"] = min(cursor + 1, total - 1)
        st.rerun()

    if prev_clicked:
        st.session_state["cursor"] = max(0, cursor - 1)
        st.rerun()

    if discard_clicked:
        st.session_state[f"canvas_nonce_{_item_id}"] = nonce + 1
        st.rerun()

    if jump_clicked:
        labels_df = _load_existing_labels(project_root, reviewer_id)
        st.session_state["cursor"] = _current_queue_index(queue_df, labels_df)
        st.rerun()


if __name__ == "__main__":
    main()
