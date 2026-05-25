"""Streamlit review application for queue labeling."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from turkey_audio_detection.spectrogram_render import (
    CANVAS_FMAX_HZ,
    CANVAS_FMIN_HZ,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DATA_BOTTOM_FRAC,
    DATA_LEFT_FRAC,
    DATA_TOP_FRAC,
    data_area_bounds,
    render_canvas_spectrogram,
)


# Custom canvas component — replaces the unmaintained streamlit-drawable-canvas which
# fails to render background images under Streamlit 1.30+ even with monkey-patches.
_CANVAS_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "canvas"
_canvas_component = components.declare_component(
    "turkey_canvas", path=str(_CANVAS_COMPONENT_DIR)
)


def turkey_canvas(
    *,
    audio_url: str,
    background_image_url: str,
    stroke_color_tom: str,
    stroke_color_hen: str,
    width: int,
    height: int,
    initial_rects: list[dict],
    initial_active_label: str,
    initial_other_birds: bool,
    initial_unsure: bool,
    frame_key: str,
    data_left_frac: float,
    data_bottom_frac: float,
    data_top_frac: float,
    fmin_hz: float,
    fmax_hz: float,
    key: str,
) -> dict:
    """Audio control + active-label toggle + Other-birds/Unsure checkboxes +
    spectrogram canvas + drawn-rectangle state, all in one Streamlit component
    iframe. Returns a dict:
        {
          "rectangles": [{left, top, width, height, stroke, label}, ...],
          "activeLabel": "Tom" | "Hen",
          "otherBirdsPresent": bool,
          "unsure": bool,
        }
    """
    default_value = {
        "rectangles": initial_rects or [],
        "activeLabel": initial_active_label,
        "otherBirdsPresent": bool(initial_other_birds),
        "unsure": bool(initial_unsure),
    }
    result = _canvas_component(
        audioUrl=audio_url,
        backgroundImageUrl=background_image_url,
        strokeColorTom=stroke_color_tom,
        strokeColorHen=stroke_color_hen,
        width=width,
        height=height,
        initialRects=initial_rects,
        initialActiveLabel=initial_active_label,
        initialOtherBirds=bool(initial_other_birds),
        initialUnsure=bool(initial_unsure),
        frameKey=frame_key,
        dataLeftFrac=float(data_left_frac),
        dataBottomFrac=float(data_bottom_frac),
        dataTopFrac=float(data_top_frac),
        fminHz=float(fmin_hz),
        fmaxHz=float(fmax_hz),
        default=default_value,
        key=key,
    )
    return result if isinstance(result, dict) else default_value

DEFAULT_CLIP_DURATION_S = 3.0

# Stroke colors chosen for maximum contrast against librosa's magma colormap
# (dark purple → red → orange → yellow). Lime green and royal blue are both
# outside magma's hue range. The canvas component also draws a white halo
# under each colored stroke so the rectangle stays visible regardless of
# background.
STROKE_COLOR_TOM = "#39ff14"  # neon lime green
STROKE_COLOR_HEN = "#4169e1"  # royal blue
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
            return int(idx)  # type: ignore[arg-type]
    return len(queue_df) - 1


def _cached_spectrogram_path(clip_path: Path, item_id: str) -> Path:
    """`data/_outputs/runs/<run_id>/clips/<item_id>.wav` →
    `data/_outputs/runs/<run_id>/spectrograms/<item_id>.png`.
    """
    return clip_path.parent.parent / "spectrograms" / f"{item_id}.png"


def _spectrogram_pil_for_clip(clip_path: Path, item_id: str) -> Image.Image | None:
    """Return the clip's canvas-band spectrogram as a PIL image.

    Reads the pre-rendered PNG cache if present (fast). Falls back to live
    mel computation when the cache is missing. Reads bytes into memory and
    constructs the image from BytesIO so the returned image is fully decoupled
    from the on-disk file handle.
    """
    import io as _io

    cached = _cached_spectrogram_path(clip_path, item_id)
    if cached.exists():
        try:
            data = cached.read_bytes()
            img = Image.open(_io.BytesIO(data))
            img.load()
            return img.convert("RGB")
        except Exception:
            pass
    return _spectrogram_for_canvas(str(clip_path), CANVAS_WIDTH, CANVAS_HEIGHT)


def _spectrogram_png_b64_for_clip(clip_path: Path, item_id: str) -> str:
    """Return base64-PNG of the clip's canvas-band spectrogram for embedding as a
    `data:image/png;base64,...` URL in the custom canvas component. Reads the
    pre-rendered PNG cache as raw bytes when present (fast, no PIL re-encode).
    """
    import base64 as _b64

    cached = _cached_spectrogram_path(clip_path, item_id)
    if cached.exists():
        try:
            return _b64.b64encode(cached.read_bytes()).decode()
        except Exception:
            pass
    pil = _spectrogram_for_canvas(str(clip_path), CANVAS_WIDTH, CANVAS_HEIGHT)
    if pil is None:
        return ""
    import io as _io

    buf = _io.BytesIO()
    pil.save(buf, format="PNG")
    return _b64.b64encode(buf.getvalue()).decode()


_BG = "#0e1117"  # Streamlit dark background


@st.cache_data(show_spinner=False)
def _spectrogram_for_canvas(audio_path_str: str, width_px: int, height_px: int) -> Image.Image | None:
    """Streamlit-cached wrapper around the shared spectrogram renderer."""
    return render_canvas_spectrogram(audio_path_str, width_px, height_px)


def pixel_y_to_hz(py: float, canvas_h: int, fmin_hz: float, fmax_hz: float,
                  canvas_w: int = CANVAS_WIDTH) -> float:
    """Convert a canvas pixel-y to Hz on the mel scale, clipped to the data area.

    The spectrogram leaves a margin at the bottom (and a thin top margin in some
    configurations) for axis labels. Pixel-y values inside the margin clip to the
    nearest data-area edge so the returned Hz is always inside [fmin, fmax].
    """
    _, data_top, _, data_bottom = data_area_bounds(canvas_w, canvas_h)
    py_clipped = max(data_top, min(data_bottom, py))
    mel_min = librosa.hz_to_mel(fmin_hz)
    mel_max = librosa.hz_to_mel(fmax_hz)
    frac = 1.0 - (py_clipped - data_top) / (data_bottom - data_top)
    frac = max(0.0, min(1.0, frac))
    mel = mel_min + frac * (mel_max - mel_min)
    return float(librosa.mel_to_hz(mel))


def hz_to_pixel_y(hz: float, canvas_h: int, fmin_hz: float, fmax_hz: float,
                  canvas_w: int = CANVAS_WIDTH) -> float:
    """Inverse of pixel_y_to_hz — returns a pixel-y inside the data area."""
    _, data_top, _, data_bottom = data_area_bounds(canvas_w, canvas_h)
    mel_min = librosa.hz_to_mel(fmin_hz)
    mel_max = librosa.hz_to_mel(fmax_hz)
    mel_target = librosa.hz_to_mel(max(fmin_hz, min(fmax_hz, hz)))
    frac = (mel_target - mel_min) / (mel_max - mel_min) if mel_max > mel_min else 0.0
    return float(data_top + (1.0 - frac) * (data_bottom - data_top))


def rect_to_region(
    obj: dict,
    canvas_w: int,
    canvas_h: int,
    clip_duration_s: float,
    fmin_hz: float,
    fmax_hz: float,
    snap_freq: bool,
) -> dict | None:
    """Convert one drawable-canvas rect into a region dict, or None if degenerate.

    Rectangles are clipped to the spectrogram data area so any portion the
    reviewer draws into the axis-label margins is discarded before computing
    (time, Hz) bounds.
    """
    if obj.get("type") != "rect":
        return None
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
    height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
    if width <= 0 or height <= 0:
        return None

    data_left, data_top, data_right, data_bottom = data_area_bounds(canvas_w, canvas_h)
    rect_left = max(data_left, left)
    rect_right = min(data_right, left + width)
    rect_top = max(data_top, top)
    rect_bottom = min(data_bottom, top + height)
    if rect_right <= rect_left or rect_bottom <= rect_top:
        return None

    data_w = data_right - data_left
    start_s = max(0.0, min(clip_duration_s, ((rect_left - data_left) / data_w) * clip_duration_s))
    end_s = max(0.0, min(clip_duration_s, ((rect_right - data_left) / data_w) * clip_duration_s))
    if end_s <= start_s:
        return None

    if snap_freq:
        freq_min_hz = float(fmin_hz)
        freq_max_hz = float(fmax_hz)
    else:
        freq_min_hz = pixel_y_to_hz(rect_bottom, canvas_h, fmin_hz, fmax_hz, canvas_w)
        freq_max_hz = pixel_y_to_hz(rect_top, canvas_h, fmin_hz, fmax_hz, canvas_w)
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
    """Build a rect dict (`left`, `top`, `width`, `height`, `stroke`) for the
    custom canvas's `initial_rects` payload.

    Region coords are expressed in (time, Hz); the rect is positioned inside the
    spectrogram data area so it lines up with the underlying image.
    """
    data_left, _, data_right, _ = data_area_bounds(canvas_w, canvas_h)
    data_w = data_right - data_left
    left = data_left + (region["start_s"] / clip_duration_s) * data_w
    width = ((region["end_s"] - region["start_s"]) / clip_duration_s) * data_w
    top = hz_to_pixel_y(region["freq_max_hz"], canvas_h, fmin_hz, fmax_hz, canvas_w)
    bottom = hz_to_pixel_y(region["freq_min_hz"], canvas_h, fmin_hz, fmax_hz, canvas_w)
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
        /* Shrink default Streamlit vertical padding so the audio + spectrogram +
           controls + nav buttons fit close together. Vertical scroll is left
           enabled (Streamlit default) so the nav buttons remain reachable on
           shorter monitors. */
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

        reviewer_id = st.text_input("Reviewer Name", value=st.session_state.get("reviewer_id", "")) or ""

        if st.button("Enter", type="primary"):
            st.session_state["reviewer_id"] = reviewer_id.strip()
            st.session_state["run_id"] = (selected_run or "").strip()
            st.session_state["session_id"] = st.session_state.get("session_id") or str(uuid.uuid4())
            st.rerun()

        st.markdown("---")
        with st.expander("How to label", expanded=False):
            st.markdown(
                "Listen, then draw rectangles around each turkey call.\n\n"
                "- **Tom** = lime green &nbsp; **Hen** = royal blue. Toggle active "
                "label between drawings.\n"
                "- **Click a rectangle** to replay its clipped audio. "
                "**Double-click** deletes.\n"
                "- **Other birds present** — any non-turkey bird audible in the clip.\n"
                "- **Unsure** — you can't reliably tell. Excluded from agreement "
                "stats by default.\n"
                "- **Save & Next** writes the snapshot and advances; saving empty = "
                "*no turkey* label.\n"
                "- **Previous** edits the last labeled clip.\n"
                "- **Reset canvas** clears drawings and the saved snapshot for "
                "this clip."
            )

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
            # The WAV filename's HHMMSS is the ARU's local clock (presumed Eastern
            # per IndexConfig.timezone_name's default of "US/Eastern"). Append "ET"
            # so reviewers don't have to guess.
            time_str = time_str + " ET"
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
    # One-shot "reset" flag — when set by the Reset canvas button, suppress the saved
    # regions for a single render so the canvas comes up empty. The flag clears
    # itself after consumption so subsequent renders of the same clip restore the
    # saved snapshot (e.g. when Previous brings the reviewer back).
    _reset_key = f"_reset_{_item_id}"
    if st.session_state.get(_reset_key):
        existing_regions = []
        st.session_state[_reset_key] = False

    def _format_summary(latest_row, regions: list[dict]) -> str:
        if latest_row is None:
            return "No Label"
        tom_n = sum(1 for r in regions if r.get("label") == "Tom")
        hen_n = sum(1 for r in regions if r.get("label") == "Hen")
        parts: list[str] = []
        if tom_n:
            parts.append(f"{tom_n} Tom")
        if hen_n:
            parts.append(f"{hen_n} Hen")
        summary = ", ".join(parts) if parts else "No turkey"
        if int(latest_row.get("other_birds_present", 0) or 0):
            summary += " · other birds"
        if int(latest_row.get("unsure", 0) or 0):
            summary += " · unsure"
        return summary

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
    # Audio is now rendered INSIDE the canvas component so audio events drive
    # the timestamp and the playhead with no cross-iframe communication.
    _audio_url = "data:audio/wav;base64," + _audio_b64

    # Per-clip canvas state — the controls (active label, other birds, unsure)
    # live inside the canvas component so we only seed initial values here; the
    # current state comes back via the component's return value on each rerun.
    init_key = f"_init_{_item_id}"
    if not st.session_state.get(init_key):
        st.session_state[f"canvas_nonce_{_item_id}"] = 0
        st.session_state[init_key] = True

    initial_other_birds = bool(
        int(_latest_row.get("other_birds_present", 0) or 0) if _latest_row is not None else 0
    )
    initial_unsure = bool(
        int(_latest_row.get("unsure", 0) or 0) if _latest_row is not None else 0
    )

    spec_b64 = _spectrogram_png_b64_for_clip(clip_path, _item_id)
    if not spec_b64:
        st.warning("Unable to render canvas spectrogram.")
        return
    spec_url = "data:image/png;base64," + spec_b64

    # Seed the canvas with rectangles from the latest saved snapshot (if any), so
    # Previous on a labeled clip restores the user's prior drawing.
    initial_rects: list[dict] = []
    for region in existing_regions:
        rect = region_to_rect(
            region, CANVAS_WIDTH, CANVAS_HEIGHT, clip_duration_s, CANVAS_FMIN_HZ, CANVAS_FMAX_HZ
        )
        initial_rects.append({
            "left": float(rect["left"]),
            "top": float(rect["top"]),
            "width": float(rect["width"]),
            "height": float(rect["height"]),
            "stroke": rect["stroke"],
            "label": region.get("label", "Tom"),
        })

    nonce = st.session_state.get(f"canvas_nonce_{_item_id}", 0)
    canvas_state = turkey_canvas(
        audio_url=_audio_url,
        background_image_url=spec_url,
        stroke_color_tom=STROKE_COLOR_TOM,
        stroke_color_hen=STROKE_COLOR_HEN,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        initial_rects=initial_rects,
        initial_active_label="Tom",
        initial_other_birds=initial_other_birds,
        initial_unsure=initial_unsure,
        frame_key=f"{_item_id}_{nonce}",
        data_left_frac=DATA_LEFT_FRAC,
        data_bottom_frac=DATA_BOTTOM_FRAC,
        data_top_frac=DATA_TOP_FRAC,
        fmin_hz=CANVAS_FMIN_HZ,
        fmax_hz=CANVAS_FMAX_HZ,
        key=f"canvas_{_item_id}_{nonce}",
    )
    canvas_rects = canvas_state.get("rectangles", []) or []
    other_birds_present = bool(canvas_state.get("otherBirdsPresent", initial_other_birds))
    unsure = bool(canvas_state.get("unsure", initial_unsure))

    action_cols = st.columns([2, 1, 1, 1])
    save_clicked = action_cols[0].button("Save & Next", type="primary", width="stretch")
    prev_clicked = action_cols[1].button("Previous", width="stretch")
    discard_clicked = action_cols[2].button("Reset canvas", width="stretch")
    jump_clicked = action_cols[3].button("Jump to first unlabeled", width="stretch")

    if save_clicked:
        regions: list[dict] = []
        for obj in canvas_rects or []:
            # The custom canvas sends rectangles in the same shape rect_to_region expects:
            # {left, top, width, height, stroke, label}. We synthesize a `type: "rect"` so
            # the rect_to_region guard accepts the row, and pass scaleX/scaleY = 1.0
            # (the custom canvas doesn't apply any scaling).
            shape = dict(obj)
            shape.setdefault("type", "rect")
            shape.setdefault("scaleX", 1.0)
            shape.setdefault("scaleY", 1.0)
            region = rect_to_region(
                shape,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                clip_duration_s,
                CANVAS_FMIN_HZ,
                CANVAS_FMAX_HZ,
                snap_freq=False,
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
        # Reset canvas wipes any saved snapshot(s) for this item from the
        # reviewer's labels CSV so the clip becomes truly unlabeled (and
        # `Jump to first unlabeled` returns to it). Combined with the
        # _reset_key flag + nonce bump, the canvas widget also re-renders
        # empty on the next pass.
        labels_path = _labels_path(project_root, reviewer_id)
        if labels_path.exists():
            try:
                df_lbl = pd.read_csv(labels_path)
                if "item_id" in df_lbl.columns and not df_lbl.empty:
                    df_lbl = df_lbl[df_lbl["item_id"].astype(str) != _item_id]
                    df_lbl.to_csv(labels_path, index=False)
            except Exception:
                pass
        st.session_state[_reset_key] = True
        st.session_state[f"canvas_nonce_{_item_id}"] = nonce + 1
        st.rerun()

    if jump_clicked:
        labels_df = _load_existing_labels(project_root, reviewer_id)
        st.session_state["cursor"] = _current_queue_index(queue_df, labels_df)
        st.rerun()


if __name__ == "__main__":
    main()
