"""Pipeline stage implementations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re
import zoneinfo

import pandas as pd
import soundfile as sf
from astral import LocationInfo
from astral.sun import sun

from turkey_audio_detection.config import BirdNetConfig, ClipConfig, IndexConfig
from turkey_audio_detection.ids import make_detection_id, make_item_id
from turkey_audio_detection.layout import RunLayout, find_aru_dirs
from turkey_audio_detection.spectrogram_render import save_canvas_spectrogram


FILENAME_PATTERN = re.compile(r"^(\w+)_(\d{8})_(\d{6})\.wav$", re.IGNORECASE)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stage_index_data(layout: RunLayout, cfg: IndexConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    quarantine: list[dict] = []

    aru_dirs = find_aru_dirs(layout.project_root)
    for aru_dir in aru_dirs:
        aru_id = aru_dir.name
        for wav_path in sorted(aru_dir.rglob(cfg.wav_glob)):
            match = FILENAME_PATTERN.match(wav_path.name)
            if not match:
                quarantine.append(
                    {
                        "filepath": str(wav_path),
                        "filename": wav_path.name,
                        "reason": "filename_not_matching_expected_pattern",
                    }
                )
                continue

            device_id = match.group(1)
            date_str = match.group(2)
            time_str = match.group(3)
            rec_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            if rec_datetime.date() < cfg.deployment_start:
                quarantine.append(
                    {
                        "filepath": str(wav_path),
                        "filename": wav_path.name,
                        "reason": "pre_deployment_recording",
                    }
                )
                continue

            records.append(
                {
                    "aru_id": aru_id,
                    "device_id": device_id,
                    "date": rec_datetime.date().isoformat(),
                    "time": rec_datetime.time().isoformat(),
                    "datetime": rec_datetime.isoformat(),
                    "filepath": str(wav_path),
                }
            )

    index_df = pd.DataFrame(
        records,
        columns=pd.Index(["aru_id", "device_id", "date", "time", "datetime", "filepath"]),
    )
    quarantine_df = pd.DataFrame(
        quarantine, columns=pd.Index(["filepath", "filename", "reason"])
    )

    if not index_df.empty:
        tz = zoneinfo.ZoneInfo(cfg.timezone_name)
        location = LocationInfo(
            name="project-site",
            region="unknown",
            timezone=cfg.timezone_name,
            latitude=cfg.latitude,
            longitude=cfg.longitude,
        )

        sunrise_cache: dict[str, datetime] = {}

        def _sunrise_for_date(date_text: str) -> datetime:
            if date_text in sunrise_cache:
                return sunrise_cache[date_text]
            d = datetime.strptime(date_text, "%Y-%m-%d").date()
            s = sun(location.observer, date=d, tzinfo=tz)
            sunrise_cache[date_text] = s["sunrise"].replace(tzinfo=None)
            return sunrise_cache[date_text]

        mins: list[float] = []
        sunrises: list[str] = []
        for _, row in index_df.iterrows():
            d_text = str(row["date"])
            sunrise_dt = _sunrise_for_date(d_text)
            rec_dt = datetime.fromisoformat(str(row["datetime"]))
            delta = (rec_dt - sunrise_dt).total_seconds() / 60.0
            mins.append(delta)
            sunrises.append(sunrise_dt.isoformat())

        index_df["sunrise"] = sunrises
        index_df["minutes_from_sunrise"] = mins
        index_df["in_prime_window"] = index_df["minutes_from_sunrise"].between(
            -cfg.prime_window_minutes_before,
            cfg.prime_window_minutes_after,
        )
    else:
        index_df["sunrise"] = []
        index_df["minutes_from_sunrise"] = []
        index_df["in_prime_window"] = []

    layout.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = layout.index_dir / "file_index.csv"
    quarantine_path = layout.index_dir / "quarantine_filenames.csv"
    index_df.to_csv(index_path, index=False)
    quarantine_df.to_csv(quarantine_path, index=False)
    return index_df, quarantine_df


def stage_run_birdnet(layout: RunLayout, cfg: BirdNetConfig) -> pd.DataFrame:
    index_path = layout.index_dir / "file_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index CSV: {index_path}")

    df_index = pd.read_csv(index_path)
    if df_index.empty:
        out = pd.DataFrame(
            columns=pd.Index([
                "detection_id",
                "project_root",
                "aru_id",
                "audio_path",
                "start_time_s",
                "end_time_s",
                "species_code",
                "species_common_name",
                "confidence",
                "birdnet_model_version",
                "source_filename",
                "source_row_index",
            ])
        )
        out.to_csv(layout.birdnet_dir / "detections_normalized.csv", index=False)
        return out

    if cfg.prime_window_only and "in_prime_window" in df_index.columns:
        df_index = df_index[df_index["in_prime_window"].astype(bool)].reset_index(drop=True)

    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    from tqdm import tqdm

    analyzer = Analyzer()
    rows: list[dict] = []
    errors: list[dict] = []

    for row_idx, row in tqdm(
        df_index.iterrows(),
        total=len(df_index),
        desc="BirdNET",
        unit="file",
        dynamic_ncols=True,
    ):
        audio_path = str(row["filepath"])
        dt_value = row.get("datetime")
        dt = None
        if isinstance(dt_value, str) and dt_value:
            try:
                dt = datetime.fromisoformat(dt_value)
            except ValueError:
                dt = None

        try:
            recording = Recording(
                analyzer,
                audio_path,
                lat=cfg.latitude,
                lon=cfg.longitude,
                date=dt,
                min_conf=cfg.min_confidence,
            )
            recording.analyze()
        except Exception as exc:
            errors.append({"filepath": audio_path, "error": str(exc)})
            continue

        for det in recording.detections:
            start_s = _safe_float(det.get("start_time"))
            end_s = _safe_float(det.get("end_time"))
            common_name = str(det.get("common_name", ""))
            sci_name = str(det.get("scientific_name", ""))
            confidence = _safe_float(det.get("confidence"))

            detection_id = make_detection_id(
                source_audio_path=audio_path,
                start_time_s=start_s,
                end_time_s=end_s,
                species_code=sci_name or common_name,
            )

            rows.append(
                {
                    "detection_id": detection_id,
                    "project_root": str(layout.project_root),
                    "aru_id": row.get("aru_id", ""),
                    "audio_path": audio_path,
                    "start_time_s": start_s,
                    "end_time_s": end_s,
                    "species_code": sci_name,
                    "species_common_name": common_name,
                    "confidence": confidence,
                    "birdnet_model_version": "birdnetlib",
                    "source_filename": Path(audio_path).name,
                    "source_row_index": int(row_idx),  # type: ignore[arg-type]
                }
            )

    detection_columns = pd.Index([
        "detection_id", "project_root", "aru_id", "audio_path",
        "start_time_s", "end_time_s", "species_code", "species_common_name",
        "confidence", "birdnet_model_version", "source_filename", "source_row_index",
    ])
    out_df = pd.DataFrame(rows, columns=detection_columns) if rows else pd.DataFrame(columns=detection_columns)
    if not out_df.empty:
        out_df.sort_values(["audio_path", "start_time_s", "end_time_s"], inplace=True, ignore_index=True)
    if errors:
        error_path = layout.birdnet_dir / "birdnet_errors.csv"
        pd.DataFrame(errors, columns=pd.Index(["filepath", "error"])).to_csv(error_path, index=False)
    layout.birdnet_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(layout.birdnet_dir / "detections_normalized.csv", index=False)
    return out_df


def _extract_clip(audio_path: Path, clip_path: Path, start_s: float, duration_s: float) -> None:
    info = sf.info(str(audio_path))
    sr = int(info.samplerate)
    n_frames = int(duration_s * sr)
    start_frame = int(start_s * sr)
    if info.frames <= n_frames:
        start_frame = 0
    else:
        start_frame = max(0, min(start_frame, info.frames - n_frames))

    data, out_sr = sf.read(  # type: ignore[misc]
        str(audio_path),
        start=start_frame,
        stop=start_frame + n_frames,
        dtype="float32",
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(clip_path), data, out_sr)


def stage_extract_clips(layout: RunLayout, cfg: ClipConfig) -> pd.DataFrame:
    detections_path = layout.birdnet_dir / "detections_normalized.csv"
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections CSV: {detections_path}")

    detections_df = pd.read_csv(detections_path)
    if detections_df.empty:
        queue_df = pd.DataFrame(
            columns=pd.Index([
                "item_id",
                "detection_id",
                "clip_path",
                "clip_start_s",
                "clip_end_s",
                "queue_order",
                "project_root",
                "aru_id",
                "source_audio_path",
                "confidence",
                "recording_datetime",
            ])
        )
        queue_df.to_csv(layout.queue_dir / "review_queue.csv", index=False)
        return queue_df

    if "species_common_name" not in detections_df.columns:
        raise ValueError("detections CSV is missing 'species_common_name' column")
    turkey_df = detections_df[
        detections_df["species_common_name"].astype(str).str.contains(
            cfg.species_match_substring, case=False, na=False
        )
    ].copy()

    rows: list[dict] = []
    for _, row in turkey_df.iterrows():
        detection_id = str(row["detection_id"])
        audio_path = Path(str(row["audio_path"]))
        start_s = _safe_float(row.get("start_time_s"))
        end_s = _safe_float(row.get("end_time_s"))
        mid = (start_s + end_s) / 2.0
        clip_start = max(0.0, mid - (cfg.clip_duration_s / 2.0))
        clip_end = clip_start + cfg.clip_duration_s
        item_id = make_item_id(detection_id, clip_start, clip_end)

        clip_name = f"{item_id}.wav"
        clip_path = layout.clips_dir / clip_name
        if not clip_path.exists():
            _extract_clip(audio_path, clip_path, clip_start, cfg.clip_duration_s)

        # Parse recording datetime from filename for display in review app
        fname_match = FILENAME_PATTERN.match(audio_path.name)
        recording_datetime = ""
        if fname_match:
            try:
                recording_datetime = datetime.strptime(
                    f"{fname_match.group(2)}_{fname_match.group(3)}", "%Y%m%d_%H%M%S"
                ).isoformat(sep=" ")
            except ValueError:
                pass

        rows.append(
            {
                "item_id": item_id,
                "detection_id": detection_id,
                "clip_path": str(clip_path),
                "clip_start_s": clip_start,
                "clip_end_s": clip_end,
                "queue_order": 0,
                "project_root": str(layout.project_root),
                "aru_id": row.get("aru_id", ""),
                "source_audio_path": str(audio_path),
                "confidence": _safe_float(row.get("confidence", 0.0)),
                "recording_datetime": recording_datetime,
            }
        )

    queue_df = pd.DataFrame(rows)
    if not queue_df.empty:
        queue_df = queue_df.sort_values(["source_audio_path", "clip_start_s", "item_id"]).reset_index(drop=True)
        queue_df["queue_order"] = queue_df.index + 1

    layout.queue_dir.mkdir(parents=True, exist_ok=True)
    queue_df.to_csv(layout.queue_dir / "review_queue.csv", index=False)
    return queue_df


def stage_cache_spectrograms(layout: RunLayout, force: bool = False) -> dict:
    """Pre-render the canvas-band spectrogram PNG for every clip in the review queue.

    Idempotent by default: skips PNGs that already exist. Pass force=True to overwrite.
    Returns a small summary dict useful for CLI logging.
    """
    queue_path = layout.queue_dir / "review_queue.csv"
    if not queue_path.exists():
        raise FileNotFoundError(f"Missing review queue: {queue_path}")
    queue_df = pd.read_csv(queue_path)
    if queue_df.empty:
        return {"total": 0, "rendered": 0, "skipped": 0, "failed": 0}

    layout.spectrograms_dir.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    rendered = 0
    skipped = 0
    failed = 0
    for _, row in tqdm(
        queue_df.iterrows(),
        total=len(queue_df),
        desc="Spectrogram cache",
        unit="clip",
        dynamic_ncols=True,
    ):
        item_id = str(row["item_id"])
        clip_path = Path(str(row["clip_path"]))
        out_path = layout.spectrograms_dir / f"{item_id}.png"
        if out_path.exists() and not force:
            skipped += 1
            continue
        if not clip_path.exists():
            failed += 1
            continue
        if save_canvas_spectrogram(clip_path, out_path):
            rendered += 1
        else:
            failed += 1

    return {"total": int(len(queue_df)), "rendered": rendered, "skipped": skipped, "failed": failed}


def stage_config_snapshot(index: IndexConfig, birdnet: BirdNetConfig, clips: ClipConfig) -> dict:
    return {
        "index": index.model_dump(mode="json"),
        "birdnet": birdnet.model_dump(mode="json"),
        "clips": clips.model_dump(mode="json"),
    }
