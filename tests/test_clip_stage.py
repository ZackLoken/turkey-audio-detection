from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from turkey_audio_detection.config import ClipConfig
from turkey_audio_detection.layout import RunLayout
from turkey_audio_detection.stages import stage_extract_clips


def test_stage_extract_clips_builds_queue(tmp_path: Path) -> None:
    layout = RunLayout.from_project_root(tmp_path, "run_20260424T010101Z")
    layout.ensure_dirs()

    audio_dir = tmp_path / "data" / "ARU_01"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "2MA09358_20260310_050001.wav"

    sr = 16000
    t = np.linspace(0, 6, sr * 6, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_path), tone.astype("float32"), sr)

    detections = pd.DataFrame(
        [
            {
                "detection_id": "det_a",
                "project_root": str(tmp_path),
                "aru_id": "ARU_01",
                "audio_path": str(wav_path),
                "start_time_s": 1.0,
                "end_time_s": 2.0,
                "species_code": "Meleagris gallopavo",
                "species_common_name": "Wild Turkey",
                "confidence": 0.9,
                "birdnet_model_version": "birdnetlib",
                "source_filename": wav_path.name,
                "source_row_index": 0,
            }
        ]
    )
    detections.to_csv(layout.birdnet_dir / "detections_normalized.csv", index=False)

    queue_df = stage_extract_clips(layout, ClipConfig())

    assert len(queue_df) == 1
    assert queue_df.iloc[0]["item_id"].startswith("itm_")
    clip_path = Path(queue_df.iloc[0]["clip_path"])
    assert clip_path.exists()
    assert (layout.queue_dir / "review_queue.csv").exists()
