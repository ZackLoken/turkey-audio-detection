"""Headless review-app navigation test using Streamlit's AppTest."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from streamlit.testing.v1 import AppTest

APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "turkey_audio_detection"
    / "app.py"
)
RUN_ID = "run_test"
N_ITEMS = 5


def _make_project(root: Path) -> None:
    run_dir = root / "data" / "_outputs" / "runs" / RUN_ID
    (run_dir / "clips").mkdir(parents=True)
    (run_dir / "queue").mkdir()
    (root / "data" / "_outputs" / "review" / "labels").mkdir(parents=True)
    sr = 16000
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(3 * sr) / sr)
    rows = []
    for i in range(N_ITEMS):
        item_id = f"itm_{i:04d}"
        sf.write(run_dir / "clips" / f"{item_id}.wav", tone, sr)
        rows.append(
            {
                "item_id": item_id,
                "detection_id": f"det_{i:04d}",
                "clip_path": f"clips\\{item_id}.wav",
                "clip_start_s": 3.0 * i,
                "clip_end_s": 3.0 * i + 3.0,
                "queue_order": i + 1,
                "project_root": str(root),
                "aru_id": "ARU_01",
                "source_audio_path": str(root / "x.wav"),
                "confidence": 0.9,
                "recording_datetime": "2026-03-04 06:15:02",
            }
        )
    pd.DataFrame(rows).to_csv(
        run_dir / "queue" / "review_queue.csv", index=False
    )


def _enter(at: AppTest, root: Path) -> AppTest:
    at.sidebar.text_input[0].set_value(str(root)).run()
    at.sidebar.selectbox[0].select(RUN_ID)
    at.sidebar.text_input[1].set_value("tester")
    enter = [b for b in at.sidebar.button if b.label == "Enter"][0]
    return enter.click().run()


def _header(at: AppTest) -> str:
    return at.subheader[0].value


def test_goto_jumps_to_typed_detection(tmp_path) -> None:
    os.environ["_TURKEY_STREAMLIT_CHILD"] = "1"
    _make_project(tmp_path)
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at = _enter(at, tmp_path)
    assert at.exception == []
    assert _header(at) == f"Detection 1 of {N_ITEMS}"

    goto = [n for n in at.number_input if n.label == "Go to #"][0]
    goto.set_value(4)
    go = [b for b in at.button if b.label == "Go"][0]
    at = go.click().run()
    assert at.exception == []
    assert _header(at) == f"Detection 4 of {N_ITEMS}"

    labels = tmp_path / "data" / "_outputs" / "review" / "labels"
    assert list(labels.iterdir()) == []
