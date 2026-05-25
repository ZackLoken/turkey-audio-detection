# turkey-audio-detection

Modular pipeline for detecting and reviewing Wild Turkey vocalizations in ARU (autonomous recording unit) audio. Runs BirdNET on WAV recordings, extracts candidate clips, and provides a Streamlit app for labeling and inter-rater adjudication.

The labeling stage produces a reviewed dataset of time–frequency regions on each 3-second candidate clip, each region tagged Tom or Hen. Clips with no turkey are saved with an empty region list (with an optional "other birds present" flag). BirdNET performs the candidate detection; training a turkey-specific classifier on the reviewed regions is a planned next phase.

## Data layout

Place raw audio under `data/ARU_<id>/` inside your project root. WAV files must follow the naming convention `<device_id>_YYYYMMDD_HHMMSS.wav` (e.g. `2MA09358_20260304_051500.wav`). Files that don't match are quarantined rather than crashing the pipeline.

All generated outputs are written under `data/_outputs/` and are excluded from version control.

```
project-root/
├── data/
│   ├── ARU_01/          ← raw audio (read-only, not committed)
│   │   └── *.wav
│   └── _outputs/        ← generated (not committed)
│       ├── runs/<run_id>/
│       └── review/
```

## Collaborator quickstart (Windows)

**Requirements:** Git for Windows, Anaconda or Miniconda.

1. Clone the repository:

   ```
   git clone https://github.com/ZackLoken/turkey-audio-detection.git
   cd turkey-audio-detection
   ```

2. Create and activate the conda environment:

   ```
   conda env create -f gobbler.yml
   conda activate gobbler
   ```

3. Install the package into the active environment:

   ```
   pip install -e .
   ```

4. Put your audio data in `data/ARU_01/` (or `data/ARU_02/`, etc.) and run the full pipeline:

   ```
   turkey-pipeline run-all --project-root "C:\path\to\your\project"
   ```

   The pipeline prints a `run_id` (e.g. `run_20260424T205153Z`) when it finishes — **note it**, you'll need it for the review app.

   > **Note:** TensorFlow and BirdNET print verbose INFO/WARNING messages to the console during startup. These are normal and can be ignored. A progress bar shows per-file status while BirdNET is running.

5. Launch the review app:

   ```
   turkey-review
   ```

   In the sidebar enter your **project root**, the **run ID** from step 4, and a **reviewer name**. For each clip, draw rectangles on the spectrogram around any Tom or Hen calls (switch the active-label radio between Tom and Hen as needed), tick **Other birds present** / **Unsure** when relevant, then click **Save & Next**. See the [Review app](#review-app) section below for details.

**Troubleshooting:**
- If `birdnetlib` fails to import, confirm ffmpeg is on your PATH: `conda install -c conda-forge ffmpeg`
- If BirdNET is slow, add `--prime-window-only` to limit processing to recordings near sunrise
- If audio playback is silent, check that your WAV files are readable: `python -c "import soundfile; print(soundfile.info('yourfile.wav'))"`

## Stage-by-stage CLI usage

All outputs are written under `data/_outputs/runs/<run_id>/`.

```
# Index recordings and compute sunrise windows
turkey-pipeline index-data --project-root .

# Run BirdNET on indexed files (use run_id from previous step)
turkey-pipeline run-birdnet --project-root . --run-id <run_id>

# Extract 3-second review clips for Wild Turkey detections
turkey-pipeline extract-clips --project-root . --run-id <run_id>

# Or run all three stages at once
turkey-pipeline run-all --project-root .

# Compute inter-rater Cohen's kappa from reviewer label files
turkey-pipeline adjudicate --project-root .
```

## Review app

```
turkey-review
```

![Turkey Call Labeler GUI](GUI_Labeler.png)

- **Sidebar:** set project root, select run ID, enter reviewer name. The collapsible *How to label* expander lives below.
- **Main panel:** custom audio control (play/pause, `seconds.milliseconds` timestamp, scrubber) and a mel spectrogram pinned to 50–14000 Hz with labeled time and frequency axes. The vertical black bar over the spectrogram tracks the audio playhead.
- **Region annotation:**
  - Toggle the active label (`Tom` = lime green, `Hen` = royal blue) between drawings to mix both call types on one clip
  - Drag rectangles on the spectrogram around each call; the rectangle's x-extent encodes time, y-extent encodes frequency. Each box is auto-previewed (audio bandpass-filtered to the box's frequency bounds) the moment you finish drawing
  - **Click** an existing rectangle to replay its band-limited audio; **double-click** to delete it
  - Tick **Other birds present** when any non-turkey bird is audible in the clip (most BirdNET candidates have at least one)
  - Tick **Unsure** when you can't reliably tell whether a turkey is in the clip — these rows are excluded from agreement stats by default
- **Save & Next** writes one row to `data/_outputs/review/labels/<reviewer_id>.csv` and advances. Saving on an empty canvas creates an explicit *no turkey* label. **Previous** revisits a labeled clip (regions re-render so you can edit them). **Reset canvas** clears drawings *and* removes any saved snapshot for the clip so it becomes unlabeled again. **Jump to first unlabeled** seeks to the next clip without a saved snapshot.
- Each CSV row contains: `item_id, detection_id, reviewer_id, reviewer_name, regions_json, other_birds_present, unsure, tom_present, hen_present, label_timestamp_utc, session_id`. `regions_json` is a JSON list of `{start_s, end_s, freq_min_hz, freq_max_hz, label}` objects; `tom_present` / `hen_present` are denormalized for cheap filtering.
- Run `adjudicate` after two reviewers finish to get pairwise Cohen's kappa **per attribute** (`tom_present` and `hen_present`) and a disagreements export tagged by attribute.

> **Note:** v0.2.0 broke the v0.1.0 label-CSV schema (single `Tom/Hen/Background/Skip` column → per-region annotation + denormalized booleans). Old label CSVs are not migrated.

## Training and classification (v0.3.0)

Once reviewers have produced labeled clips, train a region-level sound-event-detection (SED) model and run it on raw ARU recordings.

```
# Train on one or more runs' labels. Aggregates per-reviewer CSVs via majority vote,
# stratifies a train/val/test split by (ARU id, recording date) to prevent
# same-recording leakage, then finetunes a PANNs CNN14 backbone with a U-Net
# decoder + 2-channel (Tom, Hen) output head.
turkey-train --project-root . --run-id <run_id> [--run-id <run_id> ...] \
             --epochs 60 --batch-size 32 --learning-rate 1e-4

# Run a trained model on raw audio files. Slides 3-second windows across each
# WAV, stitches per-window 2D probability maps, then extracts connected-component
# events as `(start_s, end_s, freq_min_hz, freq_max_hz, label, score)` rows —
# matching the same shape as human region labels.
turkey-classify --project-root . --model-id <model_id> \
                --audio-glob "data/ARU_*/**/*.wav"
```

**Outputs:**
- `data/_outputs/models/<model_id>/checkpoint.pt` — best-validation model state
- `data/_outputs/models/<model_id>/train_metrics.csv` — epoch-level loss + per-class precision/recall/F1
- `data/_outputs/models/<model_id>/splits.csv` — which items went to train / val / test
- `data/_outputs/inference/<inference_id>/events/<source_filename>.csv` — one events CSV per input WAV
- `data/_outputs/inference/<inference_id>/summary.csv` — per-file event counts + error log

**Architecture:**
- Encoder: PANNs CNN14 pretrained on AudioSet (auto-downloaded on first use to `~/panns_data/`; ~300 MB)
- Decoder: small U-Net with skip connections, upsampling back to (n_mels × n_frames)
- Output: per-(mel-bin, frame) sigmoid logits for Tom and Hen
- Loss: pixel-level binary cross-entropy with positive-class weighting
- Training augmentation: SpecAugment + Mixup + background-mix from negative clips

**Aggregation:** by default `turkey-train` trains only on clips where reviewers reached consensus on `tom_present` and `hen_present`. Pass `--include-non-consensus` to train on disagreement-flagged clips as well.
