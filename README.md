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

- **Sidebar:** set project root, select run ID, enter reviewer name
- **Main panel:** audio player, full-band reference spectrogram, and a drawable mel spectrogram pinned to 200–6000 Hz (the turkey vocal band)
- **Region annotation:**
  - Pick the active label (`Tom` or `Hen`) — switch mid-clip to mix both kinds of calls in one save
  - Drag rectangles on the spectrogram around each call; the rectangle's x-extent encodes time, y-extent encodes frequency
  - Tick **Snap to full frequency band** when you don't want to bother with the y-axis (regions are persisted with `freq_min_hz=200, freq_max_hz=6000`)
  - Tick **Other birds present** if the clip contains non-turkey vocalizations alongside (or instead of) the turkey calls
  - Tick **Unsure** if you can't reliably tell — by default `adjudicate` excludes unsure rows from agreement stats
- **Save & Next** writes one row to `data/_outputs/review/labels/<reviewer_id>.csv` and advances. **Previous** revisits a labeled clip (regions re-render so you can edit them). **Reset canvas** clears the current clip's drawings. **Jump to first unlabeled** seeks to the next clip without a saved snapshot.
- Each CSV row contains: `item_id, detection_id, reviewer_id, reviewer_name, regions_json, other_birds_present, unsure, tom_present, hen_present, label_timestamp_utc, session_id`. `regions_json` is a JSON list of `{start_s, end_s, freq_min_hz, freq_max_hz, label}` objects; `tom_present` / `hen_present` are denormalized for cheap filtering.
- Run `adjudicate` after two reviewers finish to get pairwise Cohen's kappa **per attribute** (`tom_present` and `hen_present`) and a disagreements export tagged by attribute.

> **Note:** v0.2.0 broke the v0.1.0 label-CSV schema (single `Tom/Hen/Background/Skip` column → per-region annotation + denormalized booleans). Old label CSVs are not migrated.
