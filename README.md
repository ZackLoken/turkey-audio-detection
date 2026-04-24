# turkey-audio-detection

Modular pipeline for detecting and reviewing Wild Turkey vocalizations in ARU (autonomous recording unit) audio. Runs BirdNET on WAV recordings, extracts candidate clips, and provides a Streamlit app for labeling and inter-rater adjudication.

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

**Requirements:** Git for Windows, Anaconda or Miniconda, access to this private repository.

1. Clone the repository (handles private-repo auth via Git credential manager):

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

   In the sidebar enter your **project root**, the **run ID** from step 4, and a **reviewer ID** (e.g. your name). Label each clip as Tom, Hen, Background, or Skip — labels autosave as you go.

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

- **Sidebar:** set project root, select run ID, enter reviewer ID and display name
- **Main panel:** audio player + mel spectrogram for each queued clip
- **Buttons:** Tom / Hen / Background / Skip — autosaves and advances to next clip
- Labels are written to `data/_outputs/review/labels/<reviewer_id>.csv`
- Run `adjudicate` after two reviewers finish to get pairwise Cohen's kappa and a disagreements export
