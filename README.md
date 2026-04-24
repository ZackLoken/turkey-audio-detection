# turkey-audio-detection

This repository provides a modular, pip-installable turkey audio pipeline with CLI stages and a Streamlit review app.

## Collaborator quickstart (Windows)

**Requirements:** Git for Windows, Anaconda or miniconda, access to this private repository.

1. Clone the repository:

   git clone https://github.com/ZackLoken/turkey-audio-detection.git
   cd turkey-audio-detection

2. Install from Git (SSH):

   pip install git+ssh://git@github.com/ZackLoken/turkey-audio-detection.git

   If you do not have an SSH key set up, use HTTPS instead:

   pip install git+https://github.com/ZackLoken/turkey-audio-detection.git

3. Run the full pipeline on your project root (folder containing data/ARU_*):

   turkey-pipeline run-all --project-root "C:\path\to\your\project"

4. Launch the review app and open http://localhost:8501:

   turkey-review

**Troubleshooting:**
- If `birdnetlib` fails to import, confirm ffmpeg is on your PATH or install via conda: `conda install -c conda-forge ffmpeg`
- If audio playback is silent in the review app, check that your WAV files are readable with `soundfile.info("yourfile.wav")`
- If BirdNET takes very long, add `--prime-window-only` to `run-birdnet` to limit to recordings near sunrise

## Quick start for developers

1. Activate your existing gobbler conda environment (Python 3.10) or create a Python 3.10+ environment.
2. Install in editable mode with dev dependencies and constraints:

   pip install -c constraints.txt -e .[dev]

3. Run tests:

   pytest

## Entry points

- CLI: turkey-pipeline
- Review app: turkey-review

## Stage-by-stage CLI usage

All outputs are written under data/_outputs.

- Index recordings:

   turkey-pipeline index-data --project-root .

- Run BirdNET on indexed files:

   turkey-pipeline run-birdnet --project-root . --run-id <run_id>

- Extract review clips and queue:

   turkey-pipeline extract-clips --project-root . --run-id <run_id>

- Run full non-review pipeline in one command:

   turkey-pipeline run-all --project-root .

- Compute inter-rater adjudication from reviewer label files:

   turkey-pipeline adjudicate --project-root .

## Review app

1. Launch app:

    turkey-review

2. In sidebar, provide project root, select run ID, and enter reviewer identity.
3. Label queue items as Tom, Hen, Background, or Skip.
4. Labels autosave to data/_outputs/review/labels/<reviewer_id>.csv.

## Core modules

- src/turkey_audio_detection/cli.py
- src/turkey_audio_detection/app.py
- src/turkey_audio_detection/stages.py
- src/turkey_audio_detection/adjudication.py
- src/turkey_audio_detection/schemas.py
- src/turkey_audio_detection/ids.py
- src/turkey_audio_detection/manifest.py
