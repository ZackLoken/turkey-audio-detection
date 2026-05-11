## Plan: Turkey Audio Detection Refactor

Refactor the notebook workflow into a Windows-first, pip-installable Python package with a GUI-first review experience (Streamlit), while preserving reproducibility and modularity. The default user path will cover ARU-folder ingestion, BirdNET processing, and manual Tom/Hen/Background review; model training/tuning remains included but explicitly maintainer-only.

**Steps**
1. Phase 1: Define package boundaries and repository skeleton.
2. Create a `src/` package layout, standalone Streamlit app entrypoint, CLI entrypoint, configs, and tests directory. *Blocks all downstream steps.*
3. Add packaging metadata (`pyproject.toml`) for pip install from Git and define console scripts. *Depends on 1.*
4. Add `.gitignore` and artifact conventions so raw audio, generated clips, and run outputs stay local and out of Git by default. *Parallel with 2-3.*
5. Move existing notebook into `supplemental/` unchanged except minimal header note that it is archival-only. *Parallel with 2-4.*
6. Phase 2: Externalize config and deterministic run metadata.
7. Implement typed config models for paths, BirdNET settings, clip extraction, reviewer identity, and output schema. *Depends on 2-3.*
8. Enforce data directory contract: each project root must contain `data/ARU_*` subfolders for raw audio and `data/_outputs/` for generated artifacts; allow batching by passing one or more project roots per run. *Depends on 7.*
9. Implement run manifest writer (inputs hash/counts, config snapshot, timestamps, versions, output locations) and resume checkpoints. *Depends on 7-8.*
10. Phase 3: Modularize pipeline stages from the notebook.
11. Implement `index_data` stage (file discovery, filename parsing, deployment date filter, sunrise tagging). *Depends on 8.*
12. Implement BirdNET stage using birdnet Python API only, writing normalized detection CSV used by the review app queue. *Depends on 11.*
13. Implement clip extraction stage that materializes one review clip per BirdNET Wild Turkey detection without confidence filtering. *Depends on 12.*
14. Keep feature extraction/training/inference modules available behind maintainer commands; exclude them from standard collaborator walkthrough. *Parallel with 11-13 after shared utilities exist.*
15. Phase 4: Build GUI-first review workflow.
16. Create Streamlit review app as primary entrypoint: sequential queue, audio player + spectrogram, labels `Tom`, `Hen`, `Background`, `Skip`, autosave per label. *Depends on 13.*
17. Require reviewer identity at session start and write separate per-user CSV label files. *Depends on 16.*
18. Add adjudication utilities for inter-rater analysis (pairwise Cohen’s kappa and disagreement summaries across reviewers). *Depends on 17.*
19. Ensure app loads directly from BirdNET detection outputs and does not require notebook state. *Depends on 12, 16.*
20. Add stage-level error handling and recovery rules (missing audio, malformed filenames, BirdNET import/runtime failures, interrupted writes) with actionable logs and resumable reruns. *Depends on 9, 11-19.*
21. Phase 5: UX hardening, docs, and release readiness.
22. Write README with copy/paste Windows setup, pip install from private GitHub, one-run quickstart, and troubleshooting for audio/BirdNET installs. *Depends on 3, 9, 19-20.*
23. Add minimal tests covering filename parsing, ARU folder discovery, deterministic queue creation, label writing schema, kappa calculations, plus failure-path and resume-recovery tests for each stage. *Depends on 11-20.*
24. Add smoke-test commands for end-to-end local run (index → BirdNET → clip queue → Streamlit labels) and interrupted-run recovery validation. *Depends on 22-23.*
25. Prepare private repository publication checklist and collaborator onboarding steps for `ZackLoken/turkey-audio-detection`. *Depends on 22-24.*

**Relevant files**
- `turkey_gobble_audio_recognition.ipynb` — Source workflow to extract stage logic and constants.
- `gobbler_env.yml` — Existing dependency baseline to convert into pip-managed dependencies.
- `birdnet_detections.csv` — Current detection schema reference for normalized BirdNET output.
- `labels.csv` — Current labeling schema reference for per-reviewer CSV redesign.
- `features.h5` — Maintainer-only training artifact format reference.
- `turkey_classifier.pt` — Maintainer-only model artifact reference.

**Proposed consolidated data folder structure**
1. `data/ARU_*` remains the canonical raw-input location (read-only for pipeline stages).
2. All generated artifacts move under `data/_outputs/`.
3. Per-run stage artifacts are isolated by `run_id`.

```text
data/
	ARU_01/
		... raw audio files ...
	ARU_02/
		... raw audio files ...
	_outputs/
		runs/
			<run_id>/
				index/
					file_index.csv
					quarantine_filenames.csv
				birdnet/
					detections_normalized.csv
				queue/
					review_queue.csv
				clips/
					*.wav
				manifests/
					run_manifest.json
		review/
			labels/
				<reviewer_id>.csv
			adjudication/
				kappa_summary.csv
				disagreements.csv
```

4. Legacy root-level artifacts (`clips/`, `birdnet_detections.csv`, `labels.csv`) should be treated as migration inputs and not used as canonical output targets going forward.

**Schema contracts (must be frozen before implementation)**
1. Normalized detections CSV required columns: `detection_id`, `project_root`, `aru_id`, `audio_path`, `start_time_s`, `end_time_s`, `species_code`, `species_common_name`, `confidence`, `birdnet_model_version`, `source_filename`, `source_row_index`.
2. Review queue CSV required columns: `item_id`, `detection_id`, `clip_path`, `clip_start_s`, `clip_end_s`, `queue_order`, `project_root`, `aru_id`, `source_audio_path`.
3. Reviewer labels CSV required columns: `item_id`, `detection_id`, `reviewer_id`, `reviewer_name`, `label`, `label_timestamp_utc`, `session_id`, `app_version`.
4. Manifest JSON required fields: `run_id`, `started_at_utc`, `completed_at_utc`, `project_roots`, `config_snapshot`, `input_file_count`, `input_content_hash`, `stage_outputs`, `package_version`, `python_version`, `birdnet_version`, `status`.
5. Enum constraints: `label` must be one of `Tom`, `Hen`, `Background`, `Skip`; `Skip` is retained in raw labels but excluded by default from agreement and training datasets.

**Idempotency and resume rules**
1. Stable identity: `detection_id` is deterministic from `source_audio_path + start_time_s + end_time_s + species_code`; `item_id` is deterministic from `detection_id + clip_start_s + clip_end_s`.
2. Stage rerun behavior: indexing and BirdNET outputs are replace-on-rerun for the same `run_id`; clip extraction is upsert by `item_id`; labels are append-only per reviewer/session.
3. Config drift policy: if a rerun changes queue-defining config, write a new `run_id` and a new queue; do not mutate prior queue artifacts.
4. Partial-write recovery: each stage writes to temp files then atomic-renames; resume skips completed artifacts validated by manifest status and hash checks.
5. Conflict handling: duplicate `item_id` rows in labels are treated as latest-wins by `label_timestamp_utc` during exports, while preserving full raw history.

**Environment and dependency policy**
1. Windows-first target with Python 3.10+ baseline (validated in existing `gobbler` conda env).
2. Packaging via `pyproject.toml` with explicit minimum versions and a pinned `constraints.txt` for reproducible collaborator setup.
3. Install path of record: `pip install git+https://github.com/ZackLoken/turkey-audio-detection.git`.
4. Require version stamping in manifest for package, BirdNET, and Python to support reproducibility audits.

**Error handling and recovery expectations**
1. Missing/unreadable audio files: log as structured warnings with file path and stage, continue run, and summarize counts in manifest.
2. Malformed filenames/date parsing failures: route to quarantine report CSV for manual review instead of hard-failing full run.
3. BirdNET import/runtime errors: fail stage with actionable remediation text and preserve prior successful stage outputs.
4. Interrupted runs: resume from last completed manifest checkpoint without duplicating queue rows or relabeling completed items.
5. Label write errors in app: immediate user-visible error and local retry; never silently drop labels.

**Verification**
1. Fresh Windows virtualenv can install package via `pip install git+https://github.com/ZackLoken/turkey-audio-detection.git`.
2. CLI can discover `data/ARU_*` structure and generate deterministic file index CSV and run manifest under `data/_outputs/runs/<run_id>/`.
3. BirdNET stage runs from selected project root and produces normalized detections CSV with expected columns.
4. Clip extraction outputs one queue item per detection (no confidence filtering), and queue plus clips are reproducible under `data/_outputs/runs/<run_id>/` with fixed config.
5. Streamlit app launches locally, prompts for reviewer ID, and writes reviewer-specific CSV incrementally.
6. Two synthetic reviewer files can run through adjudication utility and output valid Cohen’s kappa summary CSV aligned by shared `item_id`.
7. Interrupted run recovery test confirms no duplicate queue items or duplicate labels after resume.
8. README quickstart is sufficient for collaborator to complete a full BirdNET + review run without notebook usage.

**Decisions**
- Included scope: Git-installable modular package, GUI-first review app, BirdNET processing on user-selected project roots containing `data/ARU_*`, reproducible manifests/checkpoints, reviewer-separated labels, CSV-only outputs consolidated under `data/_outputs/`.
- Included but maintainer-only: feature extraction, model training/tuning, model inference commands.
- Excluded from collaborator workflow: tuning guidance and automated model-development wizardry.
- Excluded for now: cross-platform support (Windows-only), Parquet/database output formats, confidence/date/ARU filtering in review app.
- Repository naming target: `turkey-audio-detection` under `ZackLoken` (private).

**Further Considerations**
1. Reviewer identity design recommendation: require `reviewer_id` + optional `reviewer_name` at app start, store both in each label row.
2. Label governance recommendation: preserve `Skip` values in raw labels but exclude from agreement and training datasets by default.
3. Publication workflow recommendation: create release tags only after smoke tests pass on a second Windows machine in the lab.