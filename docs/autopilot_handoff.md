# Autopilot Implementation Handoff

This file defines autopilot execution and validation steps for the implemented stage pipeline.

## Pre-flight

1. Confirm Python 3.10+ environment (existing gobbler env is valid).
2. Install package in editable mode.
3. Run pytest to confirm pipeline health.
4. Validate project-root data layout:
	- data/ARU_* contains raw WAV inputs only.
	- data/_outputs/runs/<run_id>/ stores stage artifacts.
	- data/_outputs/review/ stores reviewer labels and adjudication exports.

## Implemented commands

1. turkey-pipeline index-data --project-root . [--run-id ...]
2. turkey-pipeline run-birdnet --project-root . --run-id <run_id>
3. turkey-pipeline extract-clips --project-root . --run-id <run_id>
4. turkey-pipeline run-all --project-root . [--run-id ...]
5. turkey-pipeline adjudicate --project-root .
6. turkey-review

## Remaining hardening work

1. Add resume checkpoint status transitions (in-progress, failed, completed) in manifests.
2. Add per-file BirdNET error capture so one broken audio file does not fail entire stage.
3. Add stricter schema validation before writing stage outputs.
4. Add integration smoke test that executes index -> birdnet -> clips on a tiny fixture dataset.
5. Add optional maintainer-only commands for model feature extraction/training behind explicit flags.

## Definition of done gates

1. Unit tests pass for deterministic IDs, filename parsing, queue generation, and kappa logic.
2. End-to-end smoke run succeeds on one project root.
3. Interrupted run recovery test succeeds.
4. README quickstart supports collaborator run without notebook usage.
