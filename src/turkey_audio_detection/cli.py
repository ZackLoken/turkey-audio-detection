"""Command line entrypoints for the modular turkey pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from turkey_audio_detection.adjudication import adjudicate_to_csv
from turkey_audio_detection.config import BirdNetConfig, ClipConfig, IndexConfig
from turkey_audio_detection.layout import RunLayout, validate_project_layout
from turkey_audio_detection.manifest import build_stage_manifest, make_run_id, write_manifest
from turkey_audio_detection.stages import (
    stage_config_snapshot,
    stage_extract_clips,
    stage_index_data,
    stage_run_birdnet,
)


def _print(msg: str) -> None:
    print(msg, file=sys.stdout)


def _prepare_layout(project_root: Path, run_id: str) -> tuple[RunLayout, list[Path]]:
    aru_dirs = validate_project_layout(project_root)
    layout = RunLayout.from_project_root(project_root, run_id)
    layout.ensure_dirs()
    return layout, aru_dirs


def _cmd_index_data(args: argparse.Namespace) -> int:
    run_id = args.run_id or make_run_id()
    cfg = IndexConfig(
        deployment_start=args.deployment_start,
        timezone_name=args.timezone,
        latitude=args.latitude,
        longitude=args.longitude,
        prime_window_minutes_before=args.prime_before,
        prime_window_minutes_after=args.prime_after,
        wav_glob=args.wav_glob,
    )

    for project_root_raw in args.project_root:
        project_root = Path(project_root_raw).resolve()
        layout, aru_dirs = _prepare_layout(project_root, run_id)
        index_df, quarantine_df = stage_index_data(layout, cfg)

        manifest = build_stage_manifest(
            run_id=run_id,
            stage="index_data",
            project_root=project_root,
            config_snapshot={"index": cfg.model_dump(mode="json")},
            stage_outputs={
                "file_index_csv": str(layout.index_dir / "file_index.csv"),
                "quarantine_csv": str(layout.index_dir / "quarantine_filenames.csv"),
            },
            status="completed",
            input_file_count=int(len(index_df) + len(quarantine_df)),
        )
        manifest["aru_folder_count"] = len(aru_dirs)
        write_manifest(layout.manifests_dir / "index_data_manifest.json", manifest)

        _print(
            f"index-data completed for {project_root} | run_id={run_id} | "
            f"indexed={len(index_df)} quarantine={len(quarantine_df)}"
        )
    return 0


def _cmd_run_birdnet(args: argparse.Namespace) -> int:
    run_id = args.run_id or make_run_id()
    project_root = Path(args.project_root).resolve()
    layout, _aru_dirs = _prepare_layout(project_root, run_id)

    cfg = BirdNetConfig(
        min_confidence=args.min_confidence,
        prime_window_only=args.prime_window_only,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    out_df = stage_run_birdnet(layout, cfg)
    manifest = build_stage_manifest(
        run_id=run_id,
        stage="run_birdnet",
        project_root=project_root,
        config_snapshot={"birdnet": cfg.model_dump(mode="json")},
        stage_outputs={
            "detections_csv": str(layout.birdnet_dir / "detections_normalized.csv"),
        },
        status="completed",
        input_file_count=int(len(out_df)),
        birdnet_version="birdnetlib",
    )
    write_manifest(layout.manifests_dir / "run_birdnet_manifest.json", manifest)

    _print(f"run-birdnet completed for {project_root} | run_id={run_id} | detections={len(out_df)}")
    return 0


def _cmd_extract_clips(args: argparse.Namespace) -> int:
    run_id = args.run_id or make_run_id()
    project_root = Path(args.project_root).resolve()
    layout, _aru_dirs = _prepare_layout(project_root, run_id)

    cfg = ClipConfig(
        clip_duration_s=args.clip_duration,
        species_match_substring=args.species_match,
    )

    queue_df = stage_extract_clips(layout, cfg)
    manifest = build_stage_manifest(
        run_id=run_id,
        stage="extract_clips",
        project_root=project_root,
        config_snapshot={"clips": cfg.model_dump(mode="json")},
        stage_outputs={
            "review_queue_csv": str(layout.queue_dir / "review_queue.csv"),
            "clips_dir": str(layout.clips_dir),
        },
        status="completed",
        input_file_count=int(len(queue_df)),
    )
    write_manifest(layout.manifests_dir / "extract_clips_manifest.json", manifest)

    _print(f"extract-clips completed for {project_root} | run_id={run_id} | queue_items={len(queue_df)}")
    return 0


def _cmd_adjudicate(args: argparse.Namespace) -> int:
    run_id = args.run_id or make_run_id(prefix="adj")
    project_root = Path(args.project_root).resolve()
    layout, _aru_dirs = _prepare_layout(project_root, run_id)

    labels_dir = layout.review_labels_dir if args.labels_dir is None else Path(args.labels_dir).resolve()
    kappa_out = layout.review_adjudication_dir / "kappa_summary.csv"
    disagreements_out = layout.review_adjudication_dir / "disagreements.csv"

    kappa_df, disagreements_df = adjudicate_to_csv(
        labels_dir=labels_dir,
        kappa_out=kappa_out,
        disagreements_out=disagreements_out,
        include_unsure=args.include_unsure,
    )

    manifest = build_stage_manifest(
        run_id=run_id,
        stage="adjudicate",
        project_root=project_root,
        config_snapshot={"include_unsure": args.include_unsure},
        stage_outputs={
            "labels_dir": str(labels_dir),
            "kappa_summary_csv": str(kappa_out),
            "disagreements_csv": str(disagreements_out),
        },
        status="completed",
        input_file_count=int(len(kappa_df)),
    )
    write_manifest(layout.manifests_dir / "adjudicate_manifest.json", manifest)

    _print(
        f"adjudicate completed for {project_root} | run_id={run_id} | "
        f"pairs={len(kappa_df)} disagreements={len(disagreements_df)}"
    )
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    run_id = args.run_id or make_run_id()

    idx_args = argparse.Namespace(
        run_id=run_id,
        project_root=[args.project_root],
        deployment_start=args.deployment_start,
        timezone=args.timezone,
        latitude=args.latitude,
        longitude=args.longitude,
        prime_before=args.prime_before,
        prime_after=args.prime_after,
        wav_glob=args.wav_glob,
    )
    _cmd_index_data(idx_args)

    bird_args = argparse.Namespace(
        run_id=run_id,
        project_root=args.project_root,
        min_confidence=args.min_confidence,
        prime_window_only=args.prime_window_only,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    _cmd_run_birdnet(bird_args)

    clip_args = argparse.Namespace(
        run_id=run_id,
        project_root=args.project_root,
        clip_duration=args.clip_duration,
        species_match=args.species_match,
    )
    _cmd_extract_clips(clip_args)

    _print(f"run-all completed | run_id={run_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turkey-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index-data", help="Discover and index ARU recordings")
    p_index.add_argument("--project-root", action="append", required=True)
    p_index.add_argument("--run-id", required=False)
    p_index.add_argument("--deployment-start", default="2026-03-01")
    p_index.add_argument("--timezone", default="US/Eastern")
    p_index.add_argument("--latitude", type=float, default=41.7)
    p_index.add_argument("--longitude", type=float, default=-71.5)
    p_index.add_argument("--prime-before", type=float, default=90.0)
    p_index.add_argument("--prime-after", type=float, default=90.0)
    p_index.add_argument("--wav-glob", default="*.wav")
    p_index.set_defaults(func=_cmd_index_data)

    p_birdnet = sub.add_parser("run-birdnet", help="Run BirdNET and write detections")
    p_birdnet.add_argument("--project-root", required=True)
    p_birdnet.add_argument("--run-id", required=False)
    p_birdnet.add_argument("--min-confidence", type=float, default=0.1)
    p_birdnet.add_argument("--prime-window-only", action="store_true")
    p_birdnet.add_argument("--latitude", type=float, default=41.7)
    p_birdnet.add_argument("--longitude", type=float, default=-71.5)
    p_birdnet.set_defaults(func=_cmd_run_birdnet)

    p_clips = sub.add_parser("extract-clips", help="Create review clips from detections")
    p_clips.add_argument("--project-root", required=True)
    p_clips.add_argument("--run-id", required=False)
    p_clips.add_argument("--clip-duration", type=float, default=3.0)
    p_clips.add_argument("--species-match", default="Wild Turkey")
    p_clips.set_defaults(func=_cmd_extract_clips)

    p_adjudicate = sub.add_parser("adjudicate", help="Compute inter-rater agreement")
    p_adjudicate.add_argument("--project-root", required=True)
    p_adjudicate.add_argument("--run-id", required=False)
    p_adjudicate.add_argument("--labels-dir", required=False)
    p_adjudicate.add_argument("--include-unsure", action="store_true")
    p_adjudicate.set_defaults(func=_cmd_adjudicate)

    p_run_all = sub.add_parser("run-all", help="Run index -> BirdNET -> clip extraction")
    p_run_all.add_argument("--project-root", required=True)
    p_run_all.add_argument("--run-id", required=False)
    p_run_all.add_argument("--deployment-start", default="2026-03-01")
    p_run_all.add_argument("--timezone", default="US/Eastern")
    p_run_all.add_argument("--latitude", type=float, default=41.7)
    p_run_all.add_argument("--longitude", type=float, default=-71.5)
    p_run_all.add_argument("--prime-before", type=float, default=90.0)
    p_run_all.add_argument("--prime-after", type=float, default=90.0)
    p_run_all.add_argument("--wav-glob", default="*.wav")
    p_run_all.add_argument("--min-confidence", type=float, default=0.1)
    p_run_all.add_argument("--prime-window-only", action="store_true")
    p_run_all.add_argument("--clip-duration", type=float, default=3.0)
    p_run_all.add_argument("--species-match", default="Wild Turkey")
    p_run_all.set_defaults(func=_cmd_run_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
