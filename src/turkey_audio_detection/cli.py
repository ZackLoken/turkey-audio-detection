"""Command line entrypoints for the modular turkey pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from turkey_audio_detection.adjudication import adjudicate_to_csv
from turkey_audio_detection.config import BirdNetConfig, ClipConfig, IndexConfig, SedInferConfig, SedTrainConfig
from turkey_audio_detection.layout import RunLayout, validate_project_layout
from turkey_audio_detection.manifest import build_stage_manifest, make_run_id, write_manifest
from turkey_audio_detection.stages import (
    stage_cache_spectrograms,
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

    if not getattr(args, "skip_spectrogram_cache", False) and not queue_df.empty:
        _print(f"caching spectrograms for {project_root} | run_id={run_id} ...")
        summary = stage_cache_spectrograms(layout)
        _print(
            f"cache-spectrograms completed | rendered={summary['rendered']} "
            f"skipped={summary['skipped']} failed={summary['failed']}"
        )
    return 0


def _cmd_cache_spectrograms(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    layout = RunLayout.from_project_root(project_root, args.run_id)
    layout.spectrograms_dir.mkdir(parents=True, exist_ok=True)
    summary = stage_cache_spectrograms(layout, force=args.force)
    _print(
        f"cache-spectrograms completed for {project_root} | run_id={args.run_id} | "
        f"rendered={summary['rendered']} skipped={summary['skipped']} failed={summary['failed']}"
    )
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


def _cmd_train(args: argparse.Namespace) -> int:
    from turkey_audio_detection.sed_training import train_sed

    project_root = Path(args.project_root).resolve()
    model_id = args.model_id or make_run_id(prefix="model")
    cfg = SedTrainConfig(
        run_ids=list(args.run_id),
        model_id=model_id,
        clip_duration_s=args.clip_duration,
        include_non_consensus=args.include_non_consensus,
        site_map_path=args.site_map_path,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        holdout_years=list(args.holdout_year or []),
        pretrained=not args.no_pretrained,
        n_stages=args.n_stages,
        temporal=args.temporal,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
        base_lr=args.base_lr,
        base_batch_size=args.base_batch_size,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup_alpha,
        specaugment_enabled=not args.no_specaugment,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    result = train_sed(cfg, project_root)
    _print(
        f"train completed | model_id={result['model_id']} | "
        f"n_train={result['n_train']} n_val={result['n_val']} n_test={result['n_test']} | "
        f"best_score={result['best_score']:.3f}"
    )
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from turkey_audio_detection.sed_inference import infer_sed

    project_root = Path(args.project_root).resolve()
    inference_id = args.inference_id or make_run_id(prefix="inf")
    cfg = SedInferConfig(
        model_id=args.model_id,
        run_id=args.run_id,
        inference_id=inference_id,
        candidate_window_duration_s=args.candidate_window_duration,
        min_event_duration_s=args.min_event_duration,
        merge_gap_s=args.merge_gap,
        species_match_substring=args.species_match,
        batch_size=args.batch_size,
        site_map_path=args.site_map_path,
    )
    result = infer_sed(cfg, project_root)
    _print(
        f"classify completed | inference_id={result['inference_id']} | "
        f"events={result['n_events_total']}"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import torch

    from turkey_audio_detection.config import SedTrainConfig
    from turkey_audio_detection.evaluation import evaluate_table, evaluation_to_rows
    from turkey_audio_detection.layout import model_dir
    from turkey_audio_detection.sed_inference import load_sed_model
    from turkey_audio_detection.sed_training import site_year_split
    from turkey_audio_detection.sites import attach_site, load_site_map
    from turkey_audio_detection.training_labels import build_training_table

    project_root = Path(args.project_root).resolve()
    table = build_training_table(project_root, list(args.run_id), include_non_consensus=args.include_non_consensus)
    split_cfg = SedTrainConfig(
        site_map_path=args.site_map_path, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, seed=args.seed,
    )
    table = site_year_split(attach_site(table, load_site_map(project_root / args.site_map_path)), split_cfg)
    test_df = table[table["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        _print("evaluate: test split is empty")
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_sed_model(model_dir(project_root, args.model_id) / "checkpoint.pt", device)
    result = evaluate_table(
        model, test_df, payload, device,
        iou_thresholds=tuple(args.iou_thresholds), seg_s=args.seg_s, clip_duration_s=args.clip_duration,
    )
    out = model_dir(project_root, args.model_id) / "eval.csv"
    evaluation_to_rows(result).to_csv(out, index=False)
    _print(f"evaluate completed | model_id={args.model_id} | n_test={len(test_df)} | wrote {out}")
    return 0


def _cmd_hpo(args: argparse.Namespace) -> int:
    from turkey_audio_detection.config import SedTrainConfig
    from turkey_audio_detection.hpo import run_hpo
    from turkey_audio_detection.training_labels import build_training_table

    project_root = Path(args.project_root).resolve()
    table = build_training_table(project_root, list(args.run_id), include_non_consensus=args.include_non_consensus)
    base = SedTrainConfig(
        run_ids=list(args.run_id), site_map_path=args.site_map_path,
        num_workers=args.num_workers, seed=args.seed,
    )
    study = run_hpo(table, project_root, base, n_trials=args.n_trials, storage=args.storage, study_name=args.study_name)
    _print(f"hpo completed | trials={len(study.trials)} | best_value={study.best_value:.3f} | best_params={study.best_params}")
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
        skip_spectrogram_cache=getattr(args, "skip_spectrogram_cache", False),
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
    p_clips.add_argument("--skip-spectrogram-cache", action="store_true",
                         help="Skip pre-rendering review spectrograms (review app will compute on demand)")
    p_clips.set_defaults(func=_cmd_extract_clips)

    p_cache = sub.add_parser(
        "cache-spectrograms",
        help="Pre-render review-clip spectrograms to PNG so the review app loads instantly",
    )
    p_cache.add_argument("--project-root", required=True)
    p_cache.add_argument("--run-id", required=True)
    p_cache.add_argument("--force", action="store_true",
                         help="Re-render every PNG even if one already exists")
    p_cache.set_defaults(func=_cmd_cache_spectrograms)

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
    p_run_all.add_argument("--skip-spectrogram-cache", action="store_true",
                           help="Skip pre-rendering review spectrograms")
    p_run_all.set_defaults(func=_cmd_run_all)

    p_train = sub.add_parser("train", help="Train the frame-level SED model on reviewed labels")
    p_train.add_argument("--project-root", required=True)
    p_train.add_argument("--run-id", action="append", required=True,
                         help="One or more run IDs whose review queue + labels feed training")
    p_train.add_argument("--model-id")
    p_train.add_argument("--clip-duration", type=float, default=3.0)
    p_train.add_argument("--site-map-path", default="data/site_map.csv")
    p_train.add_argument("--val-fraction", type=float, default=0.15)
    p_train.add_argument("--test-fraction", type=float, default=0.15)
    p_train.add_argument("--holdout-year", action="append", type=int, help="Year(s) held out for test")
    p_train.add_argument("--include-non-consensus", action="store_true")
    p_train.add_argument("--no-pretrained", action="store_true")
    p_train.add_argument("--n-stages", type=int, default=2)
    p_train.add_argument("--temporal", choices=["bigru", "tcn"], default="bigru")
    p_train.add_argument("--hidden-size", type=int, default=256)
    p_train.add_argument("--n-layers", type=int, default=2)
    p_train.add_argument("--dropout", type=float, default=0.2)
    p_train.add_argument("--loss", choices=["focal", "bce"], default="focal")
    p_train.add_argument("--focal-gamma", type=float, default=2.0)
    p_train.add_argument("--pos-weight", type=float, default=1.0)
    p_train.add_argument("--base-lr", type=float, default=1e-3)
    p_train.add_argument("--base-batch-size", type=int, default=32)
    p_train.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--mixup-alpha", type=float, default=0.0)
    p_train.add_argument("--no-specaugment", action="store_true")
    p_train.add_argument("--early-stop-patience", type=int, default=5)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=_cmd_train)

    p_classify = sub.add_parser("classify", help="Run a trained SED model on a run's BirdNET candidates")
    p_classify.add_argument("--project-root", required=True)
    p_classify.add_argument("--model-id", required=True)
    p_classify.add_argument("--run-id", required=True, help="Run whose BirdNET candidates gate inference")
    p_classify.add_argument("--inference-id")
    p_classify.add_argument("--candidate-window-duration", type=float, default=3.0)
    p_classify.add_argument("--min-event-duration", type=float, default=0.1)
    p_classify.add_argument("--merge-gap", type=float, default=0.2)
    p_classify.add_argument("--species-match", default="Wild Turkey")
    p_classify.add_argument("--batch-size", type=int, default=16)
    p_classify.add_argument("--site-map-path", default="data/site_map.csv")
    p_classify.set_defaults(func=_cmd_classify)

    p_eval = sub.add_parser("evaluate", help="Event-level + segment evaluation on the test split")
    p_eval.add_argument("--project-root", required=True)
    p_eval.add_argument("--model-id", required=True)
    p_eval.add_argument("--run-id", action="append", required=True)
    p_eval.add_argument("--include-non-consensus", action="store_true")
    p_eval.add_argument("--site-map-path", default="data/site_map.csv")
    p_eval.add_argument("--val-fraction", type=float, default=0.15)
    p_eval.add_argument("--test-fraction", type=float, default=0.15)
    p_eval.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p_eval.add_argument("--seg-s", type=float, default=1.0)
    p_eval.add_argument("--clip-duration", type=float, default=3.0)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_hpo = sub.add_parser("hpo", help="Optuna hyperparameter search for the SED model")
    p_hpo.add_argument("--project-root", required=True)
    p_hpo.add_argument("--run-id", action="append", required=True)
    p_hpo.add_argument("--include-non-consensus", action="store_true")
    p_hpo.add_argument("--site-map-path", default="data/site_map.csv")
    p_hpo.add_argument("--n-trials", type=int, default=20)
    p_hpo.add_argument("--storage", help="SQLite path for a resumable study")
    p_hpo.add_argument("--study-name", default="sed_hpo")
    p_hpo.add_argument("--num-workers", type=int, default=2)
    p_hpo.add_argument("--seed", type=int, default=42)
    p_hpo.set_defaults(func=_cmd_hpo)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


def main_train() -> int:
    """Console-script entry point for `turkey-train` — routes to the `train` subcommand."""
    sys.argv.insert(1, "train")
    return main()


def main_classify() -> int:
    """Console-script entry point for `turkey-classify` — routes to the `classify` subcommand."""
    sys.argv.insert(1, "classify")
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
