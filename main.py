from __future__ import annotations

import argparse
from pathlib import Path

from data_process.sync_pipeline import export_synced_index, synchronize_session
from data_process.sync_inference import run_sync_and_infer


def _default_pipeline_config() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data_process"
        / "araya_eye"
        / "app"
        / "config"
        / "config_simple.yaml"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Development entrypoints for Tomii_ET.")
    sub = parser.add_subparsers(dest="command", required=True)

    sync_parser = sub.add_parser("sync", help="Synchronize a recorded calibration session.")
    sync_parser.add_argument("session_dir", type=Path, help="Path to the session directory to sync.")
    sync_parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0 / 30.0,
        help="Maximum timestamp delta (seconds) allowed between cameras (default: one frame at 30 FPS).",
    )
    sync_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path (default: <session_dir>/synced_index.csv).",
    )

    sync_run_parser = sub.add_parser(
        "sync-run",
        help="Synchronize a session, run the araya_eye pipeline, and store JSONL results.",
    )
    sync_run_parser.add_argument("session_dir", type=Path, help="Path to the session directory to process.")
    sync_run_parser.add_argument(
        "--config",
        type=Path,
        default=_default_pipeline_config(),
        help="Pipeline config YAML (default: app/config/config_simple.yaml).",
    )
    sync_run_parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0 / 30.0,
        help="Maximum timestamp delta (seconds) allowed between cameras.",
    )
    sync_run_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL output path (default: <session_dir>/synced_results.jsonl).",
    )
    sync_run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of synchronized batches to process.",
    )

    extract_parser = sub.add_parser(
        "extract-eyes",
        help="Synchronize a session, crop eyes via MediaPipe, and save images + metadata.",
    )
    extract_parser.add_argument("session_dir", type=Path, help="Session directory to process.")
    extract_parser.add_argument(
        "--config",
        type=Path,
        default=_default_pipeline_config(),
        help="Pipeline config YAML (default: app/config/config_simple.yaml).",
    )
    extract_parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0 / 30.0,
        help="Maximum timestamp delta (seconds) allowed between cameras.",
    )
    extract_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where eye crops + metadata will be written (default: <session_dir>/eye_crops).",
    )
    extract_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of synchronized batches to process.",
    )

    return parser


def _run_sync(args: argparse.Namespace) -> None:
    batches = synchronize_session(args.session_dir, tolerance_sec=args.tolerance)
    if not batches:
        print("No synchronized batches were created.")
        return
    output = args.output or (args.session_dir / "synced_index.csv")
    export_synced_index(batches, output)
    print(f"Synced {len(batches)} batches; wrote index to {output}")


def _run_sync_and_infer(args: argparse.Namespace) -> None:
    output, processed = run_sync_and_infer(
        args.session_dir,
        args.config,
        tolerance_sec=args.tolerance,
        output_path=args.output,
        limit_batches=args.limit,
    )
    print(f"Processed {processed} batches; results saved to {output}")


def _run_extract_eyes(args: argparse.Namespace) -> None:
    from data_process.sync_inference import run_sync_and_extract_eyes

    manifest, processed, saved = run_sync_and_extract_eyes(
        args.session_dir,
        args.config,
        tolerance_sec=args.tolerance,
        output_dir=args.output_dir,
        limit_batches=args.limit,
    )
    print(f"Processed {processed} batches, saved {saved} eye crops; metadata at {manifest}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "sync":
        _run_sync(args)
        return 0
    if args.command == "sync-run":
        _run_sync_and_infer(args)
        return 0
    if args.command == "extract-eyes":
        _run_extract_eyes(args)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    main()
