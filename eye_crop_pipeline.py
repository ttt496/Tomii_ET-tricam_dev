from __future__ import annotations

import argparse
import sys
from pathlib import Path

# CUDA 12.6環境を自動設定（このリポジトリでのみ有効）
sys.path.insert(0, str(Path(__file__).parent))
try:
    from scripts.setup_cuda_env import setup_cuda_env
    setup_cuda_env()
except ImportError:
    pass  # スクリプトが見つからない場合はスキップ

from data_process.sync_inference import run_sync_and_extract_eyes


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data_process"
        / "araya_eye"
        / "app"
        / "config"
        / "config_simple.yaml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize a recorded session and extract MediaPipe eye crops."
    )
    parser.add_argument("session_dir", type=Path, help="Session directory containing camera recordings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to the pipeline config YAML (default: app/config/config_simple.yaml).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0 / 30.0,
        help="Maximum timestamp delta (seconds) allowed between cameras (default: 1 frame at 30 FPS).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where crops + metadata are stored (default: original session directory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of synchronized batches to process.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path, processed_batches, saved_images = run_sync_and_extract_eyes(
        session_dir=args.session_dir,
        config_path=args.config,
        tolerance_sec=args.tolerance,
        output_dir=args.output_dir,
        limit_batches=args.limit,
    )

    print(
        f"Processed {processed_batches} batches, saved {saved_images} eye crops.\n"
        f"Metadata written to: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    USE_MANUAL_ARGS = True  # Set True for quick testing without CLI arguments.
    if USE_MANUAL_ARGS:
        BASE_DIR = Path("C:\\Users\\demo\\Tomii_ET-tricam_dev\\data")
        SESSION_DIRS = list(BASE_DIR.glob("*/*"))
        for SESSION_DIR in SESSION_DIRS:
            CONFIG = _default_config_path()
            MANIFEST, PROCESSED, SAVED = run_sync_and_extract_eyes(
                session_dir=SESSION_DIR,
                config_path=CONFIG,
                tolerance_sec=1.0 / 30.0,
                output_dir=None,
                limit_batches=None,
            )
            print(
                f"[manual] Processed {PROCESSED} batches, saved {SAVED} eye crops.\n"
                f"Metadata written to: {MANIFEST}"
            )
    else:
        raise SystemExit(main())
