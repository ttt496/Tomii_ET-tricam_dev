"""Utility to re-encode CSV files in-place or to a new file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def reencode_csv(
    path: Path,
    *,
    input_encoding: str = "utf-8",
    output_encoding: str = "utf-8",
    newline: Optional[str] = None,
    output_path: Optional[Path] = None,
    create_backup: bool = False,
) -> Path:
    """Read ``path`` using ``input_encoding`` and write it with ``output_encoding``.

    Parameters
    ----------
    path:
        CSV ファイルのパス。
    input_encoding:
        読み込み時のエンコーディング。
    output_encoding:
        書き込み時のエンコーディング。
    newline:
        書き込み時の改行コード。 ``None`` のままなら既定動作。
    output_path:
        出力先を指定したい場合。省略すると上書き。
    create_backup:
        上書き時に ``.bak`` バックアップを作成する。
    """

    source = path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    text = source.read_text(encoding=input_encoding)

    destination = output_path.expanduser().resolve() if output_path else source

    if create_backup and output_path is None:
        backup = source.with_suffix(source.suffix + ".bak")
        if not backup.exists():
            backup.write_text(text, encoding=input_encoding)

    destination.write_text(text, encoding=output_encoding, newline=newline)
    return destination


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-encode a CSV file.")
    parser.add_argument("path", type=Path, help="Path to the source CSV file.")
    parser.add_argument("--input-encoding", default="utf-8", help="Encoding to use when reading (default: utf-8).")
    parser.add_argument("--output-encoding", default="utf-8", help="Encoding to use when writing (default: utf-8).")
    parser.add_argument("--newline", choices=["\n", "\r\n", "\r", "infer"], default="infer", help="Override newline characters.")
    parser.add_argument("--output", type=Path, help="Optional destination path; overwrites in place when omitted.")
    parser.add_argument("--backup", action="store_true", help="Create a .bak backup when overwriting in place.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(None)
    newline = None if args.newline == "infer" else args.newline
    try:
        dest = reencode_csv(
            args.path,
            input_encoding=args.input_encoding,
            output_encoding=args.output_encoding,
            newline=newline,
            output_path=args.output,
            create_backup=args.backup,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Error: {exc}") from exc
    else:
        print(f"Saved re-encoded CSV to {dest}")
