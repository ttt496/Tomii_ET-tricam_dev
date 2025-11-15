#!/usr/bin/env python3
"""
Enumerate camera codec / resolution / FPS combinations via ffmpeg.

Examples
--------
    # macOS / AVFoundation (indexes 0,1,...)
    python recording/ffmpeg_probe.py --devices 0 1

    # Linux / v4l2
    python recording/ffmpeg_probe.py --devices /dev/video0 /dev/video2

    # Windows / DirectShow (device names must match ffmpeg -list_devices output)
    python recording/ffmpeg_probe.py --devices "USB Camera" "Logitech BRIO"
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class FormatRecord:
    codec: str
    pixel_fmt: str
    resolution: str
    fps: Optional[str]
    index: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "codec": self.codec,
            "pixel_format": self.pixel_fmt,
            "resolution": self.resolution,
            "fps": self.fps,
        }


@dataclass
class DeviceResult:
    device: str
    backend: str
    formats: List[FormatRecord]
    raw_output: str

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "backend": self.backend,
            "formats": [fmt.to_dict() for fmt in self.formats],
            "raw_output": self.raw_output,
        }


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------


def _detect_backend() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "avfoundation"
    if sys_name == "windows":
        return "dshow"
    return "v4l2"


def _format_device_arg(backend: str, device: str) -> str:
    if backend == "avfoundation":
        return device if ":" in device else f"{device}:none"
    if backend == "dshow":
        return f"video={device}"
    return device  # v4l2 and others


def _build_ffmpeg_command(backend: str, device: str, ffmpeg_path: str) -> List[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "info",
        "-f",
        backend,
        "-list_formats",
        "all",
        "-i",
        _format_device_arg(backend, device),
    ]


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------


# Patterns seen across avfoundation/dshow/v4l2 outputs
PATTERNS = [
    # [0] 1920x1080 30.00 fps mjpeg
    re.compile(
        r"""
        ^\s*\[\s*(?P<index>\d+)\s*\]\s+
        (?P<res>\d+x\d+)\s+
        (?P<fps>[0-9.]+)\s+fps\s+
        (?P<codec>[A-Za-z0-9_]+)
        (?:\s+\((?P<pixel>[A-Za-z0-9_ ]+)\))?
        """,
        re.VERBOSE,
    ),
    # Raw       :   yuyv422 :   YUYV 4:2:2 : 640x480 30.00 fps
    re.compile(
        r"""
        :\s*(?P<pixel>[A-Za-z0-9_]+)\s*:\s*
        (?P<codec>[A-Za-z0-9_ .+-]+?)\s*:\s*
        (?P<res>\d+x\d+)\s+
        (?P<fps>[0-9./]+)\s*fps
        """,
        re.VERBOSE,
    ),
    # fallback: codec ... 640x480 @30fps
    re.compile(
        r"""
        (?P<codec>[A-Za-z0-9_]+)[^\d]*
        (?P<res>\d+x\d+)[^\d]*
        @?\s*(?P<fps>[0-9./]+)\s*fps
        """,
        re.VERBOSE,
    ),
]


def _parse_formats(stdout: str) -> List[FormatRecord]:
    records: List[FormatRecord] = []
    seen = set()
    for line in stdout.splitlines():
        stripped = line.strip()
        if "fps" not in stripped:
            continue
        for pattern in PATTERNS:
            match = pattern.search(stripped)
            if not match:
                continue
            res = match.group("res")
            fps = match.group("fps")
            codec = (match.group("codec") or "").strip()
            pixel = (match.groupdict().get("pixel") or codec).strip()
            idx_raw = match.groupdict().get("index")
            key = (codec.lower(), pixel.lower(), res, fps)
            if key in seen:
                break
            seen.add(key)
            records.append(
                FormatRecord(
                    codec=codec or pixel,
                    pixel_fmt=pixel or codec,
                    resolution=res,
                    fps=fps,
                    index=int(idx_raw) if idx_raw is not None else None,
                )
            )
            break
    return records


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def probe_device(backend: str, device: str, ffmpeg_path: str) -> DeviceResult:
    cmd = _build_ffmpeg_command(backend, device, ffmpeg_path)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout or ""
    formats = _parse_formats(output)
    return DeviceResult(device=device, backend=backend, formats=formats, raw_output=output)


def run_probe(devices: Sequence[str], backend: Optional[str], ffmpeg_path: str) -> List[DeviceResult]:
    backend = backend or _detect_backend()
    results: List[DeviceResult] = []
    for device in devices:
        results.append(probe_device(backend, device, ffmpeg_path))
    return results


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def format_table(records: Iterable[FormatRecord]) -> str:
    records = list(records)
    if not records:
        return "  (no formats reported by ffmpeg)"
    headers = ["Codec", "PixelFmt", "Resolution", "FPS"]
    data = [
        [
            rec.codec,
            rec.pixel_fmt,
            rec.resolution,
            rec.fps or "-",
        ]
        for rec in records
    ]
    widths = [max(len(row[i]) for row in data + [headers]) for i in range(len(headers))]
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines = [header_line, sep_line]
    for row in data:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _default_devices(backend: str) -> List[str]:
    if backend == "avfoundation":
        return ["0:none"]
    if backend == "dshow":
        return ["default"]
    # v4l2 or other unix-like
    return ["/dev/video0"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List camera codec / resolution / FPS combos using ffmpeg.")
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Device identifiers (default inferred per backend: avfoundation=0, dshow=default, v4l2=/dev/video0).",
    )
    parser.add_argument(
        "--backend",
        choices=["avfoundation", "v4l2", "dshow"],
        default=None,
        help="Override ffmpeg input backend (default: auto-detect).",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path to ffmpeg binary (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    backend = args.backend or _detect_backend()
    devices = args.devices or _default_devices(backend)
    try:
        results = run_probe(devices, backend, args.ffmpeg)
    except FileNotFoundError:
        parser.error(f"ffmpeg executable not found at '{args.ffmpeg}'")
        return 2
    except Exception as exc:  # pragma: no cover - CLI convenience
        parser.error(str(exc))
        return 2

    if args.json:
        print(json.dumps([res.to_dict() for res in results], indent=2))
        return 0

    for res in results:
        print(f"=== Device: {res.device} (backend={res.backend}) ===")
        print(format_table(res.formats))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
