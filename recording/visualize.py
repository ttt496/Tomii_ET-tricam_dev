"""Live preview and playback utilities for recorded webcam content."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from recording.capture import list_available_cameras, record_from_cameras
else:
    from .capture import list_available_cameras, record_from_cameras


def live_preview(
    camera_indexes: Sequence[int],
    *,
    frame_size: Optional[Tuple[int, int]] = None,
    window_prefix: str = "Camera",
    preview_scale: float = 1.0,
    max_seconds: Optional[float] = None,
) -> None:
    """Display live streams from the specified webcams.

    This simply forwards to :func:`record_from_cameras` without writing to disk.
    Exit the preview with the ``q`` key.
    """

    record_from_cameras(
        camera_indexes,
        output_dir=None,
        frame_size=frame_size,
        show_preview=True,
        window_prefix=window_prefix,
        preview_scale=preview_scale,
        max_seconds=max_seconds,
    )


def playback_recordings(
    video_paths: Iterable[Path | str],
    *,
    window_prefix: str = "Recording",
    loop: bool = False,
) -> None:
    """Play one or more recorded video files in individual windows."""
    caps: list[tuple[cv2.VideoCapture, str, Path]] = []
    for idx, path in enumerate(video_paths):
        resolved = Path(path).expanduser().resolve()
        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {resolved}")
        window_name = f"{window_prefix} {idx}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        caps.append((cap, window_name, resolved))

    try:
        while caps:
            finished_indices: list[int] = []
            for i, (cap, window_name, resolved) in enumerate(caps):
                success, frame = cap.read()
                if not success:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        success, frame = cap.read()
                        if not success:
                            finished_indices.append(i)
                            continue
                    else:
                        finished_indices.append(i)
                        continue

                cv2.imshow(window_name, frame)

            if finished_indices:
                for offset, idx in enumerate(finished_indices):
                    cap, window_name, _ = caps.pop(idx - offset)
                    cap.release()
                    cv2.destroyWindow(window_name)
                if not loop:
                    continue

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for cap, window_name, _ in caps:
            cap.release()
            cv2.destroyWindow(window_name)
        cv2.waitKey(1)
        cv2.destroyAllWindows()


def _resolve_frame_size(width: Optional[int], height: Optional[int]) -> Optional[Tuple[int, int]]:
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("--frame-width and --frame-height must be provided together")
    return (width, height)


def _resolve_camera_indices(indices: Sequence[int], auto_discover: bool) -> list[int]:
    if indices:
        return list(indices)

    if auto_discover:
        cameras = list_available_cameras()
        if not cameras:
            raise RuntimeError("No cameras detected. Specify --cameras explicitly.")
        print("Auto-discovered cameras:", " ".join(str(cam) for cam in cameras))
        return cameras

    raise ValueError("No cameras specified. Use --cameras or --auto-discover.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-camera preview and recording helper.")
    parser.add_argument("--live", action="store_true", help="Open a live preview from the selected cameras.")
    parser.add_argument("--record", metavar="OUTPUT_DIR", help="Record cameras to OUTPUT_DIR while optionally previewing.")
    parser.add_argument("--playback", nargs="+", metavar="FILE", help="Play one or more video files.")
    parser.add_argument("--cameras", nargs="*", type=int, metavar="INDEX", help="Camera indices to use.")
    parser.add_argument("--auto-discover", action="store_true", help="Automatically select all detected cameras when --cameras is omitted.")
    parser.add_argument("--list", action="store_true", help="List available camera indices and exit.")
    parser.add_argument("--frame-width", type=int, help="Force capture width for live/record modes.")
    parser.add_argument("--frame-height", type=int, help="Force capture height for live/record modes.")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording frame rate when using --record (default: 30).")
    parser.add_argument("--codec", default="MJPG", help="FourCC codec for recordings (default: MJPG).")
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Scale factor applied to preview windows.")
    parser.add_argument("--max-seconds", type=float, help="Stop after this many seconds.")
    parser.add_argument("--no-preview", action="store_true", help="Skip on-screen preview when recording.")
    parser.add_argument("--window-prefix", help="Custom label prefix for created windows.")
    parser.add_argument("--loop", action="store_true", help="Loop playback mode for --playback.")

    args = parser.parse_args(argv)

    if args.list:
        cameras = list_available_cameras()
        if cameras:
            print("Available cameras:", " ".join(map(str, cameras)))
        else:
            print("No cameras detected.")
        if not (args.live or args.record or args.playback):
            return 0

    actions = sum(bool(flag) for flag in (args.live, args.record, args.playback))
    if actions == 0:
        print("No action specified; defaulting to live preview with auto-discovered cameras.")
        args.live = True
        args.auto_discover = True
    elif actions > 1:
        parser.error("Use only one of --live, --record, or --playback at a time.")

    frame_size: Optional[Tuple[int, int]] = None
    try:
        frame_size = _resolve_frame_size(args.frame_width, args.frame_height)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        if args.playback:
            window_prefix = args.window_prefix or "Recording"
            playback_recordings(args.playback, window_prefix=window_prefix, loop=args.loop)
            return 0

        camera_indices = _resolve_camera_indices(args.cameras or [], args.auto_discover)
        window_prefix = args.window_prefix or "Camera"

        if args.live:
            live_preview(
                camera_indices,
                frame_size=frame_size,
                window_prefix=window_prefix,
                preview_scale=args.preview_scale,
                max_seconds=args.max_seconds,
            )
            return 0

        output_dir = Path(args.record).expanduser().resolve()
        record_from_cameras(
            camera_indices,
            output_dir=output_dir,
            frame_size=frame_size,
            fps=args.fps,
            codec=args.codec,
            show_preview=not args.no_preview,
            window_prefix=window_prefix,
            preview_scale=args.preview_scale,
            max_seconds=args.max_seconds,
        )
        print("Saved recordings to", output_dir)
        return 0
    except RuntimeError as exc:  # camera or playback failure
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # bad arguments detected after parsing
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
