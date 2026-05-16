"""
General-purpose GIF creation utility for experiment images.

Usage pattern you asked for:
- Copy the images you want to animate into a folder (by default: ./input_images).
- Edit the configuration section below (paths, durations, resize mode).
- Run this script with Python to generate the GIF.

You can also optionally override some settings via CLI flags for ad‑hoc runs.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image


# ============================================================================
# Configuration (edit these defaults directly when reusing the script)
# ============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent

# Folder containing input images. You can either:
# - Copy images into ./input_images, OR
# - Point this to any existing folder (will be resolved relative to this file
#   if you give a relative path).
INPUT_DIR = SCRIPT_DIR / "input_images"

# Output GIF path (parent directory will be created if missing).
OUTPUT_GIF = SCRIPT_DIR / "output" / "animation.gif"

# Glob pattern for input images (case-sensitive). Common examples:
# "*.png", "*.jpg", "*.jpeg".
FILE_PATTERN = "*.png"

# How to order frames:
# - "numeric": look for the last integer in each filename (e.g., epoch_00100)
# - "lexicographic": sort lexicographically by filename
ORDERING_MODE = "numeric"  # "numeric" or "lexicographic"

# Regex used when ORDERING_MODE == "numeric" to extract integers from filenames.
# By default it finds all digit groups like "00100" or "10" and takes the last one.
NUMERIC_REGEX = r"(\d+)"

# Timing configuration

# Duration (in seconds) for all frames except possibly the last one.
FRAME_DURATION_SEC: float = 0.250

# If not None, the last frame will use this duration (in seconds),
# otherwise it uses FRAME_DURATION_SEC like all others.
LAST_FRAME_DURATION_SEC: Optional[float] = None

# Optional overall duration (in seconds) for one loop of the GIF.
# - If None: durations are taken from FRAME_DURATION_SEC / LAST_FRAME_DURATION_SEC.
# - If a positive float: the script rescales per-frame durations so that the
#   total duration of one loop is approximately TOTAL_DURATION_SEC, while
#   preserving the ratio between normal frames and the last frame.
TOTAL_DURATION_SEC: Optional[float] = None

# Number of loops:
# - 0 means infinite.
# - 1 means play once, 2 means play twice, etc.
LOOP: int = 0

# Size handling

# How to choose the target size for all frames:
# - "first": use size of first image
# - "max":   use (max_width, max_height) across all images
# - "fixed": use FIXED_SIZE below
RESIZE_MODE = "first"  # "first", "max", or "fixed"

# Only used when RESIZE_MODE == "fixed".
# Example: FIXED_SIZE = (1280, 720)
FIXED_SIZE: Optional[Tuple[int, int]] = None

# When resizing, which Pillow resampling filter to use.
RESAMPLE_FILTER = Image.Resampling.LANCZOS


# ============================================================================
# Implementation
# ============================================================================

@dataclass
class ImageInfo:
    path: Path
    size: Tuple[int, int]  # (width, height)


def resolve_path(base: Path, p: Path | str) -> Path:
    """Resolve `p` relative to `base` if it is not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def discover_images(input_dir: Path, pattern: str) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching pattern {pattern!r} found in {input_dir}"
        )
    return files


def extract_numeric_key(name: str, regex: str) -> Optional[int]:
    matches = re.findall(regex, name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def sort_image_paths(
    paths: Iterable[Path],
    ordering_mode: str,
    numeric_regex: str,
) -> List[Path]:
    paths = list(paths)

    if ordering_mode == "numeric":
        def key_fn(p: Path):
            k = extract_numeric_key(p.name, numeric_regex)
            # Fall back to lexicographic if no numeric key is found.
            return (0 if k is not None else 1, k if k is not None else 0, p.name)
    else:
        # Pure lexicographic
        def key_fn(p: Path):
            return (0, 0, p.name)

    return sorted(paths, key=key_fn)


def load_images(paths: Sequence[Path]) -> Tuple[List[Image.Image], List[ImageInfo]]:
    images: List[Image.Image] = []
    infos: List[ImageInfo] = []

    for p in paths:
        img = Image.open(p).convert("RGBA")
        w, h = img.size
        images.append(img)
        infos.append(ImageInfo(path=p, size=(w, h)))

    return images, infos


def determine_target_size(
    infos: Sequence[ImageInfo],
    resize_mode: str,
    fixed_size: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    if not infos:
        raise ValueError("No images provided to determine target size.")

    if resize_mode == "fixed":
        if not fixed_size:
            raise ValueError("RESIZE_MODE is 'fixed' but FIXED_SIZE is not set.")
        return fixed_size

    if resize_mode == "max":
        max_w = max(info.size[0] for info in infos)
        max_h = max(info.size[1] for info in infos)
        return (max_w, max_h)

    # Default: "first"
    return infos[0].size


def resize_images_in_place(
    images: List[Image.Image],
    infos: Sequence[ImageInfo],
    target_size: Tuple[int, int],
) -> None:
    target_w, target_h = target_size
    for i, (img, info) in enumerate(zip(images, infos)):
        if img.size != target_size:
            images[i] = img.resize((target_w, target_h), RESAMPLE_FILTER)


def summarize_sizes(infos: Sequence[ImageInfo]) -> str:
    counts = {}
    for info in infos:
        counts[info.size] = counts.get(info.size, 0) + 1
    parts = [f"{count}x {w}x{h}" for (w, h), count in sorted(counts.items())]
    return ", ".join(parts)


def build_durations_seconds(
    n_frames: int,
    frame_duration: float,
    last_frame_duration: Optional[float],
    total_duration: Optional[float],
) -> List[float]:
    if n_frames <= 0:
        raise ValueError("Cannot build durations for zero frames.")

    if n_frames == 1:
        # Single-frame GIF (essentially a still image). Use last_frame_duration
        # if provided, otherwise frame_duration.
        base = last_frame_duration if last_frame_duration is not None else frame_duration
        return [max(base, 0.01)]

    # Base durations before any total-duration scaling
    durations = [frame_duration] * n_frames
    if last_frame_duration is not None:
        durations[-1] = last_frame_duration

    # Ensure a minimum per-frame duration to avoid issues with some viewers
    durations = [max(d, 0.01) for d in durations]

    if total_duration is not None and total_duration > 0:
        current_total = sum(durations)
        if current_total > 0:
            scale = total_duration / current_total
            durations = [max(d * scale, 0.01) for d in durations]

    return durations


def seconds_to_milliseconds(durations_sec: Sequence[float]) -> List[int]:
    return [max(int(round(d * 1000)), 10) for d in durations_sec]


def create_gif(
    input_dir: Path,
    output_gif: Path,
    file_pattern: str = FILE_PATTERN,
    ordering_mode: str = ORDERING_MODE,
    numeric_regex: str = NUMERIC_REGEX,
    frame_duration_sec: float = FRAME_DURATION_SEC,
    last_frame_duration_sec: Optional[float] = LAST_FRAME_DURATION_SEC,
    total_duration_sec: Optional[float] = TOTAL_DURATION_SEC,
    resize_mode: str = RESIZE_MODE,
    fixed_size: Optional[Tuple[int, int]] = FIXED_SIZE,
    loop: int = LOOP,
) -> None:
    # Discover and sort image paths
    paths = discover_images(input_dir, file_pattern)
    paths = sort_image_paths(paths, ordering_mode, numeric_regex)

    print("=" * 72)
    print("GIF creation configuration")
    print("=" * 72)
    print(f"Input directory      : {input_dir}")
    print(f"File pattern         : {file_pattern}")
    print(f"Number of frames     : {len(paths)}")
    print(f"Ordering mode        : {ordering_mode}")
    print(f"Numeric regex        : {numeric_regex!r}")
    print(f"Resize mode          : {resize_mode}")
    print(f"Fixed size           : {fixed_size}")
    print(f"Frame duration (s)   : {frame_duration_sec}")
    print(f"Last frame duration  : {last_frame_duration_sec}")
    print(f"Total duration (s)   : {total_duration_sec}")
    print(f"Loop count           : {loop}")
    print(f"Output GIF           : {output_gif}")
    print()

    # Load images
    images, infos = load_images(paths)
    print("Input image sizes:")
    print(f"  {summarize_sizes(infos)}")

    # Determine target size and resize if necessary
    target_size = determine_target_size(infos, resize_mode, fixed_size)
    print(f"\nTarget frame size    : {target_size[0]}x{target_size[1]} pixels")

    resize_images_in_place(images, infos, target_size)

    # Build per-frame durations
    durations_sec = build_durations_seconds(
        n_frames=len(images),
        frame_duration=frame_duration_sec,
        last_frame_duration=last_frame_duration_sec,
        total_duration=total_duration_sec,
    )
    durations_ms = seconds_to_milliseconds(durations_sec)

    total_loop_duration = sum(durations_sec)
    print(f"\nComputed frame durations (s):")
    print(f"  First frame      : {durations_sec[0]:.3f}")
    if len(durations_sec) > 1:
        print(f"  Last frame       : {durations_sec[-1]:.3f}")
    print(f"  Total per loop   : {total_loop_duration:.3f} seconds")

    # Ensure output directory exists
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    # Save GIF via Pillow with per-frame durations
    print("\nSaving GIF...")
    first, *rest = images
    first.save(
        output_gif,
        save_all=True,
        append_images=rest,
        duration=durations_ms,
        loop=loop,
        optimize=False,
    )

    print(f"\n[SUCCESS] GIF created at: {output_gif}")
    print(f"  Frames: {len(images)}")
    print(f"  Size  : {target_size[0]}x{target_size[1]} pixels")
    print(f"  Loop  : {'infinite' if loop == 0 else loop}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an animated GIF from a set of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory containing images. "
             "If omitted, uses INPUT_DIR from the config section.",
    )
    parser.add_argument(
        "--output-gif",
        type=str,
        default=None,
        help="Output GIF path. If omitted, uses OUTPUT_GIF from the config section.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="Glob pattern for input images (e.g., '*.png', '*.jpg').",
    )
    parser.add_argument(
        "--ordering-mode",
        type=str,
        choices=["numeric", "lexicographic"],
        default=None,
        help="How to order frames (overrides ORDERING_MODE if provided).",
    )
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=None,
        help="Uniform frame duration in seconds for non-last frames.",
    )
    parser.add_argument(
        "--last-frame-duration",
        type=float,
        default=None,
        help="Duration in seconds for the last frame.",
    )
    parser.add_argument(
        "--total-duration",
        type=float,
        default=None,
        help="Overall duration in seconds for one GIF loop. "
             "If set, durations are rescaled to match this value.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=["first", "max", "fixed"],
        default=None,
        help="How to choose target size for frames (overrides RESIZE_MODE if provided).",
    )
    parser.add_argument(
        "--fixed-width",
        type=int,
        default=None,
        help="When resize-mode is 'fixed', fixed output width (pixels).",
    )
    parser.add_argument(
        "--fixed-height",
        type=int,
        default=None,
        help="When resize-mode is 'fixed', fixed output height (pixels).",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=None,
        help="Number of loops (0 for infinite).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve effective configuration (CLI overrides config-at-top when provided)
    input_dir = resolve_path(SCRIPT_DIR, args.input_dir) if args.input_dir else INPUT_DIR
    output_gif = (
        resolve_path(SCRIPT_DIR, args.output_gif) if args.output_gif else OUTPUT_GIF
    )
    file_pattern = args.file_pattern or FILE_PATTERN
    ordering_mode = args.ordering_mode or ORDERING_MODE

    frame_duration = args.frame_duration if args.frame_duration is not None else FRAME_DURATION_SEC
    last_frame_duration = (
        args.last_frame_duration
        if args.last_frame_duration is not None
        else LAST_FRAME_DURATION_SEC
    )
    total_duration = (
        args.total_duration if args.total_duration is not None else TOTAL_DURATION_SEC
    )

    resize_mode = args.resize_mode or RESIZE_MODE
    if resize_mode == "fixed":
        width = args.fixed_width
        height = args.fixed_height
        fixed_size = (width, height) if (width and height) else FIXED_SIZE
    else:
        fixed_size = FIXED_SIZE

    loop = args.loop if args.loop is not None else LOOP

    create_gif(
        input_dir=input_dir,
        output_gif=output_gif,
        file_pattern=file_pattern,
        ordering_mode=ordering_mode,
        numeric_regex=NUMERIC_REGEX,
        frame_duration_sec=frame_duration,
        last_frame_duration_sec=last_frame_duration,
        total_duration_sec=total_duration,
        resize_mode=resize_mode,
        fixed_size=fixed_size,
        loop=loop,
    )


if __name__ == "__main__":
    main()

