# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert an MP4 rollout video to GIF using an available ffmpeg binary."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg() -> str:
    """Return an ffmpeg executable path."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            return ffmpeg
    raise RuntimeError("Could not find ffmpeg or imageio_ffmpeg in this Python environment.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert an MP4 file to GIF.")
    parser.add_argument("--input", required=True, help="Input MP4 path.")
    parser.add_argument("--output", default=None, help="Output GIF path. Defaults to input path with .gif suffix.")
    parser.add_argument("--fps", type=int, default=15, help="Output GIF frame rate.")
    parser.add_argument("--width", type=int, default=720, help="Output GIF width in pixels.")
    return parser.parse_args()


def main() -> int:
    """Convert the requested MP4 file to GIF."""
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input MP4 does not exist: {input_path}", file=sys.stderr)
        return 2
    output_path = Path(args.output) if args.output else input_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps={args.fps},scale={args.width}:-1:flags=lanczos",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
