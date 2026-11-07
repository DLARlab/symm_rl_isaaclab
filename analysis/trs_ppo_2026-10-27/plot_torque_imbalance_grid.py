# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recreate gait_velocity_torque_balance.png using the displayed cell values.

Rows are gait templates; columns are commanded longitudinal velocities [m/s].
Values are reductions in raw front/hind torque-squared imbalance [percentage
points], computed as PPO minus PPO-TRS and transcribed to the one-decimal
precision shown in the supplied image. No experiment files are read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "gait_velocity_torque_balance_recreated"
GAITS = (
    "Trot",
    "Bound",
    "Half-bound A",
    "Half-bound B",
    "Gallop A",
    "Gallop B",
    "Gallop C",
    "Gallop D",
)
VELOCITIES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)

# Rows follow GAITS; columns follow VELOCITIES. All 144 values are embedded.
# The +0.0 cell is preserved at the precision displayed in the source image.
REDUCTIONS = {
    "Go2": (
        (-2.1, 8.5, 15.2, 8.7, 1.8, -5.5),
        (21.7, 14.3, 10.1, 9.9, -15.5, -0.9),
        (-2.1, 6.0, 6.3, 8.2, 19.8, 17.1),
        (2.2, 15.9, 6.1, 31.8, 16.0, 2.5),
        (3.6, 14.0, 16.2, 18.7, 15.5, -2.7),
        (-6.3, 8.9, 12.2, 1.1, 9.9, 4.9),
        (6.8, 4.5, 1.1, 14.7, 30.2, 29.3),
        (25.9, 18.6, 9.3, 16.4, 2.0, 2.7),
    ),
    "X1-Opposed": (
        (6.5, 2.3, 5.3, -3.1, -3.9, -9.9),
        (5.6, -12.1, 2.5, 6.8, 6.4, -4.8),
        (-14.2, -8.3, 3.5, -1.8, -1.7, 6.6),
        (-0.2, 1.1, 6.0, 0.0, 1.3, -20.4),
        (0.7, 0.4, -9.3, -13.6, 7.5, -12.1),
        (-7.5, -10.6, 7.8, -0.9, -4.8, -8.5),
        (-16.3, -10.0, 3.0, -6.9, -5.8, -0.1),
        (-9.0, -5.7, -0.6, -4.8, -7.9, 5.7),
    ),
    "X1-Aligned": (
        (-3.9, 3.8, 4.5, 0.5, -3.9, 6.7),
        (-10.1, -17.0, -6.9, 8.6, 21.8, 0.6),
        (-3.9, -7.2, -9.5, -5.6, 22.5, 10.3),
        (10.8, -1.2, -1.9, 8.1, 9.1, -3.8),
        (7.3, 7.2, -1.4, -1.5, 17.9, -2.4),
        (16.4, -4.9, 1.4, 17.3, -5.5, -6.7),
        (-19.6, -13.4, -8.1, -1.4, 3.0, 3.7),
        (5.0, -1.6, 11.1, 3.4, -3.3, -6.6),
    ),
}


def main() -> None:
    """Export the original three-panel annotated heatmap layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PNG, PDF, and SVG exports.",
    )
    parser.add_argument(
        "--dpi", type=int, default=180, help="PNG resolution in dots per inch."
    )
    parser.add_argument(
        "--cmap",
        default="RdYlGn",
        help="Matplotlib colormap; default matches the source.",
    )
    parser.add_argument(
        "--color_limit",
        type=float,
        default=25.0,
        help="Symmetric color limits [percentage points]; annotations retain full values.",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if not 0.0 < args.color_limit < float("inf"):
        parser.error("--color_limit must be positive and finite")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(14.3, 5.8), sharey=True, layout="constrained"
    )
    for ax, (robot, matrix) in zip(axes, REDUCTIONS.items(), strict=True):
        color = ax.imshow(
            matrix,
            cmap=args.cmap,
            vmin=-args.color_limit,
            vmax=args.color_limit,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(
            range(len(VELOCITIES)), [f"{v:+g}" for v in VELOCITIES], fontsize=10
        )
        ax.set_yticks(range(len(GAITS)), GAITS, fontsize=10)
        ax.set_title(robot, fontsize=12, pad=10)
        ax.set_xlabel("Commanded x velocity [m/s]")
        ax.tick_params(length=0, pad=7)
        # Match the source's division between command directions and gait families.
        ax.axvline(2.5, color="white", linewidth=3)
        for boundary in (0.5, 1.5, 3.5):
            ax.axhline(boundary, color="white", linewidth=1.5)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=8.5)

    # fig.suptitle(
    #     "Front/hind raw torque imbalance: PPO − PPO-TRS",
    #     fontsize=14,
    # )
    fig.colorbar(
        color,
        ax=list(axes),
        fraction=0.02,
        pad=0.02,
        label=f"Reduction (%)",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    description = (
        "Recreation of gait_velocity_torque_balance.png using its 144 displayed values. "
        "Raw front/hind torque-squared imbalance, PPO minus PPO-TRS [percentage points]. "
        "Values retain the source's one-decimal precision. Each cell is a mean of 100 parallel trials. "
        "No experiment recomputation. Updated robot names: Go2, X1-Opposed, X1-Aligned."
    )
    for suffix in ("png", "pdf", "svg"):
        path = args.output_dir / f"{OUTPUT_STEM}.{suffix}"
        metadata = (
            {"Subject": description}
            if suffix == "pdf"
            else {"Description": description}
        )
        fig.savefig(path, dpi=args.dpi, metadata=metadata)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
