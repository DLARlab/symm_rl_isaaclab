# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot raw torque-imbalance reductions transcribed from the supplied heatmap.

The embedded values are PPO minus PPO-TRS in percentage points, rounded to one
decimal place in the source image. Rows are gaits and columns are longitudinal
velocity commands. These are raw torque-squared imbalances, not imbalances
normalized by actuator effort limits. No experiment files are read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "gait_velocity_raw_torque_imbalance_bars"
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
# Color encodes velocity command, rather than the training method or sign of the
# reduction: warm shades are backward commands, cool shades forward commands.
VELOCITY_COLORS = ("#A66C2D", "#D19246", "#E7BE8B", "#A5A9C8", "#74799F", "#4A4F7E")

# Exact transcription of the 144 displayed numbers in the supplied image.
# Preserve the source's +0.0 cell; its unrounded value cannot be recovered here.
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
    """Export a three-panel grouped diverging bar plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PNG, PDF, and SVG exports.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution in dots per inch."
    )
    parser.add_argument(
        "--hide_values",
        action="store_true",
        help="Omit the signed numbers at bar ends for a less dense figure.",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": "#262626",
            "axes.labelcolor": "#262626",
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 9.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.85, wspace=0.10)
    bar_spacing = 0.115
    bar_height = 0.097

    for ax, (robot, rows) in zip(axes, REDUCTIONS.items(), strict=True):
        # Alternating bands keep the six velocity bars together within each gait.
        for gait_index in range(len(GAITS)):
            if gait_index % 2 == 0:
                ax.axhspan(
                    gait_index - 0.48,
                    gait_index + 0.48,
                    color="#F3F4F6",
                    zorder=0,
                )

        for velocity_index, color in enumerate(VELOCITY_COLORS):
            offset = (velocity_index - (len(VELOCITIES) - 1) / 2) * bar_spacing
            y_positions = [index + offset for index in range(len(GAITS))]
            values = [row[velocity_index] for row in rows]
            ax.barh(
                y_positions,
                values,
                height=bar_height,
                color=color,
                edgecolor="none",
                zorder=3,
            )
            if not args.hide_values:
                for y, value in zip(y_positions, values, strict=True):
                    ax.text(
                        value + (0.50 if value >= 0 else -0.50),
                        y,
                        f"{value:+.1f}",
                        ha="left" if value >= 0 else "right",
                        va="center",
                        fontsize=6.8,
                        color="#343434",
                        zorder=4,
                    )

        ax.axvline(0, color="#63676E", linewidth=0.9, zorder=2)
        ax.set_title(robot, pad=13)
        ax.set_xlim(-26, 37)
        ax.set_ylim(len(GAITS) - 0.5, -0.5)
        ax.set_yticks(range(len(GAITS)), GAITS)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.grid(axis="x", color="#DDE0E5", linewidth=0.6, zorder=1)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.tick_params(axis="x", length=3, color="#B2B6BD")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B2B6BD")

    fig.suptitle("Front/hind raw torque imbalance", y=0.965, fontsize=18)
    fig.text(
        0.545,
        0.918,
        "PPO − PPO-TRS; positive values indicate lower imbalance with PPO-TRS",
        ha="center",
        fontsize=11,
        color="#555A63",
    )
    fig.supxlabel(
        "Reduction in imbalance [percentage points]", x=0.545, y=0.126, fontsize=12
    )
    handles = [
        Patch(facecolor=color, edgecolor="none", label=f"{velocity:+g}")
        for velocity, color in zip(VELOCITIES, VELOCITY_COLORS, strict=True)
    ]
    fig.legend(
        handles=handles,
        title="Commanded longitudinal velocity [m/s]",
        loc="lower center",
        bbox_to_anchor=(0.545, 0.022),
        ncol=len(VELOCITIES),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.8,
        fontsize=10,
        title_fontsize=11,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    description = (
        "PPO minus PPO-TRS front/hind RAW torque-squared imbalance in percentage points. "
        "The 144 values were transcribed from the supplied heatmap at its displayed one-decimal precision; "
        "each source cell is a mean of 100 parallel trials. Values are not normalized by effort limits. "
        "Robot names updated to Go2, X1-Opposed, and X1-Aligned. No data clipping or experiment recomputation."
    )
    for suffix in ("png", "pdf", "svg"):
        path = args.output_dir / f"{OUTPUT_STEM}.{suffix}"
        metadata = (
            {"Subject": description}
            if suffix == "pdf"
            else {"Description": description}
        )
        fig.savefig(
            path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.10, metadata=metadata
        )
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
