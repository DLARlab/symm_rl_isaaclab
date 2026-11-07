# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the complete PPO and PPO-TRS reward histories for the three robots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/symm_locomotion"))
from analyze_trs_grid import read_scalars, rolling_mean

REWARD_TAG = "Train/mean_reward"
ROBOTS = (("go2", "Go2"), ("x1", "X1-Opposed"), ("x1_go2bend", "X1-Aligned"))
METHODS = (("ppo", "PPO", "#d19246"), ("trs", "PPO-TRS", "#4A4F7E"))


def _load_rewards(run_path: Path) -> list[tuple[int, float, float]]:
    """Read rewards and retain the latest logged value for each iteration."""
    event_paths = sorted(run_path.glob("events.out.tfevents.*"))
    if len(event_paths) != 1:
        raise ValueError(
            f"Expected one TensorBoard event file in {run_path}, found {len(event_paths)}."
        )
    series = read_scalars(event_paths[0], selected_tags={REWARD_TAG}).get(
        REWARD_TAG, []
    )
    by_step = {
        step: (step, wall_time, value)
        for step, wall_time, value in series
        if 0 <= step < 20_000
    }
    if not by_step or min(by_step) > 1 or max(by_step) != 19_999:
        raise ValueError(f"Incomplete 20,000-iteration reward history in {run_path}.")
    return [by_step[step] for step in sorted(by_step)]


def main() -> None:
    """Export the three-panel reward figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=200,
        help="Trailing-mean window in training iterations.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PDF, PNG, and SVG exports.",
    )
    args = parser.parse_args()
    if args.smoothing_window < 1:
        parser.error("--smoothing_window must be positive")

    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text())
    run_paths = {
        (source["robot"], source["method"]): ROOT / source["run"]
        for source in provenance["sources"]
    }
    curves = {}
    for robot, _title in ROBOTS:
        for method, _label, _color in METHODS:
            points = _load_rewards(run_paths[(robot, method)])
            if len(points) < args.smoothing_window:
                raise ValueError(
                    f"Smoothing window exceeds the reward history length for {robot} {method}."
                )
            curves[(robot, method)] = points
            print(
                f"{robot:12s} {method:3s}: {len(points):,} reward samples, iterations {points[0][0]}-{points[-1][0]}"
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.labelcolor": "#262626",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": "#262626",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.24, top=0.85, wspace=0.12)
    all_values = [
        value for points in curves.values() for _step, _wall_time, value in points
    ]
    y_min = min(0.0, math.floor(min(all_values)))
    y_max = math.ceil(max(all_values) * 1.04)

    for ax, (robot, title) in zip(axes, ROBOTS, strict=True):
        for method, label, color in METHODS:
            points = curves[(robot, method)]
            smooth = rolling_mean(points, args.smoothing_window)
            ax.plot(
                [step for step, _wall_time, _value in points],
                [value for _step, _wall_time, value in points],
                color=color,
                linewidth=0.55,
                alpha=0.15,
                rasterized=True,
                zorder=1,
            )
            ax.plot(
                [step for step, _wall_time, _value in smooth],
                [value for _step, _wall_time, value in smooth],
                color=color,
                linewidth=2.0,
                label=label,
                zorder=2,
            )

        ax.set_title(title, pad=12)
        ax.set_xlabel("Iterations", labelpad=7)
        ax.set_xlim(0, 20_000)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks((0, 5_000, 10_000, 15_000, 20_000))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#aaaaaa")
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(axis="both", color="#aaaaaa", length=3)

    axes[0].set_ylabel("Training reward", labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=2.7,
        columnspacing=2.5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    description = (
        f"Train/mean_reward for the complete 20,000-iteration PPO and PPO-TRS runs. "
        f"Solid lines are {args.smoothing_window}-iteration trailing means, using partial windows initially; "
        "faint lines show raw rewards. Each curve represents one training run."
    )
    for suffix in ("pdf", "png", "svg"):
        path = args.output_dir / f"training_reward_curves.{suffix}"
        metadata = (
            {"Subject": description}
            if suffix == "pdf"
            else {"Description": description}
        )
        fig.savefig(
            path, dpi=300, bbox_inches="tight", pad_inches=0.08, metadata=metadata
        )
        print(f"Saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
