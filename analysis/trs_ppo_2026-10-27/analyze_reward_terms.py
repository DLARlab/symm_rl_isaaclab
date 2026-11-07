# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare late-training reward components for the six 20,000-iteration runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/symm_locomotion"))
from analyze_trs_grid import parse_event

ROBOTS = (("go2", "Go2"), ("x1", "X1-Opposed"), ("x1_go2bend", "X1-Aligned"))
WINDOWS = (500, 1000, 2000)
STOP_ITERATION = 20_000
REWARD_PREFIX = "Episode_Reward/"
CONTEXT_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Diagnostics/ji22_positive_reward_mean",
    "Diagnostics/ji22_negative_reward_mean",
    "Diagnostics/ji22_multiplier_mean",
    "Diagnostics/joint_target_clipped_fraction",
    "Diagnostics/foot_clearance_mean_height_error",
    "Diagnostics/foot_clearance_mean_shortfall",
    "Diagnostics/foot_clearance_mean_swing_height",
    "Diagnostics/foot_clearance_mean_target_height",
    "Metrics/base_velocity/curriculum_active_bins",
    "Metrics/base_velocity/curriculum_max_abs_lin_vel_x",
    "Metrics/base_velocity/curriculum_max_abs_ang_vel_z",
    "Metrics/base_velocity/tracking_reward_lin_vel",
    "Metrics/base_velocity/tracking_reward_ang_vel",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Metrics/success_rate",
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
    "Episode_Termination/base_height",
    "Episode_Termination/base_orientation",
    "Episode_Termination/calf_height",
    "Episode_Termination/head_contact",
)


def _read_tail(event_path: Path) -> dict[str, dict[int, float]]:
    """Read relevant scalars from the final 2,000 iterations of an event file."""
    scalars = defaultdict(dict)
    start = STOP_ITERATION - max(WINDOWS)
    with event_path.open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError(f"Truncated event header: {event_path}")
            length = struct.unpack("<Q", header)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated event header checksum: {event_path}")
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise ValueError(f"Truncated event payload: {event_path}")
            _wall_time, step, values = parse_event(payload)
            if step is None or not start <= step < STOP_ITERATION:
                continue
            for tag, value in values:
                if tag.startswith(REWARD_PREFIX) or tag in CONTEXT_TAGS:
                    scalars[tag][step] = value
    if max(scalars.get("Train/mean_reward", {}), default=-1) != STOP_ITERATION - 1:
        raise ValueError(f"Missing final reward sample: {event_path}")
    error = scalars.get("Diagnostics/foot_clearance_mean_height_error", {})
    shortfall = scalars.get("Diagnostics/foot_clearance_mean_shortfall", {})
    if error and shortfall:
        # With absolute height errors, |target - height| = shortfall + overshoot.
        scalars["Derived/foot_clearance_mean_overshoot"] = {
            step: error[step] - shortfall[step]
            for step in error.keys() & shortfall.keys()
        }
    return dict(scalars)


def _mean_window(points: dict[int, float], window: int) -> tuple[float, int]:
    """Average recorded values over a fixed final iteration window."""
    values = [
        value for step, value in points.items() if step >= STOP_ITERATION - window
    ]
    return (statistics.mean(values), len(values)) if values else (math.nan, 0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write comparison rows with a shared header."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Export reward-component comparisons and supporting training diagnostics."""
    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text())
    histories = {}
    for source in provenance["sources"]:
        run = ROOT / source["run"]
        events = sorted(run.glob("events.out.tfevents.*"))
        if len(events) != 1:
            raise ValueError(f"Expected one event file in {run}, found {len(events)}.")
        histories[(source["robot"], source["method"])] = _read_tail(events[0])
        print(f"Read {source['robot']} {source['method']}", flush=True)

    reward_rows, diagnostic_rows = [], []
    report = [
        "<!--",
        "Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).",
        "All rights reserved.",
        "",
        "SPDX-License-Identifier: BSD-3-Clause",
        "-->",
        "",
        "# Late-training reward component comparison",
        "",
        "Primary window: iterations 19000-19999. Sensitivity windows: final 500 and 2000 iterations.",
        "",
        "Reward components are weighted episodic sums divided by the nominal maximum episode duration (30 s).",
        "Higher values are preferable; for negative terms, higher means a smaller penalty.",
        "The environment applies nonlinear positive/negative reward composition, so component deltas are not additive return contributions.",
        "These are descriptive comparisons of one trained policy per robot and method.",
        "Episode durations and the active curriculum bins differ between policies; this is not a matched-command evaluation.",
        "",
    ]
    for robot, label in ROBOTS:
        before = histories[(robot, "ppo")]
        after = histories[(robot, "trs")]
        before_rewards = {tag for tag in before if tag.startswith(REWARD_PREFIX)}
        after_rewards = {tag for tag in after if tag.startswith(REWARD_PREFIX)}
        if before_rewards != after_rewards:
            raise ValueError(f"Reward tags do not match for {robot}.")
        current_rewards = []
        for tag in sorted(before.keys() & after.keys()):
            for window in WINDOWS:
                ppo, ppo_count = _mean_window(before[tag], window)
                trs, trs_count = _mean_window(after[tag], window)
                row = {
                    "robot": robot,
                    "robot_label": label,
                    "window_iterations": window,
                    "tag": tag,
                    "ppo": ppo,
                    "ppo_trs": trs,
                    "delta_trs_minus_ppo": trs - ppo,
                    "delta_over_abs_ppo_percent": (
                        100 * (trs - ppo) / abs(ppo) if ppo else math.nan
                    ),
                    "ppo_samples": ppo_count,
                    "ppo_trs_samples": trs_count,
                }
                if tag.startswith(REWARD_PREFIX):
                    reward_rows.append(row)
                    if window == 1000:
                        current_rewards.append(row)
                else:
                    diagnostic_rows.append(row)
        report.extend(
            [
                f"## {label}",
                "",
                "| Reward component | PPO | PPO-TRS | Difference | Change / absolute PPO |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            current_rewards, key=lambda item: item["delta_trs_minus_ppo"], reverse=True
        ):
            term = row["tag"].removeprefix(REWARD_PREFIX)
            relative = row["delta_over_abs_ppo_percent"]
            report.append(
                f"| {term} | {row['ppo']:.6f} | {row['ppo_trs']:.6f} | {row['delta_trs_minus_ppo']:+.6f} | "
                + (f"{relative:+.2f}% |" if math.isfinite(relative) else "-- |")
            )
        report.extend(
            ["", "| Training diagnostic | PPO | PPO-TRS |", "|---|---:|---:|"]
        )
        for row in diagnostic_rows:
            if row["robot"] == robot and row["window_iterations"] == 1000:
                report.append(
                    f"| {row['tag']} | {row['ppo']:.6f} | {row['ppo_trs']:.6f} |"
                )
        report.append("")

    _write_csv(OUTPUT_DIR / "reward_term_comparison.csv", reward_rows)
    _write_csv(OUTPUT_DIR / "reward_training_diagnostics.csv", diagnostic_rows)
    (OUTPUT_DIR / "reward_term_comparison.md").write_text("\n".join(report) + "\n")
    print(
        "Saved reward_term_comparison.csv, reward_training_diagnostics.csv, reward_term_comparison.md"
    )


if __name__ == "__main__":
    main()
