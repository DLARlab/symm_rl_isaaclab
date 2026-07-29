# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot reward curves for TensorBoard runs archived under ``good_runs``."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_trs_grid import (
    LOG_ROOT,
    read_scalars,
    rolling_mean,
    threshold_crossing,
    trapezoid_auc,
)
from plot_trs_tensorboard import (
    SVG_NAMESPACE,
    _add_text,
    _format_tick,
    _nice_bounds,
    _polyline_points,
    _svg_tag,
)

GOOD_RUNS_ROOT = LOG_ROOT / "good_runs"
OUTPUT_ROOT = GOOD_RUNS_ROOT / "curated_tensorboard"
REWARD_TAG = "Train/mean_reward"
SMOOTHING_WINDOW = 200
PLOT_STRIDE = 10
ROBOT_RUN_DIRS = {
    "go2": GOOD_RUNS_ROOT / "unitree_go2_symm_flat",
    "x1": GOOD_RUNS_ROOT / "dobot_x1_symm_flat",
}
ROBOT_TITLES = {
    "go2": "Unitree Go2",
    "x1": "Dobot X1",
}
LEGACY_RUN_NAMES = {
    "2026-07-07_00-11-14_no_trs",
    "2026-07-11_02-59-13_no_trs",
    "2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral",
    "2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral",
}
UPDATED_RUN_NAMES = {
    "2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20",
    "2026-07-19_17-37-55_go2_trs_m0p1_v0p05_w500_minv0_pitch0p50_pterm1p20",
    "2026-07-20_16-23-32_go2_trs_m0p20_v0p10_w500",
    "2026-07-19_10-33-04_x1_no_trs_pitch0p35",
    "2026-07-19_17-46-27_x1_trs_m0p1_v0p05_w500_pitch0p35_pterm0p70",
    "2026-07-20_16-24-19_x1_trs_m0p20_v0p10_w500",
}
CONDITION_COLORS = {
    (0.0, 0.0): "#202124",
    (0.1, 0.05): "#0072B2",
    (0.2, 0.1): "#D97706",
}

ET.register_namespace("", SVG_NAMESPACE)


@dataclass(frozen=True)
class CuratedRun:
    """One TensorBoard run archived under ``good_runs``."""

    robot: str
    generation: str
    run_path: Path
    event_path: Path
    agent_path: Path
    seed: int
    max_iterations: int
    learning_rate: float
    schedule: str
    use_mirror_loss: bool
    mirror_coeff: float
    value_coeff: float
    warmup_iterations: int
    min_abs_command_velocity: float

    @property
    def condition(self) -> str:
        """Return the effective TRS condition."""
        if not self.use_mirror_loss:
            return "No TRS"
        return f"TRS {self.mirror_coeff:.2f}/{self.value_coeff:.2f}"

    @property
    def label(self) -> str:
        """Return the plot legend label."""
        learning_rate = f"{self.learning_rate:.0e}".replace("e-0", "e-")
        return (
            f"{self.generation} | {self.condition} | "
            f"lr {learning_rate} {self.schedule} | {self.max_iterations // 1000}k"
        )


@dataclass(frozen=True)
class CuratedCurve:
    """Smoothed reward data and summary metrics for one curated run."""

    run: CuratedRun
    iterations: tuple[float, ...]
    rewards: tuple[float, ...]
    raw_points: tuple[tuple[int, float, float], ...]
    reward_auc_first_10000: float
    reward_auc_full: float
    final_reward_mean: float
    final_reward_std: float
    reward_30_iteration: int | None
    reward_35_iteration: int | None


def _parse_yaml_scalar(path: Path, key: str) -> Any:
    """Parse one unique scalar key from a resolved YAML snapshot."""
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$")
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is not None:
            values.append(match.group(1))
    if len(values) != 1:
        raise ValueError(f"Expected one {key!r} entry in {path}, found {len(values)}.")
    value = values[0]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _generation(run_name: str) -> str:
    """Return the documented observation generation for a curated run."""
    if run_name in LEGACY_RUN_NAMES:
        return "60D"
    if run_name in UPDATED_RUN_NAMES:
        return "72D"
    raise ValueError(
        f"Curated run {run_name!r} is not classified as 60D or 72D. Update the generation inventory before plotting it."
    )


def discover_curated_runs() -> list[CuratedRun]:
    """Discover runs whose event and parameter files are physically curated."""
    runs = []
    for robot, run_root in ROBOT_RUN_DIRS.items():
        for run_path in sorted(path for path in run_root.iterdir() if path.is_dir()):
            event_paths = sorted(run_path.glob("events.out.tfevents.*"))
            agent_path = run_path / "params/agent.yaml"
            if not event_paths and not agent_path.is_file():
                continue
            if len(event_paths) != 1 or not agent_path.is_file():
                raise ValueError(
                    f"Curated TensorBoard run {run_path} must contain exactly one event file and params/agent.yaml."
                )
            runs.append(
                CuratedRun(
                    robot=robot,
                    generation=_generation(run_path.name),
                    run_path=run_path,
                    event_path=event_paths[0],
                    agent_path=agent_path,
                    seed=int(_parse_yaml_scalar(agent_path, "seed")),
                    max_iterations=int(_parse_yaml_scalar(agent_path, "max_iterations")),
                    learning_rate=float(_parse_yaml_scalar(agent_path, "learning_rate")),
                    schedule=str(_parse_yaml_scalar(agent_path, "schedule")),
                    use_mirror_loss=bool(_parse_yaml_scalar(agent_path, "use_mirror_loss")),
                    mirror_coeff=float(_parse_yaml_scalar(agent_path, "mirror_loss_coeff")),
                    value_coeff=float(_parse_yaml_scalar(agent_path, "value_loss_coeff")),
                    warmup_iterations=int(_parse_yaml_scalar(agent_path, "warmup_iterations")),
                    min_abs_command_velocity=float(_parse_yaml_scalar(agent_path, "min_abs_command_velocity")),
                )
            )
    for robot in ROBOT_RUN_DIRS:
        robot_runs = [run for run in runs if run.robot == robot]
        if len(robot_runs) != 5:
            raise ValueError(f"Expected five curated TensorBoard runs for {robot}, found {len(robot_runs)}.")
    return runs


def load_curve(run: CuratedRun) -> CuratedCurve:
    """Load one curated reward curve and compute iteration-based summaries."""
    scalar_series = read_scalars(run.event_path, selected_tags={REWARD_TAG})
    if REWARD_TAG not in scalar_series:
        raise ValueError(f"Missing TensorBoard tag {REWARD_TAG!r} in {run.event_path}.")
    raw_points = scalar_series[REWARD_TAG]
    expected_points = run.max_iterations - raw_points[0][0]
    if (
        raw_points[0][0] not in {0, 1}
        or raw_points[-1][0] != run.max_iterations - 1
        or len(raw_points) != expected_points
    ):
        raise ValueError(
            f"Incomplete reward trace in {run.event_path}: {len(raw_points)} points, "
            f"steps {raw_points[0][0]}..{raw_points[-1][0]}."
        )
    smoothed_points = rolling_mean(raw_points, SMOOTHING_WINDOW)[SMOOTHING_WINDOW - 1 :]
    sampled_points = smoothed_points[::PLOT_STRIDE]
    if sampled_points[-1] != smoothed_points[-1]:
        sampled_points.append(smoothed_points[-1])
    last_window = raw_points[-1000:]
    first_10000 = [point for point in raw_points if point[0] <= 9999]
    reward_30 = threshold_crossing(raw_points, 30.0, above=True)
    reward_35 = threshold_crossing(raw_points, 35.0, above=True)
    return CuratedCurve(
        run=run,
        iterations=tuple(float(step) for step, _wall_time, _reward in sampled_points),
        rewards=tuple(reward for _step, _wall_time, reward in sampled_points),
        raw_points=tuple(raw_points),
        reward_auc_first_10000=trapezoid_auc(first_10000),
        reward_auc_full=trapezoid_auc(raw_points),
        final_reward_mean=statistics.fmean(point[2] for point in last_window),
        final_reward_std=statistics.pstdev(point[2] for point in last_window),
        reward_30_iteration=None if reward_30 is None else int(reward_30["iteration"]),
        reward_35_iteration=None if reward_35 is None else int(reward_35["iteration"]),
    )


def _curve_sort_key(curve: CuratedCurve) -> tuple[int, float, float]:
    """Sort curves by generation and then TRS coefficients."""
    return (
        0 if curve.run.generation == "60D" else 1,
        curve.run.mirror_coeff,
        curve.run.value_coeff,
    )


def _curve_color(curve: CuratedCurve) -> str:
    """Return the coefficient-specific line color."""
    return CONDITION_COLORS[(curve.run.mirror_coeff, curve.run.value_coeff)]


def _draw_legend(root: ET.Element, curves: Sequence[CuratedCurve]) -> None:
    """Draw the parameter-rich legend below the reward plot."""
    start_x = 75.0
    start_y = 710.0
    item_width = 645.0
    for index, curve in enumerate(curves):
        row, column = divmod(index, 2)
        item_x = start_x + column * item_width
        item_y = start_y + row * 35.0
        attributes = {
            "x1": f"{item_x:.2f}",
            "x2": f"{item_x + 48.0:.2f}",
            "y1": f"{item_y:.2f}",
            "y2": f"{item_y:.2f}",
            "stroke": _curve_color(curve),
            "stroke-width": "3.0",
            "stroke-linecap": "round",
        }
        if curve.run.generation == "60D":
            attributes["stroke-dasharray"] = "9 5"
        ET.SubElement(root, _svg_tag("line"), attributes)
        _add_text(root, item_x + 59.0, item_y + 5.0, curve.run.label, size=12)


def plot_robot_curves(robot: str, curves: Sequence[CuratedCurve], output_path: Path) -> None:
    """Export one reward-versus-iteration SVG for a robot."""
    robot_curves = sorted((curve for curve in curves if curve.run.robot == robot), key=_curve_sort_key)
    if len(robot_curves) != 5:
        raise ValueError(f"Expected five curves for {robot}, found {len(robot_curves)}.")

    canvas_width = 1400.0
    canvas_height = 825.0
    plot_left = 105.0
    plot_top = 135.0
    plot_width = 1225.0
    plot_height = 500.0
    x_min = 0.0
    x_max = float(max(curve.run.max_iterations for curve in robot_curves))
    x_ticks = [float(value) for value in range(0, int(x_max) + 1, 2500)]
    reward_values = [reward for curve in robot_curves for reward in curve.rewards]
    y_min, y_max, y_ticks = _nice_bounds(reward_values)
    x_step = x_ticks[1] - x_ticks[0]
    y_step = y_ticks[1] - y_ticks[0]

    root = ET.Element(
        _svg_tag("svg"),
        {
            "viewBox": f"0 0 {canvas_width:.0f} {canvas_height:.0f}",
            "width": f"{canvas_width:.0f}",
            "height": f"{canvas_height:.0f}",
            "role": "img",
            "aria-labelledby": "plot-title plot-description",
        },
    )
    title = ET.SubElement(root, _svg_tag("title"), {"id": "plot-title"})
    title.text = f"{ROBOT_TITLES[robot]} curated TensorBoard reward by training iteration"
    description = ET.SubElement(root, _svg_tag("desc"), {"id": "plot-description"})
    description.text = (
        "Five mean-reward curves from runs physically archived under good_runs. "
        "Dashed lines are the historical 60D generation and solid lines are the 72D generation."
    )
    ET.SubElement(
        root,
        _svg_tag("rect"),
        {"width": f"{canvas_width:.0f}", "height": f"{canvas_height:.0f}", "fill": "#FFFFFF"},
    )
    definitions = ET.SubElement(root, _svg_tag("defs"))
    clip_path = ET.SubElement(definitions, _svg_tag("clipPath"), {"id": "reward-clip"})
    ET.SubElement(
        clip_path,
        _svg_tag("rect"),
        {
            "x": f"{plot_left:.2f}",
            "y": f"{plot_top:.2f}",
            "width": f"{plot_width:.2f}",
            "height": f"{plot_height:.2f}",
        },
    )

    _add_text(
        root,
        canvas_width / 2.0,
        42.0,
        f"{ROBOT_TITLES[robot]} curated runs: TensorBoard reward vs iteration",
        size=24,
        anchor="middle",
        weight=600,
    )
    _add_text(
        root,
        canvas_width / 2.0,
        72.0,
        (f"{REWARD_TAG}, {SMOOTHING_WINDOW}-iteration trailing mean · source: logs/rsl_rl/good_runs only"),
        size=13,
        anchor="middle",
        fill="#5F6368",
    )
    _add_text(
        root,
        canvas_width / 2.0,
        98.0,
        "Reward definitions changed between 60D (dashed) and 72D (solid); compare within generation.",
        size=12,
        anchor="middle",
        fill="#5F6368",
    )

    for tick in y_ticks:
        y_position = plot_top + (y_max - tick) / (y_max - y_min) * plot_height
        ET.SubElement(
            root,
            _svg_tag("line"),
            {
                "x1": f"{plot_left:.2f}",
                "x2": f"{plot_left + plot_width:.2f}",
                "y1": f"{y_position:.2f}",
                "y2": f"{y_position:.2f}",
                "stroke": "#DADCE0",
                "stroke-width": "1",
            },
        )
        _add_text(
            root,
            plot_left - 11.0,
            y_position + 5.0,
            _format_tick(tick, y_step),
            size=12,
            anchor="end",
            fill="#5F6368",
        )
    for tick in x_ticks:
        x_position = plot_left + (tick - x_min) / (x_max - x_min) * plot_width
        ET.SubElement(
            root,
            _svg_tag("line"),
            {
                "x1": f"{x_position:.2f}",
                "x2": f"{x_position:.2f}",
                "y1": f"{plot_top:.2f}",
                "y2": f"{plot_top + plot_height:.2f}",
                "stroke": "#EEF0F2",
                "stroke-width": "1",
            },
        )
        _add_text(
            root,
            x_position,
            plot_top + plot_height + 24.0,
            _format_tick(tick, x_step),
            size=12,
            anchor="middle",
            fill="#5F6368",
        )

    legacy_budget_x = plot_left + 10_000.0 / x_max * plot_width
    ET.SubElement(
        root,
        _svg_tag("line"),
        {
            "x1": f"{legacy_budget_x:.2f}",
            "x2": f"{legacy_budget_x:.2f}",
            "y1": f"{plot_top:.2f}",
            "y2": f"{plot_top + plot_height:.2f}",
            "stroke": "#7A7A7A",
            "stroke-width": "1.2",
            "stroke-dasharray": "4 4",
        },
    )
    ET.SubElement(
        root,
        _svg_tag("rect"),
        {
            "x": f"{legacy_budget_x + 5.0:.2f}",
            "y": f"{plot_top + 6.0:.2f}",
            "width": "122",
            "height": "19",
            "fill": "#FFFFFF",
            "fill-opacity": "0.88",
        },
    )
    _add_text(
        root,
        legacy_budget_x + 10.0,
        plot_top + 20.0,
        "60D budget ends",
        size=11,
        fill="#5F6368",
    )

    for attributes in (
        {
            "x1": plot_left,
            "x2": plot_left,
            "y1": plot_top,
            "y2": plot_top + plot_height,
        },
        {
            "x1": plot_left,
            "x2": plot_left + plot_width,
            "y1": plot_top + plot_height,
            "y2": plot_top + plot_height,
        },
    ):
        ET.SubElement(
            root,
            _svg_tag("line"),
            {
                **{name: f"{value:.2f}" for name, value in attributes.items()},
                "stroke": "#5F6368",
                "stroke-width": "1.2",
            },
        )

    curve_group = ET.SubElement(root, _svg_tag("g"), {"clip-path": "url(#reward-clip)"})
    for curve in robot_curves:
        attributes = {
            "points": _polyline_points(
                curve.iterations,
                curve.rewards,
                plot_left=plot_left,
                plot_top=plot_top,
                plot_width=plot_width,
                plot_height=plot_height,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            ),
            "fill": "none",
            "stroke": _curve_color(curve),
            "stroke-width": "2.4",
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
            "opacity": "0.92",
        }
        if curve.run.generation == "60D":
            attributes["stroke-dasharray"] = "9 5"
        ET.SubElement(curve_group, _svg_tag("polyline"), attributes)

    _add_text(
        root,
        plot_left + plot_width / 2.0,
        plot_top + plot_height + 61.0,
        "Training iteration",
        size=14,
        anchor="middle",
    )
    y_label_x = 29.0
    y_label_y = plot_top + plot_height / 2.0
    _add_text(
        root,
        y_label_x,
        y_label_y,
        REWARD_TAG,
        size=14,
        anchor="middle",
        transform=f"rotate(-90 {y_label_x:.2f} {y_label_y:.2f})",
    )
    _draw_legend(root, robot_curves)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def write_summary(path: Path, curves: Sequence[CuratedCurve]) -> None:
    """Write machine-readable training summaries for every curated curve."""
    fieldnames = (
        "robot",
        "generation",
        "condition",
        "run",
        "seed",
        "max_iterations",
        "learning_rate",
        "schedule",
        "mirror_loss_coeff",
        "value_loss_coeff",
        "warmup_iterations",
        "min_abs_command_velocity",
        "reward_auc_first_10000",
        "reward_auc_full",
        "final_reward_mean",
        "final_reward_std",
        "reward_30_iteration",
        "reward_35_iteration",
        "event_path",
        "agent_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for curve in sorted(curves, key=lambda item: (item.run.robot, _curve_sort_key(item))):
            writer.writerow(
                {
                    "robot": curve.run.robot,
                    "generation": curve.run.generation,
                    "condition": curve.run.condition,
                    "run": curve.run.run_path.name,
                    "seed": curve.run.seed,
                    "max_iterations": curve.run.max_iterations,
                    "learning_rate": curve.run.learning_rate,
                    "schedule": curve.run.schedule,
                    "mirror_loss_coeff": curve.run.mirror_coeff,
                    "value_loss_coeff": curve.run.value_coeff,
                    "warmup_iterations": curve.run.warmup_iterations,
                    "min_abs_command_velocity": curve.run.min_abs_command_velocity,
                    "reward_auc_first_10000": curve.reward_auc_first_10000,
                    "reward_auc_full": curve.reward_auc_full,
                    "final_reward_mean": curve.final_reward_mean,
                    "final_reward_std": curve.final_reward_std,
                    "reward_30_iteration": curve.reward_30_iteration,
                    "reward_35_iteration": curve.reward_35_iteration,
                    "event_path": curve.run.event_path.relative_to(GOOD_RUNS_ROOT).as_posix(),
                    "agent_path": curve.run.agent_path.relative_to(GOOD_RUNS_ROOT).as_posix(),
                }
            )


def main() -> None:
    """Generate plots and a summary table from the curated archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for the curated TensorBoard plots and summary CSV.",
    )
    args = parser.parse_args()

    runs = discover_curated_runs()
    curves = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index:02d}/{len(runs)}] Reading {run.robot}: {run.run_path.name}", flush=True)
        curves.append(load_curve(run))

    output_dir = args.output_dir.resolve()
    for robot in ROBOT_RUN_DIRS:
        output_path = output_dir / f"reward_vs_iterations_{robot}.svg"
        plot_robot_curves(robot, curves, output_path)
        print(f"Saved {output_path}", flush=True)
    summary_path = output_dir / "training_reward_summary.csv"
    write_summary(summary_path, curves)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
