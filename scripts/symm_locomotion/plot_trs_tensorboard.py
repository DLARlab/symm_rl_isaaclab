# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export matched no-TRS and TRS reward curves from TensorBoard logs."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from analyze_trs_grid import (
    LOG_ROOT,
    SAMPLES_PER_ITERATION,
    RunSpec,
    discover_runs,
    read_scalars,
    rolling_mean,
)

REWARD_TAG = "Train/mean_reward"
DEFAULT_SMOOTHING_WINDOW = 200
PLOT_STRIDE = 20
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ROBOT_TITLES = {
    "go2": "Unitree Go2",
    "x1": "Dobot X1",
}
COEFFICIENT_COLORS = {
    (0.10, 0.05): "#0072B2",
    (0.20, 0.10): "#D97706",
    (0.30, 0.15): "#00875A",
}
WARMUP_DASH_ARRAYS = {
    10: "2 5",
    100: "9 5",
    500: None,
}

ET.register_namespace("", SVG_NAMESPACE)


@dataclass(frozen=True)
class RewardCurve:
    """One smoothed TensorBoard reward curve."""

    run: RunSpec
    transitions_millions: tuple[float, ...]
    elapsed_hours: tuple[float, ...]
    rewards: tuple[float, ...]


def _svg_tag(name: str) -> str:
    """Return an SVG-qualified XML tag."""
    return f"{{{SVG_NAMESPACE}}}{name}"


def _event_path(run: RunSpec) -> Path:
    """Return the single TensorBoard event file for a run."""
    event_paths = sorted(run.run_path.glob("events.out.tfevents.*"))
    if len(event_paths) != 1:
        raise ValueError(f"Expected one TensorBoard event file in {run.run_path}, found {len(event_paths)}.")
    return event_paths[0]


def load_reward_curve(run: RunSpec, smoothing_window: int) -> RewardCurve:
    """Load and smooth the mean-reward TensorBoard series for one run."""
    event_path = _event_path(run)
    scalar_series = read_scalars(event_path, selected_tags={REWARD_TAG})
    if REWARD_TAG not in scalar_series:
        raise ValueError(f"Missing TensorBoard tag {REWARD_TAG!r} in {event_path}.")
    raw_points = scalar_series[REWARD_TAG]
    if len(raw_points) < smoothing_window:
        raise ValueError(
            f"TensorBoard series in {event_path} has {len(raw_points)} points, "
            f"fewer than smoothing window {smoothing_window}."
        )
    start_wall_time = raw_points[0][1]
    smoothed_points = rolling_mean(raw_points, smoothing_window)[smoothing_window - 1 :]
    sampled_points = smoothed_points[::PLOT_STRIDE]
    if sampled_points[-1] != smoothed_points[-1]:
        sampled_points.append(smoothed_points[-1])
    return RewardCurve(
        run=run,
        transitions_millions=tuple(
            step * SAMPLES_PER_ITERATION / 1_000_000.0 for step, _wall_time, _value in sampled_points
        ),
        elapsed_hours=tuple((wall_time - start_wall_time) / 3600.0 for _step, wall_time, _value in sampled_points),
        rewards=tuple(value for _step, _wall_time, value in sampled_points),
    )


def _sort_key(curve: RewardCurve) -> tuple[float, float, int]:
    """Sort the baseline first, followed by the TRS coefficient grid."""
    if not curve.run.trs_enabled:
        return (0.0, 0.0, 0)
    assert curve.run.warmup_iterations is not None
    return (
        curve.run.mirror_coeff,
        curve.run.value_coeff,
        curve.run.warmup_iterations,
    )


def _curve_color(curve: RewardCurve) -> str:
    """Return the stable line color for one curve."""
    if not curve.run.trs_enabled:
        return "#202124"
    return COEFFICIENT_COLORS[(curve.run.mirror_coeff, curve.run.value_coeff)]


def _curve_dash_array(curve: RewardCurve) -> str | None:
    """Return the warm-up-specific SVG dash pattern."""
    if not curve.run.trs_enabled:
        return None
    assert curve.run.warmup_iterations is not None
    return WARMUP_DASH_ARRAYS[curve.run.warmup_iterations]


def _curve_label(curve: RewardCurve) -> str:
    """Return a compact legend label."""
    if not curve.run.trs_enabled:
        return "No TRS"
    return f"{curve.run.mirror_coeff:.2f}/{curve.run.value_coeff:.2f}, warm-up {curve.run.warmup_iterations}"


def _nice_bounds(values: Sequence[float], tick_count: int = 6) -> tuple[float, float, list[float]]:
    """Return rounded plot bounds and evenly spaced tick values."""
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 1.0)
    raw_step = span / max(tick_count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1.0:
        nice_step = magnitude
    elif normalized <= 2.0:
        nice_step = 2.0 * magnitude
    elif normalized <= 5.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude
    lower = math.floor(minimum / nice_step) * nice_step
    upper = math.ceil(maximum / nice_step) * nice_step
    ticks = []
    value = lower
    while value <= upper + 0.5 * nice_step:
        ticks.append(value)
        value += nice_step
    return lower, upper, ticks


def _format_tick(value: float, tick_step: float) -> str:
    """Format an axis tick without unnecessary decimal places."""
    if tick_step >= 1.0:
        return f"{value:.0f}"
    decimals = max(1, int(math.ceil(-math.log10(tick_step))))
    return f"{value:.{decimals}f}"


def _add_text(
    parent: ET.Element,
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#202124",
    transform: str | None = None,
) -> ET.Element:
    """Append one SVG text element."""
    attributes = {
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        "font-size": str(size),
        "font-family": "Arial, Helvetica, sans-serif",
        "font-weight": str(weight),
        "text-anchor": anchor,
        "fill": fill,
    }
    if transform is not None:
        attributes["transform"] = transform
    text = ET.SubElement(parent, _svg_tag("text"), attributes)
    text.text = value
    return text


def _polyline_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    """Transform data coordinates into an SVG polyline point string."""
    x_span = x_max - x_min
    y_span = y_max - y_min
    return " ".join(
        f"{plot_left + (x_value - x_min) / x_span * plot_width:.2f},"
        f"{plot_top + (y_max - y_value) / y_span * plot_height:.2f}"
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )


def _draw_panel(
    root: ET.Element,
    definitions: ET.Element,
    curves: Sequence[RewardCurve],
    *,
    panel_index: int,
    x_field: str,
    title: str,
    x_label: str,
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
) -> None:
    """Draw one aligned TensorBoard reward panel."""
    plot_left = 105.0 + panel_index * 760.0
    plot_top = 135.0
    plot_width = 675.0
    plot_height = 470.0
    x_values = [value for curve in curves for value in getattr(curve, x_field)]
    x_min, x_max, x_ticks = _nice_bounds([0.0, *x_values])
    x_step = x_ticks[1] - x_ticks[0]
    y_step = y_ticks[1] - y_ticks[0]

    clip_id = f"panel-{panel_index}-clip"
    clip_path = ET.SubElement(definitions, _svg_tag("clipPath"), {"id": clip_id})
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

    _add_text(root, plot_left + plot_width / 2.0, 112.0, title, size=17, anchor="middle", weight=600)
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

    threshold_y = plot_top + (y_max - 35.0) / (y_max - y_min) * plot_height
    if plot_top <= threshold_y <= plot_top + plot_height:
        ET.SubElement(
            root,
            _svg_tag("line"),
            {
                "x1": f"{plot_left:.2f}",
                "x2": f"{plot_left + plot_width:.2f}",
                "y1": f"{threshold_y:.2f}",
                "y2": f"{threshold_y:.2f}",
                "stroke": "#7A7A7A",
                "stroke-width": "1.2",
                "stroke-dasharray": "4 4",
            },
        )
        ET.SubElement(
            root,
            _svg_tag("rect"),
            {
                "x": f"{plot_left + plot_width - 174.0:.2f}",
                "y": f"{threshold_y - 20.0:.2f}",
                "width": "170",
                "height": "18",
                "fill": "#FFFFFF",
                "fill-opacity": "0.86",
            },
        )
        _add_text(
            root,
            plot_left + plot_width - 7.0,
            threshold_y - 7.0,
            "sustained reward target",
            size=11,
            anchor="end",
            fill="#5F6368",
        )

    curve_group = ET.SubElement(root, _svg_tag("g"), {"clip-path": f"url(#{clip_id})"})
    for curve in curves:
        attributes = {
            "points": _polyline_points(
                getattr(curve, x_field),
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
            "stroke-width": "3.0" if not curve.run.trs_enabled else "1.8",
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
            "opacity": "1.0" if not curve.run.trs_enabled else "0.9",
        }
        dash_array = _curve_dash_array(curve)
        if dash_array is not None:
            attributes["stroke-dasharray"] = dash_array
        ET.SubElement(curve_group, _svg_tag("polyline"), attributes)

    _add_text(
        root,
        plot_left + plot_width / 2.0,
        plot_top + plot_height + 59.0,
        x_label,
        size=14,
        anchor="middle",
    )
    if panel_index == 0:
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


def _draw_legend(root: ET.Element, curves: Sequence[RewardCurve]) -> None:
    """Draw a two-row legend with explicit coefficient and warm-up labels."""
    columns = 5
    item_width = 300.0
    start_x = 70.0
    start_y = 710.0
    for index, curve in enumerate(curves):
        row, column = divmod(index, columns)
        item_x = start_x + column * item_width
        item_y = start_y + row * 42.0
        line_attributes = {
            "x1": f"{item_x:.2f}",
            "x2": f"{item_x + 45.0:.2f}",
            "y1": f"{item_y:.2f}",
            "y2": f"{item_y:.2f}",
            "stroke": _curve_color(curve),
            "stroke-width": "3.0" if not curve.run.trs_enabled else "2.0",
            "stroke-linecap": "round",
        }
        dash_array = _curve_dash_array(curve)
        if dash_array is not None:
            line_attributes["stroke-dasharray"] = dash_array
        ET.SubElement(root, _svg_tag("line"), line_attributes)
        _add_text(root, item_x + 55.0, item_y + 5.0, _curve_label(curve), size=12)


def plot_robot_curves(
    robot: str,
    curves: Sequence[RewardCurve],
    output_path: Path,
    smoothing_window: int,
) -> None:
    """Plot sample- and wall-clock-efficiency views for one robot."""
    robot_curves = sorted((curve for curve in curves if curve.run.robot == robot), key=_sort_key)
    if len(robot_curves) != 10:
        raise ValueError(f"Expected 10 reward curves for {robot}, found {len(robot_curves)}.")

    reward_values = [value for curve in robot_curves for value in curve.rewards]
    y_min, y_max, y_ticks = _nice_bounds([35.0, *reward_values])
    root = ET.Element(
        _svg_tag("svg"),
        {
            "viewBox": "0 0 1600 805",
            "width": "1600",
            "height": "805",
            "role": "img",
            "aria-labelledby": "plot-title plot-description",
        },
    )
    title = ET.SubElement(root, _svg_tag("title"), {"id": "plot-title"})
    title.text = f"{ROBOT_TITLES[robot]} no-TRS and TRS TensorBoard reward curves"
    description = ET.SubElement(root, _svg_tag("desc"), {"id": "plot-description"})
    description.text = (
        "The same ten smoothed mean-reward curves are plotted against environment transitions "
        "for sample efficiency and elapsed training hours for observed wall-clock efficiency."
    )
    ET.SubElement(
        root,
        _svg_tag("rect"),
        {"width": "1600", "height": "805", "fill": "#FFFFFF"},
    )
    definitions = ET.SubElement(root, _svg_tag("defs"))
    _add_text(
        root,
        800.0,
        43.0,
        f"{ROBOT_TITLES[robot]}: matched no-TRS and TRS training curves",
        size=24,
        anchor="middle",
        weight=600,
    )
    _add_text(
        root,
        800.0,
        72.0,
        (
            f"{REWARD_TAG}, {smoothing_window}-iteration trailing mean · seed 42 · "
            "512 environments · 24 steps/iteration · 20,000 iterations"
        ),
        size=13,
        anchor="middle",
        fill="#5F6368",
    )
    _draw_panel(
        root,
        definitions,
        robot_curves,
        panel_index=0,
        x_field="transitions_millions",
        title="Sample efficiency",
        x_label="Environment transitions [million]",
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
    )
    _draw_panel(
        root,
        definitions,
        robot_curves,
        panel_index=1,
        x_field="elapsed_hours",
        title="Observed wall-clock efficiency",
        x_label="Elapsed training time [h]",
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
    )
    _draw_legend(root, robot_curves)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    """Export the two robot-level TensorBoard comparison plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOG_ROOT / "good_runs/trs_grid_analysis",
        help="Directory for the exported TensorBoard plots.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Trailing mean window in training iterations.",
    )
    args = parser.parse_args()
    if args.smoothing_window < 1:
        parser.error("--smoothing-window must be positive")

    runs = discover_runs()
    curves = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index:02d}/{len(runs)}] Reading {run.key}", flush=True)
        curves.append(load_reward_curve(run, args.smoothing_window))

    output_dir = args.output_dir.resolve()
    for robot in ROBOT_TITLES:
        output_path = output_dir / f"tensorboard_reward_efficiency_{robot}.svg"
        plot_robot_curves(robot, curves, output_path, args.smoothing_window)
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
