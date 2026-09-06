# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build and render reproducible multi-run gait-closure comparisons.

The module consumes completed ``leg_usage_grid_full_v3`` evaluations and the
corresponding TensorBoard reward traces.  It deliberately keeps cohort
selection separate from metric aggregation so callers and tests can construct
comparisons with any cohort of two or more runs.  It is also the canonical
command-line entry point used by :mod:`comparison.sh` and
:mod:`comparison.ps1`.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import _tensorboard_scalars as tensorboard_scalars_module
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import study_registry as study_registry_module
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

METHOD_VERSION = "gait_closure_comparison_v1"
EVALUATION_METHOD_VERSION = "leg_usage_grid_full_v3"
DEFAULT_EVALUATION_SUBDIR = "evaluations/leg_usage_grid_full_v3"
DEFAULT_SMOOTHING_WINDOW_ITERATIONS = 200
DEFAULT_SAMPLE_STRIDE_ITERATIONS = 20
DEFAULT_THRESHOLD_REWARD = 35.0
COMPARISON_LAUNCHER_FILES = {
    "python": "comparison.py",
    "bash": "comparison.sh",
    "powershell": "comparison.ps1",
}

FAMILIES = ("trot", "bound", "half_bound", "gallop")
EXPECTED_FAMILY_COUNTS = {"trot": 6, "bound": 6, "half_bound": 24, "gallop": 24}
VELOCITY_SCOPES = ("all", "negative", "positive")
EXPECTED_SCOPE_COUNTS = {"all": 60, "negative": 30, "positive": 30}

V2_PALETTE = (
    "#202124",
    "#0072B2",
    "#56B4E9",
    "#D97706",
    "#00875A",
    "#CC79A7",
    "#8E44AD",
    "#A65E2E",
)
BASELINE_COLOR = V2_PALETTE[0]

INITIALIZATION_METADATA_SOURCE = "pre_first_rollout_initialization"
LEGACY_CONFIG_METADATA_SOURCE = "legacy_registered_run_snapshot"
LEGACY_CHECKSUM_ONLY_METADATA_SOURCE = "legacy_registered_run_snapshot_checksum_only"
LEGACY_INITIALIZATION_RECORD_TYPE = "legacy_initialization_checkpoint_checksum"

VELOCITY_METRICS: dict[str, dict[str, Any]] = {
    "velocity_vx_rmse_mps": {"display_label": "Forward-velocity RMSE", "unit": "m/s", "lower_is_better": True},
    "velocity_yaw_rmse_radps": {"display_label": "Yaw-velocity RMSE", "unit": "rad/s", "lower_is_better": True},
}
LEG_METRICS: dict[str, dict[str, Any]] = {
    "torque_squared": {
        "source_metric": "torque_squared",
        "display_label": "Torque²",
        "unit": "%",
        "lower_is_better": True,
    },
    "normalized_torque_squared": {
        "source_metric": "normalized_torque_utilization",
        "display_label": "Normalized torque²",
        "unit": "%",
        "lower_is_better": True,
    },
    "absolute_work": {
        "source_metric": "absolute_work",
        "display_label": "Absolute work",
        "unit": "%",
        "lower_is_better": True,
    },
    "vertical_grf_impulse": {
        "source_metric": "vertical_grf_impulse",
        "display_label": "Vertical GRF impulse",
        "unit": "%",
        "lower_is_better": True,
    },
}
GAIT_METRICS: dict[str, dict[str, Any]] = {
    "gait_agreement_boundary_excluded_percent": {
        "display_label": "Gait agreement",
        "unit": "%",
        "lower_is_better": False,
    },
}

STYLE: dict[str, Any] = {
    "background": "#FFFFFF",
    "text": "#202124",
    "category_text": "#3C4043",
    "muted_text": "#5F6368",
    "grid": "#E2E5E9",
    "learning_grid_y": "#DADCE0",
    "learning_grid_x": "#EEF0F2",
    "axis": "#5F6368",
    "zero_line": "#9AA0A6",
    "target_line": "#7A7A7A",
    "baseline": "#202124",
    "palette": (
        "#202124",
        "#0072B2",
        "#56B4E9",
        "#D97706",
        "#00875A",
        "#CC79A7",
        "#8E44AD",
        "#A65E2E",
    ),
    "font_family": ("Arial", "Helvetica", "DejaVu Sans", "sans-serif"),
    "title_size": 24,
    "subtitle_size": 13,
    "panel_title_size": 17,
    "axis_label_size": 14,
    "tick_size": 12,
    "legend_size": 12,
    "value_size": 10,
    "title_weight": 600,
    "matplotlib_title_weight": "semibold",
    "bar_alpha": 0.88,
    "baseline_line_width": 3.0,
    "treatment_line_width": 1.8,
    "dpi": 100,
}

FIGURE_REGISTRY: dict[str, dict[str, Any]] = {
    "fig01": {
        "stem": "fig01_training_efficiency",
        "title": "Training efficiency",
        "kind": "figure",
        "source_tables": ("training_efficiency.csv", "learning_curve_points.csv"),
    },
    "fig02": {
        "stem": "fig02_velocity_tracking_overall_direction",
        "title": "Velocity tracking: overall and by direction",
        "kind": "figure",
        "source_tables": ("velocity_tracking_summary.csv",),
    },
    "fig03_01": {
        "stem": "fig03_01_velocity_tracking_by_family_all",
        "title": "Velocity tracking by gait family: all velocities",
        "kind": "figure",
        "source_tables": ("velocity_tracking_summary.csv",),
    },
    "fig03_02": {
        "stem": "fig03_02_velocity_tracking_by_family_negative",
        "title": "Velocity tracking by gait family: negative velocities",
        "kind": "figure",
        "source_tables": ("velocity_tracking_summary.csv",),
    },
    "fig03_03": {
        "stem": "fig03_03_velocity_tracking_by_family_positive",
        "title": "Velocity tracking by gait family: positive velocities",
        "kind": "figure",
        "source_tables": ("velocity_tracking_summary.csv",),
    },
    "fig03": {
        "stem": "fig03_velocity_tracking_by_family_combined",
        "title": "Velocity tracking by gait family: combined",
        "kind": "combined",
        "children": ("fig03_01", "fig03_02", "fig03_03"),
        "source_tables": ("velocity_tracking_summary.csv",),
    },
    "fig04": {
        "stem": "fig04_leg_usage_overall_direction",
        "title": "Leg usage: overall and by direction",
        "kind": "figure",
        "source_tables": ("leg_usage_summary.csv",),
    },
    "fig05_01": {
        "stem": "fig05_01_leg_usage_by_family_all",
        "title": "Leg usage by gait family: all velocities",
        "kind": "figure",
        "source_tables": ("leg_usage_summary.csv",),
    },
    "fig05_02": {
        "stem": "fig05_02_leg_usage_by_family_negative",
        "title": "Leg usage by gait family: negative velocities",
        "kind": "figure",
        "source_tables": ("leg_usage_summary.csv",),
    },
    "fig05_03": {
        "stem": "fig05_03_leg_usage_by_family_positive",
        "title": "Leg usage by gait family: positive velocities",
        "kind": "figure",
        "source_tables": ("leg_usage_summary.csv",),
    },
    "fig05": {
        "stem": "fig05_leg_usage_by_family_combined",
        "title": "Leg usage by gait family: combined",
        "kind": "combined",
        "children": ("fig05_01", "fig05_02", "fig05_03"),
        "source_tables": ("leg_usage_summary.csv",),
    },
    "fig06": {
        "stem": "fig06_gait_fidelity_overall_direction",
        "title": "Gait fidelity: overall and by direction",
        "kind": "figure",
        "source_tables": ("gait_fidelity_summary.csv",),
    },
    "fig07_01": {
        "stem": "fig07_01_gait_fidelity_by_family_all",
        "title": "Gait fidelity by family: all velocities",
        "kind": "figure",
        "source_tables": ("gait_fidelity_summary.csv",),
    },
    "fig07_02": {
        "stem": "fig07_02_gait_fidelity_by_family_negative",
        "title": "Gait fidelity by family: negative velocities",
        "kind": "figure",
        "source_tables": ("gait_fidelity_summary.csv",),
    },
    "fig07_03": {
        "stem": "fig07_03_gait_fidelity_by_family_positive",
        "title": "Gait fidelity by family: positive velocities",
        "kind": "figure",
        "source_tables": ("gait_fidelity_summary.csv",),
    },
    "fig07": {
        "stem": "fig07_gait_fidelity_by_family_combined",
        "title": "Gait fidelity by family: combined",
        "kind": "combined",
        "children": ("fig07_01", "fig07_02", "fig07_03"),
        "source_tables": ("gait_fidelity_summary.csv",),
    },
}

VELOCITY_SCOPE_LABELS = {
    "all": "All velocities",
    "negative": "Negative velocities",
    "positive": "Positive velocities",
}
FAMILY_LABELS = {
    "trot": "Trot",
    "bound": "Bound",
    "half_bound": "Half-bound",
    "gallop": "Gallop",
}
PLOT_VELOCITY_METRICS = (
    ("forward_velocity_rmse", "Forward velocity RMSE [m/s]"),
    ("yaw_velocity_rmse", "Yaw-velocity RMSE [rad/s]"),
)
PLOT_LEG_METRICS = (
    ("torque_squared", "Torque²"),
    ("normalized_torque_squared", "Normalized torque²"),
    ("absolute_work", "Absolute work"),
    ("vertical_grf_impulse", "Vertical GRF impulse"),
)
GAIT_METRIC = "gait_agreement"
HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\Z")

_METRIC_ALIASES = {
    "forward_velocity_rmse": "forward_velocity_rmse",
    "forward_velocity_rmse_mps": "forward_velocity_rmse",
    "velocity_vx_rmse": "forward_velocity_rmse",
    "velocity_vx_rmse_mps": "forward_velocity_rmse",
    "vx_rmse": "forward_velocity_rmse",
    "yaw_velocity_rmse": "yaw_velocity_rmse",
    "yaw_velocity_rmse_radps": "yaw_velocity_rmse",
    "velocity_yaw_rmse": "yaw_velocity_rmse",
    "velocity_yaw_rmse_radps": "yaw_velocity_rmse",
    "torque_squared": "torque_squared",
    "normalized_torque_squared": "normalized_torque_squared",
    "normalized_torque_utilization": "normalized_torque_squared",
    "absolute_work": "absolute_work",
    "vertical_grf_impulse": "vertical_grf_impulse",
    "gait_agreement": "gait_agreement",
    "gait_agreement_percent": "gait_agreement",
    "gait_agreement_boundary_excluded": "gait_agreement",
    "gait_agreement_boundary_excluded_percent": "gait_agreement",
}

CORE_OUTPUT_FILES = (
    "study.json",
    "style.json",
    "training_efficiency.csv",
    "learning_curve_points.csv",
    "evaluation_cells.csv",
    "velocity_tracking_summary.csv",
    "leg_usage_summary.csv",
    "gait_fidelity_summary.csv",
    "summary.json",
    "figure_manifest.json",
    "REPORT.md",
    "reproduce.py",
    "source_manifest_snapshot.json",
)


def _rc_params() -> dict[str, Any]:
    """Return isolated Matplotlib settings matching the V2 SVG grammar."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(STYLE["font_family"][:-1]),
        "font.size": STYLE["tick_size"],
        "text.color": STYLE["text"],
        "axes.labelcolor": STYLE["text"],
        "axes.titlecolor": STYLE["text"],
        "xtick.color": STYLE["muted_text"],
        "ytick.color": STYLE["muted_text"],
        "figure.facecolor": STYLE["background"],
        "axes.facecolor": STYLE["background"],
        "savefig.facecolor": STYLE["background"],
        "svg.fonttype": "none",
        "svg.hashsalt": "isaaclab-gait-closure-v4",
    }


def _as_float(value: Any, context: str) -> float:
    """Return one finite floating-point value."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a numeric value for {context}, received {value!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite value for {context}, received {result!r}.")
    return result


def _resolve_runs(study: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate and normalize the ordered run descriptors."""
    raw_runs = study.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, str | bytes) or len(raw_runs) < 2:
        raise ValueError("study.runs must contain at least two run descriptors.")
    runs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"study.runs[{index}] must be an object.")
        run_id = str(raw.get("run_id", "")).strip()
        abbreviation = str(raw.get("abbreviation", "")).strip()
        if not run_id or not abbreviation:
            raise ValueError(f"study.runs[{index}] requires nonempty run_id and abbreviation fields.")
        is_baseline = bool(raw.get("is_baseline", False))
        supplied_color = str(raw.get("color", "")).upper()
        if HEX_COLOR.fullmatch(supplied_color) is None:
            raise ValueError(f"Run {run_id!r} requires a six-digit hexadecimal color.")
        color = STYLE["baseline"] if is_baseline else supplied_color
        if not is_baseline and color == STYLE["baseline"]:
            raise ValueError(f"Run {run_id!r} is not the baseline but uses the reserved baseline black.")
        runs.append(
            {
                "run_id": run_id,
                "abbreviation": abbreviation,
                "color": color,
                "is_baseline": is_baseline,
                "source_index": index,
            }
        )
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("study.runs contains duplicate run_id values.")
    if len({run["abbreviation"] for run in runs}) != len(runs):
        raise ValueError("study.runs contains duplicate abbreviations.")
    if sum(run["is_baseline"] for run in runs) != 1:
        raise ValueError("study.runs must mark exactly one no-TRS baseline.")
    if len({run["color"] for run in runs}) != len(runs):
        raise ValueError("study.runs must use unique colors after reserving black for the baseline.")
    runs.sort(key=lambda run: (not run["is_baseline"], run["source_index"]))
    return tuple(runs)


def _canonical_metric(metric: Any) -> str | None:
    """Return the canonical metric name, or ``None`` for an unrelated metric."""
    return _METRIC_ALIASES.get(str(metric).strip().lower())


def _normalize_velocity_scope(scope: Any) -> str:
    """Normalize one signed-velocity aggregation scope."""
    normalized = str(scope).strip().lower()
    aliases = {"backward": "negative", "forward": "positive", "overall": "all"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in VELOCITY_SCOPES:
        raise ValueError(f"Unsupported velocity_scope {scope!r}.")
    return normalized


def _normalize_family(family: Any) -> str:
    """Normalize one gait-family identifier."""
    normalized = "" if family is None else str(family).strip().lower().replace("-", "_")
    if normalized in {"", "all", "overall", "all_family_balanced"}:
        return "all_family_balanced"
    if normalized not in FAMILIES:
        raise ValueError(f"Unsupported gait family {family!r}.")
    return normalized


def _summary_index(
    rows: Sequence[Mapping[str, Any]], allowed_metrics: set[str], run_ids: set[str]
) -> dict[tuple[str, str, str, str, str], float]:
    """Index a long summary table and reject ambiguous required rows."""
    index: dict[tuple[str, str, str, str, str], float] = {}
    for row_number, row in enumerate(rows, start=1):
        metric = _canonical_metric(row.get("metric"))
        if metric not in allowed_metrics:
            continue
        run_id = str(row.get("run_id", "")).strip()
        if run_id not in run_ids:
            raise ValueError(f"Summary row {row_number} references unknown run_id {run_id!r}.")
        velocity_scope = _normalize_velocity_scope(row.get("velocity_scope"))
        aggregation = str(row.get("aggregation", row.get("aggregation_scope", ""))).strip().lower()
        if aggregation not in {"overall", "family"}:
            raise ValueError(f"Summary row {row_number} has unsupported aggregation {aggregation!r}.")
        family = _normalize_family(row.get("family"))
        if aggregation == "overall" and family != "all_family_balanced":
            raise ValueError(f"Overall summary row {row_number} must use family='all_family_balanced'.")
        if aggregation == "family" and family == "all_family_balanced":
            raise ValueError(f"Family summary row {row_number} requires an individual gait family.")
        key = (run_id, velocity_scope, aggregation, family, metric)
        if key in index:
            raise ValueError(f"Duplicate summary row for {key}.")
        value = _as_float(row.get("value"), f"summary row {row_number}")
        if value < 0.0:
            raise ValueError(f"Comparison metrics must be nonnegative; {key} has value {value}.")
        if metric == GAIT_METRIC and value > 100.0:
            raise ValueError(f"Gait agreement must be in [0, 100]; {key} has value {value}.")
        index[key] = value
    return index


def _value(
    index: Mapping[tuple[str, str, str, str, str], float],
    run_id: str,
    velocity_scope: str,
    aggregation: str,
    family: str,
    metric: str,
) -> float:
    """Return one required summary value with an actionable missing-data error."""
    key = (run_id, velocity_scope, aggregation, family, metric)
    try:
        return index[key]
    except KeyError as error:
        raise ValueError(f"Missing required comparison summary row for {key}.") from error


def _nice_bounds(values: Sequence[float], tick_count: int = 6) -> tuple[float, float, list[float]]:
    """Return the same rounded bounds used by the V2 learning plot."""
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
    ticks: list[float] = []
    tick = lower
    while tick <= upper + 0.5 * nice_step:
        ticks.append(tick)
        tick += nice_step
    return lower, upper, ticks


def _positive_upper(values: Sequence[float], *, minimum: float = 0.0) -> float:
    """Return a readable positive upper bound with label headroom."""
    if minimum < 0.0:
        raise ValueError("A positive-axis minimum cannot be negative.")
    maximum = max(values, default=minimum)
    target = max(maximum * 1.10, minimum)
    if target <= 0.0:
        return 1.0
    raw_step = target / 5.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1.0:
        nice_step = magnitude
    elif normalized <= 2.0:
        nice_step = 2.0 * magnitude
    elif normalized <= 2.5:
        nice_step = 2.5 * magnitude
    elif normalized <= 5.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude
    return max(math.ceil(target / nice_step) * nice_step, minimum)


def _figure_header(figure: Figure, title: str, subtitle: str) -> None:
    """Apply the centered V2 title and subtitle hierarchy."""
    title_y = 0.982
    subtitle_y = title_y - 0.50 / figure.get_figheight()
    figure.suptitle(
        title,
        x=0.5,
        y=title_y,
        va="top",
        fontsize=STYLE["title_size"],
        fontweight=STYLE["matplotlib_title_weight"],
        color=STYLE["text"],
    )
    figure.text(
        0.5,
        subtitle_y,
        subtitle,
        ha="center",
        va="top",
        fontsize=STYLE["subtitle_size"],
        color=STYLE["muted_text"],
    )


def _style_bar_axis(axis: Axes) -> None:
    """Apply the V2 grouped-bar axes and horizontal grid."""
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=STYLE["grid"], linewidth=1.0)
    axis.axhline(0.0, color=STYLE["zero_line"], linewidth=1.5, zorder=1)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", which="both", length=0, labelsize=STYLE["tick_size"])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))


def _legend_handles(runs: Sequence[Mapping[str, Any]], *, lines: bool = False) -> list[Any]:
    """Return run-colored handles in baseline-first order."""
    if lines:
        return [
            Line2D(
                [0],
                [0],
                color=run["color"],
                linewidth=STYLE["baseline_line_width"] if run["is_baseline"] else 2.0,
                solid_capstyle="round",
            )
            for run in runs
        ]
    return [Patch(facecolor=run["color"], edgecolor="none", alpha=STYLE["bar_alpha"]) for run in runs]


def _legend_column_order(item_count: int, column_count: int) -> list[int]:
    """Return input indices that Matplotlib renders in visual row-major order."""
    row_count = math.ceil(item_count / column_count)
    return [
        row * column_count + column
        for column in range(column_count)
        for row in range(row_count)
        if row * column_count + column < item_count
    ]


def _add_shared_legend(figure: Figure, runs: Sequence[Mapping[str, Any]], *, lines: bool = False) -> None:
    """Add a compact shared V2 legend with baseline-first visual ordering."""
    column_count = min(5, len(runs))
    order = _legend_column_order(len(runs), column_count)
    handles = _legend_handles(runs, lines=lines)
    labels = [str(run["abbreviation"]) for run in runs]
    figure.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=column_count,
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=2.8 if lines else 1.8,
        columnspacing=2.2,
        handletextpad=0.8,
    )


def _draw_grouped_bars(
    axis: Axes,
    categories: Sequence[str],
    values_by_run: Sequence[Sequence[float]],
    runs: Sequence[Mapping[str, Any]],
    *,
    value_format: str,
    y_upper: float,
) -> None:
    """Draw one V2-style grouped-bar panel."""
    _style_bar_axis(axis)
    run_count = len(runs)
    group_fraction = 0.72
    slot_width = group_fraction / run_count
    x_positions = list(range(len(categories)))
    for run_index, (run, values) in enumerate(zip(runs, values_by_run, strict=True)):
        offset = (run_index - (run_count - 1) / 2.0) * slot_width
        centers = [position + offset for position in x_positions]
        bars = axis.bar(
            centers,
            values,
            width=slot_width * 0.92,
            color=run["color"],
            alpha=STYLE["bar_alpha"],
            edgecolor="none",
            zorder=2,
        )
        for bar, value in zip(bars, values, strict=True):
            label_offset = y_upper * (0.014 + 0.013 * (run_index % 2))
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + label_offset,
                format(value, value_format),
                ha="center",
                va="bottom",
                fontsize=STYLE["value_size"],
                color=STYLE["category_text"],
                clip_on=False,
            )
    axis.set_xticks(x_positions, categories, color=STYLE["category_text"])
    axis.set_xlim(-0.5, len(categories) - 0.5)
    axis.set_ylim(0.0, y_upper)


def _save_figure(figure: Figure, output_dir: Path, figure_id: str) -> dict[str, str]:
    """Save deterministic SVG and RGB PNG artifacts for one figure."""
    entry = FIGURE_REGISTRY[figure_id]
    stem = str(entry["stem"])
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    temporary_svg = output_dir / f".{stem}.svg.tmp"
    temporary_png = output_dir / f".{stem}.png.tmp"
    figure.savefig(
        temporary_svg,
        format="svg",
        dpi=STYLE["dpi"],
        facecolor=STYLE["background"],
        metadata={"Date": None, "Creator": "IsaacLab gait-closure comparison"},
    )
    figure.savefig(
        temporary_png,
        format="png",
        dpi=STYLE["dpi"],
        facecolor=STYLE["background"],
        metadata={"Software": "IsaacLab gait-closure comparison"},
    )
    with Image.open(temporary_png) as image:
        image.convert("RGB").save(png_path, format="PNG", compress_level=6, optimize=False)
    temporary_png.unlink()
    os.replace(temporary_svg, svg_path)
    return {"figure_id": figure_id, "stem": stem, "png": png_path.name, "svg": svg_path.name}


def _learning_figure(
    output_dir: Path,
    study: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    learning_points: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Render Fig. 1 with matched sample- and wall-time learning curves."""
    points_by_run: dict[str, list[dict[str, float]]] = {str(run["run_id"]): [] for run in runs}
    for row_number, point in enumerate(learning_points, start=1):
        run_id = str(point.get("run_id", "")).strip()
        if run_id not in points_by_run:
            raise ValueError(f"Learning row {row_number} references unknown run_id {run_id!r}.")
        points_by_run[run_id].append(
            {
                "transitions_millions": _as_float(
                    point.get("transitions_millions"), f"learning row {row_number} transitions_millions"
                ),
                "elapsed_hours": _as_float(point.get("elapsed_hours"), f"learning row {row_number} elapsed_hours"),
                "smoothed_reward": _as_float(
                    point.get("smoothed_reward"), f"learning row {row_number} smoothed_reward"
                ),
            }
        )
    for run in runs:
        if len(points_by_run[str(run["run_id"])]) < 2:
            raise ValueError(f"Run {run['run_id']!r} requires at least two learning-curve points.")

    reward_target = _as_float(study.get("reward_target", 35.0), "study.reward_target")
    all_rewards = [point["smoothed_reward"] for points in points_by_run.values() for point in points]
    y_min, y_max, y_ticks = _nice_bounds([reward_target, *all_rewards])
    figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.05), sharey=True)
    display_name = str(study.get("display_name", "Unitree Go2"))
    _figure_header(
        figure,
        f"Fig. 1 · {display_name}: no-TRS and TRS learning curves",
        _learning_subtitle(study),
    )
    panel_specs = (
        ("transitions_millions", "Sample efficiency", "Environment transitions [million]"),
        ("elapsed_hours", "Observed wall-clock efficiency", "Elapsed training time [h]"),
    )
    for panel_index, (axis, (x_field, panel_title, x_label)) in enumerate(zip(axes, panel_specs, strict=True)):
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color=STYLE["learning_grid_y"], linewidth=1.0)
        axis.xaxis.grid(True, color=STYLE["learning_grid_x"], linewidth=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(STYLE["axis"])
        axis.spines["bottom"].set_color(STYLE["axis"])
        axis.spines["left"].set_linewidth(1.2)
        axis.spines["bottom"].set_linewidth(1.2)
        axis.tick_params(labelsize=STYLE["tick_size"], length=0)
        all_x = [point[x_field] for points in points_by_run.values() for point in points]
        x_min, x_max, x_ticks = _nice_bounds([0.0, *all_x])
        for run in runs:
            points = sorted(points_by_run[str(run["run_id"])], key=lambda point: point[x_field])
            axis.plot(
                [point[x_field] for point in points],
                [point["smoothed_reward"] for point in points],
                color=run["color"],
                linewidth=(STYLE["baseline_line_width"] if run["is_baseline"] else STYLE["treatment_line_width"]),
                alpha=1.0 if run["is_baseline"] else 0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )
        axis.axhline(
            reward_target,
            color=STYLE["target_line"],
            linewidth=1.2,
            linestyle=(0, (4, 4)),
            zorder=2,
        )
        axis.text(
            0.99,
            reward_target,
            "sustained reward target",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=11,
            color=STYLE["muted_text"],
            bbox={"facecolor": STYLE["background"], "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        )
        axis.set_title(
            panel_title,
            fontsize=STYLE["panel_title_size"],
            fontweight=STYLE["matplotlib_title_weight"],
            pad=18,
        )
        axis.set_xlabel(x_label, fontsize=STYLE["axis_label_size"], labelpad=16)
        if panel_index == 0:
            axis.set_ylabel("Train/mean_reward", fontsize=STYLE["axis_label_size"], labelpad=22)
        axis.set_xlim(x_min, x_max)
        axis.set_xticks(x_ticks)
        axis.set_ylim(y_min, y_max)
        axis.set_yticks(y_ticks)
    _add_shared_legend(figure, runs, lines=True)
    figure.subplots_adjust(left=0.065, right=0.97, top=0.83, bottom=0.19, wspace=0.13)
    artifact = _save_figure(figure, output_dir, "fig01")
    plt.close(figure)
    return artifact


def _learning_subtitle(study: Mapping[str, Any]) -> str:
    """Return the V2-style learning subtitle from available study metadata."""
    training = study.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    smoothing = training.get("smoothing_window_iterations", study.get("smoothing_window_iterations", 200))
    fragments = [f"Train/mean_reward, {smoothing}-iteration trailing mean"]
    optional = (
        ("seed", "seed {}"),
        ("num_envs", "{} environments"),
        ("num_steps_per_env", "{} steps/iteration"),
        ("max_iterations", "{} iterations"),
    )
    for key, template in optional:
        if key in training:
            fragments.append(template.format(training[key]))
    return " · ".join(fragments)


def _velocity_overall_figure(
    output_dir: Path,
    display_name: str,
    runs: Sequence[Mapping[str, Any]],
    index: Mapping[tuple[str, str, str, str, str], float],
) -> dict[str, str]:
    """Render Fig. 2 as three velocity scopes by two scientifically separate metrics."""
    figure, axes = plt.subplots(3, 2, figsize=(20.0, 15.0), squeeze=False)
    _figure_header(
        figure,
        f"Fig. 2 · {display_name}: velocity tracking",
        "Overall and signed-velocity comparisons · lower is better · shared run colors",
    )
    values_by_metric: dict[str, list[float]] = {metric: [] for metric, _label in PLOT_VELOCITY_METRICS}
    for metric, _label in PLOT_VELOCITY_METRICS:
        for scope in VELOCITY_SCOPES:
            values_by_metric[metric].extend(
                _value(index, str(run["run_id"]), scope, "overall", "all_family_balanced", metric) for run in runs
            )
    y_uppers = {metric: _positive_upper(values) for metric, values in values_by_metric.items()}
    for row_index, scope in enumerate(VELOCITY_SCOPES):
        for column_index, (metric, label) in enumerate(PLOT_VELOCITY_METRICS):
            axis = axes[row_index, column_index]
            values = [
                _value(index, str(run["run_id"]), scope, "overall", "all_family_balanced", metric) for run in runs
            ]
            _draw_grouped_bars(
                axis,
                [label],
                [[value] for value in values],
                runs,
                value_format=".3f",
                y_upper=y_uppers[metric],
            )
            axis.set_title(
                f"{VELOCITY_SCOPE_LABELS[scope]} · {label}",
                fontsize=STYLE["panel_title_size"],
                fontweight=STYLE["matplotlib_title_weight"],
                pad=12,
            )
            axis.set_ylabel(label, fontsize=STYLE["axis_label_size"], labelpad=16)
            axis.set_xticklabels([])
    _add_shared_legend(figure, runs)
    figure.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.105, hspace=0.42, wspace=0.19)
    artifact = _save_figure(figure, output_dir, "fig02")
    plt.close(figure)
    return artifact


def _velocity_family_figure(
    output_dir: Path,
    figure_id: str,
    display_name: str,
    scope: str,
    runs: Sequence[Mapping[str, Any]],
    index: Mapping[tuple[str, str, str, str, str], float],
    y_uppers: Mapping[str, float],
) -> dict[str, str]:
    """Render one Fig. 3 child with two gait-family metric panels."""
    figure, axes = plt.subplots(2, 1, figsize=(20.0, 10.5), squeeze=False)
    number = {"all": "3.1", "negative": "3.2", "positive": "3.3"}[scope]
    _figure_header(
        figure,
        f"Fig. {number} · {display_name}: velocity tracking by gait family",
        f"{VELOCITY_SCOPE_LABELS[scope]} · lower is better · each family is reported independently",
    )
    for row_index, (metric, label) in enumerate(PLOT_VELOCITY_METRICS):
        axis = axes[row_index, 0]
        values_by_run = [
            [_value(index, str(run["run_id"]), scope, "family", family, metric) for family in FAMILIES] for run in runs
        ]
        _draw_grouped_bars(
            axis,
            [FAMILY_LABELS[family] for family in FAMILIES],
            values_by_run,
            runs,
            value_format=".3f",
            y_upper=y_uppers[metric],
        )
        axis.set_title(
            label,
            fontsize=STYLE["panel_title_size"],
            fontweight=STYLE["matplotlib_title_weight"],
            pad=12,
        )
        axis.set_ylabel(label, fontsize=STYLE["axis_label_size"], labelpad=16)
    _add_shared_legend(figure, runs)
    figure.subplots_adjust(left=0.085, right=0.975, top=0.855, bottom=0.15, hspace=0.43)
    artifact = _save_figure(figure, output_dir, figure_id)
    plt.close(figure)
    return artifact


def _leg_overall_figure(
    output_dir: Path,
    display_name: str,
    runs: Sequence[Mapping[str, Any]],
    index: Mapping[tuple[str, str, str, str, str], float],
) -> dict[str, str]:
    """Render Fig. 4 as three velocity-scope leg-usage panels."""
    figure, axes = plt.subplots(3, 1, figsize=(20.0, 14.5), squeeze=False)
    _figure_header(
        figure,
        f"Fig. 4 · {display_name}: front/hind leg-usage balance",
        "Mean absolute front/hind imbalance · lower is more even · equal-family overall summaries",
    )
    all_values = [
        _value(index, str(run["run_id"]), scope, "overall", "all_family_balanced", metric)
        for scope in VELOCITY_SCOPES
        for metric, _label in PLOT_LEG_METRICS
        for run in runs
    ]
    y_upper = _positive_upper(all_values, minimum=10.0)
    figure.text(
        0.025,
        0.5,
        "Mean absolute front/hind imbalance [%]",
        ha="center",
        va="center",
        rotation=90,
        fontsize=STYLE["axis_label_size"],
        color=STYLE["text"],
    )
    for row_index, scope in enumerate(VELOCITY_SCOPES):
        axis = axes[row_index, 0]
        values_by_run = [
            [
                _value(index, str(run["run_id"]), scope, "overall", "all_family_balanced", metric)
                for metric, _label in PLOT_LEG_METRICS
            ]
            for run in runs
        ]
        _draw_grouped_bars(
            axis,
            [label for _metric, label in PLOT_LEG_METRICS],
            values_by_run,
            runs,
            value_format=".1f",
            y_upper=y_upper,
        )
        axis.set_title(
            VELOCITY_SCOPE_LABELS[scope],
            fontsize=STYLE["panel_title_size"],
            fontweight=STYLE["matplotlib_title_weight"],
            pad=12,
        )
    _add_shared_legend(figure, runs)
    figure.subplots_adjust(left=0.095, right=0.975, top=0.875, bottom=0.11, hspace=0.43)
    artifact = _save_figure(figure, output_dir, "fig04")
    plt.close(figure)
    return artifact


def _leg_family_figure(
    output_dir: Path,
    figure_id: str,
    display_name: str,
    scope: str,
    runs: Sequence[Mapping[str, Any]],
    index: Mapping[tuple[str, str, str, str, str], float],
    y_upper: float,
) -> dict[str, str]:
    """Render one Fig. 5 child with one four-metric panel per gait family."""
    figure, axes = plt.subplots(4, 1, figsize=(20.0, 15.0), squeeze=False)
    number = {"all": "5.1", "negative": "5.2", "positive": "5.3"}[scope]
    _figure_header(
        figure,
        f"Fig. {number} · {display_name}: leg usage by gait family",
        f"{VELOCITY_SCOPE_LABELS[scope]} · mean absolute front/hind imbalance · lower is more even",
    )
    figure.text(
        0.025,
        0.5,
        "Mean absolute front/hind imbalance [%]",
        ha="center",
        va="center",
        rotation=90,
        fontsize=STYLE["axis_label_size"],
        color=STYLE["text"],
    )
    for row_index, family in enumerate(FAMILIES):
        axis = axes[row_index, 0]
        values_by_run = [
            [_value(index, str(run["run_id"]), scope, "family", family, metric) for metric, _label in PLOT_LEG_METRICS]
            for run in runs
        ]
        _draw_grouped_bars(
            axis,
            [label for _metric, label in PLOT_LEG_METRICS],
            values_by_run,
            runs,
            value_format=".1f",
            y_upper=y_upper,
        )
        axis.set_title(
            FAMILY_LABELS[family],
            fontsize=STYLE["panel_title_size"],
            fontweight=STYLE["matplotlib_title_weight"],
            pad=10,
        )
    _add_shared_legend(figure, runs)
    figure.subplots_adjust(left=0.095, right=0.975, top=0.885, bottom=0.1, hspace=0.5)
    artifact = _save_figure(figure, output_dir, figure_id)
    plt.close(figure)
    return artifact


def _gait_overall_figure(
    output_dir: Path,
    display_name: str,
    runs: Sequence[Mapping[str, Any]],
    index: Mapping[tuple[str, str, str, str, str], float],
) -> dict[str, str]:
    """Render Fig. 6 as three overall gait-agreement panels."""
    figure, axes = plt.subplots(3, 1, figsize=(20.0, 12.5), squeeze=False)
    _figure_header(
        figure,
        f"Fig. 6 · {display_name}: gait fidelity",
        "Gait agreement only · higher is better · equal-family overall summaries",
    )
    for row_index, scope in enumerate(VELOCITY_SCOPES):
        axis = axes[row_index, 0]
        values = [
            _value(index, str(run["run_id"]), scope, "overall", "all_family_balanced", GAIT_METRIC) for run in runs
        ]
        _draw_grouped_bars(
            axis,
            ["Gait agreement"],
            [[value] for value in values],
            runs,
            value_format=".1f",
            y_upper=100.0,
        )
        axis.set_title(
            VELOCITY_SCOPE_LABELS[scope],
            fontsize=STYLE["panel_title_size"],
            fontweight=STYLE["matplotlib_title_weight"],
            pad=12,
        )
        axis.set_ylabel("Gait agreement [%]", fontsize=STYLE["axis_label_size"], labelpad=16)
        axis.set_xticklabels([])
    _add_shared_legend(figure, runs)
    figure.subplots_adjust(left=0.08, right=0.975, top=0.87, bottom=0.12, hspace=0.43)
    artifact = _save_figure(figure, output_dir, "fig06")
    plt.close(figure)
    return artifact


def _gait_family_figure(
    output_dir: Path,
    figure_id: str,
    display_name: str,
    scope: str,
    runs: Sequence[Mapping[str, Any]],
    index: Mapping[tuple[str, str, str, str, str], float],
) -> dict[str, str]:
    """Render one Fig. 7 child with gait agreement by family."""
    figure, axis = plt.subplots(1, 1, figsize=(20.0, 8.0))
    number = {"all": "7.1", "negative": "7.2", "positive": "7.3"}[scope]
    _figure_header(
        figure,
        f"Fig. {number} · {display_name}: gait fidelity by family",
        f"{VELOCITY_SCOPE_LABELS[scope]} · gait agreement only · higher is better",
    )
    values_by_run = [
        [_value(index, str(run["run_id"]), scope, "family", family, GAIT_METRIC) for family in FAMILIES] for run in runs
    ]
    _draw_grouped_bars(
        axis,
        [FAMILY_LABELS[family] for family in FAMILIES],
        values_by_run,
        runs,
        value_format=".1f",
        y_upper=100.0,
    )
    axis.set_title(
        VELOCITY_SCOPE_LABELS[scope],
        fontsize=STYLE["panel_title_size"],
        fontweight=STYLE["matplotlib_title_weight"],
        pad=12,
    )
    axis.set_ylabel("Gait agreement [%]", fontsize=STYLE["axis_label_size"], labelpad=16)
    _add_shared_legend(figure, runs)
    figure.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.2)
    artifact = _save_figure(figure, output_dir, figure_id)
    plt.close(figure)
    return artifact


def _combine_figures(output_dir: Path, figure_id: str) -> dict[str, str]:
    """Concatenate child PNGs vertically and embed that exact stack in an SVG."""
    entry = FIGURE_REGISTRY[figure_id]
    children = tuple(str(child) for child in entry["children"])
    child_paths = [output_dir / f"{FIGURE_REGISTRY[child]['stem']}.png" for child in children]
    images: list[Image.Image] = []
    try:
        for path in child_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Cannot assemble {figure_id}: missing child PNG {path.name}.")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        combined = Image.new("RGB", (width, height), STYLE["background"])
        y = 0
        for image in images:
            combined.paste(image, ((width - image.width) // 2, y))
            y += image.height
        stem = str(entry["stem"])
        png_path = output_dir / f"{stem}.png"
        combined.save(png_path, format="PNG", compress_level=6, optimize=False)
        _write_embedded_svg(output_dir / f"{stem}.svg", child_paths, images, width, height)
    finally:
        for image in images:
            image.close()
    return {"figure_id": figure_id, "stem": stem, "png": png_path.name, "svg": f"{stem}.svg"}


def _write_embedded_svg(
    path: Path, child_paths: Sequence[Path], images: Sequence[Image.Image], width: int, height: int
) -> None:
    """Write a deterministic SVG containing the literal combined PNG panels."""
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    root = ET.Element(
        f"{{{namespace}}}svg",
        {
            "viewBox": f"0 0 {width} {height}",
            "width": str(width),
            "height": str(height),
            "role": "img",
        },
    )
    ET.SubElement(root, f"{{{namespace}}}rect", {"width": str(width), "height": str(height), "fill": "#FFFFFF"})
    y = 0
    for child_path, image in zip(child_paths, images, strict=True):
        encoded = base64.b64encode(child_path.read_bytes()).decode("ascii")
        ET.SubElement(
            root,
            f"{{{namespace}}}image",
            {
                "x": str((width - image.width) // 2),
                "y": str(y),
                "width": str(image.width),
                "height": str(image.height),
                "href": f"data:image/png;base64,{encoded}",
            },
        )
        y += image.height
    temporary_path = path.with_name(f".{path.name}.tmp")
    ET.ElementTree(root).write(temporary_path, encoding="utf-8", xml_declaration=True)
    os.replace(temporary_path, path)


def generate_figures(
    output_dir: str | Path,
    study: Mapping[str, Any],
    learning_points: Sequence[Mapping[str, Any]],
    velocity_summary: Sequence[Mapping[str, Any]],
    leg_summary: Sequence[Mapping[str, Any]],
    gait_summary: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Generate the complete 16-figure gait-closure registry.

    Args:
        output_dir: Directory that receives SVG and RGB PNG artifacts.
        study: Study metadata containing display information and ordered run descriptors.
        learning_points: Smoothed TensorBoard points for every configured run.
        velocity_summary: Long-form forward- and yaw-velocity RMSE summaries.
        leg_summary: Long-form four-metric front/hind imbalance summaries.
        gait_summary: Long-form gait-agreement summaries.

    Returns:
        One artifact descriptor for every entry in :data:`FIGURE_REGISTRY`.
    """
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runs = _resolve_runs(study)
    run_ids = {str(run["run_id"]) for run in runs}
    velocity_index = _summary_index(velocity_summary, {metric for metric, _label in PLOT_VELOCITY_METRICS}, run_ids)
    leg_index = _summary_index(leg_summary, {metric for metric, _label in PLOT_LEG_METRICS}, run_ids)
    gait_index = _summary_index(gait_summary, {GAIT_METRIC}, run_ids)
    display_name = str(study.get("display_name", "Unitree Go2"))

    artifacts: dict[str, dict[str, str]] = {}
    with plt.rc_context(_rc_params()):
        artifacts["fig01"] = _learning_figure(destination, study, runs, learning_points)
        artifacts["fig02"] = _velocity_overall_figure(destination, display_name, runs, velocity_index)
        velocity_family_y_uppers = {
            metric: _positive_upper(
                [
                    _value(velocity_index, str(run["run_id"]), scope, "family", family, metric)
                    for scope in VELOCITY_SCOPES
                    for family in FAMILIES
                    for run in runs
                ]
            )
            for metric, _label in PLOT_VELOCITY_METRICS
        }
        for figure_id, scope in (
            ("fig03_01", "all"),
            ("fig03_02", "negative"),
            ("fig03_03", "positive"),
        ):
            artifacts[figure_id] = _velocity_family_figure(
                destination,
                figure_id,
                display_name,
                scope,
                runs,
                velocity_index,
                velocity_family_y_uppers,
            )
        artifacts["fig03"] = _combine_figures(destination, "fig03")
        artifacts["fig04"] = _leg_overall_figure(destination, display_name, runs, leg_index)
        leg_family_y_upper = _positive_upper(
            [
                _value(leg_index, str(run["run_id"]), scope, "family", family, metric)
                for scope in VELOCITY_SCOPES
                for family in FAMILIES
                for metric, _label in PLOT_LEG_METRICS
                for run in runs
            ],
            minimum=10.0,
        )
        for figure_id, scope in (
            ("fig05_01", "all"),
            ("fig05_02", "negative"),
            ("fig05_03", "positive"),
        ):
            artifacts[figure_id] = _leg_family_figure(
                destination,
                figure_id,
                display_name,
                scope,
                runs,
                leg_index,
                leg_family_y_upper,
            )
        artifacts["fig05"] = _combine_figures(destination, "fig05")
        artifacts["fig06"] = _gait_overall_figure(destination, display_name, runs, gait_index)
        for figure_id, scope in (
            ("fig07_01", "all"),
            ("fig07_02", "negative"),
            ("fig07_03", "positive"),
        ):
            artifacts[figure_id] = _gait_family_figure(destination, figure_id, display_name, scope, runs, gait_index)
        artifacts["fig07"] = _combine_figures(destination, "fig07")
    return [artifacts[figure_id] for figure_id in FIGURE_REGISTRY]


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 digest for canonical compact JSON."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "record_sha256"}


def _validate_record_digest(record: Mapping[str, Any], label: str) -> None:
    actual = record.get("record_sha256")
    expected = canonical_sha256(_record_payload(record))
    if actual != expected:
        raise ValueError(f"Invalid record digest for {label}: expected {expected}, found {actual!r}.")


def _sha256_value(value: Any, field: str, context: str) -> str:
    """Validate and normalize one lowercase SHA-256 digest."""
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"Expected lowercase SHA-256 {field!r} for {context}, found {value!r}.")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required JSON input does not exist: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required CSV input does not exist: {path}") from error
    if not rows:
        raise ValueError(f"Required CSV input is empty: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any, field: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected numeric {field!r} for {context}, found {value!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {field!r} for {context}, found {value!r}.")
    return result


def _integer(value: Any, field: str, context: str) -> int:
    result = _finite_float(value, field, context)
    if not result.is_integer():
        raise ValueError(f"Expected integer {field!r} for {context}, found {value!r}.")
    return int(result)


def _boolean(value: Any, field: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected true/false {field!r} for {context}, found {value!r}.")


def _optional_boolean(value: Any, field: str, context: str) -> bool | None:
    """Parse a canonical tri-state switch used by time-reversal aliases."""
    if value is None:
        return None
    return _boolean(value, field, context)


def _nonnegative_integer(value: Any, field: str, context: str) -> int:
    """Parse one nonnegative schedule iteration count."""
    result = _integer(value, field, context)
    if result < 0:
        raise ValueError(f"Expected nonnegative {field!r} for {context}, found {result!r}.")
    return result


def _resolved_treatment_term(
    symmetry: Mapping[str, Any], term: Literal["policy", "value", "augmentation"], context: str
) -> dict[str, Any]:
    """Resolve one treatment term with the same canonical-over-legacy precedence as training."""
    if term == "policy":
        schedule_source = symmetry.get("tr_policy_schedule") or {}
        canonical_enabled = _optional_boolean(
            symmetry.get("use_tr_policy_consistency"), "use_tr_policy_consistency", context
        )
        legacy_enabled = _boolean(symmetry.get("use_mirror_loss", False), "use_mirror_loss", context)
        legacy_target = _finite_float(symmetry.get("mirror_loss_coeff", 0.0), "mirror_loss_coeff", context)
        mechanism = str(symmetry.get("tr_policy_output_space", "raw_action_mean"))
        configuration = {
            "output_space": mechanism,
            "actor_mean_bound_mode": symmetry.get("actor_mean_bound_mode", "legacy_global"),
            "actor_mean_feasible_margin_fraction": symmetry.get("actor_mean_feasible_margin_fraction", 0.0),
        }
    elif term == "value":
        schedule_source = symmetry.get("tr_value_schedule") or {}
        canonical_enabled = _optional_boolean(
            symmetry.get("use_tr_value_consistency"), "use_tr_value_consistency", context
        )
        legacy_target = _finite_float(symmetry.get("value_loss_coeff", 0.0), "value_loss_coeff", context)
        legacy_enabled = legacy_target > 0.0
        mechanism = "value_consistency"
        configuration = {}
    else:
        augmentation = symmetry.get("tr_augmentation") or {}
        if not isinstance(augmentation, Mapping):
            raise ValueError(f"Resolved tr_augmentation configuration is malformed for {context}.")
        schedule_source = augmentation.get("schedule") or {}
        canonical_enabled = _optional_boolean(augmentation.get("enabled", False), "tr_augmentation.enabled", context)
        legacy_enabled = False
        legacy_target = _finite_float(augmentation.get("coefficient", 0.0), "tr_augmentation.coefficient", context)
        mechanism = str(augmentation.get("mode", "dynamics_filtered_reverse_action_supervision"))
        configuration = {
            str(key): value for key, value in augmentation.items() if key not in {"enabled", "coefficient", "schedule"}
        }
    if not isinstance(schedule_source, Mapping):
        raise ValueError(f"Resolved {term} schedule is malformed for {context}.")

    if canonical_enabled is None:
        master_enabled = (
            legacy_enabled
            if "use_time_reversal_regularization" not in symmetry
            else _boolean(
                symmetry.get("use_time_reversal_regularization"),
                "use_time_reversal_regularization",
                context,
            )
        )
        consistency_enabled = master_enabled and legacy_enabled
    else:
        consistency_enabled = canonical_enabled
    schedule_enabled = _optional_boolean(schedule_source.get("enabled"), f"{term}.schedule.enabled", context)
    enabled = consistency_enabled if schedule_enabled is None else consistency_enabled and schedule_enabled

    target_value = schedule_source.get("target_coeff")
    target_coeff = (
        legacy_target if target_value is None else _finite_float(target_value, f"{term}.schedule.target_coeff", context)
    )
    if target_coeff < 0.0:
        raise ValueError(f"Expected nonnegative {term} target coefficient for {context}, found {target_coeff!r}.")

    def schedule_value(field: str, legacy_default: Any) -> Any:
        value = schedule_source.get(field)
        return legacy_default if value is None else value

    warmup = _nonnegative_integer(
        schedule_value("warmup_iterations", symmetry.get("warmup_iterations", 0)),
        f"{term}.schedule.warmup_iterations",
        context,
    )
    rampup = _nonnegative_integer(
        schedule_value("rampup_iterations", symmetry.get("rampup_iterations", 0)),
        f"{term}.schedule.rampup_iterations",
        context,
    )
    hold = _nonnegative_integer(schedule_source.get("hold_iterations", 0), f"{term}.schedule.hold_iterations", context)
    decay = _nonnegative_integer(
        schedule_source.get("decay_iterations", 0), f"{term}.schedule.decay_iterations", context
    )
    final_scale = _finite_float(schedule_source.get("final_scale", 1.0), f"{term}.schedule.final_scale", context)
    if not 0.0 <= final_scale <= 1.0:
        raise ValueError(f"Expected {term} final scale in [0, 1] for {context}, found {final_scale!r}.")
    ramp_shape = str(schedule_value("ramp_shape", symmetry.get("ramp_shape", "linear")))
    if ramp_shape not in {"linear", "half_cosine"}:
        raise ValueError(f"Unsupported {term} ramp shape for {context}: {ramp_shape!r}.")
    return {
        "enabled": enabled,
        "target_coeff": target_coeff,
        "mechanism": mechanism,
        "configuration": configuration,
        "schedule": {
            "warmup_iterations": warmup,
            "rampup_iterations": rampup,
            "hold_iterations": hold,
            "decay_iterations": decay,
            "final_scale": final_scale,
            "ramp_shape": ramp_shape,
        },
    }


def _resolved_treatment(symmetry: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Resolve the complete policy, value, augmentation, and validity treatment contract."""
    validity = symmetry.get("tr_validity") or {}
    if not isinstance(validity, Mapping):
        raise ValueError(f"Resolved tr_validity configuration is malformed for {context}.")
    return {
        "policy": _resolved_treatment_term(symmetry, "policy", context),
        "value": _resolved_treatment_term(symmetry, "value", context),
        "augmentation": _resolved_treatment_term(symmetry, "augmentation", context),
        "validity_mode": validity.get("mode"),
        "validity": dict(validity),
    }


def _treatment_is_enabled(treatment: Mapping[str, Any]) -> bool:
    """Return whether any resolved time-reversal mechanism is enabled."""
    return any(bool(treatment.get(term, {}).get("enabled")) for term in ("policy", "value", "augmentation"))


def _validate_baseline_treatment(run: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Require a run labeled as the no-TRS baseline to disable every treatment term."""
    if run["is_baseline"] and _treatment_is_enabled(metadata["treatment"]):
        raise ValueError(
            f"The run selected as the no-TRS baseline ({run['abbreviation']}) has an enabled policy, value, "
            "or trajectory-augmentation treatment."
        )


def _treatment_metadata(treatment: Mapping[str, Any]) -> dict[str, Any]:
    """Project a resolved treatment onto compatibility and canonical metadata fields."""
    policy = treatment["policy"]
    value = treatment["value"]
    augmentation = treatment["augmentation"]
    policy_schedule = policy["schedule"]
    return {
        "trs_enabled": _treatment_is_enabled(treatment),
        "mirror_coeff": policy["target_coeff"],
        "value_coeff": value["target_coeff"],
        "warmup_iterations": policy_schedule["warmup_iterations"],
        "rampup_iterations": policy_schedule["rampup_iterations"],
        "ramp_shape": policy_schedule["ramp_shape"],
        "validity_mode": treatment.get("validity_mode"),
        "treatment": treatment,
        "policy_enabled": policy["enabled"],
        "policy_target_coeff": policy["target_coeff"],
        "policy_warmup_iterations": policy_schedule["warmup_iterations"],
        "policy_rampup_iterations": policy_schedule["rampup_iterations"],
        "value_enabled": value["enabled"],
        "value_target_coeff": value["target_coeff"],
        "value_warmup_iterations": value["schedule"]["warmup_iterations"],
        "value_rampup_iterations": value["schedule"]["rampup_iterations"],
        "augmentation_enabled": augmentation["enabled"],
        "augmentation_target_coeff": augmentation["target_coeff"],
        "augmentation_warmup_iterations": augmentation["schedule"]["warmup_iterations"],
        "augmentation_rampup_iterations": augmentation["schedule"]["rampup_iterations"],
    }


def parse_run_spec(specification: str) -> tuple[str, str]:
    """Parse one ``ABBREVIATION=RUN_NAME`` command-line specification."""
    if "=" not in specification:
        raise ValueError(f"Invalid --run specification {specification!r}; use ABBREVIATION=RUN_NAME.")
    abbreviation, run_name = specification.split("=", maxsplit=1)
    abbreviation = abbreviation.strip()
    run_name = run_name.strip()
    if not abbreviation or not run_name:
        raise ValueError(f"Invalid --run specification {specification!r}; both sides of '=' are required.")
    return abbreviation, run_name


def default_run_roots(repo_root: Path) -> tuple[Path, Path]:
    """Return the primary and archived Go2 run roots in search order."""
    return (
        repo_root / "logs" / "rsl_rl" / "unitree_go2_symm_flat",
        repo_root / "logs" / "rsl_rl" / "good_runs" / "unitree_go2_symm_flat",
    )


def _run_identity(path: Path, evaluation_subdir: str) -> dict[str, str]:
    """Return hashes for every run-local input consumed by the comparison."""
    evaluation_root = path / evaluation_subdir
    metrics_root = evaluation_root / "metrics"
    study_path = evaluation_root / "study.json"
    study = _read_json(study_path)
    checkpoint_iteration = _integer(study.get("checkpoint", {}).get("iteration"), "checkpoint iteration", str(path))
    checkpoint_path = path / f"model_{checkpoint_iteration}.pt"
    event_paths = sorted(path.glob("events.out.tfevents.*"))
    if len(event_paths) != 1:
        raise ValueError(f"Duplicate-run identity requires exactly one TensorBoard event file in {path}.")
    consumed_paths = {
        "evaluation_study": study_path,
        "evaluation_progress": evaluation_root / "progress.json",
        "cell_metrics": metrics_root / "cell_metrics.csv",
        "overall_metrics": metrics_root / "overall_metrics.json",
        "evaluation_analysis_provenance": metrics_root / "analysis_provenance.json",
        "terminal_checkpoint": checkpoint_path,
        "tensorboard_event": event_paths[0],
    }
    initialization_path = path / "provenance" / "initialization.json"
    if initialization_path.is_file():
        consumed_paths["training_initialization"] = initialization_path
    else:
        consumed_paths.update(
            {
                "training_agent": path / "params" / "agent.yaml",
                "training_environment": path / "params" / "env.yaml",
                "iteration_zero_checkpoint": path / "model_0.pt",
                "training_source_diff": path / "git" / "symm_rl_isaaclab.diff",
            }
        )
    missing = [str(input_path) for input_path in consumed_paths.values() if not input_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Duplicate-run identity inputs are missing for {path}: {missing}")
    return {name: sha256_file(input_path) for name, input_path in consumed_paths.items()}


def _resolve_run_name(
    run_name: str,
    run_roots: Sequence[Path],
    repo_root: Path,
    evaluation_subdir: str = DEFAULT_EVALUATION_SUBDIR,
) -> Path:
    raw = Path(run_name).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(repo_root / raw)
        candidates.extend(root / raw for root in run_roots)
    matches = [candidate for candidate in dict.fromkeys(path.resolve() for path in candidates) if candidate.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        identities = [_run_identity(match, evaluation_subdir) for match in matches]
        if any(identity != identities[0] for identity in identities[1:]):
            rendered = ", ".join(str(match) for match in matches)
            raise ValueError(f"Run {run_name!r} is ambiguous across roots with different content: {rendered}")
        return matches[0]
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Run {run_name!r} was not found. Checked: {rendered}")


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _comparison_launcher_provenance(repo_root: Path) -> dict[str, dict[str, str]]:
    """Return paths and hashes for the canonical comparison launcher family."""
    script_dir = Path(__file__).resolve().parent
    records: dict[str, dict[str, str]] = {}
    for launcher, filename in COMPARISON_LAUNCHER_FILES.items():
        path = script_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical {launcher} comparison launcher: {path}")
        records[launcher] = {
            "path": _repo_relative(path, repo_root),
            "sha256": sha256_file(path),
        }
    return records


def resolve_run_inputs(
    run_specs: Sequence[str],
    baseline_abbreviation: str,
    run_roots: Sequence[Path],
    repo_root: Path,
    evaluation_subdir: str = DEFAULT_EVALUATION_SUBDIR,
) -> list[dict[str, Any]]:
    """Resolve, validate, order, and color an arbitrary run cohort."""
    if len(run_specs) < 2:
        raise ValueError("At least two --run ABBREVIATION=RUN_NAME specifications are required.")
    if len(run_specs) > len(V2_PALETTE):
        raise ValueError(
            f"The v2 palette supports at most {len(V2_PALETTE)} uniquely colored runs; found {len(run_specs)}."
        )
    parsed = [parse_run_spec(specification) for specification in run_specs]
    abbreviations = [abbreviation for abbreviation, _ in parsed]
    if len(abbreviations) != len(set(abbreviations)):
        duplicates = sorted(name for name, count in Counter(abbreviations).items() if count > 1)
        raise ValueError(f"Run abbreviations must be unique; duplicates: {duplicates}.")
    if baseline_abbreviation not in abbreviations:
        raise ValueError(f"Baseline abbreviation {baseline_abbreviation!r} is not present in --run entries.")

    resolved = [
        (abbreviation, run_name, _resolve_run_name(run_name, run_roots, repo_root, evaluation_subdir))
        for abbreviation, run_name in parsed
    ]
    path_keys = [os.path.normcase(str(path.resolve())) for _, _, path in resolved]
    if len(path_keys) != len(set(path_keys)):
        raise ValueError("The comparison cohort contains duplicate resolved run directories.")

    ordered = [item for item in resolved if item[0] == baseline_abbreviation]
    ordered.extend(item for item in resolved if item[0] != baseline_abbreviation)
    nonbaseline_colors = V2_PALETTE[1:]
    runs: list[dict[str, Any]] = []
    for index, (abbreviation, run_name, path) in enumerate(ordered):
        is_baseline = index == 0
        color = BASELINE_COLOR if is_baseline else nonbaseline_colors[(index - 1) % len(nonbaseline_colors)]
        runs.append(
            {
                "run_id": f"run_{index:02d}",
                "abbreviation": abbreviation,
                "label": abbreviation,
                "run_name": run_name,
                "folder": path.name,
                "run_path": _repo_relative(path, repo_root),
                "resolved_path": str(path.resolve()),
                "is_baseline": is_baseline,
                "color": color,
            }
        )
    return runs


def velocity_scope(velocity_mps: float) -> str:
    """Return the directional scope for one nonzero commanded velocity [m/s]."""
    velocity = float(velocity_mps)
    if not math.isfinite(velocity) or velocity == 0.0:
        raise ValueError(f"Velocity scopes require a finite nonzero velocity, found {velocity_mps!r}.")
    return "negative" if velocity < 0.0 else "positive"


def _scope_includes(scope: str, velocity_mps: float) -> bool:
    direction = velocity_scope(velocity_mps)
    if scope == "all":
        return True
    if scope not in VELOCITY_SCOPES:
        raise ValueError(f"Unknown velocity scope {scope!r}.")
    return direction == scope


def _protocol_signature(study: Mapping[str, Any]) -> dict[str, Any]:
    """Project an evaluation study onto every cross-run protocol field."""
    cells = [
        {
            key: cell.get(key)
            for key in (
                "id",
                "gait_index",
                "gait_name",
                "family",
                "phases",
                "velocity_mps",
                "seed",
                "step_dt",
                "settle_steps",
                "measure_steps",
                "total_steps",
                "relative_output_dir",
            )
        }
        for cell in study.get("cells", [])
    ]
    fields = (
        "schema_version",
        "method_version",
        "protocol",
        "robot",
        "task",
        "gait_library_version",
        "velocities_mps",
        "gaits",
        "step_dt",
        "settle_s",
        "measure_s",
        "settle_steps",
        "measure_steps",
        "total_steps",
        "episode_length_s",
        "evaluation_seed",
        "nominal_profile",
        "runtime_overrides",
        "render_cell_plots",
        "source_provenance",
        "evaluation_config",
        "contact_threshold_policy",
        "effort_limit_provenance",
        "aggregation",
    )
    return {key: study.get(key) for key in fields} | {"cells": cells}


def _validate_protocol_cells(study: Mapping[str, Any], label: str) -> None:
    cells = study.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Evaluation study has no cell plan for {label}.")
    if study.get("method_version") != EVALUATION_METHOD_VERSION:
        raise ValueError(
            f"Evaluation method mismatch for {label}: expected {EVALUATION_METHOD_VERSION!r}, "
            f"found {study.get('method_version')!r}."
        )
    identifiers = [str(cell.get("id", "")) for cell in cells]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Evaluation cell identifiers are missing or duplicated for {label}.")
    family_counts = Counter(str(cell.get("family", "")) for cell in cells)
    observed_family_counts = {family: family_counts[family] for family in FAMILIES}
    if len(cells) != 60 or observed_family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(
            f"Full-v3 family coverage for {label} must be 6/6/24/24 across {FAMILIES}; found {dict(family_counts)}."
        )
    scope_counts = Counter(
        velocity_scope(_finite_float(cell.get("velocity_mps"), "velocity_mps", label)) for cell in cells
    )
    observed_scopes = {
        "all": len(cells),
        "negative": scope_counts["negative"],
        "positive": scope_counts["positive"],
    }
    if observed_scopes != EXPECTED_SCOPE_COUNTS:
        raise ValueError(f"Full-v3 velocity coverage for {label} must be 60/30/30; found {observed_scopes}.")
    velocities = tuple(float(value) for value in study.get("velocities_mps", ()))
    expected_pairs = {(int(gait["index"]), velocity) for gait in study.get("gaits", []) for velocity in velocities}
    actual_pairs = {(int(cell["gait_index"]), float(cell["velocity_mps"])) for cell in cells}
    if len(study.get("gaits", [])) != 10 or actual_pairs != expected_pairs:
        raise ValueError(f"Evaluation gait/velocity Cartesian product is incomplete for {label}.")


def _validate_checkpoint_binding(run_path: Path, study: Mapping[str, Any], label: str) -> Path:
    checkpoint = study.get("checkpoint", {})
    iteration = _integer(checkpoint.get("iteration"), "checkpoint iteration", label)
    model_path = run_path / f"model_{iteration}.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint for {label} does not exist: {model_path}")
    actual_sha = sha256_file(model_path)
    if checkpoint.get("sha256") != actual_sha:
        raise ValueError(
            f"Evaluation checkpoint SHA-256 mismatch for {label}: expected {checkpoint.get('sha256')!r}, "
            f"found {actual_sha}."
        )
    return model_path


def _validate_analysis_provenance(
    evaluation_root: Path,
    study: Mapping[str, Any],
    overall: Mapping[str, Any],
    provenance: Mapping[str, Any],
    label: str,
) -> None:
    _validate_record_digest(provenance, f"{label} evaluation analysis provenance")
    study_path = evaluation_root / "study.json"
    if provenance.get("analysis_method_version") != study.get("method_version"):
        raise ValueError(f"Evaluation analysis method provenance mismatch for {label}.")
    if provenance.get("input_study_sha256") != sha256_file(study_path):
        raise ValueError(f"Evaluation analysis provenance input-study mismatch for {label}.")
    if provenance.get("input_study_identity_sha256") != study.get("study_identity_sha256"):
        raise ValueError(f"Evaluation analysis provenance study-identity mismatch for {label}.")
    embedded = overall.get("analysis_provenance", {})
    for key in ("analyzer_sha256", "metrics_sha256", "input_study_sha256", "record_sha256"):
        if embedded.get(key) != provenance.get(key):
            raise ValueError(f"Overall metrics provenance field {key!r} mismatches for {label}.")
    for key in ("analyzer_sha256", "metrics_sha256"):
        value = str(provenance.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"Evaluation provenance has invalid {key!r} for {label}.")


def _validate_evaluation_completeness(
    progress: Mapping[str, Any], overall: Mapping[str, Any], expected_cells: int, label: str
) -> None:
    expected_progress = {
        "status": "complete",
        "total_cells": expected_cells,
        "completed_cells": expected_cells,
    }
    mismatches = {
        key: (progress.get(key), expected)
        for key, expected in expected_progress.items()
        if progress.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Incomplete full-v3 evaluation progress for {label}: {mismatches}.")
    progress_counts = {
        key: _integer(progress.get(key), key, f"{label} evaluation progress")
        for key in ("completed_cells", "successful_cells", "skipped_cells", "terminated_cells")
    }
    invalid_counts = {key: value for key, value in progress_counts.items() if value < 0}
    if progress_counts["successful_cells"] + progress_counts["terminated_cells"] != expected_cells:
        invalid_counts["successful_cells + terminated_cells"] = (
            progress_counts["successful_cells"] + progress_counts["terminated_cells"],
            expected_cells,
        )
    if progress_counts["skipped_cells"] > expected_cells:
        invalid_counts["skipped_cells"] = (progress_counts["skipped_cells"], f"<= {expected_cells}")
    if invalid_counts:
        raise ValueError(f"Inconsistent full-v3 evaluation progress for {label}: {invalid_counts}.")
    coverage = overall.get("coverage", {})
    required_counts = {
        "expected_cells": expected_cells,
        "valid_cells": expected_cells,
        "velocity_valid_cells": expected_cells,
        "heading_valid_cells": expected_cells,
        "gait_valid_cells": expected_cells,
        "raw_load_valid_cells": expected_cells,
        "grf_load_valid_cells": expected_cells,
        "normalized_load_valid_cells": expected_cells,
    }
    coverage_mismatches = {
        key: (coverage.get(key), expected) for key, expected in required_counts.items() if coverage.get(key) != expected
    }
    if coverage.get("complete") is not True or coverage_mismatches:
        raise ValueError(f"Incomplete full-v3 metric coverage for {label}: {coverage_mismatches}.")
    for metadata in LEG_METRICS.values():
        source_metric = str(metadata["source_metric"])
        metric = overall.get("metrics", {}).get(source_metric, {})
        if metric.get("complete") is not True or metric.get("valid_cells") != expected_cells:
            raise ValueError(f"Incomplete {source_metric} metric domain for {label}.")


def _validated_cell_outcome_status(cell: Mapping[str, Any], context: str) -> str:
    """Validate an analyzed cell outcome that has complete metric domains."""
    status = str(cell.get("status", ""))
    if status not in {"valid", "terminated"}:
        raise ValueError(f"Full-v3 cell has no metric-complete outcome: {context} ({status!r}).")
    terminated = _boolean(cell.get("terminated"), "terminated", context)
    if terminated != (status == "terminated"):
        raise ValueError(f"Full-v3 cell outcome fields disagree for {context}.")
    return status


def _resolved_agent(initialization: Mapping[str, Any], label: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    configs = initialization.get("resolved_configs", {})
    agent = configs.get("agent")
    environment = configs.get("environment")
    if not isinstance(agent, Mapping) or not isinstance(environment, Mapping):
        raise ValueError(f"Initialization provenance lacks resolved agent/environment configs for {label}.")
    return agent, environment


def _parse_yaml_scalar(value: str) -> Any:
    """Parse one scalar from an archived resolved-config YAML line."""
    normalized = value.strip()
    lower = normalized.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
                return normalized[1:-1]
            return normalized


def _read_yaml_scalars(path: Path, key: str) -> list[Any]:
    """Read every scalar named ``key`` from an archived resolved-config YAML file."""
    values: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            values.append(_parse_yaml_scalar(stripped.partition(":")[2]))
    return values


def _read_unique_yaml_scalar(path: Path, key: str) -> Any:
    """Read a scalar that occurs exactly once in an archived resolved config."""
    values = _read_yaml_scalars(path, key)
    if len(values) != 1:
        raise ValueError(f"Expected one scalar {key!r} in {path}, found {len(values)}.")
    return values[0]


def _read_uniform_yaml_scalar(path: Path, key: str) -> Any:
    """Read a repeated scalar whose archived values must all agree."""
    values = _read_yaml_scalars(path, key)
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"Expected one or more identical {key!r} scalars in {path}, found {values}.")
    return values[0]


def _registered_legacy_run(run_path: Path, repo_root: Path) -> dict[str, Any]:
    """Return the unique verified registry entry for one legacy run."""
    registry_dir = repo_root / "scripts" / "symm_locomotion" / "study_registry"
    matches: list[tuple[Path, dict[str, Any], Mapping[str, Any]]] = []
    for registry_path in sorted(registry_dir.glob("*.json")):
        manifest = study_registry_module.load_manifest(registry_path)
        for entry in manifest["runs"]:
            candidate = (repo_root / str(entry["path"])).resolve()
            if candidate == run_path.resolve():
                matches.append((registry_path, manifest, entry))
    if len(matches) != 1:
        raise ValueError(
            f"Legacy run snapshots must have one unique study-registry entry; found {len(matches)} for {run_path}."
        )
    registry_path, manifest, entry = matches[0]
    if entry.get("classification") != "eligible_main":
        raise ValueError(
            f"Legacy run {entry.get('run_id')!r} is not an eligible-main registry entry: "
            f"{entry.get('classification')!r}."
        )
    selected_manifest = {**manifest, "runs": [entry]}
    errors = study_registry_module.verify_manifest_artifacts(selected_manifest, repo_root)
    if errors:
        raise ValueError(
            f"Legacy registry artifacts failed verification for {registry_path}:\n- " + "\n- ".join(errors)
        )
    facts = study_registry_module.materialize_facts(manifest, entry)
    return {
        "path": str(registry_path),
        "sha256": sha256_file(registry_path),
        "cohort_id": manifest["cohort_id"],
        "run_id": entry["run_id"],
        "classification": entry["classification"],
        "facts": facts,
    }


def _load_legacy_initialization_checksum(
    record_path: Path,
    registry: Mapping[str, Any],
    label: str,
    comparison_run_id: str | None,
) -> dict[str, Any]:
    """Validate a checksum-only initialization record for a latest-only archive."""
    record = _read_json(record_path)
    _validate_record_digest(record, f"{label} legacy initialization checksum")
    expected_fields = {
        "schema_version",
        "record_type",
        "availability",
        "bytes_published",
        "checkpoint",
        "comparison_run_id",
        "registry",
        "note",
        "record_sha256",
    }
    if set(record) != expected_fields:
        raise ValueError(
            f"Legacy initialization checksum for {label} must use the exact schema; found fields {sorted(record)}."
        )
    expected_header = {
        "record_type": LEGACY_INITIALIZATION_RECORD_TYPE,
        "availability": "checksum_only",
    }
    mismatches = {key: record.get(key) for key, expected in expected_header.items() if record.get(key) != expected}
    if type(record.get("schema_version")) is not int or record.get("schema_version") != 1:
        mismatches["schema_version"] = record.get("schema_version")
    if record.get("bytes_published") is not False:
        mismatches["bytes_published"] = record.get("bytes_published")
    if mismatches:
        raise ValueError(f"Legacy initialization checksum header is invalid for {label}: {mismatches}.")

    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"filename", "iteration", "sha256"}:
        raise ValueError(f"Legacy initialization checkpoint descriptor is malformed for {label}.")
    checkpoint_iteration = checkpoint.get("iteration")
    if checkpoint.get("filename") != "model_0.pt" or type(checkpoint_iteration) is not int or checkpoint_iteration != 0:
        raise ValueError(f"Legacy initialization checksum must describe model_0.pt at iteration 0 for {label}.")
    checkpoint_sha256 = _sha256_value(checkpoint.get("sha256"), "checkpoint.sha256", label)

    registry_record = record.get("registry")
    if not isinstance(registry_record, Mapping) or set(registry_record) != {"cohort_id", "run_id"}:
        raise ValueError(f"Legacy initialization registry descriptor is malformed for {label}.")
    expected_registry = {key: registry[key] for key in ("cohort_id", "run_id")}
    if dict(registry_record) != expected_registry:
        raise ValueError(
            f"Legacy initialization checksum is bound to the wrong registry entry for {label}: "
            f"{dict(registry_record)!r} != {expected_registry!r}."
        )
    saved_comparison_run_id = str(record.get("comparison_run_id", "")).strip()
    if not saved_comparison_run_id:
        raise ValueError(f"Legacy initialization checksum has no comparison_run_id for {label}.")
    if comparison_run_id is not None and saved_comparison_run_id != comparison_run_id:
        raise ValueError(
            f"Legacy initialization checksum comparison_run_id differs for {label}: "
            f"{saved_comparison_run_id!r} != {comparison_run_id!r}."
        )
    if not isinstance(record.get("note"), str) or not str(record["note"]).strip():
        raise ValueError(f"Legacy initialization checksum has no omission note for {label}.")
    return {**record, "checkpoint_sha256": checkpoint_sha256}


def _legacy_training_metadata(
    run_path: Path,
    label: str,
    repo_root: Path,
    comparison_run_id: str | None = None,
    prefer_checksum_only: bool = False,
) -> dict[str, Any]:
    """Recover training metadata from an explicitly weaker legacy run snapshot.

    This fallback deliberately does not synthesize pre-rollout initialization
    provenance. It records the archived config and Git snapshot. When the
    latest-only publication policy omits ``model_0.pt``, it accepts only a
    registry-bound checksum record and explicitly marks the bytes unverified.
    """
    agent_path = run_path / "params" / "agent.yaml"
    environment_path = run_path / "params" / "env.yaml"
    diff_path = run_path / "git" / "symm_rl_isaaclab.diff"
    iteration_zero_checkpoint_path = run_path / "model_0.pt"
    required_paths = (agent_path, environment_path, diff_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Training metadata for {label} has neither initialization provenance nor a complete registered "
            f"legacy configuration snapshot: {missing}."
        )
    registry = _registered_legacy_run(run_path, repo_root)
    code_hashes = registry["facts"].get("code_hashes", {})
    use_mirror_loss = _boolean(_read_unique_yaml_scalar(agent_path, "use_mirror_loss"), "use_mirror_loss", label)
    mirror_coeff = _finite_float(_read_unique_yaml_scalar(agent_path, "mirror_loss_coeff"), "mirror_loss_coeff", label)
    value_coeff = _finite_float(_read_unique_yaml_scalar(agent_path, "value_loss_coeff"), "value_loss_coeff", label)
    master_values = _read_yaml_scalars(agent_path, "use_time_reversal_regularization")
    if len(master_values) > 1:
        raise ValueError(
            f"Expected at most one scalar 'use_time_reversal_regularization' in {agent_path}, "
            f"found {len(master_values)}."
        )
    master_enabled = (
        use_mirror_loss or value_coeff > 0.0
        if not master_values
        else _boolean(master_values[0], "use_time_reversal_regularization", label)
    )
    legacy_symmetry = {
        "use_time_reversal_regularization": master_enabled,
        "use_mirror_loss": use_mirror_loss,
        "mirror_loss_coeff": mirror_coeff,
        "value_loss_coeff": value_coeff,
        "warmup_iterations": _read_unique_yaml_scalar(agent_path, "warmup_iterations"),
        "rampup_iterations": _read_unique_yaml_scalar(agent_path, "rampup_iterations"),
        "ramp_shape": _read_unique_yaml_scalar(agent_path, "ramp_shape"),
    }
    treatment = _resolved_treatment(legacy_symmetry, label)
    seed = _integer(_read_unique_yaml_scalar(agent_path, "seed"), "seed", label)
    num_envs = _integer(_read_uniform_yaml_scalar(environment_path, "num_envs"), "num_envs", label)
    num_steps_per_env = _integer(_read_unique_yaml_scalar(agent_path, "num_steps_per_env"), "num_steps_per_env", label)
    max_iterations = _integer(_read_unique_yaml_scalar(agent_path, "max_iterations"), "max_iterations", label)
    checksum_record_path = run_path / "provenance" / "legacy_initialization.json"
    checksum_record = None
    if checksum_record_path.is_file():
        checksum_record = _load_legacy_initialization_checksum(checksum_record_path, registry, label, comparison_run_id)
    checkpoint_bytes_available = iteration_zero_checkpoint_path.is_file()
    if checkpoint_bytes_available:
        local_checkpoint_sha256 = sha256_file(iteration_zero_checkpoint_path)
        if checksum_record is not None and checksum_record["checkpoint_sha256"] != local_checkpoint_sha256:
            raise ValueError(
                f"Published model_0.pt SHA-256 differs from its checksum record for {label}; refusing to use "
                "the checksum-only fallback."
            )
    if checkpoint_bytes_available and not prefer_checksum_only:
        iteration_zero_checkpoint_sha256 = local_checkpoint_sha256
        training_metadata_source = LEGACY_CONFIG_METADATA_SOURCE
        checkpoint_availability = "bytes_reverified"
        checkpoint_bytes_reverified = True
        metadata_paths = [
            agent_path,
            environment_path,
            iteration_zero_checkpoint_path,
            diff_path,
            Path(registry["path"]),
        ]
        if checksum_record is not None:
            metadata_paths.append(checksum_record_path)
    else:
        if checksum_record is None:
            raise FileNotFoundError(
                f"Legacy training metadata for {label} has no model_0.pt bytes and no validated "
                f"checksum-only record: {checksum_record_path}."
            )
        iteration_zero_checkpoint_sha256 = checksum_record["checkpoint_sha256"]
        training_metadata_source = LEGACY_CHECKSUM_ONLY_METADATA_SOURCE
        checkpoint_availability = "checksum_only"
        checkpoint_bytes_reverified = False
        metadata_paths = [agent_path, environment_path, diff_path, Path(registry["path"]), checksum_record_path]

    metadata = {
        "seed": seed,
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        **_treatment_metadata(treatment),
        "repo_commit": code_hashes.get("archived_git_commit"),
        "dirty_tree_diff_sha256": code_hashes.get("source_snapshot_sha256"),
        "iteration_zero_checkpoint_sha256": iteration_zero_checkpoint_sha256,
        "iteration_zero_checkpoint_availability": checkpoint_availability,
        "iteration_zero_checkpoint_bytes_reverified": checkpoint_bytes_reverified,
        "training_metadata_source": training_metadata_source,
        "training_metadata_files": [{"path": str(path), "sha256": sha256_file(path)} for path in metadata_paths],
        "legacy_registry": {key: registry[key] for key in ("path", "sha256", "cohort_id", "run_id", "classification")},
    }
    if checksum_record is not None:
        metadata["legacy_initialization_checksum_record"] = {
            "path": str(checksum_record_path),
            "sha256": sha256_file(checksum_record_path),
            "record_sha256": checksum_record["record_sha256"],
        }
    return metadata


def _load_training_metadata(run: Mapping[str, Any], repo_root: Path, *, manifest_bound: bool = False) -> dict[str, Any]:
    run_path = Path(str(run["resolved_path"]))
    initialization_path = run_path / "provenance" / "initialization.json"
    label = str(run["abbreviation"])
    expected_source = run.get("expected_training_metadata_source") if manifest_bound else None
    if initialization_path.is_file():
        initialization = _read_json(initialization_path)
        _validate_record_digest(initialization, f"{label} initialization")
        agent, environment = _resolved_agent(initialization, label)
        algorithm = agent.get("algorithm", {})
        symmetry = algorithm.get("symmetry_cfg", {}) if isinstance(algorithm, Mapping) else {}
        scene = environment.get("scene", {})
        if not isinstance(symmetry, Mapping) or not isinstance(scene, Mapping):
            raise ValueError(f"Resolved symmetry/environment configuration is malformed for {label}.")
        treatment = _resolved_treatment(symmetry, label)
        metadata = {
            "seed": _integer(agent.get("seed"), "seed", label),
            "num_envs": _integer(scene.get("num_envs"), "num_envs", label),
            "num_steps_per_env": _integer(agent.get("num_steps_per_env"), "num_steps_per_env", label),
            "max_iterations": _integer(agent.get("max_iterations"), "max_iterations", label),
            **_treatment_metadata(treatment),
            "repo_commit": initialization.get("repo_commit"),
            "dirty_tree_diff_sha256": initialization.get("dirty_tree_diff_sha256"),
            "training_metadata_source": INITIALIZATION_METADATA_SOURCE,
            "training_metadata_files": [{"path": str(initialization_path), "sha256": sha256_file(initialization_path)}],
            "initialization_path": str(initialization_path),
            "initialization_sha256": sha256_file(initialization_path),
        }
    else:
        comparison_run_id = str(run["run_id"]) if manifest_bound else None
        metadata = _legacy_training_metadata(
            run_path,
            label,
            repo_root,
            comparison_run_id,
            prefer_checksum_only=expected_source == LEGACY_CHECKSUM_ONLY_METADATA_SOURCE,
        )
    if expected_source is not None and metadata["training_metadata_source"] != expected_source:
        raise ValueError(
            f"Training metadata source differs from the saved comparison manifest for {label}: "
            f"{metadata['training_metadata_source']!r} != {expected_source!r}."
        )
    _validate_baseline_treatment(run, metadata)
    return metadata


def _evaluation_paths(run_path: Path, evaluation_subdir: str) -> dict[str, Path]:
    root = run_path / evaluation_subdir
    metrics = root / "metrics"
    return {
        "root": root,
        "study": root / "study.json",
        "progress": root / "progress.json",
        "cell_metrics": metrics / "cell_metrics.csv",
        "overall_metrics": metrics / "overall_metrics.json",
        "analysis_provenance": metrics / "analysis_provenance.json",
    }


def _validate_cell_plan_row(source: Mapping[str, Any], planned: Mapping[str, Any], context: str) -> None:
    """Require the analyzed row metadata to match its planned evaluation cell."""
    integer_fields = ("gait_index", "seed")
    text_fields = ("gait_name", "family", "relative_output_dir")
    for field in integer_fields:
        actual = _integer(source.get(field), field, context)
        expected = _integer(planned.get(field), field, f"{context} study plan")
        if actual != expected:
            raise ValueError(f"Cell-metric field {field!r} differs from the study plan for {context}.")
    for field in text_fields:
        if str(source.get(field, "")) != str(planned.get(field, "")):
            raise ValueError(f"Cell-metric field {field!r} differs from the study plan for {context}.")
    actual_velocity = _finite_float(source.get("velocity_mps"), "velocity_mps", context)
    expected_velocity = _finite_float(planned.get("velocity_mps"), "velocity_mps", f"{context} study plan")
    if actual_velocity != expected_velocity:
        raise ValueError(f"Cell-metric field 'velocity_mps' differs from the study plan for {context}.")


def _load_evaluation(run: Mapping[str, Any], evaluation_subdir: str) -> dict[str, Any]:
    label = str(run["abbreviation"])
    run_path = Path(str(run["resolved_path"]))
    paths = _evaluation_paths(run_path, evaluation_subdir)
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing full-v3 inputs for {label}: {missing}")
    study = _read_json(paths["study"])
    progress = _read_json(paths["progress"])
    overall = _read_json(paths["overall_metrics"])
    provenance = _read_json(paths["analysis_provenance"])
    _validate_protocol_cells(study, label)
    checkpoint_path = _validate_checkpoint_binding(run_path, study, label)
    _validate_evaluation_completeness(progress, overall, len(study["cells"]), label)
    _validate_analysis_provenance(paths["root"], study, overall, provenance, label)
    cells = _read_csv(paths["cell_metrics"])
    if len(cells) != len(study["cells"]):
        raise ValueError(f"Cell-metric row count mismatch for {label}: {len(cells)} != {len(study['cells'])}.")
    planned_by_id = {str(cell["id"]): cell for cell in study["cells"]}
    expected_ids = set(planned_by_id)
    actual_ids = {str(cell.get("cell_id", "")) for cell in cells}
    if actual_ids != expected_ids or len(actual_ids) != len(cells):
        raise ValueError(f"Cell-metric identifiers do not exactly match the study plan for {label}.")
    effort_provenance = study.get("effort_limit_provenance", {})
    if not isinstance(effort_provenance, Mapping):
        raise ValueError(f"Study effort-limit provenance is malformed for {label}.")
    effort_source = effort_provenance.get("source", {})
    if effort_provenance.get("fallback_allowed") is not False or not isinstance(effort_source, Mapping):
        raise ValueError(f"Study effort-limit provenance permits fallback or is malformed for {label}.")
    expected_effort_sha256 = str(effort_source.get("sha256", ""))
    expected_effort_files = effort_source.get("files", {})
    if (
        len(expected_effort_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_effort_sha256.lower())
        or not isinstance(expected_effort_files, Mapping)
        or not expected_effort_files
    ):
        raise ValueError(f"Study effort-limit source provenance is incomplete for {label}.")
    sentinel_threshold_nm = _finite_float(
        effort_provenance.get("sentinel_rejection_threshold_nm"),
        "sentinel_rejection_threshold_nm",
        label,
    )
    cell_status_counts: Counter[str] = Counter()
    for cell in cells:
        context = f"{label}/{cell.get('cell_id')}"
        _validate_cell_plan_row(cell, planned_by_id[str(cell["cell_id"])], context)
        cell_status_counts[_validated_cell_outcome_status(cell, context)] += 1
        for field in (
            "velocity_metric_valid",
            "heading_metric_valid",
            "gait_metric_valid",
            "raw_load_metric_valid",
            "grf_load_metric_valid",
            "normalized_load_metric_valid",
        ):
            if not _boolean(cell.get(field), field, context):
                raise ValueError(f"Full-v3 domain flag {field!r} failed for {context}.")
        if _boolean(cell.get("effort_limit_fallback"), "effort_limit_fallback", context):
            raise ValueError(f"Fallback effort limits are not allowed for normalized torque²: {context}.")
        if str(cell.get("effort_limit_provenance_sha256", "")) != expected_effort_sha256:
            raise ValueError(f"Effort-limit provenance hash differs from the study plan for {context}.")
        try:
            effort_files = json.loads(str(cell.get("effort_limit_provenance_files", "")))
        except json.JSONDecodeError as error:
            raise ValueError(f"Effort-limit provenance files are malformed for {context}.") from error
        if effort_files != expected_effort_files:
            raise ValueError(f"Effort-limit provenance files differ from the study plan for {context}.")
        effort_min_nm = _finite_float(cell.get("effort_limit_min_nm"), "effort_limit_min_nm", context)
        effort_max_nm = _finite_float(cell.get("effort_limit_max_nm"), "effort_limit_max_nm", context)
        if not 0.0 < effort_min_nm <= effort_max_nm < sentinel_threshold_nm:
            raise ValueError(f"Effort-limit bounds are not finite physical values for {context}.")
    expected_status_counts = {
        "valid": _integer(progress.get("successful_cells"), "successful_cells", f"{label} evaluation progress"),
        "terminated": _integer(progress.get("terminated_cells"), "terminated_cells", f"{label} evaluation progress"),
    }
    observed_status_counts = {status: cell_status_counts[status] for status in ("valid", "terminated")}
    if observed_status_counts != expected_status_counts:
        raise ValueError(
            f"Full-v3 analyzed cell outcomes disagree with evaluation progress for {label}: "
            f"{observed_status_counts} != {expected_status_counts}."
        )
    return {
        "run": run,
        "study": study,
        "progress": progress,
        "overall": overall,
        "provenance": provenance,
        "cells": cells,
        "checkpoint_path": checkpoint_path,
        "input_paths": paths,
        "input_hashes": {key: sha256_file(path) for key, path in paths.items() if key != "root"},
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _read_training_scalars(event_path: Path) -> list[tuple[int, float, float]]:
    """Read the reward trace with the repository's dependency-free event parser."""
    scalars = tensorboard_scalars_module.read_scalars(event_path, selected_tags={"Train/mean_reward"})
    points = scalars.get("Train/mean_reward")
    if not points:
        raise ValueError(f"TensorBoard event contains no Train/mean_reward trace: {event_path}")
    return [(int(step), float(wall_time), float(value)) for step, wall_time, value in points]


def _validate_reward_trace(points: Sequence[tuple[int, float, float]], max_iterations: int, label: str) -> None:
    steps = [point[0] for point in points]
    if not steps:
        raise ValueError(f"TensorBoard reward trace is empty for {label}.")
    start = steps[0]
    if start not in {0, 1} or steps != list(range(start, max_iterations)):
        raise ValueError(
            f"Incomplete Train/mean_reward for {label}: expected contiguous steps "
            f"{start}..{max_iterations - 1}, found {len(steps)} points."
        )
    if any(not math.isfinite(wall_time) or not math.isfinite(value) for _, wall_time, value in points):
        raise ValueError(f"Train/mean_reward contains a non-finite scalar for {label}.")
    if any(right[1] < left[1] for left, right in zip(points, points[1:], strict=False)):
        raise ValueError(f"Train/mean_reward wall times are not monotonic for {label}.")


def _rolling_mean(points: Sequence[tuple[int, float, float]], window: int) -> list[tuple[int, float, float]]:
    if window <= 0:
        raise ValueError("The smoothing window must be positive.")
    result: list[tuple[int, float, float]] = []
    values: list[float] = []
    running = 0.0
    for step, wall_time, value in points:
        values.append(value)
        running += value
        if len(values) > window:
            running -= values[-window - 1]
        result.append((step, wall_time, running / min(len(values), window)))
    return result


def _trapezoid_auc(points: Sequence[tuple[int, float, float]]) -> float:
    if len(points) < 2 or points[-1][0] == points[0][0]:
        raise ValueError("At least two distinct reward points are required for AUC.")
    area = sum(
        0.5 * (left[2] + right[2]) * (right[0] - left[0]) for left, right in zip(points, points[1:], strict=False)
    )
    return area / (points[-1][0] - points[0][0])


def _build_training_data(
    runs: Sequence[Mapping[str, Any]],
    training_metadata: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    smoothing_window_iterations: int,
    sample_stride_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    if smoothing_window_iterations <= 0 or sample_stride_iterations <= 0:
        raise ValueError("Training smoothing and sample stride must both be positive.")
    summaries: list[dict[str, Any]] = []
    learning_points: list[dict[str, Any]] = []
    event_hashes: dict[str, str] = {}
    for run in runs:
        metadata = training_metadata[str(run["run_id"])]
        run_path = Path(str(run["resolved_path"]))
        event_paths = sorted(run_path.glob("events.out.tfevents.*"))
        if len(event_paths) != 1:
            raise ValueError(
                f"Expected exactly one TensorBoard event file for {run['abbreviation']}, found {len(event_paths)}."
            )
        event_path = event_paths[0]
        points = _read_training_scalars(event_path)
        max_iterations = int(metadata["max_iterations"])
        _validate_reward_trace(points, max_iterations, str(run["abbreviation"]))
        if len(points) < smoothing_window_iterations:
            raise ValueError(
                f"Reward trace for {run['abbreviation']} is shorter than the smoothing window "
                f"({len(points)} < {smoothing_window_iterations})."
            )
        transitions_per_iteration = int(metadata["num_envs"]) * int(metadata["num_steps_per_env"])
        endpoint = [point[2] for point in points[-min(1000, len(points)) :]]
        first_ten_thousand = [point for point in points if point[0] <= 10_000]
        summary = {
            **{key: run[key] for key in ("run_id", "abbreviation", "color", "is_baseline")},
            "run": run["run_path"],
            "seed": metadata["seed"],
            "num_envs": metadata["num_envs"],
            "num_steps_per_env": metadata["num_steps_per_env"],
            "iterations": points[-1][0] + 1,
            "transitions_per_iteration": transitions_per_iteration,
            "environment_transitions": (points[-1][0] + 1) * transitions_per_iteration,
            "wall_time_hours": (points[-1][1] - points[0][1]) / 3600.0,
            "reward_auc_first_10000": _trapezoid_auc(first_ten_thousand),
            "reward_auc_full": _trapezoid_auc(points),
            "reward_last_1000_mean": statistics.fmean(endpoint),
            "reward_last_1000_std": statistics.pstdev(endpoint),
            "trs_enabled": metadata["trs_enabled"],
            "mirror_coeff": metadata["mirror_coeff"],
            "value_coeff": metadata["value_coeff"],
            "warmup_iterations": metadata["warmup_iterations"],
            "rampup_iterations": metadata["rampup_iterations"],
            "policy_enabled": metadata["policy_enabled"],
            "policy_target_coeff": metadata["policy_target_coeff"],
            "policy_warmup_iterations": metadata["policy_warmup_iterations"],
            "policy_rampup_iterations": metadata["policy_rampup_iterations"],
            "value_enabled": metadata["value_enabled"],
            "value_target_coeff": metadata["value_target_coeff"],
            "value_warmup_iterations": metadata["value_warmup_iterations"],
            "value_rampup_iterations": metadata["value_rampup_iterations"],
            "augmentation_enabled": metadata["augmentation_enabled"],
            "augmentation_target_coeff": metadata["augmentation_target_coeff"],
            "augmentation_warmup_iterations": metadata["augmentation_warmup_iterations"],
            "augmentation_rampup_iterations": metadata["augmentation_rampup_iterations"],
        }
        summaries.append(summary)
        smoothed = _rolling_mean(points, smoothing_window_iterations)[smoothing_window_iterations - 1 :]
        sampled = smoothed[::sample_stride_iterations]
        if sampled[-1] != smoothed[-1]:
            sampled.append(smoothed[-1])
        start_wall_time = points[0][1]
        learning_points.extend(
            {
                **{key: run[key] for key in ("run_id", "abbreviation", "color", "is_baseline")},
                "run": run["run_path"],
                "iteration": step,
                "transitions_millions": step * transitions_per_iteration / 1_000_000.0,
                "elapsed_hours": (wall_time - start_wall_time) / 3600.0,
                "smoothed_reward": value,
            }
            for step, wall_time, value in sampled
        )
        event_hashes[str(run["run_id"])] = sha256_file(event_path)

    baseline = next(row for row in summaries if row["is_baseline"])
    for row in summaries:
        for field in (
            "wall_time_hours",
            "reward_auc_first_10000",
            "reward_auc_full",
            "reward_last_1000_mean",
        ):
            row[f"{field}_delta_from_baseline"] = float(row[field]) - float(baseline[field])
    return summaries, learning_points, event_hashes


def _build_evaluation_cells(evaluations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        run = evaluation["run"]
        for source in evaluation["cells"]:
            context = f"{run['abbreviation']}/{source.get('cell_id')}"
            velocity_mps = _finite_float(source.get("velocity_mps"), "velocity_mps", context)
            row: dict[str, Any] = {
                **{key: run[key] for key in ("run_id", "abbreviation", "color", "is_baseline")},
                "run": run["run_path"],
                "cell_id": str(source["cell_id"]),
                "status": str(source["status"]),
                "terminated": _boolean(source.get("terminated"), "terminated", context),
                "gait_index": _integer(source.get("gait_index"), "gait_index", context),
                "gait_name": str(source.get("gait_name", "")),
                "family": str(source.get("family", "")),
                "seed": _integer(source.get("seed"), "seed", context),
                "velocity_mps": velocity_mps,
                "velocity_scope": velocity_scope(velocity_mps),
                "velocity_vx_rmse_mps": _finite_float(
                    source.get("velocity_vx_rmse_mps"), "velocity_vx_rmse_mps", context
                ),
                "velocity_vx_relative_rmse_percent": 100.0
                * _finite_float(source.get("velocity_vx_relative_rmse"), "velocity_vx_relative_rmse", context),
                "velocity_yaw_rmse_radps": _finite_float(
                    source.get("velocity_yaw_rmse_radps"), "velocity_yaw_rmse_radps", context
                ),
                "gait_agreement_boundary_excluded_percent": 100.0
                * _finite_float(
                    source.get("gait_agreement_boundary_excluded"),
                    "gait_agreement_boundary_excluded",
                    context,
                ),
            }
            for field in (
                "planar_tracking_success",
                "vx_relative_tracking_success",
                "yaw_tracking_success",
                "tracking_success",
            ):
                row[field] = _boolean(source.get(field), field, context)
            for metric, metadata in LEG_METRICS.items():
                source_metric = str(metadata["source_metric"])
                for suffix in (
                    "front_integral",
                    "hind_integral",
                    "signed_imbalance_percent",
                    "abs_imbalance_percent",
                    "total_per_s",
                    "total_per_directed_m",
                ):
                    source_field = f"{source_metric}_{suffix}"
                    value = _finite_float(source.get(source_field), source_field, context)
                    row[source_field] = value
                    row[f"{metric}_{suffix}"] = value
            rows.append(row)

    baseline = {str(row["cell_id"]): row for row in rows if row["is_baseline"]}
    if len(baseline) != 60:
        raise ValueError(f"Combined evaluation table requires 60 unique baseline cells, found {len(baseline)}.")
    delta_fields = [*VELOCITY_METRICS, *GAIT_METRICS]
    delta_fields.extend(f"{metric}_abs_imbalance_percent" for metric in LEG_METRICS)
    for row in rows:
        reference = baseline.get(str(row["cell_id"]))
        if reference is None:
            raise ValueError(f"No baseline counterpart exists for cell {row['cell_id']!r}.")
        for field in delta_fields:
            row[f"{field}_delta_from_baseline"] = float(row[field]) - float(reference[field])
    return rows


def _summary_row(
    run: Mapping[str, Any],
    velocity_scope_name: str,
    aggregation: str,
    family: str,
    metric: str,
    metadata: Mapping[str, Any],
    cells: int,
    family_count: int,
    value: float,
) -> dict[str, Any]:
    return {
        **{key: run[key] for key in ("run_id", "abbreviation", "color", "is_baseline")},
        "run": run.get("run_path", run.get("run", "")),
        "velocity_scope": velocity_scope_name,
        "aggregation": aggregation,
        "family": family,
        "metric": metric,
        "source_metric": metadata.get("source_metric", metric),
        "display_label": metadata["display_label"],
        "unit": metadata["unit"],
        "lower_is_better": metadata["lower_is_better"],
        "cells": cells,
        "families": family_count,
        "value": value,
    }


def _attach_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = {
        (row["velocity_scope"], row["aggregation"], row["family"], row["metric"]): row
        for row in rows
        if row["is_baseline"]
    }
    for row in rows:
        key = (row["velocity_scope"], row["aggregation"], row["family"], row["metric"])
        if key not in baseline:
            raise ValueError(f"No baseline summary counterpart for {key}.")
        row["delta_from_baseline"] = float(row["value"]) - float(baseline[key]["value"])


def aggregate_summaries(
    cell_rows: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    families: Sequence[str] = FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate velocity, leg-use, and gait metrics without outcome filtering."""
    if not runs or sum(bool(run.get("is_baseline")) for run in runs) != 1:
        raise ValueError("Summary aggregation requires exactly one baseline run.")
    velocity_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []
    gait_rows: list[dict[str, Any]] = []
    domains = (
        (velocity_rows, VELOCITY_METRICS, lambda metric: metric),
        (leg_rows, LEG_METRICS, lambda metric: f"{metric}_abs_imbalance_percent"),
        (gait_rows, GAIT_METRICS, lambda metric: metric),
    )
    for run in runs:
        run_cells = [row for row in cell_rows if row["run_id"] == run["run_id"]]
        if not run_cells:
            raise ValueError(f"No evaluation cells were provided for {run['abbreviation']}.")
        for scope in VELOCITY_SCOPES:
            scoped = [row for row in run_cells if _scope_includes(scope, float(row["velocity_mps"]))]
            family_cells = {family: [row for row in scoped if row["family"] == family] for family in families}
            missing = [family for family, selected in family_cells.items() if not selected]
            if missing:
                raise ValueError(
                    f"Velocity scope {scope!r} has no cells for families {missing} in {run['abbreviation']}."
                )
            for target, metric_registry, field_for_metric in domains:
                family_values: dict[tuple[str, str], float] = {}
                for family in families:
                    selected = family_cells[family]
                    for metric, metadata in metric_registry.items():
                        field = field_for_metric(metric)
                        value = statistics.fmean(float(row[field]) for row in selected)
                        family_values[(family, metric)] = value
                        target.append(
                            _summary_row(
                                run,
                                scope,
                                "family",
                                family,
                                metric,
                                metadata,
                                len(selected),
                                1,
                                value,
                            )
                        )
                for metric, metadata in metric_registry.items():
                    value = statistics.fmean(family_values[(family, metric)] for family in families)
                    target.append(
                        _summary_row(
                            run,
                            scope,
                            "overall",
                            "all_family_balanced",
                            metric,
                            metadata,
                            len(scoped),
                            len(families),
                            value,
                        )
                    )
    for rows in (velocity_rows, leg_rows, gait_rows):
        _attach_baseline_deltas(rows)
    return velocity_rows, leg_rows, gait_rows


def _validate_aggregate_coverage(cell_rows: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]]) -> None:
    for run in runs:
        selected = [row for row in cell_rows if row["run_id"] == run["run_id"]]
        counts = {
            scope: sum(_scope_includes(scope, float(row["velocity_mps"])) for row in selected)
            for scope in VELOCITY_SCOPES
        }
        if counts != EXPECTED_SCOPE_COUNTS:
            raise ValueError(f"Combined cell coverage for {run['abbreviation']} must be 60/30/30; found {counts}.")


def _validate_palette(runs: Sequence[Mapping[str, Any]]) -> None:
    if len(runs) < 2:
        raise ValueError("A comparison manifest must declare at least two runs.")
    baselines = [run for run in runs if run.get("is_baseline")]
    if len(baselines) != 1:
        raise ValueError("A comparison manifest must declare exactly one baseline run.")
    if baselines[0].get("color") != BASELINE_COLOR:
        raise ValueError(f"The no-TRS baseline must use {BASELINE_COLOR}.")
    colors = [str(run.get("color", "")) for run in runs]
    if len(colors) != len(set(colors)):
        raise ValueError("Comparison runs must use unique colors from the v2 palette.")
    for run in runs:
        color = str(run.get("color", ""))
        if color not in V2_PALETTE:
            raise ValueError(f"Run {run.get('abbreviation')!r} uses color {color!r} outside the v2 palette.")
        if not run.get("is_baseline") and color == BASELINE_COLOR:
            raise ValueError("Black is reserved for the no-TRS baseline.")


def load_manifest(
    manifest_path: Path,
    repo_root: Path,
    run_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Load a generated comparison manifest and resolve its run directories."""
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("method_version") != METHOD_VERSION:
        raise ValueError(f"Unsupported gait-closure comparison manifest: {manifest_path}")
    source_runs = manifest.get("runs")
    if not isinstance(source_runs, list) or len(source_runs) < 2:
        raise ValueError(f"Comparison manifest must contain at least two runs: {manifest_path}")
    if any(not isinstance(run, Mapping) for run in source_runs):
        raise ValueError(f"Comparison manifest run descriptors must be objects: {manifest_path}")
    baselines = [run for run in source_runs if run.get("is_baseline")]
    if len(baselines) != 1:
        raise ValueError("Comparison manifest must contain exactly one baseline run.")
    roots = tuple(run_roots or ())
    if not roots:
        configured_roots = manifest.get("run_roots", [])
        roots = tuple(
            (Path(value) if Path(value).is_absolute() else repo_root / Path(value)).resolve()
            for value in configured_roots
        )
    if not roots:
        roots = default_run_roots(repo_root)
    specifications = []
    for run in source_runs:
        abbreviation = str(run.get("abbreviation", "")).strip()
        run_name = str(run.get("folder") or run.get("run_name") or run.get("run_path") or "").strip()
        specifications.append(f"{abbreviation}={run_name}")
    baseline_abbreviation = str(baselines[0]["abbreviation"])
    evaluation_subdir = str(manifest.get("evaluation_subdir", DEFAULT_EVALUATION_SUBDIR))
    resolved = resolve_run_inputs(
        specifications,
        baseline_abbreviation,
        roots,
        repo_root,
        evaluation_subdir,
    )
    by_abbreviation = {str(run["abbreviation"]): run for run in source_runs}
    for run in resolved:
        source = by_abbreviation[str(run["abbreviation"])]
        if "color" in source:
            run["color"] = str(source["color"])
        if "run_id" in source:
            run["run_id"] = str(source["run_id"])
        if "run_name" in source:
            run["run_name"] = str(source["run_name"])
        if "training_metadata_source" in source:
            run["expected_training_metadata_source"] = str(source["training_metadata_source"])
    _validate_palette(resolved)
    return {
        "source": manifest,
        "runs": resolved,
        "run_roots": roots,
        "evaluation_subdir": evaluation_subdir,
        "smoothing_window_iterations": int(
            manifest.get("training", {}).get("smoothing_window_iterations", DEFAULT_SMOOTHING_WINDOW_ITERATIONS)
        ),
        "sample_stride_iterations": int(
            manifest.get("training", {}).get("sample_stride_iterations", DEFAULT_SAMPLE_STRIDE_ITERATIONS)
        ),
        "threshold_reward": float(manifest.get("training", {}).get("threshold_reward", DEFAULT_THRESHOLD_REWARD)),
        "display_name": str(manifest.get("display_name", "Unitree Go2")),
    }


def _load_manifest_provenance_binding(manifest_path: Path) -> dict[str, Any]:
    """Load the prior analysis record that binds a reproduction manifest to its inputs."""
    provenance_path = manifest_path.parent / "analysis_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "Manifest reproduction requires the sibling analysis_provenance.json so input hashes can be "
            f"validated before output is replaced: {provenance_path}"
        )
    provenance_bytes = provenance_path.read_bytes()
    provenance = _read_json(provenance_path)
    if provenance_path.read_bytes() != provenance_bytes:
        raise RuntimeError(f"Comparison provenance changed while it was being loaded: {provenance_path}")
    _validate_record_digest(provenance, "saved gait-closure analysis provenance")
    if provenance.get("schema_version") != 1 or provenance.get("method_version") != METHOD_VERSION:
        raise ValueError(f"Unsupported saved gait-closure analysis provenance: {provenance_path}")
    manifest_sha256 = sha256_file(manifest_path)
    bound_hashes = {str(provenance.get("study_sha256", ""))}
    source_manifest = provenance.get("source_manifest")
    if isinstance(source_manifest, Mapping):
        bound_hashes.add(str(source_manifest.get("sha256", "")))
    if manifest_sha256 not in bound_hashes:
        raise ValueError(
            f"Manifest {manifest_path} is not bound by its sibling analysis provenance; "
            f"SHA-256 {manifest_sha256} was not recorded."
        )
    if not isinstance(provenance.get("inputs"), list) or not provenance["inputs"]:
        raise ValueError(f"Saved gait-closure analysis provenance has no bound input records: {provenance_path}")
    return provenance


def _validate_manifest_protocol(manifest: Mapping[str, Any], protocol: Mapping[str, Any], protocol_sha256: str) -> None:
    """Require the current evaluation protocol to equal the saved reproduction contract."""
    expected_protocol = manifest.get("evaluation_protocol")
    expected_sha256 = str(manifest.get("evaluation_protocol_sha256", ""))
    if not isinstance(expected_protocol, Mapping) or not expected_sha256:
        raise ValueError("Comparison manifest is missing its bound evaluation protocol and SHA-256.")
    actual_expected_sha256 = canonical_sha256(expected_protocol)
    if actual_expected_sha256 != expected_sha256:
        raise ValueError(
            "Comparison manifest evaluation protocol digest is invalid: "
            f"expected {expected_sha256}, found {actual_expected_sha256}."
        )
    if dict(protocol) != dict(expected_protocol) or protocol_sha256 != expected_sha256:
        raise ValueError(
            "Current evaluation inputs do not match the protocol saved in the comparison manifest: "
            f"{protocol_sha256} != {expected_sha256}."
        )


_LEGACY_TREATMENT_FIELDS = (
    "trs_enabled",
    "mirror_coeff",
    "value_coeff",
    "warmup_iterations",
    "rampup_iterations",
    "ramp_shape",
    "validity_mode",
)


def _validate_manifest_treatments(
    manifest: Mapping[str, Any], training_metadata: Mapping[str, Mapping[str, Any]]
) -> None:
    """Require every resolved treatment to equal the saved run descriptor."""
    source_runs = manifest.get("runs")
    if not isinstance(source_runs, list):
        raise ValueError("Comparison manifest has no saved run treatment descriptors.")
    expected_by_id = {str(run.get("run_id", "")): run for run in source_runs if isinstance(run, Mapping)}
    if len(expected_by_id) != len(source_runs) or "" in expected_by_id:
        raise ValueError("Comparison manifest has missing or duplicate saved run identifiers.")
    if set(expected_by_id) != set(training_metadata):
        raise ValueError("Current run identifiers do not match the saved comparison treatment cohort.")
    for run_id, metadata in training_metadata.items():
        expected = expected_by_id[run_id]
        saved_treatment = expected.get("treatment")
        if saved_treatment is not None:
            if not isinstance(saved_treatment, Mapping) or dict(saved_treatment) != metadata.get("treatment"):
                raise ValueError(f"Resolved treatment differs from the saved comparison manifest for {run_id}.")
            continue
        missing = [field for field in _LEGACY_TREATMENT_FIELDS if field not in expected]
        if missing:
            raise ValueError(f"Saved comparison treatment for {run_id} is incomplete: {missing}.")
        saved_legacy = {field: expected[field] for field in _LEGACY_TREATMENT_FIELDS}
        current_legacy = {field: metadata[field] for field in _LEGACY_TREATMENT_FIELDS}
        if saved_legacy != current_legacy:
            raise ValueError(
                f"Resolved legacy-compatible treatment differs from the saved comparison manifest for {run_id}: "
                f"{current_legacy!r} != {saved_legacy!r}."
            )


def _input_hash_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one provenance input record onto relocatable identity and content hashes."""
    run_id = str(record.get("run_id", ""))
    if not run_id:
        raise ValueError("Gait-closure input provenance has no run identifier.")
    context = f"gait-closure input provenance for {run_id}"
    evaluation_files = record.get("evaluation_files")
    training_metadata = record.get("training_metadata")
    if not isinstance(evaluation_files, Mapping) or not evaluation_files or not isinstance(training_metadata, Mapping):
        raise ValueError(f"Malformed {context}.")
    training_files = training_metadata.get("files")
    if not isinstance(training_files, list) or not training_files:
        raise ValueError(f"Missing training-metadata hashes for {run_id}.")
    if any(not isinstance(value, Mapping) for value in training_files):
        raise ValueError(f"Malformed training-metadata hash entries for {run_id}.")
    evaluation_hashes: dict[str, str] = {}
    for name, value in evaluation_files.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Malformed evaluation-file hash {name!r} for {run_id}.")
        evaluation_hashes[str(name)] = _sha256_value(value.get("sha256"), f"evaluation_files.{name}", context)
    source = str(training_metadata.get("source", ""))
    if not source:
        raise ValueError(f"Training-metadata provenance has no source class for {run_id}.")
    contract: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint_sha256": _sha256_value(record.get("checkpoint_sha256"), "checkpoint_sha256", context),
        "event_sha256": _sha256_value(record.get("event_sha256"), "event_sha256", context),
        "evaluation_analysis_record_sha256": _sha256_value(
            record.get("evaluation_analysis_record_sha256"), "evaluation_analysis_record_sha256", context
        ),
        "evaluation_files": evaluation_hashes,
        "training_metadata": {
            "source": source,
            "file_sha256": sorted(
                _sha256_value(value.get("sha256"), "training_metadata.files.sha256", context)
                for value in training_files
            ),
        },
    }
    for field in ("initialization_sha256", "iteration_zero_checkpoint_sha256"):
        if field in record:
            contract[field] = _sha256_value(record[field], field, context)
    legacy_registry = record.get("legacy_registry")
    if isinstance(legacy_registry, Mapping):
        contract["legacy_registry_sha256"] = _sha256_value(
            legacy_registry.get("sha256"), "legacy_registry.sha256", context
        )
    legacy_checksum = record.get("legacy_initialization_checksum_record")
    if isinstance(legacy_checksum, Mapping):
        contract["legacy_initialization_checksum_record_sha256"] = _sha256_value(
            legacy_checksum.get("sha256"), "legacy_initialization_checksum_record.sha256", context
        )
        contract["legacy_initialization_record_sha256"] = _sha256_value(
            legacy_checksum.get("record_sha256"),
            "legacy_initialization_checksum_record.record_sha256",
            context,
        )
    if source == INITIALIZATION_METADATA_SOURCE and "initialization_sha256" not in contract:
        raise ValueError(f"Initialization provenance is missing its record hash for {run_id}.")
    if source in {LEGACY_CONFIG_METADATA_SOURCE, LEGACY_CHECKSUM_ONLY_METADATA_SOURCE}:
        if "iteration_zero_checkpoint_sha256" not in contract or "legacy_registry_sha256" not in contract:
            raise ValueError(f"Legacy provenance is missing its checkpoint or registry hash for {run_id}.")
    if source == LEGACY_CHECKSUM_ONLY_METADATA_SOURCE and (
        "legacy_initialization_checksum_record_sha256" not in contract
        or "legacy_initialization_record_sha256" not in contract
    ):
        raise ValueError(f"Checksum-only legacy provenance is missing its checksum-record hashes for {run_id}.")
    return contract


def _validate_manifest_input_hashes(
    saved_provenance: Mapping[str, Any], current_inputs: Sequence[Mapping[str, Any]]
) -> None:
    """Require all current comparison inputs to match the hashes from the saved analysis."""
    saved_inputs = saved_provenance.get("inputs")
    if not isinstance(saved_inputs, list):
        raise ValueError("Saved comparison provenance has no input-hash records.")
    saved_contracts = {
        str(record.get("run_id", "")): _input_hash_contract(record)
        for record in saved_inputs
        if isinstance(record, Mapping)
    }
    current_contracts = {str(record.get("run_id", "")): _input_hash_contract(record) for record in current_inputs}
    if len(saved_contracts) != len(saved_inputs) or len(current_contracts) != len(current_inputs):
        raise ValueError("Comparison provenance contains missing or duplicate run identifiers.")
    if saved_contracts != current_contracts:
        differing = sorted(
            run_id
            for run_id in set(saved_contracts) | set(current_contracts)
            if saved_contracts.get(run_id) != current_contracts.get(run_id)
        )
        raise ValueError(
            "Current comparison inputs differ from the hashes bound by the saved analysis; "
            f"refusing to overwrite output. Differing runs: {differing}."
        )


@contextlib.contextmanager
def _analysis_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / ".gait_closure_analysis.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another gait-closure comparison is active in {output_dir}.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _commit_staged_outputs(stage_dir: Path, output_dir: Path) -> list[str]:
    """Install all staged files with rollback, placing provenance last."""
    relative_paths = sorted(
        (path.relative_to(stage_dir) for path in stage_dir.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    provenance = Path("analysis_provenance.json")
    if provenance not in relative_paths:
        raise ValueError("Staged output is missing analysis_provenance.json.")
    relative_paths.remove(provenance)
    relative_paths.append(provenance)
    backup_dir = output_dir / f".gait-closure-backup-{uuid.uuid4().hex}"
    backup_dir.mkdir()
    backed_up: list[Path] = []
    installed: list[Path] = []
    try:
        for relative in relative_paths:
            source = stage_dir / relative
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = backup_dir / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backed_up.append(relative)
            os.replace(source, destination)
            installed.append(relative)
    except Exception:
        for relative in reversed(installed):
            (output_dir / relative).unlink(missing_ok=True)
        for relative in reversed(backed_up):
            backup = backup_dir / relative
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
    return [path.as_posix() for path in relative_paths]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return value


def _figure_manifest(entries: Any, stage_dir: Path) -> dict[str, Any]:
    safe_entries = _json_safe(entries)
    if isinstance(safe_entries, Mapping) and "figures" in safe_entries:
        figures = safe_entries["figures"]
    elif isinstance(safe_entries, Mapping):
        figures = list(safe_entries.values())
    elif isinstance(safe_entries, list):
        figures = safe_entries
    else:
        raise ValueError("generate_figures must return figure-manifest entries.")
    enriched: list[Any] = []
    for entry in figures:
        if not isinstance(entry, Mapping):
            enriched.append(entry)
            continue
        item = dict(entry)
        figure_id = str(item.get("figure_id", item.get("id", "")))
        registry = FIGURE_REGISTRY.get(figure_id, {}) if isinstance(FIGURE_REGISTRY, Mapping) else {}
        item.setdefault("title", registry.get("title"))
        item.setdefault("source_tables", registry.get("source_tables", ()))
        for format_name in ("png", "svg"):
            filename = item.get(format_name)
            if filename:
                path = stage_dir / str(filename)
                if not path.is_file():
                    raise FileNotFoundError(f"Figure generator reported a missing {format_name.upper()}: {path}")
                item[f"{format_name}_sha256"] = sha256_file(path)
        enriched.append(item)
    return {
        "schema_version": 1,
        "method_version": METHOD_VERSION,
        "registry": _json_safe(FIGURE_REGISTRY),
        "figures": enriched,
    }


def _reproduce_script() -> str:
    return '''# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce this gait-closure comparison from its resolved manifest."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, *HERE.parents):
    script_dir = candidate / "scripts" / "symm_locomotion"
    if (script_dir / "comparison.py").is_file():
        sys.path.insert(0, str(script_dir))
        break
else:
    raise FileNotFoundError("Could not locate scripts/symm_locomotion from this analysis folder.")

from comparison import main

if __name__ == "__main__":
    raise SystemExit(main(["--manifest", str(HERE / "study.json"), "--output_dir", str(HERE)]))
'''


def _treatment_label(treatment: Mapping[str, Any]) -> str:
    """Render the resolved independent treatment terms for the report table."""
    labels: list[str] = []
    for term, short_name in (("policy", "policy"), ("value", "value"), ("augmentation", "augmentation")):
        config = treatment[term]
        if not config["enabled"]:
            continue
        schedule = config["schedule"]
        label = (
            f"{short_name}={float(config['target_coeff']):g} "
            f"(warmup={schedule['warmup_iterations']}, ramp={schedule['rampup_iterations']})"
        )
        if term == "augmentation":
            label += f", mode={config['mechanism']}"
        labels.append(label)
    return "; ".join(labels) if labels else "No TRS"


def _report_text(
    study: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    figure_manifest: Mapping[str, Any],
    warnings: Sequence[str],
) -> str:
    lines = [
        f"# {study['display_name']} gait-closure comparison",
        "",
        f"Method: `{METHOD_VERSION}`",
        "",
        "This analysis compares the ordered run cohort in `study.json`. The no-TRS baseline is always black; "
        "other runs use the gait-family-v2 palette.",
        "",
        "## Metric and aggregation contract",
        "",
        "- Velocity plots use literal forward-velocity RMSE [m/s] and yaw-velocity RMSE [rad/s].",
        "- Leg plots use per-cell absolute front/hind imbalance for torque², normalized torque², absolute "
        "work, and vertical GRF impulse.",
        "- Normalized torque² is the archived `normalized_torque_utilization` metric, computed from physical "
        "joint effort limits.",
        "- Gait fidelity uses boundary-excluded contact-pattern agreement.",
        "- All, negative, and positive scopes contain 60, 30, and 30 cells per run. No outcome or tracking-success "
        "filter is applied.",
        "- Family rows average selected cells within one family. Overall rows first compute each family mean and "
        "then weight the four gait families equally.",
        "",
        "## Run cohort",
        "",
        "| Abbreviation | Run | Color | Treatment |",
        "|---|---|---|---|",
    ]
    for run, _training in zip(study["runs"], training_rows, strict=True):
        treatment = _treatment_label(run["treatment"])
        lines.append(f"| {run['abbreviation']} | `{run['folder']}` | `{run['color']}` | {treatment} |")
    if warnings:
        lines.extend(["", "## Comparability notes", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "Figure IDs, filenames, source tables, and hashes are recorded in `figure_manifest.json`.",
            "",
        ]
    )
    for entry in figure_manifest.get("figures", []):
        if not isinstance(entry, Mapping):
            continue
        figure_id = entry.get("figure_id", entry.get("id", "Figure"))
        title = entry.get("title", entry.get("stem", ""))
        lines.append(f"- **{figure_id}** {title}")
    lines.extend(
        [
            "",
            "## Reproduction requirements",
            "",
            "Each run needs its terminal checkpoint, one `events.out.tfevents.*` file containing "
            "`Train/mean_reward`, training metadata, and a complete "
            "`evaluations/leg_usage_grid_full_v3` directory containing `study.json`, `progress.json`, "
            "`metrics/cell_metrics.csv`, `metrics/overall_metrics.json`, and "
            "`metrics/analysis_provenance.json`. Raw NPZ recordings are not reread.",
            "",
            "Training metadata normally comes from `provenance/initialization.json`. Legacy runs may instead "
            "use registered `params/agent.yaml` and `params/env.yaml` snapshots plus `model_0.pt`. A "
            "terminal-checkpoint-only archive may replace the omitted checkpoint bytes with the strictly validated "
            "`provenance/legacy_initialization.json` checksum record. These weaker, distinct provenance classes "
            "are identified in `study.json`, `analysis_provenance.json`, and the comparability notes above.",
            "",
            "Run `reproduce.py` with the Isaac Lab Python wrapper to regenerate the datasets and figures.",
            "`source_manifest_snapshot.json` preserves the exact manifest-mode input; in direct mode it mirrors "
            "the generated canonical `study.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparability_warnings(training_metadata: Mapping[str, Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    byte_verified_legacy_labels = [
        str(metadata.get("abbreviation", run_id))
        for run_id, metadata in training_metadata.items()
        if metadata.get("training_metadata_source") == LEGACY_CONFIG_METADATA_SOURCE
    ]
    if byte_verified_legacy_labels:
        warnings.append(
            "Legacy runs lack pre-first-rollout initialization provenance; training settings and the archived "
            "iteration-0 checkpoint were recovered from run snapshots. Configuration and checkpoint bytes are "
            "auditable, but pre-rollout model, optimizer, RNG, and runtime identity is unavailable: "
            f"{', '.join(byte_verified_legacy_labels)}."
        )
    checksum_only_labels = [
        str(metadata.get("abbreviation", run_id))
        for run_id, metadata in training_metadata.items()
        if metadata.get("training_metadata_source") == LEGACY_CHECKSUM_ONLY_METADATA_SOURCE
    ]
    if checksum_only_labels:
        warnings.append(
            "Legacy checksum-only runs lack pre-first-rollout initialization provenance, and their archived "
            "`model_0.pt` bytes are unavailable under the terminal-checkpoint-only publication policy. The "
            "registry-bound checksum record is auditable, but the checkpoint bytes were not reverified and the "
            "pre-rollout model, optimizer, RNG, and runtime identity cannot be audited: "
            f"{', '.join(checksum_only_labels)}."
        )
    byte_verified_checkpoints = {
        metadata.get("iteration_zero_checkpoint_sha256")
        for metadata in training_metadata.values()
        if metadata.get("training_metadata_source") == LEGACY_CONFIG_METADATA_SOURCE
    }
    if len(byte_verified_checkpoints) > 1:
        warnings.append("Legacy runs do not share one byte-identical archived `model_0.pt` checkpoint.")
    checksum_only_checkpoints = {
        metadata.get("iteration_zero_checkpoint_sha256")
        for metadata in training_metadata.values()
        if metadata.get("training_metadata_source") == LEGACY_CHECKSUM_ONLY_METADATA_SOURCE
    }
    if len(checksum_only_checkpoints) > 1:
        warnings.append(
            "Legacy checksum-only records report different `model_0.pt` digests; the omitted checkpoint bytes "
            "cannot be reverified."
        )
    fields = ("seed", "num_envs", "num_steps_per_env", "max_iterations")
    for field in fields:
        values = {metadata[field] for metadata in training_metadata.values()}
        if len(values) > 1:
            warnings.append(f"Training field `{field}` differs across runs: {sorted(values)}.")
    commits = {metadata.get("repo_commit") for metadata in training_metadata.values()}
    dirty = {metadata.get("dirty_tree_diff_sha256") for metadata in training_metadata.values()}
    if len(commits) > 1 or len(dirty) > 1:
        warnings.append(
            "Training source provenance differs across runs; baseline deltas are descriptive rather than a strict "
            "single-factor causal contrast."
        )
    return warnings


def _evaluation_outcome_warnings(evaluations: Sequence[Mapping[str, Any]]) -> list[str]:
    """Describe metric-complete terminated cells retained by outcome-free aggregation."""
    warnings: list[str] = []
    for evaluation in evaluations:
        terminated_cell_ids = [
            str(cell["cell_id"]) for cell in evaluation["cells"] if str(cell.get("status")) == "terminated"
        ]
        if not terminated_cell_ids:
            continue
        run = evaluation["run"]
        count = len(terminated_cell_ids)
        noun = "cell" if count == 1 else "cells"
        rendered_ids = ", ".join(f"`{cell_id}`" for cell_id in terminated_cell_ids)
        warnings.append(
            f"{run['abbreviation']} contains {count} metric-complete terminated evaluation {noun}. "
            "The available post-settle samples are retained by the declared outcome-free aggregation: "
            f"{rendered_ids}."
        )
    return warnings


def _serializable_study(
    runs: Sequence[Mapping[str, Any]],
    run_roots: Sequence[Path],
    repo_root: Path,
    evaluation_subdir: str,
    smoothing_window_iterations: int,
    sample_stride_iterations: int,
    threshold_reward: float,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    training_metadata: Mapping[str, Mapping[str, Any]],
    display_name: str,
) -> dict[str, Any]:
    serialized_runs = []
    for run in runs:
        metadata = training_metadata[str(run["run_id"])]
        serialized_runs.append(
            {
                key: run[key]
                for key in (
                    "run_id",
                    "abbreviation",
                    "label",
                    "run_name",
                    "folder",
                    "run_path",
                    "is_baseline",
                    "color",
                )
            }
            | {
                key: metadata[key]
                for key in (
                    "trs_enabled",
                    "mirror_coeff",
                    "value_coeff",
                    "warmup_iterations",
                    "rampup_iterations",
                    "ramp_shape",
                    "validity_mode",
                )
            }
            | {
                "training_metadata_source": metadata["training_metadata_source"],
                "treatment": metadata["treatment"],
            }
        )
    training = {
        "smoothing_window_iterations": smoothing_window_iterations,
        "sample_stride_iterations": sample_stride_iterations,
        "threshold_reward": threshold_reward,
    }
    for field in ("seed", "num_envs", "num_steps_per_env", "max_iterations"):
        values = {metadata[field] for metadata in training_metadata.values()}
        if len(values) == 1:
            training[field] = next(iter(values))
    return {
        "schema_version": 1,
        "method_version": METHOD_VERSION,
        "robot": protocol.get("robot"),
        "display_name": display_name,
        "run_roots": [_repo_relative(root, repo_root) for root in run_roots],
        "evaluation_subdir": evaluation_subdir,
        "baseline_run_id": next(run["run_id"] for run in runs if run["is_baseline"]),
        "runs": serialized_runs,
        "training": training,
        "reward_target": threshold_reward,
        "aggregation": {
            "velocity_scopes": list(VELOCITY_SCOPES),
            "scope_cells_per_run": EXPECTED_SCOPE_COUNTS,
            "families": list(FAMILIES),
            "within_family": "arithmetic mean of every selected cell in the family",
            "overall": "equal arithmetic mean of the four gait-family means",
            "outcome_filter": "none",
        },
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": protocol_sha256,
    }


def _input_provenance_records(
    repo_root: Path,
    evaluations: Sequence[Mapping[str, Any]],
    training_metadata: Mapping[str, Mapping[str, Any]],
    event_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build the per-run input records used by provenance and reproduction."""
    inputs: list[dict[str, Any]] = []
    for evaluation in evaluations:
        run = evaluation["run"]
        metadata = training_metadata[str(run["run_id"])]
        metadata_files = [
            {
                "path": _repo_relative(Path(str(entry["path"])), repo_root),
                "sha256": entry["sha256"],
            }
            for entry in metadata["training_metadata_files"]
        ]
        input_record = {
            "run_id": run["run_id"],
            "abbreviation": run["abbreviation"],
            "run_path": run["run_path"],
            "checkpoint_path": _repo_relative(evaluation["checkpoint_path"], repo_root),
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "event_sha256": event_hashes[str(run["run_id"])],
            "training_metadata": {
                "source": metadata["training_metadata_source"],
                "files": metadata_files,
            },
            "evaluation_files": {
                key: {
                    "path": _repo_relative(evaluation["input_paths"][key], repo_root),
                    "sha256": value,
                }
                for key, value in evaluation["input_hashes"].items()
            },
            "evaluation_analysis_record_sha256": evaluation["provenance"]["record_sha256"],
        }
        if metadata["training_metadata_source"] == INITIALIZATION_METADATA_SOURCE:
            input_record["initialization_path"] = _repo_relative(Path(str(metadata["initialization_path"])), repo_root)
            input_record["initialization_sha256"] = metadata["initialization_sha256"]
        else:
            registry = metadata["legacy_registry"]
            input_record["legacy_registry"] = {
                **{key: registry[key] for key in ("cohort_id", "run_id", "classification", "sha256")},
                "path": _repo_relative(Path(str(registry["path"])), repo_root),
            }
            input_record["iteration_zero_checkpoint_sha256"] = metadata["iteration_zero_checkpoint_sha256"]
            input_record["iteration_zero_checkpoint_availability"] = metadata["iteration_zero_checkpoint_availability"]
            checksum_record = metadata.get("legacy_initialization_checksum_record")
            if isinstance(checksum_record, Mapping):
                input_record["legacy_initialization_checksum_record"] = {
                    "path": _repo_relative(Path(str(checksum_record["path"])), repo_root),
                    "sha256": checksum_record["sha256"],
                    "record_sha256": checksum_record["record_sha256"],
                }
        inputs.append(input_record)
    return inputs


def _analysis_provenance(
    stage_dir: Path,
    repo_root: Path,
    study: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    training_metadata: Mapping[str, Mapping[str, Any]],
    event_hashes: Mapping[str, str],
    warnings: Sequence[str],
    source_manifest_path: Path | None,
    source_manifest_snapshot_path: Path,
) -> dict[str, Any]:
    comparison_path = Path(__file__).resolve()
    comparison_record = {
        "path": _repo_relative(comparison_path, repo_root),
        "sha256": sha256_file(comparison_path),
    }
    tensorboard_parser_path = Path(tensorboard_scalars_module.__file__).resolve()
    inputs = _input_provenance_records(repo_root, evaluations, training_metadata, event_hashes)
    output_hashes = {
        path.relative_to(stage_dir).as_posix(): sha256_file(path)
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file() and path.name != "analysis_provenance.json"
    }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "method_version": METHOD_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_script": comparison_record,
        "comparison_launchers": _comparison_launcher_provenance(repo_root),
        # Keep the historical field while binding it to this unified module.
        "plot_script": comparison_record.copy(),
        "tensorboard_scalar_parser": {
            "path": _repo_relative(tensorboard_parser_path, repo_root),
            "sha256": sha256_file(tensorboard_parser_path),
        },
        "source_manifest": None
        if source_manifest_path is None
        else {
            "path": source_manifest_snapshot_path.name,
            "original_path": _repo_relative(source_manifest_path, repo_root),
            "sha256": sha256_file(source_manifest_snapshot_path),
        },
        "study_sha256": sha256_file(stage_dir / "study.json"),
        "style_sha256": sha256_file(stage_dir / "style.json"),
        "evaluation_protocol_sha256": study["evaluation_protocol_sha256"],
        "warnings": list(warnings),
        "inputs": inputs,
        "outputs": output_hashes,
    }
    provenance["record_sha256"] = canonical_sha256(provenance)
    return provenance


def run_comparison(
    output_dir: Path,
    *,
    repo_root: Path,
    run_specs: Sequence[str] | None = None,
    baseline_abbreviation: str | None = None,
    run_roots: Sequence[Path] | None = None,
    manifest_path: Path | None = None,
    evaluation_subdir: str = DEFAULT_EVALUATION_SUBDIR,
    smoothing_window_iterations: int = DEFAULT_SMOOTHING_WINDOW_ITERATIONS,
    sample_stride_iterations: int = DEFAULT_SAMPLE_STRIDE_ITERATIONS,
    threshold_reward: float = DEFAULT_THRESHOLD_REWARD,
    display_name: str = "Unitree Go2",
) -> dict[str, Any]:
    """Generate one complete comparison and atomically install its artifacts."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    source_manifest_path: Path | None = None
    source_manifest_bytes: bytes | None = None
    saved_manifest: Mapping[str, Any] | None = None
    saved_provenance: Mapping[str, Any] | None = None
    if manifest_path is not None:
        source_manifest_path = manifest_path.resolve()
        source_manifest_bytes = source_manifest_path.read_bytes()
        loaded = load_manifest(source_manifest_path, repo_root, run_roots)
        saved_manifest = loaded["source"]
        saved_provenance = _load_manifest_provenance_binding(source_manifest_path)
        if source_manifest_path.read_bytes() != source_manifest_bytes:
            raise RuntimeError(f"Comparison manifest changed while it was being loaded: {source_manifest_path}")
        runs = loaded["runs"]
        resolved_roots = tuple(Path(root).resolve() for root in loaded["run_roots"])
        evaluation_subdir = loaded["evaluation_subdir"]
        smoothing_window_iterations = loaded["smoothing_window_iterations"]
        sample_stride_iterations = loaded["sample_stride_iterations"]
        threshold_reward = loaded["threshold_reward"]
        display_name = loaded["display_name"]
    else:
        if not baseline_abbreviation:
            raise ValueError("Direct comparison mode requires --baseline ABBREVIATION.")
        resolved_roots = tuple(Path(root).resolve() for root in (run_roots or default_run_roots(repo_root)))
        runs = resolve_run_inputs(
            run_specs or (),
            baseline_abbreviation,
            resolved_roots,
            repo_root,
            evaluation_subdir,
        )
    _validate_palette(runs)

    training_metadata: dict[str, Mapping[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    for run in runs:
        print(f"Validating {run['abbreviation']}...", flush=True)
        training = _load_training_metadata(run, repo_root, manifest_bound=saved_manifest is not None)
        training["abbreviation"] = run["abbreviation"]
        training_metadata[str(run["run_id"])] = training
        evaluations.append(_load_evaluation(run, evaluation_subdir))
    if saved_manifest is not None:
        _validate_manifest_treatments(saved_manifest, training_metadata)

    protocol = _protocol_signature(evaluations[0]["study"])
    protocol_sha256 = canonical_sha256(protocol)
    if saved_manifest is not None and saved_provenance is not None:
        _validate_manifest_protocol(saved_manifest, protocol, protocol_sha256)
        if saved_provenance.get("evaluation_protocol_sha256") != protocol_sha256:
            raise ValueError(
                "Saved comparison provenance does not bind the evaluation protocol declared by its manifest."
            )
    for evaluation in evaluations[1:]:
        candidate = _protocol_signature(evaluation["study"])
        if candidate != protocol:
            raise ValueError(
                f"Full-v3 protocol differs for {evaluation['run']['abbreviation']}: "
                f"{canonical_sha256(candidate)} != {protocol_sha256}."
            )
    analyzer_hashes = {evaluation["provenance"].get("analyzer_sha256") for evaluation in evaluations}
    metrics_hashes = {evaluation["provenance"].get("metrics_sha256") for evaluation in evaluations}
    if len(analyzer_hashes) != 1 or None in analyzer_hashes:
        raise ValueError("Comparison runs were evaluated by different or unknown analyzer versions.")
    if len(metrics_hashes) != 1 or None in metrics_hashes:
        raise ValueError("Comparison runs were evaluated by different or unknown metric implementations.")

    training_rows, learning_points, event_hashes = _build_training_data(
        runs,
        training_metadata,
        repo_root,
        smoothing_window_iterations,
        sample_stride_iterations,
    )
    if saved_provenance is not None:
        current_inputs = _input_provenance_records(repo_root, evaluations, training_metadata, event_hashes)
        _validate_manifest_input_hashes(saved_provenance, current_inputs)
    evaluation_cells = _build_evaluation_cells(evaluations)
    _validate_aggregate_coverage(evaluation_cells, runs)
    velocity_rows, leg_rows, gait_rows = aggregate_summaries(evaluation_cells, runs)
    warnings = [*_comparability_warnings(training_metadata), *_evaluation_outcome_warnings(evaluations)]
    study = _serializable_study(
        runs,
        resolved_roots,
        repo_root,
        evaluation_subdir,
        smoothing_window_iterations,
        sample_stride_iterations,
        threshold_reward,
        protocol,
        protocol_sha256,
        training_metadata,
        display_name,
    )
    study["families"] = list(FAMILIES)

    output_dir.mkdir(parents=True, exist_ok=True)
    with _analysis_lock(output_dir):
        # A normal child directory inherits the workspace ACL on Windows. Files
        # keep that ACL when atomically moved into the final directory.
        stage_dir = output_dir / f".gait-closure-stage-{uuid.uuid4().hex}"
        stage_dir.mkdir()
        try:
            _write_json(stage_dir / "study.json", study)
            source_manifest_snapshot_path = stage_dir / "source_manifest_snapshot.json"
            if source_manifest_bytes is None:
                shutil.copyfile(stage_dir / "study.json", source_manifest_snapshot_path)
            else:
                source_manifest_snapshot_path.write_bytes(source_manifest_bytes)
            _write_json(stage_dir / "style.json", _json_safe(STYLE))
            _write_csv(stage_dir / "training_efficiency.csv", training_rows)
            _write_csv(stage_dir / "learning_curve_points.csv", learning_points)
            _write_csv(stage_dir / "evaluation_cells.csv", evaluation_cells)
            _write_csv(stage_dir / "velocity_tracking_summary.csv", velocity_rows)
            _write_csv(stage_dir / "leg_usage_summary.csv", leg_rows)
            _write_csv(stage_dir / "gait_fidelity_summary.csv", gait_rows)
            print("Rendering publication figures...", flush=True)
            figure_entries = generate_figures(
                stage_dir,
                study,
                learning_points,
                velocity_rows,
                leg_rows,
                gait_rows,
            )
            figure_manifest = _figure_manifest(figure_entries, stage_dir)
            _write_json(stage_dir / "figure_manifest.json", figure_manifest)
            summary = {
                "schema_version": 1,
                "method_version": METHOD_VERSION,
                "run_count": len(runs),
                "baseline_run_id": study["baseline_run_id"],
                "evaluation_cells": len(evaluation_cells),
                "scope_cells_per_run": EXPECTED_SCOPE_COUNTS,
                "summary_rows": {
                    "velocity": len(velocity_rows),
                    "leg_usage": len(leg_rows),
                    "gait_fidelity": len(gait_rows),
                },
                "evaluation_protocol_sha256": protocol_sha256,
                "warnings": warnings,
                "overall_all_velocity": {
                    "velocity": [
                        row
                        for row in velocity_rows
                        if row["velocity_scope"] == "all" and row["aggregation"] == "overall"
                    ],
                    "leg_usage": [
                        row for row in leg_rows if row["velocity_scope"] == "all" and row["aggregation"] == "overall"
                    ],
                    "gait_fidelity": [
                        row for row in gait_rows if row["velocity_scope"] == "all" and row["aggregation"] == "overall"
                    ],
                },
            }
            _write_json(stage_dir / "summary.json", summary)
            (stage_dir / "REPORT.md").write_text(
                _report_text(study, training_rows, figure_manifest, warnings), encoding="utf-8"
            )
            (stage_dir / "reproduce.py").write_text(_reproduce_script(), encoding="utf-8")
            provenance = _analysis_provenance(
                stage_dir,
                repo_root,
                study,
                evaluations,
                training_metadata,
                event_hashes,
                warnings,
                source_manifest_path,
                source_manifest_snapshot_path,
            )
            _write_json(stage_dir / "analysis_provenance.json", provenance)
            _validate_record_digest(provenance, "generated gait-closure analysis provenance")
            installed = _commit_staged_outputs(stage_dir, output_dir)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
    return {
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "baseline_run_id": study["baseline_run_id"],
        "installed_files": installed,
        "study": study,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the canonical multi-run comparison argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare any ordered set of completed Go2 full-v3 gait evaluations and their training reward traces."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ABBREVIATION=RUN_NAME",
        help="Ordered run specification; repeat once per run.",
    )
    parser.add_argument(
        "--baseline",
        help="Abbreviation of the no-TRS baseline. The baseline is displayed first and always uses black.",
    )
    parser.add_argument(
        "--run_root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Run-directory search root; repeat to set search order. Defaults to the primary Go2 root followed by "
            "the good_runs archive."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Analysis output directory. Defaults to the manifest directory in --manifest mode.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Previously generated study.json to reproduce. Cannot be combined with --run or --baseline.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used to resolve relative run, manifest, and output paths.",
    )
    parser.add_argument(
        "--evaluation_subdir",
        default=DEFAULT_EVALUATION_SUBDIR,
        help="Evaluation subdirectory below every run in direct mode.",
    )
    parser.add_argument(
        "--smoothing_window_iterations",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW_ITERATIONS,
        help="Trailing-mean window used by the training-efficiency figure.",
    )
    parser.add_argument(
        "--sample_stride_iterations",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE_ITERATIONS,
        help="Stride used to retain plotted learning-curve points after smoothing.",
    )
    parser.add_argument(
        "--threshold_reward",
        type=float,
        default=DEFAULT_THRESHOLD_REWARD,
        help="Reward reference drawn in the training-efficiency figure.",
    )
    parser.add_argument(
        "--display_name",
        default="Unitree Go2",
        help="Robot display name used in figure and report titles.",
    )
    return parser


def _resolve_path(path: Path, repo_root: Path) -> Path:
    """Resolve a CLI path relative to the configured repository root."""
    return (path if path.is_absolute() else repo_root / path).resolve()


def main(arguments: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and generate one comparison."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    repo_root = args.repo_root.resolve()
    if args.manifest is not None and (args.run or args.baseline is not None):
        parser.error("--manifest cannot be combined with --run or --baseline.")
    if args.manifest is None and (not args.run or args.baseline is None):
        parser.error("Direct mode requires repeated --run entries and --baseline ABBREVIATION.")

    manifest_path = None if args.manifest is None else _resolve_path(args.manifest, repo_root)
    if args.output_dir is None:
        if manifest_path is None:
            parser.error("Direct mode requires --output_dir PATH.")
        output_dir = manifest_path.parent
    else:
        output_dir = _resolve_path(args.output_dir, repo_root)
    roots = (
        tuple(_resolve_path(path, repo_root) for path in args.run_root)
        if args.run_root
        else None
        if manifest_path is not None
        else default_run_roots(repo_root)
    )
    result = run_comparison(
        output_dir,
        repo_root=repo_root,
        run_specs=args.run,
        baseline_abbreviation=args.baseline,
        run_roots=roots,
        manifest_path=manifest_path,
        evaluation_subdir=args.evaluation_subdir,
        smoothing_window_iterations=args.smoothing_window_iterations,
        sample_stride_iterations=args.sample_stride_iterations,
        threshold_reward=args.threshold_reward,
        display_name=args.display_name,
    )
    print(
        f"Generated gait-closure comparison for {result['run_count']} runs in {result['output_dir']}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
