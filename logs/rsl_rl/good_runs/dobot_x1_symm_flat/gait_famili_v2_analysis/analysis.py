# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a good-runs learning and fixed-grid leg-usage comparison.

The visual grammar intentionally matches the earlier two-command good-runs
study, while the leg-usage values come from the newer 10-gait by 6-velocity
``analyze_leg_usage`` grid. Direction is kept in separate panels and gait
families receive equal weight, so signed front/hind effects cannot cancel
between forward and backward motion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO_ROOT / "scripts" / "symm_locomotion"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_trs_grid as training_analysis  # noqa: E402
import plot_trs_tensorboard as reward_plot  # noqa: E402

METHOD_VERSION = "gait_famili_v2_fixed_grid_v1"
MANIFEST_PATH = OUTPUT_DIR / "study.json"
COLORS = (
    "#202124",
    "#0072B2",
    "#D97706",
    "#00875A",
    "#CC79A7",
    "#56B4E9",
    "#8E44AD",
    "#A65E2E",
)
METRICS = (
    ("torque_squared", "Torque²", "N² m² s"),
    ("absolute_work", "Absolute work", "J"),
    ("vertical_grf_impulse", "Vertical GRF impulse", "N s"),
)
DIRECTIONS = ("backward", "forward")


@dataclass(frozen=True)
class Run:
    """One archived training run and its completed fixed-grid evaluation."""

    label: str
    slug: str
    robot: str
    folder: str
    mirror_coeff: float
    value_coeff: float
    warmup_iterations: int | None
    rampup_iterations: int
    trs_enabled: bool
    run_root: Path

    @property
    def path(self) -> Path:
        """Return the run directory."""
        return self.run_root / self.folder

    @property
    def grid_root(self) -> Path:
        """Return the fixed-grid evaluation directory."""
        return self.path / "evaluations" / "leg_usage_grid"

    @property
    def run_spec(self) -> training_analysis.RunSpec:
        """Return the descriptor used by the TensorBoard parser."""
        return training_analysis.RunSpec(
            robot=self.robot,
            run_path=self.path,
            evaluation_path=self.path / "plots" / "play" / "sim_data.npz",
            mirror_coeff=self.mirror_coeff,
            value_coeff=self.value_coeff,
            warmup_iterations=self.warmup_iterations,
            trs_enabled=self.trs_enabled,
        )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    """Return an arithmetic mean while rejecting empty groups."""
    if not values:
        raise ValueError("Cannot aggregate an empty fixed-grid group.")
    return statistics.fmean(values)


def _load_manifest() -> tuple[dict[str, Any], tuple[Run, ...]]:
    """Load the study manifest and resolve its run paths."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("study.json schema_version must be 1.")
    if payload.get("analysis_method_version") != METHOD_VERSION:
        raise ValueError("study.json analysis_method_version does not match this engine.")
    run_root = REPO_ROOT / payload["run_root"]
    runs = tuple(
        Run(
            label=str(entry["label"]),
            slug=str(entry["slug"]),
            robot=str(payload["robot"]),
            folder=str(entry["folder"]),
            mirror_coeff=float(entry["mirror_coeff"]),
            value_coeff=float(entry["value_coeff"]),
            warmup_iterations=(None if entry["warmup_iterations"] is None else int(entry["warmup_iterations"])),
            rampup_iterations=int(entry["rampup_iterations"]),
            trs_enabled=bool(entry["trs_enabled"]),
            run_root=run_root,
        )
        for entry in payload["runs"]
    )
    if not 2 <= len(runs) <= len(COLORS) or len({run.slug for run in runs}) != len(runs):
        raise ValueError(f"This comparison requires 2-{len(COLORS)} uniquely named runs.")
    return payload, runs


def _protocol_projection(study: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must match across all fixed-grid evaluations."""
    return {
        "method_version": study["method_version"],
        "robot": study["robot"],
        "task": study["task"],
        "gait_library_version": study["gait_library_version"],
        "gaits": study["gaits"],
        "velocities_mps": study["velocities_mps"],
        "settle_s": study["settle_s"],
        "measure_s": study["measure_s"],
        "settle_steps": study["settle_steps"],
        "measure_steps": study["measure_steps"],
        "step_dt": study["step_dt"],
        "evaluation_seed": study["evaluation_seed"],
        "nominal_profile": study["nominal_profile"],
        "runtime_overrides": study["runtime_overrides"],
    }


def _load_grid(
    manifest: dict[str, Any], run: Run, expected_protocol: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load and strictly validate one completed fixed-grid analysis."""
    study_path = run.grid_root / "study.json"
    metrics_dir = run.grid_root / "metrics"
    cell_path = metrics_dir / "cell_metrics.csv"
    overall_path = metrics_dir / "overall_metrics.json"
    provenance_path = metrics_dir / "analysis_provenance.json"
    progress_path = run.grid_root / "progress.json"
    required = (study_path, cell_path, overall_path, provenance_path, progress_path, run.path / "model_19999.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts for {run.label}: {missing}")

    study = json.loads(study_path.read_text(encoding="utf-8"))
    protocol = _protocol_projection(study)
    grid_cfg = manifest["fixed_grid"]
    if protocol["method_version"] != grid_cfg["method_version"]:
        raise ValueError(f"Unexpected grid method for {run.label}.")
    if protocol["robot"] != manifest["robot"] or protocol["gait_library_version"] != grid_cfg["gait_library_version"]:
        raise ValueError(f"Unexpected robot or gait library for {run.label}.")
    if protocol["velocities_mps"] != grid_cfg["velocities_mps"]:
        raise ValueError(f"Velocity grid differs for {run.label}.")
    if protocol["settle_s"] != grid_cfg["settle_s"] or protocol["measure_s"] != grid_cfg["measure_s"]:
        raise ValueError(f"Measurement timing differs for {run.label}.")
    if expected_protocol is not None and protocol != expected_protocol:
        raise ValueError(f"Fixed-grid protocol differs for {run.label}.")

    checkpoint_sha = _sha256(run.path / "model_19999.pt")
    if int(study["checkpoint"]["iteration"]) != 19999 or study["checkpoint"]["sha256"] != checkpoint_sha:
        raise ValueError(f"Checkpoint identity mismatch for {run.label}.")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if (
        progress.get("status") != "complete"
        or int(progress.get("completed_cells", -1)) != 60
        or int(progress.get("successful_cells", -1)) != 60
    ):
        raise ValueError(f"Fixed-grid progress is incomplete for {run.label}.")

    with cell_path.open("r", encoding="utf-8", newline="") as stream:
        cells = list(csv.DictReader(stream))
    if len(cells) != 60 or any(cell["status"] != "valid" for cell in cells):
        raise ValueError(f"Expected 60 valid cells for {run.label}, found {len(cells)}.")
    expected_cells = {
        (int(gait["index"]), float(velocity)) for gait in study["gaits"] for velocity in study["velocities_mps"]
    }
    actual_cells = {(int(cell["gait_index"]), float(cell["velocity_mps"])) for cell in cells}
    if len(study["gaits"]) != 10 or actual_cells != expected_cells:
        raise ValueError(f"Gait-speed coverage differs for {run.label}.")
    if any(cell["planar_tracking_success"].lower() != "true" for cell in cells):
        raise ValueError(f"Not all cells pass planar tracking for {run.label}.")

    overall = json.loads(overall_path.read_text(encoding="utf-8"))
    if not overall["coverage"]["complete"] or int(overall["coverage"]["valid_cells"]) != 60:
        raise ValueError(f"Metrics coverage is incomplete for {run.label}.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["input_study_sha256"] != _sha256(study_path):
        raise ValueError(f"Analysis provenance does not match study.json for {run.label}.")
    return (
        cells,
        protocol,
        {
            "checkpoint_sha256": checkpoint_sha,
            "grid_study_sha256": _sha256(study_path),
            "cell_metrics_sha256": _sha256(cell_path),
            "grid_analysis_provenance_sha256": _sha256(provenance_path),
            "grid_analyzer_sha256": provenance["analyzer_sha256"],
            "grid_source_provenance_sha256": study["source_provenance"]["sha256"],
            "combined_tracking_cells": sum(cell["tracking_success"].lower() == "true" for cell in cells),
            "planar_tracking_cells": 60,
            "yaw_tracking_cells": sum(cell["yaw_tracking_success"].lower() == "true" for cell in cells),
        },
    )


def _cell_metric_rows(runs: tuple[Run, ...], cells_by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Expand the fixed-grid cell table to one row per physical metric."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        for cell in cells_by_run[run.slug]:
            for metric, label, unit in METRICS:
                rows.append(
                    {
                        "run": run.folder,
                        "label": run.label,
                        "cell_id": cell["cell_id"],
                        "gait_index": int(cell["gait_index"]),
                        "gait_name": cell["gait_name"],
                        "family": cell["family"],
                        "direction": cell["velocity_sign"],
                        "velocity_mps": float(cell["velocity_mps"]),
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "front": float(cell[f"{metric}_front_integral"]),
                        "hind": float(cell[f"{metric}_hind_integral"]),
                        "signed_imbalance_percent": float(cell[f"{metric}_signed_imbalance_percent"]),
                        "absolute_imbalance_percent": float(cell[f"{metric}_abs_imbalance_percent"]),
                        "total_per_directed_m": float(cell[f"{metric}_total_per_directed_m"]),
                        "planar_tracking_success": cell["planar_tracking_success"],
                        "yaw_tracking_success": cell["yaw_tracking_success"],
                        "tracking_success": cell["tracking_success"],
                    }
                )
    return rows


def _aggregate_direction(
    manifest: dict[str, Any], runs: tuple[Run, ...], detailed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build family and equal-family directional summaries."""
    families = manifest["fixed_grid"]["families"]
    summaries: list[dict[str, Any]] = []
    for run in runs:
        for direction in DIRECTIONS:
            for metric, label, unit in METRICS:
                base = [
                    row
                    for row in detailed
                    if row["run"] == run.folder and row["direction"] == direction and row["metric"] == metric
                ]
                family_rows = []
                for family in families:
                    selected = [row for row in base if row["family"] == family]
                    expected = 3 if family in {"trot", "bound"} else 12
                    if len(selected) != expected:
                        raise ValueError(f"Unexpected {family}/{direction} coverage for {run.label}: {len(selected)}")
                    summary = {
                        "run": run.folder,
                        "label": run.label,
                        "direction": direction,
                        "family": family,
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "cells": len(selected),
                        "signed_imbalance_percent": _mean([float(row["signed_imbalance_percent"]) for row in selected]),
                        "mean_absolute_imbalance_percent": _mean(
                            [float(row["absolute_imbalance_percent"]) for row in selected]
                        ),
                        "total_per_directed_m": _mean([float(row["total_per_directed_m"]) for row in selected]),
                    }
                    summaries.append(summary)
                    family_rows.append(summary)
                summaries.append(
                    {
                        "run": run.folder,
                        "label": run.label,
                        "direction": direction,
                        "family": "all_family_balanced",
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "cells": len(base),
                        "signed_imbalance_percent": _mean(
                            [float(row["signed_imbalance_percent"]) for row in family_rows]
                        ),
                        "mean_absolute_imbalance_percent": _mean(
                            [float(row["mean_absolute_imbalance_percent"]) for row in family_rows]
                        ),
                        "total_per_directed_m": _mean([float(row["total_per_directed_m"]) for row in family_rows]),
                    }
                )
    return summaries


def _aggregate_commands(
    manifest: dict[str, Any], runs: tuple[Run, ...], detailed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build exact-velocity family and equal-family summaries."""
    families = manifest["fixed_grid"]["families"]
    summaries: list[dict[str, Any]] = []
    for run in runs:
        for velocity in manifest["fixed_grid"]["velocities_mps"]:
            direction = "backward" if velocity < 0.0 else "forward"
            for metric, label, unit in METRICS:
                base = [
                    row
                    for row in detailed
                    if row["run"] == run.folder
                    and float(row["velocity_mps"]) == float(velocity)
                    and row["metric"] == metric
                ]
                family_rows = []
                for family in families:
                    selected = [row for row in base if row["family"] == family]
                    expected = 1 if family in {"trot", "bound"} else 4
                    if len(selected) != expected:
                        raise ValueError(
                            f"Unexpected {family}/vx={velocity:+.1f} coverage for {run.label}: {len(selected)}"
                        )
                    summary = {
                        "run": run.folder,
                        "label": run.label,
                        "velocity_mps": float(velocity),
                        "direction": direction,
                        "family": family,
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "cells": len(selected),
                        "signed_imbalance_percent": _mean([float(row["signed_imbalance_percent"]) for row in selected]),
                        "mean_absolute_imbalance_percent": _mean(
                            [float(row["absolute_imbalance_percent"]) for row in selected]
                        ),
                        "total_per_directed_m": _mean([float(row["total_per_directed_m"]) for row in selected]),
                    }
                    summaries.append(summary)
                    family_rows.append(summary)
                summaries.append(
                    {
                        "run": run.folder,
                        "label": run.label,
                        "velocity_mps": float(velocity),
                        "direction": direction,
                        "family": "all_family_balanced",
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "cells": len(base),
                        "signed_imbalance_percent": _mean(
                            [float(row["signed_imbalance_percent"]) for row in family_rows]
                        ),
                        "mean_absolute_imbalance_percent": _mean(
                            [float(row["mean_absolute_imbalance_percent"]) for row in family_rows]
                        ),
                        "total_per_directed_m": _mean([float(row["total_per_directed_m"]) for row in family_rows]),
                    }
                )
    return summaries


def _overall_leg_summary(
    runs: tuple[Run, ...], directional: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Return direction-balanced headline leg-usage values."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for run in runs:
        result[run.slug] = {}
        for metric, _label, _unit in METRICS:
            rows = [
                row
                for row in directional
                if row["run"] == run.folder and row["metric"] == metric and row["family"] == "all_family_balanced"
            ]
            result[run.slug][metric] = {
                "mean_absolute_imbalance_percent": _mean(
                    [float(row["mean_absolute_imbalance_percent"]) for row in rows]
                ),
                "total_per_directed_m": _mean([float(row["total_per_directed_m"]) for row in rows]),
                "backward_signed_imbalance_percent": next(
                    float(row["signed_imbalance_percent"]) for row in rows if row["direction"] == "backward"
                ),
                "forward_signed_imbalance_percent": next(
                    float(row["signed_imbalance_percent"]) for row in rows if row["direction"] == "forward"
                ),
            }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV using their first row's stable field order."""
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_rows(runs: tuple[Run, ...]) -> tuple[list[dict[str, Any]], list[reward_plot.RewardCurve]]:
    """Analyze all training traces and load their display curves."""
    rows: list[dict[str, Any]] = []
    curves: list[reward_plot.RewardCurve] = []
    for run in runs:
        print(f"Analyzing training: {run.label}", flush=True)
        result = training_analysis.analyze_training(run.run_spec)
        threshold = result["thresholds"]["reward_35"]
        rows.append(
            {
                "run": run.folder,
                "label": run.label,
                "mirror_loss_coeff": run.mirror_coeff,
                "value_loss_coeff": run.value_coeff,
                "warmup_iterations": "" if run.warmup_iterations is None else run.warmup_iterations,
                "rampup_iterations": run.rampup_iterations,
                "iterations": result["iterations"],
                "transitions_per_iteration": training_analysis.SAMPLES_PER_ITERATION,
                "environment_transitions": result["environment_transitions"],
                "wall_time_hours": result["wall_time_hours"],
                "reward_auc_first_10000": result["mean_reward"]["auc_first_10000"],
                "reward_auc_full": result["mean_reward"]["auc_all"],
                "reward_last_1000_mean": result["mean_reward"]["last_1000_mean"],
                "reward_last_1000_std": result["mean_reward"]["last_1000_std"],
                "reward_35_iteration": "" if threshold is None else threshold["iteration"],
                "reward_35_transitions": "" if threshold is None else threshold["environment_transitions"],
                "episode_length_last_1000_mean": result["mean_episode_length"]["last_1000_mean"],
                "velocity_error_xy_last_1000_mean": result["velocity_error_xy"]["last_1000_mean"],
                "velocity_error_yaw_last_1000_mean": result["velocity_error_yaw"]["last_1000_mean"],
                "throughput_fps_mean": result["throughput_fps"]["mean"],
                "learning_seconds_per_iteration_mean": result["timing_seconds_per_iteration"]["learning_mean"],
            }
        )
        curves.append(reward_plot.load_reward_curve(run.run_spec, 200))
    return rows, curves


def _write_learning_points(runs: tuple[Run, ...], curves: list[reward_plot.RewardCurve]) -> None:
    """Write the exact smoothed points used in the learning-curve figure."""
    rows = []
    for run, curve in zip(runs, curves, strict=True):
        rows.extend(
            {
                "run": run.folder,
                "label": run.label,
                "transitions_millions": transitions,
                "elapsed_hours": hours,
                "smoothed_reward": reward,
            }
            for transitions, hours, reward in zip(
                curve.transitions_millions, curve.elapsed_hours, curve.rewards, strict=True
            )
        )
    _write_csv(OUTPUT_DIR / "learning_curve_points.csv", rows)


def _write_learning_svg(manifest: dict[str, Any], runs: tuple[Run, ...], curves: list[reward_plot.RewardCurve]) -> Path:
    """Draw the old-style two-panel sample/wall-time learning plot."""
    svg = reward_plot._svg_tag
    rewards = [value for curve in curves for value in curve.rewards]
    y_min, y_max, y_ticks = reward_plot._nice_bounds([35.0, *rewards])
    canvas_height = 805 if len(runs) > 4 else 760
    root = ET.Element(
        svg("svg"),
        {
            "viewBox": f"0 0 1600 {canvas_height}",
            "width": "1600",
            "height": str(canvas_height),
            "role": "img",
            "aria-labelledby": "plot-title plot-description",
        },
    )
    display_name = str(manifest["display_name"])
    ET.SubElement(root, svg("title"), {"id": "plot-title"}).text = f"{display_name} matched learning curves"
    ET.SubElement(root, svg("desc"), {"id": "plot-description"}).text = (
        f"The same {len(runs)} 200-iteration-smoothed reward curves are plotted against environment "
        "transitions and elapsed hours."
    )
    ET.SubElement(
        root,
        svg("rect"),
        {"width": "1600", "height": str(canvas_height), "fill": "#FFFFFF"},
    )
    definitions = ET.SubElement(root, svg("defs"))
    reward_plot._add_text(
        root,
        800.0,
        43.0,
        f"{display_name}: matched no-TRS and TRS learning curves",
        size=24,
        anchor="middle",
        weight=600,
    )
    reward_plot._add_text(
        root,
        800.0,
        72.0,
        (
            "Train/mean_reward, 200-iteration trailing mean · seed 42 · "
            "512 environments · 24 steps/iteration · 20,000 iterations"
        ),
        size=13,
        anchor="middle",
        fill="#5F6368",
    )
    reward_plot._draw_panel(
        root,
        definitions,
        curves,
        panel_index=0,
        x_field="transitions_millions",
        title="Sample efficiency",
        x_label="Environment transitions [million]",
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
        curve_colors=COLORS[: len(runs)],
        curve_dash_arrays=(None,) * len(runs),
    )
    reward_plot._draw_panel(
        root,
        definitions,
        curves,
        panel_index=1,
        x_field="elapsed_hours",
        title="Observed wall-clock efficiency",
        x_label="Elapsed training time [h]",
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
        curve_colors=COLORS[: len(runs)],
        curve_dash_arrays=(None,) * len(runs),
    )
    legend_columns = min(len(runs), 4)
    legend_width = 380.0 if len(runs) > 4 else 360.0
    legend_start_y = 700.0 if len(runs) > 4 else 710.0
    start = (1600.0 - legend_width * legend_columns) / 2.0
    for index, run in enumerate(runs):
        row, column = divmod(index, legend_columns)
        x = start + column * legend_width
        y = legend_start_y + row * 38.0
        ET.SubElement(
            root,
            svg("line"),
            {
                "x1": f"{x:.2f}",
                "x2": f"{x + 45.0:.2f}",
                "y1": f"{y:.2f}",
                "y2": f"{y:.2f}",
                "stroke": COLORS[index],
                "stroke-width": "3.0" if index == 0 else "2.0",
                "stroke-linecap": "round",
            },
        )
        reward_plot._add_text(root, x + 55.0, y + 5.0, run.label, size=12 if len(runs) > 4 else 13)
    path = OUTPUT_DIR / "learning_curve_sample_and_wall_time.svg"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _write_balance_svg(manifest: dict[str, Any], runs: tuple[Run, ...], directional: list[dict[str, Any]]) -> Path:
    """Draw old-style grouped bars from the balanced fixed-grid summaries."""
    svg = reward_plot._svg_tag
    width = 2000.0 if len(runs) > 4 else 1600.0
    height = 850.0 if len(runs) > 4 else 780.0
    panel_height, panel_top = 470.0, 135.0
    panel_gap = 85.0
    panel_margin = 105.0
    panel_width = (width - 2.0 * panel_margin - panel_gap) / 2.0
    panel_lefts = (panel_margin, panel_margin + panel_width + panel_gap)
    y_min, y_max = (float(value) for value in manifest["balance_y_range"])
    y_ticks = [float(value) for value in manifest["balance_y_ticks"]]
    root = ET.Element(
        svg("svg"),
        {
            "viewBox": f"0 0 {width:.0f} {height:.0f}",
            "width": f"{width:.0f}",
            "height": f"{height:.0f}",
            "role": "img",
            "aria-labelledby": "balance-title balance-description",
        },
    )
    display_name = str(manifest["display_name"])
    ET.SubElement(
        root, svg("title"), {"id": "balance-title"}
    ).text = f"{display_name} fixed-grid front and hind load allocation by direction"
    ET.SubElement(
        root, svg("desc"), {"id": "balance-description"}
    ).text = "Grouped bars show family-balanced signed front-hind imbalance over three speeds and ten gaits."
    ET.SubElement(
        root,
        svg("rect"),
        {"width": f"{width:.0f}", "height": f"{height:.0f}", "fill": "#FFFFFF"},
    )
    reward_plot._add_text(
        root,
        width / 2.0,
        43.0,
        f"{display_name}: front/hind load allocation in fixed gait-speed grid",
        size=24,
        anchor="middle",
        weight=600,
    )
    reward_plot._add_text(
        root,
        width / 2.0,
        72.0,
        "Signed imbalance = 100 × (front − hind) / (front + hind); equal-family mean; zero is equal usage",
        size=13,
        anchor="middle",
        fill="#5F6368",
    )
    panel_titles = {
        "backward": "Backward grid: −0.5, −1.0, −1.5 m/s",
        "forward": "Forward grid: +0.5, +1.0, +1.5 m/s",
    }
    for panel_index, direction in enumerate(DIRECTIONS):
        left = panel_lefts[panel_index]
        reward_plot._add_text(
            root,
            left + panel_width / 2.0,
            112.0,
            panel_titles[direction],
            size=17,
            anchor="middle",
            weight=600,
        )
        for tick in y_ticks:
            y = panel_top + (y_max - tick) / (y_max - y_min) * panel_height
            ET.SubElement(
                root,
                svg("line"),
                {
                    "x1": f"{left:.2f}",
                    "x2": f"{left + panel_width:.2f}",
                    "y1": f"{y:.2f}",
                    "y2": f"{y:.2f}",
                    "stroke": "#9AA0A6" if tick == 0.0 else "#E2E5E9",
                    "stroke-width": "1.5" if tick == 0.0 else "1",
                },
            )
            reward_plot._add_text(root, left - 11.0, y + 5.0, f"{tick:+.0f}", size=12, anchor="end", fill="#5F6368")
        category_width = panel_width / len(METRICS)
        group_width = category_width * 0.72
        bar_width = group_width / len(runs)
        zero_y = panel_top + y_max / (y_max - y_min) * panel_height
        for metric_index, (metric, label, _unit) in enumerate(METRICS):
            center = left + (metric_index + 0.5) * category_width
            group_left = center - group_width / 2.0
            for run_index, run in enumerate(runs):
                row = next(
                    item
                    for item in directional
                    if item["run"] == run.folder
                    and item["direction"] == direction
                    and item["family"] == "all_family_balanced"
                    and item["metric"] == metric
                )
                value = float(row["signed_imbalance_percent"])
                value_y = panel_top + (y_max - value) / (y_max - y_min) * panel_height
                bar_y = min(value_y, zero_y)
                bar_height = max(abs(zero_y - value_y), 0.8)
                ET.SubElement(
                    root,
                    svg("rect"),
                    {
                        "x": f"{group_left + run_index * bar_width + 2.0:.2f}",
                        "y": f"{bar_y:.2f}",
                        "width": f"{bar_width - 4.0:.2f}",
                        "height": f"{bar_height:.2f}",
                        "fill": COLORS[run_index],
                        "fill-opacity": "0.88",
                    },
                )
                label_gap = 8.0 + 12.0 * (run_index % 2)
                label_y = value_y - label_gap if value >= 0.0 else value_y + label_gap + 9.0
                reward_plot._add_text(
                    root,
                    group_left + (run_index + 0.5) * bar_width,
                    label_y,
                    f"{value:+.1f}",
                    size=10,
                    anchor="middle",
                    fill="#3C4043",
                )
            reward_plot._add_text(
                root, center, panel_top + panel_height + 25.0, label, size=12, anchor="middle", fill="#3C4043"
            )
        if panel_index == 0:
            reward_plot._add_text(
                root,
                29.0,
                panel_top + panel_height / 2.0,
                "Signed front/hind imbalance [%]",
                size=14,
                anchor="middle",
                transform=f"rotate(-90 29.00 {panel_top + panel_height / 2.0:.2f})",
            )
    legend_columns = min(len(runs), 4)
    legend_width = 470.0 if len(runs) > 4 else 360.0
    legend_start_y = 689.0
    start = (width - legend_width * legend_columns) / 2.0
    for index, run in enumerate(runs):
        row, column = divmod(index, legend_columns)
        x = start + column * legend_width
        y = legend_start_y + row * 36.0
        ET.SubElement(
            root,
            svg("rect"),
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "width": "28",
                "height": "14",
                "fill": COLORS[index],
            },
        )
        reward_plot._add_text(root, x + 40.0, y + 13.0, run.label, size=12 if len(runs) > 4 else 13)
    reward_plot._add_text(
        root,
        width / 2.0,
        800.0 if len(runs) > 4 else 746.0,
        (
            "Each panel averages three speeds within each gait family, then weights "
            "trot, bound, half-bound, and gallop equally."
        ),
        size=12,
        anchor="middle",
        fill="#5F6368",
    )
    path = OUTPUT_DIR / "front_hind_signed_imbalance.svg"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _svg_dimensions(path: Path) -> tuple[int, int]:
    """Return the declared SVG canvas dimensions."""
    root = ET.parse(path).getroot()
    return round(float(root.attrib["width"])), round(float(root.attrib["height"]))


def _browser_executables() -> tuple[Path, ...]:
    """Return available Chromium-family executables."""
    candidates: list[Path] = []
    for executable in ("msedge", "google-chrome", "chromium"):
        if resolved := shutil.which(executable):
            candidates.append(Path(resolved))
    if os.name == "nt":
        for environment_name, relative_path in (
            ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe"),
            ("ProgramFiles", "Microsoft/Edge/Application/msedge.exe"),
            ("LOCALAPPDATA", "Microsoft/Edge/Application/msedge.exe"),
            ("ProgramFiles", "Google/Chrome/Application/chrome.exe"),
        ):
            if root := os.environ.get(environment_name):
                candidates.append(Path(root) / relative_path)
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate.is_file()))


def _render_svg_png(svg_path: Path) -> bool:
    """Render an SVG preview through a local headless browser when available."""
    png_path = svg_path.with_suffix(".png")
    png_path.unlink(missing_ok=True)
    width, height = _svg_dimensions(svg_path)
    for browser in _browser_executables():
        try:
            with tempfile.TemporaryDirectory(prefix="gait-family-render-") as profile_dir:
                subprocess.run(
                    [
                        str(browser),
                        "--headless=new",
                        "--disable-gpu",
                        *(["--no-sandbox"] if os.name == "nt" else []),
                        "--hide-scrollbars",
                        "--allow-file-access-from-files",
                        "--force-device-scale-factor=1",
                        f"--user-data-dir={profile_dir}",
                        f"--window-size={width},{height}",
                        f"--screenshot={png_path}",
                        svg_path.resolve().as_uri(),
                    ],
                    cwd=OUTPUT_DIR,
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
        except (OSError, subprocess.SubprocessError):
            continue
        if png_path.is_file() and png_path.stat().st_size > 0:
            return True
    return False


def _write_report(
    manifest: dict[str, Any],
    runs: tuple[Run, ...],
    training_rows: list[dict[str, Any]],
    overall_leg: dict[str, dict[str, dict[str, float]]],
    validation: dict[str, dict[str, Any]],
) -> None:
    """Write the human-readable comparison report."""
    display_name = str(manifest["display_name"])
    robot_short_name = "Go2" if manifest["robot"] == "go2" else manifest["robot"].upper()
    run_word = "run" if len(runs) == 1 else "runs"
    reproduction_path = str((OUTPUT_DIR / "reproduce.py").relative_to(REPO_ROOT)).replace("/", "\\")
    default_scope = f"This study compares the latest {len(runs)} archived {robot_short_name} {run_word}."
    scope_description = str(manifest.get("scope_description", default_scope))
    lines = [
        f"# {display_name} gait-family V2 analysis",
        "",
        f"{scope_description} The learning plot retains the earlier good-runs "
        "style. The leg-usage plot retains its two-direction grouped-bar style, but replaces the incidental "
        "`-0.567/+1.682 m/s` playback samples with the controlled 10-gait grid at `vx = ±{0.5, 1.0, 1.5} m/s`.",
        "",
        "![Learning curves](learning_curve_sample_and_wall_time.png)",
        "",
        "![Fixed-grid signed front/hind imbalance](front_hind_signed_imbalance.png)",
        "",
        "## Training efficiency",
        "",
        "| Run | AUC first 10k | AUC full | Last-1k reward | Sustained reward 35 [M transitions] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        transitions = row["reward_35_transitions"]
        threshold = "N/A" if transitions == "" else f"{float(transitions) / 1_000_000:.2f}"
        lines.append(
            f"| {row['label']} | {float(row['reward_auc_first_10000']):.3f} | "
            f"{float(row['reward_auc_full']):.3f} | {float(row['reward_last_1000_mean']):.3f} ± "
            f"{float(row['reward_last_1000_std']):.3f} | {threshold} |"
        )
    lines.extend(
        [
            "",
            "## Family-balanced fixed-grid leg usage",
            "",
            "Lower is better. Imbalance is the equal-family mean of each cell's absolute front/hind imbalance; "
            "exposure is the equal-family, equal-direction mean per actual directed metre. Exact values for every "
            "signed velocity and gait family are retained in `front_hind_command_summary.csv`.",
            "",
            "| Run | Torque² imbalance | Work imbalance | GRF imbalance | Torque² / m | Work [J/m] | GRF [N·s/m] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        values = overall_leg[run.slug]
        lines.append(
            f"| {run.label} | {values['torque_squared']['mean_absolute_imbalance_percent']:.3f}% | "
            f"{values['absolute_work']['mean_absolute_imbalance_percent']:.3f}% | "
            f"{values['vertical_grf_impulse']['mean_absolute_imbalance_percent']:.3f}% | "
            f"{values['torque_squared']['total_per_directed_m']:.3f} | "
            f"{values['absolute_work']['total_per_directed_m']:.3f} | "
            f"{values['vertical_grf_impulse']['total_per_directed_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity and interpretation",
            "",
            "| Run | Valid grid cells | Planar pass | Yaw + planar pass |",
            "|---|---:|---:|---:|",
        ]
    )
    for run in runs:
        item = validation[run.slug]
        lines.append(
            f"| {run.label} | 60/60 | {item['planar_tracking_cells']}/60 | {item['combined_tracking_cells']}/60 |"
        )
    lines.extend(
        [
            "",
            f"All {len(runs) * 60} cells are valid and pass the speed-dependent planar tracking threshold. "
            "The main allocation "
            "summary therefore uses the common 60-cell planar set; heading/yaw success is reported separately and "
            "is not silently used to select a different subset for each policy.",
            "",
            str(manifest["interpretation"]["conclusion"]),
            "",
            "Normalized torque is intentionally omitted because the recorder stored PhysX sentinel effort limits "
            "(`1e9 N·m`), not physical actuator limits. Rainflow fatigue was not generated by the fixed-grid analyzer; "
            "neither quantity is reconstructed or fabricated here. Results describe one training seed and one "
            "deterministic evaluation seed, so they are checkpoint comparisons rather than statistical method claims.",
            "",
            "## Reproduction",
            "",
            "From the repository root:",
            "",
            "```powershell",
            f".\\isaaclab.bat -p .\\{reproduction_path}",
            "```",
            "",
            (
                "The SVG outputs are always regenerated. PNG previews are regenerated when a local "
                "Chromium-family browser is available."
            ),
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_outputs(runs: tuple[Run, ...]) -> None:
    """Validate the generated artifact set and table cardinalities."""
    required = (
        "study.json",
        "analysis.py",
        "reproduce.py",
        "REPORT.md",
        "summary.json",
        "analysis_provenance.json",
        "training_efficiency.csv",
        "learning_curve_points.csv",
        "front_hind_metrics.csv",
        "front_hind_command_summary.csv",
        "front_hind_direction_summary.csv",
        "learning_curve_sample_and_wall_time.svg",
        "front_hind_signed_imbalance.svg",
    )
    missing = [name for name in required if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Generated analysis is incomplete: {missing}")
    with (OUTPUT_DIR / "training_efficiency.csv").open("r", encoding="utf-8", newline="") as stream:
        if len(list(csv.DictReader(stream))) != len(runs):
            raise ValueError(f"training_efficiency.csv does not contain {len(runs)} runs.")
    with (OUTPUT_DIR / "front_hind_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        if len(list(csv.DictReader(stream))) != len(runs) * 60 * len(METRICS):
            raise ValueError("front_hind_metrics.csv has an unexpected row count.")
    with (OUTPUT_DIR / "front_hind_direction_summary.csv").open("r", encoding="utf-8", newline="") as stream:
        expected = len(runs) * len(DIRECTIONS) * len(METRICS) * 5
        if len(list(csv.DictReader(stream))) != expected:
            raise ValueError("front_hind_direction_summary.csv has an unexpected row count.")
    with (OUTPUT_DIR / "front_hind_command_summary.csv").open("r", encoding="utf-8", newline="") as stream:
        expected = len(runs) * 6 * len(METRICS) * 5
        if len(list(csv.DictReader(stream))) != expected:
            raise ValueError("front_hind_command_summary.csv has an unexpected row count.")
    ET.parse(OUTPUT_DIR / "learning_curve_sample_and_wall_time.svg")
    ET.parse(OUTPUT_DIR / "front_hind_signed_imbalance.svg")


def main(manifest_path: Path | None = None) -> None:
    """Regenerate and validate the complete analysis folder."""
    global MANIFEST_PATH, OUTPUT_DIR
    if manifest_path is not None:
        MANIFEST_PATH = manifest_path.resolve()
        OUTPUT_DIR = MANIFEST_PATH.parent
    manifest, runs = _load_manifest()
    cells_by_run: dict[str, list[dict[str, Any]]] = {}
    validation: dict[str, dict[str, Any]] = {}
    protocol = None
    for run in runs:
        print(f"Validating fixed grid: {run.label}", flush=True)
        cells, protocol, run_validation = _load_grid(manifest, run, protocol)
        cells_by_run[run.slug] = cells
        validation[run.slug] = run_validation
    if len({item["grid_analyzer_sha256"] for item in validation.values()}) != 1:
        raise ValueError("Fixed-grid analyzer source differs across runs.")
    if len({item["grid_source_provenance_sha256"] for item in validation.values()}) != 1:
        raise ValueError("Fixed-grid recording source differs across runs.")

    detailed = _cell_metric_rows(runs, cells_by_run)
    directional = _aggregate_direction(manifest, runs, detailed)
    commands = _aggregate_commands(manifest, runs, detailed)
    overall_leg = _overall_leg_summary(runs, directional)
    _write_csv(OUTPUT_DIR / "front_hind_metrics.csv", detailed)
    _write_csv(OUTPUT_DIR / "front_hind_command_summary.csv", commands)
    _write_csv(OUTPUT_DIR / "front_hind_direction_summary.csv", directional)

    training_rows, curves = _training_rows(runs)
    _write_csv(OUTPUT_DIR / "training_efficiency.csv", training_rows)
    _write_learning_points(runs, curves)
    learning_svg = _write_learning_svg(manifest, runs, curves)
    balance_svg = _write_balance_svg(manifest, runs, directional)
    _render_svg_png(learning_svg)
    _render_svg_png(balance_svg)

    generated_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "analysis_method_version": METHOD_VERSION,
        "generated_at_utc": generated_at,
        "analysis_engine": str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/"),
        "analysis_engine_sha256": _sha256(Path(__file__)),
        "analysis_entrypoint": str((OUTPUT_DIR / "analysis.py").relative_to(REPO_ROOT)).replace("\\", "/"),
        "analysis_entrypoint_sha256": _sha256(OUTPUT_DIR / "analysis.py"),
        "study_manifest_sha256": _sha256(MANIFEST_PATH),
        "training_parser_sha256": _sha256(SCRIPT_DIR / "analyze_trs_grid.py"),
        "plot_helper_sha256": _sha256(SCRIPT_DIR / "plot_trs_tensorboard.py"),
        "runs": validation,
    }
    (OUTPUT_DIR / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "method": provenance,
        "aggregation": manifest["fixed_grid"],
        "runs": {
            run.slug: {
                "label": run.label,
                "folder": run.folder,
                "training": training_rows[index],
                "leg_usage": overall_leg[run.slug],
                "validation": validation[run.slug],
            }
            for index, run in enumerate(runs)
        },
        "directional_leg_usage": directional,
        "command_leg_usage": commands,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(manifest, runs, training_rows, overall_leg, validation)
    _validate_outputs(runs)
    print(f"{len(runs)}-run {manifest['display_name']} good-runs analysis validated.", flush=True)


if __name__ == "__main__":
    main()
