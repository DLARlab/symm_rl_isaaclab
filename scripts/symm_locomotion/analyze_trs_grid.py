# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze leg allocation, durability exposure, and training efficiency for the TRS grid.

The analysis intentionally uses only the Python standard library so it can run
with the lightweight Python selected by ``isaaclab.bat -p``. It reads the raw
``sim_data.npz`` rollout archives and TensorBoard event files directly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import statistics
import struct
import zipfile
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs/rsl_rl"
SAMPLES_PER_ITERATION = 512 * 24
WINDOW_START_S = 0.5
WINDOW_STOP_S = 9.5
CONTACT_FORCE_THRESHOLD_N = 1.0
NORMALIZED_TORQUE_DWELL_THRESHOLDS = (0.50, 0.75, 0.90, 0.99)
FATIGUE_SENSITIVITY_EXPONENTS = (3, 5)
BLOCK_DURATION_S = 1.0
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260727
EQUIVALENCE_MARGIN_PP = 5.0

COEFFICIENT_PAIRS = ((0.10, 0.05), (0.20, 0.10), (0.30, 0.15))
WARMUP_ITERATIONS = (10, 100, 500)
ROBOT_RUN_DIRS = {
    "go2": LOG_ROOT / "unitree_go2_symm_flat",
    "x1": LOG_ROOT / "dobot_x1_symm_flat",
}
BASELINE_RUN_NAMES = {
    "go2": "2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20",
    "x1": "2026-07-19_10-33-04_x1_no_trs_pitch0p35",
}
CURATED_BASELINE_EVALUATIONS = {
    "go2": LOG_ROOT / "good_runs/unitree_go2_symm_flat" / BASELINE_RUN_NAMES["go2"] / "plots/play/sim_data.npz",
    "x1": LOG_ROOT / "good_runs/dobot_x1_symm_flat" / BASELINE_RUN_NAMES["x1"] / "plots/play/sim_data.npz",
}
EFFORT_LIMITS = {
    "go2": (23.7, 23.7, 35.55) * 4,
    "x1": (17.0, 17.0, 37.0, 17.0, 17.0, 37.0, 17.0, 20.0, 37.0, 17.0, 20.0, 37.0),
}
LEG_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
PAIR_BY_LEG = {
    "front_left": "front",
    "front_right": "front",
    "rear_left": "hind",
    "rear_right": "hind",
}
GRID_RUN_PATTERN = re.compile(
    r"_(?P<robot>go2|x1)_trs_m(?P<mirror>\d+p\d+)_v(?P<value>\d+p\d+)_w(?P<warmup>\d+)(?:_|$)"
)

PRIMARY_LEG_METRICS = ("torque_squared", "absolute_work", "vertical_grf_impulse")
ADDITIONAL_LEG_METRICS = (
    "normalized_torque_utilization",
    "positive_work",
    "negative_work",
    "grf_impulse",
    "contact_time",
)
METRIC_UNITS = {
    "torque_squared": ("N^2 m^2", "N^2 m^2 s"),
    "normalized_torque_utilization": ("1", "s"),
    "absolute_work": ("W", "J"),
    "positive_work": ("W", "J"),
    "negative_work": ("W", "J"),
    "grf_impulse": ("N", "N s"),
    "vertical_grf_impulse": ("N", "N s"),
    "contact_time": ("feet", "foot s"),
}

DURABILITY_PAIR_METRICS = {
    "torque_squared_exposure": {
        "mechanism": "actuator_load",
        "unit": "N^2 m^2 s",
        "per_distance_unit": "N^2 m^2 s/m",
    },
    "normalized_torque_capacity_exposure": {
        "mechanism": "actuator_capacity",
        "unit": "s",
        "per_distance_unit": "s/m",
    },
    "absolute_work": {
        "mechanism": "mechanical_work",
        "unit": "J",
        "per_distance_unit": "J/m",
    },
    "positive_work": {
        "mechanism": "mechanical_work",
        "unit": "J",
        "per_distance_unit": "J/m",
    },
    "negative_work": {
        "mechanism": "mechanical_work",
        "unit": "J",
        "per_distance_unit": "J/m",
    },
    "fatigue_proxy_m3": {
        "mechanism": "torque_cycle_fatigue_sensitivity",
        "unit": "1",
        "per_distance_unit": "1/m",
    },
    "fatigue_proxy_m5": {
        "mechanism": "torque_cycle_fatigue_sensitivity",
        "unit": "1",
        "per_distance_unit": "1/m",
    },
    "vertical_grf_impulse": {
        "mechanism": "support_load",
        "unit": "N s",
        "per_distance_unit": "N s/m",
    },
}


@dataclass(frozen=True)
class RunSpec:
    """Paths and TRS parameters for one training run."""

    robot: str
    run_path: Path
    evaluation_path: Path
    mirror_coeff: float
    value_coeff: float
    warmup_iterations: int | None
    trs_enabled: bool

    @property
    def key(self) -> str:
        """Return a stable key for machine-readable outputs."""
        if not self.trs_enabled:
            return f"{self.robot}_no_trs"
        return (
            f"{self.robot}_m{decimal_slug(self.mirror_coeff)}"
            f"_v{decimal_slug(self.value_coeff)}_w{self.warmup_iterations}"
        )

    @property
    def condition(self) -> str:
        """Return a compact condition label."""
        if not self.trs_enabled:
            return "No TRS"
        return f"{self.mirror_coeff:.2f}/{self.value_coeff:.2f}, w={self.warmup_iterations}"


def decimal_slug(value: float) -> str:
    """Return a filename-safe fixed-point decimal."""
    return f"{value:.2f}".replace(".", "p")


def parse_decimal_slug(value: str) -> float:
    """Parse a decimal whose point is represented by ``p``."""
    return float(value.replace("p", "."))


def discover_runs() -> list[RunSpec]:
    """Discover the two baselines and all expected TRS grid cells."""
    runs: list[RunSpec] = []
    expected_cells = {
        (robot, mirror, value, warmup)
        for robot in ROBOT_RUN_DIRS
        for mirror, value in COEFFICIENT_PAIRS
        for warmup in WARMUP_ITERATIONS
    }
    discovered_cells: dict[tuple[str, float, float, int], Path] = {}
    for robot, run_root in ROBOT_RUN_DIRS.items():
        baseline_path = run_root / BASELINE_RUN_NAMES[robot]
        evaluation_path = CURATED_BASELINE_EVALUATIONS[robot]
        if not evaluation_path.is_file():
            evaluation_path = baseline_path / "plots/play/sim_data.npz"
        runs.append(
            RunSpec(
                robot=robot,
                run_path=baseline_path,
                evaluation_path=evaluation_path,
                mirror_coeff=0.0,
                value_coeff=0.0,
                warmup_iterations=None,
                trs_enabled=False,
            )
        )
        for candidate in run_root.iterdir():
            if not candidate.is_dir():
                continue
            match = GRID_RUN_PATTERN.search(candidate.name)
            if match is None or match.group("robot") != robot:
                continue
            cell = (
                robot,
                parse_decimal_slug(match.group("mirror")),
                parse_decimal_slug(match.group("value")),
                int(match.group("warmup")),
            )
            if cell not in expected_cells:
                continue
            if cell in discovered_cells:
                raise ValueError(
                    f"Multiple run directories match grid cell {cell}: {discovered_cells[cell]}, {candidate}"
                )
            discovered_cells[cell] = candidate

    missing_cells = sorted(expected_cells - set(discovered_cells))
    if missing_cells:
        raise ValueError(f"Missing TRS grid cells: {missing_cells}")
    for (robot, mirror, value, warmup), run_path in sorted(discovered_cells.items()):
        curated_evaluation = (
            LOG_ROOT / "good_runs" / ROBOT_RUN_DIRS[robot].name / run_path.name / "plots/play/sim_data.npz"
        )
        evaluation_path = curated_evaluation if curated_evaluation.is_file() else run_path / "plots/play/sim_data.npz"
        runs.append(
            RunSpec(
                robot=robot,
                run_path=run_path,
                evaluation_path=evaluation_path,
                mirror_coeff=mirror,
                value_coeff=value,
                warmup_iterations=warmup,
                trs_enabled=True,
            )
        )

    for run in runs:
        if not run.run_path.is_dir():
            raise FileNotFoundError(run.run_path)
        if not run.evaluation_path.is_file():
            raise FileNotFoundError(run.evaluation_path)
        event_files = list(run.run_path.glob("events.out.tfevents.*"))
        if len(event_files) != 1:
            raise ValueError(f"Expected one TensorBoard event file in {run.run_path}, found {len(event_files)}")
    return sorted(
        runs,
        key=lambda run: (
            run.robot,
            run.trs_enabled,
            run.mirror_coeff,
            run.value_coeff,
            run.warmup_iterations or -1,
        ),
    )


def read_npy_member(archive: zipfile.ZipFile, name: str) -> tuple[list[Any], tuple[int, ...]]:
    """Read one C-order numeric, boolean, or string array from an NPZ archive."""
    with archive.open(f"{name}.npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"{name} does not have the NPY magic header")
        major, _minor = stream.read(2)
        header_size_format = "<H" if major == 1 else "<I"
        header_size = struct.unpack(header_size_format, stream.read(struct.calcsize(header_size_format)))[0]
        header = ast.literal_eval(stream.read(header_size).decode("latin1"))
        if header["fortran_order"]:
            raise ValueError(f"Fortran-order arrays are unsupported: {name}")
        dtype = header["descr"]
        shape = tuple(header["shape"])
        count = math.prod(shape) if shape else 1
        raw = stream.read()

    numeric_formats = {
        "<f4": "f",
        "<f8": "d",
        "<i4": "i",
        "<i8": "q",
        "<u4": "I",
        "<u8": "Q",
        "|b1": "?",
    }
    if dtype in numeric_formats:
        item_format = numeric_formats[dtype]
        item_size = struct.calcsize("<" + item_format)
        if len(raw) != count * item_size:
            raise ValueError(f"Unexpected data size for {name}: {len(raw)}")
        return [item[0] for item in struct.iter_unpack("<" + item_format, raw)], shape

    unicode_match = re.fullmatch(r"<U(\d+)", dtype)
    if unicode_match:
        characters = int(unicode_match.group(1))
        item_size = 4 * characters
        if len(raw) != count * item_size:
            raise ValueError(f"Unexpected Unicode data size for {name}: {len(raw)}")
        values = [
            raw[offset : offset + item_size].decode("utf-32-le").rstrip("\x00")
            for offset in range(0, len(raw), item_size)
        ]
        return values, shape

    bytes_match = re.fullmatch(r"\|S(\d+)", dtype)
    if bytes_match:
        item_size = int(bytes_match.group(1))
        if len(raw) != count * item_size:
            raise ValueError(f"Unexpected byte-string data size for {name}: {len(raw)}")
        values = [
            raw[offset : offset + item_size].rstrip(b"\x00").decode("utf-8", errors="replace")
            for offset in range(0, len(raw), item_size)
        ]
        return values, shape
    raise ValueError(f"Unsupported dtype {dtype!r} for {name}")


def reshape_rows(values: Sequence[Any], row_width: int) -> list[list[float]]:
    """Reshape a flat numeric sequence into rows."""
    return [[float(value) for value in values[start : start + row_width]] for start in range(0, len(values), row_width)]


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated sample percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def extract_reversals(values: Sequence[float]) -> list[float]:
    """Return endpoints and turning points after removing consecutive plateaus."""
    if not values:
        return []
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Rainflow input contains a non-finite value")
    if len(values) == 1:
        return [float(values[0])]

    reversals = [float(values[0])]
    previous = float(values[0])
    current = float(values[1])
    previous_slope = current - previous
    for next_value_raw in values[2:]:
        next_value = float(next_value_raw)
        if next_value == current:
            continue
        next_slope = next_value - current
        if previous_slope * next_slope < 0.0:
            reversals.append(current)
        previous = current
        current = next_value
        previous_slope = next_slope
    if current != reversals[-1]:
        reversals.append(current)
    return reversals


def rainflow_cycles(values: Sequence[float]) -> list[tuple[float, float, float]]:
    """Count ASTM E1049-style rainflow cycles as ``(range, mean, count)``.

    Residual ranges at the finite record boundaries count as half cycles.
    Zero-range cycles are omitted.
    """
    points: deque[float] = deque()
    cycles: list[tuple[float, float, float]] = []
    for point in extract_reversals(values):
        points.append(point)
        while len(points) >= 3:
            older_range = abs(points[-2] - points[-3])
            newer_range = abs(points[-1] - points[-2])
            if newer_range < older_range:
                break
            mean = 0.5 * (points[-3] + points[-2])
            if len(points) == 3:
                if older_range > 0.0:
                    cycles.append((older_range, mean, 0.5))
                points.popleft()
            else:
                if older_range > 0.0:
                    cycles.append((older_range, mean, 1.0))
                newest = points.pop()
                points.pop()
                points.pop()
                points.append(newest)
    while len(points) > 1:
        cycle_range = abs(points[1] - points[0])
        if cycle_range > 0.0:
            cycles.append((cycle_range, 0.5 * (points[0] + points[1]), 0.5))
        points.popleft()
    return cycles


def fatigue_proxy(
    cycles: Sequence[tuple[float, float, float]],
    effort_limit_nm: float,
    exponent: int,
) -> float:
    """Return a dimensionless torque-cycle fatigue sensitivity proxy.

    The proxy sums ``count * (cycle_amplitude / effort_limit)^exponent``.
    Effort limit is only a common normalization scale, not a fatigue strength.
    """
    if effort_limit_nm <= 0.0:
        raise ValueError("Effort limit must be positive")
    if exponent <= 0:
        raise ValueError("Fatigue sensitivity exponent must be positive")
    return sum(count * (0.5 * cycle_range / effort_limit_nm) ** exponent for cycle_range, _mean, count in cycles)


def fatigue_moment(cycles: Sequence[tuple[float, float, float]], exponent: int) -> float:
    """Return the exposure-preserving cycle-amplitude moment [(N m)^m]."""
    if exponent <= 0:
        raise ValueError("Fatigue sensitivity exponent must be positive")
    return sum(count * (0.5 * cycle_range) ** exponent for cycle_range, _mean, count in cycles)


def longest_threshold_dwell_s(values: Sequence[float], threshold: float, dt_s: float) -> float:
    """Return the longest contiguous time at or above ``threshold`` [s]."""
    longest_samples = 0
    current_samples = 0
    for value in values:
        if value >= threshold:
            current_samples += 1
            longest_samples = max(longest_samples, current_samples)
        else:
            current_samples = 0
    return longest_samples * dt_s


def pair_concentration(front: float, hind: float) -> tuple[str, float]:
    """Return the dominant pair and its exposure ratio to the other pair."""
    if front < 0.0 or hind < 0.0:
        raise ValueError("Pair exposures must be non-negative")
    dominant_pair = "front" if front >= hind else "hind"
    dominant = max(front, hind)
    other = min(front, hind)
    if dominant == 0.0:
        return "none", 1.0
    if other == 0.0:
        return dominant_pair, float("inf")
    return dominant_pair, dominant / other


def summarize_pair_totals(front: float, hind: float, distance_m: float) -> dict[str, float | str]:
    """Summarize total, per-distance, and concentration for a pair-additive metric."""
    if distance_m <= 0.0:
        raise ValueError("Distance must be positive")
    dominant_pair, dominant_ratio = pair_concentration(front, hind)
    signed_imbalance = pair_imbalance(front, hind)
    return {
        "front": front,
        "hind": hind,
        "total": front + hind,
        "front_per_m": front / distance_m,
        "hind_per_m": hind / distance_m,
        "total_per_m": (front + hind) / distance_m,
        "front_share_percent": 100.0 * front / (front + hind),
        "signed_imbalance_percent": signed_imbalance,
        "abs_imbalance_percent": abs(signed_imbalance),
        "dominant_pair": dominant_pair,
        "dominant_to_other_ratio": dominant_ratio,
    }


def analyze_durability_exposure(
    *,
    joint_names: Sequence[str],
    effort_limits_nm: Sequence[float],
    torque_rows: Sequence[Sequence[float]],
    power_rows: Sequence[Sequence[float]],
    force_norm_traces: Sequence[Sequence[float]],
    vertical_force_traces: Sequence[Sequence[float]],
    dt_s: float,
    duration_s: float,
    directed_distance_m: float,
) -> dict[str, Any]:
    """Compute per-component durability exposure without hiding local maxima."""
    if len(joint_names) != 12 or len(effort_limits_nm) != 12:
        raise ValueError("Durability analysis expects 12 leg-major joints")
    if len(force_norm_traces) != 4 or len(vertical_force_traces) != 4:
        raise ValueError("Durability analysis expects four foot-force traces")

    joint_results: list[dict[str, Any]] = []
    for joint_index, (joint_name, effort_limit_nm) in enumerate(zip(joint_names, effort_limits_nm, strict=True)):
        torque_trace = [float(row[joint_index]) for row in torque_rows]
        power_trace = [float(row[joint_index]) for row in power_rows]
        normalized_torque = [abs(torque) / effort_limit_nm for torque in torque_trace]
        cycles = rainflow_cycles(torque_trace)
        equivalent_cycle_count = sum(count for _range, _mean, count in cycles)
        result: dict[str, Any] = {
            "joint_index": joint_index,
            "joint_name": joint_name,
            "leg": LEG_NAMES[joint_index // 3],
            "pair": PAIR_BY_LEG[LEG_NAMES[joint_index // 3]],
            "effort_limit_nm": effort_limit_nm,
            "torque_rms_nm": math.sqrt(statistics.fmean(torque * torque for torque in torque_trace)),
            "torque_abs_p95_nm": percentile([abs(value) for value in torque_trace], 0.95),
            "torque_abs_p99_nm": percentile([abs(value) for value in torque_trace], 0.99),
            "torque_abs_peak_nm": max(abs(value) for value in torque_trace),
            "normalized_torque_rms": math.sqrt(statistics.fmean(value * value for value in normalized_torque)),
            "normalized_torque_abs_p95": percentile(normalized_torque, 0.95),
            "normalized_torque_abs_p99": percentile(normalized_torque, 0.99),
            "normalized_torque_abs_peak": max(normalized_torque),
            "torque_squared_exposure_n2m2s": sum(torque * torque for torque in torque_trace) * dt_s,
            "normalized_torque_capacity_exposure_s": (sum(value * value for value in normalized_torque) * dt_s),
            "absolute_work_j": sum(abs(power) for power in power_trace) * dt_s,
            "positive_work_j": sum(max(power, 0.0) for power in power_trace) * dt_s,
            "negative_work_j": sum(max(-power, 0.0) for power in power_trace) * dt_s,
            "absolute_power_p99_w": percentile([abs(power) for power in power_trace], 0.99),
            "absolute_power_peak_w": max(abs(power) for power in power_trace),
            "rainflow_equivalent_cycle_count": equivalent_cycle_count,
            "rainflow_cycle_count_per_s": equivalent_cycle_count / duration_s,
            "rainflow_cycle_count_per_m": equivalent_cycle_count / directed_distance_m,
            "rainflow_range_max_nm": max((cycle[0] for cycle in cycles), default=0.0),
            "rainflow_normalized_amplitude_max": max(
                (0.5 * cycle[0] / effort_limit_nm for cycle in cycles),
                default=0.0,
            ),
            "rainflow_spectrum": cycles,
        }
        for threshold in NORMALIZED_TORQUE_DWELL_THRESHOLDS:
            threshold_slug = f"{threshold:.2f}".replace(".", "p")
            dwell_samples = sum(value >= threshold for value in normalized_torque)
            result[f"dwell_fraction_at_or_above_{threshold_slug}"] = dwell_samples / len(normalized_torque)
            result[f"dwell_time_s_at_or_above_{threshold_slug}"] = dwell_samples * dt_s
            result[f"longest_dwell_s_at_or_above_{threshold_slug}"] = longest_threshold_dwell_s(
                normalized_torque,
                threshold,
                dt_s,
            )
        for exponent in FATIGUE_SENSITIVITY_EXPONENTS:
            moment = fatigue_moment(cycles, exponent)
            normalized_severity = fatigue_proxy(cycles, effort_limit_nm, exponent)
            result[f"torque_cycle_moment_m{exponent}"] = moment
            result[f"torque_cycle_moment_m{exponent}_per_s"] = moment / duration_s
            result[f"torque_cycle_moment_m{exponent}_per_m"] = moment / directed_distance_m
            result[f"fatigue_proxy_m{exponent}"] = normalized_severity
            result[f"fatigue_proxy_m{exponent}_per_s"] = normalized_severity / duration_s
            result[f"fatigue_proxy_m{exponent}_per_m"] = normalized_severity / directed_distance_m
            result[f"cycle_average_equivalent_amplitude_m{exponent}_nm"] = (
                (moment / equivalent_cycle_count) ** (1.0 / exponent) if equivalent_cycle_count > 0.0 else 0.0
            )
        for field in (
            "torque_squared_exposure_n2m2s",
            "normalized_torque_capacity_exposure_s",
            "absolute_work_j",
            "positive_work_j",
            "negative_work_j",
        ):
            result[f"{field}_per_s"] = result[field] / duration_s
            result[f"{field}_per_m"] = result[field] / directed_distance_m
        joint_results.append(result)

    leg_results: list[dict[str, Any]] = []
    for leg_index, leg_name in enumerate(LEG_NAMES):
        leg_joints = joint_results[3 * leg_index : 3 * leg_index + 3]
        force_norm_trace = list(force_norm_traces[leg_index])
        vertical_force_trace = list(vertical_force_traces[leg_index])
        positive_loading_rates = [
            max((right - left) / dt_s, 0.0)
            for left, right in zip(vertical_force_trace, vertical_force_trace[1:], strict=False)
        ]
        contact_samples = sum(force > CONTACT_FORCE_THRESHOLD_N for force in force_norm_trace)
        leg_result: dict[str, Any] = {
            "leg_index": leg_index,
            "leg": leg_name,
            "pair": PAIR_BY_LEG[leg_name],
            "torque_squared_exposure_n2m2s": sum(joint["torque_squared_exposure_n2m2s"] for joint in leg_joints),
            "normalized_torque_capacity_exposure_s": sum(
                joint["normalized_torque_capacity_exposure_s"] for joint in leg_joints
            ),
            "absolute_work_j": sum(joint["absolute_work_j"] for joint in leg_joints),
            "positive_work_j": sum(joint["positive_work_j"] for joint in leg_joints),
            "negative_work_j": sum(joint["negative_work_j"] for joint in leg_joints),
            "vertical_grf_impulse_ns": sum(vertical_force_trace) * dt_s,
            "grf_impulse_ns": sum(force_norm_trace) * dt_s,
            "contact_duty_factor": contact_samples / len(force_norm_trace),
            "vertical_force_contact_mean_n": (
                sum(
                    vertical_force
                    for vertical_force, force_norm in zip(
                        vertical_force_trace,
                        force_norm_trace,
                        strict=True,
                    )
                    if force_norm > CONTACT_FORCE_THRESHOLD_N
                )
                / contact_samples
                if contact_samples
                else 0.0
            ),
            "vertical_force_p95_n": percentile(vertical_force_trace, 0.95),
            "vertical_force_p99_n": percentile(vertical_force_trace, 0.99),
            "vertical_force_peak_n": max(vertical_force_trace),
            "positive_vertical_loading_rate_p95_nps": percentile(positive_loading_rates, 0.95),
            "positive_vertical_loading_rate_p99_nps": percentile(positive_loading_rates, 0.99),
            "positive_vertical_loading_rate_peak_nps": max(positive_loading_rates, default=0.0),
        }
        for exponent in FATIGUE_SENSITIVITY_EXPONENTS:
            leg_result[f"fatigue_proxy_m{exponent}"] = sum(joint[f"fatigue_proxy_m{exponent}"] for joint in leg_joints)
            leg_result[f"torque_cycle_moment_m{exponent}"] = sum(
                joint[f"torque_cycle_moment_m{exponent}"] for joint in leg_joints
            )
        for field in (
            "torque_squared_exposure_n2m2s",
            "normalized_torque_capacity_exposure_s",
            "absolute_work_j",
            "positive_work_j",
            "negative_work_j",
            "vertical_grf_impulse_ns",
            "grf_impulse_ns",
            "fatigue_proxy_m3",
            "fatigue_proxy_m5",
            "torque_cycle_moment_m3",
            "torque_cycle_moment_m5",
        ):
            leg_result[f"{field}_per_s"] = leg_result[field] / duration_s
            leg_result[f"{field}_per_m"] = leg_result[field] / directed_distance_m
        leg_results.append(leg_result)

    pair_field_names = {
        "torque_squared_exposure": "torque_squared_exposure_n2m2s",
        "normalized_torque_capacity_exposure": "normalized_torque_capacity_exposure_s",
        "absolute_work": "absolute_work_j",
        "positive_work": "positive_work_j",
        "negative_work": "negative_work_j",
        "fatigue_proxy_m3": "fatigue_proxy_m3",
        "fatigue_proxy_m5": "fatigue_proxy_m5",
        "vertical_grf_impulse": "vertical_grf_impulse_ns",
    }
    pair_metrics = {}
    for metric_name, field_name in pair_field_names.items():
        front = sum(leg[field_name] for leg in leg_results[:2])
        hind = sum(leg[field_name] for leg in leg_results[2:])
        pair_summary = summarize_pair_totals(front, hind, directed_distance_m)
        pair_summary.update(
            {
                "front_per_s": front / duration_s,
                "hind_per_s": hind / duration_s,
                "total_per_s": (front + hind) / duration_s,
            }
        )
        pair_metrics[metric_name] = pair_summary

    def worst_record(
        records: Sequence[dict[str, Any]],
        field_name: str,
        identity_field: str,
    ) -> dict[str, Any]:
        record = max(records, key=lambda item: item[field_name])
        return {
            "component": record[identity_field],
            "leg": record["leg"],
            "pair": record["pair"],
            "value": record[field_name],
            "field": field_name,
        }

    worst_components = {
        "joint_normalized_torque_rms": worst_record(
            joint_results,
            "normalized_torque_rms",
            "joint_name",
        ),
        "joint_normalized_torque_p99": worst_record(
            joint_results,
            "normalized_torque_abs_p99",
            "joint_name",
        ),
        "joint_normalized_torque_peak": worst_record(
            joint_results,
            "normalized_torque_abs_peak",
            "joint_name",
        ),
        "joint_torque_squared_exposure_per_m": worst_record(
            joint_results,
            "torque_squared_exposure_n2m2s_per_m",
            "joint_name",
        ),
        "joint_fatigue_proxy_m3_per_m": worst_record(
            joint_results,
            "fatigue_proxy_m3_per_m",
            "joint_name",
        ),
        "joint_fatigue_proxy_m5_per_m": worst_record(
            joint_results,
            "fatigue_proxy_m5_per_m",
            "joint_name",
        ),
        "joint_absolute_work_per_m": worst_record(
            joint_results,
            "absolute_work_j_per_m",
            "joint_name",
        ),
        "foot_vertical_force_p99": worst_record(
            leg_results,
            "vertical_force_p99_n",
            "leg",
        ),
        "foot_vertical_force_peak": worst_record(
            leg_results,
            "vertical_force_peak_n",
            "leg",
        ),
        "foot_vertical_loading_rate_p99": worst_record(
            leg_results,
            "positive_vertical_loading_rate_p99_nps",
            "leg",
        ),
        "foot_vertical_loading_rate_peak": worst_record(
            leg_results,
            "positive_vertical_loading_rate_peak_nps",
            "leg",
        ),
    }
    return {
        "joints": joint_results,
        "legs": leg_results,
        "pair_metrics": pair_metrics,
        "worst_components": worst_components,
    }


def pair_imbalance(front: float, hind: float) -> float:
    """Return signed front/hind imbalance in percent."""
    denominator = front + hind
    if denominator <= 0.0:
        return float("nan")
    return 100.0 * (front - hind) / denominator


def block_bootstrap_imbalance(
    sample_pairs: Sequence[tuple[float, float]],
    block_samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Bootstrap pair imbalance by resampling non-overlapping temporal blocks."""
    blocks = [
        (
            sum(pair[0] for pair in sample_pairs[start : start + block_samples]),
            sum(pair[1] for pair in sample_pairs[start : start + block_samples]),
        )
        for start in range(0, len(sample_pairs), block_samples)
        if len(sample_pairs[start : start + block_samples]) == block_samples
    ]
    if len(blocks) < 2:
        return float("nan"), float("nan")
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        front = 0.0
        hind = 0.0
        for _block_index in range(len(blocks)):
            block_front, block_hind = blocks[rng.randrange(len(blocks))]
            front += block_front
            hind += block_hind
        estimates.append(pair_imbalance(front, hind))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_additive_metric(
    sample_pairs: Sequence[tuple[float, float]],
    *,
    dt_s: float,
    block_samples: int,
    rng: random.Random,
) -> dict[str, float | list[float]]:
    """Summarize one additive front/hind metric."""
    front_total = sum(pair[0] for pair in sample_pairs)
    hind_total = sum(pair[1] for pair in sample_pairs)
    signed_imbalance = pair_imbalance(front_total, hind_total)
    ci_low, ci_high = block_bootstrap_imbalance(sample_pairs, block_samples, rng)
    front_samples = [pair[0] for pair in sample_pairs]
    hind_samples = [pair[1] for pair in sample_pairs]
    return {
        "front_sample_mean": statistics.fmean(front_samples),
        "hind_sample_mean": statistics.fmean(hind_samples),
        "front_integral": front_total * dt_s,
        "hind_integral": hind_total * dt_s,
        "front_share_percent": 100.0 * front_total / (front_total + hind_total),
        "signed_imbalance_percent": signed_imbalance,
        "abs_imbalance_percent": abs(signed_imbalance),
        "bootstrap_95_percent": [ci_low, ci_high],
        "front_p95": percentile(front_samples, 0.95),
        "hind_p95": percentile(hind_samples, 0.95),
        "front_p99": percentile(front_samples, 0.99),
        "hind_p99": percentile(hind_samples, 0.99),
    }


def analyze_rollout(run: RunSpec) -> dict[str, Any]:
    """Analyze a matched steady-command window in one raw rollout."""
    required_arrays = (
        "time_steps",
        "desired_lin_vel",
        "true_lin_vel",
        "base_positions",
        "joint_names",
        "joint_torques",
        "joint_powers",
        "foot_ground_reaction_forces_w",
        "foot_forces",
        "episode_done",
        "ground_reaction_force_includes_friction",
    )
    loaded: dict[str, tuple[list[Any], tuple[int, ...]]] = {}
    with zipfile.ZipFile(run.evaluation_path) as archive:
        archive_names = set(archive.namelist())
        for name in required_arrays:
            if f"{name}.npy" not in archive_names:
                raise ValueError(f"{run.evaluation_path} does not contain {name}.npy")
            loaded[name] = read_npy_member(archive, name)
        if "joint_velocities.npy" in archive_names:
            loaded["joint_velocities"] = read_npy_member(archive, "joint_velocities")

    times_flat, time_shape = loaded["time_steps"]
    sample_count = time_shape[0]
    expected_shapes = {
        "desired_lin_vel": (sample_count, 3),
        "true_lin_vel": (sample_count, 3),
        "base_positions": (sample_count, 2),
        "joint_names": (12,),
        "joint_torques": (sample_count, 12),
        "joint_powers": (sample_count, 12),
        "foot_ground_reaction_forces_w": (sample_count, 4, 3),
        "foot_forces": (sample_count, 4),
        "episode_done": (sample_count,),
        "ground_reaction_force_includes_friction": (sample_count,),
    }
    for name, expected_shape in expected_shapes.items():
        if loaded[name][1] != expected_shape:
            raise ValueError(f"Unexpected {name} shape in {run.evaluation_path}: {loaded[name][1]}")
    if "joint_velocities" in loaded and loaded["joint_velocities"][1] != (sample_count, 12):
        raise ValueError(f"Unexpected joint_velocities shape in {run.evaluation_path}: {loaded['joint_velocities'][1]}")

    times = [float(value) for value in times_flat]
    if len(times) < 2:
        raise ValueError(f"Rollout has fewer than two samples: {run.evaluation_path}")
    dt_s = statistics.median(right - left for left, right in zip(times, times[1:], strict=False))
    block_samples = round(BLOCK_DURATION_S / dt_s)
    selected = [index for index, time_s in enumerate(times) if WINDOW_START_S <= time_s < WINDOW_STOP_S]
    expected_window_samples = round((WINDOW_STOP_S - WINDOW_START_S) / dt_s)
    if len(selected) != expected_window_samples:
        raise ValueError(
            f"Expected {expected_window_samples} common-window samples, found {len(selected)} in {run.evaluation_path}"
        )

    done = [bool(value) for value in loaded["episode_done"][0]]
    if any(done[index] for index in selected):
        raise ValueError(f"Episode ended inside the common analysis window in {run.evaluation_path}")
    includes_friction = [bool(value) for value in loaded["ground_reaction_force_includes_friction"][0]]
    if not all(includes_friction[index] for index in selected):
        raise ValueError(f"GRF samples exclude friction inside the common window in {run.evaluation_path}")

    commands = reshape_rows(loaded["desired_lin_vel"][0], 3)
    true_velocities = reshape_rows(loaded["true_lin_vel"][0], 3)
    base_positions = reshape_rows(loaded["base_positions"][0], 2)
    joint_velocities = None if "joint_velocities" not in loaded else reshape_rows(loaded["joint_velocities"][0], 12)
    joint_torques = reshape_rows(loaded["joint_torques"][0], 12)
    joint_powers = reshape_rows(loaded["joint_powers"][0], 12)
    forces = reshape_rows(loaded["foot_ground_reaction_forces_w"][0], 12)
    recorded_force_norms = reshape_rows(loaded["foot_forces"][0], 4)
    joint_names = [str(value) for value in loaded["joint_names"][0]]
    effort_limits = EFFORT_LIMITS[run.robot]

    expected_leg_prefixes = (
        ("FL", "FR", "RL", "RR") if run.robot == "go2" else ("front_left", "front_right", "rear_left", "rear_right")
    )
    for leg_index, prefix in enumerate(expected_leg_prefixes):
        leg_joint_names = joint_names[3 * leg_index : 3 * leg_index + 3]
        if not all(prefix.lower() in name.lower() for name in leg_joint_names):
            raise ValueError(f"Joint order is not leg-major ({prefix}) in {run.evaluation_path}: {leg_joint_names}")

    metric_pairs: dict[str, list[tuple[float, float]]] = {
        name: []
        for name in (
            "torque_squared",
            "normalized_torque_utilization",
            "absolute_work",
            "positive_work",
            "negative_work",
            "grf_impulse",
            "vertical_grf_impulse",
            "contact_time",
        )
    }
    force_norm_traces: list[list[float]] = [[] for _ in LEG_NAMES]
    vertical_force_traces: list[list[float]] = [[] for _ in LEG_NAMES]
    max_power_identity_error = 0.0 if joint_velocities is not None else None
    max_force_norm_identity_error = 0.0
    negative_vertical_force_samples = 0
    nonzero_force_samples = 0
    for sample_index in selected:
        per_leg: dict[str, list[float]] = {name: [] for name in metric_pairs}
        sample_torques = joint_torques[sample_index]
        sample_velocities = None if joint_velocities is None else joint_velocities[sample_index]
        sample_powers = joint_powers[sample_index]
        sample_forces = forces[sample_index]
        if sample_velocities is not None:
            for torque, velocity, power in zip(sample_torques, sample_velocities, sample_powers, strict=True):
                max_power_identity_error = max(max_power_identity_error, abs(power - torque * velocity))
        for leg_index in range(4):
            joint_start = 3 * leg_index
            leg_torques = sample_torques[joint_start : joint_start + 3]
            leg_powers = sample_powers[joint_start : joint_start + 3]
            force_start = 3 * leg_index
            force_xyz = sample_forces[force_start : force_start + 3]
            force_norm = math.sqrt(sum(component * component for component in force_xyz))
            max_force_norm_identity_error = max(
                max_force_norm_identity_error,
                abs(force_norm - recorded_force_norms[sample_index][leg_index]),
            )
            vertical_force = max(force_xyz[2], 0.0)
            force_norm_traces[leg_index].append(force_norm)
            vertical_force_traces[leg_index].append(vertical_force)
            if force_norm > CONTACT_FORCE_THRESHOLD_N:
                nonzero_force_samples += 1
                if force_xyz[2] < 0.0:
                    negative_vertical_force_samples += 1
            values = {
                "torque_squared": sum(torque * torque for torque in leg_torques),
                "normalized_torque_utilization": sum(
                    (torque / effort_limits[joint_start + motor_index]) ** 2
                    for motor_index, torque in enumerate(leg_torques)
                ),
                "absolute_work": sum(abs(power) for power in leg_powers),
                "positive_work": sum(max(power, 0.0) for power in leg_powers),
                "negative_work": sum(max(-power, 0.0) for power in leg_powers),
                "grf_impulse": force_norm,
                "vertical_grf_impulse": vertical_force,
                "contact_time": float(force_norm > CONTACT_FORCE_THRESHOLD_N),
            }
            for name, value in values.items():
                per_leg[name].append(value)
        for name, leg_values in per_leg.items():
            metric_pairs[name].append((sum(leg_values[:2]), sum(leg_values[2:])))

    rng = random.Random(BOOTSTRAP_SEED + sum(ord(character) for character in run.key))
    metrics = {
        name: summarize_additive_metric(
            pairs,
            dt_s=dt_s,
            block_samples=block_samples,
            rng=rng,
        )
        for name, pairs in metric_pairs.items()
    }
    path_distance_m = sum(
        math.hypot(
            base_positions[right][0] - base_positions[left][0],
            base_positions[right][1] - base_positions[left][1],
        )
        for left, right in zip(selected, selected[1:], strict=False)
    )
    tracking_errors = [
        math.hypot(
            true_velocities[index][0] - commands[index][0],
            true_velocities[index][1] - commands[index][1],
        )
        for index in selected
    ]
    command_x = [commands[index][0] for index in selected]
    command_y = [commands[index][1] for index in selected]
    command_yaw = [commands[index][2] for index in selected]
    duration_s = len(selected) * dt_s
    mean_command_x = statistics.fmean(command_x)
    mean_command_y = statistics.fmean(command_y)
    planar_command_norm = math.hypot(mean_command_x, mean_command_y)
    if planar_command_norm <= 0.0:
        raise ValueError(f"Planar command is zero in {run.evaluation_path}")
    command_unit_x = mean_command_x / planar_command_norm
    command_unit_y = mean_command_y / planar_command_norm
    displacement_x = base_positions[selected[-1]][0] - base_positions[selected[0]][0]
    displacement_y = base_positions[selected[-1]][1] - base_positions[selected[0]][1]
    signed_directed_progress_m = displacement_x * command_unit_x + displacement_y * command_unit_y
    directed_distance_m = abs(signed_directed_progress_m)
    if directed_distance_m <= 0.0:
        raise ValueError(f"Directed progress is zero in {run.evaluation_path}")
    lateral_drift_m = abs(-displacement_x * command_unit_y + displacement_y * command_unit_x)
    selected_torque_rows = [joint_torques[index] for index in selected]
    selected_power_rows = [joint_powers[index] for index in selected]
    durability = analyze_durability_exposure(
        joint_names=joint_names,
        effort_limits_nm=effort_limits,
        torque_rows=selected_torque_rows,
        power_rows=selected_power_rows,
        force_norm_traces=force_norm_traces,
        vertical_force_traces=vertical_force_traces,
        dt_s=dt_s,
        duration_s=duration_s,
        directed_distance_m=directed_distance_m,
    )
    total_abs_work_j = metrics["absolute_work"]["front_integral"] + metrics["absolute_work"]["hind_integral"]
    total_positive_work_j = metrics["positive_work"]["front_integral"] + metrics["positive_work"]["hind_integral"]
    total_torque_sq = metrics["torque_squared"]["front_integral"] + metrics["torque_squared"]["hind_integral"]
    total_vertical_impulse = (
        metrics["vertical_grf_impulse"]["front_integral"] + metrics["vertical_grf_impulse"]["hind_integral"]
    )
    for pair_name in ("front", "hind"):
        metrics["contact_time"][f"{pair_name}_duty_factor"] = metrics["contact_time"][f"{pair_name}_integral"] / (
            duration_s * 2.0
        )
    return {
        "path": str(run.evaluation_path.relative_to(ROOT)),
        "sample_count": len(selected),
        "dt_s": dt_s,
        "duration_s": duration_s,
        "window_s": [WINDOW_START_S, WINDOW_STOP_S],
        "command": {
            "x_mean_mps": statistics.fmean(command_x),
            "x_range_mps": [min(command_x), max(command_x)],
            "y_range_mps": [min(command_y), max(command_y)],
            "yaw_range_radps": [min(command_yaw), max(command_yaw)],
        },
        "actual_distance_m": path_distance_m,
        "directed_progress_m": directed_distance_m,
        "signed_directed_progress_m": signed_directed_progress_m,
        "lateral_drift_m": lateral_drift_m,
        "planar_tracking_rmse_mps": math.sqrt(statistics.fmean(error * error for error in tracking_errors)),
        "metrics": metrics,
        "durability": durability,
        "cost_distance_definition": "absolute progress projected onto the mean commanded planar direction",
        "cost_per_distance": {
            "absolute_work_j_per_m": total_abs_work_j / directed_distance_m,
            "positive_work_j_per_m": total_positive_work_j / directed_distance_m,
            "torque_squared_integral_per_m": total_torque_sq / directed_distance_m,
            "vertical_grf_impulse_ns_per_m": total_vertical_impulse / directed_distance_m,
        },
        "validation": {
            "joint_names": joint_names,
            "episode_ends_in_window": sum(done[index] for index in selected),
            "all_grf_samples_include_friction": all(includes_friction[index] for index in selected),
            "max_abs_power_identity_error_w": max_power_identity_error,
            "max_abs_force_norm_identity_error_n": max_force_norm_identity_error,
            "negative_vertical_force_fraction_during_contact": (
                negative_vertical_force_samples / nonzero_force_samples if nonzero_force_samples else 0.0
            ),
        },
    }


def read_varint(data: bytes | memoryview, offset: int) -> tuple[int, int]:
    """Decode one protobuf varint."""
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("Invalid protobuf varint")


def iter_protobuf_fields(data: bytes | memoryview) -> Iterable[tuple[int, int, int | memoryview]]:
    """Yield protobuf fields as ``(number, wire_type, value)`` tuples."""
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        key, offset = read_varint(view, offset)
        field_number = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            value, offset = read_varint(view, offset)
        elif wire_type == 1:
            value = view[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            size, offset = read_varint(view, offset)
            value = view[offset : offset + size]
            offset += size
        elif wire_type == 5:
            value = view[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def parse_tensor_scalar(data: bytes | memoryview) -> float | None:
    """Decode the first scalar from a TensorProto."""
    dtype = None
    tensor_content = None
    typed_values: dict[int, list[float | int]] = defaultdict(list)
    for field, wire, value in iter_protobuf_fields(data):
        if field == 1 and wire == 0:
            dtype = int(value)
        elif field == 4 and wire == 2:
            tensor_content = bytes(value)
        elif field in {5, 6} and wire == 5:
            typed_values[field].append(struct.unpack("<f", value)[0])
        elif field == 6 and wire == 1:
            typed_values[field].append(struct.unpack("<d", value)[0])
        elif field in {7, 10, 11, 16, 17} and wire == 0:
            typed_values[field].append(int(value))
        elif field in {5, 6, 7, 10, 11, 16, 17} and wire == 2:
            packed = bytes(value)
            if field == 5 and len(packed) >= 4:
                typed_values[field].append(struct.unpack_from("<f", packed)[0])
            elif field == 6 and len(packed) >= 8:
                typed_values[field].append(struct.unpack_from("<d", packed)[0])
            elif field in {7, 10, 11, 16, 17} and packed:
                first, _ = read_varint(packed, 0)
                typed_values[field].append(first)
    if tensor_content:
        tensor_formats = {
            1: "f",
            2: "d",
            3: "i",
            9: "q",
            10: "?",
            22: "I",
            23: "Q",
        }
        item_format = tensor_formats.get(dtype)
        if item_format and len(tensor_content) >= struct.calcsize("<" + item_format):
            return float(struct.unpack_from("<" + item_format, tensor_content)[0])
    dtype_fields = {1: 5, 2: 6, 3: 7, 9: 10, 10: 11, 22: 16, 23: 17}
    typed_field = dtype_fields.get(dtype)
    if typed_field is not None and typed_values[typed_field]:
        return float(typed_values[typed_field][0])
    return None


def parse_summary_value(data: bytes | memoryview) -> tuple[str | None, float | None]:
    """Decode one TensorBoard Summary.Value message."""
    tag = None
    scalar = None
    for field, wire, value in iter_protobuf_fields(data):
        if field == 1 and wire == 2:
            tag = bytes(value).decode("utf-8", errors="replace")
        elif field == 2 and wire == 5:
            scalar = float(struct.unpack("<f", value)[0])
        elif field == 8 and wire == 2:
            scalar = parse_tensor_scalar(value)
    return tag, scalar


def parse_event(data: bytes) -> tuple[float | None, int | None, list[tuple[str, float]]]:
    """Decode wall time, step, and scalar summaries from one TensorBoard event."""
    wall_time = None
    step = None
    summaries: list[tuple[str, float]] = []
    for field, wire, value in iter_protobuf_fields(data):
        if field == 1 and wire == 1:
            wall_time = float(struct.unpack("<d", value)[0])
        elif field == 2 and wire == 0:
            step = int(value)
        elif field == 5 and wire == 2:
            for summary_field, summary_wire, summary_value in iter_protobuf_fields(value):
                if summary_field == 1 and summary_wire == 2:
                    tag, scalar = parse_summary_value(summary_value)
                    if tag is not None and scalar is not None and math.isfinite(scalar):
                        summaries.append((tag, scalar))
    return wall_time, step, summaries


def read_scalars(
    event_path: Path,
    selected_tags: set[str] | None = None,
) -> dict[str, list[tuple[int, float, float]]]:
    """Read scalar points keyed by TensorBoard tag.

    Args:
        event_path: TensorBoard event file to read.
        selected_tags: Optional tags to retain. All scalar tags are retained
            when omitted.

    Returns:
        Scalar points keyed by TensorBoard tag.
    """
    scalars: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with event_path.open("rb") as stream:
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {event_path}")
            record_size = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord length CRC in {event_path}")
            record = stream.read(record_size)
            if len(record) != record_size or len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord payload in {event_path}")
            wall_time, step, values = parse_event(record)
            if wall_time is None or step is None:
                continue
            for tag, value in values:
                if selected_tags is None or tag in selected_tags:
                    scalars[tag].append((step, wall_time, value))
    for points in scalars.values():
        points.sort(key=lambda point: (point[0], point[1]))
    return dict(scalars)


def rolling_mean(points: Sequence[tuple[int, float, float]], window: int) -> list[tuple[int, float, float]]:
    """Return a trailing rolling mean over scalar values."""
    result = []
    running_sum = 0.0
    values: list[float] = []
    for step, wall_time, value in points:
        values.append(value)
        running_sum += value
        if len(values) > window:
            running_sum -= values[-window - 1]
        result.append((step, wall_time, running_sum / min(len(values), window)))
    return result


def trapezoid_auc(points: Sequence[tuple[int, float, float]]) -> float:
    """Compute a step-normalized trapezoidal AUC."""
    if len(points) < 2 or points[-1][0] == points[0][0]:
        return float("nan")
    area = 0.0
    for left, right in zip(points, points[1:], strict=False):
        area += 0.5 * (left[2] + right[2]) * (right[0] - left[0])
    return area / (points[-1][0] - points[0][0])


def threshold_crossing(
    points: Sequence[tuple[int, float, float]],
    threshold: float,
    *,
    above: bool,
    rolling_window: int = 200,
    sustain_iterations: int = 500,
) -> dict[str, float | int] | None:
    """Find the first sustained threshold crossing of a rolling-mean curve."""
    smoothed = rolling_mean(points, rolling_window)
    eligible = [point for index, point in enumerate(smoothed) if index + 1 >= rolling_window]
    condition = (lambda value: value >= threshold) if above else (lambda value: value <= threshold)
    for index, point in enumerate(eligible):
        following = eligible[index : index + sustain_iterations]
        if len(following) < sustain_iterations:
            break
        if all(condition(candidate[2]) for candidate in following):
            step = point[0]
            return {
                "iteration": step,
                "environment_transitions": step * SAMPLES_PER_ITERATION,
                "wall_time_hours": (point[1] - points[0][1]) / 3600.0,
                "rolling_mean": point[2],
            }
    return None


def endpoint(points: Sequence[tuple[int, float, float]], window: int = 1000) -> dict[str, float]:
    """Summarize the endpoint and full/early AUC of one scalar curve."""
    last_window = points[-window:]
    first_half = [point for point in points if point[0] <= 10_000]
    return {
        "last_1000_mean": statistics.fmean(point[2] for point in last_window),
        "last_1000_std": statistics.pstdev(point[2] for point in last_window),
        "auc_all": trapezoid_auc(points),
        "auc_first_10000": trapezoid_auc(first_half),
    }


def analyze_training(run: RunSpec) -> dict[str, Any]:
    """Analyze sample, endpoint, and compute efficiency for one training run."""
    event_path = next(run.run_path.glob("events.out.tfevents.*"))
    scalars = read_scalars(event_path)
    required_tags = (
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Diagnostics/straight_line_reward",
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
    )
    missing_tags = [tag for tag in required_tags if tag not in scalars]
    if missing_tags:
        raise ValueError(f"Missing scalar tags in {event_path}: {missing_tags}")
    reward = scalars["Train/mean_reward"]
    episode_length = scalars["Train/mean_episode_length"]
    straight_reward = scalars["Diagnostics/straight_line_reward"]
    velocity_error_xy = scalars["Metrics/base_velocity/error_vel_xy"]
    velocity_error_yaw = scalars["Metrics/base_velocity/error_vel_yaw"]
    fps = scalars["Perf/total_fps"]
    collection_time = scalars["Perf/collection_time"]
    learning_time = scalars["Perf/learning_time"]
    expected_logged_points = 20_000 - reward[0][0]
    if reward[0][0] not in {0, 1} or reward[-1][0] != 19_999 or len(reward) != expected_logged_points:
        raise ValueError(
            f"Incomplete reward trace in {event_path}: {len(reward)} points, steps {reward[0][0]}..{reward[-1][0]}"
        )
    return {
        "event_path": str(event_path.relative_to(ROOT)),
        "iterations": reward[-1][0] + 1,
        "environment_transitions": (reward[-1][0] + 1) * SAMPLES_PER_ITERATION,
        "wall_time_hours": (reward[-1][1] - reward[0][1]) / 3600.0,
        "mean_reward": endpoint(reward),
        "mean_episode_length": endpoint(episode_length),
        "straight_line_reward": endpoint(straight_reward),
        "velocity_error_xy": endpoint(velocity_error_xy),
        "velocity_error_yaw": endpoint(velocity_error_yaw),
        "thresholds": {
            "reward_30": threshold_crossing(reward, 30.0, above=True),
            "reward_35": threshold_crossing(reward, 35.0, above=True),
            "straight_reward_1p5": threshold_crossing(straight_reward, 1.5, above=True),
            "velocity_error_xy_0p15": threshold_crossing(velocity_error_xy, 0.15, above=False),
        },
        "throughput_fps": {
            "mean": statistics.fmean(point[2] for point in fps),
            "median": statistics.median(point[2] for point in fps),
            "last_1000_mean": statistics.fmean(point[2] for point in fps[-1000:]),
        },
        "timing_seconds_per_iteration": {
            "collection_mean": statistics.fmean(point[2] for point in collection_time),
            "learning_mean": statistics.fmean(point[2] for point in learning_time),
            "collection_last_1000_mean": statistics.fmean(point[2] for point in collection_time[-1000:]),
            "learning_last_1000_mean": statistics.fmean(point[2] for point in learning_time[-1000:]),
        },
    }


def percent_change(value: float, reference: float) -> float:
    """Return percent change from ``reference`` to ``value``."""
    return 100.0 * (value / reference - 1.0)


def compare_to_baselines(
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
    training_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compare every TRS cell against its robot's no-TRS baseline."""
    comparisons: dict[str, dict[str, Any]] = {}
    baseline_runs = {run.robot: run for run in runs if not run.trs_enabled}
    for run in runs:
        if not run.trs_enabled:
            continue
        baseline = baseline_runs[run.robot]
        rollout = rollout_results[run.key]
        baseline_rollout = rollout_results[baseline.key]
        training = training_results[run.key]
        baseline_training = training_results[baseline.key]
        leg_metrics = {}
        for metric_name in (*PRIMARY_LEG_METRICS, *ADDITIONAL_LEG_METRICS):
            metric = rollout["metrics"][metric_name]
            baseline_metric = baseline_rollout["metrics"][metric_name]
            total = metric["front_integral"] + metric["hind_integral"]
            baseline_total = baseline_metric["front_integral"] + baseline_metric["hind_integral"]
            leg_metrics[metric_name] = {
                "signed_imbalance_change_pp": (
                    metric["signed_imbalance_percent"] - baseline_metric["signed_imbalance_percent"]
                ),
                "abs_imbalance_change_pp": (metric["abs_imbalance_percent"] - baseline_metric["abs_imbalance_percent"]),
                "total_exposure_change_percent": percent_change(total, baseline_total),
            }

        durability_pair_metrics = {}
        for metric_name in DURABILITY_PAIR_METRICS:
            metric = rollout["durability"]["pair_metrics"][metric_name]
            baseline_metric = baseline_rollout["durability"]["pair_metrics"][metric_name]
            durability_pair_metrics[metric_name] = {
                "dominant_pair": metric["dominant_pair"],
                "baseline_dominant_pair": baseline_metric["dominant_pair"],
                "dominant_to_other_ratio": metric["dominant_to_other_ratio"],
                "baseline_dominant_to_other_ratio": baseline_metric["dominant_to_other_ratio"],
                "dominant_to_other_ratio_change": (
                    metric["dominant_to_other_ratio"] - baseline_metric["dominant_to_other_ratio"]
                ),
                "abs_imbalance_change_pp": (metric["abs_imbalance_percent"] - baseline_metric["abs_imbalance_percent"]),
                "front_per_m_change_percent": percent_change(
                    metric["front_per_m"],
                    baseline_metric["front_per_m"],
                ),
                "hind_per_m_change_percent": percent_change(
                    metric["hind_per_m"],
                    baseline_metric["hind_per_m"],
                ),
                "total_per_m_change_percent": percent_change(
                    metric["total_per_m"],
                    baseline_metric["total_per_m"],
                ),
                "worst_pair_per_m_change_percent": percent_change(
                    max(metric["front_per_m"], metric["hind_per_m"]),
                    max(baseline_metric["front_per_m"], baseline_metric["hind_per_m"]),
                ),
            }

        joint_comparison_fields = (
            "normalized_torque_rms",
            "normalized_torque_abs_p99",
            "normalized_torque_abs_peak",
            "torque_squared_exposure_n2m2s_per_m",
            "fatigue_proxy_m3_per_m",
            "fatigue_proxy_m5_per_m",
            "absolute_work_j_per_m",
        )
        joint_comparisons = []
        for joint, baseline_joint in zip(
            rollout["durability"]["joints"],
            baseline_rollout["durability"]["joints"],
            strict=True,
        ):
            if joint["joint_name"] != baseline_joint["joint_name"]:
                raise ValueError(
                    f"Joint order differs from baseline for {run.key}: "
                    f"{joint['joint_name']} != {baseline_joint['joint_name']}"
                )
            joint_comparison = {
                "joint_index": joint["joint_index"],
                "joint_name": joint["joint_name"],
                "leg": joint["leg"],
                "pair": joint["pair"],
                "changes_percent": {
                    field: percent_change(joint[field], baseline_joint[field]) for field in joint_comparison_fields
                },
                "dwell_fraction_changes_pp": {},
            }
            for threshold in NORMALIZED_TORQUE_DWELL_THRESHOLDS:
                threshold_slug = f"{threshold:.2f}".replace(".", "p")
                field = f"dwell_fraction_at_or_above_{threshold_slug}"
                joint_comparison["dwell_fraction_changes_pp"][threshold_slug] = 100.0 * (
                    joint[field] - baseline_joint[field]
                )
            joint_comparisons.append(joint_comparison)

        leg_comparison_fields = (
            "vertical_grf_impulse_ns_per_m",
            "vertical_force_p99_n",
            "vertical_force_peak_n",
            "positive_vertical_loading_rate_p99_nps",
            "positive_vertical_loading_rate_peak_nps",
        )
        leg_comparisons = []
        for leg, baseline_leg in zip(
            rollout["durability"]["legs"],
            baseline_rollout["durability"]["legs"],
            strict=True,
        ):
            if leg["leg"] != baseline_leg["leg"]:
                raise ValueError(
                    f"Leg order differs from baseline for {run.key}: {leg['leg']} != {baseline_leg['leg']}"
                )
            leg_comparisons.append(
                {
                    "leg_index": leg["leg_index"],
                    "leg": leg["leg"],
                    "pair": leg["pair"],
                    "changes_percent": {
                        field: percent_change(leg[field], baseline_leg[field]) for field in leg_comparison_fields
                    },
                }
            )

        worst_component_comparisons = {}
        for metric_name, worst in rollout["durability"]["worst_components"].items():
            records_key = "joints" if metric_name.startswith("joint_") else "legs"
            identity_field = "joint_name" if records_key == "joints" else "leg"
            baseline_records = baseline_rollout["durability"][records_key]
            baseline_same_component = next(
                record for record in baseline_records if record[identity_field] == worst["component"]
            )
            baseline_worst = baseline_rollout["durability"]["worst_components"][metric_name]
            worst_component_comparisons[metric_name] = {
                **worst,
                "baseline_same_component_value": baseline_same_component[worst["field"]],
                "change_vs_same_component_baseline_percent": percent_change(
                    worst["value"],
                    baseline_same_component[worst["field"]],
                ),
                "baseline_worst_component": baseline_worst["component"],
                "baseline_worst_value": baseline_worst["value"],
                "global_worst_change_percent": percent_change(
                    worst["value"],
                    baseline_worst["value"],
                ),
            }

        threshold_comparisons = {}
        for threshold_name, crossing in training["thresholds"].items():
            baseline_crossing = baseline_training["thresholds"][threshold_name]
            if crossing is None or baseline_crossing is None:
                threshold_comparisons[threshold_name] = None
            else:
                threshold_comparisons[threshold_name] = {
                    "iteration_change": crossing["iteration"] - baseline_crossing["iteration"],
                    "transition_change_percent": percent_change(
                        crossing["environment_transitions"],
                        baseline_crossing["environment_transitions"],
                    ),
                    "wall_time_change_percent": percent_change(
                        crossing["wall_time_hours"],
                        baseline_crossing["wall_time_hours"],
                    ),
                }
        comparisons[run.key] = {
            "baseline_key": baseline.key,
            "leg_metrics": leg_metrics,
            "durability": {
                "pair_metrics": durability_pair_metrics,
                "joints": joint_comparisons,
                "legs": leg_comparisons,
                "worst_components": worst_component_comparisons,
            },
            "cost_per_distance": {
                name: percent_change(value, baseline_rollout["cost_per_distance"][name])
                for name, value in rollout["cost_per_distance"].items()
            },
            "training": {
                "reward_auc_change_percent": percent_change(
                    training["mean_reward"]["auc_all"],
                    baseline_training["mean_reward"]["auc_all"],
                ),
                "reward_auc_first_10000_change_percent": percent_change(
                    training["mean_reward"]["auc_first_10000"],
                    baseline_training["mean_reward"]["auc_first_10000"],
                ),
                "final_reward_change": (
                    training["mean_reward"]["last_1000_mean"] - baseline_training["mean_reward"]["last_1000_mean"]
                ),
                "straight_reward_auc_change_percent": percent_change(
                    training["straight_line_reward"]["auc_all"],
                    baseline_training["straight_line_reward"]["auc_all"],
                ),
                "final_velocity_error_xy_change_percent": percent_change(
                    training["velocity_error_xy"]["last_1000_mean"],
                    baseline_training["velocity_error_xy"]["last_1000_mean"],
                ),
                "throughput_change_percent": percent_change(
                    training["throughput_fps"]["mean"],
                    baseline_training["throughput_fps"]["mean"],
                ),
                "learning_time_change_percent": percent_change(
                    training["timing_seconds_per_iteration"]["learning_mean"],
                    baseline_training["timing_seconds_per_iteration"]["learning_mean"],
                ),
                "collection_time_change_percent": percent_change(
                    training["timing_seconds_per_iteration"]["collection_mean"],
                    baseline_training["timing_seconds_per_iteration"]["collection_mean"],
                ),
                "full_run_wall_time_change_percent": percent_change(
                    training["wall_time_hours"],
                    baseline_training["wall_time_hours"],
                ),
                "thresholds": threshold_comparisons,
            },
        }
    return comparisons


def format_number(value: float | None, digits: int = 1, signed: bool = False) -> str:
    """Format a numeric table entry."""
    if value is None or not math.isfinite(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def grid_runs(runs: Sequence[RunSpec], robot: str) -> dict[tuple[float, int], RunSpec]:
    """Return the TRS grid indexed by mirror coefficient and warm-up."""
    return {
        (run.mirror_coeff, int(run.warmup_iterations)): run for run in runs if run.robot == robot and run.trs_enabled
    }


def markdown_grid(
    runs: Sequence[RunSpec],
    robot: str,
    value_getter,
    *,
    digits: int = 1,
    signed: bool = True,
    suffix: str = "",
) -> list[str]:
    """Render a 3-by-3 TRS grid as a Markdown table."""
    indexed = grid_runs(runs, robot)
    lines = [
        "| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mirror, value_coeff in COEFFICIENT_PAIRS:
        values = []
        for warmup in WARMUP_ITERATIONS:
            value = value_getter(indexed[(mirror, warmup)])
            formatted = format_number(value, digits=digits, signed=signed)
            values.append(f"{formatted}{suffix}" if formatted != "n/a" else formatted)
        lines.append(f"| {mirror:.2f} / {value_coeff:.2f} | {' | '.join(values)} |")
    return lines


def write_leg_csv(
    path: Path,
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Write long-form front/hind metric results."""
    fieldnames = (
        "robot",
        "condition",
        "mirror_coeff",
        "value_coeff",
        "warmup_iterations",
        "metric",
        "sample_mean_unit",
        "integral_unit",
        "front_sample_mean",
        "hind_sample_mean",
        "front_integral",
        "hind_integral",
        "front_integral_per_s",
        "hind_integral_per_s",
        "front_integral_per_directed_m",
        "hind_integral_per_directed_m",
        "total_integral_per_directed_m",
        "worst_pair_integral_per_directed_m",
        "front_share_percent",
        "signed_imbalance_percent",
        "abs_imbalance_percent",
        "dominant_pair",
        "dominant_to_other_ratio",
        "concentration_factor",
        "bootstrap_ci_low_percent",
        "bootstrap_ci_high_percent",
        "abs_imbalance_change_vs_no_trs_pp",
        "total_exposure_change_vs_no_trs_percent",
        "total_per_directed_m_change_vs_no_trs_percent",
        "worst_pair_per_directed_m_change_vs_no_trs_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            rollout = rollout_results[run.key]
            duration_s = rollout["duration_s"]
            directed_distance_m = rollout["directed_progress_m"]
            for metric_name, metric in rollout["metrics"].items():
                comparison = None if not run.trs_enabled else comparisons[run.key]["leg_metrics"][metric_name]
                dominant_pair, dominant_ratio = pair_concentration(
                    metric["front_integral"],
                    metric["hind_integral"],
                )
                baseline_metric = (
                    None
                    if comparison is None
                    else rollout_results[comparisons[run.key]["baseline_key"]]["metrics"][metric_name]
                )
                total_per_m = (metric["front_integral"] + metric["hind_integral"]) / directed_distance_m
                worst_pair_per_m = (
                    max(
                        metric["front_integral"],
                        metric["hind_integral"],
                    )
                    / directed_distance_m
                )
                if baseline_metric is None:
                    total_per_m_change = None
                    worst_pair_per_m_change = None
                else:
                    baseline_rollout = rollout_results[comparisons[run.key]["baseline_key"]]
                    baseline_distance_m = baseline_rollout["directed_progress_m"]
                    baseline_total_per_m = (
                        baseline_metric["front_integral"] + baseline_metric["hind_integral"]
                    ) / baseline_distance_m
                    baseline_worst_pair_per_m = (
                        max(
                            baseline_metric["front_integral"],
                            baseline_metric["hind_integral"],
                        )
                        / baseline_distance_m
                    )
                    total_per_m_change = percent_change(total_per_m, baseline_total_per_m)
                    worst_pair_per_m_change = percent_change(
                        worst_pair_per_m,
                        baseline_worst_pair_per_m,
                    )
                writer.writerow(
                    {
                        "robot": run.robot,
                        "condition": run.condition,
                        "mirror_coeff": run.mirror_coeff,
                        "value_coeff": run.value_coeff,
                        "warmup_iterations": run.warmup_iterations,
                        "metric": metric_name,
                        "sample_mean_unit": METRIC_UNITS[metric_name][0],
                        "integral_unit": METRIC_UNITS[metric_name][1],
                        "front_sample_mean": metric["front_sample_mean"],
                        "hind_sample_mean": metric["hind_sample_mean"],
                        "front_integral": metric["front_integral"],
                        "hind_integral": metric["hind_integral"],
                        "front_integral_per_s": metric["front_integral"] / duration_s,
                        "hind_integral_per_s": metric["hind_integral"] / duration_s,
                        "front_integral_per_directed_m": (metric["front_integral"] / directed_distance_m),
                        "hind_integral_per_directed_m": (metric["hind_integral"] / directed_distance_m),
                        "total_integral_per_directed_m": total_per_m,
                        "worst_pair_integral_per_directed_m": worst_pair_per_m,
                        "front_share_percent": metric["front_share_percent"],
                        "signed_imbalance_percent": metric["signed_imbalance_percent"],
                        "abs_imbalance_percent": metric["abs_imbalance_percent"],
                        "dominant_pair": dominant_pair,
                        "dominant_to_other_ratio": dominant_ratio,
                        "concentration_factor": 2.0 * dominant_ratio / (1.0 + dominant_ratio),
                        "bootstrap_ci_low_percent": metric["bootstrap_95_percent"][0],
                        "bootstrap_ci_high_percent": metric["bootstrap_95_percent"][1],
                        "abs_imbalance_change_vs_no_trs_pp": (
                            None if comparison is None else comparison["abs_imbalance_change_pp"]
                        ),
                        "total_exposure_change_vs_no_trs_percent": (
                            None if comparison is None else comparison["total_exposure_change_percent"]
                        ),
                        "total_per_directed_m_change_vs_no_trs_percent": total_per_m_change,
                        "worst_pair_per_directed_m_change_vs_no_trs_percent": (worst_pair_per_m_change),
                    }
                )


def write_durability_pair_csv(
    path: Path,
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Write pair allocation and mission-normalized durability exposures."""
    fieldnames = (
        "robot",
        "condition",
        "mirror_coeff",
        "value_coeff",
        "warmup_iterations",
        "metric",
        "mechanism",
        "unit",
        "per_distance_unit",
        "front",
        "hind",
        "total",
        "front_per_s",
        "hind_per_s",
        "total_per_s",
        "front_per_directed_m",
        "hind_per_directed_m",
        "total_per_directed_m",
        "worst_pair_per_directed_m",
        "front_share_percent",
        "signed_imbalance_percent",
        "abs_imbalance_percent",
        "dominant_pair",
        "dominant_to_other_ratio",
        "concentration_factor",
        "abs_imbalance_change_vs_no_trs_pp",
        "total_per_directed_m_change_vs_no_trs_percent",
        "worst_pair_per_directed_m_change_vs_no_trs_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for metric_name, metric in rollout_results[run.key]["durability"]["pair_metrics"].items():
                comparison = (
                    None if not run.trs_enabled else comparisons[run.key]["durability"]["pair_metrics"][metric_name]
                )
                dominance_ratio = metric["dominant_to_other_ratio"]
                writer.writerow(
                    {
                        "robot": run.robot,
                        "condition": run.condition,
                        "mirror_coeff": run.mirror_coeff,
                        "value_coeff": run.value_coeff,
                        "warmup_iterations": run.warmup_iterations,
                        "metric": metric_name,
                        **DURABILITY_PAIR_METRICS[metric_name],
                        "front": metric["front"],
                        "hind": metric["hind"],
                        "total": metric["total"],
                        "front_per_s": metric["front_per_s"],
                        "hind_per_s": metric["hind_per_s"],
                        "total_per_s": metric["total_per_s"],
                        "front_per_directed_m": metric["front_per_m"],
                        "hind_per_directed_m": metric["hind_per_m"],
                        "total_per_directed_m": metric["total_per_m"],
                        "worst_pair_per_directed_m": max(
                            metric["front_per_m"],
                            metric["hind_per_m"],
                        ),
                        "front_share_percent": metric["front_share_percent"],
                        "signed_imbalance_percent": metric["signed_imbalance_percent"],
                        "abs_imbalance_percent": metric["abs_imbalance_percent"],
                        "dominant_pair": metric["dominant_pair"],
                        "dominant_to_other_ratio": dominance_ratio,
                        "concentration_factor": 2.0 * dominance_ratio / (1.0 + dominance_ratio),
                        "abs_imbalance_change_vs_no_trs_pp": (
                            None if comparison is None else comparison["abs_imbalance_change_pp"]
                        ),
                        "total_per_directed_m_change_vs_no_trs_percent": (
                            None if comparison is None else comparison["total_per_m_change_percent"]
                        ),
                        "worst_pair_per_directed_m_change_vs_no_trs_percent": (
                            None if comparison is None else comparison["worst_pair_per_m_change_percent"]
                        ),
                    }
                )


def write_joint_durability_csv(
    path: Path,
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Write per-joint load, capacity, work, and cyclic-fatigue descriptors."""
    base_fields = (
        "robot",
        "condition",
        "mirror_coeff",
        "value_coeff",
        "warmup_iterations",
        "joint_index",
        "joint_name",
        "leg",
        "pair",
        "effort_limit_nm",
        "torque_rms_nm",
        "torque_abs_p95_nm",
        "torque_abs_p99_nm",
        "torque_abs_peak_nm",
        "normalized_torque_rms",
        "normalized_torque_abs_p95",
        "normalized_torque_abs_p99",
        "normalized_torque_abs_peak",
        "torque_squared_exposure_n2m2s_per_s",
        "torque_squared_exposure_n2m2s_per_m",
        "normalized_torque_capacity_exposure_s_per_s",
        "normalized_torque_capacity_exposure_s_per_m",
        "absolute_work_j_per_s",
        "absolute_work_j_per_m",
        "positive_work_j_per_m",
        "negative_work_j_per_m",
        "absolute_power_p99_w",
        "absolute_power_peak_w",
        "rainflow_equivalent_cycle_count",
        "rainflow_cycle_count_per_s",
        "rainflow_cycle_count_per_m",
        "rainflow_range_max_nm",
        "rainflow_normalized_amplitude_max",
        "torque_cycle_moment_m3_per_s",
        "torque_cycle_moment_m3_per_m",
        "torque_cycle_moment_m5_per_s",
        "torque_cycle_moment_m5_per_m",
        "fatigue_proxy_m3_per_s",
        "fatigue_proxy_m3_per_m",
        "fatigue_proxy_m5_per_s",
        "fatigue_proxy_m5_per_m",
        "cycle_average_equivalent_amplitude_m3_nm",
        "cycle_average_equivalent_amplitude_m5_nm",
    )
    dwell_fields = tuple(
        field
        for threshold in NORMALIZED_TORQUE_DWELL_THRESHOLDS
        for field in (
            f"dwell_fraction_at_or_above_{str(f'{threshold:.2f}').replace('.', 'p')}",
            f"dwell_time_s_at_or_above_{str(f'{threshold:.2f}').replace('.', 'p')}",
            f"longest_dwell_s_at_or_above_{str(f'{threshold:.2f}').replace('.', 'p')}",
        )
    )
    change_fields = (
        "normalized_torque_rms_change_vs_same_joint_percent",
        "normalized_torque_abs_p99_change_vs_same_joint_percent",
        "normalized_torque_abs_peak_change_vs_same_joint_percent",
        "torque_squared_exposure_per_m_change_vs_same_joint_percent",
        "fatigue_proxy_m3_per_m_change_vs_same_joint_percent",
        "fatigue_proxy_m5_per_m_change_vs_same_joint_percent",
        "absolute_work_per_m_change_vs_same_joint_percent",
    )
    dwell_change_fields = tuple(
        f"dwell_fraction_at_or_above_{str(f'{threshold:.2f}').replace('.', 'p')}_change_pp"
        for threshold in NORMALIZED_TORQUE_DWELL_THRESHOLDS
    )
    fieldnames = (*base_fields, *dwell_fields, *change_fields, *dwell_change_fields)
    comparison_fields = {
        "normalized_torque_rms_change_vs_same_joint_percent": "normalized_torque_rms",
        "normalized_torque_abs_p99_change_vs_same_joint_percent": "normalized_torque_abs_p99",
        "normalized_torque_abs_peak_change_vs_same_joint_percent": "normalized_torque_abs_peak",
        "torque_squared_exposure_per_m_change_vs_same_joint_percent": ("torque_squared_exposure_n2m2s_per_m"),
        "fatigue_proxy_m3_per_m_change_vs_same_joint_percent": "fatigue_proxy_m3_per_m",
        "fatigue_proxy_m5_per_m_change_vs_same_joint_percent": "fatigue_proxy_m5_per_m",
        "absolute_work_per_m_change_vs_same_joint_percent": "absolute_work_j_per_m",
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            joint_comparisons = None if not run.trs_enabled else comparisons[run.key]["durability"]["joints"]
            for joint in rollout_results[run.key]["durability"]["joints"]:
                row = {
                    "robot": run.robot,
                    "condition": run.condition,
                    "mirror_coeff": run.mirror_coeff,
                    "value_coeff": run.value_coeff,
                    "warmup_iterations": run.warmup_iterations,
                    **{field: joint[field] for field in (*base_fields[5:], *dwell_fields)},
                }
                if joint_comparisons is not None:
                    joint_comparison = joint_comparisons[joint["joint_index"]]
                    for output_field, source_field in comparison_fields.items():
                        row[output_field] = joint_comparison["changes_percent"][source_field]
                    for threshold in NORMALIZED_TORQUE_DWELL_THRESHOLDS:
                        threshold_slug = f"{threshold:.2f}".replace(".", "p")
                        row[f"dwell_fraction_at_or_above_{threshold_slug}_change_pp"] = joint_comparison[
                            "dwell_fraction_changes_pp"
                        ][threshold_slug]
                writer.writerow(row)


def write_foot_impact_csv(
    path: Path,
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Write per-foot support and low-bandwidth impact-load descriptors."""
    metric_fields = (
        "vertical_grf_impulse_ns_per_s",
        "vertical_grf_impulse_ns_per_m",
        "grf_impulse_ns_per_m",
        "contact_duty_factor",
        "vertical_force_contact_mean_n",
        "vertical_force_p95_n",
        "vertical_force_p99_n",
        "vertical_force_peak_n",
        "positive_vertical_loading_rate_p95_nps",
        "positive_vertical_loading_rate_p99_nps",
        "positive_vertical_loading_rate_peak_nps",
    )
    change_source_fields = (
        "vertical_grf_impulse_ns_per_m",
        "vertical_force_p99_n",
        "vertical_force_peak_n",
        "positive_vertical_loading_rate_p99_nps",
        "positive_vertical_loading_rate_peak_nps",
    )
    fieldnames = (
        "robot",
        "condition",
        "mirror_coeff",
        "value_coeff",
        "warmup_iterations",
        "leg_index",
        "leg",
        "pair",
        *metric_fields,
        *(f"{field}_change_vs_same_foot_percent" for field in change_source_fields),
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            leg_comparisons = None if not run.trs_enabled else comparisons[run.key]["durability"]["legs"]
            for leg in rollout_results[run.key]["durability"]["legs"]:
                row = {
                    "robot": run.robot,
                    "condition": run.condition,
                    "mirror_coeff": run.mirror_coeff,
                    "value_coeff": run.value_coeff,
                    "warmup_iterations": run.warmup_iterations,
                    "leg_index": leg["leg_index"],
                    "leg": leg["leg"],
                    "pair": leg["pair"],
                    **{field: leg[field] for field in metric_fields},
                }
                if leg_comparisons is not None:
                    leg_comparison = leg_comparisons[leg["leg_index"]]
                    for field in change_source_fields:
                        row[f"{field}_change_vs_same_foot_percent"] = leg_comparison["changes_percent"][field]
                writer.writerow(row)


def write_rainflow_csv(
    path: Path,
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
) -> None:
    """Write the unaggregated per-joint rainflow cycle spectrum."""
    fieldnames = (
        "robot",
        "condition",
        "mirror_coeff",
        "value_coeff",
        "warmup_iterations",
        "joint_index",
        "joint_name",
        "leg",
        "pair",
        "effort_limit_nm",
        "torque_range_nm",
        "mean_torque_nm",
        "cycle_amplitude_nm",
        "normalized_cycle_amplitude",
        "cycle_count",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for joint in rollout_results[run.key]["durability"]["joints"]:
                for cycle_range, mean_torque, cycle_count in joint["rainflow_spectrum"]:
                    writer.writerow(
                        {
                            "robot": run.robot,
                            "condition": run.condition,
                            "mirror_coeff": run.mirror_coeff,
                            "value_coeff": run.value_coeff,
                            "warmup_iterations": run.warmup_iterations,
                            "joint_index": joint["joint_index"],
                            "joint_name": joint["joint_name"],
                            "leg": joint["leg"],
                            "pair": joint["pair"],
                            "effort_limit_nm": joint["effort_limit_nm"],
                            "torque_range_nm": cycle_range,
                            "mean_torque_nm": mean_torque,
                            "cycle_amplitude_nm": 0.5 * cycle_range,
                            "normalized_cycle_amplitude": (0.5 * cycle_range / joint["effort_limit_nm"]),
                            "cycle_count": cycle_count,
                        }
                    )


def write_training_csv(
    path: Path,
    runs: Sequence[RunSpec],
    training_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Write one-row-per-run training-efficiency results."""
    fieldnames = (
        "robot",
        "condition",
        "mirror_coeff",
        "value_coeff",
        "warmup_iterations",
        "reward_auc",
        "reward_auc_change_vs_no_trs_percent",
        "reward_auc_first_10000",
        "reward_auc_first_10000_change_vs_no_trs_percent",
        "final_reward_mean",
        "final_reward_change_vs_no_trs",
        "reward_35_iteration",
        "reward_35_transition_change_vs_no_trs_percent",
        "straight_reward_1p5_iteration",
        "straight_reward_1p5_transition_change_vs_no_trs_percent",
        "final_velocity_error_xy_mps",
        "throughput_transitions_per_second",
        "throughput_change_vs_no_trs_percent",
        "learning_seconds_per_iteration",
        "learning_time_change_vs_no_trs_percent",
        "collection_seconds_per_iteration",
        "collection_time_change_vs_no_trs_percent",
        "full_run_wall_time_hours",
        "full_run_wall_time_change_vs_no_trs_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            training = training_results[run.key]
            comparison = None if not run.trs_enabled else comparisons[run.key]["training"]
            reward_35 = training["thresholds"]["reward_35"]
            straight_1p5 = training["thresholds"]["straight_reward_1p5"]
            reward_35_comparison = None if comparison is None else comparison["thresholds"]["reward_35"]
            straight_comparison = None if comparison is None else comparison["thresholds"]["straight_reward_1p5"]
            writer.writerow(
                {
                    "robot": run.robot,
                    "condition": run.condition,
                    "mirror_coeff": run.mirror_coeff,
                    "value_coeff": run.value_coeff,
                    "warmup_iterations": run.warmup_iterations,
                    "reward_auc": training["mean_reward"]["auc_all"],
                    "reward_auc_change_vs_no_trs_percent": (
                        None if comparison is None else comparison["reward_auc_change_percent"]
                    ),
                    "reward_auc_first_10000": training["mean_reward"]["auc_first_10000"],
                    "reward_auc_first_10000_change_vs_no_trs_percent": (
                        None if comparison is None else comparison["reward_auc_first_10000_change_percent"]
                    ),
                    "final_reward_mean": training["mean_reward"]["last_1000_mean"],
                    "final_reward_change_vs_no_trs": (
                        None if comparison is None else comparison["final_reward_change"]
                    ),
                    "reward_35_iteration": None if reward_35 is None else reward_35["iteration"],
                    "reward_35_transition_change_vs_no_trs_percent": (
                        None if reward_35_comparison is None else reward_35_comparison["transition_change_percent"]
                    ),
                    "straight_reward_1p5_iteration": None if straight_1p5 is None else straight_1p5["iteration"],
                    "straight_reward_1p5_transition_change_vs_no_trs_percent": (
                        None if straight_comparison is None else straight_comparison["transition_change_percent"]
                    ),
                    "final_velocity_error_xy_mps": training["velocity_error_xy"]["last_1000_mean"],
                    "throughput_transitions_per_second": training["throughput_fps"]["mean"],
                    "throughput_change_vs_no_trs_percent": (
                        None if comparison is None else comparison["throughput_change_percent"]
                    ),
                    "learning_seconds_per_iteration": training["timing_seconds_per_iteration"]["learning_mean"],
                    "learning_time_change_vs_no_trs_percent": (
                        None if comparison is None else comparison["learning_time_change_percent"]
                    ),
                    "collection_seconds_per_iteration": training["timing_seconds_per_iteration"]["collection_mean"],
                    "collection_time_change_vs_no_trs_percent": (
                        None if comparison is None else comparison["collection_time_change_percent"]
                    ),
                    "full_run_wall_time_hours": training["wall_time_hours"],
                    "full_run_wall_time_change_vs_no_trs_percent": (
                        None if comparison is None else comparison["full_run_wall_time_change_percent"]
                    ),
                }
            )


def summarize_robot_grid(
    runs: Sequence[RunSpec],
    robot: str,
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize how consistently the grid supports each hypothesis."""
    robot_runs = [run for run in runs if run.robot == robot and run.trs_enabled]
    leg_counts = {}
    for metric_name in PRIMARY_LEG_METRICS:
        changes = [comparisons[run.key]["leg_metrics"][metric_name]["abs_imbalance_change_pp"] for run in robot_runs]
        leg_counts[metric_name] = {
            "improved_cells": sum(change < 0.0 for change in changes),
            "equivalent_within_5pp_cells": sum(abs(change) <= EQUIVALENCE_MARGIN_PP for change in changes),
            "median_change_pp": statistics.median(changes),
            "range_change_pp": [min(changes), max(changes)],
        }
    all_primary_improved = sum(
        all(
            comparisons[run.key]["leg_metrics"][metric_name]["abs_imbalance_change_pp"] < 0.0
            for metric_name in PRIMARY_LEG_METRICS
        )
        for run in robot_runs
    )
    all_primary_equivalent = sum(
        all(
            abs(comparisons[run.key]["leg_metrics"][metric_name]["abs_imbalance_change_pp"]) <= EQUIVALENCE_MARGIN_PP
            for metric_name in PRIMARY_LEG_METRICS
        )
        for run in robot_runs
    )
    reward_auc_changes = [comparisons[run.key]["training"]["reward_auc_change_percent"] for run in robot_runs]
    final_reward_changes = [comparisons[run.key]["training"]["final_reward_change"] for run in robot_runs]
    reward_35_changes = [
        comparisons[run.key]["training"]["thresholds"]["reward_35"]
        for run in robot_runs
        if comparisons[run.key]["training"]["thresholds"]["reward_35"] is not None
    ]
    throughput_changes = [comparisons[run.key]["training"]["throughput_change_percent"] for run in robot_runs]
    learning_time_changes = [comparisons[run.key]["training"]["learning_time_change_percent"] for run in robot_runs]
    cost_changes = {
        metric_name: [comparisons[run.key]["cost_per_distance"][metric_name] for run in robot_runs]
        for metric_name in (
            "torque_squared_integral_per_m",
            "absolute_work_j_per_m",
            "positive_work_j_per_m",
            "vertical_grf_impulse_ns_per_m",
        )
    }
    durability_pair_summary = {}
    for metric_name in DURABILITY_PAIR_METRICS:
        metric_comparisons = [comparisons[run.key]["durability"]["pair_metrics"][metric_name] for run in robot_runs]
        balance_changes = [comparison["abs_imbalance_change_pp"] for comparison in metric_comparisons]
        total_changes = [comparison["total_per_m_change_percent"] for comparison in metric_comparisons]
        worst_pair_changes = [comparison["worst_pair_per_m_change_percent"] for comparison in metric_comparisons]
        durability_pair_summary[metric_name] = {
            "more_even_cells": sum(change < 0.0 for change in balance_changes),
            "lower_total_per_m_cells": sum(change < 0.0 for change in total_changes),
            "lower_worst_pair_per_m_cells": sum(change < 0.0 for change in worst_pair_changes),
            "median_abs_imbalance_change_pp": statistics.median(balance_changes),
            "median_total_per_m_change_percent": statistics.median(total_changes),
            "range_total_per_m_change_percent": [min(total_changes), max(total_changes)],
            "median_worst_pair_per_m_change_percent": statistics.median(worst_pair_changes),
            "range_worst_pair_per_m_change_percent": [
                min(worst_pair_changes),
                max(worst_pair_changes),
            ],
        }
    worst_component_summary = {}
    for metric_name in comparisons[robot_runs[0].key]["durability"]["worst_components"]:
        global_changes = [
            comparisons[run.key]["durability"]["worst_components"][metric_name]["global_worst_change_percent"]
            for run in robot_runs
        ]
        same_component_changes = [
            comparisons[run.key]["durability"]["worst_components"][metric_name][
                "change_vs_same_component_baseline_percent"
            ]
            for run in robot_runs
        ]
        worst_component_summary[metric_name] = {
            "lower_global_worst_cells": sum(change < 0.0 for change in global_changes),
            "median_global_worst_change_percent": statistics.median(global_changes),
            "range_global_worst_change_percent": [
                min(global_changes),
                max(global_changes),
            ],
            "lower_vs_same_component_cells": sum(change < 0.0 for change in same_component_changes),
            "median_change_vs_same_component_percent": statistics.median(same_component_changes),
        }
    best_auc_run = max(robot_runs, key=lambda run: comparisons[run.key]["training"]["reward_auc_change_percent"])
    return {
        "primary_leg_metrics_all_improved_cells": all_primary_improved,
        "primary_leg_metrics_all_within_5pp_cells": all_primary_equivalent,
        "leg_metrics": leg_counts,
        "cost_per_distance": {
            metric_name: {
                "reduced_cells": sum(change < 0.0 for change in changes),
                "median_change_percent": statistics.median(changes),
                "range_change_percent": [min(changes), max(changes)],
            }
            for metric_name, changes in cost_changes.items()
        },
        "durability": {
            "pair_metrics": durability_pair_summary,
            "worst_components": worst_component_summary,
        },
        "training": {
            "reward_auc_improved_cells": sum(change > 0.0 for change in reward_auc_changes),
            "reward_auc_median_change_percent": statistics.median(reward_auc_changes),
            "reward_auc_range_change_percent": [min(reward_auc_changes), max(reward_auc_changes)],
            "final_reward_improved_cells": sum(change > 0.0 for change in final_reward_changes),
            "reward_35_transition_improved_cells": sum(
                comparison["transition_change_percent"] < 0.0 for comparison in reward_35_changes
            ),
            "reward_35_comparable_cells": len(reward_35_changes),
            "reward_35_failed_cells": len(robot_runs) - len(reward_35_changes),
            "throughput_improved_cells": sum(change > 0.0 for change in throughput_changes),
            "throughput_median_change_percent": statistics.median(throughput_changes),
            "learning_phase_faster_cells": sum(change < 0.0 for change in learning_time_changes),
            "learning_time_median_change_percent": statistics.median(learning_time_changes),
            "best_reward_auc_cell": best_auc_run.key,
            "best_reward_auc_change_percent": comparisons[best_auc_run.key]["training"]["reward_auc_change_percent"],
        },
    }


def build_report(
    runs: Sequence[RunSpec],
    rollout_results: dict[str, dict[str, Any]],
    training_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    grid_summary: dict[str, dict[str, Any]],
) -> str:
    """Build the human-readable Markdown report."""
    lines = [
        "# TRS 3x3 grid: leg allocation, durability exposure, and training efficiency",
        "",
        "Date: 2026-07-27",
        "",
        "This report compares the full coefficient/warm-up grid against the matched no-TRS",
        "policy for Unitree Go2 and Dobot X1. All comparisons use seed 42, 512",
        "environments, 24 rollout steps per iteration, and 20,000 PPO iterations.",
        "",
        "## Executive conclusion",
        "",
    ]
    go2_summary = grid_summary["go2"]
    x1_summary = grid_summary["x1"]
    durability_endpoints = (
        "joint_normalized_torque_rms",
        "joint_fatigue_proxy_m3_per_m",
        "joint_fatigue_proxy_m5_per_m",
        "joint_absolute_work_per_m",
        "foot_vertical_force_p99",
        "foot_vertical_loading_rate_p99",
    )
    all_durability_endpoints_lower = {
        robot: sum(
            all(
                comparisons[run.key]["durability"]["worst_components"][endpoint]["global_worst_change_percent"] < 0.0
                for endpoint in durability_endpoints
            )
            for run in runs
            if run.robot == robot and run.trs_enabled
        )
        for robot in ("go2", "x1")
    }
    lines.extend(
        [
            "| Hypothesis | Grid evidence | Verdict for this seed and rollout |",
            "| --- | --- | --- |",
            f"| TRS makes Go2 front/hind use more even | All three primary measures improved "
            f"together in {go2_summary['primary_leg_metrics_all_improved_cells']}/9 cells; "
            f"raw torque-squared, work, and vertical-GRF balance each improved in 0/9. "
            f"The cyclic-torque proxy was more even in "
            f"{go2_summary['durability']['pair_metrics']['fatigue_proxy_m3']['more_even_cells']}/9 "
            f"(m=3) and "
            f"{go2_summary['durability']['pair_metrics']['fatigue_proxy_m5']['more_even_cells']}/9 "
            "(m=5). | **Not supported across load mechanisms; contradicted by the three "
            "primary allocation measures** |",
            f"| TRS changes X1 front/hind distribution only a little | All three primary "
            f"changes stayed within an illustrative +/-{EQUIVALENCE_MARGIN_PP:.0f} pp margin in "
            f"{x1_summary['primary_leg_metrics_all_within_5pp_cells']}/9 cells. | "
            "**Not supported as a general equivalence claim** |",
            f"| TRS improves Go2 training efficiency | Reward AUC improved in "
            f"{go2_summary['training']['reward_auc_improved_cells']}/9 cells; "
            f"{go2_summary['training']['reward_35_failed_cells']}/9 TRS cells never achieved "
            "the sustained reward-35 criterion reached by no TRS. | "
            "**Not robust; supported only by the 0.30/0.15, w=500 cell** |",
            f"| TRS improves X1 training efficiency | Reward AUC improved in "
            f"{x1_summary['training']['reward_auc_improved_cells']}/9 cells and reward 35 took "
            "more transitions in every cell. | **Contradicted** |",
            f"| TRS reduces durability-oriented exposure | No cell reduced all six separately "
            f"reported worst-component proxies: Go2 "
            f"{all_durability_endpoints_lower['go2']}/9, X1 "
            f"{all_durability_endpoints_lower['x1']}/9. This is a mechanism-by-mechanism "
            "comparison, not a composite score. | **Not supported as a general claim** |",
            "",
            "There is no grid cell that supports all original hypotheses. The component-level",
            "audit also shows why pair evenness cannot be interpreted as lower break risk:",
            "total exposure or one joint/foot can rise while the pair split becomes more even.",
        ]
    )
    lines.extend(
        [
            "",
            "These are descriptive grid results, not population-level causal estimates: every",
            "cell contains one policy trained with the same seed. Grid cells are hyperparameter",
            "settings, not independent replicates.",
            "",
            "## Metric audit and corrected interpretation",
            "",
            "The earlier pair calculation is numerically correct, but pair aggregation alone is",
            "not a durability ranking. A pair can be perfectly balanced while both pairs are",
            "heavily loaded, or one actuator can be overloaded while the two pair sums match.",
            "The revised analysis therefore keeps three orthogonal questions separate:",
            "",
            "1. **Allocation:** front/hind imbalance and dominant-pair ratio.",
            "2. **Amount:** total exposure per second and per commanded-direction metre.",
            "3. **Localization:** worst individual joint/foot, peaks, capacity dwell, and cyclic",
            "   load spectrum.",
            "",
            "- `sum((torque / effort_limit)^2)` measures **normalized torque-capacity",
            "  utilization**, not motor current or copper heating. A true heating metric needs",
            "  each motor's torque constant, winding resistance, transmission efficiency, and",
            "  thermal model. The configured effort limits are simulator torque clamps, not",
            "  continuous thermal ratings or fatigue strengths.",
            "- Raw **torque-squared exposure**, RMS utilization, p95/p99/peak utilization, and",
            "  time at or above 50%, 75%, 90%, and 99% of the configured limit are computed",
            "  independently for all 12 joints before aggregation.",
            "- Torque histories are cycle-counted per joint with an ASTM E1049-style rainflow",
            "  stack. The reported m=3 and m=5 quantities are",
            "  `sum(count * (cycle_amplitude / effort_limit)^m)` per metre. They are low/high",
            "  peak-sensitivity descriptors, **not** Miner damage, lifetime, or failure",
            "  probability without component S-N curves and a stress/mean-load model.",
            "- Euclidean GRF is retained, but **vertical GRF impulse** is the cleaner support-load",
            "  metric. Per-foot p99/peak vertical force and positive finite-difference loading",
            "  rate expose localized impacts, while their 50 Hz sampling is acknowledged.",
            "- Absolute mechanical work is retained and split into positive and negative work.",
            "  Work measures drivetrain energy throughput, not fatigue, and static torque can be",
            "  damaging even when work is zero.",
            "- Mission-normalized quantities divide by endpoint displacement projected onto the",
            "  mean commanded direction. This avoids making lateral wandering look efficient;",
            "  cumulative path length and lateral drift are retained for validation.",
            "- Pair imbalance remains `100 * (front - hind) / (front + hind)`. Its magnitude is",
            "  exactly twice the deviation of the front share from 50%, so it is already the",
            "  simplest well-scaled two-pair evenness metric. Dominance ratio",
            "  `max(front, hind) / min(front, hind)` gives the same allocation in ratio form.",
            "- Heterogeneous endpoints are not averaged into an arbitrary damage score.",
            "",
            "Cycle counting follows the scope of [ASTM E1049](https://store.astm.org/standards/e1049).",
            "The exposure-preserving moment-per-distance treatment is analogous to the load",
            "spectrum workflow used in the [NREL mechanical-loads report]",
            "(https://www.nrel.gov/docs/fy15osti/63679.pdf), but no material calibration is",
            "claimed here.",
            "",
            "## Front/hind balance",
            "",
            "Every cell below is the change in absolute imbalance relative to no TRS",
            "(percentage points). Negative is more even; positive is less even.",
            "",
            "| Robot | Allocation proxy | No-TRS dominant pair (ratio) | "
            "TRS front / hind dominant cells | TRS dominance-ratio range |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    allocation_titles = {
        "torque_squared_exposure": "Torque-squared exposure",
        "absolute_work": "Absolute work",
        "vertical_grf_impulse": "Vertical GRF impulse",
        "fatigue_proxy_m3": "Cyclic-torque severity m=3",
        "fatigue_proxy_m5": "Cyclic-torque severity m=5",
    }
    for robot in ("go2", "x1"):
        baseline = next(run for run in runs if run.robot == robot and not run.trs_enabled)
        robot_trs_runs = [run for run in runs if run.robot == robot and run.trs_enabled]
        for metric_name, metric_title in allocation_titles.items():
            baseline_metric = rollout_results[baseline.key]["durability"]["pair_metrics"][metric_name]
            trs_metrics = [
                rollout_results[run.key]["durability"]["pair_metrics"][metric_name] for run in robot_trs_runs
            ]
            ratios = [metric["dominant_to_other_ratio"] for metric in trs_metrics]
            lines.append(
                f"| {robot.upper() if robot == 'x1' else 'Go2'} | {metric_title} | "
                f"{baseline_metric['dominant_pair']} "
                f"({baseline_metric['dominant_to_other_ratio']:.3f}x) | "
                f"{sum(metric['dominant_pair'] == 'front' for metric in trs_metrics)} / "
                f"{sum(metric['dominant_pair'] == 'hind' for metric in trs_metrics)} | "
                f"{min(ratios):.3f}x to {max(ratios):.3f}x |"
            )
    lines.append("")
    metric_titles = {
        "torque_squared": "Raw torque-squared exposure",
        "absolute_work": "Absolute mechanical work",
        "vertical_grf_impulse": "Vertical GRF impulse",
        "normalized_torque_utilization": "Normalized torque-capacity utilization",
    }
    for robot in ("go2", "x1"):
        robot_name = robot.upper() if robot == "x1" else "Go2"
        lines.extend([f"### {robot_name}", ""])
        for metric_name in (*PRIMARY_LEG_METRICS, "normalized_torque_utilization"):
            lines.extend(
                [
                    f"**{metric_titles[metric_name]}: change in absolute imbalance [pp]**",
                    "",
                    *markdown_grid(
                        runs,
                        robot,
                        lambda run, name=metric_name: comparisons[run.key]["leg_metrics"][name][
                            "abs_imbalance_change_pp"
                        ],
                        signed=True,
                        suffix=" pp",
                    ),
                    "",
                ]
            )
        for exponent in FATIGUE_SENSITIVITY_EXPONENTS:
            metric_name = f"fatigue_proxy_m{exponent}"
            lines.extend(
                [
                    f"**Cyclic-torque severity m={exponent}: change in absolute imbalance [pp]**",
                    "",
                    *markdown_grid(
                        runs,
                        robot,
                        lambda run, name=metric_name: comparisons[run.key]["durability"]["pair_metrics"][name][
                            "abs_imbalance_change_pp"
                        ],
                        signed=True,
                        suffix=" pp",
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Total load and cost per distance",
            "",
            "Balance and total cost are separate outcomes. The tables below normalize each",
            "rollout's exposure by progress projected onto the commanded planar direction and",
            "show percent change from no TRS. Negative is lower total exposure or cost.",
            "",
        ]
    )
    for robot in ("go2", "x1"):
        robot_name = robot.upper() if robot == "x1" else "Go2"
        cost_summary = grid_summary[robot]["cost_per_distance"]
        lines.extend(
            [
                f"### {robot_name}",
                "",
                f"Torque-squared exposure per metre fell in "
                f"{cost_summary['torque_squared_integral_per_m']['reduced_cells']}/9 cells; "
                f"absolute work per metre fell in "
                f"{cost_summary['absolute_work_j_per_m']['reduced_cells']}/9.",
                "",
                "**Torque-squared exposure per metre: change vs no TRS [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["cost_per_distance"]["torque_squared_integral_per_m"],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Absolute mechanical work per metre: change vs no TRS [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["cost_per_distance"]["absolute_work_j_per_m"],
                    signed=True,
                    suffix="%",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Durability-oriented component results",
            "",
            "Negative percentages below mean lower observed exposure than no TRS. A lower value",
            "is evidence only for that mechanism; agreement across unrelated mechanisms is",
            "reported rather than forced through a composite score.",
            "",
            "| Robot | Pair-additive proxy | Lower total / directed m | "
            "Lower worst pair / directed m | More even pair split |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    durability_pair_titles = {
        "torque_squared_exposure": "Torque-squared exposure",
        "fatigue_proxy_m3": "Cyclic-torque severity m=3",
        "fatigue_proxy_m5": "Cyclic-torque severity m=5",
        "absolute_work": "Absolute work",
        "vertical_grf_impulse": "Vertical GRF impulse",
    }
    for robot in ("go2", "x1"):
        robot_name = robot.upper() if robot == "x1" else "Go2"
        for metric_name, metric_title in durability_pair_titles.items():
            summary = grid_summary[robot]["durability"]["pair_metrics"][metric_name]
            lines.append(
                f"| {robot_name} | {metric_title} | "
                f"{summary['lower_total_per_m_cells']}/9 | "
                f"{summary['lower_worst_pair_per_m_cells']}/9 | "
                f"{summary['more_even_cells']}/9 |"
            )
    lines.extend(
        [
            "",
            "| Robot | Worst-component proxy | Lower worst component | Median change in worst component |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    worst_component_titles = {
        "joint_normalized_torque_rms": "Joint RMS torque-limit utilization",
        "joint_normalized_torque_p99": "Joint p99 torque-limit utilization",
        "joint_fatigue_proxy_m3_per_m": "Joint cyclic-torque severity m=3 / m",
        "joint_fatigue_proxy_m5_per_m": "Joint cyclic-torque severity m=5 / m",
        "joint_absolute_work_per_m": "Joint absolute work / m",
        "foot_vertical_force_p99": "Foot p99 vertical force",
        "foot_vertical_loading_rate_p99": "Foot p99 vertical loading rate",
    }
    for robot in ("go2", "x1"):
        robot_name = robot.upper() if robot == "x1" else "Go2"
        for metric_name, metric_title in worst_component_titles.items():
            summary = grid_summary[robot]["durability"]["worst_components"][metric_name]
            lines.append(
                f"| {robot_name} | {metric_title} | "
                f"{summary['lower_global_worst_cells']}/9 | "
                f"{summary['median_global_worst_change_percent']:+.1f}% |"
            )
    lines.extend(
        [
            "",
            "The pair and component conclusions can differ. For example, X1 absolute-work",
            "allocation became more even in every cell, yet the worst-pair work fell in only",
            "three cells and the worst-joint work fell in three. Go2 cyclic-torque severity",
            "improved in six cells for both m=3 and m=5, but worst-joint RMS utilization fell",
            "in only one cell, worst-joint work in none, and worst-foot p99 loading rate in",
            "none.",
            "",
        ]
    )
    for robot in ("go2", "x1"):
        robot_name = robot.upper() if robot == "x1" else "Go2"
        baseline = next(run for run in runs if run.robot == robot and not run.trs_enabled)
        baseline_saturation_dwell_percent = 100.0 * max(
            joint["dwell_fraction_at_or_above_0p99"] for joint in rollout_results[baseline.key]["durability"]["joints"]
        )
        lines.extend(
            [
                f"### {robot_name} durability grids",
                "",
                "**Cyclic-torque severity m=3, total per directed metre: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["pair_metrics"]["fatigue_proxy_m3"][
                        "total_per_m_change_percent"
                    ],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Cyclic-torque severity m=5, total per directed metre: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["pair_metrics"]["fatigue_proxy_m5"][
                        "total_per_m_change_percent"
                    ],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Worst-joint cyclic-torque severity m=5 per directed metre: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["worst_components"]["joint_fatigue_proxy_m5_per_m"][
                        "global_worst_change_percent"
                    ],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Worst-joint RMS torque-limit utilization: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["worst_components"]["joint_normalized_torque_rms"][
                        "global_worst_change_percent"
                    ],
                    signed=True,
                    suffix="%",
                ),
                "",
                f"**Maximum any-joint dwell at >=99% of configured effort limit "
                f"[% of window]; no-TRS={baseline_saturation_dwell_percent:.1f}%**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: 100.0
                    * max(
                        joint["dwell_fraction_at_or_above_0p99"]
                        for joint in rollout_results[run.key]["durability"]["joints"]
                    ),
                    signed=False,
                    suffix="%",
                ),
                "",
                "**Worst-joint absolute work per directed metre: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["worst_components"]["joint_absolute_work_per_m"][
                        "global_worst_change_percent"
                    ],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Worst-foot p99 vertical force: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["worst_components"]["foot_vertical_force_p99"][
                        "global_worst_change_percent"
                    ],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Worst-foot p99 positive vertical loading rate: change [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["durability"]["worst_components"][
                        "foot_vertical_loading_rate_p99"
                    ]["global_worst_change_percent"],
                    signed=True,
                    suffix="%",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "### Worst-component localization by grid cell",
            "",
            "The component name is as important as the pair sum. Parentheses show the absolute",
            "RMS utilization or the change in the current global maximum versus the no-TRS",
            "global maximum.",
            "",
            "| Robot | Mirror/value, warm-up | Worst RMS-utilization joint | "
            "Worst m=5 cyclic-severity joint | Worst p99-impact foot |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for run in runs:
        if not run.trs_enabled:
            continue
        rollout_worst = rollout_results[run.key]["durability"]["worst_components"]
        comparison_worst = comparisons[run.key]["durability"]["worst_components"]
        rms = rollout_worst["joint_normalized_torque_rms"]
        fatigue = comparison_worst["joint_fatigue_proxy_m5_per_m"]
        impact = comparison_worst["foot_vertical_force_p99"]
        lines.append(
            f"| {run.robot.upper() if run.robot == 'x1' else 'Go2'} | "
            f"{run.mirror_coeff:.2f}/{run.value_coeff:.2f}, w={run.warmup_iterations} | "
            f"`{rms['component']}` ({rms['value']:.3f}) | "
            f"`{fatigue['component']}` "
            f"({fatigue['global_worst_change_percent']:+.1f}%) | "
            f"`{impact['component']}` ({impact['global_worst_change_percent']:+.1f}%) |"
        )
    lines.append("")

    lines.extend(
        [
            "## Training efficiency",
            "",
            "Reward AUC is the mean return over the fixed 245.76-million-transition",
            "budget; it is the primary threshold-free sample-efficiency measure. The",
            "reward-35 metric uses a trailing 200-iteration mean and requires the next 500",
            "iterations to remain above threshold. PPO learning time isolates the optimization",
            "phase, but all wall-clock and throughput results remain secondary because machine",
            "load and paired Go2/X1 GPU contention differed between scan dates.",
            "",
            "The TensorBoard exports show the same 200-iteration trailing-mean reward",
            "curves against environment transitions (left, primary evidence) and observed",
            "wall time (right, secondary evidence). Color identifies the mirror/value pair;",
            "line style identifies the warm-up. The black curve is the matched no-TRS",
            "baseline.",
            "",
        ]
    )
    for robot in ("go2", "x1"):
        robot_name = robot.upper() if robot == "x1" else "Go2"
        training_summary = grid_summary[robot]["training"]
        lines.extend(
            [
                f"### {robot_name}",
                "",
                f"![{robot_name} TensorBoard training-efficiency curves](tensorboard_reward_efficiency_{robot}.svg)",
                "",
                f"Final reward improved in {training_summary['final_reward_improved_cells']}/9 "
                f"cells. The sustained reward-35 target was reached in "
                f"{training_summary['reward_35_comparable_cells']}/9 cells and missed in "
                f"{training_summary['reward_35_failed_cells']}/9. The PPO learning phase was "
                f"faster in {training_summary['learning_phase_faster_cells']}/9 cells.",
                "",
                "**Reward AUC change vs no TRS [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["training"]["reward_auc_change_percent"],
                    signed=True,
                    suffix="%",
                ),
                "",
                "**Transitions to sustained reward 35: change vs no TRS [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: (
                        None
                        if comparisons[run.key]["training"]["thresholds"]["reward_35"] is None
                        else comparisons[run.key]["training"]["thresholds"]["reward_35"]["transition_change_percent"]
                    ),
                    signed=True,
                    suffix="%",
                ),
                "",
                "**PPO learning time per iteration: change vs no TRS [%]**",
                "",
                *markdown_grid(
                    runs,
                    robot,
                    lambda run: comparisons[run.key]["training"]["learning_time_change_percent"],
                    signed=True,
                    suffix="%",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Baseline scales and validation",
            "",
            "| Robot | Metric | No-TRS signed imbalance | No-TRS front share |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for robot in ("go2", "x1"):
        baseline = next(run for run in runs if run.robot == robot and not run.trs_enabled)
        for metric_name in (*PRIMARY_LEG_METRICS, "normalized_torque_utilization"):
            metric = rollout_results[baseline.key]["metrics"][metric_name]
            lines.append(
                f"| {robot.upper() if robot == 'x1' else 'Go2'} | {metric_titles[metric_name]} | "
                f"{metric['signed_imbalance_percent']:+.1f}% | {metric['front_share_percent']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "| Robot | Directed progress [m] | Path length [m] | Lateral drift [m] | "
            "Tracking RMSE [m/s] | Absolute work [J/m] | "
            "Torque-squared exposure [N^2 m^2 s/m] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for robot in ("go2", "x1"):
        baseline = next(run for run in runs if run.robot == robot and not run.trs_enabled)
        rollout = rollout_results[baseline.key]
        lines.append(
            f"| {robot.upper() if robot == 'x1' else 'Go2'} | "
            f"{rollout['directed_progress_m']:.3f} | {rollout['actual_distance_m']:.3f} | "
            f"{rollout['lateral_drift_m']:.3f} | "
            f"{rollout['planar_tracking_rmse_mps']:.4f} | "
            f"{rollout['cost_per_distance']['absolute_work_j_per_m']:.2f} | "
            f"{rollout['cost_per_distance']['torque_squared_integral_per_m']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Robot | No-TRS worst component metric | Component | Value |",
            "| --- | --- | --- | ---: |",
        ]
    )
    baseline_component_units = {
        "joint_normalized_torque_rms": ("RMS torque-limit utilization", ""),
        "joint_normalized_torque_p99": ("p99 torque-limit utilization", ""),
        "joint_fatigue_proxy_m3_per_m": ("Cyclic severity m=3 / directed m", "1/m"),
        "joint_fatigue_proxy_m5_per_m": ("Cyclic severity m=5 / directed m", "1/m"),
        "joint_absolute_work_per_m": ("Absolute work / directed m", "J/m"),
        "foot_vertical_force_p99": ("Foot p99 vertical force", "N"),
        "foot_vertical_loading_rate_p99": ("Foot p99 vertical loading rate", "N/s"),
    }
    for robot in ("go2", "x1"):
        baseline = next(run for run in runs if run.robot == robot and not run.trs_enabled)
        for metric_name, (metric_title, unit) in baseline_component_units.items():
            worst = rollout_results[baseline.key]["durability"]["worst_components"][metric_name]
            suffix = f" {unit}" if unit else ""
            lines.append(
                f"| {robot.upper() if robot == 'x1' else 'Go2'} | {metric_title} | "
                f"`{worst['component']}` | {worst['value']:.4f}{suffix} |"
            )
    power_errors = [
        rollout_results[run.key]["validation"]["max_abs_power_identity_error_w"]
        for run in runs
        if rollout_results[run.key]["validation"]["max_abs_power_identity_error_w"] is not None
    ]
    max_power_error = max(power_errors)
    max_force_error = max(rollout_results[run.key]["validation"]["max_abs_force_norm_identity_error_n"] for run in runs)
    command_values = [rollout_results[run.key]["command"]["x_mean_mps"] for run in runs]
    lines.extend(
        [
            "",
            f"- All {len(runs)} evaluations contain {rollout_results[runs[0].key]['sample_count']} matched",
            f"  samples over `{WINDOW_START_S} <= t < {WINDOW_STOP_S} s`, with no reset in",
            f"  the window and a common backward sagittal command of {statistics.fmean(command_values):.6f} m/s.",
            "- Joint names validate the stored front-left, front-right, rear-left, rear-right",
            "  leg-major ordering in every archive.",
            f"- In {len(power_errors)}/{len(runs)} archives that retain joint velocity, recomputed",
            f"  `torque * joint_velocity` agrees with recorded power to at most {max_power_error:.3e} W;",
            f"  recomputed GRF norms agree in all archives to at most {max_force_error:.3e} N.",
            "- All selected GRF samples include friction; vertical impulse is recomputed from",
            "  the raw world-frame z component.",
            "",
            "## Evidence limits and next experiment",
            "",
            "The scan answers whether this seed is robust to the chosen TRS hyperparameters,",
            "but it cannot estimate the expected TRS effect. The within-rollout block bootstrap",
            "describes temporal variability in one trace; it does not make the 450 samples",
            "independent policies and does not cover training-seed uncertainty.",
            "",
            "The saved evaluations contain one reset-free nine-second backward-command window",
            "at 50 Hz. They do not resolve physics-substep impact peaks, long thermal time",
            "constants, rare events, turning, forward gait, terrain, or payload effects.",
            "Rainflow residue is retained as half cycles, but mean-load correction is omitted",
            "because material ultimate strength is unavailable.",
            "",
            "A follow-up durability study should use paired independent training seeds and",
            "repeated matched rollouts over forward/backward motion, turning, terrain, payload,",
            "and longer duty cycles. Predeclare component-level endpoints and the X1",
            "equivalence margin. Actual break probability or lifetime additionally requires",
            "motor electrical/thermal parameters, gearbox and bearing ratings, geometry and",
            "materials, stress conversion, component S-N/load-life curves, and failure labels.",
            "",
            "Machine-readable details are in `summary.json`, `leg_usage.csv`,",
            "`durability_pair_metrics.csv`, `joint_durability.csv`, `foot_impact.csv`,",
            "`rainflow_cycles.csv`, and `training_efficiency.csv` beside this report. The",
            "two `tensorboard_reward_efficiency_*.svg` files are generated directly from",
            "the raw TensorBoard event logs with:",
            "",
            "```powershell",
            r".\isaaclab.bat -p scripts\symm_locomotion\plot_trs_tensorboard.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the full grid analysis and save its report and tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOG_ROOT / "good_runs/trs_grid_analysis",
        help="Directory for the Markdown, JSON, and CSV outputs.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Use 100 bootstrap replicates for a fast smoke run.",
    )
    args = parser.parse_args()
    global BOOTSTRAP_REPLICATES
    if args.skip_bootstrap:
        BOOTSTRAP_REPLICATES = 100

    runs = discover_runs()
    rollout_results = {}
    training_results = {}
    for index, run in enumerate(runs, start=1):
        print(f"[{index:02d}/{len(runs)}] {run.key}: rollout", flush=True)
        rollout_results[run.key] = analyze_rollout(run)
        print(f"[{index:02d}/{len(runs)}] {run.key}: training", flush=True)
        training_results[run.key] = analyze_training(run)
    comparisons = compare_to_baselines(runs, rollout_results, training_results)
    grid_summary = {robot: summarize_robot_grid(runs, robot, comparisons) for robot in ROBOT_RUN_DIRS}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_runs = {
        run.key: {
            "robot": run.robot,
            "condition": run.condition,
            "run_path": str(run.run_path.relative_to(ROOT)),
            "evaluation_path": str(run.evaluation_path.relative_to(ROOT)),
            "mirror_coeff": run.mirror_coeff,
            "value_coeff": run.value_coeff,
            "warmup_iterations": run.warmup_iterations,
            "trs_enabled": run.trs_enabled,
        }
        for run in runs
    }
    summary = {
        "analysis": {
            "window_s": [WINDOW_START_S, WINDOW_STOP_S],
            "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_block_duration_s": BLOCK_DURATION_S,
            "samples_per_iteration": SAMPLES_PER_ITERATION,
            "equivalence_margin_pp_for_descriptive_count_only": EQUIVALENCE_MARGIN_PP,
            "metric_units": METRIC_UNITS,
            "durability_method_version": 1,
            "distance_normalization": (
                "absolute endpoint displacement projected onto the mean commanded planar direction"
            ),
            "normalized_torque_dwell_thresholds": NORMALIZED_TORQUE_DWELL_THRESHOLDS,
            "fatigue_sensitivity_exponents": FATIGUE_SENSITIVITY_EXPONENTS,
            "fatigue_proxy_definition": ("sum(rainflow_count * (torque_range / (2 * configured_effort_limit)) ** m)"),
            "fatigue_interpretation": (
                "uncalibrated effort-limit-normalized load-spectrum severity, not Miner damage, "
                "lifetime, or failure probability"
            ),
            "rainflow_method": ("ASTM E1049-style three-point stack; finite-record residues retained as half cycles"),
            "impact_interpretation": (
                "50 Hz world-frame vertical GRF and positive finite-difference loading-rate proxy"
            ),
        },
        "runs": serializable_runs,
        "rollouts": rollout_results,
        "training": training_results,
        "comparisons_to_no_trs": comparisons,
        "grid_summary": grid_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_leg_csv(output_dir / "leg_usage.csv", runs, rollout_results, comparisons)
    write_durability_pair_csv(
        output_dir / "durability_pair_metrics.csv",
        runs,
        rollout_results,
        comparisons,
    )
    write_joint_durability_csv(
        output_dir / "joint_durability.csv",
        runs,
        rollout_results,
        comparisons,
    )
    write_foot_impact_csv(
        output_dir / "foot_impact.csv",
        runs,
        rollout_results,
        comparisons,
    )
    write_rainflow_csv(output_dir / "rainflow_cycles.csv", runs, rollout_results)
    write_training_csv(output_dir / "training_efficiency.csv", runs, training_results, comparisons)
    report = build_report(runs, rollout_results, training_results, comparisons, grid_summary)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Saved analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
