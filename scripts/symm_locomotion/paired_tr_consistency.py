# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-aligned paired forward/backward time-reversal diagnostics.

The diagnostics in this module compare independently recorded positive- and
negative-command rollouts.  They are mechanism probes, not a proof that the
simulated or physical system is exactly reversible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD_VERSION = "paired_tr_consistency_phase_v1"
OPERATORS = ("temporal_reversal", "front_hind_reflected_temporal_reversal")
METRICS = (
    "paired_velocity_bias",
    "paired_yaw_bias",
    "paired_contact_pattern_error",
    "paired_touchdown_phase_error",
    "paired_joint_position_error",
    "paired_joint_velocity_error",
    "paired_actor_mean_consistency_error",
    "paired_normalized_target_consistency_error",
    "paired_value_consistency_error",
    "paired_torque_pattern_error",
    "paired_work_pattern_error",
    "paired_grf_pattern_error",
)
METRIC_UNITS = {
    "paired_velocity_bias": "m/s",
    "paired_yaw_bias": "rad/s",
    "paired_contact_pattern_error": "fraction",
    "paired_touchdown_phase_error": "cycle",
    "paired_joint_position_error": "rad",
    "paired_joint_velocity_error": "rad/s",
    "paired_actor_mean_consistency_error": "raw_action",
    "paired_normalized_target_consistency_error": "normalized_soft_limit_range",
    "paired_value_consistency_error": "critic_value",
    "paired_torque_pattern_error": "N*m",
    "paired_work_pattern_error": "W",
    "paired_grf_pattern_error": "N",
}
_FRONT_HIND_LEG_PERMUTATION = np.asarray((2, 3, 0, 1), dtype=np.int64)
_ROBOT_JOINT_REFLECTION_SIGNS = {
    "go2": np.asarray((1.0, 1.0, 1.0), dtype=np.float64),
    # X1 uses opposite sagittal joint coordinates for front and rear legs.
    "x1": np.asarray((1.0, -1.0, -1.0), dtype=np.float64),
}
_ROBOT_JOINT_NAMES = {
    "go2": tuple(f"{leg}_{role}_joint" for leg in ("FL", "FR", "RL", "RR") for role in ("hip", "thigh", "calf")),
    "x1": tuple(
        f"joint_{leg}_{role}"
        for leg in ("front_left", "front_right", "rear_left", "rear_right")
        for role in ("abad", "thigh_pitch", "calf_pitch")
    ),
}
_ROBOT_FOOT_BODY_NAMES = {
    "go2": ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
    "x1": (
        "link_front_left_foot",
        "link_front_right_foot",
        "link_rear_left_foot",
        "link_rear_right_foot",
    ),
}
_LEG_NAMES = ("Front Left", "Front Right", "Rear Left", "Rear Right")
_MOTOR_ROLE_NAMES = ("Hip/Abad", "Thigh", "Calf")
_VALUE_FIELDS = ("critic_values", "value_predictions", "critic_value_predictions")
_NORMALIZED_TARGET_FIELDS = (
    "normalized_requested_joint_targets",
    "requested_joint_targets_normalized",
)
_REQUESTED_TARGET_FIELDS = ("requested_joint_position_targets", "joint_position_targets_requested")
_PAIR_WINDOW_FIELDS = ("step_dt", "settle_steps", "measure_steps")
_PUBLICATION_METHODS = {"leg_usage_grid_full_v3", "leg_usage_grid_light_v2"}


@dataclass(frozen=True)
class _CellTrace:
    """One measurement-window trace copied from a cell archive."""

    phase: np.ndarray
    decision_phase: np.ndarray | None
    swing_ratio: float
    arrays: dict[str, np.ndarray]
    contacts: np.ndarray | None
    touchdown_phases: tuple[np.ndarray, ...] | None
    liftoff_phases: tuple[np.ndarray, ...] | None
    reasons: dict[str, str]
    axis_metadata_reason: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_array(data: Mapping[str, Any], name: str, length: int, shape: tuple[int, ...]) -> np.ndarray | None:
    """Return a finite time-major archive array with the requested trailing shape."""
    if name not in data:
        return None
    try:
        values = np.asarray(data[name], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.shape != (length, *shape) or not np.all(np.isfinite(values)):
        return None
    return values


def _first_array(
    data: Mapping[str, Any], names: Sequence[str], length: int, shape: tuple[int, ...]
) -> tuple[np.ndarray | None, str | None]:
    for name in names:
        values = _finite_array(data, name, length, shape)
        if values is not None:
            return values, name
    return None, None


def _static_or_time_array(data: Mapping[str, Any], name: str, length: int, width: int) -> np.ndarray | None:
    """Return a finite vector broadcast across time or an existing time-major matrix."""
    if name not in data:
        return None
    try:
        values = np.asarray(data[name], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.shape == (width,):
        values = np.broadcast_to(values, (length, width))
    if values.shape != (length, width) or not np.all(np.isfinite(values)):
        return None
    return values


def _scalar_trace(data: Mapping[str, Any], names: Sequence[str], length: int) -> tuple[np.ndarray | None, str | None]:
    """Return a scalar trace stored as either ``(T,)`` or ``(T, 1)``."""
    for name in names:
        values = _finite_array(data, name, length, ())
        if values is not None:
            return values, name
        values = _finite_array(data, name, length, (1,))
        if values is not None:
            return values[:, 0], name
    return None, None


def _yaw_rate(data: Mapping[str, Any], length: int) -> np.ndarray | None:
    for name in ("base_angular_velocities", "true_ang_vel", "base_ang_vel"):
        values = _finite_array(data, name, length, (3,))
        if values is not None:
            return values[:, 2]
    # The established rollout archive stores (vx, vy, yaw_rate) under this
    # Historical field name, matching evaluation.py's fidelity convention.
    planar_velocity = _finite_array(data, "true_lin_vel", length, (3,))
    if planar_velocity is not None:
        return planar_velocity[:, 2]
    headings = _finite_array(data, "base_headings", length, ())
    times = _finite_array(data, "time_steps", length, ())
    if headings is None or times is None or length < 3 or np.any(np.diff(times) <= 0.0):
        return None
    return np.gradient(np.unwrap(headings), times, edge_order=2)


def _normalized_requested_targets(data: Mapping[str, Any], length: int) -> np.ndarray | None:
    direct, _ = _first_array(data, _NORMALIZED_TARGET_FIELDS, length, (12,))
    if direct is not None:
        return direct
    requested, _ = _first_array(data, _REQUESTED_TARGET_FIELDS, length, (12,))
    lower = _static_or_time_array(data, "joint_position_lower_limits", length, 12)
    upper = _static_or_time_array(data, "joint_position_upper_limits", length, 12)
    if requested is None or lower is None or upper is None:
        return None
    ranges = upper - lower
    if np.any(ranges <= 0.0):
        return None
    return (2.0 * requested - (upper + lower)) / ranges


def _hysteresis_contacts(forces_n: np.ndarray, on_n: float, off_n: float) -> np.ndarray:
    if not math.isfinite(on_n) or not math.isfinite(off_n) or not on_n > off_n >= 0.0:
        raise ValueError("Contact thresholds must satisfy finite on_n > off_n >= 0 N.")
    contacts = np.zeros(forces_n.shape, dtype=bool)
    state = forces_n[0] >= on_n
    contacts[0] = state
    for index in range(1, len(forces_n)):
        state = np.where(state, forces_n[index] > off_n, forces_n[index] >= on_n)
        contacts[index] = state
    return contacts


def _event_phases(phase: np.ndarray, contacts: np.ndarray, rising: bool) -> tuple[np.ndarray, ...]:
    events: list[np.ndarray] = []
    for leg_index in range(4):
        previous = contacts[:-1, leg_index]
        current = contacts[1:, leg_index]
        selected = (~previous & current) if rising else (previous & ~current)
        events.append(np.remainder(phase[1:][selected], 1.0))
    return tuple(events)


def _axis_metadata_reason(data: Mapping[str, Any], robot: str) -> str | None:
    """Return why archived joint/foot axes are unsafe for declared reflections."""
    expected = {
        "joint_names": _ROBOT_JOINT_NAMES[robot],
        "foot_body_names": _ROBOT_FOOT_BODY_NAMES[robot],
        "leg_names": _LEG_NAMES,
        "motor_role_names": _MOTOR_ROLE_NAMES,
    }
    for field, expected_values in expected.items():
        if field not in data:
            return f"axis metadata field {field!r} is missing"
        values = tuple(str(value) for value in np.asarray(data[field]).reshape(-1))
        if values != expected_values:
            return f"axis metadata field {field!r} does not match declared {robot} order"
    return None


def _cell_archive_path(study_root: Path, cell: Mapping[str, Any]) -> Path:
    """Resolve a cell archive without allowing paths outside the study."""
    root = study_root.resolve()
    archive_path = (root / str(cell["relative_output_dir"]) / "sim_data.npz").resolve()
    try:
        archive_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Cell archive escapes the study directory: {archive_path}") from exc
    return archive_path


def _publication_archive_identity_reason(
    study: Mapping[str, Any], study_path: Path, cell: Mapping[str, Any]
) -> str | None:
    """Return why an existing publication archive is not bound to its study cell."""
    archive_path = _cell_archive_path(study_path.parent, cell)
    if not archive_path.is_file():
        return "publication archive is missing"
    plan_sha256 = _sha256_file(study_path)
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {
                "protocol_version",
                "plan_sha256",
                "cell_id",
                "gait_index",
                "evaluation_seed",
                "velocity_mps",
                "step_dt",
                "expected_steps",
                "recorded_steps",
                "settle_steps",
                "measure_steps",
                "measurement_start_step",
                "measurement_stop_step",
                "checkpoint_json",
                "common_gait_phases",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                return f"publication archive identity fields are missing: {missing}"

            def scalar(name: str) -> Any:
                return np.asarray(archive[name]).item()

            recorded_steps = len(np.asarray(archive["common_gait_phases"]).reshape(-1))
            settle_steps = int(cell.get("settle_steps", study.get("settle_steps", 0)))
            measure_steps = int(cell.get("measure_steps", study.get("measure_steps", 0)))
            expected_steps = int(cell.get("total_steps", settle_steps + measure_steps))
            step_dt = float(cell.get("step_dt", study.get("step_dt", math.nan)))
            expected_velocity = float(cell.get("velocity_mps", cell.get("vx_mps")))
            try:
                archived_checkpoint = json.loads(str(scalar("checkpoint_json")))
            except (TypeError, ValueError, json.JSONDecodeError):
                return "publication archive checkpoint_json is invalid"
            checks = (
                (str(scalar("protocol_version")) == str(study.get("method_version")), "method identity"),
                (str(scalar("plan_sha256")) == plan_sha256, "study plan SHA-256"),
                (str(scalar("cell_id")) == str(cell.get("id")), "cell identity"),
                (int(scalar("gait_index")) == int(cell.get("gait_index", -1)), "gait identity"),
                (int(scalar("evaluation_seed")) == int(cell.get("seed", 0)), "evaluation seed"),
                (
                    math.isclose(float(scalar("velocity_mps")), expected_velocity, rel_tol=0.0, abs_tol=1.0e-9),
                    "velocity",
                ),
                (math.isclose(float(scalar("step_dt")), step_dt, rel_tol=0.0, abs_tol=1.0e-9), "step_dt"),
                (int(scalar("expected_steps")) == expected_steps, "expected step count"),
                (int(scalar("recorded_steps")) == recorded_steps, "recorded step count"),
                (int(scalar("settle_steps")) == settle_steps, "settle window"),
                (int(scalar("measure_steps")) == measure_steps, "measurement window"),
                (int(scalar("measurement_start_step")) == settle_steps, "measurement start"),
                (int(scalar("measurement_stop_step")) == settle_steps + measure_steps, "measurement stop"),
                (archived_checkpoint == study.get("checkpoint", {}), "checkpoint identity"),
            )
    except (OSError, TypeError, ValueError) as exc:
        return f"publication archive identity cannot be read: {exc}"
    for matched, label in checks:
        if not matched:
            return f"publication archive {label} does not match the declared study cell"

    manifest_path = archive_path.with_name("recording_manifest.json")
    if not manifest_path.is_file():
        return "recording_manifest.json is missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return "recording_manifest.json must contain one object"
        declared_record_sha256 = manifest.pop("record_sha256", None)
        computed_record_sha256 = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        manifest_checks = (
            (declared_record_sha256 == computed_record_sha256, "record SHA-256"),
            (manifest.get("schema_version") == 1, "schema version"),
            (manifest.get("method_version") == study.get("method_version"), "method identity"),
            (manifest.get("plan_sha256") == plan_sha256, "study plan SHA-256"),
            (manifest.get("cell_id") == cell.get("id"), "cell identity"),
            (int(manifest.get("gait_index", -1)) == int(cell.get("gait_index", -2)), "gait identity"),
            (int(manifest.get("evaluation_seed", -1)) == int(cell.get("seed", 0)), "evaluation seed"),
            (int(manifest.get("recorded_steps", -1)) == recorded_steps, "recorded step count"),
            (manifest.get("archive_sha256") == _sha256_file(archive_path), "archive SHA-256"),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"recording manifest cannot be validated: {exc}"
    for matched, label in manifest_checks:
        if not matched:
            return f"recording manifest {label} is invalid"
    return None


def _load_cell_trace(study_root: Path, cell: Mapping[str, Any], robot: str) -> _CellTrace:
    archive_path = _cell_archive_path(study_root, cell)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Cell archive does not exist: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    if "common_gait_phases" not in data:
        raise ValueError(f"Cell archive lacks common_gait_phases: {archive_path}")
    phase = np.asarray(data["common_gait_phases"], dtype=np.float64).reshape(-1)
    if len(phase) < 3 or not np.all(np.isfinite(phase)) or np.any(np.diff(phase) <= 0.0):
        raise ValueError(f"Cell common gait phase must be finite and strictly increasing: {archive_path}")
    start = int(cell.get("settle_steps", 0))
    measure_steps = int(cell.get("measure_steps", len(phase) - start))
    stop = start + measure_steps
    if start < 0 or measure_steps < 2 or stop > len(phase):
        raise ValueError(f"Cell measurement window is invalid: {archive_path}")
    selected = slice(start, stop)
    length = len(phase)

    if "episode_done" in data:
        episode_done = _finite_array(data, "episode_done", length, ())
        if episode_done is None:
            raise ValueError(f"Cell episode_done trace is malformed: {archive_path}")
        if np.any(episode_done[selected] != 0.0):
            raise ValueError(f"Cell terminated during its paired measurement window: {archive_path}")

    duty = _finite_array(data, "duty_factors", length, ())
    if duty is None or np.any((duty <= 0.0) | (duty >= 1.0)):
        raise ValueError(f"Cell archive lacks valid duty factors: {archive_path}")
    selected_swing_ratio = 1.0 - duty[selected]
    if float(np.ptp(selected_swing_ratio)) > 1.0e-5:
        raise ValueError(f"Cell swing ratio varies within its paired measurement window: {archive_path}")
    swing_ratio = float(np.median(selected_swing_ratio))

    arrays: dict[str, np.ndarray] = {}
    reasons: dict[str, str] = {}
    decision_phase = _finite_array(data, "pre_decision_common_gait_phases", length, ())
    if decision_phase is None:
        reasons["decision_phase"] = "pre_decision_common_gait_phases is missing or malformed"
    elif np.any(np.diff(decision_phase) < 0.0):
        reasons["decision_phase"] = "pre-decision common gait phase is not monotonic"
        decision_phase = None
    else:
        decision_phase = decision_phase[selected]
    velocity = _finite_array(data, "true_lin_vel", length, (3,))
    if velocity is None:
        reasons["velocity"] = "true_lin_vel is missing or malformed"
    else:
        arrays["velocity"] = velocity[selected, 0]
    yaw_rate = _yaw_rate(data, length)
    if yaw_rate is None or not np.all(np.isfinite(yaw_rate)):
        reasons["yaw"] = "authentic yaw-rate or heading/time traces are unavailable"
    else:
        arrays["yaw"] = yaw_rate[selected]

    field_specs = {
        "joint_position": (("joint_positions",), (12,)),
        "joint_velocity": (("joint_velocities",), (12,)),
        "actor_mean": (("actor_means",), (12,)),
        "torque": (("joint_torques",), (12,)),
        "power": (("joint_powers",), (12,)),
        "grf": (("foot_ground_reaction_forces_w", "foot_normal_forces_w"), (4, 3)),
    }
    for label, (names, trailing_shape) in field_specs.items():
        values, source = _first_array(data, names, length, trailing_shape)
        if values is None:
            reasons[label] = f"none of {', '.join(names)} is present as a valid trace"
        else:
            arrays[label] = values[selected]
            if label == "grf":
                arrays[label] = arrays[label][..., 2]
            if label == "power":
                arrays[label] = np.abs(arrays[label])
            reasons[f"{label}_source"] = str(source)

    values, source = _scalar_trace(data, _VALUE_FIELDS, length)
    if values is None:
        reasons["value"] = f"none of {', '.join(_VALUE_FIELDS)} is present as a valid trace"
    else:
        arrays["value"] = values[selected]
        reasons["value_source"] = str(source)

    normalized_target = _normalized_requested_targets(data, length)
    if normalized_target is None:
        reasons["normalized_target"] = (
            "an unclipped requested-target trace and its soft limits are unavailable; "
            "executed/clamped targets are deliberately not substituted"
        )
    else:
        arrays["normalized_target"] = normalized_target[selected]

    contacts = None
    touchdown_phases = None
    liftoff_phases = None
    normal_forces = _finite_array(data, "foot_normal_forces_w", length, (4, 3))
    try:
        on_n = float(np.asarray(data["contact_threshold_on_n"]).item())
        off_n = float(np.asarray(data["contact_threshold_off_n"]).item())
    except (KeyError, TypeError, ValueError):
        on_n = off_n = math.nan
    if normal_forces is None:
        reasons["contact"] = "foot_normal_forces_w is missing or malformed"
    elif not on_n > off_n >= 0.0:
        reasons["contact"] = "recorded contact hysteresis thresholds are unavailable"
    else:
        full_contacts = _hysteresis_contacts(np.maximum(normal_forces[..., 2], 0.0), on_n, off_n)
        contacts = full_contacts[selected]
        touchdown_phases = _event_phases(phase[selected], contacts, rising=True)
        liftoff_phases = _event_phases(phase[selected], contacts, rising=False)

    return _CellTrace(
        phase=phase[selected],
        decision_phase=decision_phase,
        swing_ratio=swing_ratio,
        arrays=arrays,
        contacts=contacts,
        touchdown_phases=touchdown_phases,
        liftoff_phases=liftoff_phases,
        reasons=reasons,
        axis_metadata_reason=_axis_metadata_reason(data, robot),
    )


def phase_profile(
    common_phase: np.ndarray,
    values: np.ndarray,
    sample_phases: np.ndarray,
    *,
    nearest: bool = False,
) -> np.ndarray:
    """Average a trace at requested gait phases across all covered cycles.

    Args:
        common_phase: Monotonic unwrapped common phase [cycle], shape ``(T,)``.
        values: Time-major samples, shape ``(T, ...)``.
        sample_phases: Requested wrapped phases [cycle], shape ``(P,)``.
        nearest: Use nearest-neighbor samples instead of linear interpolation.

    Returns:
        Cycle-averaged samples with shape ``(P, ...)``.
    """
    common_phase = np.asarray(common_phase, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sample_phases = np.remainder(np.asarray(sample_phases, dtype=np.float64), 1.0)
    if common_phase.ndim != 1 or values.shape[0] != len(common_phase) or sample_phases.ndim != 1:
        raise ValueError("Phase, value, and requested-phase shapes are incompatible.")
    if len(common_phase) < 2 or np.any(np.diff(common_phase) <= 0.0):
        raise ValueError("Common gait phase must contain at least two strictly increasing samples.")
    flat = values.reshape(len(values), -1)
    result = np.empty((len(sample_phases), flat.shape[1]), dtype=np.float64)
    for phase_index, wrapped_phase in enumerate(sample_phases):
        first_cycle = math.ceil(float(common_phase[0]) - wrapped_phase)
        last_cycle = math.floor(float(common_phase[-1]) - wrapped_phase)
        queries = wrapped_phase + np.arange(first_cycle, last_cycle + 1, dtype=np.float64)
        queries = queries[(queries >= common_phase[0]) & (queries <= common_phase[-1])]
        if not len(queries):
            raise ValueError(f"No samples cover requested gait phase {wrapped_phase:g} cycle.")
        if nearest:
            right = np.searchsorted(common_phase, queries, side="left").clip(0, len(common_phase) - 1)
            left = np.maximum(right - 1, 0)
            choose_left = np.abs(queries - common_phase[left]) <= np.abs(common_phase[right] - queries)
            indices = np.where(choose_left, left, right)
            samples = flat[indices]
        else:
            samples = np.column_stack(
                [np.interp(queries, common_phase, flat[:, column]) for column in range(flat.shape[1])]
            )
        result[phase_index] = samples.mean(axis=0)
    return result.reshape((len(sample_phases), *values.shape[1:]))


def _reflect_legs(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 4:
        raise ValueError("Front-hind leg reflection requires a trailing four-leg axis.")
    return values[..., _FRONT_HIND_LEG_PERMUTATION]


def _reflect_joints(values: np.ndarray, robot: str) -> np.ndarray:
    if robot not in _ROBOT_JOINT_REFLECTION_SIGNS:
        raise ValueError(f"Unsupported robot for front-hind reflection: {robot!r}.")
    if values.shape[-1] != 12:
        raise ValueError("Front-hind joint reflection requires a trailing 12-joint axis.")
    reflected = values.reshape(*values.shape[:-1], 4, 3)[..., _FRONT_HIND_LEG_PERMUTATION, :]
    reflected = reflected * _ROBOT_JOINT_REFLECTION_SIGNS[robot]
    return reflected.reshape(values.shape)


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def _phase_set_error(first: np.ndarray, second: np.ndarray) -> float | None:
    if not len(first) or not len(second):
        return None
    distances = np.abs(np.remainder(first[:, None] - second[None, :] + 0.5, 1.0) - 0.5)
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def _touchdown_error(source: _CellTrace, target: _CellTrace, operator: str) -> tuple[float | None, str]:
    if source.liftoff_phases is None or target.touchdown_phases is None:
        return None, source.reasons.get("contact", target.reasons.get("contact", "contact events unavailable"))
    errors: list[float] = []
    permutation = _FRONT_HIND_LEG_PERMUTATION if operator == OPERATORS[1] else np.arange(4)
    for target_leg, source_leg in enumerate(permutation):
        transformed = np.remainder(source.swing_ratio - source.liftoff_phases[int(source_leg)], 1.0)
        error = _phase_set_error(transformed, target.touchdown_phases[target_leg])
        if error is not None:
            errors.append(error)
    if not errors:
        return None, "no paired source liftoff and target touchdown events were observed"
    return float(np.mean(errors)), ""


def _metric_profiles(
    source: _CellTrace,
    target: _CellTrace,
    label: str,
    target_grid: np.ndarray,
    *,
    nearest: bool = False,
    decision_aligned: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    source_axis = source.decision_phase if decision_aligned else source.phase
    target_axis = target.decision_phase if decision_aligned else target.phase
    if source_axis is None or target_axis is None:
        raise ValueError("Pre-decision common gait phase is unavailable.")
    source_phase = np.remainder(source.swing_ratio - target_grid, 1.0)
    source_profile = phase_profile(source_axis, source.arrays[label], source_phase, nearest=nearest)
    target_profile = phase_profile(target_axis, target.arrays[label], target_grid, nearest=nearest)
    return source_profile, target_profile


def compare_pair(
    source: _CellTrace,
    target: _CellTrace,
    *,
    robot: str,
    operator: str,
    phase_samples: int = 128,
) -> dict[str, Any]:
    """Compare one positive/negative rollout pair under a declared operator."""
    if operator not in OPERATORS:
        raise ValueError(f"Unknown paired time-reversal operator: {operator!r}.")
    if phase_samples < 8:
        raise ValueError("phase_samples must be at least eight.")
    grid = np.arange(phase_samples, dtype=np.float64) / phase_samples
    reflected = operator == OPERATORS[1]
    result: dict[str, Any] = {}

    def unavailable(metric: str, reason: str) -> None:
        result[metric] = None
        result[f"{metric}_valid"] = False
        result[f"{metric}_reason"] = reason

    def available(metric: str, value: float) -> None:
        result[metric] = float(value)
        result[f"{metric}_valid"] = True
        result[f"{metric}_reason"] = ""

    for label, metric in (("velocity", METRICS[0]), ("yaw", METRICS[1])):
        if label not in source.arrays or label not in target.arrays:
            unavailable(metric, source.reasons.get(label, target.reasons.get(label, f"{label} unavailable")))
            continue
        source_profile, target_profile = _metric_profiles(source, target, label, grid)
        transformed = -source_profile
        available(metric, abs(float(np.mean(transformed - target_profile))))

    if source.contacts is None or target.contacts is None:
        unavailable(
            METRICS[2], source.reasons.get("contact", target.reasons.get("contact", "contact trace unavailable"))
        )
    else:
        source_phase = np.remainder(source.swing_ratio - grid, 1.0)
        source_contact = phase_profile(source.phase, source.contacts, source_phase, nearest=True)
        target_contact = phase_profile(target.phase, target.contacts, grid, nearest=True)
        if reflected:
            source_contact = _reflect_legs(source_contact)
        available(METRICS[2], float(np.mean(np.abs(source_contact - target_contact))))

    touchdown_error, touchdown_reason = _touchdown_error(source, target, operator)
    if touchdown_error is None:
        unavailable(METRICS[3], touchdown_reason)
    else:
        available(METRICS[3], touchdown_error)

    joint_specs = (
        ("joint_position", METRICS[4], 1.0),
        ("joint_velocity", METRICS[5], -1.0),
        ("actor_mean", METRICS[6], 1.0),
        ("normalized_target", METRICS[7], 1.0),
        ("torque", METRICS[9], 1.0),
        ("power", METRICS[10], 1.0),
    )
    for label, metric, temporal_sign in joint_specs:
        if label not in source.arrays or label not in target.arrays:
            unavailable(metric, source.reasons.get(label, target.reasons.get(label, f"{label} unavailable")))
            continue
        decision_aligned = label in {"actor_mean", "normalized_target"}
        if decision_aligned and (source.decision_phase is None or target.decision_phase is None):
            unavailable(
                metric,
                source.reasons.get(
                    "decision_phase",
                    target.reasons.get("decision_phase", "pre-decision common gait phase unavailable"),
                ),
            )
            continue
        source_profile, target_profile = _metric_profiles(
            source,
            target,
            label,
            grid,
            decision_aligned=decision_aligned,
        )
        if reflected:
            if label == "power":
                source_profile = source_profile.reshape(phase_samples, 4, 3)[:, _FRONT_HIND_LEG_PERMUTATION, :].reshape(
                    phase_samples, 12
                )
            else:
                source_profile = _reflect_joints(source_profile, robot)
        available(metric, _rmse(temporal_sign * source_profile, target_profile))

    if "value" not in source.arrays or "value" not in target.arrays:
        unavailable(METRICS[8], source.reasons.get("value", target.reasons.get("value", "critic value unavailable")))
    elif source.decision_phase is None or target.decision_phase is None:
        unavailable(
            METRICS[8],
            source.reasons.get(
                "decision_phase",
                target.reasons.get("decision_phase", "pre-decision common gait phase unavailable"),
            ),
        )
    else:
        source_profile, target_profile = _metric_profiles(source, target, "value", grid, decision_aligned=True)
        available(METRICS[8], _rmse(source_profile, target_profile))

    if "grf" not in source.arrays or "grf" not in target.arrays:
        unavailable(METRICS[11], source.reasons.get("grf", target.reasons.get("grf", "GRF unavailable")))
    else:
        source_profile, target_profile = _metric_profiles(source, target, "grf", grid)
        if reflected:
            source_profile = _reflect_legs(source_profile)
        available(METRICS[11], _rmse(source_profile, target_profile))
    return result


def _invalid_operator_row(base: Mapping[str, Any], operator: str, reason: str) -> dict[str, Any]:
    row = {**base, "operator": operator, "status": "unavailable", "reason": reason}
    for metric in METRICS:
        row[metric] = None
        row[f"{metric}_valid"] = False
        row[f"{metric}_reason"] = reason
    return row


def analyze_paired_study(
    study: Mapping[str, Any],
    study_root: Path,
    *,
    phase_samples: int = 128,
    archive_identity_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Analyze all declared positive/partner-negative cell pairs in a study."""
    robot = str(study.get("robot", ""))
    if robot not in _ROBOT_JOINT_REFLECTION_SIGNS:
        raise ValueError(f"Paired time-reversal analysis supports {sorted(_ROBOT_JOINT_REFLECTION_SIGNS)}.")
    gaits = {int(gait["index"]): gait for gait in study.get("gaits", [])}
    cells = list(study.get("cells", []))
    by_key = {(int(cell["gait_index"]), float(cell["velocity_mps"]), int(cell.get("seed", 0))): cell for cell in cells}
    if len(by_key) != len(cells):
        raise ValueError("Paired study cells must be unique by gait, velocity, and seed.")
    rows: list[dict[str, Any]] = []
    archive_identity_reasons = archive_identity_reasons or {}
    for positive in cells:
        speed = float(positive["velocity_mps"])
        if speed <= 0.0:
            continue
        gait_index = int(positive["gait_index"])
        gait = gaits.get(gait_index)
        if gait is None:
            continue
        partner_index = int(gait.get("time_reversal_partner", gait_index))
        partner = by_key.get((partner_index, -speed, int(positive.get("seed", 0))))
        base = {
            "robot": robot,
            "protocol": study.get("protocol", "legacy-full"),
            "positive_cell_id": positive["id"],
            "negative_cell_id": None if partner is None else partner["id"],
            "gait_index": gait_index,
            "partner_gait_index": partner_index,
            "gait_name": positive.get("gait_name", gait.get("name")),
            "partner_gait_name": None if partner is None else partner.get("gait_name"),
            "family": positive["family"],
            "direction": "positive_to_negative",
            "speed_mps": speed,
            "seed": int(positive.get("seed", 0)),
            "phase_alignment": "duty_aware_reflection_and_cycle_averaged_interpolation",
            "phase_samples": phase_samples,
        }
        if partner is None:
            rows.extend(
                _invalid_operator_row(base, operator, "declared negative partner cell is missing")
                for operator in OPERATORS
            )
            continue
        identity_reason = archive_identity_reasons.get(str(positive["id"])) or archive_identity_reasons.get(
            str(partner["id"])
        )
        if identity_reason:
            rows.extend(_invalid_operator_row(base, operator, identity_reason) for operator in OPERATORS)
            continue
        window_mismatches = []
        for field in _PAIR_WINDOW_FIELDS:
            positive_value = positive.get(field, study.get(field))
            partner_value = partner.get(field, study.get(field))
            if positive_value != partner_value:
                window_mismatches.append(f"{field} ({positive_value!r} vs {partner_value!r})")
        if window_mismatches:
            reason = "paired rollout windows differ: " + ", ".join(window_mismatches)
            rows.extend(_invalid_operator_row(base, operator, reason) for operator in OPERATORS)
            continue
        try:
            source_trace = _load_cell_trace(study_root, positive, robot)
            target_trace = _load_cell_trace(study_root, partner, robot)
        except (FileNotFoundError, OSError, ValueError) as exc:
            rows.extend(_invalid_operator_row(base, operator, str(exc)) for operator in OPERATORS)
            continue
        if not math.isclose(source_trace.swing_ratio, target_trace.swing_ratio, rel_tol=0.0, abs_tol=1.0e-5):
            reason = f"paired swing ratios differ ({source_trace.swing_ratio:g} vs {target_trace.swing_ratio:g})"
            rows.extend(_invalid_operator_row(base, operator, reason) for operator in OPERATORS)
            continue
        axis_reason = source_trace.axis_metadata_reason or target_trace.axis_metadata_reason
        if axis_reason is not None:
            rows.extend(_invalid_operator_row(base, operator, axis_reason) for operator in OPERATORS)
            continue
        for operator in OPERATORS:
            metrics = compare_pair(
                source_trace,
                target_trace,
                robot=robot,
                operator=operator,
                phase_samples=phase_samples,
            )
            rows.append({**base, "operator": operator, "status": "analyzed", "reason": "", **metrics})

    summary_rows = summarize_paired_results(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot": robot,
        "study_method_version": study.get("method_version"),
        "phase_alignment": {
            "mapping": "phi_source = (swing_ratio - phi_target) mod 1",
            "aggregation": "interpolate within each covered cycle, then average cycles at each target phase",
            "physical_timestamp": "post-step common_gait_phases",
            "policy_timestamp": (
                "pre_decision_common_gait_phases for actor mean, unclipped normalized requested target, and value"
            ),
            "phase_samples": phase_samples,
        },
        "operators": {
            OPERATORS[0]: "pure temporal reversal; no leg relabeling",
            OPERATORS[1]: (
                "temporal reversal followed by front-left<->rear-left and front-right<->rear-right reflection; "
                "X1 additionally negates reflected thigh/calf coordinates"
            ),
        },
        "metric_units": METRIC_UNITS,
        "metric_definitions": {
            "paired_velocity_bias": "abs(mean(-vx_positive_aligned - vx_negative))",
            "paired_yaw_bias": "abs(mean(-yaw_rate_positive_aligned - yaw_rate_negative))",
            "paired_contact_pattern_error": "mean absolute difference of phase-conditioned contact probabilities",
            "paired_touchdown_phase_error": (
                "symmetric nearest-event circular distance between time-reversed source liftoffs and target touchdowns"
            ),
            "paired_joint_position_error": "phase-profile RMSE after the selected operator [rad]",
            "paired_joint_velocity_error": "phase-profile RMSE with temporal odd parity [rad/s]",
            "paired_actor_mean_consistency_error": ("pre-decision phase-profile RMSE of raw deterministic actor means"),
            "paired_normalized_target_consistency_error": (
                "pre-decision phase-profile RMSE of unclipped requested targets normalized by soft joint limits"
            ),
            "paired_value_consistency_error": "pre-decision phase-profile RMSE of critic value predictions",
            "paired_torque_pattern_error": "phase-profile RMSE of generalized torques [N*m]",
            "paired_work_pattern_error": "phase-profile RMSE of absolute joint power [W]",
            "paired_grf_pattern_error": "phase-profile RMSE of vertical ground reaction force [N]",
        },
        "caveat": (
            "Lower paired errors are consistency evidence under the declared measurement/operator model; "
            "they do not prove exact simulator or physical reversibility."
        ),
        "rows": rows,
        "summaries": summary_rows,
    }


def summarize_paired_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build operator-specific family, direction, speed, and robot summaries."""
    summaries: list[dict[str, Any]] = []
    for operator in OPERATORS:
        operator_rows = [row for row in rows if row.get("operator") == operator]
        definitions = (
            ("overall", lambda _row: "all"),
            ("family", lambda row: str(row["family"])),
            ("direction", lambda row: str(row["direction"])),
            ("speed", lambda row: f"{float(row['speed_mps']):g} m/s"),
            ("robot", lambda row: str(row["robot"])),
        )
        for scope, key_fn in definitions:
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in operator_rows:
                grouped[key_fn(row)].append(row)
            for label in sorted(grouped):
                selected = grouped[label]
                summary: dict[str, Any] = {
                    "operator": operator,
                    "scope": scope,
                    "stratum": label,
                    "planned_pairs": len(selected),
                    "analyzed_pairs": sum(row.get("status") == "analyzed" for row in selected),
                }
                for metric in METRICS:
                    values = [
                        float(row[metric])
                        for row in selected
                        if row.get(f"{metric}_valid") is True
                        and row.get(metric) is not None
                        and math.isfinite(float(row[metric]))
                    ]
                    summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
                    summary[f"{metric}_valid_pairs"] = len(values)
                    summary[f"{metric}_coverage_fraction"] = len(values) / len(selected) if selected else 0.0
                summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_paired_outputs(payload: Mapping[str, Any], metrics_dir: Path) -> dict[str, str]:
    """Write paired rows, stratified summaries, and provenance beside grid metrics."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows_path = metrics_dir / "paired_tr_consistency.csv"
    summary_path = metrics_dir / "paired_tr_consistency_summary.csv"
    json_path = metrics_dir / "paired_tr_consistency.json"
    _write_csv(rows_path, payload["rows"])
    _write_csv(summary_path, payload["summaries"])
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "rows": rows_path.name,
        "rows_sha256": _sha256_file(rows_path),
        "summaries": summary_path.name,
        "summaries_sha256": _sha256_file(summary_path),
        "record": json_path.name,
        "record_sha256": _sha256_file(json_path),
    }


def analyze_study_file(study_path: Path, *, phase_samples: int = 128) -> dict[str, Any]:
    """Analyze one light/full study file and write paired metric artifacts."""
    study_path = study_path.resolve()
    study = json.loads(study_path.read_text(encoding="utf-8"))
    archive_identity_reasons = {}
    if study.get("method_version") in _PUBLICATION_METHODS:
        archive_identity_reasons = {
            str(cell.get("id")): reason
            for cell in study.get("cells", [])
            if (reason := _publication_archive_identity_reason(study, study_path, cell)) is not None
        }
    payload = analyze_paired_study(
        study,
        study_path.parent,
        phase_samples=phase_samples,
        archive_identity_reasons=archive_identity_reasons,
    )
    payload["input_study"] = {
        "path": str(study_path),
        "sha256": _sha256_file(study_path),
    }
    payload["analyzer"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
    }
    payload["input_archives"] = [
        {
            "cell_id": cell.get("id"),
            "path": str(archive_path),
            "sha256": _sha256_file(archive_path) if archive_path.is_file() else None,
        }
        for cell in study.get("cells", [])
        for archive_path in (_cell_archive_path(study_path.parent, cell),)
    ]
    payload["archive_identity"] = {
        "required": study.get("method_version") in _PUBLICATION_METHODS,
        "invalid_cells": archive_identity_reasons,
        "valid": not archive_identity_reasons,
    }
    outputs = write_paired_outputs(payload, study_path.parent / "metrics")
    return {**payload, "outputs": outputs}


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Light/full leg-usage study.json to analyze.")
    parser.add_argument("--phase_samples", "--phase-samples", dest="phase_samples", type=int, default=128)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    payload = analyze_study_file(args.study, phase_samples=args.phase_samples)
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
