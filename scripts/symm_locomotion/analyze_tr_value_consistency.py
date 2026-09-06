# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze forward and backward discounted returns on recorded complete reward cycles.

This is an offline numerical diagnostic. It does not alter training or infer
per-step trajectories from aggregate reward tables. The input is an ``.npz``
archive containing the unambiguous per-step field ``step_rewards`` and an
explicit true ``complete_cycle`` marker. Rewards may have shape
``(cycles, steps)`` or shape ``(steps,)``. A one-dimensional archive must also
provide at least one identifier such as ``cycle_id``, ``episode_id``,
``command_interval_id``, or ``gait_row`` to delimit cycles.

Optional per-step fields include ``phase``, ``gait_family``, ``command_sign``,
``robot``, ``critic_values``, and ``transformed_critic_values``. Modular return
indexing is performed independently inside every complete cycle and therefore
never crosses an episode, command interval, or gait row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD_VERSION = "complete_cycle_forward_past_returns_v1"

_REWARD_FIELDS = ("step_rewards",)
_PHASE_FIELDS = ("phase", "phases", "common_gait_phases")
_CRITIC_FIELDS = ("critic_values", "value_predictions", "critic_value_predictions")
_TRANSFORMED_CRITIC_FIELDS = (
    "transformed_critic_values",
    "transformed_value_predictions",
    "transformed_critic_value_predictions",
    "time_reversed_critic_values",
)
_BOUNDARY_FIELDS = {
    "cycle_id": ("cycle_id", "cycle_ids"),
    "episode_id": ("episode_id", "episode_ids"),
    "command_interval_id": ("command_interval_id", "command_interval_ids"),
    "gait_row": ("gait_row", "gait_row_id", "gait_index"),
}
_LABEL_FIELDS = {
    "gait_family": ("gait_family", "gait_families", "family"),
    "command_sign": ("command_sign", "command_signs"),
    "robot": ("robot", "robots", "robot_name"),
}
_COMPLETE_FIELDS = ("complete_cycle", "cycle_complete", "is_complete_cycle")


@dataclass(frozen=True)
class CompleteCycle:
    """One complete, time-ordered reward cycle and its optional diagnostics."""

    rewards: np.ndarray
    phase: np.ndarray | None = None
    phase_is_inferred: bool = False
    gait_family: str = "unknown"
    command_sign: str = "unknown"
    robot: str = "unknown"
    critic_values: np.ndarray | None = None
    transformed_critic_values: np.ndarray | None = None
    cycle_id: str | int | None = None


def complete_cycle_returns(
    rewards: Sequence[float] | np.ndarray,
    *,
    gamma: float,
    horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate forward- and backward-oriented returns within one cycle.

    Args:
        rewards: Per-step rewards in temporal order for exactly one complete cycle.
        gamma: Discount factor in the closed interval ``[0, 1]``.
        horizon: Number of recorded steps in each return. The cycle length is used when omitted.

    Returns:
        The future and past returns, each with one value per cycle phase.

    Raises:
        ValueError: If rewards are not a finite per-step sequence or the horizon
            cannot be supported by the recorded cycle.
    """
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("A complete cycle requires at least two finite per-step reward samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Per-step rewards must all be finite.")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and lie in the closed interval [0, 1].")
    resolved_horizon = values.size if horizon is None else horizon
    if isinstance(resolved_horizon, bool) or not isinstance(resolved_horizon, (int, np.integer)):
        raise ValueError("horizon must be an integer when provided.")
    resolved_horizon = int(resolved_horizon)
    if not 1 <= resolved_horizon <= values.size:
        raise ValueError(
            f"horizon ({resolved_horizon}) must be between 1 and the recorded cycle length ({values.size})."
        )

    phases = np.arange(values.size, dtype=np.int64)[:, None]
    offsets = np.arange(resolved_horizon, dtype=np.int64)[None, :]
    weights = np.power(gamma, np.arange(resolved_horizon, dtype=np.float64))
    future = values[(phases + offsets) % values.size] @ weights
    past = values[(phases - offsets) % values.size] @ weights
    return future, past


# A descriptive alias for callers that prefer a verb-led name.
calculate_cycle_returns = complete_cycle_returns


def _as_step_vector(values: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (length, 1):
        array = array[:, 0]
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain one finite value for each of the {length} reward steps.")
    return array


def _normalize_command_sign(value: Any) -> str:
    if isinstance(value, (np.integer, np.floating, int, float)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("command_sign must be finite.")
        return "positive" if number > 0.0 else "negative" if number < 0.0 else "zero"
    text = str(value).strip()
    normalized = text.lower()
    if normalized in {"+", "+1", "1", "positive", "forward"}:
        return "positive"
    if normalized in {"-", "-1", "negative", "backward", "reverse"}:
        return "negative"
    if normalized in {"0", "zero"}:
        return "zero"
    return text or "unknown"


def _label_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8") or "unknown"
    return str(value) or "unknown"


def _coerce_cycle(cycle: CompleteCycle | Mapping[str, Any] | Sequence[float] | np.ndarray) -> CompleteCycle:
    if isinstance(cycle, CompleteCycle):
        source = cycle
    elif isinstance(cycle, Mapping):
        reward_field = next((name for name in _REWARD_FIELDS if name in cycle), None)
        if reward_field is None:
            raise ValueError("A cycle is missing per-step rewards.")
        phase_source, _ = _first_field(cycle, _PHASE_FIELDS)
        critic_source, _ = _first_field(cycle, _CRITIC_FIELDS)
        transformed_source, _ = _first_field(cycle, _TRANSFORMED_CRITIC_FIELDS)
        source = CompleteCycle(
            rewards=np.asarray(cycle[reward_field]),
            phase=phase_source,
            gait_family=_label_text(cycle.get("gait_family", "unknown")),
            command_sign=_normalize_command_sign(cycle.get("command_sign", "unknown")),
            robot=_label_text(cycle.get("robot", "unknown")),
            critic_values=critic_source,
            transformed_critic_values=transformed_source,
            cycle_id=cycle.get("cycle_id"),
        )
    else:
        source = CompleteCycle(rewards=np.asarray(cycle))

    rewards = np.asarray(source.rewards, dtype=np.float64)
    # complete_cycle_returns supplies the canonical reward validation and error text.
    complete_cycle_returns(rewards, gamma=1.0)
    length = rewards.size
    phase_is_inferred = source.phase is None or source.phase_is_inferred
    if source.phase is None:
        phase = np.arange(length, dtype=np.float64) / length
    else:
        phase = _as_step_vector(source.phase, name="phase", length=length)
        phase = np.remainder(phase, 1.0)
    critic_values = (
        None
        if source.critic_values is None
        else _as_step_vector(source.critic_values, name="critic_values", length=length)
    )
    transformed_critic_values = (
        None
        if source.transformed_critic_values is None
        else _as_step_vector(
            source.transformed_critic_values,
            name="transformed_critic_values",
            length=length,
        )
    )
    return CompleteCycle(
        rewards=rewards,
        phase=phase,
        phase_is_inferred=phase_is_inferred,
        gait_family=_label_text(source.gait_family),
        command_sign=_normalize_command_sign(source.command_sign),
        robot=_label_text(source.robot),
        critic_values=critic_values,
        transformed_critic_values=transformed_critic_values,
        cycle_id=None if source.cycle_id is None else _scalar_label(source.cycle_id),
    )


def _gap_summary(gaps: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(gaps.size),
        "mean": float(np.mean(gaps)),
        "mean_absolute": float(np.mean(np.abs(gaps))),
        "rmse": float(np.sqrt(np.mean(np.square(gaps)))),
    }


def _residual_summary(residuals: np.ndarray) -> dict[str, float | int]:
    return _gap_summary(residuals)


def _stratified_gaps(gaps: np.ndarray, labels: Sequence[str]) -> dict[str, dict[str, float | int]]:
    label_array = np.asarray(labels, dtype=str)
    return {label: _gap_summary(gaps[label_array == label]) for label in sorted(set(label_array.tolist()))}


def _phase_gaps(gaps: np.ndarray, phases: np.ndarray, phase_bins: int) -> dict[str, dict[str, float | int]]:
    indices = np.minimum(np.floor(np.remainder(phases, 1.0) * phase_bins).astype(np.int64), phase_bins - 1)
    result: dict[str, dict[str, float | int]] = {}
    for index in sorted(set(indices.tolist())):
        selected = gaps[indices == index]
        result[str(index)] = {
            "phase_lower": index / phase_bins,
            "phase_upper": (index + 1) / phase_bins,
            "phase_center": (index + 0.5) / phase_bins,
            **_gap_summary(selected),
        }
    return result


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def analyze_complete_cycles(
    cycles: Sequence[CompleteCycle | Mapping[str, Any] | Sequence[float] | np.ndarray],
    *,
    gamma: float = 0.99,
    horizon: int | None = None,
    phase_bins: int = 32,
) -> dict[str, Any]:
    """Analyze discounted return orientation for independent complete cycles.

    Each input element is indexed independently. Consequently, a return may
    wrap inside its own recorded cycle but cannot cross into another cycle or
    another element's episode, command interval, or gait row.

    Args:
        cycles: Complete cycles with raw per-step rewards and optional metadata.
        gamma: Discount factor in the closed interval ``[0, 1]``.
        horizon: Return horizon in steps. Each complete cycle length is used when omitted.
        phase_bins: Number of equally spaced bins used for the phase-stratified gap.

    Returns:
        A JSON-serializable diagnostic record.

    Raises:
        ValueError: If no complete per-step cycle is available or an input is malformed.
    """
    if isinstance(phase_bins, bool) or not isinstance(phase_bins, (int, np.integer)) or phase_bins < 1:
        raise ValueError("phase_bins must be a positive integer.")
    if not cycles:
        raise ValueError("No complete cycles with per-step reward information were supplied.")

    normalized = [_coerce_cycle(cycle) for cycle in cycles]
    future_parts: list[np.ndarray] = []
    past_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    gait_labels: list[str] = []
    command_labels: list[str] = []
    robot_labels: list[str] = []
    critic_residual_parts: list[np.ndarray] = []
    transformed_residual_parts: list[np.ndarray] = []
    critic_difference_parts: list[np.ndarray] = []
    phase_source_counts = {"recorded": 0, "inferred_uniform_index": 0}
    cycle_summaries: list[dict[str, Any]] = []

    for index, cycle in enumerate(normalized):
        future, past = complete_cycle_returns(cycle.rewards, gamma=gamma, horizon=horizon)
        gap = future - past
        future_parts.append(future)
        past_parts.append(past)
        assert cycle.phase is not None
        phase_parts.append(cycle.phase)
        phase_source = "inferred_uniform_index" if cycle.phase_is_inferred else "recorded"
        phase_source_counts[phase_source] += 1
        gait_labels.extend([cycle.gait_family] * len(future))
        command_labels.extend([cycle.command_sign] * len(future))
        robot_labels.extend([cycle.robot] * len(future))
        cycle_summary = {
            "cycle_id": cycle.cycle_id if cycle.cycle_id is not None else index,
            "steps": len(future),
            "phase_source": phase_source,
            "gait_family": cycle.gait_family,
            "command_sign": cycle.command_sign,
            "robot": cycle.robot,
            "future_return_mean": float(np.mean(future)),
            "past_return_mean": float(np.mean(past)),
            "future_past_return_rmse": float(np.sqrt(np.mean(np.square(gap)))),
        }
        if cycle.critic_values is not None:
            residual = cycle.critic_values - future
            critic_residual_parts.append(residual)
            cycle_summary["critic_value_minus_future_return"] = _residual_summary(residual)
        if cycle.transformed_critic_values is not None:
            residual = cycle.transformed_critic_values - past
            transformed_residual_parts.append(residual)
            cycle_summary["transformed_critic_value_minus_past_return"] = _residual_summary(residual)
        if cycle.critic_values is not None and cycle.transformed_critic_values is not None:
            difference = cycle.transformed_critic_values - cycle.critic_values
            critic_difference_parts.append(difference)
            cycle_summary["transformed_minus_original_critic_value"] = _residual_summary(difference)
        cycle_summaries.append(cycle_summary)

    future = np.concatenate(future_parts)
    past = np.concatenate(past_parts)
    phases = np.concatenate(phase_parts)
    gaps = future - past
    rmse = float(np.sqrt(np.mean(np.square(gaps))))
    pooled_return_rms = float(np.sqrt(np.mean((np.square(future) + np.square(past)) / 2.0)))
    normalized_rmse = 0.0 if pooled_return_rms == 0.0 and rmse == 0.0 else rmse / pooled_return_rms

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "gamma": float(gamma),
        "horizon": int(horizon) if horizon is not None else "complete_cycle",
        "cycle_count": len(normalized),
        "sample_count": int(future.size),
        "phase_bins": int(phase_bins),
        "phase_source_counts": phase_source_counts,
        "metric_definitions": {
            "return_gap": "G_future - G_past",
            "future_past_return_normalized_rmse": (
                "return-gap RMSE divided by the pooled RMS magnitude of future and past returns"
            ),
            "future_past_return_gap_by_phase": (
                "Uses recorded phase when supplied; otherwise uses normalized within-cycle sample index with an "
                "arbitrary zero phase"
            ),
            "critic_value_minus_future_return": "V(o_t) - G_future",
            "transformed_critic_value_minus_past_return": "V(T_O o_t) - G_past",
            "transformed_minus_original_critic_value": "V(T_O o_t) - V(o_t)",
        },
        "future_return_mean": float(np.mean(future)),
        "past_return_mean": float(np.mean(past)),
        "future_past_return_rmse": rmse,
        "future_past_return_normalized_rmse": float(normalized_rmse),
        "future_past_return_correlation": _correlation(future, past),
        "future_past_return_gap_by_phase": _phase_gaps(gaps, phases, int(phase_bins)),
        "future_past_return_gap_by_gait_family": _stratified_gaps(gaps, gait_labels),
        "future_past_return_gap_by_command_sign": _stratified_gaps(gaps, command_labels),
        "future_past_return_gap_by_robot": _stratified_gaps(gaps, robot_labels),
        "cycle_summaries": cycle_summaries,
    }
    if critic_residual_parts:
        result["critic_value_minus_future_return"] = _residual_summary(np.concatenate(critic_residual_parts))
    if transformed_residual_parts:
        result["transformed_critic_value_minus_past_return"] = _residual_summary(
            np.concatenate(transformed_residual_parts)
        )
    if critic_difference_parts:
        result["transformed_minus_original_critic_value"] = _residual_summary(np.concatenate(critic_difference_parts))
    return result


def _first_field(data: Mapping[str, Any], names: Sequence[str]) -> tuple[Any | None, str | None]:
    for name in names:
        if name in data:
            return data[name], name
    return None, None


def _scalar_label(value: Any) -> Any:
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("Expected scalar cycle metadata.")
    item = scalar.item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def _row_field(value: Any, *, row: int, cycle_count: int, step_count: int, name: str) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.shape == (cycle_count,):
        return array[row]
    if array.shape == (cycle_count, 1):
        return array[row, 0]
    if array.shape == (cycle_count, step_count):
        row_values = array[row]
        if not np.all(row_values == row_values[0]):
            raise ValueError(f"Complete cycle row {row} crosses a {name} boundary.")
        return row_values[0]
    raise ValueError(
        f"{name} metadata has shape {array.shape}; expected a scalar, ({cycle_count},), "
        f"or ({cycle_count}, {step_count})."
    )


def _row_steps(value: Any, *, row: int, cycle_count: int, step_count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (cycle_count, step_count, 1):
        return array[row, :, 0]
    if array.shape == (cycle_count, step_count):
        return array[row]
    if cycle_count == 1 and array.shape in {(step_count,), (step_count, 1)}:
        return array.reshape(step_count)
    if name == "phase" and array.shape == (step_count,):
        return array
    raise ValueError(f"{name} must have the same cycle-by-step shape as rewards.")


def _command_sign_source(data: Mapping[str, Any]) -> Any | None:
    direct, _ = _first_field(data, _LABEL_FIELDS["command_sign"])
    if direct is not None:
        return direct
    velocity, _ = _first_field(data, ("command_x", "desired_velocity_x", "velocity_mps"))
    if velocity is not None:
        return np.sign(np.asarray(velocity, dtype=np.float64))
    desired_velocity, _ = _first_field(data, ("desired_lin_vel", "desired_linear_velocity"))
    if desired_velocity is not None:
        array = np.asarray(desired_velocity, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] < 1:
            raise ValueError("desired_lin_vel must have a nonempty final coordinate dimension.")
        return np.sign(array[..., 0])
    return None


def _require_complete(value: Any, *, context: str) -> None:
    complete = np.asarray(value)
    if complete.dtype.kind == "b":
        pass
    elif complete.dtype.kind in {"i", "u", "f"}:
        if complete.dtype.kind == "f" and not np.all(np.isfinite(complete)):
            raise ValueError(f"Malformed complete-cycle marker for {context}; values must be finite booleans.")
        if not np.all((complete == 0) | (complete == 1)):
            raise ValueError(f"Malformed complete-cycle marker for {context}; expected only boolean values.")
        complete = complete.astype(bool)
    else:
        raise ValueError(f"Malformed complete-cycle marker for {context}; expected boolean values.")
    if not np.all(complete):
        raise ValueError(f"The source marks {context} as incomplete; only recorded complete cycles are valid.")


def _cycles_from_matrix(data: Mapping[str, Any], rewards: np.ndarray) -> list[CompleteCycle]:
    cycle_count, step_count = rewards.shape
    phase_source, _ = _first_field(data, _PHASE_FIELDS)
    critic_source, _ = _first_field(data, _CRITIC_FIELDS)
    transformed_source, _ = _first_field(data, _TRANSFORMED_CRITIC_FIELDS)
    complete_source, _ = _first_field(data, _COMPLETE_FIELDS)
    if complete_source is None:
        raise ValueError(
            "Matrix archives must include a true complete_cycle marker for every recorded reward-cycle row."
        )
    command_source = _command_sign_source(data)
    label_sources = {
        name: _first_field(data, aliases)[0] for name, aliases in _LABEL_FIELDS.items() if name != "command_sign"
    }
    boundary_sources = {name: _first_field(data, aliases)[0] for name, aliases in _BOUNDARY_FIELDS.items()}

    cycles: list[CompleteCycle] = []
    for row in range(cycle_count):
        marker = _row_field(
            complete_source,
            row=row,
            cycle_count=cycle_count,
            step_count=step_count,
            name="complete_cycle",
        )
        _require_complete(marker, context=f"cycle row {row}")
        labels = {
            name: "unknown"
            if source is None
            else _row_field(source, row=row, cycle_count=cycle_count, step_count=step_count, name=name)
            for name, source in label_sources.items()
        }
        command_sign = (
            "unknown"
            if command_source is None
            else _row_field(
                command_source,
                row=row,
                cycle_count=cycle_count,
                step_count=step_count,
                name="command_sign",
            )
        )
        for name, source in boundary_sources.items():
            if source is not None:
                _row_field(source, row=row, cycle_count=cycle_count, step_count=step_count, name=name)
        cycle_id_source = boundary_sources["cycle_id"]
        cycle_id = (
            row
            if cycle_id_source is None
            else _row_field(
                cycle_id_source,
                row=row,
                cycle_count=cycle_count,
                step_count=step_count,
                name="cycle_id",
            )
        )
        cycles.append(
            CompleteCycle(
                rewards=rewards[row],
                phase=(
                    None
                    if phase_source is None
                    else _row_steps(
                        phase_source,
                        row=row,
                        cycle_count=cycle_count,
                        step_count=step_count,
                        name="phase",
                    )
                ),
                gait_family=_label_text(labels["gait_family"]),
                command_sign=_normalize_command_sign(command_sign),
                robot=_label_text(labels["robot"]),
                critic_values=(
                    None
                    if critic_source is None
                    else _row_steps(
                        critic_source,
                        row=row,
                        cycle_count=cycle_count,
                        step_count=step_count,
                        name="critic_values",
                    )
                ),
                transformed_critic_values=(
                    None
                    if transformed_source is None
                    else _row_steps(
                        transformed_source,
                        row=row,
                        cycle_count=cycle_count,
                        step_count=step_count,
                        name="transformed_critic_values",
                    )
                ),
                cycle_id=_scalar_label(cycle_id),
            )
        )
    return cycles


def _step_field(data: Mapping[str, Any], names: Sequence[str], length: int, *, field_name: str) -> np.ndarray | None:
    source, _ = _first_field(data, names)
    if source is None:
        return None
    array = np.asarray(source)
    if array.ndim == 0:
        return np.full(length, array.item(), dtype=array.dtype)
    if array.shape == (length, 1):
        array = array[:, 0]
    if array.shape != (length,):
        raise ValueError(f"{field_name} must be scalar or contain one value per reward step.")
    return array


def _cycles_from_vector(data: Mapping[str, Any], rewards: np.ndarray) -> list[CompleteCycle]:
    length = len(rewards)
    explicit_boundaries = {name: _first_field(data, aliases)[0] for name, aliases in _BOUNDARY_FIELDS.items()}
    if not any(source is not None for source in explicit_boundaries.values()):
        raise ValueError(
            "One-dimensional step_rewards require cycle_id, episode_id, command_interval_id, or gait_row "
            "metadata so modular returns cannot cross an unrecorded boundary."
        )
    boundary_arrays = {
        name: _step_field(data, aliases, length, field_name=name)
        for name, aliases in {**_BOUNDARY_FIELDS, **_LABEL_FIELDS}.items()
    }
    command_source = _command_sign_source(data)
    if command_source is not None:
        command_array = np.asarray(command_source)
        if command_array.ndim == 0:
            command_array = np.full(length, command_array.item())
        if command_array.shape == (length, 1):
            command_array = command_array[:, 0]
        if command_array.shape != (length,):
            raise ValueError("command_sign must be scalar or contain one value per reward step.")
        boundary_arrays["command_sign"] = command_array

    boundaries = [0]
    for index in range(1, length):
        if any(values is not None and values[index] != values[index - 1] for values in boundary_arrays.values()):
            boundaries.append(index)
    boundaries.append(length)

    phase = _step_field(data, _PHASE_FIELDS, length, field_name="phase")
    critic = _step_field(data, _CRITIC_FIELDS, length, field_name="critic_values")
    transformed = _step_field(
        data,
        _TRANSFORMED_CRITIC_FIELDS,
        length,
        field_name="transformed_critic_values",
    )
    complete = _step_field(data, _COMPLETE_FIELDS, length, field_name="complete_cycle")
    if complete is None:
        raise ValueError(
            "One-dimensional archives must include a true complete_cycle marker for every recorded cycle segment."
        )
    cycles: list[CompleteCycle] = []
    for cycle_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        _require_complete(complete[start:stop], context=f"cycle segment {cycle_index}")

        def label(name: str) -> Any:
            values = boundary_arrays[name]
            return "unknown" if values is None else values[start]

        cycle_id = label("cycle_id")
        if cycle_id == "unknown":
            cycle_id = cycle_index
        cycles.append(
            CompleteCycle(
                rewards=rewards[start:stop],
                phase=None if phase is None else phase[start:stop],
                gait_family=_label_text(label("gait_family")),
                command_sign=_normalize_command_sign(label("command_sign")),
                robot=_label_text(label("robot")),
                critic_values=None if critic is None else critic[start:stop],
                transformed_critic_values=None if transformed is None else transformed[start:stop],
                cycle_id=_scalar_label(cycle_id),
            )
        )
    return cycles


def load_complete_cycles(path: Path) -> list[CompleteCycle]:
    """Load complete per-step reward cycles from a numerical ``.npz`` archive.

    Aggregate-only archives fail explicitly. The utility never synthesizes
    trajectories from episode sums, means, TensorBoard scalars, or summary rows.

    Args:
        path: Numerical archive containing ``step_rewards`` and explicit cycle-completeness metadata.

    Returns:
        Complete cycles kept separate at every recorded boundary.

    Raises:
        ValueError: If per-step rewards or valid complete cycles are unavailable.
    """
    source_path = Path(path).resolve()
    if source_path.suffix.lower() != ".npz":
        raise ValueError("The input must be an .npz archive with raw per-step rewards.")
    if not source_path.is_file():
        raise ValueError(f"Input archive does not exist: {source_path}")
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            reward_source, reward_field = _first_field(archive, _REWARD_FIELDS)
            if reward_source is None:
                aggregate_names = sorted(
                    name for name in archive.files if "reward" in name.lower() and name not in _REWARD_FIELDS
                )
                detail = f" Found only aggregate/unknown reward fields: {aggregate_names}." if aggregate_names else ""
                raise ValueError(
                    "Source archive does not contain sufficient per-step reward information; expected one of "
                    f"{_REWARD_FIELDS}.{detail}"
                )
            rewards = np.asarray(reward_source, dtype=np.float64)
            if rewards.ndim == 2 and rewards.shape[1] == 1:
                rewards = rewards[:, 0]
            if rewards.ndim == 1:
                cycles = _cycles_from_vector(archive, rewards)
            elif rewards.ndim == 2:
                cycles = _cycles_from_matrix(archive, rewards)
            else:
                raise ValueError(
                    f"Per-step reward field {reward_field!r} has shape {rewards.shape}; expected (steps,) or "
                    "(cycles, steps)."
                )
    except OSError as exc:
        raise ValueError(f"Could not read numerical archive {source_path}: {exc}") from exc
    if not cycles:
        raise ValueError("Source archive contains no complete per-step reward cycles.")
    # Validate before returning so errors identify archive quality rather than a later metric.
    return [_coerce_cycle(cycle) for cycle in cycles]


def analyze_archive(
    path: Path,
    *,
    gamma: float = 0.99,
    horizon: int | None = None,
    phase_bins: int = 32,
) -> dict[str, Any]:
    """Load and analyze one complete-cycle reward archive."""
    source_path = Path(path).resolve()
    result = analyze_complete_cycles(
        load_complete_cycles(source_path),
        gamma=gamma,
        horizon=horizon,
        phase_bins=phase_bins,
    )
    result["input_archive"] = {
        "path": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    return result


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="NPZ archive containing recorded complete-cycle per-step rewards.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor in [0, 1] (default: 0.99).")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Return horizon in steps (default: each complete recorded cycle length).",
    )
    parser.add_argument(
        "--phase_bins",
        "--phase-bins",
        dest="phase_bins",
        type=int,
        default=32,
        help="Number of phase bins used for stratified gap reporting (default: 32).",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional JSON output path; the report is always printed to stdout."
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    try:
        result = analyze_archive(
            args.archive,
            gamma=args.gamma,
            horizon=args.horizon,
            phase_bins=args.phase_bins,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            output_path = args.output.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
