# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transition-aligned causal-history reconstruction for time reversal.

The classes in this module are task-local and independent of RSL-RL rollout
storage.  They retain finalized decision records across rollout boundaries only
long enough to reconstruct auxiliary actor/value consistency samples.  They do
not create PPO transitions and their transient contents are never checkpointed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields, replace
from numbers import Integral, Real
from typing import Literal

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_POLICY_OBS_DIM,
    SYMM_QUADRUPED_POLICY_OBS_DIMENSIONS,
    SYMM_QUADRUPED_POLICY_OBS_LAYOUT,
    SYMM_QUADRUPED_POLICY_OBS_SCALE,
    build_reversed_causal_policy_frame,
    pack_term_major_policy_history,
    unpack_term_major_policy_history,
)

TR_CONSISTENCY_MAPPING_VERSION = "transition_aligned_causal_sequence_v1"
"""Immutable identifier for the exact causal-history reconstruction."""

ACTION_HISTORY_LENGTH = 2
"""Number of previous raw actions carried by one policy frame."""

_ACTION_DIM = SYMM_QUADRUPED_POLICY_OBS_DIMENSIONS.previous_action
_EPSILON = 1.0e-12
_VALIDITY_MODES = frozenset(
    {
        "command",
        "command_tracking",
        "command_upright_phase",
        "command_tracking_upright_phase",
    }
)


def _validate_tensor_shape(value: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise ValueError(f"{name} must have shape {expected_shape}; received {shape}.")


def _validate_float_tensor(
    value: torch.Tensor,
    expected_shape: tuple[int, ...],
    reference: torch.Tensor,
    name: str,
) -> None:
    _validate_tensor_shape(value, expected_shape, name)
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point; received {value.dtype}.")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(
            f"{name} must use {reference.dtype}/{reference.device}; received {value.dtype}/{value.device}."
        )


def _validate_scalar_limit(name: str, value: float, *, strictly_positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(float(value)):
        raise ValueError(f"{name} must be a real number; received {value!r}.")
    minimum = 0.0 if not strictly_positive else 0.0
    invalid = value < minimum if not strictly_positive else value <= minimum
    if invalid:
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}; received {value!r}.")


@dataclass(frozen=True)
class TRSimulatorValidityMetadata:
    """Simulator-only measurements used to validate a finalized decision.

    None of these fields is appended to the actor or critic observation.

    Attributes:
        body_linear_velocity_b: Measured body-frame linear velocity [m/s],
            shape ``(num_envs, 3)``.
        body_yaw_rate: Measured body-frame yaw rate [rad/s], shape ``(num_envs,)``.
        projected_gravity: Gravity direction expressed in the body frame,
            shape ``(num_envs, 3)``.
        contact_impulse: Maximum control-interval contact impulse [N s], shape
            ``(num_envs,)``.
        foot_slip: Detached nonnegative foot-slip diagnostic [m/s], shape
            ``(num_envs,)``.
        reverse_dynamics_residual: Detached nonnegative reverse-dynamics
            residual, shape ``(num_envs,)``.
        actuator_saturation: Detached actuator-saturation fraction, shape
            ``(num_envs,)``.
    """

    body_linear_velocity_b: torch.Tensor
    body_yaw_rate: torch.Tensor
    projected_gravity: torch.Tensor
    contact_impulse: torch.Tensor
    foot_slip: torch.Tensor
    reverse_dynamics_residual: torch.Tensor
    actuator_saturation: torch.Tensor

    @classmethod
    def neutral(
        cls,
        num_envs: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> TRSimulatorValidityMetadata:
        """Return finite neutral metadata for pure reconstruction tests."""
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer; received {num_envs!r}.")
        zeros = torch.zeros(int(num_envs), device=device, dtype=dtype)
        gravity = torch.zeros(int(num_envs), 3, device=device, dtype=dtype)
        gravity[:, 2] = -1.0
        return cls(
            body_linear_velocity_b=torch.zeros(int(num_envs), 3, device=device, dtype=dtype),
            body_yaw_rate=zeros.clone(),
            projected_gravity=gravity,
            contact_impulse=zeros.clone(),
            foot_slip=zeros.clone(),
            reverse_dynamics_residual=zeros.clone(),
            actuator_saturation=zeros.clone(),
        )

    def validate(self, num_envs: int, reference: torch.Tensor) -> None:
        """Validate batch shape, dtype, and device against a policy frame."""
        _validate_float_tensor(
            self.body_linear_velocity_b,
            (num_envs, 3),
            reference,
            "simulator.body_linear_velocity_b",
        )
        _validate_float_tensor(
            self.projected_gravity,
            (num_envs, 3),
            reference,
            "simulator.projected_gravity",
        )
        for name in (
            "body_yaw_rate",
            "contact_impulse",
            "foot_slip",
            "reverse_dynamics_residual",
            "actuator_saturation",
        ):
            _validate_float_tensor(
                getattr(self, name),
                (num_envs,),
                reference,
                f"simulator.{name}",
            )


@dataclass(frozen=True)
class TRSequenceValidityConfig:
    """Numerical gates for exact transition-aligned sequence candidates."""

    mode: Literal[
        "command",
        "command_tracking",
        "command_upright_phase",
        "command_tracking_upright_phase",
    ] = "command"
    min_abs_command_velocity: float = 0.0
    command_tolerance: float = 1.0e-6
    period_tolerance: float = 1.0e-6
    duty_factor_tolerance: float = 1.0e-6
    relative_gait_tolerance: float = 1.0e-5
    minimum_phase_norm: float = 1.0e-6
    tracking_abs_xy: float = 0.25
    tracking_rel_xy: float = 0.25
    tracking_abs_yaw: float = 0.25
    tracking_rel_yaw: float = 0.25
    projected_gravity_tolerance: float = 0.35
    phase_boundary_margin: float = 0.03
    maximum_contact_impulse: float = math.inf
    maximum_foot_slip: float = math.inf
    maximum_reverse_dynamics_residual: float = math.inf
    maximum_actuator_saturation: float = math.inf

    def __post_init__(self) -> None:
        """Validate all gates without importing simulator configuration."""
        if self.mode not in _VALIDITY_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALIDITY_MODES)!r}; received {self.mode!r}.")
        for name in (
            "min_abs_command_velocity",
            "command_tolerance",
            "period_tolerance",
            "duty_factor_tolerance",
            "relative_gait_tolerance",
            "tracking_abs_xy",
            "tracking_rel_xy",
            "tracking_abs_yaw",
            "tracking_rel_yaw",
            "projected_gravity_tolerance",
            "phase_boundary_margin",
        ):
            _validate_scalar_limit(name, getattr(self, name))
        _validate_scalar_limit("minimum_phase_norm", self.minimum_phase_norm, strictly_positive=True)
        if self.phase_boundary_margin >= 0.5:
            raise ValueError("phase_boundary_margin must be less than 0.5 cycle.")
        for name in (
            "maximum_contact_impulse",
            "maximum_foot_slip",
            "maximum_reverse_dynamics_residual",
            "maximum_actuator_saturation",
        ):
            _validate_scalar_limit(name, getattr(self, name), strictly_positive=True)


@dataclass(frozen=True)
class TransitionAlignedTRRecord:
    """One batched decision/action record after its transition was finalized."""

    policy_observation: torch.Tensor
    latest_frame: torch.Tensor
    action: torch.Tensor
    episode_id: torch.Tensor
    command_segment_id: torch.Tensor
    gait_segment_id: torch.Tensor
    disturbance_generation_id: torch.Tensor
    collection_update_id: torch.Tensor
    transition_valid: torch.Tensor
    done: torch.Tensor
    timeout: torch.Tensor
    history_warmup_complete: torch.Tensor
    gait_row: torch.Tensor
    simulator: TRSimulatorValidityMetadata

    def validate(self, num_envs: int, history_length: int, device: torch.device) -> None:
        """Validate the finalized record against one buffer contract."""
        expected_width = history_length * SYMM_QUADRUPED_POLICY_OBS_DIM
        _validate_tensor_shape(self.policy_observation, (num_envs, expected_width), "policy_observation")
        if not torch.is_floating_point(self.policy_observation) or self.policy_observation.device != device:
            raise ValueError("policy_observation must be floating point and reside on the buffer device.")
        _validate_float_tensor(
            self.latest_frame,
            (num_envs, SYMM_QUADRUPED_POLICY_OBS_DIM),
            self.policy_observation,
            "latest_frame",
        )
        _validate_float_tensor(self.action, (num_envs, _ACTION_DIM), self.policy_observation, "action")
        for name in (
            "episode_id",
            "command_segment_id",
            "gait_segment_id",
            "disturbance_generation_id",
            "collection_update_id",
            "gait_row",
        ):
            value = getattr(self, name)
            _validate_tensor_shape(value, (num_envs,), name)
            if value.device != device or value.dtype == torch.bool or torch.is_floating_point(value):
                raise ValueError(
                    f"{name} must be an integer tensor on {device}; received {value.dtype}/{value.device}."
                )
        # Avoid synchronizing the training CUDA stream merely to validate a
        # diagnostic id. CUDA values are checked asynchronously by the
        # candidate validity mask below and therefore fail closed.
        if self.collection_update_id.device.type == "cpu" and torch.any(self.collection_update_id < 0):
            raise ValueError("collection_update_id must be nonnegative.")
        for name in ("transition_valid", "done", "timeout", "history_warmup_complete"):
            value = getattr(self, name)
            _validate_tensor_shape(value, (num_envs,), name)
            if value.device != device or value.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean tensor on {device}.")
        self.simulator.validate(num_envs, self.policy_observation)


@dataclass(frozen=True)
class _CompactTransitionAlignedTRRecord:
    """One prepared record without a retained full policy history."""

    latest_frame: torch.Tensor
    action: torch.Tensor
    episode_id: torch.Tensor
    command_segment_id: torch.Tensor
    gait_segment_id: torch.Tensor
    disturbance_generation_id: torch.Tensor
    collection_update_id: torch.Tensor
    transition_valid: torch.Tensor
    done: torch.Tensor
    timeout: torch.Tensor
    history_warmup_complete: torch.Tensor
    gait_row: torch.Tensor
    simulator: TRSimulatorValidityMetadata
    policy_history_seed: torch.Tensor | None
    policy_history_consistent: torch.Tensor


@dataclass(frozen=True)
class _PreparedPolicyHistory:
    latest_frame: torch.Tensor
    policy_history_seed: torch.Tensor | None
    policy_history_consistent: torch.Tensor


@dataclass(frozen=True)
class TransitionAlignedTRCandidateBatch:
    """Auxiliary-only actor/value samples reconstructed from finalized records."""

    actor_source_observation: torch.Tensor
    value_source_observation: torch.Tensor
    reversed_observation: torch.Tensor
    weight: torch.Tensor
    environment_id: torch.Tensor
    episode_id: torch.Tensor
    command_segment_id: torch.Tensor
    gait_segment_id: torch.Tensor
    disturbance_generation_id: torch.Tensor
    oldest_collection_update_id: torch.Tensor
    newest_collection_update_id: torch.Tensor
    policy_version_span: torch.Tensor
    candidate_age_updates: torch.Tensor

    @property
    def count(self) -> int:
        """Return the number of reconstructed candidates."""
        return int(self.actor_source_observation.shape[0])

    def __len__(self) -> int:
        """Return :attr:`count`."""
        return self.count

    @property
    def original_actor_observation(self) -> torch.Tensor:
        """Compatibility description for the edge-aligned actor source ``H_t``."""
        return self.actor_source_observation

    @property
    def original_value_observation(self) -> torch.Tensor:
        """Compatibility description for the state-aligned value source ``H_{t+1}``."""
        return self.value_source_observation

    def index(self, selection: torch.Tensor) -> TransitionAlignedTRCandidateBatch:
        """Return a batch-preserving subset."""
        return replace(
            self,
            **{field.name: getattr(self, field.name)[selection] for field in fields(self)},
        )

    @classmethod
    def concatenate(
        cls,
        batches: list[TransitionAlignedTRCandidateBatch],
    ) -> TransitionAlignedTRCandidateBatch:
        """Concatenate nonempty candidate batches along their batch dimension."""
        if not batches:
            raise ValueError("At least one candidate batch is required.")
        return cls(
            **{field.name: torch.cat([getattr(batch, field.name) for batch in batches]) for field in fields(cls)}
        )


@dataclass
class _DenseCandidateSnapshot:
    batch: TransitionAlignedTRCandidateBatch
    valid: torch.Tensor


@dataclass
class _CompactCandidateWindow:
    records: tuple[_CompactTransitionAlignedTRRecord, ...]
    policy_frames: tuple[torch.Tensor, ...]
    records_since_clear: torch.Tensor
    valid: torch.Tensor


def build_transition_aligned_reversed_history(
    forward_frames: torch.Tensor,
    forward_actions: torch.Tensor,
    history_length: int,
) -> torch.Tensor:
    """Build ``Hbar_{t+1}`` from future forward frames and actions.

    Args:
        forward_frames: Frames ``y_{t+1}, ..., y_{t+H}``, shape
            ``(..., H, 64)``.
        forward_actions: Executed actions ``a_{t+1}, ..., a_{t+H+1}``,
            shape ``(..., H + 1, 12)``.
        history_length: Positive history length ``H``.

    Returns:
        Frames ``ybar_{t+H}, ..., ybar_{t+1}`` packed in native term-major,
        oldest-to-newest flattened order, shape ``(..., H * 64)``.

    Raises:
        ValueError: If the tensors do not match the exact ``H``/contract shapes.
    """
    if isinstance(history_length, bool) or not isinstance(history_length, Integral) or history_length <= 0:
        raise ValueError(f"history_length must be a positive integer; received {history_length!r}.")
    if not isinstance(forward_frames, torch.Tensor) or forward_frames.ndim < 2:
        shape = tuple(forward_frames.shape) if isinstance(forward_frames, torch.Tensor) else None
        raise ValueError(f"forward_frames must have shape (..., H, 64); received {shape}.")
    if forward_frames.shape[-2:] != (history_length, SYMM_QUADRUPED_POLICY_OBS_DIM):
        raise ValueError(
            "forward_frames must have trailing shape "
            f"({history_length}, {SYMM_QUADRUPED_POLICY_OBS_DIM}); received {tuple(forward_frames.shape)}."
        )
    expected_action_shape = forward_frames.shape[:-2] + (history_length + 1, _ACTION_DIM)
    if not isinstance(forward_actions, torch.Tensor) or forward_actions.shape != expected_action_shape:
        shape = tuple(forward_actions.shape) if isinstance(forward_actions, torch.Tensor) else None
        raise ValueError(f"forward_actions must have shape {tuple(expected_action_shape)}; received {shape}.")
    if not torch.is_floating_point(forward_frames) or not torch.is_floating_point(forward_actions):
        raise ValueError("forward_frames and forward_actions must be floating point.")
    if forward_actions.dtype != forward_frames.dtype or forward_actions.device != forward_frames.device:
        raise ValueError("forward_frames and forward_actions must have identical dtype and device.")

    reversed_in_forward_order = build_reversed_causal_policy_frame(
        forward_frames,
        forward_actions[..., :-1, :],
        forward_actions[..., 1:, :],
    )
    reversed_time_order = torch.flip(reversed_in_forward_order, dims=(-2,))
    return pack_term_major_policy_history(reversed_time_order)


def time_reversal_tracking_mask(
    metadata: TRSimulatorValidityMetadata,
    velocity_command: torch.Tensor,
    cfg: TRSequenceValidityConfig,
) -> torch.Tensor:
    """Return the measured XY/yaw command-tracking gate.

    Args:
        metadata: Simulator-only measured velocities.
        velocity_command: Unscaled ``(..., 3)`` command ``[vx, vy, wz]``
            with units ``[m/s, m/s, rad/s]``.
        cfg: Absolute and relative tracking tolerances.

    Returns:
        Boolean mask with shape ``velocity_command.shape[:-1]``.
    """
    if not isinstance(velocity_command, torch.Tensor) or velocity_command.ndim < 1 or velocity_command.shape[-1] != 3:
        shape = tuple(velocity_command.shape) if isinstance(velocity_command, torch.Tensor) else None
        raise ValueError(f"velocity_command must have shape (..., 3); received {shape}.")
    expected_vector_shape = velocity_command.shape
    expected_scalar_shape = velocity_command.shape[:-1]
    _validate_float_tensor(
        metadata.body_linear_velocity_b,
        expected_vector_shape,
        velocity_command,
        "metadata.body_linear_velocity_b",
    )
    _validate_float_tensor(
        metadata.body_yaw_rate,
        expected_scalar_shape,
        velocity_command,
        "metadata.body_yaw_rate",
    )
    command_xy_norm = torch.linalg.vector_norm(velocity_command[..., :2], dim=-1)
    error_xy = torch.linalg.vector_norm(
        metadata.body_linear_velocity_b[..., :2] - velocity_command[..., :2],
        dim=-1,
    )
    error_yaw = torch.abs(metadata.body_yaw_rate - velocity_command[..., 2])
    xy_limit = cfg.tracking_abs_xy + cfg.tracking_rel_xy * command_xy_norm
    yaw_limit = cfg.tracking_abs_yaw + cfg.tracking_rel_yaw * torch.abs(velocity_command[..., 2])
    finite = (
        torch.isfinite(metadata.body_linear_velocity_b).all(dim=-1)
        & torch.isfinite(metadata.body_yaw_rate)
        & torch.isfinite(velocity_command).all(dim=-1)
    )
    return finite & (error_xy <= xy_limit) & (error_yaw <= yaw_limit)


def time_reversal_reversibility_confidence(
    metadata: TRSimulatorValidityMetadata,
    cfg: TRSequenceValidityConfig,
) -> torch.Tensor:
    """Return a detached soft reversibility weight in ``[0, 1]``.

    Infinite maxima disable the corresponding factor. Finite maxima also act
    as hard validity thresholds inside :class:`TransitionAlignedTRBuffer`.
    """
    metrics_and_limits = (
        (metadata.contact_impulse, cfg.maximum_contact_impulse),
        (metadata.foot_slip, cfg.maximum_foot_slip),
        (metadata.reverse_dynamics_residual, cfg.maximum_reverse_dynamics_residual),
        (metadata.actuator_saturation, cfg.maximum_actuator_saturation),
    )
    confidence = torch.ones_like(metadata.contact_impulse)
    for metric, limit in metrics_and_limits:
        if math.isfinite(limit):
            safe_metric = torch.clamp_min(torch.nan_to_num(metric.detach(), nan=limit, posinf=limit), 0.0)
            confidence = torch.minimum(confidence, 1.0 / (1.0 + safe_metric / max(limit, _EPSILON)))
    return confidence.detach().clamp_(0.0, 1.0)


def _reversibility_validity_mask(
    metadata: TRSimulatorValidityMetadata,
    cfg: TRSequenceValidityConfig,
) -> torch.Tensor:
    valid = torch.ones_like(metadata.contact_impulse, dtype=torch.bool)
    for metric, limit in (
        (metadata.contact_impulse, cfg.maximum_contact_impulse),
        (metadata.foot_slip, cfg.maximum_foot_slip),
        (metadata.reverse_dynamics_residual, cfg.maximum_reverse_dynamics_residual),
        (metadata.actuator_saturation, cfg.maximum_actuator_saturation),
    ):
        valid &= torch.isfinite(metric) & (metric >= 0.0)
        if math.isfinite(limit):
            valid &= metric <= limit
    return valid


def _unscaled_velocity_command(frames: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(
        SYMM_QUADRUPED_POLICY_OBS_SCALE.velocity_command,
        dtype=frames.dtype,
        device=frames.device,
    )
    return frames[..., SYMM_QUADRUPED_POLICY_OBS_LAYOUT.velocity_command] / scale


def _relative_gait_signature(
    frames: torch.Tensor,
    minimum_phase_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    phase_sin = frames[..., layout.foot_phase_sin]
    phase_cos = frames[..., layout.foot_phase_cos]
    phase_norm = torch.sqrt(phase_sin.square() + phase_cos.square())
    valid_norm = (phase_norm >= minimum_phase_norm).all(dim=-1)
    safe_norm = phase_norm.clamp_min(minimum_phase_norm)
    phase_sin = phase_sin / safe_norm
    phase_cos = phase_cos / safe_norm
    reference_sin = phase_sin[..., :1]
    reference_cos = phase_cos[..., :1]
    relative_sin = phase_sin * reference_cos - phase_cos * reference_sin
    relative_cos = phase_cos * reference_cos + phase_sin * reference_sin
    return torch.stack((relative_sin, relative_cos), dim=-1), valid_norm


def _values_homogeneous(values: torch.Tensor, tolerance: float) -> torch.Tensor:
    reference = values[..., -1:, :]
    return torch.isclose(values, reference, rtol=0.0, atol=tolerance).all(dim=(-2, -1))


def _relative_gait_homogeneous(
    frames: torch.Tensor,
    cfg: TRSequenceValidityConfig,
) -> torch.Tensor:
    signature, valid_norm = _relative_gait_signature(frames, cfg.minimum_phase_norm)
    reference = signature[..., -1:, :, :]
    chord_error = torch.linalg.vector_norm(signature - reference, dim=-1)
    return valid_norm.all(dim=-1) & (chord_error <= cfg.relative_gait_tolerance).all(dim=(-2, -1))


def _policy_history_task_homogeneous(
    policy_observation: torch.Tensor,
    cfg: TRSequenceValidityConfig,
) -> torch.Tensor:
    frames = unpack_term_major_policy_history(policy_observation)
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    commands = _unscaled_velocity_command(frames)
    periods = frames[..., layout.gait_period]
    duty_factors = frames[..., layout.duty_factor]
    return (
        torch.isfinite(frames).all(dim=(-2, -1))
        & _values_homogeneous(commands, cfg.command_tolerance)
        & _values_homogeneous(periods, cfg.period_tolerance)
        & _values_homogeneous(duty_factors, cfg.duty_factor_tolerance)
        & _relative_gait_homogeneous(frames, cfg)
    )


def _phase_validity_mask(frames: torch.Tensor, cfg: TRSequenceValidityConfig) -> torch.Tensor:
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    phase = torch.remainder(
        torch.atan2(frames[..., layout.foot_phase_sin], frames[..., layout.foot_phase_cos]) / (2.0 * torch.pi),
        1.0,
    )
    swing_ratio = 1.0 - frames[..., layout.duty_factor]
    margin = cfg.phase_boundary_margin

    def circular_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.remainder(first - second + 0.5, 1.0) - 0.5)

    return (
        (circular_distance(phase, torch.zeros_like(phase)) >= margin)
        & (circular_distance(phase, swing_ratio) >= margin)
    ).all(dim=-1)


def _stack_simulator_metadata(records: tuple[TransitionAlignedTRRecord, ...]) -> TRSimulatorValidityMetadata:
    return TRSimulatorValidityMetadata(
        **{
            field.name: torch.stack([getattr(record.simulator, field.name) for record in records], dim=1)
            for field in fields(TRSimulatorValidityMetadata)
        }
    )


def _candidate_batch_where(
    condition: torch.Tensor,
    current: TransitionAlignedTRCandidateBatch,
    replacement: TransitionAlignedTRCandidateBatch,
) -> TransitionAlignedTRCandidateBatch:
    values = {}
    for field in fields(TransitionAlignedTRCandidateBatch):
        current_value = getattr(current, field.name)
        replacement_value = getattr(replacement, field.name)
        expanded_condition = condition.reshape(condition.shape + (1,) * (current_value.ndim - 1))
        values[field.name] = torch.where(expanded_condition, replacement_value, current_value)
    return TransitionAlignedTRCandidateBatch(**values)


class TransitionAlignedTRBuffer:
    """Bounded cross-rollout buffer for exact causal TR consistency.

    Exactly ``history_length + 2`` finalized decision records are needed.  The
    deque is never cleared merely because PPO starts an update, so a 30-frame
    candidate can span the ordinary 24+8 rollout boundary.  Candidate replay is
    bounded by ``candidate_max_age_updates`` and one newest eligible candidate
    per environment/collection update.
    """

    def __init__(
        self,
        num_envs: int,
        history_length: int,
        device: torch.device | str,
        validity_cfg: TRSequenceValidityConfig | None = None,
        candidate_max_age_updates: int = 1,
        allowed_policy_version_span: int = 1,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer; received {num_envs!r}.")
        if isinstance(history_length, bool) or not isinstance(history_length, Integral) or history_length <= 0:
            raise ValueError(f"history_length must be a positive integer; received {history_length!r}.")
        for name, value in (
            ("candidate_max_age_updates", candidate_max_age_updates),
            ("allowed_policy_version_span", allowed_policy_version_span),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")
        self.num_envs = int(num_envs)
        self.history_length = int(history_length)
        self.device = torch.device(device)
        self.validity_cfg = validity_cfg or TRSequenceValidityConfig()
        self.candidate_max_age_updates = int(candidate_max_age_updates)
        self.allowed_policy_version_span = int(allowed_policy_version_span)
        self.required_sequence_records = self.history_length + ACTION_HISTORY_LENGTH
        self._records: deque[_CompactTransitionAlignedTRRecord] = deque(maxlen=self.required_sequence_records)
        # A causal H-frame observation is a sliding window over instantaneous
        # frames. Retaining one full H-frame observation per pending decision
        # would waste O(H^2) device memory. Seed this ring once from the first
        # full observation and append only one 64D frame thereafter.
        self._policy_frames: deque[torch.Tensor] = deque(maxlen=2 * self.history_length + 1)
        self._records_since_clear = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._candidate_windows_by_update: dict[int, list[_CompactCandidateWindow]] = {}
        self._diagnostics: dict[str, float | torch.Tensor] = {}

    @property
    def diagnostics(self) -> dict[str, float]:
        """Return diagnostics from the most recent append/collection operation."""
        return {
            name: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in self._diagnostics.items()
        }

    @property
    def records_collected(self) -> int:
        """Return the retained finalized-record count, capped at the requirement."""
        return len(self._records)

    def _invalidate_candidate_environments(self, environment_mask: torch.Tensor) -> None:
        for windows in self._candidate_windows_by_update.values():
            for window in windows:
                window.valid = window.valid & ~environment_mask

    def clear_environment_mask(self, environment_mask: torch.Tensor) -> None:
        """Invalidate selected environments without materializing CUDA indices."""
        if (
            not isinstance(environment_mask, torch.Tensor)
            or environment_mask.shape != (self.num_envs,)
            or environment_mask.dtype != torch.bool
            or environment_mask.device != self.device
        ):
            raise ValueError(
                f"environment_mask must be a boolean tensor on the buffer device with shape ({self.num_envs},)."
            )
        self._records_since_clear = torch.where(
            environment_mask,
            torch.zeros_like(self._records_since_clear),
            self._records_since_clear,
        )
        self._invalidate_candidate_environments(environment_mask)

    def clear(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Clear all transient state or invalidate selected environments after reset."""
        if env_ids is None:
            self._records.clear()
            self._policy_frames.clear()
            self._records_since_clear = torch.zeros_like(self._records_since_clear)
            self._candidate_windows_by_update.clear()
            self._diagnostics = {}
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError(f"env_ids must be a valid one-dimensional index set for {self.num_envs} environments.")
        environment_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        environment_mask[ids] = True
        self.clear_environment_mask(environment_mask)

    def prepare_policy_history(self, policy_observation: torch.Tensor) -> _PreparedPolicyHistory:
        """Consume one full observation and return only its compact causal state."""
        expected_shape = (self.num_envs, self.history_length * SYMM_QUADRUPED_POLICY_OBS_DIM)
        _validate_tensor_shape(policy_observation, expected_shape, "policy_observation")
        if not torch.is_floating_point(policy_observation) or policy_observation.device != self.device:
            raise ValueError("policy_observation must be floating point and reside on the buffer device.")
        policy_frames = unpack_term_major_policy_history(policy_observation)
        latest_frame = policy_frames[:, -1, :].clone()
        history_consistent = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        policy_history_seed = None
        if not self._policy_frames:
            policy_history_seed = policy_frames
        elif self.history_length > 1:
            expected_prefix = torch.stack(tuple(self._policy_frames)[-(self.history_length - 1) :], dim=1)
            history_consistent &= torch.isclose(
                policy_frames[:, :-1, :],
                expected_prefix,
                rtol=0.0,
                atol=0.0,
            ).all(dim=(-2, -1))
        return _PreparedPolicyHistory(
            latest_frame=latest_frame,
            policy_history_seed=policy_history_seed,
            policy_history_consistent=history_consistent,
        )

    def prepare_record(
        self,
        record: TransitionAlignedTRRecord,
        policy_history: _PreparedPolicyHistory | None = None,
    ) -> _CompactTransitionAlignedTRRecord:
        """Validate and compact a record before its environment transition runs.

        Only the first record seeds a full H-frame history. Later records keep
        their newest 64D frame and an asynchronous per-environment overlap
        check, allowing the caller to release the full observation immediately.
        An inconsistent native history fails closed when candidates are built.
        """
        record.validate(self.num_envs, self.history_length, self.device)
        if policy_history is None:
            policy_history = self.prepare_policy_history(record.policy_observation)
        history_consistent = policy_history.policy_history_consistent & torch.isclose(
            policy_history.latest_frame,
            record.latest_frame,
            rtol=0.0,
            atol=0.0,
        ).all(dim=-1)
        return _CompactTransitionAlignedTRRecord(
            latest_frame=policy_history.latest_frame,
            action=record.action,
            episode_id=record.episode_id,
            command_segment_id=record.command_segment_id,
            gait_segment_id=record.gait_segment_id,
            disturbance_generation_id=record.disturbance_generation_id,
            collection_update_id=record.collection_update_id,
            transition_valid=record.transition_valid,
            done=record.done,
            timeout=record.timeout,
            history_warmup_complete=record.history_warmup_complete,
            gait_row=record.gait_row,
            simulator=record.simulator,
            policy_history_seed=policy_history.policy_history_seed,
            policy_history_consistent=history_consistent,
        )

    def append(
        self,
        record: TransitionAlignedTRRecord | _CompactTransitionAlignedTRRecord,
        *,
        collection_update: int | None = None,
    ) -> None:
        """Append one finalized batched record and reconstruct newly eligible candidates."""
        if isinstance(record, TransitionAlignedTRRecord):
            record = self.prepare_record(record)
        if collection_update is not None and (
            isinstance(collection_update, bool) or not isinstance(collection_update, Integral) or collection_update < 0
        ):
            raise ValueError(f"collection_update must be a nonnegative integer; received {collection_update!r}.")
        if record.policy_history_seed is not None:
            if self._policy_frames:
                raise RuntimeError("A policy-history seed is only valid for an empty sequence buffer.")
            self._policy_frames.extend(record.policy_history_seed.unbind(dim=1))
        else:
            self._policy_frames.append(record.latest_frame)
        usable = record.transition_valid & ~record.done & ~record.timeout
        segment_change = torch.zeros_like(usable)
        if self._records:
            previous = self._records[-1]
            for name in (
                "episode_id",
                "command_segment_id",
                "gait_segment_id",
                "disturbance_generation_id",
            ):
                segment_change |= getattr(previous, name) != getattr(record, name)
        boundary = ~usable | segment_change
        self._invalidate_candidate_environments(boundary)
        next_records_since_clear = torch.where(
            usable,
            self._records_since_clear + 1,
            self._records_since_clear,
        )
        self._records_since_clear = torch.where(
            boundary,
            torch.zeros_like(next_records_since_clear),
            next_records_since_clear,
        )
        self._records.append(record)

        self._diagnostics = {
            "tr_sequence/retained_records": float(len(self._records)),
            "tr_sequence/required_records": float(self.required_sequence_records),
            "tr_sequence/new_candidate_count": 0.0,
        }
        newest_update = (
            int(collection_update) if collection_update is not None else int(record.collection_update_id.max().item())
        )
        if len(self._records) == self.required_sequence_records:
            window = _CompactCandidateWindow(
                records=tuple(self._records),
                policy_frames=tuple(self._policy_frames),
                records_since_clear=self._records_since_clear.clone(),
                valid=torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            )
            self._candidate_windows_by_update.setdefault(newest_update, []).append(window)
        self._expire_candidates(newest_update)

    def _build_dense_candidate(
        self,
        window: _CompactCandidateWindow,
    ) -> tuple[TransitionAlignedTRCandidateBatch, torch.Tensor]:
        records = window.records
        latest_frames = torch.stack([record.latest_frame for record in records], dim=1)
        actions = torch.stack([record.action for record in records], dim=1)
        if len(window.policy_frames) != 2 * self.history_length + 1:
            raise RuntimeError(
                "The compact policy-frame ring is inconsistent with the finalized-record window: "
                f"received {len(window.policy_frames)} frames."
            )
        compact_frames = torch.stack(window.policy_frames, dim=1)
        actor_source = pack_term_major_policy_history(compact_frames[:, : self.history_length])
        value_source = pack_term_major_policy_history(compact_frames[:, 1 : self.history_length + 1])
        reversed_observation = build_transition_aligned_reversed_history(
            latest_frames[:, 1 : self.history_length + 1],
            actions[:, 1 : self.history_length + 2],
            self.history_length,
        )

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(record, name) for record in records], dim=1)

        episode_ids = stack("episode_id")
        command_segment_ids = stack("command_segment_id")
        gait_segment_ids = stack("gait_segment_id")
        disturbance_ids = stack("disturbance_generation_id")
        collection_updates = stack("collection_update_id")
        gait_rows = stack("gait_row")
        policy_version_span = collection_updates.max(dim=1).values - collection_updates.min(dim=1).values
        simulator = _stack_simulator_metadata(records)

        valid = window.valid & (window.records_since_clear >= self.required_sequence_records)
        for name in (
            "episode_id",
            "command_segment_id",
            "gait_segment_id",
            "disturbance_generation_id",
        ):
            values = stack(name)
            valid &= (values == values[:, :1]).all(dim=1)
        valid &= stack("transition_valid").all(dim=1)
        valid &= ~stack("done").any(dim=1)
        valid &= ~stack("timeout").any(dim=1)
        valid &= stack("history_warmup_complete").all(dim=1)
        valid &= stack("policy_history_consistent").all(dim=1)
        valid &= (collection_updates >= 0).all(dim=1)
        valid &= policy_version_span <= self.allowed_policy_version_span
        valid &= torch.isfinite(actor_source).all(dim=-1)
        valid &= torch.isfinite(value_source).all(dim=-1)
        valid &= torch.isfinite(latest_frames).all(dim=(-2, -1))
        valid &= torch.isfinite(actions).all(dim=(-2, -1))
        cfg = self.validity_cfg
        commands = _unscaled_velocity_command(latest_frames)
        layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
        valid &= _values_homogeneous(commands, cfg.command_tolerance)
        valid &= _values_homogeneous(latest_frames[..., layout.gait_period], cfg.period_tolerance)
        valid &= _values_homogeneous(latest_frames[..., layout.duty_factor], cfg.duty_factor_tolerance)
        same_gait_row = (gait_rows == gait_rows[:, :1]).all(dim=1)
        same_relative_gait = _relative_gait_homogeneous(latest_frames, cfg)
        valid &= same_gait_row | same_relative_gait
        valid &= _policy_history_task_homogeneous(actor_source, cfg)

        simulator_finite = torch.isfinite(simulator.body_linear_velocity_b).all(dim=-1)
        simulator_finite &= torch.isfinite(simulator.body_yaw_rate)
        simulator_finite &= torch.isfinite(simulator.projected_gravity).all(dim=-1)
        simulator_finite &= torch.isfinite(simulator.contact_impulse)
        simulator_finite &= torch.isfinite(simulator.foot_slip)
        simulator_finite &= torch.isfinite(simulator.reverse_dynamics_residual)
        simulator_finite &= torch.isfinite(simulator.actuator_saturation)
        valid &= simulator_finite.all(dim=1)

        command_gate = torch.abs(commands[..., 0]) >= cfg.min_abs_command_velocity
        tracking_gate = time_reversal_tracking_mask(simulator, commands, cfg)
        upright_gate = torch.isfinite(simulator.projected_gravity).all(dim=-1) & (
            torch.linalg.vector_norm(simulator.projected_gravity[..., :2], dim=-1) <= cfg.projected_gravity_tolerance
        )
        phase_gate = _phase_validity_mask(latest_frames, cfg)
        mode_gate = {
            "command": command_gate,
            "command_tracking": command_gate & tracking_gate,
            "command_upright_phase": command_gate & upright_gate & phase_gate,
            "command_tracking_upright_phase": command_gate & tracking_gate & upright_gate & phase_gate,
        }[cfg.mode]
        valid &= mode_gate.all(dim=1)
        reversibility_valid = _reversibility_validity_mask(simulator, cfg)
        valid &= reversibility_valid.all(dim=1)
        confidence = time_reversal_reversibility_confidence(simulator, cfg).amin(dim=1).unsqueeze(-1)
        valid &= confidence.squeeze(-1) > 0.0

        environment_id = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        candidate = TransitionAlignedTRCandidateBatch(
            actor_source_observation=actor_source,
            value_source_observation=value_source,
            reversed_observation=reversed_observation,
            weight=confidence.detach(),
            environment_id=environment_id,
            episode_id=episode_ids[:, 0],
            command_segment_id=command_segment_ids[:, 0],
            gait_segment_id=gait_segment_ids[:, 0],
            disturbance_generation_id=disturbance_ids[:, 0],
            oldest_collection_update_id=collection_updates.min(dim=1).values,
            newest_collection_update_id=collection_updates.max(dim=1).values,
            policy_version_span=policy_version_span,
            candidate_age_updates=zeros,
        )
        return candidate, valid

    def _expire_candidates(self, current_update: int) -> None:
        expired = [
            update
            for update in self._candidate_windows_by_update
            if current_update - update > self.candidate_max_age_updates or update > current_update
        ]
        for update in expired:
            del self._candidate_windows_by_update[update]

    def _collect_candidates(
        self,
        current_update: int,
        *,
        consume: bool,
    ) -> TransitionAlignedTRCandidateBatch | None:
        if isinstance(current_update, bool) or not isinstance(current_update, Integral) or current_update < 0:
            raise ValueError(f"current_update must be a nonnegative integer; received {current_update!r}.")
        current_update = int(current_update)
        self._expire_candidates(current_update)
        batches = []
        consumed_updates = []
        newest_candidate_count = torch.zeros((), dtype=torch.long, device=self.device)
        newest_policy_version_span = torch.zeros((), dtype=torch.long, device=self.device)
        for update, windows in sorted(self._candidate_windows_by_update.items()):
            snapshot: _DenseCandidateSnapshot | None = None
            for window in windows:
                candidate, valid = self._build_dense_candidate(window)
                valid &= candidate.newest_collection_update_id == update
                newest_candidate_count = valid.sum()
                newest_policy_version_span = torch.where(
                    valid,
                    candidate.policy_version_span,
                    torch.zeros_like(candidate.policy_version_span),
                ).max()
                if snapshot is None:
                    snapshot = _DenseCandidateSnapshot(candidate, valid)
                else:
                    snapshot.batch = _candidate_batch_where(valid, snapshot.batch, candidate)
                    snapshot.valid |= valid
            assert snapshot is not None
            if not torch.any(snapshot.valid):
                consumed_updates.append(update)
                continue
            age = current_update - update
            selected = snapshot.batch.index(snapshot.valid)
            selected = replace(
                selected,
                candidate_age_updates=torch.full_like(selected.candidate_age_updates, age),
            )
            batches.append(selected)
            if consume:
                consumed_updates.append(update)
        for update in consumed_updates:
            del self._candidate_windows_by_update[update]
        self._diagnostics["tr_sequence/new_candidate_count"] = newest_candidate_count
        self._diagnostics["tr_sequence/policy_version_span_max"] = newest_policy_version_span
        if not batches:
            self._diagnostics["tr_sequence/candidate_count"] = 0.0
            return None
        result = TransitionAlignedTRCandidateBatch.concatenate(batches)
        self._diagnostics["tr_sequence/candidate_count"] = float(result.count)
        self._diagnostics["tr_sequence/candidate_age_updates_max"] = float(result.candidate_age_updates.max().item())
        return result

    def peek_candidates(self, current_update: int) -> TransitionAlignedTRCandidateBatch | None:
        """Return currently eligible candidates without consuming them."""
        return self._collect_candidates(current_update, consume=False)

    def pop_candidates(
        self,
        current_update: int,
        consume: bool = True,
    ) -> TransitionAlignedTRCandidateBatch | None:
        """Return eligible candidates and consume their bounded replay snapshots by default."""
        return self._collect_candidates(current_update, consume=consume)

    def candidate_count(self, current_update: int) -> int:
        """Return the number of currently eligible candidates without consuming them."""
        candidates = self.peek_candidates(current_update)
        return 0 if candidates is None else candidates.count

    def state_dict(self) -> dict:
        """Reject accidental checkpointing of rollout-local future context."""
        raise RuntimeError("TransitionAlignedTRBuffer is transient and must never be checkpointed.")

    def load_state_dict(self, state: object | None = None) -> None:
        """Discard all transient records when a checkpoint is loaded."""
        del state
        self.clear()
