# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure numerical helpers for symmetric quadruped time-reversal training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS,
    SYMM_QUADRUPED_POLICY_OBS_DIM,
    SYMM_QUADRUPED_POLICY_OBS_LAYOUT,
    SYMM_QUADRUPED_POLICY_OBS_SCALE,
)

_RAMP_SHAPES = frozenset({"linear", "half_cosine"})


def validate_legacy_observation_metadata(symmetry: Mapping) -> None:
    """Validate legacy command-index aliases against centralized 72D metadata."""
    expected_index = SYMM_QUADRUPED_POLICY_OBS_LAYOUT.desired_base_twist.start
    expected_scale = SYMM_QUADRUPED_POLICY_OBS_SCALE.desired_base_twist[0]
    configured_index = symmetry.get("command_observation_index", expected_index)
    configured_scale = symmetry.get("command_observation_scale", expected_scale)
    if configured_index != expected_index:
        raise ValueError(
            "command_observation_index is a deprecated alias and must match the centralized policy layout "
            f"index {expected_index}; received {configured_index!r}."
        )
    if (
        isinstance(configured_scale, bool)
        or not isinstance(configured_scale, Real)
        or not math.isfinite(float(configured_scale))
        or not math.isclose(float(configured_scale), expected_scale, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError(
            "command_observation_scale is a deprecated alias and must match the centralized policy scale "
            f"{expected_scale}; received {configured_scale!r}."
        )


@dataclass(frozen=True)
class ResolvedTimeReversalSchedule:
    """Fully resolved schedule for one auxiliary objective."""

    enabled: bool
    target_coeff: float
    warmup_iterations: int
    rampup_iterations: int
    hold_iterations: int
    decay_iterations: int
    final_scale: float
    ramp_shape: str

    def __post_init__(self) -> None:
        """Validate the fully resolved schedule."""
        if not isinstance(self.enabled, bool):
            raise ValueError(f"schedule.enabled must be boolean; received {self.enabled!r}.")
        if (
            isinstance(self.target_coeff, bool)
            or not isinstance(self.target_coeff, Real)
            or not math.isfinite(float(self.target_coeff))
            or self.target_coeff < 0.0
        ):
            raise ValueError(f"schedule.target_coeff must be finite and nonnegative; received {self.target_coeff!r}.")

    def scale(self, iteration: int) -> float:
        """Return the coefficient scale for an absolute zero-based PPO update."""
        return time_reversal_schedule_scale(
            iteration,
            self.warmup_iterations,
            self.rampup_iterations,
            self.hold_iterations,
            self.decay_iterations,
            self.final_scale,
            self.ramp_shape,
        )

    def coefficient(self, iteration: int) -> float:
        """Return the effective coefficient for an absolute zero-based PPO update."""
        return self.target_coeff * self.scale(iteration) if self.enabled else 0.0


def _interpolate(progress: float, shape: str) -> float:
    if shape == "linear":
        return progress
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def time_reversal_schedule_scale(
    iteration: int,
    warmup_iterations: int,
    rampup_iterations: int,
    hold_iterations: int,
    decay_iterations: int,
    final_scale: float,
    ramp_shape: str,
) -> float:
    """Evaluate a warm-up/ramp/hold/decay schedule.

    Zero-duration phases are interpreted as exact boundary transitions. A
    positive ramp includes both endpoints. A positive decay starts at one and
    includes its final-scale endpoint.
    """
    if isinstance(iteration, bool) or not isinstance(iteration, Integral) or iteration < 0:
        raise ValueError(f"iteration must be a nonnegative integer; received {iteration!r}.")
    if ramp_shape not in _RAMP_SHAPES:
        raise ValueError(f"ramp_shape must be one of {sorted(_RAMP_SHAPES)!r}; received {ramp_shape!r}.")
    for name, value in (
        ("warmup_iterations", warmup_iterations),
        ("rampup_iterations", rampup_iterations),
        ("hold_iterations", hold_iterations),
        ("decay_iterations", decay_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")
    if isinstance(final_scale, bool) or not isinstance(final_scale, Real) or not math.isfinite(float(final_scale)):
        raise ValueError(f"final_scale must be finite and in [0, 1]; received {final_scale!r}.")
    if not 0.0 <= final_scale <= 1.0:
        raise ValueError(f"final_scale must be finite and in [0, 1]; received {final_scale!r}.")

    if iteration < warmup_iterations:
        return 0.0
    elapsed = iteration - warmup_iterations
    if rampup_iterations > 0 and elapsed <= rampup_iterations:
        return _interpolate(elapsed / rampup_iterations, ramp_shape)

    after_ramp = elapsed - rampup_iterations
    if after_ramp <= hold_iterations:
        return 1.0
    if decay_iterations == 0:
        return final_scale
    decay_progress = min(max((after_ramp - hold_iterations) / decay_iterations, 0.0), 1.0)
    decay_curve = _interpolate(decay_progress, ramp_shape)
    return 1.0 + (final_scale - 1.0) * decay_curve


def resolve_time_reversal_schedule(symmetry: Mapping, term: str) -> ResolvedTimeReversalSchedule:
    """Resolve a canonical policy, value, or augmentation schedule with aliases."""
    if term not in {"policy", "value", "augmentation"}:
        raise ValueError(f"term must be 'policy', 'value', or 'augmentation'; received {term!r}.")

    if term == "policy":
        schedule = symmetry.get("tr_policy_schedule") or {}
        canonical_enabled = symmetry.get("use_tr_policy_consistency")
        legacy_enabled = bool(symmetry.get("use_mirror_loss", False))
        legacy_target = float(symmetry.get("mirror_loss_coeff", 0.0))
    elif term == "value":
        schedule = symmetry.get("tr_value_schedule") or {}
        canonical_enabled = symmetry.get("use_tr_value_consistency")
        legacy_target = float(symmetry.get("value_loss_coeff", 0.0))
        legacy_enabled = legacy_target > 0.0
    else:
        augmentation = symmetry.get("tr_augmentation") or {}
        schedule = augmentation.get("schedule") or {}
        canonical_enabled = augmentation.get("enabled", False)
        legacy_enabled = False
        legacy_target = float(augmentation.get("coefficient", 0.0))

    if canonical_enabled is not None and not isinstance(canonical_enabled, bool):
        raise ValueError(f"Canonical {term} consistency enabled flag must be boolean or None.")
    if canonical_enabled is None:
        master_enabled = bool(symmetry.get("use_time_reversal_regularization", False))
        consistency_enabled = master_enabled and legacy_enabled
    else:
        consistency_enabled = bool(canonical_enabled)
    schedule_enabled = schedule.get("enabled")
    if schedule_enabled is not None and not isinstance(schedule_enabled, bool):
        raise ValueError(f"{term} schedule.enabled must be boolean or None.")
    enabled = consistency_enabled if schedule_enabled is None else consistency_enabled and bool(schedule_enabled)

    target = schedule.get("target_coeff")
    warmup = schedule.get("warmup_iterations")
    rampup = schedule.get("rampup_iterations")
    ramp_shape = schedule.get("ramp_shape")
    return ResolvedTimeReversalSchedule(
        enabled=enabled,
        target_coeff=legacy_target if target is None else target,
        warmup_iterations=symmetry.get("warmup_iterations", 0) if warmup is None else warmup,
        rampup_iterations=symmetry.get("rampup_iterations", 0) if rampup is None else rampup,
        hold_iterations=schedule.get("hold_iterations", 0),
        decay_iterations=schedule.get("decay_iterations", 0),
        final_scale=schedule.get("final_scale", 1.0),
        ramp_shape=str(symmetry.get("ramp_shape", "linear") if ramp_shape is None else ramp_shape),
    )


@dataclass(frozen=True)
class TimeReversalValidityMask:
    """Combined and component masks derived from historical policy observations."""

    combined: torch.Tensor
    command: torch.Tensor
    tracking: torch.Tensor
    upright: torch.Tensor
    phase: torch.Tensor
    command_velocity: torch.Tensor
    command_sign: torch.Tensor
    command_speed_bin: torch.Tensor
    gait_family: torch.Tensor
    num_command_speed_bins: int


def _policy_observation(observations) -> torch.Tensor:
    policy_obs = observations["policy"]
    if policy_obs.ndim != 2 or policy_obs.shape[-1] != SYMM_QUADRUPED_POLICY_OBS_DIM:
        raise ValueError(
            f"Expected historical policy observations with shape (batch, {SYMM_QUADRUPED_POLICY_OBS_DIM}), "
            f"got {tuple(policy_obs.shape)}."
        )
    return policy_obs


def _circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.remainder(a - b + 0.5, 1.0) - 0.5)


def _gait_family_indices(policy_obs: torch.Tensor) -> torch.Tensor:
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    theta = torch.remainder(
        torch.atan2(policy_obs[:, layout.foot_theta_sin], policy_obs[:, layout.foot_theta_cos]) / (2.0 * torch.pi),
        1.0,
    )
    rows = torch.as_tensor(
        SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS,
        device=policy_obs.device,
        dtype=policy_obs.dtype,
    )
    row_distance = _circular_distance(theta[:, None, :], rows[None, :, :]).mean(dim=-1)
    row_index = row_distance.argmin(dim=-1)
    family_names = tuple(dict.fromkeys(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES))
    family_by_row = torch.tensor(
        [family_names.index(name) for name in SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES],
        device=policy_obs.device,
        dtype=torch.long,
    )
    return family_by_row[row_index]


def time_reversal_validity_mask(observations, cfg: Mapping, legacy_min_abs_command: float = 0.0):
    """Build heuristic validity masks solely from a historical minibatch observation."""
    policy_obs = _policy_observation(observations)
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    scales = SYMM_QUADRUPED_POLICY_OBS_SCALE

    command_velocity = policy_obs[:, layout.desired_base_twist.start] / scales.desired_base_twist[0]
    measured_velocity = policy_obs[:, layout.measured_base_twist.start] / scales.measured_base_twist[0]
    min_command = cfg.get("min_abs_command_velocity")
    min_command = legacy_min_abs_command if min_command is None else float(min_command)
    command = torch.abs(command_velocity) >= min_command

    tracking_limit = float(cfg.get("tracking_abs_tolerance", 0.25)) + float(
        cfg.get("tracking_rel_tolerance", 0.25)
    ) * torch.abs(command_velocity)
    tracking = torch.abs(measured_velocity - command_velocity) <= tracking_limit

    gravity = policy_obs[:, layout.projected_gravity]
    gravity_scale = torch.as_tensor(scales.projected_gravity, device=policy_obs.device, dtype=policy_obs.dtype)
    gravity = gravity / gravity_scale
    upright = torch.linalg.vector_norm(gravity[:, :2], dim=-1) <= float(cfg.get("projected_gravity_tolerance", 0.35))

    phase_values = torch.remainder(
        torch.atan2(policy_obs[:, layout.foot_phase_sin], policy_obs[:, layout.foot_phase_cos]) / (2.0 * torch.pi),
        1.0,
    )
    swing_ratio = policy_obs[:, layout.swing_ratio]
    margin = float(cfg.get("phase_boundary_margin", 0.03))
    phase = (
        (_circular_distance(phase_values, torch.zeros_like(phase_values)) >= margin)
        & (_circular_distance(phase_values, swing_ratio) >= margin)
    ).all(dim=-1)

    mode = cfg.get("mode", "command")
    components = {
        "command": command,
        "command_tracking": command & tracking,
        "command_upright_phase": command & upright & phase,
        "command_tracking_upright_phase": command & tracking & upright & phase,
    }
    if mode not in components:
        raise ValueError(f"Unsupported time-reversal validity-mask mode: {mode!r}.")

    edges = torch.as_tensor(
        cfg.get("command_speed_bin_edges", (0.5, 1.0, 1.5)),
        device=policy_obs.device,
        dtype=policy_obs.dtype,
    )
    speed_bin = torch.bucketize(torch.abs(command_velocity), edges)
    command_sign = torch.sign(command_velocity).to(dtype=torch.long)
    dtype = policy_obs.dtype
    return TimeReversalValidityMask(
        combined=components[mode].unsqueeze(-1).to(dtype=dtype),
        command=command.unsqueeze(-1).to(dtype=dtype),
        tracking=tracking.unsqueeze(-1).to(dtype=dtype),
        upright=upright.unsqueeze(-1).to(dtype=dtype),
        phase=phase.unsqueeze(-1).to(dtype=dtype),
        command_velocity=command_velocity,
        command_sign=command_sign,
        command_speed_bin=speed_bin,
        gait_family=_gait_family_indices(policy_obs),
        num_command_speed_bins=len(edges) + 1,
    )


def time_reversal_mask_diagnostics(mask: TimeReversalValidityMask) -> dict[str, float]:
    """Summarize overall and stratified historical-mask acceptance."""
    diagnostics = {
        "tr_mask/command_acceptance": mask.command.mean().item(),
        "tr_mask/tracking_acceptance": mask.tracking.mean().item(),
        "tr_mask/upright_acceptance": mask.upright.mean().item(),
        "tr_mask/phase_acceptance": mask.phase.mean().item(),
        "tr_mask/combined_acceptance": mask.combined.mean().item(),
    }
    combined = mask.combined.squeeze(-1)
    family_names = tuple(dict.fromkeys(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES))
    for index, name in enumerate(family_names):
        selected = mask.gait_family == index
        diagnostics[f"tr_mask/gait_family/{name}"] = combined[selected].mean().item() if torch.any(selected) else 0.0
    for sign, name in ((-1, "negative"), (0, "zero"), (1, "positive")):
        selected = mask.command_sign == sign
        diagnostics[f"tr_mask/command_sign/{name}"] = combined[selected].mean().item() if torch.any(selected) else 0.0
    for index in range(mask.num_command_speed_bins):
        selected = mask.command_speed_bin == index
        diagnostics[f"tr_mask/command_speed_bin/{index}"] = (
            combined[selected].mean().item() if torch.any(selected) else 0.0
        )
    return diagnostics


def normalize_requested_joint_targets(
    action_mean: torch.Tensor,
    action_offset: torch.Tensor,
    action_scale: torch.Tensor,
    soft_joint_position_limits: torch.Tensor,
) -> torch.Tensor:
    """Map policy means to requested joint targets normalized by each joint's feasible interval.

    The requested target is evaluated before the simulator safety clamp.  A
    value of ``-1`` denotes the lower soft limit and ``1`` denotes the upper
    soft limit; infeasible requests intentionally remain outside that range.

    Args:
        action_mean: Raw policy action means, shape ``(..., num_joints)``.
        action_offset: Default joint positions [rad], broadcastable to ``action_mean``.
        action_scale: Joint-position action scales [rad], broadcastable to ``action_mean``.
        soft_joint_position_limits: Lower and upper soft limits [rad], shape
            broadcastable to ``(..., num_joints, 2)``.

    Returns:
        Normalized requested joint targets, shape ``(..., num_joints)``.

    Raises:
        ValueError: If the metadata dimensions, scales, or limit ranges are invalid.
    """
    if action_mean.ndim < 1:
        raise ValueError("action_mean must have at least one dimension.")
    joint_count = action_mean.shape[-1]
    if action_offset.shape[-1] != joint_count or action_scale.shape[-1] != joint_count:
        raise ValueError(
            "Action offset and scale must follow policy joint order and match the actor width; "
            f"received {action_offset.shape[-1]}, {action_scale.shape[-1]}, and {joint_count}."
        )
    if soft_joint_position_limits.shape[-2:] != (joint_count, 2):
        raise ValueError(
            "soft_joint_position_limits must have trailing shape (num_joints, 2); "
            f"received {tuple(soft_joint_position_limits.shape)} for {joint_count} joints."
        )
    if not torch.all(torch.isfinite(action_offset)) or not torch.all(torch.isfinite(action_scale)):
        raise ValueError("Action offsets and scales must be finite.")
    if torch.any(action_scale == 0.0):
        raise ValueError("Joint-position action scales must be nonzero.")
    lower = soft_joint_position_limits[..., 0]
    upper = soft_joint_position_limits[..., 1]
    limit_range = upper - lower
    if not torch.all(torch.isfinite(soft_joint_position_limits)) or torch.any(limit_range <= 0.0):
        raise ValueError("Every soft joint-position limit must be finite with upper greater than lower.")
    requested_target = action_offset + action_scale * action_mean
    epsilon = torch.finfo(action_mean.dtype).eps
    return (2.0 * requested_target - (upper + lower)) / (limit_range + epsilon)


def feasible_actor_mean_penalty(
    actor_mean: torch.Tensor,
    lower_action_bound: torch.Tensor,
    upper_action_bound: torch.Tensor,
    *,
    interior_margin_fraction: float = 0.0,
) -> torch.Tensor:
    """Return the smooth-L1 penalty outside per-joint feasible action bounds.

    Args:
        actor_mean: Raw policy action means, shape ``(..., num_joints)``.
        lower_action_bound: Feasible lower raw-action bound per joint.
        upper_action_bound: Feasible upper raw-action bound per joint.
        interior_margin_fraction: Fraction of each feasible raw-action range
            excluded at both interval ends.

    Returns:
        Scalar dimensionless smooth-L1 penalty.
    """
    if not 0.0 <= interior_margin_fraction < 0.5:
        raise ValueError(f"interior_margin_fraction must be in [0, 0.5); received {interior_margin_fraction!r}.")
    if lower_action_bound.shape[-1] != actor_mean.shape[-1] or upper_action_bound.shape[-1] != actor_mean.shape[-1]:
        raise ValueError("Per-joint feasible bounds must match the actor output width and ordering.")
    action_range = upper_action_bound - lower_action_bound
    if not torch.all(torch.isfinite(lower_action_bound)) or not torch.all(torch.isfinite(upper_action_bound)):
        raise ValueError("Per-joint feasible action bounds must be finite.")
    if torch.any(action_range <= 0.0):
        raise ValueError("Every feasible upper action bound must be greater than its lower bound.")
    interior_margin = interior_margin_fraction * action_range
    lower_overflow = torch.relu(lower_action_bound + interior_margin - actor_mean) / action_range
    upper_overflow = torch.relu(actor_mean - upper_action_bound + interior_margin) / action_range
    # Smooth-L1 against zero with beta=1.  The explicit form avoids creating a
    # target tensor and keeps the exact uncapped overflow magnitude.
    overflow = torch.cat((lower_overflow, upper_overflow), dim=-1)
    smooth_l1 = torch.where(overflow < 1.0, 0.5 * overflow.square(), overflow - 0.5)
    return smooth_l1.sum(dim=-1).mean() / actor_mean.shape[-1]


def feasible_actor_mean_diagnostics(
    actor_mean: torch.Tensor,
    lower_action_bound: torch.Tensor,
    upper_action_bound: torch.Tensor,
    *,
    interior_margin_fraction: float = 0.0,
) -> dict[str, float]:
    """Summarize actor means relative to the per-joint feasible intervals.

    The margin statistics are signed raw-action distances to the nearest
    interior boundary.  Negative values therefore identify infeasible means.
    """
    if not 0.0 <= interior_margin_fraction < 0.5:
        raise ValueError(f"interior_margin_fraction must be in [0, 0.5); received {interior_margin_fraction!r}.")
    action_range = upper_action_bound - lower_action_bound
    if lower_action_bound.shape[-1] != actor_mean.shape[-1] or upper_action_bound.shape[-1] != actor_mean.shape[-1]:
        raise ValueError("Per-joint feasible bounds must match the actor output width and ordering.")
    if torch.any(action_range <= 0.0):
        raise ValueError("Every feasible upper action bound must be greater than its lower bound.")
    interior_margin = interior_margin_fraction * action_range
    lower = lower_action_bound + interior_margin
    upper = upper_action_bound - interior_margin
    signed_margin = torch.minimum(actor_mean - lower, upper - actor_mean)
    outside = signed_margin < 0.0
    flattened_margin = signed_margin.detach().reshape(-1).to(dtype=torch.float64)
    return {
        "diagnostics/actor_mean_outside_feasible_fraction": outside.to(actor_mean.dtype).mean().item(),
        "diagnostics/actor_mean_feasible_margin_min": flattened_margin.min().item(),
        "diagnostics/actor_mean_feasible_margin_p05": torch.quantile(flattened_margin, 0.05).item(),
    }


def _loss_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if not loss.requires_grad:
        return (None,) * len(parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)


def time_reversal_gradient_diagnostics(
    main_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    effective_coefficient: float,
    prefix: str,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """Compute non-mutating gradient norms, cosine, coverage, and finiteness."""
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    main_grads = _loss_gradients(main_loss, parameters, retain_graph=True)
    auxiliary_grads = _loss_gradients(auxiliary_loss, parameters, retain_graph=True)

    main_sq = torch.zeros((), dtype=torch.float64, device=main_loss.device)
    auxiliary_sq = torch.zeros_like(main_sq)
    dot = torch.zeros_like(main_sq)
    main_parameter_count = 0
    auxiliary_parameter_count = 0
    total_parameter_count = len(parameters)
    nonfinite = False
    for main_grad, auxiliary_grad in zip(main_grads, auxiliary_grads):
        if main_grad is not None:
            nonfinite |= not bool(torch.all(torch.isfinite(main_grad)))
            safe_main = torch.nan_to_num(main_grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float64)
            main_sq += torch.sum(safe_main * safe_main)
            main_parameter_count += 1
        else:
            safe_main = None
        if auxiliary_grad is not None:
            nonfinite |= not bool(torch.all(torch.isfinite(auxiliary_grad)))
            safe_auxiliary = torch.nan_to_num(auxiliary_grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).to(
                dtype=torch.float64
            )
            auxiliary_sq += torch.sum(safe_auxiliary * safe_auxiliary)
            auxiliary_parameter_count += 1
        else:
            safe_auxiliary = None
        if safe_main is not None and safe_auxiliary is not None:
            dot += torch.sum(safe_main * safe_auxiliary)

    main_norm = torch.sqrt(main_sq)
    auxiliary_norm = torch.sqrt(auxiliary_sq)
    denom = main_norm * auxiliary_norm
    cosine = dot / (denom + epsilon)
    weighted_norm = float(effective_coefficient) * auxiliary_norm
    weighted_ratio = weighted_norm / (main_norm + epsilon)
    total_parameter_count = max(total_parameter_count, 1)
    return {
        f"tr_gradient/{prefix}/main_norm": main_norm.item(),
        f"tr_gradient/{prefix}/auxiliary_norm": auxiliary_norm.item(),
        f"tr_gradient/{prefix}/weighted_auxiliary_norm": weighted_norm.item(),
        f"tr_gradient/{prefix}/cosine": cosine.item(),
        f"tr_gradient/{prefix}/weighted_ratio": weighted_ratio.item(),
        f"tr_gradient/{prefix}/main_parameter_fraction": main_parameter_count / total_parameter_count,
        f"tr_gradient/{prefix}/auxiliary_parameter_fraction": auxiliary_parameter_count / total_parameter_count,
        f"tr_gradient/{prefix}/nonfinite": float(nonfinite),
    }
