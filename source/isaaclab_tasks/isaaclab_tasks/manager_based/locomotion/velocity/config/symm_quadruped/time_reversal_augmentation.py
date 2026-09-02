# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-derived time-reversal action supervision for symmetric quadrupeds.

This module deliberately implements an auxiliary supervised actor loss, not an
augmented PPO transition loss. A reversed transition starts at
``T_X(x_{t+1})`` and its action was not sampled by the rollout behavior policy
at that state. Consequently the original old log probability, advantage,
return, and value are invalid for the reversed sample and are never stored in
or consumed by this subsystem.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS,
    SYMM_QUADRUPED_POLICY_OBS_DIM,
    SYMM_QUADRUPED_POLICY_OBS_LAYOUT,
    SYMM_QUADRUPED_POLICY_OBS_SCALE,
    compute_dimensionless_gait_period,
)

_EPS = 1.0e-8


def _tensor(value) -> torch.Tensor:
    """Return the torch view of an Isaac Lab proxy array or a tensor."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    try:
        import warp as wp  # noqa: PLC0415

        return wp.to_torch(value)
    except (AttributeError, RuntimeError, TypeError) as error:
        raise TypeError(f"Cannot convert {type(value).__name__} to a torch tensor.") from error


def quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``(w, x, y, z)`` quaternions to a continuous 6D rotation."""
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(_EPS)
    w, x, y, z = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return rotation[..., :, :2].reshape(*quaternion.shape[:-1], 6)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert a continuous 6D rotation representation to a rotation matrix."""
    two_columns = rotation_6d.reshape(*rotation_6d.shape[:-1], 3, 2)
    first = nn.functional.normalize(two_columns[..., :, 0], dim=-1)
    second_raw = two_columns[..., :, 1]
    second = nn.functional.normalize(second_raw - (first * second_raw).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


@dataclass(frozen=True)
class TRDynamicsFeatureLayout:
    """Component indices in :meth:`TRDynamicsState.dynamics_features`."""

    rotation: slice
    actuator_target: slice
    previous_action: slice
    second_previous_action: slice
    common_phase_sin: int
    common_phase_cos: int
    foot_phase_sin: slice
    foot_phase_cos: slice
    dimension: int


def _circular_squared_error(
    prediction_sin: torch.Tensor,
    prediction_cos: torch.Tensor,
    target_sin: torch.Tensor,
    target_cos: torch.Tensor,
) -> torch.Tensor:
    prediction_radius = torch.linalg.vector_norm(torch.stack((prediction_sin, prediction_cos), dim=-1), dim=-1)
    target_radius = torch.linalg.vector_norm(torch.stack((target_sin, target_cos), dim=-1), dim=-1)
    prediction_sin = prediction_sin / prediction_radius.clamp_min(_EPS)
    prediction_cos = prediction_cos / prediction_radius.clamp_min(_EPS)
    target_sin = target_sin / target_radius.clamp_min(_EPS)
    target_cos = target_cos / target_radius.clamp_min(_EPS)
    sine_delta = prediction_sin * target_cos - prediction_cos * target_sin
    cosine_delta = prediction_cos * target_cos + prediction_sin * target_sin
    wrapped_delta_cycles = torch.atan2(sine_delta, cosine_delta) / (2.0 * torch.pi)
    radius_error = (prediction_radius - target_radius).square()
    return wrapped_delta_cycles.square() + radius_error


def time_reversal_dynamics_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    feature_variance: torch.Tensor,
    layout: TRDynamicsFeatureLayout,
) -> torch.Tensor:
    """Return one component-aware normalized dynamics residual per sample.

    Euclidean state components use empirical inverse-variance weighting. The
    redundant 6D orientation coordinates are replaced by the squared SO(3)
    geodesic angle normalized by pi, and each sine/cosine phase pair is
    replaced by its squared shortest wrapped distance in cycles plus a
    unit-circle radial penalty. Thus the weighting does not double-count
    embedding coordinates, admit a zero-vector shortcut, or introduce a
    discontinuity at phase wrap-around.
    """
    if prediction.shape != target.shape or prediction.shape[-1] != layout.dimension:
        raise ValueError("Dynamics prediction, target, and feature layout dimensions must match.")
    scale = torch.sqrt(feature_variance + 1.0e-6)
    physical_prediction = prediction * scale + feature_mean
    physical_target = target * scale + feature_mean

    euclidean = (prediction - target).square()
    euclidean[..., layout.rotation] = 0.0
    euclidean[..., layout.common_phase_sin] = 0.0
    euclidean[..., layout.common_phase_cos] = 0.0
    euclidean[..., layout.foot_phase_sin] = 0.0
    euclidean[..., layout.foot_phase_cos] = 0.0

    prediction_rotation = rotation_6d_to_matrix(physical_prediction[..., layout.rotation])
    target_rotation = rotation_6d_to_matrix(physical_target[..., layout.rotation])
    chord = torch.linalg.matrix_norm(prediction_rotation - target_rotation, dim=(-2, -1))
    rotation_angle = 2.0 * torch.asin((chord / (2.0 * math.sqrt(2.0))).clamp(0.0, 1.0))
    rotation_error = (rotation_angle / torch.pi).square()

    common_phase_error = _circular_squared_error(
        physical_prediction[..., layout.common_phase_sin],
        physical_prediction[..., layout.common_phase_cos],
        physical_target[..., layout.common_phase_sin],
        physical_target[..., layout.common_phase_cos],
    )
    foot_phase_error = _circular_squared_error(
        physical_prediction[..., layout.foot_phase_sin],
        physical_prediction[..., layout.foot_phase_cos],
        physical_target[..., layout.foot_phase_sin],
        physical_target[..., layout.foot_phase_cos],
    ).sum(dim=-1)
    foot_count = layout.foot_phase_sin.stop - layout.foot_phase_sin.start
    euclidean_count = layout.dimension - 6 - 2 - 2 * foot_count
    component_count = euclidean_count + 1 + 1 + foot_count
    return (euclidean.sum(dim=-1) + rotation_error + common_phase_error + foot_phase_error) / component_count


@dataclass(frozen=True)
class TRDynamicsState:
    """Batched Markov state used by the one-step augmentation models.

    Temporal parity is explicit: root position, rotation, joint position,
    position-controller targets, prior actions, foot offsets, duty ratios,
    gait period, gait identity, body masses, and rigid-shape material
    properties are even; linear/angular and joint velocities plus the velocity
    command are odd. Common gait phase follows the duty-aware reflection
    ``swing_ratio - phase``. The current controllers use no hidden actuator
    state, so :attr:`additional_actuator_state` has zero width and is treated as
    even. Environments with nonempty hidden actuator state must add an explicit
    parity transform before enabling augmentation.

    Root position is relative to the environment origin [m]. Root velocities
    are expressed in the body frame [m/s, rad/s]. Joint position is relative to
    the reset/default pose [rad], while joint velocity is [rad/s]. Gait phase
    and offsets are measured in cycles and gait period in seconds. Body masses
    are [kg]. Material components are static friction, dynamic friction, and
    restitution, all dimensionless.
    """

    root_position: torch.Tensor
    root_rotation_6d: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    actuator_target: torch.Tensor
    previous_action: torch.Tensor
    second_previous_action: torch.Tensor
    velocity_command: torch.Tensor
    common_gait_phase: torch.Tensor
    foot_phase_offsets: torch.Tensor
    swing_ratio: torch.Tensor
    stance_ratio: torch.Tensor
    gait_period: torch.Tensor
    gait_row: torch.Tensor
    body_mass: torch.Tensor
    material_properties: torch.Tensor
    additional_actuator_state: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Number of states in the batch."""
        return self.root_position.shape[0]

    def index(self, index) -> TRDynamicsState:
        """Return a batch-preserving indexed state."""
        if isinstance(index, int):
            index = slice(index, index + 1)
        return TRDynamicsState(**{field.name: getattr(self, field.name)[index] for field in fields(self)})

    def clone(self) -> TRDynamicsState:
        """Clone all state tensors."""
        return TRDynamicsState(**{field.name: getattr(self, field.name).clone() for field in fields(self)})

    def detach(self) -> TRDynamicsState:
        """Detach every state tensor from autograd while retaining the batch."""
        return TRDynamicsState(**{field.name: getattr(self, field.name).detach() for field in fields(self)})

    def finite_mask(self) -> torch.Tensor:
        """Return environments whose complete floating-point state is finite."""
        finite = torch.ones(self.batch_size, dtype=torch.bool, device=self.root_position.device)
        for field in fields(self):
            value = getattr(self, field.name)
            if value.dtype.is_floating_point:
                finite &= torch.isfinite(value.reshape(self.batch_size, -1)).all(dim=-1)
        return finite

    @classmethod
    def concatenate(cls, states: list[TRDynamicsState]) -> TRDynamicsState:
        """Concatenate state batches."""
        if not states:
            raise ValueError("Cannot concatenate an empty state list.")
        return cls(**{field.name: torch.cat([getattr(state, field.name) for state in states]) for field in fields(cls)})

    def dynamics_feature_layout(self) -> TRDynamicsFeatureLayout:
        """Return manifold-component indices for :meth:`dynamics_features`."""

        def width(value: torch.Tensor) -> int:
            return math.prod(value.shape[1:])

        cursor = width(self.root_position)
        rotation = slice(cursor, cursor + width(self.root_rotation_6d))
        cursor = rotation.stop
        for value in (
            self.root_linear_velocity,
            self.root_angular_velocity,
            self.joint_position,
            self.joint_velocity,
        ):
            cursor += width(value)
        actuator_target = slice(cursor, cursor + width(self.actuator_target))
        cursor = actuator_target.stop
        previous_action = slice(cursor, cursor + width(self.previous_action))
        second_previous_action = slice(previous_action.stop, previous_action.stop + width(self.second_previous_action))
        cursor = second_previous_action.stop + width(self.velocity_command)
        common_phase_sin = cursor
        common_phase_cos = cursor + 1
        cursor += 2
        foot_count = width(self.foot_phase_offsets)
        foot_phase_sin = slice(cursor, cursor + foot_count)
        cursor = foot_phase_sin.stop
        foot_phase_cos = slice(cursor, cursor + foot_count)
        return TRDynamicsFeatureLayout(
            rotation=rotation,
            actuator_target=actuator_target,
            previous_action=previous_action,
            second_previous_action=second_previous_action,
            common_phase_sin=common_phase_sin,
            common_phase_cos=common_phase_cos,
            foot_phase_sin=foot_phase_sin,
            foot_phase_cos=foot_phase_cos,
            dimension=self.dynamics_features().shape[-1],
        )

    def dynamics_features(self) -> torch.Tensor:
        """Return continuous features for normalized one-step model errors."""

        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(self.batch_size, -1)

        phase_angle = 2.0 * torch.pi * self.common_gait_phase
        offset_angle = 2.0 * torch.pi * self.foot_phase_offsets
        row_count = len(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS)
        valid_row = (self.gait_row >= 0) & (self.gait_row < row_count)
        safe_row = self.gait_row.clamp(0, row_count - 1)
        row_features = nn.functional.one_hot(safe_row, row_count).to(dtype=self.root_position.dtype)
        row_features = row_features * valid_row.unsqueeze(-1)
        return torch.cat(
            (
                self.root_position,
                self.root_rotation_6d,
                self.root_linear_velocity,
                self.root_angular_velocity,
                self.joint_position,
                self.joint_velocity,
                self.actuator_target,
                self.previous_action,
                self.second_previous_action,
                self.velocity_command,
                torch.sin(phase_angle).unsqueeze(-1),
                torch.cos(phase_angle).unsqueeze(-1),
                torch.sin(offset_angle),
                torch.cos(offset_angle),
                self.swing_ratio.unsqueeze(-1),
                self.stance_ratio.unsqueeze(-1),
                self.gait_period.unsqueeze(-1),
                row_features,
                flatten(self.body_mass),
                flatten(self.material_properties),
                flatten(self.additional_actuator_state),
            ),
            dim=-1,
        )


def time_reverse_dynamics_state(state: TRDynamicsState) -> TRDynamicsState:
    """Apply the documented involutive temporal transform ``T_X``."""
    partners = torch.as_tensor(
        SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS,
        dtype=torch.long,
        device=state.gait_row.device,
    )
    valid_row = (state.gait_row >= 0) & (state.gait_row < len(partners))
    safe_row = state.gait_row.clamp(0, len(partners) - 1)
    transformed_row = torch.where(valid_row, partners[safe_row], state.gait_row)
    return TRDynamicsState(
        root_position=state.root_position.clone(),
        root_rotation_6d=state.root_rotation_6d.clone(),
        root_linear_velocity=-state.root_linear_velocity,
        root_angular_velocity=-state.root_angular_velocity,
        joint_position=state.joint_position.clone(),
        joint_velocity=-state.joint_velocity,
        actuator_target=state.actuator_target.clone(),
        previous_action=state.previous_action.clone(),
        second_previous_action=state.second_previous_action.clone(),
        velocity_command=-state.velocity_command,
        common_gait_phase=torch.remainder(state.swing_ratio - state.common_gait_phase, 1.0),
        foot_phase_offsets=-state.foot_phase_offsets,
        swing_ratio=state.swing_ratio.clone(),
        stance_ratio=state.stance_ratio.clone(),
        gait_period=state.gait_period.clone(),
        gait_row=transformed_row,
        body_mass=state.body_mass.clone(),
        material_properties=state.material_properties.clone(),
        additional_actuator_state=state.additional_actuator_state.clone(),
    )


def time_reverse_actions(actions: torch.Tensor) -> torch.Tensor:
    """Apply ``T_A`` for even joint-position target offsets."""
    return actions.clone()


def assert_time_reverse_action_transform(action_dim: int, device: torch.device) -> None:
    """Assert the analytic action transform is involutive and volume preserving."""
    probe = torch.linspace(-1.0, 1.0, action_dim, device=device).unsqueeze(0)
    if not torch.equal(time_reverse_actions(time_reverse_actions(probe)), probe):
        raise RuntimeError("The analytic time-reversal action transform is not involutive.")
    # T_A is the identity for position targets, hence J_TA = I and |det J_TA| = 1.
    jacobian_determinant = torch.linalg.det(torch.eye(action_dim, device=device)).abs()
    if not torch.equal(jacobian_determinant, torch.ones_like(jacobian_determinant)):
        raise RuntimeError("The analytic time-reversal action transform is not volume preserving.")


def build_time_reversal_observation(
    state: TRDynamicsState,
    *,
    previous_action: torch.Tensor,
    second_previous_action: torch.Tensor,
    base_height_range: tuple[float, float],
) -> torch.Tensor:
    """Build the 64D policy observation from a reversed state and action history.

    Unlike applying ``T_O`` to an isolated stored observation, this builder
    receives the action that actually precedes the state in the constructed
    reversed sequence.
    """
    if (
        state.joint_position.shape[-1] != 12
        or previous_action.shape[-1] != 12
        or second_previous_action.shape[-1] != 12
    ):
        raise ValueError("The shared symmetric quadruped observation requires 12 joints and 12 actions.")
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    scales = SYMM_QUADRUPED_POLICY_OBS_SCALE
    observation = torch.zeros(
        state.batch_size,
        SYMM_QUADRUPED_POLICY_OBS_DIM,
        dtype=state.root_position.dtype,
        device=state.root_position.device,
    )
    rotation = rotation_6d_to_matrix(state.root_rotation_6d)
    gravity_world = torch.tensor((0.0, 0.0, -1.0), dtype=observation.dtype, device=observation.device)
    projected_gravity = torch.matmul(rotation.transpose(-1, -2), gravity_world)
    gravity_scale = torch.as_tensor(scales.projected_gravity, dtype=observation.dtype, device=observation.device)
    observation[:, layout.projected_gravity] = projected_gravity * gravity_scale

    command_scale = torch.as_tensor(scales.velocity_command, dtype=observation.dtype, device=observation.device)
    observation[:, layout.velocity_command] = state.velocity_command * command_scale
    joint_position_scale = torch.as_tensor(scales.joint_position, dtype=observation.dtype, device=observation.device)
    joint_velocity_scale = torch.as_tensor(scales.joint_velocity, dtype=observation.dtype, device=observation.device)
    previous_action_scale = torch.as_tensor(scales.previous_action, dtype=observation.dtype, device=observation.device)
    second_previous_action_scale = torch.as_tensor(
        scales.second_previous_action, dtype=observation.dtype, device=observation.device
    )
    observation[:, layout.joint_position] = state.joint_position * joint_position_scale
    observation[:, layout.joint_velocity] = state.joint_velocity * joint_velocity_scale
    observation[:, layout.previous_action] = previous_action * previous_action_scale
    observation[:, layout.second_previous_action] = second_previous_action * second_previous_action_scale
    dimensionless_period = compute_dimensionless_gait_period(state.gait_period, base_height_range)
    observation[:, layout.gait_period] = dimensionless_period.unsqueeze(-1) * scales.gait_period[0]
    observation[:, layout.duty_factor] = state.stance_ratio.unsqueeze(-1) * scales.duty_factor[0]

    foot_phase = torch.remainder(state.common_gait_phase.unsqueeze(-1) + state.foot_phase_offsets, 1.0)
    phase_sin_scale = torch.as_tensor(scales.foot_phase_sin, dtype=observation.dtype, device=observation.device)
    phase_cos_scale = torch.as_tensor(scales.foot_phase_cos, dtype=observation.dtype, device=observation.device)
    observation[:, layout.foot_phase_sin] = torch.sin(2.0 * torch.pi * foot_phase) * phase_sin_scale
    observation[:, layout.foot_phase_cos] = torch.cos(2.0 * torch.pi * foot_phase) * phase_cos_scale
    return observation


@dataclass(frozen=True)
class TRReversedSequence:
    """Complete reversed sequence after dropping two unknown-history transitions."""

    observations: torch.Tensor
    actions: torch.Tensor
    current_state: TRDynamicsState
    successor_state: TRDynamicsState


def build_reversed_sequence_segment(
    states: list[TRDynamicsState],
    actions: list[torch.Tensor],
    *,
    reversed_actions: list[torch.Tensor] | None = None,
    reversed_actuator_targets: list[torch.Tensor] | None = None,
    base_height_range: tuple[float, float],
) -> TRReversedSequence:
    """Reverse a constant-task state/action sequence with correct action history.

    Args:
        states: Chronological states ``x_0, ..., x_H``, each with the same batch size.
        actions: Chronological actions ``a_0, ..., a_{H-1}``.
        reversed_actions: Final reversed action for each chronological transition.
            ``None`` selects the analytic transform.
        reversed_actuator_targets: Processed controller target corresponding
            to each final reversed action. ``None`` reuses authentic analytic
            targets from the chronological successor states.

    Returns:
        Reversed supervision pairs. The first two transitions are dropped
        because their complete two-action reversed history is not reconstructable.
    """
    if len(states) != len(actions) + 1:
        raise ValueError("A complete segment must contain exactly one more state than action.")
    if len(actions) < 3:
        raise ValueError("At least three transitions are required after dropping unknown reversed action history.")
    if reversed_actions is None:
        reversed_actions = [time_reverse_actions(action) for action in actions]
    if reversed_actuator_targets is None:
        reversed_actuator_targets = [states[index + 1].actuator_target.clone() for index in range(len(actions))]
    if len(reversed_actions) != len(actions) or len(reversed_actuator_targets) != len(actions):
        raise ValueError("Final reversed actions and actuator targets must align one-to-one with actions.")
    observations: list[torch.Tensor] = []
    supervision_actions: list[torch.Tensor] = []
    current_states: list[TRDynamicsState] = []
    successor_states: list[TRDynamicsState] = []
    for index in range(len(actions) - 3, -1, -1):
        previous_action = reversed_actions[index + 1]
        second_previous_action = reversed_actions[index + 2]
        reversed_action = reversed_actions[index]
        # The stored state fields describe chronological action history. A
        # valid reversed Markov state instead carries the preceding reversed
        # action and its processed controller target from the adjacent state.
        current = replace(
            time_reverse_dynamics_state(states[index + 1]),
            previous_action=previous_action,
            second_previous_action=second_previous_action,
            actuator_target=reversed_actuator_targets[index + 1],
        )
        successor = replace(
            time_reverse_dynamics_state(states[index]),
            previous_action=reversed_action,
            second_previous_action=previous_action,
            actuator_target=reversed_actuator_targets[index],
        )
        observations.append(
            build_time_reversal_observation(
                current,
                previous_action=previous_action,
                second_previous_action=second_previous_action,
                base_height_range=base_height_range,
            )
        )
        supervision_actions.append(reversed_action)
        current_states.append(current)
        successor_states.append(successor)
    return TRReversedSequence(
        observations=torch.cat(observations),
        actions=torch.cat(supervision_actions),
        current_state=TRDynamicsState.concatenate(current_states),
        successor_state=TRDynamicsState.concatenate(successor_states),
    )


def time_reversal_augmentation_nll(
    log_probability: torch.Tensor,
    validity: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Return masked confidence-weighted reverse-action negative log likelihood."""
    safe_confidence = torch.nan_to_num(confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    weight = torch.where(validity, safe_confidence, torch.zeros_like(safe_confidence))
    selected_log_probability = torch.where(weight > 0.0, log_probability, torch.zeros_like(log_probability))
    denominator = weight.sum()
    if not bool(denominator > 0.0):
        return torch.zeros((), dtype=log_probability.dtype, device=log_probability.device)
    return -(weight * selected_log_probability).sum() / (denominator + _EPS)


def _gait_command_term(environment):
    command_manager = environment.command_manager
    for name in command_manager.active_terms:
        term = command_manager.get_term(name)
        required = ("foot_thetas", "duty_factors", "gait_periods", "gait_row_indices", "common_gait_phases")
        if all(hasattr(term, attribute) for attribute in required):
            return term
    raise RuntimeError("No GaitVelocityCommand term is available for TRDynamicsState capture.")


def _ordered_joint_state(environment) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_manager = environment.action_manager
    if len(action_manager.active_terms) != 1:
        raise RuntimeError("Time-reversal augmentation currently requires exactly one joint-position action term.")
    action_term = action_manager.get_term(action_manager.active_terms[0])
    if not hasattr(action_term, "processed_actions") or not hasattr(action_term, "_joint_ids"):
        raise RuntimeError("Time-reversal augmentation requires a joint-position action term with ordered joints.")
    robot = environment.scene["robot"]
    joint_ids = action_term._joint_ids
    joint_position = _tensor(robot.data.joint_pos)[:, joint_ids]
    default_joint_position = _tensor(robot.data.default_joint_pos)[:, joint_ids]
    joint_velocity = _tensor(robot.data.joint_vel)[:, joint_ids]
    actuator_target = _tensor(robot.data.joint_pos_target)[:, joint_ids]
    return joint_position - default_joint_position, joint_velocity, actuator_target


def _capture_even_dynamics_parameters(
    robot,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture randomized mass and material variables required for a Markov state."""
    try:
        body_mass = _tensor(robot.data.body_mass)
    except (AttributeError, RuntimeError, TypeError) as error:
        raise RuntimeError(
            "Time-reversal augmentation cannot capture per-environment body masses. These even dynamics "
            "variables are required because nondegenerate mass randomization may be active; augmentation "
            "therefore fails closed."
        ) from error
    root_view = getattr(robot, "root_view", None)
    material_getter = getattr(root_view, "get_material_properties", None)
    if not callable(material_getter):
        raise RuntimeError(
            "Time-reversal augmentation cannot capture per-environment rigid-shape material properties. "
            "These even dynamics variables are required because nondegenerate friction/restitution "
            "randomization may be active; augmentation therefore fails closed on this physics backend."
        )
    try:
        material_properties = _tensor(material_getter())
    except (AttributeError, RuntimeError, TypeError) as error:
        raise RuntimeError(
            "Time-reversal augmentation failed to capture per-environment rigid-shape material properties; "
            "refusing to omit latent randomized dynamics variables from TRDynamicsState."
        ) from error
    if body_mass.ndim != 2 or body_mass.shape[0] != num_envs:
        raise RuntimeError(
            f"Expected body masses with shape ({num_envs}, num_bodies); received {tuple(body_mass.shape)}."
        )
    if material_properties.ndim != 3 or material_properties.shape[0] != num_envs:
        raise RuntimeError(
            "Expected material properties with shape "
            f"({num_envs}, num_shapes, 3); received {tuple(material_properties.shape)}."
        )
    if material_properties.shape[-1] != 3:
        raise RuntimeError("Material properties must contain static friction, dynamic friction, and restitution.")
    if not bool(torch.isfinite(body_mass).all() and (body_mass > 0.0).all()):
        raise RuntimeError("Captured body masses must be finite and strictly positive [kg].")
    if not bool(torch.isfinite(material_properties).all()):
        raise RuntimeError("Captured material properties must be finite.")
    return (
        body_mass.to(device=device, dtype=dtype).detach().clone(),
        material_properties.to(device=device, dtype=dtype).detach().clone(),
    )


def capture_tr_dynamics_state(env_wrapper) -> TRDynamicsState:
    """Capture the supported flat-terrain Markov state from the wrapped live environment."""
    environment = env_wrapper.unwrapped if hasattr(env_wrapper, "unwrapped") else env_wrapper
    required = ("scene", "action_manager", "command_manager")
    if not all(hasattr(environment, attribute) for attribute in required):
        raise RuntimeError("TRDynamicsState capture requires a manager-based Isaac Lab environment.")
    robot = environment.scene["robot"]
    root_position = _tensor(robot.data.root_pos_w) - environment.scene.env_origins
    root_rotation = quaternion_to_rotation_6d(_tensor(robot.data.root_quat_w))
    root_linear_velocity = _tensor(robot.data.root_lin_vel_b)
    root_angular_velocity = _tensor(robot.data.root_ang_vel_b)
    joint_position, joint_velocity, actuator_target = _ordered_joint_state(environment)
    gait_command = _gait_command_term(environment)
    previous_action = environment.action_manager.action
    second_previous_action = environment.action_manager.prev_action
    if actuator_target.shape != previous_action.shape or second_previous_action.shape != previous_action.shape:
        raise RuntimeError(
            "Processed actuator targets and both policy action-history tensors must have identical shapes."
        )
    num_envs = root_position.shape[0]
    body_mass, material_properties = _capture_even_dynamics_parameters(
        robot,
        num_envs=num_envs,
        device=root_position.device,
        dtype=root_position.dtype,
    )
    return TRDynamicsState(
        root_position=root_position.detach().clone(),
        root_rotation_6d=root_rotation.detach().clone(),
        root_linear_velocity=root_linear_velocity.detach().clone(),
        root_angular_velocity=root_angular_velocity.detach().clone(),
        joint_position=joint_position.detach().clone(),
        joint_velocity=joint_velocity.detach().clone(),
        actuator_target=actuator_target.detach().clone(),
        previous_action=previous_action.detach().clone(),
        second_previous_action=second_previous_action.detach().clone(),
        velocity_command=gait_command.command.detach().clone(),
        common_gait_phase=torch.remainder(gait_command.common_gait_phases(), 1.0).detach().clone(),
        foot_phase_offsets=gait_command.foot_thetas.detach().clone(),
        swing_ratio=(1.0 - gait_command.duty_factors).detach().clone(),
        stance_ratio=gait_command.duty_factors.detach().clone(),
        gait_period=gait_command.gait_periods.detach().clone(),
        gait_row=gait_command.gait_row_indices.detach().clone(),
        body_mass=body_mass,
        material_properties=material_properties,
        additional_actuator_state=torch.empty(num_envs, 0, device=root_position.device, dtype=root_position.dtype),
    )


def _capture_contact_diagnostics(env_wrapper) -> tuple[torch.Tensor, torch.Tensor]:
    environment = env_wrapper.unwrapped if hasattr(env_wrapper, "unwrapped") else env_wrapper
    num_envs = environment.num_envs
    device = environment.device
    maximum_impulse = torch.zeros(num_envs, device=device)
    contact_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
    sensor_names = list(getattr(environment.scene, "sensors", {}).keys())
    contact_index = 0
    for sensor_name in sensor_names:
        sensor = environment.scene.sensors[sensor_name]
        history = getattr(sensor.data, "net_forces_w_history", None)
        if history is None:
            continue
        force = _tensor(history)
        if force.ndim < 4:
            continue
        force_norm_history = torch.linalg.vector_norm(force, dim=-1)
        # Use the history mean as the available control-interval force
        # estimate. Multiplying by step_dt integrates over the RL/control step
        # instead of treating one physics-step peak as an impulse.
        impulse_per_body = force_norm_history.mean(dim=1) * float(environment.step_dt)
        force_norm = force_norm_history.amax(dim=1).amax(dim=-1)
        maximum_impulse = torch.maximum(maximum_impulse, impulse_per_body.amax(dim=-1))
        if contact_index < 62:
            contact_mode |= (force_norm > 1.0).to(torch.long) << contact_index
        contact_index += 1
    return maximum_impulse, contact_mode


@dataclass
class TRSidecarTransition:
    """One batched rollout step retained outside upstream RSL-RL storage."""

    observation: torch.Tensor
    successor_observation: torch.Tensor
    state: TRDynamicsState
    successor_state: TRDynamicsState
    action: torch.Tensor
    actor_mean: torch.Tensor
    actor_std: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    timeout: torch.Tensor
    terminal_state_available: torch.Tensor
    command: torch.Tensor
    gait_row: torch.Tensor
    common_gait_phase: torch.Tensor
    duty_factor: torch.Tensor
    gait_period: torch.Tensor
    previous_action: torch.Tensor
    second_previous_action: torch.Tensor
    contact_impulse: torch.Tensor
    contact_mode: torch.Tensor
    environment_id: torch.Tensor
    rollout_step: torch.Tensor
    episode_id: torch.Tensor
    valid_successor: torch.Tensor
    task_continuity: torch.Tensor


@dataclass
class _PendingTransition:
    observation: torch.Tensor
    state: TRDynamicsState
    action: torch.Tensor
    actor_mean: torch.Tensor
    actor_std: torch.Tensor
    episode_id: torch.Tensor
    rollout_step: int


class TRSidecarBuffer:
    """Project-local successor-state rollout storage allocated only when enabled."""

    def __init__(self, env_wrapper) -> None:
        self.env_wrapper = env_wrapper
        environment = env_wrapper.unwrapped if hasattr(env_wrapper, "unwrapped") else env_wrapper
        self.num_envs = int(environment.num_envs)
        self.step_dt = float(environment.step_dt)
        self.device = torch.device(environment.device)
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rollout_step = 0
        self.pending: _PendingTransition | None = None
        self.transitions: list[TRSidecarTransition] = []

    def capture_before_step(
        self,
        observation: TensorDict,
        state: TRDynamicsState,
        action: torch.Tensor,
        actor_mean: torch.Tensor,
        actor_std: torch.Tensor,
    ) -> None:
        """Capture ``x_t`` immediately before the environment step."""
        if self.pending is not None:
            raise RuntimeError("The time-reversal sidecar already has an unmatched pre-step state.")
        if observation["policy"].shape[-1] != SYMM_QUADRUPED_POLICY_OBS_DIM:
            raise RuntimeError(
                "Time-reversal sidecar augmentation supports only one instantaneous policy frame; received "
                f"width {observation['policy'].shape[-1]}. Historical reversed context would require future "
                "samples from the authentic rollout and is not fabricated. Disable history or the sidecar."
            )
        self.pending = _PendingTransition(
            observation=observation["policy"].detach().clone(),
            state=state.clone(),
            action=action.detach().clone(),
            actor_mean=actor_mean.detach().clone(),
            actor_std=actor_std.detach().clone(),
            episode_id=self.episode_ids.clone(),
            rollout_step=self.rollout_step,
        )

    def capture_after_step(
        self,
        observation: TensorDict,
        successor_state: TRDynamicsState,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping,
    ) -> None:
        """Pair authentic nonterminal successors and reject auto-reset successors."""
        if self.pending is None:
            raise RuntimeError("The time-reversal sidecar received a successor without a pre-step state.")
        if observation["policy"].shape[-1] != SYMM_QUADRUPED_POLICY_OBS_DIM:
            raise RuntimeError(
                "Time-reversal sidecar augmentation received a historical successor observation. Disable history "
                "or the sidecar; current-frame-only reconstruction is intentionally unsupported."
            )
        pending = self.pending
        done = dones.to(dtype=torch.bool)
        timeout = extras.get("time_outs", torch.zeros_like(done)).to(device=done.device, dtype=torch.bool)
        episode_boundary = done | timeout
        # Isaac Lab computes returned observations after auto-reset. No authentic terminal
        # dynamics state is exposed by the wrapper, so every done/timeout pair is rejected.
        terminal_state_available = torch.zeros_like(done)
        valid_successor = ~episode_boundary
        command_close = torch.isclose(
            pending.state.velocity_command, successor_state.velocity_command, rtol=0.0, atol=1.0e-6
        ).all(dim=-1)
        gait_same = pending.state.gait_row == successor_state.gait_row
        period_same = torch.isclose(pending.state.gait_period, successor_state.gait_period, rtol=0.0, atol=1.0e-6)
        duty_same = torch.isclose(pending.state.stance_ratio, successor_state.stance_ratio, rtol=0.0, atol=1.0e-6)
        offset_error = torch.abs(
            torch.remainder(successor_state.foot_phase_offsets - pending.state.foot_phase_offsets + 0.5, 1.0) - 0.5
        )
        offsets_same = (offset_error <= 1.0e-6).all(dim=-1)
        expected_phase_delta = self.step_dt / pending.state.gait_period.clamp_min(_EPS)
        observed_phase_delta = torch.remainder(successor_state.common_gait_phase - pending.state.common_gait_phase, 1.0)
        phase_error = torch.abs(torch.remainder(observed_phase_delta - expected_phase_delta + 0.5, 1.0) - 0.5)
        phase_continuous = phase_error <= 1.0e-3
        task_continuity = command_close & gait_same & period_same & duty_same & offsets_same & phase_continuous
        contact_impulse, contact_mode = _capture_contact_diagnostics(self.env_wrapper)
        environment_id = torch.arange(self.num_envs, device=done.device)
        rollout_step = torch.full_like(environment_id, pending.rollout_step)
        self.transitions.append(
            TRSidecarTransition(
                observation=pending.observation,
                successor_observation=observation["policy"].detach().clone(),
                state=pending.state,
                successor_state=successor_state.clone(),
                action=pending.action,
                actor_mean=pending.actor_mean,
                actor_std=pending.actor_std,
                reward=rewards.detach().clone(),
                done=done.detach().clone(),
                timeout=timeout.detach().clone(),
                terminal_state_available=terminal_state_available,
                command=pending.state.velocity_command.clone(),
                gait_row=pending.state.gait_row.clone(),
                common_gait_phase=pending.state.common_gait_phase.clone(),
                duty_factor=pending.state.stance_ratio.clone(),
                gait_period=pending.state.gait_period.clone(),
                previous_action=pending.state.previous_action.clone(),
                second_previous_action=pending.state.second_previous_action.clone(),
                contact_impulse=contact_impulse.detach().clone(),
                contact_mode=contact_mode.detach().clone(),
                environment_id=environment_id,
                rollout_step=rollout_step,
                episode_id=pending.episode_id,
                valid_successor=valid_successor,
                task_continuity=task_continuity,
            )
        )
        self.episode_ids += episode_boundary.to(dtype=torch.long)
        self.rollout_step += 1
        self.pending = None

    def clear(self) -> None:
        """Clear one consumed rollout while retaining episode identity."""
        self.transitions.clear()
        self.pending = None
        self.rollout_step = 0

    def state_dict(self) -> dict:
        """Return exact reconstruction state for checkpointing."""
        return {
            "episode_ids": self.episode_ids,
            "rollout_step": self.rollout_step,
            "pending": self.pending,
            "transitions": self.transitions,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore exact reconstruction state from a checkpoint."""
        self.episode_ids = state["episode_ids"].to(self.device)
        self.rollout_step = int(state["rollout_step"])
        self.pending = state["pending"]
        self.transitions = list(state["transitions"])


class _RunningNormalizer(nn.Module):
    """Deterministic running component normalization."""

    def __init__(self, dimension: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension, device=device))
        self.register_buffer("variance", torch.ones(dimension, device=device))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Merge a batch of moments."""
        if values.numel() == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        batch_count = float(values.shape[0])
        if self.count.item() == 0.0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance.clamp_min(1.0e-6))
            self.count.fill_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        first_moment = self.variance * self.count
        second_moment = batch_variance * batch_count
        combined = first_moment + second_moment + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.variance.copy_((combined / total).clamp_min(1.0e-6))
        self.count.copy_(total)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize component-wise values."""
        return (values - self.mean) / torch.sqrt(self.variance + 1.0e-6)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Undo component-wise normalization."""
        return values * torch.sqrt(self.variance + 1.0e-6) + self.mean


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class TRAugmentationPool:
    """Detached reversed-observation supervision pool for one PPO update."""

    observations: torch.Tensor
    actions: torch.Tensor
    confidence: torch.Tensor
    residual: torch.Tensor
    gait_row: torch.Tensor
    command: torch.Tensor
    contact_mode: torch.Tensor

    @property
    def count(self) -> int:
        """Number of accepted candidates."""
        return self.actions.shape[0]


@dataclass(frozen=True)
class TRReplayCandidatePool:
    """All reversed candidates retained for caller-driven one-step replay.

    This diagnostic pool is never read by PPO. Each row contains a complete
    reversed Markov state, proposed action, target successor, learned-filter
    decision and gate outcomes, plus source rollout identity. A caller can
    therefore restore :attr:`current_state` in a separate simulator, apply
    :attr:`actions` for one control step, and compare the result with
    :attr:`successor_state` without reconstructing hidden rollout context.
    """

    observations: torch.Tensor
    successor_observations: torch.Tensor
    actions: torch.Tensor
    current_state: TRDynamicsState
    successor_state: TRDynamicsState
    learned_residual: torch.Tensor
    learned_threshold: torch.Tensor
    filter_accepted: torch.Tensor
    finite_gate: torch.Tensor
    action_gate: torch.Tensor
    phase_gate: torch.Tensor
    contact_impulse_gate: torch.Tensor
    dynamics_gate: torch.Tensor
    model_quality_gate: torch.Tensor
    contact_impulse: torch.Tensor
    contact_mode: torch.Tensor
    gait_row: torch.Tensor
    command: torch.Tensor
    environment_id: torch.Tensor
    episode_id: torch.Tensor
    rollout_step: torch.Tensor

    @property
    def count(self) -> int:
        """Total number of accepted and rejected candidates."""
        return self.actions.shape[0]

    @property
    def accepted_count(self) -> int:
        """Number of candidates accepted by the complete training filter."""
        return int(self.filter_accepted.sum().item())

    @property
    def rejected_count(self) -> int:
        """Number of candidates rejected by at least one training gate."""
        return self.count - self.accepted_count

    def index(self, index) -> TRReplayCandidatePool:
        """Return a batch-preserving subset of candidates."""
        if isinstance(index, int):
            index = slice(index, index + 1)
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.index(index) if isinstance(value, TRDynamicsState) else value[index]
        return TRReplayCandidatePool(**values)

    def sample_by_filter_decision(self, maximum_per_decision: int | None = None) -> TRReplayCandidatePool:
        """Return a deterministic sample retaining both decision classes.

        Args:
            maximum_per_decision: Maximum accepted and rejected rows retained
                independently. ``None`` retains the complete pool.

        Returns:
            Candidate rows with up to the requested count from each available
            filter-decision class.
        """
        if maximum_per_decision is None:
            return self
        if isinstance(maximum_per_decision, bool) or not isinstance(maximum_per_decision, int):
            raise TypeError("maximum_per_decision must be an integer or None.")
        if maximum_per_decision <= 0:
            raise ValueError("maximum_per_decision must be positive.")
        accepted = torch.nonzero(self.filter_accepted, as_tuple=False).flatten()[:maximum_per_decision]
        rejected = torch.nonzero(~self.filter_accepted, as_tuple=False).flatten()[:maximum_per_decision]
        indices = torch.cat((accepted, rejected)).sort().values
        return self.index(indices)


@dataclass(frozen=True)
class TRFilterMask:
    """Mandatory learned-filter decision and its individual validity gates."""

    accepted: torch.Tensor
    finite: torch.Tensor
    action: torch.Tensor
    phase: torch.Tensor
    phase_current: torch.Tensor
    phase_successor: torch.Tensor
    phase_event: torch.Tensor
    contact_impulse: torch.Tensor
    dynamics: torch.Tensor
    model_quality: torch.Tensor


def time_reversal_filter_mask(
    *,
    observation: torch.Tensor,
    action: torch.Tensor,
    reverse_residual: torch.Tensor,
    beta: torch.Tensor | float,
    foot_phase: torch.Tensor,
    successor_foot_phase: torch.Tensor,
    phase_event_crossing: torch.Tensor,
    swing_ratio: torch.Tensor,
    contact_impulse: torch.Tensor,
    model_quality: bool,
    phase_boundary_margin: float,
    maximum_contact_impulse: float,
    action_abs_limit: float,
) -> TRFilterMask:
    """Apply every mandatory dynamics-filter and physical-validity gate."""
    finite = (
        torch.isfinite(observation).all(dim=-1) & torch.isfinite(action).all(dim=-1) & torch.isfinite(reverse_residual)
    )
    action_valid = action.abs().amax(dim=-1) <= action_abs_limit

    def endpoint_valid(phase: torch.Tensor) -> torch.Tensor:
        distance_zero = torch.abs(torch.remainder(phase + 0.5, 1.0) - 0.5)
        distance_swing = torch.abs(torch.remainder(phase - swing_ratio.unsqueeze(-1) + 0.5, 1.0) - 0.5)
        return ((distance_zero >= phase_boundary_margin) & (distance_swing >= phase_boundary_margin)).all(dim=-1)

    current_phase_valid = endpoint_valid(foot_phase)
    successor_phase_valid = endpoint_valid(successor_foot_phase)
    event_phase_valid = ~phase_event_crossing
    phase_valid = current_phase_valid & successor_phase_valid & event_phase_valid
    impulse_valid = contact_impulse <= maximum_contact_impulse
    dynamics_valid = reverse_residual <= torch.as_tensor(
        beta, dtype=reverse_residual.dtype, device=reverse_residual.device
    )
    model_valid = torch.full_like(dynamics_valid, model_quality)
    accepted = finite & action_valid & phase_valid & impulse_valid & dynamics_valid & model_valid
    return TRFilterMask(
        accepted=accepted,
        finite=finite,
        action=action_valid,
        phase=phase_valid,
        phase_current=current_phase_valid,
        phase_successor=successor_phase_valid,
        phase_event=event_phase_valid,
        contact_impulse=impulse_valid,
        dynamics=dynamics_valid,
        model_quality=model_valid,
    )


def time_reversal_phase_event_crossing(
    foot_phase: torch.Tensor,
    successor_foot_phase: torch.Tensor,
    swing_ratio: torch.Tensor,
) -> torch.Tensor:
    """Return whether any foot crosses touchdown or liftoff over the forward phase interval."""
    phase_delta = torch.remainder(successor_foot_phase - foot_phase, 1.0)
    distance_touchdown = torch.remainder(-foot_phase, 1.0)
    distance_liftoff = torch.remainder(swing_ratio.unsqueeze(-1) - foot_phase, 1.0)
    positive = torch.finfo(foot_phase.dtype).eps
    touchdown_crossing = (distance_touchdown > positive) & (distance_touchdown <= phase_delta + positive)
    liftoff_crossing = (distance_liftoff > positive) & (distance_liftoff <= phase_delta + positive)
    return (touchdown_crossing | liftoff_crossing).any(dim=-1)


class TimeReversalAugmentation:
    """Dynamics-filtered reverse-action supervision controller."""

    _CHECKPOINT_SCHEMA_VERSION = 3
    _SEMANTIC_DEFAULTS = {
        "enabled": True,
        "mode": "dynamics_filtered_reverse_action_supervision",
        "action_source": "analytic",
        "dynamics_hidden_dims": (256, 256),
        "inverse_hidden_dims": (256, 256),
        "learning_rate": 1.0e-3,
        "ema_decay": 0.995,
        "validation_fraction": 0.2,
        "validation_quantile": 0.95,
        "threshold_multiplier": 2.0,
        "minimum_validation_samples": 128,
        "maximum_validation_loss": 1.0,
        "filter_enabled": True,
        "phase_boundary_margin": 0.03,
        "maximum_contact_impulse": 20.0,
        "action_abs_limit": 10.0,
        "max_augmented_to_original_ratio": 0.25,
        "use_confidence_weights": True,
        "model_updates_per_rollout": 1,
        "model_batch_size": 4096,
        "gradient_diagnostics_interval": 100,
        "gradient_diagnostics_epsilon": 1.0e-12,
        "rng_seed": 0,
    }

    def __init__(self, env_wrapper, cfg: Mapping, device: str | torch.device) -> None:
        self.env_wrapper = env_wrapper
        self.cfg = dict(cfg)
        self.device = torch.device(device)
        template_state = capture_tr_dynamics_state(env_wrapper)
        if template_state.root_position.device != self.device:
            raise RuntimeError(
                "Time-reversal augmentation requires the policy and environment on the same device; "
                f"received {self.device} and {template_state.root_position.device}."
            )
        if getattr(env_wrapper, "clip_actions", None) is not None:
            raise RuntimeError(
                "Time-reversal augmentation requires unclipped wrapper actions so sidecar actions equal applied "
                "policy actions. Set the wrapper clip_actions field to None."
            )
        if template_state.additional_actuator_state.shape[-1] != 0:
            raise RuntimeError("Nonempty actuator hidden state requires an explicit temporal parity transform.")
        environment = env_wrapper.unwrapped if hasattr(env_wrapper, "unwrapped") else env_wrapper
        command_cfg = _gait_command_term(environment).cfg
        self.base_height_range = tuple(command_cfg.base_height_range)
        # Validate the shared characteristic-length convention immediately.
        compute_dimensionless_gait_period(template_state.gait_period, self.base_height_range)
        action_dim = int(environment.action_manager.total_action_dim)
        self.action_term = environment.action_manager.get_term(environment.action_manager.active_terms[0])
        if self.cfg.get("action_source", "analytic") == "learned_inverse" and not all(
            hasattr(self.action_term, name) for name in ("_scale", "_offset", "cfg")
        ):
            raise RuntimeError(
                "Learned inverse actions require an affine joint-position action term exposing scale, offset, "
                "and clip metadata so controller targets can be reconstructed without mutating the environment."
            )
        assert_time_reverse_action_transform(action_dim, self.device)
        state_dim = template_state.dynamics_features().shape[-1]
        self.feature_layout = template_state.dynamics_feature_layout()
        excluded_inverse_width = (
            self.feature_layout.actuator_target.stop
            - self.feature_layout.actuator_target.start
            + self.feature_layout.previous_action.stop
            - self.feature_layout.previous_action.start
            + self.feature_layout.second_previous_action.stop
            - self.feature_layout.second_previous_action.start
        )
        inverse_state_dim = state_dim - excluded_inverse_width
        dynamics_hidden = tuple(self.cfg.get("dynamics_hidden_dims", (256, 256)))
        inverse_hidden = tuple(self.cfg.get("inverse_hidden_dims", (256, 256)))
        rng_seed = int(self.cfg.get("rng_seed", 0))
        self.initialization_generator = torch.Generator(device="cpu")
        self.initialization_generator.manual_seed(rng_seed)
        fork_devices = []
        if self.device.type == "cuda":
            fork_devices.append(self.device.index if self.device.index is not None else torch.cuda.current_device())
        with torch.random.fork_rng(devices=fork_devices):
            torch.set_rng_state(self.initialization_generator.get_state())
            self.forward_model = _mlp(state_dim + action_dim, state_dim, dynamics_hidden).to(self.device)
            self.inverse_model = _mlp(2 * inverse_state_dim, action_dim, inverse_hidden).to(self.device)
            self.initialization_generator.set_state(torch.get_rng_state())
        self.forward_target = copy.deepcopy(self.forward_model).requires_grad_(False)
        self.inverse_target = copy.deepcopy(self.inverse_model).requires_grad_(False)
        learning_rate = float(self.cfg.get("learning_rate", 1.0e-3))
        self.forward_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.inverse_optimizer = torch.optim.Adam(self.inverse_model.parameters(), lr=learning_rate)
        self.state_normalizer = _RunningNormalizer(state_dim, self.device)
        self.action_normalizer = _RunningNormalizer(action_dim, self.device)
        self.sidecar = TRSidecarBuffer(env_wrapper)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(rng_seed)
        self.filter_beta = torch.tensor(float("inf"), device=self.device)
        self.validation_ready = False
        self.pool: TRAugmentationPool | None = None
        self.replay_candidates: TRReplayCandidatePool | None = None
        self._last_diagnostics = self._empty_diagnostics()

    def _semantic_config(self) -> dict:
        """Return the exact canonical configuration that determines augmentation state semantics."""
        semantics = {}
        for name, default in self._SEMANTIC_DEFAULTS.items():
            value = self.cfg.get(name, default)
            semantics[name] = tuple(value) if name.endswith("hidden_dims") else value
        semantics["base_height_range"] = self.base_height_range
        return semantics

    def _action_parameter_for_environment(self, value, environment_index: int):
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == self.sidecar.num_envs:
            return value[environment_index : environment_index + 1]
        return value

    def _process_learned_actions(
        self,
        actions: list[torch.Tensor],
        environment_index: int,
    ) -> list[torch.Tensor]:
        """Apply the live joint action term's affine processing without mutating it."""
        scale = self._action_parameter_for_environment(self.action_term._scale, environment_index)
        offset = self._action_parameter_for_environment(self.action_term._offset, environment_index)
        processed = [action * scale + offset for action in actions]
        if getattr(self.action_term.cfg, "clip", None) is not None:
            clip = self._action_parameter_for_environment(self.action_term._clip, environment_index)
            processed = [torch.clamp(action, min=clip[..., 0], max=clip[..., 1]) for action in processed]
        return [action.detach() for action in processed]

    def _inverse_features(self, normalized_state_features: torch.Tensor) -> torch.Tensor:
        """Remove action-history channels that would leak the inverse-model target.

        For the memoryless joint-position controller, the successor's prior
        action and processed actuator target directly encode the supervised
        policy action. Both fields are therefore excluded from both endpoints;
        inverse dynamics receives only physical state, task, phase, and static
        dynamics parameters.
        """
        layout = self.feature_layout
        return torch.cat(
            (
                normalized_state_features[..., : layout.actuator_target.start],
                normalized_state_features[..., layout.second_previous_action.stop :],
            ),
            dim=-1,
        )

    def _learned_reversed_sequence(
        self,
        states: list[TRDynamicsState],
        analytic_actions: list[torch.Tensor],
        environment_index: int,
    ) -> TRReversedSequence:
        """Generate learned actions first, then rebuild every retained history channel from them."""
        reverse_indices = range(len(analytic_actions) - 1, -1, -1)
        provisional_current = TRDynamicsState.concatenate(
            [time_reverse_dynamics_state(states[index + 1]) for index in reverse_indices]
        )
        provisional_successor = TRDynamicsState.concatenate(
            [time_reverse_dynamics_state(states[index]) for index in reverse_indices]
        )
        current_features = self._inverse_features(
            self.state_normalizer.normalize(provisional_current.dynamics_features())
        )
        successor_features = self._inverse_features(
            self.state_normalizer.normalize(provisional_successor.dynamics_features())
        )
        with torch.no_grad():
            normalized_actions = self.inverse_target(torch.cat((current_features, successor_features), dim=-1))
            reverse_order_actions = self.action_normalizer.denormalize(normalized_actions)
        # Predictions are ordered a_{H-1}^R, ..., a_0^R. The sequence builder
        # accepts chronological alignment a_0^R, ..., a_{H-1}^R.
        learned_actions = list(reversed(list(reverse_order_actions.split(1, dim=0))))
        actuator_targets = self._process_learned_actions(learned_actions, environment_index)
        return build_reversed_sequence_segment(
            states,
            analytic_actions,
            reversed_actions=learned_actions,
            reversed_actuator_targets=actuator_targets,
            base_height_range=self.base_height_range,
        )

    def _dynamics_residual(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Evaluate forward predictions with Euclidean and manifold-aware components."""
        return time_reversal_dynamics_residual(
            prediction,
            target,
            feature_mean=self.state_normalizer.mean,
            feature_variance=self.state_normalizer.variance,
            layout=self.feature_layout,
        )

    @staticmethod
    def _empty_diagnostics() -> dict[str, float]:
        return {
            "tr_augmentation/forward_training_loss": 0.0,
            "tr_augmentation/forward_validation_loss": 0.0,
            "tr_augmentation/inverse_training_loss": 0.0,
            "tr_augmentation/inverse_validation_loss": 0.0,
            "tr_augmentation/reverse_residual_mean": 0.0,
            "tr_augmentation/reverse_residual_median": 0.0,
            "tr_augmentation/reverse_residual_p90": 0.0,
            "tr_augmentation/reverse_residual_p95": 0.0,
            "tr_augmentation/reverse_residual_p99": 0.0,
            "tr_augmentation/beta": 0.0,
            "tr_augmentation/candidate_count": 0.0,
            "tr_augmentation/accepted_count": 0.0,
            "tr_augmentation/acceptance_fraction": 0.0,
            "tr_augmentation/rejected/termination_timeout": 0.0,
            "tr_augmentation/rejected/task_discontinuity": 0.0,
            "tr_augmentation/rejected/previous_action_unknown": 0.0,
            "tr_augmentation/rejected/model_quality": 0.0,
            "tr_augmentation/rejected/dynamics_residual": 0.0,
            "tr_augmentation/rejected/phase_boundary": 0.0,
            "tr_augmentation/rejected/successor_phase_boundary": 0.0,
            "tr_augmentation/rejected/phase_event_crossing": 0.0,
            "tr_augmentation/rejected/contact_impulse": 0.0,
            "tr_augmentation/rejected/nonfinite": 0.0,
            "tr_augmentation/rejected/nonfinite_real_transition": 0.0,
            "tr_augmentation/rejected/action_limit": 0.0,
            "tr_augmentation/analytic_action_acceptance": 0.0,
            "tr_augmentation/learned_action_acceptance": 0.0,
            "tr_augmentation/gradient_norm": 0.0,
            "tr_augmentation/gradient_weighted_norm": 0.0,
            "tr_augmentation/gradient_cosine_ppo_actor": 0.0,
            "tr_augmentation/gradient_diagnostics_ran": 0.0,
        }

    def capture_before_step(
        self,
        observation: TensorDict,
        action: torch.Tensor,
        actor_mean: torch.Tensor,
        actor_std: torch.Tensor,
    ) -> None:
        """Capture the pre-step policy observation and complete Markov state."""
        state = capture_tr_dynamics_state(self.env_wrapper)
        if actor_std.ndim == 1:
            actor_std = actor_std.unsqueeze(0).expand_as(action)
        self.sidecar.capture_before_step(observation, state, action, actor_mean, actor_std)

    def capture_after_step(
        self,
        observation: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping,
    ) -> None:
        """Capture the authentic successor for non-reset environments."""
        successor_state = capture_tr_dynamics_state(self.env_wrapper)
        self.sidecar.capture_after_step(observation, successor_state, rewards, dones, extras)

    def _real_training_data(self):
        states: list[TRDynamicsState] = []
        successors: list[TRDynamicsState] = []
        actions: list[torch.Tensor] = []
        environment_ids: list[torch.Tensor] = []
        rollout_steps: list[torch.Tensor] = []
        nonfinite_count = 0
        for transition in self.sidecar.transitions:
            valid = transition.valid_successor & transition.task_continuity
            finite = (
                transition.state.finite_mask()
                & transition.successor_state.finite_mask()
                & torch.isfinite(transition.action).all(dim=-1)
            )
            nonfinite_count += int((valid & ~finite).sum())
            valid &= finite
            if torch.any(valid):
                states.append(transition.state.index(valid))
                successors.append(transition.successor_state.index(valid))
                actions.append(transition.action[valid])
                environment_ids.append(transition.environment_id[valid])
                rollout_steps.append(transition.rollout_step[valid])
        self._real_nonfinite_count = nonfinite_count
        if not states:
            return None
        return (
            TRDynamicsState.concatenate(states),
            TRDynamicsState.concatenate(successors),
            torch.cat(actions),
            torch.cat(environment_ids),
            torch.cat(rollout_steps),
        )

    def _deterministic_validation_mask(self, environment_id: torch.Tensor, rollout_step: torch.Tensor) -> torch.Tensor:
        seed = int(self.cfg.get("rng_seed", 0))
        hashed = torch.remainder(environment_id * 73_856_093 + rollout_step * 19_349_663 + seed, 10_000)
        threshold = int(float(self.cfg.get("validation_fraction", 0.2)) * 10_000)
        return hashed < threshold

    def _train_models(self) -> dict[str, float]:
        data = self._real_training_data()
        diagnostics = self._empty_diagnostics()
        if data is None:
            diagnostics["tr_augmentation/rejected/nonfinite_real_transition"] = float(
                getattr(self, "_real_nonfinite_count", 0)
            )
            self.validation_ready = False
            return diagnostics
        state, successor, action, environment_id, rollout_step = data
        diagnostics["tr_augmentation/rejected/nonfinite_real_transition"] = float(self._real_nonfinite_count)
        state_features = state.dynamics_features()
        successor_features = successor.dynamics_features()
        self.state_normalizer.update(torch.cat((state_features, successor_features)))
        self.action_normalizer.update(action)
        state_norm = self.state_normalizer.normalize(state_features)
        successor_norm = self.state_normalizer.normalize(successor_features)
        inverse_state_norm = self._inverse_features(state_norm)
        inverse_successor_norm = self._inverse_features(successor_norm)
        action_norm = self.action_normalizer.normalize(action)
        validation = self._deterministic_validation_mask(environment_id, rollout_step)
        training = ~validation
        if not torch.any(training):
            training[0] = True
            validation[0] = False

        forward_losses: list[float] = []
        inverse_losses: list[float] = []
        updates = int(self.cfg.get("model_updates_per_rollout", 1))
        maximum_batch = int(self.cfg.get("model_batch_size", 4096))
        training_indices = torch.nonzero(training, as_tuple=False).squeeze(-1)
        for _ in range(updates):
            if len(training_indices) > maximum_batch:
                permutation = torch.randperm(len(training_indices), generator=self.generator, device=self.device)
                indices = training_indices[permutation[:maximum_batch]]
            else:
                indices = training_indices
            forward_prediction = self.forward_model(torch.cat((state_norm[indices], action_norm[indices]), dim=-1))
            forward_loss = self._dynamics_residual(forward_prediction, successor_norm[indices]).mean()
            self.forward_optimizer.zero_grad()
            forward_loss.backward()
            self.forward_optimizer.step()
            forward_losses.append(forward_loss.item())

            inverse_prediction = self.inverse_model(
                torch.cat((inverse_state_norm[indices], inverse_successor_norm[indices]), dim=-1)
            )
            inverse_loss = (inverse_prediction - action_norm[indices]).square().mean()
            self.inverse_optimizer.zero_grad()
            inverse_loss.backward()
            self.inverse_optimizer.step()
            inverse_losses.append(inverse_loss.item())

        decay = float(self.cfg.get("ema_decay", 0.995))
        with torch.no_grad():
            for target, source in zip(self.forward_target.parameters(), self.forward_model.parameters()):
                target.lerp_(source, 1.0 - decay)
            for target, source in zip(self.inverse_target.parameters(), self.inverse_model.parameters()):
                target.lerp_(source, 1.0 - decay)

        diagnostics["tr_augmentation/forward_training_loss"] = sum(forward_losses) / len(forward_losses)
        diagnostics["tr_augmentation/inverse_training_loss"] = sum(inverse_losses) / len(inverse_losses)
        validation_indices = torch.nonzero(validation, as_tuple=False).squeeze(-1)
        minimum_validation = int(self.cfg.get("minimum_validation_samples", 128))
        if len(validation_indices) < minimum_validation:
            self.validation_ready = False
            return diagnostics
        with torch.no_grad():
            forward_prediction = self.forward_target(
                torch.cat((state_norm[validation_indices], action_norm[validation_indices]), dim=-1)
            )
            real_residual = self._dynamics_residual(forward_prediction, successor_norm[validation_indices])
            inverse_prediction = self.inverse_target(
                torch.cat(
                    (inverse_state_norm[validation_indices], inverse_successor_norm[validation_indices]),
                    dim=-1,
                )
            )
            inverse_residual = (inverse_prediction - action_norm[validation_indices]).square().mean(dim=-1)
            validation_loss = real_residual.mean()
            diagnostics["tr_augmentation/forward_validation_loss"] = validation_loss.item()
            diagnostics["tr_augmentation/inverse_validation_loss"] = inverse_residual.mean().item()
            quantile = float(self.cfg.get("validation_quantile", 0.95))
            multiplier = float(self.cfg.get("threshold_multiplier", 2.0))
            self.filter_beta = torch.quantile(real_residual, quantile) * multiplier
            diagnostics["tr_augmentation/beta"] = self.filter_beta.item()
            self.validation_ready = bool(
                torch.isfinite(self.filter_beta)
                and validation_loss <= float(self.cfg.get("maximum_validation_loss", 1.0))
            )
        return diagnostics

    @staticmethod
    def _transition_usable(transition: TRSidecarTransition, environment_index: int) -> bool:
        return bool(transition.valid_successor[environment_index] and transition.task_continuity[environment_index])

    def _constant_task_segments(self) -> tuple[list[list[tuple[TRSidecarTransition, int]]], dict[str, int]]:
        segments: list[list[tuple[TRSidecarTransition, int]]] = []
        rejection = {"termination_timeout": 0, "task_discontinuity": 0, "previous_action_unknown": 0}
        for environment_index in range(self.sidecar.num_envs):
            current: list[tuple[TRSidecarTransition, int]] = []
            for transition in self.sidecar.transitions:
                if not transition.valid_successor[environment_index]:
                    rejection["termination_timeout"] += 1
                    if current:
                        segments.append(current)
                        rejection["previous_action_unknown"] += min(2, len(current))
                        current = []
                    continue
                if not transition.task_continuity[environment_index]:
                    rejection["task_discontinuity"] += 1
                    if current:
                        segments.append(current)
                        rejection["previous_action_unknown"] += min(2, len(current))
                        current = []
                    continue
                episode = int(transition.episode_id[environment_index])
                step = int(transition.rollout_step[environment_index])
                if current:
                    previous = current[-1][0]
                    previous_episode = int(previous.episode_id[environment_index])
                    previous_step = int(previous.rollout_step[environment_index])
                    if episode != previous_episode or step != previous_step + 1:
                        segments.append(current)
                        rejection["previous_action_unknown"] += min(2, len(current))
                        current = []
                current.append((transition, environment_index))
            if current:
                segments.append(current)
                rejection["previous_action_unknown"] += min(2, len(current))
        return segments, rejection

    def _build_candidate_pool(self, diagnostics: dict[str, float]) -> TRAugmentationPool | None:
        segments, boundary_rejection = self._constant_task_segments()
        for name, count in boundary_rejection.items():
            diagnostics[f"tr_augmentation/rejected/{name}"] = float(count)
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        current_states: list[TRDynamicsState] = []
        next_states: list[TRDynamicsState] = []
        impulses: list[torch.Tensor] = []
        contact_modes: list[torch.Tensor] = []
        environment_ids: list[torch.Tensor] = []
        episode_ids: list[torch.Tensor] = []
        rollout_steps: list[torch.Tensor] = []
        action_source = self.cfg.get("action_source", "analytic")
        for segment in segments:
            if len(segment) < 3:
                continue
            environment_index = segment[0][1]
            selection = slice(environment_index, environment_index + 1)
            segment_states = [segment[0][0].state.index(selection)]
            segment_states.extend(transition.successor_state.index(selection) for transition, _ in segment)
            segment_actions = [transition.action[selection] for transition, _ in segment]
            if action_source == "learned_inverse":
                reversed_segment = self._learned_reversed_sequence(
                    segment_states,
                    segment_actions,
                    environment_index,
                )
            else:
                reversed_segment = build_reversed_sequence_segment(
                    segment_states,
                    segment_actions,
                    base_height_range=self.base_height_range,
                )
            observations.append(reversed_segment.observations)
            actions.append(reversed_segment.actions)
            current_states.append(reversed_segment.current_state)
            next_states.append(reversed_segment.successor_state)
            for transition, _ in reversed(segment[:-2]):
                impulses.append(transition.contact_impulse[selection])
                contact_modes.append(transition.contact_mode[selection])
                environment_ids.append(transition.environment_id[selection])
                episode_ids.append(transition.episode_id[selection])
                rollout_steps.append(transition.rollout_step[selection])
        if not observations:
            self.replay_candidates = None
            return None
        observation = torch.cat(observations)
        candidate_action = torch.cat(actions)
        current_state = TRDynamicsState.concatenate(current_states)
        next_state = TRDynamicsState.concatenate(next_states)
        contact_impulse = torch.cat(impulses)
        contact_mode = torch.cat(contact_modes)
        environment_id = torch.cat(environment_ids)
        episode_id = torch.cat(episode_ids)
        rollout_step = torch.cat(rollout_steps)
        current_features = self.state_normalizer.normalize(current_state.dynamics_features())
        next_features = self.state_normalizer.normalize(next_state.dynamics_features())
        with torch.no_grad():
            normalized_action = self.action_normalizer.normalize(candidate_action)
            predicted_next = self.forward_target(torch.cat((current_features, normalized_action), dim=-1))
            reverse_residual = self._dynamics_residual(predicted_next, next_features)

        diagnostics["tr_augmentation/candidate_count"] = float(len(candidate_action))
        foot_phase = torch.remainder(
            current_state.common_gait_phase.unsqueeze(-1) + current_state.foot_phase_offsets, 1.0
        )
        successor_foot_phase = torch.remainder(
            next_state.common_gait_phase.unsqueeze(-1) + next_state.foot_phase_offsets, 1.0
        )
        phase_event_crossing = time_reversal_phase_event_crossing(
            foot_phase,
            successor_foot_phase,
            current_state.swing_ratio,
        )
        filter_enabled = bool(self.cfg.get("filter_enabled", True))
        if not filter_enabled:
            raise RuntimeError("Dynamics-filtered augmentation cannot evaluate candidates with its filter disabled.")
        gates = time_reversal_filter_mask(
            observation=observation,
            action=candidate_action,
            reverse_residual=reverse_residual,
            beta=self.filter_beta,
            foot_phase=foot_phase,
            successor_foot_phase=successor_foot_phase,
            phase_event_crossing=phase_event_crossing,
            swing_ratio=current_state.swing_ratio,
            contact_impulse=contact_impulse,
            model_quality=self.validation_ready,
            phase_boundary_margin=float(self.cfg.get("phase_boundary_margin", 0.03)),
            maximum_contact_impulse=float(self.cfg.get("maximum_contact_impulse", 20.0)),
            action_abs_limit=float(self.cfg.get("action_abs_limit", 10.0)),
        )
        accepted = gates.accepted

        successor_observation = build_time_reversal_observation(
            next_state,
            previous_action=next_state.previous_action,
            second_previous_action=next_state.second_previous_action,
            base_height_range=self.base_height_range,
        )
        learned_threshold = torch.as_tensor(
            self.filter_beta,
            dtype=reverse_residual.dtype,
            device=reverse_residual.device,
        ).expand_as(reverse_residual)
        self.replay_candidates = TRReplayCandidatePool(
            observations=observation.detach(),
            successor_observations=successor_observation.detach(),
            actions=candidate_action.detach(),
            current_state=current_state.detach(),
            successor_state=next_state.detach(),
            learned_residual=reverse_residual.detach(),
            learned_threshold=learned_threshold.detach(),
            filter_accepted=accepted.detach(),
            finite_gate=gates.finite.detach(),
            action_gate=gates.action.detach(),
            phase_gate=gates.phase.detach(),
            contact_impulse_gate=gates.contact_impulse.detach(),
            dynamics_gate=gates.dynamics.detach(),
            model_quality_gate=gates.model_quality.detach(),
            contact_impulse=contact_impulse.detach(),
            contact_mode=contact_mode.detach(),
            gait_row=current_state.gait_row.detach(),
            command=current_state.velocity_command.detach(),
            environment_id=environment_id.detach(),
            episode_id=episode_id.detach(),
            rollout_step=rollout_step.detach(),
        )

        diagnostics["tr_augmentation/rejected/nonfinite"] = float((~gates.finite).sum())
        diagnostics["tr_augmentation/rejected/action_limit"] = float((gates.finite & ~gates.action).sum())
        diagnostics["tr_augmentation/rejected/phase_boundary"] = float(
            (gates.finite & gates.action & ~gates.phase).sum()
        )
        diagnostics["tr_augmentation/rejected/successor_phase_boundary"] = float(
            (gates.finite & gates.action & ~gates.phase_successor).sum()
        )
        diagnostics["tr_augmentation/rejected/phase_event_crossing"] = float(
            (gates.finite & gates.action & ~gates.phase_event).sum()
        )
        diagnostics["tr_augmentation/rejected/contact_impulse"] = float(
            (gates.finite & gates.action & gates.phase & ~gates.contact_impulse).sum()
        )
        diagnostics["tr_augmentation/rejected/dynamics_residual"] = float(
            (gates.finite & gates.action & gates.phase & gates.contact_impulse & ~gates.dynamics).sum()
        )
        diagnostics["tr_augmentation/rejected/model_quality"] = float((~gates.model_quality).sum())
        accepted_count = int(accepted.sum())
        diagnostics["tr_augmentation/accepted_count"] = float(accepted_count)
        diagnostics["tr_augmentation/acceptance_fraction"] = accepted.float().mean().item()
        acceptance_key = (
            "tr_augmentation/learned_action_acceptance"
            if action_source == "learned_inverse"
            else "tr_augmentation/analytic_action_acceptance"
        )
        diagnostics[acceptance_key] = diagnostics["tr_augmentation/acceptance_fraction"]
        if len(reverse_residual):
            diagnostics["tr_augmentation/reverse_residual_mean"] = reverse_residual.mean().item()
            diagnostics["tr_augmentation/reverse_residual_median"] = reverse_residual.median().item()
            for quantile, suffix in ((0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
                diagnostics[f"tr_augmentation/reverse_residual_{suffix}"] = torch.quantile(
                    reverse_residual, quantile
                ).item()
        if accepted_count == 0:
            return None
        confidence = torch.ones_like(reverse_residual)
        if bool(self.cfg.get("use_confidence_weights", True)):
            confidence = torch.exp(-reverse_residual / (self.filter_beta + _EPS))
        pool = TRAugmentationPool(
            observations=observation[accepted].detach(),
            actions=candidate_action[accepted].detach(),
            confidence=confidence[accepted].detach(),
            residual=reverse_residual[accepted].detach(),
            gait_row=current_state.gait_row[accepted].detach(),
            command=current_state.velocity_command[accepted].detach(),
            contact_mode=contact_mode[accepted].detach(),
        )
        self._add_stratified_acceptance(diagnostics, current_state, contact_mode, accepted)
        return pool

    @staticmethod
    def _add_stratified_acceptance(
        diagnostics: dict[str, float],
        state: TRDynamicsState,
        contact_mode: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        for row in torch.unique(state.gait_row).tolist():
            selected = state.gait_row == row
            diagnostics[f"tr_augmentation/acceptance/gait_row/{row}"] = accepted[selected].float().mean().item()
        for family in dict.fromkeys(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES):
            family_rows = torch.tensor(
                [
                    index
                    for index, row_family in enumerate(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES)
                    if row_family == family
                ],
                device=state.gait_row.device,
            )
            selected = torch.isin(state.gait_row, family_rows)
            diagnostics[f"tr_augmentation/acceptance/gait_family/{family}"] = (
                accepted[selected].float().mean().item() if torch.any(selected) else 0.0
            )
        command_x = state.velocity_command[:, 0]
        sign = torch.sign(command_x).to(torch.long)
        for value, name in ((-1, "negative"), (0, "zero"), (1, "positive")):
            selected = sign == value
            diagnostics[f"tr_augmentation/acceptance/command_sign/{name}"] = (
                accepted[selected].float().mean().item() if torch.any(selected) else 0.0
            )
        edges = torch.tensor((0.5, 1.0, 1.5), device=command_x.device, dtype=command_x.dtype)
        bins = torch.bucketize(command_x.abs(), edges)
        for value in range(len(edges) + 1):
            selected = bins == value
            diagnostics[f"tr_augmentation/acceptance/speed_bin/{value}"] = (
                accepted[selected].float().mean().item() if torch.any(selected) else 0.0
            )
        for value in torch.unique(contact_mode).tolist():
            selected = contact_mode == value
            diagnostics[f"tr_augmentation/acceptance/contact_mode/{value}"] = accepted[selected].float().mean().item()

    def prepare_update(self) -> dict[str, float]:
        """Train side models, calibrate the filter, and construct one detached pool."""
        diagnostics = self._train_models()
        self.replay_candidates = None
        self.pool = self._build_candidate_pool(diagnostics)
        self._last_diagnostics = diagnostics
        return diagnostics

    def actor_nll(self, actor: nn.Module, original_batch_size: int) -> torch.Tensor:
        """Return capped confidence-weighted actor NLL for an independent pool sample."""
        if self.pool is None or self.pool.count == 0:
            return torch.zeros((), device=self.device)
        maximum_ratio = float(self.cfg.get("max_augmented_to_original_ratio", 0.25))
        sample_count = min(self.pool.count, math.floor(maximum_ratio * original_batch_size))
        if sample_count <= 0:
            return torch.zeros((), device=self.device)
        indices = torch.randperm(self.pool.count, generator=self.generator, device=self.device)[:sample_count]
        observations = TensorDict(
            {"policy": self.pool.observations[indices]},
            batch_size=[sample_count],
            device=self.device,
        )
        # RSL-RL's stochastic path is the public operation that updates every
        # supported distribution (including heteroscedastic heads), but its
        # returned sample is unused. Run it with checkpointed augmentation RNG
        # inside a fork so neither CPU nor CUDA global RNG is perturbed.
        fork_devices = []
        if self.device.type == "cuda":
            fork_devices.append(self.device.index if self.device.index is not None else torch.cuda.current_device())
        with torch.random.fork_rng(devices=fork_devices):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.generator.get_state(), self.device)
            else:
                torch.set_rng_state(self.generator.get_state())
            actor(observations, stochastic_output=True)
            if self.device.type == "cuda":
                self.generator.set_state(torch.cuda.get_rng_state(self.device))
            else:
                self.generator.set_state(torch.get_rng_state())
        log_probability = actor.get_output_log_prob(self.pool.actions[indices].detach())
        confidence = self.pool.confidence[indices].detach()
        return time_reversal_augmentation_nll(
            log_probability,
            torch.ones_like(confidence, dtype=torch.bool),
            confidence,
        )

    def finish_update(self) -> None:
        """Discard training inputs while retaining the optional replay diagnostic."""
        self.sidecar.clear()
        self.pool = None

    def state_dict(self) -> dict:
        """Checkpoint models, optimizers, normalization, RNG, filter, and sidecar."""
        return {
            "schema_version": self._CHECKPOINT_SCHEMA_VERSION,
            "semantic_config": self._semantic_config(),
            "forward_model": self.forward_model.state_dict(),
            "inverse_model": self.inverse_model.state_dict(),
            "forward_target": self.forward_target.state_dict(),
            "inverse_target": self.inverse_target.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            "inverse_optimizer": self.inverse_optimizer.state_dict(),
            "state_normalizer": self.state_normalizer.state_dict(),
            "action_normalizer": self.action_normalizer.state_dict(),
            "generator_state": self.generator.get_state(),
            "initialization_generator_state": self.initialization_generator.get_state(),
            "filter_beta": self.filter_beta,
            "validation_ready": self.validation_ready,
            "sidecar": self.sidecar.state_dict(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore the complete augmentation state without random restart."""
        if not isinstance(state, Mapping):
            raise ValueError("time_reversal_augmentation_state must be a mapping.")
        required = {
            "schema_version",
            "semantic_config",
            "forward_model",
            "inverse_model",
            "forward_target",
            "inverse_target",
            "forward_optimizer",
            "inverse_optimizer",
            "state_normalizer",
            "action_normalizer",
            "generator_state",
            "initialization_generator_state",
            "filter_beta",
            "validation_ready",
            "sidecar",
        }
        allowed = required | {"schedule_iteration"}
        if not required.issubset(state) or not set(state).issubset(allowed):
            missing = sorted(required - set(state))
            unexpected = sorted(set(state) - allowed)
            raise ValueError(
                "time_reversal_augmentation_state schema fields do not match: "
                f"missing={missing}, unexpected={unexpected}."
            )
        schema_version = state.get("schema_version")
        if schema_version != self._CHECKPOINT_SCHEMA_VERSION or isinstance(schema_version, bool):
            raise ValueError(
                "Unsupported time_reversal_augmentation_state schema_version: "
                f"expected {self._CHECKPOINT_SCHEMA_VERSION}, received {schema_version!r}."
            )
        saved_semantics = state.get("semantic_config")
        expected_semantics = self._semantic_config()
        exact_semantics = isinstance(saved_semantics, Mapping) and saved_semantics.keys() == expected_semantics.keys()
        exact_semantics = exact_semantics and all(
            type(saved_semantics[name]) is type(expected_semantics[name])
            and saved_semantics[name] == expected_semantics[name]
            for name in expected_semantics
        )
        if not exact_semantics:
            raise ValueError(
                "Checkpoint augmentation semantic_config does not exactly match the configured augmentation: "
                f"expected {expected_semantics!r}, received {saved_semantics!r}."
            )
        self.forward_model.load_state_dict(state["forward_model"])
        self.inverse_model.load_state_dict(state["inverse_model"])
        self.forward_target.load_state_dict(state["forward_target"])
        self.inverse_target.load_state_dict(state["inverse_target"])
        self.forward_optimizer.load_state_dict(state["forward_optimizer"])
        self.inverse_optimizer.load_state_dict(state["inverse_optimizer"])
        self.state_normalizer.load_state_dict(state["state_normalizer"])
        self.action_normalizer.load_state_dict(state["action_normalizer"])
        self.generator.set_state(state["generator_state"].cpu())
        self.initialization_generator.set_state(state["initialization_generator_state"].cpu())
        self.filter_beta = state["filter_beta"].to(self.device)
        self.validation_ready = bool(state["validation_ready"])
        self.sidecar.load_state_dict(state["sidecar"])


def validate_filter_with_one_step_replay(
    candidates: TRReplayCandidatePool,
    replay_residual: Callable[[TRReplayCandidatePool], torch.Tensor],
    *,
    simulator_threshold: float,
    maximum_per_decision: int | None = None,
) -> dict[str, float]:
    """Offline utility comparing learned decisions with caller-provided simulator replay.

    The callback may launch GPU simulation, but this function is never invoked
    by normal training. It receives a sampled :class:`TRReplayCandidatePool`
    containing complete current/target Markov state, action, learned decision,
    individual gates, and rollout identity. It must restore and advance the
    simulator itself and return one scalar residual per candidate.
    """
    sample = candidates.sample_by_filter_decision(maximum_per_decision)
    if sample.count == 0:
        raise ValueError("One-step replay validation requires at least one candidate.")
    simulator_residual = replay_residual(sample)
    if simulator_residual.shape != sample.learned_residual.shape:
        raise ValueError("Simulator replay must return one residual per candidate.")
    if not bool(torch.isfinite(simulator_residual).all()):
        raise ValueError("Simulator replay residuals must be finite.")
    learned_accept = sample.filter_accepted
    simulator_accept = simulator_residual <= simulator_threshold
    true_positive = (learned_accept & simulator_accept).sum().item()
    false_positive = (learned_accept & ~simulator_accept).sum().item()
    false_negative = (~learned_accept & simulator_accept).sum().item()
    true_negative = (~learned_accept & ~simulator_accept).sum().item()
    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
        "agreement": (learned_accept == simulator_accept).float().mean().item(),
        "candidate_count": float(sample.count),
        "learned_accepted_count": float(learned_accept.sum().item()),
        "learned_rejected_count": float((~learned_accept).sum().item()),
        "simulator_accepted_count": float(simulator_accept.sum().item()),
    }
