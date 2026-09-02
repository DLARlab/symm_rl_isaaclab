# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared environment behavior for symmetric quadruped tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


def _broadcast_action_parameter(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Broadcast one affine action parameter to an action-shaped tensor."""
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    try:
        return torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must broadcast to action shape {tuple(reference.shape)}; received {tuple(tensor.shape)}."
        ) from exc


def compute_requested_joint_position_targets(
    raw_actions: torch.Tensor,
    action_offset: torch.Tensor | float,
    action_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Convert raw actions into requested joint-position targets before safety clamping.

    Args:
        raw_actions: Raw policy actions, shape ``(..., num_joints)``.
        action_offset: Default joint-position offset [rad].
        action_scale: Joint-position change per unit raw action [rad].

    Returns:
        Requested joint-position targets [rad] in raw-action joint order.
    """
    offset = _broadcast_action_parameter(action_offset, raw_actions, name="action_offset")
    scale = _broadcast_action_parameter(action_scale, raw_actions, name="action_scale")
    return offset + scale * raw_actions


def compute_feasible_raw_action_bounds(
    action_offset: torch.Tensor | float,
    action_scale: torch.Tensor | float,
    soft_joint_position_limits: torch.Tensor,
    interior_margin_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive feasible per-joint raw-action bounds from soft position limits.

    Args:
        action_offset: Default joint-position offset [rad].
        action_scale: Joint-position change per unit raw action [rad].
        soft_joint_position_limits: Lower and upper joint limits [rad], shape
            ``(..., num_joints, 2)``.
        interior_margin_fraction: Fraction of each joint range removed from both
            ends of the feasible target interval.

    Returns:
        The lower and upper feasible raw-action bounds in the same joint order
        as :paramref:`soft_joint_position_limits`.

    Raises:
        ValueError: If limits, scale, or the interior margin are invalid.
    """
    if not 0.0 <= interior_margin_fraction < 0.5:
        raise ValueError(f"interior_margin_fraction must be in [0, 0.5); received {interior_margin_fraction}.")
    if soft_joint_position_limits.shape[-1] != 2:
        raise ValueError(
            "soft_joint_position_limits must have a final lower/upper dimension of size 2; "
            f"received shape {tuple(soft_joint_position_limits.shape)}."
        )
    lower_limits = soft_joint_position_limits[..., 0]
    upper_limits = soft_joint_position_limits[..., 1]
    limit_range = upper_limits - lower_limits
    if torch.any(~torch.isfinite(soft_joint_position_limits)) or torch.any(limit_range <= 0.0):
        raise ValueError("soft_joint_position_limits must contain finite, strictly ordered bounds.")

    offset = _broadcast_action_parameter(action_offset, lower_limits, name="action_offset")
    scale = _broadcast_action_parameter(action_scale, lower_limits, name="action_scale")
    if torch.any(~torch.isfinite(offset)) or torch.any(~torch.isfinite(scale)) or torch.any(scale == 0.0):
        raise ValueError("action_offset and action_scale must be finite, and action_scale must be nonzero.")

    target_lower = lower_limits + interior_margin_fraction * limit_range
    target_upper = upper_limits - interior_margin_fraction * limit_range
    first_bound = (target_lower - offset) / scale
    second_bound = (target_upper - offset) / scale
    return torch.minimum(first_bound, second_bound), torch.maximum(first_bound, second_bound)


def compute_requested_target_overflow(
    requested_targets: torch.Tensor,
    soft_joint_position_limits: torch.Tensor,
    margin_fraction: float,
) -> torch.Tensor:
    """Return uncapped requested-target overflow normalized by joint range.

    Args:
        requested_targets: Requested joint-position targets before safety clamping [rad].
        soft_joint_position_limits: Lower and upper soft limits [rad], shape
            ``requested_targets.shape + (2,)``.
        margin_fraction: Interior margin as a fraction of each joint range.

    Returns:
        Nonnegative dimensionless overflow in requested-target joint order.
    """
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError(f"margin_fraction must be in [0, 0.5); received {margin_fraction}.")
    if soft_joint_position_limits.shape != requested_targets.shape + (2,):
        raise ValueError(
            "soft_joint_position_limits must match requested_targets with a final lower/upper dimension; "
            f"received {tuple(requested_targets.shape)} and {tuple(soft_joint_position_limits.shape)}."
        )
    lower_limits = soft_joint_position_limits[..., 0]
    upper_limits = soft_joint_position_limits[..., 1]
    limit_range = upper_limits - lower_limits
    margin = margin_fraction * limit_range
    lower_overflow = torch.relu(lower_limits + margin - requested_targets)
    upper_overflow = torch.relu(requested_targets - (upper_limits - margin))
    return (lower_overflow + upper_overflow) / limit_range.clamp_min(torch.finfo(requested_targets.dtype).eps)


def _joint_target_limit_margin(reward_manager) -> float:
    """Return the configured target-limit margin or the historical diagnostic default."""
    try:
        target_limit_cfg = reward_manager.get_term_cfg("joint_target_limits")
    except (KeyError, ValueError):
        target_limit_cfg = None
    return 0.05 if target_limit_cfg is None else float(target_limit_cfg.params.get("margin_fraction", 0.05))


def _clip_reward_before_termination(
    total_reward: torch.Tensor,
    termination_reward: torch.Tensor,
) -> torch.Tensor:
    """Clip nonterminal reward at zero while preserving terminal penalties."""
    running_reward = total_reward - termination_reward
    return torch.clamp_min(running_reward, 0.0) + termination_reward


def _combine_running_and_termination_reward(
    running_reward: torch.Tensor,
    termination_reward: torch.Tensor,
) -> torch.Tensor:
    """Combine rewards using the stable nonnegative-running-reward rule."""
    return _clip_reward_before_termination(running_reward + termination_reward, termination_reward)


def _clamp_joint_position_targets(
    joint_position_targets: torch.Tensor,
    soft_joint_position_limits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp joint position targets to the articulation soft limits.

    Args:
        joint_position_targets: Joint position targets [rad], shape ``(num_envs, num_joints)``.
        soft_joint_position_limits: Lower and upper joint limits [rad], shape
            ``(num_envs, num_joints, 2)``.

    Returns:
        A tuple containing the safe joint position targets [rad] and the fraction of targets that were clipped.
    """
    lower_limits = soft_joint_position_limits[..., 0]
    upper_limits = soft_joint_position_limits[..., 1]
    clamped_targets = torch.maximum(torch.minimum(joint_position_targets, upper_limits), lower_limits)
    clipped_fraction = (clamped_targets != joint_position_targets).to(joint_position_targets.dtype).mean()
    return clamped_targets, clipped_fraction


def _soft_joint_limit_diagnostics(
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    margin_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure proximity to and violation of the soft joint-position limits."""
    lower_limits = soft_joint_pos_limits[..., 0]
    upper_limits = soft_joint_pos_limits[..., 1]
    limit_range = (upper_limits - lower_limits).clamp_min(torch.finfo(joint_pos.dtype).eps)
    normalized_position = 2.0 * (joint_pos - lower_limits) / limit_range - 1.0
    normalized_magnitude = normalized_position.abs()
    near_limit_fraction = (normalized_magnitude >= 1.0 - 2.0 * margin_fraction).to(joint_pos.dtype).mean()
    limit_violation_fraction = (normalized_magnitude > 1.0).to(joint_pos.dtype).mean()
    return near_limit_fraction, limit_violation_fraction, normalized_magnitude.max()


class SymmQuadrupedManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with symmetric locomotion update ordering."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one RL step using the symmetric quadruped update ordering."""
        action = action.to(self.device)
        self.action_manager.process_action(action)
        self._clamp_processed_joint_position_targets()

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.is_rendering

        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        total_reward = self.reward_manager.compute(dt=self.step_dt)
        termination_cfg = self.reward_manager.get_term_cfg("termination_penalty")
        termination_reward = self.reset_terminated.to(total_reward.dtype) * termination_cfg.weight * self.step_dt
        running_reward = total_reward - termination_reward
        self.reward_buf = _clip_reward_before_termination(total_reward, termination_reward)
        step_diagnostics = self._compute_step_diagnostics(action, running_reward)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # Schedule the next desired signal only after scoring and recording the
        # completed transition under the command/gait that generated its action.
        self.command_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.extras.setdefault("log", {}).update(step_diagnostics)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _clamp_processed_joint_position_targets(self) -> None:
        """Clamp the processed joint-position action term to the robot soft limits."""
        joint_term = self.action_manager.get_term("joint_pos")
        joint_ids = joint_term._joint_ids
        soft_joint_position_limits = self.scene["robot"].data.soft_joint_pos_limits.torch[:, joint_ids]
        requested_targets = compute_requested_joint_position_targets(
            joint_term.raw_actions,
            joint_term._offset,
            joint_term._scale,
        )
        clamped_targets, clipped_fraction = _clamp_joint_position_targets(
            joint_term.processed_actions,
            soft_joint_position_limits,
        )
        joint_term.processed_actions.copy_(clamped_targets)
        self._requested_joint_position_targets = requested_targets.detach()
        self._executed_joint_position_targets = clamped_targets.detach()
        self._joint_target_clipped_fraction = clipped_fraction.detach()

    def get_joint_position_action_metadata(
        self,
        action_term_name: str = "joint_pos",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return affine action metadata in the resolved action-term joint order.

        Args:
            action_term_name: Name of the joint-position action term.

        Returns:
            The action offset [rad], action scale [rad], and soft joint-position
            limits [rad]. Each tensor has a leading environment dimension and
            follows ``joint_term._joint_ids`` exactly.
        """
        action_term = self.action_manager.get_term(action_term_name)
        soft_limits = self.scene["robot"].data.soft_joint_pos_limits.torch[:, action_term._joint_ids]
        reference = soft_limits[..., 0]
        offset = _broadcast_action_parameter(action_term._offset, reference, name="action_offset")
        scale = _broadcast_action_parameter(action_term._scale, reference, name="action_scale")
        return offset, scale, soft_limits

    def get_joint_position_action_feasible_bounds(
        self,
        action_term_name: str = "joint_pos",
        interior_margin_fraction: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-joint feasible raw-action bounds in action-term order.

        Args:
            action_term_name: Name of the joint-position action term.
            interior_margin_fraction: Fraction of each soft joint range
                removed from both ends of the feasible target interval.

        Returns:
            Lower and upper dimensionless raw-action bounds with a leading
            environment dimension.
        """
        offset, scale, soft_limits = self.get_joint_position_action_metadata(action_term_name)
        return compute_feasible_raw_action_bounds(
            offset,
            scale,
            soft_limits,
            interior_margin_fraction=interior_margin_fraction,
        )

    def set_training_iteration(self, iteration: int, command_name: str = "base_velocity") -> None:
        """Propagate an absolute learning iteration to the gait curriculum.

        Args:
            iteration: Absolute zero-based learning iteration.
            command_name: Name of the gait command term.
        """
        command_term = self.command_manager.get_term(command_name)
        if not hasattr(command_term, "set_training_iteration"):
            raise TypeError(f"Command term {command_name!r} does not support an iteration-based curriculum.")
        command_term.set_training_iteration(iteration)

    def get_command_curriculum_state(self, command_name: str = "base_velocity") -> dict | None:
        """Return checkpoint state for an enabled task-local command curriculum.

        Args:
            command_name: Name of the gait command term.

        Returns:
            Exact command-curriculum state, or ``None`` when the optional
            stateful curriculum is disabled.
        """
        command_term = self.command_manager.get_term(command_name)
        getter = getattr(command_term, "get_command_curriculum_state", None)
        return getter() if callable(getter) else None

    def load_command_curriculum_state(self, state: dict, command_name: str = "base_velocity") -> None:
        """Restore exact state for an enabled task-local command curriculum.

        Args:
            state: State produced by :meth:`get_command_curriculum_state`.
            command_name: Name of the gait command term.
        """
        command_term = self.command_manager.get_term(command_name)
        loader = getattr(command_term, "load_command_curriculum_state", None)
        if not callable(loader):
            raise TypeError(f"Command term {command_name!r} does not support stateful curriculum restore.")
        loader(state)

    def _compute_step_diagnostics(
        self,
        action: torch.Tensor,
        running_reward: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training diagnostics before terminated environments are reset."""
        diagnostics = {}
        action_abs = action.detach().abs()
        diagnostics["Diagnostics/action_abs_mean"] = action_abs.mean()
        diagnostics["Diagnostics/action_abs_max"] = action_abs.max()
        reward_clipped = running_reward < 0.0
        diagnostics["Diagnostics/reward_clipped_fraction"] = reward_clipped.to(running_reward.dtype).mean()
        diagnostics["Diagnostics/reward_negative_fraction"] = reward_clipped.to(running_reward.dtype).mean()

        robot = self.scene["robot"]
        joint_term = self.action_manager.get_term("joint_pos")
        joint_ids = joint_term._joint_ids
        joint_pos = robot.data.joint_pos.torch[:, joint_ids]
        soft_joint_pos_limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
        joint_pos_target = joint_term.processed_actions
        near_limit, limit_violation, normalized_max = _soft_joint_limit_diagnostics(
            joint_pos,
            soft_joint_pos_limits,
        )
        diagnostics["Diagnostics/joint_near_limit_fraction"] = near_limit
        diagnostics["Diagnostics/joint_limit_violation_fraction"] = limit_violation
        diagnostics["Diagnostics/joint_limit_normalized_max"] = normalized_max
        target_near_limit, target_limit_violation, target_normalized_max = _soft_joint_limit_diagnostics(
            joint_pos_target,
            soft_joint_pos_limits,
        )
        diagnostics["Diagnostics/joint_target_near_limit_fraction"] = target_near_limit
        diagnostics["Diagnostics/joint_target_limit_violation_fraction"] = target_limit_violation
        diagnostics["Diagnostics/joint_target_limit_normalized_max"] = target_normalized_max
        diagnostics["Diagnostics/joint_target_clipped_fraction"] = getattr(
            self,
            "_joint_target_clipped_fraction",
            torch.zeros((), device=joint_pos_target.device, dtype=joint_pos_target.dtype),
        )
        requested_target = getattr(self, "_requested_joint_position_targets", joint_pos_target)
        requested_target = requested_target.to(device=joint_pos_target.device, dtype=joint_pos_target.dtype)
        target_limit_margin = _joint_target_limit_margin(self.reward_manager)
        normalized_overflow = compute_requested_target_overflow(
            requested_target,
            soft_joint_pos_limits,
            margin_fraction=target_limit_margin,
        )
        executed_target_clipped = joint_pos_target != requested_target
        diagnostics["Diagnostics/requested_target_overflow_fraction"] = (
            (normalized_overflow > 0.0).to(joint_pos_target.dtype).mean()
        )
        diagnostics["Diagnostics/requested_target_overflow_mean"] = normalized_overflow.mean()
        diagnostics["Diagnostics/requested_target_overflow_p95"] = torch.quantile(normalized_overflow, 0.95)
        diagnostics["Diagnostics/executed_target_clipped_fraction"] = executed_target_clipped.to(
            joint_pos_target.dtype
        ).mean()

        joint_overflow_p95 = torch.quantile(normalized_overflow, 0.95, dim=0)
        joint_names = getattr(joint_term, "_joint_names", [f"joint_{index}" for index in range(joint_pos.shape[-1])])
        for joint_index, joint_name in enumerate(joint_names):
            joint_overflow = normalized_overflow[:, joint_index]
            metric_suffix = str(joint_name).replace("/", "_").replace(" ", "_")
            diagnostics[f"Diagnostics/requested_target_overflow_fraction/{metric_suffix}"] = (
                (joint_overflow > 0.0).to(joint_pos_target.dtype).mean()
            )
            diagnostics[f"Diagnostics/requested_target_overflow_mean/{metric_suffix}"] = joint_overflow.mean()
            diagnostics[f"Diagnostics/requested_target_overflow_p95/{metric_suffix}"] = joint_overflow_p95[joint_index]
            diagnostics[f"Diagnostics/executed_target_clipped_fraction/{metric_suffix}"] = (
                executed_target_clipped[:, joint_index].to(joint_pos_target.dtype).mean()
            )

        family_offsets = {"hip": 0, "thigh": 1, "calf": 2}
        for family_name, offset in family_offsets.items():
            family_overflow = normalized_overflow[:, offset::3]
            if family_overflow.shape[-1] == 0:
                continue
            diagnostics[f"Diagnostics/requested_target_overflow_fraction_{family_name}"] = (
                (family_overflow > 0.0).to(joint_pos_target.dtype).mean()
            )
            diagnostics[f"Diagnostics/requested_target_overflow_mean_{family_name}"] = family_overflow.mean()
            diagnostics[f"Diagnostics/requested_target_overflow_p95_{family_name}"] = torch.quantile(
                family_overflow, 0.95
            )
            diagnostics[f"Diagnostics/executed_target_clipped_fraction_{family_name}"] = (
                executed_target_clipped[:, offset::3].to(joint_pos_target.dtype).mean()
            )

        foot_phase_diagnostics = getattr(self, "_foot_phase_diagnostics", None)
        if foot_phase_diagnostics is not None:
            foot_phase_cfg = self.reward_manager.get_term_cfg("foot_phase")
            foot_phase_weight = float(foot_phase_cfg.weight)
            foot_phase_reduction = foot_phase_cfg.params.get("reduction", "sum")
            if foot_phase_reduction not in ("sum", "mean"):
                raise ValueError(f"Unsupported foot-phase reduction: {foot_phase_reduction!r}.")
            raw_foot_phase_sum = foot_phase_diagnostics["raw_sum"]
            raw_foot_phase_mean = foot_phase_diagnostics["raw_mean"]
            reduced_foot_phase = raw_foot_phase_sum if foot_phase_reduction == "sum" else raw_foot_phase_mean
            sum_equivalent_weight = foot_phase_weight if foot_phase_reduction == "sum" else foot_phase_weight / 4.0
            scalar = raw_foot_phase_sum.new_tensor
            diagnostics["Diagnostics/foot_phase_weight"] = scalar(foot_phase_weight)
            diagnostics["Diagnostics/foot_phase_reduction"] = scalar(float(foot_phase_reduction == "mean"))
            diagnostics["Diagnostics/foot_phase_sum_equivalent_weight"] = scalar(sum_equivalent_weight)
            diagnostics["Diagnostics/foot_phase_per_foot_effective_weight"] = scalar(sum_equivalent_weight)
            diagnostics["Diagnostics/raw_foot_phase_sum"] = raw_foot_phase_sum.mean()
            diagnostics["Diagnostics/raw_foot_phase_mean"] = raw_foot_phase_mean.mean()
            diagnostics["Diagnostics/weighted_foot_phase"] = (-foot_phase_weight * reduced_foot_phase).mean()

        # Playback reads these pre-reset tensors to retain the exact action-to-joint mapping.
        # Keep the extra copies disabled during training to avoid unnecessary GPU work.
        if getattr(self, "_capture_rollout_diagnostics", False):
            self._running_reward_negative = reward_clipped.detach().clone()
            self._last_policy_actions = action.detach().clone()
            self._last_joint_position_targets = joint_pos_target.detach().clone()
            self._last_requested_joint_position_targets = requested_target.detach().clone()
            self._last_joint_positions = joint_pos.detach().clone()
            self._last_soft_joint_pos_limits = soft_joint_pos_limits.detach().clone()
            self._last_joint_velocities = robot.data.joint_vel.torch[:, joint_ids].detach().clone()
            self._last_joint_torques = robot.data.applied_torque.torch[:, joint_ids].detach().clone()
            self._last_root_positions_w = robot.data.root_pos_w.torch[:, :2].detach().clone()
            self._last_root_headings_w = robot.data.heading_w.torch.detach().clone()
            self._last_root_lin_velocities_b = robot.data.root_lin_vel_b.torch.detach().clone()
            self._last_root_ang_velocities_b = robot.data.root_ang_vel_b.torch.detach().clone()
            command_term = self.command_manager.get_term("base_velocity")
            self._last_base_velocity_commands = command_term.command.detach().clone()
            self._last_foot_thetas = command_term.foot_thetas.detach().clone()
            self._last_gait_periods = command_term.gait_periods.detach().clone()
            self._last_duty_factors = command_term.duty_factors.detach().clone()
            self._last_common_gait_phases = command_term.common_gait_phases().detach().clone()
            self._last_periodic_force_weights = command_term.periodic_force_weights().detach().clone()
            self._last_periodic_speed_weights = command_term.periodic_speed_weights().detach().clone()
            foot_body_ids = getattr(self, "_rollout_foot_body_ids", None)
            if foot_body_ids is not None:
                self._last_foot_velocities_w = robot.data.body_lin_vel_w.torch[:, foot_body_ids].detach().clone()
            foot_sensor_names = getattr(self, "_rollout_foot_sensor_names", ())
            if foot_sensor_names:
                foot_normal_forces_w = []
                foot_ground_reaction_forces_w = []
                ground_reaction_force_includes_friction = True
                ground_filtered_force_source_valid = True
                for sensor_name in foot_sensor_names:
                    sensor_data = self.scene.sensors[sensor_name].data
                    force_matrix_w = getattr(sensor_data, "force_matrix_w", None)
                    if force_matrix_w is None:
                        ground_filtered_force_source_valid = False
                        net_forces_w = sensor_data.net_forces_w
                        if net_forces_w is None:
                            raise RuntimeError(f"Foot contact sensor '{sensor_name}' does not expose net_forces_w.")
                        normal_force_w = net_forces_w.torch[:, 0]
                        ground_reaction_force_includes_friction = False
                    else:
                        normal_force_w = force_matrix_w.torch[:, 0].sum(dim=1)
                    friction_forces_w = getattr(sensor_data, "friction_forces_w", None)
                    if friction_forces_w is None:
                        ground_reaction_force_w = normal_force_w
                        ground_reaction_force_includes_friction = False
                    else:
                        ground_reaction_force_w = normal_force_w + friction_forces_w.torch[:, 0].sum(dim=1)
                    foot_normal_forces_w.append(normal_force_w)
                    foot_ground_reaction_forces_w.append(ground_reaction_force_w)
                self._last_foot_normal_forces_w = torch.stack(foot_normal_forces_w, dim=1).detach().clone()
                self._last_foot_ground_reaction_forces_w = (
                    torch.stack(foot_ground_reaction_forces_w, dim=1).detach().clone()
                )
                self._last_foot_normal_force_is_ground_filtered = torch.full(
                    (command_term.command.shape[0],),
                    ground_filtered_force_source_valid,
                    dtype=torch.bool,
                    device=command_term.command.device,
                )
                self._last_ground_reaction_force_includes_friction = ground_reaction_force_includes_friction

        reward_diagnostics = getattr(self, "_straight_line_motion_diagnostics", {})
        for name in (
            "forward_score",
            "lateral_velocity_score",
            "yaw_rate_score",
            "roll_score",
            "straight_score",
            "lateral_position_score",
            "heading_score",
            "pose_score",
            "posture_score",
            "support_loss",
            "reward",
        ):
            value = reward_diagnostics.get(name)
            if value is not None:
                diagnostics[f"Diagnostics/straight_line_{name}"] = value.mean()

        foot_clearance_diagnostics = getattr(self, "_foot_clearance_diagnostics", {})
        for name in ("mean_swing_height", "mean_target_height", "mean_shortfall", "penalty", "reward"):
            value = foot_clearance_diagnostics.get(name)
            if value is not None:
                diagnostics[f"Diagnostics/foot_clearance_{name}"] = value.mean()
        return diagnostics
