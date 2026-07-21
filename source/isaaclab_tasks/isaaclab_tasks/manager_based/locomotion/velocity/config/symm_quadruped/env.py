# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared environment behavior for symmetric quadruped tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


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

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

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

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.extras.setdefault("log", {}).update(step_diagnostics)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _clamp_processed_joint_position_targets(self) -> None:
        """Clamp the processed joint-position action term to the robot soft limits."""
        joint_term = self.action_manager.get_term("joint_pos")
        joint_ids = joint_term._joint_ids
        soft_joint_position_limits = self.scene["robot"].data.soft_joint_pos_limits.torch[:, joint_ids]
        clamped_targets, clipped_fraction = _clamp_joint_position_targets(
            joint_term.processed_actions,
            soft_joint_position_limits,
        )
        joint_term.processed_actions.copy_(clamped_targets)
        self._joint_target_clipped_fraction = clipped_fraction.detach()

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

        # Playback reads these pre-reset tensors to retain the exact action-to-joint mapping.
        # Keep the extra copies disabled during training to avoid unnecessary GPU work.
        if getattr(self, "_capture_rollout_diagnostics", False):
            self._running_reward_negative = reward_clipped.detach().clone()
            self._last_policy_actions = action.detach().clone()
            self._last_joint_position_targets = joint_pos_target.detach().clone()
            self._last_joint_positions = joint_pos.detach().clone()
            self._last_soft_joint_pos_limits = soft_joint_pos_limits.detach().clone()
            self._last_joint_velocities = robot.data.joint_vel.torch[:, joint_ids].detach().clone()
            self._last_joint_torques = robot.data.applied_torque.torch[:, joint_ids].detach().clone()
            self._last_root_positions_w = robot.data.root_pos_w.torch[:, :2].detach().clone()
            self._last_root_lin_velocities_b = robot.data.root_lin_vel_b.torch.detach().clone()
            self._last_root_ang_velocities_b = robot.data.root_ang_vel_b.torch.detach().clone()
            command_term = self.command_manager.get_term("base_velocity")
            self._last_base_velocity_commands = command_term.command.detach().clone()
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
                for sensor_name in foot_sensor_names:
                    sensor_data = self.scene.sensors[sensor_name].data
                    force_matrix_w = getattr(sensor_data, "force_matrix_w", None)
                    if force_matrix_w is None:
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
