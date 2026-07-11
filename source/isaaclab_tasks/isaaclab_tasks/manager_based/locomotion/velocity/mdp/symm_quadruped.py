# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared symmetric quadruped gait terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


SYMM_QUADRUPED_HIP_ACTION_IDS = [0, 3, 6, 9]
"""Default hip action indices for FL, FR, RL, and RR leg-major action order."""

SYMM_QUADRUPED_LEG_JOINT_IDS = {
    "FL": [0, 1, 2],
    "FR": [3, 4, 5],
    "RL": [6, 7, 8],
    "RR": [9, 10, 11],
}
"""Default leg joint indices for a 12-DoF quadruped in FL, FR, RL, RR order."""

SYMM_QUADRUPED_LEG_PHASE_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
"""Default mapping from logical leg tags to gait phase columns."""

SYMM_QUADRUPED_LOGICAL_JOINT_SIGNS = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)
"""Default per-leg signs for converting robot joints to the logical quadruped convention."""

SYMM_QUADRUPED_JOINT_RANGES = (
    96.0 * torch.pi / 180.0,
    290.0 * torch.pi / 180.0,
    108.0 * torch.pi / 180.0,
)
"""Default hip, thigh, and calf joint ranges [rad] used to normalize morphology errors."""

SYMM_QUADRUPED_LEG_PAIRS = (
    ("FL", "FR"),
    ("RL", "RR"),
    ("FL", "RL"),
    ("FR", "RR"),
    ("FL", "RR"),
    ("RL", "FR"),
)
"""Default leg pairs compared by the morphology symmetry reward."""


@configclass
class GaitVelocityCommandCfg(CommandTermCfg):
    """Velocity command with symmetric quadruped gait phase parameters."""

    class_type: type[GaitVelocityCommand] | str = (
        "isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped:GaitVelocityCommand"
    )

    asset_name: str = MISSING
    """Name of the robot asset for which commands are generated."""

    heading_command: bool = False
    """Whether to sample a heading target and convert it to yaw velocity."""

    heading_control_stiffness: float = 1.0
    """Scale factor for converting heading error to yaw velocity."""

    rel_standing_envs: float = 0.0
    """Probability of sampling a standing command."""

    rel_heading_envs: float = 1.0
    """Probability of using heading control when :attr:`heading_command` is enabled."""

    heading_ang_vel_clip: tuple[float, float] = (-1.0, 1.0)
    """Yaw velocity clip [rad/s] used for heading control."""

    min_xy_command_norm: float = 0.0
    """Minimum sampled XY command norm [m/s]; smaller commands are zeroed."""

    resample_once_after_reset: bool = False
    """Whether velocity commands resample only once after the episode reset."""

    @configclass
    class Ranges:
        """Velocity command ranges."""

        lin_vel_x: tuple[float, float] = MISSING
        """Linear x velocity range [m/s]."""

        lin_vel_y: tuple[float, float] = MISSING
        """Linear y velocity range [m/s]."""

        ang_vel_z: tuple[float, float] = MISSING
        """Yaw velocity range [rad/s]."""

        heading: tuple[float, float] | None = None
        """Heading range [rad]."""

    ranges: Ranges = MISSING
    """Distribution ranges for velocity commands."""

    vel_xy_success_threshold: float = 0.05
    """Per-episode near-zero XY velocity error success threshold [m/s]."""

    vel_xy_success_rel_threshold: float = 0.25
    """Per-episode relative XY velocity error success threshold."""

    vel_yaw_success_threshold: float = 0.05
    """Per-episode near-zero yaw velocity error success threshold [rad/s]."""

    vel_yaw_success_rel_threshold: float = 0.25
    """Per-episode relative yaw velocity error success threshold."""

    gait_period: float = 0.45
    """Fallback gait period [s]."""

    duty_factor: float = 0.45
    """Fallback stance phase ratio."""

    resampling_time_gait: float = 20.0
    """Time between gait parameter resampling [s]."""

    resample_gait_once_after_reset: bool = False
    """Whether gait parameters resample only once after the episode reset."""

    calculate_from_sampling_curve: bool = True
    """Whether to derive period and duty factor from commanded forward speed."""

    add_noise_period: bool = True
    """Whether to add command-speed-dependent noise to sampled gait period and duty factor."""

    add_noise_theta: bool = True
    """Whether to add small integer-cycle noise to sampled foot phase offsets."""

    noise_level_theta: int = 2
    """Integer noise range used for foot phase offsets."""

    noise_scale_theta: float = 0.001
    """Phase offset noise scale."""

    kappa: float = 16.0
    """Smoothness coefficient for stance/swing indicator shaping."""

    base_height_range: tuple[float, float] = (0.35, 0.45)
    """Nominal base height range [m] used by the old period/duty-factor curve."""

    init_foot_thetas: tuple[tuple[float, float, float, float], ...] = (
        (0.0, 0.5, 0.5, 0.0),
        (0.0, 0.0, 0.5, 0.5),
        (0.13, -0.13, 0.5, 0.5),
        (-0.13, 0.13, 0.5, 0.5),
        (-0.13, 0.13, 0.63, 0.37),
        (0.13, -0.13, 0.63, 0.37),
    )
    """Foot phase offsets for FL, FR, RL, and RR."""


class GaitVelocityCommand(CommandTerm):
    """Velocity command generator with sampled symmetric gait clocks."""

    cfg: GaitVelocityCommandCfg

    def __init__(self, cfg: GaitVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_threshold_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_threshold_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self._error_xy_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_yaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        self.init_foot_thetas = torch.tensor(cfg.init_foot_thetas, dtype=torch.float32, device=self.device)
        self.foot_thetas = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.duty_factors = torch.full((self.num_envs,), cfg.duty_factor, dtype=torch.float32, device=self.device)
        self.gait_periods = torch.full((self.num_envs,), cfg.gait_period, dtype=torch.float32, device=self.device)
        self.kappa = torch.full((self.num_envs,), cfg.kappa, dtype=torch.float32, device=self.device)
        self.gait_time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.gait_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["gait_period"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["duty_factor"] = torch.zeros(self.num_envs, device=self.device)

        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/velocity"
        self.cfg.element_names = self.cfg.element_names or ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity command in the base frame."""
        return self.vel_command_b

    def compute(self, dt: float):
        self._update_metrics()
        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            if self.cfg.resample_once_after_reset:
                allowed_env_ids = resample_env_ids[self.command_counter[resample_env_ids] <= 1]
                if len(allowed_env_ids) > 0:
                    self._resample(allowed_env_ids)
                exhausted_env_ids = resample_env_ids[self.command_counter[resample_env_ids] > 1]
                if len(exhausted_env_ids) > 0:
                    self.time_left[exhausted_env_ids] = torch.inf
            else:
                self._resample(resample_env_ids)
        self._update_command()

        if self.cfg.resampling_time_gait <= 0.0:
            return
        self.gait_time_left -= dt
        env_ids = (self.gait_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            if self.cfg.resample_gait_once_after_reset:
                allowed_env_ids = env_ids[self.gait_counter[env_ids] <= 1]
                if len(allowed_env_ids) > 0:
                    self._resample_gait(allowed_env_ids)
                exhausted_env_ids = env_ids[self.gait_counter[env_ids] > 1]
                if len(exhausted_env_ids) > 0:
                    self.gait_time_left[exhausted_env_ids] = torch.inf
            else:
                self._resample_gait(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_xy = self._error_xy_sum[env_ids] / denom
        mean_error_yaw = self._error_yaw_sum[env_ids] / denom
        command = self.vel_command_b[env_ids]
        xy_success_threshold = (
            self.cfg.vel_xy_success_threshold
            + self.cfg.vel_xy_success_rel_threshold * torch.linalg.norm(command[:, :2], dim=-1)
        )
        yaw_success_threshold = self.cfg.vel_yaw_success_threshold + self.cfg.vel_yaw_success_rel_threshold * torch.abs(
            command[:, 2]
        )
        self.metrics["error_vel_xy"][env_ids] = mean_error_xy
        self.metrics["error_vel_yaw"][env_ids] = mean_error_yaw
        self.metrics["success_threshold_vel_xy"][env_ids] = xy_success_threshold
        self.metrics["success_threshold_vel_yaw"][env_ids] = yaw_success_threshold
        self.metrics["success_rate"][env_ids] = (
            (mean_error_xy < xy_success_threshold) & (mean_error_yaw < yaw_success_threshold)
        ).float()

        extras = super().reset(env_ids)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        self._error_xy_sum[env_ids] = 0.0
        self._error_yaw_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        self.gait_counter[env_ids] = 0
        self._resample_gait(env_ids)
        return extras

    def foot_phases(self) -> torch.Tensor:
        """Foot phases for FL, FR, RL, and RR in cycle units."""
        phase_ratio = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt / self.gait_periods
        phase = _wrap_phase(phase_ratio.unsqueeze(-1) + self.foot_thetas)
        phase_tr = _wrap_phase(-(phase_ratio.unsqueeze(-1) + self.foot_thetas))
        return torch.where(self.command[:, 0:1] >= 0.0, phase, phase_tr)

    def phase_ratios(self) -> torch.Tensor:
        """Swing and stance phase ratios."""
        return torch.stack((1.0 - self.duty_factors, self.duty_factors), dim=-1)

    def periodic_force_weights(self) -> torch.Tensor:
        """Swing-force penalty weights for FL, FR, RL, and RR."""
        return _von_mises_periodic_property(
            self.foot_phases(), self.duty_factors, self.kappa, c_swing=-1.0, c_stance=0.0
        )

    def periodic_speed_weights(self) -> torch.Tensor:
        """Stance-speed penalty weights for FL, FR, RL, and RR."""
        return _von_mises_periodic_property(
            self.foot_phases(), self.duty_factors, self.kappa, c_swing=0.0, c_stance=-1.0
        )

    def _resample_gait(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids_tensor) == 0:
            return

        choices = torch.randint(
            low=0, high=self.init_foot_thetas.shape[0], size=(len(env_ids_tensor),), device=self.device
        )
        self.foot_thetas[env_ids_tensor] = self.init_foot_thetas[choices]

        if self.cfg.add_noise_theta:
            theta_noise = torch.randint(
                low=-self.cfg.noise_level_theta,
                high=self.cfg.noise_level_theta,
                size=(len(env_ids_tensor), 4),
                device=self.device,
            ).to(torch.float32)
            self.foot_thetas[env_ids_tensor] += theta_noise * self.cfg.noise_scale_theta

        if self.cfg.calculate_from_sampling_curve:
            cmd_x = self.command[env_ids_tensor, 0]
            self.gait_periods[env_ids_tensor] = self._compute_period_from_forward_velocity(cmd_x).clamp_min(0.1)
            self.duty_factors[env_ids_tensor] = self._compute_duty_factor_from_forward_velocity(cmd_x).clamp(
                min=0.1, max=0.9
            )
        else:
            self.gait_periods[env_ids_tensor] = self.cfg.gait_period
            self.duty_factors[env_ids_tensor] = self.cfg.duty_factor

        self.kappa[env_ids_tensor] = self.cfg.kappa
        self.gait_time_left[env_ids_tensor] = self.cfg.resampling_time_gait
        self.gait_counter[env_ids_tensor] += 1
        self.metrics["gait_period"][env_ids_tensor] = self.gait_periods[env_ids_tensor]
        self.metrics["duty_factor"][env_ids_tensor] = self.duty_factors[env_ids_tensor]

    def _update_metrics(self):
        error_xy = torch.linalg.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._error_xy_sum += error_xy
        self._error_yaw_sum += error_yaw
        self._step_count += 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.min_xy_command_norm > 0.0:
            command_norm = torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1)
            self.vel_command_b[env_ids, :2] *= (command_norm > self.cfg.min_xy_command_norm).unsqueeze(1)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w.torch[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.heading_ang_vel_clip[0],
                max=self.cfg.heading_ang_vel_clip[1],
            )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _compute_period_from_forward_velocity(self, cmd_forward_velocity: torch.Tensor) -> torch.Tensor:
        lower_bound, upper_bound = self.cfg.base_height_range
        length = torch.tensor((lower_bound + upper_bound) * 0.5, device=self.device, dtype=torch.float32)
        gravity = torch.tensor(9.81, device=self.device, dtype=torch.float32)
        velocity_star = torch.abs(cmd_forward_velocity) / torch.sqrt(gravity * length)
        random_scale = torch.rand_like(cmd_forward_velocity) * 2.0 - 1.0 if self.cfg.add_noise_period else 0.0
        period_star = 2.55 * torch.exp(-0.975 * velocity_star) * (1.0 + random_scale * velocity_star * 0.20)
        return period_star * torch.sqrt(length / gravity)

    def _compute_duty_factor_from_forward_velocity(self, cmd_forward_velocity: torch.Tensor) -> torch.Tensor:
        lower_bound, upper_bound = self.cfg.base_height_range
        length = torch.tensor((lower_bound + upper_bound) * 0.5, device=self.device, dtype=torch.float32)
        gravity = torch.tensor(9.81, device=self.device, dtype=torch.float32)
        velocity_star = torch.abs(cmd_forward_velocity) / torch.sqrt(gravity * length)
        random_scale = torch.rand_like(cmd_forward_velocity) * 2.0 - 1.0 if self.cfg.add_noise_period else 0.0
        return 0.5588 * torch.exp(-0.681 * velocity_star) * (1.0 + random_scale * velocity_star * 0.20)


def foot_phase_sin(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sine of the commanded foot phases."""
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    return torch.sin(2.0 * torch.pi * gait_command.foot_phases())


def foot_phase_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Cosine of the commanded foot phases."""
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    return torch.cos(2.0 * torch.pi * gait_command.foot_phases())


def foot_theta_sin(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sine of the sampled foot phase offsets."""
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    return torch.sin(2.0 * torch.pi * gait_command.foot_thetas)


def foot_theta_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Cosine of the sampled foot phase offsets."""
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    return torch.cos(2.0 * torch.pi * gait_command.foot_thetas)


def phase_ratios(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Swing and stance phase ratios."""
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    return gait_command.phase_ratios()


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant alive reward."""
    return torch.ones(env.num_envs, device=env.device)


def command_tracking_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Old shaped command tracking penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    x_error = torch.abs(command[:, 0] - asset.data.root_lin_vel_b.torch[:, 0])
    y_error = torch.abs(command[:, 1] - asset.data.root_lin_vel_b.torch[:, 1])
    yaw_error = torch.abs(command[:, 2] - asset.data.root_ang_vel_b.torch[:, 2])
    x_penalty = 1.0 - torch.exp(-2.0 * x_error)
    y_penalty = 1.0 - torch.exp(-10.0 * y_error)
    yaw_penalty = 1.0 - torch.exp(-5.0 * yaw_error)
    return -(x_penalty + y_penalty + yaw_penalty)


def base_height_range_penalty(
    env: ManagerBasedRLEnv,
    height_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for base height outside the configured range."""
    asset: Articulation = env.scene[asset_cfg.name]
    lower_bound, upper_bound = height_range
    deviation = torch.clamp(lower_bound - asset.data.root_pos_w.torch[:, 2], min=0.0)
    deviation += torch.clamp(asset.data.root_pos_w.torch[:, 2] - upper_bound, min=0.0)
    return -(1.0 - torch.exp(-5.0 * deviation))


def foot_periodicity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_cfg: SceneEntityCfg,
    foot_sensor_names: Sequence[str] | None = None,
    foot_sensor_body_names: Sequence[str] | None = None,
    force_scale: float = 0.001,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for foot contact in swing and foot speed in stance."""
    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    reward = torch.zeros(env.num_envs, device=env.device)
    if foot_sensor_names is not None:
        foot_forces = _collect_single_body_contact_force_norms(env, foot_sensor_names, foot_sensor_body_names)
        reward += torch.sum(
            (1.0 - torch.exp(-force_scale * foot_forces)) * gait_command.periodic_force_weights(), dim=-1
        )
    foot_speeds = torch.linalg.norm(asset.data.body_lin_vel_w.torch[:, feet_cfg.body_ids, :], dim=-1)
    reward += torch.sum((1.0 - torch.exp(-2.0 * foot_speeds)) * gait_command.periodic_speed_weights(), dim=-1)
    return reward


def contact_collision_count(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Count selected contact bodies whose force magnitude exceeds a threshold."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    contact_force_norms = torch.linalg.norm(contact_forces, dim=-1).max(dim=1)[0]
    return torch.sum((contact_force_norms > threshold).float(), dim=-1)


def illegal_contact_any_sensor(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any named single-body contact sensor exceeds the force threshold."""
    contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in sensor_names:
        contact_sensor = env.scene.sensors[sensor_name]
        contact_forces = contact_sensor.data.net_forces_w_history.torch
        force_norms = torch.linalg.norm(contact_forces, dim=-1).max(dim=1)[0]
        contact |= torch.any(force_norms > threshold, dim=1)
    return contact


def foot_clearance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_cfg: SceneEntityCfg,
    min_height: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for insufficient foot clearance during commanded swing phases."""
    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    foot_z = asset.data.body_pos_w.torch[:, feet_cfg.body_ids, 2]
    swing_ratio = (1.0 - gait_command.duty_factors).unsqueeze(-1).clamp_min(1.0e-6)
    swing_subphase = (gait_command.foot_phases() / swing_ratio).clamp(0.0, 1.0)
    mid_swing_weight = 0.5 + 0.5 * torch.sin(torch.pi * swing_subphase)
    shortfall = torch.relu(min_height - foot_z)
    clearance_penalty = 1.0 - torch.exp(-20.0 * shortfall * mid_swing_weight)

    command = env.command_manager.get_command(command_name)
    current_x_speed = torch.abs(asset.data.root_lin_vel_b.torch[:, 0])
    desired_x_speed = torch.abs(command[:, 0]).clamp_min(1.0e-3)
    velocity_coeff = torch.sigmoid(5.0 * (current_x_speed / (desired_x_speed * 0.5) - 1.0)).clamp(0.0, 1.0)
    return torch.sum(clearance_penalty * gait_command.periodic_force_weights(), dim=-1) * velocity_coeff


def hip_action_penalty(
    env: ManagerBasedRLEnv,
    tau: float = 0.5,
    hip_action_ids: Sequence[int] = SYMM_QUADRUPED_HIP_ACTION_IDS,
) -> torch.Tensor:
    """Penalty on large hip actions using the old softmax shaping."""
    hip_actions = env.action_manager.action[:, hip_action_ids]
    hip_abs = torch.abs(hip_actions)
    weights = torch.softmax(hip_abs / tau, dim=-1)
    hip_action_magnitude = torch.sum(weights * hip_abs, dim=-1)
    return -(1.0 - torch.exp(-0.5 * hip_action_magnitude))


def _base_stability_gate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.25,
    height_scale: float = 20.0,
    orientation_scale: float = 2.0,
) -> torch.Tensor:
    """Soft gate that activates penalties once the base is tall and upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w.torch[:, 2]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    height_gate = torch.sigmoid(height_scale * (base_height - height_threshold))
    upright_gate = torch.exp(-orientation_scale * (torch.square(roll) + torch.square(pitch)))
    return torch.clamp(height_gate * upright_gate, 0.0, 1.0)


def action_rate_exp_penalty(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    use_stability_gate: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on action changes using bounded exponential shaping.

    Args:
        env: The environment instance.
        scale: Exponential scale applied to the mean squared action difference.
        use_stability_gate: Whether to soften the penalty while the robot is unstable.
        asset_cfg: The robot asset used for the stability gate.

    Returns:
        The negative action-rate penalty.
    """
    action_diff = env.action_manager.action - env.action_manager.prev_action
    action_rate = torch.mean(torch.square(action_diff), dim=-1)
    penalty = 1.0 - torch.exp(-scale * action_rate)
    if use_stability_gate:
        penalty = penalty * _base_stability_gate(env, asset_cfg=asset_cfg)
    return -penalty


def morphological_symmetry_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_ids: dict[str, Sequence[int]] | None = None,
    leg_phase_index: dict[str, int] | None = None,
    logical_joint_signs: Sequence[Sequence[float]] = SYMM_QUADRUPED_LOGICAL_JOINT_SIGNS,
    joint_ranges: Sequence[float] = SYMM_QUADRUPED_JOINT_RANGES,
    leg_pairs: Sequence[tuple[str, str]] = SYMM_QUADRUPED_LEG_PAIRS,
) -> torch.Tensor:
    """Penalty for phase-weighted quadruped joint morphology symmetry.

    Args:
        env: The environment instance.
        command_name: Name of the gait command term.
        joint_cfg: Robot joints in FL, FR, RL, RR order.
        asset_cfg: Robot articulation.
        leg_joint_ids: Mapping from logical leg tag to joint indices within :paramref:`joint_cfg`.
        leg_phase_index: Mapping from logical leg tag to gait phase column.
        logical_joint_signs: Per-leg signs that map robot joints into a common logical convention.
        joint_ranges: Hip, thigh, and calf joint ranges [rad] used to normalize errors.
        leg_pairs: Logical leg pairs to compare.

    Returns:
        The negative morphology symmetry penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    joint_pos = asset.data.joint_pos.torch[:, joint_cfg.joint_ids]
    leg_joint_ids = leg_joint_ids or SYMM_QUADRUPED_LEG_JOINT_IDS
    leg_phase_index = leg_phase_index or SYMM_QUADRUPED_LEG_PHASE_INDEX
    logical_signs = torch.tensor(logical_joint_signs, dtype=torch.float32, device=env.device)
    joint_pos = joint_pos.reshape(joint_pos.shape[0], len(logical_joint_signs), -1) * logical_signs.unsqueeze(0)
    joint_pos = joint_pos.reshape(joint_pos.shape[0], -1)
    joint_range = torch.tensor(joint_ranges, dtype=torch.float32, device=env.device)

    def morph_sym_error(tag_a: str, tag_b: str) -> torch.Tensor:
        sign = torch.tensor([-1.0, 1.0, 1.0] if tag_a[-1] != tag_b[-1] else [1.0, 1.0, 1.0], device=env.device)
        phase_a = gait_command.foot_thetas[:, leg_phase_index[tag_a]]
        phase_b = gait_command.foot_thetas[:, leg_phase_index[tag_b]]
        phase_delta = torch.atan2(torch.sin(phase_a - phase_b), torch.cos(phase_a - phase_b))
        phase_weight = torch.exp(-((phase_delta / 0.25) ** 2)).unsqueeze(-1)
        joint_a = joint_pos[:, leg_joint_ids[tag_a]]
        joint_b = joint_pos[:, leg_joint_ids[tag_b]]
        error = torch.abs(joint_a - sign.unsqueeze(0) * joint_b) / (joint_range.unsqueeze(0) + 1.0e-6)
        error = error * phase_weight
        weights = torch.softmax(error / 0.5, dim=-1)
        return torch.sum(weights * error, dim=-1)

    error_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for tag_a, tag_b in leg_pairs:
        error_sum += morph_sym_error(tag_a, tag_b)
    return -(1.0 - torch.exp(-5.0 * error_sum / max(len(leg_pairs), 1)))


class SmoothnessPenalty(ManagerTermBase):
    """Torque-difference smoothness penalty."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self._last_torques = torch.zeros_like(asset.data.applied_torque.torch)
        self._has_last_torques = torch.zeros(
            self._last_torques.shape[0], dtype=torch.bool, device=self._last_torques.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._has_last_torques[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        scale: float = 0.1,
        use_stability_gate: bool = True,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        torques = asset.data.applied_torque.torch
        torque_diff = torch.sum(torch.abs(torques - self._last_torques), dim=-1)
        torque_diff = torch.where(self._has_last_torques, torque_diff, torch.zeros_like(torque_diff))
        self._last_torques[:] = torques
        self._has_last_torques[:] = True

        penalty = 1.0 - torch.exp(-scale * torque_diff)
        if use_stability_gate:
            penalty = penalty * _base_stability_gate(env, asset_cfg=asset_cfg)
        return -penalty


def base_height_out_of_range(
    env: ManagerBasedRLEnv,
    height_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base height is outside the configured range."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w.torch[:, 2]
    return torch.logical_or(base_height < height_range[0], base_height > height_range[1])


def base_roll_pitch_out_of_range(
    env: ManagerBasedRLEnv,
    max_roll: float,
    max_pitch: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base roll or pitch exceeds configured limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return torch.logical_or(torch.abs(roll) > max_roll, torch.abs(pitch) > max_pitch)


def body_height_below(
    env: ManagerBasedRLEnv,
    min_height: float,
    body_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any selected body falls below a world height."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.any(asset.data.body_pos_w.torch[:, body_cfg.body_ids, 2] < min_height, dim=-1)


def _collect_single_body_contact_force_norms(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    expected_body_names: Sequence[str] | None = None,
) -> torch.Tensor:
    if expected_body_names is not None and len(expected_body_names) != len(sensor_names):
        raise RuntimeError(f"Expected {len(sensor_names)} contact sensor body names, got {len(expected_body_names)}.")
    contact_forces = []
    for index, sensor_name in enumerate(sensor_names):
        sensor = env.scene.sensors[sensor_name]
        if sensor.num_sensors != 1:
            raise RuntimeError(
                f"Expected contact sensor '{sensor_name}' to resolve exactly one body, got {sensor.num_sensors}."
            )
        if expected_body_names is not None:
            expected_body_name = expected_body_names[index]
            body_names = getattr(sensor, "body_names", [])
            if body_names != [expected_body_name]:
                raise RuntimeError(
                    f"Expected contact sensor '{sensor_name}' to resolve body '{expected_body_name}', got {body_names}."
                )
        force_norm = torch.linalg.norm(sensor.data.net_forces_w_history.torch, dim=-1).max(dim=1)[0]
        contact_forces.append(force_norm[:, 0])
    return torch.stack(contact_forces, dim=-1)


@torch.no_grad()
def compute_time_reversal_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment states using the shared 60D quadruped time-reversal transform."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = time_reverse_observations(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        repeat_dims = (2,) + (1,) * (actions.ndim - 1)
        actions_aug = actions.repeat(repeat_dims)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = time_reverse_actions(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def time_reverse_observations(obs: torch.Tensor) -> torch.Tensor:
    """Time-reverse the shared 60D symmetric quadruped policy observation."""
    obs_tr = obs.clone()
    obs_tr[:, 3:6] = -obs[:, 3:6]
    obs_tr[:, 6:18] = obs[:, 6:18]
    obs_tr[:, 18:30] = -obs[:, 18:30]
    obs_tr[:, 30:42] = obs[:, 30:42]
    obs_tr[:, 42:46] = -obs[:, 42:46]
    obs_tr[:, 46:50] = obs[:, 46:50]
    obs_tr[:, 50:54] = -obs[:, 50:54]
    obs_tr[:, 54:58] = obs[:, 54:58]
    obs_tr[:, 58:60] = obs[:, 58:60]
    return obs_tr


def time_reverse_actions(actions: torch.Tensor) -> torch.Tensor:
    """Time-reverse the symmetric quadruped joint-position action offsets.

    The policy action is a target joint-position offset [m or rad, depending on joint type].
    Under time reversal, joint positions are even, so the transform is the identity.
    """
    return actions.clone()


def _wrap_phase(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 1.0)


def _smooth_swing_indicator(phi: torch.Tensor, duty_factor: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    swing_end = (1.0 - duty_factor).unsqueeze(-1)
    return torch.sigmoid((swing_end - phi) * kappa.unsqueeze(-1))


def _von_mises_periodic_property(
    phi: torch.Tensor,
    duty_factor: torch.Tensor,
    kappa: torch.Tensor,
    c_swing: float,
    c_stance: float,
) -> torch.Tensor:
    """Evaluate the old Von Mises periodic property function."""
    import numpy as np  # noqa: PLC0415
    from scipy.stats import vonmises_line  # noqa: PLC0415

    phi_np = phi.detach().cpu().numpy()
    duty_factor_np = duty_factor.detach().cpu().numpy()[..., None]
    kappa_np = kappa.detach().cpu().numpy()[..., None]

    swing_start = np.zeros_like(duty_factor_np)
    swing_end = 1.0 - duty_factor_np
    stance_start = swing_end
    stance_end = np.ones_like(duty_factor_np)

    swing_prob = _von_mises_phase_indicator(phi_np, swing_start, swing_end, kappa_np, vonmises_line)
    stance_prob = _von_mises_phase_indicator(phi_np, stance_start, stance_end, kappa_np, vonmises_line)
    values = c_swing * swing_prob + c_stance * stance_prob
    return torch.as_tensor(values, dtype=phi.dtype, device=phi.device)


def _von_mises_phase_indicator(
    phi,
    start,
    end,
    kappa,
    vonmises_line,
):
    import numpy as np  # noqa: PLC0415

    bounds = np.stack([start, end, start - 1.0, end - 1.0, start + 1.0, end + 1.0], axis=0)
    probs = vonmises_line.cdf(x=2.0 * np.pi * phi[None], loc=2.0 * np.pi * bounds, kappa=kappa[None])
    return probs[0] * (1.0 - probs[1]) + probs[2] * (1.0 - probs[3]) + probs[4] * (1.0 - probs[5])
