# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared symmetric quadruped gait terms."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import MISSING, dataclass
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

SYMM_QUADRUPED_PHASE_MAPPING_VERSION = "same_gait_backward_duty_aware_trs_v2"
"""Audit identifier for the shared gait and time-reversal phase semantics."""

SYMM_QUADRUPED_POLICY_OBS_DIM = 72
"""Dimension of the shared symmetric quadruped policy observation."""


@dataclass(frozen=True)
class _SymmQuadrupedPolicyObservationLayout:
    """Slices for the shared symmetric quadruped policy observation."""

    measured_base_twist: slice
    projected_gravity: slice
    desired_base_twist: slice
    joint_position: slice
    joint_velocity: slice
    previous_action: slice
    foot_phase_sin: slice
    foot_phase_cos: slice
    foot_theta_sin: slice
    foot_theta_cos: slice
    phase_ratios: slice
    swing_ratio: slice
    stance_ratio: slice
    sagittal_plane_state: slice


SYMM_QUADRUPED_POLICY_OBS_LAYOUT = _SymmQuadrupedPolicyObservationLayout(
    measured_base_twist=slice(0, 6),
    projected_gravity=slice(6, 9),
    desired_base_twist=slice(9, 15),
    joint_position=slice(15, 27),
    joint_velocity=slice(27, 39),
    previous_action=slice(39, 51),
    foot_phase_sin=slice(51, 55),
    foot_phase_cos=slice(55, 59),
    foot_theta_sin=slice(59, 63),
    foot_theta_cos=slice(63, 67),
    phase_ratios=slice(67, 69),
    swing_ratio=slice(67, 68),
    stance_ratio=slice(68, 69),
    sagittal_plane_state=slice(69, SYMM_QUADRUPED_POLICY_OBS_DIM),
)
"""Named layout for the shared symmetric quadruped policy observation."""

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
"""Default leg pairs compared by the leg-permutation symmetry reward."""

_MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED = False


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

    phase_mapping_version: str = SYMM_QUADRUPED_PHASE_MAPPING_VERSION
    """Audit metadata identifying the gait and time-reversal phase semantics."""

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
        phase_ratio = (
            self._env.episode_length_buf.to(dtype=self.gait_periods.dtype) * self._env.step_dt / self.gait_periods
        )
        return compute_same_gait_foot_phases(phase_ratio.unsqueeze(-1), self.foot_thetas)

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


def desired_base_twist(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Expand the planar command into desired linear and angular velocities.

    The output order is ``(v_x, v_y, v_z, omega_x, omega_y, omega_z)``. The
    command generator only controls planar translation and yaw, so the other
    desired axes are explicitly zero.
    """
    command = env.command_manager.get_command(command_name)
    desired_twist = torch.zeros(command.shape[0], 6, device=command.device, dtype=command.dtype)
    desired_twist[:, :2] = command[:, :2]
    desired_twist[:, 5] = command[:, 2]
    return desired_twist


def sagittal_plane_state(
    env: ManagerBasedRLEnv,
    lateral_position_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return observable cross-track and heading error for straight-line recovery.

    The heading target is the environment's zero-yaw sagittal plane. Sine and
    cosine encode the wrapped heading error without a discontinuity at pi.
    """
    if lateral_position_scale <= 0.0:
        raise ValueError(f"Lateral-position scale must be positive, received {lateral_position_scale}.")

    asset: Articulation = env.scene[asset_cfg.name]
    lateral_position = asset.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]
    heading_error = math_utils.wrap_to_pi(asset.data.heading_w.torch)
    return torch.stack(
        (
            (lateral_position / lateral_position_scale).clamp(-1.0, 1.0),
            torch.sin(heading_error),
            torch.cos(heading_error),
        ),
        dim=-1,
    )


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


def straight_line_motion_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    forward_velocity_scale: float = 0.35,
    lateral_position_scale: float = 0.5,
    heading_scale: float = 0.5,
    lateral_velocity_scale: float = 0.25,
    yaw_rate_scale: float = 0.25,
    pose_weight: float = 0.1,
    roll_scale: float = 0.25,
    pitch_scale: float = 0.5,
    min_base_height: float = 0.35,
    height_scale: float = 0.1,
    support_loss_weight: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_weight: float = 1.0,
    straight_weight: float = 0.3,
    posture_weight: float = 0.15,
    lateral_position_deadband: float = 0.05,
    heading_deadband: float = 0.05,
) -> torch.Tensor:
    """Additively reward commanded x motion, straightness, and supported posture.

    Forward tracking is an independent term, so poor posture or temporary
    cross-track error cannot erase its learning signal. The observable lateral
    position and heading-error terms make returning to the sagittal plane
    preferable to walking straight along a displaced or rotated path.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term.
        forward_velocity_scale: Forward velocity-error scale [m/s].
        lateral_position_scale: Cross-track error scale [m].
        heading_scale: Heading-error scale [rad].
        lateral_velocity_scale: Lateral velocity-error scale [m/s].
        yaw_rate_scale: Yaw-rate error scale [rad/s].
        pose_weight: Weight of lateral-position and heading recovery.
        roll_scale: Straightness roll-error scale [rad].
        pitch_scale: Posture pitch scale and support corridor half-width [rad].
        min_base_height: Minimum supported base height [m].
        height_scale: Base height-shortfall scale [m].
        support_loss_weight: Weight of the bounded, forward-independent support loss.
        asset_cfg: Robot articulation configuration.
        forward_weight: Weight of forward command tracking.
        straight_weight: Weight of lateral velocity, yaw rate, and roll control.
        posture_weight: Weight of pitch and base-height posture.
        lateral_position_deadband: Unpenalized cross-track corridor half-width [m].
        heading_deadband: Unpenalized heading-error corridor half-width [rad].

    Returns:
        The bounded additive straight-line motion reward.
    """
    components = straight_line_motion_reward_components(
        env,
        command_name=command_name,
        forward_velocity_scale=forward_velocity_scale,
        lateral_position_scale=lateral_position_scale,
        heading_scale=heading_scale,
        lateral_velocity_scale=lateral_velocity_scale,
        yaw_rate_scale=yaw_rate_scale,
        roll_scale=roll_scale,
        pitch_scale=pitch_scale,
        min_base_height=min_base_height,
        height_scale=height_scale,
        asset_cfg=asset_cfg,
        lateral_position_deadband=lateral_position_deadband,
        heading_deadband=heading_deadband,
    )
    reward = forward_weight * components["forward_score"]
    reward += straight_weight * components["straight_score"]
    reward += pose_weight * components["pose_score"]
    reward += posture_weight * components["posture_score"]
    reward -= support_loss_weight * components["support_loss"]
    env._straight_line_motion_diagnostics = {
        name: value.detach() for name, value in (*components.items(), ("reward", reward))
    }
    return reward


def straight_line_motion_reward_components(
    env: ManagerBasedRLEnv,
    command_name: str,
    forward_velocity_scale: float = 0.35,
    lateral_position_scale: float = 0.5,
    heading_scale: float = 0.5,
    lateral_velocity_scale: float = 0.25,
    yaw_rate_scale: float = 0.25,
    roll_scale: float = 0.25,
    pitch_scale: float = 0.5,
    min_base_height: float = 0.35,
    height_scale: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_position_deadband: float = 0.05,
    heading_deadband: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Return the component tensors used by :func:`straight_line_motion_reward`.

    This helper keeps training and diagnostic calculations identical. The
    All returned values are bounded in ``[0, 1]``.
    """
    positive_scales = {
        "forward_velocity_scale": forward_velocity_scale,
        "lateral_position_scale": lateral_position_scale,
        "heading_scale": heading_scale,
        "lateral_velocity_scale": lateral_velocity_scale,
        "yaw_rate_scale": yaw_rate_scale,
        "roll_scale": roll_scale,
        "pitch_scale": pitch_scale,
        "height_scale": height_scale,
    }
    for name, value in positive_scales.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, received {value}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    forward_velocity_error = asset.data.root_lin_vel_b.torch[:, 0] - command[:, 0]
    forward_error = torch.square(forward_velocity_error / forward_velocity_scale)
    forward_score = torch.exp(-forward_error)

    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    lateral_velocity_error = asset.data.root_lin_vel_b.torch[:, 1] - command[:, 1]
    yaw_rate_error = asset.data.root_ang_vel_b.torch[:, 2] - command[:, 2]
    normalized_lateral_velocity = lateral_velocity_error / lateral_velocity_scale
    normalized_yaw_rate = yaw_rate_error / yaw_rate_scale
    normalized_roll = roll / roll_scale
    lateral_velocity_score = torch.exp(-torch.square(normalized_lateral_velocity))
    yaw_rate_score = torch.exp(-torch.square(normalized_yaw_rate))
    roll_score = torch.exp(-torch.square(normalized_roll))
    straight_error = torch.square(normalized_lateral_velocity)
    straight_error += torch.square(normalized_yaw_rate)
    straight_error += torch.square(normalized_roll)
    straight_score = torch.reciprocal(1.0 + straight_error)

    lateral_position = asset.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]
    lateral_position_excess = torch.relu(torch.abs(lateral_position) - lateral_position_deadband)
    heading_error = math_utils.wrap_to_pi(asset.data.heading_w.torch)
    heading_excess = torch.relu(torch.abs(heading_error) - heading_deadband)
    lateral_position_score = torch.exp(-torch.square(lateral_position_excess / lateral_position_scale))
    heading_score = torch.exp(-torch.square(heading_excess / heading_scale))
    pose_score = 0.5 * (lateral_position_score + heading_score)

    base_height = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    height_shortfall = torch.relu(min_base_height - base_height)
    posture_error = torch.square(pitch / pitch_scale)
    posture_error += torch.square(height_shortfall / height_scale)
    posture_score = torch.reciprocal(1.0 + posture_error)

    normalized_pitch_excess = torch.relu(torch.abs(pitch) - pitch_scale) / pitch_scale
    pitch_support_loss = -torch.expm1(-torch.square(normalized_pitch_excess))
    normalized_height_shortfall = height_shortfall / height_scale
    height_support_loss = -torch.expm1(-torch.square(normalized_height_shortfall))
    support_loss = 0.5 * (pitch_support_loss + height_support_loss)

    return {
        "forward_score": forward_score,
        "lateral_velocity_score": lateral_velocity_score,
        "yaw_rate_score": yaw_rate_score,
        "roll_score": roll_score,
        "straight_score": straight_score,
        "lateral_position_score": lateral_position_score,
        "heading_score": heading_score,
        "pose_score": pose_score,
        "posture_score": posture_score,
        "support_loss": support_loss,
    }


def sagittal_plane_penalty(
    env: ManagerBasedRLEnv,
    lateral_position_tolerance: float,
    heading_tolerance: float,
    lateral_velocity_tolerance: float,
    roll_tolerance: float,
    roll_rate_tolerance: float,
    yaw_rate_tolerance: float,
    secondary_weight: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_weight: float = 0.1,
    min_base_height: float = 0.25,
    height_tolerance: float = 0.1,
    low_height_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize low posture and motion outside a sagittal-plane corridor.

    The sagittal plane is defined by each environment origin's world x-z plane and
    a zero world heading. Forward/backward translation and pitch motion are not
    penalized. Each tolerance defines a zero-penalty corridor so normal gait sway
    is not discouraged. Absolute position and heading errors are weighted
    separately from motion errors because they are not policy observations.

    Args:
        env: The environment instance.
        lateral_position_tolerance: Lateral position normalization tolerance [m].
        heading_tolerance: Heading normalization tolerance [rad].
        lateral_velocity_tolerance: Lateral velocity normalization tolerance [m/s].
        roll_tolerance: Roll normalization tolerance [rad].
        roll_rate_tolerance: Roll-rate normalization tolerance [rad/s].
        yaw_rate_tolerance: Yaw-rate normalization tolerance [rad/s].
        secondary_weight: Weight applied to velocity, roll, and angular-rate errors.
        asset_cfg: Robot articulation configuration.
        pose_weight: Weight applied to absolute lateral-position and heading errors.
        min_base_height: Minimum base height before applying a low-posture penalty [m].
        height_tolerance: Low-height normalization tolerance [m].
        low_height_weight: Weight applied to the low-posture penalty.

    Returns:
        The negative bounded sagittal-plane and low-posture penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lateral_position_error = asset.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    heading_error = math_utils.wrap_to_pi(asset.data.heading_w.torch)
    lateral_velocity = asset.data.root_lin_vel_b.torch[:, 1]
    roll_rate = asset.data.root_ang_vel_b.torch[:, 0]
    yaw_rate = asset.data.root_ang_vel_b.torch[:, 2]

    normalized_errors = torch.stack(
        (
            lateral_position_error / lateral_position_tolerance,
            heading_error / heading_tolerance,
            lateral_velocity / lateral_velocity_tolerance,
            roll / roll_tolerance,
            roll_rate / roll_rate_tolerance,
            yaw_rate / yaw_rate_tolerance,
        ),
        dim=-1,
    )
    normalized_excess = torch.relu(torch.abs(normalized_errors) - 1.0)
    bounded_penalties = -torch.expm1(-torch.square(normalized_excess))
    pose_penalty = bounded_penalties[:, :2].mean(dim=-1)
    motion_penalty = bounded_penalties[:, 2:].mean(dim=-1)

    base_height = asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    normalized_height_shortfall = torch.relu((min_base_height - base_height) / height_tolerance)
    low_height_penalty = -torch.expm1(-torch.square(normalized_height_shortfall))
    return -(pose_weight * pose_penalty + secondary_weight * motion_penalty + low_height_weight * low_height_penalty)


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
    min_height: float = 0.08,
    height_scale: float = 0.05,
    min_command_speed: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet below a ground-relative swing-height trajectory.

    Args:
        env: The environment instance.
        command_name: Name of the gait velocity command term.
        feet_cfg: Foot body configuration in FL, FR, RL, and RR order.
        min_height: Peak commanded foot-link height above flat ground [m].
        height_scale: Clearance-shortfall shaping scale [m].
        min_command_speed: Forward command magnitude that fully enables the penalty [m/s].
        asset_cfg: Robot articulation configuration.

    Returns:
        The negative bounded swing-clearance penalty.
    """
    if min_height < 0.0:
        raise ValueError(f"Foot-clearance peak height must be non-negative, received {min_height}.")
    if height_scale <= 0.0:
        raise ValueError(f"Foot-clearance height scale must be positive, received {height_scale}.")
    if min_command_speed <= 0.0:
        raise ValueError(f"Foot-clearance command scale must be positive, received {min_command_speed}.")

    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)

    foot_height = asset.data.body_pos_w.torch[:, feet_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    swing_ratio = (1.0 - gait_command.duty_factors).unsqueeze(-1).clamp_min(1.0e-6)
    swing_subphase = (gait_command.foot_phases() / swing_ratio).clamp(0.0, 1.0)
    target_height = min_height * torch.sin(torch.pi * swing_subphase)
    swing_weight = _smooth_swing_indicator(gait_command.foot_phases(), gait_command.duty_factors, gait_command.kappa)
    shortfall = torch.relu(target_height - foot_height)
    clearance_penalty = -torch.expm1(-shortfall / height_scale)

    command = env.command_manager.get_command(command_name)
    command_gate = (torch.abs(command[:, 0]) / min_command_speed).clamp(0.0, 1.0).unsqueeze(-1)
    active_weight = swing_weight * command_gate
    weighted_penalty = clearance_penalty * active_weight
    penalty = -torch.sum(weighted_penalty, dim=-1)

    active_weight_sum = active_weight.sum(dim=-1).clamp_min(1.0e-6)
    env._foot_clearance_diagnostics = {
        "foot_height": foot_height.detach(),
        "target_height": target_height.detach(),
        "shortfall": shortfall.detach(),
        "swing_weight": active_weight.detach(),
        "mean_swing_height": ((foot_height * active_weight).sum(dim=-1) / active_weight_sum).detach(),
        "mean_target_height": ((target_height * active_weight).sum(dim=-1) / active_weight_sum).detach(),
        "mean_shortfall": ((shortfall * active_weight).sum(dim=-1) / active_weight_sum).detach(),
        "penalty": penalty.detach(),
    }
    return penalty


def foot_clearance_current_speed_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_cfg: SceneEntityCfg,
    min_height: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low swing feet only after the robot starts tracking its forward command.

    This retains the conservative clearance shaping used by the stable X1 training setup while making foot height
    ground-relative.

    Args:
        env: The environment instance.
        command_name: Name of the gait velocity command term.
        feet_cfg: Foot body configuration in FL, FR, RL, and RR order.
        min_height: Minimum swing foot-link height above flat ground [m].
        asset_cfg: Robot articulation configuration.

    Returns:
        The negative swing-clearance penalty.
    """
    if min_height < 0.0:
        raise ValueError(f"Foot-clearance minimum height must be non-negative, received {min_height}.")

    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    foot_height = asset.data.body_pos_w.torch[:, feet_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    swing_ratio = (1.0 - gait_command.duty_factors).unsqueeze(-1).clamp_min(1.0e-6)
    swing_subphase = (gait_command.foot_phases() / swing_ratio).clamp(0.0, 1.0)
    mid_swing_weight = 0.5 + 0.5 * torch.sin(torch.pi * swing_subphase)
    shortfall = torch.relu(min_height - foot_height)
    clearance_penalty = 1.0 - torch.exp(-20.0 * shortfall * mid_swing_weight)

    command = env.command_manager.get_command(command_name)
    current_x_speed = torch.abs(asset.data.root_lin_vel_b.torch[:, 0])
    desired_x_speed = torch.abs(command[:, 0]).clamp_min(1.0e-3)
    velocity_gate = torch.sigmoid(5.0 * (current_x_speed / (desired_x_speed * 0.5) - 1.0)).clamp(0.0, 1.0)
    swing_weight = -gait_command.periodic_force_weights()
    active_weight = swing_weight * velocity_gate.unsqueeze(-1)
    penalty = -torch.sum(clearance_penalty * active_weight, dim=-1)

    active_weight_sum = active_weight.sum(dim=-1).clamp_min(1.0e-6)
    target_height = torch.full_like(foot_height, min_height)
    env._foot_clearance_diagnostics = {
        "foot_height": foot_height.detach(),
        "target_height": target_height.detach(),
        "shortfall": shortfall.detach(),
        "swing_weight": active_weight.detach(),
        "mean_swing_height": ((foot_height * active_weight).sum(dim=-1) / active_weight_sum).detach(),
        "mean_target_height": ((target_height * active_weight).sum(dim=-1) / active_weight_sum).detach(),
        "mean_shortfall": ((shortfall * active_weight).sum(dim=-1) / active_weight_sum).detach(),
        "penalty": penalty.detach(),
    }
    return penalty


def foot_clearance_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_cfg: SceneEntityCfg,
    foot_sensor_names: Sequence[str],
    foot_sensor_body_names: Sequence[str] | None = None,
    target_height: float = 0.10,
    height_scale: float = 0.03,
    excess_height_margin: float = 0.03,
    excess_height_scale: float = 0.03,
    min_command_speed: float = 0.20,
    force_scale: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward every commanded swing foot for bounded height and no contact.

    A weighted harmonic mean aggregates the per-foot scores, preventing three
    well-behaved legs from hiding one dragging or overlifting leg.

    Args:
        env: The environment instance.
        command_name: Name of the gait velocity command term.
        feet_cfg: Foot body configuration in FL, FR, RL, and RR order.
        foot_sensor_names: Single-body foot contact sensor names in the same order as :paramref:`feet_cfg`.
        foot_sensor_body_names: Expected body name for each contact sensor.
        target_height: Peak commanded foot-link height above flat ground [m].
        height_scale: Clearance-shortfall shaping scale [m].
        excess_height_margin: Allowed height above the swing target [m].
        excess_height_scale: Overlift shaping scale [m].
        min_command_speed: Forward command magnitude that fully enables the reward [m/s].
        force_scale: Contact-force shaping scale [1/N].
        asset_cfg: Robot articulation configuration.

    Returns:
        The bounded swing-clearance and no-contact reward.
    """
    if target_height < 0.0:
        raise ValueError(f"Foot-clearance target height must be non-negative, received {target_height}.")
    if height_scale <= 0.0:
        raise ValueError(f"Foot-clearance height scale must be positive, received {height_scale}.")
    if excess_height_margin < 0.0:
        raise ValueError(f"Foot-clearance excess margin must be non-negative, received {excess_height_margin}.")
    if excess_height_scale <= 0.0:
        raise ValueError(f"Foot-clearance excess scale must be positive, received {excess_height_scale}.")
    if min_command_speed <= 0.0:
        raise ValueError(f"Foot-clearance command scale must be positive, received {min_command_speed}.")
    if force_scale <= 0.0:
        raise ValueError(f"Foot-contact force scale must be positive, received {force_scale}.")

    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    foot_height = asset.data.body_pos_w.torch[:, feet_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    swing_ratio = (1.0 - gait_command.duty_factors).unsqueeze(-1).clamp_min(1.0e-6)
    swing_subphase = (gait_command.foot_phases() / swing_ratio).clamp(0.0, 1.0)
    target_height_profile = target_height * torch.sin(torch.pi * swing_subphase)
    shortfall = torch.relu(target_height_profile - foot_height)
    clearance_score = torch.exp(-shortfall / height_scale)
    excess_height = torch.relu(foot_height - target_height_profile - excess_height_margin)
    excess_height_score = torch.exp(-excess_height / excess_height_scale)
    contact_forces = _collect_single_body_contact_force_norms(
        env,
        foot_sensor_names,
        foot_sensor_body_names,
    )
    no_contact_score = torch.exp(-force_scale * contact_forces)

    swing_weight = _smooth_swing_indicator(gait_command.foot_phases(), gait_command.duty_factors, gait_command.kappa)
    command = env.command_manager.get_command(command_name)
    command_gate = (torch.abs(command[:, 0]) / min_command_speed).clamp(0.0, 1.0).unsqueeze(-1)
    active_weight = swing_weight * command_gate
    active_weight_sum = active_weight.sum(dim=-1)
    per_foot_score = clearance_score * excess_height_score * no_contact_score
    harmonic_denominator = torch.sum(active_weight / per_foot_score.clamp_min(1.0e-6), dim=-1)
    reward = torch.where(
        active_weight_sum > 1.0e-6,
        active_weight_sum / harmonic_denominator.clamp_min(1.0e-6),
        torch.zeros_like(active_weight_sum),
    )
    safe_active_weight_sum = active_weight_sum.clamp_min(1.0e-6)

    env._foot_clearance_diagnostics = {
        "foot_height": foot_height.detach(),
        "target_height": target_height_profile.detach(),
        "shortfall": shortfall.detach(),
        "excess_height": excess_height.detach(),
        "per_foot_score": per_foot_score.detach(),
        "swing_weight": active_weight.detach(),
        "contact_force": contact_forces.detach(),
        "mean_swing_height": ((foot_height * active_weight).sum(dim=-1) / safe_active_weight_sum).detach(),
        "mean_target_height": ((target_height_profile * active_weight).sum(dim=-1) / safe_active_weight_sum).detach(),
        "mean_shortfall": ((shortfall * active_weight).sum(dim=-1) / safe_active_weight_sum).detach(),
        "penalty": (-reward).detach(),
        "reward": reward.detach(),
    }
    return reward


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


def joint_position_target_limit_penalty(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
    margin_fraction: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize processed joint targets inside a margin near the soft limits.

    Targets are already clamped for simulator safety. This term supplies the
    missing policy-learning signal that makes repeatedly requesting those
    clamps costly.
    """
    if not 0.0 < margin_fraction < 0.5:
        raise ValueError(f"Joint-limit margin fraction must be in (0, 0.5), received {margin_fraction}.")

    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_term_name)
    soft_limits = asset.data.soft_joint_pos_limits.torch[:, action_term._joint_ids]
    lower_limits = soft_limits[..., 0]
    upper_limits = soft_limits[..., 1]
    limit_range = (upper_limits - lower_limits).clamp_min(torch.finfo(soft_limits.dtype).eps)
    normalized_target = (2.0 * (action_term.processed_actions - lower_limits) / limit_range - 1.0).abs()
    margin_start = 1.0 - 2.0 * margin_fraction
    normalized_margin_excess = ((normalized_target - margin_start) / (1.0 - margin_start)).clamp(0.0, 1.0)
    return -torch.square(normalized_margin_excess).mean(dim=-1)


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


def leg_permutation_symmetry_penalty(
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
    """Penalize phase-aligned joint differences under leg permutations.

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
        The negative leg-permutation symmetry penalty.
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

    def leg_permutation_error(tag_a: str, tag_b: str) -> torch.Tensor:
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
        error_sum += leg_permutation_error(tag_a, tag_b)
    return -(1.0 - torch.exp(-5.0 * error_sum / max(len(leg_pairs), 1)))


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
    """Call :func:`leg_permutation_symmetry_penalty` through its deprecated name.

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
        The negative leg-permutation symmetry penalty.
    """
    global _MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED
    if not _MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED:
        warnings.warn(
            "morphological_symmetry_penalty() is deprecated; use leg_permutation_symmetry_penalty().",
            DeprecationWarning,
            stacklevel=2,
        )
        _MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED = True
    return leg_permutation_symmetry_penalty(
        env,
        command_name=command_name,
        joint_cfg=joint_cfg,
        asset_cfg=asset_cfg,
        leg_joint_ids=leg_joint_ids,
        leg_phase_index=leg_phase_index,
        logical_joint_signs=logical_joint_signs,
        joint_ranges=joint_ranges,
        leg_pairs=leg_pairs,
    )


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


def body_local_point_height_below(
    env: ManagerBasedRLEnv,
    point_b: tuple[float, float, float],
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when a body-frame point falls below a ground-relative height.

    Args:
        env: The environment instance.
        point_b: Point position in the articulation root frame [m].
        min_height: Minimum point height above the environment origin [m].
        asset_cfg: Robot articulation configuration.

    Returns:
        A mask identifying environments whose point is below :paramref:`min_height`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_position = asset.data.root_pos_w.torch
    point_position_b = torch.tensor(point_b, device=root_position.device, dtype=root_position.dtype).expand_as(
        root_position
    )
    point_position_w = root_position + math_utils.quat_apply(asset.data.root_quat_w.torch, point_position_b)
    point_height = point_position_w[:, 2] - env.scene.env_origins[:, 2]
    return point_height < min_height


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
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Pair original samples with their shared quadruped time-reversal transforms.

    RSL-RL uses this callback to construct pairs for the auxiliary policy
    equivariance and value-consistency losses. When
    ``use_data_augmentation=False``, the transformed samples do not enter the
    PPO surrogate or PPO value-regression transition batch.

    Args:
        env: Environment passed by the RSL-RL symmetry callback API.
        obs: Original observation batch.
        actions: Original action batch.

    Returns:
        The original/transformed observation and action pairs.
    """
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


def time_reverse_phase_sin_cos(
    phase_sin: torch.Tensor,
    phase_cos: torch.Tensor,
    swing_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the duty-aware time-reversal reflection to phase features.

    The reflected phase is ``remainder(swing_ratio - phase, 1.0)``. Ratios are
    used as provided and are not silently clamped.

    Args:
        phase_sin: Phase sine features with four leg channels in the final dimension.
        phase_cos: Phase cosine features matching :paramref:`phase_sin`.
        swing_ratio: Swing phase ratio with a singleton final dimension broadcastable to the phase features.

    Returns:
        The reflected phase sine and cosine features.

    Raises:
        ValueError: If shapes, devices, or dtypes are incompatible.
    """
    if phase_sin.ndim < 1 or phase_sin.shape[-1] != 4:
        raise ValueError(
            f"Expected phase_sin to have four leg channels in its final dimension, got {tuple(phase_sin.shape)}."
        )
    if phase_cos.shape != phase_sin.shape:
        raise ValueError(
            f"Expected phase_cos to match phase_sin shape, got {tuple(phase_cos.shape)} and {tuple(phase_sin.shape)}."
        )
    if swing_ratio.ndim < 1 or swing_ratio.shape[-1] != 1:
        raise ValueError(
            f"Expected swing_ratio to have a singleton final dimension, got shape {tuple(swing_ratio.shape)}."
        )
    if phase_cos.device != phase_sin.device or swing_ratio.device != phase_sin.device:
        raise ValueError(
            "Expected phase_sin, phase_cos, and swing_ratio on the same device, "
            f"got {phase_sin.device}, {phase_cos.device}, and {swing_ratio.device}."
        )
    if phase_cos.dtype != phase_sin.dtype or swing_ratio.dtype != phase_sin.dtype:
        raise ValueError(
            "Expected phase_sin, phase_cos, and swing_ratio to have the same dtype, "
            f"got {phase_sin.dtype}, {phase_cos.dtype}, and {swing_ratio.dtype}."
        )
    if not torch.is_floating_point(phase_sin):
        raise ValueError(f"Expected floating-point phase features, got dtype {phase_sin.dtype}.")

    try:
        broadcast_shape = torch.broadcast_shapes(phase_sin.shape, swing_ratio.shape)
    except RuntimeError as error:
        raise ValueError(
            "Expected swing_ratio to be broadcastable to the phase features, "
            f"got {tuple(swing_ratio.shape)} and {tuple(phase_sin.shape)}."
        ) from error
    if broadcast_shape != phase_sin.shape:
        raise ValueError(
            "Expected swing_ratio to broadcast without expanding the phase feature shape, "
            f"got broadcast shape {tuple(broadcast_shape)} from {tuple(swing_ratio.shape)} "
            f"and {tuple(phase_sin.shape)}."
        )

    alpha = 2.0 * torch.pi * swing_ratio
    sin_alpha = torch.sin(alpha)
    cos_alpha = torch.cos(alpha)
    phase_sin_tr = sin_alpha * phase_cos - cos_alpha * phase_sin
    phase_cos_tr = cos_alpha * phase_cos + sin_alpha * phase_sin
    return phase_sin_tr, phase_cos_tr


def time_reverse_observations(obs: torch.Tensor) -> torch.Tensor:
    """Time-reverse the shared 72D symmetric quadruped policy observation."""
    if obs.ndim < 1 or obs.shape[-1] != SYMM_QUADRUPED_POLICY_OBS_DIM:
        raise ValueError(
            f"Expected a {SYMM_QUADRUPED_POLICY_OBS_DIM}D symmetric quadruped policy observation, "
            f"got shape {tuple(obs.shape)}."
        )

    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    phase_sin_tr, phase_cos_tr = time_reverse_phase_sin_cos(
        obs[..., layout.foot_phase_sin],
        obs[..., layout.foot_phase_cos],
        obs[..., layout.swing_ratio],
    )
    obs_tr = obs.clone()
    obs_tr[..., layout.measured_base_twist] = -obs[..., layout.measured_base_twist]
    obs_tr[..., layout.projected_gravity] = obs[..., layout.projected_gravity]
    obs_tr[..., layout.desired_base_twist] = -obs[..., layout.desired_base_twist]
    obs_tr[..., layout.joint_position] = obs[..., layout.joint_position]
    obs_tr[..., layout.joint_velocity] = -obs[..., layout.joint_velocity]
    obs_tr[..., layout.previous_action] = obs[..., layout.previous_action]
    obs_tr[..., layout.foot_phase_sin] = phase_sin_tr
    obs_tr[..., layout.foot_phase_cos] = phase_cos_tr
    obs_tr[..., layout.foot_theta_sin] = -obs[..., layout.foot_theta_sin]
    obs_tr[..., layout.foot_theta_cos] = obs[..., layout.foot_theta_cos]
    obs_tr[..., layout.phase_ratios] = obs[..., layout.phase_ratios]
    obs_tr[..., layout.sagittal_plane_state] = obs[..., layout.sagittal_plane_state]
    return obs_tr


def time_reverse_actions(actions: torch.Tensor) -> torch.Tensor:
    """Time-reverse the symmetric quadruped joint-position action offsets.

    The policy action is a target joint-position offset [m or rad, depending on joint type].
    Under time reversal, joint positions are even, so the transform is the identity.
    """
    return actions.clone()


def compute_same_gait_foot_phases(common_phase: torch.Tensor, foot_thetas: torch.Tensor) -> torch.Tensor:
    """Combine a forward-time common gait clock with fixed foot offsets.

    Args:
        common_phase: Common gait phase in cycle units.
        foot_thetas: Per-foot phase offsets in cycle units.

    Returns:
        Wrapped foot phases with the broadcast shape of the inputs.
    """
    return _wrap_phase(common_phase + foot_thetas)


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
