# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dobot X1 symmetric gait terms adapted from the Go2 symmetric task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm as _go2_symm

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


GaitVelocityCommand = _go2_symm.GaitVelocityCommand
GaitVelocityCommandCfg = _go2_symm.GaitVelocityCommandCfg
SmoothnessPenalty = _go2_symm.SmoothnessPenalty
action_rate_exp_penalty = _go2_symm.action_rate_exp_penalty
alive_bonus = _go2_symm.alive_bonus
base_height_out_of_range = _go2_symm.base_height_out_of_range
base_height_range_penalty = _go2_symm.base_height_range_penalty
base_roll_pitch_out_of_range = _go2_symm.base_roll_pitch_out_of_range
body_height_below = _go2_symm.body_height_below
command_tracking_penalty = _go2_symm.command_tracking_penalty
compute_time_reversal_states = _go2_symm.compute_time_reversal_states
foot_clearance_penalty = _go2_symm.foot_clearance_penalty
foot_periodicity_penalty = _go2_symm.foot_periodicity_penalty
foot_phase_cos = _go2_symm.foot_phase_cos
foot_phase_sin = _go2_symm.foot_phase_sin
foot_theta_cos = _go2_symm.foot_theta_cos
foot_theta_sin = _go2_symm.foot_theta_sin
hip_action_penalty = _go2_symm.hip_action_penalty
illegal_contact_any_sensor = _go2_symm.illegal_contact_any_sensor
phase_ratios = _go2_symm.phase_ratios
time_reverse_actions = _go2_symm.time_reverse_actions
time_reverse_observations = _go2_symm.time_reverse_observations

DOBOT_X1_SYMM_LEGGED_GYM_LEG_JOINT_IDS = {
    "FL": [0, 1, 2],
    "FR": [3, 4, 5],
    "RL": [6, 7, 8],
    "RR": [9, 10, 11],
}
DOBOT_X1_SYMM_LEG_PHASE_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
)
DOBOT_X1_SYMM_JOINT_RANGES = (1.3264, 5.236, 5.06)


def morphological_symmetry_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for Dobot joint symmetry in Go2 logical leg coordinates.

    Dobot rear thigh and calf default positions have the opposite sign from the front
    legs. This term maps the rear pitch joints into the Go2 logical convention before
    applying the same phase-weighted leg-pair comparisons as the Go2 symmetric task.

    Args:
        env: The environment instance.
        command_name: Name of the gait command term.
        joint_cfg: Robot joints in FL, FR, RL, RR order.
        asset_cfg: Robot articulation.

    Returns:
        The negative morphology symmetry penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gait_command: GaitVelocityCommand = env.command_manager.get_term(command_name)
    joint_pos = asset.data.joint_pos.torch[:, joint_cfg.joint_ids]

    logical_signs = torch.tensor(DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS, dtype=torch.float32, device=env.device)
    joint_pos = joint_pos.reshape(joint_pos.shape[0], 4, 3) * logical_signs.unsqueeze(0)
    joint_pos = joint_pos.reshape(joint_pos.shape[0], 12)
    joint_range = torch.tensor(DOBOT_X1_SYMM_JOINT_RANGES, dtype=torch.float32, device=env.device)

    def morph_sym_error(tag_a: str, tag_b: str) -> torch.Tensor:
        sign = torch.tensor([-1.0, 1.0, 1.0] if tag_a[-1] != tag_b[-1] else [1.0, 1.0, 1.0], device=env.device)
        phase_a = gait_command.foot_thetas[:, DOBOT_X1_SYMM_LEG_PHASE_INDEX[tag_a]]
        phase_b = gait_command.foot_thetas[:, DOBOT_X1_SYMM_LEG_PHASE_INDEX[tag_b]]
        phase_delta = torch.atan2(torch.sin(phase_a - phase_b), torch.cos(phase_a - phase_b))
        phase_weight = torch.exp(-((phase_delta / 0.25) ** 2)).unsqueeze(-1)
        joint_a = joint_pos[:, DOBOT_X1_SYMM_LEGGED_GYM_LEG_JOINT_IDS[tag_a]]
        joint_b = joint_pos[:, DOBOT_X1_SYMM_LEGGED_GYM_LEG_JOINT_IDS[tag_b]]
        error = torch.abs(joint_a - sign.unsqueeze(0) * joint_b) / (joint_range.unsqueeze(0) + 1.0e-6)
        error = error * phase_weight
        weights = torch.softmax(error / 0.5, dim=-1)
        return torch.sum(weights * error, dim=-1)

    error_sum = (
        morph_sym_error("FL", "FR")
        + morph_sym_error("RL", "RR")
        + morph_sym_error("FL", "RL")
        + morph_sym_error("FR", "RR")
        + morph_sym_error("FL", "RR")
        + morph_sym_error("RL", "FR")
    )
    return -(1.0 - torch.exp(-5.0 * error_sum / 6.0))
