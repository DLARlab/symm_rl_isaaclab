# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""X1 configuration with matching front and rear leg bends."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg,
    DobotX1SymmFlatEnvCfg_PLAY,
)


def _configure_go2_bend(cfg: DobotX1SymmFlatEnvCfg) -> None:
    """Match rear-leg default angles, limits, and logical axes to the front legs."""
    positions = cfg.scene.robot.init_state.joint_pos.copy()
    for name, position in positions.items():
        if name.startswith("joint_rear_") and name.endswith(("_thigh_pitch", "_calf_pitch")):
            positions[name] = -position
    cfg.scene.robot.init_state.joint_pos = positions
    cfg.actions.joint_pos.clip = {**cfg.actions.joint_pos.clip, "joint_rear_.*_calf_pitch": (-2.3, -0.2)}
    cfg.events.rear_calf_joint_limits.params = {
        **cfg.events.rear_calf_joint_limits.params,
        "lower_limit_distribution_params": (-2.3, -2.3),
        "upper_limit_distribution_params": (-0.2, -0.2),
    }
    cfg.rewards.leg_permutation_symmetry.params = {
        **cfg.rewards.leg_permutation_symmetry.params,
        "logical_joint_signs": ((1.0, 1.0, 1.0),) * 4,
    }


@configclass
class DobotX1Go2BendSymmFlatEnvCfg(DobotX1SymmFlatEnvCfg):
    """X1 training task with hip/thigh/calf defaults of 0, 0.6983, and -1.2842 [rad]."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_go2_bend(self)


@configclass
class DobotX1Go2BendSymmFlatEnvCfg_PLAY(DobotX1SymmFlatEnvCfg_PLAY):
    """X1 playback task with hip/thigh/calf defaults of 0, 0.6983, and -1.2842 [rad]."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_go2_bend(self)
