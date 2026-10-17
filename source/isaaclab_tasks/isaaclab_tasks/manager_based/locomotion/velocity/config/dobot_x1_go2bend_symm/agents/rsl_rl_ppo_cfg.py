# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.agents.rsl_rl_ppo_cfg import (
    DobotX1SymmFlatPPORunnerCfg,
)


@configclass
class DobotX1Go2BendSymmFlatPPORunnerCfg(DobotX1SymmFlatPPORunnerCfg):
    """Reuse X1 PPO/TRS settings with a separate experiment directory."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "dobot_x1_go2bend_symm_flat"
