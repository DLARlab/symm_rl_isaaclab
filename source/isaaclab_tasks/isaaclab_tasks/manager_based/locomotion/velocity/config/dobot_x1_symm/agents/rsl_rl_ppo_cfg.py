# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.agents.rsl_rl_ppo_cfg import (
    configure_symm_quadruped_ppo,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import dobot_x1_symm as dobot_mdp


@configclass
class DobotX1SymmFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """RSL-RL PPO config for the flat Dobot X1 symmetric task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        configure_symm_quadruped_ppo(
            self,
            experiment_name="dobot_x1_symm_flat",
            use_data_augmentation=False,
            data_augmentation_func=dobot_mdp.compute_time_reversal_states,
            value_loss_coeff=0.05,
        )
