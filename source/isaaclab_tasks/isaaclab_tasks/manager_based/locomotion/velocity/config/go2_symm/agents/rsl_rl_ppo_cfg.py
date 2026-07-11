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
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm as go2_symm_mdp


@configclass
class UnitreeGo2SymmFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """RSL-RL PPO config for the flat Go2 symmetric migration task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        configure_symm_quadruped_ppo(
            self,
            experiment_name="unitree_go2_symm_flat",
            use_data_augmentation=False,
            data_augmentation_func=go2_symm_mdp.compute_time_reversal_states,
            value_loss_coeff=0.05,
        )
