# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import dobot_x1_symm as dobot_mdp


@configclass
class DobotX1SymmFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """RSL-RL PPO config for the flat Dobot X1 symmetric task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.max_iterations = 5000
        self.save_interval = 1000
        self.experiment_name = "dobot_x1_symm_flat"
        self.clip_actions = None
        self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]
        self.algorithm.class_name = (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.legacy_trs_ppo:LegacyTimeReversalPPO"
        )
        self.algorithm.entropy_coef = 0.01
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=dobot_mdp.compute_time_reversal_states,
            mirror_loss_coeff=0.1,
            use_legacy_time_reversal_regularization=True,
            value_loss_coeff=0.05,
            min_abs_command_velocity=0.2,
            warmup_iterations=500,
            command_observation_index=3,
            command_observation_scale=2.0,
        )
