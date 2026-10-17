# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dobot X1 tasks with all four legs using the Go2-style bend."""

import gymnasium as gym

from . import agents

for task_suffix, config_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"Isaac-Velocity-Flat-Dobot-X1-Go2Bend-Symm{task_suffix}-v0",
        entry_point=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.env:DobotX1SymmManagerBasedRLEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DobotX1Go2BendSymmFlatEnvCfg{config_suffix}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DobotX1Go2BendSymmFlatPPORunnerCfg",
        },
    )
