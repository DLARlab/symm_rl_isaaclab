# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the X1 bend variant and isolation from the original task."""

import importlib
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.agents.rsl_rl_ppo_cfg import (
    DobotX1SymmFlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg,
    DobotX1SymmFlatEnvCfg_PLAY,
)


@pytest.mark.parametrize("play", [False, True])
def test_x1_go2bend_changes_only_pose_constraints_and_symmetry(play):
    module = importlib.import_module(
        "isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_go2bend_symm.flat_env_cfg"
    )
    original = (DobotX1SymmFlatEnvCfg_PLAY if play else DobotX1SymmFlatEnvCfg)()
    expected = original.to_dict()
    cfg = getattr(module, "DobotX1Go2BendSymmFlatEnvCfg" + ("_PLAY" if play else ""))()
    positions = cfg.scene.robot.init_state.joint_pos
    assert len(positions) == 12
    for leg in ("front_left", "front_right", "rear_left", "rear_right"):
        assert positions[f"joint_{leg}_abad"] == 0.0
        assert positions[f"joint_{leg}_thigh_pitch"] == 0.6983
        assert positions[f"joint_{leg}_calf_pitch"] == -1.2842
    assert cfg.actions.joint_pos.clip == {
        "joint_front_.*_calf_pitch": (-2.3, -0.2),
        "joint_rear_.*_calf_pitch": (-2.3, -0.2),
    }
    for term_name in ("front_calf_joint_limits", "rear_calf_joint_limits"):
        params = getattr(cfg.events, term_name).params
        assert params["lower_limit_distribution_params"] == (-2.3, -2.3)
        assert params["upper_limit_distribution_params"] == (-0.2, -0.2)
        prefix = "joint_front_" if term_name.startswith("front") else "joint_rear_"
        lower = params["lower_limit_distribution_params"][0]
        upper = params["upper_limit_distribution_params"][0]
        for joint, position in positions.items():
            if joint.startswith(prefix) and joint.endswith("_calf_pitch"):
                assert lower < position < upper
    term = cfg.rewards.leg_permutation_symmetry
    assert term.params["logical_joint_signs"] == ((1.0, 1.0, 1.0),) * 4
    joint_pos = torch.tensor([[positions[name] for name in cfg.actions.joint_pos.joint_names]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joint_pos)))},
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(foot_thetas=torch.zeros(1, 4))),
    )
    params = {**term.params, "joint_cfg": SimpleNamespace(joint_ids=list(range(12)))}
    assert term.func(env, **params).item() == pytest.approx(0.0)

    actual = cfg.to_dict()
    actual["scene"]["robot"]["init_state"]["joint_pos"] = expected["scene"]["robot"]["init_state"]["joint_pos"]
    actual["actions"]["joint_pos"]["clip"] = expected["actions"]["joint_pos"]["clip"]
    actual["events"]["rear_calf_joint_limits"] = expected["events"]["rear_calf_joint_limits"]
    actual["rewards"]["leg_permutation_symmetry"] = expected["rewards"]["leg_permutation_symmetry"]
    assert actual == expected
    assert original.to_dict() == expected
    assert original.scene.robot.init_state.joint_pos["joint_rear_left_calf_pitch"] == 1.2842
    assert DobotX1SymmFlatEnvCfg().scene.robot.init_state.joint_pos["joint_rear_left_calf_pitch"] == 1.2842


def test_x1_go2bend_registers_train_play_and_its_own_experiment():
    importlib.import_module("isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_go2bend_symm")
    for suffix, class_suffix in (("", ""), ("-Play", "_PLAY")):
        spec = gym.spec(f"Isaac-Velocity-Flat-Dobot-X1-Go2Bend-Symm{suffix}-v0")
        assert spec.kwargs["env_cfg_entry_point"].endswith(f":DobotX1Go2BendSymmFlatEnvCfg{class_suffix}")
        module_name, class_name = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
        cfg = getattr(importlib.import_module(module_name), class_name)()
        assert cfg.experiment_name == "dobot_x1_go2bend_symm_flat"
        original = DobotX1SymmFlatPPORunnerCfg().to_dict()
        actual = cfg.to_dict()
        actual["experiment_name"] = original["experiment_name"]
        assert actual == original
