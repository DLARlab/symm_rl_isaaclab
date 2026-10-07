# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for evaluating archived 56D policy histories with the current tasks."""

from types import SimpleNamespace

import pytest
import torch
import yaml

from isaaclab.managers import ObservationTermCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.agents.rsl_rl_ppo_cfg import (
    DobotX1SymmFlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped._checkpoint_compat import (
    _configure_checkpoint_compatibility,
    _legacy_profile,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.observation_history import (
    FrameMajorObservationHistoryWrapper,
)

TERMS = (
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "foot_phase_sin",
    "foot_phase_cos",
    "foot_theta",
    "phase_ratios",
)


def _archive(tmp_path, history=20, input_dim=None):
    cfg = DobotX1SymmFlatEnvCfg_PLAY()
    agent = DobotX1SymmFlatPPORunnerCfg()
    joint_names = cfg.actions.joint_pos.joint_names
    policy = {name: {"func": name, "history_length": 0, "scale": None, "clip": None} for name in TERMS}
    policy["velocity_commands"]["scale"] = [2.0, 2.0, 0.25]
    policy["joint_vel"]["scale"] = 0.05
    for name in ("joint_pos", "joint_vel"):
        policy[name]["params"] = {"asset_cfg": {"joint_names": joint_names}}
    saved = {
        "observations": {"policy": policy},
        "policy_observation_history": {"history_length": history, "group_name": "policy"},
        "sim": {"dt": cfg.sim.dt},
        "decimation": cfg.decimation,
        "scene": {
            "robot": {
                "actuators": {name: {"damping": 1.2} for name in cfg.scene.robot.actuators},
                "init_state": {"joint_pos": cfg.scene.robot.init_state.joint_pos},
            }
        },
        "actions": {
            "joint_pos": {
                "joint_names": joint_names,
                "scale": 0.25,
                "offset": 0.0,
                "use_default_offset": True,
                "clip": {"joint_front_.*_calf_pitch": [-2.3, -0.2], "joint_rear_.*_calf_pitch": [0.2, 2.3]},
            }
        },
    }
    saved["events"] = {}
    for side, limits in (("front", (-2.3, -0.2)), ("rear", (0.2, 2.3))):
        saved["events"][f"{side}_calf_joint_limits"] = {
            "func": "isaaclab.envs.mdp.events:randomize_joint_parameters",
            "mode": "startup",
            "params": {
                "asset_cfg": {"joint_names": [f"joint_{side}_.*_calf_pitch"]},
                "lower_limit_distribution_params": [limits[0], limits[0]],
                "upper_limit_distribution_params": [limits[1], limits[1]],
                "operation": "abs",
                "distribution": "uniform",
            },
        }
    agent_saved = {
        name: {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
        }
        for name in ("actor", "critic")
    }
    agent_saved["actor"]["distribution_cfg"] = {"std_type": "scalar"}
    agent_saved["clip_actions"] = None
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))
    (params / "agent.yaml").write_text(yaml.safe_dump(agent_saved, sort_keys=False))
    checkpoint = tmp_path / "model_9999.pt"
    width = input_dim if input_dim is not None else 56 * history
    torch.save(
        {f"{name}_state_dict": {"mlp.0.weight": torch.zeros(512, width)} for name in ("actor", "critic")}, checkpoint
    )
    return cfg, agent, checkpoint


def _observation_env():
    data = SimpleNamespace(
        projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.1, 0.2, -0.9]])),
        joint_pos=SimpleNamespace(torch=torch.arange(12.0).unsqueeze(0)),
        default_joint_pos=SimpleNamespace(torch=torch.ones(1, 12)),
        joint_vel=SimpleNamespace(torch=torch.arange(12.0, 24.0).unsqueeze(0)),
        default_joint_vel=SimpleNamespace(torch=torch.zeros(1, 12)),
    )
    phases = torch.tensor([[0.0, 0.25, 0.5, 0.75]])
    term = SimpleNamespace(
        foot_thetas=torch.tensor([[-0.3, -0.1, 0.75, 1.25]]),
        foot_phases=lambda: phases,
        phase_ratios=lambda: torch.tensor([[0.4, 0.6]]),
    )
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        action_manager=SimpleNamespace(action=torch.arange(24.0, 36.0).unsqueeze(0)),
        command_manager=SimpleNamespace(
            get_command=lambda _: torch.tensor([[1.0, 2.0, 3.0]]),
            get_term=lambda _: term,
        ),
    )


@pytest.mark.parametrize("history", [1, 20])
def test_legacy_observation_values_order_and_history(tmp_path, history):
    cfg, agent, checkpoint = _archive(tmp_path, history=history)
    profile = _configure_checkpoint_compatibility(cfg, agent, checkpoint)
    terms = [
        (name, term) for name, term in vars(cfg.observations.policy).items() if isinstance(term, ObservationTermCfg)
    ]
    assert tuple(name for name, _ in terms) == TERMS
    env = _observation_env()
    values = []
    for _, term in terms:
        value = term.func(env, **term.params)
        if term.scale is not None:
            value = value * torch.as_tensor(term.scale)
        values.append(value)
    frame = torch.cat(values, dim=-1)
    expected = torch.cat(
        (
            torch.tensor([[0.1, 0.2, -0.9, 2.0, 4.0, 0.75]]),
            torch.arange(-1.0, 11.0).unsqueeze(0),
            (0.05 * torch.arange(12.0, 24.0)).unsqueeze(0),
            torch.arange(24.0, 36.0).unsqueeze(0),
            torch.tensor([[0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.7, -0.1, -0.25, 0.25, 0.4, 0.6]]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(frame, expected)
    manager = SimpleNamespace(
        group_obs_dim={"policy": (56,)}, compute=lambda **_: {"policy": frame}, reset=lambda **_: {}
    )
    wrapped = FrameMajorObservationHistoryWrapper(manager, num_envs=1, device="cpu", history_length=history)
    initial = wrapped.compute(update_history=True)["policy"]
    assert initial.shape == (1, 56 * history)
    assert torch.count_nonzero(initial[:, :-56]) == 0
    torch.testing.assert_close(initial[:, -56:], expected)
    wrapped.compute(update_history=True)
    wrapped.reset([0])
    torch.testing.assert_close(wrapped.compute()["policy"], initial)
    assert profile["input_dim"] == 56 * history
    assert agent.algorithm.symmetry_cfg is None


def test_legacy_restores_actuators_and_action_convention(tmp_path):
    cfg, agent, checkpoint = _archive(tmp_path)
    _configure_checkpoint_compatibility(cfg, agent, checkpoint)
    assert all(actuator.damping == 1.2 for actuator in cfg.scene.robot.actuators.values())
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.actions.joint_pos.use_default_offset
    assert cfg.actions.joint_pos.clip["joint_front_.*_calf_pitch"] == (-2.3, -0.2)
    assert cfg.actions.joint_pos.clip["joint_rear_.*_calf_pitch"] == (0.2, 2.3)
    assert cfg.events.front_calf_joint_limits.params["lower_limit_distribution_params"] == (-2.3, -2.3)
    assert cfg.events.rear_calf_joint_limits.params["upper_limit_distribution_params"] == (2.3, 2.3)
    for network in (agent.actor, agent.critic):
        assert network.hidden_dims == [512, 256, 128]
        assert network.activation == "elu"
        assert not network.obs_normalization
    assert agent.actor.distribution_cfg.std_type == "scalar"


def test_mismatched_checkpoint_is_rejected_before_mutating_task(tmp_path):
    cfg, agent, checkpoint = _archive(tmp_path, input_dim=72)
    original = cfg.observations.policy
    with pytest.raises(ValueError, match="1120D"):
        _configure_checkpoint_compatibility(cfg, agent, checkpoint)
    assert cfg.observations.policy is original


def test_unrecognized_checkpoint_keeps_task_and_runner(tmp_path):
    cfg, agent, checkpoint = _archive(tmp_path)
    path = checkpoint.parent / "params" / "env.yaml"
    saved = yaml.safe_load(path.read_text())
    del saved["observations"]["policy"]["foot_theta"]
    path.write_text(yaml.safe_dump(saved))
    original_policy = cfg.observations.policy
    original_history = cfg.policy_observation_history
    original_symmetry = agent.algorithm.symmetry_cfg
    assert _configure_checkpoint_compatibility(cfg, agent, checkpoint) is None
    assert cfg.observations.policy is original_policy
    assert agent.algorithm.symmetry_cfg is original_symmetry
    assert cfg.policy_observation_history is original_history


def test_unknown_legacy_order_is_not_silently_reinterpreted(tmp_path):
    _, _, checkpoint = _archive(tmp_path)
    path = checkpoint.parent / "params" / "env.yaml"
    saved = yaml.safe_load(path.read_text())
    policy = saved["observations"]["policy"]
    policy["joint_pos"] = policy.pop("joint_pos")
    path.write_text(yaml.safe_dump(saved, sort_keys=False))
    with pytest.raises(ValueError, match="observation order"):
        _legacy_profile(checkpoint)
