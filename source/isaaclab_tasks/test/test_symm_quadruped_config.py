# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared symmetric quadruped task configuration."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg,
    DobotX1SymmFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.flat_env_cfg import (
    UnitreeGo2SymmFlatEnvCfg,
    UnitreeGo2SymmFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped import env as symm_quadruped_env
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.agents.rsl_rl_ppo_cfg import (
    configure_symm_quadruped_ppo,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.flat_env_cfg import (
    SYMM_QUADRUPED_GROUND_COLLISION_PATH,
    SymmQuadrupedPhysicsCfg,
    SymmQuadrupedRewardsCfg,
    configure_domain_randomization,
    configure_flat_scene,
    configure_rewards,
    make_gait_velocity_command,
    make_single_body_contact_sensor,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import dobot_x1_symm, symm_quadruped


class _Scene(dict):
    pass


def _tensor_data(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def _make_clearance_gait_command(phases: torch.Tensor, duty_factor: float = 0.5) -> symm_quadruped.GaitVelocityCommand:
    command = symm_quadruped.GaitVelocityCommand.__new__(symm_quadruped.GaitVelocityCommand)
    command.gait_phase = torch.zeros(phases.shape[0], dtype=phases.dtype, device=phases.device)
    command._foot_phase_offsets = phases
    command.duty_factors = torch.full_like(command.gait_phase, duty_factor)
    command.kappa = torch.full_like(command.gait_phase, 16.0)
    return command


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("use_sensors", [False, True])
def test_foot_periodicity_diagnostics_match_reward_and_preserve_reward_time_inputs(capture, use_sensors):
    phases = torch.tensor([[0.1, 0.75, 0.2, 0.9]])
    duty_factors = torch.tensor([0.5])
    command = SimpleNamespace(
        foot_phases=lambda: phases,
        duty_factors=duty_factors,
        periodic_force_weights=lambda: torch.tensor([[-1.0, 0.0, -0.25, 0.0]]),
        periodic_speed_weights=lambda: torch.tensor([[0.0, -1.0, 0.0, -0.75]]),
    )
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(body_lin_vel_w=_tensor_data(torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.2, 0.0]] * 2])))
    )
    sensor_names = ("FL", "FR", "RL", "RR")
    scene.sensors = {
        name: SimpleNamespace(
            num_sensors=1,
            data=SimpleNamespace(
                net_forces_w_history=_tensor_data(
                    torch.tensor([[[[0.0, 0.0, 8.0]], [[0.0, 3.0, 4.0]], [[0.0, 0.0, 1.0]]]])
                )
            ),
        )
        for name in sensor_names
    }
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _: command),
        num_envs=1,
        device="cpu",
        _capture_rollout_diagnostics=capture,
    )
    reward = symm_quadruped.foot_periodicity_penalty(
        env,
        "base_velocity",
        SimpleNamespace(body_ids=[0, 1, 2, 3]),
        foot_sensor_names=sensor_names if use_sensors else None,
        force_scale=0.025,
    )
    expected_force = torch.tensor([[math.expm1(-0.2), 0.0, 0.25 * math.expm1(-0.2), 0.0]])
    if not use_sensors:
        expected_force.zero_()
    expected_speed = torch.tensor([[0.0, math.expm1(-0.4), 0.0, 0.75 * math.expm1(-0.4)]])
    torch.testing.assert_close(reward, (expected_force + expected_speed).sum(dim=-1))
    if not capture:
        assert not hasattr(env, "_foot_periodicity_diagnostics")
        return

    diagnostics = env._foot_periodicity_diagnostics
    torch.testing.assert_close(diagnostics["force_penalty"], expected_force)
    torch.testing.assert_close(diagnostics["speed_penalty"], expected_speed)
    torch.testing.assert_close(diagnostics["speeds"], torch.tensor([[5.0, 0.2, 5.0, 0.2]]))
    if use_sensors:
        torch.testing.assert_close(diagnostics["forces"], torch.full((1, 4), 8.0))
    else:
        assert torch.isnan(diagnostics["forces"]).all()
    phases.zero_()
    duty_factors.zero_()
    torch.testing.assert_close(diagnostics["phases"], torch.tensor([[0.1, 0.75, 0.2, 0.9]]))
    torch.testing.assert_close(diagnostics["swing_ratios"], torch.full((1, 4), 0.5))


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("velocity_first", [False, True])
@pytest.mark.parametrize("penalty", [False, True])
def test_split_periodicity_rewards_use_raw_magnitudes_and_preserve_diagnostics(capture, velocity_first, penalty):
    phases = torch.tensor([[0.1, 0.75, 0.2, 0.9]])
    duty_factors = torch.tensor([0.5])
    gait = SimpleNamespace(
        foot_phases=lambda: phases,
        duty_factors=duty_factors,
        periodic_force_weights=lambda: torch.tensor([[-1.0, 0.0, -0.25, 0.0]]),
        periodic_speed_weights=lambda: torch.tensor([[0.0, -1.0, 0.0, -0.75]]),
    )
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(body_lin_vel_w=_tensor_data(torch.tensor([[[0.0, 0.0, 5.0], [0.0, 0.0, 0.2]] * 2])))
    )
    scene.sensors = {
        name: SimpleNamespace(
            num_sensors=1,
            data=SimpleNamespace(
                net_forces_w_history=_tensor_data(torch.tensor([[[[0.0, 8.0, 0.0]], [[0.0, 0.0, 1.0]]]]))
            ),
        )
        for name in ("FL", "FR", "RL", "RR")
    }
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _: gait),
        _capture_rollout_diagnostics=capture,
        num_envs=1,
        device="cpu",
    )
    if penalty:
        rewards_cfg = UnitreeGo2SymmFlatEnvCfg().rewards
        force_func = rewards_cfg.foot_periodicity_force.func
        speed_func = rewards_cfg.foot_periodicity_velocity.func
        force_scale = rewards_cfg.foot_periodicity_force.params["scale"]
        speed_scale = rewards_cfg.foot_periodicity_velocity.params["scale"]
    else:
        force_func = symm_quadruped.foot_periodicity_force_exp
        speed_func = symm_quadruped.foot_periodicity_velocity_exp
        force_scale = speed_scale = 1.0
    calls = [
        ("force", lambda: force_func(env, "base_velocity", tuple(scene.sensors), scale=force_scale)),
        (
            "speed",
            lambda: speed_func(env, "base_velocity", SimpleNamespace(body_ids=[0, 1, 2, 3]), scale=speed_scale),
        ),
    ]
    if velocity_first:
        calls.reverse()
    rewards = {name: call() for name, call in calls}
    if penalty:
        # Fractional gates must multiply each exponential, and each foot must contribute independently.
        expected_force = torch.tensor([[math.expm1(-0.4), 0.0, 0.25 * math.expm1(-0.4), 0.0]])
        expected_speed = torch.tensor([[0.0, math.expm1(-0.4), 0.0, 0.75 * math.expm1(-0.4)]])
        torch.testing.assert_close(rewards["force"], expected_force.sum(dim=-1))
        torch.testing.assert_close(rewards["speed"], expected_speed.sum(dim=-1))
    else:
        assert rewards["force"].item() == pytest.approx(math.exp(-10.0))
        assert rewards["speed"].item() == pytest.approx(math.exp(-0.35))
    if capture:
        diagnostics = env._foot_periodicity_diagnostics
        if penalty:
            torch.testing.assert_close(diagnostics["force_penalty"], expected_force)
            torch.testing.assert_close(diagnostics["speed_penalty"], expected_speed)
            assert "force_cost" not in diagnostics
            assert "speed_cost" not in diagnostics
        else:
            torch.testing.assert_close(diagnostics["force_cost"], torch.tensor([[8.0, 0.0, 2.0, 0.0]]))
            torch.testing.assert_close(diagnostics["speed_cost"], torch.tensor([[0.0, 0.2, 0.0, 0.15]]))
        torch.testing.assert_close(diagnostics["forces"], torch.full((1, 4), 8.0))
        torch.testing.assert_close(diagnostics["speeds"], torch.tensor([[5.0, 0.2, 5.0, 0.2]]))
        phases.zero_()
        duty_factors.zero_()
        assert diagnostics["phases"][0, 0].item() == pytest.approx(0.1)
        torch.testing.assert_close(diagnostics["swing_ratios"], torch.full((1, 4), 0.5))
    else:
        assert not hasattr(env, "_foot_periodicity_diagnostics")


def test_configured_periodicity_penalties_are_zero_at_zero_error_and_bounded_together():
    rewards_cfg = UnitreeGo2SymmFlatEnvCfg().rewards
    swing_weights = torch.tensor([[0.0, 0.25, 0.75, 1.0]]).expand(2, -1)
    gait = SimpleNamespace(
        periodic_force_weights=lambda: -swing_weights,
        periodic_speed_weights=lambda: -(1.0 - swing_weights),
    )
    velocities = torch.zeros(2, 4, 3)
    velocities[1, :, 2] = 100.0
    scene = _Scene(robot=SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=_tensor_data(velocities))))
    scene.sensors = {
        name: SimpleNamespace(
            num_sensors=1,
            data=SimpleNamespace(
                net_forces_w_history=_tensor_data(torch.tensor([[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1000.0]]]]))
            ),
        )
        for name in ("FL", "FR", "RL", "RR")
    }
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda _: gait))
    force = rewards_cfg.foot_periodicity_force
    speed = rewards_cfg.foot_periodicity_velocity
    force_reward = force.weight * force.func(env, "base_velocity", tuple(scene.sensors), scale=force.params["scale"])
    speed_reward = speed.weight * speed.func(
        env, "base_velocity", SimpleNamespace(body_ids=[0, 1, 2, 3]), scale=speed.params["scale"]
    )
    assert force_reward.tolist() == pytest.approx([0.0, -0.6])
    assert speed_reward.tolist() == pytest.approx([0.0, -0.6])
    assert (force_reward + speed_reward).tolist() == pytest.approx([0.0, -1.2])


def test_positive_base_height_reward_uses_squared_ground_relative_error():
    scene = _Scene()
    scene.env_origins = torch.tensor([[0.0, 0.0, 2.0]] * 4)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, z] for z in (2.35, 2.25, 2.45, 2.55)])))
    )
    result = symm_quadruped.base_height_exp(SimpleNamespace(scene=scene))
    assert result.tolist() == pytest.approx([1.0, math.exp(-1.0), math.exp(-1.0), math.exp(-4.0)], rel=1e-5)


@pytest.mark.parametrize("yaw", [0.0, 1.3])
def test_positive_roll_pitch_reward_penalizes_both_angles_symmetrically(yaw):
    roll = torch.tensor([0.0, 0.1, -0.1, 0.0, 0.0, 0.1])
    pitch = torch.tensor([0.0, 0.0, 0.0, 0.1, -0.1, 0.1])
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_quat_w=_tensor_data(quat_from_euler_xyz(roll, pitch, torch.full_like(roll, yaw)))
                )
            )
        }
    )
    result = symm_quadruped.base_roll_pitch_exp(env)
    assert result.tolist() == pytest.approx([1.0] + [math.exp(-0.1)] * 4 + [math.exp(-0.2)])


def test_speed_weighted_clearance_tracks_both_sides_without_a_phase_or_command_gate():
    scene = _Scene()
    scene.env_origins = torch.tensor([[0.0, 0.0, 2.0]] * 6)
    positions = torch.full((6, 4, 3), 2.0)
    positions[:, :, 2] = 2.102
    positions[0, 0, 2] += 0.05
    positions[1, 0, 2] -= 0.05
    positions[2:4, :, 2] = 2.0
    positions[4, :2, 2] += 0.05
    velocities = torch.zeros(6, 4, 3)
    velocities[[0, 1, 4, 5], 0] = torch.tensor([3.0, 4.0, 12.0])
    velocities[4, 1] = torch.tensor([0.0, -5.0, 0.0])
    velocities[3, :, 2] = 12.0
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=_tensor_data(positions), body_lin_vel_w=_tensor_data(velocities))
    )
    env = SimpleNamespace(scene=scene)
    reward = symm_quadruped.foot_clearance_speed_exp(env, SimpleNamespace(body_ids=[0, 1, 2, 3]))
    assert reward.tolist() == pytest.approx([math.exp(-1.25), math.exp(-1.25), 1.0, 1.0, math.exp(-2.5), 1.0], rel=2e-5)
    torch.testing.assert_close(env._foot_clearance_diagnostics["target_height"], torch.full((6, 4), 0.102))
    torch.testing.assert_close(env._foot_clearance_diagnostics["reward"], reward)


@pytest.mark.parametrize(
    "cfg_cls, peak_height, foot_offset",
    [(UnitreeGo2SymmFlatEnvCfg, 0.082, 0.022), (DobotX1SymmFlatEnvCfg, 0.1075, 0.0275)],
)
def test_configured_clearance_tracks_both_sides_of_offset_trajectory_without_command_or_speed_gate(
    cfg_cls, peak_height, foot_offset
):
    term = cfg_cls().rewards.foot_clearance
    term.params["feet_cfg"].body_ids = [0, 1, 2, 3]
    scene = _Scene()
    scene.env_origins = torch.zeros(7, 3)
    scene.env_origins[1, 2] = 1.0
    positions = torch.zeros(7, 4, 3)
    positions[:, :, 2] = (
        torch.tensor(
            [
                peak_height - 0.025,
                peak_height - 0.025,
                peak_height,
                peak_height + 0.025,
                foot_offset,
                foot_offset,
                foot_offset,
            ]
        ).unsqueeze(-1)
        + scene.env_origins[:, 2:3]
    )
    velocities = torch.zeros_like(positions)
    velocities[1, :, 0] = 3.0
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=_tensor_data(positions), body_lin_vel_w=_tensor_data(velocities))
    )
    phases = torch.full((7, 4), 0.25)
    phases[4] = 0.0
    phases[5] = 0.5
    phases[6] = torch.tensor([0.0, 0.125, 0.25, 0.5])
    gait = _make_clearance_gait_command(phases)
    command = torch.zeros(7, 3)
    command[1, 0] = 1.0
    env = SimpleNamespace(
        scene=scene, command_manager=SimpleNamespace(get_term=lambda _: gait, get_command=lambda _: command)
    )
    reward = term.weight * term.func(env, **term.params)
    expected_mid_swing = 0.1 * 4.0 * math.expm1(-1.0)
    assert reward[:2].tolist() == pytest.approx([expected_mid_swing] * 2, rel=1e-5)
    assert reward[3].item() == pytest.approx(expected_mid_swing, rel=1e-5)
    torch.testing.assert_close(reward[[2, 4, 5]], torch.zeros(3))
    diagnostics = env._foot_clearance_diagnostics
    torch.testing.assert_close(diagnostics["target_height"][0], torch.full((4,), peak_height))
    torch.testing.assert_close(
        diagnostics["target_height"][6],
        torch.tensor([foot_offset, foot_offset + (peak_height - foot_offset) / 2.0, peak_height, foot_offset]),
        atol=1e-7,
        rtol=1e-5,
    )
    assert diagnostics["shortfall"][6, 2].item() == pytest.approx(peak_height - foot_offset)
    torch.testing.assert_close(diagnostics["height_error"][[0, 3]], torch.full((2, 4), 0.025))
    torch.testing.assert_close(diagnostics["shortfall"][3], torch.zeros(4))
    assert diagnostics["mean_height_error"][3].item() == pytest.approx(0.025)
    torch.testing.assert_close(diagnostics["penalty"] * term.weight, reward)


@pytest.mark.parametrize("peak_height, foot_offset", [(0.082, 0.022), (0.1075, 0.0275)])
def test_clearance_periodic_gate_and_penalty_are_continuous_across_phase_wrap(peak_height, foot_offset):
    phases = torch.tensor([1.0 - 1.0e-6, 0.0, 1.0e-6], dtype=torch.float64).unsqueeze(-1).expand(-1, 4)
    gait = _make_clearance_gait_command(phases, duty_factor=0.56)
    scene = _Scene()
    scene.env_origins = torch.zeros(3, 3, dtype=torch.float64)
    positions = torch.zeros(3, 4, 3, dtype=torch.float64)
    positions[..., 2] = foot_offset + 0.01
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(positions)))
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda _: gait))

    penalty = symm_quadruped.foot_clearance_penalty(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        min_height=peak_height,
        foot_height_offset=foot_offset,
        height_scale=0.025,
        use_absolute_error=True,
        use_command_gate=False,
    )

    torch.testing.assert_close(
        env._foot_clearance_diagnostics["swing_weight"],
        torch.full((3, 4), 0.5, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-5,
    )
    assert penalty.tolist() == pytest.approx([2.0 * math.expm1(-0.4)] * 3, abs=5.0e-5)


@pytest.mark.parametrize("peak_height, foot_offset", [(0.082, 0.022), (0.1075, 0.0275)])
def test_clearance_trajectory_preserves_peak_and_has_zero_endpoint_slope(peak_height, foot_offset):
    swing_ratio = 0.44
    increment = swing_ratio * 1.0e-4
    phases = (
        torch.tensor(
            [0.0, increment, swing_ratio / 4.0, swing_ratio / 2.0, swing_ratio - increment, swing_ratio, 0.75],
            dtype=torch.float64,
        )
        .unsqueeze(-1)
        .expand(-1, 4)
    )
    gait = _make_clearance_gait_command(phases, duty_factor=0.56)
    scene = _Scene()
    scene.env_origins = torch.zeros(7, 3, dtype=torch.float64)
    positions = torch.zeros(7, 4, 3, dtype=torch.float64)
    positions[..., 2] = foot_offset
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(positions)))
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda _: gait))

    symm_quadruped.foot_clearance_penalty(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        min_height=peak_height,
        foot_height_offset=foot_offset,
        use_command_gate=False,
    )

    target = env._foot_clearance_diagnostics["target_height"][:, 0]
    assert target[[0, 5, 6]].tolist() == pytest.approx([foot_offset] * 3)
    assert target[2].item() == pytest.approx(foot_offset + (peak_height - foot_offset) / 2.0)
    assert target[3].item() == pytest.approx(peak_height)
    assert abs((target[1] - target[0]).item() / increment) < 1.0e-3
    assert abs((target[5] - target[4]).item() / increment) < 1.0e-3


def test_training_physics_uses_proven_2048_env_pair_capacities():
    physics_cfg = SymmQuadrupedPhysicsCfg().physx

    assert physics_cfg.gpu_found_lost_pairs_capacity == 2**22
    assert physics_cfg.gpu_found_lost_aggregate_pairs_capacity == 2**27
    assert physics_cfg.gpu_total_aggregate_pairs_capacity == 2**22


def test_flat_scene_defaults_to_scalable_training_batch():
    scene = SimpleNamespace(
        num_envs=4096,
        terrain=SimpleNamespace(),
        sky_light=SimpleNamespace(),
    )
    env_cfg = SimpleNamespace(scene=scene)

    configure_flat_scene(env_cfg)

    assert env_cfg.scene.num_envs == 256


def test_single_body_contact_sensor_skips_unused_air_time_tracking():
    sensor_cfg = make_single_body_contact_sensor("{ENV_REGEX_NS}/Robot/foot")

    assert sensor_cfg.history_length == 3
    assert not sensor_cfg.track_air_time


def test_symm_quadruped_ppo_preserves_unclipped_actions():
    cfg = SimpleNamespace(
        actor=SimpleNamespace(hidden_dims=[], distribution_cfg=SimpleNamespace(init_std=1.0)),
        critic=SimpleNamespace(hidden_dims=[]),
        algorithm=SimpleNamespace(),
    )

    configure_symm_quadruped_ppo(
        cfg,
        experiment_name="test",
        data_augmentation_func=lambda **_: None,
        use_data_augmentation=False,
        value_loss_coeff=0.0,
    )

    assert cfg.clip_actions is None
    assert cfg.actor.distribution_cfg.init_std == 1.0
    assert cfg.algorithm.entropy_coef == 0.005
    assert cfg.algorithm.symmetry_cfg.command_observation_index == 3
    assert cfg.algorithm.symmetry_cfg.min_abs_command_velocity == 0.0
    assert cfg.algorithm.symmetry_cfg.warmup_iterations == 0


def test_running_reward_is_clipped_before_terminal_penalty_is_added():
    total_reward = torch.tensor([0.5, -0.2, -4.5])
    termination_reward = torch.tensor([0.0, 0.0, -4.0])

    reward = symm_quadruped_env._clip_reward_before_termination(
        total_reward,
        termination_reward,
    )

    assert torch.equal(reward, torch.tensor([0.5, 0.0, -4.0]))


def test_ji22_reward_exponentially_scales_positive_reward_before_termination():
    positive_reward = torch.tensor([0.02, 0.01, 0.02])
    negative_reward = torch.tensor([0.0, -0.02, -0.055])
    termination_reward = torch.tensor([0.0, 0.0, -4.0])

    reward = symm_quadruped_env._combine_ji22_reward(
        positive_reward,
        negative_reward,
        termination_reward,
        sigma=0.02,
    )

    expected_reward = positive_reward * torch.exp(negative_reward / 0.02) + termination_reward
    assert torch.allclose(reward, expected_reward)


@pytest.mark.parametrize(
    "reward_terms, terminated, expected_reward",
    [
        ([2.0, -1.0], False, 0.04 * math.exp(-1.0)),
        ([0.5, -0.5], False, 0.01 * math.exp(-0.5)),
        ([0.5, -1.0], False, 0.01 * math.exp(-1.0)),
        ([0.5, 0.0], True, -0.01),
        ([2.0, 0.0], True, 0.02),
    ],
)
def test_symmetric_environment_uses_ji22_and_applies_pending_commands_after_reward(
    reward_terms, terminated, expected_reward
):
    calls = []
    command_term = SimpleNamespace(command=torch.tensor([[1.0]]))

    def apply_pending_resampling():
        calls.append("apply_pending")
        command_term.command.fill_(2.0)

    command_term.apply_pending_resampling = apply_pending_resampling
    command_manager = SimpleNamespace(
        active_terms=["base_velocity"],
        compute=lambda **_: calls.append("prepare_command"),
        get_term=lambda _: command_term,
    )

    def compute_reward(**_):
        calls.append("reward")
        assert torch.equal(command_term.command, torch.tensor([[1.0]]))
        return torch.tensor([0.02 * (sum(reward_terms) - float(terminated))])

    def compute_observation(**_):
        calls.append("observation")
        assert torch.equal(command_term.command, torch.tensor([[2.0]]))
        return {"policy": torch.zeros(1, 1)}

    env = SimpleNamespace(
        device="cpu",
        action_manager=SimpleNamespace(
            process_action=lambda _action: None,
            apply_action=lambda: None,
        ),
        _clamp_processed_joint_position_targets=lambda: None,
        recorder_manager=SimpleNamespace(
            active_terms=[],
            record_pre_step=lambda: None,
            record_post_physics_decimation_step=lambda: None,
            record_pre_reset=lambda _: None,
            record_post_reset=lambda _: None,
        ),
        sim=SimpleNamespace(is_rendering=False, step=lambda **_: None),
        render_enabled=False,
        _reset_idx=lambda _: None,
        scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **_: None),
        _physics_handles_decimation=True,
        _sim_step_counter=0,
        cfg=SimpleNamespace(decimation=1, sim=SimpleNamespace(render_interval=1)),
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        common_step_counter=0,
        command_manager=command_manager,
        event_manager=SimpleNamespace(available_modes=[]),
        termination_manager=SimpleNamespace(
            compute=lambda: torch.tensor([terminated]),
            terminated=torch.tensor([terminated]),
            time_outs=torch.tensor([False]),
        ),
        reward_manager=SimpleNamespace(
            compute=compute_reward,
            get_term_cfg=lambda _: SimpleNamespace(weight=-1.0),
            _step_reward=torch.tensor([reward_terms]),
        ),
        step_dt=0.02,
        _ji22_nontermination_reward_indices=torch.tensor([0, 1]),
        _compute_step_diagnostics=lambda *_: {},
        observation_manager=SimpleNamespace(compute=compute_observation),
        extras={},
    )
    env._apply_pending_command_resampling = lambda: (
        symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv._apply_pending_command_resampling(env)
    )

    _, reward, _, _, _ = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.step(env, torch.zeros(1, 1))

    assert calls == ["prepare_command", "reward", "apply_pending", "observation"]
    assert torch.allclose(reward, torch.tensor([expected_reward]))
    assert env.extras["log"]["Diagnostics/ji22_positive_reward_mean"].item() == pytest.approx(0.02 * reward_terms[0])
    assert env.extras["log"]["Diagnostics/ji22_negative_reward_mean"].item() == pytest.approx(0.02 * reward_terms[1])
    assert env.extras["log"]["Diagnostics/ji22_multiplier_mean"].item() == pytest.approx(math.exp(reward_terms[1]))


@pytest.mark.parametrize("env_cfg_cls", [UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg])
def test_policy_observations_remove_constant_zero_dimensions(env_cfg_cls):
    env_cfg = env_cfg_cls()
    policy = env_cfg.observations.policy

    assert env_cfg.policy_observation_history.history_length == 20
    assert env_cfg.policy_observation_history.group_name == "policy"
    assert policy.base_lin_vel is None
    assert policy.base_ang_vel is None
    assert policy.velocity_commands.func is base_mdp.generated_commands
    assert policy.velocity_commands.scale == (2.0, 2.0, 0.25)
    assert policy.foot_theta.func is symm_quadruped.foot_theta
    assert getattr(policy, "foot_theta_sin", None) is None
    assert getattr(policy, "foot_theta_cos", None) is None
    assert policy.sagittal_plane_state is None
    observation_terms = [name for name, term in vars(policy).items() if hasattr(term, "func")]
    assert observation_terms == [
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
        "foot_phase_sin",
        "foot_phase_cos",
        "foot_theta",
        "phase_ratios",
    ]


def test_foot_theta_observation_uses_continuous_template_canonical_range():
    gait_command = SimpleNamespace(
        foot_thetas=torch.tensor(
            [
                [-0.13, 0.13, 0.50, 0.63],
                [0.87, -0.63, 0.75, -0.25],
            ]
        )
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: gait_command))

    theta = symm_quadruped.foot_theta(env, command_name="base_velocity")

    assert torch.allclose(
        theta,
        torch.tensor(
            [
                [-0.13, 0.13, 0.50, 0.63],
                [-0.13, 0.37, -0.25, -0.25],
            ]
        ),
    )
    assert torch.all(theta >= -0.25)
    assert torch.all(theta < 0.75)


def test_sagittal_plane_state_zero_preserves_three_zero_dimensions():
    env = SimpleNamespace(num_envs=2, device="cpu")

    state = symm_quadruped.sagittal_plane_state_zero(env)

    assert state.shape == (2, 3)
    assert torch.count_nonzero(state) == 0


def test_sagittal_plane_state_exposes_lateral_offset_and_wrapped_heading():
    scene = _Scene()
    scene.env_origins = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 1.25, 0.4], [0.0, -2.0, 0.4]])),
            heading_w=_tensor_data(torch.tensor([math.pi / 2.0, -math.pi / 2.0])),
        )
    )
    env = SimpleNamespace(scene=scene)

    state = symm_quadruped.sagittal_plane_state(env, lateral_position_scale=0.5)

    assert torch.allclose(
        state,
        torch.tensor([[0.5, 1.0, 0.0], [-1.0, -1.0, 0.0]]),
        atol=1.0e-6,
    )


def test_desired_base_twist_expands_planar_command_to_six_velocities():
    command = torch.tensor([[1.0, -0.2, 0.3], [-1.0, 0.4, -0.5]])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: command))

    desired_twist = symm_quadruped.desired_base_twist(env, command_name="base_velocity")

    assert torch.equal(
        desired_twist,
        torch.tensor([[1.0, -0.2, 0.0, 0.0, 0.0, 0.3], [-1.0, 0.4, 0.0, 0.0, 0.0, -0.5]]),
    )


def test_deprecated_morphological_symmetry_function_forwards_to_leg_permutation(monkeypatch):
    expected = torch.tensor([-0.25])
    captured = {}

    def fake_leg_permutation(env, **kwargs):
        captured["env"] = env
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(symm_quadruped, "leg_permutation_symmetry_penalty", fake_leg_permutation)
    monkeypatch.setattr(symm_quadruped, "_MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED", False)
    env = SimpleNamespace()
    joint_cfg = SimpleNamespace()

    with pytest.warns(DeprecationWarning, match="leg_permutation_symmetry_penalty"):
        result = symm_quadruped.morphological_symmetry_penalty(
            env,
            command_name="base_velocity",
            joint_cfg=joint_cfg,
            phase_sync_tolerance=0.01,
        )

    assert result is expected
    assert captured["env"] is env
    assert captured["command_name"] == "base_velocity"
    assert captured["joint_cfg"] is joint_cfg
    assert captured["phase_sync_tolerance"] == 0.01


def test_x1_leg_permutation_adapter_uses_x1_joint_convention(monkeypatch):
    expected = torch.tensor([-0.50])
    captured = {}

    def fake_leg_permutation(env, **kwargs):
        captured["env"] = env
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(symm_quadruped, "leg_permutation_symmetry_penalty", fake_leg_permutation)
    env = SimpleNamespace()
    joint_cfg = SimpleNamespace()

    result = dobot_x1_symm.leg_permutation_symmetry_penalty(
        env,
        command_name="base_velocity",
        joint_cfg=joint_cfg,
        phase_sync_tolerance=0.01,
    )

    assert result is expected
    assert captured["logical_joint_signs"] == dobot_x1_symm.DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS
    assert captured["joint_ranges"] == dobot_x1_symm.DOBOT_X1_SYMM_JOINT_RANGES
    assert captured["phase_sync_tolerance"] == 0.01


def test_deprecated_x1_morphological_symmetry_function_forwards(monkeypatch):
    expected = torch.tensor([-0.75])
    captured = {}

    def fake_leg_permutation(*_, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(dobot_x1_symm, "leg_permutation_symmetry_penalty", fake_leg_permutation)
    monkeypatch.setattr(dobot_x1_symm, "_MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED", False)

    with pytest.warns(DeprecationWarning, match="leg_permutation_symmetry_penalty"):
        result = dobot_x1_symm.morphological_symmetry_penalty(
            SimpleNamespace(),
            command_name="base_velocity",
            joint_cfg=SimpleNamespace(),
            phase_sync_tolerance=0.01,
        )

    assert result is expected
    assert captured["phase_sync_tolerance"] == 0.01


def test_leg_permutation_penalty_only_uses_synchronous_pairs():
    joint_pos = torch.zeros(2, 12)
    joint_pos[:, 6:9] = 1.0
    scene = _Scene()
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=_tensor_data(joint_pos)))
    gait_command = SimpleNamespace(
        foot_thetas=torch.tensor(
            [
                [0.00, 0.50, 0.00, 0.50],
                [0.00, 0.50, 0.03, 0.50],
            ]
        )
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _: gait_command),
    )

    penalty = symm_quadruped.leg_permutation_symmetry_penalty(
        env,
        command_name="base_velocity",
        joint_cfg=SimpleNamespace(joint_ids=list(range(12))),
        logical_joint_signs=((1.0, 1.0, 1.0),) * 4,
        joint_ranges=(1.0, 1.0, 1.0),
        leg_pairs=(("FL", "RL"),),
        phase_sync_tolerance=0.02,
    )

    assert penalty[0].item() < -0.99
    assert penalty[1].item() == 0.0


@pytest.mark.parametrize("phase_sync_tolerance", [-0.01, 0.51])
def test_leg_permutation_penalty_rejects_invalid_phase_tolerance(phase_sync_tolerance):
    joint_pos = torch.zeros(1, 12)
    scene = _Scene()
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=_tensor_data(joint_pos)))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(foot_thetas=torch.zeros(1, 4))),
    )

    with pytest.raises(ValueError, match="phase_sync_tolerance"):
        symm_quadruped.leg_permutation_symmetry_penalty(
            env,
            command_name="base_velocity",
            joint_cfg=SimpleNamespace(joint_ids=list(range(12))),
            phase_sync_tolerance=phase_sync_tolerance,
        )


def test_joint_position_targets_are_clamped_to_soft_limits():
    joint_targets = torch.tensor([[-2.0, 0.5, 3.0], [-0.5, 1.5, 1.0]])
    soft_limits = torch.tensor(
        [
            [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],
            [[-0.25, 0.25], [-2.0, 2.0], [0.5, 0.75]],
        ]
    )

    clamped_targets, clipped_fraction = symm_quadruped_env._clamp_joint_position_targets(
        joint_targets,
        soft_limits,
    )

    assert torch.equal(clamped_targets, torch.tensor([[-1.0, 0.5, 2.0], [-0.25, 1.5, 0.75]]))
    assert clipped_fraction.item() == pytest.approx(4.0 / 6.0)


def test_soft_joint_limit_diagnostics_detect_proximity_and_violation():
    joint_pos = torch.tensor([[0.0, 0.95, 1.10]])
    soft_limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]])

    near_limit, violation, normalized_max = symm_quadruped_env._soft_joint_limit_diagnostics(
        joint_pos,
        soft_limits,
    )

    assert near_limit.item() == pytest.approx(2.0 / 3.0)
    assert violation.item() == pytest.approx(1.0 / 3.0)
    assert normalized_max.item() == pytest.approx(1.10)


def test_step_diagnostics_capture_pre_reset_actions_targets_and_reward_components():
    env = SimpleNamespace(_capture_rollout_diagnostics=True, num_envs=3, device="cpu")
    joint_pos = torch.tensor([[0.0, 0.95], [0.0, 0.0], [0.0, 0.0]])
    soft_limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]] * 3)
    joint_target = torch.tensor([[0.0, 1.20], [0.0, 0.0], [0.0, 0.0]])
    joint_term = SimpleNamespace(_joint_ids=[0, 1], processed_actions=joint_target)
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=_tensor_data(joint_pos),
            joint_vel=_tensor_data(torch.zeros_like(joint_pos)),
            applied_torque=_tensor_data(torch.ones_like(joint_pos)),
            soft_joint_pos_limits=_tensor_data(soft_limits),
            root_pos_w=_tensor_data(torch.full((3, 3), 2.0)),
            root_lin_vel_b=_tensor_data(torch.full((3, 3), 3.0)),
            root_ang_vel_b=_tensor_data(torch.full((3, 3), 4.0)),
            body_lin_vel_w=_tensor_data(torch.full((3, 4, 3), 5.0)),
        )
    )
    foot_sensor_names = tuple(f"contact_{index}" for index in range(4))

    def make_foot_sensor_data(index: int) -> SimpleNamespace:
        normal_force = torch.tensor([0.0, 0.0, float(index + 1)]).reshape(1, 1, 1, 3).repeat(3, 1, 1, 1)
        friction_force = torch.tensor([0.5 * float(index + 1), 0.0, 0.0]).reshape(1, 1, 1, 3).repeat(3, 1, 1, 1)
        return SimpleNamespace(
            net_forces_w=_tensor_data(normal_force[:, :, 0]),
            force_matrix_w=_tensor_data(normal_force),
            friction_forces_w=_tensor_data(friction_force),
        )

    scene.sensors = {
        sensor_name: SimpleNamespace(data=make_foot_sensor_data(index))
        for index, sensor_name in enumerate(foot_sensor_names)
    }
    env.scene = scene
    env._rollout_foot_sensor_names = foot_sensor_names
    env._rollout_foot_body_ids = [0, 1, 2, 3]
    env.action_manager = SimpleNamespace(get_term=lambda _: joint_term)
    command_term = SimpleNamespace(
        command=torch.full((3, 3), 6.0),
        periodic_force_weights=lambda: torch.full((3, 4), 7.0),
        periodic_speed_weights=lambda: torch.full((3, 4), 8.0),
    )
    env.command_manager = SimpleNamespace(get_term=lambda _: command_term)
    env._straight_line_motion_diagnostics = {
        "forward_score": torch.tensor([1.0, 0.5, 0.0]),
        "straight_score": torch.tensor([1.0, 0.8, 0.6]),
        "posture_score": torch.tensor([1.0, 0.7, 0.4]),
        "support_loss": torch.tensor([0.0, 0.2, 0.5]),
        "reward": torch.tensor([1.0, 0.5, -0.1]),
    }
    env._foot_clearance_diagnostics = {
        "mean_height": torch.tensor([0.04, 0.05, 0.06]),
        "mean_swing_height": torch.tensor([0.04, 0.05, 0.06]),
        "mean_target_height": torch.tensor([0.08, 0.08, 0.08]),
        "mean_shortfall": torch.tensor([0.04, 0.03, 0.02]),
        "mean_height_error": torch.tensor([0.05, 0.04, 0.03]),
        "penalty": torch.tensor([-1.0, -0.5, 0.0]),
    }
    actions = torch.tensor([[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]])

    diagnostics = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv._compute_step_diagnostics(
        env,
        actions,
        torch.tensor([-0.2, 0.0, 0.1]),
    )

    assert diagnostics["Diagnostics/action_abs_mean"].item() == pytest.approx(0.5)
    assert diagnostics["Diagnostics/action_abs_max"].item() == pytest.approx(1.0)
    assert diagnostics["Diagnostics/reward_clipped_fraction"].item() == 0.0
    assert diagnostics["Diagnostics/reward_negative_fraction"].item() == pytest.approx(1.0 / 3.0)
    assert torch.equal(env._running_reward_negative, torch.tensor([True, False, False]))
    assert torch.equal(env._running_reward_clipped, torch.tensor([False, False, False]))
    assert diagnostics["Diagnostics/joint_near_limit_fraction"].item() == pytest.approx(1.0 / 6.0)
    assert diagnostics["Diagnostics/joint_target_limit_violation_fraction"].item() == pytest.approx(1.0 / 6.0)
    assert diagnostics["Diagnostics/straight_line_forward_score"].item() == pytest.approx(0.5)
    assert diagnostics["Diagnostics/foot_clearance_mean_swing_height"].item() == pytest.approx(0.05)
    assert diagnostics["Diagnostics/foot_clearance_mean_height"].item() == pytest.approx(0.05)
    assert diagnostics["Diagnostics/foot_clearance_mean_shortfall"].item() == pytest.approx(0.03)
    assert diagnostics["Diagnostics/foot_clearance_mean_height_error"].item() == pytest.approx(0.04)
    assert torch.equal(env._last_policy_actions, actions)
    assert torch.equal(env._last_joint_position_targets, joint_target)
    assert torch.equal(env._last_joint_velocities, torch.zeros_like(joint_pos))
    assert torch.equal(env._last_joint_torques, torch.ones_like(joint_pos))
    assert torch.equal(env._last_root_positions_w, torch.full((3, 2), 2.0))
    assert torch.equal(env._last_root_lin_velocities_b, torch.full((3, 3), 3.0))
    assert torch.equal(env._last_root_ang_velocities_b, torch.full((3, 3), 4.0))
    assert torch.equal(env._last_base_velocity_commands, torch.full((3, 3), 6.0))
    assert torch.equal(env._last_periodic_force_weights, torch.full((3, 4), 7.0))
    assert torch.equal(env._last_periodic_speed_weights, torch.full((3, 4), 8.0))
    assert torch.equal(env._last_foot_velocities_w, torch.full((3, 4, 3), 5.0))
    assert env._last_foot_normal_forces_w.shape == (3, 4, 3)
    assert torch.equal(env._last_foot_normal_forces_w[0, :, 2], torch.arange(1.0, 5.0))
    assert torch.equal(env._last_foot_ground_reaction_forces_w[0, :, 0], 0.5 * torch.arange(1.0, 5.0))
    assert env._last_ground_reaction_force_includes_friction


def test_gait_command_uses_21_by_1_by_21_bins_with_curriculum_enabled():
    command_cfg = make_gait_velocity_command(symm_quadruped)

    assert not command_cfg.heading_command
    assert command_cfg.ranges.lin_vel_x == (-4.0, 4.0)
    assert command_cfg.ranges.lin_vel_y == (-0.6, 0.6)
    assert command_cfg.ranges.ang_vel_z == (-4.0, 4.0)
    assert command_cfg.min_xy_command_norm == 0.1
    assert command_cfg.curriculum.enabled
    assert command_cfg.curriculum.initial_ranges.lin_vel_x == (-0.5, 0.5)
    assert command_cfg.curriculum.initial_ranges.lin_vel_y == (-0.6, 0.6)
    assert command_cfg.curriculum.initial_ranges.ang_vel_z == (-0.5, 0.5)
    assert command_cfg.curriculum.num_bins == (21, 1, 21)
    assert command_cfg.resampling_time_range == (10.0, 10.0)
    assert command_cfg.resampling_time_gait == 10.0
    assert command_cfg.resampling_transition_probabilities == pytest.approx((1.0 / 3.0,) * 3)
    assert not command_cfg.resample_once_after_reset
    assert not command_cfg.resample_gait_once_after_reset
    assert command_cfg.curriculum_tracking_lin_vel_sigma == 0.25
    assert command_cfg.curriculum_tracking_ang_vel_sigma == 0.25
    assert command_cfg.curriculum_tracking_lin_vel_threshold == 0.9
    assert command_cfg.curriculum_tracking_ang_vel_threshold == 0.9


def test_gait_command_curriculum_expands_successful_bins():
    command_cfg = make_gait_velocity_command(symm_quadruped)
    command_cfg.curriculum.enabled = True
    curriculum = symm_quadruped._VelocityCommandBinCurriculum(command_cfg, "cpu")
    initial_active_bins = curriculum.active_bin_count
    initial_max_command = curriculum.max_active_abs_command
    active_bin_ids = curriculum._weights.nonzero(as_tuple=False)
    boundary_bin_id = active_bin_ids[torch.argmax(active_bin_ids[:, 0])]

    curriculum.update(boundary_bin_id.unsqueeze(0), torch.tensor([True]))

    assert curriculum.active_bin_count > initial_active_bins
    bin_width = 8.0 / 21.0
    assert initial_max_command.tolist() == pytest.approx([4.0 / 7.0, 0.6, 4.0 / 7.0])
    assert curriculum.max_active_abs_command[0].item() == pytest.approx(initial_max_command[0].item() + bin_width)
    assert curriculum.max_active_abs_command[1].item() == pytest.approx(initial_max_command[1].item())
    assert curriculum.max_active_abs_command[2].item() == pytest.approx(initial_max_command[2].item() + bin_width)


def test_gait_command_curriculum_samples_only_active_bins():
    command_cfg = make_gait_velocity_command(symm_quadruped)
    command_cfg.curriculum.enabled = True
    curriculum = symm_quadruped._VelocityCommandBinCurriculum(command_cfg, "cpu")

    commands, _ = curriculum.sample(256)

    assert torch.all(commands[:, 0] >= -(4.0 / 7.0))
    assert torch.all(commands[:, 0] <= 4.0 / 7.0)
    assert torch.all(commands[:, 1] >= -0.6)
    assert torch.all(commands[:, 1] <= 0.6)
    assert torch.all(commands[:, 2] >= -(4.0 / 7.0))
    assert torch.all(commands[:, 2] <= 4.0 / 7.0)


def test_gait_command_disabled_curriculum_samples_all_bins_without_updating_weights():
    command_cfg = make_gait_velocity_command(symm_quadruped)
    command_cfg.curriculum.enabled = False
    curriculum = symm_quadruped._VelocityCommandBinCurriculum(command_cfg, "cpu")

    initial_weights = curriculum._weights.clone()
    commands, bin_ids = curriculum.sample(512)
    curriculum.update(bin_ids, torch.ones(len(bin_ids), dtype=torch.bool))

    assert curriculum.active_bin_count == 21 * 1 * 21
    assert curriculum.max_active_abs_command.tolist() == pytest.approx([4.0, 0.6, 4.0])
    assert torch.equal(curriculum._weights, initial_weights)
    assert torch.all(bin_ids[:, 0] >= 0)
    assert torch.all(bin_ids[:, 0] < 21)
    assert torch.all(bin_ids[:, 1] == 0)
    assert torch.all(bin_ids[:, 2] >= 0)
    assert torch.all(bin_ids[:, 2] < 21)
    assert torch.all(commands[:, 0] >= -4.0)
    assert torch.all(commands[:, 0] <= 4.0)
    assert torch.all(commands[:, 1] >= -0.6)
    assert torch.all(commands[:, 1] <= 0.6)
    assert torch.all(commands[:, 2] >= -4.0)
    assert torch.all(commands[:, 2] <= 4.0)


def test_gait_command_zeroes_forward_commands_at_threshold():
    command_cfg = make_gait_velocity_command(symm_quadruped)
    command_cfg.rel_standing_envs = -1.0
    sampled_commands = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1001, 0.0, 0.0]])
    command = SimpleNamespace(
        cfg=command_cfg,
        device="cpu",
        vel_command_b=torch.zeros_like(sampled_commands),
        _sampled_command_bins=torch.zeros(3, 3, dtype=torch.long),
        _command_curriculum=SimpleNamespace(
            sample=lambda _: (sampled_commands.clone(), torch.zeros(3, 3, dtype=torch.long))
        ),
        is_standing_env=torch.zeros(3, dtype=torch.bool),
    )

    symm_quadruped.GaitVelocityCommand._resample_command(command, torch.arange(3))

    assert torch.equal(command.vel_command_b[:2], torch.zeros(2, 3))
    assert command.vel_command_b[2, 0].item() == pytest.approx(0.1001)


def test_gait_phase_is_continuous_when_period_or_velocity_direction_changes():
    command = SimpleNamespace(
        gait_phase=torch.tensor([0.37]),
        _foot_phase_offsets=torch.tensor([[0.0, 0.5, 0.13, -0.13]]),
        vel_command_b=torch.tensor([[0.5, 0.0, 0.0]]),
    )
    phases_before = symm_quadruped.GaitVelocityCommand.foot_phases(command)

    command.gait_periods = torch.tensor([0.2])
    command.vel_command_b[:, 0] = -0.5
    phases_after = symm_quadruped.GaitVelocityCommand.foot_phases(command)

    assert torch.equal(phases_after, phases_before)


def _make_gait_direction_command(num_envs=1):
    cfg = make_gait_velocity_command(symm_quadruped)
    cfg.curriculum = None
    cfg.calculate_from_sampling_curve = False
    cfg.add_noise_theta = False
    cfg.gait_period = 0.5
    cfg.gait_transition_cycles = 0.0
    cfg.init_foot_thetas = ((-0.13, 0.13, 0.63, 0.37),)
    cfg.ranges.lin_vel_x = (0.0, 0.0)
    cfg.ranges.lin_vel_y = (0.0, 0.0)
    cfg.ranges.ang_vel_z = (0.0, 0.0)
    robot = SimpleNamespace(data=SimpleNamespace(heading_w=_tensor_data(torch.zeros(num_envs))))
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot}, extras={"log": {}})
    command = symm_quadruped.GaitVelocityCommand(cfg, env)
    command.reset(torch.arange(num_envs))
    return command


@pytest.mark.parametrize("update_method", ["timing", "command", "clock"])
def test_gait_direction_prioritizes_forward_lateral_then_yaw(update_method):
    velocities = torch.tensor(
        [
            [1.0, -0.5, -0.6],
            [-1.0, 0.5, 0.6],
            [0.0, -0.5, 0.6],
            [0.0, 0.5, -0.6],
            [0.0, 0.0, -0.6],
            [0.0, 0.0, 0.6],
            [0.05, -0.5, 0.6],
            [-0.05, 0.5, -0.6],
            [0.02, 0.05, -0.6],
            [-0.02, -0.05, 0.6],
            [0.0, 0.0, -0.0501],
            [0.0, 0.0, -0.05],
            [0.0, 0.0, 0.0],
        ]
    )
    expected = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    command = _make_gait_direction_command(len(velocities))
    command.vel_command_b[:] = velocities
    command.gait_phase[:] = 0.4
    if update_method == "timing":
        command._resample_gait_timing(torch.arange(len(velocities)), transition=False)
    elif update_method == "command":
        command._update_command()
    command._advance_gait_phase(0.02)

    expected_clock = 0.4 + expected * 0.04
    expected_offsets = expected[:, None] * command.init_foot_thetas
    torch.testing.assert_close(command.gait_phase, expected_clock)
    torch.testing.assert_close(command._foot_phase_offsets, expected_offsets)
    torch.testing.assert_close(command.foot_phases(), torch.remainder(expected_clock[:, None] + expected_offsets, 1.0))


def test_gait_direction_retains_previous_sign_in_deadband_and_resets_per_environment():
    command = _make_gait_direction_command(2)
    command.vel_command_b[:, 2] = -0.6
    command._update_command()
    command.vel_command_b[:] = torch.tensor([[0.0, 0.0, 0.0], [0.05, -0.05, 0.05]])
    command._resample_gait_timing(torch.arange(2), transition=False)
    command._advance_gait_phase(0.02)
    torch.testing.assert_close(command.gait_phase, torch.full((2,), 0.96))
    torch.testing.assert_close(command._foot_phase_offsets, -command.init_foot_thetas.expand(2, -1))

    command.reset(torch.tensor([0]))
    command.gait_phase[:] = 0.4
    command._advance_gait_phase(0.02)
    torch.testing.assert_close(command.gait_phase, torch.tensor([0.44, 0.36]))
    torch.testing.assert_close(command._foot_phase_offsets, torch.tensor([[1.0], [-1.0]]) * command.init_foot_thetas)


def test_gait_direction_change_blends_offsets_without_resampling_timing():
    command = _make_gait_direction_command(2)
    command.vel_command_b[:, 1] = -0.5
    command._update_command()
    command.cfg.gait_transition_cycles = 1.0
    command.gait_phase[:] = 0.4
    phases_before = command.foot_phases().clone()
    offsets_before = command._foot_phase_offsets.clone()
    periods_before = command.gait_periods.clone()
    duty_before = command.duty_factors.clone()
    command.vel_command_b[0, 1] = 0.5

    command._update_command()
    torch.testing.assert_close(command.foot_phases(), phases_before)
    command._advance_gait_phase(0.125)
    command._update_gait_transition(0.125)
    torch.testing.assert_close(command.gait_phase, torch.tensor([0.65, 0.15]))
    torch.testing.assert_close(command._foot_phase_offsets[0], 0.5 * offsets_before[0])
    torch.testing.assert_close(command._foot_phase_offsets[1], offsets_before[1])

    command._update_command()
    command._update_gait_transition(0.375)
    torch.testing.assert_close(command._foot_phase_offsets[0], command.init_foot_thetas[0])
    torch.testing.assert_close(command.gait_periods, periods_before)
    torch.testing.assert_close(command.duty_factors, duty_before)


def test_gait_direction_follows_heading_yaw_changes():
    command = _make_gait_direction_command()
    command.cfg.heading_command = True
    command.is_heading_env[:] = True
    command.heading_target[:] = -0.6
    command._update_command()
    command._advance_gait_phase(0.02)
    torch.testing.assert_close(command.gait_phase, torch.tensor([0.96]))
    torch.testing.assert_close(command._foot_phase_offsets, -command.init_foot_thetas)

    command.heading_target[:] = 0.6
    command._update_command()
    command.gait_phase[:] = 0.4
    command._advance_gait_phase(0.02)
    torch.testing.assert_close(command.gait_phase, torch.tensor([0.44]))
    torch.testing.assert_close(command._foot_phase_offsets, command.init_foot_thetas)


def test_gait_resampling_transition_modes_are_applied_per_environment(monkeypatch):
    command_cfg = make_gait_velocity_command(symm_quadruped)
    calls = {}
    command = SimpleNamespace(
        cfg=command_cfg,
        device="cpu",
        _transition_mode_probabilities=torch.full((3,), 1.0 / 3.0),
        time_left=torch.zeros(3),
        gait_time_left=torch.zeros(3),
        command_counter=torch.zeros(3, dtype=torch.long),
        _finalize_command_window=lambda env_ids: calls.setdefault("finalized", env_ids.clone()),
        _resample_command=lambda env_ids: calls.setdefault("velocity", env_ids.clone()),
        _update_command=lambda: None,
        _resample_gait_timing=lambda env_ids, transition: calls.setdefault("velocity_only", env_ids.clone()),
        _resample_gait=lambda env_ids, transition: calls.setdefault("gait", env_ids.clone()),
    )
    monkeypatch.setattr(torch, "multinomial", lambda *_args, **_kwargs: torch.tensor([0, 1, 2]))

    symm_quadruped.GaitVelocityCommand._resample_transition(command, torch.arange(3))

    assert torch.equal(calls["finalized"], torch.tensor([0, 2]))
    assert torch.equal(calls["velocity"], torch.tensor([0, 2]))
    assert torch.equal(calls["velocity_only"], torch.tensor([0]))
    assert torch.equal(calls["gait"], torch.tensor([1, 2]))
    assert torch.equal(command.time_left, torch.full((3,), 10.0))
    assert torch.equal(command.gait_time_left, command.time_left)


def test_gait_command_defers_expired_transition_until_after_reward():
    calls = []
    command = SimpleNamespace(
        cfg=SimpleNamespace(),
        device="cpu",
        _transition_mode_probabilities=torch.full((3,), 1.0 / 3.0),
        time_left=torch.tensor([0.01, 1.0]),
        _pending_transition_envs=torch.zeros(2, dtype=torch.bool),
        _advance_gait_phase=lambda _dt: calls.append("phase"),
        _update_gait_transition=lambda _dt: calls.append("gait_transition"),
        _update_metrics=lambda: calls.append("metrics"),
        _resample_transition=lambda env_ids: calls.append(("resample", env_ids.clone())),
        _update_command=lambda: calls.append("command"),
    )

    symm_quadruped.GaitVelocityCommand.compute(command, 0.02)

    assert calls == ["phase", "gait_transition", "metrics"]
    assert torch.equal(command._pending_transition_envs, torch.tensor([True, False]))

    symm_quadruped.GaitVelocityCommand.apply_pending_resampling(command)

    assert len(calls) == 5
    assert calls[3][0] == "resample"
    assert torch.equal(calls[3][1], torch.tensor([0]))
    assert calls[4] == "command"
    assert not torch.any(command._pending_transition_envs)


def test_gait_command_curriculum_uses_configured_reward_thresholds():
    command_cfg = make_gait_velocity_command(
        symm_quadruped,
        curriculum_tracking_lin_vel_threshold=0.85,
        curriculum_tracking_ang_vel_threshold=0.85,
    )
    command_cfg.curriculum.enabled = True
    curriculum_update = {}
    command = SimpleNamespace(
        cfg=command_cfg,
        device="cpu",
        num_envs=3,
        _command_curriculum=SimpleNamespace(
            update=lambda bin_ids, success: curriculum_update.update(bin_ids=bin_ids.clone(), success=success.clone())
        ),
        _sampled_command_bins=torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0]]),
        _command_tracking_lin_vel_reward_sum=torch.tensor([8.6, 8.4, 8.6]),
        _command_tracking_ang_vel_reward_sum=torch.tensor([8.6, 10.0, 8.4]),
        _command_window_step_count=torch.full((3,), 10.0),
        _successful_command_window_count=torch.zeros(3),
        _completed_command_window_count=torch.zeros(3),
        metrics={
            "tracking_reward_lin_vel": torch.zeros(3),
            "tracking_reward_ang_vel": torch.zeros(3),
            "success_threshold_vel_xy": torch.zeros(3),
            "success_threshold_vel_yaw": torch.zeros(3),
            "success_rate": torch.zeros(3),
        },
        _update_curriculum_metrics=lambda _env_ids: None,
    )

    symm_quadruped.GaitVelocityCommand._finalize_command_window(command, torch.arange(3))

    assert torch.equal(curriculum_update["bin_ids"], command._sampled_command_bins)
    assert torch.equal(curriculum_update["success"], torch.tensor([True, False, False]))
    assert torch.allclose(command.metrics["tracking_reward_lin_vel"], torch.tensor([0.86, 0.84, 0.86]))
    assert torch.allclose(command.metrics["tracking_reward_ang_vel"], torch.tensor([0.86, 1.0, 0.84]))
    assert torch.equal(command.metrics["success_rate"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.count_nonzero(command._command_tracking_lin_vel_reward_sum) == 0
    assert torch.count_nonzero(command._command_tracking_ang_vel_reward_sum) == 0
    assert torch.count_nonzero(command._command_window_step_count) == 0

    metrics_after_first_finalize = {name: value.clone() for name, value in command.metrics.items()}
    symm_quadruped.GaitVelocityCommand._finalize_command_window(command, torch.arange(3))
    for name, value in metrics_after_first_finalize.items():
        assert torch.equal(command.metrics[name], value)


def test_gait_command_accumulates_walk_these_ways_tracking_rewards():
    command_cfg = make_gait_velocity_command(symm_quadruped)
    command = SimpleNamespace(
        cfg=command_cfg,
        vel_command_b=torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_lin_vel_b=SimpleNamespace(torch=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])),
                root_ang_vel_b=SimpleNamespace(torch=torch.zeros(2, 3)),
            )
        ),
        _error_xy_sum=torch.zeros(2),
        _error_yaw_sum=torch.zeros(2),
        _step_count=torch.zeros(2),
        _command_tracking_lin_vel_reward_sum=torch.zeros(2),
        _command_tracking_ang_vel_reward_sum=torch.zeros(2),
        _command_window_step_count=torch.zeros(2),
    )

    symm_quadruped.GaitVelocityCommand._update_metrics(command)

    expected_tracking_reward = torch.tensor([math.exp(-1.0), 1.0])
    assert torch.allclose(command._command_tracking_lin_vel_reward_sum, expected_tracking_reward)
    assert torch.allclose(command._command_tracking_ang_vel_reward_sum, expected_tracking_reward)
    assert torch.equal(command._command_window_step_count, torch.ones(2))


def test_gait_transition_blends_offsets_and_duty_factor_over_one_cycle():
    command = SimpleNamespace(
        cfg=SimpleNamespace(gait_transition_cycles=1.0),
        device="cpu",
        gait_periods=torch.tensor([0.5]),
        _gait_transition_progress=torch.tensor([0.0]),
        foot_thetas=torch.zeros(1, 4),
        _foot_theta_transition_start=torch.zeros(1, 4),
        _foot_theta_transition_target=torch.full((1, 4), 0.5),
        _foot_phase_offsets=torch.zeros(1, 4),
        _foot_phase_offset_transition_start=torch.zeros(1, 4),
        _foot_phase_offset_transition_target=torch.full((1, 4), -0.5),
        duty_factors=torch.tensor([0.4]),
        _duty_factor_transition_start=torch.tensor([0.4]),
        _duty_factor_transition_target=torch.tensor([0.6]),
    )

    symm_quadruped.GaitVelocityCommand._update_gait_transition(command, 0.25)

    assert torch.equal(command._gait_transition_progress, torch.tensor([0.5]))
    assert torch.equal(command.foot_thetas, torch.full((1, 4), 0.25))
    assert torch.equal(command._foot_phase_offsets, torch.full((1, 4), -0.25))
    assert torch.equal(command.duty_factors, torch.tensor([0.5]))


def test_rewards_disable_straight_line_motion_and_configure_exponential_penalties():
    env_cfg = SimpleNamespace(rewards=SimpleNamespace())

    configure_rewards(
        env_cfg,
        symm_quadruped,
        joint_names=[f"joint_{index}" for index in range(12)],
        foot_body_names=[f"foot_{index}" for index in range(4)],
        foot_sensor_names=[f"contact_{index}" for index in range(4)],
        foot_sensor_body_names=[f"foot_{index}" for index in range(4)],
        base_height_range=(0.35, 0.45),
    )

    assert env_cfg.rewards.foot_periodicity is None
    assert env_cfg.rewards.foot_periodicity_force.func is symm_quadruped.foot_periodicity_force_exp_penalty
    assert env_cfg.rewards.foot_periodicity_force.weight == 0.30
    assert env_cfg.rewards.foot_periodicity_force.params["scale"] == 0.05
    assert env_cfg.rewards.foot_periodicity_velocity.func is symm_quadruped.foot_periodicity_velocity_exp_penalty
    assert env_cfg.rewards.foot_periodicity_velocity.weight == 0.30
    assert env_cfg.rewards.foot_periodicity_velocity.params["scale"] == 2.0
    assert env_cfg.rewards.base_height.func is symm_quadruped.base_height_exp
    assert env_cfg.rewards.base_height.weight == 0.15
    assert env_cfg.rewards.hip_action_penalty.func is symm_quadruped.hip_joint_position_exp_penalty
    assert env_cfg.rewards.hip_action_penalty.weight == 0.10
    assert env_cfg.rewards.hip_action_penalty.params["asset_cfg"].joint_names == [
        "joint_0",
        "joint_3",
        "joint_6",
        "joint_9",
    ]
    assert env_cfg.rewards.alive_bonus.weight == 0.0
    assert env_cfg.rewards.cmd is None
    assert env_cfg.rewards.sagittal_plane is None
    assert env_cfg.rewards.straight_line_motion is None
    assert env_cfg.rewards.termination_penalty.func is base_mdp.is_terminated
    assert env_cfg.rewards.termination_penalty.weight == 0.0
    assert env_cfg.rewards.joint_target_limits.func is symm_quadruped.joint_position_limits_exp_penalty
    assert env_cfg.rewards.joint_target_limits.weight == 2.0
    assert env_cfg.rewards.joint_target_limits.params["limit_fraction"] == 0.9
    assert env_cfg.rewards.joint_target_limits.params["scale"] == 10.0
    assert env_cfg.rewards.leg_permutation_symmetry.func is symm_quadruped.leg_permutation_symmetry_penalty
    assert env_cfg.rewards.leg_permutation_symmetry.weight == 0.1
    assert env_cfg.rewards.leg_permutation_symmetry.params["phase_sync_tolerance"] == 0.02
    assert env_cfg.rewards.smoothness.func is symm_quadruped.SmoothnessPenalty
    assert env_cfg.rewards.smoothness.weight == 0.1
    assert env_cfg.rewards.action_rate_l2.func is symm_quadruped.action_rate_l2_exp_penalty
    assert env_cfg.rewards.action_rate_l2.weight == 0.05
    assert env_cfg.rewards.action_rate_l2.params["scale"] == 0.05
    assert env_cfg.rewards.foot_clearance.weight == 0.10
    assert env_cfg.rewards.foot_clearance.func is symm_quadruped.foot_clearance_penalty
    assert env_cfg.rewards.foot_clearance.params["min_height"] == pytest.approx(0.1075)
    assert env_cfg.rewards.foot_clearance.params["height_scale"] == 0.025
    assert env_cfg.rewards.foot_clearance.params["use_command_gate"] is False
    assert env_cfg.rewards.foot_clearance.params["use_absolute_error"] is True
    assert env_cfg.rewards.foot_clearance.params["foot_height_offset"] == pytest.approx(0.0275)


def test_rewards_use_combined_xy_tracking_and_standard_yaw_and_roll_pitch_terms():
    env_cfg = SimpleNamespace(rewards=SimpleNamespace())

    configure_rewards(
        env_cfg,
        symm_quadruped,
        joint_names=[f"joint_{index}" for index in range(12)],
        foot_body_names=[f"foot_{index}" for index in range(4)],
        foot_sensor_names=[f"contact_{index}" for index in range(4)],
        foot_sensor_body_names=[f"foot_{index}" for index in range(4)],
        base_height_range=(0.35, 0.45),
    )

    assert env_cfg.rewards.track_lin_vel_xy_exp.func is base_mdp.track_lin_vel_xy_exp
    assert env_cfg.rewards.track_lin_vel_xy_exp.weight == 0.4
    assert env_cfg.rewards.track_lin_vel_xy_exp.params["std"] == 0.25
    assert env_cfg.rewards.track_lin_vel_x_exp is None
    assert env_cfg.rewards.track_lin_vel_y_exp is None
    assert env_cfg.rewards.track_ang_vel_z_exp.func is base_mdp.track_ang_vel_z_exp
    assert env_cfg.rewards.track_ang_vel_z_exp.weight == 0.3
    assert env_cfg.rewards.track_ang_vel_z_exp.params["std"] == 0.25
    assert env_cfg.rewards.base_roll_exp is None
    assert env_cfg.rewards.base_roll_pitch_exp.func is symm_quadruped.base_roll_pitch_exp
    assert env_cfg.rewards.base_roll_pitch_exp.weight == 0.15
    assert env_cfg.rewards.base_roll_pitch_exp.params["scale"] == 10.0
    assert env_cfg.rewards.straight_line_motion is None

    # Exercise the configured functions: the old custom yaw path squares the denominator.
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_lin_vel_b=_tensor_data(torch.tensor([[0.3, 0.4, 9.0]])),
                    root_ang_vel_b=_tensor_data(torch.tensor([[9.0, 9.0, 0.5]])),
                )
            )
        },
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)),
    )
    for term in (env_cfg.rewards.track_lin_vel_xy_exp, env_cfg.rewards.track_ang_vel_z_exp):
        assert term.func(env, **term.params).item() == pytest.approx(math.exp(-1.0))

    positive_names = (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "base_height",
        "base_roll_pitch_exp",
    )
    assert sum(getattr(env_cfg.rewards, name).weight for name in positive_names) == pytest.approx(1.0)


def test_configured_yaw_tracking_uses_direct_error_denominator():
    term = UnitreeGo2SymmFlatEnvCfg().rewards.track_ang_vel_z_exp
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=_tensor_data(torch.tensor([[0.0, 0.0, 0.5]]))))
        },
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(1, 3)),
    )
    assert term.func(env, **term.params).item() == pytest.approx(math.exp(-1.0))


def test_rewards_cfg_preserves_deprecated_morphological_symmetry_alias():
    cfg = SymmQuadrupedRewardsCfg()
    reward = object()
    cfg.leg_permutation_symmetry = reward

    with pytest.warns(DeprecationWarning, match="leg_permutation_symmetry"):
        assert cfg.morphological_symmetry is reward

    replacement = object()
    with pytest.warns(DeprecationWarning, match="leg_permutation_symmetry"):
        cfg.morphological_symmetry = replacement

    assert cfg.leg_permutation_symmetry is replacement
    assert "morphological_symmetry" not in cfg.__dict__

    configured_rewards = DobotX1SymmFlatEnvCfg().rewards
    with pytest.warns(DeprecationWarning, match="leg_permutation_symmetry"):
        configured_rewards.from_dict({"morphological_symmetry": {"weight": 0.42}})
    assert configured_rewards.leg_permutation_symmetry.weight == 0.42


@pytest.mark.parametrize(
    "go2_cfg_cls, x1_cfg_cls",
    [
        (UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg),
        (UnitreeGo2SymmFlatEnvCfg_PLAY, DobotX1SymmFlatEnvCfg_PLAY),
    ],
)
def test_robot_configs_share_rewards_except_robot_height_targets(go2_cfg_cls, x1_cfg_cls):
    x1_cfg = x1_cfg_cls()
    go2_cfg = go2_cfg_cls()

    for cfg, period in ((go2_cfg, 0.52), (x1_cfg, 0.60)):
        command = cfg.commands.base_velocity
        assert command.gait_period == period
        assert command.duty_factor == 0.56
        assert command.calculate_from_sampling_curve is False
        assert command.add_noise_period is False

    assert x1_cfg.rewards.leg_permutation_symmetry.func is symm_quadruped.leg_permutation_symmetry_penalty
    assert x1_cfg.rewards.leg_permutation_symmetry.weight == 0.1
    assert x1_cfg.rewards.leg_permutation_symmetry.params["phase_sync_tolerance"] == 0.02
    assert x1_cfg.rewards.track_ang_vel_z_exp.params["std"] == 0.25
    assert x1_cfg.rewards.foot_clearance.func is symm_quadruped.foot_clearance_penalty
    assert x1_cfg.rewards.foot_clearance.params["min_height"] == pytest.approx(0.1075)
    assert x1_cfg.rewards.foot_clearance.params["height_scale"] == 0.025
    assert x1_cfg.rewards.foot_clearance.params["use_command_gate"] is False
    assert x1_cfg.rewards.foot_clearance.params["use_absolute_error"] is True
    assert x1_cfg.rewards.foot_clearance.params["foot_height_offset"] == pytest.approx(0.0275)
    assert x1_cfg.rewards.foot_clearance.weight == 0.10
    assert go2_cfg.rewards.base_height.params["target_height"] == 0.30
    assert x1_cfg.rewards.base_height.params["target_height"] == 0.35
    assert go2_cfg.rewards.foot_clearance.params["min_height"] == pytest.approx(0.082)
    assert go2_cfg.rewards.foot_clearance.params["foot_height_offset"] == pytest.approx(0.022)

    # Only robot bindings and the explicitly configured height targets may differ.
    robot_binding_params = {"feet_cfg", "joint_cfg", "asset_cfg", "foot_sensor_body_names", "logical_joint_signs"}
    robot_target_params = {"base_height": {"target_height"}, "foot_clearance": {"min_height", "foot_height_offset"}}
    assert vars(go2_cfg.rewards).keys() == vars(x1_cfg.rewards).keys()
    for name, x1_term in vars(x1_cfg.rewards).items():
        go2_term = getattr(go2_cfg.rewards, name)
        if not isinstance(x1_term, RewardTermCfg):
            assert go2_term == x1_term, name
            continue
        assert go2_term.func is x1_term.func, name
        assert go2_term.weight == x1_term.weight, name
        excluded_params = robot_binding_params | robot_target_params.get(name, set())
        go2_params = {key: value for key, value in go2_term.params.items() if key not in excluded_params}
        x1_params = {key: value for key, value in x1_term.params.items() if key not in excluded_params}
        assert go2_params == x1_params, name


def test_robot_symmetry_rewards_match_x1_for_equivalent_logical_joint_states():
    logical_joint_pos = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 20.0
    gait_command = SimpleNamespace(foot_thetas=torch.zeros(2, 4))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=_Scene(),
        command_manager=SimpleNamespace(get_term=lambda _: gait_command),
    )
    joint_cfg = SimpleNamespace(joint_ids=list(range(12)))
    penalties = []
    for cfg, signs in (
        (UnitreeGo2SymmFlatEnvCfg(), symm_quadruped.SYMM_QUADRUPED_LOGICAL_JOINT_SIGNS),
        (DobotX1SymmFlatEnvCfg(), dobot_x1_symm.DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS),
    ):
        physical_joint_pos = logical_joint_pos * torch.tensor(signs).reshape(1, 12)
        env.scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=_tensor_data(physical_joint_pos)))
        term = cfg.rewards.leg_permutation_symmetry
        penalties.append(term.func(env, **{**term.params, "joint_cfg": joint_cfg}))

    # The compatibility adapter preserves X1's original reward semantics.
    x1_reference = dobot_x1_symm.leg_permutation_symmetry_penalty(
        env, command_name="base_velocity", joint_cfg=joint_cfg
    )
    assert torch.all(x1_reference < 0.0)
    assert torch.allclose(penalties[0], x1_reference)
    assert torch.allclose(penalties[1], x1_reference)


def test_play_configs_enable_ground_filtered_normal_and_friction_forces():
    for cfg in (DobotX1SymmFlatEnvCfg_PLAY(), UnitreeGo2SymmFlatEnvCfg_PLAY()):
        for sensor_name in ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot"):
            sensor_cfg = getattr(cfg.scene, sensor_name)
            assert sensor_cfg.filter_prim_paths_expr == [SYMM_QUADRUPED_GROUND_COLLISION_PATH]
            assert sensor_cfg.track_friction_forces


def test_robot_configs_use_requested_height_ranges_and_orientation_terminations():
    x1_cfg = DobotX1SymmFlatEnvCfg()
    go2_cfg = UnitreeGo2SymmFlatEnvCfg()

    assert go2_cfg.rewards.straight_line_motion is None
    assert go2_cfg.terminations.base_orientation.params["max_pitch"] == 1.20
    assert x1_cfg.rewards.straight_line_motion is None
    assert x1_cfg.terminations.base_orientation.params["max_pitch"] == 0.70

    assert x1_cfg.commands.base_velocity.base_height_range == (0.45, 0.60)
    assert x1_cfg.rewards.base_height.params["target_height"] == 0.35
    assert x1_cfg.scene.robot.init_state.pos == (0.0, 0.0, 0.5)
    default_joint_pos = x1_cfg.scene.robot.init_state.joint_pos
    assert default_joint_pos["joint_front_left_abad"] == 0.0
    assert default_joint_pos["joint_front_left_thigh_pitch"] == 0.6983
    assert default_joint_pos["joint_front_left_calf_pitch"] == -1.2842
    assert default_joint_pos["joint_rear_left_abad"] == 0.0
    assert default_joint_pos["joint_rear_left_thigh_pitch"] == -0.6983
    assert default_joint_pos["joint_rear_left_calf_pitch"] == 1.2842


def test_x1_config_sets_branch_preserving_calf_limits_at_startup():
    x1_cfg = DobotX1SymmFlatEnvCfg()

    assert x1_cfg.actions.joint_pos.clip == {
        "joint_front_.*_calf_pitch": (-2.3, -0.2),
        "joint_rear_.*_calf_pitch": (0.2, 2.3),
    }
    expected_terms = {
        "front_calf_joint_limits": {
            "joint_names": ["joint_front_.*_calf_pitch"],
            "lower_limit_distribution_params": (-2.3, -2.3),
            "upper_limit_distribution_params": (-0.2, -0.2),
        },
        "rear_calf_joint_limits": {
            "joint_names": ["joint_rear_.*_calf_pitch"],
            "lower_limit_distribution_params": (0.2, 0.2),
            "upper_limit_distribution_params": (2.3, 2.3),
        },
    }
    for term_name, expected in expected_terms.items():
        term = getattr(x1_cfg.events, term_name)
        assert term.func is base_mdp.randomize_joint_parameters
        assert term.mode == "startup"
        assert term.params["asset_cfg"].name == "robot"
        assert term.params["asset_cfg"].joint_names == expected["joint_names"]
        assert term.params["lower_limit_distribution_params"] == expected["lower_limit_distribution_params"]
        assert term.params["upper_limit_distribution_params"] == expected["upper_limit_distribution_params"]
        assert term.params["operation"] == "abs"
        assert term.params["distribution"] == "uniform"


def test_robot_configs_use_uniform_21_by_1_by_21_command_bins():
    x1_command_cfg = DobotX1SymmFlatEnvCfg().commands.base_velocity
    go2_command_cfg = UnitreeGo2SymmFlatEnvCfg().commands.base_velocity

    assert x1_command_cfg.curriculum.enabled
    assert go2_command_cfg.curriculum.enabled
    assert x1_command_cfg.curriculum.num_bins == (21, 1, 21)
    assert go2_command_cfg.curriculum.num_bins == (21, 1, 21)
    assert x1_command_cfg.ranges.lin_vel_x == (-3.0, 3.0)
    assert x1_command_cfg.ranges.ang_vel_z == (-2.0, 2.0)
    assert x1_command_cfg.curriculum_tracking_lin_vel_threshold == 0.85
    assert x1_command_cfg.curriculum_tracking_ang_vel_threshold == 0.7
    assert go2_command_cfg.ranges.lin_vel_x == (-3.0, 3.0)
    assert go2_command_cfg.ranges.ang_vel_z == (-2.0, 2.0)
    assert go2_command_cfg.curriculum_tracking_lin_vel_threshold == 0.85
    assert go2_command_cfg.curriculum_tracking_ang_vel_threshold == 0.7
    assert go2_command_cfg.curriculum.to_dict() == x1_command_cfg.curriculum.to_dict()
    assert go2_command_cfg.ranges.to_dict() == x1_command_cfg.ranges.to_dict()


def test_x1_play_config_uses_exact_nominal_joint_posture():
    x1_play_cfg = DobotX1SymmFlatEnvCfg_PLAY()

    assert x1_play_cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)


def test_x1_config_terminates_low_or_face_down_front_body_postures():
    x1_cfg = DobotX1SymmFlatEnvCfg()

    assert x1_cfg.terminations.base_height.params["height_range"] == (0.25, 0.65)
    assert x1_cfg.terminations.base_orientation.params == {"max_roll": 0.7, "max_pitch": 0.7}
    assert x1_cfg.terminations.front_body_height.func is symm_quadruped.body_local_point_height_below
    assert x1_cfg.terminations.front_body_height.params["point_b"] == (0.35, 0.0, 0.0)
    assert x1_cfg.terminations.front_body_height.params["min_height"] == 0.08


@pytest.mark.parametrize("target_height", [None, 0.30])
def test_base_height_penalty_uses_target_instead_of_accepting_a_range(target_height):
    target = 0.35 if target_height is None else target_height
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, target + offset] for offset in (0.0, -0.05, 0.05, 0.15)])),
        )
    )
    env = SimpleNamespace(scene=scene)

    params = {} if target_height is None else {"target_height": target_height}
    penalty = symm_quadruped.base_height_range_penalty(env, height_range=(0.45, 0.60), **params)

    assert penalty.tolist() == pytest.approx([0.0, math.exp(-1.0) - 1.0, math.exp(-1.0) - 1.0, math.exp(-3.0) - 1.0])


def test_body_local_point_height_detects_virtual_front_body_ground_contact():
    scene = _Scene()
    scene.env_origins = torch.zeros(2, 3)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.35], [0.0, 0.0, 0.35]])),
            root_quat_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            ),
        )
    )
    env = SimpleNamespace(scene=scene)

    terminated = symm_quadruped.body_local_point_height_below(
        env,
        point_b=(0.35, 0.0, 0.0),
        min_height=0.08,
    )

    assert torch.equal(terminated, torch.tensor([True, False]))


def test_go2_clearance_reward_favors_airborne_swing_feet_without_contact():
    body_pos_w = torch.zeros(2, 4, 3)
    body_pos_w[0, :, 2] = 0.10
    body_pos_w[1, :, 2] = 0.02
    scene = _Scene()
    scene.env_origins = torch.zeros(2, 3)
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(body_pos_w)))
    scene.sensors = {
        sensor_name: SimpleNamespace(
            num_sensors=1,
            body_names=["foot"],
            data=SimpleNamespace(
                net_forces_w_history=_tensor_data(
                    torch.tensor(
                        [
                            [[[0.0, 0.0, 0.0]]],
                            [[[0.0, 0.0, 50.0]]],
                        ]
                    )
                ),
            ),
        )
        for sensor_name in ("fl", "fr", "rl", "rr")
    }
    gait_command = SimpleNamespace(
        duty_factors=torch.full((2,), 0.45),
        kappa=torch.full((2,), 16.0),
        foot_phases=lambda: torch.full((2, 4), 0.275),
    )
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(
            get_term=lambda _: gait_command,
            get_command=lambda _: torch.tensor([[1.0, 0.0, 0.0]] * 2),
        ),
    )

    reward = symm_quadruped.foot_clearance_tracking_reward(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        foot_sensor_names=("fl", "fr", "rl", "rr"),
        foot_sensor_body_names=("foot",) * 4,
        target_height=0.10,
        height_scale=0.03,
    )

    assert reward[0] > 0.95
    assert reward[1] < 0.01
    assert env._foot_clearance_diagnostics["reward"][0] == reward[0]


def test_go2_clearance_reward_penalizes_one_low_or_overlifting_swing_foot():
    body_pos_w = torch.zeros(3, 4, 3)
    body_pos_w[0, :, 2] = 0.08
    body_pos_w[1, :, 2] = torch.tensor([0.0, 0.08, 0.08, 0.08])
    body_pos_w[2, :, 2] = torch.tensor([0.25, 0.08, 0.08, 0.08])
    scene = _Scene()
    scene.env_origins = torch.zeros(3, 3)
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(body_pos_w)))
    scene.sensors = {
        sensor_name: SimpleNamespace(
            num_sensors=1,
            body_names=["foot"],
            data=SimpleNamespace(net_forces_w_history=_tensor_data(torch.zeros(3, 1, 1, 3))),
        )
        for sensor_name in ("fl", "fr", "rl", "rr")
    }
    gait_command = SimpleNamespace(
        duty_factors=torch.full((3,), 0.45),
        kappa=torch.full((3,), 16.0),
        foot_phases=lambda: torch.full((3, 4), 0.275),
    )
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(
            get_term=lambda _: gait_command,
            get_command=lambda _: torch.tensor([[1.0, 0.0, 0.0]] * 3),
        ),
    )

    reward = symm_quadruped.foot_clearance_tracking_reward(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        foot_sensor_names=("fl", "fr", "rl", "rr"),
        foot_sensor_body_names=("foot",) * 4,
        target_height=0.08,
        height_scale=0.03,
        excess_height_margin=0.03,
        excess_height_scale=0.03,
    )

    assert reward[0] > 0.95
    assert reward[1] < 0.30
    assert reward[2] < 0.10


def test_foot_clearance_uses_ground_relative_swing_target_without_current_speed_gate():
    foot_heights = torch.tensor([0.02, 0.05, 0.08, 0.10])
    body_pos_w = torch.zeros(3, 4, 3)
    body_pos_w[:, :, 2] = foot_heights
    body_pos_w[1, :, 2] += 1.0
    scene = _Scene()
    scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(body_pos_w)))
    foot_phases = torch.full((3, 4), 0.275)
    gait_command = _make_clearance_gait_command(foot_phases, duty_factor=0.45)
    command = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(
            get_term=lambda _: gait_command,
            get_command=lambda _: command,
        ),
    )

    penalty = symm_quadruped.foot_clearance_penalty(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        min_height=0.08,
    )

    assert penalty[0] < 0.0
    assert penalty[0].item() == pytest.approx(penalty[1].item())
    assert penalty[2].item() == pytest.approx(0.0)
    assert env._foot_clearance_diagnostics["target_height"][0].tolist() == pytest.approx([0.08] * 4)
    assert env._foot_clearance_diagnostics["shortfall"][0].tolist() == pytest.approx([0.06, 0.03, 0.0, 0.0])


@pytest.mark.parametrize("command_speed, expected_gate", [(0.1, 0.5), (0.2, 1.0), (0.4, 1.0)])
def test_foot_clearance_gate_uses_planar_speed_in_every_direction(command_speed, expected_gate):
    directions = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.6, 0.8], [-0.6, 0.8], [0.6, -0.8], [-0.6, -0.8]]
    )
    num_envs = len(directions) + 2
    command = torch.zeros(num_envs, 3)
    command[0, 0] = 0.2  # Fully active forward reference.
    command[1:-1, :2] = command_speed * directions
    scene = _Scene()
    scene.env_origins = torch.zeros(num_envs, 3)
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(torch.zeros(num_envs, 4, 3))))
    gait_command = _make_clearance_gait_command(torch.full((num_envs, 4), 0.25))
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(
            get_term=lambda _: gait_command,
            get_command=lambda _: command,
        ),
    )

    penalty = symm_quadruped.foot_clearance_penalty(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        min_height=0.04,
        height_scale=0.025,
        min_command_speed=0.2,
    )

    assert penalty[0] < 0.0
    torch.testing.assert_close(penalty[1:-1], (expected_gate * penalty[0]).expand(len(directions)))
    assert penalty[-1].item() == pytest.approx(0.0)


@pytest.mark.parametrize("yaw_scale", [0.2, 0.4])
def test_foot_clearance_gate_includes_yaw_with_an_independent_scale(yaw_scale):
    command = torch.tensor(
        [
            [0.2, 0.0, 0.0],  # Fully active planar reference.
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5 * yaw_scale],
            [0.0, 0.0, -0.5 * yaw_scale],
            [0.0, 0.0, yaw_scale],
            [0.0, 0.0, -yaw_scale],
            [0.0, 0.0, 2.0 * yaw_scale],
            [0.0, 0.0, -2.0 * yaw_scale],
            [0.06, 0.08, 0.25 * yaw_scale],  # Planar speed dominates.
            [0.03, 0.04, -0.75 * yaw_scale],  # Yaw rate dominates.
            [0.06, 0.08, 0.5 * yaw_scale],  # Equally active gates must not add.
        ]
    )
    num_envs = len(command)
    scene = _Scene()
    scene.env_origins = torch.zeros(num_envs, 3)
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=_tensor_data(torch.zeros(num_envs, 4, 3))))
    gait_command = _make_clearance_gait_command(torch.full((num_envs, 4), 0.25))
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(
            get_term=lambda _: gait_command,
            get_command=lambda _: command,
        ),
    )
    penalty = symm_quadruped.foot_clearance_penalty(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
        min_height=0.04,
        height_scale=0.025,
        min_command_speed=0.2,
        min_command_yaw_rate=yaw_scale,
    )

    assert penalty[0] < 0.0
    expected_gates = torch.tensor([1.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.75, 0.5])
    torch.testing.assert_close(penalty, expected_gates * penalty[0])


def test_straight_line_motion_reward_preserves_forward_signal_and_penalizes_lost_support():
    pitch = 0.8
    scene = _Scene()
    scene.env_origins = torch.zeros(5, 3)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.4],
                        [0.0, 0.0, 0.4],
                        [0.0, 0.0, 0.4],
                        [0.0, 0.5, 0.4],
                        [0.0, 0.0, 0.15],
                    ]
                )
            ),
            root_quat_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)],
                    ]
                )
            ),
            heading_w=_tensor_data(torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0])),
            root_lin_vel_b=_tensor_data(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.6, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                )
            ),
            root_ang_vel_b=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.6],
                        [0.0, 0.0, 0.0],
                    ]
                )
            ),
        )
    )
    command = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )

    reward = symm_quadruped.straight_line_motion_reward(
        env,
        command_name="base_velocity",
        forward_velocity_scale=0.35,
        lateral_position_scale=0.50,
        heading_scale=0.50,
        lateral_velocity_scale=0.25,
        yaw_rate_scale=0.25,
        pose_weight=0.10,
        roll_scale=0.25,
        pitch_scale=0.35,
        min_base_height=0.35,
        height_scale=0.10,
    )

    assert torch.equal(reward[:2], torch.full((2,), 1.55))
    assert 0.50 < reward[2] < 0.60
    assert reward[3] < reward[0] - 0.20
    assert reward[4] < reward[0] - 0.20
    assert torch.all(reward >= -0.25)
    assert torch.all(reward <= 1.55)
    assert set(env._straight_line_motion_diagnostics) == {
        "forward_score",
        "lateral_velocity_score",
        "yaw_rate_score",
        "roll_score",
        "straight_score",
        "lateral_position_score",
        "heading_score",
        "pose_score",
        "posture_score",
        "support_loss",
        "reward",
    }
    assert torch.equal(env._straight_line_motion_diagnostics["reward"], reward)
    assert all(not value.requires_grad for value in env._straight_line_motion_diagnostics.values())


def test_independent_velocity_tracking_and_roll_rewards_only_measure_their_component():
    roll = 0.25
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0)],
                    ]
                )
            ),
            root_lin_vel_b=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.35, 0.0, 0.0],
                        [0.0, 0.20, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
            ),
            root_ang_vel_b=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.20],
                        [0.0, 0.0, 0.0],
                    ]
                )
            ),
        )
    )
    command = torch.zeros(5, 3)
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: command))
    expected = math.exp(-1.0)

    x_reward = symm_quadruped.track_lin_vel_x_exp(env, command_name="base_velocity", error_scale=0.35)
    y_reward = symm_quadruped.track_lin_vel_y_exp(env, command_name="base_velocity", error_scale=0.20)
    yaw_reward = symm_quadruped.track_ang_vel_z_exp(env, command_name="base_velocity", error_scale=0.20)
    roll_reward = symm_quadruped.base_roll_exp(env, error_scale=0.25)
    roll_penalty = symm_quadruped.base_roll_exp_penalty(env, error_scale=0.25)

    assert x_reward.tolist() == pytest.approx([1.0, expected, 1.0, 1.0, 1.0])
    assert y_reward.tolist() == pytest.approx([1.0, 1.0, expected, 1.0, 1.0])
    assert yaw_reward.tolist() == pytest.approx([1.0, 1.0, 1.0, expected, 1.0])
    assert roll_reward.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, expected])
    assert roll_penalty.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, expected - 1.0])


@pytest.mark.parametrize("yaw", [0.0, 1.3])
def test_base_roll_penalty_uses_absolute_roll_and_ignores_pitch(yaw):
    roll = torch.tensor([0.0, 0.0, 0.0, 0.125, -0.125, 0.125, 0.125])
    pitch = torch.tensor([0.0, 0.125, -0.125, 0.0, 0.0, 0.125, -0.125])
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=_tensor_data(quat_from_euler_xyz(roll, pitch, torch.full_like(roll, yaw))),
        )
    )
    env = SimpleNamespace(scene=scene)

    penalty = symm_quadruped.base_roll_exp_penalty(env, error_scale=0.25)

    roll_penalty = math.exp(-0.5) - 1.0
    assert penalty.tolist() == pytest.approx([0.0] * 3 + [roll_penalty] * 4)


def test_straight_line_motion_reward_penalizes_world_lateral_position_and_heading():
    scene = _Scene()
    scene.env_origins = torch.zeros(2, 3)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.4], [0.0, 20.0, 0.4]])),
            root_quat_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])),
            heading_w=_tensor_data(torch.tensor([0.0, math.pi])),
            root_lin_vel_b=_tensor_data(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
            root_ang_vel_b=_tensor_data(torch.zeros(2, 3)),
        )
    )
    command = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: command))

    reward = symm_quadruped.straight_line_motion_reward(env, command_name="base_velocity")

    assert reward[0] > reward[1] + 0.09


def test_straight_line_motion_reward_components_are_bounded_and_reusable():
    pitch = 0.8
    roll = 0.5
    scene = _Scene()
    scene.env_origins = torch.zeros(2, 3)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.15], [0.0, 0.0, 0.40]])),
            root_quat_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)],
                        [math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0)],
                    ]
                )
            ),
            heading_w=_tensor_data(torch.zeros(2)),
            root_lin_vel_b=_tensor_data(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
            root_ang_vel_b=_tensor_data(torch.zeros(2, 3)),
        )
    )
    command = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: command))

    components = symm_quadruped.straight_line_motion_reward_components(env, command_name="base_velocity")

    assert set(components) == {
        "forward_score",
        "lateral_velocity_score",
        "yaw_rate_score",
        "roll_score",
        "straight_score",
        "lateral_position_score",
        "heading_score",
        "pose_score",
        "posture_score",
        "support_loss",
    }
    assert torch.equal(components["forward_score"], torch.ones(2))
    assert components["straight_score"][0].item() == 1.0
    assert components["straight_score"][1].item() < 0.25
    assert components["posture_score"][0].item() < 0.15
    assert components["posture_score"][1].item() == 1.0
    assert 0.60 < components["support_loss"][0].item() < 0.70
    assert components["support_loss"][1].item() == 0.0
    assert all(torch.all((value >= 0.0) & (value <= 1.0)) for value in components.values())


@pytest.mark.parametrize(
    "cfg_cls",
    [UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg, UnitreeGo2SymmFlatEnvCfg_PLAY, DobotX1SymmFlatEnvCfg_PLAY],
)
def test_robot_negative_reward_weights_scales_and_hip_selection(cfg_cls):
    cfg = cfg_cls()
    expected = {
        "hip_action_penalty": (0.1, 1.0),
        "joint_target_limits": (2.0, 10.0),
        "action_rate_l2": (0.05, 0.05),
        "lin_vel_z_l2": (0.1, 5.0),
        "dof_vel_l2": (0.05, 0.01),
        "dof_torques_l2": (0.05, 0.004),
        "action_smoothness": (0.05, 0.05),
        "ang_vel_xy_l2": (0.05, 1.0),
        "dof_acc_l2": (0.05, 4.0e-6),
    }
    for name, (weight, scale) in expected.items():
        term = getattr(cfg.rewards, name)
        assert term.weight == weight, name
        assert term.params["scale"] == scale, name
    assert cfg.rewards.smoothness.func is symm_quadruped.SmoothnessPenalty
    assert cfg.rewards.smoothness.weight == 0.1
    assert sum(isinstance(term, RewardTermCfg) and term.weight != 0 for term in vars(cfg.rewards).values()) == 18

    # Simulate an articulation whose storage order differs from the policy joint order.
    logical_names = cfg.actions.joint_pos.joint_names
    storage_names = logical_names[1::3] + logical_names[2::3] + logical_names[::3]
    defaults = torch.arange(36, dtype=torch.float32).reshape(3, 12) * 0.01
    positions = defaults.clone()
    positions[:, :8] += 3.0  # Non-hip deviations must not enter the hip error.
    positions[1, 8:] += torch.tensor([0.1, -0.2, 0.3, -0.4])
    positions[2, 8:] -= torch.tensor([0.1, -0.2, 0.3, -0.4])
    asset = SimpleNamespace(
        joint_names=storage_names,
        num_joints=12,
        find_joints=lambda names, preserve_order: ([storage_names.index(name) for name in names], names),
        data=SimpleNamespace(joint_pos=_tensor_data(positions), default_joint_pos=_tensor_data(defaults)),
    )
    env = SimpleNamespace(scene={"robot": asset})
    term = cfg.rewards.hip_action_penalty
    term.params["asset_cfg"].resolve(env.scene)
    assert term.params["asset_cfg"].joint_ids == [8, 9, 10, 11]
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0.0, math.expm1(-0.3), math.expm1(-0.3)]))


def test_joint_limits_use_measured_positions_and_ninety_percent_of_full_runtime_range():
    positions = torch.tensor([[0.9, 3.9, 99.0], [0.95, 4.0, 99.0], [-1.0, 2.0, 99.0], [100.0, -100.0, 99.0]])
    limits = torch.tensor([[[-1.0, 1.0], [2.0, 4.0], [-200.0, 200.0]]]).repeat(4, 1, 1)
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=_tensor_data(positions),
                    joint_pos_limits=_tensor_data(limits),
                    soft_joint_pos_limits=_tensor_data(limits * 0.5),
                )
            )
        }
    )
    result = symm_quadruped.joint_position_limits_exp_penalty(env, asset_cfg=SceneEntityCfg("robot", joint_ids=[1, 0]))
    torch.testing.assert_close(result, torch.tensor([0.0, math.expm1(-1.5), math.expm1(-2.0), -1.0]))
    # Runtime limit changes, including X1's calf limits, take effect immediately.
    limits[0, 0] = torch.tensor([-2.0, 0.0])
    result = symm_quadruped.joint_position_limits_exp_penalty(env, asset_cfg=SceneEntityCfg("robot", joint_ids=[0]))
    assert result[0].item() == pytest.approx(math.expm1(-10.0))


@pytest.mark.parametrize("limit_fraction", [0.0, -0.1, 1.1])
def test_joint_limits_reject_invalid_range_fraction(limit_fraction):
    with pytest.raises(ValueError, match="Joint-limit fraction"):
        symm_quadruped.joint_position_limits_exp_penalty(SimpleNamespace(), limit_fraction=limit_fraction)


@pytest.mark.parametrize(
    "term_name, field_name, values, expected_errors",
    [
        ("lin_vel_z_l2", "root_lin_vel_b", [[9.0, 8.0, 0.0], [9.0, 8.0, -0.5]], [0.0, 0.25]),
        ("ang_vel_xy_l2", "root_ang_vel_b", [[0.0, 0.0, 9.0], [-0.3, 0.4, 9.0]], [0.0, 0.25]),
        ("dof_vel_l2", "joint_vel", [[0.0, 9.0, 0.0], [-3.0, 9.0, 4.0]], [0.0, 25.0]),
        ("dof_torques_l2", "applied_torque", [[0.0, 9.0, 0.0], [-3.0, 9.0, 4.0]], [0.0, 25.0]),
    ],
)
def test_configured_velocity_and_torque_penalties_use_selected_squared_errors(
    term_name, field_name, values, expected_errors
):
    term = getattr(UnitreeGo2SymmFlatEnvCfg().rewards, term_name)
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(**{field_name: _tensor_data(torch.tensor(values))}))}
    )
    params = {**term.params, "asset_cfg": SceneEntityCfg("robot", joint_ids=[2, 0])}
    expected = torch.tensor([math.expm1(-term.params["scale"] * error) for error in expected_errors])
    torch.testing.assert_close(term.func(env, **params), expected)


def test_action_smoothness_tracks_second_difference_and_clears_only_reset_environments():
    actions = torch.zeros(2, 12)
    previous = torch.zeros_like(actions)
    env = SimpleNamespace(action_manager=SimpleNamespace(action=actions, prev_action=previous))
    cfg = UnitreeGo2SymmFlatEnvCfg().rewards.action_smoothness
    term = cfg.func(cfg, env)

    actions[:, 0] = 1.0
    torch.testing.assert_close(term(env, **cfg.params), torch.full((2,), math.expm1(-0.05)))
    previous.copy_(actions)
    actions[:, 0] = 2.0
    torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))
    previous.copy_(actions)
    actions[:, 0] = 3.0
    torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))

    term.reset([0])
    previous.copy_(actions)
    previous[0] = 0.0  # ActionManager resets the same environment's history.
    actions[0] = 0.0
    actions[0, 0] = 1.0
    actions[1, 0] = 4.0
    torch.testing.assert_close(term(env, **cfg.params), torch.tensor([math.expm1(-0.05), 0.0]))
    term.reset()
    previous.zero_()
    actions.zero_()
    torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))


@pytest.mark.parametrize("step_dt", [0.02, 0.04])
def test_joint_acceleration_uses_control_step_difference_and_reset_velocities(step_dt):
    velocity = torch.tensor([[1.0, 99.0, -1.0], [-2.0, 99.0, 2.0]])
    env = SimpleNamespace(
        step_dt=step_dt,
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_vel=_tensor_data(velocity)))},
    )
    cfg = UnitreeGo2SymmFlatEnvCfg().rewards.dof_acc_l2
    cfg.params["asset_cfg"] = SceneEntityCfg("robot", joint_ids=[2, 0])
    term = cfg.func(cfg, env)
    velocity[:, 0] += 0.2
    velocity[:, 2] -= 0.4
    expected = math.expm1(-4.0e-6 * 0.2 / step_dt**2)
    torch.testing.assert_close(term(env, **cfg.params), torch.full((2,), expected))

    velocity[0] = torch.tensor([10.0, -99.0, -10.0])
    term.reset(torch.tensor([0]))
    velocity[:, 0] += 0.2
    velocity[:, 2] -= 0.4
    torch.testing.assert_close(term(env, **cfg.params), torch.full((2,), expected))
    term.reset()
    torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))


def test_configured_action_rate_uses_squared_changes_without_stability_gate():
    term = UnitreeGo2SymmFlatEnvCfg().rewards.action_rate_l2
    actions = torch.zeros(3, 12)
    actions[1, :2] = torch.tensor([0.25, -0.25])
    actions[2, :4] = torch.tensor([0.125, -0.125, 0.125, -0.125])
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=actions, prev_action=torch.zeros_like(actions)),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=_tensor_data(torch.zeros(3, 3)),
                    root_quat_w=_tensor_data(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)),
                )
            )
        },
    )
    # Equal L1 errors have distinct squared costs, even with the base at ground height.
    expected = torch.tensor([0.0, math.expm1(-0.05 * 0.125), math.expm1(-0.05 * 0.0625)])
    torch.testing.assert_close(term.func(env, **term.params), expected)


@pytest.mark.parametrize("use_stability_gate", [False, True])
def test_action_rate_penalty_uses_l1_change_and_preserves_stability_gate(use_stability_gate):
    action_delta = torch.zeros(5, 12)
    action_delta[1, :3] = torch.tensor([0.125, -0.25, 0.125])
    action_delta[2, :8] = torch.tensor([0.0625, -0.0625] * 4)
    action_delta[3] = -action_delta[1]
    action_delta[4] = 10.0
    previous_action = torch.full_like(action_delta, 0.25)
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.25]]).repeat(5, 1)),
            root_quat_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(5, 1)),
        )
    )
    env = SimpleNamespace(
        scene=scene,
        action_manager=SimpleNamespace(action=previous_action + action_delta, prev_action=previous_action),
    )

    penalty = symm_quadruped.action_rate_exp_penalty(env, scale=4.0, use_stability_gate=use_stability_gate)

    # Equal L1 changes must cost the same regardless of sign or distribution across joints.
    # Constant nonzero actions cost nothing; very large changes saturate at the gate magnitude.
    expected = torch.tensor([0.0, math.expm1(-2.0), math.expm1(-2.0), math.expm1(-2.0), -1.0])
    if use_stability_gate:
        expected *= 0.5  # Upright at the height threshold gives a gate of exactly one half.
    torch.testing.assert_close(penalty, expected)


def test_joint_position_target_limit_penalty_activates_only_inside_limit_margin():
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            soft_joint_pos_limits=_tensor_data(torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]]))
        )
    )
    action_term = SimpleNamespace(
        _joint_ids=[0, 1, 2],
        processed_actions=torch.tensor([[0.0, 0.9, 1.0]]),
    )
    env = SimpleNamespace(scene=scene, action_manager=SimpleNamespace(get_term=lambda _: action_term))

    penalty = symm_quadruped.joint_position_target_limit_penalty(env, margin_fraction=0.10)

    assert penalty.item() == pytest.approx(-(0.0 + 0.25 + 1.0) / 3.0)


def test_sagittal_plane_penalty_allows_gait_sway_and_rejects_low_posture():
    scene = _Scene()
    scene.env_origins = torch.zeros(4, 3)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.4],
                        [0.0, 0.2, 0.4],
                        [0.0, 0.0, 0.1],
                        [0.0, 0.6, 0.4],
                    ]
                )
            ),
            root_quat_w=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            ),
            heading_w=_tensor_data(torch.tensor([0.0, 0.2, 0.0, 0.6])),
            root_lin_vel_b=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.25, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.7, 0.0],
                    ]
                )
            ),
            root_ang_vel_b=_tensor_data(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.25],
                        [0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.7],
                    ]
                )
            ),
        )
    )
    env = SimpleNamespace(scene=scene)

    penalty = symm_quadruped.sagittal_plane_penalty(
        env,
        lateral_position_tolerance=0.30,
        heading_tolerance=0.30,
        lateral_velocity_tolerance=0.35,
        roll_tolerance=0.25,
        roll_rate_tolerance=0.75,
        yaw_rate_tolerance=0.35,
        secondary_weight=1.0,
        pose_weight=0.1,
        min_base_height=0.25,
        height_tolerance=0.10,
        low_height_weight=1.0,
    )

    assert torch.equal(penalty[:2], torch.zeros(2))
    assert penalty[2] < -0.5
    assert penalty[3] < 0.0
    assert torch.all(penalty >= -2.1)


def test_domain_randomization_resets_with_zero_lateral_and_yaw_velocity():
    events = SimpleNamespace(
        physics_material=SimpleNamespace(params={}),
        add_base_mass=SimpleNamespace(params={}),
        base_external_force_torque=SimpleNamespace(params={}),
        base_com=object(),
        reset_base=SimpleNamespace(params={}),
        push_robot=SimpleNamespace(params={}),
    )
    env_cfg = SimpleNamespace(events=events)

    configure_domain_randomization(env_cfg)

    assert env_cfg.events.reset_base.params["velocity_range"]["y"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.events.push_robot.params["velocity_range"]["y"] == (0.0, 0.0)
