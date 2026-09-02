# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared symmetric quadruped task configuration."""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ActionManager, ObservationTermCfg
from isaaclab.utils.buffers import CircularBuffer

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.agents.rsl_rl_ppo_cfg import (
    DobotX1SymmFlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg,
    DobotX1SymmFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2SymmFlatPPORunnerCfg,
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


def test_training_physics_uses_proven_aggregate_pair_capacity():
    physics_cfg = SymmQuadrupedPhysicsCfg().physx

    assert physics_cfg.gpu_total_aggregate_pairs_capacity == 2**22


def test_flat_scene_defaults_to_scalable_training_batch():
    scene = SimpleNamespace(
        num_envs=4096,
        terrain=SimpleNamespace(),
        sky_light=SimpleNamespace(),
    )
    env_cfg = SimpleNamespace(scene=scene)

    configure_flat_scene(env_cfg)

    assert env_cfg.scene.num_envs == 512


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

    assert cfg.max_iterations == 20000
    assert cfg.clip_actions is None
    assert cfg.actor.distribution_cfg.init_std == 0.5
    assert cfg.algorithm.entropy_coef == 0.005
    assert cfg.algorithm.symmetry_cfg.command_observation_index == 3
    assert cfg.algorithm.symmetry_cfg.min_abs_command_velocity == 0.0


def test_environment_rewards_and_records_old_target_before_publishing_new_observation():
    events = []
    command_target = {"value": "old"}
    env = object.__new__(symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device="cpu", is_rendering=False, step=lambda **_: events.append("physics"))
    env.cfg = SimpleNamespace(decimation=1, sim=SimpleNamespace(dt=0.02, render_interval=1))
    env._physics_handles_decimation = True
    env._sim_step_counter = 0
    env.render_enabled = False
    env.action_manager = SimpleNamespace(
        process_action=lambda _: events.append("action"),
        apply_action=lambda: None,
    )
    env._clamp_processed_joint_position_targets = lambda: None
    env.recorder_manager = SimpleNamespace(
        active_terms=("transition_trace",),
        record_pre_step=lambda: None,
        record_post_physics_decimation_step=lambda: None,
        record_post_step=lambda: events.append(f"record:{command_target['value']}"),
    )
    env.scene = SimpleNamespace(
        write_data_to_sim=lambda: None,
        update=lambda **_: None,
    )
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env.common_step_counter = 0
    env.termination_manager = SimpleNamespace(
        compute=lambda: torch.zeros(1, dtype=torch.bool),
        terminated=torch.zeros(1, dtype=torch.bool),
        time_outs=torch.zeros(1, dtype=torch.bool),
    )

    def compute_reward(*, dt):
        events.append(f"reward:{command_target['value']}")
        return torch.ones(1)

    env.reward_manager = SimpleNamespace(
        compute=compute_reward,
        get_term_cfg=lambda _: SimpleNamespace(weight=0.0),
    )

    def compute_diagnostics(*_):
        events.append(f"diagnostics:{command_target['value']}")
        return {}

    env._compute_step_diagnostics = compute_diagnostics

    def resample_command(*, dt):
        events.append("resample")
        command_target["value"] = "new"

    env.command_manager = SimpleNamespace(compute=resample_command)
    env.event_manager = SimpleNamespace(available_modes=())

    def compute_observation(*, update_history=False):
        events.append(f"observation:{command_target['value']}")
        return {"policy": torch.zeros((1, 1))}

    env.observation_manager = SimpleNamespace(compute=compute_observation)
    env.extras = {}

    env.step(torch.zeros((1, 12)))

    assert events.index("physics") < events.index("reward:old")
    assert events.index("reward:old") < events.index("diagnostics:old")
    assert events.index("diagnostics:old") < events.index("observation:old")
    assert events.index("observation:old") < events.index("record:old")
    assert events.index("record:old") < events.index("resample") < events.index("observation:new")


@pytest.mark.parametrize("env_cfg_type", [UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg])
def test_symm_quadruped_command_and_phase_mapping_configuration_is_stable(env_cfg_type):
    command_cfg = env_cfg_type().commands.base_velocity

    assert command_cfg.ranges.lin_vel_x == (-2.0, 2.0)
    assert command_cfg.min_xy_command_norm == 0.0
    assert command_cfg.phase_mapping_version == symm_quadruped.SYMM_QUADRUPED_PHASE_MAPPING_VERSION
    assert command_cfg.phase_mapping_version == "same_gait_backward_duty_aware_integrated_reward_boundary_v4"


@pytest.mark.parametrize(
    ("train_cfg_type", "play_cfg_type"),
    (
        (UnitreeGo2SymmFlatEnvCfg, UnitreeGo2SymmFlatEnvCfg_PLAY),
        (DobotX1SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg_PLAY),
    ),
)
def test_training_uses_one_shot_velocity_gait_and_push_schedule(train_cfg_type, play_cfg_type):
    train_cfg = train_cfg_type()
    train_command = train_cfg.commands.base_velocity
    play_cfg = play_cfg_type()
    play_command = play_cfg.commands.base_velocity

    assert train_cfg.episode_length_s == pytest.approx(30.0)
    assert train_command.resampling_time_range == (10.0, 10.0)
    assert train_command.resample_once_after_reset is True
    assert train_command.resampling_time_gait == pytest.approx(20.0)
    assert train_command.resample_gait_once_after_reset is True
    assert train_cfg.events.push_robot.mode == "interval"
    assert train_cfg.events.push_robot.interval_range_s == (15.0, 15.0)
    assert train_cfg.events.push_robot.params["velocity_range"] == {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
    }

    assert play_command.resampling_time_range == (10.0, 10.0)
    assert play_command.resample_once_after_reset is True
    assert play_command.resample_gait_once_after_reset is True
    assert play_cfg.events.push_robot is None


@pytest.mark.parametrize(
    "runner_cfg_type",
    [UnitreeGo2SymmFlatPPORunnerCfg, DobotX1SymmFlatPPORunnerCfg],
)
def test_robot_ppo_time_reversal_configuration_is_stable(runner_cfg_type):
    runner_cfg = runner_cfg_type()
    symmetry_cfg = runner_cfg.algorithm.symmetry_cfg

    assert runner_cfg.max_iterations == 20000
    assert not symmetry_cfg.use_data_augmentation
    assert symmetry_cfg.use_time_reversal_regularization
    assert symmetry_cfg.use_mirror_loss
    assert symmetry_cfg.mirror_loss_coeff == 0.1
    assert symmetry_cfg.value_loss_coeff == 0.05
    assert symmetry_cfg.min_abs_command_velocity == 0.0
    assert symmetry_cfg.warmup_iterations == 500


def test_running_reward_is_clipped_before_terminal_penalty_is_added():
    total_reward = torch.tensor([0.5, -0.2, -4.5])
    termination_reward = torch.tensor([0.0, 0.0, -4.0])

    reward = symm_quadruped_env._clip_reward_before_termination(
        total_reward,
        termination_reward,
    )

    assert torch.equal(reward, torch.tensor([0.5, 0.0, -4.0]))


def test_policy_observations_use_hardware_proprioception_and_native_history():
    env_cfg = UnitreeGo2SymmFlatEnvCfg()
    policy = env_cfg.observations.policy

    assert policy.base_lin_vel is None
    assert policy.base_ang_vel is None
    assert policy.velocity_commands is None
    assert policy.foot_theta_sin is None
    assert policy.foot_theta_cos is None
    assert policy.phase_ratios is None
    assert policy.sagittal_plane_state is None
    assert policy.velocity_command.func is base_mdp.generated_commands
    assert policy.velocity_command.scale == (2.0, 2.0, 0.25)
    assert policy.joint_position.func is base_mdp.joint_pos_rel
    assert policy.joint_velocity.func is base_mdp.joint_vel_rel
    assert policy.previous_action.func is base_mdp.last_action
    assert policy.second_previous_action.func is symm_quadruped.second_previous_action
    assert policy.gait_period.func is symm_quadruped.dimensionless_gait_period
    assert policy.duty_factor.func is symm_quadruped.duty_factor
    assert policy.history_length == 30
    assert policy.flatten_history_dim is True


@pytest.mark.parametrize("env_cfg_type", [UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg])
def test_policy_observation_contract_is_64d_and_in_authoritative_order(env_cfg_type):
    policy = env_cfg_type().observations.policy
    contract = symm_quadruped.SYMM_QUADRUPED_POLICY_OBSERVATION_CONTRACT
    configured_terms = tuple(name for name, value in policy.__dict__.items() if isinstance(value, ObservationTermCfg))

    assert contract.version == "hardware_proprio_history_64d_v1"
    assert contract.history_packing == "term_major_oldest_to_newest_flattened"
    assert contract.instantaneous_frame_dim == 64
    assert configured_terms == contract.term_order
    assert sum(getattr(contract.dimensions, name) for name in configured_terms) == 64
    assert policy.history_length * contract.instantaneous_frame_dim == 1920
    assert contract.slices.projected_gravity == slice(0, 3)
    assert contract.slices.velocity_command == slice(3, 6)
    assert contract.slices.joint_position == slice(6, 18)
    assert contract.slices.joint_velocity == slice(18, 30)
    assert contract.slices.previous_action == slice(30, 42)
    assert contract.slices.second_previous_action == slice(42, 54)
    assert contract.slices.gait_period == slice(54, 55)
    assert contract.slices.duty_factor == slice(55, 56)
    assert contract.slices.foot_phase_sin == slice(56, 60)
    assert contract.slices.foot_phase_cos == slice(60, 64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_policy_history_pack_unpack_round_trip_and_latest_frame(dtype):
    frames = torch.arange(2 * 3 * 5 * 64, dtype=dtype).reshape(2, 3, 5, 64)
    contract = symm_quadruped.SYMM_QUADRUPED_POLICY_OBSERVATION_CONTRACT

    packed = symm_quadruped.pack_term_major_policy_history(frames)
    manually_packed = torch.cat(
        [frames[..., getattr(contract.slices, name)].reshape(2, 3, -1) for name in contract.term_order],
        dim=-1,
    )

    assert packed.shape == (2, 3, 5 * 64)
    assert packed.dtype == dtype
    assert torch.equal(packed, manually_packed)
    assert symm_quadruped.infer_policy_history_length(packed) == 5
    assert torch.equal(symm_quadruped.unpack_term_major_policy_history(packed), frames)
    assert torch.equal(symm_quadruped.latest_policy_frame(packed), frames[..., -1, :])
    assert torch.equal(symm_quadruped.latest_policy_frame(frames[..., -1, :]), frames[..., -1, :])


@pytest.mark.parametrize("width", [0, 1, 63, 65, 1919, 1921])
def test_policy_history_rejects_invalid_flattened_width(width):
    with pytest.raises(ValueError, match="positive multiple of 64"):
        symm_quadruped.infer_policy_history_length(torch.zeros(2, width))


def test_native_history_reset_clears_only_selected_environments():
    history = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    history.append(torch.tensor([[1.0], [2.0]]))
    history.append(torch.tensor([[3.0], [4.0]]))
    preserved = history.buffer[0].clone()

    history.reset(batch_ids=[1])

    assert torch.equal(history.buffer[0], preserved)
    assert torch.equal(history.buffer[1], torch.zeros(3, 1))


def test_action_manager_exposes_previous_and_second_previous_action_values():
    class _ActionTerm:
        action_dim = 12

        def process_actions(self, actions):
            self.processed_actions = actions.clone()

        def reset(self, env_ids=None):
            self.reset_env_ids = env_ids

    manager = object.__new__(ActionManager)
    manager._resolve_terms_handle = None
    manager._env = SimpleNamespace(num_envs=2, device="cpu")
    manager._terms = {"joint_pos": _ActionTerm()}
    manager._action = torch.zeros(2, 12)
    manager._prev_action = torch.zeros(2, 12)
    env = SimpleNamespace(action_manager=manager)
    actions = [torch.full((2, 12), value) for value in (1.0, 2.0, 3.0)]

    for index, action in enumerate(actions):
        manager.process_action(action)
        expected_second_previous = torch.zeros_like(action) if index == 0 else actions[index - 1]
        assert torch.equal(base_mdp.last_action(env), action)
        assert torch.equal(symm_quadruped.second_previous_action(env), expected_second_previous)

    manager.reset(env_ids=[0])
    assert torch.equal(manager.action[0], torch.zeros(12))
    assert torch.equal(manager.prev_action[0], torch.zeros(12))
    assert torch.equal(manager.action[1], actions[-1][1])
    assert torch.equal(manager.prev_action[1], actions[-2][1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dimensionless_gait_period_uses_midpoint_characteristic_length(dtype):
    period = torch.tensor([0.45, 0.30], dtype=dtype)

    period_star = symm_quadruped.compute_dimensionless_gait_period(period, (0.35, 0.45))

    expected = period * math.sqrt(9.81 / 0.40)
    assert period_star.dtype == dtype
    assert period_star.device == period.device
    assert torch.allclose(period_star, expected)


@pytest.mark.parametrize(
    ("period", "height_range"),
    [
        (torch.tensor([0.0]), (0.35, 0.45)),
        (torch.tensor([float("nan")]), (0.35, 0.45)),
        (torch.tensor([0.45]), (0.0, 0.45)),
        (torch.tensor([0.45]), (0.45, 0.35)),
        (torch.tensor([0.45]), (0.35, float("inf"))),
    ],
)
def test_dimensionless_gait_period_rejects_invalid_inputs(period, height_range):
    with pytest.raises(ValueError):
        symm_quadruped.compute_dimensionless_gait_period(period, height_range)


def test_period_and_duty_observation_terms_preserve_shape_dtype_and_device():
    command = SimpleNamespace(
        gait_periods=torch.tensor([0.45, 0.30], dtype=torch.float64),
        duty_factors=torch.tensor([0.45, 0.60], dtype=torch.float64),
        cfg=SimpleNamespace(base_height_range=(0.35, 0.45)),
    )
    env = SimpleNamespace(
        num_envs=2,
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )

    period = symm_quadruped.dimensionless_gait_period(env, command_name="base_velocity")
    duty = symm_quadruped.duty_factor(env, command_name="base_velocity")

    assert period.shape == duty.shape == (2, 1)
    assert period.dtype == duty.dtype == torch.float64
    assert period.device == duty.device == command.gait_periods.device
    assert torch.equal(duty[:, 0], command.duty_factors)


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


def test_deprecated_foot_periodicity_function_forwards_to_foot_phase(monkeypatch):
    expected = torch.tensor([-0.25])
    captured = {}

    def fake_foot_phase(env, **kwargs):
        captured["env"] = env
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(symm_quadruped, "foot_phase_penalty", fake_foot_phase)
    monkeypatch.setattr(symm_quadruped, "_FOOT_PERIODICITY_DEPRECATION_WARNED", False)
    env = SimpleNamespace()
    feet_cfg = SimpleNamespace()
    asset_cfg = SimpleNamespace(name="test_robot")

    with pytest.warns(DeprecationWarning, match="foot_phase_penalty"):
        result = symm_quadruped.foot_periodicity_penalty(
            env,
            command_name="base_velocity",
            feet_cfg=feet_cfg,
            foot_sensor_names=("contact_FL",),
            foot_sensor_body_names=("FL_foot",),
            force_scale=0.005,
            asset_cfg=asset_cfg,
        )

    assert result is expected
    assert captured["env"] is env
    assert captured["command_name"] == "base_velocity"
    assert captured["feet_cfg"] is feet_cfg
    assert captured["foot_sensor_names"] == ("contact_FL",)
    assert captured["foot_sensor_body_names"] == ("FL_foot",)
    assert captured["force_scale"] == 0.005
    assert captured["asset_cfg"] is asset_cfg


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


def test_requested_joint_targets_are_cached_before_the_execution_clamp():
    env = SimpleNamespace()
    action_term = SimpleNamespace(
        _joint_ids=[0, 1, 2],
        _offset=torch.tensor([[0.0, 0.5, -1.0]]),
        _scale=0.25,
        raw_actions=torch.tensor([[0.0, 4.0, 12.0]]),
        processed_actions=torch.tensor([[0.0, 1.5, 2.0]]),
    )
    env.action_manager = SimpleNamespace(get_term=lambda _: action_term)
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            soft_joint_pos_limits=_tensor_data(torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-2.0, 0.0]]]))
        )
    )
    env.scene = scene

    symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv._clamp_processed_joint_position_targets(env)

    assert torch.equal(env._requested_joint_position_targets, torch.tensor([[0.0, 1.5, 2.0]]))
    assert torch.equal(action_term.processed_actions, torch.tensor([[0.0, 1.0, 0.0]]))
    assert env._joint_target_clipped_fraction.item() == pytest.approx(2.0 / 3.0)


def test_feasible_action_metadata_follows_resolved_joint_ids():
    env = object.__new__(symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv)
    env._is_closed = True
    action_term = SimpleNamespace(
        _joint_ids=[2, 0],
        _offset=torch.tensor([[0.25, -0.50]]),
        _scale=torch.tensor([[0.5, -0.25]]),
    )
    env.action_manager = SimpleNamespace(get_term=lambda _: action_term)
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            soft_joint_pos_limits=_tensor_data(torch.tensor([[[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]]))
        )
    )
    env.scene = scene

    offset, scale, limits = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.get_joint_position_action_metadata(env)
    lower, upper = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.get_joint_position_action_feasible_bounds(env)

    assert torch.equal(offset, action_term._offset)
    assert torch.equal(scale, action_term._scale)
    assert torch.equal(limits, torch.tensor([[[-3.0, 3.0], [-1.0, 1.0]]]))
    assert torch.allclose(lower, torch.tensor([[-6.5, -6.0]]))
    assert torch.allclose(upper, torch.tensor([[5.5, 2.0]]))


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
    env = SimpleNamespace(_capture_rollout_diagnostics=True)
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
            heading_w=_tensor_data(torch.full((3,), 2.5)),
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
    env.reward_manager = SimpleNamespace(
        get_term_cfg=lambda name: (
            SimpleNamespace(params={"reduction": "mean"}, weight=0.4)
            if name == "foot_phase"
            else SimpleNamespace(params={"margin_fraction": 0.05}, weight=0.05)
        )
    )
    env._foot_phase_diagnostics = {
        "raw_sum": torch.tensor([1.0, 2.0, 3.0]),
        "raw_mean": torch.tensor([0.25, 0.50, 0.75]),
    }
    command_term = SimpleNamespace(
        command=torch.full((3, 3), 6.0),
        foot_thetas=torch.full((3, 4), 0.25),
        gait_periods=torch.full((3,), 0.4),
        duty_factors=torch.full((3,), 0.6),
        common_gait_phases=lambda: torch.full((3,), 1.5),
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
        "mean_swing_height": torch.tensor([0.04, 0.05, 0.06]),
        "mean_target_height": torch.tensor([0.08, 0.08, 0.08]),
        "mean_shortfall": torch.tensor([0.04, 0.03, 0.02]),
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
    assert diagnostics["Diagnostics/reward_clipped_fraction"].item() == pytest.approx(1.0 / 3.0)
    assert diagnostics["Diagnostics/joint_near_limit_fraction"].item() == pytest.approx(1.0 / 6.0)
    assert diagnostics["Diagnostics/joint_target_limit_violation_fraction"].item() == pytest.approx(1.0 / 6.0)
    assert diagnostics["Diagnostics/requested_target_overflow_fraction"].item() == pytest.approx(1.0 / 6.0)
    assert diagnostics["Diagnostics/executed_target_clipped_fraction"].item() == 0.0
    assert diagnostics["Diagnostics/foot_phase_weight"].item() == pytest.approx(0.4)
    assert diagnostics["Diagnostics/foot_phase_reduction"].item() == 1.0
    assert diagnostics["Diagnostics/foot_phase_sum_equivalent_weight"].item() == pytest.approx(0.1)
    assert diagnostics["Diagnostics/foot_phase_per_foot_effective_weight"].item() == pytest.approx(0.1)
    assert diagnostics["Diagnostics/raw_foot_phase_sum"].item() == pytest.approx(2.0)
    assert diagnostics["Diagnostics/raw_foot_phase_mean"].item() == pytest.approx(0.5)
    assert diagnostics["Diagnostics/weighted_foot_phase"].item() == pytest.approx(-0.2)
    assert diagnostics["Diagnostics/straight_line_forward_score"].item() == pytest.approx(0.5)
    assert diagnostics["Diagnostics/foot_clearance_mean_swing_height"].item() == pytest.approx(0.05)
    assert diagnostics["Diagnostics/foot_clearance_mean_shortfall"].item() == pytest.approx(0.03)
    assert torch.equal(env._last_policy_actions, actions)
    assert torch.equal(env._last_joint_position_targets, joint_target)
    assert torch.equal(env._last_requested_joint_position_targets, joint_target)
    assert torch.equal(env._last_joint_velocities, torch.zeros_like(joint_pos))
    assert torch.equal(env._last_joint_torques, torch.ones_like(joint_pos))
    assert torch.equal(env._last_root_positions_w, torch.full((3, 2), 2.0))
    assert torch.equal(env._last_root_headings_w, torch.full((3,), 2.5))
    assert torch.equal(env._last_root_lin_velocities_b, torch.full((3, 3), 3.0))
    assert torch.equal(env._last_root_ang_velocities_b, torch.full((3, 3), 4.0))
    assert torch.equal(env._last_base_velocity_commands, torch.full((3, 3), 6.0))
    assert torch.equal(env._last_foot_thetas, torch.full((3, 4), 0.25))
    assert torch.equal(env._last_gait_periods, torch.full((3,), 0.4))
    assert torch.equal(env._last_duty_factors, torch.full((3,), 0.6))
    assert torch.equal(env._last_common_gait_phases, torch.full((3,), 1.5))
    assert torch.equal(env._last_periodic_force_weights, torch.full((3, 4), 7.0))
    assert torch.equal(env._last_periodic_speed_weights, torch.full((3, 4), 8.0))
    assert torch.equal(env._last_foot_velocities_w, torch.full((3, 4, 3), 5.0))
    assert env._last_foot_normal_forces_w.shape == (3, 4, 3)
    assert torch.equal(env._last_foot_normal_forces_w[0, :, 2], torch.arange(1.0, 5.0))
    assert torch.equal(env._last_foot_ground_reaction_forces_w[0, :, 0], 0.5 * torch.arange(1.0, 5.0))
    assert env._last_foot_normal_force_is_ground_filtered.tolist() == [True, True, True]
    assert env._last_ground_reaction_force_includes_friction

    def get_cfg_without_target_limit(name):
        if name == "joint_target_limits":
            raise ValueError("Reward term 'joint_target_limits' not found.")
        return SimpleNamespace(params={"reduction": "mean"}, weight=0.4)

    env.reward_manager = SimpleNamespace(get_term_cfg=get_cfg_without_target_limit)
    diagnostics_without_target_reward = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv._compute_step_diagnostics(
        env,
        actions,
        torch.tensor([-0.2, 0.0, 0.1]),
    )

    assert diagnostics_without_target_reward["Diagnostics/requested_target_overflow_fraction"].item() == pytest.approx(
        1.0 / 6.0
    )


def test_gait_command_uses_fixed_zero_yaw_rate():
    command_cfg = make_gait_velocity_command(symm_quadruped)

    assert not command_cfg.heading_command
    assert command_cfg.ranges.ang_vel_z == (0.0, 0.0)


def test_rewards_use_straight_line_motion_reward_and_restore_hip_action_penalty():
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

    assert env_cfg.rewards.hip_action_penalty.weight == 0.10
    assert env_cfg.rewards.alive_bonus.weight == 0.20
    assert env_cfg.rewards.foot_phase.func is symm_quadruped.foot_phase_penalty
    assert env_cfg.rewards.foot_phase.weight == 0.30
    assert env_cfg.rewards.foot_phase.params["reduction"] == "sum"
    assert "foot_periodicity" not in env_cfg.rewards.__dict__
    assert env_cfg.rewards.cmd is None
    assert env_cfg.rewards.sagittal_plane is None
    assert env_cfg.rewards.straight_line_motion.func is symm_quadruped.straight_line_motion_reward
    assert env_cfg.rewards.straight_line_motion.weight == 1.0
    assert env_cfg.rewards.straight_line_motion.params["command_name"] == "base_velocity"
    assert env_cfg.rewards.straight_line_motion.params["min_base_height"] == 0.35
    assert env_cfg.rewards.straight_line_motion.params["support_loss_weight"] == 0.25
    assert "lateral_position_scale" not in env_cfg.rewards.straight_line_motion.params
    assert "heading_scale" not in env_cfg.rewards.straight_line_motion.params
    assert env_cfg.rewards.straight_line_motion.params["pose_weight"] == 0.0
    assert env_cfg.rewards.straight_line_motion.params["pitch_scale"] == 0.50
    assert env_cfg.rewards.termination_penalty.func is base_mdp.is_terminated
    assert env_cfg.rewards.termination_penalty.weight == -200.0
    assert env_cfg.rewards.joint_target_limits.func is symm_quadruped.joint_position_target_limit_penalty
    assert env_cfg.rewards.joint_target_limits.weight == 0.05
    assert env_cfg.rewards.joint_target_limits.params["mode"] == "legacy_clamped"
    assert env_cfg.rewards.leg_permutation_symmetry.func is symm_quadruped.leg_permutation_symmetry_penalty
    assert env_cfg.rewards.leg_permutation_symmetry.weight == 0.20
    assert env_cfg.rewards.leg_permutation_symmetry.params["phase_sync_tolerance"] == 0.02
    assert env_cfg.rewards.foot_clearance.weight == 0.10
    assert env_cfg.rewards.foot_clearance.params["min_height"] == 0.08
    assert env_cfg.rewards.foot_clearance.params["height_scale"] == 0.05
    assert env_cfg.rewards.foot_clearance.params["min_command_speed"] == 0.20


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


def test_rewards_cfg_preserves_deprecated_foot_periodicity_alias():
    cfg = SymmQuadrupedRewardsCfg()
    reward = object()
    cfg.foot_phase = reward

    with pytest.warns(DeprecationWarning, match="foot_phase"):
        assert cfg.foot_periodicity is reward

    replacement = object()
    with pytest.warns(DeprecationWarning, match="foot_phase"):
        cfg.foot_periodicity = replacement

    assert cfg.foot_phase is replacement
    assert "foot_periodicity" not in cfg.__dict__
    serialized = cfg.to_dict()
    assert "foot_phase" in serialized
    assert "foot_periodicity" not in serialized

    configured_rewards = DobotX1SymmFlatEnvCfg().rewards
    with pytest.warns(DeprecationWarning, match="foot_phase"):
        configured_rewards.from_dict({"foot_periodicity": {"weight": 0.42}})
    assert configured_rewards.foot_phase.weight == 0.42


def test_robot_configs_use_canonical_foot_phase_name_without_deprecation_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*foot_periodicity.*", category=DeprecationWarning)
        for cfg in (DobotX1SymmFlatEnvCfg(), UnitreeGo2SymmFlatEnvCfg()):
            serialized = cfg.rewards.to_dict()
            assert "foot_phase" in serialized
            assert "foot_periodicity" not in serialized


def test_robot_configs_use_robot_specific_foot_clearance_shaping():
    x1_cfg = DobotX1SymmFlatEnvCfg()
    go2_cfg = UnitreeGo2SymmFlatEnvCfg()

    assert x1_cfg.rewards.foot_phase.func is symm_quadruped.foot_phase_penalty
    assert go2_cfg.rewards.foot_phase.func is symm_quadruped.foot_phase_penalty
    assert x1_cfg.rewards.foot_phase.weight == 0.30
    assert go2_cfg.rewards.foot_phase.weight == 0.30
    assert x1_cfg.rewards.foot_phase.params["reduction"] == "sum"
    assert go2_cfg.rewards.foot_phase.params["reduction"] == "sum"
    assert x1_cfg.rewards.joint_target_limits.weight == 0.05
    assert go2_cfg.rewards.joint_target_limits.weight == 0.05
    assert x1_cfg.rewards.joint_target_limits.params["mode"] == "legacy_clamped"
    assert go2_cfg.rewards.joint_target_limits.params["mode"] == "legacy_clamped"
    assert x1_cfg.rewards.hip_action_penalty.weight == 0.10
    assert go2_cfg.rewards.hip_action_penalty.weight == 0.10
    assert x1_cfg.rewards.leg_permutation_symmetry.func is dobot_x1_symm.leg_permutation_symmetry_penalty
    assert x1_cfg.rewards.leg_permutation_symmetry.weight == 0.20
    assert go2_cfg.rewards.leg_permutation_symmetry.weight == 0.20
    assert x1_cfg.rewards.leg_permutation_symmetry.params["phase_sync_tolerance"] == 0.02
    assert x1_cfg.rewards.foot_clearance.func is symm_quadruped.foot_clearance_penalty
    assert x1_cfg.rewards.foot_clearance.params["min_height"] == 0.04
    assert x1_cfg.rewards.foot_clearance.params["height_scale"] == 0.025
    assert go2_cfg.rewards.foot_clearance.func is symm_quadruped.foot_clearance_tracking_reward
    assert go2_cfg.rewards.foot_clearance.weight == 0.15
    assert go2_cfg.rewards.foot_clearance.params["target_height"] == 0.08
    assert go2_cfg.rewards.foot_clearance.params["excess_height_margin"] == 0.03


@pytest.mark.parametrize(
    ("env_cfg_type", "expected_joint_order", "expected_lower", "expected_upper"),
    (
        (
            UnitreeGo2SymmFlatEnvCfg,
            (
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ),
            (
                -3.76992,
                -8.4709,
                -4.513812,
                -3.76992,
                -8.4709,
                -4.513812,
                -3.76992,
                -4.2821,
                -4.513812,
                -3.76992,
                -4.2821,
                -4.513812,
            ),
            (
                3.76992,
                9.7505,
                2.271972,
                3.76992,
                9.7505,
                2.271972,
                3.76992,
                13.9393,
                2.271972,
                3.76992,
                13.9393,
                2.271972,
            ),
        ),
        (
            DobotX1SymmFlatEnvCfg,
            (
                "joint_front_left_abad",
                "joint_front_left_thigh_pitch",
                "joint_front_left_calf_pitch",
                "joint_front_right_abad",
                "joint_front_right_thigh_pitch",
                "joint_front_right_calf_pitch",
                "joint_rear_left_abad",
                "joint_rear_left_thigh_pitch",
                "joint_rear_left_calf_pitch",
                "joint_rear_right_abad",
                "joint_rear_right_thigh_pitch",
                "joint_rear_right_calf_pitch",
            ),
            (
                -2.38752,
                -12.218,
                -3.9712,
                -2.38752,
                -12.218,
                -3.9712,
                -2.38752,
                -6.6316,
                -14.2448,
                -2.38752,
                -6.6316,
                -14.2448,
            ),
            (
                2.38752,
                6.6316,
                14.2448,
                2.38752,
                6.6316,
                14.2448,
                2.38752,
                12.218,
                3.9712,
                2.38752,
                12.218,
                3.9712,
            ),
        ),
    ),
)
def test_robot_urdf_limits_resolve_to_expected_per_joint_raw_action_bounds(
    env_cfg_type,
    expected_joint_order,
    expected_lower,
    expected_upper,
):
    cfg = env_cfg_type()
    joint_order = tuple(cfg.actions.joint_pos.joint_names)
    assert joint_order == expected_joint_order
    assert cfg.actions.joint_pos.preserve_order
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.scene.robot.soft_joint_pos_limit_factor == 0.9

    urdf_root = ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
    urdf_limits = {
        joint.attrib["name"]: (float(joint.find("limit").attrib["lower"]), float(joint.find("limit").attrib["upper"]))
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("type") == "revolute"
    }
    default_joint_positions = []
    for joint_name in joint_order:
        matches = [
            float(value)
            for name_pattern, value in cfg.scene.robot.init_state.joint_pos.items()
            if re.fullmatch(name_pattern, joint_name)
        ]
        assert len(matches) == 1, f"Expected exactly one default-position match for {joint_name}, received {matches}."
        default_joint_positions.append(matches[0])

    hard_limits = torch.tensor([[urdf_limits[joint_name] for joint_name in joint_order]])
    limit_midpoint = hard_limits.mean(dim=-1, keepdim=True)
    limits = limit_midpoint + cfg.scene.robot.soft_joint_pos_limit_factor * (hard_limits - limit_midpoint)
    offset = torch.tensor([default_joint_positions])
    lower, upper = symm_quadruped_env.compute_feasible_raw_action_bounds(
        offset,
        cfg.actions.joint_pos.scale,
        limits,
    )

    assert lower[0].tolist() == pytest.approx(expected_lower, abs=1.0e-5)
    assert upper[0].tolist() == pytest.approx(expected_upper, abs=1.0e-5)
    calf_indices = [2, 5, 8, 11]
    if env_cfg_type is UnitreeGo2SymmFlatEnvCfg:
        assert lower[0, calf_indices].tolist() == pytest.approx([-4.513812] * 4, abs=1.0e-5)
        assert upper[0, calf_indices].tolist() == pytest.approx([2.271972] * 4, abs=1.0e-5)


def test_play_configs_enable_ground_filtered_normal_and_friction_forces():
    for cfg in (DobotX1SymmFlatEnvCfg_PLAY(), UnitreeGo2SymmFlatEnvCfg_PLAY()):
        for sensor_name in ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot"):
            sensor_cfg = getattr(cfg.scene, sensor_name)
            assert sensor_cfg.filter_prim_paths_expr == [SYMM_QUADRUPED_GROUND_COLLISION_PATH]
            assert sensor_cfg.track_friction_forces


def test_play_configs_enable_fixed_six_gait_sequence_only_for_playback():
    for train_cfg, play_cfg in (
        (DobotX1SymmFlatEnvCfg(), DobotX1SymmFlatEnvCfg_PLAY()),
        (UnitreeGo2SymmFlatEnvCfg(), UnitreeGo2SymmFlatEnvCfg_PLAY()),
    ):
        train_command = train_cfg.commands.base_velocity
        play_command = play_cfg.commands.base_velocity

        assert train_command.gait_library_version == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION
        assert train_command.init_foot_thetas == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS
        assert train_command.init_foot_theta_weights == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS
        assert train_command.gait_sampling_profile == symm_quadruped.SYMM_QUADRUPED_GAIT_SAMPLING_PROFILE_EQUAL_FAMILY
        assert train_command.gait_curriculum_iterations == 0
        assert train_cfg.commands.base_velocity.gait_sequence_enabled is False
        assert play_command.gait_library_version == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_VERSION
        expected_play_gaits = (
            (0.0, 0.5, 0.5, 0.0),
            (0.0, 0.0, 0.5, 0.5),
            (0.13, -0.13, 0.5, 0.5),
            (0.0, 0.0, 0.63, 0.37),
            (-0.13, 0.13, 0.63, 0.37),
            (0.13, -0.13, 0.63, 0.37),
        )
        assert play_command.init_foot_thetas == expected_play_gaits
        assert expected_play_gaits == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_ROWS
        assert play_command.init_foot_theta_weights is None
        assert play_command.gait_sequence_enabled is True
        assert play_command.gait_sequence_duration_s == 5.0
        assert play_command.add_noise_period is True
        assert play_command.add_noise_theta is False
        assert len(play_command.init_foot_thetas) == 6
        assert play_cfg.terminations.time_out is not None
        assert play_cfg.episode_length_s == pytest.approx(30.02)


@pytest.mark.parametrize("iteration", (0, 1, 49, 100, 1000))
def test_gait_sampling_profiles_preserve_partner_weight_equality(iteration):
    profiles_and_durations = (
        (symm_quadruped.SYMM_QUADRUPED_GAIT_SAMPLING_PROFILE_EQUAL_FAMILY, 0),
        (symm_quadruped.SYMM_QUADRUPED_GAIT_SAMPLING_PROFILE_V1_EQUIVALENT, 0),
        (symm_quadruped.SYMM_QUADRUPED_GAIT_SAMPLING_PROFILE_HALFBOUND_ANNEAL, 100),
    )
    for profile, duration in profiles_and_durations:
        weights = symm_quadruped.resolve_gait_sampling_profile_weights(
            profile,
            iteration=iteration,
            curriculum_iterations=duration,
        )
        for row_index, partner_index in enumerate(
            symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS
        ):
            assert weights[row_index] == pytest.approx(weights[partner_index])


def test_gait_sampling_curriculum_endpoints_and_absolute_iteration_resume():
    profile = symm_quadruped.SYMM_QUADRUPED_GAIT_SAMPLING_PROFILE_HALFBOUND_ANNEAL
    start = symm_quadruped.resolve_gait_sampling_profile_weights(
        profile,
        iteration=0,
        curriculum_iterations=100,
    )
    middle_before_resume = symm_quadruped.resolve_gait_sampling_profile_weights(
        profile,
        iteration=37,
        curriculum_iterations=100,
    )
    middle_after_resume = symm_quadruped.resolve_gait_sampling_profile_weights(
        profile,
        iteration=37,
        curriculum_iterations=100,
    )
    final = symm_quadruped.resolve_gait_sampling_profile_weights(
        profile,
        iteration=100,
        curriculum_iterations=100,
    )
    after_final = symm_quadruped.resolve_gait_sampling_profile_weights(
        profile,
        iteration=175,
        curriculum_iterations=100,
    )

    assert start == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_V1_EQUIVALENT_WEIGHTS
    assert middle_before_resume == middle_after_resume
    assert middle_before_resume[4] == pytest.approx(0.37)
    assert middle_before_resume[2] == pytest.approx(1.63)
    assert final == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS
    assert after_final == final
    assert all(weight > 0.0 for weight in final)


def test_gait_command_curriculum_publishes_row_and_family_probabilities():
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term._env = SimpleNamespace(device="cpu", num_envs=3)
    command_term.cfg = SimpleNamespace(
        gait_sampling_profile=symm_quadruped.SYMM_QUADRUPED_GAIT_SAMPLING_PROFILE_HALFBOUND_ANNEAL,
        gait_library_version=symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION,
        gait_curriculum_iterations=100,
    )
    command_term.init_foot_thetas = torch.tensor(symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS)
    command_term._gait_row_metric_names = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES
    command_term._gait_row_metric_families = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES
    command_term.metrics = {
        **{f"gait_row_probability_{row_name}": torch.zeros(3) for row_name in command_term._gait_row_metric_names},
        **{
            f"gait_family_probability_{family_name}": torch.zeros(3)
            for family_name in set(command_term._gait_row_metric_families)
        },
    }
    command_term.foot_theta_sampling_weights = None
    command_term._training_iteration = 0

    command_term.set_training_iteration(50)

    expected_weights = torch.tensor((4.0, 4.0, 1.5, 1.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0))
    expected_probabilities = expected_weights / expected_weights.sum()
    assert command_term.training_iteration == 50
    assert torch.allclose(command_term.foot_theta_sampling_weights, expected_probabilities)
    for row_index, row_name in enumerate(command_term._gait_row_metric_names):
        assert torch.equal(
            command_term.metrics[f"gait_row_probability_{row_name}"],
            torch.full((3,), expected_probabilities[row_index]),
        )
    for family_name in ("trot", "bound", "half_bound", "gallop"):
        assert torch.equal(
            command_term.metrics[f"gait_family_probability_{family_name}"],
            torch.full((3,), 0.25),
        )


def test_robot_configs_use_requested_pitch_and_height_postures():
    x1_cfg = DobotX1SymmFlatEnvCfg()
    go2_cfg = UnitreeGo2SymmFlatEnvCfg()

    assert go2_cfg.rewards.straight_line_motion.params["pitch_scale"] == 0.50
    assert go2_cfg.terminations.base_orientation.params["max_pitch"] == 1.20
    assert x1_cfg.rewards.straight_line_motion.params["pitch_scale"] == 0.35
    assert x1_cfg.terminations.base_orientation.params["max_pitch"] == 0.70

    assert x1_cfg.commands.base_velocity.base_height_range == (0.45, 0.60)
    assert x1_cfg.rewards.base_height.params["height_range"] == (0.45, 0.60)
    assert x1_cfg.rewards.straight_line_motion.params["min_base_height"] == 0.45
    assert x1_cfg.scene.robot.init_state.pos == (0.0, 0.0, 0.5)
    default_joint_pos = x1_cfg.scene.robot.init_state.joint_pos
    assert default_joint_pos["joint_front_left_abad"] == 0.0
    assert default_joint_pos["joint_front_left_thigh_pitch"] == 0.6983
    assert default_joint_pos["joint_front_left_calf_pitch"] == -1.2842
    assert default_joint_pos["joint_rear_left_abad"] == 0.0
    assert default_joint_pos["joint_rear_left_thigh_pitch"] == -0.6983
    assert default_joint_pos["joint_rear_left_calf_pitch"] == 1.2842


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
    gait_command = SimpleNamespace(
        duty_factors=torch.full((3,), 0.45),
        kappa=torch.full((3,), 16.0),
        foot_phases=lambda: foot_phases,
    )
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
        pose_weight=0.0,
        roll_scale=0.25,
        pitch_scale=0.35,
        min_base_height=0.35,
        height_scale=0.10,
    )

    assert torch.allclose(reward[:2], torch.full((2,), 1.45))
    assert 0.40 < reward[2] < 0.50
    assert reward[3] < reward[0] - 0.20
    assert reward[4] < reward[0] - 0.20
    assert torch.all(reward >= -0.25)
    assert torch.all(reward <= 1.45)
    assert set(env._straight_line_motion_diagnostics) == {
        "forward_score",
        "lateral_velocity_score",
        "yaw_rate_score",
        "roll_score",
        "straight_score",
        "posture_score",
        "support_loss",
        "reward",
    }
    assert torch.equal(env._straight_line_motion_diagnostics["reward"], reward)
    assert all(not value.requires_grad for value in env._straight_line_motion_diagnostics.values())


def test_straight_line_motion_reward_ignores_world_lateral_position_and_heading():
    scene = _Scene()
    scene.env_origins = torch.zeros(2, 3)
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.4], [0.0, 20.0, 0.4]])),
            root_quat_w=_tensor_data(torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])),
            heading_w=_tensor_data(torch.tensor([0.0, math.pi])),
            root_lin_vel_b=_tensor_data(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
            root_ang_vel_b=_tensor_data(torch.zeros(2, 3)),
        )
    )
    command = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: command))

    reward = symm_quadruped.straight_line_motion_reward(env, command_name="base_velocity")

    assert torch.equal(reward[0], reward[1])


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


def test_foot_phase_mean_and_sum_coefficients_have_explicit_equivalent_semantics():
    per_foot_penalties = torch.tensor(
        [
            [0.0, 0.25, 0.50, 1.0],
            [0.2, 0.4, 0.6, 0.8],
        ]
    )

    weighted_mean = -0.4 * symm_quadruped.reduce_foot_phase_penalties(per_foot_penalties, reduction="mean")
    weighted_sum = -0.1 * symm_quadruped.reduce_foot_phase_penalties(per_foot_penalties, reduction="sum")

    assert torch.equal(weighted_mean, weighted_sum)


def test_requested_target_overflow_penalty_uses_preclamp_request_and_is_uncapped():
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]])
    requested_targets = torch.tensor([[0.0, 1.0, 5.0]])
    executed_targets = requested_targets.clamp(-1.0, 1.0)
    action_term = SimpleNamespace(
        _joint_ids=[0, 1, 2],
        _offset=0.0,
        _scale=1.0,
        raw_actions=requested_targets,
        processed_actions=executed_targets,
    )
    scene = _Scene()
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=_tensor_data(limits)))
    env = SimpleNamespace(
        scene=scene,
        action_manager=SimpleNamespace(get_term=lambda _: action_term),
        _requested_joint_position_targets=requested_targets,
    )

    normalized_overflow = symm_quadruped.requested_joint_position_target_overflow(
        requested_targets,
        limits,
        margin_fraction=0.1,
    )
    requested_penalty = symm_quadruped.joint_position_target_limit_penalty(
        env,
        margin_fraction=0.1,
        mode="requested_overflow",
    )
    legacy_penalty = symm_quadruped.joint_position_target_limit_penalty(
        env,
        margin_fraction=0.1,
        mode="legacy_clamped",
    )

    assert normalized_overflow.tolist()[0] == pytest.approx([0.0, 0.1, 2.1])
    assert normalized_overflow[0, 2].item() > 1.0
    assert requested_penalty.item() == pytest.approx(-(0.0 + 0.005 + 1.6) / 3.0)
    assert legacy_penalty.item() == pytest.approx(-(0.0 + 1.0 + 1.0) / 3.0)


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


def test_domain_randomization_includes_lateral_and_yaw_velocity_disturbances():
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

    reset_velocity_range = env_cfg.events.reset_base.params["velocity_range"]
    push_velocity_range = env_cfg.events.push_robot.params["velocity_range"]
    assert reset_velocity_range["y"] == reset_velocity_range["x"] == (-0.5, 0.5)
    assert reset_velocity_range["yaw"] == reset_velocity_range["roll"] == (-0.5, 0.5)
    assert push_velocity_range["y"] == push_velocity_range["x"] == (-0.25, 0.25)
