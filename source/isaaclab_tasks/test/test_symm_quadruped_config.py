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
    assert cfg.algorithm.symmetry_cfg.command_observation_index == 9
    assert cfg.algorithm.symmetry_cfg.min_abs_command_velocity == 0.0


@pytest.mark.parametrize("env_cfg_type", [UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg])
def test_symm_quadruped_command_and_phase_mapping_configuration_is_stable(env_cfg_type):
    command_cfg = env_cfg_type().commands.base_velocity

    assert command_cfg.ranges.lin_vel_x == (-2.0, 2.0)
    assert command_cfg.min_xy_command_norm == 0.0
    assert command_cfg.phase_mapping_version == symm_quadruped.SYMM_QUADRUPED_PHASE_MAPPING_VERSION
    assert command_cfg.phase_mapping_version == "same_gait_backward_duty_aware_trs_v2"


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


def test_policy_observations_include_velocity_and_observable_sagittal_state():
    env_cfg = UnitreeGo2SymmFlatEnvCfg()
    policy = env_cfg.observations.policy

    assert policy.base_lin_vel.func is base_mdp.base_lin_vel
    assert policy.base_lin_vel.scale == (2.0, 2.0, 2.0)
    assert policy.base_ang_vel.func is base_mdp.base_ang_vel
    assert policy.base_ang_vel.scale == (0.25, 0.25, 0.25)
    assert policy.velocity_commands.func is symm_quadruped.desired_base_twist
    assert policy.velocity_commands.scale == (2.0, 2.0, 2.0, 0.25, 0.25, 0.25)
    assert policy.sagittal_plane_state.func is symm_quadruped.sagittal_plane_state
    assert policy.sagittal_plane_state.params == {"lateral_position_scale": 0.5}


@pytest.mark.parametrize("env_cfg_type", [UnitreeGo2SymmFlatEnvCfg, DobotX1SymmFlatEnvCfg])
def test_policy_observation_concatenation_remains_72d_and_in_original_order(env_cfg_type):
    policy = env_cfg_type().observations.policy
    term_dimensions = {
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 6,
        "joint_pos": 12,
        "joint_vel": 12,
        "actions": 12,
        "foot_phase_sin": 4,
        "foot_phase_cos": 4,
        "foot_theta_sin": 4,
        "foot_theta_cos": 4,
        "phase_ratios": 2,
        "sagittal_plane_state": 3,
    }
    configured_terms = tuple(
        name for name in policy.__dict__ if name in term_dimensions and getattr(policy, name) is not None
    )

    assert configured_terms == tuple(term_dimensions)
    assert sum(term_dimensions[name] for name in configured_terms) == symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_DIM


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
        )

    assert result is expected
    assert captured["env"] is env
    assert captured["command_name"] == "base_velocity"
    assert captured["joint_cfg"] is joint_cfg


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
    )

    assert result is expected
    assert captured["logical_joint_signs"] == dobot_x1_symm.DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS
    assert captured["joint_ranges"] == dobot_x1_symm.DOBOT_X1_SYMM_JOINT_RANGES


def test_deprecated_x1_morphological_symmetry_function_forwards(monkeypatch):
    expected = torch.tensor([-0.75])
    monkeypatch.setattr(dobot_x1_symm, "leg_permutation_symmetry_penalty", lambda *_, **__: expected)
    monkeypatch.setattr(dobot_x1_symm, "_MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED", False)

    with pytest.warns(DeprecationWarning, match="leg_permutation_symmetry_penalty"):
        result = dobot_x1_symm.morphological_symmetry_penalty(
            SimpleNamespace(),
            command_name="base_velocity",
            joint_cfg=SimpleNamespace(),
        )

    assert result is expected


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
    assert diagnostics["Diagnostics/straight_line_forward_score"].item() == pytest.approx(0.5)
    assert diagnostics["Diagnostics/foot_clearance_mean_swing_height"].item() == pytest.approx(0.05)
    assert diagnostics["Diagnostics/foot_clearance_mean_shortfall"].item() == pytest.approx(0.03)
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

    assert env_cfg.rewards.hip_action_penalty.weight == 0.15
    assert env_cfg.rewards.alive_bonus.weight == 0.20
    assert env_cfg.rewards.cmd is None
    assert env_cfg.rewards.sagittal_plane is None
    assert env_cfg.rewards.straight_line_motion.func is symm_quadruped.straight_line_motion_reward
    assert env_cfg.rewards.straight_line_motion.weight == 1.0
    assert env_cfg.rewards.straight_line_motion.params["command_name"] == "base_velocity"
    assert env_cfg.rewards.straight_line_motion.params["min_base_height"] == 0.35
    assert env_cfg.rewards.straight_line_motion.params["support_loss_weight"] == 0.25
    assert env_cfg.rewards.straight_line_motion.params["lateral_position_scale"] == 0.35
    assert env_cfg.rewards.straight_line_motion.params["heading_scale"] == 0.35
    assert env_cfg.rewards.straight_line_motion.params["pose_weight"] == 0.30
    assert env_cfg.rewards.straight_line_motion.params["pitch_scale"] == 0.50
    assert env_cfg.rewards.termination_penalty.func is base_mdp.is_terminated
    assert env_cfg.rewards.termination_penalty.weight == -200.0
    assert env_cfg.rewards.joint_target_limits.func is symm_quadruped.joint_position_target_limit_penalty
    assert env_cfg.rewards.joint_target_limits.weight == 0.05
    assert env_cfg.rewards.leg_permutation_symmetry.func is symm_quadruped.leg_permutation_symmetry_penalty
    assert env_cfg.rewards.leg_permutation_symmetry.weight == 0.30
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


def test_robot_configs_use_robot_specific_foot_clearance_shaping():
    x1_cfg = DobotX1SymmFlatEnvCfg()
    go2_cfg = UnitreeGo2SymmFlatEnvCfg()

    assert x1_cfg.rewards.foot_clearance.func is symm_quadruped.foot_clearance_penalty
    assert x1_cfg.rewards.foot_clearance.params["min_height"] == 0.04
    assert x1_cfg.rewards.foot_clearance.params["height_scale"] == 0.025
    assert go2_cfg.rewards.foot_clearance.func is symm_quadruped.foot_clearance_tracking_reward
    assert go2_cfg.rewards.foot_clearance.weight == 0.15
    assert go2_cfg.rewards.foot_clearance.params["target_height"] == 0.08
    assert go2_cfg.rewards.foot_clearance.params["excess_height_margin"] == 0.03


def test_play_configs_enable_ground_filtered_normal_and_friction_forces():
    for cfg in (DobotX1SymmFlatEnvCfg_PLAY(), UnitreeGo2SymmFlatEnvCfg_PLAY()):
        for sensor_name in ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot"):
            sensor_cfg = getattr(cfg.scene, sensor_name)
            assert sensor_cfg.filter_prim_paths_expr == [SYMM_QUADRUPED_GROUND_COLLISION_PATH]
            assert sensor_cfg.track_friction_forces


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
