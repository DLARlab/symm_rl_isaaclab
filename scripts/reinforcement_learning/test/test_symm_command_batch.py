# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check fixed command assignments and complete per-environment rollout exports."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_symm_rollout_plotter import _make_env


@pytest.fixture
def batch_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "rsl_rl"))
    import symm_command_batch

    return symm_command_batch


def _make_batch_env(num_envs):
    env, command, robot = _make_env()
    env.num_envs = num_envs
    env.device = "cpu"
    for name in (
        "root_pos_w",
        "heading_w",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "body_lin_vel_w",
        "joint_pos",
        "joint_vel",
        "applied_torque",
        "soft_joint_pos_limits",
    ):
        proxy = getattr(robot.data, name)
        proxy.torch = proxy.torch.expand(num_envs, *proxy.torch.shape[1:]).clone()
    action = env.action_manager.get_term("joint_pos")
    action.processed_actions = action.processed_actions.expand(num_envs, -1).clone()
    env.action_manager.action = env.action_manager.action.expand(num_envs, -1).clone()
    command.vel_command_b = command.command.expand(num_envs, -1).clone()
    command.command = command.vel_command_b
    command.is_standing_env = torch.ones(num_envs, dtype=torch.bool)
    command.is_heading_env = torch.ones(num_envs, dtype=torch.bool)
    for diagnostics in (env._straight_line_motion_diagnostics, env._foot_clearance_diagnostics):
        for name, value in diagnostics.items():
            diagnostics[name] = value.expand(num_envs, *value.shape[1:]).clone()
    env._running_reward_negative = torch.zeros(num_envs, dtype=torch.bool)
    return env, command, robot


def _cache_step(env, command, robot):
    for target, source in (
        ("_last_joint_positions", "joint_pos"),
        ("_last_joint_velocities", "joint_vel"),
        ("_last_joint_torques", "applied_torque"),
        ("_last_soft_joint_pos_limits", "soft_joint_pos_limits"),
        ("_last_root_lin_velocities_b", "root_lin_vel_b"),
        ("_last_root_ang_velocities_b", "root_ang_vel_b"),
        ("_last_foot_velocities_w", "body_lin_vel_w"),
    ):
        setattr(env, target, getattr(robot.data, source).torch.clone())
    env._last_root_positions_w = robot.data.root_pos_w.torch[:, :2].clone()
    env._last_joint_position_targets = env.action_manager.get_term("joint_pos").processed_actions.clone()
    env._last_base_velocity_commands = command.command.clone()
    env._last_periodic_force_weights = command.periodic_force_weights().expand(env.num_envs, -1).clone()
    env._last_periodic_speed_weights = command.periodic_speed_weights().expand(env.num_envs, -1).clone()
    env._last_foot_normal_forces_w = (
        torch.arange(12, dtype=torch.float32).reshape(1, 4, 3).expand(env.num_envs, -1, -1).clone()
    )
    env._last_foot_ground_reaction_forces_w = env._last_foot_normal_forces_w + 0.5
    env._last_ground_reaction_force_includes_friction = True


def test_default_assigns_600_environments_and_preserves_commands_after_reset(batch_module, tmp_path):
    env, command, robot = _make_batch_env(600)
    recorder = batch_module.SymmetricCommandBatchRecorder(env, tmp_path)
    assert recorder.total_steps == 1000
    recorder.install_command_override()
    recorder.install_command_override()
    expected = torch.tensor(((1, 0, 0), (-1, 0, 0), (0, 0.5, 0), (0, -0.5, 0), (0, 0, 0.6), (0, 0, -0.6)))
    torch.testing.assert_close(command.command, expected.repeat_interleave(100, dim=0))
    reset_ids = torch.tensor([0, 101, 202, 303, 404, 505])
    command.vel_command_b[reset_ids] = 9.0
    command.is_standing_env[reset_ids] = True
    command.is_heading_env[reset_ids] = True
    command._resample_command(reset_ids)
    torch.testing.assert_close(command.command, expected.repeat_interleave(100, dim=0))
    assert not command.is_standing_env.any() and not command.is_heading_env.any()
    _cache_step(env, command, robot)
    recorder.record()
    paths = recorder.save()
    assert len(paths) == 6
    for group, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            assert data["joint_torques"].shape == (1, 100, 12)
            assert data["episode_done"].shape == (1, 100)
            np.testing.assert_array_equal(data["env_ids"], np.arange(group * 100, (group + 1) * 100))
            np.testing.assert_allclose(data["assigned_command"], expected[group])
            np.testing.assert_allclose(data["desired_lin_vel"], np.broadcast_to(expected[group].numpy(), (1, 100, 3)))
            assert data["requested_steps"].item() == 1000
            assert not data["recording_complete"].item()


def test_completed_recordings_preserve_existing_schema_and_every_environment(batch_module, tmp_path):
    env, command, robot = _make_batch_env(12)
    recorder = batch_module.SymmetricCommandBatchRecorder(env, tmp_path, envs_per_command=2, duration=0.04)
    recorder.install_command_override()
    single = batch_module.SymmetricRolloutPlotter(env, tmp_path / "single", env_index=0)
    for step in range(2):
        robot.data.applied_torque.torch[:] = torch.arange(12)[:, None] + step + 1.0
        _cache_step(env, command, robot)
        recorder.record()
        single.record()
    assert recorder.is_complete
    recorder.record()
    paths = recorder.save()
    assert len(paths) == 6
    single_data = single._prepare_data({key: np.asarray(value) for key, value in single._data.items()})
    with np.load(paths[0], allow_pickle=False) as data:
        assert set(single_data).issubset(data.files)
        assert data["recording_complete"].item()
        assert data["joint_powers"].shape == (2, 2, 12)
        for key, value in single_data.items():
            actual = (
                data[key][:, 0]
                if key in recorder._samples
                or key.endswith("_centered_moving_mean")
                or (key.startswith("leg_") and key != "leg_names")
                or key.startswith("foot_ground_reaction_force_abs")
                or key
                in ("velocity_tracking_signed_error", "velocity_tracking_abs_error", "velocity_tracking_squared_error")
                else data[key]
            )
            if value.dtype.kind in "US":
                np.testing.assert_array_equal(actual, value)
            else:
                np.testing.assert_allclose(actual, value, atol=1e-6, equal_nan=True, err_msg=key)
    for group, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            for local_id in range(2):
                global_id = 2 * group + local_id
                np.testing.assert_allclose(data["joint_torques"][:, local_id, 0], [global_id + 1, global_id + 2])


def test_terminal_samples_use_pre_reset_state_and_smoothing_stays_within_each_episode(batch_module, tmp_path):
    env, command, robot = _make_batch_env(12)
    recorder = batch_module.SymmetricCommandBatchRecorder(env, tmp_path, envs_per_command=2, duration=0.06)
    recorder.install_command_override()
    for step, torque in enumerate((1.0, 2.0, 100.0)):
        robot.data.applied_torque.torch[0] = torque
        robot.data.applied_torque.torch[1] = (step + 1) * 10.0
        _cache_step(env, command, robot)
        done = torch.zeros(12, dtype=torch.bool)
        if step == 1:
            done[0] = True
            robot.data.applied_torque.torch[0] = -999
            command.command[0] = 9
            command._resample_command(torch.tensor([0]))
        recorder.record(dones=done)
    recorder.save()
    with np.load(tmp_path / "forward/sim_data.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["episode_done"][:, 0], [False, True, False])
        np.testing.assert_allclose(data["joint_torques"][:, 0, 0], [1, 2, 100])
        np.testing.assert_allclose(
            data["leg_joint_torque_magnitudes_centered_moving_mean"][:, 0, 0, 0], [1.5, 1.5, 100]
        )
        np.testing.assert_allclose(data["leg_joint_torque_magnitudes_centered_moving_mean"][:, 1, 0, 0], [20, 20, 20])


@pytest.mark.parametrize(
    "options",
    [{"duration": 0}, {"duration": 0.025}, {"duration": float("nan")}, {"forward_speed": -1}, {"envs_per_command": 1}],
)
def test_invalid_batch_configuration_is_rejected(batch_module, tmp_path, options):
    env, _, _ = _make_batch_env(600)
    with pytest.raises(ValueError):
        batch_module.SymmetricCommandBatchRecorder(env, tmp_path, **options)


def test_command_drift_is_not_silently_recorded(batch_module, tmp_path):
    env, command, robot = _make_batch_env(6)
    recorder = batch_module.SymmetricCommandBatchRecorder(env, tmp_path, envs_per_command=1, duration=0.02)
    recorder.install_command_override()
    _cache_step(env, command, robot)
    env._last_base_velocity_commands[3, 0] = 1
    with pytest.raises(RuntimeError, match="command changed"):
        recorder.record()
