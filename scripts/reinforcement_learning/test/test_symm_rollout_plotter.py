# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped rollout plotting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _load_plotter_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "symm_rollout_plotter.py"
    spec = importlib.util.spec_from_file_location("symm_rollout_plotter_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=tensor)


def _make_env():
    robot = MagicMock()
    robot.find_bodies.return_value = ([0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    robot.data.root_pos_w = _proxy(torch.tensor([[1.0, 2.0, 0.5]]))
    robot.data.heading_w = _proxy(torch.tensor([0.0]))
    robot.data.root_lin_vel_b = _proxy(torch.tensor([[0.4, 0.1, 0.0]]))
    robot.data.root_ang_vel_b = _proxy(torch.tensor([[0.0, 0.0, 0.2]]))
    robot.data.body_lin_vel_w = _proxy(
        torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.3],
                    [0.4, 0.0, 0.0],
                ]
            ]
        )
    )
    robot.data.joint_pos = _proxy(torch.linspace(-0.4, 0.4, 12).unsqueeze(0))
    robot.data.joint_vel = _proxy((0.1 * torch.arange(1, 13, dtype=torch.float32)).unsqueeze(0))
    torque_signs = torch.tensor([-1.0, 1.0] * 6)
    robot.data.applied_torque = _proxy((torque_signs * torch.arange(1, 13, dtype=torch.float32)).unsqueeze(0))
    robot.data.soft_joint_pos_limits = _proxy(torch.tensor([[[-1.0, 1.0]] * 12], dtype=torch.float32))

    command_term = SimpleNamespace(
        command=torch.tensor([[0.5, 0.0, 0.1]]),
        periodic_force_weights=lambda: torch.tensor([[-1.0, -0.5, 0.0, -0.25]]),
        periodic_speed_weights=lambda: torch.tensor([[0.0, -0.5, -1.0, -0.75]]),
        robot=robot,
    )
    command_manager = MagicMock()
    command_manager.get_term.return_value = command_term
    joint_names = [f"joint_{index}" for index in range(12)]
    joint_action_term = SimpleNamespace(
        _joint_ids=list(range(12)),
        _joint_names=joint_names,
        processed_actions=torch.linspace(-0.3, 0.3, 12).unsqueeze(0),
    )
    action_manager = MagicMock()
    action_manager.action = torch.linspace(-1.0, 1.0, 12).unsqueeze(0)
    action_manager.get_term.return_value = joint_action_term

    sensors = {}
    for index, (sensor_name, body_name) in enumerate(
        zip(
            ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot"),
            ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            strict=True,
        )
    ):
        normal_force = torch.tensor([0.0, 0.0, float(index + 1)]).reshape(1, 1, 1, 3)
        friction_sign = -1.0 if index % 2 == 0 else 1.0
        friction_force = torch.tensor([friction_sign * 0.1 * float(index + 1), 0.0, 0.0]).reshape(1, 1, 1, 3)
        sensors[sensor_name] = SimpleNamespace(
            num_sensors=1,
            body_names=[body_name],
            data=SimpleNamespace(
                net_forces_w=_proxy(normal_force[:, :, 0]),
                force_matrix_w=_proxy(normal_force),
                friction_forces_w=_proxy(friction_force),
            ),
        )

    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        command_manager=command_manager,
        action_manager=action_manager,
        scene=SimpleNamespace(sensors=sensors),
        _straight_line_motion_diagnostics={
            "forward_score": torch.tensor([0.8]),
            "straight_score": torch.tensor([0.7]),
            "posture_score": torch.tensor([0.6]),
            "support_loss": torch.tensor([0.2]),
            "reward": torch.tensor([0.7]),
        },
        _foot_clearance_diagnostics={
            "foot_height": torch.tensor([[0.03, 0.05, 0.07, 0.09]]),
            "target_height": torch.tensor([[0.08, 0.08, 0.08, 0.08]]),
            "shortfall": torch.tensor([[0.05, 0.03, 0.01, 0.0]]),
            "swing_weight": torch.tensor([[1.0, 1.0, 0.5, 0.0]]),
            "penalty": torch.tensor([-1.2]),
        },
        _running_reward_clipped=torch.tensor([False]),
    )
    return env, command_term, robot


def test_plotter_records_symmetric_rollout_signals(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)

    plotter.record()

    assert plotter._data["true_lin_vel"][0].tolist() == pytest.approx([0.4, 0.1, 0.2])
    assert plotter._data["desired_lin_vel"][0].tolist() == pytest.approx([0.5, 0.0, 0.1])
    assert plotter._data["E_C_frc"][0].tolist() == pytest.approx([-1.0, -0.5, 0.0, -0.25])
    assert plotter._data["E_C_spd"][0].tolist() == pytest.approx([0.0, -0.5, -1.0, -0.75])
    expected_force_norms = [float(index * np.sqrt(1.01)) for index in range(1, 5)]
    assert plotter._data["foot_forces"][0].tolist() == pytest.approx(expected_force_norms)
    assert plotter._data["foot_velocities"][0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert np.asarray(plotter._data["foot_normal_forces_w"]).shape == (1, 4, 3)
    assert np.asarray(plotter._data["foot_ground_reaction_forces_w"]).shape == (1, 4, 3)
    assert plotter._data["foot_ground_reaction_forces_w"][0][:, 0].tolist() == pytest.approx([-0.1, 0.2, -0.3, 0.4])
    assert plotter._data["foot_ground_reaction_forces_w"][0][:, 2].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert plotter._data["ground_reaction_force_includes_friction"] == [True]
    assert plotter._data["raw_actions"][0].tolist() == pytest.approx(torch.linspace(-1.0, 1.0, 12).tolist())
    assert plotter._data["joint_positions"][0].tolist() == pytest.approx(torch.linspace(-0.4, 0.4, 12).tolist())
    assert plotter._data["joint_position_targets"][0].tolist() == pytest.approx(torch.linspace(-0.3, 0.3, 12).tolist())
    expected_torques = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0]
    assert plotter._data["joint_torques"][0].tolist() == pytest.approx(expected_torques)
    assert plotter._data["joint_velocities"][0].tolist() == pytest.approx((0.1 * torch.arange(1, 13)).tolist())
    assert plotter._data["joint_powers"][0].tolist() == pytest.approx(
        (torch.tensor(expected_torques) * 0.1 * torch.arange(1, 13, dtype=torch.float32)).tolist()
    )
    assert plotter._data["forward_score"][0] == pytest.approx(0.8)
    assert plotter._data["foot_heights"][0].tolist() == pytest.approx([0.03, 0.05, 0.07, 0.09])
    assert plotter._data["foot_clearance_targets"][0].tolist() == pytest.approx([0.08] * 4)


def test_plotter_saves_isaacgym_compatible_outputs(tmp_path):
    module = _load_plotter_module()
    env, command_term, robot = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)
    plotter.record()
    command_term.command = torch.tensor([[1.0, 0.0, 0.0]])
    robot.data.root_pos_w = _proxy(torch.tensor([[1.01, 2.0, 0.5]]))
    plotter.record()

    saved_paths = plotter.save()

    expected_names = {
        "sim_data.npz",
        "figure1_linear_velocities_and_position.png",
        "figure2_E_C_frc_and_contact_forces.png",
        "figure3_E_C_spd_and_foot_velocities.png",
        "figure4_agg_E_C_frc_vs_contact.png",
        "figure5_policy_actions_and_joint_limits.png",
        "figure6_straight_line_reward_diagnostics.png",
        "figure7_foot_clearance.png",
        "figure8_leg_motor_torques.png",
        "figure9_leg_motor_powers.png",
        "figure10_leg_ground_reaction_forces.png",
    }
    assert {path.name for path in saved_paths} == expected_names
    assert all(path.stat().st_size > 0 for path in saved_paths)
    with np.load(tmp_path / "sim_data.npz") as data:
        assert data["true_lin_vel"].shape == (2, 3)
        assert data["desired_positions"].shape == (2, 2)
        assert data["foot_forces"].shape == (2, 4)
        assert data["foot_normal_forces_w"].shape == (2, 4, 3)
        assert data["foot_ground_reaction_forces_w"].shape == (2, 4, 3)
        assert data["ground_reaction_force_includes_friction"].tolist() == [True, True]
        assert data["raw_actions"].shape == (2, 12)
        assert data["actor_means"].shape == (2, 12)
        assert data["joint_positions"].shape == (2, 12)
        assert data["joint_position_targets"].shape == (2, 12)
        assert data["joint_velocities"].shape == (2, 12)
        assert data["joint_torques"].shape == (2, 12)
        assert data["joint_powers"].shape == (2, 12)
        assert data["leg_joint_torques"].shape == (2, 4, 3)
        assert data["leg_joint_powers"].shape == (2, 4, 3)
        assert data["leg_joint_torque_magnitudes"].shape == (2, 4, 3)
        assert data["leg_joint_power_magnitudes"].shape == (2, 4, 3)
        assert data["leg_torque_sums"].shape == (2, 4)
        assert data["leg_torque_magnitude_sums"].shape == (2, 4)
        assert data["leg_power_sums"].shape == (2, 4)
        assert data["leg_power_magnitude_sums"].shape == (2, 4)
        assert data["leg_torque_sums"][0].tolist() == pytest.approx([-2.0, 5.0, -8.0, 11.0])
        assert data["leg_torque_magnitude_sums"][0].tolist() == pytest.approx([6.0, 15.0, 24.0, 33.0])
        assert data["leg_power_sums"][0].tolist() == pytest.approx([-0.6, 2.7, -6.6, 12.3])
        assert data["leg_power_magnitude_sums"][0].tolist() == pytest.approx([1.4, 7.7, 19.4, 36.5])
        assert data["leg_joint_torque_magnitudes"][0, 0].tolist() == pytest.approx([1.0, 2.0, 3.0])
        assert data["leg_joint_power_magnitudes"][0, 0].tolist() == pytest.approx([0.1, 0.4, 0.9])
        assert data["foot_ground_reaction_force_abs_components"].shape == (2, 4, 3)
        assert data["foot_ground_reaction_force_abs_sums"].shape == (2, 4)
        assert data["foot_ground_reaction_force_abs_components"][0, :, 0].tolist() == pytest.approx(
            [0.1, 0.2, 0.3, 0.4]
        )
        assert data["foot_ground_reaction_force_abs_sums"][0].tolist() == pytest.approx([1.1, 2.2, 3.3, 4.4])
        assert data["episode_done"].tolist() == [False, False]
        assert data["usage_plot_smoothing_window_s"].item() == pytest.approx(1.0)
        assert data["usage_plot_smoothing_window_samples"].item() == 51
        assert data["leg_joint_torque_magnitudes_centered_moving_mean"].shape == (2, 4, 3)
        assert data["leg_torque_magnitude_sums_centered_moving_mean"].shape == (2, 4)
        assert data["leg_joint_power_magnitudes_centered_moving_mean"].shape == (2, 4, 3)
        assert data["leg_power_magnitude_sums_centered_moving_mean"].shape == (2, 4)
        assert data["foot_ground_reaction_force_abs_components_centered_moving_mean"].shape == (2, 4, 3)
        assert data["foot_ground_reaction_force_abs_sums_centered_moving_mean"].shape == (2, 4)
        assert data["joint_limit_utilization"].shape == (2, 12)
        assert data["joint_names"].tolist() == [f"joint_{index}" for index in range(12)]
        assert data["leg_names"].tolist() == ["Front Left", "Front Right", "Rear Left", "Rear Right"]
        assert data["motor_role_names"].tolist() == ["Hip/Abad", "Thigh", "Calf"]
        assert data["forward_score"].shape == (2,)
        assert data["foot_heights"].shape == (2, 4)
        assert data["foot_clearance_targets"].shape == (2, 4)
        assert data["foot_clearance_penalty"].shape == (2,)


def test_plotter_rejects_non_symmetric_command_term(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    env.command_manager.get_term.return_value = SimpleNamespace(command=torch.zeros(1, 3))

    with pytest.raises(ValueError, match="does not support symmetric plots"):
        module.SymmetricRolloutPlotter(env, tmp_path)


def test_centered_moving_mean_uses_partial_edge_windows():
    module = _load_plotter_module()
    values = np.arange(5, dtype=np.float32).reshape(-1, 1)

    smoothed = module._centered_moving_mean(values, window_samples=3)

    assert smoothed[:, 0].tolist() == pytest.approx([0.5, 1.0, 2.0, 3.0, 3.5])
    assert smoothed.dtype == np.float64
    with pytest.raises(ValueError, match="positive odd number"):
        module._centered_moving_mean(values, window_samples=2)


def test_centered_moving_mean_does_not_cross_episode_boundaries():
    module = _load_plotter_module()
    values = np.arange(6, dtype=np.float32)
    episode_ends = np.array([False, False, True, False, False, False])

    smoothed = module._centered_moving_mean(values, window_samples=5, episode_ends=episode_ends)

    assert smoothed.tolist() == pytest.approx([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])


def test_centered_one_second_window_has_no_phase_shift_at_50_hz():
    module = _load_plotter_module()
    values = np.zeros(101, dtype=np.float32)
    values[50] = 51.0

    smoothed = module._centered_moving_mean(values, window_samples=51)

    assert smoothed[24] == pytest.approx(0.0)
    assert smoothed[25:76].tolist() == pytest.approx([1.0] * 51)
    assert smoothed[76] == pytest.approx(0.0)


def test_leg_usage_plots_include_raw_and_smoothed_magnitude_curves(tmp_path, monkeypatch):
    module = _load_plotter_module()
    plotter = object.__new__(module.SymmetricRolloutPlotter)
    plotter._output_dir = tmp_path
    component_values = np.arange(1, 25, dtype=np.float64).reshape(2, 4, 3)
    component_sums = component_values.sum(axis=-1)
    data = {
        "time_steps": np.array([0.0, 0.02]),
        "joint_names": np.asarray([f"joint_{index}" for index in range(12)]),
        "usage_plot_smoothing_window_s": np.asarray(1.0),
        "leg_joint_torque_magnitudes": component_values,
        "leg_joint_torque_magnitudes_centered_moving_mean": component_values,
        "leg_torque_magnitude_sums": component_sums,
        "leg_torque_magnitude_sums_centered_moving_mean": component_sums,
        "foot_ground_reaction_force_abs_components": component_values,
        "foot_ground_reaction_force_abs_components_centered_moving_mean": component_values,
        "foot_ground_reaction_force_abs_sums": component_sums,
        "foot_ground_reaction_force_abs_sums_centered_moving_mean": component_sums,
        "ground_reaction_force_includes_friction": np.array([True, True]),
    }
    close_figure = module.plt.close
    monkeypatch.setattr(module.plt, "close", lambda _figure: None)

    plotter._save_per_leg_motor_plot(
        data,
        measurement_key="leg_joint_torque_magnitudes",
        sum_key="leg_torque_magnitude_sums",
        sum_label="Sum |torque|",
        filename="torque_test.png",
        measurement_name="Absolute Applied Joint Torque",
        measurement_unit="N m",
    )
    torque_figure = module.plt.gcf()
    for leg_index, axis in enumerate(torque_figure.axes):
        labels = [line.get_label() for line in axis.lines]
        assert len(labels) == 8
        assert f"|Hip/Abad| raw (joint_{leg_index * 3})" in labels
        assert "|Hip/Abad| 1 s mean" in labels
        assert "Sum |torque| raw" in labels
        assert "Sum |torque| 1 s mean" in labels
        assert all(np.all(np.asarray(line.get_ydata()) >= 0.0) for line in axis.lines)
    close_figure(torque_figure)

    plotter._save_leg_ground_reaction_force_plot(data)
    force_figure = module.plt.gcf()
    for axis in force_figure.axes:
        labels = [line.get_label() for line in axis.lines]
        assert len(labels) == 8
        assert "|Fx| raw" in labels
        assert "|Fx| 1 s mean" in labels
        assert "Sum |F| raw" in labels
        assert "Sum |F| 1 s mean" in labels
        assert all(np.all(np.asarray(line.get_ydata()) >= 0.0) for line in axis.lines)
    close_figure(force_figure)


def test_plotter_records_episode_end_mask(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)

    plotter.record(dones=torch.tensor([True]))

    assert plotter._data["episode_done"] == [True]


def test_plotter_prefers_pre_reset_diagnostic_cache(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    env._last_base_velocity_commands = torch.tensor([[1.2, -0.3, 0.4]])
    env._last_root_positions_w = torch.tensor([[8.0, 9.0]])
    env._last_root_lin_velocities_b = torch.tensor([[2.0, 3.0, 4.0]])
    env._last_root_ang_velocities_b = torch.tensor([[5.0, 6.0, 7.0]])
    env._last_periodic_force_weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    env._last_periodic_speed_weights = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
    env._last_foot_velocities_w = torch.full((1, 4, 3), 2.0)
    env._last_joint_velocities = torch.full((1, 12), 3.0)
    env._last_joint_torques = torch.full((1, 12), 4.0)
    env._last_foot_normal_forces_w = torch.zeros(1, 4, 3)
    env._last_foot_ground_reaction_forces_w = torch.full((1, 4, 3), 5.0)
    env._last_ground_reaction_force_includes_friction = True
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)

    plotter.record()

    assert plotter._data["desired_lin_vel"][0].tolist() == pytest.approx([1.2, -0.3, 0.4])
    assert plotter._data["true_lin_vel"][0].tolist() == pytest.approx([2.0, 3.0, 7.0])
    assert plotter._data["base_positions"][0].tolist() == pytest.approx([8.0, 9.0])
    assert plotter._data["E_C_frc"][0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert plotter._data["joint_torques"][0].tolist() == pytest.approx([4.0] * 12)
    assert plotter._data["joint_powers"][0].tolist() == pytest.approx([12.0] * 12)
    assert plotter._data["foot_forces"][0].tolist() == pytest.approx([np.sqrt(75.0)] * 4)


def test_plotter_limits_samples_for_long_play_sessions(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path, max_samples=1)

    plotter.record()
    plotter.record()

    assert len(plotter._data["time_steps"]) == 1
    assert not env._capture_rollout_diagnostics


def test_plotter_preserves_action_joint_order_for_leg_grouping(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    joint_action_term = env.action_manager.get_term("joint_pos")
    joint_ids = [3, 1, 2, 0, 7, 5, 6, 4, 11, 9, 10, 8]
    joint_action_term._joint_ids = joint_ids
    joint_action_term._joint_names = [f"joint_{index}" for index in joint_ids]
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)

    plotter.record()
    saved_paths = plotter.save()

    assert saved_paths
    expected_torques = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0]
    assert plotter._data["joint_torques"][0].tolist() == pytest.approx([expected_torques[index] for index in joint_ids])
    with np.load(tmp_path / "sim_data.npz") as data:
        assert data["joint_names"].tolist() == [f"joint_{index}" for index in joint_ids]
        assert data["leg_joint_torques"][0, 0].tolist() == pytest.approx([4.0, 2.0, -3.0])
