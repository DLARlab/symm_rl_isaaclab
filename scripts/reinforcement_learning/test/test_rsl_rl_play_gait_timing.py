# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for selectable fixed and velocity-based playback timing."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.flat_env_cfg import (
    UnitreeGo2SymmFlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import GaitVelocityCommand


@pytest.fixture
def playback_override():
    """Load playback helpers without launching the simulator."""
    script = Path(__file__).resolve().parents[1] / "rsl_rl" / "play_rsl_rl.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    names = {"_tracking_error_direction_command", "_apply_tracking_error_direction_command"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_TRACKING_ERROR_DIRECTION_SEQUENCE"
                for target in node.targets
            )
        )
    ]
    args = SimpleNamespace(
        tracking_error_direction_test=True,
        tracking_error_direction_speed=1.0,
        tracking_error_direction_lateral_speed=0.5,
        tracking_error_direction_yaw_rate=0.6,
    )
    namespace = {"torch": torch, "args_cli": args}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(script), "exec"), namespace)
    return namespace["_apply_tracking_error_direction_command"]


def _make_command(cfg_cls=UnitreeGo2SymmFlatEnvCfg_PLAY, num_envs=1):
    cfg = cfg_cls().commands.base_velocity
    cfg.curriculum = None
    cfg.add_noise_period = False
    cfg.add_noise_theta = False
    cfg.init_foot_thetas = ((-0.13, 0.13, 0.63, 0.37),)
    cfg.ranges.lin_vel_x = (0.0, 0.0)
    cfg.ranges.lin_vel_y = (0.0, 0.0)
    cfg.ranges.ang_vel_z = (0.0, 0.0)
    robot = SimpleNamespace(data=SimpleNamespace(heading_w=SimpleNamespace(torch=torch.zeros(num_envs))))
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot}, extras={"log": {}})
    command = GaitVelocityCommand(cfg, env)
    command.reset(torch.arange(num_envs))
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    return SimpleNamespace(unwrapped=env), command


def _expected_velocity_timing(command, forward_speed):
    length = sum(command.cfg.base_height_range) / 2.0
    velocity_star = abs(forward_speed) / math.sqrt(9.81 * length)
    period = 2.55 * math.exp(-0.975 * velocity_star) * math.sqrt(length / 9.81)
    duty = 0.5588 * math.exp(-0.681 * velocity_star)
    return max(period, 0.1), min(max(duty, 0.1), 0.9)


@pytest.mark.parametrize("calculate_from_sampling_curve", [False, True])
@pytest.mark.parametrize("cfg_cls", [UnitreeGo2SymmFlatEnvCfg_PLAY, DobotX1SymmFlatEnvCfg_PLAY])
@pytest.mark.parametrize(
    "segment, direction, velocity",
    [
        (0, "forward", (1.0, 0.0, 0.0)),
        (1, "backward", (-1.0, 0.0, 0.0)),
        (2, "left", (0.0, 0.5, 0.0)),
        (3, "right", (0.0, -0.5, 0.0)),
        (4, "yaw_left", (0.0, 0.0, 0.6)),
        (5, "yaw_right", (0.0, 0.0, -0.6)),
    ],
)
def test_initial_playback_respects_timing_selector_for_each_velocity(
    playback_override, cfg_cls, segment, direction, velocity, calculate_from_sampling_curve
):
    env, command = _make_command(cfg_cls)
    command.cfg.calculate_from_sampling_curve = calculate_from_sampling_curve
    command.gait_phase[:] = 0.37
    templates = command.foot_thetas.clone()

    assert playback_override(env, segment * 10, 60) == direction

    period, duty = (
        _expected_velocity_timing(command, velocity[0])
        if calculate_from_sampling_curve
        else ((0.52 if cfg_cls is UnitreeGo2SymmFlatEnvCfg_PLAY else 0.60), 0.56)
    )
    assert command.gait_periods.tolist() == pytest.approx([period])
    assert command.duty_factors.tolist() == pytest.approx([duty])
    torch.testing.assert_close(command.vel_command_b, torch.tensor([velocity]))
    torch.testing.assert_close(command.gait_phase, torch.tensor([0.37]))
    torch.testing.assert_close(command.foot_thetas, templates)
    assert torch.isinf(command.time_left).all()


@pytest.mark.parametrize("calculate_from_sampling_curve", [False, True])
def test_held_velocity_keeps_timing_without_restarting_transition(playback_override, calculate_from_sampling_curve):
    env, command = _make_command()
    command.cfg.calculate_from_sampling_curve = calculate_from_sampling_curve
    command.cfg.add_noise_period = True
    playback_override(env, 0, 60)
    periods, duties = command.gait_periods.clone(), command.duty_factors.clone()
    if not calculate_from_sampling_curve:
        assert periods.tolist() == pytest.approx([0.52])
        assert duties.tolist() == pytest.approx([0.56])
    for step in range(10):
        playback_override(env, step, 60)
        torch.testing.assert_close(command.gait_periods, periods)
        torch.testing.assert_close(command.duty_factors, duties)

    playback_override(env, 10, 60)
    command._update_gait_transition(command.gait_periods[0].item() / 4.0)
    phases = command.foot_phases().clone()
    progress = command._gait_transition_progress.clone()
    periods = command.gait_periods.clone()
    playback_override(env, 11, 60)
    torch.testing.assert_close(command.gait_periods, periods)
    torch.testing.assert_close(command.foot_phases(), phases)
    torch.testing.assert_close(command._gait_transition_progress, progress)


@pytest.mark.parametrize("calculate_from_sampling_curve", [False, True])
def test_velocity_change_applies_selected_timing_and_preserves_phase(playback_override, calculate_from_sampling_curve):
    env, command = _make_command()
    command.cfg.calculate_from_sampling_curve = calculate_from_sampling_curve
    playback_override(env, 0, 60)
    command.gait_phase[:] = 0.37
    phases = command.foot_phases().clone()
    duty_before = command.duty_factors.clone()

    playback_override(env, 20, 60)

    period, duty = _expected_velocity_timing(command, 0.0) if calculate_from_sampling_curve else (0.52, 0.56)
    assert command.gait_periods.tolist() == pytest.approx([period])
    assert command._duty_factor_transition_target.tolist() == pytest.approx([duty])
    torch.testing.assert_close(command.foot_phases(), phases)
    torch.testing.assert_close(command.duty_factors, duty_before)
    command._update_gait_transition(period)
    assert command.duty_factors.tolist() == pytest.approx([duty])


@pytest.mark.parametrize("calculate_from_sampling_curve", [False, True])
def test_episode_reset_restores_selected_timing_only_for_reset_environments(
    playback_override, calculate_from_sampling_curve
):
    env, command = _make_command(num_envs=2)
    command.cfg.calculate_from_sampling_curve = calculate_from_sampling_curve
    playback_override(env, 0, 60)
    command.gait_phase[:] = 0.37
    command.reset(torch.tensor([0]))
    playback_override(env, 1, 60)

    period, duty = _expected_velocity_timing(command, 1.0) if calculate_from_sampling_curve else (0.52, 0.56)
    assert command.gait_periods.tolist() == pytest.approx([period, period])
    assert command.duty_factors.tolist() == pytest.approx([duty, duty])
    torch.testing.assert_close(command.gait_phase, torch.tensor([0.0, 0.37]))
    torch.testing.assert_close(command._gait_transition_progress, torch.ones(2))


def test_explicit_fixed_timing_remains_supported(playback_override):
    env, command = _make_command()
    command.cfg.calculate_from_sampling_curve = False
    command.cfg.gait_period = 0.7
    command.cfg.duty_factor = 0.6

    playback_override(env, 0, 60)

    assert command.gait_periods.tolist() == pytest.approx([0.7])
    assert command.duty_factors.tolist() == pytest.approx([0.6])
