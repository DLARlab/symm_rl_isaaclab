# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check every parallel statistical trial is recorded without crossing an episode end."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging import version
from test_leg_usage_grid_runner import _load_grid_helpers
from test_symm_command_batch import _cache_step, _make_batch_env


@pytest.fixture
def batch_env(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "rsl_rl"))
    env, command, robot = _make_batch_env(100)
    robot.data.joint_effort_limits.torch = torch.ones(100, 12) * 30
    robot.data.body_mass = SimpleNamespace(torch=torch.ones(100, 1) * 20)
    robot.actuators = {
        "legs": SimpleNamespace(
            joint_names=env.action_manager.get_term("joint_pos")._joint_names,
            effort_limit=torch.ones(100, 12) * 30,
        )
    }
    for sensor in env.scene.sensors.values():
        sensor.cfg = SimpleNamespace(filter_prim_paths_expr=["/World/ground/terrain/mesh"])
    action_term = env.action_manager.get_term("joint_pos")
    action_term._scale = 0.5
    action_term._offset = torch.zeros(100, 12)
    command.command[:] = torch.tensor([1.0, 0.0, 0.0])
    command.foot_thetas = torch.tensor([[0.0, 0.0, 0.5, 0.5]]).expand(100, -1)
    command.gait_periods = torch.ones(100) * 0.6
    command.duty_factors = torch.ones(100) * 0.56
    command.phase = torch.zeros(100)
    command.common_gait_phases = lambda: command.phase
    env._last_foot_normal_force_is_ground_filtered = torch.ones(100, dtype=torch.bool)
    return env, command, robot


def _record(recorder, env, command, robot, step, dones=None):
    command.phase[:] = (step + 1) * env.step_dt / 0.6
    phases = command.phase.clone() - env.step_dt / 0.6
    if dones is None:
        dones = torch.zeros(100, dtype=torch.bool)
    # Simulate the automatic command-clock rewind after a terminal physics step.
    command.phase[dones] = 0.0
    robot.data.applied_torque.torch[:] = torch.arange(100)[:, None] + step
    _cache_step(env, command, robot)
    recorder.record(
        dones=dones,
        critic_values=torch.arange(100)[:, None],
        pre_decision_common_gait_phases=phases,
        rewards=torch.ones(100),
    )


def test_all_100_trials_retain_1000_measured_steps_after_settling(batch_env, tmp_path):
    from evaluation_batch_recorder import EvaluationBatchRecorder

    env, command, robot = batch_env
    recorder = EvaluationBatchRecorder(env, tmp_path, max_samples=1250)
    for step in range(1250):
        _record(recorder, env, command, robot, step)
    assert recorder.is_complete
    assert recorder._samples["joint_torques"].shape == (1250, 100, 12)
    for trial in range(100):
        np.testing.assert_allclose(recorder._samples["joint_torques"][250:, trial, 0], np.arange(250, 1250) + trial)
    for trial in (0, 1, 99):
        plotter = recorder.trial_plotter(trial, tmp_path / str(trial))
        assert len(plotter._data["time_steps"][250:]) == 1000
        path = plotter.save_data()
        with np.load(path, allow_pickle=False) as archive:
            assert archive["true_lin_vel"].shape == (1250, 3)
            assert archive["joint_torques"].shape == (1250, 12)
            assert archive["critic_values"].shape == (1250,)
            np.testing.assert_allclose(archive["joint_torques"][:, 0], np.arange(1250) + trial)
            assert archive["foot_normal_force_is_ground_filtered"].all()


def test_first_terminal_sample_is_preserved_and_other_trials_continue(batch_env, tmp_path):
    from evaluation_batch_recorder import EvaluationBatchRecorder

    env, command, robot = batch_env
    recorder = EvaluationBatchRecorder(env, tmp_path, max_samples=4)
    for step in range(4):
        dones = torch.zeros(100, dtype=torch.bool)
        if step in (1, 3):
            dones[0] = True
        _record(recorder, env, command, robot, step, dones)
    first = recorder.trial_plotter(0, tmp_path / "failed")
    other = recorder.trial_plotter(99, tmp_path / "completed")
    assert len(first._data["time_steps"]) == 2
    assert first._data["episode_done"] == [False, True]
    assert first._data["heading_sample_valid"] == [True, False]
    assert np.diff(first._data["common_gait_phases"])[0] > 0
    assert len(other._data["time_steps"]) == 4
    np.testing.assert_allclose(np.asarray(first._data["joint_torques"])[:, 0], [0, 1])
    np.testing.assert_allclose(np.asarray(other._data["joint_torques"])[:, 0], [99, 100, 101, 102])


def test_batch_fields_match_existing_single_trial_schema(batch_env, tmp_path):
    from evaluation_batch_recorder import EvaluationBatchRecorder
    from evaluation_rollout_plotter import SymmetricRolloutPlotter

    env, command, robot = batch_env
    batch = EvaluationBatchRecorder(env, tmp_path, max_samples=1)
    serial = SymmetricRolloutPlotter(env, tmp_path / "serial", env_index=99)
    _record(batch, env, command, robot, 0)
    serial.record(
        critic_values=torch.arange(100)[:, None],
        rewards=torch.ones(100),
        pre_decision_common_gait_phases=torch.zeros(100),
    )
    parallel = batch.trial_plotter(99, tmp_path / "parallel")
    serial_data, parallel_data = serial._prepare_data(), parallel._prepare_data()
    assert set(serial_data).issubset(parallel_data)
    for key, expected in serial_data.items():
        if expected.dtype.kind in "US":
            np.testing.assert_array_equal(parallel_data[key], expected)
        else:
            np.testing.assert_allclose(parallel_data[key], expected, rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=key)


def test_runner_batches_trials_and_resets_only_ended_policy_slots(batch_env, tmp_path):
    env, command, robot = batch_env
    env.unwrapped = env
    source = Path(__file__).resolve().parents[1] / "rsl_rl" / "play_rsl_rl.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_record_parallel_leg_usage_trials"
    )
    reset_masks = []

    class Policy:
        def __call__(self, obs):
            return env.action_manager.action

        def reset(self, mask):
            reset_masks.append(mask.clone())

    calls = []

    def step(actions):
        index = len(calls)
        calls.append(index)
        command.phase[:] = (index + 1) * env.step_dt / 0.6
        dones = torch.zeros(100, dtype=torch.bool)
        if index == 1:
            dones[0] = True
            command.phase[0] = 0
        _cache_step(env, command, robot)
        return {}, torch.ones(100), dones, {}

    env.step = step
    namespace = dict(
        torch=torch,
        np=np,
        Path=Path,
        version=version,
        installed_version="4.0.0",
        _pre_decision_common_gait_phases=lambda _: command.phase.clone(),
        _evaluate_critic_values=lambda *_: torch.ones(100),
        _active_termination_terms=lambda _, env_index: [f"fell_{env_index}"],
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    recorder, terms = namespace[function.name](env, None, Policy(), None, {}, total_steps=4, output_root=tmp_path)
    assert len(calls) == 4
    assert len(reset_masks) == 1
    assert reset_masks[0].nonzero().flatten().tolist() == [0]
    assert terms == {0: ["fell_0"]}
    assert len(recorder.trial_plotter(0, tmp_path / "0")._data["time_steps"]) == 2
    assert len(recorder.trial_plotter(99, tmp_path / "99")._data["time_steps"]) == 4


def test_grid_saves_and_resumes_all_100_trials_with_one_rollout(batch_env, tmp_path):
    env, command, robot = batch_env
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_parallel_grid_analysis", root / "scripts/symm_locomotion/evaluation.py"
    )
    analysis = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analysis
    spec.loader.exec_module(analysis)
    checkpoint = tmp_path / "run/model_9999.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    params = checkpoint.parent / "params"
    params.mkdir()
    (params / "env.yaml").write_text(
        "commands:\n  base_velocity:\n    init_foot_thetas: [[0.0,0.0,0.5,0.5]]\n", encoding="utf-8"
    )
    study = analysis.build_study(
        repo_root=root,
        checkpoint=checkpoint,
        robot="go2",
        task="task",
        step_dt=0.02,
        protocol="trained",
        velocities_mps=(-1.0, 1.0),
        settle_s=0.02,
        measure_s=0.06,
        num_envs=100,
    )
    plan_path = analysis.prepare_study(study, resume=True, analyze_only=False, dry_run=False)
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    helpers = _load_grid_helpers()
    loaded, _, _ = helpers._load_leg_usage_plan(str(plan_path))
    source = root / "scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py"
    names = {
        "_atomic_write_json",
        "_append_grid_event",
        "_write_grid_progress",
        "_format_eta",
        "_record_parallel_leg_usage_trials",
        "_run_leg_usage_grid",
    }
    nodes = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = dict(
        vars(helpers),
        torch=torch,
        np=np,
        Path=Path,
        Any=object,
        os=os,
        tempfile=tempfile,
        hashlib=hashlib,
        json=json,
        math=math,
        random=random,
        time=time,
        version=version,
        installed_version="4.0.0",
        _LEG_USAGE_PROTOCOL_VERSION="leg_usage_grid_v1",
        _LEG_USAGE_FULL_PROTOCOL_VERSION="leg_usage_grid_full_v3",
        _LEG_USAGE_LIGHT_PROTOCOL_VERSION="leg_usage_grid_light_v2",
        _LEG_USAGE_TRAINED_PROTOCOL_VERSION="leg_usage_grid_trained_v1",
        _pre_decision_common_gait_phases=lambda _: command.phase.clone(),
        _evaluate_critic_values=lambda *_: torch.ones(100),
        _active_termination_terms=lambda _, env_index=0: [],
        _reset_inference_policy=lambda *args: None,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    env.unwrapped = env
    env.seed = lambda _: None
    resets, scenarios, steps = [], [], []

    def set_scenario(vx, gait):
        scenarios.append((vx, gait))
        command.command[:, 0] = vx

    command.set_evaluation_scenario = set_scenario

    def reset():
        resets.append(True)
        command.phase.zero_()
        return {}, {}

    def step(actions):
        command.phase += env.step_dt / 0.6
        robot.data.applied_torque.torch[:] = torch.arange(100)[:, None]
        steps.append(True)
        _cache_step(env, command, robot)
        return {}, torch.ones(100), torch.zeros(100, dtype=torch.bool), {}

    env.reset, env.step = reset, step

    def policy(_):
        return env.action_manager.action

    run = namespace["_run_leg_usage_grid"]
    run(env, None, policy, None, loaded, plan_path, plan_sha256)
    assert len(resets) == len(scenarios) == 2
    assert len(steps) == 8
    for cell in loaded["cells"]:
        cell_dir = plan_path.parent / cell["relative_output_dir"]
        assert helpers._validated_completed_cell(cell_dir, cell, plan_sha256), cell["id"]
        with np.load(cell_dir / "sim_data.npz", allow_pickle=False) as data:
            assert data["trial_index"].item() == cell["trial_index"]
            np.testing.assert_allclose(data["joint_torques"][:, 0], cell["trial_index"])
    rows = [analysis.analyze_cell(study, cell, plan_path.parent) for cell in loaded["cells"]]
    assert all(row["velocity_metric_valid"] for row in rows)
    assert {row["trial_index"] for row in rows} == set(range(100))
    families, overall = analysis.aggregate_results(study, rows)
    assert overall["coverage"]["expected_cells"] == 200
    run(env, None, policy, None, loaded, plan_path, plan_sha256)
    assert len(resets) == 2
    assert len(steps) == 8
