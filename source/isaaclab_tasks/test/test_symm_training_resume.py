# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for curriculum checkpoints and continuous training histories."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.flat_env_cfg import (
    make_gait_velocity_command,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

REPO_ROOT = Path(__file__).resolve().parents[3]


def _module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resume = _module("training_resume_under_test", "scripts/reinforcement_learning/rsl_rl/training_resume.py")
merge = _module("merge_training_logs_under_test", "scripts/symm_locomotion/merge_training_logs.py")


def _curriculum():
    return symm_quadruped._VelocityCommandBinCurriculum(make_gait_velocity_command(symm_quadruped), "cpu")


def _expand(curriculum):
    indices = curriculum._weights.nonzero()
    boundary = indices[indices[:, 0].argmax()]
    curriculum.update(boundary.unsqueeze(0), torch.tensor([True]))


def test_curriculum_round_trip_keeps_expanded_bins_and_independent_cpu_weights():
    original = _curriculum()
    _expand(original)
    state = original.state_dict()
    restored = _curriculum()
    assert restored.active_bin_count < original.active_bin_count

    restored.load_state_dict(state)

    assert torch.equal(restored._weights, original._weights)
    assert restored.max_active_abs_command.tolist() == original.max_active_abs_command.tolist()
    state["weights"].zero_()
    assert restored._weights.sum() > 0
    assert original._weights.sum() > 0


@pytest.mark.parametrize("change", ["ranges", "num_bins", "shape", "nan", "negative", "zero", "large", "version"])
def test_curriculum_rejects_incompatible_or_invalid_state_without_mutating(change):
    curriculum = _curriculum()
    state = curriculum.state_dict()
    before = curriculum._weights.clone()
    if change == "ranges":
        state["ranges"] = ((-1, 1), (-1, 1), (-1, 1))
    elif change == "num_bins":
        state["num_bins"] = (3, 1, 3)
    elif change == "shape":
        state["weights"] = torch.ones(1)
    elif change == "version":
        state["version"] = 99
    else:
        state["weights"].fill_({"nan": float("nan"), "negative": -1, "zero": 0, "large": 2}[change])
    with pytest.raises(ValueError):
        curriculum.load_state_dict(state)
    assert torch.equal(curriculum._weights, before)


class _Env:
    num_envs = 2
    num_actions = 2
    device = "cpu"
    max_episode_length = 10
    cfg = {}

    def __init__(self):
        self.unwrapped = self
        self.sampler = _curriculum()
        self.command_manager = SimpleNamespace(active_terms=["base_velocity"], get_term=lambda name: self)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_weights = None

    def curriculum_state_dict(self):
        return self.sampler.state_dict()

    def curriculum_load_state_dict(self, state):
        self.sampler.load_state_dict(state)

    def get_observations(self):
        return TensorDict({"policy": torch.ones(self.num_envs, 4)}, batch_size=[self.num_envs])

    def reset(self):
        self.reset_weights = self.sampler._weights.clone()
        return self.get_observations(), {}

    def step(self, actions):
        return self.get_observations(), torch.ones(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), {}


def _config():
    return {
        "num_steps_per_env": 4,
        "save_interval": 1,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [8],
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [8]},
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
    }


def _runner(path, algorithm=None, **kwargs):
    config = _config()
    if algorithm is not None:
        config["algorithm"]["class_name"] = algorithm
    return resume._OnPolicyRunnerResume(_Env(), config, log_dir=str(path), device="cpu", **kwargs)


@pytest.mark.parametrize(
    "algorithm",
    [
        "rsl_rl.algorithms:PPO",
        "isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo:TimeReversalPPO",
    ],
)
def test_real_ppo_resume_restores_curriculum_counters_and_starts_at_next_iteration(tmp_path, algorithm):
    original = _runner(tmp_path / "first", algorithm=algorithm)
    _expand(original.env.sampler)
    original.learn(2)
    expected_steps = original.logger.tot_timesteps
    expected_time = original.logger.tot_time
    resumed = _runner(tmp_path / "second", algorithm=algorithm)

    resumed.load(str(tmp_path / "first/model_1.pt"))

    assert resumed.current_learning_iteration == 2
    assert resumed.logger.tot_timesteps == expected_steps
    assert resumed.logger.tot_time == expected_time
    assert torch.equal(resumed.env.reset_weights, original.env.sampler._weights)
    assert resumed.alg.optimizer.param_groups[0]["lr"] == resumed.alg.learning_rate
    if hasattr(resumed.alg, "_time_reversal_update_count"):
        assert resumed.alg._time_reversal_update_count == 2
    resumed.learn(1)
    assert resumed.current_learning_iteration == 2
    assert resumed.logger.tot_timesteps == expected_steps + 8
    assert (tmp_path / "second/model_2.pt").is_file()
    events = EventAccumulator(str(tmp_path / "second")).Reload()
    assert [event.step for event in events.Scalars("Perf/learning_time")] == [2]


def test_older_checkpoint_loads_with_warning_and_does_not_repeat_iteration(tmp_path):
    runner = _runner(tmp_path)
    checkpoint = runner.alg.save()
    checkpoint.update(iter=7, infos=None)
    path = tmp_path / "old.pt"
    torch.save(checkpoint, path)
    with pytest.warns(UserWarning, match="no curriculum state"):
        runner.load(str(path))
    assert runner.current_learning_iteration == 8
    runner.logger.init_logging_writer()
    runner.logger.writer.flush()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [(event.step, event.value) for event in events.Scalars("Resume/curriculum_restarted")] == [(8, 1)]
    runner.logger.writer.close()


def test_actor_only_load_does_not_restore_training_state(tmp_path):
    runner = _runner(tmp_path)
    runner.logger.init_logging_writer()
    runner.current_learning_iteration = 7
    runner.save(str(tmp_path / "model.pt"))
    target = _runner(tmp_path)
    target.load(str(tmp_path / "model.pt"), load_cfg={"actor": True})
    assert target.current_learning_iteration == 0
    assert target.env.reset_weights is None


def test_tensorboard_continuation_reuses_history_and_preserves_original_metadata(tmp_path):
    runner = _runner(tmp_path)
    runner.learn(1)
    resumed = _runner(tmp_path, resume_logging=True)
    resumed.load(str(tmp_path / "model_0.pt"))
    resumed.learn(1)
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [event.step for event in events.Scalars("Perf/learning_time")] == [0, 1]
    assert json.loads((tmp_path / "training_resume.json").read_text())["logger"] == "tensorboard"


def test_logging_continuation_rejects_history_past_checkpoint(tmp_path):
    runner = _runner(tmp_path)
    runner.learn(2)
    resumed = _runner(tmp_path, resume_logging=True)
    resumed.load(str(tmp_path / "model_0.pt"))
    with pytest.raises(ValueError, match="extends past"):
        resumed.learn(1)


def test_merge_preserves_originals_and_later_segment_wins_at_overlapping_step(tmp_path):
    first, second, output = [tmp_path / name for name in ("first", "second", "merged")]
    for directory, values in ((first, [(0, 1.0), (1, 2.0)]), (second, [(1, 3.0), (2, 4.0)])):
        with SummaryWriter(str(directory)) as writer:
            for step, value in values:
                writer.add_scalar("Reward/total", value, step)
    before = {path: path.read_bytes() for directory in (first, second) for path in directory.iterdir()}

    assert merge.main([str(first), str(second), "--output_dir", str(output), "--curriculum_restarted", "1"]) == 0

    events = EventAccumulator(str(output)).Reload()
    assert [(event.step, event.value) for event in events.Scalars("Reward/total")] == [(0, 1), (1, 3), (2, 4)]
    assert [(event.step, event.value) for event in events.Scalars("Resume/curriculum_restarted")] == [(1, 1)]
    assert all(path.read_bytes() == data for path, data in before.items())
    manifest = json.loads((output / "merge_manifest.json").read_text())
    assert manifest["duplicate_points_replaced"] == 1
    assert manifest["sources"][1]["curriculum"] == "restarted"
    with pytest.raises(SystemExit):
        merge.main([str(first), "--output_dir", str(output)])


def test_merge_reads_local_wandb_archive_without_network_and_applies_offset(tmp_path):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    source = tmp_path / "run.wandb"
    store = DataStore()
    store.open_for_write(str(source))
    for step in range(2):
        record = pb.Record()
        record.history.step.num = step
        item = record.history.item.add()
        item.nested_key.append("Train/mean_reward")
        item.value_json = json.dumps(0.5 + step)
        store.write(record)
    store.close()
    output = tmp_path / "merged"
    merge.main([str(source), "--output_dir", str(output), "--step_offsets", "100"])
    events = EventAccumulator(str(output)).Reload()
    assert [(event.step, event.value) for event in events.Scalars("Train/mean_reward")] == [(100, 0.5), (101, 1.5)]


def test_wandb_continuation_uses_saved_identity_and_allows_updated_config(tmp_path, monkeypatch):
    import wandb

    calls = []
    updates = []
    run = SimpleNamespace(id="saved_id", project="test_project", entity="test_entity", step=8)
    monkeypatch.setattr(wandb, "init", lambda **kwargs: calls.append(kwargs) or run)
    monkeypatch.setattr(wandb, "run", run)
    monkeypatch.setattr(wandb, "config", SimpleNamespace(update=lambda *args, **kwargs: updates.append((args, kwargs))))
    monkeypatch.delenv("WANDB_USERNAME", raising=False)
    logger = _runner(tmp_path, resume_logging=True).logger
    logger.cfg.update(logger="wandb", wandb_project="test_project")
    logger.next_iteration = 8
    logger.resume_state = {"wandb": {"id": "saved_id", "project": "test_project", "entity": "test_entity"}}
    monkeypatch.setattr(logger, "_store_code_state", lambda: [])

    logger.init_logging_writer()

    assert calls == [
        {
            "project": "test_project",
            "entity": "test_entity",
            "id": "saved_id",
            "resume": "must",
            "allow_val_change": True,
        }
    ]
    assert updates[0][1] == {"allow_val_change": True}
    assert json.loads((tmp_path / "training_resume.json").read_text())["wandb"]["id"] == "saved_id"
    logger.writer.close()


def test_wandb_continuation_requires_explicit_identity_for_older_runs(tmp_path):
    logger = _runner(tmp_path, resume_logging=True).logger
    logger.cfg.update(logger="wandb", wandb_project="test_project")
    with pytest.raises(ValueError, match="--wandb_run_id"):
        logger._wandb_writer()


def test_logger_continuation_preserves_the_original_code_snapshot(tmp_path, monkeypatch):
    logger = _runner(tmp_path, resume_logging=True).logger
    original = tmp_path / "git/code.diff"
    original.parent.mkdir()
    original.write_text("original snapshot")

    def store_snapshot(self):
        path = Path(self.log_dir) / "git/code.diff"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("resumed snapshot")
        return [str(path)]

    monkeypatch.setattr(resume.Logger, "_store_code_state", store_snapshot)
    snapshot = Path(logger._store_code_state()[0])
    assert original.read_text() == "original snapshot"
    assert snapshot.read_text() == "resumed snapshot"
    assert logger.log_dir == str(tmp_path)
