# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_module():
    path = Path(__file__).resolve().parents[1] / "training_provenance.py"
    spec = importlib.util.spec_from_file_location("_test_training_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROVENANCE = _load_module()


class _Config:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return copy.deepcopy(self.value)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    asset = repo / "assets" / "robot" / "urdf" / "robot.urdf"
    asset.parent.mkdir(parents=True)
    asset.write_text("<robot name='test'/>\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Isaac Lab Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Initialize test repository")
    return repo, asset


def _runner() -> SimpleNamespace:
    actor = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    critic = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1.0e-3)
    algorithm = SimpleNamespace(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        gpu_global_rank=0,
    )
    return SimpleNamespace(alg=algorithm)


def _configs(asset: Path) -> tuple[_Config, _Config]:
    env = _Config(
        {
            "seed": 7,
            "scene": {"num_envs": 8, "robot": {"spawn": {"asset_path": str(asset)}}},
            "commands": {"base_velocity": {"gait_library_version": "test-v1", "range": [-1.0, 1.0]}},
            "events": {"randomize_mass": {"range": [0.9, 1.1]}},
            "rewards": {"tracking": {"weight": 1.0}},
        }
    )
    agent = _Config(
        {
            "seed": 7,
            "max_iterations": 100,
            "num_steps_per_env": 24,
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "actor": {"hidden_dims": [4]},
            "critic": {"hidden_dims": [4]},
            "algorithm": {
                "class_name": (
                    "isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped."
                    "time_reversal_ppo:TimeReversalPPO"
                ),
                "optimizer": "adam",
                "learning_rate": 1.0e-3,
            },
        }
    )
    return env, agent


def _study_condition(seed: int = 7) -> dict:
    schedule = {
        "enabled": True,
        "warmup_iterations": 0,
        "rampup_iterations": 0,
        "hold_iterations": 0,
        "decay_iterations": 0,
        "final_scale": 1.0,
        "ramp_shape": "linear",
    }
    return {
        "robot": "go2",
        "seed": seed,
        "tr_policy": {"enabled": True, "coefficient": 0.1, "schedule": copy.deepcopy(schedule)},
        "tr_value": {"enabled": False, "coefficient": 0.0, "schedule": copy.deepcopy(schedule)},
        "validity_mask": {
            "mode": "command",
            "thresholds": {
                "min_abs_command_velocity": None,
                "tracking_abs_tolerance": 0.25,
                "tracking_rel_tolerance": 0.25,
                "projected_gravity_tolerance": 0.35,
                "phase_boundary_margin": 0.03,
                "command_speed_bin_edges": [0.5, 1.0, 1.5],
            },
        },
        "trajectory_augmentation": {
            "enabled": False,
            "mode": "dynamics_filtered_reverse_action_supervision",
            "action_source": "analytic",
            "coefficient": 0.0,
            "schedule": copy.deepcopy(schedule),
            "filter_settings": {
                "filter_enabled": True,
                "validation_fraction": 0.2,
                "validation_quantile": 0.95,
                "threshold_multiplier": 2.0,
                "minimum_validation_samples": 128,
                "maximum_validation_loss": 1.0,
                "phase_boundary_margin": 0.03,
                "maximum_contact_impulse": 20.0,
                "action_abs_limit": 10.0,
                "max_augmented_to_original_ratio": 0.25,
                "use_confidence_weights": True,
                "rng_seed": 0,
            },
        },
        "reward_profile": {"name": "test", "overrides": []},
        "leg_sync_reward_weight": 0.2,
        "training_iterations": 100,
        "num_envs": 8,
        "evaluation_protocol": "light",
    }


def _set_dotted(mapping: dict, path: str, value) -> None:
    cursor = mapping
    keys = path.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = copy.deepcopy(value)


def _align_configs_to_condition(env: _Config, agent: _Config, condition: dict) -> None:
    expected = PROVENANCE._expected_runtime_config_values(condition)
    for path, value in expected["environment"].items():
        _set_dotted(env.value, path, value)
    for path, value in expected["agent"].items():
        _set_dotted(agent.value, path, value)


def _record(tmp_path: Path, run_name: str = "run") -> tuple[Path, dict]:
    repo, asset = _repository(tmp_path)
    env, agent = _configs(asset)
    path = PROVENANCE.record_training_initialization(
        runner=_runner(),
        env_cfg=env,
        agent_cfg=agent,
        log_dir=tmp_path / run_name,
        repo_root=repo,
    )
    assert path is not None
    return path, json.loads(path.read_text(encoding="utf-8"))


def _rewrite_record(path: Path, record: dict) -> None:
    record.pop("record_sha256", None)
    record["record_sha256"] = PROVENANCE.sha256_value(record)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _study_record(tmp_path: Path, *, declared_seed: int = 7) -> tuple[Path, dict, dict]:
    repo, asset = _repository(tmp_path)
    env, agent = _configs(asset)
    run_name = "publication-run"
    agent.value["run_name"] = run_name
    condition = _study_condition(declared_seed)
    _align_configs_to_condition(env, agent, condition)
    branch = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    study_identity = PROVENANCE.sha256_value({"study": "publication"})
    runtime_argv = ["--task", "Test-Symm-v0", "--seed", str(declared_seed)]
    exact_command = ["isaaclab", "train", "--rl_library", "rsl_rl", *runtime_argv]
    context = {
        "schema_version": 1,
        "study_name": "publication",
        "study_sha256": study_identity,
        "expected_branch": branch,
        "treatment_variables": [],
        "run_name": run_name,
        "condition_sha256": PROVENANCE.sha256_value(condition),
        "resolved_condition": condition,
        "exact_training_command": exact_command,
        "evaluation_command_template": ["evaluate", "--protocol", "light"],
    }
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    record_path = PROVENANCE.record_training_initialization(
        runner=_runner(),
        env_cfg=env,
        agent_cfg=agent,
        log_dir=tmp_path / "run",
        repo_root=repo,
        study_context_path=context_path,
        runtime_argv=runtime_argv,
    )
    assert record_path is not None
    manifest = {
        "study_name": "publication",
        "study_sha256": study_identity,
        "expected_branch": branch,
        "treatment_variables": [],
        "runs": [
            {
                "run_name": run_name,
                "condition_sha256": PROVENANCE.sha256_value(condition),
                "resolved_condition": condition,
                "study_context": str(context_path.resolve()),
                "exact_training_command": exact_command,
                "evaluation_command_template": ["evaluate", "--protocol", "light"],
                "initialization_record": str(record_path),
            }
        ],
    }
    return record_path, json.loads(record_path.read_text(encoding="utf-8")), manifest


def test_state_hash_is_stable_and_changes_with_tensor_content():
    first = {"weight": torch.tensor([[1.0, 2.0]]), "step": 0}
    second = {"step": 0, "weight": torch.tensor([[1.0, 2.0]])}
    changed = {"weight": torch.tensor([[1.0, 3.0]]), "step": 0}

    assert PROVENANCE.state_sha256(first) == PROVENANCE.state_sha256(second)
    assert PROVENANCE.state_sha256(first) != PROVENANCE.state_sha256(changed)


def test_direct_run_writes_resolved_command_and_result_independent_cohort_metadata(tmp_path):
    repo, asset = _repository(tmp_path)
    env, agent = _configs(asset)
    initialization = PROVENANCE.record_training_initialization(
        runner=_runner(),
        env_cfg=env,
        agent_cfg=agent,
        log_dir=tmp_path / "run",
        repo_root=repo,
        runtime_argv=["--task", "Test-Symm-v0"],
    )
    assert initialization is not None

    command_path, cohort_path = PROVENANCE.record_resolved_run_metadata(
        env_cfg=env,
        agent_cfg=agent,
        log_dir=tmp_path / "run",
        repo_root=repo,
        runtime_argv=["--task", "Test-Symm-v0"],
        observation_dimension=64,
        action_dimension=12,
        direct_launch_context={
            "interface": r".\scripts\symm_locomotion\train.ps1",
            "argv": [r".\scripts\symm_locomotion\train.ps1", "--robot", "go2"],
        },
        initialization_path=initialization,
    )

    command = json.loads(command_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    assert command["runtime_training_argv"] == ["--task", "Test-Symm-v0"]
    assert command["direct_launch"]["argv"][0].endswith("train.ps1")
    assert cohort["eligibility_status"] == "incomplete"
    assert cohort["eligibility_is_result_independent"] is True
    assert cohort["result_quality_used_for_eligibility"] is False
    assert cohort["observation_dimension"] == 64
    assert cohort["action_dimension"] == 12
    assert cohort["initialization_record_sha256"] == PROVENANCE.sha256_file(initialization)

    numpy_state = np.arange(1000, dtype=np.uint32)
    changed_numpy_state = numpy_state.copy()
    changed_numpy_state[500] += 1
    assert PROVENANCE.state_sha256(numpy_state) != PROVENANCE.state_sha256(changed_numpy_state)


def test_policy_contract_resolves_native_history_and_rejects_width_mismatch():
    env = {"observations": {"policy": {"history_length": 30, "flatten_history_dim": True}}}
    agent = {
        "algorithm": {
            "symmetry_cfg": {
                "observation_contract_version": "hardware_proprio_history_64d_v1",
                "instantaneous_frame_dim": 64,
                "history_enabled": True,
                "history_length": 30,
                "history_packing": "term_major_oldest_to_newest_flattened",
                "history_trs_mode": "framewise_feature",
                "gait_phase_mapping_version": "phase-v4",
                "gait_library_version": "gait-v2",
            }
        }
    }

    contract = PROVENANCE.resolve_policy_contract(env, agent, 1920)

    assert contract == {
        "observation_contract_version": "hardware_proprio_history_64d_v1",
        "instantaneous_frame_dim": 64,
        "history_enabled": True,
        "history_length": 30,
        "history_packing": "term_major_oldest_to_newest_flattened",
        "history_trs_mode": "framewise_feature",
        "policy_input_dim": 1920,
        "gait_phase_mapping_version": "phase-v4",
        "gait_library_version": "gait-v2",
    }
    with pytest.raises(ValueError, match="expected 1920, received 64.*--history --history-length 30"):
        PROVENANCE.resolve_policy_contract(env, agent, 64)


def test_policy_contract_resolves_no_history_and_requires_matching_env_setting():
    env = {"observations": {"policy": {"history_length": 0, "flatten_history_dim": True}}}
    symmetry = {
        "observation_contract_version": "hardware_proprio_history_64d_v1",
        "instantaneous_frame_dim": 64,
        "history_enabled": False,
        "history_length": 0,
        "history_packing": "term_major_oldest_to_newest_flattened",
        "history_trs_mode": "framewise_feature",
        "gait_phase_mapping_version": "phase-v4",
        "gait_library_version": "gait-v2",
    }
    agent = {"algorithm": {"symmetry_cfg": symmetry}}

    assert PROVENANCE.resolve_policy_contract(env, agent, 64)["policy_input_dim"] == 64
    env["observations"]["policy"]["history_length"] = 30
    with pytest.raises(ValueError, match="Environment and algorithm policy-history settings differ"):
        PROVENANCE.resolve_policy_contract(env, agent, 64)


def test_non_time_reversal_runner_is_untouched(tmp_path):
    class ExplodingConfig:
        def to_dict(self):
            raise AssertionError("non-time-reversal config must not be serialized")

    runner = SimpleNamespace(alg=SimpleNamespace(gpu_global_rank=0))
    agent = {"algorithm": {"class_name": "rsl_rl.algorithms:PPO"}}

    result = PROVENANCE.record_training_initialization(
        runner=runner,
        env_cfg=ExplodingConfig(),
        agent_cfg=agent,
        log_dir=tmp_path / "run",
        repo_root=tmp_path,
    )

    assert result is None
    assert not (tmp_path / "run").exists()


def test_initialization_record_contains_required_pre_rollout_provenance(tmp_path):
    path, record = _record(tmp_path)

    assert path.name == "initialization.json"
    assert record["record_kind"] == "pre_first_rollout_initialization"
    for key in (
        "training_seed",
        "python_rng_state_sha256",
        "numpy_rng_state_sha256",
        "pytorch_cpu_rng_state_sha256",
        "pytorch_cuda_rng_state_sha256",
        "environment_seed",
        "actor_state_dict_sha256",
        "critic_state_dict_sha256",
        "actor_distribution_parameter_sha256",
        "optimizer_state_sha256",
        "resolved_env_config_sha256",
        "resolved_agent_config_sha256",
        "repo_commit",
        "dirty_tree_diff_sha256",
        "robot_asset_hashes",
    ):
        assert key in record
    assert record["robot_asset_hashes"]["assets"][0]["files"][0]["path"] == "urdf/robot.urdf"
    PROVENANCE._validate_record_digest(record, path)


def test_external_asset_reference_is_preserved_as_an_explicit_risk(tmp_path):
    assets = PROVENANCE.robot_asset_hashes(
        {"scene": {"robot": {"spawn": {"asset_path": "omniverse://server/Robots/test.usd"}}}},
        tmp_path,
    )

    assert assets["reference_count"] == 1
    assert assets["has_unresolved_risk"] is True
    assert assets["assets"][0]["status"] == "unresolved_external"
    assert assets["assets"][0]["reference"] == "omniverse://server/Robots/test.usd"


def test_cohort_reports_unresolved_assets_and_rejects_no_asset_references(tmp_path):
    record_path, record = _record(tmp_path)
    external = copy.deepcopy(record)
    external["robot_asset_hashes"] = PROVENANCE.robot_asset_hashes(
        {"asset_path": "omniverse://server/Robots/test.usd"}, tmp_path
    )
    _rewrite_record(record_path, external)
    manifest = {
        "treatment_variables": [],
        "runs": [
            {
                "run_name": "external",
                "resolved_condition": {"seed": 7},
                "initialization_record": str(record_path),
            }
        ],
    }

    unresolved = PROVENANCE.validate_cohort(manifest)

    assert unresolved["matched"] is False
    assert any("unresolved_external" in warning for warning in unresolved["warnings"])
    assert any("content-hashed robot assets" in error for error in unresolved["errors"])

    no_assets = copy.deepcopy(external)
    no_assets["robot_asset_hashes"] = {
        "assets": [],
        "reference_count": 0,
        "status_counts": {"hashed": 0, "unresolved_external": 0, "missing_local": 0},
        "has_unresolved_risk": False,
        "sha256": PROVENANCE.sha256_value([]),
    }
    _rewrite_record(record_path, no_assets)
    missing = PROVENANCE.validate_cohort(manifest)
    assert missing["matched"] is False
    assert any("no robot asset references" in error for error in missing["errors"])


def test_resume_links_to_and_validates_source_initialization(tmp_path):
    source_path, source_record = _record(tmp_path, "source_run")
    checkpoint = source_path.parents[1] / "model_100.pt"
    checkpoint.write_bytes(b"checkpoint")
    repo = tmp_path / "repo"
    asset = repo / "assets" / "robot" / "urdf" / "robot.urdf"
    env, agent = _configs(asset)

    resumed_path = PROVENANCE.record_training_initialization(
        runner=_runner(),
        env_cfg=env,
        agent_cfg=agent,
        log_dir=tmp_path / "resumed_run",
        repo_root=repo,
        resume_checkpoint=checkpoint,
    )

    resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resumed["resume"]["source_initialization_status"] == "validated"
    assert resumed["resume"]["source_initialization_record_sha256"] == source_record["record_sha256"]
    assert json.loads(source_path.read_text(encoding="utf-8")) == source_record


def test_cohort_rejects_different_initialization_even_if_declared_as_treatment(tmp_path):
    record_path, record = _record(tmp_path)
    second_path = tmp_path / "second.json"
    second = copy.deepcopy(record)
    second["actor_state_dict_sha256"] = "f" * 64
    _rewrite_record(second_path, second)
    manifest = {
        "treatment_variables": ["tr_policy.coefficient"],
        "runs": [
            {
                "run_name": "control",
                "resolved_condition": {"tr_policy": {"coefficient": 0.0}},
                "initialization_record": str(record_path),
            },
            {
                "run_name": "treated",
                "resolved_condition": {"tr_policy": {"coefficient": 0.1}},
                "initialization_record": str(second_path),
            },
        ],
    }

    result = PROVENANCE.validate_cohort(manifest)

    assert result["matched"] is False
    assert any("actor_state_dict_sha256 differs" in error for error in result["errors"])


def test_cohort_retains_failed_runs_and_rejects_undeclared_differences(tmp_path):
    record_path, _ = _record(tmp_path)
    manifest = {
        "treatment_variables": [],
        "runs": [
            {
                "run_name": "completed",
                "status": "completed",
                "resolved_condition": {"seed": 7},
                "initialization_record": str(record_path),
            },
            {
                "run_name": "failed",
                "status": "failed",
                "resolved_condition": {"seed": 8},
                "initialization_record": None,
            },
        ],
    }

    result = PROVENANCE.validate_cohort(manifest)

    assert result["run_count"] == 2
    assert result["validated_initialization_records"] == 1
    assert any("status=failed" in error for error in result["errors"])
    assert any("undeclared treatment difference at seed" in error for error in result["errors"])


def test_publication_record_is_bound_to_context_identity_and_seed(tmp_path):
    _, _, manifest = _study_record(tmp_path)

    matched = PROVENANCE.validate_cohort(manifest)

    assert matched["matched"] is True

    wrong_run = copy.deepcopy(manifest)
    wrong_run["runs"][0]["run_name"] = "relinked-run"
    relinked = PROVENANCE.validate_cohort(wrong_run)
    assert relinked["matched"] is False
    assert any("study-context run_name differs" in error for error in relinked["errors"])

    wrong_command = copy.deepcopy(manifest)
    wrong_command["runs"][0]["exact_training_command"] = ["different", "command"]
    command = PROVENANCE.validate_cohort(wrong_command)
    assert command["matched"] is False
    assert any("exact training command differs" in error for error in command["errors"])

    wrong_digest = copy.deepcopy(manifest)
    wrong_digest["runs"][0]["condition_sha256"] = "0" * 64
    digest = PROVENANCE.validate_cohort(wrong_digest)
    assert digest["matched"] is False
    assert any("cohort condition_sha256" in error for error in digest["errors"])

    wrong_study = copy.deepcopy(manifest)
    wrong_study["study_sha256"] = "f" * 64
    study = PROVENANCE.validate_cohort(wrong_study)
    assert study["matched"] is False
    assert any("study-context study_sha256 differs" in error for error in study["errors"])

    wrong_branch = copy.deepcopy(manifest)
    wrong_branch["expected_branch"] = "different-branch"
    branch = PROVENANCE.validate_cohort(wrong_branch)
    assert branch["matched"] is False
    assert any("study-context expected_branch differs" in error for error in branch["errors"])

    wrong_seed = copy.deepcopy(manifest)
    wrong_seed["runs"][0]["resolved_condition"]["seed"] = 8
    wrong_seed["runs"][0]["condition_sha256"] = PROVENANCE.sha256_value(wrong_seed["runs"][0]["resolved_condition"])
    seed = PROVENANCE.validate_cohort(wrong_seed)
    assert seed["matched"] is False
    assert any("environment_seed differs" in error for error in seed["errors"])


def test_publication_record_rejects_declared_seed_mismatch_before_write(tmp_path):
    with pytest.raises(ValueError, match="Study seed mismatch"):
        _study_record(tmp_path, declared_seed=8)

    assert not (tmp_path / "run" / PROVENANCE.INITIALIZATION_RELATIVE_PATH).exists()


def test_publication_record_rejects_absent_symmetry_runtime_config(tmp_path):
    repo, asset = _repository(tmp_path)
    env, agent = _configs(asset)
    condition = _study_condition()
    agent.value["run_name"] = "publication-run"
    branch = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_argv = ["--task", "Test-Symm-v0", "--seed", "7"]
    context = {
        "study_name": "publication",
        "study_sha256": "a" * 64,
        "expected_branch": branch,
        "treatment_variables": ["tr_policy.coefficient"],
        "run_name": "publication-run",
        "condition_sha256": PROVENANCE.sha256_value(condition),
        "resolved_condition": condition,
        "exact_training_command": ["isaaclab", "train", "--rl_library", "rsl_rl", *runtime_argv],
    }
    context_path = tmp_path / "missing-symmetry-context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")

    with pytest.raises(ValueError, match="condition did not resolve.*symmetry_cfg"):
        PROVENANCE.record_training_initialization(
            runner=_runner(),
            env_cfg=env,
            agent_cfg=agent,
            log_dir=tmp_path / "run",
            repo_root=repo,
            study_context_path=context_path,
            runtime_argv=runtime_argv,
        )

    assert not (tmp_path / "run" / PROVENANCE.INITIALIZATION_RELATIVE_PATH).exists()


def test_publication_record_rejects_ignored_or_mistyped_runtime_argv(tmp_path):
    repo, asset = _repository(tmp_path)
    env, agent = _configs(asset)
    condition = _study_condition()
    _align_configs_to_condition(env, agent, condition)
    agent.value["run_name"] = "publication-run"
    branch = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_argv = ["--task", "Test-Symm-v0", "--seed", "7", "agent.algorithm.clip_param=0.2"]
    context = {
        "study_name": "publication",
        "study_sha256": "b" * 64,
        "expected_branch": branch,
        "treatment_variables": [],
        "run_name": "publication-run",
        "condition_sha256": PROVENANCE.sha256_value(condition),
        "resolved_condition": condition,
        "exact_training_command": ["isaaclab", "train", "--rl_library", "rsl_rl", *expected_argv],
    }
    context_path = tmp_path / "wrong-argv-context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime command differs"):
        PROVENANCE.record_training_initialization(
            runner=_runner(),
            env_cfg=env,
            agent_cfg=agent,
            log_dir=tmp_path / "run",
            repo_root=repo,
            study_context_path=context_path,
            runtime_argv=[*expected_argv[:-1], "agent.algorithm.clip_pram=0.2"],
        )


def test_cohort_rejects_unmapped_resolved_env_and_agent_config_differences(tmp_path):
    record_path, record = _record(tmp_path)
    env_path = tmp_path / "env-different.json"
    env_record = copy.deepcopy(record)
    env_record["resolved_configs"]["environment"]["terminations"] = {"base_height": 0.1}
    env_record["resolved_env_config_sha256"] = PROVENANCE.sha256_value(env_record["resolved_configs"]["environment"])
    _rewrite_record(env_path, env_record)

    agent_path = tmp_path / "agent-different.json"
    agent_record = copy.deepcopy(record)
    agent_record["resolved_configs"]["agent"]["algorithm"]["clip_param"] = 0.3
    agent_record["resolved_agent_config_sha256"] = PROVENANCE.sha256_value(agent_record["resolved_configs"]["agent"])
    _rewrite_record(agent_path, agent_record)
    manifest = {
        "treatment_variables": [],
        "runs": [
            {"run_name": "reference", "resolved_condition": {"seed": 7}, "initialization_record": str(record_path)},
            {"run_name": "env", "resolved_condition": {"seed": 7}, "initialization_record": str(env_path)},
            {"run_name": "agent", "resolved_condition": {"seed": 7}, "initialization_record": str(agent_path)},
        ],
    }

    result = PROVENANCE.validate_cohort(manifest)

    assert result["matched"] is False
    assert any("resolved invariant environment config differs" in error for error in result["errors"])
    assert any("resolved invariant agent config differs" in error for error in result["errors"])


def test_declared_policy_treatment_masks_only_mapped_resolved_paths(tmp_path):
    first_path, record = _record(tmp_path)
    second_path = tmp_path / "policy-treated.json"
    first = copy.deepcopy(record)
    second = copy.deepcopy(record)
    for candidate, coefficient in ((first, 0.0), (second, 0.1)):
        candidate["resolved_configs"]["agent"]["algorithm"]["symmetry_cfg"] = {
            "mirror_loss_coeff": coefficient,
            "tr_policy_schedule": {"target_coeff": coefficient},
        }
        candidate["resolved_agent_config_sha256"] = PROVENANCE.sha256_value(candidate["resolved_configs"]["agent"])
    _rewrite_record(first_path, first)
    _rewrite_record(second_path, second)
    manifest = {
        "treatment_variables": ["tr_policy.coefficient"],
        "runs": [
            {
                "run_name": "control",
                "resolved_condition": {"seed": 7, "tr_policy": {"coefficient": 0.0}},
                "initialization_record": str(first_path),
            },
            {
                "run_name": "policy",
                "resolved_condition": {"seed": 7, "tr_policy": {"coefficient": 0.1}},
                "initialization_record": str(second_path),
            },
        ],
    }

    result = PROVENANCE.validate_cohort(manifest)

    assert result["matched"] is True
    assert result["runtime_treatment_paths"]["agent"] == [
        "algorithm.symmetry_cfg.mirror_loss_coeff",
        "algorithm.symmetry_cfg.tr_policy_schedule.target_coeff",
    ]


def test_reward_and_budget_treatments_mask_runtime_fields_and_signatures(tmp_path):
    first_path, record = _record(tmp_path)
    second_path = tmp_path / "reward-budget-treated.json"
    first = copy.deepcopy(record)
    second = copy.deepcopy(record)
    first["resolved_configs"]["environment"]["rewards"]["tracking"]["weight"] = 1.0
    second["resolved_configs"]["environment"]["rewards"]["tracking"]["weight"] = 2.0
    first["resolved_configs"]["environment"]["scene"]["num_envs"] = 8
    second["resolved_configs"]["environment"]["scene"]["num_envs"] = 16
    first["resolved_configs"]["agent"]["max_iterations"] = 100
    second["resolved_configs"]["agent"]["max_iterations"] = 200
    for candidate in (first, second):
        candidate["resolved_env_config_sha256"] = PROVENANCE.sha256_value(candidate["resolved_configs"]["environment"])
        candidate["resolved_agent_config_sha256"] = PROVENANCE.sha256_value(candidate["resolved_configs"]["agent"])
    second["matched_signatures"]["reward_profile"] = "1" * 64
    second["matched_signatures"]["training_budget"] = "2" * 64
    _rewrite_record(first_path, first)
    _rewrite_record(second_path, second)
    manifest = {
        "treatment_variables": ["reward_profile.overrides", "training_iterations", "num_envs"],
        "runs": [
            {
                "run_name": "control",
                "resolved_condition": {
                    "seed": 7,
                    "reward_profile": {"overrides": ["env.rewards.tracking.weight=1.0"]},
                    "training_iterations": 100,
                    "num_envs": 8,
                },
                "initialization_record": str(first_path),
            },
            {
                "run_name": "treated",
                "resolved_condition": {
                    "seed": 7,
                    "reward_profile": {"overrides": ["env.rewards.tracking.weight=2.0"]},
                    "training_iterations": 200,
                    "num_envs": 16,
                },
                "initialization_record": str(second_path),
            },
        ],
    }

    result = PROVENANCE.validate_cohort(manifest)

    assert result["matched"] is True
    assert result["treatment_masked_signatures"] == ["reward_profile", "training_budget"]
    assert result["runtime_treatment_paths"]["environment"] == [
        "rewards.tracking.weight",
        "scene.num_envs",
    ]
    assert result["runtime_treatment_paths"]["agent"] == ["max_iterations"]


def test_maintained_training_hook_runs_after_dump_and_before_learn():
    train_path = Path(__file__).resolve().parents[2] / "reinforcement_learning" / "rsl_rl" / "train_rsl_rl.py"
    source = train_path.read_text(encoding="utf-8")

    assert '"--symm_study_context"' in source
    assert source.index("dump_train_configs(log_dir, env_cfg, agent_cfg)") < source.index(
        "SYMM_PROVENANCE.record_training_initialization"
    )
    assert source.index("SYMM_PROVENANCE.record_training_initialization") < source.index("runner.learn(")
