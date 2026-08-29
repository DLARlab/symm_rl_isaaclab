# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "launch_study.py"
    spec = importlib.util.spec_from_file_location("_test_launch_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_module()


@pytest.fixture(autouse=True)
def _stable_test_branch(monkeypatch):
    monkeypatch.setattr(LAUNCHER.SYMM_CLI, "git_branch", lambda: "test-branch")


def _schedule() -> dict:
    return {
        "enabled": True,
        "warmup_iterations": 10,
        "rampup_iterations": 20,
        "hold_iterations": 30,
        "decay_iterations": 40,
        "final_scale": 0.5,
        "ramp_shape": "half_cosine",
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "study_name": "publication day 1",
        "expected_branch": "test-branch",
        "treatment_variables": ["tr_policy.coefficient"],
        "common": {
            "robot": "go2",
            "seed": 42,
            "tr_policy": {"enabled": True, "coefficient": 0.1, "schedule": _schedule()},
            "tr_value": {"enabled": True, "coefficient": 0.05, "schedule": _schedule()},
            "validity_mask": {
                "mode": "command_tracking_upright_phase",
                "thresholds": {
                    "min_abs_command_velocity": 0.1,
                    "tracking_abs_tolerance": 0.2,
                    "tracking_rel_tolerance": 0.3,
                    "projected_gravity_tolerance": 0.4,
                    "phase_boundary_margin": 0.04,
                    "command_speed_bin_edges": [0.4, 0.8, 1.2],
                },
            },
            "trajectory_augmentation": {
                "enabled": True,
                "mode": "dynamics_filtered_reverse_action_supervision",
                "action_source": "learned",
                "coefficient": 0.03,
                "schedule": _schedule(),
                "filter_settings": {
                    "filter_enabled": True,
                    "validation_fraction": 0.25,
                    "validation_quantile": 0.9,
                    "threshold_multiplier": 1.5,
                    "minimum_validation_samples": 64,
                    "maximum_validation_loss": 0.8,
                    "phase_boundary_margin": 0.02,
                    "maximum_contact_impulse": 15.0,
                    "action_abs_limit": 8.0,
                    "max_augmented_to_original_ratio": 0.2,
                    "use_confidence_weights": False,
                    "rng_seed": 123,
                },
            },
            "reward_profile": {
                "name": "publication-v1",
                "overrides": ["env.rewards.alive_bonus.weight=0.2"],
            },
            "leg_sync_reward_weight": 0.2,
            "training_iterations": 20000,
            "num_envs": 512,
            "evaluation_protocol": "light",
        },
        "conditions": [
            {"label": "control", "tr_policy": {"coefficient": 0.0}},
            {"label": "policy", "tr_policy": {"coefficient": 0.1}},
        ],
    }


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        conda_env=LAUNCHER.SYMM_CLI.CONDA_ENV,
        no_conda_run=True,
        use_conda_run=False,
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_plan_has_deterministic_names_exact_commands_and_evaluation_protocol(tmp_path):
    manifest_path = tmp_path / "study.json"
    _write_manifest(manifest_path, _manifest())
    output_dir = tmp_path / "output"

    first = LAUNCHER.load_plan(manifest_path, output_dir, _args())
    second = LAUNCHER.load_plan(manifest_path, output_dir, _args())

    assert [run["run_name"] for run in first["runs"]] == [run["run_name"] for run in second["runs"]]
    assert len({run["run_name"] for run in first["runs"]}) == 2
    command = first["runs"][1]["exact_training_command"]
    command_text = first["runs"][1]["exact_training_command_text"]
    assert "--symm_study_context" in command
    assert "agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true" in command
    assert "agent.algorithm.symmetry_cfg.use_tr_value_consistency=true" in command
    assert "agent.algorithm.symmetry_cfg.tr_policy_schedule.target_coeff=0.1" in command
    assert "agent.algorithm.symmetry_cfg.tr_value_schedule.target_coeff=0.05" in command
    assert "agent.algorithm.symmetry_cfg.tr_validity.mode=command_tracking_upright_phase" in command
    assert "agent.algorithm.symmetry_cfg.tr_augmentation.action_source=learned_inverse" in command
    assert "agent.algorithm.symmetry_cfg.tr_augmentation.filter_enabled=true" in command
    assert "env.rewards.alive_bonus.weight=0.2" in command
    assert "env.rewards.leg_permutation_symmetry.weight=0.2" in command
    assert command_text
    evaluation = first["runs"][1]["evaluation_command_template"]
    assert evaluation[2] == "evaluation"
    assert evaluation[-1] == "--no_conda_run"
    assert evaluation[evaluation.index("--expected_branch") + 1] == _manifest()["expected_branch"]
    assert evaluation[evaluation.index("--conda_env") + 1] == LAUNCHER.SYMM_CLI.CONDA_ENV
    assert "light" in evaluation
    assert first["selection_policy"].startswith("retain_all_runs")


def test_dry_run_writes_nothing(tmp_path, capsys):
    manifest_path = tmp_path / "study.json"
    _write_manifest(manifest_path, _manifest())
    output_dir = tmp_path / "never-created"

    code = LAUNCHER.main([str(manifest_path), "--output-dir", str(output_dir), "--no-conda-run", "--dry-run"])

    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert not output_dir.exists()
    assert len(output["runs"]) == 2
    assert all(run["status"] == "planned" for run in output["runs"])


def test_in_repository_output_directory_must_be_git_ignored(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".gitignore").write_text("logs/\n", encoding="utf-8")
    subprocess = LAUNCHER.subprocess
    subprocess.run(["git", "-C", str(repository), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "add", ".gitignore"], check=True, capture_output=True)
    monkeypatch.setattr(LAUNCHER.SYMM_CLI, "repo_root", lambda: repository)

    with pytest.raises(ValueError, match="must be ignored by Git"):
        LAUNCHER._validate_output_dir_is_stable(repository / "study-output")

    LAUNCHER._validate_output_dir_is_stable(repository / "logs" / "studies" / "safe-output")


def test_duplicate_resolved_conditions_are_rejected(tmp_path):
    manifest = _manifest()
    manifest["treatment_variables"] = []
    manifest["conditions"][1]["tr_policy"]["coefficient"] = 0.0
    path = tmp_path / "duplicates.json"
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="Duplicate resolved conditions"):
        LAUNCHER.load_plan(path, tmp_path / "output", _args())


def test_undeclared_condition_difference_is_rejected(tmp_path):
    manifest = _manifest()
    manifest["treatment_variables"] = ["tr_value.coefficient"]
    path = tmp_path / "undeclared.json"
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="not declared treatments"):
        LAUNCHER.load_plan(path, tmp_path / "output", _args())


def test_enabled_augmentation_cannot_disable_the_dynamics_filter():
    condition = _manifest()["common"]
    condition["trajectory_augmentation"]["filter_settings"]["filter_enabled"] = False

    with pytest.raises(ValueError, match="requires filter_settings.filter_enabled=true"):
        LAUNCHER.normalize_condition(condition)


def test_unknown_augmentation_mode_is_rejected_before_launch():
    condition = _manifest()["common"]
    condition["trajectory_augmentation"]["mode"] = "typo_reverse_mode"

    with pytest.raises(ValueError, match="trajectory_augmentation.mode"):
        LAUNCHER.normalize_condition(condition)


def test_reward_profile_name_only_cannot_create_duplicate_training_conditions(tmp_path):
    manifest = _manifest()
    manifest["treatment_variables"] = ["reward_profile.name"]
    manifest["conditions"] = [
        {"label": "profile-a"},
        {"label": "profile-b", "reward_profile": {"name": "renamed-profile", "overrides": []}},
    ]
    manifest["common"]["reward_profile"] = {"name": "profile", "overrides": []}
    path = tmp_path / "profile-labels.json"
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="Semantically duplicate training conditions"):
        LAUNCHER.load_plan(path, tmp_path / "output", _args())


def test_expected_branch_is_required_and_checked_before_writes(tmp_path):
    manifest = _manifest()
    manifest["expected_branch"] = "definitely-not-this-branch"
    path = tmp_path / "wrong-branch.json"
    output_dir = tmp_path / "output"
    _write_manifest(path, manifest)

    code = LAUNCHER.main([str(path), "--output-dir", str(output_dir), "--dry-run"])

    assert code == 2
    assert not output_dir.exists()


def test_protocol_and_robot_participate_in_condition_identity(tmp_path):
    manifest = _manifest()
    manifest["treatment_variables"] = ["evaluation_protocol", "robot"]
    manifest["conditions"][0] = {"label": "go2-light", "evaluation_protocol": "light", "robot": "go2"}
    manifest["conditions"][1] = {"label": "x1-full", "evaluation_protocol": "full", "robot": "x1"}
    path = tmp_path / "protocols.json"
    _write_manifest(path, manifest)

    plan = LAUNCHER.load_plan(path, tmp_path / "output", _args())

    names = [run["run_name"] for run in plan["runs"]]
    assert "go2" in names[0] and "light" in names[0]
    assert "x1" in names[1] and "full" in names[1]
    assert plan["runs"][0]["condition_sha256"] != plan["runs"][1]["condition_sha256"]


def test_failed_runs_are_retained_and_later_conditions_still_launch(tmp_path, monkeypatch):
    manifest_path = tmp_path / "study.json"
    _write_manifest(manifest_path, _manifest())
    output_dir = tmp_path / "retained"
    plan = LAUNCHER.load_plan(manifest_path, output_dir, _args())
    calls = []

    def fail_or_return(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise OSError("launcher unavailable")
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(LAUNCHER.subprocess, "run", fail_or_return)
    monkeypatch.setattr(LAUNCHER, "_run_snapshot", lambda condition: set())

    code = LAUNCHER.execute_plan(plan, output_dir)

    cohort = json.loads((output_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
    assert code == 1
    assert len(calls) == 2
    assert [run["status"] for run in cohort["runs"]] == ["failed", "failed"]
    assert [run["exit_code"] for run in cohort["runs"]] == [127, 9]
    assert len(list((output_dir / "contexts").glob("*.json"))) == 2


def test_resume_skips_completed_runs_and_retries_failed_runs(tmp_path, monkeypatch):
    manifest_path = tmp_path / "study.json"
    _write_manifest(manifest_path, _manifest())
    output_dir = tmp_path / "resumable"
    plan = LAUNCHER.load_plan(manifest_path, output_dir, _args())
    outcomes = iter((0, 9))
    calls = []

    def run_first_pass(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=next(outcomes))

    monkeypatch.setattr(LAUNCHER.subprocess, "run", run_first_pass)
    monkeypatch.setattr(LAUNCHER, "_run_snapshot", lambda condition: set())
    monkeypatch.setattr(LAUNCHER.PROVENANCE, "validate_cohort", lambda cohort: {"matched": True})

    assert LAUNCHER.execute_plan(plan, output_dir) == 1
    assert len(calls) == 2

    resumed_plan = LAUNCHER.load_plan(manifest_path, output_dir, _args())
    calls.clear()

    def run_resume(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(LAUNCHER.subprocess, "run", run_resume)

    assert LAUNCHER.execute_plan(resumed_plan, output_dir, resume=True) == 0

    cohort = json.loads((output_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert calls[0][0] == resumed_plan["runs"][1]["exact_training_command"]
    assert [run["status"] for run in cohort["runs"]] == ["completed", "completed"]
    assert [run["attempts"] for run in cohort["runs"]] == [1, 2]


def test_resume_rejects_changed_immutable_plan(tmp_path, monkeypatch):
    manifest_path = tmp_path / "study.json"
    _write_manifest(manifest_path, _manifest())
    output_dir = tmp_path / "tampered"
    plan = LAUNCHER.load_plan(manifest_path, output_dir, _args())
    monkeypatch.setattr(LAUNCHER.subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(LAUNCHER, "_run_snapshot", lambda condition: set())
    monkeypatch.setattr(LAUNCHER.PROVENANCE, "validate_cohort", lambda cohort: {"matched": True})
    assert LAUNCHER.execute_plan(plan, output_dir) == 0

    stored_plan_path = output_dir / "study_plan.json"
    stored_plan = json.loads(stored_plan_path.read_text(encoding="utf-8"))
    stored_plan["study_sha256"] = "0" * 64
    _write_manifest(stored_plan_path, stored_plan)
    fresh_plan = LAUNCHER.load_plan(manifest_path, output_dir, _args())

    with pytest.raises(ValueError, match="study_plan.json does not match"):
        LAUNCHER.execute_plan(fresh_plan, output_dir, resume=True)


def test_every_emitted_nested_override_exists_in_project_local_config():
    config_path = (
        LAUNCHER.SYMM_CLI.repo_root()
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "symm_quadruped"
        / "time_reversal_cfg.py"
    )
    tree = ast.parse(config_path.read_text(encoding="utf-8"))

    def fields(class_name: str) -> set[str]:
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        return {
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        }

    schedule_fields = fields("TimeReversalScheduleCfg")
    validity_fields = fields("TimeReversalValidityMaskCfg")
    augmentation_fields = fields("TimeReversalAugmentationCfg")
    symmetry_fields = fields("TimeReversalSymmetryCfg")

    assert set(LAUNCHER.SCHEDULE_DEFAULTS) | {"target_coeff"} <= schedule_fields
    assert {"tr_policy_schedule", "tr_value_schedule", "tr_validity", "tr_augmentation"} <= symmetry_fields
    assert {"mode", *LAUNCHER.VALIDITY_DEFAULTS["thresholds"]} <= validity_fields
    assert {
        "enabled",
        "mode",
        "action_source",
        "coefficient",
        "schedule",
        *LAUNCHER.AUGMENTATION_DEFAULTS["filter_settings"],
    } <= augmentation_fields
