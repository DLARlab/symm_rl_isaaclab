# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "study_registry.py"
    name = "_test_study_registry"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _facts(state: str = "complete"):
    return {
        "reward_profile": "reward_v1",
        "gait_library": "gait_v2",
        "observation_dim": 72,
        "action_dim": 12,
        "training_budget": {"max_iterations": 20000},
        "initialization": "fresh",
        "evaluation_protocol": "full_v2",
        "code_hashes": {
            "archived_git_commit": "a" * 40,
            "source_snapshot_sha256": "b" * 64,
            "agent_config_sha256": "c" * 64,
            "env_config_sha256": "d" * 64,
        },
        "checkpoint_status": {"state": state, "sha256": "e" * 64},
    }


def _protocol():
    return {
        "required": {
            "reward_profile": "reward_v1",
            "gait_library": "gait_v2",
            "observation_dim": 72,
            "action_dim": 12,
            "training_budget.max_iterations": 20000,
            "initialization": "fresh",
            "evaluation_protocol": "full_v2",
        },
        "allowed": {
            "code_hashes.archived_git_commit": ["a" * 40],
            "code_hashes.source_snapshot_sha256": ["b" * 64],
        },
        "required_present": ["checkpoint_status.sha256"],
    }


def test_cohort_eligibility_is_independent_of_result_quality():
    module = _load_module()
    first = {
        "run_id": "first",
        "design_role": "main",
        "facts": _facts(),
        "outcomes": {"final_reward": 100.0, "favorable": True},
    }
    second = deepcopy(first)
    second["run_id"] = "second"
    second["outcomes"] = {"final_reward": -100.0, "favorable": False}

    assert module.classify_run(first, _protocol()) == ("eligible_main", [])
    assert module.classify_run(second, _protocol()) == ("eligible_main", [])


def test_registry_forbids_performance_derived_eligibility_constraints():
    module = _load_module()
    run = {
        "run_id": "run",
        "path": "logs/run",
        "design_role": "main",
        "classification": "eligible_main",
        "facts": {**_facts(), "outcomes": {"final_reward": 100.0}},
    }
    protocol = deepcopy(_protocol())
    protocol["required"]["outcomes.final_reward"] = 100.0
    manifest = {
        "schema_version": 1,
        "cohort_id": "invalid_performance_selected_cohort",
        "selection_basis": "protocol_facts_only",
        "protocol": protocol,
        "runs": [run],
    }

    classification, reasons = module.classify_run(run, protocol)

    assert classification == "ineligible_protocol"
    assert any("performance-derived and forbidden" in reason for reason in reasons)
    assert any("performance-derived and forbidden" in error for error in module.validate_manifest(manifest))


def test_registry_classifies_protocol_failure_incomplete_and_historical_runs():
    module = _load_module()
    supplement = {"design_role": "supplement", "facts": _facts()}
    assert module.classify_run(supplement, _protocol())[0] == "eligible_supplement"

    failed = {"design_role": "main", "facts": _facts("failed")}
    incomplete = {"design_role": "main", "facts": _facts("incomplete")}
    mismatch = {"design_role": "main", "facts": {**_facts(), "observation_dim": 71}}
    historical = {"design_role": "historical_diagnostic", "facts": {**_facts(), "observation_dim": 71}}
    assert module.classify_run(failed, _protocol())[0] == "failed"
    assert module.classify_run(incomplete, _protocol())[0] == "incomplete"
    assert module.classify_run(mismatch, _protocol())[0] == "ineligible_protocol"
    assert module.classify_run(historical, _protocol())[0] == "historical_diagnostic"


def test_initial_registry_manifests_define_requested_cohorts_and_go2_split():
    module = _load_module()
    registry_dir = Path(__file__).resolve().parents[1] / "study_registry"
    expected = {
        "go2_gait_v2_development_sweep",
        "go2_v4_main_development",
        "x1_gait_v2_development_sweep",
        "x1_v4_main_development",
        "go2_confirmatory_cohort",
        "x1_confirmatory_cohort",
    }
    manifests = {path.stem: module.load_manifest(path) for path in registry_dir.glob("*.json")}
    assert set(manifests) == expected

    go2 = manifests["go2_gait_v2_development_sweep"]
    assert len(go2["runs"]) == 8
    assert sum(run["classification"] == "eligible_main" for run in go2["runs"]) == 5
    assert sum(run["classification"] == "eligible_supplement" for run in go2["runs"]) == 3
    archived_go2 = [run for run in go2["runs"] if run["path"].startswith("logs/rsl_rl/good_runs_72d/")]
    assert len(archived_go2) == 4
    assert all(run["artifact_availability"] == "local_only" for run in archived_go2)
    assert go2["statistical_claim_status"] == "not_confirmatory_seed_42_only"
    assert [entry["classification"] for entry in module.classify_manifest(go2)] == [
        run["classification"] for run in go2["runs"]
    ]

    assert len(manifests["x1_gait_v2_development_sweep"]["runs"]) == 3

    for cohort_id in ("go2_v4_main_development", "x1_v4_main_development"):
        development = manifests[cohort_id]
        assert len(development["runs"]) == 5
        assert all(run["classification"] == "eligible_main" for run in development["runs"])
        assert all(run["facts"]["checkpoint_status"]["state"] == "complete" for run in development["runs"])
        assert development["common_facts"]["evaluation_protocol"] == ("leg_usage_grid_full_v3_with_paired_tr_phase_v1")
        assert development["common_facts"]["evaluation_seed"] == 4242
        assert development["statistical_claim_status"] == "not_confirmatory_seed_42_only"
        assert [entry["classification"] for entry in module.classify_manifest(development)] == [
            run["classification"] for run in development["runs"]
        ]

    assert manifests["go2_confirmatory_cohort"]["runs"] == []
    assert manifests["x1_confirmatory_cohort"]["runs"] == []

    for manifest in manifests.values():
        assert module.validate_manifest(manifest) == []
        for run in manifest["runs"]:
            if run["artifact_availability"] == "tracked":
                assert run["path"].startswith("logs/rsl_rl/good_runs_64d/")


def test_malformed_facts_report_validation_errors_instead_of_crashing():
    module = _load_module()
    manifest = {
        "schema_version": 1,
        "cohort_id": "malformed",
        "selection_basis": "protocol_facts_only",
        "protocol": _protocol(),
        "runs": [
            {
                "run_id": "bad-facts",
                "path": "logs/rsl_rl/good_runs_64d/robot/run",
                "artifact_availability": "tracked",
                "design_role": "main",
                "classification": "ineligible_protocol",
                "facts": [],
                "artifacts": {},
            }
        ],
    }

    errors = module.validate_manifest(manifest)

    assert any("facts must be a mapping" in error for error in errors)


def test_protocol_fact_hashes_must_match_registered_artifacts():
    module = _load_module()
    facts = _facts()
    facts["checkpoint_status"]["final_iteration"] = 19999
    run = {
        "run_id": "bound-run",
        "path": "logs/rsl_rl/good_runs_64d/robot/run",
        "artifact_availability": "tracked",
        "design_role": "main",
        "classification": "eligible_main",
        "facts": facts,
        "artifacts": {
            "git/symm_rl_isaaclab.diff": "b" * 64,
            "params/agent.yaml": "c" * 64,
            "params/env.yaml": "d" * 64,
            "model_19999.pt": "e" * 64,
        },
    }
    manifest = {
        "schema_version": 1,
        "cohort_id": "bound",
        "selection_basis": "protocol_facts_only",
        "protocol": _protocol(),
        "runs": [run],
    }
    assert module.validate_manifest(manifest) == []

    run["artifacts"]["params/env.yaml"] = "f" * 64

    assert any("params/env.yaml" in error for error in module.validate_manifest(manifest))


def test_registry_accepts_only_the_64d_tracked_archive_root():
    module = _load_module()
    run = {
        "run_id": "bound-run",
        "path": "logs/rsl_rl/good_runs_64d/robot/run",
        "artifact_availability": "tracked",
        "design_role": "main",
        "classification": "eligible_main",
        "facts": {**_facts(), "checkpoint_status": {**_facts()["checkpoint_status"], "final_iteration": 1}},
        "artifacts": {
            "git/symm_rl_isaaclab.diff": "b" * 64,
            "params/agent.yaml": "c" * 64,
            "params/env.yaml": "d" * 64,
            "model_1.pt": "e" * 64,
        },
    }
    manifest = {
        "schema_version": 1,
        "cohort_id": "archive-roots",
        "selection_basis": "protocol_facts_only",
        "protocol": _protocol(),
        "runs": [run],
    }

    assert module.validate_manifest(manifest) == []
    run["path"] = "logs/rsl_rl/good_runs/robot/run"
    assert any("logs/rsl_rl/good_runs_64d" in error for error in module.validate_manifest(manifest))


def test_missing_local_only_artifacts_are_reported_as_skipped(tmp_path):
    module = _load_module()
    skipped = []
    manifest = {
        "runs": [
            {
                "run_id": "remote-only",
                "path": "logs/rsl_rl/robot/remote-only",
                "artifact_availability": "local_only",
                "artifacts": {},
            }
        ]
    }

    assert module.verify_manifest_artifacts(manifest, tmp_path, skipped) == []
    assert len(skipped) == 1
    assert "local-only artifact folder is unavailable" in skipped[0]


def test_tracked_run_path_cannot_traverse_outside_good_runs_64d(tmp_path):
    module = _load_module()
    manifest = {
        "schema_version": 1,
        "cohort_id": "traversal",
        "selection_basis": "protocol_facts_only",
        "protocol": _protocol(),
        "runs": [
            {
                "run_id": "escape",
                "path": "logs/rsl_rl/good_runs_64d/../../../outside",
                "artifact_availability": "tracked",
                "design_role": "main",
                "classification": "eligible_main",
                "facts": {**_facts(), "checkpoint_status": {**_facts()["checkpoint_status"], "final_iteration": 1}},
                "artifacts": {
                    "git/symm_rl_isaaclab.diff": "b" * 64,
                    "params/agent.yaml": "c" * 64,
                    "params/env.yaml": "d" * 64,
                    "model_1.pt": "e" * 64,
                },
            }
        ],
    }

    assert any("without '..'" in error for error in module.validate_manifest(manifest))
    assert any("escapes" in error for error in module.verify_manifest_artifacts(manifest, tmp_path))
