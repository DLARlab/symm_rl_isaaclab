# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the reusable gait-closure comparison."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

comparison = importlib.import_module("comparison")
tensorboard_scalars = importlib.import_module("_tensorboard_scalars")


FAMILIES = ("trot", "bound", "half_bound", "gallop")
FAMILY_GAIT_COUNTS = {"trot": 1, "bound": 1, "half_bound": 4, "gallop": 4}
VELOCITIES = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
EXPECTED_FIGURE_IDS = {
    "fig01",
    "fig02",
    "fig03",
    "fig03_01",
    "fig03_02",
    "fig03_03",
    "fig04",
    "fig05",
    "fig05_01",
    "fig05_02",
    "fig05_03",
    "fig06",
    "fig07",
    "fig07_01",
    "fig07_02",
    "fig07_03",
}


def _metric_ids(registry: Mapping[str, Any] | Sequence[Any]) -> tuple[str, ...]:
    """Return ordered public metric identifiers from a registry."""
    if isinstance(registry, Mapping):
        return tuple(str(key) for key in registry)
    result: list[str] = []
    for entry in registry:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, Mapping):
            result.append(str(entry.get("metric", entry.get("id"))))
        else:
            result.append(str(getattr(entry, "metric", None) or getattr(entry, "id")))
    return tuple(result)


def _figure_ids(registry: Mapping[str, Any] | Sequence[Any]) -> tuple[str, ...]:
    """Return figure identifiers without constraining registry implementation."""
    if isinstance(registry, Mapping):
        return tuple(str(key) for key in registry)
    result: list[str] = []
    for entry in registry:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, Mapping):
            result.append(str(entry.get("figure_id", entry.get("id"))))
        else:
            result.append(str(getattr(entry, "figure_id", None) or getattr(entry, "id")))
    return tuple(result)


def _synthetic_runs(count: int = 5) -> list[dict[str, Any]]:
    """Build the minimal public run schema accepted by the aggregator."""
    nonbaseline_palette = comparison.V2_PALETTE[1:]
    return [
        {
            "run_id": f"run_{index}",
            "abbreviation": f"R{index}",
            "color": "#202124" if index == 0 else nonbaseline_palette[(index - 1) % len(nonbaseline_palette)],
            "is_baseline": index == 0,
        }
        for index in range(count)
    ]


def _synthetic_cells(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build a complete unequal-family grid whose pooled mean is detectably wrong."""
    family_values = {"trot": 1.0, "bound": 2.0, "half_bound": 30.0, "gallop": 40.0}
    rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        for family in FAMILIES:
            for gait_variant in range(FAMILY_GAIT_COUNTS[family]):
                for velocity in VELOCITIES:
                    direction_offset = 100.0 if velocity > 0.0 else 0.0
                    value = family_values[family] + direction_offset + run_index
                    row: dict[str, Any] = {
                        **run,
                        "cell_id": f"{run['run_id']}__{family}_{gait_variant}__{velocity:+.1f}",
                        "gait_index": gait_variant,
                        "gait_name": f"{family}_{gait_variant}",
                        "family": family,
                        "velocity_mps": velocity,
                        "seed": 4242,
                        "velocity_vx_rmse_mps": value,
                        "velocity_yaw_rmse_radps": value / 10.0,
                        "gait_agreement_boundary_excluded_percent": family_values[family] + run_index,
                    }
                    for metric, metadata in comparison.LEG_METRICS.items():
                        row[f"{metric}_abs_imbalance_percent"] = value
                        row[f"{metadata['source_metric']}_abs_imbalance_percent"] = value
                    rows.append(row)
    return rows


def _one(rows: Sequence[Mapping[str, Any]], **selectors: Any) -> Mapping[str, Any]:
    """Return the unique summary row matching all selectors."""
    selected = [row for row in rows if all(row.get(key) == value for key, value in selectors.items())]
    assert len(selected) == 1, selectors
    return selected[0]


def _complete_overall_metrics(expected_cells: int) -> dict[str, Any]:
    """Build complete analyzer coverage for progress-validation tests."""
    coverage_fields = (
        "expected_cells",
        "valid_cells",
        "velocity_valid_cells",
        "heading_valid_cells",
        "gait_valid_cells",
        "raw_load_valid_cells",
        "grf_load_valid_cells",
        "normalized_load_valid_cells",
    )
    return {
        "coverage": {"complete": True, **{field: expected_cells for field in coverage_fields}},
        "metrics": {
            str(metadata["source_metric"]): {"complete": True, "valid_cells": expected_cells}
            for metadata in comparison.LEG_METRICS.values()
        },
    }


@pytest.mark.parametrize("skipped_cells", [0, 59])
def test_evaluation_completeness_accepts_metric_complete_termination_and_resume(skipped_cells: int):
    progress = {
        "status": "complete",
        "total_cells": 60,
        "completed_cells": 60,
        "successful_cells": 59,
        "terminated_cells": 1,
        "skipped_cells": skipped_cells,
    }

    comparison._validate_evaluation_completeness(progress, _complete_overall_metrics(60), 60, "History-m0.1")


def test_evaluation_completeness_rejects_inconsistent_outcome_accounting():
    progress = {
        "status": "complete",
        "total_cells": 60,
        "completed_cells": 60,
        "successful_cells": 58,
        "terminated_cells": 1,
        "skipped_cells": 0,
    }

    with pytest.raises(ValueError, match=r"successful_cells \+ terminated_cells"):
        comparison._validate_evaluation_completeness(progress, _complete_overall_metrics(60), 60, "History-m0.1")


def test_metric_complete_cell_outcome_accepts_termination_but_rejects_disagreement():
    assert comparison._validated_cell_outcome_status({"status": "valid", "terminated": "False"}, "run/cell") == "valid"
    assert (
        comparison._validated_cell_outcome_status({"status": "terminated", "terminated": "True"}, "run/cell")
        == "terminated"
    )

    with pytest.raises(ValueError, match="no metric-complete outcome"):
        comparison._validated_cell_outcome_status({"status": "failed", "terminated": "False"}, "run/cell")
    with pytest.raises(ValueError, match="outcome fields disagree"):
        comparison._validated_cell_outcome_status({"status": "terminated", "terminated": "False"}, "run/cell")


def test_evaluation_outcome_warning_names_retained_terminated_cell():
    warnings = comparison._evaluation_outcome_warnings(
        [
            {
                "run": {"abbreviation": "History-m0.1"},
                "cells": [
                    {"cell_id": "completed", "status": "valid"},
                    {"cell_id": "gait_08_gallop__vx_neg_0p5__seed_0042", "status": "terminated"},
                ],
            }
        ]
    )

    assert len(warnings) == 1
    assert "History-m0.1" in warnings[0]
    assert "gait_08_gallop__vx_neg_0p5__seed_0042" in warnings[0]
    assert "outcome-free aggregation" in warnings[0]


def test_default_run_roots_use_the_64d_history_archive(tmp_path: Path):
    assert comparison.default_run_roots(tmp_path) == (
        tmp_path / "logs/rsl_rl/unitree_go2_symm_flat",
        tmp_path / "logs/rsl_rl/good_runs_64d/unitree_go2_symm_flat",
    )


def test_parse_run_spec_is_explicit_and_preserves_user_text():
    assert comparison.parse_run_spec("NoTRS=2026-08-29_run") == ("NoTRS", "2026-08-29_run")
    assert comparison.parse_run_spec("A=folder=with=equals") == ("A", "folder=with=equals")

    for invalid in ("run_without_abbreviation", "=run", "abbr=", " = run "):
        with pytest.raises(ValueError):
            comparison.parse_run_spec(invalid)


def test_resolve_run_inputs_keeps_order_after_moving_baseline_first(tmp_path: Path):
    for name in ("run_a", "run_base", "run_c"):
        (tmp_path / name).mkdir()

    resolved = comparison.resolve_run_inputs(
        ["A=run_a", "Base=run_base", "C=run_c"],
        baseline_abbreviation="Base",
        run_roots=[tmp_path],
        repo_root=tmp_path,
    )

    assert [run["abbreviation"] for run in resolved] == ["Base", "A", "C"]
    assert [Path(run["run_path"]).name for run in resolved] == ["run_base", "run_a", "run_c"]
    assert resolved[0]["is_baseline"] is True
    assert resolved[0]["color"] == "#202124"
    assert [run["color"] for run in resolved[1:]] == list(comparison.V2_PALETTE[1:3])


def test_resolve_run_inputs_rejects_duplicates_and_missing_baseline(tmp_path: Path):
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()

    with pytest.raises(ValueError, match="At least two"):
        comparison.resolve_run_inputs(
            ["A=run_a"],
            baseline_abbreviation="A",
            run_roots=[tmp_path],
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="abbreviations.*duplicates"):
        comparison.resolve_run_inputs(
            ["A=run_a", "A=run_b"],
            baseline_abbreviation="A",
            run_roots=[tmp_path],
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="[Dd]uplicate.*(?:path|run)"):
        comparison.resolve_run_inputs(
            ["A=run_a", "B=run_a"],
            baseline_abbreviation="A",
            run_roots=[tmp_path],
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="[Bb]aseline"):
        comparison.resolve_run_inputs(
            ["A=run_a", "B=run_b"],
            baseline_abbreviation="missing",
            run_roots=[tmp_path],
            repo_root=tmp_path,
        )


def test_palette_keeps_baseline_black_and_supports_eight_unique_runs(tmp_path: Path):
    specs: list[str] = []
    for index in range(8):
        name = f"run_{index}"
        (tmp_path / name).mkdir()
        specs.append(f"R{index}={name}")

    resolved = comparison.resolve_run_inputs(
        specs,
        baseline_abbreviation="R3",
        run_roots=[tmp_path],
        repo_root=tmp_path,
    )

    assert resolved[0]["abbreviation"] == "R3"
    assert resolved[0]["color"] == "#202124"
    assert len({run["color"] for run in resolved}) == 8
    assert all(run["color"] != "#202124" for run in resolved[1:])

    duplicate_color_runs = [dict(run) for run in resolved[:2]]
    duplicate_color_runs[1]["color"] = duplicate_color_runs[0]["color"]
    with pytest.raises(ValueError, match="unique colors"):
        comparison._validate_palette(duplicate_color_runs)

    (tmp_path / "run_8").mkdir()
    with pytest.raises(ValueError, match="at most 8 uniquely colored runs"):
        comparison.resolve_run_inputs(
            [*specs, "R8=run_8"],
            baseline_abbreviation="R3",
            run_roots=[tmp_path],
            repo_root=tmp_path,
        )


def test_velocity_scope_has_no_ambiguous_zero_direction():
    assert comparison.VELOCITY_SCOPES == ("all", "negative", "positive")
    assert comparison.velocity_scope(-0.5) == "negative"
    assert comparison.velocity_scope(0.5) == "positive"
    with pytest.raises(ValueError):
        comparison.velocity_scope(0.0)


def test_canonical_treatment_resolution_tracks_independent_schedules_and_augmentation():
    symmetry = {
        "use_time_reversal_regularization": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
        "value_loss_coeff": 0.05,
        "warmup_iterations": 500,
        "rampup_iterations": 1000,
        "ramp_shape": "linear",
        "use_tr_policy_consistency": False,
        "use_tr_value_consistency": True,
        "tr_policy_output_space": "normalized_requested_joint_target",
        "actor_mean_bound_mode": "per_joint_feasible",
        "actor_mean_feasible_margin_fraction": 0.1,
        "tr_policy_schedule": {
            "enabled": True,
            "target_coeff": 0.2,
            "warmup_iterations": 10,
            "rampup_iterations": 20,
            "ramp_shape": "half_cosine",
        },
        "tr_value_schedule": {
            "enabled": True,
            "target_coeff": 0.075,
            "warmup_iterations": 30,
            "rampup_iterations": 40,
        },
        "tr_augmentation": {
            "enabled": True,
            "mode": "dynamics_filtered_reverse_action_supervision",
            "action_source": "learned_inverse",
            "coefficient": 0.3,
            "filter_enabled": True,
            "schedule": {
                "enabled": True,
                "target_coeff": 0.4,
                "warmup_iterations": 50,
                "rampup_iterations": 60,
            },
        },
        "tr_validity": {"mode": "command_and_tracking", "tracking_abs_tolerance": 0.25},
    }

    treatment = comparison._resolved_treatment(symmetry, "canonical")
    metadata = comparison._treatment_metadata(treatment)

    assert treatment["policy"] == {
        "enabled": False,
        "target_coeff": pytest.approx(0.2),
        "mechanism": "normalized_requested_joint_target",
        "configuration": {
            "output_space": "normalized_requested_joint_target",
            "actor_mean_bound_mode": "per_joint_feasible",
            "actor_mean_feasible_margin_fraction": pytest.approx(0.1),
        },
        "schedule": {
            "warmup_iterations": 10,
            "rampup_iterations": 20,
            "hold_iterations": 0,
            "decay_iterations": 0,
            "final_scale": pytest.approx(1.0),
            "ramp_shape": "half_cosine",
        },
    }
    assert treatment["value"]["enabled"] is True
    assert treatment["value"]["target_coeff"] == pytest.approx(0.075)
    assert treatment["value"]["schedule"]["warmup_iterations"] == 30
    assert treatment["augmentation"]["enabled"] is True
    assert treatment["augmentation"]["target_coeff"] == pytest.approx(0.4)
    assert treatment["augmentation"]["schedule"]["rampup_iterations"] == 60
    assert treatment["augmentation"]["configuration"] == {
        "mode": "dynamics_filtered_reverse_action_supervision",
        "action_source": "learned_inverse",
        "filter_enabled": True,
    }
    assert treatment["validity"] == {"mode": "command_and_tracking", "tracking_abs_tolerance": 0.25}
    assert metadata["trs_enabled"] is True
    assert metadata["policy_enabled"] is False
    assert metadata["value_enabled"] is True
    assert metadata["augmentation_enabled"] is True


def test_no_trs_baseline_rejects_augmentation_only_treatment():
    treatment = comparison._resolved_treatment(
        {
            "use_time_reversal_regularization": True,
            "use_tr_policy_consistency": False,
            "use_tr_value_consistency": False,
            "tr_augmentation": {
                "enabled": True,
                "coefficient": 0.25,
                "schedule": {"enabled": True, "target_coeff": 0.25},
            },
        },
        "augmentation-only",
    )
    metadata = comparison._treatment_metadata(treatment)

    with pytest.raises(ValueError, match="trajectory-augmentation"):
        comparison._validate_baseline_treatment({"is_baseline": True, "abbreviation": "NoTRS"}, metadata)


def test_training_metadata_falls_back_to_hashed_registered_legacy_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_path = tmp_path / "legacy_run"
    params = run_path / "params"
    git_dir = run_path / "git"
    params.mkdir(parents=True)
    git_dir.mkdir()
    (params / "agent.yaml").write_text(
        """seed: 42
num_steps_per_env: 24
max_iterations: 20000
algorithm:
  symmetry_cfg:
    use_mirror_loss: true
    mirror_loss_coeff: 0.1
    value_loss_coeff: 0.05
    warmup_iterations: 500
    rampup_iterations: 0
    ramp_shape: linear
""",
        encoding="utf-8",
    )
    (params / "env.yaml").write_text(
        """scene:
  num_envs: 512
replicated_scene:
  num_envs: 512
""",
        encoding="utf-8",
    )
    (run_path / "model_0.pt").write_bytes(b"matched initialization")
    (git_dir / "symm_rl_isaaclab.diff").write_text("legacy diff\n", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        comparison,
        "_registered_legacy_run",
        lambda *_: {
            "path": str(registry_path),
            "sha256": comparison.sha256_file(registry_path),
            "cohort_id": "legacy_cohort",
            "run_id": "legacy_run",
            "classification": "eligible_main",
            "facts": {
                "code_hashes": {
                    "archived_git_commit": "1" * 40,
                    "source_snapshot_sha256": "2" * 64,
                }
            },
        },
    )

    metadata = comparison._load_training_metadata(
        {"resolved_path": str(run_path), "abbreviation": "TRS", "is_baseline": False}, tmp_path
    )

    assert metadata["training_metadata_source"] == comparison.LEGACY_CONFIG_METADATA_SOURCE
    assert metadata["seed"] == 42
    assert metadata["num_envs"] == 512
    assert metadata["mirror_coeff"] == pytest.approx(0.1)
    assert metadata["value_coeff"] == pytest.approx(0.05)
    assert metadata["repo_commit"] == "1" * 40
    assert [Path(entry["path"]).name for entry in metadata["training_metadata_files"]] == [
        "agent.yaml",
        "env.yaml",
        "model_0.pt",
        "symm_rl_isaaclab.diff",
        "registry.json",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in metadata["training_metadata_files"])


def test_legacy_training_metadata_accepts_only_a_bound_checksum_when_model_zero_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_path = tmp_path / "legacy_run"
    params = run_path / "params"
    git_dir = run_path / "git"
    provenance_dir = run_path / "provenance"
    params.mkdir(parents=True)
    git_dir.mkdir()
    provenance_dir.mkdir()
    (params / "agent.yaml").write_text(
        """seed: 42
num_steps_per_env: 24
max_iterations: 20000
algorithm:
  symmetry_cfg:
    use_mirror_loss: true
    mirror_loss_coeff: 0.1
    value_loss_coeff: 0.05
    warmup_iterations: 500
    rampup_iterations: 0
    ramp_shape: linear
""",
        encoding="utf-8",
    )
    (params / "env.yaml").write_text(
        """scene:
  num_envs: 512
replicated_scene:
  num_envs: 512
""",
        encoding="utf-8",
    )
    (git_dir / "symm_rl_isaaclab.diff").write_text("legacy diff\n", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    registry = {
        "path": str(registry_path),
        "sha256": comparison.sha256_file(registry_path),
        "cohort_id": "x1_gait_v2_development_sweep",
        "run_id": "x1_trs_m0p1_v0p05_hard",
        "classification": "eligible_main",
        "facts": {
            "code_hashes": {
                "archived_git_commit": "1" * 40,
                "source_snapshot_sha256": "2" * 64,
            }
        },
    }
    monkeypatch.setattr(comparison, "_registered_legacy_run", lambda *_: registry)
    checkpoint_bytes = b"archived checkpoint"
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    record = {
        "schema_version": 1,
        "record_type": comparison.LEGACY_INITIALIZATION_RECORD_TYPE,
        "availability": "checksum_only",
        "bytes_published": False,
        "checkpoint": {"filename": "model_0.pt", "iteration": 0, "sha256": checkpoint_sha256},
        "comparison_run_id": "run_01",
        "registry": {
            "cohort_id": registry["cohort_id"],
            "run_id": registry["run_id"],
        },
        "note": (
            "The model_0.pt bytes are intentionally excluded by the terminal-checkpoint-only archive policy; "
            "this record preserves the SHA-256 observed before archival."
        ),
    }
    record["record_sha256"] = comparison.canonical_sha256(record)
    record_path = provenance_dir / "legacy_initialization.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    metadata = comparison._load_training_metadata(
        {
            "resolved_path": str(run_path),
            "run_id": "run_01",
            "abbreviation": "TRS",
            "is_baseline": False,
        },
        tmp_path,
        manifest_bound=True,
    )

    assert metadata["training_metadata_source"] == comparison.LEGACY_CHECKSUM_ONLY_METADATA_SOURCE
    assert metadata["iteration_zero_checkpoint_sha256"] == checkpoint_sha256
    assert metadata["iteration_zero_checkpoint_availability"] == "checksum_only"
    assert metadata["iteration_zero_checkpoint_bytes_reverified"] is False
    assert metadata["legacy_initialization_checksum_record"] == {
        "path": str(record_path),
        "sha256": comparison.sha256_file(record_path),
        "record_sha256": record["record_sha256"],
    }
    assert [Path(entry["path"]).name for entry in metadata["training_metadata_files"]] == [
        "agent.yaml",
        "env.yaml",
        "symm_rl_isaaclab.diff",
        "registry.json",
        "legacy_initialization.json",
    ]
    warnings = comparison._comparability_warnings({"run_01": {**metadata, "abbreviation": "TRS"}})
    assert any("checkpoint bytes were not reverified" in warning for warning in warnings)

    (run_path / "model_0.pt").write_bytes(checkpoint_bytes)
    manifest_metadata = comparison._load_training_metadata(
        {
            "resolved_path": str(run_path),
            "run_id": "run_01",
            "abbreviation": "TRS",
            "is_baseline": False,
            "expected_training_metadata_source": comparison.LEGACY_CHECKSUM_ONLY_METADATA_SOURCE,
        },
        tmp_path,
        manifest_bound=True,
    )
    assert manifest_metadata["training_metadata_source"] == comparison.LEGACY_CHECKSUM_ONLY_METADATA_SOURCE
    assert all(Path(entry["path"]).name != "model_0.pt" for entry in manifest_metadata["training_metadata_files"])

    direct_metadata = comparison._load_training_metadata(
        {
            "resolved_path": str(run_path),
            "run_id": "run_01",
            "abbreviation": "TRS",
            "is_baseline": False,
        },
        tmp_path,
    )
    assert direct_metadata["training_metadata_source"] == comparison.LEGACY_CONFIG_METADATA_SOURCE
    assert any(Path(entry["path"]).name == "model_0.pt" for entry in direct_metadata["training_metadata_files"])

    (run_path / "model_0.pt").write_bytes(b"not the archived checkpoint")
    with pytest.raises(ValueError, match="differs from its checksum record"):
        comparison._load_training_metadata(
            {
                "resolved_path": str(run_path),
                "run_id": "run_01",
                "abbreviation": "TRS",
                "is_baseline": False,
            },
            tmp_path,
            manifest_bound=True,
        )


def test_legacy_checksum_record_must_match_the_manifest_run_and_registry(tmp_path: Path):
    record_path = tmp_path / "legacy_initialization.json"
    registry = {"cohort_id": "cohort", "run_id": "registry_run"}
    record = {
        "schema_version": 1,
        "record_type": comparison.LEGACY_INITIALIZATION_RECORD_TYPE,
        "availability": "checksum_only",
        "bytes_published": False,
        "checkpoint": {"filename": "model_0.pt", "iteration": 0, "sha256": "a" * 64},
        "comparison_run_id": "run_00",
        "registry": dict(registry),
        "note": "Checkpoint bytes omitted by terminal-checkpoint-only policy.",
    }
    record["record_sha256"] = comparison.canonical_sha256(record)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="comparison_run_id differs"):
        comparison._load_legacy_initialization_checksum(record_path, registry, "TRS", "run_01")

    record["comparison_run_id"] = "run_01"
    record["registry"]["run_id"] = "other_run"
    record["record_sha256"] = comparison.canonical_sha256(comparison._record_payload(record))
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong registry entry"):
        comparison._load_legacy_initialization_checksum(record_path, registry, "TRS", "run_01")


def test_legacy_training_metadata_adds_an_explicit_comparability_warning():
    warnings = comparison._comparability_warnings(
        {
            "run_00": {
                "abbreviation": "No TRS",
                "training_metadata_source": comparison.LEGACY_CONFIG_METADATA_SOURCE,
                "iteration_zero_checkpoint_sha256": "a" * 64,
                "seed": 42,
                "num_envs": 512,
                "num_steps_per_env": 24,
                "max_iterations": 20000,
                "repo_commit": "1" * 40,
                "dirty_tree_diff_sha256": "2" * 64,
            }
        }
    )

    assert len(warnings) == 1
    assert "lack pre-first-rollout initialization provenance" in warnings[0]
    assert "No TRS" in warnings[0]


def test_leg_metric_registry_includes_raw_and_normalized_torque_squared():
    assert _metric_ids(comparison.LEG_METRICS) == (
        "torque_squared",
        "normalized_torque_squared",
        "absolute_work",
        "vertical_grf_impulse",
    )


def test_equal_family_aggregation_does_not_pool_unequal_family_counts():
    runs = _synthetic_runs(1)
    velocity_rows, leg_rows, gait_rows = comparison.aggregate_summaries(_synthetic_cells(runs), runs, FAMILIES)

    # Equal family mean: (1 + 2 + 30 + 40) / 4 = 18.25. A pooled
    # 6/6/24/24-cell mean would be 28.3 and must never appear here.
    negative_velocity = _one(
        velocity_rows,
        abbreviation="R0",
        velocity_scope="negative",
        aggregation="overall",
        family="all_family_balanced",
        metric="velocity_vx_rmse_mps",
    )
    assert negative_velocity["value"] == pytest.approx(18.25)
    assert negative_velocity["cells"] == 30
    assert negative_velocity["families"] == 4

    all_velocity = _one(
        velocity_rows,
        abbreviation="R0",
        velocity_scope="all",
        aggregation="overall",
        family="all_family_balanced",
        metric="velocity_vx_rmse_mps",
    )
    assert all_velocity["value"] == pytest.approx(68.25)

    negative_leg = _one(
        leg_rows,
        abbreviation="R0",
        velocity_scope="negative",
        aggregation="overall",
        family="all_family_balanced",
        metric="normalized_torque_squared",
    )
    negative_gait = _one(
        gait_rows,
        abbreviation="R0",
        velocity_scope="negative",
        aggregation="overall",
        family="all_family_balanced",
        metric="gait_agreement_boundary_excluded_percent",
    )
    assert negative_leg["value"] == pytest.approx(18.25)
    assert negative_gait["value"] == pytest.approx(18.25)


def test_five_run_long_form_row_counts_and_baseline_deltas():
    runs = _synthetic_runs(5)
    velocity_rows, leg_rows, gait_rows = comparison.aggregate_summaries(_synthetic_cells(runs), runs, FAMILIES)

    # run * scope * (overall + four families) * metric
    assert len(velocity_rows) == 5 * 3 * 5 * 2 == 150
    assert len(leg_rows) == 5 * 3 * 5 * 4 == 300
    assert len(gait_rows) == 5 * 3 * 5 == 75

    for rows in (velocity_rows, leg_rows, gait_rows):
        assert all(row["delta_from_baseline"] == pytest.approx(0.0) for row in rows if row["is_baseline"])

    treatment = _one(
        velocity_rows,
        abbreviation="R4",
        velocity_scope="negative",
        aggregation="overall",
        family="all_family_balanced",
        metric="velocity_vx_rmse_mps",
    )
    assert treatment["delta_from_baseline"] == pytest.approx(4.0)


def test_figure_registry_declares_sixteen_stable_unique_ids():
    figure_ids = _figure_ids(comparison.FIGURE_REGISTRY)

    assert len(figure_ids) == 16
    assert len(set(figure_ids)) == 16
    assert set(figure_ids) == EXPECTED_FIGURE_IDS
    assert comparison.FIGURE_REGISTRY["fig03"]["children"] == ("fig03_01", "fig03_02", "fig03_03")
    assert comparison.FIGURE_REGISTRY["fig05"]["children"] == ("fig05_01", "fig05_02", "fig05_03")
    assert comparison.FIGURE_REGISTRY["fig07"]["children"] == ("fig07_01", "fig07_02", "fig07_03")
    assert all(entry["source_tables"] for entry in comparison.FIGURE_REGISTRY.values())


def test_comparison_module_contains_engine_and_plotting_without_removed_module_imports():
    source_path = SCRIPT_DIR / "comparison.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported_modules.update(
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "gait_closure_comparison" not in imported_modules
    assert "gait_closure_plots" not in imported_modules
    assert comparison.run_comparison.__module__ == "comparison"
    assert comparison.generate_figures.__module__ == "comparison"
    assert not (SCRIPT_DIR / "gait_closure_comparison.py").exists()
    assert not (SCRIPT_DIR / "gait_closure_plots.py").exists()


def test_plot_bounds_scale_small_velocity_metrics_and_leg_children_consistently():
    velocity_upper = comparison._positive_upper([0.025, 0.087])

    assert 0.087 < velocity_upper < 0.2
    assert comparison._positive_upper([4.0, 51.0], minimum=10.0) == pytest.approx(60.0)
    assert comparison._legend_column_order(6, 5) == [0, 5, 1, 2, 3, 4]


def test_public_figure_generator_writes_openable_rgb_images_and_exact_vertical_composites(tmp_path: Path):
    runs = _synthetic_runs(2)
    velocity_rows, leg_rows, gait_rows = comparison.aggregate_summaries(_synthetic_cells(runs), runs, FAMILIES)
    learning_points = [
        {
            "run_id": run["run_id"],
            "transitions_millions": float(point),
            "elapsed_hours": 0.5 * point,
            "smoothed_reward": 10.0 * point + run_index,
        }
        for run_index, run in enumerate(runs)
        for point in (1, 2)
    ]
    artifacts = comparison.generate_figures(
        tmp_path,
        {
            "display_name": "Synthetic Go2",
            "reward_target": 35.0,
            "training": {"smoothing_window_iterations": 200},
            "runs": runs,
        },
        learning_points,
        velocity_rows,
        leg_rows,
        gait_rows,
    )

    assert [artifact["figure_id"] for artifact in artifacts] == list(comparison.FIGURE_REGISTRY)
    sizes: dict[str, tuple[int, int]] = {}
    for artifact in artifacts:
        png_path = tmp_path / artifact["png"]
        svg_path = tmp_path / artifact["svg"]
        with Image.open(png_path) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.width > 0 and image.height > 0
            sizes[artifact["figure_id"]] = image.size
        assert svg_path.read_bytes().lstrip().startswith(b"<?xml")

    for parent in ("fig03", "fig05", "fig07"):
        children = comparison.FIGURE_REGISTRY[parent]["children"]
        assert sizes[parent] == (
            max(sizes[child][0] for child in children),
            sum(sizes[child][1] for child in children),
        )
    figure_manifest = comparison._figure_manifest(artifacts, tmp_path)
    assert all(entry["source_tables"] for entry in figure_manifest["figures"])


def test_load_manifest_round_trips_configuration(tmp_path: Path):
    for name in ("run_base", "run_trs"):
        (tmp_path / name).mkdir()
    manifest = {
        "schema_version": 1,
        "method_version": comparison.METHOD_VERSION,
        "runs": [
            {
                "run_id": "baseline",
                "abbreviation": "Base",
                "folder": "run_base",
                "run_name": "run_base",
                "run_path": "stale/location/run_base",
                "color": "#202124",
                "is_baseline": True,
                "training_metadata_source": comparison.LEGACY_CHECKSUM_ONLY_METADATA_SOURCE,
            },
            {
                "run_id": "treatment",
                "abbreviation": "TRS",
                "folder": "run_trs",
                "run_name": "run_trs",
                "run_path": "stale/location/run_trs",
                "color": comparison.V2_PALETTE[1],
                "is_baseline": False,
                "training_metadata_source": comparison.LEGACY_CHECKSUM_ONLY_METADATA_SOURCE,
            },
        ],
        "run_roots": [str(tmp_path)],
        "evaluation_subdir": "evaluations/leg_usage_grid_full_v3",
        "training": {
            "smoothing_window_iterations": 123,
            "sample_stride_iterations": 17,
            "threshold_reward": 35.0,
        },
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = comparison.load_manifest(path, repo_root=tmp_path)

    assert loaded["runs"][0]["is_baseline"] is True
    assert [run["abbreviation"] for run in loaded["runs"]] == ["Base", "TRS"]
    assert all(
        run["expected_training_metadata_source"] == comparison.LEGACY_CHECKSUM_ONLY_METADATA_SOURCE
        for run in loaded["runs"]
    )
    assert loaded["evaluation_subdir"] == "evaluations/leg_usage_grid_full_v3"
    assert loaded["smoothing_window_iterations"] == 123
    assert loaded["sample_stride_iterations"] == 17


def test_manifest_reproduction_binds_its_saved_protocol_and_treatments():
    protocol = {
        "robot": "go2",
        "method_version": comparison.EVALUATION_METHOD_VERSION,
        "velocities_mps": [-0.5, 0.5],
    }
    protocol_sha256 = comparison.canonical_sha256(protocol)
    baseline_treatment = comparison._resolved_treatment(
        {
            "use_time_reversal_regularization": False,
            "use_tr_policy_consistency": False,
            "use_tr_value_consistency": False,
            "tr_augmentation": {"enabled": False},
        },
        "baseline",
    )
    treatment = comparison._resolved_treatment(
        {
            "use_time_reversal_regularization": True,
            "use_tr_policy_consistency": True,
            "use_tr_value_consistency": True,
            "tr_policy_schedule": {"enabled": True, "target_coeff": 0.1},
            "tr_value_schedule": {"enabled": True, "target_coeff": 0.05},
            "tr_augmentation": {"enabled": False},
        },
        "treatment",
    )
    manifest = {
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": protocol_sha256,
        "runs": [
            {"run_id": "run_00", "treatment": baseline_treatment},
            {"run_id": "run_01", "treatment": treatment},
        ],
    }
    metadata = {
        "run_00": {"treatment": baseline_treatment},
        "run_01": {"treatment": treatment},
    }

    comparison._validate_manifest_protocol(manifest, protocol, protocol_sha256)
    comparison._validate_manifest_treatments(manifest, metadata)

    changed_protocol = {**protocol, "velocities_mps": [-1.0, 1.0]}
    with pytest.raises(ValueError, match="do not match the protocol"):
        comparison._validate_manifest_protocol(
            manifest, changed_protocol, comparison.canonical_sha256(changed_protocol)
        )
    changed_metadata = json.loads(json.dumps(metadata))
    changed_metadata["run_01"]["treatment"]["policy"]["target_coeff"] = 0.2
    with pytest.raises(ValueError, match="treatment differs"):
        comparison._validate_manifest_treatments(manifest, changed_metadata)


def test_manifest_reproduction_refuses_any_bound_input_hash_drift():
    def input_record(run_id: str, fill: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "checkpoint_sha256": fill * 64,
            "event_sha256": chr(ord(fill) + 1) * 64,
            "evaluation_analysis_record_sha256": chr(ord(fill) + 2) * 64,
            "initialization_sha256": chr(ord(fill) + 3) * 64,
            "training_metadata": {
                "source": comparison.INITIALIZATION_METADATA_SOURCE,
                "files": [{"path": "provenance/initialization.json", "sha256": chr(ord(fill) + 4) * 64}],
            },
            "evaluation_files": {
                "study": {"path": "study.json", "sha256": chr(ord(fill) + 5) * 64},
                "metrics": {"path": "metrics.json", "sha256": chr(ord(fill) + 6) * 64},
            },
        }

    saved_inputs = [input_record("run_00", "1"), input_record("run_01", "2")]
    saved_provenance = {"inputs": saved_inputs}
    current_inputs = json.loads(json.dumps(saved_inputs))

    comparison._validate_manifest_input_hashes(saved_provenance, current_inputs)

    current_inputs[1]["event_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="refusing to overwrite.*run_01"):
        comparison._validate_manifest_input_hashes(saved_provenance, current_inputs)
    invalid_inputs = json.loads(json.dumps(saved_inputs))
    invalid_inputs[0]["checkpoint_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        comparison._validate_manifest_input_hashes(saved_provenance, invalid_inputs)


def test_manifest_must_be_bound_by_its_sibling_analysis_provenance(tmp_path: Path):
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "method_version": comparison.METHOD_VERSION,
        "study_sha256": comparison.sha256_file(manifest_path),
        "inputs": [{"run_id": "run_00"}],
    }
    provenance["record_sha256"] = comparison.canonical_sha256(provenance)
    provenance_path = tmp_path / "analysis_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    assert comparison._load_manifest_provenance_binding(manifest_path) == provenance

    manifest_path.write_text('{"schema_version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not bound by its sibling"):
        comparison._load_manifest_provenance_binding(manifest_path)


def test_cell_metric_metadata_must_match_the_study_plan():
    planned = {
        "gait_index": 2,
        "gait_name": "half_bound_a",
        "family": "half_bound",
        "velocity_mps": -1.0,
        "seed": 4242,
        "relative_output_dir": "cells/a",
    }
    source = {key: str(value) for key, value in planned.items()}
    comparison._validate_cell_plan_row(source, planned, "test/cell")

    source["family"] = "gallop"
    with pytest.raises(ValueError, match="family.*study plan"):
        comparison._validate_cell_plan_row(source, planned, "test/cell")


def test_load_manifest_rejects_direct_run_contract_drift(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        comparison.load_manifest(path, repo_root=tmp_path)


def test_cli_manifest_mode_defaults_output_to_manifest_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = tmp_path / "analysis" / "study.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run_comparison(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"run_count": 2, "output_dir": str(output_dir)}

    monkeypatch.setattr(comparison, "run_comparison", fake_run_comparison)

    assert comparison.main(["--manifest", str(manifest_path), "--repo_root", str(tmp_path)]) == 0
    assert captured["output_dir"] == manifest_path.parent.resolve()
    assert captured["manifest_path"] == manifest_path.resolve()
    assert captured["run_specs"] == []
    assert captured["baseline_abbreviation"] is None
    assert captured["run_roots"] is None


def test_cli_direct_mode_forwards_ordered_runs_baseline_and_default_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "analysis"
    captured: dict[str, Any] = {}

    def fake_run_comparison(destination: Path, **kwargs: Any) -> dict[str, Any]:
        captured["output_dir"] = destination
        captured.update(kwargs)
        return {"run_count": 2, "output_dir": str(destination)}

    monkeypatch.setattr(comparison, "run_comparison", fake_run_comparison)

    assert (
        comparison.main(
            [
                "--run",
                "NoTRS=run_base",
                "--run",
                "TRS=run_trs",
                "--baseline",
                "NoTRS",
                "--output_dir",
                str(output_dir),
                "--repo_root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert captured["output_dir"] == output_dir.resolve()
    assert captured["run_specs"] == ["NoTRS=run_base", "TRS=run_trs"]
    assert captured["baseline_abbreviation"] == "NoTRS"
    assert captured["run_roots"] == comparison.default_run_roots(tmp_path.resolve())
    assert captured["manifest_path"] is None


def test_comparison_launcher_names():
    assert "comparison.py" in comparison._reproduce_script()
    assert "compare_gait_closure_runs.py" not in comparison._reproduce_script()
    assert "comparison.py" in (SCRIPT_DIR / "comparison.sh").read_text(encoding="utf-8")
    assert "comparison.py" in (SCRIPT_DIR / "comparison.ps1").read_text(encoding="utf-8")


def test_comparison_launcher_provenance_binds_all_canonical_entrypoints():
    repo_root = SCRIPT_DIR.parents[1]
    records = comparison._comparison_launcher_provenance(repo_root)

    assert set(records) == {"python", "bash", "powershell"}
    for launcher, filename in comparison.COMPARISON_LAUNCHER_FILES.items():
        path = SCRIPT_DIR / filename
        assert records[launcher] == {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": comparison.sha256_file(path),
        }


def test_training_scalars_use_shared_parser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    event_path = tmp_path / "events.out.tfevents.synthetic"
    calls: list[tuple[Path, set[str] | None]] = []

    def fake_read_scalars(path: Path, selected_tags: set[str] | None = None):
        calls.append((path, selected_tags))
        return {"Train/mean_reward": [(1, 10.0, 2.5)]}

    monkeypatch.setattr(tensorboard_scalars, "read_scalars", fake_read_scalars)

    assert comparison._read_training_scalars(event_path) == [(1, 10.0, 2.5)]
    assert calls == [(event_path, {"Train/mean_reward"})]


def test_analysis_provenance_binds_tensorboard_parser(tmp_path: Path):
    (tmp_path / "study.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "style.json").write_text("{}\n", encoding="utf-8")
    repo_root = SCRIPT_DIR.parents[1]

    provenance = comparison._analysis_provenance(
        tmp_path,
        repo_root,
        {"evaluation_protocol_sha256": "protocol"},
        [],
        {},
        {},
        [],
        None,
        tmp_path / "unused-source-manifest.json",
    )

    parser_path = SCRIPT_DIR / "_tensorboard_scalars.py"
    comparison_path = SCRIPT_DIR / "comparison.py"
    comparison_record = {
        "path": comparison_path.relative_to(repo_root).as_posix(),
        "sha256": comparison.sha256_file(comparison_path),
    }
    assert provenance["comparison_script"] == comparison_record
    assert provenance["plot_script"] == comparison_record
    assert provenance["tensorboard_scalar_parser"] == {
        "path": parser_path.relative_to(repo_root).as_posix(),
        "sha256": comparison.sha256_file(parser_path),
    }
