# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the data-only leg-usage grid runner helpers."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_grid_helpers() -> SimpleNamespace:
    """Load pure helpers without importing the simulator-launching script."""
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play_rsl_rl.py"
    source = module_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(module_path))
    helper_names = {
        "_utc_timestamp",
        "_canonical_leg_usage_method_version",
        "_validate_leg_usage_gait_library",
        "_validate_immutable_leg_usage_protocol",
        "_load_leg_usage_plan",
        "_validate_leg_usage_task",
        "_sha256_file",
        "_validate_leg_usage_checkpoint",
        "_leg_usage_step_counts",
        "_ordered_leg_usage_gaits",
        "_configured_gait_index_by_id",
        "_configure_leg_usage_env",
        "_resolve_cell_output_dir",
        "_completed_cell_archive_matches",
        "_validated_completed_cell",
        "_leg_usage_recording_lock",
    }
    helper_nodes = [node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    namespace = {
        "Any": Any,
        "ManagerBasedRLEnvCfg": Any,
        "Path": Path,
        "contextlib": contextlib,
        "datetime": datetime,
        "timezone": timezone,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "np": np,
        "os": os,
        "zipfile": zipfile,
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION": "time_reversal_closed_v2",
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS": (
            (0.0, 0.5, 0.5, 0.0),
            (0.0, 0.0, 0.5, 0.5),
            (0.13, -0.13, 0.5, 0.5),
            (-0.13, 0.13, 0.5, 0.5),
            (0.0, 0.0, 0.63, 0.37),
            (0.0, 0.0, 0.37, 0.63),
            (-0.13, 0.13, 0.63, 0.37),
            (0.13, -0.13, 0.63, 0.37),
            (0.13, -0.13, 0.37, 0.63),
            (-0.13, 0.13, 0.37, 0.63),
        ),
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS": (4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES": (
            "trot",
            "bound",
            "half_bound_front_a",
            "half_bound_front_b",
            "half_bound_hind_a",
            "half_bound_hind_b",
            "gallop_a",
            "gallop_b",
            "gallop_c",
            "gallop_d",
        ),
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES": (
            "trot",
            "bound",
            "half_bound",
            "half_bound",
            "half_bound",
            "half_bound",
            "gallop",
            "gallop",
            "gallop",
            "gallop",
        ),
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS": (0, 1, 3, 2, 5, 4, 8, 9, 6, 7),
        "_LEG_USAGE_PROTOCOL_VERSION": "leg_usage_grid_v1",
        "_LEG_USAGE_FULL_PROTOCOL_VERSION": "leg_usage_grid_full_v3",
        "_LEG_USAGE_LIGHT_PROTOCOL_VERSION": "leg_usage_grid_light_v2",
        "_LEG_USAGE_PROTOCOL_VERSIONS": {
            "leg_usage_grid_v1",
            "leg_usage_grid_full_v3",
            "leg_usage_grid_light_v2",
        },
    }
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(module_path), "exec"), namespace)
    exports = {
        *helper_names,
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION",
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS",
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS",
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES",
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES",
        "SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS",
    }
    return SimpleNamespace(**{name: namespace[name] for name in exports})


def _write_sparse_plan(output_root: Path) -> Path:
    output_root.mkdir(parents=True)
    plan = {
        "schema_version": 1,
        "method_version": "leg_usage_grid_v1",
        "gait_library_version": "time_reversal_closed_v2",
        "output_root": str(output_root),
        "settle_s": 5.0,
        "measure_s": 10.0,
        "gaits": [
            {
                "index": 6,
                "name": "gallop_a",
                "family": "gallop",
                "phases": [-0.13, 0.13, 0.63, 0.37],
                "weight": 1,
                "time_reversal_partner": 8,
            },
            {
                "index": 9,
                "name": "gallop_d",
                "family": "gallop",
                "phases": [-0.13, 0.13, 0.37, 0.63],
                "weight": 1,
                "time_reversal_partner": 7,
            },
        ],
        "cells": [
            {
                "id": "g06__vx_neg_0p5__seed_0042",
                "gait_index": 6,
                "velocity_mps": -0.5,
                "relative_output_dir": "cells/gait_06/vx_neg_0p5/seed_0042",
            }
        ],
    }
    plan_path = output_root / "study.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def _write_current_full_plan(output_root: Path, helpers: SimpleNamespace) -> Path:
    """Write the exact current 10-row by six-velocity publication inventory."""
    output_root.mkdir(parents=True)
    gaits = [
        {
            "index": index,
            "name": name,
            "family": family,
            "phases": list(phases),
            "weight": weight,
            "time_reversal_partner": partner,
        }
        for index, (phases, weight, name, family, partner) in enumerate(
            zip(
                helpers.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS,
                helpers.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS,
                helpers.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES,
                helpers.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES,
                helpers.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS,
                strict=True,
            )
        )
    ]
    velocities = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
    cells = [
        {
            "id": f"gait_{gait['index']:02d}__vx_{velocity:+g}",
            "gait_index": gait["index"],
            "gait_name": gait["name"],
            "velocity_mps": velocity,
            "relative_output_dir": f"cells/gait_{gait['index']:02d}/vx_{velocity:+g}",
        }
        for gait in gaits
        for velocity in velocities
    ]
    plan = {
        "schema_version": 1,
        "method_version": "leg_usage_grid_full_v3",
        "protocol": "full",
        "gait_library_version": helpers.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION,
        "output_root": str(output_root),
        "settle_s": 5.0,
        "measure_s": 10.0,
        "velocities_mps": list(velocities),
        "gaits": gaits,
        "cells": cells,
    }
    plan_path = output_root / "study.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def test_plan_loader_accepts_sparse_stable_gait_ids(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_sparse_plan(tmp_path / "leg_usage_grid")

    plan, resolved_path, plan_sha256 = helpers._load_leg_usage_plan(str(plan_path))

    assert resolved_path == plan_path.resolve()
    assert [gait["index"] for gait in plan["gaits"]] == [6, 9]
    assert plan_sha256 == hashlib.sha256(plan_path.read_bytes()).hexdigest()


def test_plan_loader_accepts_exact_current_full_inventory(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_current_full_plan(tmp_path / "leg_usage_grid", helpers)

    plan, _, _ = helpers._load_leg_usage_plan(str(plan_path))

    assert plan["method_version"] == "leg_usage_grid_full_v3"
    assert len(plan["gaits"]) == 10
    assert len(plan["cells"]) == 60


def test_plan_loader_rejects_conflicting_protocol_version_alias(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_current_full_plan(tmp_path / "leg_usage_grid", helpers)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["protocol_version"] = "leg_usage_grid_light_v2"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="method_version/protocol_version identity mismatch"):
        helpers._load_leg_usage_plan(str(plan_path))


def test_plan_loader_retains_protocol_version_only_alias(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_current_full_plan(tmp_path / "leg_usage_grid", helpers)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["protocol_version"] = plan.pop("method_version")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    loaded, _, _ = helpers._load_leg_usage_plan(str(plan_path))

    assert helpers._canonical_leg_usage_method_version(loaded) == "leg_usage_grid_full_v3"


@pytest.mark.parametrize("stale_method", ("leg_usage_grid_full_v2", "leg_usage_grid_light_v1"))
def test_plan_loader_rejects_superseded_publication_protocols(tmp_path, stale_method):
    helpers = _load_grid_helpers()
    plan_path = _write_sparse_plan(tmp_path / "leg_usage_grid")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["method_version"] = stale_method
    plan["protocol"] = "light" if "light" in stale_method else "full"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported leg-usage method_version"):
        helpers._load_leg_usage_plan(str(plan_path))


def test_current_full_protocol_rejects_sparse_cell_inventory(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_sparse_plan(tmp_path / "leg_usage_grid")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["method_version"] = "leg_usage_grid_full_v3"
    plan["protocol"] = "full"
    plan["velocities_mps"] = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="all ten canonical gait rows"):
        helpers._load_leg_usage_plan(str(plan_path))


def test_task_and_canonical_gait_library_validation_reject_stale_plans(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_sparse_plan(tmp_path / "leg_usage_grid")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["task"] = "Isaac-Velocity-Flat-Go2-Symm-Play-v0"

    helpers._validate_leg_usage_task(plan, plan["task"])
    with pytest.raises(ValueError, match="does not match"):
        helpers._validate_leg_usage_task(plan, "Isaac-Velocity-Flat-X1-Symm-Play-v0")
    helpers._validate_leg_usage_gait_library(plan)
    plan["gaits"][0]["phases"][0] = 0.25
    with pytest.raises(ValueError, match="canonical training row"):
        helpers._validate_leg_usage_gait_library(plan)
    plan["gaits"][0]["phases"][0] = -0.13
    plan["gait_library_version"] = "stale_v1"
    with pytest.raises(ValueError, match="does not match current training code"):
        helpers._validate_leg_usage_gait_library(plan)


def test_sparse_gait_ids_map_to_sorted_installed_rows():
    helpers = _load_grid_helpers()
    plan = {"gaits": [{"index": 9}, {"index": 0}, {"index": 6}]}

    assert [gait["index"] for gait in helpers._ordered_leg_usage_gaits(plan)] == [0, 6, 9]
    assert helpers._configured_gait_index_by_id(plan) == {0: 0, 6: 1, 9: 2}


def test_fixed_grid_configuration_disables_training_gait_sampling_profile():
    helpers = _load_grid_helpers()
    command_cfg = SimpleNamespace(
        init_foot_thetas=(),
        init_foot_theta_weights=(),
        gait_library_version="stale",
        gait_sampling_profile="trclosed_v2_equal_family",
        gait_sequence_enabled=True,
        add_noise_period=True,
        add_noise_theta=True,
        resampling_time_gait=1.0,
        heading_command=True,
        rel_heading_envs=1.0,
        rel_standing_envs=1.0,
        resampling_time_range=(1.0, 2.0),
        ranges=SimpleNamespace(lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-1.0, 1.0)),
    )
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=16),
        commands=SimpleNamespace(base_velocity=command_cfg),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        episode_length_s=0.0,
    )
    plan = {
        "gait_library_version": "time_reversal_closed_v2",
        "settle_s": 5.0,
        "measure_s": 10.0,
        "nominal_profile": False,
        "gaits": [
            {"index": 9, "phases": (-0.13, 0.13, 0.37, 0.63), "weight": 1.0},
            {"index": 6, "phases": (-0.13, 0.13, 0.63, 0.37), "weight": 2.0},
        ],
    }

    helpers._configure_leg_usage_env(env_cfg, plan)

    assert env_cfg.scene.num_envs == 1
    assert command_cfg.init_foot_thetas == (
        (-0.13, 0.13, 0.63, 0.37),
        (-0.13, 0.13, 0.37, 0.63),
    )
    assert command_cfg.init_foot_theta_weights == (2.0, 1.0)
    assert command_cfg.gait_sampling_profile is None
    assert command_cfg.gait_sequence_enabled is False
    assert command_cfg.gait_library_version == "time_reversal_closed_v2"


def test_checkpoint_validation_rejects_path_and_content_mismatches(tmp_path):
    helpers = _load_grid_helpers()
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    plan = {"checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}

    helpers._validate_leg_usage_checkpoint(plan, str(checkpoint))
    with pytest.raises(ValueError, match="does not match"):
        helpers._validate_leg_usage_checkpoint(plan, str(tmp_path / "different.pt"))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        helpers._validate_leg_usage_checkpoint(plan, str(checkpoint))


def test_plan_loader_rejects_unknown_gait_and_output_root_mismatch(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_sparse_plan(tmp_path / "leg_usage_grid")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["cells"][0]["gait_index"] = 7
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown gait index"):
        helpers._load_leg_usage_plan(str(plan_path))

    plan["cells"][0]["gait_index"] = 6
    plan["output_root"] = str(tmp_path / "somewhere_else")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="directory containing the plan"):
        helpers._load_leg_usage_plan(str(plan_path))


def test_step_counts_and_cell_path_validation(tmp_path):
    helpers = _load_grid_helpers()
    plan = {"settle_s": 5.0, "measure_s": 10.0}

    assert helpers._leg_usage_step_counts(plan, 0.02) == (250, 500, 750)
    with pytest.raises(ValueError, match="total_steps =="):
        helpers._leg_usage_step_counts({**plan, "total_steps": 749}, 0.02)
    with pytest.raises(ValueError, match="escapes the study root"):
        helpers._resolve_cell_output_dir(tmp_path.resolve(), "../outside")
    with pytest.raises(ValueError, match="must be relative"):
        helpers._resolve_cell_output_dir(tmp_path.resolve(), str((tmp_path / "absolute").resolve()))


@pytest.mark.parametrize(
    (
        "outcome",
        "recorded_steps",
        "status_expected_steps",
        "episode_done",
        "termination_terms",
        "torque_width",
        "expected",
    ),
    [
        ("completed", 10, 10, [False] * 10, [], 12, True),
        ("completed", 9, 10, [False] * 9, [], 12, False),
        ("completed", 9, 9, [False] * 9, [], 12, False),
        ("completed", 10, 10, [False] * 9 + [True], [], 12, False),
        ("completed", 10, 10, [False] * 10, [], 11, False),
        ("terminated", 4, 10, [False, False, False, True], ["base_contact"], 12, True),
        ("terminated", 4, 10, [False, False, False, False], ["base_contact"], 12, False),
        ("terminated", 4, 10, [False, False, False, True], [], 12, False),
    ],
)
def test_completed_cell_validation_checks_outcome_semantics(
    tmp_path,
    outcome,
    recorded_steps,
    status_expected_steps,
    episode_done,
    termination_terms,
    torque_width,
    expected,
):
    helpers = _load_grid_helpers()
    cell = {
        "id": "cell_01",
        "gait_index": 2,
        "velocity_mps": 0.5,
        "phases": [0.13, -0.13, 0.5, 0.5],
        "step_dt": 0.02,
        "total_steps": 10,
    }
    plan_sha256 = "1" * 64
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "outcome": outcome,
                "cell_id": cell["id"],
                "plan_sha256": plan_sha256,
                "recorded_steps": recorded_steps,
                "expected_steps": status_expected_steps,
                "termination_terms": termination_terms,
            }
        ),
        encoding="utf-8",
    )
    (cell_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cell_id": cell["id"],
                "gait_index": cell["gait_index"],
                "gait_phases": cell["phases"],
                "velocity_mps": cell["velocity_mps"],
                "step_dt": cell["step_dt"],
                "expected_steps": status_expected_steps,
                "recorded_steps": recorded_steps,
                "outcome": outcome,
                "plan_sha256": plan_sha256,
            }
        ),
        encoding="utf-8",
    )
    commands = np.zeros((recorded_steps, 3), dtype=np.float64)
    commands[:, 0] = cell["velocity_mps"]
    np.savez_compressed(
        cell_dir / "sim_data.npz",
        time_steps=np.arange(recorded_steps, dtype=np.float64) * 0.02,
        desired_lin_vel=commands,
        true_lin_vel=commands,
        base_positions=np.zeros((recorded_steps, 2), dtype=np.float64),
        joint_torques=np.ones((recorded_steps, torque_width), dtype=np.float64),
        joint_powers=np.ones((recorded_steps, 12), dtype=np.float64),
        joint_effort_limits=np.full(12, 25.0, dtype=np.float64),
        foot_ground_reaction_forces_w=np.ones((recorded_steps, 4, 3), dtype=np.float64),
        episode_done=np.asarray(episode_done, dtype=bool),
        foot_thetas=np.tile(np.asarray(cell["phases"]), (recorded_steps, 1)),
        gait_periods=np.full(recorded_steps, 0.45),
        duty_factors=np.full(recorded_steps, 0.5),
        common_gait_phases=np.arange(recorded_steps, dtype=np.float64) * 0.04,
        configured_foot_thetas=np.asarray(cell["phases"]),
        gait_index=cell["gait_index"],
        velocity_mps=cell["velocity_mps"],
        step_dt=cell["step_dt"],
        expected_steps=status_expected_steps,
        recorded_steps=recorded_steps,
        cell_id=cell["id"],
        plan_sha256=plan_sha256,
    )

    assert helpers._validated_completed_cell(cell_dir, cell, plan_sha256) is expected


def test_publication_archive_validation_requires_effort_and_ground_identity(tmp_path):
    helpers = _load_grid_helpers()
    recorded_steps = 4
    cell = {
        "id": "cell_01",
        "gait_index": 0,
        "gait_name": "trot",
        "family": "trot",
        "seed": 42,
        "velocity_mps": 1.0,
        "phases": [0.0, 0.5, 0.5, 0.0],
    }
    plan_sha256 = "1" * 64
    commands = np.zeros((recorded_steps, 3), dtype=np.float64)
    commands[:, 0] = 1.0
    provenance = {
        "source": {
            "sha256": "a" * 64,
            "files": {"robot_config.py": "b" * 64},
            "missing_files": [],
        }
    }
    archive_fields = {
        "protocol_version": np.asarray("leg_usage_grid_full_v3"),
        "gait_name": np.asarray(cell["gait_name"]),
        "gait_family": np.asarray(cell["family"]),
        "evaluation_seed": np.asarray(cell["seed"]),
        "time_steps": np.arange(recorded_steps, dtype=np.float64) * 0.02,
        "desired_lin_vel": commands,
        "true_lin_vel": commands,
        "base_positions": np.zeros((recorded_steps, 2), dtype=np.float64),
        "base_headings": np.zeros(recorded_steps, dtype=np.float64),
        "desired_headings": np.zeros(recorded_steps, dtype=np.float64),
        "heading_sample_valid": np.ones(recorded_steps, dtype=bool),
        "pre_decision_common_gait_phases": np.arange(recorded_steps, dtype=np.float64) * 0.04,
        "actor_means": np.zeros((recorded_steps, 12), dtype=np.float64),
        "critic_values": np.zeros(recorded_steps, dtype=np.float64),
        "joint_positions": np.zeros((recorded_steps, 12), dtype=np.float64),
        "joint_velocities": np.zeros((recorded_steps, 12), dtype=np.float64),
        "requested_joint_position_targets": np.zeros((recorded_steps, 12), dtype=np.float64),
        "joint_position_lower_limits": np.full((recorded_steps, 12), -1.0, dtype=np.float64),
        "joint_position_upper_limits": np.full((recorded_steps, 12), 1.0, dtype=np.float64),
        "joint_torques": np.ones((recorded_steps, 12), dtype=np.float64),
        "joint_powers": np.ones((recorded_steps, 12), dtype=np.float64),
        "joint_effort_limits": np.full(12, 25.0),
        "joint_names": np.asarray([f"joint_{index}" for index in range(12)]),
        "leg_names": np.asarray(("Front Left", "Front Right", "Rear Left", "Rear Right")),
        "motor_role_names": np.asarray(("Hip/Abad", "Thigh", "Calf")),
        "foot_body_names": np.asarray(("FL_foot", "FR_foot", "RL_foot", "RR_foot")),
        "configured_joint_effort_limits": np.full(12, 25.0),
        "configured_effort_limit_source_by_joint": np.asarray([f"actuator_{index}" for index in range(12)]),
        "configured_effort_limit_fallback": np.asarray(False),
        "configured_effort_limits_valid": np.asarray(True),
        "effort_limit_provenance_json": np.asarray(json.dumps(provenance)),
        "foot_ground_reaction_forces_w": np.ones((recorded_steps, 4, 3), dtype=np.float64),
        "foot_normal_forces_w": np.ones((recorded_steps, 4, 3), dtype=np.float64),
        "foot_normal_force_is_ground_filtered": np.ones(recorded_steps, dtype=bool),
        "ground_filter_paths": np.asarray(["/World/ground/terrain/mesh"] * 4),
        "robot_mass_kg": np.asarray(20.0),
        "contact_threshold_on_n": np.asarray(20.0),
        "contact_threshold_off_n": np.asarray(10.0),
        "episode_done": np.zeros(recorded_steps, dtype=bool),
        "foot_thetas": np.tile(np.asarray(cell["phases"]), (recorded_steps, 1)),
        "gait_periods": np.full(recorded_steps, 0.5),
        "duty_factors": np.full(recorded_steps, 0.5),
        "common_gait_phases": np.arange(recorded_steps, dtype=np.float64) * 0.04,
        "configured_foot_thetas": np.asarray(cell["phases"]),
        "gait_index": np.asarray(cell["gait_index"]),
        "velocity_mps": np.asarray(cell["velocity_mps"]),
        "step_dt": np.asarray(0.02),
        "expected_steps": np.asarray(recorded_steps),
        "recorded_steps": np.asarray(recorded_steps),
        "cell_id": np.asarray(cell["id"]),
        "plan_sha256": np.asarray(plan_sha256),
    }

    def matches(fields: dict[str, np.ndarray], *, outcome: str = "completed") -> bool:
        path = tmp_path / "publication.npz"
        np.savez_compressed(path, **fields)
        with np.load(path, allow_pickle=False) as archive:
            return helpers._completed_cell_archive_matches(
                archive,
                cell,
                plan_sha256,
                recorded_steps=recorded_steps,
                expected_steps=recorded_steps,
                outcome=outcome,
                planned_step_dt=0.02,
                expected_velocity=1.0,
                expected_phases=np.asarray(cell["phases"]),
            )

    assert matches(archive_fields)
    required_mechanism_fields = {
        "pre_decision_common_gait_phases",
        "actor_means",
        "critic_values",
        "joint_positions",
        "joint_velocities",
        "requested_joint_position_targets",
        "joint_position_lower_limits",
        "joint_position_upper_limits",
        "joint_names",
        "leg_names",
        "motor_role_names",
        "foot_body_names",
    }
    for required_field in required_mechanism_fields:
        assert not matches({name: value for name, value in archive_fields.items() if name != required_field})
    assert not matches({**archive_fields, "protocol_version": np.asarray("leg_usage_grid_full_v2")})
    assert not matches({**archive_fields, "critic_values": np.full(recorded_steps, np.nan)})
    assert not matches(
        {
            **archive_fields,
            "joint_position_upper_limits": archive_fields["joint_position_lower_limits"].copy(),
        }
    )
    assert not matches({**archive_fields, "joint_names": np.asarray(["duplicate"] * 12)})
    assert not matches({**archive_fields, "configured_effort_limit_fallback": np.asarray(True)})
    assert not matches({**archive_fields, "ground_filter_paths": np.asarray(["/World/ground/not_terrain"] * 4)})
    assert not matches({**archive_fields, "effort_limit_provenance_json": np.asarray(json.dumps({"source": {}}))})
    assert not matches({name: value for name, value in archive_fields.items() if name != "base_headings"})
    assert not matches({**archive_fields, "gait_name": np.asarray("bound")})
    assert not matches({**archive_fields, "evaluation_seed": np.asarray(7)})


def test_terminated_publication_archive_rejects_post_reset_phase_rewind(tmp_path):
    helpers = _load_grid_helpers()
    recorded_steps = 4
    cell = {"id": "cell", "gait_index": 0, "velocity_mps": 1.0, "phases": [0.0, 0.5, 0.5, 0.0]}
    commands = np.zeros((recorded_steps, 3))
    commands[:, 0] = 1.0
    common = {
        "time_steps": np.arange(recorded_steps) * 0.02,
        "desired_lin_vel": commands,
        "true_lin_vel": commands,
        "base_positions": np.zeros((recorded_steps, 2)),
        "joint_torques": np.ones((recorded_steps, 12)),
        "joint_powers": np.ones((recorded_steps, 12)),
        "joint_effort_limits": np.ones(12),
        "foot_ground_reaction_forces_w": np.ones((recorded_steps, 4, 3)),
        "episode_done": np.asarray([False, False, False, True]),
        "foot_thetas": np.tile(cell["phases"], (recorded_steps, 1)),
        "gait_periods": np.full(recorded_steps, 0.5),
        "duty_factors": np.full(recorded_steps, 0.5),
        "configured_foot_thetas": np.asarray(cell["phases"]),
        "gait_index": np.asarray(0),
        "velocity_mps": np.asarray(1.0),
        "step_dt": np.asarray(0.02),
        "expected_steps": np.asarray(recorded_steps),
        "recorded_steps": np.asarray(recorded_steps),
        "cell_id": np.asarray("cell"),
        "plan_sha256": np.asarray("1" * 64),
    }

    def matches(phases: np.ndarray) -> bool:
        path = tmp_path / "terminated.npz"
        np.savez_compressed(path, common_gait_phases=phases, **common)
        with np.load(path, allow_pickle=False) as archive:
            return helpers._completed_cell_archive_matches(
                archive,
                cell,
                "1" * 64,
                recorded_steps=recorded_steps,
                expected_steps=recorded_steps,
                outcome="terminated",
                planned_step_dt=0.02,
                expected_velocity=1.0,
                expected_phases=np.asarray(cell["phases"]),
            )

    assert matches(np.asarray([0.0, 0.04, 0.08, 0.12]))
    assert not matches(np.asarray([0.0, 0.04, 0.08, 0.0]))


def test_recording_lock_is_exclusive_and_preserves_stale_lock(tmp_path):
    helpers = _load_grid_helpers()
    lock_path = tmp_path / ".recording.lock"

    with helpers._leg_usage_recording_lock(tmp_path, "a" * 64):
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="Remove this file deliberately"):
            with helpers._leg_usage_recording_lock(tmp_path, "b" * 64):
                pass
    assert not lock_path.exists()

    lock_path.write_text('{"pid": 1234}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="1234"):
        with helpers._leg_usage_recording_lock(tmp_path, "c" * 64):
            pass
    assert lock_path.read_text(encoding="utf-8") == '{"pid": 1234}\n'
