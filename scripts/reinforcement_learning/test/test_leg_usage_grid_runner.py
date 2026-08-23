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
        "_validate_leg_usage_gait_library",
        "_load_leg_usage_plan",
        "_validate_leg_usage_task",
        "_sha256_file",
        "_validate_leg_usage_checkpoint",
        "_leg_usage_step_counts",
        "_ordered_leg_usage_gaits",
        "_configured_gait_index_by_id",
        "_resolve_cell_output_dir",
        "_completed_cell_archive_matches",
        "_validated_completed_cell",
        "_leg_usage_recording_lock",
    }
    helper_nodes = [node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    namespace = {
        "Any": Any,
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
    }
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(module_path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in helper_names})


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


def test_plan_loader_accepts_sparse_stable_gait_ids(tmp_path):
    helpers = _load_grid_helpers()
    plan_path = _write_sparse_plan(tmp_path / "leg_usage_grid")

    plan, resolved_path, plan_sha256 = helpers._load_leg_usage_plan(str(plan_path))

    assert resolved_path == plan_path.resolve()
    assert [gait["index"] for gait in plan["gaits"]] == [6, 9]
    assert plan_sha256 == hashlib.sha256(plan_path.read_bytes()).hexdigest()


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
