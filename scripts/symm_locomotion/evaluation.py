# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run and summarize the symmetric locomotion policy evaluation grid."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _load_metrics_module():
    """Load the sibling pure-metrics module under file-based imports."""
    module_name = "_isaaclab_leg_usage_metrics"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module_path = Path(__file__).with_name("leg_usage_metrics.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load leg-usage metrics module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_paired_consistency_module():
    """Load the sibling paired time-reversal analysis module."""
    module_name = "_isaaclab_paired_tr_consistency"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module_path = Path(__file__).with_name("paired_tr_consistency.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load paired time-reversal analysis module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()

SCHEMA_VERSION = 1
# Retained for archived ``leg_usage_grid_v1`` consumers.  New studies always
# record one of the immutable protocol-specific versions below.
METHOD_VERSION = "leg_usage_grid_v1"
FULL_METHOD_VERSION = "leg_usage_grid_full_v3"
LIGHT_METHOD_VERSION = "leg_usage_grid_light_v2"
DEFAULT_PROTOCOL = "full"
LEGACY_PROTOCOL = "legacy"
PROTOCOL_METHOD_VERSIONS = {
    "full": FULL_METHOD_VERSION,
    "light": LIGHT_METHOD_VERSION,
    LEGACY_PROTOCOL: METHOD_VERSION,
}
# ``legacy`` is a CLI compatibility profile, not a new playback protocol.  Its
# persisted identity deliberately remains the historical v1/full pair accepted
# by existing runners and archives.
PROTOCOL_PLAN_NAMES = {"full": "full", "light": "light", LEGACY_PROTOCOL: "full"}
PROTOCOL_OUTPUT_ROOT_NAMES = {
    "full": "leg_usage_grid_full_v3",
    "light": "leg_usage_grid_light",
    LEGACY_PROTOCOL: "leg_usage_grid",
}
DEFAULT_VELOCITIES_MPS = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
DEFAULT_SETTLE_S = 5.0
DEFAULT_MEASURE_S = 10.0
DEFAULT_EVALUATION_SEED = 42
YAW_TRACKING_SUCCESS_THRESHOLD_RADPS = 0.05
FAMILY_ORDER = ("trot", "bound", "half_bound", "gallop")
PRIMARY_METRICS = ("normalized_torque_utilization", "absolute_work", "vertical_grf_impulse")
ALL_METRICS = ("torque_squared", *PRIMARY_METRICS)
SUPPORTED_ROBOTS = ("go2", "x1")
GROUND_COLLISION_PATH = "/World/ground/terrain/mesh"
METRIC_LABELS = {
    "torque_squared": "Raw torque squared",
    "normalized_torque_utilization": "Normalized torque squared",
    "absolute_work": "Absolute mechanical work",
    "vertical_grf_impulse": "Vertical GRF impulse",
}
METRIC_UNITS = {
    "torque_squared": "N^2 m^2 s",
    "normalized_torque_utilization": "s",
    "absolute_work": "J",
    "vertical_grf_impulse": "N s",
}
METRIC_DOMAIN_KEYS = {
    "torque_squared": "raw_load_metric_valid",
    "normalized_torque_utilization": "normalized_load_metric_valid",
    "absolute_work": "raw_load_metric_valid",
    "vertical_grf_impulse": "grf_load_metric_valid",
}


def canonical_training_gaits() -> tuple[str, tuple[dict[str, Any], ...]]:
    """Return the gait protocol derived from the task's canonical constants."""
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

    rows = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS
    weights = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS
    names = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES
    families = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES
    partners = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS
    lengths = {len(rows), len(weights), len(names), len(families), len(partners)}
    if len(lengths) != 1 or not rows:
        raise ValueError("Canonical symmetric training gait metadata has inconsistent lengths.")
    gaits = tuple(
        {
            "index": index,
            "name": str(name),
            "family": str(family),
            "phases": [float(value) for value in phases],
            "training_weight": float(weight),
            "weight": float(weight),
            "time_reversal_partner": int(partner),
        }
        for index, (phases, weight, name, family, partner) in enumerate(
            zip(rows, weights, names, families, partners, strict=True)
        )
    )
    return str(symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION), gaits


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _combined_file_provenance(repo_root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    files: dict[str, str] = {}
    missing: list[str] = []
    for path in paths:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            label = str(resolved)
        if resolved.is_file():
            files[label] = _sha256_file(resolved)
        else:
            missing.append(label)
    return {
        "sha256": _canonical_sha256(files),
        "files": files,
        "missing_files": sorted(missing),
    }


def _source_provenance(repo_root: Path, robot: str) -> dict[str, Any]:
    config_folder = "go2_symm" if robot == "go2" else "dobot_x1_symm"
    velocity_root = (
        repo_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "locomotion" / "velocity"
    )
    source_paths = (
        repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play_rsl_rl.py",
        repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "symm_rollout_plotter.py",
        repo_root / "scripts" / "symm_locomotion" / "evaluation.py",
        repo_root / "scripts" / "symm_locomotion" / "leg_usage_metrics.py",
        repo_root / "scripts" / "symm_locomotion" / "symm_cli.py",
        velocity_root / "mdp" / "symm_quadruped.py",
        velocity_root / "config" / config_folder / "flat_env_cfg.py",
    )
    result = _combined_file_provenance(repo_root, source_paths)
    result.update(
        {
            "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_branch": _git_value(repo_root, "branch", "--show-current"),
        }
    )
    return result


def _config_provenance(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    return _combined_file_provenance(
        repo_root,
        (run_dir / "params" / "agent.yaml", run_dir / "params" / "env.yaml"),
    )


def _steps_for_duration(duration_s: float, step_dt: float, label: str) -> int:
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"step_dt must be finite and positive; received {step_dt!r} s.")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"{label} must be finite and positive; received {duration_s!r} s.")
    steps = round(duration_s / step_dt)
    if steps < 1 or not math.isclose(duration_s, steps * step_dt, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"{label} must be an exact multiple of step_dt={step_dt:g} s; received {duration_s!r} s.")
    return steps


def velocity_slug(velocity_mps: float) -> str:
    """Return a stable directory label for a signed x velocity [m/s]."""
    sign = "pos" if velocity_mps > 0.0 else "neg"
    magnitude = f"{abs(velocity_mps):.6f}".rstrip("0").rstrip(".")
    if "." not in magnitude:
        magnitude += ".0"
    return f"{sign}_{magnitude.replace('.', 'p')}"


def _study_identity(study: dict[str, Any]) -> dict[str, Any]:
    legacy_keys = (
        "schema_version",
        "method_version",
        "robot",
        "task",
        "gait_library_version",
        "checkpoint",
        "source_provenance",
        "config_provenance",
        "step_dt",
        "settle_s",
        "measure_s",
        "settle_steps",
        "measure_steps",
        "total_steps",
        "episode_length_s",
        "evaluation_seed",
        "nominal_profile",
        "render_cell_plots",
        "runtime_overrides",
        "velocities_mps",
        "gaits",
        "cells",
        "aggregation",
    )
    if study.get("method_version") == METHOD_VERSION:
        return {key: study[key] for key in legacy_keys}
    keys = (
        *legacy_keys,
        "protocol",
        "evaluation_config",
        "contact_threshold_policy",
        "effort_limit_provenance",
    )
    return {key: study[key] for key in keys}


def validate_study_profile(study: Mapping[str, Any], profile: str) -> None:
    """Validate that a stored method/protocol pair implements a CLI profile."""
    if profile not in PROTOCOL_METHOD_VERSIONS:
        raise ValueError(f"Unknown leg-usage protocol profile {profile!r}.")
    method_version = study.get("method_version", METHOD_VERSION)
    if method_version != PROTOCOL_METHOD_VERSIONS[profile]:
        raise ValueError(
            "Existing leg-usage study belongs to a different method/profile: "
            f"requested {profile!r}, recorded {method_version!r}."
        )
    recorded_protocol = study.get("protocol", "full")
    if recorded_protocol != PROTOCOL_PLAN_NAMES[profile]:
        raise ValueError(
            "Existing leg-usage study has an incompatible persisted protocol: "
            f"requested profile {profile!r}, recorded {recorded_protocol!r}."
        )


def existing_study_manifest_path(run_dir: Path, profile: str) -> Path:
    """Resolve an existing profile manifest, including the pre-split full-v3 root."""
    if profile not in PROTOCOL_OUTPUT_ROOT_NAMES:
        raise ValueError(f"Unknown leg-usage protocol profile {profile!r}.")
    canonical = run_dir.resolve() / "evaluations" / PROTOCOL_OUTPUT_ROOT_NAMES[profile] / "study.json"
    if canonical.is_file() or profile != "full":
        return canonical
    historical = run_dir.resolve() / "evaluations" / PROTOCOL_OUTPUT_ROOT_NAMES[LEGACY_PROTOCOL] / "study.json"
    if historical.is_file():
        study = load_study_manifest(historical)
        if study.get("method_version") == FULL_METHOD_VERSION:
            return historical
    return canonical


def build_study(
    *,
    repo_root: Path,
    checkpoint: Path,
    robot: str,
    task: str,
    step_dt: float,
    settle_s: float = DEFAULT_SETTLE_S,
    measure_s: float = DEFAULT_MEASURE_S,
    evaluation_seed: int = DEFAULT_EVALUATION_SEED,
    velocities_mps: Sequence[float] = DEFAULT_VELOCITIES_MPS,
    gait_indices: Sequence[int] = tuple(range(10)),
    render_cell_plots: bool = False,
    runtime_overrides: Sequence[str] = (),
    protocol: str = DEFAULT_PROTOCOL,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the selected evaluation profile and runtime plan for one checkpoint."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported leg-usage robot: {robot}")
    if evaluation_seed < 0:
        raise ValueError("Evaluation seed must be non-negative.")
    settle_steps = _steps_for_duration(settle_s, step_dt, "Settle duration")
    measure_steps = _steps_for_duration(measure_s, step_dt, "Measurement duration")
    if protocol not in PROTOCOL_METHOD_VERSIONS:
        choices = ", ".join(repr(value) for value in PROTOCOL_METHOD_VERSIONS)
        raise ValueError(f"Unknown leg-usage protocol profile {protocol!r}; choose {choices}.")
    if protocol == LEGACY_PROTOCOL and evaluation_config is not None:
        raise ValueError("The legacy v1 compatibility profile does not accept an evaluation config.")
    if protocol == "full":
        if tuple(float(value) for value in velocities_mps) != DEFAULT_VELOCITIES_MPS:
            raise ValueError("The immutable full protocol requires its declared six-velocity grid.")
        if tuple(int(value) for value in gait_indices) != tuple(range(10)):
            raise ValueError("The immutable full protocol requires all ten canonical gait rows.")
    selected_velocities = [float(value) for value in velocities_mps]
    if not selected_velocities:
        raise ValueError("At least one x velocity is required.")
    if any(not math.isfinite(value) or value == 0.0 for value in selected_velocities):
        raise ValueError("Every x velocity must be finite and non-zero.")
    if len(set(selected_velocities)) != len(selected_velocities):
        raise ValueError("Duplicate x velocities are not allowed.")
    if len({velocity_slug(value) for value in selected_velocities}) != len(selected_velocities):
        raise ValueError("Selected x velocities must map to distinct output-directory labels.")
    gait_library_version, canonical_gaits = canonical_training_gaits()
    gait_by_index = {int(gait["index"]): gait for gait in canonical_gaits}
    if not gait_indices or len(set(gait_indices)) != len(gait_indices):
        raise ValueError("Gait indices must be a non-empty list without duplicates.")
    try:
        selected_gaits = [dict(gait_by_index[int(index)]) for index in gait_indices]
    except KeyError as exc:
        raise ValueError(f"Unknown gait index {exc.args[0]}; choose indices 0 through 9.") from exc

    light_cell_specs: list[tuple[dict[str, Any], float]] | None = None
    if protocol == "light":
        if tuple(float(value) for value in velocities_mps) != DEFAULT_VELOCITIES_MPS:
            raise ValueError("The immutable light protocol does not accept custom velocities.")
        if tuple(int(value) for value in gait_indices) != tuple(range(10)):
            raise ValueError("The immutable light protocol does not accept custom gait indices.")
        gait_by_name = {str(gait["name"]): gait for gait in canonical_gaits}

        def gait_and_partner(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
            gait = gait_by_name.get(name)
            if gait is None:
                raise ValueError(f"The canonical gait library is missing required stable row {name!r}.")
            partner_index = int(gait["time_reversal_partner"])
            partner = gait_by_index.get(partner_index)
            if partner is None or int(partner["time_reversal_partner"]) != int(gait["index"]):
                raise ValueError(f"Stable row {name!r} has invalid time-reversal partner metadata.")
            return dict(gait), dict(partner)

        trot = dict(gait_by_name["trot"])
        bound = dict(gait_by_name["bound"])
        half_bound, half_bound_partner = gait_and_partner("half_bound_front_a")
        gallop, gallop_partner = gait_and_partner("gallop_a")
        light_cell_specs = [
            (trot, 1.0),
            (trot, -1.0),
            (bound, 1.0),
            (bound, -1.0),
            (half_bound, 1.0),
            (half_bound_partner, -1.0),
            (gallop, 1.0),
            (gallop_partner, -1.0),
        ]
        selected_velocities = [1.0, -1.0]
        selected_gaits = []
        for gait, _ in light_cell_specs:
            if int(gait["index"]) not in {int(value["index"]) for value in selected_gaits}:
                selected_gaits.append(gait)

    run_dir = checkpoint.parent.resolve()
    output_root = run_dir / "evaluations" / PROTOCOL_OUTPUT_ROOT_NAMES[protocol]
    total_steps = settle_steps + measure_steps
    cells: list[dict[str, Any]] = []
    cell_specs = light_cell_specs or [(gait, velocity) for gait in selected_gaits for velocity in selected_velocities]
    for gait, velocity_mps in cell_specs:
        gait_index = int(gait["index"])
        family = str(gait["family"])
        slug = velocity_slug(velocity_mps)
        relative_output_dir = (
            Path("cells") / f"gait_{gait_index:02d}_{family}" / f"vx_{slug}" / f"seed_{evaluation_seed:04d}"
        )
        cells.append(
            {
                "id": f"gait_{gait_index:02d}_{family}__vx_{slug}__seed_{evaluation_seed:04d}",
                "gait_index": gait_index,
                "gait_name": gait["name"],
                "family": family,
                "phases": gait["phases"],
                "velocity_mps": velocity_mps,
                "seed": evaluation_seed,
                "step_dt": step_dt,
                "settle_steps": settle_steps,
                "measure_steps": measure_steps,
                "total_steps": total_steps,
                "relative_output_dir": relative_output_dir.as_posix(),
            }
        )
    checkpoint_record = {
        "path": str(checkpoint),
        "iteration": _model_iteration(checkpoint),
        "sha256": _sha256_file(checkpoint),
    }
    study: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_version": PROTOCOL_METHOD_VERSIONS[protocol],
        "protocol": PROTOCOL_PLAN_NAMES[protocol],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "robot": robot,
        "task": task,
        "gait_library_version": gait_library_version,
        "checkpoint": checkpoint_record,
        "source_provenance": _source_provenance(repo_root, robot),
        "config_provenance": _config_provenance(repo_root, run_dir),
        "step_dt": step_dt,
        "settle_s": settle_s,
        "measure_s": measure_s,
        "settle_steps": settle_steps,
        "measure_steps": measure_steps,
        "total_steps": total_steps,
        "episode_length_s": (total_steps + 2) * step_dt,
        "evaluation_seed": evaluation_seed,
        "nominal_profile": True,
        "render_cell_plots": render_cell_plots,
        "runtime_overrides": [str(value) for value in runtime_overrides],
        "velocities_mps": selected_velocities,
        "gaits": selected_gaits,
        "cells": cells,
        "aggregation": {
            "within_family": "equal mean across selected gait rows and velocities",
            "overall": "equal mean of trot, bound, half_bound, and gallop family means",
            "primary_balance_metric": "mean absolute per-cell front/hind imbalance; signed values never cancel",
            "tracking_qualified": "reported alongside all valid cells using the declared per-cell tracking threshold",
        },
    }
    if protocol != LEGACY_PROTOCOL:
        resolved_evaluation_config = _metrics.evaluation_config(evaluation_config)
        contact_config = resolved_evaluation_config["contact"]
        study.update(
            {
                "evaluation_config": resolved_evaluation_config,
                "contact_threshold_policy": {
                    "mode": contact_config["mode"],
                    "absolute_on_n": contact_config["on_n"] if contact_config["mode"] == "absolute" else None,
                    "absolute_off_n": contact_config["off_n"] if contact_config["mode"] == "absolute" else None,
                    "normalized_definition": "max(minimum_n, alpha * robot_mass_kg * 9.80665 / 4)"
                    if contact_config["mode"] == "body_weight"
                    else None,
                    "actual_threshold_archive_fields": ["contact_threshold_on_n", "contact_threshold_off_n"],
                    "actual_threshold_metrics_fields": [
                        "gait_contact_threshold_on_n",
                        "gait_contact_threshold_off_n",
                    ],
                    "actual_threshold_runtime_manifest": "<cell.relative_output_dir>/recording_manifest.json",
                },
                "effort_limit_provenance": {
                    "definition": "configured articulation actuator effort_limit resolved in action joint order",
                    "sentinel_rejection_threshold_nm": _metrics.EFFORT_LIMIT_SENTINEL_NM,
                    "source": _combined_file_provenance(
                        repo_root,
                        (
                            repo_root
                            / "source"
                            / "isaaclab_tasks"
                            / "isaaclab_tasks"
                            / "manager_based"
                            / "locomotion"
                            / "velocity"
                            / "config"
                            / ("go2_symm" if robot == "go2" else "dobot_x1_symm")
                            / "flat_env_cfg.py",
                        ),
                    ),
                    "fallback_allowed": False,
                },
            }
        )
    study["study_identity_sha256"] = _canonical_sha256(_study_identity(study))
    return study


def _model_iteration(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("model_"):
        return -1
    try:
        return int(stem.removeprefix("model_"))
    except ValueError:
        return -1


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_study_cell_dir(study_root: Path, relative_output_dir: str) -> Path:
    """Resolve one cell directory while keeping it inside the study root."""
    study_root = study_root.resolve()
    if not isinstance(relative_output_dir, str) or not relative_output_dir:
        raise ValueError("Study cell relative_output_dir must be a nonempty string.")
    relative_path = Path(relative_output_dir)
    if relative_path.is_absolute():
        raise ValueError(f"Study cell relative_output_dir must be relative: {relative_output_dir!r}.")
    cell_dir = (study_root / relative_path).resolve()
    if not cell_dir.is_relative_to(study_root):
        raise ValueError(f"Study cell output path escapes the study root: {relative_output_dir!r}.")
    return cell_dir


def _validate_study_cell_dirs(study: Mapping[str, Any], study_root: Path) -> None:
    """Validate that manifest cell directories are contained and unique."""
    cells = study.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Study manifest cells must be a nonempty list.")
    resolved_dirs: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(f"Study manifest cell {index} must be an object.")
        cell_dir = _resolve_study_cell_dir(study_root, cell.get("relative_output_dir"))
        normalized = os.path.normcase(str(cell_dir))
        if normalized in resolved_dirs:
            raise ValueError(
                "Study cell output directories must resolve uniquely inside the study root: "
                f"{cell.get('relative_output_dir')!r}."
            )
        resolved_dirs.add(normalized)


def load_study_manifest(study_path: Path) -> dict[str, Any]:
    """Load a study and verify its recorded immutable identity."""
    study_path = study_path.resolve()
    try:
        study = json.loads(study_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Leg-usage study manifest does not exist: {study_path}") from exc
    if not isinstance(study, dict):
        raise ValueError(f"Leg-usage study manifest must contain one JSON object: {study_path}")
    try:
        expected_digest = _canonical_sha256(_study_identity(study))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Study manifest is incomplete or incompatible: {study_path}") from exc
    if study.get("study_identity_sha256") != expected_digest:
        raise ValueError(f"Study identity digest does not match its protocol: {study_path}")
    declared_root = Path(str(study.get("output_root", ""))).expanduser().resolve()
    if declared_root != study_path.parent:
        raise ValueError(f"Study output_root does not match its manifest directory: {study_path}")
    _validate_study_cell_dirs(study, study_path.parent)
    return study


def validate_existing_study_for_analysis(
    study_path: Path,
    *,
    checkpoint: Path,
    robot: str,
    task: str,
    supplied_runtime_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate recorded inputs for analysis without comparing current source."""
    study_path = study_path.resolve()
    recording_lock = study_path.parent / ".recording.lock"
    if recording_lock.exists():
        raise ValueError(
            f"Leg-usage recording is already locked by another launcher: {recording_lock}. "
            "Wait for it to finish before analyzing."
        )
    study = load_study_manifest(study_path)
    checkpoint = checkpoint.resolve()
    recorded_checkpoint = study.get("checkpoint", {})
    if Path(str(recorded_checkpoint.get("path", ""))).expanduser().resolve() != checkpoint:
        raise ValueError("Existing leg-usage study belongs to a different checkpoint path.")
    if recorded_checkpoint.get("sha256") != _sha256_file(checkpoint):
        raise ValueError("Existing leg-usage study checkpoint SHA-256 does not match the selected checkpoint.")
    if study.get("robot") != robot or study.get("task") != task:
        raise ValueError("Existing leg-usage study robot/task does not match the selected utility robot.")
    supplied_overrides = [str(value) for value in supplied_runtime_overrides]
    if supplied_overrides and supplied_overrides != study.get("runtime_overrides", []):
        raise ValueError("Supplied analyze-only runtime overrides do not match the recorded study protocol.")
    return study


def validate_existing_legacy_study_for_resume(
    study_path: Path,
    *,
    checkpoint: Path,
    robot: str,
    task: str,
    step_dt: float,
    settle_s: float,
    measure_s: float,
    evaluation_seed: int,
    velocities_mps: Sequence[float],
    gait_indices: Sequence[int],
    render_cell_plots: bool,
    supplied_runtime_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Load a historical v1 plan and validate requested resume controls.

    Historical v1 identities include the source snapshot used to create the
    plan.  The compatibility module has since moved to :mod:`evaluation`, so a
    freshly rebuilt identity cannot equal an otherwise resumable archived
    plan.  Resume therefore treats the verified existing manifest as the plan
    of record and compares every runtime/grid control instead of rewriting it.
    """
    study = validate_existing_study_for_analysis(
        study_path,
        checkpoint=checkpoint,
        robot=robot,
        task=task,
        supplied_runtime_overrides=supplied_runtime_overrides,
    )
    validate_study_profile(study, LEGACY_PROTOCOL)
    requested_velocities = [float(value) for value in velocities_mps]
    requested_gaits = [int(value) for value in gait_indices]
    recorded_gaits = [int(gait["index"]) for gait in study.get("gaits", ())]
    mismatches: list[str] = []
    for label, recorded, requested in (
        ("step_dt", study.get("step_dt"), float(step_dt)),
        ("settle_s", study.get("settle_s"), float(settle_s)),
        ("measure_s", study.get("measure_s"), float(measure_s)),
        ("evaluation_seed", study.get("evaluation_seed"), int(evaluation_seed)),
        ("velocities", study.get("velocities_mps"), requested_velocities),
        ("gait_indices", recorded_gaits, requested_gaits),
        ("render_cell_plots", study.get("render_cell_plots"), bool(render_cell_plots)),
        (
            "runtime_overrides",
            study.get("runtime_overrides", []),
            [str(value) for value in supplied_runtime_overrides],
        ),
    ):
        if recorded != requested:
            mismatches.append(f"{label}: recorded {recorded!r}, requested {requested!r}")
    if mismatches:
        raise ValueError("Legacy v1 resume controls do not match the verified existing plan: " + "; ".join(mismatches))
    return study


def prepare_study(
    study: dict[str, Any],
    *,
    resume: bool,
    analyze_only: bool,
    dry_run: bool,
) -> Path:
    """Validate or create ``study.json`` without ever mixing protocols."""
    output_root = Path(study["output_root"])
    study_path = output_root / "study.json"
    recording_lock = output_root / ".recording.lock"
    if recording_lock.exists() and not dry_run:
        raise ValueError(
            f"Leg-usage recording is already locked by another launcher: {recording_lock}. "
            "Wait for it to finish; stale locks require deliberate removal after verifying no evaluator is running."
        )
    if study_path.exists():
        existing = json.loads(study_path.read_text(encoding="utf-8"))
        recorded_digest = existing.get("study_identity_sha256")
        try:
            actual_digest = _canonical_sha256(_study_identity(existing))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Existing study manifest is incomplete or incompatible: {study_path}") from exc
        if recorded_digest != actual_digest:
            raise ValueError(f"Existing study manifest has an invalid identity digest: {study_path}")
        if actual_digest != study["study_identity_sha256"]:
            raise ValueError(
                "The existing leg-usage output belongs to a different checkpoint or protocol. "
                f"Refusing to mix results in {output_root}."
            )
        if not (resume or analyze_only):
            has_cell_artifacts = (output_root / "cells").exists() and any((output_root / "cells").rglob("*"))
            has_runtime_artifacts = has_cell_artifacts or any(
                (output_root / name).exists() for name in ("events.jsonl", "progress.json", "status.json")
            )
            if has_runtime_artifacts:
                raise ValueError(f"Leg-usage results already exist in {output_root}; pass --resume to continue them.")
        study = existing
    elif analyze_only:
        raise FileNotFoundError(f"Analyze-only mode requires an existing study manifest: {study_path}")
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Refusing to initialize a study in non-empty directory without study.json: {output_root}")

    if not dry_run and not study_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(study_path, study)
    return study_path


def pair_imbalance(front: float, hind: float) -> float | None:
    """Return signed front/hind imbalance in percent, or ``None`` for no usage."""
    denominator = front + hind
    return None if denominator <= 0.0 else 100.0 * (front - hind) / denominator


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _runtime_failure(cell_dir: Path) -> str | None:
    status = _read_json_if_present(cell_dir / "status.json")
    metadata = _read_json_if_present(cell_dir / "metadata.json")
    state = str(status.get("state", status.get("status", ""))).lower()
    if state in {"failed", "failure", "invalid", "error", "interrupted", "running"}:
        return str(status.get("reason", status.get("message", status.get("error", state))))
    if status.get("terminated_early") is True or status.get("failed") is True:
        return str(status.get("reason", "runtime reported early termination"))
    for payload in (status, metadata):
        outcome = str(payload.get("outcome", "")).lower()
        if outcome in {"error", "interrupted", "failed", "failure"}:
            return str(payload.get("reason", payload.get("error", f"runtime outcome: {outcome}")))
    return None


def _period_s(data: Any, cell_dir: Path) -> float | None:
    for key in ("gait_period_s", "gait_period", "gait_periods", "command_gait_periods"):
        if key in data:
            values = np.asarray(data[key], dtype=float).reshape(-1)
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.size:
                return float(np.median(values))
    metadata = _read_json_if_present(cell_dir / "metadata.json")
    for key in ("gait_period_s", "gait_period"):
        value = metadata.get(key)
        if isinstance(value, float | int) and math.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    return None


def _metric_summary(values: np.ndarray, step_dt: float) -> dict[str, float | None]:
    front = float(values[:, :2].sum() * step_dt)
    hind = float(values[:, 2:].sum() * step_dt)
    signed = pair_imbalance(front, hind)
    return {
        "front_integral": front,
        "hind_integral": hind,
        "total_integral": front + hind,
        "signed_imbalance_percent": signed,
        "abs_imbalance_percent": None if signed is None else abs(signed),
    }


def _invalid_cell_row(cell: dict[str, Any], reason: str, status: str = "invalid") -> dict[str, Any]:
    return {
        "cell_id": cell["id"],
        "gait_index": cell["gait_index"],
        "gait_name": cell["gait_name"],
        "family": cell["family"],
        "velocity_mps": cell["velocity_mps"],
        "velocity_sign": "forward" if cell["velocity_mps"] > 0.0 else "backward",
        "seed": cell["seed"],
        "status": status,
        "reason": reason,
        "velocity_metric_valid": False,
        "velocity_metric_reason": reason,
        "heading_metric_valid": False,
        "heading_metric_reason": reason,
        "recording_manifest_valid": False,
        "recording_manifest_reason": reason,
        "gait_metric_valid": False,
        "gait_metric_reason": reason,
        "raw_load_metric_valid": False,
        "raw_load_metric_reason": reason,
        "grf_load_metric_valid": False,
        "grf_load_metric_reason": reason,
        "normalized_load_metric_valid": False,
        "load_metric_valid": False,
        "load_metric_reason": reason,
        "velocity_success_domain_valid": False,
        "gait_success_domain_valid": False,
        "joint_success_domain_valid": False,
        "gait_classification_available": False,
        "gait_row_correct": None,
        "gait_family_correct": None,
        "relative_output_dir": cell["relative_output_dir"],
    }


def _analyze_cell_legacy(study: dict[str, Any], cell: dict[str, Any], study_root: Path) -> dict[str, Any]:  # noqa: C901
    """Analyze one cell's steady-state measurement window."""
    cell_dir = _resolve_study_cell_dir(study_root, cell["relative_output_dir"])
    runtime_failure = _runtime_failure(cell_dir)
    if runtime_failure is not None:
        return _invalid_cell_row(cell, runtime_failure, "failed")
    plan_path = study_root / "study.json"
    if plan_path.is_file():
        expected_plan_sha256 = _sha256_file(plan_path)
        for metadata_path in (cell_dir / "status.json", cell_dir / "metadata.json"):
            payload = _read_json_if_present(metadata_path)
            recorded_plan_sha256 = payload.get("plan_sha256")
            if recorded_plan_sha256 is not None and recorded_plan_sha256 != expected_plan_sha256:
                return _invalid_cell_row(cell, f"{metadata_path.name} belongs to a different study plan")
    data_path = cell_dir / "sim_data.npz"
    if not data_path.is_file():
        return _invalid_cell_row(cell, "sim_data.npz is missing", "missing")
    try:
        with np.load(data_path, allow_pickle=False) as data:
            required = (
                "time_steps",
                "desired_lin_vel",
                "true_lin_vel",
                "base_positions",
                "joint_torques",
                "joint_powers",
                "joint_effort_limits",
                "foot_ground_reaction_forces_w",
                "episode_done",
            )
            missing = [key for key in required if key not in data]
            if missing:
                return _invalid_cell_row(cell, f"missing arrays: {', '.join(missing)}")
            times = np.asarray(data["time_steps"], dtype=float)
            commands = np.asarray(data["desired_lin_vel"], dtype=float)
            velocities = np.asarray(data["true_lin_vel"], dtype=float)
            positions = np.asarray(data["base_positions"], dtype=float)
            torques = np.asarray(data["joint_torques"], dtype=float)
            powers = np.asarray(data["joint_powers"], dtype=float)
            joint_effort_limits = np.asarray(data["joint_effort_limits"], dtype=float)
            ground_reaction_forces = np.asarray(data["foot_ground_reaction_forces_w"], dtype=float)
            episode_done = np.asarray(data["episode_done"], dtype=bool)
            period_s = _period_s(data, cell_dir)
            foot_thetas = np.asarray(data["foot_thetas"], dtype=float) if "foot_thetas" in data else None
            configured_foot_thetas = (
                np.asarray(data["configured_foot_thetas"], dtype=float) if "configured_foot_thetas" in data else None
            )
            common_gait_phases = (
                np.asarray(data["common_gait_phases"], dtype=float) if "common_gait_phases" in data else None
            )
            gait_period_trace = np.asarray(data["gait_periods"], dtype=float) if "gait_periods" in data else None
            archive_plan_sha256 = str(np.asarray(data["plan_sha256"]).item()) if "plan_sha256" in data else None
            archive_cell_id = str(np.asarray(data["cell_id"]).item()) if "cell_id" in data else None
    except (OSError, ValueError) as exc:
        return _invalid_cell_row(cell, f"cannot load sim_data.npz: {exc}")

    if plan_path.is_file() and archive_plan_sha256 is not None and archive_plan_sha256 != expected_plan_sha256:
        return _invalid_cell_row(cell, "sim_data.npz belongs to a different study plan")
    if archive_cell_id is not None and archive_cell_id != str(cell["id"]):
        return _invalid_cell_row(cell, f"sim_data.npz cell_id {archive_cell_id!r} does not match plan")

    expected_samples = int(cell["total_steps"])
    arrays = (times, commands, velocities, positions, torques, powers, ground_reaction_forces, episode_done)
    if any(array.shape[0] != expected_samples for array in arrays):
        counts = sorted({int(array.shape[0]) for array in arrays})
        status = "short" if min(counts) < expected_samples else "invalid"
        return _invalid_cell_row(
            cell,
            f"rollout sample count {counts} does not exactly match expected {expected_samples}",
            status,
        )
    if commands.shape[1:] != (3,) or velocities.shape[1:] != (3,) or positions.ndim != 2 or positions.shape[1] < 2:
        return _invalid_cell_row(cell, "unexpected command, velocity, or position shape")
    if torques.shape[1:] != (12,) or powers.shape[1:] != (12,):
        return _invalid_cell_row(cell, "expected 12 leg-major joints")
    if joint_effort_limits.shape == (expected_samples, 12):
        if not np.allclose(joint_effort_limits, joint_effort_limits[0], rtol=0.0, atol=1.0e-6):
            return _invalid_cell_row(cell, "joint_effort_limits changes within the fixed rollout")
        joint_effort_limits = joint_effort_limits[0]
    if (
        joint_effort_limits.shape != (12,)
        or not np.all(np.isfinite(joint_effort_limits))
        or not np.all(joint_effort_limits > 0.0)
    ):
        return _invalid_cell_row(cell, "expected 12 finite positive recorded joint effort limits")
    if ground_reaction_forces.shape[1:] != (4, 3):
        return _invalid_cell_row(cell, "expected ground reaction forces with shape (T, 4, 3)")
    if any(not np.all(np.isfinite(array[:expected_samples])) for array in arrays[:-1]):
        return _invalid_cell_row(cell, "non-finite values in required arrays")
    if bool(np.any(episode_done[:expected_samples])):
        first_done = int(np.flatnonzero(episode_done[:expected_samples])[0])
        return _invalid_cell_row(cell, f"episode ended at sample {first_done}", "failed")

    expected_velocity = float(cell["velocity_mps"])
    command_tolerance = 1.0e-5
    if not np.allclose(commands[:, 0], expected_velocity, rtol=0.0, atol=command_tolerance):
        observed_range = (float(np.min(commands[:, 0])), float(np.max(commands[:, 0])))
        return _invalid_cell_row(
            cell,
            f"x command is not fixed at {expected_velocity:g} m/s; observed range {observed_range}",
        )
    if not np.allclose(commands[:, 1:], 0.0, rtol=0.0, atol=command_tolerance):
        return _invalid_cell_row(cell, "lateral or yaw command is nonzero")
    expected_phases = np.asarray(cell["phases"], dtype=float)
    for label, recorded_phases in (
        ("foot_thetas", foot_thetas),
        ("configured_foot_thetas", configured_foot_thetas),
    ):
        if recorded_phases is None:
            continue
        if recorded_phases.shape == (4,):
            recorded_phases = recorded_phases.reshape(1, 4)
        if recorded_phases.ndim != 2 or recorded_phases.shape[1] != 4:
            return _invalid_cell_row(cell, f"unexpected {label} shape {recorded_phases.shape}")
        if label == "foot_thetas" and recorded_phases.shape[0] != expected_samples:
            return _invalid_cell_row(
                cell,
                f"foot_thetas sample count {recorded_phases.shape[0]} does not match expected {expected_samples}",
            )
        circular_error = np.remainder(recorded_phases - expected_phases + 0.5, 1.0) - 0.5
        if not np.all(np.isfinite(circular_error)) or float(np.max(np.abs(circular_error))) > 1.0e-5:
            return _invalid_cell_row(cell, f"{label} does not match planned gait row {cell['gait_index']}")
    if gait_period_trace is not None:
        gait_period_trace = gait_period_trace.reshape(-1)
        if (
            gait_period_trace.shape != (expected_samples,)
            or not np.all(np.isfinite(gait_period_trace))
            or not np.all(gait_period_trace > 0.0)
            or not np.allclose(gait_period_trace, gait_period_trace[0], rtol=0.0, atol=1.0e-6)
        ):
            return _invalid_cell_row(cell, "gait_periods is not a fixed positive per-sample trace")

    step_dt = float(study["step_dt"])
    time_deltas = np.diff(times[:expected_samples])
    if not np.all(time_deltas > 0.0) or not np.allclose(time_deltas, step_dt, rtol=0.0, atol=1.0e-6):
        observed_range = (float(np.min(time_deltas)), float(np.max(time_deltas)))
        return _invalid_cell_row(
            cell,
            f"time_steps are not strictly uniform at plan step_dt={step_dt:g}; observed delta range {observed_range}",
        )
    start = int(cell["settle_steps"])
    stop = start + int(cell["measure_steps"])
    complete_cycles = None
    phase_boundary_trimmed = False
    if common_gait_phases is not None:
        phases = np.asarray(common_gait_phases, dtype=float).reshape(-1)
        if phases.shape != (expected_samples,) or not np.all(np.isfinite(phases)):
            return _invalid_cell_row(cell, "common_gait_phases is present but invalid")
        if np.all(np.diff(phases) >= 0.0):
            first_boundary = math.ceil(float(phases[start]) - 1.0e-8)
            last_boundary = math.floor(float(phases[stop - 1]) + 1.0e-8)
            if last_boundary - first_boundary >= 1:
                first_index = int(np.searchsorted(phases, first_boundary, side="left"))
                last_index = int(np.searchsorted(phases, last_boundary, side="left"))
                if start <= first_index < last_index <= stop:
                    start = first_index
                    stop = last_index
                    complete_cycles = last_boundary - first_boundary
                    phase_boundary_trimmed = True
    if not phase_boundary_trimmed and period_s is not None:
        complete_cycles = math.floor((stop - start) * step_dt / period_s + 1.0e-9)
        if complete_cycles >= 1:
            cycle_samples = round(complete_cycles * period_s / step_dt)
            stop = min(stop, start + cycle_samples)
        else:
            complete_cycles = None
    if stop - start < 2:
        return _invalid_cell_row(cell, "measurement window has fewer than two samples")
    selected = slice(start, stop)
    selected_torques = torques[selected].reshape(-1, 4, 3)
    selected_powers = powers[selected].reshape(-1, 4, 3)
    effort_limits = joint_effort_limits.reshape(4, 3)
    metric_values = {
        "torque_squared": np.square(selected_torques).sum(axis=2),
        "normalized_torque_utilization": np.square(selected_torques / effort_limits).sum(axis=2),
        "absolute_work": np.abs(selected_powers).sum(axis=2),
        "vertical_grf_impulse": np.maximum(ground_reaction_forces[selected, :, 2], 0.0),
    }

    command_x = commands[selected, 0]
    velocity_x = velocities[selected, 0]
    command_sign = 1.0 if float(cell["velocity_mps"]) > 0.0 else -1.0
    position_displacement_x = float(positions[stop - 1, 0] - positions[start - 1, 0])
    directed_progress = command_sign * position_displacement_x
    integrated_body_x_progress = float(np.sum(command_sign * velocity_x) * step_dt)
    planar_tracking_errors = velocities[selected, :2] - commands[selected, :2]
    tracking_rmse = float(np.sqrt(np.mean(np.square(planar_tracking_errors).sum(axis=1))))
    yaw_tracking_rmse = float(np.sqrt(np.mean(np.square(velocities[selected, 2] - commands[selected, 2]))))
    tracking_success_threshold = 0.05 + 0.25 * abs(expected_velocity)
    planar_tracking_success = tracking_rmse <= tracking_success_threshold
    yaw_tracking_success = yaw_tracking_rmse <= YAW_TRACKING_SUCCESS_THRESHOLD_RADPS
    row = _invalid_cell_row(cell, "", "valid")
    row.update(
        {
            "reason": "",
            "sample_count": stop - start,
            "measurement_start_s": start * step_dt,
            "measurement_stop_s": stop * step_dt,
            "measurement_duration_s": (stop - start) * step_dt,
            "complete_cycles": complete_cycles,
            "phase_boundary_trimmed": phase_boundary_trimmed,
            "gait_period_s": period_s,
            "effort_limit_source": "recorded_joint_effort_limits",
            "effort_limit_min_nm": float(np.min(joint_effort_limits)),
            "effort_limit_max_nm": float(np.max(joint_effort_limits)),
            "command_x_mean_mps": float(np.mean(command_x)),
            "measured_x_mean_mps": float(np.mean(velocity_x)),
            "tracking_rmse_mps": tracking_rmse,
            "yaw_tracking_rmse_radps": yaw_tracking_rmse,
            "tracking_success_threshold_mps": tracking_success_threshold,
            "yaw_tracking_success_threshold_radps": YAW_TRACKING_SUCCESS_THRESHOLD_RADPS,
            "planar_tracking_success": planar_tracking_success,
            "yaw_tracking_success": yaw_tracking_success,
            "tracking_success": planar_tracking_success and yaw_tracking_success,
            "signed_directed_progress_m": directed_progress,
            "integrated_body_x_progress_m": integrated_body_x_progress,
            "position_vs_integrated_progress_difference_m": directed_progress - integrated_body_x_progress,
            "directed_progress_definition": (
                "commanded-sign world-x displacement across the same sample intervals as effort integration"
            ),
            "positive_directed_progress": directed_progress > 0.0,
        }
    )
    for metric, values in metric_values.items():
        summary = _metric_summary(values, step_dt)
        duration_s = float(row["measurement_duration_s"])
        prefix = f"{metric}_"
        for name, value in summary.items():
            row[prefix + name] = value
        total = float(summary["total_integral"])
        row[prefix + "total_per_s"] = total / duration_s
        row[prefix + "total_per_directed_m"] = total / directed_progress if directed_progress > 0.0 else None
    if directed_progress <= 0.0:
        row["status"] = "nonpositive_progress"
        row["reason"] = "commanded-direction progress is nonpositive; per-distance metrics are N/A"
    return row


def _json_cell_value(value: Any) -> Any:
    """Serialize structured metric values deterministically for flat CSV cells."""
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _effort_provenance_reason(archived: Any, expected: Any) -> str:
    """Return a fail-closed C1 provenance reason, or an empty string when valid."""
    if not isinstance(archived, dict):
        return "effort-limit provenance JSON must contain an object"
    source = archived.get("source")
    if not isinstance(source, dict):
        return "effort-limit provenance source must contain an object"
    source_sha256 = source.get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        return "effort-limit provenance source SHA-256 is missing or malformed"
    try:
        int(source_sha256, 16)
    except ValueError:
        return "effort-limit provenance source SHA-256 is not hexadecimal"
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        return "effort-limit provenance source file map is missing or empty"
    for path, sha256 in files.items():
        if not isinstance(path, str) or not path.strip():
            return "effort-limit provenance contains an empty source path"
        if not isinstance(sha256, str) or len(sha256) != 64:
            return f"effort-limit provenance hash for {path!r} is missing or malformed"
        try:
            int(sha256, 16)
        except ValueError:
            return f"effort-limit provenance hash for {path!r} is not hexadecimal"
    if source.get("missing_files") not in ([], ()):
        return "effort-limit provenance declares missing source files"
    if archived.get("fallback_allowed") is not False:
        return "effort-limit provenance does not explicitly forbid fallback limits"
    if not isinstance(expected, dict) or archived != expected:
        return "effort-limit source path/hash identity does not match the immutable study manifest"
    return ""


def _ground_filter_requirement_reason(
    evaluation_config: Mapping[str, Any],
    *,
    ground_filter_declared: bool,
    selected_ground_filtered: np.ndarray,
) -> str:
    """Return why contact forces violate the configured source-filter requirement."""
    if not bool(evaluation_config["contact"]["ground_filtered_required"]):
        return ""
    if not ground_filter_declared:
        return "foot force filters do not resolve exclusively to the declared terrain path"
    if not bool(np.all(selected_ground_filtered)):
        return "foot normal forces are not explicitly ground-filtered"
    return ""


def _tracking_qualification(
    evaluation_config: Mapping[str, Any],
    *,
    expected_velocity_mps: float,
    planar_rmse_mps: float,
    vx_relative_rmse: float,
    yaw_rmse_radps: float,
) -> dict[str, float | bool]:
    """Apply configured diagnostics while preserving the established planar rule."""
    config = evaluation_config["tracking"]
    planar_threshold = 0.05 + 0.25 * abs(expected_velocity_mps)
    yaw_threshold = float(config["yaw_rmse_limit_radps"])
    vx_relative_threshold = float(config["vx_relative_error_limit"])
    planar_success = planar_rmse_mps <= planar_threshold
    yaw_success = yaw_rmse_radps <= yaw_threshold
    return {
        "tracking_success_threshold_mps": planar_threshold,
        "yaw_tracking_success_threshold_radps": yaw_threshold,
        "vx_relative_tracking_success_threshold": vx_relative_threshold,
        "vx_relative_tracking_success": vx_relative_rmse <= vx_relative_threshold,
        "planar_tracking_success": planar_success,
        "yaw_tracking_success": yaw_success,
        "tracking_success": planar_success and yaw_success,
    }


def _recording_manifest_reason(
    *,
    cell_dir: Path,
    data_path: Path,
    study: Mapping[str, Any],
    cell: Mapping[str, Any],
    expected_plan_sha256: str,
    archive_plan_sha256: str,
    recorded_steps: int,
    robot_mass_kg: float,
    contact_threshold_on_n: float,
    contact_threshold_off_n: float,
    ground_filter_paths: Sequence[str],
    ground_filtered_samples: int,
) -> str:
    """Validate the immutable per-cell manifest of resolved runtime contact quantities."""
    path = cell_dir / "recording_manifest.json"
    if not path.is_file():
        return "recording_manifest.json is missing"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"recording_manifest.json cannot be loaded: {exc}"
    if not isinstance(manifest, dict):
        return "recording_manifest.json must contain one object"
    declared_hash = manifest.pop("record_sha256", None)
    try:
        computed_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        return f"recording manifest is not canonically serializable: {exc}"
    if declared_hash != computed_hash:
        return "recording manifest record SHA-256 is invalid"

    def integer_matches(name: str, expected: int) -> bool:
        try:
            return int(manifest[name]) == expected
        except (KeyError, TypeError, ValueError):
            return False

    checks = (
        (manifest.get("schema_version") == 1, "recording manifest schema_version is invalid"),
        (
            manifest.get("method_version") == study.get("method_version"),
            "recording manifest method identity does not match the study",
        ),
        (
            manifest.get("plan_sha256") == expected_plan_sha256 == archive_plan_sha256,
            "recording manifest plan SHA-256 does not match the study/archive",
        ),
        (manifest.get("cell_id") == cell.get("id"), "recording manifest cell identity is invalid"),
        (
            integer_matches("gait_index", int(cell.get("gait_index", -2))),
            "recording manifest gait identity is invalid",
        ),
        (
            integer_matches("evaluation_seed", int(cell.get("seed", -2))),
            "recording manifest evaluation seed is invalid",
        ),
        (
            integer_matches("recorded_steps", recorded_steps),
            "recording manifest sample count is invalid",
        ),
        (
            manifest.get("archive_sha256") == _sha256_file(data_path),
            "recording manifest archive SHA-256 is invalid",
        ),
        (
            manifest.get("contact_threshold_mode") == study["evaluation_config"]["contact"]["mode"],
            "recording manifest contact-threshold mode is invalid",
        ),
        (
            manifest.get("ground_filter_paths") == list(ground_filter_paths),
            "recording manifest ground-filter paths do not match the archive",
        ),
        (
            integer_matches("ground_filtered_samples", ground_filtered_samples),
            "recording manifest ground-filtered sample count is invalid",
        ),
    )
    for valid, reason in checks:
        if not valid:
            return reason
    for key, expected_value in (
        ("robot_mass_kg", robot_mass_kg),
        ("contact_threshold_on_n", contact_threshold_on_n),
        ("contact_threshold_off_n", contact_threshold_off_n),
    ):
        try:
            value = float(manifest[key])
        except (KeyError, TypeError, ValueError):
            return f"recording manifest {key} is missing or malformed"
        if not math.isfinite(value) or not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=1.0e-9):
            return f"recording manifest {key} does not match the archive"
    return ""


def _measurement_bounds(
    common_phase: np.ndarray,
    settle_steps: int,
    recorded_steps: int,
) -> tuple[int, int, int, bool]:
    """Trim a recorded post-settle interval to complete common gait cycles."""
    start = settle_steps
    stop = recorded_steps
    first_boundary = math.ceil(float(common_phase[start]) - 1.0e-8)
    last_boundary = math.floor(float(common_phase[stop - 1]) + 1.0e-8)
    complete_cycles = max(0, last_boundary - first_boundary)
    if complete_cycles < 1:
        return start, stop, complete_cycles, False
    trimmed_start = int(np.searchsorted(common_phase, first_boundary, side="left"))
    trimmed_stop = int(np.searchsorted(common_phase, last_boundary, side="left"))
    if start <= trimmed_start < trimmed_stop <= stop:
        return trimmed_start, trimmed_stop, complete_cycles, True
    return start, stop, 0, False


def _publication_cell(study: dict[str, Any], cell: dict[str, Any], study_root: Path) -> dict[str, Any]:  # noqa: C901
    """Analyze one full/light protocol cell while keeping metric domains independent."""
    cell_dir = _resolve_study_cell_dir(study_root, cell["relative_output_dir"])
    runtime_failure = _runtime_failure(cell_dir)
    if runtime_failure is not None:
        return _invalid_cell_row(cell, runtime_failure, "failed")
    data_path = cell_dir / "sim_data.npz"
    if not data_path.is_file():
        return _invalid_cell_row(cell, "sim_data.npz is missing", "missing")
    plan_path = study_root / "study.json"
    expected_plan_sha256 = _sha256_file(plan_path) if plan_path.is_file() else None
    required = {
        "protocol_version",
        "plan_sha256",
        "cell_id",
        "gait_index",
        "gait_name",
        "gait_family",
        "evaluation_seed",
        "velocity_mps",
        "step_dt",
        "expected_steps",
        "recorded_steps",
        "configured_foot_thetas",
        "time_steps",
        "desired_lin_vel",
        "true_lin_vel",
        "base_positions",
        "joint_torques",
        "joint_powers",
        "episode_done",
        "foot_thetas",
        "duty_factors",
        "common_gait_phases",
    }
    try:
        with np.load(data_path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                return _invalid_cell_row(cell, f"missing publication arrays: {', '.join(missing)}")
            archive_protocol_version = str(data["protocol_version"].item())
            archive_plan_sha256 = str(data["plan_sha256"].item())
            archive_cell_id = str(data["cell_id"].item())
            archive_gait_index = int(data["gait_index"].item())
            archive_gait_name = str(data["gait_name"].item())
            archive_gait_family = str(data["gait_family"].item())
            archive_evaluation_seed = int(data["evaluation_seed"].item())
            archive_velocity_mps = float(data["velocity_mps"].item())
            archive_step_dt = float(data["step_dt"].item())
            archive_expected_steps = int(data["expected_steps"].item())
            archive_recorded_steps = int(data["recorded_steps"].item())
            archive_configured_offsets = np.asarray(data["configured_foot_thetas"], dtype=float)
            times = np.asarray(data["time_steps"], dtype=float).reshape(-1)
            commands = np.asarray(data["desired_lin_vel"], dtype=float)
            velocities = np.asarray(data["true_lin_vel"], dtype=float)
            positions = np.asarray(data["base_positions"], dtype=float)
            torques = np.asarray(data["joint_torques"], dtype=float)
            powers = np.asarray(data["joint_powers"], dtype=float)
            foot_normal_forces = (
                np.asarray(data["foot_normal_forces_w"], dtype=float)
                if "foot_normal_forces_w" in data
                else np.asarray([], dtype=float)
            )
            ground_filtered = (
                np.asarray(data["foot_normal_force_is_ground_filtered"], dtype=bool).reshape(-1)
                if "foot_normal_force_is_ground_filtered" in data
                else np.asarray([], dtype=bool)
            )
            episode_done = np.asarray(data["episode_done"], dtype=bool).reshape(-1)
            foot_thetas = np.asarray(data["foot_thetas"], dtype=float)
            duty_factors = np.asarray(data["duty_factors"], dtype=float).reshape(-1)
            common_phase = np.asarray(data["common_gait_phases"], dtype=float).reshape(-1)
            joint_name_values = np.asarray(data["joint_names"]) if "joint_names" in data else np.asarray([])
            joint_names = [str(value) for value in joint_name_values.reshape(-1).tolist()]
            configured_limits_parse_reason = ""
            if "configured_joint_effort_limits" not in data:
                configured_limits = np.asarray([], dtype=float)
                configured_limits_parse_reason = "configured effort-limit vector is missing"
            else:
                try:
                    configured_limits = np.asarray(data["configured_joint_effort_limits"], dtype=float)
                except (TypeError, ValueError) as exc:
                    configured_limits = np.asarray([], dtype=float)
                    configured_limits_parse_reason = f"configured effort-limit vector is malformed: {exc}"
            effort_source_values = (
                np.asarray(data["configured_effort_limit_source_by_joint"])
                if "configured_effort_limit_source_by_joint" in data
                else np.asarray([])
            )
            effort_sources = [str(value) for value in effort_source_values.reshape(-1).tolist()]
            fallback_values = (
                np.asarray(data["configured_effort_limit_fallback"])
                if "configured_effort_limit_fallback" in data
                else np.asarray([])
            )
            valid_values = (
                np.asarray(data["configured_effort_limits_valid"])
                if "configured_effort_limits_valid" in data
                else np.asarray([])
            )
            effort_fallback = bool(fallback_values.item()) if fallback_values.shape == () else None
            effort_valid = bool(valid_values.item()) if valid_values.shape == () else None
            effort_provenance: Any = None
            effort_provenance_parse_reason = ""
            if "effort_limit_provenance_json" not in data:
                effort_provenance_parse_reason = "effort-limit provenance JSON is missing"
            else:
                effort_json_values = np.asarray(data["effort_limit_provenance_json"])
                if effort_json_values.shape != ():
                    effort_provenance_parse_reason = "effort-limit provenance JSON must be scalar"
                else:
                    try:
                        effort_provenance = json.loads(str(effort_json_values.item()))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        effort_provenance_parse_reason = f"effort-limit provenance JSON is malformed: {exc}"
            ground_filter_values = (
                np.asarray(data["ground_filter_paths"]) if "ground_filter_paths" in data else np.asarray([])
            )
            ground_filter_paths = [str(value) for value in ground_filter_values.reshape(-1).tolist()]
            robot_mass_kg = float(data["robot_mass_kg"].item()) if "robot_mass_kg" in data else math.nan
            archived_contact_on_n = (
                float(data["contact_threshold_on_n"].item()) if "contact_threshold_on_n" in data else None
            )
            archived_contact_off_n = (
                float(data["contact_threshold_off_n"].item()) if "contact_threshold_off_n" in data else None
            )
            base_headings = (
                np.asarray(data["base_headings"], dtype=float)
                if "base_headings" in data
                else np.asarray([], dtype=float)
            )
            desired_headings = (
                np.asarray(data["desired_headings"], dtype=float)
                if "desired_headings" in data
                else np.asarray([], dtype=float)
            )
            heading_valid = (
                np.asarray(data["heading_sample_valid"], dtype=bool)
                if "heading_sample_valid" in data
                else np.asarray([], dtype=bool)
            )
            soft_utilization = (
                np.asarray(data["joint_target_limit_utilization"], dtype=float)
                if "joint_target_limit_utilization" in data
                else None
            )
            action_clamped = (
                np.asarray(data["joint_action_clamped"], dtype=float) if "joint_action_clamped" in data else None
            )
            action_clamp_fraction = (
                np.asarray(data["action_clamp_fraction"], dtype=float) if "action_clamp_fraction" in data else None
            )
    except (OSError, ValueError) as exc:
        return _invalid_cell_row(cell, f"cannot load sim_data.npz: {exc}")

    if expected_plan_sha256 is None:
        return _invalid_cell_row(cell, "immutable study.json is missing")
    identity_checks = (
        (archive_protocol_version == study.get("method_version"), "archive method/protocol identity mismatch"),
        (archive_plan_sha256 == expected_plan_sha256, "archive plan SHA-256 mismatch"),
        (archive_cell_id == str(cell["id"]), "archive cell identity mismatch"),
        (archive_gait_index == int(cell["gait_index"]), "archive gait-index identity mismatch"),
        (archive_gait_name == str(cell["gait_name"]), "archive gait-row identity mismatch"),
        (archive_gait_family == str(cell["family"]), "archive gait-family identity mismatch"),
        (archive_evaluation_seed == int(cell["seed"]), "archive evaluation-seed identity mismatch"),
        (
            math.isclose(archive_velocity_mps, float(cell["velocity_mps"]), rel_tol=0.0, abs_tol=1.0e-9),
            "archive velocity identity mismatch",
        ),
        (
            math.isclose(archive_step_dt, float(study["step_dt"]), rel_tol=0.0, abs_tol=1.0e-9),
            "archive step_dt identity mismatch",
        ),
        (archive_expected_steps == int(cell["total_steps"]), "archive expected-step identity mismatch"),
        (archive_configured_offsets.shape == (4,), "archive configured gait offsets have invalid shape"),
    )
    for valid_identity, reason in identity_checks:
        if not valid_identity:
            return _invalid_cell_row(cell, reason)
    configured_offset_error = _metrics.circular_difference(
        archive_configured_offsets, np.asarray(cell["phases"], dtype=float)
    )
    if np.max(np.abs(configured_offset_error)) > 1.0e-5:
        return _invalid_cell_row(cell, "archive configured gait offsets do not match the planned row")
    expected_effort_provenance = study.get("effort_limit_provenance")
    effort_reasons: list[str] = []
    if joint_name_values.shape != (12,) or len(set(joint_names)) != 12 or any(not name for name in joint_names):
        effort_reasons.append("joint-name identity/order is missing or malformed")
    if configured_limits_parse_reason:
        effort_reasons.append(configured_limits_parse_reason)
    else:
        try:
            _metrics.validate_effort_limits(configured_limits)
        except (TypeError, ValueError) as exc:
            effort_reasons.append(f"configured effort-limit vector is invalid: {exc}")
    if effort_source_values.shape != (12,) or any(
        not source or "fallback" in source.lower() for source in effort_sources
    ):
        effort_reasons.append("per-joint configured effort-limit sources are missing or malformed")
    if effort_fallback is not False:
        effort_reasons.append("configured effort-limit fallback flag is missing, malformed, or true")
    if effort_valid is not True:
        effort_reasons.append("configured effort-limit valid flag is missing, malformed, or false")
    provenance_reason = effort_provenance_parse_reason or _effort_provenance_reason(
        effort_provenance, expected_effort_provenance
    )
    if provenance_reason:
        effort_reasons.append(provenance_reason)
    effort_provenance_reason = "; ".join(effort_reasons)
    ground_filter_declared = ground_filter_values.shape == (4,) and all(
        path == GROUND_COLLISION_PATH for path in ground_filter_paths
    )
    recorded_steps = len(times)
    expected_steps = int(cell["total_steps"])
    if archive_recorded_steps != recorded_steps:
        return _invalid_cell_row(cell, "archive recorded-step identity does not match its time-major arrays")
    time_major = (
        commands,
        velocities,
        positions,
        torques,
        powers,
        episode_done,
        foot_thetas,
        duty_factors,
        common_phase,
    )
    if recorded_steps < 1 or any(len(array) != recorded_steps for array in time_major):
        return _invalid_cell_row(cell, "time-major archive arrays have inconsistent sample counts")
    terminated = bool(np.any(episode_done))
    if recorded_steps != expected_steps and not terminated:
        status = "short" if recorded_steps < expected_steps else "invalid"
        return _invalid_cell_row(
            cell,
            f"rollout recorded {recorded_steps} samples but protocol requires {expected_steps}",
            status,
        )
    if recorded_steps > expected_steps or (terminated and (not episode_done[-1] or np.any(episode_done[:-1]))):
        return _invalid_cell_row(cell, "termination/sample-count trace is internally inconsistent")
    if (
        commands.shape != (recorded_steps, 3)
        or velocities.shape != (recorded_steps, 3)
        or positions.ndim != 2
        or positions.shape[1] < 2
        or torques.shape != (recorded_steps, 12)
        or powers.shape != (recorded_steps, 12)
        or foot_thetas.shape != (recorded_steps, 4)
    ):
        return _invalid_cell_row(cell, "publication archive arrays have unexpected shapes")
    ground_force_reason = ""
    if foot_normal_forces.size == 0:
        ground_force_reason = "foot normal-force array is missing"
    elif foot_normal_forces.shape != (recorded_steps, 4, 3):
        ground_force_reason = "foot normal-force array has unexpected shape"
    elif not np.all(np.isfinite(foot_normal_forces)):
        ground_force_reason = "foot normal-force array contains non-finite samples"
    elif ground_filtered.shape != (recorded_steps,):
        ground_force_reason = "ground-filter source-validity trace has unexpected shape"
    numeric = (
        times,
        commands,
        velocities,
        positions,
        torques,
        powers,
        foot_thetas,
        duty_factors,
        common_phase,
    )
    if any(not np.all(np.isfinite(array)) for array in numeric):
        return _invalid_cell_row(cell, "publication archive contains non-finite required samples")
    recording_manifest_reason = _recording_manifest_reason(
        cell_dir=cell_dir,
        data_path=data_path,
        study=study,
        cell=cell,
        expected_plan_sha256=expected_plan_sha256,
        archive_plan_sha256=archive_plan_sha256,
        recorded_steps=recorded_steps,
        robot_mass_kg=robot_mass_kg,
        contact_threshold_on_n=(math.nan if archived_contact_on_n is None else float(archived_contact_on_n)),
        contact_threshold_off_n=(math.nan if archived_contact_off_n is None else float(archived_contact_off_n)),
        ground_filter_paths=ground_filter_paths,
        ground_filtered_samples=int(np.count_nonzero(ground_filtered)),
    )
    step_dt = float(study["step_dt"])
    if recorded_steps > 1 and (
        np.any(np.diff(times) <= 0.0) or not np.allclose(np.diff(times), step_dt, rtol=0.0, atol=1.0e-6)
    ):
        return _invalid_cell_row(cell, "time_steps do not use the study's fixed step_dt")
    if np.any(np.diff(common_phase) < 0.0):
        return _invalid_cell_row(cell, "common gait phase is not monotonic")
    expected_velocity = float(cell["velocity_mps"])
    if not np.allclose(commands[:, 0], expected_velocity, rtol=0.0, atol=1.0e-5):
        return _invalid_cell_row(cell, "x command is not fixed at the planned velocity")
    if not np.allclose(commands[:, 1:], 0.0, rtol=0.0, atol=1.0e-5):
        return _invalid_cell_row(cell, "lateral or yaw command is nonzero")
    expected_offsets = np.asarray(cell["phases"], dtype=float)
    phase_error = _metrics.circular_difference(foot_thetas, expected_offsets[None, :])
    if np.max(np.abs(phase_error)) > 1.0e-5:
        return _invalid_cell_row(cell, "recorded foot phase offsets do not match the planned gait row")
    if np.any((duty_factors <= 0.0) | (duty_factors >= 1.0)):
        return _invalid_cell_row(cell, "duty factors are outside (0, 1)")
    heading_reason = ""
    if (
        base_headings.shape != (recorded_steps,)
        or desired_headings.shape != (recorded_steps,)
        or heading_valid.shape != (recorded_steps,)
    ):
        heading_reason = "heading arrays/validity mask have unexpected shapes"
    elif not np.all(np.isfinite(base_headings)) or not np.all(np.isfinite(desired_headings)):
        heading_reason = "heading arrays contain non-finite samples"
    settle_steps = int(cell["settle_steps"])
    if recorded_steps <= settle_steps + 1:
        reason = (
            "episode terminated before two post-settle samples" if terminated else "measurement window is too short"
        )
        return _invalid_cell_row(cell, reason, "failed" if terminated else "short")

    start, stop, complete_cycles, phase_trimmed = _measurement_bounds(common_phase, settle_steps, recorded_steps)
    if stop - start < 2:
        start, stop = settle_steps, recorded_steps
    selected = slice(start, stop)
    selected_positions = positions[selected, :2].copy()
    if start > 0:
        selected_positions[0] = positions[start - 1, :2]
    selected_heading = None if heading_reason else base_headings[selected]
    selected_desired_heading = None if heading_reason else desired_headings[selected]
    if not heading_reason and not bool(np.all(heading_valid[selected])):
        heading_reason = "one or more measurement-window heading samples lack authentic pre-reset state"
        selected_heading = None
        selected_desired_heading = None
    velocity_result = _metrics.velocity_metrics(
        commands[selected],
        velocities[selected],
        selected_positions,
        step_dt,
        base_headings_rad=selected_heading,
        desired_headings_rad=selected_desired_heading,
        terminated=terminated,
        full_commands=commands,
        full_body_velocities=velocities,
        measurement_start_index=start,
        config=study["evaluation_config"],
    )
    command_sign = 1.0 if expected_velocity > 0.0 else -1.0
    directed_progress = command_sign * float(positions[stop - 1, 0] - positions[start - 1, 0])
    velocity_result["signed_directed_progress_m"] = directed_progress
    requested_distance = float(np.sum(np.abs(commands[selected, 0])) * step_dt)
    velocity_result["progress_per_commanded_distance"] = (
        None if requested_distance <= 0.0 else directed_progress / requested_distance
    )
    velocity_result["lost_progress_m"] = max(0.0, requested_distance - directed_progress)
    vertical_force = None if ground_force_reason else np.maximum(foot_normal_forces[selected, :, 2], 0.0)
    gait_result: dict[str, Any] = {}
    contacts = None
    gait_reason = ""
    ground_filter_reason = _ground_filter_requirement_reason(
        study["evaluation_config"],
        ground_filter_declared=ground_filter_declared,
        selected_ground_filtered=ground_filtered[selected],
    )
    if ground_force_reason:
        gait_reason = ground_force_reason
    elif recording_manifest_reason:
        gait_reason = recording_manifest_reason
    elif ground_filter_reason:
        gait_reason = ground_filter_reason
    elif study["evaluation_config"]["contact"]["mode"] == "body_weight" and (
        not math.isfinite(robot_mass_kg) or robot_mass_kg <= 0.0
    ):
        gait_reason = "robot mass is unavailable for body-weight-normalized contact thresholds"
    else:
        try:
            _, gait_library = canonical_training_gaits()
            gait_result, contacts = _metrics.gait_metrics(
                vertical_force,
                common_phase[selected],
                expected_offsets,
                duty_factors[selected],
                step_dt,
                study["evaluation_config"],
                robot_mass_kg=robot_mass_kg,
                gait_library=gait_library,
            )
            gait_result["complete_cycles"] = complete_cycles
            if archived_contact_on_n is not None and archived_contact_off_n is not None:
                if not math.isclose(
                    archived_contact_on_n,
                    float(gait_result["contact_threshold_on_n"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ) or not math.isclose(
                    archived_contact_off_n,
                    float(gait_result["contact_threshold_off_n"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise ValueError("Archived contact thresholds do not match the immutable evaluation config.")
                gait_result["contact_thresholds_archive_verified"] = True
            minimum_cycles = int(study["evaluation_config"]["gait"]["minimum_complete_cycles"])
            minimum_events = int(study["evaluation_config"]["gait"]["minimum_events_per_foot"])
            coverage_reasons = []
            if complete_cycles < minimum_cycles:
                coverage_reasons.append(f"complete_cycles<{minimum_cycles}")
            if min(gait_result["touchdown_events_per_foot"]) < minimum_events:
                coverage_reasons.append(f"touchdown_events_per_foot<{minimum_events}")
            if min(gait_result["liftoff_events_per_foot"]) < minimum_events:
                coverage_reasons.append(f"liftoff_events_per_foot<{minimum_events}")
            gait_result["coverage_valid"] = not coverage_reasons
            gait_result["coverage_reason"] = "; ".join(coverage_reasons)
        except (TypeError, ValueError) as exc:
            gait_reason = str(exc)
            gait_result = {}
            contacts = None
    load_result: dict[str, Any] = {}
    load_reason = effort_provenance_reason
    raw_load_reason = ""
    grf_load_reason = "" if contacts is not None else gait_reason or "contact classification is unavailable"
    normalized_limits = None if effort_provenance_reason else configured_limits
    selected_clamped = None
    if (
        action_clamped is not None
        and action_clamped.shape == torques.shape
        and np.all(np.isfinite(action_clamped[selected]))
    ):
        selected_clamped = action_clamped[selected].astype(bool)
    selected_soft_utilization = (
        soft_utilization[selected] if soft_utilization is not None and soft_utilization.shape == torques.shape else None
    )
    try:
        load_result = _metrics.load_metrics(
            torques[selected],
            powers[selected],
            normalized_limits,
            vertical_force if contacts is not None else None,
            contacts,
            step_dt,
            complete_cycles=complete_cycles,
            directed_progress_m=directed_progress,
            soft_limit_utilization=selected_soft_utilization,
            action_clamped=selected_clamped,
            config=study["evaluation_config"],
        )
        if action_clamp_fraction is not None and np.any(np.isfinite(action_clamp_fraction[selected])):
            load_result["action_clamp_fraction"] = float(np.nanmean(action_clamp_fraction[selected]))
        for index_key, name_key in (
            ("worst_normalized_torque_joint_index", "worst_normalized_torque_joint_name"),
            ("worst_absolute_work_joint_index", "worst_absolute_work_joint_name"),
        ):
            joint_index = load_result.get(index_key)
            if joint_index is not None and len(joint_names) == 12 and 0 <= int(joint_index) < len(joint_names):
                load_result[name_key] = joint_names[int(joint_index)]
        if not bool(load_result.get("normalized_effort_available")):
            load_reason = effort_provenance_reason or str(
                load_result.get("normalized_effort_unavailable_reason", "normalized effort is unavailable")
            )
    except (TypeError, ValueError) as exc:
        raw_load_reason = str(exc)
        load_reason = load_reason or raw_load_reason

    row = _invalid_cell_row(cell, "", "valid")
    planar_error = velocities[selected, :2] - commands[selected, :2]
    planar_rmse = float(np.sqrt(np.mean(np.sum(np.square(planar_error), axis=1))))
    tracking_qualification = _tracking_qualification(
        study["evaluation_config"],
        expected_velocity_mps=expected_velocity,
        planar_rmse_mps=planar_rmse,
        vx_relative_rmse=float(velocity_result["vx_relative_rmse"]),
        yaw_rmse_radps=float(velocity_result["yaw_rmse_radps"]),
    )
    row.update(
        {
            "reason": "",
            "sample_count": stop - start,
            "recorded_sample_count": recorded_steps,
            "expected_sample_count": expected_steps,
            "measurement_start_s": start * step_dt,
            "measurement_stop_s": stop * step_dt,
            "measurement_duration_s": (stop - start) * step_dt,
            "complete_cycles": complete_cycles,
            "phase_boundary_trimmed": phase_trimmed,
            "gait_period_s": _period_s({"gait_periods": np.asarray(step_dt / np.diff(common_phase))}, cell_dir)
            if recorded_steps > 1 and np.all(np.diff(common_phase) > 0.0)
            else None,
            "terminated": terminated,
            "termination_time_s": recorded_steps * step_dt if terminated else None,
            "velocity_metric_valid": True,
            "velocity_metric_reason": "",
            "heading_metric_valid": not heading_reason,
            "heading_metric_reason": heading_reason,
            "recording_manifest_valid": not recording_manifest_reason,
            "recording_manifest_reason": recording_manifest_reason,
            "gait_metric_valid": bool(gait_result) and bool(gait_result.get("coverage_valid")),
            "gait_metric_reason": gait_reason or str(gait_result.get("coverage_reason", "")),
            "raw_load_metric_valid": bool(load_result.get("raw_actuator_load_available")),
            "raw_load_metric_reason": raw_load_reason,
            "grf_load_metric_valid": bool(load_result.get("grf_load_available")),
            "grf_load_metric_reason": grf_load_reason,
            "normalized_load_metric_valid": bool(load_result)
            and bool(load_result.get("normalized_effort_available"))
            and not effort_provenance_reason,
            "load_metric_valid": bool(load_result)
            and bool(load_result.get("normalized_effort_available"))
            and not effort_provenance_reason,
            "load_metric_reason": load_reason,
            "effort_limit_source": (
                "mixed_configured_articulation_actuator_sources"
                if len(effort_sources) == 12 and len(set(effort_sources)) > 1
                else effort_sources[0]
                if len(effort_sources) == 12
                else None
            ),
            "effort_limit_fallback": effort_fallback,
            "effort_limit_min_nm": (float(np.min(configured_limits)) if configured_limits.shape == (12,) else None),
            "effort_limit_max_nm": (float(np.max(configured_limits)) if configured_limits.shape == (12,) else None),
            "effort_limit_source_by_joint": json.dumps(effort_sources, separators=(",", ":")),
            "effort_limit_provenance_sha256": (
                effort_provenance.get("source", {}).get("sha256") if isinstance(effort_provenance, dict) else None
            ),
            "effort_limit_provenance_files": json.dumps(
                effort_provenance.get("source", {}).get("files", {}) if isinstance(effort_provenance, dict) else {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "robot_mass_kg": robot_mass_kg if math.isfinite(robot_mass_kg) else None,
            "command_x_mean_mps": float(np.mean(commands[selected, 0])),
            "measured_x_mean_mps": float(np.mean(velocities[selected, 0])),
            "tracking_rmse_mps": planar_rmse,
            "yaw_tracking_rmse_radps": velocity_result["yaw_rmse_radps"],
            **tracking_qualification,
            "signed_directed_progress_m": directed_progress,
            "integrated_body_x_progress_m": float(np.sum(command_sign * velocities[selected, 0]) * step_dt),
            "directed_progress_definition": (
                "commanded-sign world-x displacement across the same sample intervals as effort integration"
            ),
            "positive_directed_progress": directed_progress > 0.0,
        }
    )
    row["position_vs_integrated_progress_difference_m"] = (
        row["signed_directed_progress_m"] - row["integrated_body_x_progress_m"]
    )
    for prefix, result in (("velocity", velocity_result), ("gait", gait_result), ("load", load_result)):
        for key, value in result.items():
            row[f"{prefix}_{key}"] = _json_cell_value(value)

    # Preserve the original front/hind and total fields for downstream readers.
    legacy_load_mappings = {
        "torque_squared": ("torque_squared_per_leg", "raw_torque_squared"),
        "normalized_torque_utilization": ("normalized_torque_squared_per_leg", "normalized_torque_squared"),
        "absolute_work": ("absolute_work_per_leg", "absolute_work"),
        "vertical_grf_impulse": ("vertical_grf_impulse_per_leg", "vertical_grf_impulse"),
    }
    duration_s = float(row["measurement_duration_s"])
    for legacy_name, (per_leg_key, rate_prefix) in legacy_load_mappings.items():
        per_leg = load_result.get(per_leg_key)
        if per_leg is None:
            for suffix in (
                "front_integral",
                "hind_integral",
                "total_integral",
                "signed_imbalance_percent",
                "abs_imbalance_percent",
                "total_per_s",
                "total_per_directed_m",
            ):
                row[f"{legacy_name}_{suffix}"] = None
            continue
        per_leg_array = np.asarray(per_leg, dtype=float)
        front = float(np.sum(per_leg_array[:2]))
        hind = float(np.sum(per_leg_array[2:]))
        total = front + hind
        imbalance = pair_imbalance(front, hind)
        row.update(
            {
                f"{legacy_name}_front_integral": front,
                f"{legacy_name}_hind_integral": hind,
                f"{legacy_name}_total_integral": total,
                f"{legacy_name}_signed_imbalance_percent": imbalance,
                f"{legacy_name}_abs_imbalance_percent": None if imbalance is None else abs(imbalance),
                f"{legacy_name}_total_per_s": load_result.get(f"{rate_prefix}_per_s", total / duration_s),
                f"{legacy_name}_total_per_directed_m": load_result.get(f"{rate_prefix}_per_directed_m"),
                f"{legacy_name}_total_per_cycle": load_result.get(f"{rate_prefix}_per_cycle"),
            }
        )

    # Keep publication tables usable without parsing JSON array cells.  These
    # columns also make the scientific domain of each per-leg load explicit.
    per_leg_load_columns = {
        "torque_squared_per_leg": ("raw_torque_squared", "n2m2s"),
        "normalized_torque_squared_per_leg": ("normalized_torque_squared", "s"),
        "absolute_work_per_leg": ("absolute_work", "j"),
        "vertical_grf_impulse_per_leg": ("vertical_grf_impulse", "ns"),
    }
    for source_key, (label, unit) in per_leg_load_columns.items():
        values = load_result.get(source_key)
        for leg_index, leg_name in enumerate(_metrics.LEG_NAMES):
            row[f"load_{label}_{leg_name.upper()}_{unit}"] = None if values is None else float(values[leg_index])

    success_result = _metrics.success_metrics(
        velocity_result,
        gait_result,
        planned_family=str(cell["family"]),
        terminated=terminated,
        config=study["evaluation_config"],
    )
    for key, value in success_result.items():
        row[key] = _json_cell_value(value)
    gait_domain_valid = bool(gait_result) and bool(gait_result.get("coverage_valid"))
    row["velocity_success_domain_valid"] = True
    row["gait_success_domain_valid"] = gait_domain_valid
    row["joint_success_domain_valid"] = gait_domain_valid
    if not gait_domain_valid:
        for key in (
            "gait_only_success",
            "joint_velocity_and_gait_success",
            "joint_success",
            "success",
            "gait_only_success_failed_checks",
            "success_failed_checks",
            "success_gait_agreement",
            "success_correct_family",
            "success_complete_cycles",
            "success_sufficient_events",
        ):
            row[key] = None
    classified_row = str(gait_result.get("classified_row", "unclassified"))
    classified_family = str(gait_result.get("classified_family", "unclassified"))
    row["gait_classification_available"] = gait_domain_valid
    row["gait_row_correct"] = classified_row == str(cell["gait_name"]) if gait_domain_valid else None
    row["gait_family_correct"] = classified_family == str(cell["family"]) if gait_domain_valid else None
    if terminated:
        row["status"] = "terminated"
        row["reason"] = "episode terminated; available post-settle domains were retained"
    elif directed_progress <= 0.0:
        row["status"] = "nonpositive_progress"
        row["reason"] = "commanded-direction progress is nonpositive; per-distance metrics are N/A"

    joint_rows: list[dict[str, Any]] = []
    joint_arrays = {
        "normalized_torque_squared_exposure": load_result.get("joint_normalized_torque_squared"),
        "absolute_work_j": load_result.get("joint_absolute_work_j"),
        "utilization_p95": load_result.get("joint_utilization_p95"),
        "utilization_p99": load_result.get("joint_utilization_p99"),
        "saturation_fraction": load_result.get("joint_saturation_fraction"),
        "soft_limit_utilization_p95": load_result.get("joint_soft_limit_utilization_p95"),
        "soft_limit_utilization_p99": load_result.get("joint_soft_limit_utilization_p99"),
        "action_clamp_fraction": load_result.get("joint_action_clamp_fraction"),
    }
    for index in range(12):
        name = joint_names[index] if len(joint_names) == 12 else None
        joint_row = {
            "cell_id": cell["id"],
            "joint_index": index,
            "joint_name": name,
            "leg": _metrics.LEG_NAMES[index // 3],
            "motor_role": _metrics.JOINT_NAMES_PER_LEG[index % 3],
            "configured_effort_limit_nm": (
                float(configured_limits[index]) if configured_limits.shape == (12,) else None
            ),
            "effort_limit_source": effort_sources[index] if len(effort_sources) == 12 else None,
        }
        for key, values in joint_arrays.items():
            joint_row[key] = None if values is None else values[index]
        joint_rows.append(joint_row)
    foot_rows = []
    per_foot_gait = {str(metric.get("foot")): metric for metric in gait_result.get("per_foot_metrics", [])}
    for index, name in enumerate(_metrics.LEG_NAMES):
        foot_row = {
            "cell_id": cell["id"],
            "foot": name,
            "contact_mean_force_n": load_result.get(f"{name}_contact_mean_force_n"),
            "vertical_impulse_ns": (
                None
                if load_result.get("vertical_grf_impulse_per_leg") is None
                else load_result["vertical_grf_impulse_per_leg"][index]
            ),
            "force_p95_n": load_result.get(f"{name}_force_p95_n"),
            "force_p99_n": load_result.get(f"{name}_force_p99_n"),
            "force_max_n": load_result.get(f"{name}_force_max_n"),
            "impact_event_count": load_result.get(f"{name}_impact_event_count"),
            "impact_peak_mean_n": load_result.get(f"{name}_impact_peak_mean_n"),
            "impact_peak_p95_n": load_result.get(f"{name}_impact_peak_p95_n"),
            "impact_peak_max_n": load_result.get(f"{name}_impact_peak_max_n"),
            "impact_impulse_mean_ns": load_result.get(f"{name}_impact_impulse_mean_ns"),
            "impact_impulse_p95_ns": load_result.get(f"{name}_impact_impulse_p95_ns"),
            "impact_impulse_max_ns": load_result.get(f"{name}_impact_impulse_max_ns"),
        }
        foot_row.update(per_foot_gait.get(name, {}))
        foot_rows.append(foot_row)
    row["_joint_metric_rows"] = joint_rows
    row["_foot_metric_rows"] = foot_rows
    row["_confusion_row"] = {
        "cell_id": cell["id"],
        "planned_row": cell["gait_name"],
        "planned_family": cell["family"],
        "classified_row": gait_result.get("classified_row", "unclassified"),
        "classified_family": gait_result.get("classified_family", "unclassified"),
        "classification_score_cycles": gait_result.get("classified_score_cycles"),
        "classification_margin_cycles": gait_result.get("classified_margin_cycles"),
        "classification_domain_valid": gait_domain_valid,
        "classification_domain_reason": "" if gait_domain_valid else row["gait_metric_reason"],
    }
    synchronization_rows: list[dict[str, Any]] = []
    for event in ("touchdown", "liftoff"):
        for pair_record in gait_result.get(f"{event}_same_phase_sync_pair_records", []):
            synchronization_rows.append({"cell_id": cell["id"], "event": event, **pair_record})
    row["_synchronization_rows"] = synchronization_rows
    return row


def analyze_cell(study: dict[str, Any], cell: dict[str, Any], study_root: Path) -> dict[str, Any]:
    """Analyze one cell using its immutable recorded method version."""
    if study.get("method_version") == METHOD_VERSION:
        return _analyze_cell_legacy(study, cell, study_root)
    if study.get("method_version") not in {FULL_METHOD_VERSION, LIGHT_METHOD_VERSION}:
        return _invalid_cell_row(cell, f"unsupported method version {study.get('method_version')!r}")
    return _publication_cell(study, cell, study_root)


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(sum(finite) / len(finite))


def _format_number(value: float | None, format_spec: str = ".4f", *, scale: float = 1.0) -> str:
    """Format a covered scalar while preserving unavailable domains as ``N/A``."""
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return format(float(value) * scale, format_spec)


def _format_percent(value: float | None, format_spec: str = ".2f") -> str:
    """Format a covered fraction as a percent, preserving ``N/A`` exactly."""
    formatted = _format_number(value, format_spec, scale=100.0)
    return formatted if formatted == "N/A" else f"{formatted}%"


def _per_leg_load_means(
    rows: Sequence[dict[str, Any]],
    label: str,
    unit: str,
) -> dict[str, float | None]:
    """Return covered per-leg load means from explicit flat cell columns."""
    return {
        leg.upper(): _mean(row.get(f"load_{label}_{leg.upper()}_{unit}") for row in rows) for leg in _metrics.LEG_NAMES
    }


def aggregate_results(
    study: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate cells without allowing signs or gait-family sizes to bias balance."""
    expected_by_group: dict[tuple[str, float | None], int] = defaultdict(int)
    valid_by_group: dict[tuple[str, float | None], list[dict[str, Any]]] = defaultdict(list)
    rows_by_group: dict[tuple[str, float | None], list[dict[str, Any]]] = defaultdict(list)
    publication_method = study.get("method_version") in {FULL_METHOD_VERSION, LIGHT_METHOD_VERSION}
    for cell, row in zip(study["cells"], rows, strict=True):
        keys = ((cell["family"], float(cell["velocity_mps"])), (cell["family"], None))
        for key in keys:
            expected_by_group[key] += 1
            rows_by_group[key].append(row)
            if row.get("load_metric_valid") is True or (not publication_method and row["status"] == "valid"):
                valid_by_group[key].append(row)

    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        velocities: list[float | None] = [float(value) for value in study["velocities_mps"]] + [None]
        for velocity in velocities:
            key = (family, velocity)
            expected = expected_by_group.get(key, 0)
            valid = valid_by_group.get(key, [])
            group_rows = rows_by_group.get(key, [])
            tracking_qualified = [row for row in valid if row.get("tracking_success") is True]
            complete = expected > 0 and len(valid) == expected
            tracking_complete = complete and len(tracking_qualified) == expected
            result: dict[str, Any] = {
                "scope": "family" if velocity is None else "family_velocity",
                "family": family,
                "velocity_mps": velocity,
                "expected_cells": expected,
                "valid_cells": len(valid),
                "coverage_fraction": len(valid) / expected if expected else 0.0,
                "complete": complete,
                "tracking_success_cells": len(tracking_qualified),
                "tracking_success_fraction": len(tracking_qualified) / len(valid) if valid else None,
                "tracking_qualified_complete": tracking_complete,
            }
            for metric in ALL_METRICS:
                domain_key = METRIC_DOMAIN_KEYS[metric]
                metric_valid = (
                    [row for row in group_rows if row.get(domain_key) is True]
                    if publication_method
                    else [row for row in group_rows if row["status"] == "valid"]
                )
                metric_tracking = [row for row in metric_valid if row.get("tracking_success") is True]
                metric_complete = expected > 0 and len(metric_valid) == expected
                metric_tracking_complete = metric_complete and len(metric_tracking) == expected
                result[f"{metric}_valid_cells"] = len(metric_valid)
                result[f"{metric}_coverage_fraction"] = len(metric_valid) / expected if expected else 0.0
                result[f"{metric}_complete"] = metric_complete
                result[f"{metric}_tracking_success_cells"] = len(metric_tracking)
                result[f"{metric}_tracking_qualified_complete"] = metric_tracking_complete
                signed = _mean(row.get(f"{metric}_signed_imbalance_percent") for row in metric_valid)
                absolute = _mean(row.get(f"{metric}_abs_imbalance_percent") for row in metric_valid)
                result[f"{metric}_observed_mean_signed_imbalance_percent"] = signed
                result[f"{metric}_observed_mean_abs_imbalance_percent"] = absolute
                result[f"{metric}_mean_signed_imbalance_percent"] = signed if metric_complete else None
                result[f"{metric}_mean_abs_imbalance_percent"] = absolute if metric_complete else None
                tracking_signed = _mean(row.get(f"{metric}_signed_imbalance_percent") for row in metric_tracking)
                tracking_absolute = _mean(row.get(f"{metric}_abs_imbalance_percent") for row in metric_tracking)
                result[f"{metric}_tracking_qualified_observed_mean_signed_imbalance_percent"] = tracking_signed
                result[f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent"] = tracking_absolute
                result[f"{metric}_tracking_qualified_mean_signed_imbalance_percent"] = (
                    tracking_signed if metric_tracking_complete else None
                )
                result[f"{metric}_tracking_qualified_mean_abs_imbalance_percent"] = (
                    tracking_absolute if metric_tracking_complete else None
                )
                for quantity in ("total_integral", "total_per_s", "total_per_directed_m"):
                    observed_quantity = _mean(row.get(f"{metric}_{quantity}") for row in metric_valid)
                    tracking_quantity = _mean(row.get(f"{metric}_{quantity}") for row in metric_tracking)
                    result[f"{metric}_observed_mean_{quantity}"] = observed_quantity
                    result[f"{metric}_mean_{quantity}"] = observed_quantity if metric_complete else None
                    result[f"{metric}_tracking_qualified_observed_mean_{quantity}"] = tracking_quantity
                    result[f"{metric}_tracking_qualified_mean_{quantity}"] = (
                        tracking_quantity if metric_tracking_complete else None
                    )
            family_rows.append(result)

    family_summary = {row["family"]: row for row in family_rows if row["scope"] == "family"}
    metrics: dict[str, Any] = {}
    for metric in ALL_METRICS:
        family_abs = {
            family: family_summary[family].get(f"{metric}_mean_abs_imbalance_percent") for family in FAMILY_ORDER
        }
        observed_family_abs = {
            family: family_summary[family].get(f"{metric}_observed_mean_abs_imbalance_percent")
            for family in FAMILY_ORDER
        }
        family_signed = {
            family: family_summary[family].get(f"{metric}_mean_signed_imbalance_percent") for family in FAMILY_ORDER
        }
        tracking_family_abs = {
            family: family_summary[family].get(f"{metric}_tracking_qualified_mean_abs_imbalance_percent")
            for family in FAMILY_ORDER
        }
        observed_tracking_family_abs = {
            family: family_summary[family].get(f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent")
            for family in FAMILY_ORDER
        }
        complete = all(family_summary[family].get(f"{metric}_complete") is True for family in FAMILY_ORDER)
        tracking_complete = all(
            family_summary[family].get(f"{metric}_tracking_qualified_complete") is True for family in FAMILY_ORDER
        )
        metric_result = {
            "label": METRIC_LABELS[metric],
            "integral_unit": METRIC_UNITS[metric],
            "domain_key": METRIC_DOMAIN_KEYS[metric],
            "valid_cells": sum(int(family_summary[family].get(f"{metric}_valid_cells", 0)) for family in FAMILY_ORDER),
            "expected_cells": len(rows),
            "coverage_fraction": (
                sum(int(family_summary[family].get(f"{metric}_valid_cells", 0)) for family in FAMILY_ORDER) / len(rows)
                if rows
                else 0.0
            ),
            "family_mean_abs_imbalance_percent": family_abs,
            "family_mean_signed_imbalance_percent": family_signed,
            "family_balanced_mean_abs_imbalance_percent": _mean(family_abs.values()) if complete else None,
            "observed_family_balanced_mean_abs_imbalance_percent": _mean(observed_family_abs.values()),
            "complete": complete,
            "tracking_qualified_family_mean_abs_imbalance_percent": tracking_family_abs,
            "tracking_qualified_family_balanced_mean_abs_imbalance_percent": (
                _mean(tracking_family_abs.values()) if tracking_complete else None
            ),
            "tracking_qualified_observed_family_balanced_mean_abs_imbalance_percent": _mean(
                observed_tracking_family_abs.values()
            ),
            "tracking_qualified_complete": tracking_complete,
        }
        for quantity in ("total_integral", "total_per_s", "total_per_directed_m"):
            family_quantity = {
                family: family_summary[family].get(f"{metric}_mean_{quantity}") for family in FAMILY_ORDER
            }
            observed_family_quantity = {
                family: family_summary[family].get(f"{metric}_observed_mean_{quantity}") for family in FAMILY_ORDER
            }
            tracking_family_quantity = {
                family: family_summary[family].get(f"{metric}_tracking_qualified_mean_{quantity}")
                for family in FAMILY_ORDER
            }
            observed_tracking_family_quantity = {
                family: family_summary[family].get(f"{metric}_tracking_qualified_observed_mean_{quantity}")
                for family in FAMILY_ORDER
            }
            metric_result[f"family_mean_{quantity}"] = family_quantity
            metric_result[f"family_balanced_mean_{quantity}"] = _mean(family_quantity.values()) if complete else None
            metric_result[f"observed_family_balanced_mean_{quantity}"] = _mean(observed_family_quantity.values())
            metric_result[f"tracking_qualified_family_mean_{quantity}"] = tracking_family_quantity
            metric_result[f"tracking_qualified_family_balanced_mean_{quantity}"] = (
                _mean(tracking_family_quantity.values()) if tracking_complete else None
            )
            metric_result[f"tracking_qualified_observed_family_balanced_mean_{quantity}"] = _mean(
                observed_tracking_family_quantity.values()
            )
        metrics[metric] = metric_result
    if publication_method:
        valid_count = sum(
            row.get("velocity_metric_valid") is True
            and row.get("gait_metric_valid") is True
            and row.get("load_metric_valid") is True
            for row in rows
        )
        tracking_success_count = sum(
            row.get("tracking_success") is True for row in rows if row.get("velocity_metric_valid") is True
        )
    else:
        valid_count = sum(row["status"] == "valid" for row in rows)
        tracking_success_count = sum(row.get("tracking_success") is True for row in rows if row["status"] == "valid")
    status_counts = {
        status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})
    }
    success_coverage: dict[str, Any] = {}
    for label, value_key, domain_key in (
        ("velocity_only", "velocity_only_success", "velocity_success_domain_valid"),
        ("gait_only", "gait_only_success", "gait_success_domain_valid"),
        ("joint_velocity_and_gait", "joint_velocity_and_gait_success", "joint_success_domain_valid"),
    ):
        domain = [row for row in rows if row.get(domain_key) is True]
        succeeded = sum(row.get(value_key) is True for row in domain)
        success_coverage[f"{label}_success_cells"] = succeeded
        success_coverage[f"{label}_success_domain_cells"] = len(domain)
        success_coverage[f"{label}_success_fraction_of_domain"] = succeeded / len(domain) if domain else None
        success_coverage[f"{label}_success_coverage_fraction"] = len(domain) / len(rows) if rows else 0.0
    classification_domain = [row for row in rows if row.get("gait_classification_available") is True]
    row_correct = sum(row.get("gait_row_correct") is True for row in classification_domain)
    family_correct = sum(row.get("gait_family_correct") is True for row in classification_domain)
    unclassified = sum(row.get("gait_classified_row") == "unclassified" for row in classification_domain)
    classification_coverage = {
        "domain_cells": len(classification_domain),
        "coverage_fraction": len(classification_domain) / len(rows) if rows else 0.0,
        "unclassified_cells": unclassified,
        "row_correct_cells": row_correct,
        "family_correct_cells": family_correct,
        "row_accuracy": row_correct / len(classification_domain) if classification_domain else None,
        "family_accuracy": family_correct / len(classification_domain) if classification_domain else None,
    }
    tracking_config = study.get("evaluation_config", {}).get("tracking", {})
    yaw_tracking_limit = float(tracking_config.get("yaw_rmse_limit_radps", YAW_TRACKING_SUCCESS_THRESHOLD_RADPS))
    vx_relative_limit = tracking_config.get("vx_relative_error_limit")
    overall = {
        "schema_version": SCHEMA_VERSION,
        "method_version": study.get("method_version", METHOD_VERSION),
        "protocol": study.get("protocol", "legacy-full"),
        "coverage": {
            "expected_cells": len(rows),
            "valid_cells": valid_count,
            "failed_or_missing_cells": len(rows) - valid_count,
            "coverage_fraction": valid_count / len(rows) if rows else 0.0,
            "complete": valid_count == len(rows),
            "tracking_success_cells": tracking_success_count,
            "tracking_success_fraction_of_valid": (
                tracking_success_count / sum(row.get("velocity_metric_valid") is True for row in rows)
                if publication_method and any(row.get("velocity_metric_valid") is True for row in rows)
                else tracking_success_count / valid_count
                if valid_count
                else None
            ),
            "status_counts": status_counts,
            "velocity_valid_cells": sum(row.get("velocity_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "heading_valid_cells": sum(row.get("heading_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "gait_valid_cells": sum(row.get("gait_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "load_valid_cells": sum(row.get("load_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "raw_load_valid_cells": sum(row.get("raw_load_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "grf_load_valid_cells": sum(row.get("grf_load_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "normalized_load_valid_cells": sum(row.get("normalized_load_metric_valid") is True for row in rows)
            if publication_method
            else valid_count,
            "joint_success_cells": success_coverage.get("joint_velocity_and_gait_success_cells")
            if publication_method
            else None,
            **success_coverage,
        },
        "gait_classification": classification_coverage,
        "aggregation": study["aggregation"],
        "tracking_qualification": {
            "planar": "tracking_rmse_mps <= 0.05 + 0.25 * abs(velocity_mps)",
            "yaw": f"yaw_tracking_rmse_radps <= {yaw_tracking_limit:g}",
            "vx_relative": (
                None if vx_relative_limit is None else f"velocity_vx_relative_rmse <= {float(vx_relative_limit):g}"
            ),
        },
        "metrics": metrics,
    }
    return family_rows, overall


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_optional_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Replace an optional CSV, removing a stale artifact when there are no rows."""
    if rows:
        _write_csv(path, rows)
    else:
        path.unlink(missing_ok=True)


def _save_family_figure(path: Path, family: str, family_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in family_rows if row["family"] == family and row["scope"] == "family_velocity"]
    velocities = [float(row["velocity_mps"]) for row in selected]
    x_positions = np.arange(len(velocities), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        values = [row[f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent"] for row in selected]
        heights = [np.nan if value is None else value for value in values]
        axis.bar(x_positions + (metric_index - 1) * width, heights, width, label=METRIC_LABELS[metric])
    axis.set_xticks(x_positions, [f"{velocity:+g}" for velocity in velocities])
    axis.set_xlabel("Commanded x velocity [m/s]")
    axis.set_ylabel("Tracking-qualified mean absolute front/hind imbalance [%]")
    axis.set_title(f"{family.replace('_', ' ').title()} tracking-qualified leg usage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, format="svg")
    plt.close(figure)


def _save_overall_figure(path: Path, overall: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [*FAMILY_ORDER, "overall"]
    x_positions = np.arange(len(labels), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        summary = overall["metrics"][metric]
        values = [summary["tracking_qualified_family_mean_abs_imbalance_percent"][family] for family in FAMILY_ORDER]
        values.append(summary["tracking_qualified_family_balanced_mean_abs_imbalance_percent"])
        heights = [np.nan if value is None else value for value in values]
        axis.bar(x_positions + (metric_index - 1) * width, heights, width, label=METRIC_LABELS[metric])
    axis.set_xticks(x_positions, [label.replace("_", " ").title() for label in labels])
    axis.set_ylabel("Tracking-qualified mean absolute front/hind imbalance [%]")
    axis.set_title("Tracking-qualified family-balanced leg usage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, format="svg")
    plt.close(figure)


def _save_coverage_figure(path: Path, family_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {row["family"]: row for row in family_rows if row["scope"] == "family"}
    coverage = [100.0 * summary[family]["coverage_fraction"] for family in FAMILY_ORDER]
    tracking_coverage = [
        100.0 * summary[family]["tracking_success_cells"] / summary[family]["expected_cells"]
        if summary[family]["expected_cells"]
        else 0.0
        for family in FAMILY_ORDER
    ]
    figure, axis = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    x_positions = np.arange(len(FAMILY_ORDER), dtype=float)
    width = 0.38
    valid_bars = axis.bar(x_positions - width / 2, coverage, width, label="Valid rollout")
    tracking_bars = axis.bar(x_positions + width / 2, tracking_coverage, width, label="Tracking qualified")
    axis.bar_label(valid_bars, labels=[f"{value:.0f}%" for value in coverage], fontsize="small")
    axis.bar_label(tracking_bars, labels=[f"{value:.0f}%" for value in tracking_coverage], fontsize="small")
    axis.set_xticks(x_positions, [family.replace("_", " ").title() for family in FAMILY_ORDER])
    axis.set_ylim(0.0, 110.0)
    axis.set_ylabel("Valid cells [%]")
    axis.set_title("Leg-usage grid coverage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, format="svg")
    plt.close(figure)


def _training_reward(study: dict[str, Any]) -> tuple[float | None, str | None]:
    """Read the final training mean reward when TensorBoard is available."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None
    run_dir = Path(str(study["checkpoint"]["path"])).parent
    event_files = sorted(run_dir.glob("events.out.tfevents*"), key=lambda path: path.stat().st_mtime)
    if not event_files:
        return None, None
    try:
        accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        for tag in ("Train/mean_reward", "Metrics/mean_reward", "rollout/ep_rew_mean"):
            if tag in tags:
                samples = accumulator.Scalars(tag)
                if samples:
                    return float(samples[-1].value), tag
    except (OSError, RuntimeError, ValueError):
        return None, None
    return None, None


def _save_screening_figure(
    path: Path,
    study: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    overall: dict[str, Any],
) -> None:
    """Save the compact six-panel publication-screening report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reward, reward_tag = _training_reward(study)
    valid_velocity = [row for row in rows if row.get("velocity_metric_valid") is True]
    valid_heading = [row for row in rows if row.get("heading_metric_valid") is True]
    valid_gait = [row for row in rows if row.get("gait_metric_valid") is True]
    valid_raw_load = [row for row in rows if row.get("raw_load_metric_valid") is True]
    valid_grf_load = [row for row in rows if row.get("grf_load_metric_valid") is True]
    valid_normalized_load = [row for row in rows if row.get("normalized_load_metric_valid") is True]
    vx_relative_rmse = _mean(row.get("velocity_vx_relative_rmse") for row in valid_velocity)
    yaw_rmse = _mean(row.get("velocity_yaw_rmse_radps") for row in valid_velocity)
    gait_agreement = _mean(row.get("gait_agreement_boundary_excluded") for row in valid_gait)
    correct_family_count = sum(row.get("gait_classified_family") == row["family"] for row in valid_gait)
    heading_rmse = _mean(row.get("velocity_heading_rmse_rad") for row in valid_heading)
    lateral_p95 = _mean(row.get("velocity_lateral_position_p95_m") for row in valid_velocity)
    progress_ratio = _mean(row.get("velocity_progress_per_commanded_distance") for row in valid_velocity)
    work_max_share = _mean(row.get("load_absolute_work_max_share") for row in valid_raw_load)
    work_front_hind_abs = _mean(row.get("load_absolute_work_front_hind_abs") for row in valid_raw_load)
    worst_leg_counts = Counter(
        str(row["load_absolute_work_worst_leg"])
        for row in valid_raw_load
        if row.get("load_absolute_work_worst_leg") is not None
    )
    modal_worst_leg = worst_leg_counts.most_common(1)[0][0] if worst_leg_counts else "N/A"
    grf_max_share = _mean(row.get("load_vertical_grf_impulse_max_share") for row in valid_grf_load)
    normalized_max_share = _mean(row.get("load_normalized_torque_squared_max_share") for row in valid_normalized_load)
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 8.3), layout="constrained")
    panels = [
        (
            "Training reward",
            [
                "N/A (no scalar event found)" if reward is None else f"final: {reward:.3f}",
                "" if reward_tag is None else reward_tag,
            ],
        ),
        (
            "Velocity tracking",
            [
                f"vx relative RMSE: {_format_number(vx_relative_rmse, '.3f')}",
                f"yaw RMSE: {_format_number(yaw_rmse, '.3f')} rad/s",
                f"valid: {len(valid_velocity)}/{len(rows)}",
            ],
        ),
        (
            "Gait fidelity",
            [
                f"agreement: {_format_percent(gait_agreement, '.1f')}",
                f"family correct: {correct_family_count}/{len(valid_gait)}",
                f"valid: {len(valid_gait)}/{len(rows)}",
            ],
        ),
        (
            "Heading and path",
            [
                f"heading RMSE: {_format_number(heading_rmse, '.3f')} rad",
                f"heading valid: {len(valid_heading)}/{len(rows)}",
                f"lateral p95: {_format_number(lateral_p95, '.3f')} m",
                f"progress ratio: {_format_number(progress_ratio, '.3f')}",
            ],
        ),
        (
            "Load concentration",
            [
                f"work F/H imbalance: {_format_percent(work_front_hind_abs, '.1f')}",
                f"work max share: {_format_percent(work_max_share, '.1f')}",
                f"modal worst leg: {modal_worst_leg}",
                f"GRF max share: {_format_percent(grf_max_share, '.1f')}",
                f"norm max share: {_format_percent(normalized_max_share, '.1f')}",
                f"domains raw/grf/norm: {len(valid_raw_load)}/{len(valid_grf_load)}/"
                f"{len(valid_normalized_load)} of {len(rows)}",
            ],
        ),
        (
            "Joint success",
            [
                f"pass: {overall['coverage'].get('joint_success_cells', 0)}/"
                f"{overall['coverage'].get('joint_velocity_and_gait_success_domain_cells', 0)}",
                f"domain: {overall['coverage'].get('joint_velocity_and_gait_success_domain_cells', 0)}/"
                f"{len(rows)} planned",
                f"protocol: {study.get('protocol', 'legacy-full')}",
                f"method: {study.get('method_version')}",
            ],
        ),
    ]
    for axis, (title, lines) in zip(axes.flat, panels, strict=True):
        axis.axis("off")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(0.02, 0.82, "\n".join(lines), va="top", family="monospace", fontsize=10)
    figure.suptitle(
        f"Leg-usage publication screen — {study['robot']} — {overall['coverage']['expected_cells']} cells",
        fontweight="bold",
    )
    figure.savefig(path, format="svg")
    plt.close(figure)


def _write_screening_report(
    path: Path,
    study: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    overall: dict[str, Any],
) -> None:
    """Write a compact human-readable publication-screening report."""
    reward, reward_tag = _training_reward(study)
    coverage = overall["coverage"]
    velocity_rows = [row for row in rows if row.get("velocity_metric_valid") is True]
    heading_rows = [row for row in rows if row.get("heading_metric_valid") is True]
    gait_rows = [row for row in rows if row.get("gait_metric_valid") is True]
    raw_load_rows = [row for row in rows if row.get("raw_load_metric_valid") is True]
    grf_load_rows = [row for row in rows if row.get("grf_load_metric_valid") is True]
    normalized_load_rows = [row for row in rows if row.get("normalized_load_metric_valid") is True]
    reward_text = "N/A" if reward is None else f"{reward:.4g} (`{reward_tag}`)"
    vx_relative_rmse = _mean(row.get("velocity_vx_relative_rmse") for row in velocity_rows)
    yaw_rmse = _mean(row.get("velocity_yaw_rmse_radps") for row in velocity_rows)
    gait_agreement = _mean(row.get("gait_agreement_boundary_excluded") for row in gait_rows)
    heading_rmse = _mean(row.get("velocity_heading_rmse_rad") for row in heading_rows)
    work_max_share = _mean(row.get("load_absolute_work_max_share") for row in raw_load_rows)
    work_front_hind_abs = _mean(row.get("load_absolute_work_front_hind_abs") for row in raw_load_rows)
    grf_max_share = _mean(row.get("load_vertical_grf_impulse_max_share") for row in grf_load_rows)
    normalized_max_share = _mean(row.get("load_normalized_torque_squared_max_share") for row in normalized_load_rows)
    worst_leg_counts = Counter(
        str(row["load_absolute_work_worst_leg"])
        for row in raw_load_rows
        if row.get("load_absolute_work_worst_leg") is not None
    )
    modal_worst_leg = worst_leg_counts.most_common(1)[0][0] if worst_leg_counts else "N/A"
    raw_torque_per_leg = _per_leg_load_means(raw_load_rows, "raw_torque_squared", "n2m2s")
    work_per_leg = _per_leg_load_means(raw_load_rows, "absolute_work", "j")
    normalized_per_leg = _per_leg_load_means(normalized_load_rows, "normalized_torque_squared", "s")
    grf_per_leg = _per_leg_load_means(grf_load_rows, "vertical_grf_impulse", "ns")
    lines = [
        "# Publication screening report",
        "",
        f"Protocol `{study.get('protocol', 'legacy-full')}` / `{study.get('method_version')}`; "
        f"checkpoint `{study['checkpoint']['sha256']}`.",
        "",
        "| Domain | Result | Coverage |",
        "|---|---:|---:|",
        f"| Training mean reward | {reward_text} | event-file scalar |",
        f"| vx relative RMSE | {_format_number(vx_relative_rmse)} | {len(velocity_rows)}/{len(rows)} |",
        f"| Yaw RMSE [rad/s] | {_format_number(yaw_rmse)} | {len(velocity_rows)}/{len(rows)} |",
        f"| Boundary-excluded gait agreement | {_format_percent(gait_agreement)} | {len(gait_rows)}/{len(rows)} |",
        f"| Heading RMSE [rad] | {_format_number(heading_rmse)} | {len(heading_rows)}/{len(rows)} |",
        f"| Absolute-work front/hind imbalance | {_format_percent(work_front_hind_abs)} | "
        f"{len(raw_load_rows)}/{len(rows)} |",
        f"| Absolute-work worst-leg share | {_format_percent(work_max_share)} | {len(raw_load_rows)}/{len(rows)} |",
        f"| Modal absolute-work worst leg | {modal_worst_leg} | {len(raw_load_rows)}/{len(rows)} |",
        f"| Vertical-GRF worst-leg share | {_format_percent(grf_max_share)} | {len(grf_load_rows)}/{len(rows)} |",
        f"| Normalized-torque worst-leg share | {_format_percent(normalized_max_share)} | "
        f"{len(normalized_load_rows)}/{len(rows)} |",
        f"| Velocity-only success | {coverage.get('velocity_only_success_cells', 0)}/"
        f"{coverage.get('velocity_only_success_domain_cells', 0)} | valid velocity domain |",
        f"| Gait-only success | {coverage.get('gait_only_success_cells', 0)}/"
        f"{coverage.get('gait_only_success_domain_cells', 0)} | valid gait domain |",
        f"| Joint velocity+gait success | {coverage.get('joint_success_cells', 0)}/"
        f"{coverage.get('joint_velocity_and_gait_success_domain_cells', 0)} | "
        f"domain {coverage.get('joint_velocity_and_gait_success_domain_cells', 0)}/{len(rows)} planned |",
        "",
        "## Per-leg load means",
        "",
        "Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the "
        "strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.",
        "",
        "| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | "
        "Vertical GRF impulse [Ns] |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {leg} | {_format_number(raw_torque_per_leg[leg], '.6g')} | "
            f"{_format_number(work_per_leg[leg], '.6g')} | "
            f"{_format_number(normalized_per_leg[leg], '.6g')} | "
            f"{_format_number(grf_per_leg[leg], '.6g')} |"
            for leg in ("FL", "FR", "RL", "RR")
        ),
        "",
        "Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results "
        "are in `foot_metrics.csv` and `joint_metrics.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    path: Path,
    study: dict[str, Any],
    family_rows: Sequence[dict[str, Any]],
    overall: dict[str, Any],
) -> None:
    coverage = overall["coverage"]
    summary = {row["family"]: row for row in family_rows if row["scope"] == "family"}
    lines = [
        "# Leg usage grid analysis",
        "",
        f"- Robot: `{study['robot']}`",
        f"- Protocol: `{study.get('protocol', 'legacy-full')}` (`{study.get('method_version', METHOD_VERSION)}`)",
        f"- Checkpoint: `{study['checkpoint']['path']}` (SHA-256 `{study['checkpoint']['sha256']}`)",
        f"- Grid: {len(study['cells'])} planned protocol cells "
        f"({len(study['gaits'])} available gait rows, {len(study['velocities_mps'])} available velocities)",
        f"- Window: {study['settle_s']:g} s settling, then {study['measure_s']:g} s measurement",
        f"- Coverage: {coverage['valid_cells']}/{coverage['expected_cells']} valid cells "
        f"({100.0 * coverage['coverage_fraction']:.1f}%)",
        f"- Domain coverage: velocity {coverage.get('velocity_valid_cells', coverage['valid_cells'])}/"
        f"{coverage['expected_cells']}, heading {coverage.get('heading_valid_cells', coverage['valid_cells'])}/"
        f"{coverage['expected_cells']}, gait {coverage.get('gait_valid_cells', coverage['valid_cells'])}/"
        f"{coverage['expected_cells']}, raw load {coverage.get('raw_load_valid_cells', coverage['valid_cells'])}/"
        f"{coverage['expected_cells']}, GRF load {coverage.get('grf_load_valid_cells', coverage['valid_cells'])}/"
        f"{coverage['expected_cells']}, normalized load "
        f"{coverage.get('normalized_load_valid_cells', coverage['valid_cells'])}/{coverage['expected_cells']}",
        f"- Tracking quality: {coverage['tracking_success_cells']}/"
        f"{coverage.get('velocity_valid_cells', coverage['valid_cells'])} velocity-domain cells pass both "
        "`planar RMSE <= 0.05 + 0.25 * abs(vx)` [m/s] and `yaw RMSE <= 0.05` [rad/s]",
        f"- Analysis provenance: `analysis_provenance.json` "
        f"(record `{overall['analysis_provenance']['record_sha256']}`)",
        "",
        "Primary balance is the mean of per-cell absolute front/hind imbalance. Signed imbalance is retained in "
        "`cell_metrics.csv` and `family_metrics.csv`, but opposite signs never cancel in the primary score. The "
        "overall "
        "score gives each of trot, bound, half-bound, and gallop equal weight.",
        "",
        "## Coverage by family",
        "",
        "| Family | Valid | Expected | Coverage | Tracking pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILY_ORDER:
        row = summary[family]
        lines.append(
            f"| {family.replace('_', ' ')} | {row['valid_cells']} | {row['expected_cells']} | "
            f"{100.0 * row['coverage_fraction']:.1f}% | {row['tracking_success_cells']}/{row['valid_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Family-balanced primary results",
            "",
            "| Metric | All-valid mean absolute imbalance | Tracking-qualified mean | Valid complete | "
            "Tracking complete |",
            "|---|---:|---:|:---:|:---:|",
        ]
    )
    for metric in PRIMARY_METRICS:
        result = overall["metrics"][metric]
        value = result["family_balanced_mean_abs_imbalance_percent"]
        tracking_value = result["tracking_qualified_family_balanced_mean_abs_imbalance_percent"]
        formatted = "N/A" if value is None else f"{value:.3f}%"
        tracking_formatted = "N/A" if tracking_value is None else f"{tracking_value:.3f}%"
        lines.append(
            f"| {METRIC_LABELS[metric]} | {formatted} | {tracking_formatted} | "
            f"{'yes' if result['complete'] else 'no'} | "
            f"{'yes' if result['tracking_qualified_complete'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Tracking-qualified usage rates",
            "",
            "Complete-cycle trimming can produce slightly different measurement durations, so rates are compared "
            "instead of raw totals.",
            "",
            "| Metric | Family-balanced total / s | Family-balanced total / directed m |",
            "|---|---:|---:|",
        ]
    )
    for metric in PRIMARY_METRICS:
        result = overall["metrics"][metric]
        per_s = result["tracking_qualified_family_balanced_mean_total_per_s"]
        per_m = result["tracking_qualified_family_balanced_mean_total_per_directed_m"]
        formatted_per_s = "N/A" if per_s is None else f"{per_s:.6g}"
        formatted_per_m = "N/A" if per_m is None else f"{per_m:.6g}"
        lines.append(f"| {METRIC_LABELS[metric]} | {formatted_per_s} | {formatted_per_m} |")
    lines.extend(
        [
            "",
            "A cell with missing/short data, an episode termination, invalid arrays, or nonpositive "
            "commanded-direction progress remains explicit in `coverage.csv`. Cost-per-distance fields are N/A for "
            "nonpositive progress; the "
            "cell's time-normalized balance metrics remain available when its rollout is otherwise valid.",
            "Poor command tracking is retained rather than filtered and is flagged using "
            "`tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and `yaw_tracking_rmse_radps <= 0.05`. "
            "The category and overall SVG figures use the "
            "tracking-qualified aggregates; the CSV/JSON tables retain both views.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _directional_pair_metrics(study: dict[str, Any], rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return partner-paired forward/backward velocity-bias metrics."""
    directional_rows: list[dict[str, Any]] = []
    row_by_key = {
        (int(row["gait_index"]), float(row["velocity_mps"])): row
        for row in rows
        if row.get("velocity_metric_valid") is True
    }
    gait_by_index = {int(gait["index"]): gait for gait in study["gaits"]}
    for row in rows:
        velocity = float(row["velocity_mps"])
        if velocity <= 0.0 or row.get("velocity_metric_valid") is not True:
            continue
        gait = gait_by_index[int(row["gait_index"])]
        partner_index = int(gait.get("time_reversal_partner", gait["index"]))
        partner = row_by_key.get((partner_index, -velocity))
        if partner is None:
            continue
        positive_bias = row.get("velocity_vx_bias_mps")
        negative_bias = partner.get("velocity_vx_bias_mps")
        positive_mean_vx = row.get("velocity_vx_measured_mean_mps")
        negative_mean_vx = partner.get("velocity_vx_measured_mean_mps")
        pair_epsilon = float(study["evaluation_config"]["tracking"]["relative_error_epsilon_mps"])
        pair_bias = (
            None
            if positive_mean_vx is None or negative_mean_vx is None
            else abs(float(positive_mean_vx) + float(negative_mean_vx))
        )
        directional_rows.append(
            {
                "positive_cell_id": row["cell_id"],
                "negative_cell_id": partner["cell_id"],
                "gait_index": row["gait_index"],
                "partner_gait_index": partner_index,
                "family": row["family"],
                "speed_mps": velocity,
                "positive_vx_bias_mps": positive_bias,
                "negative_vx_bias_mps": negative_bias,
                "directed_positive_bias_mps": positive_bias,
                "directed_negative_bias_mps": None if negative_bias is None else -float(negative_bias),
                "paired_directional_bias_mps": (
                    None
                    if positive_bias is None or negative_bias is None
                    else 0.5 * (float(positive_bias) - float(negative_bias))
                ),
                "directional_asymmetry_mps": (
                    None
                    if positive_bias is None or negative_bias is None
                    else float(positive_bias) + float(negative_bias)
                ),
                "positive_measured_vx_mps": positive_mean_vx,
                "negative_measured_vx_mps": negative_mean_vx,
                "e_pair_bias_mps": pair_bias,
                "e_pair_bias_normalized": (
                    None if pair_bias is None else pair_bias / (2.0 * abs(velocity) + pair_epsilon)
                ),
                "e_pair_bias_epsilon_mps": pair_epsilon,
                "e_pair_bias_definition": "abs(mean_vx_pos + mean_vx_neg)",
            }
        )
    return directional_rows


def _stratified_fidelity_metrics(study: Mapping[str, Any], rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate fidelity with explicit scientific-domain counts for each requested stratum."""

    def summarize(scope: str, label: str, selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        velocity = [row for row in selected if row.get("velocity_metric_valid") is True]
        heading = [row for row in selected if row.get("heading_metric_valid") is True]
        gait = [row for row in selected if row.get("gait_metric_valid") is True]
        classification = [row for row in selected if row.get("gait_classification_available") is True]
        raw_load = [row for row in selected if row.get("raw_load_metric_valid") is True]
        grf_load = [row for row in selected if row.get("grf_load_metric_valid") is True]
        normalized_load = [row for row in selected if row.get("normalized_load_metric_valid") is True]
        velocity_success = [row for row in selected if row.get("velocity_success_domain_valid") is True]
        gait_success = [row for row in selected if row.get("gait_success_domain_valid") is True]
        joint_success = [row for row in selected if row.get("joint_success_domain_valid") is True]
        return {
            "scope": scope,
            "stratum": label,
            "planned_cells": len(selected),
            "terminated_cells": sum(row.get("terminated") is True for row in selected),
            "velocity_valid_cells": len(velocity),
            "velocity_vx_relative_rmse_mean": _mean(row.get("velocity_vx_relative_rmse") for row in velocity),
            "velocity_yaw_rmse_radps_mean": _mean(row.get("velocity_yaw_rmse_radps") for row in velocity),
            "heading_valid_cells": len(heading),
            "heading_rmse_rad_mean": _mean(row.get("velocity_heading_rmse_rad") for row in heading),
            "gait_valid_cells": len(gait),
            "gait_agreement_boundary_excluded_mean": _mean(row.get("gait_agreement_boundary_excluded") for row in gait),
            "gait_duty_factor_abs_error_mean": _mean(row.get("gait_duty_factor_abs_error_mean") for row in gait),
            "classification_valid_cells": len(classification),
            "classification_unclassified_cells": sum(
                row.get("gait_classified_row") == "unclassified" for row in classification
            ),
            "classification_row_accuracy": (
                sum(row.get("gait_row_correct") is True for row in classification) / len(classification)
                if classification
                else None
            ),
            "classification_family_accuracy": (
                sum(row.get("gait_family_correct") is True for row in classification) / len(classification)
                if classification
                else None
            ),
            "raw_load_valid_cells": len(raw_load),
            "absolute_work_total_j_mean": _mean(row.get("load_absolute_work_total_j") for row in raw_load),
            "grf_load_valid_cells": len(grf_load),
            "vertical_grf_impulse_total_ns_mean": _mean(
                row.get("load_vertical_grf_impulse_total_ns") for row in grf_load
            ),
            "normalized_load_valid_cells": len(normalized_load),
            "normalized_torque_squared_total_s_mean": _mean(
                row.get("load_normalized_torque_squared_total") for row in normalized_load
            ),
            "velocity_success_domain_cells": len(velocity_success),
            "velocity_success_cells": sum(row.get("velocity_only_success") is True for row in velocity_success),
            "gait_success_domain_cells": len(gait_success),
            "gait_success_cells": sum(row.get("gait_only_success") is True for row in gait_success),
            "joint_success_domain_cells": len(joint_success),
            "joint_success_cells": sum(row.get("joint_velocity_and_gait_success") is True for row in joint_success),
        }

    definitions: list[tuple[str, Any]] = [
        ("row", lambda row: str(row["gait_name"])),
        ("family", lambda row: str(row["family"])),
        ("velocity", lambda row: f"{float(row['velocity_mps']):+g} m/s"),
        ("direction", lambda row: str(row["velocity_sign"])),
        ("policy", lambda _row: str(study["checkpoint"]["sha256"])),
    ]
    stratified: list[dict[str, Any]] = []
    for scope, key_fn in definitions:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[key_fn(row)].append(row)
        for label in sorted(grouped):
            stratified.append(summarize(scope, label, grouped[label]))
    return stratified


def analyze_study(study_path: Path) -> dict[str, Any]:
    """Analyze all available grid cells and regenerate tables, figures, and report."""
    study_path = study_path.resolve()
    study = load_study_manifest(study_path)
    study_root = study_path.parent
    rows = [analyze_cell(study, cell, study_root) for cell in study["cells"]]
    family_rows, overall = aggregate_results(study, rows)
    metrics_dir = study_root / "metrics"
    figures_dir = study_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_source_path = Path(_metrics.__file__).resolve()
    analysis_provenance = {
        "schema_version": 1,
        "analysis_method_version": study.get("method_version", METHOD_VERSION),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analyzer_path": str(Path(__file__).resolve()),
        "analyzer_sha256": _sha256_file(Path(__file__).resolve()),
        "metrics_path": str(metrics_source_path),
        "metrics_sha256": _sha256_file(metrics_source_path),
        "git_commit": _git_value(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
        "git_branch": _git_value(Path(__file__).resolve().parents[2], "branch", "--show-current"),
        "input_study_path": str(study_path),
        "input_study_sha256": _sha256_file(study_path),
        "input_study_identity_sha256": study["study_identity_sha256"],
    }
    analysis_provenance["record_sha256"] = _canonical_sha256(analysis_provenance)
    overall["analysis_provenance"] = {
        "path": "metrics/analysis_provenance.json",
        "record_sha256": analysis_provenance["record_sha256"],
        "analyzer_sha256": analysis_provenance["analyzer_sha256"],
        "metrics_sha256": analysis_provenance["metrics_sha256"],
        "input_study_sha256": analysis_provenance["input_study_sha256"],
    }
    if study.get("method_version") in {FULL_METHOD_VERSION, LIGHT_METHOD_VERSION}:
        paired_payload = _load_paired_consistency_module().analyze_study_file(study_path)
        paired_outputs = paired_payload["outputs"]
        overall["paired_tr_consistency"] = {
            "method_version": paired_payload["method_version"],
            "rows": len(paired_payload["rows"]),
            "analyzed_rows": sum(row.get("status") == "analyzed" for row in paired_payload["rows"]),
            "outputs": {key: f"metrics/{value}" for key, value in paired_outputs.items() if key != "record_sha256"},
            "record_sha256": paired_outputs["record_sha256"],
        }
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    _write_csv(metrics_dir / "cell_metrics.csv", public_rows)
    _write_csv(metrics_dir / "family_metrics.csv", family_rows)
    stratified_rows = _stratified_fidelity_metrics(study, rows)
    _write_csv(metrics_dir / "stratified_fidelity.csv", stratified_rows)
    _write_json_atomic(
        metrics_dir / "stratified_fidelity.json",
        {
            "schema_version": 1,
            "method_version": study.get("method_version", METHOD_VERSION),
            "plan_sha256": _sha256_file(study_path),
            "rows": stratified_rows,
        },
    )
    joint_rows = [metric for row in rows for metric in row.get("_joint_metric_rows", [])]
    foot_rows = [metric for row in rows for metric in row.get("_foot_metric_rows", [])]
    classification_rows = [row["_confusion_row"] for row in rows if "_confusion_row" in row]
    _write_optional_csv(metrics_dir / "joint_metrics.csv", joint_rows)
    _write_optional_csv(metrics_dir / "foot_metrics.csv", foot_rows)
    confusion_rows: list[dict[str, Any]] = []
    if classification_rows:
        valid_classification_rows = [row for row in classification_rows if row["classification_domain_valid"]]
        for level, planned_key, classified_key in (
            ("row", "planned_row", "classified_row"),
            ("family", "planned_family", "classified_family"),
        ):
            counts = Counter((row[planned_key], row[classified_key]) for row in valid_classification_rows)
            for (planned_label, classified_label), count in sorted(counts.items()):
                confusion_rows.append(
                    {
                        "level": level,
                        "planned_label": planned_label,
                        "classified_label": classified_label,
                        "count": count,
                    }
                )
    _write_optional_csv(metrics_dir / "gait_classifications.csv", classification_rows)
    _write_optional_csv(metrics_dir / "gait_confusion_matrix.csv", confusion_rows)
    synchronization_rows = [metric for row in rows for metric in row.get("_synchronization_rows", [])]
    _write_optional_csv(metrics_dir / "same_phase_pair_metrics.csv", synchronization_rows)
    directional_rows = _directional_pair_metrics(study, rows)
    _write_optional_csv(metrics_dir / "directional_pair_metrics.csv", directional_rows)
    coverage_rows = [
        {
            "cell_id": row["cell_id"],
            "gait_index": row["gait_index"],
            "family": row["family"],
            "velocity_mps": row["velocity_mps"],
            "seed": row["seed"],
            "status": row["status"],
            "reason": row["reason"],
            "positive_directed_progress": row.get("positive_directed_progress"),
            "tracking_rmse_mps": row.get("tracking_rmse_mps"),
            "yaw_tracking_rmse_radps": row.get("yaw_tracking_rmse_radps"),
            "tracking_success_threshold_mps": row.get("tracking_success_threshold_mps"),
            "yaw_tracking_success_threshold_radps": row.get("yaw_tracking_success_threshold_radps"),
            "vx_relative_tracking_success_threshold": row.get("vx_relative_tracking_success_threshold"),
            "vx_relative_tracking_success": row.get("vx_relative_tracking_success"),
            "planar_tracking_success": row.get("planar_tracking_success"),
            "yaw_tracking_success": row.get("yaw_tracking_success"),
            "tracking_success": row.get("tracking_success"),
            "velocity_metric_valid": row.get("velocity_metric_valid"),
            "velocity_metric_reason": row.get("velocity_metric_reason"),
            "heading_metric_valid": row.get("heading_metric_valid"),
            "heading_metric_reason": row.get("heading_metric_reason"),
            "recording_manifest_valid": row.get("recording_manifest_valid"),
            "recording_manifest_reason": row.get("recording_manifest_reason"),
            "gait_metric_valid": row.get("gait_metric_valid"),
            "gait_metric_reason": row.get("gait_metric_reason"),
            "load_metric_valid": row.get("load_metric_valid"),
            "load_metric_reason": row.get("load_metric_reason"),
            "raw_load_metric_valid": row.get("raw_load_metric_valid"),
            "raw_load_metric_reason": row.get("raw_load_metric_reason"),
            "grf_load_metric_valid": row.get("grf_load_metric_valid"),
            "grf_load_metric_reason": row.get("grf_load_metric_reason"),
            "normalized_load_metric_valid": row.get("normalized_load_metric_valid"),
            "normalized_load_metric_reason": row.get("load_metric_reason"),
            "velocity_success_domain_valid": row.get("velocity_success_domain_valid"),
            "gait_success_domain_valid": row.get("gait_success_domain_valid"),
            "joint_success_domain_valid": row.get("joint_success_domain_valid"),
            "velocity_only_success": row.get("velocity_only_success"),
            "gait_only_success": row.get("gait_only_success"),
            "joint_velocity_and_gait_success": row.get("joint_velocity_and_gait_success"),
            "gait_row_correct": row.get("gait_row_correct"),
            "gait_family_correct": row.get("gait_family_correct"),
            "gait_classification_available": row.get("gait_classification_available"),
            "success": row.get("success"),
            "success_failed_checks": row.get("success_failed_checks"),
            "relative_output_dir": row["relative_output_dir"],
        }
        for row in rows
    ]
    _write_csv(metrics_dir / "coverage.csv", coverage_rows)
    _write_json_atomic(metrics_dir / "analysis_provenance.json", analysis_provenance)
    _write_json_atomic(metrics_dir / "overall_metrics.json", overall)
    for family in FAMILY_ORDER:
        _save_family_figure(figures_dir / f"{family}.svg", family, family_rows)
    _save_overall_figure(figures_dir / "overall.svg", overall)
    _save_coverage_figure(figures_dir / "coverage.svg", family_rows)
    if study.get("method_version") in {FULL_METHOD_VERSION, LIGHT_METHOD_VERSION}:
        _save_screening_figure(figures_dir / "screening_report.svg", study, rows, overall)
        _write_screening_report(metrics_dir / "SCREENING_REPORT.md", study, rows, overall)
    _write_report(metrics_dir / "REPORT.md", study, family_rows, overall)
    return overall


if __name__ == "__main__":
    from symm_cli import main

    raise SystemExit(main(["evaluation", *sys.argv[1:]]))
