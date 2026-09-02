# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable initialization provenance and matched-cohort validation.

This module deliberately has no Isaac Lab imports.  The RSL-RL training
entrypoint loads it from this checkout immediately before ``runner.learn``.
The cohort validator can therefore also run in a plain Python interpreter.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import enum
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
INITIALIZATION_RELATIVE_PATH = Path("provenance") / "initialization.json"
RESOLVED_COMMAND_RELATIVE_PATH = Path("provenance") / "resolved_command.json"
COHORT_METADATA_RELATIVE_PATH = Path("provenance") / "cohort_metadata.json"
POLICY_CONTRACT_RELATIVE_PATH = Path("policy_contract.json")
TR_CONSISTENCY_MODES = {"transition_aligned_sequence", "framewise_feature_approx", "none"}
TR_CONSISTENCY_MAPPING_VERSION = "transition_aligned_causal_sequence_v1"
POLICY_ACTION_HISTORY_LENGTH = 2
TR_SEQUENCE_ACTOR_ALIGNMENT = "edge_t_to_reverse_state_t_plus_1"
TR_SEQUENCE_VALUE_ALIGNMENT = "state_t_plus_1"
_DIRECT_CURRICULUM_CONFIG_FIELDS = {
    "mode": "command_curriculum_mode",
    "velocity_bin_count": "curriculum_velocity_bin_count",
    "ewma_coefficient": "curriculum_ewma_coefficient",
    "unlock_threshold": "curriculum_unlock_threshold",
    "initial_max_abs_speed": "curriculum_initial_max_abs_speed",
    "min_visits": "curriculum_min_visits",
    "current_cell_increment": "curriculum_current_cell_increment",
    "neighbor_increment": "curriculum_neighbor_increment",
    "exploration_floor": "curriculum_exploration_floor",
    "maximum_weight": "curriculum_maximum_weight",
    "locked_cell_weight": "curriculum_locked_cell_weight",
    "seed": "curriculum_seed",
}
_DIRECT_TR_TREATMENT_CONFIG_PATHS = {
    "mirror_loss_coeff": "algorithm.symmetry_cfg.mirror_loss_coeff",
    "value_loss_coeff": "algorithm.symmetry_cfg.value_loss_coeff",
    "use_data_augmentation": "algorithm.symmetry_cfg.use_data_augmentation",
    "trajectory_augmentation_enabled": "algorithm.symmetry_cfg.tr_augmentation.enabled",
}
_STRICT_MATCH_FIELDS = (
    "seed",
    "training_seed",
    "environment_seed",
    "actor_state_dict_sha256",
    "critic_state_dict_sha256",
    "actor_distribution_parameter_sha256",
    "optimizer_state_sha256",
    "repo_commit",
    "dirty_tree_diff_sha256",
)
_STRICT_SIGNATURES = (
    "architecture",
    "optimizer_settings",
    "reward_profile",
    "gait_library",
    "command_distribution",
    "environment_randomization",
    "training_budget",
)

# Run names and log destinations are deliberately unique per condition but do
# not affect the scientific training configuration.  Every other resolved
# configuration path is compared unless it is derived from an explicitly
# declared treatment through the allow-list below.
_ALWAYS_MASKED_CONFIG_PATHS = {
    "environment": ("log_dir",),
    "agent": ("run_name",),
}
_TREATMENT_RUNTIME_PATHS = {
    "tr_policy.enabled": (
        ("agent", "algorithm.symmetry_cfg.use_time_reversal_regularization"),
        ("agent", "algorithm.symmetry_cfg.use_mirror_loss"),
        ("agent", "algorithm.symmetry_cfg.use_tr_policy_consistency"),
        ("agent", "algorithm.symmetry_cfg.tr_policy_schedule.enabled"),
    ),
    "tr_policy.coefficient": (
        ("agent", "algorithm.symmetry_cfg.mirror_loss_coeff"),
        ("agent", "algorithm.symmetry_cfg.tr_policy_schedule.target_coeff"),
    ),
    "tr_policy.schedule.enabled": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.enabled"),),
    "tr_policy.schedule.warmup_iterations": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.warmup_iterations"),),
    "tr_policy.schedule.rampup_iterations": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.rampup_iterations"),),
    "tr_policy.schedule.hold_iterations": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.hold_iterations"),),
    "tr_policy.schedule.decay_iterations": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.decay_iterations"),),
    "tr_policy.schedule.final_scale": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.final_scale"),),
    "tr_policy.schedule.ramp_shape": (("agent", "algorithm.symmetry_cfg.tr_policy_schedule.ramp_shape"),),
    "tr_value.enabled": (
        ("agent", "algorithm.symmetry_cfg.use_time_reversal_regularization"),
        ("agent", "algorithm.symmetry_cfg.use_tr_value_consistency"),
        ("agent", "algorithm.symmetry_cfg.tr_value_schedule.enabled"),
    ),
    "tr_value.coefficient": (
        ("agent", "algorithm.symmetry_cfg.value_loss_coeff"),
        ("agent", "algorithm.symmetry_cfg.tr_value_schedule.target_coeff"),
    ),
    "tr_value.schedule.enabled": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.enabled"),),
    "tr_value.schedule.warmup_iterations": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.warmup_iterations"),),
    "tr_value.schedule.rampup_iterations": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.rampup_iterations"),),
    "tr_value.schedule.hold_iterations": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.hold_iterations"),),
    "tr_value.schedule.decay_iterations": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.decay_iterations"),),
    "tr_value.schedule.final_scale": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.final_scale"),),
    "tr_value.schedule.ramp_shape": (("agent", "algorithm.symmetry_cfg.tr_value_schedule.ramp_shape"),),
    "validity_mask.mode": (("agent", "algorithm.symmetry_cfg.tr_validity.mode"),),
    "validity_mask.thresholds.min_abs_command_velocity": (
        ("agent", "algorithm.symmetry_cfg.tr_validity.min_abs_command_velocity"),
    ),
    "validity_mask.thresholds.tracking_abs_tolerance": (
        ("agent", "algorithm.symmetry_cfg.tr_validity.tracking_abs_tolerance"),
    ),
    "validity_mask.thresholds.tracking_rel_tolerance": (
        ("agent", "algorithm.symmetry_cfg.tr_validity.tracking_rel_tolerance"),
    ),
    "validity_mask.thresholds.projected_gravity_tolerance": (
        ("agent", "algorithm.symmetry_cfg.tr_validity.projected_gravity_tolerance"),
    ),
    "validity_mask.thresholds.phase_boundary_margin": (
        ("agent", "algorithm.symmetry_cfg.tr_validity.phase_boundary_margin"),
    ),
    "validity_mask.thresholds.command_speed_bin_edges": (
        ("agent", "algorithm.symmetry_cfg.tr_validity.command_speed_bin_edges"),
    ),
    "trajectory_augmentation.enabled": (
        ("agent", "algorithm.symmetry_cfg.use_time_reversal_regularization"),
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.enabled"),
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.enabled"),
    ),
    "trajectory_augmentation.mode": (("agent", "algorithm.symmetry_cfg.tr_augmentation.mode"),),
    "trajectory_augmentation.action_source": (("agent", "algorithm.symmetry_cfg.tr_augmentation.action_source"),),
    "trajectory_augmentation.coefficient": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.coefficient"),
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.target_coeff"),
    ),
    "trajectory_augmentation.schedule.enabled": (("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.enabled"),),
    "trajectory_augmentation.schedule.warmup_iterations": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.warmup_iterations"),
    ),
    "trajectory_augmentation.schedule.rampup_iterations": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.rampup_iterations"),
    ),
    "trajectory_augmentation.schedule.hold_iterations": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.hold_iterations"),
    ),
    "trajectory_augmentation.schedule.decay_iterations": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.decay_iterations"),
    ),
    "trajectory_augmentation.schedule.final_scale": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.final_scale"),
    ),
    "trajectory_augmentation.schedule.ramp_shape": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.schedule.ramp_shape"),
    ),
    "trajectory_augmentation.filter_settings.filter_enabled": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.filter_enabled"),
    ),
    "trajectory_augmentation.filter_settings.validation_fraction": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.validation_fraction"),
    ),
    "trajectory_augmentation.filter_settings.validation_quantile": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.validation_quantile"),
    ),
    "trajectory_augmentation.filter_settings.threshold_multiplier": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.threshold_multiplier"),
    ),
    "trajectory_augmentation.filter_settings.minimum_validation_samples": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.minimum_validation_samples"),
    ),
    "trajectory_augmentation.filter_settings.maximum_validation_loss": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.maximum_validation_loss"),
    ),
    "trajectory_augmentation.filter_settings.phase_boundary_margin": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.phase_boundary_margin"),
    ),
    "trajectory_augmentation.filter_settings.maximum_contact_impulse": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.maximum_contact_impulse"),
    ),
    "trajectory_augmentation.filter_settings.action_abs_limit": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.action_abs_limit"),
    ),
    "trajectory_augmentation.filter_settings.max_augmented_to_original_ratio": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.max_augmented_to_original_ratio"),
    ),
    "trajectory_augmentation.filter_settings.use_confidence_weights": (
        ("agent", "algorithm.symmetry_cfg.tr_augmentation.use_confidence_weights"),
    ),
    "trajectory_augmentation.filter_settings.rng_seed": (("agent", "algorithm.symmetry_cfg.tr_augmentation.rng_seed"),),
    "leg_sync_reward_weight": (("environment", "rewards.leg_permutation_symmetry.weight"),),
    "training_iterations": (("agent", "max_iterations"),),
    "num_envs": (("environment", "scene.num_envs"),),
}


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation of a supported value."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_value(value: Any) -> str:
    """Hash a value after deterministic type-aware normalization."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _working_path_sha256(path: Path) -> str:
    if path.is_symlink():
        target = os.readlink(path)
        return hashlib.sha256(f"symlink\0{target}".encode("utf-8", errors="surrogateescape")).hexdigest()
    return sha256_file(path)


def state_sha256(state: Any) -> str:
    """Hash nested model, optimizer, RNG, or configuration state.

    Tensor bytes, dtypes, shapes, mapping keys, and container boundaries are
    included explicitly.  This avoids the timestamps and storage identifiers
    embedded by ``torch.save``.
    """
    digest = hashlib.sha256()
    _update_state_hash(digest, state)
    return digest.hexdigest()


def _update_state_hash(digest: Any, value: Any) -> None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - training always has NumPy
        np = None
    try:
        import torch
    except ImportError:  # pragma: no cover - training always has torch
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        return
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if np is not None and isinstance(value, np.generic):
        _update_state_hash(digest, value.item())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: canonical_json(item)):
            _update_state_hash(digest, key)
            _update_state_hash(digest, value[key])
        digest.update(b"end-mapping\0")
        return
    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_state_hash(digest, item)
        digest.update(b"end-tuple\0")
        return
    if isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_state_hash(digest, item)
        digest.update(b"end-list\0")
        return
    digest.update(canonical_json({"type": type(value).__qualname__, "value": value}).encode("utf-8"))
    digest.update(b"\0")


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - launcher need not have NumPy
        np = None
    try:
        import torch
    except ImportError:  # pragma: no cover - launcher need not have torch
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "__tensor__": state_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if np is not None and isinstance(value, np.ndarray):
        return {
            "__ndarray__": state_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if np is not None and isinstance(value, np.generic):
        return _jsonable(value.item())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=canonical_json)
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__float__": repr(value)}
        return value
    if isinstance(value, str | int | bool) or value is None:
        return value
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', repr(value))}"
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        public = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return {"__class__": f"{type(value).__module__}:{type(value).__qualname__}", **_jsonable(public)}
    return repr(value)


def _config_dict(config: Any) -> dict[str, Any]:
    normalized = _jsonable(config.to_dict() if hasattr(config, "to_dict") else config)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected a mapping-like configuration, received {type(config).__name__}.")
    return normalized


def _git_output(repo_root: Path, *args: str, text: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=False,
        text=text,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def git_provenance(repo_root: Path) -> dict[str, Any]:
    """Return commit and dirty-tree provenance, including untracked files."""
    commit = str(_git_output(repo_root, "rev-parse", "HEAD", text=True)).strip()
    branch = str(_git_output(repo_root, "branch", "--show-current", text=True)).strip()
    status_raw = bytes(_git_output(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all"))
    raw_diff = bytes(_git_output(repo_root, "diff", "--raw", "--no-abbrev", "-z", "HEAD", "--"))
    current_files: list[dict[str, str]] = []
    fields = [field for field in status_raw.split(b"\0") if field]
    field_index = 0
    while field_index < len(fields):
        field = fields[field_index]
        if len(field) < 4 or field[2:3] != b" ":
            raise RuntimeError("Unable to parse NUL-delimited Git status output.")
        status = field[:2]
        relative = field[3:].decode("utf-8", errors="surrogateescape")
        path = repo_root / relative
        if path.is_file() or path.is_symlink():
            current_files.append({"path": relative.replace("\\", "/"), "sha256": _working_path_sha256(path)})
        field_index += 2 if b"R" in status or b"C" in status else 1
    dirty_payload = {
        "raw_diff_sha256": hashlib.sha256(raw_diff).hexdigest(),
        "status_sha256": hashlib.sha256(status_raw).hexdigest(),
        "current_file_sha256": sorted(current_files, key=lambda item: item["path"]),
    }
    return {
        "repo_commit": commit,
        "repo_branch": branch,
        "dirty_tree": bool(status_raw),
        "dirty_tree_diff_sha256": sha256_value(dirty_payload),
        "dirty_tree_components": dirty_payload,
    }


def _asset_references(config: Any) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                child_path = f"{path}.{child_key}" if path else str(child_key)
                visit(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str) and path.rsplit(".", 1)[-1] in {"asset_path", "usd_path", "urdf_path"}:
            references.append({"config_path": path, "reference": value})

    visit(config)
    unique = {(item["config_path"], item["reference"]): item for item in references}
    return [unique[key] for key in sorted(unique)]


def robot_asset_hashes(env_config: Any, repo_root: Path) -> dict[str, Any]:
    """Hash every local asset and retain explicit external-reference risks."""
    assets: list[dict[str, Any]] = []
    for reference in _asset_references(env_config):
        reference_value = reference["reference"]
        asset_path = Path(reference_value)
        if not asset_path.is_file():
            status = "unresolved_external" if "://" in reference_value else "missing_local"
            assets.append(
                {
                    **reference,
                    "status": status,
                    "reference_sha256": sha256_value(reference),
                    "risk": "asset bytes were unavailable when initialization provenance was recorded",
                }
            )
            continue
        asset_path = asset_path.resolve()
        asset_root = (
            asset_path.parent.parent if asset_path.parent.name.lower() in {"urdf", "usd"} else asset_path.parent
        )
        try:
            relative_root = asset_root.relative_to(repo_root)
        except ValueError:
            tracked_files: list[Path] = []
        else:
            output = str(_git_output(repo_root, "ls-files", "--", relative_root.as_posix(), text=True))
            tracked_files = [repo_root / line for line in output.splitlines() if line]
        if not tracked_files:
            tracked_files = [path for path in asset_root.rglob("*") if path.is_file()]
        files = [
            {"path": path.relative_to(asset_root).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(tracked_files, key=str)
        ]
        assets.append(
            {
                **reference,
                "status": "hashed",
                "asset_path": str(asset_path),
                "asset_root": str(asset_root),
                "files": files,
                "sha256": sha256_value(files),
            }
        )
    status_counts = {
        status: sum(asset["status"] == status for asset in assets)
        for status in ("hashed", "unresolved_external", "missing_local")
    }
    return {
        "assets": assets,
        "reference_count": len(assets),
        "status_counts": status_counts,
        "has_unresolved_risk": any(asset["status"] != "hashed" for asset in assets),
        "sha256": sha256_value(assets),
    }


def _module_architecture(module: Any) -> dict[str, Any]:
    return {
        "class": f"{type(module).__module__}:{type(module).__qualname__}",
        "parameters": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in module.named_parameters()
        ],
        "buffers": [
            {"name": name, "shape": list(buffer.shape), "dtype": str(buffer.dtype)}
            for name, buffer in module.named_buffers()
        ],
    }


def _distribution_state(actor: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    distribution = getattr(actor, "distribution", None)
    if distribution is not None:
        state["class"] = f"{type(distribution).__module__}:{type(distribution).__qualname__}"
        if hasattr(distribution, "state_dict"):
            state["state_dict"] = distribution.state_dict()
    actor_state = actor.state_dict()
    state["actor_distribution_entries"] = {
        key: value
        for key, value in actor_state.items()
        if any(token in key.lower() for token in ("distribution", "log_std", "std"))
    }
    return state


def _optimizer_settings(optimizer: Any) -> dict[str, Any]:
    state = optimizer.state_dict()
    groups = []
    for group in state.get("param_groups", []):
        groups.append({key: value for key, value in group.items() if key != "params"})
    return {
        "class": f"{type(optimizer).__module__}:{type(optimizer).__qualname__}",
        "defaults": getattr(optimizer, "defaults", {}),
        "param_groups": groups,
    }


def _lookup(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _dotted_lookup(mapping: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """Return whether a dotted configuration path exists and its value."""
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _runtime_values_equal(actual: Any, expected: Any) -> bool:
    """Compare resolved Hydra values while accepting numeric type coercion."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) and float(actual) == float(expected)
    return _jsonable(actual) == _jsonable(expected)


def _parse_hydra_override_value(value: str) -> Any:
    """Parse the primitive/list values permitted in manifest reward overrides."""
    stripped = value.strip()
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    if stripped in {"null", "None"}:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            numeric = float(stripped)
        except ValueError:
            return stripped
        integer_literal = numeric.is_integer() and all(token not in stripped.lower() for token in (".", "e"))
        return int(numeric) if integer_literal else numeric


def _expected_runtime_config_values(condition: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Translate one normalized study condition to exact resolved config leaves."""
    required = {
        "tr_policy",
        "tr_value",
        "validity_mask",
        "trajectory_augmentation",
        "reward_profile",
        "leg_sync_reward_weight",
        "training_iterations",
        "num_envs",
    }
    missing = sorted(required - set(condition))
    if missing:
        raise ValueError(f"Study resolved_condition is missing runtime-mapped fields: {missing}")
    policy = condition["tr_policy"]
    value = condition["tr_value"]
    validity = condition["validity_mask"]
    augmentation = condition["trajectory_augmentation"]
    reward_profile = condition["reward_profile"]
    if not all(isinstance(item, Mapping) for item in (policy, value, validity, augmentation, reward_profile)):
        raise ValueError("Study resolved_condition contains malformed runtime-mapped objects.")
    policy_schedule = policy.get("schedule")
    value_schedule = value.get("schedule")
    augmentation_schedule = augmentation.get("schedule")
    thresholds = validity.get("thresholds")
    filter_settings = augmentation.get("filter_settings")
    if not all(
        isinstance(item, Mapping)
        for item in (policy_schedule, value_schedule, augmentation_schedule, thresholds, filter_settings)
    ):
        raise ValueError("Study resolved_condition contains malformed schedule, validity, or filter settings.")

    root = "algorithm.symmetry_cfg"
    any_enabled = bool(policy.get("enabled") or value.get("enabled") or augmentation.get("enabled"))
    agent: dict[str, Any] = {
        f"{root}.use_time_reversal_regularization": any_enabled,
        f"{root}.use_data_augmentation": False,
        f"{root}.use_mirror_loss": bool(policy.get("enabled")),
        f"{root}.mirror_loss_coeff": policy.get("coefficient"),
        f"{root}.value_loss_coeff": value.get("coefficient"),
        f"{root}.use_tr_policy_consistency": bool(policy.get("enabled")),
        f"{root}.use_tr_value_consistency": bool(value.get("enabled")),
        f"{root}.tr_validity.mode": validity.get("mode"),
        f"{root}.tr_augmentation.enabled": bool(augmentation.get("enabled")),
        f"{root}.tr_augmentation.mode": augmentation.get("mode"),
        f"{root}.tr_augmentation.action_source": augmentation.get("action_source"),
        f"{root}.tr_augmentation.coefficient": augmentation.get("coefficient"),
        "max_iterations": condition["training_iterations"],
    }
    for name, term, schedule in (
        ("tr_policy_schedule", policy, policy_schedule),
        ("tr_value_schedule", value, value_schedule),
        ("tr_augmentation.schedule", augmentation, augmentation_schedule),
    ):
        prefix = f"{root}.{name}"
        for field, setting in schedule.items():
            agent[f"{prefix}.{field}"] = bool(term.get("enabled")) and bool(setting) if field == "enabled" else setting
        agent[f"{prefix}.target_coeff"] = term.get("coefficient")
    for field, setting in thresholds.items():
        agent[f"{root}.tr_validity.{field}"] = setting
    for field, setting in filter_settings.items():
        agent[f"{root}.tr_augmentation.{field}"] = setting

    environment: dict[str, Any] = {
        "scene.num_envs": condition["num_envs"],
        "rewards.leg_permutation_symmetry.weight": condition["leg_sync_reward_weight"],
    }
    overrides = reward_profile.get("overrides")
    if not isinstance(overrides, list):
        raise ValueError("Study reward_profile.overrides must be a list.")
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ValueError(f"Study reward override is malformed: {override!r}")
        path, raw_value = override.split("=", 1)
        if not path.startswith("env.rewards."):
            raise ValueError(f"Study reward override is outside env.rewards: {override!r}")
        environment[path.removeprefix("env.")] = _parse_hydra_override_value(raw_value)
    return {"environment": environment, "agent": agent}


def _expected_runtime_argv(exact_training_command: Any) -> list[str]:
    """Extract the library-specific training argv from a wrapper command."""
    if not isinstance(exact_training_command, list) or not all(
        isinstance(token, str) for token in exact_training_command
    ):
        raise ValueError("Study exact_training_command must be a list of strings.")
    try:
        train_index = exact_training_command.index("train")
    except ValueError as exc:
        raise ValueError("Study exact_training_command does not contain the train action.") from exc
    arguments = list(exact_training_command[train_index + 1 :])
    try:
        library_index = arguments.index("--rl_library")
    except ValueError as exc:
        raise ValueError("Study exact_training_command does not select --rl_library.") from exc
    if library_index + 1 >= len(arguments) or arguments[library_index + 1] != "rsl_rl":
        raise ValueError("Study exact_training_command must select --rl_library rsl_rl.")
    del arguments[library_index : library_index + 2]
    return arguments


def _validate_condition_runtime_mapping(
    condition: Mapping[str, Any],
    env_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
) -> None:
    """Fail closed unless every declared condition leaf reached its runtime config."""
    expected_configs = _expected_runtime_config_values(condition)
    errors: list[str] = []
    for config_name, config in (("environment", env_config), ("agent", agent_config)):
        for path, expected in expected_configs[config_name].items():
            exists, actual = _dotted_lookup(config, path)
            if not exists:
                errors.append(f"{config_name}.{path} is absent")
            elif not _runtime_values_equal(actual, expected):
                errors.append(f"{config_name}.{path} expected {canonical_json(expected)}, got {canonical_json(actual)}")
    if errors:
        raise ValueError("Study condition did not resolve to the declared runtime configuration: " + "; ".join(errors))


def _matched_signatures(
    actor: Any,
    critic: Any,
    optimizer: Any,
    env_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> dict[str, str]:
    resolved_condition = context.get("resolved_condition", {}) if context else {}
    architecture = {
        "actor_runtime": _module_architecture(actor),
        "critic_runtime": _module_architecture(critic),
        "actor_config": agent_config.get("actor"),
        "critic_config": agent_config.get("critic"),
        "obs_groups": agent_config.get("obs_groups"),
    }
    optimizer_settings = {
        "runtime": _optimizer_settings(optimizer),
        "configured_optimizer": _lookup(agent_config, "algorithm", "optimizer"),
        "configured_learning_rate": _lookup(agent_config, "algorithm", "learning_rate"),
    }
    declared_reward_profile = resolved_condition.get("reward_profile", {})
    reward_profile = {
        # The profile name is a provenance label, not a training input.
        "declared_overrides": (
            declared_reward_profile.get("overrides") if isinstance(declared_reward_profile, Mapping) else None
        ),
        "resolved_rewards": env_config.get("rewards"),
    }
    gait_library = _lookup(env_config, "commands", "base_velocity", default={})
    training_budget = {
        "max_iterations": agent_config.get("max_iterations"),
        "num_steps_per_env": agent_config.get("num_steps_per_env"),
        "num_envs": _lookup(env_config, "scene", "num_envs"),
    }
    return {
        "architecture": sha256_value(architecture),
        "optimizer_settings": sha256_value(optimizer_settings),
        "reward_profile": sha256_value(reward_profile),
        "gait_library": sha256_value(gait_library),
        "command_distribution": sha256_value(env_config.get("commands")),
        "environment_randomization": sha256_value(env_config.get("events")),
        "training_budget": sha256_value(training_budget),
    }


def _load_context(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if path is None:
        return None
    context_path = Path(path).resolve()
    with context_path.open(encoding="utf-8") as stream:
        context = json.load(stream)
    if not isinstance(context, dict):
        raise ValueError(f"Study context must be a JSON object: {context_path}")
    context["context_path"] = str(context_path)
    return context


def _validate_initialization_context(
    context: Mapping[str, Any],
    env_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    git: Mapping[str, Any],
    runtime_argv: Sequence[str] | None,
) -> None:
    """Validate that one launcher context describes the resolved training run."""
    required = {
        "study_name",
        "study_sha256",
        "expected_branch",
        "treatment_variables",
        "run_name",
        "condition_sha256",
        "resolved_condition",
        "exact_training_command",
    }
    missing = sorted(required - set(context))
    if missing:
        raise ValueError(f"Study context is missing required fields: {missing}")
    condition = context["resolved_condition"]
    if not isinstance(condition, Mapping):
        raise ValueError("Study context resolved_condition must be an object.")
    condition_sha256 = sha256_value(condition)
    if context["condition_sha256"] != condition_sha256:
        raise ValueError(
            "Study context condition digest mismatch: "
            f"expected {context['condition_sha256']}, calculated {condition_sha256}."
        )
    declared_seed = condition.get("seed")
    training_seed = agent_config.get("seed")
    environment_seed = env_config.get("seed")
    if training_seed != declared_seed or environment_seed != declared_seed:
        raise ValueError(
            "Study seed mismatch: resolved condition, training config, and environment config must agree "
            f"({declared_seed!r}, {training_seed!r}, {environment_seed!r})."
        )
    configured_run_name = agent_config.get("run_name")
    if configured_run_name != context["run_name"]:
        raise ValueError(
            f"Study run-name mismatch: context has {context['run_name']!r}, agent config has {configured_run_name!r}."
        )
    if git.get("repo_branch") != context["expected_branch"]:
        raise ValueError(
            f"Study branch mismatch: expected {context['expected_branch']!r}, found {git.get('repo_branch')!r}."
        )
    _validate_condition_runtime_mapping(condition, env_config, agent_config)
    expected_argv = _expected_runtime_argv(context["exact_training_command"])
    if runtime_argv is None:
        raise ValueError("Study initialization requires the exact library-specific runtime argv.")
    if list(runtime_argv) != expected_argv:
        raise ValueError(
            "Study runtime command differs from exact_training_command: "
            f"expected {expected_argv!r}, received {list(runtime_argv)!r}."
        )


def _is_time_reversal_run(agent_config: Mapping[str, Any], context: Mapping[str, Any] | None) -> bool:
    class_name = str(_lookup(agent_config, "algorithm", "class_name", default=""))
    return context is not None or class_name.endswith("symm_quadruped.time_reversal_ppo:TimeReversalPPO")


def _configured_algorithm_class(agent_cfg: Any) -> str:
    if isinstance(agent_cfg, Mapping):
        algorithm = agent_cfg.get("algorithm", {})
        return str(algorithm.get("class_name", "")) if isinstance(algorithm, Mapping) else ""
    algorithm = getattr(agent_cfg, "algorithm", None)
    class_name = getattr(algorithm, "class_name", "")
    if class_name:
        return str(class_name)
    if hasattr(agent_cfg, "to_dict"):
        config = agent_cfg.to_dict()
        if isinstance(config, Mapping):
            algorithm = config.get("algorithm", {})
            if isinstance(algorithm, Mapping):
                return str(algorithm.get("class_name", ""))
    return ""


def _validate_record_digest(record: Mapping[str, Any], path: Path) -> None:
    expected = record.get("record_sha256")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    actual = sha256_value(payload)
    if expected != actual:
        raise ValueError(f"Initialization record digest mismatch at {path}: expected {expected}, calculated {actual}.")


def _write_immutable_record(path: Path, record: dict[str, Any]) -> None:
    payload = dict(record)
    payload["record_sha256"] = sha256_value(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        _validate_record_digest(existing, path)
        comparable_existing = {
            key: value for key, value in existing.items() if key not in {"recorded_at_utc", "record_sha256"}
        }
        comparable_new = {
            key: value for key, value in payload.items() if key not in {"recorded_at_utc", "record_sha256"}
        }
        if comparable_existing != comparable_new:
            raise FileExistsError(f"Refusing to replace a different immutable initialization record: {path}")


def record_training_initialization(
    *,
    runner: Any,
    env_cfg: Any,
    agent_cfg: Any,
    log_dir: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    study_context_path: str | os.PathLike[str] | None = None,
    resume_checkpoint: str | os.PathLike[str] | None = None,
    runtime_argv: Sequence[str] | None = None,
) -> Path | None:
    """Write initialization provenance before the first rollout update.

    Non-time-reversal runs and nonzero distributed ranks are left untouched.
    Resume runs validate any source initialization record and create a new,
    linked record; the source record is never modified.
    """
    if int(getattr(runner.alg, "gpu_global_rank", os.getenv("RANK", "0"))) != 0:
        return None
    context = _load_context(study_context_path)
    if context is None and not _configured_algorithm_class(agent_cfg).endswith(
        "symm_quadruped.time_reversal_ppo:TimeReversalPPO"
    ):
        return None
    agent_config = _config_dict(agent_cfg)
    if not _is_time_reversal_run(agent_config, context):
        return None
    env_config = _config_dict(env_cfg)

    import numpy as np
    import torch

    actor = runner.alg.actor
    critic = runner.alg.critic
    optimizer = runner.alg.optimizer
    root = Path(repo_root).resolve()
    resume: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        checkpoint = Path(resume_checkpoint).resolve()
        source_record_path = checkpoint.parent / INITIALIZATION_RELATIVE_PATH
        resume = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_initialization": str(source_record_path) if source_record_path.is_file() else None,
            "source_initialization_status": "missing_legacy_run",
        }
        if source_record_path.is_file():
            with source_record_path.open(encoding="utf-8") as stream:
                source_record = json.load(stream)
            _validate_record_digest(source_record, source_record_path)
            resume["source_initialization_status"] = "validated"
            resume["source_initialization_record_sha256"] = source_record["record_sha256"]

    git = git_provenance(root)
    if context is not None:
        _validate_initialization_context(context, env_config, agent_config, git, runtime_argv)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "pre_first_rollout_initialization",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_seed": agent_config.get("seed"),
        "seed": agent_config.get("seed"),
        "environment_seed": env_config.get("seed"),
        "python_rng_state_sha256": state_sha256(random.getstate()),
        "numpy_rng_state_sha256": state_sha256(np.random.get_state()),
        "pytorch_cpu_rng_state_sha256": state_sha256(torch.get_rng_state()),
        "pytorch_cuda_rng_state_sha256": state_sha256(cuda_states),
        "pytorch_cuda_rng_device_sha256": [state_sha256(state) for state in cuda_states],
        "actor_state_dict_sha256": state_sha256(actor.state_dict()),
        "critic_state_dict_sha256": state_sha256(critic.state_dict()),
        "actor_distribution_parameter_sha256": state_sha256(_distribution_state(actor)),
        "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
        "resolved_env_config_sha256": sha256_value(env_config),
        "resolved_agent_config_sha256": sha256_value(agent_config),
        "resolved_configs": {"environment": env_config, "agent": agent_config},
        "matched_signatures": _matched_signatures(actor, critic, optimizer, env_config, agent_config, context),
        "robot_asset_hashes": robot_asset_hashes(env_config, root),
        "repo_commit": git["repo_commit"],
        "dirty_tree_diff_sha256": git["dirty_tree_diff_sha256"],
        "git": git,
        "runtime_command": [sys.executable, *sys.argv],
        "runtime_training_argv": list(runtime_argv) if runtime_argv is not None else None,
        "expected_training_argv": (
            _expected_runtime_argv(context["exact_training_command"]) if context is not None else None
        ),
        "study_context": context,
        "resume": resume,
    }
    path = Path(log_dir).resolve() / INITIALIZATION_RELATIVE_PATH
    _write_immutable_record(path, record)
    return path


def _parse_direct_launch_context(value: Mapping[str, Any] | str | None) -> dict[str, Any] | None:
    """Return a validated direct-launch context supplied by the convenience CLI."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        context = copy.deepcopy(dict(value))
    elif isinstance(value, str):
        try:
            context = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("--symm_direct_launch_context must contain a JSON object.") from exc
    else:
        raise ValueError("Direct-launch context must be a mapping, JSON object string, or None.")
    if not isinstance(context, dict):
        raise ValueError("--symm_direct_launch_context must contain a JSON object.")
    schema_version = context.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version not in {1, 2}:
        raise ValueError(f"--symm_direct_launch_context schema_version must be 1 or 2; received {schema_version!r}.")
    context.setdefault("schema_version", schema_version)
    return context


def _validate_direct_launch_context(
    context: Mapping[str, Any] | None,
    env_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
) -> None:
    """Reject launcher metadata contradicted by the resolved Hydra configuration."""
    if context is None:
        return

    if context.get("schema_version") == 2:
        required_sections = {
            "run_name",
            "policy_contract",
            "time_reversal_treatment",
            "command_curriculum",
        }
        missing_sections = sorted(required_sections - set(context))
        if missing_sections:
            raise ValueError(f"Direct-launch schema 2 is missing required metadata: {missing_sections}.")

    declared_run_name = context.get("run_name")
    if declared_run_name is not None:
        resolved_run_name = agent_config.get("run_name")
        if not _runtime_values_equal(resolved_run_name, declared_run_name):
            raise ValueError(
                "Direct-launch run_name differs from resolved agent configuration: "
                f"declared {declared_run_name!r}, resolved {resolved_run_name!r}."
            )

    declared_treatment = context.get("time_reversal_treatment")
    if declared_treatment is not None:
        if not isinstance(declared_treatment, Mapping):
            raise ValueError("Direct-launch time_reversal_treatment must be a mapping.")
        unexpected = sorted(set(declared_treatment) - set(_DIRECT_TR_TREATMENT_CONFIG_PATHS))
        if unexpected:
            raise ValueError(f"Direct-launch time_reversal_treatment has unexpected fields: {unexpected}.")
        mismatches: dict[str, tuple[Any, Any]] = {}
        for metadata_name, config_path in _DIRECT_TR_TREATMENT_CONFIG_PATHS.items():
            if metadata_name not in declared_treatment:
                continue
            exists, resolved = _dotted_lookup(agent_config, config_path)
            declared = declared_treatment[metadata_name]
            if not exists or not _runtime_values_equal(resolved, declared):
                mismatches[metadata_name] = (declared, resolved if exists else "<absent>")
        if mismatches:
            raise ValueError(
                "Direct-launch time-reversal treatment differs from resolved configuration; "
                f"a forwarded Hydra override contradicted launcher-owned metadata: {mismatches!r}."
            )

    declared_curriculum = context.get("command_curriculum")
    if declared_curriculum is None:
        if context.get("schema_version") == 2:
            raise ValueError("Direct-launch schema 2 requires command_curriculum metadata.")
        return
    if not isinstance(declared_curriculum, Mapping):
        raise ValueError("Direct-launch command_curriculum must be a mapping.")
    unexpected = sorted(set(declared_curriculum) - set(_DIRECT_CURRICULUM_CONFIG_FIELDS))
    missing = sorted(set(_DIRECT_CURRICULUM_CONFIG_FIELDS) - set(declared_curriculum))
    if unexpected or (context.get("schema_version") == 2 and missing):
        raise ValueError(
            f"Direct-launch command_curriculum schema fields do not match: missing={missing}, unexpected={unexpected}."
        )

    command_cfg = _lookup(env_config, "commands", "base_velocity", default={})
    if not isinstance(command_cfg, Mapping):
        raise ValueError("Resolved environment base-velocity command configuration must be a mapping.")
    mismatches: dict[str, tuple[Any, Any]] = {}
    for metadata_name, config_name in _DIRECT_CURRICULUM_CONFIG_FIELDS.items():
        if metadata_name not in declared_curriculum:
            continue
        resolved = command_cfg.get(config_name)
        declared = declared_curriculum[metadata_name]
        if not _runtime_values_equal(resolved, declared):
            mismatches[metadata_name] = (declared, resolved)
    if mismatches:
        raise ValueError(
            "Direct-launch command curriculum differs from resolved configuration; "
            f"a forwarded Hydra override contradicted launcher-owned metadata: {mismatches!r}."
        )


def record_resolved_run_metadata(
    *,
    env_cfg: Any,
    agent_cfg: Any,
    log_dir: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    runtime_argv: Sequence[str],
    observation_dimension: int,
    action_dimension: int,
    direct_launch_context: Mapping[str, Any] | str | None = None,
    study_context_path: str | os.PathLike[str] | None = None,
    initialization_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Write immutable resolved-command and result-independent cohort facts.

    These records are emitted for direct ``train.ps1`` launches as well as
    manifest-backed launches.  Initial eligibility is ``incomplete`` because
    neither a terminal checkpoint nor an evaluation protocol exists before the
    first rollout.  A later registry audit may promote or reject the run using
    the recorded protocol facts; reward quality is intentionally absent.
    """
    if isinstance(observation_dimension, bool) or observation_dimension <= 0:
        raise ValueError("observation_dimension must be a positive integer.")
    if isinstance(action_dimension, bool) or action_dimension <= 0:
        raise ValueError("action_dimension must be a positive integer.")
    root = Path(repo_root).resolve()
    run_dir = Path(log_dir).resolve()
    env_config = _config_dict(env_cfg)
    agent_config = _config_dict(agent_cfg)
    git = git_provenance(root)
    direct_context = _parse_direct_launch_context(direct_launch_context)
    _validate_direct_launch_context(direct_context, env_config, agent_config)
    study_context = _load_context(study_context_path)
    policy_contract = resolve_policy_contract(env_config, agent_config, observation_dimension)
    if policy_contract is not None:
        declared_contract = direct_context.get("policy_contract") if direct_context is not None else None
        if declared_contract is not None:
            if not isinstance(declared_contract, Mapping):
                raise ValueError("Direct-launch policy_contract must be a mapping.")
            mismatches = {
                name: (declared_contract.get(name), policy_contract[name])
                for name in declared_contract
                if name in policy_contract and declared_contract.get(name) != policy_contract[name]
            }
            if mismatches:
                raise ValueError(f"Direct-launch policy contract differs from resolved configuration: {mismatches!r}.")
        policy_record = {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "policy_observation_contract",
            "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **policy_contract,
        }
        _write_immutable_record(run_dir / POLICY_CONTRACT_RELATIVE_PATH, policy_record)

    command_record = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "resolved_training_command",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "direct_launch": direct_context,
        "runtime_command": [sys.executable, *sys.argv],
        "runtime_training_argv": list(runtime_argv),
        "resolved_env_config_sha256": sha256_value(env_config),
        "resolved_agent_config_sha256": sha256_value(agent_config),
        "repo_commit": git["repo_commit"],
        "dirty_tree_diff_sha256": git["dirty_tree_diff_sha256"],
    }
    command_path = run_dir / RESOLVED_COMMAND_RELATIVE_PATH
    _write_immutable_record(command_path, command_record)

    initialization = Path(initialization_path).resolve() if initialization_path is not None else None
    initialization_digest = (
        sha256_file(initialization) if initialization is not None and initialization.is_file() else None
    )
    gait_cfg = _lookup(env_config, "commands", "base_velocity", default={})
    reward_profile = {"resolved_rewards": env_config.get("rewards")}
    training_budget = {
        "max_iterations": agent_config.get("max_iterations"),
        "num_steps_per_env": agent_config.get("num_steps_per_env"),
        "num_envs": _lookup(env_config, "scene", "num_envs"),
    }
    cohort_record = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "cohort_eligibility_facts",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "eligibility_status": "incomplete",
        "eligibility_is_result_independent": True,
        "result_quality_used_for_eligibility": False,
        "study_name": study_context.get("study_name") if study_context is not None else None,
        "run_name": agent_config.get("run_name"),
        "seed": agent_config.get("seed"),
        "code_and_source_hashes": {
            "repo_commit": git["repo_commit"],
            "dirty_tree_diff_sha256": git["dirty_tree_diff_sha256"],
        },
        "reward_profile_sha256": sha256_value(reward_profile),
        "gait_library_sha256": sha256_value(gait_cfg),
        "gait_library": gait_cfg,
        "observation_dimension": observation_dimension,
        "action_dimension": action_dimension,
        "training_budget": training_budget,
        "training_budget_sha256": sha256_value(training_budget),
        "initialization_record": str(initialization) if initialization is not None else None,
        "initialization_record_sha256": initialization_digest,
        "evaluation_protocol": None,
        "checkpoint_status": "pending",
        "direct_launch": direct_context,
        "study_context": study_context,
    }
    cohort_path = run_dir / COHORT_METADATA_RELATIVE_PATH
    _write_immutable_record(cohort_path, cohort_record)
    return command_path, cohort_path


def resolve_policy_contract(
    env_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    policy_input_dimension: int,
) -> dict[str, Any] | None:
    """Resolve and validate the task-local policy observation contract.

    Args:
        env_config: Resolved environment configuration.
        agent_config: Resolved RSL-RL agent configuration.
        policy_input_dimension: Actor/critic policy input width.

    Returns:
        JSON-ready policy contract metadata, or ``None`` for unrelated tasks.
    """
    symmetry = _lookup(agent_config, "algorithm", "symmetry_cfg", default={})
    if not isinstance(symmetry, Mapping):
        return None
    version_id = symmetry.get("observation_contract_version")
    if version_id is None:
        return None
    if not isinstance(version_id, str) or not version_id:
        raise ValueError("observation_contract_version must be a nonempty string.")

    frame_dimension = symmetry.get("instantaneous_frame_dim")
    history_enabled = symmetry.get("history_enabled")
    history_length = symmetry.get("history_length")
    history_packing = symmetry.get("history_packing")
    legacy_history_mode = symmetry.get("history_trs_mode")
    if legacy_history_mode is not None:
        legacy_modes = {"framewise_feature": "framewise_feature_approx", "none": "none"}
        if legacy_history_mode not in legacy_modes:
            raise ValueError(
                "history_trs_mode is deprecated and must be 'framewise_feature', 'none', or None; "
                f"received {legacy_history_mode!r}."
            )
        tr_consistency_mode = legacy_modes[legacy_history_mode]
    else:
        tr_consistency_mode = symmetry.get("tr_consistency_mode")
    tr_consistency_mapping_version = symmetry.get("tr_consistency_mapping_version")
    action_history_length = symmetry.get("action_history_length")
    candidate_max_age_updates = symmetry.get("candidate_max_age_updates")
    allowed_policy_version_span = symmetry.get("allowed_policy_version_span")
    actor_alignment = TR_SEQUENCE_ACTOR_ALIGNMENT
    value_alignment = TR_SEQUENCE_VALUE_ALIGNMENT
    if isinstance(frame_dimension, bool) or not isinstance(frame_dimension, int) or frame_dimension <= 0:
        raise ValueError(f"instantaneous_frame_dim must be a positive integer; received {frame_dimension!r}.")
    if not isinstance(history_enabled, bool):
        raise ValueError(f"history_enabled must be boolean; received {history_enabled!r}.")
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 0:
        raise ValueError(f"history_length must be a nonnegative integer; received {history_length!r}.")
    if history_enabled != (history_length > 0):
        raise ValueError("history_enabled must be true exactly when history_length is positive.")
    if not isinstance(history_packing, str) or not history_packing:
        raise ValueError("history_packing must be a nonempty string.")
    if tr_consistency_mode not in TR_CONSISTENCY_MODES:
        raise ValueError(
            f"tr_consistency_mode must be one of {sorted(TR_CONSISTENCY_MODES)}; received {tr_consistency_mode!r}."
        )
    if tr_consistency_mapping_version != TR_CONSISTENCY_MAPPING_VERSION:
        raise ValueError(
            "tr_consistency_mapping_version must identify the exact causal mapping: "
            f"expected {TR_CONSISTENCY_MAPPING_VERSION!r}, received {tr_consistency_mapping_version!r}."
        )
    if action_history_length != POLICY_ACTION_HISTORY_LENGTH:
        raise ValueError(
            f"action_history_length must be {POLICY_ACTION_HISTORY_LENGTH}; received {action_history_length!r}."
        )
    for name, value in (
        ("candidate_max_age_updates", candidate_max_age_updates),
        ("allowed_policy_version_span", allowed_policy_version_span),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")

    env_history_length = _lookup(env_config, "observations", "policy", "history_length", default=None)
    if env_history_length != history_length:
        raise ValueError(
            "Environment and algorithm policy-history settings differ: "
            f"env history_length={env_history_length!r}, algorithm history_length={history_length!r}."
        )
    expected_width = frame_dimension * (history_length if history_enabled else 1)
    if policy_input_dimension != expected_width:
        required_flags = f"--history --history-length {history_length}" if history_enabled else "--no-history"
        raise ValueError(
            "Active policy observation width is incompatible with the resolved observation contract: "
            f"expected {expected_width}, received {policy_input_dimension}. "
            f"Use {required_flags} with contract {version_id!r}."
        )
    sequence_history_length = history_length if history_enabled else 1
    return {
        "observation_contract_version": version_id,
        "instantaneous_frame_dim": frame_dimension,
        "history_enabled": history_enabled,
        "history_length": history_length,
        "history_packing": history_packing,
        "tr_consistency_mode": tr_consistency_mode,
        "tr_consistency_mapping_version": tr_consistency_mapping_version,
        "action_history_length": action_history_length,
        "sequence_history_length": sequence_history_length,
        "required_sequence_records": sequence_history_length + action_history_length,
        "candidate_max_age_updates": candidate_max_age_updates,
        "allowed_policy_version_span": allowed_policy_version_span,
        "actor_alignment": actor_alignment,
        "value_alignment": value_alignment,
        "policy_input_dim": policy_input_dimension,
        "gait_phase_mapping_version": symmetry.get("gait_phase_mapping_version"),
        "gait_library_version": symmetry.get("gait_library_version"),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
        return result
    return {prefix: value}


def _declared(path: str, treatments: Sequence[str]) -> bool:
    return any(path == treatment or path.startswith(f"{treatment}.") for treatment in treatments)


def _runtime_treatment_paths(treatments: Sequence[str], conditions: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Map declared manifest treatments to their exact resolved-config paths."""
    paths = {"environment": set(), "agent": set()}
    for logical_path, runtime_paths in _TREATMENT_RUNTIME_PATHS.items():
        if _declared(logical_path, treatments):
            for config_name, runtime_path in runtime_paths:
                paths[config_name].add(runtime_path)
    if _declared("reward_profile.overrides", treatments):
        for condition in conditions:
            reward_profile = condition.get("reward_profile", {})
            overrides = reward_profile.get("overrides", []) if isinstance(reward_profile, Mapping) else []
            for override in overrides:
                if not isinstance(override, str) or "=" not in override:
                    continue
                hydra_path = override.split("=", 1)[0]
                if hydra_path.startswith("env."):
                    paths["environment"].add(hydra_path.removeprefix("env."))
    return paths


def _masked_config(config: Mapping[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """Replace selected dotted paths with stable treatment markers."""
    masked = copy.deepcopy(dict(config))
    for path in sorted(set(paths)):
        keys = path.split(".")
        cursor: Any = masked
        for key in keys[:-1]:
            if not isinstance(cursor, Mapping) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, dict) and keys[-1] in cursor:
            cursor[keys[-1]] = {"__masked_resolved_config_path__": path}
    return masked


def _resolved_config_invariant_hashes(
    record: Mapping[str, Any], treatment_paths: Mapping[str, set[str]]
) -> dict[str, str] | None:
    resolved_configs = record.get("resolved_configs")
    if not isinstance(resolved_configs, Mapping):
        return None
    result: dict[str, str] = {}
    for config_name in ("environment", "agent"):
        config = resolved_configs.get(config_name)
        if not isinstance(config, Mapping):
            return None
        paths = [*_ALWAYS_MASKED_CONFIG_PATHS[config_name], *treatment_paths[config_name]]
        result[config_name] = sha256_value(_masked_config(config, paths))
    return result


def _validate_resolved_configs(record: Mapping[str, Any], run_name: str, errors: list[str]) -> None:
    resolved_configs = record.get("resolved_configs")
    if not isinstance(resolved_configs, Mapping):
        errors.append(f"{run_name}: initialization record is missing resolved_configs")
        return
    for config_name, hash_field in (
        ("environment", "resolved_env_config_sha256"),
        ("agent", "resolved_agent_config_sha256"),
    ):
        config = resolved_configs.get(config_name)
        if not isinstance(config, Mapping):
            errors.append(f"{run_name}: resolved {config_name} config is missing")
        elif sha256_value(config) != record.get(hash_field):
            errors.append(f"{run_name}: resolved {config_name} config digest does not match {hash_field}")


def _validate_matched_signatures(record: Mapping[str, Any], run_name: str, errors: list[str]) -> None:
    signatures = record.get("matched_signatures")
    if not isinstance(signatures, Mapping):
        errors.append(f"{run_name}: initialization record is missing matched_signatures")
        return
    for field in _STRICT_SIGNATURES:
        if signatures.get(field) is None:
            errors.append(f"{run_name}: initialization record is missing matched signature {field}")


def _validate_asset_provenance(
    record: Mapping[str, Any], run_name: str, errors: list[str], warnings: list[str]
) -> None:
    asset_provenance = record.get("robot_asset_hashes")
    if not isinstance(asset_provenance, Mapping) or asset_provenance.get("sha256") is None:
        errors.append(f"{run_name}: initialization record is missing robot asset hashes")
        asset_provenance = {}
    if not asset_provenance.get("reference_count"):
        errors.append(f"{run_name}: initialization record contains no robot asset references")
    assets = asset_provenance.get("assets", [])
    if not isinstance(assets, list) or sha256_value(assets) != asset_provenance.get("sha256"):
        errors.append(f"{run_name}: robot asset aggregate digest is invalid")
        assets = []
    if asset_provenance.get("reference_count") != len(assets):
        errors.append(f"{run_name}: robot asset reference count is inconsistent")
    for asset in assets:
        if not isinstance(asset, Mapping) or asset.get("status") == "hashed":
            continue
        detail = (
            f"{run_name}: {asset.get('status', 'unknown')} asset provenance for "
            f"{asset.get('reference', '<missing reference>')}"
        )
        warnings.append(detail)
        errors.append(f"{detail}; publication cohorts require content-hashed robot assets")


def _validate_study_context_binding(
    record: Mapping[str, Any],
    run: Mapping[str, Any],
    run_name: str,
    condition: Mapping[str, Any] | None,
    treatments: list[str],
    manifest: Mapping[str, Any],
    errors: list[str],
) -> None:
    context = record.get("study_context")
    publication_manifest = all(
        manifest.get(field) is not None for field in ("study_name", "study_sha256", "expected_branch")
    )
    if publication_manifest and not isinstance(context, Mapping):
        errors.append(f"{run_name}: publication initialization record is missing study_context")
    if not isinstance(context, Mapping):
        return
    if context.get("treatment_variables") != treatments:
        errors.append(f"{run_name}: study-context treatment variables differ from the cohort manifest")
    if condition is not None and context.get("resolved_condition") != condition:
        errors.append(f"{run_name}: study-context condition differs from the cohort manifest")
    if context.get("run_name") != run_name:
        errors.append(f"{run_name}: study-context run_name differs from the cohort manifest")
    for field in ("study_name", "study_sha256", "expected_branch"):
        if manifest.get(field) is not None and context.get(field) != manifest.get(field):
            errors.append(f"{run_name}: study-context {field} differs from the cohort manifest")
    if condition is not None:
        expected_condition_sha256 = sha256_value(condition)
        if run.get("condition_sha256") != expected_condition_sha256:
            errors.append(f"{run_name}: cohort condition_sha256 does not match resolved_condition")
        if context.get("condition_sha256") != expected_condition_sha256:
            errors.append(f"{run_name}: study-context condition_sha256 does not match resolved_condition")
        resolved_configs = record.get("resolved_configs", {})
        try:
            if not isinstance(resolved_configs, Mapping):
                raise ValueError("resolved configuration snapshots are missing")
            environment = resolved_configs.get("environment")
            agent = resolved_configs.get("agent")
            if not isinstance(environment, Mapping) or not isinstance(agent, Mapping):
                raise ValueError("resolved environment or agent configuration is missing")
            _validate_condition_runtime_mapping(condition, environment, agent)
        except ValueError as exc:
            errors.append(f"{run_name}: {exc}")
    if context.get("exact_training_command") != run.get("exact_training_command"):
        errors.append(f"{run_name}: study-context exact training command differs from the cohort manifest")
    try:
        expected_argv = _expected_runtime_argv(context.get("exact_training_command"))
    except ValueError as exc:
        errors.append(f"{run_name}: invalid study-context exact training command: {exc}")
    else:
        if record.get("expected_training_argv") != expected_argv:
            errors.append(f"{run_name}: recorded expected training argv differs from the study context")
        if record.get("runtime_training_argv") != expected_argv:
            errors.append(f"{run_name}: recorded runtime training argv differs from the exact training command")
    if "evaluation_command_template" in run and context.get("evaluation_command_template") != run.get(
        "evaluation_command_template"
    ):
        errors.append(f"{run_name}: study-context evaluation command differs from the cohort manifest")
    if run.get("study_context") is not None:
        context_path = Path(str(context.get("context_path", ""))).resolve()
        if context_path != Path(str(run["study_context"])).resolve():
            errors.append(f"{run_name}: study-context path differs from the cohort manifest")
    git = record.get("git", {})
    if isinstance(git, Mapping) and git.get("repo_branch") != context.get("expected_branch"):
        errors.append(f"{run_name}: recorded repository branch differs from the expected branch")


def _validate_record_seed(
    record: Mapping[str, Any], condition: Mapping[str, Any] | None, run_name: str, errors: list[str]
) -> None:
    if condition is None or "seed" not in condition:
        return
    declared_seed = condition["seed"]
    for field in ("seed", "training_seed", "environment_seed"):
        if record.get(field) != declared_seed:
            errors.append(f"{run_name}: {field} differs from resolved_condition.seed")


def _validate_run_initialization(
    run: Any,
    index: int,
    manifest_path: Path | None,
    treatments: list[str],
    manifest: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None, dict[str, Any] | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(run, Mapping):
        run_name = f"run[{index}]"
        return run_name, None, None, [f"{run_name}: run entry must be an object"], warnings
    run_name = str(run.get("run_name", f"run[{index}]"))
    condition = run.get("resolved_condition")
    if not isinstance(condition, Mapping):
        errors.append(f"{run_name}: missing resolved_condition")
        condition = None
    record_value = run.get("initialization_record")
    if record_value is None:
        errors.append(f"{run_name}: missing initialization_record (status={run.get('status', 'unknown')})")
        return run_name, condition, None, errors, warnings
    record_path = Path(record_value)
    if not record_path.is_absolute() and manifest_path is not None:
        record_path = manifest_path.parent / record_path
    try:
        with record_path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        if not isinstance(record, dict):
            raise ValueError("initialization record must be a JSON object")
        _validate_record_digest(record, record_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{run_name}: invalid initialization record: {exc}")
        return run_name, condition, None, errors, warnings
    if record.get("record_kind") != "pre_first_rollout_initialization":
        errors.append(f"{run_name}: initialization record has an unsupported record_kind")
    for field in _STRICT_MATCH_FIELDS:
        if record.get(field) is None:
            errors.append(f"{run_name}: initialization record is missing {field}")
    _validate_resolved_configs(record, run_name, errors)
    _validate_matched_signatures(record, run_name, errors)
    _validate_asset_provenance(record, run_name, errors, warnings)
    _validate_study_context_binding(record, run, run_name, condition, treatments, manifest, errors)
    _validate_record_seed(record, condition, run_name, errors)
    resume = record.get("resume")
    if isinstance(resume, Mapping) and resume.get("source_initialization_status") != "validated":
        warnings.append(
            f"{run_name}: resume source initialization status is "
            f"{resume.get('source_initialization_status', 'unknown')}"
        )
    return run_name, condition, record, errors, warnings


def validate_cohort(manifest_or_path: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    """Validate that a manifest's retained runs form a matched cohort."""
    manifest_path: Path | None = None
    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
    else:
        manifest_path = Path(manifest_or_path).resolve()
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    treatments = manifest.get("treatment_variables", [])
    runs = manifest.get("runs", [])
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(treatments, list) or not all(isinstance(item, str) and item for item in treatments):
        errors.append("treatment_variables must be a list of nonempty dotted paths")
        treatments = []
    if not isinstance(runs, list) or not runs:
        return {
            "matched": False,
            "errors": [*errors, "cohort has no retained runs"],
            "warnings": warnings,
            "run_count": 0,
        }

    logical = []
    logical_conditions: list[tuple[str, Mapping[str, Any]]] = []
    records: list[tuple[str, dict[str, Any]]] = []
    for index, run in enumerate(runs):
        run_name, condition, record, run_errors, run_warnings = _validate_run_initialization(
            run, index, manifest_path, treatments, manifest
        )
        errors.extend(run_errors)
        warnings.extend(run_warnings)
        if condition is not None:
            logical.append((run_name, _flatten(condition)))
            logical_conditions.append((run_name, condition))
        if record is not None:
            records.append((run_name, record))

    if logical:
        all_paths = sorted(set().union(*(values.keys() for _, values in logical)))
        for path in all_paths:
            values = {canonical_json(mapping.get(path, {"__missing__": True})) for _, mapping in logical}
            if len(values) > 1 and not _declared(path, treatments):
                errors.append(f"undeclared treatment difference at {path}")

    if records:
        treatment_paths = _runtime_treatment_paths(treatments, [condition for _, condition in logical_conditions])
        treatment_signatures = set()
        if _declared("reward_profile.overrides", treatments) or _declared("leg_sync_reward_weight", treatments):
            treatment_signatures.add("reward_profile")
        if _declared("training_iterations", treatments) or _declared("num_envs", treatments):
            treatment_signatures.add("training_budget")
        reference_name, reference = records[0]
        reference_config_hashes = _resolved_config_invariant_hashes(reference, treatment_paths)
        for run_name, record in records[1:]:
            for field in _STRICT_MATCH_FIELDS:
                if record.get(field) != reference.get(field):
                    errors.append(f"{run_name}: {field} differs from {reference_name}")
            if record.get("robot_asset_hashes", {}).get("sha256") != reference.get("robot_asset_hashes", {}).get(
                "sha256"
            ):
                errors.append(f"{run_name}: robot asset hashes differ from {reference_name}")
            reference_signatures = reference.get("matched_signatures", {})
            signatures = record.get("matched_signatures", {})
            for field in _STRICT_SIGNATURES:
                if field in treatment_signatures:
                    continue
                if signatures.get(field) != reference_signatures.get(field):
                    errors.append(f"{run_name}: matched signature {field} differs from {reference_name}")
            config_hashes = _resolved_config_invariant_hashes(record, treatment_paths)
            if config_hashes is None or reference_config_hashes is None:
                errors.append(f"{run_name}: resolved configuration snapshots are unavailable for matching")
            else:
                for config_name in ("environment", "agent"):
                    if config_hashes[config_name] != reference_config_hashes[config_name]:
                        errors.append(
                            f"{run_name}: resolved invariant {config_name} config differs from {reference_name}"
                        )
    else:
        treatment_paths = _runtime_treatment_paths(treatments, [condition for _, condition in logical_conditions])
        treatment_signatures = set()
    if len(records) != len(runs):
        errors.append(f"only {len(records)}/{len(runs)} retained runs have valid initialization records")
    return {
        "matched": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_count": len(runs),
        "validated_initialization_records": len(records),
        "strict_fields": list(_STRICT_MATCH_FIELDS),
        "strict_signatures": list(_STRICT_SIGNATURES),
        "treatment_variables": treatments,
        "runtime_treatment_paths": {key: sorted(value) for key, value in treatment_paths.items()},
        "treatment_masked_signatures": sorted(treatment_signatures),
        "always_masked_config_paths": {key: list(value) for key, value in _ALWAYS_MASKED_CONFIG_PATHS.items()},
    }


def main(argv: list[str] | None = None) -> int:
    """Validate a cohort manifest from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Cohort manifest JSON to validate.")
    args = parser.parse_args(argv)
    result = validate_cohort(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
