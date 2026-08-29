# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch a deterministic, manifest-defined symmetric-locomotion study."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_NUM_ENVS = 512
SCHEDULE_DEFAULTS = {
    "enabled": True,
    "warmup_iterations": 0,
    "rampup_iterations": 0,
    "hold_iterations": 0,
    "decay_iterations": 0,
    "final_scale": 1.0,
    "ramp_shape": "linear",
}
VALIDITY_DEFAULTS = {
    "mode": "command",
    "thresholds": {
        "min_abs_command_velocity": None,
        "tracking_abs_tolerance": 0.25,
        "tracking_rel_tolerance": 0.25,
        "projected_gravity_tolerance": 0.35,
        "phase_boundary_margin": 0.03,
        "command_speed_bin_edges": [0.5, 1.0, 1.5],
    },
}
AUGMENTATION_DEFAULTS = {
    "enabled": False,
    "mode": "dynamics_filtered_reverse_action_supervision",
    "action_source": "analytic",
    "coefficient": 0.0,
    "schedule": SCHEDULE_DEFAULTS,
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
}
REQUIRED_FIELDS = {
    "robot",
    "seed",
    "tr_policy",
    "tr_value",
    "validity_mask",
    "trajectory_augmentation",
    "reward_profile",
    "leg_sync_reward_weight",
    "training_iterations",
    "evaluation_protocol",
}
RESUMABLE_STATUSES = {"planned", "running", "failed", "interrupted"}
PLAN_IDENTITY_FIELDS = (
    "schema_version",
    "study_name",
    "study_sha256",
    "source_manifest",
    "expected_branch",
    "treatment_variables",
    "selection_policy",
)
RUN_IDENTITY_FIELDS = (
    "label",
    "run_name",
    "condition_sha256",
    "resolved_condition",
    "study_context",
    "exact_training_command",
    "exact_training_command_text",
    "evaluation_command_template",
)
VALIDITY_MODES = {
    "command",
    "command_tracking",
    "command_upright_phase",
    "command_tracking_upright_phase",
}
AUGMENTATION_ACTION_SOURCES = {"analytic", "learned", "learned_inverse"}


def _load_adjacent(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_symm_study_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROVENANCE = _load_adjacent("training_provenance")
SYMM_CLI = _load_adjacent("symm_cli")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _finite_nonnegative(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{path} must be finite and nonnegative; received {value!r}.")
    return float(value)


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer; received {value!r}.")
    return value


def _normalize_schedule(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    unknown = set(value) - set(SCHEDULE_DEFAULTS)
    if unknown:
        raise ValueError(f"{path} has unsupported fields: {sorted(unknown)}")
    schedule = _deep_merge(SCHEDULE_DEFAULTS, value)
    if not isinstance(schedule["enabled"], bool):
        raise ValueError(f"{path}.enabled must be boolean.")
    for field in ("warmup_iterations", "rampup_iterations", "hold_iterations", "decay_iterations"):
        schedule[field] = _nonnegative_integer(schedule[field], f"{path}.{field}")
    schedule["final_scale"] = _finite_nonnegative(schedule["final_scale"], f"{path}.final_scale")
    if schedule["final_scale"] > 1.0:
        raise ValueError(f"{path}.final_scale must not exceed 1.0.")
    if schedule["ramp_shape"] not in {"linear", "half_cosine"}:
        raise ValueError(f"{path}.ramp_shape must be 'linear' or 'half_cosine'.")
    return schedule


def _normalize_term(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    unknown = set(value) - {"enabled", "coefficient", "schedule"}
    if unknown:
        raise ValueError(f"{path} has unsupported fields: {sorted(unknown)}")
    if "enabled" not in value or "coefficient" not in value:
        raise ValueError(f"{path} requires enabled and coefficient.")
    if not isinstance(value["enabled"], bool):
        raise ValueError(f"{path}.enabled must be boolean.")
    return {
        "enabled": value["enabled"],
        "coefficient": _finite_nonnegative(value["coefficient"], f"{path}.coefficient"),
        "schedule": _normalize_schedule(value.get("schedule"), f"{path}.schedule"),
    }


def _normalize_validity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("validity_mask must be an object.")
    unknown = set(value) - {"mode", "thresholds"}
    if unknown:
        raise ValueError(f"validity_mask has unsupported fields: {sorted(unknown)}")
    validity = _deep_merge(VALIDITY_DEFAULTS, value)
    if validity["mode"] not in VALIDITY_MODES:
        raise ValueError(f"validity_mask.mode must be one of {sorted(VALIDITY_MODES)}.")
    thresholds = validity["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("validity_mask.thresholds must be an object.")
    unknown_thresholds = set(thresholds) - set(VALIDITY_DEFAULTS["thresholds"])
    if unknown_thresholds:
        raise ValueError(f"validity_mask.thresholds has unsupported fields: {sorted(unknown_thresholds)}")
    normalized_thresholds = dict(thresholds)
    for key, item in normalized_thresholds.items():
        if key == "command_speed_bin_edges":
            if not isinstance(item, Sequence) or isinstance(item, str | bytes):
                raise ValueError("validity_mask.thresholds.command_speed_bin_edges must be a sequence.")
            edges = [_finite_nonnegative(edge, f"validity_mask.thresholds.{key}") for edge in item]
            if edges != sorted(set(edges)):
                raise ValueError("validity_mask command_speed_bin_edges must be strictly increasing.")
            normalized_thresholds[key] = edges
        elif item is not None:
            normalized_thresholds[key] = _finite_nonnegative(item, f"validity_mask.thresholds.{key}")
    if normalized_thresholds["phase_boundary_margin"] > 0.5:
        raise ValueError("validity_mask phase_boundary_margin must not exceed 0.5 cycles.")
    return {"mode": validity["mode"], "thresholds": normalized_thresholds}


def _normalize_augmentation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("trajectory_augmentation must be an object.")
    unknown = set(value) - set(AUGMENTATION_DEFAULTS)
    if unknown:
        raise ValueError(f"trajectory_augmentation has unsupported fields: {sorted(unknown)}")
    augmentation = _deep_merge(AUGMENTATION_DEFAULTS, value)
    if not isinstance(augmentation["enabled"], bool):
        raise ValueError("trajectory_augmentation.enabled must be boolean.")
    if augmentation["mode"] != "dynamics_filtered_reverse_action_supervision":
        raise ValueError("trajectory_augmentation.mode must be 'dynamics_filtered_reverse_action_supervision'.")
    if augmentation["action_source"] not in AUGMENTATION_ACTION_SOURCES:
        raise ValueError("trajectory_augmentation.action_source must be analytic, learned, or learned_inverse.")
    if augmentation["action_source"] == "learned":
        augmentation["action_source"] = "learned_inverse"
    augmentation["coefficient"] = _finite_nonnegative(
        augmentation["coefficient"], "trajectory_augmentation.coefficient"
    )
    augmentation["schedule"] = _normalize_schedule(augmentation["schedule"], "trajectory_augmentation.schedule")
    filters = augmentation["filter_settings"]
    if not isinstance(filters, Mapping):
        raise ValueError("trajectory_augmentation.filter_settings must be an object.")
    unknown_filters = set(filters) - set(AUGMENTATION_DEFAULTS["filter_settings"])
    if unknown_filters:
        raise ValueError(f"trajectory_augmentation.filter_settings has unsupported fields: {sorted(unknown_filters)}")
    normalized_filters = dict(filters)
    for field in ("filter_enabled", "use_confidence_weights"):
        if not isinstance(normalized_filters[field], bool):
            raise ValueError(f"trajectory_augmentation.filter_settings.{field} must be boolean.")
    minimum_samples = normalized_filters["minimum_validation_samples"]
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples <= 0:
        raise ValueError(
            "trajectory_augmentation.filter_settings.minimum_validation_samples must be a positive integer."
        )
    normalized_filters["rng_seed"] = _nonnegative_integer(
        normalized_filters["rng_seed"], "trajectory_augmentation.filter_settings.rng_seed"
    )
    for field in (
        "validation_fraction",
        "validation_quantile",
        "threshold_multiplier",
        "maximum_validation_loss",
        "phase_boundary_margin",
        "maximum_contact_impulse",
        "action_abs_limit",
        "max_augmented_to_original_ratio",
    ):
        normalized_filters[field] = _finite_nonnegative(
            normalized_filters[field], f"trajectory_augmentation.filter_settings.{field}"
        )
    if not 0.0 < normalized_filters["validation_fraction"] < 1.0:
        raise ValueError("trajectory_augmentation.filter_settings.validation_fraction must be in (0, 1).")
    if not 0.0 < normalized_filters["validation_quantile"] <= 1.0:
        raise ValueError("trajectory_augmentation.filter_settings.validation_quantile must be in (0, 1].")
    if normalized_filters["phase_boundary_margin"] > 0.5:
        raise ValueError("trajectory_augmentation filter phase_boundary_margin must not exceed 0.5 cycles.")
    if normalized_filters["max_augmented_to_original_ratio"] > 1.0:
        raise ValueError("trajectory_augmentation max_augmented_to_original_ratio must not exceed 1.0.")
    if augmentation["enabled"] and not normalized_filters["filter_enabled"]:
        raise ValueError("Enabled trajectory augmentation requires filter_settings.filter_enabled=true.")
    augmentation["filter_settings"] = normalized_filters
    return augmentation


def _normalize_reward_profile(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"name": value, "overrides": []}
    if not isinstance(value, Mapping) or set(value) != {"name", "overrides"}:
        raise ValueError("reward_profile must be a string or an object with exactly name and overrides.")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise ValueError("reward_profile.name must be a nonempty string.")
    if not isinstance(value["overrides"], list) or not all(isinstance(item, str) for item in value["overrides"]):
        raise ValueError("reward_profile.overrides must be a list of Hydra override strings.")
    by_key: dict[str, str] = {}
    for override in value["overrides"]:
        if "=" not in override:
            raise ValueError(f"Reward override must contain '=': {override!r}")
        key = override.split("=", 1)[0]
        if not key.startswith("env.rewards."):
            raise ValueError(f"Reward overrides are scoped to env.rewards.*: {override!r}")
        if key == "env.rewards.leg_permutation_symmetry.weight":
            raise ValueError("Use leg_sync_reward_weight instead of overriding its reward-profile field.")
        if key in by_key:
            raise ValueError(f"Duplicate reward override key: {key}")
        by_key[key] = override
    return {"name": value["name"].strip(), "overrides": [by_key[key] for key in sorted(by_key)]}


def normalize_condition(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one fully merged study condition."""
    missing = REQUIRED_FIELDS - set(value)
    if missing:
        raise ValueError(f"Condition is missing required fields: {sorted(missing)}")
    unknown = set(value) - REQUIRED_FIELDS - {"num_envs"}
    if unknown:
        raise ValueError(f"Condition has unsupported fields: {sorted(unknown)}")
    robot = SYMM_CLI.get_robot(value["robot"]).key
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a nonnegative integer; received {seed!r}.")
    iterations = value["training_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("training_iterations must be a positive integer.")
    num_envs = value.get("num_envs", DEFAULT_NUM_ENVS)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer.")
    if value["evaluation_protocol"] not in {"light", "full"}:
        raise ValueError("evaluation_protocol must be 'light' or 'full'.")
    return {
        "robot": robot,
        "seed": seed,
        "tr_policy": _normalize_term(value["tr_policy"], "tr_policy"),
        "tr_value": _normalize_term(value["tr_value"], "tr_value"),
        "validity_mask": _normalize_validity(value["validity_mask"]),
        "trajectory_augmentation": _normalize_augmentation(value["trajectory_augmentation"]),
        "reward_profile": _normalize_reward_profile(value["reward_profile"]),
        "leg_sync_reward_weight": _finite_nonnegative(value["leg_sync_reward_weight"], "leg_sync_reward_weight"),
        "training_iterations": iterations,
        "num_envs": num_envs,
        "evaluation_protocol": value["evaluation_protocol"],
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"Value has no filename-safe characters: {value!r}")
    return slug


def deterministic_run_name(study_name: str, label: str, condition: Mapping[str, Any]) -> str:
    """Return a readable run name with a canonical condition digest."""
    digest = PROVENANCE.sha256_value(condition)[:12]
    return (
        f"{_slug(study_name)}-{_slug(label)}-{condition['robot']}-s{condition['seed']}-"
        f"{condition['evaluation_protocol']}-{digest}"
    )


def _training_behavior_identity(condition: Mapping[str, Any]) -> dict[str, Any]:
    """Return condition identity with provenance-only reward labels removed."""
    identity = copy.deepcopy(dict(condition))
    reward_profile = identity.get("reward_profile")
    if isinstance(reward_profile, dict):
        reward_profile.pop("name", None)
    return identity


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def _validate_treatments(conditions: Sequence[Mapping[str, Any]], treatments: Sequence[str]) -> None:
    if not all(isinstance(path, str) and path for path in treatments):
        raise ValueError("treatment_variables must contain nonempty dotted paths.")
    if len(set(treatments)) != len(treatments):
        raise ValueError("treatment_variables contains duplicates.")
    flattened = [_flatten(condition) for condition in conditions]
    all_paths = sorted(set().union(*(condition.keys() for condition in flattened)))
    differences = {
        path
        for path in all_paths
        if len({PROVENANCE.canonical_json(condition.get(path, {"__missing__": True})) for condition in flattened}) > 1
    }
    undeclared = sorted(
        path
        for path in differences
        if not any(path == treatment or path.startswith(f"{treatment}.") for treatment in treatments)
    )
    if undeclared:
        raise ValueError(f"Condition differences are not declared treatments: {undeclared}")
    unused = sorted(
        treatment
        for treatment in treatments
        if not any(path == treatment or path.startswith(f"{treatment}.") for path in differences)
    )
    if unused:
        raise ValueError(f"Declared treatment variables do not vary: {unused}")


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list | tuple):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _schedule_overrides(prefix: str, enabled: bool, coefficient: float, schedule: Mapping[str, Any]) -> list[str]:
    values = {**schedule, "enabled": enabled and schedule["enabled"], "target_coeff": coefficient}
    return [f"{prefix}.{key}={_hydra_value(values[key])}" for key in sorted(values)]


def condition_hydra_overrides(condition: Mapping[str, Any]) -> list[str]:
    """Translate one normalized condition into exact Hydra overrides."""
    root = "agent.algorithm.symmetry_cfg"
    policy = condition["tr_policy"]
    value = condition["tr_value"]
    augmentation = condition["trajectory_augmentation"]
    enabled = policy["enabled"] or value["enabled"] or augmentation["enabled"]
    overrides = [
        f"{root}.use_time_reversal_regularization={_hydra_value(enabled)}",
        f"{root}.use_data_augmentation=false",
        f"{root}.use_mirror_loss={_hydra_value(policy['enabled'])}",
        f"{root}.mirror_loss_coeff={_hydra_value(policy['coefficient'])}",
        f"{root}.value_loss_coeff={_hydra_value(value['coefficient'])}",
        f"{root}.use_tr_policy_consistency={_hydra_value(policy['enabled'])}",
        f"{root}.use_tr_value_consistency={_hydra_value(value['enabled'])}",
    ]
    overrides += _schedule_overrides(
        f"{root}.tr_policy_schedule", policy["enabled"], policy["coefficient"], policy["schedule"]
    )
    overrides += _schedule_overrides(
        f"{root}.tr_value_schedule", value["enabled"], value["coefficient"], value["schedule"]
    )
    validity = condition["validity_mask"]
    overrides.append(f"{root}.tr_validity.mode={validity['mode']}")
    overrides += [
        f"{root}.tr_validity.{key}={_hydra_value(value)}" for key, value in sorted(validity["thresholds"].items())
    ]
    overrides += [
        f"{root}.tr_augmentation.enabled={_hydra_value(augmentation['enabled'])}",
        f"{root}.tr_augmentation.mode={augmentation['mode']}",
        f"{root}.tr_augmentation.action_source={augmentation['action_source']}",
        f"{root}.tr_augmentation.coefficient={_hydra_value(augmentation['coefficient'])}",
    ]
    overrides += _schedule_overrides(
        f"{root}.tr_augmentation.schedule",
        augmentation["enabled"],
        augmentation["coefficient"],
        augmentation["schedule"],
    )
    overrides += [
        f"{root}.tr_augmentation.{key}={_hydra_value(value)}"
        for key, value in sorted(augmentation["filter_settings"].items())
    ]
    overrides += condition["reward_profile"]["overrides"]
    overrides.append(f"env.rewards.leg_permutation_symmetry.weight={_hydra_value(condition['leg_sync_reward_weight'])}")
    return overrides


def _launcher_namespace(args: argparse.Namespace, robot: str) -> argparse.Namespace:
    return argparse.Namespace(
        conda_env=args.conda_env,
        no_conda_run=args.no_conda_run,
        use_conda_run=args.use_conda_run,
        robot=robot,
        robot_spec=SYMM_CLI.get_robot(robot),
    )


def training_command(
    condition: Mapping[str, Any], run_name: str, context_path: Path, args: argparse.Namespace
) -> list[str]:
    """Build the exact resolved training command for one condition."""
    spec = SYMM_CLI.get_robot(condition["robot"])
    isaaclab_args = [
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        spec.train_task,
        "--num_envs",
        str(condition["num_envs"]),
        "--max_iterations",
        str(condition["training_iterations"]),
        "--run_name",
        run_name,
        "--seed",
        str(condition["seed"]),
        "--symm_study_context",
        str(context_path),
        *condition_hydra_overrides(condition),
    ]
    return SYMM_CLI.isaaclab_command(_launcher_namespace(args, condition["robot"]), isaaclab_args)


def evaluation_command_template(
    condition: Mapping[str, Any], expected_branch: str, args: argparse.Namespace
) -> list[str]:
    """Build a post-training evaluation template without running it."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("symm_cli.py")),
        "evaluation",
        "--robot",
        condition["robot"],
        "--run",
        "{run_dir}",
        "--protocol",
        condition["evaluation_protocol"],
        "--expected_branch",
        expected_branch,
        "--conda_env",
        args.conda_env,
    ]
    if args.no_conda_run:
        command.append("--no_conda_run")
    elif args.use_conda_run:
        command.append("--use_conda_run")
    return command


def _validate_output_dir_is_stable(output_dir: Path) -> None:
    """Reject in-repository study artifacts that would perturb later cohort hashes."""
    repository = SYMM_CLI.repo_root().resolve()
    resolved_output = output_dir.resolve()
    try:
        relative = resolved_output.relative_to(repository)
    except ValueError:
        return
    result = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--quiet", "--", relative.as_posix()],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            "An in-repository --output_dir must be ignored by Git so per-condition study artifacts do not "
            f"change dirty-tree initialization hashes: {resolved_output}"
        )


def load_plan(manifest_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load, validate, and resolve a study manifest without writing files."""
    _validate_output_dir_is_stable(output_dir)
    with manifest_path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    if not isinstance(source, Mapping):
        raise ValueError("Study manifest must be a JSON object.")
    if source.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    study_name = source.get("study_name")
    expected_branch = source.get("expected_branch")
    treatments = source.get("treatment_variables")
    common = source.get("common")
    conditions = source.get("conditions")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("study_name must be a nonempty string.")
    if not isinstance(expected_branch, str) or not expected_branch.strip():
        raise ValueError("expected_branch is required and must be nonempty.")
    if not isinstance(treatments, list):
        raise ValueError("treatment_variables must be an explicit list (which may be empty).")
    if not isinstance(common, Mapping):
        raise ValueError("common must be an object.")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a nonempty list.")
    if SYMM_CLI.git_branch() != expected_branch:
        actual = SYMM_CLI.git_branch() or "<detached HEAD>"
        raise ValueError(f"Expected branch '{expected_branch}', but current branch is '{actual}'.")

    resolved: list[dict[str, Any]] = []
    labels: list[str] = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            raise ValueError(f"conditions[{index}] must be an object.")
        label = condition.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"conditions[{index}].label must be nonempty.")
        labels.append(label)
        resolved.append(
            normalize_condition(_deep_merge(common, {key: value for key, value in condition.items() if key != "label"}))
        )
    canonical_conditions = [PROVENANCE.canonical_json(condition) for condition in resolved]
    if len(set(canonical_conditions)) != len(canonical_conditions):
        duplicates = sorted({value for value in canonical_conditions if canonical_conditions.count(value) > 1})
        raise ValueError(f"Duplicate resolved conditions are forbidden ({len(duplicates)} duplicate identities).")
    behavior_conditions = [PROVENANCE.canonical_json(_training_behavior_identity(condition)) for condition in resolved]
    if len(set(behavior_conditions)) != len(behavior_conditions):
        raise ValueError(
            "Semantically duplicate training conditions are forbidden; reward_profile.name is provenance only."
        )
    _validate_treatments(resolved, treatments)

    runs = []
    run_names: set[str] = set()
    for label, condition in zip(labels, resolved, strict=True):
        run_name = deterministic_run_name(study_name, label, condition)
        if run_name in run_names:
            raise ValueError(f"Duplicate deterministic run name: {run_name}")
        run_names.add(run_name)
        context_path = (output_dir / "contexts" / f"{run_name}.json").resolve()
        command = training_command(condition, run_name, context_path, args)
        runs.append(
            {
                "label": label,
                "run_name": run_name,
                "condition_sha256": PROVENANCE.sha256_value(condition),
                "resolved_condition": condition,
                "study_context": str(context_path),
                "exact_training_command": command,
                "exact_training_command_text": SYMM_CLI.command_to_string(command),
                "evaluation_command_template": evaluation_command_template(condition, expected_branch, args),
                "status": "planned",
                "exit_code": None,
                "run_dir": None,
                "initialization_record": None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_name": study_name,
        "study_sha256": PROVENANCE.sha256_value(source),
        "source_manifest": str(manifest_path.resolve()),
        "expected_branch": expected_branch,
        "treatment_variables": treatments,
        "selection_policy": "retain_all_runs_no_automatic_deletion_or_good_run_selection",
        "runs": runs,
    }


def _write_json(path: Path, value: Any, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path if exclusive else path.with_name(f".{path.name}.tmp")
    mode = "x" if exclusive else "w"
    with destination.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        os.replace(destination, path)


def _run_context(plan: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable context recorded for one study run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "study_name": plan["study_name"],
        "study_sha256": plan["study_sha256"],
        "expected_branch": plan["expected_branch"],
        "treatment_variables": plan["treatment_variables"],
        "run_name": run["run_name"],
        "condition_sha256": run["condition_sha256"],
        "resolved_condition": run["resolved_condition"],
        "exact_training_command": run["exact_training_command"],
        "evaluation_command_template": run["evaluation_command_template"],
    }


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    """Load a JSON object required for a resumable study."""
    if not path.is_file():
        raise ValueError(f"Cannot resume: {description} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Cannot resume: {description} must contain a JSON object: {path}")
    return value


def _plan_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable fields that must match before a study can resume."""
    runs = plan.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, Mapping) for run in runs):
        raise ValueError("Cannot resume: stored plan runs must be a list of objects.")
    return {
        **{field: plan.get(field) for field in PLAN_IDENTITY_FIELDS},
        "runs": [{field: run.get(field) for field in RUN_IDENTITY_FIELDS} for run in runs],
    }


def _resume_execution_state(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Validate immutable study records and restore mutable run state."""
    if not output_dir.is_dir():
        raise ValueError(f"Cannot resume: output directory does not exist: {output_dir}")
    stored_plan = _load_json_object(output_dir / "study_plan.json", "study plan")
    cohort = _load_json_object(output_dir / "cohort_manifest.json", "cohort manifest")
    expected_identity = _plan_identity(plan)
    if _plan_identity(stored_plan) != expected_identity:
        raise ValueError("Cannot resume: study_plan.json does not match the requested manifest and commands.")
    if _plan_identity(cohort) != expected_identity:
        raise ValueError("Cannot resume: cohort_manifest.json does not match the immutable study plan.")

    stored_runs = {str(run["run_name"]): run for run in cohort["runs"]}
    for run in plan["runs"]:
        context_path = Path(run["study_context"])
        context = _load_json_object(context_path, f"context for {run['run_name']}")
        if context != _run_context(plan, run):
            raise ValueError(f"Cannot resume: study context was changed for {run['run_name']}.")

        stored = stored_runs[run["run_name"]]
        status = stored.get("status")
        exit_code = stored.get("exit_code")
        if status == "completed":
            if exit_code != 0:
                raise ValueError(f"Cannot resume: completed run {run['run_name']} has exit_code {exit_code!r}.")
        elif status not in RESUMABLE_STATUSES:
            raise ValueError(f"Cannot resume: run {run['run_name']} has unsupported status {status!r}.")

        for field in ("status", "exit_code", "run_dir", "initialization_record", "failure", "attempts"):
            if field in stored:
                run[field] = copy.deepcopy(stored[field])
        if status != "completed":
            run["status"] = "planned"
    return plan


def _run_snapshot(condition: Mapping[str, Any]) -> set[Path]:
    spec = SYMM_CLI.get_robot(condition["robot"])
    root = SYMM_CLI.repo_root() / "logs" / "rsl_rl" / spec.experiment_name
    return {path.resolve() for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()


def _new_run_dir(before: set[Path], condition: Mapping[str, Any], run_name: str) -> Path | None:
    after = _run_snapshot(condition)
    candidates = [path for path in after - before if path.name.endswith(f"_{run_name}")]
    if len(candidates) == 1:
        return candidates[0]
    return None


def execute_plan(plan: dict[str, Any], output_dir: Path, *, resume: bool = False) -> int:
    """Execute retained runs, optionally resuming an immutable existing plan."""
    plan_path = output_dir / "study_plan.json"
    cohort_path = output_dir / "cohort_manifest.json"
    if resume:
        plan = _resume_execution_state(plan, output_dir)
        _write_json(cohort_path, plan, exclusive=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(plan_path, plan, exclusive=True)
        for run in plan["runs"]:
            _write_json(Path(run["study_context"]), _run_context(plan, run), exclusive=True)
        _write_json(cohort_path, plan, exclusive=True)

    for run in plan["runs"]:
        if run["status"] == "completed":
            continue
        before = _run_snapshot(run["resolved_condition"])
        run["status"] = "running"
        run["attempts"] = int(run.get("attempts", 0)) + 1
        run.pop("failure", None)
        _write_json(cohort_path, plan, exclusive=False)
        print(run["exact_training_command_text"], flush=True)
        try:
            result = subprocess.run(
                run["exact_training_command"],
                cwd=SYMM_CLI.repo_root(),
                env=SYMM_CLI.repo_subprocess_environment(),
                check=False,
            )
            run["exit_code"] = result.returncode
            run["status"] = "completed" if result.returncode == 0 else "failed"
        except OSError as exc:
            run["exit_code"] = 127
            run["status"] = "failed"
            run["failure"] = str(exc)
        except KeyboardInterrupt:
            run["exit_code"] = 130
            run["status"] = "interrupted"
            _write_json(cohort_path, plan, exclusive=False)
            raise
        run_dir = _new_run_dir(before, run["resolved_condition"], run["run_name"])
        if run_dir is not None:
            run["run_dir"] = str(run_dir)
            initialization = run_dir / PROVENANCE.INITIALIZATION_RELATIVE_PATH
            if initialization.is_file():
                run["initialization_record"] = str(initialization)
        _write_json(cohort_path, plan, exclusive=False)

    plan["matched_initialization_validation"] = PROVENANCE.validate_cohort(plan)
    _write_json(cohort_path, plan, exclusive=False)
    any_failed = any(run["status"] != "completed" or run["exit_code"] != 0 for run in plan["runs"])
    return 1 if any_failed or not plan["matched_initialization_validation"]["matched"] else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the manifest launcher parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Study manifest JSON.")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Directory for contexts and cohort records.",
    )
    parser.add_argument("--conda_env", "--conda-env", dest="conda_env", default=SYMM_CLI.CONDA_ENV)
    parser.add_argument("--use_conda_run", "--use-conda-run", dest="use_conda_run", action="store_true")
    parser.add_argument("--no_conda_run", "--no-conda-run", dest="no_conda_run", action="store_true")
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the fully resolved plan without writing or running.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue an existing matching study directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate, print, or execute a manifest-defined study."""
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    try:
        if args.use_conda_run and args.no_conda_run:
            raise ValueError("--use_conda_run and --no_conda_run are mutually exclusive.")
        if args.output_dir is None:
            source = json.loads(manifest_path.read_text(encoding="utf-8"))
            name = _slug(str(source.get("study_name", "study")))
            identity = PROVENANCE.sha256_value(source)[:12]
            output_dir = SYMM_CLI.repo_root() / "logs" / "rsl_rl" / "studies" / f"{name}-{identity}"
        else:
            output_dir = args.output_dir.resolve()
        plan = load_plan(manifest_path, output_dir, args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
            return 0
        return execute_plan(plan, output_dir, resume=args.resume)
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[symm_study] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
