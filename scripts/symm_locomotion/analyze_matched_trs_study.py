# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a manifest-defined matched TRS study and validate its inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "symm_locomotion"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_trs_grid as analysis  # noqa: E402
import plot_trs_tensorboard as reward_plot  # noqa: E402

ANALYSIS_METHOD_VERSION = "phase_mapping_v2_matched_trs_v1"
PHASE_MAPPING_VERSION = "same_gait_backward_duty_aware_trs_v2"
OUTPUT_DIR = REPO_ROOT
STUDY_MANIFEST_PATH = REPO_ROOT
ROBOT_DISPLAY_NAME = "Unconfigured robot"
ROBOT_SHORT_NAME = "Robot"
TRAINING_SEED = 42
TRAINING_NUM_ENVS = 512
TRAINING_NUM_STEPS_PER_ENV = 24
TRAINING_MAX_ITERATIONS = 20_000
BALANCE_Y_MIN = -100.0
BALANCE_Y_MAX = 100.0
BALANCE_Y_TICKS = (-100.0, -50.0, 0.0, 50.0, 100.0)
SMOOTHING_WINDOW = 200
BOOTSTRAP_REPLICATES = 20_000
MATCHED_NPZ_MEMBERS = (
    "time_steps.npy",
    "desired_lin_vel.npy",
    "desired_positions.npy",
    "episode_done.npy",
    "foot_clearance_targets.npy",
    "foot_clearance_swing_weights.npy",
    "ground_reaction_force_includes_friction.npy",
    "joint_position_lower_limits.npy",
    "joint_position_upper_limits.npy",
    "joint_names.npy",
    "leg_names.npy",
    "motor_role_names.npy",
    "E_C_frc.npy",
    "E_C_spd.npy",
)
WINDOWS: dict[str, tuple[float, float]] = {}
PRIMARY_PAIR_METRICS = (
    "torque_squared",
    "normalized_torque_utilization",
    "absolute_work",
    "positive_work",
    "negative_work",
    "vertical_grf_impulse",
    "contact_time",
)
DURABILITY_PAIR_METRICS = ("fatigue_proxy_m3", "fatigue_proxy_m5")
PAIR_METRIC_UNITS = {
    "torque_squared": "N^2 m^2 s",
    "normalized_torque_utilization": "s",
    "absolute_work": "J",
    "positive_work": "J",
    "negative_work": "J",
    "vertical_grf_impulse": "N s",
    "contact_time": "leg s",
    "fatigue_proxy_m3": "1",
    "fatigue_proxy_m5": "1",
}
BALANCE_METRICS = (
    ("torque_squared", "Torque²"),
    ("normalized_torque_utilization", "Normalized torque²"),
    ("absolute_work", "Absolute work"),
    ("vertical_grf_impulse", "Vertical GRF impulse"),
    ("fatigue_proxy_m5", "Rainflow m=5 proxy"),
)
RUN_COLORS = ("#202124", "#0072B2", "#D97706", "#00875A")
RUN_DASH_ARRAYS = (None, None, None, None)


@dataclass(frozen=True)
class StudyRun:
    """One archived policy in the four-run study."""

    label: str
    slug: str
    robot: str
    run_root: Path
    folder: str
    mirror_coeff: float
    value_coeff: float
    warmup_iterations: int | None
    trs_enabled: bool

    @property
    def path(self) -> Path:
        """Return the archived run directory."""
        return self.run_root / self.folder

    @property
    def run_spec(self) -> analysis.RunSpec:
        """Return the common analysis descriptor."""
        return analysis.RunSpec(
            robot=self.robot,
            run_path=self.path,
            evaluation_path=self.path / "plots" / "play" / "sim_data.npz",
            mirror_coeff=self.mirror_coeff,
            value_coeff=self.value_coeff,
            warmup_iterations=self.warmup_iterations,
            trs_enabled=self.trs_enabled,
        )


RUNS: tuple[StudyRun, ...] = ()


def load_study_manifest(path: Path) -> dict[str, Any]:
    """Load and validate one matched-study manifest without reading run data."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_version = int(payload["schema_version"])
        method_version = str(payload["analysis_method_version"])
        robot = str(payload["robot"])
        display_name = str(payload["display_name"])
        short_name = str(payload["short_name"])
        run_root = Path(payload["run_root"])
        training = {
            "seed": int(payload["training"]["seed"]),
            "num_envs": int(payload["training"]["num_envs"]),
            "num_steps_per_env": int(payload["training"]["num_steps_per_env"]),
            "max_iterations": int(payload["training"]["max_iterations"]),
        }
        windows = {str(name): (float(bounds[0]), float(bounds[1])) for name, bounds in payload["windows_s"].items()}
        balance_range = tuple(float(value) for value in payload["balance_y_range"])
        balance_ticks = tuple(float(value) for value in payload["balance_y_ticks"])
        run_entries = payload["runs"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid matched-study manifest {path}: {error}") from error

    if schema_version != 1:
        raise ValueError(f"Unsupported manifest schema_version={schema_version}; expected 1.")
    if method_version != ANALYSIS_METHOD_VERSION:
        raise ValueError(
            f"Manifest analysis_method_version={method_version!r} does not match {ANALYSIS_METHOD_VERSION!r}."
        )
    if robot not in {"go2", "x1"}:
        raise ValueError(f"Unsupported robot {robot!r}; expected 'go2' or 'x1'.")
    if not display_name or not short_name:
        raise ValueError("Manifest display_name and short_name must be non-empty.")
    if any(value <= 0 for key, value in training.items() if key != "seed"):
        raise ValueError("Manifest training counts must be positive.")
    if training["max_iterations"] != 20_000:
        raise ValueError("This analysis method requires exactly 20,000 training iterations.")
    if set(windows) != {"backward", "forward"} or any(start >= end for start, end in windows.values()):
        raise ValueError("Manifest windows_s must contain increasing backward and forward windows.")
    if len(balance_range) != 2 or balance_range[0] >= balance_range[1]:
        raise ValueError("Manifest balance_y_range must contain two increasing values.")
    if not balance_ticks or any(tick < balance_range[0] or tick > balance_range[1] for tick in balance_ticks):
        raise ValueError("Manifest balance_y_ticks must lie inside balance_y_range.")
    if not isinstance(run_entries, list) or len(run_entries) != 4:
        raise ValueError("A matched phase-mapping study must define exactly four runs.")

    resolved_run_root = run_root if run_root.is_absolute() else REPO_ROOT / run_root
    runs: list[StudyRun] = []
    try:
        for entry in run_entries:
            if type(entry["trs_enabled"]) is not bool:
                raise TypeError("trs_enabled must be a JSON boolean")
            warmup = entry["warmup_iterations"]
            if warmup is not None and (type(warmup) is not int or warmup <= 0):
                raise ValueError("warmup_iterations must be null or a positive integer")
            for coefficient_name in ("mirror_coeff", "value_coeff"):
                coefficient = entry[coefficient_name]
                if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
                    raise TypeError(f"{coefficient_name} must be numeric")
            runs.append(
                StudyRun(
                    label=str(entry["label"]),
                    slug=str(entry["slug"]),
                    robot=robot,
                    run_root=resolved_run_root,
                    folder=str(entry["folder"]),
                    mirror_coeff=float(entry["mirror_coeff"]),
                    value_coeff=float(entry["value_coeff"]),
                    warmup_iterations=warmup,
                    trs_enabled=entry["trs_enabled"],
                )
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid run entry in {path}: {error}") from error

    for field_name in ("label", "slug", "folder"):
        values = [getattr(run, field_name) for run in runs]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Manifest run {field_name} values must be non-empty and unique.")

    coefficient_pairs = [(run.mirror_coeff, run.value_coeff) for run in runs]
    if len(set(coefficient_pairs)) != len(coefficient_pairs):
        raise ValueError("Manifest mirror/value coefficient pairs must be unique.")
    if any(mirror < 0.0 or value < 0.0 for mirror, value in coefficient_pairs):
        raise ValueError("Manifest mirror/value coefficients must be nonnegative.")
    disabled_runs = [run for run in runs if not run.trs_enabled]
    if len(disabled_runs) != 1 or runs[0] is not disabled_runs[0]:
        raise ValueError("Manifest must place exactly one no-TRS baseline first.")
    baseline = disabled_runs[0]
    if (baseline.mirror_coeff, baseline.value_coeff, baseline.warmup_iterations) != (0.0, 0.0, None):
        raise ValueError("The no-TRS baseline must use zero coefficients and null warm-up.")
    enabled_runs = [run for run in runs if run.trs_enabled]
    if len(enabled_runs) != 3 or any(
        run.mirror_coeff <= 0.0 or run.value_coeff <= 0.0 or run.warmup_iterations is None for run in enabled_runs
    ):
        raise ValueError("Manifest must define three enabled TRS runs with positive coefficients and warm-up.")

    return {
        "robot": robot,
        "display_name": display_name,
        "short_name": short_name,
        "run_root": resolved_run_root,
        "training": training,
        "windows": windows,
        "balance_range": balance_range,
        "balance_ticks": balance_ticks,
        "runs": tuple(runs),
    }


def configure_study(manifest_path: Path, output_dir: Path | None = None) -> None:
    """Configure the analysis engine from ``manifest_path``."""
    global BALANCE_Y_MAX, BALANCE_Y_MIN, BALANCE_Y_TICKS, OUTPUT_DIR
    global ROBOT_DISPLAY_NAME, ROBOT_SHORT_NAME, RUNS, STUDY_MANIFEST_PATH, WINDOWS
    global TRAINING_MAX_ITERATIONS, TRAINING_NUM_ENVS, TRAINING_NUM_STEPS_PER_ENV, TRAINING_SEED

    STUDY_MANIFEST_PATH = manifest_path.resolve()
    study = load_study_manifest(STUDY_MANIFEST_PATH)
    OUTPUT_DIR = (output_dir or manifest_path.parent).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROBOT_DISPLAY_NAME = study["display_name"]
    ROBOT_SHORT_NAME = study["short_name"]
    TRAINING_SEED = study["training"]["seed"]
    TRAINING_NUM_ENVS = study["training"]["num_envs"]
    TRAINING_NUM_STEPS_PER_ENV = study["training"]["num_steps_per_env"]
    TRAINING_MAX_ITERATIONS = study["training"]["max_iterations"]
    analysis.SAMPLES_PER_ITERATION = TRAINING_NUM_ENVS * TRAINING_NUM_STEPS_PER_ENV
    reward_plot.SAMPLES_PER_ITERATION = analysis.SAMPLES_PER_ITERATION
    BALANCE_Y_MIN, BALANCE_Y_MAX = study["balance_range"]
    BALANCE_Y_TICKS = study["balance_ticks"]
    WINDOWS = study["windows"]
    RUNS = study["runs"]


def _single_file(directory: Path, pattern: str) -> Path:
    """Return the only file matching ``pattern`` in ``directory``."""
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern!r} under {directory}, found {len(paths)}.")
    return paths[0]


def _normalized_yaml(path: Path, normalized_keys: set[str]) -> str:
    """Return resolved YAML text with selected scalar values normalized."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        key = stripped.partition(":")[0]
        if key in normalized_keys and ":" in stripped:
            indentation = line[: len(line) - len(stripped)]
            line = f"{indentation}{key}: <normalized>"
        lines.append(line)
    return "\n".join(lines)


def _normalized_text_sha256(path: Path) -> str:
    """Hash UTF-8 text after normalizing checkout-specific line endings."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalized_training_provenance(path: Path) -> str:
    """Return training-source diff sections, excluding archive-listing noise."""
    prefixes = ("a/scripts/symm_locomotion/", "a/source/isaaclab_tasks/")
    sections = path.read_text(encoding="utf-8").split("diff --git ")
    selected = [f"diff --git {section}" for section in sections[1:] if section.startswith(prefixes)]
    if not selected:
        raise ValueError(f"No training-source diff sections found in {path}.")
    return "".join(selected)


def _read_yaml_scalars(path: Path, key: str) -> list[Any]:
    """Read every scalar matching ``key`` from a resolved YAML snapshot."""
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            values.append(yaml.safe_load(stripped.partition(":")[2].strip()))
    return values


def _read_unique_yaml_scalar(path: Path, key: str) -> Any:
    """Read one unique scalar key from an Isaac Lab resolved YAML snapshot."""
    values = _read_yaml_scalars(path, key)
    if len(values) != 1:
        raise ValueError(f"Expected one scalar {key!r} in {path}, found {len(values)}.")
    return values[0]


def _read_uniform_yaml_scalar(path: Path, key: str) -> Any:
    """Read repeated scalar keys when all resolved values are identical."""
    values = _read_yaml_scalars(path, key)
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"Expected one or more identical {key!r} scalars in {path}, got {values}.")
    return values[0]


def _validate_run_configuration_values(
    run: StudyRun,
    symmetry_cfg: dict[str, Any],
    command_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate resolved run settings against the study manifest and milestone contract."""
    expected_warmup = run.warmup_iterations if run.warmup_iterations is not None else 500
    expected = {
        "use_data_augmentation": False,
        "use_mirror_loss": run.trs_enabled,
        "mirror_loss_coeff": run.mirror_coeff,
        "use_time_reversal_regularization": True,
        "value_loss_coeff": run.value_coeff,
        "min_abs_command_velocity": 0.0,
        "warmup_iterations": expected_warmup,
        "phase_mapping_version": PHASE_MAPPING_VERSION,
        "min_xy_command_norm": 0.0,
        "seed": TRAINING_SEED,
        "num_envs": TRAINING_NUM_ENVS,
        "max_iterations": TRAINING_MAX_ITERATIONS,
        "num_steps_per_env": TRAINING_NUM_STEPS_PER_ENV,
    }
    actual = {**symmetry_cfg, **command_cfg, **training_cfg}
    mismatched = {
        key: {"expected": expected_value, "actual": actual.get(key)}
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatched:
        raise ValueError(f"Resolved settings do not match manifest for {run.label}: {mismatched}")
    return actual


def _validate_run_configuration(run: StudyRun) -> dict[str, Any]:
    """Validate one run's archived agent and environment configuration."""
    agent_path = run.path / "params" / "agent.yaml"
    env_path = run.path / "params" / "env.yaml"
    symmetry_keys = (
        "use_data_augmentation",
        "use_mirror_loss",
        "mirror_loss_coeff",
        "use_time_reversal_regularization",
        "value_loss_coeff",
        "min_abs_command_velocity",
        "warmup_iterations",
    )
    command_keys = ("phase_mapping_version", "min_xy_command_norm")
    training_keys = ("seed", "max_iterations", "num_steps_per_env")
    training_cfg = {key: _read_unique_yaml_scalar(agent_path, key) for key in training_keys}
    training_cfg["num_envs"] = _read_uniform_yaml_scalar(env_path, "num_envs")
    return _validate_run_configuration_values(
        run,
        {key: _read_unique_yaml_scalar(agent_path, key) for key in symmetry_keys},
        {key: _read_unique_yaml_scalar(env_path, key) for key in command_keys},
        training_cfg,
    )


def validate_artifacts() -> dict[str, Any]:
    """Validate completed checkpoints and byte-identical paired inputs."""
    member_hashes: dict[str, dict[str, str]] = {}
    env_snapshots: dict[str, str] = {}
    agent_snapshots: dict[str, str] = {}
    initial_checkpoint_hashes: dict[str, str] = {}
    raw_provenance_hashes: dict[str, str] = {}
    training_provenance_hashes: dict[str, str] = {}
    resolved_run_settings: dict[str, dict[str, Any]] = {}
    for run in RUNS:
        required = (
            run.path / "model_0.pt",
            run.path / "model_19999.pt",
            run.path / "params" / "agent.yaml",
            run.path / "params" / "env.yaml",
            run.path / "git" / "symm_rl_isaaclab.diff",
            run.run_spec.evaluation_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required artifacts for {run.label}: {missing}")
        _single_file(run.path, "events.out.tfevents.*")
        hashes = {}
        with zipfile.ZipFile(run.run_spec.evaluation_path) as archive:
            names = set(archive.namelist())
            for member in MATCHED_NPZ_MEMBERS:
                if member not in names:
                    raise ValueError(f"Missing {member} in {run.run_spec.evaluation_path}")
                hashes[member] = hashlib.sha256(archive.read(member)).hexdigest()
        member_hashes[run.slug] = hashes
        env_snapshots[run.slug] = _normalized_yaml(run.path / "params" / "env.yaml", {"log_dir"})
        agent_snapshots[run.slug] = _normalized_yaml(
            run.path / "params" / "agent.yaml",
            {
                "run_name",
                "use_mirror_loss",
                "mirror_loss_coeff",
                "value_loss_coeff",
                "warmup_iterations",
            },
        )
        resolved_run_settings[run.slug] = _validate_run_configuration(run)
        initial_checkpoint_hashes[run.slug] = hashlib.sha256((run.path / "model_0.pt").read_bytes()).hexdigest()
        provenance_path = run.path / "git" / "symm_rl_isaaclab.diff"
        raw_provenance_hashes[run.slug] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
        training_provenance_hashes[run.slug] = hashlib.sha256(
            _normalized_training_provenance(provenance_path).encode("utf-8")
        ).hexdigest()

    reference = member_hashes[RUNS[0].slug]
    equality = {
        member: all(member_hashes[run.slug][member] == reference[member] for run in RUNS[1:])
        for member in MATCHED_NPZ_MEMBERS
    }
    if not all(equality.values()):
        mismatched = [member for member, equal in equality.items() if not equal]
        raise ValueError(f"The supposedly matched rollout inputs differ: {mismatched}")
    env_equal = len(set(env_snapshots.values())) == 1
    agent_equal_after_trs_normalization = len(set(agent_snapshots.values())) == 1
    initial_checkpoint_equal = len(set(initial_checkpoint_hashes.values())) == 1
    raw_provenance_equal = len(set(raw_provenance_hashes.values())) == 1
    training_provenance_equal = len(set(training_provenance_hashes.values())) == 1
    if not env_equal:
        raise ValueError("Resolved environment snapshots differ beyond log_dir.")
    if not agent_equal_after_trs_normalization:
        raise ValueError("Resolved agent snapshots differ beyond run name and TRS settings.")
    if not initial_checkpoint_equal:
        raise ValueError("Initial checkpoints are not byte-identical.")
    if not training_provenance_equal:
        raise ValueError("Archived training-source provenance differs between runs.")
    return {
        "model_19999_present_for_all_runs": True,
        "environment_snapshots_equal_after_log_dir_normalization": env_equal,
        "agent_snapshots_equal_after_run_name_and_trs_normalization": agent_equal_after_trs_normalization,
        "initial_checkpoints_byte_identical": initial_checkpoint_equal,
        "initial_checkpoint_sha256": initial_checkpoint_hashes,
        "raw_code_provenance_diffs_byte_identical": raw_provenance_equal,
        "raw_code_provenance_diff_sha256": raw_provenance_hashes,
        "training_source_provenance_equal_after_archive_listing_exclusion": training_provenance_equal,
        "training_source_provenance_sha256": training_provenance_hashes,
        "resolved_run_settings": resolved_run_settings,
        "matched_npz_members": equality,
        "sha256": member_hashes,
    }


def load_recorded_limit_diagnostics(
    evaluation_path: Path,
    start_s: float,
    stop_s: float,
) -> dict[str, float]:
    """Load mean joint-limit fractions over one recorded window."""
    fields = (
        "joint_near_limit_fraction",
        "joint_limit_violation_fraction",
        "joint_target_limit_violation_fraction",
    )
    with zipfile.ZipFile(evaluation_path) as archive:
        times, time_shape = analysis.read_npy_member(archive, "time_steps")
        if len(time_shape) != 1:
            raise ValueError(f"Unexpected time_steps shape in {evaluation_path}: {time_shape}")
        selected = [index for index, time_s in enumerate(times) if start_s <= float(time_s) < stop_s]
        diagnostics = {}
        for field in fields:
            values, shape = analysis.read_npy_member(archive, field)
            if shape != time_shape:
                raise ValueError(f"Unexpected {field} shape in {evaluation_path}: {shape}")
            diagnostics[f"{field}_mean"] = statistics.fmean(float(values[index]) for index in selected)
    return diagnostics


def analyze_rollouts() -> dict[str, dict[str, dict[str, Any]]]:
    """Analyze the matched backward and forward steady-command windows."""
    old_start = analysis.WINDOW_START_S
    old_stop = analysis.WINDOW_STOP_S
    old_bootstrap = analysis.BOOTSTRAP_REPLICATES
    analysis.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
    results: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for window_name, (start_s, stop_s) in WINDOWS.items():
            analysis.WINDOW_START_S = start_s
            analysis.WINDOW_STOP_S = stop_s
            results[window_name] = {}
            for run in RUNS:
                print(f"Analyzing {window_name}: {run.label}", flush=True)
                result = analysis.analyze_rollout(run.run_spec)
                result["recorded_joint_limit_diagnostics"] = load_recorded_limit_diagnostics(
                    run.run_spec.evaluation_path,
                    start_s,
                    stop_s,
                )
                results[window_name][run.slug] = result
    finally:
        analysis.WINDOW_START_S = old_start
        analysis.WINDOW_STOP_S = old_stop
        analysis.BOOTSTRAP_REPLICATES = old_bootstrap
    return results


def analyze_training_runs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Analyze sample efficiency and selected safety diagnostics."""
    training: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    selected_tags = {
        "Loss/symmetry",
        "Loss/tr_value",
        "Diagnostics/action_abs_mean",
        "Diagnostics/joint_near_limit_fraction",
        "Diagnostics/joint_limit_violation_fraction",
        "Episode_Termination/time_out",
        "Episode_Termination/calf_height",
    }
    for run in RUNS:
        print(f"Analyzing training: {run.label}", flush=True)
        training[run.slug] = analysis.analyze_training(run.run_spec)
        event_path = _single_file(run.path, "events.out.tfevents.*")
        scalars = analysis.read_scalars(event_path, selected_tags=selected_tags)
        run_diagnostics = {}
        for tag in sorted(selected_tags):
            points = scalars.get(tag)
            if not points:
                run_diagnostics[tag] = None
                continue
            tail = points[-1000:]
            run_diagnostics[tag] = {
                "last_1000_mean": statistics.fmean(point[2] for point in tail),
                "last_value": points[-1][2],
            }
        diagnostics[run.slug] = run_diagnostics
    return training, diagnostics


def load_reward_curves() -> list[reward_plot.RewardCurve]:
    """Load the four 200-iteration-smoothed reward curves."""
    return [reward_plot.load_reward_curve(run.run_spec, SMOOTHING_WINDOW) for run in RUNS]


def write_front_hind_csv(rollouts: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Write pair allocation and temporal-bootstrap summaries."""
    path = OUTPUT_DIR / "front_hind_metrics.csv"
    fieldnames = (
        "window",
        "command_x_mps",
        "run",
        "label",
        "metric",
        "unit",
        "front",
        "hind",
        "front_per_directed_m",
        "hind_per_directed_m",
        "front_share_percent",
        "signed_imbalance_percent",
        "absolute_imbalance_percent",
        "bootstrap_95_low_percent",
        "bootstrap_95_high_percent",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for window_name in WINDOWS:
            for run in RUNS:
                result = rollouts[window_name][run.slug]
                distance = result["directed_progress_m"]
                for metric_name in PRIMARY_PAIR_METRICS:
                    metric = result["metrics"][metric_name]
                    ci_low, ci_high = metric["bootstrap_95_percent"]
                    per_distance = metric_name != "contact_time"
                    writer.writerow(
                        {
                            "window": window_name,
                            "command_x_mps": result["command"]["x_mean_mps"],
                            "run": run.folder,
                            "label": run.label,
                            "metric": metric_name,
                            "unit": PAIR_METRIC_UNITS[metric_name],
                            "front": metric["front_integral"],
                            "hind": metric["hind_integral"],
                            "front_per_directed_m": metric["front_integral"] / distance if per_distance else "",
                            "hind_per_directed_m": metric["hind_integral"] / distance if per_distance else "",
                            "front_share_percent": metric["front_share_percent"],
                            "signed_imbalance_percent": metric["signed_imbalance_percent"],
                            "absolute_imbalance_percent": metric["abs_imbalance_percent"],
                            "bootstrap_95_low_percent": ci_low,
                            "bootstrap_95_high_percent": ci_high,
                        }
                    )
                for metric_name in DURABILITY_PAIR_METRICS:
                    metric = result["durability"]["pair_metrics"][metric_name]
                    writer.writerow(
                        {
                            "window": window_name,
                            "command_x_mps": result["command"]["x_mean_mps"],
                            "run": run.folder,
                            "label": run.label,
                            "metric": metric_name,
                            "unit": PAIR_METRIC_UNITS[metric_name],
                            "front": metric["front"],
                            "hind": metric["hind"],
                            "front_per_directed_m": metric["front_per_m"],
                            "hind_per_directed_m": metric["hind_per_m"],
                            "front_share_percent": metric["front_share_percent"],
                            "signed_imbalance_percent": metric["signed_imbalance_percent"],
                            "absolute_imbalance_percent": metric["abs_imbalance_percent"],
                            "bootstrap_95_low_percent": "",
                            "bootstrap_95_high_percent": "",
                        }
                    )
    print(f"Saved {path}", flush=True)


def write_durability_csv(rollouts: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Write tracking, total exposure, and worst-component durability proxies."""
    path = OUTPUT_DIR / "durability_risk_summary.csv"
    fieldnames = (
        "window",
        "run",
        "label",
        "command_x_mps",
        "duration_s",
        "directed_progress_m",
        "lateral_drift_m",
        "tracking_rmse_mps",
        "episode_ends",
        "torque_squared_per_m",
        "absolute_work_j_per_m",
        "vertical_grf_impulse_ns_per_m",
        "fatigue_m5_total_per_m",
        "fatigue_m5_dominant_pair",
        "fatigue_m5_dominant_pair_per_m",
        "fatigue_m5_dominant_to_other_ratio",
        "worst_joint_fatigue_m5_component",
        "worst_joint_fatigue_m5_per_m",
        "worst_joint_normalized_torque_peak_component",
        "worst_joint_normalized_torque_peak",
        "worst_foot_vertical_force_component",
        "worst_foot_vertical_force_peak_n",
        "front_contact_duty",
        "hind_contact_duty",
        "recorded_joint_near_limit_fraction",
        "recorded_joint_limit_violation_fraction",
        "recorded_joint_target_limit_violation_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for window_name in WINDOWS:
            for run in RUNS:
                result = rollouts[window_name][run.slug]
                fatigue = result["durability"]["pair_metrics"]["fatigue_proxy_m5"]
                worst = result["durability"]["worst_components"]
                contact = result["metrics"]["contact_time"]
                limit = result["recorded_joint_limit_diagnostics"]
                writer.writerow(
                    {
                        "window": window_name,
                        "run": run.folder,
                        "label": run.label,
                        "command_x_mps": result["command"]["x_mean_mps"],
                        "duration_s": result["duration_s"],
                        "directed_progress_m": result["directed_progress_m"],
                        "lateral_drift_m": result["lateral_drift_m"],
                        "tracking_rmse_mps": result["planar_tracking_rmse_mps"],
                        "episode_ends": result["validation"]["episode_ends_in_window"],
                        "torque_squared_per_m": result["cost_per_distance"]["torque_squared_integral_per_m"],
                        "absolute_work_j_per_m": result["cost_per_distance"]["absolute_work_j_per_m"],
                        "vertical_grf_impulse_ns_per_m": result["cost_per_distance"]["vertical_grf_impulse_ns_per_m"],
                        "fatigue_m5_total_per_m": fatigue["total_per_m"],
                        "fatigue_m5_dominant_pair": fatigue["dominant_pair"],
                        "fatigue_m5_dominant_pair_per_m": max(fatigue["front_per_m"], fatigue["hind_per_m"]),
                        "fatigue_m5_dominant_to_other_ratio": fatigue["dominant_to_other_ratio"],
                        "worst_joint_fatigue_m5_component": worst["joint_fatigue_proxy_m5_per_m"]["component"],
                        "worst_joint_fatigue_m5_per_m": worst["joint_fatigue_proxy_m5_per_m"]["value"],
                        "worst_joint_normalized_torque_peak_component": worst["joint_normalized_torque_peak"][
                            "component"
                        ],
                        "worst_joint_normalized_torque_peak": worst["joint_normalized_torque_peak"]["value"],
                        "worst_foot_vertical_force_component": worst["foot_vertical_force_peak"]["component"],
                        "worst_foot_vertical_force_peak_n": worst["foot_vertical_force_peak"]["value"],
                        "front_contact_duty": contact["front_duty_factor"],
                        "hind_contact_duty": contact["hind_duty_factor"],
                        "recorded_joint_near_limit_fraction": limit["joint_near_limit_fraction_mean"],
                        "recorded_joint_limit_violation_fraction": limit["joint_limit_violation_fraction_mean"],
                        "recorded_joint_target_limit_violation_fraction": limit[
                            "joint_target_limit_violation_fraction_mean"
                        ],
                    }
                )
    print(f"Saved {path}", flush=True)


def _threshold_value(training: dict[str, Any], threshold: str, field: str) -> Any:
    """Return one threshold field or an empty CSV value."""
    result = training["thresholds"][threshold]
    return "" if result is None else result[field]


def write_training_csv(training: dict[str, dict[str, Any]], diagnostics: dict[str, dict[str, Any]]) -> None:
    """Write sample-, endpoint-, and observed compute-efficiency summaries."""
    path = OUTPUT_DIR / "training_efficiency.csv"
    fieldnames = (
        "run",
        "label",
        "mirror_loss_coeff",
        "value_loss_coeff",
        "warmup_iterations",
        "iterations",
        "transitions_per_iteration",
        "environment_transitions",
        "wall_time_hours",
        "reward_auc_first_10000",
        "reward_auc_full",
        "reward_last_1000_mean",
        "reward_last_1000_std",
        "reward_30_iteration",
        "reward_30_transitions",
        "reward_35_iteration",
        "reward_35_transitions",
        "straight_reward_1p5_iteration",
        "velocity_error_xy_0p15_iteration",
        "episode_length_last_1000_mean",
        "velocity_error_xy_last_1000_mean",
        "velocity_error_yaw_last_1000_mean",
        "throughput_fps_mean",
        "learning_seconds_per_iteration_mean",
        "symmetry_loss_last_1000_mean",
        "tr_value_loss_last_1000_mean",
        "action_abs_mean_last_1000_mean",
        "joint_near_limit_fraction_last_1000_mean",
        "joint_limit_violation_fraction_last_1000_mean",
        "timeout_fraction_last_1000_mean",
        "calf_height_termination_fraction_last_1000_mean",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in RUNS:
            result = training[run.slug]
            diagnostic = diagnostics[run.slug]

            def tail(tag: str) -> Any:
                value = diagnostic[tag]
                return "" if value is None else value["last_1000_mean"]

            writer.writerow(
                {
                    "run": run.folder,
                    "label": run.label,
                    "mirror_loss_coeff": run.mirror_coeff,
                    "value_loss_coeff": run.value_coeff,
                    "warmup_iterations": run.warmup_iterations if run.warmup_iterations is not None else "",
                    "iterations": result["iterations"],
                    "transitions_per_iteration": analysis.SAMPLES_PER_ITERATION,
                    "environment_transitions": result["environment_transitions"],
                    "wall_time_hours": result["wall_time_hours"],
                    "reward_auc_first_10000": result["mean_reward"]["auc_first_10000"],
                    "reward_auc_full": result["mean_reward"]["auc_all"],
                    "reward_last_1000_mean": result["mean_reward"]["last_1000_mean"],
                    "reward_last_1000_std": result["mean_reward"]["last_1000_std"],
                    "reward_30_iteration": _threshold_value(result, "reward_30", "iteration"),
                    "reward_30_transitions": _threshold_value(result, "reward_30", "environment_transitions"),
                    "reward_35_iteration": _threshold_value(result, "reward_35", "iteration"),
                    "reward_35_transitions": _threshold_value(result, "reward_35", "environment_transitions"),
                    "straight_reward_1p5_iteration": _threshold_value(result, "straight_reward_1p5", "iteration"),
                    "velocity_error_xy_0p15_iteration": _threshold_value(result, "velocity_error_xy_0p15", "iteration"),
                    "episode_length_last_1000_mean": result["mean_episode_length"]["last_1000_mean"],
                    "velocity_error_xy_last_1000_mean": result["velocity_error_xy"]["last_1000_mean"],
                    "velocity_error_yaw_last_1000_mean": result["velocity_error_yaw"]["last_1000_mean"],
                    "throughput_fps_mean": result["throughput_fps"]["mean"],
                    "learning_seconds_per_iteration_mean": result["timing_seconds_per_iteration"]["learning_mean"],
                    "symmetry_loss_last_1000_mean": tail("Loss/symmetry"),
                    "tr_value_loss_last_1000_mean": tail("Loss/tr_value"),
                    "action_abs_mean_last_1000_mean": tail("Diagnostics/action_abs_mean"),
                    "joint_near_limit_fraction_last_1000_mean": tail("Diagnostics/joint_near_limit_fraction"),
                    "joint_limit_violation_fraction_last_1000_mean": tail("Diagnostics/joint_limit_violation_fraction"),
                    "timeout_fraction_last_1000_mean": tail("Episode_Termination/time_out"),
                    "calf_height_termination_fraction_last_1000_mean": tail("Episode_Termination/calf_height"),
                }
            )
    print(f"Saved {path}", flush=True)


def write_learning_curve_csv(curves: list[reward_plot.RewardCurve]) -> None:
    """Write the exact downsampled points rendered in the learning plot."""
    path = OUTPUT_DIR / "learning_curve_points.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "label", "transitions_million", "elapsed_hours", "reward_rolling_mean_200"))
        for run, curve in zip(RUNS, curves, strict=True):
            for transitions, elapsed, reward in zip(
                curve.transitions_millions,
                curve.elapsed_hours,
                curve.rewards,
                strict=True,
            ):
                writer.writerow((run.folder, run.label, transitions, elapsed, reward))
    print(f"Saved {path}", flush=True)


def write_learning_curve_svg(curves: list[reward_plot.RewardCurve]) -> None:
    """Plot mean reward against samples and observed wall time."""
    reward_values = [value for curve in curves for value in curve.rewards]
    y_min, y_max, y_ticks = reward_plot._nice_bounds([35.0, *reward_values])
    root = ET.Element(
        reward_plot._svg_tag("svg"),
        {
            "viewBox": "0 0 1600 760",
            "width": "1600",
            "height": "760",
            "role": "img",
            "aria-labelledby": "plot-title plot-description",
        },
    )
    title = ET.SubElement(root, reward_plot._svg_tag("title"), {"id": "plot-title"})
    title.text = f"{ROBOT_DISPLAY_NAME} four-run TRS learning-curve comparison"
    description = ET.SubElement(root, reward_plot._svg_tag("desc"), {"id": "plot-description"})
    description.text = (
        "The same four smoothed reward curves are plotted against environment transitions "
        "and observed elapsed training time."
    )
    ET.SubElement(root, reward_plot._svg_tag("rect"), {"width": "1600", "height": "760", "fill": "#FFFFFF"})
    definitions = ET.SubElement(root, reward_plot._svg_tag("defs"))
    reward_plot._add_text(
        root,
        800.0,
        43.0,
        f"{ROBOT_DISPLAY_NAME}: matched no-TRS and TRS learning curves",
        size=24,
        anchor="middle",
        weight=600,
    )
    reward_plot._add_text(
        root,
        800.0,
        72.0,
        (
            f"Train/mean_reward, {SMOOTHING_WINDOW}-iteration trailing mean · seed {TRAINING_SEED} · "
            f"{TRAINING_NUM_ENVS:,} environments · {TRAINING_NUM_STEPS_PER_ENV} steps/iteration · "
            f"{TRAINING_MAX_ITERATIONS:,} iterations"
        ),
        size=13,
        anchor="middle",
        fill="#5F6368",
    )
    reward_plot._draw_panel(
        root,
        definitions,
        curves,
        panel_index=0,
        x_field="transitions_millions",
        title="Sample efficiency",
        x_label="Environment transitions [million]",
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
        curve_colors=RUN_COLORS,
        curve_dash_arrays=RUN_DASH_ARRAYS,
    )
    reward_plot._draw_panel(
        root,
        definitions,
        curves,
        panel_index=1,
        x_field="elapsed_hours",
        title="Observed wall-clock efficiency",
        x_label="Elapsed training time [h]",
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
        curve_colors=RUN_COLORS,
        curve_dash_arrays=RUN_DASH_ARRAYS,
    )
    reward_plot._draw_legend(root, curves, RUN_COLORS, RUN_DASH_ARRAYS)
    output_path = OUTPUT_DIR / "learning_curve_sample_and_wall_time.svg"
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved {output_path}", flush=True)


def _imbalance(result: dict[str, Any], metric_name: str) -> float:
    """Return signed front/hind imbalance for one result metric."""
    if metric_name in DURABILITY_PAIR_METRICS:
        return result["durability"]["pair_metrics"][metric_name]["signed_imbalance_percent"]
    return result["metrics"][metric_name]["signed_imbalance_percent"]


def write_balance_svg(rollouts: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Plot signed front/hind imbalance for both matched commands."""
    svg = reward_plot._svg_tag
    width = 1600.0
    height = 780.0
    panel_width = 675.0
    panel_height = 470.0
    panel_top = 135.0
    panel_lefts = (105.0, 865.0)
    y_min = BALANCE_Y_MIN
    y_max = BALANCE_Y_MAX
    y_ticks = BALANCE_Y_TICKS
    root = ET.Element(
        svg("svg"),
        {
            "viewBox": f"0 0 {width:.0f} {height:.0f}",
            "width": f"{width:.0f}",
            "height": f"{height:.0f}",
            "role": "img",
            "aria-labelledby": "balance-title balance-description",
        },
    )
    title = ET.SubElement(root, svg("title"), {"id": "balance-title"})
    title.text = f"{ROBOT_DISPLAY_NAME} front and hind load allocation by command direction"
    description = ET.SubElement(root, svg("desc"), {"id": "balance-description"})
    description.text = "Grouped bars show signed front-hind imbalance; zero is equal pair allocation."
    ET.SubElement(root, svg("rect"), {"width": f"{width:.0f}", "height": f"{height:.0f}", "fill": "#FFFFFF"})
    reward_plot._add_text(
        root,
        width / 2.0,
        43.0,
        f"{ROBOT_DISPLAY_NAME}: front/hind load allocation in matched playback",
        size=24,
        anchor="middle",
        weight=600,
    )
    reward_plot._add_text(
        root,
        width / 2.0,
        72.0,
        "Signed imbalance = 100 × (front − hind) / (front + hind); zero is equal usage",
        size=13,
        anchor="middle",
        fill="#5F6368",
    )
    for panel_index, window_name in enumerate(WINDOWS):
        left = panel_lefts[panel_index]
        command = rollouts[window_name][RUNS[0].slug]["command"]["x_mean_mps"]
        reward_plot._add_text(
            root,
            left + panel_width / 2.0,
            112.0,
            f"{window_name.capitalize()} command: {command:+.3f} m/s",
            size=17,
            anchor="middle",
            weight=600,
        )
        for tick in y_ticks:
            y = panel_top + (y_max - tick) / (y_max - y_min) * panel_height
            ET.SubElement(
                root,
                svg("line"),
                {
                    "x1": f"{left:.2f}",
                    "x2": f"{left + panel_width:.2f}",
                    "y1": f"{y:.2f}",
                    "y2": f"{y:.2f}",
                    "stroke": "#9AA0A6" if tick == 0.0 else "#E2E5E9",
                    "stroke-width": "1.5" if tick == 0.0 else "1",
                },
            )
            reward_plot._add_text(
                root,
                left - 11.0,
                y + 5.0,
                f"{tick:+.0f}",
                size=12,
                anchor="end",
                fill="#5F6368",
            )
        category_width = panel_width / len(BALANCE_METRICS)
        group_width = category_width * 0.78
        bar_width = group_width / len(RUNS)
        zero_y = panel_top + y_max / (y_max - y_min) * panel_height
        for metric_index, (metric_name, metric_label) in enumerate(BALANCE_METRICS):
            center = left + (metric_index + 0.5) * category_width
            group_left = center - group_width / 2.0
            for run_index, run in enumerate(RUNS):
                value = _imbalance(rollouts[window_name][run.slug], metric_name)
                value_y = panel_top + (y_max - value) / (y_max - y_min) * panel_height
                bar_y = min(value_y, zero_y)
                bar_height = max(abs(zero_y - value_y), 0.8)
                ET.SubElement(
                    root,
                    svg("rect"),
                    {
                        "x": f"{group_left + run_index * bar_width + 1.0:.2f}",
                        "y": f"{bar_y:.2f}",
                        "width": f"{bar_width - 2.0:.2f}",
                        "height": f"{bar_height:.2f}",
                        "fill": RUN_COLORS[run_index],
                        "fill-opacity": "0.88",
                    },
                )
                label_y = value_y - 7.0 if value >= 0.0 else value_y + 16.0
                reward_plot._add_text(
                    root,
                    group_left + (run_index + 0.5) * bar_width,
                    label_y,
                    f"{value:+.1f}",
                    size=10,
                    anchor="middle",
                    fill="#3C4043",
                )
            reward_plot._add_text(
                root,
                center,
                panel_top + panel_height + 25.0,
                metric_label,
                size=11,
                anchor="middle",
                fill="#3C4043",
            )
        if panel_index == 0:
            reward_plot._add_text(
                root,
                29.0,
                panel_top + panel_height / 2.0,
                "Signed front/hind imbalance [%]",
                size=14,
                anchor="middle",
                transform=f"rotate(-90 29.00 {panel_top + panel_height / 2.0:.2f})",
            )
    legend_y = 705.0
    legend_width = 350.0
    legend_start = (width - legend_width * len(RUNS)) / 2.0
    for index, run in enumerate(RUNS):
        x = legend_start + index * legend_width
        ET.SubElement(
            root,
            svg("rect"),
            {"x": f"{x:.2f}", "y": f"{legend_y - 11.0:.2f}", "width": "28", "height": "14", "fill": RUN_COLORS[index]},
        )
        reward_plot._add_text(root, x + 40.0, legend_y + 2.0, run.label, size=13)
    output_path = OUTPUT_DIR / "front_hind_signed_imbalance.svg"
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved {output_path}", flush=True)


def write_summary_json(
    artifact_validation: dict[str, Any],
    rollouts: dict[str, dict[str, dict[str, Any]]],
    training: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    """Write the complete machine-readable study summary."""
    engine_path = Path(__file__).resolve()
    payload = {
        "method": {
            "version": 2,
            "analysis_method_version": ANALYSIS_METHOD_VERSION,
            "analysis_engine": engine_path.relative_to(REPO_ROOT).as_posix(),
            "analysis_engine_sha256": _normalized_text_sha256(engine_path),
            "study_manifest": STUDY_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
            "study_manifest_sha256": _normalized_text_sha256(STUDY_MANIFEST_PATH),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_block_duration_s": analysis.BLOCK_DURATION_S,
            "smoothing_window_iterations": SMOOTHING_WINDOW,
            "threshold_sustain_iterations": 500,
            "training_seed": TRAINING_SEED,
            "training_num_envs": TRAINING_NUM_ENVS,
            "training_num_steps_per_env": TRAINING_NUM_STEPS_PER_ENV,
            "training_max_iterations": TRAINING_MAX_ITERATIONS,
            "samples_per_iteration": analysis.SAMPLES_PER_ITERATION,
            "windows_s": {name: list(window) for name, window in WINDOWS.items()},
            "imbalance_definition": "100 * (front - hind) / (front + hind)",
            "fatigue_interpretation": (
                "Effort-limit-normalized rainflow sensitivity proxy; not Miner damage, "
                "lifetime, or failure probability."
            ),
        },
        "runs": {
            run.slug: {
                "label": run.label,
                "folder": run.folder,
                "mirror_loss_coeff": run.mirror_coeff,
                "value_loss_coeff": run.value_coeff,
                "warmup_iterations": run.warmup_iterations,
                "trs_enabled": run.trs_enabled,
            }
            for run in RUNS
        },
        "artifact_validation": artifact_validation,
        "rollouts": rollouts,
        "training": training,
        "training_diagnostics": diagnostics,
    }
    path = OUTPUT_DIR / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def validate_outputs() -> None:
    """Validate the generated tables and SVG documents."""
    required = (
        "front_hind_metrics.csv",
        "durability_risk_summary.csv",
        "training_efficiency.csv",
        "learning_curve_points.csv",
        "learning_curve_sample_and_wall_time.svg",
        "front_hind_signed_imbalance.svg",
        "summary.json",
    )
    for filename in required:
        path = OUTPUT_DIR / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty generated output: {path}")
    for filename in ("learning_curve_sample_and_wall_time.svg", "front_hind_signed_imbalance.svg"):
        ET.parse(OUTPUT_DIR / filename)
    with (OUTPUT_DIR / "summary.json").open(encoding="utf-8") as stream:
        json.load(stream)


def main(argv: list[str] | None = None) -> None:
    """Run the complete four-policy analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the matched-study JSON manifest.")
    parser.add_argument("--output-dir", type=Path, help="Override the generated-artifact directory.")
    args = parser.parse_args(argv)
    configure_study(args.manifest, args.output_dir)

    artifact_validation = validate_artifacts()
    rollouts = analyze_rollouts()
    training, diagnostics = analyze_training_runs()
    curves = load_reward_curves()
    write_front_hind_csv(rollouts)
    write_durability_csv(rollouts)
    write_training_csv(training, diagnostics)
    write_learning_curve_csv(curves)
    write_learning_curve_svg(curves)
    write_balance_svg(rollouts)
    write_summary_json(artifact_validation, rollouts, training, diagnostics)
    validate_outputs()
    print(f"Four-run {ROBOT_SHORT_NAME} study outputs validated.", flush=True)


if __name__ == "__main__":
    main()
