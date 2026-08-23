# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import random
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION,
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS,
)
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument(
    "--print_gait_info",
    action="store_true",
    default=False,
    help="Print live symmetric gait command and velocity information during play, when available.",
)
parser.add_argument(
    "--print_gait_info_interval",
    type=int,
    default=50,
    help="Number of play steps between gait information terminal prints.",
)
parser.add_argument(
    "--symm_rollout_plots",
    action="store_true",
    default=False,
    help="Save symmetric quadruped rollout plots and sampled data after play.",
)
parser.add_argument(
    "--symm_rollout_plots_dir",
    default=None,
    help="Directory for symmetric rollout plots. Defaults to <run>/plots/play.",
)
parser.add_argument(
    "--symm_rollout_plot_env_index",
    type=int,
    default=0,
    help="Environment index sampled by symmetric rollout plots.",
)
parser.add_argument(
    "--symm_rollout_plot_max_steps",
    type=int,
    default=None,
    help="Maximum number of steps sampled by symmetric rollout plots.",
)
parser.add_argument(
    "--symm_leg_usage_plan",
    default=None,
    help="Run the finite symmetric leg-usage grid described by this study JSON file.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


_SYMM_GAIT_THETAS = (
    ("trot", (0.0, 0.5, 0.5, 0.0)),
    ("bound", (0.0, 0.0, 0.5, 0.5)),
    ("front-spread-half-bound", (0.13, -0.13, 0.5, 0.5)),
    ("hind-spread-half-bound", (0.0, 0.0, 0.63, 0.37)),
    ("rotary-gallop", (-0.13, 0.13, 0.63, 0.37)),
    ("transverse-gallop", (0.13, -0.13, 0.63, 0.37)),
)


def _classify_symmetric_gait(foot_thetas: torch.Tensor) -> str:
    """Return the nearest named symmetric gait for sampled foot phase offsets."""
    theta = foot_thetas.detach().to(dtype=torch.float32, device="cpu")
    best_name = "unknown"
    best_error = float("inf")
    for gait_name, gait_thetas in _SYMM_GAIT_THETAS:
        reference = torch.tensor(gait_thetas, dtype=torch.float32)
        phase_error = torch.atan2(
            torch.sin(2.0 * torch.pi * (theta - reference)),
            torch.cos(2.0 * torch.pi * (theta - reference)),
        )
        error = torch.linalg.norm(phase_error).item()
        if error < best_error:
            best_name = gait_name
            best_error = error
    return best_name


def _format_symmetric_gait_info(env, env_index: int = 0) -> str | None:
    """Format live symmetric gait command information when available."""
    try:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
    except Exception:
        return None

    command = getattr(command_term, "command", None)
    foot_thetas = getattr(command_term, "foot_thetas", None)
    duty_factors = getattr(command_term, "duty_factors", None)
    gait_periods = getattr(command_term, "gait_periods", None)
    if command is None or foot_thetas is None or duty_factors is None or gait_periods is None:
        return None

    command_b = command[env_index].detach().cpu()
    foot_theta = foot_thetas[env_index].detach().cpu()
    gait_name = _classify_symmetric_gait(foot_theta)
    duty_factor = float(duty_factors[env_index].detach().cpu())
    gait_period = float(gait_periods[env_index].detach().cpu())

    robot = getattr(command_term, "robot", None)
    if robot is not None:
        lin_vel_b = robot.data.root_lin_vel_b.torch[env_index, :2].detach().cpu()
        yaw_vel_b = float(robot.data.root_ang_vel_b.torch[env_index, 2].detach().cpu())
    else:
        lin_vel_b = torch.zeros(2)
        yaw_vel_b = 0.0

    return (
        "[symm_locomotion] "
        f"gait={gait_name} "
        f"cmd=({command_b[0]:+.2f}, {command_b[1]:+.2f}, {command_b[2]:+.2f}) "
        f"vel=({lin_vel_b[0]:+.2f}, {lin_vel_b[1]:+.2f}, {yaw_vel_b:+.2f}) "
        f"duty={duty_factor:.3f} period={gait_period:.3f}s "
        f"theta=({foot_theta[0]:+.2f}, {foot_theta[1]:+.2f}, {foot_theta[2]:+.2f}, {foot_theta[3]:+.2f})"
    )


_LEG_USAGE_PROTOCOL_VERSION = "leg_usage_grid_v1"


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON object atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _append_grid_event(output_root: Path, event: dict[str, Any]) -> None:
    """Append one durable, line-delimited grid progress event."""
    event = {"timestamp": _utc_timestamp(), **event}
    with (output_root / "events.jsonl").open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(event, sort_keys=True) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _validate_leg_usage_gait_library(plan: dict[str, Any]) -> None:
    """Require selected study rows to match the current canonical training library."""
    plan_version = plan.get("gait_library_version")
    if plan_version != SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION:
        raise ValueError(
            "Leg-usage gait library version does not match current training code: "
            f"plan={plan_version!r}, current={SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION!r}."
        )
    for gait in plan["gaits"]:
        gait_index = int(gait["index"])
        if gait_index < 0 or gait_index >= len(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS):
            raise ValueError(f"Leg-usage gait index {gait_index} is not in the current training library.")
        expected_metadata = {
            "name": SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES[gait_index],
            "family": SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES[gait_index],
            "phases": tuple(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS[gait_index]),
            "weight": float(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS[gait_index]),
            "time_reversal_partner": int(SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS[gait_index]),
        }
        received_metadata = {
            "name": gait.get("name"),
            "family": gait.get("family"),
            "phases": tuple(float(value) for value in gait["phases"]),
            "weight": float(gait.get("weight", float("nan"))),
            "time_reversal_partner": gait.get("time_reversal_partner"),
        }
        if received_metadata != expected_metadata:
            raise ValueError(
                f"Leg-usage gait {gait_index} does not match the current canonical training row: "
                f"plan={received_metadata}, current={expected_metadata}."
            )


def _load_leg_usage_plan(plan_value: str) -> tuple[dict[str, Any], Path, str]:
    """Load and validate the immutable portions of a leg-usage study plan."""
    plan_path = Path(plan_value).expanduser().resolve()
    raw_plan = plan_path.read_bytes()
    try:
        plan = json.loads(raw_plan)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Leg-usage plan is not valid JSON: {plan_path}") from exc
    if not isinstance(plan, dict):
        raise ValueError("Leg-usage plan must contain one JSON object.")
    if plan.get("schema_version") != 1:
        raise ValueError(f"Unsupported leg-usage plan schema_version: {plan.get('schema_version')!r}.")
    gaits = plan.get("gaits")
    cells = plan.get("cells")
    if not isinstance(gaits, list) or not gaits:
        raise ValueError("Leg-usage plan must define at least one gait.")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Leg-usage plan must define at least one grid cell.")

    gait_indices: set[int] = set()
    gait_weight_sum = 0.0
    for gait in gaits:
        if not isinstance(gait, dict):
            raise ValueError("Each leg-usage gait must be a JSON object.")
        gait_index = int(gait["index"])
        phases = gait.get("phases")
        if gait_index in gait_indices:
            raise ValueError(f"Duplicate leg-usage gait index: {gait_index}.")
        if not isinstance(phases, list) or len(phases) != 4 or not all(math.isfinite(float(x)) for x in phases):
            raise ValueError(f"Gait {gait_index} must define four finite phase offsets.")
        gait_weight = float(gait.get("weight", 1.0))
        if not math.isfinite(gait_weight) or gait_weight < 0.0:
            raise ValueError(f"Gait {gait_index} must define a finite, nonnegative weight.")
        gait_weight_sum += gait_weight
        gait_indices.add(gait_index)
    if gait_weight_sum <= 0.0:
        raise ValueError("Leg-usage gait weights must have a positive sum.")
    cell_ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Each leg-usage cell must be a JSON object.")
        if not isinstance(cell.get("id"), str):
            raise ValueError("Each leg-usage cell id must be a string.")
        cell_id = cell["id"]
        gait_index = int(cell["gait_index"])
        velocity_mps = float(cell.get("velocity_mps", cell.get("vx_mps", float("nan"))))
        if not cell_id or cell_id in cell_ids:
            raise ValueError(f"Leg-usage cell IDs must be unique; received {cell_id!r}.")
        if gait_index not in gait_indices:
            raise ValueError(f"Cell {cell_id!r} references unknown gait index {gait_index}.")
        if not math.isfinite(velocity_mps):
            raise ValueError(f"Cell {cell_id!r} must define a finite velocity_mps.")
        if not isinstance(cell.get("relative_output_dir"), str) or not cell["relative_output_dir"]:
            raise ValueError(f"Cell {cell_id!r} must define relative_output_dir.")
        cell_ids.add(cell_id)

    output_root = plan_path.parent
    declared_output_root = plan.get("output_root")
    if declared_output_root is not None:
        declared_path = Path(str(declared_output_root)).expanduser()
        if not declared_path.is_absolute():
            declared_path = plan_path.parent / declared_path
        if declared_path.resolve() != output_root:
            raise ValueError(
                "Leg-usage output_root must be the directory containing the plan: "
                f"{output_root}, received {declared_path.resolve()}."
            )
    settle_s = float(plan.get("settle_s", 0.0))
    measure_s = float(plan.get("measure_s", 0.0))
    if not math.isfinite(settle_s) or not math.isfinite(measure_s) or settle_s < 0.0 or measure_s <= 0.0:
        raise ValueError("Leg-usage settle_s must be nonnegative and measure_s must be positive.")
    cell_output_dirs: set[Path] = set()
    for cell in cells:
        cell_output_dir = _resolve_cell_output_dir(output_root, cell["relative_output_dir"])
        if cell_output_dir == output_root:
            raise ValueError(f"Cell {cell['id']!r} output directory must be below the study root.")
        if cell_output_dir in cell_output_dirs:
            raise ValueError(f"Leg-usage cells must use unique output directories: {cell['relative_output_dir']!r}.")
        cell_output_dirs.add(cell_output_dir)
    _validate_leg_usage_gait_library(plan)
    return plan, plan_path, hashlib.sha256(raw_plan).hexdigest()


def _load_optional_leg_usage_plan(
    plan_value: str | None,
    *,
    video_enabled: bool,
    task: str | None,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    """Load an optional leg-usage plan and reject incompatible video capture."""
    if plan_value is None:
        return None, None, None
    if video_enabled:
        raise ValueError("The leg-usage grid is data-only; --video is not supported with --symm_leg_usage_plan.")
    plan, plan_path, plan_sha256 = _load_leg_usage_plan(plan_value)
    _validate_leg_usage_task(plan, task)
    return plan, plan_path, plan_sha256


def _validate_leg_usage_task(plan: dict[str, Any], task: str | None) -> None:
    """Verify the launched environment task is the task recorded by the study."""
    planned_task = plan.get("task")
    if not isinstance(planned_task, str) or not planned_task:
        raise ValueError("Leg-usage plan must declare its task.")
    if task != planned_task:
        raise ValueError(f"Launched task {task!r} does not match the leg-usage plan task {planned_task!r}.")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_leg_usage_checkpoint(plan: dict[str, Any], resume_path: str) -> None:
    """Verify the loaded checkpoint is exactly the artifact declared by the study."""
    checkpoint = plan.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("path") or not checkpoint.get("sha256"):
        raise ValueError("Leg-usage plan checkpoint metadata must include path and sha256.")
    planned_path = Path(str(checkpoint["path"])).expanduser().resolve()
    loaded_path = Path(resume_path).expanduser().resolve()
    if loaded_path != planned_path:
        raise ValueError(
            f"Loaded checkpoint {loaded_path} does not match the leg-usage plan checkpoint {planned_path}."
        )
    planned_sha256 = str(checkpoint["sha256"]).lower()
    loaded_sha256 = _sha256_file(loaded_path)
    if loaded_sha256 != planned_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {loaded_path}: plan={planned_sha256}, loaded={loaded_sha256}."
        )


def _leg_usage_step_counts(plan: dict[str, Any], step_dt: float) -> tuple[int, int, int]:
    """Resolve settle, measurement, and total step counts for the actual environment time step."""
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"Environment step_dt must be finite and positive; received {step_dt!r}.")
    planned_step_dt = plan.get("step_dt")
    if planned_step_dt is not None and not math.isclose(float(planned_step_dt), step_dt, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Leg-usage plan step_dt={planned_step_dt} does not match environment step_dt={step_dt}.")
    settle_steps = int(plan.get("settle_steps", round(float(plan["settle_s"]) / step_dt)))
    measure_steps = int(plan.get("measure_steps", round(float(plan["measure_s"]) / step_dt)))
    total_steps = int(plan.get("total_steps", settle_steps + measure_steps))
    if settle_steps < 0 or measure_steps < 1 or total_steps != settle_steps + measure_steps:
        raise ValueError(
            "Leg-usage step counts must satisfy total_steps == settle_steps + measure_steps with a positive "
            "measurement window."
        )
    return settle_steps, measure_steps, total_steps


def _configure_nominal_leg_usage_profile(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Remove reset and perturbation randomness from a manager-based evaluation config."""
    events = getattr(env_cfg, "events", None)
    if events is not None:
        for event_name in (
            "physics_material",
            "add_base_mass",
            "base_com",
            "base_external_force_torque",
            "push_robot",
        ):
            if hasattr(events, event_name):
                setattr(events, event_name, None)

        reset_base = getattr(events, "reset_base", None)
        if reset_base is not None:
            reset_base.params["pose_range"] = {}
            reset_base.params["velocity_range"] = {
                component: (0.0, 0.0) for component in ("x", "y", "z", "roll", "pitch", "yaw")
            }
        reset_robot_joints = getattr(events, "reset_robot_joints", None)
        if reset_robot_joints is not None:
            reset_robot_joints.params["position_range"] = (1.0, 1.0)
            reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group_name in dir(observations):
            if group_name.startswith("_"):
                continue
            group_cfg = getattr(observations, group_name)
            if hasattr(group_cfg, "enable_corruption"):
                group_cfg.enable_corruption = False


def _ordered_leg_usage_gaits(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured gait rows ordered by their stable gait identifier."""
    return sorted(plan["gaits"], key=lambda gait: int(gait["index"]))


def _configured_gait_index_by_id(plan: dict[str, Any]) -> dict[int, int]:
    """Map stable gait identifiers to indices in the installed command library."""
    return {int(gait["index"]): row_index for row_index, gait in enumerate(_ordered_leg_usage_gaits(plan))}


def _configure_leg_usage_env(env_cfg: ManagerBasedRLEnvCfg, plan: dict[str, Any]) -> None:
    """Apply the fixed-command grid protocol before environment construction."""
    env_cfg.scene.num_envs = 1
    command_cfg = env_cfg.commands.base_velocity
    ordered_gaits = _ordered_leg_usage_gaits(plan)
    gait_rows = tuple(tuple(float(phase) for phase in gait["phases"]) for gait in ordered_gaits)
    gait_weights = tuple(float(gait.get("weight", 1.0)) for gait in ordered_gaits)
    command_cfg.init_foot_thetas = gait_rows
    command_cfg.init_foot_theta_weights = gait_weights
    if plan.get("gait_library_version") is not None:
        command_cfg.gait_library_version = str(plan["gait_library_version"])
    command_cfg.gait_sequence_enabled = False
    command_cfg.add_noise_period = False
    command_cfg.add_noise_theta = False
    command_cfg.resampling_time_gait = 0.0
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if command_cfg.ranges.heading is not None:
        command_cfg.ranges.heading = (0.0, 0.0)

    nominal_step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    _, _, total_steps = _leg_usage_step_counts(plan, nominal_step_dt)
    env_cfg.episode_length_s = (total_steps + 2) * nominal_step_dt
    command_cfg.resampling_time_range = (env_cfg.episode_length_s, env_cfg.episode_length_s)
    if bool(plan.get("nominal_profile", True)):
        _configure_nominal_leg_usage_profile(env_cfg)


def _resolve_cell_output_dir(output_root: Path, relative_output_dir: str) -> Path:
    """Resolve a cell directory while preventing traversal outside the study root."""
    relative_path = Path(relative_output_dir)
    if relative_path.is_absolute():
        raise ValueError(f"Cell relative_output_dir must be relative: {relative_output_dir!r}.")
    cell_output_dir = (output_root / relative_path).resolve()
    if not cell_output_dir.is_relative_to(output_root):
        raise ValueError(f"Cell output path escapes the study root: {relative_output_dir!r}.")
    return cell_output_dir


def _completed_cell_archive_matches(
    archive,
    cell: dict[str, Any],
    plan_sha256: str,
    *,
    recorded_steps: int,
    expected_steps: int,
    outcome: str,
    planned_step_dt: float,
    expected_velocity: float,
    expected_phases: np.ndarray,
) -> bool:
    """Validate required arrays and protocol metadata in a completed cell archive."""
    required_fields = {
        "time_steps",
        "desired_lin_vel",
        "true_lin_vel",
        "base_positions",
        "joint_torques",
        "joint_powers",
        "joint_effort_limits",
        "foot_ground_reaction_forces_w",
        "episode_done",
        "foot_thetas",
        "gait_periods",
        "duty_factors",
        "common_gait_phases",
        "configured_foot_thetas",
        "gait_index",
        "velocity_mps",
        "step_dt",
        "expected_steps",
        "recorded_steps",
        "cell_id",
        "plan_sha256",
    }
    if not required_fields.issubset(archive.files):
        return False
    scalar_checks = (
        str(archive["cell_id"].item()) == str(cell["id"]),
        str(archive["plan_sha256"].item()) == plan_sha256,
        int(archive["gait_index"].item()) == int(cell["gait_index"]),
        math.isclose(float(archive["velocity_mps"].item()), expected_velocity, abs_tol=1e-9),
        math.isclose(float(archive["step_dt"].item()), planned_step_dt, abs_tol=1e-9),
        int(archive["expected_steps"].item()) == expected_steps,
        int(archive["recorded_steps"].item()) == recorded_steps,
    )
    if not all(scalar_checks):
        return False

    time_steps = np.asarray(archive["time_steps"], dtype=np.float64)
    commands = np.asarray(archive["desired_lin_vel"], dtype=np.float64)
    true_velocities = np.asarray(archive["true_lin_vel"], dtype=np.float64)
    base_positions = np.asarray(archive["base_positions"], dtype=np.float64)
    joint_torques = np.asarray(archive["joint_torques"], dtype=np.float64)
    joint_powers = np.asarray(archive["joint_powers"], dtype=np.float64)
    joint_effort_limits = np.asarray(archive["joint_effort_limits"], dtype=np.float64)
    ground_reaction_forces = np.asarray(archive["foot_ground_reaction_forces_w"], dtype=np.float64)
    episode_done = np.asarray(archive["episode_done"], dtype=bool)
    foot_thetas = np.asarray(archive["foot_thetas"], dtype=np.float64)
    gait_periods = np.asarray(archive["gait_periods"], dtype=np.float64)
    duty_factors = np.asarray(archive["duty_factors"], dtype=np.float64)
    common_gait_phases = np.asarray(archive["common_gait_phases"], dtype=np.float64)
    configured_foot_thetas = np.asarray(archive["configured_foot_thetas"], dtype=np.float64)
    shapes_match = (
        time_steps.shape == (recorded_steps,)
        and commands.shape == (recorded_steps, 3)
        and true_velocities.shape == (recorded_steps, 3)
        and base_positions.ndim == 2
        and base_positions.shape[0] == recorded_steps
        and base_positions.shape[1] >= 2
        and joint_torques.shape == (recorded_steps, 12)
        and joint_powers.shape == (recorded_steps, 12)
        and joint_effort_limits.shape == (12,)
        and ground_reaction_forces.shape == (recorded_steps, 4, 3)
        and episode_done.shape == (recorded_steps,)
        and foot_thetas.shape == (recorded_steps, 4)
        and gait_periods.shape == (recorded_steps,)
        and duty_factors.shape == (recorded_steps,)
        and common_gait_phases.shape == (recorded_steps,)
        and configured_foot_thetas.shape == (4,)
    )
    if not shapes_match:
        return False
    numeric_arrays = (
        time_steps,
        commands,
        true_velocities,
        base_positions,
        joint_torques,
        joint_powers,
        joint_effort_limits,
        ground_reaction_forces,
        foot_thetas,
        gait_periods,
        duty_factors,
        common_gait_phases,
        configured_foot_thetas,
    )
    if any(not np.all(np.isfinite(values)) for values in numeric_arrays):
        return False
    if np.any(joint_effort_limits <= 0.0):
        return False
    if not np.allclose(commands[:, 0], expected_velocity, rtol=0.0, atol=1e-5):
        return False
    if not np.allclose(commands[:, 1:], 0.0, rtol=0.0, atol=1e-5):
        return False
    for recorded_phases in (configured_foot_thetas.reshape(1, 4), foot_thetas):
        circular_error = np.remainder(recorded_phases - expected_phases + 0.5, 1.0) - 0.5
        if not np.allclose(circular_error, 0.0, rtol=0.0, atol=1e-5):
            return False
    if np.any(gait_periods <= 0.0) or np.any((duty_factors <= 0.0) | (duty_factors >= 1.0)):
        return False
    if recorded_steps > 1:
        timing_is_fixed = (
            np.allclose(np.diff(time_steps), planned_step_dt, rtol=0.0, atol=1e-7)
            and np.allclose(gait_periods, gait_periods[0], rtol=0.0, atol=1e-7)
            and np.allclose(duty_factors, duty_factors[0], rtol=0.0, atol=1e-7)
        )
        if not timing_is_fixed:
            return False
    if outcome == "completed":
        return not bool(np.any(episode_done)) and not bool(np.any(np.diff(common_gait_phases) < 0.0))
    return bool(episode_done[-1]) and not bool(np.any(episode_done[:-1]))


def _validated_completed_cell(cell_dir: Path, cell: dict[str, Any], plan_sha256: str) -> bool:
    """Return whether a cell has a complete, internally consistent artifact set."""
    status_path = cell_dir / "status.json"
    metadata_path = cell_dir / "metadata.json"
    data_path = cell_dir / "sim_data.npz"
    if not status_path.is_file() or not metadata_path.is_file() or not data_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("cell_id") != cell["id"]:
            return False
        if status.get("plan_sha256") != plan_sha256 or metadata_value.get("plan_sha256") != plan_sha256:
            return False
        recorded_steps = int(status["recorded_steps"])
        expected_steps = int(status["expected_steps"])
        planned_steps = int(cell["total_steps"])
        outcome = status.get("outcome")
        if recorded_steps < 1 or expected_steps != planned_steps:
            return False
        planned_step_dt = float(cell["step_dt"])
        expected_velocity = float(cell.get("velocity_mps", cell.get("vx_mps")))
        expected_phases = np.asarray(cell["phases"], dtype=np.float64)
        metadata_checks = (
            metadata_value.get("cell_id") == cell["id"],
            int(metadata_value.get("gait_index", -1)) == int(cell["gait_index"]),
            math.isclose(float(metadata_value.get("velocity_mps", float("nan"))), expected_velocity, abs_tol=1e-9),
            math.isclose(float(metadata_value.get("step_dt", float("nan"))), planned_step_dt, abs_tol=1e-9),
            int(metadata_value.get("expected_steps", -1)) == expected_steps,
            int(metadata_value.get("recorded_steps", -1)) == recorded_steps,
            metadata_value.get("outcome") == outcome,
        )
        if not all(metadata_checks):
            return False
        metadata_phases = np.asarray(metadata_value.get("gait_phases", []), dtype=np.float64)
        if metadata_phases.shape != (4,) or not np.array_equal(metadata_phases, expected_phases):
            return False
        if outcome == "completed" and recorded_steps != expected_steps:
            return False
        if outcome == "terminated" and not status.get("termination_terms"):
            return False
        if outcome not in {"completed", "terminated"}:
            return False
        with np.load(data_path, allow_pickle=False) as archive:
            return _completed_cell_archive_matches(
                archive,
                cell,
                plan_sha256,
                recorded_steps=recorded_steps,
                expected_steps=expected_steps,
                outcome=outcome,
                planned_step_dt=planned_step_dt,
                expected_velocity=expected_velocity,
                expected_phases=expected_phases,
            )
    except (EOFError, IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
        return False
    return False


def _format_eta(seconds: float) -> str:
    """Format an ETA duration compactly."""
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    rounded_seconds = int(round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


def _active_termination_terms(env, env_index: int = 0) -> list[str]:
    """Return termination terms that fired on the most recent step."""
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    if termination_manager is None:
        return []
    active_terms: list[str] = []
    last_episode_dones = getattr(termination_manager, "_last_episode_dones", None)
    for term_index, term_name in enumerate(termination_manager.active_terms):
        try:
            fired = (
                bool(last_episode_dones[env_index, term_index].detach().cpu())
                if last_episode_dones is not None
                else bool(termination_manager.get_term(term_name)[env_index].detach().cpu())
            )
        except (IndexError, KeyError):
            fired = False
        if fired:
            active_terms.append(term_name)
    return active_terms


def _reset_inference_policy(policy, policy_nn, num_envs: int, device: str | torch.device) -> None:
    """Clear recurrent inference state for a fresh grid cell."""
    reset_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    if version.parse(installed_version) >= version.parse("4.0.0"):
        policy.reset(reset_mask)
    elif policy_nn is not None:
        policy_nn.reset(reset_mask)


def _write_grid_progress(
    output_root: Path,
    *,
    status: str,
    total_cells: int,
    completed_cells: int,
    skipped_cells: int,
    terminated_cells: int,
    current_cell_id: str | None,
    message: str | None = None,
) -> None:
    """Write the resumable root progress snapshot."""
    progress = {
        "schema_version": 1,
        "status": status,
        "updated_at": _utc_timestamp(),
        "total_cells": total_cells,
        "completed_cells": completed_cells,
        "skipped_cells": skipped_cells,
        "terminated_cells": terminated_cells,
        "successful_cells": completed_cells - terminated_cells,
        "current_cell_id": current_cell_id,
    }
    if message is not None:
        progress["message"] = message
    _atomic_write_json(output_root / "progress.json", progress)


@contextlib.contextmanager
def _leg_usage_recording_lock(output_root: Path, plan_sha256: str):
    """Hold an exclusive recording lock for one fixed study output root."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".recording.lock"
    lock_value = {
        "schema_version": 1,
        "pid": os.getpid(),
        "started_at": _utc_timestamp(),
        "plan_sha256": plan_sha256,
    }
    serialized_lock = json.dumps(lock_value, indent=2, sort_keys=True) + "\n"
    try:
        file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            existing_lock = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing_lock = "<unreadable>"
        raise RuntimeError(
            f"Leg-usage output root is locked by another or interrupted recorder: {lock_path}. "
            f"Lock contents: {existing_lock}. Remove this file deliberately only after confirming no recorder is "
            "active."
        ) from exc
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as lock_file:
            lock_file.write(serialized_lock)
            lock_file.flush()
            os.fsync(lock_file.fileno())
        yield
    finally:
        try:
            if lock_path.read_text(encoding="utf-8") == serialized_lock:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _run_leg_usage_grid(
    env,
    policy,
    policy_nn,
    plan: dict[str, Any],
    plan_path: Path,
    plan_sha256: str,
) -> None:
    """Run a fixed gait-by-velocity grid in one persistent simulator process."""
    from symm_rollout_plotter import SymmetricRolloutPlotter

    output_root = plan_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    step_dt = float(env.unwrapped.step_dt)
    settle_steps, measure_steps, total_steps = _leg_usage_step_counts(plan, step_dt)
    cells = plan["cells"]
    gait_by_index = {int(gait["index"]): gait for gait in plan["gaits"]}
    configured_gait_index_by_id = _configured_gait_index_by_id(plan)
    protocol_version = str(plan.get("protocol_version", plan.get("method_version", _LEG_USAGE_PROTOCOL_VERSION)))
    render_cell_plots = bool(plan.get("render_cell_plots", False))
    resume_enabled = bool(plan.get("resume", True))

    pending_cells: list[tuple[dict[str, Any], Path]] = []
    skipped_cells = 0
    terminated_cells = 0
    for cell in cells:
        cell_output_dir = _resolve_cell_output_dir(output_root, cell["relative_output_dir"])
        if _validated_completed_cell(cell_output_dir, cell, plan_sha256):
            if not resume_enabled:
                raise FileExistsError(
                    f"Validated results already exist for {cell['id']!r}; enable resume or use a fresh output root."
                )
            skipped_cells += 1
            status = json.loads((cell_output_dir / "status.json").read_text(encoding="utf-8"))
            terminated_cells += int(status.get("outcome") == "terminated")
        else:
            pending_cells.append((cell, cell_output_dir))

    total_cells = len(cells)
    _write_grid_progress(
        output_root,
        status="running",
        total_cells=total_cells,
        completed_cells=skipped_cells,
        skipped_cells=skipped_cells,
        terminated_cells=terminated_cells,
        current_cell_id=None,
    )
    _append_grid_event(
        output_root,
        {
            "event": "grid_started",
            "plan_sha256": plan_sha256,
            "total_cells": total_cells,
            "pending_cells": len(pending_cells),
            "skipped_cells": skipped_cells,
        },
    )
    print(
        f"[leg_usage_grid] {len(pending_cells)} pending, {skipped_cells} validated cells skipped, "
        f"{total_cells} total; {total_steps} steps/cell ({total_steps * step_dt:.2f} s).",
        flush=True,
    )
    if not pending_cells:
        _write_grid_progress(
            output_root,
            status="complete",
            total_cells=total_cells,
            completed_cells=total_cells,
            skipped_cells=skipped_cells,
            terminated_cells=terminated_cells,
            current_cell_id=None,
        )
        _append_grid_event(output_root, {"event": "grid_completed", "plan_sha256": plan_sha256})
        print("[leg_usage_grid] All cells are already complete and validated.", flush=True)
        return

    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    set_evaluation_scenario = getattr(command_term, "set_evaluation_scenario", None)
    if set_evaluation_scenario is None:
        error = RuntimeError(
            "The base_velocity command term does not implement set_evaluation_scenario(); "
            "the fixed-command leg-usage grid cannot run safely."
        )
        _write_grid_progress(
            output_root,
            status="error",
            total_cells=total_cells,
            completed_cells=skipped_cells,
            skipped_cells=skipped_cells,
            terminated_cells=terminated_cells,
            current_cell_id=None,
            message=str(error),
        )
        raise error

    grid_wall_start = time.perf_counter()
    finished_pending_cells = 0
    progress_interval_steps = max(1, round(1.0 / step_dt))
    current_cell_id = None
    try:
        for cell, cell_output_dir in pending_cells:
            cell_output_dir.mkdir(parents=True, exist_ok=True)
            cell_id = str(cell["id"])
            current_cell_id = cell_id
            gait_index = int(cell["gait_index"])
            configured_gait_index = configured_gait_index_by_id[gait_index]
            gait = gait_by_index[gait_index]
            velocity_mps = float(cell.get("velocity_mps", cell.get("vx_mps")))
            evaluation_seed = int(cell.get("seed", plan.get("evaluation_seed", 0)))
            global_index = next(index for index, candidate in enumerate(cells) if candidate["id"] == cell_id)
            cell_started_at = _utc_timestamp()
            cell_wall_start = time.perf_counter()
            running_status = {
                "schema_version": 1,
                "status": "running",
                "cell_id": cell_id,
                "plan_sha256": plan_sha256,
                "started_at": cell_started_at,
                "expected_steps": total_steps,
            }
            _atomic_write_json(cell_output_dir / "status.json", running_status)
            _write_grid_progress(
                output_root,
                status="running",
                total_cells=total_cells,
                completed_cells=skipped_cells + finished_pending_cells,
                skipped_cells=skipped_cells,
                terminated_cells=terminated_cells,
                current_cell_id=cell_id,
            )
            _append_grid_event(
                output_root,
                {
                    "event": "cell_started",
                    "cell_id": cell_id,
                    "cell_index": global_index,
                    "gait_index": gait_index,
                    "velocity_mps": velocity_mps,
                    "seed": evaluation_seed,
                },
            )

            outcome = "completed"
            termination_terms: list[str] = []
            recorded_steps = 0
            caught_exception: BaseException | None = None
            rollout_plotter = None
            try:
                random.seed(evaluation_seed)
                np.random.seed(evaluation_seed % (2**32))
                torch.manual_seed(evaluation_seed)
                env.seed(evaluation_seed)
                set_evaluation_scenario(velocity_mps, configured_gait_index)
                obs, _ = env.reset()
                _reset_inference_policy(policy, policy_nn, env.num_envs, env.unwrapped.device)
                rollout_plotter = SymmetricRolloutPlotter(
                    env.unwrapped,
                    output_dir=cell_output_dir,
                    env_index=0,
                    max_samples=total_steps,
                )
                for step_index in range(total_steps):
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, rewards, dones, _ = env.step(actions)
                    rollout_plotter.record(actions=actions, actor_means=actions, dones=dones, rewards=rewards)
                    recorded_steps = step_index + 1

                    if bool(dones[0].detach().cpu()):
                        termination_terms = _active_termination_terms(env)
                        outcome = "terminated"
                        _reset_inference_policy(policy, policy_nn, env.num_envs, env.unwrapped.device)
                        break

                    if recorded_steps % progress_interval_steps == 0 or recorded_steps == total_steps:
                        completed_equivalent = finished_pending_cells + recorded_steps / total_steps
                        elapsed = time.perf_counter() - grid_wall_start
                        seconds_per_cell = elapsed / max(completed_equivalent, 1e-9)
                        remaining_equivalent = len(pending_cells) - completed_equivalent
                        eta = _format_eta(seconds_per_cell * max(remaining_equivalent, 0.0))
                        print(
                            f"[leg_usage_grid] cell {global_index + 1}/{total_cells} {cell_id} "
                            f"sim={recorded_steps * step_dt:.1f}/{total_steps * step_dt:.1f} s "
                            f"({recorded_steps}/{total_steps}) ETA={eta}",
                            flush=True,
                        )
            except BaseException as exc:
                caught_exception = exc
                outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"

            cell_ended_at = _utc_timestamp()
            metadata_value = {
                "schema_version": 1,
                "protocol_version": protocol_version,
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha256,
                "cell_id": cell_id,
                "cell_index": global_index,
                "gait_index": gait_index,
                "configured_gait_index": configured_gait_index,
                "gait_family": str(cell.get("gait_family", gait.get("family", "unknown"))),
                "gait_name": str(cell.get("gait_name", gait.get("name", f"gait_{gait_index:02d}"))),
                "gait_phases": [float(value) for value in cell.get("phases", gait["phases"])],
                "gait_weight": float(gait.get("weight", 1.0)),
                "velocity_mps": velocity_mps,
                "evaluation_seed": evaluation_seed,
                "step_dt": step_dt,
                "settle_s": settle_steps * step_dt,
                "measure_s": measure_steps * step_dt,
                "settle_steps": settle_steps,
                "measure_steps": measure_steps,
                "expected_steps": total_steps,
                "recorded_steps": recorded_steps,
                "measurement_start_step": settle_steps,
                "measurement_stop_step": total_steps,
                "nominal_profile": bool(plan.get("nominal_profile", True)),
                "checkpoint": plan.get("checkpoint", {}),
                "outcome": outcome,
                "termination_terms": termination_terms,
                "termination_time_s": recorded_steps * step_dt if outcome == "terminated" else None,
                "started_at": cell_started_at,
                "ended_at": cell_ended_at,
                "wall_duration_s": time.perf_counter() - cell_wall_start,
            }
            if caught_exception is not None:
                metadata_value["error"] = f"{type(caught_exception).__name__}: {caught_exception}"
            _atomic_write_json(cell_output_dir / "metadata.json", metadata_value)

            save_exception: BaseException | None = None
            if recorded_steps > 0 and rollout_plotter is not None:
                extra_data = {
                    "protocol_version": protocol_version,
                    "plan_sha256": plan_sha256,
                    "cell_id": cell_id,
                    "gait_index": gait_index,
                    "configured_gait_index": configured_gait_index,
                    "gait_family": metadata_value["gait_family"],
                    "gait_name": metadata_value["gait_name"],
                    "configured_foot_thetas": np.asarray(metadata_value["gait_phases"], dtype=np.float64),
                    "gait_weight": metadata_value["gait_weight"],
                    "velocity_mps": velocity_mps,
                    "evaluation_seed": evaluation_seed,
                    "step_dt": step_dt,
                    "settle_steps": settle_steps,
                    "measure_steps": measure_steps,
                    "expected_steps": total_steps,
                    "recorded_steps": recorded_steps,
                    "measurement_start_step": settle_steps,
                    "measurement_stop_step": total_steps,
                    "nominal_profile": metadata_value["nominal_profile"],
                    "outcome": outcome,
                    "termination_terms_json": json.dumps(termination_terms, sort_keys=True),
                    "checkpoint_json": json.dumps(plan.get("checkpoint", {}), sort_keys=True),
                    "cell_metadata_json": json.dumps(metadata_value, sort_keys=True),
                }
                try:
                    rollout_plotter.save(extra_data=extra_data, render_plots=render_cell_plots)
                except BaseException as exc:
                    save_exception = exc
                    outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"

            cell_ended_at = _utc_timestamp()
            metadata_value["ended_at"] = cell_ended_at
            metadata_value["wall_duration_s"] = time.perf_counter() - cell_wall_start
            metadata_value["outcome"] = outcome
            if save_exception is not None:
                metadata_value["error"] = f"{type(save_exception).__name__}: {save_exception}"
            _atomic_write_json(cell_output_dir / "metadata.json", metadata_value)

            completed_cell = outcome in {"completed", "terminated"} and save_exception is None
            final_status = {
                **running_status,
                "status": "complete" if completed_cell else outcome,
                "outcome": outcome,
                "ended_at": cell_ended_at,
                "recorded_steps": recorded_steps,
                "termination_terms": termination_terms,
            }
            effective_exception = caught_exception if caught_exception is not None else save_exception
            if effective_exception is not None:
                final_status["error"] = f"{type(effective_exception).__name__}: {effective_exception}"
            _atomic_write_json(cell_output_dir / "status.json", final_status)
            _append_grid_event(
                output_root,
                {
                    "event": "cell_completed" if completed_cell else "cell_failed",
                    "cell_id": cell_id,
                    "status": final_status["status"],
                    "outcome": outcome,
                    "recorded_steps": recorded_steps,
                    "termination_terms": termination_terms,
                },
            )
            if effective_exception is not None:
                _write_grid_progress(
                    output_root,
                    status="interrupted" if isinstance(effective_exception, KeyboardInterrupt) else "error",
                    total_cells=total_cells,
                    completed_cells=skipped_cells + finished_pending_cells,
                    skipped_cells=skipped_cells,
                    terminated_cells=terminated_cells,
                    current_cell_id=cell_id,
                    message=str(effective_exception),
                )
                raise effective_exception

            finished_pending_cells += 1
            terminated_cells += int(outcome == "terminated")
            print(
                f"[leg_usage_grid] completed cell {global_index + 1}/{total_cells} {cell_id}: "
                f"outcome={outcome}, samples={recorded_steps}.",
                flush=True,
            )

    except KeyboardInterrupt as exc:
        _write_grid_progress(
            output_root,
            status="interrupted",
            total_cells=total_cells,
            completed_cells=skipped_cells + finished_pending_cells,
            skipped_cells=skipped_cells,
            terminated_cells=terminated_cells,
            current_cell_id=current_cell_id,
            message=str(exc) or "KeyboardInterrupt",
        )
        _append_grid_event(output_root, {"event": "grid_interrupted", "cell_id": current_cell_id})
        print("[leg_usage_grid] Interrupted; partial artifacts are saved and can be resumed.", flush=True)
        raise
    except Exception as exc:
        _write_grid_progress(
            output_root,
            status="error",
            total_cells=total_cells,
            completed_cells=skipped_cells + finished_pending_cells,
            skipped_cells=skipped_cells,
            terminated_cells=terminated_cells,
            current_cell_id=current_cell_id,
            message=str(exc),
        )
        _append_grid_event(
            output_root,
            {"event": "grid_failed", "cell_id": current_cell_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        print(f"[leg_usage_grid] Stopped after an error: {exc}", flush=True)
        raise

    _write_grid_progress(
        output_root,
        status="complete",
        total_cells=total_cells,
        completed_cells=total_cells,
        skipped_cells=skipped_cells,
        terminated_cells=terminated_cells,
        current_cell_id=None,
    )
    _append_grid_event(
        output_root,
        {
            "event": "grid_completed",
            "plan_sha256": plan_sha256,
            "wall_duration_s": time.perf_counter() - grid_wall_start,
        },
    )
    print(
        f"[leg_usage_grid] Completed {total_cells} cells in {_format_eta(time.perf_counter() - grid_wall_start)}.",
        flush=True,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    leg_usage_plan, leg_usage_plan_path, leg_usage_plan_sha256 = _load_optional_leg_usage_plan(
        args_cli.symm_leg_usage_plan,
        video_enabled=args_cli.video,
        task=args_cli.task,
    )
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        if leg_usage_plan is not None:
            if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
                raise TypeError("The symmetric leg-usage grid requires a ManagerBasedRLEnvCfg task.")
            evaluation_seed = int(leg_usage_plan.get("evaluation_seed", agent_cfg.seed))
            agent_cfg.seed = evaluation_seed
            _configure_leg_usage_env(env_cfg, leg_usage_plan)

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if leg_usage_plan is not None:
            _validate_leg_usage_checkpoint(leg_usage_plan, resume_path)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        visualizers = args_cli.visualizer or []
        if isinstance(visualizers, str):
            visualizers = [visualizers]
        render_mode = "rgb_array" if args_cli.video else ("human" if "kit" in visualizers else None)
        if leg_usage_plan is not None:
            render_mode = None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        if leg_usage_plan is not None:
            assert leg_usage_plan_path is not None
            assert leg_usage_plan_sha256 is not None
            try:
                with _leg_usage_recording_lock(leg_usage_plan_path.parent, leg_usage_plan_sha256):
                    _run_leg_usage_grid(
                        env,
                        policy,
                        policy_nn,
                        leg_usage_plan,
                        leg_usage_plan_path,
                        leg_usage_plan_sha256,
                    )
            finally:
                env.close()
            return

        rollout_plotter = None
        if args_cli.symm_rollout_plots:
            from symm_rollout_plotter import SymmetricRolloutPlotter

            plots_dir = args_cli.symm_rollout_plots_dir or os.path.join(log_dir, "plots", "play")
            rollout_plotter = SymmetricRolloutPlotter(
                env.unwrapped,
                output_dir=plots_dir,
                env_index=args_cli.symm_rollout_plot_env_index,
                max_samples=args_cli.symm_rollout_plot_max_steps,
            )

        # reset environment
        obs = env.get_observations()
        timestep = 0
        gait_info_interval = max(args_cli.print_gait_info_interval, 1)
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, rewards, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                if rollout_plotter is not None:
                    # In RSL-RL inference, the policy output is the deterministic actor mean.
                    rollout_plotter.record(actions=actions, actor_means=actions, dones=dones, rewards=rewards)

                timestep += 1
                if args_cli.print_gait_info and timestep % gait_info_interval == 0:
                    gait_info = _format_symmetric_gait_info(env)
                    if gait_info is not None:
                        print(gait_info, flush=True)
                if args_cli.video:
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            if rollout_plotter is not None:
                rollout_plotter.save()
            env.close()


if __name__ == "__main__":
    main()
