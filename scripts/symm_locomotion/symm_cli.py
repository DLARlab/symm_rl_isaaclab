# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for symmetric quadruped Isaac Lab tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CONDA_ENV = "symm_rl_isaaclab"
DEFAULT_NUM_ENVS = 512
DEFAULT_TRAINING_ITERATIONS = 20000
DEFAULT_MIRROR_LOSS_COEFF = 0.1
DEFAULT_TR_VALUE_COEFF = 0.05
DEFAULT_TR_WARMUP_ITERATIONS = 500
DEFAULT_TR_RAMPUP_ITERATIONS = 0
DEFAULT_TR_RAMP_SHAPE = "linear"
DEFAULT_TR_MIN_ABS_CMD_VEL = 0.0
DEFAULT_FOOT_PHASE_WEIGHT = 0.30
DEFAULT_FOOT_PHASE_REDUCTION = "sum"
DEFAULT_JOINT_TARGET_LIMIT_MODE = "legacy_clamped"
DEFAULT_ACTOR_MEAN_BOUND_MODE = "legacy_global"
DEFAULT_TR_POLICY_OUTPUT_SPACE = "raw_action_mean"
DEFAULT_GAIT_SAMPLING_PROFILE = "trclosed_v2_equal_family"
DEFAULT_GAIT_CURRICULUM_ITERATIONS = 0
DEFAULT_HISTORY_ENABLED = True
DEFAULT_HISTORY_LENGTH = 30
DEFAULT_HISTORY_TRS_MODE = "framewise_feature"
POLICY_OBSERVATION_CONTRACT_VERSION = "hardware_proprio_history_64d_v1"
POLICY_INSTANTANEOUS_FRAME_DIM = 64
POLICY_HISTORY_PACKING = "term_major_oldest_to_newest_flattened"
GAIT_PHASE_MAPPING_VERSION = "same_gait_backward_duty_aware_integrated_reward_boundary_v4"
GAIT_LIBRARY_VERSION = "time_reversal_closed_v2"
COMMAND_CURRICULUM_MODE_NONE = "none"
COMMAND_CURRICULUM_MODE_TR_ORBIT = "tr_orbit_reward_threshold_v1"
COMMAND_CURRICULUM_MODES = (COMMAND_CURRICULUM_MODE_NONE, COMMAND_CURRICULUM_MODE_TR_ORBIT)
DEFAULT_CURRICULUM_VELOCITY_BIN_COUNT = 11
DEFAULT_CURRICULUM_EWMA_COEFFICIENT = 0.1
DEFAULT_CURRICULUM_UNLOCK_THRESHOLD = 0.8
DEFAULT_CURRICULUM_NEIGHBOR_INCREMENT = 0.25
DEFAULT_CURRICULUM_SEED = 0
LEGACY_ABLATION_MIRROR_LOSS_COEFF = 0.2
LEGACY_ABLATION_TR_VALUE_COEFF = 0.05
DEFAULT_VIDEO_DURATION_S = 30.0
DEFAULT_GAIT_SEQUENCE_DURATION_S = 5.0
GAIT_SEQUENCE_COUNT = 6
DEFAULT_LEG_USAGE_VELOCITIES_MPS = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
DEFAULT_LEG_USAGE_SETTLE_S = 5.0
DEFAULT_LEG_USAGE_MEASURE_S = 10.0
DEFAULT_LEG_USAGE_EVALUATION_SEED = 42
LEG_USAGE_PROTECTED_RUNTIME_OPTIONS = {
    "--symm_leg_usage_plan",
    "--symm-leg-usage-plan",
    "--task",
    "--checkpoint",
    "--video",
    "--num_envs",
    "--num-envs",
    "--seed",
    "--rl_library",
    "--rl-library",
}
DEFAULT_WINDOWS_KIT_ARGS = "--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"

TR_RAMP_SHAPES = ("linear", "half_cosine")
FOOT_PHASE_REDUCTIONS = ("sum", "mean")
JOINT_TARGET_LIMIT_MODES = ("legacy_clamped", "requested_overflow")
ACTOR_MEAN_BOUND_MODES = ("legacy_global", "per_joint_feasible")
TR_POLICY_OUTPUT_SPACES = ("raw_action_mean", "normalized_requested_joint_target")
GAIT_SAMPLING_PROFILES = (
    "trclosed_v2_equal_family",
    "trclosed_v2_v1_equivalent",
    "trclosed_v2_halfbound_anneal",
)
TR_SCHEDULE_VARIANTS = ("no_trs", "hard", "linear", "delayed_linear", "half_cosine")
TR_SCHEDULE_SETTINGS = {
    "no_trs": (True, 0, 0, "linear"),
    "hard": (False, 500, 0, "linear"),
    "linear": (False, 0, 2000, "linear"),
    "delayed_linear": (False, 500, 1500, "linear"),
    "half_cosine": (False, 0, 2000, "half_cosine"),
}


@dataclass(frozen=True)
class RobotSpec:
    """Task metadata for one symmetric quadruped robot."""

    key: str
    label: str
    train_task: str
    play_task: str
    experiment_name: str
    step_dt: float


ROBOT_SPECS = {
    "go2": RobotSpec(
        key="go2",
        label="Unitree Go2",
        train_task="Isaac-Velocity-Flat-Unitree-Go2-Symm-v0",
        play_task="Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0",
        experiment_name="unitree_go2_symm_flat",
        step_dt=0.02,
    ),
    "x1": RobotSpec(
        key="x1",
        label="Dobot X1",
        train_task="Isaac-Velocity-Flat-Dobot-X1-Symm-v0",
        play_task="Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0",
        experiment_name="dobot_x1_symm_flat",
        step_dt=0.02,
    ),
}
ROBOT_ALIASES = {
    "unitree-go2": "go2",
    "unitree_go2": "go2",
    "dobot": "x1",
    "dobot-x1": "x1",
    "dobot_x1": "x1",
}
DEFAULT_ROBOT = "go2"


class _StoreExplicitAction(argparse.Action):
    """Store an argument and record that it was provided on the command line."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def repo_root() -> Path:
    """Return the Isaac Lab repository root."""
    return Path(__file__).resolve().parents[2]


def canonical_python_entrypoint(value: str) -> str:
    """Return a stable repository-relative label for the active Python entry point."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    try:
        return path.relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def repo_subprocess_environment() -> dict[str, str]:
    """Return an environment that imports packages from the launcher checkout.

    Returns:
        A copy of the current process environment with every local project under
        ``source`` placed before any existing ``PYTHONPATH`` entries.
    """
    environment = os.environ.copy()
    source_root = repo_root() / "source"
    local_projects = sorted(
        (
            path.resolve()
            for path in source_root.iterdir()
            if path.is_dir() and (path / path.name / "__init__.py").is_file()
        ),
        key=str,
    )
    inherited_paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    python_paths = list(
        dict.fromkeys([*(str(path) for path in local_projects), *(path for path in inherited_paths if path)])
    )
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def git_branch() -> str:
    """Return the branch checked out in the launcher repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_root()), "branch", "--show-current"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Unable to read the Git branch for {repo_root()}: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_expected_branch(args: argparse.Namespace) -> None:
    """Reject a launch from a branch other than the optional expected branch.

    Args:
        args: Parsed launcher arguments containing ``expected_branch``.

    Raises:
        ValueError: If the repository is detached or its branch does not match.
    """
    expected_branch = args.expected_branch
    if expected_branch is None:
        return
    branch = git_branch()
    if branch != expected_branch:
        actual = branch or "<detached HEAD>"
        raise ValueError(f"Expected branch '{expected_branch}', but current branch is '{actual}' in {repo_root()}.")


def robot_choices() -> tuple[str, ...]:
    """Return accepted robot keys and aliases."""
    return tuple(sorted((*ROBOT_SPECS.keys(), *ROBOT_ALIASES.keys())))


def get_robot(robot: str) -> RobotSpec:
    """Resolve a robot key or alias."""
    key = ROBOT_ALIASES.get(robot, robot)
    try:
        return ROBOT_SPECS[key]
    except KeyError as exc:
        choices = ", ".join(robot_choices())
        raise ValueError(f"Unknown robot '{robot}'. Choose one of: {choices}") from exc


def log_prefix(args: argparse.Namespace) -> str:
    """Return the terminal log prefix for the selected robot."""
    spec = getattr(args, "robot_spec", get_robot(getattr(args, "robot", DEFAULT_ROBOT)))
    return f"[symm_locomotion:{spec.key}] "


def coeff_label(value: float) -> str:
    """Convert a coefficient to a filename-friendly label."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def resolve_policy_history(args: argparse.Namespace) -> tuple[bool, int]:
    """Resolve and validate the policy-observation history requested by a launcher command."""
    enabled = bool(getattr(args, "history_enabled", DEFAULT_HISTORY_ENABLED))
    history_length = getattr(args, "history_length", DEFAULT_HISTORY_LENGTH)
    if isinstance(history_length, bool) or not isinstance(history_length, int):
        raise ValueError(f"--history_length must be an integer; received {history_length!r}.")
    if enabled and history_length <= 0:
        raise ValueError("--history_length must be a positive integer when history is enabled.")
    return enabled, history_length if enabled else 0


def policy_history_lab_args(args: argparse.Namespace) -> list[str]:
    """Build matching environment and TR-regularizer history overrides."""
    enabled, history_length = resolve_policy_history(args)
    enabled_label = str(enabled).lower()
    return [
        f"env.observations.policy.history_length={history_length}",
        "env.observations.policy.flatten_history_dim=true",
        f"agent.algorithm.symmetry_cfg.history_enabled={enabled_label}",
        f"agent.algorithm.symmetry_cfg.history_length={history_length}",
        f"agent.algorithm.symmetry_cfg.history_trs_mode={DEFAULT_HISTORY_TRS_MODE}",
    ]


def resolve_command_curriculum(args: argparse.Namespace) -> dict[str, int | float | str]:
    """Resolve and validate optional TR-orbit command-curriculum launcher settings."""
    mode = getattr(args, "command_curriculum_mode", COMMAND_CURRICULUM_MODE_NONE)
    if mode not in COMMAND_CURRICULUM_MODES:
        raise ValueError(f"--command_curriculum_mode must be one of {COMMAND_CURRICULUM_MODES}; received {mode!r}.")

    velocity_bin_count = getattr(args, "curriculum_velocity_bin_count", DEFAULT_CURRICULUM_VELOCITY_BIN_COUNT)
    if isinstance(velocity_bin_count, bool) or not isinstance(velocity_bin_count, int) or velocity_bin_count < 2:
        raise ValueError("--curriculum_velocity_bins must be an integer greater than or equal to two.")

    ewma_coefficient = getattr(args, "curriculum_ewma_coefficient", DEFAULT_CURRICULUM_EWMA_COEFFICIENT)
    if not math.isfinite(ewma_coefficient) or not 0.0 < ewma_coefficient <= 1.0:
        raise ValueError("--curriculum_ewma must be finite and in (0, 1].")

    unlock_threshold = getattr(args, "curriculum_unlock_threshold", DEFAULT_CURRICULUM_UNLOCK_THRESHOLD)
    if not math.isfinite(unlock_threshold) or not 0.0 <= unlock_threshold <= 1.0:
        raise ValueError("--curriculum_unlock_threshold must be finite and in [0, 1].")

    neighbor_increment = getattr(args, "curriculum_neighbor_increment", DEFAULT_CURRICULUM_NEIGHBOR_INCREMENT)
    if not math.isfinite(neighbor_increment) or neighbor_increment < 0.0:
        raise ValueError("--curriculum_neighbor_increment must be finite and nonnegative.")

    training_seed = getattr(args, "seed", None)
    curriculum_seed = DEFAULT_CURRICULUM_SEED if training_seed is None else training_seed
    if isinstance(curriculum_seed, bool) or not isinstance(curriculum_seed, int) or curriculum_seed < 0:
        if mode != COMMAND_CURRICULUM_MODE_NONE:
            raise ValueError("The command curriculum requires a nonnegative training --seed.")
        curriculum_seed = DEFAULT_CURRICULUM_SEED

    return {
        "mode": mode,
        "velocity_bin_count": velocity_bin_count,
        "ewma_coefficient": ewma_coefficient,
        "unlock_threshold": unlock_threshold,
        "neighbor_increment": neighbor_increment,
        "seed": curriculum_seed,
    }


def command_curriculum_lab_args(args: argparse.Namespace) -> list[str]:
    """Build Hydra overrides for the optional TR-orbit command curriculum."""
    settings = resolve_command_curriculum(args)
    return [
        # Quote string values because OmegaConf resolves an unquoted ``none``
        # override as null instead of the disabled curriculum mode.
        f"env.commands.base_velocity.command_curriculum_mode='{settings['mode']}'",
        f"env.commands.base_velocity.curriculum_velocity_bin_count={settings['velocity_bin_count']}",
        f"env.commands.base_velocity.curriculum_ewma_coefficient={settings['ewma_coefficient']}",
        f"env.commands.base_velocity.curriculum_unlock_threshold={settings['unlock_threshold']}",
        f"env.commands.base_velocity.curriculum_neighbor_increment={settings['neighbor_increment']}",
        f"env.commands.base_velocity.curriculum_seed={settings['seed']}",
    ]


def default_training_run_name(args: argparse.Namespace, *, suffix: str | None = None) -> str:
    """Return a compact run name encoding the primary training treatment."""
    _, history_length = resolve_policy_history(args)
    curriculum = resolve_command_curriculum(args)
    symmetry_enabled = not bool(getattr(args, "disable_symmetry", False))
    mirror_coeff = getattr(args, "mirror_loss_coeff", DEFAULT_MIRROR_LOSS_COEFF) if symmetry_enabled else 0.0
    value_coeff = getattr(args, "tr_value_coeff", DEFAULT_TR_VALUE_COEFF) if symmetry_enabled else 0.0
    seed = getattr(args, "seed", None)
    seed_label = seed if seed is not None else "default"
    curriculum_label = 0 if curriculum["mode"] == COMMAND_CURRICULUM_MODE_NONE else 1
    trs_label = "trs" if symmetry_enabled else "no_trs"
    run_name = (
        f"{args.robot_spec.key}_{trs_label}_m{coeff_label(mirror_coeff)}_v{coeff_label(value_coeff)}"
        f"_h{history_length}_cur{curriculum_label}_seed{seed_label}"
    )
    if suffix:
        run_name += f"_{suffix}"
    if bool(getattr(args, "smoke", False)):
        run_name += "_smoke"
    return run_name


def default_kit_args() -> str:
    """Return viewer Kit args for the current platform."""
    return DEFAULT_WINDOWS_KIT_ARGS if os.name == "nt" else ""


def command_to_string(command: list[str]) -> str:
    """Format a command for display."""
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def should_use_conda(args: argparse.Namespace) -> bool:
    """Return whether the command should be wrapped with ``conda run``."""
    if args.no_conda_run:
        return False
    if args.use_conda_run:
        return True
    return os.environ.get("CONDA_DEFAULT_ENV") != args.conda_env


def isaaclab_command(args: argparse.Namespace, isaaclab_args: list[str]) -> list[str]:
    """Build an Isaac Lab wrapper command for this platform."""
    root = repo_root()
    wrapper = root / ("isaaclab.bat" if os.name == "nt" else "isaaclab.sh")

    if os.name == "nt":
        base_command = ["cmd", "/c", str(wrapper), *isaaclab_args]
    else:
        base_command = [str(wrapper), *isaaclab_args]

    if should_use_conda(args):
        return ["conda", "run", "--no-capture-output", "-n", args.conda_env, *base_command]
    return base_command


def run_isaaclab(args: argparse.Namespace, isaaclab_args: list[str]) -> int:
    """Run or print an Isaac Lab command."""
    command = isaaclab_command(args, isaaclab_args)
    print(log_prefix(args) + command_to_string(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(
        command,
        cwd=repo_root(),
        env=repo_subprocess_environment(),
        check=False,
    ).returncode


def model_iteration(path: Path) -> int:
    """Return the checkpoint iteration encoded in ``model_<iteration>.pt``."""
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_run_dir(args: argparse.Namespace, run_name: str | None = None) -> Path:
    """Resolve a run directory by name or by newest modification time."""
    root = repo_root()
    experiment_dir = root / "logs" / "rsl_rl" / args.robot_spec.experiment_name
    if run_name:
        run_path = Path(run_name)
        if not run_path.is_absolute():
            run_path = experiment_dir / run_name
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")
        return run_path

    runs = [path for path in experiment_dir.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under: {experiment_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    """Resolve the checkpoint requested by play or record commands."""
    if args.checkpoint and args.checkpoint != "latest":
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = repo_root() / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    run_dir = latest_run_dir(args, args.run)
    if args.model:
        model_name = args.model if args.model.endswith(".pt") else f"model_{args.model}.pt"
        checkpoint = run_dir / model_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: (model_iteration(path), path.stat().st_mtime))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in: {run_dir}")
    return checkpoints[-1]


def play_video_snapshot(run_dir: Path) -> dict[Path, tuple[int, int]]:
    """Return modification-time and size signatures for existing play MP4s."""
    video_dir = run_dir / "videos" / "play"
    if not video_dir.exists():
        return {}
    return {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in video_dir.glob("*.mp4")}


def latest_play_video(run_dir: Path, previous_videos: dict[Path, tuple[int, int]] | None = None) -> Path | None:
    """Return the newest play MP4, optionally limited to new or modified files."""
    current_videos = play_video_snapshot(run_dir)
    videos = list(current_videos)
    if previous_videos is not None:
        videos = [path for path, signature in current_videos.items() if previous_videos.get(path) != signature]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def converter_command(args: argparse.Namespace, mp4_path: Path, gif_path: Path) -> list[str]:
    """Build a command that converts an MP4 to GIF inside the target conda env."""
    script = Path(__file__).with_name("_mp4_to_gif.py")
    command = [
        sys.executable,
        str(script),
        "--input",
        str(mp4_path),
        "--output",
        str(gif_path),
        "--fps",
        str(args.gif_fps),
        "--width",
        str(args.gif_width),
    ]
    if should_use_conda(args):
        return ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python", *command[1:]]
    return command


def convert_latest_video(
    args: argparse.Namespace,
    checkpoint: Path,
    previous_videos: dict[Path, tuple[int, int]] | None = None,
) -> int:
    """Convert the newest newly recorded MP4 for the checkpoint run to GIF."""
    mp4_path = latest_play_video(checkpoint.parent, previous_videos)
    if mp4_path is None:
        if args.dry_run:
            print(
                f"{log_prefix(args)}would convert the newest MP4 under {checkpoint.parent / 'videos' / 'play'}",
                flush=True,
            )
            return 0
        qualifier = "new or updated " if previous_videos is not None else ""
        print(
            f"{log_prefix(args)}No {qualifier}MP4 found under {checkpoint.parent / 'videos' / 'play'}",
            flush=True,
        )
        return 1
    gif_path = mp4_path.with_suffix(".gif")
    command = converter_command(args, mp4_path, gif_path)
    print(log_prefix(args) + command_to_string(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(
        command,
        cwd=repo_root(),
        env=repo_subprocess_environment(),
        check=False,
    ).returncode


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add launcher options shared by all subcommands."""
    parser.add_argument("--robot", choices=robot_choices(), default=DEFAULT_ROBOT, help="Robot task to run.")
    parser.add_argument("--conda_env", "--conda-env", default=CONDA_ENV, help="Conda env used for Isaac Lab commands.")
    parser.add_argument(
        "--use_conda_run", "--use-conda-run", action="store_true", help="Force wrapping commands with conda run."
    )
    parser.add_argument(
        "--no_conda_run", "--no-conda-run", action="store_true", help="Run the Isaac Lab wrapper directly."
    )
    parser.add_argument(
        "--expected_branch",
        "--expected-branch",
        default=None,
        help="Abort unless the launcher repository has this Git branch checked out.",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Print commands without running them.")


def add_history_args(parser: argparse.ArgumentParser) -> None:
    """Add native policy-observation history controls."""
    parser.set_defaults(history_enabled=DEFAULT_HISTORY_ENABLED)
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--history",
        dest="history_enabled",
        action="store_true",
        help="Enable native flattened policy-observation history (default).",
    )
    history_group.add_argument(
        "--no_history",
        "--no-history",
        dest="history_enabled",
        action="store_false",
        help="Disable policy-observation history and use one instantaneous frame.",
    )
    parser.add_argument(
        "--history_length",
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH,
        help="Number of native policy-observation frames when history is enabled.",
    )


def add_command_curriculum_args(parser: argparse.ArgumentParser) -> None:
    """Add controls for the optional TR-orbit command competence curriculum."""
    parser.set_defaults(command_curriculum_mode=COMMAND_CURRICULUM_MODE_NONE)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--command_curriculum",
        "--command-curriculum",
        dest="command_curriculum_mode",
        action="store_const",
        const=COMMAND_CURRICULUM_MODE_TR_ORBIT,
        help="Enable the TR-orbit reward-threshold command curriculum.",
    )
    mode_group.add_argument(
        "--no_command_curriculum",
        "--no-command-curriculum",
        dest="command_curriculum_mode",
        action="store_const",
        const=COMMAND_CURRICULUM_MODE_NONE,
        help="Disable the TR-orbit command curriculum (default).",
    )
    mode_group.add_argument(
        "--command_curriculum_mode",
        "--command-curriculum-mode",
        dest="command_curriculum_mode",
        choices=COMMAND_CURRICULUM_MODES,
        help="Select the command curriculum explicitly.",
    )
    parser.add_argument(
        "--curriculum_velocity_bins",
        "--curriculum-velocity-bins",
        "--curriculum_velocity_bin_count",
        "--curriculum-velocity-bin-count",
        dest="curriculum_velocity_bin_count",
        type=int,
        default=DEFAULT_CURRICULUM_VELOCITY_BIN_COUNT,
        help="Number of signed forward-velocity curriculum bins.",
    )
    parser.add_argument(
        "--curriculum_ewma",
        "--curriculum-ewma",
        "--curriculum_ewma_coefficient",
        "--curriculum-ewma-coefficient",
        dest="curriculum_ewma_coefficient",
        type=float,
        default=DEFAULT_CURRICULUM_EWMA_COEFFICIENT,
        help="EWMA coefficient for segment competence.",
    )
    parser.add_argument(
        "--curriculum_unlock_threshold",
        "--curriculum-unlock-threshold",
        type=float,
        default=DEFAULT_CURRICULUM_UNLOCK_THRESHOLD,
    )
    parser.add_argument(
        "--curriculum_neighbor_increment",
        "--curriculum-neighbor-increment",
        type=float,
        default=DEFAULT_CURRICULUM_NEIGHBOR_INCREMENT,
    )


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add train command options."""
    add_common_args(parser)
    add_history_args(parser)
    add_command_curriculum_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument(
        "--iterations", "--max-iterations", "--max_iterations", type=int, default=DEFAULT_TRAINING_ITERATIONS
    )
    parser.add_argument("--run-name", "--run_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mirror",
        "--mirror-loss-coeff",
        "--mirror_loss_coeff",
        "--tr-policy-coef",
        "--tr_policy_coef",
        dest="mirror_loss_coeff",
        type=float,
        default=DEFAULT_MIRROR_LOSS_COEFF,
    )
    parser.add_argument(
        "--tr-value-coef",
        "--tr_value_coef",
        dest="tr_value_coeff",
        type=float,
        default=DEFAULT_TR_VALUE_COEFF,
    )
    parser.add_argument(
        "--tr-warmup-iterations",
        "--tr_warmup_iterations",
        dest="tr_warmup_iterations",
        type=int,
        default=DEFAULT_TR_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--tr-rampup-iterations",
        "--tr_rampup_iterations",
        dest="tr_rampup_iterations",
        type=int,
        default=DEFAULT_TR_RAMPUP_ITERATIONS,
    )
    parser.add_argument(
        "--tr-ramp-shape",
        "--tr_ramp_shape",
        dest="tr_ramp_shape",
        choices=TR_RAMP_SHAPES,
        default=DEFAULT_TR_RAMP_SHAPE,
    )
    parser.add_argument(
        "--tr-min-abs-cmd-vel",
        "--tr_min_abs_cmd_vel",
        dest="tr_min_abs_cmd_vel",
        type=float,
        default=DEFAULT_TR_MIN_ABS_CMD_VEL,
    )
    parser.add_argument("--foot_phase_weight", "--foot-phase-weight", type=float, default=DEFAULT_FOOT_PHASE_WEIGHT)
    parser.add_argument(
        "--foot_phase_reduction",
        "--foot-phase-reduction",
        choices=FOOT_PHASE_REDUCTIONS,
        default=DEFAULT_FOOT_PHASE_REDUCTION,
    )
    parser.add_argument(
        "--joint_target_limit_mode",
        "--joint-target-limit-mode",
        choices=JOINT_TARGET_LIMIT_MODES,
        default=DEFAULT_JOINT_TARGET_LIMIT_MODE,
    )
    parser.add_argument(
        "--joint_target_limit_weight",
        "--joint-target-limit-weight",
        type=float,
        default=None,
        help="Override the selected robot's joint-target-limit reward weight.",
    )
    parser.add_argument(
        "--actor_mean_bound_mode",
        "--actor-mean-bound-mode",
        choices=ACTOR_MEAN_BOUND_MODES,
        default=DEFAULT_ACTOR_MEAN_BOUND_MODE,
    )
    parser.add_argument(
        "--tr_policy_output_space",
        "--tr-policy-output-space",
        choices=TR_POLICY_OUTPUT_SPACES,
        default=DEFAULT_TR_POLICY_OUTPUT_SPACE,
    )
    parser.add_argument(
        "--gait_sampling_profile",
        "--gait-sampling-profile",
        choices=GAIT_SAMPLING_PROFILES,
        default=DEFAULT_GAIT_SAMPLING_PROFILE,
    )
    parser.add_argument(
        "--gait_curriculum_iterations",
        "--gait-curriculum-iterations",
        type=int,
        default=DEFAULT_GAIT_CURRICULUM_ITERATIONS,
    )
    parser.add_argument("--no-trs", "--disable-symmetry", action="store_true", dest="disable_symmetry")
    parser.add_argument("--smoke", action="store_true", help="Run one env for one iteration.")


def train_lab_args(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Build Isaac Lab arguments for a training run."""
    foot_phase_weight = getattr(args, "foot_phase_weight", DEFAULT_FOOT_PHASE_WEIGHT)
    foot_phase_reduction = getattr(args, "foot_phase_reduction", DEFAULT_FOOT_PHASE_REDUCTION)
    joint_target_limit_mode = getattr(args, "joint_target_limit_mode", DEFAULT_JOINT_TARGET_LIMIT_MODE)
    joint_target_limit_weight = getattr(args, "joint_target_limit_weight", None)
    actor_mean_bound_mode = getattr(args, "actor_mean_bound_mode", DEFAULT_ACTOR_MEAN_BOUND_MODE)
    tr_policy_output_space = getattr(args, "tr_policy_output_space", DEFAULT_TR_POLICY_OUTPUT_SPACE)
    gait_sampling_profile = getattr(args, "gait_sampling_profile", DEFAULT_GAIT_SAMPLING_PROFILE)
    gait_curriculum_iterations = getattr(args, "gait_curriculum_iterations", DEFAULT_GAIT_CURRICULUM_ITERATIONS)
    if not math.isfinite(foot_phase_weight) or foot_phase_weight < 0.0:
        raise ValueError("--foot_phase_weight must be finite and nonnegative.")
    if joint_target_limit_weight is not None and (
        not math.isfinite(joint_target_limit_weight) or joint_target_limit_weight < 0.0
    ):
        raise ValueError("--joint_target_limit_weight must be finite and nonnegative.")
    if gait_curriculum_iterations < 0:
        raise ValueError("--gait_curriculum_iterations must be nonnegative.")
    num_envs = 1 if args.smoke else args.num_envs
    iterations = 1 if args.smoke else args.iterations
    run_name = args.run_name if args.run_name else default_training_run_name(args)

    command = [
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.robot_spec.train_task,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(iterations),
        "--run_name",
        run_name,
    ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.disable_symmetry:
        command += [
            "agent.algorithm.symmetry_cfg.use_data_augmentation=False",
            "agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false",
            "agent.algorithm.symmetry_cfg.use_mirror_loss=False",
            "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0",
            "agent.algorithm.symmetry_cfg.value_loss_coeff=0.0",
            f"agent.algorithm.symmetry_cfg.warmup_iterations={args.tr_warmup_iterations}",
            f"agent.algorithm.symmetry_cfg.rampup_iterations={args.tr_rampup_iterations}",
            f"agent.algorithm.symmetry_cfg.ramp_shape={args.tr_ramp_shape}",
            f"agent.algorithm.symmetry_cfg.min_abs_command_velocity={args.tr_min_abs_cmd_vel}",
        ]
    else:
        command += [
            "agent.algorithm.symmetry_cfg.use_data_augmentation=False",
            "agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false",
            f"agent.algorithm.symmetry_cfg.mirror_loss_coeff={args.mirror_loss_coeff}",
            f"agent.algorithm.symmetry_cfg.value_loss_coeff={args.tr_value_coeff}",
            f"agent.algorithm.symmetry_cfg.warmup_iterations={args.tr_warmup_iterations}",
            f"agent.algorithm.symmetry_cfg.rampup_iterations={args.tr_rampup_iterations}",
            f"agent.algorithm.symmetry_cfg.ramp_shape={args.tr_ramp_shape}",
            f"agent.algorithm.symmetry_cfg.min_abs_command_velocity={args.tr_min_abs_cmd_vel}",
        ]
    command += [
        f"env.rewards.foot_phase.weight={foot_phase_weight}",
        f"env.rewards.foot_phase.params.reduction={foot_phase_reduction}",
        f"env.rewards.joint_target_limits.params.mode={joint_target_limit_mode}",
        f"agent.algorithm.symmetry_cfg.actor_mean_bound_mode={actor_mean_bound_mode}",
        f"agent.algorithm.symmetry_cfg.tr_policy_output_space={tr_policy_output_space}",
        f"env.commands.base_velocity.gait_sampling_profile={gait_sampling_profile}",
        f"env.commands.base_velocity.gait_curriculum_iterations={gait_curriculum_iterations}",
    ]
    command += policy_history_lab_args(args)
    command += command_curriculum_lab_args(args)
    if joint_target_limit_weight is not None:
        command.append(f"env.rewards.joint_target_limits.weight={joint_target_limit_weight}")
    history_enabled, history_length = resolve_policy_history(args)
    curriculum = resolve_command_curriculum(args)
    resolved_runtime_argv = [*command, *extra]
    direct_argv = getattr(args, "_direct_launcher_argv", None)
    direct_interface = getattr(args, "_direct_launcher_interface", "scripts/symm_locomotion/symm_cli.py")
    direct_context = {
        "schema_version": 1,
        "interface": direct_interface,
        "argv": direct_argv,
        "resolved_runtime_argv_without_context": resolved_runtime_argv,
        "robot": args.robot_spec.key,
        "run_name": run_name,
        "policy_contract": {
            "observation_contract_version": POLICY_OBSERVATION_CONTRACT_VERSION,
            "instantaneous_frame_dim": POLICY_INSTANTANEOUS_FRAME_DIM,
            "history_enabled": history_enabled,
            "history_length": history_length,
            "history_packing": POLICY_HISTORY_PACKING,
            "history_trs_mode": DEFAULT_HISTORY_TRS_MODE,
            "gait_phase_mapping_version": GAIT_PHASE_MAPPING_VERSION,
            "gait_library_version": GAIT_LIBRARY_VERSION,
        },
        "command_curriculum": curriculum,
    }
    context_payload = json.dumps(direct_context, separators=(",", ":"), sort_keys=True)
    hydra_start = next(
        (index for index, token in enumerate(command) if token.startswith(("agent.", "env."))),
        len(command),
    )
    return [
        *command[:hydra_start],
        "--symm_direct_launch_context",
        context_payload,
        *command[hydra_start:],
        *extra,
    ]


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add checkpoint-resolution options shared by play and record."""
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--run", default=None, help="Run folder name/path. Defaults to newest run.")
    parser.add_argument("--model", default=None, help="Checkpoint iteration or file name inside the run.")


def add_rollout_plot_args(parser: argparse.ArgumentParser) -> None:
    """Add symmetric rollout plotting options."""
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save tracking, gait, joint-torque/power, and per-foot ground-reaction-force plots.",
    )
    parser.add_argument("--plots_dir", "--plots-dir", default=None, help="Override the rollout plot output directory.")
    parser.add_argument(
        "--plot_env_index",
        "--plot-env-index",
        type=int,
        default=0,
        help="Environment index sampled for rollout plots.",
    )
    parser.add_argument(
        "--plot_duration",
        "--plot-duration",
        type=float,
        default=DEFAULT_VIDEO_DURATION_S,
        help="Maximum plotted rollout duration in seconds.",
    )


def rollout_plot_lab_args(args: argparse.Namespace) -> list[str]:
    """Build Isaac Lab arguments for symmetric rollout plotting."""
    if not args.plots:
        return []
    max_steps = round(args.plot_duration / args.robot_spec.step_dt)
    if max_steps < 1:
        raise ValueError("Plot duration must be positive.")
    command = [
        "--symm_rollout_plots",
        "--symm_rollout_plot_env_index",
        str(args.plot_env_index),
        "--symm_rollout_plot_max_steps",
        str(max_steps),
    ]
    if args.plots_dir:
        command += ["--symm_rollout_plots_dir", args.plots_dir]
    return command


def add_gait_sequence_args(parser: argparse.ArgumentParser) -> None:
    """Add deterministic gait-sequence options shared by play and record."""
    gait_group = parser.add_mutually_exclusive_group()
    gait_group.add_argument(
        "--gait-sequence",
        "--gait_sequence",
        dest="gait_sequence",
        action="store_true",
        default=True,
        help="Cycle through all configured gait rows in order (default).",
    )
    gait_group.add_argument(
        "--no-gait-sequence",
        "--no_gait_sequence",
        dest="gait_sequence",
        action="store_false",
        help="Restore random gait sampling during playback.",
    )
    parser.add_argument(
        "--gait-sequence-duration",
        "--gait_sequence_duration",
        dest="gait_sequence_duration_s",
        type=float,
        default=DEFAULT_GAIT_SEQUENCE_DURATION_S,
        help="Duration [s] assigned to each configured gait row.",
    )


def gait_sequence_lab_args(args: argparse.Namespace) -> list[str]:
    """Build Hydra overrides for deterministic gait sequencing."""
    duration_s = args.gait_sequence_duration_s
    if args.gait_sequence and (not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError(f"Gait sequence duration must be finite and positive; received {duration_s!r}.")
    steps_per_gait = round(duration_s / args.robot_spec.step_dt) if args.gait_sequence else 0
    if args.gait_sequence and steps_per_gait < 1:
        raise ValueError(
            "Gait sequence duration must span at least one environment step; "
            f"received {duration_s!r} s with step_dt {args.robot_spec.step_dt!r} s."
        )
    enabled = str(args.gait_sequence).lower()
    episode_length_s = (
        (GAIT_SEQUENCE_COUNT * steps_per_gait + 1) * args.robot_spec.step_dt
        if args.gait_sequence
        else DEFAULT_VIDEO_DURATION_S
    )
    return [
        f"env.commands.base_velocity.gait_sequence_enabled={enabled}",
        f"env.commands.base_velocity.gait_sequence_duration_s={duration_s}",
        f"env.episode_length_s={episode_length_s}",
    ]


def add_play_args(parser: argparse.ArgumentParser) -> None:
    """Add play command options."""
    add_common_args(parser)
    add_history_args(parser)
    add_checkpoint_args(parser)
    add_rollout_plot_args(parser)
    add_gait_sequence_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--rendering-mode", "--rendering_mode", default="balanced")
    parser.add_argument("--kit-args", "--kit_args", default=None)
    parser.add_argument("--disable-fabric", "--disable_fabric", action="store_true")
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument(
        "--print-gait",
        "--print_gait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print current command velocity, measured velocity, gait, duty factor, and period.",
    )
    parser.add_argument(
        "--print-gait-interval",
        "--print_gait_interval",
        type=int,
        default=50,
        help="Number of play steps between gait information terminal prints.",
    )


def play_lab_args(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Build Isaac Lab arguments for policy playback."""
    checkpoint = resolve_checkpoint(args)
    print(f"{log_prefix(args)}checkpoint: {checkpoint}", flush=True)
    kit_args = default_kit_args() if args.kit_args is None else args.kit_args
    command = [
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.robot_spec.play_task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        str(checkpoint),
        "--viz",
        "kit",
        "--rendering_mode",
        args.rendering_mode,
    ]
    if not args.no_real_time:
        command.append("--real-time")
    if args.disable_fabric:
        command.append("--disable_fabric")
    if kit_args:
        command += ["--kit_args", kit_args]
    if args.print_gait:
        command += ["--print_gait_info", "--print_gait_info_interval", str(args.print_gait_interval)]
    return command + gait_sequence_lab_args(args) + rollout_plot_lab_args(args) + policy_history_lab_args(args) + extra


def add_record_args(parser: argparse.ArgumentParser) -> None:
    """Add record command options."""
    add_common_args(parser)
    add_history_args(parser)
    add_checkpoint_args(parser)
    add_rollout_plot_args(parser)
    add_gait_sequence_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument(
        "--video-length",
        "--video_length",
        type=int,
        default=None,
        help="Recording length in environment steps. Defaults to one fixed gait cycle, or 30 seconds when disabled.",
    )
    parser.add_argument("--rendering-mode", "--rendering_mode", default="balanced")
    parser.add_argument("--kit-args", "--kit_args", default=None)
    parser.add_argument("--viewer", action="store_true", help="Show the Kit viewer while recording.")
    parser.add_argument("--gif", action="store_true", help="Convert the newest MP4 to GIF after recording.")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=720)


def record_lab_args(args: argparse.Namespace, extra: list[str]) -> tuple[list[str], Path]:
    """Build Isaac Lab arguments for video recording."""
    gait_args = gait_sequence_lab_args(args)
    checkpoint = resolve_checkpoint(args)
    print(f"{log_prefix(args)}checkpoint: {checkpoint}", flush=True)
    kit_args = default_kit_args() if args.kit_args is None else args.kit_args
    video_length = (
        args.video_length
        if args.video_length is not None
        else (
            GAIT_SEQUENCE_COUNT * round(args.gait_sequence_duration_s / args.robot_spec.step_dt)
            if args.gait_sequence
            else round(DEFAULT_VIDEO_DURATION_S / args.robot_spec.step_dt)
        )
    )
    command = [
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.robot_spec.play_task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        str(checkpoint),
        "--video",
        "--video_length",
        str(video_length),
        "--rendering_mode",
        args.rendering_mode,
    ]
    if args.viewer:
        command += ["--viz", "kit", "--real-time"]
    if kit_args:
        command += ["--kit_args", kit_args]
    return command + gait_args + rollout_plot_lab_args(args) + policy_history_lab_args(args) + extra, checkpoint


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """Add fixed-grid policy evaluation and analysis options."""
    add_common_args(parser)
    add_history_args(parser)
    add_checkpoint_args(parser)
    parser.set_defaults(_protocol_explicit=False)
    parser.add_argument(
        "--protocol",
        choices=("full", "light", "legacy"),
        default="full",
        action=_StoreExplicitAction,
        help=(
            "Evaluation profile (default: immutable full 60-cell grid; light: immutable 8-cell screen; "
            "legacy: deprecated custom v1 grid)."
        ),
    )
    parser.add_argument(
        "--velocities",
        "--vx_values",
        nargs="+",
        type=float,
        default=list(DEFAULT_LEG_USAGE_VELOCITIES_MPS),
        help="Fixed x velocities [m/s] evaluated for every selected gait row.",
    )
    parser.add_argument(
        "--gait_indices",
        "--gait-indices",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Training gait-library row indices to evaluate (default: all ten rows).",
    )
    parser.add_argument(
        "--settle_s",
        "--settle-s",
        type=float,
        default=DEFAULT_LEG_USAGE_SETTLE_S,
        help="Per-cell settling duration [s], excluded from metrics.",
    )
    parser.add_argument(
        "--measure_s",
        "--measure-s",
        type=float,
        default=DEFAULT_LEG_USAGE_MEASURE_S,
        help="Per-cell steady-state measurement duration [s].",
    )
    parser.add_argument(
        "--evaluation_seed",
        "--evaluation-seed",
        type=int,
        default=DEFAULT_LEG_USAGE_EVALUATION_SEED,
        help="Reset seed shared by the fixed grid.",
    )
    parser.add_argument(
        "--render_cell_plots",
        "--render-cell-plots",
        action="store_true",
        help="Also render the plotter's detailed figures inside every cell folder.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an identical checkpoint/protocol and skip already completed cells.",
    )
    parser.add_argument(
        "--analyze_only",
        "--analyze-only",
        action="store_true",
        help="Regenerate metrics and figures from an existing identical study without simulation.",
    )
    parser.add_argument(
        "--evaluation_config",
        "--evaluation-config",
        type=Path,
        help="Optional JSON file containing nested contact, gait, tracking, load, or success overrides.",
    )


def evaluation_lab_args(
    args: argparse.Namespace,
    checkpoint: Path,
    study_path: Path,
    extra: list[str],
) -> list[str]:
    """Build Isaac Lab arguments for the data-only fixed-scenario grid."""
    return [
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.robot_spec.play_task,
        "--num_envs",
        "1",
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(args.evaluation_seed),
        "--headless",
        "--symm_leg_usage_plan",
        str(study_path),
        *policy_history_lab_args(args),
        *extra,
    ]


def validate_evaluation_runtime_overrides(extra: list[str]) -> None:
    """Reject forwarded options that could escape the immutable grid plan."""
    for token in extra:
        option = token.split("=", 1)[0]
        if option in LEG_USAGE_PROTECTED_RUNTIME_OPTIONS:
            raise ValueError(f"Evaluation controls {option} and does not allow overriding it after '--'.")


def _load_evaluation_module():
    """Load the adjacent analysis module without relying on ``sys.path`` setup."""
    module_name = "_symm_policy_evaluation"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).with_name("evaluation.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load leg-usage analysis module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_evaluation(args: argparse.Namespace, extra: list[str]) -> int:
    """Record, analyze, and report one checkpoint's fixed policy-evaluation grid."""
    validate_evaluation_runtime_overrides(extra)
    checkpoint = resolve_checkpoint(args).resolve()
    analysis = _load_evaluation_module()
    evaluation_config = None
    if args.evaluation_config is not None:
        try:
            evaluation_config = json.loads(args.evaluation_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read evaluation config {args.evaluation_config}: {exc}") from exc
        if not isinstance(evaluation_config, dict):
            raise ValueError("The evaluation config JSON root must be an object.")
    if args.protocol == "legacy" and evaluation_config is not None:
        raise ValueError("The legacy v1 compatibility profile does not accept --evaluation_config.")
    output_name = analysis.PROTOCOL_OUTPUT_ROOT_NAMES[args.protocol]
    canonical_study_path = checkpoint.parent / "evaluations" / output_name / "study.json"
    study_path = (
        analysis.existing_study_manifest_path(checkpoint.parent, args.protocol)
        if args.analyze_only or args.resume
        else canonical_study_path
    )
    if args.analyze_only:
        study = analysis.validate_existing_study_for_analysis(
            study_path,
            checkpoint=checkpoint,
            robot=args.robot_spec.key,
            task=args.robot_spec.play_task,
            supplied_runtime_overrides=extra,
        )
        analysis.validate_study_profile(study, args.protocol)
        if evaluation_config is not None and analysis._metrics.evaluation_config(evaluation_config) != study.get(
            "evaluation_config"
        ):
            raise ValueError("Supplied evaluation config does not match the recorded study protocol.")
    elif args.protocol == "legacy" and args.resume and study_path.is_file():
        study = analysis.validate_existing_legacy_study_for_resume(
            study_path,
            checkpoint=checkpoint,
            robot=args.robot_spec.key,
            task=args.robot_spec.play_task,
            step_dt=args.robot_spec.step_dt,
            settle_s=args.settle_s,
            measure_s=args.measure_s,
            evaluation_seed=args.evaluation_seed,
            velocities_mps=args.velocities,
            gait_indices=args.gait_indices,
            render_cell_plots=args.render_cell_plots,
            supplied_runtime_overrides=extra,
        )
    else:
        study = analysis.build_study(
            repo_root=repo_root(),
            checkpoint=checkpoint,
            robot=args.robot_spec.key,
            task=args.robot_spec.play_task,
            step_dt=args.robot_spec.step_dt,
            settle_s=args.settle_s,
            measure_s=args.measure_s,
            evaluation_seed=args.evaluation_seed,
            velocities_mps=args.velocities,
            gait_indices=args.gait_indices,
            render_cell_plots=args.render_cell_plots,
            runtime_overrides=extra,
            protocol=args.protocol,
            evaluation_config=evaluation_config,
        )
        if study_path != canonical_study_path:
            study["output_root"] = str(study_path.parent)
        study_path = analysis.prepare_study(
            study,
            resume=args.resume,
            analyze_only=False,
            dry_run=args.dry_run,
        )
    print(f"{log_prefix(args)}checkpoint: {checkpoint}", flush=True)
    print(f"{log_prefix(args)}output: {study_path.parent}", flush=True)
    print(
        f"{log_prefix(args)}profile: {args.protocol}; method: {study.get('method_version')} "
        f"({len(study['gaits'])} represented gait rows, {len(study['cells'])} cells)",
        flush=True,
    )
    child_code = 0
    if not args.analyze_only:
        child_code = run_isaaclab(args, evaluation_lab_args(args, checkpoint, study_path, extra))
        if args.dry_run:
            return child_code
    elif args.dry_run:
        print(f"{log_prefix(args)}would regenerate analysis from {study_path}", flush=True)
        return 0
    try:
        overall = analysis.analyze_study(study_path)
    except Exception as exc:
        if child_code != 0:
            print(
                f"{log_prefix(args)}WARNING: evaluator exited with code {child_code}, "
                f"and partial analysis failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return child_code
        raise
    coverage = overall["coverage"]
    print(
        f"{log_prefix(args)}analysis: {coverage['valid_cells']}/{coverage['expected_cells']} valid cells; "
        f"report: {study_path.parent / 'metrics' / 'REPORT.md'}",
        flush=True,
    )
    return child_code


# Compatibility aliases retained for callers using the original leg-usage names.
add_analyze_leg_usage_args = add_evaluation_args
analyze_leg_usage_lab_args = evaluation_lab_args
validate_leg_usage_runtime_overrides = validate_evaluation_runtime_overrides
_load_leg_usage_module = _load_evaluation_module
run_analyze_leg_usage = run_evaluation


def add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add ablation command options."""
    add_common_args(parser)
    add_history_args(parser)
    add_command_curriculum_args(parser)
    parser.set_defaults(_mirror_loss_coeff_explicit=False, _tr_value_coeff_explicit=False)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument(
        "--iterations", "--max-iterations", "--max_iterations", type=int, default=DEFAULT_TRAINING_ITERATIONS
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--mirror",
        "--mirror-loss-coeff",
        dest="mirror_loss_coeff",
        type=float,
        default=LEGACY_ABLATION_MIRROR_LOSS_COEFF,
        action=_StoreExplicitAction,
    )
    parser.add_argument(
        "--tr-value-coef",
        "--tr_value_coef",
        dest="tr_value_coeff",
        type=float,
        default=LEGACY_ABLATION_TR_VALUE_COEFF,
        action=_StoreExplicitAction,
    )
    parser.add_argument(
        "--tr-warmup-iterations", "--tr_warmup_iterations", dest="tr_warmup_iterations", type=int, default=1000
    )
    parser.add_argument(
        "--tr-rampup-iterations",
        "--tr_rampup_iterations",
        dest="tr_rampup_iterations",
        type=int,
        default=DEFAULT_TR_RAMPUP_ITERATIONS,
    )
    parser.add_argument(
        "--tr-ramp-shape",
        "--tr_ramp_shape",
        dest="tr_ramp_shape",
        choices=TR_RAMP_SHAPES,
        default=DEFAULT_TR_RAMP_SHAPE,
    )
    parser.add_argument(
        "--tr-min-abs-cmd-vel", "--tr_min_abs_cmd_vel", dest="tr_min_abs_cmd_vel", type=float, default=0.0
    )
    parser.add_argument("--only", choices=("both", "with_trs", "no_trs"), default="both")
    parser.add_argument(
        "--schedule-variants",
        "--schedule_variants",
        nargs="+",
        choices=TR_SCHEDULE_VARIANTS,
        default=None,
        help="Run an explicit matched schedule study; --mirror and --tr-value-coef are required.",
    )


def run_ablation(args: argparse.Namespace, extra: list[str]) -> int:
    """Run the symmetry ablation variants."""
    if args.schedule_variants is None:
        variants = ["with_trs", "no_trs"] if args.only == "both" else [args.only]
        mirror_loss_coeff = args.mirror_loss_coeff
        tr_value_coeff = args.tr_value_coeff
    else:
        if not args._mirror_loss_coeff_explicit or not args._tr_value_coeff_explicit:
            raise ValueError("Schedule ablations require explicit --mirror and --tr-value-coef values.")
        variants = args.schedule_variants
        mirror_loss_coeff = args.mirror_loss_coeff
        tr_value_coeff = args.tr_value_coeff

    for seed in args.seeds:
        for variant in variants:
            train_args = argparse.Namespace(**vars(args))
            train_args.seed = seed
            train_args.smoke = False
            train_args.mirror_loss_coeff = mirror_loss_coeff
            train_args.tr_value_coeff = tr_value_coeff
            if args.schedule_variants is None:
                train_args.disable_symmetry = variant == "no_trs"
                train_args.run_name = default_training_run_name(train_args, suffix="ablation")
            else:
                disable_symmetry, warmup_iterations, rampup_iterations, ramp_shape = TR_SCHEDULE_SETTINGS[variant]
                train_args.disable_symmetry = disable_symmetry
                train_args.tr_warmup_iterations = warmup_iterations
                train_args.tr_rampup_iterations = rampup_iterations
                train_args.tr_ramp_shape = ramp_shape
                train_args.run_name = default_training_run_name(
                    train_args,
                    suffix=f"{variant}_w{warmup_iterations}_r{rampup_iterations}_{ramp_shape}",
                )
            print(f"{log_prefix(args)}ablation variant={variant} seed={seed}", flush=True)
            code = run_isaaclab(args, train_lab_args(train_args, extra))
            if code != 0:
                return code
    return 0


def add_compare_args(parser: argparse.ArgumentParser) -> None:
    """Add compare command options."""
    add_common_args(parser)
    parser.add_argument("--robots", choices=robot_choices(), nargs="+", default=None, help="Robots to compare.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum runs shown per robot.")


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the latest model checkpoint under a run directory."""
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: (model_iteration(path), path.stat().st_mtime))
    return checkpoints[-1] if checkpoints else None


def run_compare(args: argparse.Namespace) -> int:
    """Print a compact run/checkpoint comparison table across robots."""
    robot_keys = args.robots or [args.robot]
    rows = []
    for robot_key in robot_keys:
        spec = get_robot(robot_key)
        experiment_dir = repo_root() / "logs" / "rsl_rl" / spec.experiment_name
        if not experiment_dir.exists():
            rows.append((spec.key, "-", "-", f"missing: {experiment_dir}"))
            continue
        runs = sorted(
            [path for path in experiment_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: args.limit]
        if not runs:
            rows.append((spec.key, "-", "-", f"no runs under: {experiment_dir}"))
            continue
        for run_dir in runs:
            checkpoint = _latest_checkpoint(run_dir)
            checkpoint_name = checkpoint.name if checkpoint is not None else "-"
            rows.append((spec.key, run_dir.name, checkpoint_name, str(run_dir)))

    print(f"{'robot':<8} {'run':<46} {'checkpoint':<18} path", flush=True)
    for robot_key, run_name, checkpoint_name, path in rows:
        print(f"{robot_key:<8} {run_name:<46} {checkpoint_name:<18} {path}", flush=True)
    return 0


def add_tensorboard_args(parser: argparse.ArgumentParser) -> None:
    """Add tensorboard command options."""
    add_common_args(parser)
    parser.add_argument("--robots", choices=robot_choices(), nargs="+", default=None, help="Robots to include.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)


def run_tensorboard(args: argparse.Namespace) -> int:
    """Launch TensorBoard for selected symmetric quadruped logs."""
    robot_keys = args.robots or [args.robot]
    specs = [get_robot(robot_key) for robot_key in robot_keys]
    if os.name == "nt":
        # TensorBoard's named logdir grammar conflicts with Windows drive letters.
        logdir_arg = repo_root() / "logs" / "rsl_rl" / (specs[0].experiment_name if len(specs) == 1 else "")
    else:
        logdirs = []
        for spec in specs:
            logdir = repo_root() / "logs" / "rsl_rl" / spec.experiment_name
            logdirs.append(f"{spec.key}:{logdir}")
        logdir_arg = ",".join(logdirs)
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir_arg),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if should_use_conda(args):
        command = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python", *command[1:]]
    print(log_prefix(args) + command_to_string(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(
        command,
        cwd=repo_root(),
        env=repo_subprocess_environment(),
        check=False,
    ).returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(description="Convenience scripts for symmetric quadruped Isaac Lab tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Start training.")
    add_train_args(train_parser)

    play_parser = subparsers.add_parser("play", help="Play a checkpoint in the viewer.")
    add_play_args(play_parser)

    record_parser = subparsers.add_parser("record", help="Record a checkpoint rollout.")
    add_record_args(record_parser)

    evaluation_parser = subparsers.add_parser(
        "evaluation",
        aliases=["analyze_leg_usage"],
        help="Evaluate tracking, gait fidelity, and leg usage on a fixed gait-by-velocity grid.",
    )
    add_evaluation_args(evaluation_parser)

    ablation_parser = subparsers.add_parser("ablation", help="Run symmetry ablations.")
    add_ablation_args(ablation_parser)

    compare_parser = subparsers.add_parser("compare", help="Compare recent runs across robots.")
    add_compare_args(compare_parser)

    tensorboard_parser = subparsers.add_parser("tensorboard", help="Launch TensorBoard for robot logs.")
    add_tensorboard_args(tensorboard_parser)
    return parser


def split_launcher_and_forwarded_args(argv: list[str] | None) -> tuple[list[str], list[str]]:
    """Split launcher arguments from explicitly delimited downstream arguments.

    Args:
        argv: Raw arguments, or :data:`sys.argv` when omitted.

    Returns:
        The strict launcher prefix and downstream suffix. The first literal
        ``--`` is removed and is not forwarded.
    """
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        delimiter_index = raw_args.index("--")
    except ValueError:
        return raw_args, []
    return raw_args[:delimiter_index], raw_args[delimiter_index + 1 :]


def main(argv: list[str] | None = None) -> int:
    """Run the symmetric quadruped convenience CLI."""
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    launcher_args, extra = split_launcher_and_forwarded_args(raw_args)
    if "--" in raw_args:
        args = parser.parse_args(launcher_args)
    else:
        args, extra = parser.parse_known_args(launcher_args)
        if any(token.startswith("-") for token in extra):
            parser.error(f"unrecognized arguments: {' '.join(extra)}")

    try:
        args.robot_spec = get_robot(args.robot)
        if args.command == "train":
            entrypoint = canonical_python_entrypoint(sys.argv[0])
            entrypoint_args = raw_args if Path(entrypoint).name == "symm_cli.py" else raw_args[1:]
            args._direct_launcher_interface = entrypoint
            args._direct_launcher_argv = [entrypoint, *entrypoint_args]
        validate_expected_branch(args)
        if args.command == "train":
            return run_isaaclab(args, train_lab_args(args, extra))
        if args.command == "play":
            return run_isaaclab(args, play_lab_args(args, extra))
        if args.command == "record":
            lab_args, checkpoint = record_lab_args(args, extra)
            previous_videos = play_video_snapshot(checkpoint.parent) if not args.dry_run else None
            code = run_isaaclab(args, lab_args)
            if code != 0:
                return code
            if previous_videos is not None and latest_play_video(checkpoint.parent, previous_videos) is None:
                print(
                    f"{log_prefix(args)}ERROR: recording finished without a new or updated MP4 under "
                    f"{checkpoint.parent / 'videos' / 'play'}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            if not args.gif:
                return 0
            return convert_latest_video(args, checkpoint, previous_videos)
        if args.command in {"evaluation", "analyze_leg_usage"}:
            if args.command == "analyze_leg_usage":
                if not args._protocol_explicit:
                    args.protocol = "legacy"
                print(
                    "[symm_locomotion] WARNING: 'analyze_leg_usage' is deprecated; use 'evaluation' instead.",
                    file=sys.stderr,
                )
            return run_evaluation(args, extra)
        if args.command == "ablation":
            return run_ablation(args, extra)
        if args.command == "compare":
            return run_compare(args)
        if args.command == "tensorboard":
            return run_tensorboard(args)
    except FileNotFoundError as exc:
        print(f"{log_prefix(args)}ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[symm_locomotion] ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
