# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for symmetric quadruped Isaac Lab tasks."""

from __future__ import annotations

import argparse
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
LEGACY_ABLATION_MIRROR_LOSS_COEFF = 0.2
LEGACY_ABLATION_TR_VALUE_COEFF = 0.05
DEFAULT_VIDEO_DURATION_S = 30.0
DEFAULT_GAIT_SEQUENCE_DURATION_S = 5.0
GAIT_SEQUENCE_COUNT = 6
DEFAULT_WINDOWS_KIT_ARGS = "--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"

TR_RAMP_SHAPES = ("linear", "half_cosine")
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
    script = Path(__file__).with_name("mp4_to_gif.py")
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
    parser.add_argument("--conda-env", default=CONDA_ENV, help="Conda env used for Isaac Lab commands.")
    parser.add_argument("--use-conda-run", action="store_true", help="Force wrapping commands with conda run.")
    parser.add_argument("--no-conda-run", action="store_true", help="Run the Isaac Lab wrapper directly.")
    parser.add_argument(
        "--expected-branch",
        "--expected_branch",
        default=None,
        help="Abort unless the launcher repository has this Git branch checked out.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add train command options."""
    add_common_args(parser)
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
    parser.add_argument("--no-trs", "--disable-symmetry", action="store_true", dest="disable_symmetry")
    parser.add_argument("--smoke", action="store_true", help="Run one env for one iteration.")


def train_lab_args(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Build Isaac Lab arguments for a training run."""
    num_envs = 1 if args.smoke else args.num_envs
    iterations = 1 if args.smoke else args.iterations
    if args.run_name:
        run_name = args.run_name
    elif args.disable_symmetry:
        run_name = "no_trs_smoke" if args.smoke else "no_trs"
    elif args.tr_rampup_iterations > 0:
        seed_label = args.seed if args.seed is not None else "default"
        run_name = (
            f"{args.robot_spec.key}_trs_{args.tr_ramp_shape}_m{coeff_label(args.mirror_loss_coeff)}"
            f"_v{coeff_label(args.tr_value_coeff)}_w{args.tr_warmup_iterations}"
            f"_r{args.tr_rampup_iterations}_seed{seed_label}"
        )
    else:
        run_name = ("trs_smoke_" if args.smoke else "with_trs_") + f"mirror{coeff_label(args.mirror_loss_coeff)}"

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
            f"agent.algorithm.symmetry_cfg.mirror_loss_coeff={args.mirror_loss_coeff}",
            f"agent.algorithm.symmetry_cfg.value_loss_coeff={args.tr_value_coeff}",
            f"agent.algorithm.symmetry_cfg.warmup_iterations={args.tr_warmup_iterations}",
            f"agent.algorithm.symmetry_cfg.rampup_iterations={args.tr_rampup_iterations}",
            f"agent.algorithm.symmetry_cfg.ramp_shape={args.tr_ramp_shape}",
            f"agent.algorithm.symmetry_cfg.min_abs_command_velocity={args.tr_min_abs_cmd_vel}",
        ]
    return command + extra


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
    return command + gait_sequence_lab_args(args) + rollout_plot_lab_args(args) + extra


def add_record_args(parser: argparse.ArgumentParser) -> None:
    """Add record command options."""
    add_common_args(parser)
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
    return command + gait_args + rollout_plot_lab_args(args) + extra, checkpoint


def add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add ablation command options."""
    add_common_args(parser)
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
                train_args.run_name = (
                    f"ablation_with_trs_seed{seed}_mirror{coeff_label(mirror_loss_coeff)}"
                    if variant == "with_trs"
                    else f"ablation_no_trs_seed{seed}"
                )
            else:
                disable_symmetry, warmup_iterations, rampup_iterations, ramp_shape = TR_SCHEDULE_SETTINGS[variant]
                train_args.disable_symmetry = disable_symmetry
                train_args.tr_warmup_iterations = warmup_iterations
                train_args.tr_rampup_iterations = rampup_iterations
                train_args.tr_ramp_shape = ramp_shape
                effective_mirror = 0.0 if disable_symmetry else mirror_loss_coeff
                effective_value = 0.0 if disable_symmetry else tr_value_coeff
                train_args.run_name = (
                    f"{args.robot_spec.key}_{variant}_m{coeff_label(effective_mirror)}"
                    f"_v{coeff_label(effective_value)}_w{warmup_iterations}_r{rampup_iterations}"
                    f"_{ramp_shape}_seed{seed}"
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
