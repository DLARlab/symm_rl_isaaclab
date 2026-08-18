# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for symmetric quadruped Isaac Lab tasks."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CONDA_ENV = "symm_rl_isaaclab"
DEFAULT_NUM_ENVS = 256
DEFAULT_MIRROR_LOSS_COEFF = 0.1
DEFAULT_TR_VALUE_COEFF = 0.05
DEFAULT_TR_WARMUP_ITERATIONS = 500
DEFAULT_TR_MIN_ABS_CMD_VEL = 0.0
DEFAULT_VIDEO_DURATION_S = 30.0
DEFAULT_WINDOWS_KIT_ARGS = "--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"


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


def repo_root() -> Path:
    """Return the Isaac Lab repository root."""
    return Path(__file__).resolve().parents[2]


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
    return subprocess.run(command, cwd=repo_root(), check=False).returncode


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


def checkpoint_output_name(checkpoint: Path) -> str:
    """Return the output subdirectory name for a loaded checkpoint."""
    return checkpoint.stem


def checkpoint_output_dir(checkpoint: Path) -> Path:
    """Return the shared output directory for a loaded checkpoint."""
    return checkpoint.parent / "eval" / checkpoint_output_name(checkpoint)


def play_video_snapshot(checkpoint: Path) -> dict[Path, tuple[int, int]]:
    """Return modification-time and size signatures for existing checkpoint MP4s."""
    output_dir = checkpoint_output_dir(checkpoint)
    if not output_dir.exists():
        return {}
    return {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in output_dir.glob("*.mp4")}


def latest_play_video(checkpoint: Path, previous_videos: dict[Path, tuple[int, int]] | None = None) -> Path | None:
    """Return the newest checkpoint MP4, optionally limited to new or modified files."""
    current_videos = play_video_snapshot(checkpoint)
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
    output_dir = checkpoint_output_dir(checkpoint)
    mp4_path = latest_play_video(checkpoint, previous_videos)
    if mp4_path is None:
        if args.dry_run:
            print(
                f"{log_prefix(args)}would convert the newest MP4 under {output_dir}",
                flush=True,
            )
            return 0
        qualifier = "new or updated " if previous_videos is not None else ""
        print(
            f"{log_prefix(args)}No {qualifier}MP4 found under {output_dir}",
            flush=True,
        )
        return 1
    gif_path = mp4_path.with_suffix(".gif")
    command = converter_command(args, mp4_path, gif_path)
    print(log_prefix(args) + command_to_string(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=repo_root(), check=False).returncode


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add launcher options shared by all subcommands."""
    parser.add_argument("--robot", choices=robot_choices(), default=DEFAULT_ROBOT, help="Robot task to run.")
    parser.add_argument("--conda-env", default=CONDA_ENV, help="Conda env used for Isaac Lab commands.")
    parser.add_argument("--use-conda-run", action="store_true", help="Force wrapping commands with conda run.")
    parser.add_argument("--no-conda-run", action="store_true", help="Run the Isaac Lab wrapper directly.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add train command options."""
    add_common_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--iterations", "--max-iterations", "--max_iterations", type=int, default=30000)
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
        run_name = f"{args.robot_spec.key}_" + ("no_trs_smoke" if args.smoke else "no_trs")
    else:
        run_name = (
            f"{args.robot_spec.key}_"
            + ("trs_smoke_" if args.smoke else "with_trs_")
            + f"mirror{coeff_label(args.mirror_loss_coeff)}"
        )

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
        ]
    else:
        command += [
            f"agent.algorithm.symmetry_cfg.mirror_loss_coeff={args.mirror_loss_coeff}",
            f"agent.algorithm.symmetry_cfg.value_loss_coeff={args.tr_value_coeff}",
            f"agent.algorithm.symmetry_cfg.warmup_iterations={args.tr_warmup_iterations}",
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


def add_play_args(parser: argparse.ArgumentParser) -> None:
    """Add play command options."""
    add_common_args(parser)
    add_checkpoint_args(parser)
    add_rollout_plot_args(parser)
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
    parser.add_argument(
        "--tracking-test",
        "--tracking_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override commands with a six-direction tracking-error sweep.",
    )
    parser.add_argument("--tracking-speed", "--tracking_speed", type=float, default=0.5)
    parser.add_argument("--tracking-yaw-rate", "--tracking_yaw_rate", type=float, default=0.5)


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
    if args.tracking_test:
        command += [
            "--tracking_error_direction_test",
            "--tracking_error_direction_speed",
            str(args.tracking_speed),
            "--tracking_error_direction_yaw_rate",
            str(args.tracking_yaw_rate),
        ]
    return command + rollout_plot_lab_args(args) + extra


def add_record_args(parser: argparse.ArgumentParser) -> None:
    """Add record command options."""
    add_common_args(parser)
    add_checkpoint_args(parser)
    add_rollout_plot_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument(
        "--video-length",
        "--video_length",
        type=int,
        default=None,
        help="Recording length in environment steps. Defaults to 30 seconds for the selected robot.",
    )
    parser.add_argument("--rendering-mode", "--rendering_mode", default="balanced")
    parser.add_argument("--kit-args", "--kit_args", default=None)
    parser.add_argument("--viewer", action="store_true", help="Show the Kit viewer while recording.")
    parser.add_argument(
        "--tracking-test",
        "--tracking_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override commands with a six-direction tracking-error sweep while recording.",
    )
    parser.add_argument("--tracking-speed", "--tracking_speed", type=float, default=0.5)
    parser.add_argument("--tracking-yaw-rate", "--tracking_yaw_rate", type=float, default=0.5)
    parser.add_argument("--gif", action="store_true", help="Convert the newest MP4 to GIF after recording.")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=720)


def record_lab_args(args: argparse.Namespace, extra: list[str]) -> tuple[list[str], Path]:
    """Build Isaac Lab arguments for video recording."""
    checkpoint = resolve_checkpoint(args)
    print(f"{log_prefix(args)}checkpoint: {checkpoint}", flush=True)
    kit_args = default_kit_args() if args.kit_args is None else args.kit_args
    video_length = (
        args.video_length
        if args.video_length is not None
        else round(DEFAULT_VIDEO_DURATION_S / args.robot_spec.step_dt)
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
    if args.tracking_test:
        command += [
            "--tracking_error_direction_test",
            "--tracking_error_direction_speed",
            str(args.tracking_speed),
            "--tracking_error_direction_yaw_rate",
            str(args.tracking_yaw_rate),
        ]
    if kit_args:
        command += ["--kit_args", kit_args]
    return command + rollout_plot_lab_args(args) + extra, checkpoint


def add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add ablation command options."""
    add_common_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--iterations", "--max-iterations", "--max_iterations", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--mirror", "--mirror-loss-coeff", dest="mirror_loss_coeff", type=float, default=0.2)
    parser.add_argument("--tr-value-coef", "--tr_value_coef", dest="tr_value_coeff", type=float, default=0.05)
    parser.add_argument(
        "--tr-warmup-iterations", "--tr_warmup_iterations", dest="tr_warmup_iterations", type=int, default=1000
    )
    parser.add_argument(
        "--tr-min-abs-cmd-vel", "--tr_min_abs_cmd_vel", dest="tr_min_abs_cmd_vel", type=float, default=0.0
    )
    parser.add_argument("--only", choices=("both", "with_trs", "no_trs"), default="both")


def run_ablation(args: argparse.Namespace, extra: list[str]) -> int:
    """Run the symmetry ablation variants."""
    variants = ["with_trs", "no_trs"] if args.only == "both" else [args.only]
    for seed in args.seeds:
        for variant in variants:
            train_args = argparse.Namespace(**vars(args))
            train_args.seed = seed
            train_args.smoke = False
            train_args.disable_symmetry = variant == "no_trs"
            train_args.run_name = (
                f"ablation_with_trs_seed{seed}_mirror{coeff_label(args.mirror_loss_coeff)}"
                if variant == "with_trs"
                else f"ablation_no_trs_seed{seed}"
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
    return subprocess.run(command, cwd=repo_root(), check=False).returncode


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


def main(argv: list[str] | None = None) -> int:
    """Run the symmetric quadruped convenience CLI."""
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    try:
        args.robot_spec = get_robot(args.robot)
        if args.command == "train":
            return run_isaaclab(args, train_lab_args(args, extra))
        if args.command == "play":
            return run_isaaclab(args, play_lab_args(args, extra))
        if args.command == "record":
            lab_args, checkpoint = record_lab_args(args, extra)
            previous_videos = play_video_snapshot(checkpoint) if not args.dry_run else None
            code = run_isaaclab(args, lab_args)
            if code != 0:
                return code
            if previous_videos is not None and latest_play_video(checkpoint, previous_videos) is None:
                print(
                    f"{log_prefix(args)}ERROR: recording finished without a new or updated MP4 under "
                    f"{checkpoint_output_dir(checkpoint)}",
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
