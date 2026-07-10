# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for the Go2 symmetric RL Isaac Lab task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

TRAIN_TASK = "Isaac-Velocity-Flat-Unitree-Go2-Symm-v0"
PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0"
EXPERIMENT_NAME = "unitree_go2_symm_flat"
CONDA_ENV = "go2_symm_rl_lab"
DEFAULT_NUM_ENVS = 256
DEFAULT_MIRROR_LOSS_COEFF = 0.1
DEFAULT_TR_VALUE_COEFF = 0.05
DEFAULT_TR_WARMUP_ITERATIONS = 500
DEFAULT_TR_MIN_ABS_CMD_VEL = 0.2
DEFAULT_WINDOWS_KIT_ARGS = "--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"


def repo_root() -> Path:
    """Return the Isaac Lab repository root."""
    return Path(__file__).resolve().parents[2]


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
    print("[go2_symm] " + command_to_string(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=repo_root(), check=False).returncode


def model_iteration(path: Path) -> int:
    """Return the checkpoint iteration encoded in ``model_<iteration>.pt``."""
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_run_dir(run_name: str | None = None) -> Path:
    """Resolve a run directory by name or by newest modification time."""
    root = repo_root()
    experiment_dir = root / "logs" / "rsl_rl" / EXPERIMENT_NAME
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

    run_dir = latest_run_dir(args.run)
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


def latest_play_video(run_dir: Path) -> Path | None:
    """Return the newest play MP4 for a run, if one exists."""
    video_dir = run_dir / "videos" / "play"
    if not video_dir.exists():
        return None
    videos = list(video_dir.glob("*.mp4"))
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


def convert_latest_video(args: argparse.Namespace, checkpoint: Path) -> int:
    """Convert the newest recorded MP4 for the checkpoint run to GIF."""
    mp4_path = latest_play_video(checkpoint.parent)
    if mp4_path is None:
        if args.dry_run:
            print(f"[go2_symm] would convert the newest MP4 under {checkpoint.parent / 'videos' / 'play'}", flush=True)
            return 0
        print(f"[go2_symm] No MP4 found under {checkpoint.parent / 'videos' / 'play'}", flush=True)
        return 1
    gif_path = mp4_path.with_suffix(".gif")
    command = converter_command(args, mp4_path, gif_path)
    print("[go2_symm] " + command_to_string(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=repo_root(), check=False).returncode


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add launcher options shared by all subcommands."""
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
        run_name = "no_trs_smoke" if args.smoke else "no_trs"
    else:
        run_name = ("trs_smoke_" if args.smoke else "with_trs_") + f"mirror{coeff_label(args.mirror_loss_coeff)}"

    command = [
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        TRAIN_TASK,
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


def add_play_args(parser: argparse.ArgumentParser) -> None:
    """Add play command options."""
    add_common_args(parser)
    add_checkpoint_args(parser)
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
        help="Print current Go2 command velocity, measured velocity, gait, duty factor, and period.",
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
    print(f"[go2_symm] checkpoint: {checkpoint}", flush=True)
    kit_args = default_kit_args() if args.kit_args is None else args.kit_args
    command = [
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        PLAY_TASK,
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
    return command + extra


def add_record_args(parser: argparse.ArgumentParser) -> None:
    """Add record command options."""
    add_common_args(parser)
    add_checkpoint_args(parser)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--video-length", "--video_length", type=int, default=400)
    parser.add_argument("--rendering-mode", "--rendering_mode", default="balanced")
    parser.add_argument("--kit-args", "--kit_args", default=None)
    parser.add_argument("--viewer", action="store_true", help="Show the Kit viewer while recording.")
    parser.add_argument("--gif", action="store_true", help="Convert the newest MP4 to GIF after recording.")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=720)


def record_lab_args(args: argparse.Namespace, extra: list[str]) -> tuple[list[str], Path]:
    """Build Isaac Lab arguments for video recording."""
    checkpoint = resolve_checkpoint(args)
    print(f"[go2_symm] checkpoint: {checkpoint}", flush=True)
    kit_args = default_kit_args() if args.kit_args is None else args.kit_args
    command = [
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        PLAY_TASK,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        str(checkpoint),
        "--video",
        "--video_length",
        str(args.video_length),
        "--rendering_mode",
        args.rendering_mode,
    ]
    if args.viewer:
        command += ["--viz", "kit", "--real-time"]
    if kit_args:
        command += ["--kit_args", kit_args]
    return command + extra, checkpoint


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
        "--tr-min-abs-cmd-vel", "--tr_min_abs_cmd_vel", dest="tr_min_abs_cmd_vel", type=float, default=0.2
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
            print(f"[go2_symm] ablation variant={variant} seed={seed}", flush=True)
            code = run_isaaclab(args, train_lab_args(train_args, extra))
            if code != 0:
                return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(description="Convenience scripts for Go2 Symm Isaac Lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Start training.")
    add_train_args(train_parser)

    play_parser = subparsers.add_parser("play", help="Play a checkpoint in the viewer.")
    add_play_args(play_parser)

    record_parser = subparsers.add_parser("record", help="Record a checkpoint rollout.")
    add_record_args(record_parser)

    ablation_parser = subparsers.add_parser("ablation", help="Run symmetry ablations.")
    add_ablation_args(ablation_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Go2 Symm convenience CLI."""
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    try:
        if args.command == "train":
            return run_isaaclab(args, train_lab_args(args, extra))
        if args.command == "play":
            return run_isaaclab(args, play_lab_args(args, extra))
        if args.command == "record":
            lab_args, checkpoint = record_lab_args(args, extra)
            code = run_isaaclab(args, lab_args)
            if code != 0 or not args.gif:
                return code
            return convert_latest_video(args, checkpoint)
        if args.command == "ablation":
            return run_ablation(args, extra)
    except FileNotFoundError as exc:
        print(f"[go2_symm] ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
