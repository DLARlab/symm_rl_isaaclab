# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record and analyze a Go2 Symm Isaac Lab checkpoint rollout."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import time

from isaaclab.app import AppLauncher

from go2_symm_cli import (
    PLAY_TASK,
    command_to_string,
    converter_command,
    default_kit_args,
    resolve_checkpoint,
)


FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_SENSOR_NAMES = ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot")
FOOT_LABELS = ("FL", "FR", "RL", "RR")
MAX_RECORD_DURATION_S = 30.0

GAIT_SEQUENCE = ((0.0, 0.0, 0.5, 0.5),)
VELOCITY_SEQUENCE = (
    {"vx": 0.5, "vy": 0.0, "yaw_rate": 0.0, "heading": 0.0},
    {"vx": -1.0, "vy": 0.0, "yaw_rate": 0.0, "heading": 0.0},
    {"vx": -2.0, "vy": 0.0, "yaw_rate": 0.0, "heading": 0.0},
    {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "heading": 0.0},
    {"vx": 0.5, "vy": 0.0, "yaw_rate": 0.0, "heading": 0.0},
    {"vx": 2.0, "vy": 0.0, "yaw_rate": 0.0, "heading": 0.0},
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--run", default=None, help="Run folder name/path. Defaults to newest run.")
    parser.add_argument("--model", default=None, help="Checkpoint iteration or file name inside the run.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--video-length", "--video_length", type=int, default=1500)
    parser.add_argument("--duration", type=float, default=None, help="Rollout duration in seconds, capped at 30.")
    parser.add_argument("--velocity-transition-interval", type=float, default=5.0)
    parser.add_argument("--gait-transition-interval", type=float, default=5.0)
    parser.add_argument("--plot-start", type=float, default=0.0)
    parser.add_argument("--plot-end", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--viewer", action="store_true", help="Show the Kit viewer while recording.")
    parser.add_argument("--real-time", action="store_true", help="Throttle playback to real time.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 recording and only save data/plots.")
    parser.add_argument("--gif", action="store_true", help="Convert the recorded MP4 to GIF after recording.")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--no-mat", action="store_true", help="Skip MATLAB .mat export.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable-fabric", "--disable_fabric", action="store_true")
    parser.add_argument("--conda-env", default="go2_symm_rl_lab", help=argparse.SUPPRESS)
    parser.add_argument("--use-conda-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-conda-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without launching Isaac Sim.")

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        "--rendering-mode",
        dest="rendering_mode",
        choices=("performance", "balanced", "quality"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--kit-args", dest="kit_args", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.rendering_mode is None:
        args.rendering_mode = "balanced"
    if not args.kit_args:
        args.kit_args = default_kit_args()
    if args.viewer and args.visualizer is None:
        args.visualizer = ["kit"]
    if not args.no_video:
        args.enable_cameras = True
    return args


def _output_dir(args: argparse.Namespace, checkpoint: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    return checkpoint.parent / "recordings" / checkpoint.stem


def _rollout_steps(args: argparse.Namespace, env_cfg) -> int:
    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    requested_duration = args.video_length * step_dt if args.duration is None else args.duration
    duration = min(requested_duration, MAX_RECORD_DURATION_S)
    return max(1, int(round(duration / step_dt)))


def _configure_record_env(env_cfg, args: argparse.Namespace, checkpoint: Path, rollout_steps: int) -> None:
    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    rollout_duration = rollout_steps * step_dt
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.log_dir = str(checkpoint.parent)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), rollout_duration + 1.0)

    if args.disable_fabric and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = False

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.min_xy_command_norm = 0.0
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.resampling_time_gait = 1.0e9
    command_cfg.resample_once_after_reset = True
    command_cfg.resample_gait_once_after_reset = True
    command_cfg.add_noise_period = False
    command_cfg.add_noise_theta = False

    first_velocity = VELOCITY_SEQUENCE[0]
    command_cfg.ranges.lin_vel_x = (first_velocity["vx"], first_velocity["vx"])
    command_cfg.ranges.lin_vel_y = (first_velocity["vy"], first_velocity["vy"])
    command_cfg.ranges.ang_vel_z = (first_velocity["yaw_rate"], first_velocity["yaw_rate"])
    command_cfg.ranges.heading = (first_velocity["heading"], first_velocity["heading"])
    command_cfg.init_foot_thetas = GAIT_SEQUENCE


def _scheduled_velocity(elapsed_time: float, interval: float) -> dict[str, float]:
    index = int(elapsed_time / interval) % len(VELOCITY_SEQUENCE)
    return VELOCITY_SEQUENCE[index]


def _scheduled_gait(elapsed_time: float, interval: float) -> tuple[float, float, float, float]:
    index = int(elapsed_time / interval) % len(GAIT_SEQUENCE)
    return GAIT_SEQUENCE[index]


def _set_command_and_gait(command_term, elapsed_time: float, args: argparse.Namespace) -> None:
    velocity = _scheduled_velocity(elapsed_time, args.velocity_transition_interval)
    command_term.vel_command_b[:, 0] = velocity["vx"]
    command_term.vel_command_b[:, 1] = velocity["vy"]
    command_term.vel_command_b[:, 2] = velocity["yaw_rate"]
    command_term.heading_target[:] = velocity["heading"]
    command_term.is_heading_env[:] = False
    command_term.is_standing_env[:] = False
    command_term.time_left[:] = torch.inf

    gait = torch.tensor(_scheduled_gait(elapsed_time, args.gait_transition_interval), device=command_term.device)
    command_term.foot_thetas[:] = gait
    if command_term.cfg.calculate_from_sampling_curve:
        cmd_x = command_term.command[:, 0]
        command_term.gait_periods[:] = command_term._compute_period_from_forward_velocity(cmd_x).clamp_min(0.1)
        command_term.duty_factors[:] = command_term._compute_duty_factor_from_forward_velocity(cmd_x).clamp(
            min=0.1, max=0.9
        )
    else:
        command_term.gait_periods[:] = command_term.cfg.gait_period
        command_term.duty_factors[:] = command_term.cfg.duty_factor
    command_term.kappa[:] = command_term.cfg.kappa
    command_term.gait_time_left[:] = torch.inf


def _collect_foot_forces(raw_env) -> torch.Tensor:
    foot_forces = []
    for sensor_name in FOOT_SENSOR_NAMES:
        sensor = raw_env.scene.sensors[sensor_name]
        force_norm = torch.linalg.norm(sensor.data.net_forces_w_history.torch, dim=-1).max(dim=1)[0]
        foot_forces.append(force_norm[:, 0])
    return torch.stack(foot_forces, dim=-1)


def _append_sample(data: dict[str, list], raw_env, command_term, foot_body_ids: list[int], elapsed_time: float) -> None:
    robot = raw_env.scene["robot"]
    foot_speeds = torch.linalg.norm(robot.data.body_lin_vel_w.torch[:, foot_body_ids, :], dim=-1)

    data["time"].append(elapsed_time)
    data["desired_lin_vel_b"].append(command_term.command.detach().cpu().numpy())
    data["base_lin_vel_b"].append(robot.data.root_lin_vel_b.torch.detach().cpu().numpy())
    data["base_ang_vel_b"].append(robot.data.root_ang_vel_b.torch.detach().cpu().numpy())
    data["base_pos_w"].append(robot.data.root_pos_w.torch.detach().cpu().numpy())
    data["base_quat_w"].append(robot.data.root_quat_w.torch.detach().cpu().numpy())
    data["foot_periodic_force_weights"].append(command_term.periodic_force_weights().detach().cpu().numpy())
    data["foot_periodic_speed_weights"].append(command_term.periodic_speed_weights().detach().cpu().numpy())
    data["foot_contact_forces"].append(_collect_foot_forces(raw_env).detach().cpu().numpy())
    data["foot_speeds"].append(foot_speeds.detach().cpu().numpy())
    data["foot_phases"].append(command_term.foot_phases().detach().cpu().numpy())
    data["foot_thetas"].append(command_term.foot_thetas.detach().cpu().numpy())
    data["gait_periods"].append(command_term.gait_periods.detach().cpu().numpy())
    data["duty_factors"].append(command_term.duty_factors.detach().cpu().numpy())


def _finalize_data(data: dict[str, list]) -> dict[str, np.ndarray]:
    arrays = {"time": np.asarray(data.pop("time"), dtype=np.float32)}
    arrays.update({key: np.stack(value, axis=0) for key, value in data.items()})
    return arrays


def _env0(array: np.ndarray) -> np.ndarray:
    return array[:, 0] if array.ndim >= 2 else array


def _time_mask(times: np.ndarray, plot_start: float, plot_end: float | None) -> np.ndarray:
    mask = times >= plot_start
    if plot_end is not None:
        mask &= times <= plot_end
    return mask


def _save_data_files(arrays: dict[str, np.ndarray], output_dir: Path, save_mat: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "sim_data.npz", **arrays)
    if not save_mat:
        return
    try:
        from scipy.io import savemat
    except ImportError:
        print("[go2_symm] scipy is not available; skipped sim_data.mat", flush=True)
        return
    savemat(output_dir / "sim_data.mat", arrays)


def _save_plots(arrays: dict[str, np.ndarray], output_dir: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    times = arrays["time"]
    mask = _time_mask(times, args.plot_start, args.plot_end)
    t = times[mask]

    desired_vel = _env0(arrays["desired_lin_vel_b"])[mask]
    base_vel = _env0(arrays["base_lin_vel_b"])[mask]
    base_pos = _env0(arrays["base_pos_w"])[mask]
    force_weights = _env0(arrays["foot_periodic_force_weights"])[mask]
    speed_weights = _env0(arrays["foot_periodic_speed_weights"])[mask]
    foot_forces = _env0(arrays["foot_contact_forces"])[mask]
    foot_speeds = _env0(arrays["foot_speeds"])[mask]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, desired_vel[:, 0], "--", label="cmd vx")
    axes[0].plot(t, base_vel[:, 0], label="actual vx")
    axes[0].set_ylabel("x vel [m/s]")
    axes[0].legend(loc="best")
    axes[1].plot(t, desired_vel[:, 1], "--", label="cmd vy")
    axes[1].plot(t, base_vel[:, 1], label="actual vy")
    axes[1].set_ylabel("y vel [m/s]")
    axes[1].legend(loc="best")
    axes[2].plot(t, base_pos[:, 0], label="x")
    axes[2].plot(t, base_pos[:, 1], label="y")
    axes[2].plot(t, base_pos[:, 2], label="z")
    axes[2].set_ylabel("position [m]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_linear_velocities_and_position.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for leg_id, label in enumerate(FOOT_LABELS):
        ax = axes[leg_id]
        ax.plot(t, foot_forces[:, leg_id], label=f"{label} force")
        twin = ax.twinx()
        twin.plot(t, force_weights[:, leg_id], color="tab:red", alpha=0.8, label=f"{label} E_C_frc")
        ax.set_ylabel("force [N]")
        twin.set_ylabel("weight")
        ax.legend(loc="upper left")
        twin.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_E_C_frc_and_contact_forces.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for leg_id, label in enumerate(FOOT_LABELS):
        ax = axes[leg_id]
        ax.plot(t, foot_speeds[:, leg_id], label=f"{label} speed")
        twin = ax.twinx()
        twin.plot(t, speed_weights[:, leg_id], color="tab:red", alpha=0.8, label=f"{label} E_C_spd")
        ax.set_ylabel("speed [m/s]")
        twin.set_ylabel("weight")
        ax.legend(loc="upper left")
        twin.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_E_C_spd_and_foot_velocities.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, np.sum(foot_forces, axis=1), label="sum contact force")
    twin = ax.twinx()
    twin.plot(t, np.sum(force_weights, axis=1), color="tab:red", label="sum E_C_frc")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    twin.set_ylabel("weight")
    ax.legend(loc="upper left")
    twin.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_agg_E_C_frc_vs_contact.png", dpi=160)
    plt.close(fig)


def _latest_mp4(video_dir: Path) -> Path | None:
    if not video_dir.exists():
        return None
    videos = list(video_dir.glob("*.mp4"))
    return max(videos, key=lambda path: path.stat().st_mtime) if videos else None


def _convert_gif(args: argparse.Namespace, video_dir: Path) -> int:
    mp4_path = _latest_mp4(video_dir)
    if mp4_path is None:
        print(f"[go2_symm] No MP4 found under {video_dir}", flush=True)
        return 1
    gif_path = mp4_path.with_suffix(".gif")
    command = converter_command(args, mp4_path, gif_path)
    print("[go2_symm] " + command_to_string(command), flush=True)
    return subprocess.run(command, cwd=Path.cwd(), check=False).returncode


def _reset_policy(policy, runner, dones: torch.Tensor) -> None:
    if hasattr(policy, "reset"):
        policy.reset(dones)
        return
    alg = getattr(runner, "alg", None)
    policy_nn = getattr(alg, "policy", None) or getattr(alg, "actor_critic", None)
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(dones)


def _run(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> int:
    global np, torch

    import gymnasium as gym
    import numpy as np
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = load_cfg_from_registry(PLAY_TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if args.device is not None:
        agent_cfg.device = args.device
    if args.seed is not None:
        agent_cfg.seed = args.seed

    env_cfg = parse_env_cfg(PLAY_TASK, device=args.device, num_envs=args.num_envs)
    rollout_steps = _rollout_steps(args, env_cfg)
    _configure_record_env(env_cfg, args, checkpoint, rollout_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"

    record_video = not args.no_video
    render_mode = "rgb_array" if record_video else None
    env = gym.make(PLAY_TASK, cfg=env_cfg, render_mode=render_mode)
    if record_video:
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": rollout_steps,
            "disable_logger": True,
        }
        print(f"[go2_symm] recording video to: {video_dir}", flush=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    env.reset()

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    print(f"[go2_symm] checkpoint: {checkpoint}", flush=True)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw_env.device)

    robot = raw_env.scene["robot"]
    foot_body_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
    command_term = raw_env.command_manager.get_term("base_velocity")
    data = {
        "time": [],
        "desired_lin_vel_b": [],
        "base_lin_vel_b": [],
        "base_ang_vel_b": [],
        "base_pos_w": [],
        "base_quat_w": [],
        "foot_periodic_force_weights": [],
        "foot_periodic_speed_weights": [],
        "foot_contact_forces": [],
        "foot_speeds": [],
        "foot_phases": [],
        "foot_thetas": [],
        "gait_periods": [],
        "duty_factors": [],
    }

    step_dt = raw_env.step_dt
    print(f"[go2_symm] saving analysis to: {output_dir}", flush=True)
    print(f"[go2_symm] rollout: {rollout_steps} steps, {rollout_steps * step_dt:.2f} s", flush=True)
    try:
        for step in range(rollout_steps):
            step_start_time = time.time()
            elapsed_time = step * step_dt
            _set_command_and_gait(command_term, elapsed_time, args)
            obs = env.get_observations()
            _append_sample(data, raw_env, command_term, foot_body_ids, elapsed_time)
            with torch.inference_mode():
                actions = policy(obs)
                _, _, dones, _ = env.step(actions)
                _reset_policy(policy, runner, dones)
            if args.real_time:
                sleep_time = step_dt - (time.time() - step_start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        env.close()

    arrays = _finalize_data(data)
    _save_data_files(arrays, output_dir, save_mat=not args.no_mat)
    _save_plots(arrays, output_dir, args)
    print("[go2_symm] wrote sim_data.npz, sim_data.mat, and analysis plots", flush=True)

    if args.gif and record_video:
        return _convert_gif(args, video_dir)
    return 0


def main() -> int:
    args = _parse_args()
    checkpoint = resolve_checkpoint(args)
    output_dir = _output_dir(args, checkpoint)
    if args.dry_run:
        print(f"[go2_symm] checkpoint: {checkpoint}", flush=True)
        print(f"[go2_symm] output_dir: {output_dir}", flush=True)
        print(f"[go2_symm] video_length: {args.video_length}", flush=True)
        return 0

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return _run(args, checkpoint, output_dir)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
