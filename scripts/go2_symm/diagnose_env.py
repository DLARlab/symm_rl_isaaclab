# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print short Go2 Symm environment parity diagnostics."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from go2_symm_cli import TRAIN_TASK


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TRAIN_TASK)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-mode", choices=("zero", "random"), default="zero")
    parser.add_argument("--seed", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _format_names(names: list[str]) -> str:
    return "[" + ", ".join(names) + "]"


def _print(message: str) -> None:
    print(message, flush=True)


def _term_counts(env, steps: int, action_mode: str) -> dict[str, int]:
    term_manager = env.termination_manager
    counts = {name: 0 for name in term_manager.active_terms}
    action_shape = env.action_space.shape
    device = env.device

    for _ in range(steps):
        if action_mode == "random":
            actions = 2.0 * torch.rand(action_shape, device=device) - 1.0
        else:
            actions = torch.zeros(action_shape, device=device)
        env.step(actions)
        for name in term_manager.active_terms:
            counts[name] += int(term_manager.get_term(name).sum().item())
    return counts


def main() -> None:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    _print("Creating Go2 Symm environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except BaseException as exc:
        _print(f"Environment creation failed: {type(exc).__name__}: {exc}")
        raise

    try:
        _print("Resetting Go2 Symm environment...")
        env.reset()
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        calf_ids = robot.find_bodies(["FL_calf", "FR_calf", "RL_calf", "RR_calf"], preserve_order=True)[0]
        foot_ids = robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)[0]

        _print("Go2 Symm environment diagnostic")
        _print(f"task: {args_cli.task}")
        _print(f"num_envs: {env.num_envs}")
        _print(f"terrain_type: {env_cfg.scene.terrain.terrain_type}")
        _print(f"merge_fixed_joints: {env_cfg.scene.robot.spawn.merge_fixed_joints}")
        _print(f"usd_dir: {env_cfg.scene.robot.spawn.usd_dir}")
        _print(f"usd_file_name: {env_cfg.scene.robot.spawn.usd_file_name}")
        _print(f"angular_damping: {env_cfg.scene.robot.spawn.rigid_props.angular_damping}")
        _print(f"joint_count: {len(robot.joint_names)}")
        _print(f"joint_names: {_format_names(robot.joint_names)}")
        _print(f"body_count: {len(robot.body_names)}")
        _print(f"body_names: {_format_names(robot.body_names)}")
        for sensor_name in sorted(env.scene.sensors.keys()):
            sensor = env.scene.sensors[sensor_name]
            body_names = getattr(sensor, "body_names", [])
            _print(f"sensor/{sensor_name}/body_names: {_format_names(body_names)}")
        _print(f"action_joint_ids: {list(action_term._joint_ids)}")
        _print(f"action_joint_names: {_format_names(action_term._joint_names)}")

        for actuator_name, actuator in robot.actuators.items():
            joint_indices = [int(joint_index) for joint_index in actuator.joint_indices]
            _print(f"actuator/{actuator_name}/joints: {joint_indices}")
            _print(f"actuator/{actuator_name}/effort_limit: {actuator.cfg.effort_limit}")
            _print(f"actuator/{actuator_name}/saturation_effort: {getattr(actuator.cfg, 'saturation_effort', None)}")
            _print(f"actuator/{actuator_name}/velocity_limit: {actuator.cfg.velocity_limit}")
            _print(f"actuator/{actuator_name}/stiffness: {actuator.cfg.stiffness}")
            _print(f"actuator/{actuator_name}/damping: {actuator.cfg.damping}")

        root_z = robot.data.root_pos_w[:, 2]
        calf_z = robot.data.body_pos_w[:, calf_ids, 2]
        foot_z = robot.data.body_pos_w[:, foot_ids, 2]
        _print(f"reset_root_z_min: {root_z.min().item():.6f}")
        _print(f"reset_calf_z_min: {calf_z.min().item():.6f}")
        _print(f"reset_foot_z_min: {foot_z.min().item():.6f}")

        counts = _term_counts(env, args_cli.steps, args_cli.action_mode)
        _print(f"rollout_steps: {args_cli.steps}")
        _print(f"action_mode: {args_cli.action_mode}")
        for name, count in counts.items():
            _print(f"termination_count/{name}: {count}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
