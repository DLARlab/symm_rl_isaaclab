# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record fixed six-direction command groups without averaging environments."""

from __future__ import annotations

import math
import types
from pathlib import Path

import numpy as np
import torch
from symm_rollout_plotter import SymmetricRolloutPlotter


class SymmetricCommandBatchRecorder(SymmetricRolloutPlotter):
    """Save one time-major rollout archive per direction with an environment axis."""

    def __init__(
        self,
        env,
        output_dir: str | Path,
        *,
        envs_per_command: int = 100,
        duration: float = 20.0,
        forward_speed: float = 1.0,
        lateral_speed: float = 0.5,
        yaw_rate: float = 0.6,
    ) -> None:
        """Configure the groups and reserve a fixed recording duration.

        Args:
            env: Unwrapped symmetric quadruped environment with pre-reset diagnostics.
            output_dir: Parent directory for the six command directories.
            envs_per_command: Number of independent environments assigned to each direction.
            duration: Simulated recording duration [s], an integer multiple of the control interval.
            forward_speed: Magnitude of forward and backward velocity commands [m/s].
            lateral_speed: Magnitude of left and right velocity commands [m/s].
            yaw_rate: Magnitude of left and right turning-rate commands [rad/s].
        """
        for name, value in (
            ("duration", duration),
            ("forward_speed", forward_speed),
            ("lateral_speed", lateral_speed),
            ("yaw_rate", yaw_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, received {value}.")
        if envs_per_command <= 0:
            raise ValueError("envs_per_command must be positive.")
        expected_envs = len(self._TRACKING_DIRECTION_SPECS) * envs_per_command
        if env.num_envs != expected_envs:
            raise ValueError(f"Six command groups require {expected_envs} environments, received {env.num_envs}.")
        steps = round(duration / env.step_dt)
        if steps < 1 or not math.isclose(steps * env.step_dt, duration, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("duration must be a positive integer multiple of the environment control interval.")
        super().__init__(env, output_dir, max_samples=steps)
        self._envs_per_command = envs_per_command
        self._step_count = 0
        self._samples: dict[str, np.ndarray] = {}
        self._override_installed = False
        speeds = (forward_speed, lateral_speed, yaw_rate)
        commands = torch.zeros(
            (6, 3), dtype=self._command_term.vel_command_b.dtype, device=self._command_term.vel_command_b.device
        )
        for index, (_, axis, sign, _) in enumerate(self._TRACKING_DIRECTION_SPECS):
            commands[index, axis] = sign * speeds[axis]
        self._commands = commands.repeat_interleave(envs_per_command, dim=0)
        self.reset_reference()

    @property
    def total_steps(self) -> int:
        """Number of control steps to record."""
        return self._max_samples

    @property
    def is_complete(self) -> bool:
        """Whether every requested control step has been recorded."""
        return self._step_count >= self.total_steps

    def reset_reference(self) -> None:
        """Start desired world positions [m] and headings [rad] at the current reset state."""
        self._desired_positions = self._robot.data.root_pos_w.torch[:, :2].detach().cpu().numpy().astype(np.float64)
        self._desired_headings = self._robot.data.heading_w.torch.detach().cpu().numpy().astype(np.float64)

    def install_command_override(self) -> None:
        """Keep assigned commands fixed during initial reset, resampling, and later resets."""
        if self._override_installed:
            return

        def resample_fixed_commands(command_term, env_ids) -> None:
            command_term.vel_command_b[env_ids] = self._commands[env_ids]
            for name in ("is_standing_env", "is_heading_env"):
                if hasattr(command_term, name):
                    getattr(command_term, name)[env_ids] = False

        self._command_term._resample_command = types.MethodType(resample_fixed_commands, self._command_term)
        resample_fixed_commands(self._command_term, slice(None))
        self._override_installed = True

    def _cached(self, name: str) -> torch.Tensor:
        """Require a pre-reset snapshot instead of mixing states across episode boundaries."""
        value = getattr(self._env, name, None)
        if value is None:
            raise RuntimeError(f"Batched recordings require the symmetric environment's {name} snapshot.")
        return value

    @torch.no_grad()
    def record(
        self,
        actions: torch.Tensor | None = None,
        actor_means: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> None:
        """Retain one post-step sample for every environment, including terminal samples.

        Args:
            actions: Raw policy outputs, with shape [environment, action].
            actor_means: Deterministic policy means; defaults to the supplied actions.
            dones: Episode-end flags, with shape [environment].
        """
        if self.is_complete:
            return
        if actions is None:
            actions = self._env.action_manager.action
        if actor_means is None:
            actor_means = actions
        if dones is None:
            dones = torch.zeros(self._env.num_envs, dtype=torch.bool, device=actions.device)
        if dones.shape != (self._env.num_envs,):
            raise ValueError("Expected one episode-end flag per environment.")
        command = self._cached("_last_base_velocity_commands")
        if not torch.equal(command, self._commands):
            raise RuntimeError("An environment's command changed during the fixed-command recording.")
        velocity = self._cached("_last_root_lin_velocities_b")
        angular_velocity = self._cached("_last_root_ang_velocities_b")
        positions = self._cached("_last_joint_positions")
        targets = self._cached("_last_joint_position_targets")
        joint_velocities = self._cached("_last_joint_velocities")
        torques = self._cached("_last_joint_torques")
        limits = self._cached("_last_soft_joint_pos_limits")
        limit_range = (limits[..., 1] - limits[..., 0]).clamp_min(torch.finfo(positions.dtype).eps)
        position_utilization = (2 * (positions - limits[..., 0]) / limit_range - 1).abs()
        target_utilization = (2 * (targets - limits[..., 0]) / limit_range - 1).abs()
        forces = self._cached("_last_foot_ground_reaction_forces_w")
        command_numpy = command.detach().cpu().numpy()
        self._desired_headings += command_numpy[:, 2] * self._step_dt
        cosine, sine = np.cos(self._desired_headings), np.sin(self._desired_headings)
        self._desired_positions += (
            np.stack(
                (
                    cosine * command_numpy[:, 0] - sine * command_numpy[:, 1],
                    sine * command_numpy[:, 0] + cosine * command_numpy[:, 1],
                ),
                axis=-1,
            )
            * self._step_dt
        )
        values = {
            "true_lin_vel": torch.stack((velocity[:, 0], velocity[:, 1], angular_velocity[:, 2]), dim=-1),
            "desired_lin_vel": command_numpy,
            "base_positions": self._cached("_last_root_positions_w"),
            "desired_positions": self._desired_positions,
            "E_C_frc": self._cached("_last_periodic_force_weights"),
            "E_C_spd": self._cached("_last_periodic_speed_weights"),
            "foot_normal_forces_w": self._cached("_last_foot_normal_forces_w"),
            "foot_ground_reaction_forces_w": forces,
            "ground_reaction_force_includes_friction": np.full(
                self._env.num_envs, bool(getattr(self._env, "_last_ground_reaction_force_includes_friction", False))
            ),
            "episode_done": dones.to(dtype=torch.bool),
            "foot_forces": torch.linalg.norm(forces, dim=-1),
            "foot_velocities": torch.linalg.norm(self._cached("_last_foot_velocities_w"), dim=-1),
            "raw_actions": actions,
            "actor_means": actor_means,
            "applied_actions": targets,
            "joint_positions": positions,
            "joint_position_targets": targets,
            "joint_velocities": joint_velocities,
            "joint_torques": torques,
            "joint_powers": torques * joint_velocities,
            "joint_position_lower_limits": limits[..., 0],
            "joint_position_upper_limits": limits[..., 1],
            "joint_limit_utilization": position_utilization,
            "joint_near_limit_fraction": (position_utilization >= 0.90).float().mean(dim=-1),
            "joint_limit_violation_fraction": (position_utilization > 1.0).float().mean(dim=-1),
            "joint_target_limit_utilization": target_utilization,
            "joint_target_near_limit_fraction": (target_utilization >= 0.90).float().mean(dim=-1),
            "joint_target_limit_violation_fraction": (target_utilization > 1.0).float().mean(dim=-1),
            "running_reward_clipped": getattr(
                self._env, "_running_reward_negative", np.zeros(self._env.num_envs, dtype=bool)
            ),
        }
        diagnostics = getattr(self._env, "_straight_line_motion_diagnostics", {})
        for name in ("forward_score", "straight_score", "posture_score", "support_loss", "straight_line_reward"):
            source = "reward" if name == "straight_line_reward" else name
            values[name] = diagnostics.get(source, np.full(self._env.num_envs, np.nan, dtype=np.float32))
        clearance = getattr(self._env, "_foot_clearance_diagnostics", {})
        for name, source in (
            ("foot_heights", "foot_height"),
            ("foot_clearance_targets", "target_height"),
            ("foot_clearance_shortfalls", "shortfall"),
            ("foot_clearance_swing_weights", "swing_weight"),
            ("foot_clearance_penalty", "penalty"),
        ):
            shape = (self._env.num_envs,) if source == "penalty" else (self._env.num_envs, 4)
            values[name] = clearance.get(source, np.full(shape, np.nan, dtype=np.float32))
        for name, value in values.items():
            array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if name not in self._samples:
                self._samples[name] = np.empty((self.total_steps, *array.shape), dtype=array.dtype)
            self._samples[name][self._step_count] = array
        self._step_count += 1
        # Resume the desired trajectory from each environment's post-reset pose on its next step.
        reset_ids = dones.to(dtype=torch.bool).detach().cpu().numpy()
        if reset_ids.any():
            self._desired_positions[reset_ids] = (
                self._robot.data.root_pos_w.torch[:, :2].detach().cpu().numpy()[reset_ids]
            )
            self._desired_headings[reset_ids] = self._robot.data.heading_w.torch.detach().cpu().numpy()[reset_ids]
        if self.is_complete:
            self._env._capture_rollout_diagnostics = False

    def save(self) -> list[Path]:
        """Save six archives with signal shapes [time, environment, ...].

        Returns:
            Paths to command-specific ``sim_data.npz`` files. Time [s] has shape
            [time]; ``env_ids`` identifies the environment axis. Joint positions
            [rad], velocities [rad/s], torques [N m], powers [W], foot forces [N],
            and body velocities [m/s, m/s, rad/s] retain their existing field names.
            Interrupted recordings contain their actual step count and a false
            ``recording_complete`` flag.
        """
        self._env._capture_rollout_diagnostics = False
        if not self._step_count:
            return []
        paths = []
        for index, (name, _, _, _) in enumerate(self._TRACKING_DIRECTION_SPECS):
            start = index * self._envs_per_command
            stop = start + self._envs_per_command
            data = {key: values[: self._step_count, start:stop] for key, values in self._samples.items()}
            data["time_steps"] = np.arange(self._step_count, dtype=np.float64) * self._step_dt
            data = self._prepare_data(data)
            data.update(
                env_ids=np.arange(start, stop),
                command_name=np.asarray(name),
                assigned_command=self._commands[start].detach().cpu().numpy(),
                step_dt=np.asarray(self._step_dt),
                requested_steps=np.asarray(self.total_steps),
                recorded_steps=np.asarray(self._step_count),
                recording_complete=np.asarray(self.is_complete),
            )
            path = self._output_dir / name / "sim_data.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary_path, **data)
            temporary_path.replace(path)
            paths.append(path)
            print(
                f"[symm_locomotion] Saved {name}: {self._step_count} steps x "
                f"{self._envs_per_command} environments to {path}",
                flush=True,
            )
        return paths
