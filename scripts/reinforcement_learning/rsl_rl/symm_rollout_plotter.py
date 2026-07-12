# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot symmetric quadruped rollout diagnostics during RSL-RL playback."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class SymmetricRolloutPlotter:
    """Collect and save the diagnostic plots from the IsaacGym symmetric rollout."""

    _FOOT_SENSOR_NAMES = ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot")
    _LEG_NAMES = ("Front Left", "Front Right", "Rear Left", "Rear Right")

    def __init__(
        self,
        env,
        output_dir: str | Path,
        env_index: int = 0,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the rollout collector.

        Args:
            env: Unwrapped symmetric quadruped environment.
            output_dir: Directory in which plots and sampled data are saved.
            env_index: Environment index to plot.
            max_samples: Maximum number of post-step samples to retain.
        """
        if env_index < 0 or env_index >= env.num_envs:
            raise ValueError(f"Plot env_index {env_index} is outside the valid range [0, {env.num_envs - 1}].")
        if max_samples is not None and max_samples < 1:
            raise ValueError("Plot max_samples must be positive when specified.")

        try:
            command_term = env.command_manager.get_term("base_velocity")
        except Exception as exc:
            raise ValueError("Symmetric rollout plots require a 'base_velocity' command term.") from exc
        required_command_fields = (
            "command",
            "periodic_force_weights",
            "periodic_speed_weights",
            "robot",
        )
        missing_fields = [name for name in required_command_fields if not hasattr(command_term, name)]
        if missing_fields:
            raise ValueError(f"The base_velocity command term does not support symmetric plots: {missing_fields}.")

        self._env = env
        self._env_index = env_index
        self._command_term = command_term
        self._robot = command_term.robot
        self._output_dir = Path(output_dir)
        self._step_dt = float(env.step_dt)
        self._max_samples = max_samples
        self._foot_sensors = []
        foot_body_names = []
        for sensor_name in self._FOOT_SENSOR_NAMES:
            try:
                sensor = env.scene.sensors[sensor_name]
            except KeyError as exc:
                raise ValueError(f"Symmetric rollout plots require contact sensor '{sensor_name}'.") from exc
            if sensor.num_sensors != 1 or len(sensor.body_names) != 1:
                raise ValueError(f"Contact sensor '{sensor_name}' must resolve exactly one foot body.")
            self._foot_sensors.append(sensor)
            foot_body_names.append(sensor.body_names[0])

        self._foot_body_ids, matched_body_names = self._robot.find_bodies(foot_body_names, preserve_order=True)
        if matched_body_names != foot_body_names:
            raise ValueError(f"Foot body order mismatch: expected {foot_body_names}, resolved {matched_body_names}.")

        root_position = self._robot.data.root_pos_w.torch[env_index, :2].detach().cpu().numpy()
        self._desired_position = root_position.astype(np.float64, copy=True)
        self._desired_heading = float(self._robot.data.heading_w.torch[env_index].detach().cpu())
        self._data: dict[str, list[np.ndarray | float]] = {
            "time_steps": [],
            "true_lin_vel": [],
            "desired_lin_vel": [],
            "base_positions": [],
            "desired_positions": [],
            "E_C_frc": [],
            "E_C_spd": [],
            "foot_forces": [],
            "foot_velocities": [],
        }

    @torch.no_grad()
    def record(self) -> None:
        """Record one post-step sample from the selected environment."""
        if self._max_samples is not None and len(self._data["time_steps"]) >= self._max_samples:
            return
        env_index = self._env_index
        command = self._command_term.command[env_index].detach().cpu()
        true_velocity = (
            torch.stack(
                (
                    self._robot.data.root_lin_vel_b.torch[env_index, 0],
                    self._robot.data.root_lin_vel_b.torch[env_index, 1],
                    self._robot.data.root_ang_vel_b.torch[env_index, 2],
                )
            )
            .detach()
            .cpu()
        )
        root_position = self._robot.data.root_pos_w.torch[env_index, :2].detach().cpu()

        self._desired_heading += float(command[2]) * self._step_dt
        cos_heading = np.cos(self._desired_heading)
        sin_heading = np.sin(self._desired_heading)
        desired_velocity_w = np.array(
            (
                cos_heading * float(command[0]) - sin_heading * float(command[1]),
                sin_heading * float(command[0]) + cos_heading * float(command[1]),
            )
        )
        self._desired_position += desired_velocity_w * self._step_dt

        contact_forces = []
        for sensor in self._foot_sensors:
            net_forces_w = sensor.data.net_forces_w
            if net_forces_w is None:
                raise RuntimeError("Foot contact sensor does not expose net_forces_w.")
            contact_forces.append(torch.linalg.norm(net_forces_w.torch[env_index, 0]).detach().cpu())
        foot_velocity = self._robot.data.body_lin_vel_w.torch[env_index, self._foot_body_ids]

        self._data["time_steps"].append(len(self._data["time_steps"]) * self._step_dt)
        self._data["true_lin_vel"].append(true_velocity.numpy())
        self._data["desired_lin_vel"].append(command.numpy())
        self._data["base_positions"].append(root_position.numpy())
        self._data["desired_positions"].append(self._desired_position.copy())
        self._data["E_C_frc"].append(self._command_term.periodic_force_weights()[env_index].detach().cpu().numpy())
        self._data["E_C_spd"].append(self._command_term.periodic_speed_weights()[env_index].detach().cpu().numpy())
        self._data["foot_forces"].append(torch.stack(contact_forces).numpy())
        self._data["foot_velocities"].append(torch.linalg.norm(foot_velocity, dim=-1).detach().cpu().numpy())

    def save(self) -> list[Path]:
        """Save sampled arrays and diagnostic figures.

        Returns:
            Paths of the files that were saved. The list is empty when no samples were recorded.
        """
        if not self._data["time_steps"]:
            print("[symm_locomotion] No rollout samples were collected; skipping plots.", flush=True)
            return []

        self._output_dir.mkdir(parents=True, exist_ok=True)
        data = {name: np.asarray(values) for name, values in self._data.items()}
        data_path = self._output_dir / "sim_data.npz"
        np.savez_compressed(data_path, **data)

        paths = [
            data_path,
            self._save_velocity_and_position_plot(data),
            self._save_per_foot_plot(
                data,
                weight_key="E_C_frc",
                measurement_key="foot_forces",
                filename="figure2_E_C_frc_and_contact_forces.png",
                weight_label="E_C_frc",
                measurement_label="Contact Force",
                measurement_unit="N",
            ),
            self._save_per_foot_plot(
                data,
                weight_key="E_C_spd",
                measurement_key="foot_velocities",
                filename="figure3_E_C_spd_and_foot_velocities.png",
                weight_label="E_C_spd",
                measurement_label="Foot Speed",
                measurement_unit="m/s",
            ),
            self._save_aggregate_force_plot(data),
        ]
        print(f"[symm_locomotion] Saved rollout plots to: {self._output_dir}", flush=True)
        return paths

    def _save_velocity_and_position_plot(self, data: dict[str, np.ndarray]) -> Path:
        time_steps = data["time_steps"]
        true_velocity = data["true_lin_vel"]
        desired_velocity = data["desired_lin_vel"]
        base_positions = data["base_positions"]
        desired_positions = data["desired_positions"]
        fig, axes = plt.subplots(4, 1, figsize=(10, 18))
        velocity_specs = (
            (0, "Linear X Velocity", "m/s"),
            (1, "Linear Y Velocity", "m/s"),
            (2, "Yaw Velocity", "rad/s"),
        )
        for axis, (index, title, unit) in zip(axes[:3], velocity_specs, strict=True):
            axis.plot(time_steps, true_velocity[:, index], label="True")
            axis.plot(time_steps, desired_velocity[:, index], linestyle="--", label="Desired")
            axis.set_title(f"{title} vs Time")
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(f"Velocity ({unit})")
            axis.legend()
            axis.grid(True)

        axes[3].plot(base_positions[:, 0], base_positions[:, 1], label="True Base XY")
        axes[3].plot(
            desired_positions[:, 0],
            desired_positions[:, 1],
            linestyle="--",
            label="Desired Base XY",
        )
        axes[3].set_title("Base Position Tracking")
        axes[3].set_xlabel("X (m)")
        axes[3].set_ylabel("Y (m)")
        axes[3].axis("equal")
        axes[3].legend()
        axes[3].grid(True)
        fig.tight_layout()
        path = self._output_dir / "figure1_linear_velocities_and_position.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_per_foot_plot(
        self,
        data: dict[str, np.ndarray],
        *,
        weight_key: str,
        measurement_key: str,
        filename: str,
        weight_label: str,
        measurement_label: str,
        measurement_unit: str,
    ) -> Path:
        time_steps = data["time_steps"]
        weights = data[weight_key]
        measurements = data[measurement_key]
        fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
        for index, axis_left in enumerate(axes):
            axis_right = axis_left.twinx()
            weight_lines = axis_left.plot(
                time_steps,
                weights[:, index],
                label=f"{weight_label} ({self._LEG_NAMES[index]})",
                color="tab:blue",
            )
            measurement_lines = axis_right.plot(
                time_steps,
                measurements[:, index],
                linestyle="--",
                label=f"{measurement_label} ({self._LEG_NAMES[index]})",
                color="tab:red",
            )
            axis_left.set_title(f"{self._LEG_NAMES[index]}: {weight_label} vs {measurement_label}")
            axis_left.set_xlabel("Time (s)")
            axis_left.set_ylabel(f"{weight_label} (arb.)", color="tab:blue")
            axis_right.set_ylabel(f"{measurement_label} ({measurement_unit})", color="tab:red")
            axis_left.tick_params(axis="y", colors="tab:blue")
            axis_right.tick_params(axis="y", colors="tab:red")
            lines = weight_lines + measurement_lines
            axis_left.legend(lines, [line.get_label() for line in lines], loc="upper right")
            axis_left.grid(True)
        fig.tight_layout()
        path = self._output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_aggregate_force_plot(self, data: dict[str, np.ndarray]) -> Path:
        time_steps = data["time_steps"]
        mean_force_weight = np.mean(data["E_C_frc"], axis=1)
        mean_contact_force = np.mean(data["foot_forces"], axis=1)
        fig, axis_left = plt.subplots(1, 1, figsize=(10, 6))
        axis_right = axis_left.twinx()
        axis_left.plot(time_steps, mean_force_weight, label="Mean E_C_frc", color="tab:blue")
        axis_right.plot(
            time_steps,
            mean_contact_force,
            linestyle="--",
            label="Mean Contact Force",
            color="tab:red",
        )
        axis_left.set_title("Aggregated E_C_frc vs Contact Force")
        axis_left.set_xlabel("Time (s)")
        axis_left.set_ylabel("Mean E_C_frc (arb.)", color="tab:blue")
        axis_right.set_ylabel("Mean Contact Force (N)", color="tab:red")
        axis_left.tick_params(axis="y", colors="tab:blue")
        axis_right.tick_params(axis="y", colors="tab:red")
        lines = axis_left.get_lines() + axis_right.get_lines()
        axis_left.legend(lines, [line.get_label() for line in lines], loc="upper right")
        axis_left.grid(True)
        fig.tight_layout()
        path = self._output_dir / "figure4_agg_E_C_frc_vs_contact.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
