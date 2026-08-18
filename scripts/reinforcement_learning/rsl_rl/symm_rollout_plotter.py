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


def _centered_moving_mean(
    values: np.ndarray,
    window_samples: int,
    episode_ends: np.ndarray | None = None,
) -> np.ndarray:
    """Compute an edge-corrected centered moving mean along the time axis.

    Args:
        values: Time-major values to smooth.
        window_samples: Odd number of samples in the centered window.
        episode_ends: Optional time-major mask whose true samples end an episode.
            Smoothing windows never cross these boundaries.

    Returns:
        Smoothed values as ``float64`` with the same shape as :paramref:`values`.
    """
    if window_samples < 1 or window_samples % 2 == 0:
        raise ValueError(f"Moving-mean window must be a positive odd number, received {window_samples}.")

    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError("Moving-mean values must have a time axis.")
    if values.shape[0] == 0:
        return values.astype(np.float64)

    if episode_ends is None:
        episode_ends = np.zeros(values.shape[0], dtype=bool)
    else:
        episode_ends = np.asarray(episode_ends, dtype=bool)
        if episode_ends.shape != (values.shape[0],):
            raise ValueError(f"Episode-end mask must have shape ({values.shape[0]},), received {episode_ends.shape}.")

    half_window = window_samples // 2
    result = np.empty(values.shape, dtype=np.float64)
    segment_start = 0
    segment_stops = [*(np.flatnonzero(episode_ends) + 1), values.shape[0]]
    for segment_stop in segment_stops:
        if segment_stop <= segment_start:
            continue
        segment = values[segment_start:segment_stop]
        sample_indices = np.arange(segment.shape[0])
        starts = np.maximum(sample_indices - half_window, 0)
        stops = np.minimum(sample_indices + half_window + 1, segment.shape[0])
        prefix_shape = (1, *segment.shape[1:])
        prefix_sums = np.concatenate(
            (np.zeros(prefix_shape, dtype=np.float64), np.cumsum(segment, axis=0, dtype=np.float64)),
            axis=0,
        )
        counts = (stops - starts).reshape((-1, *([1] * (values.ndim - 1))))
        result[segment_start:segment_stop] = (prefix_sums[stops] - prefix_sums[starts]) / counts
        segment_start = segment_stop
    return result


class SymmetricRolloutPlotter:
    """Collect and save the diagnostic plots from the IsaacGym symmetric rollout."""

    _FOOT_SENSOR_NAMES = ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot")
    _LEG_NAMES = ("Front Left", "Front Right", "Rear Left", "Rear Right")
    _MOTOR_ROLE_NAMES = ("Hip/Abad", "Thigh", "Calf")
    _JOINTS_PER_LEG = 3
    _USAGE_SMOOTHING_WINDOW_S = 1.0
    _TRACKING_ERROR_COMPONENTS = ("lin_x", "lin_y", "yaw")
    _TRACKING_DIRECTION_SPECS = (
        ("forward", 0, 1.0, "m/s"),
        ("backward", 0, -1.0, "m/s"),
        ("left", 1, 1.0, "m/s"),
        ("right", 1, -1.0, "m/s"),
        ("yaw_left", 2, 1.0, "rad/s"),
        ("yaw_right", 2, -1.0, "rad/s"),
    )
    _TRACKING_DIRECTION_COMMAND_THRESHOLD = 0.05

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
        self._env._capture_rollout_diagnostics = True
        self._env_index = env_index
        self._command_term = command_term
        self._robot = command_term.robot
        try:
            self._joint_action_term = env.action_manager.get_term("joint_pos")
        except Exception as exc:
            raise ValueError("Symmetric rollout plots require a 'joint_pos' action term.") from exc
        self._joint_ids = self._joint_action_term._joint_ids
        self._joint_names = tuple(self._joint_action_term._joint_names)
        expected_joint_count = len(self._LEG_NAMES) * self._JOINTS_PER_LEG
        if len(self._joint_names) != expected_joint_count:
            raise ValueError(
                "Symmetric rollout leg-usage plots require exactly "
                f"{expected_joint_count} leg-major joints, received {len(self._joint_names)}."
            )
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
        self._env._rollout_foot_sensor_names = self._FOOT_SENSOR_NAMES

        self._foot_body_ids, matched_body_names = self._robot.find_bodies(foot_body_names, preserve_order=True)
        if matched_body_names != foot_body_names:
            raise ValueError(f"Foot body order mismatch: expected {foot_body_names}, resolved {matched_body_names}.")
        self._env._rollout_foot_body_ids = self._foot_body_ids

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
            "foot_normal_forces_w": [],
            "foot_ground_reaction_forces_w": [],
            "ground_reaction_force_includes_friction": [],
            "episode_done": [],
            "foot_forces": [],
            "foot_velocities": [],
            "raw_actions": [],
            "actor_means": [],
            "applied_actions": [],
            "joint_positions": [],
            "joint_position_targets": [],
            "joint_velocities": [],
            "joint_torques": [],
            "joint_powers": [],
            "joint_position_lower_limits": [],
            "joint_position_upper_limits": [],
            "joint_limit_utilization": [],
            "joint_near_limit_fraction": [],
            "joint_limit_violation_fraction": [],
            "joint_target_limit_utilization": [],
            "joint_target_near_limit_fraction": [],
            "joint_target_limit_violation_fraction": [],
            "forward_score": [],
            "straight_score": [],
            "posture_score": [],
            "support_loss": [],
            "straight_line_reward": [],
            "running_reward_clipped": [],
            "foot_heights": [],
            "foot_clearance_targets": [],
            "foot_clearance_shortfalls": [],
            "foot_clearance_swing_weights": [],
            "foot_clearance_penalty": [],
        }

    @torch.no_grad()
    def record(
        self,
        actions: torch.Tensor | None = None,
        actor_means: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> None:
        """Record one post-step sample from the selected environment.

        Args:
            actions: Raw policy actions before the environment wrapper applies optional clipping.
            actor_means: Deterministic actor means. During inference these are the same as :paramref:`actions`.
            dones: Post-step episode-end mask. It prevents smoothing across resets.
        """
        if self._max_samples is not None and len(self._data["time_steps"]) >= self._max_samples:
            self._env._capture_rollout_diagnostics = False
            return
        env_index = self._env_index
        if actions is None:
            actions = getattr(getattr(self._env, "action_manager", None), "action", None)
        if actions is None:
            raise ValueError("Raw actions must be provided when the environment does not expose action_manager.action.")
        if actor_means is None:
            actor_means = actions
        episode_done = False if dones is None else bool(dones[env_index].detach().cpu())
        cached_commands = getattr(self._env, "_last_base_velocity_commands", None)
        command = (
            (self._command_term.command[env_index] if cached_commands is None else cached_commands[env_index])
            .detach()
            .cpu()
        )
        cached_root_lin_velocities_b = getattr(self._env, "_last_root_lin_velocities_b", None)
        cached_root_ang_velocities_b = getattr(self._env, "_last_root_ang_velocities_b", None)
        root_lin_velocity_b = (
            self._robot.data.root_lin_vel_b.torch[env_index]
            if cached_root_lin_velocities_b is None
            else cached_root_lin_velocities_b[env_index]
        )
        root_ang_velocity_b = (
            self._robot.data.root_ang_vel_b.torch[env_index]
            if cached_root_ang_velocities_b is None
            else cached_root_ang_velocities_b[env_index]
        )
        true_velocity = (
            torch.stack(
                (
                    root_lin_velocity_b[0],
                    root_lin_velocity_b[1],
                    root_ang_velocity_b[2],
                )
            )
            .detach()
            .cpu()
        )
        cached_root_positions_w = getattr(self._env, "_last_root_positions_w", None)
        root_position = (
            (
                self._robot.data.root_pos_w.torch[env_index, :2]
                if cached_root_positions_w is None
                else cached_root_positions_w[env_index]
            )
            .detach()
            .cpu()
        )

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

        cached_foot_normal_forces_w = getattr(self._env, "_last_foot_normal_forces_w", None)
        cached_foot_ground_reaction_forces_w = getattr(
            self._env,
            "_last_foot_ground_reaction_forces_w",
            None,
        )
        if cached_foot_normal_forces_w is None:
            foot_normal_forces_w = []
            foot_ground_reaction_forces_w = []
            ground_reaction_force_includes_friction = True
            for sensor in self._foot_sensors:
                force_matrix_w = getattr(sensor.data, "force_matrix_w", None)
                if force_matrix_w is None:
                    net_forces_w = sensor.data.net_forces_w
                    if net_forces_w is None:
                        raise RuntimeError("Foot contact sensor does not expose net_forces_w.")
                    normal_force_w = net_forces_w.torch[env_index, 0]
                    ground_reaction_force_includes_friction = False
                else:
                    normal_force_w = force_matrix_w.torch[env_index, 0].sum(dim=0)
                friction_forces_w = getattr(sensor.data, "friction_forces_w", None)
                if friction_forces_w is None:
                    ground_reaction_force_w = normal_force_w
                    ground_reaction_force_includes_friction = False
                else:
                    ground_reaction_force_w = normal_force_w + friction_forces_w.torch[env_index, 0].sum(dim=0)
                foot_normal_forces_w.append(normal_force_w)
                foot_ground_reaction_forces_w.append(ground_reaction_force_w)
            foot_normal_forces_w = torch.stack(foot_normal_forces_w)
            foot_ground_reaction_forces_w = torch.stack(foot_ground_reaction_forces_w)
        else:
            foot_normal_forces_w = cached_foot_normal_forces_w[env_index]
            foot_ground_reaction_forces_w = (
                foot_normal_forces_w
                if cached_foot_ground_reaction_forces_w is None
                else cached_foot_ground_reaction_forces_w[env_index]
            )
            ground_reaction_force_includes_friction = bool(
                getattr(self._env, "_last_ground_reaction_force_includes_friction", False)
            )
        foot_normal_forces_w = foot_normal_forces_w.detach().cpu()
        foot_ground_reaction_forces_w = foot_ground_reaction_forces_w.detach().cpu()
        contact_force_norms = torch.linalg.norm(foot_ground_reaction_forces_w, dim=-1)
        cached_foot_velocities_w = getattr(self._env, "_last_foot_velocities_w", None)
        foot_velocity = (
            self._robot.data.body_lin_vel_w.torch[env_index, self._foot_body_ids]
            if cached_foot_velocities_w is None
            else cached_foot_velocities_w[env_index]
        )
        cached_joint_positions = getattr(self._env, "_last_joint_positions", None)
        cached_joint_targets = getattr(self._env, "_last_joint_position_targets", None)
        cached_joint_limits = getattr(self._env, "_last_soft_joint_pos_limits", None)
        cached_joint_velocities = getattr(self._env, "_last_joint_velocities", None)
        cached_joint_torques = getattr(self._env, "_last_joint_torques", None)
        joint_positions = (
            (
                self._robot.data.joint_pos.torch[env_index, self._joint_ids]
                if cached_joint_positions is None
                else cached_joint_positions[env_index]
            )
            .detach()
            .cpu()
        )
        joint_position_targets = (
            (
                self._joint_action_term.processed_actions[env_index]
                if cached_joint_targets is None
                else cached_joint_targets[env_index]
            )
            .detach()
            .cpu()
        )
        joint_velocities = (
            (
                self._robot.data.joint_vel.torch[env_index, self._joint_ids]
                if cached_joint_velocities is None
                else cached_joint_velocities[env_index]
            )
            .detach()
            .cpu()
        )
        joint_torques = (
            (
                self._robot.data.applied_torque.torch[env_index, self._joint_ids]
                if cached_joint_torques is None
                else cached_joint_torques[env_index]
            )
            .detach()
            .cpu()
        )
        joint_powers = joint_torques * joint_velocities
        joint_limits = (
            (
                self._robot.data.soft_joint_pos_limits.torch[env_index, self._joint_ids]
                if cached_joint_limits is None
                else cached_joint_limits[env_index]
            )
            .detach()
            .cpu()
        )
        joint_limit_range = (joint_limits[:, 1] - joint_limits[:, 0]).clamp_min(torch.finfo(joint_positions.dtype).eps)
        normalized_joint_position = 2.0 * (joint_positions - joint_limits[:, 0]) / joint_limit_range - 1.0
        normalized_joint_target = 2.0 * (joint_position_targets - joint_limits[:, 0]) / joint_limit_range - 1.0
        near_joint_limit = normalized_joint_position.abs() >= 0.90
        joint_limit_violation = normalized_joint_position.abs() > 1.0
        near_joint_target_limit = normalized_joint_target.abs() >= 0.90
        joint_target_limit_violation = normalized_joint_target.abs() > 1.0

        reward_diagnostics = getattr(self._env, "_straight_line_motion_diagnostics", {})

        def reward_component(name: str) -> float:
            value = reward_diagnostics.get(name)
            return float("nan") if value is None else float(value[env_index].detach().cpu())

        reward_clipped = getattr(
            self._env,
            "_running_reward_negative",
            getattr(self._env, "_running_reward_clipped", None),
        )
        selected_reward_clipped = False if reward_clipped is None else bool(reward_clipped[env_index].detach().cpu())
        foot_clearance_diagnostics = getattr(self._env, "_foot_clearance_diagnostics", {})

        def foot_clearance_component(name: str, *, per_foot: bool = True) -> np.ndarray | float:
            value = foot_clearance_diagnostics.get(name)
            if value is None:
                return np.full(4, np.nan) if per_foot else float("nan")
            selected_value = value[env_index].detach().cpu()
            return selected_value.numpy() if per_foot else float(selected_value)

        self._data["time_steps"].append(len(self._data["time_steps"]) * self._step_dt)
        self._data["true_lin_vel"].append(true_velocity.numpy())
        self._data["desired_lin_vel"].append(command.numpy())
        self._data["base_positions"].append(root_position.numpy())
        self._data["desired_positions"].append(self._desired_position.copy())
        cached_periodic_force_weights = getattr(self._env, "_last_periodic_force_weights", None)
        periodic_force_weights = (
            self._command_term.periodic_force_weights()[env_index]
            if cached_periodic_force_weights is None
            else cached_periodic_force_weights[env_index]
        )
        cached_periodic_speed_weights = getattr(self._env, "_last_periodic_speed_weights", None)
        periodic_speed_weights = (
            self._command_term.periodic_speed_weights()[env_index]
            if cached_periodic_speed_weights is None
            else cached_periodic_speed_weights[env_index]
        )
        self._data["E_C_frc"].append(periodic_force_weights.detach().cpu().numpy())
        self._data["E_C_spd"].append(periodic_speed_weights.detach().cpu().numpy())
        self._data["foot_normal_forces_w"].append(foot_normal_forces_w.numpy())
        self._data["foot_ground_reaction_forces_w"].append(foot_ground_reaction_forces_w.numpy())
        self._data["ground_reaction_force_includes_friction"].append(ground_reaction_force_includes_friction)
        self._data["episode_done"].append(episode_done)
        self._data["foot_forces"].append(contact_force_norms.numpy())
        self._data["foot_velocities"].append(torch.linalg.norm(foot_velocity, dim=-1).detach().cpu().numpy())
        self._data["raw_actions"].append(actions[env_index].detach().cpu().numpy())
        self._data["actor_means"].append(actor_means[env_index].detach().cpu().numpy())
        self._data["applied_actions"].append(joint_position_targets.numpy())
        self._data["joint_positions"].append(joint_positions.numpy())
        self._data["joint_position_targets"].append(joint_position_targets.numpy())
        self._data["joint_velocities"].append(joint_velocities.numpy())
        self._data["joint_torques"].append(joint_torques.numpy())
        self._data["joint_powers"].append(joint_powers.numpy())
        self._data["joint_position_lower_limits"].append(joint_limits[:, 0].numpy())
        self._data["joint_position_upper_limits"].append(joint_limits[:, 1].numpy())
        self._data["joint_limit_utilization"].append(normalized_joint_position.abs().numpy())
        self._data["joint_near_limit_fraction"].append(float(near_joint_limit.to(torch.float32).mean()))
        self._data["joint_limit_violation_fraction"].append(float(joint_limit_violation.to(torch.float32).mean()))
        self._data["joint_target_limit_utilization"].append(normalized_joint_target.abs().numpy())
        self._data["joint_target_near_limit_fraction"].append(float(near_joint_target_limit.to(torch.float32).mean()))
        self._data["joint_target_limit_violation_fraction"].append(
            float(joint_target_limit_violation.to(torch.float32).mean())
        )
        self._data["forward_score"].append(reward_component("forward_score"))
        self._data["straight_score"].append(reward_component("straight_score"))
        self._data["posture_score"].append(reward_component("posture_score"))
        self._data["support_loss"].append(reward_component("support_loss"))
        self._data["straight_line_reward"].append(reward_component("reward"))
        self._data["running_reward_clipped"].append(selected_reward_clipped)
        self._data["foot_heights"].append(foot_clearance_component("foot_height"))
        self._data["foot_clearance_targets"].append(foot_clearance_component("target_height"))
        self._data["foot_clearance_shortfalls"].append(foot_clearance_component("shortfall"))
        self._data["foot_clearance_swing_weights"].append(foot_clearance_component("swing_weight"))
        self._data["foot_clearance_penalty"].append(foot_clearance_component("penalty", per_foot=False))
        if self._max_samples is not None and len(self._data["time_steps"]) >= self._max_samples:
            self._env._capture_rollout_diagnostics = False

    def save(self) -> list[Path]:
        """Save sampled arrays and diagnostic figures.

        Returns:
            Paths of the files that were saved. The list is empty when no samples were recorded.
        """
        self._env._capture_rollout_diagnostics = False
        if not self._data["time_steps"]:
            print("[symm_locomotion] No rollout samples were collected; skipping plots.", flush=True)
            return []

        self._output_dir.mkdir(parents=True, exist_ok=True)
        data = {name: np.asarray(values) for name, values in self._data.items()}
        data["joint_names"] = np.asarray(self._joint_names)
        data["leg_names"] = np.asarray(self._LEG_NAMES)
        data["motor_role_names"] = np.asarray(self._MOTOR_ROLE_NAMES)
        data["leg_joint_torques"] = data["joint_torques"].reshape(
            -1,
            len(self._LEG_NAMES),
            self._JOINTS_PER_LEG,
        )
        data["leg_joint_powers"] = data["joint_powers"].reshape(
            -1,
            len(self._LEG_NAMES),
            self._JOINTS_PER_LEG,
        )
        data["leg_torque_sums"] = data["leg_joint_torques"].sum(axis=-1)
        data["leg_joint_torque_magnitudes"] = np.abs(data["leg_joint_torques"])
        data["leg_torque_magnitude_sums"] = data["leg_joint_torque_magnitudes"].sum(axis=-1)
        data["leg_power_sums"] = data["leg_joint_powers"].sum(axis=-1)
        data["leg_joint_power_magnitudes"] = np.abs(data["leg_joint_powers"])
        data["leg_power_magnitude_sums"] = data["leg_joint_power_magnitudes"].sum(axis=-1)
        data["foot_ground_reaction_force_abs_components"] = np.abs(data["foot_ground_reaction_forces_w"])
        data["foot_ground_reaction_force_abs_sums"] = data["foot_ground_reaction_force_abs_components"].sum(axis=-1)
        data.update(self._velocity_tracking_error_data(data))
        smoothing_half_window_samples = round(0.5 * self._USAGE_SMOOTHING_WINDOW_S / self._step_dt)
        smoothing_window_samples = 2 * smoothing_half_window_samples + 1
        data["usage_plot_smoothing_window_s"] = np.asarray(self._USAGE_SMOOTHING_WINDOW_S)
        data["usage_plot_smoothing_window_samples"] = np.asarray(smoothing_window_samples)
        for key in (
            "leg_joint_torque_magnitudes",
            "leg_torque_magnitude_sums",
            "leg_joint_power_magnitudes",
            "leg_power_magnitude_sums",
            "foot_ground_reaction_force_abs_components",
            "foot_ground_reaction_force_abs_sums",
        ):
            data[f"{key}_centered_moving_mean"] = _centered_moving_mean(
                data[key],
                smoothing_window_samples,
                episode_ends=data["episode_done"],
            )
        data_path = self._output_dir / "sim_data.npz"
        np.savez_compressed(data_path, **data)
        tracking_error_path = self._save_velocity_tracking_error_summary(data)
        self._print_velocity_tracking_error_summary(data)

        paths = [
            data_path,
            tracking_error_path,
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
            self._save_policy_and_joint_limit_plot(data),
            self._save_reward_diagnostics_plot(data),
            self._save_foot_clearance_plot(data),
            self._save_per_leg_motor_plot(
                data,
                measurement_key="leg_joint_torque_magnitudes",
                sum_key="leg_torque_magnitude_sums",
                sum_label="Sum |torque|",
                filename="figure8_leg_motor_torques.png",
                measurement_name="Absolute Applied Joint Torque",
                measurement_unit="N m",
            ),
            self._save_per_leg_motor_plot(
                data,
                measurement_key="leg_joint_power_magnitudes",
                sum_key="leg_power_magnitude_sums",
                sum_label="Sum |power|",
                filename="figure9_leg_motor_powers.png",
                measurement_name="Absolute Joint Mechanical Power",
                measurement_unit="W",
            ),
            self._save_leg_ground_reaction_force_plot(data),
        ]
        print(f"[symm_locomotion] Saved rollout plots to: {self._output_dir}", flush=True)
        return paths

    def _velocity_tracking_error_data(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return velocity tracking error arrays derived from sampled command and measured velocity."""
        signed_error = data["true_lin_vel"] - data["desired_lin_vel"]
        abs_error = np.abs(signed_error)
        squared_error = np.square(signed_error)
        return {
            "velocity_tracking_signed_error": signed_error,
            "velocity_tracking_abs_error": abs_error,
            "velocity_tracking_squared_error": squared_error,
            "velocity_tracking_components": np.asarray(self._TRACKING_ERROR_COMPONENTS),
            "velocity_tracking_direction_command_threshold": np.asarray(self._TRACKING_DIRECTION_COMMAND_THRESHOLD),
        }

    def _print_velocity_tracking_error_summary(self, data: dict[str, np.ndarray]) -> None:
        """Print component-wise and direction-bucket velocity tracking errors."""
        desired_velocity = data["desired_lin_vel"]
        true_velocity = data["true_lin_vel"]
        print("[symm_locomotion] Velocity tracking error summary:", flush=True)
        for component_name, unit, mean_abs_error, rms_error, mean_signed_error in self._tracking_component_rows(data):
            print(
                "[symm_locomotion] "
                f"  {component_name:<5} all samples: "
                f"MAE={mean_abs_error:.3f} {unit}, "
                f"RMSE={rms_error:.3f} {unit}, "
                f"bias={mean_signed_error:+.3f} {unit}",
                flush=True,
            )

        threshold = self._TRACKING_DIRECTION_COMMAND_THRESHOLD
        for direction_name, component_index, sign, unit in self._TRACKING_DIRECTION_SPECS:
            signed_command = sign * desired_velocity[:, component_index]
            mask = signed_command > threshold
            if not np.any(mask):
                print(
                    f"[symm_locomotion]   {direction_name:<9}: no samples with command > {threshold:.2f} {unit}",
                    flush=True,
                )
                continue
            direction_command = signed_command[mask]
            direction_velocity = sign * true_velocity[mask, component_index]
            direction_error = direction_velocity - direction_command
            mean_abs_error = float(np.mean(np.abs(direction_error)))
            rms_error = float(np.sqrt(np.mean(np.square(direction_error))))
            mean_command = float(np.mean(direction_command))
            mean_velocity = float(np.mean(direction_velocity))
            print(
                "[symm_locomotion] "
                f"  {direction_name:<9}: "
                f"n={int(np.count_nonzero(mask))}, "
                f"cmd={mean_command:.3f} {unit}, "
                f"vel={mean_velocity:.3f} {unit}, "
                f"MAE={mean_abs_error:.3f} {unit}, "
                f"RMSE={rms_error:.3f} {unit}",
                flush=True,
            )

    def _tracking_component_rows(self, data: dict[str, np.ndarray]) -> list[tuple[str, str, float, float, float]]:
        """Return all-sample velocity tracking error rows for x, y, and yaw."""
        signed_error = data["velocity_tracking_signed_error"]
        abs_error = data["velocity_tracking_abs_error"]
        squared_error = data["velocity_tracking_squared_error"]
        rows = []
        for component_index, component_name in enumerate(self._TRACKING_ERROR_COMPONENTS):
            unit = "rad/s" if component_name == "yaw" else "m/s"
            rows.append(
                (
                    component_name,
                    unit,
                    float(np.mean(abs_error[:, component_index])),
                    float(np.sqrt(np.mean(squared_error[:, component_index]))),
                    float(np.mean(signed_error[:, component_index])),
                )
            )
        return rows

    def _save_velocity_tracking_error_summary(self, data: dict[str, np.ndarray]) -> Path:
        """Save mean absolute velocity tracking errors for x, y, and yaw to a text file."""
        path = self._output_dir / "tracking_errors.txt"
        lines = [
            "Velocity tracking errors",
            "Computed as mean(abs(actual - desired)) across all recorded timesteps.",
            "",
            "component,mean_absolute_error,unit",
        ]
        for component_name, unit, mean_abs_error, _, _ in self._tracking_component_rows(data):
            lines.append(f"{component_name},{mean_abs_error:.6f},{unit}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

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

    def _save_policy_and_joint_limit_plot(self, data: dict[str, np.ndarray]) -> Path:
        time_steps = data["time_steps"]
        raw_action_abs = np.abs(data["raw_actions"])
        actor_mean_abs = np.abs(data["actor_means"])
        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
        axes[0].plot(time_steps, raw_action_abs.mean(axis=1), label="Mean |raw action|")
        axes[0].plot(time_steps, actor_mean_abs.mean(axis=1), linestyle="--", label="Mean |actor mean|")
        axes[0].set_ylabel("Action (normalized)")
        axes[0].set_title("Policy Action Magnitude")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time_steps, raw_action_abs.max(axis=1), label="Max |raw action|")
        axes[1].plot(time_steps, actor_mean_abs.max(axis=1), linestyle="--", label="Max |actor mean|")
        axes[1].set_ylabel("Action (normalized)")
        axes[1].set_title("Peak Policy Action Magnitude")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(time_steps, data["joint_near_limit_fraction"], label="Within 5% of soft limit")
        axes[2].plot(time_steps, data["joint_limit_violation_fraction"], linestyle="--", label="Beyond soft limit")
        axes[2].plot(
            time_steps,
            data["joint_target_near_limit_fraction"],
            label="Target within 5% of soft limit",
        )
        axes[2].plot(
            time_steps,
            data["joint_target_limit_violation_fraction"],
            linestyle="--",
            label="Target beyond soft limit",
        )
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Joint fraction")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_title("Joint Soft-Limit Proximity")
        axes[2].legend()
        axes[2].grid(True)
        fig.tight_layout()
        path = self._output_dir / "figure5_policy_actions_and_joint_limits.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_reward_diagnostics_plot(self, data: dict[str, np.ndarray]) -> Path:
        time_steps = data["time_steps"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(time_steps, data["forward_score"], label="Forward score")
        axes[0].plot(time_steps, data["straight_score"], label="Straight score")
        axes[0].plot(time_steps, data["posture_score"], label="Posture score")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title("Straight-Line Reward Components")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time_steps, data["straight_line_reward"], label="Combined reward")
        axes[1].plot(time_steps, data["support_loss"], linestyle="--", label="Support loss")
        clipped_time_steps = time_steps[data["running_reward_clipped"].astype(bool)]
        if clipped_time_steps.size:
            axes[1].scatter(
                clipped_time_steps,
                np.zeros_like(clipped_time_steps),
                marker="x",
                color="tab:red",
                label="Negative running reward",
            )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Reward / loss")
        axes[1].set_title("Support Loss and Signed Running Reward")
        axes[1].legend()
        axes[1].grid(True)
        fig.tight_layout()
        path = self._output_dir / "figure6_straight_line_reward_diagnostics.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_foot_clearance_plot(self, data: dict[str, np.ndarray]) -> Path:
        time_steps = data["time_steps"]
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
        for index, axis in enumerate(axes):
            axis.plot(time_steps, data["foot_heights"][:, index], label="Measured height")
            axis.plot(
                time_steps,
                data["foot_clearance_targets"][:, index],
                linestyle="--",
                label="Swing target",
            )
            axis.fill_between(
                time_steps,
                0.0,
                data["foot_clearance_swing_weights"][:, index] * np.nanmax(data["foot_clearance_targets"]),
                alpha=0.15,
                label="Active swing weight",
            )
            axis.set_ylabel("Height (m)")
            axis.set_title(f"{self._LEG_NAMES[index]} Foot Clearance")
            axis.legend()
            axis.grid(True)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        path = self._output_dir / "figure7_foot_clearance.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_per_leg_motor_plot(
        self,
        data: dict[str, np.ndarray],
        *,
        measurement_key: str,
        sum_key: str,
        sum_label: str,
        filename: str,
        measurement_name: str,
        measurement_unit: str,
    ) -> Path:
        """Save one leg-major motor measurement plot.

        Args:
            data: Recorded rollout arrays.
            measurement_key: Key for data shaped as samples by legs by motors.
            sum_key: Key for the aggregate per-leg trace.
            sum_label: Human-readable label for the aggregate trace.
            filename: Output image filename.
            measurement_name: Human-readable measurement name.
            measurement_unit: SI unit shown on the vertical axes.

        Returns:
            Path to the saved figure.
        """
        time_steps = data["time_steps"]
        measurements = data[measurement_key]
        leg_sums = data[sum_key]
        smoothed_measurements = data[f"{measurement_key}_centered_moving_mean"]
        smoothed_leg_sums = data[f"{sum_key}_centered_moving_mean"]
        smoothing_window_s = float(data["usage_plot_smoothing_window_s"])
        joint_names = data["joint_names"]
        colors = ("tab:blue", "tab:orange", "tab:green")
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True, sharey=True)
        for leg_index, axis in enumerate(axes):
            first_joint_index = leg_index * self._JOINTS_PER_LEG
            for motor_index, (role_name, color) in enumerate(zip(self._MOTOR_ROLE_NAMES, colors, strict=True)):
                joint_name = str(joint_names[first_joint_index + motor_index])
                axis.plot(
                    time_steps,
                    measurements[:, leg_index, motor_index],
                    color=color,
                    linewidth=0.8,
                    alpha=0.30,
                    label=f"|{role_name}| raw ({joint_name})",
                )
                axis.plot(
                    time_steps,
                    smoothed_measurements[:, leg_index, motor_index],
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    label=f"|{role_name}| {smoothing_window_s:g} s mean",
                )
            axis.plot(
                time_steps,
                leg_sums[:, leg_index],
                color="black",
                linestyle=":",
                linewidth=1.0,
                alpha=0.40,
                label=f"{sum_label} raw",
            )
            axis.plot(
                time_steps,
                smoothed_leg_sums[:, leg_index],
                color="black",
                linestyle="--",
                linewidth=2.0,
                label=f"{sum_label} {smoothing_window_s:g} s mean",
            )
            axis.set_title(f"{self._LEG_NAMES[leg_index]} Leg")
            axis.set_ylabel(f"{measurement_name} ({measurement_unit})")
            axis.set_ylim(bottom=0.0)
            axis.legend(loc="upper right", ncol=2, fontsize="small")
            axis.grid(True)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{measurement_name} by Leg (FL, FR, RL, RR)")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        path = self._output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _save_leg_ground_reaction_force_plot(self, data: dict[str, np.ndarray]) -> Path:
        """Save world-frame ground-reaction-force components for every leg."""
        time_steps = data["time_steps"]
        force_components = data["foot_ground_reaction_force_abs_components"]
        force_abs_sums = data["foot_ground_reaction_force_abs_sums"]
        smoothed_force_components = data["foot_ground_reaction_force_abs_components_centered_moving_mean"]
        smoothed_force_abs_sums = data["foot_ground_reaction_force_abs_sums_centered_moving_mean"]
        smoothing_window_s = float(data["usage_plot_smoothing_window_s"])
        includes_friction = bool(np.all(data["ground_reaction_force_includes_friction"]))
        component_specs = ((0, "Fx", "tab:blue"), (1, "Fy", "tab:orange"), (2, "Fz", "tab:green"))
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True, sharey=True)
        for leg_index, axis in enumerate(axes):
            for component_index, component_name, color in component_specs:
                axis.plot(
                    time_steps,
                    force_components[:, leg_index, component_index],
                    color=color,
                    linewidth=0.8,
                    alpha=0.30,
                    label=f"|{component_name}| raw",
                )
                axis.plot(
                    time_steps,
                    smoothed_force_components[:, leg_index, component_index],
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    label=f"|{component_name}| {smoothing_window_s:g} s mean",
                )
            axis.plot(
                time_steps,
                force_abs_sums[:, leg_index],
                color="black",
                linestyle=":",
                linewidth=1.0,
                alpha=0.40,
                label="Sum |F| raw",
            )
            axis.plot(
                time_steps,
                smoothed_force_abs_sums[:, leg_index],
                color="black",
                linestyle="--",
                linewidth=2.0,
                label=f"Sum |F| {smoothing_window_s:g} s mean",
            )
            axis.set_title(f"{self._LEG_NAMES[leg_index]} Foot")
            axis.set_ylabel("Absolute Ground Reaction Force (N)" if includes_friction else "Absolute Contact Force (N)")
            axis.set_ylim(bottom=0.0)
            axis.legend(loc="upper right", ncol=2, fontsize="small")
            axis.grid(True)
        axes[-1].set_xlabel("Time (s)")
        title = (
            "Absolute World-Frame GRF Components and L1 Sum (Normal + Friction)"
            if includes_friction
            else "Absolute World-Frame Normal Contact Components and L1 Sum"
        )
        fig.suptitle(f"{title} (FL, FR, RL, RR)")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        path = self._output_dir / "figure10_leg_ground_reaction_forces.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
