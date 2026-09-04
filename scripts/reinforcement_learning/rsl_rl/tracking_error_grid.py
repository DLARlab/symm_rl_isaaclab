# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched velocity-command tracking-error evaluation."""

from __future__ import annotations

import csv
import types
from pathlib import Path

import torch


TRACKING_ERROR_COMMAND_TESTS = (
    *(
        ("x", velocity, 0.0, 0.0)
        for velocity in (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    ),
    *(("y", 0.0, velocity, 0.0) for velocity in (-0.5, 0.0, 0.5)),
    *(("yaw", 0.0, 0.0, velocity) for velocity in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)),
)
"""Ordered command tests as ``(sweep, x, y, yaw)`` rows."""


class TrackingErrorGridEvaluator:
    """Assign fixed command groups and aggregate their tracking errors."""

    _CSV_FIELDNAMES = (
        "sweep",
        "command_x_mps",
        "command_y_mps",
        "command_yaw_radps",
        "tracking_error_x_mae_mps",
        "tracking_error_y_mae_mps",
        "tracking_error_xy_mae_mps",
        "tracking_error_yaw_mae_radps",
    )

    def __init__(
        self,
        env,
        *,
        envs_per_command: int,
        warmup_steps: int,
        measurement_steps: int,
        output_path: str | Path,
    ) -> None:
        """Initialize the evaluator.

        Args:
            env: Unwrapped symmetric locomotion environment.
            envs_per_command: Number of parallel environments assigned to each command.
            warmup_steps: Number of initial environment steps excluded from error statistics.
            measurement_steps: Number of environment steps included in error statistics.
            output_path: Destination CSV path.
        """
        if envs_per_command <= 0:
            raise ValueError(f"Environments per command must be positive, received {envs_per_command}.")
        if warmup_steps < 0:
            raise ValueError(f"Warmup steps cannot be negative, received {warmup_steps}.")
        if measurement_steps <= 0:
            raise ValueError(f"Measurement steps must be positive, received {measurement_steps}.")

        self._env = env
        self._envs_per_command = envs_per_command
        self._warmup_steps = warmup_steps
        self._measurement_steps = measurement_steps
        self._output_path = Path(output_path)
        self._command_count = len(TRACKING_ERROR_COMMAND_TESTS)
        self._num_envs = self._command_count * envs_per_command
        if env.num_envs != self._num_envs:
            raise ValueError(
                f"Tracking-error grid requires {self._num_envs} environments "
                f"({self._command_count} commands x {envs_per_command}), received {env.num_envs}."
            )

        self._command_term = env.command_manager.get_term("base_velocity")
        if not hasattr(self._command_term, "vel_command_b"):
            raise ValueError("Tracking-error grid requires a base-velocity command with a 'vel_command_b' buffer.")
        self._robot = self._command_term.robot
        command_rows = torch.tensor(
            [test[1:] for test in TRACKING_ERROR_COMMAND_TESTS],
            dtype=self._command_term.vel_command_b.dtype,
            device=self._command_term.vel_command_b.device,
        )
        self._commands = torch.repeat_interleave(command_rows, envs_per_command, dim=0)
        self._error_sums = torch.zeros((self._command_count, 4), dtype=torch.float64, device=env.device)
        self._sample_counts = torch.zeros(self._command_count, dtype=torch.float64, device=env.device)
        self._step_count = 0
        self._override_installed = False

    @property
    def total_steps(self) -> int:
        """Total number of warmup and measurement steps."""
        return self._warmup_steps + self._measurement_steps

    @property
    def is_complete(self) -> bool:
        """Whether all requested measurement steps have completed."""
        return self._step_count >= self.total_steps

    def install_command_override(self) -> None:
        """Keep each environment's assigned command fixed across resampling and resets."""
        if self._override_installed:
            return

        def _resample_fixed_commands(_command_term, env_ids) -> None:
            self._assign_commands(env_ids)

        self._command_term._resample_command = types.MethodType(  # noqa: SLF001
            _resample_fixed_commands, self._command_term
        )
        self._assign_commands(slice(None))
        self._override_installed = True

    def record(self, dones: torch.Tensor) -> None:
        """Accumulate one simulation step after the configured warmup period."""
        self._step_count += 1
        if self._step_count <= self._warmup_steps or self._step_count > self.total_steps:
            return
        if dones.numel() != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} done flags, received {dones.numel()}.")

        measured_velocity = torch.stack(
            (
                self._robot.data.root_lin_vel_b.torch[:, 0],
                self._robot.data.root_lin_vel_b.torch[:, 1],
                self._robot.data.root_ang_vel_b.torch[:, 2],
            ),
            dim=-1,
        )
        error = measured_velocity - self._commands
        absolute_error = torch.abs(error)
        error_metrics = torch.stack(
            (
                absolute_error[:, 0],
                absolute_error[:, 1],
                torch.linalg.norm(error[:, :2], dim=-1),
                absolute_error[:, 2],
            ),
            dim=-1,
        ).reshape(self._command_count, self._envs_per_command, 4)
        valid = (~dones.to(dtype=torch.bool)).reshape(self._command_count, self._envs_per_command)
        self._error_sums += torch.sum(error_metrics * valid.unsqueeze(-1), dim=1, dtype=torch.float64)
        self._sample_counts += torch.sum(valid, dim=1, dtype=torch.float64)

    def write_csv(self) -> Path:
        """Write the completed tracking-error table."""
        if not self.is_complete:
            raise RuntimeError(f"Cannot write tracking errors after only {self._step_count}/{self.total_steps} steps.")
        if bool(torch.any(self._sample_counts == 0)):
            raise RuntimeError("At least one command group has no valid measurement samples.")

        mean_errors = (self._error_sums / self._sample_counts.unsqueeze(-1)).cpu()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._CSV_FIELDNAMES)
            writer.writeheader()
            for test, errors in zip(TRACKING_ERROR_COMMAND_TESTS, mean_errors, strict=True):
                writer.writerow(
                    {
                        "sweep": test[0],
                        "command_x_mps": f"{test[1]:g}",
                        "command_y_mps": f"{test[2]:g}",
                        "command_yaw_radps": f"{test[3]:g}",
                        "tracking_error_x_mae_mps": f"{float(errors[0]):.6f}",
                        "tracking_error_y_mae_mps": f"{float(errors[1]):.6f}",
                        "tracking_error_xy_mae_mps": f"{float(errors[2]):.6f}",
                        "tracking_error_yaw_mae_radps": f"{float(errors[3]):.6f}",
                    }
                )
        return self._output_path.resolve()

    def _assign_commands(self, env_ids) -> None:
        """Assign configured commands and disable standing and heading overrides."""
        self._command_term.vel_command_b[env_ids] = self._commands[env_ids]
        if hasattr(self._command_term, "is_standing_env"):
            self._command_term.is_standing_env[env_ids] = False
        if hasattr(self._command_term, "is_heading_env"):
            self._command_term.is_heading_env[env_ids] = False
