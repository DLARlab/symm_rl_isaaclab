# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record parallel statistical trials without discarding environments or joining episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from evaluation_rollout_plotter import SymmetricRolloutPlotter
from symm_command_batch import SymmetricCommandBatchRecorder


class EvaluationBatchRecorder(SymmetricCommandBatchRecorder):
    """Capture all environments together and export the existing per-trial archive format."""

    def __init__(self, env, output_dir: Path, *, max_samples: int) -> None:
        """Reserve samples and validate physical metadata before the first step.

        Args:
            env: Unwrapped symmetric quadruped environment with pre-reset diagnostics.
            output_dir: Directory containing the evaluation study.
            max_samples: Maximum number of control steps, including settling.
        """
        # The modern plotter validates effort limits, mass, foot order, and ground filters.
        template = SymmetricRolloutPlotter(
            env,
            output_dir,
            max_samples=max_samples,
            require_configured_effort_limits=True,
            require_ground_filtered_forces=True,
        )
        self._env = env
        self._command_term = template._command_term
        self._robot = template._robot
        self._action_term = template._joint_action_term
        self._step_dt = float(env.step_dt)
        self._max_samples = max_samples
        self._commands = self._command_term.command.detach().clone()
        self._samples: dict[str, np.ndarray] = {}
        self._step_count = 0
        self.first_done_steps = np.zeros(env.num_envs, dtype=np.int64)
        self.reset_reference()

    @torch.no_grad()
    def record(
        self,
        actions: torch.Tensor | None = None,
        actor_means: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
        *,
        critic_values: torch.Tensor | None = None,
        pre_decision_common_gait_phases: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
    ) -> None:
        """Capture one sample per environment, retaining terminal states from the cache.

        Args:
            actions: Policy actions, shape [environment, action].
            actor_means: Policy means with the same shape as actions.
            dones: Episode-end flags, shape [environment].
            critic_values: Optional value predictions, one scalar per environment.
            pre_decision_common_gait_phases: Gait clock [cycle] before taking each action.
            rewards: Step rewards, one scalar per environment.
        """
        if self.is_complete:
            return
        if actions is None:
            actions = self._env.action_manager.action
        if dones is None:
            dones = torch.zeros(self._env.num_envs, dtype=torch.bool, device=actions.device)
        desired_headings = self._desired_headings + self._commands[:, 2].cpu().numpy() * self._step_dt
        super().record(actions, actor_means, dones)
        index = self._step_count - 1
        if not bool(self._cached("_last_foot_normal_force_is_ground_filtered").all()):
            raise RuntimeError("Parallel evaluation requires ground-filtered foot forces for every environment.")
        done = dones.detach().cpu().numpy().astype(bool)
        self.first_done_steps[(self.first_done_steps == 0) & done] = self._step_count

        def command_value(cache_name, name, *, width=None):
            value = getattr(self._env, cache_name, None)
            if value is None:
                value = getattr(self._command_term, name, None)
                if callable(value):
                    value = value()
            if value is None:
                shape = (self._env.num_envs,) if width is None else (self._env.num_envs, width)
                return np.full(shape, np.nan)
            return value

        targets = self._cached("_last_joint_position_targets")
        requested = getattr(self._env, "_last_requested_joint_position_targets", None)
        if requested is None:
            requested = getattr(self._env, "_requested_joint_position_targets", None)
        if requested is None:
            scale = getattr(self._action_term, "_scale", None)
            offset = getattr(self._action_term, "_offset", None)
            raw_actions = getattr(self._action_term, "raw_actions", actions)
            requested = (
                torch.full_like(targets, torch.nan) if scale is None or offset is None else raw_actions * scale + offset
            )
        clamped = getattr(self._env, "_joint_target_clipped_mask", None)
        if clamped is None:
            tolerance = 8.0 * torch.finfo(targets.dtype).eps * torch.maximum(torch.ones_like(targets), requested.abs())
            clamped = ((requested - targets).abs() > tolerance).float()
            clamped = torch.where(torch.isfinite(requested), clamped, torch.nan)
        headings = getattr(self._env, "_last_root_headings_w", None)
        common_phase = command_value("_last_common_gait_phases", "common_gait_phases")
        if pre_decision_common_gait_phases is not None and getattr(self._env, "_last_common_gait_phases", None) is None:
            # Automatic reset rewinds the command clock. Reconstruct the terminal clock
            # from the phase used for the action rather than archiving the next episode.
            common_phase = common_phase.clone()
            periods = command_value("_last_gait_periods", "gait_periods")
            common_phase[dones.bool()] = (
                pre_decision_common_gait_phases[dones.bool()] + self._step_dt / periods[dones.bool()]
            )
        values = {
            "base_headings": self._robot.data.heading_w.torch if headings is None else headings,
            "desired_headings": desired_headings,
            "heading_sample_valid": ~done if headings is None else np.ones_like(done),
            "foot_thetas": command_value("_last_foot_thetas", "foot_thetas", width=4),
            "gait_periods": command_value("_last_gait_periods", "gait_periods"),
            "duty_factors": command_value("_last_duty_factors", "duty_factors"),
            "common_gait_phases": common_phase,
            "pre_decision_common_gait_phases": pre_decision_common_gait_phases,
            "foot_normal_force_is_ground_filtered": self._cached("_last_foot_normal_force_is_ground_filtered"),
            "critic_values": critic_values,
            "rewards": rewards,
            "requested_joint_position_targets": requested,
            "joint_action_clamped": clamped,
            "action_clamp_fraction": clamped.float().mean(dim=-1),
        }
        for name, value in values.items():
            if value is None:
                array = np.full(self._env.num_envs, np.nan)
            else:
                array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if name in ("critic_values", "rewards", "gait_periods", "duty_factors"):
                array = array.reshape(self._env.num_envs)
            if name not in self._samples:
                self._samples[name] = np.empty((self.total_steps, *array.shape), dtype=array.dtype)
            self._samples[name][index] = array

    def trial_plotter(self, env_index: int, output_dir: Path) -> SymmetricRolloutPlotter:
        """Return a serializer containing only the selected trial's first episode.

        Args:
            env_index: Environment assigned to this trial.
            output_dir: Destination for this trial's archive.

        Returns:
            Plotter with samples truncated at the first episode end, including its terminal sample.
        """
        plotter = SymmetricRolloutPlotter(
            self._env,
            output_dir,
            env_index=env_index,
            max_samples=self.total_steps,
            require_configured_effort_limits=True,
            require_ground_filtered_forces=True,
        )
        count = int(self.first_done_steps[env_index]) or self._step_count
        plotter._data = {name: list(array[:count, env_index]) for name, array in self._samples.items()}
        plotter._data["time_steps"] = list(np.arange(count, dtype=np.float64) * self._step_dt)
        return plotter
