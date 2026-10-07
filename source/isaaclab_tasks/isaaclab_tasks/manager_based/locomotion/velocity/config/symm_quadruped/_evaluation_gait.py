# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Fixed evaluation scenarios using the native xchen gait clock."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import GaitVelocityCommand

SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION = "time_reversal_closed_v2"

SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS = (
    (0.0, 0.5, 0.5, 0.0),
    (0.0, 0.0, 0.5, 0.5),
    (0.13, -0.13, 0.5, 0.5),
    (-0.13, 0.13, 0.5, 0.5),
    (0.0, 0.0, 0.63, 0.37),
    (0.0, 0.0, 0.37, 0.63),
    (-0.13, 0.13, 0.63, 0.37),
    (0.13, -0.13, 0.63, 0.37),
    (0.13, -0.13, 0.37, 0.63),
    (-0.13, 0.13, 0.37, 0.63),
)

SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS = (4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES = (
    "trot",
    "bound",
    "half_bound_front_a",
    "half_bound_front_b",
    "half_bound_hind_a",
    "half_bound_hind_b",
    "gallop_a",
    "gallop_b",
    "gallop_c",
    "gallop_d",
)

SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES = (
    "trot",
    "bound",
    "half_bound",
    "half_bound",
    "half_bound",
    "half_bound",
    "gallop",
    "gallop",
    "gallop",
    "gallop",
)

SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS = (0, 1, 3, 2, 5, 4, 8, 9, 6, 7)


class _EvaluationGaitVelocityCommand(GaitVelocityCommand):
    """Pin evaluation scenarios while retaining xchen's signed phase evolution."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._evaluation_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluation_velocities = torch.zeros(self.num_envs, device=self.device)
        self._evaluation_rows = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._evaluation_anchor_steps = torch.full(
            (self.num_envs,), int(env.common_step_counter), dtype=torch.long, device=self.device
        )

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if bool(torch.any((ids < 0) | (ids >= self.num_envs))):
            raise IndexError("Evaluation environment index is out of range.")
        return ids

    def set_evaluation_scenario(
        self,
        vx_mps: float,
        gait_index: int,
        env_ids: Sequence[int] | slice | None = None,
        *,
        deterministic_timing: bool = True,
    ) -> None:
        """Fix velocity [m/s] and a gait row until the next explicit scenario."""
        if not math.isfinite(vx_mps):
            raise ValueError("Evaluation velocity must be finite.")
        if isinstance(gait_index, bool) or not isinstance(gait_index, int):
            raise TypeError("Evaluation gait index must be an integer.")
        if not 0 <= gait_index < len(self.init_foot_thetas):
            raise IndexError("Evaluation gait index is out of range.")
        if not deterministic_timing:
            raise ValueError("Fixed evaluation scenarios require deterministic timing.")
        ids = self._resolve_env_ids(env_ids)
        self._evaluation_active[ids] = True
        self._evaluation_velocities[ids] = vx_mps
        self._evaluation_rows[ids] = gait_index
        self._apply_evaluation_scenario(ids)

    def _apply_evaluation_scenario(self, ids: torch.Tensor) -> None:
        ids = ids[self._evaluation_active[ids]]
        if not len(ids):
            return
        self.vel_command_b[ids] = 0.0
        self.vel_command_b[ids, 0] = self._evaluation_velocities[ids]
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False
        self._gait_foot_theta_templates[ids] = self.init_foot_thetas[self._evaluation_rows[ids]]
        self._resample_gait_timing(ids, transition=False)
        self.kappa[ids] = self.cfg.kappa
        self.time_left[ids] = torch.inf
        self.gait_time_left[ids] = torch.inf
        self._pending_transition_envs[ids] = False
        self._pending_velocity_resample_envs[ids] = False
        self._pending_gait_resample_envs[ids] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = self._resolve_env_ids(env_ids)
        self._evaluation_anchor_steps[ids] = int(self._env.common_step_counter)
        self._apply_evaluation_scenario(ids)
        return extras

    def common_gait_phases(self) -> torch.Tensor:
        """Return elapsed gait cycles, increasing for either velocity direction."""
        steps = self._env.common_step_counter - self._evaluation_anchor_steps
        return steps.to(self.gait_periods.dtype) * self._env.step_dt / self.gait_periods
