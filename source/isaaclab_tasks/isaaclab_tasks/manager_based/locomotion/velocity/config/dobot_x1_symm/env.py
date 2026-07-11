# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dobot X1 environment compatibility aliases for the shared symmetric quadruped env."""

from __future__ import annotations

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.env import (
    SymmQuadrupedManagerBasedRLEnv,
)


class DobotX1SymmManagerBasedRLEnv(SymmQuadrupedManagerBasedRLEnv):
    """Manager-based RL environment for the Dobot X1 symmetric task."""

    pass
