# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 environment compatibility aliases for the shared symmetric quadruped env."""

from __future__ import annotations

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.env import (
    SymmQuadrupedManagerBasedRLEnv,
)


class Go2SymmManagerBasedRLEnv(SymmQuadrupedManagerBasedRLEnv):
    """Manager-based RL environment for the Go2 symmetric task."""

    pass
