# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local environment behavior for the Dobot X1 symmetric task."""

from __future__ import annotations

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.env import Go2SymmManagerBasedRLEnv


class DobotX1SymmManagerBasedRLEnv(Go2SymmManagerBasedRLEnv):
    """Manager-based RL environment with the Dobot X1 symmetric reward/update ordering."""

    pass
