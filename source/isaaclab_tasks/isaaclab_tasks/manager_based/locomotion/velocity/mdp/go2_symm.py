# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 compatibility layer for shared symmetric quadruped MDP terms."""

from __future__ import annotations

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped as _symm_quadruped
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import *  # noqa: F401, F403

GO2_LEGGED_GYM_HIP_ACTION_IDS = _symm_quadruped.SYMM_QUADRUPED_HIP_ACTION_IDS
"""Hip action indices for the ported Go2 action order."""

GO2_LEGGED_GYM_LEG_JOINT_IDS = _symm_quadruped.SYMM_QUADRUPED_LEG_JOINT_IDS
"""Leg joint indices for the ported Go2 joint order."""
