# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dobot X1 adapter for shared symmetric quadruped MDP terms."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped as _symm_quadruped
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DOBOT_X1_SYMM_LEGGED_GYM_LEG_JOINT_IDS = _symm_quadruped.SYMM_QUADRUPED_LEG_JOINT_IDS
"""Leg joint indices for the Dobot X1 logical FL, FR, RL, RR order."""

DOBOT_X1_SYMM_LEG_PHASE_INDEX = _symm_quadruped.SYMM_QUADRUPED_LEG_PHASE_INDEX
"""Gait phase indices for the Dobot X1 logical FL, FR, RL, RR order."""

DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
)
"""Per-leg signs mapping Dobot joints into the shared logical quadruped convention."""

DOBOT_X1_SYMM_JOINT_RANGES = (1.3264, 5.236, 5.06)
"""Dobot X1 hip, thigh, and calf joint ranges [rad]."""

_MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED = False


def leg_permutation_symmetry_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phase_sync_tolerance: float = 0.02,
) -> torch.Tensor:
    """Penalize Dobot joint differences under phase-aligned leg permutations.

    Args:
        env: The environment instance.
        command_name: Name of the gait command term.
        joint_cfg: Robot joints in FL, FR, RL, RR order.
        asset_cfg: Robot articulation.
        phase_sync_tolerance: Maximum circular foot-offset difference [cycles] for a pair to be synchronous.

    Returns:
        The negative leg-permutation symmetry penalty.
    """
    return _symm_quadruped.leg_permutation_symmetry_penalty(
        env,
        command_name=command_name,
        joint_cfg=joint_cfg,
        asset_cfg=asset_cfg,
        leg_joint_ids=DOBOT_X1_SYMM_LEGGED_GYM_LEG_JOINT_IDS,
        leg_phase_index=DOBOT_X1_SYMM_LEG_PHASE_INDEX,
        logical_joint_signs=DOBOT_X1_SYMM_LOGICAL_JOINT_SIGNS,
        joint_ranges=DOBOT_X1_SYMM_JOINT_RANGES,
        phase_sync_tolerance=phase_sync_tolerance,
    )


def morphological_symmetry_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phase_sync_tolerance: float = 0.02,
) -> torch.Tensor:
    """Call :func:`leg_permutation_symmetry_penalty` through its deprecated name.

    Args:
        env: The environment instance.
        command_name: Name of the gait command term.
        joint_cfg: Robot joints in FL, FR, RL, RR order.
        asset_cfg: Robot articulation.
        phase_sync_tolerance: Maximum circular foot-offset difference [cycles] for a pair to be synchronous.

    Returns:
        The negative leg-permutation symmetry penalty.
    """
    global _MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED
    if not _MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED:
        warnings.warn(
            "morphological_symmetry_penalty() is deprecated; use leg_permutation_symmetry_penalty().",
            DeprecationWarning,
            stacklevel=2,
        )
        _MORPHOLOGICAL_SYMMETRY_DEPRECATION_WARNED = True
    return leg_permutation_symmetry_penalty(
        env,
        command_name=command_name,
        joint_cfg=joint_cfg,
        asset_cfg=asset_cfg,
        phase_sync_tolerance=phase_sync_tolerance,
    )
