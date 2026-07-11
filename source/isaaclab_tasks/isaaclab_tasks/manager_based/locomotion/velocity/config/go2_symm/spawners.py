# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local spawner aliases for the Go2 symmetric task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.spawners import (
    spawn_nested_urdf_with_contact_sensors,
)

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg


def spawn_go2_symm_from_urdf(
    prim_path: str,
    cfg: UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the Go2 URDF and enable contact reporting on nested links."""
    return spawn_nested_urdf_with_contact_sensors(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
