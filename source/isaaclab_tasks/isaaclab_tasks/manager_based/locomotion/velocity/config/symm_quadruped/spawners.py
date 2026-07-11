# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared spawners for symmetric quadruped URDF tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg


def spawn_nested_urdf_with_contact_sensors(
    prim_path: str,
    cfg: UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a nested URDF and enable contact reporting on all rigid-body descendants."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf  # noqa: PLC0415
    from isaaclab.sim.utils import find_matching_prims  # noqa: PLC0415

    activate_contact_sensors = cfg.activate_contact_sensors
    cfg.activate_contact_sensors = False
    prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    cfg.activate_contact_sensors = activate_contact_sensors
    if activate_contact_sensors:
        robot_prims = find_matching_prims(prim_path) or [prim]
        for robot_prim in robot_prims:
            _activate_nested_rigid_body_contact_sensors(robot_prim, threshold=0.0)
    return prim


def _activate_nested_rigid_body_contact_sensors(prim: Usd.Prim, threshold: float = 0.0) -> None:
    """Activate contact reports on every rigid-body descendant in the imported tree."""
    from pxr import UsdPhysics  # noqa: PLC0415

    from isaaclab.sim.utils import safe_set_attribute_on_usd_prim  # noqa: PLC0415

    all_prims = [prim]
    while all_prims:
        child_prim = all_prims.pop(0)
        if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            applied_schemas = child_prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied_schemas:
                child_prim.AddAppliedSchema("PhysxRigidBodyAPI")
            if "PhysxContactReportAPI" not in applied_schemas:
                child_prim.AddAppliedSchema("PhysxContactReportAPI")
            safe_set_attribute_on_usd_prim(child_prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
            safe_set_attribute_on_usd_prim(child_prim, "physxContactReport:threshold", threshold, camel_case=False)
        all_prims += child_prim.GetChildren()
