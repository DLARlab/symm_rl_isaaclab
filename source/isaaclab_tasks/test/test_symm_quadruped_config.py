# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared symmetric quadruped task configuration."""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.flat_env_cfg import (
    configure_domain_randomization,
)


def test_domain_randomization_resets_with_zero_lateral_and_yaw_velocity():
    events = SimpleNamespace(
        physics_material=SimpleNamespace(params={}),
        add_base_mass=SimpleNamespace(params={}),
        base_external_force_torque=SimpleNamespace(params={}),
        base_com=object(),
        reset_base=SimpleNamespace(params={}),
        push_robot=SimpleNamespace(params={}),
    )
    env_cfg = SimpleNamespace(events=events)

    configure_domain_randomization(env_cfg)

    assert env_cfg.events.reset_base.params["velocity_range"]["y"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.events.push_robot.params["velocity_range"]["y"] == (0.0, 0.0)
