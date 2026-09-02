# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free compatibility tests for symmetric quadruped environment helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_joint_target_limit_margin():
    repository = Path(__file__).resolve().parents[3]
    module_path = (
        repository
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "symm_quadruped"
        / "env.py"
    )
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    helper_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_joint_target_limit_margin"
    )
    namespace = {}
    exec(compile(ast.Module(body=[helper_node], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["_joint_target_limit_margin"]


def _load_maximum_stance_foot_slip():
    repository = Path(__file__).resolve().parents[3]
    module_path = (
        repository
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "symm_quadruped"
        / "env.py"
    )
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    helper_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_maximum_stance_foot_slip"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[helper_node], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["_maximum_stance_foot_slip"]


def test_joint_target_limit_margin_uses_configured_reward_term():
    margin = _load_joint_target_limit_margin()
    reward_manager = SimpleNamespace(
        get_term_cfg=lambda _name: SimpleNamespace(params={"margin_fraction": 0.125}),
    )

    assert margin(reward_manager) == pytest.approx(0.125)


@pytest.mark.parametrize("error_type", (KeyError, ValueError))
def test_joint_target_limit_margin_falls_back_when_reward_term_is_absent(error_type):
    margin = _load_joint_target_limit_margin()

    def missing_term(_name):
        raise error_type("joint_target_limits")

    assert margin(SimpleNamespace(get_term_cfg=missing_term)) == pytest.approx(0.05)


def test_foot_slip_uses_only_feet_with_contact_force():
    maximum_slip = _load_maximum_stance_foot_slip()
    horizontal_velocity = torch.tensor(
        [
            [[3.0, 4.0], [0.3, 0.4], [9.0, 0.0]],
            [[8.0, 0.0], [6.0, 8.0], [1.0, 0.0]],
        ]
    )
    contact_force = torch.tensor([[0.0, 2.0, 0.5], [0.0, 0.0, 0.0]])

    slip = maximum_slip(horizontal_velocity, contact_force)

    torch.testing.assert_close(slip, torch.tensor([0.5, 0.0]))
