# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for RSL-RL play checkpoint selection."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging import version
from rsl_rl.algorithms import Distillation


def _load_checkpoint_helpers(script_name: str) -> SimpleNamespace:
    """Load the pure checkpoint helpers without launching the simulator script."""
    repository = Path(__file__).resolve().parents[3]
    module_path = repository / "scripts" / "reinforcement_learning" / "rsl_rl" / script_name
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    helper_names = {"_inference_load_cfg", "_load_runner_checkpoint"}
    helper_nodes = [node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    namespace = {"version": version}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(module_path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in helper_names})


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls = []

    def load(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.mark.parametrize("script_name", ("play.py", "play_rsl_rl.py"))
@pytest.mark.parametrize("load_critic", (False, True))
def test_current_on_policy_inference_loads_only_requested_networks(script_name, load_critic):
    helpers = _load_checkpoint_helpers(script_name)
    runner = _RecordingRunner()

    helpers._load_runner_checkpoint(
        runner,
        "model.pt",
        "OnPolicyRunner",
        "5.0.1",
        load_critic=load_critic,
    )

    assert runner.calls == [
        (
            ("model.pt",),
            {
                "load_cfg": {
                    "actor": True,
                    "critic": load_critic,
                    "optimizer": False,
                    "iteration": False,
                    "environment_iteration": True,
                    "rnd": False,
                    "augmentation": False,
                }
            },
        )
    ]


@pytest.mark.parametrize("script_name", ("play.py", "play_rsl_rl.py"))
def test_current_distillation_inference_loads_student_policy(script_name):
    helpers = _load_checkpoint_helpers(script_name)
    runner = _RecordingRunner()

    helpers._load_runner_checkpoint(runner, "model.pt", "DistillationRunner", "5.0.1")

    assert runner.calls == [
        (
            ("model.pt",),
            {
                "load_cfg": {
                    "student": True,
                    "teacher": False,
                    "optimizer": False,
                    "iteration": False,
                }
            },
        )
    ]


@pytest.mark.parametrize("script_name", ("play.py", "play_rsl_rl.py"))
@pytest.mark.parametrize("runner_class_name", ("OnPolicyRunner", "DistillationRunner"))
def test_pre_v4_checkpoint_loading_omits_unsupported_load_cfg(script_name, runner_class_name):
    helpers = _load_checkpoint_helpers(script_name)
    runner = _RecordingRunner()

    helpers._load_runner_checkpoint(runner, "model.pt", runner_class_name, "3.9.9")

    assert runner.calls == [(("model.pt",), {})]


@pytest.mark.parametrize("script_name", ("play.py", "play_rsl_rl.py"))
def test_custom_runner_retains_generic_checkpoint_loading(script_name):
    helpers = _load_checkpoint_helpers(script_name)
    runner = _RecordingRunner()

    helpers._load_runner_checkpoint(runner, "model.pt", "ProjectRunner", "5.0.1")

    assert runner.calls == [(("model.pt",), {})]


@pytest.mark.parametrize("script_name", ("play.py", "play_rsl_rl.py"))
def test_distillation_rejects_protocols_that_require_a_critic(script_name):
    helpers = _load_checkpoint_helpers(script_name)

    with pytest.raises(ValueError, match="do not provide the critic"):
        helpers._inference_load_cfg("DistillationRunner", "5.0.1", load_critic=True)


def test_distillation_load_cfg_restores_the_actual_student_network():
    helpers = _load_checkpoint_helpers("play.py")
    algorithm = Distillation.__new__(Distillation)
    algorithm.student = torch.nn.Linear(1, 1, bias=False)
    algorithm.teacher = torch.nn.Linear(1, 1, bias=False)
    algorithm.optimizer = torch.optim.SGD(algorithm.student.parameters(), lr=0.1)
    checkpoint_student = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        algorithm.student.weight.fill_(1.0)
        checkpoint_student.weight.fill_(7.0)
    checkpoint = {"student_state_dict": checkpoint_student.state_dict()}

    load_cfg = helpers._inference_load_cfg("DistillationRunner", "5.0.1")
    loaded_iteration = algorithm.load(checkpoint, load_cfg, strict=True)

    assert loaded_iteration is False
    assert algorithm.student.weight.item() == pytest.approx(7.0)
