# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped time-reversal PPO."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

_PPO_MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "symm_quadruped"
    / "time_reversal_ppo.py"
)
_PPO_MODULE_SPEC = importlib.util.spec_from_file_location("_symm_time_reversal_ppo", _PPO_MODULE_PATH)
assert _PPO_MODULE_SPEC is not None and _PPO_MODULE_SPEC.loader is not None
_PPO_MODULE = importlib.util.module_from_spec(_PPO_MODULE_SPEC)
_PPO_MODULE_SPEC.loader.exec_module(_PPO_MODULE)
TimeReversalPPO = _PPO_MODULE.TimeReversalPPO
time_reversal_loss_scale = _PPO_MODULE.time_reversal_loss_scale
time_reversal_weighted_losses = _PPO_MODULE.time_reversal_weighted_losses


class _DummyDistribution:
    def __init__(self):
        self.std_param = torch.nn.Parameter(torch.tensor([-1.0, float("nan"), float("inf"), 5.0]))


class _DummyActor:
    def __init__(self):
        self.distribution = _DummyDistribution()


class _CountingActor(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.distribution = SimpleNamespace(std_param=None)
        self.is_recurrent = False
        self.events = events
        self.forward_calls = 0

    def forward(self, observations, **_kwargs):
        self.events.append("actor")
        self.forward_calls += 1
        batch_size = observations["policy"].shape[0]
        mean = self.weight.expand(batch_size, 12)
        self.output_distribution_params = (mean,)
        self.output_entropy = self.weight.expand(batch_size)
        return mean

    def get_output_log_prob(self, actions):
        return self.weight.expand(actions.shape[0])


class _CountingCritic(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.is_recurrent = False
        self.events = events
        self.forward_calls = 0

    def forward(self, observations, **_kwargs):
        self.events.append("critic")
        self.forward_calls += 1
        return self.weight.expand(observations["policy"].shape[0], 1)


class _TwoBatchStorage:
    def __init__(self, batch):
        self.batch = batch
        self.actions = torch.zeros(2, 12)
        self.distribution_params = (torch.zeros(2, 12),)
        self.cleared = False

    def mini_batch_generator(self, _num_mini_batches, _num_learning_epochs):
        return iter((self.batch, self.batch))

    def clear(self):
        self.cleared = True


def _schedule_algorithm(
    iteration: int,
    *,
    warmup_iterations: int = 0,
    rampup_iterations: int = 2000,
    ramp_shape: str = "linear",
    enabled: bool = True,
) -> TimeReversalPPO:
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.current_learning_iteration = iteration
    algorithm._time_reversal_update_count = iteration
    algorithm.symmetry = {
        "use_time_reversal_regularization": enabled,
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.20,
        "value_loss_coeff": 0.10,
        "warmup_iterations": warmup_iterations,
        "rampup_iterations": rampup_iterations,
        "ramp_shape": ramp_shape,
    }
    return algorithm


@pytest.mark.parametrize("ramp_shape", ["linear", "half_cosine"])
@pytest.mark.parametrize(("iteration", "expected"), [(499, 0.0), (500, 1.0), (501, 1.0)])
def test_legacy_hard_step_schedule(iteration, expected, ramp_shape):
    assert time_reversal_loss_scale(iteration, 500, 0, ramp_shape) == expected


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(0, 0.0), (500, 0.25), (1000, 0.5), (1500, 0.75), (2000, 1.0), (3000, 1.0)],
)
def test_immediate_linear_schedule(iteration, expected):
    assert time_reversal_loss_scale(iteration, 0, 2000, "linear") == expected


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(499, 0.0), (500, 0.0), (1250, 0.5), (2000, 1.0)],
)
def test_delayed_linear_schedule(iteration, expected):
    assert time_reversal_loss_scale(iteration, 500, 1500, "linear") == expected


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, 0.0),
        (500, 0.1464466094),
        (1000, 0.5),
        (1500, 0.8535533906),
        (2000, 1.0),
    ],
)
def test_half_cosine_schedule(iteration, expected):
    assert time_reversal_loss_scale(iteration, 0, 2000, "half_cosine") == pytest.approx(expected)


@pytest.mark.parametrize(
    ("warmup_iterations", "rampup_iterations"),
    [(-1, 0), (0, -1)],
)
def test_schedule_rejects_negative_iteration_counts(warmup_iterations, rampup_iterations):
    with pytest.raises(ValueError, match="must be a nonnegative integer"):
        time_reversal_loss_scale(0, warmup_iterations, rampup_iterations, "linear")


def test_schedule_rejects_unsupported_ramp_shape_even_for_hard_step():
    with pytest.raises(ValueError, match="ramp_shape"):
        time_reversal_loss_scale(500, 500, 0, "quadratic")


def test_actor_and_value_coefficients_share_scale_without_mutating_targets():
    algorithm = _schedule_algorithm(500)
    configured_targets = dict(algorithm.symmetry)

    scale, mirror_coeff, value_coeff = algorithm._effective_time_reversal_coefficients()

    assert scale == 0.25
    assert mirror_coeff == 0.05
    assert value_coeff == 0.025
    assert algorithm.symmetry == configured_targets


def test_effective_coefficients_cover_exact_linear_schedule_values():
    expected = {
        0: (0.0, 0.0, 0.0),
        500: (0.25, 0.05, 0.025),
        1000: (0.5, 0.10, 0.05),
        1500: (0.75, 0.15, 0.075),
        2000: (1.0, 0.20, 0.10),
        3000: (1.0, 0.20, 0.10),
    }

    for iteration, values in expected.items():
        assert _schedule_algorithm(iteration)._effective_time_reversal_coefficients() == pytest.approx(values)


def test_zero_scale_computes_schedule_once_and_skips_extra_time_reversal_forwards():
    events = []
    actor = _CountingActor(events)
    critic = _CountingCritic(events)
    transform_calls = 0

    def transform(**_kwargs):
        nonlocal transform_calls
        transform_calls += 1
        raise AssertionError("zero-scale update must not compute transformed observations")

    batch_size = 2
    batch = SimpleNamespace(
        observations=TensorDict({"policy": torch.zeros(batch_size, 72)}, batch_size=[batch_size]),
        actions=torch.zeros(batch_size, 12),
        old_actions_log_prob=torch.zeros(batch_size),
        values=torch.zeros(batch_size, 1),
        advantages=torch.ones(batch_size, 1),
        returns=torch.zeros(batch_size, 1),
        masks=None,
        hidden_states=(None, None),
        old_distribution_params=(),
    )
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry["data_augmentation_func"] = transform
    algorithm.symmetry["_env"] = None
    algorithm.actor = actor
    algorithm.critic = critic
    algorithm.storage = _TwoBatchStorage(batch)
    algorithm.optimizer = torch.optim.SGD((*actor.parameters(), *critic.parameters()), lr=0.01)
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.num_mini_batches = 2
    algorithm.num_learning_epochs = 1
    algorithm.normalize_advantage_per_mini_batch = False
    algorithm.desired_kl = None
    algorithm.schedule = "fixed"
    algorithm.use_clipped_value_loss = False
    algorithm.value_loss_coef = 1.0
    algorithm.entropy_coef = 0.0
    algorithm.clip_param = 0.2
    algorithm.max_grad_norm = 1.0
    algorithm.device = "cpu"
    algorithm.is_multi_gpu = False
    algorithm._actor_mean_abort_count = 0
    effective_coeff_calls = 0
    effective_coefficients = algorithm._effective_time_reversal_coefficients

    def counted_effective_coefficients():
        nonlocal effective_coeff_calls
        effective_coeff_calls += 1
        events.append("effective_coefficients")
        return effective_coefficients()

    algorithm._effective_time_reversal_coefficients = counted_effective_coefficients

    losses = algorithm.update()

    assert effective_coeff_calls == 1
    assert events[0] == "effective_coefficients"
    assert actor.forward_calls == 2
    assert critic.forward_calls == 2
    assert transform_calls == 0
    assert algorithm.storage.cleared
    assert losses["symmetry"] == 0.0
    assert losses["tr_value"] == 0.0
    assert losses["trs_scale"] == 0.0
    assert losses["effective_mirror_coeff"] == 0.0
    assert losses["effective_tr_value_coeff"] == 0.0
    assert losses["weighted_symmetry"] == 0.0
    assert losses["weighted_tr_value"] == 0.0
    assert losses["weighted_trs_total"] == 0.0


def test_zero_scale_has_exactly_zero_weighted_auxiliary_objective():
    scale, mirror_coeff, value_coeff = _schedule_algorithm(0)._effective_time_reversal_coefficients()

    weighted = time_reversal_weighted_losses(mirror_coeff, value_coeff, 123.0, 456.0)

    assert scale == 0.0
    assert mirror_coeff == 0.0
    assert value_coeff == 0.0
    assert weighted == (0.0, 0.0, 0.0)


def test_weighted_auxiliary_logs_use_mean_raw_losses():
    assert time_reversal_weighted_losses(0.05, 0.025, 2.0, 4.0) == pytest.approx((0.10, 0.10, 0.20))


def test_disabled_trs_has_zero_scale_coefficients_and_weighted_losses():
    scale, mirror_coeff, value_coeff = _schedule_algorithm(2000, enabled=False)._effective_time_reversal_coefficients()

    assert (scale, mirror_coeff, value_coeff) == (0.0, 0.0, 0.0)
    assert time_reversal_weighted_losses(mirror_coeff, value_coeff, 1.0, 1.0) == (0.0, 0.0, 0.0)


def test_resume_uses_update_after_last_completed_checkpoint_iteration(monkeypatch, tmp_path):
    monkeypatch.setattr(PPO, "save", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)

    checkpoint_path = tmp_path / "model_499.pt"
    save_runner = OnPolicyRunner.__new__(OnPolicyRunner)
    save_runner.alg = _schedule_algorithm(499, warmup_iterations=500, rampup_iterations=0)
    save_runner.current_learning_iteration = 499
    save_runner.logger = SimpleNamespace(save_model=lambda *_args, **_kwargs: None)
    save_runner.save(str(checkpoint_path))
    assert torch.load(checkpoint_path, weights_only=False)["iter"] == 499

    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    resume_runner = OnPolicyRunner.__new__(OnPolicyRunner)
    resume_runner.alg = algorithm
    resume_runner.current_learning_iteration = 0

    resume_runner.load(str(checkpoint_path))

    # RSL-RL saves the last completed index. TimeReversalPPO advances the
    # ephemeral loaded mapping before OnPolicyRunner restores its counter.
    assert resume_runner.current_learning_iteration == 500
    assert algorithm.current_learning_iteration == 500
    assert algorithm._time_reversal_update_count == 500
    assert algorithm._effective_time_reversal_coefficients() == (1.0, 0.20, 0.10)
    assert torch.load(checkpoint_path, weights_only=False)["iter"] == 499


def test_partial_checkpoint_load_does_not_advance_schedule_counter(monkeypatch):
    algorithm = _schedule_algorithm(123)
    checkpoint = {"iter": 499}
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: False)

    loaded_iteration = algorithm.load(checkpoint, load_cfg={"iteration": False}, strict=True)

    assert not loaded_iteration
    assert checkpoint["iter"] == 499
    assert algorithm.current_learning_iteration == 123
    assert algorithm._time_reversal_update_count == 123


@pytest.mark.parametrize(
    ("field", "value"),
    [("warmup_iterations", -1), ("rampup_iterations", -1), ("mirror_loss_coeff", float("nan"))],
)
def test_algorithm_runtime_validation_rejects_invalid_overrides(field, value):
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry[field] = value

    with pytest.raises(ValueError, match=field):
        algorithm._validate_time_reversal_configuration()


def test_zeroed_no_trs_configuration_disables_time_reversal_update_path():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "use_time_reversal_regularization": True,
        "use_data_augmentation": False,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "value_loss_coeff": 0.0,
    }

    assert not algorithm._time_reversal_enabled()


def test_clamp_actor_std_keeps_scalar_gaussian_std_positive_and_finite():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.actor = _DummyActor()

    algorithm._clamp_actor_std()

    std = algorithm.actor.distribution.std_param
    assert torch.all(torch.isfinite(std))
    assert torch.all(std >= algorithm._MIN_ACTOR_STD)
    assert torch.all(std <= algorithm._MAX_ACTOR_STD)
    assert algorithm._MAX_ACTOR_STD == 1.0


def test_action_diagnostics_use_exact_rollout_samples_and_distribution_means():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.storage = SimpleNamespace(
        actions=torch.tensor([[[-2.0, 1.0]], [[0.0, 1.0]]]),
        distribution_params=(torch.tensor([[[-0.5, 0.25]], [[0.0, 0.25]]]),),
    )

    diagnostics = algorithm._action_diagnostics_from_storage()

    assert diagnostics["diagnostics/action_abs_mean"] == pytest.approx(1.0)
    assert diagnostics["diagnostics/action_abs_max"] == pytest.approx(2.0)
    assert diagnostics["diagnostics/actor_mean_abs_mean"] == pytest.approx(0.25)
    assert diagnostics["diagnostics/actor_mean_abs_max"] == pytest.approx(0.5)


def test_actor_mean_bound_loss_only_penalizes_extreme_policy_means():
    actor_mean = torch.tensor([[0.0, 10.0, 12.0, -14.0]])

    loss = TimeReversalPPO._actor_mean_bound_loss(actor_mean)

    assert loss.item() == pytest.approx(5.0)
    assert TimeReversalPPO._ACTOR_MEAN_BOUND == 10.0
    assert TimeReversalPPO._ACTOR_MEAN_BOUND_LOSS_COEFF == 1.0e-2


def test_actor_mean_safety_aborts_sustained_divergence_and_recovers_after_safe_update():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm._actor_mean_abort_count = 0

    for _ in range(algorithm._ACTOR_MEAN_ABORT_PATIENCE - 1):
        algorithm._update_actor_mean_safety(algorithm._ACTOR_MEAN_ABORT_BOUND + 1.0)

    with pytest.raises(RuntimeError, match="actor mean diverged"):
        algorithm._update_actor_mean_safety(algorithm._ACTOR_MEAN_ABORT_BOUND + 1.0)

    algorithm._update_actor_mean_safety(0.0)
    assert algorithm._actor_mean_abort_count == 0


def test_time_reversal_mask_includes_zero_velocity_commands():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "command_observation_index": 9,
        "command_observation_scale": 2.0,
        "min_abs_command_velocity": 0.0,
    }
    observations = {"policy": torch.zeros(3, 72)}

    mask = algorithm._time_reversal_mask(observations)

    assert torch.equal(mask, torch.ones(3, 1))


def test_masked_mse_with_no_selected_commands_is_finite_zero():
    prediction = torch.full((2, 3), float("nan"), requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 1)

    loss = TimeReversalPPO._masked_mse(prediction, target, mask)
    loss.backward()

    assert math.isfinite(loss.item())
    assert loss.item() == 0.0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
