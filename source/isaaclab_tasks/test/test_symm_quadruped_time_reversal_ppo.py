# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped time-reversal PPO."""

from __future__ import annotations

import copy
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
feasible_actor_mean_penalty = _PPO_MODULE.feasible_actor_mean_penalty
normalize_requested_joint_targets = _PPO_MODULE.normalize_requested_joint_targets


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


class _LegacyRegressionActor(torch.nn.Module):
    """Small deterministic policy used to freeze the audited-base PPO update."""

    def __init__(self):
        super().__init__()
        self.log_prob_weight = torch.nn.Parameter(torch.tensor(0.2))
        self.log_prob_bias = torch.nn.Parameter(torch.tensor(-0.1))
        self.distribution = SimpleNamespace(std_param=None)
        self.is_recurrent = False

    def forward(self, observations, **_kwargs):
        feature = observations["policy"][:, 0]
        mean = (self.log_prob_weight * feature + self.log_prob_bias).unsqueeze(-1).expand(-1, 12)
        self.output_distribution_params = (mean,)
        self.output_entropy = (0.3 + 0.25 * self.log_prob_weight - 0.5 * self.log_prob_bias).expand(
            observations.batch_size[0]
        )
        return mean

    def get_output_log_prob(self, actions):
        return self.log_prob_weight * actions[:, 0] + self.log_prob_bias


class _LegacyRegressionCritic(torch.nn.Module):
    """Small deterministic value function used by the compatibility fixture."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))
        self.bias = torch.nn.Parameter(torch.tensor(-0.2))
        self.is_recurrent = False

    def forward(self, observations, **_kwargs):
        value = self.weight * observations["policy"][:, 1] + self.bias
        return value.unsqueeze(-1)


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


class _SingleBatchStorage:
    def __init__(self, batch, rollout_means):
        self.batch = batch
        self.actions = batch.actions.detach().clone()
        self.distribution_params = (rollout_means.detach().clone(),)
        self.cleared = False

    def mini_batch_generator(self, _num_mini_batches, _num_learning_epochs):
        return iter((self.batch,))

    def clear(self):
        self.cleared = True


class _TrackingRegressionCritic(_LegacyRegressionCritic):
    """Record critic inputs and autograd state without changing its predictions."""

    def __init__(self):
        super().__init__()
        self.forward_inputs = []
        self.forward_grad_enabled = []

    def forward(self, observations, **kwargs):
        self.forward_inputs.append(observations["policy"].detach().clone())
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        return super().forward(observations, **kwargs)


def _regression_time_reversal_transform(*, obs=None, actions=None, env=None):
    del env
    augmented_observations = None
    augmented_actions = None
    if obs is not None:
        transformed_policy = obs["policy"].detach().clone()
        transformed_policy[:, 1].neg_()
        augmented_observations = TensorDict(
            {"policy": torch.cat((obs["policy"], transformed_policy))},
            batch_size=[2 * obs.batch_size[0]],
        )
    if actions is not None:
        augmented_actions = torch.cat((actions, -actions))
    return augmented_observations, augmented_actions


def _run_regression_update(
    *,
    value_loss_coeff: float,
    use_tr_value_consistency: bool | None,
    log_disabled_raw_consistency: bool = False,
    mirror_loss_coeff: float = 0.0,
    use_tr_policy_consistency: bool | None = False,
):
    """Run one deterministic PPO update with independently controlled TR terms."""
    actor = _LegacyRegressionActor()
    critic = _TrackingRegressionCritic()
    observations = torch.zeros(4, 72)
    observations[:, 0] = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    observations[:, 1] = torch.tensor([0.5, -1.0, 2.0, -0.5])
    observations = TensorDict({"policy": observations}, batch_size=[4])
    actions = torch.zeros(4, 12)
    actions[:, 0] = torch.tensor([-0.5, 0.25, 1.0, -1.5])
    batch = SimpleNamespace(
        observations=observations,
        actions=actions,
        old_actions_log_prob=torch.tensor([-0.25, -0.05, 0.1, -0.2]),
        values=torch.tensor([[0.1], [-0.2], [0.4], [-0.1]]),
        advantages=torch.tensor([[1.2], [-0.7], [0.5], [-1.1]]),
        returns=torch.tensor([[0.4], [-0.3], [1.2], [-0.8]]),
        masks=None,
        hidden_states=(None, None),
        old_distribution_params=(),
    )
    with torch.no_grad():
        rollout_means = actor(observations)
    storage = _SingleBatchStorage(batch, rollout_means)
    parameters = tuple(actor.parameters()) + tuple(critic.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.03, momentum=0.9)

    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "use_time_reversal_regularization": True,
        "use_data_augmentation": False,
        "use_mirror_loss": mirror_loss_coeff > 0.0,
        "mirror_loss_coeff": mirror_loss_coeff,
        "value_loss_coeff": value_loss_coeff,
        "use_tr_policy_consistency": use_tr_policy_consistency,
        "use_tr_value_consistency": use_tr_value_consistency,
        "log_disabled_raw_consistency": log_disabled_raw_consistency,
        "warmup_iterations": 0,
        "rampup_iterations": 0,
        "ramp_shape": "linear",
        "tr_gradient_diagnostics": {"enabled": False},
        "tr_augmentation": {
            "enabled": False,
            "coefficient": 0.0,
            "schedule": {"enabled": False, "target_coeff": 0.0},
        },
        "data_augmentation_func": _regression_time_reversal_transform,
        "_env": None,
    }
    algorithm.actor = actor
    algorithm.critic = critic
    algorithm.storage = storage
    algorithm.optimizer = optimizer
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.num_mini_batches = 1
    algorithm.num_learning_epochs = 1
    algorithm.normalize_advantage_per_mini_batch = False
    algorithm.desired_kl = None
    algorithm.schedule = "fixed"
    algorithm.use_clipped_value_loss = True
    algorithm.value_loss_coef = 0.7
    algorithm.entropy_coef = 0.05
    algorithm.clip_param = 0.2
    algorithm.max_grad_norm = 100.0
    algorithm.device = "cpu"
    algorithm.is_multi_gpu = False
    algorithm._actor_mean_abort_count = 0
    algorithm._time_reversal_update_count = 0
    algorithm.current_learning_iteration = 0
    algorithm._tr_augmentation = None
    initial_actor = tuple(parameter.detach().clone() for parameter in actor.parameters())
    initial_critic = tuple(parameter.detach().clone() for parameter in critic.parameters())

    losses = algorithm.update()
    return SimpleNamespace(
        actor=actor,
        critic=critic,
        initial_actor=initial_actor,
        initial_critic=initial_critic,
        losses=losses,
        optimizer=optimizer,
        storage=storage,
    )


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


def test_zero_scale_computes_schedule_once_and_skips_extra_time_reversal_forwards(monkeypatch):
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

    def diagnostics_disabled(*_args, **_kwargs):
        raise AssertionError("disabled diagnostics must preserve the previous update path")

    monkeypatch.setattr(_PPO_MODULE, "time_reversal_gradient_diagnostics", diagnostics_disabled)

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


def test_actor_only_trs_does_not_evaluate_critic_on_transformed_observations():
    result = _run_regression_update(
        value_loss_coeff=0.0,
        use_tr_value_consistency=None,
        mirror_loss_coeff=0.1,
        use_tr_policy_consistency=None,
    )

    assert result.losses["effective_tr_policy_coeff"] == pytest.approx(0.1)
    assert result.losses["effective_tr_value_coeff"] == 0.0
    assert result.losses["tr_value_consistency"] == 0.0
    assert result.losses["weighted_tr_value"] == 0.0
    assert len(result.critic.forward_inputs) == 1
    assert result.critic.forward_grad_enabled == [True]


def test_disabled_value_diagnostic_is_detached_and_preserves_ppo_outputs():
    disabled = _run_regression_update(
        value_loss_coeff=0.0,
        use_tr_value_consistency=None,
        log_disabled_raw_consistency=False,
        mirror_loss_coeff=0.1,
        use_tr_policy_consistency=None,
    )
    diagnostic = _run_regression_update(
        value_loss_coeff=0.0,
        use_tr_value_consistency=None,
        log_disabled_raw_consistency=True,
        mirror_loss_coeff=0.1,
        use_tr_policy_consistency=None,
    )

    assert disabled.critic.forward_grad_enabled == [True]
    assert diagnostic.critic.forward_grad_enabled == [True, False]
    assert diagnostic.losses["tr_value_consistency"] == pytest.approx(0.495)
    assert diagnostic.losses["tr_value"] == diagnostic.losses["tr_value_consistency"]
    assert diagnostic.losses["effective_tr_value_coeff"] == 0.0
    assert diagnostic.losses["weighted_tr_value"] == 0.0
    for key in ("value", "surrogate", "entropy", "actor_bound", "weighted_tr_policy", "weighted_tr_total"):
        assert diagnostic.losses[key] == pytest.approx(disabled.losses[key], abs=1.0e-8)
    for disabled_parameter, diagnostic_parameter in zip(
        (*disabled.actor.parameters(), *disabled.critic.parameters()),
        (*diagnostic.actor.parameters(), *diagnostic.critic.parameters()),
    ):
        torch.testing.assert_close(diagnostic_parameter, disabled_parameter, rtol=0.0, atol=1.0e-8)
        torch.testing.assert_close(diagnostic_parameter.grad, disabled_parameter.grad, rtol=0.0, atol=1.0e-8)


def test_positive_value_coefficient_preserves_legacy_ablation_enablement():
    inferred = _run_regression_update(value_loss_coeff=0.05, use_tr_value_consistency=None)
    disabled = _run_regression_update(value_loss_coeff=0.05, use_tr_value_consistency=False)
    explicit = _run_regression_update(value_loss_coeff=0.05, use_tr_value_consistency=True)

    for enabled in (inferred, explicit):
        assert enabled.critic.forward_grad_enabled == [True, True]
        assert enabled.losses["effective_tr_value_coeff"] == pytest.approx(0.05)
        assert enabled.losses["tr_value_consistency"] == pytest.approx(0.495)
        assert enabled.losses["weighted_tr_value"] == pytest.approx(0.02475)
        assert enabled.losses["tr_value"] == enabled.losses["tr_value_consistency"]
        assert enabled.losses["effective_mirror_coeff"] == enabled.losses["effective_tr_policy_coeff"]
        assert enabled.losses["weighted_trs_total"] == enabled.losses["weighted_tr_total"]

    assert disabled.critic.forward_grad_enabled == [True]
    assert disabled.losses["effective_tr_value_coeff"] == 0.0
    assert disabled.losses["tr_value_consistency"] == 0.0
    assert disabled.losses["weighted_tr_value"] == 0.0
    torch.testing.assert_close(
        torch.stack(tuple(inferred.critic.parameters())),
        torch.stack(tuple(explicit.critic.parameters())),
        rtol=0.0,
        atol=1.0e-8,
    )
    # Frozen gradient result for (V(T(o)) - stop_gradient(V(o)))^2. If the
    # original value target were not detached, both the weight and bias
    # updates would differ from these values.
    torch.testing.assert_close(
        torch.stack(tuple(inferred.critic.parameters())),
        torch.tensor([0.3145875036716461, -0.18432500958442688]),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert not torch.equal(
        torch.stack(tuple(inferred.critic.parameters())),
        torch.stack(tuple(disabled.critic.parameters())),
    )


def test_deprecated_data_augmentation_still_duplicates_ppo_minibatches():
    events = []
    actor = _CountingActor(events)
    critic = _CountingCritic(events)
    batch_size = 2
    observations = TensorDict({"policy": torch.zeros(batch_size, 72)}, batch_size=[batch_size])
    batch = SimpleNamespace(
        observations=observations,
        actions=torch.zeros(batch_size, 12),
        old_actions_log_prob=torch.tensor([0.1, 0.2]),
        values=torch.tensor([[0.3], [0.4]]),
        advantages=torch.tensor([[0.5], [0.6]]),
        returns=torch.tensor([[0.7], [0.8]]),
        masks=None,
        hidden_states=(None, None),
        old_distribution_params=(),
    )
    transform_calls = []
    filtered_augmentation_batch_sizes = []

    class FakeFilteredAugmentation:
        def prepare_update(self):
            return {}

        def actor_nll(self, actor_module, original_batch_size):
            filtered_augmentation_batch_sizes.append(original_batch_size)
            return actor_module.weight * 0.0

        def finish_update(self):
            pass

    def duplicate_time_reversal(*, obs, actions, env):
        transform_calls.append((obs.batch_size[0], actions.shape[0], env))
        augmented_observations = TensorDict(
            {"policy": torch.cat((obs["policy"], obs["policy"] + 1.0))},
            batch_size=[2 * obs.batch_size[0]],
        )
        augmented_actions = torch.cat((actions, actions + 1.0))
        return augmented_observations, augmented_actions

    algorithm = _schedule_algorithm(1, rampup_iterations=0)
    algorithm.symmetry.update(
        {
            "use_time_reversal_regularization": True,
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "mirror_loss_coeff": 0.0,
            "value_loss_coeff": 0.0,
            "use_tr_policy_consistency": False,
            "use_tr_value_consistency": False,
            "tr_gradient_diagnostics": {"enabled": False},
            "tr_augmentation": {
                "enabled": True,
                "coefficient": 0.5,
                "schedule": {"enabled": True, "warmup_iterations": 0, "rampup_iterations": 0},
                "gradient_diagnostics_interval": 100,
            },
            "data_augmentation_func": duplicate_time_reversal,
            "_env": "environment",
        }
    )
    storage = _SingleBatchStorage(batch, torch.zeros(batch_size, 12))
    algorithm.actor = actor
    algorithm.critic = critic
    algorithm.storage = storage
    algorithm.optimizer = torch.optim.SGD((*actor.parameters(), *critic.parameters()), lr=0.01)
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.num_mini_batches = 1
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
    algorithm._tr_augmentation = FakeFilteredAugmentation()

    algorithm.update()

    assert transform_calls == [(2, 2, "environment")]
    assert filtered_augmentation_batch_sizes == [2]
    assert batch.observations.batch_size[0] == 4
    assert batch.actions.shape == (4, 12)
    assert batch.old_actions_log_prob.tolist() == pytest.approx([0.1, 0.2, 0.1, 0.2])
    assert batch.values.squeeze(-1).tolist() == pytest.approx([0.3, 0.4, 0.3, 0.4])
    assert batch.advantages.squeeze(-1).tolist() == pytest.approx([0.5, 0.6, 0.5, 0.6])
    assert batch.returns.squeeze(-1).tolist() == pytest.approx([0.7, 0.8, 0.7, 0.8])
    assert actor.forward_calls == 1
    assert critic.forward_calls == 1
    assert storage.cleared


def test_all_additive_options_disabled_preserve_frozen_legacy_ppo_update():
    """Freeze audited-base PPO losses, gradients, and optimizer behavior."""
    actor = _LegacyRegressionActor()
    critic = _LegacyRegressionCritic()
    observations = torch.zeros(4, 72)
    observations[:, 0] = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    observations[:, 1] = torch.tensor([0.5, -1.0, 2.0, -0.5])
    observations = TensorDict({"policy": observations}, batch_size=[4])
    actions = torch.zeros(4, 12)
    actions[:, 0] = torch.tensor([-0.5, 0.25, 1.0, -1.5])
    batch = SimpleNamespace(
        observations=observations,
        actions=actions,
        old_actions_log_prob=torch.tensor([-0.25, -0.05, 0.1, -0.2]),
        values=torch.tensor([[0.1], [-0.2], [0.4], [-0.1]]),
        advantages=torch.tensor([[1.2], [-0.7], [0.5], [-1.1]]),
        returns=torch.tensor([[0.4], [-0.3], [1.2], [-0.8]]),
        masks=None,
        hidden_states=(None, None),
        old_distribution_params=(),
    )

    def disabled_transform(**_kwargs):
        raise AssertionError("disabled additive paths must not transform PPO data")

    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        # The publication fields are additive. Explicitly disabling every new
        # path must leave the audited-base PPO update numerically unchanged.
        "use_time_reversal_regularization": False,
        "use_data_augmentation": False,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "value_loss_coeff": 0.0,
        "use_tr_policy_consistency": False,
        "use_tr_value_consistency": False,
        "log_disabled_raw_consistency": False,
        "tr_policy_schedule": {"enabled": False, "target_coeff": 0.0},
        "tr_value_schedule": {"enabled": False, "target_coeff": 0.0},
        "tr_gradient_diagnostics": {"enabled": False},
        "tr_augmentation": {
            "enabled": False,
            "coefficient": 0.0,
            "schedule": {"enabled": False, "target_coeff": 0.0},
        },
        "data_augmentation_func": disabled_transform,
        "_env": None,
    }
    with torch.no_grad():
        rollout_means = actor(observations)
    storage = _SingleBatchStorage(batch, rollout_means)
    parameters = tuple(actor.parameters()) + tuple(critic.parameters())
    initial_critic_parameters = tuple(parameter.detach().clone() for parameter in critic.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.03, momentum=0.9)
    algorithm.actor = actor
    algorithm.critic = critic
    algorithm.storage = storage
    algorithm.optimizer = optimizer
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.num_mini_batches = 1
    algorithm.num_learning_epochs = 1
    algorithm.normalize_advantage_per_mini_batch = False
    algorithm.desired_kl = None
    algorithm.schedule = "fixed"
    algorithm.use_clipped_value_loss = True
    algorithm.value_loss_coef = 0.7
    algorithm.entropy_coef = 0.05
    algorithm.clip_param = 0.2
    algorithm.max_grad_norm = 100.0
    algorithm.device = "cpu"
    algorithm.is_multi_gpu = False
    algorithm._actor_mean_abort_count = 0
    algorithm._time_reversal_update_count = 0
    algorithm.current_learning_iteration = 0
    algorithm._tr_augmentation = None

    losses = algorithm.update()

    # Frozen from the audited-base surrogate, clipped-value, and entropy
    # equations. These are literal goldens, not values recomputed by helpers
    # shared with the implementation under test.
    assert losses["surrogate"] == pytest.approx(-0.040230363607406616, abs=1.0e-7)
    assert losses["value"] == pytest.approx(0.28312501311302185, abs=1.0e-7)
    assert losses["entropy"] == pytest.approx(0.40000003576278687, abs=1.0e-7)
    assert losses["actor_bound"] == 0.0
    assert losses["trs_scale"] == 0.0
    assert losses["weighted_tr_total"] == 0.0
    assert losses["weighted_tr_augmentation"] == 0.0
    assert algorithm.value_loss_coef == 0.7
    assert all(parameter.grad is not None and torch.any(parameter.grad != 0.0) for parameter in critic.parameters())
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad)) for parameter in critic.parameters()
    )
    assert any(
        not torch.equal(parameter, initial)
        for parameter, initial in zip(critic.parameters(), initial_critic_parameters)
    )
    torch.testing.assert_close(
        torch.stack(tuple(parameters)),
        torch.tensor([0.20821358263492584, -0.09954308718442917, 0.31706249713897705, -0.18477500975131989]),
        rtol=0.0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        torch.stack(tuple(optimizer.state[parameter]["momentum_buffer"] for parameter in parameters)),
        torch.tensor([-0.2737857699394226, -0.015230363234877586, -0.5687500238418579, -0.5074999928474426]),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert storage.cleared
    assert algorithm.current_learning_iteration == 1


def test_augmentation_gradient_diagnostics_have_independent_low_frequency_cadence(monkeypatch):
    diagnostic_calls = []
    original_diagnostics = _PPO_MODULE.time_reversal_gradient_diagnostics

    def counted_diagnostics(*args, **kwargs):
        diagnostic_calls.append(kwargs["prefix"])
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(_PPO_MODULE, "time_reversal_gradient_diagnostics", counted_diagnostics)

    class FakeAugmentation:
        def prepare_update(self):
            return _PPO_MODULE.TimeReversalAugmentation._empty_diagnostics()

        def actor_nll(self, actor, _original_batch_size):
            return (actor.weight - 1.0).square()

        def finish_update(self):
            pass

    def run_update(iteration):
        events = []
        actor = _CountingActor(events)
        critic = _CountingCritic(events)
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
        algorithm = _schedule_algorithm(iteration)
        algorithm.symmetry.update(
            {
                "use_tr_policy_consistency": False,
                "use_tr_value_consistency": False,
                "tr_gradient_diagnostics": {"enabled": False},
                "tr_augmentation": {
                    "enabled": True,
                    "coefficient": 0.5,
                    "schedule": {"enabled": True, "warmup_iterations": 0, "rampup_iterations": 0},
                    "gradient_diagnostics_interval": 2,
                    "gradient_diagnostics_epsilon": 1.0e-12,
                },
            }
        )
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
        algorithm._tr_augmentation = FakeAugmentation()
        losses = algorithm.update()
        assert batch.observations.batch_size[0] == batch_size
        assert batch.actions.shape == (batch_size, 12)
        assert batch.old_actions_log_prob.shape[0] == batch_size
        assert batch.advantages.shape[0] == batch_size
        assert batch.returns.shape[0] == batch_size
        return losses

    cadence_losses = run_update(0)
    between_cadence_losses = run_update(1)

    assert diagnostic_calls == ["augmentation_actor"]
    assert cadence_losses["tr_augmentation/gradient_diagnostics_ran"] == 1.0
    assert between_cadence_losses["tr_augmentation/gradient_diagnostics_ran"] == 0.0
    for losses in (cadence_losses, between_cadence_losses):
        for key in (
            "tr_augmentation/gradient_norm",
            "tr_augmentation/gradient_weighted_norm",
            "tr_augmentation/gradient_cosine_ppo_actor",
        ):
            assert key in losses
            assert math.isfinite(losses[key])


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
    # The algorithm counter points at the next absolute update while the
    # runner serializes the index of the update that just completed.
    save_runner.alg.current_learning_iteration = 500
    save_runner.alg._time_reversal_update_count = 500
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


def _time_reversal_checkpoint(algorithm, completed_iteration):
    policy_schedule, value_schedule = algorithm._resolved_time_reversal_schedules()
    return {
        "iter": completed_iteration,
        "time_reversal_state": {
            "schema_version": algorithm._TIME_REVERSAL_STATE_SCHEMA_VERSION,
            "last_completed_update": completed_iteration,
            "next_absolute_update": completed_iteration + 1,
            "policy_schedule": vars(policy_schedule),
            "value_schedule": vars(value_schedule),
            "augmentation_schedule": vars(algorithm._resolved_time_reversal_augmentation_schedule()),
        },
    }


def test_positive_coefficient_checkpoint_remains_loadable(monkeypatch):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    algorithm.symmetry["mirror_loss_coeff"] = 0.1
    algorithm.symmetry["value_loss_coeff"] = 0.05
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=19_999)
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)

    loaded_iteration = algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert loaded_iteration is True
    assert algorithm.current_learning_iteration == 20_000
    assert algorithm._effective_time_reversal_coefficients() == pytest.approx((1.0, 0.1, 0.05))


def test_checkpoint_schema_accepts_exact_schedule_at_hard_step_boundary(monkeypatch):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=499)
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)

    loaded_iteration = algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert loaded_iteration
    assert checkpoint["iter"] == 500
    assert algorithm.current_learning_iteration == 500
    assert algorithm._effective_time_reversal_coefficients() == (1.0, 0.20, 0.10)


def test_load_immediate_save_and_reload_preserves_same_pending_update(monkeypatch):
    monkeypatch.setattr(PPO, "save", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=499)
    algorithm.load(checkpoint, load_cfg=None, strict=True)

    immediate_resave = algorithm.save()
    immediate_resave["iter"] = checkpoint["iter"]
    restored = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    restored.load(immediate_resave, load_cfg=None, strict=True)

    assert immediate_resave["time_reversal_state"]["last_completed_update"] == 499
    assert immediate_resave["time_reversal_state"]["next_absolute_update"] == 500
    assert immediate_resave["iter"] == 500
    assert restored.current_learning_iteration == 500
    assert restored._effective_time_reversal_coefficients() == (1.0, 0.20, 0.10)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda state: state.update(schema_version=1), "schema_version"),
        (lambda state: state.update(next_absolute_update=499), "next_absolute_update"),
        (lambda state: state["policy_schedule"].update(target_coeff=0.21), "policy_schedule"),
        (lambda state: state["value_schedule"].update(warmup_iterations=500.0), "value_schedule"),
        (lambda state: state.pop("augmentation_schedule"), "schema fields"),
    ],
)
def test_checkpoint_schema_rejects_mismatch_before_upstream_load(monkeypatch, mutation, error):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=499)
    mutation(checkpoint["time_reversal_state"])
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return True

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match=error):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert not upstream_load_called


def test_checkpoint_without_time_reversal_state_retains_legacy_iteration_fallback(monkeypatch):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = {"iter": 499}
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)

    algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert checkpoint["iter"] == 500
    assert algorithm.current_learning_iteration == 500


def test_partial_checkpoint_load_does_not_advance_schedule_counter(monkeypatch):
    algorithm = _schedule_algorithm(123)
    checkpoint = {"iter": 499}
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: False)

    loaded_iteration = algorithm.load(checkpoint, load_cfg={"iteration": False}, strict=True)

    assert not loaded_iteration
    assert checkpoint["iter"] == 499
    assert algorithm.current_learning_iteration == 123
    assert algorithm._time_reversal_update_count == 123


def test_actor_only_inference_load_accepts_nondefault_checkpoint_but_exact_resume_rejects(monkeypatch):
    source = _schedule_algorithm(0, warmup_iterations=17, rampup_iterations=23)
    checkpoint = _time_reversal_checkpoint(source, completed_iteration=41)
    target = _schedule_algorithm(9, warmup_iterations=500, rampup_iterations=0)
    received_environment_iterations = []
    target.symmetry["_env"] = SimpleNamespace(
        unwrapped=SimpleNamespace(set_training_iteration=received_environment_iterations.append)
    )
    upstream_calls = []

    def upstream_load(_algorithm, _checkpoint, load_cfg, strict):
        upstream_calls.append((copy.deepcopy(load_cfg), strict))
        return bool(load_cfg and load_cfg.get("iteration", False))

    monkeypatch.setattr(PPO, "load", upstream_load)
    with pytest.raises(ValueError, match="policy_schedule"):
        target.load(copy.deepcopy(checkpoint), load_cfg=None, strict=True)

    inference_cfg = {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "environment_iteration": True,
        "rnd": False,
        "augmentation": False,
    }
    loaded_iteration = target.load(copy.deepcopy(checkpoint), load_cfg=inference_cfg, strict=True)

    assert loaded_iteration is False
    assert upstream_calls == [(inference_cfg, True)]
    assert target.current_learning_iteration == 9
    assert target._time_reversal_update_count == 9
    assert received_environment_iterations == [42]


def test_actor_only_inference_restores_environment_iteration_from_legacy_checkpoint(monkeypatch):
    target = _schedule_algorithm(9, warmup_iterations=500, rampup_iterations=0)
    received_environment_iterations = []
    target.symmetry["_env"] = SimpleNamespace(
        unwrapped=SimpleNamespace(set_training_iteration=received_environment_iterations.append)
    )
    checkpoint = {"iter": 499}
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: False)

    loaded_iteration = target.load(
        checkpoint,
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "environment_iteration": True,
            "rnd": False,
            "augmentation": False,
        },
        strict=True,
    )

    assert loaded_iteration is False
    assert checkpoint["iter"] == 499
    assert target.current_learning_iteration == 9
    assert target._time_reversal_update_count == 9
    assert received_environment_iterations == [500]


def test_actor_only_environment_iteration_restore_rejects_checkpoint_without_iter(monkeypatch):
    target = _schedule_algorithm(9)
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return False

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match="environment_iteration.*iter"):
        target.load(
            {},
            load_cfg={"actor": True, "iteration": False, "environment_iteration": True},
            strict=True,
        )

    assert upstream_load_called is False


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


def test_raw_action_and_normalized_requested_target_consistency_are_distinct():
    original_mean = torch.zeros(2, 2)
    transformed_mean = torch.tensor([[1.0, 1.0], [2.0, -2.0]])
    offset = torch.tensor([0.0, 0.0])
    scale = torch.tensor([0.5, 0.1])
    limits = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])

    normalized_original = normalize_requested_joint_targets(original_mean, offset, scale, limits)
    normalized_transformed = normalize_requested_joint_targets(transformed_mean, offset, scale, limits)
    raw_loss = torch.mean((transformed_mean - original_mean).square())
    normalized_loss = torch.mean((normalized_transformed - normalized_original).square())

    assert raw_loss == pytest.approx(2.5)
    assert normalized_loss == pytest.approx(1.30)
    assert normalized_transformed[1, 0] > 1.0  # requested targets are not hard-clipped before comparison


def test_per_joint_feasible_actor_penalty_is_smooth_and_uncapped():
    lower = torch.tensor([-1.0, -2.0])
    upper = torch.tensor([1.0, 2.0])

    inside = feasible_actor_mean_penalty(torch.tensor([[0.0, 1.5]]), lower, upper)
    slight = feasible_actor_mean_penalty(torch.tensor([[1.2, 2.4]]), lower, upper)
    severe = feasible_actor_mean_penalty(torch.tensor([[5.0, 10.0]]), lower, upper)

    assert inside == 0.0
    assert 0.0 < slight < severe
    assert severe > 1.0


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


def test_environment_curriculum_receives_absolute_resumed_iteration():
    received = []
    environment = SimpleNamespace(set_training_iteration=received.append)
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {"_env": SimpleNamespace(unwrapped=environment)}
    algorithm.current_learning_iteration = 137

    algorithm._sync_environment_training_iteration()

    assert received == [137]


def test_masked_mse_with_no_selected_commands_is_finite_zero():
    prediction = torch.full((2, 3), float("nan"), requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 1)

    loss = TimeReversalPPO._masked_mse(prediction, target, mask)
    loss.backward()

    assert math.isfinite(loss.item())
    assert loss.item() == 0.0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
