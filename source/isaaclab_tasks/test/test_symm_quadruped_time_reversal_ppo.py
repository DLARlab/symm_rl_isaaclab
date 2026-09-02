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

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_POLICY_OBS_DIM,
)

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
TransitionAlignedTRCandidateBatch = _PPO_MODULE.TransitionAlignedTRCandidateBatch
TransitionAlignedTRBuffer = _PPO_MODULE.TransitionAlignedTRBuffer
TransitionAlignedTRRecord = _PPO_MODULE.TransitionAlignedTRRecord
TRSimulatorValidityMetadata = _PPO_MODULE.TRSimulatorValidityMetadata
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


class _SequenceActor(torch.nn.Module):
    """Small actor exposing every forward input used by an exact-sequence update."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.distribution = SimpleNamespace(std_param=None)
        self.is_recurrent = False
        self.forward_observations = []
        self.kl_batch_sizes = []

    def forward(self, observations, **_kwargs):
        policy = observations["policy"]
        self.forward_observations.append(policy.detach().clone())
        mean = (self.weight * policy[:, :1]).expand(-1, 12)
        self.output_distribution_params = (mean,)
        self.output_entropy = (self.weight * 0.0).expand(policy.shape[0])
        return mean

    def get_output_log_prob(self, actions):
        return self.weight * actions[:, 0]

    def get_kl_divergence(self, old_distribution_params, distribution_params):
        old_batch_size = old_distribution_params[0].shape[0]
        current_batch_size = distribution_params[0].shape[0]
        self.kl_batch_sizes.append((old_batch_size, current_batch_size))
        return torch.full((current_batch_size,), 0.01, device=distribution_params[0].device)


class _SequenceCritic(torch.nn.Module):
    """Small critic exposing every forward input used by an exact-sequence update."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.is_recurrent = False
        self.forward_observations = []

    def forward(self, observations, **_kwargs):
        policy = observations["policy"]
        self.forward_observations.append(policy.detach().clone())
        return self.weight * policy[:, 1:2]


class _SequenceCandidateBuffer:
    """One-shot candidate source used to isolate the PPO auxiliary path."""

    def __init__(self, candidate):
        self.candidate = candidate
        self.diagnostics = {"tr_sequence/test_diagnostic": 1.0}
        self.pop_iterations = []

    def pop_candidates(self, current_update):
        self.pop_iterations.append(current_update)
        return self.candidate


class _SequenceContextEnvironment:
    """Return deterministic pre/post sequence contexts in call order."""

    def __init__(self, *contexts):
        self.contexts = list(contexts)

    def get_time_reversal_sequence_context(self):
        return self.contexts.pop(0)


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


def _sequence_candidates(
    actor_source: torch.Tensor,
    value_source: torch.Tensor,
    reversed_observation: torch.Tensor,
) -> TransitionAlignedTRCandidateBatch:
    """Build a fully populated exact-sequence candidate batch."""
    count = actor_source.shape[0]
    assert value_source.shape == actor_source.shape
    assert reversed_observation.shape == actor_source.shape
    integer = torch.arange(count, dtype=torch.long, device=actor_source.device)
    return TransitionAlignedTRCandidateBatch(
        actor_source_observation=actor_source,
        value_source_observation=value_source,
        reversed_observation=reversed_observation,
        weight=torch.ones(count, 1, dtype=actor_source.dtype, device=actor_source.device),
        environment_id=integer,
        episode_id=integer + 10,
        command_segment_id=integer + 20,
        gait_segment_id=integer + 30,
        disturbance_generation_id=integer + 40,
        oldest_collection_update_id=torch.zeros_like(integer),
        newest_collection_update_id=torch.ones_like(integer),
        policy_version_span=torch.ones_like(integer),
        candidate_age_updates=torch.zeros_like(integer),
    )


def _sequence_context(**overrides) -> dict[str, torch.Tensor]:
    """Build one complete single-environment causal-sequence context."""
    context = {
        "episode_id": torch.zeros(1, dtype=torch.long),
        "command_segment_id": torch.zeros(1, dtype=torch.long),
        "gait_segment_id": torch.zeros(1, dtype=torch.long),
        "disturbance_generation_id": torch.zeros(1, dtype=torch.long),
        "manual_reset_generation": torch.zeros(1, dtype=torch.long),
        "episode_step": torch.zeros(1, dtype=torch.long),
        "command_segment_age": torch.zeros(1, dtype=torch.long),
        "gait_segment_age": torch.zeros(1, dtype=torch.long),
        "disturbance_segment_age": torch.zeros(1, dtype=torch.long),
        "gait_row": torch.zeros(1, dtype=torch.long),
        "body_linear_velocity_b": torch.zeros(1, 3),
        "body_yaw_rate": torch.zeros(1),
        "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        "contact_impulse": torch.zeros(1),
        "foot_slip": torch.zeros(1),
        "reverse_dynamics_residual": torch.zeros(1),
        "actuator_saturation": torch.zeros(1),
    }
    context.update(overrides)
    return context


def _finalized_sequence_record(index: int, history_length: int = 1) -> TransitionAlignedTRRecord:
    """Build one valid decision record for buffer-reset tests."""
    latest_frame = torch.zeros(1, SYMM_QUADRUPED_POLICY_OBS_DIM)
    layout = go2_symm.SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    latest_frame[:, layout.velocity_command.start] = 1.0
    latest_frame[:, layout.gait_period] = 1.5
    latest_frame[:, layout.duty_factor] = 0.6
    latest_frame[:, layout.foot_phase_cos] = 1.0
    observation = go2_symm.pack_term_major_policy_history(latest_frame[:, None, :].repeat(1, history_length, 1))
    integer = torch.zeros(1, dtype=torch.long)
    boolean = torch.zeros(1, dtype=torch.bool)
    return TransitionAlignedTRRecord(
        policy_observation=observation,
        latest_frame=latest_frame,
        action=torch.full((1, 12), float(index)),
        episode_id=integer.clone(),
        command_segment_id=integer.clone(),
        gait_segment_id=integer.clone(),
        disturbance_generation_id=integer.clone(),
        collection_update_id=integer.clone(),
        transition_valid=torch.ones(1, dtype=torch.bool),
        done=boolean.clone(),
        timeout=boolean.clone(),
        history_warmup_complete=torch.ones(1, dtype=torch.bool),
        gait_row=integer.clone(),
        simulator=TRSimulatorValidityMetadata.neutral(1, device="cpu", dtype=observation.dtype),
    )


def _exact_sequence_update_algorithm(
    actor: _SequenceActor,
    critic: _SequenceCritic,
    candidate: TransitionAlignedTRCandidateBatch | None,
    *,
    batch_size: int = 4,
    policy_coefficient: float = 1.0,
    value_coefficient: float = 1.0,
    adaptive_kl: bool = False,
):
    """Return a minimal feed-forward PPO fixture with exact causal candidates."""
    policy_width = 2 * SYMM_QUADRUPED_POLICY_OBS_DIM
    policy = torch.zeros(batch_size, policy_width)
    policy[:, 0] = torch.linspace(-0.4, 0.4, batch_size)
    policy[:, 1] = torch.linspace(0.3, -0.3, batch_size)
    observations = TensorDict({"policy": policy}, batch_size=[batch_size])
    actions = torch.zeros(batch_size, 12)
    actions[:, 0] = torch.linspace(-0.2, 0.2, batch_size)
    batch = SimpleNamespace(
        observations=observations,
        actions=actions,
        old_actions_log_prob=torch.linspace(-0.3, 0.1, batch_size),
        values=torch.linspace(-0.2, 0.2, batch_size).unsqueeze(-1),
        advantages=torch.linspace(0.5, -0.5, batch_size).unsqueeze(-1),
        returns=torch.linspace(0.1, 0.4, batch_size).unsqueeze(-1),
        masks=None,
        hidden_states=(None, None),
        old_distribution_params=(torch.zeros(batch_size, 12),),
    )
    algorithm = _schedule_algorithm(1, rampup_iterations=0)
    algorithm.symmetry.update(
        tr_consistency_mode="transition_aligned_sequence",
        history_enabled=True,
        history_length=2,
        use_tr_policy_consistency=policy_coefficient > 0.0,
        use_tr_value_consistency=value_coefficient > 0.0,
        tr_policy_schedule={
            "enabled": policy_coefficient > 0.0,
            "target_coeff": policy_coefficient,
            "warmup_iterations": 0,
            "rampup_iterations": 0,
        },
        tr_value_schedule={
            "enabled": value_coefficient > 0.0,
            "target_coeff": value_coefficient,
            "warmup_iterations": 0,
            "rampup_iterations": 0,
        },
        tr_gradient_diagnostics={"enabled": False},
        tr_augmentation={
            "enabled": False,
            "coefficient": 0.0,
            "schedule": {"enabled": False, "target_coeff": 0.0},
        },
        log_disabled_raw_consistency=False,
        tr_policy_output_space="raw_action_mean",
        _env=None,
    )
    storage = _SingleBatchStorage(batch, torch.zeros(batch_size, 12))
    algorithm.actor = actor
    algorithm.critic = critic
    algorithm.storage = storage
    algorithm.optimizer = torch.optim.SGD((*actor.parameters(), *critic.parameters()), lr=0.0)
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.num_mini_batches = 1
    algorithm.num_learning_epochs = 1
    algorithm.normalize_advantage_per_mini_batch = False
    algorithm.desired_kl = 0.01 if adaptive_kl else None
    algorithm.schedule = "adaptive" if adaptive_kl else "fixed"
    algorithm.learning_rate = 0.0
    algorithm.use_clipped_value_loss = False
    algorithm.value_loss_coef = 1.0
    algorithm.entropy_coef = 0.0
    algorithm.clip_param = 0.2
    algorithm.max_grad_norm = 100.0
    algorithm.device = "cpu"
    algorithm.is_multi_gpu = False
    algorithm.gpu_global_rank = 0
    algorithm._actor_mean_abort_count = 0
    algorithm._tr_augmentation = None
    algorithm._tr_sequence_pending = None
    algorithm._tr_sequence_buffer = _SequenceCandidateBuffer(candidate)
    return algorithm, batch, storage


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
        # Most fixtures below freeze the audited framewise compatibility path.
        # Exact-sequence tests opt into the new primary mode explicitly.
        "tr_consistency_mode": "framewise_feature_approx",
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
        observations=TensorDict(
            {"policy": torch.zeros(batch_size, SYMM_QUADRUPED_POLICY_OBS_DIM)}, batch_size=[batch_size]
        ),
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


def test_history_actor_and_value_consistency_losses_are_finite_and_fully_transformed():
    events = []
    actor = _CountingActor(events)
    critic = _CountingCritic(events)
    batch_size = 2
    policy_width = 30 * SYMM_QUADRUPED_POLICY_OBS_DIM
    observations = TensorDict({"policy": torch.zeros(batch_size, policy_width)}, batch_size=[batch_size])
    batch = SimpleNamespace(
        observations=observations,
        actions=torch.zeros(batch_size, 12),
        old_actions_log_prob=torch.zeros(batch_size),
        values=torch.zeros(batch_size, 1),
        advantages=torch.ones(batch_size, 1),
        returns=torch.zeros(batch_size, 1),
        masks=None,
        hidden_states=(None, None),
        old_distribution_params=(),
    )
    algorithm = _schedule_algorithm(2000)
    algorithm.symmetry.update(
        history_enabled=True,
        history_length=30,
        history_trs_mode="framewise_feature",
        data_augmentation_func=go2_symm.compute_time_reversal_states,
        _env=None,
    )
    algorithm.actor = actor
    algorithm.critic = critic
    algorithm.storage = _SingleBatchStorage(batch, torch.zeros(batch_size, 12))
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
    algorithm._tr_augmentation = None

    losses = algorithm.update()

    assert actor.forward_calls == 2
    assert critic.forward_calls == 2
    assert math.isfinite(losses["raw_tr_policy_residual"])
    assert math.isfinite(losses["raw_tr_value_residual"])
    assert losses["history_enabled"] == 1.0
    assert losses["history_length"] == 30.0
    assert losses["policy_input_dim"] == 1920.0
    assert losses["instantaneous_frame_dim"] == 64.0
    assert losses["observation_contract_version/hardware_proprio_history_64d_v1"] == 1.0
    assert losses["history_trs_mode/framewise_feature"] == 1.0


def test_exact_sequence_uses_edge_actor_source_and_next_state_value_source():
    actor = _SequenceActor()
    critic = _SequenceCritic()
    width = 2 * SYMM_QUADRUPED_POLICY_OBS_DIM
    actor_source = torch.full((2, width), -11.0)
    value_source = torch.full((2, width), -22.0)
    reversed_observation = torch.full((2, width), -33.0)
    actor_source[:, 0] = torch.tensor([1.5, -2.0])
    value_source[:, 1] = torch.tensor([0.75, -1.25])
    reversed_observation[:, 0] = actor_source[:, 0]
    reversed_observation[:, 1] = value_source[:, 1]
    candidate = _sequence_candidates(actor_source, value_source, reversed_observation)
    algorithm, _, _ = _exact_sequence_update_algorithm(actor, critic, candidate)

    losses = algorithm.update()

    assert len(actor.forward_observations) == 3
    assert len(critic.forward_observations) == 3
    torch.testing.assert_close(actor.forward_observations[1], actor_source)
    torch.testing.assert_close(actor.forward_observations[2], reversed_observation)
    torch.testing.assert_close(critic.forward_observations[1], value_source)
    torch.testing.assert_close(critic.forward_observations[2], reversed_observation)
    assert losses["tr_policy_consistency"] == pytest.approx(0.0, abs=1.0e-12)
    assert losses["tr_value_consistency"] == pytest.approx(0.0, abs=1.0e-12)
    assert losses["weighted_tr_total"] == pytest.approx(0.0, abs=1.0e-12)
    assert losses["tr_sequence/candidate_pool_count"] == 2.0
    assert losses["tr_sequence/test_diagnostic"] == 1.0
    assert algorithm._tr_sequence_buffer.pop_iterations == [1]


def test_sequence_finalization_uses_post_step_diagnostics_and_rejects_new_boundary():
    pre_context = _sequence_context()
    post_context = _sequence_context(
        command_segment_id=torch.ones(1, dtype=torch.long),
        contact_impulse=torch.tensor([7.0]),
        foot_slip=torch.tensor([0.75]),
    )
    environment = _SequenceContextEnvironment(pre_context, post_context)
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(
        tr_consistency_mode="transition_aligned_sequence",
        _env=SimpleNamespace(unwrapped=environment),
    )
    algorithm.device = "cpu"
    algorithm._tr_sequence_buffer = TransitionAlignedTRBuffer(1, 1, "cpu")
    algorithm._tr_sequence_pending = None
    algorithm._tr_sequence_manual_reset_generation = torch.zeros(1, dtype=torch.long)
    observations = TensorDict(
        {"policy": torch.zeros(1, SYMM_QUADRUPED_POLICY_OBS_DIM)},
        batch_size=[1],
    )

    algorithm._capture_time_reversal_sequence_before_step(observations, torch.zeros(1, 12))
    algorithm._finalize_time_reversal_sequence_step(torch.zeros(1, dtype=torch.bool), {})

    record = algorithm._tr_sequence_buffer._records[-1]
    assert not record.transition_valid.item()
    assert record.simulator.contact_impulse.item() == 7.0
    assert record.simulator.foot_slip.item() == 0.75


def test_sequence_capture_clears_saved_candidates_after_manual_environment_reset():
    buffer = TransitionAlignedTRBuffer(1, 1, "cpu")
    for index in range(3):
        buffer.append(_finalized_sequence_record(index))
    assert buffer.candidate_count(0) == 1

    environment = _SequenceContextEnvironment(
        _sequence_context(manual_reset_generation=torch.ones(1, dtype=torch.long))
    )
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(
        tr_consistency_mode="transition_aligned_sequence",
        _env=SimpleNamespace(unwrapped=environment),
    )
    algorithm.device = "cpu"
    algorithm._tr_sequence_buffer = buffer
    algorithm._tr_sequence_pending = None
    algorithm._tr_sequence_manual_reset_generation = torch.zeros(1, dtype=torch.long)
    observations = TensorDict(
        {"policy": torch.zeros(1, SYMM_QUADRUPED_POLICY_OBS_DIM)},
        batch_size=[1],
    )

    algorithm._capture_time_reversal_sequence_before_step(observations, torch.zeros(1, 12))

    assert buffer.candidate_count(0) == 0
    assert buffer._records_since_clear.item() == 0


def test_sequence_update_polls_manual_reset_before_consuming_candidate():
    history_length = 2
    buffer = TransitionAlignedTRBuffer(1, history_length, "cpu")
    for index in range(history_length + 2):
        buffer.append(_finalized_sequence_record(index, history_length))
    assert buffer.candidate_count(1) == 1

    actor = _SequenceActor()
    critic = _SequenceCritic()
    algorithm, _, _ = _exact_sequence_update_algorithm(actor, critic, None)
    algorithm._tr_sequence_buffer = buffer
    algorithm._tr_sequence_manual_reset_generation = torch.zeros(1, dtype=torch.long)
    algorithm.symmetry["_env"] = SimpleNamespace(
        unwrapped=_SequenceContextEnvironment(
            _sequence_context(manual_reset_generation=torch.ones(1, dtype=torch.long))
        )
    )

    losses = algorithm.update()

    assert losses["tr_sequence/candidate_pool_available"] == 0.0
    assert len(actor.forward_observations) == 1
    assert len(critic.forward_observations) == 1


@pytest.mark.parametrize(
    "young_segment",
    ["command_segment_age", "gait_segment_age", "disturbance_segment_age"],
)
def test_sequence_history_warmup_requires_episode_and_task_segment_ages(young_segment):
    history_length = 30
    segment_ages = {
        "command_segment_age": torch.full((1,), 100, dtype=torch.long),
        "gait_segment_age": torch.full((1,), 100, dtype=torch.long),
        "disturbance_segment_age": torch.full((1,), 100, dtype=torch.long),
    }
    segment_ages[young_segment] = torch.zeros(1, dtype=torch.long)
    environment = _SequenceContextEnvironment(
        _sequence_context(
            episode_step=torch.full((1,), 100, dtype=torch.long),
            **segment_ages,
        )
    )
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(
        tr_consistency_mode="transition_aligned_sequence",
        _env=SimpleNamespace(unwrapped=environment),
    )
    algorithm.device = "cpu"
    algorithm._tr_sequence_buffer = TransitionAlignedTRBuffer(1, history_length, "cpu")
    algorithm._tr_sequence_pending = None
    algorithm._tr_sequence_manual_reset_generation = torch.zeros(1, dtype=torch.long)
    observations = TensorDict(
        {"policy": torch.zeros(1, history_length * SYMM_QUADRUPED_POLICY_OBS_DIM)},
        batch_size=[1],
    )

    algorithm._capture_time_reversal_sequence_before_step(observations, torch.zeros(1, 12))

    assert not algorithm._tr_sequence_pending.history_warmup_complete.item()


def test_sequence_activation_matches_auxiliary_consumers_not_sidecar():
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(
        tr_consistency_mode="transition_aligned_sequence",
        use_mirror_loss=False,
        use_tr_policy_consistency=True,
        use_tr_value_consistency=False,
        value_loss_coeff=0.0,
        tr_policy_schedule={"enabled": True, "target_coeff": 0.0},
        tr_value_schedule={"enabled": False, "target_coeff": 0.0},
        tr_gradient_diagnostics={"enabled": True},
        log_disabled_raw_consistency=False,
        tr_augmentation={"enabled": False},
    )
    assert algorithm._sequence_consistency_enabled()

    algorithm.symmetry.update(
        tr_policy_schedule={"enabled": False, "target_coeff": 0.0},
        use_tr_policy_consistency=False,
        tr_gradient_diagnostics={"enabled": False},
        tr_augmentation={"enabled": True},
    )
    assert algorithm._time_reversal_enabled()
    assert not algorithm._sequence_consistency_enabled()


def test_exact_sequence_rejects_wrapper_action_clipping_before_buffer_allocation():
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(
        tr_consistency_mode="transition_aligned_sequence",
        use_tr_policy_consistency=True,
        tr_policy_schedule={"enabled": True, "target_coeff": 0.1},
        history_enabled=False,
        history_length=1,
        _env=SimpleNamespace(clip_actions=1.0),
    )
    algorithm.device = "cpu"
    algorithm._tr_sequence_buffer = None
    algorithm._tr_sequence_manual_reset_generation = None

    with pytest.raises(ValueError, match="clip_actions=None"):
        algorithm._initialize_time_reversal_sequence_buffer()


def test_exact_sequence_targets_are_stop_gradient_but_reversed_branch_is_differentiable():
    actor = _SequenceActor()
    critic = _SequenceCritic()
    width = 2 * SYMM_QUADRUPED_POLICY_OBS_DIM
    actor_source = torch.zeros(2, width)
    value_source = torch.zeros(2, width)
    reversed_observation = torch.zeros(2, width)
    actor_source[:, 0] = torch.tensor([1.0, -1.0])
    value_source[:, 1] = torch.tensor([2.0, -2.0])
    reversed_observation[:, 0] = torch.tensor([3.0, -3.0])
    reversed_observation[:, 1] = torch.tensor([5.0, -5.0])
    actor_source.requires_grad_()
    value_source.requires_grad_()
    reversed_observation.requires_grad_()
    candidate = _sequence_candidates(actor_source, value_source, reversed_observation)
    algorithm, _, _ = _exact_sequence_update_algorithm(actor, critic, candidate)

    losses = algorithm.update()

    assert losses["tr_policy_consistency"] > 0.0
    assert losses["tr_value_consistency"] > 0.0
    assert actor_source.grad is None
    assert value_source.grad is None
    assert reversed_observation.grad is not None
    assert torch.count_nonzero(reversed_observation.grad[:, 0]).item() == 2
    assert torch.count_nonzero(reversed_observation.grad[:, 1]).item() == 2


@pytest.mark.parametrize(
    ("policy_coefficient", "value_coefficient", "actor_forwards", "critic_forwards"),
    [(0.0, 1.0, 1, 3), (1.0, 0.0, 3, 1)],
)
def test_exact_sequence_zero_coefficient_skips_that_network_auxiliary_forwards(
    policy_coefficient,
    value_coefficient,
    actor_forwards,
    critic_forwards,
):
    actor = _SequenceActor()
    critic = _SequenceCritic()
    width = 2 * SYMM_QUADRUPED_POLICY_OBS_DIM
    candidate = _sequence_candidates(
        torch.zeros(2, width),
        torch.zeros(2, width),
        torch.ones(2, width),
    )
    algorithm, _, _ = _exact_sequence_update_algorithm(
        actor,
        critic,
        candidate,
        policy_coefficient=policy_coefficient,
        value_coefficient=value_coefficient,
    )

    losses = algorithm.update()

    assert len(actor.forward_observations) == actor_forwards
    assert len(critic.forward_observations) == critic_forwards
    if policy_coefficient == 0.0:
        assert losses["tr_policy_consistency"] == 0.0
    if value_coefficient == 0.0:
        assert losses["tr_value_consistency"] == 0.0


def test_exact_sequence_without_candidate_has_zero_losses_and_no_auxiliary_forwards():
    actor = _SequenceActor()
    critic = _SequenceCritic()
    algorithm, _, _ = _exact_sequence_update_algorithm(actor, critic, None)

    losses = algorithm.update()

    assert len(actor.forward_observations) == 1
    assert len(critic.forward_observations) == 1
    assert losses["tr_policy_consistency"] == 0.0
    assert losses["tr_value_consistency"] == 0.0
    assert losses["weighted_tr_total"] == 0.0
    assert losses["tr_sequence/candidate_pool_available"] == 0.0
    assert losses["tr_sequence/candidate_pool_count"] == 0.0
    assert algorithm._tr_sequence_buffer.pop_iterations == [1]


def test_exact_candidates_do_not_change_ppo_samples_or_adaptive_kl_batch_size():
    actor = _SequenceActor()
    critic = _SequenceCritic()
    width = 2 * SYMM_QUADRUPED_POLICY_OBS_DIM
    candidate = _sequence_candidates(
        torch.zeros(2, width),
        torch.zeros(2, width),
        torch.ones(2, width),
    )
    algorithm, batch, storage = _exact_sequence_update_algorithm(
        actor,
        critic,
        candidate,
        batch_size=5,
        adaptive_kl=True,
    )
    frozen_ppo_fields = {
        name: getattr(batch, name).detach().clone()
        for name in ("old_actions_log_prob", "advantages", "returns", "values")
    }
    frozen_old_distribution = batch.old_distribution_params[0].detach().clone()

    algorithm.update()

    for name, expected in frozen_ppo_fields.items():
        torch.testing.assert_close(getattr(batch, name), expected)
    torch.testing.assert_close(batch.old_distribution_params[0], frozen_old_distribution)
    assert actor.kl_batch_sizes == [(5, 5)]
    assert [observations.shape[0] for observations in actor.forward_observations] == [5, 2, 2]
    assert [observations.shape[0] for observations in critic.forward_observations] == [5, 2, 2]
    assert storage.actions.shape[0] == 5
    assert storage.distribution_params[0].shape[0] == 5


def test_deprecated_data_augmentation_still_duplicates_ppo_minibatches():
    events = []
    actor = _CountingActor(events)
    critic = _CountingCritic(events)
    batch_size = 2
    observations = TensorDict(
        {"policy": torch.zeros(batch_size, SYMM_QUADRUPED_POLICY_OBS_DIM)}, batch_size=[batch_size]
    )
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
    observations = torch.zeros(4, SYMM_QUADRUPED_POLICY_OBS_DIM)
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
            observations=TensorDict(
                {"policy": torch.zeros(batch_size, SYMM_QUADRUPED_POLICY_OBS_DIM)}, batch_size=[batch_size]
            ),
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
            "policy_contract": algorithm._policy_contract_metadata(),
        },
    }


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


def test_command_curriculum_state_round_trips_through_algorithm_checkpoint(monkeypatch):
    monkeypatch.setattr(PPO, "save", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)
    saved_curriculum = {
        "training_iteration": 0,
        "weights": torch.tensor([[1.0, 2.0]]),
        "rng_state": torch.arange(4),
    }
    restored_states = []
    environment = SimpleNamespace(
        get_command_curriculum_state=lambda: copy.deepcopy(saved_curriculum),
        load_command_curriculum_state=restored_states.append,
    )
    source = _schedule_algorithm(0)
    source.symmetry["_env"] = SimpleNamespace(unwrapped=environment)
    checkpoint = source.save()
    checkpoint["iter"] = 0

    target = _schedule_algorithm(0)
    target.symmetry["_env"] = SimpleNamespace(unwrapped=environment)
    target.load(checkpoint, load_cfg=None, strict=True)

    assert len(restored_states) == 1
    assert torch.equal(restored_states[0]["weights"], saved_curriculum["weights"])
    assert torch.equal(restored_states[0]["rng_state"], saved_curriculum["rng_state"])


def test_command_curriculum_iteration_mismatch_is_rejected_before_upstream_load(monkeypatch):
    restored_states = []
    environment = SimpleNamespace(
        get_command_curriculum_state=lambda: {"training_iteration": 0},
        validate_command_curriculum_state=lambda _state: None,
        load_command_curriculum_state=restored_states.append,
    )
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry["_env"] = SimpleNamespace(unwrapped=environment)
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=41)
    checkpoint["command_curriculum_state"] = {"training_iteration": 41}
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return True

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match="training_iteration.*expected 42, received 41"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert upstream_load_called is False
    assert restored_states == []


def test_enabled_command_curriculum_rejects_checkpoint_without_state(monkeypatch):
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry["_env"] = SimpleNamespace(
        unwrapped=SimpleNamespace(
            get_command_curriculum_state=lambda: {"weights": torch.ones(1)},
            load_command_curriculum_state=lambda _state: None,
        )
    )
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=0)
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return True

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match="command_curriculum_state.*silently restart"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert upstream_load_called is False


def test_disabled_command_curriculum_rejects_full_resume_with_saved_state(monkeypatch):
    algorithm = _schedule_algorithm(0)
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=0)
    checkpoint["command_curriculum_state"] = {"weights": torch.ones(1)}
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return True

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match="contains command_curriculum_state.*disabled"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert upstream_load_called is False


def test_actor_only_inference_does_not_restore_or_require_curriculum_runtime(monkeypatch):
    restored_states = []
    received_environment_iterations = []
    algorithm = _schedule_algorithm(9)
    algorithm.symmetry["_env"] = SimpleNamespace(
        unwrapped=SimpleNamespace(
            get_command_curriculum_state=lambda: {"active_num_envs": 1},
            load_command_curriculum_state=restored_states.append,
            set_training_iteration=received_environment_iterations.append,
        )
    )
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=41)
    checkpoint["command_curriculum_state"] = {"saved_num_envs": 4096}
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: False)

    loaded_iteration = algorithm.load(
        checkpoint,
        load_cfg={"actor": True, "iteration": False, "environment_iteration": True},
        strict=True,
    )

    assert loaded_iteration is False
    assert restored_states == []
    assert received_environment_iterations == [42]


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


def test_rejected_augmentation_checkpoint_does_not_mutate_upstream_state(monkeypatch):
    class RejectingAugmentation:
        def __init__(self):
            self.value = torch.tensor(3.0)

        def state_dict(self):
            return {"accepted": True, "value": self.value.clone()}

        def load_state_dict(self, state):
            self.value.copy_(state["value"])
            if not state["accepted"]:
                raise ValueError("invalid augmentation checkpoint")

    algorithm = _schedule_algorithm(0)
    algorithm.actor = torch.nn.Linear(2, 2)
    algorithm.critic = torch.nn.Linear(2, 1)
    algorithm.optimizer = torch.optim.SGD((*algorithm.actor.parameters(), *algorithm.critic.parameters()), lr=0.1)
    algorithm._tr_augmentation = RejectingAugmentation()
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=0)
    checkpoint["time_reversal_augmentation_state"] = {
        "accepted": False,
        "value": torch.tensor(9.0),
        "schedule_iteration": 1,
    }
    actor_before = copy.deepcopy(algorithm.actor.state_dict())
    critic_before = copy.deepcopy(algorithm.critic.state_dict())
    optimizer_before = copy.deepcopy(algorithm.optimizer.state_dict())
    upstream_load_called = False

    def mutating_upstream_load(target, *_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        with torch.no_grad():
            for parameter in (*target.actor.parameters(), *target.critic.parameters()):
                parameter.fill_(42.0)
        target.optimizer.param_groups[0]["lr"] = 42.0
        return True

    monkeypatch.setattr(PPO, "load", mutating_upstream_load)

    with pytest.raises(ValueError, match="invalid augmentation checkpoint"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert upstream_load_called is False
    assert all(torch.equal(value, actor_before[name]) for name, value in algorithm.actor.state_dict().items())
    assert all(torch.equal(value, critic_before[name]) for name, value in algorithm.critic.state_dict().items())
    assert algorithm.optimizer.state_dict() == optimizer_before
    assert algorithm._tr_augmentation.value.item() == 3.0


def test_checkpoint_without_time_reversal_state_retains_legacy_iteration_fallback(monkeypatch):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = {"iter": 499}
    monkeypatch.setattr(PPO, "load", lambda *_args, **_kwargs: True)

    algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert checkpoint["iter"] == 500
    assert algorithm.current_learning_iteration == 500


def test_legacy_72d_checkpoint_is_rejected_before_upstream_load(monkeypatch):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = {
        "iter": 499,
        "actor_state_dict": {"architecture.0.weight": torch.zeros(12, 72)},
    }
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return True

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match="expected 64, received 72.*not migrated"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert upstream_load_called is False


def test_declared_policy_contract_does_not_mask_serialized_actor_width_mismatch(monkeypatch):
    algorithm = _schedule_algorithm(0, warmup_iterations=500, rampup_iterations=0)
    checkpoint = _time_reversal_checkpoint(algorithm, completed_iteration=499)
    assert checkpoint["time_reversal_state"]["policy_contract"]["policy_input_dim"] == 64
    checkpoint["actor_state_dict"] = {"architecture.0.weight": torch.zeros(12, 72)}
    upstream_load_called = False

    def upstream_load(*_args, **_kwargs):
        nonlocal upstream_load_called
        upstream_load_called = True
        return True

    monkeypatch.setattr(PPO, "load", upstream_load)

    with pytest.raises(ValueError, match="declared policy contract.*expected 64, received 72"):
        algorithm.load(checkpoint, load_cfg=None, strict=True)

    assert upstream_load_called is False


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


def test_none_consistency_mode_disables_history_auxiliary_losses():
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(history_enabled=True, history_length=30, tr_consistency_mode="none")

    algorithm._validate_time_reversal_configuration()
    assert algorithm._effective_time_reversal_coefficients() == (0.0, 0.0, 0.0)


def test_history_and_model_sidecar_are_rejected_before_allocation():
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(
        history_enabled=True,
        history_length=30,
        history_trs_mode="framewise_feature",
        tr_augmentation={"enabled": True, "mode": "dynamics_filtered_reverse_action_supervision"},
    )

    with pytest.raises(ValueError, match="future samples.*--no-history"):
        algorithm._validate_time_reversal_configuration()


def test_distributed_ppo_rejects_active_command_curriculum(monkeypatch):
    environment = SimpleNamespace(get_command_curriculum_state=lambda: {"training_iteration": 0})

    def initialize_base(algorithm, *_args, **_kwargs):
        algorithm.symmetry = {
            "_env": SimpleNamespace(unwrapped=environment),
            "tr_augmentation": {"enabled": False},
        }
        algorithm.is_multi_gpu = True
        algorithm.device = "cpu"
        algorithm.actor = _DummyActor()

    monkeypatch.setattr(PPO, "__init__", initialize_base)
    monkeypatch.setattr(TimeReversalPPO, "_validate_time_reversal_configuration", lambda _algorithm: None)

    with pytest.raises(ValueError, match="TR-orbit command curriculum.*distributed PPO"):
        TimeReversalPPO()


def test_policy_observation_width_reports_required_history_flags():
    algorithm = _schedule_algorithm(0)
    algorithm.symmetry.update(history_enabled=True, history_length=30, history_trs_mode="framewise_feature")
    algorithm._policy_observation_width_validated = False

    with pytest.raises(ValueError, match="expected 1920, received 64.*--history --history-length 30"):
        algorithm._validate_policy_observation_width(torch.zeros(2, SYMM_QUADRUPED_POLICY_OBS_DIM))


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
        "min_abs_command_velocity": 0.0,
    }
    observations = {"policy": torch.zeros(3, SYMM_QUADRUPED_POLICY_OBS_DIM)}

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
