# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-reversal PPO regularization for symmetric quadruped RSL-RL tasks."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

_TRS_RAMP_SHAPES = frozenset({"linear", "half_cosine"})


def time_reversal_loss_scale(
    iteration: int,
    warmup_iterations: int,
    rampup_iterations: int,
    ramp_shape: str,
) -> float:
    """Return the time-reversal auxiliary-loss scale for one PPO update.

    Args:
        iteration: Absolute zero-based PPO learning iteration.
        warmup_iterations: Number of fully unregularized PPO updates before the ramp starts.
        rampup_iterations: Number of PPO updates in the coefficient ramp. Zero selects a hard switch.
        ramp_shape: Ramp interpolation shape, either ``"linear"`` or ``"half_cosine"``.

    Returns:
        The coefficient scale in the closed interval [0, 1].

    Raises:
        ValueError: If a schedule setting is invalid.
    """
    for name, value in (
        ("warmup_iterations", warmup_iterations),
        ("rampup_iterations", rampup_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")
    if ramp_shape not in _TRS_RAMP_SHAPES:
        raise ValueError(f"ramp_shape must be one of {sorted(_TRS_RAMP_SHAPES)!r}; received {ramp_shape!r}.")

    if iteration < warmup_iterations:
        return 0.0
    if rampup_iterations == 0:
        return 1.0

    progress = min(max((iteration - warmup_iterations) / rampup_iterations, 0.0), 1.0)
    if ramp_shape == "linear":
        return progress
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def time_reversal_weighted_losses(
    effective_mirror_coeff: float,
    effective_value_coeff: float,
    mean_symmetry_loss: float | None,
    mean_tr_value_loss: float | None,
) -> tuple[float, float, float]:
    """Return weighted time-reversal loss contributions for per-update logging.

    Args:
        effective_mirror_coeff: Effective policy-equivariance loss coefficient.
        effective_value_coeff: Effective value-consistency loss coefficient.
        mean_symmetry_loss: Mean raw policy-equivariance loss for the PPO update.
        mean_tr_value_loss: Mean raw value-consistency loss for the PPO update.

    Returns:
        Weighted policy, weighted value, and total auxiliary contributions.
    """
    weighted_symmetry = (
        effective_mirror_coeff * mean_symmetry_loss
        if effective_mirror_coeff > 0.0 and mean_symmetry_loss is not None
        else 0.0
    )
    weighted_tr_value = (
        effective_value_coeff * mean_tr_value_loss
        if effective_value_coeff > 0.0 and mean_tr_value_loss is not None
        else 0.0
    )
    return weighted_symmetry, weighted_tr_value, weighted_symmetry + weighted_tr_value


class TimeReversalPPO(PPO):
    """PPO with scheduled auxiliary time-reversal policy and value losses."""

    _MIN_ACTOR_STD = 1.0e-6
    _MAX_ACTOR_STD = 1.0
    _ACTOR_MEAN_BOUND = 10.0
    _ACTOR_MEAN_BOUND_LOSS_COEFF = 1.0e-2
    _ACTOR_MEAN_ABORT_BOUND = 50.0
    _ACTOR_MEAN_ABORT_PATIENCE = 25

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._validate_time_reversal_configuration()
        self.current_learning_iteration = 0
        self._time_reversal_update_count = 0
        self._actor_mean_abort_count = 0
        self._clamp_actor_std()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions after ensuring the scalar action std is valid."""
        self._clamp_actor_std()
        return super().act(obs)

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO updates with optional time-reversal regularization."""
        time_reversal_enabled = self._time_reversal_enabled()
        trs_scale, effective_mirror_coeff, effective_tr_value_coeff = self._effective_time_reversal_coefficients()
        time_reversal_active = (
            time_reversal_enabled
            and trs_scale > 0.0
            and (
                bool(self.symmetry["use_data_augmentation"])
                or effective_mirror_coeff > 0.0
                or effective_tr_value_coeff > 0.0
            )
        )

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_actor_bound_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_tr_value_loss = 0 if time_reversal_enabled else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            use_data_augmentation = time_reversal_active and self.symmetry and self.symmetry["use_data_augmentation"]
            if use_data_augmentation:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self._clamp_actor_std()
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]
            actor_bound_loss = self._actor_mean_bound_loss(distribution_params[0])

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss += self._ACTOR_MEAN_BOUND_LOSS_COEFF * actor_bound_loss

            symmetry_loss = torch.zeros((), device=self.device)
            tr_value_loss = torch.zeros((), device=self.device)
            if time_reversal_active and self.symmetry:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                augmented_observations = batch.observations
                augmented_values = values
                if not use_data_augmentation:
                    augmented_observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                    augmented_values = self.critic(augmented_observations)

                mean_actions = self.actor(augmented_observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                time_reversal_mask = self._time_reversal_mask(augmented_observations[:original_batch_size])
                symmetry_loss = self._masked_mse(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                    time_reversal_mask,
                )
                tr_value_loss = self._masked_mse(
                    augmented_values[original_batch_size:],
                    augmented_values[:original_batch_size].detach(),
                    time_reversal_mask,
                )

                if effective_mirror_coeff > 0.0:
                    loss += effective_mirror_coeff * symmetry_loss
                if effective_tr_value_coeff > 0.0:
                    loss += effective_tr_value_coeff * tr_value_loss

            elif self.symmetry:
                symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_actor_std()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_actor_bound_loss += actor_bound_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if mean_tr_value_loss is not None:
                mean_tr_value_loss += tr_value_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_actor_bound_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_tr_value_loss is not None:
            mean_tr_value_loss /= num_updates

        weighted_symmetry, weighted_tr_value, weighted_trs_total = time_reversal_weighted_losses(
            effective_mirror_coeff,
            effective_tr_value_coeff,
            mean_symmetry_loss,
            mean_tr_value_loss,
        )

        action_diagnostics = self._action_diagnostics_from_storage()
        self._update_actor_mean_safety(action_diagnostics["diagnostics/actor_mean_abs_max"])
        self.storage.clear()
        self._time_reversal_update_count += 1
        self.current_learning_iteration = self._time_reversal_update_count

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_bound": mean_actor_bound_loss,
            "trs_scale": trs_scale,
            "effective_mirror_coeff": effective_mirror_coeff,
            "effective_tr_value_coeff": effective_tr_value_coeff,
            "weighted_symmetry": weighted_symmetry,
            "weighted_tr_value": weighted_tr_value,
            "weighted_trs_total": weighted_trs_total,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_tr_value_loss is not None:
            loss_dict["tr_value"] = mean_tr_value_loss
        loss_dict.update(action_diagnostics)

        return loss_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load algorithm state and align the TRS schedule to the next PPO update.

        RSL-RL 5.0.1 saves ``iter`` after completing that indexed update. The
        loaded mapping is ephemeral, so it is advanced here before the runner
        restores its counter. This keeps the runner, TensorBoard step, and
        auxiliary schedule aligned to the next absolute update.
        """
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and "iter" in loaded_dict:
            next_iteration = int(loaded_dict["iter"]) + 1
            loaded_dict["iter"] = next_iteration
            self.current_learning_iteration = next_iteration
            self._time_reversal_update_count = self.current_learning_iteration
        return load_iteration

    def _validate_time_reversal_configuration(self) -> None:
        """Validate target coefficients and schedule settings without mutating them."""
        if self.symmetry is None:
            return

        for name in ("mirror_loss_coeff", "value_loss_coeff"):
            value = self.symmetry.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative; received {value!r}.")

        time_reversal_loss_scale(
            iteration=0,
            warmup_iterations=self.symmetry.get("warmup_iterations", 0),
            rampup_iterations=self.symmetry.get("rampup_iterations", 0),
            ramp_shape=self.symmetry.get("ramp_shape", "linear"),
        )

    def _effective_time_reversal_coefficients(self) -> tuple[float, float, float]:
        """Return the shared schedule scale and effective actor/value coefficients."""
        if not self._time_reversal_enabled():
            return 0.0, 0.0, 0.0

        scale = time_reversal_loss_scale(
            iteration=self.current_learning_iteration,
            warmup_iterations=self.symmetry.get("warmup_iterations", 0),
            rampup_iterations=self.symmetry.get("rampup_iterations", 0),
            ramp_shape=self.symmetry.get("ramp_shape", "linear"),
        )
        mirror_coeff = (
            float(self.symmetry.get("mirror_loss_coeff", 0.0)) * scale if self.symmetry["use_mirror_loss"] else 0.0
        )
        value_coeff = float(self.symmetry.get("value_loss_coeff", 0.0)) * scale
        return scale, mirror_coeff, value_coeff

    def _time_reversal_enabled(self) -> bool:
        if self.symmetry is None:
            return False
        if not self.symmetry.get("use_time_reversal_regularization", False):
            return False
        return (
            self.symmetry["use_data_augmentation"]
            or self.symmetry["use_mirror_loss"]
            or float(self.symmetry.get("value_loss_coeff", 0.0)) > 0.0
        )

    def _warmup_iterations(self) -> int:
        if self.symmetry is None:
            return 0
        return int(self.symmetry.get("warmup_iterations", 0))

    def _time_reversal_mask(self, observations) -> torch.Tensor:
        policy_obs = observations["policy"]
        command_index = int(self.symmetry.get("command_observation_index", 9))
        command_scale = float(self.symmetry.get("command_observation_scale", 1.0))
        min_abs_command = float(self.symmetry.get("min_abs_command_velocity", 0.0))
        command = policy_obs[:, command_index] / command_scale
        return (torch.abs(command) >= min_abs_command).unsqueeze(-1).to(dtype=policy_obs.dtype)

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(prediction)
        denom = mask.sum().clamp_min(1.0)
        selected = mask > 0.0
        safe_prediction = torch.where(selected, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(selected, target, torch.zeros_like(target))
        squared_error = (safe_prediction - safe_target).pow(2)
        return squared_error.sum() / denom

    def _clamp_actor_std(self) -> None:
        """Keep RSL-RL's scalar Gaussian std parameter positive and finite."""
        distribution = getattr(self.actor, "distribution", None)
        std_param = getattr(distribution, "std_param", None)
        if std_param is None:
            return
        with torch.no_grad():
            std_param.nan_to_num_(nan=self._MIN_ACTOR_STD, posinf=self._MAX_ACTOR_STD, neginf=self._MIN_ACTOR_STD)
            std_param.clamp_(min=self._MIN_ACTOR_STD, max=self._MAX_ACTOR_STD)

    @classmethod
    def _actor_mean_bound_loss(cls, actor_mean: torch.Tensor) -> torch.Tensor:
        """Penalize actor means only after they exceed the normal locomotion range."""
        excess = torch.relu(actor_mean.abs() - cls._ACTOR_MEAN_BOUND)
        return excess.square().mean()

    def _update_actor_mean_safety(self, actor_mean_abs_max: float) -> None:
        """Abort training after sustained actor-mean divergence."""
        if actor_mean_abs_max <= self._ACTOR_MEAN_ABORT_BOUND:
            self._actor_mean_abort_count = 0
            return

        self._actor_mean_abort_count += 1
        if self._actor_mean_abort_count >= self._ACTOR_MEAN_ABORT_PATIENCE:
            raise RuntimeError(
                "actor mean diverged above "
                f"{self._ACTOR_MEAN_ABORT_BOUND:g} for {self._ACTOR_MEAN_ABORT_PATIENCE} consecutive PPO updates; "
                "stop this run and restart training from a stable checkpoint or a fresh policy"
            )

    def _action_diagnostics_from_storage(self) -> dict[str, float]:
        """Summarize exact sampled actions and actor means retained by rollout storage."""
        sampled_action_abs = self.storage.actions.abs()
        actor_mean_abs = self.storage.distribution_params[0].abs()
        return {
            "diagnostics/action_abs_mean": sampled_action_abs.mean().item(),
            "diagnostics/action_abs_max": sampled_action_abs.max().item(),
            "diagnostics/actor_mean_abs_mean": actor_mean_abs.mean().item(),
            "diagnostics/actor_mean_abs_max": actor_mean_abs.max().item(),
        }
