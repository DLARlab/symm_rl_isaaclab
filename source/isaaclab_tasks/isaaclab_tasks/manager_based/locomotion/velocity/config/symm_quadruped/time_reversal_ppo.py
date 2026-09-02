# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-reversal PPO regularization for symmetric quadruped RSL-RL tasks."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from numbers import Integral, Real

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_augmentation import (
    TimeReversalAugmentation,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_core import (
    ResolvedTimeReversalSchedule,
    feasible_actor_mean_diagnostics,
    feasible_actor_mean_penalty,
    normalize_requested_joint_targets,
    resolve_time_reversal_schedule,
    time_reversal_gradient_diagnostics,
    time_reversal_mask_diagnostics,
    time_reversal_schedule_scale,
    time_reversal_validity_mask,
    validate_legacy_observation_metadata,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION,
    SYMM_QUADRUPED_OBSERVATION_CONTRACT_VERSION,
    SYMM_QUADRUPED_PHASE_MAPPING_VERSION,
    SYMM_QUADRUPED_POLICY_OBS_DIM,
    SYMM_QUADRUPED_POLICY_OBSERVATION_CONTRACT,
    infer_policy_history_length,
)


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
    return time_reversal_schedule_scale(
        iteration=iteration,
        warmup_iterations=warmup_iterations,
        rampup_iterations=rampup_iterations,
        hold_iterations=0,
        decay_iterations=0,
        final_scale=1.0,
        ramp_shape=ramp_shape,
    )


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
    _TIME_REVERSAL_STATE_SCHEMA_VERSION = 3

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._joint_action_metadata_cache: tuple[torch.Tensor, ...] | None = None
        self._validate_time_reversal_configuration()
        self.current_learning_iteration = 0
        self._time_reversal_update_count = 0
        self._actor_mean_abort_count = 0
        self._policy_observation_width_validated = False
        self._tr_augmentation: TimeReversalAugmentation | None = None
        augmentation_cfg = self.symmetry.get("tr_augmentation", {}) if self.symmetry else {}
        if augmentation_cfg.get("enabled", False):
            if self.is_multi_gpu:
                raise ValueError(
                    "tr_augmentation is not supported in distributed PPO until side-model synchronization is "
                    "implemented. Disable augmentation or train on one process."
                )
            self._tr_augmentation = TimeReversalAugmentation(self.symmetry["_env"], augmentation_cfg, self.device)
        self._clamp_actor_std()
        self._sync_environment_training_iteration()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions after ensuring the scalar action std is valid."""
        self._validate_policy_observation_width(obs["policy"])
        self._clamp_actor_std()
        actions = super().act(obs)
        augmentation = getattr(self, "_tr_augmentation", None)
        if augmentation is not None:
            augmentation.capture_before_step(
                obs,
                actions,
                self.actor.output_distribution_params[0],
                self.actor.output_std,
            )
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Record the standard PPO transition and an optional authentic successor sidecar."""
        augmentation = getattr(self, "_tr_augmentation", None)
        if augmentation is not None:
            augmentation.capture_after_step(obs, rewards, dones, extras)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO updates with optional time-reversal regularization."""
        policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
        trs_scale, effective_tr_policy_coeff, effective_tr_value_coeff = self._effective_time_reversal_coefficients()
        legacy_data_augmentation_scale = self._legacy_data_augmentation_scale()
        legacy_data_augmentation_active = legacy_data_augmentation_scale > 0.0
        trs_scale = max(trs_scale, legacy_data_augmentation_scale)
        log_disabled_raw = bool(self.symmetry and self.symmetry.get("log_disabled_raw_consistency", False))
        compute_policy_raw = effective_tr_policy_coeff > 0.0 or log_disabled_raw
        compute_value_raw = effective_tr_value_coeff > 0.0 or log_disabled_raw
        gradient_cfg = self.symmetry.get("tr_gradient_diagnostics", {}) if self.symmetry else {}
        gradient_diagnostics_enabled = bool(gradient_cfg.get("enabled", False))
        gradient_diagnostics_update = gradient_diagnostics_enabled and (
            self.current_learning_iteration % int(gradient_cfg.get("interval", 100)) == 0
        )
        augmentation_schedule = self._resolved_time_reversal_augmentation_schedule()
        effective_tr_augmentation_coeff = augmentation_schedule.coefficient(self.current_learning_iteration)
        augmentation = getattr(self, "_tr_augmentation", None)
        augmentation_cfg = self.symmetry.get("tr_augmentation", {}) if self.symmetry else {}
        augmentation_gradient_update = augmentation is not None and (
            self.current_learning_iteration % int(augmentation_cfg.get("gradient_diagnostics_interval", 100)) == 0
        )
        augmentation_diagnostics = augmentation.prepare_update() if augmentation is not None else {}

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_actor_bound_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_raw_action_consistency = 0.0
        mean_normalized_target_consistency = 0.0
        raw_action_consistency_count = 0
        normalized_target_consistency_count = 0
        mean_tr_value_loss = 0 if self.symmetry else None
        mean_tr_augmentation_loss = 0.0
        mask_diagnostics_sum: dict[str, float] = {}
        mask_diagnostics_count = 0
        gradient_diagnostics: dict[str, float] = {}

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch_index, batch in enumerate(generator):
            original_batch_size = batch.observations.batch_size[0]
            original_observations = batch.observations

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if legacy_data_augmentation_active:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=original_observations,
                    actions=batch.actions,
                )
                augmented_batch_size = batch.observations.batch_size[0]
                if augmented_batch_size % original_batch_size != 0 or batch.actions.shape[0] != augmented_batch_size:
                    raise RuntimeError(
                        "Legacy time-reversal data augmentation must return equally sized observation and action "
                        f"batches containing the originals; got {augmented_batch_size} observations and "
                        f"{batch.actions.shape[0]} actions for {original_batch_size} inputs."
                    )
                augmentation_factor = augmented_batch_size // original_batch_size
                for field_name in ("old_actions_log_prob", "values", "advantages", "returns"):
                    field = getattr(batch, field_name)
                    repeats = (augmentation_factor,) + (1,) * (field.ndim - 1)
                    setattr(batch, field_name, field.repeat(repeats))

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
            actor_bound_loss = self._configured_actor_mean_bound_loss(distribution_params[0])

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
            raw_action_consistency_loss = torch.zeros((), device=self.device)
            normalized_target_consistency_loss: torch.Tensor | None = None
            tr_value_loss = torch.zeros((), device=self.device)
            actor_diagnostic_loss = surrogate_loss - self.entropy_coef * entropy.mean()
            critic_diagnostic_loss = self.value_loss_coef * value_loss
            diagnostic_minibatch = gradient_diagnostics_update and batch_index == 0
            augmentation_diagnostic_minibatch = augmentation_gradient_update and batch_index == 0
            need_policy_raw = compute_policy_raw or (diagnostic_minibatch and policy_schedule.enabled)
            need_value_raw = compute_value_raw or (diagnostic_minibatch and value_schedule.enabled)
            if self.symmetry and (need_policy_raw or need_value_raw):
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                if legacy_data_augmentation_active:
                    augmented_observations = batch.observations
                else:
                    augmented_observations, _ = data_augmentation_func(
                        obs=original_observations, actions=None, env=self.symmetry["_env"]
                    )
                if augmented_observations.batch_size[0] != 2 * original_batch_size:
                    raise RuntimeError(
                        "Time-reversal consistency requires exactly one transformed observation per historical "
                        f"sample; got {augmented_observations.batch_size[0]} outputs for {original_batch_size} inputs."
                    )
                transformed_observations = augmented_observations[original_batch_size:]
                validity = self._time_reversal_validity(original_observations)
                time_reversal_mask = validity.combined
                minibatch_mask_diagnostics = time_reversal_mask_diagnostics(validity)
                for name, value in minibatch_mask_diagnostics.items():
                    mask_diagnostics_sum[name] = mask_diagnostics_sum.get(name, 0.0) + value
                mask_diagnostics_count += 1

                if need_policy_raw:
                    transformed_action_mean = self.actor(transformed_observations.detach().clone())
                    _, augmented_action_targets = data_augmentation_func(
                        obs=None,
                        actions=distribution_params[0].detach(),
                        env=self.symmetry["_env"],
                    )
                    if augmented_action_targets.shape[0] != 2 * original_batch_size:
                        raise RuntimeError(
                            "Time-reversal consistency requires exactly one transformed action per historical "
                            f"sample; got {augmented_action_targets.shape[0]} outputs for {original_batch_size} inputs."
                        )
                    raw_action_consistency_loss = self._masked_mse(
                        transformed_action_mean,
                        augmented_action_targets[original_batch_size:].detach(),
                        time_reversal_mask,
                    )
                    metadata = self._joint_action_metadata(required=False)
                    if metadata is not None:
                        action_offset, action_scale, soft_limits, _, _ = metadata
                        normalized_original = normalize_requested_joint_targets(
                            distribution_params[0].detach(),
                            action_offset,
                            action_scale,
                            soft_limits,
                        )
                        normalized_transformed = normalize_requested_joint_targets(
                            transformed_action_mean,
                            action_offset,
                            action_scale,
                            soft_limits,
                        )
                        _, augmented_normalized_targets = data_augmentation_func(
                            obs=None,
                            actions=normalized_original,
                            env=self.symmetry["_env"],
                        )
                        if augmented_normalized_targets.shape[0] != 2 * original_batch_size:
                            raise RuntimeError(
                                "Normalized-target time-reversal consistency requires exactly one transformed "
                                f"target per historical sample; got {augmented_normalized_targets.shape[0]} outputs "
                                f"for {original_batch_size} inputs."
                            )
                        normalized_target_consistency_loss = self._masked_mse(
                            normalized_transformed,
                            augmented_normalized_targets[original_batch_size:].detach(),
                            time_reversal_mask,
                        )
                    output_space = self.symmetry.get("tr_policy_output_space", "raw_action_mean")
                    if output_space == "raw_action_mean":
                        symmetry_loss = raw_action_consistency_loss
                    elif output_space == "normalized_requested_joint_target":
                        if normalized_target_consistency_loss is None:
                            raise RuntimeError(
                                "tr_policy_output_space='normalized_requested_joint_target' requires ordered joint "
                                "action offsets, scales, and soft limits from the symmetric quadruped environment."
                            )
                        symmetry_loss = normalized_target_consistency_loss
                    else:
                        raise ValueError(f"Unsupported tr_policy_output_space: {output_space!r}.")
                if need_value_raw:
                    transformed_values = (
                        values[original_batch_size:]
                        if legacy_data_augmentation_active
                        else self.critic(transformed_observations)
                    )
                    tr_value_loss = self._masked_mse(
                        transformed_values,
                        values[:original_batch_size].detach(),
                        time_reversal_mask,
                    )

                if effective_tr_policy_coeff > 0.0:
                    loss += effective_tr_policy_coeff * symmetry_loss
                if effective_tr_value_coeff > 0.0:
                    loss += effective_tr_value_coeff * tr_value_loss

                if diagnostic_minibatch:
                    epsilon = float(gradient_cfg.get("epsilon", 1.0e-12))
                    if policy_schedule.enabled and need_policy_raw:
                        gradient_diagnostics.update(
                            time_reversal_gradient_diagnostics(
                                actor_diagnostic_loss,
                                symmetry_loss,
                                tuple(self.actor.parameters()),
                                effective_coefficient=effective_tr_policy_coeff,
                                prefix="actor",
                                epsilon=epsilon,
                            )
                        )
                    if value_schedule.enabled and need_value_raw:
                        gradient_diagnostics.update(
                            time_reversal_gradient_diagnostics(
                                critic_diagnostic_loss,
                                tr_value_loss,
                                tuple(self.critic.parameters()),
                                effective_coefficient=effective_tr_value_coeff,
                                prefix="critic",
                                epsilon=epsilon,
                            )
                        )

            tr_augmentation_loss = torch.zeros((), device=self.device)
            if augmentation is not None and effective_tr_augmentation_coeff > 0.0:
                tr_augmentation_loss = augmentation.actor_nll(self.actor, original_batch_size)
                loss += effective_tr_augmentation_coeff * tr_augmentation_loss
                if augmentation_diagnostic_minibatch and tr_augmentation_loss.requires_grad:
                    augmentation_gradient = time_reversal_gradient_diagnostics(
                        actor_diagnostic_loss,
                        tr_augmentation_loss,
                        tuple(self.actor.parameters()),
                        effective_coefficient=effective_tr_augmentation_coeff,
                        prefix="augmentation_actor",
                        epsilon=float(augmentation_cfg.get("gradient_diagnostics_epsilon", 1.0e-12)),
                    )
                    gradient_diagnostics.update(augmentation_gradient)
                    augmentation_diagnostics["tr_augmentation/gradient_norm"] = augmentation_gradient[
                        "tr_gradient/augmentation_actor/auxiliary_norm"
                    ]
                    augmentation_diagnostics["tr_augmentation/gradient_weighted_norm"] = augmentation_gradient[
                        "tr_gradient/augmentation_actor/weighted_auxiliary_norm"
                    ]
                    augmentation_diagnostics["tr_augmentation/gradient_cosine_ppo_actor"] = augmentation_gradient[
                        "tr_gradient/augmentation_actor/cosine"
                    ]
                    augmentation_diagnostics["tr_augmentation/gradient_diagnostics_ran"] = 1.0

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(original_observations)
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
            if mean_symmetry_loss is not None and compute_policy_raw:
                mean_symmetry_loss += symmetry_loss.item()
                mean_raw_action_consistency += raw_action_consistency_loss.item()
                raw_action_consistency_count += 1
                if normalized_target_consistency_loss is not None:
                    mean_normalized_target_consistency += normalized_target_consistency_loss.item()
                    normalized_target_consistency_count += 1
            if mean_tr_value_loss is not None and compute_value_raw:
                mean_tr_value_loss += tr_value_loss.item()
            mean_tr_augmentation_loss += tr_augmentation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_actor_bound_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if raw_action_consistency_count:
            mean_raw_action_consistency /= raw_action_consistency_count
        if normalized_target_consistency_count:
            mean_normalized_target_consistency /= normalized_target_consistency_count
        if mean_tr_value_loss is not None:
            mean_tr_value_loss /= num_updates
        mean_tr_augmentation_loss /= num_updates

        weighted_symmetry, weighted_tr_value, weighted_trs_total = time_reversal_weighted_losses(
            effective_tr_policy_coeff,
            effective_tr_value_coeff,
            mean_symmetry_loss,
            mean_tr_value_loss,
        )

        action_diagnostics = self._action_diagnostics_from_storage()
        self._update_actor_mean_safety(action_diagnostics["diagnostics/actor_mean_abs_max"])
        if augmentation is not None:
            augmentation.finish_update()
        self.storage.clear()
        self._time_reversal_update_count += 1
        self.current_learning_iteration = self._time_reversal_update_count
        self._sync_environment_training_iteration()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_bound": mean_actor_bound_loss,
            "trs_scale": trs_scale,
            "tr_policy_consistency": mean_symmetry_loss or 0.0,
            "tr_policy_consistency_raw_action_mean": mean_raw_action_consistency,
            "tr_policy_consistency_normalized_requested_joint_target": mean_normalized_target_consistency,
            "tr_policy_consistency_normalized_target_available": float(normalized_target_consistency_count > 0),
            "tr_value_consistency": mean_tr_value_loss or 0.0,
            "effective_tr_policy_coeff": effective_tr_policy_coeff,
            "effective_tr_value_coeff": effective_tr_value_coeff,
            "weighted_tr_policy": weighted_symmetry,
            "weighted_tr_value": weighted_tr_value,
            "weighted_tr_total": weighted_trs_total,
            "tr_augmentation_nll": mean_tr_augmentation_loss,
            "effective_tr_augmentation_coeff": effective_tr_augmentation_coeff,
            "weighted_tr_augmentation": effective_tr_augmentation_coeff * mean_tr_augmentation_loss,
            "history_enabled": float(self._history_enabled()),
            "history_length": float(self._configured_history_length()),
            "policy_input_dim": float(self._expected_policy_input_dim()),
            "instantaneous_frame_dim": float(SYMM_QUADRUPED_POLICY_OBS_DIM),
            "raw_tr_policy_residual": mean_symmetry_loss or 0.0,
            "weighted_tr_policy_contribution": weighted_symmetry,
            "raw_tr_value_residual": mean_tr_value_loss or 0.0,
            "weighted_tr_value_contribution": weighted_tr_value,
            f"observation_contract_version/{SYMM_QUADRUPED_OBSERVATION_CONTRACT_VERSION}": 1.0,
            f"history_trs_mode/{self._history_trs_mode()}": 1.0,
            # Deprecated TensorBoard aliases retained for archived analyses.
            "effective_mirror_coeff": effective_tr_policy_coeff,
            "weighted_symmetry": weighted_symmetry,
            "weighted_trs_total": weighted_trs_total,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_tr_value_loss is not None:
            loss_dict["tr_value"] = mean_tr_value_loss
        if mask_diagnostics_count:
            loss_dict.update({name: value / mask_diagnostics_count for name, value in mask_diagnostics_sum.items()})
        loss_dict.update(gradient_diagnostics)
        loss_dict.update(augmentation_diagnostics)
        loss_dict.update(action_diagnostics)

        return loss_dict

    def _history_enabled(self) -> bool:
        """Return the policy-history switch from the resolved symmetry config."""
        return bool(self.symmetry and self.symmetry.get("history_enabled", False))

    def _configured_history_length(self) -> int:
        """Return the configured native history length, using zero for one-frame input."""
        if not self._history_enabled():
            return 0
        return int(self.symmetry.get("history_length", 30))

    def _history_trs_mode(self) -> str:
        """Return the configured feature-level time-reversal history mode."""
        return str(self.symmetry.get("history_trs_mode", "framewise_feature")) if self.symmetry else "framewise_feature"

    def _expected_policy_input_dim(self) -> int:
        """Return the flattened actor/critic policy input width."""
        history_frames = self._configured_history_length() if self._history_enabled() else 1
        return history_frames * SYMM_QUADRUPED_POLICY_OBS_DIM

    def _policy_contract_metadata(self) -> dict[str, object]:
        """Return deterministic metadata required to interpret a policy checkpoint."""
        return {
            "observation_contract_version": SYMM_QUADRUPED_OBSERVATION_CONTRACT_VERSION,
            "instantaneous_frame_dim": SYMM_QUADRUPED_POLICY_OBS_DIM,
            "history_enabled": self._history_enabled(),
            "history_length": self._configured_history_length(),
            "history_packing": SYMM_QUADRUPED_POLICY_OBSERVATION_CONTRACT.history_packing,
            "history_trs_mode": self._history_trs_mode(),
            "policy_input_dim": self._expected_policy_input_dim(),
            "gait_phase_mapping_version": SYMM_QUADRUPED_PHASE_MAPPING_VERSION,
            "gait_library_version": SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION,
        }

    def _validate_policy_observation_width(self, policy_obs: torch.Tensor) -> None:
        """Fail before rollout when active observations do not match the policy contract."""
        if getattr(self, "_policy_observation_width_validated", False):
            return
        width = policy_obs.shape[-1]
        received_history_length = infer_policy_history_length(policy_obs)
        expected_width = self._expected_policy_input_dim()
        expected_history_length = self._configured_history_length() if self._history_enabled() else 1
        if width != expected_width or received_history_length != expected_history_length:
            required_flag = (
                f"--history --history-length {expected_history_length}" if self._history_enabled() else "--no-history"
            )
            raise ValueError(
                "Policy observation width is incompatible with the active checkpoint/config contract: "
                f"expected {expected_width}, received {width}; observation_contract_version="
                f"{SYMM_QUADRUPED_OBSERVATION_CONTRACT_VERSION!r}. Use {required_flag}."
            )
        self._policy_observation_width_validated = True

    def _validate_saved_policy_contract(self, saved_contract: object) -> None:
        """Validate checkpoint policy metadata without padding or projecting inputs."""
        expected = self._policy_contract_metadata()
        if not isinstance(saved_contract, Mapping):
            raise ValueError("Checkpoint policy_contract must be a mapping.")
        required = set(expected)
        if set(saved_contract) != required:
            missing = sorted(required - set(saved_contract))
            unexpected = sorted(set(saved_contract) - required)
            raise ValueError(
                f"Checkpoint policy_contract schema fields do not match: missing={missing}, unexpected={unexpected}."
            )
        saved_width = saved_contract["policy_input_dim"]
        if saved_width != expected["policy_input_dim"] or isinstance(saved_width, bool):
            required_flag = (
                f"--history --history-length {self._configured_history_length()}"
                if self._history_enabled()
                else "--no-history"
            )
            raise ValueError(
                "Checkpoint actor input width is incompatible with the active policy observation: "
                f"expected {expected['policy_input_dim']}, received {saved_width}; saved observation-contract "
                f"version={saved_contract.get('observation_contract_version')!r}. Use {required_flag}."
            )
        exact_match = all(
            type(saved_contract[name]) is type(expected[name]) and saved_contract[name] == expected[name]
            for name in expected
        )
        if not exact_match:
            raise ValueError(
                "Checkpoint policy_contract does not exactly match the active policy configuration: "
                f"expected {expected!r}, received {dict(saved_contract)!r}."
            )

    @staticmethod
    def _checkpoint_actor_input_width(loaded_dict: Mapping) -> int | None:
        """Best-effort extraction of a legacy checkpoint actor's first linear width."""
        for state_name in ("actor_state_dict", "model_state_dict"):
            state = loaded_dict.get(state_name)
            if not isinstance(state, Mapping):
                continue
            candidates: list[tuple[str, torch.Tensor]] = []
            for name, value in state.items():
                if not isinstance(value, torch.Tensor) or value.ndim != 2 or not name.endswith("weight"):
                    continue
                if state_name == "actor_state_dict" or "actor" in name.lower():
                    candidates.append((str(name), value))
            if candidates:
                _, first_weight = min(candidates, key=lambda item: item[0])
                return int(first_weight.shape[1])
        return None

    def _validate_checkpoint_policy_width(self, loaded_dict: Mapping) -> None:
        """Reject incompatible V5 and legacy actor widths before upstream loading."""
        state = loaded_dict.get("time_reversal_state")
        if isinstance(state, Mapping) and "policy_contract" in state:
            self._validate_saved_policy_contract(state["policy_contract"])
            return
        saved_width = self._checkpoint_actor_input_width(loaded_dict)
        if saved_width is None:
            return
        expected_width = self._expected_policy_input_dim()
        if saved_width != expected_width:
            required_flag = (
                f"--history --history-length {self._configured_history_length()}"
                if self._history_enabled()
                else "--no-history"
            )
            raise ValueError(
                "Legacy checkpoint actor input width is incompatible with the active policy observation: "
                f"expected {expected_width}, received {saved_width}; saved observation-contract version is "
                f"unavailable. Use {required_flag}. 72D checkpoints are not migrated automatically."
            )

    def _command_curriculum_state(self):
        """Return optional task-local command-curriculum checkpoint state."""
        symmetry = getattr(self, "symmetry", None)
        environment = symmetry.get("_env") if symmetry else None
        unwrapped = getattr(environment, "unwrapped", environment)
        getter = getattr(unwrapped, "get_command_curriculum_state", None)
        return getter() if callable(getter) else None

    def _restore_command_curriculum_state(self, state) -> None:
        """Restore optional task-local command-curriculum checkpoint state."""
        symmetry = getattr(self, "symmetry", None)
        environment = symmetry.get("_env") if symmetry else None
        unwrapped = getattr(environment, "unwrapped", environment)
        loader = getattr(unwrapped, "load_command_curriculum_state", None)
        if not callable(loader):
            raise ValueError("Enabled command curriculum does not expose load_command_curriculum_state().")
        loader(state)

    def save(self) -> dict:
        """Save PPO state with absolute schedules and the policy input contract."""
        saved_dict = super().save()
        policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
        augmentation_schedule = self._resolved_time_reversal_augmentation_schedule()
        saved_dict["time_reversal_state"] = {
            "schema_version": self._TIME_REVERSAL_STATE_SCHEMA_VERSION,
            "last_completed_update": self.current_learning_iteration - 1,
            "next_absolute_update": self.current_learning_iteration,
            "policy_schedule": vars(policy_schedule),
            "value_schedule": vars(value_schedule),
            "augmentation_schedule": vars(augmentation_schedule),
            "policy_contract": self._policy_contract_metadata(),
        }
        augmentation = getattr(self, "_tr_augmentation", None)
        if augmentation is not None:
            augmentation_state = augmentation.state_dict()
            augmentation_state["schedule_iteration"] = self.current_learning_iteration
            saved_dict["time_reversal_augmentation_state"] = augmentation_state
        curriculum_state = self._command_curriculum_state()
        if curriculum_state is not None:
            saved_dict["command_curriculum_state"] = curriculum_state
        return saved_dict

    def _validate_time_reversal_checkpoint_state(
        self,
        loaded_dict: Mapping,
        *,
        validate_schedules: bool = True,
    ) -> int | None:
        """Validate and return the next update for checkpoints carrying the publication schema."""
        if "time_reversal_state" not in loaded_dict:
            # Checkpoints predating the publication schema retain the legacy
            # iteration-only fallback below.
            return None
        state = loaded_dict["time_reversal_state"]
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint time_reversal_state must be a mapping.")
        common_required = {
            "schema_version",
            "last_completed_update",
            "next_absolute_update",
            "policy_schedule",
            "value_schedule",
            "augmentation_schedule",
        }
        schema_version = state.get("schema_version")
        if schema_version == 2 and not isinstance(schema_version, bool):
            required = common_required
        elif schema_version == self._TIME_REVERSAL_STATE_SCHEMA_VERSION and not isinstance(schema_version, bool):
            required = common_required | {"policy_contract"}
        else:
            raise ValueError(
                "Unsupported time_reversal_state schema_version: expected legacy 2 or "
                f"{self._TIME_REVERSAL_STATE_SCHEMA_VERSION}, received {schema_version!r}."
            )
        if set(state) != required:
            missing = sorted(required - set(state))
            unexpected = sorted(set(state) - required)
            raise ValueError(
                "Checkpoint time_reversal_state schema fields do not match: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if schema_version == self._TIME_REVERSAL_STATE_SCHEMA_VERSION:
            self._validate_saved_policy_contract(state["policy_contract"])
        if "iter" not in loaded_dict:
            raise ValueError("A checkpoint with time_reversal_state must also contain the runner iter field.")
        runner_iteration = loaded_dict["iter"]
        if isinstance(runner_iteration, bool) or not isinstance(runner_iteration, Integral) or runner_iteration < 0:
            raise ValueError(f"Checkpoint runner iter must be a nonnegative integer; received {runner_iteration!r}.")
        last_completed_update = state["last_completed_update"]
        next_absolute_update = state["next_absolute_update"]
        if (
            isinstance(last_completed_update, bool)
            or not isinstance(last_completed_update, Integral)
            or last_completed_update < -1
            or isinstance(next_absolute_update, bool)
            or not isinstance(next_absolute_update, Integral)
            or next_absolute_update != last_completed_update + 1
        ):
            raise ValueError(
                "Checkpoint time_reversal_state must contain consecutive last_completed_update and "
                "next_absolute_update integers; received "
                f"{last_completed_update!r} and {next_absolute_update!r}."
            )
        if runner_iteration not in (last_completed_update, next_absolute_update):
            raise ValueError(
                "Checkpoint runner iter must identify either the last completed update or the same pending next "
                f"update after an immediate resave; received iter={runner_iteration!r}, "
                f"last_completed_update={last_completed_update!r}, next_absolute_update={next_absolute_update!r}."
            )
        if (
            isinstance(next_absolute_update, bool)
            or not isinstance(next_absolute_update, Integral)
            or next_absolute_update < 0
        ):
            raise ValueError(
                "Checkpoint time_reversal_state.next_absolute_update must be a nonnegative integer; "
                f"received {next_absolute_update!r}."
            )
        if validate_schedules:
            policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
            expected_schedules = {
                "policy_schedule": vars(policy_schedule),
                "value_schedule": vars(value_schedule),
                "augmentation_schedule": vars(self._resolved_time_reversal_augmentation_schedule()),
            }
            for name, expected in expected_schedules.items():
                saved = state[name]
                saved = dict(saved) if isinstance(saved, Mapping) else None
                exact_match = saved is not None and saved.keys() == expected.keys()
                exact_match = exact_match and all(
                    type(saved[key]) is type(expected[key]) and saved[key] == expected[key] for key in expected
                )
                if not exact_match:
                    raise ValueError(
                        f"Checkpoint {name} does not exactly match the resolved training configuration: "
                        f"expected {expected!r}, received {saved!r}."
                    )
        return int(next_absolute_update)

    def _checkpoint_next_iteration(self, loaded_dict: Mapping, *, validate_schedules: bool) -> int | None:
        """Return the checkpoint's next update while retaining the legacy iter-only fallback."""
        next_iteration = self._validate_time_reversal_checkpoint_state(
            loaded_dict,
            validate_schedules=validate_schedules,
        )
        if next_iteration is not None or "iter" not in loaded_dict:
            return next_iteration
        runner_iteration = loaded_dict["iter"]
        if isinstance(runner_iteration, bool) or not isinstance(runner_iteration, Integral) or runner_iteration < 0:
            raise ValueError(f"Checkpoint runner iter must be a nonnegative integer; received {runner_iteration!r}.")
        return int(runner_iteration) + 1

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load algorithm state and align the TRS schedule to the next PPO update.

        RSL-RL 5.0.1 saves ``iter`` after completing that indexed update. The
        loaded mapping is ephemeral, so it is advanced here before the runner
        restores its counter. This keeps the runner, TensorBoard step, and
        auxiliary schedule aligned to the next absolute update.
        """
        self._validate_checkpoint_policy_width(loaded_dict)
        load_iteration_requested = load_cfg is None or bool(load_cfg.get("iteration", False))
        restore_environment_iteration = load_cfg is not None and bool(load_cfg.get("environment_iteration", False))
        load_augmentation = load_cfg is None or bool(load_cfg.get("augmentation", load_cfg.get("iteration", False)))
        active_curriculum_state = self._command_curriculum_state()
        # ``environment_iteration`` is also requested by actor-only playback so
        # gait schedules match the checkpoint iteration.  It is not a training
        # resume and must not restore or require per-environment curriculum
        # runtime, whose shape commonly differs between train and inference.
        full_resume_requested = load_iteration_requested
        checkpoint_has_curriculum = "command_curriculum_state" in loaded_dict
        restore_curriculum = active_curriculum_state is not None and full_resume_requested
        if full_resume_requested:
            if active_curriculum_state is not None and not checkpoint_has_curriculum:
                raise ValueError(
                    "The active command curriculum has no command_curriculum_state in this checkpoint. "
                    "Refusing to silently restart curriculum competence state."
                )
            if active_curriculum_state is None and checkpoint_has_curriculum:
                raise ValueError(
                    "The checkpoint contains command_curriculum_state, but the active command curriculum is "
                    "disabled. Enable the saved curriculum mode for a full resume, or request weights-only "
                    "transfer without iteration/environment state."
                )
        augmentation = getattr(self, "_tr_augmentation", None)
        require_schedule_match = load_iteration_requested or (augmentation is not None and load_augmentation)
        checkpoint_next_iteration = None
        if require_schedule_match or restore_environment_iteration:
            checkpoint_next_iteration = self._checkpoint_next_iteration(
                loaded_dict,
                validate_schedules=require_schedule_match,
            )
        if restore_environment_iteration and checkpoint_next_iteration is None:
            raise ValueError("environment_iteration checkpoint loading requires the runner iter field.")
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if augmentation is not None and load_augmentation:
            if "time_reversal_augmentation_state" not in loaded_dict:
                raise ValueError(
                    "The checkpoint has no time_reversal_augmentation_state. Refusing to silently restart enabled "
                    "augmentation models; disable augmentation for transfer loading or provide a matching checkpoint."
                )
            augmentation_state = loaded_dict["time_reversal_augmentation_state"]
            if load_iteration and "iter" in loaded_dict:
                expected_iteration = checkpoint_next_iteration
                saved_iteration = augmentation_state.get("schedule_iteration")
                if saved_iteration != expected_iteration:
                    raise ValueError(
                        "Augmentation schedule checkpoint is inconsistent with the runner iteration: "
                        f"expected {expected_iteration}, received {saved_iteration!r}."
                    )
            augmentation.load_state_dict(augmentation_state)
        if restore_curriculum:
            self._restore_command_curriculum_state(loaded_dict["command_curriculum_state"])
        if load_iteration and "iter" in loaded_dict:
            next_iteration = checkpoint_next_iteration
            assert next_iteration is not None
            loaded_dict["iter"] = next_iteration
            self.current_learning_iteration = next_iteration
            self._time_reversal_update_count = self.current_learning_iteration
        environment_iteration = checkpoint_next_iteration if restore_environment_iteration else None
        self._sync_environment_training_iteration(environment_iteration)
        return load_iteration

    def _validate_time_reversal_configuration(self) -> None:
        """Validate target coefficients and schedule settings without mutating them."""
        if self.symmetry is None:
            return

        augmentation_cfg = self.symmetry.get("tr_augmentation") or {}
        if self.symmetry.get("use_data_augmentation", False):
            warnings.warn(
                "use_data_augmentation is deprecated, but its legacy PPO minibatch duplication remains active "
                "during the compatibility window. Set use_data_augmentation=False and configure the independent "
                "project-local tr_augmentation path for filtered reverse-action supervision.",
                DeprecationWarning,
                stacklevel=2,
            )

        for name in ("mirror_loss_coeff", "value_loss_coeff"):
            value = self.symmetry.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative; received {value!r}.")

        output_space = self.symmetry.get("tr_policy_output_space", "raw_action_mean")
        if output_space not in {"raw_action_mean", "normalized_requested_joint_target"}:
            raise ValueError(
                "tr_policy_output_space must be 'raw_action_mean' or "
                f"'normalized_requested_joint_target'; received {output_space!r}."
            )
        actor_bound_mode = self.symmetry.get("actor_mean_bound_mode", "legacy_global")
        if actor_bound_mode not in {"legacy_global", "per_joint_feasible"}:
            raise ValueError(
                f"actor_mean_bound_mode must be 'legacy_global' or 'per_joint_feasible'; received {actor_bound_mode!r}."
            )
        feasible_margin = self.symmetry.get("actor_mean_feasible_margin_fraction", 0.0)
        if (
            isinstance(feasible_margin, bool)
            or not isinstance(feasible_margin, Real)
            or not math.isfinite(float(feasible_margin))
            or not 0.0 <= float(feasible_margin) < 0.5
        ):
            raise ValueError(
                f"actor_mean_feasible_margin_fraction must be finite and in [0, 0.5); received {feasible_margin!r}."
            )

        validate_legacy_observation_metadata(self.symmetry)

        history_enabled = self.symmetry.get("history_enabled", False)
        history_length = self.symmetry.get("history_length", 30 if history_enabled else 0)
        history_trs_mode = self.symmetry.get("history_trs_mode", "framewise_feature")
        if not isinstance(history_enabled, bool):
            raise ValueError(f"history_enabled must be boolean; received {history_enabled!r}.")
        if isinstance(history_length, bool) or not isinstance(history_length, Integral):
            raise ValueError(f"history_length must be an integer; received {history_length!r}.")
        if history_enabled and history_length <= 0:
            raise ValueError("history_length must be positive when history_enabled=True.")
        if not history_enabled and history_length != 0:
            raise ValueError("history_length must be zero when history_enabled=False.")
        if history_trs_mode not in {"none", "framewise_feature"}:
            raise ValueError(f"history_trs_mode must be 'none' or 'framewise_feature'; received {history_trs_mode!r}.")

        resolved_schedules = {
            term: resolve_time_reversal_schedule(self.symmetry, term) for term in ("policy", "value", "augmentation")
        }
        for schedule in resolved_schedules.values():
            schedule.scale(0)
        actor_or_value_tr_active = any(
            resolved_schedules[term].enabled and resolved_schedules[term].target_coeff > 0.0
            for term in ("policy", "value")
        )
        if history_enabled and history_trs_mode == "none" and actor_or_value_tr_active:
            raise ValueError(
                "history_trs_mode='none' is invalid when history is enabled and actor/value time-reversal "
                "consistency is active. Use history_trs_mode='framewise_feature'."
            )
        if history_enabled and augmentation_cfg.get("enabled", False):
            raise ValueError(
                "tr_augmentation with observation history is unsupported: reversed rollouts do not provide the "
                "future samples needed to reconstruct complete causal historical context. Use --no-history for "
                "the experimental sidecar, or disable tr_augmentation for the primary history-aware method."
            )

        immutable_metadata = self._policy_contract_metadata()
        for name in (
            "observation_contract_version",
            "instantaneous_frame_dim",
            "history_packing",
            "gait_phase_mapping_version",
            "gait_library_version",
        ):
            if name in self.symmetry:
                expected = immutable_metadata[name]
                received = self.symmetry[name]
                if type(received) is not type(expected) or received != expected:
                    raise ValueError(f"{name} is immutable: expected {expected!r}, received {received!r}.")

        if augmentation_cfg.get("enabled", False):
            if augmentation_cfg.get("mode") != "dynamics_filtered_reverse_action_supervision":
                raise ValueError(
                    "Enabled tr_augmentation requires mode='dynamics_filtered_reverse_action_supervision'."
                )
            if self.actor.is_recurrent or self.critic.is_recurrent:
                raise ValueError(
                    "tr_augmentation rejects recurrent policies because reversed hidden-state semantics are not "
                    "implemented."
                )
            if not augmentation_cfg.get("filter_enabled", True):
                raise ValueError(
                    "Enabled dynamics_filtered_reverse_action_supervision requires filter_enabled=True; "
                    "unfiltered reversed candidates are not a maintained training mode."
                )
            augmentation_gradient_interval = augmentation_cfg.get("gradient_diagnostics_interval", 100)
            if (
                isinstance(augmentation_gradient_interval, bool)
                or not isinstance(augmentation_gradient_interval, int)
                or augmentation_gradient_interval <= 0
            ):
                raise ValueError(
                    "tr_augmentation.gradient_diagnostics_interval must be a positive integer; "
                    f"received {augmentation_gradient_interval!r}."
                )
            augmentation_gradient_epsilon = augmentation_cfg.get("gradient_diagnostics_epsilon", 1.0e-12)
            if (
                isinstance(augmentation_gradient_epsilon, bool)
                or not isinstance(augmentation_gradient_epsilon, Real)
                or not math.isfinite(float(augmentation_gradient_epsilon))
                or augmentation_gradient_epsilon <= 0.0
            ):
                raise ValueError(
                    "tr_augmentation.gradient_diagnostics_epsilon must be finite and positive; "
                    f"received {augmentation_gradient_epsilon!r}."
                )

        gradient_cfg = self.symmetry.get("tr_gradient_diagnostics") or {}
        interval = gradient_cfg.get("interval", 100)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError(f"tr_gradient_diagnostics.interval must be a positive integer; received {interval!r}.")
        epsilon = gradient_cfg.get("epsilon", 1.0e-12)
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(float(epsilon)):
            raise ValueError(f"tr_gradient_diagnostics.epsilon must be finite and positive; received {epsilon!r}.")
        if epsilon <= 0.0:
            raise ValueError(f"tr_gradient_diagnostics.epsilon must be finite and positive; received {epsilon!r}.")

    def _resolved_time_reversal_schedules(
        self,
    ) -> tuple[ResolvedTimeReversalSchedule, ResolvedTimeReversalSchedule]:
        """Resolve independent policy and value schedules from canonical fields and aliases."""
        if self.symmetry is None:
            disabled = ResolvedTimeReversalSchedule(False, 0.0, 0, 0, 0, 0, 1.0, "linear")
            return disabled, disabled
        return (
            resolve_time_reversal_schedule(self.symmetry, "policy"),
            resolve_time_reversal_schedule(self.symmetry, "value"),
        )

    def _resolved_time_reversal_augmentation_schedule(self) -> ResolvedTimeReversalSchedule:
        """Resolve the independent reverse-action-supervision schedule."""
        if self.symmetry is None:
            return ResolvedTimeReversalSchedule(False, 0.0, 0, 0, 0, 0, 1.0, "linear")
        return resolve_time_reversal_schedule(self.symmetry, "augmentation")

    def _legacy_data_augmentation_scale(self) -> float:
        """Return whether deprecated PPO batch duplication passed its historical warmup."""
        if not self.symmetry:
            return 0.0
        if not self.symmetry.get("use_time_reversal_regularization", False):
            return 0.0
        if not self.symmetry.get("use_data_augmentation", False):
            return 0.0
        # The online implementation switched this binary augmentation on at
        # the warmup boundary.  ``rampup_iterations`` scales auxiliary loss
        # coefficients, but cannot partially duplicate a PPO minibatch.
        warmup_iterations = int(self.symmetry.get("warmup_iterations", 0))
        return float(self.current_learning_iteration >= warmup_iterations)

    def _effective_time_reversal_coefficients(self) -> tuple[float, float, float]:
        """Return the shared schedule scale and effective actor/value coefficients."""
        policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
        policy_scale = policy_schedule.scale(self.current_learning_iteration) if policy_schedule.enabled else 0.0
        value_scale = value_schedule.scale(self.current_learning_iteration) if value_schedule.enabled else 0.0
        policy_coeff = policy_schedule.target_coeff * policy_scale
        value_coeff = value_schedule.target_coeff * value_scale
        alias_scale = policy_scale if policy_schedule.enabled else value_scale
        return alias_scale, policy_coeff, value_coeff

    def _time_reversal_enabled(self) -> bool:
        policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
        augmentation_cfg = self.symmetry.get("tr_augmentation", {}) if self.symmetry else {}
        return (
            (policy_schedule.enabled and policy_schedule.target_coeff > 0.0)
            or (value_schedule.enabled and value_schedule.target_coeff > 0.0)
            or bool(
                self.symmetry
                and self.symmetry.get("use_time_reversal_regularization", False)
                and self.symmetry.get("use_data_augmentation", False)
            )
            or bool(augmentation_cfg.get("enabled", False))
            or bool(self.symmetry and self.symmetry.get("log_disabled_raw_consistency", False))
        )

    def _warmup_iterations(self) -> int:
        if self.symmetry is None:
            return 0
        return int(self.symmetry.get("warmup_iterations", 0))

    def _time_reversal_mask(self, observations) -> torch.Tensor:
        return self._time_reversal_validity(observations).combined

    def _time_reversal_validity(self, observations):
        """Return heuristic stable-phase validity masks from historical observations only."""
        validity_cfg = self.symmetry.get("tr_validity", {}) if self.symmetry else {}
        legacy_minimum = float(self.symmetry.get("min_abs_command_velocity", 0.0)) if self.symmetry else 0.0
        return time_reversal_validity_mask(observations, validity_cfg, legacy_minimum)

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

    def _sync_environment_training_iteration(self, iteration: int | None = None) -> None:
        """Apply the absolute PPO update to resume-stable environment curricula."""
        symmetry = getattr(self, "symmetry", None)
        environment = symmetry.get("_env") if symmetry else None
        unwrapped = getattr(environment, "unwrapped", environment)
        setter = getattr(unwrapped, "set_training_iteration", None)
        if callable(setter):
            if iteration is None:
                iteration = int(getattr(self, "current_learning_iteration", 0))
            setter(int(iteration))

    @staticmethod
    def _static_joint_metadata(tensor: torch.Tensor, trailing_dimensions: int, name: str) -> torch.Tensor:
        """Collapse identical per-environment joint metadata to its static action-order value."""
        if tensor.ndim < trailing_dimensions:
            raise RuntimeError(
                f"{name} must have at least {trailing_dimensions} dimensions; received shape {tuple(tensor.shape)}."
            )
        if tensor.ndim == trailing_dimensions:
            return tensor
        trailing_shape = tensor.shape[-trailing_dimensions:]
        flattened = tensor.reshape(-1, *trailing_shape)
        reference = flattened[0]
        if not torch.allclose(flattened, reference.expand_as(flattened), rtol=0.0, atol=0.0):
            raise RuntimeError(
                f"{name} differs across environments, so flattened historical PPO samples cannot be normalized "
                "without retaining environment indices."
            )
        return reference

    def _joint_action_metadata(
        self,
        *,
        required: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return static action-order offset, scale, limits, and feasible bounds."""
        cached = getattr(self, "_joint_action_metadata_cache", None)
        if cached is not None:
            return cached  # type: ignore[return-value]
        symmetry = getattr(self, "symmetry", None)
        environment = symmetry.get("_env") if symmetry else None
        unwrapped = getattr(environment, "unwrapped", environment)
        metadata_getter = getattr(unwrapped, "get_joint_position_action_metadata", None)
        bounds_getter = getattr(unwrapped, "get_joint_position_action_feasible_bounds", None)
        if not callable(metadata_getter) or not callable(bounds_getter):
            if required:
                raise RuntimeError(
                    "Per-joint policy geometry requires a symmetric quadruped environment exposing ordered "
                    "joint-position action metadata and feasible bounds."
                )
            return None
        action_offset, action_scale, soft_limits = metadata_getter()
        lower_bound, upper_bound = bounds_getter(interior_margin_fraction=0.0)
        device = self.device
        dtype = next(self.actor.parameters()).dtype
        action_offset = self._static_joint_metadata(
            torch.as_tensor(action_offset, device=device, dtype=dtype), 1, "action_offset"
        )
        action_scale = self._static_joint_metadata(
            torch.as_tensor(action_scale, device=device, dtype=dtype), 1, "action_scale"
        )
        soft_limits = self._static_joint_metadata(
            torch.as_tensor(soft_limits, device=device, dtype=dtype), 2, "soft_joint_position_limits"
        )
        lower_bound = self._static_joint_metadata(
            torch.as_tensor(lower_bound, device=device, dtype=dtype), 1, "lower_action_bound"
        )
        upper_bound = self._static_joint_metadata(
            torch.as_tensor(upper_bound, device=device, dtype=dtype), 1, "upper_action_bound"
        )
        joint_count = action_offset.shape[-1]
        expected_shapes = {
            "action_scale": action_scale.shape[-1],
            "soft_joint_position_limits": soft_limits.shape[-2],
            "lower_action_bound": lower_bound.shape[-1],
            "upper_action_bound": upper_bound.shape[-1],
        }
        mismatched = {name: width for name, width in expected_shapes.items() if width != joint_count}
        if mismatched:
            raise RuntimeError(
                "Joint action metadata does not share one ordered action width: "
                f"offset={joint_count}, mismatched={mismatched}."
            )
        resolved = (action_offset, action_scale, soft_limits, lower_bound, upper_bound)
        self._joint_action_metadata_cache = resolved
        return resolved

    def _configured_actor_mean_bound_loss(self, actor_mean: torch.Tensor) -> torch.Tensor:
        """Return the selected legacy-global or per-joint-feasible actor-mean penalty."""
        mode = self.symmetry.get("actor_mean_bound_mode", "legacy_global") if self.symmetry else "legacy_global"
        if mode == "legacy_global":
            return self._actor_mean_bound_loss(actor_mean)
        if mode != "per_joint_feasible":
            raise ValueError(f"Unsupported actor_mean_bound_mode: {mode!r}.")
        metadata = self._joint_action_metadata(required=True)
        assert metadata is not None
        _, _, _, lower_bound, upper_bound = metadata
        return feasible_actor_mean_penalty(
            actor_mean,
            lower_bound,
            upper_bound,
            interior_margin_fraction=float(self.symmetry.get("actor_mean_feasible_margin_fraction", 0.0)),
        )

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
        diagnostics = {
            "diagnostics/action_abs_mean": sampled_action_abs.mean().item(),
            "diagnostics/action_abs_max": sampled_action_abs.max().item(),
            "diagnostics/actor_mean_abs_mean": actor_mean_abs.mean().item(),
            "diagnostics/actor_mean_abs_max": actor_mean_abs.max().item(),
        }
        metadata = self._joint_action_metadata(required=False)
        if metadata is not None:
            _, _, _, lower_bound, upper_bound = metadata
            symmetry = getattr(self, "symmetry", None)
            diagnostics.update(
                feasible_actor_mean_diagnostics(
                    self.storage.distribution_params[0],
                    lower_bound,
                    upper_bound,
                    interior_margin_fraction=float(
                        symmetry.get("actor_mean_feasible_margin_fraction", 0.0) if symmetry else 0.0
                    ),
                )
            )
        return diagnostics
