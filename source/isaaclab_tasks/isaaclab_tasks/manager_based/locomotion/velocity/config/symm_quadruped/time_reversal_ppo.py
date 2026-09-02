# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-reversal PPO regularization for symmetric quadruped RSL-RL tasks."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral, Real

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_augmentation import (
    TimeReversalAugmentation,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_cfg import (
    TR_CONSISTENCY_MAPPING_VERSION,
    resolve_tr_consistency_mode,
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
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_sequence import (
    TransitionAlignedTRBuffer,
    TransitionAlignedTRCandidateBatch,
    TransitionAlignedTRRecord,
    TRSequenceValidityConfig,
    TRSimulatorValidityMetadata,
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
    _TIME_REVERSAL_STATE_SCHEMA_VERSION = 4
    _TR_SEQUENCE_CONTEXT_KEYS = frozenset(
        {
            "episode_id",
            "command_segment_id",
            "gait_segment_id",
            "disturbance_generation_id",
            "manual_reset_generation",
            "episode_step",
            "command_segment_age",
            "gait_segment_age",
            "disturbance_segment_age",
            "gait_row",
            "body_linear_velocity_b",
            "body_yaw_rate",
            "projected_gravity",
            "contact_impulse",
            "foot_slip",
            "reverse_dynamics_residual",
            "actuator_saturation",
        }
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._joint_action_metadata_cache: tuple[torch.Tensor, ...] | None = None
        self._validate_time_reversal_configuration()
        self.current_learning_iteration = 0
        self._time_reversal_update_count = 0
        self._actor_mean_abort_count = 0
        self._policy_observation_width_validated = False
        self._tr_sequence_buffer: TransitionAlignedTRBuffer | None = None
        self._tr_sequence_pending = None
        self._tr_sequence_manual_reset_generation: torch.Tensor | None = None
        self._tr_augmentation: TimeReversalAugmentation | None = None
        augmentation_cfg = self.symmetry.get("tr_augmentation", {}) if self.symmetry else {}
        if augmentation_cfg.get("enabled", False):
            if self.is_multi_gpu:
                raise ValueError(
                    "tr_augmentation is not supported in distributed PPO until side-model synchronization is "
                    "implemented. Disable augmentation or train on one process."
                )
            self._tr_augmentation = TimeReversalAugmentation(self.symmetry["_env"], augmentation_cfg, self.device)
        if self.is_multi_gpu and self._command_curriculum_state() is not None:
            raise ValueError(
                "The task-local TR-orbit command curriculum is not supported in distributed PPO until curriculum "
                "outcomes and sampling state are synchronized across ranks. Disable the curriculum or use one process."
            )
        self._initialize_time_reversal_sequence_buffer()
        self._clamp_actor_std()
        self._sync_environment_training_iteration()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions after ensuring the scalar action std is valid."""
        self._validate_policy_observation_width(obs["policy"])
        self._clamp_actor_std()
        actions = super().act(obs)
        self._capture_time_reversal_sequence_before_step(obs, actions)
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
        self._finalize_time_reversal_sequence_step(dones, extras)
        super().process_env_step(obs, rewards, dones, extras)

    def _initialize_time_reversal_sequence_buffer(self) -> None:
        """Allocate the transient exact-sequence buffer when it can be used."""
        if self._tr_consistency_mode() != "transition_aligned_sequence" or not self._sequence_consistency_enabled():
            return
        environment = self.symmetry.get("_env") if self.symmetry else None
        if getattr(environment, "clip_actions", None) is not None:
            raise ValueError(
                "transition_aligned_sequence requires wrapper clip_actions=None so sampled actions exactly match "
                "the raw actions stored in causal policy history."
            )
        unwrapped = getattr(environment, "unwrapped", environment)
        context_getter = getattr(unwrapped, "get_time_reversal_sequence_context", None)
        num_envs = getattr(unwrapped, "num_envs", None)
        if not callable(context_getter) or isinstance(num_envs, bool) or not isinstance(num_envs, Integral):
            raise ValueError(
                "transition_aligned_sequence requires a task environment exposing "
                "get_time_reversal_sequence_context() and a positive num_envs."
            )
        if num_envs <= 0:
            raise ValueError(f"The sequence-consistency environment must have positive num_envs; received {num_envs}.")
        validity = self.symmetry.get("tr_validity", {})
        minimum_command = validity.get("min_abs_command_velocity")
        if minimum_command is None:
            minimum_command = self.symmetry.get("min_abs_command_velocity", 0.0)
        validity_cfg = TRSequenceValidityConfig(
            mode=validity.get("mode", "command"),
            min_abs_command_velocity=float(minimum_command),
            tracking_abs_xy=float(validity.get("tracking_abs_tolerance", 0.25)),
            tracking_rel_xy=float(validity.get("tracking_rel_tolerance", 0.25)),
            tracking_abs_yaw=float(validity.get("tracking_abs_yaw_tolerance", 0.25)),
            tracking_rel_yaw=float(validity.get("tracking_rel_yaw_tolerance", 0.25)),
            projected_gravity_tolerance=float(validity.get("projected_gravity_tolerance", 0.35)),
            phase_boundary_margin=float(validity.get("phase_boundary_margin", 0.03)),
            maximum_contact_impulse=float(validity.get("maximum_contact_impulse", math.inf)),
            maximum_foot_slip=float(validity.get("maximum_foot_slip", math.inf)),
            maximum_reverse_dynamics_residual=float(validity.get("maximum_reverse_dynamics_residual", math.inf)),
            maximum_actuator_saturation=float(validity.get("maximum_actuator_saturation", math.inf)),
        )
        self._tr_sequence_buffer = TransitionAlignedTRBuffer(
            num_envs=int(num_envs),
            history_length=self._configured_history_length() if self._history_enabled() else 1,
            device=self.device,
            validity_cfg=validity_cfg,
            candidate_max_age_updates=int(self.symmetry.get("candidate_max_age_updates", 1)),
            allowed_policy_version_span=int(self.symmetry.get("allowed_policy_version_span", 1)),
        )
        context = self._time_reversal_sequence_context()
        self._tr_sequence_manual_reset_generation = self._sequence_context_integer_vector(
            context,
            "manual_reset_generation",
        )

    def _time_reversal_sequence_context(self) -> Mapping[str, torch.Tensor]:
        """Return and validate the task-local causal-sequence context schema."""
        environment = self.symmetry.get("_env") if self.symmetry else None
        unwrapped = getattr(environment, "unwrapped", environment)
        context = unwrapped.get_time_reversal_sequence_context()
        if not isinstance(context, Mapping) or not self._TR_SEQUENCE_CONTEXT_KEYS.issubset(context):
            received = sorted(context) if isinstance(context, Mapping) else type(context).__name__
            raise RuntimeError(
                "get_time_reversal_sequence_context() returned an incompatible schema: "
                f"expected {sorted(self._TR_SEQUENCE_CONTEXT_KEYS)!r}, received {received!r}."
            )
        return context

    def _sequence_context_integer_vector(
        self,
        context: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """Copy one per-environment integer context vector onto the buffer device."""
        sequence_buffer = self._tr_sequence_buffer
        if sequence_buffer is None:
            raise RuntimeError("Sequence context was requested without an active sequence buffer.")
        value = context[name].detach().to(device=sequence_buffer.device, dtype=torch.long).reshape(-1)
        if value.shape != (sequence_buffer.num_envs,):
            raise RuntimeError(
                f"Sequence context field {name!r} must contain one scalar per environment; "
                f"received {tuple(value.shape)}."
            )
        return value.clone()

    def _synchronize_time_reversal_manual_resets(
        self,
        context: Mapping[str, torch.Tensor] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """Invalidate transient candidates after any explicit environment reset."""
        sequence_buffer = self._tr_sequence_buffer
        if sequence_buffer is None:
            return {} if context is None else context
        if context is None:
            context = self._time_reversal_sequence_context()
        manual_reset_generation = self._sequence_context_integer_vector(context, "manual_reset_generation")
        previous_reset_generation = self._tr_sequence_manual_reset_generation
        if previous_reset_generation is not None:
            sequence_buffer.clear_environment_mask(manual_reset_generation != previous_reset_generation)
        self._tr_sequence_manual_reset_generation = manual_reset_generation
        return context

    def _capture_time_reversal_sequence_before_step(self, obs: TensorDict, actions: torch.Tensor) -> None:
        """Capture ``(H_t, y_t, a_t)`` and simulator-only validity metadata."""
        sequence_buffer = getattr(self, "_tr_sequence_buffer", None)
        if sequence_buffer is None:
            return
        context = self._synchronize_time_reversal_manual_resets(self._time_reversal_sequence_context())
        if self._tr_sequence_pending is not None:
            raise RuntimeError("A sequence decision is already pending transition finalization.")
        # prepare_record() consumes this detached view synchronously on the
        # device and retains only a compact frame, so a full H-frame clone is
        # unnecessary on every environment step.
        policy_observation = obs["policy"].detach()
        policy_history = sequence_buffer.prepare_policy_history(policy_observation)
        sampled_action = actions.detach().clone()
        num_envs = sequence_buffer.num_envs
        if policy_observation.shape[0] != num_envs or sampled_action.shape[0] != num_envs:
            raise RuntimeError(
                "Sequence capture requires one observation and sampled action per environment; "
                f"received {policy_observation.shape[0]} observations and {sampled_action.shape[0]} actions "
                f"for {num_envs} environments."
            )
        bool_zeros = torch.zeros(num_envs, dtype=torch.bool, device=sequence_buffer.device)
        collection_update_id = torch.full(
            (num_envs,),
            int(self.current_learning_iteration),
            dtype=torch.long,
            device=sequence_buffer.device,
        )
        simulator = TRSimulatorValidityMetadata(
            body_linear_velocity_b=context["body_linear_velocity_b"].detach().clone(),
            body_yaw_rate=context["body_yaw_rate"].detach().clone(),
            projected_gravity=context["projected_gravity"].detach().clone(),
            contact_impulse=context["contact_impulse"].detach().clone(),
            foot_slip=context["foot_slip"].detach().clone(),
            reverse_dynamics_residual=context["reverse_dynamics_residual"].detach().clone(),
            actuator_saturation=context["actuator_saturation"].detach().clone(),
        )
        pending = TransitionAlignedTRRecord(
            policy_observation=policy_observation,
            latest_frame=policy_history.latest_frame,
            action=sampled_action,
            episode_id=context["episode_id"].detach().clone().to(dtype=torch.long),
            command_segment_id=context["command_segment_id"].detach().clone().to(dtype=torch.long),
            gait_segment_id=context["gait_segment_id"].detach().clone().to(dtype=torch.long),
            disturbance_generation_id=context["disturbance_generation_id"].detach().clone().to(dtype=torch.long),
            collection_update_id=collection_update_id,
            transition_valid=bool_zeros.clone(),
            done=bool_zeros.clone(),
            timeout=bool_zeros.clone(),
            history_warmup_complete=(
                context["episode_step"].detach().clone().to(dtype=torch.long) >= sequence_buffer.history_length - 1
            )
            & (
                context["command_segment_age"].detach().clone().to(dtype=torch.long)
                >= sequence_buffer.history_length - 1
            )
            & (context["gait_segment_age"].detach().clone().to(dtype=torch.long) >= sequence_buffer.history_length - 1)
            & (
                context["disturbance_segment_age"].detach().clone().to(dtype=torch.long)
                >= sequence_buffer.history_length - 1
            ),
            gait_row=context["gait_row"].detach().clone().to(dtype=torch.long),
            simulator=simulator,
        )
        self._tr_sequence_pending = sequence_buffer.prepare_record(pending, policy_history)

    def _finalize_time_reversal_sequence_step(
        self,
        dones: torch.Tensor,
        extras: Mapping[str, torch.Tensor],
    ) -> None:
        """Finalize the sampled action only after its forward transition ran."""
        sequence_buffer = getattr(self, "_tr_sequence_buffer", None)
        if sequence_buffer is None:
            return
        pending = self._tr_sequence_pending
        if pending is None:
            raise RuntimeError("Sequence transition finalization has no pending decision record.")
        done = dones.detach().to(device=sequence_buffer.device, dtype=torch.bool).reshape(-1)
        timeout_value = extras.get("time_outs")
        timeout = (
            torch.zeros_like(done)
            if timeout_value is None
            else timeout_value.detach().to(device=sequence_buffer.device, dtype=torch.bool).reshape(-1)
        )
        explicit_valid = extras.get("tr_transition_valid")
        transition_valid = (
            torch.ones_like(done)
            if explicit_valid is None
            else explicit_valid.detach().to(device=sequence_buffer.device, dtype=torch.bool).reshape(-1)
        )
        if (
            done.shape != (sequence_buffer.num_envs,)
            or timeout.shape != done.shape
            or transition_valid.shape != done.shape
        ):
            raise RuntimeError(
                "Sequence transition flags must contain one scalar per environment; received "
                f"done={tuple(done.shape)}, timeout={tuple(timeout.shape)}, valid={tuple(transition_valid.shape)}."
            )
        post_context = self._time_reversal_sequence_context()
        post_episode_id = self._sequence_context_integer_vector(post_context, "episode_id")
        post_command_segment_id = self._sequence_context_integer_vector(post_context, "command_segment_id")
        post_gait_segment_id = self._sequence_context_integer_vector(post_context, "gait_segment_id")
        post_disturbance_generation_id = self._sequence_context_integer_vector(
            post_context,
            "disturbance_generation_id",
        )
        post_manual_reset_generation = self._sequence_context_integer_vector(
            post_context,
            "manual_reset_generation",
        )
        transition_valid &= ~done & ~timeout & torch.isfinite(pending.action).all(dim=-1)
        transition_valid &= post_episode_id == pending.episode_id
        transition_valid &= post_command_segment_id == pending.command_segment_id
        transition_valid &= post_gait_segment_id == pending.gait_segment_id
        transition_valid &= post_disturbance_generation_id == pending.disturbance_generation_id
        transition_valid &= post_manual_reset_generation == self._tr_sequence_manual_reset_generation
        simulator = TRSimulatorValidityMetadata(
            body_linear_velocity_b=post_context["body_linear_velocity_b"].detach().clone(),
            body_yaw_rate=post_context["body_yaw_rate"].detach().clone(),
            projected_gravity=post_context["projected_gravity"].detach().clone(),
            contact_impulse=post_context["contact_impulse"].detach().clone(),
            foot_slip=post_context["foot_slip"].detach().clone(),
            reverse_dynamics_residual=post_context["reverse_dynamics_residual"].detach().clone(),
            actuator_saturation=post_context["actuator_saturation"].detach().clone(),
        )
        finalized = replace(
            pending,
            transition_valid=transition_valid,
            done=done,
            timeout=timeout,
            simulator=simulator,
        )
        self._tr_sequence_pending = None
        sequence_buffer.append(finalized, collection_update=int(self.current_learning_iteration))

    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO updates with optional time-reversal regularization."""
        consistency_mode = self._tr_consistency_mode()
        policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
        trs_scale, effective_tr_policy_coeff, effective_tr_value_coeff = self._effective_time_reversal_coefficients()
        legacy_data_augmentation_scale = self._legacy_data_augmentation_scale()
        legacy_data_augmentation_active = legacy_data_augmentation_scale > 0.0
        trs_scale = max(trs_scale, legacy_data_augmentation_scale)
        log_disabled_raw = bool(self.symmetry and self.symmetry.get("log_disabled_raw_consistency", False))
        compute_policy_raw = consistency_mode != "none" and (effective_tr_policy_coeff > 0.0 or log_disabled_raw)
        compute_value_raw = consistency_mode != "none" and (effective_tr_value_coeff > 0.0 or log_disabled_raw)
        gradient_cfg = self.symmetry.get("tr_gradient_diagnostics", {}) if self.symmetry else {}
        gradient_diagnostics_enabled = bool(gradient_cfg.get("enabled", False))
        gradient_diagnostics_update = gradient_diagnostics_enabled and (
            self.current_learning_iteration % int(gradient_cfg.get("interval", 100)) == 0
        )
        sequence_candidates: TransitionAlignedTRCandidateBatch | None = None
        sequence_buffer = getattr(self, "_tr_sequence_buffer", None)
        sequence_diagnostic_raw = gradient_diagnostics_update and (policy_schedule.enabled or value_schedule.enabled)
        if consistency_mode == "transition_aligned_sequence" and sequence_buffer is not None:
            if isinstance(sequence_buffer, TransitionAlignedTRBuffer):
                self._synchronize_time_reversal_manual_resets()
            if compute_policy_raw or compute_value_raw or sequence_diagnostic_raw:
                sequence_candidates = sequence_buffer.pop_candidates(self.current_learning_iteration)
        sequence_diagnostics = sequence_buffer.diagnostics if sequence_buffer is not None else {}
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
            if (
                self.symmetry
                and consistency_mode == "transition_aligned_sequence"
                and (need_policy_raw or need_value_raw)
                and sequence_candidates is not None
            ):
                candidate_count = sequence_candidates.count
                selected_count = min(original_batch_size, candidate_count)
                start = (batch_index * selected_count) % candidate_count
                selection = torch.remainder(
                    torch.arange(selected_count, device=self.device) + start,
                    candidate_count,
                )
                selected_candidates = sequence_candidates.index(selection)
                actor_source_observations = TensorDict(
                    {"policy": selected_candidates.actor_source_observation},
                    batch_size=[selected_count],
                )
                value_source_observations = TensorDict(
                    {"policy": selected_candidates.value_source_observation},
                    batch_size=[selected_count],
                )
                reversed_observations = TensorDict(
                    {"policy": selected_candidates.reversed_observation},
                    batch_size=[selected_count],
                )
                sequence_weight = selected_candidates.weight.detach()

                if need_policy_raw:
                    with torch.no_grad():
                        original_action_mean = self.actor(actor_source_observations).detach()
                    reversed_action_mean = self.actor(reversed_observations)
                    raw_action_consistency_loss = self._masked_mse(
                        reversed_action_mean,
                        original_action_mean,
                        sequence_weight,
                    )
                    metadata = self._joint_action_metadata(required=False)
                    if metadata is not None:
                        action_offset, action_scale, soft_limits, _, _ = metadata
                        normalized_original = normalize_requested_joint_targets(
                            original_action_mean,
                            action_offset,
                            action_scale,
                            soft_limits,
                        )
                        normalized_reversed = normalize_requested_joint_targets(
                            reversed_action_mean,
                            action_offset,
                            action_scale,
                            soft_limits,
                        )
                        normalized_target_consistency_loss = self._masked_mse(
                            normalized_reversed,
                            normalized_original.detach(),
                            sequence_weight,
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
                    with torch.no_grad():
                        original_values = self.critic(value_source_observations).detach()
                    reversed_values = self.critic(reversed_observations)
                    tr_value_loss = self._masked_mse(
                        reversed_values,
                        original_values,
                        sequence_weight,
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

            elif (
                self.symmetry and consistency_mode == "framewise_feature_approx" and (need_policy_raw or need_value_raw)
            ):
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
            f"tr_consistency_mode/{self._tr_consistency_mode()}": 1.0,
            "tr_consistency/is_approximate": float(self._tr_consistency_mode() == "framewise_feature_approx"),
            "tr_consistency/action_history_reconstructed": float(
                self._tr_consistency_mode() == "transition_aligned_sequence"
            ),
            "tr_consistency/history_order_reversed": float(
                self._tr_consistency_mode() == "transition_aligned_sequence"
            ),
            "tr_sequence/candidate_pool_available": float(sequence_candidates is not None),
            "tr_sequence/candidate_pool_count": float(0 if sequence_candidates is None else sequence_candidates.count),
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
        if consistency_mode == "framewise_feature_approx":
            # Deprecated dashboard key retained for archived runs during the
            # configuration-alias compatibility window.
            loss_dict["history_trs_mode/framewise_feature"] = 1.0
        if mask_diagnostics_count:
            loss_dict.update({name: value / mask_diagnostics_count for name, value in mask_diagnostics_sum.items()})
        loss_dict.update(gradient_diagnostics)
        loss_dict.update(augmentation_diagnostics)
        loss_dict.update(sequence_diagnostics)
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

    def _tr_consistency_mode(self, *, warn_legacy: bool = False) -> str:
        """Return the resolved actor/value consistency construction."""
        return resolve_tr_consistency_mode(self.symmetry or {}, warn_legacy=warn_legacy)

    def _expected_policy_input_dim(self) -> int:
        """Return the flattened actor/critic policy input width."""
        history_frames = self._configured_history_length() if self._history_enabled() else 1
        return history_frames * SYMM_QUADRUPED_POLICY_OBS_DIM

    def _policy_contract_metadata(self) -> dict[str, object]:
        """Return deterministic metadata required to interpret a policy checkpoint."""
        sequence_history_length = self._configured_history_length() if self._history_enabled() else 1
        return {
            "observation_contract_version": SYMM_QUADRUPED_OBSERVATION_CONTRACT_VERSION,
            "instantaneous_frame_dim": SYMM_QUADRUPED_POLICY_OBS_DIM,
            "history_enabled": self._history_enabled(),
            "history_length": self._configured_history_length(),
            "history_packing": SYMM_QUADRUPED_POLICY_OBSERVATION_CONTRACT.history_packing,
            "policy_input_dim": self._expected_policy_input_dim(),
            "gait_phase_mapping_version": SYMM_QUADRUPED_PHASE_MAPPING_VERSION,
            "gait_library_version": SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION,
            "tr_consistency_mode": self._tr_consistency_mode(),
            "tr_consistency_mapping_version": TR_CONSISTENCY_MAPPING_VERSION,
            "action_history_length": 2,
            "sequence_history_length": sequence_history_length,
            "required_sequence_records": sequence_history_length + 2,
            "candidate_max_age_updates": int(self.symmetry.get("candidate_max_age_updates", 1)),
            "allowed_policy_version_span": int(self.symmetry.get("allowed_policy_version_span", 1)),
            "actor_alignment": "edge_t_to_reverse_state_t_plus_1",
            "value_alignment": "state_t_plus_1",
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

    def _validate_saved_policy_contract(self, saved_contract: object, *, training_resume: bool = True) -> None:
        """Validate checkpoint policy metadata without padding or projecting inputs."""
        expected = self._policy_contract_metadata()
        if not isinstance(saved_contract, Mapping):
            raise ValueError("Checkpoint policy_contract must be a mapping.")
        observation_fields = {
            "observation_contract_version",
            "instantaneous_frame_dim",
            "history_enabled",
            "history_length",
            "history_packing",
            "policy_input_dim",
            "gait_phase_mapping_version",
            "gait_library_version",
        }
        required = set(expected) if training_resume else observation_fields
        if not required.issubset(saved_contract):
            missing = sorted(required - set(saved_contract))
            raise ValueError(f"Checkpoint policy_contract schema fields do not match: missing={missing}.")
        if training_resume and set(saved_contract) != required:
            unexpected = sorted(set(saved_contract) - required)
            raise ValueError(f"Checkpoint policy_contract has unexpected fields: {unexpected}.")
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
            for name in required
        )
        if not exact_match:
            purpose = "training" if training_resume else "observation"
            raise ValueError(
                f"Checkpoint {purpose} policy_contract does not exactly match the active policy configuration: "
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
        has_policy_contract = isinstance(state, Mapping) and "policy_contract" in state
        saved_contract = state["policy_contract"] if has_policy_contract else None
        saved_width = self._checkpoint_actor_input_width(loaded_dict)
        if isinstance(saved_contract, Mapping):
            declared_width = saved_contract.get("policy_input_dim")
            if type(declared_width) is int and saved_width is not None and saved_width != declared_width:
                required_flag = (
                    f"--history --history-length {self._configured_history_length()}"
                    if self._history_enabled()
                    else "--no-history"
                )
                raise ValueError(
                    "Checkpoint actor input width is inconsistent with the declared policy contract: "
                    f"expected {declared_width}, received {saved_width}. Use {required_flag}; checkpoint metadata "
                    "does not override the serialized actor architecture."
                )
        if has_policy_contract:
            self._validate_saved_policy_contract(saved_contract, training_resume=False)
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

    def _preflight_command_curriculum_state(self, state: object) -> None:
        """Validate curriculum compatibility before upstream model mutation."""
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint command_curriculum_state must be a mapping.")
        symmetry = getattr(self, "symmetry", None)
        environment = symmetry.get("_env") if symmetry else None
        unwrapped = getattr(environment, "unwrapped", environment)
        validator = getattr(unwrapped, "validate_command_curriculum_state", None)
        if callable(validator):
            validator(state)

    @staticmethod
    def _preflight_time_reversal_augmentation_state(
        augmentation: TimeReversalAugmentation,
        state: object,
    ) -> None:
        """Validate augmentation state before upstream model mutation."""
        active_state = copy.deepcopy(augmentation.state_dict())
        try:
            augmentation.load_state_dict(state)
        finally:
            augmentation.load_state_dict(active_state)

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
            if validate_schedules and self._tr_consistency_mode() == "transition_aligned_sequence":
                raise ValueError(
                    "A full transition_aligned_sequence resume requires causal mapping metadata. This legacy "
                    "checkpoint may be loaded actor-only or weights-only, but its training iteration cannot be resumed."
                )
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
        elif schema_version in (3, self._TIME_REVERSAL_STATE_SCHEMA_VERSION) and not isinstance(schema_version, bool):
            required = common_required | {"policy_contract"}
        else:
            raise ValueError(
                "Unsupported time_reversal_state schema_version: expected legacy 2/3 or "
                f"{self._TIME_REVERSAL_STATE_SCHEMA_VERSION}, received {schema_version!r}."
            )
        if set(state) != required:
            missing = sorted(required - set(state))
            unexpected = sorted(set(state) - required)
            raise ValueError(
                "Checkpoint time_reversal_state schema fields do not match: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if schema_version == 2 and validate_schedules and self._tr_consistency_mode() == "transition_aligned_sequence":
            raise ValueError(
                "Checkpoint schema 2 has no causal consistency mapping metadata and cannot be resumed under "
                "transition_aligned_sequence. Use actor-only/weights-only loading or the approximate "
                "compatibility mode."
            )
        if schema_version == self._TIME_REVERSAL_STATE_SCHEMA_VERSION:
            self._validate_saved_policy_contract(state["policy_contract"], training_resume=validate_schedules)
        elif schema_version == 3:
            self._validate_saved_policy_contract(state["policy_contract"], training_resume=False)
            if validate_schedules:
                saved_mode = state["policy_contract"].get("history_trs_mode")
                active_mode = self._tr_consistency_mode()
                compatible = saved_mode == "framewise_feature" and active_mode == "framewise_feature_approx"
                if not compatible:
                    raise ValueError(
                        "Checkpoint schema 3 used the legacy framewise feature mapping and cannot be resumed under "
                        f"tr_consistency_mode={active_mode!r}. Use actor-only/weights-only loading, or explicitly "
                        "select framewise_feature_approx for compatibility."
                    )
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
        if restore_curriculum:
            curriculum_state = loaded_dict["command_curriculum_state"]
            self._preflight_command_curriculum_state(curriculum_state)
            saved_curriculum_iteration = curriculum_state.get("training_iteration")
            if type(saved_curriculum_iteration) is not int or saved_curriculum_iteration != checkpoint_next_iteration:
                raise ValueError(
                    "Command-curriculum training_iteration is inconsistent with the checkpoint next update: "
                    f"expected {checkpoint_next_iteration!r}, received {saved_curriculum_iteration!r}."
                )
        augmentation_state = None
        if augmentation is not None and load_augmentation:
            if "time_reversal_augmentation_state" not in loaded_dict:
                raise ValueError(
                    "The checkpoint has no time_reversal_augmentation_state. Refusing to silently restart enabled "
                    "augmentation models; disable augmentation for transfer loading or provide a matching checkpoint."
                )
            augmentation_state = loaded_dict["time_reversal_augmentation_state"]
            if load_iteration_requested and "iter" in loaded_dict:
                expected_iteration = checkpoint_next_iteration
                saved_iteration = (
                    augmentation_state.get("schedule_iteration") if isinstance(augmentation_state, Mapping) else None
                )
                if saved_iteration != expected_iteration:
                    raise ValueError(
                        "Augmentation schedule checkpoint is inconsistent with the runner iteration: "
                        f"expected {expected_iteration}, received {saved_iteration!r}."
                    )
            self._preflight_time_reversal_augmentation_state(augmentation, augmentation_state)
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if augmentation is not None and load_augmentation:
            assert augmentation_state is not None
            augmentation.load_state_dict(augmentation_state)
        if restore_curriculum:
            self._restore_command_curriculum_state(loaded_dict["command_curriculum_state"])
        sequence_buffer = getattr(self, "_tr_sequence_buffer", None)
        if sequence_buffer is not None:
            sequence_buffer.clear()
            self._tr_sequence_pending = None
        if load_iteration and "iter" in loaded_dict:
            next_iteration = checkpoint_next_iteration
            assert next_iteration is not None
            loaded_dict["iter"] = next_iteration
            self.current_learning_iteration = next_iteration
            self._time_reversal_update_count = self.current_learning_iteration
        environment_iteration = checkpoint_next_iteration if restore_environment_iteration else None
        self._sync_environment_training_iteration(environment_iteration)
        return load_iteration

    def _validate_time_reversal_augmentation_configuration(self, augmentation_cfg: Mapping) -> None:
        """Validate the optional model-based reverse-action sidecar."""
        if not augmentation_cfg.get("enabled", False):
            return
        if augmentation_cfg.get("mode") != "dynamics_filtered_reverse_action_supervision":
            raise ValueError("Enabled tr_augmentation requires mode='dynamics_filtered_reverse_action_supervision'.")
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
        gradient_interval = augmentation_cfg.get("gradient_diagnostics_interval", 100)
        if isinstance(gradient_interval, bool) or not isinstance(gradient_interval, int) or gradient_interval <= 0:
            raise ValueError(
                "tr_augmentation.gradient_diagnostics_interval must be a positive integer; "
                f"received {gradient_interval!r}."
            )
        gradient_epsilon = augmentation_cfg.get("gradient_diagnostics_epsilon", 1.0e-12)
        if (
            isinstance(gradient_epsilon, bool)
            or not isinstance(gradient_epsilon, Real)
            or not math.isfinite(float(gradient_epsilon))
            or gradient_epsilon <= 0.0
        ):
            raise ValueError(
                "tr_augmentation.gradient_diagnostics_epsilon must be finite and positive; "
                f"received {gradient_epsilon!r}."
            )

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
        tr_consistency_mode = self._tr_consistency_mode(warn_legacy=True)
        if not isinstance(history_enabled, bool):
            raise ValueError(f"history_enabled must be boolean; received {history_enabled!r}.")
        if isinstance(history_length, bool) or not isinstance(history_length, Integral):
            raise ValueError(f"history_length must be an integer; received {history_length!r}.")
        if history_enabled and history_length <= 0:
            raise ValueError("history_length must be positive when history_enabled=True.")
        if not history_enabled and history_length != 0:
            raise ValueError("history_length must be zero when history_enabled=False.")
        resolved_schedules = {
            term: resolve_time_reversal_schedule(self.symmetry, term) for term in ("policy", "value", "augmentation")
        }
        for schedule in resolved_schedules.values():
            schedule.scale(0)
        if tr_consistency_mode == "none" and self.symmetry.get("use_data_augmentation", False):
            raise ValueError("tr_consistency_mode='none' cannot be combined with legacy PPO data augmentation.")
        if tr_consistency_mode == "transition_aligned_sequence" and self.symmetry.get("use_data_augmentation", False):
            raise ValueError(
                "transition_aligned_sequence supplies auxiliary-only samples and cannot be combined with legacy "
                "PPO transition duplication. Keep use_data_augmentation=False."
            )
        validity_mode = (self.symmetry.get("tr_validity") or {}).get("mode", "command")
        if tr_consistency_mode == "framewise_feature_approx" and "tracking" in validity_mode:
            raise ValueError(
                "Tracking-labelled validity modes require transition-aligned simulator metadata and are not "
                "available in framewise_feature_approx. Use mode='command'/'command_upright_phase' or select "
                "transition_aligned_sequence."
            )
        if tr_consistency_mode == "transition_aligned_sequence" and (
            self.actor.is_recurrent or self.critic.is_recurrent
        ):
            raise ValueError(
                "transition_aligned_sequence requires feed-forward actor and critic networks; reversed recurrent "
                "hidden-state semantics are not implemented."
            )
        if tr_consistency_mode == "transition_aligned_sequence" and self.is_multi_gpu:
            raise ValueError(
                "transition_aligned_sequence is not supported in distributed PPO until auxiliary candidate "
                "sampling is synchronized across ranks. Use one process or select framewise_feature_approx."
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
            "tr_consistency_mapping_version",
            "action_history_length",
        ):
            if name in self.symmetry:
                expected = immutable_metadata[name]
                received = self.symmetry[name]
                if type(received) is not type(expected) or received != expected:
                    raise ValueError(f"{name} is immutable: expected {expected!r}, received {received!r}.")

        self._validate_time_reversal_augmentation_configuration(augmentation_cfg)

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
        if self._tr_consistency_mode() == "none":
            return 0.0, 0.0, 0.0
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

    def _sequence_consistency_enabled(self) -> bool:
        """Return whether exact actor/value candidates or diagnostics can be consumed."""
        policy_schedule, value_schedule = self._resolved_time_reversal_schedules()
        gradient_cfg = self.symmetry.get("tr_gradient_diagnostics", {}) if self.symmetry else {}
        return (
            (policy_schedule.enabled and policy_schedule.target_coeff > 0.0)
            or (value_schedule.enabled and value_schedule.target_coeff > 0.0)
            or bool(self.symmetry and self.symmetry.get("log_disabled_raw_consistency", False))
            or bool(gradient_cfg.get("enabled", False) and (policy_schedule.enabled or value_schedule.enabled))
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
        weight = mask.detach().to(dtype=prediction.dtype).expand_as(prediction)
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        denom = weight.sum().clamp_min(1.0)
        selected = weight > 0.0
        safe_prediction = torch.where(selected, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(selected, target, torch.zeros_like(target))
        squared_error = (safe_prediction - safe_target).pow(2)
        return (weight * squared_error).sum() / denom

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
