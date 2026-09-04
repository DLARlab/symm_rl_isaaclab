# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-reversal PPO regularization for symmetric quadruped RSL-RL tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict


class TimeReversalPPO(PPO):
    """PPO with time-reversal warmup, action, and value losses."""

    _OBSERVATION_FRAME_DIM = 56
    _MIN_ACTOR_STD = 1.0e-6
    _NONFINITE_ACTOR_STD_FALLBACK = 1.0
    _ACTOR_MEAN_BOUND = 10.0
    _ACTOR_MEAN_BOUND_LOSS_COEFF = 1.0e-2
    _ACTOR_MEAN_ABORT_BOUND = 50.0
    _ACTOR_MEAN_ABORT_PATIENCE = 25

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        time_reversal_active = time_reversal_enabled and self.current_learning_iteration >= self._warmup_iterations()
        if time_reversal_enabled and self.symmetry["use_data_augmentation"]:
            raise ValueError(
                "Temporally aligned time reversal cannot reuse PPO advantages and returns as generic data "
                "augmentation; set use_data_augmentation=False and use the actor/value consistency losses."
            )

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_actor_bound_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_tr_value_loss = 0 if time_reversal_enabled else None
        time_reversal_pair_generator = None
        time_reversal_valid_pair_fraction = 0.0
        if time_reversal_active:
            reversed_observations, reference_observations, time_reversal_pair_mask = self._time_reversal_pairs()
            time_reversal_valid_pair_fraction = time_reversal_pair_mask.mean().item()
            time_reversal_pair_generator = self._time_reversal_pair_mini_batch_generator(
                reversed_observations,
                reference_observations,
                time_reversal_pair_mask,
            )

        if self.actor.is_recurrent or self.critic.is_recurrent:
            ppo_generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            ppo_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in ppo_generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

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
                if time_reversal_pair_generator is None:
                    raise RuntimeError("Time-reversal pair generator was not initialized.")
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                reversed_batch, reference_batch, pair_mask = next(time_reversal_pair_generator)
                reversed_action_mean = self.actor(reversed_batch)
                with torch.no_grad():
                    reference_action_mean = self.actor(reference_batch)
                _, transformed_reference_action_mean = data_augmentation_func(
                    obs=None,
                    actions=reference_action_mean,
                    env=self.symmetry["_env"],
                )
                reference_batch_size = reference_action_mean.shape[0]
                symmetry_loss = self._masked_mse(
                    reversed_action_mean,
                    transformed_reference_action_mean[reference_batch_size:].detach(),
                    pair_mask,
                )
                reversed_values = self.critic(reversed_batch)
                with torch.no_grad():
                    reference_values = self.critic(reference_batch)
                tr_value_loss = self._masked_mse(
                    reversed_values,
                    reference_values.detach(),
                    pair_mask,
                )

                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                value_loss_coeff = float(self.symmetry.get("value_loss_coeff", 0.0))
                if value_loss_coeff > 0.0:
                    loss += value_loss_coeff * tr_value_loss

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
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_tr_value_loss is not None:
            loss_dict["tr_value"] = mean_tr_value_loss
        if time_reversal_enabled:
            loss_dict["trs/valid_pair_fraction"] = time_reversal_valid_pair_fraction
        loss_dict.update(action_diagnostics)

        return loss_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load algorithm state and keep the TRS warmup counter aligned when possible."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and "iter" in loaded_dict:
            self.current_learning_iteration = int(loaded_dict["iter"])
            self._time_reversal_update_count = self.current_learning_iteration
        return load_iteration

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

    def _time_reversal_pairs(self) -> tuple[TensorDict, TensorDict, torch.Tensor]:
        """Build temporally aligned reversed/reference pairs before PPO shuffles the rollout."""
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError(
                "Historical time-reversal pairing currently supports feed-forward actors and critics only."
            )

        policy_obs = self.storage.observations["policy"]
        frame_dim = int(self.symmetry.get("observation_frame_dim", self._OBSERVATION_FRAME_DIM))
        if frame_dim < 1 or policy_obs.shape[-1] % frame_dim != 0:
            raise ValueError(
                f"Policy observation dimension {policy_obs.shape[-1]} must be a multiple of the configured "
                f"frame dimension {frame_dim}."
            )
        history_length = policy_obs.shape[-1] // frame_dim
        num_steps, num_envs = policy_obs.shape[:2]
        if history_length >= num_steps:
            raise ValueError(
                f"Historical TRS requires num_steps_per_env ({num_steps}) to exceed the observation history length "
                f"({history_length})."
            )

        num_pair_steps = num_steps - history_length
        current_observations = self.storage.observations[history_length:].flatten(0, 1)
        reference_observations = self.storage.observations[:-history_length].flatten(0, 1)
        current_actions = self.storage.actions[history_length:].flatten(0, 1)
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        augmented_observations, _ = data_augmentation_func(
            env=self.symmetry["_env"],
            obs=current_observations,
            actions=current_actions,
        )
        pair_count = current_observations.batch_size[0]
        reversed_observations = augmented_observations[pair_count:]

        done_windows = self.storage.dones[:-1].squeeze(-1).bool().unfold(0, history_length, 1)
        crosses_reset = done_windows.any(dim=-1)
        current_frames = policy_obs[history_length:].reshape(
            num_pair_steps,
            num_envs,
            history_length,
            frame_dim,
        )
        reference_frames = policy_obs[:-history_length].reshape(
            num_pair_steps,
            num_envs,
            history_length,
            frame_dim,
        )
        current_history_is_full = current_frames[..., 0, :].ne(0.0).any(dim=-1)
        reference_history_is_full = reference_frames[..., 0, :].ne(0.0).any(dim=-1)

        command_index = int(self.symmetry.get("command_observation_index", 3))
        command_end = command_index + 3
        if command_index < 0 or command_end > frame_dim:
            raise ValueError(
                f"Velocity-command observation slice [{command_index}:{command_end}] is outside a {frame_dim}D frame."
            )
        reference_command = reference_frames[..., -1, command_index:command_end]
        current_oldest_command = current_frames[..., 0, command_index:command_end]
        command_is_continuous = torch.isclose(
            reference_command,
            current_oldest_command,
            rtol=0.0,
            atol=1.0e-6,
        ).all(dim=-1)

        valid_pair = (
            ~crosses_reset & current_history_is_full & reference_history_is_full & command_is_continuous
        ).reshape(-1, 1)
        command_mask = self._time_reversal_mask(reference_observations).bool()
        pair_mask = (valid_pair & command_mask).to(dtype=policy_obs.dtype)
        return reversed_observations, reference_observations, pair_mask

    def _time_reversal_pair_mini_batch_generator(
        self,
        reversed_observations: TensorDict,
        reference_observations: TensorDict,
        pair_mask: torch.Tensor,
    ):
        """Yield every temporal TRS pair once per PPO learning epoch."""
        pair_count = reversed_observations.batch_size[0]
        if pair_count < self.num_mini_batches:
            raise ValueError(
                f"Historical TRS produced {pair_count} pairs, fewer than {self.num_mini_batches} mini-batches."
            )
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(pair_count, device=self.device)
            for batch_indices in torch.tensor_split(indices, self.num_mini_batches):
                yield (
                    reversed_observations[batch_indices],
                    reference_observations[batch_indices],
                    pair_mask[batch_indices],
                )

    def _time_reversal_mask(self, observations) -> torch.Tensor:
        policy_obs = observations["policy"]
        frame_dim = int(self.symmetry.get("observation_frame_dim", self._OBSERVATION_FRAME_DIM))
        if frame_dim < 1 or policy_obs.shape[-1] % frame_dim != 0:
            raise ValueError(
                f"Policy observation dimension {policy_obs.shape[-1]} must be a multiple of the configured "
                f"frame dimension {frame_dim}."
            )
        command_index = int(self.symmetry.get("command_observation_index", 3))
        if command_index < 0 or command_index >= frame_dim:
            raise ValueError(f"Command observation index {command_index} is outside a {frame_dim}D frame.")
        command_scale = float(self.symmetry.get("command_observation_scale", 1.0))
        min_abs_command = float(self.symmetry.get("min_abs_command_velocity", 0.0))
        latest_command_index = policy_obs.shape[-1] - frame_dim + command_index
        command = policy_obs[:, latest_command_index] / command_scale
        return (torch.abs(command) >= min_abs_command).unsqueeze(-1).to(dtype=policy_obs.dtype)

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(prediction)
        denom = mask.sum().clamp_min(1.0)
        return ((prediction - target).pow(2) * mask).sum() / denom

    def _clamp_actor_std(self) -> None:
        """Keep RSL-RL's scalar Gaussian std parameter positive and finite."""
        distribution = getattr(self.actor, "distribution", None)
        std_param = getattr(distribution, "std_param", None)
        if std_param is None:
            return
        with torch.no_grad():
            std_param.nan_to_num_(
                nan=self._MIN_ACTOR_STD,
                posinf=self._NONFINITE_ACTOR_STD_FALLBACK,
                neginf=self._MIN_ACTOR_STD,
            )
            std_param.clamp_min_(self._MIN_ACTOR_STD)

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
