# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy Go2 time-reversal PPO regularization for RSL-RL."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


class LegacyTimeReversalPPO(PPO):
    """PPO with the old Go2 TRS warmup, action, and value losses."""

    _MIN_ACTOR_STD = 1.0e-6
    _MAX_ACTOR_STD = 10.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_learning_iteration = 0
        self._legacy_update_count = 0
        self._clamp_actor_std()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions after ensuring the scalar action std is valid."""
        self._clamp_actor_std()
        return super().act(obs)

    def update(self) -> dict[str, float]:
        """Run PPO updates with optional legacy time-reversal regularization."""
        legacy_trs_enabled = self._legacy_trs_enabled()
        legacy_trs_active = legacy_trs_enabled and self.current_learning_iteration >= self._legacy_warmup_iterations()

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_tr_value_loss = 0 if legacy_trs_enabled else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            use_data_augmentation = legacy_trs_active and self.symmetry and self.symmetry["use_data_augmentation"]
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

            symmetry_loss = torch.zeros((), device=self.device)
            tr_value_loss = torch.zeros((), device=self.device)
            if legacy_trs_active and self.symmetry:
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

                trs_mask = self._legacy_trs_mask(augmented_observations[:original_batch_size])
                symmetry_loss = self._masked_mse(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                    trs_mask,
                )
                tr_value_loss = self._masked_mse(
                    augmented_values[original_batch_size:],
                    augmented_values[:original_batch_size].detach(),
                    trs_mask,
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
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_tr_value_loss is not None:
            mean_tr_value_loss /= num_updates

        self.storage.clear()
        self._legacy_update_count += 1
        self.current_learning_iteration = self._legacy_update_count

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_tr_value_loss is not None:
            loss_dict["tr_value"] = mean_tr_value_loss

        return loss_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load algorithm state and keep the TRS warmup counter aligned when possible."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and "iter" in loaded_dict:
            self.current_learning_iteration = int(loaded_dict["iter"])
            self._legacy_update_count = self.current_learning_iteration
        return load_iteration

    def _legacy_trs_enabled(self) -> bool:
        if self.symmetry is None:
            return False
        if not self.symmetry.get("use_legacy_time_reversal_regularization", False):
            return False
        return (
            self.symmetry["use_data_augmentation"]
            or self.symmetry["use_mirror_loss"]
            or float(self.symmetry.get("value_loss_coeff", 0.0)) > 0.0
        )

    def _legacy_warmup_iterations(self) -> int:
        if self.symmetry is None:
            return 0
        return int(self.symmetry.get("warmup_iterations", 0))

    def _legacy_trs_mask(self, observations) -> torch.Tensor:
        policy_obs = observations["policy"]
        command_index = int(self.symmetry.get("command_observation_index", 3))
        command_scale = float(self.symmetry.get("command_observation_scale", 1.0))
        min_abs_command = float(self.symmetry.get("min_abs_command_velocity", 0.0))
        command = policy_obs[:, command_index] / command_scale
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
            std_param.nan_to_num_(nan=self._MIN_ACTOR_STD, posinf=self._MAX_ACTOR_STD, neginf=self._MIN_ACTOR_STD)
            std_param.clamp_(min=self._MIN_ACTOR_STD, max=self._MAX_ACTOR_STD)
