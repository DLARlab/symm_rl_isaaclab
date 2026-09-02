# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared RSL-RL PPO setup for symmetric quadruped locomotion tasks."""

from collections.abc import Callable
from typing import Literal

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_cfg import (
    TimeReversalSymmetryCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_POLICY_OBS_LAYOUT,
    SYMM_QUADRUPED_POLICY_OBS_SCALE,
)


def configure_symm_quadruped_ppo(
    cfg,
    *,
    experiment_name: str,
    data_augmentation_func: Callable,
    use_data_augmentation: bool,
    value_loss_coeff: float,
    mirror_loss_coeff: float = 0.1,
    min_abs_command_velocity: float = 0.0,
    warmup_iterations: int = 500,
    rampup_iterations: int = 0,
    ramp_shape: str = "linear",
    history_enabled: bool = True,
    history_length: int = 30,
    history_trs_mode: Literal["none", "framewise_feature"] = "framewise_feature",
) -> None:
    """Apply the shared symmetric quadruped PPO/TRS defaults to a runner config.

    Args:
        cfg: Runner config to mutate.
        experiment_name: RSL-RL experiment directory name.
        data_augmentation_func: Time-reversal data augmentation function.
        use_data_augmentation: Whether to duplicate mini-batch samples with time-reversed states. This legacy
            compatibility path is deprecated; prefer the filtered ``tr_augmentation`` configuration.
        value_loss_coeff: Weight for the value-function TRS consistency loss.
        mirror_loss_coeff: Weight for the policy mirror loss.
        min_abs_command_velocity: Minimum forward command velocity [m/s] for TRS losses.
        warmup_iterations: Number of fully unregularized PPO updates before applying TRS losses.
        rampup_iterations: Number of PPO updates used to ramp the TRS loss coefficients.
        ramp_shape: Shape of the TRS loss coefficient ramp.
        history_enabled: Whether native policy observation history is enabled.
        history_length: Number of policy frames, or zero when history is disabled.
        history_trs_mode: Feature-level transform applied to policy history.
    """
    cfg.max_iterations = 20000
    cfg.save_interval = 1000
    cfg.experiment_name = experiment_name
    cfg.clip_actions = None
    cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    cfg.actor.hidden_dims = [512, 256, 128]
    cfg.actor.distribution_cfg.init_std = 0.5
    cfg.critic.hidden_dims = [512, 256, 128]
    cfg.algorithm.class_name = (
        "isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo:TimeReversalPPO"
    )
    cfg.algorithm.entropy_coef = 0.005
    cfg.algorithm.symmetry_cfg = TimeReversalSymmetryCfg(
        use_data_augmentation=use_data_augmentation,
        use_mirror_loss=True,
        data_augmentation_func=data_augmentation_func,
        mirror_loss_coeff=mirror_loss_coeff,
        use_time_reversal_regularization=True,
        value_loss_coeff=value_loss_coeff,
        min_abs_command_velocity=min_abs_command_velocity,
        warmup_iterations=warmup_iterations,
        rampup_iterations=rampup_iterations,
        ramp_shape=ramp_shape,
        history_enabled=history_enabled,
        history_length=history_length,
        history_trs_mode=history_trs_mode,
        # Deprecated RSL-RL aliases remain schema-derived during the compatibility window.
        command_observation_index=SYMM_QUADRUPED_POLICY_OBS_LAYOUT.velocity_command.start,
        command_observation_scale=SYMM_QUADRUPED_POLICY_OBS_SCALE.velocity_command[0],
    )
