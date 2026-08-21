# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared RSL-RL PPO setup for symmetric quadruped locomotion tasks."""

from collections.abc import Callable

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


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
) -> None:
    """Apply the shared symmetric quadruped PPO/TRS defaults to a runner config.

    Args:
        cfg: Runner config to mutate.
        experiment_name: RSL-RL experiment directory name.
        data_augmentation_func: Time-reversal data augmentation function.
        use_data_augmentation: Whether to duplicate mini-batch samples with time-reversed states.
        value_loss_coeff: Weight for the value-function TRS consistency loss.
        mirror_loss_coeff: Weight for the policy mirror loss.
        min_abs_command_velocity: Minimum forward command velocity [m/s] for TRS losses.
        warmup_iterations: Number of fully unregularized PPO updates before applying TRS losses.
        rampup_iterations: Number of PPO updates used to ramp the TRS loss coefficients.
        ramp_shape: Shape of the TRS loss coefficient ramp.
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
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
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
        command_observation_index=9,
        command_observation_scale=2.0,
    )
