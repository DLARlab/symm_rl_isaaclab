# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from numbers import Integral, Real
from typing import Literal

from isaaclab.utils.configclass import configclass


@configclass
class RslRlSymmetryCfg:
    """Configuration for the symmetry-augmentation in the training.

    When :meth:`use_data_augmentation` is True, the :meth:`data_augmentation_func` is used to generate
    augmented observations and actions. These are then used to train the model.

    When :meth:`use_mirror_loss` is True, the :meth:`mirror_loss_coeff` is used to weight the
    symmetry-mirror loss. This loss is directly added to the agent's loss function.

    If both :meth:`use_data_augmentation` and :meth:`use_mirror_loss` are False, then no symmetry-based
    training is enabled. However, the :meth:`data_augmentation_func` is called to compute and log
    symmetry metrics. This is useful for performing ablations.

    For more information, please check the work from :cite:`mittal2024symmetry`.
    """

    use_data_augmentation: bool = False
    """Whether to use symmetry-based data augmentation. Defaults to False."""

    use_mirror_loss: bool = False
    """Whether to use the symmetry-augmentation loss. Defaults to False."""

    data_augmentation_func: callable = MISSING
    """The symmetry data augmentation function.

    The function signature should be as follows:

    Args:

        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (tensordict.TensorDict | None): The observation tensor dictionary. If None, the observation is not used.
        action (torch.Tensor | None): The action tensor. If None, the action is not used.

    Returns:
        A tuple containing the augmented observation dictionary and action tensors. The tensors can be None,
        if their respective inputs are None.
    """

    mirror_loss_coeff: float = 0.0
    """The weight for the symmetry-mirror loss. Defaults to 0.0."""

    use_time_reversal_regularization: bool = False
    """Whether to use time-reversal regularization."""

    value_loss_coeff: float = 0.0
    """The weight for the value-function time-reversal consistency loss. Defaults to 0.0."""

    min_abs_command_velocity: float = 0.0
    """Minimum absolute command velocity [m/s] for applying time-reversal losses."""

    warmup_iterations: int = 0
    """Number of fully unregularized PPO updates before a hard switch or coefficient ramp starts."""

    rampup_iterations: int = 0
    """Number of PPO updates used to ramp the time-reversal loss coefficients.

    A value of zero preserves the legacy hard-switch behavior after :attr:`warmup_iterations`.
    """

    ramp_shape: Literal["linear", "half_cosine"] = "linear"
    """Shape of the time-reversal loss coefficient ramp."""

    command_observation_index: int = 3
    """Index of the forward velocity command in the policy observation."""

    command_observation_scale: float = 1.0
    """Scale applied to the command observation."""

    def __post_init__(self) -> None:
        """Validate the time-reversal loss schedule configuration."""
        self.validate_config()

    def validate_config(self) -> None:
        """Validate coefficient and schedule values.

        Raises:
            ValueError: If a coefficient or schedule setting is invalid.
        """
        for name, value in (
            ("warmup_iterations", self.warmup_iterations),
            ("rampup_iterations", self.rampup_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")

        for name, value in (
            ("mirror_loss_coeff", self.mirror_loss_coeff),
            ("value_loss_coeff", self.value_loss_coeff),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative; received {value!r}.")

        if self.ramp_shape not in {"linear", "half_cosine"}:
            raise ValueError(f"ramp_shape must be 'linear' or 'half_cosine'; received {self.ramp_shape!r}.")
