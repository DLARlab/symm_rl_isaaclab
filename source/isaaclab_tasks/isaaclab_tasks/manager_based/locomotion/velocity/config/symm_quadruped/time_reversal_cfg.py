# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local configuration for symmetric quadruped time reversal."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Literal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


def _validate_optional_nonnegative_integer(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None; received {value!r}.")


def _validate_optional_nonnegative_real(name: str, value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative or None; received {value!r}.")


@configclass
class TimeReversalScheduleCfg:
    """Independent coefficient schedule for one time-reversal objective.

    ``None`` values inherit the corresponding legacy shared schedule or
    coefficient. This keeps archived command lines numerically identical while
    allowing each new canonical schedule field to be overridden independently.
    """

    enabled: bool | None = None
    """Whether the scheduled contribution is enabled, or ``None`` to inherit its consistency switch."""

    target_coeff: float | None = None
    """Peak coefficient, or ``None`` to inherit the legacy coefficient field."""

    warmup_iterations: int | None = None
    """Number of zero-coefficient PPO updates, or ``None`` to inherit the shared warm-up."""

    rampup_iterations: int | None = None
    """Number of PPO updates in the ramp, or ``None`` to inherit the shared ramp."""

    hold_iterations: int = 0
    """Number of PPO updates to hold the peak before decay."""

    decay_iterations: int = 0
    """Number of PPO updates in the decay to :attr:`final_scale`."""

    final_scale: float = 1.0
    """Final coefficient scale after decay."""

    ramp_shape: Literal["linear", "half_cosine"] | None = None
    """Ramp/decay interpolation, or ``None`` to inherit the shared shape."""

    def validate_config(self, name: str) -> None:
        """Validate this schedule.

        Args:
            name: Configuration path used in validation errors.
        """
        _validate_optional_nonnegative_real(f"{name}.target_coeff", self.target_coeff)
        for field_name in ("warmup_iterations", "rampup_iterations"):
            _validate_optional_nonnegative_integer(f"{name}.{field_name}", getattr(self, field_name))
        for field_name in ("hold_iterations", "decay_iterations"):
            _validate_optional_nonnegative_integer(f"{name}.{field_name}", getattr(self, field_name))
        if not isinstance(self.final_scale, Real) or isinstance(self.final_scale, bool):
            raise ValueError(f"{name}.final_scale must be a finite value in [0, 1]; received {self.final_scale!r}.")
        if not math.isfinite(float(self.final_scale)) or not 0.0 <= self.final_scale <= 1.0:
            raise ValueError(f"{name}.final_scale must be a finite value in [0, 1]; received {self.final_scale!r}.")
        if self.ramp_shape not in (None, "linear", "half_cosine"):
            raise ValueError(
                f"{name}.ramp_shape must be None, 'linear', or 'half_cosine'; received {self.ramp_shape!r}."
            )


@configclass
class TimeReversalValidityMaskCfg:
    """Historical-observation validity mask for time-reversal consistency losses."""

    mode: Literal[
        "command",
        "command_tracking",
        "command_upright_phase",
        "command_tracking_upright_phase",
    ] = "command"
    """Combination of heuristic stable-phase validity components."""

    min_abs_command_velocity: float | None = None
    """Minimum command magnitude [m/s], or ``None`` to inherit the legacy field."""

    tracking_abs_tolerance: float = 0.25
    """Absolute forward-velocity tracking tolerance [m/s]."""

    tracking_rel_tolerance: float = 0.25
    """Relative forward-velocity tracking tolerance."""

    projected_gravity_tolerance: float = 0.35
    """Maximum horizontal projected-gravity norm."""

    phase_boundary_margin: float = 0.03
    """Minimum circular phase distance from touchdown or liftoff [cycles]."""

    command_speed_bin_edges: tuple[float, ...] = (0.5, 1.0, 1.5)
    """Absolute command-speed bin edges [m/s] used only for diagnostics."""

    def validate_config(self) -> None:
        """Validate mask modes and thresholds."""
        modes = {
            "command",
            "command_tracking",
            "command_upright_phase",
            "command_tracking_upright_phase",
        }
        if self.mode not in modes:
            raise ValueError(f"validity_mask.mode must be one of {sorted(modes)!r}; received {self.mode!r}.")
        for name, value in (
            ("validity_mask.min_abs_command_velocity", self.min_abs_command_velocity),
            ("validity_mask.tracking_abs_tolerance", self.tracking_abs_tolerance),
            ("validity_mask.tracking_rel_tolerance", self.tracking_rel_tolerance),
            ("validity_mask.projected_gravity_tolerance", self.projected_gravity_tolerance),
            ("validity_mask.phase_boundary_margin", self.phase_boundary_margin),
        ):
            _validate_optional_nonnegative_real(name, value)
        if self.phase_boundary_margin > 0.5:
            raise ValueError("validity_mask.phase_boundary_margin must not exceed 0.5 cycles.")
        previous = -math.inf
        for edge in self.command_speed_bin_edges:
            _validate_optional_nonnegative_real("validity_mask.command_speed_bin_edges", edge)
            if edge <= previous:
                raise ValueError("validity_mask.command_speed_bin_edges must be strictly increasing.")
            previous = edge


@configclass
class TimeReversalGradientDiagnosticsCfg:
    """Low-frequency gradient-conflict diagnostics."""

    enabled: bool = False
    """Whether diagnostics are enabled."""

    interval: int = 100
    """PPO-update interval. Diagnostics run only on the first minibatch."""

    epsilon: float = 1.0e-12
    """Denominator floor used for norms and cosine similarities."""

    def validate_config(self) -> None:
        """Validate diagnostic cadence and numerical tolerance."""
        if isinstance(self.interval, bool) or not isinstance(self.interval, Integral) or self.interval <= 0:
            raise ValueError(f"gradient_diagnostics.interval must be a positive integer; received {self.interval!r}.")
        if not isinstance(self.epsilon, Real) or not math.isfinite(float(self.epsilon)) or self.epsilon <= 0.0:
            raise ValueError(f"gradient_diagnostics.epsilon must be finite and positive; received {self.epsilon!r}.")


@configclass
class TimeReversalAugmentationCfg:
    """Trajectory-derived reverse-action supervision settings.

    This auxiliary path is disabled by default and never contributes samples to
    PPO ratios, GAE, returns, KL control, entropy averaging, or critic fitting.
    """

    enabled: bool = False
    """Whether to allocate and train the project-local trajectory sidecar."""

    mode: Literal["dynamics_filtered_reverse_action_supervision"] = "dynamics_filtered_reverse_action_supervision"
    """Maintained augmentation mode; reversed samples supervise only the actor likelihood."""

    action_source: Literal["analytic", "learned_inverse"] = "analytic"
    """Source of candidate reversed actions: the analytic involution or the learned inverse model."""

    coefficient: float = 0.0
    """Peak dimensionless coefficient multiplying the reverse-action negative log likelihood."""

    schedule: TimeReversalScheduleCfg = TimeReversalScheduleCfg()
    """Independent absolute-update schedule for :attr:`coefficient`."""

    dynamics_hidden_dims: tuple[int, ...] = (256, 256)
    """Hidden-layer widths of the one-step forward dynamics model."""

    inverse_hidden_dims: tuple[int, ...] = (256, 256)
    """Hidden-layer widths of the inverse dynamics model."""

    learning_rate: float = 1.0e-3
    """Dimensionless Adam learning rate for both side models."""

    ema_decay: float = 0.995
    """Dimensionless exponential moving-average decay applied to target-model parameters."""

    validation_fraction: float = 0.2
    """Deterministic fraction of authentic transitions reserved for validation."""

    validation_quantile: float = 0.95
    """Forward-residual quantile used to calibrate the learned acceptance threshold."""

    threshold_multiplier: float = 2.0
    """Dimensionless multiplier applied to the validation residual quantile."""

    minimum_validation_samples: int = 128
    """Minimum held-out transition count before candidates may be accepted."""

    maximum_validation_loss: float = 1.0
    """Maximum dimensionless component-aware validation residual for model-quality acceptance."""

    filter_enabled: bool = True
    """Whether the mandatory learned dynamics filter is enabled; enabled augmentation requires ``True``."""

    phase_boundary_margin: float = 0.03
    """Minimum circular distance from touchdown and liftoff boundaries [cycles]."""

    maximum_contact_impulse: float = 20.0
    """Largest admissible contact impulse integrated over one RL/control step [N·s]."""

    action_abs_limit: float = 10.0
    """Largest admissible absolute normalized policy action [dimensionless]."""

    max_augmented_to_original_ratio: float = 0.25
    """Dimensionless maximum ratio of auxiliary candidates to authentic PPO samples."""

    use_confidence_weights: bool = True
    """Whether actor likelihood terms are weighted by detached filter confidence."""

    model_updates_per_rollout: int = 1
    """Side-model optimizer steps performed after each rollout."""

    model_batch_size: int = 4096
    """Maximum authentic transition count in one side-model optimizer step."""

    gradient_diagnostics_interval: int = 100
    """PPO-update interval for actor/augmentation gradient diagnostics on the first minibatch."""

    gradient_diagnostics_epsilon: float = 1.0e-12
    """Positive denominator floor for augmentation gradient norms and cosine similarity."""

    rng_seed: int = 0
    """Nonnegative seed for side-model batching and candidate sampling."""

    def validate_config(self) -> None:
        """Validate trajectory augmentation settings."""
        if self.mode != "dynamics_filtered_reverse_action_supervision":
            raise ValueError(f"Unsupported tr_augmentation.mode: {self.mode!r}.")
        if self.action_source not in {"analytic", "learned_inverse"}:
            raise ValueError(f"Unsupported tr_augmentation.action_source: {self.action_source!r}.")
        for name, value in (
            ("coefficient", self.coefficient),
            ("learning_rate", self.learning_rate),
            ("threshold_multiplier", self.threshold_multiplier),
            ("maximum_validation_loss", self.maximum_validation_loss),
            ("phase_boundary_margin", self.phase_boundary_margin),
            ("maximum_contact_impulse", self.maximum_contact_impulse),
            ("action_abs_limit", self.action_abs_limit),
            ("max_augmented_to_original_ratio", self.max_augmented_to_original_ratio),
        ):
            _validate_optional_nonnegative_real(f"tr_augmentation.{name}", value)
        for name, value in (
            ("ema_decay", self.ema_decay),
            ("validation_fraction", self.validation_fraction),
            ("validation_quantile", self.validation_quantile),
        ):
            if not isinstance(value, Real) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"tr_augmentation.{name} must be finite and in [0, 1]; received {value!r}.")
        if self.validation_fraction <= 0.0 or self.validation_fraction >= 1.0:
            raise ValueError("tr_augmentation.validation_fraction must be strictly between 0 and 1.")
        if self.validation_quantile <= 0.0:
            raise ValueError("tr_augmentation.validation_quantile must be strictly positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("tr_augmentation.learning_rate must be strictly positive.")
        if self.phase_boundary_margin > 0.5:
            raise ValueError("tr_augmentation.phase_boundary_margin must not exceed 0.5 cycles.")
        if self.max_augmented_to_original_ratio > 1.0:
            raise ValueError("tr_augmentation.max_augmented_to_original_ratio must not exceed 1.0.")
        for name in ("filter_enabled", "use_confidence_weights"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"tr_augmentation.{name} must be boolean.")
        if self.enabled and not self.filter_enabled:
            raise ValueError("Enabled dynamics_filtered_reverse_action_supervision requires filter_enabled=True.")
        for name in (
            "minimum_validation_samples",
            "model_updates_per_rollout",
            "model_batch_size",
            "gradient_diagnostics_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"tr_augmentation.{name} must be a positive integer; received {value!r}.")
        gradient_epsilon = self.gradient_diagnostics_epsilon
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
        if isinstance(self.rng_seed, bool) or not isinstance(self.rng_seed, Integral) or self.rng_seed < 0:
            raise ValueError(f"tr_augmentation.rng_seed must be a nonnegative integer; received {self.rng_seed!r}.")
        for name in ("dynamics_hidden_dims", "inverse_hidden_dims"):
            hidden_dims = getattr(self, name)
            if not hidden_dims or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in hidden_dims
            ):
                raise ValueError(f"tr_augmentation.{name} must contain positive integers.")
        self.schedule.validate_config("tr_augmentation.schedule")


@configclass
class TimeReversalSymmetryCfg(RslRlSymmetryCfg):
    """Project-local extension of RSL-RL symmetry configuration.

    The inherited RSL-RL fields remain accepted as deprecated aliases. The
    global RSL-RL symmetry implementation is not modified.
    """

    use_tr_policy_consistency: bool | None = None
    """Canonical policy-consistency switch, or ``None`` to inherit :attr:`use_mirror_loss`."""

    use_tr_value_consistency: bool | None = None
    """Canonical value-consistency switch, or ``None`` to infer it from the legacy value coefficient."""

    log_disabled_raw_consistency: bool = False
    """Whether to compute raw consistency losses for explicitly disabled terms."""

    tr_policy_output_space: Literal["raw_action_mean", "normalized_requested_joint_target"] = "raw_action_mean"
    """Policy output space used by the optimized time-reversal consistency loss.

    ``"raw_action_mean"`` preserves the historical loss exactly.  The target-space
    option maps the unclipped requested joint targets into each joint's soft-limit
    interval before applying the configured temporal action operator.
    """

    actor_mean_bound_mode: Literal["legacy_global", "per_joint_feasible"] = "legacy_global"
    """Actor-mean regularizer: historical global bound or per-joint feasible intervals."""

    actor_mean_feasible_margin_fraction: float = 0.0
    """Fraction of each feasible raw-action range reserved as an interior margin."""

    tr_policy_schedule: TimeReversalScheduleCfg = TimeReversalScheduleCfg()
    tr_value_schedule: TimeReversalScheduleCfg = TimeReversalScheduleCfg()
    tr_validity: TimeReversalValidityMaskCfg = TimeReversalValidityMaskCfg()
    tr_gradient_diagnostics: TimeReversalGradientDiagnosticsCfg = TimeReversalGradientDiagnosticsCfg()
    tr_augmentation: TimeReversalAugmentationCfg = TimeReversalAugmentationCfg()

    def validate_config(self) -> None:
        """Validate inherited aliases and all project-local controls."""
        super().validate_config()
        for name in ("use_tr_policy_consistency", "use_tr_value_consistency"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or None; received {value!r}.")
        if not isinstance(self.log_disabled_raw_consistency, bool):
            raise ValueError("log_disabled_raw_consistency must be boolean.")
        if self.tr_policy_output_space not in {"raw_action_mean", "normalized_requested_joint_target"}:
            raise ValueError(
                "tr_policy_output_space must be 'raw_action_mean' or "
                f"'normalized_requested_joint_target'; received {self.tr_policy_output_space!r}."
            )
        if self.actor_mean_bound_mode not in {"legacy_global", "per_joint_feasible"}:
            raise ValueError(
                "actor_mean_bound_mode must be 'legacy_global' or 'per_joint_feasible'; "
                f"received {self.actor_mean_bound_mode!r}."
            )
        margin = self.actor_mean_feasible_margin_fraction
        if (
            isinstance(margin, bool)
            or not isinstance(margin, Real)
            or not math.isfinite(float(margin))
            or not 0.0 <= margin < 0.5
        ):
            raise ValueError(
                f"actor_mean_feasible_margin_fraction must be finite and in [0, 0.5); received {margin!r}."
            )
        self.tr_policy_schedule.validate_config("tr_policy_schedule")
        self.tr_value_schedule.validate_config("tr_value_schedule")
        self.tr_validity.validate_config()
        self.tr_gradient_diagnostics.validate_config()
        self.tr_augmentation.validate_config()
