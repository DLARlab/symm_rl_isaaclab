# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RSL-RL symmetry configuration validation."""

from __future__ import annotations

import pytest

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


def test_time_reversal_ramp_defaults_preserve_hard_switch_behavior():
    cfg = RslRlSymmetryCfg(data_augmentation_func=lambda **_: None)

    assert cfg.warmup_iterations == 0
    assert cfg.rampup_iterations == 0
    assert cfg.ramp_shape == "linear"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_iterations", -1),
        ("rampup_iterations", -1),
        ("mirror_loss_coeff", -0.1),
        ("mirror_loss_coeff", float("nan")),
        ("mirror_loss_coeff", float("inf")),
        ("value_loss_coeff", -0.1),
        ("value_loss_coeff", float("nan")),
        ("value_loss_coeff", float("inf")),
    ],
)
def test_time_reversal_configuration_rejects_invalid_numeric_values(field, value):
    with pytest.raises(ValueError, match=field):
        RslRlSymmetryCfg(data_augmentation_func=lambda **_: None, **{field: value})


def test_time_reversal_configuration_rejects_unsupported_ramp_shape():
    with pytest.raises(ValueError, match="ramp_shape"):
        RslRlSymmetryCfg(data_augmentation_func=lambda **_: None, ramp_shape="quadratic")
