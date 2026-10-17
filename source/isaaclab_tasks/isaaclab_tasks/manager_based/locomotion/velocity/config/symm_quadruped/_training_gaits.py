# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read checkpoint gait metadata without constructing objects from YAML tags."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def _read_training_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.parent / "params" / "env.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Saved training configuration is required: {path}")
    try:
        config = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid saved training configuration: {path}") from exc
    commands = config.get("commands") if isinstance(config, dict) else None
    if not isinstance(commands, dict) or not isinstance(commands.get("base_velocity"), dict):
        raise ValueError(f"Saved training configuration has no base_velocity command: {path}")
    return config


def _trained_foot_thetas(config: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
    command = config["commands"]["base_velocity"]
    raw_rows = command.get("init_foot_thetas")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Saved training gait templates must be a nonempty list.")
    try:
        rows = tuple(tuple(float(value) for value in row) for row in raw_rows)
    except (TypeError, ValueError) as exc:
        raise ValueError("Saved training gait templates must contain four finite phase offsets per foot row.") from exc
    if any(len(row) != 4 or not all(math.isfinite(value) for value in row) for row in rows):
        raise ValueError("Saved training gait templates must contain four finite phase offsets per foot row.")
    weights = command.get("init_foot_theta_weights")
    if weights not in (None, "null", "None"):
        if not isinstance(weights, list) or len(weights) != len(rows):
            raise ValueError("Saved training gait weights must match the template count.")
        values = tuple(float(value) for value in weights)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Saved training gait weights must be finite and nonnegative.")
        rows = tuple(row for row, weight in zip(rows, values, strict=True) if weight > 0.0)
    if not rows:
        raise ValueError("Saved training gait templates must contain at least one sampled gait.")
    return rows
