# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the dependency-free TensorBoard scalar reader."""

from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

scalars = importlib.import_module("_tensorboard_scalars")


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _length_delimited(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _event(step: int, wall_time: float, values: dict[str, float]) -> bytes:
    summary = b""
    for tag, value in values.items():
        summary_value = _length_delimited(1, tag.encode()) + _varint((2 << 3) | 5) + struct.pack("<f", value)
        summary += _length_delimited(1, summary_value)
    return (
        _varint((1 << 3) | 1)
        + struct.pack("<d", wall_time)
        + _varint(2 << 3)
        + _varint(step)
        + _length_delimited(5, summary)
    )


def _tfrecord(record: bytes) -> bytes:
    return struct.pack("<Q", len(record)) + b"\0" * 4 + record + b"\0" * 4


def test_read_scalars_filters_and_sorts_event_points(tmp_path: Path) -> None:
    event_path = tmp_path / "events.out.tfevents.synthetic"
    event_path.write_bytes(
        _tfrecord(_event(2, 12.0, {"Train/mean_reward": 4.5, "Loss/value": 3.0}))
        + _tfrecord(_event(1, 11.0, {"Train/mean_reward": 2.5}))
    )

    result = scalars.read_scalars(event_path, selected_tags={"Train/mean_reward"})

    assert set(result) == {"Train/mean_reward"}
    assert result["Train/mean_reward"] == pytest.approx([(1, 11.0, 2.5), (2, 12.0, 4.5)])


def test_read_scalars_rejects_truncated_tfrecord(tmp_path: Path) -> None:
    event_path = tmp_path / "events.out.tfevents.truncated"
    event_path.write_bytes(b"short")

    with pytest.raises(ValueError, match="Truncated TFRecord length"):
        scalars.read_scalars(event_path)
