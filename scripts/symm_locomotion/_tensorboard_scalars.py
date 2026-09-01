# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read scalar summaries from TensorBoard event files without TensorBoard."""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


def read_varint(data: bytes | memoryview, offset: int) -> tuple[int, int]:
    """Decode one protobuf varint."""
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("Invalid protobuf varint")


def iter_protobuf_fields(data: bytes | memoryview) -> Iterable[tuple[int, int, int | memoryview]]:
    """Yield protobuf fields as ``(number, wire_type, value)`` tuples."""
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        key, offset = read_varint(view, offset)
        field_number = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            value, offset = read_varint(view, offset)
        elif wire_type == 1:
            value = view[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            size, offset = read_varint(view, offset)
            value = view[offset : offset + size]
            offset += size
        elif wire_type == 5:
            value = view[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def parse_tensor_scalar(data: bytes | memoryview) -> float | None:
    """Decode the first scalar from a TensorProto."""
    dtype = None
    tensor_content = None
    typed_values: dict[int, list[float | int]] = defaultdict(list)
    for field, wire, value in iter_protobuf_fields(data):
        if field == 1 and wire == 0:
            dtype = int(value)
        elif field == 4 and wire == 2:
            tensor_content = bytes(value)
        elif field in {5, 6} and wire == 5:
            typed_values[field].append(struct.unpack("<f", value)[0])
        elif field == 6 and wire == 1:
            typed_values[field].append(struct.unpack("<d", value)[0])
        elif field in {7, 10, 11, 16, 17} and wire == 0:
            typed_values[field].append(int(value))
        elif field in {5, 6, 7, 10, 11, 16, 17} and wire == 2:
            packed = bytes(value)
            if field == 5 and len(packed) >= 4:
                typed_values[field].append(struct.unpack_from("<f", packed)[0])
            elif field == 6 and len(packed) >= 8:
                typed_values[field].append(struct.unpack_from("<d", packed)[0])
            elif field in {7, 10, 11, 16, 17} and packed:
                first, _ = read_varint(packed, 0)
                typed_values[field].append(first)
    if tensor_content:
        tensor_formats = {
            1: "f",
            2: "d",
            3: "i",
            9: "q",
            10: "?",
            22: "I",
            23: "Q",
        }
        item_format = tensor_formats.get(dtype)
        if item_format and len(tensor_content) >= struct.calcsize("<" + item_format):
            return float(struct.unpack_from("<" + item_format, tensor_content)[0])
    dtype_fields = {1: 5, 2: 6, 3: 7, 9: 10, 10: 11, 22: 16, 23: 17}
    typed_field = dtype_fields.get(dtype)
    if typed_field is not None and typed_values[typed_field]:
        return float(typed_values[typed_field][0])
    return None


def parse_summary_value(data: bytes | memoryview) -> tuple[str | None, float | None]:
    """Decode one TensorBoard Summary.Value message."""
    tag = None
    scalar = None
    for field, wire, value in iter_protobuf_fields(data):
        if field == 1 and wire == 2:
            tag = bytes(value).decode("utf-8", errors="replace")
        elif field == 2 and wire == 5:
            scalar = float(struct.unpack("<f", value)[0])
        elif field == 8 and wire == 2:
            scalar = parse_tensor_scalar(value)
    return tag, scalar


def parse_event(data: bytes) -> tuple[float | None, int | None, list[tuple[str, float]]]:
    """Decode wall time, step, and scalar summaries from one TensorBoard event."""
    wall_time = None
    step = None
    summaries: list[tuple[str, float]] = []
    for field, wire, value in iter_protobuf_fields(data):
        if field == 1 and wire == 1:
            wall_time = float(struct.unpack("<d", value)[0])
        elif field == 2 and wire == 0:
            step = int(value)
        elif field == 5 and wire == 2:
            for summary_field, summary_wire, summary_value in iter_protobuf_fields(value):
                if summary_field == 1 and summary_wire == 2:
                    tag, scalar = parse_summary_value(summary_value)
                    if tag is not None and scalar is not None and math.isfinite(scalar):
                        summaries.append((tag, scalar))
    return wall_time, step, summaries


def read_scalars(
    event_path: Path,
    selected_tags: set[str] | None = None,
) -> dict[str, list[tuple[int, float, float]]]:
    """Read scalar points keyed by TensorBoard tag.

    Args:
        event_path: TensorBoard event file to read.
        selected_tags: Optional tags to retain. All scalar tags are retained
            when omitted.

    Returns:
        Scalar points keyed by TensorBoard tag.
    """
    scalars: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with event_path.open("rb") as stream:
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {event_path}")
            record_size = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord length CRC in {event_path}")
            record = stream.read(record_size)
            if len(record) != record_size or len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord payload in {event_path}")
            wall_time, step, values = parse_event(record)
            if wall_time is None or step is None:
                continue
            for tag, value in values:
                if selected_tags is None or tag in selected_tags:
                    scalars[tag].append((step, wall_time, value))
    for points in scalars.values():
        points.sort(key=lambda point: (point[0], point[1]))
    return dict(scalars)
