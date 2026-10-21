# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Combine local TensorBoard or W&B scalar histories without changing their source runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _tensorboard_rows(source: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util.tensor_util import make_ndarray

    events = EventAccumulator(str(source), size_guidance={"scalars": 0, "tensors": 0}).Reload()
    for tag in events.Tags()["scalars"]:
        for event in events.Scalars(tag):
            yield tag, event.step, event.value, event.wall_time
    for tag in events.Tags()["tensors"]:
        for event in events.Tensors(tag):
            value = make_ndarray(event.tensor_proto)
            if value.size == 1 and value.dtype.kind in "biuf":
                yield tag, event.step, float(value.item()), event.wall_time


def _wandb_rows(source: Path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(source))
    try:
        while True:
            blob = store.scan_data()
            if blob is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(blob)
            if not record.HasField("history"):
                continue
            values = {}
            for item in record.history.item:
                tag = item.key or "/".join(item.nested_key)
                values[tag] = json.loads(item.value_json)
            step = values.get("_step")
            if step is None and record.history.HasField("step"):
                step = record.history.step.num
            if step is None:
                raise ValueError(f"W&B history record has no explicit step: {source}")
            wall_time = values.get("_timestamp", 0.0)
            for tag, value in values.items():
                if not tag.startswith("_") and isinstance(value, (int, float)) and math.isfinite(value):
                    yield tag, int(step), float(value), float(wall_time)
    finally:
        store.close()


def _source_rows(source: Path, input_type: str):
    if input_type != "tensorboard":
        archives = [source] if source.suffix == ".wandb" else sorted(source.glob("*.wandb"))
        if archives:
            if len(archives) != 1:
                raise ValueError(f"Select one local W&B run per input, rather than its parent directory: {source}")
            yield from _wandb_rows(archives[0])
            return
        if input_type == "wandb":
            raise ValueError(f"No local .wandb archive found in {source}")
    if not source.is_file() and not list(source.glob("events.out.tfevents.*")):
        raise ValueError(f"No TensorBoard event files found in {source}")
    yield from _tensorboard_rows(source)


def main(argv: list[str] | None = None) -> int:
    """Write a derived CSV, TensorBoard history, and source/boundary manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Training folders, W&B run folders, or event files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="New output directory; must not exist.")
    parser.add_argument("--input_type", choices=("auto", "tensorboard", "wandb"), default="auto")
    parser.add_argument("--step_offsets", nargs="+", type=int, help="One explicit iteration offset per input.")
    parser.add_argument(
        "--curriculum_restarted",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based input indices with curriculum resets.",
    )
    args = parser.parse_args(argv)
    offsets = args.step_offsets if args.step_offsets is not None else [0] * len(args.inputs)
    if len(offsets) != len(args.inputs):
        parser.error("--step_offsets requires one value per input.")
    if any(index < 0 or index >= len(args.inputs) for index in args.curriculum_restarted):
        parser.error("--curriculum_restarted indices must identify input runs.")
    if args.output_dir.exists():
        parser.error("--output_dir already exists; choose a new directory to preserve existing results.")

    merged = {}
    sources = []
    duplicates = 0
    for index, (source, offset) in enumerate(zip(args.inputs, offsets)):
        steps = []
        for tag, step, value, wall_time in _source_rows(source, args.input_type):
            # These TensorBoard tags use elapsed seconds, not iterations. Do not apply an iteration offset to them.
            if tag.endswith("/time"):
                continue
            step += offset
            if step < 0:
                raise ValueError("Step offsets must not produce negative iteration indices.")
            key = (tag, step)
            duplicates += int(key in merged)
            merged[key] = (float(value), wall_time, index)
            steps.append(step)
        if not steps:
            raise ValueError(f"No scalar training history found in {source}")
        sources.append(
            {
                "path": str(source.resolve()),
                "step_offset": offset,
                "first_step": min(steps),
                "last_step": max(steps),
                "curriculum": "restarted" if index in args.curriculum_restarted else "unknown",
            }
        )

    from torch.utils.tensorboard import SummaryWriter

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with SummaryWriter(log_dir=str(args.output_dir)) as writer:
        with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
            table = csv.writer(stream)
            table.writerow(["tag", "step", "value", "wall_time", "source_index"])
            for (tag, step), (value, wall_time, index) in sorted(
                merged.items(), key=lambda item: (item[0][1], item[0][0])
            ):
                table.writerow([tag, step, value, wall_time, index])
                writer.add_scalar(tag, value, step, walltime=wall_time or None)
        for index, source in enumerate(sources):
            if index in args.curriculum_restarted:
                writer.add_scalar("Resume/curriculum_restarted", 1.0, source["first_step"])
    manifest = {
        "sources": sources,
        "overlap_policy": "later input wins for each metric/iteration",
        "duplicate_points_replaced": duplicates,
        "scalar_points": len(merged),
        "elapsed_time_tags": "omitted: elapsed-time axes require separate offsets, not iteration offsets",
        "note": "Derived offline history; original files and online W&B runs are unchanged.",
    }
    (args.output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(merged)} scalar points to {args.output_dir}; replaced {duplicates} overlapping points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
