# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare individual gaits using the existing, verified aggregate input tables.

Run ``./isaaclab.sh -p analysis/trs_ppo_2026-10-27/gait_analysis.py`` after
``analyze.py``. This script also runs at the end of ``analyze.py``. The saved
phase-clock caveat applies to phase-dependent metrics and reversal diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ROBOTS = ("go2", "x1", "x1_go2bend")
ROBOT_LABELS = {"go2": "Go2", "x1": "X1", "x1_go2bend": "X1 Go2-bend"}
GAITS = (
    "trot",
    "bound",
    "half_bound_front_a",
    "half_bound_front_b",
    "gallop_a",
    "gallop_b",
    "gallop_c",
    "gallop_d",
)
GAIT_LABELS = dict(
    zip(
        GAITS,
        (
            "Trot",
            "Bound",
            "Half-bound A",
            "Half-bound B",
            "Gallop A",
            "Gallop B",
            "Gallop C",
            "Gallop D",
        ),
        strict=True,
    )
)
VELOCITIES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
PLOTTED = (
    ("velocity_vx_rmse_mps", "Body-x\nRMSE"),
    ("velocity_vy_rmse_mps", "Lateral\nRMSE"),
    ("velocity_yaw_rmse_radps", "Yaw-rate\nRMSE"),
    ("torque_squared_abs_imbalance_percent", "Torque\nimbalance"),
    ("normalized_torque_utilization_abs_imbalance_percent", "Norm. torque\nimbalance"),
    ("absolute_work_abs_imbalance_percent", "Work\nimbalance"),
    ("normalized_torque_utilization_total_per_s", "Norm. torque\nuse"),
    ("absolute_work_total_per_s", "Mechanical\npower"),
    ("load_action_clamp_fraction", "Target\nclipping"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _relative_change(first: float, second: float) -> float:
    if abs(first) <= 1e-12:
        return 0.0 if abs(second) <= 1e-12 else math.nan
    return 100.0 * (second / first - 1.0)


def _aggregate_gaits(conditions: list[dict], metadata: dict) -> list[dict]:
    groups = defaultdict(list)
    for row in conditions:
        groups[(row["robot"], row["gait_name"], row["metric"])].append(row)
    comparisons = []
    for (robot, gait, field), rows in groups.items():
        assert len(rows) == 6
        assert {row["velocity_mps"] for row in rows} == set(VELOCITIES)
        for scope in ("all", "backward", "forward"):
            selected = [
                row
                for row in rows
                if scope == "all"
                or ((row["velocity_mps"] < 0) == (scope == "backward"))
            ]
            before = float(np.mean([row["ppo"] for row in selected]))
            after = float(np.mean([row["trs"] for row in selected]))
            direction = 1.0 if metadata[field]["lower_is_better"] == "True" else -1.0
            deltas = (
                np.asarray([row["trs"] - row["ppo"] for row in selected]) * direction
            )
            comparisons.append(
                {
                    "robot": robot,
                    "gait_index": rows[0]["gait_index"],
                    "gait_name": gait,
                    "family": rows[0]["family"],
                    "scope": scope,
                    "metric": field,
                    "label": metadata[field]["label"],
                    "unit": metadata[field]["unit"],
                    "lower_is_better": metadata[field]["lower_is_better"],
                    "ppo": before,
                    "trs": after,
                    "delta_trs_minus_ppo": after - before,
                    "relative_change_percent": _relative_change(before, after),
                    "improved_conditions": int(np.sum(deltas < -1e-12)),
                    "worsened_conditions": int(np.sum(deltas > 1e-12)),
                    "unchanged_conditions": int(np.sum(np.abs(deltas) <= 1e-12)),
                    "conditions": len(selected),
                }
            )
    assert len(comparisons) == len(ROBOTS) * len(GAITS) * len(metadata) * 3
    return comparisons


def _plot_gait_overviews(lookup: dict) -> None:
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dedede")
    for robot in ROBOTS:
        matrix = np.asarray(
            [
                [
                    -lookup[(robot, gait, field)]["relative_change_percent"]
                    for field, _ in PLOTTED
                ]
                for gait in GAITS
            ]
        )
        fig, ax = plt.subplots(figsize=(13.0, 5.9), layout="constrained")
        color = ax.imshow(matrix, cmap=cmap, vmin=-50, vmax=50, aspect="auto")
        ax.set_xticks(range(len(PLOTTED)), [label for _, label in PLOTTED], fontsize=9)
        ax.set_yticks(
            range(len(GAITS)), [GAIT_LABELS[gait] for gait in GAITS], fontsize=10
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=8)
        for y in range(len(GAITS)):
            for x in range(len(PLOTTED)):
                value = matrix[y, x]
                ax.text(
                    x,
                    y,
                    f"{value:+.1f}%" if math.isfinite(value) else "N/A",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        for index in (0.5, 1.5, 3.5):
            ax.axhline(index, color="white", linewidth=2)
        ax.set_title(
            f"{ROBOT_LABELS[robot]}: TRS-PPO improvement by individual gait\n"
            "Relative reduction of six-command means; positive = lower error or load",
            pad=58,
            fontsize=13,
        )
        fig.colorbar(
            color,
            ax=ax,
            fraction=0.025,
            pad=0.025,
            label="Reduction [%]; colors clipped at ±50%",
        )
        fig.savefig(OUT / f"{robot}_gait_improvements.png", dpi=180)
        plt.close(fig)


def _plot_velocity_grid(
    conditions: list[dict],
    field: str,
    title: str,
    unit: str,
    limit: float,
    filename: str,
) -> None:
    lookup = {
        (row["robot"], row["gait_name"], row["velocity_mps"]): row
        for row in conditions
        if row["metric"] == field
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(14.3, 5.8), sharey=True, layout="constrained"
    )
    for ax, robot in zip(axes, ROBOTS, strict=True):
        matrix = np.asarray(
            [
                [
                    lookup[(robot, gait, velocity)]["ppo"]
                    - lookup[(robot, gait, velocity)]["trs"]
                    for velocity in VELOCITIES
                ]
                for gait in GAITS
            ]
        )
        color = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(6), [f"{v:+g}" for v in VELOCITIES], fontsize=10)
        ax.set_yticks(range(8), [GAIT_LABELS[gait] for gait in GAITS], fontsize=10)
        ax.set_title(ROBOT_LABELS[robot], fontsize=12, pad=10)
        ax.set_xlabel("Commanded x velocity [m/s]")
        ax.tick_params(length=0, pad=7)
        ax.axvline(2.5, color="white", linewidth=3)
        for index in (0.5, 1.5, 3.5):
            ax.axhline(index, color="white", linewidth=1.5)
        for y in range(8):
            for x in range(6):
                value = matrix[y, x]
                label = f"{value:+.3f}" if unit == "m/s" else f"{value:+.1f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=8.5)
    fig.suptitle(
        f"{title}: PPO − TRS-PPO\nPositive = improvement; means of 100 parallel trials per cell",
        fontsize=14,
    )
    fig.colorbar(
        color,
        ax=list(axes),
        fraction=0.02,
        pad=0.02,
        label=f"Reduction [{unit}]; colors clipped at ±{limit:g}",
    )
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def _write_tables(comparisons: list[dict], metadata: dict) -> None:
    lookup = {
        (r["robot"], r["gait_name"], r["metric"]): r
        for r in comparisons
        if r["scope"] == "all"
    }
    sections = [
        "# PPO versus TRS-PPO by individual gait",
        "",
        "Updated 2026-10-28. Each gait mean equally weights the six signed velocity commands after averaging",
        "100 parallel trials per command. These are 34 selected metrics, matching the main report's scope.",
        "Changes are TRS-PPO minus PPO; the direction indicating improvement is stated for each metric.",
        "Speed-condition win counts describe the six commands and are not independent training replications.",
        "",
        "**Phase-clock caveat:** saved desired-contact agreement, classification, gait-only success, and",
        "joint success need signed-clock reanalysis, particularly for backward commands. Their numbers",
        "are provided for traceability. See the main report for the archive audit.",
        "",
    ]
    for field, meta in metadata.items():
        direction = (
            "Lower is better"
            if meta["lower_is_better"] == "True"
            else "Higher is better"
        )
        sections.extend(
            [
                f"## {meta['label']} [{meta['unit']}]",
                "",
                f"{direction}.",
                "",
                "| Robot | Gait | PPO | TRS-PPO | Change | Relative change | Commands improved / 6 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for robot in ROBOTS:
            for gait in GAITS:
                row = lookup[(robot, gait, field)]
                relative = (
                    f"{row['relative_change_percent']:+.1f}%"
                    if math.isfinite(row["relative_change_percent"])
                    else "N/A (PPO = 0)"
                )
                sections.append(
                    f"| {ROBOT_LABELS[robot]} | `{gait}` | {row['ppo']:.6g} | {row['trs']:.6g} | "
                    f"{row['delta_trs_minus_ppo']:+.6g} | {relative} | {row['improved_conditions']}/6 |"
                )
        sections.append("")
    (OUT / "all_gait_metric_tables.md").write_text("\n".join(sections) + "\n")

    compact = []
    for robot in ROBOTS:
        compact.extend(
            [
                f"### {ROBOT_LABELS[robot]}",
                "",
                "| Gait | Body-x RMSE [m/s] | Torque imbalance [%] | Work imbalance [%] | Mechanical power [W] | Target clipping [%] |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for gait in GAITS:
            values = []
            for field, precision, scale in (
                ("velocity_vx_rmse_mps", 3, 1),
                ("torque_squared_abs_imbalance_percent", 2, 1),
                ("absolute_work_abs_imbalance_percent", 2, 1),
                ("absolute_work_total_per_s", 1, 1),
                ("load_action_clamp_fraction", 2, 100),
            ):
                row = lookup[(robot, gait, field)]
                values.append(
                    f"{row['ppo'] * scale:.{precision}f} → {row['trs'] * scale:.{precision}f}"
                )
            compact.append(f"| `{gait}` | " + " | ".join(values) + " |")
        compact.append("")
    (OUT / "gait_key_tables.md").write_text("\n".join(compact) + "\n")


def _compare_reversal_gaits(provenance: dict, condition_rows: list[dict]) -> None:
    groups = defaultdict(list)
    partners = {}
    names_by_index = {
        (r["robot"], r["gait_index"]): r["gait_name"] for r in condition_rows
    }
    for source in provenance["sources"]:
        directory = ROOT / source["run"] / "evaluations"
        matches = list(directory.glob("*/metrics/paired_tr_consistency.csv"))
        assert len(matches) == 1
        for row in _read_csv(matches[0]):
            groups[
                (
                    "saved_full",
                    source["robot"],
                    source["method"],
                    row["gait_name"],
                    row["operator"],
                )
            ].append(row)
            partners[(source["robot"], row["gait_name"])] = row["partner_gait_name"]
    for row in _read_csv(OUT / "signed_clock_reversal_trial0.csv"):
        gait = names_by_index[(row["robot"], int(row["gait_index"]))]
        groups[
            ("signed_clock_trial0", row["robot"], row["method"], gait, row["operator"])
        ].append(row)
    metrics = [
        key
        for key in next(iter(groups.values()))[0]
        if key.startswith("paired_") and not key.endswith(("_valid", "_reason"))
    ]
    results = []
    for protocol in ("saved_full", "signed_clock_trial0"):
        for robot in ROBOTS:
            for gait in GAITS:
                for operator in (
                    "temporal_reversal",
                    "front_hind_reflected_temporal_reversal",
                ):
                    before_rows = groups[(protocol, robot, "ppo", gait, operator)]
                    after_rows = groups[(protocol, robot, "trs", gait, operator)]
                    assert (
                        len(before_rows)
                        == len(after_rows)
                        == (300 if protocol == "saved_full" else 3)
                    )
                    for field in metrics:
                        before = float(
                            np.mean([float(row[field]) for row in before_rows])
                        )
                        after = float(
                            np.mean([float(row[field]) for row in after_rows])
                        )
                        results.append(
                            {
                                "protocol": protocol,
                                "robot": robot,
                                "positive_gait_name": gait,
                                "negative_partner_gait_name": partners[(robot, gait)],
                                "operator": operator,
                                "metric": field,
                                "ppo": before,
                                "trs": after,
                                "delta_trs_minus_ppo": after - before,
                                "relative_change_percent": _relative_change(
                                    before, after
                                ),
                                "pairs_per_policy": len(before_rows),
                            }
                        )
    assert len(metrics) == 12 and len(results) == 1152
    _write_csv("gait_reversal_comparison.csv", results)


def main() -> None:
    """Regenerate gait comparisons and five figures from the base analysis outputs."""
    provenance = json.loads((OUT / "provenance.json").read_text())
    for source in provenance["sources"]:
        for relative, expected_hash in source["input_sha256"].items():
            assert (
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                == expected_hash
            ), relative
    aggregate_rows = _read_csv(OUT / "metric_comparison.csv")
    metadata = {r["metric"]: r for r in aggregate_rows}
    conditions = _read_csv(OUT / "condition_comparison.csv")
    for row in conditions:
        row["gait_index"] = int(row["gait_index"])
        for field in ("velocity_mps", "ppo", "trs", "delta_trs_minus_ppo"):
            row[field] = float(row[field])
    comparisons = _aggregate_gaits(conditions, metadata)
    for aggregate in aggregate_rows:
        selected = [
            row
            for row in comparisons
            if row["robot"] == aggregate["robot"]
            and row["metric"] == aggregate["metric"]
            and row["scope"] == "all"
        ]
        for method in ("ppo", "trs"):
            reconstructed = np.mean(
                [
                    np.mean(
                        [row[method] for row in selected if row["family"] == family]
                    )
                    for family in ("trot", "bound", "half_bound", "gallop")
                ]
            )
            assert math.isclose(
                reconstructed, float(aggregate[method]), rel_tol=1e-11, abs_tol=1e-11
            )
    _write_csv("gait_metric_comparison.csv", comparisons)
    _write_tables(comparisons, metadata)
    lookup = {
        (r["robot"], r["gait_name"], r["metric"]): r
        for r in comparisons
        if r["scope"] == "all"
    }
    _plot_gait_overviews(lookup)
    _plot_velocity_grid(
        conditions,
        "torque_squared_abs_imbalance_percent",
        "Front/hind raw torque imbalance",
        "percentage points",
        25,
        "gait_velocity_torque_balance.png",
    )
    _plot_velocity_grid(
        conditions,
        "velocity_vx_rmse_mps",
        "Body-x velocity RMSE",
        "m/s",
        0.15,
        "gait_velocity_tracking.png",
    )
    _compare_reversal_gaits(provenance, conditions)
    print(
        f"Wrote {len(comparisons)} gait/direction comparisons, 1,152 reversal comparisons, and five gait figures."
    )


if __name__ == "__main__":
    main()
