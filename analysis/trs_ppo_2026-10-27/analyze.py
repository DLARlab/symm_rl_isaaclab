# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce the tables and figures for the six final-checkpoint evaluations.

Run from the repository root with ``./isaaclab.sh -p analysis/trs_ppo_2026-10-27/analyze.py``.
Existing evaluation outputs are read without modification. Confidence intervals
resample gait/velocity conditions within families, not individual parallel trials.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/symm_locomotion"))
import _leg_usage_metrics as metrics
import paired_tr_consistency as paired

ROBOTS = ("go2", "x1", "x1_go2bend")
FAMILIES = ("trot", "bound", "half_bound", "gallop")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261027
# key, human-readable label, units, whether a lower value is desirable
FIELDS = (
    ("velocity_vx_rmse_mps", "Body-x velocity RMSE", "m/s", True),
    ("velocity_vx_relative_rmse", "Relative body-x velocity RMSE", "fraction", True),
    ("velocity_vy_rmse_mps", "Lateral velocity RMSE", "m/s", True),
    ("velocity_yaw_rmse_radps", "Yaw-rate RMSE", "rad/s", True),
    ("velocity_heading_rmse_rad", "Heading RMSE", "rad", True),
    ("velocity_lateral_position_rmse_m", "Lateral position RMSE", "m", True),
    (
        "velocity_progress_per_commanded_distance",
        "World-x progress / commanded distance",
        "fraction",
        False,
    ),
    (
        "gait_agreement_boundary_excluded",
        "Saved gait contact agreement (phase caveat)",
        "fraction",
        False,
    ),
    ("gait_duty_factor_abs_error_mean", "Duty-factor absolute error", "fraction", True),
    ("gait_row_correct", "Saved gait-row classification accuracy", "fraction", False),
    (
        "gait_family_correct",
        "Saved gait-family classification accuracy",
        "fraction",
        False,
    ),
    ("velocity_only_success", "Strict velocity-only success", "fraction", False),
    ("gait_only_success", "Saved gait-only success (phase caveat)", "fraction", False),
    (
        "joint_velocity_and_gait_success",
        "Saved joint success (phase caveat)",
        "fraction",
        False,
    ),
    ("success_positive_progress", "Positive world-x progress", "fraction", False),
    ("success_vx_relative_rmse", "Relative vx RMSE <= 0.20", "fraction", False),
    ("success_yaw_rmse", "Yaw RMSE <= 0.05 rad/s", "fraction", False),
    (
        "torque_squared_abs_imbalance_percent",
        "Raw torque front/hind imbalance",
        "%",
        True,
    ),
    (
        "normalized_torque_utilization_abs_imbalance_percent",
        "Normalized torque front/hind imbalance",
        "%",
        True,
    ),
    ("absolute_work_abs_imbalance_percent", "Work front/hind imbalance", "%", True),
    (
        "vertical_grf_impulse_abs_imbalance_percent",
        "Vertical support front/hind imbalance",
        "%",
        True,
    ),
    ("torque_squared_total_per_s", "Total raw torque-squared rate", "N^2 m^2", True),
    (
        "normalized_torque_utilization_total_per_s",
        "Total normalized torque-squared rate",
        "1",
        True,
    ),
    ("absolute_work_total_per_s", "Absolute mechanical power", "W", True),
    ("load_torque_squared_cv", "Four-leg raw torque CV", "1", True),
    ("load_normalized_torque_squared_cv", "Four-leg normalized torque CV", "1", True),
    ("load_absolute_work_cv", "Four-leg work CV", "1", True),
    ("load_vertical_grf_impulse_cv", "Four-leg support CV", "1", True),
    ("load_absolute_work_max_share", "Largest leg's work share", "fraction", True),
    (
        "load_normalized_torque_squared_max_share",
        "Largest leg's normalized torque share",
        "fraction",
        True,
    ),
    ("load_action_clamp_fraction", "Action target clipping fraction", "fraction", True),
    (
        "joint_saturation_mean",
        "Mean joint torque saturation fraction",
        "fraction",
        True,
    ),
    (
        "joint_utilization_p95_mean",
        "Mean joint torque utilization p95",
        "fraction",
        True,
    ),
    ("foot_impact_peak_p95_max", "Largest per-foot impact peak p95", "N", True),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: str | float | None) -> float:
    if value in ("True", "False"):
        return float(value == "True")
    try:
        return float(value)
    except (ValueError, TypeError):
        return math.nan


def mean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    valid = values[np.isfinite(values)]
    return float(np.mean(valid)) if valid.size else math.nan


def family_mean(rows: list[dict], field: str) -> float:
    return mean(
        mean(r[field] for r in rows if r["family"] == family) for family in FAMILIES
    )


def flatten(value, prefix="") -> dict:
    if isinstance(value, dict):
        return {
            key: item
            for name, child in value.items()
            for key, item in flatten(child, prefix + "/" + name).items()
        }
    return {prefix: value}


def scenario_interval(
    ppo: list[dict], trs: list[dict], field: str
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = np.zeros(BOOTSTRAP_REPLICATES)
    for family in FAMILIES:
        first = {r["condition"]: r[field] for r in ppo if r["family"] == family}
        second = {r["condition"]: r[field] for r in trs if r["family"] == family}
        delta = np.asarray([second[key] - first[key] for key in sorted(first)])
        if not np.all(np.isfinite(delta)):
            return math.nan, math.nan
        indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
        replicates += np.mean(delta[indices], axis=1) / len(FAMILIES)
    return tuple(float(v) for v in np.quantile(replicates, [0.025, 0.975]))


def audit_phases(studies: dict) -> tuple[list[dict], list[dict]]:
    """Audit one representative archive per gait/velocity condition."""
    phase_rows, pair_rows = [], []
    for (robot, method), entry in studies.items():
        study, directory = entry["study"], entry["directory"]
        selected = [cell for cell in study["cells"] if cell["trial_index"] == 0]
        traces = {}
        by_id = {row["cell_id"]: row for row in entry["rows"]}
        for cell in selected:
            saved = by_id[cell["id"]]
            archive = directory / cell["relative_output_dir"] / "sim_data.npz"
            with np.load(archive) as arrays:
                times = arrays["time_steps"]
                window = (times >= saved["measurement_start_s"] - 1e-6) & (
                    times <= saved["measurement_stop_s"] + 1e-6
                )
                actual_phase = arrays["foot_periodicity_phases"][window]
                unsigned = np.remainder(
                    arrays["common_gait_phases"][window, None]
                    + arrays["foot_thetas"][window],
                    1.0,
                )
                signed = np.remainder(
                    np.sign(cell["velocity_mps"])
                    * (
                        arrays["common_gait_phases"][window, None]
                        + arrays["foot_thetas"][window]
                    ),
                    1.0,
                )
                unsigned_error = np.abs(
                    np.remainder(actual_phase - unsigned + 0.5, 1.0) - 0.5
                )
                signed_error = np.abs(
                    np.remainder(actual_phase - signed + 0.5, 1.0) - 0.5
                )
                dwell = max(
                    1,
                    int(
                        math.ceil(
                            study["evaluation_config"]["contact"]["minimum_dwell_s"]
                            / study["step_dt"]
                            - 1e-12
                        )
                    ),
                )
                contacts = metrics.hysteresis_contacts(
                    np.maximum(arrays["foot_normal_forces_w"][window, :, 2], 0.0),
                    float(arrays["contact_threshold_on_n"]),
                    float(arrays["contact_threshold_off_n"]),
                    dwell,
                )
                stance_start = 1.0 - arrays["duty_factors"][window, None]
                boundary = study["evaluation_config"]["gait"][
                    "boundary_exclusion_cycles"
                ]
                mask = (np.minimum(actual_phase, 1.0 - actual_phase) > boundary) & (
                    np.abs(np.remainder(actual_phase - stance_start + 0.5, 1.0) - 0.5)
                    > boundary
                )
                phase_rows.append(
                    {
                        "robot": robot,
                        "method": method,
                        "gait_name": cell["gait_name"],
                        "family": cell["family"],
                        "velocity_mps": cell["velocity_mps"],
                        "trial_index": 0,
                        "unsigned_phase_error_cycles": float(np.mean(unsigned_error)),
                        "signed_phase_error_cycles": float(np.mean(signed_error)),
                        "saved_agreement": saved["gait_agreement_boundary_excluded"],
                        "recorded_phase_agreement": float(
                            np.mean((contacts == (actual_phase >= stance_start))[mask])
                        ),
                    }
                )
                assert float(np.mean(signed_error)) < 1e-3
            traces[(cell["gait_index"], cell["velocity_mps"])] = (
                paired._load_cell_trace(directory, cell, robot)
            )
        gaits = {gait["index"]: gait for gait in study["gaits"]}
        for cell in selected:
            if cell["velocity_mps"] <= 0.0:
                continue
            source = traces[(cell["gait_index"], cell["velocity_mps"])]
            partner = gaits[cell["gait_index"]]["time_reversal_partner"]
            target = traces[(partner, -cell["velocity_mps"])]
            for operator in paired.OPERATORS:
                # Override only the diagnostic's alignment intercept, using the
                # observed signed-clock relationship phi_source = -phi_target.
                # This is an offline sensitivity probe, not a production change.
                result = paired.compare_pair(
                    replace(source, swing_ratio=0.0),
                    replace(target, swing_ratio=0.0),
                    robot=robot,
                    operator=operator,
                )
                pair_rows.append(
                    {
                        "robot": robot,
                        "method": method,
                        "gait_index": cell["gait_index"],
                        "family": cell["family"],
                        "speed_mps": cell["velocity_mps"],
                        "operator": operator,
                        **{key: result[key] for key in paired.METRICS},
                    }
                )
        print(f"Audited 48 representative archives: {robot} {method}", flush=True)
    return phase_rows, pair_rows


def main() -> None:
    studies, sources = {}, []
    for overall_path in sorted(
        (ROOT / "logs/rsl_rl").glob("*/*/evaluations/*/metrics/overall_metrics.json")
    ):
        directory = overall_path.parent.parent
        study_path = directory / "study.json"
        study = json.loads(study_path.read_text())
        robot = study["robot"]
        run = directory.parents[1]
        method = "trs" if "with_trs" in run.name else "ppo"
        if method == "ppo" and "merged_0-19999" not in run.name:
            continue
        assert (robot, method) not in studies
        assert study["checkpoint"]["iteration"] == 19999
        raw_rows = read_csv(overall_path.parent / "cell_metrics.csv")
        assert len(raw_rows) == len(study["cells"]) == 4800
        assert {int(row["seed"]) for row in raw_rows} == {42}
        rows = []
        for raw in raw_rows:
            row = {key: number(value) for key, value in raw.items()}
            for key in ("cell_id", "gait_name", "family", "relative_output_dir"):
                row[key] = raw[key]
            row["condition"] = (int(raw["gait_index"]), float(raw["velocity_mps"]))
            for source, target in (
                ("load_joint_saturation_fraction", "joint_saturation_mean"),
                ("load_joint_utilization_p95", "joint_utilization_p95_mean"),
            ):
                row[target] = mean(json.loads(raw[source])) if raw[source] else math.nan
            row["foot_impact_peak_p95_max"] = max(
                number(raw[f"load_{leg}_impact_peak_p95_n"])
                for leg in ("fl", "fr", "rl", "rr")
            )
            rows.append(row)
        by_condition = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
        conditions = [
            {
                "condition": key,
                "gait_index": key[0],
                "velocity_mps": key[1],
                "gait_name": group[0]["gait_name"],
                "family": group[0]["family"],
                **{field: mean(row[field] for row in group) for field, *_ in FIELDS},
            }
            for key, group in sorted(by_condition.items())
        ]
        assert len(conditions) == 48 and all(
            len(group) == 100 for group in by_condition.values()
        )
        overall = json.loads(overall_path.read_text())
        for field in (
            "torque_squared",
            "normalized_torque_utilization",
            "absolute_work",
            "vertical_grf_impulse",
        ):
            computed = family_mean(conditions, field + "_abs_imbalance_percent")
            expected = overall["metrics"][field][
                "family_balanced_mean_abs_imbalance_percent"
            ]
            assert math.isclose(computed, expected, rel_tol=1e-11, abs_tol=1e-11)
        studies[(robot, method)] = {
            "directory": directory,
            "study": study,
            "rows": rows,
            "conditions": conditions,
            "overall": overall,
        }
        inputs = {}
        for path in (
            study_path,
            overall_path,
            overall_path.parent / "cell_metrics.csv",
            overall_path.parent / "paired_tr_consistency.csv",
        ):
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        sources.append(
            {
                "robot": robot,
                "method": method,
                "run": str(run.relative_to(ROOT)),
                "checkpoint_sha256": study["checkpoint"]["sha256"],
                "input_sha256": inputs,
                "duplicate_vx_rmse_conditions": sum(
                    len({row["velocity_vx_rmse_mps"] for row in group}) == 1
                    for group in by_condition.values()
                ),
            }
        )
    assert set(studies) == {
        (robot, method) for robot in ROBOTS for method in ("ppo", "trs")
    }

    (
        summary,
        condition_comparisons,
        strata,
        common_support,
        paired_summary,
        config_audit,
    ) = ([], [], [], [], [], [])
    for robot in ROBOTS:
        ppo, trs = (studies[(robot, method)] for method in ("ppo", "trs"))
        assert {row["cell_id"] for row in ppo["rows"]} == {
            row["cell_id"] for row in trs["rows"]
        }
        for section in (
            "runtime_overrides",
            "evaluation_config",
            "source_provenance",
            "policy_compatibility",
            "gaits",
        ):
            assert ppo["study"][section] == trs["study"][section]
        for name in ("agent.yaml", "env.yaml"):
            configs = []
            for entry in (ppo, trs):
                path = entry["directory"].parents[1] / "params" / name
                configs.append(
                    flatten(yaml.load(path.read_text(), Loader=yaml.BaseLoader))
                )
            first, second = configs
            config_audit.extend(
                {
                    "robot": robot,
                    "config": name,
                    "key": key,
                    "ppo": first.get(key),
                    "trs": second.get(key),
                }
                for key in sorted(first.keys() | second.keys())
                if first.get(key) != second.get(key)
            )
        first = {row["condition"]: row for row in ppo["conditions"]}
        second = {row["condition"]: row for row in trs["conditions"]}
        for field, label, unit, lower in FIELDS:
            before, after = (
                family_mean(entry["conditions"], field) for entry in (ppo, trs)
            )
            lo, hi = scenario_interval(ppo["conditions"], trs["conditions"], field)
            differences = np.asarray(
                [second[key][field] - first[key][field] for key in sorted(first)]
            )
            finite = np.isfinite(differences)
            directed = differences * (1 if lower else -1)
            summary.append(
                {
                    "robot": robot,
                    "metric": field,
                    "label": label,
                    "unit": unit,
                    "lower_is_better": lower,
                    "ppo": before,
                    "trs": after,
                    "delta_trs_minus_ppo": after - before,
                    "relative_change_percent": (
                        100.0 * (after / before - 1.0) if before else math.nan
                    ),
                    "scenario_interval_low": lo,
                    "scenario_interval_high": hi,
                    "improved_conditions": int(np.sum(directed[finite] < -1e-12)),
                    "worsened_conditions": int(np.sum(directed[finite] > 1e-12)),
                    "available_conditions": int(np.sum(finite)),
                }
            )
            for key in sorted(first):
                condition_comparisons.append(
                    {
                        "robot": robot,
                        "gait_index": key[0],
                        "gait_name": first[key]["gait_name"],
                        "family": first[key]["family"],
                        "velocity_mps": key[1],
                        "metric": field,
                        "ppo": first[key][field],
                        "trs": second[key][field],
                        "delta_trs_minus_ppo": second[key][field] - first[key][field],
                    }
                )
            for scope, names in (
                ("family", FAMILIES),
                ("direction", ("backward", "forward")),
                ("speed", (0.5, 1.0, 2.0)),
            ):
                for name in names:

                    def selected(row):
                        if scope == "family":
                            return row["family"] == name
                        if scope == "direction":
                            return (row["velocity_mps"] < 0.0) == (name == "backward")
                        return abs(row["velocity_mps"]) == name

                    values = [
                        family_mean(
                            [row for row in entry["conditions"] if selected(row)], field
                        )
                        for entry in (ppo, trs)
                    ]
                    strata.append(
                        {
                            "robot": robot,
                            "scope": scope,
                            "stratum": name,
                            "metric": field,
                            "ppo": values[0],
                            "trs": values[1],
                            "delta": values[1] - values[0],
                        }
                    )
        second_rows = {row["cell_id"]: row for row in trs["rows"]}
        for name in ("positive_progress", "tracking_success", "velocity_only_success"):
            first_rows = [
                row
                for row in ppo["rows"]
                if (
                    row["signed_directed_progress_m"] > 0
                    and second_rows[row["cell_id"]]["signed_directed_progress_m"] > 0
                )
                and (
                    name == "positive_progress"
                    or (row[name] == 1 and second_rows[row["cell_id"]][name] == 1)
                )
            ]
            other_rows = [second_rows[row["cell_id"]] for row in first_rows]
            for field in (
                "absolute_work_total_per_s",
                "absolute_work_total_per_directed_m",
                "torque_squared_abs_imbalance_percent",
                "absolute_work_abs_imbalance_percent",
                "normalized_torque_utilization_total_per_s",
            ):
                supported = []
                for rows in (first_rows, other_rows):
                    grouped = defaultdict(list)
                    for row in rows:
                        grouped[row["condition"]].append(row)
                    supported.append(
                        [
                            {
                                "family": group[0]["family"],
                                field: mean(row[field] for row in group),
                            }
                            for group in grouped.values()
                        ]
                    )
                before, after = (family_mean(rows, field) for rows in supported)
                common_support.append(
                    {
                        "robot": robot,
                        "filter": name,
                        "rows": len(first_rows),
                        "conditions": len({row["condition"] for row in first_rows}),
                        "families": ",".join(
                            f
                            for f in FAMILIES
                            if any(row["family"] == f for row in first_rows)
                        ),
                        "metric": field,
                        "ppo_observed_family_mean": before,
                        "trs_observed_family_mean": after,
                    }
                )
        for method, entry in (("ppo", ppo), ("trs", trs)):
            rows = read_csv(entry["directory"] / "metrics/paired_tr_consistency.csv")
            for operator in paired.OPERATORS:
                chosen = [row for row in rows if row["operator"] == operator]
                for field in paired.METRICS:
                    assert all(row[field + "_valid"] == "True" for row in chosen)
                    paired_summary.append(
                        {
                            "robot": robot,
                            "method": method,
                            "operator": operator,
                            "metric": field,
                            "family_balanced_mean": mean(
                                mean(
                                    number(row[field])
                                    for row in chosen
                                    if row["family"] == family
                                )
                                for family in FAMILIES
                            ),
                            "raw_mean": mean(number(row[field]) for row in chosen),
                            "valid_pairs": len(chosen),
                        }
                    )

    write_csv("metric_comparison.csv", summary)
    write_csv("condition_comparison.csv", condition_comparisons)
    write_csv("stratified_comparison.csv", strata)
    write_csv("common_support.csv", common_support)
    write_csv("saved_reversal_comparison.csv", paired_summary)
    write_csv("config_differences.csv", config_audit)
    phase_rows, signed_pairs = audit_phases(studies)
    write_csv("phase_clock_audit.csv", phase_rows)
    write_csv("signed_clock_reversal_trial0.csv", signed_pairs)
    write_csv(
        "success_counts.csv",
        [
            {
                "robot": robot,
                "method": method,
                **{
                    key: entry["overall"]["coverage"][key]
                    for key in (
                        "expected_cells",
                        "valid_cells",
                        "tracking_success_cells",
                        "velocity_only_success_cells",
                        "gait_only_success_cells",
                        "joint_success_cells",
                    )
                },
                "terminated_cells": sum(
                    row["terminated"] == 1 for row in entry["rows"]
                ),
            }
            for (robot, method), entry in studies.items()
        ],
    )
    manifest = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "weighting": "equal families; equal gait/velocity conditions within each family; mean 100 trials per condition",
        "phase_audit": "trial 000 for all 48 conditions in each of six studies",
        "sources": sources,
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")

    plotted = [
        field
        for field, *_ in FIELDS
        if field
        in (
            "velocity_vx_rmse_mps",
            "velocity_vy_rmse_mps",
            "velocity_yaw_rmse_radps",
            "velocity_heading_rmse_rad",
            "gait_duty_factor_abs_error_mean",
            "torque_squared_abs_imbalance_percent",
            "normalized_torque_utilization_abs_imbalance_percent",
            "absolute_work_abs_imbalance_percent",
            "vertical_grf_impulse_abs_imbalance_percent",
            "torque_squared_total_per_s",
            "normalized_torque_utilization_total_per_s",
            "absolute_work_total_per_s",
            "load_absolute_work_cv",
            "load_action_clamp_fraction",
        )
    ]
    lookup = {(row["robot"], row["metric"]): row for row in summary}
    matrix = np.asarray(
        [
            [-lookup[(robot, field)]["relative_change_percent"] for robot in ROBOTS]
            for field in plotted
        ]
    )
    fig, ax = plt.subplots(figsize=(10.2, 8.5), layout="constrained")
    color = ax.imshow(matrix, cmap="RdYlGn", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(range(3), ("Go2", "X1", "X1 Go2-bend"))
    ax.set_yticks(
        range(len(plotted)), [lookup[("go2", field)]["label"] for field in plotted]
    )
    for y in range(len(plotted)):
        for x in range(3):
            ax.text(x, y, f"{matrix[y,x]:+.1f}%", ha="center", va="center", fontsize=10)
    ax.set_title(
        "TRS-PPO relative improvement over PPO\nFamily-balanced means; positive values indicate lower error/load",
        pad=14,
    )
    fig.colorbar(
        color,
        ax=ax,
        shrink=0.55,
        label="Relative reduction [%]; colors limited to ±50%",
    )
    fig.savefig(OUT / "metric_improvements.png", dpi=180)
    plt.close(fig)

    sections = []
    for field, label, unit, _ in FIELDS:
        sections += [
            f"### {label} [{unit}]",
            "",
            "| Robot | PPO | TRS-PPO | Change | Relative change | Conditions improved | Scenario interval for change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for robot in ROBOTS:
            row = lookup[(robot, field)]
            sections.append(
                f"| {robot} | {row['ppo']:.6g} | {row['trs']:.6g} | {row['delta_trs_minus_ppo']:+.6g} | {row['relative_change_percent']:+.2f}% | {row['improved_conditions']}/{row['available_conditions']} | [{row['scenario_interval_low']:+.6g}, {row['scenario_interval_high']:+.6g}] |"
            )
        sections.append("")
    (OUT / "all_metric_tables.md").write_text("\n".join(sections) + "\n")
    print(
        f"Wrote {len(summary)} aggregate comparisons and {len(condition_comparisons)} matched-condition comparisons."
    )
    from gait_analysis import main as analyze_gaits

    analyze_gaits()


if __name__ == "__main__":
    main()
