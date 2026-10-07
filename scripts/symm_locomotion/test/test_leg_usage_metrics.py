# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic tests for the simulator-independent leg-usage metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gaits() -> tuple[dict, ...]:
    names = (
        "trot",
        "bound",
        "half_bound_front_a",
        "half_bound_front_b",
        "half_bound_hind_a",
        "half_bound_hind_b",
        "gallop_a",
        "gallop_b",
        "gallop_c",
        "gallop_d",
    )
    families = (
        "trot",
        "bound",
        "half_bound",
        "half_bound",
        "half_bound",
        "half_bound",
        "gallop",
        "gallop",
        "gallop",
        "gallop",
    )
    phases = (
        (0.0, 0.5, 0.5, 0.0),
        (0.0, 0.0, 0.5, 0.5),
        (0.0, 0.25, 0.5, 0.75),
        (0.0, 0.75, 0.5, 0.25),
        (0.0, 0.75, 0.25, 0.5),
        (0.0, 0.25, 0.75, 0.5),
        (0.0, 0.15, 0.55, 0.75),
        (0.0, 0.25, 0.65, 0.85),
        (0.0, 0.85, 0.45, 0.25),
        (0.0, 0.75, 0.35, 0.15),
    )
    partners = (0, 1, 3, 2, 5, 4, 8, 9, 6, 7)
    return tuple(
        {
            "index": index,
            "name": names[index],
            "family": families[index],
            "phases": list(phases[index]),
            "weight": 1.0,
            "training_weight": 1.0,
            "time_reversal_partner": partners[index],
        }
        for index in range(10)
    )


def test_hysteresis_rejects_chatter_and_enforces_dwell():
    metrics = _load("leg_usage_metrics_hysteresis", "_leg_usage_metrics.py")
    force = np.zeros((9, 4))
    force[:, 0] = [0.0, 12.0, 0.0, 12.0, 13.0, 11.0, 4.0, 3.0, 0.0]

    contact = metrics.hysteresis_contacts(force, 10.0, 5.0, minimum_dwell_samples=2)

    assert contact[:, 0].tolist() == [False, False, False, True, True, True, False, False, False]
    assert not np.any(contact[:, 1:])


def test_contact_error_rates_use_desired_state_conditional_denominators():
    metrics = _load("leg_usage_metrics_conditional_rates", "_leg_usage_metrics.py")
    desired = np.asarray([False, False, True, True, True, True])
    actual = np.asarray([True, False, False, True, True, True])

    result = metrics._classification_metrics(actual, desired, np.ones_like(actual, dtype=bool))

    assert result["false_swing_contact"] == pytest.approx(0.5)
    assert result["missed_stance_contact"] == pytest.approx(0.25)
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.75)


def test_event_matching_never_crosses_common_cycle_identity():
    metrics = _load("leg_usage_metrics_cycle_matching", "_leg_usage_metrics.py")
    common = np.asarray([0.99, 1.01])
    foot_phase = np.asarray([0.50, 0.50])
    target = np.asarray([0.50, 0.50])

    errors, unmatched_expected, unmatched_actual = metrics._match_events(
        np.asarray([1]),
        np.asarray([0]),
        common,
        foot_phase,
        target,
        0.20,
    )

    assert errors == []
    assert unmatched_expected == 1
    assert unmatched_actual == 1


def test_ideal_trot_recovers_contact_events_order_and_classifier():
    metrics = _load("leg_usage_metrics_gait", "_leg_usage_metrics.py")
    step_dt = 0.01
    common = np.arange(0.0, 4.0, step_dt)
    offsets = np.asarray(_gaits()[0]["phases"])
    desired, boundary_mask, _ = metrics.desired_stance_states(common, offsets, 0.5, 0.02)
    force = np.where(desired, 100.0, 0.0)
    config = metrics.evaluation_config(
        {
            "contact": {"mode": "absolute", "on_n": 20.0, "off_n": 10.0, "minimum_dwell_s": 0.01},
            "gait": {"classifier_min_margin_cycles": 0.0},
        }
    )

    result, contact = metrics.gait_metrics(
        force,
        common,
        offsets,
        0.5,
        step_dt,
        config,
        robot_mass_kg=None,
        gait_library=_gaits(),
    )

    assert np.array_equal(contact, desired)
    assert 0.0 < 1.0 - float(np.mean(boundary_mask)) < 0.2
    assert result["agreement"] == pytest.approx(1.0)
    assert result["agreement_boundary_excluded"] == pytest.approx(1.0)
    assert result["contact_f1"] == pytest.approx(1.0)
    assert result["touchdown_abs_error_p95_cycles"] == pytest.approx(0.0, abs=0.011)
    assert result["liftoff_abs_error_p95_cycles"] == pytest.approx(0.0, abs=0.011)
    assert result["touchdown_unmatched_expected"] == 0
    assert result["same_phase_sync_p95_s"] == pytest.approx(0.0)
    assert result["touchdown_same_phase_sync_p95_cycles"] == pytest.approx(0.0)
    assert result["liftoff_same_phase_sync_p95_s"] == pytest.approx(0.0)
    assert result["measured_duty_factor_per_foot"] == pytest.approx([0.5] * 4)
    assert result["duty_factor_abs_error_per_foot"] == pytest.approx([0.0] * 4)
    assert result["touchdown_unmatched_fraction"] == pytest.approx(0.0)
    assert result["liftoff_unmatched_fraction"] == pytest.approx(0.0)
    assert result["measured_cyclic_order"] in {"fl+rr>fr+rl", "fr+rl>fl+rr"}
    assert result["classified_row"] == "trot"
    assert result["classified_family"] == "trot"
    assert len(result["classified_pairwise_touchdown_phase_differences"]) == 6
    assert result["complete_cycles"] == 4
    assert result["coverage_valid"] is True
    for foot in result["per_foot_metrics"]:
        assert foot["agreement_boundary_excluded"] == pytest.approx(1.0)
        assert foot["contact_f1"] == pytest.approx(1.0)
        assert foot["duty_factor_abs_error"] == pytest.approx(0.0)
        assert foot["touchdown_abs_error_p95_cycles"] == pytest.approx(0.0, abs=0.011)
        assert foot["liftoff_abs_error_p95_cycles"] == pytest.approx(0.0, abs=0.011)
        assert foot["touchdown_unmatched_fraction"] == pytest.approx(0.0)
        assert foot["liftoff_unmatched_fraction"] == pytest.approx(0.0)
    touchdown_pairs = result["touchdown_same_phase_sync_pair_records"]
    liftoff_pairs = result["liftoff_same_phase_sync_pair_records"]
    assert {(record["first_foot"], record["second_foot"]) for record in touchdown_pairs} == {
        ("fl", "rr"),
        ("fr", "rl"),
    }
    assert sorted(record["matched_complete_cycles"] for record in touchdown_pairs) == [3, 4]
    assert all(record["eligible_complete_cycles"] == 4 for record in touchdown_pairs + liftoff_pairs)
    assert all(record["unmatched_complete_cycles"] >= 0 for record in touchdown_pairs + liftoff_pairs)
    assert result["touchdown_same_phase_sync_eligible_pair_cycles"] == 8
    assert result["touchdown_same_phase_sync_matched_pair_cycles"] == 7
    assert result["touchdown_same_phase_sync_unmatched_pair_cycles"] == 1
    assert all(record["p95_s"] == pytest.approx(0.0) for record in touchdown_pairs + liftoff_pairs)


def test_gait_classifier_fails_closed_without_minimum_events():
    metrics = _load("leg_usage_metrics_gait_coverage", "_leg_usage_metrics.py")
    step_dt = 0.01
    common = np.arange(0.0, 4.0, step_dt)
    force = np.zeros((len(common), 4))

    result, contact = metrics.gait_metrics(
        force,
        common,
        _gaits()[0]["phases"],
        0.5,
        step_dt,
        metrics.evaluation_config(),
        robot_mass_kg=20.0,
        gait_library=_gaits(),
    )

    assert not np.any(contact)
    assert result["coverage_valid"] is False
    assert "touchdown_events_per_foot" in result["coverage_reason"]
    assert result["classified_row"] == "unclassified"
    assert result["classified_family"] == "unclassified"
    assert result["classified_reason"] == result["coverage_reason"]


def test_duty_error_uses_commanded_beta_not_sampled_binary_fraction():
    metrics = _load("leg_usage_metrics_commanded_duty", "_leg_usage_metrics.py")
    common = np.arange(0.50, 1.00, 0.10)
    offsets = np.zeros(4)
    desired, _, _ = metrics.desired_stance_states(common, offsets, 0.5, 0.0)
    force = np.where(desired, 100.0, 0.0)
    config = metrics.evaluation_config(
        {
            "contact": {"mode": "absolute", "on_n": 20.0, "off_n": 10.0, "minimum_dwell_s": 0.0},
            "gait": {"minimum_complete_cycles": 0, "minimum_events_per_foot": 0},
        }
    )

    result, contacts = metrics.gait_metrics(
        force,
        common,
        offsets,
        0.5,
        0.1,
        config,
        robot_mass_kg=None,
    )

    assert np.all(contacts)
    assert result["commanded_duty_factor"] == pytest.approx(0.5)
    assert result["sampled_desired_stance_fraction"] == pytest.approx(1.0)
    assert result["measured_duty_factor"] == pytest.approx(1.0)
    assert result["duty_factor_error"] == pytest.approx(0.5)
    assert result["duty_factor_abs_error_per_foot"] == pytest.approx([0.5] * 4)


def test_cyclic_order_merges_wraparound_simultaneous_events_and_canonicalizes_rotation():
    metrics = _load("leg_usage_metrics_cyclic_wrap", "_leg_usage_metrics.py")
    common = np.arange(0.0, 2.0, 0.01)
    touchdowns = [
        np.asarray([99, 199]),
        np.asarray([1, 101]),
        np.asarray([25, 125]),
        np.asarray([50, 150]),
    ]

    orders = metrics._cyclic_orders(touchdowns, common, 0.03)

    assert orders == ["fl+fr>rl>rr", "fl+fr>rl>rr"]


def test_cyclic_order_agreement_aligns_unequal_traces_by_common_cycle():
    metrics = _load("leg_usage_metrics_cyclic_alignment", "_leg_usage_metrics.py")
    step_dt = 0.01
    common = np.arange(0.0, 4.0, step_dt)
    offsets = np.asarray(_gaits()[0]["phases"])
    actual_contacts = np.zeros((len(common), 4), dtype=bool)
    for cycle in range(4):
        for leg, phase in enumerate((0.1, 0.3, 0.5, 0.7)):
            start = int((cycle + phase) / step_dt)
            actual_contacts[start : start + 10, leg] = True
    force = np.where(actual_contacts, 100.0, 0.0)
    config = metrics.evaluation_config(
        {
            "contact": {"mode": "absolute", "on_n": 20.0, "off_n": 10.0, "minimum_dwell_s": 0.0},
            "gait": {"minimum_complete_cycles": 0, "minimum_events_per_foot": 0},
        }
    )

    result, _ = metrics.gait_metrics(
        force,
        common,
        offsets,
        0.5,
        step_dt,
        config,
        robot_mass_kg=None,
    )

    assert result["cyclic_order_complete_cycles"] == 4
    assert result["cyclic_order_expected_complete_cycles"] == 3
    assert result["cyclic_order_comparable_cycles"] == 3
    assert result["cyclic_order_agreement"] == pytest.approx(0.0)


def test_velocity_metrics_preserve_signed_bias_heading_and_path():
    metrics = _load("leg_usage_metrics_velocity", "_leg_usage_metrics.py")
    count = 100
    commands = np.zeros((count, 3))
    commands[:, 0] = -1.0
    velocity = np.zeros_like(commands)
    velocity[:, 0] = -0.8
    velocity[:, 1] = 0.1
    velocity[:, 2] = 0.02
    positions = np.zeros((count, 2))
    positions[:, 0] = np.linspace(0.0, -1.6, count)
    positions[:, 1] = np.linspace(0.0, 0.2, count)
    headings = np.linspace(np.pi - 0.02, np.pi + 0.02, count)
    desired_headings = np.full(count, -np.pi + 0.01)

    result = metrics.velocity_metrics(
        commands,
        velocity,
        positions,
        0.02,
        base_headings_rad=headings,
        desired_headings_rad=desired_headings,
    )

    assert result["vx_rmse_mps"] == pytest.approx(0.2)
    assert result["vx_bias_mps"] == pytest.approx(0.2)
    assert result["vx_gain"] == pytest.approx(0.8)
    assert result["vx_relative_rmse"] == pytest.approx(0.2)
    assert result["vx_sign_error_fraction"] == 0.0
    assert result["vy_rmse_mps"] == pytest.approx(0.1)
    assert result["signed_directed_progress_m"] == pytest.approx(1.6)
    assert result["progress_per_commanded_distance"] == pytest.approx(0.8)
    assert result["heading_p95_rad"] < 0.06


def test_velocity_transient_fraction_uses_measurement_window_overlap():
    metrics = _load("leg_usage_metrics_velocity_transient", "_leg_usage_metrics.py")
    commands = np.zeros((6, 3))
    commands[:, 0] = 1.0
    velocity = np.zeros_like(commands)
    velocity[2:, 0] = 1.0
    positions = np.zeros((6, 2))

    result = metrics.velocity_metrics(
        commands[1:],
        velocity[1:],
        positions[1:],
        0.1,
        full_commands=commands,
        full_body_velocities=velocity,
        measurement_start_index=1,
    )

    assert result["settling_time_s"] == pytest.approx(0.2)
    assert result["measurement_window_start_s"] == pytest.approx(0.1)
    assert result["post_settle_in_band_fraction"] == pytest.approx(4.0 / 5.0)


def test_load_metrics_report_concentration_worst_joint_and_sentinel_rejection():
    metrics = _load("leg_usage_metrics_load", "_leg_usage_metrics.py")
    count = 20
    torque = np.tile(np.repeat([1.0, 2.0, 3.0, 4.0], 3), (count, 1))
    power = 2.0 * torque
    force = np.tile([40.0, 50.0, 60.0, 100.0], (count, 1))
    contact = np.ones((count, 4), dtype=bool)
    contact[:2] = False
    limits = np.full(12, 10.0)

    result = metrics.load_metrics(
        torque,
        power,
        limits,
        force,
        contact,
        0.02,
        complete_cycles=2,
        directed_progress_m=1.0,
    )

    assert result["normalized_effort_available"] is True
    assert result["torque_squared_worst_leg"] == "rr"
    assert result["absolute_work_worst_leg"] == "rr"
    assert result["vertical_grf_impulse_worst_leg"] == "rr"
    assert result["worst_normalized_torque_joint_index"] == 9
    assert result["worst_absolute_work_joint_index"] == 9
    assert result["torque_squared_max_share"] > 0.4
    assert result["absolute_work_front_hind_signed"] < 0.0
    assert result["absolute_work_worst_leg_value"] == pytest.approx(max(result["absolute_work_per_leg"]))
    assert result["rr_impact_peak_max_n"] == pytest.approx(100.0)
    assert result["rr_impact_impulse_max_ns"] > 0.0
    assert result["raw_torque_squared_per_cycle"] == pytest.approx(result["raw_torque_squared_total"] / 2.0)

    unavailable = metrics.load_metrics(
        torque,
        power,
        np.full(12, 1.0e9),
        force,
        contact,
        0.02,
        complete_cycles=2,
        directed_progress_m=1.0,
    )
    assert unavailable["normalized_effort_available"] is False
    assert "sentinel" in unavailable["normalized_effort_unavailable_reason"]
    assert unavailable["absolute_work_total_j"] > 0.0
    assert unavailable["worst_absolute_work_joint_index"] == 9
    assert unavailable["worst_absolute_work_j"] > 0.0

    raw_only = metrics.load_metrics(
        torque,
        power,
        None,
        None,
        None,
        0.02,
        complete_cycles=2,
        directed_progress_m=1.0,
    )
    assert raw_only["raw_actuator_load_available"] is True
    assert raw_only["grf_load_available"] is False
    assert raw_only["normalized_effort_available"] is False
    assert raw_only["absolute_work_total_j"] > 0.0
    assert "vertical_grf_impulse_total_ns" not in raw_only

    zero_leg = metrics._concentration(np.asarray([0.0, 1.0, 2.0, 3.0]), "synthetic")
    assert zero_leg["synthetic_max_to_min"] == pytest.approx(3.0e12)
    assert zero_leg["synthetic_max_to_min_reason"] == "epsilon_regularized_zero_leg_value"
    assert zero_leg["synthetic_zero_leg_count"] == 1
    assert zero_leg["synthetic_worst_leg_value"] == pytest.approx(3.0)


def test_light_protocol_uses_exact_stable_rows_and_separate_identity(tmp_path, monkeypatch):
    analysis = _load("evaluation_light_protocol", "evaluation.py")
    monkeypatch.setattr(analysis, "canonical_training_gaits", lambda: ("synthetic_v1", _gaits()))
    checkpoint = tmp_path / "run" / "model_1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    common = {
        "repo_root": tmp_path,
        "checkpoint": checkpoint,
        "robot": "go2",
        "task": "synthetic-play-task",
        "step_dt": 0.02,
    }

    full = analysis.build_study(**common)
    light = analysis.build_study(**common, protocol="light")

    assert full["method_version"] == "leg_usage_grid_full_v3"
    assert len(full["cells"]) == 60
    assert Path(full["output_root"]).name == "leg_usage_grid_full_v3"
    assert light["method_version"] == "leg_usage_grid_light_v2"
    assert Path(light["output_root"]).name == "leg_usage_grid_light"
    assert [(cell["gait_name"], cell["velocity_mps"]) for cell in light["cells"]] == [
        ("trot", 1.0),
        ("trot", -1.0),
        ("bound", 1.0),
        ("bound", -1.0),
        ("half_bound_front_a", 1.0),
        ("half_bound_front_b", -1.0),
        ("gallop_a", 1.0),
        ("gallop_c", -1.0),
    ]
    assert full["study_identity_sha256"] != light["study_identity_sha256"]
    with pytest.raises(ValueError, match="immutable light protocol"):
        analysis.build_study(**common, protocol="light", velocities_mps=(-1.0, 1.0))
    with pytest.raises(ValueError, match="immutable light protocol"):
        analysis.build_study(**common, protocol="light", gait_indices=(0, 1))


def test_cli_file_loader_can_load_analyzer_and_sibling_metrics():
    cli = _load("symm_cli_metrics_loader", "symm_cli.py")

    analysis = cli._load_evaluation_module()

    assert analysis.FULL_METHOD_VERSION == "leg_usage_grid_full_v3"
    assert analysis._metrics.evaluation_config()["contact"]["mode"] == "body_weight"


def test_directional_pair_metric_uses_mean_velocity_equation_with_recorded_epsilon():
    analysis = _load("evaluation_directional_pair", "evaluation.py")
    config = analysis._metrics.evaluation_config({"tracking": {"relative_error_epsilon_mps": 0.01}})
    study = {
        "evaluation_config": config,
        "gaits": [{"index": 0, "time_reversal_partner": 0}],
    }
    common = {"gait_index": 0, "family": "trot", "velocity_metric_valid": True}
    rows = [
        {
            **common,
            "cell_id": "positive",
            "velocity_mps": 1.0,
            "velocity_vx_bias_mps": -0.1,
            "velocity_vx_measured_mean_mps": 0.9,
        },
        {
            **common,
            "cell_id": "negative",
            "velocity_mps": -1.0,
            "velocity_vx_bias_mps": 0.2,
            "velocity_vx_measured_mean_mps": -0.8,
        },
    ]

    result = analysis._directional_pair_metrics(study, rows)

    assert len(result) == 1
    assert result[0]["e_pair_bias_mps"] == pytest.approx(abs(0.9 - 0.8))
    assert result[0]["e_pair_bias_normalized"] == pytest.approx(0.1 / 2.01)
    assert result[0]["e_pair_bias_epsilon_mps"] == pytest.approx(0.01)


def test_success_metrics_publish_independent_velocity_gait_and_joint_flags():
    metrics = _load("leg_usage_metrics_success_domains", "_leg_usage_metrics.py")
    velocity = {
        "signed_directed_progress_m": 1.0,
        "vx_relative_rmse": 0.05,
        "yaw_rmse_radps": 0.01,
    }
    gait = {
        "agreement_boundary_excluded": 0.5,
        "classified_family": "bound",
        "complete_cycles": 4,
        "touchdown_events_per_foot": [4] * 4,
        "liftoff_events_per_foot": [4] * 4,
    }

    result = metrics.success_metrics(velocity, gait, planned_family="trot", terminated=False)

    assert result["velocity_only_success"] is True
    assert result["gait_only_success"] is False
    assert result["joint_velocity_and_gait_success"] is False
    assert result["joint_success"] is False
    assert result["success"] is False
    assert "correct_family" in result["gait_only_success_failed_checks"]
