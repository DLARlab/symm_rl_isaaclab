# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TRS grid durability-analysis helpers."""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_analysis_module():
    module_path = Path(__file__).resolve().parents[1] / "analyze_trs_grid.py"
    spec = importlib.util.spec_from_file_location("analyze_trs_grid_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load_analysis_module()


def _range_counts(values: list[float]) -> list[tuple[float, float]]:
    return sorted((cycle_range, count) for cycle_range, _mean, count in analysis.rainflow_cycles(values))


class TestDurabilityAnalysis(unittest.TestCase):
    """Validate cycle counting and pair-allocation helpers."""

    def assert_pairs_almost_equal(
        self,
        actual: list[tuple[float, float]],
        expected: list[tuple[float, float]],
    ) -> None:
        """Compare ordered ``(range, count)`` pairs."""
        self.assertEqual(len(actual), len(expected))
        for actual_pair, expected_pair in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_pair[0], expected_pair[0])
            self.assertAlmostEqual(actual_pair[1], expected_pair[1])

    def test_rainflow_matches_published_astm_e1049_example(self) -> None:
        """Match the public reference results used by the ``rainflow`` package."""
        time = [4.0 * index / 200 for index in range(201)]
        signal = [
            0.2 + 0.5 * math.sin(value) + 0.2 * math.cos(10.0 * value) + 0.2 * math.sin(4.0 * value) for value in time
        ]
        expected = [
            (0.04258965150708488, 0.5),
            (0.10973439445727551, 1.0),
            (0.11294628078612906, 0.5),
            (0.2057106991158965, 1.0),
            (0.21467990941625242, 1.0),
            (0.4388985979776988, 1.0),
            (0.48305748051348263, 0.5),
            (0.5286423866535466, 0.5),
            (0.7809330293159786, 0.5),
            (1.4343610172143002, 0.5),
        ]
        self.assert_pairs_almost_equal(_range_counts(signal), expected)

    def test_rainflow_boundaries_plateaus_and_nested_cycles(self) -> None:
        cases = [
            ([2.0, 2.0, 2.0], [], 0.0, 0.0),
            ([0.0, 2.0], [(2.0, 0.5)], 0.5, 0.5),
            ([0.0, 2.0, 0.0], [(2.0, 0.5), (2.0, 0.5)], 1.0, 1.0),
            (
                [0.0, 4.0, 1.0, 3.0, 0.0],
                [(2.0, 1.0), (4.0, 0.5), (4.0, 0.5)],
                9.0,
                33.0,
            ),
            ([0.0, 4.0, 1.0], [(3.0, 0.5), (4.0, 0.5)], 5.6875, 19.796875),
            ([0.0, 0.0, 2.0, 2.0, 0.0, 0.0], [(2.0, 0.5), (2.0, 0.5)], 1.0, 1.0),
        ]
        for signal, expected_cycles, expected_m3, expected_m5 in cases:
            with self.subTest(signal=signal):
                cycles = analysis.rainflow_cycles(signal)
                self.assert_pairs_almost_equal(_range_counts(signal), expected_cycles)
                self.assertAlmostEqual(analysis.fatigue_proxy(cycles, 1.0, 3), expected_m3)
                self.assertAlmostEqual(analysis.fatigue_proxy(cycles, 1.0, 5), expected_m5)

    def test_fatigue_proxy_is_offset_and_sign_invariant(self) -> None:
        signal = [0.0, 4.0, 1.0, 3.0, 0.0]
        reference = analysis.fatigue_proxy(analysis.rainflow_cycles(signal), 5.0, 5)
        shifted = analysis.fatigue_proxy(
            analysis.rainflow_cycles([value + 10.0 for value in signal]),
            5.0,
            5,
        )
        inverted = analysis.fatigue_proxy(analysis.rainflow_cycles([-value for value in signal]), 5.0, 5)
        self.assertAlmostEqual(shifted, reference)
        self.assertAlmostEqual(inverted, reference)

    def test_pair_concentration_separates_balance_from_total_exposure(self) -> None:
        pair, ratio = analysis.pair_concentration(8.0, 4.0)
        self.assertEqual(pair, "front")
        self.assertAlmostEqual(ratio, 2.0)
        low = analysis.summarize_pair_totals(1.0, 1.0, distance_m=2.0)
        high = analysis.summarize_pair_totals(10.0, 10.0, distance_m=2.0)
        self.assertAlmostEqual(low["abs_imbalance_percent"], 0.0)
        self.assertAlmostEqual(high["abs_imbalance_percent"], 0.0)
        self.assertAlmostEqual(high["total_per_m"], 10.0 * low["total_per_m"])


class TestRolloutLeftRightAllocation(unittest.TestCase):
    """Check physical leg grouping and the sign of side allocation."""

    def _analyze_rollout(self, leg_order: tuple[int, ...] = (0, 1, 2, 3)) -> dict:
        amplitudes = (1.0, 2.0, 3.0, 4.0)
        foot_forces = ((10.0, 20.0, 30.0, 0.0), (10.0, 20.0, 30.0, 0.0), (10.0, 0.0, 30.0, 0.0), (10.0, 0.0, 0.0, 40.0))
        torques = [amplitudes[leg] * step for step in range(1, 5) for leg in leg_order for _ in range(3)]
        velocities = [1.0, -1.0, 1.0] * 16
        forces = [force_row[leg] for force_row in foot_forces for leg in leg_order]
        arrays = {
            "time_steps": ([0.0, 1.0, 2.0, 3.0], (4,)),
            "desired_lin_vel": ([1.0, 0.0, 0.0] * 4, (4, 3)),
            "true_lin_vel": ([1.0, 0.0, 0.0] * 4, (4, 3)),
            "base_positions": ([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0], (4, 2)),
            "joint_names": (
                [f"joint_{leg}_{motor}" for leg in analysis.LEG_NAMES for motor in ("abad", "thigh", "calf")],
                (12,),
            ),
            "joint_torques": (torques, (4, 12)),
            "joint_velocities": (velocities, (4, 12)),
            "joint_powers": (
                [torque * velocity for torque, velocity in zip(torques, velocities, strict=True)],
                (4, 12),
            ),
            "foot_ground_reaction_forces_w": (
                [component for force in forces for component in (0.0, 0.0, force)],
                (4, 4, 3),
            ),
            "foot_forces": (forces, (4, 4)),
            "episode_done": ([False] * 4, (4,)),
            "ground_reaction_force_includes_friction": ([True] * 4, (4,)),
        }
        run = analysis.RunSpec("x1", analysis.ROOT, analysis.ROOT / "synthetic.npz", 0.0, 0.0, 500, False)
        with (
            mock.patch.object(analysis.zipfile, "ZipFile") as archive,
            mock.patch.object(analysis, "read_npy_member", side_effect=lambda archive, name: arrays[name]),
            mock.patch.multiple(analysis, WINDOW_START_S=0.0, WINDOW_STOP_S=4.0, BOOTSTRAP_REPLICATES=40),
        ):
            archive.return_value.__enter__.return_value.namelist.return_value = [f"{name}.npy" for name in arrays]
            return analysis.analyze_rollout(run)

    def test_left_right_totals_use_alternating_leg_indices(self) -> None:
        result = self._analyze_rollout()
        side_metrics = result["metrics_left_right"]
        self.assertEqual(set(side_metrics), set(result["metrics"]))
        torque = side_metrics["torque_squared"]
        self.assertAlmostEqual(torque["left_integral"], 900.0)
        self.assertAlmostEqual(torque["right_integral"], 1800.0)
        self.assertAlmostEqual(torque["signed_imbalance_percent"], -100.0 / 3.0)
        self.assertAlmostEqual(side_metrics["absolute_work"]["signed_imbalance_percent"], -20.0)
        self.assertAlmostEqual(side_metrics["vertical_grf_impulse"]["left_integral"], 130.0)
        self.assertAlmostEqual(side_metrics["vertical_grf_impulse"]["right_integral"], 80.0)
        self.assertAlmostEqual(side_metrics["contact_time"]["left_duty_factor"], 7.0 / 8.0)
        self.assertAlmostEqual(side_metrics["contact_time"]["right_duty_factor"], 3.0 / 8.0)
        self.assertAlmostEqual(result["metrics"]["contact_time"]["front_duty_factor"], 6.0 / 8.0)
        self.assertAlmostEqual(result["metrics"]["contact_time"]["hind_duty_factor"], 4.0 / 8.0)
        for name, sides in side_metrics.items():
            front_hind = result["metrics"][name]
            self.assertAlmostEqual(
                sides["left_integral"] + sides["right_integral"],
                front_hind["front_integral"] + front_hind["hind_integral"],
            )

    def test_mirroring_sides_flips_sign_and_preserves_front_hind_results(self) -> None:
        original = self._analyze_rollout()
        mirrored = self._analyze_rollout((1, 0, 3, 2))
        self.assertEqual(original["metrics"], mirrored["metrics"])
        for name, left_right in original["metrics_left_right"].items():
            swapped = mirrored["metrics_left_right"][name]
            self.assertAlmostEqual(left_right["signed_imbalance_percent"], -swapped["signed_imbalance_percent"])
            self.assertAlmostEqual(left_right["abs_imbalance_percent"], swapped["abs_imbalance_percent"])

    def test_additive_summary_preserves_defaults_and_supports_side_labels(self) -> None:
        with mock.patch.object(analysis, "BOOTSTRAP_REPLICATES", 40):
            default = analysis.summarize_additive_metric(
                [(3.0, 1.0)] * 4, dt_s=0.5, block_samples=2, rng=random.Random(1)
            )
            sides = analysis.summarize_additive_metric(
                [(3.0, 1.0)] * 4, dt_s=0.5, block_samples=2, rng=random.Random(1), pair_names=("left", "right")
            )
        self.assertEqual(sides["left_integral"], 6.0)
        self.assertEqual(sides["right_integral"], 2.0)
        self.assertEqual(sides["left_share_percent"], 75.0)
        self.assertEqual(sides["signed_imbalance_percent"], 50.0)
        self.assertEqual(sides["bootstrap_95_percent"], [50.0, 50.0])
        self.assertEqual(
            default, {key.replace("left", "front").replace("right", "hind"): value for key, value in sides.items()}
        )


if __name__ == "__main__":
    unittest.main()
