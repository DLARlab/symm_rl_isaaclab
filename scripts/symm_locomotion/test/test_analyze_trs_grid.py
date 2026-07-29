# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TRS grid durability-analysis helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
