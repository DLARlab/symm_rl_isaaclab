# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the curated TensorBoard reward plotter."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[1] / "plot_good_runs_tensorboard.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("plot_good_runs_tensorboard_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plotter = _load_plot_module()


class TestCuratedTensorBoardPlotter(unittest.TestCase):
    """Validate curated-run discovery and resolved-parameter parsing."""

    def test_generation_inventory_rejects_unclassified_runs(self) -> None:
        """Require every plotted run to have a documented generation."""
        self.assertEqual(plotter._generation("2026-07-07_00-11-14_no_trs"), "60D")
        self.assertEqual(
            plotter._generation("2026-07-20_16-24-19_x1_trs_m0p20_v0p10_w500"),
            "72D",
        )
        with self.assertRaisesRegex(ValueError, "not classified"):
            plotter._generation("new_unreviewed_run")

    def test_yaml_scalar_parser_preserves_resolved_types(self) -> None:
        """Parse bool, numeric, and string scalars without a YAML dependency."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "agent.yaml"
            path.write_text(
                "seed: 42\nlearning_rate: 0.001\nuse_mirror_loss: true\nschedule: adaptive\n",
                encoding="utf-8",
            )
            self.assertEqual(plotter._parse_yaml_scalar(path, "seed"), 42)
            self.assertEqual(plotter._parse_yaml_scalar(path, "learning_rate"), 0.001)
            self.assertIs(plotter._parse_yaml_scalar(path, "use_mirror_loss"), True)
            self.assertEqual(plotter._parse_yaml_scalar(path, "schedule"), "adaptive")

    def test_discovery_is_limited_to_five_curated_runs_per_robot(self) -> None:
        """Find only the ten traces in the fixed legacy comparison inventory."""
        runs = plotter.discover_curated_runs()
        self.assertEqual(len(runs), 10)
        for robot in plotter.ROBOT_RUN_DIRS:
            self.assertEqual(sum(run.robot == robot for run in runs), 5)
        for run in runs:
            self.assertTrue(run.event_path.is_relative_to(plotter.GOOD_RUNS_ROOT))
            self.assertTrue(run.agent_path.is_relative_to(plotter.GOOD_RUNS_ROOT))
            self.assertEqual(run.seed, 42)


if __name__ == "__main__":
    unittest.main()
