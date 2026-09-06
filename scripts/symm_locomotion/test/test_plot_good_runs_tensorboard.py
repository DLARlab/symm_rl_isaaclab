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
        """Require every plotted run to have a documented cohort."""
        self.assertEqual(
            plotter._generation("2026-09-03_11-19-54_m5_x1_actor_only_trs_m0p2_v0_w500_r0_x1def_s42"),
            "actor-trs-v5",
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

    def test_yaml_scalar_parser_supports_exact_mapping_scopes(self) -> None:
        """Disambiguate repeated resolved-config keys using their mapping path."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "agent.yaml"
            path.write_text(
                "algorithm:\n"
                "  learning_rate: 0.001\n"
                "  symmetry_cfg:\n"
                "    rampup_iterations: 500\n"
                "    nested:\n"
                "      rampup_iterations: null\n",
                encoding="utf-8",
            )
            self.assertEqual(
                plotter._parse_yaml_scalar(path, "learning_rate", section=("algorithm",)),
                0.001,
            )
            self.assertEqual(
                plotter._parse_yaml_scalar(path, "rampup_iterations", section=("algorithm", "symmetry_cfg")),
                500,
            )

    def test_discovery_is_limited_to_actor_trs_v5_runs(self) -> None:
        """Find only the ten traces in the retained Actor TRS V5 inventory."""
        runs = plotter.discover_curated_runs()
        self.assertEqual(len(runs), 10)
        self.assertEqual(sum(run.robot == "go2" for run in runs), 6)
        self.assertEqual(sum(run.robot == "x1" for run in runs), 4)
        for run in runs:
            self.assertTrue(run.event_path.is_relative_to(plotter.GOOD_RUNS_ROOT))
            self.assertTrue(run.agent_path.is_relative_to(plotter.GOOD_RUNS_ROOT))
            self.assertNotIn("legacy", run.run_path.parts)
            self.assertEqual(run.generation, "actor-trs-v5")
            self.assertEqual(run.value_coeff, 0.0)
            self.assertEqual(run.seed, 43 if run.robot == "go2" else 42)

        actor_runs = [run for run in runs if run.use_mirror_loss]
        self.assertEqual(len(actor_runs), 8)
        self.assertTrue(all(run.warmup_iterations == 500 for run in actor_runs))
        self.assertEqual(sum(run.rampup_iterations == 500 for run in actor_runs), 2)
        self.assertTrue(all(run.robot == "go2" for run in actor_runs if run.rampup_iterations == 500))


if __name__ == "__main__":
    unittest.main()
