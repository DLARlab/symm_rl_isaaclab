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
        self.assertEqual(plotter._generation("2026-07-31_22-48-10_go2_no_trs_20k_512"), "phase-v2")
        self.assertEqual(
            plotter._generation("2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2"),
            "gait-v2",
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

    def test_discovery_is_limited_to_six_matched_runs_per_robot(self) -> None:
        """Find only the twelve traces in a matched retained inventory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_root = Path(temporary_directory) / "good_runs_72d"
            run_dirs = plotter._robot_run_dirs(archive_root)
            for run_name in sorted(plotter.PHASE_V2_RUN_NAMES | plotter.GAIT_V2_RUN_NAMES):
                robot = "go2" if "_go2_" in run_name else "x1"
                generation = plotter._generation(run_name)
                run_root = run_dirs[robot][0 if generation == "phase-v2" else 1]
                run_path = run_root / run_name
                params_path = run_path / "params"
                params_path.mkdir(parents=True)
                (run_path / "events.out.tfevents.synthetic").touch()
                if "no_trs" in run_name:
                    use_mirror_loss, mirror_coeff, value_coeff = False, 0.0, 0.0
                elif "m0p1" in run_name:
                    use_mirror_loss, mirror_coeff, value_coeff = True, 0.1, 0.05
                else:
                    use_mirror_loss, mirror_coeff, value_coeff = True, 0.2, 0.1
                (params_path / "agent.yaml").write_text(
                    "seed: 42\n"
                    "max_iterations: 20000\n"
                    "learning_rate: 0.001\n"
                    "schedule: adaptive\n"
                    f"use_mirror_loss: {str(use_mirror_loss).lower()}\n"
                    f"mirror_loss_coeff: {mirror_coeff}\n"
                    f"value_loss_coeff: {value_coeff}\n"
                    "warmup_iterations: 500\n"
                    "min_abs_command_velocity: 0.0\n",
                    encoding="utf-8",
                )

            runs = plotter.discover_curated_runs(archive_root)

            self.assertEqual(len(runs), 12)
            for robot in run_dirs:
                self.assertEqual(sum(run.robot == robot for run in runs), 6)
            for run in runs:
                self.assertTrue(run.event_path.is_relative_to(archive_root))
                self.assertTrue(run.agent_path.is_relative_to(archive_root))
                self.assertEqual(run.seed, 42)


if __name__ == "__main__":
    unittest.main()
