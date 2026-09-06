# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the manifest-driven matched TRS study utility."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


def _load_analysis_module():
    module_path = Path(__file__).resolve().parents[1] / "analyze_matched_trs_study.py"
    spec = importlib.util.spec_from_file_location("analyze_matched_trs_study_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study_analysis = _load_analysis_module()


def _manifest_payload(robot: str) -> dict:
    """Return a minimal valid matched-study manifest for ``robot``."""
    display_name = "Unitree Go2" if robot == "go2" else "Dobot X1"
    return {
        "schema_version": 1,
        "analysis_method_version": study_analysis.ANALYSIS_METHOD_VERSION,
        "robot": robot,
        "display_name": display_name,
        "short_name": robot.upper(),
        "run_root": f"synthetic/{robot}",
        "training": {
            "seed": 42,
            "num_envs": 512,
            "num_steps_per_env": 24,
            "max_iterations": 20000,
        },
        "windows_s": {"backward": [0.5, 9.5], "forward": [10.5, 29.5]},
        "balance_y_range": [-90.0, 90.0],
        "balance_y_ticks": [-80.0, -40.0, 0.0, 40.0, 80.0],
        "runs": [
            {
                "label": "No TRS",
                "slug": "no_trs",
                "folder": "no_trs",
                "mirror_coeff": 0.0,
                "value_coeff": 0.0,
                "warmup_iterations": None,
                "trs_enabled": False,
            },
            {
                "label": "TRS 0.10/0.05",
                "slug": "trs_0p10_0p05",
                "folder": "trs_0p10_0p05",
                "mirror_coeff": 0.1,
                "value_coeff": 0.05,
                "warmup_iterations": 500,
                "trs_enabled": True,
            },
            {
                "label": "TRS 0.20/0.10",
                "slug": "trs_0p20_0p10",
                "folder": "trs_0p20_0p10",
                "mirror_coeff": 0.2,
                "value_coeff": 0.1,
                "warmup_iterations": 500,
                "trs_enabled": True,
            },
            {
                "label": "TRS 0.30/0.15",
                "slug": "trs_0p30_0p15",
                "folder": "trs_0p30_0p15",
                "mirror_coeff": 0.3,
                "value_coeff": 0.15,
                "warmup_iterations": 500,
                "trs_enabled": True,
            },
        ],
    }


class TestMatchedTrsStudyManifest(unittest.TestCase):
    """Validate the stable matched-study manifest schema for both robots."""

    def setUp(self) -> None:
        """Create manifests without depending on unpublished legacy artifacts."""
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        self.manifests = {}
        for robot in ("go2", "x1"):
            path = root / f"{robot}_study.json"
            path.write_text(json.dumps(_manifest_payload(robot)), encoding="utf-8")
            self.manifests[robot] = path

    def test_milestone_manifests_define_matched_four_run_studies(self) -> None:
        """Load both robot schemas with unique runs and canonical coefficients."""
        for robot, path in self.manifests.items():
            with self.subTest(robot=robot):
                study = study_analysis.load_study_manifest(path)
                self.assertEqual(study["robot"], robot)
                self.assertEqual(len(study["runs"]), 4)
                self.assertEqual(
                    study["training"],
                    {"seed": 42, "num_envs": 512, "num_steps_per_env": 24, "max_iterations": 20000},
                )
                self.assertEqual(
                    [(run.mirror_coeff, run.value_coeff) for run in study["runs"]],
                    [(0.0, 0.0), (0.1, 0.05), (0.2, 0.1), (0.3, 0.15)],
                )
                self.assertEqual(
                    [run.warmup_iterations for run in study["runs"]],
                    [None, 500, 500, 500],
                )

        self.assertEqual(
            set(study_analysis.MATCHED_NPZ_MEMBERS),
            {
                "time_steps.npy",
                "desired_lin_vel.npy",
                "desired_positions.npy",
                "episode_done.npy",
                "foot_clearance_targets.npy",
                "foot_clearance_swing_weights.npy",
                "ground_reaction_force_includes_friction.npy",
                "joint_position_lower_limits.npy",
                "joint_position_upper_limits.npy",
                "joint_names.npy",
                "leg_names.npy",
                "motor_role_names.npy",
                "E_C_frc.npy",
                "E_C_spd.npy",
            },
        )

    def test_duplicate_run_slug_is_rejected(self) -> None:
        """Reject ambiguous manifests before reading expensive run artifacts."""
        payload = json.loads(self.manifests["go2"].read_text(encoding="utf-8"))
        payload["runs"][1]["slug"] = payload["runs"][0]["slug"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "slug.*unique"):
                study_analysis.load_study_manifest(path)

    def test_method_version_mismatch_is_rejected(self) -> None:
        """Prevent silent reinterpretation by a future analysis implementation."""
        payload = json.loads(self.manifests["x1"].read_text(encoding="utf-8"))
        payload["analysis_method_version"] = "unknown"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "analysis_method_version"):
                study_analysis.load_study_manifest(path)

    def test_inconsistent_no_trs_baseline_is_rejected(self) -> None:
        """Require one baseline-first zero-coefficient no-TRS condition."""
        payload = json.loads(self.manifests["go2"].read_text(encoding="utf-8"))
        payload["runs"][0]["mirror_coeff"] = 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no-TRS baseline"):
                study_analysis.load_study_manifest(path)

    def test_manifest_coefficient_must_match_resolved_run_configuration(self) -> None:
        """Reject a mislabeled coefficient before generating authoritative outputs."""
        run = study_analysis.load_study_manifest(self.manifests["x1"])["runs"][1]
        symmetry_cfg = {
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.1,
            "use_time_reversal_regularization": True,
            "value_loss_coeff": 0.05,
            "min_abs_command_velocity": 0.0,
            "warmup_iterations": 500,
        }
        command_cfg = {
            "phase_mapping_version": study_analysis.PHASE_MAPPING_VERSION,
            "min_xy_command_norm": 0.0,
        }
        training_cfg = {"seed": 42, "num_envs": 512, "max_iterations": 20000, "num_steps_per_env": 24}

        study_analysis._validate_run_configuration_values(run, symmetry_cfg, command_cfg, training_cfg)
        with self.assertRaisesRegex(ValueError, "mirror_loss_coeff"):
            study_analysis._validate_run_configuration_values(
                replace(run, mirror_coeff=0.123),
                symmetry_cfg,
                command_cfg,
                training_cfg,
            )

    def test_initial_checkpoint_validation_supports_latest_only_archives(self) -> None:
        """Validate complete sets, skip absent sets, and reject partial sets."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_paths = {slug: root / slug / "model_0.pt" for slug in ("a", "b", "c", "d")}

            equal, hashes, validation = study_analysis._validate_initial_checkpoints(checkpoint_paths)
            self.assertIsNone(equal)
            self.assertEqual(hashes, {})
            self.assertEqual(validation, "unavailable_latest_only_archive")

            checkpoint_paths["a"].parent.mkdir()
            checkpoint_paths["a"].write_bytes(b"same initialization")
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                study_analysis._validate_initial_checkpoints(checkpoint_paths)

            for path in checkpoint_paths.values():
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"same initialization")
            equal, hashes, validation = study_analysis._validate_initial_checkpoints(checkpoint_paths)
            self.assertTrue(equal)
            self.assertEqual(len(set(hashes.values())), 1)
            self.assertEqual(validation, "validated")

            checkpoint_paths["d"].write_bytes(b"different initialization")
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                study_analysis._validate_initial_checkpoints(checkpoint_paths)


if __name__ == "__main__":
    unittest.main()
