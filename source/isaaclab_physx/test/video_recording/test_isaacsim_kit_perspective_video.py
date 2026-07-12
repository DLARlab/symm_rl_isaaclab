# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac Sim Kit perspective video recorder."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import IsaacsimKitPerspectiveVideo


def test_update_camera_updates_config_and_viewport():
    """Camera updates are applied to both stored config and the Kit viewport."""
    cfg = SimpleNamespace(
        camera_prim_path="/OmniverseKit_Persp",
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
    )
    recorder = IsaacsimKitPerspectiveVideo(cfg)
    viewport_manager = MagicMock()

    rendering_manager = SimpleNamespace(ViewportManager=viewport_manager)
    with patch.dict(sys.modules, {"isaacsim.core.rendering_manager": rendering_manager}):
        recorder.update_camera((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    assert cfg.eye == (1.0, 2.0, 3.0)
    assert cfg.lookat == (4.0, 5.0, 6.0)
    viewport_manager.set_camera_view.assert_called_once_with(
        "/OmniverseKit_Persp",
        eye=[1.0, 2.0, 3.0],
        target=[4.0, 5.0, 6.0],
    )
