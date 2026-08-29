# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCHEDULER = Path(__file__).resolve().parents[1] / "scheduler.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is unavailable")
def test_child_exit_does_not_terminate_continue_on_error_schedule():
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(SCHEDULER),
            "--continue_on_error",
            "--delay",
            "0s",
            "--command",
            "exit 7",
            "--delay",
            "0s",
            "--command",
            "Write-Output SCHEDULER_SECOND_STEP_RAN",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 7
    assert "SCHEDULER_SECOND_STEP_RAN" in result.stdout
    assert "Step 1/2 failed with exit code 7" in result.stdout
    assert "Completed step 2/2" in result.stdout
