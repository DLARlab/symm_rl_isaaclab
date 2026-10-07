# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Write-Warning "analyze_leg_usage.ps1 is deprecated; use evaluation.ps1 instead."
& "$PSScriptRoot\evaluation.ps1" --protocol legacy @args
exit $LASTEXITCODE
