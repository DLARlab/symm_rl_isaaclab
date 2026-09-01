# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

Write-Warning "compare.ps1 is deprecated; use symm_locomotion.ps1 compare instead."
& "$PSScriptRoot\_run.ps1" "symm_cli.py" "compare" @RemainingArgs
exit $LASTEXITCODE
