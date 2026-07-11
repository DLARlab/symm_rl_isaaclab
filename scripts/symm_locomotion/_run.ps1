# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptName,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$DefaultCondaEnv = "symm_rl_isaaclab"
$CondaPrefix = $env:CONDA_PREFIX
if (-not $CondaPrefix) {
    $CondaPrefix = Join-Path $HOME ".conda\envs\$DefaultCondaEnv"
}

$PythonExe = Join-Path $CondaPrefix "python.exe"
$ScriptPath = Join-Path $PSScriptRoot $ScriptName

if (Test-Path $PythonExe) {
    $env:CONDA_PREFIX = $CondaPrefix
    if (-not $env:CONDA_DEFAULT_ENV) {
        $env:CONDA_DEFAULT_ENV = Split-Path $CondaPrefix -Leaf
    }
    & $PythonExe $ScriptPath @RemainingArgs
} else {
    python $ScriptPath @RemainingArgs
}
exit $LASTEXITCODE
