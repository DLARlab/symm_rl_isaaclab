# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

$DefaultCondaEnv = "go2_symm_rl_lab"
$CondaPrefix = $env:CONDA_PREFIX
if (-not $CondaPrefix) {
    $CondaPrefix = Join-Path $HOME ".conda\envs\$DefaultCondaEnv"
}

$PythonExe = Join-Path $CondaPrefix "python.exe"
if (Test-Path $PythonExe) {
    $env:CONDA_PREFIX = $CondaPrefix
    if (-not $env:CONDA_DEFAULT_ENV) {
        $env:CONDA_DEFAULT_ENV = Split-Path $CondaPrefix -Leaf
    }
    & $PythonExe "$PSScriptRoot\record.py" @args
} else {
    python "$PSScriptRoot\record.py" @args
}
exit $LASTEXITCODE
