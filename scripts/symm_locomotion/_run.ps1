# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

if ($args.Count -lt 1) {
    Write-Error "Usage: _run.ps1 SCRIPT_NAME [ARGUMENT ...]"
    exit 2
}

$ScriptName = [string]$args[0]
$RemainingArgs = @($args | Select-Object -Skip 1)

$DefaultCondaEnv = "symm_rl_isaaclab"
$CondaPrefix = $null
$ActiveCondaPrefix = $env:CONDA_PREFIX
if ($ActiveCondaPrefix) {
    $ActiveCondaPython = Join-Path $ActiveCondaPrefix "python.exe"
    if (Test-Path -LiteralPath $ActiveCondaPython) {
        $CondaPrefix = $ActiveCondaPrefix
    }
}

if (-not $CondaPrefix) {
    $DefaultCondaPrefix = Join-Path $HOME ".conda\envs\$DefaultCondaEnv"
    $DefaultCondaPython = Join-Path $DefaultCondaPrefix "python.exe"
    if (Test-Path -LiteralPath $DefaultCondaPython) {
        $CondaPrefix = $DefaultCondaPrefix
    }
}

if (-not $CondaPrefix) {
    Write-Error (
        "Could not find the '$DefaultCondaEnv' Conda environment. " +
        "Activate it or install it at '$DefaultCondaPrefix'."
    )
    exit 2
}

$PythonExe = Join-Path $CondaPrefix "python.exe"
$ScriptPath = Join-Path $PSScriptRoot $ScriptName
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LocalSourcePaths = Get-ChildItem -LiteralPath (Join-Path $RepoRoot "source") -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName (Join-Path $_.Name "__init__.py")) } |
    Sort-Object -Property FullName |
    ForEach-Object { $_.FullName }
$LocalPythonPath = $LocalSourcePaths -join [System.IO.Path]::PathSeparator
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = $LocalPythonPath + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
} else {
    $env:PYTHONPATH = $LocalPythonPath
}

$env:CONDA_PREFIX = $CondaPrefix
$env:CONDA_DEFAULT_ENV = Split-Path $CondaPrefix -Leaf
& $PythonExe $ScriptPath @RemainingArgs
exit $LASTEXITCODE
