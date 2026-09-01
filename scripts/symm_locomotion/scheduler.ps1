# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$RemainingArgs
)

$ForwardedArgs = [System.Collections.Generic.List[string]]::new()
$ForwardedArgs.Add("--shell")
$ForwardedArgs.Add("powershell")
$CurrentPowerShellExecutable = (Get-Process -Id $PID).Path
if (-not [string]::IsNullOrWhiteSpace($CurrentPowerShellExecutable)) {
    $ForwardedArgs.Add("--shell_executable")
    $ForwardedArgs.Add($CurrentPowerShellExecutable)
}
foreach ($Argument in $RemainingArgs) {
    if ($Argument -is [scriptblock]) {
        $ForwardedArgs.Add($Argument.ToString())
    } else {
        $ForwardedArgs.Add([string]$Argument)
    }
}

$PythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
$PythonArgs = @()
if ($null -eq $PythonCommand) {
    $PythonCommand = Get-Command "py" -ErrorAction SilentlyContinue
    $PythonArgs = @("-3")
}
if ($null -eq $PythonCommand) {
    Write-Error "Could not find 'python' or the Windows 'py' launcher on PATH."
    exit 2
}

$SchedulerPath = Join-Path $PSScriptRoot "scheduler.py"
& $PythonCommand.Source @PythonArgs $SchedulerPath @ForwardedArgs
exit $LASTEXITCODE
