# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

function Show-Usage {
    @'
Schedule one or more PowerShell commands with a delay before each command.

Usage:
  .\scripts\symm_locomotion\scheduler.ps1 [OPTIONS] `
    --delay DURATION --command { COMMAND } `
    [--delay DURATION --command { COMMAND } ...]

Options:
  --delay DURATION            Delay before the next command.
  --command { COMMAND }       PowerShell command to run after the delay.
                              A quoted command string is also accepted.
  --update_interval DURATION  Countdown update interval (default: 5m).
  --working_directory PATH    Directory in which all commands run.
  --continue_on_error         Continue after a command returns an error.
  --dry_run                   Validate and display the schedule without waiting.
  --help                      Show this help text.

Durations accept short or long unit suffixes (for example, 500ms, 10min, or
5hours), as well as TimeSpan text such as 01:30:00.

Each delay begins only after the preceding command has finished. Commands run
synchronously, in the order given. By default, a failed command stops the
remaining schedule and the scheduler returns that command's exit code.

Example:
  .\scripts\symm_locomotion\scheduler.ps1 `
    --update_interval 5m `
    --delay 5h --command { .\scripts\symm_locomotion\train.ps1 --robot go2 } `
    --delay 10m --command { .\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest } `
    --delay 10m --command { .\scripts\symm_locomotion\evaluation.ps1 --robot go2 --checkpoint latest }
'@ | Write-Host
}

function ConvertTo-Duration {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Value,

        [Parameter(Mandatory = $true)]
        [string]$OptionName
    )

    if ($Value -is [TimeSpan]) {
        $Duration = $Value
    } else {
        $Text = ([string]$Value).Trim()
        $DurationMatch = [regex]::Match(
            $Text,
            (
                "^(?<amount>[0-9]+(?:\.[0-9]+)?)\s*" +
                "(?<unit>ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?)$"
            ),
            [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
        )

        if ($DurationMatch.Success) {
            $Amount = [double]::Parse(
                $DurationMatch.Groups["amount"].Value,
                [System.Globalization.CultureInfo]::InvariantCulture
            )
            $Unit = $DurationMatch.Groups["unit"].Value.ToLowerInvariant()
            $Duration = switch -Regex ($Unit) {
                "^(ms|millisecond|milliseconds)$" { [TimeSpan]::FromMilliseconds($Amount) }
                "^(s|sec|secs|second|seconds)$" { [TimeSpan]::FromSeconds($Amount) }
                "^(m|min|mins|minute|minutes)$" { [TimeSpan]::FromMinutes($Amount) }
                "^(h|hr|hrs|hour|hours)$" { [TimeSpan]::FromHours($Amount) }
                "^(d|day|days)$" { [TimeSpan]::FromDays($Amount) }
            }
        } else {
            $ParsedDuration = [TimeSpan]::Zero
            if (-not [TimeSpan]::TryParse($Text, [ref]$ParsedDuration)) {
                throw "$OptionName expects a duration such as '10min', '5hours', or '01:30:00'; received '$Text'."
            }
            $Duration = $ParsedDuration
        }
    }

    if ($Duration.Ticks -lt 0) {
        throw "$OptionName must not be negative."
    }
    return $Duration
}

function Format-Duration {
    param(
        [Parameter(Mandatory = $true)]
        [TimeSpan]$Duration
    )

    if ($Duration.TotalMilliseconds -lt 1000) {
        return ("{0:0} ms" -f $Duration.TotalMilliseconds)
    }
    if ($Duration.Days -gt 0) {
        return ("{0}d {1:00}:{2:00}:{3:00}" -f $Duration.Days, $Duration.Hours, $Duration.Minutes, $Duration.Seconds)
    }
    return ("{0:00}:{1:00}:{2:00}" -f ([Math]::Floor($Duration.TotalHours)), $Duration.Minutes, $Duration.Seconds)
}

function Format-Command {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    return (($Command.ToString().Trim() -replace "\s+", " "))
}

function Invoke-IsolatedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,

        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory
    )

    # Project wrappers intentionally use ``exit $LASTEXITCODE`` so that direct
    # shell invocations propagate failures. Run every scheduled step in a child
    # PowerShell process to keep that exit from terminating the scheduler.
    $PowerShellExecutable = (Get-Process -Id $PID).Path
    if ([string]::IsNullOrWhiteSpace($PowerShellExecutable)) {
        throw "Unable to resolve the current PowerShell executable."
    }

    $EscapedWorkingDirectory = $WorkingDirectory.Replace("'", "''")
    $CommandSource = $Command.ToString()
    $ChildSource = @"
`$ErrorActionPreference = "Stop"
Set-Location -LiteralPath '$EscapedWorkingDirectory'
try {
    & {
$CommandSource
    }
    `$CommandSucceeded = `$?
    `$CommandExitCode = if (`$null -eq `$LASTEXITCODE) { 0 } else { [int]`$LASTEXITCODE }
    if (-not `$CommandSucceeded -and `$CommandExitCode -eq 0) {
        `$CommandExitCode = 1
    }
    exit `$CommandExitCode
} catch {
    [Console]::Error.WriteLine(`$_.Exception.Message)
    exit 1
}
"@
    $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($ChildSource))
    & $PowerShellExecutable -NoLogo -NoProfile -NonInteractive -EncodedCommand $EncodedCommand | Out-Host
    $ChildExitCode = [int]$LASTEXITCODE
    return $ChildExitCode
}

function Wait-ForScheduledStep {
    param(
        [Parameter(Mandatory = $true)]
        [TimeSpan]$Delay,

        [Parameter(Mandatory = $true)]
        [TimeSpan]$UpdateInterval,

        [Parameter(Mandatory = $true)]
        [int]$StepNumber,

        [Parameter(Mandatory = $true)]
        [int]$StepCount
    )

    if ($Delay -eq [TimeSpan]::Zero) {
        Write-Host ("[scheduler] Step {0}/{1} has no delay." -f $StepNumber, $StepCount)
        return
    }

    $RunAt = [DateTimeOffset]::Now.Add($Delay)
    while ($true) {
        $Now = [DateTimeOffset]::Now
        $Remaining = $RunAt - $Now
        if ($Remaining -le [TimeSpan]::Zero) {
            break
        }

        $DisplayRemaining = [TimeSpan]::FromSeconds([Math]::Ceiling($Remaining.TotalSeconds))
        Write-Host (
            "[{0:yyyy-MM-dd HH:mm:ss zzz}] Step {1}/{2} starts at {3:yyyy-MM-dd HH:mm:ss zzz}; remaining {4}." -f
            $Now,
            $StepNumber,
            $StepCount,
            $RunAt,
            (Format-Duration $DisplayRemaining)
        )

        $SleepMilliseconds = [Math]::Min($Remaining.TotalMilliseconds, $UpdateInterval.TotalMilliseconds)
        $SleepMilliseconds = [Math]::Min($SleepMilliseconds, [int]::MaxValue)
        Start-Sleep -Milliseconds ([int][Math]::Max(1, [Math]::Ceiling($SleepMilliseconds)))
    }
}

$UpdateInterval = [TimeSpan]::FromMinutes(5)
$WorkingDirectory = $null
$ContinueOnError = $false
$DryRun = $false
$PendingDelay = $null
$Steps = [System.Collections.Generic.List[object]]::new()

try {
    for ($Index = 0; $Index -lt $args.Count; $Index++) {
        $Argument = [string]$args[$Index]
        switch ($Argument) {
            { $_ -in "--help", "-h" } {
                Show-Usage
                exit 0
            }
            { $_ -in "--update_interval", "--update-interval" } {
                if ($Index + 1 -ge $args.Count) {
                    throw "$Argument requires a duration."
                }
                $Index++
                $UpdateInterval = ConvertTo-Duration $args[$Index] $Argument
                if ($UpdateInterval -le [TimeSpan]::Zero) {
                    throw "$Argument must be greater than zero."
                }
            }
            { $_ -in "--working_directory", "--working-directory" } {
                if ($Index + 1 -ge $args.Count) {
                    throw "$Argument requires a path."
                }
                $Index++
                $WorkingDirectory = [string]$args[$Index]
            }
            { $_ -in "--continue_on_error", "--continue-on-error" } {
                $ContinueOnError = $true
            }
            { $_ -in "--dry_run", "--dry-run" } {
                $DryRun = $true
            }
            "--delay" {
                if ($null -ne $PendingDelay) {
                    throw "Each --delay must be followed by one --command before another --delay."
                }
                if ($Index + 1 -ge $args.Count) {
                    throw "--delay requires a duration."
                }
                $Index++
                $PendingDelay = ConvertTo-Duration $args[$Index] "--delay"
            }
            "--command" {
                if ($null -eq $PendingDelay) {
                    throw "--command requires a preceding --delay."
                }
                if ($Index + 1 -ge $args.Count) {
                    throw "--command requires a script block or quoted command string."
                }
                $Index++
                $CommandValue = $args[$Index]
                if ($CommandValue -is [scriptblock]) {
                    $Command = $CommandValue
                } elseif ($CommandValue -is [string] -and -not [string]::IsNullOrWhiteSpace($CommandValue)) {
                    $Command = [scriptblock]::Create($CommandValue)
                } else {
                    throw "--command requires a nonempty script block or quoted command string."
                }

                $Steps.Add(
                    [pscustomobject]@{
                        Delay = $PendingDelay
                        Command = $Command
                    }
                )
                $PendingDelay = $null
            }
            default {
                throw "Unknown argument '$Argument'."
            }
        }
    }

    if ($null -ne $PendingDelay) {
        throw "The final --delay is missing its --command."
    }
    if ($Steps.Count -eq 0) {
        throw "At least one --delay/--command pair is required."
    }

    $ResolvedWorkingDirectory = $null
    if ($null -ne $WorkingDirectory) {
        if (-not (Test-Path -LiteralPath $WorkingDirectory -PathType Container)) {
            throw "Working directory '$WorkingDirectory' does not exist or is not a directory."
        }
        $ResolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
    }
} catch {
    Write-Host ("[scheduler] {0}" -f $_.Exception.Message) -ForegroundColor Red
    Write-Host ""
    Show-Usage
    exit 2
}

Write-Host (
    "[scheduler] Loaded {0} step(s); countdown updates every {1}." -f
    $Steps.Count,
    (Format-Duration $UpdateInterval)
)
if ($null -ne $ResolvedWorkingDirectory) {
    Write-Host ("[scheduler] Working directory: {0}" -f $ResolvedWorkingDirectory)
}
for ($Index = 0; $Index -lt $Steps.Count; $Index++) {
    $Step = $Steps[$Index]
    Write-Host (
        "[scheduler]   {0}. wait {1}, then: {2}" -f
        ($Index + 1),
        (Format-Duration $Step.Delay),
        (Format-Command $Step.Command)
    )
}

if ($DryRun) {
    Write-Host "[scheduler] Dry run complete; no delays or commands were run."
    exit 0
}

$ChangedLocation = $false
$OverallExitCode = 0
try {
    if ($null -ne $ResolvedWorkingDirectory) {
        Push-Location -LiteralPath $ResolvedWorkingDirectory
        $ChangedLocation = $true
    }

    for ($Index = 0; $Index -lt $Steps.Count; $Index++) {
        $Step = $Steps[$Index]
        $StepNumber = $Index + 1
        Wait-ForScheduledStep $Step.Delay $UpdateInterval $StepNumber $Steps.Count

        $CommandText = Format-Command $Step.Command
        $StartedAt = [DateTimeOffset]::Now
        Write-Host (
            "[{0:yyyy-MM-dd HH:mm:ss zzz}] Starting step {1}/{2}: {3}" -f
            $StartedAt,
            $StepNumber,
            $Steps.Count,
            $CommandText
        )

        $CommandExitCode = 0
        try {
            $CommandExitCode = Invoke-IsolatedCommand $Step.Command (Get-Location).ProviderPath
        } catch {
            $CommandExitCode = 1
            Write-Host (
                "[scheduler] Step {0}/{1} raised an error: {2}" -f
                $StepNumber,
                $Steps.Count,
                $_.Exception.Message
            ) -ForegroundColor Red
        }

        $Elapsed = [DateTimeOffset]::Now - $StartedAt
        if ($CommandExitCode -eq 0) {
            Write-Host (
                "[{0:yyyy-MM-dd HH:mm:ss zzz}] Completed step {1}/{2} in {3}." -f
                [DateTimeOffset]::Now,
                $StepNumber,
                $Steps.Count,
                (Format-Duration $Elapsed)
            )
            continue
        }

        if ($OverallExitCode -eq 0) {
            $OverallExitCode = $CommandExitCode
        }
        Write-Host (
            "[scheduler] Step {0}/{1} failed with exit code {2}." -f
            $StepNumber,
            $Steps.Count,
            $CommandExitCode
        ) -ForegroundColor Red

        if (-not $ContinueOnError) {
            Write-Host "[scheduler] Stopping; remaining steps were not run."
            break
        }
        Write-Host "[scheduler] Continuing because --continue_on_error was supplied."
    }
} finally {
    if ($ChangedLocation) {
        Pop-Location
    }
}

if ($OverallExitCode -eq 0) {
    Write-Host "[scheduler] Schedule completed successfully."
} else {
    Write-Host ("[scheduler] Schedule completed with failures (exit code {0})." -f $OverallExitCode) -ForegroundColor Red
}
exit $OverallExitCode
