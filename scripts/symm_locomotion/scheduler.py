# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run delayed shell commands sequentially.

The Python entry point contains the platform-neutral scheduling logic. The
matching ``scheduler.sh`` and ``scheduler.ps1`` launchers select the native
command shell and forward their arguments here.
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

_DURATION_PATTERN = re.compile(
    r"^(?P<amount>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?)$",
    re.IGNORECASE,
)
_TIMESPAN_PATTERN = re.compile(
    r"^(?:(?P<days>[0-9]+)\.)?"
    r"(?P<hours>[0-9]+):(?P<minutes>[0-9]{1,2}):(?P<seconds>[0-9]{1,2}(?:\.[0-9]+)?)$"
)
_ZERO_PATTERN = re.compile(r"^0+(?:\.0+)?$")
_SHELL_KINDS = ("auto", "powershell", "bash", "sh")

_ShellKind = Literal["auto", "powershell", "bash", "sh"]


@dataclass(frozen=True)
class _Step:
    delay_seconds: float
    command: str


class _ScheduleEventAction(argparse.Action):
    """Record interleaved ``--delay`` and ``--command`` arguments."""

    def __init__(self, *args, event_kind: Literal["delay", "command"], **kwargs):
        self.event_kind = event_kind
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        events = list(getattr(namespace, self.dest, None) or [])
        events.append((self.event_kind, values))
        setattr(namespace, self.dest, events)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Schedule one or more shell commands with a delay before each command.",
        epilog=(
            "Durations accept values such as 500ms, 10min, 5hours, or 01:30:00. "
            "Each delay starts after the preceding command finishes."
        ),
    )
    parser.set_defaults(schedule_events=[])
    parser.add_argument(
        "--delay",
        dest="schedule_events",
        action=_ScheduleEventAction,
        event_kind="delay",
        metavar="DURATION",
        help="delay before the next command; repeat with --command for every step",
    )
    parser.add_argument(
        "--command",
        dest="schedule_events",
        action=_ScheduleEventAction,
        event_kind="command",
        metavar="COMMAND",
        help="shell command to run after the preceding --delay",
    )
    parser.add_argument(
        "--update_interval",
        "--update-interval",
        default="5m",
        metavar="DURATION",
        help="countdown update interval (default: 5m)",
    )
    parser.add_argument(
        "--working_directory",
        "--working-directory",
        type=Path,
        metavar="PATH",
        help="directory in which all commands run",
    )
    parser.add_argument(
        "--continue_on_error",
        "--continue-on-error",
        action="store_true",
        help="continue after a command returns a nonzero exit code",
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="validate and display the schedule without waiting or running commands",
    )
    parser.add_argument(
        "--shell",
        choices=_SHELL_KINDS,
        default="auto",
        help="command shell (default: PowerShell on Windows, bash otherwise)",
    )
    parser.add_argument(
        "--shell_executable",
        "--shell-executable",
        metavar="PATH",
        help="override the executable used for the selected command shell",
    )
    return parser


def _duration_error(option_name: str, value: str) -> ValueError:
    return ValueError(f"{option_name} expects a duration such as '10min', '5hours', or '01:30:00'; received '{value}'.")


def _parse_duration(value: str, option_name: str) -> float:
    text = str(value).strip()
    match = _DURATION_PATTERN.fullmatch(text)
    if match is not None:
        amount = float(match.group("amount"))
        unit = match.group("unit").lower()
        if unit in {"ms", "millisecond", "milliseconds"}:
            seconds = amount / 1000.0
        elif unit in {"s", "sec", "secs", "second", "seconds"}:
            seconds = amount
        elif unit in {"m", "min", "mins", "minute", "minutes"}:
            seconds = amount * 60.0
        elif unit in {"h", "hr", "hrs", "hour", "hours"}:
            seconds = amount * 3600.0
        else:
            seconds = amount * 86400.0
    elif _ZERO_PATTERN.fullmatch(text) is not None:
        # PowerShell converts the literal ``0s`` to numeric zero before a
        # script receives it, so retain the original launcher's zero-delay
        # behavior when scheduler.ps1 forwards that value.
        seconds = 0.0
    else:
        timespan = _TIMESPAN_PATTERN.fullmatch(text)
        if timespan is None:
            raise _duration_error(option_name, text)
        days = int(timespan.group("days") or 0)
        hours = int(timespan.group("hours"))
        minutes = int(timespan.group("minutes"))
        seconds_part = float(timespan.group("seconds"))
        if minutes >= 60 or seconds_part >= 60 or (days > 0 and hours >= 24):
            raise _duration_error(option_name, text)
        seconds = days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds_part

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{option_name} must be a finite, nonnegative duration.")
    return seconds


def _format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"

    total_seconds = int(math.ceil(seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _format_command(command: str) -> str:
    return " ".join(command.split())


def _build_steps(
    parser: argparse.ArgumentParser,
    events: Sequence[tuple[Literal["delay", "command"], str]],
) -> list[_Step]:
    steps: list[_Step] = []
    pending_delay: float | None = None
    for event_kind, value in events:
        if event_kind == "delay":
            if pending_delay is not None:
                parser.error("Each --delay must be followed by one --command before another --delay.")
            try:
                pending_delay = _parse_duration(value, "--delay")
            except ValueError as error:
                parser.error(str(error))
        else:
            if pending_delay is None:
                parser.error("--command requires a preceding --delay.")
            command = str(value).strip()
            if not command:
                parser.error("--command requires a nonempty command string.")
            steps.append(_Step(delay_seconds=pending_delay, command=command))
            pending_delay = None

    if pending_delay is not None:
        parser.error("The final --delay is missing its --command.")
    if not steps:
        parser.error("At least one --delay/--command pair is required.")
    return steps


def _resolve_working_directory(parser: argparse.ArgumentParser, value: Path | None) -> Path:
    if value is None:
        return Path.cwd()
    try:
        resolved = value.expanduser().resolve(strict=True)
    except OSError:
        parser.error(f"Working directory '{value}' does not exist or is not a directory.")
    if not resolved.is_dir():
        parser.error(f"Working directory '{value}' does not exist or is not a directory.")
    return resolved


def _resolve_shell(kind: _ShellKind, executable_override: str | None) -> tuple[_ShellKind, str]:
    resolved_kind: _ShellKind = kind
    if resolved_kind == "auto":
        resolved_kind = "powershell" if os.name == "nt" else "bash"

    if executable_override:
        executable_path = Path(executable_override).expanduser()
        if executable_path.parent != Path("."):
            if not executable_path.is_file():
                raise RuntimeError(f"Shell executable '{executable_override}' does not exist or is not a file.")
            return resolved_kind, str(executable_path.resolve())
        resolved_override = shutil.which(executable_override)
        if resolved_override is None:
            raise RuntimeError(f"Shell executable '{executable_override}' was not found on PATH.")
        return resolved_kind, resolved_override

    candidates = {
        "powershell": ("pwsh", "powershell"),
        "bash": ("bash",),
        "sh": ("sh",),
    }[resolved_kind]
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable is not None:
            return resolved_kind, executable
    raise RuntimeError(f"No executable was found for the '{resolved_kind}' command shell.")


def _powershell_source(command: str) -> str:
    return f"""\
$ErrorActionPreference = "Stop"
try {{
    & {{
{command}
    }}
    $CommandSucceeded = $?
    $CommandExitCode = if ($null -eq $LASTEXITCODE) {{ 0 }} else {{ [int]$LASTEXITCODE }}
    if (-not $CommandSucceeded -and $CommandExitCode -eq 0) {{
        $CommandExitCode = 1
    }}
    exit $CommandExitCode
}} catch {{
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}}
"""


def _command_argv(kind: _ShellKind, executable: str, command: str) -> list[str]:
    if kind == "powershell":
        source = _powershell_source(command)
        encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
        return [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ]
    return [executable, "-c", command]


def _run_command(kind: _ShellKind, executable: str, command: str, working_directory: Path) -> int:
    try:
        result = subprocess.run(
            _command_argv(kind, executable, command),
            cwd=working_directory,
            check=False,
        )
    except OSError as error:
        print(
            f"[scheduler] Unable to start command shell: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if result.returncode < 0:
        return 128 + abs(result.returncode)
    return result.returncode


def _timestamp(value: datetime) -> str:
    text = value.strftime("%Y-%m-%d %H:%M:%S %z")
    if len(text) >= 5:
        text = f"{text[:-2]}:{text[-2:]}"
    return text


def _wait_for_step(
    delay_seconds: float,
    update_interval_seconds: float,
    step_number: int,
    step_count: int,
) -> None:
    if delay_seconds == 0:
        print(f"[scheduler] Step {step_number}/{step_count} has no delay.", flush=True)
        return

    started_at = datetime.now().astimezone()
    run_at = started_at + timedelta(seconds=delay_seconds)
    deadline = time.monotonic() + delay_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        now = datetime.now().astimezone()
        print(
            f"[{_timestamp(now)}] Step {step_number}/{step_count} starts at {_timestamp(run_at)}; "
            f"remaining {_format_duration(remaining)}.",
            flush=True,
        )
        time.sleep(min(remaining, update_interval_seconds))


def _run_schedule(
    steps: Sequence[_Step],
    *,
    update_interval_seconds: float,
    working_directory: Path,
    continue_on_error: bool,
    shell_kind: _ShellKind,
    shell_executable: str | None,
) -> int:
    try:
        resolved_kind, executable = _resolve_shell(shell_kind, shell_executable)
    except RuntimeError as error:
        print(f"[scheduler] {error}", file=sys.stderr, flush=True)
        return 2

    overall_exit_code = 0
    for index, step in enumerate(steps):
        step_number = index + 1
        _wait_for_step(step.delay_seconds, update_interval_seconds, step_number, len(steps))

        command_text = _format_command(step.command)
        started_at = datetime.now().astimezone()
        started_monotonic = time.monotonic()
        print(
            f"[{_timestamp(started_at)}] Starting step {step_number}/{len(steps)}: {command_text}",
            flush=True,
        )
        command_exit_code = _run_command(resolved_kind, executable, step.command, working_directory)
        elapsed = time.monotonic() - started_monotonic
        if command_exit_code == 0:
            print(
                f"[{_timestamp(datetime.now().astimezone())}] Completed step {step_number}/{len(steps)} "
                f"in {_format_duration(elapsed)}.",
                flush=True,
            )
            continue

        if overall_exit_code == 0:
            overall_exit_code = command_exit_code
        print(
            f"[scheduler] Step {step_number}/{len(steps)} failed with exit code {command_exit_code}.",
            file=sys.stderr,
            flush=True,
        )
        if not continue_on_error:
            print(
                "[scheduler] Stopping; remaining steps were not run.",
                file=sys.stderr,
                flush=True,
            )
            break
        print(
            "[scheduler] Continuing because --continue_on_error was supplied.",
            flush=True,
        )

    if overall_exit_code == 0:
        print("[scheduler] Schedule completed successfully.", flush=True)
    else:
        print(
            f"[scheduler] Schedule completed with failures (exit code {overall_exit_code}).",
            file=sys.stderr,
            flush=True,
        )
    return overall_exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run the delayed-command scheduler CLI."""
    parser = _create_parser()
    args = parser.parse_args(argv)
    steps = _build_steps(parser, args.schedule_events)
    try:
        update_interval_seconds = _parse_duration(args.update_interval, "--update_interval")
    except ValueError as error:
        parser.error(str(error))
    if update_interval_seconds <= 0:
        parser.error("--update_interval must be greater than zero.")
    working_directory = _resolve_working_directory(parser, args.working_directory)

    print(
        f"[scheduler] Loaded {len(steps)} step(s); countdown updates every "
        f"{_format_duration(update_interval_seconds)}.",
        flush=True,
    )
    if args.working_directory is not None:
        print(f"[scheduler] Working directory: {working_directory}", flush=True)
    print(f"[scheduler] Command shell: {args.shell}.", flush=True)
    for index, step in enumerate(steps):
        print(
            f"[scheduler]   {index + 1}. wait {_format_duration(step.delay_seconds)}, "
            f"then: {_format_command(step.command)}",
            flush=True,
        )

    if args.dry_run:
        print("[scheduler] Dry run complete; no delays or commands were run.", flush=True)
        return 0

    return _run_schedule(
        steps,
        update_interval_seconds=update_interval_seconds,
        working_directory=working_directory,
        continue_on_error=args.continue_on_error,
        shell_kind=args.shell,
        shell_executable=args.shell_executable,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[scheduler] Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
