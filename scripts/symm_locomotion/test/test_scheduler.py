# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCHEDULER_PY = SCRIPT_DIR / "scheduler.py"
SCHEDULER_PS1 = SCRIPT_DIR / "scheduler.ps1"
SCHEDULER_SH = SCRIPT_DIR / "scheduler.sh"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _find_bash() -> str | None:
    executable = shutil.which("bash")
    if executable is not None:
        return executable
    git = shutil.which("git")
    if os.name == "nt" and git is not None:
        candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


BASH = _find_bash()


def _load_scheduler():
    spec = importlib.util.spec_from_file_location("_test_scheduler", SCHEDULER_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCHEDULER = _load_scheduler()


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("0", 0.0),
        ("500ms", 0.5),
        ("10min", 600.0),
        ("1.5hours", 5400.0),
        ("01:30:00", 5400.0),
        ("2.01:02:03", 176523.0),
    ],
)
def test_duration_formats(text, seconds):
    assert SCHEDULER._parse_duration(text, "--delay") == pytest.approx(seconds)


def test_dry_run_does_not_resolve_or_execute_shell():
    result = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER_PY),
            "--dry_run",
            "--shell",
            "bash",
            "--shell_executable",
            "definitely_missing_scheduler_shell",
            "--delay",
            "0s",
            "--command",
            "exit 93",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "Dry run complete" in result.stdout


def test_rejects_non_interleaved_delay_command_pairs():
    result = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER_PY),
            "--dry_run",
            "--delay",
            "0s",
            "--delay",
            "1s",
            "--command",
            "echo unreachable",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Each --delay must be followed by one --command" in result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is unavailable")
def test_python_scheduler_runs_posix_commands_and_preserves_first_failure():
    result = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER_PY),
            "--shell",
            "bash",
            "--shell_executable",
            BASH,
            "--continue_on_error",
            "--delay",
            "0s",
            "--command",
            "exit 7",
            "--delay",
            "0s",
            "--command",
            "printf 'SCHEDULER_SECOND_STEP_RAN\\n'",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 7
    assert "SCHEDULER_SECOND_STEP_RAN" in output
    assert "Step 1/2 failed with exit code 7" in output
    assert "Completed step 2/2" in output


@pytest.mark.skipif(BASH is None, reason="bash is unavailable")
def test_bash_wrapper_executes_canonical_scheduler():
    environment = os.environ.copy()
    environment["PYTHON"] = sys.executable.replace("\\", "/")
    result = subprocess.run(
        [
            BASH,
            str(SCHEDULER_SH).replace("\\", "/"),
            "--delay",
            "0s",
            "--command",
            "printf 'SCHEDULER_SH_RAN\\n'",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "SCHEDULER_SH_RAN" in output
    assert "Schedule completed successfully" in output


def test_default_failure_stops_remaining_steps(tmp_path):
    if os.name == "nt" and POWERSHELL is not None:
        first_command = "exit 7"
        second_command = "Set-Content -LiteralPath should_not_exist.txt -Value ran"
    elif BASH is not None:
        first_command = "exit 7"
        second_command = "printf ran > should_not_exist.txt"
    else:
        pytest.skip("the platform's default command shell is unavailable")

    result = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER_PY),
            "--working_directory",
            str(tmp_path),
            "--delay",
            "0s",
            "--command",
            first_command,
            "--delay",
            "0s",
            "--command",
            second_command,
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 7
    assert not (tmp_path / "should_not_exist.txt").exists()
    assert "Stopping; remaining steps were not run" in output


def test_working_directory_is_applied_without_changing_scheduler_process_directory(tmp_path, monkeypatch):
    observed_directories = []

    def _record_command(kind, executable, command, working_directory):
        del kind, executable, command
        observed_directories.append(working_directory)
        return 0

    monkeypatch.setattr(SCHEDULER, "_resolve_shell", lambda kind, executable: ("sh", "unused"))
    monkeypatch.setattr(SCHEDULER, "_run_command", _record_command)
    original_directory = Path.cwd()
    exit_code = SCHEDULER._run_schedule(
        [SCHEDULER._Step(0.0, "unused")],
        update_interval_seconds=300.0,
        working_directory=tmp_path,
        continue_on_error=False,
        shell_kind="auto",
        shell_executable=None,
    )

    assert exit_code == 0
    assert observed_directories == [tmp_path]
    assert Path.cwd() == original_directory


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is unavailable")
def test_powershell_wrapper_forwards_scriptblocks_and_isolates_exit():
    scheduler_path = str(SCHEDULER_PS1).replace("'", "''")
    invocation = (
        f"& '{scheduler_path}' --continue_on_error "
        "--delay 0s --command { exit 7 } "
        "--delay 0s --command { Write-Output SCHEDULER_SECOND_STEP_RAN }; "
        "exit $LASTEXITCODE"
    )
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            invocation,
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 7
    assert "SCHEDULER_SECOND_STEP_RAN" in output
    assert "Step 1/2 failed with exit code 7" in output
    assert "Completed step 2/2" in output


def test_wrapper_files_forward_to_canonical_python_entrypoint():
    assert "scheduler.py" in SCHEDULER_PS1.read_text(encoding="utf-8")
    assert "scheduler.py" in SCHEDULER_SH.read_text(encoding="utf-8")
