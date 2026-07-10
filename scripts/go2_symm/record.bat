@echo off
setlocal EnableExtensions

rem Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
rem All rights reserved.
rem
rem SPDX-License-Identifier: BSD-3-Clause

set "DEFAULT_CONDA_ENV=go2_symm_rl_lab"
if not defined CONDA_PREFIX set "CONDA_PREFIX=%USERPROFILE%\.conda\envs\%DEFAULT_CONDA_ENV%"

set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if exist "%PYTHON_EXE%" (
    if not defined CONDA_DEFAULT_ENV set "CONDA_DEFAULT_ENV=%DEFAULT_CONDA_ENV%"
    "%PYTHON_EXE%" "%~dp0record.py" %*
) else (
    python "%~dp0record.py" %*
)

exit /b %ERRORLEVEL%
