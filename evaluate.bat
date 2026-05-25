@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: evaluate.bat [--cpu] ^<run_dir^> [--fold N] [--split val^|train]
    echo.
    echo   --cpu   Force CPU inference ^(GPU auto-detected by default^)
    echo.
    echo Examples:
    echo   evaluate.bat outputs\runs\myrun_fold0
    echo   evaluate.bat --cpu outputs\runs\myrun_fold0 --fold 1
    exit /b 1
)

set "ARGS="

:loop
if "%~1"=="" goto :run
if /i "%~1"=="--cpu" (
    set "CUDA_VISIBLE_DEVICES=-1"
    echo [INFO] Forcing CPU inference
) else (
    set "ARGS=!ARGS! %~1"
)
shift
goto :loop

:run
python scripts\evaluate.py !ARGS!
