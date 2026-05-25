@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: evaluate_test.bat [--cpu] ^<run_dir^> [--out PATH]
    echo        evaluate_test.bat [--cpu] --cv-summary ^<cv_dir^> [--out PATH]
    echo.
    echo   --cpu   Force CPU inference ^(GPU auto-detected by default^)
    echo.
    echo Examples:
    echo   evaluate_test.bat outputs\runs\myrun_fold0
    echo   evaluate_test.bat --cv-summary outputs\cv_summary\myrun
    echo   evaluate_test.bat --cpu --cv-summary outputs\cv_summary\myrun --out outputs\test_evaluation
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
python scripts\evaluate_test.py !ARGS!
