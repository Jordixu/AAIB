@echo off
setlocal enabledelayedexpansion
pushd "%~dp0.."

set "ARGS="

:loop
if "%~1"=="" goto :run
if /i "%~1"=="--cpu" (
    set "CUDA_VISIBLE_DEVICES=-1"
    echo [INFO] Forcing CPU training
) else (
    set "ARGS=!ARGS! %~1"
)
shift
goto :loop

:run
python scripts\train_cv.py --config configs\efficientnet_v2_l.yaml !ARGS!
