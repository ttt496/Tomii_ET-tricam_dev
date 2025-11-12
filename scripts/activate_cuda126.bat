@echo off
REM Use CUDA 12.6 only within this shell for this repo
REM Prepend CUDA and cuDNN paths so DLL lookup uses 12.6 first

SETLOCAL ENABLEDELAYEDEXPANSION

REM Adjust these paths if CUDA/cuDNN are installed elsewhere
set "CUDA_BASE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDNN_BIN=C:\tools\cudnn\bin"

if not exist "%CUDA_BASE%\bin\nvcc.exe" (
  echo [WARN] nvcc not found at "%CUDA_BASE%\bin\nvcc.exe". Please verify CUDA 12.6 installation path.
)

REM Export helpful vars
set "CUDA_PATH=%CUDA_BASE%"
set "CUDA_HOME=%CUDA_BASE%"

REM Prepend CUDA/cuDNN bins to PATH just for this process
set "PATH=%CUDA_BASE%\bin;%CUDA_BASE%\libnvvp;%CUDNN_BIN%;%PATH%"

echo.
echo CUDA configured for this shell:
echo  - CUDA_PATH=%CUDA_PATH%
echo  - Using bin: %CUDA_BASE%\bin
echo  - cuDNN bin: %CUDNN_BIN%
echo.
echo To run, use:
echo   .venv\Scripts\python.exe eye_crop_pipeline.py
echo.

REM Keep shell open for interactive use
cmd /K


