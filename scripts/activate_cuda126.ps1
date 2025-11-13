<# 
Use CUDA 12.6 only within this PowerShell session (repo-local).
Adjust paths if installed elsewhere.
#>

$CudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$CuDnnBin = "C:\tools\cudnn\bin"

if (-not (Test-Path "$CudaBase\bin\nvcc.exe")) {
  Write-Warning "nvcc not found at $CudaBase\bin\nvcc.exe. Check CUDA 12.6 installation."
}

$env:CUDA_PATH = $CudaBase
$env:CUDA_HOME = $CudaBase

# Prepend CUDA/cuDNN to PATH for this session
$prepend = @("$CudaBase\bin", "$CudaBase\libnvvp", "$CuDnnBin") -join ";"
if ($env:PATH -notlike "$prepend*") {
  $env:PATH = "$prepend;$env:PATH"
}

Write-Host ""
Write-Host "CUDA configured for this session:"
Write-Host " - CUDA_PATH=$($env:CUDA_PATH)"
Write-Host " - Using bin: $CudaBase\bin"
Write-Host " - cuDNN bin: $CuDnnBin"
Write-Host ""
Write-Host "Run your script, e.g.:"
Write-Host "   .\.venv\Scripts\python.exe eye_crop_pipeline.py"
Write-Host ""


