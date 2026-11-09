param(
    [switch]$SkipTests,
    [switch]$SkipZip,
    [switch]$ReuseFrozen,
    [ValidateSet("CPU", "GPU")]
    [string]$Flavor = "GPU"
)

$ErrorActionPreference = "Stop"
$platform = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$project = [System.IO.Path]::GetFullPath((Join-Path $platform ".."))
$releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $platform "release"))
$portable = [System.IO.Path]::GetFullPath((Join-Path $releaseRoot "NeuroCIM-Windows-x64"))
$buildRoot = [System.IO.Path]::GetFullPath((Join-Path $releaseRoot ".build"))
$distRoot = Join-Path $buildRoot "dist"
$workRoot = Join-Path $buildRoot "work"
$spec = Join-Path $platform "packaging\neurocim.spec"

foreach ($target in @($portable, $buildRoot)) {
    if (-not $target.StartsWith($releaseRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe release target: $target"
    }
}

if (-not $SkipTests) {
    Push-Location $platform
    try {
        python -m pytest -q tests/test_gui_smoke.py tests/test_paths.py tests/test_model_registry.py
        if ($LASTEXITCODE -ne 0) { throw "Release preflight tests failed" }
    } finally {
        Pop-Location
    }
}

# Never publish a GPU-labelled archive from a CPU-only or non-functional CUDA
# environment. The frozen executable is checked again after packaging. CPU
# releases use a separate gate in verify_frozen_app.py.
New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
if ($Flavor -eq "GPU") {
    $sourceGpuReport = Join-Path $releaseRoot "source_gpu_smoke.json"
    python (Join-Path $platform "scripts\gpu_release_smoke.py") --output $sourceGpuReport
    if ($LASTEXITCODE -ne 0) { throw "Source CUDA release gate failed" }
} else {
    $torchCuda = python -c "import torch; print(torch.version.cuda or '')"
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect source CPU PyTorch runtime" }
    if (($torchCuda | Out-String).Trim()) {
        throw "CPU release requires a CPU-only PyTorch build"
    }
}

$appTarget = Join-Path $portable "NeuroCIM"
if (-not $ReuseFrozen) {
    if (Test-Path -LiteralPath $portable) { Remove-Item -LiteralPath $portable -Recurse -Force }
    if (Test-Path -LiteralPath $buildRoot) { Remove-Item -LiteralPath $buildRoot -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $releaseRoot, $portable, $buildRoot | Out-Null

    Push-Location $platform
    try {
        $previousQtApi = $env:QT_API
        $env:QT_API = "pyside6"
        python -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $workRoot $spec
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
    } finally {
        $env:QT_API = $previousQtApi
        Pop-Location
    }

    $builtApp = Join-Path $distRoot "NeuroCIM"
    if (-not (Test-Path -LiteralPath (Join-Path $builtApp "NeuroCIM.exe"))) {
        throw "PyInstaller output is missing NeuroCIM.exe"
    }
    Copy-Item -LiteralPath $builtApp -Destination $appTarget -Recurse
} elseif (-not (Test-Path -LiteralPath (Join-Path $appTarget "NeuroCIM.exe"))) {
    throw "Cannot reuse frozen application because NeuroCIM.exe is missing"
}

# The Windows CPU wheel links fbgemm.dll and torch_cpu.dll against LLVM OpenMP,
# but PyInstaller does not discover libomp140.x86_64.dll when it is supplied by
# the Windows/Visual Studio runtime. Bundle it beside the importing Torch DLLs
# so a clean Windows computer does not fail with WinError 126.
if ($Flavor -eq "CPU") {
    $openMpTarget = Join-Path $appTarget "_internal\torch\lib\libomp140.x86_64.dll"
    $openMpCandidates = @(
        (Join-Path $env:WINDIR "System32\libomp140.x86_64.dll"),
        (Get-Command "libomp140.x86_64.dll" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
    $openMpSource = $openMpCandidates | Select-Object -First 1
    if (-not $openMpSource) {
        throw "CPU release requires libomp140.x86_64.dll from the LLVM/MSVC runtime"
    }
    Copy-Item -LiteralPath $openMpSource -Destination $openMpTarget -Force
}

$neurosimTarget = Join-Path $appTarget "outputs\neurosim\bin"
New-Item -ItemType Directory -Force -Path $neurosimTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $platform "outputs\neurosim\bin\neurosim.exe") -Destination $neurosimTarget
$mingwBin = Join-Path $platform ".tools\mingw\Library\mingw-w64\bin"
foreach ($dll in @("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")) {
    Copy-Item -LiteralPath (Join-Path $mingwBin $dll) -Destination $neurosimTarget
}

$resourceDirectories = @(
    "3_models\LeNet_5\chekpoint",
    "3_models\BNN\checkpoint",
    "3_models\Resnet20\checkpoint",
    "3_models\Resnet20\output\formal",
    "3_models\TinyTransformer\checkpoint",
    "3_models\VisionZoo\checkpoint",
    "3_models\GPT2\checkpoint\tokenizer",
    "data\mnist",
    "data\cifar10",
    "0_data"
)
foreach ($relative in $resourceDirectories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $portable $relative) | Out-Null
}

function Copy-ResourceTree {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Required bundled resource directory is missing: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($file in Get-ChildItem -LiteralPath $Source -Recurse -File) {
        $relative = $file.FullName.Substring($Source.Length).TrimStart('\')
        $target = Join-Path $Destination $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
    }
}

Copy-ResourceTree (Join-Path $project "3_models\LeNet_5\chekpoint") (Join-Path $portable "3_models\LeNet_5\chekpoint")
Copy-ResourceTree (Join-Path $project "3_models\BNN\checkpoint") (Join-Path $portable "3_models\BNN\checkpoint")
Copy-ResourceTree (Join-Path $project "3_models\Resnet20\checkpoint") (Join-Path $portable "3_models\Resnet20\checkpoint")
Copy-Item -LiteralPath (Join-Path $project "3_models\Resnet20\output\formal\input_scales.json") -Destination (Join-Path $portable "3_models\Resnet20\output\formal\input_scales.json") -Force
Copy-ResourceTree (Join-Path $project "3_models\TinyTransformer\checkpoint") (Join-Path $portable "3_models\TinyTransformer\checkpoint")
Copy-ResourceTree (Join-Path $project "3_models\VisionZoo\checkpoint") (Join-Path $portable "3_models\VisionZoo\checkpoint")
Copy-ResourceTree (Join-Path $project "3_models\GPT2\checkpoint") (Join-Path $portable "3_models\GPT2\checkpoint")
# GPT-2-large runs from the verified FP16 checkpoint.  Shipping the original
# FP32 checkpoint as well adds 3.25 GB without providing another runtime path.
$redundantGpt2Large = Join-Path $portable "3_models\GPT2\checkpoint\gpt2-large\model.safetensors"
if (Test-Path -LiteralPath $redundantGpt2Large) {
    Remove-Item -LiteralPath $redundantGpt2Large -Force
}
$measurementTarget = Join-Path $portable "0_data"
if (Test-Path -LiteralPath $measurementTarget) {
    Remove-Item -LiteralPath $measurementTarget -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $measurementTarget | Out-Null
$measurementFiles = Get-ChildItem -LiteralPath (Join-Path $project "0_data") -Filter "*.csv" -File
if ($measurementFiles.Count -lt 5) {
    throw "Expected at least five device measurement CSV files"
}
foreach ($file in $measurementFiles) {
    Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $measurementTarget $file.Name) -Force
}
Copy-Item -LiteralPath (Join-Path $platform "model_resource_manifest.json") -Destination $portable -Force

$mnistSource = Join-Path $project "data\mnist\MNIST\raw"
$mnistTarget = Join-Path $portable "data\mnist\MNIST\raw"
New-Item -ItemType Directory -Force -Path $mnistTarget | Out-Null
foreach ($name in @("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")) {
    Copy-Item -LiteralPath (Join-Path $mnistSource $name) -Destination (Join-Path $mnistTarget $name) -Force
}
Copy-ResourceTree (Join-Path $project "data\cifar10\cifar-10-batches-py") (Join-Path $portable "data\cifar10\cifar-10-batches-py")

python (Join-Path $platform "scripts\verify_release_resources.py") $portable --flavor $Flavor
if ($LASTEXITCODE -ne 0) { throw "Bundled resource verification failed" }

$docsTarget = Join-Path $portable "docs"
$licensesTarget = Join-Path $portable "licenses"
New-Item -ItemType Directory -Force -Path $docsTarget, $licensesTarget | Out-Null
foreach ($name in @("USER_MANUAL.md", "ENVIRONMENT_SETUP.md", "TECHNICAL_MANUAL.md", "RELEASE_GUIDE.md")) {
    Copy-Item -LiteralPath (Join-Path $platform "docs\$name") -Destination $docsTarget
}
$resourceGuide = Get-ChildItem -LiteralPath (Join-Path $platform "packaging") -Filter "*.md" -File | Select-Object -First 1
if ($null -eq $resourceGuide) { throw "External resource guide is missing" }
Copy-Item -LiteralPath $resourceGuide.FullName -Destination $portable
Copy-Item -LiteralPath (Join-Path $platform "THIRD_PARTY_NOTICES.md") -Destination $portable
Copy-Item -LiteralPath (Join-Path $platform "vendor\neurosim\README.md") -Destination (Join-Path $licensesTarget "NeuroSim_README.md")
Copy-Item -LiteralPath (Join-Path $project "3_models\TinyTransformer\LICENSE") -Destination (Join-Path $licensesTarget "Compact_Transformers_LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $project "3_models\VisionZoo\LICENSE-chenyaofo.txt") -Destination (Join-Path $licensesTarget "pytorch-cifar-models_LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $project "3_models\VisionZoo\LICENSE-torchvision.txt") -Destination (Join-Path $licensesTarget "torchvision_LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $project "3_models\GPT2\LICENSE") -Destination (Join-Path $licensesTarget "nanoGPT_LICENSE.txt")
Copy-ResourceTree (Join-Path $project "3_models\GPT2\licenses") (Join-Path $licensesTarget "GPT2_and_WikiText2")

$smokeImage = Join-Path $releaseRoot "release_smoke.png"
python (Join-Path $platform "scripts\verify_frozen_app.py") `
    (Join-Path $appTarget "NeuroCIM.exe") $smokeImage --flavor $Flavor
if ($LASTEXITCODE -ne 0) { throw "Frozen application smoke test failed" }
if ($Flavor -eq "GPU") {
    Copy-Item -LiteralPath (Join-Path $releaseRoot "release_gpu_smoke.json") -Destination (Join-Path $portable "gpu_build_diagnostics.json") -Force
} else {
    Copy-Item -LiteralPath (Join-Path $releaseRoot "release_cpu_smoke.json") -Destination (Join-Path $portable "cpu_build_diagnostics.json") -Force
}

$gpuDiagnosticsLauncher = @'
@echo off
setlocal
set "OUTPUT=%~dp0gpu_diagnostics.json"
"%~dp0NeuroCIM\NeuroCIM.exe" --gpu-smoke "%OUTPUT%"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo GPU check passed. Details: "%OUTPUT%"
) else (
  echo GPU check failed. NeuroCIM can still run with device set to auto or cpu.
  echo Details: "%OUTPUT%"
)
pause
exit /b %RESULT%
'@
if ($Flavor -eq "GPU") {
    [System.IO.File]::WriteAllText(
        (Join-Path $portable "GPU-Diagnostics.bat"),
        $gpuDiagnosticsLauncher,
        [System.Text.Encoding]::ASCII
    )
}

$manifestFiles = Get-ChildItem -LiteralPath $portable -Recurse -File
$manifest = [ordered]@{
    format_version = 1
    product = "NeuroCIM DRAM CIM Simulation Platform"
    platform = if ($Flavor -eq "GPU") { "Windows x64 (CPU fallback + NVIDIA CUDA)" } else { "Windows x64 CPU" }
    built_at = (Get-Date).ToString("o")
    python = (python --version 2>&1 | Out-String).Trim()
    file_count = $manifestFiles.Count
    total_bytes = ($manifestFiles | Measure-Object Length -Sum).Sum
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $appTarget "NeuroCIM.exe")).Hash.ToLowerInvariant()
    external_assets = @()
    bundled_assets = @("ten model checkpoints", "MNIST", "CIFAR-10", "WikiText-2 raw-v1", "device measurement CSV")
    neurosim_license = "CC BY-NC 4.0"
    runtime_validation = if ($Flavor -eq "GPU") {
        Get-Content -LiteralPath (Join-Path $portable "gpu_build_diagnostics.json") -Raw | ConvertFrom-Json
    } else {
        Get-Content -LiteralPath (Join-Path $portable "cpu_build_diagnostics.json") -Raw | ConvertFrom-Json
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $portable "release_manifest.json") -Encoding UTF8

$zipPath = Join-Path $releaseRoot "NeuroCIM-Windows-x64.zip"
if (-not $SkipZip) {
    python (Join-Path $platform "scripts\create_release_zip.py") $portable $zipPath
    if ($LASTEXITCODE -ne 0) { throw "Release ZIP creation failed" }
}

$hashLines = @()
$hashLines += "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $appTarget "NeuroCIM.exe")).Hash.ToLowerInvariant(), "NeuroCIM-Windows-x64/NeuroCIM/NeuroCIM.exe"
if (Test-Path -LiteralPath $zipPath) {
    $hashLines += "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant(), "NeuroCIM-Windows-x64.zip"
}
$hashLines | Set-Content -LiteralPath (Join-Path $releaseRoot "SHA256SUMS.txt") -Encoding ASCII

if (Test-Path -LiteralPath $buildRoot) { Remove-Item -LiteralPath $buildRoot -Recurse -Force }

$size = (Get-ChildItem -LiteralPath $portable -Recurse -File | Measure-Object Length -Sum).Sum
Write-Output ("Release ready: {0}" -f $portable)
Write-Output ("Files: {0}; size: {1:N2} GB" -f ((Get-ChildItem -LiteralPath $portable -Recurse -File).Count), ($size / 1GB))
if (Test-Path -LiteralPath $zipPath) { Write-Output ("ZIP: {0}" -f $zipPath) }
