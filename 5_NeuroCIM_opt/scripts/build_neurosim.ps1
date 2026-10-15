param(
    [string]$CondaPrefix = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$platformRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $CondaPrefix) {
    $CondaPrefix = Join-Path $platformRoot ".tools\mingw"
}
if (-not $Output) {
    $Output = Join-Path $platformRoot "outputs\neurosim\bin\neurosim.exe"
}

$sourceRoot = Join-Path $platformRoot "vendor\neurosim\Inference_pytorch\NeuroSIM"
$sources = Get-ChildItem -LiteralPath $sourceRoot -Filter "*.cpp" | ForEach-Object { $_.FullName }
if (-not $sources) {
    throw "No NeuroSim C++ sources found under $sourceRoot"
}
if (-not (Test-Path -LiteralPath $CondaPrefix)) {
    conda create -y -p $CondaPrefix m2w64-toolchain
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the isolated MinGW environment" }
}

$outputDirectory = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
conda run -p $CondaPrefix g++ -fopenmp -O3 -std=c++11 -w @sources -o $Output
if ($LASTEXITCODE -ne 0) { throw "NeuroSim compilation failed" }
Write-Output (Resolve-Path $Output).Path
