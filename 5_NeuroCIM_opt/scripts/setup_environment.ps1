param(
    [string]$Python = "py -3.11",
    [string]$Environment = ""
)

$ErrorActionPreference = "Stop"
$platformRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Environment) {
    $Environment = Join-Path $platformRoot ".venv"
}

$pythonParts = $Python -split " ", 2
if ($pythonParts.Count -eq 2) {
    & $pythonParts[0] $pythonParts[1] -m venv $Environment
} else {
    & $Python -m venv $Environment
}
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required to create the environment" }

$venvPython = Join-Path $Environment "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $platformRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Write-Output "Environment ready: $Environment"
Write-Output "Activate with: $Environment\Scripts\Activate.ps1"
Write-Output "Configure datasets with NEUROCIM_DATA_ROOT or %APPDATA%\NeuroCIM\config.json"
