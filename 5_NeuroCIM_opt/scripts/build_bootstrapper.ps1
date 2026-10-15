param([string]$ClientId = $env:NEUROCIM_GITHUB_CLIENT_ID)

$ErrorActionPreference = "Stop"
$platform = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $ClientId) { throw "Set NEUROCIM_GITHUB_CLIENT_ID or pass -ClientId" }
Push-Location $platform
try {
    python scripts\configure_release.py --client-id $ClientId
    python -m PyInstaller --noconfirm --clean --distpath release\bootstrapper --workpath release\.bootstrapper-build packaging\bootstrapper.spec
    if ($LASTEXITCODE -ne 0) { throw "Bootstrapper build failed" }
} finally {
    Pop-Location
}
Write-Output (Join-Path $platform "release\bootstrapper\NeuroCIM-Installer.exe")
