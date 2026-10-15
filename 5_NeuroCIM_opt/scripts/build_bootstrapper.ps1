param([string]$ClientId = $env:NEUROCIM_GITHUB_CLIENT_ID)

$ErrorActionPreference = "Stop"
$platform = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $ClientId) { throw "Set NEUROCIM_GITHUB_CLIENT_ID or pass -ClientId" }
$releaseConfig = Join-Path $platform "neurocim_opt\release_config.py"
$originalReleaseConfig = [System.IO.File]::ReadAllText($releaseConfig)
Push-Location $platform
try {
    python scripts\configure_release.py --client-id $ClientId
    if ($LASTEXITCODE -ne 0) { throw "Release configuration failed" }
    python -m PyInstaller --noconfirm --clean --distpath release\bootstrapper --workpath release\.bootstrapper-build packaging\bootstrapper.spec
    if ($LASTEXITCODE -ne 0) { throw "Bootstrapper build failed" }
} finally {
    [System.IO.File]::WriteAllText(
        $releaseConfig,
        $originalReleaseConfig,
        [System.Text.UTF8Encoding]::new($false)
    )
    Pop-Location
}
Write-Output (Join-Path $platform "release\bootstrapper\NeuroCIM-Installer.exe")
