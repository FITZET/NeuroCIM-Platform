param(
    [ValidateSet("CPU", "GPU")]
    [Parameter(Mandatory = $true)]
    [string]$Flavor,
    [ValidatePattern("^\d+\.\d+\.\d+$")]
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$ClientId = $env:NEUROCIM_GITHUB_CLIENT_ID,
    [switch]$SkipTests,
    [switch]$ReuseFrozen
)

$ErrorActionPreference = "Stop"
$platform = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$releaseRoot = Join-Path $platform "release"
$portable = Join-Path $releaseRoot "NeuroCIM-Windows-x64"
$artifactRoot = Join-Path $releaseRoot ("artifacts\" + $Flavor.ToLowerInvariant())
$iss = Join-Path $platform "packaging\neurocim-installer.iss"
if (-not $ClientId) {
    throw "Set NEUROCIM_GITHUB_CLIENT_ID or pass -ClientId so authenticated updates can work."
}
$releaseConfig = Join-Path $platform "neurocim_opt\release_config.py"
$versionModule = Join-Path $platform "neurocim_opt\version.py"
$originalReleaseConfig = [System.IO.File]::ReadAllText($releaseConfig)
$originalVersionModule = [System.IO.File]::ReadAllText($versionModule)
$originalExpectedVersion = $env:NEUROCIM_RELEASE_EXPECTED_VERSION
try {
    [System.IO.File]::WriteAllText(
        $versionModule,
        "`"`"`"Application version shared by the GUI, installer and update client.`"`"`"`r`n`r`n__version__ = `"$Version`"`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    $env:NEUROCIM_RELEASE_EXPECTED_VERSION = $Version
    python (Join-Path $platform "scripts\configure_release.py") --client-id $ClientId
    if ($LASTEXITCODE -ne 0) { throw "Release configuration failed" }

    $torchFlavor = python -c "import torch; print('GPU' if torch.version.cuda else 'CPU')"
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect the installed PyTorch build" }
    $torchFlavor = ($torchFlavor | Out-String).Trim()
    if ($torchFlavor -ne $Flavor) {
        throw "This environment contains the $torchFlavor PyTorch build; use a $Flavor build environment."
    }
    if ($Flavor -eq "GPU") {
        # GitHub-hosted Windows runners do not expose an NVIDIA device. This
        # gate verifies the CUDA wheel and torch_cuda.dll without claiming that
        # real CUDA execution occurred; run gpu_release_smoke.py without
        # --build-only on a local/self-hosted NVIDIA runner before publication.
        python (Join-Path $platform "scripts\gpu_release_smoke.py") --build-only
        if ($LASTEXITCODE -ne 0) { throw "GPU PyTorch build/DLL validation failed" }
    }

    $buildArgs = @{
        SkipTests = [bool]$SkipTests
        SkipZip = $true
        ReuseFrozen = [bool]$ReuseFrozen
    }
    & (Join-Path $platform "scripts\build_release.ps1") @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "Portable release build failed" }
}
finally {
    [System.IO.File]::WriteAllText(
        $releaseConfig,
        $originalReleaseConfig,
        [System.Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::WriteAllText(
        $versionModule,
        $originalVersionModule,
        [System.Text.UTF8Encoding]::new($false)
    )
    $env:NEUROCIM_RELEASE_EXPECTED_VERSION = $originalExpectedVersion
}

if ($Flavor -eq "GPU") {
    $bundledCudaDll = Join-Path $portable "NeuroCIM\_internal\torch\lib\torch_cuda.dll"
    if (-not (Test-Path -LiteralPath $bundledCudaDll -PathType Leaf)) {
        throw "Frozen GPU application is missing torch_cuda.dll: $bundledCudaDll"
    }
    Write-Output "Frozen GPU CUDA DLL verified: $bundledCudaDll"
}

$manifestPath = Join-Path $portable "release_manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifest | Add-Member -NotePropertyName version -NotePropertyValue $Version -Force
$manifest | Add-Member -NotePropertyName flavor -NotePropertyValue $Flavor.ToLowerInvariant() -Force
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$isccCandidates = @(
    (Get-Command iscc.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$iscc = $isccCandidates | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup 6 is required. Install it with: winget install JRSoftware.InnoSetup"
}

New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
Get-ChildItem -LiteralPath $artifactRoot -File -ErrorAction SilentlyContinue | Remove-Item -Force
& $iscc "/DAppVersion=$Version" "/DBuildFlavor=$Flavor" "/DSourceRoot=$portable" "/DOutputRoot=$artifactRoot" $iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed" }

python (Join-Path $platform "scripts\generate_update_manifest.py") `
    --version $Version --flavor $Flavor --artifact-dir $artifactRoot
if ($LASTEXITCODE -ne 0) { throw "Update manifest generation failed" }

Write-Output "Installer artifacts ready: $artifactRoot"

