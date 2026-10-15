param()

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is not installed."
}

gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "Please sign in with 'gh auth login' first."
}

Write-Host "Paste the fine-grained GitHub token. Input is hidden, then press Enter."
$secureToken = Read-Host -AsSecureString
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "Token cannot be empty."
    }

    $token | gh secret set DISTRIBUTION_TOKEN --repo FITZET/NeuroCIM-Platform
    if ($LASTEXITCODE -ne 0) { throw "Could not set DISTRIBUTION_TOKEN." }

    $token | gh secret set ACCESS_RELEASE_TOKEN --repo FITZET/NeuroCIM-Platform
    if ($LASTEXITCODE -ne 0) { throw "Could not set ACCESS_RELEASE_TOKEN." }

    $token | gh secret set DISTRIBUTION_ADMIN_TOKEN --repo FITZET/NeuroCIM-Access
    if ($LASTEXITCODE -ne 0) { throw "Could not set DISTRIBUTION_ADMIN_TOKEN." }
}
finally {
    $token = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
}

Write-Host "Configured secrets:"
gh secret list --repo FITZET/NeuroCIM-Platform
gh secret list --repo FITZET/NeuroCIM-Access
Write-Host "Done. Close this window and tell Codex that configuration is complete."
