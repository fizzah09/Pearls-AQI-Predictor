<#
PowerShell wrapper to run the daily_store_features.py script.
Loads a simple .env file (KEY=VALUE) from the repo root if present,
sets env vars for the current process, and runs the Python script while
redirecting output to a log file.

Usage:
  ./scripts/run_daily_store.ps1 [-DaysBack 1] [-LogDir .\logs] [-TimeStamp (Get-Date -Format o)]
#>

param(
    [int]$DaysBack = 1,
    [string]$LogDir = ".\logs",
    [string]$TimeStamp = $(Get-Date -Format "yyyyMMdd_HHmmss")
)

Set-StrictMode -Version Latest
Push-Location $PSScriptRoot
$repoRoot = (Resolve-Path "..\").ProviderPath
Pop-Location

# Load .env if present
$envPath = Join-Path $repoRoot ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        if ($line -match "^([^=]+)=(.*)$") {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim().Trim('"').Trim("'")
            if (-not [string]::IsNullOrEmpty($k)) { $env:$k = $v }
        }
    }
}

# Ensure log dir
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$logFile = Join-Path $LogDir ("daily_store_$TimeStamp.log")

Write-Output "Running daily_store_features.py; log -> $logFile"

# Optional: adjust path to Python executable if needed
$python = "python"

# Set DAYS_BACK env for script
$env:DAYS_BACK = $DaysBack.ToString()

# Run script and capture output
& $python "$repoRoot\scripts\daily_store_features.py" *>&1 | Tee-Object -FilePath $logFile

if ($LASTEXITCODE -eq 0) { Write-Output "daily_store completed successfully" } else { Write-Error "daily_store failed (exit $LASTEXITCODE)" }
