$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "outputs\baseline_$ts"
$logOut = "outputs\baseline_$ts.log"
$logErr = "outputs\baseline_$ts.err"
$pidFile = "outputs\baseline_$ts.pid"

$proc = Start-Process `
    -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList @(
        "scripts\run_baseline.py",
        "--output", $runDir,
        "--max-new-tokens", "768",
        "--temperature", "0.0"
    ) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File -Encoding ascii $pidFile

Write-Host "Started PID $($proc.Id)"
Write-Host "Run dir : $runDir"
Write-Host "Log     : $logOut"
Write-Host "Err     : $logErr"
Write-Host "PID file: $pidFile"
Write-Host ""
Write-Host "Tail log : Get-Content $logOut -Wait -Tail 20"
Write-Host "Check    : Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue"
Write-Host "Kill     : Stop-Process -Id (Get-Content $pidFile)"
