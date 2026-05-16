# Launch a long-running Python job via Windows Task Scheduler (S4U principal).
# Survives SSH disconnect, console close, and user logoff.
#
# Usage:
#   .\scripts\run_scheduled.ps1 <tag> <python_args...>
# Example:
#   .\scripts\run_scheduled.ps1 baseline scripts\run_baseline.py `
#       --output outputs\baseline_run --max-new-tokens 768 --temperature 0.0
#
# Requirements:
#   - Must be invoked from an ELEVATED PowerShell (admin) the first time, because
#     -LogonType S4U requires SeTcbPrivilege. After registration the task runs as
#     the current user without needing a password or an active session.
#
# Outputs (under outputs\):
#   <tag>_<ts>.log    stdout
#   <tag>_<ts>.err    stderr
#   <tag>_<ts>.pid    PID of the wrapper PowerShell (Stop-ScheduledTask kills the tree)
#   <tag>_<ts>.task   the scheduled task name (use to stop / unregister)
#   <tag>_<ts>.wrap.ps1  generated wrapper that pins cwd, runs python, redirects IO
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Tag,

    [Parameter(Mandatory = $true, Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$base = "${Tag}_${ts}"
$taskName = "ARC_${base}"

$outDir = Join-Path $repoRoot "outputs"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$logOut      = Join-Path $outDir "${base}.log"
$logErr      = Join-Path $outDir "${base}.err"
$pidFile     = Join-Path $outDir "${base}.pid"
$taskFile    = Join-Path $outDir "${base}.task"
$wrapperFile = Join-Path $outDir "${base}.wrap.ps1"
$python      = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python. Did you create .venv? See CLAUDE.md Environment section."
    exit 1
}

# Join python args; quote any token that contains whitespace.
$pyArgsJoined = ($PyArgs | ForEach-Object {
    if ($_ -match '\s') { "`"$_`"" } else { $_ }
}) -join ' '

# Wrapper PowerShell: records its own PID, pins cwd, runs python with redirects.
$wrapper = @"
`$PID | Out-File -Encoding ascii '$pidFile'
Set-Location '$repoRoot'
& '$python' $pyArgsJoined 1>'$logOut' 2>'$logErr'
"@
$wrapper | Out-File -Encoding utf8 $wrapperFile

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapperFile`"" `
    -WorkingDirectory $repoRoot

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

try {
    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType S4U `
        -RunLevel Highest
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
} catch {
    Write-Error @"
Failed to register S4U scheduled task: $_

S4U requires admin (SeTcbPrivilege) to register. Without S4U the task would die
when your SSH session ends, defeating the purpose.

Fix: open an ELEVATED PowerShell (Run as Administrator) and retry. After the
task is registered once, the python process itself runs as your normal user
without needing the elevated shell again.

If you cannot get admin, edit this script to switch -LogonType to Interactive
and accept that the task dies on SSH disconnect.
"@
    exit 1
}

$taskName | Out-File -Encoding ascii $taskFile

Start-ScheduledTask -TaskName $taskName

Start-Sleep -Seconds 2
$info = Get-ScheduledTaskInfo -TaskName $taskName

Write-Host ""
Write-Host "Scheduled task : $taskName"
Write-Host "Last run time  : $($info.LastRunTime)"
Write-Host "Last task code : $($info.LastTaskResult)  (0 / 267009 = OK)"
Write-Host "Log out        : $logOut"
Write-Host "Log err        : $logErr"
Write-Host "PID file       : $pidFile"
Write-Host "Wrapper script : $wrapperFile"
Write-Host ""
Write-Host "Tail log  : Get-Content '$logOut' -Wait -Tail 20"
Write-Host "Status    : Get-ScheduledTaskInfo -TaskName '$taskName'"
Write-Host "Stop      : Stop-ScheduledTask -TaskName '$taskName'"
Write-Host "Cleanup   : Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
