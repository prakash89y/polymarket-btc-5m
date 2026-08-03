<#
.SYNOPSIS
    Remove the pmbtc collector supervisor Scheduled Task and stop collection.

.DESCRIPTION
    Stops the task, then the supervisor and any collector it owns. Order
    matters: killing the collector first would only prompt the supervisor to
    start a replacement.

    The dataset, archive and logs are never touched. Collection can be resumed
    at any time by re-running install_supervisor.ps1, and the append-only store
    picks up where it left off.

.PARAMETER TaskName
    Scheduled Task name. Default: pmbtc-collector.

.PARAMETER KeepRunning
    Unregister the task but leave the current processes alive, so collection
    continues until the next reboot.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\uninstall_supervisor.ps1
#>
[CmdletBinding()]
param(
    [string]$TaskName = "pmbtc-collector",
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is not reliably populated while parameter defaults are being
# bound, so the root is resolved here instead, with a fallback for every
# invocation style (-File, dot-sourcing, or piping into powershell).
if (-not $ProjectRoot) {
    $here = $PSScriptRoot
    if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
    if (-not $here) { $here = (Get-Location).Path }
    $ProjectRoot = Split-Path -Parent $here
}


$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    if ($task.State -eq "Running") {
        Write-Host "Stopping task '$TaskName'..." -ForegroundColor Yellow
        Stop-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 2
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Unregistered scheduled task '$TaskName'." -ForegroundColor Green
} else {
    Write-Host "No scheduled task named '$TaskName'." -ForegroundColor DarkGray
}

$link = Join-Path ([Environment]::GetFolderPath('Startup')) "pmbtc-collector.lnk"
if (Test-Path $link) {
    Remove-Item $link -Force
    Write-Host "Removed Startup-folder shortcut." -ForegroundColor Green
}

if ($KeepRunning) {
    Write-Host "-KeepRunning set: leaving the running supervisor and collector alone." -ForegroundColor Cyan
    exit 0
}

# Supervisor first, then the collector - the reverse order just triggers a
# restart of the thing we are trying to stop.
$stopped = 0
foreach ($pattern in @("supervise", "cli run")) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "pmbtc\.cli\s+$pattern" } |
        ForEach-Object {
            Write-Host "  stopping pid $($_.ProcessId) ($pattern)" -ForegroundColor Yellow
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    Start-Sleep -Seconds 2
}

if ($stopped -eq 0) {
    Write-Host "No supervisor or collector processes were running." -ForegroundColor DarkGray
} else {
    Write-Host "Stopped $stopped process(es)." -ForegroundColor Green
}
Write-Host ""
Write-Host "Data, archive and logs are untouched. Re-install to resume collection." -ForegroundColor Cyan
