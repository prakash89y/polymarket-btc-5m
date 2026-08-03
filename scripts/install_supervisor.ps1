<#
.SYNOPSIS
    Install the pmbtc collector supervisor as a Windows Scheduled Task.

.DESCRIPTION
    The collector has died twice, both times because the host shut down and
    nothing brought it back (Windows event VSS 8193, hr=0x8007045b, "A system
    shutdown is in progress"). This registers `pmbtc supervise` to run at boot
    and at logon, so collection resumes without anyone being present.

    A Scheduled Task is used rather than a service because the collector runs as
    a normal user process against a user-owned venv and data directory, and
    because it needs no elevation. NSSM would add a dependency for no benefit.

    Duplicate protection does not rely on the task definition: the supervisor
    takes a machine-wide named mutex, so even if the task fires twice the second
    invocation exits without starting a collector.

.PARAMETER TaskName
    Scheduled Task name. Default: pmbtc-collector.

.PARAMETER ProjectRoot
    Repository root. Defaults to the parent of this script's directory.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_supervisor.ps1
#>
[CmdletBinding()]
param(
    [string]$TaskName = "pmbtc-collector",
    [string]$ProjectRoot
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


$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found at $python. Create the venv first: python -m venv .venv"
}

# -m pmbtc.cli rather than pmbtc.exe: the console script launcher spawns a
# second process, so the PID the supervisor holds would not be the PID writing
# the heartbeat, and the identity check could never be exact.
$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-m pmbtc.cli supervise" `
    -WorkingDirectory $ProjectRoot

# An at-startup trigger is a machine-scope registration and Windows requires
# elevation for it. Without admin we fall back to a logon trigger, which still
# recovers collection after a reboot as soon as the user signs in - but the gap
# between boot and logon is uncovered, so the elevated install is preferred.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
   ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($isAdmin) {
    $triggers = @(
        (New-ScheduledTaskTrigger -AtStartup),
        (New-ScheduledTaskTrigger -AtLogOn)
    )
} else {
    Write-Host "Not running elevated - registering a LOGON trigger only." -ForegroundColor Yellow
    Write-Host "  For true at-boot start, re-run this script from an elevated PowerShell:" -ForegroundColor Yellow
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\install_supervisor.ps1" -ForegroundColor Yellow
    $triggers = @( (New-ScheduledTaskTrigger -AtLogOn) )
}

# RestartCount/RestartInterval are the belt to the supervisor's braces: they
# cover the supervisor process itself dying, which its own restart loop cannot.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew

# Highest privileges are deliberately NOT requested: the collector reads public
# market data and writes into the project directory. It needs nothing more.
# S4U lets the task run without a stored password but is itself an elevated
# registration, so the unelevated path uses an interactive principal.
if ($isAdmin) {
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
} else {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive -RunLevel Limited
}

# --------------------------------------------------------------------------- #
# Registration. Scheduled Task first; Startup-folder shortcut as the fallback.
#
# Task Scheduler refuses registration outright on a locked-down machine ("Access
# is denied" from both Register-ScheduledTask and schtasks.exe, even for a
# logon-only task). The Startup folder is user-writable, needs no privilege, and
# still restores collection after a reboot at sign-in - so the supervisor gets
# installed either way rather than the install simply failing.
# --------------------------------------------------------------------------- #
function Install-StartupShortcut {
    param([string]$Python, [string]$Root)
    $startup  = [Environment]::GetFolderPath('Startup')
    $linkPath = Join-Path $startup "pmbtc-collector.lnk"
    $shell    = New-Object -ComObject WScript.Shell
    $link     = $shell.CreateShortcut($linkPath)
    $link.TargetPath       = $Python
    $link.Arguments        = "-m pmbtc.cli supervise"
    $link.WorkingDirectory = $Root
    $link.WindowStyle      = 7          # minimised
    $link.Description      = "pmbtc collector supervisor"
    $link.Save()
    return $linkPath
}

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Task '$TaskName' already exists - replacing it." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$registered = $false
try {
    Register-ScheduledTask -TaskName $TaskName `
        -Action $action -Trigger $triggers -Settings $settings -Principal $principal `
        -Description "Supervises the pmbtc BTC 5-minute collector: starts at boot, restarts on exit, refuses duplicates." `
        -ErrorAction Stop | Out-Null
    $registered = $true
    Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
} catch {
    Write-Host "Task Scheduler refused registration: $($_.Exception.Message)" -ForegroundColor Yellow
    $link = Install-StartupShortcut -Python $python -Root $ProjectRoot
    Write-Host "Installed Startup-folder fallback instead:" -ForegroundColor Green
    Write-Host "  $link"
    Write-Host "  This starts the supervisor at sign-in. For start-at-boot (before" -ForegroundColor Yellow
    Write-Host "  any user logs in), re-run this script from an elevated PowerShell." -ForegroundColor Yellow
}
Write-Host "  python : $python"
Write-Host "  workdir: $ProjectRoot"
Write-Host ("  trigger: " + $(if ($isAdmin) { "at startup, at logon" } else { "at logon only (unelevated install)" }))
Write-Host ""
Write-Host "Starting it now so collection does not wait for a reboot..." -ForegroundColor Cyan
if ($registered) {
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 8
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "  last run   : $($info.LastRunTime)"
    Write-Host "  last result: $($info.LastTaskResult)"
} else {
    # The supervisor's own lock makes this safe even if one is already running:
    # the duplicate exits with code 2 rather than starting a second collector.
    Start-Process -FilePath $python -ArgumentList "-m","pmbtc.cli","supervise" `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 8
    Write-Host "  supervisor started in the background."
}
Write-Host ""
Write-Host "Verify with: powershell -File scripts\verify_supervisor.ps1" -ForegroundColor Cyan
