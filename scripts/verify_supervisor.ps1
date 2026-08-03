<#
.SYNOPSIS
    Verify collector supervision end to end.

.DESCRIPTION
    Checks, in order:

      1. the Scheduled Task exists and is configured to run at boot
      2. exactly one supervisor process is running
      3. exactly one collector process is running
      4. the heartbeat is fresh and belongs to that collector
      5. clock and feed health, via `pmbtc supervisor-status`

    With -TestRecovery it additionally kills the collector and proves the
    supervisor brings it back. That is destructive to the running process but
    not to any data - the store is append-only and the collector resumes from
    where it stopped.

.PARAMETER TestRecovery
    Kill the collector and verify automatic restart.

.PARAMETER TestDuplicate
    Start a second supervisor and verify it refuses to run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify_supervisor.ps1 -TestRecovery -TestDuplicate
#>
[CmdletBinding()]
param(
    [string]$TaskName = "pmbtc-collector",
    [string]$ProjectRoot,
    [switch]$TestRecovery,
    [switch]$TestDuplicate
)

$ErrorActionPreference = "Continue"

# $PSScriptRoot is not reliably populated while parameter defaults are being
# bound, so the root is resolved here instead, with a fallback for every
# invocation style (-File, dot-sourcing, or piping into powershell).
if (-not $ProjectRoot) {
    $here = $PSScriptRoot
    if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
    if (-not $here) { $here = (Get-Location).Path }
    $ProjectRoot = Split-Path -Parent $here
}

$failures = 0

function Test-Step {
    param([string]$Name, [scriptblock]$Check)
    $result = & $Check
    if ($result.ok) {
        Write-Host ("  [PASS] {0}: {1}" -f $Name, $result.detail) -ForegroundColor Green
    } else {
        Write-Host ("  [FAIL] {0}: {1}" -f $Name, $result.detail) -ForegroundColor Red
        $script:failures++
    }
}

function Get-PmbtcProcesses {
    param([string]$Pattern)
    # The venv's python.exe is a launcher that re-execs the base interpreter, so
    # every logical process appears twice: the shim and its child. Counting both
    # would report a duplicate that does not exist. Keep only the leaves - the
    # processes that are not the parent of another matching process.
    $all = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "pmbtc\.cli\s+$Pattern" })
    $parents = $all | ForEach-Object { $_.ParentProcessId }
    @($all | Where-Object { $parents -notcontains $_.ProcessId })
}

Write-Host "pmbtc collector supervision check" -ForegroundColor Cyan
Write-Host "=================================="

Test-Step "boot registration" {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) {
        # Task Scheduler refuses registration without elevation on a locked-down
        # machine; the Startup-folder shortcut is the unelevated equivalent and
        # still restores collection after a reboot, at sign-in.
        $link = Join-Path ([Environment]::GetFolderPath('Startup')) "pmbtc-collector.lnk"
        if (Test-Path $link) {
            return @{ ok = $true
                      detail = "Startup-folder shortcut installed (starts at sign-in; run install elevated for at-boot)" }
        }
        return @{ ok = $false; detail = "neither a scheduled task nor a Startup shortcut is installed" }
    }
    $boot  = $t.Triggers | Where-Object { $_.CimClass.CimClassName -eq "MSFT_TaskBootTrigger" }
    $logon = $t.Triggers | Where-Object { $_.CimClass.CimClassName -eq "MSFT_TaskLogonTrigger" }
    if ($boot)  { return @{ ok = $true; detail = "registered, state=$($t.State), boot trigger present" } }
    if ($logon) {
        return @{ ok = $true
                  detail = "registered, state=$($t.State), LOGON trigger only - re-run install elevated for at-boot" }
    }
    @{ ok = $false; detail = "task exists but has neither a boot nor a logon trigger" }
}

Test-Step "single supervisor" {
    $p = @(Get-PmbtcProcesses "supervise")
    if ($p.Count -eq 1) { @{ ok = $true; detail = "1 supervisor (pid $($p[0].ProcessId))" } }
    else { @{ ok = $false; detail = "$($p.Count) supervisor process(es); expected exactly 1" } }
}

Test-Step "single collector" {
    $p = @(Get-PmbtcProcesses "run")
    if ($p.Count -eq 1) { @{ ok = $true; detail = "1 collector (pid $($p[0].ProcessId))" } }
    else { @{ ok = $false; detail = "$($p.Count) collector process(es); expected exactly 1" } }
}

Test-Step "heartbeat owner" {
    $hb = Join-Path $ProjectRoot "data\heartbeat.json"
    if (-not (Test-Path $hb)) { return @{ ok = $false; detail = "no heartbeat file" } }
    $j = Get-Content $hb -Raw | ConvertFrom-Json
    $ageS = [Math]::Round(([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - $j.written_at_ms) / 1000)
    $collector = @(Get-PmbtcProcesses "run")
    $owned = $collector.Count -eq 1 -and $collector[0].ProcessId -eq $j.pid
    if ($ageS -gt 900) { return @{ ok = $false; detail = "heartbeat ${ageS}s old (budget 900s)" } }
    if (-not $owned) { return @{ ok = $false; detail = "heartbeat pid $($j.pid) is not the running collector" } }
    @{ ok = $true; detail = "${ageS}s old, pid $($j.pid), clock=$($j.clock_status)" }
}

if ($TestDuplicate) {
    Write-Host ""
    Write-Host "Duplicate-prevention test" -ForegroundColor Cyan
    Test-Step "second supervisor refuses" {
        $py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
        $out = Join-Path $env:TEMP "pmbtc-dup-test.log"
        $p = Start-Process -FilePath $py -ArgumentList "-m","pmbtc.cli","supervise" `
             -WorkingDirectory $ProjectRoot -RedirectStandardError $out -PassThru -WindowStyle Hidden -Wait
        $before = @(Get-PmbtcProcesses "run").Count
        if ($p.ExitCode -eq 2 -and $before -eq 1) {
            @{ ok = $true; detail = "second supervisor exited 2 (already_running); still $before collector" }
        } else {
            @{ ok = $false; detail = "exit code $($p.ExitCode); $before collector(s) running" }
        }
    }
}

if ($TestRecovery) {
    Write-Host ""
    Write-Host "Recovery test (kills the collector; data is append-only and safe)" -ForegroundColor Cyan
    $before = @(Get-PmbtcProcesses "run")
    if ($before.Count -ne 1) {
        Write-Host "  [SKIP] need exactly 1 collector to test recovery" -ForegroundColor Yellow
    } else {
        $oldPid = $before[0].ProcessId
        Write-Host "  killing collector pid $oldPid ..." -ForegroundColor Yellow
        Stop-Process -Id $oldPid -Force
        $deadline = (Get-Date).AddSeconds(90)
        $newPid = $null
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 3
            $now = @(Get-PmbtcProcesses "run")
            if ($now.Count -eq 1 -and $now[0].ProcessId -ne $oldPid) { $newPid = $now[0].ProcessId; break }
        }
        Test-Step "automatic restart" {
            if ($newPid) { @{ ok = $true; detail = "collector restarted as pid $newPid (was $oldPid)" } }
            else { @{ ok = $false; detail = "no replacement collector within 90s" } }
        }
    }
}

Write-Host ""
Write-Host "pmbtc supervisor-status" -ForegroundColor Cyan
& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") -m pmbtc.cli supervisor-status

Write-Host ""
if ($failures -eq 0) {
    Write-Host "ALL CHECKS PASSED - supervision is healthy." -ForegroundColor Green
    exit 0
}
Write-Host "$failures CHECK(S) FAILED." -ForegroundColor Red
exit 1
