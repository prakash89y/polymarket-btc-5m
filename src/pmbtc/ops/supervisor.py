"""Collector supervision — the last operational blocker.

The collector was never unreliable in the way it looked. It died twice, and both
times the cause was the same and was not a defect in it: the host shut down.
Windows event ``VSS 8193, hr=0x8007045b`` ("A system shutdown is in progress")
was logged at 14:01:35 UTC, matching the collector's final line at 14:01:23 UTC,
and the machine booted again eighteen hours later with nothing to bring
collection back. A process launched by hand from a shell survives exactly as
long as the machine does.

So this module owns three jobs and deliberately no others:

**Start on boot.** A Scheduled Task runs ``pmbtc supervise`` at startup; see
``scripts/install_supervisor.ps1``.

**Restart on exit.** Bounded-exponential backoff, and a *hung* collector is
restarted too — a process that is running but no longer writing a heartbeat is
the failure that a PID check alone would miss, and it is the one the collector's
own docs call out as worth alerting on.

**Refuse to be the second instance.** Two collectors writing one append-only
store is a data-integrity problem, not a performance one. The lock is an OS
primitive (a named mutex on Windows, ``flock`` elsewhere) so it is released by
the kernel when the holder dies, however it dies — a PID file left behind by a
killed process would block every future start.

Everything else is reused: health rules come from :mod:`pmbtc.ops.heartbeat`,
alerting from :mod:`pmbtc.ops.alerts`, and the restart policy from the existing
``service.*`` configuration. No new config section, no new dependency, no second
implementation of any alert condition.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pmbtc.config import Config
from pmbtc.logging_setup import get_logger
from pmbtc.metrics import METRICS
from pmbtc.ops.alerts import Alert, AlertKind, AlertState, Severity, alert_state_path, dispatch
from pmbtc.ops.heartbeat import Heartbeat, read_heartbeat
from pmbtc.utils.timeutils import utc_now_ms

log = get_logger("pmbtc.ops.supervisor")

supervisor_restarts = METRICS.counter(
    "pmbtc_supervisor_restarts_total", "Collector restarts performed by the supervisor."
)

#: Identity of the single permitted collector on this machine.
LOCK_NAME = "pmbtc-collector"

#: Environment variable carrying the supervisor run token to the collector.
#: Named here so the writer (the service) and the reader (this module)
#: cannot drift apart.
TOKEN_ENV = "PMBTC_SUPERVISOR_TOKEN"

#: A child that dies sooner than this is failing to start, not crashing after a
#: good run. Only those increment the consecutive-failure counter, so a
#: collector that runs for hours and then dies gets a clean restart rather than
#: inheriting the history of a bad patch weeks earlier.
HEALTHY_RUN_SECONDS = 300.0


# --------------------------------------------------------------------------- #
# Single instance
# --------------------------------------------------------------------------- #
class InstanceLock:
    """Whole-machine mutual exclusion, released by the OS on process death.

    Windows uses a named kernel mutex and POSIX an advisory ``flock``. Both are
    held by the *process*, so a supervisor that is killed -9 releases its claim
    immediately. A PID file would not: it would survive the death it is meant to
    describe and lock out every subsequent start.
    """

    def __init__(self, name: str = LOCK_NAME, directory: Path | None = None) -> None:
        self.name = name
        self.directory = directory
        self._handle: Any = None
        self._file: Any = None

    def acquire(self) -> bool:
        if os.name == "nt":
            return self._acquire_windows()
        return self._acquire_posix()

    def _acquire_windows(self) -> bool:
        import ctypes
        from ctypes import wintypes

        ERROR_ALREADY_EXISTS = 183
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        # "Global\\" would need elevation; the session namespace is the right
        # scope for a per-user collector.
        handle = kernel32.CreateMutexW(None, True, f"Local\\{self.name}")
        if not handle:
            return False
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def _acquire_posix(self) -> bool:
        # fcntl is POSIX-only; this branch never runs on Windows, but mypy
        # type-checks both branches on whichever platform it is invoked from.
        import fcntl

        directory = self.directory or Path(os.environ.get("TMPDIR", "/tmp"))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.lock"
        handle = path.open("w")
        try:
            exclusive_nonblocking = fcntl.LOCK_EX | fcntl.LOCK_NB  # type: ignore[attr-defined]
            fcntl.flock(handle.fileno(), exclusive_nonblocking)  # type: ignore[attr-defined]
        except OSError:
            handle.close()
            return False
        handle.write(str(os.getpid()))
        handle.flush()
        self._file = handle
        return True

    def release(self) -> None:
        if self._handle is not None and os.name == "nt":
            import ctypes

            ctypes.WinDLL("kernel32").CloseHandle(self._handle)
            self._handle = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> InstanceLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    healthy: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'OK' if self.healthy else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class HealthReport:
    """The five things the supervisor verifies on every tick."""

    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return bool(self.checks) and all(c.healthy for c in self.checks)

    @property
    def failures(self) -> list[HealthCheck]:
        return [c for c in self.checks if not c.healthy]

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "checks": [
                {"name": c.name, "healthy": c.healthy, "detail": c.detail}
                for c in self.checks
            ],
        }

    def render(self) -> str:
        return "\n".join(str(c) for c in self.checks)


class ProcessHandle(Protocol):
    """The subset of :class:`subprocess.Popen` the supervisor needs."""

    pid: int

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def collector_command(config_path: Path | None = None) -> list[str]:
    """Launch the collector as a module, not through the console script.

    ``pmbtc.exe`` is a launcher that spawns a *second* process, so the PID the
    supervisor holds would not be the PID writing the heartbeat and the identity
    check below could never be exact. Invoking the module directly makes the
    child we supervise the child we measure.
    """
    command = [sys.executable, "-m", "pmbtc.cli", "run", "--minutes", "0"]
    if config_path is not None:
        command += ["--config", str(config_path)]
    return command


def check_health(
    config: Config,
    *,
    child: ProcessHandle | None,
    lock_held: bool,
    heartbeat: Heartbeat | None = None,
    now_ms: int | None = None,
    supervisor_token: str = "",
) -> HealthReport:
    """Verify process, instance, heartbeat, clock and feeds.

    The heartbeat staleness rule is taken from the alert engine's own budget
    (``status_interval_seconds * 3``) rather than a second threshold invented
    here — two definitions of "stale" would eventually disagree, and the
    disagreement would surface as an alert nobody can reproduce.
    """
    now = now_ms if now_ms is not None else utc_now_ms()
    beat = heartbeat if heartbeat is not None else read_heartbeat(config)
    report = HealthReport()

    # 1. Single instance
    report.checks.append(
        HealthCheck("single_instance", lock_held, "supervisor holds the machine lock"
                    if lock_held else "another supervisor holds the lock")
    )

    # 2. Collector process
    if child is None:
        report.checks.append(HealthCheck("collector_process", False, "no child process"))
    else:
        code = child.poll()
        report.checks.append(
            HealthCheck(
                "collector_process",
                code is None,
                f"pid {child.pid} running" if code is None else f"pid {child.pid} exited ({code})",
            )
        )

    # 3. Heartbeat freshness
    budget_ms = config.service.status_interval_seconds * 3 * 1000
    if beat is None:
        report.checks.append(HealthCheck("heartbeat", False, "no heartbeat file"))
    else:
        age = beat.age_ms(now)
        report.checks.append(
            HealthCheck(
                "heartbeat",
                age <= budget_ms,
                f"{age / 1000:.0f}s old (budget {budget_ms / 1000:.0f}s)",
            )
        )
        # 3b. The heartbeat must belong to the run we supervise. A stale file
        # from a previous run would otherwise read as perfect health.
        #
        # Matched on a token rather than a PID: the venv's python.exe is a
        # launcher that re-execs the base interpreter, so the PID we hold is the
        # shim's and the heartbeat is written by its child. Measured on this
        # machine: supervised child 16132, heartbeat writer 18556. A token
        # passed through the environment survives every such hop.
        if supervisor_token:
            owned = beat.supervisor_token == supervisor_token
            report.checks.append(
                HealthCheck(
                    "heartbeat_owner",
                    owned,
                    "heartbeat belongs to this supervisor run"
                    if owned
                    else f"heartbeat token {beat.supervisor_token or '<none>'} is not ours",
                )
            )
        elif child is not None:
            # No token: a hand-started collector. Fall back to the PID, which is
            # correct whenever no launcher shim sits in between.
            owned = beat.pid == child.pid
            report.checks.append(
                HealthCheck(
                    "heartbeat_owner",
                    owned,
                    f"heartbeat pid {beat.pid} matches supervised child"
                    if owned
                    else f"heartbeat pid {beat.pid} is not our child {child.pid}",
                )
            )

    # 4. Clock health
    if beat is not None:
        ok = beat.clock_status == "healthy"
        report.checks.append(
            HealthCheck("clock", ok, f"clock is {beat.clock_status}")
        )

    # 5. Feed health
    if beat is not None:
        feeds = beat.feeds or []
        if not feeds:
            report.checks.append(HealthCheck("feeds", True, "no feeds open (between windows)"))
        else:
            # A feed whose *transport* is down is a failure. A feed that is
            # merely quiet is not — that distinction is the whole of v0.12.0.
            down = [
                f.get("name", "?")
                for f in feeds
                if not f.get("transport_connected", f.get("state") == "live")
            ]
            report.checks.append(
                HealthCheck(
                    "feeds",
                    not down,
                    f"{len(feeds)} feed(s) connected" if not down else f"transport down: {down}",
                )
            )
    return report


# --------------------------------------------------------------------------- #
# The supervisor
# --------------------------------------------------------------------------- #
@dataclass
class SupervisorStats:
    started_at_ms: int = 0
    starts: int = 0
    restarts: int = 0
    consecutive_failures: int = 0
    alerts_fired: int = 0
    last_exit_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at_ms": self.started_at_ms,
            "starts": self.starts,
            "restarts": self.restarts,
            "consecutive_failures": self.consecutive_failures,
            "alerts_fired": self.alerts_fired,
            "last_exit_code": self.last_exit_code,
        }


class CollectorSupervisor:
    """Keeps exactly one collector running, and says so when it cannot."""

    def __init__(
        self,
        config: Config,
        *,
        spawn: Callable[[], ProcessHandle] | None = None,
        lock: InstanceLock | None = None,
        check_interval_s: float | None = None,
        startup_grace_s: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.lock = lock if lock is not None else InstanceLock(
            directory=config.resolved_path(config.app.data_dir)
        )
        self.spawn = spawn or (lambda: self._spawn_collector(config_path))
        # Reuses the collector's own liveness cadence rather than adding a knob.
        self.check_interval_s = (
            check_interval_s
            if check_interval_s is not None
            else float(config.service.heartbeat_interval_seconds)
        )
        #: Seconds after a spawn before health is judged. Two heartbeat
        #: intervals: long enough for the new collector to publish its first
        #: heartbeat, short enough that a failure to start is still caught fast.
        self.startup_grace_s = (
            startup_grace_s
            if startup_grace_s is not None
            else float(config.service.heartbeat_interval_seconds) * 2
        )
        self.stats = SupervisorStats()
        self.child: ProcessHandle | None = None
        #: Unique per supervisor run, handed to the collector through the
        #: environment and echoed back in its heartbeat.
        self.token = uuid.uuid4().hex

    def _spawn_collector(self, config_path: Path | None) -> ProcessHandle:
        log_dir = self.config.resolved_path(self.config.app.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / "collector.supervised.out.log").open("a", encoding="utf-8")
        stderr = (log_dir / "collector.supervised.err.log").open("a", encoding="utf-8")
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        # The run token travels in the environment so it survives the venv's
        # python.exe launcher re-execing the base interpreter — the hop that
        # makes PID-based ownership impossible.
        environment = {**os.environ, TOKEN_ENV: self.token}
        return subprocess.Popen(
            collector_command(config_path),
            stdout=stdout,
            stderr=stderr,
            cwd=str(self.config.resolved_path(Path("."))),
            creationflags=creation,
            env=environment,
        )

    # ------------------------------------------------------------------ #
    def backoff_seconds(self, failures: int) -> float:
        base = self.config.service.restart_backoff_s
        return min(base * (2 ** max(0, failures - 1)), 300.0)

    def _raise_restart_alert(self, detail: dict[str, Any]) -> None:
        """Alert through the existing framework — no second implementation."""
        try:
            state = AlertState(alert_state_path(self.config))
            alert = Alert(
                kind=AlertKind.SUPERVISOR_RESTART_FAILING,
                severity=Severity.CRITICAL,
                message=(
                    f"Collector failed to stay up across "
                    f"{self.stats.consecutive_failures} consecutive restarts"
                ),
                detail={"check": "supervisor", **detail},
                raised_at_ms=utc_now_ms(),
            )
            if dispatch(self.config, state, [alert]):
                self.stats.alerts_fired += 1
        except Exception as exc:  # pragma: no cover - alerting must never crash us
            log.warning("supervisor.alert_failed", error=f"{type(exc).__name__}: {exc}")

    def stop_child(self, timeout: float = 20.0) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        self.child.terminate()
        try:
            self.child.wait(timeout=timeout)
        except Exception:
            self.child.kill()

    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        max_restarts: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> int:
        """Supervise until interrupted. Returns a process exit code.

        ``max_restarts`` bounds the loop for tests and for the verification
        script; left unset it runs forever, which is the point.
        """
        if not self.lock.acquire():
            log.error(
                "supervisor.already_running",
                detail="another supervisor holds the lock; refusing to start a second collector",
            )
            return 2

        self.stats.started_at_ms = utc_now_ms()
        log.info(
            "supervisor.started",
            check_interval_s=self.check_interval_s,
            max_consecutive_failures=self.config.service.max_consecutive_errors,
        )
        try:
            while True:
                started = time.monotonic()
                self.child = self.spawn()
                self.stats.starts += 1
                log.info("supervisor.collector_started", pid=self.child.pid)

                exit_code = self._watch(sleep)
                ran_for = time.monotonic() - started
                self.stats.last_exit_code = exit_code

                if ran_for >= HEALTHY_RUN_SECONDS:
                    self.stats.consecutive_failures = 0
                else:
                    self.stats.consecutive_failures += 1

                log.warning(
                    "supervisor.collector_exited",
                    exit_code=exit_code,
                    ran_for_s=round(ran_for, 1),
                    consecutive_failures=self.stats.consecutive_failures,
                )

                if self.stats.consecutive_failures >= self.config.service.max_consecutive_errors:
                    self._raise_restart_alert(
                        {
                            "exit_code": exit_code,
                            "ran_for_s": round(ran_for, 1),
                            "consecutive_failures": self.stats.consecutive_failures,
                        }
                    )

                if max_restarts is not None and self.stats.restarts >= max_restarts:
                    return exit_code or 0

                delay = self.backoff_seconds(self.stats.consecutive_failures)
                self.stats.restarts += 1
                supervisor_restarts.inc()
                log.info("supervisor.restarting", in_s=delay)
                sleep(delay)
        except KeyboardInterrupt:  # pragma: no cover - operator interrupt
            log.info("supervisor.interrupted")
            self.stop_child()
            return 0
        finally:
            self.lock.release()

    def _watch(self, sleep: Callable[[float], None]) -> int:
        """Block until the child exits, or until it is hung and we kill it.

        Health is not judged during the startup grace: for the first seconds a
        freshly spawned collector has not yet written a heartbeat, so the file
        on disk still belongs to its predecessor and every check would report a
        failure that resolves itself. Alarming on a known-transient state is how
        an operator learns to ignore the log.
        """
        grace_s = self.startup_grace_s
        started = time.monotonic()
        while True:
            code = self.child.poll() if self.child else 0
            if code is not None:
                return code

            if time.monotonic() - started < grace_s:
                sleep(self.check_interval_s)
                continue

            report = check_health(
                self.config,
                child=self.child,
                lock_held=True,
                supervisor_token=self.token,
            )
            hung = any(c.name == "heartbeat" and not c.healthy for c in report.checks)
            if hung:
                # Running but not collecting. This is the failure a PID check
                # cannot see, and leaving it alone means a process that looks
                # alive while the dataset stops growing.
                log.error(
                    "supervisor.collector_hung",
                    detail=[str(c) for c in report.failures],
                )
                self.stop_child()
                code = self.child.poll() if self.child else None
                # A hung collector that ignored SIGTERM has no exit code of its
                # own; -1 marks "we ended it" distinctly from any code it chose.
                return code if code is not None else -1
            if not report.healthy:
                log.warning("supervisor.degraded", failures=[str(c) for c in report.failures])
            sleep(self.check_interval_s)
