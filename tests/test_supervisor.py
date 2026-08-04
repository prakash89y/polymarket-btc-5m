"""Tests for collector supervision.

The failure being prevented is concrete: the collector died twice, both times
because the host shut down (Windows event VSS 8193, hr=0x8007045b) and nothing
brought it back. These pin the four behaviours that make unattended running
possible — restart on exit, restart when hung, refuse duplicates, and alert when
restarting is not working.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pmbtc.config import Config
from pmbtc.ops.alerts import AlertKind
from pmbtc.ops.supervisor import (
    HEALTHY_RUN_SECONDS,
    CollectorSupervisor,
    InstanceLock,
    check_health,
    collector_command,
)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(app={"base_dir": str(tmp_path)})


def _isolated_lock(config: Config, name: str) -> InstanceLock:
    """A lock name unique to this test.

    The Windows mutex is machine-scoped by design, so a test using the default
    name would contend with a real supervisor running on the same box - and
    that is exactly the behaviour the lock is meant to have.
    """
    return InstanceLock(
        name=f"pmbtc-test-{name}-{os.getpid()}",
        directory=config.resolved_path(config.app.data_dir),
    )


class _FakeProcess:
    """A child that exits after a scripted number of polls."""

    def __init__(self, pid: int = 4321, exit_after_polls: int = 1, code: int = 1) -> None:
        self.pid = pid
        self._polls = 0
        self._exit_after = exit_after_polls
        self._code = code
        self.terminated = False

    def poll(self) -> int | None:
        self._polls += 1
        return self._code if self._polls > self._exit_after else None

    def terminate(self) -> None:
        self.terminated = True
        self._exit_after = -1

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        return self._code


def _write_heartbeat(config: Config, pid: int, *, age_ms: int = 0, clock: str = "healthy",
                     feeds: list | None = None, token: str = "") -> None:
    from pmbtc.utils.timeutils import utc_now_ms

    path = config.resolved_path(config.app.data_dir) / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "written_at_ms": utc_now_ms() - age_ms,
        "pid": pid, "sessions": 1, "snapshots": 10, "labels": 1, "errors": 0,
        "archived_frames": 100, "clock_status": clock,
        "supervisor_token": token,
        "feeds": feeds if feeds is not None else [
            {"name": "clob:x", "state": "live", "transport_connected": True}
        ],
    }), encoding="utf-8")


# --------------------------------------------------------------------------- #
class TestInstanceLock:
    """Requirement 4: the restart mechanism must not create duplicates."""

    def test_second_acquisition_is_refused(self, tmp_path: Path) -> None:
        first = InstanceLock(name="pmbtc-test-lock", directory=tmp_path)
        second = InstanceLock(name="pmbtc-test-lock", directory=tmp_path)
        assert first.acquire()
        try:
            assert not second.acquire(), "a second instance was allowed to start"
        finally:
            first.release()

    def test_the_lock_is_reusable_after_release(self, tmp_path: Path) -> None:
        """A released lock must not block the next start — a stale PID file
        would, which is why an OS primitive is used instead."""
        lock = InstanceLock(name="pmbtc-test-lock2", directory=tmp_path)
        assert lock.acquire()
        lock.release()
        again = InstanceLock(name="pmbtc-test-lock2", directory=tmp_path)
        assert again.acquire()
        again.release()

    def test_supervisor_exits_when_another_holds_the_lock(self, config: Config) -> None:
        holder = InstanceLock(
            name="pmbtc-dup", directory=config.resolved_path(config.app.data_dir)
        )
        assert holder.acquire()
        try:
            spawned: list[int] = []
            supervisor = CollectorSupervisor(
                config,
                spawn=lambda: (spawned.append(1), _FakeProcess())[1],
                lock=InstanceLock(
                    name="pmbtc-dup",
                    directory=config.resolved_path(config.app.data_dir),
                ),
            )
            assert supervisor.run(max_restarts=0) == 2
            assert not spawned, "a duplicate collector was started"
        finally:
            holder.release()


class TestMachineWideLock:
    r"""Regression: the lock must exclude across Windows *sessions*, not just
    within one.

    Production, 2026-08-03: a Scheduled Task (S4U, session 0) and a
    Startup-folder shortcut (session 1) each acquired their own ``Local\``
    named mutex, because that namespace is per-logon-session. Two supervisors
    ran, each killed the other's collector on a foreign heartbeat token, and the
    restart loop reached 24 consecutive failures.
    """

    def test_the_lock_is_not_session_scoped(self) -> None:
        r"""The specific defect: a ``Local\`` mutex is per-session."""
        import inspect

        from pmbtc.ops import supervisor

        source = inspect.getsource(supervisor.InstanceLock)
        # CreateMutexW is the unambiguous marker: every named-mutex namespace on
        # Windows is session- or privilege-scoped. (The string "Local\\" also
        # appears in the docstring explaining the old bug, so it cannot be the
        # test's signal.)
        assert "CreateMutexW" not in source, (
            "named-mutex locking is session-scoped; the lock must key on a path"
        )
        assert "CreateFileW" in source, "expected the exclusive lock-file implementation"

    def test_the_lock_is_keyed_by_an_absolute_path(self, tmp_path: Path) -> None:
        """A path is identical across sessions; a session namespace is not."""
        lock = InstanceLock(name="pmbtc-path", directory=tmp_path)
        path = lock.lock_path()
        assert path.is_absolute()
        assert path.parent == tmp_path

    def test_a_separate_process_cannot_acquire_the_same_lock(
        self, tmp_path: Path
    ) -> None:
        """The real property, proven across a genuine process boundary.

        A second OS process is the closest testable analogue of a second logon
        session: it shares nothing with this one except the filesystem, which is
        exactly what the lock must now key on.
        """
        import subprocess
        import sys
        import textwrap

        held = InstanceLock(name="pmbtc-xproc", directory=tmp_path)
        assert held.acquire()
        try:
            probe = textwrap.dedent(
                f"""
                from pathlib import Path
                from pmbtc.ops.supervisor import InstanceLock
                lock = InstanceLock(name="pmbtc-xproc", directory=Path(r"{tmp_path}"))
                print("ACQUIRED" if lock.acquire() else "REFUSED")
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60
            )
            assert "REFUSED" in result.stdout, (
                f"a second process acquired the lock: {result.stdout!r} {result.stderr!r}"
            )
        finally:
            held.release()

    def test_the_lock_is_reacquirable_after_the_holder_releases(
        self, tmp_path: Path
    ) -> None:
        first = InstanceLock(name="pmbtc-reacq", directory=tmp_path)
        assert first.acquire()
        first.release()
        second = InstanceLock(name="pmbtc-reacq", directory=tmp_path)
        assert second.acquire(), "a released lock must not block the next start"
        second.release()


class _StubSummary:
    """Minimal stand-in for the daily summary `evaluate` reads.

    Only two attributes are touched, and supplying them directly keeps this
    test about alert ownership rather than about dataset construction.
    """

    class _Readiness:
        checks: list = []

    class _Stats:
        quality_distribution: dict = {}

    readiness = _Readiness()
    stats = _Stats()


class TestAlertOwnership:
    """Regression: `evaluate` must not retract another producer's alert.

    Production, 2026-08-03: the supervisor raised SUPERVISOR_RESTART_FAILING at
    24 consecutive failures; the collector's next status tick called `evaluate`,
    which cleared every active key it had not itself re-raised, erasing the
    alert. `pmbtc watch` showed nothing while the restart loop continued.
    """

    def test_evaluate_does_not_own_the_supervisor_alert(self) -> None:
        from pmbtc.ops.alerts import EVALUATED_KINDS

        assert AlertKind.SUPERVISOR_RESTART_FAILING.value not in EVALUATED_KINDS

    def test_evaluate_preserves_a_supervisor_alert(self, config: Config) -> None:
        from pmbtc.ops.alerts import (
            Alert,
            AlertState,
            Severity,
            alert_state_path,
            dispatch,
            evaluate,
        )
        from pmbtc.utils.timeutils import utc_now_ms

        state = AlertState(alert_state_path(config))
        alert = Alert(
            kind=AlertKind.SUPERVISOR_RESTART_FAILING,
            severity=Severity.CRITICAL,
            message="collector will not stay up",
            detail={"check": "supervisor"},
            raised_at_ms=utc_now_ms(),
        )
        assert dispatch(config, state, [alert]), "the supervisor alert did not fire"
        key = alert.key
        assert key in state.active

        # A healthy collector tick: evaluate finds none of *its* conditions.
        _write_heartbeat(config, pid=1, token="t")
        reloaded = AlertState(alert_state_path(config))
        evaluate(config, reloaded, heartbeat=None, summary=_StubSummary())
        assert key in reloaded.active, "evaluate erased another producer's alert"

    def test_evaluate_still_clears_its_own_conditions(self, config: Config) -> None:
        """The dedup behaviour that made the bug subtle must be preserved."""
        from pmbtc.ops.alerts import AlertState, alert_state_path, dispatch, evaluate

        state = AlertState(alert_state_path(config))
        # No heartbeat -> collection_stopped fires and is recorded.
        raised = evaluate(config, state, heartbeat=None, summary=_StubSummary())
        assert dispatch(config, state, raised)
        assert any(k.startswith("collection_stopped") for k in state.active)

        # Heartbeat healthy -> the condition clears and must be forgotten.
        _write_heartbeat(config, pid=1, token="t")
        from pmbtc.ops.heartbeat import read_heartbeat

        evaluate(config, state, heartbeat=read_heartbeat(config), summary=_StubSummary())
        assert not any(k.startswith("collection_stopped") for k in state.active)


class TestHealthChecks:
    """Requirement 5: verify heartbeat, PID, instance, clock, feeds."""

    def _names(self, report) -> set[str]:
        return {c.name for c in report.checks}

    def test_all_five_subjects_are_verified(self, config: Config) -> None:
        child = _FakeProcess(pid=99, exit_after_polls=10)
        _write_heartbeat(config, pid=99)
        report = check_health(config, child=child, lock_held=True)
        assert {"single_instance", "collector_process", "heartbeat",
                "heartbeat_owner", "clock", "feeds"} <= self._names(report)
        assert report.healthy, report.render()

    def test_a_stale_heartbeat_is_unhealthy(self, config: Config) -> None:
        child = _FakeProcess(pid=99, exit_after_polls=10)
        budget = config.service.status_interval_seconds * 3 * 1000
        _write_heartbeat(config, pid=99, age_ms=budget + 60_000)
        report = check_health(config, child=child, lock_held=True)
        assert not report.healthy
        assert any(c.name == "heartbeat" for c in report.failures)

    def test_a_foreign_heartbeat_is_detected_by_pid(self, config: Config) -> None:
        """No token (hand-started collector): fall back to the PID."""
        child = _FakeProcess(pid=99, exit_after_polls=10)
        _write_heartbeat(config, pid=12345)
        report = check_health(config, child=child, lock_held=True)
        assert any(c.name == "heartbeat_owner" for c in report.failures)

    def test_ownership_uses_the_token_not_the_pid(self, config: Config) -> None:
        """The venv python.exe is a launcher that re-execs the interpreter, so
        the supervised PID is never the PID that writes the heartbeat.
        Measured on the real machine: child 16132, writer 18556."""
        child = _FakeProcess(pid=16132, exit_after_polls=10)
        _write_heartbeat(config, pid=18556, token="abc123")
        report = check_health(
            config, child=child, lock_held=True, supervisor_token="abc123"
        )
        assert report.healthy, report.render()

    def test_a_heartbeat_from_a_previous_supervisor_run_is_rejected(
        self, config: Config
    ) -> None:
        child = _FakeProcess(pid=16132, exit_after_polls=10)
        _write_heartbeat(config, pid=18556, token="an-older-run")
        report = check_health(
            config, child=child, lock_held=True, supervisor_token="this-run"
        )
        assert any(c.name == "heartbeat_owner" for c in report.failures)

    def test_an_unhealthy_clock_is_reported(self, config: Config) -> None:
        child = _FakeProcess(pid=99, exit_after_polls=10)
        _write_heartbeat(config, pid=99, clock="stale")
        report = check_health(config, child=child, lock_held=True)
        assert any(c.name == "clock" for c in report.failures)

    def test_a_quiet_feed_is_not_a_failure(self, config: Config) -> None:
        """v0.12.0's distinction holds here too: quiet data with a live
        transport is not a supervision problem."""
        child = _FakeProcess(pid=99, exit_after_polls=10)
        _write_heartbeat(config, pid=99, feeds=[
            {"name": "clob:quiet", "state": "stale", "transport_connected": True}
        ])
        report = check_health(config, child=child, lock_held=True)
        assert not any(c.name == "feeds" for c in report.failures), report.render()

    def test_a_disconnected_transport_is_a_failure(self, config: Config) -> None:
        child = _FakeProcess(pid=99, exit_after_polls=10)
        _write_heartbeat(config, pid=99, feeds=[
            {"name": "clob:dead", "state": "reconnecting", "transport_connected": False}
        ])
        report = check_health(config, child=child, lock_held=True)
        assert any(c.name == "feeds" for c in report.failures)

    def test_no_heartbeat_at_all_is_unhealthy(self, config: Config) -> None:
        report = check_health(config, child=None, lock_held=True, heartbeat=None)
        assert not report.healthy


class TestRestart:
    """Requirements 1-3: restart on exit, and alert when that stops working."""

    def test_the_collector_is_restarted_after_it_exits(self, config: Config) -> None:
        starts: list[int] = []

        def spawn() -> _FakeProcess:
            starts.append(1)
            _write_heartbeat(config, pid=500 + len(starts))
            return _FakeProcess(pid=500 + len(starts), exit_after_polls=0)

        supervisor = CollectorSupervisor(
            config, spawn=spawn, check_interval_s=0.0,
            startup_grace_s=0.0,
            lock=_isolated_lock(config, "restart"),
        )
        supervisor.run(max_restarts=3, sleep=lambda _s: None)
        assert len(starts) >= 4, f"collector started only {len(starts)} time(s)"
        assert supervisor.stats.restarts == 3

    def test_a_hung_collector_is_killed_and_replaced(self, config: Config) -> None:
        """Running but not collecting is the failure a PID check cannot see."""
        budget = config.service.status_interval_seconds * 3 * 1000
        children: list[_FakeProcess] = []

        def spawn() -> _FakeProcess:
            # Alive forever, but its heartbeat is far past the budget.
            child = _FakeProcess(pid=700 + len(children), exit_after_polls=10**6)
            _write_heartbeat(config, pid=child.pid, age_ms=budget + 120_000)
            children.append(child)
            return child

        supervisor = CollectorSupervisor(
            config, spawn=spawn, check_interval_s=0.0,
            startup_grace_s=0.0,
            lock=_isolated_lock(config, "hung"),
        )
        supervisor.run(max_restarts=1, sleep=lambda _s: None)
        assert children[0].terminated, "a hung collector was left running"

    def test_repeated_fast_failures_raise_an_alert(self, config: Config) -> None:
        cfg = Config(
            app={"base_dir": str(config.resolved_path(config.app.base_dir))},
            service={"max_consecutive_errors": 3},
        )
        supervisor = CollectorSupervisor(
            cfg,
            spawn=lambda: _FakeProcess(exit_after_polls=0, code=1),
            check_interval_s=0.0,
            startup_grace_s=0.0,
            lock=_isolated_lock(cfg, "alert"),
        )
        supervisor.run(max_restarts=5, sleep=lambda _s: None)
        assert supervisor.stats.consecutive_failures >= 3
        assert supervisor.stats.alerts_fired >= 1

        state = json.loads(
            (cfg.resolved_path(cfg.app.data_dir) / "alert_state.json").read_text()
        )
        assert any(
            AlertKind.SUPERVISOR_RESTART_FAILING.value in key
            for key in state.get("active", {})
        ), "the supervisor alert was not dispatched through the existing engine"

    def test_backoff_grows_and_is_bounded(self, config: Config) -> None:
        supervisor = CollectorSupervisor(
            config, spawn=lambda: _FakeProcess(),
            lock=_isolated_lock(config, "backoff"),
        )
        base = config.service.restart_backoff_s
        assert supervisor.backoff_seconds(1) == pytest.approx(base)
        assert supervisor.backoff_seconds(3) > supervisor.backoff_seconds(2)
        assert supervisor.backoff_seconds(50) <= 300.0

    def test_a_long_run_resets_the_failure_counter(self, config: Config, monkeypatch) -> None:
        """A collector that ran for hours must not inherit an old failure
        streak; only fast failures count as failures to start."""
        import pmbtc.ops.supervisor as sup

        clock = {"t": 0.0}
        monkeypatch.setattr(sup.time, "monotonic", lambda: clock["t"])

        def spawn() -> _FakeProcess:
            clock["t"] += HEALTHY_RUN_SECONDS + 1
            _write_heartbeat(config, pid=888)
            return _FakeProcess(pid=888, exit_after_polls=0)

        supervisor = CollectorSupervisor(
            config, spawn=spawn, check_interval_s=0.0,
            startup_grace_s=0.0,
            lock=_isolated_lock(config, "reset"),
        )
        supervisor.stats.consecutive_failures = 7
        supervisor.run(max_restarts=1, sleep=lambda _s: None)
        assert supervisor.stats.consecutive_failures == 0


def test_collector_command_avoids_the_launcher_shim() -> None:
    """pmbtc.exe spawns a second process, so its PID is not the heartbeat's."""
    command = collector_command()
    assert command[1:4] == ["-m", "pmbtc.cli", "run"]
    assert "--minutes" in command and "0" in command
    assert not any(part.endswith("pmbtc.exe") for part in command)
