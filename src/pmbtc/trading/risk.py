"""Portfolio risk limits — the layer that survives a wrong model.

Sizing asks "how much is this trade worth". Risk asks a different and more
important question: "how much has already gone wrong today, and should this
account still be trading at all". The two must stay separate, because the
failure mode being defended against is precisely a model that is confident and
wrong for a sustained stretch — in which case per-trade sizing will keep saying
yes, correctly by its own lights, all the way down.

Every limit here is a stand-down, not a resize. Crossing one stops trading and
names itself; nothing in this module can make a position larger.

Determinism: the ledger never reads the clock. Every method takes ``now_ms``
from the caller, so a backtest, a paper run, and a replay of the same sequence
produce byte-identical state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmbtc.config import Config, RiskConfig
from pmbtc.constants import SkipReason
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.trading.risk")


def _day_key(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, UTC).strftime("%Y-%m-%d")


def _week_key(ms: int) -> str:
    iso = datetime.fromtimestamp(ms / 1000.0, UTC).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


@dataclass
class RiskState:
    """Everything the limits are computed from."""

    bankroll_usdc: float
    starting_bankroll_usdc: float
    peak_bankroll_usdc: float = 0.0
    day_key: str = ""
    day_pnl_usdc: float = 0.0
    day_start_bankroll_usdc: float = 0.0
    week_key: str = ""
    week_pnl_usdc: float = 0.0
    week_start_bankroll_usdc: float = 0.0
    consecutive_losses: int = 0
    open_exposure_usdc: float = 0.0
    open_positions: int = 0
    cooldown_until_ms: int = 0
    halted: bool = False
    halt_reason: str = ""
    trades: int = 0
    wins: int = 0

    @property
    def drawdown(self) -> float:
        if self.peak_bankroll_usdc <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.bankroll_usdc / self.peak_bankroll_usdc)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bankroll_usdc": round(self.bankroll_usdc, 4),
            "peak_bankroll_usdc": round(self.peak_bankroll_usdc, 4),
            "drawdown": round(self.drawdown, 6),
            "day_key": self.day_key,
            "day_pnl_usdc": round(self.day_pnl_usdc, 4),
            "week_key": self.week_key,
            "week_pnl_usdc": round(self.week_pnl_usdc, 4),
            "consecutive_losses": self.consecutive_losses,
            "open_exposure_usdc": round(self.open_exposure_usdc, 4),
            "open_positions": self.open_positions,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "trades": self.trades,
            "wins": self.wins,
        }


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    allowed: bool
    reason: SkipReason | None = None
    detail: str = ""

    def __bool__(self) -> bool:
        return self.allowed


_OK = RiskVerdict(allowed=True)


@dataclass
class RiskLedger:
    """Tracks exposure and losses, and refuses trades that breach a limit."""

    config: RiskConfig
    state: RiskState
    #: Checked on every decision so an operator can stop the bot without a
    #: deploy. Absent in backtests, where there is no operator to intervene.
    honour_kill_switch: bool = True
    _kill_switch_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        starting_bankroll_usdc: float,
        honour_kill_switch: bool = True,
    ) -> RiskLedger:
        return cls(
            config=config.risk,
            state=RiskState(
                bankroll_usdc=starting_bankroll_usdc,
                starting_bankroll_usdc=starting_bankroll_usdc,
                peak_bankroll_usdc=starting_bankroll_usdc,
                day_start_bankroll_usdc=starting_bankroll_usdc,
                week_start_bankroll_usdc=starting_bankroll_usdc,
            ),
            honour_kill_switch=honour_kill_switch,
            _kill_switch_path=config.resolved_path(config.risk.kill_switch_file),
        )

    # ------------------------------------------------------------------ #
    # Period rollover
    # ------------------------------------------------------------------ #
    def roll_periods(self, now_ms: int) -> None:
        """Reset the daily and weekly counters when the UTC period turns over.

        Called before every check so a limit breached yesterday does not silence
        the bot forever, and so a backtest replaying months of history applies
        daily limits daily rather than once.
        """
        day, week = _day_key(now_ms), _week_key(now_ms)
        if day != self.state.day_key:
            self.state.day_key = day
            self.state.day_pnl_usdc = 0.0
            self.state.day_start_bankroll_usdc = self.state.bankroll_usdc
            # A daily stand-down expires with its day; a manual halt does not.
            if self.state.halted and self.state.halt_reason.startswith("daily"):
                self.state.halted = False
                self.state.halt_reason = ""
        if week != self.state.week_key:
            self.state.week_key = week
            self.state.week_pnl_usdc = 0.0
            self.state.week_start_bankroll_usdc = self.state.bankroll_usdc

    # ------------------------------------------------------------------ #
    # The gate
    # ------------------------------------------------------------------ #
    def check(self, *, now_ms: int, stake_usdc: float) -> RiskVerdict:
        """May a position of ``stake_usdc`` be opened right now?"""
        self.roll_periods(now_ms)
        cfg, st = self.config, self.state

        if self.honour_kill_switch and self.kill_switch_engaged():
            return RiskVerdict(False, SkipReason.KILL_SWITCH, "kill switch file present")
        if st.halted:
            return RiskVerdict(False, SkipReason.RISK_LIMIT, st.halt_reason)
        if now_ms < st.cooldown_until_ms:
            remaining = (st.cooldown_until_ms - now_ms) / 60_000.0
            return RiskVerdict(
                False, SkipReason.RISK_LIMIT, f"cooling down for {remaining:.1f} more minutes"
            )
        if st.bankroll_usdc <= 0.0:
            return RiskVerdict(False, SkipReason.RISK_LIMIT, "bankroll exhausted")

        if st.open_positions >= cfg.max_concurrent_positions:
            return RiskVerdict(
                False,
                SkipReason.DUPLICATE_POSITION,
                f"{st.open_positions} position(s) open, limit {cfg.max_concurrent_positions}",
            )
        if st.open_exposure_usdc + stake_usdc > cfg.max_total_exposure_usdc:
            return RiskVerdict(
                False,
                SkipReason.RISK_LIMIT,
                f"exposure {st.open_exposure_usdc + stake_usdc:.2f} would exceed "
                f"{cfg.max_total_exposure_usdc:.2f}",
            )

        # Loss limits are evaluated against the stake at risk, not only against
        # losses already taken: a trade that *could* breach the daily limit is
        # refused before it is placed, not regretted afterwards.
        day_loss = -(st.day_pnl_usdc - stake_usdc)
        if day_loss > cfg.max_daily_loss_usdc:
            return RiskVerdict(
                False,
                SkipReason.RISK_LIMIT,
                f"worst-case daily loss {day_loss:.2f} > {cfg.max_daily_loss_usdc:.2f}",
            )
        day_fraction = day_loss / max(st.day_start_bankroll_usdc, 1e-9)
        if day_fraction > cfg.max_daily_loss_fraction:
            return RiskVerdict(
                False,
                SkipReason.RISK_LIMIT,
                f"worst-case daily loss {day_fraction:.2%} > {cfg.max_daily_loss_fraction:.2%}",
            )
        week_loss = -(st.week_pnl_usdc - stake_usdc)
        week_fraction = week_loss / max(st.week_start_bankroll_usdc, 1e-9)
        if week_fraction > cfg.max_weekly_loss_fraction:
            return RiskVerdict(
                False,
                SkipReason.RISK_LIMIT,
                f"worst-case weekly loss {week_fraction:.2%} > "
                f"{cfg.max_weekly_loss_fraction:.2%}",
            )

        if st.consecutive_losses >= cfg.max_consecutive_losses:
            return RiskVerdict(
                False,
                SkipReason.RISK_LIMIT,
                f"{st.consecutive_losses} consecutive losses, limit {cfg.max_consecutive_losses}",
            )
        return _OK

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def register_open(self, *, cost_usdc: float) -> None:
        self.state.bankroll_usdc -= cost_usdc
        self.state.open_exposure_usdc += cost_usdc
        self.state.open_positions += 1

    def register_close(self, *, now_ms: int, cost_usdc: float, payoff_usdc: float) -> float:
        """Settle one position and return its P&L.

        Rolls the period first. Without this the ledger could attribute P&L to
        an uninitialised day and then have the *next* rollover clear a halt that
        had only just fired — a daily loss limit that silently lasted no time
        at all.
        """
        self.roll_periods(now_ms)
        pnl = payoff_usdc - cost_usdc
        st = self.state
        st.bankroll_usdc += payoff_usdc
        st.open_exposure_usdc = max(0.0, st.open_exposure_usdc - cost_usdc)
        st.open_positions = max(0, st.open_positions - 1)
        st.peak_bankroll_usdc = max(st.peak_bankroll_usdc, st.bankroll_usdc)
        st.day_pnl_usdc += pnl
        st.week_pnl_usdc += pnl
        st.trades += 1

        if pnl > 0:
            st.wins += 1
            st.consecutive_losses = 0
        else:
            st.consecutive_losses += 1
            if st.consecutive_losses >= self.config.max_consecutive_losses:
                self.stand_down(
                    now_ms=now_ms,
                    reason=f"{st.consecutive_losses} consecutive losses",
                )

        realised_day_loss = -st.day_pnl_usdc
        if (
            realised_day_loss > self.config.max_daily_loss_usdc
            or realised_day_loss / max(st.day_start_bankroll_usdc, 1e-9)
            > self.config.max_daily_loss_fraction
        ):
            self.halt(f"daily loss limit breached ({realised_day_loss:.2f} USDC)")
        return pnl

    def stand_down(self, *, now_ms: int, reason: str) -> None:
        """Pause for the configured cool-off rather than halting outright.

        The losing streak is cleared at the same time. The cool-off *is* the
        penalty; leaving the counter at its limit would make the streak gate
        fire again the instant the cooldown expired, turning a pause into a
        permanent stop that only a win could lift — and a win is exactly what a
        stopped bot cannot get.
        """
        self.state.cooldown_until_ms = now_ms + self.config.cooldown_minutes * 60_000
        self.state.consecutive_losses = 0
        log.warning("risk.stand_down", reason=reason, until_ms=self.state.cooldown_until_ms)

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        log.error("risk.halted", reason=reason)

    def kill_switch_engaged(self) -> bool:
        return self._kill_switch_path is not None and self._kill_switch_path.exists()
