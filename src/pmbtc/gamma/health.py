"""Per-market health checks.

Settlement verification answers "does this market resolve the way my model
thinks". Health answers the separate question "is this market *usable*" -- is
anyone quoting it, is the venue accepting orders, are the identifiers sane.

Both must pass. A market can settle exactly as expected and still be untradeable
because its book is empty, and it can be liquid and healthy while resolving off
the wrong feed.

Every issue is an enum member, so the rejection-reason metric is a fixed
vocabulary rather than free text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pmbtc.config import Config
from pmbtc.gamma.liquidity import LiquiditySnapshot
from pmbtc.settlement.spec import SettlementSpecification, TieRule


class HealthIssue(StrEnum):
    MISSING_METADATA = "missing_metadata"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"
    INVALID_TIMESTAMP = "invalid_timestamp"
    UNEXPECTED_RECURRENCE = "unexpected_recurrence"
    MISSING_LIQUIDITY = "missing_liquidity"
    TRADING_SUSPENDED = "trading_suspended"
    SETTLEMENT_AMBIGUITY = "settlement_ambiguity"
    NO_ORDER_BOOK = "no_order_book"
    ONE_SIDED_QUOTE = "one_sided_quote"
    SPREAD_TOO_WIDE = "spread_too_wide"
    ALREADY_SETTLED = "already_settled"


@dataclass(frozen=True, slots=True)
class HealthResult:
    healthy: bool
    issues: tuple[HealthIssue, ...]
    details: tuple[str, ...]

    @property
    def primary_issue(self) -> HealthIssue | None:
        return self.issues[0] if self.issues else None

    def __str__(self) -> str:
        return "healthy" if self.healthy else "; ".join(self.details)


class MarketHealthChecker:
    """Applies the configured health gates to one market."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg = config.health

    def check(
        self,
        market: dict[str, Any],
        spec: SettlementSpecification,
        snapshot: LiquiditySnapshot,
        *,
        now_ms: int,
        seen_condition_ids: set[str] | None = None,
        seen_slugs: set[str] | None = None,
    ) -> HealthResult:
        issues: list[HealthIssue] = []
        details: list[str] = []

        def fail(issue: HealthIssue, detail: str) -> None:
            issues.append(issue)
            details.append(f"{issue.value}: {detail}")

        # --- identity ------------------------------------------------- #
        if not spec.condition_id or not spec.slug:
            fail(HealthIssue.MISSING_METADATA, "condition id or slug absent")
        if not spec.question:
            fail(HealthIssue.MISSING_METADATA, "no question/title")
        if seen_condition_ids is not None and spec.condition_id in seen_condition_ids:
            fail(
                HealthIssue.DUPLICATE_IDENTIFIER,
                f"condition id {spec.condition_id[:14]} repeated",
            )
        if seen_slugs is not None and spec.slug in seen_slugs:
            fail(HealthIssue.DUPLICATE_IDENTIFIER, f"slug {spec.slug} repeated")

        # --- timestamps ----------------------------------------------- #
        if not spec.window_open_ms or not spec.window_close_ms:
            fail(HealthIssue.INVALID_TIMESTAMP, "window boundaries incomplete")
        elif spec.window_close_ms <= spec.window_open_ms:
            fail(HealthIssue.INVALID_TIMESTAMP, "settlement is not after the open")
        elif self.cfg.reject_past_settlement and spec.window_close_ms <= now_ms:
            fail(
                HealthIssue.ALREADY_SETTLED,
                f"settled {(now_ms - spec.window_close_ms) / 1000:.0f}s ago",
            )

        # --- cadence --------------------------------------------------- #
        expected = self.config.app.window_seconds
        if spec.interval_seconds != expected:
            fail(
                HealthIssue.UNEXPECTED_RECURRENCE,
                f"{spec.interval_seconds}s window, expected {expected}s",
            )
        elif spec.window_open_ms:
            skew = spec.window_open_ms % (expected * 1000)
            if skew > self.cfg.max_boundary_skew_ms:
                fail(
                    HealthIssue.INVALID_TIMESTAMP,
                    f"window open is {skew}ms off the {expected}s grid",
                )

        # --- venue state ----------------------------------------------- #
        if self.cfg.require_accepting_orders and market.get("acceptingOrders") is False:
            fail(HealthIssue.TRADING_SUSPENDED, "venue is not accepting orders")
        if market.get("closed") is True:
            fail(HealthIssue.TRADING_SUSPENDED, "market is closed")
        if market.get("active") is False:
            fail(HealthIssue.TRADING_SUSPENDED, "market is inactive")
        if self.cfg.require_order_book and market.get("enableOrderBook") is False:
            fail(HealthIssue.NO_ORDER_BOOK, "order book disabled for this market")

        # --- liquidity -------------------------------------------------- #
        if snapshot.liquidity_usdc is not None and snapshot.liquidity_usdc < (
            self.cfg.min_liquidity_usdc
        ):
            fail(
                HealthIssue.MISSING_LIQUIDITY,
                f"{snapshot.liquidity_usdc:.0f} USDC < {self.cfg.min_liquidity_usdc:.0f}",
            )
        if self.cfg.require_two_sided_quote and not snapshot.two_sided:
            fail(HealthIssue.ONE_SIDED_QUOTE, "no two-sided quote")
        elif snapshot.spread is not None and snapshot.spread > self.cfg.max_spread:
            fail(
                HealthIssue.SPREAD_TOO_WIDE,
                f"spread {snapshot.spread:.3f} > {self.cfg.max_spread:.3f}",
            )

        # --- settlement ambiguity --------------------------------------- #
        # Duplicated from the verifier on purpose: health is evaluated before
        # verification in the pipeline, and a market with an ambiguous source
        # should never reach the more expensive checks.
        if spec.conflicts:
            fail(
                HealthIssue.SETTLEMENT_AMBIGUITY,
                "; ".join(spec.conflicts.values()),
            )
        elif spec.tie_rule is TieRule.UNKNOWN:
            fail(HealthIssue.SETTLEMENT_AMBIGUITY, "tie rule could not be determined")

        return HealthResult(not issues, tuple(issues), tuple(details))
