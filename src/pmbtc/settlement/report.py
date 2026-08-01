"""The startup settlement verification report.

Trading may not begin until this report passes. It is rendered two ways from one
source of truth: a table for a human at a terminal, and a structured log line
per market for the audit trail and the dashboard.

The report deliberately shows *rejected* markets too, with the failing gate
named. A silent empty report and a report full of rejections mean very different
things, and an operator must be able to tell them apart at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from pmbtc.logging_setup import get_logger
from pmbtc.settlement.verifier import VerificationResult
from pmbtc.utils.timeutils import isoformat

log = get_logger("pmbtc.settlement.report")


@dataclass(frozen=True, slots=True)
class SettlementReport:
    """Aggregated verification results for one startup scan."""

    results: tuple[VerificationResult, ...]

    @property
    def verified(self) -> tuple[VerificationResult, ...]:
        return tuple(r for r in self.results if r.trading_enabled)

    @property
    def rejected(self) -> tuple[VerificationResult, ...]:
        return tuple(r for r in self.results if not r.trading_enabled)

    @property
    def all_passed(self) -> bool:
        """True only if at least one market verified and none were rejected.

        "Nothing was scanned" is not a pass: a scan that silently found no
        markets would otherwise look identical to a clean bill of health.
        """
        return bool(self.results) and not self.rejected

    @property
    def any_tradeable(self) -> bool:
        return bool(self.verified)

    # ------------------------------------------------------------------ #
    def rows(self) -> list[dict[str, str]]:
        """One dict per market, matching the agreed report fields."""
        rows: list[dict[str, str]] = []
        for result in self.results:
            spec = result.spec
            rows.append(
                {
                    "market_id": spec.market_id or spec.condition_id[:12],
                    "slug": spec.slug,
                    "resolution_source": spec.resolution_source_url or "(none published)",
                    "provider": spec.provider.value,
                    "venue": spec.venue or "-",
                    "trading_pair": spec.trading_pair or "-",
                    "interval": f"{spec.interval_seconds}s" if spec.interval_seconds else "-",
                    "settlement_timestamp": isoformat(spec.window_close_ms)
                    if spec.window_close_ms
                    else "-",
                    "tie_rule": spec.tie_rule.value,
                    "confidence": f"{spec.confidence:.2f}",
                    "status": result.status.value,
                    "trading_enabled": "YES" if result.trading_enabled else "NO",
                    "failures": "; ".join(result.reasons),
                }
            )
        return rows

    def render(self, console: Console | None = None) -> None:
        """Print the operator-facing table."""
        console = console or Console()
        table = Table(
            title="Settlement verification report",
            caption="Trading is blocked for every row not marked YES.",
            show_lines=False,
        )
        for column in (
            "market",
            "provider",
            "pair",
            "interval",
            "settles (UTC)",
            "tie",
            "conf",
            "status",
            "trade",
        ):
            table.add_column(column, overflow="fold")

        for row in self.rows():
            enabled = row["trading_enabled"] == "YES"
            table.add_row(
                row["market_id"],
                f"{row['provider']} ({row['venue']})",
                row["trading_pair"],
                row["interval"],
                row["settlement_timestamp"],
                row["tie_rule"],
                row["confidence"],
                f"[green]{row['status']}[/]" if enabled else f"[red]{row['status']}[/]",
                "[green]YES[/]" if enabled else "[red]NO[/]",
            )
        console.print(table)

        for result in self.rejected:
            console.print(
                f"[red]blocked[/] {result.spec.slug or result.spec.condition_id[:12]}: "
                f"{'; '.join(result.reasons)}"
            )
        if not self.results:
            console.print("[yellow]No markets scanned — nothing verified, nothing tradeable.[/]")

    def emit_log(self) -> None:
        """One structured record per market, for the audit trail."""
        for result in self.results:
            spec = result.spec
            log.info(
                "settlement.verification",
                market_id=spec.market_id,
                condition_id=spec.condition_id,
                slug=spec.slug,
                provider=spec.provider.value,
                resolution_source=spec.resolution_source_url,
                trading_pair=spec.trading_pair,
                interval_seconds=spec.interval_seconds,
                settlement_timestamp_ms=spec.window_close_ms,
                settlement_timestamp=isoformat(spec.window_close_ms)
                if spec.window_close_ms
                else None,
                tie_rule=spec.tie_rule.value,
                confidence=spec.confidence,
                spec_hash=spec.spec_hash,
                evidence=spec.evidence_summary(),
                status=result.status.value,
                trading_enabled=result.trading_enabled,
                failures=list(result.reasons),
            )
