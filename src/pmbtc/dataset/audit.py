"""Pre-training data audit.

The last gate before any model sees the data. Readiness asks "is there *enough*
data"; this asks "is the data *sound*". They fail differently and both must pass.

Eight checks, each corresponding to a way a dataset silently becomes worthless:

``label_leakage``
    Any observation carrying information from after its own snapshot instant, or
    any outcome-derived column masquerading as a feature.
``duplicate_markets``
    The same window recorded twice under different condition ids — which would
    put correlated rows on both sides of a train/test split.
``duplicate_snapshots``
    The same ``(market, horizon)`` twice.
``timestamp_inconsistencies``
    Windows whose open/lock/settlement ordering is impossible, snapshots outside
    their own window, or misaligned boundaries.
``missing_settlement_labels``
    Markets long past settlement with no official outcome.
``feature_schema_drift``
    The set of features present has changed across the collection period, so
    early and late rows are not the same measurement.
``provider_mismatches``
    Markets settling on a provider other than the configured one.
``class_imbalance``
    A skew extreme enough that accuracy becomes meaningless.

A failing audit **aborts training** with the full report. There is no override
flag: the correct response to a failed audit is to fix the data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pmbtc.config import Config
from pmbtc.dataset.leakage import LeakageGuard, scan_monotonicity, scan_snapshots
from pmbtc.dataset.schema import FeatureSnapshot, MarketRecord
from pmbtc.exceptions import DataError
from pmbtc.logging_setup import get_logger

log = get_logger("pmbtc.dataset.audit")


class AuditSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    FAILURE = "failure"


@dataclass
class AuditFinding:
    check: str
    severity: AuditSeverity
    message: str
    #: A bounded sample of offending records — enough to act on, not a dump.
    examples: list[str] = field(default_factory=list)
    count: int = 0

    def __str__(self) -> str:
        head = f"[{self.severity.value.upper()}] {self.check}: {self.message}"
        if self.examples:
            head += "\n    e.g. " + "; ".join(self.examples[:5])
        return head


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    markets: int = 0
    snapshots: int = 0

    @property
    def failures(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity is AuditSeverity.FAILURE]

    @property
    def warnings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity is AuditSeverity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.passed and not self.warnings:
            return (
                f"audit clean: {self.markets} market(s), {self.snapshots} snapshot(s), "
                f"{len(self.findings)} check(s) with nothing to report"
            )
        lines = [
            f"audit: {len(self.failures)} failure(s), {len(self.warnings)} warning(s) "
            f"over {self.markets} market(s) / {self.snapshots} snapshot(s)"
        ]
        lines += [f"  {finding}" for finding in (*self.failures, *self.warnings)]
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Abort training with the full report attached."""
        if self.passed:
            return
        raise DataError(
            "Pre-training data audit failed; training aborted",
            context={
                "failures": len(self.failures),
                "report": "; ".join(f.check + ": " + f.message for f in self.failures),
            },
        )


# --------------------------------------------------------------------------- #
def audit_dataset(
    config: Config,
    markets: list[MarketRecord],
    snapshots: list[FeatureSnapshot],
) -> AuditReport:
    """Run every audit check. All of them run, so the report is complete."""
    report = AuditReport(markets=len(markets), snapshots=len(snapshots))
    by_id = {m.condition_id: m for m in markets}

    def add(
        check: str, severity: AuditSeverity, message: str, examples: list[str], count: int = 0
    ) -> None:
        report.findings.append(AuditFinding(check, severity, message, examples, count))

    # --- 1. Label leakage --------------------------------------------- #
    guard = LeakageGuard(config.dataset.observation_tolerance_ms)
    leaks = scan_snapshots(snapshots, guard)
    ordering = scan_monotonicity(snapshots)
    if leaks or ordering:
        add(
            "label_leakage",
            AuditSeverity.FAILURE,
            f"{len(leaks)} leaked observation(s), {len(ordering)} ordering violation(s)",
            [str(f) for f in (*leaks, *ordering)[:5]],
            len(leaks) + len(ordering),
        )
    else:
        add("label_leakage", AuditSeverity.INFO, "no leakage detected", [])

    # --- 2. Duplicate markets ------------------------------------------ #
    # Same settlement instant recorded under two condition ids: correlated rows
    # that a train/test split would put on both sides.
    by_window: dict[tuple[int, str], list[str]] = {}
    for market in markets:
        by_window.setdefault(
            (market.settlement_time_ms, market.series_slug), []
        ).append(market.condition_id)
    duplicates = {k: v for k, v in by_window.items() if len(v) > 1}
    if duplicates:
        add(
            "duplicate_markets",
            AuditSeverity.FAILURE,
            f"{len(duplicates)} settlement instant(s) mapped to multiple markets",
            [f"{ts}: {ids}" for (ts, _), ids in list(duplicates.items())[:5]],
            len(duplicates),
        )
    else:
        add("duplicate_markets", AuditSeverity.INFO, "no duplicate windows", [])

    # --- 3. Duplicate snapshots ----------------------------------------- #
    keys = Counter((s.condition_id, s.horizon_seconds) for s in snapshots)
    repeated = [k for k, n in keys.items() if n > 1]
    if repeated:
        add(
            "duplicate_snapshots",
            AuditSeverity.FAILURE,
            f"{len(repeated)} (market, horizon) pair(s) recorded more than once",
            [f"{cid[:14]}@T-{h}" for cid, h in repeated[:5]],
            len(repeated),
        )
    else:
        add("duplicate_snapshots", AuditSeverity.INFO, "all snapshots unique", [])

    # --- 4. Timestamp inconsistencies ------------------------------------ #
    bad_time: list[str] = []
    window_ms = config.app.window_seconds * 1000
    for market in markets:
        if not (market.open_time_ms < market.lock_time_ms <= market.settlement_time_ms):
            bad_time.append(f"{market.slug}: open/lock/settle out of order")
        elif market.settlement_time_ms - market.open_time_ms != window_ms:
            bad_time.append(
                f"{market.slug}: window is "
                f"{(market.settlement_time_ms - market.open_time_ms) / 1000:.0f}s"
            )
        if market.discovery_time_ms > market.settlement_time_ms:
            bad_time.append(f"{market.slug}: discovered after settlement")
    for snapshot in snapshots:
        parent = by_id.get(snapshot.condition_id)
        if parent is None:
            continue
        if not (parent.open_time_ms <= snapshot.snapshot_time_ms <= parent.settlement_time_ms):
            bad_time.append(
                f"{snapshot.slug}@T-{snapshot.horizon_seconds}: instant outside its window"
            )
    if bad_time:
        add(
            "timestamp_inconsistencies",
            AuditSeverity.FAILURE,
            f"{len(bad_time)} timestamp problem(s)",
            bad_time[:5],
            len(bad_time),
        )
    else:
        add("timestamp_inconsistencies", AuditSeverity.INFO, "timestamps consistent", [])

    # --- 5. Missing settlement labels ------------------------------------ #
    grace_ms = config.dataset.resolution_max_wait_s * 1000
    newest = max((m.settlement_time_ms for m in markets), default=0)
    unlabelled = [
        m
        for m in markets
        if not m.is_labelled and newest - m.settlement_time_ms > grace_ms
    ]
    if unlabelled:
        add(
            "missing_settlement_labels",
            AuditSeverity.FAILURE,
            f"{len(unlabelled)} market(s) past the resolution grace period with no outcome",
            [m.slug for m in unlabelled[:5]],
            len(unlabelled),
        )
    else:
        pending = sum(1 for m in markets if not m.is_labelled)
        add(
            "missing_settlement_labels",
            AuditSeverity.INFO,
            f"all settled markets labelled ({pending} still within the grace period)",
            [],
        )

    # --- 6. Feature schema drift ------------------------------------------ #
    # Early and late rows must be the same measurement. A feature that appears
    # or disappears mid-collection splits the dataset into incomparable halves.
    if snapshots:
        ordered = sorted(snapshots, key=lambda s: s.snapshot_time_ms)
        half = max(1, len(ordered) // 2)
        early = {o.name for s in ordered[:half] for o in s.observations}
        late = {o.name for s in ordered[half:] for o in s.observations}
        appeared, vanished = late - early, early - late
        if appeared or vanished:
            severity = (
                AuditSeverity.FAILURE if vanished else AuditSeverity.WARNING
            )
            add(
                "feature_schema_drift",
                severity,
                f"{len(vanished)} feature(s) disappeared, {len(appeared)} appeared "
                "mid-collection",
                [f"-{n}" for n in sorted(vanished)[:3]]
                + [f"+{n}" for n in sorted(appeared)[:3]],
                len(vanished) + len(appeared),
            )
        else:
            add("feature_schema_drift", AuditSeverity.INFO, "feature set stable", [])

    # --- 7. Provider mismatches -------------------------------------------- #
    expected = config.settlement.expected_source
    wrong = [m for m in markets if m.settlement_provider is not expected]
    if wrong:
        add(
            "provider_mismatches",
            AuditSeverity.FAILURE,
            f"{len(wrong)} market(s) settle on a provider other than {expected.value}",
            [f"{m.slug}: {m.settlement_provider.value}" for m in wrong[:5]],
            len(wrong),
        )
    else:
        add(
            "provider_mismatches",
            AuditSeverity.INFO,
            f"all markets settle on {expected.value}",
            [],
        )
    hashes = {m.settlement_spec_hash for m in markets if m.settlement_spec_hash}
    if len(hashes) > 1:
        add(
            "provider_mismatches",
            AuditSeverity.FAILURE,
            f"{len(hashes)} distinct settlement fingerprints in one dataset",
            sorted(hashes)[:5],
            len(hashes),
        )

    # --- 8. Class imbalance -------------------------------------------------- #
    labelled = [m for m in markets if m.label is not None]
    if labelled:
        ups = sum(1 for m in labelled if m.label == 1)
        rate = ups / len(labelled)
        minority = min(rate, 1.0 - rate)
        limit = config.training.min_class_balance
        if minority < limit:
            add(
                "class_imbalance",
                AuditSeverity.FAILURE,
                f"minority class is {minority:.1%} of labels (limit {limit:.1%}); "
                "accuracy is not a meaningful metric at this skew",
                [f"up={ups}/{len(labelled)}"],
                len(labelled),
            )
        else:
            add(
                "class_imbalance",
                AuditSeverity.INFO,
                f"balanced: up={rate:.1%}, minority={minority:.1%}",
                [],
            )
    else:
        add("class_imbalance", AuditSeverity.FAILURE, "no labelled markets to audit", [])

    log.info(
        "audit.completed",
        markets=len(markets),
        snapshots=len(snapshots),
        failures=len(report.failures),
        warnings=len(report.warnings),
    )
    return report


def audit_store(config: Config, store: Any) -> AuditReport:
    return audit_dataset(config, store.markets(), store.snapshots())
