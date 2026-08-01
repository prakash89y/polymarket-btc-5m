"""Verify a detected specification against configuration and recorded history.

The verifier answers exactly one question: **may this market be traded?** It
answers "no" by default and requires every check to pass. Each check produces a
human-readable line so the startup report can show an operator precisely which
gate failed, rather than a bare boolean.

The checks are ordered cheapest-and-most-fundamental first, and evaluation
continues through all of them so the report lists every problem at once instead
of one per run.
"""

from __future__ import annotations

from dataclasses import dataclass

from pmbtc.config import Config
from pmbtc.constants import RunMode
from pmbtc.exceptions import SettlementSourceMismatch
from pmbtc.settlement.providers import canonical_pair, descriptor_for
from pmbtc.settlement.spec import (
    REQUIRED_FIELDS,
    SettlementSpecification,
    TieRule,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class Check:
    """One verification gate."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    spec: SettlementSpecification
    status: VerificationStatus
    checks: tuple[Check, ...]
    #: The single authoritative permission flag. Nothing downstream may trade
    #: without it, and nothing may recompute it from the parts.
    trading_enabled: bool

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f"{c.name}: {c.detail}" for c in self.failures)

    @property
    def summary(self) -> str:
        if self.trading_enabled:
            return f"VERIFIED ({self.spec.provider.value}, {self.spec.trading_pair})"
        return f"{self.status.value}: {'; '.join(self.reasons) or 'unspecified'}"


class SettlementVerifier:
    """Config-driven verification of parsed settlement specifications."""

    def __init__(self, config: Config, *, known_hash: str | None = None) -> None:
        """
        Args:
            config: validated bot configuration.
            known_hash: the settlement fingerprint previously recorded for this
                series. When supplied and different from the spec's hash, the
                family has changed how it resolves and trading stops until a
                human looks at it. Supplied by
                :class:`~pmbtc.settlement.store.SettlementSpecStore`.
        """
        self.config = config
        self.known_hash = known_hash

    # ------------------------------------------------------------------ #
    def verify(self, spec: SettlementSpecification) -> VerificationResult:
        cfg = self.config.settlement
        checks: list[Check] = []

        # 1. Conflicts between structured metadata and prose.
        conflicts = spec.conflicts
        checks.append(
            Check(
                "no_source_conflict",
                not conflicts,
                "; ".join(f"{k}: {v}" for k, v in conflicts.items())
                or "structured and prose agree",
            )
        )

        # 2. Completeness.
        missing = spec.missing_fields
        checks.append(
            Check(
                "required_fields_present",
                not missing,
                f"missing: {', '.join(missing)}" if missing else
                f"all {len(REQUIRED_FIELDS)} required fields extracted",
            )
        )

        # 3. Provider identity.
        expected = cfg.expected_source
        provider_ok = spec.provider is expected
        checks.append(
            Check(
                "provider_matches_config",
                provider_ok,
                f"detected {spec.provider.value}, expected {expected.value}",
            )
        )

        # 4. Trading pair, against what the provider is an authority for.
        pair_ok, pair_detail = self._check_pair(spec)
        checks.append(Check("trading_pair_supported", pair_ok, pair_detail))

        # 5. Interval must be the instrument we model.
        want_seconds = self.config.app.window_seconds
        interval_ok = spec.interval_seconds == want_seconds
        checks.append(
            Check(
                "interval_matches_config",
                interval_ok,
                f"market is {spec.interval_seconds}s, bot trades {want_seconds}s",
            )
        )

        # 6. Timing self-consistency.
        timing_ok, timing_detail = self._check_timing(spec, want_seconds)
        checks.append(Check("window_timing_consistent", timing_ok, timing_detail))

        # 7. Tie rule must be known -- it decides the exact-equality case.
        tie_ok = spec.tie_rule is not TieRule.UNKNOWN
        checks.append(Check("tie_rule_known", tie_ok, f"tie resolves {spec.tie_rule.value}"))

        # 8. Confidence gate.
        confidence = spec.confidence
        conf_ok = confidence >= cfg.min_detection_confidence
        checks.append(
            Check(
                "detection_confidence",
                conf_ok,
                f"{confidence:.2f} vs required {cfg.min_detection_confidence:.2f} "
                f"({_weakest(spec)})",
            )
        )

        # 9. The family must not have silently changed how it resolves.
        hash_ok = self.known_hash is None or self.known_hash == spec.spec_hash
        checks.append(
            Check(
                "settlement_unchanged",
                hash_ok,
                f"recorded {self.known_hash} vs current {spec.spec_hash}"
                if not hash_ok
                else f"fingerprint {spec.spec_hash}",
            )
        )

        status = self._status_for(checks, spec)
        return VerificationResult(
            spec=spec,
            status=status,
            checks=tuple(checks),
            trading_enabled=self._trading_enabled(status),
        )

    # ------------------------------------------------------------------ #
    def _check_pair(self, spec: SettlementSpecification) -> tuple[bool, str]:
        if not spec.trading_pair:
            return False, "no trading pair established"
        try:
            descriptor = descriptor_for(spec.provider)
        except Exception:
            return False, f"no descriptor for provider {spec.provider.value}"
        if not descriptor.supports(spec.trading_pair):
            return False, (
                f"{descriptor.display_name} is not an authority for {spec.trading_pair}"
            )
        configured = self.config.settlement.providers.get(spec.provider.value)
        if configured and configured.symbol:
            want = canonical_pair(configured.symbol) or configured.symbol
            if want != spec.trading_pair:
                return False, f"config expects {want}, market settles on {spec.trading_pair}"
        return True, f"{spec.trading_pair} via {descriptor.display_name}"

    def _check_timing(self, spec: SettlementSpecification, want_seconds: int) -> tuple[bool, str]:
        if not spec.window_open_ms or not spec.window_close_ms:
            return False, "window open/close not both established"
        if spec.window_close_ms <= spec.window_open_ms:
            return False, "settlement instant is not after the window open"
        if spec.duration_seconds != spec.interval_seconds:
            return False, (
                f"published window is {spec.duration_seconds}s but the declared "
                f"interval is {spec.interval_seconds}s"
            )
        # An unaligned open means our window arithmetic and the venue's disagree.
        if want_seconds and spec.window_open_ms % (want_seconds * 1000) != 0:
            return False, f"window open {spec.window_open_ms} is not aligned to {want_seconds}s"
        return True, (
            f"{spec.duration_seconds}s window, settles at {spec.window_close_ms} (UTC ms)"
        )

    def _status_for(
        self, checks: list[Check], spec: SettlementSpecification
    ) -> VerificationStatus:
        """Map the first meaningful failure to a specific status."""
        failed = {c.name for c in checks if not c.passed}
        if not failed:
            return VerificationStatus.VERIFIED
        # Ordered by how fundamental the failure is.
        if "no_source_conflict" in failed:
            return VerificationStatus.REJECTED_CONFLICT
        if "settlement_unchanged" in failed:
            return VerificationStatus.REJECTED_PROVIDER_CHANGED
        if "provider_matches_config" in failed:
            return VerificationStatus.REJECTED_PROVIDER_MISMATCH
        if "interval_matches_config" in failed:
            return VerificationStatus.REJECTED_INTERVAL_MISMATCH
        if "trading_pair_supported" in failed:
            return VerificationStatus.REJECTED_PAIR_MISMATCH
        if "window_timing_consistent" in failed:
            return VerificationStatus.REJECTED_TIMING
        if "required_fields_present" in failed or "tie_rule_known" in failed:
            return VerificationStatus.REJECTED_INCOMPLETE
        if "detection_confidence" in failed:
            return VerificationStatus.REJECTED_LOW_CONFIDENCE
        del spec
        return VerificationStatus.REJECTED_INCOMPLETE

    def _trading_enabled(self, status: VerificationStatus) -> bool:
        if status.is_tradeable:
            return True
        # Backtests may replay markets whose source cannot be re-verified from
        # an archived payload. No money can move, and Module 1's config gates
        # already forbid this combination outside backtest mode.
        cfg = self.config.settlement
        return self.config.app.mode is RunMode.BACKTEST and not cfg.require_verified

    # ------------------------------------------------------------------ #
    def verify_or_raise(self, spec: SettlementSpecification) -> VerificationResult:
        """Verify and raise :class:`SettlementSourceMismatch` on refusal.

        Use in the live path, where a refusal must abort the window rather than
        be inspected.
        """
        result = self.verify(spec)
        if not result.trading_enabled:
            raise SettlementSourceMismatch(
                f"Settlement verification failed for {spec.slug or spec.condition_id}",
                context={
                    "status": result.status.value,
                    "reasons": "; ".join(result.reasons),
                    "detected": spec.provider.value,
                    "expected": self.config.settlement.expected_source.value,
                },
            )
        return result


def _weakest(spec: SettlementSpecification) -> str:
    """Name the field dragging overall confidence down -- the actionable part."""
    scored = [
        (spec.evidence[name].confidence, name)
        for name in REQUIRED_FIELDS
        if name in spec.evidence
    ]
    if not scored:
        return "no evidence"
    confidence, name = min(scored)
    return f"weakest: {name}@{confidence:.2f}"
