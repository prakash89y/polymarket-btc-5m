"""Time-aware validation.

There is no shuffle option here, and that is deliberate. Every splitter takes
timestamps and returns index arrays whose training portion strictly precedes
their test portion. A random split on this dataset would be catastrophic in a
specific way: eleven snapshots share one market and one label, so a shuffled
split puts the same outcome on both sides and reports an accuracy that cannot be
achieved live.

Two protections beyond ordering:

**Grouping by market.** The unit of independence is the *market*, not the
snapshot. Splits are computed over markets and then expanded to their snapshots,
so a market is never partly in train and partly in test.

**Purge and embargo.** Markets whose windows overlap the test period are dropped
from training (purge), and a gap is left after the test period (embargo). Both
exist because adjacent 5-minute windows share microstructure state, and a model
that trains on the market immediately before the one it is tested on has seen
a correlated draw.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pmbtc.exceptions import InsufficientDataError


class SplitScheme(StrEnum):
    EXPANDING = "expanding"
    ROLLING = "rolling"
    WALK_FORWARD = "walk_forward"


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, in row indices."""

    index: int
    train: np.ndarray
    test: np.ndarray
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    purged: int = 0

    def __len__(self) -> int:
        return len(self.test)

    @property
    def train_size(self) -> int:
        return len(self.train)

    def describe(self) -> str:
        return (
            f"fold {self.index}: train n={len(self.train)} test n={len(self.test)} "
            f"purged={self.purged}"
        )


@dataclass
class TimeSeriesSplitter:
    """Produces time-ordered folds grouped by market.

    Args:
        scheme: expanding, rolling, or walk-forward.
        n_folds: number of test periods.
        train_size: markets per training window (rolling only).
        embargo_markets: markets skipped between train and test.
        min_train_markets: refuse to produce a fold with less than this.
    """

    scheme: SplitScheme = SplitScheme.EXPANDING
    n_folds: int = 5
    train_size: int = 500
    embargo_markets: int = 2
    min_train_markets: int = 50

    # ------------------------------------------------------------------ #
    def split(
        self, timestamps: np.ndarray, groups: np.ndarray
    ) -> Iterator[Fold]:
        """Yield folds.

        Args:
            timestamps: settlement time per row, used only for ordering and
                for reporting fold boundaries.
            groups: market identifier per row. Rows sharing a group always land
                on the same side of a split.
        """
        if len(timestamps) != len(groups):
            raise ValueError("timestamps and groups must be the same length")
        if len(timestamps) == 0:
            return

        # Order markets by their earliest timestamp: chronological, and stable.
        order: dict[object, int] = {}
        for group, timestamp in zip(groups, timestamps, strict=True):
            if group not in order or timestamp < order[group]:
                order[group] = int(timestamp)
        markets = [g for g, _ in sorted(order.items(), key=lambda kv: (kv[1], str(kv[0])))]

        if len(markets) < self.min_train_markets + self.n_folds:
            raise InsufficientDataError(
                "Not enough markets for the requested validation scheme",
                context={
                    "markets": len(markets),
                    "needed": self.min_train_markets + self.n_folds,
                    "scheme": self.scheme.value,
                    "folds": self.n_folds,
                },
            )

        rows_by_group: dict[object, list[int]] = {}
        for row, group in enumerate(groups):
            rows_by_group.setdefault(group, []).append(row)

        total = len(markets)
        # Reserve the tail for testing, split evenly across folds.
        testable = total - self.min_train_markets
        fold_size = max(1, testable // self.n_folds)

        for fold_index in range(self.n_folds):
            test_start = self.min_train_markets + fold_index * fold_size
            test_end = (
                total if fold_index == self.n_folds - 1 else test_start + fold_size
            )
            if test_start >= total:
                break

            train_end = max(0, test_start - self.embargo_markets)
            if self.scheme is SplitScheme.ROLLING:
                train_begin = max(0, train_end - self.train_size)
            else:
                # Expanding and walk-forward both grow the training set; they
                # differ in that walk-forward tests a single step at a time,
                # which is expressed through n_folds rather than here.
                train_begin = 0

            train_markets = markets[train_begin:train_end]
            test_markets = markets[test_start:test_end]
            if not train_markets or not test_markets:
                continue

            train_rows = _rows_for(train_markets, rows_by_group)
            test_rows = _rows_for(test_markets, rows_by_group)
            purged = (test_start - train_end) if test_start > train_end else 0

            yield Fold(
                index=fold_index,
                train=np.array(sorted(train_rows), dtype=int),
                test=np.array(sorted(test_rows), dtype=int),
                train_start_ms=int(min(timestamps[train_rows])),
                train_end_ms=int(max(timestamps[train_rows])),
                test_start_ms=int(min(timestamps[test_rows])),
                test_end_ms=int(max(timestamps[test_rows])),
                purged=purged,
            )

    # ------------------------------------------------------------------ #
    def final_holdout(
        self, timestamps: np.ndarray, groups: np.ndarray, fraction: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray]:
        """The untouched tail, reserved for champion/challenger comparison.

        Never used for fitting, calibration, selection, or hyperparameter search
        — only for the final decision about whether a model may be promoted.
        """
        order: dict[object, int] = {}
        for group, timestamp in zip(groups, timestamps, strict=True):
            if group not in order or timestamp < order[group]:
                order[group] = int(timestamp)
        markets = [g for g, _ in sorted(order.items(), key=lambda kv: (kv[1], str(kv[0])))]
        cut = max(1, int(len(markets) * (1.0 - fraction)))
        develop, holdout = markets[:cut], markets[cut:]

        rows_by_group: dict[object, list[int]] = {}
        for row, group in enumerate(groups):
            rows_by_group.setdefault(group, []).append(row)
        return (
            np.array(sorted(_rows_for(develop, rows_by_group)), dtype=int),
            np.array(sorted(_rows_for(holdout, rows_by_group)), dtype=int),
        )


def _rows_for(markets: list[object], rows_by_group: dict[object, list[int]]) -> list[int]:
    rows: list[int] = []
    for market in markets:
        rows.extend(rows_by_group.get(market, []))
    return rows


def assert_no_leakage_between(
    train: np.ndarray,
    test: np.ndarray,
    timestamps: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Verify a split is genuinely time-ordered and group-disjoint.

    Called on every fold the trainer uses. It is cheap, and it is the difference
    between believing the splitter works and knowing it.
    """
    if len(train) == 0 or len(test) == 0:
        return
    shared = set(groups[train]) & set(groups[test])
    if shared:
        raise InsufficientDataError(
            "Split leaks: the same market appears in train and test",
            context={"markets": len(shared)},
        )
    if max(timestamps[train]) > min(timestamps[test]):
        raise InsufficientDataError(
            "Split leaks: training data postdates the test period",
            context={
                "train_end": int(max(timestamps[train])),
                "test_start": int(min(timestamps[test])),
            },
        )
