# Architecture

## The shape of the problem

A Polymarket BTC 5-minute Up/Down market asks one question: will the settlement
price at the end of a 300-second window be at or above the price at its start?
Ties resolve **Up**.

Two facts drive every design decision:

1. **The market is close to efficient.** A liquid book's own price is a
   well-informed forecast. Any system here is trying to beat that, and mostly
   should not trade.
2. **Settlement is decided by a specific feed.** The 5m and 15m families settle
   on **Chainlink** BTC/USD; the sibling **hourly** family settles on **Binance**
   BTCUSDT. Forecasting the wrong series is an invisible, compounding loss.

## Data flow

```
Gamma API ──► discovery ──► settlement verification ──┐
(metadata,      (series-driven,   (9 gates; ambiguity  │
 29-72s stale)   no slug guessing) means no trade)     │
                                                       ▼
CLOB WebSocket ──► order book + trade tape ──► feature engine ──► snapshot
(~155 frames/s)         │                      (70 features,       (T-300…T-1,
                        │                       deterministic)      immutable)
Binance WebSocket ──────┤                            │                  │
(reference price)       │                            │                  ▼
                        ▼                            │            dataset store
                   tick archive ────────────────────►┘            (append-only)
                   (gzipped, replayable)                                │
                                                                        ▼
                                          official Polymarket outcome ──► label
```

## Layers

| layer | responsibility | key invariant |
|---|---|---|
| `settlement` | Which feed decides this market | Ambiguity ⇒ no trade |
| `gamma` | Discovery, metadata, lifecycle, health | No slug is ever constructed |
| `live` | Order book and trade tape | Silence is failure; a gapped book is discarded |
| `features` | The **only** path from data to features | Deterministic, bit-for-bit |
| `dataset` | Storage, quality, labelling | Snapshots immutable; no leakage |
| `models` | Training, evaluation, promotion | No override path |
| `ops` | Liveness, alerts, reporting | Alert only on the four things that matter |

## The invariants

These are enforced by code and tests, not by convention. Each has already caught
a real defect.

**Settlement is verified per market.** Nine gates; the source must be confirmed
by both structured metadata *and* the rules prose, and they must agree. A market
whose family changes its resolution source is blocked until reviewed.

**No feature can postdate the moment it describes.** The leakage guard runs at
write time and again at export. Missing values are legitimate; invented ones are
not.

**One canonical feature pipeline.** Live collection, archive replay, dataset
export, and the training loader are proven to produce identical vectors —
compared as a single fingerprint, not a tolerance.

**Labels come from Polymarket's own resolution.** Never reconstructed from a
price feed. External feeds audit; they do not decide.

**Time-aware validation only.** Splits are grouped by market and ordered in
time, with purge and embargo. There is no shuffle parameter to misuse.

**Calibration over accuracy.** Position size is a function of the predicted
probability, so a miscalibrated model sizes wrongly in exactly the cases where
it is most confident.

**Gates have no overrides.** Settlement verification, the readiness gate, and
the promotion gate cannot be bypassed by a flag. Changing a threshold is a
reviewable commit.

## Safety

Live trading is gated three ways and off by default:

```
app.mode: live   AND   live.enabled: true   AND   PMBTC_I_UNDERSTAND_LIVE_RISK=yes
```

`live.dry_run` defaults true even then, and `data/KILL_SWITCH` halts new orders
immediately without a restart.

## Module status

| # | module | state |
|---|---|---|
| 1 | Infrastructure | done |
| 2 | Settlement verification | done |
| 3 | Market discovery | done |
| 4 | Historical dataset | done |
| 5 | Live market data | done |
| 6 | Feature engineering | done |
| 7 | Model training framework | done, gated on data |
| 8 | Backtesting | pending |
| 9 | Paper trading | pending |
| 10 | Live execution | pending |
| 11 | Monitoring dashboard | pending |
| 12 | Continuous retraining | pending |
