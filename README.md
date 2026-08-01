# pmbtc — Polymarket BTC 5-minute Up/Down bot

Autonomous, self-learning trading system for Polymarket's Bitcoin 5-minute
Up/Down markets. Forecasts whether the official settlement price at the end of
each 5-minute window will be **higher or lower** than the window's reference
price, prices that forecast against the Polymarket book, and trades only when
the edge survives costs, liquidity, and risk limits.

## Status

Built module by module. Each module is reviewed and approved before the next
begins.

| # | Module | Status |
|---|--------|--------|
| 1 | Project architecture (config, logging, time, errors, Docker, tests) | **done** |
| 2 | Settlement verification engine | **done** |
| 3 | Gamma API market parser | **done** |
| 4 | Historical data collector | **done** |
| 5 | Live market data collector | **done** |
| 6 | Feature engineering pipeline | **done** |
| 7 | Model training (GBDT / sequence / ensemble + calibration) | **framework done, gated** |
| 8 | Trading layer + backtesting framework | **done** |
| 9 | Paper trading engine | pending |
| 10 | Live execution engine | pending |
| 11 | Monitoring dashboard | pending |
| 12 | Continuous retraining pipeline | pending |

## Principles this codebase is built on

**Settlement is immutable and verified per market.** The bot forecasts exactly
one price series. Every market's resolution source is detected from its Gamma
payload and must match the configured source with full confidence; anything
ambiguous is skipped, never traded. `settlement.require_verified` and
`settlement.allow_unknown_source` cannot be relaxed outside backtest mode — the
config layer rejects it (see `tests/test_config.py::TestSettlementGates`).

This is not theoretical. Verified against live Gamma payloads on 2026-07-31:

| family | slug shape | resolves off |
|---|---|---|
| `btc-up-or-down-5m` | `btc-updown-5m-<open_epoch>` | **Chainlink** BTC/USD data stream |
| `btc-up-or-down-15m` | `btc-updown-15m-<open_epoch>` | Chainlink BTC/USD |
| `btc-up-or-down-hourly` | `bitcoin-up-or-down-<date>` | **Binance** BTCUSDT |

Three markets in one product line, two different settlement authorities. A bot
that matched on the title "Bitcoin Up or Down" would eventually forecast
Chainlink and get paid on Binance. The 5-minute family also resolves ties **up**
(`>=`), while older quarterly BTC markets resolve an exact tie 50-50 — so even
the tie rule is per-family and must be parsed, not assumed.

**No hand-assigned feature importance.** Features are grouped into
microstructure / short-term-context / regime tiers, which control what gets
computed and how fresh it must be. What survives into the model is decided by
SHAP, mutual information, permutation importance, and RFE, refitted on training
folds at every retrain.

**The production model is never updated per trade.** Every window — traded or
skipped — is appended to `logs/decisions.jsonl` with its features, probabilities,
and eventual settlement. Retraining runs on a schedule; a challenger is promoted
only if it beats the champion on an untouched holdout by a statistically
significant margin, and then shadow-runs before it sizes real money.

**Discovery never constructs a market name.** It asks Gamma for the *series*
(`/events/pagination?series_slug=btc-up-or-down-5m&end_date_min=<now>`) and takes
what comes back, so a change in Polymarket's naming conventions cannot break it.
The observed slug shape is learned and persisted only to keep the emergency
fallback current. Every discovered market runs one gauntlet — schema, parse,
liquidity snapshot, health, settlement verification, lifecycle — and is accepted
only if all of it passes.

**The clock is a service, not a call to `time()`.** Median offset across
independent references, RTT-corrected, with uncertainty tracked and treated
pessimistically at the boundary. It fails closed: never synced, stale, drifted,
or merely uncertain all block order submission. Measured while building this:
the dev machine ran **~0.7–1.3 s behind** true UTC, which is 0.4% of the
instrument.

**A parser change can never silently rewrite history.** Every API response is
archived, and `tests/fixtures/replay_baseline.json` records how the parser reads
50 real markets. `pytest` replays them on every run; a changed interpretation
fails with the market and field named. Accepting it means regenerating the
baseline — a reviewable diff.

**No feature can postdate the moment it claims to describe.** Every observation
carries `event_time`, `observation_time`, `latency`, and `data_source`. The
leakage guard refuses, at write time, any snapshot containing information from
after its own instant — including the subtle case of an event that predates the
instant but only *arrived* afterwards. The scan runs again at export, and the
detector itself is tested against deliberately poisoned data. Snapshots are
immutable and horizon-addressed; a late capture is skipped and counted, never
backdated.

**The label is what Polymarket paid out.** Never a reconstruction from
Chainlink, Binance, or our own tape. Those are audit inputs, and where they
disagree with the official outcome the *disagreement* is recorded and alarmed —
the label does not move. A model trained on a reconstructed label optimises a
different objective than the one that pays.

**Live order flow is the signal; Gamma is metadata.** The CLOB WebSocket
delivered **3,891 frames in 25 seconds** for a single market. Gamma's cached
quotes measured **29–72 seconds stale**. So the CLOB book and trade tape are the
primary microstructure source, and Gamma supplies discovery, metadata,
settlement information, and historical context only — never a low-latency
signal. A feed that goes silent past its budget is treated as dead, and an
incrementally-maintained book is *discarded* on reconnect, because after a gap
it is wrong rather than merely old.

**Every feature declares its tier and its provenance.** `real_time`,
`near_real_time`, `delayed`, `derived`, or `static` — carried into the dataset
so the model always knows how fresh each input was. Unregistered features cannot
be recorded at all. Each one also declares how it can be reconstructed; a
stream-derived feature is reproducible precisely because the raw frames are
archived as they arrive, and one that cannot be reconstructed is excluded from
training by policy rather than by accident.

**Brier score is not money.** Every gate before Module 8 scores forecasts; a
model can beat the market's own forecast on Brier and still lose on every trade
after paying to cross a 1-3 cent spread on a 50-cent contract. So the backtest
prices entries at the touch plus slippage, caps fills at the depth that was
actually resting, and treats an unknown book as no book. Break-even is the
average price paid, not 50% — a contract bought at 0.62 has to win 62% of the
time to return the stake. And a result on fewer than `backtest.min_trades`
trades is reported as **inconclusive** and fails the gate however good it looks,
because ten lucky trades is not evidence. A model is never promoted for a better
Brier score alone: `positive_ev_after_costs` is a separate, mandatory gate.

**Every run says where the money went.** The report is a waterfall from the raw
forecast edge to realised P&L, built as six counterfactual worlds that each
change exactly one thing, so

```
raw_edge − spread − slippage − fees − risk_limits − missed_fills == net_profit
```

holds *exactly* and is asserted against the ledger on every run. Cost lines may
come out negative, and that is information: risk limits which block a losing
trade show up as a saving. This is not decoration — on the collected data it
showed a strategy with **+239 USDC of genuine forecast edge** losing money, with
42 going to spread and slippage and 212 to trades the risk engine refused.

**The cost model is written once.** `pmbtc.trading` holds the whole path from a
probability to a position — decision gates, cost model, sizing, risk ledger —
and the backtest, the paper engine, and live execution all drive that same code.
A cost model that lived inside the simulator would be re-implemented, slightly
differently, by the live path, and the difference would only ever show up in the
P&L.

**No model ships without beating the benchmarks.** Random, market-favourite,
market-underdog, logistic, and gradient boosting — scored on Brier and log loss,
not accuracy, on a chronological holdout. The market-favourite baseline is the
one that matters: beating a liquid book's own forecast is the whole thesis.
Training is additionally gated on dataset size, timeline completeness, feature
coverage, class balance, and quality; `pmbtc readiness` reports exactly which
threshold is short.

**Abstention is the default.** A 5-minute BTC coin flip is close to fair. The
gates (`prediction.min_confidence`, `min_edge`, `min_ev`) are deliberately
strict: the expected number of trades per day is small, and every skip is logged
with its reason so "the bot stopped trading" is always diagnosable.

**Live trading is triple-gated.** `mode: live` **and** `live.enabled: true`
**and** `PMBTC_I_UNDERSTAND_LIVE_RISK=yes` in the environment. `live.dry_run`
defaults to true even then. Module 9's promotion gate must also pass.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env
pmbtc doctor
```

```bash
pytest
```

## Configuration

Precedence: **environment → `.env` → `config/config.yaml` → code defaults.**
Nested keys use a double underscore:

```bash
PMBTC_APP__MODE=backtest
PMBTC_PREDICTION__MIN_EDGE=0.06
```

Secrets never appear in YAML. They are read from the environment into
`BotSecrets` as `SecretStr` and redacted from every log sink, including nested
dicts, lists, and `logging.extra` from third-party libraries.

A YAML section **replaces** the corresponding code default wholesale rather than
merging into it, so every section written in `config/config.yaml` is written in
full.

## Layout

```
config/config.yaml        reviewable behaviour description (no secrets)
src/pmbtc/config.py       typed, layered, validated-at-startup configuration
src/pmbtc/constants.py    domain vocabulary (Outcome, Side, WindowPhase, ...)
src/pmbtc/exceptions.py   typed errors carrying `retryable` / `halts_trading`
src/pmbtc/logging_setup.py structured logs, redaction, append-only decision log
src/pmbtc/utils/timeutils.py UTC-millisecond window arithmetic
src/pmbtc/settlement/     settlement verification engine (Module 2)
  providers.py            pluggable provider descriptors + price protocol
  parser.py               Gamma payload -> spec, structured-first, cross-examined
  spec.py                 canonical SettlementSpecification + evidence model
  verifier.py             the nine gates that decide trading_enabled
  store.py                append-only spec history + change detection
  report.py               startup verification report
src/pmbtc/gamma/          discovery, lifecycle, health, schema, replay (Module 3)
  client.py               async client + archival capture of every response
  discovery.py            series-driven discovery + the acceptance pipeline
  schema.py               payload contracts, versioning, fail-safe drift checks
  lifecycle.py            discovered -> open -> near settlement -> settled
  health.py               identifiers, timestamps, cadence, venue state, book
  liquidity.py            snapshots, captured for accepted and rejected alike
  replay.py               golden-baseline regression over archived payloads
src/pmbtc/dataset/        historical dataset (Module 4)
  schema.py               Observation / FeatureSnapshot / MarketRecord
  leakage.py              write-time guard + dataset-wide scanners
  quality.py              per-source latency budgets -> quality flags
  providers.py            feature providers with mandatory provenance
  collector.py            T-300 ... T-1 countdown scheduler
  labels.py               official Polymarket outcome only (+ audit compare)
  store.py                append-only; snapshots immutable, labels write-once
  manifest.py             dataset/schema versions, git commit, config hash
  export.py               parquet (default) / arrow / csv / sqlite
  stats.py                coverage, balance, latency, quality report
src/pmbtc/live/           live market data (Module 5)
  feed.py                 self-healing socket; silence counts as failure
  clob.py                 Polymarket CLOB book + tape — primary microstructure
  binance.py              BTC reference price and flow
  book.py                 L2 state: imbalance, microprice, depth, slippage
  tape.py                 CVD, volume delta, VWAP, realised vol
  archive.py              gzipped raw frames — makes live features reproducible
  providers.py            live observations for the collector
  service.py              continuous long-running collection
src/pmbtc/features/       feature registry: tiers, provenance, reproducibility
src/pmbtc/models/         readiness gate + benchmark baselines
src/pmbtc/trading/        probability -> position (Module 8, shared with 9/10)
  costs.py                one cost model: spread, slippage, fees, payoff
  decision.py             the abstention gates; every refusal is named
  sizing.py               fractional Kelly, capped well below it
  risk.py                 daily/weekly loss, streaks, exposure, kill switch
src/pmbtc/backtest/       the simulator (Module 8)
  engine.py               chronological replay; the label only settles
  fills.py                touch/mid/aggressive, depth-capped, pessimistic
  metrics.py              money + forecast quality (calibration reused from 7)
  attribution.py          the edge waterfall; reconciles exactly, or says so
  report.py               the deployment gate, incl. positive EV after costs
  walkforward.py          lookback gating + strict refit-per-fold walk-forward
  adapters.py             any estimator -> a probability callable
src/pmbtc/clock.py        UTC sync, drift detection, pre-settlement safety gate
src/pmbtc/metrics.py      counters/gauges/histograms + Prometheus rendering
src/pmbtc/cli.py          `doctor` / `discover` / `train` / `backtest` / ...
tests/fixtures/gamma/     real archived Gamma payloads (2 providers, 3 eras)
tests/fixtures/archive/   captured responses for the replay regression test
data/                     local history, parquet, sqlite, KILL_SWITCH
logs/decisions.jsonl      the bot's memory — never rotated, never purged
```

## Operational notes

- `data/KILL_SWITCH`: create this file to block all new orders immediately.
- `logs/decisions.jsonl` is the training set. Back it up; deleting it costs the
  bot everything it has learned.
- Docker: `docker compose up --build` (paper mode, JSON logs, volumes for data
  and logs). Build with `--build-arg EXTRAS=".[data,ml,trade]"` for the full
  runtime.

## Risk

This trades real money against a market that is close to efficient at a
5-minute horizon. Paper trading must clear the statistical gate in
`config.paper` before live mode will arm. Position sizing is quarter-Kelly by
default with hard caps above it. Use a dedicated wallet funded with only the
bankroll you are prepared to lose, and confirm that trading on Polymarket is
permitted in your jurisdiction.
