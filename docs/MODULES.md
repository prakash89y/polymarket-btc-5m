# Module reference

Generated from package docstrings. Edit the code, not this file.

## `pmbtc.settlement`

Settlement verification engine — the immutable core of the system.

Pipeline:

    raw Gamma market
        -> parse_settlement_spec()      canonical spec + per-field evidence
        -> SettlementSpecStore.record() durable history, change detection
        -> SettlementVerifier.verify()  config gates -> trading_enabled
        -> SettlementReport             operator report + audit log

Nothing downstream may place an order without a
:class:`~pmbtc.settlement.verifier.VerificationResult` whose
``trading_enabled`` is true.

| submodule | summary |
|---|---|
| `parser` | Gamma market -> canonical :class:`SettlementSpecification`. |
| `providers` | Pluggable settlement price providers. |
| `report` | The startup settlement verification report. |
| `spec` | The canonical settlement specification and its evidence model. |
| `store` | Durable record of how every market settled. |
| `verifier` | Verify a detected specification against configuration and recorded history. |

## `pmbtc.gamma`

Gamma API layer: discovery, lifecycle, health, liquidity, schema, replay.

GammaClient          network + archival capture
  -> SchemaChecker   fail safe on incompatible payloads
  -> parser          canonical settlement spec (Module 2)
  -> LiquiditySnapshot
  -> MarketHealthChecker
  -> SettlementVerifier
  -> LifecycleTracker
ReplaySession        regression-test the parser against the archive

| submodule | summary |
|---|---|
| `client` | Async Gamma API client with archival capture. |
| `discovery` | Dynamic market discovery and the full acceptance pipeline. |
| `health` | Per-market health checks. |
| `lifecycle` | Market lifecycle tracking. |
| `liquidity` | Liquidity snapshots. |
| `replay` | Replay archived Gamma responses through the current parser. |
| `runtime` | Assembly of the discovery stack. |
| `schema` | Gamma payload schema contracts, fingerprinting, and drift detection. |

## `pmbtc.dataset`

Historical dataset: collection, quality, labelling, versioning, export.

discovery (Module 3)
  -> HistoricalCollector   countdown scheduler, T-300 ... T-1
  -> FeatureProvider(s)    observations with full provenance
  -> QualityScorer         latency budgets -> quality flags
  -> LeakageGuard          write-time refusal of future information
  -> DatasetStore          append-only, snapshots immutable
  -> resolve_label()       official Polymarket outcome only
  -> DatasetExporter       parquet / arrow / csv / sqlite + manifest
  -> DatasetStats          coverage, balance, latency, quality report

| submodule | summary |
|---|---|
| `audit` | Pre-training data audit. |
| `collector` | The historical collector. |
| `export` | Dataset export. |
| `labels` | Label resolution -- from the official Polymarket outcome, and nothing else. |
| `leakage` | Leakage prevention. |
| `manifest` | Dataset versioning: what a training run can point at and say "this, exactly". |
| `providers` | Feature providers. |
| `quality` | Latency budgets and observation quality. |
| `schema` | Dataset record types. |
| `stats` | Dataset statistics. |
| `store` | Append-only dataset store. |

## `pmbtc.live`

Live market data — the primary microstructure source.

    WebSocketFeed            self-healing socket; silence counts as failure
      -> PolymarketMarketFeed  CLOB book + trade tape (primary)
      -> BinanceMarketFeed     reference price + flow
      -> TickArchive           raw frames, so live features stay reproducible
      -> Live*Provider         observations for the Module 4 collector
      -> CollectionService     continuous, long-running collection

Gamma is not a feed. Measured at 29-72 seconds stale, it supplies discovery,
metadata, settlement information, and historical context only.

| submodule | summary |
|---|---|
| `archive` | Raw stream archive — the price of using live features at all. |
| `binance` | Binance BTC feed — the reference-price microstructure source. |
| `book` | Level-2 order book state and microstructure metrics. |
| `clob` | Polymarket CLOB market feed — the primary microstructure source. |
| `feed` | Resilient WebSocket feed base class. |
| `fixtures` | Archive source resolution: committed fixture versus live collection. |
| `providers` | Live feature providers. |
| `service` | Continuous collection service. |
| `tape` | Trade tape: cumulative volume delta, flow, and realised volatility. |

## `pmbtc.features`

Feature declarations and the registry that enforces them.

Module 6 builds the engineering pipeline on top of this; Module 5 already
depends on it, because no feature may be recorded without first declaring its
tier, source, and reproducibility policy.

| submodule | summary |
|---|---|
| `base` | Feature definitions and the computation context. |
| `consistency` | Cross-source consistency. |
| `docs` | Automatic feature documentation. |
| `drift` | Feature drift monitoring. |
| `engine` | The feature engine: deterministic, incremental computation. |
| `families` | Feature families. |
| `graph` | Feature dependency graph: ordering, cycle detection, and impact analysis. |
| `matrix` | The versioned feature matrix — Module 6's output. |
| `provider` | The canonical feature provider — the only path from market data to features. |
| `registry` | Feature registry: tiers, provenance, and reproducibility policy. |
| `selection` | Evidence-based feature selection. |

## `pmbtc.models`

Modelling: readiness gates and benchmark baselines.

Module 7 (training) is deliberately gated behind both:

* :func:`~pmbtc.models.readiness.check_readiness` refuses to train until the
  dataset is large, complete, balanced, and clean enough for a result to mean
  anything.
* :mod:`~pmbtc.models.baselines` defines the bar every model must clear on
  unseen data before it may be promoted -- above all, the market's own forecast.

| submodule | summary |
|---|---|
| `baselines` | Benchmark baselines. |
| `calibration` | Calibration metrics and calibrators. |
| `explain` | Explainability for promoted models. |
| `promotion` | The promotion gate. |
| `readiness` | Model readiness gate. |
| `registry` | Model registry: identity, reproducibility, and artifacts. |
| `search` | Hyperparameter search — inside the training folds, never outside them. |
| `shadow` | Shadow validation: replay a trained model across history, in order. |
| `statistics` | Statistical comparison of models. |
| `train` | Training orchestration. |
| `validation` | Time-aware validation. |

## `pmbtc.trading`

The trading layer: the path from a probability to a position.

Modules 1-7 answer "what will happen". This package answers "does that belong
in a position, at what price, and for how much" — and it is deliberately
separate from :mod:`pmbtc.backtest`, because the same four questions have to be
answered identically in simulation, in paper, and with real money. A cost model
that lives inside the simulator is a cost model the live path will eventually
re-implement, slightly differently, and the difference will only show up in the
P&L.

    Quote + model probability
      -> DecisionEngine   gates -> trade or a named SkipReason
      -> CostModel        entry price, fees, round-trip cost
      -> PositionSizer    fractional Kelly, capped
      -> RiskLedger       portfolio limits, drawdown, kill switch

Nothing here touches the network, the clock, or a random number generator.
Every function is pure in its inputs, so the same market state produces the same
decision in a backtest as it does at 3 a.m. against the live book.

| submodule | summary |
|---|---|
| `costs` | The cost model. |
| `decision` | The decision gate: probability in, position or a named refusal out. |
| `risk` | Portfolio risk limits — the layer that survives a wrong model. |
| `sizing` | Position sizing: fractional Kelly, with the caps that matter more than Kelly. |

## `pmbtc.backtest`

Module 8 — backtesting: does the edge survive contact with the book?

Every gate before this one scores *forecasts*. Brier score and log loss say how
well the model knows the world; they say nothing about money. On a market where
the spread is 1-3 cents and the whole edge is a few percentage points, the
difference is decisive: a model can beat the market's own forecast on Brier and
still lose on every single trade after paying to cross the spread. This module
is the first thing in the system that measures P&L, and therefore the first
thing that can answer whether to trade at all.

    labelled rows (Module 4/6)
      -> BacktestEngine     one decision per window, chronological
           DecisionEngine   trade or a named SkipReason      (pmbtc.trading)
           PositionSizer    fractional Kelly, capped         (pmbtc.trading)
           FillModel        what the book would actually have given us
           RiskLedger       daily/weekly/streak stand-downs  (pmbtc.trading)
      -> BacktestMetrics    ROI, profit factor, Sharpe, drawdown, Brier
      -> BacktestReport     the deployment gate + walk-forward stability

Three properties are deliberate:

*It is pessimistic by construction.* Fills cross the spread, pay slippage, and
are capped by the depth that was actually resting. Missing depth data is treated
as no depth, not as infinite depth.

*It is deterministic.* No clock reads, no RNG, no set iteration order. The same
dataset and config produce byte-identical results, which is what makes a
regression in the strategy detectable rather than arguable.

*It cannot conclude from noise.* A result computed on fewer than
``backtest.min_trades`` trades is reported as inconclusive and fails the gate,
however good the numbers look.

| submodule | summary |
|---|---|
| `adapters` | Adapters from "a thing that predicts" to the engine's probability callable. |
| `attribution` | Edge decomposition: where the money actually came from, and where it went. |
| `engine` | The simulator. |
| `fills` | What the book would actually have given us. |
| `metrics` | Performance metrics for a completed run. |
| `report` | The deployment gate. |
| `walkforward` | Walk-forward evaluation over several lookbacks. |

## `pmbtc.ops`

Operations: liveness, daily reporting, and alerting.

write_heartbeat / read_heartbeat   is collection actually running
build_summary                      daily dataset + readiness report
evaluate / dispatch                the four alert conditions worth firing on

| submodule | summary |
|---|---|
| `alerts` | Alerting — only for the four conditions worth waking someone for. |
| `gitinfo` | Git provenance for experiment tracking. |
| `heartbeat` | Liveness heartbeat. |
| `summary` | Daily dataset summary and readiness progress. |
