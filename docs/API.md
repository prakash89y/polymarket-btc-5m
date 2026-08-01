# API reference

The public surface of each package, taken from `__all__`.

## `pmbtc.settlement`

- **`Check`** *(class)* — One verification gate.
- **`EvidenceSource`** *(class)* — Where a extracted fact came from, ordered by trustworthiness.
- **`FieldEvidence`** *(class)* — Provenance for a single extracted fact.
- **`PARSER_VERSION`** *(object)* — str(object='') -> str
- **`PRICE_PROVIDERS`** *(object)* — Late-bound registry of concrete price clients.
- **`PROVIDER_REGISTRY`** *(object)* — dict() -> new empty dictionary
- **`ProviderDescriptor`** *(class)* — Declarative identity of one settlement authority.
- **`REQUIRED_FIELDS`** *(object)* — Built-in immutable sequence.
- **`SettlementChange`** *(class)* — A detected change in how a series resolves.
- **`SettlementParseError`** *(class)* — The Gamma payload could not be read as a market at all.
- **`SettlementPriceProvider`** *(class)* — Runtime access to a settlement authority's price series.
- **`SettlementReport`** *(class)* — Aggregated verification results for one startup scan.
- **`SettlementSpecStore`** *(class)* — Append-only store of settlement specifications, indexed in memory.
- **`SettlementSpecification`** *(class)* — Normalised, provider-agnostic description of how one market settles.
- **`SettlementVerifier`** *(class)* — Config-driven verification of parsed settlement specifications.
- **`TieRule`** *(class)* — What happens when the settlement price exactly equals the reference.
- **`VerificationResult`** *(class)* — VerificationResult(spec: 'SettlementSpecification', status: 'VerificationStatus', checks: 'tuple[Check, ...]', trading_enabled: 'bool')
- **`VerificationStatus`** *(class)* — Outcome of verifying a detected spec against configuration.
- **`canonical_pair`** *(function)* — Normalise a pair as written in market text to a canonical form.
- **`descriptor_for`** *(function)* — Look up a descriptor, or fail loudly.
- **`open_spec_store`** *(function)* — Open the store at its configured location.
- **`parse_settlement_spec`** *(function)* — Build a canonical specification from a raw Gamma market payload.

## `pmbtc.gamma`

- **`BookLevel`** *(class)* — !!! abstract "Usage Documentation"
- **`DiscoveryResult`** *(class)* — DiscoveryResult(candidates: 'list[MarketCandidate]' = <factory>, rejected: 'list[RejectedMarket]' = <factory>, scanned: 'int' = 0, source: 'str' = 'series', scan_ms: 'float' = 0.0)
- **`DriftSeverity`** *(class)* — Enum where members are also (and must be) strings
- **`GammaClient`** *(class)* — Thin async client for the Gamma and CLOB read APIs.
- **`HealthIssue`** *(class)* — Enum where members are also (and must be) strings
- **`HealthResult`** *(class)* — HealthResult(healthy: 'bool', issues: 'tuple[HealthIssue, ...]', details: 'tuple[str, ...]')
- **`LifecycleTracker`** *(class)* — In-memory state per market, with an append-only transition log.
- **`LiquiditySnapshot`** *(class)* — Point-in-time view of one market's tradeability.
- **`LiquidityStore`** *(class)* — Append-only JSONL sink for snapshots.
- **`MarketCandidate`** *(class)* — A market that passed every gate and may be traded.
- **`MarketDiscovery`** *(class)* — Series-driven discovery with the full acceptance pipeline.
- **`MarketHealthChecker`** *(class)* — Applies the configured health gates to one market.
- **`MarketState`** *(class)* — Where a market is in its life.
- **`RejectedMarket`** *(class)* — RejectedMarket(slug: 'str', condition_id: 'str', stage: 'str', reason: 'str', detail: 'str', snapshot: 'LiquiditySnapshot | None' = None)
- **`ReplayReport`** *(class)* — ReplayReport(parsed: 'int' = 0, unparseable: 'int' = 0, new_markets: 'list[str]' = <factory>, missing_markets: 'list[str]' = <factory>, differences: 'list[Difference]' = <factory>)
- **`ReplaySession`** *(class)* — Loads archived payloads and replays them through the current parser.
- **`ResponseArchive`** *(class)* — Append-only capture of raw API responses, partitioned by UTC day.
- **`SchemaCheckResult`** *(class)* — SchemaCheckResult(fingerprint: 'str', version: 'int', drifts: 'tuple[SchemaDrift, ...]', known: 'bool')
- **`SchemaChecker`** *(class)* — Validates payloads against the contracts and tracks shape versions.
- **`SchemaRegistry`** *(class)* — Persisted record of every payload shape we have seen.
- **`SchemaViolation`** *(class)* — A payload no longer satisfies the contract the parser was written for.
- **`Transition`** *(class)* — Transition(condition_id: 'str', slug: 'str', previous: 'MarketState | None', current: 'MarketState', at_ms: 'int', reason: 'str' = '')
- **`classify`** *(function)* — Derive the current state from time and venue flags.
- **`interpretation_of`** *(function)* — The fields whose meaning must never drift silently.
- **`open_lifecycle_tracker`** *(function)* — 
- **`open_liquidity_store`** *(function)* — 
- **`open_schema_checker`** *(function)* — Build a checker from configuration.
- **`snapshot_from_market`** *(function)* — Build a snapshot from a Gamma market payload.

## `pmbtc.dataset`

- **`BinanceReferenceProvider`** *(class)* — Reference BTC price and displacement from the window open.
- **`CollectionStats`** *(class)* — CollectionStats(markets_tracked: 'int' = 0, snapshots_written: 'int' = 0, snapshots_missed: 'int' = 0, labels_written: 'int' = 0, leakage_blocked: 'int' = 0, misses_by_reason: 'dict[str, int]' = <factory>)
- **`DATASET_RECORD_VERSION`** *(object)* — str(object='') -> str
- **`DatasetExporter`** *(class)* — Writes the training table in the supported formats.
- **`DatasetManifest`** *(class)* — Everything needed to reproduce a training run.
- **`DatasetStats`** *(class)* — DatasetStats(markets: 'int' = 0, labelled: 'int' = 0, unlabelled: 'int' = 0, void: 'int' = 0, snapshots: 'int' = 0, period_start_ms: 'int | None' = None, period_end_ms: 'int | None' = None, class_balance: 'dict[str, int]' = <factory>, settlement_providers: 'dict[str, int]' = <factory>, snapshots_by_horizon: 'dict[int, int]' = <factory>, timeline_completeness: 'float' = 0.0, feature_coverage: 'dict[str, float]' = <factory>, missing_pct_by_source: 'dict[str, float]' = <factory>, mean_latency_by_source: 'dict[str, float]' = <factory>, quality_distribution: 'dict[str, int]' = <factory>, flag_counts: 'dict[str, int]' = <factory>, mean_jitter_ms: 'float' = 0.0)
- **`DatasetStore`** *(class)* — Durable, append-only home for markets and their feature timelines.
- **`ExportResult`** *(class)* — ExportResult(path: 'Path', manifest: 'DatasetManifest', rows: 'int', columns: 'int', format: 'str', excluded_rows: 'int' = 0)
- **`FEATURE_SCHEMA_VERSION`** *(object)* — str(object='') -> str
- **`FeatureProvider`** *(class)* — Contract every feature source implements.
- **`FeatureSnapshot`** *(class)* — All features for one market at one point on the countdown.
- **`HistoricalCollector`** *(class)* — Builds the dataset one market at a time.
- **`ImmutableRecordError`** *(class)* — An attempt to overwrite something the dataset guarantees is immutable.
- **`LabelAudit`** *(class)* — Comparison of the official outcome against an external reference.
- **`LabelResolution`** *(class)* — The outcome of trying to label one market.
- **`LeakageFinding`** *(class)* — LeakageFinding(kind: 'LeakageKind', condition_id: 'str', horizon_seconds: 'int', feature: 'str', detail: 'str')
- **`LeakageGuard`** *(class)* — Write-time enforcement of the leakage invariant.
- **`LeakageKind`** *(class)* — Enum where members are also (and must be) strings
- **`MarketRecord`** *(class)* — Everything known about one market: metadata, timing, and the label.
- **`Observation`** *(class)* — One measured value, with full provenance.
- **`PolymarketQuoteProvider`** *(class)* — The market's own implied probability and book state.
- **`QualityFilter`** *(class)* — Training-time exclusion rules.
- **`QualityFlag`** *(class)* — Why an observation is less than perfect.
- **`QualityScorer`** *(class)* — Applies budgets to raw readings, producing flagged observations.
- **`SnapshotContext`** *(class)* — Everything a provider is allowed to know at snapshot time.
- **`SourceBudget`** *(class)* — Latency and staleness limits for one data source.
- **`TimeFeatureProvider`** *(class)* — Deterministic features of the clock itself.
- **`assert_no_label_leakage`** *(function)* — Confirm no snapshot postdates the resolution of its own market.
- **`audit_against_reference`** *(function)* — Derive what an external feed *would* have said, for comparison only.
- **`build_manifest`** *(function)* — 
- **`build_rows`** *(function)* — Join snapshots to their market metadata and label.
- **`compute_stats`** *(function)* — 
- **`config_hash`** *(function)* — Stable fingerprint of the collection-relevant configuration.
- **`git_commit`** *(function)* — Current commit, or ``"unavailable"`` outside a git checkout.
- **`open_dataset_store`** *(function)* — 
- **`render_stats`** *(function)* — Operator-facing report.
- **`resolve_label`** *(function)* — Extract the official outcome from a resolved Gamma market payload.
- **`scan_monotonicity`** *(function)* — Check that earlier snapshots do not contain later information.
- **`scan_snapshots`** *(function)* — Re-check a whole dataset. Used at export and in the test suite.
- **`stats_for_store`** *(function)* — 

## `pmbtc.live`

- **`BinanceMarketFeed`** *(class)* — Top-of-book and trade tape for one Binance symbol.
- **`BookPair`** *(class)* — The two complementary outcome books of one market.
- **`CollectionService`** *(class)* — Long-running dataset collection with live feeds.
- **`FeedHealth`** *(class)* — Everything needed to decide whether to trust this feed right now.
- **`FeedState`** *(class)* — Enum where members are also (and must be) strings
- **`Level`** *(class)* — Level(price: 'float', size: 'float')
- **`LivePolymarketProvider`** *(class)* — Microstructure from the CLOB stream — the primary source.
- **`LiveReferenceProvider`** *(class)* — BTC reference price and flow from the Binance stream.
- **`MarketSession`** *(class)* — One market's live streaming session.
- **`OrderBook`** *(class)* — One outcome token's book.
- **`PolymarketMarketFeed`** *(class)* — Streams one market's two outcome books and its trade tape.
- **`ReplayFeed`** *(class)* — Feed that replays archived frames instead of connecting.
- **`ServiceStats`** *(class)* — ServiceStats(windows_streamed: 'int' = 0, snapshots: 'int' = 0, labels: 'int' = 0, errors: 'int' = 0, feeds_opened: 'int' = 0, by_state: 'dict[str, int]' = <factory>)
- **`TickArchive`** *(class)* — Append-only gzipped archive of raw feed frames.
- **`Trade`** *(class)* — Trade(price: 'float', size: 'float', sign: 'int', timestamp_ms: 'int')
- **`TradeTape`** *(class)* — Rolling window of trades with flow and volatility metrics.
- **`WebSocketFeed`** *(class)* — Base class for a self-healing streaming feed.
- **`open_tick_archive`** *(function)* — 

## `pmbtc.features`

- **`FeatureRegistry`** *(class)* — The set of features this system is allowed to record.
- **`FeatureSpec`** *(class)* — The declaration every feature must have before it can be recorded.
- **`FeatureTier`** *(class)* — How fresh a feature is, by construction.
- **`REGISTRY`** *(object)* — The set of features this system is allowed to record.
- **`ReproducibilityPolicy`** *(class)* — How a feature can be reconstructed for a backtest.
- **`register`** *(function)* — Convenience wrapper around :meth:`FeatureRegistry.register`.

## `pmbtc.models`

- **`Baseline`** *(class)* — A predictor that emits P(Up).
- **`BaselineReport`** *(class)* — BaselineReport(scores: 'list[ScoreCard]')
- **`CalibrationCurve`** *(class)* — CalibrationCurve(bins: 'list[CalibrationBin]' = <factory>, samples: 'int' = 0)
- **`Calibrator`** *(class)* — Maps raw model scores onto calibrated probabilities.
- **`DEFAULT_SPACES`** *(object)* — dict() -> new empty dictionary
- **`EvaluationMetrics`** *(class)* — The full metric set every model reports.
- **`Explanation`** *(class)* — Global importance, SHAP summary, interactions, and calibration.
- **`Fold`** *(class)* — One train/test split, in row indices.
- **`GateCheck`** *(class)* — GateCheck(name: 'str', passed: 'bool', detail: 'str')
- **`GradientBoostingBaseline`** *(class)* — Default-hyperparameter GBDT — the "just use sklearn" bar.
- **`HyperparameterSearch`** *(class)* — Cross-validated search over the development set.
- **`LogisticBaseline`** *(class)* — Regularised linear model on the feature matrix.
- **`MarketFavourite`** *(class)* — Take the market's own probability at face value.
- **`MarketUnderdog`** *(class)* — The exact inverse of the market's forecast.
- **`ModelCard`** *(class)* — Everything needed to identify, reproduce, and judge one model.
- **`ModelRegistry`** *(class)* — Immutable model artifacts plus a mutable production pointer.
- **`ModelTrainer`** *(class)* — Runs the full training and evaluation pipeline.
- **`PromotionDecision`** *(class)* — PromotionDecision(checks: 'list[GateCheck]' = <factory>, tests: 'list[TestResult]' = <factory>)
- **`RandomPredictor`** *(class)* — Coin flip at a constant 0.5. The floor every model must clear.
- **`ReadinessCheck`** *(class)* — ReadinessCheck(name: 'str', passed: 'bool', required: 'float', actual: 'float', detail: 'str' = '')
- **`ReadinessReport`** *(class)* — ReadinessReport(checks: 'list[ReadinessCheck]' = <factory>, stats: 'DatasetStats | None' = None)
- **`ScoreCard`** *(class)* — How a predictor did on a set of markets.
- **`SearchMethod`** *(class)* — Enum where members are also (and must be) strings
- **`SearchResult`** *(class)* — SearchResult(best: 'Trial | None', trials: 'list[Trial]' = <factory>, method: 'str' = 'grid', note: 'str' = '')
- **`ShadowResult`** *(class)* — Outcome of a chronological replay.
- **`SplitScheme`** *(class)* — Enum where members are also (and must be) strings
- **`TestResult`** *(class)* — Outcome of one statistical comparison.
- **`TimeSeriesSplitter`** *(class)* — Produces time-ordered folds grouped by market.
- **`TrainingData`** *(class)* — A prepared, ordered training set.
- **`TrainingResult`** *(class)* — TrainingResult(card: 'ModelCard | None' = None, readiness: 'ReadinessReport | None' = None, audit: 'AuditReport | None' = None, selection: 'SelectionResult | None' = None, search: 'SearchResult | None' = None, cv_metrics: 'list[EvaluationMetrics]' = <factory>, holdout: 'EvaluationMetrics | None' = None, baselines: 'BaselineReport | None' = None, explanation: 'Explanation | None' = None, shadow: 'ShadowResult | None' = None, decision: 'PromotionDecision | None' = None, blocked_reason: 'str' = '')
- **`assert_no_leakage_between`** *(function)* — Verify a split is genuinely time-ordered and group-disjoint.
- **`binomial_vs_breakeven`** *(function)* — Is a win rate above break-even by more than chance?
- **`brier_score`** *(function)* — Mean squared error of the probability. Lower is better.
- **`calibration_curve`** *(function)* — Reliability diagram with equal-width probability bins.
- **`check_readiness`** *(function)* — Evaluate every gate. Runs all checks so the report is complete.
- **`complete_timeline_count`** *(function)* — Markets that are labelled *and* have every scheduled snapshot.
- **`default_baselines`** *(function)* — Every baseline a candidate model must beat.
- **`evaluate`** *(function)* — Compute every reported metric in one pass.
- **`evaluate_promotion`** *(function)* — Run every gate condition. No path through this returns early on success.
- **`explain_model`** *(function)* — Produce the full explanation bundle for a promoted model.
- **`log_loss`** *(function)* — 
- **`make_model_id`** *(function)* — Human-scannable, collision-resistant model identifier.
- **`mcnemar`** *(function)* — Exact McNemar test on discordant classification pairs.
- **`paired_bootstrap`** *(function)* — Is model A's metric better than model B's, beyond sampling noise?
- **`readiness_for_store`** *(function)* — Convenience wrapper for a :class:`~pmbtc.dataset.store.DatasetStore`.
- **`run_baselines`** *(function)* — Fit and score every baseline on an unseen test set.
- **`run_shadow_validation`** *(function)* — Replay chronologically and verify the four properties.
- **`score`** *(function)* — Accuracy, Brier, and log loss for probabilistic predictions.
- **`stable_hash`** *(function)* — Deterministic hash of any JSON-serialisable structure.
- **`write_report`** *(function)* — Write the full evaluation report next to the model artifact.

## `pmbtc.trading`

- **`CostModel`** *(class)* — Prices an entry against a quote. Pure: no clock, no network, no state.
- **`Decision`** *(class)* — What to do about one window, and why.
- **`DecisionEngine`** *(class)* — Applies the abstention gates to one window.
- **`Fill`** *(class)* — The realised economics of one entry.
- **`PositionSizer`** *(class)* — Turns a decision into an amount of money. Pure and deterministic.
- **`Quote`** *(class)* — A two-sided quote for the UP token, with the depth standing behind it.
- **`RiskLedger`** *(class)* — Tracks exposure and losses, and refuses trades that breach a limit.
- **`RiskState`** *(class)* — Everything the limits are computed from.
- **`Stake`** *(class)* — How much to stake, and what decided it.

## `pmbtc.backtest`

- **`BacktestEngine`** *(class)* — Runs one strategy over one set of settled windows.
- **`BacktestMetrics`** *(class)* — Everything the deployment gate and the operator need to see.
- **`BacktestReport`** *(class)* — Metrics, the gate's verdict, and the statistics behind it.
- **`BacktestResult`** *(class)* — Every window the engine looked at, in order, plus how it ended.
- **`ColumnMap`** *(class)* — Which dataset columns carry the book. Named once, in one place.
- **`ConstantModel`** *(class)* — Always the same probability. Used to exercise the gates in tests.
- **`EstimatorModel`** *(class)* — Wraps a fitted estimator, pinning the column order it was trained on.
- **`FillModel`** *(class)* — Prices and sizes an entry against the book that was recorded.
- **`FillOutcome`** *(class)* — A fill, or a named reason there wasn't one.
- **`GateCheck`** *(class)* — GateCheck(name: 'str', passed: 'bool', detail: 'str')
- **`MarketProbabilityModel`** *(class)* — The book's own forecast. The null strategy, and the bar to beat.
- **`ProbabilityModel`** *(object)* — 
- **`WalkForwardReport`** *(class)* — One report per lookback, plus the combined verdict.
- **`WindowResult`** *(class)* — One evaluated window — traded or not.
- **`compute_metrics`** *(function)* — Summarise a completed run. Pure function of the run's windows.
- **`evaluate_backtest`** *(function)* — Score a completed run against ``config.backtest``.
- **`quote_from_row`** *(function)* — Reconstruct the book as it stood at the snapshot instant.
- **`run_walk_forward`** *(function)* — Run and gate the backtest over every configured lookback.
- **`slice_recent`** *(function)* — Rows settling within ``days`` of the last settlement in the data.

## `pmbtc.ops`

- **`Alert`** *(class)* — Alert(kind: 'AlertKind', severity: 'Severity', message: 'str', detail: 'dict[str, Any]', raised_at_ms: 'int')
- **`AlertKind`** *(class)* — Enum where members are also (and must be) strings
- **`AlertState`** *(class)* — Remembers what is already firing, so a standing problem alerts once.
- **`DailySummary`** *(class)* — DailySummary(generated_at_ms: 'int', stats: 'DatasetStats', readiness: 'ReadinessReport', markets_last_day: 'int' = 0, snapshots_last_day: 'int' = 0, labelled_last_day: 'int' = 0, feed_health: 'list[dict[str, Any]]' = <factory>)
- **`Heartbeat`** *(class)* — Heartbeat(written_at_ms: 'int', pid: 'int', sessions: 'int', snapshots: 'int', labels: 'int', errors: 'int', archived_frames: 'int', clock_status: 'str', feeds: 'list[dict[str, Any]]')
- **`Severity`** *(class)* — Enum where members are also (and must be) strings
- **`alert_state_path`** *(function)* — 
- **`build_summary`** *(function)* — 
- **`dispatch`** *(function)* — Log and record new alerts. Returns the ones actually fired.
- **`evaluate`** *(function)* — Return the alerts that should fire now.
- **`heartbeat_path`** *(function)* — 
- **`read_heartbeat`** *(function)* — 
- **`write_heartbeat`** *(function)* — Write atomically: a torn heartbeat would read as a dead service.
- **`write_summary`** *(function)* — Persist one day's summary. Append-only by filename, never overwritten.
