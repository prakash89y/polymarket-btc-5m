# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Four schema versions move independently of the package version — feature,
dataset, settlement, and model. Each release records all four, because a
package upgrade that leaves the schemas alone is a very different upgrade from
one that does not.

## [Unreleased]

---

## [0.13.0] — 2026-08-03 — Collector supervision

The last operational blocker. v0.12.0 fixed everything wrong *inside* the
collector; this makes it survive the machine it runs on.

The collector died twice, and neither time was a defect in it. Windows event
``VSS 8193, hr=0x8007045b`` ("A system shutdown is in progress") was logged at
14:01:35 UTC against a final collector line at 14:01:23 UTC, and the host booted
again eighteen hours later with nothing to bring collection back. A process
started by hand from a shell survives exactly as long as the machine does.

### Added

**`pmbtc supervise`** — keeps exactly one collector running. Restarts it on
exit with bounded-exponential backoff, and restarts it when *hung*: a process
that is running but no longer writing a heartbeat is the failure a PID check
cannot see. Consecutive fast failures raise
`AlertKind.SUPERVISOR_RESTART_FAILING` through the **existing**
`ops.alerts` engine — no second implementation of any condition.

**`pmbtc supervisor-status`** — reports the five supervised checks read-only.

**Single-instance guarantee.** A named kernel mutex on Windows, `flock`
elsewhere. Both are held by the process, so a supervisor killed with `-9`
releases its claim immediately; a PID file would survive the death it describes
and lock out every future start. Two collectors writing one append-only store is
a data-integrity problem, not a performance one.

**Scripts.** `install_supervisor.ps1`, `uninstall_supervisor.ps1`,
`verify_supervisor.ps1`.

### Two real bugs found during validation, both from launcher shims

**The venv's `python.exe` is itself a launcher** that re-execs the base
interpreter, so every logical process appears twice in the process table and the
supervised PID is never the PID that writes the heartbeat — measured here: child
`16132`, writer `18556`. Ownership is therefore established with a **run token**
minted per supervisor start, passed through the environment and echoed back in
the heartbeat. It survives any number of exec hops and is strictly stronger than
a PID: it identifies the *run*, so a stale heartbeat from a previous supervisor
is rejected too. (`collector_command()` already avoided the `pmbtc.exe` shim for
this reason; the venv shim was the same trap one level down.)

**`verify_supervisor.ps1` miscounted** for the same reason, reporting two
supervisors and two collectors where there was one of each. It now counts leaf
processes only. A second PowerShell defect was found alongside it: a
single-element array is unrolled on return, so `.Count` was empty rather than
`1`; call sites now force array context.

A **startup grace period** was added after observing one spurious
`supervisor.degraded` line 18 ms after each spawn — the new collector had not yet
written its first heartbeat, so the file on disk still belonged to its
predecessor. Alarming on a known-transient state is how an operator learns to
ignore the log.

### Validated end to end, against live processes

| Check | Result |
|---|---|
| single supervisor | PASS — 1 (pid 7820) |
| single collector | PASS — 1 (pid 11888) |
| heartbeat owner | PASS — 10s old, token matched, clock healthy |
| duplicate prevention | PASS — second supervisor exited 2, still 1 collector |
| **kill recovery** | **PASS — collector restarted as pid 1392 (was 11888)** |
| steady state | 0 `supervisor.degraded` lines after the grace fix |

### Known limitation

Task Scheduler refuses registration without elevation on this host — "Access is
denied" from both `Register-ScheduledTask` and `schtasks.exe`, even for a
logon-only task. The install script falls back to a **Startup-folder shortcut**,
which needs no privilege and restores collection at sign-in. The window between
boot and sign-in is uncovered; re-running the installer from an elevated
PowerShell registers the at-boot trigger and closes it.

---

## [0.12.0] — 2026-08-02 — Production data pipeline reliability

No new functionality. Seven operational root causes, each proven by measurement
before a line was changed, and each fixed at the cause rather than the symptom.

Six competing explanations were **disproven** and deliberately not fixed:
event-loop starvation (loop lag p99 **17.1 ms** under full production shape,
`dropped=0`), receive starvation, archive I/O (0 reconnects in a 240 s probe
with archiving on), network instability (18,158 frames / 0 reconnects
standalone), the websocket library, and the feed implementation (raw library and
our feed behave identically). No queue, thread or extra concurrency was added.

### Fixed

**1-2. The clock synchronised once and never again.** `ClockService.run_forever`
existed from the first version of the module and was **never started by
anything** — dead code behind `# pragma: no cover`. A 13.5-hour run logged
exactly one `clock.synced` line, so `_last_sync_ms` never advanced and the
status was `STALE` for all but the first five minutes. It is now supervised by
`discovery_stack`, with every iteration guarded so one bad network minute cannot
kill the loop.

**3. A degraded sample became permanent.** That single startup sample
self-reported `DRIFTED` *and* `UNCERTAIN` (offset 1,636 ms against
`max_drift_ms` 1,500; uncertainty 927 ms against `max_uncertainty_ms` 750) and
was applied for 13.5 hours. The measured true offset at the time was **297 ms**,
so every timestamp the collector wrote was 1.34 s late. `add_sample` now refuses
a sample that is *worse* than a still-usable incumbent — while never letting
precision outrank freshness, since preferring an accurate memory over a fresh
reading is what froze the offset in the first place.

**4-5. Transport liveness was conflated with data freshness.** One budget drove
both the reconnect trigger and the freshness gate. Across **1,353,586 archived
CLOB frames not one inter-frame gap exceeded the 15-second budget** — the socket
never stalls. What does happen is that **17% of BTC 5-minute markets are thin
enough to say nothing for 15 seconds**, and a market outside its window says
nothing at all. Silence tore down a healthy connection, `on_disconnect` then
correctly discarded the book, and the book was destroyed and rebuilt on a loop.

A quiet feed now stays connected. **No threshold moved and no gate weakened**:
the budget is unchanged, the feed is still marked `STALE`, `is_fresh()` still
refuses it and trading still fails closed. Only the response changed. Because
data arrival no longer doubles as a liveness signal, CLOB keepalive pings are
enabled and `ping_timeout` is bounded — an unanswered ping is now the only thing
that can prove the transport is gone.

**6. Subscriptions outlived their markets.** Sessions were closed at
`settlement + cooldown`; all eleven horizons complete before settlement, so the
cooldown held a subscription open on a market that had stopped existing. Now
closed at settlement.

**7. Alerts were unreachable from the process they describe.**
`ops.alerts.evaluate`/`dispatch` were called from exactly one site —
`cli.py`, inside the manual `pmbtc watch` command. A collector that died left a
262-minute-stale heartbeat and nothing said a word. The **existing** engine is
now driven from the service status tick (no second implementation of any
condition), plus a heartbeat self-check for the one blind spot the heartbeat
file cannot report: its own failure to be written.

### Measured, before vs after

| | before | after |
|---|---|---|
| `clock.synced` rate | 0.07/h (1 in 13.5 h) | **68/h** |
| clock status | `stale` 1665, `drifted` 11, healthy **0** | **healthy 100%** |
| distinct clock offsets | **1** (frozen) | 5 and moving |
| `feed.stale` | 46.5/h | **0** |
| `feed.error` | 13.9/h | **0** |
| `feed.connected` | 66.2/h | 28.3/h |
| CLOB observation latency | median 18,989 ms, p90 149,778 ms | **median 0 ms, p90 77 ms** |
| snapshot quality (median) | 0.471 — below the 0.5 floor | **0.814** |
| snapshots rejected | **85%** | **17%** |
| `out_of_budget` + `stale` flags | 49% of observations | **0%** |
| `ok` flags | 33% | **77%** |

### Notes
`tests/test_pipeline_reliability.py` pins each root cause, including an explicit
assertion that the staleness budgets and quality floors are unchanged. Replay
remains byte-identical and all module validators pass.

---

## [0.11.0] — 2026-08-01 — Module 8.5: market edge validation

The statistical review of Module 8 found a specific, reproducible pathology.
Every threshold in the system was a **floor** — a minimum confidence, a minimum
edge, a minimum expected value — and a model's *apparent* edge grows with its
error. The worse the forecast, the larger the disagreement with the book, the
more eagerly the system traded, and the more reliably it took the opposite side
of a market that turned out to be right.

Measured at T-30 on collected data: median claimed edge **0.83 probability
points**, median expected value **+680% per five-minute trade**, and **84% of
intents taken against books quoted beyond 0.80**. One trade bought DOWN at 0.035
against a market pricing UP at 0.975, on a model probability of 0.239, and
reported +2,074% EV. It lost the full stake.

This release supplies the missing ceilings. It adds no execution code, no paper
trading, and no live trading.

### Added

**`pmbtc.trading.validation` — three ceilings.**
- **Disagreement bound**, measured in **log-odds** rather than probability
  points. Calling a 0.50 market at 0.75 is an ordinary opinion; calling a 0.97
  market at 0.72 is an extraordinary one. Both are 0.25 in probability space and
  indistinguishable there; in log-odds they are 1.10 and 2.53. Penalising
  disagreement with confident books falls out of the arithmetic rather than
  needing a special case. Default ceiling 1.5 logits.
- **Plausibility band** (`min_plausible_prob`, default 0.02). No honest forecast
  of a five-minute coin flip reaches 0.999; the logistic baseline emitted
  exactly 0.001 on real data.
- **Robust anomaly monitor**, median/MAD rather than mean/stdev — because the
  thing being detected is precisely what corrupts a mean and inflates a standard
  deviation until the next outlier looks unremarkable. Returns *no opinion*
  rather than "normal" before it has enough history, so a cold start cannot wave
  everything through.

**Calibration-adjusted EV shrinks toward the market, not toward 0.5.** The
review tested every implied-probability bucket at every horizon and found none
mispriced (Bonferroni p >= 0.25 throughout). When the prior is "the book is
right", the correct shrinkage target for an untrusted model is the book's price.
Shrinking toward 0.5 would *manufacture* edge against confident markets — the
exact error being corrected. `prediction.model_trust` defaults to 1.0, so the
adjusted view is reported before it is enforced; what value is justified is an
empirical question and `pmbtc edge-scan` is what answers it.

**`pmbtc.backtest.edgescan` and `pmbtc edge-scan`.** Horizon choice by strict
walk-forward only, with the model refit inside every fold at every candidate
horizon — picking a horizon from the whole history would be the overfitting this
module exists to prevent. A horizon is called STABLE only if it is profitable,
profitable in **most folds**, *and* beats the book's own forecast: profit from
one fold is a coincidence, a good Brier without profit is the trap Module 8 was
built to expose, and profit without beating the market is unexplained. Also
produces the edge stability report and the disagreement distribution histogram.

Three new `SkipReason` members: `EXCESSIVE_DISAGREEMENT`,
`IMPLAUSIBLE_PROBABILITY`, `ANOMALOUS_EDGE`.

### Changed
The disagreement ceiling is a deliberate behaviour change and it binds. Tests
and the Module 8 validator that used `ConstantModel(0.9)`/`(0.95)` against a
0.50 book (2.20 and 2.94 logits) now use 0.80 (1.39 logits) — they exercise
fills, metrics and risk, not disagreement, and 0.80 is the strongest claim that
clears the ceiling. CI additionally asserts the reviewed pathological trade
cannot become a trade, *and* that an ordinary claim still can, so the ceiling
cannot be tightened into a system that never trades.

### Notes
No schema moved. Six new config fields, all additive; the ceilings default ON
because they prevent pathology, while blending defaults OFF because it is a
modelling choice that must be evidenced first.

Run against the collected data (98 markets), `pmbtc edge-scan` reports **NO
EVIDENCE OF EDGE at any of 6 horizons**. The disagreement histogram explains
why: the logistic baseline sits a median of **2.5–3.3 logits** from the book at
every horizon, with **74–83% of windows over the ceiling** — categorically
broken, at every horizon, and now visible in one report rather than derivable by
hand. T-60 showed +82% return on stake, which the stability test correctly
refuses: 3 trades, two folds with none, and Brier skill −0.59. Without the
three-condition test somebody would have deployed it.

---

## [0.10.0] — 2026-08-01 — Edge decomposition, strict walk-forward, EV gate

Module 8 completed to its full brief. v0.9.0 built the probability-to-money
pipeline; this release makes it *explain itself*, and adds the two gates that
stop a good forecast being mistaken for a good strategy.

### Added

**Edge decomposition (`backtest/attribution.py`).** A waterfall from the raw
forecast edge to realised P&L, built as six counterfactual worlds applied in
runtime order — mid → spread → slippage → fees → risk limits → fill
constraints. Because each line is the difference of two adjacent worlds, the
identity

    raw_edge - spread - slippage - fees - risk - missed_fills == net_profit

holds exactly rather than approximately, and is asserted against the engine's
own bankroll on every run. Cost lines may be negative: risk limits that block a
losing trade *save* money, and reporting only the cases where a constraint hurt
would make the risk engine look like pure overhead.

**Strict walk-forward (`run_strict_walk_forward`).** Refits the model at every
fold and only ever trades forward. Splitting is delegated to
`models.validation.TimeSeriesSplitter` rather than reimplemented — it already
groups by market and purges the adjacent windows, and adjacent 5-minute markets
share microstructure state. Bankroll resets each fold, so one lucky early fold
cannot inflate the size of every later trade.

**The gates the brief exists to enforce.** `positive_ev_after_costs` requires
net profit *and* positive return on stake after execution; a better Brier score
is explicitly not a reason to deploy. `decomposition_reconciles` verifies the
attribution against the ledger, so a broken decomposition fails loudly rather
than mis-attributing quietly.

**Metrics completed.** Calibration, reliability (ECE/MCE + the diagram), fill
rate, average holding time, average quoted spread, average spread paid, average
slippage, and trade frequency. `--strict` and `--folds` added to `pmbtc backtest`.

**Latency (`costs.assumed_latency_ms`, default 500ms).** The book must still be
fresh when the order *lands*, not when we looked at it. **Confidence shrinkage
(`prediction.confidence_shrinkage`, default 0.0)** pulls probabilities toward
0.5 in proportion to recent calibration error.

### Fixed

**A real bug, found by the decomposition on live data.** The fair (spread-free)
reference price was recorded as the **UP** mid regardless of which side was
bought. A DOWN token is the complement, so its fair price is `1 - mid_up`, and
against the wrong reference the ladder reported *crossing the spread as a
+146 USDC profit*. Spread cost is now non-negative in both directions and the
regression test asserts it across UP/DOWN, winning/losing and skewed books. The
bound is `>= 0` rather than `> 0` deliberately: a losing trade forfeits its
stake whatever it paid, so entry price only moves the winning branch.

**A duplication introduced in v0.9.0.** `backtest/metrics.py` had defined its own
`_brier`, `_log_loss` and `_accuracy`, duplicating `models/calibration.py` —
which the promotion gate uses. Two implementations would eventually disagree,
and a model would pass one gate and fail the other for no visible reason.
Deleted in favour of `calibration.evaluate()`, which also supplied the required
reliability curve for free.

### Notes
Schemas are unchanged — feature, dataset, settlement and model. The two new
config fields are additive with behaviour-preserving defaults.

Measured on the collected data (77 labelled markets), strict walk-forward over 4
folds: pooled ROI -1.45% on 15 trades, 1 of 4 folds profitable, INCONCLUSIVE as
expected below `min_trades`. The decomposition is the interesting part — raw
forecast edge **+239.65 USDC**, of which spread took 21.86 and slippage 20.09,
and the risk engine's 16 refusals cost 212.23, netting -14.53. The forecast had
edge; the execution and the limits ate it. That sentence was not available
before this release.

---

## [0.9.0] — 2026-08-01 — Module 8: the trading layer and the backtest

The first module that measures money. Everything before it scores *forecasts* —
Brier, log loss, calibration — and on this instrument that is not the same
question. The spread is 1-3 cents on a contract worth about 50, so a model can
beat the market's own forecast and still lose on every trade it places. Until
now the system had no way to detect that.

### Added

**`pmbtc.trading` — the shared path from a probability to a position.**
Deliberately a separate package from the simulator, because the same four
questions have to be answered identically in backtest, in paper, and with real
money. `CostConfig`'s own docstring already said "one cost model, used
identically by backtest, paper, and live"; putting it inside the simulator would
have guaranteed a second, subtly different implementation in Module 10.

- `costs.py` — entry prices, fees, round-trip cost, binary payoff. Buying UP
  lifts the UP ask; buying DOWN pays `1 - bid_up`, which no-arbitrage between
  the paired tokens makes the DOWN ask. Both directions cross the spread, which
  is the conservative reading and the correct one.
- `decision.py` — the abstention gates, ordered data-validity → market
  conditions → model opinion, so a stale book reports `stale_data` rather than a
  misleading `edge_too_small` computed from prices that were never real. Every
  refusal carries a `SkipReason`; none of the gates can *start* a trade.
- `sizing.py` — fractional Kelly with the caps that matter more than Kelly. At
  `p=0.95, c=0.50` full Kelly asks for 90% of bankroll on one coin flip;
  `max_risk_per_trade` is what stands between that belief and the account.
- `risk.py` — daily/weekly loss limits, consecutive-loss stand-downs, exposure
  caps, kill switch. Never reads the clock: `now_ms` comes from the caller, so a
  backtest and a live run produce identical state transitions.

**`pmbtc.backtest` — the simulator and the deployment gate.**

- `engine.py` — chronological replay, one decision per window. The label is used
  for exactly one thing: settling a position that was already opened.
- `fills.py` — `touch` by default; fills are capped at the depth actually
  resting, and under `pessimistic_fill` an *unknown* depth is treated as no
  depth. `mid` and `aggressive` exist as sensitivity analyses — the gap between
  `touch` and `mid` is the honest measure of how much of a strategy is really a
  bet on getting maker fills.
- `metrics.py` — money and forecast quality reported side by side, because when
  they disagree the disagreement is the finding. Break-even is the average price
  paid, not 0.5.
- `report.py` — the deployment gate, modelled on `models/promotion.py`: every
  check must pass, no force flag.
- `walkforward.py` — every lookback in `backtest.windows_days` gated separately,
  so a strategy whose entire profit came from one good fortnight cannot hide
  inside an average.
- `adapters.py` — `MarketProbabilityModel` is the null control: it forecasts
  what the book forecasts, so it must place zero trades. If it ever trades, the
  edge calculation is wrong.

- `pmbtc backtest [--model market|logistic|gradient_boosting] [--horizon N]
  [--fill touch|mid|aggressive] [--walk-forward]`. The train/test split is on a
  **market** boundary, so a market whose T-240 row fitted the model never has
  its T-60 row evaluated by it.
- `scripts/validate_module8.py` — 21 checks, each one a property that would let
  the system report a profit it did not earn. Wired into CI alongside a
  deployment-gate check that mirrors the readiness-gate check: a gate that
  approves a strategy on ten lucky trades is not a gate.

### Fixed
Two defects found by the tests written for this module, both in new code:
- A depth-capped fill recorded the *capped* stake as the requested amount, so
  `Fill.partial` was always false — partial fills could never have been detected.
- `RiskLedger` attributed P&L to an uninitialised day, so the next period
  rollover cleared a halt that had only just fired. A daily loss limit would have
  lasted no time at all. `register_close` now rolls the period first.

### Notes
The six abstention thresholds, five sizing caps, and seven risk limits this
module makes live were already written in `config.yaml` — and, as the gap
analysis for this release found, had **zero code consumers**: `sizing`,
`backtest` and `paper` were referenced nowhere in `src/`, and `costs`,
`execution` and `risk` only by their own validators. `constants.py` had likewise
defined `SkipReason`, `TradeStatus` and `OrderType` since Module 1 with nothing
using them. Module 8 is where that vocabulary starts running.

No schema moved: feature, dataset, settlement and model schemas are all
unchanged, and this module reads the dataset through the existing `build_rows`
rather than introducing a second loader. Readiness gates and training discipline
are untouched — the backtest deliberately does **not** require a trained model,
which is what makes it usable now, while the dataset is still hours old.

Measured on the 59 labelled markets collected so far, the null control places 0
trades and the gate returns INCONCLUSIVE on every strategy tried. That is the
correct answer, and it will stay the correct answer until roughly 200 trades'
worth of history exists.

---

## [0.8.3] — 2026-08-01 — Dependency audit on a clean runner

### Fixed
- `pip-audit` failed with *"pmbtc: Dependency not found on PyPI and could not be
  audited"*. The local project is installed editable and is not published, so
  dependency collection fails on it, and `--strict` — which means *fail if
  collection fails on any dependency* — makes that fatal.

  `--skip-editable` is **not** the fix: skipping is itself a collection failure
  under `--strict`, so the combination still fails. It would also be the wrong
  mechanism, blanket-skipping *every* editable install — a third-party package
  installed editable would silently drop out of the audit, which is exactly the
  hole a security check must not have.

  Instead `scripts/audit_dependencies.py` builds the audit set explicitly: every
  installed **non-editable** distribution, pinned, audited with
  `--strict --no-deps`. `pip list` already reports the full transitive closure,
  so the set is complete by construction rather than by resolution. Two guards
  keep the exclusion honest — the only editable distribution must be the local
  project, and the set must be non-trivially large so a broken `pip list` cannot
  pass by auditing nothing.
- `.gitignore` matched only exactly-named virtualenvs, so a venv called
  `.venv-ci` or `venv313` would be scanned and could be committed. Now globbed.
  Found by running the hygiene check against a throwaway environment.

### Notes
- Strictness is unchanged; only the *set* being audited changed. Verified: 92
  packages locally and 76 in a fresh clean-clone environment, `pmbtc` absent,
  exit 0. Negative control — `jinja2==2.11.2` reports 5 vulnerabilities — so the
  scan is armed rather than merely passing.

---

## [0.8.2] — 2026-08-01 — Deterministic CI without collected data

### Fixed
- The release workflow failed on a clean runner because Module 6 validation
  required a live CLOB archive, which exists only on a machine that has been
  collecting. That conflated two distinct properties, so they are now split
  rather than either being weakened:
  - **Determinism is a property of the code.** Verifying it needs a fixed input
    replayed twice, and a committed fixture is a *better* fixed input than a
    live archive — byte-identical on every machine forever, where a live archive
    differs by host and by hour.
  - **A live archive proves an operational property**: that the collector is
    running and emitting replayable output. Meaningful only where collection
    happens.
- `site/` (MkDocs build output) was committed by an earlier `git add -A` and is
  now untracked and ignored. The blobs remain in history at v0.8.1.

### Added
- `tests/fixtures/ticks/clob_sample.jsonl.gz` — a real 42 KB slice of the CLOB
  stream covering `book`, `price_change`, and `last_trade_price` for both
  outcome tokens. Built by `scripts/make_tick_fixture.py` with `mtime=0` so
  rebuilds are byte-stable.
- `pmbtc.live.fixtures` resolves the archive source: the committed fixture under
  `GITHUB_ACTIONS`/`CI` or when no live archive exists, the live archive
  otherwise. The chosen source is always reported, never silently substituted.
- `validate_module6.py --require-live` refuses the fixture entirely — the local
  production check that the collector really is producing replayable output.
- `tests/test_fixtures_resolution.py` covers resolution, fixture coverage of all
  event types, and determinism against the committed bytes.

---

## [0.8.1] — 2026-08-01 — Documentation site

### Fixed
- MkDocs strict mode rejected four links that escaped the `docs/` tree
  (`index.md` → `../CHANGELOG.md`, `../CONTRIBUTING.md`, `../SECURITY.md`, and
  `GITHUB_WORKFLOW.md` → `../.github/branch-protection.md`). Those links are
  genuinely broken on the published site even though they resolve when browsing
  GitHub, so **strict mode stays on** and the documentation architecture was
  fixed instead: `scripts/generate_docs.py` now mirrors the root-level documents
  into `docs/` and rewrites their internal links for the flattened layout. The
  sources remain at the repository root, where GitHub and contributors expect
  them; the copies carry a generated-from banner and CI fails if they drift.
- Generated documentation was written with platform-native line endings, so the
  same generator emitted different bytes on Windows than on a Linux CI runner
  and the staleness check saw phantom drift. All generators now write explicit
  LF.

### Added
- `generate_docs.py` asserts every relative link in `docs/` resolves inside
  `docs/`, failing before MkDocs is reached.
- `validate_github.py` checks strict mode is enabled, the mirrored pages exist
  and are marked generated, and no documentation link escapes.

---

## [0.8.0] — 2026-08-01 — Engineering Workflow

**Schemas:** feature `ade558d616b58821` · dataset `4.0` · settlement `2.0` · model `1.1`

No change to trading logic, the feature pipeline, or the models. This release is
about making the previous seven reproducible and hard to regress.

### Added
- Git repository with a documented branch strategy (`main`, `develop`,
  `feature/*`, `release/*`, `hotfix/*`) and branch-protection rules recorded as
  a reviewable document rather than only in a web UI.
- CI on every push and pull request: ruff, mypy, unit and integration tests on
  Python 3.12 and 3.13, replay determinism, feature/dataset/model schema
  validation, readiness validation, Gitleaks across full history, and
  `pip-audit --strict`. A single aggregate `CI complete` check gates merges, so
  a job can never be silently dropped from the required set.
- **Experiment tracking**: every model card now records the commit, branch, tag,
  and working-tree cleanliness it was trained from, and the commit is part of
  the experiment identity hash. A model trained from a dirty tree is marked
  unreproducible rather than quietly accepted.
- Tag-driven release workflow that re-runs every gate before publishing, since a
  tag can be pushed to any commit.
- Generated documentation — features, schemas, modules, API — built from the
  code, with CI failing if the committed copy is stale. Architecture,
  operations, and deployment guides. MkDocs site for GitHub Pages.
- Security policy, Dependabot with numerically-sensitive packages held for
  manual review, CODEOWNERS, issue and pull-request templates.
- `scripts/validate_repo_hygiene.py`, `validate_schemas.py`, `validate_github.py`.

### Changed
- `ModelCard` gained four git provenance fields (**model schema 1.0 → 1.1**).
- New `models` and `explain` extras so CI can install scikit-learn without
  pulling the ~3 GB deep-learning stack.

### Notes
- **Git LFS deliberately not enabled.** Frame archives grow ~95 GB/year and LFS
  never garbage-collects; the archives are reproducible by re-collection and
  pinned by a 24 KB committed baseline. Reasoning recorded in
  `docs/GITHUB_WORKFLOW.md`.
- `ruff format` is **not** enforced. Adopting it means one reformat commit across
  ~60 files — a deliberate decision, not a side effect of adding CI.

---

## [0.7.0] — 2026-08-01 — Model Training Framework

**Schemas:** feature `ade558d616b58821` · dataset `4.0` · settlement `2.0` · model `1.0`

### Added
- Time-aware validation only: expanding, rolling, and walk-forward splitters
  grouped by market, with purge and embargo. No shuffle parameter exists.
- Calibration as a first-class metric: Brier, log loss, ECE, MCE, Brier skill,
  reliability curves, and isotonic/sigmoid calibrators fitted on a held-back
  slice.
- Model registry with immutable artifacts and a mutable production pointer;
  model IDs derived from dataset, feature schema, config, and hyperparameters.
- Hyperparameter search (grid, random, optional Bayesian) confined to training
  folds.
- Statistical comparison: paired bootstrap, exact McNemar, Diebold-Mariano,
  exact binomial.
- Explainability: permutation importance, SHAP with honest fallback, interaction
  analysis, calibration plots.
- Shadow validation: determinism, artifact reproducibility, per-block
  calibration drift, inference latency.
- Promotion gate with five conditions and **no override path**.
- One canonical feature pipeline: live collection, archive replay, dataset
  export, and the training loader proven to produce identical vectors.

### Fixed
- Archive partitioned by wall-clock time instead of data timestamps, which
  reordered replayed frames and silently changed 12 of 70 feature values.
- Cold-start volatility produced ±85-sigma displacements and a fair value pinned
  at 0 or 1; estimates now require 20 trades spanning half the window.

### Notes
- Production training remains **blocked** by the readiness gate. Thresholds
  unchanged: 2,000 labelled markets, 1,000 complete timelines, 5,000 snapshots.

## [0.6.0] — 2026-08-01 — Feature Engineering

**Schemas:** feature `ade558d616b58821` · dataset `4.0` · settlement `2.0`

### Added
- 70 features across 10 independent family modules, each declaring formula,
  inputs, units, freshness budget, reproducibility policy, and purpose —
  enforced by the constructor signature.
- Dependency graph with deterministic topological ordering, cycle detection, and
  transitive impact analysis for incremental updates (15 of 70 recomputed on a
  clock tick).
- Bit-for-bit determinism between live and replayed computation.
- Evidence-based selection: SHAP, mutual information, permutation, RFE — on
  training folds only.
- Drift monitoring (PSI and z-score) that alerts and never retrains.
- Cross-source consistency annotating affected features from the graph.
- Generated feature documentation (`docs/features.md`).

### Fixed
- `math.log` shadowed by a logger, breaking PSI entirely.
- Drift self-masking: the reference window included the recent window.
- Archive replay crashed on the partially-written current-hour file.

## [0.5.0] — 2026-07-31 — Live Market Data

**Schemas:** dataset `4.0` · settlement `2.0`

### Added
- Polymarket CLOB WebSocket as the primary microstructure source
  (3,891 frames / 25 s measured, versus 29–72 s Gamma quote staleness).
- Self-healing feeds treating silence as failure; books discarded on reconnect.
- Order book (imbalance, microprice, depth, slippage) and trade tape
  (CVD, VWAP, realised volatility).
- Gzipped raw tick archive making stream features reproducible.
- Feature registry with tiers and reproducibility policies.
- Continuous collection service, benchmark baselines, readiness gate.

### Fixed
- Feed receive timestamps used the raw local clock, reporting negative transport
  latency; now clock-corrected (~220 ms mean to both venues).

## [0.4.0] — 2026-07-31 — Historical Dataset

**Schemas:** dataset `4.0` · settlement `2.0`

### Added
- T-300 → T-1 snapshot timeline, immutable and horizon-addressed.
- Leakage guard enforced at write *and* export, with a tested detector.
- Per-source latency budgets and quality scoring.
- Labels from the official Polymarket outcome only; external feeds audit-only.
- Parquet/Arrow/CSV/SQLite export with a reproducibility manifest.
- Eight-check pre-training data audit.

### Fixed
- Gamma `/markets` excludes closed markets unless `closed=true`, so label
  backfill could never see a settled market.

## [0.3.0] — 2026-07-31 — Market Discovery

### Added
- Series-driven discovery — no slug is ever constructed on the happy path.
- Market lifecycle tracking, health checks, liquidity snapshots.
- Payload schema versioning with fail-safe drift detection.
- Replay regression against a committed baseline of 50 real markets.
- Clock service with drift detection and a pre-settlement safety gate.

## [0.2.0] — 2026-07-31 — Settlement Verification

### Added
- Per-market settlement source detection from Gamma, structured-first and
  cross-examined against the rules prose.
- Nine verification gates; ambiguity means no trade.
- Append-only spec store detecting provider changes within a series.

### Notes
- Verified live: BTC **5m and 15m** settle on **Chainlink**, while the sibling
  **hourly** family settles on **Binance**. One product line, two authorities.

## [0.1.0] — 2026-07-31 — Infrastructure

### Added
- Layered validated configuration, structured logging with recursive secret
  redaction, UTC window arithmetic, typed errors, Docker, test suite.
- Live trading triple-gated and off by default.

[Unreleased]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.8.3...HEAD
[0.8.3]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/prakash89y/polymarket-btc-5m/releases/tag/v0.1.0
