<!-- Generated from CHANGELOG.md by scripts/generate_docs.py. Edit the source, not this copy. -->

# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Four schema versions move independently of the package version — feature,
dataset, settlement, and model. Each release records all four, because a
package upgrade that leaves the schemas alone is a very different upgrade from
one that does not.

## [Unreleased]

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

[Unreleased]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.8.2...HEAD
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
