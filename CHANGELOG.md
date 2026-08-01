# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Four schema versions move independently of the package version — feature,
dataset, settlement, and model. Each release records all four, because a
package upgrade that leaves the schemas alone is a very different upgrade from
one that does not.

## [Unreleased]

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

[Unreleased]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/prakash89y/polymarket-btc-5m/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/prakash89y/polymarket-btc-5m/releases/tag/v0.1.0
