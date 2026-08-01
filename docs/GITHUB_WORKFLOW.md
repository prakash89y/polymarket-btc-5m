# Engineering workflow

This repository trades real money on a 5-minute instrument. The workflow exists
to make two things very hard: shipping code that has not been validated, and
producing a model nobody can rebuild.

## Branches

| branch | purpose | protected | merges from |
|---|---|---|---|
| `main` | Released, deployable. Every commit is tagged. | yes | `release/*`, `hotfix/*` |
| `develop` | Integration. Always green, not always released. | yes | `feature/*`, `release/*` back-merge |
| `feature/<slug>` | One change. Short-lived. | no | branched from `develop` |
| `release/<version>` | Stabilisation, changelog, version bump. | no | branched from `develop` |
| `hotfix/<slug>` | Production defect. | no | branched from `main` |

```
main     ─●────────────────●──────────────●─────  tags: v0.1.0, v0.2.0
           \              /              /
release/    \        ●──●              /
             \      /                 /
develop  ─●───●────●────●────●───────●──────────
            \      /      \    /
feature/     ●────●        ●──●
```

### Rules

**`main` and `develop` are protected.** No direct pushes, no force-pushes, no
deletion. Every change arrives by pull request with green CI and one approving
review. Configure this under *Settings → Branches* using the rules in
[branch protection](BRANCH_PROTECTION.md), whose source lives at
`.github/branch-protection.md`.

**Feature branches are short-lived.** A branch open for two weeks is a merge
conflict with extra steps. Split the work.

**Hotfixes branch from `main`, not `develop`.** A production defect must not
drag unreleased work along with it. Merge to `main`, tag, then back-merge to
`develop` so the fix is not lost at the next release.

### Naming

```
feature/module-8-backtesting
feature/fix-clock-drift-alert
release/0.8.0
hotfix/settlement-source-mismatch
```

Use the imperative and name the *change*, not the file.

## Commits

Conventional Commits, because the changelog is generated from them:

```
feat(features): add order-flow imbalance slope
fix(archive): partition by data timestamp, not wall clock
docs(workflow): document the hotfix path
test(integration): assert four-way feature identity
chore(deps): bump ruff to 0.6
```

Scopes match the package layout: `settlement`, `gamma`, `dataset`, `features`,
`live`, `models`, `ops`, `ci`, `docs`.

A commit that changes a feature formula, a dataset schema, or a settlement rule
must say so in the body. Those three change the *meaning* of stored data, and
the reader of a diff six months from now needs to know without re-deriving it.

## Pull requests

CI must be green. That is not a formality: the checks include replay
determinism and the four-way feature-identity test, which are the only things
standing between a refactor and a silently corrupted dataset.

The PR template asks whether the change affects a schema. Answer honestly — a
schema change requires a version bump, and an unversioned schema change makes
every prior row incomparable to every later one.

## Versioning

Four version numbers move independently, because they answer different
questions:

| version | lives in | changes when |
|---|---|---|
| Package | `pyproject.toml` | any release |
| Feature schema | `features/matrix.py` (`feature_set_version`) | a feature's formula, inputs, or units change |
| Dataset schema | `dataset/schema.py` (`FEATURE_SCHEMA_VERSION`) | the snapshot/market record shape changes |
| Settlement schema | `settlement/parser.py` (`PARSER_VERSION`) | settlement detection or normalisation changes |

The feature-schema version is a *hash of the declarations*, so it changes
automatically and cannot be forgotten. The other three are hand-maintained and
are checked by `scripts/validate_schemas.py`.

Releases are tagged `v<major>.<minor>.<patch>` on `main`. Each completed module
gets a tag, a changelog entry, and release notes.

## Large files

**Git LFS is deliberately not enabled.** The reasoning, so it is not
relitigated:

- Frame archives grow ~11 MB/hour, about **95 GB/year**.
- LFS stores every version and does not garbage-collect without rewriting
  history, so that figure is cumulative.
- GitHub's LFS quota is 1 GB free; this would exhaust it in under four days.
- The archives are **reproducible by re-collection**, and their interpretation
  is pinned by `tests/fixtures/replay_baseline.json` — 24 KB of committed JSON
  that fails CI if the parser ever reads them differently.

LFS is the right tool for artifacts you cannot regenerate. These are not. If
durable archive retention becomes a requirement, use object storage with
lifecycle rules; that is a separate concern from source control.

What *is* committed: a small archive fixture under `tests/fixtures/archive/`
(308 KB) because the replay determinism test cannot run without it.

## Running the checks locally

```bash
make check
```

Runs ruff, mypy, and the full test suite — the same three that gate a PR. The
heavier validation scripts run in CI and can be run by hand:

```bash
python scripts/validate_repo_hygiene.py
python scripts/validate_schemas.py
python scripts/validate_module6.py
python scripts/validate_module7.py
```
