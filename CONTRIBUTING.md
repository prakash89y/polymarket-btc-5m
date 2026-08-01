# Contributing

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[data,models,dev]"
cp .env.example .env
pmbtc doctor
```

`.[data,models,dev]` is what CI installs. `.[ml]` adds torch and the boosted-tree
stack (~3 GB) and is only needed for local research.

## Before you push

```bash
make check
```

Ruff, mypy, and the full suite — the same three that gate a pull request.

## The three rules that matter

**1. Do not weaken a gate to make something pass.**

The settlement verifier, the leakage guard, the readiness gate, and the
promotion gate exist because each one has already caught a real defect. If one
blocks you, it is more likely right than wrong. Changing a threshold is a
reviewable config commit with a stated reason — never an inline override, and
never a `force` flag.

**2. A schema change needs a version bump and a plan for existing data.**

Four schemas move independently: feature, dataset, settlement, model. Change one
without bumping it and old rows become silently incomparable to new ones — no
test necessarily fails. The PR template asks about this; answer it honestly.

If a change makes existing collected data incompatible, say what happens to it.
The usual answer is "archive the generation and restart collection", which has
already happened twice on purpose.

**3. Determinism is not negotiable.**

Live collection, archive replay, dataset export, and the training loader must
produce identical feature vectors. `tests/test_integration_pipeline.py` asserts
this as one fingerprint comparison. If it fails, something real broke — the last
two times it caught frame reordering and a cold-start volatility collapse.

The replay baseline (`tests/fixtures/replay_baseline.json`) records how the
parser reads 50 real archived markets. Regenerating it means historical
interpretation moved; do it deliberately, with `pmbtc replay --update`, and
review the diff.

## Style

Ruff enforces the mechanics. Beyond that:

- **Comments explain *why*.** The code already says what. A comment that
  restates the line below is noise; one that records why an obvious approach was
  rejected is worth more than the function.
- **Returning `None` is a legitimate answer.** An empty book has no imbalance.
  Inventing a zero is a lie the model will learn.
- **Prefer failing loudly to degrading quietly**, everywhere data integrity is
  involved.

## Testing

Every feature has a stated formula, so every feature gets a test against that
formula on hand-built state whose answer is computable on paper. Tests that
assert a function returns *something* are not worth writing.

Tests must not touch the network. The suite is offline by construction; CI runs
`-m "not network"` to make that a property of the run rather than a convention.

## Branching

See [docs/GITHUB_WORKFLOW.md](docs/GITHUB_WORKFLOW.md). Short version: branch
from `develop`, name it `feature/<slug>`, open a PR, keep it small.

## Commits

Conventional Commits — the changelog is generated from them.

```
feat(features): add order-flow imbalance slope
fix(archive): partition by data timestamp, not wall clock
```

If a commit changes a feature formula, a dataset schema, or a settlement rule,
say so in the body. Someone reading the diff in six months needs to know without
re-deriving it.
