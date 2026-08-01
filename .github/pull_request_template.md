## What and why

<!-- What changes, and what problem it solves. Not a restatement of the diff. -->

## Schema impact

Tick every box that applies. **A schema change without a version bump makes
every prior row incomparable to every later one**, and no test will necessarily
catch it.

- [ ] No schema change
- [ ] **Feature schema** — a feature's formula, inputs, or units changed
      (`feature_set_version` will change automatically; confirm it did)
- [ ] **Dataset schema** — snapshot or market record shape changed
      (bump `FEATURE_SCHEMA_VERSION` / `DATASET_RECORD_VERSION`)
- [ ] **Settlement schema** — detection or normalisation changed
      (bump `PARSER_VERSION`)
- [ ] **Model schema** — model card fields changed

If any box above is ticked, say what happens to already-collected data:

<!-- e.g. "existing rows keep schema 4.0; the audit will flag the mix and the
     dataset generation must be archived before collection resumes" -->

## Safety

- [ ] No change to settlement verification, leakage guards, readiness gate, or
      the promotion gate
- [ ] If any of those changed, explain why the safety property still holds:

<!-- ... -->

## Validation

- [ ] `make check` passes locally (ruff, mypy, tests)
- [ ] Determinism holds — replay baseline unchanged, or deliberately
      regenerated with the diff reviewed
- [ ] New behaviour has a test that fails without the change

<!-- Paste anything worth seeing: a validation script's output, a metric
     before/after, a fingerprint comparison. -->

## Risk

- [ ] Low — tooling, docs, tests
- [ ] Medium — collection or feature logic
- [ ] High — anything that can place an order, or that changes stored meaning
