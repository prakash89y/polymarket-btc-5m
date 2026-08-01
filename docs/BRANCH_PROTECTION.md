<!-- Generated from .github/branch-protection.md by scripts/generate_docs.py. Edit the source, not this copy. -->

# Branch protection

These rules cannot be set from a file — GitHub requires them to be configured on
the repository. This document is the source of truth for what they should be, so
the configuration is reviewable even though it lives in a web UI.

Apply under **Settings → Branches → Add branch ruleset**.

## `main`

| setting | value | why |
|---|---|---|
| Require a pull request | yes, 1 approval | Nothing reaches production unreviewed |
| Dismiss stale approvals on push | yes | An approval is for the diff that was read |
| Require review from Code Owners | yes | Schema and safety paths need the owner |
| Require status checks | **`CI complete`** | One aggregate check; new jobs need no rule change |
| Require branches up to date | yes | Prevents semantic conflicts that merge cleanly |
| Require conversation resolution | yes | |
| Require linear history | yes | Bisecting a trading bug should not fight merge commits |
| Require signed commits | recommended | |
| Allow force pushes | **no** | |
| Allow deletions | **no** | |
| Restrict who can push | maintainers only | |

## `develop`

Same as `main`, with two relaxations:

- Approvals: 1, Code Owner review **not** required for non-owned paths.
- Linear history: not required — feature merges are expected.

Force pushes and deletions remain blocked. `CI complete` remains required.

## Why one aggregate check

`CI complete` depends on every other job and fails if any did not succeed. The
protection rule names only that check, so adding a CI job never requires editing
branch protection — and, more importantly, a job cannot be silently dropped from
the required set by forgetting to add it.

## Tags

Protect the `v*` pattern under **Settings → Tags**: tags are release identity
and must not move.

## Verifying

```bash
gh api repos/:owner/:repo/branches/main/protection --jq '.required_status_checks.contexts'
```

Should list `CI complete`.
