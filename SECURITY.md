# Security policy

This repository controls a wallet that can move funds. Treat it accordingly.

## Reporting a vulnerability

Do **not** open a public issue. Use GitHub's private vulnerability reporting
(*Security → Report a vulnerability*) or contact the maintainer directly.

Please include what an attacker could achieve, not just that something looks
wrong — the difference between "this key is logged" and "this key is logged and
the log ships to a third party" determines how fast it has to be fixed.

Expect an acknowledgement within 72 hours.

## What is in scope

| Area | Why it matters |
|---|---|
| Credential handling | `POLYMARKET_PRIVATE_KEY` can move funds |
| Log redaction | A leaked key in a log file is a lost wallet |
| Settlement verification | Bypassing it means trading the wrong instrument |
| Promotion / live-trading gates | Bypassing them means untested code sizing real money |
| Dependency supply chain | An upstream compromise reaches the signing key |

## Design commitments

These are enforced by tests, not by convention:

- **Secrets never enter configuration files.** `Config` and `BotSecrets` share no
  field name, so there is nowhere in `config.yaml` for a credential to live.
  Asserted by `tests/test_config.py::TestSecrets`.
- **Secrets are redacted from every log sink**, recursively, including
  `logging.extra` from third-party libraries. Asserted by
  `tests/test_logging.py::TestRedaction`.
- **Live trading is triple-gated**: `mode: live` **and** `live.enabled: true`
  **and** `PMBTC_I_UNDERSTAND_LIVE_RISK=yes`, with `dry_run` defaulting true
  even then.
- **Settlement verification cannot be disabled outside backtests.** The config
  layer rejects it.
- **No promotion override exists.** There is no `force` parameter anywhere in
  the training or promotion path, asserted by signature inspection.

## Operating guidance

- Use a **dedicated wallet** funded only with the bankroll you can lose. Never
  the wallet holding your main balance.
- Keep `.env` out of git. It is ignored, and CI fails if it is ever staged.
- `data/KILL_SWITCH` halts all new orders immediately; create the file, no
  restart required.
- Rotate the Polymarket API key if a log file was ever shared.

## Automated scanning

Every push runs, and must pass:

- `scripts/validate_repo_hygiene.py` — forbidden paths, file sizes, and a
  credential-shaped-string scan over tracked content.
- **Gitleaks** across full history, not just the tip commit.
- **pip-audit** in `--strict` mode; a known CVE in any shipped dependency fails
  the build.
- **Dependabot** weekly, with major bumps of numerically-sensitive packages held
  for manual review.
