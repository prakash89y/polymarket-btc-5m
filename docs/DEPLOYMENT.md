# Deployment guide

## Requirements

- Python 3.12 or 3.13
- ~100 GB disk per year of continuous collection
- A stable network path to `clob.polymarket.com` and `stream.binance.com`
- **A clock that is actually synchronised.** The service corrects for offset,
  but it fails closed when drift exceeds 1.5 s — a host without NTP will simply
  stop trading.

## Install

```bash
git clone <repo> && cd polymarket-btc-5m
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,models]"
cp .env.example .env
pmbtc doctor
```

`doctor` is the pre-flight: config validity, paths, credential presence
(never values), and the live-trading interlocks.

## Docker

```bash
docker compose up --build
```

Paper mode, JSON logs, volumes for `data`, `logs`, and `artifacts`. The image is
two-stage and runs as a non-root user — it will eventually hold a funded key.

```bash
docker build --build-arg EXTRAS=".[data,models]" .
```

`EXTRAS` keeps a collector image from pulling torch.

## Configuration

Precedence: environment → `.env` → `config/config.yaml` → defaults. Nested keys
use a double underscore:

```bash
PMBTC_APP__MODE=paper
PMBTC_FEEDS__ARCHIVE_TICKS=true
```

Secrets never appear in YAML. `Config` and `BotSecrets` share no field name, so
there is nowhere in the config file for a credential to live.

## Running as a service

### systemd

```ini
[Unit]
Description=pmbtc collector
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pmbtc
WorkingDirectory=/opt/pmbtc
Environment=PMBTC_CONFIG=/opt/pmbtc/config/config.yaml
ExecStart=/opt/pmbtc/.venv/bin/pmbtc run --minutes 0
Restart=always
RestartSec=10
# The service recovers from transient errors internally; systemd handles the
# case where the process dies outright.

[Install]
WantedBy=multi-user.target
```

Watchdog, from cron or a timer:

```bash
*/5 * * * * cd /opt/pmbtc && .venv/bin/pmbtc watch || logger -t pmbtc "alert active"
0 6 * * *   cd /opt/pmbtc && .venv/bin/pmbtc daily-summary
```

## Going live — read this first

Live trading is gated three ways and **off by default**:

1. `app.mode: live` in config
2. `live.enabled: true` in config
3. `PMBTC_I_UNDERSTAND_LIVE_RISK=yes` in the environment

`live.dry_run` still defaults to `true` even with all three. Module 9's paper
promotion gate must also pass.

Before arming:

- [ ] `pmbtc readiness` passes every threshold
- [ ] `pmbtc audit` clean
- [ ] A model is promoted, and its card says `reproducible_from_git: true`
- [ ] Paper trading cleared its statistical gate
- [ ] A **dedicated wallet**, funded only with a loseable bankroll
- [ ] `data/KILL_SWITCH` tested — create it, confirm orders stop
- [ ] Trading on Polymarket is permitted in your jurisdiction

## Backups

| path | why |
|---|---|
| `data/dataset/` | The dataset. Irreplaceable — it is weeks of wall-clock time |
| `logs/decisions.jsonl` | Audit trail and training input; never rotated |
| `artifacts/models/` | Model artifacts and cards |
| `.env` | Back up **securely**, or not at all |

`data/raw/ticks/` is large and regenerable by re-collection; back it up only if
you need to re-derive features for windows already past.

## Upgrading

```bash
git pull && pip install -e ".[data,models]" && python scripts/validate_schemas.py
```

If the feature schema version changed, the collected dataset is a **different
schema**. Archive the generation before resuming, or the audit will block
training on a mixed dataset — correctly.
