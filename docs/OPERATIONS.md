# Operations guide

## Running the collector

```bash
pmbtc run --minutes 0
```

Runs until stopped. Detached on Windows:

```powershell
Start-Process -FilePath ".\.venv\Scripts\pmbtc.exe" `
  -ArgumentList "run","--minutes","0" -WindowStyle Hidden `
  -RedirectStandardOutput "logs\collector.out.log" `
  -RedirectStandardError  "logs\collector.err.log"
```

The dataset is the asset. It should keep growing while later modules are built —
the readiness gate needs roughly a week of continuous collection.

## Checking it is alive

```bash
pmbtc watch
```

Reads `data/heartbeat.json` (written every 15 seconds) and evaluates the four
alert conditions. A **process that is running but no longer collecting** is the
failure worth catching, which is why liveness is a heartbeat file and not a PID
check.

Exit code 1 means an alert is active.

## Daily report

```bash
pmbtc daily-summary
```

Growth over 24 hours, class balance, quality distribution, and progress toward
each readiness threshold with an ETA at the current collection rate.

## Alerts

Only four conditions fire, deliberately. A system that pages on every reconnect
teaches its operator to ignore it.

| condition | meaning | action |
|---|---|---|
| `collection_stopped` | Heartbeat stale or absent | Restart the collector |
| `feed_degraded` | Feed disconnected, or p95 latency blown out | Check network, then venue status |
| `readiness_regressed` | A **counter** went backwards | Investigate data loss — this should be impossible |
| `quality_below_threshold` | Too many low-quality snapshots | Check feed health first |

Only monotonic counters trigger regression alerts. Ratios like class balance
fluctuate legitimately and are excluded.

## Data integrity

```bash
pmbtc audit
```

Eight checks: label leakage, duplicate markets, duplicate snapshots, timestamp
inconsistencies, missing settlement labels, feature schema drift, provider
mismatches, class imbalance. Training aborts on any failure and there is no
override.

**Feature schema drift is the one to take seriously.** It means early and late
rows are different measurements. The correct response is to archive the dataset
generation and restart collection — this has happened twice on purpose, once
when switching from Gamma-only to live feeds and once when the canonical feature
pipeline landed.

```bash
mv data/dataset data/dataset-genN-<reason>
```

## Labels

Resolution lags settlement, so backfill runs on its own cadence:

```bash
pmbtc dataset-label
```

The service does this automatically every five minutes. Gamma excludes closed
markets unless asked for them explicitly — a settled market is invisible to an
unfiltered query, which is why the client asks for the closed set first.

## Readiness

```bash
pmbtc readiness
```

Reports every threshold with have/need. Training is blocked until all pass.
**Do not lower these thresholds.** At a few hundred markets, a 2% edge and pure
noise are indistinguishable.

## Kill switch

```bash
touch data/KILL_SWITCH
```

Blocks all new orders immediately. No restart needed. Delete the file to resume.

## Log files

`logs/pmbtc.log` rotates at 64 MB, 14 backups. Secrets are redacted recursively
before any sink, including `logging.extra` from third-party libraries.

`logs/decisions.jsonl` **never rotates**. It is the training set and the audit
trail. Back it up; deleting it costs everything the system has learned.

## Clock

The clock service medians several references, corrects for round-trip time, and
fails closed. On a 300-second instrument a one-second error is 0.33% of the
window applied to every label.

```bash
pmbtc doctor
```

Shows offset, uncertainty, and whether the safety window would currently block
an order. Measured on the development machine: ~0.7–1.3 s behind true UTC, which
is why feed timestamps use the corrected clock rather than `time.time()`.

## Archive growth

Roughly 11 MB/hour gzipped, ~95 GB/year. Not versioned in git — see
[GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md) for why LFS is the wrong tool. Plan
disk accordingly, or prune old days once the corresponding dataset rows are
exported and verified.
