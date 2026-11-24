# Drift Monitor + Retraining Pipeline (simulated production replay)

Replays two public datasets as a simulated production stream and measures,
end to end, whether hand-rolled drift detectors catch known data failures and
whether retraining on the drifted data actually helps.

> **Scope: this is a simulation.** Historical data is replayed day-by-day /
> month-by-month to make monitoring and retraining behavior measurable and
> reproducible. Nothing here ran in production, and no production-readiness
> or state-of-the-art claim is made. See [Limitations](#limitations).

## What it does

- **Simulated stream.** UCI Bike Sharing (hourly, replayed day-by-day) and
  UCI Online Shoppers (replayed month-by-month) are replayed in chronological
  order. Ground-truth outcomes mature only after a fixed delay (7 days / 1
  month), so outcome monitoring and retraining only ever use labels that
  would have arrived.
- **Hand-rolled detectors (numpy/scipy only).** Per-feature Population
  Stability Index (PSI) on all monitored inputs, two-sample KS statistics on
  numeric inputs, PSI on the model's score distribution, and PSI on delayed
  outcome distributions.
- **Season-matched references (bike).** Each 7-day monitoring window is
  compared against the *same calendar month* of the training year, so
  ordinary seasonal variation is not flagged as drift. This design is backed
  by an ablation, not asserted: see [Results](#results). Calendar fields
  whose window concentration is known in advance (season, month, weekday,
  ...) are model inputs but excluded from monitoring, with the reason
  documented in `driftmon/datasets.py`.
- **Validation-tuned, then frozen.** Alert thresholds (per-signal validation
  quantile, q95) and the alert cooldown are tuned on the validation window
  only and frozen before the evaluation window starts. Enforced by tests.
- **Planted-drift evaluation.** Three known synthetic failures are injected
  into a copy of the evaluation window - truncated sensor range, inverted
  feature sign, shifted seasonality - and alert precision/recall is measured
  against the planted events.
- **Retraining trigger.** Alerts trigger refits of the same model family
  (scikit-learn histogram gradient boosting, consistent with the batch-1
  bike-demand and purchase-intent projects) on all *matured* data seen so
  far, with identical hyperparameters. Before/after test metrics are logged
  for every retrain, including the retrains that do not help.

## Data (downloaded at build time, never committed)

Both datasets are CC BY 4.0 and fetched by `scripts/download_data.py`, which
verifies pinned SHA-256 checksums of the UCI archives. Full citations,
license texts links, source URLs and checksums: **[DATA_LICENSES.md](DATA_LICENSES.md)**.

## Reproduce

```bash
pip install -r requirements.txt
make repro        # download data (checksum-verified), run experiments, run tests
make tables       # re-print the README result tables from committed artifacts
make run-smoke    # short CI variant (truncated bike window, fewer boosting rounds)
```

Python 3.10, CPU only. Full reproduction of every number below takes about
**68 seconds** wall-clock (measured on this machine, including the reference
ablation; your hardware will differ). All seeds are fixed (`SEED=42`);
scikit-learn histogram gradient boosting is deterministic for a pinned
version, which is why `requirements.txt` pins exact versions.

## Results

Every number below is regenerated from committed artifacts by
`python scripts/print_readme_tables.py` (which reads
`artifacts/summary.json` and `artifacts/bike/reference_ablation.json`).

**Headlines:**

- All 6 planted failures (3 per dataset) are detected; the strict
  sent-alerts-only recall is 3/3 on bike and 2/3 on shoppers.
- **Bike precision is the honest weak point: 0.58** (14 false-positive alerts
  on the planted stream, and 16 alerts on 306 days of the *clean* stream -
  about 5% of normal days page someone). See the discussion below.
- **Retraining is not a free lunch: 12 of 37 bike retrains made the next
  14-day window worse.** Net effect is still positive (MAE 98.23 -> 90.56).

### How detection is scored (the only place these are defined)

An event occupies stream periods `[start, end]`; an alert or firing at period
`p` counts toward it if `start <= p <= end + grace` with `grace = 2` periods.

- A **firing** is any period where at least one monitored signal crosses its
  frozen threshold, whether or not the cooldown lets an alert through.
- A **sent alert** is a firing that survived the cooldown - what a human
  would actually receive.
- **Precision (sent)** = `#{sent alerts inside an event window + grace} / #{sent alerts}`.
- **Recall (any firing)** = `#{events with >= 1 firing in window + grace} / #{planted events}`.
- **Recall (strict, sent-only)** = same but counting only sent alerts. This
  is the conservative variant: a cooldown-suppressed firing does not count.
- **Clean-stream alerts** = alerts the same frozen policy sends while
  replaying the evaluation window with *nothing planted*; the natural
  false-alarm background behind the FP count.

### Detection of planted drift (frozen validation-tuned policy)

| Dataset | Sent alerts | TP | FP | Precision (sent) | Recall (any firing) | Recall (strict: sent-only) | Clean-stream alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| bike | 33 | 19 | 14 | 0.58 | 1.00 | 1.00 | 16 |
| shoppers | 2 | 2 | 0 | 1.00 | 1.00 | 0.67 | 2 |

The shoppers strict recall (0.67) is the cooldown doing its job: the December
visitor-type event fired one month after the November alert, inside the
cooldown, so no second alert was sent - the monitor *noticed* (firing) but
stayed quiet (no sent alert). That trade-off is deliberate and recorded in
`artifacts/shoppers/detection_eval.json` under `strict`.

**The bike false-positive rate is the weakest result in this repository and
we say so plainly.** With a q95 validation threshold, 16 of 306 clean-stream
days (~5%) still alert - weather in 2012 differs enough from the same month
of 2011 to cross the threshold - and on the planted stream this becomes 14
false positives against 19 true positives (precision 0.58). Raising the
threshold would trade away the 1.00 recall. Detection latency was 0, 5 and 2
days for the truncate, sign-flip and season-shift events.

### Reference ablation (bike): why season-matched references

The season-matched reference is a measured decision, not an asserted one.
Rerunning the identical pipeline with a naive single full-year reference
(`python -m driftmon.run_all` writes both to
`artifacts/bike/reference_ablation.json`):

| Reference | Alerts | TP | FP | Events detected | Precision | Recall | Clean-stream alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| season-matched (used) | 33 | 19 | 14 | 3/3 | 0.58 | 1.00 | 16 |
| naive full-year | 44 | 14 | 30 | 2/3 | 0.32 | 0.67 | 38 |

Against a full-year reference, ordinary seasonal weeks dominate the
thresholds: the season-shift event goes undetected, false positives more than
double, and the natural alert rate reaches 38 of 306 days (~12%).

### Retrain on alert

| Dataset | Stream read | Never-retrain | Retrain-on-alert |
|---|---|---|---|
| bike | planted (corrupted) | MAE 98.23 / RMSE 139.08 | MAE 90.56 / RMSE 129.36 |
| bike | clean holdout | MAE 96.71 / RMSE 135.34 | MAE 92.57 / RMSE 130.14 |
| shoppers | planted (corrupted) | AUC 0.8152 / Brier 0.1248 | AUC 0.8240 / Brier 0.1147 |
| shoppers | clean holdout | AUC 0.8118 / Brier 0.1341 | AUC 0.8273 / Brier 0.1198 |

- **Planted (corrupted) read**: models are scored on the drifted stream
  itself - what production would actually serve. **Clean holdout read**: the
  same models scored on the uncorrupted copy of the same periods, showing
  whether refits absorbed corrupted training rows. Both exist for every
  retrain (per-retrain `old_*`/`new_*`/`old_clean_*`/`new_clean_*` columns in
  `artifacts/*/retraining_log.csv`).
- **12 of 37 bike retrains hurt** the following 14-day window on the
  corrupted read (25 helped). On the clean holdout the policy still nets
  positive (96.71 -> 92.57 MAE) but by less than the corrupted read suggests
  (98.23 -> 90.56) - part of the "improvement" on the corrupted stream is
  the refit adapting to corrupted inputs, exactly what the companion reading
  exists to surface.
- The shoppers November retrain improved Brier (0.1499 -> 0.1457) while
  *lowering* AUC (0.8160 -> 0.7967): better calibrated, worse ranked.
  Reported as-is.
- Retrains use only matured labels (delay: 7 days bike / 1 month shoppers);
  enforced by tests.

### Figures

- `artifacts/bike/monitor.png`, `artifacts/shoppers/monitor.png`: PSI
  timelines with frozen thresholds, planted-event spans and sent alerts.
- `artifacts/bike/retraining.png`, `artifacts/shoppers/retraining.png`:
  per-retrain before/after headline metric (corrupted-stream read).

## Layout

```
driftmon/
  datasets.py    specs, loaders, chronological splits, monitored-feature policy
  detectors.py   hand-rolled PSI / KS, frozen ReferenceProfile
  stream.py      delayed-outcome mechanics
  alerting.py    thresholds + cooldown; tuning on validation only
  planting.py    planted drift events (evaluation window only)
  models.py      model family, metrics, pooled policy evaluation
  experiment.py  monitoring, retraining, clean-holdout and ablation runners
  run_all.py     one-command entry point
scripts/
  download_data.py         checksum-pinned dataset download
  print_readme_tables.py   regenerates the README tables from artifacts
  evidently_crosscheck.py  OPTIONAL third-party eyeball comparison (below)
tests/                     23 tests, leakage-focused (see CI)
DATA_LICENSES.md           dataset citations, licenses, checksums
```

## Testing and CI

`pytest` (23 tests) covers detector correctness, cooldown semantics, and the
leakage invariants that matter for a monitoring pipeline:

- reference bins/profiles are frozen after fitting on the training window;
- planted events never touch training/validation rows (asserted column by column);
- retraining only sees matured labels, and its test windows never overlap
  its training rows;
- alert thresholds are reachable from validation data alone and the policy
  object is immutable after tuning;
- split chronology and target-leakage guards (`casual`/`registered` are
  never features, since they sum to the target).

`.github/workflows/ci.yml` runs the tests plus a smoke end-to-end run on
every push.

## Optional third-party cross-check

`pip install -r requirements-optional.txt && make crosscheck` renders
Evidently (`evidently==0.7.23`) DataDriftPreset HTML reports for a handful of
sampled periods next to our own signal values. This is an eyeball comparison
of a third-party tool with its own drift definitions; no Evidently output
feeds any number reported above.

## Limitations

- **Simulated stream.** Replay of historical data, twice. No live
  predictions, no serving, no production claims.
- **Bike false positives.** ~5% of clean-stream days alert; precision 0.58 is
  the honest cost of a q95 validation threshold at 1.00 recall, not a number
  we tuned post-hoc to look better.
- **PSI/KS are heuristics.** A drift statistic crossing a threshold is not
  proof of accuracy degradation; outcome monitoring uses delayed labels with
  a fixed assumed delay, and real label delays are rarely fixed.
- **Univariate monitors have blind spots.** A bare relabeling of a calendar
  field (e.g. month+3) is invisible to per-feature marginal checks, which is
  why the bike seasonality event shifts the *weather* columns instead; a
  production system would add multivariate or performance-based detectors.
- **Shoppers tuning window is tiny** (3 months), so its thresholds are rough;
  and its third planted event swaps visitor-type labels instead of shifting
  seasonality, because the dataset's 10 distinct months make any held-out
  month an unseen category by construction (documented in
  `driftmon/planting.py`).
- **Retraining on drifted data can hurt**, and does in 12 of 37 bike cases;
  the policy shown here is deliberately naive (retrain on every alert) to
  make that visible. The corrupted-stream before/after read conflates
  recovery with adaptation to corrupted inputs; the clean-holdout companion
  read bounds that effect but cannot remove it (the refits themselves were
  trained on corrupted rows).
- Results are exact only for the pinned dependency versions; the model
  family, seeds and hyperparameters are fixed, but nothing here claims
  superiority over any other approach.

## AI-assistance disclosure

This repository was built with AI assistance (design, implementation and
documentation were drafted by an AI agent under human direction) and reviewed
by the portfolio team before publication.

## License and attribution

Code is MIT licensed. Datasets are CC BY 4.0 by their respective authors,
downloaded from the UCI Machine Learning Repository at build time and not
redistributed here - full citations and license details in
[DATA_LICENSES.md](DATA_LICENSES.md).
