"""End-to-end experiments.

run_monitor:    replay the stream, tune the alert policy on the validation
                window only, freeze it, then measure alert precision/recall
                against planted drift events in the evaluation window.
run_retraining: replay the planted stream with a retrain-on-alert policy and
                log before/after test metrics for every retrain, including
                the retrains that do not help.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alerting import Alerter, tune
from .config import (ARTIFACTS_DIR, DETECTION_GRACE_PERIODS, MONITOR_WINDOW,
                     PSI_EPS, SEED)
from .datasets import SPECS, DatasetSpec, load, ordered_periods, split_periods
from .detectors import ReferenceProfile, psi, quantile_edges
from .models import (MAX_ITER, SMOKE_MAX_ITER, build_model, evaluate,
                     evaluate_pooled, fit, prepare_xy, raw_scores)
from .planting import apply_events, bike_events, shoppers_events
from .stream import matured_index

SMOKE_BIKE_EVAL_PERIODS = 130
RETRAIN_TEST_HORIZON = {"bike": 14, "shoppers": 1}  # periods


@dataclass
class Context:
    spec: DatasetSpec
    df: pd.DataFrame
    periods: list
    n_tr: int
    n_val: int
    n_eval: int
    events: list
    smoke: bool

    @property
    def eval_start(self) -> int:
        return self.n_tr + self.n_val

    @property
    def last_idx(self) -> int:
        return self.n_tr + self.n_val + self.n_eval


def _setup(spec_name: str, smoke: bool) -> Context:
    spec = SPECS[spec_name]
    df = load(spec)
    periods = ordered_periods(df, spec)
    train_p, val_p, eval_p = split_periods(periods, spec)
    if smoke and spec.name == "bike":
        eval_p = eval_p[:SMOKE_BIKE_EVAL_PERIODS]
    used = train_p + val_p + eval_p
    df = df[df[spec.period_col].isin(list(used))].copy()
    df["_pidx"] = df[spec.period_col].map({p: i for i, p in enumerate(used)}).astype(int)
    n_tr, n_val, n_eval = len(train_p), len(val_p), len(eval_p)
    if spec.name == "bike":
        events = bike_events(n_tr, n_val, n_eval, smoke)
    else:
        events = shoppers_events(n_tr, n_val, n_eval, smoke)
    return Context(spec, df, used, n_tr, n_val, n_eval, events, smoke)


def _clipped(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    props = counts / total if total > 0 else np.full(len(counts), 1.0 / len(counts))
    return np.maximum(props, PSI_EPS)


def _frozen_numeric(values: np.ndarray, n_bins: int = 10):
    edges = quantile_edges(values, n_bins)
    counts, _ = np.histogram(values, bins=edges)
    return edges, _clipped(counts)


def _y_values(df: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    _, y = prepare_xy(df, spec)
    return y


def _psi_numeric(cur: np.ndarray, ref) -> float:
    edges, ref_props = ref
    counts, _ = np.histogram(np.asarray(cur, dtype=float), bins=edges)
    return psi(ref_props, _clipped(counts))


def _outcome_ref(y_train: np.ndarray, spec: DatasetSpec):
    if spec.task == "regression":
        return ("numeric", _frozen_numeric(y_train))
    props = np.maximum(pd.Series(y_train.astype(int)).value_counts(normalize=True)
                       .reindex([0, 1], fill_value=0.0).to_numpy(), PSI_EPS)
    return ("categorical", props)


def _outcome_psi(kind_ref, y_period: np.ndarray) -> float:
    kind, ref = kind_ref
    if kind == "numeric":
        return _psi_numeric(y_period, ref)
    props = np.maximum(pd.Series(y_period.astype(int)).value_counts(normalize=True)
                       .reindex([0, 1], fill_value=0.0).to_numpy(), PSI_EPS)
    return psi(ref, props)


def _signals(ctx: Context, df: pd.DataFrame, idx: int, refs: dict, model):
    """All monitored signals at one stream step, plus the feature table.

    Input and score distributions are computed over a trailing window of
    MONITOR_WINDOW periods ending at idx; the outcome signal uses the most
    recent matured window of the same length (delayed ground truth). The
    reference bundle is the season-matched one when the spec defines a
    reference group, so ordinary seasonal variation is not flagged.
    """
    bundle = _refs_for(ctx, refs, idx)
    profile, score_ref, outcome_ref = bundle["profile"], bundle["score_ref"], bundle["outcome_ref"]
    w = MONITOR_WINDOW[ctx.spec.name]
    pdf = df[df["_pidx"].between(idx - w + 1, idx)]
    ft = profile.feature_psi_ks(pdf)
    scores = raw_scores(model, pdf, ctx.spec)
    m = matured_index(idx, ctx.spec.label_delay_periods)
    outcome = None
    if m >= 0:
        matured = df[df["_pidx"].between(max(m - w + 1, 0), m)]
        outcome = _outcome_psi(outcome_ref, _y_values(matured, ctx.spec))
    row = {
        "pidx": int(idx),
        "period": str(ctx.periods[idx]),
        "n_rows": int(len(pdf)),
        "input_psi": float(ft["psi"].max()),
        "input_ks": float(ft["ks"].max(skipna=True)),
        "score_psi": _psi_numeric(scores, score_ref),
        "outcome_psi": outcome,
    }
    return row, ft.assign(pidx=int(idx))


def _reference_bundle(train_df: pd.DataFrame, spec: DatasetSpec, model):
    profile = ReferenceProfile(spec.monitored_numeric, spec.monitored_categorical).fit(train_df)
    score_ref = _frozen_numeric(raw_scores(model, train_df, spec))
    outcome_ref = _outcome_ref(_y_values(train_df, spec), spec)
    return {"profile": profile, "score_ref": score_ref, "outcome_ref": outcome_ref}


def _prepare(spec_name: str, smoke: bool, force_global_reference: bool = False):
    ctx = _setup(spec_name, smoke)
    spec = ctx.spec
    train_df = ctx.df[ctx.df["_pidx"] < ctx.n_tr]
    max_iter = SMOKE_MAX_ITER if smoke else MAX_ITER
    model = fit(build_model(spec, max_iter=max_iter), train_df, spec)
    if spec.reference_group and not force_global_reference:
        # Season-matched references: one frozen bundle per calendar value of
        # the group column (e.g. per month), all fit on training data only.
        refs = {int(g): _reference_bundle(grp, spec, model)
                for g, grp in train_df.groupby(spec.reference_group)}
    else:
        refs = {None: _reference_bundle(train_df, spec, model)}
    return ctx, refs, model, max_iter


def _refs_for(ctx: Context, refs: dict, idx: int):
    if None in refs:
        return refs[None]
    period = ctx.periods[idx]
    key = int(pd.Timestamp(period).month) if ctx.spec.name == "bike" else int(period)
    return refs[key]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2))


def run_monitor(spec_name: str, smoke: bool = False, out_root: Path = ARTIFACTS_DIR,
                force_global_reference: bool = False) -> dict:
    t0 = time.perf_counter()
    ctx, refs, model, _ = _prepare(spec_name, smoke, force_global_reference)
    spec = ctx.spec
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: clean replay of validation + evaluation windows.
    rows, feat_rows = [], []
    for idx in range(ctx.n_tr, ctx.last_idx):
        row, ft = _signals(ctx, ctx.df, idx, refs, model)
        row["phase"] = "val" if idx < ctx.eval_start else "eval"
        rows.append(row)
        feat_rows.append(ft)
    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / "drift_stats_clean.csv", index=False)
    pd.concat(feat_rows).to_csv(out_dir / "feature_drift_clean.csv", index=False)

    # Tune on validation rows only, then freeze.
    policy = tune([r for r in rows if r["phase"] == "val"])

    # Natural alert rate on the clean evaluation window (context for FPs).
    natural = Alerter(policy)
    n_natural = sum(
        1 for i, r in enumerate(rows) if r["phase"] == "eval" and natural.update(i, r)[0])

    # Pass 2: planted-drift replay of the evaluation window, frozen policy.
    dfp = apply_events(ctx.df, ctx.events)
    planted_rows, planted_feats, alerts, firings = [], [], [], []
    alerter = Alerter(policy)
    for local_i, idx in enumerate(range(ctx.eval_start, ctx.last_idx)):
        row, ft = _signals(ctx, dfp, idx, refs, model)
        alert, fired = alerter.update(local_i, row)
        row["alert"] = bool(alert)
        row["fired_signals"] = ",".join(fired)
        planted_rows.append(row)
        planted_feats.append(ft)
        if fired:
            firings.append({"pidx": idx, "fired": fired})
        if alert:
            alerts.append({"pidx": idx, "period": row["period"], "fired": fired,
                           "input_psi": row["input_psi"], "score_psi": row["score_psi"]})
    planted = pd.DataFrame(planted_rows)
    planted.to_csv(out_dir / "drift_stats_planted_eval.csv", index=False)
    pd.concat(planted_feats).to_csv(out_dir / "feature_drift_planted_eval.csv", index=False)
    pd.DataFrame(alerts).to_csv(out_dir / "alerts_planted_eval.csv", index=False)

    # Precision/recall against planted events (alert inside an event window,
    # plus a grace of DETECTION_GRACE_PERIODS after it ends, counts as a hit).
    grace = DETECTION_GRACE_PERIODS

    def hits_event(pidx):
        return any(ev.start <= pidx <= ev.end + grace for ev in ctx.events)

    # Precision is over alerts actually sent (cooldown applied): each sent
    # alert inside an event window (plus grace) is a true positive.
    tp = sum(1 for a in alerts if hits_event(a["pidx"]))
    fp = len(alerts) - tp
    # Recall credits any threshold firing, sent or cooldown-suppressed: the
    # monitor noticed the event even when rate-limiting muted the alert.
    per_event = []
    for ev in ctx.events:
        hits = [f for f in firings if ev.start <= f["pidx"] <= ev.end + grace]
        per_event.append({
            "event_id": ev.event_id, "kind": ev.kind, "feature": ev.feature,
            "start": ev.start, "end": ev.end, "description": ev.description,
            "detected": bool(hits),
            "latency_periods": (min(f["pidx"] for f in hits) - ev.start) if hits else None,
            "fired_signals": sorted({s for f in hits for s in f["fired"]}),
        })
    # Strict variant: a sent alert (cooldown applied) must land in the window.
    strict_per_event = []
    for ev in ctx.events:
        hits = [a for a in alerts if ev.start <= a["pidx"] <= ev.end + grace]
        strict_per_event.append({"event_id": ev.event_id, "detected": bool(hits)})
    detection = {
        "n_events": len(ctx.events),
        "events_detected": sum(1 for e in per_event if e["detected"]),
        "recall": sum(1 for e in per_event if e["detected"]) / len(ctx.events),
        "n_alerts": len(alerts),
        "true_positive_alerts": tp,
        "false_positive_alerts": fp,
        "precision": (tp / len(alerts)) if alerts else None,
        "grace_periods": grace,
        "natural_eval_alerts_clean_stream": n_natural,
        "recall_basis": "any threshold firing, sent or cooldown-suppressed",
        "precision_basis": "sent alerts only",
        "strict": {
            "events_detected": sum(1 for e in strict_per_event if e["detected"]),
            "recall": sum(1 for e in strict_per_event if e["detected"]) / len(ctx.events),
            "precision": (tp / len(alerts)) if alerts else None,
            "basis": "sent alerts only, on both sides",
            "per_event": strict_per_event,
        },
        "reference": "season_matched" if not force_global_reference else "global_full_year",
        "per_event": per_event,
    }
    _dump({
        "thresholds": policy.thresholds, "cooldown": policy.cooldown,
        "quantile": policy.quantile, "tuned_on": policy.tuned_on, **policy.metadata,
    }, out_dir / "tuning.json")
    _dump(detection, out_dir / "detection_eval.json")
    _plot_monitor(ctx, planted, policy, alerts, out_dir / "monitor.png")
    elapsed = time.perf_counter() - t0
    return {"dataset": spec.name, "policy": {"thresholds": policy.thresholds,
            "cooldown": policy.cooldown, **policy.metadata},
            "detection": detection, "seconds": elapsed}


def run_retraining(spec_name: str, smoke: bool = False, out_root: Path = ARTIFACTS_DIR) -> dict:
    t0 = time.perf_counter()
    ctx, refs, base_model, max_iter = _prepare(spec_name, smoke)
    spec = ctx.spec
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rebuild the same frozen policy from the clean validation pass.
    val_rows = []
    for idx in range(ctx.n_tr, ctx.eval_start):
        row, _ = _signals(ctx, ctx.df, idx, refs, base_model)
        val_rows.append(row)
    policy = tune(val_rows)

    dfp = apply_events(ctx.df, ctx.events)
    alerter = Alerter(policy)
    # roster: (active_from_pidx, model, trained_through_pidx, n_train_rows)
    roster = [(ctx.eval_start, base_model, ctx.n_tr - 1, int((ctx.df["_pidx"] < ctx.n_tr).sum()))]
    retrains = []
    for local_i, idx in enumerate(range(ctx.eval_start, ctx.last_idx)):
        row, _ = _signals(ctx, dfp, idx, refs, roster[-1][1])
        alert, fired = alerter.update(local_i, row)
        if not alert:
            continue
        m = matured_index(idx, spec.label_delay_periods)
        if m <= roster[-1][2]:
            continue  # no newly matured data to learn from
        train_slice = dfp[dfp["_pidx"] <= m]
        new_model = fit(build_model(spec, max_iter=max_iter), train_slice, spec)
        roster.append((idx, new_model, m, int(len(train_slice))))
        retrains.append({"pidx": int(idx), "period": row["period"],
                         "trained_through_pidx": int(m), "n_train_rows": int(len(train_slice)),
                         "fired_signals": fired})

    # Before/after: for each retrain, the old and new models are scored on the
    # next RETRAIN_TEST_HORIZON periods of the planted stream (what production
    # would actually serve), using matured outcomes. A companion reading on the
    # CLEAN stream copy of the same periods shows whether a refit absorbed
    # corrupted training rows instead of recovering.
    horizon = RETRAIN_TEST_HORIZON[spec.name]
    before_after = []
    for i, r in enumerate(retrains):
        old_model, new_model = roster[i][1], roster[i + 1][1]
        w_lo, w_hi = r["pidx"], min(r["pidx"] + horizon - 1, ctx.last_idx - 1)
        w = dfp[dfp["_pidx"].between(w_lo, w_hi)]
        wc = ctx.df[ctx.df["_pidx"].between(w_lo, w_hi)]
        entry = {"retrain_pidx": r["pidx"], "period": r["period"],
                 "test_periods": [int(w_lo), int(w_hi)],
                 "n_test_rows": int(len(w)),
                 "old": evaluate(old_model, w, spec), "new": evaluate(new_model, w, spec),
                 "old_clean": evaluate(old_model, wc, spec), "new_clean": evaluate(new_model, wc, spec)}
        before_after.append(entry)
    pd.DataFrame([{
        "retrain_pidx": r["retrain_pidx"], "period": r["period"],
        **{f"old_{k}": v for k, v in r["old"].items()},
        **{f"new_{k}": v for k, v in r["new"].items()},
        **{f"old_clean_{k}": v for k, v in r["old_clean"].items()},
        **{f"new_clean_{k}": v for k, v in r["new_clean"].items()},
    } for r in before_after]).to_csv(out_dir / "retraining_log.csv", index=False)

    # Whole-window comparison: never-retrain baseline vs retrain-on-alert policy,
    # on both the planted stream and its clean-stream copy.
    eval_df = dfp[dfp["_pidx"] >= ctx.eval_start]
    eval_clean = ctx.df[ctx.df["_pidx"] >= ctx.eval_start]
    baseline = evaluate_pooled([(base_model, eval_df)], spec)
    segments, segments_clean = [], []
    for j, (start, model_j, _, _) in enumerate(roster):
        end = roster[j + 1][0] if j + 1 < len(roster) else ctx.last_idx
        segments.append((model_j, dfp[dfp["_pidx"].between(start, end - 1)]))
        segments_clean.append((model_j, ctx.df[ctx.df["_pidx"].between(start, end - 1)]))
    policy_metrics = evaluate_pooled(segments, spec)
    baseline_clean = evaluate_pooled([(base_model, eval_clean)], spec)
    policy_clean = evaluate_pooled(segments_clean, spec)

    headline = "mae" if spec.task == "regression" else "brier"
    better_when_lower = headline in ("mae", "brier")
    for e in before_after:
        delta = e["new"][headline] - e["old"][headline]
        e[f"delta_{headline}"] = delta
        e["verdict"] = ("no change" if abs(delta) < 1e-9 else
                        ("helps" if (delta < 0) == better_when_lower else "hurts"))
    summary = {
        "n_alerts_triggering_retrain": len(retrains),
        "retrains": retrains,
        "before_after": before_after,
        "headline_metric": headline,
        "verdict_counts": {v: sum(1 for e in before_after if e["verdict"] == v)
                           for v in ("helps", "hurts", "no change")},
        "whole_window": {"never_retrain": baseline, "retrain_on_alert": policy_metrics},
        "whole_window_clean": {"never_retrain": baseline_clean, "retrain_on_alert": policy_clean},
        "test_horizon_periods": horizon,
        "note": ("before/after metrics are computed on the planted (corrupted) "
                 "stream window following each retrain (what production would "
                 "serve), with a companion clean-stream reading of the same "
                 "periods (old_clean/new_clean); the retrain-on-alert policy "
                 "retrains on all matured data, including corrupted rows"),
    }
    _dump(summary, out_dir / "retraining_summary.json")
    _plot_retraining(spec, before_after, out_dir / "retraining.png")
    elapsed = time.perf_counter() - t0
    return {"dataset": spec.name, "summary": summary, "seconds": elapsed}


def _period_labels(ctx: Context, planted: pd.DataFrame) -> list:
    if ctx.spec.name == "bike":
        return [pd.Timestamp(p).strftime("%Y-%m") for p in planted["period"]]
    return list(planted["period"])


def _plot_monitor(ctx: Context, planted: pd.DataFrame, policy, alerts, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x = np.arange(len(planted))
    for ev in ctx.events:
        axes[0].axvspan(ev.start - ctx.eval_start, ev.end - ctx.eval_start,
                        color="orange", alpha=0.2)
        axes[1].axvspan(ev.start - ctx.eval_start, ev.end - ctx.eval_start,
                        color="orange", alpha=0.2)
    axes[0].plot(x, planted["input_psi"], label="max input PSI", lw=1)
    axes[0].plot(x, planted["score_psi"], label="score PSI", lw=1)
    axes[0].axhline(policy.thresholds["input_psi"], color="C0", ls="--", lw=0.8)
    axes[0].axhline(policy.thresholds["score_psi"], color="C1", ls="--", lw=0.8)
    for a in alerts:
        axes[0].axvline(a["pidx"] - ctx.eval_start, color="red", alpha=0.5, lw=0.8)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("PSI")
    axes[0].set_title(f"{ctx.spec.name}: planted-drift replay (shaded = planted events, red = alerts)")
    axes[1].plot(x, planted["outcome_psi"], color="C2", lw=1, label="outcome PSI (delayed labels)")
    axes[1].axhline(policy.thresholds["outcome_psi"], color="C2", ls="--", lw=0.8)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("PSI")
    labels = _period_labels(ctx, planted)
    step = max(1, len(labels) // 12)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
    axes[1].set_xlabel(f"stream period ({ctx.spec.period_unit})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_retraining(spec: DatasetSpec, before_after: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not before_after:
        return
    metric = "mae" if spec.task == "regression" else "brier"
    labels = [f"retrain@{e['period'][:10]}" for e in before_after]
    old = [e["old"][metric] for e in before_after]
    new = [e["new"][metric] for e in before_after]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 4))
    ax.bar(x - 0.2, old, width=0.4, label="before retrain")
    ax.bar(x + 0.2, new, width=0.4, label="after retrain")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{spec.name}: before/after {metric.upper()} on the {RETRAIN_TEST_HORIZON[spec.name]}-period post-retrain window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_reference_ablation(spec_name: str = "bike", smoke: bool = False,
                           out_root: Path = ARTIFACTS_DIR) -> dict:
    """Evidence for the season-matched reference design: rerun monitoring with
    a single naive full-year reference and record how detection degrades.
    Writes artifacts/<dataset>/reference_ablation.json."""
    matched = run_monitor(spec_name, smoke=smoke, out_root=out_root)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        naive = run_monitor(spec_name, smoke=smoke, out_root=Path(tmp),
                            force_global_reference=True)
    def brief(run):
        d = run["detection"]
        return {"thresholds": run["policy"]["thresholds"], "cooldown": run["policy"]["cooldown"],
                "n_alerts": d["n_alerts"], "true_positive_alerts": d["true_positive_alerts"],
                "false_positive_alerts": d["false_positive_alerts"],
                "events_detected": d["events_detected"], "n_events": d["n_events"],
                "precision": d["precision"], "recall": d["recall"],
                "natural_eval_alerts_clean_stream": d["natural_eval_alerts_clean_stream"]}
    out = {"dataset": spec_name,
           "season_matched_reference": brief(matched),
           "naive_full_year_reference": brief(naive)}
    _dump(out, out_root / spec_name / "reference_ablation.json")
    return out
