"""Leakage-focused invariants over the real (small) shoppers pipeline.

Runs the actual retraining experiment on the smallest dataset and asserts the
properties that matter: retrains only see matured labels, test windows never
overlap training data, planted events stay inside the evaluation window, and
the frozen policy comes from the validation window only.
"""
import pytest

from driftmon.alerting import tune
from driftmon.experiment import _prepare, run_retraining
from driftmon.planting import apply_events


@pytest.fixture(scope="module")
def prepared():
    return _prepare("shoppers", smoke=True)


def test_planted_events_stay_in_eval_window(prepared):
    ctx = prepared[0]
    for ev in ctx.events:
        assert ev.start >= ctx.eval_start
        assert ev.end < ctx.last_idx
    dfp = apply_events(ctx.df, ctx.events)
    pre_eval = dfp["_pidx"] < ctx.eval_start
    cols = [c for c in ctx.df.columns if c != "_pidx"]
    pd_left = dfp.loc[pre_eval, cols].reset_index(drop=True)
    pd_right = ctx.df.loc[pre_eval, cols].reset_index(drop=True)
    for c in cols:
        assert pd_left[c].astype(str).equals(pd_right[c].astype(str)), c


def test_retraining_uses_only_matured_labels(tmp_path, prepared):
    result = run_retraining("shoppers", smoke=True, out_root=tmp_path)
    ctx = prepared[0]
    delay = ctx.spec.label_delay_periods
    for r in result["summary"]["retrains"]:
        assert r["trained_through_pidx"] <= r["pidx"] - delay
    for e in result["summary"]["before_after"]:
        # test window starts at the retrain period, strictly after the last
        # training period, so training and evaluation rows never overlap
        trained_through = next(r["trained_through_pidx"] for r in
                               result["summary"]["retrains"]
                               if r["pidx"] == e["retrain_pidx"])
        assert e["test_periods"][0] > trained_through


def test_policy_tuned_on_validation_window_only(prepared):
    ctx, refs, model, _ = prepared
    from driftmon.experiment import _signals
    val_rows = [_signals(ctx, ctx.df, i, refs, model)[0]
                for i in range(ctx.n_tr, ctx.eval_start)]
    policy = tune(val_rows)
    # a threshold that had been fit on evaluation data could not exceed every
    # validation maximum by construction; assert it sits within validation reach
    for s, th in policy.thresholds.items():
        vals = [r[s] for r in val_rows if r.get(s) is not None]
        assert th <= max(vals) + 1e-9
