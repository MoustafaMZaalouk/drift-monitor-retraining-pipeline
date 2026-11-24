"""Print the README results tables from committed artifacts.

Every number in the README's results section comes from artifacts/summary.json
(plus artifacts/bike/reference_ablation.json) via this script: run
`python scripts/print_readme_tables.py` after `python -m driftmon.run_all`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "artifacts" / "summary.json"
ABLATION = ROOT / "artifacts" / "bike" / "reference_ablation.json"


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    s = json.loads(SUMMARY.read_text())
    abl = json.loads(ABLATION.read_text())

    print("### Detection of planted drift (frozen validation-tuned policy)\n")
    print("| Dataset | Sent alerts | TP | FP | Precision (sent) | Recall (any firing) | Recall (strict: sent-only) | Clean-stream alerts |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in s["datasets"].items():
        det = d["monitor"]["detection"]
        print(f"| {name} | {det['n_alerts']} | {det['true_positive_alerts']} | {det['false_positive_alerts']} | "
              f"{f(det['precision'])} | {f(det['recall'])} | {f(det['strict']['recall'])} | "
              f"{det['natural_eval_alerts_clean_stream']} |")

    print("\n### Reference ablation (bike): why season-matched references\n")
    print("| Reference | Alerts | TP | FP | Events detected | Precision | Recall | Clean-stream alerts |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in (("season_matched_reference", "season-matched (used)"),
                       ("naive_full_year_reference", "naive full-year")):
        b = abl[key]
        print(f"| {label} | {b['n_alerts']} | {b['true_positive_alerts']} | {b['false_positive_alerts']} | "
              f"{b['events_detected']}/{b['n_events']} | {f(b['precision'])} | {f(b['recall'])} | "
              f"{b['natural_eval_alerts_clean_stream']} |")

    print("\n### Retrain on alert: whole evaluation window\n")
    print("| Dataset | Stream read | Never-retrain | Retrain-on-alert |")
    print("|---|---|---|---|")
    for name, d in s["datasets"].items():
        r = d["retraining"]
        if r["headline_metric"] == "mae":
            for stream, ww in (("planted (corrupted)", r["whole_window"]),
                               ("clean holdout", r["whole_window_clean"])):
                nw, po = ww["never_retrain"], ww["retrain_on_alert"]
                print(f"| {name} | {stream} | MAE {f(nw['mae'])} / RMSE {f(nw['rmse'])} | "
                      f"MAE {f(po['mae'])} / RMSE {f(po['rmse'])} |")
        else:
            for stream, ww in (("planted (corrupted)", r["whole_window"]),
                               ("clean holdout", r["whole_window_clean"])):
                nw, po = ww["never_retrain"], ww["retrain_on_alert"]
                print(f"| {name} | {stream} | AUC {f(nw['roc_auc'],4)} / Brier {f(nw['brier'],4)} | "
                      f"AUC {f(po['roc_auc'],4)} / Brier {f(po['brier'],4)} |")
    print()
    for name, d in s["datasets"].items():
        r = d["retraining"]
        print(f"- {name}: {r['n_retrains']} retrains, verdicts {r['verdict_counts']}")
    print(f"\nTotal wall-clock for both datasets (incl. ablation): {s['total_seconds']}s (CPU only).")


if __name__ == "__main__":
    main()
