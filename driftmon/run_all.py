"""One-command reproduction: python -m driftmon.run_all [--smoke]

Runs the monitoring and retraining experiments for both datasets, writes all
artifacts under artifacts/, and prints the summary tables that the README
embeds (scripts/print_readme_tables.py re-prints them from artifacts).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import ARTIFACTS_DIR, SEED
from .experiment import _dump, run_monitor, run_reference_ablation, run_retraining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="short run for CI: truncated bike window, fewer boosting rounds")
    args = parser.parse_args()
    t0 = time.perf_counter()
    summary = {"seed": SEED, "smoke": args.smoke, "datasets": {}}
    for name in ("bike", "shoppers"):
        if name == "bike":
            abl = run_reference_ablation(name, smoke=args.smoke)
            mon = run_monitor(name, smoke=args.smoke)  # rewrite canonical artifacts after the ablation
        else:
            abl = None
            mon = run_monitor(name, smoke=args.smoke)
        ret = run_retraining(name, smoke=args.smoke)
        summary["datasets"][name] = {
            "monitor": {"detection": mon["detection"], "policy": mon["policy"],
                        "seconds": round(mon["seconds"], 2)},
            "retraining": {
                "n_retrains": ret["summary"]["n_alerts_triggering_retrain"],
                "verdict_counts": ret["summary"]["verdict_counts"],
                "headline_metric": ret["summary"]["headline_metric"],
                "whole_window": ret["summary"]["whole_window"],
                "whole_window_clean": ret["summary"]["whole_window_clean"],
                "before_after": ret["summary"]["before_after"],
                "seconds": round(ret["seconds"], 2),
            },
            "reference_ablation": abl,
        }
    summary["total_seconds"] = round(time.perf_counter() - t0, 2)
    _dump(summary, ARTIFACTS_DIR / "summary.json")
    print(json.dumps(summary["total_seconds"]))
    print(f"done in {summary['total_seconds']}s; artifacts under {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
