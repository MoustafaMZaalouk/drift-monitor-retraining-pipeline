"""OPTIONAL third-party cross-check with Evidently (not part of core results).

Renders Evidently's DataDriftPreset as a standalone HTML report for a few
sampled periods (one clean evaluation period plus the first period of each
planted event), alongside our own signal values, for eyeball comparison only.
Evidently is an external library with its own drift definitions and defaults;
no Evidently number feeds any result in this repository, and verdict
extraction from its report objects is intentionally not automated.

Requires: pip install -r requirements-optional.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driftmon.config import ARTIFACTS_DIR, SEED
from driftmon.experiment import _prepare, _signals  # noqa: E402
from driftmon.planting import apply_events  # noqa: E402


def main():
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("evidently not installed; pip install -r requirements-optional.txt")
        return 0
    out = {}
    for name in ("bike", "shoppers"):
        ctx, refs, model, _ = _prepare(name, smoke=False)
        spec = ctx.spec
        dfp = apply_events(ctx.df, ctx.events)
        ref = ctx.df[ctx.df["_pidx"] < ctx.n_tr][spec.features]
        if len(ref) > 5000:
            ref = ref.sample(5000, random_state=SEED)
        clean_idx = next(i for i in range(ctx.eval_start, ctx.last_idx)
                         if not any(ev.start <= i <= ev.end for ev in ctx.events))
        sampled = [clean_idx] + [ev.start for ev in ctx.events]
        rows = []
        for idx in sampled:
            cur = dfp[dfp["_pidx"] == idx][spec.features]
            report = Report([DataDriftPreset()])
            result = report.run(reference_data=ref, current_data=cur)
            html_path = ARTIFACTS_DIR / spec.name / f"evidently_period_{idx}.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            result.save_html(str(html_path))
            our_row, _ = _signals(ctx, dfp, idx, refs, model)
            rows.append({"pidx": idx,
                         "our_input_psi": our_row["input_psi"],
                         "our_score_psi": our_row["score_psi"],
                         "evidently_html": str(html_path.relative_to(ARTIFACTS_DIR.parent)),
                         "note": "Evidently verdicts are read from the HTML by a human; not automated"})
        out[spec.name] = rows
    (ARTIFACTS_DIR / "evidently_crosscheck.json").write_text(json.dumps(out, indent=2))
    print("wrote artifacts/evidently_crosscheck.json and per-period HTML reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
