"""Alert policy: per-signal thresholds plus a cooldown, tuned on the
validation window only and then frozen for the evaluation window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ALERT_BUDGET_FRAC, COOLDOWN_GRID, TUNE_QUANTILE

SIGNALS = ["input_psi", "input_ks", "score_psi", "outcome_psi"]


@dataclass(frozen=True)
class AlertPolicy:
    thresholds: dict
    cooldown: int
    tuned_on: str = "validation"
    quantile: float = TUNE_QUANTILE
    metadata: dict = field(default_factory=dict)


class Alerter:
    """Stateful alert tracker: applies a frozen policy with a cooldown."""

    def __init__(self, policy: AlertPolicy):
        self.policy = policy
        self._last_alert_index: int | None = None

    def update(self, period_index: int, signals: dict) -> tuple[bool, list[str]]:
        fired = [
            s for s in SIGNALS
            if signals.get(s) is not None
            and not np.isnan(signals[s])
            and signals[s] > self.policy.thresholds[s]
        ]
        if not fired:
            return False, []
        if self._last_alert_index is not None and period_index - self._last_alert_index <= self.policy.cooldown:
            return False, fired  # suppressed by cooldown
        self._last_alert_index = period_index
        return True, fired


def _alert_count(rows: list[dict], thresholds: dict, cooldown: int) -> int:
    policy = AlertPolicy(thresholds=thresholds, cooldown=cooldown)
    alerter = Alerter(policy)
    return sum(1 for i, r in enumerate(rows) if alerter.update(i, r)[0])


def tune(val_rows: list[dict], quantile: float = TUNE_QUANTILE,
         cooldown_grid=COOLDOWN_GRID, budget_frac: float = ALERT_BUDGET_FRAC) -> AlertPolicy:
    """Tune thresholds and cooldown on validation-period signal rows only.

    Thresholds are the per-signal validation quantile. The cooldown is the
    smallest grid value that keeps validation alerts within the budget.
    The returned policy is frozen: it is never refit on evaluation data.
    """
    if not val_rows:
        raise ValueError("no validation rows to tune on")
    thresholds = {}
    for s in SIGNALS:
        values = np.array([r[s] for r in val_rows if r.get(s) is not None and not np.isnan(r[s])], dtype=float)
        if len(values) == 0:
            raise ValueError(f"signal {s} has no validation observations")
        thresholds[s] = float(np.quantile(values, quantile, method="higher"))
    budget = max(1, int(np.floor(budget_frac * len(val_rows))))
    chosen = cooldown_grid[-1]
    for cd in cooldown_grid:
        if _alert_count(val_rows, thresholds, cd) <= budget:
            chosen = cd
            break
    return AlertPolicy(
        thresholds=thresholds,
        cooldown=int(chosen),
        metadata={
            "n_val_periods": len(val_rows),
            "val_alert_budget": budget,
            "val_alerts_at_policy": _alert_count(val_rows, thresholds, chosen),
            "cooldown_grid": list(cooldown_grid),
        },
    )
