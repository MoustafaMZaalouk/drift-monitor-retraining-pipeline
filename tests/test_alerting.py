import pytest

from driftmon.alerting import Alerter, AlertPolicy, tune

TH = {"input_psi": 0.2, "input_ks": 0.3, "score_psi": 0.2, "outcome_psi": 0.2}


def _row(v):
    return {"input_psi": v, "input_ks": 0.0, "score_psi": 0.0, "outcome_psi": 0.0}


def test_alert_only_above_threshold():
    a = Alerter(AlertPolicy(thresholds=TH, cooldown=2))
    assert a.update(0, _row(0.1))[0] is False
    assert a.update(1, _row(0.5))[0] is True


def test_cooldown_suppresses_then_recovers():
    a = Alerter(AlertPolicy(thresholds=TH, cooldown=2))
    assert a.update(0, _row(0.9))[0] is True
    assert a.update(1, _row(0.9))[0] is False  # inside cooldown
    assert a.update(2, _row(0.9))[0] is False  # still inside cooldown
    assert a.update(3, _row(0.9))[0] is True   # cooldown expired


def test_missing_signal_never_alerts():
    a = Alerter(AlertPolicy(thresholds=TH, cooldown=1))
    row = {"input_psi": None, "input_ks": None, "score_psi": None, "outcome_psi": None}
    assert a.update(0, row) == (False, [])


def test_tune_uses_only_the_rows_it_is_given_and_freezes():
    val_rows = [_row(0.05 * i) for i in range(40)]
    policy = tune(val_rows)
    import numpy as np
    expected = np.quantile([r["input_psi"] for r in val_rows], 0.95, method="higher")
    assert policy.thresholds["input_psi"] == pytest.approx(expected)
    assert policy.cooldown in (1, 2, 4, 7, 14)
    with pytest.raises(Exception):
        policy.thresholds = {}  # frozen dataclass


def test_tune_rejects_empty_validation():
    with pytest.raises(ValueError):
        tune([])
