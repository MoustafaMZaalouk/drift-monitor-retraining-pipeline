import numpy as np
import pandas as pd

from driftmon.datasets import (BIKE, SHOPPERS, SHOPPERS_MONTH_ORDER, load,
                               ordered_periods, split_periods)
from driftmon.stream import matured_index


def test_splits_are_contiguous_disjoint_and_chronological():
    for spec in (BIKE, SHOPPERS):
        df = load(spec)
        periods = ordered_periods(df, spec)
        train, val, eval_ = split_periods(periods, spec)
        assert train + val + eval_ == periods
        assert len(train) == spec.n_train_periods
        assert len(val) == spec.n_val_periods


def test_shoppers_months_chronological_not_alphabetical():
    df = load(SHOPPERS)
    assert ordered_periods(df, SHOPPERS) == SHOPPERS_MONTH_ORDER


def test_bike_leakage_columns_not_in_features():
    for leaked in ("casual", "registered", "cnt", "instant", "yr"):
        assert leaked not in BIKE.features


def test_delayed_outcomes_never_from_the_future():
    for t in range(10):
        assert matured_index(t, 7) < t
    assert matured_index(3, 7) < 0  # nothing matured yet
