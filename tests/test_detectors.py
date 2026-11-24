import numpy as np
import pandas as pd
import pytest

from driftmon.detectors import (ReferenceProfile, ks_statistic, psi,
                                psi_categorical, psi_numeric, quantile_edges)


def test_psi_identical_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    edges = quantile_edges(ref)
    counts, _ = np.histogram(ref, bins=edges)
    props = np.maximum(counts / counts.sum(), 1e-4)
    cur = rng.normal(0, 1, 2000)
    assert psi_numeric(cur, edges, props) < 0.01


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    edges = quantile_edges(ref)
    counts, _ = np.histogram(ref, bins=edges)
    props = np.maximum(counts / counts.sum(), 1e-4)
    small = psi_numeric(rng.normal(0.5, 1, 2000), edges, props)
    large = psi_numeric(rng.normal(2.0, 1, 2000), edges, props)
    assert 0 < small < large


def test_quantile_edges_are_frozen_covering():
    edges = quantile_edges(np.arange(100, dtype=float))
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)


def test_categorical_psi_handles_unseen_category():
    ref_props = pd.Series({"a": 0.5, "b": 0.49, "__OTHER__": 0.01})
    base = psi_categorical(["a", "b", "a", "b"], ref_props)
    unseen = psi_categorical(["a", "b", "zzz"], ref_props)
    assert unseen > base >= 0


def test_ks_statistic_bounds_and_sensitivity():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 1000)
    assert 0.0 <= ks_statistic(a, rng.normal(0, 1, 500)) <= 1.0
    assert ks_statistic(a, rng.normal(3, 1, 500)) > 0.8


def test_reference_profile_is_frozen_after_fit():
    df = pd.DataFrame({"x": np.arange(200, dtype=float), "c": ["a", "b"] * 100})
    profile = ReferenceProfile(["x"], ["c"]).fit(df)
    edges_before = profile._num_edges["x"].copy()
    # scoring extreme future data must not change frozen state
    profile.feature_psi_ks(pd.DataFrame({"x": [1e9] * 10, "c": ["zzz"] * 10}))
    np.testing.assert_array_equal(profile._num_edges["x"], edges_before)
    with pytest.raises(RuntimeError):
        profile.fit(df)
