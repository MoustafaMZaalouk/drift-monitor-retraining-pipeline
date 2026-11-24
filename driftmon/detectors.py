"""Hand-rolled drift detectors: Population Stability Index (PSI) and the
two-sample Kolmogorov-Smirnov statistic, numpy/scipy only.

PSI is a heuristic distribution-shift indicator. It is not a calibrated
probability of accuracy degradation, and we treat it as such throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import PSI_EPS, PSI_N_BINS

OTHER = "__OTHER__"


def quantile_edges(values: np.ndarray, n_bins: int = PSI_N_BINS) -> np.ndarray:
    """Bin edges from reference values only. Called once, then frozen."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        raise ValueError("cannot fit bins on empty reference")
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _clipped_props(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    props = counts / total if total > 0 else np.full_like(counts, 1.0 / max(len(counts), 1), dtype=float)
    return np.maximum(props, PSI_EPS)


def psi(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    ref = np.maximum(np.asarray(ref_props, dtype=float), PSI_EPS)
    cur = np.maximum(np.asarray(cur_props, dtype=float), PSI_EPS)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(cur_values: np.ndarray, edges: np.ndarray, ref_props: np.ndarray) -> float:
    cur_counts, _ = np.histogram(np.asarray(cur_values, dtype=float), bins=edges)
    return psi(ref_props, _clipped_props(cur_counts))


def psi_categorical(cur_values, ref_props: pd.Series) -> float:
    cats = list(ref_props.index)
    counts = pd.Series(cur_values).value_counts()
    aligned = np.array([counts.get(c, 0) for c in cats[:-1]] + [counts.get(OTHER, 0) + sum(
        v for k, v in counts.items() if k not in cats)], dtype=float)
    return psi(ref_props.to_numpy(dtype=float), _clipped_props(aligned))


def ks_statistic(ref_values: np.ndarray, cur_values: np.ndarray) -> float:
    """Two-sample KS statistic (not the p-value: at our sample sizes the
    p-value flags trivially small shifts, so we monitor the statistic)."""
    ref = np.asarray(ref_values, dtype=float)
    cur = np.asarray(cur_values, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    return float(stats.ks_2samp(ref, cur).statistic)


class ReferenceProfile:
    """Frozen reference-window state for all drift signals.

    Fit once on the training (reference) window; never refit during the
    stream. score_period() only reads the frozen state.
    """

    def __init__(self, numeric: list[str], categorical: list[str], n_bins: int = PSI_N_BINS):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.n_bins = n_bins
        self._num_edges: dict[str, np.ndarray] = {}
        self._num_ref_props: dict[str, np.ndarray] = {}
        self._num_ref_raw: dict[str, np.ndarray] = {}
        self._cat_ref_props: dict[str, pd.Series] = {}
        self._fitted = False

    def fit(self, ref_df: pd.DataFrame) -> "ReferenceProfile":
        if self._fitted:
            raise RuntimeError("ReferenceProfile is frozen after fit; refusing to refit")
        for col in self.numeric:
            values = ref_df[col].to_numpy(dtype=float)
            edges = quantile_edges(values, self.n_bins)
            counts, _ = np.histogram(values, bins=edges)
            self._num_edges[col] = edges
            self._num_ref_props[col] = _clipped_props(counts)
            self._num_ref_raw[col] = values[~np.isnan(values)]
        for col in self.categorical:
            props = ref_df[col].value_counts(normalize=True)
            # Rare reference categories collapse into OTHER to keep PSI stable.
            keep = props[props >= 0.01]
            other = 1.0 - float(keep.sum())
            keep[OTHER] = max(other, PSI_EPS)
            self._cat_ref_props[col] = np.maximum(keep, PSI_EPS)
        self._fitted = True
        return self

    def feature_psi_ks(self, period_df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature PSI (all features) and KS statistic (numeric only)."""
        if not self._fitted:
            raise RuntimeError("fit on the reference window first")
        rows = []
        for col in self.numeric:
            cur = period_df[col].to_numpy(dtype=float)
            rows.append({
                "feature": col, "kind": "numeric",
                "psi": psi_numeric(cur, self._num_edges[col], self._num_ref_props[col]),
                "ks": ks_statistic(self._num_ref_raw[col], cur),
            })
        for col in self.categorical:
            rows.append({
                "feature": col, "kind": "categorical",
                "psi": psi_categorical(period_df[col].tolist(), self._cat_ref_props[col]),
                "ks": np.nan,
            })
        return pd.DataFrame(rows)

    # Distribution helpers reused for score and outcome signals.
    def frozen_numeric(self, values: np.ndarray):
        """Return (edges, ref_props) fitted on the given reference array."""
        edges = quantile_edges(values, self.n_bins)
        counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
        return edges, _clipped_props(counts)
