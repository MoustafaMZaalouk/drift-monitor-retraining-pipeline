"""Model training and evaluation.

Gradient-boosted tree models (scikit-learn histogram gradient boosting, the
same model family as the batch-1 bike-demand and purchase-intent projects),
CPU only, fixed seeds, fixed hyperparameters across initial training and all
retrains so that before/after differences come from data, not tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import SEED
from .datasets import DatasetSpec

MAX_ITER = 200
SMOKE_MAX_ITER = 60


def build_model(spec: DatasetSpec, seed: int = SEED, max_iter: int = MAX_ITER) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", spec.numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec.categorical),
    ])
    if spec.task == "regression":
        est = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=0.08, early_stopping=False, random_state=seed)
    else:
        est = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=0.08, early_stopping=False, random_state=seed)
    return Pipeline([("pre", pre), ("est", est)])


def prepare_xy(df: pd.DataFrame, spec: DatasetSpec):
    X = df[spec.features].copy()
    y = df[spec.target].to_numpy(dtype=float)
    if spec.target_transform == "log1p":
        y = np.log1p(y)
    return X, y


def fit(model: Pipeline, df: pd.DataFrame, spec: DatasetSpec) -> Pipeline:
    X, y = prepare_xy(df, spec)
    model.fit(X, y)
    return model


def raw_scores(model: Pipeline, df: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    """Model-output scores used for the score-drift signal: predicted log1p
    count (regression) or predicted purchase probability (classification)."""
    X = df[spec.features]
    if spec.task == "regression":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def evaluate(model: Pipeline, df: pd.DataFrame, spec: DatasetSpec) -> dict:
    X, y = prepare_xy(df, spec)
    if spec.task == "regression":
        pred = np.expm1(model.predict(X))
        pred = np.clip(pred, 0.0, None)
        actual = np.expm1(y)
        return {
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "n": int(len(df)),
        }
    proba = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, proba)),
        "n": int(len(df)),
    }


def evaluate_pooled(segments: list, spec: DatasetSpec) -> dict:
    """Pool predictions from (model, dataframe) segments into one metric set.

    Used to compare a fixed baseline against a retrain-on-alert policy whose
    active model changes over the stream, on identical rows.
    """
    actuals, preds = [], []
    for model, df in segments:
        if len(df) == 0:
            continue
        X, y = prepare_xy(df, spec)
        if spec.task == "regression":
            p = np.clip(np.expm1(model.predict(X)), 0.0, None)
            a = np.expm1(y)
        else:
            p = model.predict_proba(X)[:, 1]
            a = y
        actuals.append(a)
        preds.append(p)
    a = np.concatenate(actuals)
    p = np.concatenate(preds)
    if spec.task == "regression":
        return {
            "mae": float(mean_absolute_error(a, p)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "n": int(len(a)),
        }
    return {
        "roc_auc": float(roc_auc_score(a, p)) if len(np.unique(a)) > 1 else float("nan"),
        "brier": float(brier_score_loss(a, p)),
        "n": int(len(a)),
    }
