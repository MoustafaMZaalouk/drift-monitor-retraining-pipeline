"""Dataset specs, loaders, and chronological stream splits.

Both datasets are downloaded at build time by scripts/download_data.py with a
pinned SHA-256 of the UCI archive. Nothing derived from them is committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DATA_DIR

# Chronological order of the Month column in Online Shoppers (UCI documents
# the sessions as spanning Feb..Dec of one year; April and January are absent).
SHOPPERS_MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str  # "regression" or "classification"
    period_col: str
    numeric: list[str]
    categorical: list[str]
    target: str
    target_transform: str | None  # "log1p" or None
    n_train_periods: int
    n_val_periods: int
    label_delay_periods: int
    period_unit: str  # human label, "day" or "month"
    # Features excluded from drift monitoring (never from model inputs).
    monitor_exclude: tuple = ()
    # When set, drift references are fit per calendar value of this column
    # (e.g. per month) and each window is compared against the matching
    # season's reference, so ordinary seasonal variation is not flagged.
    reference_group: str | None = None
    features: list[str] = field(init=False)
    monitored_features: list[str] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "features", list(self.numeric) + list(self.categorical))
        object.__setattr__(self, "monitored_features",
                           [f for f in self.features if f not in self.monitor_exclude])

    @property
    def monitored_numeric(self):
        return [c for c in self.numeric if c in self.monitored_features]

    @property
    def monitored_categorical(self):
        return [c for c in self.categorical if c in self.monitored_features]


BIKE = DatasetSpec(
    name="bike",
    task="regression",
    period_col="dteday",
    numeric=["temp", "atemp", "hum", "windspeed"],
    categorical=["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"],
    target="cnt",
    target_transform="log1p",
    # A full year as the reference window: all four seasons are represented,
    # so validation thresholds reflect ordinary seasonal variation rather
    # than the first winter-to-summer transition.
    n_train_periods=365,
    n_val_periods=60,
    label_delay_periods=7,
    period_unit="day",
    # Calendar fields stay model inputs but are excluded from drift
    # monitoring: their values are known in advance, so any trailing window
    # concentrates on one or two of them by calendar construction (a single
    # season, month boundary, or holiday), not because anything drifted.
    # Left in, their constant saturation PSI sets the alert threshold and
    # drowns the signals a monitor actually exists for (weather, outcomes).
    monitor_exclude=("season", "mnth", "hr", "weekday", "workingday", "holiday"),
    reference_group="mnth",
)

SHOPPERS = DatasetSpec(
    name="shoppers",
    task="classification",
    period_col="Month",
    numeric=[
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
        "PageValues", "SpecialDay",
    ],
    categorical=["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"],
    target="Revenue",
    target_transform=None,
    n_train_periods=3,   # Feb, Mar, May
    n_val_periods=3,     # June, Jul, Aug
    label_delay_periods=1,
    period_unit="month",
    # Month stays a model input but is excluded from drift monitoring: the
    # dataset spans only 10 distinct months, so every held-out month is an
    # unseen category by construction of the replay (its PSI saturates at a
    # constant value and would drown every other signal). That is an artifact
    # of replaying 10 months, not real drift.
    monitor_exclude=("Month",),
)

SPECS = {s.name: s for s in (BIKE, SHOPPERS)}


def load_bike(data_dir=DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "hour.csv", parse_dates=["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    # Leakage guard: casual + registered sum to the target cnt, so they are
    # never used as predictors. instant/yr/dteday are identifiers or leak the
    # long-run trend and are excluded from features.
    return df


def load_shoppers(data_dir=DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "online_shoppers_intention.csv")
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)
    df["Month"] = pd.Categorical(df["Month"], categories=SHOPPERS_MONTH_ORDER, ordered=True)
    df = df.sort_values("Month").reset_index(drop=True)
    return df


def load(spec: DatasetSpec, data_dir=DATA_DIR) -> pd.DataFrame:
    if spec.name == "bike":
        return load_bike(data_dir)
    if spec.name == "shoppers":
        return load_shoppers(data_dir)
    raise ValueError(spec.name)


def ordered_periods(df: pd.DataFrame, spec: DatasetSpec) -> list:
    """Period identifiers in chronological order."""
    if spec.name == "bike":
        return list(pd.Series(df[spec.period_col].unique()).sort_values())
    return [m for m in SHOPPERS_MONTH_ORDER if m in set(df[spec.period_col].astype(str))]


def split_periods(periods: list, spec: DatasetSpec):
    """Contiguous, disjoint train / validation / evaluation period lists."""
    n_tr, n_val = spec.n_train_periods, spec.n_val_periods
    train = periods[:n_tr]
    val = periods[n_tr:n_tr + n_val]
    eval_ = periods[n_tr + n_val:]
    assert set(train).isdisjoint(val) and set(train).isdisjoint(eval_) and set(val).isdisjoint(eval_)
    return train, val, eval_
