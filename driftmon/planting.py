"""Planted drift events: known, synthetic corruptions applied to a copy of
the evaluation-window stream so detector precision/recall can be measured
against ground truth. Training, reference, and validation data are never
touched; the planted stream exists only for evaluation.

Periods are addressed by integer stream index (the `_pidx` column), so
events can never leak outside the evaluation window by accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd



@dataclass(frozen=True)
class PlantedEvent:
    event_id: str
    kind: str          # "truncate" | "sign_flip" | "month_shift" | "category_remap"
    feature: str
    start: int         # stream period index, inclusive
    end: int           # stream period index, inclusive
    params: dict
    description: str


def apply_events(df: pd.DataFrame, events: list[PlantedEvent]) -> pd.DataFrame:
    """Apply all events to a copy of df that carries an integer `_pidx` column."""
    out = df.copy()
    for ev in events:
        mask = out["_pidx"].between(ev.start, ev.end)
        col = ev.feature
        if ev.kind == "truncate":
            out.loc[mask, col] = out.loc[mask, col].clip(upper=ev.params["cap"])
        elif ev.kind == "sign_flip":
            out.loc[mask, col] = -out.loc[mask, col]
        elif ev.kind == "month_shift":
            shift = ev.params["shift"]
            out.loc[mask, col] = ((out.loc[mask, col].astype(int) - 1 + shift) % 12) + 1
        elif ev.kind == "season_shift":
            # Weather columns are replaced by values from `shift_rows` earlier
            # rows (the stream is hourly and near-regular), simulating the
            # environment's seasonality moving while the calendar stays put.
            shift = ev.params["shift_rows"]
            for c in ev.params["columns"]:
                out.loc[mask, c] = out[c].shift(shift).loc[mask].to_numpy()
        elif ev.kind == "category_remap":
            out.loc[mask, col] = out.loc[mask, col].astype(str).map(ev.params["mapping"]).values
        else:
            raise ValueError(ev.kind)
    return out


def bike_events(n_train: int, n_val: int, n_eval: int, smoke: bool = False) -> list[PlantedEvent]:
    """Event offsets are relative to the evaluation window so the smoke run
    can truncate the window and still keep every event inside it."""
    spans = [(20, 27), (60, 67), (100, 107)] if smoke else [(30, 44), (150, 164), (280, 294)]
    base = n_train + n_val
    kinds = [
        ("truncate", "temp", {"cap": 0.35}, "temperature clipped above 0.35 (simulated sensor fault)"),
        ("sign_flip", "windspeed", {}, "windspeed sign inverted (simulated pipeline bug)"),
        ("season_shift", "temp",
         {"shift_rows": 91 * 24, "columns": ["temp", "atemp", "hum", "windspeed"]},
         "weather columns replay values from 91 days earlier (simulated seasonality shift "
         "of the environment; a bare relabeling of the month field was considered and "
         "rejected: univariate marginal monitors are blind to which value a calendar "
         "field concentrates on)"),
    ]
    events = []
    for (s, e), (kind, feat, params, desc) in zip(spans, kinds):
        events.append(PlantedEvent(
            event_id=f"bike_{kind}_{feat}", kind=kind, feature=feat,
            start=base + s, end=base + min(e, n_eval - 1), params=params, description=desc))
    return events


def shoppers_events(n_train: int, n_val: int, n_eval: int, smoke: bool = False) -> list[PlantedEvent]:
    # Eval months are Sep, Oct, Nov, Dec; Oct (second eval month) stays clean.
    base = n_train + n_val
    return [
        PlantedEvent("shoppers_truncate_PageValues", "truncate", "PageValues",
                     base + 0, base + 0, {"cap": 50.0},
                     "PageValues clipped above 50 (simulated tracking change)"),
        PlantedEvent("shoppers_sign_flip_ExitRates", "sign_flip", "ExitRates",
                     base + 2, base + 2, {},
                     "ExitRates sign inverted (simulated pipeline bug)"),
        PlantedEvent("shoppers_category_remap_VisitorType", "category_remap", "VisitorType",
                     base + 3, base + 3,
                     {"mapping": {"Returning_Visitor": "New_Visitor",
                                  "New_Visitor": "Returning_Visitor",
                                  "Other": "Other"}},
                     "visitor-type labels swapped (simulated tracking bug; replaces the "
                     "seasonality shift used for bike, because the replayed Month field is "
                     "degenerate under univariate drift checks: later calendar months never "
                     "appear in the reference window, so remapping them changes nothing)"),
    ]
