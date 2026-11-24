import numpy as np
import pandas as pd

from driftmon.planting import PlantedEvent, apply_events


def _df():
    return pd.DataFrame({
        "_pidx": [0, 1, 1, 2, 3, 3],
        "temp": [0.1, 0.6, 0.9, 0.3, 0.8, 0.2],
        "windspeed": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "mnth": [1, 2, 2, 3, 4, 4],
        "Month": ["Sep", "Sep", "Oct", "Nov", "Dec", "Dec"],
    })


def test_truncate_only_inside_window():
    ev = PlantedEvent("e", "truncate", "temp", 1, 2, {"cap": 0.5}, "")
    out = apply_events(_df(), [ev])
    assert out.loc[out["_pidx"].between(1, 2), "temp"].max() <= 0.5
    assert out.loc[out["_pidx"] == 3, "temp"].max() == 0.8  # untouched
    assert out.loc[out["_pidx"] == 0, "temp"].iloc[0] == 0.1


def test_sign_flip_negates_only_window():
    ev = PlantedEvent("e", "sign_flip", "windspeed", 2, 2, {}, "")
    out = apply_events(_df(), [ev])
    assert (out.loc[out["_pidx"] == 2, "windspeed"] < 0).all()
    assert (out.loc[out["_pidx"] != 2, "windspeed"] > 0).all()


def test_month_shift_stays_in_range():
    ev = PlantedEvent("e", "month_shift", "mnth", 0, 3, {"shift": 3}, "")
    out = apply_events(_df(), [ev])
    assert out["mnth"].between(1, 12).all()
    assert (out["mnth"] != _df()["mnth"]).all()


def test_category_remap_swaps_labels_only_inside_window():
    df = _df()
    df["VisitorType"] = ["Returning_Visitor", "New_Visitor", "Returning_Visitor",
                         "Returning_Visitor", "New_Visitor", "Other"]
    ev = PlantedEvent("e", "category_remap", "VisitorType", 3, 3,
                      {"mapping": {"Returning_Visitor": "New_Visitor",
                                   "New_Visitor": "Returning_Visitor",
                                   "Other": "Other"}}, "")
    out = apply_events(df, [ev])
    assert list(out.loc[out["_pidx"] == 3, "VisitorType"]) == ["Returning_Visitor", "Other"]
    assert list(out.loc[out["_pidx"] != 3, "VisitorType"]) == list(
        df.loc[df["_pidx"] != 3, "VisitorType"])


def test_season_shift_replaces_weather_from_earlier_rows():
    n = 300
    df = pd.DataFrame({
        "_pidx": [i // 24 for i in range(n)],
        "temp": [float(i % 97) for i in range(n)],
        "atemp": [float(i % 89) for i in range(n)],
    })
    ev = PlantedEvent("e", "season_shift", "temp", 5, 6,
                      {"shift_rows": 48, "columns": ["temp"]}, "")
    out = apply_events(df, [ev])
    mask = out["_pidx"].between(5, 6)
    import numpy as np
    np.testing.assert_array_equal(out.loc[mask, "temp"].to_numpy(),
                                  df["temp"].shift(48).loc[mask].to_numpy())
    # untouched outside the window
    assert out.loc[~mask, "temp"].equals(df.loc[~mask, "temp"])
    assert out.loc[~mask, "atemp"].equals(df.loc[~mask, "atemp"])
