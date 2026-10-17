"""Tests for the step 6 helpers: route groups, comparisons with 2025, elimination and adapting.

    pytest tests/test_routes.py

The elimination test is worked through by hand: given last year's weight and containers, this
year's containers and air cargo, the minimum for other routes must come out exactly. The adapting
test checks the "came back x% of the way" formula, including totals that fell further or ended
above normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from routes import (BEYOND, GCC, HEAVY, LIGHT, MIDDLE, VEHICLES, adapting, compare_with_2025, elimination,
                    origin_route, weight_class)


def test_origin_routes():
    assert origin_route("GCC") == GCC
    assert origin_route("Other Middle East") == "Other Middle East"
    assert origin_route("Europe") == BEYOND
    assert origin_route("Other and not specified") == BEYOND


def test_weight_classes():
    assert weight_class(16, 9.4) == LIGHT
    assert weight_class(16, 5.0) == LIGHT           # the threshold itself counts as light
    assert weight_class(15, 0.3) == HEAVY
    assert weight_class(4, 2.0) == MIDDLE
    assert weight_class(17, 5.2) == VEHICLES        # vehicles are kept apart whatever their value per kg
    assert weight_class(7, np.nan) == MIDDLE


def monthly(values_2025, values_2026):
    dates = list(pd.date_range("2025-01-01", periods=12, freq="MS")) + \
        list(pd.date_range("2026-01-01", periods=len(values_2026), freq="MS"))
    return pd.Series(list(values_2025) + list(values_2026), index=dates, dtype=float)


def test_compare_with_2025_uses_only_months_with_data():
    s = monthly([10.0] * 12, [10.0, 10.0, 5.0, np.nan])
    full = compare_with_2025(s, [1, 2])
    assert full["pct_of_normal"] == pytest.approx(100.0)
    part = compare_with_2025(s, [3, 4])                 # April 2026 missing: compare March with March only
    assert part["months"] == [3]
    assert part["pct_of_normal"] == pytest.approx(50.0)
    assert np.isnan(compare_with_2025(s, [5, 6])["pct_of_normal"])


def test_elimination_minimum():
    # 2025: 100 t of core goods a month, 10 containers. 2026 March: 70 t arrived, 3 containers, 1 t by air.
    dates = list(pd.date_range("2025-01-01", periods=12, freq="MS")) + [pd.Timestamp("2026-03-01")]
    core = pd.DataFrame({"date": dates, "region": ["GCC"] * 13,
                         "weight_kg": [100_000.0] * 12 + [70_000.0]})
    route = pd.DataFrame({"date": dates, "port_import_teu": [10.0] * 12 + [3.0],
                          "air_import_t": [2.0] * 12 + [1.0], "port_general_cargo_ft": [5.0] * 13})
    out = elimination(core, route, {"March": [3]}).iloc[0]
    # Containers at most 100 x 3/10 = 30 t; planes 1 t; so at least 70 - 30 - 1 = 39 t another way.
    assert out["containers_max_t"] == pytest.approx(30.0)
    assert out["other_min_t"] == pytest.approx(39.0)
    assert out["other_min_share"] == pytest.approx(39 / 70 * 100)
    assert out["gcc_share_2026"] == pytest.approx(100.0)


def test_adapting_came_back():
    rows = []
    for label, values in {"Bahrain": (46, 64, 82), "GCC": (85, 108, 119), "Africa": (64, 81, 55)}.items():
        for period, v in zip(["March", "Apr-May", "Jun-Jul"], values):
            rows.append({"level": "x", "label": label, "period": period, "actual": v, "forecast": 100.0,
                         "outside": True, "range_pct": 8.0})
        rows.append({"level": "x", "label": label, "period": "Since the war (Mar-Jul)", "actual": 1.0,
                     "forecast": 100.0, "outside": True, "range_pct": 4.0})
    out = adapting(pd.DataFrame(rows), ["March", "Apr-May", "Jun-Jul"]).set_index("label")
    assert out.loc["Bahrain", "came_back_pct"] == pytest.approx((82 - 46) / (100 - 46) * 100)
    assert out.loc["Bahrain", "worst_period"] == "March"
    assert out.loc["GCC", "came_back_pct"] > 100           # ended above normal
    assert out.loc["Africa", "worst_period"] == "Jun-Jul"   # fell further: nothing came back
    assert out.loc["Africa", "came_back_pct"] == pytest.approx(0.0)
