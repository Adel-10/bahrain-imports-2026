"""Tests for step 4: the training table, the models and the scoring.

Run with:  pytest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import coverage, miss, ranges_from
from features import NUMERIC, Panel, build_panel, make_examples
from models import LinearModel, SameMonth, SimpleAverage


def toy_core() -> pd.DataFrame:
    """Three HS2 chapters in one section and region, Jan 2021 - Dec 2023.
    Chapter 01 has imports every month, 02 only every third month (patchy), 03 never after 2021."""
    months = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
    rows = []
    for i, m in enumerate(months):
        rows.append({"date": m, "section_id": 1, "hs2": "01", "region": "GCC", "value_bd": 100.0 + i})
        if i % 3 == 0:
            rows.append({"date": m, "section_id": 1, "hs2": "02", "region": "GCC", "value_bd": 30.0})
        if m.year == 2021:
            rows.append({"date": m, "section_id": 1, "hs2": "03", "region": "GCC", "value_bd": 5.0})
    return pd.DataFrame(rows)


def toy_calendar() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    return pd.DataFrame({"days_in_month": months.days_in_month, "ramadan_days": 0, "eid_al_fitr": 0, "eid_al_adha": 0,
                         "covid_restrictions": 0, "covid_closure_days": 0}, index=months)


def test_patchy_series_are_pooled_and_totals_kept():
    core = toy_core()
    panel = build_panel(core, "2021-01-01", "2022-12-01", "2023-12-01")
    assert list(panel.info["hs2"]) == ["01", "other"]
    assert panel.values.sum() == pytest.approx(core["value_bd"].sum())


def test_features_only_use_the_past():
    core = toy_core()
    panel = build_panel(core, "2021-01-01", "2022-12-01", "2023-12-01")
    origin = pd.Timestamp("2022-12-01")
    before = make_examples(panel, toy_calendar(), [origin])
    # Change every value after the origin: the features must not move, only the targets.
    future = Panel(values=panel.values.copy(), months=panel.months, info=panel.info)
    future.values[:, panel.months > origin] *= 50
    after = make_examples(future, toy_calendar(), [origin])
    features = [c for c in NUMERIC if c in before]
    pd.testing.assert_frame_equal(before[features], after[features])
    assert not np.allclose(before["actual"], after["actual"])


def test_training_rows_stop_before_the_test_year():
    core = toy_core()
    panel = build_panel(core, "2021-01-01", "2022-12-01", "2023-12-01")
    origins = list(pd.date_range("2021-12-01", "2022-11-01", freq="MS"))
    rows = make_examples(panel, toy_calendar(), origins, last_target=pd.Timestamp("2022-12-01"))
    assert rows["target_month"].max() == pd.Timestamp("2022-12-01")
    assert set(rows["horizon"]) <= set(range(1, 13))
    first = rows[(rows["series"] == 0) & (rows["origin"] == origins[0]) & (rows["horizon"] == 1)].iloc[0]
    assert first["scale"] == pytest.approx(np.mean(100 + np.arange(12)))          # average of Jan-Dec 2021
    assert first["actual"] == pytest.approx(112)                                   # January 2022


def test_gradient_descent_matches_normal_equations():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 4))])
    y = X @ np.array([1.0, 0.5, -0.3, 0.0, 0.2]) + rng.normal(scale=0.1, size=200)
    w = rng.uniform(0.5, 1.5, size=200)
    w = w / w.sum()
    penalty = np.array([0.0, 0.01, 0.01, 0.01, 0.01])
    exact = LinearModel.solve_normal_equations(X, y, w, penalty)
    gd = LinearModel(n_iter=3000).gradient_descent(X, y, w, penalty)
    np.testing.assert_allclose(gd, exact, atol=1e-8)
    model = LinearModel()
    model.gradient_descent(X, y, w, penalty)
    assert all(a >= b - 1e-12 for a, b in zip(model.loss_history, model.loss_history[1:]))   # never goes uphill


def test_linear_model_on_a_table():
    # The target is exactly 1.1 x last month (as a ratio); the model should get very close with a tiny penalty.
    rng = np.random.default_rng(1)
    n = 500
    rows = pd.DataFrame({name: rng.uniform(0.5, 1.5, n) for name in NUMERIC})
    rows["section_id"] = rng.integers(1, 4, n)
    rows["region_id"] = rng.integers(0, 3, n)
    rows["hs2_id"] = 0
    rows["scale"] = rng.uniform(1, 100, n)
    rows["target"] = 1.1 * rows["last_1"]
    model = LinearModel(ridge=1e-8).fit(rows)
    np.testing.assert_allclose(model.predict(rows), rows["target"], atol=1e-6)


def test_simple_rules():
    rows = pd.DataFrame({"same_month_1y": [0.8, np.nan, 1.3]})
    assert SimpleAverage().predict(rows).tolist() == [1.0, 1.0, 1.0]
    assert SameMonth().predict(rows).tolist() == [0.8, 1.0, 1.3]


def test_miss_ranges_and_coverage():
    months = pd.date_range("2024-01-01", periods=4, freq="MS")
    frame = pd.DataFrame({"target_month": list(months) * 2, "series": [0] * 4 + [1] * 4,
                          "actual": [100, 100, 100, 100, 50, 50, 50, 50],
                          "forecast": [110, 90, 100, 100, 50, 50, 50, 50]})
    # Nationally the totals are 160, 140, 150, 150 against 150 each month: misses of 10 + 10 out of 600.
    assert miss(frame, []) == pytest.approx(20 / 600 * 100)
    errors = pd.DataFrame({"level": "National total", "name": "Bahrain", "target_month": months,
                           "error": [0.05, -0.10, 0.02, 0.30]})
    ranges = ranges_from(errors)
    assert ranges["range"].iloc[0] == pytest.approx(np.quantile([0.05, 0.10, 0.02, 0.30], 0.9))
    assert coverage(errors, ranges)["share inside"].iloc[0] == pytest.approx(0.75)
