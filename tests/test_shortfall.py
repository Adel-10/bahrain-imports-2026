"""Tests for step 5: periods, ranges for longer periods, and gaps.

    pytest tests/test_shortfall.py

Checks that new months published by the portal create new periods automatically, that ranges
for longer periods come out narrower when monthly errors partly cancel, and that gaps and BD lost
(with their likely range) are computed as described in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reference import war_periods_2026
from shortfall import build_periods, period_gaps, simple_rule_gaps, window_ranges


def test_periods_extend_when_new_months_arrive():
    assert war_periods_2026(7) == {"March": [3], "Apr-May": [4, 5], "Jun-Jul": [6, 7]}
    assert war_periods_2026(5) == {"March": [3], "Apr-May": [4, 5]}
    assert war_periods_2026(9)["Aug-Sep"] == [8, 9]
    assert war_periods_2026(8)["Aug"] == [8]
    periods, since = build_periods(9)
    assert list(periods)[0] == "Jan-Feb"
    assert since == {"Since the war (Mar-Sep)": [3, 4, 5, 6, 7, 8, 9]}


def toy_frame(year: int, forecast_scale: float = 1.0) -> pd.DataFrame:
    """One series in one section and region: actual 100 every month, forecast alternating 90 / 110 (x scale)."""
    months = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    return pd.DataFrame({"series": 0, "target_month": months, "actual": 100.0,
                         "forecast": [90.0 * forecast_scale if i % 2 == 0 else 110.0 * forecast_scale for i in range(12)],
                         "section_id": 1, "region": "GCC", "hs2": "01"})


def test_longer_periods_get_narrower_ranges():
    backtest = pd.concat([toy_frame(2024).assign(year=2024), toy_frame(2025).assign(year=2025)])
    ranges = window_ranges(backtest, {1, 2})
    national = ranges[ranges["level"] == "National total"].set_index("months")["range"]
    assert national[1] == pytest.approx(100 / 90 - 1)          # single months miss by up to 10/90
    assert national[2] == pytest.approx(0.0)                     # two months always add up to 200 against 200


def test_gaps_and_lost_imports():
    forecast = toy_frame(2026)
    forecast["actual"] = [100.0, 100.0, 50.0] + [100.0] * 9      # March is half its usual size
    ranges = pd.DataFrame({"level": ["National total"], "name": ["Bahrain"], "months": [1], "range": [0.1]})
    gaps = period_gaps(forecast, ranges, {"March": [3]})
    march = gaps[gaps["level"] == "National total"].iloc[0]
    assert march["forecast"] == 90 and march["actual"] == 50
    assert march["gap_pct"] == pytest.approx((50 / 90 - 1) * 100)
    assert bool(march["outside"])
    assert march["lost"] == pytest.approx(40)
    assert march["lost_low"] == pytest.approx(90 * 0.9 - 50) and march["lost_high"] == pytest.approx(90 * 1.1 - 50)


def test_simple_rule_gaps():
    months = pd.date_range("2022-01-01", "2026-07-01", freq="MS")
    values = np.where(months.year == 2026, 50.0, 100.0)
    w = pd.DataFrame([values], index=["Iron ore"], columns=months)
    gaps = simple_rule_gaps(w, {"March": [3], "Apr-May": [4, 5]}).set_index("period")
    assert gaps.loc["March", "gap_pct"] == pytest.approx(-50.0)
    assert gaps.loc["Apr-May", "forecast"] == pytest.approx(200.0)
    assert gaps.loc["March", "range_pct"] == pytest.approx(0.0)   # 2023-2025 never strayed from the year before
    assert bool(gaps.loc["March", "outside"])
