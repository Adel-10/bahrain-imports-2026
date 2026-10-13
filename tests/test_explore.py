"""Tests for the step 3 helpers: origin regions, analysis groups and the simple-forecast measures.

Run with:  pytest
"""

from __future__ import annotations

import pandas as pd
import pytest

from explore import change_vs_2025, gap, months_of, normal_swing, wide
from reference import BULK, CORE, OTHER_REGION, PRECIOUS, REGION_CODES, add_analysis_group, add_region, region_of


def test_each_code_has_one_region():
    codes = [c for group in REGION_CODES.values() for c in group.split()]
    assert len(codes) == len(set(codes))


def test_regions():
    df = add_region(pd.DataFrame({"iso2": pd.Categorical(["SA", "FZ", "CN", "TR", "NA", "QX", "XB"])}))
    assert df["region"].astype(str).tolist() == [
        "GCC", "GCC", "East and South-East Asia", "Other Middle East", "Africa", OTHER_REGION, OTHER_REGION]
    assert region_of("ZZ") == OTHER_REGION


def test_analysis_groups():
    df = add_analysis_group(pd.DataFrame({
        "product_group": ["General goods", "General goods", "Iron ore", "General goods"],
        "hs2": ["84", "71", "26", "02"],
    }))
    assert df["group"].astype(str).tolist() == [CORE, PRECIOUS, BULK, CORE]


def monthly_table() -> pd.DataFrame:
    # Two series over 2024-2026: "flat" is 100 every month; "jumpy" alternates 50 / 150.
    dates = pd.date_range("2024-01-01", "2026-07-01", freq="MS")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "key": "flat", "value_bd": 100.0})
        rows.append({"date": d, "key": "jumpy", "value_bd": 50.0 if i % 2 == 0 else 150.0})
    df = pd.DataFrame(rows)
    df.loc[(df["key"] == "flat") & (df["date"] >= "2026-03-01"), "value_bd"] = 40.0   # a 60% drop from March 2026
    return wide(df, ["key"])


def test_change_vs_2025():
    w = monthly_table()
    assert change_vs_2025(w, [1, 2])["flat"] == pytest.approx(0.0)
    assert change_vs_2025(w, [3])["flat"] == pytest.approx(-60.0)


def test_gap_and_normal_swing():
    w = monthly_table()
    # The flat series is forecast perfectly; the jumpy one misses by 50 every month (forecast 100).
    assert gap(w.loc[["flat"]], 2025) == pytest.approx(0.0)
    assert gap(w.loc[["jumpy"]], 2025) == pytest.approx(50.0)
    assert gap(w, 2025) == pytest.approx(25.0)                       # 50 missed out of 200 forecast per month
    assert gap(w.loc[["flat"]], 2026, [3, 4]) == pytest.approx(60.0)
    # "Same month last year" is perfect for the alternating series (12 months = an even number of steps).
    assert gap(w.loc[["jumpy"]], 2025, forecast="same month") == pytest.approx(0.0)
    swing = normal_swing(w, years=(2025,))
    assert swing["flat"] == pytest.approx(0.0)
    assert swing["jumpy"] == pytest.approx(50.0)
    assert months_of(w, 2026).shape[1] == 7


def test_glossary_tables_use_known_terms():
    import glossary
    for step, keys in {3: ["core_goods", "national_total", "normal", "normal_miss", "normal_range", "containers"],
                       4: ["national_total", "normal_year", "miss", "overall_score"],
                       5: ["forecast", "normal", "gap", "normal_range", "since_war"]}.items():
        table = glossary.table(step, keys)
        assert table.count("\n") == len(keys) + 1
    assert "**National total** (also *nationally* or *All of Bahrain*)" in glossary.table(5, ["national_total"])
