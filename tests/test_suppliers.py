"""Tests for the supplier-concentration extra.

    pytest tests/test_suppliers.py

Checks the effective number of suppliers against hand-worked examples, the supplier profile of a
small product table, that the weighted regression recovers a known line, and that switching to
minor and brand-new suppliers is counted correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from suppliers import effective_suppliers, supplier_profile, switching, weighted_least_squares


def test_effective_suppliers():
    assert effective_suppliers(pd.Series([50.0, 50.0])) == pytest.approx(2.0)
    assert effective_suppliers(pd.Series([90.0, 10.0])) == pytest.approx(1 / (0.81 + 0.01))
    assert effective_suppliers(pd.Series([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4.0)
    assert effective_suppliers(pd.Series([7.0])) == pytest.approx(1.0)


def toy_imports() -> pd.DataFrame:
    # Product 0101: half from China, half from Japan (far away). Product 0201: all from Saudi Arabia.
    rows = []
    for year in (2023, 2024, 2025, 2026):
        for month in range(1, 13):
            date = pd.Timestamp(year, month, 1)
            rows += [(date, 1, "0101", "CN", "East and South-East Asia", 50.0, 10.0),
                     (date, 1, "0101", "JP", "East and South-East Asia", 50.0, 10.0),
                     (date, 4, "0201", "SA", "GCC", 100.0, 100.0)]
    df = pd.DataFrame(rows, columns=["date", "section_id", "hs4", "iso2", "region", "value_bd", "weight_kg"])
    # In 2026 product 0101 also buys from the UAE, a supplier it never used before.
    extra = df[(df["date"].dt.year == 2026) & (df["hs4"] == "0101") & (df["iso2"] == "CN")].assign(
        iso2="AE", region="GCC", value_bd=25.0)
    return pd.concat([df, extra], ignore_index=True)


def test_supplier_profile():
    profile = supplier_profile(toy_imports())
    assert profile.loc["0101", "effective_suppliers"] == pytest.approx(2.0)
    assert profile.loc["0201", "effective_suppliers"] == pytest.approx(1.0)
    assert profile.loc["0101", "beyond_share"] == pytest.approx(1.0)
    assert profile.loc["0201", "beyond_share"] == pytest.approx(0.0)
    assert profile.loc["0101", "bd_per_kg"] == pytest.approx(5.0)
    assert profile.loc["0101", "supplier_group"] == "2 to 3"
    assert profile.loc["0201", "beyond_group"] == "Under 25%"


def test_weighted_least_squares_recovers_a_known_line():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    X = np.column_stack([np.ones(200), x])
    y = 3.0 - 2.0 * x
    assert weighted_least_squares(X, y, rng.uniform(0.5, 2.0, 200)) == pytest.approx([3.0, -2.0])


def test_switching_counts_new_suppliers():
    data = toy_imports()
    out = switching(data, supplier_profile(data), years=(2025, 2026), months=(3,)).set_index(["year", "scope"])
    assert out.loc[(2025, "All products"), "minor_share"] == pytest.approx(0.0)
    # March 2026: 25 of 225 BD came from the UAE, which had supplied nothing before.
    assert out.loc[(2026, "All products"), "minor_share"] == pytest.approx(25 / 225 * 100)
    assert out.loc[(2026, "Far-dependent products"), "minor_from_GCC"] == pytest.approx(25 / 125 * 100)
    assert out.loc[(2026, "Far-dependent products"), "new_share"] == pytest.approx(25 / 125 * 100)
