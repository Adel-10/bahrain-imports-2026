"""Tests for the step 2 cleaning functions and reference tables.

Run with:  pytest
"""

from __future__ import annotations

import pandas as pd
import pytest

from build_dataset import add_flags, add_hierarchy, latest_wording, merge_repeats
from reference import HS_SECTIONS, build_calendar, chapter_to_section


def raw_rows() -> pd.DataFrame:
    # Two shipments of the same product from the same country in the same month,
    # plus one unrelated row and one with the "#N/A" unit seen in the source.
    return pd.DataFrame({
        "year": [2025, 2025, 2025, 2025],
        "month": [8, 8, 8, 9],
        "hs8": ["03028400", "03028400", "20059911", "84819000"],
        "iso2": ["SA", "SA", "SA", "CN"],
        "value_bd": [11495.620, 80726.555, 152.64, 10.0],
        "value_usd": [30573.4, 214698.3, 405.9, 26.6],
        "weight_kg": [5600.0, 54000.0, 16.0, 0.0],
        "quantity": [5600.0, 54000.0, 16.0, 3.0],
        "unit": ["KG", "KG", "KG", "#N/A"],
    })


def test_merge_repeats_adds_shipments_and_keeps_totals():
    raw = raw_rows()
    merged = merge_repeats(raw)
    assert len(merged) == 3
    seabass = merged[merged["hs8"] == "03028400"].iloc[0]
    assert seabass["value_bd"] == pytest.approx(11495.620 + 80726.555)
    assert seabass["weight_kg"] == 59600.0
    assert seabass["n_shipments"] == 2
    assert merged["value_bd"].sum() == pytest.approx(raw["value_bd"].sum())
    assert pd.isna(merged.loc[merged["hs8"] == "84819000", "unit"].iloc[0])   # "#N/A" becomes missing


def test_merge_repeats_refuses_mixed_units():
    raw = raw_rows()
    raw.loc[1, "unit"] = "NO"
    with pytest.raises(ValueError, match="mix units"):
        merge_repeats(raw)


def test_hierarchy_and_groups():
    df = add_hierarchy(pd.DataFrame({"hs8": ["01012910", "26011100", "84819000", "71131900", "98010000"]}))
    assert df["hs4"].tolist() == ["0101", "2601", "8481", "7113", "9801"]
    assert df["section_id"].tolist() == [1, 5, 16, 14, 22]
    assert df["product_group"].tolist() == ["General goods", "Iron ore", "General goods", "General goods", "General goods"]


def test_every_chapter_has_a_section():
    for chapter in range(1, 100):
        number, name = chapter_to_section(chapter)
        assert 1 <= number <= len(HS_SECTIONS) and name


def test_flags():
    df = add_flags(pd.DataFrame({"value_bd": [100.0, 0.0, 5e6, 50.0], "weight_kg": [10.0, 5.0, 1.0, 0.0]}))
    assert df["flag_zero_weight"].tolist() == [False, False, False, True]
    assert df["flag_nonpositive_value"].tolist() == [False, True, False, False]
    assert df["flag_implausible_unit_value"].tolist() == [False, False, True, False]


def test_latest_wording_prefers_latest_year_then_most_common():
    table = pd.DataFrame({
        "hs8": ["01", "01", "01", "02"],
        "commodity_en": ["old name", "new name ", "new nme", "only"],
        "n": [50, 10, 2, 1],
        "year": [2021, 2025, 2025, 2023],
    })
    wording = latest_wording(table, "hs8", "commodity_en")
    assert wording["01"] == "new name"
    assert wording["02"] == "only"


def test_calendar_matches_known_dates():
    cal = build_calendar("2021-01", "2026-12").set_index("date")
    # Ramadan 1447 ran 18 Feb - 19 Mar 2026; Eid al-Fitr 20 Mar; Eid al-Adha 27 May.
    assert cal.loc["2026-02-01", "ramadan_days"] == 11
    assert cal.loc["2026-03-01", "ramadan_days"] == 19
    assert cal.loc["2026-03-01", "eid_al_fitr"] == 1
    assert cal.loc["2026-05-01", "eid_al_adha"] == 1
    # Ramadan 1442 began 13 April 2021.
    assert cal.loc["2021-04-01", "ramadan_days"] == 18
    # COVID restrictions end with March 2022; closures 27 May - 9 June 2021.
    assert cal.loc["2022-03-01", "covid_restrictions"] == 1
    assert cal.loc["2022-04-01", "covid_restrictions"] == 0
    assert cal.loc["2021-05-01", "covid_closure_days"] == 5
    assert cal.loc["2021-06-01", "covid_closure_days"] == 9
    # Every year has about 29-30 Ramadan days in total.
    per_year = cal.groupby(cal.index.year)["ramadan_days"].sum()
    assert per_year.between(29, 30).all()
