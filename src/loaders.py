"""Read the portal's CSV files into one consistent shape.

The six yearly import files were published with small differences, and the analysis should
never have to know about them:

* months are written "January" (2021-2022), "01 January" (2023, 2025, 2026) or
  "01-January" (2024);
* the 2024 file has no year column, and its value, weight, quantity and unit columns have
  transliterated Arabic names (for example ``qym_lwrdt_dynr_bhryny`` for the value in BD);
* the 2023 file has no row-number column.

``load_imports`` hides those differences and returns the same columns for every year.
``load_route`` does the same for the port and airport files, which come in two layouts: a
"long" file with a column naming the series (container type, vessel type, ...) and a
single-series file with just a date and a number.

Two reading rules protect the data:

* Everything is read as text first and converted deliberately. Otherwise pandas would turn
  product codes like "01012910" into the number 1012910 (losing the leading zero, which is
  part of the code), and Namibia's country code "NA" into a missing value.
* A file's year column is checked against the year it is supposed to contain, so a file
  saved under the wrong name is caught immediately.
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path

import pandas as pd

MONTH_NUMBER = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}   # "january" -> 1

# Portal column name -> project column name. Both naming schemes map to the same names.
IMPORT_RENAME = {
    "commodity_no": "hs8",
    "commodity": "commodity_en",
    "un_code": "iso2",
    "country_name": "country_en",
    "import_value_bd": "value_bd",
    "import_value_usa": "value_usd",
    "import_weight_kg": "weight_kg",
    "import_quantity": "quantity",
    "um": "unit",
    # 2024 file (Arabic names written in Latin letters)
    "qym_lwrdt_dynr_bhryny": "value_bd",
    "qym_lwrdt_dwlr_mryky": "value_usd",
    "wzn_lwrdt_kjm": "weight_kg",
    "kmy_lwrdt": "quantity",
    "whd_lqys": "unit",
}

# Only these source columns are read. The Arabic text columns and row numbers are skipped,
# which roughly halves the memory needed for a 320,000-row file.
IMPORT_SOURCE_COLUMNS = {"year", "month", *IMPORT_RENAME}

IMPORT_COLUMNS = [
    "year", "month", "hs8", "commodity_en", "iso2", "country_en",
    "value_bd", "value_usd", "weight_kg", "quantity", "unit",
]
NUMERIC_COLUMNS = ["value_bd", "value_usd", "weight_kg", "quantity"]
TEXT_COLUMNS = ["hs8", "commodity_en", "iso2", "country_en", "unit"]

# For each route file: (column holding the series name, column holding the number).
# The RoRo and general cargo files hold a single series, so they have no series column;
# the number column's name is used as the series name instead.
ROUTE_COLUMNS = {
    "port_containers": ("container_type", "teus"),
    "port_vessels": ("vessel_type", "vessels"),
    "airport_cargo": ("data_type", "value"),
    "port_roro": (None, "roro_units_vehicles"),
    "port_general_cargo": (None, "general_cargo_volume_ft"),
}


def parse_month(label: str) -> int:
    """'January', '01 January' or '01-January' -> 1. Anything else raises an error."""
    name = re.sub(r"^\s*\d{1,2}\s*[-\s]\s*", "", str(label)).strip().lower()   # drop a leading "01 " or "01-"
    try:
        return MONTH_NUMBER[name]
    except KeyError:
        raise ValueError(f"Unrecognised month label: {label!r}") from None


def _read_csv(path: str | Path, usecols=None) -> pd.DataFrame:
    """Read a portal CSV with every column as text.

    keep_default_na=False stops pandas treating "NA" (Namibia) as missing; only truly empty
    cells become missing. utf-8-sig copes with or without a byte-order mark at the start.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig", usecols=usecols)


def _month_numbers(labels: pd.Series) -> pd.Series:
    # A file has only a dozen distinct month labels, so each is parsed once and then looked up.
    lookup = {label: parse_month(label) for label in labels.dropna().unique()}
    return labels.map(lookup).astype("int8")


def load_imports(path: str | Path, year: int) -> pd.DataFrame:
    """Load one yearly imports file into the standard columns listed in IMPORT_COLUMNS."""
    raw = _read_csv(path, usecols=lambda col: col in IMPORT_SOURCE_COLUMNS)

    # Guard against a file saved under the wrong year (the 2024 file has no year column to check).
    if "year" in raw.columns:
        years_in_file = set(raw["year"].dropna().str.strip().unique())
        if years_in_file != {str(year)}:
            raise ValueError(f"{Path(path).name}: expected year {year}, file contains {sorted(years_in_file)}")

    df = raw.rename(columns=IMPORT_RENAME)
    missing = [c for c in IMPORT_COLUMNS if c not in df.columns and c != "year"]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing}; found {list(raw.columns)}")

    df["year"] = year
    df["month"] = _month_numbers(df["month"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")    # unreadable numbers become missing
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().astype("category")                 # categories save a lot of memory
    df["year"] = df["year"].astype("int16")
    return df[IMPORT_COLUMNS]


def load_route(path: str | Path, name: str) -> pd.DataFrame:
    """Load a port or airport file into: year, month, date, series, value (one row per series and month)."""
    series_col, value_col = ROUTE_COLUMNS[name]
    raw = _read_csv(path)
    df = pd.DataFrame({
        "year": pd.to_numeric(raw["year"].str[:4]).astype("int16"),
        "month": _month_numbers(raw["month"]),
        "series": raw[series_col].str.strip() if series_col else value_col,
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    # Some files include blank placeholder rows for months not yet published
    # (the airport file has empty rows for the rest of 2026), so rows with no value are dropped.
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    return df[["year", "month", "date", "series", "value"]].sort_values(["series", "date"], ignore_index=True)
