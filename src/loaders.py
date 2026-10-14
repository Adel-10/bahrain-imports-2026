"""Read the portal's CSV files into one consistent shape.

The yearly import files were published with small differences:
  * months are written "January" (2021-2022), "01 January" (2023, 2025, 2026)
    or "01-January" (2024);
  * the 2024 file has no year column and uses transliterated Arabic names
    for the value, weight, quantity and unit columns;
  * the 2023 file has no row-number column.

`load_imports` hides those differences and returns the same columns for every
year. `load_route` does the same for the port and airport files.
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path

import pandas as pd

MONTH_NUMBER = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}

# Portal column name -> our column name. Both naming schemes map to the same names.
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
    # 2024 file
    "qym_lwrdt_dynr_bhryny": "value_bd",
    "qym_lwrdt_dwlr_mryky": "value_usd",
    "wzn_lwrdt_kjm": "weight_kg",
    "kmy_lwrdt": "quantity",
    "whd_lqys": "unit",
}

# Only these source columns are read; the Arabic text columns and row numbers are skipped to save memory.
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
    """'January', '01 January' or '01-January' -> 1."""
    name = re.sub(r"^\s*\d{1,2}\s*[-\s]\s*", "", str(label)).strip().lower()
    try:
        return MONTH_NUMBER[name]
    except KeyError:
        raise ValueError(f"Unrecognised month label: {label!r}") from None


def _read_csv(path: str | Path, usecols=None) -> pd.DataFrame:
    # Everything is read as text first, so nothing is silently converted:
    #  * product codes keep their leading zeros ("01012910", not 1012910);
    #  * keep_default_na=False stops pandas turning Namibia's code "NA" into a missing value.
    # utf-8-sig copes with or without a byte-order mark at the start of the file.
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig", usecols=usecols)


def _month_numbers(labels: pd.Series) -> pd.Series:
    # Only a dozen distinct labels, so parse each once and map.
    lookup = {label: parse_month(label) for label in labels.dropna().unique()}
    return labels.map(lookup).astype("int8")


def load_imports(path: str | Path, year: int) -> pd.DataFrame:
    """Load one yearly imports file into the standard columns listed in IMPORT_COLUMNS."""
    raw = _read_csv(path, usecols=lambda col: col in IMPORT_SOURCE_COLUMNS)

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
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().astype("category")
    df["year"] = df["year"].astype("int16")
    return df[IMPORT_COLUMNS]


def load_route(path: str | Path, name: str) -> pd.DataFrame:
    """Load a port or airport file into: year, month, date, series, value."""
    series_col, value_col = ROUTE_COLUMNS[name]
    raw = _read_csv(path)
    df = pd.DataFrame({
        "year": pd.to_numeric(raw["year"].str[:4]).astype("int16"),
        "month": _month_numbers(raw["month"]),
        "series": raw[series_col].str.strip() if series_col else value_col,
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    # Some files include blank placeholder rows for months not yet published
    # (the airport file has empty Sep-Dec 2026 rows), so drop rows with no value.
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    return df[["year", "month", "date", "series", "value"]].sort_values(["series", "date"], ignore_index=True)
