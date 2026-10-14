"""Step 1b: confirm the downloaded files are complete and readable.

    python src/check_downloads.py

For every imports file it loads the data with the project's loader, then compares
each month's total value and weight with the figures measured on the portal on
2026-10-11 (config.REFERENCE_MONTHLY_IMPORTS). It also reports things that
matter later: product codes that aren't 8 digits, unreadable numbers, and
product-country-month combinations that appear more than once.
"""

from __future__ import annotations

import sys

import pandas as pd

from config import DATASETS, MANIFEST_PATH, RAW_DIR, REFERENCE_MONTHLY_IMPORTS, REFERENCE_ROUTE_POINTS
from loaders import load_imports, load_route

TOLERANCE = 1  # reference figures are rounded to whole millions / thousand tonnes


def check_imports() -> tuple[list[str], list[str]]:
    problems, notes = [], []
    print("IMPORTS")
    print(f"{'year':<6}{'rows':>10}{'months':>8}{'value (BD m)':>14}{'weight (kt)':>13}   vs portal")
    for ds in (d for d in DATASETS if d.kind == "imports"):
        path = RAW_DIR / f"{ds.name}.csv"
        if not path.exists():
            problems.append(f"{ds.name}: file missing, run src/download_data.py")
            print(f"{ds.year:<6}{'missing':>10}")
            continue
        try:
            df = load_imports(path, ds.year)
        except ValueError as err:
            problems.append(f"{ds.name}: could not be read: {err}")
            print(f"{ds.year:<6}{'unreadable':>10}")
            continue

        monthly = df.groupby("month")[["value_bd", "weight_kg"]].sum()
        value_m = (monthly["value_bd"] / 1e6).round().astype(int)
        weight_kt = (monthly["weight_kg"] / 1e6).round().astype(int)

        reference = REFERENCE_MONTHLY_IMPORTS.get(ds.year, [])
        matched, mismatched = 0, []
        for month, (ref_value, ref_weight) in enumerate(reference, 1):
            if month not in monthly.index:
                mismatched.append(f"month {month} missing")
            elif abs(value_m[month] - ref_value) <= TOLERANCE and abs(weight_kt[month] - ref_weight) <= TOLERANCE:
                matched += 1
            else:
                mismatched.append(f"month {month}: {value_m[month]}/{weight_kt[month]} vs {ref_value}/{ref_weight}")
        extra = sorted(set(monthly.index) - set(range(1, len(reference) + 1)))

        verdict = f"{matched}/{len(reference)} months match"
        if extra:
            verdict += f", plus new month(s) {extra}"
        print(f"{ds.year:<6}{len(df):>10,}{monthly.shape[0]:>8}{value_m.sum():>14,}{weight_kt.sum():>13,}   {verdict}")

        if mismatched:
            problems.append(f"{ds.name}: " + "; ".join(mismatched))
        if extra:
            notes.append(f"{ds.name}: the portal has added month(s) {extra} since 2026-10-11, so there is new data to use")

        bad_codes = (~df["hs8"].astype(object).str.fullmatch(r"\d{8}")).sum()
        if bad_codes:
            notes.append(f"{ds.name}: {bad_codes:,} row(s) with a product code that isn't 8 digits")
        unreadable = df[["value_bd", "weight_kg"]].isna().any(axis=1).sum()
        if unreadable:
            notes.append(f"{ds.name}: {unreadable:,} row(s) with a missing or unreadable value/weight")
        repeats = df.duplicated(["month", "hs8", "iso2"], keep=False).sum()
        if repeats:
            notes.append(f"{ds.name}: {repeats:,} rows share a month+product+country with another row (to handle in cleaning)")
    return problems, notes


def check_routes() -> tuple[list[str], list[str]]:
    problems, notes = [], []
    print("\nROUTE TOTALS")
    loaded: dict[str, pd.DataFrame] = {}
    for ds in (d for d in DATASETS if d.kind == "route"):
        path = RAW_DIR / f"{ds.name}.csv"
        if not path.exists():
            problems.append(f"{ds.name}: file missing, run src/download_data.py")
            print(f"{ds.name:<18}missing")
            continue
        df = load_route(path, ds.name)
        loaded[ds.name] = df
        first, last = df["date"].min(), df["date"].max()
        print(f"{ds.name:<18}{len(df):>5} rows   {first:%b %Y} to {last:%b %Y}   series: {df['series'].nunique()}")
        if ds.name == "port_general_cargo":
            typos = df[(df["value"] > 0) & (df["value"] < 1_000)]
            if len(typos):
                months = ", ".join(f"{d:%b %Y}" for d in typos["date"])
                notes.append(f"port_general_cargo: {len(typos)} month(s) look like thousands typed with a decimal point "
                             f"(for example {typos['value'].iloc[0]:g}): {months}")

    for name, series, year, months, expected in REFERENCE_ROUTE_POINTS:
        if name not in loaded:
            continue
        df = loaded[name]
        actual = df.loc[(df["series"] == series) & (df["year"] == year) & (df["month"].isin(months)), "value"].sum()
        if abs(actual - expected) > 0.5:
            problems.append(f"{name}: {series} {year} months {list(months)} = {actual:,.0f}, expected {expected:,.0f}")
    return problems, notes


def main() -> int:
    if not MANIFEST_PATH.exists():
        print("No data/raw/MANIFEST.json yet. Run: python src/download_data.py")
        return 1

    problems, notes = check_imports()
    route_problems, route_notes = check_routes()
    problems += route_problems
    notes += route_notes

    if notes:
        print("\nNOTES (expected; we deal with these in step 2)")
        for n in notes:
            print(f"  - {n}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll checks passed: every file is complete and matches the portal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
