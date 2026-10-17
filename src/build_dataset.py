"""Step 2: clean the raw downloads and build the tables the analysis uses.

    python src/build_dataset.py

Reads data/raw/ and writes:
  data/processed/imports_clean.parquet  one row per month x 8-digit product x origin country
  data/processed/route_monthly.parquet  port and airport totals, one row per month
  data/reference/calendar.csv           Ramadan, Eid and COVID features, one row per month
  data/reference/hs_sections.csv        HS chapter -> section lookup
  reports/step2_cleaning_report.md      what was changed, and the checks that passed

The guiding rule: change as little as possible, and make every change visible
------------------------------------------------------------------------------
Every decision below came from profiling the real files first. The official monthly totals
must survive cleaning exactly, so nothing is dropped. Rows that look wrong are flagged
instead, and each analysis decides what to do with them.

1. **Combine the years.** ``loaders.load_imports`` reads the six yearly files into the same
   columns. Each year is processed separately (no product-country-month spans two years),
   which keeps memory low.
2. **Add up repeated rows.** A few products appear twice for the same country and month (6
   pairs in 1.76 million rows). They have different values, so they are separate shipments,
   not duplicates; they are added together so each product-country-month is one row. The
   script stops if two rows for the same key use different units, because their quantities
   could not be added.
3. **Check the totals.** Monthly value and weight are compared before and after, to 12
   significant figures. The script stops if anything changed.
4. **One description per code.** Product and country descriptions are worded slightly
   differently in different years, so each code gets the wording from the latest year it
   appears in.
5. **Add the classification.** Each 8-digit code gets its 6-digit, 4-digit and 2-digit
   parents, its HS section and its product group (see reference.py).
6. **Flag odd rows.** Zero weight, zero or negative value, and value per kg above BD 100,000
   (more than gold) are flagged, not removed. Together they are about 0.2% of value.
7. **Route figures.** The port and airport files are put side by side, one row per month. The
   port's general cargo file has 2026 months typed with a decimal point for a thousands
   separator; those are repaired and marked (see ``fix_general_cargo``).

Pros of this approach: totals still match the official statistics, every change is recorded,
and an analysis can always go back to the unflagged rows. Cons: flagged rows stay in the
totals, so an unusual row can still affect a small series (the models in step 4 reduce this
by working with larger groups).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from config import DATASETS, RAW_DIR, REFERENCE_DIR, ROOT
from loaders import load_imports, load_route
from reference import DEFAULT_GROUP, PRODUCT_GROUPS, build_calendar, chapter_to_section, hs_sections_table
from reporting import footer, md_table, write_markdown

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "step2_cleaning_report.md"

KEYS = ["year", "month", "hs8", "iso2"]   # one clean row per product, origin country and month
IMPLAUSIBLE_BD_PER_KG = 100_000  # above the price of gold (~30-50k BD/kg); almost certainly a data-entry error

# Series names in the portal's route files -> short column names in route_monthly.parquet.
# Any series not listed here stops the script, so a renamed or added series is noticed.
ROUTE_SERIES = {
    "port_containers": {
        "Imports Containers (TEUs)": "port_import_teu",
        "Export Containers (TEUs)": "port_export_teu",
        "Transshipment containers (TEUs)": "port_transship_teu",
        "Restowed Containers (TEUs)": "port_restow_teu",
        "Total Container Throughput (TEUs)": "port_total_teu",
    },
    "port_vessels": {
        "Container Vessel Traffic": "port_container_vessels",
        "General Cargo Vessel Traffic": "port_general_cargo_vessels",
        "Cruise Vessels Traffic": "port_cruise_vessels",
        "Total Vessel Visit": "port_total_vessels",
    },
    "airport_cargo": {
        "Import Cargo & Mail (In Tonnes)": "air_import_t",
        "ExportCargo & Mail (In Tonnes)": "air_export_t",
        "Transshipment Cargo & Mail (In Tonnes)": "air_transship_t",
    },
    "port_roro": {"roro_units_vehicles": "port_roro_vehicles"},
    "port_general_cargo": {"general_cargo_volume_ft": "port_general_cargo_ft"},
}
GENERAL_CARGO_TYPO_BELOW = 1_000   # every correctly typed month since 2019 is above 3,000 freight tonnes


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
def merge_repeats(raw: pd.DataFrame) -> pd.DataFrame:
    """Add together rows with the same year, month, product and country.

    Value, weight and quantity are summed. ``n_shipments`` records how many rows were combined,
    so merged rows can always be found again.
    """
    # "#N/A" in the unit column means no unit was recorded; treat it as missing.
    raw = raw.assign(unit=raw["unit"].replace({"#N/A": np.nan, "nan": np.nan}))
    units_per_key = raw.groupby(KEYS, sort=False)["unit"].nunique()
    if (units_per_key > 1).any():
        raise ValueError("some product-country-months mix units; quantities can't be added")
    merged = raw.groupby(KEYS, sort=False).agg(
        value_bd=("value_bd", "sum"),
        value_usd=("value_usd", "sum"),
        weight_kg=("weight_kg", "sum"),
        quantity=("quantity", "sum"),
        unit=("unit", "first"),
        n_shipments=("value_bd", "size"),
    )
    return merged.reset_index()


def latest_wording(table: pd.DataFrame, key: str, text: str) -> pd.Series:
    """For each code, the wording from the latest year it appears (most common wording if several).

    Sorting by code, then newest year first, then most rows first, puts the preferred wording on
    the first row of each code; drop_duplicates keeps exactly that row.
    """
    sort_cols = ["year", "n"] if "n" in table else ["year"]
    ordered = table.sort_values([key, *sort_cols], ascending=[True] + [False] * len(sort_cols))
    return ordered.drop_duplicates(key).set_index(key)[text].str.strip()


def add_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Parent codes for each 8-digit product, its HS section and its product group.

    Worked out once per distinct code (about 7,400) and then looked up, rather than
    slicing strings on every one of the 1.76M rows.
    """
    codes = pd.DataFrame({"hs8": pd.unique(df["hs8"].astype(object))})
    codes["hs6"], codes["hs4"], codes["hs2"] = codes["hs8"].str[:6], codes["hs8"].str[:4], codes["hs8"].str[:2]
    sections = codes["hs2"].map(chapter_to_section)
    codes["section_id"] = sections.str[0].astype("int8")
    codes["section_name"] = sections.str[1]
    codes["product_group"] = codes["hs4"].map(PRODUCT_GROUPS).fillna(DEFAULT_GROUP)
    lookup = codes.set_index("hs8")

    df = df.copy()
    keys = df["hs8"].astype(object)
    for col in ["hs6", "hs4", "hs2", "section_id", "section_name", "product_group"]:
        df[col] = keys.map(lookup[col]).to_numpy()
    df["section_id"] = df["section_id"].astype("int8")
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows that look wrong without removing them (see the report for how many there are)."""
    df = df.copy()
    df["flag_zero_weight"] = df["weight_kg"] <= 0
    df["flag_nonpositive_value"] = df["value_bd"] <= 0
    with np.errstate(divide="ignore", invalid="ignore"):     # zero weights give inf/nan here; they are flagged above
        bd_per_kg = df["value_bd"] / df["weight_kg"]
    df["flag_implausible_unit_value"] = (df["weight_kg"] > 0) & (bd_per_kg > IMPLAUSIBLE_BD_PER_KG)
    return df


def clean_year(year: int, name: str) -> dict:
    """Load and merge one year. Repeats never span two years, so year-by-year gives the same result."""
    df = load_imports(RAW_DIR / f"{name}.csv", year)
    # Grouping on categorical columns would create every combination of categories, so the
    # key columns are turned back into plain Python strings for the merge.
    for col in ("hs8", "commodity_en", "iso2", "country_en", "unit"):
        df[col] = df[col].astype(object)  # not astype(str): that builds a huge fixed-width array for long text
    # Everything needed later is collected now, so the raw rows can be freed after each year.
    out = {
        "rows": len(df),
        "totals": df.groupby(["year", "month"])[["value_bd", "weight_kg"]].sum(),
        "descriptions": df.value_counts(["hs8", "commodity_en"]).rename("n").reset_index().assign(year=year),
        "countries": df[["iso2", "country_en"]].drop_duplicates().assign(year=year),
        "merged": merge_repeats(df.drop(columns=["commodity_en", "country_en"])),
    }
    print(f"  {year}: {out['rows']:,} rows -> {len(out['merged']):,}")
    return out


def concat_categorical(frames: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    """Stack frames, giving text columns one shared set of categories so they stay compact."""
    for col in cols:
        categories = sorted(set().union(*(set(f[col].dropna()) for f in frames)))
        for f in frames:
            f[col] = pd.Categorical(f[col], categories=categories)
    return pd.concat(frames, ignore_index=True)


def build_imports() -> tuple[pd.DataFrame, dict]:
    """Clean every year, combine them, check the totals and add the classification and flags."""
    years = [clean_year(ds.year, ds.name) for ds in DATASETS if ds.kind == "imports"]
    clean = concat_categorical([y["merged"] for y in years], ["hs8", "iso2", "unit"])
    descriptions = pd.concat([y["descriptions"] for y in years])
    countries = pd.concat([y["countries"] for y in years])

    # Totals must survive cleaning exactly, month by month.
    before = pd.concat([y["totals"] for y in years])
    after = clean.groupby(["year", "month"])[["value_bd", "weight_kg"]].sum()
    if not np.allclose(before.values, after.loc[before.index].values, rtol=1e-12, atol=1e-6):
        raise AssertionError("monthly totals changed during cleaning")
    if clean.duplicated(KEYS).any():
        raise AssertionError("duplicate product-country-months remain")

    # One description per product code and per country code (step 4 of the module docstring).
    clean["commodity_en"] = clean["hs8"].astype(object).map(latest_wording(descriptions, "hs8", "commodity_en"))
    clean["country_en"] = clean["iso2"].astype(object).map(latest_wording(countries, "iso2", "country_en"))
    clean = add_flags(add_hierarchy(clean))
    clean["date"] = pd.to_datetime({"year": clean["year"], "month": clean["month"], "day": 1})

    columns = [
        "date", "year", "month",
        "hs8", "hs6", "hs4", "hs2", "section_id", "section_name", "product_group", "commodity_en",
        "iso2", "country_en",
        "value_bd", "value_usd", "weight_kg", "quantity", "unit", "n_shipments",
        "flag_zero_weight", "flag_nonpositive_value", "flag_implausible_unit_value",
    ]
    clean = clean[columns].sort_values(["date", "hs8", "iso2"], ignore_index=True)
    # Store repeated text as categories: the parquet file is several times smaller and loads faster.
    for col in ["hs8", "hs6", "hs4", "hs2", "section_name", "product_group", "commodity_en", "iso2", "country_en", "unit"]:
        clean[col] = clean[col].astype("category")

    stats = {"raw_rows_by_year": pd.Series({y["totals"].index[0][0]: y["rows"] for y in years})}
    return clean, stats


# ---------------------------------------------------------------------------
# Route totals
# ---------------------------------------------------------------------------
def build_route_monthly() -> pd.DataFrame:
    """Port and airport series side by side, one row per month."""
    parts = []
    for name, mapping in ROUTE_SERIES.items():
        df = load_route(RAW_DIR / f"{name}.csv", name)
        unknown = set(df["series"]) - set(mapping)
        if unknown:
            raise ValueError(f"{name}: unexpected series {sorted(unknown)}")
        # Exact repeats are harmless (port_vessels lists 'Cruise Vessels Traffic' Aug 2025 twice, both 0);
        # two *different* values for the same month and series would be a real problem, so stop on those.
        df = df.drop_duplicates(["date", "series", "value"])
        conflicts = df[df.duplicated(["date", "series"], keep=False)]
        if len(conflicts):
            raise ValueError(f"{name}: conflicting values for the same month:\n{conflicts}")
        # Long (date, series, value) -> wide: one column per series, one row per month.
        parts.append(df.assign(series=df["series"].map(mapping)).pivot(index="date", columns="series", values="value"))
    out = pd.concat(parts, axis=1).sort_index()
    out.columns.name = None
    return fix_general_cargo(out.reset_index())


def fix_general_cargo(route: pd.DataFrame) -> pd.DataFrame:
    """Repair two problems in the port's general cargo figures, and note every month changed.

    1. From February 2026, several months read like 56.186 or 170.881 freight tonnes, while every
       other month since 2019 is between 3,000 and 194,000 (June 2026 itself reads 112609). They are
       thousands typed with a decimal point (56.186 means 56,186), so they are multiplied by 1,000.
    2. A month recorded as 0 while general cargo ships called at the port (March 2026: 17 ships) is
       treated as missing rather than as nothing unloaded.

    Pro: the repaired months line up with the ship counts and with the months around them.
    Con: it is a judgement about someone else's typing, so the original value is kept in the note
    and no conclusion in the project rests on these figures alone.
    """
    col = "port_general_cargo_ft"
    route = route.copy()
    note = pd.Series("", index=route.index, dtype=object)
    typo = (route[col] > 0) & (route[col] < GENERAL_CARGO_TYPO_BELOW)
    note[typo] = "decimal point fixed: " + route.loc[typo, col].map(lambda v: f"{v:g}") + " read as thousands"
    route.loc[typo, col] = (route.loc[typo, col] * 1000).round()
    missing = (route[col] == 0) & (route["port_general_cargo_vessels"] > 0)
    note[missing] = "recorded as 0 although general cargo ships called; treated as missing"
    route.loc[missing, col] = np.nan
    route["port_general_cargo_note"] = note
    return route


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def code_churn(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Codes that vanish or appear between consecutive years, and the share of value they carry.

    Customs codes are revised (the HS 2022 revision came into force in 2022). If many codes
    change between years, a model working at that level would see series stop and start for
    reasons that have nothing to do with trade. This table shows which level is stable enough.
    """
    codes = df[level].astype(object)
    rows = []
    years = sorted(df["year"].unique())
    for y0, y1 in zip(years, years[1:]):
        in_a, in_b = df["year"] == y0, df["year"] == y1
        codes_a, codes_b = codes[in_a], codes[in_b]
        value_a, value_b = df.loc[in_a, "value_bd"], df.loc[in_b, "value_bd"]
        gone, new = set(codes_a) - set(codes_b), set(codes_b) - set(codes_a)
        rows.append({
            "years": f"{y0}->{y1}",
            "codes": codes_a.nunique(),
            "gone": len(gone),
            "gone_%_value": 100 * value_a[codes_a.isin(gone)].sum() / value_a.sum(),
            "new": len(new),
            "new_%_value": 100 * value_b[codes_b.isin(new)].sum() / value_b.sum(),
        })
    return pd.DataFrame(rows)


def series_density(df: pd.DataFrame, keys: list[str], months: int) -> dict:
    """How many series at a given level have (nearly) every month, and how much value they cover.

    A series that only has trade in a few months is hard to forecast, so this decides how finely
    the data can be split for the models (2021-2025 only, so the war doesn't affect the choice).
    """
    hist = df[df["year"] <= 2025]
    g = hist.groupby(keys, observed=True).agg(n=("date", "nunique"), v=("value_bd", "sum"))
    dense = g[g["n"] >= months - 6]
    return {"level": " x ".join(keys), "series": len(g), "dense": len(dense), "dense_%_value": 100 * dense["v"].sum() / g["v"].sum()}


def write_report(clean: pd.DataFrame, stats: dict, route: pd.DataFrame, calendar: pd.DataFrame) -> str:
    """The step 2 report: what was changed, why, and the checks that passed."""
    raw_rows = int(stats["raw_rows_by_year"].sum())
    by_year = pd.DataFrame({
        "Year": stats["raw_rows_by_year"].index.astype(str),
        "Raw rows": stats["raw_rows_by_year"].values,
        "Clean rows": clean.groupby("year").size().values,
        "Value (BD m)": (clean.groupby("year")["value_bd"].sum() / 1e6).values,
        "Weight (kt)": (clean.groupby("year")["weight_kg"].sum() / 1e6).values,
    })
    merged = clean[clean["n_shipments"] > 1]

    flags = []
    for col, label in [("flag_zero_weight", "Weight is zero"), ("flag_nonpositive_value", "Value is zero or negative"),
                       ("flag_implausible_unit_value", f"Value per kg above BD {IMPLAUSIBLE_BD_PER_KG:,}")]:
        rows = clean[clean[col]]
        share = round(100 * rows["value_bd"].sum() / clean["value_bd"].sum(), 3) + 0.0  # + 0.0 avoids printing "-0.000"
        flags.append({"Flag": label, "Rows": len(rows), "% of value": share})
    flag_value = sum(f["% of value"] for f in flags)
    missing_units = int(clean["unit"].isna().sum())

    # Share of value and weight by product group, on the two most recent normal years.
    hist = clean[clean["year"].between(2024, 2025)]
    groups = hist.groupby("product_group", observed=True)[["value_bd", "weight_kg"]].sum()
    groups = (100 * groups / groups.sum()).sort_values("weight_kg", ascending=False).reset_index()
    groups.columns = ["Product group", "% of value", "% of weight"]
    bulk = groups[groups["Product group"] != DEFAULT_GROUP]

    churn = {lvl: code_churn(clean, lvl) for lvl in ["hs8", "hs6", "hs4"]}
    for table in churn.values():
        table.columns = ["Years", "Codes", "Gone", "Gone (% of value)", "New", "New (% of value)"]
    first_pair = churn["hs8"].iloc[0]
    hs4_max = max(churn["hs4"]["Gone (% of value)"].max(), churn["hs4"]["New (% of value)"].max())
    density = pd.DataFrame([series_density(clean, keys, 60) for keys in
                            (["hs2"], ["hs4"], ["hs2", "iso2"], ["hs4", "iso2"], ["hs8", "iso2"])])
    density["level"] = density["level"].map({
        "hs2": "Chapter (2-digit)", "hs4": "Heading (4-digit)", "hs2 x iso2": "Chapter x country",
        "hs4 x iso2": "Heading x country", "hs8 x iso2": "8-digit product x country"})
    density.columns = ["Level", "Series", "Series with 54+ of 60 months", "% of value they cover"]
    hs2_country = density.iloc[2]

    last_import = clean["date"].max()
    route_cov = {c: f"{route.loc[route[c].notna(), 'date'].min():%b %Y} to {route.loc[route[c].notna(), 'date'].max():%b %Y}"
                 for c in ["port_import_teu", "port_total_vessels", "air_import_t", "port_roro_vehicles",
                           "port_general_cargo_ft"]}
    fixes = route[route["port_general_cargo_note"] != ""]
    fix_rows = pd.DataFrame({"Month": fixes["date"].dt.strftime("%b %Y"),
                             "Freight tonnes used": fixes["port_general_cargo_ft"].map(
                                 lambda v: "missing" if pd.isna(v) else f"{v:,.0f}"),
                             "What was done": fixes["port_general_cargo_note"].str.capitalize()})

    return f"""
# Step 2: Cleaning the data

The six yearly import files ({raw_rows:,} rows, January 2021 to {last_import:%B %Y}) were combined into
one table with {len(clean):,} rows: one row per month, 8-digit product and origin country. The aim was to
change as little as possible. Nothing was dropped, the monthly totals still match the official figures
exactly, and every change is recorded.

## Key points

- **Totals are unchanged.** Monthly value and weight were checked before and after cleaning; the script
  stops if they differ.
- **{len(merged)} product-country-months appeared more than once** with different values. They are
  separate shipments, so they were added together rather than treated as duplicates.
- **{sum(f["Rows"] for f in flags):,} odd rows are flagged, not removed.** They carry about
  {flag_value:.1f}% of value.
- **Bulk raw materials are about {bulk["% of weight"].sum():.0f}% of import weight but only
  {bulk["% of value"].sum():.0f}% of value**, so they are analysed separately from everyday goods.
- **The models work at chapter (2-digit) level or coarser.** Customs code revisions barely affect it, and
  its series are complete enough to forecast.
- **{len(fix_rows)} months of the port's general cargo figures were repaired** (a thousands separator typed
  as a decimal point, and one zero that should be missing).

## Rows and totals by year

{md_table(by_year, ",.0f")}

The 2026 figures cover January to {last_import:%B} only.

## Repeated rows

{len(merged)} product-country-months had more than one row ({int(merged['n_shipments'].sum())} rows in
total). The rows have different values and weights, so they look like separate shipments rather than
copies of the same record. Adding them together gives one row per product, country and month, which is
what every later step expects. The number of rows combined is kept in the column `n_shipments`.

## Rows that look wrong

Rows with zero weight, zero or negative value, or an impossible price per kg are flagged with a column
each and kept. Removing them would make the totals disagree with the published statistics, and each
analysis can still leave them out if it needs to. The downside is that a wrong row stays in the data, but
together they carry so little value that no result in the project changes.

{md_table(pd.DataFrame(flags), ".3f")}

A value above BD {IMPLAUSIBLE_BD_PER_KG:,} per kg is more than the price of gold, so it is almost
certainly a typing error in the weight. {missing_units} rows have no unit recorded (`#N/A` in the source).

## Bulk materials

A few raw materials dominate Bahrain's import weight. They arrive in whole shiploads for a handful of
plants (the aluminium smelter, steel and pelletising plants, construction), so one ship more or less moves
the monthly figure a lot. Analysing them with everyday goods would let a single ore carrier hide what
happened to food, machinery or clothing. They are kept as their own group in every later step.

{md_table(groups, ".1f")}

*Shares of 2024-2025 value and weight. Gold and jewellery are also separated later, in step 3, because
they follow the gold price.*

## Choosing the level of detail

Customs codes are revised every few years. The HS 2022 revision shows up as {first_pair['New']} new
8-digit codes in 2022 carrying {first_pair['New (% of value)']:.1f}% of that year's value, with many old
codes only disappearing in 2023. A model working with 8-digit codes would see series stop and start for
purely administrative reasons. At the 4-digit level, no year's changes affect more than {hs4_max:.2f}% of
value.

The other consideration is how complete each series is. A series (a product, optionally from one
country) that only has trade in a few months cannot be forecast sensibly. Splitting by chapter and
country gives {hs2_country['Series']:,} series, of which {hs2_country['Series with 54+ of 60 months']:,}
have trade in nearly every month and cover {hs2_country['% of value they cover']:.0f}% of value. Splitting
more finely quickly leaves most value in patchy series.

The models in step 4 therefore work with chapters (2-digit codes) by origin region, and 8-digit codes are
kept for looking up individual products. The cost is that the forecasts cannot see a shift between two
products in the same chapter; the benefit is stable, complete series.

{md_table(density, ".1f")}

*2021-2025. A series is counted as complete with trade in at least 54 of the 60 months.*

### Code changes between years, at each level

**8-digit**

{md_table(churn["hs8"])}

**6-digit**

{md_table(churn["hs6"])}

**4-digit**

{md_table(churn["hs4"])}

2025->2026 compares a full year with seven war-affected months, so more codes are "gone".

## Port general cargo: months repaired

In the port's general cargo file (freight tonnes per month), several 2026 months were typed with a decimal
point where a thousands separator belongs: 56.186 instead of 56,186. Every correctly typed month since
2019 is above 3,000, so values between 0 and {GENERAL_CARGO_TYPO_BELOW:,} were multiplied by 1,000. A month
recorded as 0 while general cargo ships called at the port is treated as missing, since ships did unload.
Each change is marked in the column `port_general_cargo_note`. Because these are repairs to someone else's
figures, no conclusion in later steps rests on general cargo alone.

{md_table(fix_rows) if len(fix_rows) else "No months needed repairing."}

## Tables produced

- `data/processed/imports_clean.parquet`: the cleaned imports, with product hierarchy, product group and flags.
- `data/processed/route_monthly.parquet`: port containers ({route_cov['port_import_teu']}), ship visits
  ({route_cov['port_total_vessels']}), air cargo ({route_cov['air_import_t']}), vehicles unloaded from car
  carriers ({route_cov['port_roro_vehicles']}) and general cargo ({route_cov['port_general_cargo_ft']}).
- `data/reference/calendar.csv`: {len(calendar)} months ({calendar['date'].min():%b %Y} to
  {calendar['date'].max():%b %Y}) with Ramadan days, Eid flags and COVID flags.
- `data/reference/hs_sections.csv`: chapter to section lookup.

{footer("src/build_dataset.py")}
"""


def main() -> int:
    start = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Imports")
    clean, stats = build_imports()
    clean.to_parquet(PROCESSED_DIR / "imports_clean.parquet", index=False)
    print(f"  wrote imports_clean.parquet: {len(clean):,} rows")

    print("Route totals")
    route = build_route_monthly()
    route.to_parquet(PROCESSED_DIR / "route_monthly.parquet", index=False)
    print(f"  wrote route_monthly.parquet: {len(route)} months")

    print("Reference tables")
    calendar = build_calendar()
    calendar.to_csv(REFERENCE_DIR / "calendar.csv", index=False)
    hs_sections_table().to_csv(REFERENCE_DIR / "hs_sections.csv", index=False)
    print("  wrote calendar.csv and hs_sections.csv")

    write_markdown(REPORT_PATH, write_report(clean, stats, route, calendar))
    print(f"  wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"\nDone in {time.time() - start:.0f}s. All totals preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
