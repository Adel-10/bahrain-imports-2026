"""Step 2: clean the raw downloads and build the tables the analysis uses.

    python src/build_dataset.py

Reads data/raw/ and writes:
  data/processed/imports_clean.parquet  one row per month x 8-digit product x origin country
  data/processed/route_monthly.parquet  port and airport totals, one row per month
  data/reference/calendar.csv           Ramadan, Eid and COVID features, one row per month
  data/reference/hs_sections.csv        HS chapter -> section lookup
  reports/step2_cleaning_report.md      what was changed, and the checks that passed

What "cleaning" means here (all decisions are based on profiling the real files):
  * the six yearly files are combined into one table with the same columns;
  * separate shipments of the same product from the same country in the same
    month are added together (12 rows in 6 pairs);
  * each 8-digit code gets its parents: 6-digit, 4-digit, chapter and section;
  * each product and country gets one consistent description;
  * bulk raw materials (iron ore, alumina, coke, construction minerals) are
    labelled, because they dominate import weight;
  * odd rows are flagged, never dropped, so totals still match the official figures.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from config import DATASETS, RAW_DIR, REFERENCE_DIR, ROOT
from loaders import load_imports, load_route
from reference import DEFAULT_GROUP, PRODUCT_GROUPS, build_calendar, chapter_to_section, hs_sections_table

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "step2_cleaning_report.md"

KEYS = ["year", "month", "hs8", "iso2"]
IMPLAUSIBLE_BD_PER_KG = 100_000  # above the price of gold (~30-50k BD/kg); almost certainly a data-entry error

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
}


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
def merge_repeats(raw: pd.DataFrame) -> pd.DataFrame:
    """Add together rows with the same year, month, product and country."""
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
    """For each code, the wording from the latest year it appears (most common wording if several)."""
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
    df = df.copy()
    df["flag_zero_weight"] = df["weight_kg"] <= 0
    df["flag_nonpositive_value"] = df["value_bd"] <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        bd_per_kg = df["value_bd"] / df["weight_kg"]
    df["flag_implausible_unit_value"] = (df["weight_kg"] > 0) & (bd_per_kg > IMPLAUSIBLE_BD_PER_KG)
    return df


def clean_year(year: int, name: str) -> dict:
    """Load and merge one year. Repeats never span two years, so year-by-year gives the same result."""
    df = load_imports(RAW_DIR / f"{name}.csv", year)
    for col in ("hs8", "commodity_en", "iso2", "country_en", "unit"):
        df[col] = df[col].astype(object)  # not astype(str): that builds a huge fixed-width array for long text
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
        parts.append(df.assign(series=df["series"].map(mapping)).pivot(index="date", columns="series", values="value"))
    out = pd.concat(parts, axis=1).sort_index()
    out.columns.name = None
    return out.reset_index()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def code_churn(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Codes that vanish or appear between consecutive years, and the share of value they carry."""
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
    hist = df[df["year"] <= 2025]
    g = hist.groupby(keys, observed=True).agg(n=("date", "nunique"), v=("value_bd", "sum"))
    dense = g[g["n"] >= months - 6]
    return {"level": " x ".join(keys), "series": len(g), "dense": len(dense), "dense_%_value": 100 * dense["v"].sum() / g["v"].sum()}


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    def fmt(x):
        if isinstance(x, float):
            return format(x, floatfmt)
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        return str(x)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_report(clean: pd.DataFrame, stats: dict, route: pd.DataFrame, calendar: pd.DataFrame) -> str:
    raw_rows = int(stats["raw_rows_by_year"].sum())
    by_year = pd.DataFrame({
        "year": stats["raw_rows_by_year"].index.astype(str),
        "raw rows": stats["raw_rows_by_year"].values,
        "clean rows": clean.groupby("year").size().values,
        "value (BD m)": (clean.groupby("year")["value_bd"].sum() / 1e6).values,
        "weight (kt)": (clean.groupby("year")["weight_kg"].sum() / 1e6).values,
    })
    merged = clean[clean["n_shipments"] > 1]

    flags = []
    for col, label in [("flag_zero_weight", "weight is zero"), ("flag_nonpositive_value", "value is zero or negative"),
                       ("flag_implausible_unit_value", f"value per kg above {IMPLAUSIBLE_BD_PER_KG:,} BD")]:
        rows = clean[clean[col]]
        share = round(100 * rows["value_bd"].sum() / clean["value_bd"].sum(), 3) + 0.0  # + 0.0 avoids printing "-0.000"
        flags.append({"flag": label, "rows": len(rows), "% of value": share})
    missing_units = int(clean["unit"].isna().sum())

    hist = clean[clean["year"].between(2024, 2025)]
    groups = hist.groupby("product_group", observed=True)[["value_bd", "weight_kg"]].sum()
    groups = (100 * groups / groups.sum()).sort_values("weight_kg", ascending=False).reset_index()
    groups.columns = ["product group", "% of value", "% of weight"]

    churn = {lvl: code_churn(clean, lvl) for lvl in ["hs8", "hs6", "hs4"]}
    first_pair = churn["hs8"].iloc[0]
    hs4_max = max(churn["hs4"]["gone_%_value"].max(), churn["hs4"]["new_%_value"].max())
    density = pd.DataFrame([series_density(clean, keys, 60) for keys in
                            (["hs2"], ["hs4"], ["hs2", "iso2"], ["hs4", "iso2"], ["hs8", "iso2"])])
    density.columns = ["level", "series", "series with >= 54 of 60 months", "% of value they cover"]

    last_import = clean["date"].max()
    route_cov = {c: f"{route.loc[route[c].notna(), 'date'].min():%b %Y} to {route.loc[route[c].notna(), 'date'].max():%b %Y}"
                 for c in ["port_import_teu", "port_total_vessels", "air_import_t"]}

    lines = [
        "# Step 2: cleaning report",
        "",
        f"Generated by `src/build_dataset.py`. Raw rows: {raw_rows:,}. Clean rows: {len(clean):,}. "
        f"Imports cover Jan 2021 to {last_import:%b %Y}.",
        "",
        "## Rows and totals by year",
        "Totals are identical before and after cleaning (checked month by month in the script).",
        "",
        md_table(by_year, ",.0f"),
        "",
        "## Repeated shipments",
        f"{len(merged)} product-country-months had more than one row ({int(merged['n_shipments'].sum())} rows in total). "
        "They are separate shipments with different values, so they were added together.",
        "",
        "## Flagged rows (kept, not dropped)",
        md_table(pd.DataFrame(flags), ".3f"),
        "",
        f"Rows with no unit recorded (`#N/A` in the source): {missing_units}.",
        "",
        "## What dominates import weight (2024-2025)",
        "A few bulk raw materials make up most of the weight, so weight analyses separate them from general goods.",
        "",
        md_table(groups, ".1f"),
        "",
        "## Customs-code changes between years",
        f"The HS 2022 revision shows up as {first_pair['new']} new 8-digit codes in 2022 carrying "
        f"{first_pair['new_%_value']:.1f}% of that year's value, with many old codes only disappearing in 2023. "
        f"At the 4-digit level no year's changes affect more than {hs4_max:.2f}% of value, so models work at "
        "4-digit level or coarser and 8-digit codes are kept for drill-downs. (2025->2026 compares a full year "
        "with seven crisis-hit months, so more codes are 'gone'.)",
        "",
        "**8-digit**", "", md_table(churn["hs8"]), "",
        "**6-digit**", "", md_table(churn["hs6"]), "",
        "**4-digit**", "", md_table(churn["hs4"]), "",
        "## How complete the series are (2021-2025)",
        "A series is a product (at the given level), optionally split by origin country.",
        "",
        md_table(density, ".1f"),
        "",
        "## Other tables built",
        f"- `route_monthly.parquet`: port containers {route_cov['port_import_teu']}, ship visits "
        f"{route_cov['port_total_vessels']}, air cargo {route_cov['air_import_t']}.",
        f"- `calendar.csv`: {len(calendar)} months ({calendar['date'].min():%b %Y} to {calendar['date'].max():%b %Y}) "
        "with Ramadan days, Eid flags and COVID flags.",
        "- `hs_sections.csv`: chapter to section lookup.",
        "",
    ]
    return "\n".join(lines)


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

    REPORT_PATH.write_text(write_report(clean, stats, route, calendar), encoding="utf-8")
    print(f"  wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"\nDone in {time.time() - start:.0f}s. All totals preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
