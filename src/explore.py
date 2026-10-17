"""Step 3: explore the cleaned data before building any models.

    python src/explore.py

Reads the step 2 tables and writes:
  * reports/figures/step3_*.png    six charts
  * reports/step3_exploration.md   what they show, and the decisions taken for step 4

Purpose
-------
Before choosing a model, it pays to understand the data: which imports behave alike, how much
they move in a normal year, and how finely they can be split before the noise drowns the signal.
This script answers those questions and takes a first look at 2026.

The yardstick used here
-----------------------
A 2026 month is compared with the **average month of 2025** for the same thing (a region, a
section, a product group). Two measures say whether a change is large:

* the **normal miss**: forecast every month of a year as the previous year's average month and
  measure the total absolute miss as a share of the total forecast (``gap``);
* the **normal range**: the distance from the previous year's average month that 9 months out of
  10 stayed within in 2023-2025 (``normal_swing``).

Pros: simple, transparent and easy to check by hand. Imports turn out to have little
seasonality, so the average month is a steadier yardstick than the same month a year earlier
(section 2 of the report shows this). Cons: it ignores growth and the calendar. Step 4 replaces
it with proper forecasts for each month, and step 5 measures the 2026 shortfall against those.

Code layout
-----------
1. Data helpers: tables of series x month, and the yardstick calculations.
2. Figures: one function per chart (shared styling comes from charts.py).
3. Analyses: each returns a table used by the charts and the report.
4. Report: the markdown template, filled in with the numbers computed above.

Every number in the report is computed here, so re-running after the portal adds a month
refreshes it. The written interpretation was checked against data up to July 2026.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from charts import (AQUA, BASELINE, BLUE, FIG_DIR, GRID, INK, INK2, MUTED, ORANGE, SHADE, SURFACE, WAR_START, YELLOW,
                    add_axes, dot_handle, end_dot, end_labels, legend, line_handle, month_axis, new_figure, rounded_bar,
                    save, source_note, tidy)
from config import REFERENCE_DIR, ROOT
from reference import (BULK, CORE, OTHER_REGION, PRE_WAR_2026, PRECIOUS, WAR_PERIODS_2026, add_analysis_group,
                       add_region, period_event, war_periods_2026)
from reporting import footer, md_table, signed, write_markdown

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "step3_exploration.md"

# 2026 periods, by month (from data/reference/events.csv via reference.py). The war began on 28 February.
# main() extends PERIODS when the portal has published months after July.
PRE_WAR = PRE_WAR_2026["Jan-Feb"]
PERIODS = dict(WAR_PERIODS_2026)  # label -> months of 2026
POST_WAR = [m for months in PERIODS.values() for m in months]
COVID_END = pd.Timestamp("2022-03-28")        # COVID restrictions lifted (shaded on the first chart)

MIN_SECTION_SHARE = 0.5                       # % of 2025 core value; smaller sections are left out of the chart
SKIP_SECTIONS = {"National special codes"}    # confidential and special codes, not a supply chain
AIRCRAFT_HS4 = {"8409", "8411", "8802", "8803"}   # engines, engine parts, aircraft and aircraft parts

# Short names for 4-digit codes that come up in the report (the data only names 8-digit codes).
HS4_NAMES = {
    "0207": "Poultry meat", "1001": "Wheat", "1701": "Sugar", "2402": "Cigarettes and cigars",
    "3002": "Vaccines and blood products", "3004": "Medicines", "7106": "Silver", "7108": "Gold",
    "7113": "Jewellery", "7207": "Semi-finished steel", "7403": "Refined copper", "7409": "Copper plates and sheets",
    "7601": "Unwrought aluminium", "8409": "Aircraft engine parts", "8411": "Jet engines and gas turbines",
    "8417": "Industrial furnaces", "8471": "Computers", "8504": "Transformers and power supplies",
    "8517": "Phones and network equipment", "8535": "High-voltage switchgear", "8544": "Insulated wire and cable",
    "8545": "Carbon electrodes", "8703": "Cars", "8708": "Car parts", "8802": "Aircraft",
    "9898": "Confidential goods (national code)",
}

PERIOD_COLOURS = dict(zip(PERIODS, [BLUE, ORANGE, AQUA, YELLOW]))   # one colour per 2026 period on the dot plots


def extend_periods(last_month: pd.Timestamp) -> None:
    """Bring PERIODS (and the lists built from it) up to the latest month the portal has published.

    The lists are changed in place, so every function that already refers to them sees the update.
    """
    if last_month.year != 2026:
        return
    PERIODS.clear()
    PERIODS.update(war_periods_2026(last_month.month))
    POST_WAR[:] = [m for months in PERIODS.values() for m in months]
    PERIOD_COLOURS.clear()
    PERIOD_COLOURS.update(zip(PERIODS, [BLUE, ORANGE, AQUA, YELLOW]))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_imports() -> pd.DataFrame:
    """The cleaned imports with only the columns this step needs, plus origin region and analysis group."""
    cols = ["date", "year", "month", "section_id", "section_name", "product_group", "hs2", "hs4",
            "iso2", "country_en", "value_bd", "weight_kg"]
    df = pd.read_parquet(PROCESSED_DIR / "imports_clean.parquet", columns=cols)
    add_region(df)
    add_analysis_group(df)
    return df


def wide(df: pd.DataFrame, keys: list[str], col: str = "value_bd") -> pd.DataFrame:
    """Series x month table: one row per series, one column per month, missing months = 0.

    `keys` defines a series (for example ["region"] gives one row per origin region). A month with
    no trade really means nothing was imported, so it is filled with 0 rather than left missing.
    With no keys the result is a single "total" row.
    """
    if not keys:
        return df.groupby("date")[col].sum().to_frame("total").T
    return df.groupby([*keys, "date"], observed=True)[col].sum().unstack("date", fill_value=0.0)


def months_of(w: pd.DataFrame, year: int, months: list[int] | None = None) -> pd.DataFrame:
    """The columns of a wide table for one year (optionally only some months of it)."""
    mask = w.columns.year == year
    if months is not None:
        mask &= w.columns.month.isin(months)
    return w.loc[:, mask]


def change_vs_2025(w: pd.DataFrame, months: list[int]) -> pd.Series:
    """% change of the average 2026 month in `months` vs the 2025 monthly average, per row.

    This is the yardstick used throughout step 3: -53 means the period's average month was 53%
    below an average month of 2025.
    """
    base = months_of(w, 2025).mean(axis=1)
    return (months_of(w, 2026, months).mean(axis=1) / base - 1) * 100


def normal_swing(w: pd.DataFrame, years=(2023, 2024, 2025), q: float = 0.9) -> pd.Series:
    """Per row: 9 months out of 10 in `years` were within +/- this % of the previous year's average.

    For each normal year, every month is compared with the previous year's average month. The
    absolute deviations of all those months are pooled and the 90th percentile is taken. A 2026
    change inside this range could be ordinary noise; one outside it is unusual. Three years give
    36 months per series, enough for a rough 90% range but not a precise one.
    """
    swings = [months_of(w, y).div(months_of(w, y - 1).mean(axis=1), axis=0) - 1 for y in years]
    return pd.concat(swings, axis=1).abs().quantile(q, axis=1) * 100


def gap(w: pd.DataFrame, year: int, months: list[int] | None = None, forecast: str = "average") -> float:
    """Average gap between actual and a simple forecast, as % of the forecast (summed over all series).

    forecast="average": every month of `year` is forecast as the previous year's monthly average.
    forecast="same month": every month is forecast as the same month a year earlier.
    """
    # Summing the absolute misses over every series and month, then dividing by the total forecast,
    # weights each series by its size: a 10% miss on machinery counts for more than one on paper.
    actual = months_of(w, year, months)
    if forecast == "average":
        fc = np.repeat(months_of(w, year - 1).mean(axis=1).to_numpy()[:, None], actual.shape[1], axis=1)
    else:
        fc = months_of(w, year - 1, list(actual.columns.month)).to_numpy()
    return float(np.abs(actual.to_numpy() - fc).sum() / fc.sum() * 100)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_monthly(df: pd.DataFrame) -> None:
    """Figure 1: monthly value and weight by analysis group, 2021 to latest month.

    Two panels share the time axis: value on top (all three groups) and weight below (precious
    metals weigh almost nothing, so they are left out). The COVID months and the war are shaded.
    """
    value = df.groupby(["date", "group"], observed=True)["value_bd"].sum().unstack() / 1e6
    weight = df.groupby(["date", "group"], observed=True)["weight_kg"].sum().unstack() / 1e6
    colours = {CORE: BLUE, BULK: ORANGE, PRECIOUS: AQUA}
    latest = value.index.max()

    fig = new_figure(10, 8.05, "Imports fell sharply after the war began on 28 February 2026",
                     f"Monthly non-oil imports, January 2021 to {latest:%B %Y}, split into three groups that behave differently.")
    legend(fig, [line_handle(colours[g]) for g in (CORE, BULK, PRECIOUS)], [CORE, BULK, PRECIOUS], top=0.98, ncol=3)
    panels = [(value, [CORE, BULK, PRECIOUS], "Value (BD million)", 1.6, 4.15),
              (weight, [BULK, CORE], "Weight (thousand tonnes)", 4.95, 0.8)]
    for data, groups, label, top, bottom in panels:       # top/bottom: the panel's margins in inches
        ax = add_axes(fig, left=0.75, right=1.55, top=top, bottom=bottom)
        tidy(ax)
        x_lo, x_hi = data.index.min() - pd.Timedelta(days=15), data.index.max() + pd.Timedelta(days=15)
        ax.axvspan(x_lo, COVID_END, color=SHADE, lw=0, zorder=0)
        ax.axvspan(WAR_START, x_hi, color=SHADE, lw=0, zorder=0)
        for g in groups:
            ax.plot(data.index, data[g], color=colours[g], zorder=3)
            end_dot(ax, data.index[-1], data[g].iloc[-1], colours[g])
        ax.set_ylim(0, data[groups].max().max() * 1.12)
        ax.set_xlim(x_lo, x_hi)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        month_axis(ax)
        ax.set_title(label, loc="left", fontsize=10, color=INK2, pad=8)
        ymax = ax.get_ylim()[1]
        ax.text(x_lo + pd.Timedelta(days=20), ymax * 0.97, "COVID rules", fontsize=8.5, color=MUTED, va="top")
        ax.text(WAR_START + pd.Timedelta(days=10), ymax * 0.97, "War", fontsize=8.5, color=MUTED, va="top")
        end_labels(ax, [(data.index[-1], data[g].iloc[-1], f"{g}  {data[g].iloc[-1]:,.0f}") for g in groups])
    source_note(fig, "Precious metals weigh almost nothing, so they are left out of the weight panel. "
                     "Source: Bahrain Open Data Portal (iGA). Analysis: own.")
    save(fig, "step3_1_monthly_imports.png")


def fig_routes(route_idx: pd.DataFrame) -> None:
    """Figure 2: goods arriving vs port containers, ship visits and air cargo, as % of the 2025 average.

    Putting every series on the same "% of its own 2025 average" scale lets very different units
    (BD, tonnes, containers, ship visits) share one chart. If goods kept arriving while containers
    and air cargo did not, they must have come another way.
    """
    series = [
        ("core_weight", "Core goods arriving (weight)", "Core goods", BLUE),
        ("port_import_teu", "Containers unloaded at the port", "Containers", ORANGE),
        ("port_total_vessels", "Ships calling at the port", "Ships", AQUA),
        ("air_import_t", "Air cargo unloaded", "Air cargo", YELLOW),
    ]
    data = route_idx.loc["2025-07-01":]
    fig = new_figure(10, 5.8, "Sea containers and air cargo fell much further than the goods arriving",
                     "Each month as % of that series' 2025 monthly average. Core goods = all imports except bulk materials "
                     "and precious metals.\nWhen the blue line is far above the others, more goods than usual were arriving "
                     "another way, most likely by road.")
    legend(fig, [line_handle(c) for *_, c in series], [s[1] for s in series], top=1.25, ncol=4)
    ax = add_axes(fig, left=0.75, right=1.9, top=1.75, bottom=0.8)
    tidy(ax)
    x_hi = data.index.max() + pd.Timedelta(days=12)
    ax.axvspan(WAR_START, x_hi, color=SHADE, lw=0, zorder=0)
    ax.axhline(100, color=BASELINE, lw=1, zorder=1)
    items = []
    for col, _, short, colour in series:
        s = data[col].dropna()
        ax.plot(s.index, s, color=colour, zorder=3)
        end_dot(ax, s.index[-1], s.iloc[-1], colour)
        items.append((s.index[-1], s.iloc[-1], f"{short} {s.iloc[-1]:.0f}%  ({s.index[-1]:%b})"))
    ax.set_ylim(0, 150)
    ax.set_xlim(data.index.min() - pd.Timedelta(days=12), x_hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    month_axis(ax, months=range(1, 13), first=data.index.min())
    ax.text(WAR_START + pd.Timedelta(days=6), 146, "War", fontsize=8.5, color=MUTED, va="top")
    end_labels(ax, items, x_text=data.index.max() + pd.Timedelta(days=22), gap_pt=14)
    source_note(fig, "Port: Khalifa Bin Salman Port. Air: Bahrain International Airport. "
                     "Source: Bahrain Open Data Portal (iGA). Analysis: own.")
    save(fig, "step3_2_routes.png")


def dot_plot(table: pd.DataFrame, band: pd.Series, title: str, subtitle: str, name: str, xlim: tuple) -> None:
    """Rows = series (top to bottom in table order), one dot per 2026 period, grey band = normal swing.

    Used for figures 3 (origin regions) and 4 (sections). The band makes it easy to see which
    changes are bigger than that series' normal month-to-month movement. Dots beyond the axis
    limits are drawn at the edge so they stay visible.
    """
    n = len(table)
    fig = new_figure(10, 1.75 + 0.37 * n + 0.75, title, subtitle)
    legend(fig, [dot_handle(c) for c in PERIOD_COLOURS.values()] + [Rectangle((0, 0), 1, 1, color=GRID, lw=0)],
           [f"{p} 2026" for p in PERIODS] + ["Normal range (2023-2025)"], top=1.25, ncol=4)
    ax = add_axes(fig, left=3.0, right=0.45, top=1.7, bottom=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.5, -0.5)
    tidy(ax, grid_axis="x")
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    for i, label in enumerate(table.index):
        swing = min(band[label], -xlim[0], xlim[1])
        ax.add_patch(Rectangle((-swing, i - 0.17), 2 * swing, 0.34, color=GRID, lw=0, zorder=1))
        for period, colour in PERIOD_COLOURS.items():
            x = table.loc[label, period]
            if pd.notna(x):
                ax.plot([np.clip(x, *xlim)], [i], "o", ms=8, color=colour, mec=SURFACE, mew=2, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(table.index, color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
    source_note(fig)
    save(fig, name)


def fig_patterns(core: pd.DataFrame) -> None:
    """Figure 5: month-of-year pattern by year (index) and yearly totals.

    Left: each month divided by its own year's average, so years of different size can be
    compared on one scale; a strong seasonal pattern would show as lines moving together.
    Right: yearly totals, to show the trend.
    """
    monthly = core.groupby("date")["value_bd"].sum()
    hist = monthly[monthly.index.year <= 2025]
    idx = hist / hist.groupby(hist.index.year).transform("mean") * 100
    table = idx.to_frame("idx").assign(year=idx.index.year, month=idx.index.month).pivot(
        index="month", columns="year", values="idx")
    avg = table[[y for y in table.columns if y >= 2022]].mean(axis=1)
    yearly = hist.groupby(hist.index.year).sum() / 1e6

    fig = new_figure(10, 5.0, "Core imports have only a weak seasonal pattern; 2021 was smaller but not unusual in shape",
                     "Left: each month as % of that year's monthly average. Right: yearly total.")
    legend(fig, [line_handle(BLUE), line_handle(ORANGE), line_handle(BASELINE, 1)],
           ["Average 2022-2025", "2021 (COVID rules)", "2022, 2023, 2024 and 2025"], top=0.98, ncol=3)
    ax = add_axes(fig, left=0.75, right=4.0, top=1.5, bottom=0.6)
    tidy(ax)
    ax.axhline(100, color=BASELINE, lw=1, zorder=1)
    for y in table.columns:
        if y >= 2022:
            ax.plot(table.index, table[y], color=BASELINE, lw=1, zorder=2)
    ax.plot(table.index, table[2021], color=ORANGE, zorder=3)
    ax.plot(table.index, avg, color=BLUE, zorder=4)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "", "Mar", "", "May", "", "Jul", "", "Sep", "", "Nov", ""])
    ax.set_xlim(0.7, 12.3)
    ax.set_ylim(80, 125)

    ax2 = add_axes(fig, left=6.75, right=0.3, top=1.5, bottom=0.6)
    tidy(ax2)
    ax2.set_xlim(2020.4, 2025.6)
    ax2.set_ylim(0, yearly.max() * 1.18)
    for year, v in yearly.items():
        rounded_bar(ax2, 0, v, year, 0.5, BLUE, horizontal=False)
        ax2.text(year, v + yearly.max() * 0.03, f"{v:,.0f}", ha="center", fontsize=8.5, color=INK2)
    ax2.set_xticks(yearly.index)
    ax2.set_xticklabels([str(y) for y in yearly.index])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.set_title("BD million per year", loc="left", fontsize=9.5, color=INK2, pad=8)
    source_note(fig)
    save(fig, "step3_5_patterns.png")


def fig_levels(levels: pd.DataFrame, chosen: str) -> None:
    """Figure 6: normal forecast miss vs the 2026 gap, by level of detail.

    The war is easy to see where the 2026 gap (dot) is far beyond the normal miss (bar), and
    hard to see where the two are close. This is the main input to choosing the model level.
    """
    n = len(levels)
    fig = new_figure(10, 1.75 + 0.42 * n + 0.8, "The more detail, the bigger the normal forecast miss",
                     "Bars: average monthly gap in 2024-2025 between actual imports and a simple forecast "
                     "(last year's monthly average).\nDots: the same gap in March-July 2026. Both as % of the forecast; "
                     "core goods only.")
    legend(fig, [Rectangle((0, 0), 1, 1, color=BLUE, lw=0), dot_handle(ORANGE)],
           ["Normal miss (2024-2025)", "Gap in Mar-Jul 2026"], top=1.25, ncol=2)
    ax = add_axes(fig, left=2.6, right=0.5, top=1.7, bottom=0.8)
    xmax = max(levels["normal miss %"].max(), levels["2026 gap %"].max()) * 1.12
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.5, -0.5)
    tidy(ax, grid_axis="x")
    for i, (level, normal, crisis) in enumerate(zip(levels["level"], levels["normal miss %"], levels["2026 gap %"])):
        rounded_bar(ax, 0, normal, i, 0.38, BLUE)
        ax.plot([crisis], [i], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
        if level == chosen:
            ax.annotate("model level for step 4", (crisis, i), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{lvl}  ({s:,} series)" for lvl, s in zip(levels["level"], levels["series"])], color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    source_note(fig)
    save(fig, "step3_6_levels.png")


# ---------------------------------------------------------------------------
# Analyses (each returns tables used by the figures and the report)
# ---------------------------------------------------------------------------
def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Size, normal miss and 2026 changes for core goods, bulk materials and precious metals."""
    w, ww = wide(df, ["group"]), wide(df, ["group"], "weight_kg")
    v25, w25 = months_of(w, 2025).sum(axis=1), months_of(ww, 2025).sum(axis=1)
    out = pd.DataFrame({
        "2025 value (BD m)": v25 / 1e6,
        "% of 2025 value": v25 / v25.sum() * 100,
        "% of 2025 weight": w25 / w25.sum() * 100,
        "normal miss %": [np.mean([gap(w.loc[[g]], y) for y in (2024, 2025)]) for g in w.index],
        "Jan-Feb": change_vs_2025(w, PRE_WAR),
        **{p: change_vs_2025(w, m) for p, m in PERIODS.items()},
    })
    out.index = out.index.astype(str)
    return out


def route_index(df: pd.DataFrame, route: pd.DataFrame) -> pd.DataFrame:
    """Core goods (value and weight) and the route series, each as % of its own 2025 monthly average."""
    core = df[df["group"] == CORE].groupby("date")[["value_bd", "weight_kg"]].sum()
    core.columns = ["core_value", "core_weight"]
    data = route[["port_import_teu", "port_total_vessels", "air_import_t"]].join(core, how="outer")
    return data / data[data.index.year == 2025].mean() * 100


def bulk_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bulk materials by weight: changes by material, and the main origins before and during the war."""
    bulk = df[df["group"] == BULK].copy()
    bulk["product_group"] = bulk["product_group"].astype(str)
    bulk["country"] = bulk["country_en"].astype(str).str.title()
    w = wide(bulk, ["product_group"], "weight_kg")
    changes = pd.DataFrame({"kt per month (2025)": months_of(w, 2025).mean(axis=1) / 1e6,
                            "Jan-Feb": change_vs_2025(w, PRE_WAR),
                            **{p: change_vs_2025(w, m) for p, m in PERIODS.items()}})
    changes = changes.sort_values("kt per month (2025)", ascending=False)

    wo = wide(bulk, ["product_group", "country"], "weight_kg")
    origins = pd.DataFrame({"kt per month, 2025": months_of(wo, 2025).mean(axis=1) / 1e6,
                            "kt per month, Mar-Jul 2026": months_of(wo, 2026, POST_WAR).mean(axis=1) / 1e6})
    # Keep the origins that mattered: at least 10% of a material's weight in either period, and at least 1 kt a month.
    group_total = origins.groupby(level=0).transform("sum")
    keep = ((origins / group_total) >= 0.1).any(axis=1) & (origins.max(axis=1) >= 1)
    origins = origins[keep].reset_index()
    origins["order"] = origins["product_group"].map({g: i for i, g in enumerate(changes.index)})
    origins = origins.sort_values(["order", "kt per month, 2025"], ascending=[True, False]).drop(columns="order")
    return changes, origins.rename(columns={"product_group": "material", "country": "origin"})


def change_table(core: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.Series]:
    """Share of 2025 value and change in each 2026 period, by `key` (region or section), with the normal range."""
    w = wide(core, [key])
    w.index = w.index.astype(str)
    share = months_of(w, 2025).sum(axis=1) / months_of(w, 2025).to_numpy().sum() * 100
    table = pd.DataFrame({"% of 2025": share, "Jan-Feb": change_vs_2025(w, PRE_WAR),
                          **{p: change_vs_2025(w, m) for p, m in PERIODS.items()}})
    order = table.sort_values("% of 2025", ascending=False).index
    return table.loc[order], normal_swing(w).loc[order]


# Ways of splitting core goods into series, ordered from least to most detailed. "region" is origin region,
# "section_id" the HS section, "hs2"/"hs4" the 2- and 4-digit customs codes, "iso2" the origin country.
LEVELS = {
    "National total": [], "Origin region": ["region"], "Section": ["section_id"], "HS2": ["hs2"],
    "Section x region": ["section_id", "region"], "HS2 x region": ["hs2", "region"],
    "HS4": ["hs4"], "HS4 x region": ["hs4", "region"], "HS4 x country": ["hs4", "iso2"],
}
CHOSEN_LEVEL = "HS2 x region"   # the level the models in step 4 forecast (see section 6 of the report)


def level_table(core: pd.DataFrame) -> pd.DataFrame:
    """For each level of detail: how many series, how complete they are, the normal miss and the 2026 gap.

    "Regular" series have trade in at least 54 of the 60 months of 2021-2025. The normal miss is
    averaged over 2024 and 2025; the 2026 gap is the same calculation for the war months.
    """
    rows = []
    for name, keys in LEVELS.items():
        w = wide(core, keys)
        hist = w.loc[:, w.columns.year <= 2025]
        regular = (hist > 0).sum(axis=1) >= hist.shape[1] - 6          # at least 54 of 60 months
        rows.append({
            "level": name, "series": len(w), "regular series": int(regular.sum()),
            "% of value in regular series": hist[regular].to_numpy().sum() / hist.to_numpy().sum() * 100,
            "normal miss %": np.mean([gap(w, y) for y in (2024, 2025)]),
            "2026 gap %": gap(w, 2026, POST_WAR),
            "same-month forecast miss %": np.mean([gap(w, y, forecast="same month") for y in (2024, 2025)]),
        })
    return pd.DataFrame(rows)


def seasonality(core: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    """How much of the month-to-month movement is a calendar pattern, and is there a Ramadan effect?

    Seasonal share: each month is divided by its year's average (2022-2025), then the share of
    the remaining variation explained by the month-of-year average is computed, like an R-squared.

    Ramadan: a small least-squares regression of that monthly index on the fraction of the month
    that is Ramadan, and on the fraction of the *next* month that is Ramadan (goods are often
    imported ahead of it). Done for all core goods and for food (sections 1-4), where an effect
    would be most likely. Four years of data make this a rough check, not a precise estimate.
    """
    out = {}
    ramadan = calendar.set_index("date")["ramadan_days"]
    for name, part in (("core", core), ("food", core[core["section_id"] <= 4])):
        monthly = part.groupby("date")["value_bd"].sum()
        monthly = monthly[(monthly.index.year >= 2022) & (monthly.index.year <= 2025)]
        idx = monthly / monthly.groupby(monthly.index.year).transform("mean")
        month_mean = idx.groupby(idx.index.month).transform("mean")
        out[f"{name}_explained"] = (1 - ((idx - month_mean) ** 2).sum() / ((idx - 1) ** 2).sum()) * 100
        # Columns: intercept, share of this month in Ramadan, share of next month in Ramadan.
        X = np.column_stack([np.ones(len(idx)), ramadan.reindex(idx.index) / 30, ramadan.shift(-1).reindex(idx.index) / 30])
        coef = np.linalg.lstsq(X, idx.to_numpy(), rcond=None)[0]
        out[f"{name}_ramadan_this"], out[f"{name}_ramadan_next"] = coef[1] * 100, coef[2] * 100
    monthly = core.groupby("date")["value_bd"].sum()
    hist = monthly[monthly.index.year <= 2025]
    out["yearly"] = hist.groupby(hist.index.year).sum()
    out["growth"] = out["yearly"].pct_change() * 100
    idx = hist / hist.groupby(hist.index.year).transform("mean") * 100
    later = idx.index.year >= 2022
    out["dec_index"] = idx[later & (idx.index.month == 12)].mean()
    out["may_jun_2021"] = idx[(idx.index.year == 2021) & idx.index.month.isin([5, 6])].mean()
    out["may_jun_other"] = idx[later & idx.index.month.isin([5, 6])].mean()
    return out


def jumpy_products(df: pd.DataFrame) -> pd.DataFrame:
    """Which 4-digit products cause most of the simple forecast's misses (general goods, 2024-2025)?

    A product whose share of the misses is much bigger than its share of value is "jumpy": large,
    occasional purchases such as aircraft engines, or price-driven goods such as gold.
    """
    w = wide(df[df["group"] != BULK], ["hs4"])
    w.index = w.index.astype(str)
    miss = sum(months_of(w, y).sub(months_of(w, y - 1).mean(axis=1), axis=0).abs().sum(axis=1) for y in (2024, 2025))
    value = months_of(w, 2024).sum(axis=1) + months_of(w, 2025).sum(axis=1)
    out = pd.DataFrame({"% of misses": miss / miss.sum() * 100, "% of value": value / value.sum() * 100})
    out = out.sort_values("% of misses", ascending=False).head(8)
    out.insert(0, "product", [HS4_NAMES.get(code, code) for code in out.index])
    return out


def biggest_increases(core: pd.DataFrame, region: str, months: list[int], n: int = 3) -> pd.Series:
    """Biggest 4-digit increases for one region: 2026 months vs the same months of 2025 (BD m)."""
    w = wide(core[core["region"] == region], ["hs4"])
    w.index = w.index.astype(str)
    d = (months_of(w, 2026, months).sum(axis=1) - months_of(w, 2025, months).sum(axis=1)) / 1e6
    return d.sort_values(ascending=False).head(n)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def vs_normal(x: float) -> str:
    """-53.2 -> '53% below', 7.1 -> '7% above'."""
    r = int(round(x))
    return "about the same as" if r == 0 else f"{abs(r)}% {'above' if r > 0 else 'below'}"


def names(series: pd.Series) -> str:
    """4-digit codes with BD changes as a phrase: 'copper plates and sheets (+BD 4.6m) and ...'."""
    parts = [f"{HS4_NAMES.get(code, 'HS ' + code).lower()} ({'+' if v >= 0 else '-'}BD {abs(v):.1f}m)" for code, v in series.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def table_md(df: pd.DataFrame, signed_cols=(), int_cols=(), one_dp_cols=()) -> str:
    """A markdown table with per-column number formats (signed whole numbers, whole numbers, one decimal)."""
    out = df.copy()
    for c in signed_cols:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else signed(v))
    for c in int_cols:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    for c in one_dp_cols:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    return md_table(out)


def yardstick_examples(core: pd.DataFrame, route: pd.DataFrame) -> dict:
    """Worked numbers used to explain the yardstick, the normal miss and the 2026 periods in the report."""
    monthly = core.groupby("date")["value_bd"].sum() / 1e6
    by_year = {y: monthly[monthly.index.year == y] for y in (2023, 2024, 2025, 2026)}
    avg = {y: s.mean() for y, s in by_year.items()}
    a25 = by_year[2025]
    misses = (a25 - avg[2024]).abs()
    miss_pct = {y: (by_year[y] - avg[y - 1]).abs().sum() / (avg[y - 1] * 12) * 100 for y in (2024, 2025)}
    month = lambda y, m: by_year[y][by_year[y].index.month == m].iloc[0]
    return {
        "avg": avg,
        "table": pd.DataFrame({"Month": a25.index.strftime("%b"), "Actual": a25.to_numpy(),
                               "Forecast": avg[2024], "Miss": misses.to_numpy()}),
        "miss_sum": misses.sum(), "forecast_sum": avg[2024] * 12, "miss_pct": miss_pct,
        "march_2026": month(2026, 3),
        "ruler": {m: {"2026": month(2026, m), "2025": month(2025, m)} for m in (6, 7)},
        "last_import": monthly.index.max(),
        "last_port": route["port_import_teu"].dropna().index.max(),
        "last_air": route["air_import_t"].dropna().index.max(),
    }


def write_report(ctx: dict) -> str:
    """The step 3 report. Numbers come from `ctx`; the interpretation was written for data to July 2026."""
    g = ctx["groups"]
    core_row, bulk_row, prec_row = g.loc[CORE], g.loc[BULK], g.loc[PRECIOUS]
    s, lv, ri, ex = ctx["season"], ctx["levels"].set_index("level"), ctx["routes"], ctx["examples"]
    regions, sections = ctx["regions"], ctx["sections"]
    gcc, asia, europe, americas = (regions.loc[r] for r in ("GCC", "East and South-East Asia", "Europe", "Americas"))
    gcc_share = ctx["gcc_share"]
    bulk_changes, bulk_origins = ctx["bulk_changes"], ctx["bulk_origins"]
    period_cols = ["Jan-Feb", *PERIODS]
    y, gr = s["yearly"] / 1e6, s["growth"]
    jj = ri.loc["2026-06-01":"2026-07-01"].mean()           # June-July average, as % of normal
    avg25 = ex["avg"][2025]
    march_change = (ex["march_2026"] / avg25 - 1) * 100
    r6, r7 = ex["ruler"][6], ex["ruler"][7]
    pct_vs = lambda a, b: signed((a / b - 1) * 100)
    chosen = lv.loc[CHOSEN_LEVEL]

    # --- tables, with readable column names ---
    groups_md = g.reset_index().rename(columns={"index": "Group", "group": "Group"})
    groups_md["Normal range (+/-%)"] = ctx["group_band"].reindex(groups_md["Group"]).to_numpy()
    groups_md = groups_md.rename(columns={"2025 value (BD m)": "2025 value (BD m)", "% of 2025 value": "% of value",
                                          "% of 2025 weight": "% of weight", "normal miss %": "Normal miss (%)"})
    groups_md = groups_md[["Group", "2025 value (BD m)", "% of value", "% of weight", "Normal miss (%)",
                           "Normal range (+/-%)", *period_cols]]

    example_md = ex["table"].copy()
    total_row = pd.DataFrame([{"Month": "**Total**", "Actual": np.nan, "Forecast": ex["forecast_sum"], "Miss": ex["miss_sum"]}])
    example_md = pd.concat([example_md, total_row], ignore_index=True)
    for c in ("Actual", "Forecast", "Miss"):
        example_md[c] = example_md[c].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}")

    route_md = ri.loc["2026-01-01":, ["core_value", "core_weight", "port_import_teu", "port_total_vessels", "air_import_t"]]
    route_md.columns = ["Core goods (value)", "Core goods (weight)", "Containers unloaded", "Ships calling",
                        "Air cargo unloaded"]
    route_md.insert(0, "Month", route_md.index.strftime("%b %Y"))

    def changes_md(table, band, label):
        out = table.copy()
        out.insert(0, label, out.index)
        out["Normal range (+/-%)"] = band
        out = out.rename(columns={"% of 2025": "% of 2025 value"})
        return table_md(out[[label, "% of 2025 value", "Normal range (+/-%)", *period_cols]], signed_cols=period_cols,
                        int_cols=["Normal range (+/-%)"], one_dp_cols=["% of 2025 value"])

    bulk_md = bulk_changes.copy()
    bulk_md.insert(0, "Material", bulk_md.index)
    bulk_md = bulk_md.rename(columns={"kt per month (2025)": "kt a month (2025)"})
    origins_md = bulk_origins.rename(columns={"material": "Material", "origin": "Origin",
                                              "kt per month, 2025": "kt a month, 2025",
                                              "kt per month, Mar-Jul 2026": "kt a month, Mar-Jul 2026"})
    jumpy_md = ctx["jumpy"].copy()
    jumpy_md.insert(0, "HS4", jumpy_md.index)
    jumpy_md = jumpy_md.rename(columns={"product": "Product", "% of misses": "% of misses", "% of value": "% of value"})
    levels_md = ctx["levels"][["level", "series", "regular series", "% of value in regular series", "normal miss %",
                               "same-month forecast miss %", "2026 gap %"]].copy()
    levels_md.columns = ["Level", "Series", "Regular series", "% of value in regular series", "Normal miss (%)",
                         "Same-month rule miss (%)", "Gap in Mar-Jul 2026 (%)"]

    return f"""
# Step 3: Exploring the data

Before building a forecasting model I wanted to understand the data well enough to answer three practical
questions. Should all imports be modelled together? What does a normal year look like, and so what does a
model need to know? And how finely can imports be split before ordinary noise is as large as the war itself?
Along the way this step takes a first look at what happened in 2026.

It uses imports from January 2021 to {ex['last_import']:%B %Y}, the latest month published. Port figures run
to {ex['last_port']:%B %Y} and airport figures to {ex['last_air']:%B %Y}, because they are published sooner.

## Key findings

- **Three kinds of imports behave very differently.** Everyday "core" goods are {core_row['% of 2025 value']:.0f}%
  of import value and fairly steady; bulk raw materials are {bulk_row['% of 2025 weight']:.0f}% of weight but
  arrive in lumpy shiploads; gold and jewellery swing with the gold price. They are analysed separately.
- **Imports are only weakly seasonal but they grow**, so a model needs the recent level of imports more than
  the calendar.
- **Core goods fell {abs(core_row['March']):.0f}% below normal in March 2026** and were still
  {abs(core_row['Jun-Jul']):.0f}% below in June-July.
- **Goods from GCC countries rose** while those from East Asia and the Americas stayed far below normal, a
  first sign that route mattered more than product.
- **Containers and air cargo fell much further than the goods arriving.** In June-July containers unloaded
  were at {jj['port_import_teu']:.0f}% of normal while core goods were back to {jj['core_weight']:.0f}% by
  weight, so a large share must have come another way, most likely by road.
- **The models will forecast each 2-digit customs chapter from each origin region.** Finer splits have so
  much normal noise that the war barely stands out.

## How changes are measured in this step

Every change in this report compares a 2026 figure with the average month of 2025 for the same thing. For
example, core goods averaged BD {avg25:,.1f}m a month in 2025 and were BD {ex['march_2026']:,.1f}m in March
2026, so March is reported as {signed(march_change)}%.

Comparing with the same month a year earlier sounds more natural, but a single month carries its own noise,
and that noise leaks into the comparison:

| Month | 2026 | Same month in 2025 | Against the same month | Against the 2025 average month |
|---|---|---|---|---|
| June | {r6['2026']:,.0f} | {r6['2025']:,.0f} | {pct_vs(r6['2026'], r6['2025'])}% | {pct_vs(r6['2026'], avg25)}% |
| July | {r7['2026']:,.0f} | {r7['2025']:,.0f} | {pct_vs(r7['2026'], r7['2025'])}% | {pct_vs(r7['2026'], avg25)}% |

*Core goods, BD million.*

June 2025 happened to be a weak month and July 2025 a strong one, so the same-month comparison flatters June
2026 and penalises July 2026. Tested on two normal years, the same-month rule also missed by more at every
level of detail (section 6). The average month is only a yardstick for exploring, though. From step 5 onward,
each month of 2026 is compared with a model's forecast for that specific month, which can allow for growth,
December peaks and Ramadan.

A change only means something if it is bigger than the normal ups and downs, so two measures of normal
movement are used:

- **Normal miss.** Forecast every month of a year as the previous year's average month, add up the misses
  (ignoring whether each month was above or below, so they can't cancel out) and divide by the total
  forecast. For core goods this was {ex['miss_pct'][2024]:.1f}% in 2024 and {ex['miss_pct'][2025]:.1f}% in
  2025, so about {core_row['normal miss %']:.0f}%. The 2025 calculation is shown below.
- **Normal range.** A single month can stray further than the average miss. For each month of 2023-2025 I
  measured how far it was from the previous year's average month; the normal range is the distance that 9
  months out of 10 stayed within. It is the grey band on the charts, and a dot outside the band is more than
  ordinary noise.

{md_table(example_md)}

*Core goods in 2025, forecast = the 2024 average month (BD million). Miss = {ex['miss_sum']:,.1f} /
{ex['forecast_sum']:,.0f} = {ex['miss_pct'][2025]:.1f}%.*

The months of 2026 are grouped by what was happening (dates and sources in `data/reference/events.csv`):

| Period | What was happening |
|---|---|
{chr(10).join(f"| **{p}** | {period_event(p)} |" for p in period_cols)}

## 1. Three kinds of imports

![Monthly imports](figures/step3_1_monthly_imports.png)

Imports were split into three groups:

- **Core goods**: everything except the two groups below, from machinery and vehicles to food, clothing and
  chemicals. These are the everyday supply chains and the main subject of the project.
- **Bulk materials**: iron ore, alumina, coke and construction minerals, delivered in shiploads to a few plants.
- **Precious metals**: gold, silver and jewellery (customs chapter 71).

{table_md(groups_md, signed_cols=period_cols, int_cols=["2025 value (BD m)", "% of value", "% of weight", "Normal miss (%)", "Normal range (+/-%)"])}

*The last four columns compare each 2026 period with the 2025 average month.*

Core goods are {core_row['% of 2025 value']:.0f}% of import value and move relatively little in a normal year
(a normal miss of about {core_row['normal miss %']:.0f}%). They were {vs_normal(core_row['Jan-Feb'])} normal in
January-February, {vs_normal(core_row['March'])} normal in March and still {vs_normal(core_row['Jun-Jul'])}
normal by June-July. Bulk materials are only {bulk_row['% of 2025 value']:.0f}% of value but
{bulk_row['% of 2025 weight']:.0f}% of weight; they fell further and stayed down
({vs_normal(bulk_row['Jun-Jul'])} normal in June-July). Precious metals follow the gold price. They were already
{vs_normal(prec_row['Jan-Feb'])} normal before the war, and their normal miss of about
{prec_row['normal miss %']:.0f}% is {prec_row['normal miss %'] / core_row['normal miss %']:.0f} times that of core goods.

Mixing the three would let one ore carrier or one gold consignment decide whether supply chains looked
better or worse than they were. From here on core goods are the main subject and the other two groups are
reported separately. The cost is that bulk materials and precious metals get a simpler treatment.

## 2. What a normal year looks like

![Seasonal pattern](figures/step3_5_patterns.png)

Two things matter for forecasting: how seasonal imports are, and whether they trend.

Seasonality is weak. When each month is compared with its own year's average (left chart), only about
{s['core_explained']:.0f}% of the month-to-month movement in core goods repeats from year to year. The
clearest pattern is December, about {s['dec_index'] - 100:.0f}% above a typical month. Ramadan makes no clear
difference either: for food, a full month of Ramadan goes with {signed(s['food_ramadan_this'])}% that month
and {signed(s['food_ramadan_next'])}% the month before, well within the normal ups and downs. With only four
years of data this is a rough check rather than a precise estimate.

Imports do trend. Core goods changed by {signed(gr[2023])}% in 2023, {signed(gr[2024])}% in 2024 and
{signed(gr[2025])}% in 2025, and were {vs_normal(core_row['Jan-Feb'])} the 2025 average in January-February
2026. The year 2021, under COVID rules, was smaller (BD {y[2021]:,.0f}m against BD {y[2022]:,.0f}m in 2022) but
had a similar shape. Even the closures of malls and restaurants in May-June 2021 barely show: those months
were {s['may_jun_2021']:.0f}% of that year's average month, against {s['may_jun_other']:.0f}% in other years.

For the models this means the recent level of imports matters more than the calendar, so they are given
averages of the last 3, 6 and 12 months. Calendar effects are small but are still offered as inputs, and 2021
is tested both with and without.

## 3. Where goods came from

![Origins](figures/step3_3_regions.png)

The customs data records where goods came from, not how they travelled, so origin is the best available clue
to route. Countries are grouped into regions by how goods normally reach Bahrain: GCC goods can come by road
over the King Fahd Causeway, while goods from East Asia, Europe or the Americas normally come by sea through
the Strait of Hormuz, or by air.

{changes_md(regions, ctx['region_band'], 'Origin region')}

*Core goods. "Other and not specified" (under 1% of value) is left out.*

Goods from GCC countries went up: {vs_normal(gcc['Apr-May'])} normal in April-May and
{vs_normal(gcc['Jun-Jul'])} normal in June-July. Their share of core goods rose from {gcc_share['2025']:.0f}% in
2025 to {gcc_share['Apr-May']:.0f}% in April-May. That fits the geography: GCC suppliers can send goods by road
through Saudi Arabia, and the UAE's and Oman's ports on the Gulf of Oman and the Arabian Sea are outside the
Strait. The biggest increases in June-July, compared with June-July 2025, were {names(ctx['gcc_drivers'])}.

Goods from East and South-East Asia, the biggest origin at {asia['% of 2025']:.0f}% of core goods, stayed down
longest: {vs_normal(asia['Apr-May'])} normal in April-May and still {vs_normal(asia['Jun-Jul'])} in June-July.
These goods normally come by container ship through Hormuz. The Americas were similar
({vs_normal(americas['Jun-Jul'])} normal in June-July).

Goods from Europe came back faster ({vs_normal(europe['Jun-Jul'])} normal in June-July), partly because of a
few large aircraft-related deliveries: {names(ctx['europe_drivers'])}. Leaving out aircraft, engines and their
parts, Europe was {vs_normal(ctx['europe_ex_aircraft'])} normal, still well ahead of Asia. Step 6 looks at why.

## 4. Which goods were hit

![Sections](figures/step3_4_sections.png)

Core goods were grouped into product sections, the top level of the customs classification.

{changes_md(sections, ctx['section_band'], 'Section')}

*Sections under {MIN_SECTION_SHARE}% of 2025 value, and the national special codes (confidential goods), are
left out.*

Every section fell in March. By June-July most were back inside their normal range, with food first: animal
products, vegetable products and fats and oils were at or above normal. Vehicles and clothing were still well
below normal and outside their range.

### Bulk materials, by weight

{table_md(bulk_md, signed_cols=period_cols, int_cols=["kt a month (2025)"])}

The main origins (any with at least 10% of a material's weight in 2025 or in March-July 2026):

{table_md(origins_md, one_dp_cols=["kt a month, 2025", "kt a month, Mar-Jul 2026"])}

Construction minerals show route exposure most clearly: supplies from Oman, which have to pass through the
Strait, stopped, while supplies from the UAE, inside the Gulf, went up. Alumina from Australia fell sharply and
new suppliers appeared. But bulk materials also mix in a second effect. Iron ore was already
{vs_normal(bulk_changes.loc['Iron ore', 'Jan-Feb'])} normal in January-February, before the war, and coke from
Kuwait, which doesn't need the Strait, also stopped in May-July, possibly because plants needed less (the
aluminium smelter, Alba, was attacked on 28 March). Because blocked routes and lower demand can't be told apart
here, bulk materials stay separate from core goods.

## 5. How goods arrived

![Routes](figures/step3_2_routes.png)

{table_md(route_md, int_cols=list(route_md.columns[1:]))}

*Each month as % of that series' 2025 monthly average. Port = Khalifa Bin Salman Port, Bahrain's container
port. Import figures stop at {ex['last_import']:%B}; port and airport figures are published sooner.*

In June-July, core goods were back to {jj['core_value']:.0f}% of normal by value and {jj['core_weight']:.0f}%
by weight. Containers unloaded at the port, though, were only at {jj['port_import_teu']:.0f}% of normal and
air cargo at {jj['air_import_t']:.0f}%. A much bigger share than usual must therefore have arrived another way.
The data covers the container port and the airport, so the most likely other way is by road over the King
Fahd Causeway from Saudi Arabia, although general cargo ships and smaller ports are possible too. The rise in
goods from GCC countries points the same way. This is the main route finding so far, and step 6 tests it
properly.

## 6. How much detail the models can handle

![Levels](figures/step3_6_levels.png)

The last question is how finely to split core goods for the models. I worked out the normal miss at levels
of detail from the national total down to each 4-digit product from each country. A series is regular if it
has imports in at least 54 of the 60 months of 2021-2025. The "same-month rule" uses the same month a year
earlier as the forecast, and the 2026 gap is the same calculation for March-July 2026.

{table_md(levels_md, int_cols=["Normal miss (%)", "Same-month rule miss (%)", "Gap in Mar-Jul 2026 (%)"], one_dp_cols=["% of value in regular series"])}

The national total is easy to forecast (a normal miss of about {lv.loc['National total', 'normal miss %']:.0f}%),
and the 2026 gap of about {lv.loc['National total', '2026 gap %']:.0f}% stands out clearly. The more detail,
though, the bigger the normal miss, because in a large total one product's high month cancels another's low
month, and splitting the total stops that cancelling. At section level the 2026 gap is
{lv.loc['Section', '2026 gap %'] / lv.loc['Section', 'normal miss %']:.1f} times the normal miss; at 4-digit
level it is only {lv.loc['HS4', '2026 gap %'] / lv.loc['HS4', 'normal miss %']:.1f} times; and for each product
from each country the normal miss ({lv.loc['HS4 x country', 'normal miss %']:.0f}%) is almost as large as the
war's effect. The average-month rule beats the same-month rule at every level, so it is the simple forecast the
models have to beat.

A few products cause a disproportionate share of the misses (general goods including precious metals,
2024-2025):

{table_md(jumpy_md, one_dp_cols=["% of misses", "% of value"])}

Gold and jewellery are part of why precious metals are kept separate. Aircraft engines and parts are large,
occasional purchases; they stay in core goods, but step 5 checks them so that one delivery isn't mistaken for a
recovery.

## Decisions for step 4

1. **Core goods are the main analysis.** Bulk materials (by weight) and precious metals are reported
   separately, because one shipment can swing them.
2. **The models forecast each 2-digit chapter from each origin region.** That gives
   {int(chosen['regular series']):,} regular series covering {chosen['% of value in regular series']:.1f}% of core
   goods, at a level where the war is still clearly bigger than normal noise. Patchy series are pooled into one
   "other" series per section and region. Adding the forecasts up gives totals by section, by region and for the
   whole country that agree with each other. The cost is that a shift between two products in the same chapter
   is invisible to the models.
3. **Results are reported nationally, by section and by region**, where the effect is clearly bigger than the
   normal miss. More detailed results are only shown with their uncertainty.
4. **The simple forecast to beat is the previous year's average month**, with the same-month rule reported too.
5. **The models get the recent level and trend** (averages over the last 3, 6 and 12 months), because imports
   were growing going into 2026.
6. **2021 is tested both ways**: kept in with a COVID flag, and left out.

## Limitations

- The yardstick in this step ignores growth. Imports were rising going into 2026, so falls measured against
  2025 understate how far imports were below where they would otherwise have been. Step 5 corrects this by using
  model forecasts.
- The normal range is based on three years (36 months per series), so it is a rough guide, not a precise 90%
  interval.
- Origin region is a proxy for route. A good from China could have come by air, or have been landed in Saudi
  Arabia and trucked in.
- Customs records the month goods were declared, which is not always the month they arrived. Periods of two
  months soften this.

{footer("src/explore.py")}
"""


def main() -> int:
    """Load the data, run every analysis, draw the six charts and write the report."""
    df = load_imports()
    route = pd.read_parquet(PROCESSED_DIR / "route_monthly.parquet").set_index("date")
    calendar = pd.read_csv(REFERENCE_DIR / "calendar.csv", parse_dates=["date"])
    core = df[df["group"] == CORE]

    extend_periods(df["date"].max())          # add a period for any months published after July
    ctx: dict = {"latest_month": df["date"].max()}
    ctx["groups"] = group_summary(df)
    wg = wide(df, ["group"])
    wg.index = wg.index.astype(str)
    ctx["group_band"] = normal_swing(wg)
    ctx["examples"] = yardstick_examples(core, route)
    ctx["routes"] = route_index(df, route)
    ctx["bulk_changes"], ctx["bulk_origins"] = bulk_tables(df)
    regions, region_band = change_table(core, "region")
    ctx["regions"], ctx["region_band"] = regions.drop(index=OTHER_REGION), region_band.drop(index=OTHER_REGION)
    sections, section_band = change_table(core, "section_name")
    keep = (sections["% of 2025"] >= MIN_SECTION_SHARE) & ~sections.index.isin(SKIP_SECTIONS)
    ctx["sections"], ctx["section_band"] = sections[keep], section_band[keep]
    ctx["levels"] = level_table(core)
    ctx["season"] = seasonality(core, calendar)
    ctx["jumpy"] = jumpy_products(df)
    ctx["gcc_drivers"] = biggest_increases(core, "GCC", [6, 7])
    ctx["europe_drivers"] = biggest_increases(core, "Europe", [6, 7], n=2)
    # Europe again without aircraft, engines and their parts, to check its recovery isn't just a few deliveries.
    europe = core[(core["region"] == "Europe") & ~core["hs4"].astype(str).isin(AIRCRAFT_HS4)]
    ctx["europe_ex_aircraft"] = change_vs_2025(wide(europe, []), [6, 7]).iloc[0]
    # GCC share of core goods in 2025 and in each 2026 period.
    wr = wide(core, ["region"])
    ctx["gcc_share"] = {"2025": months_of(wr, 2025).loc["GCC"].sum() / months_of(wr, 2025).to_numpy().sum() * 100}
    for p, months in {"Jan-Feb": PRE_WAR, **PERIODS}.items():
        part = months_of(wr, 2026, months)
        ctx["gcc_share"][p] = part.loc["GCC"].sum() / part.to_numpy().sum() * 100

    fig_monthly(df)
    fig_routes(ctx["routes"])
    # Both dot plots share one x-axis so they can be compared; it runs from -100% to just past the largest rise.
    shown = pd.concat([ctx["regions"], ctx["sections"]])[list(PERIODS)]
    xlim = (-100, max(50, np.ceil(shown.max().max() / 10) * 10 + 10))
    for table, band, title, name, dp in [
        (ctx["regions"], ctx["region_band"],
         "Goods from GCC countries rose, while those from Asia and the Americas stayed far below normal",
         "step3_3_regions.png", 0),
        (ctx["sections"], ctx["section_band"],
         "Every section fell in March; by June-July vehicles and clothing were still well below normal",
         "step3_4_sections.png", 1),
    ]:
        labelled = table.copy()
        labelled.index = [f"{r}  ({table.loc[r, '% of 2025']:.{dp}f}%)" for r in table.index]
        dot_plot(labelled, pd.Series(band.to_numpy(), index=labelled.index), title,
                 "Core goods (share of 2025 value in brackets): % change against the 2025 monthly average.\n"
                 "Grey band: in 2023-2025, 9 months out of 10 were within this range of the previous year's average.",
                 name, xlim)
    fig_patterns(core)
    fig_levels(ctx["levels"], CHOSEN_LEVEL)

    write_markdown(REPORT_PATH, write_report(ctx))
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and 6 figures in {FIG_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
