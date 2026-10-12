"""Step 3: explore the cleaned data before building any models.

    python src/explore.py

Reads the step 2 tables and writes:
  * reports/figures/step3_*.png    six charts
  * reports/step3_exploration.md   what they show, and the decisions for step 4

Every number in the report is computed here, so re-running after the portal adds
a month refreshes the numbers. The chart titles and the written interpretation
were checked against data up to July 2026.

Throughout, "normal" means 2025: a month's change is measured against the 2025
monthly average of the same series. Imports have little seasonality (shown
below), so the average is a steadier yardstick than a single month a year earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter

from build_dataset import md_table
from config import REFERENCE_DIR, ROOT
from reference import BULK, CORE, OTHER_REGION, PRECIOUS, add_analysis_group, add_region

PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "step3_exploration.md"

# 2026 periods, by month (from data/reference/events.csv). The war began on 28 February.
PRE_WAR = [1, 2]
PERIODS = {                       # label -> months of 2026
    "March": [3],                 # shock: Hormuz and airspace closed, port struck
    "Apr-May": [4, 5],            # blockade: airspace reopens 8 Apr, Hormuz still closed
    "Jun-Jul": [6, 7],            # brief partial reopening 18 Jun - 10 Jul, then closed again
}
POST_WAR = [m for months in PERIODS.values() for m in months]
WAR_START = pd.Timestamp("2026-02-28")
COVID_END = pd.Timestamp("2022-03-28")

MIN_SECTION_SHARE = 0.5                       # % of 2025 core value; smaller sections are left out of the chart
SKIP_SECTIONS = {"National special codes"}    # confidential and special codes, not a supply chain
AIRCRAFT_HS4 = {"8409", "8411", "8802", "8803"}

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

# Chart style (dataviz reference palette; colours checked with validate_palette.js).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SHADE = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PERIOD_COLOURS = dict(zip(PERIODS, [BLUE, ORANGE, AQUA]))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 100, "savefig.dpi": 200,
    "axes.edgecolor": BASELINE, "axes.linewidth": 1, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9.5,
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_imports() -> pd.DataFrame:
    cols = ["date", "year", "month", "section_id", "section_name", "product_group", "hs2", "hs4",
            "iso2", "country_en", "value_bd", "weight_kg"]
    df = pd.read_parquet(PROCESSED_DIR / "imports_clean.parquet", columns=cols)
    add_region(df)
    add_analysis_group(df)
    return df


def wide(df: pd.DataFrame, keys: list[str], col: str = "value_bd") -> pd.DataFrame:
    """Series x month table: one row per series, one column per month, missing months = 0."""
    if not keys:
        return df.groupby("date")[col].sum().to_frame("total").T
    return df.groupby([*keys, "date"], observed=True)[col].sum().unstack("date", fill_value=0.0)


def months_of(w: pd.DataFrame, year: int, months: list[int] | None = None) -> pd.DataFrame:
    mask = w.columns.year == year
    if months is not None:
        mask &= w.columns.month.isin(months)
    return w.loc[:, mask]


def change_vs_2025(w: pd.DataFrame, months: list[int]) -> pd.Series:
    """% change of the average 2026 month in `months` vs the 2025 monthly average, per row."""
    base = months_of(w, 2025).mean(axis=1)
    return (months_of(w, 2026, months).mean(axis=1) / base - 1) * 100


def normal_swing(w: pd.DataFrame, years=(2023, 2024, 2025), q: float = 0.9) -> pd.Series:
    """Per row: 9 months out of 10 in `years` were within +/- this % of the previous year's average."""
    swings = [months_of(w, y).div(months_of(w, y - 1).mean(axis=1), axis=0) - 1 for y in years]
    return pd.concat(swings, axis=1).abs().quantile(q, axis=1) * 100


def gap(w: pd.DataFrame, year: int, months: list[int] | None = None, forecast: str = "average") -> float:
    """Average gap between actual and a simple forecast, as % of the forecast (summed over all series).

    forecast="average": every month of `year` is forecast as the previous year's monthly average.
    forecast="same month": every month is forecast as the same month a year earlier.
    """
    actual = months_of(w, year, months)
    if forecast == "average":
        fc = np.repeat(months_of(w, year - 1).mean(axis=1).to_numpy()[:, None], actual.shape[1], axis=1)
    else:
        fc = months_of(w, year - 1, list(actual.columns.month)).to_numpy()
    return float(np.abs(actual.to_numpy() - fc).sum() / fc.sum() * 100)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def new_figure(width: float, height: float, title: str, subtitle: str):
    fig = plt.figure(figsize=(width, height))
    fig.text(0.6 / width, 1 - 0.32 / height, title, fontsize=14, fontweight="semibold", color=INK, va="top")
    fig.text(0.6 / width, 1 - 0.62 / height, subtitle, fontsize=10, color=INK2, va="top", linespacing=1.45)
    return fig


def add_axes(fig, left: float, right: float, top: float, bottom: float):
    """Axes placed by its distance in inches from each edge of the figure."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (top + bottom) / h])


def legend(fig, handles, labels, top: float, ncol: int):
    """Legend in a row, `top` inches below the top of the figure, aligned with the title."""
    w, h = fig.get_size_inches()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.52 / w, 1 - top / h), ncol=ncol,
               handlelength=1.4, columnspacing=1.6, borderaxespad=0)


def tidy(ax, grid_axis: str = "y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if grid_axis == "x":
        ax.spines["bottom"].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def source_note(fig, text: str = "Source: Bahrain Open Data Portal (iGA), www.data.gov.bh. Analysis: own."):
    w, h = fig.get_size_inches()
    fig.text(0.6 / w, 0.14 / h, text, fontsize=8, color=MUTED)


def line_handle(colour, lw=2):
    return plt.Line2D([], [], color=colour, lw=lw)


def dot_handle(colour):
    return plt.Line2D([], [], ls="", marker="o", ms=8, color=colour, mec=SURFACE, mew=2)


def end_dot(ax, x, y, colour):
    ax.plot([x], [y], "o", ms=8, color=colour, mec=SURFACE, mew=2, zorder=5)


def end_labels(ax, items, x_text=None, gap_pt: float = 13):
    """Label line ends without overlaps.

    items: (x, y, text) per line. Labels sit to the right of their point, or in a shared
    column at data-x `x_text`. Labels that would overlap are spread apart vertically and
    joined to their point by a thin leader line.
    """
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    pts = []
    for x, y, text in items:
        xn = mdates.date2num(x) if isinstance(x, pd.Timestamp) else x
        px, py = to_px((xn, y))
        pts.append({"x": x, "y": y, "text": text, "px": px, "py": py})
    gap_px = gap_pt * fig.dpi / 72
    pts.sort(key=lambda p: -p["py"])
    ys = [p["py"] for p in pts]
    for k in range(1, len(ys)):
        ys[k] = min(ys[k], ys[k - 1] - gap_px)
    shift = np.mean([p["py"] for p in pts]) - np.mean(ys)        # keep the block centred on the points
    col_px = to_px((mdates.date2num(x_text) if isinstance(x_text, pd.Timestamp) else x_text, 0))[0] if x_text is not None else None
    for p, ty in zip(pts, ys):
        ty += shift
        tx = col_px if col_px is not None else p["px"] + 9 * fig.dpi / 72
        dx, dy = (tx - p["px"]) * 72 / fig.dpi, (ty - p["py"]) * 72 / fig.dpi
        leader = abs(dy) > 1.5 or col_px is not None
        ax.annotate(p["text"], (p["x"], p["y"]), xytext=(dx, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8, shrinkA=3, shrinkB=6) if leader else None)


def rounded_bar(ax, start, length, centre, thickness, colour, horizontal=True, radius_px=4):
    """Bar with a 4px rounded data end and a square end at the baseline (call after setting limits)."""
    fig = ax.figure
    box = ax.get_window_extent(fig.canvas.get_renderer())
    x_per_px = abs(np.ptp(ax.get_xlim())) / box.width
    y_per_px = abs(np.ptp(ax.get_ylim())) / box.height
    M, L, C, Z = Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CLOSEPOLY
    codes = [M, L, C, C, L, C, C, L, Z]
    if horizontal:
        ra, rc = min(radius_px * x_per_px, length / 2), min(radius_px * y_per_px, thickness / 2)
        x0, x1, y0, y1 = start, start + length, centre - thickness / 2, centre + thickness / 2
        verts = [(x0, y0), (x1 - ra, y0), (x1, y0), (x1, y0 + rc), (x1, y1 - rc), (x1, y1), (x1 - ra, y1), (x0, y1), (x0, y0)]
    else:
        ra, rc = min(radius_px * y_per_px, length / 2), min(radius_px * x_per_px, thickness / 2)
        x0, x1, y0, y1 = centre - thickness / 2, centre + thickness / 2, start, start + length
        verts = [(x1, y0), (x1, y1 - ra), (x1, y1), (x1 - rc, y1), (x0 + rc, y1), (x0, y1), (x0, y1 - ra), (x0, y0), (x1, y0)]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=colour, lw=0, zorder=3))


def month_axis(ax, months=(1, 7), first=None):
    """Month ticks; January (and the first tick, if given) also shows the year."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))

    def fmt(x, _):
        d = mdates.num2date(x)
        show_year = d.month == 1 or (first is not None and d.year == first.year and d.month == first.month)
        return d.strftime("%b\n%Y") if show_year else d.strftime("%b")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_monthly(df: pd.DataFrame) -> None:
    """Figure 1: monthly value and weight by analysis group, 2021 to latest month."""
    value = df.groupby(["date", "group"], observed=True)["value_bd"].sum().unstack() / 1e6
    weight = df.groupby(["date", "group"], observed=True)["weight_kg"].sum().unstack() / 1e6
    colours = {CORE: BLUE, BULK: ORANGE, PRECIOUS: AQUA}
    latest = value.index.max()

    fig = new_figure(10, 8.05, "Imports fell sharply after the war began on 28 February 2026",
                     f"Monthly non-oil imports, January 2021 to {latest:%B %Y}, split into three groups that behave differently.")
    legend(fig, [line_handle(colours[g]) for g in (CORE, BULK, PRECIOUS)], [CORE, BULK, PRECIOUS], top=0.98, ncol=3)
    panels = [(value, [CORE, BULK, PRECIOUS], "Value (BD million)", 1.6, 4.15),
              (weight, [BULK, CORE], "Weight (thousand tonnes)", 4.95, 0.8)]
    for data, groups, label, top, bottom in panels:
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
    """Figure 2: goods arriving vs port containers, ship visits and air cargo, as % of the 2025 average."""
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
    """Rows = series (top to bottom in table order), one dot per 2026 period, grey band = normal swing."""
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
    """Figure 5: month-of-year pattern by year (index) and yearly totals."""
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
    """Figure 6: normal forecast miss vs the 2026 gap, by level of detail."""
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
    group_total = origins.groupby(level=0).transform("sum")
    keep = ((origins / group_total) >= 0.1).any(axis=1) & (origins.max(axis=1) >= 1)
    origins = origins[keep].reset_index()
    origins["order"] = origins["product_group"].map({g: i for i, g in enumerate(changes.index)})
    origins = origins.sort_values(["order", "kt per month, 2025"], ascending=[True, False]).drop(columns="order")
    return changes, origins.rename(columns={"product_group": "material", "country": "origin"})


def change_table(core: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.Series]:
    w = wide(core, [key])
    w.index = w.index.astype(str)
    share = months_of(w, 2025).sum(axis=1) / months_of(w, 2025).to_numpy().sum() * 100
    table = pd.DataFrame({"% of 2025": share, "Jan-Feb": change_vs_2025(w, PRE_WAR),
                          **{p: change_vs_2025(w, m) for p, m in PERIODS.items()}})
    order = table.sort_values("% of 2025", ascending=False).index
    return table.loc[order], normal_swing(w).loc[order]


LEVELS = {        # ordered from least to most detailed
    "National total": [], "Origin region": ["region"], "Section": ["section_id"], "HS2": ["hs2"],
    "Section x region": ["section_id", "region"], "HS2 x region": ["hs2", "region"],
    "HS4": ["hs4"], "HS4 x region": ["hs4", "region"], "HS4 x country": ["hs4", "iso2"],
}
CHOSEN_LEVEL = "HS2 x region"


def level_table(core: pd.DataFrame) -> pd.DataFrame:
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
    """How much of the month-to-month movement is a calendar pattern, and is there a Ramadan effect?"""
    out = {}
    ramadan = calendar.set_index("date")["ramadan_days"]
    for name, part in (("core", core), ("food", core[core["section_id"] <= 4])):
        monthly = part.groupby("date")["value_bd"].sum()
        monthly = monthly[(monthly.index.year >= 2022) & (monthly.index.year <= 2025)]
        idx = monthly / monthly.groupby(monthly.index.year).transform("mean")
        month_mean = idx.groupby(idx.index.month).transform("mean")
        out[f"{name}_explained"] = (1 - ((idx - month_mean) ** 2).sum() / ((idx - 1) ** 2).sum()) * 100
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
    """Which 4-digit products cause most of the simple forecast's misses (general goods, 2024-2025)?"""
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
def signed(x: float) -> str:
    r = int(round(x))
    return "0" if r == 0 else f"{r:+d}"


def vs_normal(x: float) -> str:
    """-53.2 -> '53% below', 7.1 -> '7% above'."""
    r = int(round(x))
    return "about the same as" if r == 0 else f"{abs(r)}% {'above' if r > 0 else 'below'}"


def names(series: pd.Series) -> str:
    parts = [f"{HS4_NAMES.get(code, 'HS ' + code).lower()} ({'+' if v >= 0 else '-'}BD {abs(v):.1f}m)" for code, v in series.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def table_md(df: pd.DataFrame, signed_cols=(), int_cols=(), one_dp_cols=()) -> str:
    out = df.copy()
    for c in signed_cols:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else signed(v))
    for c in int_cols:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    for c in one_dp_cols:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    return md_table(out)


def write_report(ctx: dict) -> str:
    g = ctx["groups"]
    core_row, bulk_row, prec_row = g.loc[CORE], g.loc[BULK], g.loc[PRECIOUS]
    s, lv, ri = ctx["season"], ctx["levels"].set_index("level"), ctx["routes"]
    regions, sections = ctx["regions"], ctx["sections"]
    gcc, asia, europe = regions.loc["GCC"], regions.loc["East and South-East Asia"], regions.loc["Europe"]
    gcc_share = ctx["gcc_share"]
    bulk_changes, bulk_origins = ctx["bulk_changes"], ctx["bulk_origins"]
    period_cols = ["Jan-Feb", *PERIODS]
    y, gr = s["yearly"] / 1e6, s["growth"]
    jj = ri.loc["2026-06-01":"2026-07-01"].mean()

    groups_md = g.reset_index().rename(columns={"index": "group"})
    route_md = ri.loc["2026-01-01":, ["core_value", "core_weight", "port_import_teu", "port_total_vessels", "air_import_t"]]
    route_md.columns = ["core goods value", "core goods weight", "containers unloaded", "ships calling", "air cargo unloaded"]
    route_md.insert(0, "month", route_md.index.strftime("%b %Y"))

    def changes_md(table, band, label):
        out = table.copy()
        out.insert(0, label, out.index)
        out["normal range +/-"] = band
        return table_md(out[[label, "% of 2025", *period_cols, "normal range +/-"]], signed_cols=period_cols,
                        int_cols=["normal range +/-"], one_dp_cols=["% of 2025"])

    bulk_md = bulk_changes.copy()
    bulk_md.insert(0, "material", bulk_md.index)

    jumpy_md = ctx["jumpy"].copy()
    jumpy_md.insert(0, "HS4", jumpy_md.index)

    levels_md = ctx["levels"][["level", "series", "regular series", "% of value in regular series", "normal miss %",
                               "same-month forecast miss %", "2026 gap %"]]

    return f"""# Step 3: exploring the data

Generated by `src/explore.py` from the step 2 tables. Imports run from Jan 2021 to {ctx['latest_month']:%b %Y}.

**How to read the numbers.** Unless stated otherwise, a change is measured against the **2025 monthly average** of the same thing,
so "-50" in March means March 2026 was half an average 2025 month. The 2026 periods follow the events in
`data/reference/events.csv`: **Jan-Feb** (before the war), **March** (shock: Strait of Hormuz and airspace closed, port struck),
**Apr-May** (airspace open again, Hormuz still closed, US blockade in force), **Jun-Jul** (a brief, partial reopening
from 18 June, then closed again on 11 July).

## 1. The big picture

![Monthly imports](figures/step3_1_monthly_imports.png)

Imports split into three groups that behave very differently, so from now on they are analysed separately:

{table_md(groups_md, signed_cols=period_cols, int_cols=["2025 value (BD m)", "% of 2025 value", "% of 2025 weight", "normal miss %"])}

*Normal miss: on average, how far a simple forecast (last year's monthly average) was from each month's actual value in
2024-2025, as % of the forecast. The last four columns are % changes against the 2025 monthly average.*

- **Core goods** (everything except bulk materials and precious metals) are the everyday supply chains and the main subject
  of the project. They were {vs_normal(core_row['Jan-Feb'])} the 2025 average in Jan-Feb, {vs_normal(core_row['March'])}
  it in March, and {vs_normal(core_row['Jun-Jul'])} it by Jun-Jul.
- **Bulk materials** (iron ore, alumina, coke and construction minerals) arrive as whole shiploads for a few factories.
  They collapsed and stayed down: {vs_normal(bulk_row['Jun-Jul'])} normal in Jun-Jul (section 4).
- **Precious metals** (gold, silver, jewellery) follow the gold price and a few very large shipments. They were already
  {vs_normal(prec_row['Jan-Feb'])} normal in Jan-Feb, before the war, and their normal miss is about
  {prec_row['normal miss %']:.0f}%. They say little about supply chains, so they are reported separately and kept out of core goods.

## 2. What "normal" looks like

![Seasonal pattern](figures/step3_5_patterns.png)

- **Seasonality is weak.** Only about {s['core_explained']:.0f}% of the month-to-month movement in core goods (2022-2025)
  is a repeating calendar pattern; the rest is noise. The clearest pattern is December, about
  {s['dec_index'] - 100:.0f}% above the year's average. This is why "last year's monthly average" beats
  "the same month last year" as a forecast at every level of detail (section 6).
- **No clear Ramadan effect.** For food (HS sections I-IV), a full month of Ramadan changes imports by about
  {signed(s['food_ramadan_this'])}% that month and {signed(s['food_ramadan_next'])}% the month before, well within the noise.
  The calendar features stay in the models, but they are not expected to matter much.
- **Imports grow over time.** Core goods changed by {signed(gr[2023])}% in 2023, {signed(gr[2024])}% in 2024 and
  {signed(gr[2025])}% in 2025, and were still rising going into 2026 ({vs_normal(core_row['Jan-Feb'])} the 2025 average in Jan-Feb).
  The models need features for the recent level and trend, not just last year's pattern.
- **2021 (COVID rules) was smaller but not unusual in shape.** Core goods were BD {y[2021]:,.0f}m in 2021 against
  {y[2022]:,.0f}m in 2022 (2022 was {gr[2022]:.0f}% higher). May-June 2021, when malls and restaurants were closed, averaged
  {s['may_jun_2021']:.0f}% of that year's monthly average, against {s['may_jun_other']:.0f}% in 2022-2025, so the closures barely show.
  Whether to train on 2021 is tested in step 4, as planned.

## 3. Which origins were hit, and where goods came from instead

![Origins](figures/step3_3_regions.png)

{changes_md(regions, ctx['region_band'], 'origin region')}

*Core goods. Normal range: in 2023-2025, 9 months out of 10 were within this many % of the previous year's average, so
changes inside it are not unusual. "Other and not specified" (under 1% of value) is left out.*

- **Goods from GCC countries rose.** They were {vs_normal(gcc['Apr-May'])} normal in Apr-May and {vs_normal(gcc['Jun-Jul'])}
  in Jun-Jul, and their share of core goods went from {gcc_share['2025']:.0f}% in 2025 to {gcc_share['Apr-May']:.0f}% in Apr-May
  and {gcc_share['Jun-Jul']:.0f}% in Jun-Jul. GCC suppliers can reach Bahrain by road through Saudi Arabia, and UAE and Omani ports
  on the Gulf of Oman and Arabian Sea are outside the Strait of Hormuz. The largest increases in Jun-Jul (against the same months of
  2025) were {names(ctx['gcc_drivers'])}.
- **East and South-East Asia**, the biggest origin ({asia['% of 2025']:.0f}% of core goods), stayed down longest:
  {vs_normal(asia['Apr-May'])} normal in Apr-May and still {vs_normal(asia['Jun-Jul'])} in Jun-Jul. These goods normally come
  by container ship through Hormuz.
- **Europe** came back faster ({vs_normal(europe['Jun-Jul'])} normal in Jun-Jul). Part of that is a few large aircraft-related
  deliveries: {names(ctx['europe_drivers'])} against the same months of 2025. Even without aircraft, engines and their parts,
  Europe was {vs_normal(ctx['europe_ex_aircraft'])} normal, well ahead of Asia.

## 4. Which supply chains were hit

![Sections](figures/step3_4_sections.png)

{changes_md(sections, ctx['section_band'], 'section')}

*Core goods by HS section. Sections under {MIN_SECTION_SHARE}% of 2025 value, and the national special codes (confidential goods),
are left out of the chart and table.*

Every section fell in March. By Jun-Jul most were back inside their normal range, but vehicles and clothing were still well below it.
Food came back first: animal products, vegetable products and fats and oils were at or above normal by Jun-Jul.

**Bulk materials, by weight**

{table_md(bulk_md, signed_cols=period_cols, int_cols=["kt per month (2025)"])}

Main origins (each listed origin supplied at least 10% of that material in 2025 or in Mar-Jul 2026):

{table_md(bulk_origins, one_dp_cols=["kt per month, 2025", "kt per month, Mar-Jul 2026"])}

- **Construction minerals show route exposure cleanly**: supplies from Oman, which have to pass through the Strait of Hormuz,
  stopped, while those from the UAE, inside the Gulf, rose.
- **Alumina from Australia** fell sharply and new suppliers appeared.
- **Iron ore** was already {vs_normal(bulk_changes.loc['Iron ore', 'Jan-Feb'])} normal in Jan-Feb, before the war, so not all of its fall is the war.
- **Coke from Kuwait**, which does not need the Strait, had no shipments in May-July either. That may reflect lower demand from
  the factories (Alba, the aluminium smelter, was attacked on 28 March) rather than routes alone. Bulk materials mix route and
  factory effects, which is another reason to keep them separate from core goods.

## 5. Routes: did goods stop coming by sea and air?

![Routes](figures/step3_2_routes.png)

{table_md(route_md, int_cols=list(route_md.columns[1:]))}

*Each figure is % of that series' 2025 monthly average. Port = Khalifa Bin Salman Port, Bahrain's container port.*

In Jun-Jul, core goods were back to {jj['core_value']:.0f}% of normal by value and {jj['core_weight']:.0f}% by weight,
but containers unloaded at the port were only {jj['port_import_teu']:.0f}% of normal and air cargo {jj['air_import_t']:.0f}%.
So a much larger share than usual arrived some other way. The data covers the container port and the airport, so the
remaining route is mostly overland, by truck over the King Fahd Causeway from Saudi Arabia (smaller ports are also possible).
Section 3 points the same way: imports from GCC countries rose. Step 6 tests this properly.

## 6. How detailed can the forecasts be?

![Levels](figures/step3_6_levels.png)

{table_md(levels_md, int_cols=["normal miss %", "same-month forecast miss %", "2026 gap %"], one_dp_cols=["% of value in regular series"])}

*Core goods. A "series" is one line of monthly numbers, for example "HS2 chapter 84 from Europe". Regular series have imports in
at least 54 of the 60 months of 2021-2025. Normal miss and same-month forecast miss: average monthly gap in 2024-2025 between actual
imports and, respectively, last year's monthly average and the same month last year, as % of the forecast.*

- The national total is easy to forecast (normal miss about {lv.loc['National total', 'normal miss %']:.0f}%), and the 2026 gap
  (about {lv.loc['National total', '2026 gap %']:.0f}%) is far bigger than that.
- The more detail, the bigger the normal miss. At section level the 2026 gap is about
  {lv.loc['Section', '2026 gap %'] / lv.loc['Section', 'normal miss %']:.1f} times the normal miss; at HS4 it is only
  {lv.loc['HS4', '2026 gap %'] / lv.loc['HS4', 'normal miss %']:.1f} times, and at HS4 x country the normal miss
  ({lv.loc['HS4 x country', 'normal miss %']:.0f}%) is almost as big as the crisis itself.
- "Last year's monthly average" beats "the same month last year" at every level, so it is the simple forecast the models must beat.

Products that make the totals jumpy (general goods including precious metals, 2024-2025):

{table_md(jumpy_md, one_dp_cols=["% of misses", "% of value"])}

*% of misses: share of all the simple forecast's monthly misses that came from this product.* Gold and jewellery are why precious
metals are split out. Aircraft engines and parts are large, irregular purchases; they stay in core goods but are watched in step 5,
so that one delivery isn't mistaken for a recovery.

## Decisions for step 4

1. **Three groups, analysed separately**: core goods (the main analysis), bulk materials (by weight; few series, driven by
   shiploads and factory demand) and precious metals (reported, not used to judge supply chains).
2. **The simple forecast to beat** is last year's monthly average, with "same month last year" also reported.
3. **Bottom level for the models: HS2 chapter x origin region.** Its {lv.loc['HS2 x region', 'regular series']:,} regular series
   cover {lv.loc['HS2 x region', '% of value in regular series']:.1f}% of core goods; the patchy ones are pooled into one "other"
   series per section and region. Adding the forecasts up gives consistent totals by section, by origin region and for the whole
   country. HS4 forecasts are a check and are used for drill-downs (for example cars or phones).
4. **Results are reported at national, section and region level**, where the crisis is clearly bigger than the normal miss.
   Finer detail is shown only with its uncertainty range.
5. **Features for recent level and trend** (3-, 6- and 12-month averages), not just last year's same month, because imports
   were growing into 2026.
6. **2021 is tested both ways** (kept with a COVID flag, and left out).
"""


def main() -> int:
    df = load_imports()
    route = pd.read_parquet(PROCESSED_DIR / "route_monthly.parquet").set_index("date")
    calendar = pd.read_csv(REFERENCE_DIR / "calendar.csv", parse_dates=["date"])
    core = df[df["group"] == CORE]

    ctx: dict = {"latest_month": df["date"].max()}
    ctx["groups"] = group_summary(df)
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
    europe = core[(core["region"] == "Europe") & ~core["hs4"].astype(str).isin(AIRCRAFT_HS4)]
    ctx["europe_ex_aircraft"] = change_vs_2025(wide(europe, []), [6, 7]).iloc[0]
    wr = wide(core, ["region"])
    ctx["gcc_share"] = {"2025": months_of(wr, 2025).loc["GCC"].sum() / months_of(wr, 2025).to_numpy().sum() * 100}
    for p, months in {"Jan-Feb": PRE_WAR, **PERIODS}.items():
        part = months_of(wr, 2026, months)
        ctx["gcc_share"][p] = part.loc["GCC"].sum() / part.to_numpy().sum() * 100

    fig_monthly(df)
    fig_routes(ctx["routes"])
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

    REPORT_PATH.write_text(write_report(ctx), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and 6 figures in {FIG_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
