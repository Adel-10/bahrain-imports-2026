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


def yardstick_examples(core: pd.DataFrame, route: pd.DataFrame) -> dict:
    """Worked numbers used to explain the yardstick, the normal miss and the 2026 periods in plain words."""
    monthly = core.groupby("date")["value_bd"].sum() / 1e6
    by_year = {y: monthly[monthly.index.year == y] for y in (2023, 2024, 2025, 2026)}
    avg = {y: s.mean() for y, s in by_year.items()}
    a25 = by_year[2025]
    misses = (a25 - avg[2024]).abs()
    miss_pct = {y: (by_year[y] - avg[y - 1]).abs().sum() / (avg[y - 1] * 12) * 100 for y in (2024, 2025)}
    month = lambda y, m: by_year[y][by_year[y].index.month == m].iloc[0]
    return {
        "avg": avg,
        "table": pd.DataFrame({"month": a25.index.strftime("%b"), "actual": a25.to_numpy(),
                               "forecast": avg[2024], "miss": misses.to_numpy()}),
        "miss_sum": misses.sum(), "forecast_sum": avg[2024] * 12, "miss_pct": miss_pct,
        "march_2026": month(2026, 3),
        "ruler": {m: {"2026": month(2026, m), "2025": month(2025, m)} for m in (6, 7)},
        "last_import": monthly.index.max(),
        "last_port": route["port_import_teu"].dropna().index.max(),
        "last_air": route["air_import_t"].dropna().index.max(),
    }


def write_report(ctx: dict) -> str:
    g = ctx["groups"]
    core_row, bulk_row, prec_row = g.loc[CORE], g.loc[BULK], g.loc[PRECIOUS]
    s, lv, ri, ex = ctx["season"], ctx["levels"].set_index("level"), ctx["routes"], ctx["examples"]
    regions, sections = ctx["regions"], ctx["sections"]
    gcc, asia, europe = regions.loc["GCC"], regions.loc["East and South-East Asia"], regions.loc["Europe"]
    gcc_share = ctx["gcc_share"]
    bulk_changes, bulk_origins = ctx["bulk_changes"], ctx["bulk_origins"]
    period_cols = ["Jan-Feb", *PERIODS]
    y, gr = s["yearly"] / 1e6, s["growth"]
    jj = ri.loc["2026-06-01":"2026-07-01"].mean()
    avg25 = ex["avg"][2025]
    march_change = (ex["march_2026"] / avg25 - 1) * 100
    r6, r7 = ex["ruler"][6], ex["ruler"][7]
    pct_vs = lambda a, b: signed((a / b - 1) * 100)

    groups_md = g.reset_index().rename(columns={"index": "group"})
    groups_md["normal range +/-"] = ctx["group_band"].reindex(groups_md["group"]).to_numpy()
    groups_md = groups_md[["group", "2025 value (BD m)", "% of 2025 value", "% of 2025 weight", "normal miss %",
                           "normal range +/-", *period_cols]]

    example_md = ex["table"].copy()
    total_row = pd.DataFrame([{"month": "**Total**", "actual": np.nan, "forecast": ex["forecast_sum"], "miss": ex["miss_sum"]}])
    example_md = pd.concat([example_md, total_row], ignore_index=True)
    for c in ("actual", "forecast", "miss"):
        example_md[c] = example_md[c].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}")

    route_md = ri.loc["2026-01-01":, ["core_value", "core_weight", "port_import_teu", "port_total_vessels", "air_import_t"]]
    route_md.columns = ["core goods value", "core goods weight", "containers unloaded", "ships calling", "air cargo unloaded"]
    route_md.insert(0, "month", route_md.index.strftime("%b %Y"))

    def changes_md(table, band, label):
        out = table.copy()
        out.insert(0, label, out.index)
        out["normal range +/-"] = band
        return table_md(out[[label, "% of 2025", "normal range +/-", *period_cols]], signed_cols=period_cols,
                        int_cols=["normal range +/-"], one_dp_cols=["% of 2025"])

    bulk_md = bulk_changes.copy()
    bulk_md.insert(0, "material", bulk_md.index)
    jumpy_md = ctx["jumpy"].copy()
    jumpy_md.insert(0, "HS4", jumpy_md.index)
    levels_md = ctx["levels"][["level", "series", "regular series", "% of value in regular series", "normal miss %",
                               "same-month forecast miss %", "2026 gap %"]]

    return f"""# Step 3: Exploring the data

*Generated by `src/explore.py` from the step 2 tables. Re-running the script refreshes every number.*

Before building any forecasting models, this step looks closely at the data to answer six questions:

1. Do all imports behave the same way, or should some be treated separately?
2. What does a normal year look like?
3. Which origin countries were hit in 2026, and where did goods come from instead?
4. Which kinds of goods were hit hardest?
5. Did goods stop arriving by sea and air, and if so, how did they get here?
6. How detailed can the forecasting models be?

The answers set the plan for step 4.

**What data this uses.** Imports from January 2021 to {ex['last_import']:%B %Y} (the latest month the portal has published).
Port figures run to {ex['last_port']:%B %Y} and airport figures to {ex['last_air']:%B %Y}, because those are published sooner.

---

## Before you start: how to read the numbers

### Step 1: "Normal" means an average month in 2025

To say whether a month in 2026 was unusually low, we need something to compare it with. Throughout this report that is
**the average month in 2025** for the same thing.

> Example: in 2025, core goods imports (defined in section 1) averaged **BD {avg25:,.1f}m a month**. In March 2026 they
> were BD {ex['march_2026']:,.1f}m. So March 2026 was **{signed(march_change)}%** compared with normal.

Every "+" or "-" number in the tables is worked out this way: the 2026 figure compared with the 2025 average month.

### Step 2: Why not compare with the same month last year?

Comparing March 2026 with March 2025 sounds more natural, but one month carries its own random ups and downs, and those
leak into the answer. For core goods:

| | 2026 | Same month in 2025 | Compared with same month | Compared with 2025 average month |
|---|---|---|---|---|
| June | {r6['2026']:,.0f} | {r6['2025']:,.0f} | {pct_vs(r6['2026'], r6['2025'])}% | {pct_vs(r6['2026'], avg25)}% |
| July | {r7['2026']:,.0f} | {r7['2025']:,.0f} | {pct_vs(r7['2026'], r7['2025'])}% | {pct_vs(r7['2026'], avg25)}% |

*BD million.* June 2025 happened to be a weak month, so June 2026 looks better against it. July 2025 was a strong month,
so July 2026 looks worse against it. The answer depends on whether last year's month was a good or a bad one.
We also tested both methods on 2024 and 2025, two normal years: the "same month last year" method missed by more
(section 6). So the average month is the steadier yardstick.

This is only a yardstick for exploring. In step 5 each month of 2026 is compared with a **model's forecast for that
specific month**, which can allow for growth, December peaks and Ramadan.

### Step 3: The "normal miss": how much imports wobble in a normal year

A 2026 drop only matters if it is bigger than the usual ups and downs. To measure those, we pretend to forecast a
normal year with the simple rule above and see how far off it is:

1. **Forecast:** take the average month of the previous year, and use it as the forecast for every month.
2. **Miss:** for each month, work out how far the actual value was from the forecast, ignoring whether it was above or below.
3. **Normal miss:** add up the misses and divide by the forecasts added up, giving a percentage.

Worked example, core goods in 2025 (forecast = the 2024 average month, BD million):

{md_table(example_md)}

2025 miss = {ex['miss_sum']:,.1f} / {ex['forecast_sum']:,.0f} = **{ex['miss_pct'][2025]:.1f}%**. The same calculation for 2024
gives {ex['miss_pct'][2024]:.1f}%. The **normal miss** is the average of the two years: about {core_row['normal miss %']:.0f}%.

The miss ignores whether a month is above or below the forecast, so a high month and a low month can't cancel out
and make a bad forecast look good.

### Step 4: The "normal range": how far a single month can stray

The normal miss is an *average* over months. A single month can stray further than the average even in a normal year,
so single months (and two-month periods) are compared with the **normal range** instead:

- For each month in 2023, 2024 and 2025, work out how far it was from the previous year's average month.
- The normal range is the distance that 9 months out of 10 stayed within.

> Example: if a section's normal range is +/-20%, a 2026 month that is 15% below normal is not unusual,
> but one that is 50% below is.

In the charts this range is the **grey band**. A dot outside the band is a real change, not just normal ups and downs.

### Step 5: The 2026 periods

The months of 2026 are grouped by what was happening (dates from `data/reference/events.csv`):

| Period | What was happening |
|---|---|
| **Jan-Feb** | Before the war (it began on 28 February) |
| **March** | The shock: Strait of Hormuz and Bahrain's airspace closed, Mina Salman Port struck |
| **Apr-May** | Airspace reopened on 8 April, but Hormuz stayed closed and a US naval blockade began |
| **Jun-Jul** | A brief, partial reopening of Hormuz from 18 June, then closed again on 11 July |

---

## 1. Not all imports behave the same

![Monthly imports](figures/step3_1_monthly_imports.png)

**What we did:** split imports into three groups and checked how each one behaves.

| Group | What it is |
|---|---|
| **Core goods** | Everything except the two groups below: machinery, cars, food, clothing, chemicals and so on. The everyday supply chains. |
| **Bulk materials** | Iron ore, alumina, coke and construction minerals. Huge shiploads for a few factories. |
| **Precious metals** | Gold, silver and jewellery. |

**What we found:**

{table_md(groups_md, signed_cols=period_cols, int_cols=["2025 value (BD m)", "% of 2025 value", "% of 2025 weight", "normal miss %", "normal range +/-"])}

*The last four columns compare each 2026 period with the 2025 average month (step 1).*

- **Core goods** are {core_row['% of 2025 value']:.0f}% of import value. They were {vs_normal(core_row['Jan-Feb'])} normal in Jan-Feb,
  {vs_normal(core_row['March'])} normal in March, and {vs_normal(core_row['Jun-Jul'])} normal by Jun-Jul.
- **Bulk materials** are only {bulk_row['% of 2025 value']:.0f}% of the value but {bulk_row['% of 2025 weight']:.0f}% of the weight.
  They collapsed and stayed down: {vs_normal(bulk_row['Jun-Jul'])} normal in Jun-Jul.
- **Precious metals** follow the gold price. They were already {vs_normal(prec_row['Jan-Feb'])} normal in Jan-Feb, before
  the war, and their normal miss is about {prec_row['normal miss %']:.0f}%.

**Why it matters:** mixing the three would hide what happened. A jump in gold or a single iron ore ship could make
supply chains look better or worse than they were. From here on, **core goods are the main subject**, and the other two
groups are reported separately.

---

## 2. What a normal year looks like

![Seasonal pattern](figures/step3_5_patterns.png)

**What we did:** compared each month with its own year's average (left chart), and looked at yearly totals (right chart).

**What we found:**

1. **Seasons barely matter.** Only about {s['core_explained']:.0f}% of the month-to-month movement in core goods repeats
   every year. The rest is random. The clearest pattern is December, about {s['dec_index'] - 100:.0f}% above a typical month.
2. **Ramadan makes no clear difference.** For food, a full month of Ramadan changes imports by about
   {signed(s['food_ramadan_this'])}% that month and {signed(s['food_ramadan_next'])}% the month before. That is within the normal ups and downs.
3. **Imports grow over time.** Core goods changed by {signed(gr[2023])}% in 2023, {signed(gr[2024])}% in 2024 and
   {signed(gr[2025])}% in 2025, and were still rising going into 2026 ({vs_normal(core_row['Jan-Feb'])} normal in Jan-Feb).
4. **2021 (COVID rules) was smaller but had a similar shape.** Core goods were BD {y[2021]:,.0f}m in 2021 and
   BD {y[2022]:,.0f}m in 2022. When malls and restaurants closed in May-June 2021, imports were
   {s['may_jun_2021']:.0f}% of that year's average month, against {s['may_jun_other']:.0f}% in other years, so the closures barely show.

**Why it matters:**
- Weak seasons explain why "the same month last year" is a worse yardstick than "the average month" (before-you-start, step 2).
- Growth means the models need to know the *recent* level of imports, not just last year's pattern.
- Step 4 will test the models both with and without 2021.

---

## 3. Which origins were hit, and where goods came from instead

![Origins](figures/step3_3_regions.png)

**What we did:** grouped core goods by the region they came from and compared each period of 2026 with normal.

**What we found:**

{changes_md(regions, ctx['region_band'], 'Origin region')}

*"Normal range" is explained in before-you-start, step 4: a change inside it is not unusual. "Other and not specified"
(under 1% of value) is left out.*

1. **Goods from GCC countries went up.** They were {vs_normal(gcc['Apr-May'])} normal in Apr-May and {vs_normal(gcc['Jun-Jul'])}
   normal in Jun-Jul. Their share of core goods rose from {gcc_share['2025']:.0f}% in 2025 to {gcc_share['Apr-May']:.0f}% in Apr-May.
   - Why this makes sense: GCC suppliers can send goods by road through Saudi Arabia, and UAE and Omani ports on the
     Gulf of Oman and Arabian Sea are outside the Strait of Hormuz.
   - The biggest increases in Jun-Jul (compared with Jun-Jul 2025) were {names(ctx['gcc_drivers'])}.
2. **Goods from East and South-East Asia stayed down longest.** This is the biggest origin ({asia['% of 2025']:.0f}% of core goods).
   It was {vs_normal(asia['Apr-May'])} normal in Apr-May and still {vs_normal(asia['Jun-Jul'])} normal in Jun-Jul. These goods
   normally come by container ship through Hormuz.
3. **Goods from Europe came back faster**: {vs_normal(europe['Jun-Jul'])} normal in Jun-Jul. Part of this was a few large
   aircraft-related deliveries: {names(ctx['europe_drivers'])}. Even leaving out aircraft, engines and their parts,
   Europe was {vs_normal(ctx['europe_ex_aircraft'])} normal, still well ahead of Asia.

**Why it matters:** where goods come from is our best clue to how they travel. This is the starting point for step 6 (routes).

---

## 4. Which kinds of goods were hit

![Sections](figures/step3_4_sections.png)

**What we did:** grouped core goods into product sections (the top level of the customs code) and compared each period
of 2026 with normal.

**What we found:**

{changes_md(sections, ctx['section_band'], 'Section')}

*Sections under {MIN_SECTION_SHARE}% of 2025 value, and the national special codes (confidential goods), are left out.*

1. **Every section fell in March.**
2. **By Jun-Jul most were back inside their normal range.** Food came back first: animal products, vegetable products
   and fats and oils were at or above normal.
3. **Vehicles and clothing were still well below normal** in Jun-Jul, outside their normal range.

### Bulk materials (by weight)

{table_md(bulk_md, signed_cols=period_cols, int_cols=["kt per month (2025)"])}

Where they came from (any origin with at least 10% of that material, in 2025 or in Mar-Jul 2026):

{table_md(bulk_origins, one_dp_cols=["kt per month, 2025", "kt per month, Mar-Jul 2026"])}

1. **Construction minerals show route exposure clearly.** Supplies from Oman, which have to pass through the Strait of
   Hormuz, stopped. Supplies from the UAE, inside the Gulf, went up.
2. **Alumina from Australia fell sharply**, and new suppliers appeared.
3. **Iron ore was already {vs_normal(bulk_changes.loc['Iron ore', 'Jan-Feb'])} normal in Jan-Feb**, before the war, so not all
   of its fall was caused by the war.
4. **Coke from Kuwait** doesn't need the Strait, but it also had no shipments in May-July. That may be because the factories
   needed less (Alba, the aluminium smelter, was attacked on 28 March), not just because of routes.

**Why it matters:** bulk materials mix two effects, blocked routes and factories using less, so they are kept separate
from core goods.

---

## 5. How did goods arrive?

![Routes](figures/step3_2_routes.png)

**What we did:** compared goods arriving with the port and airport figures. Each number is that month as % of the 2025 average month.

{table_md(route_md, int_cols=list(route_md.columns[1:]))}

*Port = Khalifa Bin Salman Port, Bahrain's container port. Import figures stop at {ex['last_import']:%B}; port and airport
figures are published sooner.*

**What we found:**

1. In Jun-Jul, core goods were back to **{jj['core_value']:.0f}%** of normal by value and **{jj['core_weight']:.0f}%** by weight.
2. But containers unloaded at the port were only **{jj['port_import_teu']:.0f}%** of normal, and air cargo **{jj['air_import_t']:.0f}%**.
3. So a much bigger share than usual must have arrived another way.
4. The data covers the container port and the airport, so the most likely other way is **by road over the King Fahd
   Causeway** from Saudi Arabia (smaller ports are also possible).
5. Section 3 points the same way: goods from GCC countries went up.

**Why it matters:** this is the main route finding so far. Step 6 tests it properly.

---

## 6. How detailed can the models be?

![Levels](figures/step3_6_levels.png)

**What we did:** worked out the normal miss (before-you-start, step 3) at different levels of detail, from one national total
down to every product from every country.

- A **series** is one line of monthly numbers, for example "HS2 chapter 84 (machinery) from Europe".
- A **regular** series has imports in at least 54 of the 60 months of 2021-2025.
- **Same-month forecast miss** is the normal miss if you use "the same month last year" as the forecast instead.
- **2026 gap** is the same calculation for March-July 2026.

**What we found:**

{table_md(levels_md, int_cols=["normal miss %", "same-month forecast miss %", "2026 gap %"], one_dp_cols=["% of value in regular series"])}

1. **The national total is easy to forecast** (normal miss about {lv.loc['National total', 'normal miss %']:.0f}%), and the
   2026 gap (about {lv.loc['National total', '2026 gap %']:.0f}%) is far bigger. The crisis clearly shows.
2. **The more detail, the bigger the normal miss.** Big totals hide the ups and downs of individual products, because one
   product's high month cancels another's low month. Split them up and the cancelling stops.
   - At section level, the 2026 gap is {lv.loc['Section', '2026 gap %'] / lv.loc['Section', 'normal miss %']:.1f} times the normal miss.
   - At HS4 level, it is only {lv.loc['HS4', '2026 gap %'] / lv.loc['HS4', 'normal miss %']:.1f} times.
   - For each product from each country, the normal miss ({lv.loc['HS4 x country', 'normal miss %']:.0f}%) is almost as big as the crisis itself.
3. **"The average month" beats "the same month last year" at every level**, so it is the simple forecast the models must beat.

A few products cause a lot of the ups and downs (general goods including precious metals, 2024-2025):

{table_md(jumpy_md, one_dp_cols=["% of misses", "% of value"])}

*% of misses: the share of all the simple forecast's misses that came from this product.*

- Gold and jewellery are why precious metals are kept separate.
- Aircraft engines and parts are large, occasional purchases. They stay in core goods, but step 5 will check them so that
  one delivery isn't mistaken for a recovery.

**Why it matters:** this decides how detailed the models in step 4 should be.

---

## 7. What this means for step 4

1. **Three groups, handled separately.** Core goods are the main analysis. Bulk materials (by weight) and precious metals
   are reported separately.
2. **The simple forecast to beat** is "the previous year's average month". "The same month last year" is reported too.
3. **The models forecast each HS2 chapter from each origin region** ({lv.loc['HS2 x region', 'regular series']:,} regular series
   covering {lv.loc['HS2 x region', '% of value in regular series']:.1f}% of core goods). Patchy series are pooled into one
   "other" series per section and region. Adding the forecasts up gives totals by section, by region and for the whole
   country that all agree with each other. HS4 forecasts are used as a check and for close-ups (for example cars or phones).
4. **Results are reported for the whole country, by section and by region**, where the crisis is clearly bigger than the
   normal miss. More detailed results are only shown with their uncertainty range.
5. **The models get features for the recent level and trend** (averages over the last 3, 6 and 12 months), because imports
   were growing going into 2026.
6. **2021 is tested both ways**: kept in with a COVID flag, and left out.
"""


def main() -> int:
    df = load_imports()
    route = pd.read_parquet(PROCESSED_DIR / "route_monthly.parquet").set_index("date")
    calendar = pd.read_csv(REFERENCE_DIR / "calendar.csv", parse_dates=["date"])
    core = df[df["group"] == CORE]

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
