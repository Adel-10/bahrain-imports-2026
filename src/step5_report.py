"""Charts and the written report for step 5 (called by shortfall.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import glossary
from build_dataset import md_table
from explore import (AQUA, BASELINE, BLUE, GRID, INK, INK2, MUTED, ORANGE, SHADE, SURFACE, add_axes, end_dot, end_labels,
                     legend, line_handle, month_axis, new_figure, rounded_bar, save, source_note, tidy)
from reference import period_event
from shortfall import AIRCRAFT_HS4, CLOSE_UPS, REPORT_PATH

RED = "#e34948"
# Diverging steps for the month grids: shortfalls in red, months above the forecast in blue,
# grey when the gap is inside the normal range (not a real change).
RED_STEPS = ["#f3b4b2", RED, "#a92b2b"]
BLUE_STEPS = ["#9ec5f4", BLUE, "#184f95"]
STEP_EDGES = [25, 50]                  # |gap| below 25%, 25-50%, 50% or more
MIN_SHARE = 0.5                        # sections smaller than this (% of the national forecast) are left out of tables
SKIP_SECTIONS = {"National special codes"}
WAR_START = pd.Timestamp("2026-02-28")


def pct(x: float, dp: int = 0) -> str:
    r = round(x, dp)
    return "0%" if r == 0 else f"{r:+.{dp}f}%"


def bd_m(x: float) -> str:
    return f"{x / 1e6:,.0f}"


def verdict(outside: bool, gap: float) -> str:
    if not outside:
        return "no (inside the normal range)"
    return "**yes, below normal**" if gap < 0 else "**yes, above normal**"


def and_list(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def cell_colour(gap: float, outside: bool) -> tuple[str, str]:
    """(fill, text colour) for one grid cell."""
    if not outside or np.isnan(gap):
        return SHADE, INK2
    steps = RED_STEPS if gap < 0 else BLUE_STEPS
    i = int(np.searchsorted(STEP_EDGES, abs(gap), side="right"))
    return steps[i], (INK if i == 0 else "#ffffff")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_national(results: dict) -> None:
    core = results["core"]
    monthly = results["monthly"]
    nat = monthly[monthly["level"] == "National total"].set_index("target_month")
    nat_trees = results["monthly_trees"]
    nat_trees = nat_trees[nat_trees["level"] == "National total"].set_index("target_month")
    history = core[(core["date"] >= "2025-01-01")].groupby("date")["value_bd"].sum() / 1e6
    since_label = list(results["since"])[0]
    lost = results["gaps"]
    lost = lost[(lost["level"] == "National total") & (lost["period"] == since_label)].iloc[0]

    fig = new_figure(10, 8.1, f"Core imports were about BD {lost['lost'] / 1e6:,.0f} million below normal from March to "
                              f"{pd.Timestamp(2026, results['last_month'], 1):%B}",
                     "Top: actual imports against the model's forecast of imports without the war, made at the end of "
                     "December 2025.\nThe shaded band is the forecast's normal range for a single month. Bottom: the "
                     "gap each month. Core goods, BD million.")
    legend(fig, [line_handle(BLUE), line_handle(ORANGE), Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.12, lw=0),
                 line_handle(AQUA)],
           ["Actual", "Forecast without the war (linear model)", "Normal range", "Second opinion (trees)"],
           top=1.25, ncol=4)
    ax = add_axes(fig, left=0.75, right=1.55, top=1.75, bottom=3.55)
    tidy(ax)
    x_lo = history.index.min() - pd.Timedelta(days=15)
    x_hi = history.index.max() + pd.Timedelta(days=15)
    ax.axvspan(WAR_START, x_hi, color=SHADE, lw=0, zorder=0)
    ax.text(WAR_START + pd.Timedelta(days=6), history.max() * 1.17, "War", fontsize=8.5, color=MUTED, va="top")
    fc = nat["forecast"] / 1e6
    band_lo, band_hi = fc * (1 - nat["range"]), fc * (1 + nat["range"])
    ax.fill_between(fc.index, band_lo, band_hi, color=ORANGE, alpha=0.12, lw=0, zorder=1)
    ax.plot(history.index, history, color=BLUE, zorder=4)
    ax.plot(fc.index, fc, color=ORANGE, zorder=3)
    ax.plot(nat_trees.index, nat_trees["forecast"] / 1e6, color=AQUA, lw=1.5, zorder=2)
    last = fc.index[-1]
    for series, colour in ((history, BLUE), (fc, ORANGE)):
        end_dot(ax, last, series.loc[last], colour)
    end_labels(ax, [(last, history.loc[last], f"Actual {history.loc[last]:,.0f}"),
                    (last, fc.loc[last], f"Forecast {fc.loc[last]:,.0f}")])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, history.max() * 1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    month_axis(ax, months=(1, 4, 7, 10))

    ax2 = add_axes(fig, left=0.75, right=1.55, top=5.5, bottom=0.8)
    tidy(ax2)
    gap = nat["gap"] / 1e6
    ax2.set_xlim(x_lo, x_hi)
    lim = max(abs(gap.min()), abs(gap.max())) * 1.25
    ax2.set_ylim(-lim, max(gap.max() * 1.6, lim * 0.25))
    ax2.axhline(0, color=BASELINE, lw=1, zorder=1)
    width = 18  # days
    for month, value in gap.items():
        centre = month
        colour = RED if value < 0 else BLUE
        x0 = mdates_num(centre)
        rounded_bar(ax2, 0, value, x0, width, colour, horizontal=False)
        ax2.text(centre, value + (lim * 0.05 if value >= 0 else -lim * 0.05), f"{value:+,.0f}", ha="center",
                 va="bottom" if value >= 0 else "top", fontsize=8.5, color=INK2)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}" if v else "0"))
    month_axis(ax2, months=(1, 4, 7, 10))
    ax2.set_title("Gap: actual minus forecast (BD million)", loc="left", fontsize=10, color=INK2, pad=8)
    source_note(fig)
    save(fig, "step5_1_national.png")


def mdates_num(ts: pd.Timestamp) -> float:
    import matplotlib.dates as mdates
    return mdates.date2num(ts)


def grid_chart(rows: list[tuple[str, dict]], columns: list[str], title: str, subtitle: str, name: str,
               first_col_width: float = 3.3, bold_columns: set[str] | None = None, cell_width: float = 0.78) -> None:
    """A table-like chart: one row per total, one column per month (and period), coloured only when outside the range.

    rows: (label, {column: (gap_pct, outside)}). Column headers in `bold_columns` are bold (by default, any
    starting "Mar": the "since the war" column).
    """
    n_rows, n_cols = len(rows), len(columns)
    cell_w, cell_h = cell_width, 0.36
    width = first_col_width + n_cols * cell_w + 0.5
    height = 1.95 + n_rows * cell_h + 0.85
    fig = new_figure(width, height, title, subtitle)
    w, h = fig.get_size_inches()
    # Colour key
    key_y = 1 - 1.3 / h
    x = 0.6 / w
    items = [(RED_STEPS[0], "below by up to 25%"), (RED_STEPS[1], "25-50%"), (RED_STEPS[2], "50% or more"),
             (BLUE_STEPS[1], "above normal"), (SHADE, "inside the normal range")]
    for colour, label in items:
        fig.patches.append(Rectangle((x, key_y - 0.06 / h), 0.22 / w, 0.16 / h, transform=fig.transFigure,
                                     color=colour, lw=0))
        fig.text(x + 0.3 / w, key_y + 0.02 / h, label, fontsize=9, color=INK2, va="center")
        x += (0.3 + 0.075 * len(label) + 0.35) / w
    ax = fig.add_axes([first_col_width / w, 0.75 / h, n_cols * cell_w / w, n_rows * cell_h / h])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.axis("off")
    gap_px = 2 / (cell_w * fig.dpi)            # a 2px surface gap between cells
    gap_py = 2 / (cell_h * fig.dpi)
    for i, (label, cells) in enumerate(rows):
        bold = label.startswith("**")
        clean = label.strip("*")
        fig.text((first_col_width - 0.12) / w, (0.75 + (n_rows - i - 0.5) * cell_h) / h, clean, ha="right",
                 va="center", fontsize=9.5, color=INK, fontweight="semibold" if bold else "normal")
        for j, col in enumerate(columns):
            gap, outside = cells.get(col, (np.nan, False))
            fill, text = cell_colour(gap, outside)
            ax.add_patch(Rectangle((j + gap_px, i + gap_py), 1 - 2 * gap_px, 1 - 2 * gap_py, color=fill, lw=0))
            if not np.isnan(gap):
                ax.text(j + 0.5, i + 0.52, f"{gap:+.0f}" if round(gap) else "0", ha="center", va="center",
                        fontsize=8.5, color=text)
    for j, col in enumerate(columns):
        fig.text((first_col_width + (j + 0.5) * cell_w) / w, (0.75 + n_rows * cell_h + 0.12) / h, col, ha="center",
                 va="bottom", fontsize=9, color=INK2,
                 fontweight="semibold" if (col in bold_columns if bold_columns is not None else col.startswith("Mar"))
                 else "normal")
    source_note(fig)
    save(fig, name)


def grid_rows(monthly: pd.DataFrame, gaps: pd.DataFrame, level: str, labels: list[str], since_label: str,
              columns: list[str], national: bool = True) -> list:
    rows = []
    month_cols = columns[:-1]
    keep = [("National total", "Bahrain", "**All of Bahrain**")] if national else []
    keep += [(level, l, l) for l in labels]
    for lvl, label, shown in keep:
        cells = {}
        m = monthly[(monthly["level"] == lvl) & (monthly["label"] == label)]
        for _, r in m.iterrows():
            cells[f"{r['target_month']:%b}"] = (r["gap_pct"], bool(r["outside"]))
        g = gaps[(gaps["level"] == lvl) & (gaps["label"] == label) & (gaps["period"] == since_label)].iloc[0]
        cells[columns[-1]] = (g["gap_pct"], bool(g["outside"]))
        rows.append((shown, cells))
    return rows


def fig_second_opinion(gaps: pd.DataFrame, gaps_trees: pd.DataFrame, since_label: str, labels: dict) -> None:
    a = gaps[gaps["period"] == since_label].set_index(["level", "label"])["gap_pct"]
    b = gaps_trees[gaps_trees["period"] == since_label].set_index(["level", "label"])["gap_pct"]
    both = pd.DataFrame({"linear": a, "trees": b}).dropna()
    both = both[[(lvl in ("Origin region", "Section") and lab in labels.get(lvl, [])) or lvl == "National total"
                 for lvl, lab in both.index]]
    months_text = since_label.split("(")[-1].rstrip(")").replace("Mar-", "March to ")
    fig = new_figure(10, 6.2, "The second opinion agrees: the trees give almost the same gaps",
                     f"Gap from {months_text}, % of each model's forecast. Each dot is one total (the whole country, an "
                     "origin region or a product section);\ndots on the diagonal mean both models agree exactly.")
    legend(fig, [plt.Line2D([], [], ls="", marker="o", ms=8, color=c, mec=SURFACE, mew=2) for c in (BLUE, ORANGE, INK)],
           ["Origin regions", "Product sections", "All of Bahrain"], top=1.15, ncol=3)
    ax = add_axes(fig, left=0.9, right=0.5, top=1.6, bottom=0.8)
    lo = min(both.min().min(), -10) - 5
    hi = max(both.max().max(), 10) + 5
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    tidy(ax)
    ax.grid(True, axis="both", zorder=0)
    ax.plot([lo, hi], [lo, hi], color=BASELINE, lw=1, zorder=1)
    ax.axhline(0, color=BASELINE, lw=1, zorder=1)
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    colours = {"Origin region": BLUE, "Section": ORANGE, "National total": INK}
    for (lvl, lab), r in both.iterrows():
        ax.plot([r["linear"]], [r["trees"]], "o", ms=8, color=colours[lvl], mec=SURFACE, mew=2,
                zorder=4 if lvl == "National total" else 3)
        if lvl == "National total":
            ax.annotate("All of Bahrain", (r["linear"], r["trees"]), xytext=(-60, 30), textcoords="offset points",
                        fontsize=9, color=INK, ha="right",
                        arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8, shrinkA=2, shrinkB=6))
    ax.set_xlabel("Linear model (main forecast)", color=INK2)
    ax.set_ylabel("Gradient-boosted trees (second opinion)", color=INK2)
    fmt = FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    source_note(fig)
    save(fig, "step5_4_second_opinion.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write(results: dict) -> None:
    gaps, gaps_trees, monthly = results["gaps"], results["gaps_trees"], results["monthly"]
    periods, since = results["periods"], results["since"]
    since_label = list(since)[0]
    last_month = results["last_month"]
    month_cols = [f"{pd.Timestamp(2026, m, 1):%b}" for m in range(1, last_month + 1)]
    since_short = f"Mar-{pd.Timestamp(2026, last_month, 1):%b}"
    grid_cols = month_cols + [since_short]
    war_periods = [p for p in periods if p != "Jan-Feb"]

    nat = gaps[gaps["level"] == "National total"].set_index("period")
    nat_m = monthly[monthly["level"] == "National total"].set_index("target_month")
    nat_trees = gaps_trees[gaps_trees["level"] == "National total"].set_index("period")
    total = nat.loc[since_label]

    # Which sections and regions to show
    nat_fc = total["forecast"]
    sec = gaps[(gaps["level"] == "Section") & (gaps["period"] == since_label)].copy()
    sec = sec[(sec["forecast"] / nat_fc * 100 >= MIN_SHARE) & ~sec["label"].isin(SKIP_SECTIONS)]
    sec = sec.sort_values("gap_pct")
    reg = gaps[(gaps["level"] == "Origin region") & (gaps["period"] == since_label)].copy()
    reg = reg[reg["label"] != "Other and not specified"].sort_values("gap_pct")
    labels = {"Section": list(sec["label"]), "Origin region": list(reg["label"])}

    # Charts
    fig_national(results)
    grid_chart(grid_rows(monthly, gaps, "Section", labels["Section"], since_label, grid_cols), grid_cols,
               f"Almost every product section fell; {sec.iloc[0]['label'].lower()} fell furthest",
               "Gap between actual imports and the forecast without the war, % of the forecast, for each month of 2026\n"
               f"and for {since_short} together. Coloured only where the gap is outside the normal range (a real change);\n"
               "grey where it could be normal ups and downs. Sorted by the overall gap.",
               "step5_2_sections.png")
    grid_chart(grid_rows(monthly, gaps, "Origin region", labels["Origin region"], since_label, grid_cols), grid_cols,
               "Goods from the Americas and East Asia fell furthest; goods from GCC countries held up",
               "Gap between actual imports and the forecast without the war, % of the forecast. Coloured only where "
               "the gap is outside the normal range.",
               "step5_3_regions.png")
    fig_second_opinion(gaps, gaps_trees, since_label, labels)

    cu = results["closeups"]["table"]
    cu_rows = []
    for code, product in CLOSE_UPS.items():
        cells = {}
        part = cu[cu["hs4"] == code]
        for _, r in part.iterrows():
            key = f"{pd.Timestamp(2026, int(r['period']), 1):%b}" if isinstance(r["period"], (int, np.integer)) else r["period"]
            if key == since_label:
                key = since_short
            cells[key] = (r["gap_pct"], bool(r["outside"]))
        cu_rows.append((product, cells))
    grid_chart(cu_rows, grid_cols, "Cars stayed far below normal, while rice and poultry bounced back",
               "Six everyday products (HS4 level, all origins): gap between actual imports and the forecast without "
               "the war, % of the forecast.\nColoured only where the gap is outside that product's normal range, "
               "which is wide for single products.", "step5_5_products.png", first_col_width=2.9)

    # --- Tables --------------------------------------------------------------
    nat_ranges = results["ranges"][results["ranges"]["level"] == "National total"].set_index("months")["range"] * 100
    ranges_md = md_table(pd.DataFrame({
        "Length of the period": [f"{k} month{'s' if k > 1 else ''}" for k in nat_ranges.index],
        "Normal range for the national total": [f"+/-{v:.0f}%" for v in nat_ranges],
    }))
    periods_md = md_table(pd.DataFrame({
        "Period": [f"**{p}**" for p in periods],
        "What was happening": [period_event(p) for p in periods],
    }))

    def nat_period_rows(labels_):
        return md_table(pd.DataFrame({
            "Period": labels_,
            "Forecast (BD m)": [bd_m(nat.loc[p, "forecast"]) for p in labels_],
            "Actual (BD m)": [bd_m(nat.loc[p, "actual"]) for p in labels_],
            "Gap": [pct(nat.loc[p, "gap_pct"]) for p in labels_],
            "Normal range": [f"+/-{nat.loc[p, 'range_pct']:.0f}%" for p in labels_],
            "Real shortfall?": [verdict(nat.loc[p, "outside"], nat.loc[p, "gap_pct"]) for p in labels_],
        }))

    monthly_md = md_table(pd.DataFrame({
        "Month": [f"{m:%B}" for m in nat_m.index],
        "Forecast (BD m)": [bd_m(v) for v in nat_m["forecast"]],
        "Actual (BD m)": [bd_m(v) for v in nat_m["actual"]],
        "Gap (BD m)": [f"{v / 1e6:+,.0f}" for v in nat_m["gap"]],
        "Gap": [pct(v) for v in nat_m["gap_pct"]],
        "Normal range": [f"+/-{v:.0f}%" for v in nat_m["range_pct"]],
        "Outside the range?": ["yes" if o else "no" for o in nat_m["outside"]],
    }))

    lost_md = md_table(pd.DataFrame({
        "Period": [*war_periods, f"**{since_label}**"],
        "Lost imports (BD m)": [bd_m(nat.loc[p, "lost"]) for p in war_periods] + [f"**{bd_m(total['lost'])}**"],
        "Likely between (BD m)": [f"{bd_m(nat.loc[p, 'lost_low'])} and {bd_m(nat.loc[p, 'lost_high'])}" for p in war_periods]
                                 + [f"**{bd_m(total['lost_low'])} and {bd_m(total['lost_high'])}**"],
    }))

    def ranking_md(df, label):
        out = pd.DataFrame({
            label: df["label"],
            "Forecast (BD m)": [bd_m(v) for v in df["forecast"]],
            "Lost (BD m)": [f"{v / 1e6:,.0f}" for v in df["lost"]],
            "Gap": [pct(v) for v in df["gap_pct"]],
            "Normal range": [f"+/-{v:.0f}%" for v in df["range_pct"]],
            "Real shortfall?": [verdict(o, g) for o, g in zip(df["outside"], df["gap_pct"])],
        })
        return md_table(out)

    by_bd = sec.sort_values("lost", ascending=False)
    top2_bd = by_bd.head(2)
    top2_share = top2_bd["lost"].sum() / total["lost"] * 100
    sec_real = sec[sec["outside"] & (sec["gap_pct"] < 0)]
    sec_normal = sec[~sec["outside"]]

    def row(df, label):
        return df[df["label"] == label].iloc[0]

    reg_gcc = row(reg, "GCC")
    early = [p for p in ("March", "Apr-May") if p in nat.index]
    early_share = sum(nat.loc[p, "lost"] for p in early) / total["lost"] * 100

    def month_gap(level, label, month):
        m = monthly[(monthly["level"] == level) & (monthly["label"] == label) & (monthly["target_month"].dt.month == month)]
        return m.iloc[0] if not m.empty else None

    food = ["Animal products", "Vegetable products", "Food, drinks and tobacco", "Fats and oils"]
    food_parts = []
    for f in food:
        mar, jul = month_gap("Section", f, 3), month_gap("Section", f, last_month)
        if mar is not None and jul is not None:
            food_parts.append(f"{f.lower()} went from {pct(mar['gap_pct'])} in March to {pct(jul['gap_pct'])} in "
                              f"{pd.Timestamp(2026, last_month, 1):%B}")
    food_sentence = "; ".join(food_parts[:3]) + "."
    food_sentence = food_sentence[0].upper() + food_sentence[1:]
    eu_last = month_gap("Origin region", "Europe", last_month)
    as_last = month_gap("Origin region", "East and South-East Asia", last_month)
    europe_sentence = (f"by {pd.Timestamp(2026, last_month, 1):%B} it was {pct(eu_last['gap_pct'])}, against "
                       f"{pct(as_last['gap_pct'])} for East and South-East Asia.")
    asia = row(reg, "East and South-East Asia")
    americas = row(reg, "Americas")
    europe = row(reg, "Europe")

    # Second opinion
    a = gaps[gaps["period"] == since_label].set_index(["level", "label"])
    b = gaps_trees[gaps_trees["period"] == since_label].set_index(["level", "label"])
    shown = [("National total", "Bahrain")] + [("Section", l) for l in labels["Section"]] + \
            [("Origin region", l) for l in labels["Origin region"]]
    diff = (a.loc[shown, "gap_pct"] - b.loc[shown, "gap_pct"]).abs()
    same_verdict = (a.loc[shown, "outside"] == b.loc[shown, "outside"])
    disagree = [(lvl, lab) for (lvl, lab), ok in same_verdict.items() if not ok]
    if disagree:
        parts = [f"{lab.lower() if lvl == 'Section' else lab} (linear {pct(a.loc[(lvl, lab), 'gap_pct'])} against a range of +/-{a.loc[(lvl, lab), 'range_pct']:.0f}%; "
                 f"trees {pct(b.loc[(lvl, lab), 'gap_pct'])} against +/-{b.loc[(lvl, lab), 'range_pct']:.0f}%)" for lvl, lab in disagree]
        disagree_text = (f" They disagree only for {and_list(parts)}: borderline cases whose gaps sit right at the edge of "
                         "the normal range.")
    else:
        disagree_text = ""

    # Aircraft
    air = results["aircraft"]
    air_md = md_table(pd.DataFrame({
        "Month": [f"{m:%B}" for m in air.index],
        "Aircraft, engines and parts (BD m)": [f"{v / 1e6:,.1f}" for v in air["aircraft"]],
        "Their normal month (BD m)": [f"{v / 1e6:,.1f}" for v in air["aircraft_normal"]],
        "Gap, as measured": [pct(v) for v in air["gap_pct"]],
        "Gap if they had been normal": [pct(v) for v in air["gap_pct_without_aircraft_swings"]],
    }))
    march = air.loc[pd.Timestamp("2026-03-01")]

    # Bulk materials and precious metals
    sg = results["simple_groups"]
    bulk = sg["bulk"]
    bulk_md = md_table(pd.DataFrame([
        {"Material": n, **{p: (f"**{pct(g.iloc[0]['gap_pct'])}**" if g.iloc[0]["outside"] else pct(g.iloc[0]["gap_pct"]))
                          for p in [*war_periods, since_label]
                          for g in [bulk[(bulk["name"] == n) & (bulk["period"] == p)]] if not g.empty}}
        for n in ["All bulk materials", "Iron ore", "Alumina", "Petroleum and pitch coke", "Construction minerals"]
    ]))
    prec = sg["precious"].set_index("period")
    prec_md = md_table(pd.DataFrame({
        "Period": [*periods, since_label],
        "Gap against the 2025 average month": [pct(prec.loc[p, "gap_pct"]) for p in [*periods, since_label]],
        "Normal range": [f"+/-{prec.loc[p, 'range_pct']:.0f}%" for p in [*periods, since_label]],
        "Outside?": ["yes" if prec.loc[p, "outside"] else "no" for p in [*periods, since_label]],
    }))

    # Close-ups
    misses = results["closeups"]["misses"]
    cu_since = cu[cu["period"] == since_label].set_index("product")
    cu_jj = cu[cu["period"] == "Jun-Jul"].set_index("product") if "Jun-Jul" in set(cu["period"]) else None

    def cu_val(product, period):
        r = cu[(cu["product"] == product) & (cu["period"] == period)].iloc[0]
        return pct(r["gap_pct"])

    text = f"""# Step 5: How far below normal were imports in 2026?

*Generated by `src/shortfall.py` (charts and text by `src/step5_report.py`). Re-running the script refreshes every number.*

This is the main result of the project. Step 4 built a model that forecasts what imports would have been **without the war**,
and tested it on normal years. Here, each month of 2026 is compared with that forecast to answer three questions:

1. How far below normal were imports, and for how long?
2. How much was lost in total?
3. Which supply chains (product sections and origins) were hit hardest?

## The short answer

| Question | Answer |
|---|---|
| How far below normal? | **{pct(nat.loc['March', 'gap_pct'])} in March**, {pct(nat.loc['Apr-May', 'gap_pct'])} in April-May and {pct(nat.loc['Jun-Jul', 'gap_pct'])} in June-July (core goods). Every one of these is a real shortfall, well outside the normal range. |
| Did it get better? | **Yes, steadily.** By July imports were {pct(nat_m['gap_pct'].iloc[-1])} against the forecast, which on its own is within a normal month's ups and downs. |
| How much was lost? | **About BD {bd_m(total['lost'])} million** of core goods from March to {pd.Timestamp(2026, last_month, 1):%B} ({pct(total['gap_pct'])}); likely between BD {bd_m(total['lost_low'])} and {bd_m(total['lost_high'])} million. |
| Which products were hit hardest? | **{sec.iloc[0]['label']}** fell furthest ({pct(sec.iloc[0]['gap_pct'])}). **{top2_bd.iloc[0]['label']}** and **{top2_bd.iloc[1]['label'].lower()}** account for {top2_share:.0f}% of the money lost. |
| Which origins? | **The Americas ({pct(americas['gap_pct'])}) and East and South-East Asia ({pct(asia['gap_pct'])})**, which normally ship through the Strait of Hormuz. Goods from **GCC countries held up** ({pct(reg_gcc['gap_pct'])}, inside the normal range). |
| Do the trees agree? | **Yes.** The second opinion gives the national gap as {pct(nat_trees.loc[since_label, 'gap_pct'])} (against {pct(total['gap_pct'])}), and every section and region gap within {diff.max():.0f} percentage points. |

## Words used in this report

{glossary.table(5, ["core_goods", "national_total", "origin_region", "section", "hs_codes", "periods", "since_war",
                    "forecast", "normal", "gap", "normal_year", "normal_range", "real_shortfall", "lost",
                    "second_opinion", "units"])}

---

## Before you start: how the comparison works

### Step 1: The forecast is "imports without the war"

Step 4's linear model stood at the end of December 2025 and forecast every month of 2026, using only what was known then.
It was right for January and February (within 4%), before the war began on 28 February. So for March onwards, its
forecast is our best estimate of what would have arrived without the war.

### Step 2: The gap

For every month and every total (the whole country, each origin region, each product section):

> **gap = actual imports - forecast**, also shown as % of the forecast.

### Step 3: Is the gap real?

The model is never exactly right, even in normal years. A gap only counts as a **real shortfall** if it is bigger than the
**normal range** for that total.

Longer periods have **narrower** ranges, because over several months the model's ups and downs partly cancel out. For the
national total:

{ranges_md}

*From step 4's tests on 2024 and 2025: for every run of months of that length, how far the model's total was from the
actual total, 9 times out of 10.*

### Step 4: The periods

{periods_md}

---

## 1. The whole country, month by month

![National](figures/step5_1_national.png)

**What we found:**

{monthly_md}

1. **January and February were normal**, slightly above the forecast.
2. **March was the worst month:** imports were less than half the forecast ({pct(nat_m.loc[pd.Timestamp('2026-03-01'), 'gap_pct'])}).
3. **Each month after that was a little better.** By July the gap was {pct(nat_m['gap_pct'].iloc[-1])}, inside the
   normal range for a single month, so July on its own could be ordinary ups and downs.

By period:

{nat_period_rows([*periods, since_label])}

**Why it matters:** the fall is far too large to be noise. Even the smallest war-time gap (June-July, {pct(nat.loc['Jun-Jul', 'gap_pct'])})
is {abs(nat.loc['Jun-Jul', 'gap_pct']) / nat.loc['Jun-Jul', 'range_pct']:.1f} times the normal range for two months.

---

## 2. How much was lost in total?

**What we did:** added up the gap from March to {pd.Timestamp(2026, last_month, 1):%B}. The "likely between" figures use the
normal range: what would have arrived without the war could have been a little above or below the forecast.

{lost_md}

**What we found:** about **BD {bd_m(total['lost'])} million** of core goods didn't arrive between March and
{pd.Timestamp(2026, last_month, 1):%B}: {abs(total['gap_pct']):.0f}% of what was expected. {early_share:.0f}% of it was lost in March, April and May.

---

## 3. Which kinds of goods were hit hardest?

![Sections](figures/step5_2_sections.png)

**What we did:** compared each product section with its own forecast, month by month and for {since_short} together.

**What we found ({since_short} together):**

{ranking_md(sec, 'Section')}

*Sections under {MIN_SHARE}% of the national forecast, and national special codes, are left out.*

1. **{len(sec_real)} of {len(sec)} sections had a real shortfall.** Only {and_list(sec_normal['label'].str.lower())}
   stayed inside their normal range.
2. **{sec.iloc[0]['label']} fell furthest** ({pct(sec.iloc[0]['gap_pct'])}), followed by
   {and_list(f"{l.lower()} ({pct(g)})" for l, g in zip(sec['label'].iloc[1:5], sec['gap_pct'].iloc[1:5]))}.
3. **Most of the money was lost in two sections:** {top2_bd.iloc[0]['label'].lower()} (BD {bd_m(top2_bd.iloc[0]['lost'])}m)
   and {top2_bd.iloc[1]['label'].lower()} (BD {bd_m(top2_bd.iloc[1]['lost'])}m), {top2_share:.0f}% of the total.
4. **Food was hit early, then recovered.** {food_sentence}

**Why it matters:** the goods that fell furthest (vehicles, machinery, clothing) are mostly made in Asia, Europe and the
Americas and shipped in containers or on car carriers. Food can more easily be bought from nearby countries. Step 6 tests this.

---

## 4. Which origins were hit hardest?

![Regions](figures/step5_3_regions.png)

**What we found ({since_short} together):**

{ranking_md(reg, 'Origin region')}

1. **The Americas ({pct(americas['gap_pct'])}) and East and South-East Asia ({pct(asia['gap_pct'])}) fell furthest.**
   Asia is the biggest origin, so it also lost the most money (BD {bd_m(asia['lost'])}m).
2. **Europe fell less ({pct(europe['gap_pct'])})**: {europe_sentence}
3. **GCC countries held up:** {pct(reg_gcc['gap_pct'])}, inside the normal range. They can reach Bahrain by road, or by sea
   without passing through the Strait of Hormuz.

**Why it matters:** this is the clearest sign yet that the route mattered more than the product. Step 6 looks at it directly.

---

## 5. Second opinion: do the trees agree?

![Second opinion](figures/step5_4_second_opinion.png)

**What we did:** repeated everything with step 4's best gradient-boosted trees, each judged against its own normal ranges.

**What we found:**

- **Nationally:** {pct(nat_trees.loc[since_label, 'gap_pct'])} with the trees, against {pct(total['gap_pct'])} with the linear model.
- **Sections and regions:** the two models' gaps differ by {diff.mean():.1f} percentage points on average, and by at most
  {diff.max():.1f}.
- **Real shortfall or not:** the two models agree for {int(same_verdict.sum())} of {len(same_verdict)} totals.{disagree_text}

**Why it matters:** two very different kinds of model tell the same story, so the findings don't depend on the choice of model.

---

## 6. Close-ups: six everyday products

![Products](figures/step5_5_products.png)

**What we did:** rebuilt the same linear model at HS4 level, where each series is one product from all origins together, and
compared six everyday products with their forecasts.

*At this level of detail the model barely beats the simple rule (it missed by {misses[(2025, 'linear')]:.1f}% against
{misses[(2025, 'simple')]:.1f}% in the 2025 test), so the normal ranges are wide and these are illustrations rather than headline results.*

**What we found:**

1. **Cars fell hardest and stayed down:** {cu_val('Cars', 'March')} in March and still {cu_val('Cars', 'Jun-Jul')} in June-July.
   Cars normally arrive on car-carrier ships through the Strait of Hormuz.
2. **Phones and computers** were well below normal throughout ({cu_val('Phones and network equipment', since_label)} and
   {cu_val('Computers', since_label)} from March to {pd.Timestamp(2026, last_month, 1):%B}).
3. **Rice and poultry bounced back.** Rice was {cu_val('Rice', 'March')} in March but {cu_val('Rice', 'Jun-Jul')}
   in June-July, above normal, which looks like restocking. Poultry went from {cu_val('Poultry meat', 'March')} to
   {cu_val('Poultry meat', 'Jun-Jul')}.
4. **Medicines** dipped in March ({cu_val('Medicines', 'March')}) and then came in above normal in April-May
   ({cu_val('Medicines', 'Apr-May')}), which looks like catching up or building stocks.

---

## 7. Checking the aircraft effect

Step 3 found that aircraft, engines and parts are large, irregular purchases that can make a month look better or worse than
everyday supply chains really were. Here is the national gap with and without their swings:

{air_md}

- **In March, they made the drop look deeper.** Almost none arrived (BD {march['aircraft'] / 1e6:,.1f}m against a normal
  BD {march['aircraft_normal'] / 1e6:,.1f}m). Without that swing, March would have been {pct(march['gap_pct_without_aircraft_swings'])}
  instead of {pct(march['gap_pct'])}.
- **From May onwards they were close to normal**, so they don't explain the later months or the recovery.

---

## 8. Bulk materials and precious metals

These groups have too few series for the models, so they are compared with the simple rule (the 2025 average month), using
the normal ranges from step 3. Bold means outside the normal range.

**Bulk materials, by weight:**

{bulk_md}

- **Iron ore almost stopped** and alumina fell by about half. Both come by sea through the Strait of Hormuz, from Brazil
  and Australia.
- **Construction minerals didn't change:** they come mainly from the UAE, inside the Gulf.

**Precious metals, by value:**

{prec_md}

Apart from March, precious metals stayed within their (very wide) normal range: they follow the gold price more than supply routes.

---

## What this means for step 6

1. **Core goods were about BD {bd_m(total['lost'])} million below normal from March to {pd.Timestamp(2026, last_month, 1):%B}**,
   with the worst in March and a steady recovery after.
2. **Origin mattered a lot:** goods that normally pass through the Strait of Hormuz fell furthest, while GCC goods held up.
3. **Step 6** will test the route explanation directly: grouping goods by how they normally travel, and comparing with the
   port, airport, RoRo and general cargo figures.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
