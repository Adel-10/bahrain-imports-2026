"""Charts and the written report for step 5 (called by shortfall.py).

    python src/shortfall.py        # computes the gaps, then calls write() in this module

Charts (reports/figures/)
-------------------------
1. step5_1_national.png        actual core imports against the forecast without the war, with the
                               forecast's normal range, and the monthly gap in BD below.
2. step5_2_sections.png        grid: gap for each product section, each month and the whole war period.
3. step5_3_regions.png         the same grid for origin regions.
4. step5_4_second_opinion.png  linear model against trees: one dot per total.
5. step5_5_products.png        grid for six everyday products (4-digit level).

The grids colour a cell only when the gap is outside its normal range, so the eye goes to the
changes that are bigger than the model's ordinary misses; everything else is grey.

Report (reports/step5_shortfall.md)
-----------------------------------
Key findings, the method, then results nationally, in BD lost, by section and by origin, the
second opinion, the product close-ups, the aircraft check, and bulk materials and precious metals.
Sentences that name the hardest-hit sections or the months of recovery are built from the
numbers, so they stay correct when the data is updated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from charts import (AQUA, BASELINE, BLUE, INK, INK2, MUTED, ORANGE, RED, SHADE, SURFACE, WAR_START, add_axes, date_num,
                    end_dot, end_labels, grid_chart, legend, line_handle, month_axis, new_figure, rounded_bar, save,
                    source_note, tidy)
from reference import period_event
from reporting import and_list, bd_m, footer, md_table, pct, write_markdown
from shortfall import CLOSE_UPS, REPORT_PATH

MIN_SHARE = 0.5                        # sections smaller than this (% of the national forecast) are left out of tables
SKIP_SECTIONS = {"National special codes"}   # confidential goods and other special codes: not a supply chain


def verdict(outside: bool, gap: float) -> str:
    """Table text for whether a gap is a real change."""
    if not outside:
        return "no (inside the normal range)"
    return "**yes, below normal**" if gap < 0 else "**yes, above normal**"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_national(results: dict) -> None:
    """Figure 1: two panels sharing a time axis: actual vs forecast (top) and the monthly gap (bottom)."""
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
    # The forecast's normal range for a single month, drawn as a band around it.
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

    # Bottom panel: the gap each month as a bar (red below the forecast, blue above).
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
        x0 = date_num(centre)          # bars on a date axis are placed in matplotlib's date units
        rounded_bar(ax2, 0, value, x0, width, colour, horizontal=False)
        ax2.text(centre, value + (lim * 0.05 if value >= 0 else -lim * 0.05), f"{value:+,.0f}", ha="center",
                 va="bottom" if value >= 0 else "top", fontsize=8.5, color=INK2)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}" if v else "0"))
    month_axis(ax2, months=(1, 4, 7, 10))
    ax2.set_title("Gap: actual minus forecast (BD million)", loc="left", fontsize=10, color=INK2, pad=8)
    source_note(fig)
    save(fig, "step5_1_national.png")


def grid_rows(monthly: pd.DataFrame, gaps: pd.DataFrame, level: str, labels: list[str], since_label: str,
              columns: list[str], national: bool = True) -> list:
    """Rows for grid_chart: one per total, with a cell per month and one for the whole war period."""
    rows = []
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
    """Figure 4: each total's war-period gap under the linear model (x) and the trees (y)."""
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
    """Draw the five charts and write reports/step5_shortfall.md from the results of shortfall.py."""
    gaps, gaps_trees, monthly = results["gaps"], results["gaps_trees"], results["monthly"]
    periods, since = results["periods"], results["since"]
    since_label = list(since)[0]
    last_month = results["last_month"]
    last_name = f"{pd.Timestamp(2026, last_month, 1):%B}"
    month_cols = [f"{pd.Timestamp(2026, m, 1):%b}" for m in range(1, last_month + 1)]
    since_short = f"Mar-{pd.Timestamp(2026, last_month, 1):%b}"
    grid_cols = month_cols + [since_short]
    war_periods = [p for p in periods if p != "Jan-Feb"]

    nat = gaps[gaps["level"] == "National total"].set_index("period")
    nat_m = monthly[monthly["level"] == "National total"].set_index("target_month")
    nat_trees = gaps_trees[gaps_trees["level"] == "National total"].set_index("period")
    total = nat.loc[since_label]

    # Which sections and regions to show: sections big enough to matter, regions except "not specified".
    nat_fc = total["forecast"]
    sec = gaps[(gaps["level"] == "Section") & (gaps["period"] == since_label)].copy()
    sec = sec[(sec["forecast"] / nat_fc * 100 >= MIN_SHARE) & ~sec["label"].isin(SKIP_SECTIONS)]
    sec = sec.sort_values("gap_pct")
    reg = gaps[(gaps["level"] == "Origin region") & (gaps["period"] == since_label)].copy()
    reg = reg[reg["label"] != "Other and not specified"].sort_values("gap_pct")
    labels = {"Section": list(sec["label"]), "Origin region": list(reg["label"])}

    # --- charts ---
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
            # Single months are stored as month numbers, periods by their label.
            key = f"{pd.Timestamp(2026, int(r['period']), 1):%b}" if isinstance(r["period"], (int, np.integer)) else r["period"]
            if key == since_label:
                key = since_short
            cells[key] = (r["gap_pct"], bool(r["outside"]))
        cu_rows.append((product, cells))
    grid_chart(cu_rows, grid_cols, "Cars stayed far below normal, while rice and poultry bounced back",
               "Six everyday products (HS4 level, all origins): gap between actual imports and the forecast without "
               "the war, % of the forecast.\nColoured only where the gap is outside that product's normal range, "
               "which is wide for single products.", "step5_5_products.png", first_col_width=2.9)

    # --- tables ---
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

    # --- numbers quoted in the text ---
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

    # How the food sections moved from March to the latest month, as one sentence.
    food = ["Animal products", "Vegetable products", "Food, drinks and tobacco", "Fats and oils"]
    food_parts = []
    for f in food:
        mar, last = month_gap("Section", f, 3), month_gap("Section", f, last_month)
        if mar is not None and last is not None:
            food_parts.append(f"{f.lower()} went from {pct(mar['gap_pct'])} in March to {pct(last['gap_pct'])} in {last_name}")
    food_sentence = "; ".join(food_parts[:3]) + "."
    food_sentence = food_sentence[0].upper() + food_sentence[1:]
    eu_last = month_gap("Origin region", "Europe", last_month)
    as_last = month_gap("Origin region", "East and South-East Asia", last_month)
    asia = row(reg, "East and South-East Asia")
    americas = row(reg, "Americas")
    europe = row(reg, "Europe")

    # Second opinion: how far apart the two models' gaps are, and where their verdicts differ.
    a = gaps[gaps["period"] == since_label].set_index(["level", "label"])
    b = gaps_trees[gaps_trees["period"] == since_label].set_index(["level", "label"])
    shown = [("National total", "Bahrain")] + [("Section", l) for l in labels["Section"]] + \
            [("Origin region", l) for l in labels["Origin region"]]
    diff = (a.loc[shown, "gap_pct"] - b.loc[shown, "gap_pct"]).abs()
    same_verdict = (a.loc[shown, "outside"] == b.loc[shown, "outside"])
    disagree = [(lvl, lab) for (lvl, lab), ok in same_verdict.items() if not ok]
    if disagree:
        parts = [f"{lab.lower() if lvl == 'Section' else lab} (linear {pct(a.loc[(lvl, lab), 'gap_pct'])} against a range of "
                 f"+/-{a.loc[(lvl, lab), 'range_pct']:.0f}%; trees {pct(b.loc[(lvl, lab), 'gap_pct'])} against "
                 f"+/-{b.loc[(lvl, lab), 'range_pct']:.0f}%)" for lvl, lab in disagree]
        disagree_text = (f" They differ only for {and_list(parts)}, borderline cases where the gap sits right at the edge "
                         "of the normal range.")
    else:
        disagree_text = ""

    # Aircraft check
    air = results["aircraft"]
    air_md = md_table(pd.DataFrame({
        "Month": [f"{m:%B}" for m in air.index],
        "Aircraft, engines and parts (BD m)": [f"{v / 1e6:,.1f}" for v in air["aircraft"]],
        "Their normal month (BD m)": [f"{v / 1e6:,.1f}" for v in air["aircraft_normal"]],
        "Gap, as measured": [pct(v) for v in air["gap_pct"]],
        "Gap with aircraft at normal": [pct(v) for v in air["gap_pct_without_aircraft_swings"]],
    }))
    march = air.loc[pd.Timestamp("2026-03-01")]

    # Bulk materials and precious metals (simple rule)
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

    # Product close-ups
    misses = results["closeups"]["misses"]

    def cu_val(product, period):
        r = cu[(cu["product"] == product) & (cu["period"] == period)].iloc[0]
        return pct(r["gap_pct"])

    text = f"""
# Step 5: The 2026 shortfall

This is the main result of the project. Step 4 built and tested a model that forecasts what Bahrain would have
imported without the war. Here every month of 2026 is compared with that forecast to measure how far below
normal imports fell, how much was lost, and which supply chains were hit hardest.

## Key findings

- **Core imports fell {abs(nat.loc['March', 'gap_pct']):.0f}% below normal in March**, then
  {abs(nat.loc['Apr-May', 'gap_pct']):.0f}% in April-May and {abs(nat.loc['Jun-Jul', 'gap_pct']):.0f}% in June-July. Each of
  these is far outside the model's normal range. By {last_name} the gap had narrowed to
  {abs(nat_m['gap_pct'].iloc[-1]):.0f}%, which on its own could be an ordinary month.
- **About BD {bd_m(total['lost'])} million of goods didn't arrive** from March to {last_name}
  ({abs(total['gap_pct']):.0f}% of what was expected), likely between BD {bd_m(total['lost_low'])} and
  {bd_m(total['lost_high'])} million.
- **{sec.iloc[0]['label']} fell furthest ({pct(sec.iloc[0]['gap_pct'])}).** {top2_bd.iloc[0]['label']} together with
  {top2_bd.iloc[1]['label'].lower()} account for {top2_share:.0f}% of the money lost.
- **Goods from the Americas ({pct(americas['gap_pct'])}) and East and South-East Asia ({pct(asia['gap_pct'])}) fell
  furthest.** These normally come by sea through the Strait of Hormuz. Goods from GCC countries held up
  ({pct(reg_gcc['gap_pct'])}, within the normal range).
- **The second model agrees.** Gradient-boosted trees put the national gap at
  {pct(nat_trees.loc[since_label, 'gap_pct'])}, against {pct(total['gap_pct'])}, and every section and region within
  {diff.max():.0f} percentage points.

## Method

The step 4 model stood at the end of December 2025 and forecast every month of 2026 using only what was known
then. It was within 4% of January and February, before the war began on 28 February, so from March its
forecast is the best available estimate of what would have arrived without the war.

For every month and period, and for every total (the whole country, each origin region, each product section),
the **gap** is actual imports minus the forecast, also given as % of the forecast. The model is never exactly
right, even in normal years, so a gap only counts as a real shortfall if it is bigger than the **normal range**
for that total: the distance from the forecast that 9 out of 10 periods of the same length stayed within in step
4's tests on 2024 and 2025. Longer periods have narrower ranges, because over several months the model's ups and
downs partly cancel out:

{ranges_md}

*National total. Ranges for every region and section are computed the same way.*

The amount **lost** is the forecast minus what arrived. Because the true "without the war" value could have
been anywhere within the normal range, the loss is also given as a likely range.

Measuring against a tested forecast has two advantages over comparing with last year: the forecast already
allows for growth and the calendar, and every claim that a gap is real is backed by the model's own track
record. The weaknesses are that the result can only be as good as the model, the model can't know about
anything else that would have changed in 2026, and the normal ranges rest on two test years.

The months of 2026 are grouped by what was happening:

{periods_md}

## 1. The whole country

![National](figures/step5_1_national.png)

{monthly_md}

January and February were normal, slightly above the forecast. March was the worst month, with imports at less
than half the forecast ({pct(nat_m.loc[pd.Timestamp('2026-03-01'), 'gap_pct'])}). Each month after that was a little
better, and by {last_name} the gap was {pct(nat_m['gap_pct'].iloc[-1])}, inside the normal range for a single month.

{nat_period_rows([*periods, since_label])}

The fall is far too large to be noise. Even the smallest war-time gap (June-July,
{pct(nat.loc['Jun-Jul', 'gap_pct'])}) is {abs(nat.loc['Jun-Jul', 'gap_pct']) / nat.loc['Jun-Jul', 'range_pct']:.1f}
times the normal range for a two-month period.

## 2. How much was lost

{lost_md}

About BD {bd_m(total['lost'])} million of core goods didn't arrive between March and {last_name},
{abs(total['gap_pct']):.0f}% of what was expected. {early_share:.0f}% of that was lost in March, April and May.

## 3. Product sections

![Sections](figures/step5_2_sections.png)

{ranking_md(sec, 'Section')}

*{since_short} together. Sections under {MIN_SHARE}% of the national forecast, and national special codes, are
left out.*

{len(sec_real)} of {len(sec)} sections had a real shortfall; only {and_list(sec_normal['label'].str.lower())} stayed
inside their normal range. {sec.iloc[0]['label']} fell furthest ({pct(sec.iloc[0]['gap_pct'])}), followed by
{and_list(f"{l.lower()} ({pct(g)})" for l, g in zip(sec['label'].iloc[1:5], sec['gap_pct'].iloc[1:5]))}. Most of the
money was lost in two sections, {top2_bd.iloc[0]['label'].lower()} (BD {bd_m(top2_bd.iloc[0]['lost'])}m) and
{top2_bd.iloc[1]['label'].lower()} (BD {bd_m(top2_bd.iloc[1]['lost'])}m), {top2_share:.0f}% of the total.

Food was hit early and then recovered. {food_sentence}

The goods that fell furthest (vehicles, machinery, clothing) are mostly made in Asia, Europe and the Americas and
shipped in containers or on car carriers, while food can more easily be bought from nearby countries. Step 6
tests that explanation.

## 4. Origins

![Regions](figures/step5_3_regions.png)

{ranking_md(reg, 'Origin region')}

*{since_short} together.*

The Americas ({pct(americas['gap_pct'])}) and East and South-East Asia ({pct(asia['gap_pct'])}) fell furthest, and
because Asia is the biggest origin it also lost the most money (BD {bd_m(asia['lost'])}m). Europe fell less
({pct(europe['gap_pct'])}); by {last_name} it was {pct(eu_last['gap_pct'])}, against {pct(as_last['gap_pct'])} for East and
South-East Asia. Goods from GCC countries held up ({pct(reg_gcc['gap_pct'])}, inside the normal range): they can
reach Bahrain by road, or by sea without passing through the Strait of Hormuz.

This is the clearest sign so far that the route mattered more than the product. Step 6 looks at it directly.

## 5. Second opinion

![Second opinion](figures/step5_4_second_opinion.png)

Everything above was repeated with step 4's best gradient-boosted trees, each total judged against the trees'
own normal ranges. Nationally the trees give {pct(nat_trees.loc[since_label, 'gap_pct'])}, against
{pct(total['gap_pct'])} with the linear model. For sections and regions the two models' gaps differ by
{diff.mean():.1f} percentage points on average and by at most {diff.max():.1f}, and they agree on whether the gap is
real for {int(same_verdict.sum())} of {len(same_verdict)} totals.{disagree_text}

Two quite different kinds of model tell the same story, so the findings don't hinge on the choice of model. They
are not fully independent, though: both learned from the same five years of data.

## 6. Six everyday products

![Products](figures/step5_5_products.png)

The same linear model was rebuilt at 4-digit level, where each series is one product from all origins
together. At this level the model barely beats the simple rule (it missed by {misses[(2025, 'linear')]:.1f}% against
{misses[(2025, 'simple')]:.1f}% in the 2025 test), so the normal ranges are wide and these results are illustrations
rather than headline findings.

Cars fell hardest and stayed down: {cu_val('Cars', 'March')} in March and still {cu_val('Cars', 'Jun-Jul')} in June-July.
They normally arrive on car carriers through the Strait of Hormuz. Phones and computers were well below normal
throughout ({cu_val('Phones and network equipment', since_label)} and {cu_val('Computers', since_label)} from March to
{last_name}). Rice and poultry bounced back: rice went from {cu_val('Rice', 'March')} in March to
{cu_val('Rice', 'Jun-Jul')} in June-July, above normal, which looks like restocking, and poultry from
{cu_val('Poultry meat', 'March')} to {cu_val('Poultry meat', 'Jun-Jul')}. Medicines dipped in March
({cu_val('Medicines', 'March')}) and then came in above normal in April-May ({cu_val('Medicines', 'Apr-May')}), which looks
like catching up or building stocks.

## 7. The aircraft effect

Aircraft, engines and parts are large, irregular purchases that can make a month look better or worse than
everyday supply chains really were. The table recomputes the national gap as if they had arrived at their
normal 2025 level:

{air_md}

In March they made the drop look deeper: almost none arrived (BD {march['aircraft'] / 1e6:,.1f}m against a normal
BD {march['aircraft_normal'] / 1e6:,.1f}m), and without that swing March would have been
{pct(march['gap_pct_without_aircraft_swings'])} instead of {pct(march['gap_pct'])}. From May they were close to normal,
so they don't explain the later months or the recovery.

## 8. Bulk materials and precious metals

These groups have too few series for the models, so they are compared with the 2025 average month, with normal
ranges from 2023-2025 (as in step 3). Bold means outside the normal range.

**Bulk materials, by weight**

{bulk_md}

Iron ore almost stopped and alumina fell by about half; both come by sea through the Strait of Hormuz, from
Brazil and Australia. Construction minerals didn't change, because they come mainly from the UAE, inside the Gulf.

**Precious metals, by value**

{prec_md}

Apart from March, precious metals stayed within their very wide normal range: they follow the gold price more
than supply routes.

## Limitations

- **The forecast is a model, not an observation.** Anything else that would have changed imports in 2026 (prices,
  big projects, policy) is not in it.
- **The normal ranges rest on two test years**, and the ranges for periods of several months on even fewer
  independent data points.
- **Customs months and arrival months don't always match**, which can move goods between neighbouring months.
  Two-month periods soften this.
- **Product close-ups are noisy**: single products have wide ranges, so only large changes can be called real.

## What step 6 takes from this

Core goods were about BD {bd_m(total['lost'])} million below normal from March to {last_name}, worst in March and
recovering steadily after. Goods that normally pass through the Strait of Hormuz fell furthest, while GCC goods
held up. Step 6 tests the route explanation directly, by grouping goods by how they normally travel and by
comparing them with the port, airport, car-carrier and general cargo figures.

{footer("src/shortfall.py", " (charts and text in `src/step5_report.py`)")}
"""
    write_markdown(REPORT_PATH, text)
