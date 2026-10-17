"""Charts and the written report for the supplier-concentration extra (called by suppliers.py).

    python src/suppliers.py        # runs the analysis, then calls write() in this module

Charts (reports/figures/)
-------------------------
1. extra_suppliers_1_groups.png     % of normal for products grouped by number of suppliers (left)
                                    and by share from beyond the Middle East (right).
2. extra_suppliers_2_effects.png    the regression's two main effects under three weightings, each
                                    with its 90% bootstrap range.
3. extra_suppliers_3_switching.png  share of far-dependent products' imports from minor suppliers,
                                    2024-2026, split by where those suppliers are.

Report (reports/extra_suppliers.md)
-----------------------------------
Key findings, the method (products, effective suppliers, distance, the regression), the group
comparisons, the regression, switching, and the limitations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from charts import (AQUA, BASELINE, BLUE, GRID, INK, INK2, ORANGE, RED, SURFACE, YELLOW, add_axes, dot_handle, legend,
                    new_figure, rounded_bar, save, source_note, tidy)
from reporting import footer, md_table, pon, short_period, write_markdown
from suppliers import BEYOND_GROUPS, BEYOND_LEVEL, MINOR_SHARE, REPORT_PATH, SUPPLIER_GROUPS, SUPPLIER_LEVEL


def status_colour(outside: bool, gap: float) -> str:
    """Dot colour: aqua inside the normal range, red below it, blue above it."""
    if not outside:
        return AQUA
    return RED if gap < 0 else BLUE


# ---------------------------------------------------------------------------
# Chart 1: the two groupings side by side
# ---------------------------------------------------------------------------
def fig_groups(gaps: pd.DataFrame, since_label: str) -> None:
    """Figure 1: two panels of lollipops from 100% (normal) to each group's % of normal, over its normal range."""
    g = gaps[gaps["period"] == since_label].set_index(["level", "name"])
    fig = new_figure(10, 5.4, "What mattered was where suppliers were, not how many there were",
                     f"Imports from {short_period(since_label)} 2026 as % of normal (the forecast without the war), "
                     "for products grouped two ways. The grey band is the\nnormal range: a dot inside it could be "
                     "ordinary ups and downs.")
    legend(fig, [dot_handle(RED), dot_handle(AQUA), Rectangle((0, 0), 1, 1, color=GRID, lw=0)],
           ["Below normal (a real shortfall)", "Inside the normal range", "Normal range"], top=1.3, ncol=3)
    w, h = fig.get_size_inches()
    panels = [(SUPPLIER_LEVEL, SUPPLIER_GROUPS, "Number of supplier countries (effective)"),
              (BEYOND_LEVEL, BEYOND_GROUPS, "Share of the product from beyond the Middle East")]
    left, gap_in, right, top, bottom = 1.55, 1.75, 0.35, 2.25, 0.8
    pw = (w - left - gap_in - right) / 2
    for k, (level, groups, title) in enumerate(panels):
        x0 = left + k * (pw + gap_in)
        ax = fig.add_axes([x0 / w, bottom / h, pw / w, (h - top - bottom) / h])
        fig.text(x0 / w - 1.0 / w, (h - top + 0.32) / h, title, fontsize=10.5, fontweight="semibold", color=INK)
        n = len(groups)
        ax.set_xlim(40, 120)
        ax.set_ylim(n - 0.5, -0.5)
        tidy(ax, grid_axis="x")
        ax.axvline(100, color=BASELINE, lw=1, zorder=1)
        for i, name in enumerate(groups):
            row = g.loc[(level, name)]
            value = row["actual"] / row["forecast"] * 100
            rng = row["range_pct"]
            ax.add_patch(Rectangle((100 - rng, i - 0.3), 2 * rng, 0.6, color=GRID, lw=0, zorder=1))
            ax.plot([100, value], [i, i], color=BASELINE, lw=2, zorder=2, solid_capstyle="butt")
            ax.plot([value], [i], "o", ms=9, color=status_colour(row["outside"], row["gap_pct"]), mec=SURFACE,
                    mew=2, zorder=4)
            ax.text(value + (-3 if value < 100 else 3), i, f"{value:.0f}%", ha="right" if value < 100 else "left",
                    va="center", fontsize=8.5, color=INK2, zorder=5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(groups, color=INK, fontsize=9.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(axis="x", labelsize=8.5)
    source_note(fig)
    save(fig, "extra_suppliers_1_groups.png")


# ---------------------------------------------------------------------------
# Chart 2: the regression, both effects under three weightings
# ---------------------------------------------------------------------------
def fig_effects(effects: pd.DataFrame) -> None:
    """Figure 2: estimate and 90% range of the two main effects, one row per weighting."""
    shown = [("suppliers", "Each doubling of the number of supplier countries", BLUE),
             ("beyond", "Each extra 10% of the product from beyond the Middle East", ORANGE)]
    weightings = list(dict.fromkeys(effects["weighting_label"]))
    rows = []                                    # (kind, text, colour, row): headers, then one row per weighting
    for key, label, colour in shown:
        rows.append(("header", label, colour, None))
        for wl in weightings:
            rows.append(("row", wl, colour,
                         effects[(effects["effect"] == key) & (effects["weighting_label"] == wl)].iloc[0]))
    n = len(rows)
    fig = new_figure(10, 1.9 + 0.44 * n + 0.95,
                     "Compared fairly, the number of suppliers makes no difference; distance does",
                     "Change in % of normal (March-July 2026) linked to each feature of a product, with the other "
                     "held fixed. The dot is the estimate and the line\nits likely range (90%). A line that crosses "
                     "zero means no effect can be told apart from none.")
    ax = add_axes(fig, left=3.3, right=0.45, top=1.55, bottom=0.85)
    data = [r for kind, *_, r in rows if kind == "row"]
    lo = min(-8.0, min(r["low"] for r in data) - 1)
    hi = max(8.0, max(r["high"] for r in data) + 1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(n - 0.5, -0.7)
    tidy(ax, grid_axis="x")
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    labels = []
    for i, (kind, text, colour, r) in enumerate(rows):
        if kind == "header":
            ax.text(lo, i + 0.1, text, ha="left", va="center", fontsize=10, fontweight="semibold", color=INK,
                    zorder=6, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
            labels.append("")
            continue
        ax.plot([r["low"], r["high"]], [i, i], color=colour, lw=2.2, zorder=3, solid_capstyle="round")
        ax.plot([r["points"]], [i], "o", ms=9, color=colour, mec=SURFACE, mew=2, zorder=4)
        ax.text(r["points"], i - 0.24, f"{r['points']:+.1f}  ({r['low']:+.1f} to {r['high']:+.1f})", ha="center",
                va="bottom", fontsize=8.5, color=INK2, zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        labels.append(text)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax.set_xlabel("percentage points of normal", fontsize=9, color=INK2)
    source_note(fig)
    save(fig, "extra_suppliers_2_effects.png")


# ---------------------------------------------------------------------------
# Chart 3: switching to minor suppliers
# ---------------------------------------------------------------------------
def fig_switching(switch: pd.DataFrame, months_label: str) -> None:
    """Figure 3: stacked bars, one per year, of the minor-supplier share split by supplier location."""
    far = switch[switch["scope"] == "Far-dependent products"].set_index("year")
    parts = [("minor_from_Beyond the Middle East", "Other countries beyond the Middle East", BLUE),
             ("minor_from_Other Middle East", "Rest of the Middle East", YELLOW),
             ("minor_from_GCC", "GCC countries", ORANGE)]
    years = list(far.index)
    fig = new_figure(10, 5.0, "Buyers of far-away products turned more to suppliers they had used little before",
                     f"Far-dependent products: share of {months_label} imports that came from minor suppliers "
                     f"(under {MINOR_SHARE:.0%} of the product in the two\nyears before), split by where those "
                     "suppliers are.")
    legend(fig, [Rectangle((0, 0), 1, 1, color=c, lw=0) for *_, c in parts], [p[1] for p in parts], top=1.3, ncol=3)
    ax = add_axes(fig, left=0.9, right=0.45, top=1.9, bottom=0.8)
    top_val = max(far["minor_share"]) * 1.25
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(0, top_val)
    tidy(ax)
    fig.canvas.draw()
    gap = 2 * top_val / ax.get_window_extent().height
    for i, year in enumerate(years):
        base = 0.0
        for k, (col, _, colour) in enumerate(parts):
            v = far.loc[year, col]
            if k == len(parts) - 1:
                rounded_bar(ax, base, v, i, 0.55, colour, horizontal=False)
            else:
                ax.add_patch(Rectangle((i - 0.275, base), 0.55, max(v - gap, 0), color=colour, lw=0, zorder=3))
            base += v
        ax.text(i, base + top_val * 0.02, f"{far.loc[year, 'minor_share']:.0f}%", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="semibold")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"{y}\n{'(war)' if y == years[-1] else '(normal)'}" for y in years], color=INK)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    source_note(fig)
    save(fig, "extra_suppliers_3_switching.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write(r: dict) -> None:
    """Draw the three charts and write reports/extra_suppliers.md from the results of suppliers.py."""
    gaps, products, medians, eff, switch = r["gaps"], r["products"], r["medians"], r["effects"], r["switching"]
    since_label = r["since_label"]
    since_short = short_period(since_label)
    g = gaps[gaps["period"] == since_label].set_index(["level", "name"])
    med = medians.set_index(["level", "name"])
    main = eff[eff["weighting"] == "size"].set_index("effect")
    others = eff[eff["weighting"] != "size"]
    sup, far_eff = main.loc["suppliers"], main.loc["beyond"]
    far_switch = switch[switch["scope"] == "Far-dependent products"].set_index("year")
    all_switch = switch[switch["scope"] == "All products"].set_index("year")
    war_year = int(far_switch.index.max())
    normal_years = [y for y in far_switch.index if y != war_year]
    nat = r["national_pct"]
    n_products = len(products)
    coverage = products["value_bd"].sum() / r["core_value_2024_25"] * 100

    def pon_group(level, name):
        """A group's total as % of its forecast over the war period."""
        row = g.loc[(level, name)]
        return row["actual"] / row["forecast"] * 100

    sup_pons = {n: pon_group(SUPPLIER_LEVEL, n) for n in SUPPLIER_GROUPS}
    far_pons = {n: pon_group(BEYOND_LEVEL, n) for n in BEYOND_GROUPS}
    all_clear_beyond = bool(others[others["effect"] == "beyond"]["clear"].all())
    none_clear_sup = not bool(eff[eff["effect"] == "suppliers"]["clear"].any())

    since_months = list(range(3, r["last_month"] + 1))
    months_long = f"{pd.Timestamp(2026, since_months[0], 1):%B}-{pd.Timestamp(2026, since_months[-1], 1):%B}"

    # ---- charts ----
    fig_groups(gaps, since_label)
    fig_effects(eff)
    fig_switching(switch, months_long)

    # ---- tables ----
    def group_md(level, groups, first_col):
        return md_table(pd.DataFrame({
            first_col: groups,
            "Products": [int(med.loc[(level, n), "products"]) for n in groups],
            "Share of value": [f"{med.loc[(level, n), 'value_share']:.0f}%" for n in groups],
            "From beyond the Middle East" if level == SUPPLIER_LEVEL else "Typical number of suppliers":
                [f"{med.loc[(level, n), 'beyond_share']:.0f}%" if level == SUPPLIER_LEVEL
                 else f"{med.loc[(level, n), 'median_suppliers']:.1f}" for n in groups],
            "% of normal (group total)": [pon(pon_group(level, n)) for n in groups],
            "Median product": [pon(med.loc[(level, n), "median_pct_of_normal"]) for n in groups],
            "Normal range": [f"+/-{g.loc[(level, n), 'range_pct']:.0f}%" for n in groups],
            "Real shortfall?": ["**yes**" if g.loc[(level, n), "outside"] and g.loc[(level, n), "gap_pct"] < 0
                                else ("above normal" if g.loc[(level, n), "outside"] else "no") for n in groups],
        }))

    effects_md = md_table(pd.DataFrame({
        "Feature of the product": main["label"].tolist(),
        "Change in % of normal": [f"{v:+.1f} points" for v in main["points"]],
        "90% range": [f"{lo:+.1f} to {hi:+.1f}" for lo, hi in zip(main["low"], main["high"])],
        "Clear effect?": ["**yes**" if c else "no (the range includes zero)" for c in main["clear"]],
    }))

    def effect_cell(wl, key):
        row = eff[(eff["weighting_label"] == wl) & (eff["effect"] == key)].iloc[0]
        return f"{row['points']:+.1f} ({row['low']:+.1f} to {row['high']:+.1f})"

    checks_md = md_table(pd.DataFrame({
        "Weighting": list(dict.fromkeys(eff["weighting_label"])),
        "Each doubling of supplier countries": [effect_cell(wl, "suppliers") for wl in dict.fromkeys(eff["weighting_label"])],
        "Each extra 10% from beyond the Middle East": [effect_cell(wl, "beyond")
                                                       for wl in dict.fromkeys(eff["weighting_label"])],
    }))
    switch_md = md_table(pd.DataFrame({
        "": [f"{y} ({'war' if y == war_year else 'normal year'})" for y in far_switch.index],
        "All products: from minor suppliers": [f"{v:.0f}%" for v in all_switch["minor_share"]],
        "Far-dependent products: from minor suppliers": [f"{v:.0f}%" for v in far_switch["minor_share"]],
        "... of which minor suppliers in GCC countries": [f"{v:.1f}%" for v in far_switch["minor_from_GCC"]],
        "Far-dependent products: all GCC suppliers": [f"{v:.1f}%" for v in far_switch["gcc_share"]],
    }))
    few, names = r["few"], r["names"]
    few_md = md_table(pd.DataFrame({
        "Product": [names["products"][c] for c in few.index],
        "Main supplier": [f"{names['countries'][t]} ({s:.0%})" for t, s in zip(few["top_country"], few["top_share"])],
        "From beyond the Middle East": [f"{v:.0%}" for v in few["beyond_share"]],
        "% of normal": [pon(v) for v in few["pct_of_normal"]],
    }))

    # ---- numbers and sentences that depend on the data ----
    worst_group = min(sup_pons, key=sup_pons.get)
    low_med = med.loc[(SUPPLIER_LEVEL, SUPPLIER_GROUPS[0]), "median_suppliers"]
    high_med = med.loc[(SUPPLIER_LEVEL, SUPPLIER_GROUPS[-1]), "median_suppliers"]
    # The largest effect of suppliers still consistent with the data: the top of its range, times the
    # number of doublings between a typical "under 2" product and a typical "8 or more" one.
    sup_span = sup["high"] * np.log2(high_med / low_med)
    lumpy = {"8703", "8409", "8411"}                     # cars and aircraft engines
    lean_products = products[~products.index.isin(lumpy)]
    lean = {n: lean_products.loc[lean_products["supplier_group"] == n, "actual"].sum()
            / lean_products.loc[lean_products["supplier_group"] == n, "forecast"].sum() * 100 for n in SUPPLIER_GROUPS}
    if "8703" in products.index:
        cars = products.loc["8703"]
        cars_sentence = (f"cars, one of the biggest products and at {pon(cars['pct_of_normal'])} of normal, sit in the "
                         f"\"{cars['supplier_group']}\" group.")
    else:
        cars_sentence = "a handful of large products, such as cars, dominate some groups."

    sup_lo, sup_hi = min(sup_pons.values()), max(sup_pons.values())
    far_lo_name, far_hi_name = BEYOND_GROUPS[0], BEYOND_GROUPS[-1]
    full_far_effect = -far_eff["points"] * 10            # 0% -> 100% from beyond the Middle East
    normal_minor = f"{min(far_switch.loc[normal_years, 'minor_share']):.0f}-{max(far_switch.loc[normal_years, 'minor_share']):.0f}%"
    gcc_before = f"{min(far_switch.loc[normal_years, 'gcc_share']):.0f}-{max(far_switch.loc[normal_years, 'gcc_share']):.0f}%"
    robust = " The result holds however the products are weighted." if all_clear_beyond and none_clear_sup else ""
    nearby_inside = "inside" if not g.loc[(BEYOND_LEVEL, far_lo_name), "outside"] else "outside"

    text = f"""
# Extra: Did relying on few supplier countries matter?

A common lesson from supply shocks is to spread purchases across many supplier countries, so that losing one
doesn't matter much. Steps 5 and 6 showed which goods and origins were hit hardest in 2026. This extra asks
whether products that depended on one or two countries were hit harder than products with many suppliers, and if
not, what did matter.

## Key findings

- **The number of supplier countries made no measurable difference.** Grouped by how many countries they relied
  on, products fell to between {pon(sup_lo)} and {pon(sup_hi)} of normal over {since_short}, in no particular order.
  Compared fairly, each doubling of the number of suppliers changed a product's result by {sup['points']:+.1f}
  points (90% range {sup['low']:+.1f} to {sup['high']:+.1f}).
- **Distance did.** Products bought mostly from nearby countries were at {pon(far_pons[far_lo_name])} of normal;
  products almost entirely from beyond the Middle East were at {pon(far_pons[far_hi_name])}. Each extra 10% from
  beyond the Middle East meant about {-far_eff['points']:.0f} points further below normal (90% range
  {-far_eff['high']:.1f} to {-far_eff['low']:.1f}).{robust}
- **Buyers switched a little.** For products that normally come from far away,
  {far_switch.loc[war_year, 'minor_share']:.0f}% of what arrived came from countries that had been minor suppliers,
  against {normal_minor} in normal years. Most of those were other distant countries.
- **The lesson for Bahrain is to diversify routes, not just countries.** Five suppliers in East Asia all ship
  through the same strait.

## Method

**Products and forecasts.** Each product is one 4-digit HS heading from all origins together. {n_products} products
had imports in almost every month before the war; together they are {coverage:.0f}% of core goods by value. Step 4's
linear model, rebuilt at this level (as for step 5's close-ups), forecasts each one for 2026 as if there had been
no war, and its 2024-2025 tests give each group of products a normal range. Added up, the products were at
{pon(nat)} of normal over {since_short}, in line with step 5.

**Reliance on few countries.** Counting supplier countries isn't enough: a product with one big supplier and ten
tiny ones really relies on one. Each product therefore gets an effective number of supplier countries from its
2024-2025 imports, 1 / (sum of each country's squared share), which is the inverse of the Herfindahl-Hirschman
index. Two countries with 50% each give 2; 90% from one and 10% from another gives 1.2; four countries with 25%
each give 4.

**Distance.** Each product also gets its share from beyond the Middle East in 2024-2025: the part that normally
has to pass the Strait of Hormuz or fly.

**Comparing fairly.** The two measures could be tangled together: if products with few suppliers also tended to
come from far away, they would look worse for the wrong reason. So after comparing groups one way at a time, a
weighted regression looks at both at once, holding value per kg (light goods could fly) and vehicles (car carriers
stopped) constant. Products are weighted by size, and % of normal is capped at 200% so a few tiny products with
huge swings can't dominate. Re-running the regression on 2,000 random re-draws of the products (a bootstrap) gives
each effect a 90% range without assuming the errors follow a normal distribution.

The approach measures every product against its own tested forecast and separates the two explanations, but it
shows association rather than cause, and single-product forecasts are noisy. The regression is therefore repeated
with two other weightings as a check.

## 1. Number of supplier countries

![Groups](figures/extra_suppliers_1_groups.png)

{group_md(SUPPLIER_LEVEL, SUPPLIER_GROUPS, "Supplier countries (effective)")}

*{since_short}. "Median product" is the middle product of each group, which is less affected by a few big
products than the group total.*

Every group had a real shortfall, falling to between {pon(sup_lo)} and {pon(sup_hi)} of normal, and the order
doesn't follow the number of suppliers: products with under 2 effective suppliers ({pon(sup_pons[SUPPLIER_GROUPS[0]])})
did better than the {worst_group} group ({pon(sup_pons[worst_group])}). Single large products move these totals;
{cars_sentence} Leaving out cars and aircraft engines, the groups range from {pon(min(lean.values()))} to
{pon(max(lean.values()))}, with a slight edge for products with more suppliers. Section 3 checks whether that edge
survives once distance is allowed for (it doesn't clearly). The groups are also about equally dependent on distant
suppliers, each getting around {med.xs(SUPPLIER_LEVEL)['beyond_share'].mean():.0f}% of its value from beyond the
Middle East, a first hint that distance rather than the number of countries decided the result.

The biggest products that relied on one or two countries:

{few_md}

Even these ranged from {pon(few['pct_of_normal'].min())} to {pon(few['pct_of_normal'].max())} of normal; relying on one
country didn't decide how they fared.

## 2. Distance

{group_md(BEYOND_LEVEL, BEYOND_GROUPS, "From beyond the Middle East")}

*{since_short}.*

Here there is a clear staircase. Products bought mostly nearby were at {pon(far_pons[far_lo_name])} of normal
({nearby_inside} the normal range). The more a product depended on distant suppliers, the further it fell, down to
{pon(far_pons[far_hi_name])} for products almost entirely from beyond the Middle East. The median product tells the
same story, so this isn't driven by a few big products.

## 3. Both at once

![Effects](figures/extra_suppliers_2_effects.png)

{effects_md}

*Weighted by size. Supplier count and value per kg enter as doublings, distance per 10 percentage points.*

The same regression with other weightings:

{checks_md}

The number of supplier countries makes no measurable difference: every range includes zero. Even at the top of
its main range ({sup['high']:+.1f} points per doubling), going from a typical "under 2" product ({low_med:.1f}
suppliers) to a typical "8 or more" one ({high_med:.1f}) would be worth about {sup_span:.0f} points, well below the
{full_far_effect:.0f} points between buying everything nearby and everything from far away. Distance makes a clear
difference under every weighting, about {-far_eff['points']:.0f} points per extra 10% from beyond the Middle East.
Vehicles did worse still, even after allowing for distance, because car carriers stopped altogether (step 6).

## 4. Did buyers switch suppliers?

![Switching](figures/extra_suppliers_3_switching.png)

For each product, the countries that had supplied under {MINOR_SHARE:.0%} of it in the two years before count as
minor suppliers. The table shows how much of each year's {months_long} imports came from them, so 2026 can be
compared with the normal churn among suppliers.

{switch_md}

*"Far-dependent" products get at least 75% of their value from beyond the Middle East.*

Some switching happened for far-dependent products: {far_switch.loc[war_year, 'minor_share']:.0f}% of what arrived
came from minor suppliers, against {normal_minor} in normal years. Most of it went to other distant countries rather
than nearby ones; GCC suppliers' share of these products rose from {gcc_before} to
{far_switch.loc[war_year, 'gcc_share']:.0f}%, almost double but still small. Brand-new suppliers were rare:
{far_switch.loc[war_year, 'new_share']:.1f}% of far-dependent imports came from countries that hadn't supplied the
product at all in the two years before. Buyers adapted by leaning on countries they already bought a little from,
mostly distant ones whose goods could still get through.

## Limitations

- **Country of origin isn't the shipping route.** The customs data records where goods were made, not how they
  travelled; a Chinese product bought from a Dubai warehouse counts as Chinese.
- **One heading can hold many items.** HS heading 8471 (computers) mixes laptops and parts, which may have
  different suppliers.
- **Single products are hard to forecast**, so this analysis compares groups and uses ranges and re-draws rather
  than reading much into any one product.
- **Association, not cause.** The regression holds a few things constant, but other differences between products
  could still play a part.
- **The normal ranges come from two test years**, as in the rest of the project.

{footer("src/suppliers.py", " (charts and text in `src/suppliers_report.py`)")}
"""
    write_markdown(REPORT_PATH, text)
