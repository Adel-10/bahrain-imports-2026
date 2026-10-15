"""Charts and the written report for the supplier-concentration extra (called by suppliers.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import glossary
from build_dataset import md_table
from explore import (AQUA, BASELINE, BLUE, GRID, INK, INK2, MUTED, ORANGE, SHADE, SURFACE, YELLOW, add_axes,
                     dot_handle, legend, new_figure, rounded_bar, save, source_note, tidy)
from step5_report import RED, and_list, pct
from step6_report import pon, short_period
from suppliers import (BEYOND_GROUPS, BEYOND_LEVEL, FAR_FROM, MINOR_SHARE, REPORT_PATH, SUPPLIER_GROUPS,
                       SUPPLIER_LEVEL)


def status_colour(outside: bool, gap: float) -> str:
    if not outside:
        return AQUA
    return RED if gap < 0 else BLUE


# ---------------------------------------------------------------------------
# Chart 1: the two groupings side by side
# ---------------------------------------------------------------------------
def fig_groups(gaps: pd.DataFrame, since_label: str) -> None:
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
    gaps, products, medians, eff, switch = r["gaps"], r["products"], r["medians"], r["effects"], r["switching"]
    since_label = r["since_label"]
    since_short = short_period(since_label)
    months_label = f"{since_short.replace('-', ' to ')}"
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
        row = g.loc[(level, name)]
        return row["actual"] / row["forecast"] * 100

    sup_pons = {n: pon_group(SUPPLIER_LEVEL, n) for n in SUPPLIER_GROUPS}
    far_pons = {n: pon_group(BEYOND_LEVEL, n) for n in BEYOND_GROUPS}
    all_clear_beyond = bool(others[others["effect"] == "beyond"]["clear"].all())
    none_clear_sup = not bool(eff[eff["effect"] == "suppliers"]["clear"].any())

    since_months = list(range(3, r["last_month"] + 1))
    months_long = f"{pd.Timestamp(2026, since_months[0], 1):%B}-{pd.Timestamp(2026, since_months[-1], 1):%B}"

    # ---- charts --------------------------------------------------------------
    fig_groups(gaps, since_label)
    fig_effects(eff)
    fig_switching(switch, months_long)

    # ---- tables --------------------------------------------------------------
    def group_md(level, groups, first_col):
        return md_table(pd.DataFrame({
            first_col: groups,
            "Products": [int(med.loc[(level, n), "products"]) for n in groups],
            "Share of value": [f"{med.loc[(level, n), 'value_share']:.0f}%" for n in groups],
            "From beyond the Middle East" if level == SUPPLIER_LEVEL else "Typical number of suppliers":
                [f"{med.loc[(level, n), 'beyond_share']:.0f}%" if level == SUPPLIER_LEVEL
                 else f"{med.loc[(level, n), 'median_suppliers']:.1f}" for n in groups],
            "% of normal (group total)": [pon(pon_group(level, n)) for n in groups],
            "Typical product": [pon(med.loc[(level, n), "median_pct_of_normal"]) for n in groups],
            "Normal range": [f"+/-{g.loc[(level, n), 'range_pct']:.0f}%" for n in groups],
            "Real shortfall?": ["**yes**" if g.loc[(level, n), "outside"] and g.loc[(level, n), "gap_pct"] < 0
                                else ("above normal" if g.loc[(level, n), "outside"] else "no") for n in groups],
        }))

    effects_md = md_table(pd.DataFrame({
        "Feature of the product": main["label"].tolist(),
        "Change in % of normal": [f"{v:+.1f} points" for v in main["points"]],
        "Likely range": [f"{lo:+.1f} to {hi:+.1f}" for lo, hi in zip(main["low"], main["high"])],
        "A clear effect?": ["**yes**" if c else "no (the range includes zero)" for c in main["clear"]],
    }))
    checks_md = md_table(pd.DataFrame({
        "Weighting": [wl for wl in dict.fromkeys(eff["weighting_label"])],
        "Each doubling of supplier countries": [
            f"{eff[(eff['weighting_label'] == wl) & (eff['effect'] == 'suppliers')]['points'].iloc[0]:+.1f} "
            f"({eff[(eff['weighting_label'] == wl) & (eff['effect'] == 'suppliers')]['low'].iloc[0]:+.1f} to "
            f"{eff[(eff['weighting_label'] == wl) & (eff['effect'] == 'suppliers')]['high'].iloc[0]:+.1f})"
            for wl in dict.fromkeys(eff["weighting_label"])],
        "Each extra 10% from beyond the Middle East": [
            f"{eff[(eff['weighting_label'] == wl) & (eff['effect'] == 'beyond')]['points'].iloc[0]:+.1f} "
            f"({eff[(eff['weighting_label'] == wl) & (eff['effect'] == 'beyond')]['low'].iloc[0]:+.1f} to "
            f"{eff[(eff['weighting_label'] == wl) & (eff['effect'] == 'beyond')]['high'].iloc[0]:+.1f})"
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

    # ---- sentences ----------------------------------------------------------
    worst_group = min(sup_pons, key=sup_pons.get)
    low_med = med.loc[(SUPPLIER_LEVEL, SUPPLIER_GROUPS[0]), "median_suppliers"]
    high_med = med.loc[(SUPPLIER_LEVEL, SUPPLIER_GROUPS[-1]), "median_suppliers"]
    sup_span = sup["high"] * np.log2(high_med / low_med)
    lumpy = {"8703", "8409", "8411"}                     # cars and aircraft engines
    lean_products = products[~products.index.isin(lumpy)]
    lean = {n: lean_products.loc[lean_products["supplier_group"] == n, "actual"].sum()
            / lean_products.loc[lean_products["supplier_group"] == n, "forecast"].sum() * 100 for n in SUPPLIER_GROUPS}
    if "8703" in products.index:
        cars = products.loc["8703"]
        cars_sentence = (f"cars, at {pon(cars['pct_of_normal'])} of normal and one of the biggest products, sit in the "
                         f"\"{cars['supplier_group']}\" group.")
    else:
        cars_sentence = "a handful of large products, such as cars, dominate some groups."

    sup_lo, sup_hi = min(sup_pons.values()), max(sup_pons.values())
    far_lo_name, far_hi_name = BEYOND_GROUPS[0], BEYOND_GROUPS[-1]
    full_far_effect = -far_eff["points"] * 10
    normal_minor = f"{min(far_switch.loc[normal_years, 'minor_share']):.0f}-{max(far_switch.loc[normal_years, 'minor_share']):.0f}%"
    gcc_before = f"{min(far_switch.loc[normal_years, 'gcc_share']):.0f}-{max(far_switch.loc[normal_years, 'gcc_share']):.0f}%"

    text = f"""# Extra: Were products that relied on few supplier countries hit harder?

*Generated by `src/suppliers.py` (charts and text by `src/suppliers_report.py`). Re-running the script refreshes every
number.*

Steps 5 and 6 showed which kinds of goods and which origins were hit hardest. A common piece of supply-chain advice is
to **spread purchases across many supplier countries**, so that losing one doesn't matter much. This extra asks whether
that advice would have helped Bahrain in 2026:

1. Were products bought from only one or two countries hit harder than products with many suppliers?
2. If not, what did matter?
3. Did buyers switch to other suppliers during the war?

## The short answer

| Question | Answer |
|---|---|
| Were products with few supplier countries hit harder? | **No clear effect.** Every group fell a long way, to between {pon(sup_lo)} and {pon(sup_hi)} of normal from {since_short}, in no particular order. Compared fairly (with the share from far away held fixed), each doubling of the number of supplier countries changed a product's result by {sup['points']:+.1f} points, likely between {sup['low']:+.1f} and {sup['high']:+.1f}: **no measurable effect**, and small next to distance. |
| What did matter? | **Where the suppliers were.** Products bought mostly nearby (under 25% from beyond the Middle East) were at {pon(far_pons[far_lo_name])} of normal; products almost entirely from beyond the Middle East at {pon(far_pons[far_hi_name])}. Each extra 10% from beyond the Middle East meant about **{-far_eff['points']:.0f} points further below normal** (likely {-far_eff['high']:.1f} to {-far_eff['low']:.1f}).{" The same holds however the products are weighted." if all_clear_beyond and none_clear_sup else ""} |
| Did buyers switch suppliers? | **A little.** For products that normally come from far away, {far_switch.loc[war_year, 'minor_share']:.0f}% of what arrived came from countries that had been minor suppliers, against {normal_minor} in normal years. Most of those were other distant countries; GCC suppliers' share of these products rose from {gcc_before} to {far_switch.loc[war_year, 'gcc_share']:.0f}%. |
| Why it matters | **Spreading purchases across many countries doesn't protect a supply chain if they all ship through the same strait.** Having suppliers on a different route does. |

## Words used in this report

{glossary.table(8, ["product", "forecast", "pct_of_normal", "normal_range", "real_shortfall", "since_war",
                    "effective_suppliers", "beyond_share", "beyond_me", "median_product", "regression", "likely_range",
                    "minor_supplier", "far_dependent"])}

---

## Before you start: how this extra works

### Step 1: Products and their forecasts

Each product is one 4-digit HS heading from all origins together. {n_products} products had imports in almost every
month before the war; together they are {coverage:.0f}% of core goods by value. Step 4's linear model, rebuilt at this
level (as for step 5's close-ups), forecasts each one for 2026 as if there had been no war. Added up, the products were at
{pon(nat)} of normal from {since_short}, matching step 5.

### Step 2: How many countries did a product rely on?

Counting countries isn't enough: a product with one big supplier and ten tiny ones really relies on one. So each product
gets an **effective number of supplier countries** from its 2024-2025 imports:

> effective number = 1 / (sum of each country's share, squared)

* two countries with 50% each: 1 / (0.5² + 0.5²) = **2**;
* 90% from one and 10% from another: 1 / (0.9² + 0.1²) = **1.2**;
* four countries with 25% each: **4**.

### Step 3: How far away were its suppliers?

Each product also gets its **share from beyond the Middle East** in 2024-2025: the part that normally has to pass the
Strait of Hormuz or fly.

### Step 4: Comparing fairly

The two could be tangled together. If products with few suppliers also tended to come from far away, they would look
worse for the wrong reason. So, after comparing groups one way at a time, a **regression** looks at both at once, along
with value per kg (light goods could fly) and whether a product is a vehicle (car carriers stopped). Re-running it on
2,000 random re-draws of the products gives each effect a **likely range**.

---

## 1. Did relying on few countries matter?

![Groups](figures/extra_suppliers_1_groups.png)

**What we did:** grouped products by their effective number of supplier countries and compared each group's total with
its forecast, from {since_short}.

**What we found:**

{group_md(SUPPLIER_LEVEL, SUPPLIER_GROUPS, "Supplier countries (effective)")}

1. **Every group had a real shortfall**, falling to between {pon(sup_lo)} and {pon(sup_hi)} of normal.
2. **The order doesn't follow the number of suppliers.** Products with under 2 effective suppliers
   ({pon(sup_pons[SUPPLIER_GROUPS[0]])}) did better than the {worst_group} group ({pon(sup_pons[worst_group])}). Big single
   products move the totals: {cars_sentence}
3. **Leaving out cars and aircraft engines**, the groups range from {pon(min(lean.values()))} to {pon(max(lean.values()))},
   with a slight edge for products with more suppliers. Section 3 checks whether that edge survives once distance is
   allowed for. It doesn't clearly.
4. **The groups are about equally far-dependent**: each gets around {med.xs(SUPPLIER_LEVEL)['beyond_share'].mean():.0f}% of its
   value from beyond the Middle East. That is a first hint that distance, not the number of countries, decides the result.

**The biggest products that relied on one or two countries ({since_short}):**

{few_md}

Even these ranged from {pon(few['pct_of_normal'].min())} to {pon(few['pct_of_normal'].max())} of normal. Relying on one
country didn't decide how they fared.

---

## 2. What did matter?

**What we did:** grouped the same products by their share from beyond the Middle East.

**What we found:**

{group_md(BEYOND_LEVEL, BEYOND_GROUPS, "From beyond the Middle East")}

1. **A clear staircase.** Products bought mostly nearby were at {pon(far_pons[far_lo_name])} of normal
   ({"inside the normal range" if not g.loc[(BEYOND_LEVEL, far_lo_name), "outside"] else "outside the normal range"}).
   The more a product depended on far-away suppliers, the further it fell, down to {pon(far_pons[far_hi_name])} for
   products almost entirely from beyond the Middle East.
2. **The typical product tells the same story**, so it isn't just a few big products.

---

## 3. Both at once

![Effects](figures/extra_suppliers_2_effects.png)

**What we did:** a regression of each product's % of normal on its number of supplier countries (doubling), its share from
beyond the Middle East (per 10%), its value per kg (doubling) and whether it is a vehicle. Products are weighted by size,
so a BD 10 million product counts more than a BD 10,000 one. % of normal is capped at 200% so a few tiny products with
huge swings can't dominate.

**What we found:**

{effects_md}

**The same check with other weightings:**

{checks_md}

1. **The number of supplier countries makes no measurable difference.** Every likely range includes zero. Even at
   the top of its main range ({sup['high']:+.1f} points per doubling), going from a typical "under 2" product
   ({low_med:.1f} suppliers) to a typical "8 or more" one ({high_med:.1f}) would be worth about {sup_span:.0f} points,
   well below the {full_far_effect:.0f} points between buying everything nearby and everything from far away.
2. **Distance makes a clear difference** under every weighting: about {-far_eff['points']:.0f} points per extra 10% from
   beyond the Middle East. A product entirely from far away did about {full_far_effect:.0f} points worse than one
   bought entirely nearby, everything else being equal.
3. **Vehicles did worse still**, even after allowing for distance: car carriers stopped altogether (step 6).

**Why it matters:** in this shock the danger was the **route**, not reliance on one country. Five suppliers in East Asia
all ship through the same strait, so having five of them gave no protection.

---

## 4. Did buyers switch suppliers?

![Switching](figures/extra_suppliers_3_switching.png)

**What we did:** for each product, found the countries that had supplied under {MINOR_SHARE:.0%} of it in the two years
before, and measured how much of each year's {months_long} imports came from them, in the war year and in normal years.

**What we found:**

{switch_md}

1. **Some switching happened, for the far-dependent products.** {far_switch.loc[war_year, 'minor_share']:.0f}% of what
   arrived came from minor suppliers, against {normal_minor} in normal years.
2. **Most of it went to other distant countries**, not nearby ones. GCC suppliers' share of these products rose
   from {gcc_before} to {far_switch.loc[war_year, 'gcc_share']:.0f}%: almost double, but still small.
3. **Brand-new suppliers were rare**: {far_switch.loc[war_year, 'new_share']:.1f}% of far-dependent imports came from
   countries that hadn't supplied the product at all in the two years before.

**Why it matters:** buyers adapted by leaning on countries they already bought a little from, mostly other distant
ones whose goods could still get through, rather than by finding brand-new suppliers or switching to nearby ones.

---

## What this can't tell us

1. **Country of origin isn't the shipping route.** The customs data records where goods were made, not how they
   travelled. A Chinese product bought from a Dubai warehouse counts as Chinese.
2. **One heading can hold many items.** HS heading 8471 (computers) mixes laptops and parts, which may have different
   suppliers.
3. **Single products are hard to forecast**, so this extra compares groups and uses ranges and re-draws rather than
   reading much into any one product.
4. **The normal ranges come from two test years**, as in the rest of the project.

## What this means

The usual advice to "diversify suppliers" needs a twist for Bahrain: **diversify routes, not just countries.** Products
bought mostly from nearby countries, which can come by road, held up; products bought from far away fell a long way
whether they had one supplier or many.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
