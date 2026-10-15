"""Step 7: the whole project in one read.

    python src/summary.py          (run steps 4 to 6 first: it reads their saved results)

Pulls the main results of steps 4, 5 and 6 into reports/summary.md, in the same easy style as the
step reports. Every number is read from the files those steps save, so re-running the pipeline
keeps the summary in step with the detailed reports. It draws no new charts; it reuses the main
ones from the step reports.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

import glossary
from backtest import aggregate_errors, coverage, ranges_from
from build_dataset import md_table
from config import ROOT
from features import load_core
from reference import period_event
from step5_report import MIN_SHARE, SKIP_SECTIONS, and_list, bd_m, pct
from step6_report import pon, region_text, share_words, short_period
from suppliers import BEYOND_GROUPS

INTERIM_DIR = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "reports" / "summary.md"
NEEDED = ["step4_backtest.parquet", "step5_gaps.csv", "step6_indicators.csv", "step6_route_gaps.csv",
          "step6_elimination.csv", "step6_adapting.csv", "step6_engines.csv"]


# ---------------------------------------------------------------------------
# Results from the earlier steps
# ---------------------------------------------------------------------------
def model_scores(backtest: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    """National miss in each test year for the simple rule, the linear model and the trees.

    The simple rule forecasts every series as last year's average month, so its national
    forecast is simply last year's national average month.
    """
    rows = []
    national = core.groupby("date")["value_bd"].sum()
    for year in sorted(backtest["year"].unique()):
        actual = national[national.index.year == year]
        forecast = national[national.index.year == year - 1].mean()
        rows.append({"forecast": "Simple rule (last year's average month)", "year": year,
                     "miss": (actual - forecast).abs().sum() / actual.sum() * 100})
        for model, name in (("linear", "Linear model"), ("trees", "Gradient-boosted trees")):
            g = backtest[(backtest["model"] == model) & (backtest["year"] == year)]
            n = g.groupby("target_month")[["actual", "forecast"]].sum()
            rows.append({"forecast": name, "year": year,
                         "miss": (n["actual"] - n["forecast"]).abs().sum() / n["actual"].sum() * 100})
    return pd.DataFrame(rows).pivot(index="forecast", columns="year", values="miss")


def range_coverage(backtest: pd.DataFrame) -> pd.Series:
    """Ranges set on the first test year, checked on the second: share of months inside, by level."""
    linear = backtest[backtest["model"] == "linear"]
    first, second = sorted(linear["year"].unique())[:2]
    ranges = ranges_from(aggregate_errors(linear[linear["year"] == first]))
    return coverage(aggregate_errors(linear[linear["year"] == second]), ranges)["share inside"] * 100


def load_results() -> dict:
    missing = [f for f in NEEDED if not (INTERIM_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in data/interim. Run src/backtest.py, src/shortfall.py and src/routes.py "
                         "first (or python src/run_all.py).")
    backtest = pd.read_parquet(INTERIM_DIR / "step4_backtest.parquet")
    core = load_core()
    gaps = pd.read_csv(INTERIM_DIR / "step5_gaps.csv")
    return {
        "scores": model_scores(backtest, core),
        "coverage": range_coverage(backtest),
        "n_series": int(backtest[backtest["year"] == backtest["year"].max()]["series"].nunique()),
        "gaps": gaps[gaps["model"] == "linear"],
        "gaps_trees": gaps[gaps["model"] == "trees"],
        "indicators": pd.read_csv(INTERIM_DIR / "step6_indicators.csv"),
        "route_gaps": pd.read_csv(INTERIM_DIR / "step6_route_gaps.csv"),
        "elimination": pd.read_csv(INTERIM_DIR / "step6_elimination.csv").set_index("period"),
        "adapting": pd.read_csv(INTERIM_DIR / "step6_adapting.csv"),
        "engines": pd.read_csv(INTERIM_DIR / "step6_engines.csv").set_index("period"),
        "all_rows": pq.ParquetFile(ROOT / "data" / "processed" / "imports_clean.parquet").metadata.num_rows,
        "last_import": core["date"].max(),
        "suppliers": load_suppliers(),
    }


def load_suppliers() -> dict | None:
    """Results of the supplier-concentration extra, if it has been run."""
    files = {k: INTERIM_DIR / f"extra_suppliers_{k}.csv" for k in ("groups", "effects", "switching")}
    if not all(f.exists() for f in files.values()):
        return None
    return {k: pd.read_csv(f) for k, f in files.items()}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
SHORT_NAMES = {"Machinery and electrical": "machinery", "Vehicles and transport": "vehicles"}


def short_name(section: str) -> str:
    return SHORT_NAMES.get(section, section.lower())


def write(r: dict) -> str:
    gaps, ind, rg, el, ad, eng = (r["gaps"], r["indicators"], r["route_gaps"], r["elimination"], r["adapting"],
                                  r["engines"])
    periods = list(dict.fromkeys(gaps["period"]))
    since = [p for p in periods if p.startswith("Since")][0]
    war = [p for p in periods if p not in ("Jan-Feb", since)]
    latest = war[-1]
    since_short = short_period(since)
    nat = gaps[gaps["level"] == "National total"].set_index("period")
    total = nat.loc[since]
    trees_total = r["gaps_trees"][(r["gaps_trees"]["level"] == "National total")].set_index("period").loc[since]

    # Sections and regions, as in step 5
    sec = gaps[(gaps["level"] == "Section") & (gaps["period"] == since)]
    sec = sec[(sec["forecast"] / total["forecast"] * 100 >= MIN_SHARE) & ~sec["label"].isin(SKIP_SECTIONS)]
    sec = sec.sort_values("gap_pct")
    top_lost = sec.sort_values("lost", ascending=False).head(2)
    top_lost_share = top_lost["lost"].sum() / total["lost"] * 100
    reg = gaps[(gaps["level"] == "Origin region") & (gaps["period"] == since) & (gaps["label"] != "Other and not specified")]
    reg = reg.sort_values("gap_pct").set_index("label")

    def ind_pon(name, period):
        row = ind[(ind["indicator"] == name) & (ind["period"] == period)]
        return row["pct_of_normal"].iloc[0] if len(row) else float("nan")

    def route(level, name, period, col="gap_pct"):
        return rg[(rg["level"] == level) & (rg["name"] == name) & (rg["period"] == period)][col].iloc[0]

    origin_lvl, weight_lvl = "Origin route", "Beyond the Middle East, by weight"
    after = [p for p in dict.fromkeys(ind["period"]) if p not in periods and p != "Jan-Feb"]
    after_label = after[-1] if after else None
    nat_ad = ad[ad["level"] == "National total"].iloc[0]
    secs_ad = ad[(ad["level"] == "Section") & ad["label"].isin(sec["label"])]
    regs_ad = ad[(ad["level"] == "Origin region") & (ad["label"] != "Other and not specified")]
    still = secs_ad[~secs_ad["latest_inside_range"] & (secs_ad["latest"] < 100)].sort_values("latest")
    regs_still = regs_ad[~regs_ad["latest_inside_range"] & (regs_ad["latest"] < 100)].sort_values("latest")
    regs_inside = regs_ad[regs_ad["latest_inside_range"]].sort_values("forecast", ascending=False)
    eall = el.loc[since]
    gcc_row = regs_ad[regs_ad["label"] == "GCC"]
    gcc_above = (f"; goods from GCC countries were above normal ({pon(gcc_row['latest'].iloc[0])})"
                 if len(gcc_row) and not gcc_row["latest_inside_range"].iloc[0] and gcc_row["latest"].iloc[0] > 100
                 else "")
    scores, cov = r["scores"], r["coverage"]
    years = list(scores.columns)
    final_year = years[-1]
    lin, simple = scores.loc["Linear model", final_year], scores.loc["Simple rule (last year's average month)", final_year]
    cont, air = "Containers unloaded", "Air cargo and mail unloaded"
    light_am, heavy_am = eng.loc["Apr-May", "gap_pct_without_engines"], route(weight_lvl, "Heavy", "Apr-May")

    scores_md = md_table(pd.DataFrame({
        "Forecast": [f"**{f}**" if f == "Linear model" else f for f in scores.index],
        **{f"Miss in {y}": [f"{v:.1f}%" for v in scores[y]] for y in years},
    }).sort_values(f"Miss in {final_year}"))
    periods_md = md_table(pd.DataFrame({
        "Period": [f"**{p}**" for p in [*periods[:-1], *after]],
        "What was happening": [period_event(p) for p in [*periods[:-1], *after]],
    }))
    nat_md = md_table(pd.DataFrame({
        "Period": [*periods[:-1], f"**{since}**"],
        "Forecast (BD m)": [bd_m(nat.loc[p, "forecast"]) for p in periods],
        "Actual (BD m)": [bd_m(nat.loc[p, "actual"]) for p in periods],
        "Gap": [pct(nat.loc[p, "gap_pct"]) for p in periods],
        "Normal range": [f"+/-{nat.loc[p, 'range_pct']:.0f}%" for p in periods],
        "Real shortfall?": ["**yes**" if nat.loc[p, "outside"] and nat.loc[p, "gap_pct"] < 0
                            else "no" for p in periods],
    }))
    hit_md = md_table(pd.DataFrame({
        "Section": sec["label"].head(5).tolist(),
        "Gap": [pct(v) for v in sec["gap_pct"].head(5)],
        "Lost (BD m)": [bd_m(v) for v in sec["lost"].head(5)],
    }))
    reg_md = md_table(pd.DataFrame({
        "Origin region": [region_text(l)[0].upper() + region_text(l)[1:] for l in reg.index],
        "Gap": [pct(v) for v in reg["gap_pct"]],
        "Real change?": ["yes" if o else "no (inside the normal range)" for o in reg["outside"]],
    }))
    routes_md = md_table(pd.DataFrame({
        "Route": ["Containers unloaded at the port", "Vehicles unloaded from car carriers", "Air cargo and mail",
                  "Core goods arriving, by weight"],
        **{p: [pon(ind_pon(n, p)) for n in [cont, "Vehicles unloaded from car carriers", air,
                                            "Core goods imported, by weight"]] for p in [*war, *after]},
    }))

    sup = r["suppliers"]
    if sup is not None:
        sg = sup["groups"][sup["groups"]["period"] == since].set_index(["level", "name"])
        se = sup["effects"][sup["effects"]["weighting"] == "size"].set_index("effect")
        sw = sup["switching"][sup["switching"]["scope"] == "Far-dependent products"].set_index("year")
        by_count = sg.xs("Supplier countries")
        by_far = sg.xs("From beyond the Middle East")
        count_pons = by_count["actual"] / by_count["forecast"] * 100
        far_pons = by_far["actual"] / by_far["forecast"] * 100
        se_s, se_b = se.loc["suppliers"], se.loc["beyond"]
        war_year = int(sw.index.max())
        normal_sw = sw.drop(index=war_year)["minor_share"]
        suppliers_row = (f"| Did relying on few supplier countries matter? | **No clear effect.** Products bought from one or "
                         f"two countries fell about as far as products with many suppliers. **What mattered was distance**: "
                         f"each extra 10% of a product from beyond the Middle East meant about {-se_b['points']:.0f} points "
                         f"further below normal. |\n")
        suppliers_section = f"""
---

## 7. Extra: did relying on few supplier countries matter?

![Suppliers](figures/extra_suppliers_1_groups.png)

* **The number of supplier countries made no measurable difference.** Grouped by how many countries they relied on,
  products fell to between {pon(count_pons.min())} and {pon(count_pons.max())} of normal, in no particular order.
  Allowing for distance, each doubling of supplier countries changed the result by {se_s['points']:+.1f} points
  (likely {se_s['low']:+.1f} to {se_s['high']:+.1f}).
* **Distance did.** Products bought mostly nearby were at {pon(far_pons[BEYOND_GROUPS[0]])} of normal;
  products almost entirely from beyond the Middle East at {pon(far_pons[BEYOND_GROUPS[-1]])}. Each extra
  10% from beyond the Middle East meant about {-se_b['points']:.0f} points further below normal.
* **Some switching:** {sw.loc[war_year, 'minor_share']:.0f}% of far-dependent products' imports came from countries that had
  been minor suppliers, against {normal_sw.min():.0f}-{normal_sw.max():.0f}% in normal years, mostly other distant
  countries.
* **The lesson:** diversify routes, not just countries.
"""
        suppliers_link = "| Extra | [Supplier concentration](extra_suppliers.md) | `src/suppliers.py` |\n"
        next_items = """1. **A "what if" range for the road's own share**, using reasonable assumptions about how much a container holds and how
   much general cargo was imports.
2. **Update with August onwards** when the portal publishes it, the first months with Hormuz closed and no reopening."""
    else:
        suppliers_row = suppliers_section = suppliers_link = ""
        next_items = """1. **A "what if" range for the road's own share**, using reasonable assumptions about how much a container holds and how
   much general cargo was imports.
2. **Did products that relied on only a few supplier countries get hit harder?**
3. **Update with August onwards** when the portal publishes it, the first months with Hormuz closed and no reopening."""

    text = f"""# Shock and Recovery: Bahrain's import supply chains in 2026

*The whole project in one read. Generated by `src/summary.py` from the results of steps 4 to 6; the detailed reports
are linked at the end.*

On 28 February 2026, war in the Gulf closed the Strait of Hormuz and Bahrain's airspace. Everything that reaches
Bahrain by sea from outside the Gulf passes through that strait. This project uses Bahrain's official monthly import data to answer three
questions:

1. How far did imports fall below what they would have been without the war?
2. Which supply chains were hit hardest, and which routes broke?
3. How far had they come back, and how?

## The short answer

| Question | Answer |
|---|---|
| How far did imports fall? | **{pct(nat.loc['March', 'gap_pct'])} in March**, {pct(nat.loc['Apr-May', 'gap_pct'])} in April-May and {pct(nat.loc['Jun-Jul', 'gap_pct'])} in June-July, against a forecast of imports without the war. About **BD {bd_m(total['lost'])} million** of everyday goods didn't arrive from {since_short} ({pct(total['gap_pct'])}); likely between BD {bd_m(total['lost_low'])} and {bd_m(total['lost_high'])} million. |
| Which supply chains were hit hardest? | **{sec.iloc[0]['label']}** ({pct(sec.iloc[0]['gap_pct'])}). {short_name(top_lost.iloc[0]['label']).capitalize()} and {short_name(top_lost.iloc[1]['label'])} together were {top_lost_share:.0f}% of the money lost. By origin, **{region_text(reg.index[0])} ({pct(reg['gap_pct'].iloc[0])}) and {region_text(reg.index[1])} ({pct(reg['gap_pct'].iloc[1])})**; goods from **GCC countries held up** ({pct(reg.loc['GCC', 'gap_pct'])}). |
| Which routes broke? | **Container ships** (about a third of the usual containers from March to {pd.Timestamp(2026, int(r['last_import'].month), 1):%B}), **car carriers** (none at all from April) and, in March, **planes**. |
| How did goods still get in? | **By other routes.** Containers and planes could have carried at most {eall['containers_air_max_share']:.0f}% of what arrived (by weight), so **at least {eall['other_min_share']:.0f}% came over the King Fahd Causeway or by general cargo ship**. Light goods from far away held up while planes could fly: {pct(light_am)} in April-May (leaving out aircraft engines), against {pct(heavy_am)} for heavy goods. |
| How far had they come back? | **About {share_words(nat_ad['came_back_pct'])} of the way** by {latest} (from {pon(nat_ad['worst'])} of normal to {pon(nat_ad['latest'])}). Vehicles, clothing, East Asia and the Americas lagged.{" Port figures suggest supply chains kept adapting after Hormuz closed again on 11 July." if after_label else ""} |
{suppliers_row}| How do we know it's not normal ups and downs? | The forecast was **tested on {and_list([str(y) for y in years])} as if it were January**: nationally it missed by {lin:.1f}% a month in {final_year}. Its normal ranges held {cov.min():.0f}-{cov.max():.0f}% of the time in testing, and every war-time gap is far outside them. |

## Words used in this summary

{glossary.table(7, ["core_goods", "national_total", "origin_region", "section", "series", "forecast", "miss",
                    "normal_range", "gap", "real_shortfall", "lost", "since_war", "normal", "pct_of_normal", "route",
                    "causeway", "elimination", "came_back", "adapting", "units"])}

---

## Before you start: the idea in four steps

1. **Forecast imports without the war.** A model learns from 2021-2025 how monthly imports usually move, then, standing
   at the end of December 2025, forecasts every month of 2026 as if nothing had happened.
2. **Test the forecast on normal years first.** The same model is run on 2024 and 2025 as if it were January of each
   year. How far off it was sets a **normal range** for every total.
3. **Compare 2026 with the forecast.** A gap only counts as a **real shortfall** if it's bigger than the normal range.
4. **Look at the routes.** Port and airport figures show which routes broke. The road has no figures, so its share is
   worked out by **elimination**.

The periods used throughout:

{periods_md}

---

## 1. The data

* **Imports:** {r['all_rows'] / 1e6:.2f} million rows, one per month, 8-digit product code and origin country, from
  January 2021 to {r['last_import']:%B %Y}, from the Bahrain Open Data Portal.
* **Routes:** monthly figures for Khalifa Bin Salman Port (containers, ship visits, vehicles from car carriers, general
  cargo) and Bahrain International Airport (air cargo), which run two months ahead of the import data.
* **Three groups, analysed separately:** *core goods* (the everyday supply chains, about three-quarters of value);
  *bulk materials* (iron ore, alumina, coke, construction minerals: most of the weight, for a few factories); and
  *precious metals*, which follow the gold price. This summary is about core goods.
* **Cleaning:** six yearly files in three formats combined into one table; repeated shipments added together; totals
  checked month by month against the portal; port general cargo figures typed with a decimal point for a thousands
  separator repaired and marked.

---

## 2. The model: forecasting imports without the war

![Model comparison](figures/step4_2_model_comparison.png)

**What we did:** split core goods into {r['n_series']} series (one HS2 chapter from one origin region) and compared four
forecasts, each tested on {and_list([str(y) for y in years])} as if it were January:

1. the **simple rule** (every month like last year's average month);
2. **the same month last year**;
3. a **linear model** (ridge regression) written from scratch in NumPy, solved two ways: exactly (normal equations) and
   by gradient descent, which agree to computer precision;
4. **gradient-boosted trees** (scikit-learn).

**What we found (national miss, % of actual imports):**

{scores_md}

1. **The linear model won** in both years, at every level. Most of its gain comes from getting the level right while
   imports were growing.
2. **The trees never beat the linear model**, and in the {final_year} final check they did no better than the simple rule
   ({scores.loc['Gradient-boosted trees', final_year]:.1f}%): with five years of noisy monthly data, their flexibility
   mostly fitted noise.
3. **The ranges work:** set on {years[0]} and checked on {final_year}, {cov.min():.0f}-{cov.max():.0f}% of months fell inside
   them (the target was 90%).
4. **It got January-February 2026 right**, before the war: {pct(nat.loc['Jan-Feb', 'gap_pct'])}, inside the normal range.

**Why it matters:** the whole project rests on this forecast, so it had to be tested before being trusted with 2026.

---

## 3. How far below normal were imports?

![National](figures/step5_1_national.png)

{nat_md}

* **March was the worst month.** Each period after that was better, but every war-time period was a real shortfall.
* **About BD {bd_m(total['lost'])} million was lost from {since_short}** ({abs(total['gap_pct']):.0f}% of what was expected).
* **A second model agrees:** the gradient-boosted trees put the national gap at {pct(trees_total['gap_pct'])}.

---

## 4. Which supply chains were hit hardest?

**The five sections that fell furthest ({since_short}):**

{hit_md}

**By origin ({since_short}):**

{reg_md}

The pattern is about **routes**: goods made far away and shipped through Hormuz fell furthest, while goods from GCC
countries, which can come by road, held up.

---

## 5. Which routes broke, and how did goods get around them?

![Route groups](figures/step6_2_route_groups.png)

**Routes (% of the same months in 2025):**

{routes_md}

1. **Containers fell to about a third, and car carriers stopped.** Planes were grounded in March and came back slowly,
   never above normal.
2. **Goods with another way in held up.** Goods from GCC countries were {pct(route(origin_lvl, 'GCC', since))} against the
   forecast; goods from beyond the Middle East {pct(route(origin_lvl, 'Beyond the Middle East', since))}. In April-May,
   with the airspace open but Hormuz shut, light goods from beyond the Middle East were {pct(light_am)} (leaving out
   aircraft engines) against {pct(heavy_am)} for heavy goods.

![Elimination](figures/step6_3_elimination.png)

3. **At least {eall['other_min_share']:.0f}% came by road or general cargo ship.** By weight, imports were at
   {pon(eall['weight_pct_of_2025'])} of their 2025 level while containers ran at {pon(eall['containers_pct_of_2025'])}.
   Even assuming every import had come in a container in 2025, containers and planes could have carried at most
   {eall['containers_air_max_share']:.0f}% of what arrived; the rest, about {eall['other_min_t'] / 1000:,.0f} thousand tonnes,
   must have come over the causeway or by general cargo ship. Cars made in East Asia and Europe kept arriving from May,
   with no car carrier calling.

---

## 6. How far had supply chains come back?

![Adapting](figures/step6_4_adapting.png)

* **Nationally, about {share_words(nat_ad['came_back_pct'])} of the way** by {latest}: from {pon(nat_ad['worst'])} of normal in
  {nat_ad['worst_period']} to {pon(nat_ad['latest'])}.
* **Back inside the normal range:** {and_list(regs_inside['label'].map(region_text))} among origins{gcc_above}.
* **Still far below:** {and_list(f"{l.lower()} ({pon(v)})" for l, v in zip(still['label'].head(4), still['latest'].head(4)))};
  {and_list(f"{region_text(l)} ({pon(v)})" for l, v in zip(regs_still['label'], regs_still['latest']))}.
* **Part of the bounce came when Hormuz briefly reopened** (18 June to 11 July), so it isn't all adapting.{f'''
* **After July:** with Hormuz closed again, containers unloaded rose from {pon(ind_pon(cont, latest))} of normal in {latest}
  to {pon(ind_pon(cont, after_label))} in {after_label}, a sign that supply chains kept finding other ways in.''' if after_label else ""}

{suppliers_section}
---

## What this project can't tell us

1. **The road has no figures**, so its share is a minimum worked out by elimination, not a measurement.
2. **The normal ranges come from two test years.** More normal years would make them more reliable.
3. **One kind of forecast decides the answer.** The trees give almost the same result, but both learn from the same
   five years.
4. **The port's general cargo figures needed repairs**, so conclusions don't lean on them.

## What I'd do next

{next_items}

## The detailed reports

| Step | Report | Script |
|---|---|---|
| 2. Clean | [Cleaning report](step2_cleaning_report.md) | `src/build_dataset.py` |
| 3. Explore | [Exploration](step3_exploration.md) | `src/explore.py` |
| 4. Model | [Forecasting models](step4_models.md) | `src/backtest.py` |
| 5. Shortfall | [The 2026 shortfall](step5_shortfall.md) | `src/shortfall.py` |
| 6. Routes | [Routes and adapting](step6_routes.md) | `src/routes.py` |
{suppliers_link}
*Source: Bahrain Open Data Portal (iGA), www.data.gov.bh. All analysis is my own and is not attributed to the iGA.*
"""
    return text


def main() -> int:
    results = load_results()
    REPORT_PATH.write_text(write(results), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
