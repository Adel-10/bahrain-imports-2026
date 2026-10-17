"""Step 7: the whole project in one read.

    python src/summary.py          (run steps 4 to 6 first: it reads their saved results)

Pulls the main results of steps 4, 5 and 6 (and the supplier extra, if it has been run) into
reports/summary.md. Every number is read from the files those steps save in data/interim/, so
re-running the pipeline keeps the summary in step with the detailed reports. It draws no new
charts; it reuses the main ones from the step reports.

Two numbers are recomputed here rather than read: the national miss of each forecast in the
test years (``model_scores``) and the check of 2024-based ranges on 2025 (``range_coverage``).
Both come from the saved test forecasts, using the same functions as step 4.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from backtest import aggregate_errors, coverage, ranges_from
from config import ROOT
from features import load_core
from reference import period_event
from reporting import and_list, bd_m, footer, md_table, pct, pon, region_text, share_words, short_period, write_markdown
from step5_report import MIN_SHARE, SKIP_SECTIONS
from suppliers import BEYOND_GROUPS

INTERIM_DIR = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "reports" / "summary.md"
NEEDED = [                       # files the earlier steps must have written"step4_backtest.parquet", "step5_gaps.csv", "step6_indicators.csv", "step6_route_gaps.csv",
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
    """Read everything the summary needs from data/interim/ (and two figures from the cleaned data)."""
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
    """Section names as they read inside a sentence: 'Machinery and electrical' -> 'machinery'."""
    return SHORT_NAMES.get(section, section.lower())


def write(r: dict) -> str:
    """The summary report, built from the saved results of the earlier steps."""
    gaps, ind, rg, el, ad, eng = (r["gaps"], r["indicators"], r["route_gaps"], r["elimination"], r["adapting"],
                                  r["engines"])
    periods = list(dict.fromkeys(gaps["period"]))
    since = [p for p in periods if p.startswith("Since")][0]
    war = [p for p in periods if p not in ("Jan-Feb", since)]
    latest = war[-1]
    since_short = short_period(since)
    last_name = f"{pd.Timestamp(2026, int(r['last_import'].month), 1):%B}"
    nat = gaps[gaps["level"] == "National total"].set_index("period")
    total = nat.loc[since]
    trees_total = r["gaps_trees"][(r["gaps_trees"]["level"] == "National total")].set_index("period").loc[since]

    # Sections and regions, chosen as in step 5.
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
    gcc_above = (f", and goods from GCC countries were above normal ({pon(gcc_row['latest'].iloc[0])})"
                 if len(gcc_row) and not gcc_row["latest_inside_range"].iloc[0] and gcc_row["latest"].iloc[0] > 100
                 else "")
    scores, cov = r["scores"], r["coverage"]
    years = list(scores.columns)
    final_year = years[-1]
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

    # The supplier extra is optional: its finding, section and link only appear if it has been run.
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
        suppliers_finding = (f"- **Relying on few supplier countries didn't matter; relying on distant ones did.** Products "
                             f"bought from one or two countries fell about as far as products with many suppliers, while "
                             f"each extra 10% of a product from beyond the Middle East meant about {-se_b['points']:.0f} "
                             f"points further below normal. The lesson is to diversify routes, not just countries.")
        suppliers_section = f"""
## 7. Did relying on few supplier countries matter?

![Suppliers](figures/extra_suppliers_1_groups.png)

Grouped by how many countries they relied on (measured as an effective number of suppliers), products fell to
between {pon(count_pons.min())} and {pon(count_pons.max())} of normal, in no particular order. Allowing for
distance in a weighted regression, each doubling of the number of supplier countries changed a product's result by
{se_s['points']:+.1f} points (90% bootstrap range {se_s['low']:+.1f} to {se_s['high']:+.1f}), which is no measurable
effect. Distance mattered: products bought mostly nearby were at {pon(far_pons[BEYOND_GROUPS[0]])} of normal, and
products almost entirely from beyond the Middle East at {pon(far_pons[BEYOND_GROUPS[-1]])}. Buyers switched a
little: {sw.loc[war_year, 'minor_share']:.0f}% of far-dependent products' imports came from countries that had been
minor suppliers, against {normal_sw.min():.0f}-{normal_sw.max():.0f}% in normal years, mostly other distant countries.
Five suppliers in East Asia all ship through the same strait, so having five of them gave no protection.
"""
        suppliers_link = "| Extra | [Supplier concentration](extra_suppliers.md) | `src/suppliers.py` |\n"
        next_items = """- **A "what if" range for the road's own share**, under stated assumptions about container loads and how much
  general cargo was imports.
- **Update with August 2026 onwards** when the portal publishes it: the first months with Hormuz closed and no
  reopening."""
    else:
        suppliers_finding = suppliers_section = suppliers_link = ""
        next_items = """- **A "what if" range for the road's own share**, under stated assumptions about container loads and how much
  general cargo was imports.
- **Whether products that relied on only a few supplier countries were hit harder.**
- **Update with August 2026 onwards** when the portal publishes it: the first months with Hormuz closed and no
  reopening."""

    g0, g1 = pct(reg['gap_pct'].iloc[0]), pct(reg['gap_pct'].iloc[1])
    two_gaps = f"both {g0}" if g0 == g1 else f"{g0} and {g1}"
    after_finding = (f" Port figures suggest they kept adapting after Hormuz closed again on 11 July: containers "
                     f"unloaded rose from {pon(ind_pon(cont, latest))} of normal in {latest} to "
                     f"{pon(ind_pon(cont, after_label))} in {after_label}." if after_label else "")

    text = f"""
# Shock and Recovery: Bahrain's import supply chains in 2026

On 28 February 2026, war in the Gulf closed the Strait of Hormuz and, for more than a month, Bahrain's airspace.
Everything that reaches Bahrain by sea from outside the Gulf passes through that strait. I used Bahrain's official
monthly import statistics ({r['all_rows'] / 1e6:.2f} million rows, January 2021 to {r['last_import']:%B %Y}) together with
port and airport figures to measure how far imports fell below what they would otherwise have been, which supply
chains and routes broke, and how far they had come back.

## Key findings

- **Everyday imports fell {abs(nat.loc['March', 'gap_pct']):.0f}% below normal in March 2026**, then
  {abs(nat.loc['Apr-May', 'gap_pct']):.0f}% in April-May and {abs(nat.loc['Jun-Jul', 'gap_pct']):.0f}% in June-July,
  against a tested forecast of imports without the war. About BD {bd_m(total['lost'])} million of goods didn't arrive
  from March to {last_name} ({abs(total['gap_pct']):.0f}% of what was expected; likely between BD
  {bd_m(total['lost_low'])} and {bd_m(total['lost_high'])} million).
- **{sec.iloc[0]['label']} were hit hardest ({pct(sec.iloc[0]['gap_pct'])})**, and
  {short_name(top_lost.iloc[0]['label'])} and {short_name(top_lost.iloc[1]['label'])} together were
  {top_lost_share:.0f}% of the money lost. Goods from {region_text(reg.index[0])} and {region_text(reg.index[1])} fell
  furthest ({two_gaps}); goods from GCC countries held up
  ({pct(reg.loc['GCC', 'gap_pct'])}, within the normal range).
- **Container ships and car carriers were the routes that broke.** Containers unloaded ran at about a third of normal
  from March to {last_name}, and car carriers unloaded no vehicles at all from April.
- **Goods got in another way.** Containers and planes could have carried at most
  {eall['containers_air_max_share']:.0f}% of what arrived by weight, so at least {eall['other_min_share']:.0f}% came over
  the King Fahd Causeway or by general cargo ship. While planes could fly but ships couldn't pass Hormuz, light goods
  from far away fell {abs(light_am):.0f}% (leaving out aircraft engines) and heavy goods {abs(heavy_am):.0f}%.
- **Supply chains came back about {share_words(nat_ad['came_back_pct'])} of the way by {latest}.** Vehicles, clothing,
  East Asia and the Americas lagged.{after_finding}
{suppliers_finding}

![Imports against the forecast without the war](figures/step5_1_national.png)

## Approach

1. **Forecast imports without the war.** Core goods were split into about {round(r['n_series'], -1)} monthly series (one customs chapter
   from one origin region). A ridge regression written from scratch learned from 2021-2025 how these usually move and,
   standing at the end of December 2025, forecast every month of 2026 as if nothing had happened.
2. **Test the forecast on normal years first.** The same model was run on {and_list([str(y) for y in years])} as if it were
   January of each year, using only data available at the time. Its misses set a normal range for every total and
   period length.
3. **Compare 2026 with the forecast.** A gap only counts as a real shortfall if it is bigger than the normal range. A
   second model (gradient-boosted trees) gives a second opinion.
4. **Follow the routes.** Port and airport figures show which routes broke. The causeway publishes nothing, so its share
   is bounded by elimination: the most containers and planes could have carried, taken away from what arrived.

The [methodology note](../docs/methodology.md) explains each choice and its trade-offs. The periods used throughout:

{periods_md}

## 1. Data

The import data has one row per month, 8-digit product code and origin country, from the Bahrain Open Data Portal.
Six yearly files in three slightly different formats were combined into one table, repeated shipments added together,
and every month's totals checked against the portal's own figures; nothing was dropped. Imports were split into three
groups that behave very differently: core goods (the everyday supply chains, about three-quarters of value), bulk
materials (iron ore, alumina, coke and building minerals, most of the weight, for a few plants) and precious metals
(which follow the gold price). This summary is about core goods. The route data are monthly figures for Khalifa Bin
Salman Port and Bahrain International Airport, which run two months ahead of the import data. Several months of the
port's general cargo figures had a decimal point typed for a thousands separator and were repaired and marked.

## 2. The forecast

![Model comparison](figures/step4_2_model_comparison.png)

Four forecasts were compared, each tested on {and_list([str(y) for y in years])} as if it were January: the simple rule
(every month like last year's average month), the same month last year, the ridge regression (solved both with the
normal equations and by gradient descent, which agree to floating-point precision), and gradient-boosted trees
(scikit-learn).

{scores_md}

*National miss: the total absolute monthly error as % of actual imports.*

The linear model won in both years and at every level, mostly because it gets the level right while imports are
growing. The trees never beat it, and in the {final_year} final check they did no better than the simple rule
({scores.loc['Gradient-boosted trees', final_year]:.1f}%): with five years of noisy monthly data their flexibility mostly
fitted noise. The normal ranges work: set on {years[0]} and checked on {final_year}, {cov.min():.0f}-{cov.max():.0f}% of
months fell inside them, against a target of 90%. The model also got January-February 2026 right, before the war
({pct(nat.loc['Jan-Feb', 'gap_pct'])}, inside the normal range).

## 3. How far imports fell

{nat_md}

March was the worst month. Each period after that was better, but every war-time period was a real shortfall, and in
total about BD {bd_m(total['lost'])} million was lost from {since_short}. The gradient-boosted trees put the national gap
at {pct(trees_total['gap_pct'])}, against {pct(total['gap_pct'])} for the linear model.

## 4. Which supply chains were hit hardest

The five sections that fell furthest over {since_short}:

{hit_md}

By origin:

{reg_md}

The pattern is about routes more than products: goods made far away and shipped through Hormuz fell furthest, while
goods from GCC countries, which can come by road, held up.

## 5. Which routes broke, and how goods got around them

![Route groups](figures/step6_2_route_groups.png)

{routes_md}

*% of the same months of 2025.*

Containers fell to about a third of normal and car carriers stopped. Planes were grounded in March and came back slowly,
never above normal. Goods with another way in held up: goods from GCC countries were
{pct(route(origin_lvl, 'GCC', since))} against the forecast, while goods from beyond the Middle East were
{pct(route(origin_lvl, 'Beyond the Middle East', since))}. In April-May, with the airspace open but Hormuz shut, light
goods from beyond the Middle East were {pct(light_am)} (leaving out aircraft engines) against {pct(heavy_am)} for heavy
goods.

![Elimination](figures/step6_3_elimination.png)

By weight, core goods arrived at {pon(eall['weight_pct_of_2025'])} of their 2025 level while containers ran at
{pon(eall['containers_pct_of_2025'])}. Even assuming every import had come in a container in 2025, containers and
planes could have carried at most {eall['containers_air_max_share']:.0f}% of what arrived. The rest, about
{eall['other_min_t'] / 1000:,.0f} thousand tonnes, must have come over the causeway or by general cargo ship. Cars made in
East Asia and Europe kept arriving from May even though no car carrier called.

## 6. How far supply chains came back

![Adapting](figures/step6_4_adapting.png)

Nationally, imports came back about {share_words(nat_ad['came_back_pct'])} of the way by {latest}, from
{pon(nat_ad['worst'])} of normal in {nat_ad['worst_period']} to {pon(nat_ad['latest'])}. Back inside their normal range
were {and_list(regs_inside['label'].map(region_text))} among origins{gcc_above}. Still far below normal were
{and_list(f"{l.lower()} ({pon(v)})" for l, v in zip(still['label'].head(4), still['latest'].head(4)))} among sections,
and {and_list(f"{region_text(l)} ({pon(v)})" for l, v in zip(regs_still['label'], regs_still['latest']))} among origins. Part of the bounce
came while Hormuz was briefly open (18 June to 11 July), so not all of it was adapting.{f''' After July, with Hormuz
closed again, containers unloaded rose from {pon(ind_pon(cont, latest))} of normal in {latest} to
{pon(ind_pon(cont, after_label))} in {after_label}, a sign that supply chains kept finding other ways in.''' if after_label else ""}
{suppliers_section}
## Limitations

- **The causeway has no published figures**, so the road's share is a minimum worked out by elimination, not a
  measurement.
- **The normal ranges come from two test years.** More normal years would make them more reliable.
- **Both models learn from the same five years**, so their agreement is reassuring but not independent proof.
- **Customs months and arrival months don't always line up.** Periods of two months or more soften this.
- **The port's general cargo figures needed repairs** and may include goods loaded for export, so no conclusion rests on
  them alone.

## What I'd do next

{next_items}

## Detailed reports

| Step | Report | Script |
|---|---|---|
| 2. Clean | [Cleaning the data](step2_cleaning_report.md) | `src/build_dataset.py` |
| 3. Explore | [Exploring the data](step3_exploration.md) | `src/explore.py` |
| 4. Forecast | [Forecasting imports without the war](step4_models.md) | `src/backtest.py` |
| 5. Shortfall | [The 2026 shortfall](step5_shortfall.md) | `src/shortfall.py` |
| 6. Routes | [Routes and adapting](step6_routes.md) | `src/routes.py` |
{suppliers_link}
{footer("src/summary.py", " from the saved results of steps 4 to 6")}

*All analysis is my own and is not attributed to the iGA.*
"""
    return text


def main() -> int:
    """Read the saved results and write reports/summary.md."""
    results = load_results()
    write_markdown(REPORT_PATH, write(results))
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
