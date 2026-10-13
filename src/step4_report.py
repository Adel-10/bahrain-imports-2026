"""Charts and the written report for step 4 (called by backtest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from backtest import (CHECK_YEAR, LEVELS, RANGE_COVERAGE, REPORT_LEVELS, REPORT_PATH, TUNE_YEAR, aggregate_errors,
                      describe, totals)
import glossary
from build_dataset import md_table
from explore import (AQUA, BASELINE, BLUE, INK, INK2, MUTED, ORANGE, SURFACE, YELLOW, add_axes, dot_handle, end_dot,
                     end_labels, legend, line_handle, month_axis, new_figure, rounded_bar, save, source_note, tidy)
from features import CATEGORICAL, NUMERIC
from models import LinearModel, clipped_target
from reference import HS_SECTIONS

RED = "#e34948"
FAMILY_LABELS = {"Simple rule": "Simple rule: last year's average month", "Same month": "Same month last year",
                 "Linear": "Linear model (from scratch)", "Trees": "Gradient-boosted trees"}
FAMILY_COLOURS = {"Simple rule": BLUE, "Same month": ORANGE, "Linear": AQUA, "Trees": YELLOW}
SECTION_NAMES = {num: name for num, _, name, _, _ in HS_SECTIONS}
PLAIN = {
    "horizon": "Months ahead", "month": "Month of the year", "days_in_month": "Days in the month",
    "ramadan_days": "Days of Ramadan in the month", "eid_al_fitr": "Eid al-Fitr in the month",
    "eid_al_adha": "Eid al-Adha in the month", "december": "December", "covid_target": "COVID rules in the month",
    "covid_window": "COVID rules in the last 12 months", "log_scale": "Size of the series",
    "last_1": "Last month", "mean_3": "Average of the last 3 months", "mean_6": "Average of the last 6 months",
    "prev_12": "Average month in the year before", "growth": "Growth over the last year",
    "volatility": "How jumpy the series is", "zero_share": "Share of months with no imports",
    "same_month_1y": "Same month last year", "same_month_2y": "Same month two years ago",
}


def signed(x: float, dp: int = 0) -> str:
    r = round(x, dp)
    return "0" if r == 0 else f"{r:+.{dp}f}"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_national(results: dict, core: pd.DataFrame) -> None:
    choice = results["choice"]
    win_key = (choice["winner"], describe(choice["best"][choice["winner"]]))
    actual = core[(core["date"] >= "2023-01-01") & (core["date"] <= "2025-12-01")].groupby("date")["value_bd"].sum() / 1e6
    fc = {name: pd.concat([totals(results["frames"][y][key], [])["forecast"] for y in (TUNE_YEAR, CHECK_YEAR)]) / 1e6
          for name, key in (("model", win_key), ("simple", ("Simple rule", "")))}

    fig = new_figure(10, 5.2, "The linear model follows the level of imports more closely than the simple rule",
                     "Core goods imports per month (BD million). Each forecast was made at the end of the previous December,\n"
                     "using only data known by then.")
    legend(fig, [line_handle(BLUE), line_handle(ORANGE), line_handle(BASELINE)],
           ["Actual", "Linear model forecast", "Simple rule forecast (last year's average month)"], top=1.25, ncol=3)
    ax = add_axes(fig, left=0.75, right=1.6, top=1.75, bottom=0.8)
    tidy(ax)
    ax.plot(actual.index, actual, color=BLUE, zorder=4)
    for year in (TUNE_YEAR, CHECK_YEAR):
        for series, colour, z in ((fc["simple"], BASELINE, 2), (fc["model"], ORANGE, 3)):
            part = series[series.index.year == year]
            ax.plot(part.index, part, color=colour, zorder=z)
    ax.axvline(pd.Timestamp("2023-12-16"), color=BASELINE, lw=1)
    ax.axvline(pd.Timestamp("2024-12-16"), color=BASELINE, lw=1)
    ymin = min(actual.min(), fc["model"].min()) * 0.85
    ax.set_ylim(ymin, actual.max() * 1.08)
    for year, label in ((TUNE_YEAR, "2024 test (used to choose)"), (CHECK_YEAR, "2025 test (final check)")):
        ax.text(pd.Timestamp(f"{year}-01-10"), actual.max() * 1.06, label, fontsize=8.5, color=MUTED, va="top")
    ax.set_xlim(actual.index.min() - pd.Timedelta(days=15), actual.index.max() + pd.Timedelta(days=15))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    month_axis(ax, months=(1, 4, 7, 10))
    last = actual.index[-1]
    for series, colour, label in ((actual, BLUE, "Actual"), (fc["model"], ORANGE, "Model"), (fc["simple"], BASELINE, "Simple rule")):
        end_dot(ax, last, series.loc[last], colour)
    end_labels(ax, [(last, actual.loc[last], f"Actual {actual.loc[last]:,.0f}"),
                    (last, fc["model"].loc[last], f"Model {fc['model'].loc[last]:,.0f}"),
                    (last, fc["simple"].loc[last], f"Simple rule {fc['simple'].loc[last]:,.0f}")])
    source_note(fig)
    save(fig, "step4_1_national_backtest.png")


def fig_comparison(check: pd.DataFrame, first_year: int) -> None:
    data = check[(check["year"] == CHECK_YEAR) & (check["first_year"] == first_year)].set_index("family")
    families = ["Simple rule", "Same month", "Linear", "Trees"]
    levels = list(LEVELS)
    fig = new_figure(10, 5.2, "In the 2025 final check, the linear model missed least at every level",
                     "Average monthly miss of each forecast in 2025, as % of actual imports (lower is better). Core goods.")
    short = {"Simple rule": "Simple rule", "Same month": "Same month last year", "Linear": "Linear model",
             "Trees": "Gradient-boosted trees"}
    legend(fig, [Rectangle((0, 0), 1, 1, color=FAMILY_COLOURS[f], lw=0) for f in families],
           [short[f] for f in families], top=1.0, ncol=4)
    w, h = fig.get_size_inches()
    panel_w, gap, left = 2.05, 0.3, 0.6
    for k, level in enumerate(levels):
        ax = fig.add_axes([(left + k * (panel_w + gap)) / w, 0.75 / h, panel_w / w, 2.6 / h])
        values = data.loc[families, level].to_numpy()
        ax.set_xlim(-0.6, len(families) - 0.4)
        ax.set_ylim(0, values.max() * 1.25)
        tidy(ax)
        ax.set_yticks([])
        ax.grid(False)
        for i, (fam, v) in enumerate(zip(families, values)):
            rounded_bar(ax, 0, v, i, 0.62, FAMILY_COLOURS[fam], horizontal=False)
            ax.text(i, v + values.max() * 0.03, f"{v:.1f}", ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks([])
        ax.set_title(level, loc="left", fontsize=9.5, color=INK, pad=6)
    source_note(fig)
    save(fig, "step4_2_model_comparison.png")


def coefficient_table(model: LinearModel) -> pd.DataFrame:
    names = model.feature_names()
    coef = pd.Series(model.coef_, index=names).drop("intercept")
    return coef


def fig_coefficients(model: LinearModel) -> pd.Series:
    coef = coefficient_table(model)
    numeric = coef[[n for n in coef.index if n in PLAIN]]
    top = numeric.reindex(numeric.abs().sort_values(ascending=False).index[:10])
    n = len(top)
    fig = new_figure(10, 1.75 + 0.42 * n + 0.8, "What the linear model relies on most",
                     "How much the forecast moves (in % of a series' average month) when each clue goes up by a typical amount.\n"
                     "Blue pushes the forecast up, red pulls it down. Clues are measured at the end of the previous December.")
    ax = add_axes(fig, left=3.2, right=0.6, top=1.4, bottom=0.8)
    lim = top.abs().max() * 1.3 * 100
    ax.set_xlim(-lim, lim)
    ax.set_ylim(n - 0.5, -0.5)
    tidy(ax, grid_axis="x")
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    for i, (name, value) in enumerate(top.items()):
        v = value * 100
        rounded_bar(ax, 0, v, i, 0.5, BLUE if v >= 0 else RED)
        ax.text(v + (lim * 0.02 if v >= 0 else -lim * 0.02), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([PLAIN[name] for name in top.index], color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    source_note(fig)
    save(fig, "step4_3_coefficients.png")
    return top


def gradient_descent_runs(train: pd.DataFrame, ridges=(1.0, 0.01), n_iter: int = 300) -> dict:
    """Loss after each gradient descent step, relative to the exact (normal equations) minimum."""
    out = {}
    for ridge in ridges:
        exact = LinearModel(ridge=ridge).fit(train)
        gd = LinearModel(ridge=ridge, method="gd", n_iter=n_iter).fit(train)
        X = exact._design(train)
        y = clipped_target(train)
        w = train["scale"].to_numpy(dtype=float)
        w = w / w.sum()
        penalty = np.full(X.shape[1], ridge)
        penalty[0] = 0.0
        best = LinearModel.loss(X, y, w, penalty, exact.coef_)
        history = np.array(gd.loss_history)
        out[ridge] = {"gap": (history - best) / best, "max_coef_diff": float(np.abs(gd.coef_ - exact.coef_).max()),
                      "within_0.1%": int(np.argmax(history <= best * 1.001)) + 1 if (history <= best * 1.001).any() else None,
                      "exact": int(np.argmax((history - best) / best <= 1e-14)) + 1 if ((history - best) / best <= 1e-14).any() else None}
    return out


def fig_gradient_descent(runs: dict, chosen_ridge: float) -> None:
    fig = new_figure(10, 5.0, "Gradient descent reaches the same answer as the normal equations",
                     "How far the loss after each gradient descent step is above the exact minimum from the normal equations,\n"
                     "as a fraction of that minimum (log scales). A bigger ridge penalty makes the problem easier, so descent gets there faster.")
    colours = {chosen_ridge: BLUE}
    others = [r for r in runs if r != chosen_ridge]
    for r, c in zip(others, [ORANGE, AQUA]):
        colours[r] = c
    labels = {r: f"Ridge penalty {r:g}" + (" (chosen)" if r == chosen_ridge else "") for r in runs}
    legend(fig, [line_handle(colours[r]) for r in runs], [labels[r] for r in runs], top=1.25, ncol=len(runs))
    ax = add_axes(fig, left=0.9, right=0.4, top=1.75, bottom=0.8)
    tidy(ax)
    floor = 1e-14                 # differences this small are rounding error in the computer's arithmetic
    for r, run in runs.items():
        gap = np.maximum(run["gap"], floor)
        ax.plot(np.arange(1, len(gap) + 1), gap, color=colours[r], zorder=3)
        if run["gap"].min() <= floor:
            first = int(np.argmax(run["gap"] <= floor)) + 1
            ax.annotate("exact, to computer precision", (first, floor), xytext=(8, 10), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor / 3, 1e1)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "1" if round(np.log10(v)) == 0 else f"$10^{{{round(np.log10(v))}}}$"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Gradient descent step", color=INK2)
    source_note(fig, "Fitted on the final training table (2021-2025). Source: Bahrain Open Data Portal (iGA). Analysis: own.")
    save(fig, "step4_4_gradient_descent.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write(results: dict, core: pd.DataFrame, calendar: pd.DataFrame) -> None:
    choice = results["choice"]
    first_year = choice["first_year"]
    tuning, check = results["tuning"], results["check"]
    best = choice["best"]
    details = choice["details"][first_year]

    fig_national(results, core)
    fig_comparison(check, first_year)
    fig_coefficients(results["model"])
    runs = gradient_descent_runs(results["train_2026"])
    fig_gradient_descent(runs, best["Linear"]["ridge"])

    key = "Reported levels (average)"
    score_cols = {"National total": "National", "Origin region": "By origin region", "Section": "By section",
                  key: "Overall score"}

    def too_low_high(v: float) -> str:
        r = round(v, 1)
        return "about right" if r == 0 else f"{abs(r):.1f}% too {'low' if r < 0 else 'high'}"

    def simple_table(df: pd.DataFrame, label_col: str, labels) -> str:
        out = pd.DataFrame({label_col: labels})
        for src, name in score_cols.items():
            out[name] = [f"{v:.1f}%" for v in df[src]]
        out["Total for the year"] = [too_low_high(v) for v in df["Bias %"]]
        return md_table(out)

    def family_rows(df):
        rows = []
        for fam in ("Simple rule", "Same month", "Linear", "Trees"):
            part = df[df["family"] == fam]
            if fam in best:
                part = part[part["settings"] == describe(best[fam])]
            rows.append(part.iloc[0])
        return pd.DataFrame(rows).set_index("family")

    tune_main = tuning[tuning["first_year"] == first_year]
    check_rows = {y: family_rows(check[(check["year"] == y) & (check["first_year"] == first_year)])
                  for y in (TUNE_YEAR, CHECK_YEAR)}
    lin = {y: check_rows[y].loc["Linear"] for y in check_rows}
    simple = {y: check_rows[y].loc["Simple rule"] for y in check_rows}
    same = {y: check_rows[y].loc["Same month"] for y in check_rows}
    trees = {y: check_rows[y].loc["Trees"] for y in check_rows}
    families = ["Simple rule", "Same month", "Linear", "Trees"]
    family_names = ["Simple rule (last year's average month)", "Same month last year", "**Linear model**",
                    "Gradient-boosted trees"]

    def tree_label(params: dict) -> str:
        aim = {"absolute_error": "typical month", "poisson": "average month (Poisson method)",
               "squared_error": "average month (squared-miss method)"}.get(params["loss"], params["loss"])
        size = {31: "big", 15: "medium", 4: "very small"}.get(params["max_leaf_nodes"], f"{params['max_leaf_nodes']}-leaf")
        label = f"Aims at the {aim}, {size} trees"
        if params.get("weight_power", 1.0) < 1:
            label += ", less weight on the biggest series"
        return f"**{label} (chosen)**" if params == best["Trees"] else label

    tree_rows = tune_main[tune_main["family"] == "Trees"]
    tree_labels = [tree_label(p) for p in tree_rows["params"]]

    jf_all = results["jan_feb_all"]
    jf_labels = []
    for family, params in zip(jf_all["family"], jf_all["params"]):
        if family == "Trees":
            jf_labels.append("Trees: " + tree_label(params).replace("**", ""))
        else:
            jf_labels.append({"Simple rule": "Simple rule", "Same month": "Same month last year",
                              "Linear": "**Linear model (chosen)**"}[family])
    jf_lin = jf_all[jf_all["family"] == "Linear"].iloc[0]
    jf_best_tree = jf_all[(jf_all["family"] == "Trees")
                          & (jf_all["settings"] == describe(best["Trees"]))].iloc[0]
    history_md = md_table(pd.DataFrame({
        "Test": ["2024", "2025", "January-February 2026"],
        "Years of history to learn from": [3, 4, 5],
        "Linear model (overall score)": [f"{lin[TUNE_YEAR][key]:.1f}%", f"{lin[CHECK_YEAR][key]:.1f}%", f"{jf_lin[key]:.1f}%"],
        "Best trees (overall score)": [f"{trees[TUNE_YEAR][key]:.1f}%", f"{trees[CHECK_YEAR][key]:.1f}%", f"{jf_best_tree[key]:.1f}%"],
    }))
    abs_err_tree = tree_rows[tree_rows["settings"].str.contains("absolute_error")].iloc[0]

    linear_rows = tune_main[tune_main["family"] == "Linear"]
    strength = {0.01: "very weak", 0.1: "weak", 1.0: "strong", 10.0: "very strong"}
    linear_labels = []
    for params in linear_rows["params"]:
        label = f"{params['ridge']:g} ({strength.get(params['ridge'], '')})"
        linear_labels.append(f"**{label}, chosen**" if params == best["Linear"] else label)

    history = [tuning[(tuning["first_year"] == fy) & (tuning["family"] == "Linear")
                      & (tuning["settings"] == describe(best["Linear"]))].iloc[0] for fy in (2021, 2022)]
    history_df = pd.DataFrame(history)

    ranges = results["ranges"].copy()
    calib = results["calibration"]
    nat_range = ranges.loc[ranges["level"] == "National total", "range"].iloc[0] * 100
    region_ranges = ranges[ranges["level"] == "Origin region"].sort_values("range")
    section_ranges = ranges[ranges["level"] == "Section"].copy()
    section_ranges["name"] = section_ranges["name"].astype(int).map(SECTION_NAMES)
    section_ranges = section_ranges.sort_values("range")

    errors26 = aggregate_errors(results["forecast_2026"]).merge(results["ranges"], on=["level", "name"])
    errors26["inside"] = errors26["error"].abs() <= errors26["range"]
    jf = errors26[errors26["target_month"].dt.month.isin([1, 2])]
    jf_nat = jf[jf["level"] == "National total"].copy()
    jf_nat_md = md_table(pd.DataFrame({
        "Month": jf_nat["target_month"].dt.strftime("%B %Y"),
        "Forecast (BD m)": (jf_nat["forecast"] / 1e6).map(lambda v: f"{v:,.1f}"),
        "Actual (BD m)": (jf_nat["actual"] / 1e6).map(lambda v: f"{v:,.1f}"),
        "Difference": (jf_nat["error"] * 100).map(lambda v: f"{signed(v, 1)}%"),
        "Range": (jf_nat["range"] * 100).map(lambda v: f"+/-{v:.0f}%"),
        "Inside the range?": jf_nat["inside"].map({True: "yes", False: "no"}),
    }))
    jf_detail = jf[jf["level"] != "National total"]
    jf_out = jf_detail[~jf_detail["inside"]]
    n_out_above = int((jf_out["error"] > 0).sum())
    jf_within = int(np.ceil(jf_nat["error"].abs().max() * 100))

    ranges_md = md_table(pd.concat([
        pd.DataFrame({"Total": ["**Bahrain (national)**"], "Range for a single month": [f"**+/-{nat_range:.0f}%**"]}),
        pd.DataFrame({"Total": region_ranges["name"],
                      "Range for a single month": (region_ranges["range"] * 100).map(lambda v: f"+/-{v:.0f}%")}),
    ]))
    calib_md = md_table(pd.DataFrame({
        "Level": calib.index,
        "Months checked": calib["months"].astype(int).to_numpy(),
        "Months inside the range": (calib["share inside"] * 100).map(lambda v: f"{v:.0f}%").to_numpy(),
    }))
    calib_low, calib_high = calib["share inside"].min() * 100, calib["share inside"].max() * 100

    ridge = best["Linear"]["ridge"]
    gd_chosen, gd_weak = runs[ridge], runs[min(runs)]
    coef = coefficient_table(results["model"])
    growth_allowance = (results["model"].coef_[0] - 1) * 100
    improvement = (1 - lin[CHECK_YEAR]["National total"] / simple[CHECK_YEAR]["National total"]) * 100

    text = f"""# Step 4: Building and testing the forecasting models

*Generated by `src/backtest.py` (charts and text by `src/step4_report.py`). Re-running the script refreshes every number.*

The whole project depends on one estimate: what imports **would have been in 2026 without the war**. This step builds
forecasting models to make that estimate. Before trusting them with 2026, each model is tested on two normal years,
2024 and 2025, to see how accurate it is and how far off it usually is.

## The short answer

| Question | Answer |
|---|---|
| Which forecast is most accurate in normal years? | **The linear model.** Nationally it missed by {lin[CHECK_YEAR]['National total']:.1f}% a month on average in 2025, against {simple[CHECK_YEAR]['National total']:.1f}% for the simple rule. |
| Should 2021 (the COVID year) be used? | **Yes.** The model is more accurate with it. |
| What does the model rely on? | **Mostly the recent level of imports, plus an allowance for growth.** Everything else makes only small adjustments. |
| How far off is it usually? | **A single month nationally can be up to {nat_range:.0f}% off** even in a normal year. These ranges were tested and they work. |
| Does it get January and February 2026 right? | **Yes.** Both months were within {jf_within}% of the forecast, before the war began on 28 February. |

## Words used in this report

{glossary.table(4, ["core_goods", "national_total", "origin_region", "section", "series", "normal_year", "simple_rule",
                    "clue", "weight", "miss", "overall_score", "total_for_year", "range", "units"])}

---

## Before you start: how the test works

### Step 1: Pretend it's January

Each test copies exactly what will be done for 2026:

1. Stand at the end of December 2023 and use only data known by then.
2. Forecast every month of 2024.
3. Compare the forecasts with what actually arrived.
4. Repeat from the end of December 2024 for 2025.

### Step 2: What gets forecast

- The models forecast **core goods** (everything except bulk materials and precious metals, as decided in step 3).
- Core goods are split into **series**: one HS2 chapter from one origin region.
- A series needs imports in at least 90% of past months to be forecast on its own. The patchy ones are added together into
  one "other" series per section and region.
- That gives **{details['n_series']:,} series** ({details['n_pooled']} of them "other").
- Adding the series' forecasts together gives forecasts for each section, each origin region and the whole country.

### Step 3: The clues

For each series, the models get these clues, all known at the end of December:

| Clue | What it means |
|---|---|
| Recent level | Last month, and the average of the last 3, 6 and 12 months |
| Trend | Growth over the last year, and the average month in the year before that |
| Same month before | The same month last year and two years ago |
| How regular the series is | How jumpy it is, and how often it has months with no imports |
| Size and identity | How big the series is, its section and its origin region |
| Calendar | Month of the year, December, days of Ramadan, Eid, days in the month |
| COVID | Whether COVID rules applied in the month, or in the past 12 months |
| Months ahead | How far ahead the forecast is (1 to 12 months) |

Every clue is measured relative to the series' own average month, so a small series and a big one look alike to the models.

### Step 4: The four forecasts

1. **Simple rule:** every month will be like last year's average month. This is the yardstick from step 3.
2. **Same month last year:** every month will be like the same month a year earlier.
3. **Linear model**, written from scratch with NumPy:
   - It multiplies each clue by a weight and adds them up.
   - It picks the weights that make the squared misses on past data as small as possible.
   - A **penalty** stops any weight from getting too big. That pulls the model back towards the simple rule whenever
     the evidence for a clue is weak. (The technical name is ridge regression.)
4. **Gradient-boosted trees** (from scikit-learn):
   - Many small decision trees, each one correcting the mistakes of the ones before.
   - One model learns from all the series at once.
   - This is the method behind LightGBM, often the strongest choice for tables like this one.

When the models learn, big series count for more than small ones, because what matters in the end is the miss in Bahraini dinars.

### Step 5: How the forecasts are scored

The miss works like the normal miss in step 3, with one change: it is divided by the **actual** imports, not by the
forecast. With several models to compare, the actual is the fair yardstick because it is the same for all of them.
Dividing by the forecast would let a model look better just by forecasting too high.

### Step 6: Choose on 2024, confirm on 2025

1. **Choosing (2024):** every choice was made using the 2024 test only: each model's settings, and whether to use 2021.
2. **Final check (2025):** 2025 was used once, at the end, with the choices already fixed. Nothing was tuned on it, so it
   shows how the models do on a year they haven't seen. That is the situation 2026 will be in.

---

## 1. Which forecast is most accurate?

**What we did:** ran all four forecasts on 2024 and 2025.

**What we found:**

**2025: the final check**

{simple_table(check_rows[CHECK_YEAR].loc[families], "Forecast", family_names)}

**2024: used for choosing**

{simple_table(check_rows[TUNE_YEAR].loc[families], "Forecast", family_names)}

![Model comparison](figures/step4_2_model_comparison.png)

1. **The linear model wins in both years, at every level.** Nationally it missed by {lin[CHECK_YEAR]['National total']:.1f}%
   in 2025, against {simple[CHECK_YEAR]['National total']:.1f}% for the simple rule: about {improvement:.0f}% smaller.
2. **Much of the gain comes from getting the level right.** Imports were growing, so the simple rule was
   {abs(simple[CHECK_YEAR]['Bias %']):.1f}% too low over 2025. The linear model was {abs(lin[CHECK_YEAR]['Bias %']):.1f}% too low.
3. **"Same month last year" is the worst in both years**, as step 3 found.

![National backtest](figures/step4_1_national_backtest.png)

**Why it matters:** the gain is modest, because most of a month's ups and downs are random and no model can predict them.
But it is consistent, and it held up on 2025, the year nothing was tuned on.

---

## 2. Why didn't the trees win?

Gradient-boosted trees are usually the strongest model for this kind of table, so it is worth explaining why they weren't here.

**Reason 1: set up the usual way, they forecast too low.**
- Trained to make the plain misses as small as possible, a model forecasts the *typical* (median) month of each series.
- In a lopsided series, a few big months sit among many small ones, so the typical month is below the average.
- Added up over hundreds of series, the trees' forecasts came out {abs(abs_err_tree['Bias %']):.1f}% too low in 2024.
- Training them to make the *squared* misses small instead makes them aim at the average month, which fixed it.

**Reason 2: there isn't enough history for their flexibility to pay off.**
- Trees can learn complicated patterns. With only four to five years of noisy monthly data, most of those patterns are noise.
- The best trees were the most restricted ones tried, with only 4 leaves each.
- In 2025 they still missed more than the linear model ({trees[CHECK_YEAR][key]:.1f}% against {lin[CHECK_YEAR][key]:.1f}%
  overall), and slightly more than the simple rule ({simple[CHECK_YEAR][key]:.1f}%).

Every tree setup tried on 2024:

{simple_table(tree_rows, "Tree setup", tree_labels)}

*Tree results can differ slightly (in the first decimal place) between scikit-learn versions; the conclusions don't change.*

---

## 3. How strong should the linear model's penalty be?

**What we did:** tried four strengths of penalty on 2024. A stronger penalty keeps the weights smaller, so the model stays
closer to the simple rule.

{simple_table(linear_rows, "Penalty strength", linear_labels)}

**What we found:** a strong penalty ({ridge:g}) works best.

**Why it matters:** apart from the recent level and growth, the evidence for most clues is weak, and a strong penalty
stops the model from chasing noise in the past data.

---

## 4. Should 2021 be used?

**What we did:** ran the chosen linear model on 2024 twice: learning from 2021-2023, and learning from 2022-2023 only.

{simple_table(history_df, "History used", ["2021-2023 (with 2021)", "2022-2023 (without 2021)"])}

**What we found:** the model is more accurate with 2021. Without it, there is only one year of history to learn from in
the 2024 test, which isn't enough to learn about trends.

**Why it matters:** 2021 stays in. The COVID clues tell the model which of its months were unusual.

---

## 5. What does the linear model rely on?

![Coefficients](figures/step4_3_coefficients.png)

*How to read the chart: each bar shows how much the forecast moves, in % of the series' average month, when that clue
goes up by a typical amount (one standard deviation). From the final model, trained on 2021-2025. Section and region
adjustments are not shown.*

1. **A built-in allowance for growth.** When every clue is at its typical value, the model forecasts {growth_allowance:.1f}%
   above the series' average month over the last 12 months. That is roughly how much imports grew from one year to the next,
   and it fixes most of the simple rule's "too low" problem.
2. **Recent months pull the forecast their way.** If last month, or the last 3 or 6 months, were higher than usual, the
   forecast goes up a little.
3. **Part of last year's change tends to unwind.** If a series was higher in the year before than it is now, the forecast
   goes up a little ({coef['prev_12'] * 100:+.1f} on the chart). If it grew fast, the forecast comes down a little
   ({coef['growth'] * 100:+.1f}).
4. **The same month two years ago** adds a little seasonal memory ({coef['same_month_2y'] * 100:+.1f}).
5. **Holidays:** a month with Eid al-Adha is about {abs(coef['eid_al_adha']) * 100:.1f}% lower. December is about
   {coef['december'] * 100:.1f}% higher.

**Why it matters:** all the adjustments are small, a few % at most. That is the penalty at work: in normal years most of a
month's movement is noise, so the model only moves away from "recent level plus growth" when the evidence is clear.
Some clues overlap (for example the last 3 and the last 6 months), so the penalty shares the credit between them, and one
weight on its own should be read with care.

---

## 6. Checking the maths: two ways to find the weights

The linear model's weights were found in two different ways, both coded from scratch:

1. **Solve it directly ("normal equations").** The best weights satisfy one matrix equation, which can be solved in one go.
2. **Walk downhill ("gradient descent").** Start with every weight at zero. At each step, work out which direction lowers
   the misses fastest, and take a small step that way. Repeat.

If the code is right, both should end up with the same weights.

![Gradient descent](figures/step4_4_gradient_descent.png)

**What we found:**

- With the chosen penalty, walking downhill gets within 0.1% of the best answer after {gd_chosen['within_0.1%']} steps,
  and matches it exactly (to the computer's precision) after {gd_chosen['exact']} steps.
- With a very weak penalty ({min(runs):g}), overlapping clues make the slope shallow near the bottom, so progress slows
  down. After {len(gd_weak['gap'])} steps it is close, but still not exact.

*For readers who know the terms: the step size is 1/L, where L is the largest eigenvalue of the Hessian 2(X'WX + P), the
safe maximum for a quadratic loss. A bigger penalty improves the conditioning of X'WX, which is why descent converges faster.*

---

## 7. How far off is the model usually?

**What we did:**

1. Took the model's monthly misses in the 2024 and 2025 tests, for the national total, each origin region and each section.
2. For each total, found the distance that 9 months out of 10 stayed within. That is its **range**, the same idea as the
   normal range in step 3.

**Does the range work?** As a fair test, ranges were first set using 2024 only, then checked against 2025:

{calib_md}

Between {calib_low:.0f}% and {calib_high:.0f}% of months fell inside, close to the 90% aimed for, so the ranges can be trusted.

**The final ranges** (set from 2024 and 2025 together):

{ranges_md}

*Sections range from +/-{section_ranges['range'].min() * 100:.0f}% ({section_ranges['name'].iloc[0]}) to
+/-{section_ranges['range'].max() * 100:.0f}% ({section_ranges['name'].iloc[-1]}). All of them are saved in `data/interim/step4_ranges.csv`.*

**Why it matters:**
- In step 5, a 2026 month only counts as a real shortfall if it falls **outside** its range.
- Nationally, a single month can be {nat_range:.0f}% off even in a normal year. That is bigger than the average miss
  ({lin[CHECK_YEAR]['National total']:.1f}%), because the range has to cover the unusual months too, such as December.
- Small or jumpy origins have wide ranges, so a change there needs to be large before it counts.

---

## 8. Sanity check: January and February 2026

**What we did:** retrained the model on 2021-2025 and forecast 2026 from the end of December 2025, exactly as step 5 will.
The war began on 28 February, so January and February should look normal.

{jf_nat_md}

**What we found:**

1. **Nationally, both months land close to the forecast**, well inside the range.
2. **{int(jf_detail['inside'].sum())} of {len(jf_detail)}** origin-region and section months were also inside their range.
3. Of the {len(jf_out)} outside it, **{n_out_above} were above the forecast**, mostly in February. That could be stockpiling
   as tensions rose before the war, which step 5 will keep in mind.

**Why it matters:** the model gets the months before the war right, which supports using it for the months after.

### Extra check: every forecast on January-February 2026 (for information only)

Trees need a lot of history, so we also tested every forecast with all five years (2021-2025) to learn from, on the only
2026 months before the war:

{simple_table(jf_all, "Forecast", jf_labels).replace("Total for the year", "Total for the two months")}

With five years of history, the best trees did about as well as the linear model. Across all three tests, though, more
history doesn't steadily favour the trees:

{history_md}

This check is **not used to change the choice**, for four reasons:

1. **Two months is too few.** A difference of a few tenths of a percent over two months is well within normal noise.
2. **February may not be a normal month.** The war began on 28 February, and several totals were higher than expected
   (possibly stockpiling).
3. **It would use up the independent check.** Right now January-February is a clean test the model was never tuned on.
4. **Picking the best of several setups on a small test flatters the winner.** One of them will come out on top by luck.

Instead, step 5 uses the linear model for the main results and the best trees as a **second opinion**. If both models
tell the same story about 2026, the findings are stronger. If they disagree somewhere, that is reported.

---

## What this means for step 5

1. **The linear model** (strong penalty, trained on 2021-2025) gives the "without the war" forecast for each month of 2026.
   The forecasts are saved in `data/interim/step4_forecast_2026.parquet`. The best trees' forecasts, the second opinion,
   are saved alongside them.
2. **Each month of 2026 is compared with its own forecast**, for the whole country, each origin region and each section.
3. **A month only counts as a real shortfall if it falls outside its range** (saved in `data/interim/step4_ranges.csv`).
4. **Bulk materials and precious metals** have too few series for these models. Step 5 uses the simple rule for them,
   with their normal ranges from step 3.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
