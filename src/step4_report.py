"""Charts and the written report for step 4 (called by backtest.py once the models are tested).

    python src/backtest.py        # runs the backtest, then calls write() in this module

Kept separate from backtest.py so the modelling code stays short and readable. Nothing here
changes a result: it only draws and describes what backtest.py computed.

Charts (reports/figures/)
-------------------------
1. step4_1_national_backtest.png  actual core imports in 2024-2025 against the linear model's and
                                  the simple rule's forecasts, each made the previous December.
2. step4_2_model_comparison.png   the 2025 miss of each forecast at each level of detail.
3. step4_3_coefficients.png       the linear model's ten largest coefficients, as % of a normal month.
4. step4_4_gradient_descent.png   how fast gradient descent reaches the normal-equations answer.

Report (reports/step4_models.md)
--------------------------------
The method (what is forecast, the features, the four forecasts and how they are tested), then the
results: which forecast wins, why the trees didn't, the tuning choices, what the linear model
relies on, the solver check, the normal ranges and a check on January-February 2026.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from backtest import CHECK_YEAR, LEVELS, REPORT_PATH, TUNE_YEAR, aggregate_errors, describe, totals
from charts import (AQUA, BASELINE, BLUE, INK, INK2, MUTED, ORANGE, RED, YELLOW, add_axes, end_dot, end_labels, legend,
                    line_handle, month_axis, new_figure, rounded_bar, save, source_note, tidy)
from models import LinearModel, clipped_target
from reference import HS_SECTIONS
from reporting import footer, md_table, signed, write_markdown

FAMILY_COLOURS = {"Simple rule": BLUE, "Same month": ORANGE, "Linear": AQUA, "Trees": YELLOW}
SECTION_NAMES = {num: name for num, _, name, _, _ in HS_SECTIONS}
# Readable names for the model's features, used on the coefficient chart.
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


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def fig_national(results: dict, core: pd.DataFrame) -> None:
    """Figure 1: actual core imports in 2023-2025 with the 2024 and 2025 test forecasts."""
    choice = results["choice"]
    win_key = (choice["winner"], describe(choice["best"][choice["winner"]]))
    actual = core[(core["date"] >= "2023-01-01") & (core["date"] <= "2025-12-01")].groupby("date")["value_bd"].sum() / 1e6
    # National forecasts for the two test years, from the chosen model and from the simple rule.
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
    """Figure 2: one small bar chart per level of detail, comparing the four forecasts' 2025 miss."""
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
    panel_w, gap, left = 2.05, 0.3, 0.6           # panel width, gap between panels and left margin, in inches
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


def coefficient_table(model: LinearModel) -> pd.Series:
    """The linear model's coefficients by feature name (without the intercept).

    Features were standardised before fitting, so each coefficient is the change in the scaled
    forecast when that feature rises by one standard deviation, and the coefficients can be
    compared with each other directly.
    """
    names = model.feature_names()
    coef = pd.Series(model.coef_, index=names).drop("intercept")
    return coef


def fig_coefficients(model: LinearModel) -> pd.Series:
    """Figure 3: the ten largest coefficients of the numeric features (section and region dummies left out)."""
    coef = coefficient_table(model)
    numeric = coef[[n for n in coef.index if n in PLAIN]]
    top = numeric.reindex(numeric.abs().sort_values(ascending=False).index[:10])
    n = len(top)
    fig = new_figure(10, 1.75 + 0.42 * n + 0.8, "What the linear model relies on most",
                     "How much the forecast moves (in % of a series' average month) when each feature goes up by one standard deviation.\n"
                     "Blue pushes the forecast up, red pulls it down. Features are measured at the end of the previous December.")
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
    """Loss after each gradient descent step, relative to the exact (normal equations) minimum.

    For each ridge strength the model is fitted both ways on the same table. The relative gap
    (loss - best) / best shows how close descent is after each step; "exact" is the first step at
    which the gap is below 1e-14, i.e. equal up to floating-point rounding.
    """
    out = {}
    for ridge in ridges:
        exact = LinearModel(ridge=ridge).fit(train)
        gd = LinearModel(ridge=ridge, method="gd", n_iter=n_iter).fit(train)
        # Rebuild exactly the problem the model solved, to evaluate the loss at the exact answer.
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
    """Figure 4: the gap to the exact minimum after each gradient descent step (log-log scales)."""
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
    """Draw the four charts and write reports/step4_models.md from the backtest results."""
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

    # --- helpers for the score tables ---
    key = "Reported levels (average)"
    score_cols = {"National total": "National", "Origin region": "By origin region", "Section": "By section",
                  key: "Average of the three"}

    def too_low_high(v: float) -> str:
        r = round(v, 1)
        return "about right" if r == 0 else f"{abs(r):.1f}% too {'low' if r < 0 else 'high'}"

    def simple_table(df: pd.DataFrame, label_col: str, labels) -> str:
        """Miss at each level plus the bias over the year, one row per forecast or setting."""
        out = pd.DataFrame({label_col: labels})
        for src, name in score_cols.items():
            out[name] = [f"{v:.1f}%" for v in df[src]]
        out["Total for the year"] = [too_low_high(v) for v in df["Bias %"]]
        return md_table(out)

    def family_rows(df):
        """One row per forecast family: the chosen settings for the linear model and the trees."""
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
    trees = {y: check_rows[y].loc["Trees"] for y in check_rows}
    families = ["Simple rule", "Same month", "Linear", "Trees"]
    family_names = ["Simple rule (last year's average month)", "Same month last year", "**Linear model**",
                    "Gradient-boosted trees"]

    def tree_label(params: dict) -> str:
        """A tree set-up described in words, e.g. 'Squared error, small trees, sqrt weights'."""
        aim = {"absolute_error": "Absolute error (aims at the median)", "poisson": "Poisson (aims at the mean)",
               "squared_error": "Squared error (aims at the mean)"}.get(params["loss"], params["loss"])
        size = {31: "31 leaves", 15: "15 leaves", 4: "4 leaves"}.get(params["max_leaf_nodes"],
                                                                     f"{params['max_leaf_nodes']} leaves")
        label = f"{aim}, {size}"
        if params.get("weight_power", 1.0) < 1:
            label += ", square-root weights"
        return f"**{label} (chosen)**" if params == best["Trees"] else label

    tree_rows = tune_main[tune_main["family"] == "Trees"]
    tree_labels = [tree_label(p) for p in tree_rows["params"]]

    # January-February 2026 check of every candidate (for information only).
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
        "Years of history": [3, 4, 5],
        "Linear model": [f"{lin[TUNE_YEAR][key]:.1f}%", f"{lin[CHECK_YEAR][key]:.1f}%", f"{jf_lin[key]:.1f}%"],
        "Best trees": [f"{trees[TUNE_YEAR][key]:.1f}%", f"{trees[CHECK_YEAR][key]:.1f}%", f"{jf_best_tree[key]:.1f}%"],
    }))
    abs_err_tree = tree_rows[tree_rows["settings"].str.contains("absolute_error")].iloc[0]

    linear_rows = tune_main[tune_main["family"] == "Linear"]
    linear_labels = [f"**{p['ridge']:g} (chosen)**" if p == best["Linear"] else f"{p['ridge']:g}"
                     for p in linear_rows["params"]]

    history = [tuning[(tuning["first_year"] == fy) & (tuning["family"] == "Linear")
                      & (tuning["settings"] == describe(best["Linear"]))].iloc[0] for fy in (2021, 2022)]
    history_df = pd.DataFrame(history)

    # Normal ranges and the check of 2024-based ranges on 2025.
    ranges = results["ranges"].copy()
    calib = results["calibration"]
    nat_range = ranges.loc[ranges["level"] == "National total", "range"].iloc[0] * 100
    region_ranges = ranges[ranges["level"] == "Origin region"].sort_values("range")
    section_ranges = ranges[ranges["level"] == "Section"].copy()
    section_ranges["name"] = section_ranges["name"].astype(int).map(SECTION_NAMES)
    section_ranges = section_ranges.sort_values("range")

    # January-February 2026 against the final model's forecast.
    errors26 = aggregate_errors(results["forecast_2026"]).merge(results["ranges"], on=["level", "name"])
    errors26["inside"] = errors26["error"].abs() <= errors26["range"]
    jf = errors26[errors26["target_month"].dt.month.isin([1, 2])]
    jf_nat = jf[jf["level"] == "National total"].copy()
    jf_nat_md = md_table(pd.DataFrame({
        "Month": jf_nat["target_month"].dt.strftime("%B %Y"),
        "Forecast (BD m)": (jf_nat["forecast"] / 1e6).map(lambda v: f"{v:,.1f}"),
        "Actual (BD m)": (jf_nat["actual"] / 1e6).map(lambda v: f"{v:,.1f}"),
        "Difference": (jf_nat["error"] * 100).map(lambda v: f"{signed(v, 1)}%"),
        "Normal range": (jf_nat["range"] * 100).map(lambda v: f"+/-{v:.0f}%"),
        "Inside?": jf_nat["inside"].map({True: "yes", False: "no"}),
    }))
    jf_detail = jf[jf["level"] != "National total"]
    jf_out = jf_detail[~jf_detail["inside"]]
    n_out_above = int((jf_out["error"] > 0).sum())
    jf_within = int(np.ceil(jf_nat["error"].abs().max() * 100))

    ranges_md = md_table(pd.concat([
        pd.DataFrame({"Total": ["**Bahrain (national)**"], "Normal range for a single month": [f"**+/-{nat_range:.0f}%**"]}),
        pd.DataFrame({"Total": region_ranges["name"],
                      "Normal range for a single month": (region_ranges["range"] * 100).map(lambda v: f"+/-{v:.0f}%")}),
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

    text = f"""
# Step 4: Forecasting imports without the war

Everything in this project rests on one estimate: what Bahrain would have imported in 2026 if the war had
not happened. This step builds the forecasting models that make that estimate, and tests them on two normal
years, 2024 and 2025, to find out how accurate they are and how far off they normally are. Those normal
misses are what later steps use to decide whether a 2026 shortfall is real or just noise.

## Key findings

- **A ridge regression written from scratch was the most accurate forecast.** On the 2025 final check it
  missed national core imports by {lin[CHECK_YEAR]['National total']:.1f}% a month on average, against
  {simple[CHECK_YEAR]['National total']:.1f}% for the simple rule (last year's average month), and it was the most
  accurate at every level of detail in both test years.
- **Gradient-boosted trees didn't beat it.** With four or five years of noisy monthly data their extra
  flexibility mostly fitted noise; the best set-up was the most constrained one tried.
- **The model relies mainly on the recent level of imports plus an allowance for growth** (about
  {growth_allowance:.1f}% above the last 12 months' average). Calendar effects are small.
- **A single month can be {nat_range:.0f}% off nationally even in a normal year.** Ranges like this were set for every
  total, and when set on 2024 they held for {calib_low:.0f}-{calib_high:.0f}% of months in 2025, close to the 90% aimed for.
- **January and February 2026, before the war, were within {jf_within}% of the forecast**, which supports using
  the model for the months after.

## Method

### What is forecast

The models forecast core goods (everything except bulk materials and precious metals, as decided in step 3).
Core goods are split into series, each one 2-digit customs chapter from one origin region, for example
"chapter 84, machinery, from Europe". A series needs imports in at least 90% of past months to be forecast on
its own; the patchy ones are pooled into one "other" series per section and region. That gives
{details['n_series']:,} series ({details['n_pooled']} of them pooled). Adding up the series' forecasts gives a
forecast for every section, every origin region and the whole country, and these always agree with each
other.

### Features

Each forecast is made at the end of a December for each of the next 12 months, using only what was known at
that point:

| Feature group | What it contains |
|---|---|
| Recent level | Last month, and the average of the last 3, 6 and 12 months |
| Trend | Growth over the last year, and the average month in the year before that |
| Same month before | The same calendar month one and two years earlier |
| Behaviour of the series | How variable it is, and how often it has months with no imports |
| Size and identity | The size of the series, its section and its origin region |
| Calendar | Month of the year, December, days of Ramadan, Eid, days in the month |
| COVID | Whether COVID rules applied in the target month, or in the past 12 months |
| Horizon | How far ahead the forecast is (1 to 12 months) |

Every value is divided by the series' own average month over the last 12 months, so the target becomes "this
month as a multiple of a normal recent month". This lets one model learn from all {details['n_series']:,} series at
once, small and large alike, which gives it far more examples than any single series could. The forecast is
multiplied back afterwards. When the models learn, big series count for more than small ones, because the
errors that matter in the end are in dinars.

### The four forecasts

1. **Simple rule**: every month will look like last year's average month. This is the baseline from step 3,
   and the bar any model has to clear.
2. **Same month last year**: the other obvious baseline.
3. **Ridge regression, written from scratch in NumPy.** It adds up the features with one coefficient each,
   choosing the coefficients that minimise the weighted squared error on past data plus a penalty on their size.
   The penalty pulls the model back towards the simple rule whenever the evidence for a feature is weak. It
   is solved two ways, directly with the normal equations and step by step with gradient descent, as a check on
   the code. *Pros:* transparent, fast and hard to overfit. *Cons:* it can only combine features in straight
   lines, so it can't discover interactions on its own.
4. **Gradient-boosted trees** (scikit-learn's HistGradientBoosting, the same method as LightGBM). Many small
   decision trees, each correcting the errors of the ones before. *Pros:* learns non-linear effects and
   interactions without being told about them, and is usually the strongest method for tabular data. *Cons:*
   harder to explain, more settings to tune, and with little data it can fit noise.

### How the test works

Each test copies exactly what is done for 2026. Standing at the end of December 2023, with only the data known
by then, every model forecasts each month of 2024; the forecasts are then compared with what arrived. The same
is repeated from the end of December 2024 for 2025.

All choices (each model's settings, and whether to train on the COVID year 2021) were made on 2024 only. The
2025 test was run once, at the end, with every choice already fixed, so it shows how the models do on a year
nothing was tuned on, which is the situation 2026 is in. The models were scored on the average miss for the
three levels the later reports use: the national total, the nine origin regions and the sections.

The miss is the weighted absolute percentage error: the total of the absolute monthly errors divided by total
actual imports. Unlike an average of percentage errors, small totals can't blow it up, and it answers the
question that matters here: how many dinars off was the forecast? (Step 3 divided by the forecast instead.
With several models to compare, the actual is fairer, because it is the same for all of them; dividing by the
forecast would let a model look better just by forecasting high.)

## Results

### 1. Which forecast is most accurate?

**2025: the final check**

{simple_table(check_rows[CHECK_YEAR].loc[families], "Forecast", family_names)}

**2024: used for choosing**

{simple_table(check_rows[TUNE_YEAR].loc[families], "Forecast", family_names)}

![Model comparison](figures/step4_2_model_comparison.png)

The linear model wins in both years and at every level. Nationally it missed by
{lin[CHECK_YEAR]['National total']:.1f}% in 2025, against {simple[CHECK_YEAR]['National total']:.1f}% for the simple
rule, about {improvement:.0f}% less. Much of the gain comes from getting the level right: imports were growing, so
the simple rule was {abs(simple[CHECK_YEAR]['Bias %']):.1f}% too low over 2025, while the linear model was
{abs(lin[CHECK_YEAR]['Bias %']):.1f}% too low. The same-month rule was the worst in both years, as step 3 found.

The gain is modest because most of a month's movement is random and no model can predict it. But it is
consistent, and it held up on the year nothing was tuned on.

![National backtest](figures/step4_1_national_backtest.png)

### 2. Why the trees didn't win

Gradient-boosted trees are usually the strongest model for this kind of table, so their result needs
explaining. There were two problems.

First, trained the usual robust way, with absolute error, they forecast too low. Absolute error makes a model
aim at the median month of each series, and in a lopsided series a few big months sit among many smaller ones,
so the median is below the mean. Added up over hundreds of series, those forecasts came out
{abs(abs_err_tree['Bias %']):.1f}% too low in 2024. Switching to squared error, which aims at the mean, fixed the bias.

Second, there isn't enough history for their flexibility to pay off. With four or five years of noisy monthly
data, most of the patterns trees can find are noise. The best set-up was the most constrained one tried (4
leaves per tree, at least 1,000 rows per leaf), and in 2025 it still missed more than the linear model
({trees[CHECK_YEAR][key]:.1f}% against {lin[CHECK_YEAR][key]:.1f}% on the average of the three levels) and slightly more
than the simple rule ({simple[CHECK_YEAR][key]:.1f}%).

Every tree set-up tried on 2024:

{simple_table(tree_rows, "Tree set-up", tree_labels)}

*Tree results can differ slightly (in the first decimal place) between scikit-learn versions; the conclusions
don't change.*

### 3. The ridge penalty

A stronger penalty keeps the coefficients smaller, so the model stays closer to the simple rule. Four
strengths were tried on 2024:

{simple_table(linear_rows, "Penalty", linear_labels)}

A fairly strong penalty ({ridge:g}) worked best. Apart from the recent level and growth, the evidence for most
features is weak, and the penalty stops the model from chasing noise. Several features also overlap (the
averages of the last 3, 6 and 12 months move together), and without a penalty their coefficients can grow
large and cancel each other out.

### 4. Training on 2021

The chosen linear model was run on 2024 twice, learning from 2021-2023 and from 2022-2023 only:

{simple_table(history_df, "History used", ["2021-2023 (with 2021)", "2022-2023 (without 2021)"])}

It is more accurate with 2021. Without it there is only one year of history in the 2024 test, which isn't
enough to learn about trends. So 2021 stays in, and the COVID features tell the model which of its months were
unusual.

### 5. What the linear model relies on

![Coefficients](figures/step4_3_coefficients.png)

*Each bar shows how much the forecast moves, in % of the series' average month, when that feature rises by one
standard deviation. Final model, trained on 2021-2025. Section and region adjustments are not shown.*

The intercept is a built-in allowance for growth: with every feature at its typical value, the model forecasts
{growth_allowance:.1f}% above the series' average month over the last 12 months. That is roughly how much imports
grew from one year to the next, and it fixes most of the simple rule's tendency to forecast too low. Beyond
that, recent months pull the forecast their way: if the last month or the last 3 or 6 months were higher than
usual, the forecast goes up a little. Part of last year's change tends to unwind. A series that was higher in
the year before than it is now gets a slightly higher forecast ({coef['prev_12'] * 100:+.1f} on the chart), and one
that grew fast gets a slightly lower one ({coef['growth'] * 100:+.1f}). The same month two years earlier adds a little
seasonal memory ({coef['same_month_2y'] * 100:+.1f}). A month with Eid al-Adha is about
{abs(coef['eid_al_adha']) * 100:.1f}% lower, and December about {coef['december'] * 100:.1f}% higher.

All these adjustments are small, a few percent at most. That is the penalty at work: in normal years most of a
month's movement is noise, so the model only moves away from "recent level plus growth" when the evidence is
clear. Because some features overlap, the penalty shares the credit between them, so any single coefficient
should be read with care.

### 6. Checking the maths: normal equations and gradient descent

The ridge coefficients were found two ways, both coded from scratch. The normal equations solve
(X'WX + P) b = X'Wy directly. Gradient descent starts with every coefficient at zero and repeatedly steps
downhill, with a step size of 1/L, where L is the largest eigenvalue of the Hessian 2(X'WX + P); that is the
largest step that is guaranteed not to overshoot for a quadratic loss. If the code is right, both must reach the
same answer.

![Gradient descent](figures/step4_4_gradient_descent.png)

With the chosen penalty, gradient descent gets within 0.1% of the minimum after {gd_chosen['within_0.1%']} steps
and matches it to floating-point precision after {gd_chosen['exact']} steps. With a very weak penalty
({min(runs):g}), the overlapping features make the problem badly conditioned: the loss surface is almost flat
in some directions, so progress slows, and after {len(gd_weak['gap'])} steps it is close but not exact. A larger
penalty improves the conditioning of X'WX, which is why descent converges faster.

### 7. How far off the model normally is

For each reported total (national, each origin region, each section) I took the model's monthly errors in the
2024 and 2025 tests and found the distance that 9 months out of 10 stayed within. That is its normal range. A
2026 month only counts as a real shortfall if it falls outside it.

To check that the ranges work on a year they weren't built from, they were first set from 2024 alone and
tested on 2025:

{calib_md}

Between {calib_low:.0f}% and {calib_high:.0f}% of months fell inside, close to the 90% aimed for. The final ranges
are set from both years together:

{ranges_md}

*Sections range from +/-{section_ranges['range'].min() * 100:.0f}% ({section_ranges['name'].iloc[0]}) to
+/-{section_ranges['range'].max() * 100:.0f}% ({section_ranges['name'].iloc[-1]}). All are saved in
`data/interim/step4_ranges.csv`.*

A single national month can be {nat_range:.0f}% off in a normal year, much more than the average miss of
{lin[CHECK_YEAR]['National total']:.1f}%, because the range has to cover unusual months such as December as well.
Small or erratic origins have wide ranges, so a change there has to be large before it counts.

These ranges are empirical: they make no assumption about the shape of the errors, they simply read off how big
the misses were. The trade-off is that with 24 months per total, the 90th percentile is itself uncertain.

### 8. January and February 2026

The model was retrained on 2021-2025 and used to forecast 2026 from the end of December 2025, exactly as step
5 does. The war began on 28 February, so January and February should look normal:

{jf_nat_md}

Both months land close to the forecast and well inside the range. Of the {len(jf_detail)} origin-region and
section months, {int(jf_detail['inside'].sum())} were inside their range too. Of the {len(jf_out)} outside it,
{n_out_above} were above the forecast, mostly in February, which could be stockpiling as tensions rose.

Because trees benefit from more history, every candidate was also run on these two months with all five years
of training data:

{simple_table(jf_all, "Forecast", jf_labels).replace("Total for the year", "Total for the two months")}

With five years of history the best trees did about as well as the linear model, but across the three tests
more history doesn't steadily favour them:

{history_md}

*Average miss over the three reported levels.*

This check was not used to change the choice. Two months is too few to separate models that differ by a few
tenths of a percent; February may already have been unusual; and picking the best of several set-ups on a
small test flatters whichever wins by luck. Instead, step 5 uses the linear model for the main results and the
best trees as a second opinion. Where the two agree, the findings are stronger; where they differ, that is
reported.

## Limitations

- **Two test years.** The normal ranges rest on 24 months per total. More normal years would make them more
  reliable.
- **Tuning on a single year** can favour settings that happened to suit 2024. The untouched 2025 check is the
  guard against this, and it confirmed the choice.
- **Both models learn from the same five years**, so their agreement in step 5 is reassuring but not independent
  proof.
- **Pooling patchy series** into "other" means their individual products can't be forecast separately.
- **The 2026 forecast is made once, from December 2025.** It can't know about anything that would have changed
  in 2026 without the war, such as price moves or new projects.

## What step 5 uses

- The linear model's 2026 forecasts (`data/interim/step4_forecast_2026.parquet`) and the best trees' forecasts as a
  second opinion.
- The normal ranges (`data/interim/step4_ranges.csv`) to decide which gaps are real.
- The 2024-2025 test forecasts (`data/interim/step4_backtest.parquet`) to build ranges for periods longer than one
  month.
- Bulk materials and precious metals have too few series for these models; step 5 uses the simple rule for them,
  with their normal ranges from step 3.

{footer("src/backtest.py", " (charts and text in `src/step4_report.py`)")}
"""
    write_markdown(REPORT_PATH, text)
