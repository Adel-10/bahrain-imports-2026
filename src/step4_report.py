"""Charts and the written report for step 4 (called by backtest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from backtest import (CHECK_YEAR, LEVELS, RANGE_COVERAGE, REPORT_LEVELS, REPORT_PATH, TUNE_YEAR, aggregate_errors,
                      describe, totals)
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
    top_coef = fig_coefficients(results["model"])
    runs = gradient_descent_runs(results["train_2026"])
    fig_gradient_descent(runs, best["Linear"]["ridge"])

    key = "Reported levels (average)"
    level_cols = [*LEVELS, "Bias %", key]

    def family_rows(df):
        rows = []
        for fam in ("Simple rule", "Same month", "Linear", "Trees"):
            part = df[df["family"] == fam]
            if fam in best:
                part = part[part["settings"] == describe(best[fam])]
            rows.append(part.iloc[0])
        out = pd.DataFrame(rows)
        out.insert(0, "forecast", out["family"].map(FAMILY_LABELS))
        return out[["forecast", *level_cols]]

    def fmt(df, cols):
        df = df.copy()
        for c in cols:
            df[c] = df[c].map(lambda v: f"{v:.1f}" if c != "Bias %" else signed(v, 1))
        return df

    tune_main = tuning[tuning["first_year"] == first_year]
    linear_rows = tune_main[tune_main["family"] == "Linear"][["settings", *level_cols]]
    tree_rows = tune_main[tune_main["family"] == "Trees"][["settings", *level_cols]]
    history_rows = []
    for fy in (2021, 2022):
        part = tuning[(tuning["first_year"] == fy) & (tuning["family"] == "Linear") & (tuning["settings"] == describe(best["Linear"]))]
        history_rows.append({"history used": "2021-2023 (with 2021)" if fy == 2021 else "2022-2023 (without 2021)",
                             **part[level_cols].iloc[0].to_dict()})
    check_rows = {y: family_rows(check[(check["year"] == y) & (check["first_year"] == first_year)]) for y in (TUNE_YEAR, CHECK_YEAR)}
    lin = {y: check_rows[y].set_index("forecast").loc[FAMILY_LABELS["Linear"]] for y in check_rows}
    simple = {y: check_rows[y].set_index("forecast").loc[FAMILY_LABELS["Simple rule"]] for y in check_rows}
    trees = {y: check_rows[y].set_index("forecast").loc[FAMILY_LABELS["Trees"]] for y in check_rows}
    abs_err_tree = tune_main[(tune_main["family"] == "Trees") & tune_main["settings"].str.contains("absolute_error")].iloc[0]

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
    jf_nat_md = pd.DataFrame({
        "month": jf_nat["target_month"].dt.strftime("%B %Y"),
        "forecast (BD m)": (jf_nat["forecast"] / 1e6).map(lambda v: f"{v:,.1f}"),
        "actual (BD m)": (jf_nat["actual"] / 1e6).map(lambda v: f"{v:,.1f}"),
        "difference": (jf_nat["error"] * 100).map(lambda v: f"{signed(v, 1)}%"),
        "range": (jf_nat["range"] * 100).map(lambda v: f"+/-{v:.0f}%"),
        "inside?": jf_nat["inside"].map({True: "yes", False: "no"}),
    })
    jf_levels = jf.groupby("level")["inside"].agg(["sum", "size"])
    jf_out = jf[~jf["inside"] & (jf["level"] != "National total")].copy()
    jf_out["name"] = np.where(jf_out["level"] == "Section", jf_out["name"].map(lambda s: SECTION_NAMES.get(int(s), s) if str(s).isdigit() else s), jf_out["name"])
    n_out_above = int((jf_out["error"] > 0).sum())

    ranges_md = pd.concat([
        pd.DataFrame({"total": ["Bahrain (national)"], "range": [f"+/-{nat_range:.0f}%"]}),
        pd.DataFrame({"total": region_ranges["name"], "range": (region_ranges["range"] * 100).map(lambda v: f"+/-{v:.0f}%")}),
    ])
    calib_md = calib.reset_index().rename(columns={"level": "level", "share inside": "months inside the range", "months": "months checked"})
    calib_md["months inside the range"] = (calib_md["months inside the range"] * 100).map(lambda v: f"{v:.0f}%")
    calib_md["months checked"] = calib_md["months checked"].astype(int)

    ridge = best["Linear"]["ridge"]
    gd_chosen = runs[ridge]
    gd_weak = runs[min(runs)]
    n_coef = len(results["model"].coef_)
    coef = coefficient_table(results["model"])
    growth_allowance = (results["model"].coef_[0] - 1) * 100

    text = f"""# Step 4: Forecasting models, tested on 2024 and 2025

*Generated by `src/backtest.py` (charts and text by `src/step4_report.py`). Re-running the script refreshes every number.*

The whole project depends on one thing: a good estimate of what imports **would have been in 2026 without the war**.
This step builds models to make that estimate and tests them on two normal years first, so we know how far off they
usually are. The questions:

1. Which forecast is most accurate in normal years?
2. Does keeping 2021 (the COVID year) help or hurt?
3. What does the winning model rely on?
4. How far off is it usually, and can we put a reliable range around its forecasts?
5. Does it get January and February 2026 (before the war) right?

---

## Before you start: how the test works

### Step 1: Pretend it's January

Each test copies exactly what will be done for 2026. Standing at the end of December 2023, using only data known by then,
each model forecasts every month of 2024. Then we compare with what actually arrived. The same is repeated from the end of
December 2024 for 2025.

### Step 2: What gets forecast

The models forecast core goods (as decided in step 3), split into **series**: one HS2 chapter from one origin region, for
example "chapter 84 (machinery) from Europe".

- A series needs imports in at least 90% of past months to be forecast on its own.
- The patchy ones are added together into one "other" series per section and region.
- That gives {details['n_series']:,} series ({details['n_pooled']} of them "other").

Adding the series' forecasts up gives forecasts for each section, each origin region and the whole country, which always
agree with each other.

### Step 3: The clues the models get

For each series, the models see clues that were known at the end of December:

| Clue | What it means |
|---|---|
| Recent level | Last month, and the average of the last 3, 6 and 12 months |
| Trend | Growth over the last year, and the average month in the year before |
| Same month before | The same month last year and two years ago |
| How regular the series is | How jumpy it is, and how often it has months with no imports |
| Size and identity | How big the series is, its section and its origin region |
| Calendar | Month of the year, December, days of Ramadan, Eid, days in the month |
| COVID | Whether COVID rules applied in the month, or in the past 12 months |
| Months ahead | How far ahead the forecast is (1 to 12 months) |

Everything is measured relative to the series' own average month, so a small series and a big one look alike to the models.

### Step 4: The four forecasts

1. **Simple rule:** last year's average month (the yardstick from step 3).
2. **Same month last year.**
3. **Linear model:** a weighted linear regression with a ridge penalty, written from scratch in NumPy. It adds up each
   clue times a weight, and the weights are chosen to make the squared misses as small as possible. The ridge penalty
   stops any weight getting too big, which pulls the model towards the simple rule when the evidence is weak.
4. **Gradient-boosted trees:** many small decision trees, each one correcting the mistakes of the ones before
   (scikit-learn's HistGradientBoosting, the same method LightGBM made popular). One model learns from all the series at once.

During training, big series count for more than small ones, because the models are judged on misses in Bahraini dinars.

### Step 5: How the forecasts are scored

The **miss** is the same idea as in step 3: add up how far each forecast was from the actual value, ignoring above or below.
There is one change: it is now divided by the **actual** imports instead of the forecast. With several models, the actual
is the fair yardstick, because it is the same for everyone. Dividing by the forecast would let a model look better just by
forecasting too high.

The miss is measured at four levels: the national total, each origin region, each section, and each series. The three
levels step 5 will report (national, region, section) are averaged into one score for choosing the winner.

### Step 6: Choose on 2024, confirm on 2025

- **Tuning:** each model's settings, and whether to keep 2021, were chosen using the 2024 test only.
- **Final check:** 2025 was used once, at the end, with the chosen settings. Because nothing was tuned on it, it is an
  honest test of how the models will do on a year they haven't seen.

---

## 1. Which forecast is most accurate?

**What we did:** ran all four forecasts on the 2024 and 2025 tests.

**What we found:**

**2025 (final check)**

{md_table(fmt(check_rows[CHECK_YEAR], level_cols))}

**2024 (used for choosing)**

{md_table(fmt(check_rows[TUNE_YEAR], level_cols))}

*Average monthly miss as % of actual imports (lower is better). Bias: total forecast compared with total actual imports
over the year (negative = too low).*

![Model comparison](figures/step4_2_model_comparison.png)

1. **The linear model wins in both years.** Nationally it missed by {lin[CHECK_YEAR]['National total']:.1f}% in 2025,
   against {simple[CHECK_YEAR]['National total']:.1f}% for the simple rule
   ({lin[TUNE_YEAR]['National total']:.1f}% against {simple[TUNE_YEAR]['National total']:.1f}% in 2024).
2. **The gain is modest but consistent.** It is better at every level in both years.
3. **Much of the gain comes from getting the level right.** The simple rule's forecasts were
   {abs(simple[CHECK_YEAR]['Bias %']):.1f}% too low over 2025, because imports were growing. The linear model's bias was
   {signed(lin[CHECK_YEAR]['Bias %'], 1)}%.
4. **"Same month last year" is the worst everywhere**, confirming step 3.

![National backtest](figures/step4_1_national_backtest.png)

---

## 2. Why didn't the trees win?

Gradient-boosted trees are often the strongest model for this kind of table, so it is worth explaining why they weren't here.

1. **They first undershot by a lot.** Trained on absolute misses, the trees forecast each series' *median* (typical) month.
   For lopsided series, where a few big months sit among many small ones, the median is below the average. Added up across
   hundreds of series, the forecasts came out {abs(abs_err_tree['Bias %']):.1f}% too low in 2024. Switching to squared
   misses, which aim at the *average*, fixed that.
2. **Even fixed, there isn't enough history for their flexibility to pay off.** Trees can learn complicated patterns, but
   with four or five years of noisy monthly data, most of those patterns are noise. The best tree settings were the most
   restricted ones tried (trees with only 4 leaves). In 2025 they missed more than the linear model
   ({trees[CHECK_YEAR][key]:.1f}% against {lin[CHECK_YEAR][key]:.1f}%), and slightly more than the simple rule
   ({simple[CHECK_YEAR][key]:.1f}%).

All tree settings tried on 2024:

{md_table(fmt(tree_rows, level_cols))}

*Tree results can differ slightly (in the first decimal place) between scikit-learn versions; the conclusions don't change.*

The linear model's ridge penalty, also chosen on 2024:

{md_table(fmt(linear_rows, level_cols))}

A stronger penalty (ridge = {ridge:g}) works best: the evidence for anything beyond "recent level plus trend" is weak, and
the penalty keeps the model from chasing noise.

---

## 3. Keep 2021 or not?

**What we did:** ran the chosen linear model on 2024 twice, once learning from 2021-2023 and once from 2022-2023.

{md_table(fmt(pd.DataFrame(history_rows), level_cols))}

**What we found:** keeping 2021 helps. Without it, the model has only one year of history to learn from in the 2024 test,
which isn't enough to learn about trends. 2021 stays in, with the COVID clues marking its unusual months.

---

## 4. What does the linear model rely on?

![Coefficients](figures/step4_3_coefficients.png)

*From the final model, trained on 2021-2025. The {n_coef} weights include section and region adjustments, not shown.*

1. **A built-in growth allowance.** When every clue is at its typical value, the model forecasts {growth_allowance:.1f}%
   above a series' average month over the last 12 months: on average, that is how much imports ran above the year before.
   This alone fixes most of the simple rule's "too low" bias.
2. **Recent months pull the forecast their way.** Last month and the averages of the last 3 and 6 months all push the
   forecast in the same direction they moved.
3. **Part of last year's change tends to unwind.** If a series was higher in the year before than recently, the forecast
   moves up a little ({coef['prev_12'] * 100:+.1f} on the chart); if it grew fast, the forecast is pulled down a little
   ({coef['growth'] * 100:+.1f}).
4. **The same month two years ago** carries a little seasonal memory ({coef['same_month_2y'] * 100:+.1f}).
5. **Holidays:** a month with Eid al-Adha is about {abs(coef['eid_al_adha']) * 100:.1f}% lower, and December about
   {coef['december'] * 100:.1f}% higher.

All the adjustments are small. The ridge penalty keeps them that way, because in normal years most of a month's movement
is noise. Several clues overlap (for example the last 3 and last 6 months), so the penalty shares the credit between them
and individual weights should be read with care.

**Checking the maths.** The weights can be found two ways, and both were coded from scratch:

1. **Normal equations:** solve one matrix equation, which gives the exact minimum in one go.
2. **Gradient descent:** start with all weights at zero and repeatedly take a small step downhill. The step size is set
   from the largest eigenvalue of the Hessian (how sharply the loss curves), so steps never overshoot.

![Gradient descent](figures/step4_4_gradient_descent.png)

- With the chosen penalty, gradient descent gets within 0.1% of the exact minimum after {gd_chosen['within_0.1%']} steps, and is
  exact to the computer's precision after {gd_chosen['exact']} steps. Its weights match the normal equations' to within
  {gd_chosen['max_coef_diff']:.0e}.
- With a weak penalty (ridge = {min(runs):g}), the overlapping clues make the problem harder. It also gets close quickly,
  but after {len(gd_weak['gap'])} steps it is still {gd_weak['gap'][-1]:.0e} above the minimum, and its weights are still
  up to {gd_weak['max_coef_diff']:.0e} off.

---

## 5. How far off is the model usually?

**What we did:** used the model's monthly misses in the 2024 and 2025 tests to set a **range** around each forecast,
using the same idea as the normal range in step 3: the distance that 9 months out of 10 stayed within.

**Does the range work?** As a fair check, ranges were first set using 2024 only, then tested on 2025:

{md_table(calib_md)}

Close to the 90% target at every level, so the ranges can be trusted.

**The final ranges** (set from 2024 and 2025 together), for a single month:

{md_table(ranges_md)}

*Sections range from +/-{section_ranges['range'].min() * 100:.0f}% ({section_ranges['name'].iloc[0]}) to
+/-{section_ranges['range'].max() * 100:.0f}% ({section_ranges['name'].iloc[-1]}); all are saved in `data/interim/step4_ranges.csv`.*

- A single month nationally can be {nat_range:.0f}% away from the forecast even in a normal year. That's bigger than the
  average miss ({lin[CHECK_YEAR]['National total']:.1f}%), because ranges cover the unusual months too (December, for example).
- Small or jumpy origins have wide ranges. A change there needs to be large before it counts as real.

---

## 6. Sanity check: January and February 2026

The final model was retrained on 2021-2025 and forecast 2026 from the end of December 2025. The war began on 28 February,
so January and February should look normal:

{md_table(jf_nat_md)}

- **Nationally, both months land close to the forecast**, well inside the range.
- For origin regions and sections, {int(jf_levels.loc[['Origin region', 'Section'], 'sum'].sum())} of
  {int(jf_levels.loc[['Origin region', 'Section'], 'size'].sum())} region and section months were inside their range.
  Of those outside, {n_out_above} of {len(jf_out)} were *above* the forecast, mostly in February. That could be
  stockpiling as tensions rose before the war; step 5 will keep it in mind.

---

## What this means for step 5

1. **The linear model (ridge = {ridge:g}, trained on 2021-2025)** gives the "without the war" forecast for 2026.
   Its forecasts are saved in `data/interim/step4_forecast_2026.parquet`.
2. **Each month of 2026 is compared with its own forecast**, nationally, by origin region and by section.
3. **A month counts as a real shortfall only if it falls outside the range.** The ranges are saved in `data/interim/step4_ranges.csv`.
4. **Bulk materials and precious metals** have too few series for these models. Step 5 uses the simple rule for them,
   with their normal ranges from step 3.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
