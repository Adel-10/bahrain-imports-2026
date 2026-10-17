"""Step 4c: test the models on normal years, choose one, and measure how far off it normally is.

    python src/backtest.py

Why a backtest
--------------
The whole project rests on one comparison: what arrived in 2026 against what would have arrived
without the war. "What would have arrived" is a forecast, so its accuracy has to be known before
any gap can be trusted. The only honest way to know it is to make the same kind of forecast for
years where the answer is known, using only the data that would have been available at the time.

How it works ("pretend it's January")
-------------------------------------
Standing at the end of December 2023, every model forecasts each month of 2024 using only data
up to then; the same is repeated, one year later, for 2025.

  1. **Tuning** on 2024 only: model settings, and whether to keep the COVID year 2021 in the
     training data, are chosen by the average miss on the totals the reports use (national,
     each origin region, each section).
  2. **Final check** on 2025, used once with the chosen settings. Because nothing was chosen
     using 2025, it gives an unbiased view of how well the choice works on a new year.
  3. **Ranges.** The chosen model's misses in both years set a "9 months out of 10" range for
     every reported total. The ranges are also set from 2024 alone and checked on 2025, to see
     whether they hold on a year they weren't built from.
  4. **2026.** The chosen model is retrained on 2021-2025 and forecasts 2026 from the end of
     December 2025. Only January and February (before the war) are checked here; step 5 uses
     the rest.

Pros: no information from the test year leaks into the forecast or the settings, and the test
mimics exactly how the 2026 forecast is made. Cons: only two test years, so the normal ranges
rest on 24 months per total, and tuning on a single year can favour settings that happened to
suit 2024 (the 2025 check guards against this).

Writes reports/step4_models.md, four charts in reports/figures/, and for step 5 (in data/interim/):
the 2026 forecasts of the chosen model and of the best trees (a second opinion), their ranges,
and their 2024-2025 test forecasts.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from config import ROOT
from features import fold_tables, load_calendar, load_core
from models import BoostedTrees, LinearModel, SameMonth, SimpleAverage

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice means on brand-new series; handled as NaN

INTERIM_DIR = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "reports" / "step4_models.md"
TUNE_YEAR, CHECK_YEAR, FORECAST_YEAR = 2024, 2025, 2026
LEVELS = {"National total": [], "Origin region": ["region"], "Section": ["section_id"], "Series (HS2 x region)": ["series"]}
REPORT_LEVELS = ["National total", "Origin region", "Section"]      # what step 5 reports, so what tuning optimises
RANGE_COVERAGE = 0.9                                                # "9 months out of 10"

# Candidates tried on the 2024 test. Four ridge strengths for the linear model, and four tree set-ups
# running from large flexible trees (first) to small heavily constrained ones (last).
LINEAR_GRID = [dict(ridge=r) for r in (0.01, 0.1, 1.0, 10.0)]
TREE_GRID = [
    dict(loss="absolute_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=50, weight_power=1.0),
    dict(loss="poisson", learning_rate=0.03, max_iter=200, max_leaf_nodes=15, min_samples_leaf=200, weight_power=1.0),
    dict(loss="squared_error", learning_rate=0.03, max_iter=200, max_leaf_nodes=15, min_samples_leaf=200, weight_power=1.0),
    dict(loss="squared_error", learning_rate=0.03, max_iter=100, max_leaf_nodes=4, min_samples_leaf=1000, weight_power=0.5),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def forecasts_frame(panel, test: pd.DataFrame, ratio: np.ndarray) -> pd.DataFrame:
    """Test rows with the forecast in BD and the series' section and region attached."""
    out = test[["series", "target_month", "horizon", "scale"]].copy()
    if "actual" in test:
        out["actual"] = test["actual"].to_numpy()
    out["forecast"] = ratio * out["scale"].to_numpy()
    return out.join(panel.info[["section_id", "region", "hs2"]], on="series")


def totals(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Actual and forecast added up to the totals given by `keys` (e.g. by region), month by month."""
    cols = [c for c in ("actual", "forecast") if c in frame]
    return frame.groupby([*keys, "target_month"], observed=True)[cols].sum()


def miss(frame: pd.DataFrame, keys: list[str]) -> float:
    """Average monthly miss: sum of |actual - forecast| over all totals and months, as % of the actual.

    (Step 3 divided by the forecast. With several models the actual is the fairer yardstick:
    it is the same for every model, so a model can't look better by forecasting too high.)

    This is the weighted absolute percentage error (WAPE). Compared with the average of
    percentage errors (MAPE), it isn't blown up by tiny totals and it weights each total by its
    size, which matches the question "how many dinars off was the forecast?".
    """
    t = totals(frame, keys)
    return float((t["actual"] - t["forecast"]).abs().sum() / t["actual"].sum() * 100)


def score(frame: pd.DataFrame) -> dict:
    """Miss at every level, overall bias (forecast too high or too low) and the tuning score."""
    out = {level: miss(frame, keys) for level, keys in LEVELS.items()}
    out["Bias %"] = float((frame["forecast"].sum() / frame["actual"].sum() - 1) * 100)
    out["Reported levels (average)"] = float(np.mean([out[level] for level in REPORT_LEVELS]))
    return out


def describe(params: dict) -> str:
    """Settings as readable text, e.g. 'ridge=1.0'."""
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def run_candidates(core, calendar, year: int, first_year: int, linear_grid, tree_grid) -> tuple[list[dict], dict]:
    """Fit every candidate on the training table for `year` and score its forecasts of `year`."""
    panel, train, test = fold_tables(core, calendar, year, first_year=first_year)
    rows, frames = [], {}
    candidates = [("Simple rule", SimpleAverage(), {}), ("Same month", SameMonth(), {})]
    candidates += [("Linear", LinearModel(**p), p) for p in linear_grid]
    candidates += [("Trees", BoostedTrees(**p), p) for p in tree_grid]
    for family, model, params in candidates:
        model.fit(train)
        frame = forecasts_frame(panel, test, model.predict(test))
        rows.append({"year": year, "first_year": first_year, "family": family, "settings": describe(params),
                     "params": params, **score(frame)})
        frames[(family, describe(params))] = frame
    info = {"panel": panel, "n_train": len(train), "n_series": len(panel.info),
            "n_pooled": int((panel.info["hs2"] == "other").sum())}
    return rows, {"frames": frames, **info}


def tune(core, calendar) -> tuple[pd.DataFrame, dict]:
    """Try every candidate on 2024, with and without 2021, and pick the best of each family."""
    results, details = [], {}
    for first_year in (2021, 2022):
        rows, info = run_candidates(core, calendar, TUNE_YEAR, first_year, LINEAR_GRID, TREE_GRID)
        results += rows
        details[first_year] = info
    tuning = pd.DataFrame(results)
    # Tuning score: the average miss on the three levels the reports use (national, region, section).
    key = "Reported levels (average)"
    best_first_year = int(tuning[tuning["family"].isin(["Linear", "Trees"])].sort_values(key).iloc[0]["first_year"])
    chosen = tuning[tuning["first_year"] == best_first_year]
    # Best settings within each family, then the better of the two families.
    best = {fam: chosen[chosen["family"] == fam].sort_values(key).iloc[0]["params"] for fam in ("Linear", "Trees")}
    winner = min(("Linear", "Trees"), key=lambda fam: chosen[(chosen["family"] == fam)
                                                             & (chosen["settings"] == describe(best[fam]))][key].iloc[0])
    return tuning, {"first_year": best_first_year, "best": best, "winner": winner, "details": details}


def final_check(core, calendar, choice: dict) -> tuple[pd.DataFrame, dict]:
    """Score the chosen settings of every family on 2024 and 2025 (2025 for the first time)."""
    out, frames = [], {}
    for year in (TUNE_YEAR, CHECK_YEAR):
        for first_year in (2021, 2022):
            rows, info = run_candidates(core, calendar, year, first_year,
                                        [choice["best"]["Linear"]], [choice["best"]["Trees"]])
            out += rows
            if first_year == choice["first_year"]:
                frames[year] = info["frames"]
    return pd.DataFrame(out), frames


def aggregate_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly % errors (actual / forecast - 1) of every reported total.

    The error is measured relative to the forecast because that is how the 2026 gaps are
    expressed ("x% below the forecast"), so the ranges and the gaps are on the same scale.
    """
    parts = []
    for level, keys in LEVELS.items():
        if level not in REPORT_LEVELS:
            continue
        t = totals(frame, keys).reset_index()
        t["level"] = level
        t["name"] = t[keys[0]].astype(str) if keys else "Bahrain"
        parts.append(t[["level", "name", "target_month", "actual", "forecast"]])
    errors = pd.concat(parts, ignore_index=True)
    errors["error"] = errors["actual"] / errors["forecast"] - 1
    return errors


def ranges_from(errors: pd.DataFrame) -> pd.DataFrame:
    """Half-width of the "9 months out of 10" range for each total: 90th percentile of |error|.

    This is an empirical interval: it makes no assumption about the shape of the errors (no
    normal distribution), it just reads off how big the misses actually were. The downside is
    that with 24 months per total the 90th percentile is itself uncertain.
    """
    return (errors.assign(abs_error=errors["error"].abs())
            .groupby(["level", "name"])["abs_error"].quantile(RANGE_COVERAGE).rename("range").reset_index())


def coverage(errors: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Share of months whose actual value fell inside forecast +/- range, by level."""
    merged = errors.merge(ranges, on=["level", "name"])
    merged["inside"] = merged["error"].abs() <= merged["range"]
    return merged.groupby("level")["inside"].agg(["mean", "size"]).rename(columns={"mean": "share inside", "size": "months"})


def forecast_2026(core, calendar, choice: dict):
    """Retrain on 2021-2025 and forecast 2026 from the end of December 2025.

    Returns the chosen model and its forecasts, the best trees and theirs (the "second opinion"
    step 5 uses), the training table, and every candidate's score on January-February 2026.
    That last score is for information only: it is not used to choose anything.
    """
    panel, train, test = fold_tables(core, calendar, FORECAST_YEAR, first_year=choice["first_year"])
    winner_params = choice["best"][choice["winner"]]
    model = (LinearModel if choice["winner"] == "Linear" else BoostedTrees)(**winner_params).fit(train)
    trees = BoostedTrees(**choice["best"]["Trees"]).fit(train)
    frame = forecasts_frame(panel, test, model.predict(test))
    frame_trees = forecasts_frame(panel, test, trees.predict(test))

    # January-February 2026 were before the war, so they are a small extra out-of-sample check.
    pre_war = test[test["target_month"].dt.month.isin([1, 2])].reset_index(drop=True)
    candidates = [("Simple rule", SimpleAverage(), {}), ("Same month", SameMonth(), {}),
                  ("Linear", model, winner_params)]
    candidates += [("Trees", BoostedTrees(**p).fit(train), p) for p in TREE_GRID]
    jan_feb = []
    for family, m, params in candidates:
        scored = score(forecasts_frame(panel, pre_war, m.predict(pre_war)))
        jan_feb.append({"family": family, "settings": describe(params), "params": params, **scored})
    return model, frame, frame_trees, train, pd.DataFrame(jan_feb)


def backtest_predictions(frames: dict, keys: dict) -> pd.DataFrame:
    """The 2024 and 2025 test forecasts of the chosen models, kept for step 5's ranges."""
    parts = []
    for year, by_key in frames.items():
        for label, key in keys.items():
            parts.append(by_key[key].assign(model=label, year=year))
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    """Tune, check, set the ranges, forecast 2026, save the results for step 5 and write the report."""
    t0 = time.time()
    core, calendar = load_core(), load_calendar()

    print("1/4 Tuning on 2024 ...", flush=True)
    tuning, choice = tune(core, calendar)
    print(f"    chosen: {choice['winner']} ({describe(choice['best'][choice['winner']])}), "
          f"history from {choice['first_year']}  [{time.time() - t0:.0f}s]", flush=True)

    print("2/4 Final check on 2025 ...", flush=True)
    check, frames = final_check(core, calendar, choice)

    print("3/4 Ranges ...", flush=True)
    winner_key = (choice["winner"], describe(choice["best"][choice["winner"]]))
    trees_key = ("Trees", describe(choice["best"]["Trees"]))
    errors = {year: aggregate_errors(frames[year][winner_key]) for year in (TUNE_YEAR, CHECK_YEAR)}
    # Do ranges set on 2024 hold on 2025? (Ideally about 90% of 2025 months fall inside.)
    calib = coverage(errors[CHECK_YEAR], ranges_from(errors[TUNE_YEAR]))
    # The ranges used from step 5 onward come from both years, for more months per total.
    ranges = ranges_from(pd.concat(errors.values()))
    ranges_trees = ranges_from(pd.concat(aggregate_errors(frames[y][trees_key]) for y in (TUNE_YEAR, CHECK_YEAR)))

    print("4/4 Forecast for 2026 ...", flush=True)
    model, frame26, frame26_trees, train26, jan_feb = forecast_2026(core, calendar, choice)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    frame26.to_parquet(INTERIM_DIR / "step4_forecast_2026.parquet", index=False)
    frame26_trees.to_parquet(INTERIM_DIR / "step4_forecast_2026_trees.parquet", index=False)
    ranges.to_csv(INTERIM_DIR / "step4_ranges.csv", index=False)
    ranges_trees.to_csv(INTERIM_DIR / "step4_ranges_trees.csv", index=False)
    backtest_predictions(frames, {"linear": winner_key, "trees": trees_key}).to_parquet(
        INTERIM_DIR / "step4_backtest.parquet", index=False)

    results = {"tuning": tuning, "choice": choice, "check": check, "frames": frames, "errors": errors,
               "calibration": calib, "ranges": ranges, "model": model, "forecast_2026": frame26,
               "train_2026": train26, "jan_feb_all": jan_feb}
    import step4_report                                   # charts and report live in their own module
    step4_report.write(results, core, calendar)
    print(f"Done in {time.time() - t0:.0f}s. Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
