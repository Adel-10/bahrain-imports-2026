"""Step 4c: test the models on normal years, choose one, and measure how far off it normally is.

    python src/backtest.py

"Pretend it's January": standing at the end of December 2023, every model forecasts
each month of 2024 using only data up to then; the same is repeated for 2025.

  1. Tuning: model settings (and whether to keep 2021) are chosen on the 2024 test only.
  2. Final check: 2025 is used once, with the chosen settings, to confirm the choice.
  3. Ranges: the chosen model's misses in those tests set a "9 months out of 10" range
     for every total we report (national, each origin region, each section).
  4. 2026: the chosen model is retrained on 2021-2025 and forecasts 2026. Only January
     and February (before the war) are checked here; step 5 uses the rest.

Writes reports/step4_models.md, four charts in reports/figures/, and the 2026 forecasts
and ranges in data/interim/ for step 5.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from features import FEATURES, NUMERIC, PROCESSED_DIR, fold_tables, load_calendar, load_core
from models import RATIO_FEATURES, BoostedTrees, LinearModel, SameMonth, SimpleAverage, clipped_target
from config import ROOT

warnings.filterwarnings("ignore", category=RuntimeWarning)

INTERIM_DIR = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "reports" / "step4_models.md"
TUNE_YEAR, CHECK_YEAR, FORECAST_YEAR = 2024, 2025, 2026
LEVELS = {"National total": [], "Origin region": ["region"], "Section": ["section_id"], "Series (HS2 x region)": ["series"]}
REPORT_LEVELS = ["National total", "Origin region", "Section"]      # what step 5 reports, so what tuning optimises
RANGE_COVERAGE = 0.9                                                # "9 months out of 10"

# Candidates tried on the 2024 test. The first entry of each family is the default.
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
    cols = [c for c in ("actual", "forecast") if c in frame]
    return frame.groupby([*keys, "target_month"], observed=True)[cols].sum()


def miss(frame: pd.DataFrame, keys: list[str]) -> float:
    """Average monthly miss: sum of |actual - forecast| over all totals and months, as % of the actual.

    (Step 3 divided by the forecast. With several models the actual is the fairer yardstick:
    it is the same for every model, so a model can't look better by forecasting too high.)
    """
    t = totals(frame, keys)
    return float((t["actual"] - t["forecast"]).abs().sum() / t["actual"].sum() * 100)


def score(frame: pd.DataFrame) -> dict:
    out = {level: miss(frame, keys) for level, keys in LEVELS.items()}
    out["Bias %"] = float((frame["forecast"].sum() / frame["actual"].sum() - 1) * 100)
    out["Reported levels (average)"] = float(np.mean([out[level] for level in REPORT_LEVELS]))
    return out


def describe(params: dict) -> str:
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
    key = "Reported levels (average)"
    best_first_year = int(tuning[tuning["family"].isin(["Linear", "Trees"])].sort_values(key).iloc[0]["first_year"])
    chosen = tuning[tuning["first_year"] == best_first_year]
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
    """Monthly % errors (actual / forecast - 1) of every reported total."""
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
    """Half-width of the "9 months out of 10" range for each total: 90th percentile of |error|."""
    return (errors.assign(abs_error=errors["error"].abs())
            .groupby(["level", "name"])["abs_error"].quantile(RANGE_COVERAGE).rename("range").reset_index())


def coverage(errors: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Share of months whose actual value fell inside forecast +/- range, by level."""
    merged = errors.merge(ranges, on=["level", "name"])
    merged["inside"] = merged["error"].abs() <= merged["range"]
    return merged.groupby("level")["inside"].agg(["mean", "size"]).rename(columns={"mean": "share inside", "size": "months"})


def forecast_2026(core, calendar, choice: dict):
    """Retrain the chosen model on 2021-2025 and forecast 2026 from the end of December 2025."""
    panel, train, test = fold_tables(core, calendar, FORECAST_YEAR, first_year=choice["first_year"])
    params = choice["best"][choice["winner"]]
    model = (LinearModel if choice["winner"] == "Linear" else BoostedTrees)(**params).fit(train)
    frame = forecasts_frame(panel, test, model.predict(test))
    return model, frame, train


def main() -> int:
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
    errors = {year: aggregate_errors(frames[year][winner_key]) for year in (TUNE_YEAR, CHECK_YEAR)}
    calib = coverage(errors[CHECK_YEAR], ranges_from(errors[TUNE_YEAR]))
    ranges = ranges_from(pd.concat(errors.values()))

    print("4/4 Forecast for 2026 ...", flush=True)
    model, frame26, train26 = forecast_2026(core, calendar, choice)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    frame26.to_parquet(INTERIM_DIR / "step4_forecast_2026.parquet", index=False)
    ranges.to_csv(INTERIM_DIR / "step4_ranges.csv", index=False)

    results = {"tuning": tuning, "choice": choice, "check": check, "frames": frames, "errors": errors,
               "calibration": calib, "ranges": ranges, "model": model, "forecast_2026": frame26, "train_2026": train26}
    import step4_report                                   # charts and report live in their own module
    step4_report.write(results, core, calendar)
    print(f"Done in {time.time() - t0:.0f}s. Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
