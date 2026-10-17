"""Step 5: how far below normal were imports in 2026, and which supply chains were hit hardest?

    python src/shortfall.py          (run src/backtest.py first: it saves the forecasts used here)

Method: a counterfactual forecast
---------------------------------
Step 4's model, standing at the end of December 2025, forecast every month of 2026 as if
nothing had happened. That forecast is the counterfactual: imports without the war. Each
period's actual imports are compared with it:

* the **gap** is actual minus forecast, in BD and as % of the forecast;
* the gap only counts as a real shortfall if it is bigger than the model's normal misses. Those
  set a **range** for each total and each period length (``window_ranges``). Longer periods get
  narrower ranges, because over several months the model's ups and downs partly cancel out. The
  ranges come from step 4's tests on 2024 and 2025, so they describe how wrong this exact model
  is in normal years.
* the BD **lost** in a period is given with a likely range too: the normal value could have been
  anywhere in forecast x (1 +/- range), so the loss could have been anywhere between the two ends.

Pros: the gap is measured against what would have been expected for each specific month
(including growth and the calendar), not against last year, and every claim of a "real"
shortfall is backed by the model's own track record. Cons: the counterfactual is only as good
as the model, and the model can't know about anything else that would have changed in 2026;
the ranges rest on two test years.

Further checks
--------------
* **Second opinion**: the same gaps with the best gradient-boosted trees from step 4.
* **Aircraft**: aircraft, engines and parts are large, irregular purchases, so the national gap is
  recomputed with them held at their normal level, to make sure one delivery isn't read as a
  recovery.
* **Product close-ups**: the same linear model rebuilt at 4-digit level for six everyday products
  (cars, phones, computers, medicines, rice, poultry), each with its own tested range.
* **Bulk materials and precious metals**: too few series for the models, so they are compared with
  the simple rule (the 2025 average month), with ranges from 2023-2025.

Writes reports/step5_shortfall.md, five charts, and data/interim/step5_gaps.csv.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

from backtest import aggregate_errors, forecasts_frame
from config import ROOT
from features import fold_tables, load_calendar, load_core
from models import LinearModel, SimpleAverage
from reference import (BULK, HS_SECTIONS, PRE_WAR_2026, PRECIOUS, add_analysis_group, war_periods_2026)

warnings.filterwarnings("ignore", category=RuntimeWarning)   # ratios with a zero forecast become NaN, as intended

INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "step5_shortfall.md"
COVERAGE = 0.9                                         # ranges hold 9 times out of 10
SECTION_NAMES = {num: name for num, _, name, _, _ in HS_SECTIONS}
AIRCRAFT_HS4 = {"8409", "8411", "8802", "8803"}       # aircraft, engines and their parts: large, irregular purchases
CLOSE_UPS = {                                          # everyday products looked at one by one (HS4 headings)
    "8703": "Cars", "8517": "Phones and network equipment", "8471": "Computers",
    "3004": "Medicines", "1006": "Rice", "0207": "Poultry meat",
}


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
def build_periods(last_month: int) -> tuple[dict, dict]:
    """(periods to report, the "since the war" period). Periods only cover months that exist."""
    war = war_periods_2026(last_month)
    periods = {**PRE_WAR_2026, **war}
    since = {f"Since the war (Mar-{pd.Timestamp(2026, last_month, 1):%b})": list(range(3, last_month + 1))}
    return periods, since


# ---------------------------------------------------------------------------
# Totals, ranges and gaps
# ---------------------------------------------------------------------------
def name_of(level: str, name: str) -> str:
    """Readable label for a total: section numbers become section names."""
    return SECTION_NAMES.get(int(name), name) if level == "Section" else name


def window_ranges(backtest: pd.DataFrame, lengths: set[int], totals=aggregate_errors) -> pd.DataFrame:
    """Range for a k-month total, for every reported total and every k in `lengths`.

    From the 2024 and 2025 tests: take every run of k consecutive months, compare the model's
    total with the actual total, and find the distance that 9 runs out of 10 stayed within.
    `totals` turns series forecasts into the reported totals (columns level, name, target_month,
    actual, forecast); step 6 passes its own groupings.

    Runs are taken within each test year only (a run never spans 2024 and 2025), because each
    year's forecasts were made from a different December. Two years give 24 one-month errors per
    total but only 2 x 8 = 16 five-month runs, and overlapping runs are not independent, so the
    ranges for long periods are rougher than those for single months.
    """
    rows = []
    for _, frame in backtest.groupby("year"):
        errors = totals(frame).sort_values("target_month")
        for (level, name), g in errors.groupby(["level", "name"]):
            for k in lengths:
                a = g["actual"].rolling(k).sum()
                f = g["forecast"].rolling(k).sum()
                for err in ((a / f) - 1).dropna():
                    rows.append((level, name, k, abs(err)))
    df = pd.DataFrame(rows, columns=["level", "name", "months", "abs_error"])
    return df.groupby(["level", "name", "months"])["abs_error"].quantile(COVERAGE).rename("range").reset_index()


def period_gaps(frame: pd.DataFrame, ranges: pd.DataFrame, periods: dict, totals=aggregate_errors) -> pd.DataFrame:
    """Gap for every reported total and period: BD, % of forecast, its range, and whether it is outside.

    Actual and forecast are added up over the period first, then compared, so a period's gap is
    the gap of its total rather than an average of monthly gaps.
    """
    totals = totals(frame)
    rows = []
    for label, months in periods.items():
        part = totals[totals["target_month"].dt.month.isin(months) & (totals["target_month"].dt.year == 2026)]
        summed = part.groupby(["level", "name"])[["actual", "forecast"]].sum().reset_index()
        summed["period"] = label
        summed["months"] = len(months)
        rows.append(summed)
    out = pd.concat(rows, ignore_index=True).merge(ranges, on=["level", "name", "months"], how="left")
    out["gap"] = out["actual"] - out["forecast"]
    out["gap_pct"] = (out["actual"] / out["forecast"] - 1) * 100
    out["range_pct"] = out["range"] * 100
    out["outside"] = out["gap_pct"].abs() > out["range_pct"]
    # BD lost, with its range: the "normal" value could have been anywhere in forecast x (1 +/- range).
    out["lost"] = -out["gap"]
    out["lost_low"] = out["forecast"] * (1 - out["range"]) - out["actual"]
    out["lost_high"] = out["forecast"] * (1 + out["range"]) - out["actual"]
    out["label"] = [name_of(l, n) for l, n in zip(out["level"], out["name"])]
    return out


def monthly_gaps(frame: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Gap for every reported total and every month of 2026 (with the single-month range)."""
    totals = aggregate_errors(frame)
    totals = totals[totals["target_month"].dt.year == 2026]
    one = ranges[ranges["months"] == 1][["level", "name", "range"]]
    out = totals.merge(one, on=["level", "name"], how="left")
    out["gap"] = out["actual"] - out["forecast"]
    out["gap_pct"] = (out["actual"] / out["forecast"] - 1) * 100
    out["range_pct"] = out["range"] * 100
    out["outside"] = out["gap_pct"].abs() > out["range_pct"]
    out["label"] = [name_of(l, n) for l, n in zip(out["level"], out["name"])]
    return out


# ---------------------------------------------------------------------------
# Extra checks
# ---------------------------------------------------------------------------
def aircraft_effect(core: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """How much aircraft, engines and parts moved the national gap each month.

    Their 2026 value is compared with their 2025 average month; the national gap is then
    recomputed as if they had been at that normal level. If the two gaps differ a lot in a month,
    that month's national figure was moved by an aircraft-related delivery (or its absence).
    """
    air = core[core["hs4"].isin(AIRCRAFT_HS4)].groupby("date")["value_bd"].sum()
    normal = air[air.index.year == 2025].mean()
    nat = national.set_index("target_month")
    out = pd.DataFrame({"aircraft": air.reindex(nat.index).fillna(0.0), "aircraft_normal": normal,
                        "actual": nat["actual"], "forecast": nat["forecast"]})
    out["gap_pct"] = (out["actual"] / out["forecast"] - 1) * 100
    adjusted = out["actual"] - (out["aircraft"] - out["aircraft_normal"])
    out["gap_pct_without_aircraft_swings"] = (adjusted / out["forecast"] - 1) * 100
    return out


def simple_rule_gaps(w: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """Gaps against the simple rule (2025 average month), for groups too small for the models.

    Ranges come from 2023-2025: for every run of k consecutive months, the total against
    k x the previous year's average month; 9 runs out of 10 stayed within the range.

    `w` is a wide table (one row per group, one column per month). This is the same yardstick as
    step 3, extended to periods of several months.
    """
    def months_of(year, months=None):
        mask = w.columns.year == year
        if months is not None:
            mask &= w.columns.month.isin(months)
        return w.loc[:, mask]

    base = months_of(2025).mean(axis=1)
    lengths = {len(m) for m in periods.values()}
    ranges = {}
    for k in lengths:
        errs = []
        for y in (2023, 2024, 2025):
            actual = months_of(y)
            prev = months_of(y - 1).mean(axis=1)
            sums = actual.T.rolling(k).sum().T.iloc[:, k - 1:]          # every run of k months within the year
            errs.append(sums.div(prev * k, axis=0) - 1)
        ranges[k] = pd.concat(errs, axis=1).abs().quantile(COVERAGE, axis=1)
    rows = []
    for label, months in periods.items():
        part = months_of(2026, months)
        if part.shape[1] == 0:
            continue
        actual = part.sum(axis=1)
        forecast = base * part.shape[1]
        for name in w.index:
            gap_pct = (actual[name] / forecast[name] - 1) * 100
            rng = ranges[len(months)][name] * 100
            rows.append({"name": name, "period": label, "actual": actual[name], "forecast": forecast[name],
                         "gap_pct": gap_pct, "range_pct": rng, "outside": abs(gap_pct) > rng})
    return pd.DataFrame(rows)


def close_ups(core, calendar, periods: dict) -> dict:
    """The step 4 linear model rebuilt at HS4 level (all origins together) for a few everyday products.

    The same backtest as step 4 is repeated at this level (2024 and 2025 tests), both to check
    that the model still beats the simple rule here and to give each product its own range.
    Origins are combined because one product from one region is usually too patchy to forecast.
    """
    frames, misses = {}, {}
    for year in (2024, 2025, 2026):
        panel, train, test = fold_tables(core, calendar, year, product="hs4", by_region=False)
        model = LinearModel(ridge=1.0).fit(train)
        frame = forecasts_frame(panel, test, model.predict(test))
        frame["hs4"] = frame["series"].map(panel.info["hs4"])
        frames[year] = frame
        if year < 2026:
            simple = forecasts_frame(panel, test, SimpleAverage().predict(test))
            for label, f in (("linear", frame), ("simple", simple)):
                misses[(year, label)] = float((f["actual"] - f["forecast"]).abs().sum() / f["actual"].sum() * 100)
    # Ranges for each close-up product, from its 2024-2025 test misses (single months and longer runs).
    lengths = {1} | {len(m) for m in periods.values()}
    rows = []
    for year in (2024, 2025):
        f = frames[year]
        for code in CLOSE_UPS:
            g = f[f["hs4"] == code].sort_values("target_month")
            for k in lengths:
                errs = (g["actual"].rolling(k).sum() / g["forecast"].rolling(k).sum() - 1).dropna().abs()
                rows += [(code, k, e) for e in errs]
    ranges = (pd.DataFrame(rows, columns=["hs4", "months", "abs_error"])
              .groupby(["hs4", "months"])["abs_error"].quantile(COVERAGE).rename("range").reset_index())
    f26 = frames[2026]
    f26 = f26[f26["hs4"].isin(CLOSE_UPS)]
    out_rows = []
    for code in CLOSE_UPS:
        g = f26[f26["hs4"] == code]
        for label, months in {**{m: [m] for m in sorted(g["target_month"].dt.month.unique())}, **periods}.items():
            part = g[g["target_month"].dt.month.isin(months)]
            if part.empty:
                continue
            k = len(months)
            rng = ranges[(ranges["hs4"] == code) & (ranges["months"] == k)]["range"].iloc[0] * 100
            gap_pct = (part["actual"].sum() / part["forecast"].sum() - 1) * 100
            out_rows.append({"hs4": code, "product": CLOSE_UPS[code], "period": label, "months": k,
                             "actual": part["actual"].sum(), "forecast": part["forecast"].sum(),
                             "gap_pct": gap_pct, "range_pct": rng, "outside": abs(gap_pct) > rng})
    return {"table": pd.DataFrame(out_rows), "misses": misses}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Compute every gap and check, save them, and write the step 5 report."""
    t0 = time.time()
    forecast = pd.read_parquet(INTERIM_DIR / "step4_forecast_2026.parquet")
    forecast_trees = pd.read_parquet(INTERIM_DIR / "step4_forecast_2026_trees.parquet")
    backtest = pd.read_parquet(INTERIM_DIR / "step4_backtest.parquet")
    # The latest month with actual data decides which periods exist (more appear as the portal publishes).
    last_month = int(forecast["target_month"].max().month)
    periods, since = build_periods(last_month)
    all_periods = {**periods, **since}
    lengths = {1} | {len(m) for m in all_periods.values()}       # period lengths that need a range

    print("1/4 Gaps and ranges ...", flush=True)
    ranges = window_ranges(backtest[backtest["model"] == "linear"], lengths)
    ranges_trees = window_ranges(backtest[backtest["model"] == "trees"], lengths)
    gaps = period_gaps(forecast, ranges, all_periods)
    gaps_trees = period_gaps(forecast_trees, ranges_trees, all_periods)
    monthly = monthly_gaps(forecast, ranges)
    monthly_trees = monthly_gaps(forecast_trees, ranges_trees)

    print("2/4 Aircraft check, bulk materials and precious metals ...", flush=True)
    core = load_core()
    national = monthly[monthly["level"] == "National total"]
    aircraft = aircraft_effect(core, national)
    cols = ["date", "product_group", "hs2", "value_bd", "weight_kg"]
    imports = add_analysis_group(pd.read_parquet(PROCESSED_DIR / "imports_clean.parquet", columns=cols))
    # Bulk materials are measured by weight (a shipload is a shipload), precious metals by value.
    bulk = imports[imports["group"] == BULK]
    bulk_w = bulk.groupby([bulk["product_group"].astype(str), "date"])["weight_kg"].sum().unstack(fill_value=0.0)
    bulk_w.loc["All bulk materials"] = bulk_w.sum()
    precious = imports[imports["group"] == PRECIOUS].groupby("date")["value_bd"].sum().to_frame("Precious metals").T
    simple_groups = {"bulk": simple_rule_gaps(bulk_w, all_periods), "precious": simple_rule_gaps(precious, all_periods)}

    print("3/4 Product close-ups ...", flush=True)
    closeups = close_ups(core, load_calendar(), all_periods)

    print("4/4 Report ...", flush=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([gaps.assign(model="linear"), gaps_trees.assign(model="trees")]).to_csv(
        INTERIM_DIR / "step5_gaps.csv", index=False)
    results = {"periods": periods, "since": since, "gaps": gaps, "gaps_trees": gaps_trees, "monthly": monthly,
               "monthly_trees": monthly_trees, "ranges": ranges, "aircraft": aircraft, "simple_groups": simple_groups,
               "closeups": closeups, "forecast": forecast, "forecast_trees": forecast_trees, "core": core,
               "last_month": last_month}
    import step5_report
    step5_report.write(results)
    print(f"Done in {time.time() - t0:.0f}s. Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
