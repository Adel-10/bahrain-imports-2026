"""Step 4a: turn the cleaned imports into a table the forecasting models can learn from.

The forecasting set-up
----------------------
* **Series.** A series is one line of monthly numbers: the value of one HS2 chapter from one
  origin region (core goods only, as decided in step 3), for example "chapter 84, machinery,
  from Europe". Series with imports in fewer than 90% of months are too patchy to forecast on
  their own, so they are added together into one "other" series per section and region. That
  gives about 660 series, and their forecasts add up to every total the reports need.
* **Origin and horizon.** Pretend it is the end of a month (the "origin"). Using only what was
  known by then, describe each series (its recent level, trend, the same month last year, ...)
  and ask the model to predict each of the next 12 months (the "horizon", 1-12).
* **Training and testing.** Training uses every month-end as an origin, as long as the target
  month is before the test year. Testing uses the end of December only, exactly as the 2026
  forecast is made: standing at the end of December 2025, forecast January-December 2026.

Each row of the resulting table is one (series, origin, horizon) with the features known at the
origin and, when it is known, the actual value in the target month.

Why the numbers are divided by each series' scale
-------------------------------------------------
Series range from a few thousand to tens of millions of dinars a month. Every value is
divided by the series' own recent level (its average month over the last 12 months, the
"scale"), so the target becomes "this month as a multiple of a normal recent month" and a small
series and a big one look alike. One model can then learn from all 660 series at once, which
gives it far more examples than any one series could. Predictions are multiplied back by the
scale afterwards.

Pros: shared learning across series, a target that is roughly comparable everywhere, and no
need for a separate model per series. Cons: the scale itself is a noisy 12-month average, and
a series that recently had one huge month gets a temporarily inflated scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import REFERENCE_DIR, ROOT
from reference import CORE, REGIONS, add_analysis_group, add_region

PROCESSED_DIR = ROOT / "data" / "processed"
REGULAR_SHARE = 0.9          # a series needs imports in at least 90% of history months to stand alone
HORIZONS = range(1, 13)      # forecast 1 to 12 months ahead

# Feature columns, in the order the models see them. Ratios are relative to the series' scale.
#   horizon            months ahead (1-12)
#   month ... eid      calendar of the target month (Ramadan days, Eid flags, December flag)
#   covid_target       COVID restrictions in force in the target month
#   covid_window       share of the last 12 months under COVID restrictions
#   log_scale          size of the series (log of its recent average month)
#   last_1, mean_3/6   the latest month and the averages of the last 3 and 6 months
#   prev_12            the average month of the 12 months before that (year-ago level)
#   growth             recent 12-month level / previous 12-month level
#   volatility         how much the last 12 months varied (standard deviation / scale)
#   zero_share         share of the last 24 months with no imports at all
#   same_month_1y/2y   the target's calendar month one and two years earlier
CATEGORICAL = ["section_id", "region_id", "hs2_id"]
NUMERIC = [
    "horizon", "month", "days_in_month", "ramadan_days", "eid_al_fitr", "eid_al_adha", "december",
    "covid_target", "covid_window",
    "log_scale", "last_1", "mean_3", "mean_6", "prev_12", "growth", "volatility", "zero_share",
    "same_month_1y", "same_month_2y",
]
FEATURES = CATEGORICAL + NUMERIC


@dataclass
class Panel:
    """Monthly values for every series: `values` is series x months, `info` describes each series."""
    values: np.ndarray            # BD, shape (n_series, n_months)
    months: pd.DatetimeIndex
    info: pd.DataFrame            # one row per series: series_id, section_id, region, hs2, region_id, hs2_id


def load_core() -> pd.DataFrame:
    """Core goods rows with section, HS2 chapter, HS4 heading, origin region, month and value."""
    cols = ["date", "section_id", "product_group", "hs2", "hs4", "iso2", "value_bd"]
    df = pd.read_parquet(PROCESSED_DIR / "imports_clean.parquet", columns=cols)
    add_analysis_group(df)
    add_region(df)
    core = df[df["group"] == CORE]
    return core[["date", "section_id", "hs2", "hs4", "region", "value_bd"]].assign(
        hs2=core["hs2"].astype(str), hs4=core["hs4"].astype(str), region=core["region"].astype(str))


def load_calendar() -> pd.DataFrame:
    """The monthly calendar features built in step 2, indexed by month."""
    return pd.read_csv(REFERENCE_DIR / "calendar.csv", parse_dates=["date"]).set_index("date")


ALL_ORIGINS = "All origins"     # region label when series are not split by origin (a check in step 4)


def build_panel(core: pd.DataFrame, first_month: str, last_history_month: str, last_month: str,
                product: str = "hs2", by_region: bool = True) -> Panel:
    """Series x month panel from `first_month` to `last_month`.

    A series is one `product` code (HS2 chapter by default, or HS4 heading) from one origin
    region, or from all origins together when `by_region` is False. Which series stand
    alone is decided using only months up to `last_history_month` (what would have been
    known at forecast time). Patchy series are pooled into an "other" series for their
    section and region.
    """
    months = pd.date_range(first_month, last_month, freq="MS")      # "MS" = the first day of each month
    data = core[(core["date"] >= months[0]) & (core["date"] <= months[-1])]
    if not by_region:
        data = data.assign(region=ALL_ORIGINS)
    wide = (data.groupby(["section_id", "region", product, "date"], observed=True)["value_bd"].sum()
            .unstack("date", fill_value=0.0).reindex(columns=months, fill_value=0.0))
    wide.index = wide.index.rename(["section_id", "region", "hs2"])     # the product column is called hs2 below
    # Decide which series are regular using history only, so the test year can't influence the choice.
    history = wide.loc[:, wide.columns <= pd.Timestamp(last_history_month)]
    regular = (history > 0).sum(axis=1) >= np.ceil(REGULAR_SHARE * history.shape[1])

    standalone = wide[regular]
    pooled = wide[~regular].groupby(level=["section_id", "region"]).sum()
    pooled.index = pd.MultiIndex.from_tuples([(s, r, "other") for s, r in pooled.index], names=wide.index.names)
    pooled = pooled[pooled.sum(axis=1) > 0]
    both = pd.concat([standalone, pooled]).sort_index()

    # Describe each series. The numeric ids are what the models see; series_id is a readable label.
    info = both.index.to_frame(index=False)
    info["series_id"] = info["hs2"].where(info["hs2"] != "other", "other-s" + info["section_id"].astype(str)) + " | " + info["region"]
    info["region_id"] = info["region"].map({r: i for i, r in enumerate([*REGIONS, ALL_ORIGINS])}).astype(int)
    info["hs2_id"] = info["hs2"].map(lambda h: 0 if h == "other" else int(h[:2])).astype(int)
    if product != "hs2":
        info = info.rename(columns={"hs2": product})
        info["hs2"] = info[product].map(lambda h: "other" if h == "other" else h[:2])
    return Panel(values=both.to_numpy(dtype=float), months=months, info=info)


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b, with missing (NaN) wherever b is zero, instead of infinity or a warning."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def make_examples(panel: Panel, calendar: pd.DataFrame, origins: list[pd.Timestamp],
                  horizons=HORIZONS, last_target: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (series, origin, horizon) with features known at the origin.

    Rows whose target month is after `last_target` (or after the panel) are dropped,
    so a training table never peeks at the test year. Each row keeps `scale`
    (the series' average month over the 12 months up to the origin) and, where known,
    `actual` (the value in the target month, in BD) and `target` (actual / scale).
    """
    v = panel.values
    months = panel.months
    last_target = min(last_target or months[-1], months[-1])
    # The calendar must reach 12 months past the last origin, because targets can be that far ahead.
    cal = calendar.reindex(months.union(pd.date_range(months[0], periods=len(months) + 12, freq="MS")))
    covid = cal["covid_restrictions"].to_numpy()
    n_series = len(panel.info)
    frames = []
    for origin in origins:
        i = months.get_loc(origin)                    # column of the origin month in the panel
        if i < 11:
            raise ValueError(f"origin {origin:%b %Y} needs 12 months of history")
        # Features that describe each series at the origin. They are the same for every horizon.
        # Only columns up to i (the origin) are used, so nothing from the future leaks in.
        last12 = v[:, i - 11:i + 1]
        scale = last12.mean(axis=1)
        prev12 = v[:, i - 23:i - 11].mean(axis=1) if i >= 23 else np.full(n_series, np.nan)
        base = {
            "log_scale": np.log1p(scale),
            "last_1": _safe_ratio(v[:, i], scale),
            "mean_3": _safe_ratio(v[:, i - 2:i + 1].mean(axis=1), scale),
            "mean_6": _safe_ratio(v[:, i - 5:i + 1].mean(axis=1), scale),
            "prev_12": _safe_ratio(prev12, scale),
            "growth": _safe_ratio(scale, prev12),
            "volatility": _safe_ratio(last12.std(axis=1), scale),
            "zero_share": (v[:, max(0, i - 23):i + 1] == 0).mean(axis=1),
            "covid_window": covid[i - 11:i + 1].mean(),
        }
        # One block of rows per horizon: the same series features plus the target month's calendar.
        for h in horizons:
            t = i + h                                 # column of the target month
            target_month = origin + pd.DateOffset(months=h)
            if target_month > last_target:
                break
            row = cal.loc[target_month]
            frame = {
                "series": np.arange(n_series), "origin": origin, "target_month": target_month, "horizon": h,
                "month": target_month.month, "days_in_month": row["days_in_month"],
                "ramadan_days": row["ramadan_days"], "eid_al_fitr": row["eid_al_fitr"],
                "eid_al_adha": row["eid_al_adha"], "december": int(target_month.month == 12),
                "covid_target": row["covid_restrictions"],
                "same_month_1y": _safe_ratio(v[:, t - 12], scale),
                "same_month_2y": _safe_ratio(v[:, t - 24], scale) if t - 24 >= 0 else np.nan,
                "scale": scale,
                **base,
            }
            if t < v.shape[1]:                        # the target is known (it isn't for future months)
                frame["actual"] = v[:, t]
                frame["target"] = _safe_ratio(v[:, t], scale)
            frames.append(pd.DataFrame(frame))
    out = pd.concat(frames, ignore_index=True)
    info = panel.info[["section_id", "region_id", "hs2_id"]].reset_index(drop=True)
    out = out.join(info, on="series")
    return out


def fold_tables(core: pd.DataFrame, calendar: pd.DataFrame, test_year: int, first_year: int = 2021,
                last_month: str | None = None, product: str = "hs2",
                by_region: bool = True) -> tuple[Panel, pd.DataFrame, pd.DataFrame]:
    """Panel, training table and test table for one "pretend it's January of `test_year`" run.

    Training rows use every month-end origin whose targets fall before `test_year`.
    Test rows use the end of December of the previous year as the origin.

    The first origin is December of `first_year`, because the features need 12 months of history.
    This is a rolling-origin (or "expanding window") backtest: the model for 2025 learns from
    2021-2024, the model for 2026 from 2021-2025, and so on.
    """
    end_of_history = pd.Timestamp(f"{test_year - 1}-12-01")
    last_month = last_month or f"{test_year}-12-01"
    available = core["date"].max()
    last_month = min(pd.Timestamp(last_month), available)
    panel = build_panel(core, f"{first_year}-01-01", end_of_history.strftime("%Y-%m-%d"), last_month.strftime("%Y-%m-%d"),
                        product=product, by_region=by_region)
    first_origin = pd.Timestamp(f"{first_year}-12-01")
    train_origins = list(pd.date_range(first_origin, end_of_history - pd.DateOffset(months=1), freq="MS"))
    train = make_examples(panel, calendar, train_origins, last_target=end_of_history)
    test = make_examples(panel, calendar, [end_of_history])
    train = train[train["scale"] > 0].reset_index(drop=True)      # nothing to learn from a series with no recent imports
    return panel, train, test
