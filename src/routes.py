"""Step 6: which routes were cut, how goods got around them, and how far supply chains had come back.

    python src/routes.py          (run src/backtest.py first: it saves the forecasts used here)

The idea in plain words
-----------------------
Goods reach Bahrain in five ways: container ships, car carriers and general cargo ships (all at
Khalifa Bin Salman Port), planes (Bahrain International Airport), and trucks over the King Fahd
Causeway from Saudi Arabia. The port and the airport publish monthly figures; the causeway doesn't.

1. **Routes.** The port and airport figures for 2026 are compared with the same months of 2025.
2. **Exposure.** Step 5's comparison (actual imports against the forecast without the war, with
   the same normal ranges) is repeated for groups that depend on different routes: goods from GCC
   countries (which can come by road), from the rest of the Middle East, and from beyond; and, for
   goods from beyond the Middle East, light goods (worth flying) against heavy ones.
3. **Elimination.** Containers and planes can't have carried more than a certain share of what
   arrived; the rest must have come by road or general cargo ship.
4. **Adapting.** How far each supply chain had come back by the latest period, and what the port
   and airport figures (published two months ahead of the import data) say about August and later.

Writes reports/step6_routes.md, four charts, and data/interim/step6_*.csv (read by summary.py).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from backtest import aggregate_errors
from config import ROOT
from reference import CORE, PRE_WAR_2026, add_analysis_group, add_region, war_periods_2026
from shortfall import build_periods, period_gaps, window_ranges

warnings.filterwarnings("ignore", category=RuntimeWarning)

INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "step6_routes.md"

# Origin groups by how goods can reach Bahrain.
GCC, OTHER_ME, BEYOND = "GCC", "Other Middle East", "Beyond the Middle East"
ORIGIN_ROUTES = [GCC, OTHER_ME, BEYOND]

# Weight classes, by value per kilogram of each series in 2024-2025.
LIGHT_FROM = 5.0          # BD per kg (about $13): worth flying
HEAVY_BELOW = 1.0         # BD per kg
LIGHT, MIDDLE, HEAVY = "Light", "Middle", "Heavy"
VEHICLES = "Vehicles and transport"   # HS section 17: own kind of ship (car carriers), kept apart
VEHICLES_SECTION = 17
WEIGHT_CLASSES = [LIGHT, MIDDLE, HEAVY, VEHICLES]
AIRCRAFT_ENGINES = {"8409", "8411"}   # aircraft engines and their parts: large, irregular purchases (see step 5)

ORIGIN_LEVEL = "Origin route"
WEIGHT_LEVEL = "Beyond the Middle East, by weight"

# Port and airport indicators: column, name, unit.
INDICATORS = [
    ("port_import_teu", "Containers unloaded", "TEU"),
    ("port_container_vessels", "Container ship visits", "ships"),
    ("port_roro_vehicles", "Vehicles unloaded from car carriers", "vehicles"),
    ("port_general_cargo_ft", "General cargo handled", "freight tonnes"),
    ("port_general_cargo_vessels", "General cargo ship visits", "ships"),
    ("air_import_t", "Air cargo and mail unloaded", "tonnes"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_core_weight() -> pd.DataFrame:
    """Core goods rows with value, weight, origin country and region."""
    cols = ["date", "section_id", "product_group", "hs2", "hs4", "iso2", "value_bd", "weight_kg"]
    df = pd.read_parquet(PROCESSED_DIR / "imports_clean.parquet", columns=cols)
    add_analysis_group(df)
    add_region(df)
    core = df[df["group"] == CORE]
    return core[["date", "section_id", "hs2", "hs4", "iso2", "region", "value_bd", "weight_kg"]].assign(
        hs2=core["hs2"].astype(str), hs4=core["hs4"].astype(str), iso2=core["iso2"].astype(str),
        region=core["region"].astype(str))


def origin_route(region: str) -> str:
    return region if region in (GCC, OTHER_ME) else BEYOND


def weight_class(section_id: int, bd_per_kg: float) -> str:
    if section_id == VEHICLES_SECTION:
        return VEHICLES
    if np.isnan(bd_per_kg):          # a handful of tiny pooled series with no weight recorded
        return MIDDLE
    return LIGHT if bd_per_kg >= LIGHT_FROM else (HEAVY if bd_per_kg < HEAVY_BELOW else MIDDLE)


def series_key(core: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """For each import row, the model series it belongs to: (section, region, chapter or "other")."""
    standalone = set(zip(frame["section_id"], frame["region"], frame["hs2"].astype(str)))
    keys = zip(core["section_id"], core["region"], core["hs2"])
    return pd.Series([(s, r, h if (s, r, h) in standalone else "other") for s, r, h in keys], index=core.index)


def value_density(core: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """BD per kg of each model series in 2024-2025, from rows with a recorded weight."""
    hist = core[core["date"].dt.year.isin([2024, 2025]) & (core["weight_kg"] > 0)]
    keyed = hist.assign(key=series_key(hist, frame))
    sums = keyed.groupby("key")[["value_bd", "weight_kg"]].sum()
    return sums["value_bd"] / sums["weight_kg"]


def tag_series(frame: pd.DataFrame, density: pd.Series) -> pd.DataFrame:
    """Add each series' origin route and weight class."""
    out = frame.copy()
    keys = list(zip(out["section_id"], out["region"], out["hs2"].astype(str)))
    out["origin_route"] = out["region"].map(origin_route)
    out["bd_per_kg"] = [density.get(k, np.nan) for k in keys]
    out["weight_class"] = [weight_class(s, d) for s, d in zip(out["section_id"], out["bd_per_kg"])]
    return out


# ---------------------------------------------------------------------------
# 2. Exposure: gaps for route groups
# ---------------------------------------------------------------------------
def route_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly totals for the route groups, in the shape step 5's functions expect."""
    parts = []
    t = frame.groupby(["origin_route", "target_month"])[["actual", "forecast"]].sum().reset_index()
    parts.append(t.rename(columns={"origin_route": "name"}).assign(level=ORIGIN_LEVEL))
    beyond = frame[frame["origin_route"] == BEYOND]
    t = beyond.groupby(["weight_class", "target_month"])[["actual", "forecast"]].sum().reset_index()
    parts.append(t.rename(columns={"weight_class": "name"}).assign(level=WEIGHT_LEVEL))
    out = pd.concat(parts, ignore_index=True)
    out["error"] = out["actual"] / out["forecast"] - 1
    return out[["level", "name", "target_month", "actual", "forecast", "error"]]


def weight_class_shares(core: pd.DataFrame, frame: pd.DataFrame, density: pd.Series) -> pd.DataFrame:
    """Each weight class's share of the value and weight of goods from beyond the Middle East, 2024-2025."""
    hist = core[core["date"].dt.year.isin([2024, 2025])]
    keys = series_key(hist, frame)
    beyond = [origin_route(k[1]) == BEYOND for k in keys]
    classes = pd.Series([weight_class(k[0], density.get(k, np.nan)) for k in keys], index=hist.index)
    sums = hist[beyond].groupby(classes[beyond])[["value_bd", "weight_kg"]].sum()
    return sums / sums.sum() * 100


def engines_check(core: pd.DataFrame, frame: pd.DataFrame, density: pd.Series, gaps: pd.DataFrame) -> pd.DataFrame:
    """Light goods from beyond the Middle East, with aircraft engines held at their 2025 average month.

    Aircraft engines are light, valuable and bought irregularly, so one airline's purchases can
    move the light group by a lot. As in step 5, their actual value is swapped for their normal value.
    """
    rows = core[core["hs4"].isin(AIRCRAFT_ENGINES)]
    rows = rows.assign(key=series_key(rows, frame))
    rows = rows[[origin_route(k[1]) == BEYOND and weight_class(k[0], density.get(k, np.nan)) == LIGHT
                 for k in rows["key"]]]
    monthly = rows.groupby("date")["value_bd"].sum()
    normal = monthly[monthly.index.year == 2025].mean()
    light = gaps[(gaps["level"] == WEIGHT_LEVEL) & (gaps["name"] == LIGHT)].copy()
    adjusted = []
    for _, g in light.iterrows():
        months = [pd.Timestamp(2026, m, 1) for m in g["month_list"]]
        swing = sum(monthly.get(m, 0.0) - normal for m in months)
        adjusted.append((g["actual"] - swing) / g["forecast"] * 100 - 100)
    light["gap_pct_without_engines"] = adjusted
    return light[["period", "gap_pct", "gap_pct_without_engines"]]


# ---------------------------------------------------------------------------
# 1. Routes: port and airport against the same months of 2025
# ---------------------------------------------------------------------------
def months_in(series: pd.Series, year: int, months: list[int]) -> pd.Series:
    s = series.dropna()
    return s[(s.index.year == year) & s.index.month.isin(months)]


def compare_with_2025(series: pd.Series, months: list[int]) -> dict:
    """2026 total for the months that have data, against the same months of 2025."""
    now = months_in(series, 2026, months)
    if now.empty:
        return {"months": [], "now": np.nan, "before": np.nan, "pct_of_normal": np.nan}
    have = list(now.index.month)
    before = months_in(series, 2025, have).sum()
    return {"months": have, "now": now.sum(), "before": before,
            "pct_of_normal": now.sum() / before * 100 if before else np.nan}


def route_periods(route: pd.DataFrame) -> dict[str, list[int]]:
    """The 2026 periods the port and airport figures cover (they run ahead of the import data)."""
    last = int(route.set_index("date")[[c for c, _, _ in INDICATORS]].dropna(how="all").index.max().month)
    return {**PRE_WAR_2026, **war_periods_2026(last)}


def indicator_table(route: pd.DataFrame, core: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """% of normal (same months of 2025) for every indicator and period, plus core goods by weight."""
    r = route.set_index("date")
    series = {name: r[col] for col, name, _ in INDICATORS}
    series["Core goods imported, by weight"] = core.groupby("date")["weight_kg"].sum() / 1e3
    rows = []
    for name, s in series.items():
        for label, months in periods.items():
            rows.append({"indicator": name, "period": label, **compare_with_2025(s, months)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Elimination
# ---------------------------------------------------------------------------
def elimination(core: pd.DataFrame, route: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """The most containers and planes can have carried, and so the least that came another way.

    For each period, by weight of core goods:
      * containers: at most last year's weight of core goods x (containers unloaded this year /
        last year). That is the same as assuming every import came in a container last year, the
        most generous assumption possible, and that a container held no more than usual;
      * planes: all air cargo and mail unloaded, as if every tonne were core goods;
      * whatever is left must have come another way: by truck over the causeway or by general
        cargo ship (or through a port the data doesn't cover).
    """
    r = route.set_index("date")
    weight = core.groupby("date")["weight_kg"].sum() / 1e3                  # tonnes
    gcc = core[core["region"] == GCC].groupby("date")["weight_kg"].sum() / 1e3
    rows = []
    for label, months in periods.items():
        w0, w1 = months_in(weight, 2025, months).sum(), months_in(weight, 2026, months).sum()
        c0, c1 = months_in(r["port_import_teu"], 2025, months).sum(), months_in(r["port_import_teu"], 2026, months).sum()
        air1 = months_in(r["air_import_t"], 2026, months).sum()
        gc = months_in(r["port_general_cargo_ft"], 2026, months)
        containers_max = w0 * c1 / c0
        other_min = w1 - containers_max - air1
        rows.append({
            "period": label, "months": len(months),
            "weight_2025_t": w0, "weight_2026_t": w1, "weight_pct_of_2025": w1 / w0 * 100,
            "containers_pct_of_2025": c1 / c0 * 100,
            "air_pct_of_2025": air1 / months_in(r["air_import_t"], 2025, months).sum() * 100,
            "containers_max_t": containers_max, "air_t": air1,
            "containers_air_max_share": (containers_max + air1) / w1 * 100,
            "other_min_t": other_min, "other_min_share": other_min / w1 * 100,
            "general_cargo_ft": gc.sum() if len(gc) == len(months) else np.nan,
            "gcc_share_2025": months_in(gcc, 2025, months).sum() / w0 * 100,
            "gcc_share_2026": months_in(gcc, 2026, months).sum() / w1 * 100,
            "gcc_weight_pct_of_2025": months_in(gcc, 2026, months).sum() / months_in(gcc, 2025, months).sum() * 100,
        })
    return pd.DataFrame(rows)


def cars_without_car_carriers(core: pd.DataFrame, route: pd.DataFrame) -> pd.DataFrame:
    """Cars (HS 8703) from East Asia and Europe each month of 2026, beside vehicles unloaded from car carriers."""
    cars = core[(core["hs4"] == "8703") & core["region"].isin(["East and South-East Asia", "Europe"])]
    monthly = cars.groupby("date")["value_bd"].sum()
    normal = monthly[monthly.index.year == 2025].mean()
    r = route.set_index("date")["port_roro_vehicles"]
    roro_normal = r[r.index.year == 2025].mean()
    m26 = monthly[monthly.index.year == 2026]
    return pd.DataFrame({"cars_bd": m26, "cars_pct_of_2025_avg": m26 / normal * 100,
                         "roro_vehicles": r.reindex(m26.index), "roro_pct_of_2025_avg": r.reindex(m26.index) / roro_normal * 100})


# ---------------------------------------------------------------------------
# 4. Adapting
# ---------------------------------------------------------------------------
def adapting(gaps: pd.DataFrame, war_periods: list[str]) -> pd.DataFrame:
    """For every total: % of normal in each war period, the worst one, the latest one, and how far back it came.

    "Came back x% of the way" = (latest - worst) / (100 - worst), in % of normal.
    """
    g = gaps[gaps["period"].isin(war_periods)].copy()
    g["pct_of_normal"] = g["actual"] / g["forecast"] * 100
    wide = g.pivot_table(index=["level", "label"], columns="period", values="pct_of_normal", sort=False)[war_periods]
    latest = war_periods[-1]
    inside = g[g["period"] == latest].set_index(["level", "label"])
    out = wide.copy()
    out["worst"] = wide.min(axis=1)
    out["worst_period"] = wide.idxmin(axis=1)
    out["latest"] = wide[latest]
    out["latest_inside_range"] = ~inside["outside"].reindex(out.index).astype(bool)
    out["latest_range_pct"] = inside["range_pct"].reindex(out.index)
    out["forecast"] = gaps[gaps["period"].str.startswith("Since")].set_index(["level", "label"])["forecast"].reindex(out.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["came_back_pct"] = np.where(out["worst"] < 100, (out["latest"] - out["worst"]) / (100 - out["worst"]) * 100,
                                        np.nan)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    forecast = pd.read_parquet(INTERIM_DIR / "step4_forecast_2026.parquet")
    backtest = pd.read_parquet(INTERIM_DIR / "step4_backtest.parquet")
    backtest = backtest[backtest["model"] == "linear"]
    route = pd.read_parquet(PROCESSED_DIR / "route_monthly.parquet")
    missing = [c for c, _, _ in INDICATORS if c not in route]
    if missing:
        raise SystemExit(f"route_monthly.parquet has no {missing}. Run: python src/download_data.py, "
                         "then python src/build_dataset.py")
    last_month = int(forecast["target_month"].max().month)
    periods, since = build_periods(last_month)
    all_periods = {**periods, **since}
    war_periods = [p for p in periods if p not in PRE_WAR_2026]
    lengths = {1} | {len(m) for m in all_periods.values()}

    print("1/4 Port and airport figures ...", flush=True)
    core = load_core_weight()
    r_periods = route_periods(route)
    indicators = indicator_table(route, core, r_periods)

    print("2/4 Route groups: gaps and ranges ...", flush=True)
    density = value_density(core, forecast)
    frame = tag_series(forecast, density)
    tests = tag_series(backtest, density)
    ranges = window_ranges(tests, lengths, route_totals)
    gaps = period_gaps(frame, ranges, all_periods, route_totals)
    gaps["month_list"] = gaps["period"].map(all_periods)
    engines = engines_check(core, forecast, density, gaps)
    class_share = weight_class_shares(core, forecast, density)

    print("3/4 Elimination ...", flush=True)
    elim = elimination(core, route, {**{p: periods[p] for p in war_periods}, **since})
    cars = cars_without_car_carriers(core, route)

    print("4/4 Adapting, and the report ...", flush=True)
    step5_ranges = window_ranges(backtest, lengths)
    step5_gaps = period_gaps(forecast, step5_ranges, all_periods)
    adapt = adapting(pd.concat([step5_gaps, gaps.drop(columns="month_list")], ignore_index=True), war_periods)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    indicators.drop(columns="months").to_csv(INTERIM_DIR / "step6_indicators.csv", index=False)
    gaps.drop(columns="month_list").to_csv(INTERIM_DIR / "step6_route_gaps.csv", index=False)
    elim.to_csv(INTERIM_DIR / "step6_elimination.csv", index=False)
    adapt.to_csv(INTERIM_DIR / "step6_adapting.csv", index=False)
    engines.to_csv(INTERIM_DIR / "step6_engines.csv", index=False)

    results = {"route": route, "core": core, "indicators": indicators, "route_periods": r_periods,
               "periods": periods, "since": since, "war_periods": war_periods, "gaps": gaps, "ranges": ranges,
               "engines": engines, "class_share": class_share, "elimination": elim, "cars": cars,
               "adapting": adapt, "nat_gaps": step5_gaps[step5_gaps["level"] == "National total"],
               "last_month": last_month}
    import step6_report
    step6_report.write(results)
    print(f"Done in {time.time() - t0:.0f}s. Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
