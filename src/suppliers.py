"""Extra: were products that relied on only a few supplier countries hit harder?

    python src/suppliers.py          (needs step 2's tables; about 30 seconds)

The question
------------
A common lesson from supply shocks is "don't depend on one country". This script tests whether
that was the lesson of 2026 for Bahrain, or whether what mattered was where the suppliers were.

Method
------
1. **Products.** Each 4-digit HS heading (for example 8703, cars) with regular imports is one
   product. Step 4's linear model, rebuilt at this level as in step 5's close-ups, forecasts each
   product's 2026 imports without the war, and is tested on 2024 and 2025 for normal ranges.
2. **Reliance on few countries.** For each product, the *effective number of supplier countries*
   in 2024-2025: 1 / (sum of each country's squared share), the inverse of the
   Herfindahl-Hirschman index. Two equal suppliers give 2; one supplier with 90% and another with
   10% gives about 1.2. Unlike a simple count, it isn't inflated by many tiny suppliers.
3. **Route.** For each product, the share of its 2024-2025 imports from beyond the Middle East
   (normally shipped through the Strait of Hormuz, or flown).
4. **Comparison.** Products are grouped both ways and compared with their forecasts. Because the
   two measures could be tangled together (far-away goods might also have fewer suppliers), a
   weighted regression then looks at both at once, holding value per kg and vehicles constant.
   Bootstrapping (re-running it on 2,000 random re-draws of the products) gives a 90% range for
   each effect without assuming the errors are normally distributed.
5. **Switching.** How much of each product came from countries that had been minor suppliers
   before (under 5%), in 2026 and in normal years, to see whether importers changed suppliers.

Pros: every product is measured against its own tested forecast, and the regression separates
the two explanations. Cons: product-level forecasts are noisy; the regression shows association,
not cause; supplier shares are by country of origin, which can differ from where goods were
shipped from; and it is weighted by size, so a few large products carry a lot of weight (the
report shows two other weightings as a check).

Writes reports/extra_suppliers.md, three charts, and data/interim/extra_suppliers_*.csv.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from backtest import forecasts_frame
from config import ROOT
from features import fold_tables, load_calendar, load_core
from models import LinearModel
from routes import BEYOND, GCC, load_core_weight, origin_route
from shortfall import build_periods, period_gaps, window_ranges

warnings.filterwarnings("ignore", category=RuntimeWarning)   # ratios with zero denominators become NaN, as intended

INTERIM_DIR = ROOT / "data" / "interim"
REPORT_PATH = ROOT / "reports" / "extra_suppliers.md"

PROFILE_YEARS = (2024, 2025)          # supplier shares are measured before the war
TEST_YEARS = (2024, 2025)             # normal years, for the ranges
VEHICLES_SECTION = 17
SPECIAL_CHAPTERS = {"98", "99"}       # national special and confidential codes: not real products
MINOR_SHARE = 0.05                    # a country with under 5% of a product before counts as a minor supplier
FAR_FROM = 0.75                       # "far-dependent": at least 75% from beyond the Middle East
N_BOOT = 2000                         # bootstrap re-draws for the regression ranges
SEED = 0                              # fixed so the ranges are the same on every run

# Group boundaries. Effective suppliers are grouped on a roughly doubling scale; the beyond-Middle-East
# share in quarters, with the top quarter split because most value sits there.
SUPPLIER_LEVEL = "Supplier countries"
BEYOND_LEVEL = "From beyond the Middle East"
SUPPLIER_BINS = [0, 2, 3, 5, 8, np.inf]
SUPPLIER_GROUPS = ["Under 2", "2 to 3", "3 to 5", "5 to 8", "8 or more"]
BEYOND_BINS = [0, 0.25, 0.5, 0.75, 0.9, 1.0 + 1e-9]
BEYOND_GROUPS = ["Under 25%", "25-50%", "50-75%", "75-90%", "90% or more"]

# The regression's explanatory variables, in the units the report quotes them in.
EFFECTS = [
    ("suppliers", "Each doubling of the number of supplier countries"),
    ("beyond", "Each extra 10% of the product from beyond the Middle East"),
    ("density", "Each doubling of value per kg"),
    ("vehicles", "Vehicles and transport (compared with other goods)"),
]
WEIGHTINGS = [("size", "Weighted by size (main)"), ("sqrt", "Weighted by the square root of size"),
              ("equal", "Every product counts equally")]


# ---------------------------------------------------------------------------
# 1. Product forecasts
# ---------------------------------------------------------------------------
def product_forecasts(core: pd.DataFrame, calendar: pd.DataFrame, years=(*TEST_YEARS, 2026)) -> pd.DataFrame:
    """Step 4's linear model at HS4 level (all origins together): one "pretend it's January" run per year.

    Pooled "other" series mix many products, so they are left out, as are the national special
    codes (chapters 98-99), which are not real products.
    """
    frames = []
    for year in years:
        panel, train, test = fold_tables(core, calendar, year, product="hs4", by_region=False)
        model = LinearModel(ridge=1.0).fit(train)
        frame = forecasts_frame(panel, test, model.predict(test))
        frame["hs4"] = frame["series"].map(panel.info["hs4"])
        frames.append(frame.assign(year=year))
    out = pd.concat(frames, ignore_index=True)
    keep = (out["hs4"] != "other") & ~out["hs4"].str[:2].isin(SPECIAL_CHAPTERS)
    return out[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2-3. Supplier profile of each product
# ---------------------------------------------------------------------------
def effective_suppliers(shares: pd.Series) -> float:
    """1 / sum of squared shares: the number of equal-sized suppliers that would be as concentrated."""
    s = shares / shares.sum()
    return float(1.0 / (s ** 2).sum())


def supplier_profile(core_w: pd.DataFrame, years=PROFILE_YEARS) -> pd.DataFrame:
    """For each HS4 product, from the chosen years: value, effective supplier countries, top supplier,
    share from beyond the Middle East, value per kg and whether it is a vehicle.

    Measured on 2024-2025 so the profile describes the supply chain as it was before the war.
    """
    hist = core_w[core_w["date"].dt.year.isin(years)]
    by_country = hist.groupby(["hs4", "iso2"])["value_bd"].sum()
    by_country = by_country[by_country > 0]                       # negative or zero rows can't be shares
    shares = by_country / by_country.groupby(level=0).transform("sum")
    profile = pd.DataFrame({
        "value_bd": by_country.groupby(level=0).sum(),
        "effective_suppliers": 1.0 / (shares ** 2).groupby(level=0).sum(),
        "top_share": shares.groupby(level=0).max(),
        "top_country": shares.groupby(level=0).idxmax().str[1],
    })
    far = hist["region"].map(origin_route) == BEYOND
    profile["beyond_share"] = (hist["value_bd"].where(far, 0.0).groupby(hist["hs4"]).sum()
                               / hist.groupby("hs4")["value_bd"].sum())
    weighed = hist[hist["weight_kg"] > 0].groupby("hs4")[["value_bd", "weight_kg"]].sum()
    profile["bd_per_kg"] = weighed["value_bd"] / weighed["weight_kg"]
    sections = hist.groupby("hs4")["section_id"].first()
    profile["vehicles"] = (sections == VEHICLES_SECTION).astype(float)
    profile["supplier_group"] = pd.cut(profile["effective_suppliers"], SUPPLIER_BINS, labels=SUPPLIER_GROUPS,
                                       right=False).astype(str)
    profile["beyond_group"] = pd.cut(profile["beyond_share"], BEYOND_BINS, labels=BEYOND_GROUPS,
                                     right=False).astype(str)
    return profile


# ---------------------------------------------------------------------------
# 4. Comparisons
# ---------------------------------------------------------------------------
def group_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly totals for both groupings, in the shape step 5's functions expect (as in routes.py)."""
    parts = []
    for level, col in ((SUPPLIER_LEVEL, "supplier_group"), (BEYOND_LEVEL, "beyond_group")):
        t = frame.groupby([col, "target_month"])[["actual", "forecast"]].sum().reset_index()
        parts.append(t.rename(columns={col: "name"}).assign(level=level))
    out = pd.concat(parts, ignore_index=True)
    out["error"] = out["actual"] / out["forecast"] - 1
    return out[["level", "name", "target_month", "actual", "forecast", "error"]]


def product_results(frame26: pd.DataFrame, profile: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Each product's actual and forecast over the given months of 2026, with its supplier profile."""
    part = frame26[frame26["target_month"].dt.month.isin(months)]
    sums = part.groupby("hs4")[["actual", "forecast"]].sum()
    out = profile.join(sums, how="inner")
    out = out[out["forecast"] > 0]
    out["pct_of_normal"] = out["actual"] / out["forecast"] * 100
    return out


def group_medians(products: pd.DataFrame) -> pd.DataFrame:
    """Typical (median) product in each group, plus counts and shares of value.

    The median is used because single products can be far above normal (a restocking surge) and
    would pull an average around; % of normal is capped at 200 for the same reason.
    """
    rows = []
    for level, col, groups in ((SUPPLIER_LEVEL, "supplier_group", SUPPLIER_GROUPS),
                               (BEYOND_LEVEL, "beyond_group", BEYOND_GROUPS)):
        for g in groups:
            sub = products[products[col] == g]
            rows.append({"level": level, "name": g, "products": len(sub),
                         "value_share": sub["value_bd"].sum() / products["value_bd"].sum() * 100,
                         "median_pct_of_normal": sub["pct_of_normal"].clip(0, 200).median(),
                         "beyond_share": (sub["beyond_share"] * sub["value_bd"]).sum() / sub["value_bd"].sum() * 100,
                         "median_suppliers": sub["effective_suppliers"].median()})
    return pd.DataFrame(rows)


def design(products: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """y = % of normal (capped at 0-200); X = constant and the four effects in EFFECTS, in their units.

    Supplier count and value per kg enter as log2, so their coefficient is the effect of a doubling;
    the beyond share is multiplied by 10 so its coefficient is the effect of 10 percentage points.
    """
    y = products["pct_of_normal"].clip(0, 200).to_numpy()
    X = np.column_stack([
        np.ones(len(products)),
        np.log2(products["effective_suppliers"]),
        products["beyond_share"] * 10,
        np.log2(products["bd_per_kg"].fillna(products["bd_per_kg"].median()).clip(lower=1e-3)),
        products["vehicles"],
    ])
    return X, y


def weighted_least_squares(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Coefficients minimising sum(w * (y - Xb)^2), solved on sqrt(w)-scaled rows.

    Multiplying each row of X and y by sqrt(w) turns weighted least squares into ordinary least
    squares, which np.linalg.lstsq solves stably.
    """
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def effects(products: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Each effect, for each weighting, with a 90% bootstrap range (products re-drawn with replacement).

    The same 2,000 re-draws are used for every weighting so the three can be compared fairly. An
    effect is "clear" when its whole 90% range is on one side of zero.

    Weightings: by size (forecast BD, the main one, because the question is about the money); by
    the square root of size (a compromise); and equal (every product counts the same, so small
    products have as much say as large ones). If an effect only appears under one weighting, it
    depends on a few large products.
    """
    X, y = design(products)
    size = products["forecast"].to_numpy()
    weights = {"size": size, "sqrt": np.sqrt(size), "equal": np.ones(len(products))}
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(products), len(products)) for _ in range(n_boot)]
    rows = []
    for wkey, wlabel in WEIGHTINGS:
        w = weights[wkey]
        coef = weighted_least_squares(X, y, w)
        boot = np.array([weighted_least_squares(X[i], y[i], w[i]) for i in draws])
        low, high = np.percentile(boot, [5, 95], axis=0)
        for j, (key, label) in enumerate(EFFECTS, start=1):
            rows.append({"weighting": wkey, "weighting_label": wlabel, "effect": key, "label": label,
                         "points": coef[j], "low": low[j], "high": high[j],
                         "clear": bool(low[j] > 0 or high[j] < 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Switching
# ---------------------------------------------------------------------------
def switching(core_w: pd.DataFrame, profile: pd.DataFrame, years=(2024, 2025, 2026), months=(3, 4, 5, 6, 7),
              minor: float = MINOR_SHARE) -> pd.DataFrame:
    """Share of each year's imports (chosen months) that came from minor suppliers of that product.

    Normal years (2024, 2025) are computed the same way, so 2026 can be compared with the usual
    amount of churn among suppliers.

    A minor supplier had under `minor` of the product's imports in the two years before. Computed for
    all products and for far-dependent ones (at least FAR_FROM from beyond the Middle East), with
    the minor-supplier share split by where those suppliers are.
    """
    rows = []
    products = set(profile.index)
    data = core_w[core_w["hs4"].isin(products)]
    route_of = data.drop_duplicates("iso2").set_index("iso2")["region"].map(origin_route)
    far_products = set(profile.index[profile["beyond_share"] >= FAR_FROM])
    for year in years:
        # Each country's share of each product in the two years before `year`.
        before = data[data["date"].dt.year.isin([year - 2, year - 1])].groupby(["hs4", "iso2"])["value_bd"].sum()
        shares = (before / before.groupby(level=0).transform("sum")).rename("share_before")
        now = data[(data["date"].dt.year == year) & data["date"].dt.month.isin(months)]
        now = now.groupby(["hs4", "iso2"])["value_bd"].sum().rename("value").reset_index()
        now = now.join(shares, on=["hs4", "iso2"])
        now["share_before"] = now["share_before"].fillna(0.0)
        now["minor"] = now["share_before"] < minor
        now["route"] = now["iso2"].map(route_of)
        for scope, sub in (("All products", now), ("Far-dependent products", now[now["hs4"].isin(far_products)])):
            total = sub["value"].sum()
            row = {"year": year, "scope": scope, "minor_share": sub.loc[sub["minor"], "value"].sum() / total * 100,
                   "new_share": sub.loc[sub["share_before"] == 0, "value"].sum() / total * 100,
                   "gcc_share": sub.loc[sub["route"] == GCC, "value"].sum() / total * 100}
            for route in (GCC, "Other Middle East", BEYOND):
                row[f"minor_from_{route}"] = sub.loc[sub["minor"] & (sub["route"] == route), "value"].sum() / total * 100
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Names for the products shown in the report
# ---------------------------------------------------------------------------
PRODUCT_NAMES = {"1006": "Rice", "2804": "Silicon", "7303": "Cast-iron pipes", "8702": "Buses",
                 "9101": "Wristwatches (precious metal)", "9102": "Wristwatches (other)"}
SMALL_WORDS = {"of", "and", "the", "for"}


def tidy_name(text: str) -> str:
    """'DEMOCRATIC REPUBLIC OF CONGO' -> 'Democratic Republic of Congo'."""
    words = str(text).strip().lower().split()
    return " ".join(w if (i and w in SMALL_WORDS) else w.capitalize() for i, w in enumerate(words))


def product_names(codes: list[str]) -> dict[str, str]:
    """Short names: hand-written where available, otherwise the description of the product's biggest 8-digit line."""
    from explore import HS4_NAMES
    names = {**HS4_NAMES, **PRODUCT_NAMES}
    missing = [c for c in codes if c not in names]
    if missing:
        rows = pd.read_parquet(ROOT / "data" / "processed" / "imports_clean.parquet",
                               columns=["hs4", "commodity_en", "value_bd"], filters=[("hs4", "in", missing)])
        top = rows.groupby(["hs4", "commodity_en"], observed=True)["value_bd"].sum().reset_index()
        top = top.sort_values("value_bd", ascending=False).drop_duplicates("hs4").set_index("hs4")["commodity_en"]
        names.update({c: str(top[c]).capitalize() for c in missing if c in top.index})
    return {c: names.get(c, c) for c in codes}


def country_names(codes: list[str]) -> dict[str, str]:
    """Country codes -> tidied country names from the cleaned data."""
    rows = pd.read_parquet(ROOT / "data" / "processed" / "imports_clean.parquet", columns=["iso2", "country_en"],
                           filters=[("iso2", "in", list(codes))])
    lookup = rows.drop_duplicates("iso2").set_index("iso2")["country_en"]
    return {c: tidy_name(lookup[c]) if c in lookup.index else c for c in codes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Forecast every product, build the supplier profiles, run the comparisons and write the report."""
    t0 = time.time()
    print("1/4 Product forecasts (HS4) for 2024, 2025 and 2026 ...", flush=True)
    core = load_core()
    frames = product_forecasts(core, load_calendar())
    last_month = int(frames.loc[frames["year"] == 2026, "target_month"].max().month)
    periods, since = build_periods(last_month)
    all_periods = {**periods, **since}
    since_label, since_months = list(since.items())[0]

    print("2/4 Supplier profiles ...", flush=True)
    core_w = load_core_weight()
    profile = supplier_profile(core_w)
    frames = frames.join(profile[["supplier_group", "beyond_group"]], on="hs4", how="inner")

    print("3/4 Comparisons, ranges and the regression ...", flush=True)
    lengths = {1} | {len(m) for m in all_periods.values()}
    tests = frames[frames["year"].isin(TEST_YEARS)]
    ranges = window_ranges(tests, lengths, group_totals)
    frame26 = frames[frames["year"] == 2026]
    gaps = period_gaps(frame26, ranges, all_periods, group_totals)
    products = product_results(frame26, profile, since_months)
    medians = group_medians(products)
    effect_table = effects(products)
    totals_all = frame26[frame26["target_month"].dt.month.isin(since_months)][["actual", "forecast"]].sum()

    print("4/4 Switching, and the report ...", flush=True)
    switch = switching(core_w, profile, months=tuple(since_months))

    # The largest products with fewer than two effective suppliers, shown as examples in the report.
    few = products[products["supplier_group"] == SUPPLIER_GROUPS[0]].sort_values("value_bd", ascending=False).head(6)
    names = {"products": product_names(list(few.index)), "countries": country_names(list(few["top_country"]))}

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    products.to_csv(INTERIM_DIR / "extra_suppliers_products.csv")
    gaps.to_csv(INTERIM_DIR / "extra_suppliers_groups.csv", index=False)
    effect_table.to_csv(INTERIM_DIR / "extra_suppliers_effects.csv", index=False)
    switch.to_csv(INTERIM_DIR / "extra_suppliers_switching.csv", index=False)

    results = {"profile": profile, "products": products, "gaps": gaps, "ranges": ranges, "medians": medians,
               "effects": effect_table, "switching": switch, "since_label": since_label, "periods": periods,
               "national_pct": totals_all["actual"] / totals_all["forecast"] * 100,
               "core_value_2024_25": core[core["date"].dt.year.isin(PROFILE_YEARS)]["value_bd"].sum(),
               "few": few, "names": names, "last_month": last_month}
    import suppliers_report
    suppliers_report.write(results)
    print(f"Done in {time.time() - t0:.0f}s. Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
