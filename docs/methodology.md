# Methodology

This note explains how the project works and why each choice was made, including the alternatives I considered and
what each choice costs. The step reports in [`reports/`](../reports) show the results; the code in [`src/`](../src) is
commented with the same reasoning at the point where each choice is implemented.

## 1. The question and the overall design

The question is how far Bahrain's imports fell because of the 2026 war, which supply chains and routes were hit, and
how far they had come back. "Because of the war" is the hard part. Imports grow over time, move with the seasons and
jump around from month to month, so a fall against last year is not the same as a fall caused by the war.

The project therefore uses a **counterfactual forecast**. A model learns from 2021-2025 how imports normally behave,
and then, standing at the end of December 2025, forecasts every month of 2026 as if nothing had happened. The gap
between that forecast and what actually arrived is the estimate of the war's effect. The same model is first tested
on two normal years, so its usual error is known, and a gap only counts as real if it is larger than that.

**Alternatives considered**

| Approach | Why not the main method |
|---|---|
| Compare with the same months of 2025 | Simple, but ignores growth (imports rose 5% in 2025) and lets one unusual month in 2025 distort the comparison. Used only for port and airport figures, which have no model behind them. |
| Compare with the 2025 average month | Steadier than a single month, but still ignores growth and the calendar. Used as the yardstick for exploring in step 3. |
| Difference-in-differences or synthetic control | Needs comparable places that were not affected. Every Gulf economy was hit by the same closure, and the data covers only Bahrain. |
| A classical model per series (ARIMA, exponential smoothing) | About 660 series with five years of monthly history each is short for per-series models, and they cannot share what they learn. One model across all series (a "global" model) learns from far more examples. |

**Cost of the chosen approach.** The estimate is only as good as the forecast, and the forecast cannot know about
anything else that would have changed in 2026 without the war (prices, large projects, policy). Testing on normal
years limits the first problem; nothing in the data can fix the second.

## 2. Data and cleaning

**Sources.** Bahrain Open Data Portal: monthly non-oil imports by 8-digit product and origin country (one file per
year, 2021-2026), plus monthly figures for Khalifa Bin Salman Port and Bahrain International Airport. Every download
is recorded with its date and checksum in `data/raw/MANIFEST.json`, and every month's totals are checked against
figures read independently from the portal's API (`src/check_downloads.py`).

**Keep everything, flag what looks wrong.** Rows with zero weight, zero or negative value, or a value per kg above
gold are flagged but kept, so the totals still equal the official statistics. *Pro:* nothing silently disappears and
any analysis can still exclude flagged rows. *Con:* a wrong row stays in the data; here they carry about 0.2% of value.

**Repeated rows are added together.** Six product-country-months appear twice with different values. They look like
separate shipments, so they are summed rather than de-duplicated. Totals are checked month by month before and after.

**Three groups that behave differently.**

- *Core goods*: the everyday supply chains (machinery, vehicles, food, clothing, chemicals...), about three-quarters
  of value and the subject of the forecasts.
- *Bulk materials*: iron ore, alumina, coke and building minerals, about 85% of weight but a fifth of value, delivered
  in shiploads to a few plants.
- *Precious metals*: gold, silver and jewellery, which follow the gold price and whose normal forecast miss is about
  seven times that of core goods.

*Pro:* one ore carrier or gold consignment can't decide whether supply chains look better or worse. *Con:* bulk
materials and precious metals get a simpler treatment (compared with the 2025 average month).

**Repairs to the port's general cargo figures.** Several 2026 months were typed with a decimal point for a thousands
separator (56.186 for 56,186) and are multiplied by 1,000; a month recorded as zero while 17 general cargo ships
called is treated as missing. Each change is marked in the data. Because these are judgements about someone else's
figures, no conclusion rests on general cargo alone.

## 3. What is forecast

**Series.** Core goods are split into one series per 2-digit customs chapter and origin region (for example, chapter
84, machinery, from Europe). Series with imports in fewer than 90% of months are pooled into one "other" series per
section and region, giving about 660 series.

**Why this level.** Step 3 measured how much a simple forecast misses at each level of detail. The finer the split,
the larger the normal miss, because in a large total one product's high month cancels another's low month. At the
national level the war's effect is about five times the normal miss; for each 4-digit product from each country the
normal miss is almost as large as the war's effect. Chapter by region keeps the war clearly visible while still
allowing results by product section and by origin.

**Totals that add up.** Forecasts are made for the series and simply added up to every reported total (national, by
section, by region), so the totals are always consistent with each other. *Alternative:* forecast each level
separately and reconcile them statistically (for example MinT reconciliation). That can be slightly more accurate but
is more complex; with a model that already beats the baselines at every level, the simple bottom-up sum was enough.

**Cost.** The models cannot see shifts between products within a chapter, and pooled series cannot be broken down.

## 4. Features and scaling

Each forecast is made at the end of a month (the "origin") for each of the next 12 months, using only what was known
then: recent levels (last month, averages of the last 3, 6 and 12 months), trend (growth over the year, the level a
year earlier), the same calendar month one and two years before, how variable the series is and how often it has
empty months, its size, section and region, the calendar of the target month (month, December, days of Ramadan, Eid,
days in the month), COVID restrictions, and how far ahead the forecast is.

**Every value is divided by the series' average month over the last 12 months.** The target becomes "this month as a
multiple of a normal recent month", so a BD 20,000 series and a BD 20 million series look alike and one model can learn
from all of them. *Pro:* far more training examples than any one series has, and a target that means the same thing
everywhere. *Con:* the scale is itself a noisy 12-month average, inflated for a while after one exceptional month.
Training targets are capped at 10 times the scale so that a few one-off deliveries cannot dominate a squared-error fit.

Training rows are weighted by the series' scale, because the errors that matter are in dinars.

## 5. Models

Four forecasts are compared. All predict the scaled target and share the same interface.

| Forecast | What it does | Why it is here |
|---|---|---|
| Simple rule | Every month equals last year's average month | The baseline from step 3; any model has to beat it. |
| Same month last year | Every month equals the same month a year earlier | The other obvious baseline. It loses because imports are only weakly seasonal. |
| Ridge regression | A weighted linear model with a penalty on coefficient size, written from scratch in NumPy | Transparent, fast and hard to overfit with little data. |
| Gradient-boosted trees | scikit-learn's HistGradientBoosting | Learns non-linear effects and interactions on its own; usually strong on tabular data. |

**Ridge regression, from scratch.** Features are standardised (weighted mean 0, standard deviation 1), missing values
are filled with a neutral value plus a "was missing" flag, and section and region become 0/1 columns. The coefficients
minimise the weighted squared error plus λ times the sum of squared coefficients, with the intercept left unpenalised.
They are found two ways: directly, by solving the normal equations (X'WX + P)b = X'Wy, and by gradient descent with a
step size of 1/L, where L is the largest eigenvalue of the Hessian. The two agree to floating-point precision, which
checks the code. The penalty matters because several features overlap (the 3-, 6- and 12-month averages move together);
without it, their coefficients can become large and cancel out. λ = 1 was chosen from {0.01, 0.1, 1, 10} on 2024.

*Pros:* every coefficient can be read, and the fitted model is easy to explain: forecast the recent level, add an
allowance for growth (about 3.6%), and make small adjustments. *Cons:* it can only add effects in straight lines, so
interactions have to be built in as features.

**Gradient-boosted trees.** Four set-ups were tried on 2024, from large flexible trees to small constrained ones. Two
lessons came out. First, absolute error (often chosen for robustness) makes each series' forecast aim at its median
month, and in lopsided series the median is below the mean, so the forecasts added up to 4.6% too little. Squared error
aims at the mean and fixed this. Second, with five years of noisy monthly data, the most constrained set-up (4 leaves,
at least 1,000 rows per leaf, square-root weights) did best; extra flexibility mostly fitted noise. The trees did not
beat the ridge regression on the untouched 2025 check, so they are used as a second opinion.

## 6. Validation

**Rolling-origin backtests.** Standing at the end of December 2023, using only data up to then, each model forecasts
every month of 2024; the same is repeated a year later for 2025. Training rows only use origins whose target months
fall before the test year, and features only use months up to the origin. Both rules are covered by tests.

**Tune on one year, check on another.** Every choice (ridge strength, tree set-up, whether to train on the COVID year
2021) was made on 2024. The 2025 test was run once, with the choices fixed, to see how they do on a year nothing was
tuned on, which is the situation 2026 is in.

**Scoring.** The miss is the weighted absolute percentage error: the total of absolute monthly errors divided by total
actual imports, averaged over the national total, the origin regions and the sections (the levels the reports use).
*Why not MAPE:* an average of percentage errors is dominated by tiny totals and treats a 10% miss on paper the same as
a 10% miss on machinery. *Why divide by actual:* it is the same for every model, so no model can look better by
forecasting high.

**Results.** On 2025 the ridge regression missed national core imports by 5.5% a month against 6.2% for the simple
rule, and it was best at every level in both years. January and February 2026, before the war, were within 4% of its
forecast.

**Cost.** Two test years is not many. Tuning on one year can favour settings that happened to suit it; the untouched
2025 check is the guard, and it confirmed the choice.

## 7. Normal ranges

For every reported total, the model's errors in the 2024 and 2025 tests (actual / forecast - 1) give a "9 times out of
10" range: the 90th percentile of the absolute error. Periods longer than a month get their own, narrower ranges,
computed the same way from every run of that many consecutive months within each test year, because errors partly
cancel over several months.

**Does it work?** Ranges set from 2024 alone held for 88-92% of months in 2025, close to the 90% aimed for.

*Pros:* no assumption about the shape of the error distribution, and each total gets a range that reflects its own
volatility. This is close in spirit to split conformal prediction, without its small-sample correction. *Cons:* with
24 months per total (and fewer independent runs for long periods), the 90th percentile is itself uncertain, and the
ranges assume 2026's ordinary noise would have looked like 2024-2025's.

## 8. Measuring the shortfall

For each period and total, the **gap** is actual minus forecast, in BD and as % of the forecast; it counts as a real
change if it is outside the normal range for a period of that length. The **BD lost** is given with a likely range,
because the true no-war value could have been anywhere in forecast × (1 ± range).

Three checks support the headline numbers:

- **Second opinion.** The whole comparison is repeated with the best trees and their own ranges. The national gap is
  -33% against -32%, and every section and region is within about 4 points. (Both models learn from the same data, so
  this is reassuring rather than independent.)
- **Aircraft.** Aircraft, engines and parts are large, irregular purchases. Holding them at their normal level shows
  they made March look deeper than everyday supply chains were (-45% instead of -54%) but don't explain the later months.
- **Products.** The same model rebuilt at 4-digit level for six everyday products gives illustrations (cars far below
  normal, rice and poultry bouncing back), with wide ranges because single products are noisy.

## 9. Routes

The customs data says where goods came from but not how they travelled, so the route analysis combines several pieces
of partial evidence.

**Port and airport figures.** Containers, car-carrier vehicles, general cargo and air cargo in 2026 against the same
months of 2025.

**Groups that depend on different routes.** Goods are grouped by origin (GCC countries, which can use the road; the rest
of the Middle East; beyond the Middle East, normally by sea through Hormuz or by air) and, for goods from beyond the
Middle East, by value per kg: light (at least BD 5/kg, worth flying), heavy (under BD 1/kg, almost never flown), middle,
and vehicles separately because they travel on car carriers. The thresholds follow from the cost of air freight, a few
dollars per kg. Step 5's gaps and ranges are then computed for these groups. If air helped, light goods should have
held up better than heavy ones in April-May, when the airspace was open but Hormuz was closed, and they did (-13%
against -54%, leaving out aircraft engines). *Cost:* origin is a proxy for route, and a series' class is set by its
average value per kg, so some heavy items sit in light series.

**An elimination bound for the road.** The King Fahd Causeway publishes no freight figures. For each period, by weight
of core goods: the most containers could have carried is last year's weight scaled by the change in containers
unloaded, as if everything had come in containers last year; add all air cargo, as if every tonne were core goods;
whatever is left must have come by road or general cargo ship. Every assumption favours containers and planes, so the
result (at least 54% over March-July) is a minimum. The only assumption that could push the other way is that a
container held no more than usual. *Cost:* it bounds the road and general cargo together and cannot separate them;
splitting them needs assumptions about container loads and how much general cargo was imports.

**How far supply chains came back.** For each total, % of normal in the latest period compared with its worst:
(latest - worst) / (100 - worst). It measures recovery relative to the size of the fall, so a shallow dip and a deep
fall that both fully recovered score the same.

## 10. Supplier concentration

**Products.** Each 4-digit heading with regular imports is a product (844 products, 98% of core goods), forecast with
the same model and tested on 2024-2025.

**Reliance on few countries.** The effective number of supplier countries, 1 / Σ(share²), the inverse of the
Herfindahl-Hirschman index, measured on 2024-2025. Unlike a simple count, it isn't inflated by many tiny suppliers.

**Separating the explanations.** Products with few suppliers might also tend to come from far away. A weighted least
squares regression of each product's % of normal (capped at 200%) on log2(effective suppliers), the share from beyond
the Middle East, log2(value per kg) and a vehicles flag separates the two. Products are weighted by size; the same
regression with square-root and equal weights checks that the result doesn't depend on a few large products. Ranges
come from 2,000 bootstrap re-draws of the products (5th to 95th percentiles), which needs no assumption that the errors
are normally distributed.

**Result.** No measurable effect of the number of suppliers (+1.7 points per doubling, range -3.7 to +7.3) and a clear
effect of distance (about 5 points per 10% from beyond the Middle East) under every weighting. *Cost:* association,
not cause; country of origin is not the shipping route; and a heading can mix items with different suppliers.

## 11. Reproducibility and testing

- One command (`python src/run_all.py`) downloads the data and regenerates every table, chart and report, so the text
  can never drift from the numbers. Sentences that name the hardest-hit sections or regions are built from the data.
- `data/raw/MANIFEST.json` records each file's download date, row count and checksum. `requirements-lock.txt` pins the
  package versions used; the trees' results can differ slightly with other scikit-learn versions.
- 49 tests (pytest) cover the loaders, the cleaning, the honesty rules of the backtest (no future data in features or
  training rows), the solvers, the range and gap calculations, the elimination bound and the supplier measures. They
  use small hand-made tables, so they run without the data, and they run on every push through GitHub Actions.

## 12. Main limitations

1. The causeway publishes no figures, so the road's share is a minimum, not a measurement.
2. The normal ranges rest on two test years.
3. Both models learn from the same five years, so their agreement is not independent proof.
4. Customs months and arrival months don't always line up; periods of two months or more soften this.
5. The port's general cargo figures needed repairs and may include goods loaded for export.
6. The data runs to July 2026. August onwards, the first months with Hormuz closed and no reopening, will extend the
   results when the portal publishes them.
