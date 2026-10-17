# Step 4: Forecasting imports without the war

Everything in this project rests on one estimate: what Bahrain would have imported in 2026 if the war had not happened. This step builds the forecasting models that make that estimate, and tests them on two normal years, 2024 and 2025, to find out how accurate they are and how far off they normally are. Those normal misses are what later steps use to decide whether a 2026 shortfall is real or just noise.

## Key findings

- **A ridge regression written from scratch was the most accurate forecast.** On the 2025 final check it missed national core imports by 5.5% a month on average, against 6.2% for the simple rule (last year's average month), and it was the most accurate at every level of detail in both test years.
- **Gradient-boosted trees didn't beat it.** With four or five years of noisy monthly data their extra flexibility mostly fitted noise; the best set-up was the most constrained one tried.
- **The model relies mainly on the recent level of imports plus an allowance for growth** (about 3.6% above the last 12 months' average). Calendar effects are small.
- **A single month can be 15% off nationally even in a normal year.** Ranges like this were set for every total, and when set on 2024 they held for 88-92% of months in 2025, close to the 90% aimed for.
- **January and February 2026, before the war, were within 4% of the forecast**, which supports using the model for the months after.

## Method

### What is forecast

The models forecast core goods (everything except bulk materials and precious metals, as decided in step 3). Core goods are split into series, each one 2-digit customs chapter from one origin region, for example "chapter 84, machinery, from Europe". A series needs imports in at least 90% of past months to be forecast on its own; the patchy ones are pooled into one "other" series per section and region. That gives 663 series (114 of them pooled). Adding up the series' forecasts gives a forecast for every section, every origin region and the whole country, and these always agree with each other.

### Features

Each forecast is made at the end of a December for each of the next 12 months, using only what was known at that point:

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

Every value is divided by the series' own average month over the last 12 months, so the target becomes "this month as a multiple of a normal recent month". This lets one model learn from all 663 series at once, small and large alike, which gives it far more examples than any single series could. The forecast is multiplied back afterwards. When the models learn, big series count for more than small ones, because the errors that matter in the end are in dinars.

### The four forecasts

1. **Simple rule**: every month will look like last year's average month. This is the baseline from step 3, and the bar any model has to clear.
2. **Same month last year**: the other obvious baseline.
3. **Ridge regression, written from scratch in NumPy.** It adds up the features with one coefficient each, choosing the coefficients that minimise the weighted squared error on past data plus a penalty on their size. The penalty pulls the model back towards the simple rule whenever the evidence for a feature is weak. It is solved two ways, directly with the normal equations and step by step with gradient descent, as a check on the code. *Pros:* transparent, fast and hard to overfit. *Cons:* it can only combine features in straight lines, so it can't discover interactions on its own.
4. **Gradient-boosted trees** (scikit-learn's HistGradientBoosting, the same method as LightGBM). Many small decision trees, each correcting the errors of the ones before. *Pros:* learns non-linear effects and interactions without being told about them, and is usually the strongest method for tabular data. *Cons:* harder to explain, more settings to tune, and with little data it can fit noise.

### How the test works

Each test copies exactly what is done for 2026. Standing at the end of December 2023, with only the data known by then, every model forecasts each month of 2024; the forecasts are then compared with what arrived. The same is repeated from the end of December 2024 for 2025.

All choices (each model's settings, and whether to train on the COVID year 2021) were made on 2024 only. The 2025 test was run once, at the end, with every choice already fixed, so it shows how the models do on a year nothing was tuned on, which is the situation 2026 is in. The models were scored on the average miss for the three levels the later reports use: the national total, the nine origin regions and the sections.

The miss is the weighted absolute percentage error: the total of the absolute monthly errors divided by total actual imports. Unlike an average of percentage errors, small totals can't blow it up, and it answers the question that matters here: how many dinars off was the forecast? (Step 3 divided by the forecast instead. With several models to compare, the actual is fairer, because it is the same for all of them; dividing by the forecast would let a model look better just by forecasting high.)

## Results

### 1. Which forecast is most accurate?

**2025: the final check**

| Forecast | National | By origin region | By section | Average of the three | Total for the year |
|---|---|---|---|---|---|
| Simple rule (last year's average month) | 6.2% | 12.1% | 13.6% | 10.7% | 4.4% too low |
| Same month last year | 7.9% | 14.3% | 16.7% | 13.0% | 4.4% too low |
| **Linear model** | 5.5% | 11.1% | 12.9% | 9.9% | 2.0% too low |
| Gradient-boosted trees | 6.3% | 12.1% | 14.0% | 10.8% | 0.4% too low |

**2024: used for choosing**

| Forecast | National | By origin region | By section | Average of the three | Total for the year |
|---|---|---|---|---|---|
| Simple rule (last year's average month) | 6.2% | 10.8% | 13.2% | 10.1% | 3.3% too low |
| Same month last year | 7.7% | 15.0% | 15.9% | 12.9% | 3.3% too low |
| **Linear model** | 5.4% | 10.4% | 13.0% | 9.6% | 1.0% too low |
| Gradient-boosted trees | 5.7% | 10.7% | 13.1% | 9.8% | 0.6% too high |

![Model comparison](figures/step4_2_model_comparison.png)

The linear model wins in both years and at every level. Nationally it missed by 5.5% in 2025, against 6.2% for the simple rule, about 11% less. Much of the gain comes from getting the level right: imports were growing, so the simple rule was 4.4% too low over 2025, while the linear model was 2.0% too low. The same-month rule was the worst in both years, as step 3 found.

The gain is modest because most of a month's movement is random and no model can predict it. But it is consistent, and it held up on the year nothing was tuned on.

![National backtest](figures/step4_1_national_backtest.png)

### 2. Why the trees didn't win

Gradient-boosted trees are usually the strongest model for this kind of table, so their result needs explaining. There were two problems.

First, trained the usual robust way, with absolute error, they forecast too low. Absolute error makes a model aim at the median month of each series, and in a lopsided series a few big months sit among many smaller ones, so the median is below the mean. Added up over hundreds of series, those forecasts came out 4.6% too low in 2024. Switching to squared error, which aims at the mean, fixed the bias.

Second, there isn't enough history for their flexibility to pay off. With four or five years of noisy monthly data, most of the patterns trees can find are noise. The best set-up was the most constrained one tried (4 leaves per tree, at least 1,000 rows per leaf), and in 2025 it still missed more than the linear model (10.8% against 9.9% on the average of the three levels) and slightly more than the simple rule (10.7%).

Every tree set-up tried on 2024:

| Tree set-up | National | By origin region | By section | Average of the three | Total for the year |
|---|---|---|---|---|---|
| Absolute error (aims at the median), 31 leaves | 6.2% | 11.0% | 14.4% | 10.5% | 4.6% too low |
| Poisson (aims at the mean), 15 leaves | 6.9% | 11.2% | 14.5% | 10.9% | 1.1% too high |
| Squared error (aims at the mean), 15 leaves | 7.0% | 11.4% | 14.8% | 11.1% | 1.3% too high |
| **Squared error (aims at the mean), 4 leaves, square-root weights (chosen)** | 5.7% | 10.7% | 13.1% | 9.8% | 0.6% too high |

*Tree results can differ slightly (in the first decimal place) between scikit-learn versions; the conclusions don't change.*

### 3. The ridge penalty

A stronger penalty keeps the coefficients smaller, so the model stays closer to the simple rule. Four strengths were tried on 2024:

| Penalty | National | By origin region | By section | Average of the three | Total for the year |
|---|---|---|---|---|---|
| 0.01 | 6.5% | 11.2% | 13.6% | 10.4% | 0.3% too high |
| 0.1 | 6.2% | 11.0% | 13.4% | 10.2% | 0.2% too high |
| **1 (chosen)** | 5.4% | 10.4% | 13.0% | 9.6% | 1.0% too low |
| 10 | 5.6% | 10.5% | 13.1% | 9.7% | 0.7% too low |

A fairly strong penalty (1) worked best. Apart from the recent level and growth, the evidence for most features is weak, and the penalty stops the model from chasing noise. Several features also overlap (the averages of the last 3, 6 and 12 months move together), and without a penalty their coefficients can grow large and cancel each other out.

### 4. Training on 2021

The chosen linear model was run on 2024 twice, learning from 2021-2023 and from 2022-2023 only:

| History used | National | By origin region | By section | Average of the three | Total for the year |
|---|---|---|---|---|---|
| 2021-2023 (with 2021) | 5.4% | 10.4% | 13.0% | 9.6% | 1.0% too low |
| 2022-2023 (without 2021) | 5.5% | 10.8% | 14.3% | 10.2% | 2.0% too low |

It is more accurate with 2021. Without it there is only one year of history in the 2024 test, which isn't enough to learn about trends. So 2021 stays in, and the COVID features tell the model which of its months were unusual.

### 5. What the linear model relies on

![Coefficients](figures/step4_3_coefficients.png)

*Each bar shows how much the forecast moves, in % of the series' average month, when that feature rises by one standard deviation. Final model, trained on 2021-2025. Section and region adjustments are not shown.*

The intercept is a built-in allowance for growth: with every feature at its typical value, the model forecasts 3.6% above the series' average month over the last 12 months. That is roughly how much imports grew from one year to the next, and it fixes most of the simple rule's tendency to forecast too low. Beyond that, recent months pull the forecast their way: if the last month or the last 3 or 6 months were higher than usual, the forecast goes up a little. Part of last year's change tends to unwind. A series that was higher in the year before than it is now gets a slightly higher forecast (+2.1 on the chart), and one that grew fast gets a slightly lower one (-1.5). The same month two years earlier adds a little seasonal memory (+2.0). A month with Eid al-Adha is about 1.3% lower, and December about 0.7% higher.

All these adjustments are small, a few percent at most. That is the penalty at work: in normal years most of a month's movement is noise, so the model only moves away from "recent level plus growth" when the evidence is clear. Because some features overlap, the penalty shares the credit between them, so any single coefficient should be read with care.

### 6. Checking the maths: normal equations and gradient descent

The ridge coefficients were found two ways, both coded from scratch. The normal equations solve (X'WX + P) b = X'Wy directly. Gradient descent starts with every coefficient at zero and repeatedly steps downhill, with a step size of 1/L, where L is the largest eigenvalue of the Hessian 2(X'WX + P); that is the largest step that is guaranteed not to overshoot for a quadratic loss. If the code is right, both must reach the same answer.

![Gradient descent](figures/step4_4_gradient_descent.png)

With the chosen penalty, gradient descent gets within 0.1% of the minimum after 19 steps and matches it to floating-point precision after 71 steps. With a very weak penalty (0.01), the overlapping features make the problem badly conditioned: the loss surface is almost flat in some directions, so progress slows, and after 300 steps it is close but not exact. A larger penalty improves the conditioning of X'WX, which is why descent converges faster.

### 7. How far off the model normally is

For each reported total (national, each origin region, each section) I took the model's monthly errors in the 2024 and 2025 tests and found the distance that 9 months out of 10 stayed within. That is its normal range. A 2026 month only counts as a real shortfall if it falls outside it.

To check that the ranges work on a year they weren't built from, they were first set from 2024 alone and tested on 2025:

| Level | Months checked | Months inside the range |
|---|---|---|
| National total | 12 | 92% |
| Origin region | 108 | 88% |
| Section | 252 | 90% |

Between 88% and 92% of months fell inside, close to the 90% aimed for. The final ranges are set from both years together:

| Total | Normal range for a single month |
|---|---|
| **Bahrain (national)** | **+/-15%** |
| GCC | +/-11% |
| East and South-East Asia | +/-12% |
| Other Middle East | +/-14% |
| South Asia | +/-23% |
| Americas | +/-23% |
| Europe | +/-29% |
| Africa | +/-45% |
| Oceania | +/-71% |
| Other and not specified | +/-109% |

*Sections range from +/-14% (Plastics and rubber) to +/-153% (Arms and ammunition). All are saved in `data/interim/step4_ranges.csv`.*

A single national month can be 15% off in a normal year, much more than the average miss of 5.5%, because the range has to cover unusual months such as December as well. Small or erratic origins have wide ranges, so a change there has to be large before it counts.

These ranges are empirical: they make no assumption about the shape of the errors, they simply read off how big the misses were. The trade-off is that with 24 months per total, the 90th percentile is itself uncertain.

### 8. January and February 2026

The model was retrained on 2021-2025 and used to forecast 2026 from the end of December 2025, exactly as step 5 does. The war began on 28 February, so January and February should look normal:

| Month | Forecast (BD m) | Actual (BD m) | Difference | Normal range | Inside? |
|---|---|---|---|---|---|
| January 2026 | 397.6 | 406.4 | +2.2% | +/-15% | yes |
| February 2026 | 386.5 | 399.9 | +3.5% | +/-15% | yes |

Both months land close to the forecast and well inside the range. Of the 60 origin-region and section months, 50 were inside their range too. Of the 10 outside it, 8 were above the forecast, mostly in February, which could be stockpiling as tensions rose.

Because trees benefit from more history, every candidate was also run on these two months with all five years of training data:

| Forecast | National | By origin region | By section | Average of the three | Total for the two months |
|---|---|---|---|---|---|
| Simple rule | 6.4% | 10.4% | 11.6% | 9.5% | 6.4% too low |
| Same month last year | 10.9% | 13.8% | 18.1% | 14.3% | 10.9% too low |
| **Linear model (chosen)** | 2.8% | 8.0% | 11.2% | 7.3% | 2.8% too low |
| Trees: Absolute error (aims at the median), 31 leaves | 7.2% | 11.3% | 14.8% | 11.1% | 7.2% too low |
| Trees: Poisson (aims at the mean), 15 leaves | 2.0% | 9.6% | 11.9% | 7.8% | 0.4% too high |
| Trees: Squared error (aims at the mean), 15 leaves | 2.1% | 9.7% | 11.5% | 7.8% | 0.6% too high |
| Trees: Squared error (aims at the mean), 4 leaves, square-root weights (chosen) | 2.3% | 7.8% | 11.2% | 7.1% | 2.3% too low |

With five years of history the best trees did about as well as the linear model, but across the three tests more history doesn't steadily favour them:

| Test | Years of history | Linear model | Best trees |
|---|---|---|---|
| 2024 | 3 | 9.6% | 9.8% |
| 2025 | 4 | 9.9% | 10.8% |
| January-February 2026 | 5 | 7.3% | 7.1% |

*Average miss over the three reported levels.*

This check was not used to change the choice. Two months is too few to separate models that differ by a few tenths of a percent; February may already have been unusual; and picking the best of several set-ups on a small test flatters whichever wins by luck. Instead, step 5 uses the linear model for the main results and the best trees as a second opinion. Where the two agree, the findings are stronger; where they differ, that is reported.

## Limitations

- **Two test years.** The normal ranges rest on 24 months per total. More normal years would make them more reliable.
- **Tuning on a single year** can favour settings that happened to suit 2024. The untouched 2025 check is the guard against this, and it confirmed the choice.
- **Both models learn from the same five years**, so their agreement in step 5 is reassuring but not independent proof.
- **Pooling patchy series** into "other" means their individual products can't be forecast separately.
- **The 2026 forecast is made once, from December 2025.** It can't know about anything that would have changed in 2026 without the war, such as price moves or new projects.

## What step 5 uses

- The linear model's 2026 forecasts (`data/interim/step4_forecast_2026.parquet`) and the best trees' forecasts as a second opinion.
- The normal ranges (`data/interim/step4_ranges.csv`) to decide which gaps are real.
- The 2024-2025 test forecasts (`data/interim/step4_backtest.parquet`) to build ranges for periods longer than one month.
- Bulk materials and precious metals have too few series for these models; step 5 uses the simple rule for them, with their normal ranges from step 3.

---

*This report is generated by `src/backtest.py` (charts and text in `src/step4_report.py`). Re-running the pipeline (`python src/run_all.py`) refreshes every number and chart. Data: Bahrain Open Data Portal (iGA), www.data.gov.bh.*
