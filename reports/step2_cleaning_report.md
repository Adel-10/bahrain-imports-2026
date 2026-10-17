# Step 2: Cleaning the data

The six yearly import files (1,758,299 rows, January 2021 to July 2026) were combined into one table with 1,758,293 rows: one row per month, 8-digit product and origin country. The aim was to change as little as possible. Nothing was dropped, the monthly totals still match the official figures exactly, and every change is recorded.

## Key points

- **Totals are unchanged.** Monthly value and weight were checked before and after cleaning; the script stops if they differ.
- **6 product-country-months appeared more than once** with different values. They are separate shipments, so they were added together rather than treated as duplicates.
- **584 odd rows are flagged, not removed.** They carry about 0.2% of value.
- **Bulk raw materials are about 84% of import weight but only 22% of value**, so they are analysed separately from everyday goods.
- **The models work at chapter (2-digit) level or coarser.** Customs code revisions barely affect it, and its series are complete enough to forecast.
- **7 months of the port's general cargo figures were repaired** (a thousands separator typed as a decimal point, and one zero that should be missing).

## Rows and totals by year

| Year | Raw rows | Clean rows | Value (BD m) | Weight (kt) |
|---|---|---|---|---|
| 2021 | 314,055 | 314,054 | 5,316 | 26,420 |
| 2022 | 320,747 | 320,747 | 5,842 | 26,693 |
| 2023 | 321,376 | 321,376 | 5,778 | 26,994 |
| 2024 | 322,062 | 322,062 | 5,872 | 25,948 |
| 2025 | 324,818 | 324,816 | 6,185 | 27,553 |
| 2026 | 155,241 | 155,238 | 2,682 | 8,608 |

The 2026 figures cover January to July only.

## Repeated rows

6 product-country-months had more than one row (12 rows in total). The rows have different values and weights, so they look like separate shipments rather than copies of the same record. Adding them together gives one row per product, country and month, which is what every later step expects. The number of rows combined is kept in the column `n_shipments`.

## Rows that look wrong

Rows with zero weight, zero or negative value, or an impossible price per kg are flagged with a column each and kept. Removing them would make the totals disagree with the published statistics, and each analysis can still leave them out if it needs to. The downside is that a wrong row stays in the data, but together they carry so little value that no result in the project changes.

| Flag | Rows | % of value |
|---|---|---|
| Weight is zero | 287 | 0.000 |
| Value is zero or negative | 21 | 0.000 |
| Value per kg above BD 100,000 | 276 | 0.224 |

A value above BD 100,000 per kg is more than the price of gold, so it is almost certainly a typing error in the weight. 13 rows have no unit recorded (`#N/A` in the source).

## Bulk materials

A few raw materials dominate Bahrain's import weight. They arrive in whole shiploads for a handful of plants (the aluminium smelter, steel and pelletising plants, construction), so one ship more or less moves the monthly figure a lot. Analysing them with everyday goods would let a single ore carrier hide what happened to food, machinery or clothing. They are kept as their own group in every later step.

| Product group | % of value | % of weight |
|---|---|---|
| Iron ore | 9.6 | 48.4 |
| Construction minerals | 0.7 | 21.4 |
| General goods | 78.5 | 15.5 |
| Alumina | 9.1 | 11.5 |
| Petroleum and pitch coke | 2.2 | 3.2 |

*Shares of 2024-2025 value and weight. Gold and jewellery are also separated later, in step 3, because they follow the gold price.*

## Choosing the level of detail

Customs codes are revised every few years. The HS 2022 revision shows up as 722 new 8-digit codes in 2022 carrying 2.7% of that year's value, with many old codes only disappearing in 2023. A model working with 8-digit codes would see series stop and start for purely administrative reasons. At the 4-digit level, no year's changes affect more than 0.12% of value.

The other consideration is how complete each series is. A series (a product, optionally from one country) that only has trade in a few months cannot be forecast sensibly. Splitting by chapter and country gives 8,706 series, of which 2,045 have trade in nearly every month and cover 93% of value. Splitting more finely quickly leaves most value in patchy series.

The models in step 4 therefore work with chapters (2-digit codes) by origin region, and 8-digit codes are kept for looking up individual products. The cost is that the forecasts cannot see a shift between two products in the same chapter; the benefit is stable, complete series.

| Level | Series | Series with 54+ of 60 months | % of value they cover |
|---|---|---|---|
| Chapter (2-digit) | 97 | 96 | 100.0 |
| Heading (4-digit) | 1,210 | 865 | 99.3 |
| Chapter x country | 8,706 | 2,045 | 93.4 |
| Heading x country | 43,806 | 5,319 | 82.8 |
| 8-digit product x country | 138,460 | 7,966 | 66.4 |

*2021-2025. A series is counted as complete with trade in at least 54 of the 60 months.*

### Code changes between years, at each level

**8-digit**

| Years | Codes | Gone | Gone (% of value) | New | New (% of value) |
|---|---|---|---|---|---|
| 2021->2022 | 6,178 | 343 | 0.07 | 722 | 2.68 |
| 2022->2023 | 6,557 | 518 | 1.76 | 421 | 0.24 |
| 2023->2024 | 6,460 | 377 | 0.39 | 347 | 0.13 |
| 2024->2025 | 6,430 | 420 | 0.16 | 355 | 0.28 |
| 2025->2026 | 6,365 | 748 | 0.98 | 180 | 0.18 |

**6-digit**

| Years | Codes | Gone | Gone (% of value) | New | New (% of value) |
|---|---|---|---|---|---|
| 2021->2022 | 4,503 | 210 | 0.04 | 468 | 2.13 |
| 2022->2023 | 4,761 | 336 | 1.48 | 282 | 0.20 |
| 2023->2024 | 4,707 | 262 | 0.37 | 223 | 0.09 |
| 2024->2025 | 4,668 | 266 | 0.06 | 227 | 0.25 |
| 2025->2026 | 4,629 | 494 | 0.90 | 117 | 0.10 |

**4-digit**

| Years | Codes | Gone | Gone (% of value) | New | New (% of value) |
|---|---|---|---|---|---|
| 2021->2022 | 1,152 | 26 | 0.00 | 37 | 0.12 |
| 2022->2023 | 1,163 | 18 | 0.04 | 23 | 0.02 |
| 2023->2024 | 1,168 | 24 | 0.00 | 19 | 0.00 |
| 2024->2025 | 1,163 | 29 | 0.00 | 20 | 0.00 |
| 2025->2026 | 1,154 | 48 | 0.00 | 12 | 0.03 |

2025->2026 compares a full year with seven war-affected months, so more codes are "gone".

## Port general cargo: months repaired

In the port's general cargo file (freight tonnes per month), several 2026 months were typed with a decimal point where a thousands separator belongs: 56.186 instead of 56,186. Every correctly typed month since 2019 is above 3,000, so values between 0 and 1,000 were multiplied by 1,000. A month recorded as 0 while general cargo ships called at the port is treated as missing, since ships did unload. Each change is marked in the column `port_general_cargo_note`. Because these are repairs to someone else's figures, no conclusion in later steps rests on general cargo alone.

| Month | Freight tonnes used | What was done |
|---|---|---|
| Feb 2026 | 56,186 | Decimal point fixed: 56.186 read as thousands |
| Mar 2026 | missing | Recorded as 0 although general cargo ships called; treated as missing |
| Apr 2026 | 59,386 | Decimal point fixed: 59.386 read as thousands |
| May 2026 | 105,553 | Decimal point fixed: 105.553 read as thousands |
| Jul 2026 | 170,881 | Decimal point fixed: 170.881 read as thousands |
| Aug 2026 | 134,420 | Decimal point fixed: 134.42 read as thousands |
| Sep 2026 | 165,861 | Decimal point fixed: 165.861 read as thousands |

## Tables produced

- `data/processed/imports_clean.parquet`: the cleaned imports, with product hierarchy, product group and flags.
- `data/processed/route_monthly.parquet`: port containers (Jan 2019 to Sep 2026), ship visits (Jan 2019 to Sep 2026), air cargo (Jan 2023 to Aug 2026), vehicles unloaded from car carriers (Jan 2019 to Sep 2026) and general cargo (Jan 2019 to Sep 2026).
- `data/reference/calendar.csv`: 108 months (Jan 2019 to Dec 2027) with Ramadan days, Eid flags and COVID flags.
- `data/reference/hs_sections.csv`: chapter to section lookup.

---

*This report is generated by `src/build_dataset.py`. Re-running the pipeline (`python src/run_all.py`) refreshes every number and chart. Data: Bahrain Open Data Portal (iGA), www.data.gov.bh.*
