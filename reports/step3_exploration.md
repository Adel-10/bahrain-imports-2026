# Step 3: Exploring the data

Before building a forecasting model I wanted to understand the data well enough to answer three practical questions. Should all imports be modelled together? What does a normal year look like, and so what does a model need to know? And how finely can imports be split before ordinary noise is as large as the war itself? Along the way this step takes a first look at what happened in 2026.

It uses imports from January 2021 to July 2026, the latest month published. Port figures run to September 2026 and airport figures to August 2026, because they are published sooner.

## Key findings

- **Three kinds of imports behave very differently.** Everyday "core" goods are 73% of import value and fairly steady; bulk raw materials are 85% of weight but arrive in lumpy shiploads; gold and jewellery swing with the gold price. They are analysed separately.
- **Imports are only weakly seasonal but they grow**, so a model needs the recent level of imports more than the calendar.
- **Core goods fell 53% below normal in March 2026** and were still 15% below in June-July.
- **Goods from GCC countries rose** while those from East Asia and the Americas stayed far below normal, a first sign that route mattered more than product.
- **Containers and air cargo fell much further than the goods arriving.** In June-July containers unloaded were at 31% of normal while core goods were back to 86% by weight, so a large share must have come another way, most likely by road.
- **The models will forecast each 2-digit customs chapter from each origin region.** Finer splits have so much normal noise that the war barely stands out.

## How changes are measured in this step

Every change in this report compares a 2026 figure with the average month of 2025 for the same thing. For example, core goods averaged BD 377.4m a month in 2025 and were BD 178.2m in March 2026, so March is reported as -53%.

Comparing with the same month a year earlier sounds more natural, but a single month carries its own noise, and that noise leaks into the comparison:

| Month | 2026 | Same month in 2025 | Against the same month | Against the 2025 average month |
|---|---|---|---|---|
| June | 286 | 337 | -15% | -24% |
| July | 359 | 403 | -11% | -5% |

*Core goods, BD million.*

June 2025 happened to be a weak month and July 2025 a strong one, so the same-month comparison flatters June 2026 and penalises July 2026. Tested on two normal years, the same-month rule also missed by more at every level of detail (section 6). The average month is only a yardstick for exploring, though. From step 5 onward, each month of 2026 is compared with a model's forecast for that specific month, which can allow for growth, December peaks and Ramadan.

A change only means something if it is bigger than the normal ups and downs, so two measures of normal movement are used:

- **Normal miss.** Forecast every month of a year as the previous year's average month, add up the misses (ignoring whether each month was above or below, so they can't cancel out) and divide by the total forecast. For core goods this was 6.4% in 2024 and 6.5% in 2025, so about 6%. The 2025 calculation is shown below.
- **Normal range.** A single month can stray further than the average miss. For each month of 2023-2025 I measured how far it was from the previous year's average month; the normal range is the distance that 9 months out of 10 stayed within. It is the grey band on the charts, and a dot outside the band is more than ordinary noise.

| Month | Actual | Forecast | Miss |
|---|---|---|---|
| Jan | 353.5 | 360.7 | 7.3 |
| Feb | 365.0 | 360.7 | 4.2 |
| Mar | 359.6 | 360.7 | 1.2 |
| Apr | 377.9 | 360.7 | 17.2 |
| May | 394.1 | 360.7 | 33.4 |
| Jun | 336.7 | 360.7 | 24.1 |
| Jul | 403.0 | 360.7 | 42.2 |
| Aug | 351.8 | 360.7 | 9.0 |
| Sep | 376.9 | 360.7 | 16.1 |
| Oct | 362.1 | 360.7 | 1.4 |
| Nov | 412.9 | 360.7 | 52.1 |
| Dec | 435.1 | 360.7 | 74.4 |
| **Total** |  | 4,329.0 | 282.5 |

*Core goods in 2025, forecast = the 2024 average month (BD million). Miss = 282.5 / 4,329 = 6.5%.*

The months of 2026 are grouped by what was happening (dates and sources in `data/reference/events.csv`):

| Period | What was happening |
|---|---|
| **Jan-Feb** | Before the war (it began on 28 February) |
| **March** | The shock: Strait of Hormuz and Bahrain's airspace closed, Mina Salman Port struck |
| **Apr-May** | Airspace reopened on 8 April, but Hormuz stayed closed and a US naval blockade began |
| **Jun-Jul** | A brief, partial reopening of Hormuz from 18 June, then closed again on 11 July |

## 1. Three kinds of imports

![Monthly imports](figures/step3_1_monthly_imports.png)

Imports were split into three groups:

- **Core goods**: everything except the two groups below, from machinery and vehicles to food, clothing and chemicals. These are the everyday supply chains and the main subject of the project.
- **Bulk materials**: iron ore, alumina, coke and construction minerals, delivered in shiploads to a few plants.
- **Precious metals**: gold, silver and jewellery (customs chapter 71).

| Group | 2025 value (BD m) | % of value | % of weight | Normal miss (%) | Normal range (+/-%) | Jan-Feb | March | Apr-May | Jun-Jul |
|---|---|---|---|---|---|---|---|---|---|
| Core goods | 4,528 | 73 | 15 | 6 | 13 | +7 | -53 | -34 | -15 |
| Precious metals | 363 | 6 | 0 | 44 | 79 | +83 | -95 | +3 | +41 |
| Bulk materials | 1,294 | 21 | 85 | 16 | 35 | -38 | -62 | -81 | -63 |

*The last four columns compare each 2026 period with the 2025 average month.*

Core goods are 73% of import value and move relatively little in a normal year (a normal miss of about 6%). They were 7% above normal in January-February, 53% below normal in March and still 15% below normal by June-July. Bulk materials are only 21% of value but 85% of weight; they fell further and stayed down (63% below normal in June-July). Precious metals follow the gold price. They were already 83% above normal before the war, and their normal miss of about 44% is 7 times that of core goods.

Mixing the three would let one ore carrier or one gold consignment decide whether supply chains looked better or worse than they were. From here on core goods are the main subject and the other two groups are reported separately. The cost is that bulk materials and precious metals get a simpler treatment.

## 2. What a normal year looks like

![Seasonal pattern](figures/step3_5_patterns.png)

Two things matter for forecasting: how seasonal imports are, and whether they trend.

Seasonality is weak. When each month is compared with its own year's average (left chart), only about 17% of the month-to-month movement in core goods repeats from year to year. The clearest pattern is December, about 6% above a typical month. Ramadan makes no clear difference either: for food, a full month of Ramadan goes with +1% that month and -2% the month before, well within the normal ups and downs. With only four years of data this is a rough check rather than a precise estimate.

Imports do trend. Core goods changed by 0% in 2023, +3% in 2024 and +5% in 2025, and were 7% above the 2025 average in January-February 2026. The year 2021, under COVID rules, was smaller (BD 3,728m against BD 4,185m in 2022) but had a similar shape. Even the closures of malls and restaurants in May-June 2021 barely show: those months were 92% of that year's average month, against 99% in other years.

For the models this means the recent level of imports matters more than the calendar, so they are given averages of the last 3, 6 and 12 months. Calendar effects are small but are still offered as inputs, and 2021 is tested both with and without.

## 3. Where goods came from

![Origins](figures/step3_3_regions.png)

The customs data records where goods came from, not how they travelled, so origin is the best available clue to route. Countries are grouped into regions by how goods normally reach Bahrain: GCC goods can come by road over the King Fahd Causeway, while goods from East Asia, Europe or the Americas normally come by sea through the Strait of Hormuz, or by air.

| Origin region | % of 2025 value | Normal range (+/-%) | Jan-Feb | March | Apr-May | Jun-Jul |
|---|---|---|---|---|---|---|
| East and South-East Asia | 32.3 | 14 | +12 | -48 | -56 | -36 |
| Europe | 27.4 | 30 | +2 | -74 | -27 | -8 |
| GCC | 15.0 | 14 | +8 | -14 | +9 | +23 |
| Americas | 10.9 | 28 | -9 | -69 | -47 | -36 |
| South Asia | 6.9 | 28 | +31 | -58 | -47 | -5 |
| Other Middle East | 3.4 | 15 | +17 | -33 | -19 | -3 |
| Africa | 2.0 | 48 | -4 | -38 | -21 | -45 |
| Oceania | 1.4 | 63 | -3 | -84 | -62 | +24 |

*Core goods. "Other and not specified" (under 1% of value) is left out.*

Goods from GCC countries went up: 9% above normal in April-May and 23% above normal in June-July. Their share of core goods rose from 15% in 2025 to 25% in April-May. That fits the geography: GCC suppliers can send goods by road through Saudi Arabia, and the UAE's and Oman's ports on the Gulf of Oman and the Arabian Sea are outside the Strait. The biggest increases in June-July, compared with June-July 2025, were copper plates and sheets (+BD 4.6m), semi-finished steel (+BD 2.2m) and poultry meat (+BD 2.1m).

Goods from East and South-East Asia, the biggest origin at 32% of core goods, stayed down longest: 56% below normal in April-May and still 36% below in June-July. These goods normally come by container ship through Hormuz. The Americas were similar (36% below normal in June-July).

Goods from Europe came back faster (8% below normal in June-July), partly because of a few large aircraft-related deliveries: jet engines and gas turbines (+BD 29.3m) and aircraft engine parts (+BD 13.6m). Leaving out aircraft, engines and their parts, Europe was 16% below normal, still well ahead of Asia. Step 6 looks at why.

## 4. Which goods were hit

![Sections](figures/step3_4_sections.png)

Core goods were grouped into product sections, the top level of the customs classification.

| Section | % of 2025 value | Normal range (+/-%) | Jan-Feb | March | Apr-May | Jun-Jul |
|---|---|---|---|---|---|---|
| Machinery and electrical | 29.2 | 31 | +2 | -69 | -42 | -20 |
| Vehicles and transport | 13.0 | 29 | +5 | -66 | -74 | -41 |
| Chemicals | 9.5 | 15 | +2 | -34 | +7 | -4 |
| Base metals | 8.9 | 28 | +38 | -22 | -23 | -7 |
| Food, drinks and tobacco | 7.4 | 18 | -13 | -52 | -35 | -12 |
| Animal products | 6.3 | 21 | +24 | -37 | -18 | +2 |
| Vegetable products | 4.9 | 18 | +5 | -38 | -24 | +19 |
| Plastics and rubber | 4.2 | 22 | -3 | -40 | -34 | -4 |
| Textiles and clothing | 3.4 | 23 | +19 | -62 | -38 | -33 |
| Instruments and clocks | 3.2 | 25 | +10 | -68 | -12 | -14 |
| Miscellaneous manufactures | 2.3 | 23 | +1 | -43 | -29 | -22 |
| Stone, ceramics and glass | 1.9 | 17 | +6 | -41 | -30 | -5 |
| Paper | 1.3 | 39 | -6 | -35 | -26 | -9 |
| Wood and cork | 0.9 | 36 | +4 | -52 | -47 | -37 |
| Fats and oils | 0.9 | 34 | +19 | -39 | -15 | +14 |
| Minerals and fuels | 0.9 | 34 | +35 | -46 | -43 | -2 |
| Footwear and headgear | 0.8 | 35 | +11 | -62 | -47 | -28 |
| Hides, leather and travel goods | 0.6 | 32 | 0 | -67 | -30 | -26 |

*Sections under 0.5% of 2025 value, and the national special codes (confidential goods), are left out.*

Every section fell in March. By June-July most were back inside their normal range, with food first: animal products, vegetable products and fats and oils were at or above normal. Vehicles and clothing were still well below normal and outside their range.

### Bulk materials, by weight

| Material | kt a month (2025) | Jan-Feb | March | Apr-May | Jun-Jul |
|---|---|---|---|---|---|
| Iron ore | 1,080 | -53 | -69 | -100 | -95 |
| Construction minerals | 545 | -3 | -3 | -8 | +11 |
| Alumina | 262 | -12 | -64 | -73 | -34 |
| Petroleum and pitch coke | 67 | +6 | -9 | -79 | -83 |

The main origins (any with at least 10% of a material's weight in 2025 or in March-July 2026):

| Material | Origin | kt a month, 2025 | kt a month, Mar-Jul 2026 |
|---|---|---|---|
| Iron ore | Brazil | 760.9 | 87.0 |
| Iron ore | Canada | 155.0 | 0.0 |
| Iron ore | Chile | 136.4 | 0.0 |
| Construction minerals | United Arab Emirates | 387.3 | 490.0 |
| Construction minerals | Sultanate Oman | 68.1 | 0.1 |
| Construction minerals | Saudi Arabia | 61.0 | 55.2 |
| Alumina | Australia | 255.4 | 64.1 |
| Alumina | Saudi Arabia | 2.2 | 16.1 |
| Alumina | India | 0.0 | 14.0 |
| Petroleum and pitch coke | Kuwait | 21.1 | 11.2 |
| Petroleum and pitch coke | Spain | 20.8 | 6.4 |
| Petroleum and pitch coke | China | 9.4 | 2.4 |
| Petroleum and pitch coke | Sultanate Oman | 7.0 | 0.2 |

Construction minerals show route exposure most clearly: supplies from Oman, which have to pass through the Strait, stopped, while supplies from the UAE, inside the Gulf, went up. Alumina from Australia fell sharply and new suppliers appeared. But bulk materials also mix in a second effect. Iron ore was already 53% below normal in January-February, before the war, and coke from Kuwait, which doesn't need the Strait, also stopped in May-July, possibly because plants needed less (the aluminium smelter, Alba, was attacked on 28 March). Because blocked routes and lower demand can't be told apart here, bulk materials stay separate from core goods.

## 5. How goods arrived

![Routes](figures/step3_2_routes.png)

| Month | Core goods (value) | Core goods (weight) | Containers unloaded | Ships calling | Air cargo unloaded |
|---|---|---|---|---|---|
| Jan 2026 | 108 | 117 | 115 | 99 | 93 |
| Feb 2026 | 106 | 125 | 91 | 94 | 98 |
| Mar 2026 | 47 | 58 | 36 | 43 | 1 |
| Apr 2026 | 64 | 71 | 37 | 46 | 11 |
| May 2026 | 68 | 58 | 22 | 71 | 52 |
| Jun 2026 | 76 | 69 | 30 | 62 | 71 |
| Jul 2026 | 95 | 104 | 33 | 61 | 64 |
| Aug 2026 |  |  | 47 | 66 | 67 |
| Sep 2026 |  |  | 48 | 59 |  |

*Each month as % of that series' 2025 monthly average. Port = Khalifa Bin Salman Port, Bahrain's container port. Import figures stop at July; port and airport figures are published sooner.*

In June-July, core goods were back to 85% of normal by value and 86% by weight. Containers unloaded at the port, though, were only at 31% of normal and air cargo at 67%. A much bigger share than usual must therefore have arrived another way. The data covers the container port and the airport, so the most likely other way is by road over the King Fahd Causeway from Saudi Arabia, although general cargo ships and smaller ports are possible too. The rise in goods from GCC countries points the same way. This is the main route finding so far, and step 6 tests it properly.

## 6. How much detail the models can handle

![Levels](figures/step3_6_levels.png)

The last question is how finely to split core goods for the models. I worked out the normal miss at levels of detail from the national total down to each 4-digit product from each country. A series is regular if it has imports in at least 54 of the 60 months of 2021-2025. The "same-month rule" uses the same month a year earlier as the forecast, and the 2026 gap is the same calculation for March-July 2026.

| Level | Series | Regular series | % of value in regular series | Normal miss (%) | Same-month rule miss (%) | Gap in Mar-Jul 2026 (%) |
|---|---|---|---|---|---|---|
| National total | 1 | 1 | 100.0 | 6 | 8 | 30 |
| Origin region | 9 | 9 | 100.0 | 12 | 15 | 35 |
| Section | 21 | 20 | 100.0 | 14 | 17 | 33 |
| HS2 | 96 | 95 | 100.0 | 19 | 24 | 38 |
| Section x region | 186 | 160 | 100.0 | 23 | 30 | 41 |
| HS2 x region | 836 | 551 | 99.3 | 31 | 39 | 48 |
| HS4 | 1,185 | 846 | 99.1 | 35 | 43 | 50 |
| HS4 x region | 8,243 | 2,692 | 94.1 | 51 | 62 | 62 |
| HS4 x country | 44,245 | 5,268 | 82.7 | 68 | 80 | 75 |

The national total is easy to forecast (a normal miss of about 6%), and the 2026 gap of about 30% stands out clearly. The more detail, though, the bigger the normal miss, because in a large total one product's high month cancels another's low month, and splitting the total stops that cancelling. At section level the 2026 gap is 2.4 times the normal miss; at 4-digit level it is only 1.4 times; and for each product from each country the normal miss (68%) is almost as large as the war's effect. The average-month rule beats the same-month rule at every level, so it is the simple forecast the models have to beat.

A few products cause a disproportionate share of the misses (general goods including precious metals, 2024-2025):

| HS4 | Product | % of misses | % of value |
|---|---|---|---|
| 8409 | Aircraft engine parts | 7.9 | 5.2 |
| 7108 | Gold | 5.6 | 3.1 |
| 8411 | Jet engines and gas turbines | 5.4 | 2.2 |
| 8703 | Cars | 3.6 | 8.6 |
| 8517 | Phones and network equipment | 2.9 | 3.5 |
| 7113 | Jewellery | 2.4 | 2.9 |
| 3004 | Medicines | 1.4 | 2.9 |
| 7601 | Unwrought aluminium | 1.3 | 0.7 |

Gold and jewellery are part of why precious metals are kept separate. Aircraft engines and parts are large, occasional purchases; they stay in core goods, but step 5 checks them so that one delivery isn't mistaken for a recovery.

## Decisions for step 4

1. **Core goods are the main analysis.** Bulk materials (by weight) and precious metals are reported separately, because one shipment can swing them.
2. **The models forecast each 2-digit chapter from each origin region.** That gives 551 regular series covering 99.3% of core goods, at a level where the war is still clearly bigger than normal noise. Patchy series are pooled into one "other" series per section and region. Adding the forecasts up gives totals by section, by region and for the whole country that agree with each other. The cost is that a shift between two products in the same chapter is invisible to the models.
3. **Results are reported nationally, by section and by region**, where the effect is clearly bigger than the normal miss. More detailed results are only shown with their uncertainty.
4. **The simple forecast to beat is the previous year's average month**, with the same-month rule reported too.
5. **The models get the recent level and trend** (averages over the last 3, 6 and 12 months), because imports were growing going into 2026.
6. **2021 is tested both ways**: kept in with a COVID flag, and left out.

## Limitations

- The yardstick in this step ignores growth. Imports were rising going into 2026, so falls measured against 2025 understate how far imports were below where they would otherwise have been. Step 5 corrects this by using model forecasts.
- The normal range is based on three years (36 months per series), so it is a rough guide, not a precise 90% interval.
- Origin region is a proxy for route. A good from China could have come by air, or have been landed in Saudi Arabia and trucked in.
- Customs records the month goods were declared, which is not always the month they arrived. Periods of two months soften this.

---

*This report is generated by `src/explore.py`. Re-running the pipeline (`python src/run_all.py`) refreshes every number and chart. Data: Bahrain Open Data Portal (iGA), www.data.gov.bh.*
