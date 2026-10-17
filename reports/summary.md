# Shock and Recovery: Bahrain's import supply chains in 2026

On 28 February 2026, war in the Gulf closed the Strait of Hormuz and, for more than a month, Bahrain's airspace. Everything that reaches Bahrain by sea from outside the Gulf passes through that strait. I used Bahrain's official monthly import statistics (1.76 million rows, January 2021 to July 2026) together with port and airport figures to measure how far imports fell below what they would otherwise have been, which supply chains and routes broke, and how far they had come back.

## Key findings

- **Everyday imports fell 54% below normal in March 2026**, then 36% in April-May and 18% in June-July, against a tested forecast of imports without the war. About BD 624 million of goods didn't arrive from March to July (32% of what was expected; likely between BD 539 and 710 million).
- **Vehicles and transport were hit hardest (-62%)**, and machinery and vehicles together were 64% of the money lost. Goods from the Americas and East and South-East Asia fell furthest (both -48%); goods from GCC countries held up (+8%, within the normal range).
- **Container ships and car carriers were the routes that broke.** Containers unloaded ran at about a third of normal from March to July, and car carriers unloaded no vehicles at all from April.
- **Goods got in another way.** Containers and planes could have carried at most 46% of what arrived by weight, so at least 54% came over the King Fahd Causeway or by general cargo ship. While planes could fly but ships couldn't pass Hormuz, light goods from far away fell 13% (leaving out aircraft engines) and heavy goods 54%.
- **Supply chains came back about two-thirds of the way by Jun-Jul.** Vehicles, clothing, East Asia and the Americas lagged. Port figures suggest they kept adapting after Hormuz closed again on 11 July: containers unloaded rose from 30% of normal in Jun-Jul to 47% in Aug-Sep.
- **Relying on few supplier countries didn't matter; relying on distant ones did.** Products bought from one or two countries fell about as far as products with many suppliers, while each extra 10% of a product from beyond the Middle East meant about 5 points further below normal. The lesson is to diversify routes, not just countries.

![Imports against the forecast without the war](figures/step5_1_national.png)

## Approach

1. **Forecast imports without the war.** Core goods were split into about 660 monthly series (one customs chapter from one origin region). A ridge regression written from scratch learned from 2021-2025 how these usually move and, standing at the end of December 2025, forecast every month of 2026 as if nothing had happened.
2. **Test the forecast on normal years first.** The same model was run on 2024 and 2025 as if it were January of each year, using only data available at the time. Its misses set a normal range for every total and period length.
3. **Compare 2026 with the forecast.** A gap only counts as a real shortfall if it is bigger than the normal range. A second model (gradient-boosted trees) gives a second opinion.
4. **Follow the routes.** Port and airport figures show which routes broke. The causeway publishes nothing, so its share is bounded by elimination: the most containers and planes could have carried, taken away from what arrived.

The [methodology note](../docs/methodology.md) explains each choice and its trade-offs. The periods used throughout:

| Period | What was happening |
|---|---|
| **Jan-Feb** | Before the war (it began on 28 February) |
| **March** | The shock: Strait of Hormuz and Bahrain's airspace closed, Mina Salman Port struck |
| **Apr-May** | Airspace reopened on 8 April, but Hormuz stayed closed and a US naval blockade began |
| **Jun-Jul** | A brief, partial reopening of Hormuz from 18 June, then closed again on 11 July |
| **Aug-Sep** | Hormuz closed again from 11 July (still closed in October) |

## 1. Data

The import data has one row per month, 8-digit product code and origin country, from the Bahrain Open Data Portal. Six yearly files in three slightly different formats were combined into one table, repeated shipments added together, and every month's totals checked against the portal's own figures; nothing was dropped. Imports were split into three groups that behave very differently: core goods (the everyday supply chains, about three-quarters of value), bulk materials (iron ore, alumina, coke and building minerals, most of the weight, for a few plants) and precious metals (which follow the gold price). This summary is about core goods. The route data are monthly figures for Khalifa Bin Salman Port and Bahrain International Airport, which run two months ahead of the import data. Several months of the port's general cargo figures had a decimal point typed for a thousands separator and were repaired and marked.

## 2. The forecast

![Model comparison](figures/step4_2_model_comparison.png)

Four forecasts were compared, each tested on 2024 and 2025 as if it were January: the simple rule (every month like last year's average month), the same month last year, the ridge regression (solved both with the normal equations and by gradient descent, which agree to floating-point precision), and gradient-boosted trees (scikit-learn).

| Forecast | Miss in 2024 | Miss in 2025 |
|---|---|---|
| **Linear model** | 5.4% | 5.5% |
| Simple rule (last year's average month) | 6.2% | 6.2% |
| Gradient-boosted trees | 5.7% | 6.3% |

*National miss: the total absolute monthly error as % of actual imports.*

The linear model won in both years and at every level, mostly because it gets the level right while imports are growing. The trees never beat it, and in the 2025 final check they did no better than the simple rule (6.3%): with five years of noisy monthly data their flexibility mostly fitted noise. The normal ranges work: set on 2024 and checked on 2025, 88-92% of months fell inside them, against a target of 90%. The model also got January-February 2026 right, before the war (+3%, inside the normal range).

## 3. How far imports fell

| Period | Forecast (BD m) | Actual (BD m) | Gap | Normal range | Real shortfall? |
|---|---|---|---|---|---|
| Jan-Feb | 784 | 806 | +3% | +/-8% | no |
| March | 385 | 178 | -54% | +/-15% | **yes** |
| Apr-May | 770 | 495 | -36% | +/-8% | **yes** |
| Jun-Jul | 788 | 645 | -18% | +/-8% | **yes** |
| **Since the war (Mar-Jul)** | 1,942 | 1,318 | -32% | +/-4% | **yes** |

March was the worst month. Each period after that was better, but every war-time period was a real shortfall, and in total about BD 624 million was lost from Mar-Jul. The gradient-boosted trees put the national gap at -33%, against -32% for the linear model.

## 4. Which supply chains were hit hardest

The five sections that fell furthest over Mar-Jul:

| Section | Gap | Lost (BD m) |
|---|---|---|
| Vehicles and transport | -62% | 165 |
| Footwear and headgear | -45% | 8 |
| Wood and cork | -43% | 7 |
| Machinery and electrical | -41% | 238 |
| Textiles and clothing | -41% | 26 |

By origin:

| Origin region | Gap | Real change? |
|---|---|---|
| The Americas | -48% | yes |
| East and South-East Asia | -48% | yes |
| Africa | -33% | yes |
| Europe | -32% | yes |
| South Asia | -32% | yes |
| Oceania | -31% | no (inside the normal range) |
| The rest of the Middle East | -16% | yes |
| GCC | +8% | no (inside the normal range) |

The pattern is about routes more than products: goods made far away and shipped through Hormuz fell furthest, while goods from GCC countries, which can come by road, held up.

## 5. Which routes broke, and how goods got around them

![Route groups](figures/step6_2_route_groups.png)

| Route | March | Apr-May | Jun-Jul | Aug-Sep |
|---|---|---|---|---|
| Containers unloaded at the port | 33% | 32% | 30% | 47% |
| Vehicles unloaded from car carriers | 16% | 0% | 0% | 0% |
| Air cargo and mail | 1% | 31% | 70% | 69% |
| Core goods arriving, by weight | 65% | 65% | 79% | - |

*% of the same months of 2025.*

Containers fell to about a third of normal and car carriers stopped. Planes were grounded in March and came back slowly, never above normal. Goods with another way in held up: goods from GCC countries were +8% against the forecast, while goods from beyond the Middle East were -40%. In April-May, with the airspace open but Hormuz shut, light goods from beyond the Middle East were -13% (leaving out aircraft engines) against -54% for heavy goods.

![Elimination](figures/step6_3_elimination.png)

By weight, core goods arrived at 71% of their 2025 level while containers ran at 31%. Even assuming every import had come in a container in 2025, containers and planes could have carried at most 46% of what arrived. The rest, about 660 thousand tonnes, must have come over the causeway or by general cargo ship. Cars made in East Asia and Europe kept arriving from May even though no car carrier called.

## 6. How far supply chains came back

![Adapting](figures/step6_4_adapting.png)

Nationally, imports came back about two-thirds of the way by Jun-Jul, from 46% of normal in March to 82%. Back inside their normal range were Europe, South Asia, the rest of the Middle East and Oceania among origins, and goods from GCC countries were above normal (119%). Still far below normal were vehicles and transport (54%), wood and cork (64%), textiles and clothing (66%) and footwear and headgear (67%) among sections, and Africa (55%), East and South-East Asia (61%) and the Americas (61%) among origins. Part of the bounce came while Hormuz was briefly open (18 June to 11 July), so not all of it was adapting. After July, with Hormuz closed again, containers unloaded rose from 30% of normal in Jun-Jul to 47% in Aug-Sep, a sign that supply chains kept finding other ways in.

## 7. Did relying on few supplier countries matter?

![Suppliers](figures/extra_suppliers_1_groups.png)

Grouped by how many countries they relied on (measured as an effective number of suppliers), products fell to between 61% and 76% of normal, in no particular order. Allowing for distance in a weighted regression, each doubling of the number of supplier countries changed a product's result by +1.7 points (90% bootstrap range -3.7 to +7.3), which is no measurable effect. Distance mattered: products bought mostly nearby were at 103% of normal, and products almost entirely from beyond the Middle East at 59%. Buyers switched a little: 27% of far-dependent products' imports came from countries that had been minor suppliers, against 19-23% in normal years, mostly other distant countries. Five suppliers in East Asia all ship through the same strait, so having five of them gave no protection.

## Limitations

- **The causeway has no published figures**, so the road's share is a minimum worked out by elimination, not a measurement.
- **The normal ranges come from two test years.** More normal years would make them more reliable.
- **Both models learn from the same five years**, so their agreement is reassuring but not independent proof.
- **Customs months and arrival months don't always line up.** Periods of two months or more soften this.
- **The port's general cargo figures needed repairs** and may include goods loaded for export, so no conclusion rests on them alone.

## What I'd do next

- **A "what if" range for the road's own share**, under stated assumptions about container loads and how much general cargo was imports.
- **Update with August 2026 onwards** when the portal publishes it: the first months with Hormuz closed and no reopening.

## Detailed reports

| Step | Report | Script |
|---|---|---|
| 2. Clean | [Cleaning the data](step2_cleaning_report.md) | `src/build_dataset.py` |
| 3. Explore | [Exploring the data](step3_exploration.md) | `src/explore.py` |
| 4. Forecast | [Forecasting imports without the war](step4_models.md) | `src/backtest.py` |
| 5. Shortfall | [The 2026 shortfall](step5_shortfall.md) | `src/shortfall.py` |
| 6. Routes | [Routes and adapting](step6_routes.md) | `src/routes.py` |
| Extra | [Supplier concentration](extra_suppliers.md) | `src/suppliers.py` |

---

*This report is generated by `src/summary.py` from the saved results of steps 4 to 6. Re-running the pipeline (`python src/run_all.py`) refreshes every number and chart. Data: Bahrain Open Data Portal (iGA), www.data.gov.bh.*

*All analysis is my own and is not attributed to the iGA.*
