# Extra: Did relying on few supplier countries matter?

A common lesson from supply shocks is to spread purchases across many supplier countries, so that losing one doesn't matter much. Steps 5 and 6 showed which goods and origins were hit hardest in 2026. This extra asks whether products that depended on one or two countries were hit harder than products with many suppliers, and if not, what did matter.

## Key findings

- **The number of supplier countries made no measurable difference.** Grouped by how many countries they relied on, products fell to between 61% and 76% of normal over Mar-Jul, in no particular order. Compared fairly, each doubling of the number of suppliers changed a product's result by +1.7 points (90% range -3.7 to +7.3).
- **Distance did.** Products bought mostly from nearby countries were at 103% of normal; products almost entirely from beyond the Middle East were at 59%. Each extra 10% from beyond the Middle East meant about 5 points further below normal (90% range 3.6 to 6.6). The result holds however the products are weighted.
- **Buyers switched a little.** For products that normally come from far away, 27% of what arrived came from countries that had been minor suppliers, against 19-23% in normal years. Most of those were other distant countries.
- **The lesson for Bahrain is to diversify routes, not just countries.** Five suppliers in East Asia all ship through the same strait.

## Method

**Products and forecasts.** Each product is one 4-digit HS heading from all origins together. 844 products had imports in almost every month before the war; together they are 98% of core goods by value. Step 4's linear model, rebuilt at this level (as for step 5's close-ups), forecasts each one for 2026 as if there had been no war, and its 2024-2025 tests give each group of products a normal range. Added up, the products were at 67% of normal over Mar-Jul, in line with step 5.

**Reliance on few countries.** Counting supplier countries isn't enough: a product with one big supplier and ten tiny ones really relies on one. Each product therefore gets an effective number of supplier countries from its 2024-2025 imports, 1 / (sum of each country's squared share), which is the inverse of the Herfindahl-Hirschman index. Two countries with 50% each give 2; 90% from one and 10% from another gives 1.2; four countries with 25% each give 4.

**Distance.** Each product also gets its share from beyond the Middle East in 2024-2025: the part that normally has to pass the Strait of Hormuz or fly.

**Comparing fairly.** The two measures could be tangled together: if products with few suppliers also tended to come from far away, they would look worse for the wrong reason. So after comparing groups one way at a time, a weighted regression looks at both at once, holding value per kg (light goods could fly) and vehicles (car carriers stopped) constant. Products are weighted by size, and % of normal is capped at 200% so a few tiny products with huge swings can't dominate. Re-running the regression on 2,000 random re-draws of the products (a bootstrap) gives each effect a 90% range without assuming the errors follow a normal distribution.

The approach measures every product against its own tested forecast and separates the two explanations, but it shows association rather than cause, and single-product forecasts are noisy. The regression is therefore repeated with two other weightings as a check.

## 1. Number of supplier countries

![Groups](figures/extra_suppliers_1_groups.png)

| Supplier countries (effective) | Products | Share of value | From beyond the Middle East | % of normal (group total) | Median product | Normal range | Real shortfall? |
|---|---|---|---|---|---|---|---|
| Under 2 | 108 | 12% | 82% | 70% | 67% | +/-11% | **yes** |
| 2 to 3 | 165 | 14% | 84% | 76% | 59% | +/-21% | **yes** |
| 3 to 5 | 279 | 30% | 78% | 63% | 62% | +/-8% | **yes** |
| 5 to 8 | 218 | 28% | 82% | 61% | 64% | +/-6% | **yes** |
| 8 or more | 74 | 16% | 79% | 76% | 65% | +/-7% | **yes** |

*Mar-Jul. "Median product" is the middle product of each group, which is less affected by a few big products than the group total.*

Every group had a real shortfall, falling to between 61% and 76% of normal, and the order doesn't follow the number of suppliers: products with under 2 effective suppliers (70%) did better than the 5 to 8 group (61%). Single large products move these totals; cars, one of the biggest products and at 35% of normal, sit in the "5 to 8" group. Leaving out cars and aircraft engines, the groups range from 67% to 76%, with a slight edge for products with more suppliers. Section 3 checks whether that edge survives once distance is allowed for (it doesn't clearly). The groups are also about equally dependent on distant suppliers, each getting around 81% of its value from beyond the Middle East, a first hint that distance rather than the number of countries decided the result.

The biggest products that relied on one or two countries:

| Product | Main supplier | From beyond the Middle East | % of normal |
|---|---|---|---|
| Phones and network equipment | China (70%) | 100% | 57% |
| Refined copper | Democratic Republic of Congo (85%) | 98% | 46% |
| Wristwatches (other) | Switzerland (91%) | 100% | 90% |
| Rice | India (84%) | 100% | 88% |
| Wristwatches (precious metal) | Switzerland (97%) | 100% | 91% |
| Silicon | China (95%) | 96% | 50% |

Even these ranged from 46% to 91% of normal; relying on one country didn't decide how they fared.

## 2. Distance

| From beyond the Middle East | Products | Share of value | Typical number of suppliers | % of normal (group total) | Median product | Normal range | Real shortfall? |
|---|---|---|---|---|---|---|---|
| Under 25% | 61 | 6% | 1.9 | 103% | 98% | +/-6% | no |
| 25-50% | 71 | 10% | 4.3 | 91% | 83% | +/-9% | **yes** |
| 50-75% | 132 | 13% | 5.6 | 69% | 68% | +/-7% | **yes** |
| 75-90% | 141 | 17% | 4.9 | 68% | 59% | +/-6% | **yes** |
| 90% or more | 439 | 55% | 3.6 | 59% | 57% | +/-7% | **yes** |

*Mar-Jul.*

Here there is a clear staircase. Products bought mostly nearby were at 103% of normal (inside the normal range). The more a product depended on distant suppliers, the further it fell, down to 59% for products almost entirely from beyond the Middle East. The median product tells the same story, so this isn't driven by a few big products.

## 3. Both at once

![Effects](figures/extra_suppliers_2_effects.png)

| Feature of the product | Change in % of normal | 90% range | Clear effect? |
|---|---|---|---|
| Each doubling of the number of supplier countries | +1.7 points | -3.7 to +7.3 | no (the range includes zero) |
| Each extra 10% of the product from beyond the Middle East | -5.1 points | -6.6 to -3.6 | **yes** |
| Each doubling of value per kg | +1.8 points | -0.8 to +4.6 | no (the range includes zero) |
| Vehicles and transport (compared with other goods) | -24.4 points | -31.8 to -9.7 | **yes** |

*Weighted by size. Supplier count and value per kg enter as doublings, distance per 10 percentage points.*

The same regression with other weightings:

| Weighting | Each doubling of supplier countries | Each extra 10% from beyond the Middle East |
|---|---|---|
| Weighted by size (main) | +1.7 (-3.7 to +7.3) | -5.1 (-6.6 to -3.6) |
| Weighted by the square root of size | -0.6 (-3.2 to +2.4) | -4.8 (-5.8 to -3.7) |
| Every product counts equally | +0.4 (-1.9 to +3.0) | -4.3 (-5.3 to -3.4) |

The number of supplier countries makes no measurable difference: every range includes zero. Even at the top of its main range (+7.3 points per doubling), going from a typical "under 2" product (1.5 suppliers) to a typical "8 or more" one (9.3) would be worth about 19 points, well below the 51 points between buying everything nearby and everything from far away. Distance makes a clear difference under every weighting, about 5 points per extra 10% from beyond the Middle East. Vehicles did worse still, even after allowing for distance, because car carriers stopped altogether (step 6).

## 4. Did buyers switch suppliers?

![Switching](figures/extra_suppliers_3_switching.png)

For each product, the countries that had supplied under 5% of it in the two years before count as minor suppliers. The table shows how much of each year's March-July imports came from them, so 2026 can be compared with the normal churn among suppliers.

|  | All products: from minor suppliers | Far-dependent products: from minor suppliers | ... of which minor suppliers in GCC countries | Far-dependent products: all GCC suppliers |
|---|---|---|---|---|
| 2024 (normal year) | 23% | 23% | 1.8% | 3.6% |
| 2025 (normal year) | 20% | 19% | 1.3% | 3.0% |
| 2026 (war) | 24% | 27% | 3.4% | 5.7% |

*"Far-dependent" products get at least 75% of their value from beyond the Middle East.*

Some switching happened for far-dependent products: 27% of what arrived came from minor suppliers, against 19-23% in normal years. Most of it went to other distant countries rather than nearby ones; GCC suppliers' share of these products rose from 3-4% to 6%, almost double but still small. Brand-new suppliers were rare: 0.5% of far-dependent imports came from countries that hadn't supplied the product at all in the two years before. Buyers adapted by leaning on countries they already bought a little from, mostly distant ones whose goods could still get through.

## Limitations

- **Country of origin isn't the shipping route.** The customs data records where goods were made, not how they travelled; a Chinese product bought from a Dubai warehouse counts as Chinese.
- **One heading can hold many items.** HS heading 8471 (computers) mixes laptops and parts, which may have different suppliers.
- **Single products are hard to forecast**, so this analysis compares groups and uses ranges and re-draws rather than reading much into any one product.
- **Association, not cause.** The regression holds a few things constant, but other differences between products could still play a part.
- **The normal ranges come from two test years**, as in the rest of the project.

---

*This report is generated by `src/suppliers.py` (charts and text in `src/suppliers_report.py`). Re-running the pipeline (`python src/run_all.py`) refreshes every number and chart. Data: Bahrain Open Data Portal (iGA), www.data.gov.bh.*
