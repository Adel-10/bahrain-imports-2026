"""Plain-language definitions shared by the step 3 to 6 reports.

Each report picks the terms it uses, so the same word always has the same definition.
Where a word means something slightly different in one step (for example "normal"),
that step passes its own wording.
"""

from __future__ import annotations

TERMS: dict[str, tuple[str, str]] = {
    "core_goods": (
        "Core goods",
        "All imports except bulk materials (iron ore, alumina, coke and construction minerals) and precious metals "
        "(gold, silver and jewellery). These are the everyday supply chains, about three-quarters of import value.",
    ),
    "national_total": (
        "National total (also *nationally* or *All of Bahrain*)",
        "All core goods imports into Bahrain in a month, added together: every product from every origin, in one "
        "number. It doesn't include bulk materials or precious metals, or oil (the portal's import data covers non-oil "
        "imports only).",
    ),
    "origin_region": (
        "Origin region",
        "Where goods came from, grouped into 9 regions by how they normally reach Bahrain. **GCC** means Saudi Arabia, "
        "the UAE, Kuwait, Qatar and Oman, including the Jebel Ali and Oman free zones.",
    ),
    "section": (
        "Section",
        "One of the 21 top-level groups of the customs (HS) code, such as \"Machinery and electrical\" or \"Vehicles and "
        "transport\". Each section is made up of HS2 chapters.",
    ),
    "hs_codes": (
        "HS2 chapter, HS4 heading",
        "Finer levels of the customs (HS) code: 2 digits (96 chapters, for example 84 = machinery) and 4 digits "
        "(about 1,200 headings, for example 8703 = cars).",
    ),
    "series": (
        "Series",
        "One line of monthly numbers: one HS2 chapter from one origin region, for example \"chapter 84 (machinery) from "
        "Europe\".",
    ),
    "units": (
        "BD m, kt",
        "Million Bahraini dinars (value) and thousand tonnes (weight).",
    ),
    "periods": (
        "Jan-Feb, March, Apr-May, Jun-Jul",
        "Groups of months in 2026, by what was happening at the time (see the periods table below).",
    ),
    "normal_year": (
        "Normal year",
        "A year without a major shock to imports. Here that means 2024 and 2025, the years used to test the models.",
    ),
    "simple_rule": (
        "Simple rule",
        "The simplest forecast: every month will be like last year's average month.",
    ),
}

# Words whose meaning depends on the step.
STEP_TERMS: dict[int, dict[str, tuple[str, str]]] = {
    3: {
        "normal": (
            "Normal",
            "The 2025 average month of the same thing. \"-50%\" in March means March 2026 was half an average 2025 month.",
        ),
        "normal_miss": (
            "Normal miss",
            "How far the simple rule's forecasts were from the actual values in 2024 and 2025, on average over the "
            "months, as % of the forecast (see before-you-start, step 3).",
        ),
        "normal_range": (
            "Normal range",
            "How far a single month strayed from the previous year's average month, 9 times out of 10, in 2023-2025 "
            "(see before-you-start, step 4). The grey band in the charts.",
        ),
        "containers": (
            "Containers unloaded",
            "Shipping containers taken off ships at Khalifa Bin Salman Port, Bahrain's container port, counted in TEUs "
            "(one TEU is one twenty-foot container).",
        ),
        "regular": (
            "Regular series",
            "A series with imports in at least 54 of the 60 months of 2021-2025.",
        ),
    },
    4: {
        "clue": (
            "Clue",
            "A piece of information the model uses to make a forecast, such as \"the average of the last 3 months\". "
            "Machine learning calls these *features*.",
        ),
        "weight": (
            "Weight",
            "How much the linear model counts each clue. Machine learning calls these *coefficients*.",
        ),
        "miss": (
            "Miss",
            "How far a forecast was from what actually arrived, ignoring whether it was too high or too low. Averaged "
            "over the months and shown as % of actual imports. Lower is better. (Step 3's \"normal miss\" divided by the "
            "forecast instead; see before-you-start, step 5.)",
        ),
        "overall_score": (
            "Overall score",
            "The average of the miss for the national total, for origin regions and for sections: the three levels "
            "step 5 reports. Used to pick the winner.",
        ),
        "total_for_year": (
            "Total for the year",
            "Whether the forecasts, added up over the whole year, were too low or too high.",
        ),
        "range": (
            "Range",
            "How far from the model's forecast a single month can be in a normal year: 9 months out of 10 stay inside it.",
        ),
    },
    5: {
        "forecast": (
            "Forecast",
            "What the model expected imports to be **without the war**. It was made at the end of December 2025, so it "
            "never saw any 2026 data.",
        ),
        "normal": (
            "Normal, below normal",
            "In this report, \"normal\" means the forecast: \"30% below normal\" means 30% below what the model expected "
            "without the war.",
        ),
        "gap": (
            "Gap",
            "Actual imports minus the forecast: in BD, or as % of the forecast. Negative means below normal.",
        ),
        "normal_range": (
            "Normal range",
            "How far from the forecast a total can be in a normal year (9 times out of 10), measured in step 4's tests. "
            "Longer periods have narrower ranges.",
        ),
        "real_shortfall": (
            "Real shortfall",
            "A gap bigger than the normal range: too big to be ordinary ups and downs.",
        ),
        "lost": (
            "Lost imports",
            "The forecast minus what actually arrived, in BD.",
        ),
        "since_war": (
            "Since the war",
            "March 2026 to the latest month the portal has published, added together.",
        ),
        "second_opinion": (
            "Second opinion",
            "The same comparison using step 4's best gradient-boosted trees instead of the linear model.",
        ),
    },
}


STEP_TERMS[6] = {
    **{k: STEP_TERMS[5][k] for k in ("forecast", "gap", "normal_range", "real_shortfall", "since_war")},
    "normal": (
        "Normal",
        "Two meanings, one for each kind of data. For **imports**, normal means step 5's forecast: what was expected "
        "without the war. For **port and airport figures**, which have no forecast, normal means the same months of "
        "2025.",
    ),
    "pct_of_normal": (
        "% of normal",
        "Actual as a percentage of normal: 100% is normal, 50% is half. (Step 5's gap of -54% is 46% of normal.)",
    ),
    "route": (
        "Route",
        "How goods physically reach Bahrain: by container ship, car carrier or general cargo ship (all at Khalifa Bin "
        "Salman Port), by plane (Bahrain International Airport), or by truck over the King Fahd Causeway.",
    ),
    "teu": (
        "TEU",
        "One twenty-foot shipping container. A standard forty-foot container counts as 2 TEU.",
    ),
    "car_carrier": (
        "Car carrier (RoRo)",
        "A \"roll-on, roll-off\" ship that vehicles are driven on and off. The port counts the vehicles unloaded.",
    ),
    "general_cargo": (
        "General cargo",
        "Goods that travel neither in containers nor loose in bulk, loaded piece by piece: steel, timber, machinery, "
        "bagged food, livestock. Counted in freight tonnes.",
    ),
    "freight_tonne": (
        "Freight tonne",
        "A shipping unit: one tonne, or one cubic metre if the goods are bulky, whichever is larger. So freight tonnes "
        "are never less than the weight in tonnes.",
    ),
    "causeway": (
        "King Fahd Causeway",
        "The 25 km road bridge between Saudi Arabia and Bahrain, Bahrain's only road link to the mainland. No monthly "
        "freight figures for it are published.",
    ),
    "beyond_me": (
        "Beyond the Middle East",
        "Goods from Asia, Europe, the Americas, Africa and Oceania (plus a small \"not specified\" group). Before the "
        "war these came mostly by sea through the Strait of Hormuz, or by plane.",
    ),
    "weight_class": (
        "Light, middle and heavy goods",
        "Series grouped by their value per kilogram in 2024-2025. **Light**: BD 5 per kg or more (about $13), worth "
        "flying: machinery and parts, medicines, instruments, watches, clothes. **Heavy**: under BD 1 per kg: steel, "
        "plastics, meat, fruit, rice, building materials. **Middle**: in between. Vehicles are kept apart because "
        "they travel on their own kind of ship.",
    ),
    "elimination": (
        "Elimination",
        "Working out how much came by a route with no figures (the road) by taking away the most the other routes "
        "could have carried.",
    ),
    "came_back": (
        "Came back ...% of the way",
        "How much of the drop had been made up by the latest period. From 46% of normal at the worst to 82% is "
        "(82 - 46) / (100 - 46) = two-thirds of the way. Over 100% means it ended above normal.",
    ),
    "adapting": (
        "Adapting",
        "Finding other ways to get goods in while the usual route is shut. Used instead of \"recovery\" because the "
        "Strait of Hormuz closed again on 11 July and was still closed in October.",
    ),
}


# The project summary (written in step 7) uses words from every step.
STEP_TERMS[7] = {
    **STEP_TERMS[6],
    "lost": STEP_TERMS[5]["lost"],
    "miss": (
        "Miss",
        "How far a forecast was from what actually arrived, ignoring whether it was too high or too low, as % of "
        "actual imports. Lower is better.",
    ),
}


def label(name: str) -> str:
    """'National total (also ...)' -> '**National total** (also ...)'."""
    if " (" in name:
        main, rest = name.split(" (", 1)
        return f"**{main}** ({rest}"
    return f"**{name}**"


def table(step: int, keys: list[str]) -> str:
    """A two-column markdown table of the chosen terms, in the given order."""
    lookup = {**TERMS, **STEP_TERMS.get(step, {})}
    rows = ["| Word | What it means here |", "|---|---|"]
    rows += [f"| {label(lookup[k][0])} | {lookup[k][1]} |" for k in keys]
    return "\n".join(rows)
