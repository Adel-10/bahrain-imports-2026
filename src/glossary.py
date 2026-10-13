"""Plain-language definitions shared by the step 3, 4 and 5 reports.

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
