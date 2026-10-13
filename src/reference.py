"""Small reference tables the pipeline needs: HS sections, product groups, calendar.

Everything here is hand-checked data rather than downloaded data, so it lives in
code where changes are visible in version control.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

import pandas as pd
from hijridate import Gregorian, Hijri

# ---------------------------------------------------------------------------
# HS sections: the 21 top-level groups of the Harmonized System.
# (section number, roman numeral, short name, first chapter, last chapter)
# ---------------------------------------------------------------------------
HS_SECTIONS = [
    (1, "I", "Animal products", 1, 5),
    (2, "II", "Vegetable products", 6, 14),
    (3, "III", "Fats and oils", 15, 15),
    (4, "IV", "Food, drinks and tobacco", 16, 24),
    (5, "V", "Minerals and fuels", 25, 27),
    (6, "VI", "Chemicals", 28, 38),
    (7, "VII", "Plastics and rubber", 39, 40),
    (8, "VIII", "Hides, leather and travel goods", 41, 43),
    (9, "IX", "Wood and cork", 44, 46),
    (10, "X", "Paper", 47, 49),
    (11, "XI", "Textiles and clothing", 50, 63),
    (12, "XII", "Footwear and headgear", 64, 67),
    (13, "XIII", "Stone, ceramics and glass", 68, 70),
    (14, "XIV", "Precious metals and jewellery", 71, 71),
    (15, "XV", "Base metals", 72, 83),
    (16, "XVI", "Machinery and electrical", 84, 85),
    (17, "XVII", "Vehicles and transport", 86, 89),
    (18, "XVIII", "Instruments and clocks", 90, 92),
    (19, "XIX", "Arms and ammunition", 93, 93),
    (20, "XX", "Miscellaneous manufactures", 94, 96),
    (21, "XXI", "Art and antiques", 97, 97),
    (22, "-", "National special codes", 98, 99),  # not part of the international HS
]

_CHAPTER_TO_SECTION = {ch: (num, name) for num, _, name, lo, hi in HS_SECTIONS for ch in range(lo, hi + 1)}


def chapter_to_section(chapter: str | int) -> tuple[int, str]:
    """'84' -> (16, 'Machinery and electrical')."""
    return _CHAPTER_TO_SECTION[int(chapter)]


def hs_sections_table() -> pd.DataFrame:
    rows = [
        {"chapter": f"{ch:02d}", "section_id": num, "section_roman": roman, "section_name": name}
        for num, roman, name, lo, hi in HS_SECTIONS
        for ch in range(lo, hi + 1)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Product groups. A handful of bulk raw materials make up most of Bahrain's
# import WEIGHT (iron ore alone is about half), so they get their own groups;
# otherwise they would drown out every weight-based comparison.
# Measured on 2024-2025 data: these HS4 codes = about 85% of import weight.
# ---------------------------------------------------------------------------
PRODUCT_GROUPS = {
    "2601": "Iron ore",
    "2818": "Alumina",
    "2713": "Petroleum and pitch coke",
    "2708": "Petroleum and pitch coke",
    "2517": "Construction minerals",   # gravel and crushed stone
    "2521": "Construction minerals",   # limestone
    "2505": "Construction minerals",   # silica sand
    "2522": "Construction minerals",   # quicklime
    "2523": "Construction minerals",   # cement
}
DEFAULT_GROUP = "General goods"

# Analysis groups (decided in step 3). Three kinds of imports behave very differently,
# so they are forecast and reported separately:
#   * Core goods: general goods except precious metals; the everyday supply chains.
#   * Precious metals (HS chapter 71: gold, silver, jewellery): driven by the gold price
#     and by a few very large shipments; normal forecast error is about 40% a year.
#   * Bulk materials: the PRODUCT_GROUPS above; arrive in whole shiploads for a few factories.
CORE, PRECIOUS, BULK = "Core goods", "Precious metals", "Bulk materials"
ANALYSIS_GROUPS = [CORE, PRECIOUS, BULK]
PRECIOUS_CHAPTER = "71"


def add_analysis_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical `group` column (Core goods / Precious metals / Bulk materials).

    Needs `product_group` and `hs2` columns (as produced by build_dataset.py).
    """
    bulk = df["product_group"].astype(object) != DEFAULT_GROUP
    precious = df["hs2"].astype(object) == PRECIOUS_CHAPTER
    group = pd.Series(CORE, index=df.index, dtype=object)
    group[precious] = PRECIOUS
    group[bulk] = BULK
    df["group"] = pd.Categorical(group, categories=ANALYSIS_GROUPS)
    return df


# ---------------------------------------------------------------------------
# Origin regions. Grouped by how goods normally reach Bahrain, because the
# trade data never says which route a shipment took:
#   * GCC (incl. the Jebel Ali and Oman free zones): can come overland through
#     Saudi Arabia and the King Fahd Causeway, or by short sea trips inside the Gulf.
#   * Other Middle East (incl. Turkey and Egypt): long overland routes are possible.
#   * Everyone else: normally by sea through the Strait of Hormuz, or by air.
# Codes not listed (Central Asia, the Caucasus, "not specified", returned
# Bahraini goods, duty-free) fall into "Other and not specified".
# ---------------------------------------------------------------------------
REGION_CODES = {
    "GCC": "SA AE FZ KW QA OM XO",
    "Other Middle East": "IQ IR IL JO LB PS SY YE TR EG",
    "East and South-East Asia": "CN HK MO TW JP KR KP MN TH VN ID MY SG PH KH LA MM BN TP",
    "South Asia": "IN PK BD LK NP BT MV AF",
    "Europe": (
        "AD AL AT BA BE BG BY CH CS CY CZ DE DK EE ES FI FO FR GB GI GR HR HU IE IS IT KV LI LT LU "
        "LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SJ SK SM UA VA"
    ),
    "Americas": (
        "US CA MX AG AI AN AR AW BB BL BM BO BR BS BZ CL CO CR CU CW DM DO EC FK GD GF GL GP GS GT GY "
        "HN HT JM KN KY LC MQ MS NI PA PE PM PR PY SR SV TC TT UM UY VC VE VG VI"
    ),
    "Africa": (
        "AO BF BI BJ BW CD CF CG CI CM CV DJ DZ ER ET GA GH GM GN GQ KE KM LR LS LY MA MG ML MR MU MW "
        "MZ NA NE NG RE RW SC SD SH SL SN SO SS ST SZ TD TG TN TZ UG YT ZA ZM ZW"
    ),
    "Oceania": "AS AU CC CK CX FJ FM GU KI MH MP NC NF NR NU NZ PF PG PN PW SB TA TK TO VU WF WS",
}
OTHER_REGION = "Other and not specified"
REGIONS = [*REGION_CODES, OTHER_REGION]
COUNTRY_REGION = {code: region for region, codes in REGION_CODES.items() for code in codes.split()}


def region_of(iso2: str) -> str:
    """'SA' -> 'GCC'; unknown or unlisted codes -> 'Other and not specified'."""
    return COUNTRY_REGION.get(str(iso2), OTHER_REGION)


def add_region(df: pd.DataFrame, code_col: str = "iso2") -> pd.DataFrame:
    """Add a categorical `region` column (mapped once per distinct code)."""
    codes = df[code_col].astype(object)
    lookup = {code: region_of(code) for code in codes.dropna().unique()}
    df["region"] = pd.Categorical(codes.map(lookup).fillna(OTHER_REGION), categories=REGIONS)
    return df


# ---------------------------------------------------------------------------
# 2026 periods, by month (dates from data/reference/events.csv). The war began on
# 28 February 2026. Months after July (when the portal publishes them) are grouped
# automatically: the Strait of Hormuz was closed again from 11 July.
# ---------------------------------------------------------------------------
PRE_WAR_2026 = {"Jan-Feb": [1, 2]}
WAR_PERIODS_2026 = {"March": [3], "Apr-May": [4, 5], "Jun-Jul": [6, 7]}
PERIOD_EVENTS_2026 = {
    "Jan-Feb": "Before the war (it began on 28 February)",
    "March": "The shock: Strait of Hormuz and Bahrain's airspace closed, Mina Salman Port struck",
    "Apr-May": "Airspace reopened on 8 April, but Hormuz stayed closed and a US naval blockade began",
    "Jun-Jul": "A brief, partial reopening of Hormuz from 18 June, then closed again on 11 July",
    "later": "Hormuz closed again from 11 July (still closed in October)",
}


def war_periods_2026(last_month: int) -> dict[str, list[int]]:
    """The war-time periods up to `last_month` (1-12), adding one for any months after July."""
    periods = {label: [m for m in months if m <= last_month]
               for label, months in WAR_PERIODS_2026.items() if months[0] <= last_month}
    if last_month >= 8:
        label = "Aug" if last_month == 8 else f"Aug-{pd.Timestamp(2026, last_month, 1):%b}"
        periods[label] = list(range(8, last_month + 1))
    return periods


def period_event(label: str) -> str:
    return PERIOD_EVENTS_2026.get(label, PERIOD_EVENTS_2026["later"])


# ---------------------------------------------------------------------------
# Monthly calendar features.
# Ramadan and Eid dates come from the Umm al-Qura calendar (hijridate); Bahrain's
# announced dates can differ by a day, which doesn't matter at monthly level.
# COVID dates come from data/reference/events.csv.
# ---------------------------------------------------------------------------
COVID_RESTRICTIONS = ("2020-03", "2022-03")                 # restrictions in force (system suspended 28 Mar 2022)
COVID_CLOSURE_DAYS = {"2021-05": 5, "2021-06": 9}           # 27 May - 9 June 2021 closures of malls, restaurants etc.


def build_calendar(start: str = "2019-01", end: str = "2027-12") -> pd.DataFrame:
    """One row per month with Ramadan days, Eid flags and COVID flags."""
    ramadan_days: Counter = Counter()
    eid_fitr, eid_adha = set(), set()
    first, last = pd.Period(start, "M"), pd.Period(end, "M")
    first_hijri = Gregorian(first.year, 1, 1).to_hijri().year
    last_hijri = Gregorian(last.year, 12, 31).to_hijri().year
    for hijri_year in range(first_hijri - 1, last_hijri + 1):     # one extra year so a Ramadan spanning Jan is caught
        ramadan_start = Hijri(hijri_year, 9, 1).to_gregorian()
        fitr = Hijri(hijri_year, 10, 1).to_gregorian()           # Eid al-Fitr = 1 Shawwal, day after Ramadan
        adha = Hijri(hijri_year, 12, 10).to_gregorian()          # Eid al-Adha = 10 Dhu al-Hijjah
        day = ramadan_start
        while day < fitr:
            ramadan_days[(day.year, day.month)] += 1
            day += timedelta(days=1)
        eid_fitr.add((fitr.year, fitr.month))
        eid_adha.add((adha.year, adha.month))

    months = pd.period_range(first, last, freq="M")
    covid_lo, covid_hi = (pd.Period(p, "M") for p in COVID_RESTRICTIONS)
    return pd.DataFrame({
        "date": months.to_timestamp(),
        "year": months.year,
        "month": months.month,
        "days_in_month": months.days_in_month,
        "ramadan_days": [ramadan_days[(p.year, p.month)] for p in months],
        "eid_al_fitr": [int((p.year, p.month) in eid_fitr) for p in months],
        "eid_al_adha": [int((p.year, p.month) in eid_adha) for p in months],
        "covid_restrictions": [int(covid_lo <= p <= covid_hi) for p in months],
        "covid_closure_days": [COVID_CLOSURE_DAYS.get(str(p), 0) for p in months],
    })
