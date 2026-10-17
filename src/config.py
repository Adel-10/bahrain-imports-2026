"""Project settings: where files live, which portal datasets are used, and reference figures.

Every other script imports its paths and dataset ids from here, so they are defined once.

Datasets
--------
All data comes from the Bahrain Open Data Portal (data.gov.bh), which runs on Opendatasoft and
offers each dataset as a CSV export through a public API. Two kinds of dataset are used:

* **Imports** (one file per year, 2021-2026): every row is one 8-digit customs code from one
  origin country in one month, with value in Bahraini dinars (BD) and US dollars, weight in kg
  and quantity. These are the official non-oil import statistics.
* **Routes** (port and airport): monthly national totals for Khalifa Bin Salman Port
  (containers, ship visits, vehicles unloaded from car carriers, general cargo tonnage) and for
  Bahrain International Airport (air cargo tonnage). They say nothing about individual products,
  but they show which routes were working.

Reference figures
-----------------
``check_downloads.py`` compares each downloaded file against totals that were read directly
from the portal's API when the project was set up (October 2026). If a download is truncated
or the portal changes its format, those checks fail loudly instead of the analysis quietly
running on bad data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the project folder (this file is in src/)
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

PORTAL = "https://www.data.gov.bh"
EXPORT_URL = PORTAL + "/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/csv"   # full dataset as one CSV
META_URL = PORTAL + "/api/explore/v2.1/catalog/datasets/{dataset_id}"                 # title, last-modified date


@dataclass(frozen=True)
class Dataset:
    name: str           # short name used in this project; the file is saved as data/raw/<name>.csv
    dataset_id: str     # the portal's id for the dataset
    kind: str           # "imports" (product x country x month) or "route" (national monthly totals)
    year: int | None    # imports are published as one file per year
    rows_seen: int      # row count on the portal when checked
    checked: str = "2026-10-11"   # the date rows_seen was checked


DATASETS: list[Dataset] = [
    Dataset("imports_2021", "05-import-2021", "imports", 2021, 314_055),
    Dataset("imports_2022", "04-import-2022", "imports", 2022, 320_747),
    Dataset("imports_2023", "01-import-non-oil-classified-by-commodity-and-country-for-2023", "imports", 2023, 321_376),
    Dataset("imports_2024", "import-2024", "imports", 2024, 322_062),
    Dataset("imports_2025", "import-1-2025", "imports", 2025, 324_818),
    Dataset("imports_2026", "import-2026", "imports", 2026, 155_241),
    Dataset("port_containers", "01-container-statistics-at-khalifa-bin-salman-port", "route", None, 460),
    Dataset("port_vessels", "02-vessel-traffic-statistics-at-khalifa-bin-salman-port", "route", None, 368),
    Dataset("airport_cargo", "02-caa-monthly-air-cargo-movement-at-bahrain-international-airport", "route", None, 144),
    Dataset("port_roro", "03-roro-units-statistics-at-khalifa-bin-salman-port", "route", None, 93, "2026-10-13"),
    Dataset("port_general_cargo", "04-general-cargo-volume-statistics-at-khalifa-bin-salman-port", "route", None, 93,
            "2026-10-13"),
]

DATASETS_BY_NAME = {d.name: d for d in DATASETS}


# ---------------------------------------------------------------------------
# Reference figures for check_downloads.py
#
# Monthly import totals read from the portal's own API on 2026-10-11, as
# (value in BD millions, weight in thousand tonnes), both rounded. If a downloaded
# file adds up to the same numbers month by month, it arrived complete.
# ---------------------------------------------------------------------------
REFERENCE_MONTHLY_IMPORTS: dict[int, list[tuple[int, int]]] = {
    2021: [(409, 2223), (429, 2732), (474, 2009), (457, 2704), (402, 2337), (389, 1932),
           (418, 1965), (496, 2420), (435, 1940), (452, 2208), (453, 1867), (500, 2082)],
    2022: [(449, 1940), (469, 2420), (445, 2037), (505, 2491), (518, 2075), (576, 2634),
           (468, 2497), (520, 2128), (483, 2200), (500, 2140), (440, 2182), (470, 1947)],
    2023: [(465, 2412), (435, 1785), (531, 2442), (523, 3079), (475, 1575), (472, 2524),
           (441, 1719), (534, 2867), (427, 1598), (561, 3309), (454, 1561), (461, 2124)],
    2024: [(535, 1745), (475, 2018), (487, 2173), (481, 2790), (466, 2382), (464, 2296),
           (464, 2035), (494, 2241), (516, 2385), (463, 1578), (478, 2149), (551, 2156)],
    2025: [(496, 1788), (515, 2071), (515, 2504), (575, 2742), (498, 1523), (457, 2253),
           (544, 2972), (489, 3109), (468, 1561), (520, 2562), (538, 2172), (570, 2296)],
    2026: [(528, 1514), (523, 1984), (221, 1215), (288, 779), (312, 842), (372, 1108),
           (438, 1168)],  # January-July: the months published as of 2026-10-11
}

# Spot checks for the route files: (dataset, series name in the file, year, months, expected total).
REFERENCE_ROUTE_POINTS: list[tuple[str, str, int, tuple[int, ...], float]] = [
    ("port_containers", "Imports Containers (TEUs)", 2025, (7,), 20_510),
    ("port_containers", "Imports Containers (TEUs)", 2026, (3,), 6_147),
    ("port_vessels", "Container Vessel Traffic", 2025, (3, 4, 5, 6, 7), 149),
    ("port_vessels", "Total Vessel Visit", 2026, (3, 4, 5, 6, 7), 191),
    ("airport_cargo", "Import Cargo & Mail (In Tonnes)", 2025, (12,), 14_187),
    ("airport_cargo", "Import Cargo & Mail (In Tonnes)", 2026, (3,), 129),
    ("port_roro", "roro_units_vehicles", 2025, (1,), 5_365),
    ("port_roro", "roro_units_vehicles", 2026, (3,), 609),
    ("port_general_cargo", "general_cargo_volume_ft", 2025, (2,), 105_028),
    ("port_general_cargo", "general_cargo_volume_ft", 2026, (6,), 112_609),
]
