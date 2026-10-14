"""Tests for loaders.py, using real rows copied from each yearly file's format.

Run with:  pytest
"""

from __future__ import annotations

import pandas as pd
import pytest

from loaders import IMPORT_COLUMNS, load_imports, load_route, parse_month

# 2021-2022 format: row number, "January", English + Arabic text, quoted commas.
CSV_2021 = (
    "n,year,month,lshhr,commodity_no,commodity,lsl,un_code,country_name,ldwl,"
    "import_value_bd,import_value_usa,import_weight_kg,import_quantity,um\r\n"
    "1,2021,January,يناير,01012910,HORSES FOR SPORT,خيول للرياضة,AE,UNITED ARAB EMIRATES,"
    "الامارات العربية المتحدة,6191.212399999999,16465.990425531912,4000.0,11.0,NO\r\n"
    '15,2021,January,يناير,01051100,"FOWLS OF GALLUS DOMESTICUS, NOT MORE THAN 185 GRAMS",'
    '"ديوك ودجاجات من جنس جالوس دومستكوس بوزن لا يزيد عن 185 غرام                ,",SA,SAUDI ARABIA,'
    "المملكة العربية السعودية,27402.818945,72879.83761968085,2437.0,68500.0,NO\r\n"
    "99,2021,June,يونيو,08045010,MANGOES,مانجو,NA,NAMIBIA,ناميبيا,100.0,265.9,50.0,50.0,KG\r\n"
)

# 2023 format: no row number, "01 January", a field with doubled quotes inside.
CSV_2023 = (
    "year,month,lshhr,commodity_no,commodity,lsl,un_code,country_name,ldwl,"
    "import_value_bd,import_value_usa,import_weight_kg,import_quantity,um\r\n"
    "2023,01 January,01 يناير,48062000,GREASEPROOF PAPERS,ورق زبدة ( ورق كتيم للشحوم ),AE,"
    "UNITED ARAB EMIRATES,الامارات العربية المتحدة,3012.1409999999996,8011.0132978723395,2741.5,2741.5,KG\r\n"
    '2023,01 January,01 يناير,48101300,"PAPER AND PAPERBOARD IN ROLS OF A KIND USED FOR WRITING, PRINTING",'
    '"ورق وورق مقوى لفات \"\"جرافيك\"\" غير محتوي",SA,SAUDI ARABIA,المملكة العربية السعودية,'
    "10472.821999999998,27853.249999999996,5920.0,5920.0,KG\r\n"
)

# 2024 format: no year column, "01-January", transliterated Arabic column names.
CSV_2024 = (
    "n,month,lshhr,commodity_no,commodity,lsl,un_code,country_name,ldwl,"
    "qym_lwrdt_dynr_bhryny,qym_lwrdt_dwlr_mryky,wzn_lwrdt_kjm,kmy_lwrdt,whd_lqys\r\n"
    "1,01-January,01 يناير,01012910,HORSES FOR SPORT,خيول للرياضة,NL,NETHERLANDS,هولندا,"
    "1483.37,3945.14,400.0,2.0,NO\r\n"
    "5,12-December,12 ديسمبر,01022900,OTHER BOVINE,غيرها من ابقار,OM,SULTANATE OMAN,سلطنة عمان,"
    "289893.48,770993.29,206600.0,1053.0,NO\r\n"
)

CSV_CONTAINERS = (
    "n,year,month,lshhr,container_type,nw_lhwy,teus\r\n"
    "1.0,2019,01 January,01 يناير,Imports Containers (TEUs),الحاويات الواردة (حاوية نمطية),17131.0\r\n"
    "2.0,2019,02 February,02 فبراير,Imports Containers (TEUs),الحاويات الواردة (حاوية نمطية),15827.0\r\n"
)

CSV_AIR = (
    "n,year,month,lshhr,data_type,nw_lm_lwm,value\r\n"
    "1.0,2023,01 January,01 يناير,Import Cargo & Mail (In Tonnes),استيراد البضائع والبريد (بالأطنان),10323.0\r\n"
    "2.0,2026,09 September,09 سبتمبر,Import Cargo & Mail (In Tonnes),استيراد البضائع والبريد (بالأطنان),\r\n"
)

# Single-series file: no series column, so the number column's name becomes the series name.
CSV_GENERAL_CARGO = (
    "n,year,month,lshhr,general_cargo_volume_ft\r\n"
    "86.0,2026,01 January,01 يناير,40487.0\r\n"
    "87.0,2026,02 February,02 فبراير,56.186\r\n"
)


def write(tmp_path, name: str, text: str, bom: bool = False):
    path = tmp_path / name
    path.write_bytes(("﻿" if bom else "").encode("utf-8") + text.encode("utf-8"))
    return path


@pytest.mark.parametrize("label, expected", [
    ("January", 1), ("01 January", 1), ("01-January", 1), ("12 December", 12), (" september ", 9),
])
def test_parse_month(label, expected):
    assert parse_month(label) == expected


def test_parse_month_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_month("Thirteenember")


def test_2021_format(tmp_path):
    df = load_imports(write(tmp_path, "i2021.csv", CSV_2021, bom=True), 2021)
    assert list(df.columns) == IMPORT_COLUMNS
    assert len(df) == 3
    assert df["hs8"].iloc[0] == "01012910"          # leading zero kept
    assert df["commodity_en"].iloc[1] == "FOWLS OF GALLUS DOMESTICUS, NOT MORE THAN 185 GRAMS"
    assert df["month"].tolist() == [1, 1, 6]
    assert df["value_bd"].iloc[0] == pytest.approx(6191.2124)
    assert df["iso2"].iloc[2] == "NA"               # Namibia, not a missing value
    assert df["iso2"].isna().sum() == 0


def test_2023_format(tmp_path):
    df = load_imports(write(tmp_path, "i2023.csv", CSV_2023), 2023)
    assert len(df) == 2
    assert (df["year"] == 2023).all()
    assert df["weight_kg"].sum() == pytest.approx(2741.5 + 5920.0)


def test_2024_format(tmp_path):
    df = load_imports(write(tmp_path, "i2024.csv", CSV_2024), 2024)
    assert list(df.columns) == IMPORT_COLUMNS
    assert (df["year"] == 2024).all()               # added: the file has no year column
    assert df["month"].tolist() == [1, 12]
    assert df["value_bd"].tolist() == pytest.approx([1483.37, 289893.48])
    assert df["unit"].tolist() == ["NO", "NO"]


def test_wrong_year_is_caught(tmp_path):
    with pytest.raises(ValueError, match="expected year 2022"):
        load_imports(write(tmp_path, "i.csv", CSV_2021), 2022)


def test_route_files(tmp_path):
    containers = load_route(write(tmp_path, "c.csv", CSV_CONTAINERS), "port_containers")
    assert containers["value"].tolist() == [17131.0, 15827.0]
    assert containers["date"].tolist() == [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01")]

    air = load_route(write(tmp_path, "a.csv", CSV_AIR), "airport_cargo")
    assert air.loc[0, "series"] == "Import Cargo & Mail (In Tonnes)"
    assert air.loc[0, "value"] == 10323.0
    assert len(air) == 1                            # blank placeholder month dropped


def test_single_series_route_file(tmp_path):
    cargo = load_route(write(tmp_path, "g.csv", CSV_GENERAL_CARGO), "port_general_cargo")
    assert cargo["series"].unique().tolist() == ["general_cargo_volume_ft"]
    assert cargo["value"].tolist() == [40487.0, 56.186]          # read as published; step 2 repairs it
