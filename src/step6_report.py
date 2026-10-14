"""Charts and the written report for step 6 (called by routes.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import glossary
from build_dataset import md_table
from explore import (AQUA, BASELINE, BLUE, GRID, INK, INK2, MUTED, ORANGE, SHADE, SURFACE, add_axes, dot_handle,
                     legend, line_handle, new_figure, rounded_bar, save, source_note, tidy)
from reference import PERIOD_EVENTS_2026, period_event
from routes import (BEYOND, GCC, HEAVY, INDICATORS, LIGHT, LIGHT_FROM, HEAVY_BELOW, MIDDLE, ORIGIN_LEVEL, OTHER_ME,
                    REPORT_PATH, VEHICLES, WEIGHT_LEVEL)
from step5_report import MIN_SHARE, RED, SKIP_SECTIONS, and_list, grid_chart, pct

PORT_AIR_SOURCE = ("Port: Khalifa Bin Salman Port. Air: Bahrain International Airport. "
                   "Source: Bahrain Open Data Portal (iGA). Analysis: own.")
SMALL_SECTIONS = {"Arms and ammunition", "Art and antiques"}
SHOW_ROUTE_ORDER = [GCC, OTHER_ME, BEYOND]
WEIGHT_ROWS = [(LIGHT, "Light goods"), (MIDDLE, "Middle"), (HEAVY, "Heavy goods"), (VEHICLES, "Vehicles and transport")]
ORIGIN_ROWS = {GCC: "From GCC countries", OTHER_ME: "From the rest of the Middle East", BEYOND: "From beyond the Middle East"}
REPAIRED = "#ffffff"


def k_fmt(v: float, _=None) -> str:
    return f"{v / 1000:,.0f}k" if abs(v) >= 1000 else f"{v:,.0f}"


def short_period(label: str) -> str:
    """'Since the war (Mar-Jul)' -> 'Mar-Jul'."""
    return label.split("(")[1].rstrip(")") if label.startswith("Since") else label


def share_words(p: float) -> str:
    """66.4 -> 'two-thirds'; numbers not close to a simple fraction stay as percentages."""
    for value, words in ((25, "a quarter"), (100 / 3, "a third"), (50, "half"), (200 / 3, "two-thirds"),
                         (75, "three-quarters")):
        if abs(p - value) <= 3:
            return words
    return f"{p:.0f}%"


def region_text(label: str) -> str:
    """Region names as they read in a sentence."""
    return {"Other Middle East": "the rest of the Middle East", "Americas": "the Americas"}.get(label, label)


def times(x: float) -> str:
    """292 (% of normal) -> 'about 3 times'."""
    return f"about {x / 100:.0f} times" if x >= 150 else f"{x:.0f}% of"


def pon(x: float) -> str:
    """% of normal, as text."""
    return "-" if pd.isna(x) else f"{x:.0f}%"


# ---------------------------------------------------------------------------
# Chart 1: port and airport, 2026 against 2025
# ---------------------------------------------------------------------------
def fig_indicators(route: pd.DataFrame, last_route: pd.Timestamp) -> None:
    r = route.set_index("date")
    repaired = r["port_general_cargo_note"].str.startswith("decimal")
    missing = r["port_general_cargo_note"].str.startswith("recorded as 0")
    fig = new_figure(10, 8.7, "Containers fell to a third of normal and car carriers stopped; general cargo rose",
                     "Each month of 2026 (blue) against the same month of 2025 (grey), at Bahrain's port and airport. "
                     "The shaded months are after the war began.")
    legend(fig, [line_handle(BLUE), line_handle(BASELINE),
                 plt.Line2D([], [], ls="", marker="o", ms=7, mfc=REPAIRED, mec=BLUE, mew=1.6)],
           ["2026", "2025", "Figure repaired (decimal point typed for a thousands separator)"], top=1.05, ncol=3)
    w, h = fig.get_size_inches()
    cols, rows = 2, 3
    left, right, top, bottom, hgap, vgap = 0.75, 0.35, 1.85, 0.7, 0.75, 0.75
    pw = (w - left - right - hgap) / cols
    ph = (h - top - bottom - (rows - 1) * vgap) / rows
    for i, (col, name, unit) in enumerate(INDICATORS):
        rr, cc = divmod(i, cols)
        x0 = left + cc * (pw + hgap)
        y0 = h - top - (rr + 1) * ph - rr * vgap
        ax = fig.add_axes([x0 / w, y0 / h, pw / w, ph / h])
        tidy(ax)
        s25 = r.loc[r.index.year == 2025, col]
        s26 = r.loc[(r.index.year == 2026) & (r.index <= last_route), col]
        ax.axvspan(2.5, 12.5, color=SHADE, lw=0, zorder=0)
        ax.plot(s25.index.month, s25.values, color=BASELINE, lw=2, zorder=2)
        ax.plot(s26.index.month, s26.values, color=BLUE, lw=2, zorder=3)
        top_val = np.nanmax([s25.max(), s26.max()])
        ax.set_ylim(0, top_val * 1.15)
        fixed = s26[repaired.reindex(s26.index, fill_value=False)] if col == "port_general_cargo_ft" else s26.iloc[:0]
        valid = s26.dropna()
        if len(valid) and valid.index[-1] not in fixed.index:
            ax.plot([valid.index[-1].month], [valid.iloc[-1]], "o", ms=7, color=BLUE, mec=SURFACE, mew=2, zorder=5)
        if col == "port_general_cargo_ft":
            ax.plot(fixed.index.month, fixed.values, "o", ms=6, mfc=REPAIRED, mec=BLUE, mew=1.6, zorder=6)
            for d in s26.index[missing.reindex(s26.index, fill_value=False)]:
                ax.text(d.month, top_val * 1.15 * 0.03, "not\nrecorded", ha="center", va="bottom", fontsize=7.5,
                        color=MUTED, linespacing=1.1, zorder=7)
        ax.set_xlim(0.6, 12.4)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8.5)
        ax.yaxis.set_major_formatter(FuncFormatter(k_fmt))
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(f"{name} ({unit})", loc="left", fontsize=10, color=INK, pad=6, fontweight="semibold")
    source_note(fig, PORT_AIR_SOURCE)
    save(fig, "step6_1_routes.png")


# ---------------------------------------------------------------------------
# Chart 2: route groups (grid)
# ---------------------------------------------------------------------------
def route_group_rows(gaps: pd.DataFrame, nat: pd.DataFrame, columns: dict[str, str]) -> list:
    def cells(df):
        return {short: (df.loc[p, "gap_pct"], bool(df.loc[p, "outside"])) for p, short in columns.items()}
    rows = [("**All of Bahrain**", cells(nat))]
    for name in SHOW_ROUTE_ORDER:
        g = gaps[(gaps["level"] == ORIGIN_LEVEL) & (gaps["name"] == name)].set_index("period")
        rows.append((ORIGIN_ROWS[name], cells(g)))
    for name, shown in WEIGHT_ROWS:
        g = gaps[(gaps["level"] == WEIGHT_LEVEL) & (gaps["name"] == name)].set_index("period")
        rows.append((f"of which {shown[0].lower()}{shown[1:]}", cells(g)))
    return rows


# ---------------------------------------------------------------------------
# Chart 3: elimination
# ---------------------------------------------------------------------------
def fig_elimination(elim: pd.DataFrame) -> None:
    n = len(elim)
    fig = new_figure(10, 1.95 + 0.62 * n + 0.75,
                     "At least half of what arrived came by road or general cargo ship",
                     "Core goods arriving, by weight. The most that container ships and planes could have carried "
                     "(blue), and so the least that must have\ncome another way (orange). Bars are labelled with the "
                     "share of what arrived in each period.")
    legend(fig, [Rectangle((0, 0), 1, 1, color=BLUE, lw=0), Rectangle((0, 0), 1, 1, color=ORANGE, lw=0)],
           ["Container ships and planes: at most", "Road (King Fahd Causeway) or general cargo ship: at least"],
           top=1.3, ncol=2)
    ax = add_axes(fig, left=1.75, right=0.45, top=1.85, bottom=0.75)
    ax.set_xlim(0, 100)
    ax.set_ylim(n - 0.5, -0.5)
    tidy(ax, grid_axis="x")
    fig.canvas.draw()
    gap = 2 * 100 / (ax.get_window_extent().width)        # a 2px surface gap between the two parts
    for i, row in enumerate(elim.itertuples()):
        a = row.containers_air_max_share
        ax.add_patch(Rectangle((0, i - 0.3), a - gap / 2, 0.6, color=BLUE, lw=0, zorder=3))
        rounded_bar(ax, a + gap / 2, 100 - a - gap / 2, i, 0.6, ORANGE)
        ax.text(a / 2, i, f"{a:.0f}%", ha="center", va="center", fontsize=10, color="#ffffff", fontweight="semibold",
                zorder=4)
        ax.text(a + (100 - a) / 2, i, f"{row.other_min_share:.0f}%  ({row.other_min_t / 1000:,.0f} kt)", ha="center",
                va="center", fontsize=10, color="#ffffff", fontweight="semibold", zorder=4)
    ax.set_yticks(range(n))
    labels = [short_period(p) for p in elim["period"]]
    ax.set_yticklabels([f"{l}\ntogether" if l.startswith("Mar-") else l for l in labels], color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    source_note(fig, "Assumes a container held no more than usual. Customs import data, port and airport figures: "
                     "Bahrain Open Data Portal (iGA). Analysis: own.")
    save(fig, "step6_3_elimination.png")


# ---------------------------------------------------------------------------
# Chart 4: adapting (dumbbells)
# ---------------------------------------------------------------------------
def status_colour(row) -> str:
    if row["latest_inside_range"]:
        return AQUA
    return BLUE if row["latest"] > 100 else RED


def fig_adapting(panels: list[tuple[str, pd.DataFrame]], latest: str) -> None:
    n_total = sum(len(df) for _, df in panels)
    row_h, head = 0.33, 0.55
    height = 1.95 + n_total * row_h + head * len(panels) + 0.35 * (len(panels) - 1) + 0.95
    fig = new_figure(10, height, "Most supply chains had come a long way back; vehicles and clothing hadn't",
                     f"% of normal in the worst war-time period and in {latest}. The grey band is the normal "
                     f"range for {latest}: a dot inside it\ncould be ordinary ups and downs.")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=MUTED, mew=1.8),
               dot_handle(RED), dot_handle(AQUA), dot_handle(BLUE), Rectangle((0, 0), 1, 1, color=GRID, lw=0)]
    legend(fig, handles, ["Worst point", f"{latest}: still below normal", "back inside the normal range",
                          "above normal", "Normal range"], top=1.3, ncol=5)
    w, h = fig.get_size_inches()
    y_top = h - 1.85
    x_lo, x_hi = 0, 140
    for title, df in panels:
        n = len(df)
        ax = fig.add_axes([3.0 / w, (y_top - head - n * row_h) / h, (w - 3.45) / w, n * row_h / h])
        fig.text(0.6 / w, (y_top - 0.2) / h, title, fontsize=10.5, fontweight="semibold", color=INK, va="top")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(n - 0.5, -0.5)
        tidy(ax, grid_axis="x")
        ax.axvline(100, color=BASELINE, lw=1, zorder=1)
        for i, (_, row) in enumerate(df.iterrows()):
            rng = row["latest_range_pct"]
            lo, hi = max(100 - rng, x_lo), min(100 + rng, x_hi)
            ax.add_patch(Rectangle((lo, i - 0.32), hi - lo, 0.64, color=GRID, lw=0, zorder=1))
            worst, now = row["worst"], min(row["latest"], x_hi)
            ax.plot([worst, now], [i, i], color=BASELINE, lw=2, zorder=2, solid_capstyle="butt")
            ax.plot([worst], [i], "o", ms=8, mfc=SURFACE, mec=MUTED, mew=1.8, zorder=3)
            ax.plot([now], [i], "o", ms=8, color=status_colour(row), mec=SURFACE, mew=2, zorder=4)
            if row["latest"] > x_hi:
                ax.text(x_hi - 1, i - 0.38, f"{row['latest']:.0f}%", ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_yticks(range(n))
        ax.set_yticklabels(df["shown"], color=INK, fontsize=9.5)
        for lab, bold in zip(ax.get_yticklabels(), df["bold"]):
            if bold:
                lab.set_fontweight("semibold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(axis="x", labelsize=8.5)
        y_top -= head + n * row_h + 0.35
    source_note(fig)
    save(fig, "step6_4_adapting.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write(results: dict) -> None:
    route, gaps, elim, adapt = results["route"], results["gaps"], results["elimination"], results["adapting"]
    ind, periods, since = results["indicators"], results["periods"], results["since"]
    war_periods, last_month = results["war_periods"], results["last_month"]
    since_label = list(since)[0]
    since_short = short_period(since_label)
    latest = war_periods[-1]
    r_periods = list(results["route_periods"])
    last_route = route.set_index("date")[[c for c, _, _ in INDICATORS]].dropna(how="all").index.max()

    # ---- numbers used in the text ------------------------------------------
    def ind_pon(name, period):
        row = ind[(ind["indicator"] == name) & (ind["period"] == period)]
        return row["pct_of_normal"].iloc[0] if len(row) else np.nan

    def g(level, name, period, col="gap_pct"):
        return gaps[(gaps["level"] == level) & (gaps["name"] == name) & (gaps["period"] == period)][col].iloc[0]

    nat_gaps = results["nat_gaps"].set_index("period")
    ad = adapt.set_index(["level", "label"])
    nat = ad.loc[("National total", "Bahrain")]
    eng = results["engines"].set_index("period")
    cs = results["class_share"]
    el = elim.set_index("period")
    el_all = el.loc[since_label]
    cars = results["cars"]
    after = [p for p in r_periods if p not in periods and p not in ("Jan-Feb",)]   # periods only the port data reach
    after_label = after[-1] if after else None
    cont, air, roro, gc = ("Containers unloaded", "Air cargo and mail unloaded", "Vehicles unloaded from car carriers",
                           "General cargo handled")
    war_cont = [ind_pon(cont, p) for p in war_periods]
    r = route.set_index("date")
    gc_2025_avg = r.loc[r.index.year == 2025, "port_general_cargo_ft"].mean()
    gc_after_avg = (r.loc[(r.index.year == 2026) & r.index.month.isin(results["route_periods"][after_label]),
                          "port_general_cargo_ft"].mean() if after_label else np.nan)
    gc_months = r.loc[r.index.year == 2025, "port_general_cargo_ft"]
    first_roro_zero = r.index[(r.index.year == 2026) & (r["port_roro_vehicles"] == 0)].min()

    # ---- charts --------------------------------------------------------------
    fig_indicators(route, last_route)

    grid_cols = {p: short_period(p) for p in [*periods, since_label]}
    grid_chart(route_group_rows(gaps, nat_gaps, grid_cols), list(grid_cols.values()),
               "Goods that could use open routes held up; heavy goods from far away fell furthest",
               "Gap between actual imports and the forecast without the war, % of the forecast. Coloured only where the "
               "gap is outside the normal range\n(a real change); grey where it could be normal ups and downs. Light, "
               "middle and heavy goods are all from beyond the Middle East.",
               "step6_2_route_groups.png", first_col_width=3.1, bold_columns={since_short}, cell_width=1.28)
    fig_elimination(elim)

    regions = adapt[(adapt["level"] == "Origin region") & (adapt["label"] != "Other and not specified")].copy()
    regions["shown"], regions["bold"] = regions["label"], False
    nat_row = adapt[adapt["level"] == "National total"].assign(shown="All of Bahrain", bold=True)
    regions = pd.concat([nat_row, regions.sort_values("latest")])
    nat_fc = nat["forecast"]
    secs = adapt[(adapt["level"] == "Section") & (adapt["forecast"] / nat_fc * 100 >= MIN_SHARE)
                 & ~adapt["label"].isin(SKIP_SECTIONS | SMALL_SECTIONS)].copy()
    secs["shown"], secs["bold"] = secs["label"], False
    secs = secs.sort_values("latest")
    fig_adapting([("By origin region", regions), ("By product section", secs)], latest)

    # ---- tables --------------------------------------------------------------
    route_rows = pd.DataFrame({
        "Route": ["**Container ships**", "**Car carriers (RoRo)**", "**General cargo ships**", "**Planes**",
                  "**Trucks over the King Fahd Causeway**"],
        "Where": ["Khalifa Bin Salman Port", "Khalifa Bin Salman Port", "Khalifa Bin Salman Port",
                  "Bahrain International Airport", "From Saudi Arabia"],
        "What it carries": ["Most manufactured goods from Asia, Europe and the Americas",
                            "New cars and other vehicles", "Steel, timber, machinery, bagged food, livestock",
                            "Light, valuable or urgent goods", "Goods from GCC countries, and goods landed at their ports"],
        "Monthly figures?": ["Yes: containers unloaded (TEU), ship visits", "Yes: vehicles unloaded",
                             "Yes: freight tonnes handled, ship visits", "Yes: tonnes unloaded", "**No**"],
    })
    periods_md = md_table(pd.DataFrame({
        "Period": [f"**{p}**" for p in r_periods],
        "What was happening": [period_event(p) for p in r_periods],
        "Import data?": ["yes" if p in periods else "not yet (port and airport figures only)" for p in r_periods],
    }))

    ind_names = [name for _, name, _ in INDICATORS] + ["Core goods imported, by weight"]
    ind_md = md_table(pd.DataFrame(
        {"Indicator": ind_names,
         **{p: [pon(ind_pon(n, p)) for n in ind_names] for p in r_periods}}))

    origin_md = md_table(pd.DataFrame({
        "Goods from": ["GCC countries", "the rest of the Middle East", "beyond the Middle East"],
        "Share of normal imports": [f"{g(ORIGIN_LEVEL, n, since_label, 'forecast') / nat_gaps.loc[since_label, 'forecast'] * 100:.0f}%"
                                    for n in SHOW_ROUTE_ORDER],
        **{short_period(p): [pct(g(ORIGIN_LEVEL, n, p)) for n in SHOW_ROUTE_ORDER] for p in [*war_periods, since_label]},
        f"Normal range ({since_short})": [f"+/-{g(ORIGIN_LEVEL, n, since_label, 'range_pct'):.0f}%" for n in SHOW_ROUTE_ORDER],
    }))
    weight_md = md_table(pd.DataFrame({
        "Goods from beyond the Middle East": [s for _, s in WEIGHT_ROWS],
        "Share of value": [f"{cs.loc[n, 'value_bd']:.0f}%" for n, _ in WEIGHT_ROWS],
        "Share of weight": [f"{cs.loc[n, 'weight_kg']:.0f}%" for n, _ in WEIGHT_ROWS],
        **{short_period(p): [pct(g(WEIGHT_LEVEL, n, p)) for n, _ in WEIGHT_ROWS] for p in [*war_periods, since_label]},
    }))
    engines_md = md_table(pd.DataFrame({
        "Light goods from beyond the Middle East": ["As measured", "With aircraft engines at their normal level"],
        **{short_period(p): [pct(eng.loc[p, "gap_pct"]), pct(eng.loc[p, "gap_pct_without_engines"])]
           for p in [*war_periods, since_label]},
    }))

    elim_md = md_table(pd.DataFrame({
        "": ["Core goods arriving, by weight (kt)", "... as % of the same months of 2025",
             "Containers unloaded, % of the same months of 2025", "Air cargo unloaded, % of the same months of 2025",
             "**Most that containers and planes could have carried**", "**Least that came by road or general cargo ship**",
             "Goods from GCC countries, share of the weight (2025 -> 2026)"],
        **{short_period(p): [f"{row.weight_2026_t / 1000:,.0f}", pon(row.weight_pct_of_2025),
                             pon(row.containers_pct_of_2025), pon(row.air_pct_of_2025),
                             f"**{row.containers_air_max_share:.0f}%**",
                             f"**{row.other_min_share:.0f}% ({row.other_min_t / 1000:,.0f} kt)**",
                             f"{row.gcc_share_2025:.0f}% -> {row.gcc_share_2026:.0f}%"]
           for p, row in el.iterrows()},
    }))

    cars_md = md_table(pd.DataFrame({
        "Month (2026)": [f"{d:%B}" for d in cars.index],
        "Vehicles unloaded from car carriers": [f"{v:,.0f}" for v in cars["roro_vehicles"]],
        "... % of the 2025 average month": [pon(v) for v in cars["roro_pct_of_2025_avg"]],
        "Cars from East Asia and Europe (BD m)": [f"{v / 1e6:,.1f}" for v in cars["cars_bd"]],
        "... % of the 2025 average month ": [pon(v) for v in cars["cars_pct_of_2025_avg"]],
    }))

    def adapt_md(df, label):
        return md_table(pd.DataFrame({
            label: [f"**{s}**" if b else s for s, b in zip(df["shown"], df["bold"])],
            **{p: [pon(v) for v in df[p]] for p in war_periods},
            "Came back": ["above normal" if l > 100 else ("no: fell further" if wp == latest else
                                                           f"{c:.0f}% of the way" if pd.notna(c) else "-")
                          for c, l, wp in zip(df["came_back_pct"], df["latest"], df["worst_period"])],
            f"Inside the normal range in {latest}?": ["yes" if x else "no" for x in df["latest_inside_range"]],
        }))

    regions_md = adapt_md(regions, "Origin")
    sections_md = adapt_md(secs.sort_values("latest"), "Section")

    # ---- sentences that depend on the data --------------------------------
    sec_inside = secs[secs["latest_inside_range"]]
    sec_above = secs[~secs["latest_inside_range"] & (secs["latest"] > 100)]
    still = secs[~secs["latest_inside_range"] & (secs["latest"] < 100)].sort_values("latest")
    origins = regions[regions["level"] == "Origin region"]
    reg_inside = origins[origins["latest_inside_range"]]
    reg_above = origins[~origins["latest_inside_range"] & (origins["latest"] > 100)]
    reg_still = origins[~origins["latest_inside_range"] & (origins["latest"] < 100)]
    gcc_above_text = " (goods from GCC countries were above normal)" if GCC in set(reg_above["label"]) else ""
    above = [f"goods from {l if l != GCC else 'GCC countries'} ({pon(v)})" for l, v in zip(reg_above["label"], reg_above["latest"])]
    above += [f"{l.lower()} ({pon(v)})" for l, v in zip(sec_above["label"], sec_above["latest"])]
    above_text = f" **Above normal:** {and_list(above)}." if above else ""
    gcc_latest = g(ORIGIN_LEVEL, GCC, latest)
    gcc_outside = bool(g(ORIGIN_LEVEL, GCC, latest, "outside"))
    light_am = eng.loc["Apr-May", "gap_pct_without_engines"] if "Apr-May" in eng.index else np.nan
    heavy_am = g(WEIGHT_LEVEL, HEAVY, "Apr-May") if "Apr-May" in periods else np.nan
    cars_from = cars[(cars.index.month >= 5)]
    cars_low, cars_high = cars_from["cars_pct_of_2025_avg"].min(), cars_from["cars_pct_of_2025_avg"].max()
    if "Jun-Jul" in el.index and pd.notna(el.loc["Jun-Jul", "general_cargo_ft"]):
        jj = el.loc["Jun-Jul"]
        gc_sentence = (f"In June-July, for example, at least {jj['other_min_t'] / 1000:,.0f} kt came by road or general "
                       f"cargo ship, and the port handled {jj['general_cargo_ft'] / 1000:,.0f} thousand freight tonnes of "
                       "general cargo. If every one of those were core goods imports, general cargo ships could have "
                       "carried most of it; if few were, the causeway carried almost all of it.")
    else:
        gc_sentence = ""
    beyond_lost = g(ORIGIN_LEVEL, BEYOND, since_label, "lost")
    nat_lost = nat_gaps.loc[since_label, "lost"]
    gcc_lost = g(ORIGIN_LEVEL, GCC, since_label, "lost")
    if beyond_lost >= nat_lost and gcc_lost < 0:
        beyond_lost_sentence = (f"They account for more than all of the money lost: BD {beyond_lost / 1e6:,.0f}m, against "
                                f"BD {nat_lost / 1e6:,.0f}m nationally, because goods from GCC countries arrived "
                                f"BD {-gcc_lost / 1e6:,.0f}m above normal.")
    else:
        beyond_lost_sentence = (f"They account for {beyond_lost / nat_lost * 100:.0f}% of the money lost "
                                f"(BD {beyond_lost / 1e6:,.0f}m of BD {nat_lost / 1e6:,.0f}m).")
    after_cont = ind_pon(cont, after_label) if after_label else np.nan
    after_air = ind_pon(air, after_label) if after_label else np.nan
    last_war_cont = ind_pon(cont, latest)

    text = f"""# Step 6: Routes: how goods got around the blockade, and how far supply chains came back

*Generated by `src/routes.py` (charts and text by `src/step6_report.py`). Re-running the script refreshes every number.*

Step 5 measured how far imports fell below normal. This step looks at **how** goods reach Bahrain, and asks four questions:

1. Which routes were cut, and for how long?
2. Did goods that could use routes that stayed open hold up better?
3. How much must have come by road, a route with no published figures?
4. How far had each supply chain come back, and is it still adapting?

## The short answer

| Question | Answer |
|---|---|
| Which routes were cut? | **Container ships:** containers unloaded at the port fell to about a third of normal from March to July ({and_list([pon(v) for v in war_cont])} in {and_list(war_periods)}). **Car carriers stopped completely** from {first_roro_zero:%B} (still none in {last_route:%B}). **Planes:** air cargo was almost zero in March, when the airspace was closed, and back to {pon(ind_pon(air, latest))} of normal by {latest}. |
| Did goods that could use open routes hold up better? | **Yes.** Goods from **GCC countries**, which can come by road, were {pct(g(ORIGIN_LEVEL, GCC, since_label))} against the forecast from {since_short} (inside the normal range), and {pct(gcc_latest)} in {latest}. Goods from **beyond the Middle East** were {pct(g(ORIGIN_LEVEL, BEYOND, since_label))}. Among those, while Hormuz was shut but the airspace open (April-May), **light goods held up far better than heavy goods**: {pct(light_am)} against {pct(heavy_am)} (leaving out aircraft engines, which are bought irregularly). |
| How much must have come by road? | **At least half.** Container ships and planes could have carried at most {el_all['containers_air_max_share']:.0f}% of core goods by weight from {since_short}. So **at least {el_all['other_min_share']:.0f}% (about {el_all['other_min_t'] / 1000:,.0f} thousand tonnes) came by road over the King Fahd Causeway, or by general cargo ship.** The data can't fully separate those two. |
| How far had supply chains come back? | **About {share_words(nat['came_back_pct'])} of the way** by {latest}: from {pon(nat['worst'])} of normal in {nat['worst_period']} to {pon(nat['latest'])}. {and_list(reg_inside.sort_values('forecast', ascending=False)['label'].head(3).map(region_text))} were back inside their normal range{gcc_above_text}, as were {len(sec_inside)} of {len(secs)} product sections; **vehicles ({pon(ad.loc[('Section', 'Vehicles and transport'), 'latest'])}), East Asia ({pon(ad.loc[('Origin region', 'East and South-East Asia'), 'latest'])}) and the Americas ({pon(ad.loc[('Origin region', 'Americas'), 'latest'])}) were still far below**. |
| Is it still adapting? | {"**The port figures say yes.** They run two months ahead of the import data: even after Hormuz closed again on 11 July, containers unloaded rose from " + pon(last_war_cont) + " of normal in " + latest + " to " + pon(after_cont) + " in " + after_label + ". Car carriers still hadn't returned." if after_label else "The port figures don't yet reach past the import data."} |

## Words used in this report

{glossary.table(6, ["core_goods", "national_total", "origin_region", "section", "series", "route", "causeway", "teu",
                    "car_carrier", "general_cargo", "freight_tonne", "normal", "pct_of_normal", "forecast", "gap",
                    "normal_range", "real_shortfall", "since_war", "beyond_me", "weight_class", "elimination",
                    "came_back", "adapting", "units"])}

---

## Before you start: how this step works

### Step 1: The five routes into Bahrain

Bahrain is an island. Goods arrive in one of five ways:

{md_table(route_rows)}

Everything that reaches Bahrain by sea from outside the Gulf passes through the **Strait of Hormuz**. The road comes
from Saudi Arabia, so it carries goods from GCC countries, and goods from further away that were landed at Saudi, UAE or
Omani ports and driven the rest of the way. **The road is the one route with no monthly figures**, so it has to be
worked out from the others (step 4 below).

### Step 2: Two kinds of "normal"

* **Imports** are compared with step 5's forecast of what would have arrived without the war, with the same normal ranges
  (a gap only counts as a real change when it is bigger than the model's normal misses).
* **Port and airport figures** have no forecast, so each 2026 month is compared with **the same month of 2025**. The
  changes are so large (often more than half) that the choice of comparison doesn't change the picture.

### Step 3: The periods

{periods_md}

The port and airport publish their figures about two months before the import data, so they show what happened in
{and_list(after) if after else "the latest months"} too.

### Step 4: Elimination, in plain words

Imagine 100 tonnes of goods arrived in a month, and the port says it unloaded a third as many containers as usual.

1. **The most the containers could have carried.** Pretend that last year *everything* came in containers. That's
   more than is true, so it gives the most containers could carry now: a third of last year's goods.
2. **Add everything that came by plane** (as if every tonne of air cargo were core goods, again more than is true).
3. **Whatever is left must have come another way:** by truck over the causeway, or by general cargo ship.

Because each step is deliberately generous to containers and planes, the answer is a **minimum** for the road and
general cargo ships together. It needs one assumption: that a container held no more than usual.

---

## 1. Which routes were cut?

![Routes](figures/step6_1_routes.png)

**What we did:** compared each port and airport figure for 2026 with the same months of 2025, and core goods imports by
weight with the same months of 2025.

**What we found (% of the same months in 2025):**

{ind_md}

*A dash means no figure yet. General cargo jumps around from month to month (in 2025, from {gc_months.min():,.0f} to
{gc_months.max():,.0f} freight tonnes){f", so against the 2025 average month, {after_label} 2026 was {gc_after_avg / gc_2025_avg:.1f} times normal" if after_label else ""}.*

1. **Containers fell by about two-thirds and stayed there.** From March to July the port unloaded about a third of the
   usual containers in every period. Fewer container ships called, and those that did unloaded less.
2. **Car carriers stopped.** Vehicles unloaded from car carriers fell to {pon(ind_pon(roro, 'March'))} of normal in March and to **zero from {first_roro_zero:%B}**, still zero in {last_route:%B}.
3. **Planes came back slowly.** Air cargo was {pon(ind_pon(air, 'March'))} of normal in March (the airspace was closed until 8 April),
   {pon(ind_pon(air, 'Apr-May'))} in April-May and {pon(ind_pon(air, latest))} in {latest}. It never rose above normal.
4. **General cargo went the other way.** Fewer general cargo ships called, but in April-May they handled {times(ind_pon(gc, 'Apr-May'))}
   the usual tonnage, and more after that.
5. **Core goods by weight fell much less than any of these:** {pon(ind_pon('Core goods imported, by weight', 'March'))} of normal in
   March, {pon(ind_pon('Core goods imported, by weight', latest))} in {latest}.

**A data problem fixed along the way.** In the general cargo file, several 2026 months were typed with a decimal point
where a thousands separator belongs (56.186 instead of 56,186). Step 2 now repairs them and marks each one; they are the
open circles in the chart. March 2026 was recorded as 0 although 17 general cargo ships called, so it is treated as
missing.

**Why it matters:** goods kept arriving at {pon(el_all['weight_pct_of_2025'])} of their 2025 weight from
{since_short}, while containers ran at a third. The difference had to come in some other way. Sections 2 and 3 work out how.

---

## 2. Did goods that could use open routes hold up better?

![Route groups](figures/step6_2_route_groups.png)

**What we did:** repeated step 5's comparison (actual imports against the forecast without the war, with the normal
ranges from step 4's tests) for groups of goods that depend on different routes.

### 2a. By where the goods came from

{origin_md}

1. **Goods from GCC countries held up**: {pct(g(ORIGIN_LEVEL, GCC, since_label))} from {since_short}, inside the normal range.
   {"In " + latest + " they were " + pct(gcc_latest) + ", **above** normal (outside the range)." if gcc_outside and gcc_latest > 0 else ""}
2. **Goods from the rest of the Middle East fell, but less**: {pct(g(ORIGIN_LEVEL, OTHER_ME, since_label))}.
3. **Goods from beyond the Middle East fell furthest**: {pct(g(ORIGIN_LEVEL, BEYOND, since_label))}, and
   {pct(g(ORIGIN_LEVEL, BEYOND, 'March'))} in March. {beyond_lost_sentence}

### 2b. Goods from beyond the Middle East: light against heavy

Light goods are worth flying; heavy ones must go by sea or road. If planes helped, light goods should have held up better
once the airspace reopened on 8 April, while Hormuz was still shut.

{weight_md}

One correction matters here. **Aircraft engines** are light and valuable, and one airline's purchases can move the light
group a lot: they almost stopped in March and came back irregularly (step 5 showed the same effect nationally). Holding
them at their normal level:

{engines_md}

1. **In March everything fell**, light goods included: the airspace was closed as well as the sea route.
2. **In April-May light goods held up far better:** {pct(light_am)} (without aircraft engines) against
   **{pct(heavy_am)} for heavy goods.** That is the period when planes could fly but ships couldn't pass Hormuz.
3. **In June-July heavy goods bounced back** to {pct(g(WEIGHT_LEVEL, HEAVY, 'Jun-Jul'))}. This was when Hormuz partly reopened
   (18 June to 11 July), and when general cargo, the usual way to ship heavy goods such as steel, timber and bagged rice,
   was {times(ind_pon(gc, 'Jun-Jul'))} its usual tonnage.
4. **Vehicles stayed deepest**: {pct(g(WEIGHT_LEVEL, VEHICLES, since_label))} from {since_short}. Section 3 shows why.

Air cargo into Bahrain itself was only {pon(ind_pon(air, 'Apr-May'))} of normal in April-May, so planes into Bahrain can't
be the whole story for light goods. Some probably flew into nearby Saudi airports and came over the causeway, or came in
the containers that still arrived.

**Why it matters:** the supply chains that held up were the ones with **another way in**: the road for goods from GCC
countries, planes (into Bahrain or nearby) for light goods. Heavy goods from far away had no other way in until ships could pass again.

---

## 3. How much must have come by road?

![Elimination](figures/step6_3_elimination.png)

**What we did:** the elimination from before-you-start, step 4, for each period, using core goods by weight.

**What we found:**

{elim_md}

1. **At least {el_all['other_min_share']:.0f}% of what arrived from {since_short} came by road or general cargo ship**:
   about {el_all['other_min_t'] / 1000:,.0f} thousand tonnes. In every period it was at least
   {el[el.index != since_label]['other_min_share'].min():.0f}%.
2. **This is a minimum.** It assumes that last year everything came in containers, which isn't true: goods from GCC
   countries alone were about {el_all['gcc_share_2025']:.0f}% of the weight in 2025, and most of those normally come by
   road. The real share was higher.

**Road or general cargo ship?** The data can't fully split the two. {gc_sentence} Two clues point to the road
carrying a large part:

* **Goods from GCC countries kept arriving at their usual weight** ({pon(el_all['gcc_weight_pct_of_2025'])} of the same
  months of 2025), so their share of everything that arrived rose from {el_all['gcc_share_2025']:.0f}% to
  {el_all['gcc_share_2026']:.0f}%. Most of these goods come from Saudi Arabia, over the causeway.
* **Cars kept arriving without car carriers.** From April no car carrier unloaded at the port, yet cars made in East
  Asia and Europe kept being imported from May, at {cars_low:.0f}-{cars_high:.0f}% of their 2025 average month:

{cars_md}

  They must have come another way: most likely landed at another country's port and driven or trucked over the
  causeway, or packed in containers.

**Why it matters:** the King Fahd Causeway, and possibly short trips by general cargo ships within the Gulf, kept a large
share of Bahrain's supply chains running while the sea route was shut. Putting a number on the road alone needs
assumptions (how much a container holds, and how much of the general cargo was imports). That is the "what if" range
planned for after step 7.

---

## 4. How far had each supply chain come back?

![Adapting](figures/step6_4_adapting.png)

**What we did:** for every origin region and product section, took step 5's gaps as **% of normal** in each war period,
found the worst one, and measured how much of the drop had been made up by {latest}.

**What we found:**

{regions_md}

{sections_md}

*Sections under {MIN_SHARE}% of the national forecast, and national special codes, are left out.*

1. **Nationally, imports came back about {share_words(nat['came_back_pct'])} of the way**: from
   {pon(nat['worst'])} of normal in {nat['worst_period']} to {pon(nat['latest'])} in {latest}.
2. **Back inside the normal range by {latest}:** {and_list(reg_inside.sort_values('forecast', ascending=False)['label'].map(region_text))}
   among origins, and {len(sec_inside)} of {len(secs)} sections, including {and_list(sec_inside.sort_values('forecast', ascending=False)['label'].str.lower().head(4))}.{above_text}
3. **Still far below normal:** {and_list(f"{l.lower()} ({pon(v)})" for l, v in zip(still['label'].head(5), still['latest'].head(5)))}
   among sections; {and_list(f"{region_text(l)} ({pon(v)})" for l, v in zip(reg_still.sort_values('latest')['label'], reg_still.sort_values('latest')['latest']))} among origins.

**A word of caution.** Part of the bounce in {latest} came when Hormuz **partly reopened** (18 June to 11 July), so it
isn't all adapting: some ships that had been waiting got through. The clearest signs of adapting are the things that
improved **while Hormuz was shut**: light goods in April-May, goods from GCC countries above normal, and cars arriving
without car carriers.

---

## 5. What happened after July?

The import data stop at {pd.Timestamp(2026, last_month, 1):%B}, but the port and airport figures reach {last_route:%B}.
{"Hormuz closed again on 11 July, so " + after_label + " is the first test of how well supply chains adapt to a closed strait without any help from a reopening." if after_label else ""}

{md_table(pd.DataFrame({"Indicator": [name for _, name, _ in INDICATORS], **{p: [pon(ind_pon(n, p)) for _, n, _ in INDICATORS] for p in [latest, *after]}}))  if after_label else ""}

{"1. **Containers kept climbing**: from " + pon(last_war_cont) + " of normal in " + latest + " to " + pon(after_cont) + " in " + after_label + ", with Hormuz closed. More container ships called (" + pon(ind_pon('Container ship visits', after_label)) + "). With the strait closed, they most likely came on ships from other ports inside the Gulf, carrying containers that had crossed land from ports outside it." if after_label else ""}
{"2. **Air cargo held at about " + pon(after_air) + " of normal.**" if after_label else ""}
{"3. **Car carriers still hadn't returned**, and general cargo stayed high." if after_label else ""}

**Why it matters:** step 5's shortfall was shrinking while the strait was partly open. The port figures suggest the
supply chains kept adapting after it closed again. The import data for August, when published, will show whether the
goods themselves followed.

---

## What this step can't tell us

1. **The road has no figures.** Everything about the causeway is worked out from the other routes, so it is a minimum,
   not a measurement.
2. **The general cargo figures are the least reliable.** Several months had to be repaired, the file doesn't say whether
   it counts only goods unloaded or loaded too, and freight tonnes can be larger than the weight.
3. **Months don't line up exactly.** A ship's arrival and the customs record of its goods can fall in different months.
   Periods of two months or more soften this.
4. **Light, middle and heavy are rough groups.** They are decided for each series (one HS2 chapter from one region), so
   a few heavy items sit in light series and the other way round.

## What this means for step 7

Step 7 writes up the project. The main findings from this step:

1. **Containers and car carriers were cut; the road and general cargo ships carried the difference.** At least half of
   what arrived from {since_short} came by road or general cargo ship.
2. **Goods with another way in held up**: GCC goods by road, light goods probably by air.
3. **Imports had come back about {share_words(nat['came_back_pct'])} of the way by {latest}**, with vehicles,
   clothing, East Asia and the Americas lagging; port figures suggest adapting continued after Hormuz closed again.

Still to do after step 7, as agreed: the "what if" range for the road's own share, and whether products that relied on
only a few supplier countries were hit harder.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")
