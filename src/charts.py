"""Shared chart style and drawing helpers used by every report.

Every chart in the project is drawn with matplotlib using the helpers in this file, so the
figures look like one family: the same fonts, colours, margins, legend placement and source
note. Keeping the style in one module also means a report script only has to say *what* to
draw, not how to make it look right.

What is in here
---------------
* **Palette and global style.** A small, colour-blind-checked palette (blue, orange, aqua,
  yellow, plus a red for shortfalls) and neutral greys for text, grid lines and shading.
  The matplotlib defaults are replaced once, when this module is imported.
* **Layout helpers.** Figures are laid out in inches rather than matplotlib's fractions, so a
  title, subtitle, legend and chart area can be placed precisely and consistently.
  ``new_figure`` draws the title block, ``add_axes`` places a chart area by its margins and
  ``legend`` puts the legend in a row under the subtitle.
* **Mark helpers.** Bars with a rounded data end (``rounded_bar``), end-of-line dots and
  labels that move apart instead of overlapping (``end_labels``), and month tick labels.
* **Grid chart.** A table-like chart used in steps 5 and 6: one row per total, one column
  per period, coloured only where a gap is outside its normal range.

Design choices
--------------
* Static PNGs rather than interactive charts: they render directly in GitHub markdown, which
  is where the reports are read. The cost is that readers can't hover for exact values, so the
  reports always put the key numbers in tables next to the charts.
* Colour is only used to carry meaning (a series, or below/inside/above normal); everything
  else is grey. Direct labels are preferred to legends where there is room.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # draw to files only; no window is opened (works on servers and in CI)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter

from config import ROOT

FIG_DIR = ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# Palette. Text and structure use greys; colour is reserved for data.
# The four categorical colours were checked for colour-blind separation.
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"   # background, main text, secondary text, notes
GRID, BASELINE, SHADE = "#e1e0d9", "#c3c2b7", "#f0efec"                  # grid lines, zero lines, shaded periods
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical series, always in this order
RED = "#e34948"                                                          # shortfalls (below normal)

# Steps for the grid chart: shades of red for gaps below normal, blue above, grey inside the range.
RED_STEPS = ["#f3b4b2", RED, "#a92b2b"]
BLUE_STEPS = ["#9ec5f4", BLUE, "#184f95"]
STEP_EDGES = [25, 50]                   # |gap| under 25%, 25-50%, 50% or more

WAR_START = pd.Timestamp("2026-02-28")  # used to shade the war months on time-series charts

plt.rcParams.update({
    "font.family": "sans-serif",
    # Inter is preferred; the others are fallbacks so the charts still render on any machine.
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 100, "savefig.dpi": 200,          # layout is in inches at 100 dpi; files are saved at 200 dpi
    "axes.edgecolor": BASELINE, "axes.linewidth": 1, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9.5,
})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def new_figure(width: float, height: float, title: str, subtitle: str):
    """A figure of the given size (inches) with the title and subtitle drawn at the top left.

    The title states the finding ("Containers fell to a third of normal") and the subtitle says
    what is plotted, so a reader who only looks at the chart still gets the point.
    """
    fig = plt.figure(figsize=(width, height))
    fig.text(0.6 / width, 1 - 0.32 / height, title, fontsize=14, fontweight="semibold", color=INK, va="top")
    fig.text(0.6 / width, 1 - 0.62 / height, subtitle, fontsize=10, color=INK2, va="top", linespacing=1.45)
    return fig


def add_axes(fig, left: float, right: float, top: float, bottom: float):
    """Add a chart area placed by its distance in inches from each edge of the figure."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (top + bottom) / h])


def legend(fig, handles, labels, top: float, ncol: int):
    """A one-row legend, `top` inches below the top of the figure, aligned with the title."""
    w, h = fig.get_size_inches()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.52 / w, 1 - top / h), ncol=ncol,
               handlelength=1.4, columnspacing=1.6, borderaxespad=0)


def tidy(ax, grid_axis: str = "y"):
    """Remove chart borders except the baseline and draw light grid lines on one axis."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if grid_axis == "x":                     # horizontal charts read against vertical grid lines, so no baseline
        ax.spines["bottom"].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def source_note(fig, text: str = "Source: Bahrain Open Data Portal (iGA), www.data.gov.bh. Analysis: own."):
    """The small source line at the bottom left of every chart."""
    w, h = fig.get_size_inches()
    fig.text(0.6 / w, 0.14 / h, text, fontsize=8, color=MUTED)


def save(fig, name: str) -> None:
    """Save a figure to reports/figures/ and free its memory."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Marks and labels
# ---------------------------------------------------------------------------
def line_handle(colour, lw=2):
    """A short line, for legends."""
    return plt.Line2D([], [], color=colour, lw=lw)


def dot_handle(colour):
    """A dot with a white ring, for legends."""
    return plt.Line2D([], [], ls="", marker="o", ms=8, color=colour, mec=SURFACE, mew=2)


def end_dot(ax, x, y, colour):
    """Mark the last point of a line."""
    ax.plot([x], [y], "o", ms=8, color=colour, mec=SURFACE, mew=2, zorder=5)


def end_labels(ax, items, x_text=None, gap_pt: float = 13):
    """Label the ends of several lines without the labels overlapping.

    items: (x, y, text) for each line. Labels sit just right of their point, or in a shared
    column at data-x `x_text`. Working in screen pixels, labels are sorted top to bottom and
    pushed down until each is at least `gap_pt` points below the one above; the whole block
    is then shifted so it stays centred on the points. A label that had to move is joined to
    its point by a thin leader line.
    """
    fig = ax.figure
    fig.canvas.draw()                          # the transforms are only final after a draw
    to_px = ax.transData.transform
    pts = []
    for x, y, text in items:
        xn = mdates.date2num(x) if isinstance(x, pd.Timestamp) else x
        px, py = to_px((xn, y))
        pts.append({"x": x, "y": y, "text": text, "px": px, "py": py})
    gap_px = gap_pt * fig.dpi / 72
    pts.sort(key=lambda p: -p["py"])           # top of the chart first
    ys = [p["py"] for p in pts]
    for k in range(1, len(ys)):
        ys[k] = min(ys[k], ys[k - 1] - gap_px)
    shift = np.mean([p["py"] for p in pts]) - np.mean(ys)
    col_px = None
    if x_text is not None:
        col_px = to_px((mdates.date2num(x_text) if isinstance(x_text, pd.Timestamp) else x_text, 0))[0]
    for p, ty in zip(pts, ys):
        ty += shift
        tx = col_px if col_px is not None else p["px"] + 9 * fig.dpi / 72
        dx, dy = (tx - p["px"]) * 72 / fig.dpi, (ty - p["py"]) * 72 / fig.dpi   # offsets in points
        leader = abs(dy) > 1.5 or col_px is not None
        ax.annotate(p["text"], (p["x"], p["y"]), xytext=(dx, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8, shrinkA=3, shrinkB=6) if leader else None)


def rounded_bar(ax, start, length, centre, thickness, colour, horizontal=True, radius_px=4):
    """A bar with a rounded data end and a square end on the baseline.

    Matplotlib's bars are plain rectangles, so the outline is built as a path: straight edges,
    with quadratic curves at the two corners of the data end. The radius is given in pixels and
    converted to data units for each axis, so the corners look the same on any chart. Call it
    after the axis limits are set. A negative length draws the bar the other way.
    """
    fig = ax.figure
    box = ax.get_window_extent(fig.canvas.get_renderer())
    x_per_px = abs(np.ptp(ax.get_xlim())) / box.width
    y_per_px = abs(np.ptp(ax.get_ylim())) / box.height
    M, L, C, Z = Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CLOSEPOLY
    codes = [M, L, C, C, L, C, C, L, Z]
    sign = -1.0 if length < 0 else 1.0
    if horizontal:
        ra, rc = sign * min(radius_px * x_per_px, abs(length) / 2), min(radius_px * y_per_px, thickness / 2)
        x0, x1, y0, y1 = start, start + length, centre - thickness / 2, centre + thickness / 2
        verts = [(x0, y0), (x1 - ra, y0), (x1, y0), (x1, y0 + rc), (x1, y1 - rc), (x1, y1), (x1 - ra, y1), (x0, y1), (x0, y0)]
    else:
        ra, rc = sign * min(radius_px * y_per_px, abs(length) / 2), min(radius_px * x_per_px, thickness / 2)
        x0, x1, y0, y1 = centre - thickness / 2, centre + thickness / 2, start, start + length
        verts = [(x1, y0), (x1, y1 - ra), (x1, y1), (x1 - rc, y1), (x0 + rc, y1), (x0, y1), (x0, y1 - ra), (x0, y0), (x1, y0)]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=colour, lw=0, zorder=3))


def month_axis(ax, months=(1, 7), first=None):
    """Month tick labels on a date axis; January (and the first tick, if given) also shows the year."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))

    def fmt(x, _):
        d = mdates.num2date(x)
        show_year = d.month == 1 or (first is not None and d.year == first.year and d.month == first.month)
        return d.strftime("%b\n%Y") if show_year else d.strftime("%b")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def date_num(ts: pd.Timestamp) -> float:
    """A date as matplotlib's internal number, for drawing shapes on a date axis."""
    return mdates.date2num(ts)


# ---------------------------------------------------------------------------
# Grid chart (steps 5 and 6)
# ---------------------------------------------------------------------------
def cell_colour(gap: float, outside: bool) -> tuple[str, str]:
    """(fill colour, text colour) for one grid cell.

    Grey when the gap is inside the normal range (it could be ordinary noise); otherwise a shade
    of red (below normal) or blue (above), darker for bigger gaps.
    """
    if not outside or np.isnan(gap):
        return SHADE, INK2
    steps = RED_STEPS if gap < 0 else BLUE_STEPS
    i = int(np.searchsorted(STEP_EDGES, abs(gap), side="right"))
    return steps[i], (INK if i == 0 else "#ffffff")


def grid_chart(rows: list[tuple[str, dict]], columns: list[str], title: str, subtitle: str, name: str,
               first_col_width: float = 3.3, bold_columns: set[str] | None = None, cell_width: float = 0.78) -> None:
    """A table-like chart: one row per total, one column per month or period.

    rows: (label, {column: (gap_pct, outside)}). A label wrapped in ** is drawn in bold (used
    for "All of Bahrain"). Column headers in `bold_columns` are bold; by default that is any
    header starting with "Mar", i.e. the whole-war column.

    The chart is drawn on an axis running 0..n_cols across and 0..n_rows down, so every cell
    is a unit square; a 2-pixel gap of background colour separates neighbouring cells.
    """
    n_rows, n_cols = len(rows), len(columns)
    cell_w, cell_h = cell_width, 0.36
    width = first_col_width + n_cols * cell_w + 0.5
    height = 1.95 + n_rows * cell_h + 0.85
    fig = new_figure(width, height, title, subtitle)
    w, h = fig.get_size_inches()

    # Colour key, drawn as small squares with labels in one row under the subtitle.
    key_y = 1 - 1.3 / h
    x = 0.6 / w
    items = [(RED_STEPS[0], "below by up to 25%"), (RED_STEPS[1], "25-50%"), (RED_STEPS[2], "50% or more"),
             (BLUE_STEPS[1], "above normal"), (SHADE, "inside the normal range")]
    for colour, label in items:
        fig.patches.append(Rectangle((x, key_y - 0.06 / h), 0.22 / w, 0.16 / h, transform=fig.transFigure,
                                     color=colour, lw=0))
        fig.text(x + 0.3 / w, key_y + 0.02 / h, label, fontsize=9, color=INK2, va="center")
        x += (0.3 + 0.075 * len(label) + 0.35) / w          # rough text width, so the next item starts after this one

    ax = fig.add_axes([first_col_width / w, 0.75 / h, n_cols * cell_w / w, n_rows * cell_h / h])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)                                   # first row at the top
    ax.axis("off")
    gap_px = 2 / (cell_w * fig.dpi)
    gap_py = 2 / (cell_h * fig.dpi)
    for i, (label, cells) in enumerate(rows):
        bold = label.startswith("**")
        fig.text((first_col_width - 0.12) / w, (0.75 + (n_rows - i - 0.5) * cell_h) / h, label.strip("*"), ha="right",
                 va="center", fontsize=9.5, color=INK, fontweight="semibold" if bold else "normal")
        for j, col in enumerate(columns):
            gap, outside = cells.get(col, (np.nan, False))
            fill, text = cell_colour(gap, outside)
            ax.add_patch(Rectangle((j + gap_px, i + gap_py), 1 - 2 * gap_px, 1 - 2 * gap_py, color=fill, lw=0))
            if not np.isnan(gap):
                ax.text(j + 0.5, i + 0.52, f"{gap:+.0f}" if round(gap) else "0", ha="center", va="center",
                        fontsize=8.5, color=text)
    for j, col in enumerate(columns):
        is_bold = col in bold_columns if bold_columns is not None else col.startswith("Mar")
        fig.text((first_col_width + (j + 0.5) * cell_w) / w, (0.75 + n_rows * cell_h + 0.12) / h, col, ha="center",
                 va="bottom", fontsize=9, color=INK2, fontweight="semibold" if is_bold else "normal")
    source_note(fig)
    save(fig, name)
