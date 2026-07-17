"""
plot_sp500_equity_curves.py
Regenerate the 2 S&P 500 equity-curve + drawdown PNGs with avg DD overlay.

Outputs (data/analysis/):
  sp500_equity_curve_1901_2026.png
  sp500_equity_curve_2000_2026.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from metrics import wealth_index, drawdown_series

OUT_DIR = ROOT / "data" / "analysis"
CSV     = OUT_DIR / "sp500_bh_timing_1901_2026.csv"

# ── colours matching original PNGs ──
COL_BH     = "#3d3d3d"   # dark grey
COL_TIMING = "#2563eb"   # blue

WINDOWS = [
    (
        "1901_2026",
        "1901-01-01", "2026-06-30",
        "S&P 500: Buy & Hold vs. Timing (SMA-10) — 1901 to Jun 2026",
        "$1 invested January 1901  ·  log scale  ·  total return  ·  cash = T-bills when out of market",
        "dot-com crash, GFC (2009), COVID (2020)",   # unused for full window
    ),
    (
        "2000_2026",
        "2000-01-01", "2026-06-30",
        "S&P 500: Buy & Hold vs. Timing (SMA-10) — 2000 to Jun 2026",
        "$1 invested January 2000  ·  log scale  ·  dot-com crash, GFC (2009), COVID (2020)",
        "",
    ),
]


def load(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(CSV, index_col=0, parse_dates=True)
    bh  = df["monthly_ret_bh"].loc[start:end].dropna()
    tm  = df["monthly_ret_timing"].loc[start:end].dropna()
    return bh, tm


def plot_window(
    slug: str,
    start: str, end: str,
    title: str, subtitle: str,
) -> None:
    bh_ret, tm_ret = load(start, end)

    wi_bh  = wealth_index(bh_ret)
    wi_tm  = wealth_index(tm_ret)
    dd_bh  = drawdown_series(bh_ret) * 100
    dd_tm  = drawdown_series(tm_ret) * 100

    avg_bh = float(dd_bh.mean())
    avg_tm = float(dd_tm.mean())
    max_bh = float(dd_bh.min())
    max_tm = float(dd_tm.min())
    term_bh = float(wi_bh.iloc[-1])
    term_tm = float(wi_tm.iloc[-1])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.5),
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.06},
    )
    fig.patch.set_facecolor("white")

    # ── top: equity curve ──
    ax1.semilogy(wi_bh.index, wi_bh.values, color=COL_BH,     lw=1.4, label=f"Buy & Hold    1→{term_bh:,.0f}")
    ax1.semilogy(wi_tm.index, wi_tm.values, color=COL_TIMING,  lw=1.8, label=f"Timing SMA-10  1→{term_tm:,.0f}")

    # terminal labels on right
    ax1.annotate(f"${term_tm:,.0f}", xy=(wi_tm.index[-1], term_tm),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=9, fontweight="bold", color=COL_TIMING, va="center")
    ax1.annotate(f"${term_bh:,.0f}", xy=(wi_bh.index[-1], term_bh),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=COL_BH, va="center")

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax1.text(0.5, 0.985, subtitle, transform=ax1.transAxes,
             ha="center", va="top", fontsize=8, color="#57606a", style="italic")
    ax1.set_ylabel("Portfolio value ($, log scale)", fontsize=9)
    # Use only clean power-of-10 and half-decade ticks; suppress minor labels
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:,.0f}" if y >= 100 else (f"${y:.0f}" if y >= 10 else (f"${y:.1f}" if y >= 0.5 else ""))
    ))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=[2, 5], numticks=20))
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.legend(fontsize=9.5, framealpha=0.95, loc="upper left")
    ax1.grid(True, which="both", lw=0.4, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── bottom: drawdown ──
    ax2.fill_between(dd_bh.index, dd_bh.values, 0, color=COL_BH,    alpha=0.35, zorder=2)
    ax2.fill_between(dd_tm.index, dd_tm.values, 0, color=COL_TIMING, alpha=0.40, zorder=3)
    ax2.plot(dd_bh.index, dd_bh.values, color=COL_BH,    lw=0.9, zorder=4)
    ax2.plot(dd_tm.index, dd_tm.values, color=COL_TIMING, lw=1.1, zorder=5)

    # ── avg DD horizontal lines ──
    ax2.axhline(avg_bh, color=COL_BH,    lw=1.3, ls=":", zorder=6, alpha=0.9)
    ax2.axhline(avg_tm, color=COL_TIMING, lw=1.3, ls=":", zorder=6, alpha=0.9)

    # right-margin labels for avg DD lines
    xmax = dd_bh.index[-1]
    ax2.annotate(f"avg {avg_bh:.1f}%",
                 xy=(1.0, avg_bh), xycoords=("axes fraction", "data"),
                 xytext=(5, 0), textcoords="offset points",
                 fontsize=7.5, color=COL_BH, va="center")
    ax2.annotate(f"avg {avg_tm:.1f}%",
                 xy=(1.0, avg_tm), xycoords=("axes fraction", "data"),
                 xytext=(5, 0), textcoords="offset points",
                 fontsize=7.5, color=COL_TIMING, va="center")

    # legend with max + avg DD
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COL_BH,    lw=1.5, label=f"B&H    max {max_bh:.1f}%    avg {avg_bh:.1f}%"),
        Line2D([0], [0], color=COL_TIMING, lw=1.5, label=f"Timing max {max_tm:.1f}%   avg {avg_tm:.1f}%"),
        Line2D([0], [0], color="gray", lw=1.2, ls=":", label="avg DD (dotted line)"),
    ]
    ax2.legend(handles=legend_handles, fontsize=8.5, framealpha=0.95, loc="lower left")

    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.set_xlabel("Year", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.grid(True, lw=0.4, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # x-axis ticks
    start_yr = int(start[:4])
    end_yr   = int(end[:4])
    span     = end_yr - start_yr
    step     = 10 if span > 60 else 5
    ticks    = [pd.Timestamp(f"{y}-01-01") for y in range(start_yr, end_yr + 1, step)]
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.text(0.5, 0.005,
             "Data: Shiller S&P 500 TR spliced with SPY ETF  |  "
             "Timing: price > SMA-10, T-bill cash, 1-month lag, no look-ahead",
             ha="center", fontsize=7.5, color="#9ca3af")
    out = OUT_DIR / f"sp500_equity_curve_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {out.name}  (avg DD: B&H {avg_bh:.1f}%  Timing {avg_tm:.1f}%)")


def main() -> None:
    for slug, start, end, title, subtitle, _ in WINDOWS:
        plot_window(slug, start, end, title, subtitle)


if __name__ == "__main__":
    main()
