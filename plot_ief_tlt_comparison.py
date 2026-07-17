"""
plot_ief_tlt_comparison.py
IEF vs TLT equity-curve + drawdown comparison PNGs.

Two periods:
  1963–Jun 2026  (full spliced history, FRED proxy + ETF)
  2000–Jun 2026  (modern era)

Each PNG: log-scale equity curve (top) + drawdown with avg DD lines (bottom).

Outputs (data/analysis/):
  ief_vs_tlt_1963-jun_2026.png
  ief_vs_tlt_2000-jun_2026.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from metrics import wealth_index, drawdown_series, cagr, annualised_volatility, \
    sharpe_ratio, max_drawdown, calmar_ratio, ulcer_index

PROC    = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"

COL_IEF = "#3b82d4"   # blue  — matches existing HTML
COL_TLT = "#7c5cd8"   # purple — matches existing HTML

WINDOWS = [
    ("1963-jun_2026", "1963-01-01", "2026-06-30",
     "IEF vs TLT — 1963 to Jun 2026",
     "FRED DGS10/DGS20 yield-derived total-return proxy (1962–) spliced with IEF/TLT ETF (2002–present)"),
    ("2000-jun_2026", "2000-01-01", "2026-06-30",
     "IEF vs TLT — 2000 to Jun 2026",
     "Predominantly real ETF data · IEF from Jul 2002, TLT from Jul 2002 · pre-ETF: spliced proxy"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True).sort_index()
    ief_px = df["IEF"].dropna()
    tlt_px = df["TLT"].dropna()
    tbill  = df["BIL"].dropna().pct_change().dropna()

    # pct_change on full series then trim → preserves first month's return
    ief_ret = ief_px.pct_change().loc[start:end].dropna()
    tlt_ret = tlt_px.pct_change().loc[start:end].dropna()
    rf      = tbill.reindex(ief_ret.index).ffill().fillna(0.0)

    return ief_ret, tlt_ret, rf


# ---------------------------------------------------------------------------
# Metrics summary dict
# ---------------------------------------------------------------------------

def calc_metrics(ret: pd.Series, rf: pd.Series, label: str) -> dict:
    dd   = drawdown_series(ret) * 100
    wi   = wealth_index(ret)
    return {
        "label":   label,
        "cagr":    round(cagr(ret) * 100, 2),
        "vol":     round(annualised_volatility(ret, "monthly") * 100, 2),
        "sharpe":  round(sharpe_ratio(ret, rf, "monthly"), 3),
        "max_dd":  round(max_drawdown(ret) * 100, 2),
        "avg_dd":  round(float(dd.mean()), 2),
        "calmar":  round(calmar_ratio(ret, "monthly"), 3),
        "ulcer":   round(ulcer_index(ret), 2),
        "terminal": round(float(wi.iloc[-1]), 2),
        "wi":  wi,
        "dd":  dd,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(slug: str, start: str, end: str, title: str, subtitle: str) -> None:
    ief_ret, tlt_ret, rf = load(start, end)
    ief = calc_metrics(ief_ret, rf, "IEF")
    tlt = calc_metrics(tlt_ret, rf, "TLT")

    start_yr = int(start[:4])
    end_yr   = int(end[:4])
    span     = end_yr - start_yr

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1.8], "hspace": 0.07},
    )
    fig.patch.set_facecolor("white")

    # ── top: equity curve (log scale) ──
    ax1.semilogy(ief["wi"].index, ief["wi"].values,
                 color=COL_IEF, lw=2.2,
                 label=f"IEF  $1 → ${ief['terminal']:.2f}")
    ax1.semilogy(tlt["wi"].index, tlt["wi"].values,
                 color=COL_TLT, lw=1.8, ls="--",
                 label=f"TLT  $1 → ${tlt['terminal']:.2f}")

    # terminal value annotations at the end of each line
    last_date = ief["wi"].index[-1]
    ax1.annotate(
        f"${ief['terminal']:.2f}",
        xy=(last_date, ief["wi"].iloc[-1]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=9, fontweight="bold", color=COL_IEF, va="center",
    )
    ax1.annotate(
        f"${tlt['terminal']:.2f}",
        xy=(last_date, tlt["wi"].iloc[-1]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=9, fontweight="bold", color=COL_TLT, va="center",
    )

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=7)
    ax1.text(0.5, 0.988, subtitle,
             transform=ax1.transAxes, ha="center", va="top",
             fontsize=8, color="#57606a", style="italic")

    ax1.set_ylabel("Growth of $1 (log scale)", fontsize=9)
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:,.0f}" if y >= 100 else (
                      f"${y:.0f}" if y >= 10 else (
                      f"${y:.1f}" if y >= 0.5 else ""))
    ))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=[2, 5], numticks=20))
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.legend(fontsize=10, framealpha=0.95, loc="upper left")
    ax1.grid(True, which="both", lw=0.35, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── bottom: drawdown + avg DD lines ──
    ax2.fill_between(ief["dd"].index, ief["dd"].values, 0,
                     color=COL_IEF, alpha=0.20, zorder=2)
    ax2.fill_between(tlt["dd"].index, tlt["dd"].values, 0,
                     color=COL_TLT, alpha=0.18, zorder=2)
    ax2.plot(ief["dd"].index, ief["dd"].values, color=COL_IEF, lw=1.4, zorder=3)
    ax2.plot(tlt["dd"].index, tlt["dd"].values, color=COL_TLT, lw=1.2,
             ls="--", zorder=3)

    # average drawdown horizontal lines
    for m, col in [(ief, COL_IEF), (tlt, COL_TLT)]:
        ax2.axhline(m["avg_dd"], color=col, lw=1.3, ls=":", zorder=4, alpha=0.9)
        ax2.annotate(
            f"avg {m['avg_dd']:.1f}%",
            xy=(1.0, m["avg_dd"]),
            xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            fontsize=7.5, color=col, va="center",
        )

    # legend: max DD + avg DD per asset
    legend_handles = [
        Line2D([0], [0], color=COL_IEF, lw=2.0,
               label=f"IEF   max {ief['max_dd']:.1f}%   avg {ief['avg_dd']:.1f}%"),
        Line2D([0], [0], color=COL_TLT, lw=1.6, ls="--",
               label=f"TLT   max {tlt['max_dd']:.1f}%   avg {tlt['avg_dd']:.1f}%"),
        Line2D([0], [0], color="gray", lw=1.2, ls=":",
               label="avg DD (dotted)"),
    ]
    ax2.legend(handles=legend_handles, fontsize=8.5, framealpha=0.95, loc="lower left")

    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.set_xlabel("")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.grid(True, lw=0.35, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # x-axis ticks
    step = 10 if span > 40 else 5
    ticks = [pd.Timestamp(f"{y}-01-01")
             for y in range(start_yr, end_yr + 1, step)]
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), ha="center")

    # compact stats table below the chart
    rows = [
        ("",        "IEF",                               "TLT"),
        ("CAGR",    f"{ief['cagr']:+.2f}%",              f"{tlt['cagr']:+.2f}%"),
        ("Vol",     f"{ief['vol']:.2f}%",                f"{tlt['vol']:.2f}%"),
        ("Sharpe",  f"{ief['sharpe']:.3f}",              f"{tlt['sharpe']:.3f}"),
        ("Max DD",  f"{ief['max_dd']:.2f}%",             f"{tlt['max_dd']:.2f}%"),
        ("Avg DD",  f"{ief['avg_dd']:.2f}%",             f"{tlt['avg_dd']:.2f}%"),
        ("Calmar",  f"{ief['calmar']:.3f}",              f"{tlt['calmar']:.3f}"),
        ("Ulcer",   f"{ief['ulcer']:.2f}",               f"{tlt['ulcer']:.2f}"),
        ("$1 →",    f"${ief['terminal']:,.2f}",          f"${tlt['terminal']:,.2f}"),
    ]
    col_labels = [r[0] for r in rows[1:]]
    ief_vals   = [r[1] for r in rows[1:]]
    tlt_vals   = [r[2] for r in rows[1:]]

    table = ax2.table(
        cellText=[ief_vals, tlt_vals],
        rowLabels=["IEF", "TLT"],
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=[0.0, -0.62, 1.0, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if row == 0:
            cell.set_facecolor("#f7f8fa")
            cell.set_text_props(fontweight="bold", color="#57606a", fontsize=7.5)
        elif col == -1:   # row labels
            cell.set_text_props(
                fontweight="bold",
                color=COL_IEF if row == 1 else COL_TLT,
            )
            cell.set_facecolor("#f7f8fa")
        else:
            cell.set_facecolor("white")

    fig.text(
        0.5, 0.0,
        "Data: FRED DGS10/DGS20 yield-derived total-return index spliced with IEF/TLT ETF  |  "
        "Monthly total return, no costs",
        ha="center", fontsize=7.5, color="#9ca3af",
    )
    fig.subplots_adjust(bottom=0.22)
    out = OUT_DIR / f"ief_vs_tlt_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  saved → {out.name}")
    print(f"    IEF: CAGR {ief['cagr']:+.2f}%  Vol {ief['vol']:.2f}%  "
          f"MaxDD {ief['max_dd']:.2f}%  AvgDD {ief['avg_dd']:.2f}%  "
          f"$1→${ief['terminal']:.2f}")
    print(f"    TLT: CAGR {tlt['cagr']:+.2f}%  Vol {tlt['vol']:.2f}%  "
          f"MaxDD {tlt['max_dd']:.2f}%  AvgDD {tlt['avg_dd']:.2f}%  "
          f"$1→${tlt['terminal']:.2f}")


def main() -> None:
    for slug, start, end, title, subtitle in WINDOWS:
        plot(slug, start, end, title, subtitle)


if __name__ == "__main__":
    main()
