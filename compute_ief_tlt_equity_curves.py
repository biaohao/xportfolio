"""
compute_ief_tlt_equity_curves.py
Regenerates the 6 IEF/TLT equity-curve PNGs:
  ief_equity_curve_1963-jun_2026.png
  ief_equity_curve_1973-jun_2026.png
  ief_equity_curve_2000-jun_2026.png
  tlt_equity_curve_1963-jun_2026.png
  tlt_equity_curve_1973-jun_2026.png
  tlt_equity_curve_2000-jun_2026.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

OUT_DIR = Path("data/analysis")
DATA    = Path("data/processed/prices_monthly_spliced.csv")
SMA     = 10

BLUE   = "#3b82d4"   # IEF
PURPLE = "#7c5cd8"   # TLT

WINDOWS = [
    # (slug,        asset, start,        colour,  footnote_asset)
    ("1963-jun_2026", "IEF", "1963-01-01", BLUE,   "IEF"),
    ("1973-jun_2026", "IEF", "1973-01-01", BLUE,   "IEF"),
    ("2000-jun_2026", "IEF", "2000-01-01", BLUE,   "IEF"),
    ("1963-jun_2026", "TLT", "1963-01-01", PURPLE, "TLT"),
    ("1973-jun_2026", "TLT", "1973-01-01", PURPLE, "TLT"),
    ("2000-jun_2026", "TLT", "2000-01-01", PURPLE, "TLT"),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def run_timing(prices: pd.Series, tbill_ret: pd.Series,
               start: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    px = prices.dropna()
    px = px[px.index >= start]
    sma = px.rolling(SMA).mean()
    signal = (px > sma).astype(float).shift(1)   # 1-month lag
    ret_asset = px.pct_change()
    ret_bh    = ret_asset[px.index >= start].dropna()
    tb        = tbill_ret.reindex(ret_bh.index).fillna(0)
    sig       = signal.reindex(ret_bh.index).fillna(0)
    ret_timing = sig * ret_bh + (1 - sig) * tb
    return ret_bh, ret_timing, sig


def wealth(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def drawdown(ret: pd.Series) -> pd.Series:
    wi = wealth(ret)
    peak = wi.cummax()
    return (wi / peak - 1) * 100


def stats(ret: pd.Series, tbill: pd.Series) -> dict:
    ann = 12
    n = len(ret)
    total = (1 + ret).prod()
    years = n / ann
    cagr  = (total ** (1 / years) - 1) * 100 if years > 0 else float("nan")
    vol   = ret.std() * np.sqrt(ann) * 100
    rf    = tbill.reindex(ret.index).fillna(0).mean() * ann
    sharpe = (ret.mean() * ann - rf) / (ret.std() * np.sqrt(ann)) if ret.std() > 0 else 0
    dd    = drawdown(ret)
    max_dd = dd.min()
    avg_dd = dd.mean()
    terminal = wealth(ret).iloc[-1]
    return dict(cagr=cagr, vol=vol, sharpe=sharpe,
                max_dd=max_dd, avg_dd=avg_dd, terminal=terminal)


def plot_curve(slug: str, asset: str, start: str, colour: str, footnote_asset: str,
               prices: pd.DataFrame) -> None:
    tbill_ret = prices["BIL"].pct_change().dropna()
    px = prices[asset].dropna()

    ret_bh, ret_t, sig = run_timing(px, tbill_ret, start)
    wi_bh = wealth(ret_bh)
    wi_t  = wealth(ret_t)
    dd_bh = drawdown(ret_bh)
    dd_t  = drawdown(ret_t)
    m_bh  = stats(ret_bh, tbill_ret)
    m_t   = stats(ret_t, tbill_ret)

    # ── figure ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
        facecolor="white",
    )

    end_yr = wi_bh.index[-1].year
    start_label = start[:4]

    # top: equity curve (log scale)
    ax1.plot(wi_bh.index, wi_bh.values, color="black", lw=1.2,
             label=f"Buy & Hold  1→{m_bh['terminal']:.2f}")
    ax1.plot(wi_t.index,  wi_t.values,  color=colour,  lw=1.5,
             label=f"Timing SMA-{SMA}  1→{m_t['terminal']:.2f}")
    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio value ($, log scale)", fontsize=9)
    ax1.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax1.grid(True, lw=0.4, color="#e5e7eb")
    ax1.tick_params(labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title(
        f"{asset}: Buy & Hold vs. Timing (SMA-{SMA}) — {start_label} to Jun 2026",
        fontsize=13, fontweight="bold", pad=10,
    )
    subtitle = (f"$1 invested January {start_label}  ·  log scale  ·  "
                "total return  ·  cash = T-bills when out of market")
    ax1.text(0.5, 1.01, subtitle, transform=ax1.transAxes,
             ha="center", fontsize=8, color="#57606a")
    # terminal labels
    ax1.annotate(f"${m_bh['terminal']:.2f}",
                 xy=(wi_bh.index[-1], wi_bh.iloc[-1]),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=8, color="black", va="center")
    ax1.annotate(f"${m_t['terminal']:.2f}",
                 xy=(wi_t.index[-1], wi_t.iloc[-1]),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=8, color=colour, va="center", fontweight="bold")

    # bottom: drawdown
    ax2.fill_between(dd_bh.index, dd_bh.values, 0, color="black", alpha=0.15)
    ax2.fill_between(dd_t.index,  dd_t.values,  0, color=colour,  alpha=0.25)
    ax2.plot(dd_bh.index, dd_bh.values, color="black", lw=0.8)
    ax2.plot(dd_t.index,  dd_t.values,  color=colour,  lw=0.8, ls="--")
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.grid(True, lw=0.4, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(
        [
            plt.Line2D([], [], color="black", lw=1.2),
            plt.Line2D([], [], color=colour,  lw=1.5),
        ],
        [
            f"B&H     max {m_bh['max_dd']:.1f}%",
            f"Timing max {m_t['max_dd']:.1f}%",
        ],
        fontsize=8, framealpha=0.9, loc="lower left",
    )

    # shared x ticks
    span = end_yr - int(start_label)
    step = 10 if span > 40 else 5
    ticks = pd.date_range(f"{start_label}-01-01", f"{end_yr+1}-01-01", freq=f"{step}YS")
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xticks(ticks)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

    fig.tight_layout()

    # footnote (centre) + @biaohao (top-right)
    bond_label = "DGS10" if asset == "IEF" else "DGS20"
    fig.text(
        0.5, 0.003,
        f"Data: FRED {bond_label} yield-derived TR index spliced with {footnote_asset} ETF  |  "
        "Timing: price > SMA-10, T-bill cash, 1-month lag, no look-ahead",
        ha="center", fontsize=7.5, color="#9ca3af",
    )
    out = OUT_DIR / f"{asset.lower()}_equity_curve_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {out.name}  "
          f"(B&H CAGR {m_bh['cagr']:.2f}%  Timing CAGR {m_t['cagr']:.2f}%)")


def main() -> None:
    print("Loading data…")
    prices = load()
    for slug, asset, start, colour, footnote_asset in WINDOWS:
        plot_curve(slug, asset, start, colour, footnote_asset, prices)


if __name__ == "__main__":
    main()
