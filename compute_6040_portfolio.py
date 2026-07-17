"""
compute_6040_portfolio.py
60/40 portfolio (60% S&P 500 · 40% IEF) — Buy & Hold vs SMA-10 Timing
Annual rebalancing at year-end.

Periods:
  1966–Jun 2026
  2000–Jun 2026

Outputs (data/analysis/):
  6040_equity_curve_1966-jun_2026.png
  6040_equity_curve_2000-jun_2026.png
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from metrics import (
    wealth_index, drawdown_series, cagr,
    annualised_volatility, sharpe_ratio, max_drawdown, ulcer_index,
)

PROC    = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMA     = 10
BLUE    = "#2563eb"   # B&H
ORANGE  = "#e07b39"   # Timing

WINDOWS = [
    ("1966-jun_2026", "1966-01-01"),
    ("1973-jun_2026", "1973-01-01"),
    ("2000-jun_2026", "2000-01-01"),
]

# Portfolio configurations: (w_spy, w_ief, short_label, file_prefix, html_name)
PORTFOLIOS = [
    (0.60, 0.40, "60/40", "6040", "6040-bh-vs-timing-1966-2026.html"),
    (0.40, 0.60, "40/60", "4060", "4060-bh-vs-timing-1966-2026.html"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True).sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


# ---------------------------------------------------------------------------
# Portfolio engine
# ---------------------------------------------------------------------------

def build_portfolio(prices: pd.DataFrame, start: str,
                    w_spy: float, w_ief: float) -> tuple[pd.Series, pd.Series]:
    """
    Returns (ret_bh, ret_timing) monthly return series starting from `start`.
    w_spy + w_ief must equal 1.0.
    Annual rebalance every January; each leg independently SMA-timed.
    """
    warmup = "1962-01-01"
    spy = prices["SPY"].dropna()
    ief = prices["IEF"].dropna()
    bil = prices["BIL"].dropna()

    common = spy.index.intersection(ief.index).intersection(bil.index)
    common = common[common >= warmup]
    spy = spy.reindex(common)
    ief = ief.reindex(common)
    bil = bil.reindex(common)

    r_spy = spy.pct_change()
    r_ief = ief.pct_change()
    r_bil = bil.pct_change()

    sig_spy = (spy > spy.rolling(SMA).mean()).astype(float).shift(1)
    sig_ief = (ief > ief.rolling(SMA).mean()).astype(float).shift(1)

    idx = common[common >= start]
    r_spy   = r_spy.reindex(idx)
    r_ief   = r_ief.reindex(idx)
    r_bil   = r_bil.reindex(idx)
    sig_spy = sig_spy.reindex(idx).fillna(0)
    sig_ief = sig_ief.reindex(idx).fillna(0)

    # ── B&H with annual rebalance ──────────────────────────────────────────
    val_spy_bh = w_spy
    val_ief_bh = w_ief
    ret_bh = []
    for dt in idx:
        if dt.month == 1 and dt != idx[0]:
            total = val_spy_bh + val_ief_bh
            val_spy_bh = total * w_spy
            val_ief_bh = total * w_ief
        rs = r_spy.loc[dt] if not np.isnan(r_spy.loc[dt]) else 0.0
        ri = r_ief.loc[dt] if not np.isnan(r_ief.loc[dt]) else 0.0
        port_ret = (val_spy_bh * rs + val_ief_bh * ri) / (val_spy_bh + val_ief_bh)
        ret_bh.append(port_ret)
        val_spy_bh *= (1 + rs)
        val_ief_bh *= (1 + ri)
    ret_bh = pd.Series(ret_bh, index=idx, name="ret_bh")

    # ── Timing with annual rebalance of target weights ─────────────────────
    val_spy_t = w_spy
    val_ief_t = w_ief
    ret_timing = []
    for dt in idx:
        if dt.month == 1 and dt != idx[0]:
            total = val_spy_t + val_ief_t
            val_spy_t = total * w_spy
            val_ief_t = total * w_ief
        rs  = r_spy.loc[dt] if not np.isnan(r_spy.loc[dt]) else 0.0
        ri  = r_ief.loc[dt] if not np.isnan(r_ief.loc[dt]) else 0.0
        rb  = r_bil.loc[dt] if not np.isnan(r_bil.loc[dt]) else 0.0
        ss  = sig_spy.loc[dt]
        si  = sig_ief.loc[dt]
        r_leg_spy = ss * rs + (1 - ss) * rb
        r_leg_ief = si * ri + (1 - si) * rb
        port_ret  = (val_spy_t * r_leg_spy + val_ief_t * r_leg_ief) / (val_spy_t + val_ief_t)
        ret_timing.append(port_ret)
        val_spy_t *= (1 + r_leg_spy)
        val_ief_t *= (1 + r_leg_ief)
    ret_timing = pd.Series(ret_timing, index=idx, name="ret_timing")

    return ret_bh, ret_timing


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def plot_window(slug: str, start: str, port_label: str, file_prefix: str,
                ret_bh: pd.Series, ret_timing: pd.Series) -> dict:
    title = (f"{port_label} Portfolio (S&P 500 / IEF)  |  "
             f"{start[:4]}–Jun 2026\n"
             "Annual rebalancing · SMA-10 timing · T-bill cash")

    wi_bh = wealth_index(ret_bh)
    wi_t  = wealth_index(ret_timing)
    dd_bh = drawdown_series(ret_bh) * 100
    dd_t  = drawdown_series(ret_timing) * 100

    avg_dd_bh = dd_bh.mean()
    avg_dd_t  = dd_t.mean()

    # Stats
    stats_bh = dict(
        cagr   = cagr(ret_bh) * 100,
        vol    = annualised_volatility(ret_bh, "monthly") * 100,
        sharpe = sharpe_ratio(ret_bh, 0.0, "monthly"),
        maxdd  = max_drawdown(ret_bh) * 100,
        avgdd  = avg_dd_bh,
        ulcer  = ulcer_index(ret_bh),
        term   = wi_bh.iloc[-1],
        n_months = len(ret_bh),
    )
    stats_t = dict(
        cagr   = cagr(ret_timing) * 100,
        vol    = annualised_volatility(ret_timing, "monthly") * 100,
        sharpe = sharpe_ratio(ret_timing, 0.0, "monthly"),
        maxdd  = max_drawdown(ret_timing) * 100,
        avgdd  = avg_dd_t,
        ulcer  = ulcer_index(ret_timing),
        term   = wi_t.iloc[-1],
        n_months = len(ret_timing),
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.07},
        facecolor="white",
    )

    # ── Top: equity curve ──
    ax1.semilogy(wi_bh.index, wi_bh.values, color=BLUE,   lw=1.6,
                 label=f"B&H  ${stats_bh['term']:.2f}  "
                       f"CAGR {stats_bh['cagr']:.2f}%  "
                       f"Vol {stats_bh['vol']:.1f}%  "
                       f"MaxDD {stats_bh['maxdd']:.1f}%")
    ax1.semilogy(wi_t.index,  wi_t.values,  color=ORANGE, lw=1.6, ls="--",
                 label=f"Timing  ${stats_t['term']:.2f}  "
                       f"CAGR {stats_t['cagr']:.2f}%  "
                       f"Vol {stats_t['vol']:.1f}%  "
                       f"MaxDD {stats_t['maxdd']:.1f}%")
    ax1.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Growth of $1 (log)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if y >= 1 else f"${y:.2f}"
    ))
    ax1.legend(fontsize=8.5, framealpha=0.95, loc="upper left")
    ax1.grid(True, which="both", lw=0.4, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Bottom: drawdown + avg DD lines ──
    ax2.fill_between(dd_bh.index, dd_bh.values, 0, color=BLUE,   alpha=0.20)
    ax2.fill_between(dd_t.index,  dd_t.values,  0, color=ORANGE, alpha=0.30)
    ax2.plot(dd_bh.index, dd_bh.values, color=BLUE,   lw=0.8)
    ax2.plot(dd_t.index,  dd_t.values,  color=ORANGE, lw=0.8, ls="--")

    # Average drawdown dotted lines
    ax2.axhline(avg_dd_bh, color=BLUE,   lw=1.0, ls=":", zorder=4)
    ax2.axhline(avg_dd_t,  color=ORANGE, lw=1.0, ls=":", zorder=4)
    ax2.text(dd_bh.index[-1], avg_dd_bh - 0.3, f"avg {avg_dd_bh:.1f}%",
             ha="right", va="top", fontsize=7.5, color=BLUE)
    ax2.text(dd_t.index[-1],  avg_dd_t  + 0.3, f"avg {avg_dd_t:.1f}%",
             ha="right", va="bottom", fontsize=7.5, color=ORANGE)

    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.legend(
        [plt.Line2D([], [], color=BLUE, lw=1.4),
         plt.Line2D([], [], color=ORANGE, lw=1.4, ls="--"),
         plt.Line2D([], [], color="#888", lw=1.0, ls=":")],
        [f"B&H     max {stats_bh['maxdd']:.1f}%   avg {avg_dd_bh:.1f}%",
         f"Timing  max {stats_t['maxdd']:.1f}%   avg {avg_dd_t:.1f}%",
         "avg DD (dotted)"],
        fontsize=8, framealpha=0.9, loc="lower left",
    )
    ax2.grid(True, lw=0.4, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # Shared x-ticks
    start_yr = wi_bh.index[0].year
    end_yr   = wi_bh.index[-1].year
    step     = 10 if (end_yr - start_yr) > 40 else 5
    ticks    = pd.date_range(f"{start_yr}-01-01", f"{end_yr+1}-01-01", freq=f"{step}YS")
    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

    fig.tight_layout()
    fig.text(0.5, 0.003,
             f"Data: Shiller/FRED long-history spliced with SPY/IEF/BIL ETFs  |  "
             f"{port_label} SPY/IEF · annual rebalance (Jan) · Timing: price > SMA-10, 1-month lag",
             ha="center", fontsize=7.5, color="#9ca3af")

    out = OUT_DIR / f"{file_prefix}_equity_curve_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {out.name}")
    print(f"    B&H    CAGR {stats_bh['cagr']:.2f}%  Vol {stats_bh['vol']:.1f}%  "
          f"MaxDD {stats_bh['maxdd']:.1f}%  AvgDD {avg_dd_bh:.1f}%  $1→${stats_bh['term']:.2f}")
    print(f"    Timing CAGR {stats_t['cagr']:.2f}%  Vol {stats_t['vol']:.1f}%  "
          f"MaxDD {stats_t['maxdd']:.1f}%  AvgDD {avg_dd_t:.1f}%  $1→${stats_t['term']:.2f}")

    return {"bh": stats_bh, "timing": stats_t, "slug": slug, "start": start,
            "port_label": port_label, "file_prefix": file_prefix}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _dd_cls(v: float) -> str:
    if v <= -20: return "neg"
    if v <= -10: return "warn"
    return ""

def _pct(v: float, pos: bool = True) -> str:
    sign = "+" if v >= 0 and pos else ""
    return f"{sign}{v:.2f}%"

def build_html(results: list[dict], port_label: str, file_prefix: str) -> str:
    period_labels = {
        "1966-jun_2026": ("1966–Jun 2026", "Jan 1966 – Jun 2026 (60 years)"),
        "1973-jun_2026": ("1973–Jun 2026", "Jan 1973 – Jun 2026 (53 years)"),
        "2000-jun_2026": ("2000–Jun 2026", "Jan 2000 – Jun 2026 (26 years)"),
    }

    def period_table(r: dict) -> str:
        label, sub = period_labels[r["slug"]]
        bh = r["bh"]
        t  = r["timing"]
        dd_cls_bh = _dd_cls(bh["maxdd"])
        dd_cls_t  = _dd_cls(t["maxdd"])
        return f"""
<div class="period-block">
  <div class="period-label">{label}</div>
  <div class="period-sub">{sub} &middot; {bh['n_months']} months</div>
  <table>
    <thead><tr>
      <th>Strategy</th>
      <th class="num">CAGR</th>
      <th class="num">Volatility</th>
      <th class="num">Sharpe</th>
      <th class="num">Max DD</th>
      <th class="num">Avg DD</th>
      <th class="num hi">Ulcer</th>
      <th class="num hi">$1 &rarr;</th>
    </tr></thead>
    <tbody>
<tr class="bh-row"><td>Buy &amp; Hold</td>
  <td class="num">{_pct(bh['cagr'])}</td>
  <td class="num">{bh['vol']:.2f}%</td>
  <td class="num">{bh['sharpe']:.3f}</td>
  <td class="num{' ' + dd_cls_bh if dd_cls_bh else ''}">{_pct(bh['maxdd'], False)}</td>
  <td class="num">{_pct(bh['avgdd'], False)}</td>
  <td class="num">{bh['ulcer']:.2f}</td>
  <td class="num end-dollar">${bh['term']:.2f}</td></tr>
<tr class="timing-row"><td>Timing SMA-10</td>
  <td class="num">{_pct(t['cagr'])}</td>
  <td class="num">{t['vol']:.2f}%</td>
  <td class="num">{t['sharpe']:.3f}</td>
  <td class="num{' ' + dd_cls_t if dd_cls_t else ''}">{_pct(t['maxdd'], False)}</td>
  <td class="num">{_pct(t['avgdd'], False)}</td>
  <td class="num">{t['ulcer']:.2f}</td>
  <td class="num end-dollar">${t['term']:.2f}</td></tr>
    </tbody>
  </table>
</div>"""

    tables = "\n".join(period_table(r) for r in results)

    figures = ""
    for i, r in enumerate(results, 1):
        label, _ = period_labels[r["slug"]]
        bh = r["bh"]
        t  = r["timing"]
        figures += f"""
<h2>Figure {i} — {label}</h2>
<p class="chart-note">$1 invested &middot; log scale (top) &middot; drawdown + avg DD lines (bottom)
&middot; B&amp;H ${bh['term']:.2f} | Timing ${t['term']:.2f}</p>
<img class="chart-img" src="{file_prefix}_equity_curve_{r['slug']}.png"
     alt="{port_label} Buy &amp; Hold vs Timing SMA-10, {label}">
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{port_label} Portfolio (S&P 500 / IEF) — Buy &amp; Hold vs Timing (SMA-10)</title>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;font-size:14px;
        line-height:1.6;background:#fff;color:#1f2328;padding:28px 16px 48px}}
  .page{{max-width:760px;margin:0 auto}}
  h1{{font-size:1.35rem;font-weight:700;margin-bottom:4px}}
  h2{{font-size:1.05rem;font-weight:600;margin:28px 0 8px;color:#1f2328;
      border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
  .subtitle{{color:#57606a;font-size:.88rem;margin-bottom:20px}}
  .meta{{font-size:.82rem;color:#57606a;margin-bottom:16px}}
  .period-block{{margin-bottom:24px}}
  .period-label{{font-size:.9rem;font-weight:600;color:#1f2328;margin-bottom:3px}}
  .period-sub{{font-size:.8rem;color:#57606a;margin-bottom:6px}}
  table{{width:100%;border-collapse:collapse;font-size:.82rem}}
  th{{background:#f7f8fa;border:1px solid #e5e7eb;padding:6px 8px;text-align:left;
      font-weight:600;color:#57606a;white-space:nowrap}}
  td{{border:1px solid #e5e7eb;padding:5px 8px}}
  .num{{text-align:right;font-variant-numeric:tabular-nums}}
  .end-dollar{{font-weight:600}}
  .bh-row{{background:#fff}}
  .timing-row{{background:#f0f6ff}}
  .neg{{color:#cf222e;font-weight:600}}
  .warn{{color:#9a6700}}
  th.hi{{color:#3b82d4}}
  .chart-img{{width:100%;height:auto;display:block;border:1px solid #e5e7eb;
              border-radius:4px;margin-bottom:6px}}
  .chart-note{{font-size:.78rem;color:#57606a;text-align:center;margin-bottom:16px}}
  .method-box{{background:#f7f8fa;border:1px solid #e5e7eb;border-radius:4px;
               padding:14px 16px;font-size:.82rem;color:#57606a;margin-top:6px}}
  .method-box p{{margin-bottom:6px}}
  .method-box p:last-child{{margin-bottom:0}}
  .method-box b{{color:#1f2328}}
  footer{{margin-top:48px;padding-top:12px;border-top:1px solid #e5e7eb;
          text-align:center;font-size:12px;color:#57606a}}
</style>
</head>
<body>
<div class="page">

<h1>{port_label} Portfolio (S&amp;P 500 / IEF): Buy &amp; Hold vs. Timing (SMA-10)</h1>
<p class="subtitle">{port_label} S&amp;P 500 / IEF · Annual rebalancing at year-end</p>
<p class="meta">Data: Shiller S&amp;P 500 total-return index (1871–) and FRED DGS10 yield-derived bond return (1962–),
both spliced with SPY/IEF ETFs at their respective launch dates.
Cash: FRED TB3MS 3-month T-bill spliced with BIL ETF.
SMA warm-up uses 1962 prices; first reported signal Jan 1963 (1966 study uses 4 years of warm-up).
Annual rebalance: portfolio reset to {port_label} target weights every January.</p>

<h2>Performance Summary by Period</h2>
<p class="meta">Timing rule: each leg independently timed — invested when prior month-end price &gt; SMA-10, else T-bills.
1-month lag, no look-ahead. Portfolio rebalanced to {port_label} each January.</p>

{tables}

<h2>Charts</h2>
{figures}

<h2>Methodology</h2>
<div class="method-box">
  <p><b>Portfolio construction:</b> {port_label} S&amp;P 500 / IEF, rebalanced annually at the start of each January.
     Between rebalances the weights drift with market returns.</p>
  <p><b>Timing rule:</b> Each leg is timed independently. At each month-end compare that asset's price to its
     10-month SMA. Price &gt; SMA &rarr; invested. Price &le; SMA &rarr; exit to T-bills.
     Signal at close of t&minus;1, applied to return of month t (1-month lag).</p>
  <p><b>Rebalancing interaction:</b> Annual rebalance resets the <em>notional allocation</em> between legs;
     the timing signal within each leg is unaffected.</p>
  <p><b>Bond return model:</b> Yield-dependent modified duration + convexity correction on the FRED DGS10 series.
     Spliced with IEF ETF total-return from 2002.</p>
  <p><b>Cash:</b> FRED TB3MS 3-month T-bill (1934–). Spliced with BIL ETF from 2007.</p>
  <p><b>No transaction costs, taxes or leverage.</b></p>
</div>

</div>
<footer>Made with IBM Bob</footer>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data…")
    prices = load_data()
    print(f"  SPY: {prices['SPY'].dropna().index[0].date()} – {prices['SPY'].dropna().index[-1].date()}")
    print(f"  IEF: {prices['IEF'].dropna().index[0].date()} – {prices['IEF'].dropna().index[-1].date()}")

    for w_spy, w_ief, port_label, file_prefix, html_name in PORTFOLIOS:
        print(f"\n{'='*50}")
        print(f"  {port_label} Portfolio")
        print(f"{'='*50}")
        results = []
        for slug, start in WINDOWS:
            print(f"\n  {slug}")
            ret_bh, ret_timing = build_portfolio(prices, start, w_spy, w_ief)
            r = plot_window(slug, start, port_label, file_prefix, ret_bh, ret_timing)
            results.append(r)

        html = build_html(results, port_label, file_prefix)
        out  = OUT_DIR / html_name
        out.write_text(html, encoding="utf-8")
        print(f"\n  [html] saved → {out.name}")


if __name__ == "__main__":
    main()
