"""
compute_sp500_only_portfolio.py
100% S&P 500 — Buy & Hold vs SMA-10 Timing.

SMA warm-up: full history from 1963 used to seed the 10-month SMA.
Base date: Dec 1972 close → first return = Jan 1973 (no return lost).

Periods:
  1973–Jun 2026
  2000–Jun 2026

Outputs (data/analysis/):
  sp500only_equity_curve_1973-jun_2026.png
  sp500only_equity_curve_2000-jun_2026.png
  sp500only-bh-vs-timing-1973-2026.html
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

SMA    = 10
WARMUP = "1963-01-01"
BLUE   = "#2563eb"
ORANGE = "#e07b39"

WINDOWS = [
    ("1901-jun_2026", "1901-01-01",
     "S&P 500 (100%)  |  1901–Jun 2026\n"
     "SMA-10 timing · T-bill cash when out of market"),
    ("1973-jun_2026", "1973-01-01",
     "S&P 500 (100%)  |  1973–Jun 2026\n"
     "SMA-10 timing · T-bill cash when out of market"),
    ("2000-jun_2026", "2000-01-01",
     "S&P 500 (100%)  |  2000–Jun 2026\n"
     "SMA-10 timing · T-bill cash when out of market"),
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True).sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def build_portfolio(prices: pd.DataFrame, start: str) -> tuple[pd.Series, pd.Series]:
    # Use SPY's full index as the backbone (goes back to 1871)
    spy_full = prices["SPY"].dropna()
    bil_full = prices["BIL"].dropna()

    # Align on SPY index; BIL gets 0 return where unavailable (pre-1934)
    spy = spy_full
    bil = bil_full.reindex(spy.index).fillna(0.0)

    ret_spy = spy.pct_change()
    ret_bil = bil.pct_change().fillna(0.0)

    # SMA warm-up: restrict computation to WARMUP start but compute on full series
    sig = (spy > spy.rolling(SMA).mean()).astype(float).shift(1)

    idx     = spy.index[spy.index >= start]
    ret_spy = ret_spy.reindex(idx)
    ret_bil = ret_bil.reindex(idx)
    sig     = sig.reindex(idx).fillna(0.0)

    ret_bh     = ret_spy.rename("ret_bh")
    ret_timing = (sig * ret_spy + (1 - sig) * ret_bil).rename("ret_timing")

    return ret_bh, ret_timing


def compute_stats(ret: pd.Series) -> dict:
    dd = drawdown_series(ret) * 100
    return dict(
        cagr     = cagr(ret) * 100,
        vol      = annualised_volatility(ret, "monthly") * 100,
        sharpe   = sharpe_ratio(ret, 0.0, "monthly"),
        maxdd    = max_drawdown(ret) * 100,
        avgdd    = dd.mean(),
        ulcer    = ulcer_index(ret),
        terminal = wealth_index(ret).iloc[-1],
        n_months = len(ret),
    )


def plot_window(slug: str, title: str,
                ret_bh: pd.Series, ret_timing: pd.Series) -> dict:

    wi_bh = wealth_index(ret_bh)
    wi_t  = wealth_index(ret_timing)
    dd_bh = drawdown_series(ret_bh) * 100
    dd_t  = drawdown_series(ret_timing) * 100

    s_bh = compute_stats(ret_bh)
    s_t  = compute_stats(ret_timing)
    avg_dd_bh = s_bh["avgdd"]
    avg_dd_t  = s_t["avgdd"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.07},
        facecolor="white",
    )

    ax1.semilogy(wi_bh.index, wi_bh.values, color=BLUE,   lw=1.6,
                 label=f"B&H  ${s_bh['terminal']:.2f}  "
                       f"CAGR {s_bh['cagr']:.2f}%  "
                       f"Vol {s_bh['vol']:.1f}%  "
                       f"MaxDD {s_bh['maxdd']:.1f}%")
    ax1.semilogy(wi_t.index,  wi_t.values,  color=ORANGE, lw=1.6, ls="--",
                 label=f"Timing  ${s_t['terminal']:.2f}  "
                       f"CAGR {s_t['cagr']:.2f}%  "
                       f"Vol {s_t['vol']:.1f}%  "
                       f"MaxDD {s_t['maxdd']:.1f}%")
    ax1.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    ax1.set_ylabel("Growth of $1 (log)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if y >= 1 else f"${y:.2f}"
    ))
    ax1.legend(fontsize=8.5, framealpha=0.95, loc="upper left")
    ax1.grid(True, which="both", lw=0.4, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.fill_between(dd_bh.index, dd_bh.values, 0, color=BLUE,   alpha=0.20)
    ax2.fill_between(dd_t.index,  dd_t.values,  0, color=ORANGE, alpha=0.30)
    ax2.plot(dd_bh.index, dd_bh.values, color=BLUE,   lw=0.8)
    ax2.plot(dd_t.index,  dd_t.values,  color=ORANGE, lw=0.8, ls="--")
    ax2.axhline(avg_dd_bh, color=BLUE,   lw=1.0, ls=":", zorder=4)
    ax2.axhline(avg_dd_t,  color=ORANGE, lw=1.0, ls=":", zorder=4)
    ax2.text(dd_bh.index[-1], avg_dd_bh - 0.3, f"avg {avg_dd_bh:.1f}%",
             ha="right", va="top", fontsize=7.5, color=BLUE)
    ax2.text(dd_t.index[-1],  avg_dd_t  + 0.3, f"avg {avg_dd_t:.1f}%",
             ha="right", va="bottom", fontsize=7.5, color=ORANGE)
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.legend(
        [plt.Line2D([], [], color=BLUE,   lw=1.4),
         plt.Line2D([], [], color=ORANGE, lw=1.4, ls="--"),
         plt.Line2D([], [], color="#888", lw=1.0, ls=":")],
        [f"B&H     max {s_bh['maxdd']:.1f}%   avg {avg_dd_bh:.1f}%",
         f"Timing  max {s_t['maxdd']:.1f}%   avg {avg_dd_t:.1f}%",
         "avg DD (dotted)"],
        fontsize=8, framealpha=0.9, loc="lower left",
    )
    ax2.grid(True, lw=0.4, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

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
             "Data: Shiller S&P 500 TR spliced with SPY ETF  |  "
             "Timing: price > SMA-10, T-bill cash, 1-month lag, no look-ahead",
             ha="center", fontsize=7.5, color="#9ca3af")

    out = OUT_DIR / f"sp500only_equity_curve_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {out.name}")
    print(f"    B&H    CAGR {s_bh['cagr']:.2f}%  Vol {s_bh['vol']:.1f}%  "
          f"MaxDD {s_bh['maxdd']:.1f}%  AvgDD {avg_dd_bh:.1f}%  "
          f"Sharpe {s_bh['sharpe']:.3f}  $1→${s_bh['terminal']:.2f}")
    print(f"    Timing CAGR {s_t['cagr']:.2f}%  Vol {s_t['vol']:.1f}%  "
          f"MaxDD {s_t['maxdd']:.1f}%  AvgDD {avg_dd_t:.1f}%  "
          f"Sharpe {s_t['sharpe']:.3f}  $1→${s_t['terminal']:.2f}")

    return {"bh": s_bh, "timing": s_t, "slug": slug}


def _dd_cls(v: float) -> str:
    if v <= -40: return "neg"
    if v <= -20: return "warn"
    return ""

def _pct(v: float, pos: bool = True) -> str:
    sign = "+" if v >= 0 and pos else ""
    return f"{sign}{v:.2f}%"

def build_html(results: list[dict]) -> str:
    period_labels = {
        "1901-jun_2026": ("1901–Jun 2026", "Jan 1901 – Jun 2026 (125 years)"),
        "1973-jun_2026": ("1973–Jun 2026", "Jan 1973 – Jun 2026 (53 years)"),
        "2000-jun_2026": ("2000–Jun 2026", "Jan 2000 – Jun 2026 (26 years)"),
    }

    def period_table(r: dict) -> str:
        label, sub = period_labels[r["slug"]]
        bh = r["bh"]; t = r["timing"]
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
  <td class="num{' ' + _dd_cls(bh['maxdd']) if _dd_cls(bh['maxdd']) else ''}">{_pct(bh['maxdd'], False)}</td>
  <td class="num">{_pct(bh['avgdd'], False)}</td>
  <td class="num">{bh['ulcer']:.2f}</td>
  <td class="num end-dollar">${bh['terminal']:.2f}</td></tr>
<tr class="timing-row"><td>Timing SMA-10</td>
  <td class="num">{_pct(t['cagr'])}</td>
  <td class="num">{t['vol']:.2f}%</td>
  <td class="num">{t['sharpe']:.3f}</td>
  <td class="num{' ' + _dd_cls(t['maxdd']) if _dd_cls(t['maxdd']) else ''}">{_pct(t['maxdd'], False)}</td>
  <td class="num">{_pct(t['avgdd'], False)}</td>
  <td class="num">{t['ulcer']:.2f}</td>
  <td class="num end-dollar">${t['terminal']:.2f}</td></tr>
    </tbody>
  </table>
</div>"""

    tables  = "\n".join(period_table(r) for r in results)
    figures = ""
    for i, r in enumerate(results, 1):
        label, _ = period_labels[r["slug"]]
        bh = r["bh"]; t = r["timing"]
        figures += f"""
<h2>Figure {i} — {label}</h2>
<p class="chart-note">$1 invested &middot; log scale (top) &middot; drawdown + avg DD lines (bottom)
&middot; B&amp;H ${bh['terminal']:.2f} | Timing ${t['terminal']:.2f}</p>
<img class="chart-img" src="sp500only_equity_curve_{r['slug']}.png"
     alt="S&P 500 100% B&amp;H vs Timing, {label}">
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S&P 500 (100%) — Buy &amp; Hold vs Timing (SMA-10)</title>
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

<h1>S&amp;P 500 (100%): Buy &amp; Hold vs. Timing (SMA-10)</h1>
<p class="subtitle">100% S&amp;P 500 total return · no diversification · benchmark for portfolio comparisons</p>
<p class="meta">Data: Shiller S&amp;P 500 total-return index (1871–) spliced with SPY ETF.
Cash: FRED TB3MS 3-month T-bill spliced with BIL ETF.
SMA warm-up uses data from 1963; first signal Jan 1973 is based on the Dec 1972 price vs 10-month SMA — no look-ahead.</p>

<h2>Performance Summary by Period</h2>
<p class="meta">Timing rule: invested when prior month-end price &gt; SMA-10, else T-bills. 1-month lag, no look-ahead.</p>

{tables}

<h2>Charts</h2>
{figures}

<h2>Methodology</h2>
<div class="method-box">
  <p><b>Timing rule:</b> At each month-end compare price to 10-month SMA.
     Price &gt; SMA &rarr; invested in S&amp;P 500. Price &le; SMA &rarr; exit to T-bills.
     Signal at close of month t&minus;1, applied to return of month t (1-month lag, no look-ahead).</p>
  <p><b>Warm-up &amp; base date:</b> SMA seeded using monthly prices from Jan 1963.
     For the 1973 study, the first return (Jan 1973) uses the Dec 1972 → Jan 1973 price change — no return discarded.</p>
  <p><b>Data:</b> Shiller S&amp;P 500 TR (dividend-adjusted) spliced with SPY ETF at its 1993 launch.</p>
  <p><b>Cash:</b> FRED TB3MS 3-month T-bill (1934–). Spliced with BIL ETF from 2007.</p>
  <p><b>No transaction costs, taxes or leverage.</b></p>
</div>

</div>
<footer>Made with IBM Bob</footer>
</body></html>"""


def main() -> None:
    print("Loading data…")
    prices = load_data()
    print(f"  SPY: {prices['SPY'].dropna().index[0].date()} – {prices['SPY'].dropna().index[-1].date()}")
    print(f"  BIL: {prices['BIL'].dropna().index[0].date()} – {prices['BIL'].dropna().index[-1].date()}")

    results = []
    for slug, start, title in WINDOWS:
        print(f"\n{slug}")
        ret_bh, ret_timing = build_portfolio(prices, start)
        r = plot_window(slug, title, ret_bh, ret_timing)
        results.append(r)

    html = build_html(results)
    out  = OUT_DIR / "sp500only-bh-vs-timing-1901-2026.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n  [html] saved → {out.name}")


if __name__ == "__main__":
    main()
