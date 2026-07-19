"""
compute_602020_portfolio.py
60/20/20 Portfolio: 60% S&P 500 · 20% TLT · 20% GLD
Buy & Hold vs SMA-10 Timing, annual rebalancing at year-end (every January).

SMA warm-up: data from 1963 used to seed the 10-month SMA.
Base date: Dec 1972 close → first return = Jan 1973 (no return lost).
Cash (BIL) used only as exit destination when a leg's timing signal is off.

Outputs (data/analysis/):
  602020_equity_curve_1973-jun_2026.png
  602020_equity_curve_2000-jun_2026.png
  602020-portfolio-bh-vs-timing-1973-2026.html
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
WARMUP  = "1963-01-01"
WEIGHTS = {"SPY": 0.60, "TLT": 0.20, "GLD": 0.20}
BLUE    = "#2563eb"
ORANGE  = "#e07b39"

WINDOWS = [
    ("1973-jun_2026", "1973-01-01",
     "60/20/20 Portfolio (60% SP500 · 20% TLT · 20% GLD)  |  1973–Jun 2026\n"
     "Annual rebalancing · SMA-10 timing · T-bill cash when out of market"),
    ("2000-jun_2026", "2000-01-01",
     "60/20/20 Portfolio (60% SP500 · 20% TLT · 20% GLD)  |  2000–Jun 2026\n"
     "Annual rebalancing · SMA-10 timing · T-bill cash when out of market"),
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

def build_portfolio(prices: pd.DataFrame, start: str) -> tuple[pd.Series, pd.Series]:
    assets = list(WEIGHTS.keys())

    all_cols = assets + ["BIL"]
    common = prices[all_cols[0]].dropna().index
    for c in all_cols[1:]:
        common = common.intersection(prices[c].dropna().index)
    common = common[common >= WARMUP]

    px  = {a: prices[a].reindex(common) for a in assets}
    bil = prices["BIL"].reindex(common)

    ret     = {a: px[a].pct_change() for a in assets}
    ret_bil = bil.pct_change()

    sig = {a: (px[a] > px[a].rolling(SMA).mean()).astype(float).shift(1)
           for a in assets}

    idx = common[common >= start]
    for a in assets:
        ret[a] = ret[a].reindex(idx)
        sig[a] = sig[a].reindex(idx).fillna(0.0)
    ret_bil = ret_bil.reindex(idx)

    # ── B&H with annual rebalance ──────────────────────────────────────────
    vals_bh = {a: WEIGHTS[a] for a in assets}
    ret_bh_list = []
    for dt in idx:
        if dt.month == 1 and dt != idx[0]:
            total = sum(vals_bh.values())
            for a in assets:
                vals_bh[a] = total * WEIGHTS[a]
        total    = sum(vals_bh.values())
        port_ret = sum(
            vals_bh[a] / total * (ret[a].loc[dt] if not np.isnan(ret[a].loc[dt]) else 0.0)
            for a in assets
        )
        ret_bh_list.append(port_ret)
        for a in assets:
            r = ret[a].loc[dt] if not np.isnan(ret[a].loc[dt]) else 0.0
            vals_bh[a] *= (1 + r)
    ret_bh = pd.Series(ret_bh_list, index=idx, name="ret_bh")

    # ── Timing with annual rebalance ───────────────────────────────────────
    vals_t = {a: WEIGHTS[a] for a in assets}
    ret_t_list = []
    for dt in idx:
        if dt.month == 1 and dt != idx[0]:
            total = sum(vals_t.values())
            for a in assets:
                vals_t[a] = total * WEIGHTS[a]
        total = sum(vals_t.values())
        rb    = ret_bil.loc[dt] if not np.isnan(ret_bil.loc[dt]) else 0.0
        port_ret = 0.0
        for a in assets:
            ra    = ret[a].loc[dt] if not np.isnan(ret[a].loc[dt]) else 0.0
            s     = sig[a].loc[dt]
            r_leg = s * ra + (1 - s) * rb
            port_ret += vals_t[a] / total * r_leg
            vals_t[a] *= (1 + r_leg)
        ret_t_list.append(port_ret)
    ret_timing = pd.Series(ret_t_list, index=idx, name="ret_timing")

    return ret_bh, ret_timing


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def plot_window(slug: str, title: str,
                ret_bh: pd.Series, ret_timing: pd.Series) -> dict:

    wi_bh = wealth_index(ret_bh)
    wi_t  = wealth_index(ret_timing)
    dd_bh = drawdown_series(ret_bh) * 100
    dd_t  = drawdown_series(ret_timing) * 100
    s_bh  = compute_stats(ret_bh)
    s_t   = compute_stats(ret_timing)
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
    ax2.text(dd_bh.index[-1], avg_dd_bh - 0.2, f"avg {avg_dd_bh:.1f}%",
             ha="right", va="top", fontsize=7.5, color=BLUE)
    ax2.text(dd_t.index[-1],  avg_dd_t  + 0.2, f"avg {avg_dd_t:.1f}%",
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
             "Data: Shiller/FRED long-history spliced with SPY/TLT/GLD/BIL ETFs  |  "
             "60/20/20 · annual rebalance (Jan) · Timing: price > SMA-10, 1-month lag",
             ha="center", fontsize=7.5, color="#9ca3af")

    out = OUT_DIR / f"602020_equity_curve_{slug}.png"
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

def build_html(results: list[dict]) -> str:
    period_labels = {
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
<img class="chart-img" src="602020_equity_curve_{r['slug']}.png"
     alt="60/20/20 Portfolio B&amp;H vs Timing, {label}">
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>60/20/20 Portfolio — Buy &amp; Hold vs Timing (SMA-10)</title>
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

<h1>60/20/20 Portfolio: Buy &amp; Hold vs. Timing (SMA-10)</h1>
<p class="subtitle">60% S&amp;P 500 · 20% Long Bond (TLT) · 20% Gold (GLD) · Annual rebalancing</p>
<p class="meta">Data: Shiller S&amp;P 500 TR (1871–), FRED DGS20 bond return (1962–), gold spot (1963–),
all spliced with SPY/TLT/GLD ETFs at their launch dates. Cash (BIL) used only as exit destination for timed legs.
SMA warm-up uses data from 1963; first signal Jan 1973 is based on the Dec 1972 price vs 10-month SMA — no look-ahead.</p>

<h2>Performance Summary by Period</h2>
<p class="meta">Timing rule: SPY, TLT, GLD each independently timed — invested when prior month-end price &gt; SMA-10,
else T-bills. 1-month lag, no look-ahead. Portfolio rebalanced to 60/20/20 each January.</p>

{tables}

<h2>Charts</h2>
{figures}

<h2>Methodology</h2>
<div class="method-box">
  <p><b>Portfolio construction:</b> 60% S&amp;P 500, 20% 20-year Treasury bond, 20% gold.
     Rebalanced annually at the start of each January. Between rebalances weights drift with market returns.</p>
  <p><b>Timing rule:</b> Each leg independently timed via its own 10-month SMA.
     Price &gt; SMA &rarr; invested in that asset. Price &le; SMA &rarr; that leg moves to T-bills (BIL).
     Signal at close of month t&minus;1, applied to return of month t (1-month lag, no look-ahead).</p>
  <p><b>Warm-up &amp; base date:</b> SMA seeded using monthly prices from Jan 1963 onwards.
     For the 1973 study, the first return (Jan 1973) uses the Dec 1972 → Jan 1973 price change,
     so no return month is discarded. Signal for Jan 1973 is based on Dec 1972 price vs SMA.</p>
  <p><b>Annual rebalance:</b> Notional leg values reset to 60/20/20 each January.
     The timing signal within each leg is unaffected by rebalancing.</p>
  <p><b>Bond return model:</b> FRED DGS20 yield-derived total return with convexity correction.
     Spliced with TLT ETF from 2002.</p>
  <p><b>Gold:</b> Kitco/GitHub spot price from 1963. Spliced with GLD ETF from Nov 2004.</p>
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
    for col in ["SPY", "TLT", "GLD", "BIL"]:
        s = prices[col].dropna()
        print(f"  {col}: {s.index[0].date()} – {s.index[-1].date()}")

    results = []
    for slug, start, title in WINDOWS:
        print(f"\n{slug}")
        ret_bh, ret_timing = build_portfolio(prices, start)
        r = plot_window(slug, title, ret_bh, ret_timing)
        results.append(r)

    html = build_html(results)
    out  = OUT_DIR / "602020-portfolio-bh-vs-timing-1973-2026.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n  [html] saved → {out.name}")


if __name__ == "__main__":
    main()
