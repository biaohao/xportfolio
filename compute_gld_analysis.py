"""
compute_gld_analysis.py
GLD (Gold) Buy & Hold vs Timing (SMA-10) deep-dive.

Mirrors the IEF analysis in data/analysis/ief-bh-vs-timing-1963-2026.html.

Periods studied:
  1973–Jun 2026  (full free-float gold era, uses spot-price proxy)
  2000–Jun 2026  (modern era, GLD ETF from Nov 2004)

SMA warm-up: 1963 prices used to seed the SMA. First valid signal ~Jan 1973
(after at least 10 months of data inside the reporting window).

Outputs (all in data/analysis/):
  gld_bh_timing_1973_2026.csv          — monthly rows with all series
  gld_equity_curve_1973-jun_2026.png   — equity curve + drawdown chart
  gld_equity_curve_2000-jun_2026.png   — equity curve + drawdown chart
  gld-bh-vs-timing-1973-2026.html      — HTML report (same style as IEF)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from metrics import (
    cagr, annualised_volatility, sharpe_ratio, max_drawdown,
    calmar_ratio, ulcer_index, wealth_index, drawdown_series,
)

PROC     = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMA_PERIOD  = 10
WARMUP_START = "1963-01-01"   # long-history data starts here
PERIOD_END  = "2026-06-30"

PERIODS = [
    ("1973–Jun 2026", "1973-01-01", PERIOD_END),
    ("2000–Jun 2026", "2000-01-01", PERIOD_END),
]

CHART_WINDOWS = [
    ("1973-jun_2026", "1973-01-01", PERIOD_END,
     "$1 invested January 1973 · log scale (top) · drawdown (bottom)"),
    ("2000-jun_2026", "2000-01-01", PERIOD_END,
     "$1 invested January 2000 · log scale (top) · drawdown (bottom)"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.Series, pd.Series]:
    """Return (gld_prices, tbill_returns) from the spliced monthly file."""
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True).sort_index()
    gld = df["GLD"].dropna()
    bil = df["BIL"].dropna()
    tbill_ret = bil.pct_change().dropna()
    return gld, tbill_ret


# ---------------------------------------------------------------------------
# Timing engine
# ---------------------------------------------------------------------------

def run_gld_timing(
    prices: pd.Series,
    tbill_ret: pd.Series,
    start: str,
    end: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (bh_returns, timing_returns, signal) for the given window.

    Signal computed on the full price history from WARMUP_START so the SMA
    is fully seeded.  Returns are trimmed to [start, end].
    cash-when-out: T-bill return.
    """
    # Use full history from warmup for SMA seeding
    s = prices.loc[WARMUP_START:]
    s = s.loc[:end] if end else s

    sma   = s.rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    raw_sig = (s > sma).astype(float)
    sig   = raw_sig.shift(1)          # 1-month lag: signal at t applied at t+1

    rets  = s.pct_change()
    tb    = tbill_ret.reindex(s.index).ffill().fillna(0.0)

    # Trim to reporting window
    rets_w = rets.loc[start:end].dropna()
    sig_w  = sig.reindex(rets_w.index)
    tb_w   = tb.reindex(rets_w.index).fillna(0.0)

    bh     = rets_w.copy()
    timing = rets_w.where(sig_w == 1.0, tb_w)

    return bh, timing, sig_w


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def period_metrics(ret: pd.Series, tbill: pd.Series) -> dict:
    rf = tbill.reindex(ret.index).fillna(0.0)
    c   = cagr(ret)
    v   = annualised_volatility(ret, "monthly")
    sh  = sharpe_ratio(ret, rf, "monthly")
    md  = max_drawdown(ret)
    cal = calmar_ratio(ret, "monthly")
    ui  = ulcer_index(ret)
    wi  = wealth_index(ret)
    terminal = round(float(wi.iloc[-1]), 2) if not wi.empty else float("nan")

    def _r(x, d=2):
        return round(x, d) if not np.isnan(x) else float("nan")

    return {
        "cagr":     _r(c * 100),
        "vol":      _r(v * 100),
        "sharpe":   _r(sh, 3),
        "max_dd":   _r(md * 100),
        "calmar":   _r(cal, 3),
        "ulcer":    _r(ui, 2),
        "terminal": terminal,
    }


# ---------------------------------------------------------------------------
# CSV export (same columns as ief_bh_timing_1963_2026.csv)
# ---------------------------------------------------------------------------

def build_csv_df(
    prices: pd.Series,
    tbill_ret: pd.Series,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Build the full monthly DataFrame for CSV export."""
    s   = prices.loc[WARMUP_START:end]
    sma = s.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
    raw_sig = (s > sma).astype(float)
    sig = raw_sig.shift(1)
    tb  = tbill_ret.reindex(s.index).ffill().fillna(0.0)
    rets = s.pct_change()

    # Trim to reporting window
    idx = rets.loc[start:end].dropna().index
    price_w = s.reindex(idx)
    sma_w   = sma.reindex(idx)
    sig_w   = sig.reindex(idx).fillna(0.0)
    rets_w  = rets.reindex(idx)
    tb_w    = tb.reindex(idx).fillna(0.0)

    timing_w = rets_w.where(sig_w == 1.0, tb_w)

    wi_bh     = wealth_index(rets_w)
    wi_timing = wealth_index(timing_w)
    dd_bh     = drawdown_series(rets_w) * 100
    dd_timing = drawdown_series(timing_w) * 100

    df = pd.DataFrame({
        "price":               price_w,
        "sma10":               sma_w,
        "signal":              sig_w,
        "monthly_ret_bh":      rets_w,
        "monthly_ret_timing":  timing_w,
        "equity_bh":           wi_bh,
        "equity_timing":       wi_timing,
        "drawdown_bh_pct":     dd_bh.round(4),
        "drawdown_timing_pct": dd_timing.round(4),
    })
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Chart (equity curve + drawdown) — same style as IEF PNGs
# ---------------------------------------------------------------------------

BLUE   = "#2563eb"
ORANGE = "#e07b39"

def _compute_stats(ret: pd.Series) -> dict:
    from metrics import cagr, annualised_volatility, max_drawdown
    dd = drawdown_series(ret) * 100
    return dict(
        cagr   = cagr(ret) * 100,
        vol    = annualised_volatility(ret, "monthly") * 100,
        maxdd  = max_drawdown(ret) * 100,
        avgdd  = dd.mean(),
    )

def plot_equity_drawdown(
    bh_ret: pd.Series,
    timing_ret: pd.Series,
    title: str,
    out_path: Path,
    terminal_bh: float,
    terminal_timing: float,
) -> None:
    """
    Two-panel figure: log-scale equity curve (top) + drawdown (bottom).
    Matches the sp500only chart style.
    """
    wi_bh  = wealth_index(bh_ret)
    wi_t   = wealth_index(timing_ret)
    dd_bh  = drawdown_series(bh_ret) * 100
    dd_t   = drawdown_series(timing_ret) * 100

    s_bh = _compute_stats(bh_ret)
    s_t  = _compute_stats(timing_ret)
    avg_dd_bh = s_bh["avgdd"]
    avg_dd_t  = s_t["avgdd"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.07},
        facecolor="white",
    )

    # ── Top: equity curve (log scale) ──
    ax1.semilogy(wi_bh.index, wi_bh.values, color=BLUE,   lw=1.6,
                 label=f"B&H  ${terminal_bh:.2f}  "
                       f"CAGR {s_bh['cagr']:.2f}%  "
                       f"Vol {s_bh['vol']:.1f}%  "
                       f"MaxDD {s_bh['maxdd']:.1f}%")
    ax1.semilogy(wi_t.index,  wi_t.values,  color=ORANGE, lw=1.6, ls="--",
                 label=f"Timing  ${terminal_timing:.2f}  "
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

    # ── Bottom: drawdown ──
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

    # x-axis: shared year ticks
    start_yr = wi_bh.index[0].year
    end_yr   = wi_bh.index[-1].year
    step     = 10 if (end_yr - start_yr) > 40 else 5
    ticks    = pd.date_range(f"{start_yr}-01-01", f"{end_yr+1}-01-01", freq=f"{step}YS")
    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

    fig.tight_layout()
    fig.text(0.5, 0.003,
             "Data: Kitco/GitHub gold spot spliced with GLD ETF (Nov 2004)  |  "
             "Timing: price > SMA-10, T-bill cash, 1-month lag, no look-ahead",
             ha="center", fontsize=7.5, color="#9ca3af")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [chart] saved → {out_path.name}")


# ---------------------------------------------------------------------------
# HTML report (same layout as ief-bh-vs-timing-1963-2026.html)
# ---------------------------------------------------------------------------

def fmt_pct(v: float, pos: bool = True) -> str:
    sign = "+" if v >= 0 and pos else ""
    return f"{sign}{v:.2f}%"


def td_neg(v: float) -> str:
    cls = "neg" if v < -15 else "warn" if v < 0 else ""
    return f'<td class="num{" " + cls if cls else ""}">{fmt_pct(v, pos=False)}</td>'


def build_html(
    period_stats: dict,   # {label: {bh: {...}, timing: {...}}}
    pct_invested: dict,   # {label: float}
    n_months: dict,       # {label: int}
) -> str:

    CSS = """
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;font-size:14px;
        line-height:1.6;background:#fff;color:#1f2328;padding:28px 16px 48px}
  .page{max-width:760px;margin:0 auto}
  h1{font-size:1.35rem;font-weight:700;margin-bottom:4px}
  h2{font-size:1.05rem;font-weight:600;margin:28px 0 8px;color:#1f2328;
      border-bottom:1px solid #e5e7eb;padding-bottom:4px}
  .subtitle{color:#57606a;font-size:.88rem;margin-bottom:20px}
  .meta{font-size:.82rem;color:#57606a;margin-bottom:16px}
  .period-block{margin-bottom:24px}
  .period-label{font-size:.9rem;font-weight:600;color:#1f2328;margin-bottom:3px}
  .period-sub{font-size:.8rem;color:#57606a;margin-bottom:6px}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{background:#f7f8fa;border:1px solid #e5e7eb;padding:6px 8px;text-align:left;
      font-weight:600;color:#57606a;white-space:nowrap}
  td{border:1px solid #e5e7eb;padding:5px 8px}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .end-dollar{font-weight:600}
  .bh-row{background:#fff}
  .timing-row{background:#f0f6ff}
  .neg{color:#cf222e;font-weight:600}
  .warn{color:#9a6700}
  th.hi{color:#3b82d4}
  .chart-img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;
              border-radius:4px;margin-bottom:6px}
  .chart-note{font-size:.78rem;color:#57606a;text-align:center;margin-bottom:16px}
  .method-box{background:#f7f8fa;border:1px solid #e5e7eb;border-radius:4px;
               padding:14px 16px;font-size:.82rem;color:#57606a;margin-top:6px}
  .method-box p{margin-bottom:6px}
  .method-box p:last-child{margin-bottom:0}
  .method-box b{color:#1f2328}
  .dl-box{background:#f7f8fa;border:1px solid #e5e7eb;border-radius:4px;
           padding:10px 14px;font-size:.82rem;margin:8px 0 20px}
  footer{margin-top:48px;padding-top:12px;border-top:1px solid #e5e7eb;
          text-align:center;font-size:12px;color:#57606a}
"""

    def period_table(label: str, sub: str) -> str:
        bh = period_stats[label]["bh"]
        tm = period_stats[label]["timing"]
        pct = pct_invested[label]
        rows = (
            f'<tr class="bh-row"><td>Buy &amp; Hold</td>\n'
            f'  <td class="num">{fmt_pct(bh["cagr"])}</td>\n'
            f'  <td class="num">{bh["vol"]:.2f}%</td>\n'
            f'  <td class="num">{bh["sharpe"]:.3f}</td>\n'
            f'  {td_neg(bh["max_dd"])}\n'
            f'  <td class="num">{bh["ulcer"]:.2f}</td>\n'
            f'  <td class="num end-dollar">${bh["terminal"]:,.2f}</td></tr>\n'
            f'<tr class="timing-row"><td>Timing SMA-10 ({pct:.0f}% invested)</td>\n'
            f'  <td class="num">{fmt_pct(tm["cagr"])}</td>\n'
            f'  <td class="num">{tm["vol"]:.2f}%</td>\n'
            f'  <td class="num">{tm["sharpe"]:.3f}</td>\n'
            f'  {td_neg(tm["max_dd"])}\n'
            f'  <td class="num">{tm["ulcer"]:.2f}</td>\n'
            f'  <td class="num end-dollar">${tm["terminal"]:,.2f}</td></tr>\n'
        )
        return f"""
<div class="period-block">
  <div class="period-label">{label}</div>
  <div class="period-sub">{sub}</div>
  <table>
    <thead><tr>
      <th>Strategy</th>
      <th class="num">CAGR</th><th class="num">Volatility</th>
      <th class="num">Sharpe</th><th class="num">Max DD</th>
      <th class="num hi">Ulcer Index</th><th class="num hi">$1 &rarr;</th>
    </tr></thead>
    <tbody>
{rows}    </tbody>
  </table>
</div>"""

    p1 = period_stats["1973–Jun 2026"]
    p2 = period_stats["2000–Jun 2026"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GLD &mdash; Buy &amp; Hold vs Timing (SMA-10) | 1973&ndash;2026</title>
<style>{CSS}
</style>
</head>
<body>
<div class="page">

<h1>GLD: Buy &amp; Hold vs. Timing (SMA-10)</h1>
<p class="subtitle">Gold — 53 years of monthly data, 1973–Jun 2026</p>
<p class="meta">Data: GitHub datasets/gold-prices monthly spot price (1963–present) spliced with GLD ETF (Nov 2004–present).
Gold prices were fixed under Bretton Woods until Aug 1971; free-float era used for backtest (1973+).
Cash: FRED TB3MS T-bill (1934–present) spliced with BIL ETF from 2007.
SMA warm-up uses 1963–1972 prices; first reported signal Jan 1973.</p>

<div class="dl-box">
  <b>Raw data export:</b> <code>gld_bh_timing_1973_2026.csv</code> &mdash; {n_months["1973–Jun 2026"]} monthly rows (Jan 1973&ndash;Jun 2026) with columns:
  <code>price, sma10, signal, monthly_ret_bh, monthly_ret_timing, equity_bh, equity_timing,
  drawdown_bh_pct, drawdown_timing_pct</code>
</div>

<h2>Performance Summary by Period</h2>
<p class="meta">Rule: invested when prior month-end price &gt; SMA-10, else T-bills. 1-month lag, no look-ahead.</p>

{period_table("1973–Jun 2026", "Jan 1973 – Jun 2026 (53 years)")}

{period_table("2000–Jun 2026", "Jan 2000 – Jun 2026 (26 years)")}


<h2>Figure 1 &mdash; 1973&ndash;Jun 2026</h2>
<p class="chart-note">$1 invested January 1973 &middot; log scale (top) &middot; drawdown (bottom) &middot; B&amp;H ${p1["bh"]["terminal"]:,.2f} | Timing ${p1["timing"]["terminal"]:,.2f}</p>
<img class="chart-img" src="gld_equity_curve_1973-jun_2026.png"
     alt="GLD Buy &amp; Hold vs Timing SMA-10, 1973&ndash;Jun 2026 &mdash; equity curve and drawdown">

<h2>Figure 2 &mdash; 2000&ndash;Jun 2026</h2>
<p class="chart-note">$1 invested January 2000 &middot; log scale (top) &middot; drawdown (bottom) &middot; B&amp;H ${p2["bh"]["terminal"]:,.2f} | Timing ${p2["timing"]["terminal"]:,.2f}</p>
<img class="chart-img" src="gld_equity_curve_2000-jun_2026.png"
     alt="GLD Buy &amp; Hold vs Timing SMA-10, 2000&ndash;Jun 2026 &mdash; equity curve and drawdown">


<h2>Methodology</h2>
<div class="method-box">
  <p><b>Timing rule:</b> At each month-end compare price to 10-month SMA. Price &gt; SMA &rarr; invested in GLD.
     Price &le; SMA &rarr; exit to T-bills. Signal at close of t&minus;1, applied to return of month t (1-month lag).</p>
  <p><b>SMA warm-up:</b> Full 1963&ndash;1972 prices used to seed the SMA. First signal = Jan 1973. No look-ahead bias.</p>
  <p><b>Base date:</b> For a period starting Jan YYYY, $1 is invested at Dec (YYYY&minus;1) close.
     <code>pct_change()</code> is called on the full price series; the return series is then trimmed to the window start,
     so the first month&rsquo;s return is never lost.</p>
  <p><b>Gold price series:</b> Monthly spot price from GitHub datasets/gold-prices (original source: World Gold Council / LBMA).
     Pre-1971 data reflects the Bretton Woods fixed peg ($35/oz); free-float begins Aug 1971.
     Backtest uses 1973+ only. GLD ETF (adjusted close) takes over from Nov 2004 at the splice point.</p>
  <p><b>Cash:</b> FRED TB3MS 3-month T-bill (1934&ndash;). Spliced with BIL ETF from 2007.</p>
  <p><b>No costs, taxes or leverage.</b></p>
</div>

</div>
<footer>Made with IBM Bob</footer>
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data…")
    gld, tbill_ret = load_data()
    print(f"  GLD:   {gld.index[0].date()} – {gld.index[-1].date()}  ({len(gld)} rows)")
    print(f"  TBill: {tbill_ret.index[0].date()} – {tbill_ret.index[-1].date()}")

    period_stats: dict[str, dict] = {}
    pct_invested: dict[str, float] = {}
    n_months:     dict[str, int]   = {}

    all_returns: dict[str, tuple[pd.Series, pd.Series]] = {}

    for label, start, end in PERIODS:
        bh, timing, sig = run_gld_timing(gld, tbill_ret, start, end)
        m_bh     = period_metrics(bh, tbill_ret)
        m_timing = period_metrics(timing, tbill_ret)
        period_stats[label] = {"bh": m_bh, "timing": m_timing}
        pct_invested[label] = round(float(sig.dropna().mean() * 100), 1)
        n_months[label]     = int(bh.dropna().shape[0])
        all_returns[label]  = (bh, timing)

        print(f"\n{label}  ({n_months[label]} months)")
        print(f"  B&H    CAGR={m_bh['cagr']:5.2f}%  Vol={m_bh['vol']:5.2f}%  "
              f"Sharpe={m_bh['sharpe']:.3f}  MaxDD={m_bh['max_dd']:7.2f}%  "
              f"Ulcer={m_bh['ulcer']:.2f}  $1→${m_bh['terminal']:,.2f}")
        print(f"  Timing CAGR={m_timing['cagr']:5.2f}%  Vol={m_timing['vol']:5.2f}%  "
              f"Sharpe={m_timing['sharpe']:.3f}  MaxDD={m_timing['max_dd']:7.2f}%  "
              f"Ulcer={m_timing['ulcer']:.2f}  $1→${m_timing['terminal']:,.2f}  "
              f"Invested={pct_invested[label]:.0f}%")

    # ── CSV export (primary period: 1973–Jun 2026) ──
    csv_df  = build_csv_df(gld, tbill_ret, "1973-01-01", PERIOD_END)
    csv_out = OUT_DIR / "gld_bh_timing_1973_2026.csv"
    csv_df.to_csv(csv_out)
    print(f"\n  [csv]  saved → {csv_out.name}  ({len(csv_df)} rows)")

    # ── Charts ──
    print()
    for win_id, start, end, note in CHART_WINDOWS:
        bh, timing, _ = run_gld_timing(gld, tbill_ret, start, end)
        m_bh     = period_metrics(bh, tbill_ret)
        m_timing = period_metrics(timing, tbill_ret)
        label    = next(lbl for lbl, s, e in PERIODS if s == start and e == end)
        term_bh  = m_bh["terminal"]
        term_t   = m_timing["terminal"]
        title    = (
            f"GLD: Buy & Hold vs Timing SMA-{SMA_PERIOD}  |  "
            f"{start[:4]}–Jun 2026\n{note}"
        )
        png_out = OUT_DIR / f"gld_equity_curve_{win_id}.png"
        plot_equity_drawdown(bh, timing, title, png_out, term_bh, term_t)

    # ── HTML report ──
    html = build_html(period_stats, pct_invested, n_months)
    html_out = OUT_DIR / "gld-bh-vs-timing-1973-2026.html"
    html_out.write_text(html, encoding="utf-8")
    print(f"  [html] saved → {html_out.name}")


if __name__ == "__main__":
    main()
