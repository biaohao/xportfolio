"""
compute_asset_statistics.py
5-asset class statistics: SP500, TLT, IEF, GLD, T-Bill.

Mirrors the style of data/analysis/asset-class-statistics-faber-fig4-1973-2026.html
but covers 3 periods and adds TLT.

Periods:
  1973–Jun 2026  (full free-float era)
  2000–Jun 2026  (modern / tech-era)
  1973–2012      (Faber paper window)

For each period: combined equity-curve + drawdown PNG (2-panel), plus a single
multi-period HTML report with the stats table.

Outputs (data/analysis/):
  asset_stats_equity_1973-jun_2026.png
  asset_stats_equity_2000-jun_2026.png
  asset_stats_equity_1973-2012.png
  asset-class-statistics-5asset-3period.html
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

from metrics import (
    cagr, annualised_volatility, sharpe_ratio, max_drawdown,
    calmar_ratio, ulcer_index, wealth_index, drawdown_series,
    best_worst_year, dollars_to,
)

PROC    = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette & asset config  (order = display order)
# ---------------------------------------------------------------------------
ASSETS = [
    ("SPY", "S&P 500",       "#2563eb"),
    ("TLT", "20yr Bond (TLT)","#7c3aed"),
    ("IEF", "10yr Bond (IEF)","#16a34a"),
    ("GLD", "Gold",           "#d97706"),
    ("BIL", "T-Bills",        "#9ca3af"),
]

PERIODS = [
    ("1973–Jun 2026", "1973-01-01", "2026-06-30", "1973-jun_2026"),
    ("2000–Jun 2026", "2000-01-01", "2026-06-30", "2000-jun_2026"),
    ("1973–2012",     "1973-01-01", "2012-12-31", "1973-2012"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices() -> dict[str, pd.Series]:
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True).sort_index()
    result = {}
    for ticker, _, _ in ASSETS:
        s = df[ticker].dropna()
        s.name = ticker
        result[ticker] = s
    return result


# ---------------------------------------------------------------------------
# Per-asset metrics for one window
# ---------------------------------------------------------------------------

def _tbill_ret(prices: dict[str, pd.Series], start: str, end: str) -> pd.Series:
    bil = prices["BIL"].pct_change().dropna()
    return bil.loc[start:end]


def asset_metrics(prices: dict[str, pd.Series], start: str, end: str) -> dict[str, dict]:
    tbill = _tbill_ret(prices, start, end)
    out = {}
    for ticker, label, color in ASSETS:
        s = prices[ticker].dropna()
        ret = s.pct_change()
        # pct_change on full series, then trim → preserves first month return
        ret_w = ret.loc[start:end].dropna()
        if ret_w.empty:
            continue

        rf = tbill.reindex(ret_w.index).ffill().fillna(0.0)

        # Sharpe: BIL vs itself = 0 by construction
        if ticker == "BIL":
            sh = 0.0
        else:
            sh = sharpe_ratio(ret_w, rf, "monthly")

        wi     = wealth_index(ret_w)
        dd     = drawdown_series(ret_w)
        mdd    = float(dd.min())
        avg_dd = float((dd * 100).mean())         # mean over all months (0 at peaks)
        ui     = ulcer_index(ret_w)
        c      = cagr(ret_w)
        v      = annualised_volatility(ret_w, "monthly")
        term   = float(wi.iloc[-1])
        best, worst = best_worst_year(ret_w)

        # Max-drawdown episode
        trough_idx = dd.idxmin()
        peak_idx   = wi.loc[:trough_idx].idxmax()
        # Recovery: first date AFTER trough where wi >= wi[peak]
        peak_val   = float(wi.loc[peak_idx])
        after_trough = wi.loc[trough_idx:]
        recovered  = after_trough[after_trough >= peak_val]
        recovery_date = recovered.index[0] if not recovered.empty else None

        out[ticker] = {
            "label":    label,
            "color":    color,
            "cagr":     round(c * 100, 2),
            "vol":      round(v * 100, 2),
            "sharpe":   round(sh, 3),
            "max_dd":   round(mdd * 100, 2),
            "avg_dd":   round(avg_dd, 2),
            "ulcer":    round(ui, 2),
            "terminal": round(term, 2),
            "best_yr":  best,
            "worst_yr": worst,
            "dd_peak":   peak_idx.strftime("%Y-%m-%d"),
            "dd_trough": trough_idx.strftime("%Y-%m-%d"),
            "dd_dur_m":  int(round((trough_idx - peak_idx).days / 30.44)),
            "dd_recovery": recovery_date.strftime("%Y-%m-%d") if recovery_date else None,
            "wi":  wi,
            "dd":  dd,
            "ret": ret_w,
        }
    return out


# ---------------------------------------------------------------------------
# Combined equity-curve + drawdown PNG
# ---------------------------------------------------------------------------

def plot_combined(
    metrics_by_ticker: dict[str, dict],
    period_label: str,
    start: str,
    out_path: Path,
) -> None:
    """
    Two-panel figure: log-scale growth of $1 (top) + drawdown % (bottom).
    All 5 assets on both panels.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.07},
    )

    start_year = int(start[:4])

    # ── Top: equity curves ──
    for ticker, _, _ in ASSETS:
        m = metrics_by_ticker.get(ticker)
        if m is None:
            continue
        wi   = m["wi"]
        col  = m["color"]
        term = m["terminal"]
        lbl  = f"{m['label']}  ${term:,.2f}"
        ls   = "--" if ticker == "BIL" else "-"
        lw   = 1.2 if ticker == "BIL" else 1.7
        ax1.semilogy(wi.index, wi.values, color=col, lw=lw, ls=ls, label=lbl)

    ax1.set_title(
        f"5-Asset Class Statistics  |  {period_label}\n"
        f"Growth of $1 — log scale (top)  ·  Drawdown from peak % (bottom)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax1.set_ylabel("Growth of $1 (log)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if y >= 1 else f"${y:.2f}"
    ))
    ax1.legend(fontsize=8.5, framealpha=0.92, loc="upper left",
               ncol=2 if len(ASSETS) > 3 else 1)
    ax1.grid(True, which="both", lw=0.35, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Bottom: drawdown + average drawdown horizontal lines ──
    for ticker, _, _ in ASSETS:
        m = metrics_by_ticker.get(ticker)
        if m is None:
            continue
        dd     = m["dd"] * 100
        col    = m["color"]
        avg_dd = m["avg_dd"]          # already in % (negative float)
        ls  = "--" if ticker == "BIL" else "-"
        lw  = 1.0 if ticker == "BIL" else 1.4
        ax2.fill_between(dd.index, dd.values, 0, color=col, alpha=0.12)
        ax2.plot(dd.index, dd.values, color=col, lw=lw, ls=ls)
        # Average drawdown horizontal line — dashed, slightly thicker
        ax2.axhline(avg_dd, color=col, lw=1.1, ls=":", alpha=0.85)
        # Label on the right margin
        ax2.annotate(
            f"{avg_dd:.1f}%",
            xy=(1.0, avg_dd),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=col,
            va="center",
        )

    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.grid(True, lw=0.35, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # Shared x-axis ticks
    end_year_approx = max(
        m["wi"].index[-1].year
        for _, m in metrics_by_ticker.items() if m
    )
    span = end_year_approx - start_year
    step = 5 if span > 30 else (3 if span > 15 else 2)
    tick_years = range(start_year, end_year_approx + 1, step)
    tick_dates = [pd.Timestamp(f"{y}-01-01") for y in tick_years]
    ax2.set_xticks(tick_dates)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    fig.tight_layout()
    fig.text(0.5, 0.005,
             "Data: Shiller/FRED long-history spliced with SPY/IEF/TLT/GLD/BIL ETFs  |  "
             "Monthly total return, no costs",
             ha="center", fontsize=7.5, color="#9ca3af")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [png] {out_path.name}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _pct(v: float, pos=True) -> str:
    sign = "+" if v > 0 and pos else ""
    return f"{sign}{v:.2f}%"


def _td(v: float, cls: str = "", fmt: str = "") -> str:
    if fmt:
        inner = fmt.format(v)
    else:
        inner = str(v)
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}>{inner}</td>"


def stats_table_html(metrics_by_ticker: dict[str, dict]) -> str:
    rows = ""
    for ticker, _, _ in ASSETS:
        m = metrics_by_ticker.get(ticker)
        if m is None:
            rows += f'<tr><td><span class="dot" style="background:{_}"></span>{ticker} — no data</td></tr>\n'
            continue

        cagr_cls   = "pos" if m["cagr"] >= 0 else "neg"
        sharpe_cls = "pos" if m["sharpe"] > 0 else "neg"
        dd_cls     = "neg"
        term_cls   = "pos"
        rec = m["dd_recovery"] if m["dd_recovery"] else '<em style="color:#d97706">Not yet</em>'
        dot_style  = f'background:{m["color"]}'

        rows += (
            f'<tr>'
            f'<td><span class="dot" style="{dot_style}"></span>{m["label"]} ({ticker})</td>'
            f'<td class="{cagr_cls}">{_pct(m["cagr"])}</td>'
            f'<td>{m["vol"]:.2f}%</td>'
            f'<td class="{sharpe_cls}">{m["sharpe"]:.3f}</td>'
            f'<td class="{dd_cls}">{_pct(m["max_dd"], pos=False)}</td>'
            f'<td>{m["ulcer"]:.2f}</td>'
            f'<td class="{term_cls}">${m["terminal"]:,.2f}</td>'
            f'<td class="pos">{m["best_yr"]}</td>'
            f'<td class="neg">{m["worst_yr"]}</td>'
            f'<td class="neu">{m["dd_peak"]}</td>'
            f'<td class="neu">{m["dd_trough"]}</td>'
            f'<td class="neu">{m["dd_dur_m"]}m</td>'
            f'<td class="neu">{rec}</td>'
            f'</tr>\n'
        )
    return rows


def build_html(all_metrics: dict[str, dict[str, dict]]) -> str:
    """all_metrics = {period_slug: {ticker: metrics_dict}}"""

    CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;font-size:14px;line-height:1.6;background:#fff;color:#1f2328;padding:28px 16px 48px}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:19px;font-weight:700;margin-bottom:4px}
.subtitle{color:#57606a;font-size:12px;margin-bottom:20px}
h2{font-size:14px;font-weight:700;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid #e5e7eb}
h3{font-size:12.5px;font-weight:700;margin:20px 0 6px;color:#57606a;text-transform:uppercase;letter-spacing:.04em}
.chart-wrap{background:#fafafa;border:1px solid #e5e7eb;border-radius:6px;padding:8px;margin-bottom:6px}
.chart-img{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 18px;font-size:12px;padding-left:4px}
.leg-item{display:flex;align-items:center;gap:6px}
.leg-line{width:20px;height:3px;border-radius:2px;flex-shrink:0}
table{border-collapse:collapse;width:100%;font-size:11.5px;margin:8px 0 6px}
th{background:#f7f8fa;border:1px solid #e5e7eb;padding:5px 7px;text-align:right;font-weight:600;color:#57606a;font-size:10.5px;text-transform:uppercase;white-space:nowrap;letter-spacing:.03em}
th:first-child{text-align:left;min-width:180px}
td{border:1px solid #e5e7eb;padding:5px 7px;text-align:right;white-space:nowrap}
td:first-child{text-align:left;font-weight:600}
tr:hover td{background:#f7f8fa}
.pos{color:#1a7f37;font-weight:600}
.neg{color:#cf222e;font-weight:600}
.neu{color:#57606a}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
.note{font-size:11px;color:#57606a;font-style:italic;margin:8px 0 0;line-height:1.5}
footer{text-align:center;font-size:12px;color:#57606a;margin-top:40px;padding-top:14px;border-top:1px solid #e5e7eb}
"""

    legend_html = " ".join(
        f'<span class="leg-item"><span class="leg-line" style="background:{col}"></span>{lbl} ({t})</span>'
        for t, lbl, col in ASSETS
    )

    period_blocks = ""
    for period_label, start, end, slug in PERIODS:
        m = all_metrics.get(slug, {})
        period_blocks += f"""
<h2>{period_label}</h2>
<div class="chart-wrap">
  <img class="chart-img" src="asset_stats_equity_{slug}.png"
       alt="5-asset equity curves and drawdowns, {period_label}">
</div>
<div class="legend">{legend_html}</div>

<h3>Summary Statistics &mdash; {period_label}</h3>
<table>
  <tr>
    <th>Asset</th><th>CAGR</th><th>Vol</th><th>Sharpe</th>
    <th>Max DD</th><th>Ulcer</th><th>$1 &rarr;</th>
    <th>Best Year</th><th>Worst Year</th>
    <th>DD Peak</th><th>DD Trough</th><th>Duration</th><th>Recovery</th>
  </tr>
  {stats_table_html(m)}
</table>
"""

    note = (
        "CAGR = compound annual growth rate. "
        "Vol = annualised std dev of monthly returns. "
        "Sharpe = excess return over T-bills ÷ vol (dynamic T-bill rate; BIL Sharpe = 0 by construction). "
        "Max DD = maximum peak-to-trough decline. "
        "Ulcer Index = RMS of all below-peak deviations — captures depth and duration of drawdowns. "
        "$1 → = terminal value of $1 invested at period start. "
        "DD Peak / Trough = calendar dates of the worst drawdown episode. "
        "Duration = months from peak to trough. "
        "Recovery = first month wealth returned to prior peak (\"Not yet\" = still below peak as of period end). "
        "Data: SPY = Shiller TR spliced with SPY ETF · TLT = FRED DGS20 duration+convexity spliced with TLT ETF · "
        "IEF = FRED DGS10 duration+convexity spliced with IEF ETF · GLD = LBMA/WGC gold spot spliced with GLD ETF · "
        "BIL = FRED TB3MS spliced with BIL ETF."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>5-Asset Class Statistics — 3 Periods</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>5-Asset Class Statistics — SP500 · TLT · IEF · Gold · T-Bills</h1>
<p class="subtitle">Three periods: 1973–Jun 2026 · 2000–Jun 2026 · 1973–2012 &nbsp;·&nbsp;
Monthly total-return series, no costs or taxes &nbsp;·&nbsp;
SPY = Shiller TR+ETF splice · TLT/IEF = FRED yield-derived total-return+ETF splice ·
GLD = LBMA gold spot+ETF splice · BIL = FRED TB3MS+ETF splice</p>

{period_blocks}

<p class="note">{note}</p>

<footer>Made with IBM Bob</footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading prices…")
    prices = load_prices()
    for t, _, _ in ASSETS:
        s = prices[t]
        print(f"  {t}: {s.index[0].date()} – {s.index[-1].date()}  ({len(s)} rows)")

    all_metrics: dict[str, dict[str, dict]] = {}

    for period_label, start, end, slug in PERIODS:
        print(f"\n{'='*60}")
        print(f"Period: {period_label}  ({start[:4]}–{end[:4]})")
        print(f"{'='*60}")

        m = asset_metrics(prices, start, end)
        all_metrics[slug] = m

        # Print summary
        print(f"  {'Asset':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} {'Ulcer':>7} {'$1→':>9}")
        print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*9}")
        for ticker, _, _ in ASSETS:
            if ticker not in m:
                continue
            r = m[ticker]
            print(f"  {r['label']:<22} {r['cagr']:>6.2f}% {r['vol']:>6.2f}% "
                  f"{r['sharpe']:>7.3f} {r['max_dd']:>7.2f}% {r['ulcer']:>7.2f} "
                  f"${r['terminal']:>8,.2f}")

        # Generate PNG
        png_path = OUT_DIR / f"asset_stats_equity_{slug}.png"
        plot_combined(m, period_label, start, png_path)

    # Generate HTML
    html = build_html(all_metrics)
    html_path = OUT_DIR / "asset-class-statistics-5asset-3period.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\n  [html] {html_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
