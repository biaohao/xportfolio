"""
compute_gld_sma_sensitivity.py

GLD SMA sensitivity sweep: SMA-6 through SMA-15.
For each SMA period and each of 3 chart windows:
  - Equity curve (log scale, top)
  - Drawdown panel (bottom)
  - Green dots = entry signals, Red dots = exit signals on the price chart

3 periods × 10 SMA values = 30 PNGs
1 HTML report with stats table + entry/exit tables per SMA

Outputs (data/analysis/):
  gld_sma{N}_{slug}.png           (30 files)
  btc-sma-sensitivity-6-to-15.html   (existing filename reused)
  gld-sma-sensitivity-6-to-15.html   (new report)
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

ROOT    = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from metrics import (
    cagr, annualised_volatility, sharpe_ratio, max_drawdown,
    ulcer_index, wealth_index, drawdown_series,
)

PROC    = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP  = "1963-01-01"
END     = "2026-06-30"
BLUE    = "#2563eb"
ORANGE  = "#e07b39"
GREEN   = "#16a34a"
RED     = "#dc2626"

SMA_RANGE = list(range(6, 16))   # 6..15

WINDOWS = [
    ("1973-jun_2026", "1973-01-01", END),
    ("2000-jun_2026", "2000-01-01", END),
    ("2011-jun_2026", "2011-01-01", END),
]

WINDOW_TITLES = {
    "1973-jun_2026": "1973–Jun 2026",
    "2000-jun_2026": "2000–Jun 2026",
    "2011-jun_2026": "2011–Jun 2026",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True).sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    gld = df["GLD"].dropna()
    bil = df["BIL"].dropna()
    return gld, bil.pct_change().dropna()


# ── Timing engine ─────────────────────────────────────────────────────────────

def run_timing(prices: pd.Series, tbill: pd.Series,
               sma: int, start: str, end: str
               ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (bh_ret, timing_ret, signal_lagged, price_window)."""
    s     = prices.loc[WARMUP:end]
    sma_s = s.rolling(sma, min_periods=sma).mean()
    raw   = (s > sma_s).astype(float)
    sig   = raw.shift(1)

    rets = s.pct_change()
    tb   = tbill.reindex(s.index).ffill().fillna(0.0)

    rets_w = rets.loc[start:end].dropna()
    sig_w  = sig.reindex(rets_w.index).fillna(0.0)
    tb_w   = tb.reindex(rets_w.index).fillna(0.0)
    px_w   = s.reindex(rets_w.index)

    bh     = rets_w.copy()
    timing = rets_w.where(sig_w == 1.0, tb_w)
    return bh, timing, sig_w, px_w


# ── Entry/Exit detection ──────────────────────────────────────────────────────

def get_transitions(sig: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (entry_dates_prices_index, exit_dates_prices_index) as boolean masks."""
    prev   = sig.shift(1).fillna(0.0)
    entry  = (sig == 1.0) & (prev == 0.0)   # 0→1
    exit_  = (sig == 0.0) & (prev == 1.0)   # 1→0
    return entry, exit_


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(ret: pd.Series) -> dict:
    dd = drawdown_series(ret) * 100
    return dict(
        cagr     = cagr(ret) * 100,
        vol      = annualised_volatility(ret, "monthly") * 100,
        sharpe   = sharpe_ratio(ret, 0.0, "monthly"),
        maxdd    = max_drawdown(ret) * 100,
        avgdd    = dd.mean(),
        ulcer    = ulcer_index(ret),
        terminal = wealth_index(ret).iloc[-1],
        n        = len(ret),
    )


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_chart(slug: str, sma: int,
               bh_ret: pd.Series, tim_ret: pd.Series,
               sig: pd.Series, px: pd.Series) -> Path:

    wi_bh = wealth_index(bh_ret)
    wi_t  = wealth_index(tim_ret)
    dd_bh = drawdown_series(bh_ret) * 100
    dd_t  = drawdown_series(tim_ret) * 100
    s_bh  = stats(bh_ret)
    s_t   = stats(tim_ret)

    entry_mask, exit_mask = get_transitions(sig)
    entry_dates = sig.index[entry_mask]
    exit_dates  = sig.index[exit_mask]

    # Map entry/exit dates to wealth-index values for dot placement
    wi_entry = wi_t.reindex(entry_dates).dropna()
    wi_exit  = wi_t.reindex(exit_dates).dropna()
    # Also map to price series for alternate placement check
    px_entry = px.reindex(entry_dates).dropna()
    px_exit  = px.reindex(exit_dates).dropna()

    win_title = WINDOW_TITLES[slug]
    start_yr  = wi_bh.index[0].year
    end_yr    = wi_bh.index[-1].year

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5),
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.07},
        facecolor="white",
    )

    # ── Top: equity curves ───────────────────────────────────────────────────
    ax1.semilogy(wi_bh.index, wi_bh.values, color=BLUE,   lw=1.4, alpha=0.7,
                 label=f"B&H  ${s_bh['terminal']:.2f}  "
                       f"CAGR {s_bh['cagr']:.2f}%  "
                       f"Vol {s_bh['vol']:.1f}%  "
                       f"MaxDD {s_bh['maxdd']:.1f}%")
    ax1.semilogy(wi_t.index,  wi_t.values,  color=ORANGE, lw=1.6, ls="--",
                 label=f"SMA-{sma}  ${s_t['terminal']:.2f}  "
                       f"CAGR {s_t['cagr']:.2f}%  "
                       f"Vol {s_t['vol']:.1f}%  "
                       f"MaxDD {s_t['maxdd']:.1f}%")

    # Entry/Exit dots anchored on the B&H (GLD price) equity curve
    wi_bh_entry = wi_bh.reindex(entry_dates).dropna()
    wi_bh_exit  = wi_bh.reindex(exit_dates).dropna()
    if len(wi_bh_entry):
        ax1.scatter(wi_bh_entry.index, wi_bh_entry.values,
                    color=GREEN, s=40, zorder=6, marker="o",
                    label=f"Entry ({len(wi_bh_entry)})")
    if len(wi_bh_exit):
        ax1.scatter(wi_bh_exit.index, wi_bh_exit.values,
                    color=RED,   s=40, zorder=6, marker="o",
                    label=f"Exit ({len(wi_bh_exit)})")

    ax1.set_title(
        f"GLD — SMA-{sma} Timing  |  {win_title}\n"
        f"Dots on B&H curve — green = entry · red = exit  (SMA-{sma} crossover)",
        fontsize=10, fontweight="bold", pad=8)
    ax1.set_ylabel("Growth of $1 (log)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if y >= 1 else f"${y:.2f}"
    ))
    ax1.legend(fontsize=7.8, framealpha=0.95, loc="upper left", ncol=2)
    ax1.grid(True, which="both", lw=0.4, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Bottom: drawdown ─────────────────────────────────────────────────────
    avg_bh = s_bh["avgdd"]
    avg_t  = s_t["avgdd"]

    ax2.fill_between(dd_bh.index, dd_bh.values, 0, color=BLUE,   alpha=0.20)
    ax2.fill_between(dd_t.index,  dd_t.values,  0, color=ORANGE, alpha=0.30)
    ax2.plot(dd_bh.index, dd_bh.values, color=BLUE,   lw=0.8)
    ax2.plot(dd_t.index,  dd_t.values,  color=ORANGE, lw=0.8, ls="--")
    ax2.axhline(avg_bh, color=BLUE,   lw=0.9, ls=":", zorder=4)
    ax2.axhline(avg_t,  color=ORANGE, lw=0.9, ls=":", zorder=4)
    ax2.text(dd_bh.index[-1], avg_bh - 0.3, f"avg {avg_bh:.1f}%",
             ha="right", va="top",    fontsize=7, color=BLUE)
    ax2.text(dd_t.index[-1],  avg_t  + 0.3, f"avg {avg_t:.1f}%",
             ha="right", va="bottom", fontsize=7, color=ORANGE)

    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.legend(
        [plt.Line2D([], [], color=BLUE,   lw=1.4),
         plt.Line2D([], [], color=ORANGE, lw=1.4, ls="--"),
         plt.Line2D([], [], color="#888", lw=0.9, ls=":")],
        [f"B&H    max {s_bh['maxdd']:.1f}%  avg {avg_bh:.1f}%",
         f"SMA-{sma}  max {s_t['maxdd']:.1f}%  avg {avg_t:.1f}%",
         "avg DD (dotted)"],
        fontsize=7.8, framealpha=0.9, loc="lower left",
    )
    ax2.grid(True, lw=0.4, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # Shared x-ticks
    step  = 10 if (end_yr - start_yr) > 40 else 5 if (end_yr - start_yr) > 20 else 2
    ticks = pd.date_range(f"{start_yr}-01-01", f"{end_yr+1}-01-01", freq=f"{step}YS")
    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

    fig.tight_layout()
    fig.text(0.5, 0.003,
             f"Data: Kitco/GitHub gold spot spliced with GLD ETF (Nov 2004)  |  "
             f"Timing: price > SMA-{sma}, T-bill cash, 1-month lag, no look-ahead",
             ha="center", fontsize=7.5, color="#9ca3af")

    out = OUT_DIR / f"gld_sma{sma}_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── HTML helpers ──────────────────────────────────────────────────────────────

def pct(v: float, pos: bool = True) -> str:
    sign = "+" if (v >= 0 and pos) else ""
    return f"{sign}{v:.2f}%"

def dd_cls(v: float) -> str:
    if v <= -40: return " neg"
    if v <= -20: return " warn"
    return ""

CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;font-size:13px;
     line-height:1.55;background:#fff;color:#1f2328;padding:24px 14px 48px}
.page{max-width:1000px;margin:0 auto}
h1{font-size:1.2rem;font-weight:700;margin-bottom:4px}
h2{font-size:.97rem;font-weight:700;margin:28px 0 8px;color:#1f2328;
   border-bottom:2px solid #e5e7eb;padding-bottom:4px}
h3{font-size:.88rem;font-weight:600;margin:18px 0 6px;color:#57606a}
.sub{color:#57606a;font-size:.82rem;margin-bottom:22px}
.meta{font-size:.80rem;color:#57606a;margin-bottom:14px}

/* summary table */
.stbl{width:100%;border-collapse:collapse;font-size:.78rem;margin-bottom:6px}
.stbl th{background:#f7f8fa;border:1px solid #e5e7eb;padding:5px 7px;
         text-align:center;font-weight:600;color:#57606a;white-space:nowrap}
.stbl th.left{text-align:left}
.stbl td{border:1px solid #e5e7eb;padding:4px 7px;text-align:right;
         font-variant-numeric:tabular-nums;white-space:nowrap}
.stbl td.lbl{text-align:left;font-weight:600}
.stbl tr.bh{background:#fff}
.stbl tr.tim{background:#f0f6ff}
.neg{color:#cf222e;font-weight:700}
.warn{color:#9a6700;font-weight:600}
.good{color:#15803d;font-weight:700}

/* per-SMA block */
.sma-block{margin-bottom:36px;border:1px solid #e5e7eb;border-radius:6px;
           padding:14px 16px}
.sma-title{font-size:.95rem;font-weight:700;margin-bottom:8px;color:#1f2328}
.chart-img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;
           border-radius:4px;margin-bottom:10px}

/* entry/exit table */
.ext{width:100%;border-collapse:collapse;font-size:.76rem;margin-bottom:6px}
.ext th{background:#f7f8fa;border:1px solid #e5e7eb;padding:4px 8px;
        text-align:left;font-weight:600;color:#57606a}
.ext td{border:1px solid #e5e7eb;padding:3px 8px}
.entry{color:#15803d;font-weight:600}
.exit {color:#cf222e;font-weight:600}

footer{margin-top:40px;padding-top:10px;border-top:1px solid #e5e7eb;
       text-align:center;font-size:11px;color:#57606a}
"""

def entry_exit_table(sig: pd.Series, px: pd.Series, slug: str) -> str:
    entry_mask, exit_mask = get_transitions(sig)
    rows = []
    # merge and sort
    for dt in sig.index[entry_mask]:
        rows.append((dt, "Entry", px.get(dt, float("nan"))))
    for dt in sig.index[exit_mask]:
        rows.append((dt, "Exit",  px.get(dt, float("nan"))))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return "<p class='meta'>No transitions in this window.</p>"

    html = ('<table class="ext"><thead><tr>'
            '<th>#</th><th>Date</th><th>Type</th><th>GLD Price</th>'
            '</tr></thead><tbody>')
    for i, (dt, typ, px_val) in enumerate(rows, 1):
        cls  = "entry" if typ == "Entry" else "exit"
        dot  = "●"
        html += (f'<tr><td>{i}</td>'
                 f'<td>{dt.strftime("%b %Y")}</td>'
                 f'<td class="{cls}">{dot} {typ}</td>'
                 f'<td>${px_val:.2f}</td></tr>')
    html += "</tbody></table>"
    return html


def summary_table_html(all_stats: dict) -> str:
    """Cross-SMA summary for one window: rows = metrics, cols = SMA values."""
    metrics = [
        ("CAGR",     "cagr",  True),
        ("Vol",      "vol",   False),
        ("Sharpe",   "sharpe",True),
        ("Max DD",   "maxdd", False),
        ("Avg DD",   "avgdd", False),
        ("Ulcer",    "ulcer", False),
        ("$1 →",     "terminal", True),
    ]
    # all_stats[sma] = {"bh": {...}, "tim": {...}}
    smas = sorted(all_stats.keys())

    html  = '<table class="stbl"><thead>'
    html += '<tr><th class="left">Strategy</th><th class="left">Metric</th>'
    for s in smas:
        html += f'<th>SMA-{s}</th>'
    html += '</tr></thead><tbody>'

    for label, key, higher_better in metrics:
        # B&H row
        html += f'<tr class="bh"><td class="lbl">B&amp;H</td><td>{label}</td>'
        for s in smas:
            v = all_stats[s]["bh"][key]
            if key in ("cagr", "maxdd", "avgdd"):
                html += f'<td>{pct(v, pos=(key=="cagr"))}</td>'
            elif key == "terminal":
                html += f'<td>${v:.2f}</td>'
            else:
                html += f'<td>{v:.3f}</td>' if key == "sharpe" else f'<td>{v:.2f}</td>'
        html += '</tr>'

        # Timing row
        html += f'<tr class="tim"><td class="lbl">SMA-N</td><td>{label}</td>'
        for s in smas:
            v   = all_stats[s]["tim"][key]
            bv  = all_stats[s]["bh"][key]
            # colour: green if improved vs B&H
            improved = (v > bv) if higher_better else (v < bv)
            cls = ' class="good"' if improved else (' class="neg"' if (not improved and v != bv) else "")
            if key in ("cagr", "maxdd", "avgdd"):
                html += f'<td{cls}>{pct(v, pos=(key=="cagr"))}</td>'
            elif key == "terminal":
                html += f'<td{cls}>${v:.2f}</td>'
            else:
                html += f'<td{cls}>{v:.3f}</td>' if key == "sharpe" else f'<td{cls}>{v:.2f}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data…")
    gld, tbill = load_data()
    print(f"  GLD: {gld.index[0].date()} – {gld.index[-1].date()}")

    # Store results: results[slug][sma] = {bh, tim, sig, px, entry, exit}
    results: dict[str, dict] = {w[0]: {} for w in WINDOWS}
    all_pngs: dict[str, dict] = {w[0]: {} for w in WINDOWS}

    for slug, start, end in WINDOWS:
        print(f"\n── {WINDOW_TITLES[slug]} ──")
        for sma in SMA_RANGE:
            bh, tim, sig, px = run_timing(gld, tbill, sma, start, end)
            s_bh = stats(bh)
            s_t  = stats(tim)
            results[slug][sma] = {"bh": s_bh, "tim": s_t, "sig": sig, "px": px}

            png = plot_chart(slug, sma, bh, tim, sig, px)
            all_pngs[slug][sma] = png
            n_entry = int(get_transitions(sig)[0].sum())
            n_exit  = int(get_transitions(sig)[1].sum())
            print(f"  SMA-{sma:2d}  B&H {s_bh['cagr']:+.2f}%  "
                  f"Tim {s_t['cagr']:+.2f}%  "
                  f"MaxDD_bh {s_bh['maxdd']:+.1f}%  "
                  f"MaxDD_tim {s_t['maxdd']:+.1f}%  "
                  f"entries={n_entry} exits={n_exit}")

    # ── Watermark ──────────────────────────────────────────────────────────
    print("\nApplying watermarks…")
    from PIL import Image

    wm_src = Image.open(OUT_DIR / "@biaohao.png").convert("RGBA")
    wm_arr = np.array(wm_src)
    r, g, b, a = wm_arr[:,:,0], wm_arr[:,:,1], wm_arr[:,:,2], wm_arr[:,:,3]
    wm_arr[(r > 220) & (g > 220) & (b > 220), 3] = 0
    wm_clean = Image.fromarray(wm_arr)

    for slug in results:
        for sma in SMA_RANGE:
            path = all_pngs[slug][sma]
            chart = Image.open(path).convert("RGBA")
            W, H = chart.size
            wm_w = int(W * 0.15)
            wm_h = int(wm_clean.height * wm_w / wm_clean.width)
            wm = wm_clean.resize((wm_w, wm_h), Image.LANCZOS)
            wa = np.array(wm).astype(float)
            wa[:,:,3] *= 0.30
            wm = Image.fromarray(wa.astype(np.uint8))
            margin = int(16 * W / 1400)
            canvas = chart.copy()
            canvas.paste(wm, (W - wm_w - margin, H - wm_h - margin), mask=wm)
            canvas.save(path)

    # ── HTML ───────────────────────────────────────────────────────────────
    print("\nBuilding HTML…")

    win_sections = ""
    for slug, start, end in WINDOWS:
        win_label = WINDOW_TITLES[slug]
        win_sections += f'<h2>{win_label}</h2>\n'
        win_sections += f'<h3>Summary: B&amp;H vs Timing across SMA-6 to SMA-15</h3>\n'
        win_sections += summary_table_html({s: results[slug][s] for s in SMA_RANGE})
        win_sections += '<p class="meta" style="margin-top:6px;margin-bottom:18px">Green = timing improves metric vs B&amp;H. Red = timing worsens.</p>\n'

        for sma in SMA_RANGE:
            r   = results[slug][sma]
            sig = r["sig"]
            px  = r["px"]
            s_bh = r["bh"]
            s_t  = r["tim"]
            n_e  = int(get_transitions(sig)[0].sum())
            n_x  = int(get_transitions(sig)[1].sum())
            png_name = all_pngs[slug][sma].name

            win_sections += f'''
<div class="sma-block">
<div class="sma-title">SMA-{sma} &nbsp;·&nbsp; {win_label}
  &nbsp;|&nbsp; B&amp;H: CAGR {pct(s_bh["cagr"])}  MaxDD {pct(s_bh["maxdd"], False)}
  &nbsp;|&nbsp; Timing: CAGR {pct(s_t["cagr"])}  MaxDD {pct(s_t["maxdd"], False)}
  &nbsp;|&nbsp; {n_e} entries · {n_x} exits
</div>
<img class="chart-img" src="{png_name}" alt="GLD SMA-{sma} {win_label}">
<h3>Entry / Exit Signals &mdash; SMA-{sma}, {win_label}</h3>
{entry_exit_table(sig, px, slug)}
</div>
'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GLD SMA Sensitivity — SMA-6 to SMA-15</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">

<h1>GLD — SMA Timing Sensitivity: SMA-6 to SMA-15</h1>
<p class="sub">Gold buy &amp; hold vs SMA-N timing across 3 periods.
Green/red dots on the B&amp;H (GLD price) curve — green = entry · red = exit.
T-bill cash when out of market · 1-month lag · no look-ahead.</p>

<p class="meta">
<b>Why did timing struggle for GLD in 2012–2021?</b><br>
Gold peaked in Sep 2011 (~$1,900/oz) and entered a long, choppy bear market that lasted until
late 2018. During this period the price repeatedly crossed its SMA both ways — a classic
whipsaw environment — causing timing models to repeatedly exit near lows and re-enter near
temporary recoveries, accumulating trading losses relative to simply holding T-bills.
Longer SMAs generate fewer signals (less whipsaw) but also exit later (more of the initial
drawdown absorbed). Shorter SMAs react faster but create more false signals.
The entry/exit tables below show exactly when each SMA fired.
</p>

{win_sections}

<h2>Methodology</h2>
<div style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:4px;
            padding:12px 14px;font-size:.80rem;color:#57606a">
  <p><b>Signal:</b> Invest in GLD when prior month-end price &gt; SMA-N, else hold T-bills.
     1-month lag (signal at close of month t−1, applied to return of month t). No look-ahead.</p>
  <p style="margin-top:6px"><b>SMA warm-up:</b> Full history from Jan 1963 used to seed the SMA
     before the first reporting month. First signal for the 1973 window is based on Dec 1972 price.</p>
  <p style="margin-top:6px"><b>Data:</b> Kitco/GitHub monthly gold spot (1963–) spliced with GLD ETF (Nov 2004–).
     Cash: FRED TB3MS 3-month T-bill spliced with BIL ETF.</p>
  <p style="margin-top:6px"><b>Entry dot:</b> signal transitions from 0→1 (invested).
     <b>Exit dot:</b> signal transitions from 1→0 (exit to T-bills). Dots plotted on the B&amp;H (GLD price) equity curve so you can see the exact price level at which each signal fired.</p>
</div>

</div>
<footer>Made with IBM Bob</footer>
</body></html>"""

    out = OUT_DIR / "gld-sma-sensitivity-6-to-15.html"
    out.write_text(html, encoding="utf-8")
    print(f"  [html] saved → {out.name}")
    print(f"\nTotal PNGs: {len(SMA_RANGE) * len(WINDOWS)}")


if __name__ == "__main__":
    main()
