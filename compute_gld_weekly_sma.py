"""
compute_gld_weekly_sma.py

GLD weekly SMA timing sensitivity sweep: SMA-24w to SMA-60w (step 2).

Data construction:
  - Warmup: monthly gold spot (1998-01-01 onward) forward-filled to weekly
  - Live:   GLD ETF daily prices (2004-11-18 onward) resampled to weekly (Friday close)
  - Splice: monthly→weekly up to the first GLD ETF week, then GLD ETF weekly
  - T-bill: FRED TB3MS monthly (tbill_fred.csv) forward-filled to weekly for
            pre-BIL period; BIL ETF daily resampled weekly from 2007-06 onward

Signal:
  - Invested when prior week's close > SMA-N, else T-bills
  - 1-week lag (signal at close of week t-1, applied to week t return)
  - No look-ahead

Periods:
  2000-Jun 2026  (weekly from 2000-01-07 onward; ~1.5yr warmup before that)
  2011-Jun 2026  (focused on the choppy bear era)

SMA values: 24, 26, 28, ..., 60  (19 values)

Outputs (data/analysis/):
  gld_w_sma{N}_{slug}.png    (38 PNGs)
  gld-weekly-sma-sensitivity.html
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
RAW     = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_START = "1998-01-01"
PERIOD_END   = "2026-06-30"
BLUE         = "#2563eb"
ORANGE       = "#e07b39"
GREEN        = "#16a34a"
RED          = "#dc2626"

SMA_RANGE = list(range(24, 62, 2))   # 24,26,...,60  → 19 values

WINDOWS = [
    ("2000-jun_2026", "2000-01-01", PERIOD_END),
    ("2011-jun_2026", "2011-01-01", PERIOD_END),
]
WINDOW_TITLES = {
    "2000-jun_2026": "2000–Jun 2026",
    "2011-jun_2026": "2011–Jun 2026",
}


# ── Data construction ─────────────────────────────────────────────────────────

def build_weekly_series() -> tuple[pd.Series, pd.Series]:
    """
    Returns (gld_weekly, tbill_weekly_return) aligned on Friday-end weekly index
    from WARMUP_START through PERIOD_END.
    """
    # ── GLD price ──────────────────────────────────────────────────────────
    # 1) Monthly gold spot → forward-fill to weekly (pre-ETF warmup)
    monthly = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                          index_col=0, parse_dates=True).sort_index()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    gld_m = monthly["GLD"].dropna()

    # Build weekly index from WARMUP_START to PERIOD_END (Fridays = W-FRI)
    weekly_idx = pd.date_range(WARMUP_START, PERIOD_END, freq="W-FRI")

    # Forward-fill monthly values to weekly
    gld_monthly_weekly = (gld_m
                          .reindex(gld_m.index.union(weekly_idx))
                          .ffill()
                          .reindex(weekly_idx))

    # 2) GLD ETF daily → resample to weekly Friday close
    gld_daily = pd.read_csv(RAW / "etf" / "GLD.csv",
                            index_col=0, parse_dates=True).sort_index()["GLD"]
    gld_etf_weekly = (gld_daily
                      .reindex(gld_daily.index.union(weekly_idx))
                      .ffill()
                      .reindex(weekly_idx))

    # 3) Splice: use monthly-filled before first ETF week, ETF after
    first_etf = gld_daily.index[0]
    first_etf_friday = weekly_idx[weekly_idx >= first_etf][0]

    gld_w = gld_monthly_weekly.copy()
    gld_w[gld_w.index >= first_etf_friday] = gld_etf_weekly[gld_w.index >= first_etf_friday]

    # ── T-bill weekly return ────────────────────────────────────────────────
    # 1) FRED monthly T-bill total-return index → weekly return
    tb_m = pd.read_csv(RAW / "longhistory" / "tbill_fred.csv",
                       index_col=0, parse_dates=True).sort_index()["TBILL_TR"]
    # Monthly returns from the TR index
    tb_m_ret = tb_m.pct_change().dropna()
    # Convert monthly return to approximate weekly: (1+r_m)^(1/4.33)-1
    tb_m_weekly = (1 + tb_m_ret) ** (1 / 4.33) - 1
    # Spread to weekly by forward-filling
    tb_weekly_from_monthly = (tb_m_weekly
                               .reindex(tb_m_weekly.index.union(weekly_idx))
                               .ffill()
                               .reindex(weekly_idx))

    # 2) BIL ETF daily → weekly return (from 2007-06 onward)
    bil_daily = pd.read_csv(RAW / "etf" / "BIL.csv",
                            index_col=0, parse_dates=True).sort_index()["BIL"]
    bil_weekly_px = (bil_daily
                     .reindex(bil_daily.index.union(weekly_idx))
                     .ffill()
                     .reindex(weekly_idx))
    bil_weekly_ret = bil_weekly_px.pct_change()

    # 3) Splice: use FRED-derived before first BIL week, BIL after
    first_bil = bil_daily.index[0]
    first_bil_friday = weekly_idx[weekly_idx >= first_bil][0]

    tb_w = tb_weekly_from_monthly.copy()
    tb_w[tb_w.index >= first_bil_friday] = bil_weekly_ret[tb_w.index >= first_bil_friday]
    tb_w = tb_w.fillna(0.0)

    return gld_w, tb_w


# ── Timing engine ─────────────────────────────────────────────────────────────

def run_timing(gld_w: pd.Series, tb_w: pd.Series,
               sma: int, start: str, end: str
               ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (bh_ret, timing_ret, signal_lagged, price_window)."""
    s    = gld_w.loc[WARMUP_START:end].dropna()
    sma_s = s.rolling(sma, min_periods=sma).mean()
    raw   = (s > sma_s).astype(float)
    sig   = raw.shift(1)   # 1-week lag

    rets = s.pct_change()
    tb   = tb_w.reindex(s.index).ffill().fillna(0.0)

    rets_w = rets.loc[start:end].dropna()
    sig_w  = sig.reindex(rets_w.index).fillna(0.0)
    tb_w2  = tb.reindex(rets_w.index).fillna(0.0)
    px_w   = s.reindex(rets_w.index)

    bh     = rets_w.copy()
    timing = rets_w.where(sig_w == 1.0, tb_w2)
    return bh, timing, sig_w, px_w


# ── Entry/Exit transitions ────────────────────────────────────────────────────

def get_transitions(sig: pd.Series) -> tuple[pd.Series, pd.Series]:
    prev  = sig.shift(1).fillna(0.0)
    entry = (sig == 1.0) & (prev == 0.0)
    exit_ = (sig == 0.0) & (prev == 1.0)
    return entry, exit_


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(ret: pd.Series) -> dict:
    dd = drawdown_series(ret) * 100
    return dict(
        cagr     = cagr(ret) * 100,
        vol      = annualised_volatility(ret, "weekly") * 100,
        sharpe   = sharpe_ratio(ret, 0.0, "weekly"),
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

    wi_bh_entry = wi_bh.reindex(entry_dates).dropna()
    wi_bh_exit  = wi_bh.reindex(exit_dates).dropna()

    win_title = WINDOW_TITLES[slug]
    start_yr  = wi_bh.index[0].year
    end_yr    = wi_bh.index[-1].year

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5),
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.07},
        facecolor="white",
    )

    # ── Top: equity curves ──────────────────────────────────────────────────
    ax1.semilogy(wi_bh.index, wi_bh.values, color=BLUE,   lw=1.4, alpha=0.75,
                 label=f"B&H  ${s_bh['terminal']:.2f}  "
                       f"CAGR {s_bh['cagr']:.2f}%  "
                       f"Vol {s_bh['vol']:.1f}%  "
                       f"MaxDD {s_bh['maxdd']:.1f}%")
    ax1.semilogy(wi_t.index,  wi_t.values,  color=ORANGE, lw=1.6, ls="--",
                 label=f"SMA-{sma}w  ${s_t['terminal']:.2f}  "
                       f"CAGR {s_t['cagr']:.2f}%  "
                       f"Vol {s_t['vol']:.1f}%  "
                       f"MaxDD {s_t['maxdd']:.1f}%")

    # Entry/exit dots on the B&H curve
    if len(wi_bh_entry):
        ax1.scatter(wi_bh_entry.index, wi_bh_entry.values,
                    color=GREEN, s=28, zorder=6, marker="o",
                    label=f"Entry ({len(wi_bh_entry)})")
    if len(wi_bh_exit):
        ax1.scatter(wi_bh_exit.index, wi_bh_exit.values,
                    color=RED,   s=28, zorder=6, marker="o",
                    label=f"Exit ({len(wi_bh_exit)})")

    ax1.set_title(
        f"GLD Weekly — SMA-{sma}w Timing  |  {win_title}\n"
        f"Dots on B&H curve — green = entry · red = exit  (SMA-{sma}w crossover)",
        fontsize=10, fontweight="bold", pad=8)
    ax1.set_ylabel("Growth of $1 (log)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if y >= 1 else f"${y:.2f}"
    ))
    ax1.legend(fontsize=7.8, framealpha=0.95, loc="upper left", ncol=2)
    ax1.grid(True, which="both", lw=0.4, color="#e5e7eb")
    ax1.tick_params(labelbottom=False, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Bottom: drawdown ────────────────────────────────────────────────────
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
         f"SMA-{sma}w  max {s_t['maxdd']:.1f}%  avg {avg_t:.1f}%",
         "avg DD (dotted)"],
        fontsize=7.8, framealpha=0.9, loc="lower left",
    )
    ax2.grid(True, lw=0.4, color="#e5e7eb")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # Shared x-ticks
    step  = 5 if (end_yr - start_yr) > 20 else 2
    ticks = pd.date_range(f"{start_yr}-01-01", f"{end_yr+1}-01-01", freq=f"{step}YS")
    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

    fig.tight_layout()
    fig.text(0.5, 0.003,
             f"Data: GLD ETF weekly (2004–) spliced with monthly gold spot fwd-filled (pre-2004)  |  "
             f"SMA-{sma}w timing · T-bill cash when out · 1-week lag · no look-ahead",
             ha="center", fontsize=7.5, color="#9ca3af")

    out = OUT_DIR / f"gld_w_sma{sma}_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ── HTML helpers ──────────────────────────────────────────────────────────────

def pct(v: float, pos: bool = True) -> str:
    sign = "+" if (v >= 0 and pos) else ""
    return f"{sign}{v:.2f}%"

CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;font-size:13px;
     line-height:1.55;background:#fff;color:#1f2328;padding:24px 14px 48px}
.page{max-width:1050px;margin:0 auto}
h1{font-size:1.2rem;font-weight:700;margin-bottom:4px}
h2{font-size:.97rem;font-weight:700;margin:28px 0 8px;color:#1f2328;
   border-bottom:2px solid #e5e7eb;padding-bottom:4px}
h3{font-size:.88rem;font-weight:600;margin:16px 0 5px;color:#57606a}
.sub{color:#57606a;font-size:.82rem;margin-bottom:22px}
.meta{font-size:.80rem;color:#57606a;margin-bottom:14px}
.stbl{width:100%;border-collapse:collapse;font-size:.76rem;margin-bottom:6px}
.stbl th{background:#f7f8fa;border:1px solid #e5e7eb;padding:5px 6px;
         text-align:center;font-weight:600;color:#57606a;white-space:nowrap}
.stbl th.left{text-align:left}
.stbl td{border:1px solid #e5e7eb;padding:4px 6px;text-align:right;
         font-variant-numeric:tabular-nums;white-space:nowrap}
.stbl td.lbl{text-align:left;font-weight:600}
.stbl tr.bh{background:#fff}
.stbl tr.tim{background:#f0f6ff}
.neg{color:#cf222e;font-weight:700}
.warn{color:#9a6700;font-weight:600}
.good{color:#15803d;font-weight:700}
.sma-block{margin-bottom:36px;border:1px solid #e5e7eb;border-radius:6px;padding:14px 16px}
.sma-title{font-size:.92rem;font-weight:700;margin-bottom:8px;color:#1f2328}
.chart-img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;
           border-radius:4px;margin-bottom:10px}
.ext{width:100%;border-collapse:collapse;font-size:.74rem;margin-bottom:6px}
.ext th{background:#f7f8fa;border:1px solid #e5e7eb;padding:3px 8px;
        text-align:left;font-weight:600;color:#57606a}
.ext td{border:1px solid #e5e7eb;padding:3px 8px}
.entry{color:#15803d;font-weight:600}
.exit{color:#cf222e;font-weight:600}
.ext-wrap{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px}
footer{margin-top:40px;padding-top:10px;border-top:1px solid #e5e7eb;
       text-align:center;font-size:11px;color:#57606a}
"""

def entry_exit_table(sig: pd.Series, px: pd.Series) -> str:
    entry_mask, exit_mask = get_transitions(sig)
    rows = []
    for dt in sig.index[entry_mask]:
        rows.append((dt, "Entry", float(px.get(dt, float("nan")))))
    for dt in sig.index[exit_mask]:
        rows.append((dt, "Exit",  float(px.get(dt, float("nan")))))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return "<p class='meta'>No transitions.</p>"
    html = ('<table class="ext"><thead><tr>'
            '<th>#</th><th>Week ending</th><th>Type</th><th>GLD $</th>'
            '</tr></thead><tbody>')
    for i, (dt, typ, pv) in enumerate(rows, 1):
        cls = "entry" if typ == "Entry" else "exit"
        html += (f'<tr><td>{i}</td>'
                 f'<td>{dt.strftime("%d %b %Y")}</td>'
                 f'<td class="{cls}">{"●"} {typ}</td>'
                 f'<td>${pv:.2f}</td></tr>')
    html += "</tbody></table>"
    return html


def summary_table(all_stats: dict) -> str:
    metrics = [
        ("CAGR",    "cagr",     True),
        ("Vol",     "vol",      False),
        ("Sharpe",  "sharpe",   True),
        ("Max DD",  "maxdd",    False),
        ("Avg DD",  "avgdd",    False),
        ("Ulcer",   "ulcer",    False),
        ("$1 →",    "terminal", True),
    ]
    smas = sorted(all_stats.keys())
    html  = '<table class="stbl"><thead>'
    html += '<tr><th class="left">Strategy</th><th class="left">Metric</th>'
    for s in smas:
        html += f'<th>SMA-{s}w</th>'
    html += '</tr></thead><tbody>'
    for label, key, higher_better in metrics:
        # B&H row (same for all SMAs — just show once per metric)
        html += f'<tr class="bh"><td class="lbl">B&amp;H</td><td>{label}</td>'
        for s in smas:
            v = all_stats[s]["bh"][key]
            html += _fmt_cell(v, key)
        html += '</tr>'
        # Timing row
        html += f'<tr class="tim"><td class="lbl">SMA-Nw</td><td>{label}</td>'
        for s in smas:
            v  = all_stats[s]["tim"][key]
            bv = all_stats[s]["bh"][key]
            improved = (v > bv) if higher_better else (v < bv)
            cls = ' class="good"' if improved else (' class="neg"' if v != bv else "")
            html += _fmt_cell(v, key, cls)
        html += '</tr>'
    html += '</tbody></table>'
    return html

def _fmt_cell(v: float, key: str, cls: str = "") -> str:
    if key in ("cagr", "maxdd", "avgdd"):
        return f'<td{cls}>{pct(v, pos=(key=="cagr"))}</td>'
    elif key == "terminal":
        return f'<td{cls}>${v:.2f}</td>'
    elif key == "sharpe":
        return f'<td{cls}>{v:.3f}</td>'
    else:
        return f'<td{cls}>{v:.2f}</td>'


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building weekly GLD + T-bill series…")
    gld_w, tb_w = build_weekly_series()
    print(f"  GLD weekly: {gld_w.dropna().index[0].date()} – {gld_w.dropna().index[-1].date()}  ({gld_w.dropna().shape[0]} weeks)")
    print(f"  T-bill weekly: {tb_w.dropna().index[0].date()} – {tb_w.dropna().index[-1].date()}")

    # results[slug][sma] = {bh, tim, sig, px}
    results  = {w[0]: {} for w in WINDOWS}
    all_pngs = {w[0]: {} for w in WINDOWS}

    for slug, start, end in WINDOWS:
        print(f"\n── {WINDOW_TITLES[slug]} ──")
        for sma in SMA_RANGE:
            bh, tim, sig, px = run_timing(gld_w, tb_w, sma, start, end)
            s_bh = stats(bh)
            s_t  = stats(tim)
            results[slug][sma] = {"bh": s_bh, "tim": s_t, "sig": sig, "px": px}

            png = plot_chart(slug, sma, bh, tim, sig, px)
            all_pngs[slug][sma] = png

            n_e = int(get_transitions(sig)[0].sum())
            n_x = int(get_transitions(sig)[1].sum())
            print(f"  SMA-{sma:2d}w  B&H {s_bh['cagr']:+.2f}%  "
                  f"Tim {s_t['cagr']:+.2f}%  "
                  f"MaxDD_bh {s_bh['maxdd']:+.1f}%  "
                  f"MaxDD_tim {s_t['maxdd']:+.1f}%  "
                  f"entries={n_e} exits={n_x}")

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
            path  = all_pngs[slug][sma]
            chart = Image.open(path).convert("RGBA")
            W, H  = chart.size
            wm_w  = int(W * 0.15)
            wm_h  = int(wm_clean.height * wm_w / wm_clean.width)
            wm    = wm_clean.resize((wm_w, wm_h), Image.LANCZOS)
            wa    = np.array(wm).astype(float)
            wa[:,:,3] *= 0.30
            wm    = Image.fromarray(wa.astype(np.uint8))
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
        win_sections += f'<h3>Summary across SMA-24w to SMA-60w (step 2)</h3>\n'
        win_sections += summary_table({s: results[slug][s] for s in SMA_RANGE})
        win_sections += '<p class="meta" style="margin-top:5px;margin-bottom:18px">Green = timing improves vs B&amp;H · Red = worsens · Weekly frequency · 1-week lag</p>\n'

        for sma in SMA_RANGE:
            r    = results[slug][sma]
            sig  = r["sig"]
            px   = r["px"]
            s_bh = r["bh"]
            s_t  = r["tim"]
            n_e  = int(get_transitions(sig)[0].sum())
            n_x  = int(get_transitions(sig)[1].sum())
            png_name = all_pngs[slug][sma].name

            win_sections += f"""
<div class="sma-block">
<div class="sma-title">SMA-{sma}w &nbsp;·&nbsp; {win_label}
  &nbsp;|&nbsp; B&amp;H: CAGR {pct(s_bh['cagr'])} · MaxDD {pct(s_bh['maxdd'], False)}
  &nbsp;|&nbsp; Timing: CAGR {pct(s_t['cagr'])} · MaxDD {pct(s_t['maxdd'], False)}
  &nbsp;|&nbsp; {n_e} entries · {n_x} exits
</div>
<img class="chart-img" src="{png_name}" alt="GLD weekly SMA-{sma}w {win_label}">
<h3>Entry / Exit Signals — SMA-{sma}w, {win_label}</h3>
{entry_exit_table(sig, px)}
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GLD Weekly SMA Sensitivity — SMA-24w to SMA-60w</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">

<h1>GLD — Weekly SMA Timing Sensitivity: SMA-24w to SMA-60w</h1>
<p class="sub">Weekly GLD prices · SMA-24w to SMA-60w (step 2, 19 values) · 2 periods ·
Green/red dots on B&amp;H curve = entry/exit signals · T-bill cash when out · 1-week lag</p>

<p class="meta"><b>Motivation:</b> Monthly SMA-10 (~43 weeks) may be too slow to react to
gold's frequent trend reversals. Weekly signals allow finer-grained entry/exit timing.
SMA-24w (~6 months) to SMA-60w (~14 months) brackets the monthly SMA-10 equivalent
(~43 weeks) and tests whether faster or slower weekly signals improve risk-adjusted returns
during the choppy 2012–2021 gold bear market and the broader 2000–2026 period.</p>

{win_sections}

<h2>Methodology</h2>
<div style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:4px;
            padding:12px 14px;font-size:.80rem;color:#57606a">
  <p><b>Price series:</b> GLD ETF daily prices (Nov 2004–) resampled to Friday weekly close.
     Pre-2004 warmup: monthly gold spot price forward-filled to weekly from Jan 1998.</p>
  <p style="margin-top:5px"><b>T-bill proxy:</b> FRED TB3MS monthly total-return index
     converted to weekly (geometric) and forward-filled for pre-2007 period.
     BIL ETF daily prices resampled to weekly returns from Jun 2007 onward.</p>
  <p style="margin-top:5px"><b>Signal:</b> Invested when prior week's close &gt; SMA-N weeks,
     else T-bills. 1-week lag — signal at Friday close of week t−1 applied to week t. No look-ahead.</p>
  <p style="margin-top:5px"><b>SMA warm-up:</b> Weekly series starts Jan 1998, giving
     ~100 weeks of history before the first 2000 reporting date — fully seeded for all SMA values.</p>
  <p style="margin-top:5px"><b>Annualisation:</b> Weekly returns annualised using 52 weeks/year.
     CAGR computed from compounded weekly returns.</p>
  <p style="margin-top:5px"><b>Entry dot:</b> signal 0→1. <b>Exit dot:</b> signal 1→0.
     Both plotted on the B&amp;H equity curve at the signal week's price level.</p>
</div>

</div>
<footer>Made with IBM Bob</footer>
</body></html>"""

    out = OUT_DIR / "gld-weekly-sma-sensitivity.html"
    out.write_text(html, encoding="utf-8")
    print(f"  [html] saved → {out.name}")
    print(f"\nTotal PNGs: {len(SMA_RANGE) * len(WINDOWS)}")


if __name__ == "__main__":
    main()
