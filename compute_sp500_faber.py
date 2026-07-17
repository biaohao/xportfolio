"""
compute_sp500_faber.py
Recomputes all SP500 B&H vs Timing data for the Faber Fig 7/8 article illustration.

Periods:
  1901–1972
  1901–2012
  1973–Jun 2026
  2000–Jun 2026

Full equity-curve data:
  1901–Jun 2026
  2000–Jun 2026

SMA warm-up: 1900 prices used (10-month SMA → first valid signal is Jan 1901).
Cash when out of market: FRED TB3MS tbill series; pre-1934 = 0%.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from metrics import (
    cagr, annualised_volatility, sharpe_ratio, max_drawdown,
    calmar_ratio, ulcer_index, wealth_index, drawdown_series,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_spy_and_tbill() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(ROOT / "data/processed/prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True)
    spy = df["SPY"].dropna()

    # T-bill returns: BIL column is the spliced tbill series
    bil = df["BIL"].dropna()
    tbill_ret = bil.pct_change().dropna()
    # Pre-1934: set to 0 (conservative, matches Faber approach)
    tbill_ret = tbill_ret.reindex(spy.index)
    tbill_ret[tbill_ret.index.year < 1934] = 0.0
    tbill_ret = tbill_ret.fillna(0.0)

    return spy, tbill_ret


# ---------------------------------------------------------------------------
# Timing engine (single-asset SMA-10, 1-month lag)
# ---------------------------------------------------------------------------

def run_sp500_timing(spy: pd.Series, tbill_ret: pd.Series,
                     start: str, end: str,
                     warmup_start: str = "1900-01-01",
                     sma: int = 10) -> tuple[pd.Series, pd.Series]:
    """
    Returns (bh_returns, timing_returns) for the requested window.
    Data from warmup_start is used to seed the SMA; reported returns
    begin at `start`.
    """
    # Slice with warmup
    s = spy.loc[warmup_start:]
    if end:
        s = s.loc[:end]

    prices = s.copy()

    # SMA and signal (shift 1: signal at end of t → trade at start of t+1)
    sma_vals = prices.rolling(window=sma, min_periods=sma).mean()
    raw_sig = (prices > sma_vals).astype(float)
    sig = raw_sig.shift(1)   # NaN for first `sma` rows

    # Period returns
    rets = prices.pct_change()

    # Align tbill to same index
    tb = tbill_ret.reindex(prices.index).fillna(0.0)

    # Trim to reporting window
    rets_window   = rets.loc[start:end]
    sig_window    = sig.loc[start:end]
    tb_window     = tb.loc[start:end]

    # B&H returns (simply SPY returns)
    bh = rets_window.copy()

    # Timing returns: invested when sig==1 → earn SPY return; else earn tbill
    timing = rets_window.where(sig_window == 1.0, tb_window)

    return bh, timing


# ---------------------------------------------------------------------------
# Metric dict for one strategy in one period
# ---------------------------------------------------------------------------

def period_metrics(returns: pd.Series, tbill: pd.Series, label: str) -> dict:
    rf = tbill.reindex(returns.index).fillna(0.0)
    c  = cagr(returns)
    v  = annualised_volatility(returns, "monthly")
    sh = sharpe_ratio(returns, rf, "monthly")
    md = max_drawdown(returns)
    cal = calmar_ratio(returns, "monthly")
    ui = ulcer_index(returns)
    w  = wealth_index(returns)
    terminal = round(float(w.iloc[-1]), 2) if not w.empty else float("nan")

    # date range
    r_clean = returns.dropna()
    yrs = (r_clean.index[-1] - r_clean.index[0]).days / 365.25 if len(r_clean) > 1 else float("nan")

    return {
        "label":       label,
        "start":       r_clean.index[0].strftime("%Y-%m") if not r_clean.empty else "",
        "end":         r_clean.index[-1].strftime("%Y-%m") if not r_clean.empty else "",
        "years":       round(yrs, 1),
        "cagr":        round(c * 100, 2) if not np.isnan(c) else float("nan"),
        "vol":         round(v * 100, 2) if not np.isnan(v) else float("nan"),
        "sharpe":      round(sh, 3) if not np.isnan(sh) else float("nan"),
        "max_dd":      round(md * 100, 2) if not np.isnan(md) else float("nan"),
        "calmar":      round(cal, 3) if not np.isnan(cal) else float("nan"),
        "ulcer":       round(ui, 2) if not np.isnan(ui) else float("nan"),
        "terminal":    terminal,
    }


# ---------------------------------------------------------------------------
# Equity-curve helpers → list of {date, bh, timing} for JSON
# ---------------------------------------------------------------------------

def equity_curve_data(bh: pd.Series, timing: pd.Series) -> list[dict]:
    wi_bh = wealth_index(bh)
    wi_t  = wealth_index(timing)
    rows = []
    for dt in wi_bh.index:
        bv = wi_bh.get(dt, float("nan"))
        tv = wi_t.get(dt, float("nan"))
        if not (np.isnan(bv) or np.isnan(tv)):
            rows.append({
                "date":   dt.strftime("%Y-%m"),
                "bh":     round(float(bv), 6),
                "timing": round(float(tv), 6),
            })
    return rows


def drawdown_data(bh: pd.Series, timing: pd.Series) -> list[dict]:
    dd_bh = drawdown_series(bh)
    dd_t  = drawdown_series(timing)
    rows = []
    for dt in dd_bh.index:
        bv = dd_bh.get(dt, float("nan"))
        tv = dd_t.get(dt, float("nan"))
        if not (np.isnan(bv) or np.isnan(tv)):
            rows.append({
                "date":    dt.strftime("%Y-%m"),
                "bh_dd":   round(float(bv) * 100, 4),
                "timing_dd": round(float(tv) * 100, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PERIODS = [
    ("1901–1972",     "1901-01-01", "1972-12-31"),
    ("1901–2012",     "1901-01-01", "2012-12-31"),
    ("1973–Jun 2026", "1973-01-01", "2026-06-30"),
    ("2000–Jun 2026", "2000-01-01", "2026-06-30"),
]

EQUITY_WINDOWS = [
    ("full",  "1901-01-01", "2026-06-30"),
    ("2000s", "2000-01-01", "2026-06-30"),
]


def main() -> None:
    spy, tbill_ret = load_spy_and_tbill()

    out = {}

    # ---- Period metrics table ----
    for period_name, start, end in PERIODS:
        # Warmup: always start from 1900 for SMA seed
        bh, timing = run_sp500_timing(spy, tbill_ret, start, end,
                                      warmup_start="1900-01-01")
        bh_m = period_metrics(bh, tbill_ret, "Buy & Hold")
        t_m  = period_metrics(timing, tbill_ret, "Timing SMA-10")
        out[period_name] = {"bh": bh_m, "timing": t_m}
        print(f"\n{period_name}")
        print(f"  B&H   : CAGR={bh_m['cagr']:5.2f}%  Vol={bh_m['vol']:5.2f}%  "
              f"Sharpe={bh_m['sharpe']:6.3f}  MaxDD={bh_m['max_dd']:7.2f}%  "
              f"Ulcer={bh_m['ulcer']:5.2f}  $1→${bh_m['terminal']:,.0f}")
        print(f"  Timing: CAGR={t_m['cagr']:5.2f}%  Vol={t_m['vol']:5.2f}%  "
              f"Sharpe={t_m['sharpe']:6.3f}  MaxDD={t_m['max_dd']:7.2f}%  "
              f"Ulcer={t_m['ulcer']:5.2f}  $1→${t_m['terminal']:,.0f}")

    # ---- Equity curves ----
    for win_name, start, end in EQUITY_WINDOWS:
        bh, timing = run_sp500_timing(spy, tbill_ret, start, end,
                                      warmup_start="1900-01-01")
        out[f"_eq_{win_name}"]  = equity_curve_data(bh, timing)
        out[f"_dd_{win_name}"]  = drawdown_data(bh, timing)
        print(f"\nEquity window '{win_name}': {len(out[f'_eq_{win_name}'])} points")

    # Save JSON
    out_path = ROOT / "data/analysis/sp500_faber_fig7_data.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
