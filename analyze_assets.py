"""
analyze_assets.py

Comprehensive analysis of IEF, TLT, GLD, SP500 individually and as
3-asset portfolios (SP500+IEF+GLD, SP500+TLT+GLD).

Two time periods:
  - Period 1: Starting from 1973-01-01 (gold/long-history data, spliced)
  - Period 2: Starting from 2004-12-31 (when all ETF data are available;
              GLD ETF launched Nov 2004, IEF/TLT ETF launched Jul 2002)

Both Buy & Hold and Timing (SMA-10) variants.
End date: 2026-07-31

Outputs:
  data/analysis/asset_analysis.json  — all metrics for the dashboard
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from metrics import (
    cagr, annualised_volatility as vol_fn, sharpe_ratio as sharpe_fn,
    max_drawdown as max_dd, calmar_ratio as calmar, ulcer_index as ulcer,
    win_rate, best_worst_year as best_worst_yr, dollars_to, wealth_index,
)

PROC    = Path("data/processed")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Period definitions
# ---------------------------------------------------------------------------

PERIOD_1_START = "1973-01-01"   # Gold data available from 1973
PERIOD_2_START = "2004-12-31"   # All ETFs available (GLD ETF Nov 2004)
PERIOD_END     = "2026-07-31"


def vol(ret: pd.Series) -> float:
    return vol_fn(ret, "monthly")


def sharpe(ret: pd.Series, rf_annual: float = 0.0) -> float:
    return sharpe_fn(ret, rf_annual, "monthly")


def annual_returns(ret: pd.Series) -> dict:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}
    ann = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    return {int(idx.year): round(float(v) * 100, 2) for idx, v in ann.items() if not np.isnan(v)}


def cumulative_wealth(ret: pd.Series) -> dict:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}
    w = wealth_index(r)
    return {str(idx.date()): round(float(v), 4) for idx, v in w.items()}


def pct_invested_fn(sig: pd.Series) -> float:
    return round(float(sig.mean() * 100), 1) if not sig.empty else 100.0


# ---------------------------------------------------------------------------
# SMA timing signal
# ---------------------------------------------------------------------------

def sma_signal(prices: pd.Series, period: int = 10) -> pd.Series:
    sma = prices.rolling(period).mean()
    sig = (prices > sma).astype(float)
    return sig.shift(1)


# ---------------------------------------------------------------------------
# Stats row builder
# ---------------------------------------------------------------------------

def stats_row(label: str, ret: pd.Series, period: str,
              sig: pd.Series | None = None) -> dict:
    best, worst = best_worst_yr(ret)
    return {
        "label":       label,
        "period":      period,
        "cagr":        round(cagr(ret) * 100, 2),
        "vol":         round(vol(ret) * 100, 2),
        "sharpe":      round(sharpe(ret), 3),
        "max_dd":      round(max_dd(ret) * 100, 2),
        "calmar":      round(calmar(ret, "monthly"), 3),
        "ulcer":       round(ulcer(ret), 2),
        "win_rate":    round(win_rate(ret), 1),
        "dollars_to":  dollars_to(ret),
        "best_yr":     best,
        "worst_yr":    worst,
        "pct_invested": pct_invested_fn(sig) if sig is not None else 100.0,
        "annual":      annual_returns(ret),
        "wealth":      cumulative_wealth(ret),
        "n_months":    int(ret.dropna().shape[0]),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    spliced = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                          index_col=0, parse_dates=True).sort_index()
    tlt = spliced["TLT"].dropna()
    tlt.name = "TLT"
    gold = pd.read_csv(PROC / "gold_monthly.csv",
                       index_col=0, parse_dates=True).squeeze("columns").sort_index()
    gold.name = "GLD"
    # BIL column in the spliced file is the T-bill total-return proxy (FRED TB3MS pre-2007,
    # BIL ETF from 2007 onward).  Convert the price index to per-period returns here so
    # callers can pass it straight into timing functions.
    bil_prices = spliced["BIL"].dropna()
    cash_returns = bil_prices.pct_change().dropna()
    return spliced, tlt, gold, cash_returns


# ---------------------------------------------------------------------------
# Build analysis
# ---------------------------------------------------------------------------

def build_price_df(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Align, forward-fill gaps up to 1 month, require all assets present."""
    df = pd.DataFrame(price_dict).sort_index().dropna(how="all")
    df = df.ffill(limit=1)
    df = df.dropna(how="any")
    return df


def _apply_cash(ret: pd.Series, sig: pd.Series,
                cash_ret: pd.Series) -> pd.Series:
    """
    Per-period return for one asset slot under the timing rule:
        invested  (sig=1): earn asset return
        in cash   (sig=0): earn T-bill return

    Formula:  r_timing = ret * sig  +  cash_ret * (1 - sig)
    """
    sig_a = sig.reindex(ret.index).fillna(0)
    cr    = cash_ret.reindex(ret.index).ffill().fillna(0)
    return ret * sig_a + cr * (1 - sig_a)


def analyse_single(prices: pd.Series, name: str, period_label: str,
                   start: str, end: str,
                   cash_returns: pd.Series | None = None) -> list[dict]:
    """B&H and Timing stats for a single asset in a given window."""
    # Compute pct_change on the FULL series before slicing so the first
    # month's return (e.g. Jan 2000 when start="2000-01-01") is not lost.
    # The $1 base is implicitly the Dec 1999 close, not Jan 2000.
    ret = prices.pct_change()
    ret = ret.loc[start:end].dropna()
    if len(ret) < 24:
        return []

    sig = sma_signal(prices).loc[ret.index]

    # Drop early NaN signals (SMA warm-up)
    first_valid = sig.first_valid_index()
    if first_valid is None:
        return []
    ret = ret.loc[first_valid:]
    sig = sig.loc[first_valid:]

    rows = []
    rows.append(stats_row(f"{name} B&H", ret, period_label))
    if cash_returns is not None:
        timed = _apply_cash(ret, sig, cash_returns)
    else:
        timed = ret * sig.reindex(ret.index).fillna(0)
    rows.append(stats_row(f"{name} Timing", timed, period_label, sig=sig))
    return rows


def analyse_portfolio(price_dict: dict[str, pd.Series], port_name: str,
                      period_label: str, start: str, end: str,
                      weights: dict[str, float] | None = None,
                      cash_returns: pd.Series | None = None) -> list[dict]:
    """
    Build portfolio from price_dict within the window.

    weights: dict of {col: weight} that sum to 1.0; None → equal weight.
    cash_returns: T-bill per-period return series; when provided the cash
                  slot earns this rate instead of 0%.
    Returns portfolio B&H + portfolio Timing stats rows.
    """
    df = build_price_df(price_dict)
    # Compute pct_change and SMA signals on the FULL price history, then
    # trim the return/signal series to the window.  This preserves the
    # first month's return (e.g. Jan 2000 uses Dec 1999 as its base).
    rets_full = df.pct_change()
    sigs_full = pd.DataFrame({col: sma_signal(df[col]) for col in df.columns})

    rets = rets_full.loc[start:end].dropna()
    if len(rets) < 24:
        return []

    sigs = sigs_full.reindex(rets.index)

    # Drop SMA warm-up period
    first_valid = sigs.dropna(how="any").index[0] if not sigs.dropna(how="any").empty else None
    if first_valid is None:
        return []
    rets = rets.loc[first_valid:]
    sigs = sigs.loc[first_valid:].fillna(0)

    # Resolve weight vector
    if weights is None:
        w = pd.Series({col: 1.0 / len(df.columns) for col in df.columns})
    else:
        w = pd.Series({col: weights.get(col, 0.0) for col in df.columns})
        w = w / w.sum()   # normalise to 1

    rows = []

    # Portfolio B&H (weighted, monthly rebalance)
    bh = (rets * w).sum(axis=1)

    # Portfolio Timing: invested slice earns asset return, cash slice earns T-bill
    if cash_returns is not None:
        cr = cash_returns.reindex(rets.index).ffill().fillna(0)
        # Each asset column: ret*sig*w  +  cash_ret*(1-sig)*w
        timing = ((rets * sigs + cr.values.reshape(-1, 1) * (1 - sigs)) * w).sum(axis=1)
    else:
        timing = (rets * sigs * w).sum(axis=1)

    # Weighted-average % invested
    avg_sig = (sigs * w).sum(axis=1)

    rows.append(stats_row(f"{port_name} B&H", bh, period_label))
    rows.append(stats_row(f"{port_name} Timing", timing, period_label, sig=avg_sig))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data…")
    spliced, tlt, gold, cash_returns = load_data()

    spy = spliced["SPY"].dropna()
    ief = spliced["IEF"].dropna()

    # Gold: use spliced GLD column (gold spot proxy from 1973)
    gld = spliced["GLD"].dropna()

    print(f"  SPY:  {spy.index[0].date()} – {spy.index[-1].date()}")
    print(f"  IEF:  {ief.index[0].date()} – {ief.index[-1].date()}")
    print(f"  TLT:  {tlt.index[0].date()} – {tlt.index[-1].date()}")
    print(f"  GLD:  {gld.index[0].date()} – {gld.index[-1].date()}")

    periods = [
        ("1973–2026", PERIOD_1_START, PERIOD_END),
        ("ETF Era (2004–2026)", PERIOD_2_START, PERIOD_END),
    ]

    all_rows: list[dict] = []

    for period_label, start, end in periods:
        print(f"\n--- Period: {period_label} ({start[:4]}–{end[:4]}) ---")

        # --- Individual assets ---
        for name, prices in [("SP500", spy), ("IEF", ief), ("TLT", tlt), ("GLD", gld)]:
            rows = analyse_single(prices, name, period_label, start, end,
                                  cash_returns=cash_returns)
            all_rows.extend(rows)
            if rows:
                bh_cagr  = rows[0]["cagr"]
                tm_cagr  = rows[1]["cagr"]
                print(f"  {name:8s}  B&H CAGR={bh_cagr:6.2f}%  Timing CAGR={tm_cagr:6.2f}%")

        # --- 3-asset SP500 + IEF + GLD  (equal weight 33/33/33) ---
        port_ief = {"SP500": spy, "IEF": ief, "GLD": gld}
        rows_ief = analyse_portfolio(port_ief, "SP500+IEF+GLD (EW)", period_label, start, end,
                                     cash_returns=cash_returns)
        all_rows.extend(rows_ief)
        if rows_ief:
            print(f"  {'SP500+IEF+GLD EW':22s}  B&H CAGR={rows_ief[0]['cagr']:6.2f}%  "
                  f"Timing CAGR={rows_ief[1]['cagr']:6.2f}%")

        # --- 3-asset SP500 + IEF + GLD  (50/25/25) ---
        w_ief = {"SP500": 0.50, "IEF": 0.25, "GLD": 0.25}
        rows_ief_w = analyse_portfolio(port_ief, "SP500+IEF+GLD (50/25/25)", period_label, start, end,
                                       weights=w_ief, cash_returns=cash_returns)
        all_rows.extend(rows_ief_w)
        if rows_ief_w:
            print(f"  {'SP500+IEF+GLD 50/25/25':22s}  B&H CAGR={rows_ief_w[0]['cagr']:6.2f}%  "
                  f"Timing CAGR={rows_ief_w[1]['cagr']:6.2f}%")

        # --- 3-asset SP500 + TLT + GLD  (equal weight 33/33/33) ---
        port_tlt = {"SP500": spy, "TLT": tlt, "GLD": gld}
        rows_tlt = analyse_portfolio(port_tlt, "SP500+TLT+GLD (EW)", period_label, start, end,
                                     cash_returns=cash_returns)
        all_rows.extend(rows_tlt)
        if rows_tlt:
            print(f"  {'SP500+TLT+GLD EW':22s}  B&H CAGR={rows_tlt[0]['cagr']:6.2f}%  "
                  f"Timing CAGR={rows_tlt[1]['cagr']:6.2f}%")

        # --- 3-asset SP500 + TLT + GLD  (50/25/25) ---
        w_tlt = {"SP500": 0.50, "TLT": 0.25, "GLD": 0.25}
        rows_tlt_w = analyse_portfolio(port_tlt, "SP500+TLT+GLD (50/25/25)", period_label, start, end,
                                       weights=w_tlt, cash_returns=cash_returns)
        all_rows.extend(rows_tlt_w)
        if rows_tlt_w:
            print(f"  {'SP500+TLT+GLD 50/25/25':22s}  B&H CAGR={rows_tlt_w[0]['cagr']:6.2f}%  "
                  f"Timing CAGR={rows_tlt_w[1]['cagr']:6.2f}%")

    # -------------------------------------------------------------------------
    # Output JSON
    # -------------------------------------------------------------------------
    output = {
        "metadata": {
            "periods": [
                {"label": "1973–2026", "start": PERIOD_1_START, "end": PERIOD_END,
                 "description": "Full history (gold/spliced data from 1973)"},
                {"label": "ETF Era (2004–2026)", "start": PERIOD_2_START, "end": PERIOD_END,
                 "description": "All ETFs available (GLD ETF launched Nov 2004)"},
            ],
            "assets": ["SP500", "IEF", "TLT", "GLD"],
            "portfolios": ["SP500+IEF+GLD (EW)", "SP500+IEF+GLD (50/25/25)",
                           "SP500+TLT+GLD (EW)", "SP500+TLT+GLD (50/25/25)"],
            "strategy": "SMA-10 monthly; EW = equal weight 33/33/33; 50/25/25 = SP500 50%, bond 25%, GLD 25%",
            "note_period1": "Uses spliced/synthetic data: SP500 from Shiller 1871+, IEF from 1962+, TLT spliced from 1977, GLD gold spot from 1973",
            "note_period2": "Uses actual ETF price history (GLD ETF Nov 2004, IEF/TLT ETF Jul 2002, SPY ETF Jan 1993)",
        },
        "rows": all_rows,
    }

    out_path = OUT_DIR / "asset_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved: {out_path}  ({len(all_rows)} rows)")

    # Also print summary table
    print("\n" + "="*100)
    print(f"{'Label':<35} {'Period':<25} {'CAGR%':>7} {'Vol%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'$1→':>8} {'%Inv':>6}")
    print("="*100)
    for r in all_rows:
        print(f"{r['label']:<35} {r['period']:<25} {r['cagr']:>7.2f} {r['vol']:>7.2f} "
              f"{r['sharpe']:>7.3f} {r['max_dd']:>8.2f} {r['dollars_to']:>8.2f} {r['pct_invested']:>6.1f}")


if __name__ == "__main__":
    main()
