"""
triple_asset.py — Two-part analysis

PART 1: Fix + recompute SP500 / IEF clean stats for all three windows
         (replaces the inf-overflow rows in sp500_ief_5050.csv)

PART 2: Three-asset portfolios
         - SP500 + Gold + IEF   (monthly, three windows)
         - SP500 + Gold + TLT   (monthly, three windows)
         Equal-weight rebalanced monthly; B&H and Timing (SMA-10) variants.

Data:
  data/processed/prices_monthly_spliced.csv  — SPY col (1871+), IEF col (1962+), TLT col (1962+), BIL col (1934+)
  data/processed/gold_monthly.csv            — Gold spot prices (1973+)

Outputs:
  results/sp500_ief_5050_fixed.csv    — Part 1: clean IEF/SP500 stats (all 3 windows)
  results/triple_asset.csv            — Part 2: 3-asset portfolio metrics
  results/triple_asset_annual.csv     — Part 2: annual returns table

No look-ahead bias: SMA computed over full history; window trimmed post-hoc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from metrics import (
    cagr, annualised_volatility as vol_fn, sharpe_ratio as sharpe_fn,
    max_drawdown as max_dd, calmar_ratio as calmar, ulcer_index as ulcer,
    win_rate, best_worst_year as best_worst_yr, dollars_to,
)

PROC    = Path("data/processed")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

_FACTOR = 12   # monthly data throughout


def vol(ret: pd.Series) -> float:
    return vol_fn(ret, "monthly")


def sharpe(ret: pd.Series, rf_annual: float = 0.0) -> float:
    return sharpe_fn(ret, rf_annual, "monthly")


def pct_invested(signal: pd.Series) -> float:
    return float(signal.mean() * 100)


def stats_row(label: str, ret: pd.Series, window: str,
              sig: pd.Series | None = None, rf: float = 0.0) -> dict:
    best, worst = best_worst_yr(ret)
    row = {
        "Portfolio":  label,
        "CAGR%":      round(cagr(ret) * 100, 2),
        "Vol%":       round(vol(ret) * 100, 2),
        "Sharpe":     round(sharpe(ret, rf), 3),
        "Max DD%":    round(max_dd(ret) * 100, 2),
        "Calmar":     round(calmar(ret, "monthly"), 3),
        "Ulcer%":     round(ulcer(ret), 2),
        "Win%":       round(win_rate(ret), 1),
        "$1→":        dollars_to(ret),
        "Best Yr":    best,
        "Worst Yr":   worst,
        "Window":     window,
        "% Inv":      round(pct_invested(sig), 1) if sig is not None else 100.0,
    }
    return row


# ---------------------------------------------------------------------------
# SMA timing helper
# ---------------------------------------------------------------------------

def sma_signal(prices: pd.Series, period: int = 10) -> pd.Series:
    """Return lagged SMA signal (1=invested, 0=cash). No look-ahead."""
    sma = prices.rolling(period).mean()
    sig = (prices > sma).astype(float)
    return sig.shift(1)          # signal known at end-of-prior-month


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_spliced() -> pd.DataFrame:
    df = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                     index_col=0, parse_dates=True)
    return df.sort_index()


def load_tlt() -> pd.Series:
    spliced = pd.read_csv(PROC / "prices_monthly_spliced.csv",
                          index_col=0, parse_dates=True)
    s = spliced["TLT"].dropna()
    s.name = "TLT"
    return s.sort_index()


def load_gold() -> pd.Series:
    s = pd.read_csv(PROC / "gold_monthly.csv",
                    index_col=0, parse_dates=True).squeeze("columns")
    s.name = "Gold"
    return s.sort_index()


# ---------------------------------------------------------------------------
# Core 2-asset / 3-asset portfolio builder
# ---------------------------------------------------------------------------

def build_portfolio_returns(
    price_dict: dict[str, pd.Series],
    sma_period: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a dict of {name: price_series}, compute:
      - returns DataFrame (one column per asset)
      - signals DataFrame (SMA signals, one column per asset)

    Returns aligned on common non-NaN index.
    """
    # Align all series to common dates
    df = pd.DataFrame(price_dict).sort_index().dropna(how="all")
    # Forward-fill very small gaps (month-end alignment) up to 5 days
    df = df.ffill(limit=1)
    # Require all assets present
    df = df.dropna(how="any")

    rets = df.pct_change().dropna()
    sigs = pd.DataFrame(index=df.index)
    for col in df.columns:
        sigs[col] = sma_signal(df[col], sma_period)

    # Align
    rets, sigs = rets.align(sigs, join="inner", axis=0)
    sigs = sigs.reindex(rets.index).fillna(0)

    return rets, sigs


def portfolio_bh(rets: pd.DataFrame) -> pd.Series:
    """Equal-weight buy-and-hold return each period."""
    w = 1.0 / len(rets.columns)
    return rets.mean(axis=1)     # same as sum * w / (1/n) — equal weight


def portfolio_timing_correct(
    rets: pd.DataFrame,
    sigs: pd.DataFrame,
    cash_returns: pd.Series | None = None,
) -> pd.Series:
    """
    Each asset gets equal weight 1/N.
    If signal=1: that slot earns the asset return.
    If signal=0: that slot earns the T-bill return (cash_returns) when provided,
                 or 0% when cash_returns is None.
    Portfolio return = mean over assets of (ret*sig + cash_ret*(1-sig)).
    """
    sig_aligned = sigs.reindex(rets.index).fillna(0)
    if cash_returns is not None:
        cr = cash_returns.reindex(rets.index).ffill().fillna(0)
        timed = rets * sig_aligned + cr.values.reshape(-1, 1) * (1 - sig_aligned)
    else:
        timed = rets * sig_aligned
    return timed.mean(axis=1)


# ---------------------------------------------------------------------------
# Window runner
# ---------------------------------------------------------------------------

WINDOWS_IEF = [
    ("Full (1962–2026)",   "1962-01-01", "2026-06-30"),
    ("Faber (1977–2012)",  "1977-01-01", "2012-12-31"),
    ("Modern (2007–2026)", "2007-01-01", "2026-06-30"),
]

WINDOWS_TRIPLE = [
    ("Full (1973–2026)",   "1973-01-01", "2026-06-30"),
    ("Faber (1973–2012)",  "1973-01-01", "2012-12-31"),
    ("Modern (2007–2026)", "2007-01-01", "2026-06-30"),
]

SMA = 10


def run_window(
    rets: pd.DataFrame,
    sigs: pd.DataFrame,
    window_label: str,
    start: str,
    end: str,
    asset_names: dict[str, str] | None = None,
    cash_returns: pd.Series | None = None,
) -> list[dict]:
    """Slice to window and compute all portfolio variants."""
    r = rets.loc[start:end].copy()
    s = sigs.loc[start:end].copy()

    if len(r) < 24:
        return []

    cols = list(r.columns)
    an = asset_names or {c: c for c in cols}

    rows = []

    # Individual assets B&H + Timing
    for col in cols:
        label = an.get(col, col)
        rows.append(stats_row(f"{label} B&H", r[col], window_label))
        sig_col = s[col].reindex(r.index).fillna(0)
        if cash_returns is not None:
            cr = cash_returns.reindex(r.index).ffill().fillna(0)
            timed_col = r[col] * sig_col + cr * (1 - sig_col)
        else:
            timed_col = r[col] * sig_col
        rows.append(stats_row(f"{label} Timing", timed_col, window_label, sig=s[col]))

    # Portfolio B&H (equal weight, rebalanced monthly)
    bh  = portfolio_bh(r)
    tim = portfolio_timing_correct(r, s, cash_returns=cash_returns)

    # Combined signal for % invested
    avg_sig = s.mean(axis=1)

    n = len(cols)
    portfolio_name = "/".join(an.get(c, c) for c in cols)
    rows.append(stats_row(f"{n}-asset B&H ({portfolio_name})",    bh,  window_label))
    rows.append(stats_row(f"{n}-asset Timing ({portfolio_name})", tim, window_label,
                          sig=avg_sig))

    return rows


# ---------------------------------------------------------------------------
# PART 1 — Recompute SP500 / IEF (fix overflow)
# ---------------------------------------------------------------------------

def _cash_returns(spliced: pd.DataFrame) -> pd.Series:
    """Extract T-bill per-period returns from the BIL column of the spliced DataFrame."""
    return spliced["BIL"].dropna().pct_change().dropna()


def part1_ief(spliced: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PART 1 — SP500 / IEF  (overflow-safe recompute)")
    print("="*70)

    spy = spliced["SPY"].dropna()
    ief = spliced["IEF"].dropna()
    cr  = _cash_returns(spliced)

    price_dict = {"SPY": spy, "IEF": ief}
    rets, sigs = build_portfolio_returns(price_dict, sma_period=SMA)

    an = {"SPY": "SP500", "IEF": "IEF"}

    all_rows: list[dict] = []

    win_labels = [
        ("Full (1962–2026)",   "1962-01-01", "2026-06-30"),
        ("Faber (1977–2012)",  "1977-01-01", "2012-12-31"),
        ("Modern (2007–2026)", "2007-01-01", "2026-06-30"),
    ]

    for label, start, end in win_labels:
        rows = run_window(rets, sigs, label, start, end, asset_names=an, cash_returns=cr)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(df[["Portfolio", "CAGR%", "Vol%", "Sharpe", "Max DD%",
              "Calmar", "Ulcer%", "$1→", "Window"]].to_string(index=False))

    out = RESULTS / "sp500_ief_5050_fixed.csv"
    df.to_csv(out, index=False)
    print(f"\n  → saved: {out}")
    return df


# ---------------------------------------------------------------------------
# PART 2 — Triple-asset portfolios
# ---------------------------------------------------------------------------

def part2_triple(spliced: pd.DataFrame, tlt: pd.Series, gold: pd.Series) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PART 2 — Triple-asset: SP500 + Gold + IEF  /  SP500 + Gold + TLT")
    print("="*70)

    spy = spliced["SPY"].dropna()
    ief = spliced["IEF"].dropna()
    cr  = _cash_returns(spliced)

    all_rows: list[dict] = []

    # --- Combo A: SP500 + Gold + IEF ---
    print("\n--- Combo A: SP500 + Gold + IEF ---")
    pA = {"SPY": spy, "Gold": gold, "IEF": ief}
    rA, sA = build_portfolio_returns(pA, sma_period=SMA)
    anA = {"SPY": "SP500", "Gold": "Gold", "IEF": "IEF"}

    for label, start, end in WINDOWS_TRIPLE:
        rows = run_window(rA, sA, label, start, end, asset_names=anA, cash_returns=cr)
        all_rows.extend(rows)

    # Summarise Combo A quickly
    for label, start, end in WINDOWS_TRIPLE:
        r = rA.loc[start:end]
        if len(r) < 12:
            continue
        s_slice = sA.loc[start:end]
        bh  = portfolio_bh(r)
        tim = portfolio_timing_correct(r, s_slice, cash_returns=cr)
        c_bh  = cagr(bh)
        c_tim = cagr(tim)
        sh_bh  = sharpe(bh)
        sh_tim = sharpe(tim)
        dd_bh  = max_dd(bh)
        dd_tim = max_dd(tim)
        print(f"  {label}: B&H CAGR={c_bh*100:.2f}% Sharpe={sh_bh:.3f} MaxDD={dd_bh*100:.2f}%"
              f"  |  Timing CAGR={c_tim*100:.2f}% Sharpe={sh_tim:.3f} MaxDD={dd_tim*100:.2f}%")

    # --- Combo B: SP500 + Gold + TLT ---
    print("\n--- Combo B: SP500 + Gold + TLT ---")
    pB = {"SPY": spy, "Gold": gold, "TLT": tlt}
    rB, sB = build_portfolio_returns(pB, sma_period=SMA)
    anB = {"SPY": "SP500", "Gold": "Gold", "TLT": "TLT"}

    win_B = [
        ("Full (1977–2026)",   "1977-01-01", "2026-06-30"),
        ("Faber (1977–2012)",  "1977-01-01", "2012-12-31"),
        ("Modern (2007–2026)", "2007-01-01", "2026-06-30"),
    ]

    for label, start, end in win_B:
        rows = run_window(rB, sB, label, start, end, asset_names=anB, cash_returns=cr)
        all_rows.extend(rows)

    for label, start, end in win_B:
        r = rB.loc[start:end]
        if len(r) < 12:
            continue
        s_slice = sB.loc[start:end]
        bh  = portfolio_bh(r)
        tim = portfolio_timing_correct(r, s_slice, cash_returns=cr)
        c_bh  = cagr(bh)
        c_tim = cagr(tim)
        sh_bh  = sharpe(bh)
        sh_tim = sharpe(tim)
        dd_bh  = max_dd(bh)
        dd_tim = max_dd(tim)
        print(f"  {label}: B&H CAGR={c_bh*100:.2f}% Sharpe={sh_bh:.3f} MaxDD={dd_bh*100:.2f}%"
              f"  |  Timing CAGR={c_tim*100:.2f}% Sharpe={sh_tim:.3f} MaxDD={dd_tim*100:.2f}%")

    df = pd.DataFrame(all_rows)
    out = RESULTS / "triple_asset.csv"
    df.to_csv(out, index=False)
    print(f"\n  → saved: {out}")

    # Print full summary table
    print("\n" + "-"*90)
    print(df[["Portfolio", "CAGR%", "Vol%", "Sharpe", "Max DD%",
              "Ulcer%", "$1→", "Window", "% Inv"]].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# PART 3 — Annual returns table for the triple-asset combos (Faber + Modern)
# ---------------------------------------------------------------------------

def part3_annual(spliced: pd.DataFrame, tlt: pd.Series, gold: pd.Series) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PART 3 — Annual returns table for 3-asset combos (1977–2026)")
    print("="*70)

    spy = spliced["SPY"].dropna()
    ief = spliced["IEF"].dropna()
    cr  = _cash_returns(spliced)

    # Combo A: SP500 + Gold + IEF (1973+)
    pA = {"SPY": spy, "Gold": gold, "IEF": ief}
    rA, sA = build_portfolio_returns(pA, sma_period=SMA)
    rA = rA.loc["1973-01-01":"2026-06-30"]
    sA = sA.loc["1973-01-01":"2026-06-30"]

    bhA  = portfolio_bh(rA)
    timA = portfolio_timing_correct(rA, sA, cash_returns=cr)

    # Combo B: SP500 + Gold + TLT (1977+)
    pB = {"SPY": spy, "Gold": gold, "TLT": tlt}
    rB, sB = build_portfolio_returns(pB, sma_period=SMA)
    rB = rB.loc["1977-01-01":"2026-06-30"]
    sB = sB.loc["1977-01-01":"2026-06-30"]

    bhB  = portfolio_bh(rB)
    timB = portfolio_timing_correct(rB, sB, cash_returns=cr)

    # SP500 for reference
    spy_rets = spy.pct_change().dropna().loc["1973-01-01":"2026-06-30"]

    def annual(s: pd.Series, label: str) -> pd.Series:
        return (s.resample("YE").apply(lambda x: (1 + x).prod() - 1)
                 .rename(lambda idx: idx.year)
                 .rename(label))

    ann = pd.concat([
        annual(spy_rets, "SP500 B&H"),
        annual(bhA,  "SP+Gold+IEF B&H"),
        annual(timA, "SP+Gold+IEF Timing"),
        annual(bhB,  "SP+Gold+TLT B&H"),
        annual(timB, "SP+Gold+TLT Timing"),
    ], axis=1)

    # Format as percentages
    ann_pct = (ann * 100).round(1)

    print(ann_pct.to_string())

    out = RESULTS / "triple_asset_annual.csv"
    ann_pct.to_csv(out)
    print(f"\n  → saved: {out}")
    return ann_pct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data…")
    spliced = load_spliced()
    tlt     = load_tlt()
    gold    = load_gold()

    print(f"  Spliced: {spliced.index[0].date()} – {spliced.index[-1].date()}"
          f"  ({len(spliced)} months)")
    print(f"  TLT:     {tlt.index[0].date()} – {tlt.index[-1].date()}"
          f"  ({len(tlt)} months)")
    print(f"  Gold:    {gold.index[0].date()} – {gold.index[-1].date()}"
          f"  ({len(gold)} months)")

    df1 = part1_ief(spliced)
    df2 = part2_triple(spliced, tlt, gold)
    df3 = part3_annual(spliced, tlt, gold)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
