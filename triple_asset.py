"""
triple_asset.py — Two-part analysis

PART 1: Fix + recompute SP500 / IEF clean stats for all three windows
         (replaces the inf-overflow rows in sp500_ief_5050.csv)

PART 2: Three-asset portfolios
         - SP500 + Gold + IEF   (monthly, three windows)
         - SP500 + Gold + TLT   (monthly, three windows)
         Equal-weight rebalanced monthly; B&H and Timing (SMA-10) variants.

Data:
  data/processed/prices_monthly_spliced.csv  — SPY col (1871+), IEF col (1962+), BIL col (1934+)
  data/processed/tlt_monthly_spliced.csv     — TLT col (1977+)
  data/processed/gold_monthly.csv            — Gold spot prices (1973+)

Outputs:
  results/sp500_ief_5050_fixed.csv    — Part 1: clean IEF/SP500 stats (all 3 windows)
  results/triple_asset.csv            — Part 2: 3-asset portfolio metrics
  results/triple_asset_annual.csv     — Part 2: annual returns table

No look-ahead bias: SMA computed over full history; window trimmed post-hoc.
Overflow fix: log-space cumprod for wealth index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROC    = Path("data/processed")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------
_FACTOR = 12   # monthly data throughout

# ---------------------------------------------------------------------------
# Safe metric helpers — use log-space wealth to prevent float64 overflow
# over 60+ year windows
# ---------------------------------------------------------------------------

def _wealth(ret: pd.Series) -> pd.Series:
    """Cumulative wealth index via log-space accumulation (overflow-safe)."""
    r = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r = r.clip(-0.9999)          # prevent log(0)
    return np.exp(np.log1p(r).cumsum())


def cagr(ret: pd.Series) -> float:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return np.nan
    w = _wealth(r)
    years = (r.index[-1] - r.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float(w.iloc[-1] ** (1.0 / years) - 1.0)


def vol(ret: pd.Series) -> float:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.std() * np.sqrt(_FACTOR))


def sharpe(ret: pd.Series, rf_annual: float = 0.0) -> float:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    rf_pp = (1 + rf_annual) ** (1.0 / _FACTOR) - 1.0
    excess = r - rf_pp
    s = excess.std()
    if s == 0 or np.isnan(s):
        return np.nan
    return float(excess.mean() / s * np.sqrt(_FACTOR))


def max_dd(ret: pd.Series) -> float:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    w = _wealth(r)
    dd = (w - w.cummax()) / w.cummax()
    return float(dd.min())


def calmar(ret: pd.Series) -> float:
    c = cagr(ret)
    md = max_dd(ret)
    if np.isnan(c) or np.isnan(md) or md == 0:
        return np.nan
    return float(c / abs(md))


def ulcer(ret: pd.Series) -> float:
    """Ulcer Index = RMS of drawdown series (%)."""
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    w = _wealth(r)
    dd_pct = ((w - w.cummax()) / w.cummax()) * 100
    return float(np.sqrt((dd_pct ** 2).mean()))


def win_rate(ret: pd.Series) -> float:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    return float((r > 0).mean() * 100)


def best_worst_yr(ret: pd.Series):
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return "n/a", "n/a"
    annual = r.resample("A").apply(lambda x: (1 + x).prod() - 1)
    best_y  = annual.idxmax().year
    worst_y = annual.idxmin().year
    best_v  = annual.max()
    worst_v = annual.min()
    sign_b  = "+" if best_v >= 0 else ""
    sign_w  = "+" if worst_v >= 0 else ""
    return (f"{sign_b}{best_v*100:.1f}% ({best_y})",
            f"{sign_w}{worst_v*100:.1f}% ({worst_y})")


def dollars_to(ret: pd.Series) -> float:
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    w = _wealth(r)
    return round(float(w.iloc[-1]), 2)


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
        "Calmar":     round(calmar(ret), 3),
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
    s = pd.read_csv(PROC / "tlt_monthly_spliced.csv",
                    index_col=0, parse_dates=True).squeeze("columns")
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


def portfolio_timing(rets: pd.DataFrame, sigs: pd.DataFrame) -> pd.Series:
    """
    Timing rule: each asset is invested (weight = 1/N) if signal=1, else cash (0 return).
    """
    n = len(rets.columns)
    w = 1.0 / n
    timed = rets * sigs.reindex(rets.index).fillna(0)
    return timed.sum(axis=1) * w / w    # sum of weighted returns


# Actually: equal-weight means each slot = 1/N of portfolio.
# If timing puts asset in cash, that 1/N earns 0%.
def portfolio_timing_correct(rets: pd.DataFrame, sigs: pd.DataFrame) -> pd.Series:
    """
    Each asset gets weight 1/N.  If signal=0, that slice earns 0 (cash).
    Portfolio return = mean over assets of (ret * signal).
    """
    sig_aligned = sigs.reindex(rets.index).fillna(0)
    return (rets * sig_aligned).mean(axis=1)


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
    asset_names: dict[str, str] | None = None,  # col → display name
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
        rows.append(stats_row(f"{label} B&H",    r[col], window_label))
        timed_col = r[col] * s[col].reindex(r.index).fillna(0)
        rows.append(stats_row(f"{label} Timing", timed_col, window_label,
                              sig=s[col]))

    # Portfolio B&H (equal weight, rebalanced monthly)
    bh  = portfolio_bh(r)
    tim = portfolio_timing_correct(r, s)

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

def part1_ief(spliced: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PART 1 — SP500 / IEF  (overflow-safe recompute)")
    print("="*70)

    spy = spliced["SPY"].dropna()
    ief = spliced["IEF"].dropna()

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
        rows = run_window(rets, sigs, label, start, end, asset_names=an)
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

    all_rows: list[dict] = []

    # --- Combo A: SP500 + Gold + IEF ---
    print("\n--- Combo A: SP500 + Gold + IEF ---")
    pA = {"SPY": spy, "Gold": gold, "IEF": ief}
    rA, sA = build_portfolio_returns(pA, sma_period=SMA)
    anA = {"SPY": "SP500", "Gold": "Gold", "IEF": "IEF"}

    for label, start, end in WINDOWS_TRIPLE:
        rows = run_window(rA, sA, label, start, end, asset_names=anA)
        all_rows.extend(rows)

    # Summarise Combo A quickly
    for label, start, end in WINDOWS_TRIPLE:
        r = rA.loc[start:end]
        if len(r) < 12:
            continue
        s_slice = sA.loc[start:end]
        bh  = portfolio_bh(r)
        tim = portfolio_timing_correct(r, s_slice)
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
        rows = run_window(rB, sB, label, start, end, asset_names=anB)
        all_rows.extend(rows)

    for label, start, end in win_B:
        r = rB.loc[start:end]
        if len(r) < 12:
            continue
        s_slice = sB.loc[start:end]
        bh  = portfolio_bh(r)
        tim = portfolio_timing_correct(r, s_slice)
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

    # Combo A: SP500 + Gold + IEF (1973+)
    pA = {"SPY": spy, "Gold": gold, "IEF": ief}
    rA, sA = build_portfolio_returns(pA, sma_period=SMA)
    rA = rA.loc["1973-01-01":"2026-06-30"]
    sA = sA.loc["1973-01-01":"2026-06-30"]

    bhA  = portfolio_bh(rA)
    timA = portfolio_timing_correct(rA, sA)

    # Combo B: SP500 + Gold + TLT (1977+)
    pB = {"SPY": spy, "Gold": gold, "TLT": tlt}
    rB, sB = build_portfolio_returns(pB, sma_period=SMA)
    rB = rB.loc["1977-01-01":"2026-06-30"]
    sB = sB.loc["1977-01-01":"2026-06-30"]

    bhB  = portfolio_bh(rB)
    timB = portfolio_timing_correct(rB, sB)

    # SP500 for reference
    spy_rets = spy.pct_change().dropna().loc["1973-01-01":"2026-06-30"]

    def annual(s: pd.Series, label: str) -> pd.Series:
        return (s.resample("A").apply(lambda x: (1 + x).prod() - 1)
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
