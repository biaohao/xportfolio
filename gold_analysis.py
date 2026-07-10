"""
gold_analysis.py — Analyse gold as a standalone asset and as a portfolio addition.

Produces:
  results/gold_statistics.csv     — CAGR, vol, Sharpe, max-DD per decade since 1973
  results/gold_correlations.csv   — rolling and full-period correlation with other assets
  results/gold_timing.csv         — B&H vs SMA-timing for gold alone (monthly/weekly/daily)
  Console summary table

Data sources used:
  Monthly (1973–):  data/processed/gold_monthly.csv  (GitHub datasets/gold-prices)
  Daily (2000–):    data/processed/gold_daily.csv    (GCF splice + GLD)
  Other assets:     data/processed/prices_daily.csv  (ETF adjusted close)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
RESULTS = Path("results")


# ---------------------------------------------------------------------------
# Metric helpers (same convention as metrics.py)
# ---------------------------------------------------------------------------

def _freq_factor(returns: pd.Series) -> float:
    """Infer annualisation factor from median gap between observations."""
    if len(returns) < 2:
        return 252.0
    median_days = returns.index.to_series().diff().dt.days.median()
    if median_days is None or np.isnan(median_days):
        return 252.0
    if median_days >= 25:   # monthly
        return 12.0
    if median_days >= 5:    # weekly
        return 52.0
    return 252.0             # daily


def cagr(ret: pd.Series) -> float:
    n = _freq_factor(ret)
    total = (1 + ret).prod()
    years = len(ret) / n
    return float(total ** (1 / years) - 1) if years > 0 else float("nan")


def vol(ret: pd.Series) -> float:
    return float(ret.std() * np.sqrt(_freq_factor(ret)))


def sharpe(ret: pd.Series, rf: float = 0.0) -> float:
    excess = ret - rf / _freq_factor(ret)
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(_freq_factor(ret)))


def max_dd(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())


def calmar(ret: pd.Series) -> float:
    c = cagr(ret)
    md = max_dd(ret)
    return float(c / abs(md)) if md != 0 else float("nan")


def stats_row(label: str, ret: pd.Series) -> dict:
    return {
        "label": label,
        "start": ret.index[0].date(),
        "end": ret.index[-1].date(),
        "years": round(len(ret) / _freq_factor(ret), 1),
        "CAGR%": round(cagr(ret) * 100, 2),
        "Vol%": round(vol(ret) * 100, 2),
        "Sharpe": round(sharpe(ret), 3),
        "MaxDD%": round(max_dd(ret) * 100, 2),
        "Calmar": round(calmar(ret), 3),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_gold_monthly() -> pd.Series:
    path = PROC / "gold_monthly.csv"
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.name = "Gold"
    return s.sort_index()


def load_gold_daily() -> pd.Series:
    path = PROC / "gold_daily.csv"
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.name = "Gold"
    return s.sort_index()


def load_etf_prices() -> pd.DataFrame:
    path = PROC / "prices_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


# ---------------------------------------------------------------------------
# 1. Decade-by-decade gold statistics (monthly)
# ---------------------------------------------------------------------------

def decade_stats(gold_monthly: pd.Series) -> pd.DataFrame:
    rets = gold_monthly.pct_change().dropna()

    rows = []
    # Full period
    rows.append(stats_row("Full (1973–2026)", rets))

    # Decade slices
    decades = [
        ("1970s (1973–1979)", "1973-01-01", "1979-12-31"),
        ("1980s", "1980-01-01", "1989-12-31"),
        ("1990s", "1990-01-01", "1999-12-31"),
        ("2000s", "2000-01-01", "2009-12-31"),
        ("2010s", "2010-01-01", "2019-12-31"),
        ("2020s (2020–2026)", "2020-01-01", "2026-12-31"),
    ]
    for label, start, end in decades:
        slice_ = rets.loc[start:end]
        if len(slice_) < 6:
            continue
        rows.append(stats_row(label, slice_))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Correlation analysis (daily, rolling 3-year)
# ---------------------------------------------------------------------------

def correlation_analysis(gold_daily: pd.Series, etf_prices: pd.DataFrame) -> pd.DataFrame:
    """Full-period pairwise correlation between gold and each ETF."""
    # Align on common dates, compute log returns
    common = etf_prices.join(gold_daily.rename("GLD_calc"), how="inner").dropna(how="all")
    rets = common.pct_change().dropna(how="all")

    # Only keep assets with >80% non-NaN overlap with gold
    gold_col = "GLD_calc" if "GLD_calc" in rets.columns else "GLD"
    if gold_col not in rets.columns:
        return pd.DataFrame()

    rows = []
    for col in rets.columns:
        if col == gold_col:
            continue
        pair = rets[[gold_col, col]].dropna()
        if len(pair) < 30:
            continue
        corr_full = pair.corr().iloc[0, 1]
        # Rolling 3-year (756 trading days)
        roll = pair[gold_col].rolling(756).corr(pair[col]).dropna()
        rows.append({
            "Asset": col,
            "Full_period_corr_with_gold": round(corr_full, 3),
            "Min_3yr_rolling_corr": round(roll.min(), 3),
            "Max_3yr_rolling_corr": round(roll.max(), 3),
            "Mean_3yr_rolling_corr": round(roll.mean(), 3),
            "Observations": len(pair),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Gold SMA timing (monthly)
# ---------------------------------------------------------------------------

def gold_timing_monthly(gold_monthly: pd.Series, sma_period: int = 10) -> pd.DataFrame:
    """Apply SMA timing rule to gold alone; compare B&H vs timing."""
    prices = gold_monthly.copy()
    rets = prices.pct_change()
    sma = prices.rolling(sma_period).mean()
    # Signal: 1 if price > SMA, else 0; shifted by 1 to avoid look-ahead
    signal = (prices > sma).astype(int).shift(1)

    rows = []
    for label, start, end in [
        ("Full 1973–2026", "1973-01-01", "2026-12-31"),
        ("Paper period 1973–2012", "1973-01-01", "2012-12-31"),
        ("Modern 2007–2026", "2007-01-01", "2026-12-31"),
    ]:
        r = rets.loc[start:end].dropna()
        s = signal.loc[start:end].reindex(r.index)

        bh = r
        timing = r * s.fillna(0)

        if len(bh) < 12:
            continue
        rows.append({**stats_row(f"B&H — {label}", bh), "strategy": "B&H"})
        rows.append({**stats_row(f"Timing — {label}", timing), "strategy": "Timing", "pct_invested": round(s.mean() * 100, 1)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=== Gold Analysis ===\n")

    # Load data
    gold_monthly = load_gold_monthly()
    gold_daily = load_gold_daily()
    etf_prices = load_etf_prices()

    # 1. Decade statistics
    print("1. Decade-by-decade statistics (monthly, 1973–2026)")
    print("-" * 60)
    decade_df = decade_stats(gold_monthly)
    print(decade_df.to_string(index=False))
    decade_df.to_csv(RESULTS / "gold_statistics.csv", index=False)
    print(f"\n   → saved: results/gold_statistics.csv\n")

    # 2. Correlations
    print("2. Correlation with other assets (daily)")
    print("-" * 60)
    corr_df = correlation_analysis(gold_daily, etf_prices)
    if not corr_df.empty:
        print(corr_df.to_string(index=False))
        corr_df.to_csv(RESULTS / "gold_correlations.csv", index=False)
        print(f"\n   → saved: results/gold_correlations.csv\n")
    else:
        print("  (Could not compute correlations — check data availability)\n")

    # 3. Gold timing
    print("3. SMA-10 timing for gold alone (monthly)")
    print("-" * 60)
    timing_df = gold_timing_monthly(gold_monthly)
    print(timing_df[["label", "years", "CAGR%", "Vol%", "Sharpe", "MaxDD%"]].to_string(index=False))
    timing_df.to_csv(RESULTS / "gold_timing.csv", index=False)
    print(f"\n   → saved: results/gold_timing.csv\n")


if __name__ == "__main__":
    main()
