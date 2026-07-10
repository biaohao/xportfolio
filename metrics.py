"""
metrics.py — Sub-Task 4 (part 1)
Performance metric functions for TAA backtest results.

All functions accept a pandas Series of period-level simple returns
(e.g. monthly, weekly, or daily) and an optional frequency string.

Public API:
    annualisation_factor(frequency)  → int (periods per year)
    cagr(returns)
    annualised_volatility(returns, frequency)
    sharpe_ratio(returns, rf_series_or_rate, frequency)
    max_drawdown(returns)
    calmar_ratio(returns, frequency)
    pct_time_invested(signals_df)
    compute_all_metrics(returns, rf, frequency, signals_df)  → dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ANN_FACTORS = {
    "monthly": 12,
    "weekly": 52,
    "daily": 252,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def annualisation_factor(frequency: str) -> int:
    """Return the number of periods per year for a given frequency."""
    if frequency not in _ANN_FACTORS:
        raise ValueError(
            f"Unknown frequency '{frequency}'. Choose from: {list(_ANN_FACTORS.keys())}"
        )
    return _ANN_FACTORS[frequency]


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaN and inf values from a return series."""
    s = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return s


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (end_value / start_value) ^ (1 / years) - 1
    Computed from period returns, inferred from the index dates.
    """
    r = _clean(returns)
    if r.empty or len(r) < 2:
        return np.nan

    growth = (1 + r).prod()
    # Infer number of years from index
    n_days = (r.index[-1] - r.index[0]).days
    years = n_days / 365.25
    if years <= 0:
        return np.nan
    return float(growth ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series, frequency: str) -> float:
    """
    Annualised standard deviation of period returns.
    vol_annual = std(returns) × √(periods_per_year)
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    factor = annualisation_factor(frequency)
    return float(r.std() * np.sqrt(factor))


def sharpe_ratio(
    returns: pd.Series,
    rf: pd.Series | float,
    frequency: str,
) -> float:
    """
    Annualised Sharpe ratio.
    If rf is a Series (same index as returns), compute excess returns per period.
    If rf is a float (annual rate), convert to per-period rate.
    Sharpe = mean(excess_returns) / std(excess_returns) × √(periods_per_year)
    """
    r = _clean(returns)
    if r.empty:
        return np.nan

    factor = annualisation_factor(frequency)

    if isinstance(rf, pd.Series):
        rf_aligned = rf.reindex(r.index).ffill().fillna(0.0)
        excess = r - rf_aligned
    else:
        # Convert annual rate to per-period rate
        rf_per_period = (1 + float(rf)) ** (1.0 / factor) - 1.0
        excess = r - rf_per_period

    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(factor))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative number, e.g. -0.46 means -46%).
    Computed on the cumulative return (wealth index).
    """
    r = _clean(returns)
    if r.empty:
        return np.nan

    wealth = (1 + r).cumprod()
    rolling_max = wealth.cummax()
    dd = (wealth - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(returns: pd.Series, frequency: str) -> float:
    """
    Calmar ratio = CAGR / |max_drawdown|.
    Higher is better; common target for trend-following is 0.5–2.0.
    """
    c = cagr(returns)
    md = max_drawdown(returns)
    if pd.isna(c) or pd.isna(md) or md == 0:
        return np.nan
    return float(c / abs(md))


def pct_time_invested(signals_df: pd.DataFrame) -> float:
    """
    Fraction of periods where the portfolio is at least partially invested.
    = mean of per-period mean signal across all assets.
    Value of 1.0 → always fully invested; 0.7 → 70% of the time on average.
    """
    if signals_df.empty:
        return np.nan
    return float(signals_df.mean(axis=1).mean())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Return the full drawdown time series (for plotting).
    Each value is the drawdown from the prior peak, as a fraction.
    """
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    wealth = (1 + r).cumprod()
    rolling_max = wealth.cummax()
    return (wealth - rolling_max) / rolling_max


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Return the cumulative wealth index (starting at 1.0).
    """
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    return (1 + r).cumprod()


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    returns: pd.Series,
    rf: pd.Series | float,
    frequency: str,
    signals_df: pd.DataFrame | None = None,
    label: str = "",
) -> dict:
    """
    Compute the full set of metrics used in the paper and return as a dict.
    """
    c = cagr(returns)
    v = annualised_volatility(returns, frequency)
    s = sharpe_ratio(returns, rf, frequency)
    md = max_drawdown(returns)
    cal = calmar_ratio(returns, frequency)
    pct_inv = pct_time_invested(signals_df) if signals_df is not None else np.nan

    r = _clean(returns)
    n_years = (r.index[-1] - r.index[0]).days / 365.25 if len(r) > 1 else np.nan
    start_date = r.index[0].date() if not r.empty else None
    end_date = r.index[-1].date() if not r.empty else None

    return {
        "label": label,
        "frequency": frequency,
        "start_date": start_date,
        "end_date": end_date,
        "years": round(n_years, 1) if not np.isnan(n_years) else np.nan,
        "cagr": round(c * 100, 2) if not np.isnan(c) else np.nan,
        "volatility": round(v * 100, 2) if not np.isnan(v) else np.nan,
        "sharpe": round(s, 3) if not np.isnan(s) else np.nan,
        "max_drawdown": round(md * 100, 2) if not np.isnan(md) else np.nan,
        "calmar": round(cal, 3) if not np.isnan(cal) else np.nan,
        "pct_time_invested": round(pct_inv * 100, 1) if not np.isnan(pct_inv) else np.nan,
    }
