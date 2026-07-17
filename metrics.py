"""
metrics.py — Sub-Task 4 (part 1)
Performance metric functions for TAA backtest results.

All functions accept a pandas Series of period-level simple returns
(e.g. monthly, weekly, or daily) and an optional frequency string.

Public API:
    annualisation_factor(frequency)  → int (periods per year)
    wealth_index(returns)            → overflow-safe cumulative wealth Series
    cagr(returns)
    annualised_volatility(returns, frequency)
    sharpe_ratio(returns, rf_series_or_rate, frequency)
    max_drawdown(returns)
    calmar_ratio(returns, frequency)
    ulcer_index(returns)
    win_rate(returns)
    best_worst_year(returns)
    dollars_to(returns)
    pct_time_invested(signals_df)
    drawdown_series(returns)
    cumulative_returns(returns)
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
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def wealth_index(returns: pd.Series) -> pd.Series:
    """
    Cumulative wealth index starting at 1.0, computed in log-space to avoid
    float64 overflow on very long histories (e.g. Shiller 1871–2025).

    Uses log1p for numerical accuracy near zero returns and clips at -0.9999
    to prevent log(0) on a near-total-loss period.
    """
    r = _clean(returns).clip(-0.9999)
    return np.exp(np.log1p(r).cumsum())


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = terminal_wealth ^ (1 / years) - 1
    Years are inferred from the index date range.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    years = (r.index[-1] - r.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float(wealth_index(r).iloc[-1] ** (1.0 / years) - 1.0)


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
    If rf is a float (annual rate), convert to per-period rate first.
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
        rf_per_period = (1 + float(rf)) ** (1.0 / factor) - 1.0
        excess = r - rf_per_period

    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(factor))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative number, e.g. -0.46 means -46%).
    Computed on the log-space wealth index.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    w = wealth_index(r)
    dd = (w - w.cummax()) / w.cummax()
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


def ulcer_index(returns: pd.Series) -> float:
    """
    Ulcer Index — root mean square of the drawdown series (in %).
    Measures both depth and duration of drawdowns; lower is better.
    Unlike max_drawdown, a long shallow drawdown scores worse than a
    brief deep one of the same peak loss.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    w = wealth_index(r)
    dd_pct = ((w - w.cummax()) / w.cummax()) * 100
    return float(np.sqrt((dd_pct ** 2).mean()))


def win_rate(returns: pd.Series) -> float:
    """
    Percentage of periods with a positive return.
    e.g. 0.62 means 62% of months were up months.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    return float((r > 0).mean() * 100)


def best_worst_year(returns: pd.Series) -> tuple[str, str]:
    """
    Return formatted strings for the best and worst calendar year.
    e.g. ("+32.4% (1995)", "-38.1% (2008)")
    """
    r = _clean(returns)
    if r.empty:
        return "n/a", "n/a"
    annual = r.resample("YE").apply(lambda x: (1 + x).prod() - 1).dropna()
    if annual.empty:
        return "n/a", "n/a"
    best_v, worst_v = annual.max(), annual.min()
    best_y, worst_y = annual.idxmax().year, annual.idxmin().year
    fmt = lambda v, y: f"{'+' if v >= 0 else ''}{v * 100:.1f}% ({y})"
    return fmt(best_v, best_y), fmt(worst_v, worst_y)


def dollars_to(returns: pd.Series) -> float:
    """
    Terminal value of $1 invested at the start of the return series.
    e.g. 4.72 means $1 grew to $4.72.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    return round(float(wealth_index(r).iloc[-1]), 2)


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
    Full drawdown time series for plotting.
    Each value is the current drawdown from the prior peak, as a fraction.
    """
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    w = wealth_index(r)
    return (w - w.cummax()) / w.cummax()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Cumulative wealth index starting at 1.0 (alias for wealth_index on clean data).
    """
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    return wealth_index(r)


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
    c   = cagr(returns)
    v   = annualised_volatility(returns, frequency)
    s   = sharpe_ratio(returns, rf, frequency)
    md  = max_drawdown(returns)
    cal = calmar_ratio(returns, frequency)
    ui  = ulcer_index(returns)
    wr  = win_rate(returns)
    pct_inv = pct_time_invested(signals_df) if signals_df is not None else np.nan

    r = _clean(returns)
    n_years    = (r.index[-1] - r.index[0]).days / 365.25 if len(r) > 1 else np.nan
    start_date = r.index[0].date() if not r.empty else None
    end_date   = r.index[-1].date() if not r.empty else None

    def _r(val, decimals=2):
        return round(val, decimals) if not np.isnan(val) else np.nan

    return {
        "label":            label,
        "frequency":        frequency,
        "start_date":       start_date,
        "end_date":         end_date,
        "years":            _r(n_years, 1),
        "cagr":             _r(c * 100),
        "volatility":       _r(v * 100),
        "sharpe":           _r(s, 3),
        "max_drawdown":     _r(md * 100),
        "calmar":           _r(cal, 3),
        "ulcer_index":      _r(ui),
        "win_rate":         _r(wr, 1),
        "pct_time_invested": _r(pct_inv * 100, 1),
    }
