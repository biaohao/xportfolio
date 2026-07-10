"""
strategy.py — Sub-Task 3
Core SMA timing engine implementing Faber's GTAA strategy.

Works identically across monthly, weekly, and daily frequencies.
No hardcoded frequency assumptions — all parameters come from config.yaml.

Public API:
    run_backtest(prices_df, frequency, sma_period, cash_returns) → BacktestResult
    resample_prices(daily_df, frequency)
    compute_sma(prices, period)
    generate_signals(prices, sma_period)
    compute_period_returns(prices)
    apply_timing_model(returns, signals, cash_returns, weights)
    apply_buy_hold(returns, weights)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Supported frequency aliases for pandas resample
_RESAMPLE_MAP = {
    "monthly": "M",   # month-end
    "weekly": "W-FRI",  # Friday close
    "daily": "D",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """All outputs from a single backtest run."""
    frequency: str
    sma_period: int

    # Per-period portfolio returns (not cumulative)
    returns_bh: pd.Series       # Buy & Hold
    returns_timing: pd.Series   # Timing model

    # Per-asset signals at each rebalance date (1 = invested, 0 = cash)
    signals: pd.DataFrame

    # Per-asset per-period returns (asset level, before portfolio aggregation)
    asset_returns: pd.DataFrame

    # Actual cash return per period (T-bill proxy)
    cash_returns: pd.Series


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def resample_prices(daily_df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample a daily price DataFrame to the target frequency.

    For 'daily': returns the input unchanged (already daily).
    For 'weekly': takes Friday close (W-FRI).
    For 'monthly': takes last trading day of each month (M).

    NaN values are forward-filled within each resample period before taking
    the period-end value, so weekends/holidays do not create gaps.
    """
    if frequency not in _RESAMPLE_MAP:
        raise ValueError(
            f"Unsupported frequency '{frequency}'. "
            f"Choose from: {list(_RESAMPLE_MAP.keys())}"
        )

    if frequency == "daily":
        return daily_df.copy()

    rule = _RESAMPLE_MAP[frequency]
    # ffill daily data first, then take last value of each period
    resampled = daily_df.ffill().resample(rule).last()
    # Drop rows where ALL assets are NaN (empty periods, e.g. holidays straddling a week boundary)
    resampled = resampled.dropna(how="all")
    return resampled


def compute_sma(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Compute a simple moving average over `period` periods for each column.
    The rolling window is backward-looking (no look-ahead).
    Returns NaN for the first (period-1) observations.
    """
    return prices.rolling(window=period, min_periods=period).mean()


def generate_signals(prices: pd.DataFrame, sma_period: int) -> pd.DataFrame:
    """
    Generate binary signals for each asset:
        signal[t] = 1  if price[t] > SMA[t]
        signal[t] = 0  otherwise

    IMPORTANT: The signal at time T is applied to the return at time T+1.
    This shift is applied here so callers work with aligned data without
    needing to remember the look-ahead rule.

    Returns a DataFrame of 0/1 signals shifted forward by 1 period.
    """
    sma = compute_sma(prices, sma_period)

    # Raw signal: 1 where price is strictly above SMA, else 0
    raw_signal = (prices > sma).astype(float)

    # Shift forward: signal observed at end of T → applied to return of T+1
    signal_shifted = raw_signal.shift(1)

    return signal_shifted


def compute_period_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple period-over-period returns: (P[t] - P[t-1]) / P[t-1].
    First row is NaN (no prior period).
    """
    return prices.pct_change()


def _normalise_weights(weights_raw: dict[str, float] | str,
                       available_cols: list[str]) -> dict[str, float]:
    """
    Resolve weights for the available columns.
    'equal' → 1/N for each available asset.
    dict → use provided weights, filtering to available columns and renormalising.
    """
    if weights_raw == "equal":
        n = len(available_cols)
        return {col: 1.0 / n for col in available_cols}

    filtered = {k: v for k, v in weights_raw.items() if k in available_cols}
    total = sum(filtered.values())
    if total == 0:
        n = len(available_cols)
        return {col: 1.0 / n for col in available_cols}
    return {k: v / total for k, v in filtered.items()}


def apply_buy_hold(
    returns: pd.DataFrame,
    weights_cfg: dict[str, float] | str = "equal",
) -> pd.Series:
    """
    Compute equal-weight (or configured-weight) Buy & Hold portfolio returns.

    At each period, the weight is renormalised across assets that have valid
    (non-NaN) returns, so the portfolio handles staggered asset start dates
    gracefully.

    Returns a Series of portfolio-level period returns.
    """
    port_returns = pd.Series(index=returns.index, dtype=float)

    for dt in returns.index:
        row = returns.loc[dt]
        available = row.dropna()
        if available.empty:
            port_returns[dt] = np.nan
            continue
        w = _normalise_weights(weights_cfg, list(available.index))
        port_returns[dt] = sum(available[k] * v for k, v in w.items())

    return port_returns


def apply_timing_model(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    cash_returns: pd.Series,
    weights_cfg: dict[str, float] | str = "equal",
) -> pd.Series:
    """
    Compute Faber's timing-model portfolio returns.

    For each period:
      - If signal[asset] = 1 → asset earns its return × weight
      - If signal[asset] = 0 → that weight earns the cash return

    Weights are renormalised at each period across assets with valid data.

    Returns a Series of portfolio-level period returns.
    """
    port_returns = pd.Series(index=returns.index, dtype=float)

    for dt in returns.index:
        ret_row = returns.loc[dt]
        sig_row = signals.loc[dt] if dt in signals.index else pd.Series(dtype=float)
        cash_ret = cash_returns.get(dt, 0.0)
        if pd.isna(cash_ret):
            cash_ret = 0.0

        # Assets with valid return data at this period
        available = ret_row.dropna().index.tolist()
        # Filter: only assets whose signal is also available
        available = [a for a in available if a in sig_row.index and not pd.isna(sig_row[a])]

        if not available:
            port_returns[dt] = np.nan
            continue

        w = _normalise_weights(weights_cfg, available)
        period_return = 0.0
        for asset in available:
            sig = sig_row[asset]
            asset_ret = ret_row[asset]
            if sig == 1.0:
                period_return += w[asset] * asset_ret
            else:
                period_return += w[asset] * cash_ret

        port_returns[dt] = period_return

    return port_returns


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_backtest(
    prices_df: pd.DataFrame,
    cash_col: str,
    frequency: str,
    sma_period: int,
    weights_cfg: dict[str, float] | str = "equal",
) -> BacktestResult:
    """
    Run a full backtest for a given frequency and SMA period.

    Parameters
    ----------
    prices_df   : Daily (or pre-resampled) price DataFrame. Must include
                  asset columns AND a cash proxy column.
    cash_col    : Column name for the cash/T-bill proxy (e.g. 'BIL').
    frequency   : 'monthly', 'weekly', or 'daily'.
    sma_period  : Number of periods for the SMA look-back.
    weights_cfg : 'equal' or a dict of ticker → target weight.

    Returns
    -------
    BacktestResult with returns, signals, and asset-level returns.
    """
    # 1. Resample to target frequency
    resampled = resample_prices(prices_df, frequency)

    # 2. Separate asset prices from cash proxy
    if cash_col not in resampled.columns:
        raise ValueError(f"Cash proxy column '{cash_col}' not found in prices DataFrame.")

    asset_cols = [c for c in resampled.columns if c != cash_col]
    asset_prices = resampled[asset_cols]
    cash_prices = resampled[cash_col]

    # 3. Per-period returns
    asset_returns = compute_period_returns(asset_prices)
    cash_returns_raw = compute_period_returns(cash_prices)

    # 4. Signals (already shifted by 1 period inside generate_signals)
    signals = generate_signals(asset_prices, sma_period)

    # 5. Portfolio returns
    returns_bh = apply_buy_hold(asset_returns, weights_cfg)
    returns_timing = apply_timing_model(
        asset_returns, signals, cash_returns_raw, weights_cfg
    )

    # Drop the first row (NaN from pct_change) and the SMA warm-up rows
    # where signals are entirely NaN
    first_valid = signals.first_valid_index()
    if first_valid is not None:
        returns_bh = returns_bh.loc[first_valid:]
        returns_timing = returns_timing.loc[first_valid:]
        asset_returns = asset_returns.loc[first_valid:]
        signals = signals.loc[first_valid:]
        cash_returns_raw = cash_returns_raw.loc[first_valid:]

    return BacktestResult(
        frequency=frequency,
        sma_period=sma_period,
        returns_bh=returns_bh,
        returns_timing=returns_timing,
        signals=signals,
        asset_returns=asset_returns,
        cash_returns=cash_returns_raw,
    )
