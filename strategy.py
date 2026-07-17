"""
strategy.py — Sub-Task 3
Core SMA timing engine implementing Faber's GTAA strategy.

Works identically across monthly, weekly, and daily frequencies.
No hardcoded frequency assumptions — all parameters come from config.yaml.

Public API:
    run_backtest(prices_df, frequency, sma_period, cash_returns) → BacktestResult
    resample_prices(daily_df, frequency)
    compute_sma(prices, period)
    compute_signals(prices, sma_period)
    compute_period_returns(prices)
    apply_timing_model(returns, signals, cash_returns, weights, rebalance_period_months)
    apply_buy_hold(returns, weights)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Supported frequency aliases for pandas resample
_RESAMPLE_MAP = {
    "monthly": "ME",   # month-end
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
    rebalance_period_months: int

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
    For 'monthly': takes last trading day of each month (ME).

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


def compute_signals(prices: pd.DataFrame, sma_period: int) -> pd.DataFrame:
    """
    Compute binary SMA signals for each asset:
        raw_signal[t] = 1  if price[t] > SMA[t],  else 0

    Per Faber: "all entry and exit prices are on the day of the signal at
    the close."  The raw signal is therefore shifted forward by 1 period so
    that signals[t] holds the signal computed at close of t-1.  The
    simulation can then trade at close of t-1 and earn the return of period t.

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

    Weights are renormalised each period across assets that have valid (non-NaN)
    returns, so the portfolio handles staggered asset start dates gracefully.
    Assets with NaN returns in a given period are excluded from that period's
    weight pool — equivalent to the row-by-row loop but fully vectorised.

    Returns a Series of portfolio-level period returns.
    """
    # Build a (dates × assets) weight matrix from the static config
    static_w = _normalise_weights(weights_cfg, list(returns.columns))
    w_vec = pd.Series(static_w)                    # one weight per column

    # Mask: 1.0 where return is valid, NaN elsewhere
    valid = returns.notna().astype(float).replace(0.0, np.nan)

    # Per-row effective weight: static weight, zeroed where return is NaN,
    # then renormalised so the row always sums to 1 across available assets.
    w_matrix = valid.multiply(w_vec, axis=1)        # zero out missing assets
    row_sum = w_matrix.sum(axis=1)                  # sum of valid weights per row
    w_matrix = w_matrix.divide(row_sum, axis=0)     # renormalise each row to 1

    # Weighted sum — NaN returns treated as 0 after weights are set
    return (returns.fillna(0.0) * w_matrix).sum(axis=1).where(row_sum > 0, np.nan)


def apply_timing_model(
    prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    signals: pd.DataFrame,
    cash_returns: pd.Series,
    weights_cfg: dict[str, float] | str = "equal",
    rebalance_period_months: int = 1,
) -> pd.Series:
    """
    Dollar-value simulation of Faber's timing model with configurable rebalance cadence.

    Rules
    -----
    Every period (month-end):
      - Evaluate the SMA signal for each asset (already shifted in `signals`).
      - EXIT signal (1→0): sell the full position in that asset → cash.
      - ENTRY signal (0→1): deploy min(target_weight × portfolio_value, cash_balance)
        from cash into that asset.
      - No other trades happen mid-period.

    Every N months (rebalance_period_months):
      - True-up all *currently invested* assets to equal weight of total portfolio value.
      - Any excess cash stays in cash; any asset over-weight is trimmed to cash.

    When rebalance_period_months=1 this reproduces Faber's original monthly behaviour
    exactly (every period is both a signal check AND a rebalance).

    Parameters
    ----------
    prices              : Monthly asset price DataFrame (used to detect period index).
    asset_returns       : Per-period simple returns for each asset.
    signals             : Binary 0/1 signals, already shifted by 1 period.
    cash_returns        : Per-period T-bill / cash proxy returns.
    weights_cfg         : 'equal' or dict of target weights.
    rebalance_period_months : How many periods between full equal-weight rebalances.

    Returns
    -------
    pd.Series of portfolio-level period returns.
    """
    dates = asset_returns.index
    assets = asset_returns.columns.tolist()
    n_assets = len(assets)

    # Resolve target weight per asset (equal weight across all assets, not just invested)
    target_w = _normalise_weights(weights_cfg, assets)  # e.g. 0.20 each for 5 assets

    # --- Portfolio state ---
    # holdings[asset] = current dollar value invested in each asset
    # cash            = dollar value sitting in T-bill / cash proxy
    holdings: dict[str, float] = {a: 0.0 for a in assets}
    portfolio_value = 1.0
    cash = portfolio_value  # start fully in cash

    port_returns = pd.Series(index=dates, dtype=float)

    # Track which period index this is (for rebalance cadence)
    period_count = 0

    # Track prior signal to detect entry/exit transitions
    prev_signals: dict[str, float] = {a: np.nan for a in assets}

    for dt in dates:
        ret_row = asset_returns.loc[dt]
        sig_row = signals.loc[dt] if dt in signals.index else pd.Series(dtype=float)
        cash_ret = cash_returns.get(dt, 0.0)
        if pd.isna(cash_ret):
            cash_ret = 0.0

        # ----------------------------------------------------------------
        # 1. Process entry/exit signals — trade at the close of t-1.
        #
        # signals[t] = raw signal computed at close of t-1 (compute_signals
        # shifts by 1).  Faber: "all entry and exit prices are on the day of
        # the signal at the close."  So we set the position now (close of t-1
        # = start of period t) and then let it earn the return of period t.
        # ----------------------------------------------------------------
        portfolio_value = cash + sum(holdings.values())

        for asset in assets:
            if asset not in sig_row.index or pd.isna(sig_row[asset]):
                continue

            sig = sig_row[asset]
            prev = prev_signals[asset]

            if pd.isna(prev):
                # First valid signal period — enter if signal=1, else stay in cash
                if sig == 1.0 and holdings[asset] == 0.0:
                    deploy = min(target_w[asset] * portfolio_value, cash)
                    holdings[asset] += deploy
                    cash -= deploy
                prev_signals[asset] = sig
                continue

            if prev == 1.0 and sig == 0.0:
                # EXIT: sell entire position → cash
                cash += holdings[asset]
                holdings[asset] = 0.0

            elif prev == 0.0 and sig == 1.0:
                # ENTRY: deploy target weight of current portfolio from cash
                deploy = min(target_w[asset] * portfolio_value, cash)
                holdings[asset] += deploy
                cash -= deploy

            prev_signals[asset] = sig

        # ----------------------------------------------------------------
        # 2. Periodic rebalance — true-up at the same close of t-1,
        #    so the rebalanced weights are exposed to period-t returns.
        # ----------------------------------------------------------------
        period_count += 1
        is_rebalance_period = (period_count % rebalance_period_months == 0)

        if is_rebalance_period:
            portfolio_value = cash + sum(holdings.values())
            invested_assets = [a for a in assets if holdings[a] > 0]

            if invested_assets:
                # Each invested asset should hold target_w[asset] × portfolio_value.
                # Trim over-weight to cash first, then deploy to under-weight.
                for asset in invested_assets:
                    target_val = target_w[asset] * portfolio_value
                    diff = holdings[asset] - target_val
                    if diff > 0:
                        holdings[asset] = target_val
                        cash += diff

                for asset in invested_assets:
                    target_val = target_w[asset] * portfolio_value
                    diff = target_val - holdings[asset]
                    if diff > 0:
                        deploy = min(diff, cash)
                        holdings[asset] += deploy
                        cash -= deploy

        # ----------------------------------------------------------------
        # 3. Grow holdings and cash by period-t returns (close t-1 → close t).
        # ----------------------------------------------------------------
        value_before = cash + sum(holdings.values())

        cash *= (1.0 + cash_ret)
        for asset in assets:
            if holdings[asset] > 0:
                r = ret_row.get(asset, np.nan)
                if pd.isna(r):
                    r = 0.0
                holdings[asset] *= (1.0 + r)

        # ----------------------------------------------------------------
        # 4. Record the period-t return.
        # ----------------------------------------------------------------
        portfolio_value_new = cash + sum(holdings.values())
        port_returns[dt] = (portfolio_value_new / value_before) - 1.0 if value_before > 0 else 0.0
        portfolio_value = portfolio_value_new

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
    rebalance_period_months: int = 1,
) -> BacktestResult:
    """
    Run a full backtest for a given frequency and SMA period.

    Parameters
    ----------
    prices_df              : Daily (or pre-resampled) price DataFrame. Must include
                             asset columns AND a cash proxy column.
    cash_col               : Column name for the cash/T-bill proxy (e.g. 'BIL').
    frequency              : 'monthly', 'weekly', or 'daily'.
    sma_period             : Number of periods for the SMA look-back.
    weights_cfg            : 'equal' or a dict of ticker → target weight.
    rebalance_period_months: Periods between full equal-weight rebalances.
                             1 = Faber original (every period).
                             12 = annual. Only meaningful for monthly frequency.

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

    # 4. Signals (already shifted by 1 period inside compute_signals)
    signals = compute_signals(asset_prices, sma_period)

    # 5. Portfolio returns
    returns_bh = apply_buy_hold(asset_returns, weights_cfg)

    # rebalance_period_months only applies meaningfully to monthly frequency;
    # for weekly/daily keep the original per-period behaviour (period=1).
    reb_months = rebalance_period_months if frequency == "monthly" else 1
    returns_timing = apply_timing_model(
        prices=asset_prices,
        asset_returns=asset_returns,
        signals=signals,
        cash_returns=cash_returns_raw,
        weights_cfg=weights_cfg,
        rebalance_period_months=reb_months,
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
        rebalance_period_months=reb_months,
        returns_bh=returns_bh,
        returns_timing=returns_timing,
        signals=signals,
        asset_returns=asset_returns,
        cash_returns=cash_returns_raw,
    )
