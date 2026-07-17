"""
signals.py — Sub-Task 6
Live BUY/SELL signal generator.

Fetches the latest price data for all configured assets and computes
the current SMA signal for each asset × frequency combination.

Usage:
    python3 signals.py                   # all configured frequencies
    python3 signals.py --freq monthly    # one frequency only

Outputs:
    results/current_signals.csv          — current signals with price, SMA, % vs SMA
    results/last_signals.csv             — copy of previous run (for change detection)
    Console: formatted signal table
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

from config import load_config, ROOT

RESULTS_DIR = ROOT / "results"

# Resample rules per frequency
_RESAMPLE = {
    "monthly": "ME",
    "weekly": "W-FRI",
    "daily": "D",
}

# Minimum history needed to compute the SMA (add a 50% buffer for safety)
_LOOKBACK_BUFFER = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_fresh(tickers: list[str], sma_period: int, frequency: str) -> pd.DataFrame:
    """
    Fetch enough recent history to compute the SMA.
    Returns a daily DataFrame of adjusted close prices.
    """
    # Number of calendar days needed (convert periods → calendar days)
    periods_needed = int(sma_period * _LOOKBACK_BUFFER)
    if frequency == "monthly":
        cal_days = periods_needed * 35      # ~1 month = 30-31 days + buffer
    elif frequency == "weekly":
        cal_days = periods_needed * 8       # ~1 week = 7 days + buffer
    else:
        cal_days = int(periods_needed * 1.5)  # daily + weekends

    start = (date.today() - timedelta(days=cal_days)).isoformat()

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns from multi-ticker download
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw[["Close"]] if "Close" in raw.columns else raw

    return raw.ffill()


def _compute_signal(prices_series: pd.Series, frequency: str, sma_period: int) -> dict:
    """
    Given a daily price series for one asset, resample to the target frequency
    and compute the current SMA signal.

    Returns a dict with: latest_price, sma_value, signal, pct_vs_sma, last_date.
    """
    if prices_series.dropna().empty:
        return {
            "latest_price": np.nan,
            "sma_value": np.nan,
            "signal": "N/A",
            "pct_vs_sma": np.nan,
            "last_date": None,
        }

    rule = _RESAMPLE[frequency]
    if frequency == "daily":
        resampled = prices_series.dropna()
    else:
        resampled = prices_series.ffill().resample(rule).last().dropna()

    if len(resampled) < sma_period:
        return {
            "latest_price": float(resampled.iloc[-1]) if not resampled.empty else np.nan,
            "sma_value": np.nan,
            "signal": "INSUFFICIENT DATA",
            "pct_vs_sma": np.nan,
            "last_date": resampled.index[-1].date() if not resampled.empty else None,
        }

    sma = resampled.rolling(window=sma_period).mean()
    latest_price = float(resampled.iloc[-1])
    latest_sma = float(sma.iloc[-1])
    last_date = resampled.index[-1].date()

    pct_vs_sma = (latest_price - latest_sma) / latest_sma * 100.0 if latest_sma else np.nan

    if latest_price > latest_sma:
        signal = "BUY / HOLD"
    else:
        signal = "SELL / CASH"

    return {
        "latest_price": round(latest_price, 4),
        "sma_value": round(latest_sma, 4),
        "signal": signal,
        "pct_vs_sma": round(pct_vs_sma, 2),
        "last_date": last_date,
    }


def _days_to_next_rebalance(frequency: str) -> int:
    """Approximate calendar days until the next rebalance date."""
    today = date.today()
    if frequency == "monthly":
        # Next month-end
        if today.month == 12:
            next_month_end = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            next_month_end = date(today.year, today.month + 1, 1) - timedelta(days=1)
        return (next_month_end - today).days
    elif frequency == "weekly":
        # Next Friday
        days_ahead = 4 - today.weekday()   # Friday = 4
        if days_ahead <= 0:
            days_ahead += 7
        return days_ahead
    else:
        return 0  # daily — today


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_signals(frequencies: list[str] | None = None) -> pd.DataFrame:
    cfg = load_config()

    if frequencies is None:
        frequencies = cfg["strategy"]["rebalance_frequencies"]

    asset_tickers: list[str] = list(cfg["assets"].keys())
    asset_names: dict = cfg["assets"]
    sma_periods: dict = cfg["strategy"]["sma_periods"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for freq in frequencies:
        sma = sma_periods.get(freq)
        if sma is None:
            log.warning("No SMA period configured for '%s'; skipping.", freq)
            continue

        log.info("Fetching live data for frequency=%s  SMA=%d …", freq, sma)
        prices_df = _fetch_fresh(asset_tickers, sma, freq)

        if prices_df.empty:
            log.error("Could not fetch live price data.")
            continue

        days_to_next = _days_to_next_rebalance(freq)

        for ticker in asset_tickers:
            if ticker not in prices_df.columns:
                log.warning("  %s not found in downloaded data; skipping.", ticker)
                continue

            sig = _compute_signal(prices_df[ticker], freq, sma)
            rows.append({
                "frequency": freq,
                "ticker": ticker,
                "name": asset_names.get(ticker, ticker),
                "latest_price": sig["latest_price"],
                "sma_value": sig["sma_value"],
                "signal": sig["signal"],
                "pct_vs_sma": sig["pct_vs_sma"],
                "last_date": sig["last_date"],
                "sma_period": sma,
                "days_to_next_rebalance": days_to_next,
                "generated_at": date.today().isoformat(),
            })

    if not rows:
        log.error("No signals generated.")
        return pd.DataFrame()

    signals_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Detect changes vs. last run
    # ------------------------------------------------------------------
    last_path = RESULTS_DIR / "last_signals.csv"
    current_path = RESULTS_DIR / "current_signals.csv"

    changed_col = []
    if last_path.exists():
        last_df = pd.read_csv(last_path)
        for _, row in signals_df.iterrows():
            mask = (
                (last_df["frequency"] == row["frequency"]) &
                (last_df["ticker"] == row["ticker"])
            )
            if mask.any():
                prev_signal = last_df.loc[mask, "signal"].iloc[0]
                changed_col.append("YES" if prev_signal != row["signal"] else "")
            else:
                changed_col.append("NEW")
    else:
        changed_col = [""] * len(signals_df)

    signals_df["signal_changed"] = changed_col

    # Rotate: current → last
    if current_path.exists():
        import shutil
        shutil.copy(current_path, last_path)

    # Save current
    signals_df.to_csv(current_path, index=False)
    log.info("Signals saved → %s", current_path)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    _print_signals(signals_df, cfg)
    return signals_df


def _print_signals(df: pd.DataFrame, cfg: dict) -> None:
    """Pretty-print the signal table to console."""
    freqs = df["frequency"].unique()
    print()
    print("=" * 100)
    print("  GTAA Live Signals — as of", date.today().isoformat())
    print("=" * 100)

    for freq in freqs:
        sub = df[df["frequency"] == freq].copy()
        sma = sub["sma_period"].iloc[0]
        days = sub["days_to_next_rebalance"].iloc[0]

        print(f"\n  [{freq.upper()}]  SMA-{sma}  |  Next rebalance in: {days} day(s)")
        print(f"  {'Ticker':<8}  {'Name':<35}  {'Price':>10}  {'SMA':>10}  {'% vs SMA':>10}  {'Signal':<16}  Changed?")
        print(f"  {'-'*8}  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*8}")

        for _, row in sub.iterrows():
            pct_str = f"{row['pct_vs_sma']:+.2f}%" if pd.notna(row['pct_vs_sma']) else "   N/A"
            price_str = f"{row['latest_price']:.4f}" if pd.notna(row['latest_price']) else "N/A"
            sma_str = f"{row['sma_value']:.4f}" if pd.notna(row['sma_value']) else "N/A"
            changed = row.get("signal_changed", "")
            print(f"  {row['ticker']:<8}  {row['name']:<35}  {price_str:>10}  {sma_str:>10}  {pct_str:>10}  {row['signal']:<16}  {changed}")

    print()
    print("=" * 100)
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Generate live BUY/SELL signals for GTAA assets.")
    parser.add_argument(
        "--freq",
        choices=["monthly", "weekly", "daily"],
        default=None,
        help="Show signals for a single frequency only.",
    )
    args = parser.parse_args()
    freqs = [args.freq] if args.freq else None
    generate_signals(frequencies=freqs)
