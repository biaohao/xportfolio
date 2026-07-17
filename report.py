"""
report.py — Sub-Task 4 (part 2)
Orchestrates backtests across all configured frequencies and produces
the comparison summary table.

Usage:
    python3 report.py                  # run all frequencies, print + save CSV
    python3 report.py --freq monthly   # run only one frequency

Outputs:
    results/summary_table.csv   — metrics for all strategies × frequencies
    Console: formatted comparison table
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from metrics import compute_all_metrics
from strategy import run_backtest

# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

from config import load_config, ROOT, CONFIG_PATH

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_window(cfg: dict) -> tuple[str | None, str | None]:
    """
    Read backtest_start_date and backtest_end_date from config.
    Returns (start, end) as ISO strings, or None when the value is 'latest'/null.
    """
    from datetime import date as _date

    def _resolve(val) -> str | None:
        if val is None or str(val).lower() == "latest":
            return None
        return str(val)

    start = _resolve(cfg["data"].get("backtest_start_date"))
    end = _resolve(cfg["data"].get("backtest_end_date"))
    return start, end


def load_prices(frequency: str, end: str | None = None) -> pd.DataFrame:
    """
    Load the full available price DataFrame for the given frequency, trimmed
    only at the end date (so the SMA warm-up always has access to pre-window
    data).  The start trim is applied *after* the backtest runs — see
    run_all_backtests() which slices results to backtest_start_date.

    - monthly : use spliced long-history series (broadest history)
    - weekly  : use ETF daily prices (resampled to weekly by strategy engine)
    - daily   : use ETF daily prices
    """
    if frequency == "monthly":
        path = PROCESSED / "prices_monthly_spliced.csv"
    else:
        path = PROCESSED / "prices_daily.csv"

    if not path.exists():
        log.error("Price file not found: %s — run fetch_etf.py / splice_data.py first.", path)
        sys.exit(1)

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Apply end trim on the price data so we don't use future prices
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if df.empty:
        log.error(
            "No data up to end date %s for frequency '%s'.",
            end or "latest", frequency,
        )
        sys.exit(1)

    return df


def resolve_rf(cfg: dict, cash_returns: pd.Series) -> pd.Series | float:
    """
    Return the risk-free rate per the config setting.
    'dynamic' → use actual T-bill returns from the backtest cash proxy.
    A numeric value → use that fixed annual rate.
    """
    rf_cfg = cfg["reporting"]["risk_free_rate"]
    if str(rf_cfg).lower() == "dynamic":
        return cash_returns
    return float(rf_cfg)


# ---------------------------------------------------------------------------
# Core reporting function
# ---------------------------------------------------------------------------

def _trim_result_to_window(result, bt_start: str | None) -> None:
    """
    Trim BacktestResult series in-place to backtest_start_date.
    The backtest runs over the full history (for SMA warm-up), then we
    discard any periods before bt_start so reported metrics cover exactly
    the requested window.
    """
    if not bt_start:
        return
    cutoff = pd.Timestamp(bt_start)
    for attr in ("returns_bh", "returns_timing", "cash_returns"):
        s = getattr(result, attr)
        setattr(result, attr, s[s.index >= cutoff])
    result.signals = result.signals[result.signals.index >= cutoff]
    result.asset_returns = result.asset_returns[result.asset_returns.index >= cutoff]


def run_all_backtests(
    frequencies: list[str],
    cfg: dict,
) -> tuple[list[dict], list]:
    """
    Run backtests for all requested frequencies.

    Returns
    -------
    rows    : list of metric dicts (one per strategy × frequency)
    results : list of BacktestResult objects (one per frequency), in the same
              frequency order — can be passed directly to plot.generate_all_plots()
              to avoid re-running the backtests a second time.

    SMA warm-up strategy:
      - Price data is loaded from the earliest available date (no start trim).
      - The backtest engine computes the SMA over the full history.
      - Results (returns, signals) are then trimmed to backtest_start_date so
        the first traded period is exactly at that date, with a fully-warmed SMA.
    """
    cash_ticker = cfg["cash_proxy"]
    sma_periods: dict = cfg["strategy"]["sma_periods"]
    weights_cfg = cfg["strategy"]["weights"]
    rebalance_period_months: int = int(cfg["strategy"].get("rebalance_period_months", 1))
    bt_start, bt_end = resolve_window(cfg)
    rows = []
    results = []

    for freq in frequencies:
        sma = sma_periods.get(freq)
        if sma is None:
            log.warning("No SMA period configured for frequency '%s'; skipping.", freq)
            continue

        log.info(
            "Running backtest: frequency=%s  SMA=%d  rebalance_every=%d_months  window=%s → %s",
            freq, sma, rebalance_period_months if freq == "monthly" else 1,
            bt_start or "earliest", bt_end or "latest",
        )
        # Load full history up to end date — start trim happens AFTER backtest
        prices = load_prices(freq, end=bt_end)

        if cash_ticker not in prices.columns:
            log.warning(
                "Cash proxy '%s' not in %s price data; skipping %s.",
                cash_ticker, freq, freq,
            )
            continue

        result = run_backtest(
            prices_df=prices,
            cash_col=cash_ticker,
            frequency=freq,
            sma_period=sma,
            weights_cfg=weights_cfg,
            rebalance_period_months=rebalance_period_months,
        )

        # Trim results to the requested backtest window start
        _trim_result_to_window(result, bt_start)

        if result.returns_bh.empty:
            log.warning(
                "No returns after trimming to window start %s for %s; skipping.",
                bt_start, freq,
            )
            continue

        results.append(result)

        rf = resolve_rf(cfg, result.cash_returns)

        # Buy & Hold metrics
        bh_metrics = compute_all_metrics(
            returns=result.returns_bh,
            rf=rf,
            frequency=freq,
            signals_df=None,
            label="Buy & Hold",
        )
        rows.append(bh_metrics)

        # Timing model metrics — label includes rebalance cadence for clarity
        reb = result.rebalance_period_months
        if reb == 1:
            timing_label = "Timing (GTAA monthly)"
        elif reb == 12:
            timing_label = "Timing (GTAA annual)"
        elif reb == 3:
            timing_label = "Timing (GTAA quarterly)"
        elif reb == 6:
            timing_label = "Timing (GTAA semi-annual)"
        else:
            timing_label = f"Timing (GTAA {reb}m rebal)"
        timing_metrics = compute_all_metrics(
            returns=result.returns_timing,
            rf=rf,
            frequency=freq,
            signals_df=result.signals,
            label=timing_label,
        )
        rows.append(timing_metrics)

        log.info(
            "  B&H:    CAGR=%5.2f%%  Vol=%5.2f%%  Sharpe=%5.3f  MaxDD=%6.2f%%",
            bh_metrics["cagr"], bh_metrics["volatility"],
            bh_metrics["sharpe"], bh_metrics["max_drawdown"],
        )
        log.info(
            "  Timing: CAGR=%5.2f%%  Vol=%5.2f%%  Sharpe=%5.3f  MaxDD=%6.2f%%  Time Invested=%5.1f%%",
            timing_metrics["cagr"], timing_metrics["volatility"],
            timing_metrics["sharpe"], timing_metrics["max_drawdown"],
            timing_metrics["pct_time_invested"],
        )
        log.info("")

    return rows, results


def print_summary_table(rows: list[dict]) -> None:
    """Pretty-print the summary table to the console."""
    df = pd.DataFrame(rows)
    col_order = [
        "frequency", "label", "start_date", "end_date", "years",
        "cagr", "volatility", "sharpe", "max_drawdown", "calmar",
        "pct_time_invested",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.columns = [
        "Freq", "Strategy", "Start", "End", "Years",
        "CAGR%", "Vol%", "Sharpe", "MaxDD%", "Calmar",
        "% Invested",
    ]

    # Print as a wide table
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))
    print()
    print("=" * 120)
    print("  GTAA Replication — Performance Summary")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    print()

    # Paper benchmarks for reference
    print("Paper benchmarks (Faber 2013, 5-asset, 1973-2012):")
    print("  Buy & Hold:    CAGR ~9.9%   Vol ~12.0%  Sharpe ~0.5   MaxDD ~-46%")
    print("  GTAA Timing:   CAGR ~10.5%  Vol  ~7.0%  Sharpe ~0.8   MaxDD ~-10%  ~70% invested")
    print()


def main(
    frequencies: list[str] | None = None,
) -> tuple[list[dict], list]:
    """
    Run all backtests, print the summary table, save CSV.

    Returns (rows, results) so the caller (main.py) can pass the
    BacktestResult objects directly to plot.main() without re-running.
    """
    cfg = load_config()

    if frequencies is None:
        frequencies = cfg["strategy"]["rebalance_frequencies"]

    RESULTS.mkdir(parents=True, exist_ok=True)

    rows, results = run_all_backtests(frequencies, cfg)

    if not rows:
        log.error("No backtest results produced.")
        return rows, results

    print_summary_table(rows)

    # Save to CSV
    summary_df = pd.DataFrame(rows)
    out_path = RESULTS / "summary_table.csv"
    summary_df.to_csv(out_path, index=False)
    log.info("Summary table saved → %s", out_path)

    return rows, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run GTAA backtests and produce summary table.")
    parser.add_argument(
        "--freq",
        choices=["monthly", "weekly", "daily"],
        default=None,
        help="Run a single frequency only (default: all configured frequencies).",
    )
    args = parser.parse_args()
    freqs = [args.freq] if args.freq else None
    main(frequencies=freqs)
