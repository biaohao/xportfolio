"""
portfolio_compare.py — Compare 5-asset (Faber original) vs 6-asset (+ Gold) portfolios
across three time windows and three frequencies.

Produces:
  results/portfolio_comparison.csv   — full comparison table
  Console: formatted table

The script re-runs the strategy engine in-process with two different asset lists,
reading config.yaml for all other parameters (SMA periods, backtest window, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from metrics import compute_all_metrics
from strategy import run_backtest
from report import load_prices, resolve_window, resolve_rf, _trim_result_to_window

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Portfolio definitions
# ---------------------------------------------------------------------------

PORTFOLIOS = {
    "5-asset (original)": ["SPY", "EFA", "IEF", "GSG", "VNQ"],
    "6-asset (+Gold)":    ["SPY", "EFA", "IEF", "GSG", "VNQ", "GLD"],
}

# Time windows: (label, start, end)
WINDOWS = [
    ("Full ETF (2007–2026)", "2007-05-30", "2025-12-31"),
    ("Out-of-sample (2006–2012)", "2006-01-01", "2012-12-31"),
    ("Full paper (1973–2012)", "1973-01-01", "2012-12-31"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_portfolio(
    prices_full: pd.DataFrame,
    asset_cols: list[str],
    cash_ticker: str,
    freq: str,
    sma: int,
    weights_cfg,
    bt_start: str,
    bt_end: str,
    cfg: dict,
) -> dict | None:
    """
    Run backtest for a given subset of assets and return metric row for both
    B&H and Timing, or None if not enough data.
    """
    # Keep only the required asset columns + cash proxy
    keep = [c for c in asset_cols if c in prices_full.columns] + [cash_ticker]
    prices = prices_full[keep].copy()

    # Filter to the window (end only; start trimmed after backtest)
    if bt_end:
        prices = prices[prices.index <= pd.Timestamp(bt_end)]
    if prices.empty:
        return None

    result = run_backtest(
        prices_df=prices,
        cash_col=cash_ticker,
        frequency=freq,
        sma_period=sma,
        weights_cfg=weights_cfg,
    )

    _trim_result_to_window(result, bt_start)

    if result.returns_bh.empty or len(result.returns_bh) < 12:
        return None

    rf = resolve_rf(cfg, result.cash_returns)

    bh = compute_all_metrics(
        returns=result.returns_bh, rf=rf, frequency=freq,
        signals_df=None, label="B&H",
    )
    timing = compute_all_metrics(
        returns=result.returns_timing, rf=rf, frequency=freq,
        signals_df=result.signals, label="Timing",
    )
    return {"bh": bh, "timing": timing}


def main():
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    cash_ticker = cfg["cash_proxy"]
    sma_periods: dict = cfg["strategy"]["sma_periods"]
    weights_cfg = cfg["strategy"]["weights"]
    frequencies: list = cfg["strategy"]["rebalance_frequencies"]

    RESULTS.mkdir(exist_ok=True)

    rows = []

    for win_label, win_start, win_end in WINDOWS:
        for freq in frequencies:
            sma = sma_periods.get(freq)
            if sma is None:
                continue

            # Load full price history up to window end (no start trim)
            prices_full = load_prices(freq, end=win_end)

            for port_label, asset_list in PORTFOLIOS.items():
                metrics = run_portfolio(
                    prices_full=prices_full,
                    asset_cols=asset_list,
                    cash_ticker=cash_ticker,
                    freq=freq,
                    sma=sma,
                    weights_cfg=weights_cfg,
                    bt_start=win_start,
                    bt_end=win_end,
                    cfg=cfg,
                )
                if metrics is None:
                    print(f"  [SKIP] {win_label} / {freq} / {port_label} — insufficient data")
                    continue

                for strat_key, m in [("B&H", metrics["bh"]), ("Timing", metrics["timing"])]:
                    rows.append({
                        "Window": win_label,
                        "Freq": freq,
                        "Portfolio": port_label,
                        "Strategy": strat_key,
                        "Start": m["start_date"],
                        "End": m["end_date"],
                        "Years": m["years"],
                        "CAGR%": m["cagr"],
                        "Vol%": m["volatility"],
                        "Sharpe": m["sharpe"],
                        "MaxDD%": m["max_drawdown"],
                        "Calmar": m["calmar"],
                        "%Invested": m.get("pct_time_invested", ""),
                    })

    df = pd.DataFrame(rows)

    # Print formatted table
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print()
    print("=" * 140)
    print("  Portfolio Comparison: 5-asset (Original Faber) vs 6-asset (+Gold)")
    print("=" * 140)
    print(df.to_string(index=False))
    print("=" * 140)
    print()

    # Save
    out = RESULTS / "portfolio_comparison.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
