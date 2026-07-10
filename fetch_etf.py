"""
fetch_etf.py — Sub-Task 2a
Download adjusted-close ETF price history from Yahoo Finance (yfinance)
for all tickers defined in config.yaml.

Usage:
    python3 fetch_etf.py           # download only if cache is absent
    python3 fetch_etf.py --refresh # force re-download even if cache exists

Outputs:
    data/raw/etf/<TICKER>.csv           — per-ticker adjusted close
    data/processed/prices_daily.csv     — all tickers merged, daily, forward-filled
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
RAW_DIR = ROOT / "data" / "raw" / "etf"
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def resolve_end_date(end_date_str: str) -> str:
    """Return today's date string when config says 'latest'."""
    if str(end_date_str).lower() == "latest":
        return date.today().isoformat()
    return str(end_date_str)


def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
    refresh: bool,
) -> pd.Series:
    """
    Download adjusted-close for a single ticker and cache to CSV.
    Returns a Series named after the ticker.
    """
    cache_path = raw_dir / f"{ticker}.csv"

    if cache_path.exists() and not refresh:
        log.info("  %s  — loading from cache (%s)", ticker, cache_path)
        series = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        series.name = ticker
        log.info("  %s  — cached range: %s → %s  (%d rows)",
                 ticker, series.index.min().date(), series.index.max().date(), len(series))
        return series

    log.info("  %s  — downloading from Yahoo Finance …", ticker)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        log.warning("  %s  — no data returned; skipping", ticker)
        return pd.Series(name=ticker, dtype=float)

    # yfinance may return a MultiIndex column when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    series = raw["Close"].copy()
    series.name = ticker

    # Persist raw adjusted close
    raw_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(cache_path, header=[ticker])

    log.info("  %s  — downloaded: %s → %s  (%d rows)",
             ticker, series.index.min().date(), series.index.max().date(), len(series))
    return series


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(refresh: bool = False) -> None:
    cfg = load_config()

    tickers: list[str] = list(cfg["assets"].keys()) + [cfg["cash_proxy"]]
    start: str = cfg["data"]["etf_start_date"]
    end: str = resolve_end_date(cfg["data"]["end_date"])

    log.info("=== ETF data fetch ===")
    log.info("Tickers  : %s", ", ".join(tickers))
    log.info("Range    : %s → %s", start, end)
    log.info("Refresh  : %s", refresh)
    log.info("")

    series_list: list[pd.Series] = []
    for ticker in tickers:
        s = fetch_ticker(ticker, start, end, RAW_DIR, refresh)
        if not s.empty:
            series_list.append(s)

    if not series_list:
        log.error("No data fetched. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Merge into a single daily DataFrame
    # ------------------------------------------------------------------
    daily = pd.concat(series_list, axis=1)
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"

    # Forward-fill across weekends and holidays (standard market convention).
    # NaN at the very start of a ticker's history remains NaN (not filled backwards).
    daily = daily.ffill()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "prices_daily.csv"
    daily.to_csv(out_path)

    # ------------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=== Coverage summary ===")
    log.info("%-8s  %-12s  %-12s  %8s  %s", "Ticker", "First date", "Last date", "Rows", "Name")
    log.info("%-8s  %-12s  %-12s  %8s  %s", "-"*8, "-"*12, "-"*12, "-"*8, "-"*30)
    asset_names: dict = cfg["assets"]
    asset_names[cfg["cash_proxy"]] = "Cash Proxy"
    for col in daily.columns:
        col_data = daily[col].dropna()
        if col_data.empty:
            log.info("%-8s  %-12s  %-12s  %8s  %s",
                     col, "N/A", "N/A", 0, asset_names.get(col, ""))
        else:
            log.info("%-8s  %-12s  %-12s  %8d  %s",
                     col,
                     col_data.index.min().date(),
                     col_data.index.max().date(),
                     len(col_data),
                     asset_names.get(col, ""))

    log.info("")
    log.info("Merged daily prices saved → %s  (%d rows × %d cols)",
             out_path, len(daily), len(daily.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ETF adjusted-close data from Yahoo Finance.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if cached CSV files exist.",
    )
    args = parser.parse_args()
    main(refresh=args.refresh)
