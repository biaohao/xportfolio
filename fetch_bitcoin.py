"""
fetch_bitcoin.py — Download Bitcoin price history from CoinMetrics (GitHub) and
cache it as a monthly total-return index suitable for splicing with the BTC-USD
ETF (Yahoo Finance) in splice_data.py.

Sources:
  Long-history : CoinMetrics open data (GitHub) — daily USD price from 2010-07-18
                 URL: https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv
  ETF-equivalent: BTC-USD on Yahoo Finance — daily from 2014-09-17 (handled by fetch_etf.py)

The two series agree to within ~0.7% at month-end in the overlap period (2014-09 → 2026-05),
so the splice is clean.

Usage:
    python3 fetch_bitcoin.py            # download only if cache is absent
    python3 fetch_bitcoin.py --refresh  # force re-download

Output:
    data/raw/longhistory/btc_coinmetrics.csv  — monthly total-return index (BTC_TR), base 1.0
"""

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
from config import ROOT

RAW_DIR = ROOT / "data" / "raw" / "longhistory"
CACHE_PATH = RAW_DIR / "btc_coinmetrics.csv"

COINMETRICS_URL = (
    "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
)

# Bitcoin first traded on exchanges 2010-07-17 (Mt.Gox).  Rows before this date
# are 0 or NaN and are dropped.
BTC_MARKET_START = pd.Timestamp("2010-07-17")


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_btc_coinmetrics(cache_path: Path, refresh: bool) -> pd.Series:
    """
    Download CoinMetrics BTC daily price data and convert to a monthly
    total-return index normalised to 1.0 at the first observation.

    Bitcoin has no dividend income, so:
        R_monthly = P[t] / P[t-1] - 1

    The resulting series is stored as BTC_TR in the cache CSV.
    """
    if cache_path.exists() and not refresh:
        log.info("  BTC (CoinMetrics) — loading from cache (%s)", cache_path)
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        s.name = "BTC_TR"
        log.info("  BTC (CoinMetrics) — cached range: %s → %s  (%d rows)",
                 s.index.min().date(), s.index.max().date(), len(s))
        return s

    log.info("  BTC (CoinMetrics) — downloading from %s …", COINMETRICS_URL)
    resp = requests.get(COINMETRICS_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), parse_dates=["time"])
    df = df.set_index("time").sort_index()

    # Use PriceUSD column; drop rows with missing / zero price
    price_daily = df["PriceUSD"].dropna()
    price_daily = price_daily[price_daily > 0]
    price_daily = price_daily[price_daily.index >= BTC_MARKET_START]

    # Resample to month-end (last available price of each calendar month)
    price_monthly = price_daily.resample("ME").last().dropna()

    # Build total-return index from monthly price relatives
    # (Bitcoin pays no dividends, so price change = total return)
    monthly_ret = price_monthly.pct_change().dropna()
    tr = (1 + monthly_ret).cumprod()
    tr = tr / tr.iloc[0]   # normalise to 1.0 at first observation
    tr.name = "BTC_TR"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tr.to_csv(cache_path, header=["BTC_TR"])
    log.info("  BTC (CoinMetrics) — saved: %s → %s  (%d rows)",
             tr.index.min().date(), tr.index.max().date(), len(tr))
    return tr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(refresh: bool = False) -> None:
    log.info("=== Bitcoin long-history fetch ===")
    log.info("Source   : CoinMetrics open data (GitHub)")
    log.info("Refresh  : %s", refresh)
    log.info("")

    tr = fetch_btc_coinmetrics(CACHE_PATH, refresh)

    log.info("")
    log.info("=== Coverage summary ===")
    log.info("  %-10s  %s → %s  (%d monthly observations)",
             tr.name, tr.index.min().date(), tr.index.max().date(), len(tr))
    log.info("")
    log.info("Next steps:")
    log.info("  1.  python3 fetch_etf.py --refresh   # download BTC-USD from Yahoo Finance")
    log.info("  2.  python3 splice_data.py           # rebuild spliced monthly series")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Fetch Bitcoin long-history price data from CoinMetrics (GitHub)."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if cached CSV exists.",
    )
    args = parser.parse_args()
    main(refresh=args.refresh)
