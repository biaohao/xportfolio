"""
fetch_longhistory.py — Sub-Task 2b (part 1)
Download and process long-history monthly data from free public sources:
  • Robert Shiller's online data  → S&P 500 total-return index (back to 1871)
  • FRED DGS10                   → approximate US 10-year bond monthly return (back to 1953)
  • FRED TB3MS                   → 3-month T-bill monthly return (back to 1934)

Usage:
    python3 fetch_longhistory.py           # download only if cache is absent
    python3 fetch_longhistory.py --refresh # force re-download

Outputs (all monthly, total-return indices normalised to 1.0 at first observation):
    data/raw/longhistory/sp500_shiller.csv   — S&P 500 total-return index
    data/raw/longhistory/bonds_fred.csv      — approximate 10yr bond total-return index
    data/raw/longhistory/tbill_fred.csv      — 3-month T-bill total-return index
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import yaml

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
RAW_DIR = ROOT / "data" / "raw" / "longhistory"

# Shiller data URL (publicly hosted on Yale Economics website)
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

# FRED series IDs
FRED_DGS10 = "DGS10"    # 10-year Treasury constant maturity yield (%, daily → resample monthly)
FRED_TB3MS = "TB3MS"    # 3-month T-bill secondary market rate (%, monthly)

# Modified duration approximation for IEF (7–10yr Treasury blend, ~8.5yr avg maturity).
# Empirically calibrated against IEF ETF returns 2002-2026:
#   optimal fixed D that minimises RMSE vs actual IEF monthly returns = 7.35yr
#   (the constant-8.0 value over-estimated duration by ~9%).
# Used to convert yield changes to price returns: P_ret ≈ -D * Δy + y/12
BOND_DURATION = 7.35


# ---------------------------------------------------------------------------
# S&P 500 Total Return — Robert Shiller
# ---------------------------------------------------------------------------

def fetch_shiller_sp500(cache_path: Path, refresh: bool) -> pd.Series:
    """
    Download Shiller's ie_data.xls and construct a monthly total-return index
    for the S&P 500 going back to January 1871.

    Total return formula (monthly reinvestment of dividends):
        TR[t] = TR[t-1] * (P[t] + D[t]/12) / P[t-1]

    where P = S&P price level, D = annual dividend (divided by 12 for monthly).
    """
    if cache_path.exists() and not refresh:
        log.info("  S&P 500 (Shiller) — loading from cache (%s)", cache_path)
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        s.name = "SP500_TR"
        log.info("  S&P 500 (Shiller) — cached range: %s → %s  (%d rows)",
                 s.index.min().date(), s.index.max().date(), len(s))
        return s

    log.info("  S&P 500 (Shiller) — downloading from %s …", SHILLER_URL)
    resp = requests.get(SHILLER_URL, timeout=60)
    resp.raise_for_status()

    # Shiller's file is old-format BIFF8 .xls — must use xlrd engine (not openpyxl).
    # Sheet "Data" layout (verified against live file 2024-2026):
    #   Row 7 (0-based): header row — Date | P | D | E | CPI | ...
    #   Row 8 onward   : monthly data starting 1871.01
    # Columns: A=Date (fractional year e.g.1871.01), B=Price, C=Dividend (annual)
    xls = pd.ExcelFile(io.BytesIO(resp.content), engine="xlrd")
    raw = pd.read_excel(
        xls,
        sheet_name="Data",
        header=7,      # row index 7 is the header; data starts at row 8
        usecols=[0, 1, 2],
    )

    # Rename columns regardless of what Shiller calls them in this edition
    raw.columns = ["date_frac", "price", "dividend"]

    # Drop rows where date or price is missing (footer notes etc.)
    raw = raw.dropna(subset=["date_frac", "price"])
    raw = raw[pd.to_numeric(raw["date_frac"], errors="coerce").notna()]
    raw["date_frac"] = raw["date_frac"].astype(float)
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["dividend"] = pd.to_numeric(raw["dividend"], errors="coerce").fillna(0.0)
    raw = raw.dropna(subset=["price"])

    # Convert fractional year to period-end date
    # e.g. 1871.01 → 1871-01-31, 1871.10 → 1871-10-31
    def frac_to_date(f: float) -> pd.Timestamp:
        year = int(f)
        month = round((f - year) * 100)
        if month == 0:
            month = 1
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

    raw["date"] = raw["date_frac"].apply(frac_to_date)
    raw = raw.set_index("date").sort_index()

    # Build total-return index (base = 1.0 at first observation)
    prices = raw["price"].values
    dividends = raw["dividend"].values   # annual dividend; divide by 12 for monthly

    n = len(prices)
    tr = np.ones(n)
    for i in range(1, n):
        if prices[i - 1] > 0:
            # Monthly reinvested return: price appreciation + dividend income
            tr[i] = tr[i - 1] * (prices[i] + dividends[i] / 12.0) / prices[i - 1]
        else:
            tr[i] = tr[i - 1]

    series = pd.Series(tr, index=raw.index, name="SP500_TR")

    # Persist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(cache_path, header=["SP500_TR"])
    log.info("  S&P 500 (Shiller) — saved: %s → %s  (%d rows)",
             series.index.min().date(), series.index.max().date(), len(series))
    return series


# ---------------------------------------------------------------------------
# US 10-Year Bond Return — FRED DGS10
# ---------------------------------------------------------------------------

def fetch_fred_bonds(cache_path: Path, start: str, refresh: bool) -> pd.Series:
    """
    Fetch 10-year Treasury yield (DGS10) from FRED and construct an approximate
    monthly total-return index using the modified-duration price approximation:

        R_monthly ≈ -Duration × Δyield/100 + yield_prev/1200

    where:
      -Duration × Δyield/100  is the price change from yield movement
      yield_prev/1200          is the coupon income (annualised yield / 12, in decimal)

    This is a first-order approximation sufficient for long-term backtesting.
    It will differ from an exact bond total-return index but captures the
    direction and magnitude of bond returns well.
    """
    if cache_path.exists() and not refresh:
        log.info("  US 10yr Bonds (FRED DGS10) — loading from cache (%s)", cache_path)
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        s.name = "BONDS_TR"
        log.info("  US 10yr Bonds (FRED DGS10) — cached range: %s → %s  (%d rows)",
                 s.index.min().date(), s.index.max().date(), len(s))
        return s

    log.info("  US 10yr Bonds (FRED DGS10) — downloading from FRED …")
    raw = web.DataReader(FRED_DGS10, "fred", start=start)
    raw.columns = ["yield_pct"]

    # Resample daily yield to month-end (take last available value in month)
    monthly_yield = raw["yield_pct"].resample("ME").last().dropna()

    # Approximate monthly bond return
    yield_prev = monthly_yield.shift(1)
    delta_yield = monthly_yield - yield_prev

    # Price return from duration + coupon income
    monthly_return = -BOND_DURATION * (delta_yield / 100.0) + yield_prev / 1200.0
    monthly_return = monthly_return.dropna()

    # Build cumulative total-return index (base = 1.0)
    tr = (1 + monthly_return).cumprod()
    tr = tr / tr.iloc[0]   # normalise to 1.0 at first observation
    tr.name = "BONDS_TR"
    # index is already month-end after resample("M") — no offset needed

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tr.to_csv(cache_path, header=["BONDS_TR"])
    log.info("  US 10yr Bonds (FRED DGS10) — saved: %s → %s  (%d rows)",
             tr.index.min().date(), tr.index.max().date(), len(tr))
    return tr


# ---------------------------------------------------------------------------
# 3-Month T-Bill Return — FRED TB3MS
# ---------------------------------------------------------------------------

def fetch_fred_tbills(cache_path: Path, start: str, refresh: bool) -> pd.Series:
    """
    Fetch 3-month T-bill rate (TB3MS) from FRED and convert to a monthly
    total-return index:

        R_monthly = (1 + rate/100)^(1/12) - 1

    where rate is the annualised T-bill rate in percent.
    """
    if cache_path.exists() and not refresh:
        log.info("  T-Bills (FRED TB3MS) — loading from cache (%s)", cache_path)
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
        s.name = "TBILL_TR"
        log.info("  T-Bills (FRED TB3MS) — cached range: %s → %s  (%d rows)",
                 s.index.min().date(), s.index.max().date(), len(s))
        return s

    log.info("  T-Bills (FRED TB3MS) — downloading from FRED …")
    raw = web.DataReader(FRED_TB3MS, "fred", start=start)
    raw.columns = ["rate_pct"]
    raw = raw.dropna()
    raw.index = raw.index + pd.offsets.MonthEnd(0)   # align to month-end

    # Monthly return from annualised rate
    monthly_return = (1 + raw["rate_pct"] / 100.0) ** (1.0 / 12.0) - 1.0

    # Cumulative total-return index (base = 1.0)
    tr = (1 + monthly_return).cumprod()
    tr = tr / tr.iloc[0]
    tr.name = "TBILL_TR"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tr.to_csv(cache_path, header=["TBILL_TR"])
    log.info("  T-Bills (FRED TB3MS) — saved: %s → %s  (%d rows)",
             tr.index.min().date(), tr.index.max().date(), len(tr))
    return tr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(refresh: bool = False) -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    start = cfg["data"]["longhistory_start_date"]

    log.info("=== Long-history data fetch ===")
    log.info("Start    : %s", start)
    log.info("Refresh  : %s", refresh)
    log.info("")

    sp500 = fetch_shiller_sp500(
        cache_path=RAW_DIR / "sp500_shiller.csv",
        refresh=refresh,
    )

    bonds = fetch_fred_bonds(
        cache_path=RAW_DIR / "bonds_fred.csv",
        start=start,
        refresh=refresh,
    )

    tbills = fetch_fred_tbills(
        cache_path=RAW_DIR / "tbill_fred.csv",
        start=start,
        refresh=refresh,
    )

    log.info("")
    log.info("=== Long-history coverage summary ===")
    for s in [sp500, bonds, tbills]:
        log.info("  %-12s  %s → %s  (%d monthly observations)",
                 s.name, s.index.min().date(), s.index.max().date(), len(s))

    log.info("")
    log.info("Note: MSCI EAFE, GSCI, and NAREIT have no reliable free long-history "
             "sources and are covered by ETF data only (Sub-Task 2a).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch long-history monthly data from Shiller and FRED."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if cached CSV files exist.",
    )
    args = parser.parse_args()
    main(refresh=args.refresh)
