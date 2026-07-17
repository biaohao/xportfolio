"""
splice_data.py — Sub-Task 2b (part 2)
Stitch long-history monthly series (Shiller / FRED) with ETF adjusted-close data
so the monthly backtest can use the longest possible history.

Rules:
  • ETF data takes precedence from its launch date onward (more reliable, dividend-adjusted).
  • Long-history data is used for the period before the ETF launch.
  • At the splice point the two series are normalised so they meet at 1.0,
    ensuring a smooth, seamless join.

Assets with no long-history source (MSCI EAFE / GSCI / NAREIT) are included
from their ETF launch date only; earlier rows are NaN.

Outputs:
    data/processed/prices_monthly_long.csv    — long-history raw series only (monthly)
    data/processed/prices_monthly_spliced.csv — spliced series, monthly, aligned
    data/processed/data_sources.csv           — coverage metadata per asset
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config import load_config, ROOT

RAW_ETF = ROOT / "data" / "raw" / "etf"
RAW_LONG = ROOT / "data" / "raw" / "longhistory"
PROCESSED = ROOT / "data" / "processed"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_etf_monthly(ticker: str) -> pd.Series:
    """Load ETF daily prices, resample to month-end, return price series."""
    path = RAW_ETF / f"{ticker}.csv"
    if not path.exists():
        log.warning("  ETF cache missing for %s — run fetch_etf.py first", ticker)
        return pd.Series(name=ticker, dtype=float)
    raw = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    raw.name = ticker
    # Resample to month-end (last trading day of each month)
    monthly = raw.resample("ME").last()
    return monthly


def load_long_history(col: str, path: Path) -> pd.Series:
    """Load a long-history total-return index CSV."""
    if not path.exists():
        log.warning("  Long-history file missing: %s — run fetch_longhistory.py first", path)
        return pd.Series(dtype=float)
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.name = col
    return s


def splice(long_series: pd.Series, etf_series: pd.Series) -> pd.Series:
    """
    Splice long_series (pre-ETF) with etf_series (post-ETF).

    Both series are assumed to be total-return indices (not returns).

    Scaling rule: anchor the long-history level at the ETF's splice month
    (long[M] * scale == etf[M]), then keep only the long-history rows
    *before* month M so that the ETF's first month return is preserved.

    Previous bug: anchoring at M-1 (long[M-1] * scale == etf[M]) forced
    the pct_change across the boundary to 0%, erasing the real splice-month
    return for every spliced asset.
    """
    if etf_series.empty:
        return long_series

    etf_start = etf_series.first_valid_index()

    # Long-history rows strictly before the ETF launch month
    pre_etf = long_series[long_series.index < etf_start]

    if pre_etf.empty:
        # No pre-ETF data; use ETF series as-is (converted to a growth index)
        etf_idx = etf_series / etf_series.iloc[0]
        return etf_idx

    # Anchor: scale so long_series[etf_start] == etf_series[etf_start].
    # If the long-history has a value at the splice month, use it; otherwise
    # fall back to the last pre-ETF value (same result, just a safety net).
    if etf_start in long_series.index:
        long_at_splice = long_series.loc[etf_start]
    else:
        long_at_splice = pre_etf.iloc[-1]

    etf_at_splice = etf_series.iloc[0]

    if long_at_splice == 0 or pd.isna(long_at_splice):
        scale = 1.0
    else:
        scale = etf_at_splice / long_at_splice

    # Apply scale only to the strictly pre-ETF rows (ETF takes over from etf_start)
    pre_etf_scaled = pre_etf * scale

    # Concatenate: pre-ETF long-history (scaled) + ETF from launch onward
    combined = pd.concat([pre_etf_scaled, etf_series])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.name = etf_series.name
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_long_history_map(cfg: dict) -> dict[str, tuple[str, Path]]:
    """
    Build the long-history map from config.yaml at runtime.

    Returns {ticker: (column_name, Path)} for every entry under
    long_history_sources.  Paths are resolved relative to ROOT so they
    work regardless of the working directory.
    """
    sources = cfg.get("long_history_sources", {})
    result: dict[str, tuple[str, Path]] = {}
    for ticker, entry in sources.items():
        result[ticker] = (entry["column"], ROOT / entry["file"])
    return result


def main() -> None:
    cfg = load_config()

    all_tickers = list(cfg["assets"].keys())
    cash_ticker = cfg["cash_proxy"]

    # Build the long-history map from config — no hardcoded paths in code.
    long_history_map = _build_long_history_map(cfg)

    log.info("=== Splice long-history + ETF data ===")

    # ------------------------------------------------------------------
    # 1.  Build long-history monthly DataFrame (raw, no splicing yet)
    # ------------------------------------------------------------------
    long_frames: dict[str, pd.Series] = {}
    for ticker, (col, path) in long_history_map.items():
        s = load_long_history(col, path)
        if not s.empty:
            s.name = ticker
            long_frames[ticker] = s
            log.info("  Long-history loaded: %-6s  %s → %s  (%d rows)",
                     ticker, s.index.min().date(), s.index.max().date(), len(s))

    if long_frames:
        prices_monthly_long = pd.DataFrame(long_frames)
        prices_monthly_long.index.name = "Date"
        out = PROCESSED / "prices_monthly_long.csv"
        prices_monthly_long.to_csv(out)
        log.info("  Saved long-history monthly → %s", out)
    else:
        log.warning("  No long-history series found. Run fetch_longhistory.py first.")

    # ------------------------------------------------------------------
    # 2.  Build spliced series for every asset (incl. cash proxy)
    # ------------------------------------------------------------------
    spliced: dict[str, pd.Series] = {}
    source_records: list[dict] = []

    for ticker in all_tickers + [cash_ticker]:
        etf_monthly = load_etf_monthly(ticker)
        long_s = long_frames.get(ticker, pd.Series(dtype=float))

        if long_s.empty:
            # No long-history: use ETF only, convert to growth index
            if etf_monthly.empty:
                log.warning("  %s — no data at all; skipping", ticker)
                continue
            growth = etf_monthly / etf_monthly.iloc[0]
            growth.name = ticker
            spliced[ticker] = growth
            source_records.append({
                "ticker": ticker,
                "source": "ETF only",
                "etf_start": etf_monthly.first_valid_index().date() if not etf_monthly.empty else None,
                "long_history_start": None,
                "splice_date": None,
            })
            log.info("  %-6s  ETF only  from %s", ticker,
                     etf_monthly.first_valid_index().date())
        else:
            merged = splice(long_s, etf_monthly)
            spliced[ticker] = merged
            etf_start = etf_monthly.first_valid_index()
            source_records.append({
                "ticker": ticker,
                "source": "Long-history + ETF splice",
                "etf_start": etf_start.date() if etf_start else None,
                "long_history_start": long_s.first_valid_index().date() if not long_s.empty else None,
                "splice_date": etf_start.date() if etf_start else None,
            })
            log.info("  %-6s  Spliced: long-history from %s, ETF from %s",
                     ticker,
                     long_s.first_valid_index().date(),
                     etf_start.date() if etf_start else "N/A")

    # ------------------------------------------------------------------
    # 3.  Save merged spliced DataFrame
    # ------------------------------------------------------------------
    spliced_df = pd.DataFrame(spliced)
    spliced_df.index.name = "Date"
    out_splice = PROCESSED / "prices_monthly_spliced.csv"
    spliced_df.to_csv(out_splice)
    log.info("")
    log.info("Spliced monthly prices saved → %s  (%d rows × %d cols)",
             out_splice, len(spliced_df), len(spliced_df.columns))

    # ------------------------------------------------------------------
    # 4.  Save data source metadata
    # ------------------------------------------------------------------
    sources_df = pd.DataFrame(source_records)
    out_sources = PROCESSED / "data_sources.csv"
    sources_df.to_csv(out_sources, index=False)
    log.info("Data source metadata saved → %s", out_sources)

    # ------------------------------------------------------------------
    # 5.  Print coverage summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=== Spliced coverage summary ===")
    log.info("%-8s  %-12s  %-12s  %8s  %s",
             "Ticker", "First date", "Last date", "Rows", "Source")
    log.info("%-8s  %-12s  %-12s  %8s  %s",
             "-"*8, "-"*12, "-"*12, "-"*8, "-"*35)
    asset_names = dict(cfg["assets"])
    asset_names[cash_ticker] = "Cash Proxy"
    for ticker, s in spliced.items():
        valid = s.dropna()
        rec = next((r for r in source_records if r["ticker"] == ticker), {})
        log.info("%-8s  %-12s  %-12s  %8d  %s",
                 ticker,
                 valid.index.min().date() if not valid.empty else "N/A",
                 valid.index.max().date() if not valid.empty else "N/A",
                 len(valid),
                 rec.get("source", ""))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
