"""
fetch_gold.py — Download gold price data from multiple free sources and produce
a clean long-history monthly series and a daily series.

Sources (in priority order):
1. GitHub datasets/gold-prices — monthly USD spot price 1833–present (free, no auth)
   URL: https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv
   Note: Pre-1968 was fixed at $35/oz (Bretton Woods); free-float starts Aug 1971.
   Paper uses 1973+ (after the first oil shock normalised the free market).

2. GLD ETF (yfinance) — daily adjusted close, 2004-11-18 to present.
   Best daily buy-and-hold proxy; no futures roll cost.

3. GC=F (COMEX gold futures, yfinance) — daily 2000-08 to present.
   Used to fill 2000–2004 gap where GLD doesn't exist.

Output files:
  data/raw/longhistory/gold_monthly_github.csv  — raw monthly (1833–present)
  data/raw/etf/GLD.csv                          — already fetched by fetch_etf.py (skip if exists)
  data/raw/etf/GCF.csv                          — GC=F daily (saved as GCF to avoid '=' in filename)
  data/processed/gold_daily.csv                 — splice: GC=F 2000-2004, GLD 2004-present
  data/processed/gold_monthly.csv               — from GitHub monthly series (1973–present)
"""

import os
import requests
import pandas as pd
import yfinance as yf
from io import StringIO

RAW_LH = "data/raw/longhistory"
RAW_ETF = "data/raw/etf"
PROC = "data/processed"


# ---------------------------------------------------------------------------
# 1. Monthly gold from GitHub (1833 → present)
# ---------------------------------------------------------------------------

def fetch_monthly_github(force=False):
    out = os.path.join(RAW_LH, "gold_monthly_github.csv")
    if os.path.exists(out) and not force:
        print(f"[gold] Monthly GitHub data already cached: {out}")
        return pd.read_csv(out, index_col=0, parse_dates=True)["Price"]

    url = "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv"
    print(f"[gold] Downloading monthly gold from GitHub...")
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index = df.index + pd.offsets.MonthEnd(0)  # align to month-end

    df.to_csv(out)
    print(f"[gold] Saved monthly data: {df.index[0].date()} → {df.index[-1].date()}, {len(df)} rows")
    return df["Price"]


# ---------------------------------------------------------------------------
# 2. GLD ETF daily (2004 → present)
# ---------------------------------------------------------------------------

def fetch_gld_daily(force=False):
    out = os.path.join(RAW_ETF, "GLD.csv")
    if os.path.exists(out) and not force:
        df = pd.read_csv(out, index_col=0, parse_dates=True)
        # Handle both single-column and multi-column saved formats
        col = df.columns[0] if len(df.columns) > 0 else None
        if col:
            s = df[col]
            print(f"[gold] GLD ETF already cached: {s.index[0].date()} → {s.index[-1].date()}, {len(s)} rows")
            return s.rename("GLD")
    # Re-download
    print("[gold] Downloading GLD ETF (yfinance)...")
    raw = yf.download("GLD", start="2004-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for GLD")
    close = raw["Close"].squeeze()
    close.name = "GLD"
    close.to_csv(out, header=True)
    print(f"[gold] Saved GLD: {close.index[0].date()} → {close.index[-1].date()}, {len(close)} rows")
    return close


# ---------------------------------------------------------------------------
# 3. GC=F (COMEX gold futures) daily (2000 → 2004 gap fill)
# ---------------------------------------------------------------------------

def fetch_gcf_daily(force=False):
    out = os.path.join(RAW_ETF, "GCF.csv")
    if os.path.exists(out) and not force:
        df = pd.read_csv(out, index_col=0, parse_dates=True)
        col = df.columns[0]
        s = df[col].rename("GCF")
        print(f"[gold] GC=F already cached: {s.index[0].date()} → {s.index[-1].date()}, {len(s)} rows")
        return s
    print("[gold] Downloading GC=F (COMEX gold futures, yfinance)...")
    raw = yf.download("GC=F", start="1999-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for GC=F")
    close = raw["Close"].squeeze()
    close.name = "GCF"
    close.to_csv(out, header=True)
    print(f"[gold] Saved GC=F: {close.index[0].date()} → {close.index[-1].date()}, {len(close)} rows")
    return close


# ---------------------------------------------------------------------------
# 4. Splice daily series: GC=F 2000-2004, GLD 2004-present
# ---------------------------------------------------------------------------

def build_daily_splice(gcf: pd.Series, gld: pd.Series) -> pd.Series:
    """
    Splice GC=F and GLD into a single daily gold price series.
    Method: use GLD from its first trading day onwards; backfill with GC=F,
    rescaled so both series match at the handoff date.
    """
    gld_start = gld.index[0]

    # Scale GC=F to match GLD at the handoff date
    gcf_at_handoff = gcf.asof(gld_start)
    gld_at_handoff = gld.iloc[0]
    scale = gld_at_handoff / gcf_at_handoff

    gcf_scaled = gcf * scale
    gcf_pre = gcf_scaled[gcf_scaled.index < gld_start]

    spliced = pd.concat([gcf_pre, gld])
    spliced.name = "Gold"
    spliced = spliced.sort_index().dropna()
    return spliced


# ---------------------------------------------------------------------------
# 5. Build monthly gold series from GitHub data (1973 → present)
# ---------------------------------------------------------------------------

def build_monthly_gold(monthly_raw: pd.Series) -> pd.Series:
    """
    Extract 1973-01 onward (post-Bretton-Woods free-float era).
    Return as month-end indexed price series suitable for splicing.
    """
    cutoff = pd.Timestamp("1973-01-31")
    s = monthly_raw[monthly_raw.index >= cutoff].copy()
    s.name = "Gold"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force=False):
    os.makedirs(RAW_LH, exist_ok=True)
    os.makedirs(RAW_ETF, exist_ok=True)
    os.makedirs(PROC, exist_ok=True)

    # 1. Monthly from GitHub
    monthly_raw = fetch_monthly_github(force=force)

    # 2. GLD daily
    gld = fetch_gld_daily(force=force)

    # 3. GC=F daily
    gcf = fetch_gcf_daily(force=force)

    # 4. Splice daily
    daily_gold = build_daily_splice(gcf, gld)
    daily_out = os.path.join(PROC, "gold_daily.csv")
    daily_gold.to_csv(daily_out, header=True)
    print(f"[gold] Daily splice saved: {daily_gold.index[0].date()} → {daily_gold.index[-1].date()}, {len(daily_gold)} rows")

    # 5. Monthly gold (1973+)
    monthly_gold = build_monthly_gold(monthly_raw)
    monthly_out = os.path.join(PROC, "gold_monthly.csv")
    monthly_gold.to_csv(monthly_out, header=True)
    print(f"[gold] Monthly (1973+) saved: {monthly_gold.index[0].date()} → {monthly_gold.index[-1].date()}, {len(monthly_gold)} rows")

    # Summary
    print("\n[gold] ── Coverage Summary ──")
    print(f"  Monthly (GitHub, 1973+) : {monthly_gold.index[0].strftime('%Y-%m')} → {monthly_gold.index[-1].strftime('%Y-%m')}")
    print(f"  Daily (GCF+GLD splice)  : {daily_gold.index[0].date()} → {daily_gold.index[-1].date()}")
    print(f"  GLD ETF only            : {gld.index[0].date()} → {gld.index[-1].date()}")

    return monthly_gold, daily_gold


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fetch gold price data from free sources")
    p.add_argument("--refresh", action="store_true", help="Force re-download even if cached")
    args = p.parse_args()
    main(force=args.refresh)
