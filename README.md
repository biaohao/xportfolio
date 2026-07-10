# Tactical Asset Allocation (TAA) — Faber GTAA Replication

A Python implementation of Mebane Faber's *"A Quantitative Approach to Tactical Asset
Allocation"* (SSRN-id962461, 2013 update), extended with daily/weekly frequency analysis
and a live BUY/SELL signal generator.

## Strategy in one sentence

Hold an asset class when its price is **above** its simple moving average (SMA); move to
cash (T-bills) when it is **below**.  Applied equally across 5 global asset classes, this
historically produced equity-like returns with bond-like risk.

## Project structure

```
.
├── config.yaml              # All parameters — edit this to change tickers/settings
├── requirements.txt
├── main.py                  # Single entry point (--fetch / --backtest / --signals / --all)
├── fetch_etf.py             # Sub-Task 2a: download ETF adjusted-close data (yfinance)
├── fetch_longhistory.py     # Sub-Task 2b: download Shiller + FRED long-history data
├── splice_data.py           # Sub-Task 2b: stitch long-history + ETF series
├── strategy.py              # Sub-Task 3:  SMA timing engine, multi-frequency
├── metrics.py               # Sub-Task 4:  CAGR, volatility, Sharpe, drawdown, etc.
├── report.py                # Sub-Task 4:  run all backtests, produce summary table
├── plot.py                  # Sub-Task 5:  equity curves, drawdowns, return bars, heatmaps
├── signals.py               # Sub-Task 6:  current BUY/SELL signal generator
├── data/
│   ├── raw/
│   │   ├── etf/             # Per-ticker CSVs from yfinance
│   │   └── longhistory/     # Per-source CSVs from Shiller / FRED
│   └── processed/
│       ├── prices_daily.csv          # All ETF tickers, daily, aligned
│       ├── prices_monthly_long.csv   # Long-history monthly series
│       ├── prices_monthly_spliced.csv# Long-history + ETF splice
│       └── data_sources.csv          # Coverage metadata
├── results/
│   ├── summary_table.csv    # Metrics for all strategies × frequencies
│   ├── current_signals.csv  # Latest BUY/SELL signals
│   └── validation_notes.md  # Comparison against paper's published numbers
└── plots/
    ├── equity_curve_monthly.png
    ├── equity_curve_weekly.png
    ├── equity_curve_daily.png
    ├── drawdown_*.png
    ├── yearly_returns_*.png
    └── asset_signals_*.png
```

## Quick start

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Fetch all data (ETF + long-history)
python3 main.py --fetch

# 3. Run backtests and generate all reports + plots
python3 main.py --backtest

# 4. View current BUY/SELL signals
python3 main.py --signals

# 5. Run full pipeline at once
python3 main.py --all
```

## Configuration

All parameters live in [`config.yaml`](config.yaml).

| Section | Key | Description |
|---|---|---|
| `assets` | ticker → name | Asset universe. Add/remove tickers here freely. |
| `cash_proxy` | ticker | ETF used as T-bill cash return (default: `BIL`). |
| `data.etf_start_date` | date string | Earliest date to request from yfinance. |
| `strategy.sma_periods` | monthly/weekly/daily | SMA look-back per frequency. |
| `strategy.rebalance_frequencies` | list | Which frequencies to run. |
| `reporting.risk_free_rate` | `"dynamic"` or float | Sharpe ratio risk-free rate. |

## Data sources

| Asset | ETF (yfinance) | ETF Start | Long-History Source | Starts |
|---|---|---|---|---|
| US Large Cap | SPY | Jan 1993 | Shiller `ie_data.xls` (total return) | 1871 |
| Foreign Developed | EFA | Aug 2001 | None — ETF only | 2001 |
| US 10-Year Bonds | IEF | Jul 2002 | FRED `DGS10` yield → approx. return | 1953 |
| Commodities | GSG | Jul 2006 | None — ETF only | 2006 |
| US REITs | VNQ | Sep 2004 | None — ETF only | 2004 |
| Cash Proxy | BIL | May 2007 | FRED `TB3MS` rate → monthly return | 1934 |

**Full 5-asset common start (ETF era):** ~July 2007
**Monthly spliced backtest:** S&P 500 + bonds + cash back to ~1953; EAFE/GSCI/NAREIT ETF-only

## Paper benchmarks (Faber 2013, 1973–2012)

| Metric | Buy & Hold (5-asset) | GTAA Timing |
|---|---|---|
| CAGR | ~9.9% | ~10.5% |
| Annualized Vol | ~12% | ~7% |
| Sharpe Ratio | ~0.5 | ~0.8 |
| Max Drawdown | ~−46% | ~−10% |
| % Time Invested | 100% | ~70% |

Our replication uses ETF data from 2007 and spliced long-history data for S&P/bonds.
Minor discrepancies from the paper are expected (different data source, shorter history).
See [`results/validation_notes.md`](results/validation_notes.md) after running `--backtest`.

## Frequency → SMA period mapping

| Frequency | Resampling | SMA Period | Rationale |
|---|---|---|---|
| Monthly | Month-end close | 10 | Directly from paper |
| Weekly | Friday close | 43 | 10 months × 4.3 weeks |
| Daily | Daily close | 200 | Standard 200-day MA |

## Extending to more asset classes

Add any ticker to the `assets` dict in `config.yaml`, then re-run `--fetch` and
`--backtest`. The strategy engine normalizes weights automatically.  The paper's full
13-asset GTAA Moderate/Aggressive allocation is ready to implement — simply add the
additional tickers (e.g., `IJR`, `EEM`, `TIP`, `BWX`, `IAU`) to `config.yaml`.
