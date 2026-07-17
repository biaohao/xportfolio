# TAA Replication & Extension Plan

## Top-Level Overview

**Goal:** Replicate Mebane Faber's "A Quantitative Approach to Tactical Asset Allocation" (SSRN-id962461) as a Python project, then extend it with daily/weekly frequency analysis and live BUY/SELL signal generation.

**Scope:**
1. **Replication** — Fetch historical data for the paper's 5 core asset classes as far back as possible, implement the 10-month SMA timing rule, and reproduce the paper's key statistics (CAGR, volatility, Sharpe ratio, max drawdown) for both Buy & Hold and the Timing model.
2. **Multi-frequency extension** — Run the same strategy logic at monthly, weekly, and daily frequencies and compare results side-by-side.
3. **Live signals** — Produce a current BUY/SELL signal report for each configured asset class.
4. **Configurability** — All asset tickers and strategy parameters are driven by a single config file so the portfolio can be extended at any time without touching the core logic.

**Non-goals (v1):**
- The 13-asset GTAA Moderate/Aggressive extensions (addressed in config design but not backtested)
- Leverage overlays
- Cash management alternatives (T-bills vs. 10yr bonds)
- Live trading or broker integration

**Paper's core rules (to replicate):**
- 5 asset classes: US Large Cap (SPY), Foreign Developed (EFA), US 10-Year Bonds (IEF), Commodities (DJP or GSG), US REITs (VNQ)
- **Buy rule:** End-of-period price > 10-period SMA → hold asset (20% weight)
- **Sell rule:** End-of-period price < 10-period SMA → move to cash (T-bill proxy: BIL/SHV)
- Equal-weight, rebalance every period
- Total-return series (adjusted prices including dividends)

**Tech stack:**
- Language: Python 3.x, plain scripts (no notebooks)
- Data: `yfinance` (ETF history), `pandas-datareader` (FRED), direct HTTP download (Shiller)
- Analysis: `pandas`, `numpy`
- Metrics: `quantstats`
- Plotting: `matplotlib`
- Config: YAML file (`config.yaml`)

---

## Sub-Task 1: Project Scaffold & Configuration

**Intent:** Establish the project skeleton — directory structure, dependency file, and a `config.yaml` that drives all downstream scripts. Getting this right first ensures every later sub-task follows the same conventions.

**Expected Outcomes:**
- `requirements.txt` listing all locked dependencies
- `config.yaml` with the 5 core tickers, strategy parameters, and date range
- Directory structure: `data/`, `results/`, `plots/`
- `README.md` explaining how to run the project

**Todo List:**
1. Create top-level directory structure: `data/`, `results/`, `plots/`
2. Write `config.yaml` with:
   - `assets` dict: ticker → human-readable name (e.g., `SPY: "US Large Cap"`)
   - `cash_proxy` ticker (e.g., `BIL`)
   - `strategy.sma_periods` list (e.g., `[10]` for monthly; derived for weekly/daily)
   - `strategy.rebalance_frequencies` list: `["monthly", "weekly", "daily"]`
   - `data.etf_start_date` (e.g., `"1973-01-01"` — pulls as far back as each ETF allows)
   - `data.end_date`: `"latest"`
   - `data.backtest_start_date`: controls the start of the backtest window (`"latest"` = full history)
   - `data.backtest_end_date`: controls the end of the backtest window (`"latest"` = today)
3. Write `requirements.txt` with pinned latest-stable versions of: `yfinance`, `pandas`, `numpy`, `quantstats`, `matplotlib`, `pyyaml`
4. Write `README.md` with install and run instructions

**Relevant Context:**
- Paper uses Global Financial Data (paid); we use yfinance adjusted close as free equivalent
- SPY launches 1993, EFA 2001, VNQ 2004, GSG 2006 — earliest common history is ~2006; for longer history see Sub-Task 2b (long-history data)
- Config must allow adding new tickers without code changes
- `requirements.txt` must also include `pandas-datareader` and `requests` (for Shiller download)

**Status:** [x] complete

---

## Sub-Task 2a: ETF Data Fetching Module

**Intent:** Download and persist adjusted-close price history for all configured tickers using `yfinance`. This is the primary data source for the multi-frequency backtest (daily/weekly/monthly) and for live signal generation. Saves data locally so subsequent runs don't re-download.

**Expected Outcomes:**
- `fetch_etf.py` script that reads `config.yaml` and downloads ETF data
- Raw adjusted-close CSV files saved to `data/raw/etf/<TICKER>.csv`
- A merged `data/processed/prices_daily.csv` with all ETF tickers aligned on trading days
- Graceful handling of different ticker start dates (forward-fill NaN strategy)
- Script logs the actual date range and observation count per ticker

**Todo List:**
1. Write `fetch_etf.py`:
   - Load `config.yaml` to get ticker list and date range
   - For each ticker: download adjusted close via `yfinance.download(auto_adjust=True)`, save to `data/raw/etf/<TICKER>.csv`
   - Merge all tickers into a single daily DataFrame indexed by date
   - Forward-fill weekends/holidays (standard market convention)
   - Save merged daily prices to `data/processed/prices_daily.csv`
2. Add a `--refresh` flag to force re-download even if cached data exists
3. Log the actual start/end date and count of observations per ticker

**Relevant Context:**
- Use `auto_adjust=True` to get dividends baked into adjusted close (critical — paper uses total return series)
- Cash proxy: `BIL` (SPDR Bloomberg 1-3 Month T-Bill ETF, launched May 2007) or `SHV`
- ETF effective common start date for all 5 assets + cash proxy: ~July 2007
- This module is the **only** data source for daily and weekly frequency backtests (long-history data in Sub-Task 2b is monthly only)

**Status:** [x] complete

---

## Sub-Task 2b: Long-History Data Fetching Module

**Intent:** Extend the backtest as far back as possible toward the paper's 1973 start date by fetching free long-history monthly data from FRED (Federal Reserve) and Robert Shiller's public dataset. This enables a closer replication of the paper's 40-year results. The output is a separate monthly price series that is spliced with ETF data where available.

**Expected Outcomes:**
- `fetch_longhistory.py` script that downloads and processes long-history monthly data
- Separate CSV files per asset saved to `data/raw/longhistory/<ASSET>.csv`
- A merged `data/processed/prices_monthly_long.csv` covering the longest available history per asset
- A `data/processed/prices_monthly_spliced.csv` that stitches long-history series with ETF data (ETF takes precedence from its launch date)
- A `data/processed/data_sources.csv` documenting which source covers which date range for each asset

**Data Sources and Coverage:**

| Asset | Free Source | Series / Method | Approx. Start |
|---|---|---|---|
| US Large Cap (S&P 500) | Robert Shiller (irrationalexuberance.com) | `ie_data.xls`: price + dividends → total return | 1871 (monthly) |
| US Large Cap (S&P 500) | FRED via `pandas-datareader` | `SP500` price index (no dividends; use Shiller for total return) | 1928 |
| US 10-Year Bonds | FRED | `DGS10` yield → construct price return; or use `BAMLCC0A0CMTRIV` as proxy | 1962 (daily) |
| Cash / T-Bills | FRED | `TB3MS` (3-month T-bill rate, monthly) | 1934 |
| Foreign Developed (MSCI EAFE) | No reliable free long-history source | Use EFA ETF only; mark pre-2001 as unavailable | 2001 (ETF only) |
| Commodities (GSCI) | No reliable free long-history source | Use GSG ETF only; mark pre-2006 as unavailable | 2006 (ETF only) |
| US REITs (NAREIT) | No reliable free long-history source | Use VNQ ETF only; mark pre-2004 as unavailable | 2004 (ETF only) |

**Todo List:**
1. Write `fetch_longhistory.py`:
   - **Shiller S&P 500 total return:**
     - Download `ie_data.xls` from `http://www.econ.yale.edu/~shiller/data/ie_data.xls` via `requests` + `openpyxl`
     - Extract monthly columns: `Date`, `P` (price), `D` (dividend), `CPI`
     - Construct monthly total return index: `TR[t] = TR[t-1] × (P[t] + D[t]/12) / P[t-1]`
     - Save to `data/raw/longhistory/sp500_shiller.csv`
   - **FRED bond yield → return:**
     - Fetch `DGS10` (10-year Treasury constant maturity yield) via `pandas-datareader.DataReader('DGS10', 'fred', ...)`
     - Resample to month-end
     - Approximate monthly bond return from yield change: `R ≈ -Duration × Δyield + yield/12` (modified duration ~8 for 10yr)
     - Save to `data/raw/longhistory/bonds_fred.csv`
   - **FRED T-bill cash proxy:**
     - Fetch `TB3MS` (3-month T-bill secondary market rate, monthly)
     - Convert annualized rate to monthly return: `R_monthly = (1 + rate/100)^(1/12) - 1`
     - Save to `data/raw/longhistory/tbill_fred.csv`
2. Write `splice_data.py`:
   - For each asset with both long-history and ETF series: align on date index, use long-history pre-ETF-launch and ETF post-launch
   - Normalize both series to the same base (set index = 1.0 at ETF launch date to ensure smooth splice)
   - Save merged series to `data/processed/prices_monthly_spliced.csv`
   - Save source-tracking metadata to `data/processed/data_sources.csv`
3. Log coverage summary: per asset, show which source covers which date range

**Relevant Context:**
- Shiller's `ie_data.xls` file structure: sheet "Data", row 8 onward, columns A (fractional year date), B (S&P price), C (dividend), E (earnings), G (CPI). Verify sheet layout on download as it is occasionally updated.
- `pandas-datareader` FRED access is free and requires no API key for most series; install with `pip install pandas-datareader`
- Bond return construction from yields is an approximation; for the paper's replication purpose this is sufficient. Document the approximation formula in a comment.
- MSCI EAFE, GSCI, and NAREIT do not have reliable free long-history sources; the plan intentionally scopes these to ETF data only and documents this limitation clearly
- The `prices_monthly_spliced.csv` file is the input to the monthly-frequency backtest; `prices_daily.csv` (from Sub-Task 2a) is the input for daily and weekly backtests
- `pandas-datareader` must be added to `requirements.txt`; `openpyxl` must be added for reading the Shiller Excel file

**Status:** [x] complete

---

## Sub-Task 3: Strategy Engine (Core Backtesting Logic)

**Intent:** Implement the paper's timing model as a reusable engine that works across monthly, weekly, and daily frequencies. The engine produces per-period returns for both Buy & Hold and the Timing model.

**Expected Outcomes:**
- `strategy.py` module with a `run_backtest(prices_df, frequency, sma_period, weights)` function
- Returns a DataFrame of portfolio-level period returns for B&H and Timing
- Intermediate output: per-asset signals DataFrame (1=invested, 0=cash) at each rebalance date
- Works correctly for all three frequencies without hardcoded assumptions

**Todo List:**
1. Write `strategy.py`:
   - `resample_prices(daily_df, frequency)` → resamples daily data to month-end / week-end / daily
   - `compute_sma(price_series, period)` → rolling SMA on resampled data
   - `generate_signals(prices, sma_period)` → for each asset: 1 if price > SMA else 0 (signal is evaluated at period close, applied next period to avoid look-ahead bias)
   - `compute_period_returns(prices_resampled)` → pct_change per period per asset
   - `apply_timing_model(returns, signals, cash_returns, weights)` → timing portfolio return per period
   - `apply_buy_hold(returns, weights)` → equal-weight B&H portfolio return per period
2. Ensure no look-ahead bias: signal generated at end of period T applies to return in period T+1
3. Handle NaN correctly when tickers have different start dates (only invest in available assets, renormalize weights)
4. Write a `run_backtest()` orchestrator that calls steps above in order

**Relevant Context:**
- Paper's exact rule: "Buy when monthly price > 10-month SMA; Sell when monthly price < 10-month SMA"
- SMA period translation across frequencies: 10-month ≈ 43-week ≈ 200-day (standard convention)
- These equivalent periods should be configurable in `config.yaml` (not hardcoded)
- Cash return proxy: use BIL/SHV period return when signal=0

**Status:** [x] complete

---

## Sub-Task 4: Performance Metrics & Reporting

**Intent:** Compute the same statistics the paper reports (CAGR, annualized volatility, Sharpe ratio, max drawdown, percent time invested) and produce a clean comparison table for all three frequencies side-by-side.

**Expected Outcomes:**
- `metrics.py` module with all metric functions
- `report.py` script that orchestrates running all backtests and printing/saving comparison tables
- A `results/summary_table.csv` with all metrics per strategy × frequency
- Human-readable console output showing the comparison

**Todo List:**
1. Write `metrics.py` with:
   - `cagr(returns_series)` → annualized compound growth rate
   - `annualized_volatility(returns_series, freq)` → annualized std dev (correctly annualizes by frequency: ×√12, ×√52, ×√252)
   - `sharpe_ratio(returns_series, risk_free_rate, freq)` → annualized Sharpe
   - `max_drawdown(returns_series)` → peak-to-trough decline
   - `calmar_ratio(returns_series)` → CAGR / |max drawdown|
   - `pct_time_invested(signals_df)` → average fraction of periods with signal=1 across assets
2. Write `report.py`:
   - For each frequency in `[monthly, weekly, daily]`:
     - Run `run_backtest()` from `strategy.py`
     - Compute all metrics for both B&H and Timing
   - Build a Pandas DataFrame summary table
   - Print formatted table to console
   - Save to `results/summary_table.csv`
3. Optionally use `quantstats.reports.metrics()` for validation/cross-check

**Relevant Context:**
- Paper reports for GTAA (5-asset, 1973-2012): Timing CAGR ~10.5%, Vol ~7%, Sharpe ~0.8, Max DD ~-10%; B&H CAGR ~9.9%, Vol ~12%, Sharpe ~0.5, Max DD ~-46%
- These are benchmarks to validate the replication is correct
- Annualization factor must match the frequency of the input returns series

**Status:** [x] complete

---

## Sub-Task 5: Visualization

**Intent:** Produce the key charts from the paper — equity curve, drawdown chart, and yearly return bar chart — for all three frequencies, saved as PNG files.

**Expected Outcomes:**
- `plot.py` script producing the following plots saved to `plots/`:
  - `equity_curve_<freq>.png`: log-scale cumulative return for B&H vs Timing
  - `drawdown_<freq>.png`: drawdown over time for B&H vs Timing
  - `yearly_returns_<freq>.png`: bar chart of annual returns B&H vs Timing
  - `asset_signals_<freq>.png`: heatmap showing per-asset signal (invested vs. cash) over time

**Todo List:**
1. Write `plot.py` with helper functions:
   - `plot_equity_curves(returns_bh, returns_timing, freq, output_path)` → log-scale line chart
   - `plot_drawdowns(returns_bh, returns_timing, freq, output_path)` → area chart of drawdown
   - `plot_annual_returns(returns_bh, returns_timing, freq, output_path)` → grouped bar chart
   - `plot_signal_heatmap(signals_df, freq, output_path)` → green/red heatmap per asset per period
2. Use `matplotlib` with clean, publication-style formatting
3. Call `plot.py` from `report.py` or as a standalone script

**Relevant Context:**
- Figures 7, 8, 13, 14 in the paper are the target outputs to match visually
- Log scale equity curve is critical for long-horizon data (as used in the paper)

**Status:** [x] complete

---

## Sub-Task 6: Live Signal Generator

**Intent:** Produce a current-state BUY/SELL signal report for each configured asset class using the most recent available data. This is the forward-looking tool the user can run periodically to inform portfolio decisions.

**Expected Outcomes:**
- `signals.py` script that fetches the latest prices and outputs the current signal for each asset
- Console output and `results/current_signals.csv` showing per-asset: current price, SMA value, signal (BUY/HOLD/SELL), and % above/below SMA
- Works for all three configured frequencies

**Todo List:**
1. Write `signals.py`:
   - Load `config.yaml` to get tickers and SMA parameters
   - Fetch the last N periods of data (enough to compute the SMA) via `yfinance`
   - For each asset and each frequency: compute SMA, compare to latest price, emit signal
   - Output a clean table: `Asset | Freq | Price | SMA | Signal | % vs SMA | Last Updated`
   - Save to `results/current_signals.csv`
2. Add a `--frequency` CLI argument to filter output to one frequency
3. Mark signals that changed since the previous run (compare to a saved `results/last_signals.csv`)

**Relevant Context:**
- This script should be runnable in isolation (no need to run full backtest first)
- The most actionable output for the user's own portfolio monitoring
- For monthly: signal is only meaningful at month-end; add a note showing days until next rebalance date

**Status:** [x] complete

---

## Sub-Task 7: Integration & Validation

**Intent:** Wire all modules together into a single entry point (`main.py`), validate the replication against the paper's published numbers, and document any discrepancies (expected due to data source differences).

**Expected Outcomes:**
- `main.py` that runs the full pipeline: fetch → backtest → report → plots → signals
- A `results/validation_notes.md` documenting how close the replication is to the paper's numbers and why any differences exist
- All scripts runnable from the command line with sensible defaults

**Todo List:**
1. Write `main.py` with `argparse`:
   - `python main.py --fetch` → run data download
   - `python main.py --backtest` → run all backtests and generate reports/plots
   - `python main.py --signals` → run signal generator only
   - `python main.py --all` → run full pipeline
2. Write `results/validation_notes.md` comparing output metrics against the paper's Table in Figure 13/18
3. Add a brief section in `README.md` explaining known data differences (ETF history vs. Global Financial Data, adjusted close vs. total return indices)

**Relevant Context:**
- Expect minor discrepancies: paper uses GFD data back to 1973; ETF data starts ~2007 for common history; spliced long-history series extends monthly backtest further but EAFE/GSCI/NAREIT still limited to ETF launch dates
- Adjusted close in yfinance includes dividends but may differ from GFD total return construction
- Bond return construction from FRED yields (Sub-Task 2b) is an approximation; document clearly
- The validation step is important for building confidence before applying to live portfolios

**Status:** [x] complete

---

## Data Source Notes

| Asset Class | Paper Index | ETF Proxy (yfinance) | ETF Start | Long-History Free Source | Long-History Start |
|---|---|---|---|---|---|
| US Large Cap | S&P 500 TR | SPY | Jan 1993 | Shiller `ie_data.xls` (total return) | Jan 1871 |
| Foreign Developed | MSCI EAFE | EFA | Aug 2001 | None (ETF only) | Aug 2001 |
| US 10-Year Bonds | GFD 10yr T-Bond TR | IEF | Jul 2002 | FRED `DGS10` yield → approx. return | Apr 1953 |
| Commodities | GSCI | GSG | Jul 2006 | None (ETF only) | Jul 2006 |
| US REITs | NAREIT | VNQ | Sep 2004 | None (ETF only) | Sep 2004 |
| Cash Proxy | 90-day T-bills | BIL | May 2007 | FRED `TB3MS` rate → monthly return | Apr 1934 |

**ETF common start date (all 5 assets + cash): ~July 2007**
**Monthly spliced common start (S&P + bonds + cash only): ~April 1953 (3-asset subset); full 5-asset set limited to ETF dates**

---

## Frequency → SMA Period Mapping

| Frequency | Resampling | SMA Period | Rationale |
|---|---|---|---|
| Monthly | Month-end close | 10 periods | Directly from paper |
| Weekly | Week-end close | 43 periods | 10 months × 4.3 weeks/month |
| Daily | Daily close | 200 periods | Standard "200-day MA" equivalent |
