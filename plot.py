"""
plot.py — Sub-Task 5
Produce the key charts from the Faber GTAA paper:
  1. Equity curves (log scale) — B&H vs Timing
  2. Drawdown chart            — B&H vs Timing
  3. Annual returns bar chart  — B&H vs Timing side by side
  4. Asset signal heatmap      — invested (green) / cash (red) per asset per period

All plots are saved to the plots/ directory as PNG files.

Usage:
    python3 plot.py                  # generate plots for all configured frequencies
    python3 plot.py --freq monthly   # one frequency only
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

from metrics import cumulative_returns, drawdown_series
from report import load_prices, resolve_rf, resolve_window
from strategy import run_backtest

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
PLOTS_DIR = ROOT / "plots"

# Consistent colour palette
C_BH = "#2563eb"      # blue  — Buy & Hold
C_TM = "#16a34a"      # green — Timing
C_INV = "#16a34a"     # green — invested
C_CASH = "#ef4444"    # red   — cash


# ---------------------------------------------------------------------------
# 1. Equity Curve
# ---------------------------------------------------------------------------

def plot_equity_curves(
    returns_bh: pd.Series,
    returns_timing: pd.Series,
    freq: str,
    output_path: Path,
) -> None:
    cum_bh = cumulative_returns(returns_bh)
    cum_tm = cumulative_returns(returns_timing)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(cum_bh.index, cum_bh.values, color=C_BH, linewidth=1.2, label="Buy & Hold")
    ax.semilogy(cum_tm.index, cum_tm.values, color=C_TM, linewidth=1.5, label="GTAA Timing")

    ax.set_title(f"Equity Curve — {freq.capitalize()} ({cum_bh.index[0].year}–{cum_bh.index[-1].year})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1 (log scale)", fontsize=10)
    ax.set_xlabel("")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", output_path)


# ---------------------------------------------------------------------------
# 2. Drawdown Chart
# ---------------------------------------------------------------------------

def plot_drawdowns(
    returns_bh: pd.Series,
    returns_timing: pd.Series,
    freq: str,
    output_path: Path,
) -> None:
    dd_bh = drawdown_series(returns_bh) * 100
    dd_tm = drawdown_series(returns_timing) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd_bh.index, dd_bh.values, 0, alpha=0.35, color=C_BH, label="Buy & Hold")
    ax.fill_between(dd_tm.index, dd_tm.values, 0, alpha=0.5, color=C_TM, label="GTAA Timing")
    ax.plot(dd_bh.index, dd_bh.values, color=C_BH, linewidth=0.8)
    ax.plot(dd_tm.index, dd_tm.values, color=C_TM, linewidth=1.0)

    ax.set_title(f"Drawdowns — {freq.capitalize()} ({dd_bh.index[0].year}–{dd_bh.index[-1].year})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", output_path)


# ---------------------------------------------------------------------------
# 3. Annual Returns Bar Chart
# ---------------------------------------------------------------------------

def plot_annual_returns(
    returns_bh: pd.Series,
    returns_timing: pd.Series,
    freq: str,
    output_path: Path,
) -> None:
    def annual(r: pd.Series) -> pd.Series:
        return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) * 100

    bh_ann = annual(returns_bh.dropna())
    tm_ann = annual(returns_timing.dropna())

    years = sorted(set(bh_ann.index) | set(tm_ann.index))
    x = np.arange(len(years))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(14, len(years) * 0.35), 5))
    bars_bh = ax.bar(x - width / 2, [bh_ann.get(y, 0) for y in years],
                     width, label="Buy & Hold", color=C_BH, alpha=0.8)
    bars_tm = ax.bar(x + width / 2, [tm_ann.get(y, 0) for y in years],
                     width, label="GTAA Timing", color=C_TM, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=90, fontsize=7)
    ax.set_title(f"Annual Returns — {freq.capitalize()}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Return (%)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", output_path)


# ---------------------------------------------------------------------------
# 4. Signal Heatmap
# ---------------------------------------------------------------------------

def plot_signal_heatmap(
    signals_df: pd.DataFrame,
    asset_names: dict[str, str],
    freq: str,
    output_path: Path,
) -> None:
    if signals_df.empty:
        log.warning("  Empty signals DataFrame; skipping heatmap for %s", freq)
        return

    # For readability, cap to last 20 years if very long
    cutoff = signals_df.index[-1] - pd.DateOffset(years=20)
    df = signals_df[signals_df.index >= cutoff].copy()

    labels = [asset_names.get(c, c) for c in df.columns]
    data = df.values.T   # shape: (n_assets, n_periods)

    fig, ax = plt.subplots(figsize=(14, max(3, len(df.columns) * 0.6)))
    cmap = matplotlib.colors.ListedColormap([C_CASH, C_INV])
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[mdates.date2num(df.index[0]), mdates.date2num(df.index[-1]),
                      -0.5, len(df.columns) - 0.5])

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    ax.set_title(f"Asset Signals — {freq.capitalize()} (green=invested, red=cash)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_plots(frequencies: list[str], cfg: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cash_ticker = cfg["cash_proxy"]
    sma_periods: dict = cfg["strategy"]["sma_periods"]
    weights_cfg = cfg["strategy"]["weights"]
    asset_names: dict = cfg["assets"]
    bt_start, bt_end = resolve_window(cfg)

    for freq in frequencies:
        sma = sma_periods.get(freq)
        if sma is None:
            log.warning("No SMA period configured for '%s'; skipping plots.", freq)
            continue

        log.info("Generating plots: frequency=%s  SMA=%d  window=%s → %s",
                 freq, sma, bt_start or "earliest", bt_end or "latest")
        # Load full history up to end — start trim applied after backtest (warm-up)
        prices = load_prices(freq, end=bt_end)

        if cash_ticker not in prices.columns:
            log.warning("Cash proxy '%s' not found in %s data; skipping.", cash_ticker, freq)
            continue

        result = run_backtest(
            prices_df=prices,
            cash_col=cash_ticker,
            frequency=freq,
            sma_period=sma,
            weights_cfg=weights_cfg,
        )

        # Trim to backtest window start (after SMA warm-up over full history)
        from report import _trim_result_to_window
        _trim_result_to_window(result, bt_start)
        if result.returns_bh.empty:
            log.warning("No data after trimming to %s for %s; skipping plots.", bt_start, freq)
            continue

        plot_equity_curves(
            result.returns_bh, result.returns_timing, freq,
            PLOTS_DIR / f"equity_curve_{freq}.png",
        )
        plot_drawdowns(
            result.returns_bh, result.returns_timing, freq,
            PLOTS_DIR / f"drawdown_{freq}.png",
        )
        plot_annual_returns(
            result.returns_bh, result.returns_timing, freq,
            PLOTS_DIR / f"yearly_returns_{freq}.png",
        )
        plot_signal_heatmap(
            result.signals, asset_names, freq,
            PLOTS_DIR / f"asset_signals_{freq}.png",
        )
        log.info("")


def main(frequencies: list[str] | None = None) -> None:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    if frequencies is None:
        frequencies = cfg["strategy"]["rebalance_frequencies"]

    generate_all_plots(frequencies, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GTAA backtest plots.")
    parser.add_argument(
        "--freq",
        choices=["monthly", "weekly", "daily"],
        default=None,
        help="Generate plots for a single frequency only.",
    )
    args = parser.parse_args()
    freqs = [args.freq] if args.freq else None
    main(frequencies=freqs)
