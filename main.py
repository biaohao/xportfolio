"""
main.py — Sub-Task 7
Single entry point for the TAA replication project.

Usage:
    python3 main.py --fetch      # download ETF + long-history data
    python3 main.py --backtest   # run backtests, generate reports and plots
    python3 main.py --signals    # generate live BUY/SELL signals
    python3 main.py --all        # run full pipeline (fetch + backtest + signals)

Individual scripts can also be run directly:
    python3 fetch_etf.py [--refresh]
    python3 fetch_longhistory.py [--refresh]
    python3 splice_data.py
    python3 report.py [--freq monthly|weekly|daily]
    python3 plot.py [--freq monthly|weekly|daily]
    python3 signals.py [--freq monthly|weekly|daily]
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_fetch(refresh: bool = False) -> None:
    log.info("=== STEP 1/3: Fetching data ===")
    import fetch_etf
    fetch_etf.main(refresh=refresh)

    import fetch_longhistory
    fetch_longhistory.main(refresh=refresh)

    import fetch_bitcoin
    fetch_bitcoin.main(refresh=refresh)

    import splice_data
    splice_data.main()
    log.info("")


def step_backtest() -> None:
    log.info("=== STEP 2/3: Running backtests ===")
    import report
    import plot
    _, results = report.main()
    plot.main(results=results)
    log.info("")


def step_signals(frequencies=None) -> None:
    log.info("=== STEP 3/3: Generating live signals ===")
    import signals
    signals.generate_signals(frequencies=frequencies)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAA Replication — Faber GTAA strategy backtester and signal generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fetch",
        action="store_true",
        help="Download ETF and long-history data.",
    )
    group.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtests across all frequencies and generate reports + plots.",
    )
    group.add_argument(
        "--signals",
        action="store_true",
        help="Generate current BUY/SELL signals from live price data.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: fetch → backtest → signals.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="(With --fetch or --all) Force re-download of cached data.",
    )
    parser.add_argument(
        "--freq",
        choices=["monthly", "weekly", "daily"],
        default=None,
        help="(With --backtest or --signals) Limit to a single frequency.",
    )

    args = parser.parse_args()

    if args.fetch:
        step_fetch(refresh=args.refresh)

    elif args.backtest:
        import report
        import plot
        freqs = [args.freq] if args.freq else None
        _, results = report.main(frequencies=freqs)
        plot.main(frequencies=freqs, results=results)

    elif args.signals:
        freqs = [args.freq] if args.freq else None
        step_signals(frequencies=freqs)

    elif args.all:
        step_fetch(refresh=args.refresh)
        step_backtest()
        step_signals()


if __name__ == "__main__":
    main()
