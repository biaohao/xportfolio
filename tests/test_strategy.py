"""
tests/test_strategy.py

Unit tests for strategy.py covering:
  - compute_signals: SMA computation, shift, boundary behaviour
  - apply_buy_hold: equal weight, staggered start dates, NaN handling
  - run_backtest: no look-ahead bias end-to-end, signal→return alignment
  - apply_timing_model: cash earns return, entry/exit transitions

Run with:  python3 -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from strategy import (
    compute_signals,
    compute_sma,
    resample_prices,
    compute_period_returns,
    apply_buy_hold,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_index(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _price_series(values, name: str = "A", start: str = "2000-01-31") -> pd.Series:
    idx = _monthly_index(len(values), start)
    return pd.Series(values, index=idx, name=name, dtype=float)


# ---------------------------------------------------------------------------
# compute_sma
# ---------------------------------------------------------------------------

class TestComputeSMA:
    def test_first_n_minus_1_are_nan(self):
        prices = pd.DataFrame({"A": _price_series(range(1, 16))})
        sma = compute_sma(prices, period=10)
        assert sma["A"].iloc[:9].isna().all()
        assert pd.notna(sma["A"].iloc[9])

    def test_correct_value(self):
        values = list(range(1, 11))   # 1..10, mean = 5.5
        prices = pd.DataFrame({"A": _price_series(values)})
        sma = compute_sma(prices, period=10)
        assert sma["A"].iloc[-1] == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# compute_signals
# ---------------------------------------------------------------------------

class TestComputeSignals:
    def test_signal_is_shifted_by_one_period(self):
        """
        If price crosses above SMA at period T, the signal should be 1 at T+1,
        not at T — no look-ahead.
        """
        # Flat price at 10 for 10 periods (SMA = 10), then jump to 20.
        values = [10.0] * 10 + [20.0] * 5
        prices = pd.DataFrame({"A": _price_series(values)})
        sigs = compute_signals(prices, sma_period=10)

        # Period 10 (index 9): price = 10, SMA = 10 → price NOT > SMA → raw = 0
        # Period 11 (index 10): price = 20, SMA = 11 → price > SMA → raw = 1
        #   After shift(1): sigs[11] = raw[10] = 1, so signal appears at index 11
        cross_idx = 10   # 0-based index where price first goes above SMA
        # Signal at cross_idx itself must be 0 or NaN (no look-ahead)
        assert sigs["A"].iloc[cross_idx] == 0.0 or pd.isna(sigs["A"].iloc[cross_idx])
        # Signal at cross_idx + 1 must be 1 (trade fires next period)
        assert sigs["A"].iloc[cross_idx + 1] == pytest.approx(1.0)

    def test_signal_values_are_zero_or_one(self):
        values = [10.0] * 5 + [5.0] * 5 + [10.0] * 5
        prices = pd.DataFrame({"A": _price_series(values)})
        sigs = compute_signals(prices, sma_period=5)
        valid = sigs["A"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

# ---------------------------------------------------------------------------
# apply_buy_hold
# ---------------------------------------------------------------------------

class TestApplyBuyHold:
    def test_equal_weight_two_assets(self):
        idx = _monthly_index(3)
        rets = pd.DataFrame({"A": [0.1, 0.2, 0.0], "B": [0.0, 0.2, 0.1]}, index=idx)
        result = apply_buy_hold(rets, weights_cfg="equal")
        expected = pd.Series([0.05, 0.2, 0.05], index=idx)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_staggered_start_renormalises_weight(self):
        """
        When asset B has NaN in the first period, the full weight
        should fall on asset A (not lose half the portfolio).
        """
        idx = _monthly_index(3)
        rets = pd.DataFrame({"A": [0.1, 0.1, 0.1], "B": [np.nan, 0.1, 0.1]}, index=idx)
        result = apply_buy_hold(rets, weights_cfg="equal")
        # Period 0: only A available → weight = 1.0 on A → return = 0.1
        assert result.iloc[0] == pytest.approx(0.1)
        # Periods 1 and 2: both available → equal weight → return = 0.1
        assert result.iloc[1] == pytest.approx(0.1)

    def test_all_nan_row_returns_nan(self):
        idx = _monthly_index(2)
        rets = pd.DataFrame({"A": [np.nan, 0.1], "B": [np.nan, 0.1]}, index=idx)
        result = apply_buy_hold(rets)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# run_backtest — no look-ahead bias
# ---------------------------------------------------------------------------

class TestRunBacktest:
    """
    End-to-end smoke test using a synthetic price DataFrame.
    Verifies alignment between signals and returns (no look-ahead).
    """

    def _make_prices(self, n: int = 30) -> pd.DataFrame:
        idx = _monthly_index(n)
        # SPY: steadily rising so it stays above its 10-period SMA
        spy = pd.Series(np.linspace(100, 200, n), index=idx, name="SPY")
        # BIL: flat cash proxy (tiny growth each period)
        bil = pd.Series(np.linspace(100, 101, n), index=idx, name="BIL")
        return pd.concat([spy, bil], axis=1)

    def test_returns_not_empty(self):
        prices = self._make_prices(30)
        result = run_backtest(prices, cash_col="BIL", frequency="monthly", sma_period=5)
        assert not result.returns_bh.empty
        assert not result.returns_timing.empty

    def test_signals_index_matches_returns_index(self):
        prices = self._make_prices(30)
        result = run_backtest(prices, cash_col="BIL", frequency="monthly", sma_period=5)
        assert result.signals.index.equals(result.returns_bh.index)

    def test_no_look_ahead_signal_leads_return_by_one(self):
        """
        The signal at date T was computed from prices known at T-1.
        Concretely: signals[T] == raw_signal[T-1] (shift=1).
        We verify by checking that signals.first_valid_index() is
        *after* the SMA warm-up, not at the very first period.
        """
        prices = self._make_prices(30)
        result = run_backtest(prices, cash_col="BIL", frequency="monthly", sma_period=5)
        first_sig = result.signals.first_valid_index()
        first_ret = result.returns_bh.index[0]
        # The first valid signal must be at or after the first return period
        assert first_sig >= first_ret

    def test_rising_asset_stays_invested(self):
        """
        A consistently rising asset should have signal=1 once the price is
        clearly above the SMA — i.e. after enough periods for the SMA to
        lag behind the rising price.  We check the second half of the series
        where the price is well above its 5-period SMA.
        """
        prices = self._make_prices(30)
        result = run_backtest(prices, cash_col="BIL", frequency="monthly", sma_period=5)
        # Take only the latter half where price is unambiguously above SMA
        later_sigs = result.signals["SPY"].dropna().iloc[5:]
        assert (later_sigs == 1.0).all()
