"""
tests/test_metrics.py

Unit tests for metrics.py covering:
  - cagr: known input → expected output
  - annualised_volatility: correct annualisation factor
  - sharpe_ratio: zero excess return edge case
  - max_drawdown: peak-to-trough correctness
  - ulcer_index: always non-negative
  - win_rate: fraction of positive periods
  - best_worst_year: correct year labels
  - dollars_to: terminal wealth
  - wealth_index: log-space overflow safety

Run with:  python3 -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from metrics import (
    cagr,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    ulcer_index,
    win_rate,
    best_worst_year,
    dollars_to,
    wealth_index,
    drawdown_series,
    cumulative_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_returns(values, start="2000-01-31") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# wealth_index
# ---------------------------------------------------------------------------

class TestWealthIndex:
    def test_starts_at_first_growth_factor(self):
        r = _monthly_returns([0.1, 0.0, -0.1])
        w = wealth_index(r)
        assert w.iloc[0] == pytest.approx(1.1)

    def test_matches_cumprod_for_small_returns(self):
        """Log-space and cumprod should agree for normal return magnitudes."""
        r = _monthly_returns([0.01, -0.02, 0.03, -0.01])
        w_log = wealth_index(r)
        w_cum = (1 + r).cumprod()
        pd.testing.assert_series_equal(w_log, w_cum, check_names=False, rtol=1e-10)

    def test_no_overflow_on_long_history(self):
        """150 years of 10% annual monthly returns should not overflow."""
        monthly_r = (1.10) ** (1 / 12) - 1
        r = _monthly_returns([monthly_r] * 1800)
        w = wealth_index(r)
        assert np.isfinite(w.iloc[-1])
        assert w.iloc[-1] > 1e6   # substantial growth


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_10pct_annual_monthly(self):
        """
        Monthly returns that compound to 10% annually should produce a CAGR
        close to 10%.  We use a wide tolerance (±0.5pp) because cagr() infers
        years from calendar days and month-end dates don't land on exactly
        integer year boundaries.
        """
        monthly_r = (1.10) ** (1 / 12) - 1
        r = _monthly_returns([monthly_r] * 24)
        assert cagr(r) == pytest.approx(0.10, abs=5e-3)

    def test_zero_return(self):
        r = _monthly_returns([0.0] * 24)
        assert cagr(r) == pytest.approx(0.0, abs=1e-6)

    def test_empty_returns_nan(self):
        assert np.isnan(cagr(pd.Series([], dtype=float)))

    def test_single_row_nan(self):
        r = _monthly_returns([0.05])
        assert np.isnan(cagr(r))


# ---------------------------------------------------------------------------
# annualised_volatility
# ---------------------------------------------------------------------------

class TestAnnualisedVolatility:
    def test_monthly_annualisation(self):
        np.random.seed(42)
        r = _monthly_returns(np.random.normal(0, 0.01, 120))
        vol = annualised_volatility(r, "monthly")
        assert vol == pytest.approx(r.std() * np.sqrt(12), rel=1e-10)

    def test_daily_annualisation(self):
        idx = pd.date_range("2000-01-01", periods=252, freq="B")
        r = pd.Series(np.random.normal(0, 0.01, 252), index=idx)
        vol = annualised_volatility(r, "daily")
        assert vol == pytest.approx(r.std() * np.sqrt(252), rel=1e-10)

    def test_unknown_frequency_raises(self):
        r = _monthly_returns([0.01] * 12)
        with pytest.raises(ValueError):
            annualised_volatility(r, "hourly")


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_risk_free_positive_returns(self):
        r = _monthly_returns([0.01] * 24)
        s = sharpe_ratio(r, rf=0.0, frequency="monthly")
        assert np.isnan(s)   # zero std → nan (no meaningful Sharpe)

    def test_higher_returns_higher_sharpe(self):
        r_low  = _monthly_returns([0.005] * 60 + [-0.01] * 60)
        r_high = _monthly_returns([0.02]  * 60 + [-0.01] * 60)
        assert sharpe_ratio(r_high, 0.0, "monthly") > sharpe_ratio(r_low, 0.0, "monthly")

    def test_series_rf_aligns_on_index(self):
        r = _monthly_returns([0.01, 0.02, 0.03])
        rf = r * 0.5   # rf is half the return
        s = sharpe_ratio(r, rf=rf, frequency="monthly")
        assert np.isfinite(s)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Strictly rising returns → max drawdown = 0."""
        r = _monthly_returns([0.01] * 24)
        assert max_drawdown(r) == pytest.approx(0.0, abs=1e-10)

    def test_50pct_drawdown(self):
        """A 50% drop from peak."""
        # Price path: 100 → 200 → 100
        # Returns:  +100%,  -50%
        r = _monthly_returns([1.0, -0.5])
        assert max_drawdown(r) == pytest.approx(-0.5, rel=1e-6)

    def test_returns_negative_number(self):
        r = _monthly_returns([0.1, -0.2, 0.1])
        assert max_drawdown(r) < 0


# ---------------------------------------------------------------------------
# ulcer_index
# ---------------------------------------------------------------------------

class TestUlcerIndex:
    def test_always_non_negative(self):
        np.random.seed(7)
        r = _monthly_returns(np.random.normal(0, 0.05, 60))
        assert ulcer_index(r) >= 0

    def test_zero_for_no_drawdown(self):
        r = _monthly_returns([0.01] * 24)
        assert ulcer_index(r) == pytest.approx(0.0, abs=1e-6)

    def test_larger_for_deeper_drawdown(self):
        shallow = _monthly_returns([0.02, -0.01] * 24)
        deep    = _monthly_returns([0.02, -0.05] * 24)
        assert ulcer_index(deep) > ulcer_index(shallow)


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_positive(self):
        r = _monthly_returns([0.01] * 12)
        assert win_rate(r) == pytest.approx(100.0)

    def test_half_positive(self):
        r = _monthly_returns([0.01, -0.01] * 12)
        assert win_rate(r) == pytest.approx(50.0)

    def test_empty_nan(self):
        assert np.isnan(win_rate(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# best_worst_year
# ---------------------------------------------------------------------------

class TestBestWorstYear:
    def test_identifies_correct_years(self):
        # 2000: 12 months of +1% → good year
        # 2001: 12 months of -5% → bad year
        r2000 = [0.01] * 12
        r2001 = [-0.05] * 12
        idx = pd.date_range("2000-01-31", periods=24, freq="ME")
        r = pd.Series(r2000 + r2001, index=idx)
        best, worst = best_worst_year(r)
        assert "2000" in best
        assert "2001" in worst

    def test_empty_returns_na(self):
        b, w = best_worst_year(pd.Series([], dtype=float))
        assert b == "n/a"
        assert w == "n/a"


# ---------------------------------------------------------------------------
# dollars_to
# ---------------------------------------------------------------------------

class TestDollarsTo:
    def test_doubling(self):
        """100% return → $1 becomes $2."""
        r = _monthly_returns([1.0])
        assert dollars_to(r) == pytest.approx(2.0)

    def test_flat(self):
        r = _monthly_returns([0.0] * 12)
        assert dollars_to(r) == pytest.approx(1.0, abs=1e-4)
