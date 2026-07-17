"""
tests/test_splice.py

Unit tests for splice_data.py covering:
  - splice(): ETF's first-month return is preserved (no boundary erasure)
  - splice(): long-history-only case (no ETF data)
  - splice(): ETF-only case (no pre-ETF data)
  - splice(): scale factor correct at junction
  - _build_long_history_map(): reads paths from config correctly

Run with:  python3 -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from splice_data import splice, _build_long_history_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_series(values, start, name="X") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name=name, dtype=float)


# ---------------------------------------------------------------------------
# splice()
# ---------------------------------------------------------------------------

class TestSplice:
    def test_etf_first_month_return_preserved(self):
        """
        The ETF's return for its launch month must not be erased.

        Previous bug: anchoring at M-1 set long[M-1] * scale == etf[M],
        which forced pct_change across the boundary to 0%.
        Fix: anchor at M (long[M] * scale == etf[M]) and exclude M from
        the pre-ETF slice, so etf[M] is the first ETF row in the output.
        """
        # Long history: 10 months, values 10..19
        long = _monthly_series(range(10, 20), "2000-01-31", "LH")
        # ETF starts at month 11 (2000-11-30), values 25, 30, 35
        etf  = _monthly_series([25.0, 30.0, 35.0], "2000-11-30", "LH")

        combined = splice(long, etf)

        # The ETF's first return: (30 - 25) / 25 = 20%
        etf_first_return = (etf.iloc[1] - etf.iloc[0]) / etf.iloc[0]

        # In the spliced series the same two dates should give the same return
        splice_at_etf = combined.loc[etf.index[0]]
        splice_next   = combined.loc[etf.index[1]]
        spliced_return = (splice_next - splice_at_etf) / splice_at_etf

        assert spliced_return == pytest.approx(etf_first_return, rel=1e-9)

    def test_etf_only_when_no_pre_etf_data(self):
        """When long history starts at or after ETF launch, use ETF only."""
        # Long starts same month as ETF → no pre-ETF rows
        etf  = _monthly_series([100.0, 110.0, 121.0], "2000-01-31", "A")
        long = _monthly_series([100.0, 110.0, 121.0], "2000-01-31", "A")

        combined = splice(long, etf)

        # Output should be the ETF series (possibly rescaled to start at 1)
        # Verify returns are identical
        assert combined.pct_change().dropna().tolist() == pytest.approx(
            etf.pct_change().dropna().tolist(), rel=1e-9
        )

    def test_long_only_when_etf_empty(self):
        """When ETF series is empty, return the long-history series unchanged."""
        long = _monthly_series([1.0, 1.1, 1.2], "2000-01-31", "A")
        etf  = pd.Series(name="A", dtype=float)

        combined = splice(long, etf)
        pd.testing.assert_series_equal(combined, long)

    def test_scale_factor_at_junction(self):
        """
        The last pre-ETF value (scaled) and the ETF's first value should
        be correctly anchored: long[M] * scale == etf[M].
        """
        long = _monthly_series([10.0, 20.0, 30.0], "2000-01-31", "A")
        # ETF starts 2000-03-31 with value 60 (long[M=2000-03-31] = 30)
        etf  = _monthly_series([60.0, 66.0], "2000-03-31", "A")

        combined = splice(long, etf)

        # At the ETF start date the combined value must equal the ETF value
        assert combined.loc[etf.index[0]] == pytest.approx(etf.iloc[0], rel=1e-9)

    def test_output_has_no_duplicate_index(self):
        long = _monthly_series(range(10, 20), "2000-01-31", "A")
        etf  = _monthly_series([25.0, 30.0], "2000-11-30", "A")
        combined = splice(long, etf)
        assert not combined.index.duplicated().any()

    def test_output_is_sorted(self):
        long = _monthly_series(range(10, 20), "2000-01-31", "A")
        etf  = _monthly_series([25.0, 30.0], "2000-11-30", "A")
        combined = splice(long, etf)
        assert combined.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# _build_long_history_map()
# ---------------------------------------------------------------------------

class TestBuildLongHistoryMap:
    def test_reads_entries_from_config(self):
        from config import load_config, ROOT
        cfg = load_config()
        m = _build_long_history_map(cfg)
        sources = cfg.get("long_history_sources", {})
        assert set(m.keys()) == set(sources.keys())

    def test_paths_are_absolute(self):
        from config import load_config
        cfg = load_config()
        m = _build_long_history_map(cfg)
        for ticker, (col, path) in m.items():
            assert path.is_absolute(), f"Path for {ticker} is not absolute: {path}"

    def test_columns_match_config(self):
        from config import load_config
        cfg = load_config()
        m = _build_long_history_map(cfg)
        sources = cfg.get("long_history_sources", {})
        for ticker, (col, _) in m.items():
            assert col == sources[ticker]["column"]

    def test_empty_config_returns_empty_map(self):
        cfg = {"long_history_sources": {}}
        assert _build_long_history_map(cfg) == {}

    def test_missing_section_returns_empty_map(self):
        cfg = {}
        assert _build_long_history_map(cfg) == {}
