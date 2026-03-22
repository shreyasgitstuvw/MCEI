"""Unit tests for ETFArbitrageAnalyzer and SmartMoneyTracker."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analytics.etf_smart_money import ETFArbitrageAnalyzer, SmartMoneyTracker

BASE_DATE = date(2026, 2, 3)


def make_index_df():
    """Index data with security_name matching ETF underlying_map values."""
    from tests.conftest import DATES
    rows = []
    underlyings = {"Nifty 50": 20000.0, "Nifty Bank": 45000.0, "GOLD": 60000.0}
    for name, base in underlyings.items():
        for d in DATES:
            close = base * (1 + np.random.default_rng(0).normal(0, 0.005))
            rows.append({
                "symbol": name.replace(" ", ""),
                "security_name": name,
                "trade_date": d,
                "close_price": round(close, 2),
                "prev_close": round(close * 0.999, 2),
                "net_traded_qty": 1_000_000,
                "net_traded_value": round(close * 1_000_000, 2),
                "is_index": True,
                "is_valid": True,
            })
    return pd.DataFrame(rows)


class TestSmartMoneyTracker:

    def test_identify_institutional_buying_returns_dataframe(self, price_df, etf_df, top_traded_df):
        tracker = SmartMoneyTracker(price_df, top_traded_df, etf_df)
        result = tracker.identify_institutional_buying()
        assert isinstance(result, pd.DataFrame)

    def test_identify_institutional_buying_empty_when_no_conditions_met(self, price_df, etf_df, top_traded_df):
        tracker = SmartMoneyTracker(price_df, top_traded_df, etf_df)
        result = tracker.identify_institutional_buying(volume_threshold=100.0, value_threshold=1e20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_track_top_traded_changes_returns_dict(self, price_df, etf_df):
        top = pd.DataFrame({
            "trade_date": [BASE_DATE, BASE_DATE, BASE_DATE + timedelta(days=1),
                           BASE_DATE + timedelta(days=1)],
            "security_name": ["RELIANCE", "TCS", "RELIANCE", "INFY"],
            "net_traded_value": [1e10, 8e9, 1e10, 7e9],
            "rank": [1, 2, 1, 2],
        })
        tracker = SmartMoneyTracker(price_df, top, etf_df)
        result = tracker.track_top_traded_changes()
        assert isinstance(result, dict)
        assert "date" in result
        assert "new_entries" in result
        assert "exits" in result

    def test_calculate_money_flow_index_returns_dataframe(self, price_df, etf_df, top_traded_df):
        tracker = SmartMoneyTracker(price_df, top_traded_df, etf_df)
        result = tracker.calculate_money_flow_index(window=5)
        assert isinstance(result, pd.DataFrame)
        assert "mfi" in result.columns

    def test_safe_mode_handles_all_nan_category(self, price_df, etf_df, top_traded_df):
        """Regression: mode()[0] on all-NaN Categorical raised KeyError: 0.
        Uses ETF symbols that match underlying_map so flow_status is populated."""
        tracker = SmartMoneyTracker(price_df, top_traded_df, etf_df)
        result = tracker.analyze_etf_institutional_flow()
        assert isinstance(result, pd.DataFrame)


class TestETFArbitrageAnalyzer:

    def test_calculate_tracking_error_returns_dataframe(self, etf_df):
        index_df = make_index_df()
        analyzer = ETFArbitrageAnalyzer(etf_df, index_df)
        result = analyzer.calculate_tracking_error()
        assert isinstance(result, pd.DataFrame)

    def test_detect_premium_discount_returns_dataframe(self, etf_df):
        index_df = make_index_df()
        analyzer = ETFArbitrageAnalyzer(etf_df, index_df)
        result = analyzer.detect_premium_discount()
        assert isinstance(result, pd.DataFrame)

    def test_analyze_etf_flows_returns_dataframe(self, etf_df):
        index_df = make_index_df()
        analyzer = ETFArbitrageAnalyzer(etf_df, index_df)
        result = analyzer.analyze_etf_flows()
        assert isinstance(result, pd.DataFrame)
        assert "flow_status" in result.columns

    def test_categorize_etf_known_types(self):
        # _categorize_etf is a static method on SmartMoneyTracker
        cat = SmartMoneyTracker._categorize_etf
        assert cat("Nifty Bank") == "Banking"
        assert cat("GOLD") == "Commodities"
        assert cat("Nifty 50") == "Large Cap"
        assert cat(None) == "Other"
        assert cat(float("nan")) == "Other"
