"""Unit tests for BreakoutAnalyzer and BreakoutScreener."""
import pandas as pd
import numpy as np
import pytest
from src.analytics.breakout_analysis import BreakoutAnalyzer, BreakoutScreener


class TestBreakoutAnalyzer:

    @pytest.fixture
    def analyzer(self, price_df, breakout_df):
        return BreakoutAnalyzer(breakout_df, price_df)

    def test_init(self, price_df, breakout_df):
        a = BreakoutAnalyzer(breakout_df, price_df)
        assert hasattr(a, "merged_data")

    def test_calculate_breakout_strength_returns_dataframe(self, analyzer):
        result = analyzer.calculate_breakout_strength()
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"symbol", "trade_date", "high_low", "breakout_margin",
                         "volume_ratio", "strength"}
        assert expected_cols.issubset(result.columns)

    def test_analyze_breakout_success_rate_empty_data(self, price_df):
        # Empty breakout_df should not crash — previously raised KeyError: 'breakout_type'
        empty_breakout = pd.DataFrame(columns=["symbol", "series", "trade_date", "high_low", "hl_type"])
        a = BreakoutAnalyzer(empty_breakout, price_df)
        summary, details = a.analyze_breakout_success_rate()
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(details, pd.DataFrame)
        # Empty inputs → empty outputs
        assert len(details) == 0

    def test_analyze_breakout_success_rate_returns_two_dataframes(self, analyzer):
        summary, details = analyzer.analyze_breakout_success_rate([5, 10])
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(details, pd.DataFrame)

    def test_identify_false_breakouts_empty_data(self, price_df):
        empty_breakout = pd.DataFrame(columns=["symbol", "series", "trade_date", "high_low", "hl_type"])
        a = BreakoutAnalyzer(empty_breakout, price_df)
        result = a.identify_false_breakouts()
        assert isinstance(result, pd.DataFrame)

    def test_analyze_consolidation_before_breakout_empty(self, price_df):
        # Should not crash when no pre-breakout data found
        empty_breakout = pd.DataFrame(columns=["symbol", "series", "trade_date", "high_low", "hl_type"])
        a = BreakoutAnalyzer(empty_breakout, price_df)
        result = a.analyze_consolidation_before_breakout()
        assert isinstance(result, pd.DataFrame)
        assert "symbol" in result.columns

    def test_identify_breakout_clusters_returns_dataframe(self, analyzer):
        result = analyzer.identify_breakout_clusters()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_breakout_momentum_score_returns_dataframe(self, analyzer):
        # Previously crashed due to column name mismatch in merge
        result = analyzer.calculate_breakout_momentum_score()
        assert isinstance(result, pd.DataFrame)
        assert "momentum_score" in result.columns or len(result) == 0


class TestBreakoutScreener:

    def test_screen_near_52w_high_returns_dataframe(self, price_df):
        # Add required columns
        df = price_df.copy()
        df["high_52_week"] = df["high_price"] * 1.05
        df["low_52_week"] = df["low_price"] * 0.95
        screener = BreakoutScreener(df)
        result = screener.screen_near_52w_high(threshold_pct=10.0)
        assert isinstance(result, pd.DataFrame)

    def test_screen_near_52w_low_returns_dataframe(self, price_df):
        df = price_df.copy()
        df["high_52_week"] = df["high_price"] * 1.05
        df["low_52_week"] = df["low_price"] * 0.95
        screener = BreakoutScreener(df)
        result = screener.screen_near_52w_low(threshold_pct=10.0)
        assert isinstance(result, pd.DataFrame)
