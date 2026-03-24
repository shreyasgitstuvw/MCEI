"""Unit tests for LeadLagAnalyzer."""
import pytest
import pandas as pd
import numpy as np

from src.analytics.causality_analysis import LeadLagAnalyzer


class TestLeadLagAnalyzerInit:

    def test_init_succeeds(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        assert analyzer.returns is not None
        assert len(analyzer.symbols) > 0

    def test_init_raises_on_empty_dataframe(self):
        empty = pd.DataFrame(columns=["trade_date", "symbol", "close_price", "is_index"])
        with pytest.raises(ValueError, match="price_data is empty"):
            LeadLagAnalyzer(empty)

    def test_init_filters_index_rows(self, price_df, index_df):
        combined = pd.concat([price_df, index_df], ignore_index=True)
        analyzer = LeadLagAnalyzer(combined)
        # index_df symbols should not appear in analyzer.symbols
        assert "Nifty 50" not in analyzer.symbols

    def test_returns_are_log_returns(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        # Log returns should be small for small price changes
        assert analyzer.returns.abs().max().max() < 0.5

    def test_market_return_is_series(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        assert isinstance(analyzer._market_return, pd.Series)
        assert len(analyzer._market_return) > 0


class TestCorrelationMatrix:

    def test_returns_dataframe(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.correlation_matrix()
        assert isinstance(result, pd.DataFrame)

    def test_is_square(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.correlation_matrix()
        assert result.shape[0] == result.shape[1]

    def test_diagonal_is_one(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.correlation_matrix()
        diag = np.diag(result.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-6)

    def test_top_n_limits_size(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        top_n = 3
        result = analyzer.correlation_matrix(top_n=top_n)
        assert result.shape[0] <= top_n

    def test_values_in_valid_range(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.correlation_matrix()
        assert (result.values >= -1.0 - 1e-6).all()
        assert (result.values <= 1.0 + 1e-6).all()


class TestLeadLagCorrelation:

    def test_returns_dataframe(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.lead_lag_correlation("RELIANCE", max_lag=2)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.lead_lag_correlation("RELIANCE", max_lag=2)
        for col in ["lag", "symbol", "correlation"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_raises_on_unknown_symbol(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        with pytest.raises(ValueError, match="not found in price data"):
            analyzer.lead_lag_correlation("UNKNOWN_SYM", max_lag=2)

    def test_lag_range_is_correct(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        max_lag = 3
        result = analyzer.lead_lag_correlation("RELIANCE", max_lag=max_lag)
        if not result.empty:
            assert result["lag"].min() >= -max_lag
            assert result["lag"].max() <= max_lag

    def test_excludes_self_correlation(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.lead_lag_correlation("RELIANCE", max_lag=2)
        assert "RELIANCE" not in result["symbol"].values


class TestFindMarketLeaders:

    def test_returns_dataframe(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.find_market_leaders()
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.find_market_leaders()
        for col in ["symbol", "lag1_corr", "direction", "strength"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_top_n_limits_rows(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        top_n = 3
        result = analyzer.find_market_leaders(top_n=top_n)
        assert len(result) <= top_n

    def test_direction_values_are_valid(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.find_market_leaders()
        if not result.empty:
            valid_directions = {"Leading +", "Leading −"}
            assert set(result["direction"].unique()).issubset(valid_directions)

    def test_strength_values_are_valid(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.find_market_leaders()
        if not result.empty:
            valid_strengths = {"Strong", "Moderate", "Weak"}
            assert set(result["strength"].unique()).issubset(valid_strengths)


class TestRollingCorrelation:

    def test_returns_series(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.rolling_correlation("RELIANCE", "TCS", window=5)
        assert isinstance(result, pd.Series)

    def test_values_in_valid_range(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.rolling_correlation("RELIANCE", "TCS", window=5)
        assert (result.values >= -1.0 - 1e-6).all()
        assert (result.values <= 1.0 + 1e-6).all()

    def test_raises_on_unknown_sym1(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        with pytest.raises(ValueError, match="not in data"):
            analyzer.rolling_correlation("UNKNOWN", "TCS", window=5)

    def test_raises_on_unknown_sym2(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        with pytest.raises(ValueError, match="not in data"):
            analyzer.rolling_correlation("RELIANCE", "UNKNOWN", window=5)


class TestClusteredCorrelationMatrix:

    def test_returns_dataframe(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.clustered_correlation_matrix()
        assert isinstance(result, pd.DataFrame)

    def test_is_square(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.clustered_correlation_matrix()
        assert result.shape[0] == result.shape[1]

    def test_same_symbols_as_correlation_matrix(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        standard = set(analyzer.correlation_matrix().columns)
        clustered = set(analyzer.clustered_correlation_matrix().columns)
        assert standard == clustered


class TestLagProfile:

    def test_returns_dataframe(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.lag_profile(max_lag=3)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.lag_profile(max_lag=3)
        for col in ["lag", "avg_abs_corr", "n_pairs"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_matches_lag_range(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        max_lag = 3
        result = analyzer.lag_profile(max_lag=max_lag)
        expected_rows = 2 * max_lag + 1  # -max_lag to +max_lag inclusive
        assert len(result) == expected_rows

    def test_lag_zero_has_highest_avg_corr(self, price_df):
        analyzer = LeadLagAnalyzer(price_df)
        result = analyzer.lag_profile(max_lag=3).dropna(subset=["avg_abs_corr"])
        if not result.empty:
            lag0_corr = result.loc[result["lag"] == 0, "avg_abs_corr"].iloc[0]
            assert lag0_corr == result["avg_abs_corr"].max()
