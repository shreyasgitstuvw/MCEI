"""Unit tests for MarketRegimeClassifier and RegimeBasedStrategy."""
import sys
import pytest
import pandas as pd

from src.analytics.market_regime import MarketRegimeClassifier, RegimeBasedStrategy

# KMeans (via sklearn/OpenMP) deadlocks during process-based thread init on Windows.
# The rule-based tests (28 of 31) cover the core functionality.
_skip_ml_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="sklearn KMeans OpenMP thread initialization deadlocks on Windows in pytest"
)


class TestMarketRegimeClassifier:

    # ── construction ──────────────────────────────────────────────────────────

    def test_init_succeeds(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        assert clf.nifty is not None
        assert len(clf.nifty) == len(index_df)

    def test_init_raises_when_no_nifty_data(self, price_df, index_df):
        bad_index = index_df.copy()
        bad_index["security_name"] = "Some Other Index"
        with pytest.raises(ValueError, match="Nifty 50 data not found"):
            MarketRegimeClassifier(price_df, bad_index)

    # ── trend indicators ──────────────────────────────────────────────────────

    def test_calculate_trend_indicators_returns_dataframe(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_trend_indicators()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_trend_indicators_has_sma_columns(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_trend_indicators()
        for col in ["sma_20", "sma_50", "sma_200"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_trend_indicators_has_adx(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_trend_indicators()
        assert "adx" in result.columns

    def test_calculate_trend_indicators_has_above_sma_flags(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_trend_indicators()
        for col in ["above_sma20", "above_sma50", "above_sma200"]:
            assert col in result.columns

    # ── volatility indicators ─────────────────────────────────────────────────

    def test_calculate_volatility_indicators_returns_dataframe(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_volatility_indicators()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_volatility_indicators_has_expected_columns(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_volatility_indicators()
        for col in ["volatility", "atr", "atr_pct", "bb_width"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_volatility_atr_pct_non_negative(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_volatility_indicators()
        valid = result["atr_pct"].dropna()
        assert (valid >= 0).all()

    # ── momentum indicators ───────────────────────────────────────────────────

    def test_calculate_momentum_indicators_returns_dataframe(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_momentum_indicators()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_momentum_indicators_has_expected_columns(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_momentum_indicators()
        for col in ["rsi", "macd", "macd_signal", "macd_hist"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_momentum_rsi_in_range(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_momentum_indicators()
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    # ── breadth indicators ────────────────────────────────────────────────────

    def test_calculate_breadth_indicators_returns_dataframe(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_breadth_indicators()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_breadth_indicators_has_expected_columns(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_breadth_indicators()
        for col in ["advances", "declines", "ad_ratio", "ad_line", "advance_pct"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_breadth_advance_pct_in_range(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.calculate_breadth_indicators()
        pct = result["advance_pct"].dropna()
        assert (pct >= 0).all() and (pct <= 100).all()

    # ── rule-based classification ─────────────────────────────────────────────

    def test_classify_regime_rule_based_returns_dataframe(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.classify_regime_rule_based()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_classify_regime_rule_based_has_regime_columns(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.classify_regime_rule_based()
        for col in ["trend", "volatility_regime", "market_regime"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_classify_regime_rule_based_trend_values_are_valid(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.classify_regime_rule_based()
        valid_trends = {
            "Strong Uptrend", "Moderate Uptrend",
            "Strong Downtrend", "Moderate Downtrend", "Sideways"
        }
        assert set(result["trend"].unique()).issubset(valid_trends)

    def test_classify_regime_rule_based_market_regime_is_string(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.classify_regime_rule_based()
        assert result["market_regime"].dtype == object

    # ── ML-based classification ───────────────────────────────────────────────
    # These tests use large_price_df / large_index_df (65 dates) so that
    # roc_20 = pct_change(20) and adx rolling(14,14) produce non-NaN rows
    # that KMeans can cluster.

    @_skip_ml_on_windows
    def test_classify_regime_ml_based_returns_dataframe(self, large_price_df, large_index_df):
        clf = MarketRegimeClassifier(large_price_df, large_index_df)
        result = clf.classify_regime_ml_based(n_regimes=2)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @_skip_ml_on_windows
    def test_classify_regime_ml_based_has_cluster_and_label(self, large_price_df, large_index_df):
        clf = MarketRegimeClassifier(large_price_df, large_index_df)
        result = clf.classify_regime_ml_based(n_regimes=2)
        assert "regime_cluster" in result.columns
        assert "regime_label" in result.columns

    @_skip_ml_on_windows
    def test_classify_regime_ml_based_cluster_count(self, large_price_df, large_index_df):
        n = 2
        clf = MarketRegimeClassifier(large_price_df, large_index_df)
        result = clf.classify_regime_ml_based(n_regimes=n)
        assert result["regime_cluster"].nunique() == n

    # ── regime changes ────────────────────────────────────────────────────────

    def test_detect_regime_changes_returns_dataframe(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.detect_regime_changes()
        assert isinstance(result, pd.DataFrame)

    def test_detect_regime_changes_has_expected_columns(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.detect_regime_changes()
        for col in ["trade_date", "market_regime", "close_price"]:
            assert col in result.columns, f"Missing column: {col}"

    # ── dashboard data generation ─────────────────────────────────────────────

    def test_generate_regime_dashboard_data_returns_dict(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.generate_regime_dashboard_data()
        assert isinstance(result, dict)

    def test_generate_regime_dashboard_data_has_all_keys(self, price_df, index_df):
        clf = MarketRegimeClassifier(price_df, index_df)
        result = clf.generate_regime_dashboard_data()
        expected_keys = {
            "current_regime", "regime_history", "regime_changes",
            "trend_indicators", "volatility_indicators",
            "momentum_indicators", "breadth_indicators",
        }
        assert expected_keys.issubset(result.keys())


class TestRegimeBasedStrategy:

    def test_get_strategy_returns_dict(self):
        result = RegimeBasedStrategy.get_strategy_for_regime("Strong Uptrend / Low Volatility")
        assert isinstance(result, dict)

    def test_get_strategy_has_required_keys(self):
        result = RegimeBasedStrategy.get_strategy_for_regime("Strong Uptrend / Low Volatility")
        for key in ["action", "instruments", "position_size", "stop_loss", "description"]:
            assert key in result, f"Missing key: {key}"

    def test_get_strategy_instruments_is_list(self):
        result = RegimeBasedStrategy.get_strategy_for_regime("Sideways / Low Volatility")
        assert isinstance(result["instruments"], list)

    def test_get_strategy_unknown_regime_returns_neutral(self):
        result = RegimeBasedStrategy.get_strategy_for_regime("Unknown / Unknown")
        assert result["action"] == "Neutral"

    def test_get_strategy_all_known_regimes_return_non_empty(self):
        known_regimes = [
            "Strong Uptrend / Low Volatility",
            "Strong Uptrend / High Volatility",
            "Strong Downtrend / Low Volatility",
            "Strong Downtrend / High Volatility",
            "Sideways / Low Volatility",
            "Sideways / High Volatility",
        ]
        for regime in known_regimes:
            result = RegimeBasedStrategy.get_strategy_for_regime(regime)
            assert result["action"] != "Neutral", f"Expected non-neutral for {regime}"
