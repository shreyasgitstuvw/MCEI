"""Unit tests for MarketMicrostructureAnalyzer."""
import pandas as pd
import pytest
from src.analytics.market_microstructure import MarketMicrostructureAnalyzer


class TestMarketMicrostructureAnalyzer:

    def test_init(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        assert len(analyzer.data) == len(price_df)

    def test_calculate_overnight_gap_returns_dataframe(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.calculate_overnight_gap()
        assert isinstance(result, pd.DataFrame)
        assert "gap_pct" in result.columns

    def test_analyze_intraday_price_discovery_returns_dataframe(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_intraday_price_discovery()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_liquidity_metrics_returns_dataframe(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.calculate_liquidity_metrics()
        assert isinstance(result, pd.DataFrame)
        # Should include a spread proxy or turnover
        assert not result.empty

    def test_detect_volume_clusters_returns_dataframe(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.detect_volume_clusters(threshold_pct=1.0)
        assert isinstance(result, pd.DataFrame)

    def test_analyze_price_volume_relationship_returns_dataframe(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_price_volume_relationship()
        assert isinstance(result, pd.DataFrame)

    def test_calculate_market_depth_proxy_returns_dataframe(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.calculate_market_depth_proxy()
        assert isinstance(result, pd.DataFrame)

    def test_single_symbol(self, price_df):
        single = price_df[price_df["symbol"] == "RELIANCE"].copy()
        analyzer = MarketMicrostructureAnalyzer(single)
        result = analyzer.calculate_overnight_gap()
        assert isinstance(result, pd.DataFrame)

    # ── new method tests ──────────────────────────────────────────────────────

    def test_analyze_delivery_quality_no_column_returns_empty(self, price_df):
        df = price_df.drop(columns=["delivery_pct"], errors="ignore")
        analyzer = MarketMicrostructureAnalyzer(df)
        result = analyzer.analyze_delivery_quality()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_analyze_delivery_quality_with_data(self, price_df):
        import numpy as np
        df = price_df.copy()
        df["delivery_pct"] = np.linspace(10, 90, len(df))
        analyzer = MarketMicrostructureAnalyzer(df)
        result = analyzer.analyze_delivery_quality()
        assert isinstance(result, pd.DataFrame)
        assert "delivery_signal" in result.columns
        assert set(result["delivery_signal"]).issubset({
            "Institutional Accumulation", "Institutional Distribution",
            "Speculative Rally", "Speculative Selloff", "Mixed / Neutral",
        })

    def test_analyze_candle_structure_returns_expected_columns(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_candle_structure()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        for col in ["range_pct", "body_pct", "upper_wick_pct",
                    "lower_wick_pct", "body_direction", "candle_type"]:
            assert col in result.columns

    def test_analyze_candle_structure_body_le_range(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_candle_structure()
        # body cannot exceed range
        assert (result["body_pct"] <= result["range_pct"] + 0.01).all()

    def test_calculate_price_position_range_0_100(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.calculate_price_position()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "price_position" in result.columns
        assert (result["price_position"].between(0, 100, inclusive="both")).all()

    def test_calculate_price_position_conviction_labels(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.calculate_price_position()
        valid = {"Strong Bullish", "Weak Bullish", "Strong Bearish",
                 "Weak Bearish", "Neutral"}
        assert set(result["conviction"]).issubset(valid)

    def test_classify_momentum_volume_quadrant_returns_four_labels(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.classify_momentum_volume_quadrant()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "quadrant" in result.columns
        valid = {
            "Q1: Confirmed Momentum", "Q2: Suspect Rally",
            "Q3: Distribution",       "Q4: Low-Conviction Pullback",
        }
        assert set(result["quadrant"]).issubset(valid)

    def test_classify_momentum_volume_quadrant_volume_ratio_positive(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.classify_momentum_volume_quadrant()
        assert (result["volume_vs_median"] >= 0).all()

    def test_analyze_volatility_returns_expected_columns(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_volatility()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        for col in ["intraday_range_pct", "true_range_pct", "parkinson_vol",
                    "price_change_pct", "volatility_class"]:
            assert col in result.columns

    def test_analyze_volatility_range_non_negative(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_volatility()
        assert (result["intraday_range_pct"] >= 0).all()
        assert (result["parkinson_vol"] >= 0).all()

    def test_analyze_volatility_class_labels(self, price_df):
        analyzer = MarketMicrostructureAnalyzer(price_df)
        result = analyzer.analyze_volatility()
        assert set(result["volatility_class"]).issubset({"Low Vol", "Mid Vol", "High Vol"})

    def test_analyze_volatility_missing_ohlc_returns_empty(self):
        import pandas as pd
        df = pd.DataFrame({"symbol": ["A"], "net_traded_qty": [100]})
        analyzer = MarketMicrostructureAnalyzer(df)
        result = analyzer.analyze_volatility()
        assert result.empty
