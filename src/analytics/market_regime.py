"""
Market Regime Classification Module
Identifies market conditions: Bull, Bear, Sideways, High/Low Volatility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MarketRegimeClassifier:
    """
    Classify market regimes using multiple indicators
    """

    def __init__(self, price_data: pd.DataFrame, index_data: pd.DataFrame):
        """
        Args:
            price_data: Individual stock price data
            index_data: Market index data (Nifty 50, etc.)
        """
        self.price_data = price_data.copy()
        self.index_data = index_data.copy()
        self._prepare_index_data()

    def _prepare_index_data(self):
        """Prepare index data for analysis"""
        # Filter for main index
        self.nifty = self.index_data[
            self.index_data['security_name'] == 'Nifty 50'
            ].sort_values('trade_date').copy()

        if len(self.nifty) == 0:
            raise ValueError("Nifty 50 data not found")

    def calculate_trend_indicators(self) -> pd.DataFrame:
        """
        Calculate trend indicators: Moving averages, trend strength
        """
        df = self.nifty.copy()

        # Moving averages
        for period in [20, 50, 200]:
            df[f'sma_{period}'] = df['close_price'].rolling(period, min_periods=1).mean()

        # Price position relative to MAs
        df['above_sma20'] = df['close_price'] > df['sma_20']
        df['above_sma50'] = df['close_price'] > df['sma_50']
        df['above_sma200'] = df['close_price'] > df['sma_200']

        # MA slopes (rate of change)
        for period in [20, 50, 200]:
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(5) * 100

        # Trend strength (ADX-like)
        df['price_change'] = df['close_price'].pct_change()
        df['positive_dm'] = np.where(df['price_change'] > 0, abs(df['price_change']), 0)
        df['negative_dm'] = np.where(df['price_change'] < 0, abs(df['price_change']), 0)

        window = 14
        df['di_plus'] = df['positive_dm'].rolling(window).mean()
        df['di_minus'] = df['negative_dm'].rolling(window).mean()
        df['dx'] = abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus']) * 100
        df['adx'] = df['dx'].rolling(window).mean()

        return df

    def calculate_volatility_indicators(self) -> pd.DataFrame:
        """
        Calculate volatility indicators
        """
        df = self.nifty.copy()

        # Historical volatility (20-day)
        df['returns'] = df['close_price'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252) * 100

        # ATR (Average True Range)
        df['high_low'] = df['high_price'] - df['low_price']
        df['high_close'] = abs(df['high_price'] - df['close_price'].shift(1))
        df['low_close'] = abs(df['low_price'] - df['close_price'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close_price']) * 100

        # Bollinger Band width
        df['sma_20'] = df['close_price'].rolling(20).mean()
        df['std_20'] = df['close_price'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (2 * df['std_20'])
        df['bb_lower'] = df['sma_20'] - (2 * df['std_20'])
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['sma_20']) * 100

        return df

    def calculate_momentum_indicators(self) -> pd.DataFrame:
        """
        Calculate momentum indicators
        """
        df = self.nifty.copy()

        # RSI
        delta = df['close_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close_price'].pct_change(period) * 100

        # MACD
        ema_12 = df['close_price'].ewm(span=12).mean()
        ema_26 = df['close_price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def calculate_breadth_indicators(self) -> pd.DataFrame:
        """
        Calculate market breadth indicators
        """
        # Daily advance/decline
        market_data = self.price_data.copy()
        market_data['price_change'] = market_data.groupby('symbol')['close_price'].pct_change()

        breadth = market_data.groupby('trade_date').apply(
            lambda x: pd.Series({
                'advances': (x['price_change'] > 0).sum(),
                'declines': (x['price_change'] < 0).sum(),
                'unchanged': (x['price_change'] == 0).sum(),
                'total_stocks': len(x)
            }), include_groups=False
        ).reset_index()

        # Advance/Decline ratio
        breadth['ad_ratio'] = breadth['advances'] / breadth['declines'].replace(0, 1)
        breadth['advance_pct'] = (breadth['advances'] / breadth['total_stocks']) * 100

        # Advance/Decline line (cumulative)
        breadth['ad_diff'] = breadth['advances'] - breadth['declines']
        breadth['ad_line'] = breadth['ad_diff'].cumsum()

        return breadth

    def classify_regime_rule_based(self) -> pd.DataFrame:
        """
        Classify regime using rule-based approach
        """
        # Get all indicators
        trend = self.calculate_trend_indicators()
        vol = self.calculate_volatility_indicators()
        momentum = self.calculate_momentum_indicators()

        # Merge
        regime_data = trend[['trade_date', 'close_price', 'above_sma20', 'above_sma50',
                             'above_sma200', 'sma_20_slope', 'sma_50_slope', 'adx']].copy()

        regime_data = pd.merge(
            regime_data,
            vol[['trade_date', 'volatility', 'atr_pct']],
            on='trade_date'
        )

        regime_data = pd.merge(
            regime_data,
            momentum[['trade_date', 'rsi', 'macd_hist']],
            on='trade_date'
        )

        # Trend classification
        regime_data['trend'] = 'Sideways'

        # Strong uptrend
        regime_data.loc[
            (regime_data['above_sma20']) &
            (regime_data['above_sma50']) &
            (regime_data['sma_20_slope'] > 0) &
            (regime_data['sma_50_slope'] > 0) &
            (regime_data['adx'] > 25),
            'trend'
        ] = 'Strong Uptrend'

        # Moderate uptrend
        regime_data.loc[
            (regime_data['above_sma20']) &
            (regime_data['sma_20_slope'] > 0) &
            (regime_data['trend'] == 'Sideways'),
            'trend'
        ] = 'Moderate Uptrend'

        # Strong downtrend
        regime_data.loc[
            (~regime_data['above_sma20']) &
            (~regime_data['above_sma50']) &
            (regime_data['sma_20_slope'] < 0) &
            (regime_data['sma_50_slope'] < 0) &
            (regime_data['adx'] > 25),
            'trend'
        ] = 'Strong Downtrend'

        # Moderate downtrend
        regime_data.loc[
            (~regime_data['above_sma20']) &
            (regime_data['sma_20_slope'] < 0) &
            (regime_data['trend'] == 'Sideways'),
            'trend'
        ] = 'Moderate Downtrend'

        # Volatility classification
        vol_percentile = regime_data['volatility'].rank(pct=True) * 100
        regime_data['volatility_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 33, 67, 100],
            labels=['Low Volatility', 'Normal Volatility', 'High Volatility']
        )

        # Overall market regime
        regime_data['market_regime'] = (
                regime_data['trend'] + ' / ' + regime_data['volatility_regime'].astype(str)
        )

        return regime_data

    def classify_regime_ml_based(self, n_regimes: int = 4) -> pd.DataFrame:
        """
        Classify regime using machine learning (K-means clustering)
        Requires scikit-learn: pip install scikit-learn
        """
        # Prevent OpenMP thread-pool deadlock on Windows.
        # Must be set before sklearn imports its C extensions.
        import sys
        if sys.platform == "win32":
            import os
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn is required for ML-based regime classification.\n"
                "Install it with:  pip install scikit-learn"
            )
        # Get all indicators
        trend = self.calculate_trend_indicators()
        vol = self.calculate_volatility_indicators()
        momentum = self.calculate_momentum_indicators()

        # Features for clustering
        features = pd.merge(
            trend[['trade_date', 'sma_20_slope', 'sma_50_slope', 'adx']],
            vol[['trade_date', 'volatility', 'atr_pct']],
            on='trade_date'
        )
        features = pd.merge(
            features,
            momentum[['trade_date', 'rsi', 'roc_20', 'macd_hist']],
            on='trade_date'
        )

        # Drop NaN
        features = features.dropna()

        # Prepare feature matrix
        X = features[['sma_20_slope', 'sma_50_slope', 'volatility',
                      'rsi', 'roc_20', 'adx']].values

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        features['regime_cluster'] = kmeans.fit_predict(X_scaled)

        # Label clusters based on characteristics
        cluster_profiles = features.groupby('regime_cluster').agg({
            'sma_20_slope': 'mean',
            'volatility': 'mean',
            'rsi': 'mean'
        })

        # Create regime labels
        regime_labels = {}
        for cluster in range(n_regimes):
            profile = cluster_profiles.loc[cluster]

            if profile['sma_20_slope'] > 0.5 and profile['volatility'] < 15:
                label = 'Bull Market (Low Vol)'
            elif profile['sma_20_slope'] > 0.5 and profile['volatility'] >= 15:
                label = 'Bull Market (High Vol)'
            elif profile['sma_20_slope'] < -0.5 and profile['volatility'] < 15:
                label = 'Bear Market (Low Vol)'
            elif profile['sma_20_slope'] < -0.5 and profile['volatility'] >= 15:
                label = 'Bear Market (High Vol)'
            else:
                label = 'Sideways Market'

            regime_labels[cluster] = label

        features['regime_label'] = features['regime_cluster'].map(regime_labels)

        return features

    def detect_regime_changes(self) -> pd.DataFrame:
        """
        Detect when market regime changes
        """
        regime_data = self.classify_regime_rule_based()

        # Detect changes
        regime_data['regime_changed'] = (
                regime_data['market_regime'] != regime_data['market_regime'].shift(1)
        )

        # Get change points
        changes = regime_data[regime_data['regime_changed']].copy()
        changes['previous_regime'] = changes['market_regime'].shift(1)

        return changes[['trade_date', 'previous_regime', 'market_regime', 'close_price']]

    def generate_regime_dashboard_data(self) -> Dict:
        """
        Generate all data needed for regime analysis dashboard
        """
        return {
            'current_regime': self.classify_regime_rule_based().iloc[-1],
            'regime_history': self.classify_regime_rule_based(),
            'regime_changes': self.detect_regime_changes(),
            'trend_indicators': self.calculate_trend_indicators(),
            'volatility_indicators': self.calculate_volatility_indicators(),
            'momentum_indicators': self.calculate_momentum_indicators(),
            'breadth_indicators': self.calculate_breadth_indicators()
        }


class RegimeBasedStrategy:
    """
    Strategy recommendations based on market regime
    """

    @staticmethod
    def get_strategy_for_regime(regime: str) -> Dict:
        """
        Get recommended strategy for given regime
        """
        strategies = {
            'Strong Uptrend / Low Volatility': {
                'action': 'Aggressive Long',
                'instruments': ['Momentum stocks', 'Growth stocks', 'Call options'],
                'position_size': 'High (80-100%)',
                'stop_loss': 'Wide (5-8%)',
                'description': 'Best market for aggressive long positions'
            },
            'Strong Uptrend / High Volatility': {
                'action': 'Cautious Long',
                'instruments': ['Quality large caps', 'Covered calls'],
                'position_size': 'Medium (60-70%)',
                'stop_loss': 'Medium (3-5%)',
                'description': 'Trending but volatile - take profits regularly'
            },
            'Strong Downtrend / Low Volatility': {
                'action': 'Short / Defensive',
                'instruments': ['Defensive stocks', 'Put options', 'Gold'],
                'position_size': 'Low (20-30%)',
                'stop_loss': 'Tight (2-3%)',
                'description': 'Preserve capital, minimal long exposure'
            },
            'Strong Downtrend / High Volatility': {
                'action': 'Cash / Hedged',
                'instruments': ['Cash', 'Bonds', 'Gold', 'Put spreads'],
                'position_size': 'Very Low (0-10%)',
                'stop_loss': 'Very Tight (1-2%)',
                'description': 'Worst market - stay in cash or heavily hedged'
            },
            'Sideways / Low Volatility': {
                'action': 'Range Trading',
                'instruments': ['Iron condors', 'Range-bound stocks'],
                'position_size': 'Medium (50-60%)',
                'stop_loss': 'Medium (3-4%)',
                'description': 'Trade ranges, sell options premium'
            },
            'Sideways / High Volatility': {
                'action': 'Volatility Trading',
                'instruments': ['Straddles', 'Strangles', 'Mean reversion'],
                'position_size': 'Low (30-40%)',
                'stop_loss': 'Medium (3-5%)',
                'description': 'Profit from volatility, quick trades'
            }
        }

        return strategies.get(regime, {
            'action': 'Neutral',
            'instruments': ['Balanced portfolio'],
            'position_size': 'Medium (50%)',
            'stop_loss': 'Medium (3-5%)',
            'description': 'Unknown regime - maintain balanced approach'
        })


if __name__ == "__main__":
    print("Market Regime Classification Module")
    print("=" * 50)
    print("\nFeatures:")
    print("1. Trend Indicators (MA, ADX)")
    print("2. Volatility Indicators (HV, ATR, BB Width)")
    print("3. Momentum Indicators (RSI, MACD, ROC)")
    print("4. Breadth Indicators (A/D Line)")
    print("5. Rule-Based Regime Classification")
    print("6. ML-Based Regime Classification")
    print("7. Regime Change Detection")
    print("8. Strategy Recommendations")