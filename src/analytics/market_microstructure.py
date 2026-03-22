"""
Market Microstructure - Liquidity Analysis
===========================================
Calculates real liquidity scores based on:
- Delivery percentage (institutional vs retail activity)
- Trading volume
- Bid-ask spread proxy
- Price impact

The delivery percentage is KEY:
- High delivery % = Long-term investors (more liquid, stable)
- Low delivery % = Intraday traders (less liquid, volatile)
"""

import pandas as pd
import numpy as np


def calculate_liquidity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate liquidity scores from 0-100 based on multiple factors.

    Args:
        df: DataFrame with columns:
            - symbol
            - net_traded_qty (volume)
            - net_traded_value (turnover)
            - delivery_pct (CRITICAL - from full bhavcopy)
            - close_price

    Returns:
        DataFrame with symbol and liquidity_score
    """
    if df.empty:
        return pd.DataFrame(columns=['symbol', 'liquidity_score'])

    result = df[['symbol']].copy()

    # Component 1: Delivery Percentage (40% weight)
    # Higher delivery = more institutional = more liquid
    # Scale: 0-100
    if 'delivery_pct' in df.columns and df['delivery_pct'].notna().any():
        # Normalize delivery % to 0-100 scale
        delivery_score = df['delivery_pct'].fillna(0).clip(0, 100)
    else:
        # Fallback if no delivery data
        delivery_score = pd.Series(50, index=df.index)  # Neutral

    # Component 2: Volume Score (30% weight)
    # Higher volume = more liquid
    if 'net_traded_qty' in df.columns and df['net_traded_qty'].notna().any():
        volume = df['net_traded_qty'].fillna(0)
        # Log scale to handle wide range
        volume_log = np.log1p(volume)
        # Normalize to 0-100
        volume_score = (volume_log / volume_log.max() * 100) if volume_log.max() > 0 else pd.Series(0, index=df.index)
    else:
        volume_score = pd.Series(0, index=df.index)

    # Component 3: Turnover Score (20% weight)
    # Higher value traded = more liquid
    if 'net_traded_value' in df.columns and df['net_traded_value'].notna().any():
        turnover = df['net_traded_value'].fillna(0)
        turnover_log = np.log1p(turnover)
        turnover_score = (turnover_log / turnover_log.max() * 100) if turnover_log.max() > 0 else pd.Series(0,
                                                                                                            index=df.index)
    else:
        turnover_score = pd.Series(0, index=df.index)

    # Component 4: Price Stability (10% weight)
    # Lower volatility = more liquid (stable)
    if 'close_price' in df.columns and 'net_traded_value' in df.columns:
        # Price impact proxy: lower is better
        price_impact = (df['close_price'] / (df['net_traded_value'] + 1)) * 1000000
        # Invert and normalize
        stability_score = 100 - (price_impact / price_impact.max() * 100) if price_impact.max() > 0 else pd.Series(50,
                                                                                                                   index=df.index)
        stability_score = stability_score.fillna(50).clip(0, 100)
    else:
        stability_score = pd.Series(50, index=df.index)

    # Weighted combination
    liquidity_score = (
            delivery_score * 0.40 +  # 40% weight on delivery
            volume_score * 0.30 +  # 30% weight on volume
            turnover_score * 0.20 +  # 20% weight on turnover
            stability_score * 0.10  # 10% weight on stability
    )

    result['liquidity_score'] = liquidity_score.clip(0, 100).round(1)

    return result


def classify_liquidity(score: float) -> str:
    """Classify liquidity score into categories."""
    if score >= 80:
        return "HIGHLY_LIQUID"
    elif score >= 60:
        return "LIQUID"
    elif score >= 40:
        return "MODERATE"
    elif score >= 20:
        return "ILLIQUID"
    else:
        return "HIGHLY_ILLIQUID"


def identify_volume_clusters(df: pd.DataFrame, threshold_pct: float = 2.0) -> pd.DataFrame:
    """
    Identify stocks with volume spikes (clusters).

    Args:
        df: DataFrame with net_traded_qty and average volume
        threshold_pct: Spike threshold (e.g., 2.0 = 200% of average)

    Returns:
        DataFrame of stocks with volume spikes
    """
    # This would need historical data - simplified for now
    if df.empty or 'net_traded_qty' not in df.columns:
        return pd.DataFrame()

    # Use current volume vs median as proxy
    median_vol = df['net_traded_qty'].median()

    spikes = df[df['net_traded_qty'] > median_vol * threshold_pct].copy()
    spikes['volume_ratio'] = spikes['net_traded_qty'] / median_vol

    return spikes[['symbol', 'net_traded_qty', 'volume_ratio']].sort_values('volume_ratio', ascending=False)


def calculate_overnight_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overnight gap percentages.

    Args:
        df: DataFrame with open_price and prev_close

    Returns:
        DataFrame with gap percentages
    """
    if df.empty or 'open_price' not in df.columns or 'prev_close' not in df.columns:
        return pd.DataFrame()

    result = df[['symbol', 'open_price', 'prev_close']].copy()

    result['gap_pct'] = ((result['open_price'] - result['prev_close']) / result['prev_close'] * 100).round(2)

    # Filter significant gaps (>1%)
    significant = result[abs(result['gap_pct']) > 1.0].copy()
    significant['gap_direction'] = significant['gap_pct'].apply(lambda x: 'GAP_UP' if x > 0 else 'GAP_DOWN')

    return significant.sort_values('gap_pct', ascending=False, key=abs)


class MarketMicrostructureAnalyzer:
    """
    DataFrame-based analyzer for market microstructure patterns.
    Accepts a price DataFrame directly (no DB required).
    """

    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data.copy()

    def calculate_overnight_gap(self) -> pd.DataFrame:
        """Significant overnight gaps between prev_close and open."""
        return calculate_overnight_gaps(self.data)

    def analyze_intraday_price_discovery(self) -> pd.DataFrame:
        """Intraday open-to-close price movement as price-discovery proxy."""
        df = self.data.copy()
        if 'open_price' not in df.columns or 'close_price' not in df.columns:
            return pd.DataFrame()
        result = df[['symbol']].copy()
        result['open_to_close_pct'] = (
            (df['close_price'] - df['open_price']) / df['open_price'].replace(0, np.nan) * 100
        ).round(2)
        result['price_discovery_direction'] = result['open_to_close_pct'].apply(
            lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Flat')
        )
        return result.dropna(subset=['open_to_close_pct'])

    def calculate_liquidity_metrics(self) -> pd.DataFrame:
        """Full liquidity scoring with classification."""
        scores = calculate_liquidity_scores(self.data)
        scores['liquidity_class'] = scores['liquidity_score'].apply(classify_liquidity)
        return scores

    def detect_volume_clusters(self, threshold_pct: float = 2.0) -> pd.DataFrame:
        """Stocks with volume spikes above threshold × median."""
        return identify_volume_clusters(self.data, threshold_pct)

    def analyze_price_volume_relationship(self) -> pd.DataFrame:
        """Correlation between price change and volume."""
        df = self.data.copy()
        if 'close_price' not in df.columns or 'net_traded_qty' not in df.columns:
            return pd.DataFrame()
        result = df[['symbol', 'close_price', 'net_traded_qty']].copy()
        if 'prev_close' in df.columns:
            result['price_change_pct'] = (
                (df['close_price'] - df['prev_close']) / df['prev_close'].replace(0, np.nan) * 100
            ).round(2)
        result['volume_category'] = pd.qcut(
            df['net_traded_qty'].rank(method='first'), q=4,
            labels=['Low', 'Below Avg', 'Above Avg', 'High']
        )
        return result.dropna()

    def calculate_market_depth_proxy(self) -> pd.DataFrame:
        """Amihud illiquidity proxy: |return| / volume."""
        df = self.data.copy()
        if 'close_price' not in df.columns or 'net_traded_qty' not in df.columns:
            return pd.DataFrame()
        result = df[['symbol', 'close_price', 'net_traded_qty']].copy()
        if 'prev_close' in df.columns:
            abs_return = ((df['close_price'] - df['prev_close']).abs()
                          / df['prev_close'].replace(0, np.nan))
        else:
            abs_return = pd.Series(np.nan, index=df.index)
        result['amihud_illiquidity'] = (
            abs_return / df['net_traded_qty'].replace(0, np.nan) * 1e6
        ).round(6)
        result['depth_class'] = pd.qcut(
            result['amihud_illiquidity'].rank(method='first'), q=3,
            labels=['Deep', 'Moderate', 'Shallow']
        )
        return result.dropna(subset=['amihud_illiquidity'])

    def analyze_delivery_quality(self) -> pd.DataFrame:
        """
        Classify stocks by delivery % vs price change to surface
        institutional accumulation / distribution signals.

        Signal logic:
          High delivery + price up   → Institutional Accumulation
          High delivery + price down → Institutional Distribution
          Low delivery  + price up   → Speculative Rally (fragile)
          Low delivery  + price down → Speculative Selloff
        """
        df = self.data.copy()
        if 'delivery_pct' not in df.columns or df['delivery_pct'].isna().all():
            return pd.DataFrame()

        result = df[['symbol']].copy()
        result['delivery_pct'] = df['delivery_pct'].fillna(0)

        if 'close_price' in df.columns and 'prev_close' in df.columns:
            result['price_change_pct'] = (
                (df['close_price'] - df['prev_close'])
                / df['prev_close'].replace(0, np.nan) * 100
            ).round(2)
        else:
            result['price_change_pct'] = np.nan

        if 'net_traded_value' in df.columns:
            result['net_traded_value'] = df['net_traded_value']
        if 'close_price' in df.columns:
            result['close_price'] = df['close_price']

        def _signal(row):
            dp = row['delivery_pct']
            pc = row.get('price_change_pct', 0) or 0
            if dp >= 60 and pc > 0:
                return 'Institutional Accumulation'
            elif dp >= 60 and pc < 0:
                return 'Institutional Distribution'
            elif dp < 30 and pc > 2:
                return 'Speculative Rally'
            elif dp < 30 and pc < -2:
                return 'Speculative Selloff'
            else:
                return 'Mixed / Neutral'

        result['delivery_signal'] = result.apply(_signal, axis=1)
        return result.dropna(subset=['delivery_pct']).sort_values('delivery_pct', ascending=False)

    def analyze_candle_structure(self) -> pd.DataFrame:
        """
        Decompose each candle into body, upper wick, lower wick as % of range.
        Classifies candle type useful for reading intraday sentiment.

        Returns columns: symbol, range_pct, body_pct, upper_wick_pct,
                         lower_wick_pct, body_direction, candle_type
        """
        df = self.data.copy()
        req = ['open_price', 'high_price', 'low_price', 'close_price']
        if not all(c in df.columns for c in req):
            return pd.DataFrame()

        result = df[['symbol']].copy()
        close = df['close_price'].replace(0, np.nan)
        hi_lo = (df['high_price'] - df['low_price']).replace(0, np.nan)

        result['range_pct']      = (hi_lo / close * 100).round(2)
        result['body_pct']       = (abs(df['close_price'] - df['open_price']) / close * 100).round(2)
        candle_top = df[['open_price', 'close_price']].max(axis=1)
        candle_bot = df[['open_price', 'close_price']].min(axis=1)
        result['upper_wick_pct'] = ((df['high_price'] - candle_top) / close * 100).round(2)
        result['lower_wick_pct'] = ((candle_bot - df['low_price'])  / close * 100).round(2)
        result['body_direction'] = (df['close_price'] >= df['open_price']).map(
            {True: 'Bullish', False: 'Bearish'}
        )

        if 'net_traded_value' in df.columns:
            result['net_traded_value'] = df['net_traded_value']

        def _candle_type(row):
            rng  = row['range_pct']
            body = row['body_pct']
            uw   = row['upper_wick_pct']
            lw   = row['lower_wick_pct']
            if rng < 0.3:
                return 'Doji'
            body_ratio = body / rng if rng else 0
            if body_ratio > 0.7:
                return 'Marubozu' if row['body_direction'] == 'Bullish' else 'Bearish Marubozu'
            if lw > body * 2 and uw < body * 0.5:
                return 'Hammer / Dragonfly'
            if uw > body * 2 and lw < body * 0.5:
                return 'Shooting Star / Gravestone'
            if body_ratio < 0.3:
                return 'Spinning Top'
            return ('Normal Bullish' if row['body_direction'] == 'Bullish'
                    else 'Normal Bearish')

        result['candle_type'] = result.apply(_candle_type, axis=1)
        return result.dropna(subset=['range_pct'])

    def calculate_price_position(self) -> pd.DataFrame:
        """
        Where did close land within the day's high-low range?
          price_position = (close - low) / (high - low) × 100
          100 = closed at day high (max bullish), 0 = closed at day low.

        Combines position with volume to produce a conviction label.
        """
        df = self.data.copy()
        if not all(c in df.columns for c in ['high_price', 'low_price', 'close_price']):
            return pd.DataFrame()

        result = df[['symbol']].copy()
        hl_range = (df['high_price'] - df['low_price']).replace(0, np.nan)
        result['price_position'] = (
            (df['close_price'] - df['low_price']) / hl_range * 100
        ).round(1)

        if 'prev_close' in df.columns:
            result['price_change_pct'] = (
                (df['close_price'] - df['prev_close'])
                / df['prev_close'].replace(0, np.nan) * 100
            ).round(2)
        if 'close_price' in df.columns:
            result['close_price'] = df['close_price']

        if 'net_traded_qty' in df.columns:
            result['net_traded_qty'] = df['net_traded_qty']
            median_vol = df['net_traded_qty'].median()
            vol_above  = df['net_traded_qty'] > median_vol
        else:
            vol_above = pd.Series(False, index=df.index)

        def _conviction(pos, va):
            if pos >= 70 and va:
                return 'Strong Bullish'
            elif pos >= 70:
                return 'Weak Bullish'
            elif pos <= 30 and va:
                return 'Strong Bearish'
            elif pos <= 30:
                return 'Weak Bearish'
            else:
                return 'Neutral'

        result['conviction'] = [
            _conviction(p, v)
            for p, v in zip(result['price_position'], vol_above)
        ]
        return result.dropna(subset=['price_position'])

    def classify_momentum_volume_quadrant(self) -> pd.DataFrame:
        """
        Four-quadrant classification for strategy selection:
          Q1 (+price, high vol) — Confirmed Momentum (trend-follow)
          Q2 (+price, low  vol) — Suspect Rally / Low Conviction
          Q3 (-price, high vol) — Distribution / Institutional Selling
          Q4 (-price, low  vol) — Low-Conviction Pullback / Potential Base
        """
        df = self.data.copy()
        if not all(c in df.columns for c in ['close_price', 'prev_close', 'net_traded_qty']):
            return pd.DataFrame()

        result = df[['symbol']].copy()
        result['price_change_pct'] = (
            (df['close_price'] - df['prev_close'])
            / df['prev_close'].replace(0, np.nan) * 100
        ).round(2)
        result['net_traded_qty']   = df['net_traded_qty']
        if 'net_traded_value' in df.columns:
            result['net_traded_value'] = df['net_traded_value']
        if 'close_price' in df.columns:
            result['close_price'] = df['close_price']

        median_vol = df['net_traded_qty'].median()
        vol_above  = df['net_traded_qty'] > median_vol
        price_pos  = result['price_change_pct'] > 0

        quad_map = {
            (True,  True):  'Q1: Confirmed Momentum',
            (True,  False): 'Q2: Suspect Rally',
            (False, True):  'Q3: Distribution',
            (False, False): 'Q4: Low-Conviction Pullback',
        }
        result['quadrant'] = [
            quad_map[(pp, va)]
            for pp, va in zip(price_pos, vol_above)
        ]
        result['volume_vs_median'] = (df['net_traded_qty'] / median_vol).round(2)
        return result.dropna(subset=['price_change_pct'])

    def analyze_volatility(self) -> pd.DataFrame:
        """
        Single-day per-stock volatility decomposition.

        Columns returned:
          symbol, intraday_range_pct, true_range_pct, parkinson_vol,
          price_change_pct, avg_trade_size, volatility_class, net_traded_value, close_price

        Notes:
          - intraday_range_pct = (H - L) / prev_close × 100  (normalised range)
          - true_range_pct     = max(H-L, |H-PC|, |L-PC|) / prev_close × 100
          - parkinson_vol      = sqrt(1/(4·ln2)) · ln(H/L) × 100
          - avg_trade_size     = net_traded_value / total_trades  (institutional proxy)
          - volatility_class   = Low Vol / Mid Vol / High Vol  (today's terciles)
        """
        df = self.data.copy()
        if not all(c in df.columns for c in ['high_price', 'low_price', 'close_price']):
            return pd.DataFrame()

        result = df[['symbol']].copy()
        close  = df['close_price'].replace(0, np.nan)
        base   = df['prev_close'].replace(0, np.nan) if 'prev_close' in df.columns else close

        result['intraday_range_pct'] = (
            (df['high_price'] - df['low_price']) / base * 100
        ).round(2)

        if 'prev_close' in df.columns:
            pc = df['prev_close'].replace(0, np.nan)
            tr = pd.concat([
                (df['high_price'] - df['low_price']).abs(),
                (df['high_price'] - pc).abs(),
                (df['low_price']  - pc).abs(),
            ], axis=1).max(axis=1)
            result['true_range_pct']   = (tr / pc * 100).round(2)
            result['price_change_pct'] = (
                (df['close_price'] - pc) / pc * 100
            ).round(2)
        else:
            result['true_range_pct']   = result['intraday_range_pct']
            result['price_change_pct'] = np.nan

        # Parkinson estimator — H/L ratio floored at 1 to avoid log(0)
        hl_ratio = (df['high_price'] / df['low_price'].replace(0, np.nan)).clip(lower=1.0)
        result['parkinson_vol'] = (
            np.sqrt(1.0 / (4.0 * np.log(2))) * np.log(hl_ratio) * 100
        ).round(3)

        # Average trade size as institutional proxy (large = block trades)
        if 'net_traded_value' in df.columns and 'total_trades' in df.columns:
            result['avg_trade_size'] = (
                df['net_traded_value'] / df['total_trades'].replace(0, np.nan)
            ).round(0)

        # Volatility class from today's terciles
        valid_range = result['intraday_range_pct'].dropna()
        if len(valid_range) >= 6:
            q33, q67 = valid_range.quantile([0.33, 0.67]).values
            result['volatility_class'] = pd.cut(
                result['intraday_range_pct'],
                bins=[-np.inf, q33, q67, np.inf],
                labels=['Low Vol', 'Mid Vol', 'High Vol'],
            ).astype(str)
        else:
            result['volatility_class'] = 'Mid Vol'

        for col in ('net_traded_value', 'close_price'):
            if col in df.columns:
                result[col] = df[col]

        return result.dropna(subset=['intraday_range_pct'])


if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'SMALLCAP'],
        'net_traded_qty': [1000000, 500000, 800000, 300000, 10000],
        'net_traded_value': [100000000, 80000000, 90000000, 50000000, 500000],
        'delivery_pct': [75.5, 65.2, 70.1, 55.0, 25.3],
        'close_price': [2500, 3200, 1400, 1600, 45],
    })

    scores = calculate_liquidity_scores(sample_data)
    print("\nLiquidity Scores (with delivery data):")
    print(scores)
    print(f"\nAverage score: {scores['liquidity_score'].mean():.1f}")
    print(f"This should NOT be 50.0 if using real data!")