"""
52-Week High/Low Breakout Analysis Module
Analyzes breakout patterns and success rates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BreakoutAnalyzer:
    """
    Analyzes 52-week high/low breakouts
    """
    
    def __init__(self, breakout_data: pd.DataFrame, price_data: pd.DataFrame):
        """
        Initialize with breakout and price data
        
        Args:
            breakout_data: DataFrame from HL file (securities hitting 52w high/low)
            price_data: DataFrame with historical price data
        """
        self.breakout_data = breakout_data.copy()
        self.price_data = price_data.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and categorize breakout data"""
        # Merge with price data
        self.merged_data = pd.merge(
            self.breakout_data,
            self.price_data,
            on=['symbol', 'trade_date'],
            how='inner'
        )
    
    def calculate_breakout_strength(self) -> pd.DataFrame:
        """
        Calculate strength of breakout
        - Volume on breakout day
        - Price above/below 52-week level
        - Consolidation period before breakout
        """
        df = self.merged_data.copy()
        
        # Calculate how far above 52-week high (or below 52-week low)
        df['breakout_margin'] = np.where(
            df['hl_type'] == 'H',
            ((df['close_price'] - df['high_52_week']) / df['high_52_week']) * 100,
            ((df['low_52_week'] - df['close_price']) / df['low_52_week']) * 100
        )
        
        # Volume relative to average
        avg_volume = self.price_data.groupby('symbol')['net_traded_qty'].mean()
        df['avg_volume'] = df['symbol'].map(avg_volume)
        df['volume_ratio'] = df['net_traded_qty'] / df['avg_volume']
        
        # Classify breakout strength
        df['strength'] = pd.cut(
            df['volume_ratio'],
            bins=[0, 1, 2, 3, np.inf],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        return df[['symbol', 'trade_date', 'hl_type', 'breakout_margin',
                  'volume_ratio', 'strength']]
    
    def analyze_breakout_success_rate(self, 
                                     forward_days: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Analyze success rate of breakouts
        What % of 52w high breakouts continue higher?
        """
        df = self.merged_data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        results = []
        
        for idx, row in df.iterrows():
            symbol = row['symbol']
            breakout_date = row['trade_date']
            breakout_type = row['hl_type']  # 'H' or 'L'
            breakout_price = row['close_price']
            
            # Get future prices
            future_prices = self.price_data[
                (self.price_data['symbol'] == symbol) &
                (self.price_data['trade_date'] > breakout_date)
            ].sort_values('trade_date').head(max(forward_days))
            
            if len(future_prices) < max(forward_days):
                continue
            
            result = {
                'symbol': symbol,
                'breakout_date': breakout_date,
                'breakout_type': '52W High' if breakout_type == 'H' else '52W Low',
                'breakout_price': breakout_price
            }
            
            # Calculate returns and success for each time period
            for days in forward_days:
                if len(future_prices) >= days:
                    future_price = future_prices.iloc[days - 1]['close_price']
                    returns = ((future_price - breakout_price) / breakout_price) * 100
                    
                    # Success = price continued in breakout direction
                    if breakout_type == 'H':
                        success = future_price > breakout_price
                    else:  # 'L'
                        success = future_price < breakout_price
                    
                    result[f'return_{days}d'] = returns
                    result[f'success_{days}d'] = success
                    
                    # Max favorable and adverse excursion
                    period_data = future_prices.head(days)
                    if breakout_type == 'H':
                        result[f'max_gain_{days}d'] = ((period_data['high_price'].max() - breakout_price) / breakout_price) * 100
                        result[f'max_loss_{days}d'] = ((period_data['low_price'].min() - breakout_price) / breakout_price) * 100
                    else:
                        result[f'max_gain_{days}d'] = ((breakout_price - period_data['low_price'].min()) / breakout_price) * 100
                        result[f'max_loss_{days}d'] = ((breakout_price - period_data['high_price'].max()) / breakout_price) * 100
            
            results.append(result)
        
        # Ensure consistent columns even when results is empty
        result_cols = ['symbol', 'breakout_date', 'breakout_type', 'breakout_price']
        for days in forward_days:
            result_cols += [f'return_{days}d', f'success_{days}d',
                            f'max_gain_{days}d', f'max_loss_{days}d']
        results_df = pd.DataFrame(results, columns=result_cols) if results else pd.DataFrame(columns=result_cols)

        # Calculate summary statistics
        summary = []
        for breakout_type in ['52W High', '52W Low']:
            type_data = results_df[results_df['breakout_type'] == breakout_type]
            
            for days in forward_days:
                summary.append({
                    'breakout_type': breakout_type,
                    'period_days': days,
                    'sample_size': len(type_data),
                    'success_rate': type_data[f'success_{days}d'].mean() * 100,
                    'avg_return': type_data[f'return_{days}d'].mean(),
                    'median_return': type_data[f'return_{days}d'].median(),
                    'avg_max_gain': type_data[f'max_gain_{days}d'].mean(),
                    'avg_max_loss': type_data[f'max_loss_{days}d'].mean()
                })
        
        return pd.DataFrame(summary), results_df
    
    def identify_false_breakouts(self, 
                                reversal_threshold: float = -5.0,
                                reversal_days: int = 5) -> pd.DataFrame:
        """
        Identify false breakouts
        Breakouts that reversed within X days
        """
        _, detailed_results = self.analyze_breakout_success_rate([reversal_days])
        
        # False breakout = reversed by threshold% within reversal_days
        false_breakouts = detailed_results[
            ((detailed_results['breakout_type'] == '52W High') & 
             (detailed_results[f'return_{reversal_days}d'] < reversal_threshold)) |
            ((detailed_results['breakout_type'] == '52W Low') & 
             (detailed_results[f'return_{reversal_days}d'] > -reversal_threshold))
        ].copy()
        
        return false_breakouts[['symbol', 'breakout_date', 'breakout_type',
                               f'return_{reversal_days}d', f'max_loss_{reversal_days}d']]
    
    def analyze_consolidation_before_breakout(self, 
                                             lookback_days: int = 20) -> pd.DataFrame:
        """
        Analyze price consolidation period before breakout
        Longer consolidation = stronger breakout?
        """
        df = self.merged_data.copy()
        
        results = []
        
        for idx, row in df.iterrows():
            symbol = row['symbol']
            breakout_date = row['trade_date']
            
            # Get pre-breakout prices
            pre_breakout = self.price_data[
                (self.price_data['symbol'] == symbol) &
                (self.price_data['trade_date'] < breakout_date)
            ].sort_values('trade_date').tail(lookback_days)
            
            if len(pre_breakout) < lookback_days:
                continue
            
            # Calculate consolidation metrics
            price_range = (pre_breakout['high_price'].max() - pre_breakout['low_price'].min())
            avg_price = pre_breakout['close_price'].mean()
            consolidation_tightness = (price_range / avg_price) * 100
            
            # Volume trend during consolidation
            volume_trend = np.polyfit(
                range(len(pre_breakout)), 
                pre_breakout['net_traded_qty'].values, 
                1
            )[0]
            
            results.append({
                'symbol': symbol,
                'breakout_date': breakout_date,
                'breakout_type': row['hl_type'],
                'consolidation_days': len(pre_breakout),
                'consolidation_tightness_pct': consolidation_tightness,
                'volume_trend': 'Increasing' if volume_trend > 0 else 'Decreasing',
                'avg_volume_pre': pre_breakout['net_traded_qty'].mean()
            })

        _consolidation_cols = ['symbol', 'breakout_date', 'breakout_type',
                               'consolidation_days', 'consolidation_tightness_pct',
                               'volume_trend', 'avg_volume_pre']
        return (pd.DataFrame(results, columns=_consolidation_cols)
                if results else pd.DataFrame(columns=_consolidation_cols))
    
    def identify_breakout_clusters(self, 
                                  date_window: int = 5) -> pd.DataFrame:
        """
        Identify when multiple stocks break out together
        Indicates sector/market momentum
        """
        df = self.breakout_data.copy()
        
        # Count breakouts by date and type
        breakout_counts = df.groupby(['trade_date', 'hl_type']).size().reset_index(name='count')
        
        # Identify high-count days
        breakout_counts['is_cluster'] = breakout_counts['count'] > breakout_counts['count'].quantile(0.75)
        
        # Get symbols for cluster days
        clusters = []
        for idx, row in breakout_counts[breakout_counts['is_cluster']].iterrows():
            symbols = df[
                (df['trade_date'] == row['trade_date']) &
                (df['hl_type'] == row['hl_type'])
            ]['symbol'].tolist()

            clusters.append({
                'date': row['trade_date'],
                'type': '52W High' if row['hl_type'] == 'H' else '52W Low',
                'count': row['count'],
                'symbols': ', '.join(symbols[:10])  # First 10 symbols
            })
        
        return pd.DataFrame(clusters)
    
    def calculate_breakout_momentum_score(self) -> pd.DataFrame:
        """
        Calculate a composite momentum score for breakouts
        Combines multiple factors:
        - Volume
        - Price strength
        - Consolidation quality
        """
        strength = self.calculate_breakout_strength()
        # rename trade_date → breakout_date and hl_type → breakout_type to match consolidation
        strength = strength.rename(columns={'trade_date': 'breakout_date', 'hl_type': 'breakout_type'})
        consolidation = self.analyze_consolidation_before_breakout()
        # consolidation also carries breakout_type; drop it before merge to avoid _x/_y conflict
        consolidation = consolidation.drop(columns=['breakout_type'], errors='ignore')

        # Merge
        momentum = pd.merge(
            strength,
            consolidation,
            on=['symbol', 'breakout_date'],
            how='inner'
        )
        
        # Score components (0-100 each)
        momentum['volume_score'] = np.clip(momentum['volume_ratio'] / 5 * 100, 0, 100)
        momentum['tightness_score'] = np.clip((1 - momentum['consolidation_tightness_pct'] / 20) * 100, 0, 100)
        momentum['margin_score'] = np.clip(abs(momentum['breakout_margin']) / 2 * 100, 0, 100)
        
        # Composite score
        momentum['momentum_score'] = (
            momentum['volume_score'] * 0.4 +
            momentum['tightness_score'] * 0.3 +
            momentum['margin_score'] * 0.3
        )
        
        # Rank
        momentum['momentum_rank'] = momentum['momentum_score'].rank(pct=True) * 100
        
        return momentum[['symbol', 'breakout_date', 'breakout_type', 
                        'momentum_score', 'momentum_rank']]


class BreakoutScreener:
    """
    Screen for potential breakout candidates
    """
    
    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data.copy()
    
    def screen_near_52w_high(self, threshold_pct: float = 5.0) -> pd.DataFrame:
        """
        Find stocks near 52-week high
        """
        df = self.data.copy()
        
        # Calculate distance from 52-week high
        df['distance_from_52w_high'] = ((df['high_52_week'] - df['close_price']) / 
                                        df['close_price']) * 100
        
        # Filter
        near_high = df[df['distance_from_52w_high'] <= threshold_pct].copy()
        near_high = near_high.sort_values('distance_from_52w_high')
        
        return near_high[['symbol', 'security_name', 'close_price', 
                         'high_52_week', 'distance_from_52w_high']]
    
    def screen_near_52w_low(self, threshold_pct: float = 5.0) -> pd.DataFrame:
        """
        Find stocks near 52-week low
        """
        df = self.data.copy()
        
        # Calculate distance from 52-week low
        df['distance_from_52w_low'] = ((df['close_price'] - df['low_52_week']) / 
                                       df['close_price']) * 100
        
        # Filter
        near_low = df[df['distance_from_52w_low'] <= threshold_pct].copy()
        near_low = near_low.sort_values('distance_from_52w_low')
        
        return near_low[['symbol', 'security_name', 'close_price',
                        'low_52_week', 'distance_from_52w_low']]


if __name__ == "__main__":
    print("52-Week Breakout Analysis Module")
    print("=" * 50)
    print("\nFeatures:")
    print("1. Breakout Strength Calculation")
    print("2. Success Rate Analysis")
    print("3. False Breakout Detection")
    print("4. Consolidation Pattern Analysis")
    print("5. Breakout Clustering")
    print("6. Momentum Score Calculation")
    print("7. Breakout Screening")