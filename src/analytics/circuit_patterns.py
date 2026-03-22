"""
Circuit Breaker & Volume Pattern Detection Module
Analyzes price band hits and unusual trading patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class CircuitBreakerAnalyzer:
    """
    Analyzes circuit breaker hits and associated volume patterns
    """
    
    def __init__(self, circuit_data: pd.DataFrame, price_data: pd.DataFrame):
        """
        Initialize with circuit breaker and price data
        
        Args:
            circuit_data: DataFrame from BH file (SYMBOL, SERIES, SECURITY, HIGH/LOW)
            price_data: DataFrame from PR/PD file with price and volume data
        """
        self.circuit_data = circuit_data.copy()
        self.price_data = price_data.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and merge circuit and price data"""
        # Add trade_date to circuit data if not present
        if 'trade_date' not in self.circuit_data.columns:
            self.circuit_data['trade_date'] = pd.to_datetime('today').date()
        
        # Merge with price data
        self.merged_data = pd.merge(
            self.circuit_data,
            self.price_data,
            on=['symbol', 'trade_date'],
            how='inner'
        )
    
    def analyze_circuit_patterns(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Analyze patterns of securities hitting circuits
        """
        df = self.merged_data.copy()
        
        # Count circuit hits by symbol
        circuit_frequency = df.groupby('symbol').agg({
            'trade_date': 'count',
            'high_low': lambda x: (x == 'H').sum()  # Count upper circuit hits
        }).rename(columns={'trade_date': 'total_circuits', 'high_low': 'upper_circuits'})
        
        circuit_frequency['lower_circuits'] = (
            circuit_frequency['total_circuits'] - circuit_frequency['upper_circuits']
        )
        
        # Calculate circuit hit ratio
        circuit_frequency['upper_circuit_ratio'] = (
            circuit_frequency['upper_circuits'] / circuit_frequency['total_circuits']
        )
        
        # Classify circuit pattern
        circuit_frequency['circuit_pattern'] = pd.cut(
            circuit_frequency['upper_circuit_ratio'],
            bins=[-0.1, 0.3, 0.7, 1.1],
            labels=['Mostly Lower Circuits', 'Mixed', 'Mostly Upper Circuits']
        )
        
        return circuit_frequency.reset_index()
    
    def detect_consecutive_circuits(self, min_consecutive: int = 2) -> pd.DataFrame:
        """
        Detect securities with consecutive circuit hits
        Often indicates strong momentum or manipulation
        """
        df = self.circuit_data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Create groups of consecutive dates
        df['date_diff'] = df.groupby('symbol')['trade_date'].diff().dt.days
        df['new_streak'] = (df['date_diff'] != 1).astype(int)
        df['streak_id'] = df.groupby('symbol')['new_streak'].cumsum()
        
        # Count streak length
        streak_counts = df.groupby(['symbol', 'streak_id']).agg({
            'trade_date': ['min', 'max', 'count'],
            'high_low': lambda x: x.mode()[0] if len(x) > 0 else None
        })
        
        streak_counts.columns = ['start_date', 'end_date', 'consecutive_days', 'direction']
        streak_counts = streak_counts[streak_counts['consecutive_days'] >= min_consecutive]
        
        return streak_counts.reset_index()
    
    def analyze_volume_on_circuit_days(self) -> pd.DataFrame:
        """
        Analyze volume patterns on circuit hit days vs normal days
        """
        df = self.merged_data.copy()
        
        # Calculate average volume for each symbol
        avg_volume = self.price_data.groupby('symbol')['net_traded_qty'].mean()
        avg_volume.name = 'avg_volume_normal'
        
        # Merge with circuit data
        circuit_volume = df.groupby('symbol').agg({
            'net_traded_qty': 'mean',
            'total_trades': 'mean',
            'high_low': 'count'
        })
        circuit_volume.columns = ['avg_volume_circuit', 'avg_trades_circuit', 'circuit_hits']
        
        # Combine
        volume_analysis = pd.concat([circuit_volume, avg_volume], axis=1)
        
        # Calculate volume amplification
        volume_analysis['volume_amplification'] = (
            volume_analysis['avg_volume_circuit'] / 
            volume_analysis['avg_volume_normal'].replace(0, np.nan)
        )
        
        # Classify
        volume_analysis['volume_pattern'] = pd.cut(
            volume_analysis['volume_amplification'],
            bins=[0, 0.8, 1.2, 2, np.inf],
            labels=['Low Volume Circuit', 'Normal Volume', 
                   'High Volume Circuit', 'Extreme Volume Circuit']
        )
        
        return volume_analysis.reset_index()
    
    def identify_pump_and_dump_candidates(self, 
                                         min_circuits: int = 3,
                                         volume_threshold: float = 3.0) -> pd.DataFrame:
        """
        Identify potential pump and dump schemes
        Characteristics:
        - Multiple upper circuits
        - Extremely high volume
        - Often followed by lower circuits
        """
        # Get circuit patterns
        patterns = self.analyze_circuit_patterns()
        volume_patterns = self.analyze_volume_on_circuit_days()
        
        # Merge
        analysis = pd.merge(patterns, volume_patterns, on='symbol', how='inner')
        
        # Flag suspicious patterns
        analysis['pump_dump_score'] = 0
        
        # High number of upper circuits
        analysis.loc[analysis['upper_circuits'] >= min_circuits, 'pump_dump_score'] += 30
        
        # Extreme volume amplification
        analysis.loc[analysis['volume_amplification'] > volume_threshold, 'pump_dump_score'] += 30
        
        # Consecutive circuits
        consecutive = self.detect_consecutive_circuits()
        consecutive_upper = consecutive[consecutive['direction'] == 'H']['symbol'].value_counts()
        analysis['consecutive_upper'] = analysis['symbol'].map(consecutive_upper).fillna(0)
        analysis.loc[analysis['consecutive_upper'] >= 2, 'pump_dump_score'] += 40
        
        # Classify risk
        analysis['manipulation_risk'] = pd.cut(
            analysis['pump_dump_score'],
            bins=[-1, 30, 60, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        return analysis[analysis['pump_dump_score'] > 0].sort_values(
            'pump_dump_score', ascending=False
        )
    
    def analyze_circuit_reversal_patterns(self, forward_days: int = 5) -> pd.DataFrame:
        """
        Analyze what happens after circuit hits
        Do upper circuits continue or reverse?
        """
        df = self.merged_data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Get future returns
        for i in range(1, forward_days + 1):
            df[f'return_t+{i}'] = df.groupby('symbol')['close_price'].pct_change(i).shift(-i) * 100
        
        # Separate by circuit direction
        upper_circuits = df[df['high_low'] == 'H'].copy()
        lower_circuits = df[df['high_low'] == 'L'].copy()
        
        # Calculate average returns after circuit
        results = []
        
        for direction, data in [('Upper Circuit', upper_circuits), 
                               ('Lower Circuit', lower_circuits)]:
            avg_returns = {}
            for i in range(1, forward_days + 1):
                avg_returns[f'Day_{i}'] = data[f'return_t+{i}'].mean()
            
            results.append({
                'circuit_type': direction,
                **avg_returns,
                'success_rate_d1': (data['return_t+1'] > 0).mean() * 100,
                'success_rate_d5': (data['return_t+5'] > 0).mean() * 100,
                'avg_volume': data['net_traded_qty'].mean()
            })
        
        return pd.DataFrame(results)
    
    def circuit_hit_heatmap_data(self) -> pd.DataFrame:
        """
        Generate data for heatmap visualization
        Shows which securities hit circuits on which dates
        """
        df = self.circuit_data.copy()
        
        # Create pivot for heatmap
        heatmap = df.pivot_table(
            values='high_low',
            index='symbol',
            columns='trade_date',
            aggfunc=lambda x: 1 if 'H' in x.values else -1 if 'L' in x.values else 0,
            fill_value=0
        )
        
        return heatmap
    
    def sector_circuit_analysis(self, sector_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Analyze circuit hits by sector
        Identifies if circuits are concentrated in specific sectors
        """
        df = self.merged_data.copy()
        df['sector'] = df['symbol'].map(sector_mapping)
        
        sector_analysis = df.groupby(['sector', 'high_low']).agg({
            'symbol': 'count',
            'net_traded_qty': 'sum'
        }).reset_index()
        
        sector_analysis.columns = ['sector', 'direction', 'circuit_count', 'total_volume']
        
        # Calculate percentage
        total_circuits = sector_analysis.groupby('sector')['circuit_count'].sum()
        sector_analysis['pct_of_sector'] = (
            sector_analysis.apply(
                lambda x: x['circuit_count'] / total_circuits[x['sector']] * 100, 
                axis=1
            )
        )
        
        return sector_analysis


class VolumePatternDetector:
    """
    Advanced volume pattern detection
    """
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize with price data
        """
        self.data = price_data.copy()
    
    def detect_volume_breakout(self, multiplier: float = 2.0, 
                              lookback: int = 20) -> pd.DataFrame:
        """
        Detect volume breakouts (volume > multiplier * average)
        """
        df = self.data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Calculate rolling average volume
        df['volume_ma'] = df.groupby('symbol')['net_traded_qty'].transform(
            lambda x: x.rolling(lookback, min_periods=1).mean()
        )
        
        # Detect breakout
        df['volume_breakout'] = df['net_traded_qty'] > (df['volume_ma'] * multiplier)
        
        # Calculate magnitude
        df['breakout_magnitude'] = df['net_traded_qty'] / df['volume_ma']
        
        # Get price change on breakout day
        df['price_change'] = df.groupby('symbol')['close_price'].pct_change() * 100
        
        breakouts = df[df['volume_breakout']].copy()
        
        return breakouts[['symbol', 'trade_date', 'net_traded_qty', 'volume_ma',
                         'breakout_magnitude', 'price_change']]
    
    def detect_volume_dry_up(self, threshold: float = 0.3, 
                            lookback: int = 20) -> pd.DataFrame:
        """
        Detect volume dry-up (volume < threshold * average)
        Often precedes major moves
        """
        df = self.data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Calculate rolling average volume
        df['volume_ma'] = df.groupby('symbol')['net_traded_qty'].transform(
            lambda x: x.rolling(lookback, min_periods=1).mean()
        )
        
        # Detect dry-up
        df['volume_dryup'] = df['net_traded_qty'] < (df['volume_ma'] * threshold)
        
        # Count consecutive dry-up days
        df['dryup_streak'] = df.groupby('symbol')['volume_dryup'].transform(
            lambda x: x.groupby((~x).cumsum()).cumsum()
        )
        
        dryups = df[df['volume_dryup'] & (df['dryup_streak'] >= 3)].copy()
        
        return dryups[['symbol', 'trade_date', 'net_traded_qty', 
                      'volume_ma', 'dryup_streak']]
    
    def analyze_climactic_volume(self) -> pd.DataFrame:
        """
        Detect climactic volume (exhaustion moves)
        Very high volume with wide range - often marks turning points
        """
        df = self.data.copy()
        
        # Calculate range
        df['price_range_pct'] = ((df['high_price'] - df['low_price']) / 
                                df['low_price']) * 100
        
        # Calculate volume and range percentiles
        df['volume_percentile'] = df.groupby('trade_date')['net_traded_qty'].rank(pct=True) * 100
        df['range_percentile'] = df.groupby('trade_date')['price_range_pct'].rank(pct=True) * 100
        
        # Climactic volume = top 10% in both volume and range
        df['climactic_volume'] = (
            (df['volume_percentile'] > 90) & 
            (df['range_percentile'] > 90)
        )
        
        # Price direction
        df['price_direction'] = np.where(
            df['close_price'] > df['open_price'], 'Buying Climax', 'Selling Climax'
        )
        
        climax = df[df['climactic_volume']].copy()
        
        return climax[['symbol', 'trade_date', 'net_traded_qty', 
                      'price_range_pct', 'price_direction']]


def generate_circuit_dashboard_data(circuit_analyzer: CircuitBreakerAnalyzer) -> Dict:
    """
    Generate all data needed for circuit breaker dashboard
    """
    return {
        'circuit_patterns': circuit_analyzer.analyze_circuit_patterns(),
        'consecutive_circuits': circuit_analyzer.detect_consecutive_circuits(),
        'volume_analysis': circuit_analyzer.analyze_volume_on_circuit_days(),
        'pump_dump_candidates': circuit_analyzer.identify_pump_and_dump_candidates(),
        'reversal_patterns': circuit_analyzer.analyze_circuit_reversal_patterns(),
        'heatmap_data': circuit_analyzer.circuit_hit_heatmap_data()
    }


if __name__ == "__main__":
    print("Circuit Breaker & Volume Pattern Detection Module")
    print("=" * 50)
    print("\nFeatures:")
    print("1. Circuit Hit Pattern Analysis")
    print("2. Consecutive Circuit Detection")
    print("3. Volume Analysis on Circuit Days")
    print("4. Pump & Dump Detection")
    print("5. Circuit Reversal Analysis")
    print("6. Volume Breakout Detection")
    print("7. Volume Dry-up Detection")
    print("8. Climactic Volume Detection")