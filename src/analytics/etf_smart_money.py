"""
ETF Analysis & Smart Money Tracking Module
Analyzes ETF premium/discount and institutional flows
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ETFArbitrageAnalyzer:
    """
    Analyze ETF vs underlying index/stocks for arbitrage opportunities
    """
    
    def __init__(self, etf_data: pd.DataFrame, index_data: pd.DataFrame):
        """
        Args:
            etf_data: DataFrame from ETF file
            index_data: DataFrame with underlying index prices
        """
        self.etf_data = etf_data.copy()
        self.index_data = index_data.copy()
        self._match_etf_to_underlying()
    
    def _match_etf_to_underlying(self):
        """Match ETFs to their underlying indices"""
        # Create mapping dictionary (simplified - in production, use comprehensive mapping)
        self.underlying_map = {
            'NIFTYBEES': 'Nifty 50',
            'BANKBEES': 'Nifty Bank',
            'JUNIORBEES': 'NIFTY NEXT 50',
            'GOLDBEES': 'GOLD',
            # Add more mappings
        }
        
        self.etf_data['underlying'] = self.etf_data['symbol'].map(self.underlying_map)
    
    def calculate_tracking_error(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate tracking error (how closely ETF follows underlying)
        """
        results = []
        
        for symbol in self.etf_data['symbol'].unique():
            etf_prices = self.etf_data[self.etf_data['symbol'] == symbol].sort_values('trade_date')
            
            if len(etf_prices) < window:
                continue
            
            underlying_name = self.underlying_map.get(symbol)
            if not underlying_name:
                continue
            
            underlying_prices = self.index_data[
                self.index_data['security_name'] == underlying_name
            ].sort_values('trade_date')
            
            # Merge on date
            merged = pd.merge(
                etf_prices[['trade_date', 'close_price']],
                underlying_prices[['trade_date', 'close_price']],
                on='trade_date',
                suffixes=('_etf', '_index')
            )
            
            if len(merged) < window:
                continue
            
            # Calculate returns
            merged['return_etf'] = merged['close_price_etf'].pct_change()
            merged['return_index'] = merged['close_price_index'].pct_change()
            
            # Tracking difference
            merged['tracking_diff'] = merged['return_etf'] - merged['return_index']
            
            # Rolling tracking error (std of tracking difference)
            tracking_error = merged['tracking_diff'].rolling(window).std() * np.sqrt(252) * 100
            
            results.append({
                'etf_symbol': symbol,
                'underlying': underlying_name,
                'latest_tracking_error': tracking_error.iloc[-1],
                'avg_tracking_error': tracking_error.mean(),
                'correlation': merged['return_etf'].corr(merged['return_index'])
            })
        
        return pd.DataFrame(results)
    
    def detect_premium_discount(self) -> pd.DataFrame:
        """
        Detect when ETF trades at premium/discount to NAV
        """
        # Note: Ideal calculation requires actual NAV data
        # Here we use close price vs index as proxy
        
        results = []
        
        for symbol in self.etf_data['symbol'].unique():
            etf_current = self.etf_data[self.etf_data['symbol'] == symbol].iloc[-1]
            
            underlying_name = self.underlying_map.get(symbol)
            if not underlying_name:
                continue
            
            underlying_current = self.index_data[
                self.index_data['security_name'] == underlying_name
            ].iloc[-1]
            
            # Calculate implied premium/discount
            # This is simplified - in reality, need to account for ETF units per index point
            etf_return = ((etf_current['close_price'] - etf_current['prev_close']) / 
                         etf_current['prev_close']) * 100
            index_return = ((underlying_current['close_price'] - underlying_current['prev_close']) / 
                           underlying_current['prev_close']) * 100
            
            premium_discount = etf_return - index_return
            
            results.append({
                'etf_symbol': symbol,
                'etf_name': etf_current['security_name'],
                'underlying': underlying_name,
                'etf_price': etf_current['close_price'],
                'etf_return': etf_return,
                'index_return': index_return,
                'premium_discount': premium_discount,
                'etf_volume': etf_current['net_traded_qty'],
                'etf_value': etf_current['net_traded_value']
            })
        
        df = pd.DataFrame(results)
        
        # Flag arbitrage opportunities
        df['arbitrage_opportunity'] = abs(df['premium_discount']) > 0.5
        df['arb_direction'] = np.where(
            df['premium_discount'] > 0.5, 'Short ETF / Long Index',
            np.where(df['premium_discount'] < -0.5, 'Long ETF / Short Index', 'None')
        )
        
        return df
    
    def analyze_etf_flows(self, lookback: int = 5) -> pd.DataFrame:
        """
        Analyze ETF volume trends as proxy for institutional flows
        """
        df = self.etf_data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Calculate rolling average volume
        df['volume_ma'] = df.groupby('symbol')['net_traded_qty'].transform(
            lambda x: x.rolling(lookback, min_periods=1).mean()
        )
        
        # Volume trend
        df['volume_change'] = df.groupby('symbol')['net_traded_qty'].pct_change() * 100
        
        # Value trend (volume * price)
        df['traded_value'] = df['net_traded_qty'] * df['close_price']
        df['value_change'] = df.groupby('symbol')['traded_value'].pct_change() * 100
        
        # Classify flow
        latest = df.groupby('symbol').last().reset_index()
        
        latest['flow_status'] = pd.cut(
            latest['volume_change'],
            bins=[-np.inf, -20, -5, 5, 20, np.inf],
            labels=['Heavy Outflow', 'Outflow', 'Stable', 'Inflow', 'Heavy Inflow']
        )
        
        return latest[['symbol', 'security_name', 'underlying', 'net_traded_qty',
                      'volume_ma', 'volume_change', 'flow_status']]


class SmartMoneyTracker:
    """
    Track smart money (institutional) activity
    """
    
    def __init__(self, price_data: pd.DataFrame, top_traded: pd.DataFrame, 
                 etf_data: pd.DataFrame):
        """
        Args:
            price_data: Main price data
            top_traded: Data from TT file (top 25 by value)
            etf_data: ETF trading data
        """
        self.price_data = price_data.copy()
        self.top_traded = top_traded.copy()
        self.etf_data = etf_data.copy()
    
    def identify_institutional_buying(self, 
                                     volume_threshold: float = 2.0,
                                     value_threshold: float = 100_000_000) -> pd.DataFrame:
        """
        Identify stocks with institutional buying patterns
        High volume + high value + price strength
        """
        df = self.price_data.copy()
        
        # Calculate metrics
        df['total_value'] = df['net_traded_qty'] * df['close_price']
        df['price_change'] = df.groupby('symbol')['close_price'].pct_change() * 100
        
        # Average volume
        avg_vol = df.groupby('symbol')['net_traded_qty'].transform('mean')
        df['volume_ratio'] = df['net_traded_qty'] / avg_vol
        
        # Filter for institutional patterns
        institutional = df[
            (df['volume_ratio'] > volume_threshold) &
            (df['total_value'] > value_threshold) &
            (df['price_change'] > 0)  # Buying (price up)
        ].copy()
        
        # Calculate institutional score
        institutional['inst_score'] = (
            np.clip(institutional['volume_ratio'] / 5 * 40, 0, 40) +
            np.clip(institutional['total_value'] / 1_000_000_000 * 40, 0, 40) +
            np.clip(institutional['price_change'] / 5 * 20, 0, 20)
        )
        
        return institutional.sort_values('inst_score', ascending=False)[
            ['symbol', 'security_name', 'trade_date', 'close_price', 'price_change',
             'net_traded_qty', 'total_value', 'volume_ratio', 'inst_score']
        ]
    
    def track_top_traded_changes(self) -> pd.DataFrame:
        """
        Track changes in top 25 traded stocks
        New entries indicate shift in institutional focus
        """
        df = self.top_traded.copy()
        df = df.sort_values('trade_date')
        
        # Get current and previous top 25
        latest_date = df['trade_date'].max()
        prev_date = df[df['trade_date'] < latest_date]['trade_date'].max()
        
        latest_stocks = set(df[df['trade_date'] == latest_date]['security_name'])
        prev_stocks = set(df[df['trade_date'] == prev_date]['security_name']) if pd.notna(prev_date) else set()
        
        # New entries and exits
        new_entries = latest_stocks - prev_stocks
        exits = prev_stocks - latest_stocks
        
        return {
            'date': latest_date,
            'new_entries': list(new_entries),
            'exits': list(exits),
            'stable': list(latest_stocks & prev_stocks)
        }
    
    def analyze_etf_institutional_flow(self) -> pd.DataFrame:
        """
        Analyze ETF flows as indicator of institutional positioning
        Sector ETFs show sector rotation
        """
        etf_analyzer = ETFArbitrageAnalyzer(self.etf_data, self.price_data)
        flows = etf_analyzer.analyze_etf_flows()
        
        # Categorize by ETF type
        flows['etf_category'] = flows['underlying'].apply(self._categorize_etf)
        
        # Aggregate by category
        def _safe_mode(x):
            m = x.mode()
            return m.iloc[0] if len(m) > 0 else 'Unknown'

        category_flows = flows.groupby('etf_category').agg({
            'volume_change': 'mean',
            'flow_status': _safe_mode,
        }).reset_index()
        
        return category_flows
    
    @staticmethod
    def _categorize_etf(underlying: str) -> str:
        """Categorize ETF by type"""
        if pd.isna(underlying):
            return 'Other'
        underlying_upper = str(underlying).upper()
        
        if 'BANK' in underlying_upper:
            return 'Banking'
        elif 'IT' in underlying_upper or 'TECH' in underlying_upper:
            return 'Technology'
        elif 'GOLD' in underlying_upper:
            return 'Commodities'
        elif '50' in underlying_upper:
            return 'Large Cap'
        elif 'MIDCAP' in underlying_upper or 'SMALL' in underlying_upper:
            return 'Mid/Small Cap'
        else:
            return 'Sector/Thematic'
    
    def detect_smart_money_rotation(self) -> Dict:
        """
        Detect where smart money is rotating
        """
        # ETF flows by category
        etf_rotation = self.analyze_etf_institutional_flow()
        
        # Top traded analysis
        top_changes = self.track_top_traded_changes()
        
        # Institutional buying stocks
        inst_buying = self.identify_institutional_buying()
        
        return {
            'etf_rotation': etf_rotation,
            'top_traded_changes': top_changes,
            'hot_stocks': inst_buying.head(10)
        }
    
    def calculate_money_flow_index(self, window: int = 14) -> pd.DataFrame:
        """
        Calculate Money Flow Index (MFI) - volume-weighted RSI
        Values > 80 = overbought, < 20 = oversold
        """
        df = self.price_data.copy()
        df = df.sort_values(['symbol', 'trade_date'])
        
        # Typical price
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        
        # Raw money flow
        df['raw_money_flow'] = df['typical_price'] * df['net_traded_qty']
        
        # Positive and negative money flow
        df['price_change'] = df.groupby('symbol')['typical_price'].diff()
        
        df['positive_flow'] = np.where(df['price_change'] > 0, df['raw_money_flow'], 0)
        df['negative_flow'] = np.where(df['price_change'] < 0, df['raw_money_flow'], 0)
        
        # Sum over window
        df['positive_mf'] = df.groupby('symbol')['positive_flow'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        df['negative_mf'] = df.groupby('symbol')['negative_flow'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        
        # Money flow ratio and index
        df['mf_ratio'] = df['positive_mf'] / df['negative_mf'].replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + df['mf_ratio']))
        
        # Classify
        df['mfi_signal'] = pd.cut(
            df['mfi'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Oversold', 'Weak', 'Neutral', 'Strong', 'Overbought']
        )
        
        return df[['symbol', 'trade_date', 'mfi', 'mfi_signal']]


if __name__ == "__main__":
    print("ETF Analysis & Smart Money Tracking Module")
    print("=" * 50)
    print("\nFeatures:")
    print("1. ETF Tracking Error Analysis")
    print("2. Premium/Discount Detection")
    print("3. ETF Flow Analysis")
    print("4. Institutional Buying Detection")
    print("5. Top Traded Changes Tracking")
    print("6. Smart Money Rotation Detection")
    print("7. Money Flow Index Calculation")