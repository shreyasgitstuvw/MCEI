"""
Market Microstructure Analyzer
===============================
High-level analysis functions for the dashboard.
Wraps the core market_microstructure.py module.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Import core liquidity calculator
from .market_microstructure import (
    calculate_liquidity_scores,
    classify_liquidity,
    identify_volume_clusters,
    calculate_overnight_gaps
)


class MarketMicrostructureAnalyzer:
    """Analyzer for market microstructure patterns."""
    
    def __init__(self, engine):
        """Initialize with database engine."""
        self.engine = engine
    
    def get_liquidity_distribution(self, trade_date: date = None) -> pd.DataFrame:
        """Get liquidity score distribution for a trading day."""
        if trade_date is None:
            trade_date = self._get_latest_date()
        
        # Fetch data
        query = f"""
        SELECT symbol, series, net_traded_qty, net_traded_value,
               delivery_pct, close_price
        FROM fact_daily_prices
        WHERE trade_date = '{trade_date}'
          AND series = 'EQ'
          AND delivery_pct IS NOT NULL
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate scores
        scores = calculate_liquidity_scores(df)
        
        # Add classification
        scores['liquidity_class'] = scores['liquidity_score'].apply(classify_liquidity)
        
        return scores
    
    def get_volume_anomalies(self, trade_date: date = None, threshold: float = 2.0) -> pd.DataFrame:
        """Identify stocks with unusual volume."""
        if trade_date is None:
            trade_date = self._get_latest_date()
        
        query = f"""
        SELECT symbol, series, net_traded_qty, net_traded_value,
               delivery_pct, close_price
        FROM fact_daily_prices
        WHERE trade_date = '{trade_date}'
          AND series = 'EQ'
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return pd.DataFrame()
        
        return identify_volume_clusters(df, threshold)
    
    def get_overnight_gaps(self, trade_date: date = None) -> pd.DataFrame:
        """Get stocks with significant overnight gaps."""
        if trade_date is None:
            trade_date = self._get_latest_date()
        
        query = f"""
        SELECT symbol, open_price, prev_close, close_price
        FROM fact_daily_prices
        WHERE trade_date = '{trade_date}'
          AND series = 'EQ'
          AND open_price IS NOT NULL
          AND prev_close IS NOT NULL
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return pd.DataFrame()
        
        return calculate_overnight_gaps(df)
    
    def get_delivery_trends(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get delivery percentage trends for a symbol."""
        end_date = self._get_latest_date()
        start_date = end_date - timedelta(days=days)
        
        query = f"""
        SELECT trade_date, delivery_pct, net_traded_qty, close_price
        FROM fact_daily_prices
        WHERE symbol = '{symbol}'
          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND delivery_pct IS NOT NULL
        ORDER BY trade_date
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        return df
    
    def get_liquidity_summary(self, trade_date: date = None) -> Dict:
        """Get overall market liquidity summary."""
        scores = self.get_liquidity_distribution(trade_date)
        
        if scores.empty:
            return {
                'avg_liquidity': 0,
                'highly_liquid': 0,
                'liquid': 0,
                'moderate': 0,
                'illiquid': 0,
                'total_stocks': 0
            }
        
        return {
            'avg_liquidity': scores['liquidity_score'].mean(),
            'highly_liquid': len(scores[scores['liquidity_class'] == 'HIGHLY_LIQUID']),
            'liquid': len(scores[scores['liquidity_class'] == 'LIQUID']),
            'moderate': len(scores[scores['liquidity_class'] == 'MODERATE']),
            'illiquid': len(scores[scores['liquidity_class'] == 'ILLIQUID']),
            'total_stocks': len(scores)
        }
    
    def _get_latest_date(self) -> date:
        """Get the latest trading date in database."""
        query = "SELECT MAX(trade_date) as latest FROM fact_daily_prices"
        
        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn)
        
        return result.iloc[0]['latest']


# Convenience functions for dashboard
def analyze_liquidity(engine, trade_date: date = None) -> pd.DataFrame:
    """Quick liquidity analysis."""
    analyzer = MarketMicrostructureAnalyzer(engine)
    return analyzer.get_liquidity_distribution(trade_date)


def get_market_summary(engine, trade_date: date = None) -> Dict:
    """Quick market summary."""
    analyzer = MarketMicrostructureAnalyzer(engine)
    return analyzer.get_liquidity_summary(trade_date)


def find_unusual_volume(engine, trade_date: date = None, threshold: float = 2.0) -> pd.DataFrame:
    """Find stocks with unusual volume."""
    analyzer = MarketMicrostructureAnalyzer(engine)
    return analyzer.get_volume_anomalies(trade_date, threshold)