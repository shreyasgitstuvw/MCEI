"""
Corporate Action Event Study Module
Analyzes price impact of corporate actions (dividends, splits, bonuses, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CorporateActionEventStudy:
    """
    Event study analysis for corporate actions
    """
    
    def __init__(self, corporate_action_data: pd.DataFrame, price_data: pd.DataFrame):
        """
        Initialize with corporate action and price data
        
        Args:
            corporate_action_data: DataFrame from BC file
            price_data: DataFrame with price history
        """
        self.ca_data = corporate_action_data.copy()
        self.price_data = price_data.copy()
        self._parse_corporate_actions()
    
    def _parse_corporate_actions(self):
        """Parse and categorize corporate actions"""
        # Extract action type from PURPOSE column
        self.ca_data['action_category'] = self.ca_data['purpose'].apply(
            self._categorize_action
        )
        
        # Parse dividend amount where applicable
        self.ca_data['dividend_amount'] = self.ca_data['purpose'].str.extract(
            r'RS\s*([\d.]+)'
        )[0].astype(float)
        
        # Convert ex_date to datetime
        self.ca_data['ex_dt'] = pd.to_datetime(self.ca_data['ex_dt'])
    
    @staticmethod
    def _categorize_action(purpose: str) -> str:
        """Categorize corporate action"""
        purpose_upper = str(purpose).upper()
        
        if 'DIVIDEND' in purpose_upper or 'INTDIV' in purpose_upper:
            return 'Dividend'
        elif 'BONUS' in purpose_upper:
            return 'Bonus'
        elif 'SPLIT' in purpose_upper or 'SPLT' in purpose_upper:
            return 'Stock Split'
        elif 'RIGHTS' in purpose_upper or 'RGHTS' in purpose_upper:
            return 'Rights Issue'
        elif 'BUYBACK' in purpose_upper:
            return 'Buyback'
        elif 'INTEREST' in purpose_upper:
            return 'Interest Payment'
        elif 'REDEMPTION' in purpose_upper:
            return 'Redemption'
        else:
            return 'Other'
    
    def calculate_abnormal_returns(self, 
                                   event_window: Tuple[int, int] = (-5, 5),
                                   estimation_window: int = 60) -> pd.DataFrame:
        """
        Calculate abnormal returns around corporate action announcements
        
        Args:
            event_window: (days_before, days_after) event
            estimation_window: days to calculate expected return
        """
        results = []
        
        for idx, row in self.ca_data.iterrows():
            symbol = row['symbol']
            ex_date = row['ex_dt']
            
            # Get price data for this symbol
            symbol_prices = self.price_data[
                self.price_data['symbol'] == symbol
            ].sort_values('trade_date').copy()
            
            if len(symbol_prices) < estimation_window:
                continue
            
            # Get event date index
            event_idx = symbol_prices[
                symbol_prices['trade_date'] == ex_date
            ].index
            
            if len(event_idx) == 0:
                continue
            
            event_idx = event_idx[0]
            event_pos = symbol_prices.index.get_loc(event_idx)
            
            # Calculate returns
            symbol_prices['return'] = symbol_prices['close_price'].pct_change() * 100
            
            # Estimation period (before event window)
            est_start = max(0, event_pos - estimation_window - event_window[0])
            est_end = max(0, event_pos - event_window[0])
            
            if est_end <= est_start:
                continue
            
            expected_return = symbol_prices.iloc[est_start:est_end]['return'].mean()
            
            # Event window returns
            event_start = max(0, event_pos + event_window[0])
            event_end = min(len(symbol_prices), event_pos + event_window[1] + 1)
            
            for i in range(event_start, event_end):
                day = i - event_pos
                actual_return = symbol_prices.iloc[i]['return']
                abnormal_return = actual_return - expected_return
                
                results.append({
                    'symbol': symbol,
                    'ex_date': ex_date,
                    'action_type': row['action_category'],
                    'event_day': day,
                    'actual_return': actual_return,
                    'expected_return': expected_return,
                    'abnormal_return': abnormal_return,
                    'price': symbol_prices.iloc[i]['close_price']
                })
        
        return pd.DataFrame(results)
    
    def calculate_cumulative_abnormal_returns(self, 
                                             abnormal_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Cumulative Abnormal Returns (CAR)
        """
        # Sort by symbol, ex_date, and event_day
        ar_sorted = abnormal_returns.sort_values(['symbol', 'ex_date', 'event_day'])
        
        # Calculate CAR for each event
        ar_sorted['car'] = ar_sorted.groupby(['symbol', 'ex_date'])['abnormal_return'].cumsum()
        
        return ar_sorted
    
    def analyze_by_action_type(self, 
                               abnormal_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze average abnormal returns by corporate action type
        """
        # Calculate average AR and CAR by action type and event day
        summary = abnormal_returns.groupby(['action_type', 'event_day']).agg({
            'abnormal_return': ['mean', 'median', 'std', 'count'],
            'actual_return': ['mean', 'median']
        }).reset_index()
        
        summary.columns = ['action_type', 'event_day', 
                          'ar_mean', 'ar_median', 'ar_std', 'count',
                          'actual_return_mean', 'actual_return_median']
        
        # Calculate cumulative AR
        summary['car_mean'] = summary.groupby('action_type')['ar_mean'].cumsum()
        
        # Statistical significance (t-test)
        summary['t_statistic'] = summary['ar_mean'] / (
            summary['ar_std'] / np.sqrt(summary['count'])
        )
        summary['is_significant'] = abs(summary['t_statistic']) > 1.96  # 95% confidence
        
        return summary
    
    def analyze_dividend_impact(self) -> pd.DataFrame:
        """
        Specific analysis for dividend announcements
        """
        dividend_data = self.ca_data[
            self.ca_data['action_category'] == 'Dividend'
        ].copy()
        
        if len(dividend_data) == 0:
            return pd.DataFrame()
        
        # Calculate dividend yield
        results = []
        
        for idx, row in dividend_data.iterrows():
            symbol = row['symbol']
            ex_date = row['ex_dt']
            div_amount = row['dividend_amount']
            
            if pd.isna(div_amount):
                continue
            
            # Get price on ex-date
            price_on_ex = self.price_data[
                (self.price_data['symbol'] == symbol) &
                (self.price_data['trade_date'] == ex_date)
            ]
            
            if len(price_on_ex) == 0:
                continue
            
            price = price_on_ex.iloc[0]['close_price']
            div_yield = (div_amount / price) * 100
            
            # Get price 1 day after ex-date
            next_day_price = self.price_data[
                (self.price_data['symbol'] == symbol) &
                (self.price_data['trade_date'] > ex_date)
            ].sort_values('trade_date').head(1)
            
            if len(next_day_price) > 0:
                price_drop = ((next_day_price.iloc[0]['close_price'] - price) / price) * 100
            else:
                price_drop = None
            
            results.append({
                'symbol': symbol,
                'ex_date': ex_date,
                'dividend_amount': div_amount,
                'price_on_ex': price,
                'dividend_yield': div_yield,
                'price_drop_next_day': price_drop,
                'adjusted_for_dividend': price_drop + div_yield if price_drop else None
            })
        
        return pd.DataFrame(results)
    
    def detect_announcement_leakage(self, 
                                   lookback_days: int = 10) -> pd.DataFrame:
        """
        Detect potential information leakage before announcements
        Abnormal price movement before ex-date suggests leakage
        """
        ar_df = self.calculate_abnormal_returns(
            event_window=(-lookback_days, 0)
        )

        if ar_df.empty or 'event_day' not in ar_df.columns:
            return pd.DataFrame()

        # Focus on days before announcement (negative event_day)
        pre_announcement = ar_df[ar_df['event_day'] < 0].copy()
        
        # Calculate average AR in pre-announcement period
        leakage_analysis = pre_announcement.groupby(['symbol', 'ex_date', 'action_type']).agg({
            'abnormal_return': ['mean', 'sum', 'std'],
            'event_day': 'count'
        }).reset_index()
        
        leakage_analysis.columns = ['symbol', 'ex_date', 'action_type',
                                    'avg_ar_pre', 'total_ar_pre', 'std_ar_pre', 'days']
        
        # Flag potential leakage (positive AR before good news)
        leakage_analysis['potential_leakage'] = (
            (leakage_analysis['total_ar_pre'] > 2) &  # >2% cumulative abnormal return
            (leakage_analysis['action_type'].isin(['Dividend', 'Bonus', 'Stock Split']))
        )
        
        return leakage_analysis[leakage_analysis['potential_leakage']].sort_values(
            'total_ar_pre', ascending=False
        )
    
    def analyze_post_action_drift(self, 
                                  drift_window: int = 30) -> pd.DataFrame:
        """
        Analyze post-announcement drift
        Do prices continue to move in the same direction after announcement?
        """
        ar_df = self.calculate_abnormal_returns(
            event_window=(0, drift_window)
        )
        
        # Calculate CAR
        car_df = self.calculate_cumulative_abnormal_returns(ar_df)
        
        # Get final CAR for each event
        final_car = car_df.groupby(['symbol', 'ex_date', 'action_type']).last().reset_index()
        
        # Classify drift
        final_car['drift_direction'] = pd.cut(
            final_car['car'],
            bins=[-np.inf, -5, -1, 1, 5, np.inf],
            labels=['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive']
        )
        
        # Summary by action type
        drift_summary = final_car.groupby(['action_type', 'drift_direction']).size().reset_index(name='count')
        
        return drift_summary
    
    def compare_action_effectiveness(self) -> pd.DataFrame:
        """
        Compare effectiveness of different corporate actions
        Which actions create the most shareholder value?
        """
        # Calculate AR for all actions
        ar_df = self.calculate_abnormal_returns(event_window=(-5, 20))
        car_df = self.calculate_cumulative_abnormal_returns(ar_df)
        
        # Get final CAR at day +20
        final_impact = car_df[car_df['event_day'] == 20].copy()
        
        # Summary by action type
        effectiveness = final_impact.groupby('action_type').agg({
            'car': ['mean', 'median', 'std', 'count'],
            'abnormal_return': 'sum'
        }).reset_index()
        
        effectiveness.columns = ['action_type', 'avg_car_20d', 'median_car_20d', 
                                'std_car_20d', 'sample_size', 'total_ar']
        
        # Calculate success rate (positive CAR)
        success_rate = final_impact.groupby('action_type').apply(
            lambda x: (x['car'] > 0).mean() * 100, include_groups=False
        ).reset_index(name='success_rate')
        
        effectiveness = pd.merge(effectiveness, success_rate, on='action_type')
        
        return effectiveness.sort_values('avg_car_20d', ascending=False)


class CorporateActionCalendar:
    """
    Manage corporate action calendar and upcoming events
    """
    
    def __init__(self, ca_data: pd.DataFrame):
        self.ca_data = ca_data.copy()
        self.ca_data['ex_dt'] = pd.to_datetime(self.ca_data['ex_dt'])
    
    def get_upcoming_actions(self, days_ahead: int = 30) -> pd.DataFrame:
        """
        Get upcoming corporate actions
        """
        today = pd.Timestamp.now().normalize()
        future_date = today + timedelta(days=days_ahead)
        
        upcoming = self.ca_data[
            (self.ca_data['ex_dt'] >= today) &
            (self.ca_data['ex_dt'] <= future_date)
        ].copy()
        
        upcoming = upcoming.sort_values('ex_dt')
        
        return upcoming[['symbol', 'security', 'ex_dt', 'purpose']]
    
    def get_actions_by_date_range(self, 
                                  start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """
        Get corporate actions in a date range
        """
        actions = self.ca_data[
            (self.ca_data['ex_dt'] >= start_date) &
            (self.ca_data['ex_dt'] <= end_date)
        ].copy()
        
        return actions.sort_values('ex_dt')


if __name__ == "__main__":
    print("Corporate Action Event Study Module")
    print("=" * 50)
    print("\nFeatures:")
    print("1. Abnormal Returns Calculation")
    print("2. Cumulative Abnormal Returns (CAR)")
    print("3. Analysis by Action Type")
    print("4. Dividend Impact Analysis")
    print("5. Information Leakage Detection")
    print("6. Post-Announcement Drift Analysis")
    print("7. Action Effectiveness Comparison")
    print("8. Corporate Action Calendar")