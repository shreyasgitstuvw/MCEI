import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def load_sample_data():
    """Load sample CSV files"""
    try:
        # Load price data
        pr_data = pd.read_csv('data/raw/sample/pr_sample.csv')
        pr_data['trade_date'] = pd.to_datetime('2026-02-13')
        pr_data['symbol'] = pr_data['SECURITY'].str.split().str[0]
        pr_data = pr_data.rename(columns={
            'PREV_CL_PR': 'prev_close',
            'OPEN_PRICE': 'open_price',
            'HIGH_PRICE': 'high_price',
            'LOW_PRICE': 'low_price',
            'CLOSE_PRICE': 'close_price',
            'NET_TRDVAL': 'net_traded_value',
            'NET_TRDQTY': 'net_traded_qty',
            'TRADES': 'total_trades',
            'HI_52_WK': 'high_52_week',
            'LO_52_WK': 'low_52_week',
            'SECURITY': 'security_name'
        })
        
        # Load circuit data
        bh_data = pd.read_csv('data/raw/sample/bh_sample.csv')
        bh_data['trade_date'] = pd.to_datetime('2026-02-13')
        bh_data = bh_data.rename(columns={'HIGH/LOW': 'high_low'})
        bh_data['symbol'] = bh_data['SYMBOL']
        
        # Load corporate action data
        bc_data = pd.read_csv('data/raw/sample/bc_sample.csv')
        bc_data = bc_data.rename(columns={
            'SYMBOL': 'symbol',
            'SECURITY': 'security',
            'EX_DT': 'ex_dt',
            'PURPOSE': 'purpose'
        })
        
        print("Sample data loaded successfully!")
        return pr_data, bh_data, bc_data
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return None, None, None

if __name__ == "__main__":
    pr, bh, bc = load_sample_data()
    if pr is not None:
        print(f"\nPrice Data: {len(pr)} records")
        print(f"Circuit Data: {len(bh)} records")
        print(f"Corporate Action Data: {len(bc)} records")
