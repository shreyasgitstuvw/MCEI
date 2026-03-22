"""
Deep Debug - Find Exact Problematic Row
========================================
Test the exact data flow from transformation to database insert.
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from datetime import date
from src.etl.multi_transformer import transform_full_bhavcopy
from src.database.connection import get_engine
from sqlalchemy import text
import pandas as pd

# PostgreSQL BIGINT limits
BIGINT_MIN = -9223372036854775808
BIGINT_MAX = 9223372036854775807

def deep_debug():
    file_path = Path("data/raw/20260223/sec_bhavdata_full.csv")
    trade_date = date(2026, 2, 23)
    
    print(f"\n{'='*70}")
    print(f"  DEEP DEBUG - Finding Exact Issue")
    print(f"{'='*70}\n")
    
    # Step 1: Transform
    print("Step 1: Transforming data...")
    result = transform_full_bhavcopy(file_path, trade_date)
    print(f"  Rows: {len(result)}")
    
    # Step 2: Check dtypes
    print(f"\nStep 2: Checking data types...")
    int_cols = ['net_traded_qty', 'net_traded_value', 'total_trades', 'delivery_qty']
    for col in int_cols:
        print(f"  {col}: {result[col].dtype}")
    
    # Step 3: Sample the data that will be sent to DB
    print(f"\nStep 3: First 3 rows as they'll be sent to DB...")
    sample = result.head(3)
    for idx, row in sample.iterrows():
        print(f"\n  Row {idx} ({row['symbol']}):")
        for col in int_cols:
            val = row[col]
            print(f"    {col}: {val} (type: {type(val).__name__})")
    
    # Step 4: Convert to dict (what gets sent to DB)
    print(f"\nStep 4: Converting to dict (simulates DB insert)...")
    
    # This is what the loader does
    records = result.to_dict('records')
    
    print(f"  First record:")
    first_record = records[0]
    for col in int_cols:
        val = first_record[col]
        print(f"    {col}: {val} (type: {type(val).__name__})")
    
    # Step 5: Try inserting ONE row
    print(f"\nStep 5: Testing single row insert...")
    
    engine = get_engine()
    test_row = {
        'trade_date': result.iloc[0]['trade_date'],
        'symbol': 'TEST_DEBUG',
        'series': 'EQ',
        'security_name': 'Test',
        'prev_close': float(result.iloc[0]['prev_close']),
        'open_price': float(result.iloc[0]['open_price']),
        'high_price': float(result.iloc[0]['high_price']),
        'low_price': float(result.iloc[0]['low_price']),
        'close_price': float(result.iloc[0]['close_price']),
        'net_traded_qty': result.iloc[0]['net_traded_qty'],
        'net_traded_value': result.iloc[0]['net_traded_value'],
        'total_trades': result.iloc[0]['total_trades'],
        'delivery_qty': result.iloc[0]['delivery_qty'],
        'delivery_pct': float(result.iloc[0]['delivery_pct']) if pd.notna(result.iloc[0]['delivery_pct']) else None,
        'is_index': False,
        'is_valid': True,
    }
    
    print(f"\n  Test row types:")
    for k, v in test_row.items():
        if k in int_cols or k in ['trade_date', 'symbol']:
            print(f"    {k}: {v} (type: {type(v).__name__})")
    
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO fact_daily_prices 
                (trade_date, symbol, series, security_name, prev_close, open_price, 
                 high_price, low_price, close_price, net_traded_qty, net_traded_value, 
                 total_trades, delivery_qty, delivery_pct, is_index, is_valid)
                VALUES 
                (:trade_date, :symbol, :series, :security_name, :prev_close, :open_price,
                 :high_price, :low_price, :close_price, :net_traded_qty, :net_traded_value,
                 :total_trades, :delivery_qty, :delivery_pct, :is_index, :is_valid)
            """), test_row)
        print(f"\n  ✅ Single row inserted successfully!")
        
        # Clean up
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fact_daily_prices WHERE symbol = 'TEST_DEBUG'"))
        
    except Exception as e:
        print(f"\n  ❌ Single row insert FAILED!")
        print(f"     Error: {e}")
        print(f"\n  This tells us the issue is with data types")
    
    # Step 6: Try bulk insert (what actually happens)
    print(f"\nStep 6: Testing bulk insert (first 5 rows)...")
    
    bulk_data = result.head(5).to_dict('records')
    
    # Check types in bulk data
    print(f"\n  Types in bulk_data[0]:")
    for col in int_cols:
        val = bulk_data[0][col]
        print(f"    {col}: {type(val).__name__} = {val}")
    
    # The issue: pandas to_dict might preserve float dtypes!
    # Let's check
    print(f"\n  Checking if any values are still float...")
    for i, record in enumerate(bulk_data[:3]):
        has_float = False
        for col in int_cols:
            val = record[col]
            if isinstance(val, float) and val is not None and not pd.isna(val):
                print(f"  ⚠️  Row {i} ({record['symbol']}): {col} is float: {val}")
                has_float = True
        if not has_float:
            print(f"  ✅ Row {i} ({record['symbol']}): All int columns are proper ints")
    
    print(f"\n{'='*70}")
    print("  DEBUG COMPLETE")
    print(f"{'='*70}\n")
    
    return result


if __name__ == "__main__":
    result = deep_debug()