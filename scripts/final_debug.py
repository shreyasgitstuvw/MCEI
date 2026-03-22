"""
Final Debug - Test Exact Insert
================================
Take the actual transformed data and try inserting it row by row
to find which exact row/column fails.
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from datetime import date
from src.etl.multi_transformer import transform_full_bhavcopy
from src.database.connection import get_engine
from sqlalchemy import text
import pandas as pd

def test_exact_insert():
    file_path = Path("data/raw/20260223/sec_bhavdata_full.csv")
    trade_date = date(2026, 2, 23)
    
    print(f"\n{'='*70}")
    print(f"  Testing Exact Insert Row by Row")
    print(f"{'='*70}\n")
    
    # Transform
    result = transform_full_bhavcopy(file_path, trade_date)
    print(f"Transformed: {len(result)} rows\n")
    
    engine = get_engine()
    
    # Convert to records (what the loader does)
    records = result.to_dict('records')
    
    # Try inserting first 10 rows one by one
    print("Attempting to insert first 10 rows...\n")
    
    for i, record in enumerate(records[:10]):
        symbol = record['symbol']
        
        # Convert numpy types to Python types (what loader should do)
        import numpy as np
        for key, value in record.items():
            if value is not None:
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    record[key] = float(value)
        
        # Show the record
        print(f"Row {i}: {symbol}")
        print(f"  net_traded_qty: {record['net_traded_qty']} ({type(record['net_traded_qty']).__name__})")
        print(f"  net_traded_value: {record['net_traded_value']} ({type(record['net_traded_value']).__name__})")
        print(f"  total_trades: {record['total_trades']} ({type(record['total_trades']).__name__})")
        print(f"  delivery_qty: {record['delivery_qty']} ({type(record['delivery_qty']).__name__})")
        
        # Try insert
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
                """), record)
            print(f"  ✅ SUCCESS\n")
            
            # Clean up
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM fact_daily_prices WHERE symbol = :s AND trade_date = :d"),
                           {'s': symbol, 'd': trade_date})
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)[:200]}\n")
            
            # Check which value is problematic
            print(f"  Checking value ranges:")
            print(f"    net_traded_qty: {record['net_traded_qty']:,}")
            print(f"    net_traded_value: {record['net_traded_value']:,.2f}")
            print(f"    total_trades: {record['total_trades']:,}")
            if record['delivery_qty'] is not None:
                print(f"    delivery_qty: {record['delivery_qty']:,}")
            
            # Check if total_trades exceeds INTEGER max (2,147,483,647)
            if record['total_trades'] > 2147483647:
                print(f"  ⚠️  total_trades EXCEEDS INTEGER MAX!")
            
            # Check if delivery_qty exceeds BIGINT max
            if record['delivery_qty'] and record['delivery_qty'] > 9223372036854775807:
                print(f"  ⚠️  delivery_qty EXCEEDS BIGINT MAX!")
            
            break
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    test_exact_insert()