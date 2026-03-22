"""
Test Actual Transformation
===========================
Run the exact transformation and find which row fails.
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from datetime import date
from src.etl.multi_transformer import transform_full_bhavcopy
import pandas as pd

# PostgreSQL BIGINT limits
BIGINT_MAX = 9223372036854775807

def test_transformation():
    file_path = Path("data/raw/20260223/sec_bhavdata_full.csv")
    trade_date = date(2026, 2, 23)
    
    print(f"\n{'='*70}")
    print(f"  Testing Actual Transformation")
    print(f"{'='*70}\n")
    
    # Run the transformation
    result = transform_full_bhavcopy(file_path, trade_date)
    
    print(f"Transformed rows: {len(result):,}")
    print(f"Columns: {list(result.columns)}\n")
    
    # Check each numeric column
    numeric_cols = ['net_traded_qty', 'net_traded_value', 'total_trades', 'delivery_qty', 'delivery_pct']
    
    for col in numeric_cols:
        if col not in result.columns:
            print(f"⚠️  {col} not in result")
            continue
        
        print(f"\n--- {col} ---")
        print(f"  Type: {result[col].dtype}")
        print(f"  Non-null: {result[col].notna().sum():,}")
        print(f"  Min: {result[col].min()}")
        print(f"  Max: {result[col].max()}")
        
        # Check for overflow
        if result[col].dtype in ['float64', 'int64', 'Int64']:
            exceeds = result[col] > BIGINT_MAX
            if exceeds.any():
                print(f"\n  ❌ OVERFLOW: {exceeds.sum()} values exceed BIGINT_MAX")
                print(f"\n  Problematic stocks:")
                problem = result[exceeds][['symbol', col]].head(10)
                for idx, row in problem.iterrows():
                    print(f"    {row['symbol']}: {row[col]:,.0f}")
                    print(f"      Exceeds by: {row[col] - BIGINT_MAX:,.0f}")
        
        # Check for NaN that might cause issues
        has_nan = result[col].isna()
        if has_nan.any():
            print(f"  ℹ️  {has_nan.sum()} NaN values (will be NULL in DB)")
    
    # Check for infinity
    print(f"\n--- Checking for Infinity Values ---")
    for col in numeric_cols:
        if col not in result.columns:
            continue
        if result[col].dtype == 'float64':
            inf_count = (result[col] == float('inf')).sum()
            neg_inf_count = (result[col] == float('-inf')).sum()
            if inf_count > 0:
                print(f"  ❌ {col}: {inf_count} +inf values")
            if neg_inf_count > 0:
                print(f"  ❌ {col}: {neg_inf_count} -inf values")
    
    # Try to identify the exact failing row
    print(f"\n--- Simulating Database Insert ---")
    print("  Testing if values can be converted to int...")
    
    for col in ['net_traded_qty', 'net_traded_value', 'total_trades', 'delivery_qty']:
        if col not in result.columns:
            continue
        
        try:
            # Try converting to int (what database expects)
            test_series = result[col].fillna(0).astype('int64')
            print(f"  ✅ {col}: Can convert to int64")
        except Exception as e:
            print(f"  ❌ {col}: CANNOT convert - {e}")
            
            # Find problematic values
            for idx, val in result[col].items():
                if pd.notna(val):
                    try:
                        int(val)
                    except:
                        print(f"      Problem at row {idx}: {result.loc[idx, 'symbol']} = {val}")
                        break
    
    print(f"\n{'='*70}")
    print("  Test Complete")
    print(f"{'='*70}\n")
    
    # Return first 5 rows for inspection
    print("\nFirst 5 rows of transformed data:")
    print(result[['symbol', 'net_traded_qty', 'net_traded_value', 'delivery_qty', 'delivery_pct']].head())
    
    return result


if __name__ == "__main__":
    result = test_transformation()