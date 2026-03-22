"""
Data Diagnostic Script
======================
Find which value is causing the BIGINT overflow error.
"""

import pandas as pd
from pathlib import Path
import sys

# PostgreSQL BIGINT limits
BIGINT_MIN = -9223372036854775808
BIGINT_MAX = 9223372036854775807

def diagnose_file(file_path: Path):
    """Diagnose the full bhavcopy file for problematic values."""
    
    print(f"\n{'='*70}")
    print(f"  Diagnosing: {file_path.name}")
    print(f"{'='*70}\n")
    
    # Read the file
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Check each numeric column
    numeric_cols = ['TTL_TRD_QNTY', 'TURNOVER_LACS', 'NO_OF_TRADES', 'DELIV_QTY', 'DELIV_PER']
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"⚠️  Column {col} not found")
            continue
        
        print(f"\n--- {col} ---")
        
        # Convert to numeric
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        # Stats
        print(f"  Non-null values: {numeric_series.notna().sum():,}")
        print(f"  Min: {numeric_series.min()}")
        print(f"  Max: {numeric_series.max()}")
        print(f"  Mean: {numeric_series.mean():.2f}")
        
        # Check for overflow after turnover conversion
        if col == 'TURNOVER_LACS':
            converted = numeric_series * 100000
            print(f"\n  After × 100000:")
            print(f"    Min: {converted.min()}")
            print(f"    Max: {converted.max()}")
            
            # Find values that exceed BIGINT
            overflow = converted > BIGINT_MAX
            if overflow.any():
                print(f"\n  ❌ OVERFLOW FOUND! {overflow.sum()} values exceed BIGINT_MAX")
                print(f"\n  Problematic rows:")
                problem_rows = df[overflow][['SYMBOL', 'TURNOVER_LACS', 'TTL_TRD_QNTY']]
                print(problem_rows.head(10))
                
                # Show the actual converted values
                print(f"\n  Converted values:")
                for idx in df[overflow].head(5).index:
                    symbol = df.loc[idx, 'SYMBOL']
                    turnover_lacs = numeric_series.loc[idx]
                    converted_val = turnover_lacs * 100000
                    print(f"    {symbol}: {turnover_lacs} lacs → {converted_val:,.0f}")
                    print(f"      Exceeds BIGINT by: {converted_val - BIGINT_MAX:,.0f}")
        
        # Check if values themselves exceed BIGINT
        exceeds_max = numeric_series > BIGINT_MAX
        exceeds_min = numeric_series < BIGINT_MIN
        
        if exceeds_max.any():
            print(f"  ⚠️  {exceeds_max.sum()} values > BIGINT_MAX directly")
            print(f"  Examples: {numeric_series[exceeds_max].head().tolist()}")
        
        if exceeds_min.any():
            print(f"  ⚠️  {exceeds_min.sum()} values < BIGINT_MIN")
    
    # Check for NaN values that become problematic
    print(f"\n--- Checking Non-Numeric Values ---")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        # Find non-numeric values
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        
        if non_numeric.any():
            print(f"\n  {col}: {non_numeric.sum()} non-numeric values")
            print(f"  Examples: {df[non_numeric][col].unique()[:5].tolist()}")
    
    print(f"\n{'='*70}")
    print("  Diagnosis Complete")
    print(f"{'='*70}\n")


def main():
    file_path = Path("data/raw/20260223/sec_bhavdata_full.csv")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    diagnose_file(file_path)
    
    # Recommendation
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 70)
    print("\n1. If TURNOVER_LACS overflow found:")
    print("   → Change net_traded_value to use raw TURNOVER_LACS without × 100000")
    print("   → Or cap values at BIGINT_MAX")
    print("\n2. If other columns overflow:")
    print("   → Filter out those rows")
    print("   → Or cap at BIGINT_MAX")
    print()


if __name__ == "__main__":
    main()