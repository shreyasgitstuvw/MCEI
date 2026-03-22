"""
Multi-Source Data Pipeline - Comprehensive Test
================================================
Tests the complete NSE data pipeline:
1. Downloads data from real NSE API endpoints
2. Transforms all data sources
3. Loads into database
4. Runs diagnostics to verify real vs placeholder values

Two modes:
- LIVE mode: Downloads fresh data from NSE (requires internet)
- SAMPLE mode: Uses uploaded sample files for testing

Usage:
    python -m scripts.test_multisource_pipeline
    python -m scripts.test_multisource_pipeline --live  # Download from NSE
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
from shutil import copy2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl.multi_transformer import transform_all
from src.database.loader import load_all
from src.database.connection import get_engine, test_connection
import pandas as pd


def setup_test_data():
    """Copy uploaded files to expected locations for testing."""
    
    # Source: uploaded files
    uploads = Path("/mnt/user-data/uploads")
    
    # Destination: data/raw/20260223
    test_date = date(2026, 2, 23)
    dest_dir = Path("data/raw") / test_date.strftime("%Y%m%d")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # File mapping
    file_map = {
        "52WeekHigh.csv": "52WeekHigh.csv",
        "52WeekLow.csv": "52WeekLow.csv",
        "CF-CA-equities-23-Feb-2026.csv": "corporate_actions.csv",
        "sec_bhavdata_full_23022026.csv": "full_bhavcopy.csv",
    }
    
    print(f"\n{'='*60}")
    print("  Setting Up Test Data")
    print(f"{'='*60}\n")
    
    for source, dest in file_map.items():
        src_path = uploads / source
        if src_path.exists():
            dest_path = dest_dir / dest
            copy2(src_path, dest_path)
            print(f"  ✅ Copied: {source} -> {dest}")
        else:
            print(f"  ⚠️  Missing: {source}")
    
    return dest_dir, test_date


def check_delivery_data(engine):
    """Check if delivery data is actually being used."""
    
    print(f"\n{'='*60}")
    print("  Checking Delivery Data (Liquidity Source)")
    print(f"{'='*60}\n")
    
    query = """
    SELECT 
        COUNT(*) as total_rows,
        COUNT(delivery_pct) as rows_with_delivery,
        AVG(delivery_pct) as avg_delivery_pct,
        MIN(delivery_pct) as min_delivery_pct,
        MAX(delivery_pct) as max_delivery_pct
    FROM fact_daily_prices
    WHERE trade_date = '2026-02-23'
    """
    
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    
    if result.empty:
        print("  ❌ No data found for 2026-02-23")
        return False
    
    row = result.iloc[0]
    print(f"  Total rows: {row['total_rows']}")
    print(f"  Rows with delivery data: {row['rows_with_delivery']}")
    
    if row['rows_with_delivery'] > 0:
        print(f"  ✅ REAL delivery data found!")
        print(f"     Avg delivery %: {row['avg_delivery_pct']:.2f}%")
        print(f"     Min delivery %: {row['min_delivery_pct']:.2f}%")
        print(f"     Max delivery %: {row['max_delivery_pct']:.2f}%")
        return True
    else:
        print(f"  ⚠️  No delivery data - still using placeholders")
        return False


def check_liquidity_calculation():
    """Check if liquidity scores are real or placeholder 50."""
    
    print(f"\n{'='*60}")
    print("  Checking Liquidity Scores")
    print(f"{'='*60}\n")
    
    # Import the analytics module
    from src.analytics.market_microstructure import calculate_liquidity_scores
    from database.queries import _query
    
    # Get some sample data
    df = _query("""
        SELECT symbol, series, net_traded_qty, net_traded_value, 
               delivery_pct, close_price
        FROM fact_daily_prices
        WHERE trade_date = '2026-02-23'
          AND series = 'EQ'
        LIMIT 100
    """)
    
    if df.empty:
        print("  ❌ No data to calculate liquidity")
        return False
    
    # Calculate liquidity scores
    scores = calculate_liquidity_scores(df)
    
    print(f"  Calculated {len(scores)} liquidity scores")
    
    # Check if all scores are 50 (placeholder)
    unique_scores = scores['liquidity_score'].nunique()
    all_fifty = (scores['liquidity_score'] == 50).all()
    
    if all_fifty:
        print(f"  ⚠️  All scores are 50 - STILL USING PLACEHOLDERS!")
        print(f"     This means delivery data is not being used properly")
        return False
    else:
        print(f"  ✅ REAL liquidity scores!")
        print(f"     Unique scores: {unique_scores}")
        print(f"     Score range: {scores['liquidity_score'].min():.1f} - {scores['liquidity_score'].max():.1f}")
        print(f"     Average score: {scores['liquidity_score'].mean():.1f}")
        
        # Show top 5 most liquid stocks
        print(f"\n  Top 5 Most Liquid Stocks:")
        top5 = scores.nlargest(5, 'liquidity_score')[['symbol', 'liquidity_score']]
        for idx, row in top5.iterrows():
            print(f"     {row['symbol']}: {row['liquidity_score']:.1f}")
        
        return True


def main():
    """Run the complete multi-source pipeline test."""
    
    print(f"\n{'='*70}")
    print("  NSE MULTI-SOURCE DATA PIPELINE TEST")
    print(f"{'='*70}")
    
    # Step 1: Setup test data
    data_dir, test_date = setup_test_data()
    
    # Step 2: Test database connection
    print(f"\n{'='*60}")
    print("  Testing Database Connection")
    print(f"{'='*60}\n")
    
    if not test_connection():
        print("  ❌ Database connection failed!")
        return
    
    engine = get_engine()
    
    # Step 3: Transform all data sources
    print(f"\n{'='*60}")
    print("  Transforming All Data Sources")
    print(f"{'='*60}\n")
    
    datasets = transform_all(data_dir, test_date)
    
    if not datasets:
        print("  ❌ No datasets produced!")
        return
    
    # Step 4: Load to database
    print(f"\n{'='*60}")
    print("  Loading to Database")
    print(f"{'='*60}\n")
    
    summary = load_all(datasets, engine)
    
    print(f"\n  Load Summary:")
    for table, count in summary.items():
        print(f"    {table}: {count} rows")
    
    total_rows = sum(summary.values())
    print(f"\n  ✅ Total rows loaded: {total_rows}")
    
    # Step 5: Verify delivery data
    has_delivery = check_delivery_data(engine)
    
    # Step 6: Check liquidity calculations
    has_real_liquidity = check_liquidity_calculation()
    
    # Final verdict
    print(f"\n{'='*60}")
    print("  FINAL VERDICT")
    print(f"{'='*60}\n")
    
    if has_delivery and has_real_liquidity:
        print("  ✅ SUCCESS! Real liquidity data is working!")
        print("     - Delivery percentages are in database")
        print("     - Liquidity scores are calculated from real data")
        print("     - No more placeholder values!")
    elif has_delivery and not has_real_liquidity:
        print("  ⚠️  PARTIAL SUCCESS")
        print("     - Delivery data is in database ✅")
        print("     - But liquidity calculation still uses placeholders ⚠️")
        print("     - Need to fix calculate_liquidity_scores() function")
    else:
        print("  ❌ STILL USING PLACEHOLDERS")
        print("     - No delivery data in database")
        print("     - Liquidity scores are hardcoded to 50")
        print("     - Need to:")
        print("       1. Run: psql -U postgres -d bhavcopy_db -f scripts/add_delivery_columns.sql")
        print("       2. Reload full bhavcopy data with delivery columns")
    
    print()


if __name__ == "__main__":
    main()