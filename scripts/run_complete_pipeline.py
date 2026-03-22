"""
NSE Analytics - Complete Pipeline
==================================
Orchestrates the entire data pipeline from download to database to dashboard.

Steps:
1. Download data from NSE API (or use sample files)
2. Transform to database format
3. Add delivery columns to database (if needed)
4. Load data to PostgreSQL
5. Verify real liquidity scores (not placeholder 50.0)
6. Display summary statistics

Usage:
    # Use sample files (uploaded)
    python -m scripts.run_complete_pipeline

    # Download fresh data from NSE
    python -m scripts.run_complete_pipeline --live 2026-02-23

    # Just verify current data
    python -m scripts.run_complete_pipeline --verify-only
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime
from shutil import copy2
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.nse_real_api_downloader import NSERealAPIDownloader
from src.etl.multi_transformer import transform_all
from src.database.loader import load_all
from src.database.connection import get_engine, test_connection
import pandas as pd


def add_delivery_columns(engine):
    """Add delivery columns to database if they don't exist."""
    from sqlalchemy import text

    print(f"\n{'=' * 60}")
    print("  Checking Database Schema")
    print(f"{'=' * 60}\n")

    try:
        with engine.connect() as conn:
            # Check if columns exist
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'fact_daily_prices' 
                AND column_name IN ('delivery_qty', 'delivery_pct')
            """))

            existing = [row[0] for row in result]

            if 'delivery_qty' in existing and 'delivery_pct' in existing:
                print("  ✅ Delivery columns already exist")
                return True

            print("  ⚠️  Delivery columns missing - adding them now...")

            # Add columns
            conn.execute(text("""
                ALTER TABLE fact_daily_prices 
                ADD COLUMN IF NOT EXISTS delivery_qty BIGINT,
                ADD COLUMN IF NOT EXISTS delivery_pct DECIMAL(5,2)
            """))

            # Create index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_prices_delivery 
                ON fact_daily_prices(delivery_pct) 
                WHERE delivery_pct IS NOT NULL
            """))

            conn.commit()

            print("  ✅ Delivery columns added successfully")
            return True

    except Exception as e:
        print(f"  ❌ Schema update failed: {e}")
        print(f"     Please run manually:")
        print(f"     psql -U postgres -d bhavcopy_db -f scripts/add_delivery_columns.sql")
        return False


def setup_sample_data():
    """Copy uploaded sample files."""
    uploads = Path("/mnt/user-data/uploads")
    test_date = date(2026, 2, 23)
    dest_dir = Path("data/raw") / test_date.strftime("%Y%m%d")
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "52WeekHigh.csv": "52WeekHigh.csv",
        "52WeekLow.csv": "52WeekLow.csv",
        "CF-CA-equities-23-Feb-2026.csv": "CF-CA-equities.csv",
        "sec_bhavdata_full_23022026.csv": "sec_bhavdata_full.csv",
        "MW-NIFTY-50-23-Feb-2026.csv": "MW-NIFTY-50.csv",
        "MW-ETF-23-Feb-2026.csv": "MW-ETF.csv",
    }

    print(f"\n{'=' * 60}")
    print("  Using Sample Data Files")
    print(f"{'=' * 60}\n")

    copied = 0
    for source, dest in file_map.items():
        src_path = uploads / source
        if src_path.exists():
            dest_path = dest_dir / dest
            copy2(src_path, dest_path)
            print(f"  ✅ {source}")
            copied += 1

    print(f"\n  Copied {copied}/{len(file_map)} files")
    return dest_dir, test_date


def download_live_data(trade_date: date):
    """Download fresh data from NSE."""
    print(f"\n{'=' * 60}")
    print(f"  Downloading Live Data - {trade_date}")
    print(f"{'=' * 60}\n")

    dl = NSERealAPIDownloader()
    results = dl.download_all(trade_date)

    success_count = sum(1 for f in results.values() if f and f.exists())

    if success_count == 0:
        print("\n  ❌ All downloads failed!")
        print("     Possible reasons:")
        print("     - Network issue")
        print("     - Weekend/holiday (no data)")
        print("     - NSE website down")
        print("\n     Falling back to sample data...")
        return setup_sample_data()

    dest_dir = Path("data/raw") / trade_date.strftime("%Y%m%d")
    print(f"\n  ✅ Downloaded {success_count}/{len(results)} files")
    return dest_dir, trade_date


def verify_liquidity_data(engine, trade_date: date):
    """Verify that real liquidity data is being used."""

    print(f"\n{'=' * 60}")
    print("  Verifying Liquidity Data")
    print(f"{'=' * 60}\n")

    # Check delivery data
    query_delivery = f"""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(delivery_pct) as with_delivery,
        AVG(delivery_pct) as avg_delivery,
        MIN(delivery_pct) as min_delivery,
        MAX(delivery_pct) as max_delivery
    FROM fact_daily_prices
    WHERE trade_date = '{trade_date}'
    """

    with engine.connect() as conn:
        df_del = pd.read_sql(query_delivery, conn)

    if df_del.empty or df_del.iloc[0]['total_rows'] == 0:
        print(f"  ❌ No data found for {trade_date}")
        return False

    row = df_del.iloc[0]
    print(f"  📊 Data Statistics:")
    print(f"     Total rows: {row['total_rows']:,}")
    print(f"     Rows with delivery data: {row['with_delivery']:,}")

    has_delivery = row['with_delivery'] > 0

    if has_delivery:
        print(f"\n  ✅ DELIVERY DATA PRESENT!")
        print(f"     Average delivery %: {row['avg_delivery']:.2f}%")
        print(f"     Range: {row['min_delivery']:.2f}% - {row['max_delivery']:.2f}%")
    else:
        print(f"\n  ⚠️  No delivery data in database")
        return False

    # Calculate liquidity scores
    print(f"\n  🧮 Calculating Liquidity Scores...")

    from src.analytics.market_microstructure import calculate_liquidity_scores

    query_sample = f"""
    SELECT symbol, net_traded_qty, net_traded_value, 
           delivery_pct, close_price
    FROM fact_daily_prices
    WHERE trade_date = '{trade_date}'
      AND series = 'EQ'
      AND delivery_pct IS NOT NULL
    LIMIT 500
    """

    with engine.connect() as conn:
        df_sample = pd.read_sql(query_sample, conn)

    if df_sample.empty:
        print(f"  ⚠️  No equity data to calculate liquidity")
        return False

    scores = calculate_liquidity_scores(df_sample)

    # Check if scores are real
    unique_scores = scores['liquidity_score'].nunique()
    avg_score = scores['liquidity_score'].mean()

    if unique_scores == 1 and abs(avg_score - 50.0) < 0.1:
        print(f"\n  ❌ STILL USING PLACEHOLDERS!")
        print(f"     All liquidity scores = 50.0")
        print(f"     The delivery data is not being used properly")
        return False

    print(f"\n  ✅ REAL LIQUIDITY SCORES!")
    print(f"     Unique scores: {unique_scores}")
    print(f"     Average: {avg_score:.1f}")
    print(f"     Range: {scores['liquidity_score'].min():.1f} - {scores['liquidity_score'].max():.1f}")

    # Show top 5 most liquid
    print(f"\n  🏆 Top 5 Most Liquid Stocks:")
    top5 = scores.nlargest(5, 'liquidity_score')
    for idx, row in top5.iterrows():
        print(f"     {row['symbol']:12s} {row['liquidity_score']:5.1f}")

    return True


def main():
    """Run the complete pipeline."""

    parser = argparse.ArgumentParser(description='NSE Analytics Pipeline')
    parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD), defaults to 2026-02-23')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing data')
    parser.add_argument('--sample', action='store_true', help='Use sample files instead of downloading')
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  NSE ANALYTICS - COMPLETE PIPELINE")
    print(f"{'=' * 70}")

    # Test database
    if not test_connection():
        print("\n❌ Database connection failed!")
        return

    engine = get_engine()

    # Add delivery columns if needed
    if not add_delivery_columns(engine):
        print("\n⚠️  Continuing without delivery columns...")

    # Verify-only mode
    if args.verify_only:
        test_date = date(2026, 2, 23)
        verify_liquidity_data(engine, test_date)
        return

    # Determine date
    if args.date:
        trade_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        trade_date = date(2026, 2, 23)  # Default test date

    # Setup data - default is to download live data
    if args.sample:
        print("\n  📂 Using sample files mode")
        data_dir, trade_date = setup_sample_data()
    else:
        print(f"\n  🌐 Downloading live data for {trade_date}")
        data_dir, trade_date = download_live_data(trade_date)

    # Transform
    print(f"\n{'=' * 60}")
    print("  Transforming Data")
    print(f"{'=' * 60}\n")

    datasets = transform_all(data_dir, trade_date)

    if not datasets:
        print("  ❌ No datasets to load")
        return

    # Load
    print(f"\n{'=' * 60}")
    print("  Loading to Database")
    print(f"{'=' * 60}\n")

    summary = load_all(datasets, engine)

    total_rows = sum(summary.values())
    for table, count in summary.items():
        print(f"  {table:20s} {count:,} rows")

    print(f"\n  ✅ Total: {total_rows:,} rows loaded")

    # Verify
    success = verify_liquidity_data(engine, trade_date)

    # Final verdict
    print(f"\n{'=' * 70}")
    print("  PIPELINE STATUS")
    print(f"{'=' * 70}\n")

    if success:
        print("  🎉 SUCCESS! Pipeline is working perfectly!")
        print("     ✅ Data downloaded/loaded")
        print("     ✅ Delivery data present")
        print("     ✅ Real liquidity scores calculated")
        print("     ✅ No placeholder values")
        print("\n     Dashboard should now show real metrics!")
    else:
        print("  ⚠️  Pipeline completed but issues detected:")
        print("     - Check delivery data loading")
        print("     - Verify liquidity calculation")
        print("     - Review database schema")

    print()


if __name__ == "__main__":
    main()