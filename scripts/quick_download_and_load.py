"""
Quick Download and Load - NSE Data
===================================
Simple script to download and load NSE data in one go.

Usage:
    python -m scripts.quick_download_and_load
    python -m scripts.quick_download_and_load --date 2026-02-21
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.nse_real_api_downloader import NSERealAPIDownloader
from src.etl.multi_transformer import transform_all
from src.database.loader import load_all
from src.database.connection import get_engine, test_connection
from sqlalchemy import text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2026-02-23', help='Date (YYYY-MM-DD)')
    args = parser.parse_args()

    trade_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    print(f"\n{'=' * 70}")
    print(f"  NSE Quick Download & Load - {trade_date.strftime('%d-%b-%Y')}")
    print(f"{'=' * 70}\n")

    # Step 1: Test database
    print("1️⃣  Testing database connection...")
    if not test_connection():
        print("   ❌ Database failed!")
        return
    print("   ✅ Database connected\n")

    engine = get_engine()

    # Step 2: Ensure delivery columns exist
    print("2️⃣  Checking database schema...")
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE fact_daily_prices 
                ADD COLUMN IF NOT EXISTS delivery_qty BIGINT,
                ADD COLUMN IF NOT EXISTS delivery_pct DECIMAL(5,2)
            """))
            conn.commit()
        print("   ✅ Schema ready\n")
    except Exception as e:
        print(f"   ⚠️  Schema warning: {e}\n")

    # Step 3: Download data
    print("3️⃣  Downloading from NSE APIs...")
    dl = NSERealAPIDownloader()
    results = dl.download_all(trade_date)

    success_count = sum(1 for f in results.values() if f and f.exists())
    if success_count == 0:
        print("\n   ❌ All downloads failed!")
        print("   Try a different date (weekday/trading day)")
        return

    print(f"\n   ✅ Downloaded {success_count}/{len(results)} files\n")

    # Step 4: Transform
    print("4️⃣  Transforming data...")
    data_dir = Path("data/raw") / trade_date.strftime("%Y%m%d")
    datasets = transform_all(data_dir, trade_date)

    if not datasets:
        print("   ❌ No data to transform")
        return

    for name, df in datasets.items():
        print(f"   ✅ {name}: {len(df):,} rows")

    print()

    # Step 5: Load to database
    print("5️⃣  Loading to database...")
    summary = load_all(datasets, engine)

    total_rows = sum(summary.values())
    for table, count in summary.items():
        print(f"   {table}: {count:,} rows")

    print(f"\n   ✅ Loaded {total_rows:,} total rows\n")

    # Step 6: Quick verification
    print("6️⃣  Verifying delivery data...")

    query = f"""
    SELECT 
        COUNT(*) as total,
        COUNT(delivery_pct) as with_delivery,
        AVG(delivery_pct) as avg_delivery
    FROM fact_daily_prices
    WHERE trade_date = '{trade_date}'
    """

    import pandas as pd
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if not df.empty and df.iloc[0]['with_delivery'] > 0:
        row = df.iloc[0]
        print(f"   ✅ Delivery data loaded!")
        print(f"      {row['with_delivery']:,} / {row['total']:,} stocks have delivery data")
        print(f"      Average delivery: {row['avg_delivery']:.1f}%")
    else:
        print(f"   ⚠️  No delivery data found")

    print(f"\n{'=' * 70}")
    print("  🎉 COMPLETE! Data is ready.")
    print(f"{'=' * 70}")
    print(f"\n  Next step: streamlit run src/dashboard/app.py\n")


if __name__ == "__main__":
    main()