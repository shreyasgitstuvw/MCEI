"""
Database Diagnostic Script
Tests connection and shows table structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database.connection import get_engine, test_connection
from sqlalchemy import text
import pandas as pd

print("="*60)
print("  DATABASE DIAGNOSTIC")
print("="*60)

# Test 1: Connection
print("\n1. Testing connection...")
if not test_connection():
    print("\n❌ Cannot proceed - fix connection first")
    print("\nCheck your .env file:")
    print("  DB_HOST=localhost")
    print("  DB_PORT=5432")
    print("  DB_NAME=nse_analytics")
    print("  DB_USER=postgres  (or your actual DB user)")
    print("  DB_PASSWORD=your_password")
    sys.exit(1)

engine = get_engine()

# Test 2: List tables
print("\n2. Checking tables...")
with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    ))
    tables = [r[0] for r in result.fetchall()]
    
    if not tables:
        print("❌ No tables found!")
        print("   Run: psql -U postgres -d nse_analytics -f scripts/setup_database.sql")
    else:
        print(f"✅ Found {len(tables)} tables:")
        for t in tables:
            print(f"   - {t}")

# Test 3: Check fact_daily_prices structure
print("\n3. Checking fact_daily_prices columns...")
try:
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='fact_daily_prices' "
            "ORDER BY ordinal_position"
        ))
        cols = result.fetchall()
        
        if not cols:
            print("❌ Table fact_daily_prices doesn't exist or has no columns")
        else:
            print(f"✅ Table has {len(cols)} columns:")
            for name, dtype in cols:
                if name not in ['id', 'created_at']:
                    print(f"   - {name:<20} ({dtype})")
except Exception as e:
    print(f"❌ Error checking table: {e}")

# Test 4: Check if table is empty
print("\n4. Checking data...")
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM fact_daily_prices"))
        count = result.scalar()
        print(f"   fact_daily_prices: {count:,} rows")
        
        result = conn.execute(text("SELECT COUNT(*) FROM fact_etf_prices"))
        count = result.scalar()
        print(f"   fact_etf_prices:   {count:,} rows")
except Exception as e:
    print(f"❌ Error querying data: {e}")

# Test 5: Simulate what the loader does
print("\n5. Testing loader logic...")
from database.loader import _table_columns

table_cols = _table_columns(engine, "fact_daily_prices")
print(f"   Columns loader sees in fact_daily_prices:")
keep = table_cols - {"id", "created_at"}
print(f"   {sorted(keep)}")

# Simulate transformer output
transformer_cols = {'trade_date', 'symbol', 'series', 'security_name', 'isin',
                    'prev_close', 'open_price', 'high_price', 'low_price',
                    'close_price', 'net_traded_value', 'net_traded_qty',
                    'total_trades', 'is_index', 'is_valid'}

matched = [c for c in transformer_cols if c in keep]
print(f"\n   Columns that will be loaded: {len(matched)}")
print(f"   {sorted(matched)}")

missing_in_table = transformer_cols - keep
if missing_in_table:
    print(f"\n   ⚠️  Columns in transformer but NOT in table:")
    print(f"   {sorted(missing_in_table)}")

missing_in_transformer = keep - transformer_cols
if missing_in_transformer:
    print(f"\n   ⚠️  Columns in table but NOT in transformer:")
    print(f"   {sorted(missing_in_transformer)}")

print("\n" + "="*60)
print("  DIAGNOSIS COMPLETE")
print("="*60 + "\n")