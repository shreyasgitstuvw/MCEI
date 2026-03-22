"""
Check Database Schema
=====================
Find the exact column definitions to understand constraints.
"""

from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv
import os

load_dotenv()

def check_schema():
    # Get database connection
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    
    print(f"\n{'='*70}")
    print(f"  Database Schema for fact_daily_prices")
    print(f"{'='*70}\n")
    
    # Get column info
    inspector = inspect(engine)
    columns = inspector.get_columns('fact_daily_prices')
    
    print(f"Column definitions:\n")
    for col in columns:
        col_name = col['name']
        col_type = str(col['type'])
        nullable = "NULL" if col['nullable'] else "NOT NULL"
        print(f"  {col_name:20s} {col_type:20s} {nullable}")
    
    # Check for any CHECK constraints
    print(f"\n{'='*70}")
    print(f"  Checking for constraints...")
    print(f"{'='*70}\n")
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT conname, pg_get_constraintdef(oid) 
            FROM pg_constraint 
            WHERE conrelid = 'fact_daily_prices'::regclass
        """))
        
        for row in result:
            print(f"  {row[0]}: {row[1]}")
    
    print(f"\n{'='*70}")
    print(f"  Testing sample insert with extreme values...")
    print(f"{'='*70}\n")
    
    # Try inserting a test row with various values
    test_values = [
        ('TEST1', 1, 1, 1, 1),  # Tiny values
        ('TEST2', 1000000, 100000000, 100000, 50000),  # Medium values  
        ('TEST3', 999999999, 99999999999, 999999, 999999),  # Large values
    ]
    
    for symbol, qty, value, trades, del_qty in test_values:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO fact_daily_prices 
                    (trade_date, symbol, series, security_name, close_price, 
                     net_traded_qty, net_traded_value, total_trades, delivery_qty, 
                     is_index, is_valid)
                    VALUES 
                    ('2026-01-01', :sym, 'EQ', 'Test', 100.0,
                     :qty, :val, :trades, :del_qty,
                     false, true)
                """), {
                    'sym': symbol,
                    'qty': qty,
                    'val': value,
                    'trades': trades,
                    'del_qty': del_qty
                })
            print(f"  ✅ {symbol}: qty={qty:,}, value={value:,}, trades={trades:,}, del_qty={del_qty:,}")
            
            # Clean up
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM fact_daily_prices WHERE symbol = :sym"), {'sym': symbol})
                
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)[:100]}")
    
    print()


if __name__ == "__main__":
    check_schema()