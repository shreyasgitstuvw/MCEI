"""
Clean Database for Testing
===========================
Remove existing data for 2026-02-23 so we can test fresh insert.
"""

from sqlalchemy import text
from src.database.connection import get_engine

def clean_data():
    engine = get_engine()
    
    print("\n🧹 Cleaning existing data for 2026-02-23...\n")
    
    with engine.begin() as conn:
        result = conn.execute(text("""
            DELETE FROM fact_daily_prices 
            WHERE trade_date = '2026-02-23'
        """))
        
        print(f"  ✅ Deleted {result.rowcount} rows\n")

if __name__ == "__main__":
    clean_data()