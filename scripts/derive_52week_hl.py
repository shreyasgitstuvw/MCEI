"""
Derive and backfill 52-week data in fact_daily_prices and fact_hl_hits.

Step 1: Compute rolling 252-trading-day high_52_week / low_52_week for every
        row in fact_daily_prices where those columns are NULL.

Step 2: Populate fact_hl_hits — a symbol "hits" a 52-week high on a date when
        its close_price equals its rolling 252-day max (i.e. it IS the new high),
        and symmetrically for lows.  This eliminates the NSE live-API dependency
        for fact_hl_hits entirely.

Usage:
    python scripts/derive_52week_hl.py            # both steps
    python scripts/derive_52week_hl.py --hl-only  # Step 1 only
    python scripts/derive_52week_hl.py --hits-only # Step 2 only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database.connection import get_engine
from sqlalchemy import text

UPDATE_SQL = """
WITH ranked AS (
    SELECT
        ctid,
        symbol,
        trade_date,
        MAX(close_price) OVER (
            PARTITION BY symbol
            ORDER BY trade_date
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        ) AS h52,
        MIN(close_price) OVER (
            PARTITION BY symbol
            ORDER BY trade_date
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        ) AS l52
    FROM fact_daily_prices
    WHERE close_price IS NOT NULL
)
UPDATE fact_daily_prices fp
SET
    high_52_week = r.h52,
    low_52_week  = r.l52
FROM ranked r
WHERE fp.ctid = r.ctid
  AND (fp.high_52_week IS NULL OR fp.low_52_week IS NULL);
"""

# Insert into fact_hl_hits for any (symbol, date) where:
#   H hit: close_price = high_52_week  (today's close IS the rolling 252-day max)
#   L hit: close_price = low_52_week   (today's close IS the rolling 252-day min)
# Skip dates already present to make this idempotent.
HITS_SQL = """
INSERT INTO fact_hl_hits (trade_date, symbol, series, security_name, hl_type, price, prev_high_low)
SELECT
    fp.trade_date,
    fp.symbol,
    fp.series,
    fp.security_name,
    hit.hl_type,
    fp.close_price                              AS price,
    CASE hit.hl_type
        WHEN 'H' THEN fp.high_52_week
        ELSE           fp.low_52_week
    END                                         AS prev_high_low
FROM fact_daily_prices fp
CROSS JOIN (VALUES ('H'), ('L')) AS hit(hl_type)
WHERE fp.high_52_week IS NOT NULL
  AND fp.low_52_week  IS NOT NULL
  AND fp.series IN ('EQ', 'BE', 'BZ')          -- equity series only
  AND (
        (hit.hl_type = 'H' AND fp.close_price = fp.high_52_week)
     OR (hit.hl_type = 'L' AND fp.close_price = fp.low_52_week)
  )
  AND NOT EXISTS (
        SELECT 1 FROM fact_hl_hits h
        WHERE h.trade_date = fp.trade_date
          AND h.symbol     = fp.symbol
          AND h.hl_type    = hit.hl_type
  );
"""


def step1_derive_hl(engine) -> int:
    """Compute rolling 52W H/L and backfill NULL cells in fact_daily_prices."""
    print("\n[Step 1] Deriving 52-week high/low from existing price history ...")

    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT COUNT(*) FROM fact_daily_prices WHERE high_52_week IS NULL"
        )).fetchone()
        null_before = row[0]
        print(f"  Rows with NULL high_52_week: {null_before:,}")

        if null_before == 0:
            print("  Already fully populated — nothing to do.")
            return 0

        result = conn.execute(text(UPDATE_SQL))
        updated = result.rowcount
        print(f"  Rows updated: {updated:,}")

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT COUNT(*) FROM fact_daily_prices WHERE high_52_week IS NULL"
        )).fetchone()
        print(f"  Rows still NULL: {row[0]:,}")

    return updated


def step2_populate_hits(engine) -> int:
    """Insert into fact_hl_hits based on computed high_52_week / low_52_week."""
    print("\n[Step 2] Populating fact_hl_hits from computed 52W values ...")

    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM fact_hl_hits")).fetchone()[0]
    print(f"  Rows in fact_hl_hits before: {before:,}")

    with engine.begin() as conn:
        result = conn.execute(text(HITS_SQL))
        inserted = result.rowcount
    print(f"  New rows inserted: {inserted:,}")

    with engine.connect() as conn:
        after = conn.execute(text("SELECT COUNT(*) FROM fact_hl_hits")).fetchone()[0]
        sample = conn.execute(text("""
            SELECT hl_type, COUNT(*) AS n
            FROM fact_hl_hits
            GROUP BY hl_type
            ORDER BY hl_type
        """)).fetchall()
    print(f"  Total rows after:  {after:,}")
    for hl_type, n in sample:
        label = "52W Highs" if hl_type == "H" else "52W Lows"
        print(f"    {label}: {n:,}")

    return inserted


def main():
    parser = argparse.ArgumentParser(description="Derive 52W H/L and populate hl_hits")
    parser.add_argument("--hl-only",   action="store_true", help="Step 1 only (update fact_daily_prices)")
    parser.add_argument("--hits-only", action="store_true", help="Step 2 only (populate fact_hl_hits)")
    args = parser.parse_args()

    run_step1 = not args.hits_only
    run_step2 = not args.hl_only

    engine = get_engine()

    if run_step1:
        step1_derive_hl(engine)

    if run_step2:
        step2_populate_hits(engine)

    # Sanity check sample
    if run_step1:
        with engine.connect() as conn:
            sample = conn.execute(text("""
                SELECT symbol, trade_date, close_price, high_52_week, low_52_week
                FROM fact_daily_prices
                WHERE series = 'EQ'
                  AND high_52_week IS NOT NULL
                ORDER BY trade_date DESC, symbol
                LIMIT 5
            """)).fetchall()
        if sample:
            print("\nSample fact_daily_prices (latest 5):")
            print(f"  {'Symbol':<15} {'Date':<12} {'Close':>10} {'52W High':>10} {'52W Low':>10}")
            for row in sample:
                sym, dt, close, h52, l52 = row
                print(f"  {sym:<15} {str(dt):<12} {close:>10.2f} {h52:>10.2f} {l52:>10.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
