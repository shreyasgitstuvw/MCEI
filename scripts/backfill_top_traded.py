"""
Backfill fact_top_traded from existing fact_daily_prices.

For each trading date in fact_daily_prices, computes the top 25 stocks
by net_traded_value and inserts into fact_top_traded.

Usage:
    python scripts/backfill_top_traded.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine


def backfill_top_traded(top_n: int = 25) -> None:
    engine = get_engine()
    with engine.connect() as conn:
        dates = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT trade_date FROM fact_daily_prices ORDER BY trade_date"
        )).fetchall()]
        print(f"Found {len(dates)} trading dates to backfill")

    total = 0
    for trade_date in dates:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT trade_date, security_name, net_traded_value "
                "FROM fact_daily_prices "
                "WHERE trade_date = :d "
                "  AND is_index = FALSE "
                "  AND net_traded_value IS NOT NULL "
                "  AND net_traded_value > 0 "
                "ORDER BY net_traded_value DESC "
                f"LIMIT {top_n}"
            ), {"d": trade_date}).fetchall()

        if not rows:
            print(f"  {trade_date}: no data, skipping")
            continue

        df = pd.DataFrame(rows, columns=["trade_date", "security_name", "net_traded_value"])
        df["rank"] = range(1, len(df) + 1)
        df = df[["trade_date", "rank", "security_name", "net_traded_value"]]

        records = df.to_dict("records")
        for r in records:
            r["net_traded_value"] = float(r["net_traded_value"]) if r["net_traded_value"] is not None else None

        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fact_top_traded WHERE trade_date = :d"), {"d": trade_date})
            conn.execute(
                text("INSERT INTO fact_top_traded (trade_date, rank, security_name, net_traded_value) "
                     "VALUES (:trade_date, :rank, :security_name, :net_traded_value)"),
                records,
            )
        total += len(records)
        print(f"  {trade_date}: inserted {len(records)} rows")

    print(f"\nDone. Total rows inserted: {total}")


if __name__ == "__main__":
    backfill_top_traded()
