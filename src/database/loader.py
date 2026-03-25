"""
Database Loader
Upserts transformed DataFrames into PostgreSQL.
Re-runnable: ON CONFLICT DO UPDATE means running twice is safe.
"""

from __future__ import annotations
import os
import time
from datetime import date
from typing import Dict

import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# reuse the engine from connection.py
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.connection import get_engine


# ── helpers ───────────────────────────────────────────────────────────────────

def _table_columns(engine, table: str) -> set:
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=:t"
        ), {"t": table}).fetchall()
    return {r[0] for r in rows}


def _upsert(df: pd.DataFrame, table: str,
            conflict_cols: list[str], engine) -> int:
    """Bulk upsert. Returns rows written."""
    if df is None or df.empty:
        return 0

    # Only keep columns that exist in the DB table
    keep = _table_columns(engine, table) - {"id", "created_at"}
    df = df[[c for c in df.columns if c in keep]].copy()
    if df.empty:
        return 0

    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(
        f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict_cols
    )
    conflict_str = ", ".join(conflict_cols)

    sql = text(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_str}) DO UPDATE SET {updates}"
    )

    # Convert DataFrame to dict, replacing NaN with None
    records = df.where(pd.notna(df), other=None).to_dict("records")

    # CRITICAL: Convert types for PostgreSQL compatibility
    # 1. psycopg2 can't handle numpy types (numpy.int64, numpy.float64)
    # 2. PostgreSQL BIGINT columns reject Python float (even whole numbers like 33003.0)
    import numpy as np

    # Columns that are BIGINT in database (must be integers)
    bigint_columns = {'net_traded_qty', 'total_trades', 'delivery_qty'}

    for record in records:
        for key, value in record.items():
            if value is not None and not isinstance(value, bool):  # Skip None and bool
                # Handle numpy integers
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    record[key] = int(value)
                # Handle numpy floats AND Python floats
                elif isinstance(value, (np.floating, np.float64, np.float32, float)):
                    if value != value:  # NaN check (NaN != NaN is always True)
                        record[key] = None
                    elif key in bigint_columns:
                        # For BIGINT columns, convert to int (PostgreSQL rejects floats for BIGINT)
                        record[key] = int(value)
                    else:
                        # For NUMERIC/DECIMAL columns, keep as Python float
                        record[key] = float(value)

    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def _log(engine, trade_date: date, dataset: str,
         status: str, n: int, err: str = None, secs: int = 0):
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO log_ingestion "
                "(trade_date,file_name,status,records_processed,"
                "error_message,processing_seconds) "
                "VALUES (:d,:f,:s,:r,:e,:t)"
            ), dict(d=trade_date, f=dataset, s=status, r=n, e=err, t=secs))
    except Exception:
        pass  # never let logging crash the pipeline


# ── individual loaders ────────────────────────────────────────────────────────

def load_prices(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    # If security_name is absent the DataFrame only has delivery/numeric updates
    # (from full bhavcopy).  Use a targeted UPDATE to avoid NOT NULL violation.
    if 'security_name' not in df.columns:
        return _update_delivery(df, trade_date, e)
    n = _upsert(df, "fact_daily_prices", ["symbol", "trade_date"], e)
    _log(e, trade_date, "price", "SUCCESS", n)
    return n


def _update_delivery(df: pd.DataFrame, trade_date: date, engine) -> int:
    """UPDATE delivery_qty / delivery_pct for rows that already exist."""
    import numpy as np
    delivery_cols = [c for c in ('delivery_qty', 'delivery_pct') if c in df.columns]
    if not delivery_cols:
        return 0
    set_clause = ", ".join(f"{c}=:{c}" for c in delivery_cols)
    sql = text(
        f"UPDATE fact_daily_prices SET {set_clause} "
        "WHERE symbol=:symbol AND trade_date=:trade_date"
    )
    records = df[['symbol', 'trade_date'] + delivery_cols].where(
        pd.notna(df[['symbol', 'trade_date'] + delivery_cols]), other=None
    ).to_dict('records')
    for rec in records:
        for k, v in rec.items():
            if v is not None and not isinstance(v, bool):
                if isinstance(v, (np.integer, np.int64, np.int32)):
                    rec[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32, float)):
                    if v != v:  # NaN check
                        rec[k] = None
                    elif k == 'delivery_qty':
                        rec[k] = int(v)
                    else:
                        rec[k] = float(v)
    with engine.begin() as conn:
        conn.execute(sql, records)
    _log(engine, trade_date, "delivery", "SUCCESS", len(records))
    return len(records)


def load_circuits(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    n = _upsert(df, "fact_circuit_hits",
                ["symbol", "trade_date", "circuit_type"], e)
    _log(e, trade_date, "circuits", "SUCCESS", n)
    return n


def load_corporate_actions(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    # no single unique key — replace today's rows atomically
    keep = _table_columns(e, "fact_corporate_actions") - {"id", "created_at"}
    out = df[[col for col in df.columns if col in keep]].copy()
    records = out.where(pd.notna(out), other=None).to_dict("records")
    with e.begin() as c:
        c.execute(text(
            "DELETE FROM fact_corporate_actions WHERE trade_date=:d"
        ), {"d": trade_date})
        if records:
            c.execute(text(
                "INSERT INTO fact_corporate_actions "
                f"({', '.join(out.columns)}) "
                f"VALUES ({', '.join(':' + col for col in out.columns)})"
            ), records)
    _log(e, trade_date, "corp_actions", "SUCCESS", len(records))
    return len(records)


def load_etf(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    n = _upsert(df, "fact_etf_prices", ["symbol", "trade_date"], e)
    _log(e, trade_date, "etf", "SUCCESS", n)
    return n


def load_mcap(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    n = _upsert(df, "fact_market_cap", ["symbol", "trade_date"], e)
    _log(e, trade_date, "mcap", "SUCCESS", n)
    return n


def load_hl(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    keep = _table_columns(e, "fact_hl_hits") - {"id", "created_at"}
    out = df[[col for col in df.columns if col in keep]].copy()
    records = out.where(pd.notna(out), other=None).to_dict("records")
    with e.begin() as c:
        c.execute(text(
            "DELETE FROM fact_hl_hits WHERE trade_date=:d"
        ), {"d": trade_date})
        if records:
            c.execute(text(
                f"INSERT INTO fact_hl_hits ({', '.join(out.columns)}) "
                f"VALUES ({', '.join(':' + col for col in out.columns)})"
            ), records)
    _log(e, trade_date, "hl", "SUCCESS", len(records))
    return len(records)


def load_top_traded(df: pd.DataFrame, trade_date: date) -> int:
    e = get_engine()
    keep = _table_columns(e, "fact_top_traded") - {"id", "created_at"}
    out = df[[col for col in df.columns if col in keep]].copy()
    records = out.where(pd.notna(out), other=None).to_dict("records")
    with e.begin() as c:
        c.execute(text(
            "DELETE FROM fact_top_traded WHERE trade_date=:d"
        ), {"d": trade_date})
        if records:
            c.execute(text(
                f"INSERT INTO fact_top_traded ({', '.join(out.columns)}) "
                f"VALUES ({', '.join(':' + col for col in out.columns)})"
            ), records)
    _log(e, trade_date, "top_traded", "SUCCESS", len(records))
    return len(records)


# ── bulk entry point ──────────────────────────────────────────────────────────

LOADER_MAP = {
    "price": load_prices,
    "price_det": load_prices,
    "full_bhavcopy": load_prices,  # NEW: Full bhavcopy goes to prices
    "nifty50": load_prices,  # NEW: Nifty 50 constituents go to prices
    "circuits": load_circuits,
    "corp_act": load_corporate_actions,
    "corporate_actions": load_corporate_actions,  # NEW: Alternative name
    "etf": load_etf,
    "mcap": load_mcap,
    "hl": load_hl,
    "hl_hits": load_hl,  # NEW: 52-week high/low hits
    "hl_high": load_hl,  # individual file fallback (combined by multi_transformer normally)
    "hl_low":  load_hl,  # individual file fallback
    "top_traded": load_top_traded,
}


def load_all(datasets: Dict[str, pd.DataFrame],
             engine=None) -> Dict[str, int]:
    """Load every dataset. Returns {table_name: rows_written}."""
    if engine is None:
        engine = get_engine()

    results = {}

    # Get the trade_date from the first non-empty dataset
    trade_date = None
    for df in datasets.values():
        if df is not None and not df.empty and 'trade_date' in df.columns:
            trade_date = df['trade_date'].iloc[0]
            break

    if trade_date is None:
        print("  ⚠️  No trade_date found in datasets")
        return results

    # Dataset keys intentionally without a loader (no DB table for them yet)
    _silently_skip = {'bonds', 'indices'}

    for key, df in datasets.items():
        fn = LOADER_MAP.get(key)
        if fn is None:
            if key not in _silently_skip:
                print(f"  ⚠️  No loader for dataset: {key}")
            continue
        if df is None or df.empty:
            continue

        t0 = time.time()
        try:
            n = fn(df, trade_date)

            # Map dataset key to table name for results
            if key in ['full_bhavcopy', 'nifty50', 'price', 'price_det']:
                table_name = 'fact_daily_prices'
            elif key in ['corporate_actions', 'corp_act']:
                table_name = 'fact_corporate_actions'
            elif key in ['hl_hits', 'hl']:
                table_name = 'fact_hl_hits'
            elif key == 'etf':
                table_name = 'fact_etf_prices'
            elif key == 'circuits':
                table_name = 'fact_circuit_hits'
            elif key == 'mcap':
                table_name = 'fact_market_cap'
            elif key == 'top_traded':
                table_name = 'fact_top_traded'
            else:
                table_name = key

            # Accumulate rows for same table
            if table_name in results:
                results[table_name] += n
            else:
                results[table_name] = n

        except Exception as exc:
            print(f"  ❌ Failed to load {key}: {exc}")
            _log(engine, trade_date, key, "FAILED", 0,
                 str(exc), int(time.time() - t0))

    return results