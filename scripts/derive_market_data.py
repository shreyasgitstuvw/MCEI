"""
Derive proxy data for empty DB tables from fact_daily_prices.

Tables populated:
  fact_circuit_hits  — stocks whose close is within 0.1% of day high/low with a
                       price move ≥ 2%.  This is a heuristic; no official NSE
                       circuit-breaker feed is available.
  fact_hl_hits       — stocks making a new rolling 52-week high or low based on
                       the available history in fact_daily_prices (up to 365 days).

Usage:
    python scripts/derive_market_data.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]
                                         [--only circuits|hl]
"""
from __future__ import annotations
import os, sys, argparse
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import text

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from database.connection import get_engine


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_prices_range(engine, from_date: date, to_date: date) -> pd.DataFrame:
    sql = """
        SELECT symbol, series, security_name, trade_date,
               open_price, high_price, low_price, close_price, prev_close,
               net_traded_qty, net_traded_value, total_trades, is_index
        FROM fact_daily_prices
        WHERE trade_date BETWEEN :f AND :t
          AND is_index = false
        ORDER BY symbol, trade_date
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"f": from_date, "t": to_date})


def _upsert_circuits(engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = text("""
        INSERT INTO fact_circuit_hits
            (trade_date, symbol, series, security_name, circuit_type,
             prev_close, high_price, low_price, close_price, net_traded_qty,
             net_traded_value, total_trades)
        VALUES
            (:trade_date, :symbol, :series, :security_name, :circuit_type,
             :prev_close, :high_price, :low_price, :close_price, :net_traded_qty,
             :net_traded_value, :total_trades)
        ON CONFLICT (trade_date, symbol, series) DO UPDATE SET
            circuit_type     = EXCLUDED.circuit_type,
            close_price      = EXCLUDED.close_price,
            net_traded_qty   = EXCLUDED.net_traded_qty,
            net_traded_value = EXCLUDED.net_traded_value
    """)
    records = df.to_dict("records")
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def _upsert_hl(engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = text("""
        INSERT INTO fact_hl_hits
            (trade_date, symbol, series, security_name,
             close_price, high_52_week, low_52_week, hit_type,
             net_traded_qty, net_traded_value)
        VALUES
            (:trade_date, :symbol, :series, :security_name,
             :close_price, :high_52_week, :low_52_week, :hit_type,
             :net_traded_qty, :net_traded_value)
        ON CONFLICT (trade_date, symbol, series) DO UPDATE SET
            hit_type       = EXCLUDED.hit_type,
            close_price    = EXCLUDED.close_price,
            high_52_week   = EXCLUDED.high_52_week,
            low_52_week    = EXCLUDED.low_52_week
    """)
    records = df.to_dict("records")
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


# ── circuit proxy ─────────────────────────────────────────────────────────────

def derive_circuits(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic: a stock is considered at upper circuit if
      - close is within 0.1% of the day high, AND
      - price moved ≥ 2% from prev_close (typical NSE lower circuit limit is 2%+).
    Lower circuit: close within 0.1% of day low AND move ≤ -2%.
    """
    df = prices.copy()
    df["range"]   = df["high_price"] - df["low_price"]
    df["move_pct"] = (df["close_price"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan) * 100

    # Upper circuit
    df["dist_to_high"] = (df["high_price"] - df["close_price"]) / df["high_price"].replace(0, np.nan) * 100
    df["dist_to_low"]  = (df["close_price"] - df["low_price"])  / df["close_price"].replace(0, np.nan) * 100

    uc = df[(df["dist_to_high"] <= 0.1) & (df["move_pct"] >= 2.0)].copy()
    uc["circuit_type"] = "H"

    lc = df[(df["dist_to_low"]  <= 0.1) & (df["move_pct"] <= -2.0)].copy()
    lc["circuit_type"] = "L"

    result = pd.concat([uc, lc], ignore_index=True)
    keep = ["trade_date","symbol","series","security_name","circuit_type",
            "prev_close","high_price","low_price","close_price",
            "net_traded_qty","net_traded_value","total_trades"]
    return result[[c for c in keep if c in result.columns]]


# ── HL hits ───────────────────────────────────────────────────────────────────

def derive_hl_hits(prices: pd.DataFrame, window_days: int = 252) -> pd.DataFrame:
    """
    For each trading date, flag stocks making a new rolling N-day high or low.
    Uses whatever history is available (up to window_days).
    """
    df = prices.sort_values(["symbol","trade_date"]).copy()
    df["rolling_high"] = (df.groupby("symbol")["close_price"]
                            .transform(lambda s: s.expanding(min_periods=2).max()
                                                   .shift(1)))
    df["rolling_low"]  = (df.groupby("symbol")["close_price"]
                            .transform(lambda s: s.expanding(min_periods=2).min()
                                                   .shift(1)))

    new_high = df[df["close_price"] >= df["rolling_high"]].copy()
    new_high["hit_type"]    = "HIGH"
    new_high["high_52_week"] = new_high["rolling_high"]
    new_high["low_52_week"]  = new_high["rolling_low"]

    new_low = df[df["close_price"] <= df["rolling_low"]].copy()
    new_low["hit_type"]    = "LOW"
    new_low["high_52_week"] = new_low["rolling_high"]
    new_low["low_52_week"]  = new_low["rolling_low"]

    result = pd.concat([new_high, new_low], ignore_index=True).dropna(subset=["rolling_high"])
    keep = ["trade_date","symbol","series","security_name",
            "close_price","high_52_week","low_52_week","hit_type",
            "net_traded_qty","net_traded_value"]
    return result[[c for c in keep if c in result.columns]]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Derive proxy market data from fact_daily_prices")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (default: earliest in DB)")
    parser.add_argument("--to",   dest="to_date",   default=None,
                        help="End date   YYYY-MM-DD (default: today)")
    parser.add_argument("--only", choices=["circuits","hl"], default=None,
                        help="Populate only one table")
    args = parser.parse_args()

    engine = get_engine()

    # Determine date range
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MIN(trade_date), MAX(trade_date) FROM fact_daily_prices")).fetchone()
    min_date, max_date = row[0], row[1]
    if min_date is None:
        print("fact_daily_prices is empty — run the ETL first.")
        return

    # Load extra history for rolling window (up to 1 year before from_date)
    from_date = date.fromisoformat(args.from_date) if args.from_date else min_date
    to_date   = date.fromisoformat(args.to_date)   if args.to_date   else max_date
    history_from = from_date - timedelta(days=365)

    print(f"Loading prices {history_from} → {to_date} ...")
    prices = _load_prices_range(engine, history_from, to_date)
    print(f"  {len(prices):,} rows loaded ({prices['symbol'].nunique()} symbols, "
          f"{prices['trade_date'].nunique()} dates)")

    if not args.only or args.only == "circuits":
        # Only process dates in the requested range
        target_prices = prices[prices["trade_date"] >= pd.Timestamp(from_date)]
        circuits = derive_circuits(target_prices)
        print(f"\nCircuit proxy: {len(circuits)} rows ({(circuits['circuit_type']=='H').sum()} upper, "
              f"{(circuits['circuit_type']=='L').sum()} lower)")
        n = _upsert_circuits(engine, circuits)
        print(f"  Upserted {n} rows into fact_circuit_hits")

    if not args.only or args.only == "hl":
        hl = derive_hl_hits(prices)
        target_hl = hl[hl["trade_date"] >= pd.Timestamp(from_date)]
        print(f"\nHL hits: {len(target_hl)} rows ({(target_hl['hit_type']=='HIGH').sum()} highs, "
              f"{(target_hl['hit_type']=='LOW').sum()} lows)")
        n = _upsert_hl(engine, target_hl)
        print(f"  Upserted {n} rows into fact_hl_hits")

    print("\nDone.")


if __name__ == "__main__":
    main()
