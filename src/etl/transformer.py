"""
ETL Transformer – cleans and standardises raw NSE bhavcopy CSV files
into analysis-ready DataFrames.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import date
from typing import Dict, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns that should be numeric in the price files
PRICE_COLS = ["PREV_CL_PR", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
              "CLOSE_PRICE", "HI_52_WK", "LO_52_WK"]
VOLUME_COLS = ["NET_TRDVAL", "NET_TRDQTY", "TRADES"]


def _strip_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df


def _to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def transform_price_data(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """
    Clean and transform the PR / PD bhavcopy file.

    Adds:
        trade_date, symbol (if missing), day_change, day_change_pct,
        intraday_range, intraday_range_pct, is_index
    """
    df = _strip_df(df.copy())
    df = _to_numeric(df, PRICE_COLS + VOLUME_COLS)

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]

    rename = {
        "SECURITY": "security_name",
        "PREV_CL_PR": "prev_close",
        "OPEN_PRICE": "open_price",
        "HIGH_PRICE": "high_price",
        "LOW_PRICE": "low_price",
        "CLOSE_PRICE": "close_price",
        "NET_TRDVAL": "net_traded_value",
        "NET_TRDQTY": "net_traded_qty",
        "TRADES": "total_trades",
        "HI_52_WK": "high_52_week",
        "LO_52_WK": "low_52_week",
        "MKT": "market",
        "IND_SEC": "ind_sec",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["trade_date"] = pd.to_datetime(trade_date)

    # Derive symbol from SYMBOL column (PD file) or first word of security_name
    if "SYMBOL" in df.columns:
        df = df.rename(columns={"SYMBOL": "symbol"})
    else:
        df["symbol"] = df["security_name"].str.split().str[0]

    df["series"] = df.get("SERIES", pd.Series(["EQ"] * len(df)))

    # is_index flag
    df["is_index"] = (df.get("market", "N") == "Y") | \
                     (df.get("ind_sec", "N") == "Y")

    # Derived metrics
    df["day_change"] = df["close_price"] - df["prev_close"]
    df["day_change_pct"] = (df["day_change"] / df["prev_close"].replace(0, np.nan)) * 100

    df["intraday_range"] = df["high_price"] - df["low_price"]
    df["intraday_range_pct"] = (df["intraday_range"] / df["low_price"].replace(0, np.nan)) * 100

    # Validation – flag invalid rows instead of dropping
    df["is_valid"] = (
            (df["high_price"] >= df["low_price"]) &
            (df["close_price"] > 0) &
            (df["net_traded_qty"].fillna(0) >= 0)
    )

    invalid = (~df["is_valid"]).sum()
    if invalid:
        log.warning(f"transform_price_data: {invalid} invalid rows flagged")

    log.info(f"Transformed price data: {len(df)} rows, trade_date={trade_date}")
    return df


def transform_circuit_data(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Clean BH (price bands hit) file."""
    df = _strip_df(df.copy())
    df.columns = [c.strip().upper() for c in df.columns]

    rename = {
        "SYMBOL": "symbol",
        "SERIES": "series",
        "SECURITY": "security_name",
        "HIGH/LOW": "circuit_type",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(trade_date)
    df["circuit_type"] = df["circuit_type"].str.upper().str.strip()
    df = df[df["circuit_type"].isin(["H", "L"])]

    log.info(f"Transformed circuit data: {len(df)} rows")
    return df


def transform_corporate_actions(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Clean BC (corporate actions) file."""
    df = _strip_df(df.copy())
    df.columns = [c.strip().upper() for c in df.columns]

    rename = {
        "SYMBOL": "symbol",
        "SERIES": "series",
        "SECURITY": "security_name",
        "RECORD_DT": "record_date",
        "EX_DT": "ex_date",
        "PURPOSE": "purpose",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(trade_date)

    for col in ["record_date", "ex_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Categorise action
    def _categorise(p: str) -> str:
        p = str(p).upper()
        if "DIVIDEND" in p or "INTDIV" in p:  return "Dividend"
        if "BONUS" in p:                    return "Bonus"
        if "SPLIT" in p or "SPLT" in p:    return "Stock Split"
        if "RIGHTS" in p or "RGHTS" in p:   return "Rights Issue"
        if "BUYBACK" in p:                    return "Buyback"
        if "INTEREST" in p:                    return "Interest Payment"
        if "REDEMPTION" in p:                  return "Redemption"
        return "Other"

    df["action_type"] = df["purpose"].apply(_categorise)

    # Parse dividend amount
    df["dividend_amount"] = (
        df["purpose"].str.extract(r"RS\s*([\d.]+)", flags=re.IGNORECASE)[0]
        .astype(float)
    )

    log.info(f"Transformed corporate actions: {len(df)} rows")
    return df


def transform_etf_data(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Clean ETF file."""
    df = _strip_df(df.copy())
    df.columns = [c.strip().upper() for c in df.columns]

    rename = {
        "SYMBOL": "symbol",
        "SECURITY": "security_name",
        "PREVIOUS CLOSE PRICE": "prev_close",
        "OPEN PRICE": "open_price",
        "HIGH PRICE": "high_price",
        "LOW PRICE": "low_price",
        "CLOSE PRICE": "close_price",
        "NET TRADED VALUE": "net_traded_value",
        "NET TRADED QTY": "net_traded_qty",
        "TRADES": "total_trades",
        "52 WEEK HIGH": "high_52_week",
        "52 WEEK LOW": "low_52_week",
        "UNDERLYING": "underlying",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(trade_date)

    num_cols = ["prev_close", "open_price", "high_price", "low_price", "close_price",
                "net_traded_value", "net_traded_qty", "total_trades", "high_52_week", "low_52_week"]
    df = _to_numeric(df, num_cols)

    log.info(f"Transformed ETF data: {len(df)} rows")
    return df


def transform_market_cap(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Clean MCAP file."""
    df = _strip_df(df.copy())
    df.columns = [c.strip().upper() for c in df.columns]

    rename = {
        "SYMBOL": "symbol",
        "SERIES": "series",
        "SECURITY NAME": "security_name",
        "FACE VALUE(RS.)": "face_value",
        "ISSUE SIZE": "issue_size",
        "CLOSE PRICE/PAID UP VALUE(RS.)": "close_price",
        "MARKET CAP(RS.)": "market_cap",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(trade_date)

    for col in ["face_value", "issue_size", "close_price", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Transformed market cap data: {len(df)} rows")
    return df


def transform_hl_data(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Clean HL (52-week highs/lows) file."""
    df = _strip_df(df.copy())
    df.columns = [c.strip().upper() for c in df.columns]

    rename = {"SYMBOL": "symbol", "SERIES": "series",
              "SECURITY": "security_name", "HIGH/LOW": "hl_type"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(trade_date)
    df["hl_type"] = df["hl_type"].str.upper().str.strip()

    log.info(f"Transformed HL data: {len(df)} rows")
    return df


def transform_top_traded(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Clean TT (top 25 traded) file."""
    df = _strip_df(df.copy())
    df.columns = [c.strip().upper() for c in df.columns]
    df["trade_date"] = pd.to_datetime(trade_date)

    if "NET TRDVAL" in df.columns:
        df = df.rename(columns={"NET TRDVAL": "net_traded_value"})
    if "SECURITY" in df.columns:
        df = df.rename(columns={"SECURITY": "security_name"})

    log.info(f"Transformed top-traded data: {len(df)} rows")
    return df


def run_full_transform(raw_dir: Path, trade_date: date) -> Dict[str, pd.DataFrame]:
    """
    Transform all files found in raw_dir for the given trade_date.
    Handles both old (ddmmyy) and new (ddmmyyyy) NSE filename formats.
    Returns a dict keyed by dataset name.
    """
    # NSE used ddmmyy (6 digits) until ~2025, then switched to ddmmyyyy (8 digits)
    date_short = trade_date.strftime("%d%m%y")  # 130226
    date_long = trade_date.strftime("%d%m%Y")  # 13022026

    results = {}

    file_map = {
        "pr": ("price", transform_price_data),
        "pd": ("price_det", transform_price_data),
        "bh": ("circuits", transform_circuit_data),
        "bc": ("corp_act", transform_corporate_actions),
        "etf": ("etf", transform_etf_data),
        "mcap": ("mcap", transform_market_cap),
        "hl": ("hl", transform_hl_data),
        "tt": ("top_traded", transform_top_traded),
    }

    for prefix, (key, fn) in file_map.items():
        found = None

        # Try all known NSE filename patterns
        candidates = [
            raw_dir / f"{prefix}{date_long}1.csv",  # NEW: pr130220261.csv (2026+)
            raw_dir / f"{prefix}{date_long}.csv",  # NEW: pr13022026.csv
            raw_dir / f"{prefix}{date_short}1.csv",  # OLD: pr1302261.csv
            raw_dir / f"{prefix}{date_short}.csv",  # OLD: pr130226.csv
            raw_dir / f"{prefix}{date_long}_1.csv",  # Variant with underscore
            raw_dir / f"{prefix}{date_short}_1.csv",  # Variant
        ]

        # Also try glob as fallback
        glob_patterns = [
            f"{prefix}{date_long}*.csv",
            f"{prefix}{date_short}*.csv",
        ]

        for path in candidates:
            if path.exists():
                found = path
                break

        if not found:
            for pattern in glob_patterns:
                matches = list(raw_dir.glob(pattern))
                if matches:
                    found = matches[0]
                    break

        if found:
            try:
                log.info(f"Reading {found.name}")
                df = pd.read_csv(found, dtype=str)
                results[key] = fn(df, trade_date)
            except Exception as e:
                log.error(f"Failed to transform {found.name}: {e}")
        else:
            log.debug(f"File not found for prefix '{prefix}' date {trade_date}")

    log.info(f"Transformation complete. Datasets: {list(results.keys())}")
    return results