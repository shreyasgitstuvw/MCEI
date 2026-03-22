"""
NSE UDIFF Bhavcopy Transformer
===============================
For the new unified format introduced in July 2024.

Single CSV contains ALL instrument types:
- STK (Stocks): Series EQ, BE, BZ, ST, etc.
- ETF (Exchange Traded Funds)
- BOND (Government & Corporate Bonds)
- IDX (Indices)

Filename pattern: BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv
Example: BhavCopy_NSE_CM_0_0_0_20260204_F_0000.csv
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from typing import Dict

# Column mapping: UDIFF → Our schema
UDIFF_COL_MAP = {
    "TradDt":           "trade_date",
    "BizDt":            "biz_date",
    "Sgmt":             "segment",
    "Src":              "source",
    "FinInstrmTp":      "instrument_type",   # STK, ETF, BOND, IDX
    "FinInstrmId":      "instrument_id",
    "ISIN":             "isin",
    "TckrSymb":         "symbol",
    "SctySrs":          "series",            # EQ, BE, BZ, ST, GB, etc.
    "FinInstrmNm":      "security_name",
    "OpnPric":          "open_price",
    "HghPric":          "high_price",
    "LwPric":           "low_price",
    "ClsPric":          "close_price",
    "LastPric":         "last_price",
    "PrvsClsgPric":     "prev_close",
    "TtlTradgVol":      "net_traded_qty",
    "TtlTrfVal":        "net_traded_value",
    "TtlNbOfTxsExctd":  "total_trades",
}

# Equity series (for price data)
EQUITY_SERIES = {"EQ", "BE", "BZ", "BL", "GC", "IL", "SM", "ST"}

# Special series
BOND_SERIES = {"GB", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"}
INDEX_TYPES = {"IDX"}


def _clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to numeric, coercing errors to NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all string columns."""
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Public transformer
# ══════════════════════════════════════════════════════════════════════════════

def transform_udiff_bhavcopy(csv_path: Path, trade_date: date) -> Dict[str, pd.DataFrame]:
    """
    Transform the NSE UDIFF unified bhavcopy CSV into multiple datasets.
    
    Returns:
        {
            "price":      Equity price data,
            "indices":    Index values,
            "etf":        ETF data,
            "bonds":      Bond data (gold bonds, etc.)
        }
    """
    
    # ── 1. Load raw CSV ───────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, dtype=str)
    df = _strip_strings(df)
    
    # Rename columns to our schema
    df = df.rename(columns=UDIFF_COL_MAP)
    
    # Ensure trade_date column
    df["trade_date"] = pd.to_datetime(trade_date)
    
    # Clean numeric columns
    numeric_cols = [
        "open_price", "high_price", "low_price", "close_price",
        "last_price", "prev_close", "net_traded_qty",
        "net_traded_value", "total_trades"
    ]
    df = _clean_numeric(df, numeric_cols)
    
    results = {}
    
    # ── 2. Extract Equity Prices ──────────────────────────────────────────────
    equity_mask = (df["instrument_type"] == "STK") & (df["series"].isin(EQUITY_SERIES))
    equity_df   = df[equity_mask].copy()
    
    if not equity_df.empty:
        # Add derived columns
        equity_df["day_change"]     = equity_df["close_price"] - equity_df["prev_close"]
        equity_df["day_change_pct"] = (
            (equity_df["day_change"] / equity_df["prev_close"].replace(0, np.nan)) * 100
        )
        equity_df["intraday_range"]     = equity_df["high_price"] - equity_df["low_price"]
        equity_df["intraday_range_pct"] = (
            (equity_df["intraday_range"] / equity_df["low_price"].replace(0, np.nan)) * 100
        )
        
        # Validation
        equity_df["is_valid"] = (
            (equity_df["high_price"] >= equity_df["low_price"]) &
            (equity_df["close_price"] > 0) &
            (equity_df["net_traded_qty"].fillna(0) >= 0)
        )
        
        equity_df["is_index"] = False
        
        # Select final columns for DB
        keep_cols = [
            "trade_date", "symbol", "series", "security_name", "isin",
            "prev_close", "open_price", "high_price", "low_price", "close_price",
            "net_traded_value", "net_traded_qty", "total_trades",
            "is_index", "is_valid"
        ]
        results["price"] = equity_df[[c for c in keep_cols if c in equity_df.columns]]
    
    # ── 3. Extract ETFs ───────────────────────────────────────────────────────
    etf_mask = (df["instrument_type"] == "ETF") | (
        (df["instrument_type"] == "STK") & 
        (df["symbol"].str.contains("BEES|GOLD|SILVER", case=False, na=False))
    )
    etf_df = df[etf_mask].copy()
    
    if not etf_df.empty:
        keep_cols = [
            "trade_date", "symbol", "security_name", "isin",
            "prev_close", "open_price", "high_price", "low_price", "close_price",
            "net_traded_value", "net_traded_qty", "total_trades"
        ]
        results["etf"] = etf_df[[c for c in keep_cols if c in etf_df.columns]]
    
    # ── 4. Extract Indices ────────────────────────────────────────────────────
    # Indices are typically in a separate file in UDIFF, but if present:
    idx_mask = df["instrument_type"].isin(INDEX_TYPES)
    idx_df   = df[idx_mask].copy()
    
    if not idx_df.empty:
        idx_df["is_index"] = True
        keep_cols = [
            "trade_date", "symbol", "security_name",
            "open_price", "high_price", "low_price", "close_price", "prev_close",
            "is_index"
        ]
        results["indices"] = idx_df[[c for c in keep_cols if c in idx_df.columns]]
    
    # ── 5. Extract Bonds ──────────────────────────────────────────────────────
    bond_mask = (
        (df["instrument_type"] == "BOND") |
        (df["series"].isin(BOND_SERIES)) |
        (df["symbol"].str.contains("SGB|GOLDBOND", case=False, na=False))
    )
    bond_df = df[bond_mask].copy()
    
    if not bond_df.empty:
        keep_cols = [
            "trade_date", "symbol", "series", "security_name", "isin",
            "open_price", "high_price", "low_price", "close_price", "prev_close",
            "net_traded_value", "net_traded_qty", "total_trades"
        ]
        results["bonds"] = bond_df[[c for c in keep_cols if c in bond_df.columns]]
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Discovery helper
# ══════════════════════════════════════════════════════════════════════════════

def find_udiff_file(directory: Path, trade_date: date) -> Path | None:
    """
    Find the UDIFF bhavcopy CSV for the given date.
    Tries multiple filename patterns.
    """
    date_str = trade_date.strftime("%Y%m%d")
    
    patterns = [
        f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv",
        f"BhavCopy_NSE_CM_0_0_0_{date_str}*.csv",
        f"*{date_str}*.csv",
    ]
    
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    
    return None