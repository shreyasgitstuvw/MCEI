"""
Shared pytest fixtures for NSE analytics platform tests.
"""
from __future__ import annotations
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# Make src/ importable
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


# ── Date helpers ──────────────────────────────────────────────────────────────

BASE_DATE = date(2026, 2, 3)
DATES = [BASE_DATE + timedelta(days=i) for i in range(20)]
SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]


# ── Core price fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def price_df() -> pd.DataFrame:
    """
    Minimal multi-symbol, multi-date price DataFrame matching fact_daily_prices schema.
    20 rows × 5 symbols = 100 rows.
    """
    rng = np.random.default_rng(42)
    rows = []
    for sym in SYMBOLS:
        base_price = {"RELIANCE": 1200.0, "TCS": 3500.0, "INFY": 1800.0,
                      "HDFC": 1600.0, "ICICIBANK": 900.0}[sym]
        for d in DATES:
            close = base_price * (1 + rng.normal(0, 0.01))
            high = close * (1 + abs(rng.normal(0, 0.005)))
            low = close * (1 - abs(rng.normal(0, 0.005)))
            vol = int(rng.integers(100_000, 1_000_000))
            rows.append({
                "symbol": sym,
                "security_name": f"{sym} Ltd",
                "series": "EQ",
                "trade_date": d,
                "open_price": round(close * 0.999, 2),
                "high_price": round(high, 2),
                "low_price": round(low, 2),
                "close_price": round(close, 2),
                "prev_close": round(close * 0.995, 2),
                "net_traded_qty": vol,
                "net_traded_value": round(vol * close, 2),
                "total_trades": int(rng.integers(500, 5000)),
                "delivery_qty": int(vol * rng.uniform(0.3, 0.8)),
                "delivery_pct": round(rng.uniform(30, 80), 2),
                "high_52_week": round(close * 1.15, 2),
                "low_52_week": round(close * 0.85, 2),
                "is_index": False,
                "is_valid": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def breakout_df() -> pd.DataFrame:
    """Minimal breakout data (hl_hits format)."""
    return pd.DataFrame({
        "symbol": ["RELIANCE", "TCS", "INFY"],
        "series": ["EQ", "EQ", "EQ"],
        "trade_date": [BASE_DATE, BASE_DATE, BASE_DATE],
        "high_low": ["H", "H", "L"],
        "hl_type": ["H", "H", "L"],
    })


@pytest.fixture
def corporate_actions_df() -> pd.DataFrame:
    """Minimal corporate actions data."""
    return pd.DataFrame({
        "symbol": ["RELIANCE", "TCS"],
        "series": ["EQ", "EQ"],
        "security_name": ["Reliance Industries Ltd", "Tata Consultancy Services"],
        "trade_date": [BASE_DATE, BASE_DATE],
        "ex_date": [BASE_DATE + timedelta(days=5), BASE_DATE + timedelta(days=10)],
        "record_date": [BASE_DATE + timedelta(days=5), BASE_DATE + timedelta(days=10)],
        "action_type": ["DIVIDEND", "BONUS"],
        "action_amount": [10.0, None],
        "action_details": ["dividend rs. 10 per share", "bonus 1:1"],
    })


@pytest.fixture
def top_traded_df() -> pd.DataFrame:
    """Minimal top-traded data."""
    return pd.DataFrame({
        "rank": list(range(1, 6)),
        "security_name": ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"],
        "net_traded_value": [1e10, 8e9, 7e9, 6e9, 5e9],
        "trade_date": [BASE_DATE] * 5,
    })


@pytest.fixture
def etf_df() -> pd.DataFrame:
    """Minimal ETF price data."""
    rng = np.random.default_rng(99)
    rows = []
    etfs = ["NIFTYBEES", "BANKBEES", "GOLDBEES"]
    for sym in etfs:
        for d in DATES:
            close = 100.0 * (1 + rng.normal(0, 0.01))
            rows.append({
                "symbol": sym,
                "security_name": f"{sym} ETF",
                "trade_date": d,
                "open_price": round(close * 0.999, 2),
                "high_price": round(close * 1.005, 2),
                "low_price": round(close * 0.995, 2),
                "close_price": round(close, 2),
                "prev_close": round(close * 0.998, 2),
                "net_traded_qty": int(rng.integers(10_000, 100_000)),
                "net_traded_value": round(close * rng.integers(10_000, 100_000), 2),
            })
    return pd.DataFrame(rows)
