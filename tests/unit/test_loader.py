"""Unit tests for database loader functions."""
from datetime import date
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# We test the type-coercion logic in isolation — no real DB needed
from src.database.loader import _upsert, _update_delivery


TRADE_DATE = date(2026, 2, 3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_engine(columns=("symbol", "trade_date", "close_price", "net_traded_qty")):
    """Return a mock SQLAlchemy engine whose _table_columns returns `columns`."""
    engine = MagicMock()
    conn_ctx = MagicMock()
    conn = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn_ctx
    engine.begin.return_value = conn_ctx

    # Make _table_columns return the desired set
    conn.execute.return_value.fetchall.return_value = [(c,) for c in columns]
    return engine, conn


# ── _upsert ───────────────────────────────────────────────────────────────────

class TestUpsert:

    def test_empty_dataframe_returns_zero(self):
        engine = MagicMock()
        result = _upsert(pd.DataFrame(), "fact_daily_prices", ["symbol", "trade_date"], engine)
        assert result == 0

    def test_none_dataframe_returns_zero(self):
        engine = MagicMock()
        result = _upsert(None, "fact_daily_prices", ["symbol", "trade_date"], engine)
        assert result == 0

    def test_numpy_int64_is_converted(self):
        """numpy.int64 values must be converted to Python int before execution."""
        engine, conn = _make_engine(["symbol", "trade_date", "net_traded_qty"])
        df = pd.DataFrame({
            "symbol": ["RELIANCE"],
            "trade_date": [TRADE_DATE],
            "net_traded_qty": np.array([1_000_000], dtype=np.int64),
        })

        _upsert(df, "fact_daily_prices", ["symbol", "trade_date"], engine)

        # Capture the records passed to conn.execute
        exec_call = conn.execute.call_args_list[-1]
        records = exec_call[0][1]  # positional arg 1
        assert isinstance(records[0]["net_traded_qty"], int), "numpy.int64 should be converted to Python int"

    def test_nan_is_converted_to_none(self):
        """NaN values must become None (not float('nan')) for PostgreSQL."""
        engine, conn = _make_engine(["symbol", "trade_date", "close_price"])
        df = pd.DataFrame({
            "symbol": ["RELIANCE"],
            "trade_date": [TRADE_DATE],
            "close_price": [float("nan")],
        })

        _upsert(df, "fact_daily_prices", ["symbol", "trade_date"], engine)

        exec_call = conn.execute.call_args_list[-1]
        records = exec_call[0][1]
        assert records[0]["close_price"] is None

    def test_bigint_column_float_is_converted_to_int(self):
        """PostgreSQL BIGINT columns reject Python float — must be int."""
        engine, conn = _make_engine(["symbol", "trade_date", "net_traded_qty"])
        df = pd.DataFrame({
            "symbol": ["RELIANCE"],
            "trade_date": [TRADE_DATE],
            "net_traded_qty": [33_003.0],  # float whole number → must become int
        })

        _upsert(df, "fact_daily_prices", ["symbol", "trade_date"], engine)

        exec_call = conn.execute.call_args_list[-1]
        records = exec_call[0][1]
        assert isinstance(records[0]["net_traded_qty"], int)
        assert records[0]["net_traded_qty"] == 33_003

    def test_returns_row_count(self):
        engine, conn = _make_engine(["symbol", "trade_date", "close_price"])
        df = pd.DataFrame({
            "symbol": ["RELIANCE", "TCS"],
            "trade_date": [TRADE_DATE, TRADE_DATE],
            "close_price": [1200.5, 3500.0],
        })
        result = _upsert(df, "fact_daily_prices", ["symbol", "trade_date"], engine)
        assert result == 2


# ── _update_delivery ──────────────────────────────────────────────────────────

class TestUpdateDelivery:

    def test_empty_delivery_cols_returns_zero(self):
        df = pd.DataFrame({"symbol": ["RELIANCE"], "trade_date": [TRADE_DATE]})
        engine = MagicMock()
        result = _update_delivery(df, TRADE_DATE, engine)
        assert result == 0

    def test_nan_delivery_qty_becomes_none(self):
        conn_ctx = MagicMock()
        conn = MagicMock()
        conn_ctx.__enter__ = MagicMock(return_value=conn)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        engine = MagicMock()
        engine.begin.return_value = conn_ctx

        df = pd.DataFrame({
            "symbol": ["RELIANCE"],
            "trade_date": [TRADE_DATE],
            "delivery_qty": [float("nan")],
            "delivery_pct": [55.5],
        })

        _update_delivery(df, TRADE_DATE, engine)

        exec_call = conn.execute.call_args_list[0]
        records = exec_call[0][1]
        assert records[0]["delivery_qty"] is None, "NaN delivery_qty must be None"
        assert isinstance(records[0]["delivery_pct"], float)

    def test_numpy_delivery_qty_converted_to_int(self):
        conn_ctx = MagicMock()
        conn = MagicMock()
        conn_ctx.__enter__ = MagicMock(return_value=conn)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        engine = MagicMock()
        engine.begin.return_value = conn_ctx

        df = pd.DataFrame({
            "symbol": ["RELIANCE"],
            "trade_date": [TRADE_DATE],
            "delivery_qty": np.array([50_000.0]),
            "delivery_pct": np.array([65.5]),
        })

        _update_delivery(df, TRADE_DATE, engine)

        exec_call = conn.execute.call_args_list[0]
        records = exec_call[0][1]
        assert isinstance(records[0]["delivery_qty"], int)
        assert records[0]["delivery_qty"] == 50_000
