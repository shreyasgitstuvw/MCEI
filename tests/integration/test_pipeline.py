"""
Integration tests for the ETL pipeline.

Tests run against the real PostgreSQL database. They are skipped automatically
if the DB is unreachable, so the suite stays green in CI without a DB.

Run manually:
    pytest tests/integration/ -v
"""
from __future__ import annotations
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)


# ── skip marker if DB is not available ───────────────────────────────────────

def _db_available() -> bool:
    try:
        from src.database.connection import test_connection
        return test_connection()
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(),
    reason="PostgreSQL database not reachable",
)


# ── helpers ───────────────────────────────────────────────────────────────────

TRADE_DATE = date(2026, 2, 3)


def _make_price_datasets():
    """Minimal dataset dict that load_all() accepts."""
    df = pd.DataFrame({
        "symbol": ["TEST_SYM"],
        "series": ["EQ"],
        "security_name": ["Test Symbol Ltd"],
        "trade_date": [TRADE_DATE],
        "open_price": [100.0],
        "high_price": [105.0],
        "low_price": [98.0],
        "close_price": [103.0],
        "prev_close": [101.0],
        "net_traded_qty": [10_000],
        "net_traded_value": [1_030_000.0],
        "total_trades": [500],
        "is_index": [False],
        "is_valid": [True],
    })
    return {"price": df}


# ── load_all integration ──────────────────────────────────────────────────────

@requires_db
class TestLoadAll:

    def test_load_all_returns_dict_with_row_counts(self):
        from src.database.loader import load_all
        datasets = _make_price_datasets()
        results = load_all(datasets)
        assert isinstance(results, dict)
        assert "fact_daily_prices" in results
        assert results["fact_daily_prices"] >= 1

    def test_load_all_is_idempotent(self):
        """Running load_all twice for the same date should not grow the row count."""
        from src.database.loader import load_all
        datasets = _make_price_datasets()
        results1 = load_all(datasets)
        results2 = load_all(datasets)
        assert results1 == results2

    def test_load_all_skips_empty_datasets(self):
        from src.database.loader import load_all
        datasets = {"price": pd.DataFrame(), "circuits": None}
        results = load_all(datasets)
        assert isinstance(results, dict)

    def test_load_all_handles_no_trade_date_column(self):
        from src.database.loader import load_all
        datasets = {"price": pd.DataFrame({"symbol": ["X"]})}
        results = load_all(datasets)
        assert results == {}


# ── pipeline smoke test ───────────────────────────────────────────────────────

class TestPipelineWithMock:
    """Test pipeline flow without hitting NSE or the real DB."""

    def test_run_for_date_returns_no_data_when_download_fails_and_no_csvs(self):
        from src.etl.pipeline import run_for_date
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = run_for_date(
                trade_date=date(2099, 12, 31),
                skip_download=True,
                raw_root=Path(tmp),
            )
        assert result["status"] == "NO_DATA"

    def test_run_for_date_uses_existing_csvs(self):
        """If CSVs already exist, pipeline skips download and proceeds to transform."""
        from src.etl.pipeline import run_for_date
        import tempfile, shutil

        # Copy an existing date's directory to a temp location
        real_dir = Path("data/raw/20260203")
        if not real_dir.exists():
            pytest.skip("data/raw/20260203 not available")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "20260203"
            shutil.copytree(real_dir, dest)

            result = run_for_date(
                trade_date=date(2026, 2, 3),
                skip_download=True,
                raw_root=Path(tmp),
            )

        # Status is OK or LOAD_FAILED (if DB not available) — never DOWNLOAD_FAILED
        assert result["status"] in ("OK", "LOAD_FAILED", "TRANSFORM_FAILED")
        assert result.get("status") != "DOWNLOAD_FAILED"
