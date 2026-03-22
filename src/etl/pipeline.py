"""
ETL Pipeline Orchestrator
Ties together: Download → Transform → Load

Usage:
    python -m src.etl.pipeline              # today
    python -m src.etl.pipeline 2026-02-13   # specific date
    python -m src.etl.pipeline backfill 2026-02-01 2026-02-13
"""

from __future__ import annotations
import os, sys, time
from datetime import date, datetime, timedelta
from pathlib import Path

# project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ingestion.udiff_downloader import UDIFFDownloader
from src.etl.udiff_transformer import transform_udiff_bhavcopy, find_udiff_file
from src.database.loader import load_all
from src.database.connection import get_engine, test_connection
from src.utils.logger import get_logger

log = get_logger("pipeline")


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_trading_day(d: date) -> bool:
    """Rough check: skip weekends. Does not account for NSE holidays."""
    return d.weekday() < 5  # Mon–Fri


def _find_csv_dir(raw_root: Path, d: date) -> Path | None:
    """Return the directory containing extracted CSVs for date d."""
    date_str = d.strftime("%Y%m%d")
    candidate = raw_root / date_str
    if candidate.exists() and any(candidate.glob("*.csv")):
        return candidate
    return None


# ── single-day pipeline ───────────────────────────────────────────────────────

def run_for_date(trade_date: date,
                 skip_download: bool = False,
                 raw_root: Path | None = None) -> dict:
    """
    Full pipeline for one trading date.
    Returns summary dict with row counts and status.
    """
    if raw_root is None:
        raw_root = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

    t_start = time.time()
    summary = {"date": str(trade_date), "status": "OK", "datasets": {}}

    log.info(f"{'=' * 60}")
    log.info(f"Pipeline starting for {trade_date}")
    log.info(f"{'=' * 60}")

    # ── 1. Find or download CSVs ────────────────────────────────────
    csv_dir = None

    # Check if CSVs already exist
    csv_dir = _find_csv_dir(raw_root, trade_date)
    if csv_dir:
        log.info(f"✅  CSVs already present in {csv_dir}")

    elif not skip_download:
        # Try downloading with UDIFF downloader
        log.info("Attempting UDIFF download from NSE …")
        try:
            dl = UDIFFDownloader(download_dir=raw_root)
            csv_dir = dl.download_and_extract(trade_date)
        except FileNotFoundError as exc:
            log.warning(f"UDIFF download failed: {exc}")
            # Check one more time in case files exist but weren't detected
            csv_dir = _find_csv_dir(raw_root, trade_date)
            if not csv_dir:
                summary["status"] = "NO_DATA"
                summary["error"] = "Download failed and no local CSVs found"
                log.error(
                    f"No data available for {trade_date}. "
                    "Either NSE hasn't published it yet, or it's a holiday/weekend. "
                    f"If you have the CSVs, put them in: {raw_root / trade_date.strftime('%Y%m%d')}"
                )
                return summary
        except Exception as exc:
            log.error(f"Download error: {exc}")
            csv_dir = _find_csv_dir(raw_root, trade_date)
            if not csv_dir:
                summary["status"] = "DOWNLOAD_FAILED"
                summary["error"] = str(exc)
                return summary
    else:
        # skip_download=True but no CSVs found
        if csv_dir is None:
            log.error(f"skip_download=True but no CSVs found for {trade_date}")
            summary["status"] = "NO_DATA"
            return summary

    # ── 2. Transform ───────────────────────────────────────────────
    log.info(f"Transforming CSVs from {csv_dir} …")

    # Find UDIFF CSV file
    udiff_csv = find_udiff_file(csv_dir, trade_date)

    if not udiff_csv:
        log.error(f"No UDIFF CSV found in {csv_dir}")
        summary["status"] = "NO_FILES_MATCHED"
        summary["error"] = f"No matching CSV files found in {csv_dir}"
        return summary

    try:
        log.info(f"Using UDIFF transformer on {udiff_csv.name}")
        datasets = transform_udiff_bhavcopy(udiff_csv, trade_date)
        if not datasets:
            log.warning("No datasets produced by transformer")
            summary["status"] = "NO_DATA"
            summary["error"] = "Transformer produced no datasets"
            return summary
    except Exception as exc:
        log.error(f"Transform failed: {exc}")
        summary["status"] = "TRANSFORM_FAILED"
        summary["error"] = str(exc)
        return summary

    for key, df in datasets.items():
        log.info(f"  {key}: {len(df)} rows")

    # ── 3. Load to DB ──────────────────────────────────────────────
    log.info("Loading to database …")
    try:
        results = load_all(datasets)
        summary["datasets"] = results
        total = sum(results.values())
        log.info(f"  ✅  Loaded {total:,} rows across {len(results)} tables")
    except Exception as exc:
        log.error(f"DB load failed: {exc}")
        summary["status"] = "LOAD_FAILED"
        summary["error"] = str(exc)
        return summary

    elapsed = round(time.time() - t_start, 1)
    log.info(f"Pipeline complete in {elapsed}s  ✅")
    summary["elapsed_sec"] = elapsed
    return summary


# ── backfill ──────────────────────────────────────────────────────────────────

def backfill(start: date, end: date,
             raw_root: Path | None = None) -> list[dict]:
    """
    Run pipeline for every trading day between start and end (inclusive).
    Already-downloaded dates are skipped automatically.
    """
    results = []
    d = start
    while d <= end:
        if _is_trading_day(d):
            r = run_for_date(d, raw_root=raw_root)
            results.append(r)
            if r["status"] not in ("SKIPPED",):
                time.sleep(3)  # be polite to NSE servers
        d += timedelta(days=1)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not test_connection():
        print("❌  Cannot reach database. Check .env settings.")
        sys.exit(1)

    args = sys.argv[1:]

    if args and args[0] == "backfill":
        start = datetime.strptime(args[1], "%Y-%m-%d").date()
        end = datetime.strptime(args[2], "%Y-%m-%d").date()
        print(f"\nBackfilling {start} → {end}")
        summaries = backfill(start, end)
        for s in summaries:
            datasets_str = ", ".join(f"{k}:{v}" for k, v in s.get("datasets", {}).items())
            print(f"  {s['date']}  {s['status']}  [{datasets_str}]")

    else:
        d = (datetime.strptime(args[0], "%Y-%m-%d").date()
             if args else date.today())
        result = run_for_date(d)
        print(f"\nResult: {result}")