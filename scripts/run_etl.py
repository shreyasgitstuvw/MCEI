"""
Unified ETL Pipeline
====================
Auto-detects NSE bhavcopy format (old multi-file vs new UDIFF).

For dates < July 2024: Uses legacy transformer (11 separate CSVs)
For dates >= July 2024: Uses UDIFF transformer (single unified CSV)

Usage:
    python -m scripts.run_etl 2026-02-04           # auto-detects format
    python -m scripts.run_etl backfill 2024-01-01 2026-02-04
"""

from __future__ import annotations
import os, sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.udiff_downloader import UDIFFDownloader
from src.etl.udiff_transformer import transform_udiff_bhavcopy, find_udiff_file
from src.database.loader import load_all
from src.database.connection import test_connection
from src.utils.logger import get_logger

log = get_logger("etl_pipeline")

# Cutoff: NSE switched to UDIFF in July 2024
UDIFF_CUTOFF = date(2024, 7, 1)


def _find_csv_dir(raw_root: Path, d: date) -> Path | None:
    """Return directory containing CSVs for date d."""
    date_dir = raw_root / d.strftime("%Y%m%d")
    if date_dir.exists() and any(date_dir.glob("*.csv")):
        return date_dir
    return None


def run_for_date(trade_date: date, raw_root: Path | None = None) -> dict:
    """
    Full ETL for one date. Auto-detects old vs UDIFF format.
    Returns summary dict.
    """
    if raw_root is None:
        raw_root = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

    summary = {"date": str(trade_date), "status": "OK", "datasets": {}}

    log.info(f"{'='*60}")
    log.info(f"Pipeline for {trade_date} (auto-detect format)")
    log.info(f"{'='*60}")

    # ── 1. Find or download CSVs ──────────────────────────────────────────────
    csv_dir = _find_csv_dir(raw_root, trade_date)
    
    if csv_dir:
        log.info(f"✅  CSVs already present in {csv_dir}")
    else:
        # Try downloading
        if trade_date >= UDIFF_CUTOFF:
            log.info("Attempting UDIFF download (post-July 2024 format) …")
            try:
                dl      = UDIFFDownloader(download_dir=raw_root)
                csv_dir = dl.download_and_extract(trade_date)
            except FileNotFoundError as exc:
                log.warning(f"UDIFF download failed: {exc}")
                csv_dir = _find_csv_dir(raw_root, trade_date)
                if not csv_dir:
                    summary["status"] = "NO_DATA"
                    summary["error"] = str(exc)
                    return summary
        else:
            # OLD format — would need the original downloader
            log.warning(
                f"{trade_date} is before UDIFF cutoff ({UDIFF_CUTOFF}). "
                "Old format download not implemented in this version. "
                f"If you have the CSVs, put them in: {raw_root / trade_date.strftime('%Y%m%d')}"
            )
            summary["status"] = "OLD_FORMAT_NOT_SUPPORTED"
            return summary

    # ── 2. Transform ──────────────────────────────────────────────────────────
    log.info(f"Transforming CSVs from {csv_dir} …")
    
    # Check if UDIFF file exists
    udiff_csv = find_udiff_file(csv_dir, trade_date)
    
    if udiff_csv:
        log.info(f"Detected UDIFF format: {udiff_csv.name}")
        try:
            datasets = transform_udiff_bhavcopy(udiff_csv, trade_date)
        except Exception as exc:
            log.error(f"UDIFF transform failed: {exc}")
            summary["status"] = "TRANSFORM_FAILED"
            summary["error"] = str(exc)
            return summary
    else:
        # OLD format — would use legacy transformer
        log.error(
            f"No UDIFF file found in {csv_dir}. "
            "If this is old-format data, the legacy transformer is needed."
        )
        summary["status"] = "NO_FILES_MATCHED"
        return summary

    if not datasets:
        log.warning("No datasets produced by transformer")
        summary["status"] = "NO_DATA"
        return summary

    for key, df in datasets.items():
        log.info(f"  {key}: {len(df)} rows")

    # ── 3. Load to DB ─────────────────────────────────────────────────────────
    log.info("Loading to database …")
    try:
        results = load_all(datasets, trade_date)
        summary["datasets"] = results
        total = sum(results.values())
        log.info(f"  ✅  Loaded {total:,} rows across {len(results)} tables")
    except Exception as exc:
        log.error(f"DB load failed: {exc}")
        summary["status"] = "LOAD_FAILED"
        summary["error"] = str(exc)
        return summary

    summary["status"] = "OK"
    return summary


def backfill(start: date, end: date, raw_root: Path | None = None) -> list[dict]:
    """Run ETL for every business day between start and end."""
    results = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon–Fri
            r = run_for_date(d, raw_root)
            results.append(r)
            if r["status"] == "OK":
                import time
                time.sleep(3)  # be polite to NSE
        d += timedelta(days=1)
    return results


if __name__ == "__main__":
    if not test_connection():
        print("❌  Database unreachable")
        sys.exit(1)

    args = sys.argv[1:]

    if args and args[0] == "backfill":
        start = datetime.strptime(args[1], "%Y-%m-%d").date()
        end   = datetime.strptime(args[2], "%Y-%m-%d").date()
        print(f"\nBackfilling {start} → {end}")
        summaries = backfill(start, end)
        for s in summaries:
            ds = ", ".join(f"{k}:{v}" for k,v in s.get("datasets",{}).items())
            print(f"  {s['date']}  {s['status']}  [{ds}]")
    else:
        d = (datetime.strptime(args[0], "%Y-%m-%d").date()
             if args else date.today())
        result = run_for_date(d)
        print(f"\nResult: {result}")