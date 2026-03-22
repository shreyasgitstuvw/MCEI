"""
Daily ETL Runner
================
Schedule this script to run every weekday after 6 PM IST
(NSE publishes bhavcopy by ~5:30 PM on trading days).

Windows Task Scheduler:
  Action:   python -m scripts.run_daily_etl
  Trigger:  Daily at 18:30, Monday–Friday

Linux cron (add via `crontab -e`):
  30 18 * * 1-5 cd /path/to/project && venv/bin/python -m scripts.run_daily_etl

Usage:
  python -m scripts.run_daily_etl                    # today
  python -m scripts.run_daily_etl 2026-02-13         # specific date
  python -m scripts.run_daily_etl backfill 2026-01-01 2026-02-13
"""

from __future__ import annotations
import sys, os
from datetime import datetime, date

# put project root on path
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.database.connection import test_connection
from src.etl.pipeline        import run_for_date, backfill
from src.utils.logger        import get_logger

log = get_logger("daily_runner")


def _print_summary(result: dict):
    status  = result.get("status", "?")
    elapsed = result.get("elapsed_sec", "?")
    datasets = result.get("datasets", {})
    total    = sum(datasets.values())

    icon = "✅" if status == "OK" else "⚠️ " if status == "SKIPPED" else "❌"
    print(f"\n{icon}  {result['date']}  |  status={status}  |  {total:,} rows  |  {elapsed}s")
    if datasets:
        for k, v in sorted(datasets.items()):
            print(f"    {k:<15} {v:>6} rows")


def main():
    args = sys.argv[1:]

    print("=" * 60)
    print("  NSE Analytics — Daily ETL Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Database check
    if not test_connection():
        print("\n Database unreachable. Check .env and that PostgreSQL is running.")
        print("    Hint: psql -U postgres -c 'SELECT 1'")
        sys.exit(1)
    print(" Database connected")

    # Dispatch
    if args and args[0] == "backfill":
        if len(args) < 3:
            print("Usage: python -m scripts.run_daily_etl backfill YYYY-MM-DD YYYY-MM-DD")
            sys.exit(1)
        start    = datetime.strptime(args[1], "%Y-%m-%d").date()
        end      = datetime.strptime(args[2], "%Y-%m-%d").date()
        print(f"\nBackfilling {start} → {end} …\n")
        summaries = backfill(start, end)
        for r in summaries:
            _print_summary(r)
        ok  = sum(1 for r in summaries if r["status"] == "OK")
        skp = sum(1 for r in summaries if r["status"] == "SKIPPED")
        print(f"\nDone: {ok} loaded, {skp} skipped, "
              f"{len(summaries)-ok-skp} errors")

    else:
        d = (datetime.strptime(args[0], "%Y-%m-%d").date()
             if args else date.today())
        result = run_for_date(d)
        _print_summary(result)

    print()


if __name__ == "__main__":
    main()