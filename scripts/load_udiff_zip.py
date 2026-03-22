"""
Quick UDIFF Loader
==================
Loads NSE UDIFF bhavcopy (ZIP, folder, or CSV).

Usage:
    python -m scripts.load_udiff_zip BhavCopy_NSE_CM_0_0_0_20260204_F_0000_csv.zip
    python -m scripts.load_udiff_zip BhavCopy_NSE_CM_0_0_0_20260204_F_0000_csv (folder)
    python -m scripts.load_udiff_zip BhavCopy_NSE_CM_0_0_0_20260204_F_0000.csv
"""

import sys
import os
import zipfile
import tempfile
import shutil
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.etl.udiff_transformer import transform_udiff_bhavcopy
from src.database.loader import load_all
from src.database.connection import test_connection


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.load_udiff_zip <path>")
        print("\nExamples:")
        print("  python -m scripts.load_udiff_zip BhavCopy_NSE_CM_0_0_0_20260204_F_0000_csv.zip")
        print("  python -m scripts.load_udiff_zip BhavCopy_NSE_CM_0_0_0_20260204_F_0000_csv")
        print("  python -m scripts.load_udiff_zip BhavCopy_NSE_CM_0_0_0_20260204_F_0000.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"❌  File/folder not found: {input_path}")
        sys.exit(1)

    if not test_connection():
        print("❌  Database unreachable. Check .env settings.")
        sys.exit(1)

    # Extract date from filename
    name = input_path.stem if input_path.is_file() else input_path.name
    # Remove .csv from stem if present (for .csv.zip files)
    if name.endswith("_csv"):
        name = name[:-4]

    parts = name.split("_")
    date_str = None
    for part in parts:
        if len(part) == 8 and part.isdigit():
            date_str = part
            break

    if not date_str:
        print(f"❌  Cannot extract date from: {input_path.name}")
        print("    Expected pattern: BhavCopy_NSE_CM_0_0_0_YYYYMMDD...")
        sys.exit(1)

    trade_date = datetime.strptime(date_str, "%Y%m%d").date()
    print(f"\n📅  Trade date: {trade_date}")

    tmpdir = None
    csv_file = None

    try:
        # Determine input type and find CSV
        if input_path.suffix == ".zip":
            print(f"📦  Extracting ZIP: {input_path.name}")
            tmpdir = tempfile.mkdtemp()
            tmppath = Path(tmpdir)

            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(tmppath)

            csvs = list(tmppath.glob("*.csv"))
            if not csvs:
                print(f"❌  No CSV found in ZIP")
                sys.exit(1)

            csv_file = csvs[0]
            print(f"✅  Extracted: {csv_file.name} ({csv_file.stat().st_size:,} bytes)")

        elif input_path.is_dir():
            print(f"📂  Reading folder: {input_path.name}")
            csvs = list(input_path.glob("*.csv"))
            if not csvs:
                print(f"❌  No CSV found in folder")
                sys.exit(1)

            csv_file = csvs[0]
            print(f"✅  Found: {csv_file.name} ({csv_file.stat().st_size:,} bytes)")

        elif input_path.suffix == ".csv":
            print(f"📄  Reading CSV: {input_path.name}")
            csv_file = input_path
            print(f"✅  Size: {csv_file.stat().st_size:,} bytes")

        else:
            print(f"❌  Unsupported file type: {input_path.suffix}")
            print("    Expected: .zip, folder, or .csv")
            sys.exit(1)

        # Transform
        print(f"\n🔄  Transforming UDIFF format …")
        datasets = transform_udiff_bhavcopy(csv_file, trade_date)

        if not datasets:
            print("❌  No datasets produced")
            sys.exit(1)

        print(f"✅  Transformation complete:")
        for key, df in datasets.items():
            print(f"    {key:<12} {len(df):>6,} rows")

        # Load
        print(f"\n💾  Loading to database …")
        results = load_all(datasets, trade_date)
        total = sum(results.values())

        if total == 0:
            print("⚠️  Warning: 0 rows loaded")
            print("   Check that tables exist: psql -f scripts/create_schema.sql")
        else:
            print(f"\n✅  Success — {total:,} rows loaded")

        print("\nBreakdown:")
        for k, v in sorted(results.items()):
            print(f"    {k:<15} {v:>6,} rows")

    except Exception as exc:
        print(f"\n❌  Error: {exc}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

    finally:
        if tmpdir and Path(tmpdir).exists():
            shutil.rmtree(tmpdir)

    print("\n" + "=" * 60)
    print("  ✅ Done! View data in dashboard:")
    print("     streamlit run dashboard/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()