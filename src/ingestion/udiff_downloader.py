"""
NSE UDIFF Bhavcopy Downloader (Updated Feb 2026)
=================================================
Uses the actual NSE API endpoint for downloading UDIFF bhavcopy.

Usage:
    python -m src.ingestion.udiff_downloader 2026-02-12
"""

from __future__ import annotations
import os, sys, time, json
import requests
from datetime import datetime, date
from pathlib import Path
from urllib.parse import quote

RAW_DIR = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

NSE_BASE = "https://www.nseindia.com"
NSE_API = "https://www.nseindia.com/api/reports"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": NSE_BASE + "/",
    "Connection": "keep-alive",
    "DNT": "1",
}


class UDIFFDownloader:
    def __init__(self, download_dir=None):
        self.download_dir = Path(download_dir) if download_dir else RAW_DIR
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._primed = False

    def _prime(self):
        """Prime session by visiting NSE homepage to get cookies."""
        if self._primed:
            return
        try:
            print("  🔐 Priming session (getting cookies) …")
            self.session.get(NSE_BASE, timeout=15)
            time.sleep(2)
            self.session.get(NSE_BASE + "/all-reports", timeout=15)
            time.sleep(1.5)
            self._primed = True
        except Exception as exc:
            print(f"  ⚠️  Warning: {exc}")

    def _build_url(self, d):
        """Build NSE API URL with proper parameters."""
        archives = [{
            "name": "CM-UDiFF Common Bhavcopy Final (zip)",
            "type": "daily-reports",
            "category": "capital-market",
            "section": "equities"
        }]

        date_str = d.strftime("%d-%b-%Y")
        archives_encoded = quote(json.dumps(archives))

        url = (
            f"{NSE_API}?"
            f"archives={archives_encoded}&"
            f"date={date_str}&"
            f"type=equities&"
            f"mode=single"
        )
        return url

    def download(self, trade_date=None):
        """Download UDIFF bhavcopy ZIP."""
        if trade_date is None:
            trade_date = date.today()

        date_str = trade_date.strftime("%Y%m%d")
        dest_dir = self.download_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / f"udiff_{date_str}.zip"

        if zip_path.exists() and zip_path.stat().st_size > 10000:
            print(f"  ✅ Already downloaded: {zip_path.name}")
            return zip_path

        print(f"\n📅 Downloading for {trade_date.strftime('%d-%b-%Y')}")
        self._prime()

        url = self._build_url(trade_date)
        print(f"  🌐 API: {NSE_API}")
        print(f"  📆 Date: {trade_date.strftime('%d-%b-%Y')}")

        try:
            print(f"  ⬇️  Requesting …")
            resp = self.session.get(url, timeout=60, allow_redirects=True)

            if resp.status_code != 200:
                raise FileNotFoundError(f"NSE returned status {resp.status_code}")

            content_type = resp.headers.get("Content-Type", "")

            if "json" in content_type:
                try:
                    error_data = resp.json()
                    raise FileNotFoundError(f"NSE API error: {error_data}")
                except json.JSONDecodeError:
                    raise FileNotFoundError("NSE returned JSON error")

            if len(resp.content) < 10000:
                raise FileNotFoundError("Downloaded file too small")

            zip_path.write_bytes(resp.content)
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Downloaded: {zip_path.name} ({size_mb:.2f} MB)")

            return zip_path

        except requests.RequestException as exc:
            raise FileNotFoundError(f"Download failed: {exc}")

    def extract(self, zip_path):
        """Extract ZIP and return directory."""
        import zipfile

        dest = zip_path.parent
        print(f"  📦 Extracting …")

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name or not name.endswith(".csv"):
                    continue
                (dest / name).write_bytes(zf.open(member).read())
                print(f"  ✅ Extracted: {name}")

        return dest

    def download_and_extract(self, trade_date=None):
        """Download + extract."""
        zip_path = self.download(trade_date)
        return self.extract(zip_path)


if __name__ == "__main__":
    dl = UDIFFDownloader()

    if len(sys.argv) > 1:
        d = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    else:
        d = date.today()

    print(f"\n{'=' * 60}")
    print(f"  NSE UDIFF Bhavcopy Downloader")
    print(f"{'=' * 60}")

    try:
        csv_dir = dl.download_and_extract(d)
        print(f"\n✅ Complete! Files in: {csv_dir}")
    except FileNotFoundError as e:
        print(f"\n❌ Failed: {e}")
        sys.exit(1)