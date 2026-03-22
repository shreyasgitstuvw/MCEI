"""
NSE Bhavcopy Downloader
Downloads the daily equity bhavcopy ZIP from the NSE website.

NSE requires a primed session (cookies from homepage) before file downloads work.
We try the new API endpoint first, then fall back to the legacy static URL.

Usage:
    python -m src.ingestion.nse_downloader              # today
    python -m src.ingestion.nse_downloader 2026-02-13   # specific date
"""

from __future__ import annotations
import os, sys, time, zipfile
import requests
from datetime import datetime, date
from pathlib import Path

RAW_DIR = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

MONTH_ABBR = {
    1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
    7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC",
}

NSE_BASE = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         NSE_BASE + "/",
    "Connection":      "keep-alive",
}


class NSEDownloader:
    def __init__(self, download_dir: str | Path | None = None):
        self.download_dir = Path(download_dir) if download_dir else RAW_DIR
        self.session      = requests.Session()
        self.session.headers.update(HEADERS)
        self._primed      = False

    # ── session priming ───────────────────────────────────────────────────────

    def _prime(self):
        """Hit NSE homepage + all-reports page to get necessary cookies."""
        if self._primed:
            return
        try:
            self.session.get(NSE_BASE, timeout=15)
            time.sleep(1.5)
            self.session.get(NSE_BASE + "/all-reports", timeout=15)
            time.sleep(1.0)
            self._primed = True
        except Exception as exc:
            print(f"  ⚠  Session prime warning (continuing): {exc}")

    # ── URL builders ──────────────────────────────────────────────────────────

    def _new_api_url(self, d: date) -> str:
        """NSE's JSON report API — used since 2023."""
        return (
            f"{NSE_BASE}/api/reports"
            "?archives=%5B%7B%22name%22%3A%22CM%20-%20Bhavcopy%22%2C"
            "%22type%22%3A%22daily%22%2C%22category%22%3A%22capital-market%22%2C"
            "%22section%22%3A%22equities%22%7D%5D"
            f"&date={d.strftime('%d-%m-%Y')}&type=equities&mode=single"
        )

    def _legacy_url(self, d: date) -> str:
        """Static file URL — works for older dates."""
        mon = MONTH_ABBR[d.month]
        return (
            f"{NSE_BASE}/content/historical/EQUITIES/"
            f"{d.year}/{mon}/cm{d.strftime('%d')}{mon}{d.year}bhav.csv.zip"
        )

    # ── download ──────────────────────────────────────────────────────────────

    def _attempt(self, url: str, dest: Path) -> bool:
        try:
            r = self.session.get(url, timeout=60, stream=True)
            if r.status_code != 200:
                return False
            ct = r.headers.get("content-type", "")
            # NSE returns HTML error pages with 200 — detect by content-type
            if "zip" not in ct and "octet" not in ct and len(r.content) < 5000:
                return False
            dest.write_bytes(r.content)
            return dest.stat().st_size > 5000   # sanity: real ZIPs are >5 KB
        except Exception:
            return False

    def download(self, trade_date: date | None = None) -> Path:
        """
        Download bhavcopy ZIP for *trade_date*.
        Returns path of the saved ZIP.
        Raises FileNotFoundError if all attempts fail.
        """
        if trade_date is None:
            trade_date = date.today()

        date_str = trade_date.strftime("%Y%m%d")
        dest_dir = self.download_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / f"bhav_{date_str}.zip"

        if zip_path.exists() and zip_path.stat().st_size > 5000:
            print(f"  ✅  Already downloaded: {zip_path}")
            return zip_path

        print(f"  ⬇  Downloading bhavcopy for {trade_date} …")
        self._prime()

        for url in [self._new_api_url(trade_date), self._legacy_url(trade_date)]:
            if self._attempt(url, zip_path):
                print(f"  ✅  Saved {zip_path}  ({zip_path.stat().st_size:,} bytes)")
                return zip_path
            time.sleep(1.5)

        raise FileNotFoundError(
            f"Could not download bhavcopy for {trade_date}.\n"
            "Possible reasons: holiday / weekend / not yet published (after 6 PM IST)."
        )

    def extract(self, zip_path: Path) -> Path:
        """Extract ZIP into its parent directory. Returns that directory."""
        dest = zip_path.parent
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name:
                    continue
                (dest / name).write_bytes(zf.open(member).read())
        print(f"  ✅  Extracted to {dest}")
        return dest

    def download_and_extract(self, trade_date: date | None = None) -> Path:
        """Download + extract. Returns directory of CSV files."""
        return self.extract(self.download(trade_date))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dl = NSEDownloader()
    d  = (datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
          if len(sys.argv) > 1 else date.today())
    out = dl.download_and_extract(d)
    print(f"\nFiles ready in: {out}")