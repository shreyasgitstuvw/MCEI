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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.logger import get_logger

log = get_logger("udiff_downloader")

RAW_DIR = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

NSE_BASE     = "https://www.nseindia.com"
NSE_API      = "https://www.nseindia.com/api/reports"
NSE_ARCHIVES = "https://nsearchives.nseindia.com"

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
            log.info("Priming NSE session (getting cookies) ...")
            self.session.get(NSE_BASE, timeout=15)
            time.sleep(2)
            self.session.get(NSE_BASE + "/all-reports", timeout=15)
            time.sleep(1.5)
            self._primed = True
        except Exception as exc:
            log.warning(f"Session priming warning: {exc}")

    def _build_api_url(self, d):
        """Build NSE live API URL (requires session cookies)."""
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

    def _build_archive_url(self, d):
        """Build direct nsearchives URL — no session cookies required."""
        date_str = d.strftime("%Y%m%d")
        return (
            f"{NSE_ARCHIVES}/content/cm/"
            f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.zip"
        )

    def _try_archive_download(self, trade_date, zip_path):
        """Try downloading via public nsearchives URL (no cookies)."""
        url = self._build_archive_url(trade_date)
        log.info(f"Trying archive URL: {url}")
        try:
            resp = self.session.get(url, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 10000:
                content_type = resp.headers.get("Content-Type", "")
                snippet = resp.content[:4]
                # ZIP magic bytes: PK (0x50 0x4B)
                if snippet[:2] == b"PK":
                    zip_path.write_bytes(resp.content)
                    size_mb = zip_path.stat().st_size / (1024 * 1024)
                    log.info(f"Archive download OK: {zip_path.name} ({size_mb:.2f} MB)")
                    return zip_path
                if b"<html" in resp.content[:200].lower():
                    log.debug("Archive URL returned HTML (not yet published or wrong date)")
                    return None
            log.debug(f"Archive URL returned status {resp.status_code} / size {len(resp.content)}")
            return None
        except Exception as exc:
            log.debug(f"Archive download attempt failed: {exc}")
            return None

    def download(self, trade_date=None):
        """Download UDIFF bhavcopy ZIP."""
        if trade_date is None:
            trade_date = date.today()

        date_str = trade_date.strftime("%Y%m%d")
        dest_dir = self.download_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / f"udiff_{date_str}.zip"

        if zip_path.exists() and zip_path.stat().st_size > 10000:
            log.info(f"Already downloaded: {zip_path.name}")
            return zip_path

        log.info(f"Downloading UDIFF bhavcopy for {trade_date.strftime('%d-%b-%Y')}")

        # ── Attempt 1: public nsearchives URL (no session cookies, works from CI) ──
        result = self._try_archive_download(trade_date, zip_path)
        if result:
            return result

        # ── Attempt 2: live NSE API (requires session cookies, may be blocked in CI) ──
        log.info("Archive URL failed — falling back to live NSE API ...")
        self._prime()

        url = self._build_api_url(trade_date)
        log.debug(f"API: {NSE_API}")
        log.info(f"Date: {trade_date.strftime('%d-%b-%Y')}")

        try:
            log.info("Requesting ...")
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
            log.info(f"Downloaded: {zip_path.name} ({size_mb:.2f} MB)")

            return zip_path

        except requests.RequestException as exc:
            raise FileNotFoundError(f"Download failed: {exc}")

    def extract(self, zip_path):
        """Extract ZIP and return directory."""
        import zipfile

        dest = zip_path.parent
        log.info("Extracting ZIP ...")

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name or not name.endswith(".csv"):
                    continue
                (dest / name).write_bytes(zf.open(member).read())
                log.info(f"Extracted: {name}")

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

    log.info("=" * 60)
    log.info("  NSE UDIFF Bhavcopy Downloader")
    log.info("=" * 60)

    try:
        csv_dir = dl.download_and_extract(d)
        log.info(f"Complete! Files in: {csv_dir}")
    except FileNotFoundError as e:
        log.error(f"Failed: {e}")
        sys.exit(1)