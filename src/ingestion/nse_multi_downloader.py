"""
NSE Multi-Source Downloader
============================
Downloads all NSE data sources for a given trading date:
- 52-week high/low hits
- Corporate actions
- Nifty 50 index data
- Full bhavcopy with delivery data
- ETF prices
- Market summary

Usage:
    python -m src.ingestion.nse_multi_downloader 2026-02-23
"""

from __future__ import annotations
import os, sys, time
import requests
from datetime import datetime, date
from pathlib import Path

RAW_DIR = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

NSE_BASE = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": NSE_BASE + "/",
    "Connection": "keep-alive",
}


class NSEMultiDownloader:
    """Downloads all NSE data sources for comprehensive market analysis."""
    
    def __init__(self, download_dir=None):
        self.download_dir = Path(download_dir) if download_dir else RAW_DIR
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._primed = False

    def _prime(self):
        """Prime session with cookies."""
        if self._primed:
            return
        try:
            print("  🔐 Priming session...")
            self.session.get(NSE_BASE, timeout=15)
            time.sleep(1.5)
            self._primed = True
        except Exception as e:
            print(f"  ⚠️  Prime warning: {e}")

    def _download_file(self, url: str, filename: str, trade_date: date) -> Path | None:
        """Generic file download method."""
        date_str = trade_date.strftime("%Y%m%d")
        dest_dir = self.download_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / filename

        if dest_file.exists() and dest_file.stat().st_size > 100:
            print(f"  ✅ Cached: {filename}")
            return dest_file

        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  ❌ {filename}: HTTP {resp.status_code}")
                return None
            
            if len(resp.content) < 100:
                print(f"  ❌ {filename}: Too small")
                return None

            dest_file.write_bytes(resp.content)
            size_kb = dest_file.stat().st_size / 1024
            print(f"  ✅ Downloaded: {filename} ({size_kb:.1f} KB)")
            return dest_file
            
        except Exception as e:
            print(f"  ❌ {filename}: {e}")
            return None

    def download_52week_hits(self, trade_date: date) -> dict[str, Path | None]:
        """Download 52-week high and low hit files."""
        print("\n📊 Downloading 52-week hit data...")
        
        # Format: 23-Feb-2026
        date_str = trade_date.strftime("%d-%b-%Y")
        
        results = {}
        
        # 52-week highs
        high_url = f"{NSE_BASE}/api/live-analysis-variations?index=SECURITIES%20IN%20F%26O&date={date_str}&type=52high"
        results['high'] = self._download_file(high_url, "52WeekHigh.csv", trade_date)
        
        # 52-week lows  
        low_url = f"{NSE_BASE}/api/live-analysis-variations?index=SECURITIES%20IN%20F%26O&date={date_str}&type=52low"
        results['low'] = self._download_file(low_url, "52WeekLow.csv", trade_date)
        
        return results

    def download_corporate_actions(self, trade_date: date) -> Path | None:
        """Download corporate actions file."""
        print("\n📋 Downloading corporate actions...")
        
        date_str = trade_date.strftime("%d-%b-%Y")
        url = f"{NSE_BASE}/api/corporates-corporateActions?index=equities&from_date={date_str}&to_date={date_str}"
        
        return self._download_file(url, "corporate_actions.csv", trade_date)

    def download_nifty50(self, trade_date: date) -> Path | None:
        """Download Nifty 50 constituent prices."""
        print("\n📈 Downloading Nifty 50 data...")
        
        # MW = Market Watch format
        url = f"{NSE_BASE}/api/equity-stockIndices?index=NIFTY%2050"
        
        return self._download_file(url, "nifty50_constituents.csv", trade_date)

    def download_full_bhavcopy(self, trade_date: date) -> Path | None:
        """Download full bhavcopy with delivery data."""
        print("\n📦 Downloading full bhavcopy (with delivery data)...")
        
        # Format: DDMMYYYY
        date_str = trade_date.strftime("%d%m%Y")
        url = f"{NSE_BASE}/content/historical/EQUITIES/{trade_date.year}/{trade_date.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"
        
        zip_file = self._download_file(url, f"full_bhav_{date_str}.csv.zip", trade_date)
        
        if zip_file and zip_file.exists():
            # Extract the ZIP
            import zipfile
            try:
                with zipfile.ZipFile(zip_file) as zf:
                    for member in zf.namelist():
                        if member.endswith('.csv'):
                            extracted = zip_file.parent / "full_bhavcopy.csv"
                            extracted.write_bytes(zf.read(member))
                            print(f"  📂 Extracted: {extracted.name}")
                            return extracted
            except Exception as e:
                print(f"  ⚠️  Extract failed: {e}")
        
        return None

    def download_etf_data(self, trade_date: date) -> Path | None:
        """Download ETF price data."""
        print("\n🏦 Downloading ETF data...")
        
        url = f"{NSE_BASE}/api/equity-stockIndices?index=NIFTY%20ETF"
        
        return self._download_file(url, "etf_prices.csv", trade_date)

    def download_all(self, trade_date: date = None) -> dict:
        """Download all available data sources."""
        if trade_date is None:
            trade_date = date.today()

        print(f"\n{'='*60}")
        print(f"  NSE Multi-Source Download: {trade_date.strftime('%d-%b-%Y')}")
        print(f"{'='*60}")

        self._prime()
        
        results = {
            '52week': self.download_52week_hits(trade_date),
            'corporate_actions': self.download_corporate_actions(trade_date),
            'nifty50': self.download_nifty50(trade_date),
            'full_bhavcopy': self.download_full_bhavcopy(trade_date),
            'etf': self.download_etf_data(trade_date),
        }

        # Summary
        print(f"\n{'='*60}")
        print("  Download Summary:")
        print(f"{'='*60}")
        
        for source, files in results.items():
            if isinstance(files, dict):
                for sub_name, file_path in files.items():
                    status = "✅" if file_path and file_path.exists() else "❌"
                    print(f"  {status} {source}/{sub_name}")
            else:
                status = "✅" if files and files.exists() else "❌"
                print(f"  {status} {source}")
        
        return results


if __name__ == "__main__":
    dl = NSEMultiDownloader()
    
    if len(sys.argv) > 1:
        d = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    else:
        d = date.today()
    
    results = dl.download_all(d)
    
    date_dir = RAW_DIR / d.strftime("%Y%m%d")
    print(f"\n✅ Files saved to: {date_dir}")