"""
NSE Real API Downloader
=======================
Downloads all NSE data sources using actual API endpoints discovered.

Key Features:
- Uses real NSE API endpoints (not guessed URLs)
- Handles JSON responses with download links for 52-week data
- Proper session management with cookies
- CSV format downloads

API Endpoints:
1. ETF Data: /api/etf?csv=true
2. SLB Data: /api/live-analysis-slb?series=03&csv=true
3. Market Summary: /archives/equities/mkt/MADDMMYY.csv
4. Full Bhavcopy: /products/content/sec_bhavdata_full_DDMMYYYY.csv
5. Nifty 50: /api/equity-stockIndices?csv=true&index=NIFTY%2050
6. Corporate Actions: /api/corporates-corporateActions?index=equities&csv=true
7. 52-Week High/Low: /api/live-analysis-variations (returns JSON with CSV link)

Usage:
    python -m src.ingestion.nse_real_api_downloader 2026-02-23
"""

from __future__ import annotations
import os, sys, time
import requests
import json
from datetime import datetime, date
from pathlib import Path

RAW_DIR = Path(os.getenv("RAW_DATA_PATH", "data/raw"))

NSE_BASE = "https://www.nseindia.com"
NSE_ARCHIVES = "https://nsearchives.nseindia.com"

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
    "X-Requested-With": "XMLHttpRequest",
}


class NSERealAPIDownloader:
    """Downloads NSE data using actual discovered API endpoints."""
    
    def __init__(self, download_dir=None):
        self.download_dir = Path(download_dir) if download_dir else RAW_DIR
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._primed = False

    def _prime(self):
        """Prime session by visiting homepage to get cookies."""
        if self._primed:
            return
        try:
            print("  🔐 Priming session (getting cookies)...")
            # Visit homepage
            self.session.get(NSE_BASE, timeout=15)
            time.sleep(1.5)
            # Visit market data page
            self.session.get(f"{NSE_BASE}/market-data/live-equity-market", timeout=15)
            time.sleep(1)
            self._primed = True
            print("  ✅ Session primed")
        except Exception as e:
            print(f"  ⚠️  Prime warning: {e}")

    def _download_direct_csv(self, url: str, filename: str, trade_date: date) -> Path | None:
        """Download CSV file directly from URL."""
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
            
            # Check if response is actually CSV (not error page)
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'text/csv' not in content_type and 'application/csv' not in content_type:
                # Might still be valid CSV without proper content-type
                if not resp.text.strip().startswith('"') and not resp.text.strip().startswith('SYMBOL'):
                    print(f"  ⚠️  {filename}: Not a CSV file (Content-Type: {content_type})")
                    # Try to save anyway and let transformer handle it
            
            if len(resp.content) < 100:
                print(f"  ❌ {filename}: Too small ({len(resp.content)} bytes)")
                return None

            dest_file.write_bytes(resp.content)
            size_kb = dest_file.stat().st_size / 1024
            print(f"  ✅ Downloaded: {filename} ({size_kb:.1f} KB)")
            return dest_file
            
        except Exception as e:
            print(f"  ❌ {filename}: {e}")
            return None

    def _download_json_then_csv(self, json_url: str, csv_filename: str, trade_date: date) -> Path | None:
        """
        Download JSON response that contains a CSV download link.
        Used for 52-week high/low data.
        """
        date_str = trade_date.strftime("%Y%m%d")
        dest_dir = self.download_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / csv_filename

        if dest_file.exists() and dest_file.stat().st_size > 100:
            print(f"  ✅ Cached: {csv_filename}")
            return dest_file

        try:
            # First, get JSON response
            resp = self.session.get(json_url, timeout=30)
            if resp.status_code != 200:
                print(f"  ❌ {csv_filename}: HTTP {resp.status_code}")
                return None
            
            # Parse JSON to find CSV download link
            try:
                data = resp.json()
                
                # NSE returns CSV link in response
                if 'url' in data:
                    csv_url = data['url']
                elif 'downloadUrl' in data:
                    csv_url = data['downloadUrl']
                elif 'csvUrl' in data:
                    csv_url = data['csvUrl']
                else:
                    # Data might be directly in JSON - convert to CSV
                    print(f"  ℹ️  {csv_filename}: No CSV link, converting JSON to CSV")
                    return self._json_to_csv(data, dest_file)
                
                # Download CSV from the link
                if not csv_url.startswith('http'):
                    csv_url = NSE_BASE + csv_url
                
                csv_resp = self.session.get(csv_url, timeout=30)
                if csv_resp.status_code == 200 and len(csv_resp.content) > 100:
                    dest_file.write_bytes(csv_resp.content)
                    size_kb = dest_file.stat().st_size / 1024
                    print(f"  ✅ Downloaded: {csv_filename} ({size_kb:.1f} KB)")
                    return dest_file
                else:
                    print(f"  ❌ {csv_filename}: CSV download failed")
                    return None
                    
            except json.JSONDecodeError:
                # Maybe it's already CSV?
                if resp.text.strip().startswith('"') or resp.text.strip().startswith('SYMBOL'):
                    dest_file.write_text(resp.text)
                    print(f"  ✅ Downloaded: {csv_filename} (direct CSV)")
                    return dest_file
                else:
                    print(f"  ❌ {csv_filename}: Invalid JSON response")
                    return None
            
        except Exception as e:
            print(f"  ❌ {csv_filename}: {e}")
            return None

    def _json_to_csv(self, json_data: dict, dest_file: Path) -> Path:
        """Convert JSON data to CSV format."""
        import pandas as pd
        
        # Extract data array from JSON
        if isinstance(json_data, dict):
            if 'data' in json_data:
                data = json_data['data']
            elif 'records' in json_data:
                data = json_data['records']
            else:
                data = [json_data]
        else:
            data = json_data
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(data)
        df.to_csv(dest_file, index=False)
        print(f"  ✅ Converted JSON to CSV: {dest_file.name}")
        return dest_file

    def download_etf_data(self, trade_date: date) -> Path | None:
        """Download ETF data."""
        print("\n🏦 Downloading ETF Data...")
        url = f"{NSE_BASE}/api/etf?csv=true&selectValFormat=crores"
        return self._download_direct_csv(url, "MW-ETF.csv", trade_date)

    def download_slb_data(self, trade_date: date) -> Path | None:
        """Download Securities Lending & Borrowing data."""
        print("\n📊 Downloading SLB Data...")
        url = f"{NSE_BASE}/api/live-analysis-slb?series=03&csv=true"
        return self._download_direct_csv(url, "MW-SLB-03.csv", trade_date)

    def download_market_summary(self, trade_date: date) -> Path | None:
        """Download market activity summary."""
        print("\n📈 Downloading Market Summary...")
        # Format: MADDMMYY (e.g., MA230226 for 23-Feb-2026)
        date_code = trade_date.strftime("MA%d%m%y")
        url = f"{NSE_ARCHIVES}/archives/equities/mkt/{date_code}.csv"
        return self._download_direct_csv(url, f"{date_code}.csv", trade_date)

    def download_full_bhavcopy(self, trade_date: date) -> Path | None:
        """Download full bhavcopy with delivery data (CRITICAL!)."""
        print("\n📦 Downloading Full Bhavcopy (with delivery data)...")
        # Format: sec_bhavdata_full_DDMMYYYY.csv
        date_str = trade_date.strftime("%d%m%Y")
        url = f"{NSE_ARCHIVES}/products/content/sec_bhavdata_full_{date_str}.csv"
        return self._download_direct_csv(url, "sec_bhavdata_full.csv", trade_date)

    def download_nifty50(self, trade_date: date) -> Path | None:
        """Download Nifty 50 constituent data."""
        print("\n📊 Downloading Nifty 50 Data...")
        url = f"{NSE_BASE}/api/equity-stockIndices?csv=true&index=NIFTY%2050&selectValFormat=crores"
        return self._download_direct_csv(url, "MW-NIFTY-50.csv", trade_date)

    def download_corporate_actions(self, trade_date: date) -> Path | None:
        """Download corporate actions."""
        print("\n📋 Downloading Corporate Actions...")
        url = f"{NSE_BASE}/api/corporates-corporateActions?index=equities&csv=true"
        return self._download_direct_csv(url, "CF-CA-equities.csv", trade_date)

    def download_52week_high(self, trade_date: date) -> Path | None:
        """Download 52-week high data (JSON response with CSV link)."""
        print("\n📈 Downloading 52-Week High...")
        # Format date: DD-MMM-YYYY (e.g., 23-Feb-2026)
        date_str = trade_date.strftime("%d-%b-%Y")
        url = f"{NSE_BASE}/api/live-analysis-variations?index=SECURITIES%20IN%20F%26O&date={date_str}&type=52high"
        return self._download_json_then_csv(url, "52WeekHigh.csv", trade_date)

    def download_52week_low(self, trade_date: date) -> Path | None:
        """Download 52-week low data (JSON response with CSV link)."""
        print("\n📉 Downloading 52-Week Low...")
        date_str = trade_date.strftime("%d-%b-%Y")
        url = f"{NSE_BASE}/api/live-analysis-variations?index=SECURITIES%20IN%20F%26O&date={date_str}&type=52low"
        return self._download_json_then_csv(url, "52WeekLow.csv", trade_date)

    def download_all(self, trade_date: date = None) -> dict:
        """Download all available data sources."""
        if trade_date is None:
            trade_date = date.today()

        print(f"\n{'='*70}")
        print(f"  NSE Real API Downloader - {trade_date.strftime('%d-%b-%Y')}")
        print(f"{'='*70}")

        self._prime()
        
        results = {
            'etf': self.download_etf_data(trade_date),
            'slb': self.download_slb_data(trade_date),
            'market_summary': self.download_market_summary(trade_date),
            'full_bhavcopy': self.download_full_bhavcopy(trade_date),  # CRITICAL - has delivery data
            'nifty50': self.download_nifty50(trade_date),
            'corporate_actions': self.download_corporate_actions(trade_date),
            '52week_high': self.download_52week_high(trade_date),
            '52week_low': self.download_52week_low(trade_date),
        }

        # Summary
        print(f"\n{'='*70}")
        print("  Download Summary:")
        print(f"{'='*70}")
        
        success_count = 0
        for source, file_path in results.items():
            if file_path and file_path.exists():
                status = "✅"
                success_count += 1
            else:
                status = "❌"
            print(f"  {status} {source}")
        
        print(f"\n  Total: {success_count}/{len(results)} sources downloaded")
        
        return results


if __name__ == "__main__":
    dl = NSERealAPIDownloader()
    
    if len(sys.argv) > 1:
        d = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    else:
        d = date.today()
    
    results = dl.download_all(d)
    
    date_dir = RAW_DIR / d.strftime("%Y%m%d")
    print(f"\n✅ Files saved to: {date_dir}\n")