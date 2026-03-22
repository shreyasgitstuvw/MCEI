"""
NSE API Endpoint Test
=====================
Quick test to verify all NSE API endpoints are working.

Tests:
1. ETF Data - CSV direct download
2. SLB Data - CSV direct download  
3. Market Summary - Archive CSV
4. Full Bhavcopy - Archive CSV (with delivery data!)
5. Nifty 50 - CSV direct download
6. Corporate Actions - CSV direct download
7. 52-Week High - JSON response (special handling)
8. 52-Week Low - JSON response (special handling)

Usage:
    python -m scripts.test_api_endpoints
"""

import sys
import time
import requests
from datetime import date
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.nse_real_api_downloader import NSERealAPIDownloader


def test_single_endpoint(name: str, url: str, session: requests.Session, is_json: bool = False):
    """Test a single API endpoint."""
    print(f"\n{'─'*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"{'─'*60}")
    
    try:
        resp = session.get(url, timeout=30)
        print(f"Status Code: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('Content-Type', 'Unknown')}")
        print(f"Content Length: {len(resp.content)} bytes ({len(resp.content)/1024:.1f} KB)")
        
        if resp.status_code == 200:
            if is_json:
                # Try to parse JSON
                try:
                    data = resp.json()
                    print(f"✅ Valid JSON response")
                    print(f"   Keys: {list(data.keys()) if isinstance(data, dict) else 'Array'}")
                    
                    # Check for CSV download link
                    if isinstance(data, dict):
                        if 'url' in data or 'downloadUrl' in data or 'csvUrl' in data:
                            csv_link = data.get('url') or data.get('downloadUrl') or data.get('csvUrl')
                            print(f"   📎 CSV Download Link: {csv_link}")
                        elif 'data' in data:
                            print(f"   📊 Data records: {len(data['data'])}")
                    
                    return True
                except:
                    print(f"⚠️  Response is not JSON")
                    print(f"   First 200 chars: {resp.text[:200]}")
                    return False
            else:
                # Check if CSV
                first_line = resp.text.split('\n')[0][:100]
                print(f"   First line: {first_line}")
                
                if ',' in first_line or 'SYMBOL' in first_line:
                    print(f"✅ Valid CSV response")
                    return True
                else:
                    print(f"⚠️  May not be CSV")
                    return False
        else:
            print(f"❌ Request failed")
            print(f"   Response: {resp.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test all NSE API endpoints."""
    
    print(f"\n{'='*70}")
    print(f"  NSE API Endpoint Verification")
    print(f"{'='*70}")
    
    # Create session and prime it
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    
    print("\n🔐 Priming session...")
    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(2)
    print("✅ Session primed\n")
    
    # Test date
    test_date = date(2026, 2, 23)
    date_str = test_date.strftime("%d-%b-%Y")
    date_code = test_date.strftime("MA%d%m%y")
    date_full = test_date.strftime("%d%m%Y")
    
    # Define endpoints
    endpoints = [
        ("ETF Data", 
         "https://www.nseindia.com/api/etf?csv=true&selectValFormat=crores",
         False),
        
        ("SLB Data",
         "https://www.nseindia.com/api/live-analysis-slb?series=03&csv=true",
         False),
        
        ("Market Summary",
         f"https://nsearchives.nseindia.com/archives/equities/mkt/{date_code}.csv",
         False),
        
        ("Full Bhavcopy (DELIVERY DATA!)",
         f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_full}.csv",
         False),
        
        ("Nifty 50",
         "https://www.nseindia.com/api/equity-stockIndices?csv=true&index=NIFTY%2050&selectValFormat=crores",
         False),
        
        ("Corporate Actions",
         "https://www.nseindia.com/api/corporates-corporateActions?index=equities&csv=true",
         False),
        
        ("52-Week High (JSON)",
         f"https://www.nseindia.com/api/live-analysis-variations?index=SECURITIES%20IN%20F%26O&date={date_str}&type=52high",
         True),
        
        ("52-Week Low (JSON)",
         f"https://www.nseindia.com/api/live-analysis-variations?index=SECURITIES%20IN%20F%26O&date={date_str}&type=52low",
         True),
    ]
    
    # Test each endpoint
    results = {}
    for name, url, is_json in endpoints:
        success = test_single_endpoint(name, url, session, is_json)
        results[name] = success
        time.sleep(1.5)  # Rate limiting
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  Test Summary")
    print(f"{'='*70}\n")
    
    success_count = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {success_count}/{total} endpoints working")
    
    if success_count == total:
        print(f"\n  🎉 All endpoints verified! Ready to download data.")
    elif success_count > 0:
        print(f"\n  ⚠️  Some endpoints failed. Check network/date validity.")
    else:
        print(f"\n  ❌ All endpoints failed. Possible issues:")
        print(f"     - Network connectivity")
        print(f"     - NSE website down")
        print(f"     - Invalid test date (weekend/holiday)")
        print(f"     - Rate limiting (wait and retry)")
    
    print()
    
    # Now test the downloader
    if success_count >= 4:  # At least half working
        print(f"\n{'='*70}")
        print(f"  Testing Real API Downloader")
        print(f"{'='*70}\n")
        
        dl = NSERealAPIDownloader()
        # Just download one file as test
        print("Testing full bhavcopy download (most important)...")
        result = dl.download_full_bhavcopy(test_date)
        
        if result and result.exists():
            print(f"\n✅ Downloader working! File saved: {result}")
            print(f"   Size: {result.stat().st_size / 1024:.1f} KB")
        else:
            print(f"\n⚠️  Downloader test failed, but endpoints work.")
            print(f"   This might be a date issue - try a recent trading day.")


if __name__ == "__main__":
    main()