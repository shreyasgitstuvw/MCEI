"""
NSE Multi-Source Transformer
=============================
Transforms all NSE data sources into database-ready DataFrames.

Handles:
- 52-week high/low hits
- Corporate actions
- Nifty 50 index data
- Full bhavcopy with delivery data
- ETF prices
"""

from __future__ import annotations
import pandas as pd
from datetime import date, datetime
from pathlib import Path


def _read_csv_auto_encoding(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Read CSV with automatic encoding detection and error handling.
    NSE files can have various encodings and malformed fields.
    """
    # Try common encodings in order
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            # Add error handling for malformed fields
            return pd.read_csv(
                file_path,
                encoding=encoding,
                on_bad_lines='skip',  # Skip malformed lines
                **kwargs
            )
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            # Try with quoting
            try:
                return pd.read_csv(
                    file_path,
                    encoding=encoding,
                    quoting=1,  # QUOTE_ALL
                    on_bad_lines='skip',
                    **kwargs
                )
            except:
                continue

    # Last resort: read as binary and decode errors='ignore'
    try:
        return pd.read_csv(
            file_path,
            encoding='latin-1',
            on_bad_lines='skip',
            **kwargs
        )
    except Exception as e:
        raise Exception(f"Could not read {file_path.name} with any encoding: {e}")


def transform_52week_hits(file_path: Path, trade_date: date, hit_type: str) -> pd.DataFrame:
    """
    Transform 52-week high/low CSV into hl_hits format.

    Args:
        file_path: Path to 52WeekHigh.csv or 52WeekLow.csv
        trade_date: Trading date
        hit_type: 'HIGH' or 'LOW'

    Returns:
        DataFrame ready for fact_hl_hits table
    """
    df = _read_csv_auto_encoding(file_path, skipinitialspace=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Map to database schema
    result = pd.DataFrame({
        'trade_date': trade_date,
        'symbol': df['Symbol'].str.strip(),
        'series': df['Series'].str.strip(),
        'security_name': None,  # Not in source file
        'hl_type': hit_type,
        'price': df['LTP'],
        'prev_high_low': df['Prev.High'] if hit_type == 'HIGH' else df['Prev.Low'],
    })

    # Filter valid rows
    result = result[result['symbol'].notna() & (result['symbol'] != '')]

    return result


def transform_corporate_actions(file_path: Path, trade_date: date) -> pd.DataFrame:
    """
    Transform corporate actions CSV.

    Returns:
        DataFrame ready for fact_corporate_actions table
    """
    df = _read_csv_auto_encoding(file_path)

    # Parse action purpose to extract type and amount
    def parse_purpose(purpose: str) -> tuple:
        """Extract action type and details from purpose string."""
        purpose = str(purpose).lower()

        if 'dividend' in purpose:
            action_type = 'DIVIDEND'
        elif 'split' in purpose:
            action_type = 'SPLIT'
        elif 'bonus' in purpose:
            action_type = 'BONUS'
        elif 'rights' in purpose:
            action_type = 'RIGHTS'
        else:
            action_type = 'OTHER'

        # Try to extract amount (simplified)
        import re
        amount_match = re.search(r'Rs\.?\s*([\d.]+)', purpose, re.IGNORECASE)
        amount = float(amount_match.group(1)) if amount_match else None

        return action_type, amount, purpose

    # Apply parsing
    parsed = df['PURPOSE'].apply(parse_purpose)

    result = pd.DataFrame({
        'trade_date': trade_date,
        'symbol': df['SYMBOL'].str.strip(),
        'series': df['SERIES'].str.strip(),
        'security_name': df['COMPANY NAME'].str.strip(),
        'ex_date': pd.to_datetime(df['EX-DATE'], format='%d-%b-%Y', errors='coerce'),
        'record_date': pd.to_datetime(df['RECORD DATE'], format='%d-%b-%Y', errors='coerce'),
        'action_type': parsed.apply(lambda x: x[0]),
        'action_amount': parsed.apply(lambda x: x[1]),
        'action_details': parsed.apply(lambda x: x[2]),
    })

    return result


def transform_nifty50(file_path: Path, trade_date: date) -> pd.DataFrame:
    """
    Transform Nifty 50 constituent data.

    Returns:
        DataFrame ready for fact_daily_prices (with is_index=True)
    """
    # Try CSV first (new API format)
    try:
        df = _read_csv_auto_encoding(file_path)

        # CSV format from API
        result = pd.DataFrame({
            'trade_date': trade_date,
            'symbol': df['SYMBOL'].str.strip(),
            'series': 'EQ',
            'security_name': df.get('NAME OF COMPANY', df.get('COMPANY', '')),
            'open_price': df.get('OPEN', df.get('Open')),
            'high_price': df.get('HIGH', df.get('High')),
            'low_price': df.get('LOW', df.get('Low')),
            'close_price': df.get('LTP', df.get('Last Price', df.get('LAST PRICE'))),
            'prev_close': df.get('PREV. CLOSE', df.get('Previous Close')),
            'net_traded_value': df.get('VALUE (Rs. Cr.)', 0) * 10000000,  # Convert crores to rupees
            'net_traded_qty': df.get('VOLUME', df.get('TRADED QUANTITY', 0)),
            'is_index': False,  # These are constituents
            'is_valid': True,
        })

        return result

    except Exception as e:
        print(f"CSV parse failed ({e}), trying JSON")
        # Fall back to JSON format
        try:
            import json
            data = json.loads(file_path.read_text(encoding='utf-8'))

            # Extract the data array
            if 'data' in data:
                stocks = data['data']
            else:
                stocks = data

            result = pd.DataFrame({
                'trade_date': trade_date,
                'symbol': [s.get('symbol', '').strip() for s in stocks],
                'series': 'EQ',
                'security_name': [s.get('meta', {}).get('companyName', '') for s in stocks],
                'open_price': [s.get('open') for s in stocks],
                'high_price': [s.get('dayHigh') for s in stocks],
                'low_price': [s.get('dayLow') for s in stocks],
                'close_price': [s.get('lastPrice') for s in stocks],
                'prev_close': [s.get('previousClose') for s in stocks],
                'net_traded_value': [s.get('totalTradedValue') for s in stocks],
                'net_traded_qty': [s.get('totalTradedVolume') for s in stocks],
                'is_index': False,
                'is_valid': True,
            })

            return result

        except Exception as e2:
            print(f"Warning: Could not parse Nifty50 data: {e2}")
            return pd.DataFrame()


def transform_full_bhavcopy(file_path: Path, trade_date: date) -> pd.DataFrame:
    """
    Transform full bhavcopy with delivery data.

    This has CRITICAL delivery percentage data for liquidity analysis!

    Returns:
        DataFrame ready for fact_daily_prices (enhanced)
    """
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Pre-process delivery columns to handle '-' and convert to proper types
    df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'], errors='coerce')
    df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce')

    # Map to schema
    result = pd.DataFrame({
        'trade_date': pd.to_datetime(df['DATE1'], format='%d-%b-%Y', errors='coerce'),
        'symbol': df['SYMBOL'].str.strip(),
        'series': df[' SERIES'].str.strip() if ' SERIES' in df.columns else df['SERIES'].str.strip(),
        # security_name intentionally omitted: full bhavcopy has no names,
        # and including an empty value here would overwrite real names on upsert.
        'prev_close': pd.to_numeric(df['PREV_CLOSE'], errors='coerce'),
        'open_price': pd.to_numeric(df['OPEN_PRICE'], errors='coerce'),
        'high_price': pd.to_numeric(df['HIGH_PRICE'], errors='coerce'),
        'low_price': pd.to_numeric(df['LOW_PRICE'], errors='coerce'),
        'close_price': pd.to_numeric(df['CLOSE_PRICE'], errors='coerce'),
        'net_traded_qty': pd.to_numeric(df['TTL_TRD_QNTY'], errors='coerce'),
        'net_traded_value': pd.to_numeric(df['TURNOVER_LACS'], errors='coerce') * 100000,
        'total_trades': pd.to_numeric(df['NO_OF_TRADES'], errors='coerce'),
        'delivery_qty': df['DELIV_QTY'],  # Already converted above
        'delivery_pct': df['DELIV_PER'],  # Already converted above
        'is_index': False,
        'is_valid': True,
    })

    # Filter to equity series only
    equity_series = ['EQ', 'BE', 'BZ', 'BL', 'GC', 'IL', 'SM', 'ST']
    result = result[result['series'].isin(equity_series)]

    # Filter out rows with invalid data
    result = result.dropna(subset=['symbol', 'close_price'])

    # Note: Loader will convert numpy types to Python types
    # Just ensure int columns are int64, float columns are float64

    return result


def transform_etf_data(file_path: Path, trade_date: date) -> pd.DataFrame:
    """
    Transform ETF price data.

    Returns:
        DataFrame ready for fact_etf_prices
    """
    # Try CSV first
    try:
        df = _read_csv_auto_encoding(file_path)

        result = pd.DataFrame({
            'trade_date': trade_date,
            'symbol': df['SYMBOL'].str.strip(),
            'security_name': df.get('NAME OF COMPANY', df.get('COMPANY', '')),
            'open_price': df.get('OPEN', df.get('Open')),
            'high_price': df.get('HIGH', df.get('High')),
            'low_price': df.get('LOW', df.get('Low')),
            'close_price': df.get('LTP', df.get('Last Price')),
            'prev_close': df.get('PREV. CLOSE', df.get('Previous Close')),
            'net_traded_value': df.get('VALUE (Rs. Cr.)', 0) * 10000000,
            'net_traded_qty': df.get('VOLUME', df.get('TRADED QUANTITY', 0)),
        })

        return result

    except Exception as e:
        print(f"CSV parse failed ({e}), trying JSON")
        # Fall back to JSON
        try:
            import json
            data = json.loads(file_path.read_text(encoding='utf-8'))

            if 'data' in data:
                etfs = data['data']
            else:
                etfs = data

            result = pd.DataFrame({
                'trade_date': trade_date,
                'symbol': [e.get('symbol', '').strip() for e in etfs],
                'security_name': [e.get('meta', {}).get('companyName', '') for e in etfs],
                'open_price': [e.get('open') for e in etfs],
                'high_price': [e.get('dayHigh') for e in etfs],
                'low_price': [e.get('dayLow') for e in etfs],
                'close_price': [e.get('lastPrice') for e in etfs],
                'prev_close': [e.get('previousClose') for e in etfs],
                'net_traded_value': [e.get('totalTradedValue') for e in etfs],
                'net_traded_qty': [e.get('totalTradedVolume') for e in etfs],
            })

            return result

        except Exception as e2:
            print(f"Warning: Could not parse ETF data: {e2}")
            return pd.DataFrame()


def transform_all(data_dir: Path, trade_date: date) -> dict:
    """
    Transform all available data sources in a directory.

    Args:
        data_dir: Directory containing downloaded NSE files
        trade_date: Trading date

    Returns:
        Dict of dataset_name -> DataFrame
    """
    datasets = {}

    print(f"  📂 Looking in: {data_dir}")

    if not data_dir.exists():
        print(f"  ❌ Directory does not exist!")
        return datasets

    # List all files
    all_files = list(data_dir.glob("*"))
    print(f"  📄 Found {len(all_files)} files")
    for f in all_files:
        print(f"     - {f.name}")

    # 52-week highs (try multiple possible names)
    for filename in ["52WeekHigh.csv", "52week_high.csv"]:
        high_file = data_dir / filename
        if high_file.exists():
            try:
                datasets['hl_high'] = transform_52week_hits(high_file, trade_date, 'HIGH')
                print(f"  ✅ Transformed 52W High: {len(datasets['hl_high'])} rows")
                break
            except Exception as e:
                print(f"  ⚠️  52W High failed: {e}")

    # 52-week lows
    for filename in ["52WeekLow.csv", "52week_low.csv"]:
        low_file = data_dir / filename
        if low_file.exists():
            try:
                datasets['hl_low'] = transform_52week_hits(low_file, trade_date, 'LOW')
                print(f"  ✅ Transformed 52W Low: {len(datasets['hl_low'])} rows")
                break
            except Exception as e:
                print(f"  ⚠️  52W Low failed: {e}")

    # Combine HL data (works even if only one file was available)
    hl_parts = []
    for hl_key in ('hl_high', 'hl_low'):
        if hl_key in datasets:
            hl_parts.append(datasets.pop(hl_key))
    if hl_parts:
        datasets['hl_hits'] = pd.concat(hl_parts, ignore_index=True)
        print(f"  ✅ Combined HL hits: {len(datasets['hl_hits'])} rows")

    # Corporate actions (try multiple names)
    for filename in ["CF-CA-equities.csv", "corporate_actions.csv"]:
        ca_file = data_dir / filename
        if ca_file.exists():
            try:
                datasets['corporate_actions'] = transform_corporate_actions(ca_file, trade_date)
                print(f"  ✅ Transformed Corp Actions: {len(datasets['corporate_actions'])} rows")
                break
            except Exception as e:
                print(f"  ⚠️  Corp Actions failed: {e}")

    # Nifty 50 (try multiple names)
    for filename in ["MW-NIFTY-50.csv", "nifty50_constituents.csv", "nifty50.csv"]:
        nifty_file = data_dir / filename
        if nifty_file.exists():
            try:
                datasets['nifty50'] = transform_nifty50(nifty_file, trade_date)
                print(f"  ✅ Transformed Nifty 50: {len(datasets['nifty50'])} rows")
                break
            except Exception as e:
                print(f"  ⚠️  Nifty 50 failed: {e}")

    # Full bhavcopy (CRITICAL - has delivery data!)
    for filename in ["sec_bhavdata_full.csv", "full_bhavcopy.csv"]:
        full_bhav = data_dir / filename
        if full_bhav.exists():
            try:
                datasets['full_bhavcopy'] = transform_full_bhavcopy(full_bhav, trade_date)
                print(f"  ✅ Transformed Full Bhavcopy (WITH DELIVERY): {len(datasets['full_bhavcopy'])} rows")
                break
            except Exception as e:
                print(f"  ⚠️  Full Bhavcopy failed: {e}")

    # ETF data
    for filename in ["MW-ETF.csv", "etf_prices.csv", "etf.csv"]:
        etf_file = data_dir / filename
        if etf_file.exists():
            try:
                datasets['etf'] = transform_etf_data(etf_file, trade_date)
                print(f"  ✅ Transformed ETF: {len(datasets['etf'])} rows")
                break
            except Exception as e:
                print(f"  ⚠️  ETF failed: {e}")

    return datasets


if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) < 2:
        print("Usage: python -m src.etl.multi_transformer <data_dir> [date]")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    trade_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else date.today()

    print(f"\n{'=' * 60}")
    print(f"  Multi-Source Transformation: {trade_date}")
    print(f"{'=' * 60}\n")

    datasets = transform_all(data_dir, trade_date)

    print(f"\n{'=' * 60}")
    print("  Summary:")
    print(f"{'=' * 60}")
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")