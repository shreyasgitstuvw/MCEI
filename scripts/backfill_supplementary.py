"""
Backfill Supplementary Data
============================
Downloads and loads supplementary data for all dates already in fact_daily_prices:

1. Delivery data  (sec_bhavdata_full from nsearchives - no session needed)
2. Market cap     (MCAP file from nsearchives - no session needed)
3. 52-week HL     (live-analysis-variations API - session needed)
4. Corporate acts (corporates-corporateActions API - session needed)

Usage:
    python scripts/backfill_supplementary.py                  # all four sources
    python scripts/backfill_supplementary.py --delivery       # delivery only
    python scripts/backfill_supplementary.py --mcap           # mcap only
    python scripts/backfill_supplementary.py --hl             # 52-week HL only
    python scripts/backfill_supplementary.py --corp           # corporate actions only
    python scripts/backfill_supplementary.py --date 2026-02-23  # single date
"""

from __future__ import annotations
import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_engine
from src.database import connection as _db_conn
from src.database.loader import load_hl, load_mcap, load_corporate_actions
from src.etl.multi_transformer import (
    transform_full_bhavcopy,
    transform_52week_hits,
)
from src.etl.transformer import transform_market_cap
from sqlalchemy import text

NSE_ARCHIVES = "https://nsearchives.nseindia.com"
NSE_BASE     = "https://www.nseindia.com"

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
}

RAW_DIR = Path("data/raw")


def _reconnect(retries: int = 5, delay: int = 15):
    """Dispose the global engine and reconnect — used after Neon drops a connection."""
    for attempt in range(1, retries + 1):
        try:
            if _db_conn._engine is not None:
                _db_conn._engine.dispose()
                _db_conn._engine = None
            eng = get_engine()
            with eng.connect() as c:
                c.execute(text("SELECT 1"))
            return eng
        except Exception as e:
            print(f"\n  Reconnect attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(delay)


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_loaded_dates() -> list[date]:
    engine = get_engine()
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT DISTINCT trade_date FROM fact_daily_prices ORDER BY trade_date"
        )).fetchall()
    return [r[0] for r in rows]


def _make_session(prime: bool = False) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    if prime:
        print("  Priming session (getting NSE cookies)…")
        try:
            s.get(NSE_BASE, timeout=15)
            time.sleep(1.5)
            s.get(f"{NSE_BASE}/market-data/live-equity-market", timeout=15)
            time.sleep(1.0)
            print("  Session primed.")
        except Exception as e:
            print(f"  Prime warning: {e}")
    return s


def _download(session: requests.Session, url: str, dest: Path,
              min_bytes: int = 500) -> Path | None:
    if dest.exists() and dest.stat().st_size > min_bytes:
        return dest
    try:
        r = session.get(url, timeout=60)
        if r.status_code != 200 or len(r.content) < min_bytes:
            return None
        # Reject obvious HTML error pages
        snippet = r.content[:200].lower()
        if b"<html" in snippet or b"<!doctype" in snippet:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
    except Exception as e:
        print(f"    Download error: {e}")
        return None


# ── 1. Delivery data ──────────────────────────────────────────────────────────

def backfill_delivery(dates: list[date], session: requests.Session):
    print(f"\n{'='*60}")
    print(f"  Delivery Data Backfill ({len(dates)} dates)")
    print(f"{'='*60}")

    engine = get_engine()
    ok = skip = fail = 0

    for d in dates:
        date_str = d.strftime("%d%m%Y")
        dest = RAW_DIR / d.strftime("%Y%m%d") / "sec_bhavdata_full.csv"
        url  = f"{NSE_ARCHIVES}/products/content/sec_bhavdata_full_{date_str}.csv"

        print(f"\n  {d}  ", end="", flush=True)

        # Check if already loaded — direct psycopg2 with 30s timeout to avoid pool hangs
        import os, psycopg2 as _pg
        _chk = _pg.connect(
            host=os.getenv("DB_HOST", "").strip(), port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "").strip(), user=os.getenv("DB_USER", "").strip(),
            password=os.getenv("DB_PASSWORD", "").strip(),
            sslmode=os.getenv("DB_SSLMODE", "require").strip(), connect_timeout=30,
        )
        with _chk.cursor() as _cur:
            _cur.execute(
                "SELECT COUNT(*) FROM fact_daily_prices WHERE trade_date=%s AND delivery_pct IS NOT NULL",
                (d,)
            )
            already = _cur.fetchone()[0]
        _chk.close()
        if already > 100:
            print(f"already loaded ({already} rows with delivery)")
            skip += 1
            continue

        f = _download(session, url, dest)
        if not f:
            print(f"download failed → {url}")
            fail += 1
            continue

        try:
            df = transform_full_bhavcopy(f, d)
            # Only update delivery columns, don't overwrite prices
            delivery_cols = [c for c in ("delivery_qty", "delivery_pct") if c in df.columns]
            if not delivery_cols:
                print("no delivery columns in file")
                fail += 1
                continue

            upd = df[["symbol", "trade_date"] + delivery_cols].dropna(subset=["delivery_pct"]).copy()
            if "delivery_qty" in upd.columns:
                upd["delivery_qty"] = pd.to_numeric(upd["delivery_qty"], errors="coerce").round().astype("Int64")

            # Bulk update via COPY → temp table → single UPDATE FROM
            # Avoids ~3000 serial round-trips; completes in ~1s instead of ~5min.
            import io, os, psycopg2
            buf = io.StringIO()
            upd.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            col_list = "symbol, trade_date, " + ", ".join(delivery_cols)
            set_clause = ", ".join(f"{c}=t.{c}" for c in delivery_cols)
            col_defs = ", ".join(
                "delivery_qty BIGINT" if c == "delivery_qty" else "delivery_pct NUMERIC(6,2)"
                for c in delivery_cols
            )
            # Direct psycopg2 connection with explicit timeout — avoids pool hangs
            pg_conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "").strip(),
                port=int(os.getenv("DB_PORT", "5432")),
                dbname=os.getenv("DB_NAME", "").strip(),
                user=os.getenv("DB_USER", "").strip(),
                password=os.getenv("DB_PASSWORD", "").strip(),
                sslmode=os.getenv("DB_SSLMODE", "require").strip(),
                connect_timeout=30,
            )
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        f"CREATE TEMP TABLE _delivery_tmp "
                        f"(symbol TEXT, trade_date DATE, {col_defs}) ON COMMIT DROP"
                    )
                    cur.copy_expert(
                        f"COPY _delivery_tmp ({col_list}) FROM STDIN WITH CSV NULL '\\N'",
                        buf,
                    )
                    cur.execute(
                        f"UPDATE fact_daily_prices SET {set_clause} "
                        "FROM _delivery_tmp t "
                        "WHERE fact_daily_prices.symbol = t.symbol "
                        "  AND fact_daily_prices.trade_date = t.trade_date"
                    )
                pg_conn.commit()
                n = len(upd)
            finally:
                pg_conn.close()
            print(f"updated {n} rows with delivery data  ({f.stat().st_size//1024} KB)")
            ok += 1
        except Exception as e:
            print(f"transform/load error: {e}")
            engine = _reconnect()
            fail += 1

    print(f"\n  Done: {ok} loaded, {skip} skipped, {fail} failed")


# ── 2. Market cap ─────────────────────────────────────────────────────────────

MCAP_URL_PATTERNS = [
    # Pattern 1: bulk_bhavcopy directory (DDMMYYYY)
    NSE_ARCHIVES + "/web/sites/default/files/bulk_bhavcopy/MCAP{ddmmyyyy}.csv",
    # Pattern 2: products directory
    NSE_ARCHIVES + "/products/content/MCAP{ddmmyyyy}.csv",
    # Pattern 3: older ddmmyy format
    NSE_ARCHIVES + "/web/sites/default/files/bulk_bhavcopy/MCAP{ddmmyy}.csv",
]


def backfill_mcap(dates: list[date], session: requests.Session):
    print(f"\n{'='*60}")
    print(f"  Market Cap Backfill ({len(dates)} dates)")
    print(f"{'='*60}")

    engine = get_engine()
    ok = skip = fail = 0

    for d in dates:
        print(f"\n  {d}  ", end="", flush=True)

        # Check if already loaded
        with engine.connect() as c:
            already = c.execute(text(
                "SELECT COUNT(*) FROM fact_market_cap WHERE trade_date=:d"
            ), {"d": d}).scalar()
        if already > 100:
            print(f"already loaded ({already} rows)")
            skip += 1
            continue

        date_dir = RAW_DIR / d.strftime("%Y%m%d")
        dest = date_dir / "MCAP.csv"

        f = None
        for pattern in MCAP_URL_PATTERNS:
            url = pattern.format(
                ddmmyyyy=d.strftime("%d%m%Y"),
                ddmmyy=d.strftime("%d%m%y"),
            )
            f = _download(session, url, dest)
            if f:
                print(f"downloaded ({f.stat().st_size//1024} KB)  ", end="")
                break

        if not f:
            print("MCAP file not found at any known URL")
            fail += 1
            continue

        try:
            raw = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
            df  = transform_market_cap(raw, d)
            df  = df[df["symbol"].notna() & (df["symbol"] != "")]
            n   = load_mcap(df, d)
            print(f"loaded {n} rows")
            ok += 1
        except Exception as e:
            print(f"transform/load error: {e}")
            fail += 1

    print(f"\n  Done: {ok} loaded, {skip} skipped, {fail} failed")


# ── 3. 52-week HL ─────────────────────────────────────────────────────────────

def _fetch_hl_json(session: requests.Session, d: date, hit_type: str) -> pd.DataFrame | None:
    """
    NSE API returns either:
    - JSON with 'data' array  → convert directly
    - JSON with 'url' key     → download that CSV
    - Direct CSV text
    """
    date_str = d.strftime("%d-%b-%Y")
    url = (
        f"{NSE_BASE}/api/live-analysis-variations"
        f"?index=SECURITIES%20IN%20F%26O&date={date_str}&type={hit_type}"
    )
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 50:
            return None

        # Try JSON first
        try:
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
                return df
            if isinstance(data, dict) and "url" in data:
                csv_url = data["url"]
                if not csv_url.startswith("http"):
                    csv_url = NSE_BASE + csv_url
                r2 = session.get(csv_url, timeout=30)
                if r2.status_code == 200 and len(r2.content) > 50:
                    from io import StringIO
                    return pd.read_csv(StringIO(r2.text))
        except Exception:
            pass

        # Fallback: maybe it's direct CSV
        if b"Symbol" in r.content[:200] or b"SYMBOL" in r.content[:200]:
            from io import StringIO
            return pd.read_csv(StringIO(r.text))

        return None
    except Exception as e:
        print(f"    HL fetch error: {e}")
        return None


def _map_hl_columns(df: pd.DataFrame, hit_type: str, d: date) -> pd.DataFrame:
    """Normalize column names from different API response formats."""
    df.columns = df.columns.str.strip()

    # Detect format and map to schema
    col_map: dict = {}
    cols = {c.upper() for c in df.columns}

    # Symbol
    for candidate in ("SYMBOL", "Symbol"):
        if candidate in df.columns:
            col_map[candidate] = "symbol"
            break
    # Series
    for candidate in ("SERIES", "Series"):
        if candidate in df.columns:
            col_map[candidate] = "series"
            break
    # Security name
    for candidate in ("SECURITY NAME", "Security Name", "SECURITY", "Security",
                      "COMPANY", "COMPANY NAME", "companyName"):
        if candidate in df.columns:
            col_map[candidate] = "security_name"
            break

    df = df.rename(columns=col_map)
    df["trade_date"] = d
    df["hl_type"]    = "H" if hit_type == "52high" else "L"

    keep = [c for c in ("trade_date", "symbol", "series", "security_name", "hl_type")
            if c in df.columns]
    return df[keep]


def backfill_hl(dates: list[date], session: requests.Session):
    print(f"\n{'='*60}")
    print(f"  52-Week High/Low Backfill ({len(dates)} dates)")
    print(f"{'='*60}")
    print("  Note: API only returns current-day data reliably. Historical dates may return empty.")

    engine = get_engine()
    ok = skip = fail = 0

    for d in dates:
        print(f"\n  {d}  ", end="", flush=True)

        with engine.connect() as c:
            already = c.execute(text(
                "SELECT COUNT(*) FROM fact_hl_hits WHERE trade_date=:d"
            ), {"d": d}).scalar()
        if already > 0:
            print(f"already loaded ({already} rows)")
            skip += 1
            continue

        parts = []
        for hit_type in ("52high", "52low"):
            raw = _fetch_hl_json(session, d, hit_type)
            if raw is not None and not raw.empty:
                mapped = _map_hl_columns(raw, hit_type, d)
                if not mapped.empty:
                    parts.append(mapped)

        if not parts:
            print("no data returned (historical HL not available via live API)")
            fail += 1
            continue

        df = pd.concat(parts, ignore_index=True)
        df = df[df["symbol"].notna() & (df["symbol"].astype(str).str.strip() != "")]
        n  = load_hl(df, d)
        highs = sum(1 for p in parts if "H" in p["hl_type"].values)
        print(f"loaded {n} rows ({highs}/2 hit types)")
        ok += 1
        time.sleep(1.5)  # rate-limit

    print(f"\n  Done: {ok} loaded, {skip} skipped, {fail} failed")


# ── 4. Corporate actions ──────────────────────────────────────────────────────

def _parse_corp_action_type(purpose: str) -> str:
    p = str(purpose).lower()
    if "dividend" in p:  return "DIVIDEND"
    if "split"    in p:  return "SPLIT"
    if "bonus"    in p:  return "BONUS"
    if "rights"   in p:  return "RIGHTS"
    return "OTHER"


def backfill_corporate_actions(dates: list[date], session: requests.Session):
    print(f"\n{'='*60}")
    print(f"  Corporate Actions Backfill ({len(dates)} dates)")
    print(f"{'='*60}")

    engine = get_engine()
    ok = skip = fail = 0

    for d in dates:
        print(f"\n  {d}  ", end="", flush=True)

        with engine.connect() as c:
            already = c.execute(text(
                "SELECT COUNT(*) FROM fact_corporate_actions WHERE trade_date=:d"
            ), {"d": d}).scalar()
        if already > 0:
            print(f"already loaded ({already} rows)")
            skip += 1
            continue

        from_str = to_str = d.strftime("%d-%b-%Y")

        # Try JSON endpoint (preferred — binary-free)
        url = (
            f"{NSE_BASE}/api/corporates-corporateActions"
            f"?index=equities&from_date={from_str}&to_date={to_str}"
        )
        try:
            r = session.get(url, timeout=30)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}")
                fail += 1
                continue

            try:
                data = r.json()
            except Exception:
                print("non-JSON response")
                fail += 1
                continue

            # API returns list directly or nested under a key
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("data") or data.get("records") or []
            else:
                rows = []

            if not rows:
                print("no actions returned")
                ok += 1
                continue

            raw = pd.DataFrame(rows)
            raw.columns = raw.columns.str.strip()

            # Map known column names
            col_map = {}
            for candidate in ("symbol", "SYMBOL", "Symbol"):
                if candidate in raw.columns: col_map[candidate] = "symbol"; break
            for candidate in ("series", "SERIES", "Series"):
                if candidate in raw.columns: col_map[candidate] = "series"; break
            for candidate in ("companyName", "COMPANY NAME", "Company", "SECURITY"):
                if candidate in raw.columns: col_map[candidate] = "security_name"; break
            for candidate in ("exDate", "EX-DATE", "Ex Date", "ex_date"):
                if candidate in raw.columns: col_map[candidate] = "ex_date_raw"; break
            for candidate in ("recordDate", "RECORD DATE", "Record Date"):
                if candidate in raw.columns: col_map[candidate] = "record_date_raw"; break
            for candidate in ("purpose", "PURPOSE", "Purpose"):
                if candidate in raw.columns: col_map[candidate] = "purpose"; break

            raw = raw.rename(columns=col_map)
            raw["trade_date"]  = d
            raw["action_type"] = raw.get("purpose", pd.Series(dtype=str)).apply(_parse_corp_action_type)

            for date_col, target in (("ex_date_raw", "ex_date"), ("record_date_raw", "record_date")):
                if date_col in raw.columns:
                    raw[target] = pd.to_datetime(raw[date_col], errors="coerce")

            keep = [c for c in ("trade_date","symbol","series","security_name",
                                "record_date","ex_date","purpose","action_type")
                    if c in raw.columns]
            df = raw[keep].copy()
            df = df[df["symbol"].notna()]

            n = load_corporate_actions(df, d)
            print(f"loaded {n} rows")
            ok += 1

        except Exception as e:
            print(f"error: {e}")
            fail += 1

        time.sleep(1.5)

    print(f"\n  Done: {ok} loaded, {skip} skipped, {fail} failed")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill supplementary NSE data")
    parser.add_argument("--delivery", action="store_true", help="Delivery data only")
    parser.add_argument("--mcap",     action="store_true", help="Market cap only")
    parser.add_argument("--hl",       action="store_true", help="52-week HL only")
    parser.add_argument("--corp",     action="store_true", help="Corporate actions only")
    parser.add_argument("--date",     help="Single date YYYY-MM-DD")
    args = parser.parse_args()

    # Determine which sources to run
    run_all = not any([args.delivery, args.mcap, args.hl, args.corp])

    # Determine dates
    if args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    else:
        dates = _get_loaded_dates()
        print(f"Found {len(dates)} dates in DB: {dates[0]} → {dates[-1]}")

    # Archives don't need session priming; live API does
    need_prime = run_all or args.hl or args.corp
    session = _make_session(prime=need_prime)

    if run_all or args.delivery:
        backfill_delivery(dates, session)

    if run_all or args.mcap:
        backfill_mcap(dates, session)

    if run_all or args.hl:
        backfill_hl(dates, session)

    if run_all or args.corp:
        backfill_corporate_actions(dates, session)

    print("\n\nBackfill complete.")


if __name__ == "__main__":
    main()
