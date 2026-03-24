"""
Daily Pipeline Orchestrator
============================
Runs all data ingestion steps for a given trading date:

  Step 1: UDIFF bhavcopy ETL          (always — no session needed)
  Step 2: Delivery data backfill       (always — no session needed)
  Step 3: 52-week HL from NSE API      (best-effort — requires live NSE session)
  Step 4: Corporate actions from NSE   (best-effort — requires live NSE session)

Usage:
    python scripts/run_full_daily.py                   # today
    python scripts/run_full_daily.py 2026-03-21        # specific date
    python scripts/run_full_daily.py --skip-live       # skip Steps 3+4
    python scripts/run_full_daily.py --delivery-only   # Step 2 only
"""

from __future__ import annotations
import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import requests


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Prime an NSE session by hitting the homepage first."""
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
    }
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(NSE_BASE, timeout=10)
        time.sleep(1)
    except Exception as e:
        print(f"  ⚠️  NSE session priming failed: {e}")
    return session


def _parse_date(s: str | None) -> date:
    if s is None:
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()


# ── pipeline steps ────────────────────────────────────────────────────────────

def step1_udiff(target_date: date) -> str:
    """Run UDIFF bhavcopy ETL for target_date."""
    from src.etl.pipeline import run_for_date
    try:
        result = run_for_date(target_date)
        if result:
            datasets = result.get("datasets", {})
            status = result.get("status", "")
            if status not in ("", None) and not str(status).startswith("OK") and datasets == {}:
                err = result.get("error", status)
                return f"FAILED: {err}"
            rows = sum(datasets.values()) if datasets else 0
            return f"OK ({rows} rows across {len(datasets)} tables)"
        return "OK (0 rows — date may already be loaded or file unavailable)"
    except Exception as exc:
        return f"FAILED: {exc}"


def step2_delivery(target_date: date) -> str:
    """Download and load delivery data for target_date."""
    from backfill_supplementary import backfill_delivery, _make_session as _bs_make_session
    session = _bs_make_session()
    try:
        loaded, skipped = backfill_delivery([target_date], session)
        return f"OK ({loaded} dates loaded, {skipped} skipped)"
    except TypeError:
        # backfill_delivery may not return a tuple — handle gracefully
        try:
            from backfill_supplementary import backfill_delivery as bd
            bd([target_date], session)
            return "OK"
        except Exception as exc:
            return f"FAILED: {exc}"
    except Exception as exc:
        return f"FAILED: {exc}"


def step3_hl(target_date: date, session: requests.Session) -> str:
    """Fetch 52-week HL data for target_date from live NSE API."""
    from backfill_supplementary import backfill_hl
    try:
        backfill_hl([target_date], session)
        return "OK"
    except Exception as exc:
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            return "SKIPPED (timeout)"
        return f"SKIPPED ({exc})"


def step4_corp(target_date: date, session: requests.Session) -> str:
    """Fetch corporate actions for target_date from live NSE API."""
    from backfill_supplementary import backfill_corporate_actions
    try:
        backfill_corporate_actions([target_date], session)
        return "OK"
    except Exception as exc:
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            return "SKIPPED (timeout)"
        return f"SKIPPED ({exc})"


# ── orchestrator ──────────────────────────────────────────────────────────────

def run(target_date: date, skip_live: bool = False,
        delivery_only: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}

    if not delivery_only:
        print(f"\n[Step 1] UDIFF bhavcopy ETL for {target_date} ...")
        t0 = time.time()
        results["step1_udiff"] = step1_udiff(target_date)
        print(f"         {results['step1_udiff']}  ({time.time()-t0:.1f}s)")

    print(f"\n[Step 2] Delivery data for {target_date} ...")
    t0 = time.time()
    results["step2_delivery"] = step2_delivery(target_date)
    print(f"         {results['step2_delivery']}  ({time.time()-t0:.1f}s)")

    if not skip_live and not delivery_only:
        print(f"\n[Step 3] 52-week HL data for {target_date} ...")
        session = _make_session()
        t0 = time.time()
        results["step3_hl"] = step3_hl(target_date, session)
        print(f"         {results['step3_hl']}  ({time.time()-t0:.1f}s)")

        print(f"\n[Step 4] Corporate actions for {target_date} ...")
        t0 = time.time()
        results["step4_corp"] = step4_corp(target_date, session)
        print(f"         {results['step4_corp']}  ({time.time()-t0:.1f}s)")
    else:
        results["step3_hl"]   = "SKIPPED (--skip-live)"
        results["step4_corp"] = "SKIPPED (--skip-live)"

    print("\n── Summary ──────────────────────────────────────────")
    for k, v in results.items():
        status_icon = "✅" if v.startswith("OK") else ("⏭️" if v.startswith("SKIPPED") else "❌")
        print(f"  {status_icon}  {k:<20} {v}")
    print()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE daily pipeline orchestrator")
    parser.add_argument("date", nargs="?", help="Trading date YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip Steps 3+4 (no NSE live API session required)")
    parser.add_argument("--delivery-only", action="store_true",
                        help="Run Step 2 only (delivery data update)")
    args = parser.parse_args()

    target = _parse_date(args.date)
    print(f"NSE Daily Pipeline — {target}")
    run(target, skip_live=args.skip_live, delivery_only=args.delivery_only)
