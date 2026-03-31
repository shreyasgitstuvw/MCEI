"""
Pre-compute Analytics
======================
Runs heavy analytics modules and stores results in precomp_* tables.
Intended to run nightly via GitHub Actions after the ETL pipeline,
so the dashboard can do fast SELECTs instead of on-demand computation.

Tables written:
  precomp_regime           — market regime timeline (classify_regime_rule_based)
  precomp_volume_patterns  — volume breakouts, dry-ups, climactic events
  precomp_causality        — lead-lag market leaders

Usage:
    python scripts/precompute_analytics.py            # uses latest date in DB
    python scripts/precompute_analytics.py --date 2026-03-21
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from database.connection import get_engine


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_prices(engine) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql(
            text("SELECT * FROM fact_daily_prices ORDER BY trade_date"),
            c,
        )


def _delete_computed(engine, table: str, computed_date: date) -> None:
    with engine.begin() as c:
        c.execute(text(f"DELETE FROM {table} WHERE computed_date = :d"),
                  {"d": computed_date})


def _insert(engine, table: str, records: list[dict]) -> int:
    if not records:
        return 0
    cols = list(records[0].keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})")
    with engine.begin() as c:
        c.execute(sql, records)
    return len(records)


def _log(engine, computed_date: date, dataset: str,
         status: str, n: int, err: str | None, secs: int) -> None:
    try:
        with engine.begin() as c:
            c.execute(
                text(
                    "INSERT INTO log_ingestion "
                    "(trade_date, file_name, status, records_processed, "
                    "error_message, processing_seconds) "
                    "VALUES (:d, :f, :s, :r, :e, :t)"
                ),
                dict(d=computed_date, f=dataset, s=status,
                     r=n, e=err, t=secs),
            )
    except Exception:
        pass


def _to_python(val):
    """Convert numpy scalars to native Python types for psycopg2."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:   # NaN
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {k: _to_python(v) for k, v in row.items()}
        for row in df.where(pd.notna(df), other=None).to_dict("records")
    ]


# ── Step 2: regime ────────────────────────────────────────────────────────────

def precompute_regime(engine, pr: pd.DataFrame, computed_date: date) -> int:
    from analytics.market_regime import MarketRegimeClassifier

    eq  = pr[~pr["is_index"].fillna(False)].copy()
    idx = pr[pr["is_index"].fillna(False)].copy()

    # Build synthetic Nifty proxy if no real index rows exist
    if idx.empty or "Nifty 50" not in idx.get("security_name", pd.Series()).values:
        top100 = (eq.groupby("symbol")["net_traded_value"]
                    .sum().nlargest(100).index)
        eq100 = eq[eq["symbol"].isin(top100)]
        proxy = (eq100.groupby("trade_date")
                      .apply(lambda g: pd.Series({
                          "open_price":       (g["open_price"]  * g["net_traded_value"]).sum() / g["net_traded_value"].sum(),
                          "high_price":       (g["high_price"]  * g["net_traded_value"]).sum() / g["net_traded_value"].sum(),
                          "low_price":        (g["low_price"]   * g["net_traded_value"]).sum() / g["net_traded_value"].sum(),
                          "close_price":      (g["close_price"] * g["net_traded_value"]).sum() / g["net_traded_value"].sum(),
                          "prev_close":       (g["prev_close"]  * g["net_traded_value"]).sum() / g["net_traded_value"].sum(),
                          "net_traded_value": g["net_traded_value"].sum(),
                          "net_traded_qty":   int(g["net_traded_qty"].sum()),
                          "total_trades":     int(g["total_trades"].sum()),
                      }), include_groups=False)
                      .reset_index())
        proxy["symbol"]        = "NIFTY50_PROXY"
        proxy["security_name"] = "Nifty 50"
        proxy["series"]        = "INDEX"
        proxy["is_index"]      = True
        idx = proxy

    clf = MarketRegimeClassifier(eq, idx)
    regime_history = clf.classify_regime_rule_based()

    if regime_history.empty:
        return 0

    keep_cols = ["trade_date", "close_price", "trend", "volatility_regime",
                 "market_regime", "adx", "rsi", "macd_hist", "atr_pct"]
    out = regime_history[[c for c in keep_cols if c in regime_history.columns]].copy()
    out["computed_date"] = computed_date

    _delete_computed(engine, "precomp_regime", computed_date)
    recs = _records(out)
    return _insert(engine, "precomp_regime", recs)


# ── Step 3: volume patterns ───────────────────────────────────────────────────

def precompute_volume_patterns(engine, pr: pd.DataFrame, computed_date: date) -> int:
    from analytics.circuit_patterns import VolumePatternDetector

    eq = pr[~pr["is_index"].fillna(False)].copy()
    det = VolumePatternDetector(eq)

    total = 0
    _delete_computed(engine, "precomp_volume_patterns", computed_date)

    # Breakouts
    bk = det.detect_volume_breakout(multiplier=2.0, lookback=20)
    if not bk.empty:
        bk["pattern_type"]  = "breakout"
        bk["computed_date"] = computed_date
        bk["dryup_streak"]  = None
        bk["price_range_pct"] = None
        bk["price_direction"] = None
        total += _insert(engine, "precomp_volume_patterns", _records(bk))

    # Dry-ups
    du = det.detect_volume_dry_up(threshold=0.3, lookback=20)
    if not du.empty:
        du["pattern_type"]      = "dryup"
        du["computed_date"]     = computed_date
        du["breakout_magnitude"] = None
        du["price_change"]      = None
        du["price_range_pct"]   = None
        du["price_direction"]   = None
        total += _insert(engine, "precomp_volume_patterns", _records(du))

    # Climactic
    cl = det.analyze_climactic_volume()
    if not cl.empty:
        cl["pattern_type"]      = "climax"
        cl["computed_date"]     = computed_date
        cl["volume_ma"]         = None
        cl["breakout_magnitude"] = None
        cl["price_change"]      = None
        cl["dryup_streak"]      = None
        total += _insert(engine, "precomp_volume_patterns", _records(cl))

    return total


# ── Step 4: causality ─────────────────────────────────────────────────────────

def precompute_causality(engine, pr: pd.DataFrame, computed_date: date) -> int:
    from analytics.causality_analysis import LeadLagAnalyzer

    analyzer = LeadLagAnalyzer(pr)
    leaders  = analyzer.find_market_leaders(top_n=50)

    if leaders.empty:
        return 0

    corr_mat = analyzer.correlation_matrix(top_n=30)
    upper    = corr_mat.values[np.triu_indices(len(corr_mat), k=1)]
    avg_corr = float(np.nanmean(np.abs(upper))) if len(upper) else 0.0
    n_dates  = len(analyzer.dates)

    leaders["computed_date"] = computed_date
    leaders["avg_corr"]      = avg_corr
    leaders["n_dates"]       = n_dates

    _delete_computed(engine, "precomp_causality", computed_date)
    return _insert(engine, "precomp_causality", _records(leaders))


# ── orchestrator ──────────────────────────────────────────────────────────────

def run(computed_date: date | None = None) -> dict[str, int]:
    engine = get_engine()

    print("Loading price data …")
    t0 = time.time()
    pr = _load_prices(engine)
    print(f"  {len(pr):,} rows loaded in {time.time()-t0:.1f}s")

    if pr.empty:
        print("No price data — aborting.")
        return {}

    if computed_date is None:
        computed_date = pr["trade_date"].max()
        if hasattr(computed_date, "date"):
            computed_date = computed_date.date()

    print(f"Computed date: {computed_date}")
    results: dict[str, int] = {}

    # Regime
    print("\n[Step 2] Market regime …")
    t0 = time.time()
    try:
        n = precompute_regime(engine, pr, computed_date)
        results["regime"] = n
        _log(engine, computed_date, "precomp_regime", "SUCCESS", n, None, int(time.time()-t0))
        print(f"         OK  ({n} rows, {time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"         FAILED: {exc}")
        _log(engine, computed_date, "precomp_regime", "FAILED", 0, str(exc), int(time.time()-t0))

    # Volume patterns
    print("\n[Step 3] Volume patterns …")
    t0 = time.time()
    try:
        n = precompute_volume_patterns(engine, pr, computed_date)
        results["volume_patterns"] = n
        _log(engine, computed_date, "precomp_volume_patterns", "SUCCESS", n, None, int(time.time()-t0))
        print(f"         OK  ({n} rows, {time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"         FAILED: {exc}")
        _log(engine, computed_date, "precomp_volume_patterns", "FAILED", 0, str(exc), int(time.time()-t0))

    # Causality
    print("\n[Step 4] Causality / lead-lag …")
    t0 = time.time()
    try:
        n = precompute_causality(engine, pr, computed_date)
        results["causality"] = n
        _log(engine, computed_date, "precomp_causality", "SUCCESS", n, None, int(time.time()-t0))
        print(f"         OK  ({n} rows, {time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"         FAILED: {exc}")
        _log(engine, computed_date, "precomp_causality", "FAILED", 0, str(exc), int(time.time()-t0))

    print("\n-- Summary --------------------------------------------------")
    for k, v in results.items():
        print(f"  OK  {k:<22} {v} rows")
    print()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute analytics into DB tables")
    parser.add_argument("--date", help="Override computed_date (YYYY-MM-DD)")
    args = parser.parse_args()

    cd = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    run(cd)
