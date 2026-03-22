"""
Smoke-test all analytics modules against sample data.

Usage:
    python -m scripts.test_analytics

Run generate_sample_data.py first if you haven't already.
"""

import sys
import os

# Make sure src/ is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

SAMPLE_DIR = "data/raw/sample"


# ─── helpers ──────────────────────────────────────────────────────────────────

def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def ok(msg):
    print(f"  \u2705  {msg}")

def warn(msg):
    print(f"  \u26a0\ufe0f   {msg}")

def fail(msg, err):
    print(f"  \u274c  {msg}")
    print(f"      {type(err).__name__}: {err}")


# ─── loaders ──────────────────────────────────────────────────────────────────

def load_price_data():
    df = pd.read_csv(f"{SAMPLE_DIR}/pr_sample.csv")
    df["trade_date"] = pd.to_datetime("2026-02-13")
    symbol_map = {
        "RELIANCE INDUSTRIES LTD":   "RELIANCE",
        "TATA CONSULTANCY SERV LT":  "TCS",
        "INFOSYS LTD":               "INFY",
        "HDFC BANK LTD":             "HDFCBANK",
        "ICICI BANK LIMITED":        "ICICIBANK",
        "WIPRO LTD":                 "WIPRO",
        "STATE BANK OF INDIA":       "SBIN",
        "BHARTI AIRTEL LTD":         "BHARTIARTL",
        "ITC LIMITED":               "ITC",
        "LARSEN & TOUBRO LTD":       "LT",
        "HINDUSTAN UNILEVER LTD.":   "HINDUNILVR",
        "BAJAJ FINANCE LIMITED":     "BAJFINANCE",
        "ASIAN PAINTS LIMITED":      "ASIANPAINT",
        "TITAN COMPANY LIMITED":     "TITAN",
        "NESTLE INDIA LIMITED":      "NESTLEIND",
        "HINDALCO INDUSTRIES LTD":   "HINDALCO",
        "COAL INDIA LTD":            "COALINDIA",
        "ADANI ENTERPRISES LIMITED": "ADANIENT",
        "OIL AND NATURAL GAS CORP.": "ONGC",
        "HINDUSTAN AERONAUTICS LTD": "HAL",
    }
    df["symbol"] = df["SECURITY"].map(symbol_map).fillna(
        df["SECURITY"].str.split().str[0]
    )
    df = df.rename(columns={
        "SECURITY":    "security_name",
        "PREV_CL_PR":  "prev_close",
        "OPEN_PRICE":  "open_price",
        "HIGH_PRICE":  "high_price",
        "LOW_PRICE":   "low_price",
        "CLOSE_PRICE": "close_price",
        "NET_TRDVAL":  "net_traded_value",
        "NET_TRDQTY":  "net_traded_qty",
        "TRADES":      "total_trades",
        "HI_52_WK":    "high_52_week",
        "LO_52_WK":    "low_52_week",
    })
    df = df[df["MKT"] == "N"].copy()
    return df


def load_circuit_data():
    df = pd.read_csv(f"{SAMPLE_DIR}/bh_sample.csv")
    df["trade_date"] = pd.to_datetime("2026-02-13")
    df = df.rename(columns={"HIGH/LOW": "high_low", "SYMBOL": "symbol"})
    return df


def load_ca_data():
    df = pd.read_csv(f"{SAMPLE_DIR}/bc_sample.csv")
    # Rename ALL columns to lowercase so CorporateActionEventStudy finds them
    df = df.rename(columns={
        "SERIES":     "series",
        "SYMBOL":     "symbol",
        "SECURITY":   "security",
        "RECORD_DT":  "record_dt",
        "BC_STRT_DT": "bc_strt_dt",
        "BC_END_DT":  "bc_end_dt",
        "EX_DT":      "ex_dt",
        "ND_STRT_DT": "nd_strt_dt",
        "ND_END_DT":  "nd_end_dt",
        "PURPOSE":    "purpose",
    })
    return df


def load_etf_data():
    df = pd.read_csv(f"{SAMPLE_DIR}/etf_sample.csv")
    df["trade_date"] = pd.to_datetime("2026-02-13")
    df = df.rename(columns={
        "SYMBOL":               "symbol",
        "SECURITY":             "security_name",
        "PREVIOUS CLOSE PRICE": "prev_close",
        "OPEN PRICE":           "open_price",
        "HIGH PRICE":           "high_price",
        "LOW PRICE":            "low_price",
        "CLOSE PRICE":          "close_price",
        "NET TRADED VALUE":     "net_traded_value",
        "NET TRADED QTY":       "net_traded_qty",
        "TRADES":               "total_trades",
        "52 WEEK HIGH":         "high_52_week",
        "52 WEEK LOW":          "low_52_week",
        "UNDERLYING":           "underlying",
    })
    return df


def load_hl_data():
    df = pd.read_csv(f"{SAMPLE_DIR}/hl_sample.csv")
    df["trade_date"] = pd.to_datetime("2026-02-13")
    df = df.rename(columns={"SYMBOL": "symbol", "HIGH/LOW": "high_low"})
    return df


def load_tt_data():
    df = pd.read_csv(f"{SAMPLE_DIR}/tt_sample.csv")
    df["trade_date"] = pd.to_datetime("2026-02-13")
    df = df.rename(columns={"SECURITY": "security_name"})
    return df


# ─── tests ────────────────────────────────────────────────────────────────────

def test_microstructure(pr):
    header("Market Microstructure Analysis")
    from src.analytics.market_microstructure import MarketMicrostructureAnalyzer
    try:
        ana = MarketMicrostructureAnalyzer(pr)
        ok("Analyzer created")
    except Exception as e:
        fail("Analyzer creation", e); return

    for method, label in [
        ("calculate_overnight_gap",           "Overnight gap"),
        ("analyze_intraday_price_discovery",  "Price discovery"),
        ("calculate_liquidity_metrics",       "Liquidity metrics"),
        ("detect_volume_clusters",            "Volume clusters"),
        ("analyze_price_volume_relationship", "Price-volume relationship"),
        ("calculate_market_depth_proxy",      "Market depth proxy"),
    ]:
        try:
            result = getattr(ana, method)()
            ok(f"{label}: {len(result)} rows")
        except Exception as e:
            fail(label, e)


def test_circuit_patterns(bh, pr):
    header("Circuit Breaker & Volume Pattern Detection")
    from src.analytics.circuit_patterns import CircuitBreakerAnalyzer, VolumePatternDetector
    try:
        ana = CircuitBreakerAnalyzer(bh, pr)
        ok("CircuitBreakerAnalyzer created")
    except Exception as e:
        fail("CircuitBreakerAnalyzer creation", e); return

    for method, label in [
        ("analyze_circuit_patterns",       "Circuit patterns"),
        ("detect_consecutive_circuits",    "Consecutive circuits"),
        ("analyze_volume_on_circuit_days", "Volume on circuit days"),
    ]:
        try:
            result = getattr(ana, method)()
            ok(f"{label}: {len(result)} rows")
        except Exception as e:
            fail(label, e)

    try:
        vpd = VolumePatternDetector(pr)
        result = vpd.detect_volume_breakout()
        ok(f"Volume breakout detection: {len(result)} rows")
    except Exception as e:
        fail("VolumePatternDetector", e)


def test_corporate_actions(bc, pr):
    header("Corporate Action Event Study")
    from src.analytics.corporate_actions import CorporateActionEventStudy, CorporateActionCalendar
    try:
        study = CorporateActionEventStudy(bc, pr)
        ok("CorporateActionEventStudy created")
    except Exception as e:
        fail("CorporateActionEventStudy creation", e); return

    try:
        div = study.analyze_dividend_impact()
        ok(f"Dividend impact analysis: {len(div)} rows")
    except Exception as e:
        fail("Dividend impact analysis", e)

    try:
        cal = CorporateActionCalendar(bc)
        upcoming = cal.get_upcoming_actions(days_ahead=60)
        ok(f"Corporate action calendar: {len(upcoming)} upcoming events")
    except Exception as e:
        fail("Corporate action calendar", e)


def test_breakout_analysis(hl, pr):
    header("52-Week Breakout Analysis")
    from src.analytics.breakout_analysis import BreakoutAnalyzer, BreakoutScreener
    try:
        ana = BreakoutAnalyzer(hl, pr)
        ok("BreakoutAnalyzer created")
    except Exception as e:
        fail("BreakoutAnalyzer creation", e); return

    try:
        strength = ana.calculate_breakout_strength()
        ok(f"Breakout strength: {len(strength)} rows")
    except Exception as e:
        fail("Breakout strength", e)

    try:
        screener = BreakoutScreener(pr)
        near_high = screener.screen_near_52w_high(threshold_pct=10)
        near_low  = screener.screen_near_52w_low(threshold_pct=10)
        ok(f"Near 52W high: {len(near_high)} stocks  |  Near 52W low: {len(near_low)} stocks")
    except Exception as e:
        fail("Breakout screener", e)


def test_etf_smart_money(etf, pr, tt):
    header("ETF Arbitrage & Smart Money Tracking")
    from src.analytics.etf_smart_money import SmartMoneyTracker
    try:
        tracker = SmartMoneyTracker(pr, tt, etf)
        ok("SmartMoneyTracker created")
    except Exception as e:
        fail("SmartMoneyTracker creation", e); return

    try:
        inst = tracker.identify_institutional_buying(
            volume_threshold=1.0, value_threshold=100_000
        )
        ok(f"Institutional buying: {len(inst)} candidates")
    except Exception as e:
        fail("Institutional buying detection", e)

    try:
        mfi = tracker.calculate_money_flow_index(window=3)
        ok(f"Money Flow Index: {len(mfi)} rows")
    except Exception as e:
        fail("Money Flow Index", e)


def test_market_regime(pr):
    header("Market Regime Classification")

    # Import inside function so missing sklearn doesn't crash at module load
    try:
        from src.analytics.market_regime import MarketRegimeClassifier, RegimeBasedStrategy
    except ModuleNotFoundError as e:
        warn(f"Import skipped — {e}")
        warn("Fix:  pip install scikit-learn")
        return
    except Exception as e:
        fail("market_regime import", e); return

    dates  = pd.date_range("2025-09-01", periods=60, freq="B")
    prices = 25000 + np.cumsum(np.random.randn(60) * 100)
    index_df = pd.DataFrame({
        "trade_date":     dates,
        "security_name":  "Nifty 50",
        "open_price":     prices * 0.99,
        "high_price":     prices * 1.01,
        "low_price":      prices * 0.98,
        "close_price":    prices,
        "prev_close":     np.roll(prices, 1),
        "net_traded_qty": (np.random.uniform(1_000_000, 5_000_000, 60)).astype(int),
        "total_trades":   (np.random.uniform(10_000,    50_000,    60)).astype(int),
    })
    pr_multi = pd.concat(
        [pr.assign(trade_date=d) for d in dates[-10:]],
        ignore_index=True,
    )

    try:
        clf = MarketRegimeClassifier(pr_multi, index_df)
        ok("MarketRegimeClassifier created")
    except Exception as e:
        fail("MarketRegimeClassifier creation", e); return

    for method, label in [
        ("calculate_trend_indicators",      "Trend indicators"),
        ("calculate_volatility_indicators", "Volatility indicators"),
        ("calculate_momentum_indicators",   "Momentum indicators"),
        ("calculate_breadth_indicators",    "Breadth indicators"),
        ("classify_regime_rule_based",      "Rule-based regime classification"),
    ]:
        try:
            result = getattr(clf, method)()
            ok(f"{label}: {len(result)} rows")
        except Exception as e:
            fail(label, e)

    # ML clustering — optional, needs scikit-learn
    try:
        result = clf.classify_regime_ml_based(n_regimes=4)
        ok(f"ML-based regime classification: {len(result)} rows")
    except ImportError:
        warn("ML regime skipped — run:  pip install scikit-learn")
    except Exception as e:
        fail("ML regime classification", e)

    try:
        strategy = RegimeBasedStrategy.get_strategy_for_regime(
            "Strong Uptrend / Low Volatility"
        )
        ok(f"Strategy recommendation: '{strategy['action']}'")
    except Exception as e:
        fail("Strategy recommendation", e)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(f"{SAMPLE_DIR}/pr_sample.csv"):
        print("\u274c  Sample data not found.")
        print("    Run:  python -m scripts.generate_sample_data")
        sys.exit(1)

    print("\n\U0001f52c  NSE Analytics \u2013 Module Smoke Tests")
    print(f"    Sample data: {SAMPLE_DIR}/")
    print(f"    Python:      {sys.version.split()[0]}")

    pr  = load_price_data()
    bh  = load_circuit_data()
    bc  = load_ca_data()
    etf = load_etf_data()
    hl  = load_hl_data()
    tt  = load_tt_data()

    test_microstructure(pr)
    test_circuit_patterns(bh, pr)
    test_corporate_actions(bc, pr)
    test_breakout_analysis(hl, pr)
    test_etf_smart_money(etf, pr, tt)
    test_market_regime(pr)

    print("\n" + "=" * 60)
    print("  All tests complete.")
    print("  \u2705 = working   \u26a0\ufe0f  = skipped (install package)   \u274c = needs fix")
    print("=" * 60 + "\n")