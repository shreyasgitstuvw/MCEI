"""
Generate sample NSE bhavcopy data for local testing.
Run this FIRST before anything else to create test data.

Usage:
    python scripts/generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

OUTPUT_DIR = "data/raw/sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADE_DATE = "2026-02-13"
SYMBOLS = [
    ("RELIANCE",  "RELIANCE INDUSTRIES LTD",    1419.60),
    ("TCS",       "TATA CONSULTANCY SERV LT",   2692.20),
    ("INFY",      "INFOSYS LTD",                1801.50),
    ("HDFCBANK",  "HDFC BANK LTD",               903.90),
    ("ICICIBANK", "ICICI BANK LIMITED",           800.25),
    ("WIPRO",     "WIPRO LTD",                   214.09),
    ("SBIN",      "STATE BANK OF INDIA",         1198.60),
    ("BHARTIARTL","BHARTI AIRTEL LTD",           1680.00),
    ("ITC",       "ITC LIMITED",                 415.90),
    ("LT",        "LARSEN & TOUBRO LTD",        3550.00),
    ("HINDUNILVR","HINDUSTAN UNILEVER LTD.",    2305.20),
    ("BAJFINANCE", "BAJAJ FINANCE LIMITED",     1024.75),
    ("ASIANPAINT","ASIAN PAINTS LIMITED",       2366.40),
    ("TITAN",     "TITAN COMPANY LIMITED",      4179.20),
    ("NESTLEIND", "NESTLE INDIA LIMITED",       1282.60),
    ("HINDALCO",  "HINDALCO INDUSTRIES LTD",    909.00),
    ("COALINDIA", "COAL INDIA LTD",             408.95),
    ("ADANIENT",  "ADANI ENTERPRISES LIMITED", 2136.60),
    ("ONGC",      "OIL AND NATURAL GAS CORP.", 267.40),
    ("HAL",       "HINDUSTAN AERONAUTICS LTD", 4212.40),
]

# Index rows
INDICES = [
    ("Nifty 50",       25471.10, 25807.20),
    ("NIFTY BANK",     52150.00, 52800.00),
    ("NIFTY MIDCAP 150", 21884.35, 22252.40),
    ("NIFTY IT",       36200.00, 36800.00),
]

np.random.seed(42)


def make_ohlc(base, prev):
    """Generate realistic OHLC from a base and previous close."""
    pct = np.random.uniform(-0.04, 0.04)
    close = round(prev * (1 + pct), 2)
    high  = round(max(base, close) * np.random.uniform(1.00, 1.03), 2)
    low   = round(min(base, close) * np.random.uniform(0.97, 1.00), 2)
    open_ = round(prev * np.random.uniform(0.99, 1.01), 2)
    return open_, high, low, close


# ── 1. Price data (pr) ────────────────────────────────────────────────────────
def gen_pr():
    rows = []

    # Indices
    for name, close, prev in INDICES:
        o, h, l, c = make_ohlc(close, prev)
        rows.append({
            "MKT": "Y", "SECURITY": name, "PREV_CL_PR": prev,
            "OPEN_PRICE": o, "HIGH_PRICE": h, "LOW_PRICE": l, "CLOSE_PRICE": c,
            "NET_TRDVAL": 0, "NET_TRDQTY": 0,
            "IND_SEC": "Y", "CORP_IND": " ", "TRADES": 0,
            "HI_52_WK": round(prev * 1.15, 2), "LO_52_WK": round(prev * 0.80, 2),
        })

    # Equities
    for sym, name, base in SYMBOLS:
        prev  = round(base * np.random.uniform(0.97, 1.03), 2)
        o, h, l, c = make_ohlc(base, prev)
        vol    = int(np.random.uniform(200_000, 15_000_000))
        val    = round(vol * c, 2)
        trades = int(np.random.uniform(2_000, 120_000))
        rows.append({
            "MKT": "N", "SECURITY": name, "PREV_CL_PR": prev,
            "OPEN_PRICE": o, "HIGH_PRICE": h, "LOW_PRICE": l, "CLOSE_PRICE": c,
            "NET_TRDVAL": val, "NET_TRDQTY": vol,
            "IND_SEC": "N", "CORP_IND": " ", "TRADES": trades,
            "HI_52_WK": round(base * 1.30, 2), "LO_52_WK": round(base * 0.70, 2),
        })

    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/pr_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 2. Detailed price data (pd) ───────────────────────────────────────────────
def gen_pd():
    pr = pd.read_csv(f"{OUTPUT_DIR}/pr_sample.csv")
    pr.insert(1, "SERIES", " ")
    pr.insert(2, "SYMBOL", "")

    for i, row in pr.iterrows():
        name = row["SECURITY"].split()[0]
        match = [s[0] for s in SYMBOLS if s[1].startswith(name)]
        pr.at[i, "SYMBOL"] = match[0] if match else name
        pr.at[i, "SERIES"] = "EQ" if row["MKT"] == "N" else " "

    path = f"{OUTPUT_DIR}/pd_sample.csv"
    pr.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(pr)} rows)")


# ── 3. Price bands hit (bh) ───────────────────────────────────────────────────
def gen_bh():
    upper = [
        ("APEX",     "EQ", "APEX FROZEN FOODS LIMITED",    "H"),
        ("BLUECHIP", "EQ", "BLUE CHIP INDIA LIMITED",      "H"),
        ("FAZE3Q",   "EQ", "FAZE THREE LIMITED",           "H"),
        ("GOKEX",    "EQ", "GOKALDAS EXPORTS LTD.",        "H"),
        ("AHIMSA",   "ST", "AHIMSA INDUSTRIES LTD.",       "H"),
        ("BRANDMAN", "ST", "BRANDMAN RETAIL LIMITED",      "H"),
        ("MAXVOLT",  "ST", "MAXVOLT ENERGY INDUS L",       "H"),
    ]
    lower = [
        ("BLUEJET",  "EQ", "BLUE JET HEALTHCARE LTD",     "L"),
        ("RELINFRA", "BE", "RELIANCE INFRASTRUCTU LTD",   "L"),
        ("JPASSOCIAT","BE","JAIPRAKASH ASSOCIATES LTD",   "L"),
        ("UEL",      "EQ", "UJAAS ENERGY LIMITED",        "L"),
        ("SADBHIN",  "EQ", "SADBHAV INFRA PROJ LTD.",     "L"),
    ]
    rows = [{"SYMBOL": s, "SERIES": sr, "SECURITY": sec, "HIGH/LOW": hl}
            for s, sr, sec, hl in upper + lower]
    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/bh_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 4. Corporate actions (bc) ─────────────────────────────────────────────────
def gen_bc():
    rows = [
        ("EQ","RELIANCE","RELIANCE INDUSTRIES LTD","2026-02-14","","","2026-02-13","","","INTDIV - RS 10 PER SH"),
        ("EQ","TCS",     "TATA CONSULTANCY SERV LT","2026-02-18","","","2026-02-18","","","INTDIV - RS 28 PER SH"),
        ("EQ","INFY",    "INFOSYS LTD",            "2026-02-20","","","2026-02-20","","","INTDIV - RS 21 PER SH"),
        ("EQ","SBIN",    "STATE BANK OF INDIA",     "2026-02-25","","","2026-02-25","","","INTDIV - RS 13.70 PER SH"),
        ("EQ","ITC",     "ITC LIMITED",             "2026-02-13","","","2026-02-13","","","INTDIV - RS 7.50 PER SH"),
        ("EQ","WIPRO",   "WIPRO LTD",              "2026-02-16","","","2026-02-16","","","BONUS 1:1"),
        ("EQ","TITAN",   "TITAN COMPANY LIMITED",   "2026-02-19","","","2026-02-19","","","FV SPLT FRM RS 10 TO RS 1"),
    ]
    df = pd.DataFrame(rows, columns=[
        "SERIES","SYMBOL","SECURITY","RECORD_DT","BC_STRT_DT","BC_END_DT",
        "EX_DT","ND_STRT_DT","ND_END_DT","PURPOSE"
    ])
    path = f"{OUTPUT_DIR}/bc_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 5. Gainers and losers (gl) ────────────────────────────────────────────────
def gen_gl():
    rows = []
    rows.append({"GAIN_LOSS":" ","SECURITY":"Nifty 50 Sec.","CLOSE_PRIC":" ","PREV_CL_PR":" ","PERCENT_CG":" "})
    gainers = [("BAJFINANCE",1024.75,999.10,2.57),("HAL",4212.40,4158.90,1.28),
               ("SBIN",1198.60,1192.40,0.52),("BHARTIARTL",1680.00,1660.00,1.20)]
    losers  = [("HINDUNILVR",2305.20,2409.70,-4.34),("HINDALCO",909.00,964.40,-5.74),
               ("ADANIENT",2136.60,2211.80,-3.40),("ONGC",267.40,276.35,-3.24)]
    for sym,c,p,pct in gainers:
        rows.append({"GAIN_LOSS":"G","SECURITY":sym,"CLOSE_PRIC":c,"PREV_CL_PR":p,"PERCENT_CG":pct})
    for sym,c,p,pct in losers:
        rows.append({"GAIN_LOSS":"L","SECURITY":sym,"CLOSE_PRIC":c,"PREV_CL_PR":p,"PERCENT_CG":pct})
    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/gl_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 6. 52-week high / low (hl) ────────────────────────────────────────────────
def gen_hl():
    rows = [
        ("HAL",      "EQ","HINDUSTAN AERONAUTICS LTD","H"),
        ("BHARTIARTL","EQ","BHARTI AIRTEL LTD",        "H"),
        ("BAJFINANCE","EQ","BAJAJ FINANCE LIMITED",     "H"),
        ("RELINFRA",  "BE","RELIANCE INFRASTRUCTU LTD","L"),
        ("JPASSOCIAT","BE","JAIPRAKASH ASSOCIATES LTD","L"),
    ]
    df = pd.DataFrame(rows, columns=["SYMBOL","SERIES","SECURITY","HIGH/LOW"])
    path = f"{OUTPUT_DIR}/hl_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 7. Market cap (mcap) ──────────────────────────────────────────────────────
def gen_mcap():
    rows = []
    for sym, name, base in SYMBOLS:
        # Use int() on uniform() to avoid NumPy int32 overflow on Windows
        issue_size = int(np.random.uniform(500_000_000, 5_000_000_000))
        close = round(base * np.random.uniform(0.98, 1.02), 2)
        rows.append({
            "Trade Date": TRADE_DATE, "Symbol": sym, "Series": "EQ",
            "Security Name": name, "Category": "Listed",
            "Last Trade Date": TRADE_DATE, "Face Value(Rs.)": 1.0,
            "Issue Size": issue_size, "Close Price/Paid up value(Rs.)": close,
            "Market Cap(Rs.)": round(issue_size * close, 2),
        })
    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/mcap_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 8. Top traded (tt) ────────────────────────────────────────────────────────
def gen_tt():
    top = sorted(SYMBOLS, key=lambda x: x[2] * np.random.uniform(0.8, 1.2), reverse=True)[:10]
    rows = [{"RANK":i+1,"SECURITY":name,"NET_TRADED_VALUE":round(base*int(np.random.uniform(500000,5000000)),2)}
            for i,(sym,name,base) in enumerate(top)]
    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/tt_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── 9. ETF data ───────────────────────────────────────────────────────────────
def gen_etf():
    etfs = [
        ("NIFTYBEES","NIPPON INDIA ETF NIFTY 50 BEES",291.92,290.00,"Nifty 50"),
        ("BANKBEES", "NIPPON INDIA ETF NIFTY BANK BEES",626.62,630.00,"Nifty Bank"),
        ("GOLDBEES", "NIPPON INDIA ETF GOLD BEES",128.36,127.00,"Gold"),
        ("JUNIORBEES","NIPPON INDIA ETF NIFTY NEXT 50",751.54,748.00,"NIFTY NEXT 50"),
        ("ICICIB22", "ICICI PRUDENTIAL NIFTY BEES",    18.50, 18.20,"Nifty 50"),
    ]
    rows = []
    for sym, name, close, prev, underlying in etfs:
        o, h, l, c = make_ohlc(close, prev)
        rows.append({
            "MARKET":"N","SERIES":"EQ","SYMBOL":sym,"SECURITY":name,
            "PREVIOUS CLOSE PRICE":prev,"OPEN PRICE":o,"HIGH PRICE":h,
            "LOW PRICE":l,"CLOSE PRICE":c,
            "NET TRADED VALUE":round(c*int(np.random.uniform(10000,500000)),2),
            "NET TRADED QTY":int(np.random.uniform(10000,500000)),
            "TRADES":int(np.random.uniform(100,5000)),
            "52 WEEK HIGH":round(close*1.20,2),"52 WEEK LOW":round(close*0.80,2),
            "UNDERLYING":underlying,
        })
    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/etf_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ {path}  ({len(df)} rows)")


# ── Run all generators ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating sample NSE data → {OUTPUT_DIR}/\n")
    gen_pr()
    gen_pd()
    gen_bh()
    gen_bc()
    gen_gl()
    gen_hl()
    gen_mcap()
    gen_tt()
    gen_etf()
    print(f"\n✅  All sample files written to {OUTPUT_DIR}/")
    print("Next step: python scripts/test_analytics.py")