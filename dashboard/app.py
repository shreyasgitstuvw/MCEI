"""
NSE Advanced Analytics Dashboard
All pages wired to real database + analytics modules.
Falls back to sample data gracefully if DB is unavailable.
"""

from __future__ import annotations
import os, sys
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# â”€â”€ path setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from analytics.market_microstructure  import MarketMicrostructureAnalyzer
from analytics.etf_smart_money        import SmartMoneyTracker
from analytics.market_regime          import MarketRegimeClassifier, RegimeBasedStrategy
from analytics.breakout_analysis      import BreakoutAnalyzer
from analytics.circuit_patterns       import VolumePatternDetector
from analytics.causality_analysis     import LeadLagAnalyzer

# â”€â”€ page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="NSE Advanced Analytics",
    page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1f77b4;
             border-bottom:3px solid #1f77b4;padding-bottom:.5rem;margin-bottom:1.5rem}
.metric-card{background:linear-gradient(135deg,#667eea,#764ba2);
             padding:1rem;border-radius:10px;color:#fff;margin-bottom:.5rem}
.success-card{background:linear-gradient(135deg,#11998e,#38ef7d)}
.warning-card{background:linear-gradient(135deg,#f093fb,#f5576c)}
.info-card   {background:linear-gradient(135deg,#4facfe,#00f2fe)}
</style>
""", unsafe_allow_html=True)


# â”€â”€ DB helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(ttl=60)   # recheck DB connection once per minute
def _db_available() -> bool:
    try:
        from database.connection import get_engine
        from sqlalchemy import text
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception as e:
        # Store error for sidebar display
        import streamlit as _st
        _st.session_state["_db_error"] = str(e)
        return False


@st.cache_data(ttl=300)   # refresh every 5 min
def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    from database.connection import get_engine
    from sqlalchemy import text
    with get_engine().connect() as c:
        return pd.read_sql(text(sql), c, params=params or {})


@st.cache_data(ttl=300)
def _latest_date() -> date:
    try:
        row = _query("SELECT MAX(trade_date) AS d FROM fact_daily_prices")
        return row["d"].iloc[0].date() if not row.empty else date.today()
    except Exception:
        return date.today()


# â”€â”€ data loaders (DB â†’ DataFrame) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(ttl=300)
def load_prices(trade_date: date) -> pd.DataFrame:
    return _query(
        "SELECT * FROM fact_daily_prices WHERE trade_date=:d",
        {"d": trade_date}
    )


@st.cache_data(ttl=300)
def load_prices_range(days: int = 120) -> pd.DataFrame:
    """Load up to `days` of price history. All pages share the 120-day cache entry;
    pages needing a shorter window filter by date in Python."""
    return _query(
        "SELECT * FROM fact_daily_prices "
        "WHERE trade_date >= CURRENT_DATE - :n ORDER BY trade_date",
        {"n": days}
    )


def _tail_dates(df: pd.DataFrame, n_dates: int) -> pd.DataFrame:
    """Return only the last `n_dates` distinct trade_dates from a price range DataFrame."""
    if df.empty or n_dates <= 0:
        return df
    cutoff_dates = sorted(df["trade_date"].unique())[-n_dates:]
    return df[df["trade_date"].isin(cutoff_dates)]


@st.cache_data(ttl=300)
def load_circuits(trade_date: date) -> pd.DataFrame:
    return _query(
        "SELECT * FROM fact_circuit_hits WHERE trade_date=:d",
        {"d": trade_date}
    )


@st.cache_data(ttl=300)
def load_circuits_range(days: int = 30) -> pd.DataFrame:
    return _query(
        "SELECT * FROM fact_circuit_hits "
        "WHERE trade_date >= CURRENT_DATE - :n ORDER BY trade_date",
        {"n": days}
    )


@st.cache_data(ttl=300)
def load_corp_actions(days: int = 30) -> pd.DataFrame:
    df = _query(
        "SELECT * FROM fact_corporate_actions "
        "WHERE trade_date >= CURRENT_DATE - :n ORDER BY ex_date",
        {"n": days}
    )
    # rename to match analytics module expectations
    return df.rename(columns={"ex_date": "ex_dt", "security_name": "security",
                               "action_type": "action_type"})


@st.cache_data(ttl=300)
def load_etf(trade_date: date) -> pd.DataFrame:
    return _query(
        "SELECT * FROM fact_etf_prices WHERE trade_date=:d",
        {"d": trade_date}
    )


@st.cache_data(ttl=300)
def load_top_traded(trade_date: date) -> pd.DataFrame:
    return _query(
        "SELECT * FROM fact_top_traded WHERE trade_date=:d ORDER BY rank",
        {"d": trade_date}
    )


@st.cache_data(ttl=300)
def load_hl(trade_date: date) -> pd.DataFrame:
    return _query(
        "SELECT * FROM fact_hl_hits WHERE trade_date=:d",
        {"d": trade_date}
    )


@st.cache_data(ttl=300)
def load_breadth(days: int = 30) -> pd.DataFrame:
    return _query(
        "SELECT * FROM v_market_breadth ORDER BY trade_date DESC LIMIT :n",
        {"n": days}
    )


@st.cache_data(ttl=300)
def load_precomp_regime() -> pd.DataFrame:
    return _query("SELECT * FROM precomp_regime ORDER BY trade_date")


@st.cache_data(ttl=300)
def load_precomp_volume(pattern_type: str) -> pd.DataFrame:
    return _query(
        "SELECT * FROM precomp_volume_patterns WHERE pattern_type=:t "
        "ORDER BY trade_date, symbol",
        {"t": pattern_type},
    )


@st.cache_data(ttl=300)
def load_precomp_causality() -> pd.DataFrame:
    return _query(
        "SELECT * FROM precomp_causality "
        "WHERE computed_date = (SELECT MAX(computed_date) FROM precomp_causality) "
        "ORDER BY lag1_corr DESC"
    )


# â”€â”€ sample-data fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _sample_prices() -> pd.DataFrame:
    import glob
    files = glob.glob("data/raw/sample/pr_sample.csv")
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    df["trade_date"] = pd.to_datetime(date.today())
    col_map = {
        "SECURITY":"security_name","PREV_CL_PR":"prev_close","OPEN_PRICE":"open_price",
        "HIGH_PRICE":"high_price","LOW_PRICE":"low_price","CLOSE_PRICE":"close_price",
        "NET_TRDVAL":"net_traded_value","NET_TRDQTY":"net_traded_qty",
        "TRADES":"total_trades","HI_52_WK":"high_52_week","LO_52_WK":"low_52_week",
    }
    df = df.rename(columns=col_map)
    df["is_index"] = df.get("MKT", "N") == "Y"
    df["symbol"]   = df["security_name"].str.split().str[0]
    return df[df["is_index"] == False].copy()


# â”€â”€ sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _sidebar() -> tuple[str, date, bool, str]:
    with st.sidebar:
        st.markdown("## ðŸ“Š NSE Analytics")
        st.markdown("---")

        db_ok = _db_available()
        if db_ok:
            st.success("DB online")
        else:
            st.warning("DB offline -- using sample data")
            err = st.session_state.get("_db_error", "")
            if err:
                with st.expander("DB error details"):
                    st.code(err)

        latest = _latest_date() if db_ok else date.today()
        sel_date = st.date_input("ðŸ“… Trading Date", value=latest,
                                 max_value=date.today())

        view_period = st.radio("ðŸ“… View Period", ["Daily", "Weekly", "Monthly"],
                               horizontal=True)

        if st.button("ðŸ”„ Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        page = st.radio("Navigation", [
            "ðŸ  Market Overview",
            "ðŸ“‹ Market Summary",
            "ðŸ”¬ Microstructure",
            "ðŸ’° Smart Money",
            "ðŸ“ˆ Market Regime",
            "ðŸš€ Breakout Analysis",
            "ðŸ“Š Volume Patterns",
            "ðŸ”— Causality Analysis",
            "ðŸ—„ï¸ Data Explorer",
        ])

        if db_ok:
            st.markdown("---")
            st.caption("âš¡ Run ETL manually:")
            if st.button("â–¶ Run today's ETL"):
                with st.spinner("Running pipeline â€¦"):
                    try:
                        from etl.pipeline import run_for_date
                        r = run_for_date(date.today())
                        st.success(f"Done: {r['status']}")
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(str(exc))

    return page, sel_date, db_ok, view_period


# â”€â”€ page: market overview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def page_overview(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ  Market Overview</div>',
                unsafe_allow_html=True)

    pr_df = load_prices(sel_date) if db_ok else _sample_prices()
    breadth_df = load_breadth() if db_ok else pd.DataFrame()

    if pr_df.empty:
        st.warning(f"No price data for {sel_date}. Run the ETL first.")
        return

    eq = pr_df[pr_df.get("is_index", pd.Series(False)) == False]

    # â”€â”€ top metrics â”€â”€
    advances  = int((eq["close_price"] > eq["prev_close"]).sum())
    declines  = int((eq["close_price"] < eq["prev_close"]).sum())
    unchanged = int(len(eq) - advances - declines)
    total_val = eq["net_traded_value"].sum() / 1e9

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ðŸŸ¢ Advances",    advances)
    c2.metric("ðŸ”´ Declines",    declines)
    c3.metric("â¬œ Unchanged",   unchanged)
    c4.metric("ðŸ’° Turnover (â‚¹B)", f"{total_val:.1f}")

    # â”€â”€ breadth chart â”€â”€
    if not breadth_df.empty:
        bdf = breadth_df.copy()
        bdf["trade_date"] = pd.to_datetime(bdf["trade_date"])
        if view_period != "Daily":
            freq = "W-FRI" if view_period == "Weekly" else "ME"
            bdf = (bdf.set_index("trade_date")
                      .resample(freq)
                      .agg({"advances": "sum", "declines": "sum"})
                      .reset_index())
        period_label = f"â€” {view_period} View" if view_period != "Daily" else ""
        fig = go.Figure()
        fig.add_bar(x=bdf["trade_date"], y=bdf["advances"],
                    name="Advances", marker_color="#2ecc71")
        fig.add_bar(x=bdf["trade_date"], y=bdf["declines"],
                    name="Declines", marker_color="#e74c3c")
        fig.update_layout(barmode="group",
                          title=f"Market Breadth â€” Advances vs Declines {period_label}",
                          height=350, xaxis_title="Date", yaxis_title="Stocks")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Breadth history needs multiple days of data in the database.")

    # â”€â”€ top gainers / losers â”€â”€
    eq2 = eq.copy()
    eq2["pct_change"] = ((eq2["close_price"] - eq2["prev_close"]) /
                          eq2["prev_close"].replace(0, np.nan)) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ðŸ† Top 10 Gainers")
        g = (eq2.nlargest(10, "pct_change")
               [["symbol","close_price","pct_change","net_traded_value"]]
               .rename(columns={"pct_change":"Chg %","close_price":"Close",
                                 "net_traded_value":"Value (â‚¹)"}))
        st.dataframe(g.style.format({"Chg %":"{:.2f}","Close":"{:.2f}",
                                     "Value (â‚¹)":"{:,.0f}"}),
                     use_container_width=True, hide_index=True)
    with col2:
        st.subheader("ðŸ“‰ Top 10 Losers")
        l = (eq2.nsmallest(10, "pct_change")
               [["symbol","close_price","pct_change","net_traded_value"]]
               .rename(columns={"pct_change":"Chg %","close_price":"Close",
                                 "net_traded_value":"Value (â‚¹)"}))
        st.dataframe(l.style.format({"Chg %":"{:.2f}","Close":"{:.2f}",
                                     "Value (â‚¹)":"{:,.0f}"}),
                     use_container_width=True, hide_index=True)


# â”€â”€ page: market summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def page_market_summary(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ“‹ Market Summary</div>',
                unsafe_allow_html=True)

    pr_df     = load_prices(sel_date)     if db_ok else _sample_prices()
    tt_df     = load_top_traded(sel_date) if db_ok else pd.DataFrame()
    etf_df    = load_etf(sel_date)        if db_ok else pd.DataFrame()
    pr_range  = _tail_dates(load_prices_range(), 30) if db_ok else pd.DataFrame()

    if pr_df.empty:
        st.warning(f"No price data for {sel_date}. Run the ETL first.")
        return

    eq = pr_df[pr_df.get("is_index", pd.Series(False)) == False].copy()
    eq["pct_change"] = ((eq["close_price"] - eq["prev_close"]) /
                         eq["prev_close"].replace(0, np.nan)) * 100

    # â”€â”€ market pulse â”€â”€
    st.subheader("ðŸ“¡ Market Pulse")
    advances  = int((eq["close_price"] > eq["prev_close"]).sum())
    declines  = int((eq["close_price"] < eq["prev_close"]).sum())
    unchanged = int(len(eq) - advances - declines)
    total_val = eq["net_traded_value"].sum() / 1e9
    avg_chg   = eq["pct_change"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ðŸŸ¢ Advances",      advances)
    c2.metric("ðŸ”´ Declines",      declines)
    c3.metric("â¬œ Unchanged",     unchanged)
    c4.metric("ðŸ’° Turnover (â‚¹B)", f"{total_val:.1f}")
    c5.metric("ðŸ“Š Avg Move",      f"{avg_chg:+.2f}%")

    # A/D ratio trend
    if not pr_range.empty:
        eq_r = pr_range[pr_range.get("is_index", pd.Series(False)) == False].copy()
        eq_r["trade_date"] = pd.to_datetime(eq_r["trade_date"])
        breadth = (eq_r.groupby("trade_date")
                       .apply(lambda g: pd.Series({
                           "advances": int((g["close_price"] > g["prev_close"]).sum()),
                           "declines": int((g["close_price"] < g["prev_close"]).sum()),
                       }))
                       .reset_index())
        if not breadth.empty:
            if view_period != "Daily":
                freq = "W-FRI" if view_period == "Weekly" else "ME"
                breadth["trade_date"] = pd.to_datetime(breadth["trade_date"])
                breadth = (breadth.set_index("trade_date")
                                  .resample(freq)
                                  .agg({"advances": "sum", "declines": "sum"})
                                  .reset_index())
            breadth["ad_ratio"] = breadth["advances"] / (breadth["declines"].replace(0, 1))
            period_label = f"({view_period})" if view_period != "Daily" else "â€” Last 30 Days"
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Advances vs Declines", "A/D Ratio"),
                                row_heights=[0.6, 0.4])
            fig.add_bar(row=1, col=1, x=breadth["trade_date"], y=breadth["advances"],
                        name="Advances", marker_color="#2ecc71")
            fig.add_bar(row=1, col=1, x=breadth["trade_date"], y=-breadth["declines"],
                        name="Declines", marker_color="#e74c3c")
            fig.add_scatter(row=2, col=1, x=breadth["trade_date"], y=breadth["ad_ratio"],
                            name="A/D Ratio", line=dict(color="#3498db", width=2))
            fig.update_layout(height=400, barmode="relative",
                              showlegend=True, title=f"Market Breadth {period_label}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # â”€â”€ momentum scan â”€â”€
    st.subheader("ðŸš€ Multi-Period Momentum")
    if not pr_range.empty:
        eq_r = pr_range[pr_range.get("is_index", pd.Series(False)) == False].copy()
        dates_avail = sorted(eq_r["trade_date"].unique())
        latest_date = dates_avail[-1] if dates_avail else sel_date

        def _period_return(eq_all, n_days):
            if len(dates_avail) < n_days + 1:
                s = pd.Series(dtype=float, name=f"ret_{n_days}d")
                s.index.name = "symbol"
                return s
            past_date = dates_avail[-(n_days + 1)]
            cur  = eq_all[eq_all["trade_date"] == latest_date][["symbol","close_price"]].set_index("symbol")
            past = eq_all[eq_all["trade_date"] == past_date][["symbol","close_price"]].set_index("symbol")
            joined = cur.join(past, lsuffix="_cur", rsuffix="_past").dropna()
            return ((joined["close_price_cur"] - joined["close_price_past"]) /
                    joined["close_price_past"].replace(0, np.nan) * 100).rename(f"ret_{n_days}d")

        r1  = _period_return(eq_r, 1)
        r5  = _period_return(eq_r, 5)
        r20 = _period_return(eq_r, 20)

        mom = pd.concat([r1, r5, r20], axis=1).dropna(how="all").reset_index()
        if not mom.empty:
            col_labels = {c: c.replace("ret_", "").replace("d", "D Ret%") for c in mom.columns if isinstance(c, str) and c != "symbol"}
            mom = mom.rename(columns=col_labels)
            if "1D Ret%" in mom.columns:
                tab_g, tab_l = st.tabs(["ðŸ† Top Gainers (5D)", "ðŸ“‰ Top Losers (5D)"])
                with tab_g:
                    if "5D Ret%" in mom.columns:
                        st.dataframe(mom.nlargest(15, "5D Ret%").style.format(
                            {c: "{:.2f}" for c in mom.columns if isinstance(c, str) and "Ret" in c}),
                            use_container_width=True, hide_index=True)
                with tab_l:
                    if "5D Ret%" in mom.columns:
                        st.dataframe(mom.nsmallest(15, "5D Ret%").style.format(
                            {c: "{:.2f}" for c in mom.columns if isinstance(c, str) and "Ret" in c}),
                            use_container_width=True, hide_index=True)
        else:
            st.info("Need price data across multiple dates for momentum calculation.")
    else:
        st.info("Price history unavailable.")

    st.markdown("---")

    # â”€â”€ volume anomalies â”€â”€
    st.subheader("ðŸ“Š Volume Anomalies Today")
    if not eq.empty and not pr_range.empty:
        eq_r = pr_range[pr_range.get("is_index", pd.Series(False)) == False].copy()
        avg_vol = (eq_r.groupby("symbol")["net_traded_qty"]
                       .mean().rename("avg_qty"))
        today_vol = eq[["symbol","net_traded_qty","net_traded_value","close_price","pct_change"]].set_index("symbol")
        vol_comp = today_vol.join(avg_vol).dropna(subset=["avg_qty"])
        vol_comp["vol_ratio"] = vol_comp["net_traded_qty"] / vol_comp["avg_qty"].replace(0, np.nan)
        anomalies = vol_comp[vol_comp["vol_ratio"] > 2].sort_values("vol_ratio", ascending=False)
        st.metric("Stocks with 2x+ avg volume", len(anomalies))
        if not anomalies.empty:
            fig = px.scatter(anomalies.head(30).reset_index(), x="symbol", y="vol_ratio",
                             size="net_traded_value", color="pct_change",
                             color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                             title="Volume Anomalies (bubble size = traded value)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(anomalies.head(20).reset_index()
                         [["symbol","vol_ratio","pct_change","net_traded_value"]]
                         .style.format({"vol_ratio":"{:.1f}x","pct_change":"{:+.2f}%",
                                        "net_traded_value":"{:,.0f}"}),
                         use_container_width=True, hide_index=True)
    else:
        st.info("Need multi-day price data for volume anomaly comparison.")

    st.markdown("---")

    # â”€â”€ top traded â”€â”€
    st.subheader("ðŸ’¹ Top Traded by Value")
    if not tt_df.empty:
        disp = [c for c in ["rank","symbol","security_name","close_price",
                             "net_traded_value","net_traded_qty"] if c in tt_df.columns]
        st.dataframe(tt_df[disp].style.format(
            {c: "{:,.0f}" for c in ["net_traded_value","net_traded_qty"] if c in tt_df.columns}),
            use_container_width=True, hide_index=True)
    else:
        top25 = eq.nlargest(25, "net_traded_value")[
            ["symbol","close_price","net_traded_value","pct_change"]
        ].reset_index(drop=True)
        top25.index += 1
        st.caption("(Derived from today's prices â€” run backfill to populate fact_top_traded)")
        st.dataframe(top25.style.format(
            {"net_traded_value":"{:,.0f}","pct_change":"{:+.2f}%","close_price":"{:.2f}"}),
            use_container_width=True)

    # â”€â”€ etf snapshot â”€â”€
    if not etf_df.empty:
        st.markdown("---")
        st.subheader("ðŸ”„ ETF Snapshot")
        etf_eq = etf_df.copy()
        if "prev_close" in etf_eq.columns:
            etf_eq["pct_chg"] = ((etf_eq["close_price"] - etf_eq["prev_close"]) /
                                  etf_eq["prev_close"].replace(0, np.nan)) * 100
        disp = [c for c in ["symbol","security_name","close_price","pct_chg",
                             "net_traded_value","underlying"] if c in etf_eq.columns]
        st.dataframe(etf_eq[disp].style.format(
            {"close_price":"{:.2f}","pct_chg":"{:+.2f}%",
             "net_traded_value":"{:,.0f}"}),
            use_container_width=True, hide_index=True)


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _resample_prices(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Resample a multi-date price DataFrame to weekly or monthly periods.

    - close_price / prev_close: last / first value of period
    - high_price: max; low_price: min
    - net_traded_value / net_traded_qty / total_trades: sum

    Returns a DataFrame with the same columns; trade_date = period-end date.
    """
    if period == "Daily" or df.empty:
        return df
    freq = "W-FRI" if period == "Weekly" else "ME"
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    agg: dict = {
        "open_price":       "first",
        "high_price":       "max",
        "low_price":        "min",
        "close_price":      "last",
        "prev_close":       "first",
        "net_traded_value": "sum",
        "net_traded_qty":   "sum",
        "total_trades":     "sum",
    }
    agg_use = {k: v for k, v in agg.items() if k in df.columns}
    result = (
        df.groupby(["symbol", pd.Grouper(key="trade_date", freq=freq)])
          .agg(agg_use)
          .reset_index()
    )
    # Carry forward non-aggregated columns (series, security_name, is_index â€¦)
    static_cols = [c for c in ["series", "security_name", "is_index"]
                   if c in df.columns]
    if static_cols:
        meta = df.drop_duplicates("symbol")[["symbol"] + static_cols]
        result = result.merge(meta, on="symbol", how="left")
    return result


def _safe_size(series: pd.Series, scale: float = 1e6) -> pd.Series:
    """Return a size series safe for plotly (strictly positive, reasonable range)."""
    s = series.fillna(scale).clip(lower=1)
    return (s / scale).clip(lower=0.05, upper=200)


def _vc_df(series: pd.Series) -> pd.DataFrame:
    """
    value_counts() â†’ tidy DataFrame with columns [series.name, 'count'].
    Works identically across pandas 1.x and 2.x.
    """
    counts = series.value_counts()
    return pd.DataFrame({series.name: counts.index, "count": counts.values})


# â”€â”€ page: microstructure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def page_microstructure(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ”¬ Market Microstructure</div>',
                unsafe_allow_html=True)

    pr_df   = load_prices(sel_date)       if db_ok else _sample_prices()
    pr_range = _tail_dates(load_prices_range(), 30) if db_ok else pd.DataFrame()

    if pr_df.empty:
        st.warning("No price data available.")
        return

    is_idx = pr_df["is_index"] if "is_index" in pr_df.columns else pd.Series(False, index=pr_df.index)
    eq = pr_df[~is_idx.astype(bool)].copy().reset_index(drop=True)
    ana = MarketMicrostructureAnalyzer(eq)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "ðŸ’§ Liquidity & Depth",
        "ðŸ“¦ Delivery Quality",
        "ðŸ•¯ Candle Structure",
        "âš– Price Position",
        "ðŸ“Š Volume Analysis",
        "ðŸ”„ Momentum-Volume Matrix",
        "ðŸŒ™ Overnight Gaps",
        "ðŸ“‰ Volatility",
    ])

    # â”€â”€ Tab 1: Liquidity & Depth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab1:
        try:
            liq   = ana.calculate_liquidity_metrics()
            depth = ana.calculate_market_depth_proxy()

            if not liq.empty:
                hl  = int((liq["liquidity_class"] == "HIGHLY_LIQUID").sum()) if "liquidity_class" in liq.columns else 0
                ill = int((liq["liquidity_class"] == "ILLIQUID").sum())      if "liquidity_class" in liq.columns else 0
                hil = int((liq["liquidity_class"] == "HIGHLY_ILLIQUID").sum()) if "liquidity_class" in liq.columns else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Liquidity Score", f"{liq['liquidity_score'].mean():.1f}")
                c2.metric("Highly Liquid",   hl)
                c3.metric("Illiquid",        ill)
                c4.metric("Highly Illiquid", hil)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.bar(
                        liq.nlargest(20, "liquidity_score"),
                        x="symbol", y="liquidity_score",
                        color="liquidity_score", color_continuous_scale="RdYlGn",
                        title="Top 20 Most Liquid Stocks",
                    )
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    class_counts = liq["liquidity_class"].value_counts().reset_index()
                    class_counts.columns = ["class", "count"]
                    fig2 = px.pie(class_counts, names="class", values="count",
                                  title="Liquidity Class Distribution",
                                  color_discrete_sequence=px.colors.diverging.RdYlGn)
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Liquidity Scores â€” Full Table")
                disp = [c for c in ["symbol", "liquidity_score", "liquidity_class"] if c in liq.columns]
                st.dataframe(liq[disp], use_container_width=True, hide_index=True)

            if not depth.empty:
                st.markdown("---")
                st.subheader("ðŸ“ Market Depth â€” Amihud Illiquidity")
                st.caption("Lower Amihud score = deeper market = lower price impact per trade.")
                shallowest = depth.nlargest(15, "amihud_illiquidity")
                deepest    = depth.nsmallest(15, "amihud_illiquidity")

                col_c, col_d = st.columns(2)
                with col_c:
                    fig3 = px.bar(deepest, x="symbol", y="amihud_illiquidity",
                                  title="15 Deepest Markets (lowest impact)",
                                  color_discrete_sequence=["#2ecc71"])
                    fig3.update_layout(height=300, xaxis_tickangle=-45)
                    st.plotly_chart(fig3, use_container_width=True)
                with col_d:
                    fig4 = px.bar(shallowest, x="symbol", y="amihud_illiquidity",
                                  title="15 Shallowest Markets (highest impact)",
                                  color_discrete_sequence=["#e74c3c"])
                    fig4.update_layout(height=300, xaxis_tickangle=-45)
                    st.plotly_chart(fig4, use_container_width=True)

                disp_d = [c for c in ["symbol", "amihud_illiquidity", "depth_class"] if c in depth.columns]
                st.dataframe(depth[disp_d], use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Liquidity analysis error: {exc}")

    # â”€â”€ Tab 2: Delivery Quality â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab2:
        try:
            dq = ana.analyze_delivery_quality()
            has_delivery = not dq.empty

            # â”€â”€ Delivery-based analysis (only when delivery_pct is populated) â”€â”€
            if has_delivery:
                st.success("Delivery % data available for this date.")
                signal_counts = dq["delivery_signal"].value_counts()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Institutional Accumulation",
                          int(signal_counts.get("Institutional Accumulation", 0)))
                c2.metric("Institutional Distribution",
                          int(signal_counts.get("Institutional Distribution", 0)))
                c3.metric("Speculative Rally",
                          int(signal_counts.get("Speculative Rally", 0)))
                c4.metric("Speculative Selloff",
                          int(signal_counts.get("Speculative Selloff", 0)))

                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.histogram(dq, x="delivery_pct", nbins=30,
                                       color="delivery_signal",
                                       title="Delivery % Distribution by Signal",
                                       barmode="overlay", opacity=0.7)
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    if "price_change_pct" in dq.columns:
                        fig2 = px.scatter(
                            dq.dropna(subset=["price_change_pct"]),
                            x="delivery_pct", y="price_change_pct",
                            color="delivery_signal", hover_data=["symbol"],
                            title="Delivery % vs Price Change",
                            labels={"delivery_pct": "Delivery %",
                                    "price_change_pct": "Price Change %"},
                        )
                        fig2.add_vline(x=60, line_dash="dash", line_color="gray",
                                       annotation_text="60% threshold")
                        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig2.update_layout(height=350)
                        st.plotly_chart(fig2, use_container_width=True)

                col_c, col_d = st.columns(2)
                with col_c:
                    st.subheader("Institutional Accumulation")
                    acc = dq[dq["delivery_signal"] == "Institutional Accumulation"]
                    if not acc.empty:
                        disp = [c for c in ["symbol","delivery_pct","price_change_pct",
                                            "close_price","net_traded_value"] if c in acc.columns]
                        st.dataframe(acc[disp].style.format(
                            {"delivery_pct":"{:.1f}%","price_change_pct":"{:+.2f}%",
                             "close_price":"{:.2f}","net_traded_value":"{:,.0f}"}),
                            use_container_width=True, hide_index=True)
                with col_d:
                    st.subheader("Distribution Warning")
                    dist = dq[dq["delivery_signal"] == "Institutional Distribution"]
                    if not dist.empty:
                        disp = [c for c in ["symbol","delivery_pct","price_change_pct",
                                            "close_price","net_traded_value"] if c in dist.columns]
                        st.dataframe(dist[disp].style.format(
                            {"delivery_pct":"{:.1f}%","price_change_pct":"{:+.2f}%",
                             "close_price":"{:.2f}","net_traded_value":"{:,.0f}"}),
                            use_container_width=True, hide_index=True)

            else:
                st.info("Delivery % not available for this date â€” showing trade-size proxy instead.\n\n"
                        "**Why?** `delivery_qty` is only populated when the full bhavcopy is loaded. "
                        "Enable it by running the ETL with the complete bhavcopy file.")

            # â”€â”€ Always shown: Average Trade Size proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.markdown("---")
            st.subheader("ðŸ¦ Institutional Activity Proxy â€” Average Trade Size")
            st.caption("High avg trade size (â‚¹/trade) suggests block/institutional orders. "
                       "Combine with price direction for signal interpretation.")

            if "net_traded_value" in eq.columns and "total_trades" in eq.columns:
                proxy = eq[["symbol","net_traded_value","total_trades",
                             "close_price","prev_close"]].copy()
                proxy = proxy[proxy["total_trades"] > 0]
                proxy["avg_trade_size"] = proxy["net_traded_value"] / proxy["total_trades"]
                proxy["price_change_pct"] = (
                    (proxy["close_price"] - proxy["prev_close"])
                    / proxy["prev_close"].replace(0, np.nan) * 100
                ).round(2)

                med_ts  = proxy["avg_trade_size"].median()
                ts_high = proxy["avg_trade_size"] > med_ts * 2

                def _proxy_signal(row):
                    big = row["avg_trade_size"] > med_ts * 2
                    pc  = row["price_change_pct"]
                    if big and pc > 0:   return "Large-Block Buy"
                    if big and pc < 0:   return "Large-Block Sell"
                    if not big and pc > 2: return "Retail-Driven Rally"
                    if not big and pc < -2: return "Retail-Driven Selloff"
                    return "Normal Activity"

                proxy["activity_signal"] = proxy.apply(_proxy_signal, axis=1)

                sig_counts = proxy["activity_signal"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Large-Block Buys",   int(sig_counts.get("Large-Block Buy",  0)))
                c2.metric("Large-Block Sells",  int(sig_counts.get("Large-Block Sell", 0)))
                c3.metric("Median Trade Size",  f"â‚¹{med_ts:,.0f}")

                col_e, col_f = st.columns(2)
                with col_e:
                    fig3 = px.scatter(
                        proxy.dropna(subset=["price_change_pct"]),
                        x="avg_trade_size", y="price_change_pct",
                        color="activity_signal", hover_data=["symbol"],
                        title="Avg Trade Size vs Price Change",
                        labels={"avg_trade_size": "Avg Trade Size (â‚¹)",
                                "price_change_pct": "Price Change %"},
                        log_x=True,
                    )
                    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig3.add_vline(x=med_ts * 2, line_dash="dash", line_color="gray",
                                   annotation_text="2Ã— median")
                    fig3.update_layout(height=370)
                    st.plotly_chart(fig3, use_container_width=True)

                with col_f:
                    sig_df = _vc_df(proxy["activity_signal"])
                    fig4 = px.bar(sig_df, x="activity_signal", y="count",
                                  color="activity_signal",
                                  title="Activity Signal Breakdown",
                                  labels={"activity_signal": "Signal", "count": "Stocks"})
                    fig4.update_layout(height=370, showlegend=False, xaxis_tickangle=-20)
                    st.plotly_chart(fig4, use_container_width=True)

                col_g, col_h = st.columns(2)
                with col_g:
                    st.subheader("Large-Block Buys")
                    lb = proxy[proxy["activity_signal"] == "Large-Block Buy"].nlargest(
                        20, "avg_trade_size")
                    disp = [c for c in ["symbol","avg_trade_size","price_change_pct",
                                        "close_price","net_traded_value"] if c in lb.columns]
                    st.dataframe(lb[disp].style.format(
                        {"avg_trade_size":"{:,.0f}","price_change_pct":"{:+.2f}%",
                         "close_price":"{:.2f}","net_traded_value":"{:,.0f}"}),
                        use_container_width=True, hide_index=True)
                with col_h:
                    st.subheader("Large-Block Sells")
                    ls = proxy[proxy["activity_signal"] == "Large-Block Sell"].nsmallest(
                        20, "price_change_pct")
                    disp = [c for c in ["symbol","avg_trade_size","price_change_pct",
                                        "close_price","net_traded_value"] if c in ls.columns]
                    st.dataframe(ls[disp].style.format(
                        {"avg_trade_size":"{:,.0f}","price_change_pct":"{:+.2f}%",
                         "close_price":"{:.2f}","net_traded_value":"{:,.0f}"}),
                        use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Delivery analysis error: {exc}")

    # â”€â”€ Tab 3: Candle Structure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab3:
        try:
            candles = ana.analyze_candle_structure()

            if candles.empty:
                st.warning("OHLC data not available.")
            else:
                type_counts = _vc_df(candles["candle_type"]).sort_values("count", ascending=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.bar(type_counts, x="count", y="candle_type",
                                 orientation="h", title="Candle Type Frequency",
                                 color="count", color_continuous_scale="Blues")
                    fig.update_layout(height=380, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    sort_col = "net_traded_value" if "net_traded_value" in candles.columns \
                               else "range_pct"
                    top100 = (candles.dropna(subset=["body_pct", "range_pct"])
                                     .nlargest(100, sort_col))
                    fig2 = px.scatter(
                        top100, x="range_pct", y="body_pct",
                        color="body_direction",
                        color_discrete_map={"Bullish": "#2ecc71", "Bearish": "#e74c3c"},
                        hover_data=["symbol", "candle_type"],
                        title="Body vs Range % â€” top 100 stocks by value",
                        labels={"range_pct": "Day Range %", "body_pct": "Body Size %"},
                    )
                    fig2.update_layout(height=380)
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.subheader("ðŸ”¨ Hammer / Dragonfly (Reversal Signals)")
                    h = candles[candles["candle_type"] == "Hammer / Dragonfly"]
                    if not h.empty:
                        disp = [c for c in ["symbol", "candle_type", "lower_wick_pct",
                                            "body_pct", "body_direction",
                                            "net_traded_value"] if c in h.columns]
                        st.dataframe(h[disp].sort_values(
                            "net_traded_value", ascending=False
                        ).head(20).style.format(
                            {"lower_wick_pct": "{:.2f}%", "body_pct": "{:.2f}%",
                             "net_traded_value": "{:,.0f}"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.info("No hammer/dragonfly patterns today.")

                with col_d:
                    st.subheader("â­ Shooting Star / Gravestone (Reversal Signals)")
                    s = candles[candles["candle_type"] == "Shooting Star / Gravestone"]
                    if not s.empty:
                        disp = [c for c in ["symbol", "candle_type", "upper_wick_pct",
                                            "body_pct", "body_direction",
                                            "net_traded_value"] if c in s.columns]
                        st.dataframe(s[disp].sort_values(
                            "net_traded_value", ascending=False
                        ).head(20).style.format(
                            {"upper_wick_pct": "{:.2f}%", "body_pct": "{:.2f}%",
                             "net_traded_value": "{:,.0f}"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.info("No shooting star/gravestone patterns today.")

                st.markdown("---")
                st.subheader("Full Candle Table")
                disp_all = [c for c in ["symbol", "candle_type", "body_direction",
                                        "range_pct", "body_pct", "upper_wick_pct",
                                        "lower_wick_pct"] if c in candles.columns]
                st.dataframe(candles[disp_all].style.format(
                    {c: "{:.2f}%" for c in ["range_pct","body_pct",
                                             "upper_wick_pct","lower_wick_pct"]}),
                    use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Candle structure error: {exc}")

    # â”€â”€ Tab 4: Price Position & Conviction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab4:
        try:
            pp = ana.calculate_price_position()

            if pp.empty:
                st.warning("Price data not available.")
            else:
                conv_counts = pp["conviction"].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Strong Bullish",  int(conv_counts.get("Strong Bullish", 0)))
                c2.metric("Weak Bullish",    int(conv_counts.get("Weak Bullish",   0)))
                c3.metric("Strong Bearish",  int(conv_counts.get("Strong Bearish", 0)))
                c4.metric("Weak Bearish",    int(conv_counts.get("Weak Bearish",   0)))

                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.histogram(pp, x="price_position", nbins=20,
                                       color="conviction",
                                       title="Price Position Distribution (0=day low, 100=day high)",
                                       labels={"price_position": "Price Position %"})
                    fig.add_vline(x=70, line_dash="dash", line_color="green",
                                  annotation_text="Bullish zone")
                    fig.add_vline(x=30, line_dash="dash", line_color="red",
                                  annotation_text="Bearish zone")
                    fig.update_layout(height=370)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    if "price_change_pct" in pp.columns:
                        plot_pp = pp.dropna(subset=["price_change_pct"]).copy()
                        size_kwargs = {}
                        if "net_traded_qty" in plot_pp.columns:
                            plot_pp["_sz"] = _safe_size(plot_pp["net_traded_qty"], 1e5)
                            size_kwargs = {"size": "_sz", "size_max": 20}
                        fig2 = px.scatter(
                            plot_pp, x="price_position", y="price_change_pct",
                            color="conviction", hover_data=["symbol"],
                            title="Price Position vs Price Change (bubble = volume)",
                            labels={"price_position": "Position in Range %",
                                    "price_change_pct": "Price Change %"},
                            **size_kwargs,
                        )
                        fig2.add_vline(x=50, line_dash="dot", line_color="gray")
                        fig2.add_hline(y=0,  line_dash="dot", line_color="gray")
                        fig2.update_layout(height=370)
                        st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.subheader("ðŸ’ª Strong Bullish Closes")
                    sb = pp[pp["conviction"] == "Strong Bullish"].sort_values(
                        "price_position", ascending=False)
                    disp = [c for c in ["symbol", "price_position", "price_change_pct",
                                        "close_price"] if c in sb.columns]
                    st.dataframe(sb[disp].head(20).style.format(
                        {"price_position": "{:.1f}%", "price_change_pct": "{:+.2f}%",
                         "close_price": "{:.2f}"}),
                        use_container_width=True, hide_index=True)

                with col_d:
                    st.subheader("ðŸ”» Strong Bearish Closes")
                    sb2 = pp[pp["conviction"] == "Strong Bearish"].sort_values("price_position")
                    disp2 = [c for c in ["symbol", "price_position", "price_change_pct",
                                         "close_price"] if c in sb2.columns]
                    st.dataframe(sb2[disp2].head(20).style.format(
                        {"price_position": "{:.1f}%", "price_change_pct": "{:+.2f}%",
                         "close_price": "{:.2f}"}),
                        use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Price position error: {exc}")

    # â”€â”€ Tab 5: Volume Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab5:
        try:
            vol = ana.detect_volume_clusters()
            pvr = ana.analyze_price_volume_relationship()

            if not vol.empty:
                threshold = vol["volume_ratio"].quantile(0.9)
                spikes    = vol[vol["volume_ratio"] > threshold]
                c1, c2 = st.columns(2)
                c1.metric("High Volume Stocks (top 10%)", len(spikes))
                c2.metric("Median Volume Ratio",
                          f"{vol['volume_ratio'].median():.2f}Ã—")

                fig = px.bar(
                    vol.head(30), x="symbol", y="volume_ratio",
                    color="volume_ratio", color_continuous_scale="OrRd",
                    title="Volume Ratio vs Median â€” Top 30 Stocks",
                    labels={"volume_ratio": "Vol Ratio"},
                )
                fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                              annotation_text="90th pct threshold")
                fig.update_layout(height=350, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            if not pvr.empty and "price_change_pct" in pvr.columns:
                st.markdown("---")
                st.subheader("Price Change by Volume Quartile")
                fig2 = px.box(
                    pvr, x="volume_category", y="price_change_pct",
                    color="volume_category",
                    category_orders={"volume_category":
                                     ["Low", "Below Avg", "Above Avg", "High"]},
                    title="Price Change Distribution across Volume Quartiles",
                    labels={"price_change_pct": "Price Change %",
                            "volume_category": "Volume Quartile"},
                )
                fig2.add_hline(y=0, line_dash="dot", line_color="gray")
                fig2.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                avg_by_vol = (pvr.groupby("volume_category")["price_change_pct"]
                                 .agg(["mean", "median", "std", "count"])
                                 .reset_index()
                                 .rename(columns={"mean": "Avg Chg %",
                                                  "median": "Median Chg %",
                                                  "std": "Std Dev",
                                                  "count": "Stocks"}))
                st.dataframe(avg_by_vol.style.format(
                    {"Avg Chg %": "{:+.2f}%", "Median Chg %": "{:+.2f}%",
                     "Std Dev": "{:.2f}"}),
                    use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Volume analysis error: {exc}")

    # â”€â”€ Tab 6: Momentum-Volume Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab6:
        try:
            mv = ana.classify_momentum_volume_quadrant()

            if mv.empty:
                st.warning("Price/volume data not available.")
            else:
                q_counts = mv["quadrant"].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Q1 Confirmed Momentum",
                          int(q_counts.get("Q1: Confirmed Momentum", 0)))
                c2.metric("Q2 Suspect Rally",
                          int(q_counts.get("Q2: Suspect Rally", 0)))
                c3.metric("Q3 Distribution",
                          int(q_counts.get("Q3: Distribution", 0)))
                c4.metric("Q4 Low-Conv. Pullback",
                          int(q_counts.get("Q4: Low-Conviction Pullback", 0)))

                st.markdown("""
                **Reading the matrix:**
                - **Q1 (â†‘ price, â†‘ volume)** â€” Trend-follow candidates; institutional demand confirmed by volume
                - **Q2 (â†‘ price, â†“ volume)** â€” Be cautious; rally lacks participation, prone to reversal
                - **Q3 (â†“ price, â†‘ volume)** â€” Institutional selling / distribution; avoid longs
                - **Q4 (â†“ price, â†“ volume)** â€” Weak pullback; potential base-building if delivery is high
                """)

                QUAD_COLORS = {
                    "Q1: Confirmed Momentum":      "#2ecc71",
                    "Q2: Suspect Rally":           "#f39c12",
                    "Q3: Distribution":            "#e74c3c",
                    "Q4: Low-Conviction Pullback": "#95a5a6",
                }
                col_a, col_b = st.columns(2)
                with col_a:
                    plot_mv = mv.dropna(subset=["price_change_pct"]).copy()
                    size_col = "net_traded_value" if "net_traded_value" in plot_mv.columns \
                               else "net_traded_qty"
                    plot_mv["_sz"] = _safe_size(plot_mv[size_col], 1e7)
                    fig = px.scatter(
                        plot_mv, x="volume_vs_median", y="price_change_pct",
                        color="quadrant", size="_sz", size_max=25,
                        hover_data=["symbol"],
                        title="Momentum-Volume Scatter (bubble = traded value)",
                        labels={"volume_vs_median": "Volume vs Median (Ã—)",
                                "price_change_pct": "Price Change %"},
                        color_discrete_map=QUAD_COLORS,
                    )
                    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                                  annotation_text="median vol")
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(height=420)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    # Use _vc_df to avoid pandas version column-name differences
                    qc_df = _vc_df(mv["quadrant"])
                    fig2 = px.bar(
                        qc_df, x="quadrant", y="count",
                        color="quadrant", title="Stock Count by Quadrant",
                        labels={"quadrant": "Quadrant", "count": "Stocks"},
                        color_discrete_map=QUAD_COLORS,
                    )
                    fig2.update_layout(height=420, showlegend=False,
                                       xaxis_tickangle=-20)
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")
                st.subheader("Q1 â€” Best Momentum Candidates")
                q1 = mv[mv["quadrant"] == "Q1: Confirmed Momentum"].sort_values(
                    "price_change_pct", ascending=False)
                disp = [c for c in ["symbol", "price_change_pct", "volume_vs_median",
                                    "close_price", "net_traded_value"] if c in q1.columns]
                st.dataframe(q1[disp].head(25).style.format(
                    {"price_change_pct": "{:+.2f}%", "volume_vs_median": "{:.1f}Ã—",
                     "close_price": "{:.2f}", "net_traded_value": "{:,.0f}"}),
                    use_container_width=True, hide_index=True)

                st.subheader("Q3 â€” Distribution Watch (avoid longs)")
                q3 = mv[mv["quadrant"] == "Q3: Distribution"].sort_values(
                    "price_change_pct")
                disp3 = [c for c in ["symbol", "price_change_pct", "volume_vs_median",
                                     "close_price", "net_traded_value"] if c in q3.columns]
                st.dataframe(q3[disp3].head(25).style.format(
                    {"price_change_pct": "{:+.2f}%", "volume_vs_median": "{:.1f}Ã—",
                     "close_price": "{:.2f}", "net_traded_value": "{:,.0f}"}),
                    use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Momentum-volume matrix error: {exc}")

    # â”€â”€ Tab 7: Overnight Gaps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab7:
        try:
            gaps = ana.calculate_overnight_gap()
            disc = ana.analyze_intraday_price_discovery()

            if not gaps.empty:
                gap_up   = gaps[gaps.get("gap_direction", pd.Series()) == "GAP_UP"]   \
                           if "gap_direction" in gaps.columns else pd.DataFrame()
                gap_down = gaps[gaps.get("gap_direction", pd.Series()) == "GAP_DOWN"] \
                           if "gap_direction" in gaps.columns else pd.DataFrame()

                c1, c2, c3 = st.columns(3)
                c1.metric("Gap Up Stocks",   len(gap_up))
                c2.metric("Gap Down Stocks", len(gap_down))
                c3.metric("Avg Gap Size",
                          f"{gaps['gap_pct'].abs().mean():.2f}%" if "gap_pct" in gaps.columns else "â€”")

                fig = px.bar(
                    gaps.sort_values("gap_pct", ascending=False),
                    x="symbol", y="gap_pct",
                    color="gap_pct",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    title="Overnight Gaps > 1% (sorted largest to smallest)",
                    labels={"gap_pct": "Gap %"},
                )
                fig.add_hline(y=0, line_color="gray", line_dash="dot")
                fig.update_layout(height=380, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("â¬† Top Gap-Ups")
                    disp = [c for c in ["symbol", "gap_pct", "open_price", "prev_close"]
                            if c in gaps.columns]
                    st.dataframe(gap_up.head(20)[disp].style.format(
                        {"gap_pct": "{:+.2f}%", "open_price": "{:.2f}",
                         "prev_close": "{:.2f}"}),
                        use_container_width=True, hide_index=True)
                with col_b:
                    st.subheader("â¬‡ Top Gap-Downs")
                    disp2 = [c for c in ["symbol", "gap_pct", "open_price", "prev_close"]
                             if c in gaps.columns]
                    st.dataframe(gap_down.head(20)[disp2].style.format(
                        {"gap_pct": "{:+.2f}%", "open_price": "{:.2f}",
                         "prev_close": "{:.2f}"}),
                        use_container_width=True, hide_index=True)
            else:
                st.info("No significant overnight gaps (>1%) today.")

            if not disc.empty:
                st.markdown("---")
                st.subheader("ðŸ” Intraday Price Discovery (Open â†’ Close)")
                st.caption("Positive = market absorbed the gap / continued in open direction. "
                           "Negative = gap fade / mean reversion.")
                dir_counts = disc["price_discovery_direction"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive Discovery", int(dir_counts.get("Positive", 0)))
                c2.metric("Negative Discovery", int(dir_counts.get("Negative", 0)))
                c3.metric("Flat",               int(dir_counts.get("Flat", 0)))

                fig2 = px.histogram(
                    disc, x="open_to_close_pct", nbins=40,
                    color="price_discovery_direction",
                    color_discrete_map={"Positive": "#2ecc71",
                                        "Negative": "#e74c3c",
                                        "Flat":     "#95a5a6"},
                    title="Open-to-Close % Distribution",
                    labels={"open_to_close_pct": "Openâ†’Close %"},
                )
                fig2.add_vline(x=0, line_dash="dot", line_color="gray")
                fig2.update_layout(height=320)
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as exc:
            st.error(f"Overnight gaps error: {exc}")

    # â”€â”€ Tab 8: Volatility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab8:
        try:
            vdf = ana.analyze_volatility()

            if vdf.empty:
                st.warning("Price data not available for volatility analysis.")
            else:
                avg_rng  = vdf["intraday_range_pct"].mean()
                avg_tr   = vdf["true_range_pct"].mean() if "true_range_pct" in vdf.columns else 0
                high_vol = int((vdf["volatility_class"] == "High Vol").sum()) if "volatility_class" in vdf.columns else 0
                low_vol  = int((vdf["volatility_class"] == "Low Vol").sum())  if "volatility_class" in vdf.columns else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Intraday Range",  f"{avg_rng:.2f}%")
                c2.metric("Avg True Range",      f"{avg_tr:.2f}%")
                c3.metric("High-Vol Stocks",     high_vol)
                c4.metric("Low-Vol Stocks",      low_vol)

                # â”€â”€ Distribution + Risk-Return â”€â”€
                col_a, col_b = st.columns(2)
                with col_a:
                    vc_col = "volatility_class" if "volatility_class" in vdf.columns else None
                    fig = px.histogram(
                        vdf, x="intraday_range_pct", nbins=40,
                        color=vc_col,
                        color_discrete_map={"Low Vol":"#2ecc71","Mid Vol":"#f39c12","High Vol":"#e74c3c"},
                        title="Intraday Range Distribution",
                        labels={"intraday_range_pct": "Range %"},
                    )
                    fig.add_vline(x=avg_rng, line_dash="dash", line_color="navy",
                                  annotation_text=f"Avg {avg_rng:.2f}%")
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    if "price_change_pct" in vdf.columns:
                        plot_v = vdf.dropna(subset=["price_change_pct"]).copy()
                        if "net_traded_value" in plot_v.columns:
                            plot_v["_sz"] = _safe_size(plot_v["net_traded_value"], 1e8)
                            size_kw = {"size": "_sz", "size_max": 20}
                        else:
                            size_kw = {}
                        fig2 = px.scatter(
                            plot_v, x="intraday_range_pct", y="price_change_pct",
                            color=vc_col, hover_data=["symbol"],
                            title="Risk-Return Quadrant",
                            labels={"intraday_range_pct": "Range %",
                                    "price_change_pct": "Price Chg %"},
                            color_discrete_map={"Low Vol":"#2ecc71","Mid Vol":"#f39c12","High Vol":"#e74c3c"},
                            **size_kw,
                        )
                        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
                        fig2.add_vline(x=avg_rng, line_dash="dot", line_color="gray",
                                       annotation_text="avg range")
                        # Quadrant labels
                        fig2.add_annotation(text="High Vol Gainers", x=vdf["intraday_range_pct"].quantile(0.9),
                                            y=vdf["price_change_pct"].quantile(0.9) if "price_change_pct" in vdf.columns else 3,
                                            showarrow=False, font=dict(size=9, color="gray"))
                        fig2.update_layout(height=360)
                        st.plotly_chart(fig2, use_container_width=True)

                # â”€â”€ True Range leaders â”€â”€
                st.markdown("---")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.subheader("ðŸ”¥ Most Volatile â€” Highest True Range")
                    disp = [c for c in ["symbol","intraday_range_pct","true_range_pct",
                                        "parkinson_vol","price_change_pct","close_price",
                                        "net_traded_value"] if c in vdf.columns]
                    top_v = vdf.nlargest(20, "true_range_pct" if "true_range_pct" in vdf.columns
                                         else "intraday_range_pct")
                    st.dataframe(top_v[disp].style.format(
                        {c: "{:.2f}%" for c in ["intraday_range_pct","true_range_pct",
                                                 "parkinson_vol","price_change_pct"] if c in disp}
                        | {"close_price": "{:.2f}", "net_traded_value": "{:,.0f}"}),
                        use_container_width=True, hide_index=True)

                with col_d:
                    st.subheader("ðŸ§Š Calmest â€” Low Range, Active Market")
                    calm = vdf.copy()
                    if "net_traded_value" in calm.columns:
                        # Keep only stocks above 30th percentile of value (filtering micro-caps)
                        val_thresh = calm["net_traded_value"].quantile(0.30)
                        calm = calm[calm["net_traded_value"] >= val_thresh]
                    bot_v = calm.nsmallest(20, "intraday_range_pct")
                    disp2 = [c for c in ["symbol","intraday_range_pct","price_change_pct",
                                         "close_price","net_traded_value"] if c in bot_v.columns]
                    st.dataframe(bot_v[disp2].style.format(
                        {"intraday_range_pct": "{:.2f}%", "price_change_pct": "{:+.2f}%",
                         "close_price": "{:.2f}", "net_traded_value": "{:,.0f}"}),
                        use_container_width=True, hide_index=True)

                # â”€â”€ Historical Volatility (multi-day) â”€â”€
                if not pr_range.empty:
                    st.markdown("---")
                    st.subheader("ðŸ“ˆ Historical Volatility â€” 30-Day Rolling")

                    eq_r = pr_range[~pr_range["is_index"].astype(bool)].copy() \
                           if "is_index" in pr_range.columns else pr_range.copy()

                    if eq_r["trade_date"].nunique() >= 5:
                        pivot = eq_r.pivot_table(
                            index="trade_date", columns="symbol", values="close_price"
                        )
                        hist_vol = (pivot.pct_change() * 100).std().reset_index()
                        hist_vol.columns = ["symbol", "hist_vol_30d"]

                        # Merge with today's range
                        today_rng = vdf[["symbol","intraday_range_pct"]].rename(
                            columns={"intraday_range_pct": "today_range"})
                        hist_vol = hist_vol.merge(today_rng, on="symbol", how="inner")
                        hist_vol["vol_ratio"] = (
                            hist_vol["today_range"]
                            / hist_vol["hist_vol_30d"].replace(0, np.nan)
                        ).round(2)

                        avg_hv = hist_vol["hist_vol_30d"].mean()
                        col_e, col_f = st.columns(2)
                        with col_e:
                            fig3 = px.histogram(
                                hist_vol, x="hist_vol_30d", nbins=30,
                                title=f"30-Day Historical Vol Distribution (avg {avg_hv:.2f}%/day)",
                                labels={"hist_vol_30d": "Hist Vol (daily %)"},
                                color_discrete_sequence=["#3498db"],
                            )
                            fig3.add_vline(x=avg_hv, line_dash="dash",
                                           annotation_text=f"Avg {avg_hv:.2f}%")
                            fig3.update_layout(height=320)
                            st.plotly_chart(fig3, use_container_width=True)

                        with col_f:
                            max_v = max(hist_vol["hist_vol_30d"].max(),
                                        hist_vol["today_range"].max())
                            fig4 = px.scatter(
                                hist_vol, x="hist_vol_30d", y="today_range",
                                hover_data=["symbol"],
                                title="Today's Range vs 30D Historical Vol",
                                labels={"hist_vol_30d": "30D Hist Vol %",
                                        "today_range": "Today Range %"},
                            )
                            fig4.add_shape(type="line", x0=0, y0=0,
                                           x1=max_v, y1=max_v,
                                           line=dict(dash="dash", color="gray"))
                            fig4.add_annotation(text="Above line = unusually high today",
                                                x=max_v * 0.6, y=max_v * 0.9,
                                                showarrow=False, font=dict(size=9, color="gray"))
                            fig4.update_layout(height=320)
                            st.plotly_chart(fig4, use_container_width=True)

                        st.subheader("Unusual Volatility â€” Today vs 30-Day History")
                        st.caption("vol_ratio > 1 â†’ stock is more volatile than usual today; "
                                   "vol_ratio < 0.5 â†’ unusually calm.")
                        unusual = hist_vol.nlargest(25, "vol_ratio")
                        st.dataframe(unusual.style.format(
                            {"hist_vol_30d": "{:.2f}%", "today_range": "{:.2f}%",
                             "vol_ratio": "{:.2f}Ã—"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.info("Need at least 5 days of price history for historical vol. "
                                "Run the ETL for more dates.")

        except Exception as exc:
            st.error(f"Volatility analysis error: {exc}")


# â”€â”€ page: smart money â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def page_smart_money(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ’° Smart Money & ETF Tracking</div>',
                unsafe_allow_html=True)

    pr_df  = load_prices(sel_date)     if db_ok else _sample_prices()
    etf_df = load_etf(sel_date)        if db_ok else pd.DataFrame()
    tt_df  = load_top_traded(sel_date) if db_ok else pd.DataFrame()

    if pr_df.empty:
        st.warning("No price data available.")
        return

    eq = pr_df[pr_df.get("is_index", pd.Series(False)) == False].copy()

    try:
        tracker = SmartMoneyTracker(eq, tt_df, etf_df)

        tab1, tab2, tab3 = st.tabs(["ðŸ› Institutional Buying", "ðŸ“Š Money Flow", "ðŸ”„ ETF Analysis"])

        with tab1:
            inst = tracker.identify_institutional_buying()
            st.metric("Institutional Buying Candidates", len(inst))
            if not inst.empty:
                st.dataframe(inst.head(20), use_container_width=True, hide_index=True)
            else:
                st.info("No strong institutional signals today.")

        with tab2:
            mfi = tracker.calculate_money_flow_index(window=5)
            if not mfi.empty:
                overbought = mfi[mfi["mfi"] > 80]
                oversold   = mfi[mfi["mfi"] < 20]
                c1,c2 = st.columns(2)
                c1.metric("Overbought (MFI > 80)", len(overbought))
                c2.metric("Oversold  (MFI < 20)", len(oversold))
                st.dataframe(mfi.head(20), use_container_width=True, hide_index=True)

        with tab3:
            if not etf_df.empty:
                st.subheader("ETF Prices Today")
                disp = [c for c in ["symbol","security_name","close_price",
                                     "net_traded_value","underlying"] if c in etf_df.columns]
                st.dataframe(etf_df[disp], use_container_width=True, hide_index=True)
            else:
                st.info("No ETF data for this date.")

    except Exception as exc:
        st.error(f"Smart money analysis error: {exc}")




# â”€â”€ cached computation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(ttl=300)
def _compute_regime_data(_pr_range: pd.DataFrame) -> dict:
    """Compute market regime indicators. Builds synthetic Nifty proxy since is_index=False."""
    eq_data = _pr_range[_pr_range["is_index"].astype(bool) == False].copy()
    if eq_data.empty or len(eq_data["trade_date"].unique()) < 5:
        return {"error": "Need at least 5 trading dates"}

    top_syms = (eq_data.groupby("symbol")["net_traded_value"].sum()
                       .nlargest(100).index)
    top_eq = eq_data[eq_data["symbol"].isin(top_syms)].copy()

    def _wt_avg(g):
        w = g["net_traded_value"].fillna(1).clip(lower=1)
        return pd.Series({
            "close_price": float(np.average(g["close_price"].fillna(0), weights=w)),
            "high_price":  float(g["high_price"].max()),
            "low_price":   float(g["low_price"].min()),
            "open_price":  float(np.average(g["open_price"].fillna(0), weights=w)),
        })

    nifty_proxy = top_eq.groupby("trade_date").apply(_wt_avg).reset_index()
    nifty_proxy["security_name"] = "Nifty 50"

    try:
        clf = MarketRegimeClassifier(eq_data, nifty_proxy)
        rh  = clf.classify_regime_rule_based()
        return {
            "regime_history": rh,
            "trend":          clf.calculate_trend_indicators(),
            "volatility":     clf.calculate_volatility_indicators(),
            "momentum":       clf.calculate_momentum_indicators(),
            "breadth":        clf.calculate_breadth_indicators(),
            "regime_changes": clf.detect_regime_changes(),
        }
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=300)
def _compute_volume_patterns(_pr_range: pd.DataFrame) -> dict:
    """Compute VolumePatternDetector signals across all dates."""
    eq_r = _pr_range[_pr_range["is_index"].astype(bool) == False].copy()
    if eq_r.empty:
        return {"breakouts": pd.DataFrame(), "dryups": pd.DataFrame(), "climax": pd.DataFrame()}
    try:
        det = VolumePatternDetector(eq_r)
        return {
            "breakouts": det.detect_volume_breakout(multiplier=2.0, lookback=20),
            "dryups":    det.detect_volume_dry_up(threshold=0.3, lookback=20),
            "climax":    det.analyze_climactic_volume(),
        }
    except Exception as exc:
        return {"breakouts": pd.DataFrame(), "dryups": pd.DataFrame(),
                "climax": pd.DataFrame(), "error": str(exc)}


# â”€â”€ page: market regime â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_TREND_COLORS = {
    "Strong Uptrend":     "#22c55e",
    "Moderate Uptrend":   "#86efac",
    "Sideways":           "#94a3b8",
    "Moderate Downtrend": "#fca5a5",
    "Strong Downtrend":   "#ef4444",
}


def page_market_regime(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ“ˆ Market Regime</div>', unsafe_allow_html=True)

    if not db_ok:
        st.warning("Database required for regime analysis.")
        return

    regime_history = load_precomp_regime()
    data = {}
    if regime_history.empty:
        # fallback to live compute if pre-computed table not yet populated
        st.info("Pre-computed regime data not yet available — computing live…")
        pr_range = load_prices_range()
        if pr_range.empty:
            st.warning("No price data available.")
            return
        with st.spinner("Computing regime indicators…"):
            data = _compute_regime_data(pr_range)
        if "error" in data:
            st.error(f"Regime computation error: {data['error']}")
            return
        regime_history = data["regime_history"]
        if regime_history.empty:
            st.info("Insufficient data for regime classification.")
            return
    else:
        # Build tab data from precomputed columns
        rh = regime_history.copy()
        rh["trade_date"] = pd.to_datetime(rh["trade_date"])
        data["trend"]    = rh[["trade_date", "close_price", "adx"]].dropna(subset=["close_price"])
        data["volatility"] = rh[["trade_date", "atr_pct"]].rename(columns={"atr_pct": "volatility"}).dropna(subset=["volatility"])
        data["momentum"] = rh[["trade_date", "rsi", "macd_hist"]].dropna(subset=["rsi"])
        data["breadth"]  = pd.DataFrame()
        regime_shifted   = rh["market_regime"].shift(1)
        changes          = rh[rh["market_regime"] != regime_shifted].copy()
        data["regime_changes"] = changes[["trade_date", "market_regime"]].rename(
            columns={"market_regime": "new_regime"}) if not changes.empty else pd.DataFrame()

    current      = regime_history.iloc[-1]
    trend_label  = str(current.get("trend", "Unknown"))
    vol_label    = str(current.get("volatility_regime", "Unknown"))
    regime_label = str(current.get("market_regime", "Unknown"))
    adx_val      = current.get("adx")

    # â”€â”€ metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Regime",    regime_label)
    c2.metric("Trend",             trend_label)
    c3.metric("Volatility",        vol_label)
    c4.metric("ADX",               f"{adx_val:.1f}" if pd.notna(adx_val) else "N/A")
    st.caption("â„¹ï¸ Index proxy built from value-weighted average of top-100 stocks (no Nifty 50 rows in DB).")
    st.markdown("---")

    # â”€â”€ timeline chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    period_lbl = f" ({view_period})" if view_period != "Daily" else ""
    st.subheader(f"Regime Timeline{period_lbl}")
    rh = regime_history.copy()
    rh["trade_date"] = pd.to_datetime(rh["trade_date"])
    if view_period != "Daily" and not rh.empty:
        freq = "W-FRI" if view_period == "Weekly" else "ME"
        rh = (rh.set_index("trade_date")
                .resample(freq)
                .agg({"close_price": "last", "trend": "last",
                      "market_regime": "last", "volatility_regime": "last"})
                .reset_index()
                .dropna(subset=["close_price"]))
    rh["color"] = rh["trend"].map(_TREND_COLORS).fillna("#94a3b8")
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=rh["trade_date"], y=rh["close_price"], mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(color=rh["color"].tolist(), size=8, line=dict(width=1, color="#333")),
        text=rh["trend"],
        hovertemplate="%{x}<br>Level: %{y:.1f}<br>Trend: %{text}<extra></extra>",
        name="Synthetic Nifty",
    ))
    fig_tl.update_layout(height=320, xaxis_title="Date", yaxis_title="Index Level",
                         plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                         font_color="#fff", margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_tl, use_container_width=True)

    # â”€â”€ strategy recommendation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    strategy = RegimeBasedStrategy.get_strategy_for_regime(regime_label)
    with st.expander("ðŸ“‹ Strategy Recommendation", expanded=True):
        s1, s2, s3 = st.columns(3)
        s1.metric("Action",        strategy.get("action", "â€”"))
        s2.metric("Position Size", strategy.get("position_size", "â€”"))
        s3.metric("Stop Loss",     strategy.get("stop_loss", "â€”"))
        st.caption(strategy.get("description", ""))
        instr = strategy.get("instruments", [])
        if instr:
            st.write("**Instruments:** " + ", ".join(instr))

    st.markdown("---")

    # â”€â”€ tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tab_tr, tab_vol, tab_mom, tab_br, tab_ch = st.tabs(
        ["ðŸ“ˆ Trend", "âš¡ Volatility", "ðŸš€ Momentum", "ðŸ“Š Breadth", "ðŸ”„ Regime Changes"]
    )

    with tab_tr:
        trend_df = data["trend"]
        if not trend_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_df["trade_date"], y=trend_df["close_price"],
                                     name="Price", line=dict(color="#1f77b4", width=2)))
            for period, color in [(20, "#f59e0b"), (50, "#a855f7")]:
                col = f"sma_{period}"
                if col in trend_df.columns:
                    fig.add_trace(go.Scatter(x=trend_df["trade_date"], y=trend_df[col],
                                             name=f"SMA {period}",
                                             line=dict(color=color, dash="dash")))
            fig.update_layout(height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="#fff", margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
            if "adx" in trend_df.columns:
                fig_adx = px.bar(trend_df.dropna(subset=["adx"]), x="trade_date", y="adx",
                                 title="ADX (>25 = strong trend)",
                                 color_discrete_sequence=["#22c55e"])
                fig_adx.add_hline(y=25, line_dash="dash", line_color="#f59e0b",
                                  annotation_text="Strong Trend (25)")
                fig_adx.update_layout(height=220, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                      font_color="#fff", margin=dict(l=40, r=20, t=30, b=40))
                st.plotly_chart(fig_adx, use_container_width=True)

    with tab_vol:
        vol_df = data["volatility"]
        if not vol_df.empty and "volatility" in vol_df.columns:
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=vol_df["trade_date"], y=vol_df["volatility"],
                                       name="HV 20-day (%)", fill="tozeroy",
                                       line=dict(color="#f59e0b")))
            if "atr_pct" in vol_df.columns:
                fig_v.add_trace(go.Scatter(x=vol_df["trade_date"], y=vol_df["atr_pct"],
                                           name="ATR %", line=dict(color="#a855f7", dash="dash")))
            fig_v.update_layout(height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font_color="#fff", margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_v, use_container_width=True)
            if "bb_width" in vol_df.columns:
                fig_bb = px.line(vol_df.dropna(subset=["bb_width"]), x="trade_date", y="bb_width",
                                 title="Bollinger Band Width (%)",
                                 color_discrete_sequence=["#4facfe"])
                fig_bb.update_layout(height=220, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                     font_color="#fff", margin=dict(l=40, r=20, t=30, b=40))
                st.plotly_chart(fig_bb, use_container_width=True)

    with tab_mom:
        mom_df = data["momentum"]
        if not mom_df.empty:
            fig_m = make_subplots(specs=[[{"secondary_y": True}]])
            if "rsi" in mom_df.columns:
                fig_m.add_trace(go.Scatter(x=mom_df["trade_date"], y=mom_df["rsi"],
                                           name="RSI", line=dict(color="#22c55e")),
                                secondary_y=False)
                fig_m.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                                annotation_text="OB 70")
                fig_m.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                                annotation_text="OS 30")
            if "macd_hist" in mom_df.columns:
                bar_colors = ["#22c55e" if v >= 0 else "#ef4444"
                              for v in mom_df["macd_hist"].fillna(0)]
                fig_m.add_trace(go.Bar(x=mom_df["trade_date"], y=mom_df["macd_hist"],
                                       name="MACD Hist", marker_color=bar_colors, opacity=0.7),
                                secondary_y=True)
            fig_m.update_layout(height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font_color="#fff", margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_m, use_container_width=True)

    with tab_br:
        breadth_df = data["breadth"]
        if not breadth_df.empty:
            lb = breadth_df.iloc[-1]
            c1b, c2b = st.columns(2)
            c1b.metric("Advances (latest)", int(lb.get("advances", 0)))
            c2b.metric("Declines (latest)", int(lb.get("declines", 0)))
            fig_br = go.Figure()
            fig_br.add_trace(go.Bar(x=breadth_df["trade_date"], y=breadth_df["advances"],
                                    name="Advances", marker_color="#22c55e", opacity=0.8))
            fig_br.add_trace(go.Bar(x=breadth_df["trade_date"], y=-breadth_df["declines"],
                                    name="Declines",  marker_color="#ef4444", opacity=0.8))
            if "ad_line" in breadth_df.columns:
                fig_br.add_trace(go.Scatter(x=breadth_df["trade_date"], y=breadth_df["ad_line"],
                                            name="A/D Line", line=dict(color="#f59e0b", width=2),
                                            yaxis="y2"))
            fig_br.update_layout(barmode="relative", height=300,
                                 plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                 font_color="#fff", margin=dict(l=40, r=20, t=20, b=40),
                                 yaxis2=dict(overlaying="y", side="right"))
            st.plotly_chart(fig_br, use_container_width=True)

    with tab_ch:
        rc = data["regime_changes"]
        if not rc.empty:
            st.metric("Regime Changes Detected", len(rc))
            st.dataframe(rc, use_container_width=True, hide_index=True)
        else:
            st.info("No regime changes detected in available data window.")


# â”€â”€ page: breakout analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def page_breakout_analysis(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸš€ Breakout Analysis</div>', unsafe_allow_html=True)

    if not db_ok:
        st.warning("Database required for breakout analysis.")
        return

    pr_range = load_prices_range()
    pr_df    = load_prices(sel_date)
    if pr_range.empty or pr_df.empty:
        st.warning("No price data available.")
        return

    eq_r = pr_range[pr_range["is_index"].astype(bool) == False].copy()
    dates_avail = sorted(eq_r["trade_date"].unique())
    latest_date = dates_avail[-1] if dates_avail else sel_date

    # â”€â”€ period high/low screener (52W columns are NULL in UDIFF) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    period_stats = eq_r.groupby("symbol").agg(
        period_high=("high_price", "max"),
        period_low=("low_price",  "min"),
    ).reset_index()

    latest_eq = eq_r[eq_r["trade_date"] == latest_date][
        ["symbol", "security_name", "close_price", "net_traded_value"]
    ].copy().merge(period_stats, on="symbol", how="inner")

    latest_eq["dist_from_period_high_pct"] = (
        (latest_eq["period_high"] - latest_eq["close_price"])
        / latest_eq["close_price"].replace(0, np.nan) * 100
    ).round(2)
    latest_eq["dist_from_period_low_pct"] = (
        (latest_eq["close_price"] - latest_eq["period_low"])
        / latest_eq["close_price"].replace(0, np.nan) * 100
    ).round(2)

    near_high = latest_eq[latest_eq["dist_from_period_high_pct"] <= 5].sort_values(
        "dist_from_period_high_pct")
    near_low  = latest_eq[latest_eq["dist_from_period_low_pct"]  <= 5].sort_values(
        "dist_from_period_low_pct")

    # â”€â”€ period momentum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _period_ret(n_days):
        if len(dates_avail) < n_days + 1:
            s = pd.Series(dtype=float, name=f"ret_{n_days}d")
            s.index.name = "symbol"
            return s
        past_date = dates_avail[-(n_days + 1)]
        cur  = eq_r[eq_r["trade_date"] == latest_date][["symbol", "close_price"]].set_index("symbol")
        past = eq_r[eq_r["trade_date"] == past_date][["symbol", "close_price"]].set_index("symbol")
        joined = cur.join(past, lsuffix="_cur", rsuffix="_past").dropna()
        return ((joined["close_price_cur"] - joined["close_price_past"]) /
                joined["close_price_past"].replace(0, np.nan) * 100).rename(f"ret_{n_days}d")

    r1, r5, r20 = _period_ret(1), _period_ret(5), _period_ret(20)
    mom = pd.concat([r1, r5, r20], axis=1).dropna(how="all").reset_index()
    col_labels = {c: c.replace("ret_", "").replace("d", "D Ret%")
                  for c in mom.columns if isinstance(c, str) and c != "symbol"}
    mom = mom.rename(columns=col_labels)

    # â”€â”€ volume breakouts (sel_date) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vol_bk = load_precomp_volume("breakout")
    if vol_bk.empty:
        vp_data = _compute_volume_patterns(pr_range)
        vol_bk  = vp_data.get("breakouts", pd.DataFrame())
    vol_today = vol_bk[vol_bk["trade_date"] == pd.Timestamp(sel_date)] if not vol_bk.empty else pd.DataFrame()

    # â”€â”€ 52W HL from DB (only if populated) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    hl_df = load_hl(sel_date)

    # â”€â”€ metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Near Period High (â‰¤5%)", len(near_high))
    c2.metric(f"Near Period Low  (â‰¤5%)", len(near_low))
    c3.metric("Vol Breakouts Today",     len(vol_today))
    c4.metric("52W HL Hits (DB)",        len(hl_df))
    st.caption(
        f"â„¹ï¸ 52-week H/L not in UDIFF data. Showing period high/low across "
        f"{len(dates_avail)} trading dates ({dates_avail[0]} â€“ {dates_avail[-1]})."
    )
    st.markdown("---")

    # â”€â”€ tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tab_labels = ["ðŸ“ˆ Near Period High", "ðŸ“‰ Near Period Low",
                  "ðŸƒ Period Momentum",  "ðŸ’¥ Volume Breakouts"]
    if not hl_df.empty:
        tab_labels.append("âš¡ 52W HL Hits")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if near_high.empty:
            st.info("No stocks within 5% of their period high.")
        else:
            disp = [c for c in ["symbol", "security_name", "close_price",
                                 "period_high", "dist_from_period_high_pct"]
                    if c in near_high.columns]
            st.dataframe(near_high[disp].head(50), use_container_width=True, hide_index=True)
            fig = px.bar(near_high.head(25), x="symbol", y="dist_from_period_high_pct",
                         title="Distance from Period High % (lower = closer to high)",
                         color="dist_from_period_high_pct",
                         color_continuous_scale="RdYlGn_r")
            fig.update_layout(height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="#fff", margin=dict(l=40, r=20, t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if near_low.empty:
            st.info("No stocks within 5% of their period low.")
        else:
            disp = [c for c in ["symbol", "security_name", "close_price",
                                 "period_low", "dist_from_period_low_pct"]
                    if c in near_low.columns]
            st.dataframe(near_low[disp].head(50), use_container_width=True, hide_index=True)
            fig = px.bar(near_low.head(25), x="symbol", y="dist_from_period_low_pct",
                         title="Distance from Period Low % (lower = near multi-period low)",
                         color="dist_from_period_low_pct",
                         color_continuous_scale="RdYlGn")
            fig.update_layout(height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="#fff", margin=dict(l=40, r=20, t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if mom.empty:
            st.info("Insufficient dates for momentum calculation.")
        else:
            ret_cols = [c for c in mom.columns if isinstance(c, str) and "Ret%" in c]
            sort_col = ret_cols[0] if ret_cols else (mom.columns[1] if len(mom.columns) > 1 else mom.columns[0])
            fmt = {c: "{:.2f}" for c in ret_cols}
            c_top, c_bot = st.columns(2)
            with c_top:
                st.subheader("Top 20 Momentum")
                st.dataframe(mom.nlargest(20, sort_col).style.format(fmt),
                             use_container_width=True, hide_index=True)
            with c_bot:
                st.subheader("Bottom 20 Momentum")
                st.dataframe(mom.nsmallest(20, sort_col).style.format(fmt),
                             use_container_width=True, hide_index=True)
            if len(ret_cols) >= 2:
                fig_sc = px.scatter(
                    mom.dropna(subset=ret_cols[:2]),
                    x=ret_cols[0], y=ret_cols[1],
                    hover_data=["symbol"] if "symbol" in mom.columns else None,
                    title=f"{ret_cols[0]} vs {ret_cols[1]}",
                    color=ret_cols[0], color_continuous_scale="RdYlGn",
                )
                fig_sc.update_layout(height=350, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                     font_color="#fff", margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_sc, use_container_width=True)

    with tabs[3]:
        if vol_today.empty:
            st.info(f"No volume breakouts (â‰¥2Ã— 20-day avg) on {sel_date}.")
        else:
            st.metric("Volume Breakouts", len(vol_today))
            fmt_vb = {c: "{:.2f}" for c in ["breakout_magnitude", "price_change"]
                      if c in vol_today.columns}
            st.dataframe(vol_today.sort_values("breakout_magnitude", ascending=False)
                                  .style.format(fmt_vb),
                         use_container_width=True, hide_index=True)
            fig_vb = px.scatter(
                vol_today.dropna(subset=["breakout_magnitude", "price_change"]),
                x="symbol", y="breakout_magnitude",
                color="price_change", color_continuous_scale="RdYlGn",
                size="breakout_magnitude", size_max=40,
                title="Volume Spike Magnitude (colour = price change %)",
            )
            fig_vb.update_layout(height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                 font_color="#fff", margin=dict(l=40, r=20, t=40, b=80))
            st.plotly_chart(fig_vb, use_container_width=True)

    if not hl_df.empty and len(tabs) > 4:
        with tabs[4]:
            highs = hl_df[hl_df.get("hl_type", pd.Series(dtype=str)) == "H"]
            lows  = hl_df[hl_df.get("hl_type", pd.Series(dtype=str)) == "L"]
            ch, cl = st.columns(2)
            ch.metric("52W Highs", len(highs))
            cl.metric("52W Lows",  len(lows))
            disp_hl = [c for c in ["symbol", "series", "security_name", "hl_type"]
                       if c in hl_df.columns]
            st.dataframe(hl_df[disp_hl], use_container_width=True, hide_index=True)


# â”€â”€ page: volume patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def page_volume_patterns(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ“Š Volume Patterns</div>', unsafe_allow_html=True)

    if not db_ok:
        st.warning("Database required for volume pattern analysis.")
        return

    vol_bk = load_precomp_volume("breakout")
    dryups = load_precomp_volume("dryup")
    climax = load_precomp_volume("climax")

    if vol_bk.empty and dryups.empty and climax.empty:
        # fallback to live compute if pre-computed table not yet populated
        st.info("Pre-computed volume data not yet available — computing live…")
        pr_range = load_prices_range()
        if pr_range.empty:
            st.warning("No price data available.")
            return
        with st.spinner("Analysing volume patterns across all dates…"):
            vp = _compute_volume_patterns(pr_range)
        if "error" in vp:
            st.error(f"Volume pattern error: {vp['error']}")
        vol_bk = vp.get("breakouts", pd.DataFrame())
        dryups = vp.get("dryups",    pd.DataFrame())
        climax = vp.get("climax",    pd.DataFrame())

    def _today(df):
        if df.empty or "trade_date" not in df.columns:
            return pd.DataFrame()
        return df[df["trade_date"] == sel_date]

    bk_today  = _today(vol_bk)
    dry_today = _today(dryups)
    clx_today = _today(climax)

    # â”€â”€ metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vol Breakouts Today",       len(bk_today))
    c2.metric("Total Breakouts (history)", len(vol_bk))
    c3.metric("Dry-Up Signals (streakâ‰¥3)", len(dry_today))
    c4.metric("Climactic Events Today",    len(clx_today))
    st.markdown("---")

    tab_bk, tab_dry, tab_clx, tab_heat = st.tabs(
        ["ðŸ’¥ Volume Breakouts", "ðŸ“‰ Volume Dry-Up", "ðŸŒŠ Climactic Volume", "ðŸ—º Historical Heatmap"]
    )

    with tab_bk:
        st.subheader(f"Volume Breakouts on {sel_date}  (â‰¥ 2Ã— 20-day average)")
        if bk_today.empty:
            st.info(f"No volume breakouts on {sel_date}.")
        else:
            fmt_bk = {c: "{:.2f}" for c in ["breakout_magnitude", "price_change"]
                      if c in bk_today.columns}
            st.dataframe(bk_today.sort_values("breakout_magnitude", ascending=False)
                                 .style.format(fmt_bk),
                         use_container_width=True, hide_index=True)
            fig = px.scatter(
                bk_today.dropna(subset=["breakout_magnitude", "price_change"]),
                x="symbol", y="breakout_magnitude",
                color="price_change", color_continuous_scale="RdYlGn",
                size="breakout_magnitude", size_max=40,
                title="Volume Spike Magnitude (colour = price change %)",
            )
            fig.update_layout(height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="#fff", margin=dict(l=40, r=20, t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("ðŸ“… All-dates volume breakouts (top 100)"):
            if not vol_bk.empty:
                fmt_all = {c: "{:.2f}" for c in ["breakout_magnitude", "price_change"]
                           if c in vol_bk.columns}
                st.dataframe(vol_bk.sort_values("breakout_magnitude", ascending=False)
                                   .head(100).style.format(fmt_all),
                             use_container_width=True, hide_index=True)

    with tab_dry:
        st.subheader("Volume Dry-Up Signals  (streak â‰¥ 3 days)")
        st.caption("Unusually low volume after a price move often precedes a directional breakout.")
        if dry_today.empty:
            st.info(f"No dry-up signals on {sel_date}.")
        else:
            st.dataframe(dry_today.sort_values("dryup_streak", ascending=False),
                         use_container_width=True, hide_index=True)
        if not dryups.empty:
            with st.expander("All dry-up signals (all dates, top 100)"):
                st.dataframe(dryups.sort_values("dryup_streak", ascending=False).head(100),
                             use_container_width=True, hide_index=True)

    with tab_clx:
        st.subheader("Climactic Volume Events")
        st.caption("Top 10% in both volume AND price range â€” often marks exhaustion / turning points.")
        if clx_today.empty:
            st.info(f"No climactic events on {sel_date}.")
        else:
            buying  = clx_today[clx_today.get("price_direction", pd.Series(dtype=str)) == "Buying Climax"]
            selling = clx_today[clx_today.get("price_direction", pd.Series(dtype=str)) == "Selling Climax"]
            cb, cs = st.columns(2)
            cb.metric("Buying Climax",  len(buying))
            cs.metric("Selling Climax", len(selling))
            disp_c = [c for c in ["symbol", "trade_date", "net_traded_qty",
                                   "price_range_pct", "price_direction"]
                      if c in clx_today.columns]
            st.dataframe(clx_today[disp_c], use_container_width=True, hide_index=True)
        if not climax.empty:
            with st.expander("All climactic events (all dates)"):
                disp_ca = [c for c in ["symbol", "trade_date", "price_direction", "price_range_pct"]
                           if c in climax.columns]
                st.dataframe(climax[disp_ca].head(200), use_container_width=True, hide_index=True)

    with tab_heat:
        period_heat_lbl = f" ({view_period})" if view_period != "Daily" else ""
        st.subheader(f"Volume Breakout Heatmap â€” Top 50 Symbols by Frequency{period_heat_lbl}")
        if vol_bk.empty:
            st.info("No volume breakout data available.")
        else:
            top50 = vol_bk["symbol"].value_counts().head(50).index.tolist()
            heat_data = vol_bk[vol_bk["symbol"].isin(top50)].copy()
            heat_data["trade_date"] = pd.to_datetime(heat_data["trade_date"])
            if view_period != "Daily" and not heat_data.empty:
                freq = "W-FRI" if view_period == "Weekly" else "ME"
                heat_data = (heat_data.groupby(
                                 ["symbol", pd.Grouper(key="trade_date", freq=freq)]
                             )["breakout_magnitude"]
                             .max()
                             .reset_index())
            if not heat_data.empty:
                pivot = heat_data.pivot_table(
                    index="symbol", columns="trade_date",
                    values="breakout_magnitude", aggfunc="max"
                ).fillna(0)
                fig_heat = px.imshow(
                    pivot,
                    color_continuous_scale="YlOrRd",
                    title=f"Volume Breakout Magnitude by Symbol Ã— Date{period_heat_lbl}",
                    labels=dict(color="Breakout Mag (Ã—avg)"),
                )
                fig_heat.update_layout(
                    height=max(400, len(top50) * 14),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="#fff", margin=dict(l=80, r=20, t=40, b=80),
                )
                st.plotly_chart(fig_heat, use_container_width=True)


# â”€â”€ page: causality analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(ttl=300)
def _compute_causality(_pr_range: pd.DataFrame) -> dict:
    """Cache wrapper for LeadLagAnalyzer (expensive on ~100k rows)."""
    try:
        analyzer = LeadLagAnalyzer(_pr_range)
        return {
            "analyzer": analyzer,
            "corr_matrix": analyzer.correlation_matrix(top_n=30),
            "clustered":   analyzer.clustered_correlation_matrix(top_n=30),
            "leaders":     analyzer.find_market_leaders(top_n=15),
            "lag_profile": analyzer.lag_profile(max_lag=5),
            "symbols":     analyzer.symbols,
            "n_dates":     len(analyzer.dates),
            "avg_corr":    float(analyzer.correlation_matrix(top_n=30).values[
                               np.triu_indices(min(30, len(analyzer.symbols)), k=1)
                           ].mean()) if len(analyzer.symbols) >= 2 else 0.0,
        }
    except Exception as exc:
        return {"error": str(exc)}


def page_causality_analysis(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ”— Causality Analysis</div>',
                unsafe_allow_html=True)

    if not db_ok:
        st.warning("Database required for causality analysis.")
        return

    leaders = load_precomp_causality()

    # scalar metadata stored in each row
    avg_corr = float(leaders["avg_corr"].iloc[0]) if not leaders.empty else 0.0
    n_dates  = int(leaders["n_dates"].iloc[0])    if not leaders.empty else 0
    symbols  = leaders["symbol"].tolist()          if not leaders.empty else []

    # For correlation matrix and explorer tabs we still need the live analyzer
    # (pre-compute only stores the leaders table, not the full matrix)
    corr_mat  = pd.DataFrame()
    clustered = pd.DataFrame()
    analyzer  = None
    lag_profile = pd.DataFrame()

    if leaders.empty:
        # fallback to live compute if pre-computed table not yet populated
        st.info("Pre-computed causality data not yet available — computing live…")
        pr_range = load_prices_range()
        if pr_range.empty:
            st.warning("No price data available.")
            return
        with st.spinner("Computing lead-lag relationships…"):
            data = _compute_causality(pr_range)
        if "error" in data:
            st.error(f"Causality computation error: {data['error']}")
            return
        symbols     = data["symbols"]
        n_dates     = data["n_dates"]
        avg_corr    = data["avg_corr"]
        leaders     = data["leaders"]
        lag_profile = data["lag_profile"]
        corr_mat    = data["corr_matrix"]
        clustered   = data["clustered"]
        analyzer    = data["analyzer"]
    else:
        # Load live analyzer for correlation matrix + explorer (fast — cached)
        pr_range = load_prices_range()
        if not pr_range.empty:
            try:
                with st.spinner("Loading correlation matrix…"):
                    live = _compute_causality(pr_range)
                corr_mat    = live.get("corr_matrix",  pd.DataFrame())
                clustered   = live.get("clustered",    pd.DataFrame())
                analyzer    = live.get("analyzer")
                lag_profile = live.get("lag_profile",  pd.DataFrame())
            except Exception:
                pass

    # â”€â”€ metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    c1, c2, c3 = st.columns(3)
    c1.metric("Stocks Analyzed",       len(symbols))
    c2.metric("Trading Dates",         n_dates)
    c3.metric("Avg Cross-Correlation", f"{avg_corr:.3f}")
    st.caption("Lead-lag analysis uses log-returns. A positive lag means the source stock "
               "moves *before* the market.")
    st.markdown("---")

    tab_corr, tab_leaders, tab_explorer, tab_cluster = st.tabs([
        "ðŸ“Š Correlation Matrix",
        "ðŸ† Market Leaders",
        "ðŸ” Lead-Lag Explorer",
        "ðŸŒ Cluster View",
    ])

    # â”€â”€ Tab 1: Correlation Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab_corr:
        st.subheader("Contemporaneous Return Correlation (Lag 0)")
        if corr_mat.empty:
            st.info("Not enough data to compute correlations.")
        else:
            fig = px.imshow(
                corr_mat,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Pearson Correlation Matrix â€” Top 30 Stocks",
                labels=dict(color="Correlation"),
            )
            fig.update_layout(
                height=600,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fff", margin=dict(l=100, r=20, t=40, b=100),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Values close to 1 = strong co-movement. Values near 0 = independent movement.")

    # â”€â”€ Tab 2: Market Leaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab_leaders:
        st.subheader("Stocks That Lead the Market (Lag-1 Correlation)")
        st.caption("A stock is a 'leader' if its return yesterday predicts the market return today.")
        if leaders.empty:
            st.info("Not enough data to identify leaders.")
        else:
            fig = px.bar(
                leaders,
                x="lag1_corr", y="symbol",
                orientation="h",
                color="lag1_corr",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="Lag-1 Correlation with Market Return (top 15)",
                labels={"lag1_corr": "Lag-1 Correlation", "symbol": "Stock"},
            )
            fig.update_layout(
                height=450, yaxis=dict(autorange="reversed"),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fff", margin=dict(l=80, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            disp = [c for c in ["symbol", "lag1_corr", "direction", "strength"]
                    if c in leaders.columns]
            st.dataframe(
                leaders[disp].style.format({"lag1_corr": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )

        if not lag_profile.empty:
            st.markdown("---")
            st.subheader("Market-Wide Lag Profile")
            st.caption("Average absolute correlation between each stock and the market "
                       "at each lag. Higher bars = stronger lead/lag structure.")
            fig2 = px.bar(
                lag_profile, x="lag", y="avg_abs_corr",
                color="avg_abs_corr", color_continuous_scale="Blues",
                title="Avg |Correlation| vs. Market at Each Lag",
                labels={"lag": "Lag (days)", "avg_abs_corr": "Avg |Corr|"},
                text="n_pairs",
            )
            fig2.update_layout(
                height=300,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fff", margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # â”€â”€ Tab 3: Lead-Lag Explorer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab_explorer:
        st.subheader("Pair Lead-Lag Explorer")
        if len(symbols) < 2:
            st.info("Need at least 2 symbols for pair analysis.")
        else:
            col_a, col_b = st.columns(2)
            sym_a = col_a.selectbox("Symbol A", symbols, index=0, key="ll_sym_a")
            default_b = symbols[1] if len(symbols) > 1 else symbols[0]
            sym_b = col_b.selectbox("Symbol B", symbols,
                                    index=symbols.index(default_b), key="ll_sym_b")

            if sym_a == sym_b:
                st.warning("Select two different symbols.")
            else:
                try:
                    ll_df = analyzer.lead_lag_correlation(sym_a, max_lag=5)
                    pair_df = ll_df[ll_df["symbol"] == sym_b].copy()
                    if pair_df.empty:
                        st.info(f"No lead-lag data for {sym_a} â†’ {sym_b}.")
                    else:
                        fig_ll = px.bar(
                            pair_df, x="lag", y="correlation",
                            color="correlation",
                            color_continuous_scale="RdYlGn",
                            color_continuous_midpoint=0,
                            title=f"Lead-Lag: {sym_a} vs {sym_b}",
                            labels={"lag": "Lag (days, +ve = A leads)",
                                    "correlation": "Correlation"},
                        )
                        fig_ll.update_layout(
                            height=300,
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="#fff", margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig_ll, use_container_width=True)

                    roll_window = st.slider("Rolling window (days)", 5, 20, 10,
                                            key="ll_window")
                    try:
                        roll_corr = analyzer.rolling_correlation(sym_a, sym_b,
                                                                  window=roll_window)
                        if not roll_corr.empty:
                            fig_roll = px.line(
                                x=roll_corr.index, y=roll_corr.values,
                                title=f"Rolling {roll_window}-day Correlation: {sym_a} Ã— {sym_b}",
                                labels={"x": "Date", "y": "Correlation"},
                            )
                            fig_roll.add_hline(y=0, line_dash="dot", line_color="grey")
                            fig_roll.update_layout(
                                height=280,
                                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font_color="#fff", margin=dict(l=40, r=20, t=40, b=40),
                            )
                            st.plotly_chart(fig_roll, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Rolling correlation error: {e}")
                except Exception as e:
                    st.error(f"Lead-lag computation error: {e}")

    # â”€â”€ Tab 4: Cluster View â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab_cluster:
        st.subheader("Hierarchical Cluster View")
        st.caption("Stocks are reordered by Ward linkage clustering on 1 âˆ’ |correlation|. "
                   "Clusters indicate groups that move together.")
        if clustered.empty:
            st.info("Not enough data for clustering.")
        else:
            fig_cl = px.imshow(
                clustered,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Clustered Correlation Matrix",
                labels=dict(color="Correlation"),
            )
            fig_cl.update_layout(
                height=620,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fff", margin=dict(l=100, r=20, t=40, b=100),
            )
            st.plotly_chart(fig_cl, use_container_width=True)


# â”€â”€ page: data explorer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_table(table: str, start: date, end: date,
                symbol: str | None, limit: int | None) -> pd.DataFrame:
    """Load a DB table with optional date-range and symbol filters."""
    conditions = ["trade_date BETWEEN :start AND :end"]
    params: dict = {"start": start, "end": end}
    if symbol:
        conditions.append("UPPER(symbol) LIKE :sym")
        params["sym"] = f"%{symbol.upper()}%"
    where = " AND ".join(conditions)
    lim = f" LIMIT {limit}" if limit else ""
    return _query(f"SELECT * FROM {table} WHERE {where} ORDER BY trade_date DESC{lim}",
                  params)


def _load_log(limit: int | None) -> pd.DataFrame:
    lim = f" LIMIT {limit}" if limit else ""
    return _query(f"SELECT * FROM log_ingestion ORDER BY id DESC{lim}")


_TABLE_INFO = {
    "fact_daily_prices":      ("ðŸ“ˆ Daily Prices",    True,  "Core price data â€” populated via UDIFF bhavcopy ETL"),
    "fact_etf_prices":        ("ðŸ”„ ETF Prices",      True,  "ETF OHLCV â€” populated from UDIFF (ETF series)"),
    "fact_top_traded":        ("ðŸ† Top Traded",       True,  "Top stocks by value â€” backfilled from fact_daily_prices"),
    "fact_circuit_hits":      ("âš¡ Circuit Hits",     False, "Empty â€” needs NSE live API session (circuit filter data)"),
    "fact_corporate_actions": ("ðŸ“‹ Corp Actions",     False, "Empty â€” needs NSE CF-CA-equities.csv or live API"),
    "fact_hl_hits":           ("ðŸŽ¯ 52W HL Hits",      False, "Empty â€” needs NSE live-analysis-variations API"),
    "fact_market_cap":        ("ðŸ’Ž Market Cap",       False, "Empty â€” needs NSE MCAP file (no working archive URL found)"),
}


def page_data_explorer(sel_date: date, db_ok: bool, view_period: str = "Daily"):
    st.markdown('<div class="main-header">ðŸ—„ï¸ Data Explorer</div>',
                unsafe_allow_html=True)

    if not db_ok:
        st.warning("Database required for data explorer.")
        return

    # â”€â”€ global filters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Filters")
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    default_start = date(2026, 2, 1)
    start_d = fc1.date_input("From", value=default_start, key="de_start")
    end_d   = fc2.date_input("To",   value=date.today(),  key="de_end")
    sym_filter = fc3.text_input("Symbol (optional)", placeholder="e.g. RELIANCE",
                                 key="de_sym").strip()
    limit_opt = fc4.selectbox("Row limit", [100, 500, 1000, "All"], index=0,
                               key="de_limit")
    limit_val = None if limit_opt == "All" else int(limit_opt)
    st.markdown("---")

    # â”€â”€ tabs â€” one per table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tab_names = [info[0] for info in _TABLE_INFO.values()] + ["ðŸ“œ Ingestion Log"]
    tabs = st.tabs(tab_names)

    for tab_obj, (table, (label, populated, note)) in zip(tabs, _TABLE_INFO.items()):
        with tab_obj:
            if not populated:
                st.info(f"**{label}** â€” {note}")
                # Still show schema row count
                try:
                    cnt = _query(f"SELECT COUNT(*) AS n FROM {table}")
                    st.metric("Rows in table", int(cnt["n"].iloc[0]))
                except Exception:
                    pass
                continue

            try:
                df = _load_table(table, start_d, end_d,
                                 sym_filter or None, limit_val)
            except Exception as exc:
                st.error(f"Query error: {exc}")
                continue

            r1, r2 = st.columns(2)
            r1.metric("Rows returned", len(df))
            r2.metric("Columns", len(df.columns))

            if df.empty:
                st.info("No rows match the current filters.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"â¬‡ Download {label} CSV",
                    data=csv_bytes,
                    file_name=f"{table}_{start_d}_{end_d}.csv",
                    mime="text/csv",
                    key=f"dl_{table}",
                )

    # â”€â”€ Ingestion Log tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[-1]:
        st.subheader("ðŸ“œ Ingestion Log")
        try:
            log_df = _load_log(limit_val)
        except Exception as exc:
            st.error(f"Query error: {exc}")
            log_df = pd.DataFrame()

        st.metric("Log entries", len(log_df))
        if log_df.empty:
            st.info("No ingestion log entries found.")
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            csv_bytes = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="â¬‡ Download Log CSV",
                data=csv_bytes,
                file_name=f"log_ingestion_{date.today()}.csv",
                mime="text/csv",
                key="dl_log",
            )


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    page, sel_date, db_ok, view_period = _sidebar()

    PAGE_MAP = {
        "ðŸ  Market Overview":   page_overview,
        "ðŸ“‹ Market Summary":    page_market_summary,
        "ðŸ”¬ Microstructure":    page_microstructure,
        "ðŸ’° Smart Money":       page_smart_money,
        "ðŸ“ˆ Market Regime":     page_market_regime,
        "ðŸš€ Breakout Analysis": page_breakout_analysis,
        "ðŸ“Š Volume Patterns":   page_volume_patterns,
        "ðŸ”— Causality Analysis": page_causality_analysis,
        "ðŸ—„ï¸ Data Explorer":     page_data_explorer,
    }

    fn = PAGE_MAP.get(page)
    if fn:
        fn(sel_date, db_ok, view_period)


if __name__ == "__main__":
    main()
