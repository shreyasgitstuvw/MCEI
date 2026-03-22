import sys, os, traceback, numpy as np
sys.path.insert(0, 'src')
from database.connection import get_engine
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analytics.market_microstructure import MarketMicrostructureAnalyzer
from sqlalchemy import text

def _safe_size(series, scale=1e6):
    s = series.fillna(scale).clip(lower=1)
    return (s / scale).clip(lower=0.05, upper=200)

def _vc_df(series):
    counts = series.value_counts()
    return pd.DataFrame({series.name: counts.index, "count": counts.values})

engine = get_engine()
with engine.connect() as conn:
    sel_date = pd.read_sql(text("SELECT MAX(trade_date) as d FROM fact_daily_prices"), conn).iloc[0]["d"]
    pr_df = pd.read_sql(text("SELECT * FROM fact_daily_prices WHERE trade_date=:d"), conn, params={"d": sel_date})
    pr_range = pd.read_sql(text("SELECT * FROM fact_daily_prices WHERE trade_date >= CURRENT_DATE - 30 ORDER BY trade_date"), conn)

print(f"sel_date={sel_date}, pr_df={pr_df.shape}, pr_range={pr_range.shape}")
is_idx = pr_df["is_index"] if "is_index" in pr_df.columns else pd.Series(False, index=pr_df.index)
eq = pr_df[~is_idx.astype(bool)].copy().reset_index(drop=True)
ana = MarketMicrostructureAnalyzer(eq)
errors = {}

# TAB 2
try:
    dq = ana.analyze_delivery_quality()
    has_delivery = not dq.empty
    if has_delivery:
        signal_counts = dq["delivery_signal"].value_counts()
        fig = px.histogram(dq, x="delivery_pct", nbins=30, color="delivery_signal", barmode="overlay", opacity=0.7)
        if "price_change_pct" in dq.columns:
            fig2 = px.scatter(dq.dropna(subset=["price_change_pct"]), x="delivery_pct", y="price_change_pct", color="delivery_signal", hover_data=["symbol"])
            fig2.add_vline(x=60, line_dash="dash", line_color="gray", annotation_text="60% threshold")
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    if "net_traded_value" in eq.columns and "total_trades" in eq.columns:
        proxy = eq[["symbol","net_traded_value","total_trades","close_price","prev_close"]].copy()
        proxy = proxy[proxy["total_trades"] > 0]
        proxy["avg_trade_size"] = proxy["net_traded_value"] / proxy["total_trades"]
        proxy["price_change_pct"] = ((proxy["close_price"] - proxy["prev_close"]) / proxy["prev_close"].replace(0, np.nan) * 100).round(2)
        med_ts = proxy["avg_trade_size"].median()
        def _proxy_signal(row):
            big = row["avg_trade_size"] > med_ts * 2
            pc = row["price_change_pct"]
            if big and pc > 0: return "Large-Block Buy"
            if big and pc < 0: return "Large-Block Sell"
            if not big and pc > 2: return "Retail-Driven Rally"
            if not big and pc < -2: return "Retail-Driven Selloff"
            return "Normal Activity"
        proxy["activity_signal"] = proxy.apply(_proxy_signal, axis=1)
        sig_df = _vc_df(proxy["activity_signal"])
        fig3 = px.scatter(proxy.dropna(subset=["price_change_pct"]), x="avg_trade_size", y="price_change_pct", color="activity_signal", hover_data=["symbol"], log_x=True)
        fig3.add_hline(y=0, line_dash="dot", line_color="gray")
        fig3.add_vline(x=med_ts * 2, line_dash="dash", line_color="gray", annotation_text="2x median")
        fig4 = px.bar(sig_df, x="activity_signal", y="count", color="activity_signal")
        lb = proxy[proxy["activity_signal"] == "Large-Block Buy"].nlargest(20, "avg_trade_size")
        disp = [c for c in ["symbol","avg_trade_size","price_change_pct","close_price","net_traded_value"] if c in lb.columns]
        _ = lb[disp].style.format({"avg_trade_size":"{:,.0f}","price_change_pct":"{:+.2f}%","close_price":"{:.2f}","net_traded_value":"{:,.0f}"}).to_html()
        ls = proxy[proxy["activity_signal"] == "Large-Block Sell"].nsmallest(20, "price_change_pct")
        _ = ls[disp].style.format({"avg_trade_size":"{:,.0f}","price_change_pct":"{:+.2f}%","close_price":"{:.2f}","net_traded_value":"{:,.0f}"}).to_html()
    print("Tab2: OK")
except Exception:
    errors["Tab2"] = traceback.format_exc()
    print(f"Tab2: ERROR")

# TAB 3
try:
    candles = ana.analyze_candle_structure()
    if not candles.empty:
        type_counts = _vc_df(candles["candle_type"]).sort_values("count", ascending=True)
        fig = px.bar(type_counts, x="count", y="candle_type", orientation="h", color="count", color_continuous_scale="Blues")
        sort_col = "net_traded_value" if "net_traded_value" in candles.columns else "range_pct"
        top100 = candles.dropna(subset=["body_pct","range_pct"]).nlargest(100, sort_col)
        fig2 = px.scatter(top100, x="range_pct", y="body_pct", color="body_direction", color_discrete_map={"Bullish":"#2ecc71","Bearish":"#e74c3c"}, hover_data=["symbol","candle_type"])
        h = candles[candles["candle_type"] == "Hammer / Dragonfly"]
        if not h.empty:
            disp = [c for c in ["symbol","candle_type","lower_wick_pct","body_pct","body_direction","net_traded_value"] if c in h.columns]
            _ = h[disp].sort_values("net_traded_value", ascending=False).head(20).style.format({"lower_wick_pct":"{:.2f}%","body_pct":"{:.2f}%","net_traded_value":"{:,.0f}"}).to_html()
        s = candles[candles["candle_type"] == "Shooting Star / Gravestone"]
        if not s.empty:
            disp = [c for c in ["symbol","candle_type","upper_wick_pct","body_pct","body_direction","net_traded_value"] if c in s.columns]
            _ = s[disp].sort_values("net_traded_value", ascending=False).head(20).style.format({"upper_wick_pct":"{:.2f}%","body_pct":"{:.2f}%","net_traded_value":"{:,.0f}"}).to_html()
        disp_all = [c for c in ["symbol","candle_type","body_direction","range_pct","body_pct","upper_wick_pct","lower_wick_pct"] if c in candles.columns]
        _ = candles[disp_all].style.format({c: "{:.2f}%" for c in ["range_pct","body_pct","upper_wick_pct","lower_wick_pct"]}).to_html()
    print("Tab3: OK")
except Exception:
    errors["Tab3"] = traceback.format_exc()
    print("Tab3: ERROR")

# TAB 4
try:
    pp = ana.calculate_price_position()
    if not pp.empty:
        fig = px.histogram(pp, x="price_position", nbins=20, color="conviction")
        fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Bullish zone")
        fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Bearish zone")
        if "price_change_pct" in pp.columns:
            plot_pp = pp.dropna(subset=["price_change_pct"]).copy()
            size_kwargs = {}
            if "net_traded_qty" in plot_pp.columns:
                plot_pp["_sz"] = _safe_size(plot_pp["net_traded_qty"], 1e5)
                size_kwargs = {"size": "_sz", "size_max": 20}
            fig2 = px.scatter(plot_pp, x="price_position", y="price_change_pct", color="conviction", hover_data=["symbol"], **size_kwargs)
            fig2.add_vline(x=50, line_dash="dot", line_color="gray")
            fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        sb = pp[pp["conviction"] == "Strong Bullish"].sort_values("price_position", ascending=False)
        disp = [c for c in ["symbol","price_position","price_change_pct","close_price"] if c in sb.columns]
        _ = sb[disp].head(20).style.format({"price_position":"{:.1f}%","price_change_pct":"{:+.2f}%","close_price":"{:.2f}"}).to_html()
        sb2 = pp[pp["conviction"] == "Strong Bearish"].sort_values("price_position")
        _ = sb2[disp].head(20).style.format({"price_position":"{:.1f}%","price_change_pct":"{:+.2f}%","close_price":"{:.2f}"}).to_html()
    print("Tab4: OK")
except Exception:
    errors["Tab4"] = traceback.format_exc()
    print("Tab4: ERROR")

# TAB 6
try:
    mv = ana.classify_momentum_volume_quadrant()
    if not mv.empty:
        QUAD_COLORS = {"Q1: Confirmed Momentum":"#2ecc71","Q2: Suspect Rally":"#f39c12","Q3: Distribution":"#e74c3c","Q4: Low-Conviction Pullback":"#95a5a6"}
        plot_mv = mv.dropna(subset=["price_change_pct"]).copy()
        size_col = "net_traded_value" if "net_traded_value" in plot_mv.columns else "net_traded_qty"
        plot_mv["_sz"] = _safe_size(plot_mv[size_col], 1e7)
        fig = px.scatter(plot_mv, x="volume_vs_median", y="price_change_pct", color="quadrant", size="_sz", size_max=25, hover_data=["symbol"], color_discrete_map=QUAD_COLORS)
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="median vol")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        qc_df = _vc_df(mv["quadrant"])
        fig2 = px.bar(qc_df, x="quadrant", y="count", color="quadrant", color_discrete_map=QUAD_COLORS)
        q1 = mv[mv["quadrant"] == "Q1: Confirmed Momentum"].sort_values("price_change_pct", ascending=False)
        disp = [c for c in ["symbol","price_change_pct","volume_vs_median","close_price","net_traded_value"] if c in q1.columns]
        _ = q1[disp].head(25).style.format({"price_change_pct":"{:+.2f}%","volume_vs_median":"{:.1f}x","close_price":"{:.2f}","net_traded_value":"{:,.0f}"}).to_html()
        q3 = mv[mv["quadrant"] == "Q3: Distribution"].sort_values("price_change_pct")
        disp3 = [c for c in ["symbol","price_change_pct","volume_vs_median","close_price","net_traded_value"] if c in q3.columns]
        _ = q3[disp3].head(25).style.format({"price_change_pct":"{:+.2f}%","volume_vs_median":"{:.1f}x","close_price":"{:.2f}","net_traded_value":"{:,.0f}"}).to_html()
    print("Tab6: OK")
except Exception:
    errors["Tab6"] = traceback.format_exc()
    print("Tab6: ERROR")

# TAB 8
try:
    vdf = ana.analyze_volatility()
    if not vdf.empty:
        avg_rng = vdf["intraday_range_pct"].mean()
        vc_col = "volatility_class" if "volatility_class" in vdf.columns else None
        fig = px.histogram(vdf, x="intraday_range_pct", nbins=40, color=vc_col, color_discrete_map={"Low Vol":"#2ecc71","Mid Vol":"#f39c12","High Vol":"#e74c3c"})
        fig.add_vline(x=avg_rng, line_dash="dash", line_color="navy", annotation_text=f"Avg {avg_rng:.2f}%")
        if "price_change_pct" in vdf.columns:
            plot_v = vdf.dropna(subset=["price_change_pct"]).copy()
            plot_v["_sz"] = _safe_size(plot_v["net_traded_value"], 1e8)
            fig2 = px.scatter(plot_v, x="intraday_range_pct", y="price_change_pct", color=vc_col, hover_data=["symbol"], size="_sz", size_max=20, color_discrete_map={"Low Vol":"#2ecc71","Mid Vol":"#f39c12","High Vol":"#e74c3c"})
            fig2.add_hline(y=0, line_dash="dot", line_color="gray")
            fig2.add_vline(x=avg_rng, line_dash="dot", line_color="gray", annotation_text="avg range")
            fig2.add_annotation(text="High Vol Gainers", x=vdf["intraday_range_pct"].quantile(0.9), y=vdf["price_change_pct"].quantile(0.9), showarrow=False, font=dict(size=9, color="gray"))
        disp = [c for c in ["symbol","intraday_range_pct","true_range_pct","parkinson_vol","price_change_pct","close_price","net_traded_value"] if c in vdf.columns]
        top_v = vdf.nlargest(20, "true_range_pct" if "true_range_pct" in vdf.columns else "intraday_range_pct")
        _ = top_v[disp].style.format({c: "{:.2f}%" for c in ["intraday_range_pct","true_range_pct","parkinson_vol","price_change_pct"] if c in disp} | {"close_price": "{:.2f}", "net_traded_value": "{:,.0f}"}).to_html()
        calm = vdf.copy()
        if "net_traded_value" in calm.columns:
            calm = calm[calm["net_traded_value"] >= calm["net_traded_value"].quantile(0.30)]
        bot_v = calm.nsmallest(20, "intraday_range_pct")
        disp2 = [c for c in ["symbol","intraday_range_pct","price_change_pct","close_price","net_traded_value"] if c in bot_v.columns]
        _ = bot_v[disp2].style.format({"intraday_range_pct": "{:.2f}%", "price_change_pct": "{:+.2f}%", "close_price": "{:.2f}", "net_traded_value": "{:,.0f}"}).to_html()
        if not pr_range.empty:
            eq_r = pr_range[~pr_range["is_index"].astype(bool)].copy() if "is_index" in pr_range.columns else pr_range.copy()
            if eq_r["trade_date"].nunique() >= 5:
                pivot = eq_r.pivot_table(index="trade_date", columns="symbol", values="close_price")
                hist_vol = (pivot.pct_change() * 100).std().reset_index()
                hist_vol.columns = ["symbol", "hist_vol_30d"]
                today_rng = vdf[["symbol","intraday_range_pct"]].rename(columns={"intraday_range_pct":"today_range"})
                hist_vol = hist_vol.merge(today_rng, on="symbol", how="inner")
                hist_vol["vol_ratio"] = (hist_vol["today_range"] / hist_vol["hist_vol_30d"].replace(0, np.nan)).round(2)
                max_v = max(hist_vol["hist_vol_30d"].max(), hist_vol["today_range"].max())
                fig4 = px.scatter(hist_vol, x="hist_vol_30d", y="today_range", hover_data=["symbol"])
                fig4.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash", color="gray"))
                unusual = hist_vol.nlargest(25, "vol_ratio")
                _ = unusual.style.format({"hist_vol_30d": "{:.2f}%", "today_range": "{:.2f}%", "vol_ratio": "{:.2f}x"}).to_html()
    print("Tab8: OK")
except Exception:
    errors["Tab8"] = traceback.format_exc()
    print("Tab8: ERROR")

print()
for tab, tb in errors.items():
    print(f"=== {tab} TRACEBACK ===")
    print(tb)
