import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "symbol": ["A","B","C","D","E"],
    "price": [100.5, 200.3, 50.1, 300.2, 150.8],
    "change_pct": [1.5, -2.3, 0.5, -1.1, 3.2],
    "volume": [1000000, 500000, 200000, 800000, 300000],
})

st.title("Styler Test")

st.subheader("Plain dataframe (should work)")
try:
    st.dataframe(df, hide_index=True)
    st.success("Plain OK")
except Exception as e:
    st.error(f"Plain failed: {e}")

st.subheader("Styled dataframe with hide_index=True (failing tabs pattern)")
try:
    styled = df.style.format({"price": "{:.2f}", "change_pct": "{:+.2f}%", "volume": "{:,.0f}"})
    st.dataframe(styled, hide_index=True)
    st.success("Styled OK")
except Exception as e:
    import traceback
    st.error(f"Styled failed: {e}")
    st.code(traceback.format_exc())

st.subheader("Styled dataframe WITHOUT hide_index (alternative)")
try:
    styled2 = df.style.format({"price": "{:.2f}", "change_pct": "{:+.2f}%", "volume": "{:,.0f}"}).hide(axis="index")
    st.dataframe(styled2, use_container_width=True)
    st.success("Styled without hide_index OK")
except Exception as e:
    import traceback
    st.error(f"Styled without hide_index failed: {e}")
    st.code(traceback.format_exc())
