"""
Market Causality & Lead-Lag Analysis
=====================================
Uses return-series correlation at various time lags to identify:
  - Which stocks move together (correlation matrix)
  - Which stocks lead or lag the broader market
  - Pair-wise lead-lag relationships

No statsmodels required — uses numpy/scipy for all calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


class LeadLagAnalyzer:
    """
    Compute lead-lag relationships between NSE equities.

    Parameters
    ----------
    price_data : pd.DataFrame
        Multi-date DataFrame from fact_daily_prices.
        Must contain columns: trade_date, symbol, close_price, is_index.
    """

    def __init__(self, price_data: pd.DataFrame) -> None:
        if price_data.empty:
            raise ValueError("price_data is empty")

        # Filter out index rows
        eq = price_data[~price_data["is_index"].fillna(False)].copy()
        eq["trade_date"] = pd.to_datetime(eq["trade_date"])

        # Drop duplicate (date, symbol) pairs before pivoting
        eq = eq.drop_duplicates(subset=["trade_date", "symbol"], keep="last")

        # Pivot: rows = trade_date, cols = symbol
        pivot = eq.pivot_table(
            index="trade_date", columns="symbol", values="close_price", aggfunc="last"
        )
        # Ensure unique date index
        pivot = pivot[~pivot.index.duplicated(keep="last")]
        # Daily log-returns (more stationary than price levels)
        self.returns: pd.DataFrame = np.log(pivot / pivot.shift(1)).dropna(how="all")
        self.symbols: list[str] = list(self.returns.columns)
        self.dates: list = list(self.returns.index)

        # Precompute market proxy = equal-weighted average return
        self._market_return: pd.Series = self.returns.mean(axis=1)

    # ── public methods ─────────────────────────────────────────────────────────

    def correlation_matrix(self, top_n: int = 30) -> pd.DataFrame:
        """
        Pearson correlation matrix (lag 0) for the top_n most-observed symbols.

        Returns
        -------
        pd.DataFrame  (top_n × top_n)
        """
        # Pick symbols with the most non-NaN observations
        counts = self.returns.count().nlargest(top_n)
        sel = self.returns[counts.index].dropna(how="all")
        return sel.corr(method="pearson")

    def lead_lag_correlation(
        self, symbol: str, max_lag: int = 5
    ) -> pd.DataFrame:
        """
        Correlate `symbol`'s returns (at various lags) against every other symbol.

        A positive lag means `symbol` leads: its return at t predicts others at t+lag.
        A negative lag means `symbol` lags: others at t predict symbol at t-lag.

        Returns
        -------
        pd.DataFrame  columns: lag, other_symbol, correlation
            Sorted by abs(correlation) descending within each lag.
        """
        if symbol not in self.returns.columns:
            raise ValueError(f"Symbol '{symbol}' not found in price data")

        sym_ret = self.returns[symbol]
        others = self.returns.drop(columns=[symbol])
        records = []

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                # symbol leads: correlate shifted symbol against others at t
                corrs = others.corrwith(sym_ret.shift(lag))
            else:
                # symbol lags: correlate symbol against shifted others at t
                corrs = others.shift(-lag).corrwith(sym_ret)

            for sym, c in corrs.items():
                if not np.isnan(c):
                    records.append({"lag": lag, "symbol": sym, "correlation": c})

        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["abs_corr"] = df["correlation"].abs()
        return df.sort_values(["lag", "abs_corr"], ascending=[True, False]).drop(
            columns="abs_corr"
        )

    def find_market_leaders(self, top_n: int = 15) -> pd.DataFrame:
        """
        Identify stocks whose return at t-1 best predicts the market return at t.

        'Market return' = equal-weighted average of all equities.

        Returns
        -------
        pd.DataFrame  columns: symbol, lag1_corr, direction, strength
            Sorted by abs(lag1_corr) descending.
        """
        market = self._market_return
        # Correlate each stock's lag-1 return with the market return at t
        corrs = self.returns.shift(1).corrwith(market).dropna()
        if corrs.empty:
            return pd.DataFrame(columns=["symbol", "lag1_corr", "direction", "strength"])

        top = corrs.abs().nlargest(top_n)
        corrs = corrs.loc[top.index].sort_values(key=abs, ascending=False)

        df = pd.DataFrame({
            "symbol": corrs.index,
            "lag1_corr": corrs.round(4).values,
        })
        df["direction"] = df["lag1_corr"].apply(lambda c: "Leading +" if c > 0 else "Leading \u2212")
        df["strength"] = df["lag1_corr"].abs().apply(
            lambda c: "Strong" if c > 0.4 else ("Moderate" if c > 0.2 else "Weak")
        )
        return df.reset_index(drop=True)

    def rolling_correlation(
        self, sym1: str, sym2: str, window: int = 10
    ) -> pd.Series:
        """
        Rolling Pearson correlation between two symbols' returns.

        Returns
        -------
        pd.Series  indexed by trade_date
        """
        if sym1 not in self.returns.columns:
            raise ValueError(f"Symbol '{sym1}' not in data")
        if sym2 not in self.returns.columns:
            raise ValueError(f"Symbol '{sym2}' not in data")
        combined = self.returns[[sym1, sym2]].dropna()
        return combined[sym1].rolling(window).corr(combined[sym2]).dropna()

    def clustered_correlation_matrix(self, top_n: int = 30) -> pd.DataFrame:
        """
        Correlation matrix reordered by hierarchical clustering (Ward linkage).
        Use with px.imshow for the cluster heatmap.

        Returns
        -------
        pd.DataFrame  reordered correlation matrix
        """
        corr = self.correlation_matrix(top_n=top_n)
        if corr.empty or len(corr) < 2:
            return corr

        # Fill NaN correlations (insufficient overlap) with 0 before clustering
        corr = corr.fillna(0)

        # Convert correlation distance (1 - |corr|) to condensed form
        dist = 1 - corr.abs()
        np.fill_diagonal(dist.values, 0)
        condensed = squareform(dist.values, checks=False)
        condensed = np.clip(condensed, 0, 2)  # ensure finite [0, 2]
        condensed = np.nan_to_num(condensed, nan=1.0, posinf=2.0, neginf=0.0)

        linkage = hierarchy.linkage(condensed, method="ward")
        order = hierarchy.leaves_list(linkage)

        reordered = corr.iloc[order, :].iloc[:, order]
        return reordered

    def lag_profile(self, max_lag: int = 5) -> pd.DataFrame:
        """
        For each lag -max_lag..+max_lag, compute the average absolute correlation
        across all stock pairs. Shows whether the market has persistent lead-lag structure.

        Returns
        -------
        pd.DataFrame  columns: lag, avg_abs_corr, n_pairs
        """
        market = self._market_return
        records = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                # stocks lead: shifted stock returns vs market at t
                corrs = self.returns.shift(lag).corrwith(market).dropna().abs()
            else:
                # stocks lag: stock returns vs shifted market
                corrs = self.returns.corrwith(market.shift(-lag)).dropna().abs()

            records.append(
                {
                    "lag": lag,
                    "avg_abs_corr": round(float(corrs.mean()), 4) if len(corrs) > 0 else np.nan,
                    "n_pairs": len(corrs),
                }
            )
        return pd.DataFrame(records)
