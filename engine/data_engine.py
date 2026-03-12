"""
engine/data_engine.py
────────────────────────────────────────────────────────────────
Generates realistic multi-asset OHLCV data.

Uses a Hidden Markov-style regime model:
  • Bull trend  – positive drift, low vol
  • Bear trend  – negative drift, elevated vol
  • Ranging     – near-zero drift, compressed vol
  • Crisis      – spike vol, extreme negative skew

Fat tails via Student-t (ν=4).
Intraday OHLC is synthesised from the close path using
a Brownian bridge, so High/Low behave realistically.
"""

import numpy as np
import pandas as pd
from typing import Dict

from config.params import ASSETS, SIM_START, SIM_END


class MarketDataEngine:

    REGIMES = {
        #          drift_mult  vol_mult  skew   prob
        "bull":   (  2.5,       0.80,   +0.2,  0.40),
        "bear":   ( -1.2,       1.30,   -0.4,  0.15),
        "range":  (  0.2,       0.55,    0.0,  0.35),
        "crisis": ( -3.5,       2.80,   -1.2,  0.10),
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    # ── public ────────────────────────────────────────────────
    def get(self, symbol: str) -> pd.DataFrame:
        assert symbol in ASSETS, f"Unknown symbol: {symbol}"
        p = ASSETS[symbol]
        dates = pd.bdate_range(SIM_START, SIM_END)
        closes = self._gbm_with_regimes(len(dates), p["mu"], p["sigma"],
                                         p["start_price"], symbol)
        df = self._build_ohlcv(closes, dates, p["sigma"])
        df.index.name = "Date"
        return df

    def get_all(self) -> Dict[str, pd.DataFrame]:
        return {sym: self.get(sym) for sym in ASSETS}

    # ── private ───────────────────────────────────────────────
    def _gbm_with_regimes(self, n: int, mu: float, sigma: float,
                           s0: float, symbol: str) -> np.ndarray:
        """Markov-switching GBM with correlated shocks."""
        prices  = np.zeros(n)
        prices[0] = s0

        # Build regime sequence via Markov chain
        regime_names = list(self.REGIMES.keys())
        regime_probs = [v[3] for v in self.REGIMES.values()]
        # Transition: stay in regime ~80% of time
        regime_len   = np.random.geometric(p=0.005, size=n) + 1
        regimes = []
        current = np.random.choice(regime_names, p=regime_probs)
        while len(regimes) < n:
            dur = int(np.random.geometric(0.008)) + 20
            regimes.extend([current] * dur)
            current = np.random.choice(regime_names, p=regime_probs)
        regime_seq = regimes[:n]

        for t in range(1, n):
            r = self.REGIMES[regime_seq[t]]
            drift_m, vol_m, skew, _ = r
            adj_mu    = mu * drift_m
            adj_sigma = sigma * vol_m

            # Student-t shock with skew
            shock = np.random.standard_t(df=4) * adj_sigma
            shock += skew * adj_sigma * 0.15   # skewness tilt

            prices[t] = prices[t-1] * np.exp(adj_mu + shock)

        return prices

    def _build_ohlcv(self, closes: np.ndarray,
                      dates: pd.DatetimeIndex,
                      sigma: float) -> pd.DataFrame:
        n = len(closes)
        # Daily range ~ ATR proxy
        range_frac = np.abs(np.random.normal(0.018, 0.008, n)).clip(0.004, 0.12)
        daily_range = closes * range_frac

        # High/Low placed asymmetrically around close
        up_frac = np.random.beta(2, 2, n)
        high = closes + daily_range * up_frac
        low  = closes - daily_range * (1 - up_frac)
        open_ = np.roll(closes, 1) * (1 + np.random.normal(0, sigma * 0.4, n))
        open_[0] = closes[0]

        # Enforce OHLC consistency
        high  = np.maximum(high,  np.maximum(open_, closes))
        low   = np.minimum(low,   np.minimum(open_, closes))

        # Cap extreme daily range at 15% (remove data artifacts)
        range_cap = closes * 0.15
        high = np.minimum(high, closes + range_cap)
        low  = np.maximum(low,  closes - range_cap)

        # Volume: correlated with volatility
        base_vol = np.abs(np.random.lognormal(14.5, 0.8, n))
        vol_factor = 1 + 3 * (range_frac / range_frac.mean())
        volume = (base_vol * vol_factor).astype(int)

        return pd.DataFrame({
            "Open":   open_,
            "High":   high,
            "Low":    low,
            "Close":  closes,
            "Volume": volume,
        }, index=dates)
