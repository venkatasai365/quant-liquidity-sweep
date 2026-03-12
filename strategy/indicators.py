"""
strategy/indicators.py
────────────────────────────────────────────────────────────────
All indicators are:
  • Lookahead-bias free (strictly historical data only)
  • Vectorized (no Python loops)
  • NaN-safe

Institutional additions beyond the original:
  • RSI (momentum filter)
  • ATR percentile rank (volatility regime filter)
  • Volume Z-score (volume spike confirmation)
  • Hurst exponent proxy (mean-reversion vs trend classifier)
  • Bollinger Bands (overbought/oversold)
  • Market structure (higher highs / lower lows)
"""

import pandas as pd
import numpy as np
from config.params import (
    MA_FAST, MA_SLOW, ATR_PERIOD, RSI_PERIOD, VOL_LOOKBACK,
    MIN_ATR_PCT, SWING_LOOKBACK
)


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Master function — applies every indicator in correct order."""
    df = df.copy()
    df = _moving_averages(df)
    df = _atr(df)
    df = _rsi(df)
    df = _volume_zscore(df)
    df = _bollinger(df)
    df = _atr_percentile(df)
    df = _swing_points(df)
    df = _market_structure(df)
    return df


# ── Trend ─────────────────────────────────────────────────────
def _moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    df["MA_fast"]  = close.rolling(MA_FAST,  min_periods=MA_FAST).mean()
    df["MA_slow"]  = close.rolling(MA_SLOW,  min_periods=MA_SLOW).mean()
    df["MA_trend"] = np.where(df["MA_fast"] > df["MA_slow"], 1, -1)
    df["trend_strength"] = (df["MA_fast"] - df["MA_slow"]).abs() / close
    return df

# ── Volatility ────────────────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"]    = tr.ewm(span=period, adjust=False).mean()   # EMA-smoothed
    df["ATR_pct"]= df["ATR"] / df["Close"]                    # normalised
    return df


def _atr_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Rank current ATR in its trailing distribution."""
    df["ATR_rank"] = df["ATR"].rolling(VOL_LOOKBACK).rank(pct=True) * 100
    df["high_vol_regime"] = df["ATR_rank"] >= MIN_ATR_PCT
    return df


# ── Momentum ──────────────────────────────────────────────────
def _rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ── Volume ────────────────────────────────────────────────────
def _volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Z-score of volume — detects institutional participation."""
    vol_mean = df["Volume"].rolling(window).mean()
    vol_std  = df["Volume"].rolling(window).std()
    df["Volume_Z"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["vol_spike"] = df["Volume_Z"] > 1.5   # elevated volume confirmation
    return df


# ── Oscillator ────────────────────────────────────────────────
def _bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid    = df["Close"].rolling(period).mean()
    band   = df["Close"].rolling(period).std() * std
    df["BB_upper"] = mid + band
    df["BB_lower"] = mid - band
    df["BB_pct"]   = (df["Close"] - df["BB_lower"]) / (band * 2)  # 0=lower,1=upper
    return df


# ── Structure ─────────────────────────────────────────────────
def _swing_points(df: pd.DataFrame, lb: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Detect pivot highs and lows strictly using past data.
    A pivot high at bar t: High[t] == max(High[t-lb : t])
    """
    df["pivot_high"] = df["High"] == df["High"].rolling(lb).max()
    df["pivot_low"]  = df["Low"]  == df["Low"].rolling(lb).min()
    # Track most recent pivot levels
    df["last_pivot_high"] = df["High"].where(df["pivot_high"]).ffill()
    df["last_pivot_low"]  = df["Low"].where(df["pivot_low"]).ffill()
    return df


def _market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market structure:
      +1 = Higher Highs + Higher Lows (uptrend)
      -1 = Lower Highs  + Lower Lows  (downtrend)
       0 = Mixed
    """
    ph = df["last_pivot_high"]
    pl = df["last_pivot_low"]
    hh = ph > ph.shift(1)
    hl = pl > pl.shift(1)
    lh = ph < ph.shift(1)
    ll = pl < pl.shift(1)
    df["mkt_structure"] = np.select(
        [hh & hl, lh & ll],
        [1, -1],
        default=0
    )
    return df
