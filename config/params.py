"""
config/params.py
────────────────────────────────────────────────────────────────
Single source of truth for every parameter in the pipeline.
Change here → propagates everywhere. No magic numbers in code.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataclasses import dataclass, field
from typing import List

# ── Assets ────────────────────────────────────────────────────
ASSETS = {
    "BTC-USD": dict(start_price=7000,  mu=0.0014, sigma=0.038, asset_class="crypto"),
    "SPY":     dict(start_price=280,   mu=0.0007, sigma=0.011, asset_class="equity"),
    "EURUSD=X":  dict(start_price=1.12,  mu=0.00005,sigma=0.004, asset_class="fx"),
    "GLD":     dict(start_price=145,   mu=0.0005, sigma=0.009, asset_class="commodity"),
}

SIM_START = "2018-01-01"
SIM_END   = "2024-06-01"

# ── Indicator params ──────────────────────────────────────────
MA_FAST       = 50
MA_SLOW       = 200
SWING_LOOKBACK= 12      # bars each side for swing detection
ATR_PERIOD    = 14
VOL_LOOKBACK  = 30      # ATR percentile window

# ── Signal params ─────────────────────────────────────────────
SWEEP_LOOKBACK  = 15    # rolling window for prev high/low
SWEEP_CONFIRM   = True  # require close to retrace back inside range
MIN_ATR_PCT     = 35    # only trade when ATR > Nth percentile (avoid chop)
RSI_PERIOD      = 14
RSI_OB          = 62    # overbought threshold (short filter)
RSI_OS          = 38    # oversold threshold  (long filter)

# ── Risk / execution ──────────────────────────────────────────
RISK_PER_TRADE   = 0.010   # 1.0% equity risk per trade
MAX_PORTFOLIO_RISK=0.06    # max 6% total portfolio risk at once
TRANSACTION_COST = 0.0006  # 0.06% per side (institutional rate)
SLIPPAGE_PCT     = 0.0002  # 0.02% slippage

# ── Trade management ──────────────────────────────────────────
ATR_SL_MULT   = 2.5    # wider stop: close-based exits need more room
ATR_TP_MULT   = 3.5    # TP at 3.5xATR, R:R=1.4
MAX_HOLD_BARS = 15     # exit at 15 bars = right where win rate peaks

# ── Walk-forward ──────────────────────────────────────────────
WF_TRAIN_BARS = 504    # ~2 years training
WF_TEST_BARS  = 126    # ~6 months out-of-sample
WF_STEP_BARS  = 63     # roll every quarter

# ── Monte Carlo ───────────────────────────────────────────────
MC_SIMULATIONS = 2000
MC_SEED        = 42

# ── Portfolio ─────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000   # $1M notional
CORRELATION_LOOKBACK = 60