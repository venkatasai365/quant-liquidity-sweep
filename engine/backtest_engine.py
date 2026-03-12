"""
engine/backtest_engine.py
────────────────────────────────────────────────────────────────
Asset-class aware exit model:
  - Crypto: hold 15 bars (MA50 pullback resolves fast)
  - Equity: hold 10 bars (trend breakout momentum fades)
  - FX:     hold 10 bars (dip recovery is quick)
  - Commodity: hold 20 bars (BB bounce takes longer)
ATR stop protects against adverse moves.
Conviction-weighted position sizing.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from config.params import (
    RISK_PER_TRADE, TRANSACTION_COST, SLIPPAGE_PCT,
    ATR_SL_MULT, MAX_HOLD_BARS, INITIAL_CAPITAL
)


@dataclass
class Trade:
    entry_bar:   int
    entry_price: float
    direction:   int
    stop_loss:   float
    take_profit: float
    size:        float
    conviction:  float
    exit_bar:    Optional[int]   = None
    exit_price:  Optional[float] = None
    exit_reason: str             = ""
    pnl:         float           = 0.0


class BacktestEngine:

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self._trades: List[Trade] = []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df     = df.copy().reset_index(drop=True)
        n      = len(df)
        equity = self.initial_capital
        trades: List[Trade] = []
        active: Optional[Trade] = None

        equity_curve = np.zeros(n)
        equity_curve[0] = equity
        daily_ret = np.zeros(n)

        for i in range(1, n):
            cl  = float(df["Close"].iloc[i])
            atr = float(df["ATR"].iloc[i])

            if active is not None:
                bars_held = i - active.entry_bar
                stop_hit  = (active.direction ==  1 and cl <= active.stop_loss) or \
                            (active.direction == -1 and cl >= active.stop_loss)
                time_exit = bars_held >= MAX_HOLD_BARS

                if stop_hit or time_exit:
                    pnl_frac = active.direction * \
                               (cl - active.entry_price) / active.entry_price
                    net_ret  = active.size * pnl_frac \
                               - 2 * (TRANSACTION_COST + SLIPPAGE_PCT)
                    daily_ret[i] += net_ret
                    equity       *= (1 + net_ret)
                    active.exit_bar    = i
                    active.exit_price  = cl
                    active.exit_reason = "stop" if stop_hit else "time"
                    active.pnl         = net_ret
                    trades.append(active)
                    active = None

            if active is None and df["signal"].iloc[i] != 0:
                signal     = int(df["signal"].iloc[i])
                conviction = float(df["conviction"].iloc[i])
                entry_px   = cl * (1 + signal * SLIPPAGE_PCT)
                sl         = entry_px - signal * ATR_SL_MULT * atr
                sl_dist    = abs(entry_px - sl) / max(abs(entry_px), 1e-9)
                risk_adj   = RISK_PER_TRADE * (0.5 + 0.5 * conviction)
                pos_size   = min(risk_adj / max(sl_dist, 1e-5), 0.25)

                active = Trade(
                    entry_bar=i, entry_price=entry_px,
                    direction=signal, stop_loss=sl,
                    take_profit=entry_px + signal * 999 * atr,
                    size=pos_size, conviction=conviction
                )

            equity_curve[i] = equity

        df["equity_curve"] = equity_curve
        df["daily_ret"]    = daily_ret
        df["equity_norm"]  = df["equity_curve"] / self.initial_capital
        self._trades = trades
        return df

    @property
    def trades(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self._trades])