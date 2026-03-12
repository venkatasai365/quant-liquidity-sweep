"""
research/walk_forward.py
────────────────────────────────────────────────────────────────
Walk-forward analysis + Monte Carlo simulation.

Walk-forward:
  • Splits data into rolling train/test windows
  • Optimises min_score threshold on train set
  • Applies best params to out-of-sample test window
  • Stitches OOS returns together for unbiased performance

Monte Carlo:
  • Bootstraps trade returns with replacement (N=2000 paths)
  • Reports percentile bands for equity curves
  • Estimates probability of ruin
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from config.params import (
    WF_TRAIN_BARS, WF_TEST_BARS, WF_STEP_BARS,
    MC_SIMULATIONS, MC_SEED
)
from strategy.indicators import add_all
from strategy.signals     import generate_signals
from engine.backtest_engine import BacktestEngine


# ─────────────────────────────────────────────────────────────
#  Walk-Forward
# ─────────────────────────────────────────────────────────────

def walk_forward(df_raw: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      oos_returns : stitched out-of-sample daily returns
      wf_log      : DataFrame logging each window's params + Sharpe
    """
    total  = len(df_raw)
    log    = []
    oos_chunks: List[pd.Series] = []

    start = 0
    while start + WF_TRAIN_BARS + WF_TEST_BARS <= total:
        train_end = start + WF_TRAIN_BARS
        test_end  = train_end + WF_TEST_BARS

        train_raw = df_raw.iloc[start:train_end].copy()
        test_raw  = df_raw.iloc[train_end:test_end].copy()

        # Optimise min_score on training window
        best_score_thresh, best_sharpe = _optimise_threshold(train_raw)

        # Apply to test window
        test_df = add_all(test_raw)
        test_df = generate_signals(test_df, min_score=best_score_thresh)
        eng     = BacktestEngine()
        test_df = eng.run(test_df)

        oos_ret = test_df["daily_ret"].fillna(0)
        oos_chunks.append(oos_ret)

        sharpe_oos = (oos_ret.mean() * 252) / (oos_ret.std() * 252**0.5 + 1e-9)
        log.append({
            "window_start":   df_raw.index[start],
            "train_end":      df_raw.index[train_end - 1],
            "test_end":       df_raw.index[test_end - 1],
            "best_threshold": best_score_thresh,
            "train_sharpe":   round(best_sharpe, 3),
            "oos_sharpe":     round(sharpe_oos, 3),
        })

        start += WF_STEP_BARS

    oos_returns = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float)
    wf_log      = pd.DataFrame(log)
    return oos_returns, wf_log


def _optimise_threshold(df_raw: pd.DataFrame) -> Tuple[int, float]:
    """Grid search over min_score [4,5,6] on training data."""
    best_thresh, best_sh = 5, -999
    df_ind = add_all(df_raw)
    for thresh in [4, 5, 6]:
        df_sig = generate_signals(df_ind, min_score=thresh)
        eng    = BacktestEngine()
        result = eng.run(df_sig)
        r = result["daily_ret"].fillna(0)
        sh = (r.mean() * 252) / (r.std() * 252**0.5 + 1e-9)
        if sh > best_sh:
            best_sh, best_thresh = sh, thresh
    return best_thresh, best_sh


# ─────────────────────────────────────────────────────────────
#  Monte Carlo
# ─────────────────────────────────────────────────────────────

def monte_carlo(trade_returns: pd.Series,
                n_trades_forward: int = 100) -> Dict:
    """
    Bootstrap trade PnL to simulate N paths forward.
    Returns percentile bands and probability of ruin (<-20%).
    """
    np.random.seed(MC_SEED)
    trade_ret = trade_returns[trade_returns != 0].values
    if len(trade_ret) < 5:
        return {"error": "insufficient trades for MC"}

    paths = np.zeros((MC_SIMULATIONS, n_trades_forward + 1))
    paths[:, 0] = 1.0

    for sim in range(MC_SIMULATIONS):
        sample = np.random.choice(trade_ret, size=n_trades_forward, replace=True)
        paths[sim, 1:] = np.cumprod(1 + sample)

    final_equity = paths[:, -1]
    prob_ruin    = (final_equity < 0.80).mean()   # <20% loss = ruin threshold

    return {
        "paths":         paths,
        "p5":            np.percentile(paths, 5,  axis=0),
        "p25":           np.percentile(paths, 25, axis=0),
        "p50":           np.percentile(paths, 50, axis=0),
        "p75":           np.percentile(paths, 75, axis=0),
        "p95":           np.percentile(paths, 95, axis=0),
        "prob_ruin":     round(prob_ruin * 100, 1),
        "median_return": round((np.median(final_equity) - 1) * 100, 1),
        "p5_return":     round((np.percentile(final_equity, 5) - 1) * 100, 1),
        "p95_return":    round((np.percentile(final_equity, 95) - 1) * 100, 1),
        "n_trades_sim":  n_trades_forward,
    }
