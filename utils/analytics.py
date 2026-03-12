"""
utils/analytics.py
────────────────────────────────────────────────────────────────
Institutional performance analytics.

Metrics computed:
  Core:        Sharpe, Sortino, Calmar, Max Drawdown, CAGR
  Risk:        VaR (95/99), CVaR, Omega ratio, Tail ratio
  Trade stats: Win rate, Avg W/L, Profit factor, Expectancy
  Style:       Skewness, Kurtosis, Hit rate by exit type
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any


def compute_metrics(df: pd.DataFrame,
                    trades_df: pd.DataFrame,
                    symbol: str = "") -> Dict[str, Any]:

    ret = df["daily_ret"].dropna()
    eq  = df["equity_norm"].dropna()

    ann = 252
    m = {}

    # ── Core ──────────────────────────────────────────────────
    cagr = (eq.iloc[-1] ** (ann / len(eq))) - 1
    m["CAGR %"]        = _pct(cagr)
    m["Total Return %"]= _pct(eq.iloc[-1] - 1)

    vol = ret.std() * np.sqrt(ann)
    m["Ann. Volatility %"] = _pct(vol)

    sharpe = (ret.mean() * ann) / (ret.std() * np.sqrt(ann) + 1e-9)
    m["Sharpe Ratio"] = round(sharpe, 3)

    neg = ret[ret < 0]
    sortino = (ret.mean() * ann) / (neg.std() * np.sqrt(ann) + 1e-9)
    m["Sortino Ratio"] = round(sortino, 3)

    # ── Drawdown ──────────────────────────────────────────────
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    m["Max Drawdown %"] = _pct(dd.min())
    m["Avg Drawdown %"] = _pct(dd[dd < 0].mean() if (dd < 0).any() else 0)

    calmar = cagr / abs(dd.min() + 1e-9)
    m["Calmar Ratio"] = round(calmar, 3)

    # ── Drawdown duration ─────────────────────────────────────
    in_dd = dd < 0
    max_dur = 0
    cur_dur = 0
    for x in in_dd:
        cur_dur = cur_dur + 1 if x else 0
        max_dur = max(max_dur, cur_dur)
    m["Max DD Duration (days)"] = max_dur

    # ── Risk measures ─────────────────────────────────────────
    m["VaR 95% (daily)"] = _pct(np.percentile(ret, 5))
    m["CVaR 95% (daily)"]= _pct(ret[ret <= np.percentile(ret, 5)].mean())
    m["VaR 99% (daily)"] = _pct(np.percentile(ret, 1))

    # Omega ratio (threshold=0)
    gains = ret[ret > 0].sum()
    losses= ret[ret < 0].abs().sum()
    m["Omega Ratio"] = round(gains / (losses + 1e-9), 3)

    # ── Distribution ─────────────────────────────────────────
    m["Return Skew"]    = round(float(stats.skew(ret)), 3)
    m["Return Kurtosis"]= round(float(stats.kurtosis(ret)), 3)

    # ── Trade analytics ───────────────────────────────────────
    if trades_df is not None and len(trades_df) > 0:
        t = trades_df
        m["Total Trades"]  = len(t)
        wins   = t[t["pnl"] > 0]
        losses_t = t[t["pnl"] < 0]
        m["Win Rate %"]    = _pct(len(wins) / len(t))
        m["Avg Win %"]     = _pct(wins["pnl"].mean())   if len(wins) else "0%"
        m["Avg Loss %"]    = _pct(losses_t["pnl"].mean()) if len(losses_t) else "0%"

        if len(wins) > 0 and len(losses_t) > 0:
            w2l = abs(wins["pnl"].mean() / losses_t["pnl"].mean())
            m["Win/Loss Ratio"] = round(w2l, 2)
            pf = wins["pnl"].sum() / (losses_t["pnl"].abs().sum() + 1e-9)
            m["Profit Factor"]  = round(pf, 3)
            exp = (len(wins)/len(t)) * wins["pnl"].mean() + \
                  (len(losses_t)/len(t)) * losses_t["pnl"].mean()
            m["Expectancy %"]   = _pct(exp)
        else:
            m["Win/Loss Ratio"] = "—"
            m["Profit Factor"]  = "—"
            m["Expectancy %"]   = "—"

        if "exit_reason" in t.columns:
            for reason in ["tp", "stop", "time"]:
                sub = t[t["exit_reason"] == reason]
                m[f"Exit:{reason} %"] = _pct(len(sub)/len(t)) if len(t) else "0%"
    else:
        m["Total Trades"] = 0

    return m


def _pct(x: float) -> str:
    if isinstance(x, float):
        return f"{x*100:.2f}%"
    return str(x)


def compute_portfolio_metrics(asset_rets: Dict[str, pd.Series],
                               weights: Dict[str, float] = None) -> Dict:
    """Combine per-asset returns into portfolio-level analytics."""
    df = pd.DataFrame(asset_rets).dropna()
    if weights is None:
        weights = {k: 1/len(df.columns) for k in df.columns}

    w = np.array([weights[c] for c in df.columns])
    port_ret = df.values @ w

    ann = 252
    sharpe = (port_ret.mean() * ann) / (port_ret.std() * np.sqrt(ann) + 1e-9)
    corr   = df.corr()

    return {
        "Portfolio Sharpe":   round(sharpe, 3),
        "Portfolio CAGR":     f"{((1+port_ret.mean())**ann - 1)*100:.2f}%",
        "Avg Cross-Corr":     round(corr.values[np.triu_indices_from(corr.values, k=1)].mean(), 3),
        "Diversification":    round(1 - corr.values[np.triu_indices_from(corr.values, k=1)].mean(), 3),
        "correlation_matrix": corr,
    }
