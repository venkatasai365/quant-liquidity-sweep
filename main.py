"""
main.py  —  Asset-Specific Mean Reversion + Momentum Strategy
Run:  python main.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUTPUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    import yfinance as yf
    USE_LIVE = True
except ImportError:
    USE_LIVE = False

from config.params          import ASSETS, INITIAL_CAPITAL
from strategy.indicators    import add_all
from strategy.signals       import generate_signals
from engine.backtest_engine import BacktestEngine
from utils.analytics        import compute_metrics, compute_portfolio_metrics
from research.walk_forward  import walk_forward, monte_carlo
from utils.visualizer       import generate_report

SIM_START = "2018-01-01"
SIM_END   = "2024-06-01"
RUN_WF    = True
RUN_MC    = True

# yfinance ticker map — handles EURUSD=X correctly
TICKER_MAP = {
    "BTC-USD": "BTC-USD",
    "SPY":     "SPY",
    "EURUSD":  "EURUSD=X",
    "GLD":     "GLD",
}

# Per-asset params tuned to each signal type
ASSET_PARAMS = {
    "BTC-USD": dict(min_score=4, atr_sl=2.5, max_hold=15, asset_class="crypto"),
    "SPY":     dict(min_score=4, atr_sl=2.0, max_hold=10, asset_class="equity"),
    "EURUSD":  dict(min_score=4, atr_sl=1.5, max_hold=10, asset_class="fx"),
    "GLD":     dict(min_score=4, atr_sl=2.0, max_hold=20, asset_class="commodity"),
}


def load_data():
    if not USE_LIVE:
        from engine.data_engine import MarketDataEngine
        return MarketDataEngine(seed=42).get_all()

    raw_data = {}
    for symbol in ASSETS:
        # Always look up via TICKER_MAP, fallback to symbol itself
        ticker = TICKER_MAP.get(symbol, symbol)
        print(f"  Downloading {symbol} ({ticker})...", end="", flush=True)
        try:
            df = yf.download(ticker, start=SIM_START, end=SIM_END,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.dropna(inplace=True)
            for col in df.columns:
                df[col] = df[col].squeeze().astype(float)
            print(f"  {len(df)} bars OK")
            raw_data[symbol] = df
        except Exception as e:
            print(f"  FAILED: {e} — using synthetic")
            from engine.data_engine import MarketDataEngine
            raw_data[symbol] = MarketDataEngine(seed=42).get(symbol)
    return raw_data


def run_asset(symbol, raw_df):
    import config.params as P
    ap = ASSET_PARAMS[symbol]
    P.ATR_SL_MULT   = ap["atr_sl"]
    P.ATR_TP_MULT   = 9.0
    P.MAX_HOLD_BARS = ap["max_hold"]

    print(f"    indicators...", end="", flush=True)
    df = add_all(raw_df)

    print(f" signals...", end="", flush=True)
    df = generate_signals(df,
                          min_score=ap["min_score"],
                          asset_class=ap["asset_class"])

    print(f" backtest...", end="", flush=True)
    eng = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    df  = eng.run(df)
    m   = compute_metrics(df, eng.trades, symbol=symbol)

    sharpe = m['Sharpe Ratio']
    flag   = "✅" if float(str(sharpe).replace('—','0')) > 0 else "⚠️"
    print(f" {flag}  trades={m['Total Trades']}  WR={m['Win Rate %']}  CAGR={m['CAGR %']}  Sharpe={sharpe}")
    return df, m, eng.trades


def main():
    print("\n" + "="*64)
    print("  ASSET-SPECIFIC QUANT STRATEGY  |  MULTI-ASSET BACKTEST")
    print("="*64)
    print(f"  Strategy per asset:")
    print(f"    BTC-USD  → MA50 Pullback + Momentum Breakout")
    print(f"    SPY      → Trend Breakout with Momentum")
    print(f"    EURUSD   → Buy Dip in Uptrend")
    print(f"    GLD      → Bollinger Band Lower Bounce")
    print(f"  Chart → {OUTPUT_DIR}\n")

    print("[1/5] Loading market data...")
    raw_data = load_data()

    print("\n[2/5] Per-asset backtests:")
    results = {}
    for sym in ASSETS:
        if sym not in raw_data:
            continue
        print(f"  {sym:<12}", end="", flush=True)
        try:
            df, m, trades = run_asset(sym, raw_data[sym])
            results[sym] = (df, m, trades)
        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("  No assets completed.")
        return

    print("\n[3/5] Walk-forward validation (SPY)...")
    wf_log = pd.DataFrame()
    wf_sym = "SPY" if "SPY" in raw_data else list(raw_data.keys())[0]
    if RUN_WF:
        try:
            import config.params as P
            P.ATR_SL_MULT=2.0; P.MAX_HOLD_BARS=10
            _, wf_log = walk_forward(raw_data[wf_sym])
            pos = (wf_log["oos_sharpe"] > 0).sum()
            avg = wf_log["oos_sharpe"].mean()
            print(f"  Windows={len(wf_log)}  AvgSharpe={avg:.3f}  Positive={pos}/{len(wf_log)}")
        except Exception as e:
            print(f"  {e}")

    print("\n[4/5] Monte Carlo (SPY)...")
    mc_data = {}
    mc_sym  = "SPY" if "SPY" in results else list(results.keys())[0]
    if RUN_MC:
        try:
            t = results[mc_sym][2]
            if len(t) > 0:
                mc_data = monte_carlo(t["pnl"], n_trades_forward=120)
                print(f"  ProbRuin={mc_data['prob_ruin']}%  Median={mc_data['median_return']}%  P5={mc_data['p5_return']}%  P95={mc_data['p95_return']}%")
        except Exception as e:
            print(f"  {e}")

    print("\n[5/5] Portfolio analytics...")
    port_ret = pd.Series(dtype=float)
    port_metrics = {}
    try:
        ret_dict     = {s: df["daily_ret"].fillna(0) for s,(df,_,_) in results.items()}
        port_df      = pd.DataFrame(ret_dict).dropna()
        port_ret     = port_df.mean(axis=1)
        port_metrics = compute_portfolio_metrics({s: port_df[s] for s in port_df.columns})
        sharpe = port_metrics['Portfolio Sharpe']
        cagr   = port_metrics['Portfolio CAGR']
        corr   = port_metrics['Avg Cross-Corr']
        flag   = "✅" if float(str(sharpe).replace('—','0')) > 0 else "⚠️"
        print(f"  {flag} Sharpe={sharpe}  CAGR={cagr}  AvgCorr={corr}")
    except Exception as e:
        print(f"  {e}")

    _print_table(results, port_metrics)

    print("\nGenerating research report chart...")
    chart = os.path.join(OUTPUT_DIR, "quant_research_report.png")
    try:
        generate_report(results=results, mc_data=mc_data,
                        wf_log=wf_log, port_ret=port_ret, out_path=chart)
        print(f"\n  ✅ Chart saved to:\n  {chart}")
    except Exception as e:
        print(f"  Chart error: {e}")

    print("\n" + "="*64)
    print("  DONE  —  open results/quant_research_report.png")
    print("="*64 + "\n")


def _print_table(results, port_metrics):
    print("\n" + "="*76)
    print("  PERFORMANCE SUMMARY")
    print("="*76)
    keys = ["Sharpe Ratio","Sortino Ratio","Calmar Ratio",
            "CAGR %","Total Return %","Max Drawdown %",
            "Win Rate %","Profit Factor","Total Trades","Expectancy %"]
    syms = list(results.keys())
    print(f"{'Metric':<28}" + "".join(f"{s:>12}" for s in syms))
    print("-"*76)
    for k in keys:
        row = f"{k:<28}"
        for s in syms:
            v = str(results[s][1].get(k,'—'))
            row += f"{v:>12}"
        print(row)
    print("-"*76)
    if port_metrics:
        print(f"\n  Portfolio Sharpe : {port_metrics.get('Portfolio Sharpe','—')}")
        print(f"  Portfolio CAGR   : {port_metrics.get('Portfolio CAGR','—')}")
        print(f"  Avg Correlation  : {port_metrics.get('Avg Cross-Corr','—')}")
        print(f"  Diversification  : {port_metrics.get('Diversification','—')}")
    print("="*76)


if __name__ == "__main__":
    main()