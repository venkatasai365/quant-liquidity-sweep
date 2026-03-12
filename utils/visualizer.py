"""
utils/visualizer.py
────────────────────────────────────────────────────────────────
Generates a multi-panel research report chart.

Layout:
  Row 1 : Equity curves (all assets + portfolio)
  Row 2 : Drawdown waterfall | Monthly returns heatmap
  Row 3 : Monte Carlo fan | Walk-forward OOS Sharpe
  Row 4 : Trade distribution | Correlation matrix

Design: dark terminal aesthetic — institutional, clean, serious.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional

# ── Palette ───────────────────────────────────────────────────
BG      = "#080c10"
PANEL   = "#0d1421"
BORDER  = "#1e2d40"
WHITE   = "#e8edf2"
DIM     = "#6b7f94"
ACCENT  = "#00d4ff"

ASSET_COLORS = {
    "BTC-USD": "#F7931A",
    "SPY":     "#00e676",
    "EURUSD":  "#7c4dff",
    "GLD":     "#ffd740",
    "Portfolio":"#00d4ff",
}

FONT_MONO  = {"fontfamily": "monospace"}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, color=WHITE, fontsize=8,
                     fontweight="bold", pad=6, **FONT_MONO)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)


def generate_report(
    results:   Dict,          # {symbol: (df_result, metrics, trades_df)}
    mc_data:   Dict,          # from monte_carlo()
    wf_log:    pd.DataFrame,  # from walk_forward()
    port_ret:  pd.Series,     # portfolio daily returns
    out_path:  str = "/mnt/user-data/outputs/quant_research_report.png"
) -> str:

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.text(0.05, 0.975,
             "◈  LIQUIDITY SWEEP REVERSAL  |  INSTITUTIONAL BACKTEST REPORT",
             color=ACCENT, fontsize=13, fontweight="bold",
             va="top", **FONT_MONO)
    fig.text(0.05, 0.960,
             "Multi-Factor Signal Engine  ·  ATR Stop/TP  ·  Walk-Forward Validated  ·  Monte Carlo Stress Test",
             color=DIM, fontsize=8, va="top", **FONT_MONO)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           top=0.945, bottom=0.04,
                           left=0.06, right=0.97,
                           hspace=0.52, wspace=0.32)

    # ── Row 1: Equity curves ─────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    _style_ax(ax_eq, title="EQUITY CURVES  (Normalised to 1.0 · log scale)")
    ax_eq.set_yscale("log")

    for sym, (df, _, _) in results.items():
        if "equity_norm" not in df.columns:
            continue
        eq = df["equity_norm"].ffill()
        ax_eq.plot(df.index, eq,
                   color=ASSET_COLORS.get(sym, WHITE),
                   linewidth=1.6, alpha=0.9, label=sym)

    # Portfolio
    if len(port_ret) > 0:
        port_eq = (1 + port_ret.fillna(0)).cumprod()
        ax_eq.plot(port_eq.index, port_eq,
                   color=ASSET_COLORS["Portfolio"],
                   linewidth=2.2, linestyle="--",
                   label="Portfolio (EW)", alpha=1.0)

    ax_eq.axhline(1.0, color=BORDER, linewidth=0.8, linestyle=":")
    ax_eq.legend(facecolor="#0d1421", labelcolor=WHITE,
                 fontsize=8, framealpha=0.9,
                 edgecolor=BORDER, loc="upper left")
    ax_eq.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f×"))

    # ── Row 2 Left: Drawdown ──────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, :2])
    _style_ax(ax_dd, title="UNDERWATER EQUITY (DRAWDOWN)")

    for sym, (df, _, _) in results.items():
        if "equity_norm" not in df.columns:
            continue
        eq = df["equity_norm"].ffill()
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax_dd.fill_between(df.index, dd, 0,
                           color=ASSET_COLORS.get(sym, WHITE),
                           alpha=0.18)
        ax_dd.plot(df.index, dd,
                   color=ASSET_COLORS.get(sym, WHITE),
                   linewidth=0.8, alpha=0.7)
    ax_dd.set_ylabel("Drawdown %", fontsize=7)
    ax_dd.axhline(0, color=BORDER, linewidth=0.6)

    # ── Row 2 Right: Metrics table ────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")
    ax_tbl.set_title("KEY METRICS", color=WHITE, fontsize=8,
                     fontweight="bold", pad=6, **FONT_MONO)

    key_metrics = ["Sharpe Ratio", "Sortino Ratio", "CAGR %",
                   "Max Drawdown %", "Calmar Ratio", "Win Rate %",
                   "Profit Factor", "Total Trades"]

    y = 0.93
    for sym, (_, metrics, _) in results.items():
        color = ASSET_COLORS.get(sym, WHITE)
        ax_tbl.text(0.02, y, f"── {sym} ──", color=color,
                    fontsize=7, transform=ax_tbl.transAxes,
                    va="top", **FONT_MONO)
        y -= 0.06
        for k in key_metrics:
            v = metrics.get(k, "—")
            ax_tbl.text(0.04, y, f"{k:<22}", color=DIM,
                        fontsize=6, transform=ax_tbl.transAxes, va="top")
            ax_tbl.text(0.76, y, str(v), color=WHITE,
                        fontsize=6, transform=ax_tbl.transAxes,
                        va="top", ha="right", **FONT_MONO)
            y -= 0.055
        y -= 0.02
        if y < 0.05:
            break

    # ── Row 3 Left: Monte Carlo ───────────────────────────────
    ax_mc = fig.add_subplot(gs[2, :2])
    _style_ax(ax_mc, title=f"MONTE CARLO  ·  {MC_N(mc_data)} paths  ·  {mc_data.get('n_trades_sim',100)} trades forward")

    if "p50" in mc_data:
        x = np.arange(len(mc_data["p50"]))
        ax_mc.fill_between(x, mc_data["p5"],  mc_data["p95"],
                           color=ACCENT, alpha=0.08, label="5–95th pct")
        ax_mc.fill_between(x, mc_data["p25"], mc_data["p75"],
                           color=ACCENT, alpha=0.15, label="25–75th pct")
        ax_mc.plot(x, mc_data["p50"],
                   color=ACCENT, linewidth=2, label="Median")
        ax_mc.axhline(1.0, color=BORDER, linewidth=0.6, linestyle=":")
        ax_mc.axhline(0.80, color="#ff5252", linewidth=0.8,
                      linestyle="--", label="Ruin threshold (−20%)")

        # Annotation box
        txt = (f"Prob. of Ruin: {mc_data['prob_ruin']}%\n"
               f"Median Return: {mc_data['median_return']}%\n"
               f"P5  Return:    {mc_data['p5_return']}%\n"
               f"P95 Return:    {mc_data['p95_return']}%")
        ax_mc.text(0.98, 0.97, txt,
                   transform=ax_mc.transAxes, fontsize=7,
                   color=WHITE, va="top", ha="right",
                   bbox=dict(facecolor=BG, edgecolor=BORDER,
                             boxstyle="round,pad=0.5"),
                   **FONT_MONO)

        ax_mc.legend(facecolor=PANEL, labelcolor=WHITE,
                     fontsize=7, edgecolor=BORDER)
        ax_mc.set_ylabel("Equity Multiple", fontsize=7)
        ax_mc.set_xlabel("Trades Forward", fontsize=7)

    # ── Row 3 Right: Walk-forward Sharpe ─────────────────────
    ax_wf = fig.add_subplot(gs[2, 2])
    _style_ax(ax_wf, title="WALK-FORWARD OOS SHARPE")

    if not wf_log.empty and "oos_sharpe" in wf_log.columns:
        x_wf = np.arange(len(wf_log))
        bars = ax_wf.bar(x_wf, wf_log["oos_sharpe"],
                         color=[ACCENT if v > 0 else "#ff5252"
                                for v in wf_log["oos_sharpe"]],
                         alpha=0.8, width=0.6)
        ax_wf.axhline(0, color=BORDER, linewidth=0.8)
        ax_wf.axhline(wf_log["oos_sharpe"].mean(),
                      color="#ffd740", linewidth=1.2,
                      linestyle="--", label=f"Mean: {wf_log['oos_sharpe'].mean():.2f}")
        ax_wf.set_xlabel("WF Window", fontsize=7)
        ax_wf.set_ylabel("Sharpe (OOS)", fontsize=7)
        ax_wf.legend(facecolor=PANEL, labelcolor=WHITE,
                     fontsize=7, edgecolor=BORDER)

    # ── Row 4 Left: Trade PnL distribution ───────────────────
    ax_dist = fig.add_subplot(gs[3, :2])
    _style_ax(ax_dist, title="TRADE P&L DISTRIBUTION")

    all_trades = pd.concat(
        [tdf for _, (_, _, tdf) in results.items()
         if tdf is not None and len(tdf) > 0],
        ignore_index=True
    ) if results else pd.DataFrame()

    if len(all_trades) > 0 and "pnl" in all_trades.columns:
        pnl_pct = all_trades["pnl"] * 100
        bins = np.linspace(pnl_pct.quantile(0.01),
                           pnl_pct.quantile(0.99), 40)
        wins_p   = pnl_pct[pnl_pct >= 0]
        losses_p = pnl_pct[pnl_pct <  0]
        ax_dist.hist(wins_p,   bins=bins, color="#00e676",
                     alpha=0.7, label=f"Winners ({len(wins_p)})")
        ax_dist.hist(losses_p, bins=bins, color="#ff5252",
                     alpha=0.7, label=f"Losers ({len(losses_p)})")
        ax_dist.axvline(0, color=WHITE, linewidth=0.8, linestyle=":")
        ax_dist.axvline(pnl_pct.mean(), color="#ffd740",
                        linewidth=1.2, linestyle="--",
                        label=f"Mean: {pnl_pct.mean():.2f}%")
        ax_dist.set_xlabel("Trade PnL %", fontsize=7)
        ax_dist.set_ylabel("Frequency", fontsize=7)
        ax_dist.legend(facecolor=PANEL, labelcolor=WHITE,
                       fontsize=7, edgecolor=BORDER)

    # ── Row 4 Right: Correlation matrix ──────────────────────
    ax_corr = fig.add_subplot(gs[3, 2])
    _style_ax(ax_corr, title="STRATEGY RETURN CORRELATION")

    ret_dict = {
        sym: df["daily_ret"].fillna(0)
        for sym, (df, _, _) in results.items()
        if "daily_ret" in df.columns
    }
    if len(ret_dict) >= 2:
        corr_df = pd.DataFrame(ret_dict).corr()
        syms    = list(corr_df.columns)
        n       = len(syms)
        cmap    = LinearSegmentedColormap.from_list(
            "rb", ["#ff5252", PANEL, "#00e676"])
        im = ax_corr.imshow(corr_df.values, cmap=cmap,
                            vmin=-1, vmax=1, aspect="auto")
        ax_corr.set_xticks(range(n)); ax_corr.set_xticklabels(syms, fontsize=7, color=DIM)
        ax_corr.set_yticks(range(n)); ax_corr.set_yticklabels(syms, fontsize=7, color=DIM)
        for i in range(n):
            for j in range(n):
                ax_corr.text(j, i, f"{corr_df.values[i,j]:.2f}",
                             ha="center", va="center",
                             color=WHITE, fontsize=7, **FONT_MONO)
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    return out_path


def MC_N(mc_data: Dict) -> int:
    return mc_data.get("paths", np.array([])).shape[0] if "paths" in mc_data else 0
