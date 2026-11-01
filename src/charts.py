"""Charts for the README / notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecasting import ROOT
from viz_style import BLUES, MUTED, SERIES, TEXT, TEXT_2, apply_style, finish

IMG = ROOT / "images"
apply_style()


def diwali_shift(cv, cal, uid="PUN|SNK02", path=IMG / "diwali_shift.png"):
    d = cv[(cv.unique_id == uid) & (cv.ds >= "2025-08-01") & (cv.ds <= "2025-12-31")].drop_duplicates("ds")
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    ax.plot(d.ds, d.y, color=TEXT, lw=2.2, label="Actual demand")
    ax.plot(d.ds, d.SeasonalNaive, color=SERIES[1], lw=1.8, label="Seasonal naive (same week last year)")
    ax.plot(d.ds, d.LightGBM, color=SERIES[0], lw=1.8, label="LightGBM with festival calendar")
    ax.axvline(pd.Timestamp("2025-10-20"), color=TEXT_2, lw=1)
    ax.text(pd.Timestamp("2025-10-17"), d.y.max() * 1.02, "Diwali 2025 (20 Oct)", ha="right", fontsize=8.5, color=TEXT_2)
    ax.axvline(pd.Timestamp("2025-10-27"), color=SERIES[1], lw=1, alpha=0.6)
    ax.text(pd.Timestamp("2025-10-30"), d.y.max() * 1.02, "52 weeks after Diwali 2024 (1 Nov)", fontsize=8.5, color=SERIES[1])
    ax.set_ylabel("Units per week")
    ax.set_ylim(0, d.y.max() * 1.12)
    ax.legend(loc="upper left")
    finish(fig, ax, "Diwali moved 12 days earlier in 2025; 'same week last year' missed it",
           "Aloo Bhujia 400g, Pune warehouse (no promotions around either Diwali): backtest forecasts vs actual", path)


def wape_chart(overall, festive, path=IMG / "model_wape.png"):
    o = overall.set_index("model").wape_pct
    f = festive.set_index("model").wape_pct
    order = o.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(order))
    ax.hlines(y, o[order] / 100, f[order] / 100, color=MUTED, lw=2)
    ax.scatter(o[order] / 100, y, s=80, color=SERIES[0], label="All weeks", zorder=3)
    ax.scatter(f[order] / 100, y, s=80, color=SERIES[1], label="Diwali weeks only", zorder=3)
    for yi, m in enumerate(order):
        ax.text(max(o[m], f[m]) / 100 + 0.006, yi, f"{o[m]:.1f}%  /  {f[m]:.1f}%", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y, order)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0.15, max(f.max(), o.max()) / 100 + 0.07)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel("WAPE (lower is better), 4 × 13-week rolling backtests over 2025")
    ax.legend(loc="upper right")
    finish(fig, ax, "A global LightGBM beats every statistical model, most of all at Diwali",
           "Weighted absolute percentage error by model, 120 SKU-warehouse series", path)


def abc_xyz_chart(cl, path=IMG / "abc_xyz.png"):
    m = cl.pivot_table(index="abc", columns="xyz", values="value_inr", aggfunc="sum").reindex(index=list("ABC"), columns=list("XYZ"))
    n = cl.pivot_table(index="abc", columns="xyz", values="unique_id", aggfunc="count").reindex(index=list("ABC"), columns=list("XYZ"))
    share = m / m.sum().sum()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    from matplotlib.colors import LinearSegmentedColormap
    ax.imshow(share.fillna(0).to_numpy(), cmap=LinearSegmentedColormap.from_list("b", BLUES), vmin=0, vmax=0.5)
    ax.set_xticks(range(3), ["X  stable", "Y  variable", "Z  erratic"])
    ax.set_yticks(range(3), ["A  top 80% value", "B  next 15%", "C  last 5%"])
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            if not np.isnan(n.iloc[i, j]):
                v = share.iloc[i, j]
                ax.text(j, i, f"{int(n.iloc[i, j])} series\n{v:.0%} of value", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.25 else TEXT)
    finish(fig, ax, "ABC-XYZ: where forecasting effort pays", "Series by value (ABC) and weekly variability (XYZ), last 52 weeks", path)


def inventory_tradeoff(curves, path=IMG / "inventory_tradeoff.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for i, (m, c) in enumerate(curves.items()):
        ax.plot(c.avg_inventory_value_lakh, c.fill_rate_pct, marker="o", markersize=5, color=SERIES[i], label=m)
    ax.axhline(99.5, color=TEXT_2, lw=1)
    ax.text(ax.get_xlim()[0] + 2, 99.53, "99.5% fill target", fontsize=8.5, color=TEXT_2)
    ax.set_xlabel("Average inventory held (Rs lakh)")
    ax.set_ylabel("Fill rate (% of units served)")
    ax.legend(loc="lower right")
    finish(fig, ax, "Better forecasts = same service with ~23% less stock",
           "Inventory simulation over 2025 at different safety-stock service levels (regular SKUs)", path)


def next_quarter(df, fc, uid="MUM|BEV01", path=IMG / "next_quarter_forecast.png"):
    h = df[(df.unique_id == uid) & (df.ds >= "2024-07-01")]
    f = fc[fc.unique_id == uid]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(h.ds, h.y, color=TEXT, lw=1.8, label="History (demand)")
    ax.fill_between(f.ds, f["LightGBM-lo-95"], f["LightGBM-hi-95"], color=BLUES[0], lw=0, label="95% interval")
    ax.fill_between(f.ds, f["LightGBM-lo-80"], f["LightGBM-hi-80"], color=BLUES[2], lw=0, label="80% interval")
    ax.plot(f.ds, f.LightGBM, color=SERIES[0], lw=2.2, label="Forecast")
    ax.set_ylabel("Units per week")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncols=2)
    finish(fig, ax, "Q1-2026 forecast with conformal prediction intervals",
           "Cola 2L, Mumbai warehouse · no promotions planned · demand starts rising into summer", path)
