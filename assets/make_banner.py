"""Render assets/banner.png: every SKU's normalised weekly demand as a glowing ribbon (python assets/make_banner.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from banner import glow_line, make_banner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "data" / "raw" / "weekly_sales.csv", parse_dates=["week"])
s = s[(s.warehouse == "MUM") & ~s.sku.str.startswith("FST")]
keep = ["BEV01", "BEV03", "ICE01", "TEA01", "NDL01", "SNK02", "SNK04", "STP03", "STP04", "DRY01", "PC04", "HC04"]
wide = s[s.sku.isin(keep)].pivot(index="week", columns="sku", values="units_sold").rolling(4, center=True, min_periods=1).mean()
wide = (wide - wide.mean()) / wide.std()


def art(ax, acc):
    x = np.arange(len(wide))
    for i, c in enumerate(wide.columns):
        y = wide[c].to_numpy() * 0.32 + i * 0.62
        glow_line(ax, x, y, acc[i % len(acc)], lw=1.4, layers=4, alpha=0.8)
    for d in ["2022-10-24", "2023-11-13", "2024-10-28", "2025-10-20"]:
        k = wide.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        ax.axvline(k, color="white", alpha=0.18, lw=1)
        ax.text(k, -1.4, "Diwali", color="white", alpha=0.6, fontsize=8.5, ha="center")
    ax.set_xlim(-2, len(x) + 2)
    ax.set_ylim(-1.8, len(wide.columns) * 0.62 + 0.6)


make_banner(Path(__file__).parent / "banner.png", "Demand Forecasting",
            "120 weekly series · 8 forecasters · backtests · inventory policy",
            ["statsforecast", "LightGBM / mlforecast", "Backtesting", "Inventory sim"], art, theme="ember",
            emoji_free_kicker="Time series · FMCG supply chain")
