"""Synthetic weekly sell-in for an FMCG distributor in Maharashtra: 3 warehouses x 40 SKUs, 2022-2025.

Fully synthetic. Built-in features that a forecaster has to handle:
* yearly seasonality that differs by category (summer beverages / ice cream, monsoon tea & noodles, winter ghee)
* MOVING festivals: Diwali falls on 24-Oct-22, 12-Nov-23, 1-Nov-24, 20-Oct-25, so "same week last year" is wrong
* promotions with uplift and a post-promotion dip (customers stocked up)
* stock-outs that CENSOR demand (sales recorded < true demand), flagged so they can be cleaned
* intermittent slow movers (festive gift packs, diyas) that are zero most weeks
* a price increase in Apr-2024 and a competitor launch that permanently lowers one SKU's level

Usage: python src/generate_data.py --out data/raw --seed 31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2022-01-03", "2025-12-29"       # Mondays
FESTIVALS = {
    "diwali": ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"],
    "holi": ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04"],
    "ganesh": ["2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14"],
    "eid": ["2022-05-03", "2023-04-22", "2024-04-11", "2025-03-31", "2026-03-20"],
}
# monthly seasonal profiles (Jan..Dec)
PROFILES = {
    "summer":  [0.75, 0.85, 1.05, 1.35, 1.50, 1.30, 0.95, 0.90, 0.90, 0.85, 0.80, 0.80],
    "monsoon": [1.00, 0.95, 0.90, 0.85, 0.85, 1.10, 1.30, 1.30, 1.15, 1.00, 0.95, 1.05],
    "winter":  [1.25, 1.15, 0.95, 0.85, 0.80, 0.80, 0.85, 0.90, 0.95, 1.10, 1.20, 1.30],
    "flat":    [1.00] * 12,
    "mild":    [0.95, 0.95, 1.00, 1.05, 1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 0.98, 1.02],
}
SKUS = [  # sku, name, category, profile, base weekly units, diwali sensitivity, intermittent
    ("BEV01", "Cola 2L", "Beverages", "summer", 900, 1.2, 0), ("BEV02", "Mango Drink 1.2L", "Beverages", "summer", 650, 1.1, 0),
    ("BEV03", "Packaged Water 1L x12", "Beverages", "summer", 1400, 1.0, 0), ("BEV04", "Energy Drink 250ml", "Beverages", "mild", 300, 1.1, 0),
    ("BEV05", "Lemon Soda 600ml", "Beverages", "summer", 420, 1.1, 0),
    ("ICE01", "Vanilla Ice Cream 700ml", "Ice Cream", "summer", 260, 1.1, 0), ("ICE02", "Kulfi Pack x6", "Ice Cream", "summer", 180, 1.3, 0),
    ("TEA01", "Premium Tea 500g", "Tea & Coffee", "monsoon", 520, 1.3, 0), ("TEA02", "Instant Coffee 100g", "Tea & Coffee", "winter", 210, 1.2, 0),
    ("TEA03", "Masala Chai Premix", "Tea & Coffee", "monsoon", 160, 1.1, 0),
    ("NDL01", "Instant Noodles 4-pack", "Instant Food", "monsoon", 1100, 1.0, 0), ("NDL02", "Cup Noodles", "Instant Food", "monsoon", 380, 1.0, 0),
    ("SNK01", "Salted Chips 52g x10", "Snacks", "mild", 760, 1.5, 0), ("SNK02", "Aloo Bhujia 400g", "Snacks", "mild", 540, 1.9, 0),
    ("SNK03", "Cream Biscuits x12", "Snacks", "flat", 980, 1.4, 0), ("SNK04", "Glucose Biscuits x24", "Snacks", "flat", 1250, 1.2, 0),
    ("SNK05", "Roasted Makhana 80g", "Snacks", "mild", 140, 1.6, 0),
    ("STP01", "Whole Wheat Atta 10kg", "Staples", "flat", 620, 1.3, 0), ("STP02", "Basmati Rice 5kg", "Staples", "flat", 430, 1.6, 0),
    ("STP03", "Sunflower Oil 5L", "Staples", "flat", 510, 1.7, 0), ("STP04", "Sugar 5kg", "Staples", "flat", 470, 1.9, 0),
    ("STP05", "Toor Dal 1kg", "Staples", "flat", 690, 1.3, 0), ("STP06", "Besan 1kg", "Staples", "flat", 240, 2.2, 0),
    ("DRY01", "Cow Ghee 1L", "Dairy", "winter", 330, 2.4, 0), ("DRY02", "Milk Powder 1kg", "Dairy", "winter", 190, 1.2, 0),
    ("PC01", "Bath Soap 4-pack", "Personal Care", "flat", 870, 1.3, 0), ("PC02", "Shampoo 340ml", "Personal Care", "flat", 360, 1.2, 0),
    ("PC03", "Toothpaste 200g", "Personal Care", "flat", 940, 1.1, 0), ("PC04", "Sunscreen SPF50", "Personal Care", "summer", 120, 1.0, 0),
    ("PC05", "Body Lotion 400ml", "Personal Care", "winter", 150, 1.2, 0),
    ("HC01", "Detergent Powder 4kg", "Home Care", "flat", 560, 1.6, 0), ("HC02", "Dishwash Liquid 750ml", "Home Care", "flat", 640, 1.3, 0),
    ("HC03", "Floor Cleaner 2L", "Home Care", "flat", 280, 2.0, 0), ("HC04", "Mosquito Repellent Refill", "Home Care", "monsoon", 450, 1.0, 0),
    ("FST01", "Kaju Katli Gift Box", "Festive", "flat", 6, 30.0, 1), ("FST02", "Dry Fruit Hamper", "Festive", "flat", 5, 35.0, 1),
    ("FST03", "Clay Diya Set", "Festive", "flat", 3, 60.0, 1), ("FST04", "Rangoli Colours", "Festive", "flat", 3, 45.0, 1),
    ("FST05", "Puja Thali Kit", "Festive", "flat", 8, 12.0, 1), ("FST06", "Holi Colour Pack", "Festive", "flat", 2, 0.0, 1),
]
WAREHOUSES = {"MUM": 1.0, "PUN": 0.62, "NGP": 0.41}


def festival_weight(weeks: pd.DatetimeIndex, dates: list[str], lead_weeks: int = 2) -> np.ndarray:
    """1.0 in the festival week, decaying over the `lead_weeks` before it (people buy ahead)."""
    w = np.zeros(len(weeks))
    for d in pd.to_datetime(dates):
        wk = (d - pd.Timedelta(days=int(d.dayofweek))).normalize()
        for k in range(lead_weeks + 1):
            w[weeks.to_numpy() == np.datetime64(wk - pd.Timedelta(weeks=k), "ns")] = max(0.0, 1 - 0.3 * k)
    return w


def generate(seed: int = 31) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(START, END, freq="W-MON")
    n = len(weeks)
    month = weeks.month.to_numpy() - 1
    diwali = festival_weight(weeks, FESTIVALS["diwali"])
    holi = festival_weight(weeks, FESTIVALS["holi"], 1)
    ganesh = festival_weight(weeks, FESTIVALS["ganesh"], 1)
    eid = festival_weight(weeks, FESTIVALS["eid"], 1)
    t = np.arange(n) / 52
    price_up = weeks >= "2024-04-01"
    rows = []
    for sku, _name, cat, prof, base, dsens, intermittent in SKUS:
        season = np.array(PROFILES[prof])[month]
        # smooth the monthly step profile so it does not jump at month boundaries
        season = pd.Series(season).rolling(5, center=True, min_periods=1).mean().to_numpy()
        growth = rng.uniform(0.02, 0.12)
        for wh, scale in WAREHOUSES.items():
            level = base * scale * rng.lognormal(0, 0.08)
            mu = level * (1 + growth) ** t * season
            mu *= 1 + (dsens - 1) * diwali if dsens >= 1 else 1
            if sku == "FST06":
                mu = mu * (1 + 80 * holi)
            if cat in ("Snacks", "Staples", "Dairy"):
                mu *= 1 + 0.25 * ganesh + 0.2 * eid
            mu *= np.where(price_up, 0.94, 1.0)                    # price increase dents volume
            if sku == "SNK01":                                     # competitor launch: permanent level drop
                mu *= np.where(weeks >= "2024-09-02", 0.72, 1.0)
            promo = np.zeros(n, dtype=int)
            if not intermittent:
                k = rng.choice(np.arange(4, n - 2), size=int(n * 0.07), replace=False)
                promo[k] = 1
                uplift = rng.uniform(1.3, 1.8)
                mu = mu * np.where(promo == 1, uplift, 1.0) * np.where(np.roll(promo, 1) == 1, 0.85, 1.0)
            # gamma-Poisson noise (over-dispersed counts)
            shape = 2.0 if intermittent else 25.0
            demand = rng.poisson(rng.gamma(shape, mu / shape))
            # stock-outs censor sales: in ~3% of weeks only part of demand could be served
            oos = (rng.random(n) < (0.0 if intermittent else 0.03))
            fill = np.where(oos, rng.uniform(0.2, 0.7, n), 1.0)
            sales = np.floor(demand * fill).astype(int)
            rows.append(pd.DataFrame({"week": weeks, "warehouse": wh, "sku": sku, "units_sold": sales,
                                      "stockout_flag": oos.astype(int), "on_promotion": promo}))
    sales = pd.concat(rows, ignore_index=True)
    sku_master = pd.DataFrame([(s, nm, c, p, it) for s, nm, c, p, _, _, it in SKUS],
                              columns=["sku", "product_name", "category", "seasonality", "intermittent"])
    cost = rng.uniform(40, 650, len(SKUS)).round(0)
    sku_master["unit_cost_inr"] = cost
    sku_master["lead_time_weeks"] = np.where(sku_master.category.isin(["Festive", "Staples"]), 3, 2)
    # the festival calendar is known in advance, so it extends one quarter beyond the sales history
    cal_weeks = pd.date_range(START, "2026-03-30", freq="W-MON")
    cal = pd.DataFrame({"week": cal_weeks, **{f"{k}_weight": festival_weight(cal_weeks, v, 2 if k == "diwali" else 1)
                                              for k, v in FESTIVALS.items()}})
    return {"weekly_sales": sales, "sku_master": sku_master, "festival_calendar": cal}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in generate(a.seed).items():
        v.to_csv(out / f"{k}.csv", index=False)
        print(f"{k:<18}{len(v):>8,} rows")


if __name__ == "__main__":
    main()
