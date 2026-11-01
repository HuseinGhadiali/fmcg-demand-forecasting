import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecasting as F  # noqa: E402
from generate_data import generate  # noqa: E402


@pytest.fixture(scope="module")
def data():
    d = generate(seed=2)
    s = d["weekly_sales"].copy()
    s["week"] = pd.to_datetime(s.week)
    cal = d["festival_calendar"].copy()
    cal["week"] = pd.to_datetime(cal.week)
    return s, d["sku_master"], cal


def test_festival_calendar_marks_diwali_week(data):
    _, _, cal = data
    c = cal.set_index("week")
    assert c.loc["2025-10-20", "diwali_weight"] == 1.0      # Diwali 2025 = Mon 20 Oct
    assert c.loc["2024-10-28", "diwali_weight"] == 1.0      # Diwali 2024 = Fri 1 Nov, week starts 28 Oct
    assert c.loc["2025-10-27", "diwali_weight"] == 0.0
    assert cal.week.max() >= pd.Timestamp("2026-03-23")     # calendar extends into the forecast horizon


def test_censored_demand_only_raises_stockout_weeks(data):
    s, _, _ = data
    c = F.correct_censored_demand(s)
    assert (c.demand >= c.units_sold).all()
    assert (c.loc[c.stockout_flag == 0, "demand_correction"] == 0).all()
    assert c.loc[c.stockout_flag == 1, "demand_correction"].sum() > 0


def test_abc_xyz_classes(data):
    s, sku, _ = data
    cl = F.abc_xyz(F.correct_censored_demand(s), sku)
    assert set(cl.abc) <= {"A", "B", "C"} and set(cl.xyz) <= {"X", "Y", "Z"}
    a_share = cl[cl.abc == "A"].value_inr.sum() / cl.value_inr.sum()
    assert 0.6 < a_share <= 0.81
    assert (cl[cl.sku.str.startswith("FST")].xyz == "Z").all()       # festive SKUs are erratic


def test_metrics():
    y, f = pd.Series([10.0, 20, 30]), pd.Series([12.0, 18, 30])
    assert F.wape(y, f) == pytest.approx(4 / 60)
    assert F.bias(y, f) == pytest.approx(0)


def test_backtest_small(data):
    s, sku, cal = data
    df = F.to_long(F.correct_censored_demand(s), cal)
    df = df[df.unique_id.isin(["MUM|BEV01", "PUN|SNK02", "NGP|FST01"])]
    cv = F.backtest(df, n_windows=1)
    assert set(F.MODELS) <= set(cv.columns)
    assert (cv[F.MODELS] >= 0).all().all()
    assert cv.groupby("unique_id").size().eq(F.H).all()
    pol_lo = F.safety_stock_policy(cv, "LightGBM", sku, 0.90)
    pol_hi = F.safety_stock_policy(cv, "LightGBM", sku, 0.99)
    assert (pol_hi.safety_stock_units >= pol_lo.safety_stock_units).all()
    sim = F.simulate_inventory(cv, "LightGBM", sku, 0.95)
    assert 0 < sim["fill_rate_pct"] <= 100


def test_no_future_leakage_in_cutoffs(data):
    s, _, cal = data
    df = F.to_long(F.correct_censored_demand(s), cal)
    df = df[df.unique_id == "MUM|BEV01"]
    cv = F.backtest(df, n_windows=2)
    assert (cv.ds > cv.cutoff).all()
    assert np.isclose(cv.y.to_numpy(), df.set_index("ds").loc[cv.ds, "y"].to_numpy()).all()
