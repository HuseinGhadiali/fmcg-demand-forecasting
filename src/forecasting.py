"""Demand forecasting pipeline: cleaning censored demand, ABC-XYZ, statistical + ML models, backtesting, inventory policy."""
from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import GlobalSklearnTransformer
from scipy.stats import norm
from sklearn.preprocessing import FunctionTransformer
from statsforecast import StatsForecast
from statsforecast.models import MSTL, TSB, AutoETS, AutoTheta, CrostonOptimized, SeasonalNaive

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
H = 13                 # forecast horizon: one quarter of weeks
N_WINDOWS = 4          # rolling-origin backtest windows (covers all of 2025)
SEASON = 52
EXOG = ["on_promotion", "diwali_weight", "holi_weight", "ganesh_weight", "eid_weight"]


# ------------------------------------------------------------------ data
def load(raw: Path = ROOT / "data" / "raw"):
    s = pd.read_csv(raw / "weekly_sales.csv", parse_dates=["week"])
    sku = pd.read_csv(raw / "sku_master.csv")
    cal = pd.read_csv(raw / "festival_calendar.csv", parse_dates=["week"])
    return s, sku, cal


def correct_censored_demand(s: pd.DataFrame) -> pd.DataFrame:
    """Sales in stock-out weeks understate demand. Replace them with the median of the surrounding
    in-stock, non-promo weeks (±3) so models learn demand, not supply problems."""
    s = s.sort_values(["warehouse", "sku", "week"]).copy()
    s["demand"] = s.units_sold.astype(float)
    clean = s.units_sold.where((s.stockout_flag == 0) & (s.on_promotion == 0))
    ref = clean.groupby([s.warehouse, s.sku]).transform(lambda x: x.rolling(7, center=True, min_periods=2).median())
    fix = (s.stockout_flag == 1) & ref.notna()
    s.loc[fix, "demand"] = np.maximum(ref[fix], s.units_sold[fix])
    s["demand_correction"] = s.demand - s.units_sold
    return s


def to_long(s: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    df = s.merge(cal, on="week")
    df["unique_id"] = df.warehouse + "|" + df.sku
    return df.rename(columns={"week": "ds", "demand": "y"})[["unique_id", "ds", "y", *EXOG]].sort_values(["unique_id", "ds"])


# ------------------------------------------------------------------ ABC-XYZ
def abc_xyz(s: pd.DataFrame, sku: pd.DataFrame, last_weeks: int = 52) -> pd.DataFrame:
    recent = s[s.week > s.week.max() - pd.Timedelta(weeks=last_weeks)]
    g = recent.groupby(["warehouse", "sku"]).agg(units=("demand", "sum"), mean=("demand", "mean"), std=("demand", "std"),
                                                 zero_weeks=("demand", lambda x: (x == 0).mean()))
    g = g.join(sku.set_index("sku")[["unit_cost_inr", "category"]], on="sku")
    g["value_inr"] = g.units * g.unit_cost_inr
    g = g.sort_values("value_inr", ascending=False)
    share = g.value_inr.cumsum() / g.value_inr.sum()
    g["abc"] = np.select([share <= 0.80, share <= 0.95], ["A", "B"], "C")
    g["cv"] = g["std"] / g["mean"].replace(0, np.nan)
    g["xyz"] = np.select([g.cv <= 0.25, g.cv <= 0.5], ["X", "Y"], "Z")   # weekly CV thresholds
    g["class"] = g.abc + g.xyz
    g = g.reset_index()
    g["unique_id"] = g.warehouse + "|" + g.sku
    return g


# ------------------------------------------------------------------ models
def stat_models():
    return [SeasonalNaive(season_length=SEASON, alias="SeasonalNaive"),
            AutoETS(season_length=SEASON, alias="AutoETS"),
            AutoTheta(season_length=SEASON, alias="AutoTheta"),
            MSTL(season_length=SEASON, trend_forecaster=AutoETS(model="ZZN"), alias="MSTL"),
            CrostonOptimized(alias="Croston"), TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB")]


def ml_model() -> MLForecast:
    log = GlobalSklearnTransformer(FunctionTransformer(func=np.log1p, inverse_func=np.expm1))
    return MLForecast(
        models={"LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)},
        freq="W-MON",
        lags=[1, 2, 3, 4, 13, 52],
        lag_transforms={1: [RollingMean(4), RollingMean(13), RollingStd(13)], 52: [RollingMean(3)]},
        date_features=["week", "month"],
        target_transforms=[log],
    )


def backtest(df: pd.DataFrame, n_windows: int = N_WINDOWS, h: int = H) -> pd.DataFrame:
    """Rolling-origin cross-validation for all models; returns one row per series x cutoff x week."""
    sf = StatsForecast(models=stat_models(), freq="W-MON", n_jobs=-1)
    cv_stat = sf.cross_validation(df=df[["unique_id", "ds", "y"]], h=h, n_windows=n_windows, step_size=h)
    mlf = ml_model()
    cv_ml = mlf.cross_validation(df=df, h=h, n_windows=n_windows, step_size=h, static_features=[])
    cv = cv_stat.merge(cv_ml.drop(columns="y"), on=["unique_id", "ds", "cutoff"])
    models = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    cv[models] = cv[models].clip(lower=0)
    # simple ensemble of the two strongest families
    cv["Ensemble"] = cv[["LightGBM", "AutoETS", "MSTL"]].median(axis=1)
    return cv


MODELS = ["SeasonalNaive", "AutoETS", "AutoTheta", "MSTL", "Croston", "TSB", "LightGBM", "Ensemble"]


def model_columns(cv: pd.DataFrame) -> list[str]:
    return [m for m in MODELS if m in cv.columns]


# ------------------------------------------------------------------ metrics
def wape(y, f) -> float:
    return float(np.abs(y - f).sum() / np.abs(y).sum())


def bias(y, f) -> float:
    return float((f - y).sum() / y.sum())


def scores(cv: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    models = model_columns(cv)
    groups = [("All", cv)] if by is None else list(cv.groupby(by))
    rows = []
    for g, d in groups:
        for m in models:
            rows.append({"group": g, "model": m, "wape_pct": 100 * wape(d.y, d[m]), "bias_pct": 100 * bias(d.y, d[m])})
    return pd.DataFrame(rows).round(1)


def festival_scores(cv: pd.DataFrame, cal: pd.DataFrame, festival: str = "diwali") -> pd.DataFrame:
    """Accuracy in festival weeks only: this is where 'same week last year' breaks, because festival dates move."""
    d = cv.merge(cal.rename(columns={"week": "ds"})[["ds", f"{festival}_weight"]], on="ds")
    return scores(d[d[f"{festival}_weight"] > 0])


def best_model_per_class(cv: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Model selection by ABC-XYZ class: pick the lowest-WAPE model for each class (on backtest)."""
    d = cv.merge(classes[["unique_id", "class"]], on="unique_id")
    sc = scores(d, by="class")
    best = sc.loc[sc.groupby("group").wape_pct.idxmin()].set_index("group")
    sn = sc[sc.model == "SeasonalNaive"].set_index("group").wape_pct
    best["seasonal_naive_wape_pct"] = sn
    best["series"] = classes.groupby("class").size()
    return best.rename_axis("class")


# ------------------------------------------------------------------ inventory
def safety_stock_policy(cv: pd.DataFrame, model: str, sku: pd.DataFrame, service_level: float = 0.95) -> pd.DataFrame:
    """Safety stock = z * sigma(forecast error) * sqrt(lead time); reorder point = forecast over lead time + SS."""
    z = norm.ppf(service_level)
    err = cv.assign(e=cv.y - cv[model]).groupby("unique_id").e.std().rename("sigma_weekly_error")
    p = err.to_frame()
    p["sku"] = p.index.str.split("|").str[1]
    p = p.join(sku.set_index("sku")[["lead_time_weeks", "unit_cost_inr"]], on="sku")
    p["safety_stock_units"] = np.ceil(z * p.sigma_weekly_error * np.sqrt(p.lead_time_weeks))
    p["safety_stock_value_inr"] = p.safety_stock_units * p.unit_cost_inr
    return p


def simulate_inventory(cv: pd.DataFrame, model: str, sku: pd.DataFrame, service_level: float = 0.95,
                       exclude_prefix: str | None = "FST") -> dict:
    """Weekly order-up-to simulation over the backtest period.
    Each week: receive the order placed `lead_time` weeks ago, serve demand, then order up to
    (forecast over lead time + 1 review week) + safety stock.
    Festive gift SKUs are excluded by default: they are planned as one-off events, not replenished weekly.
    Returns the unit-weighted fill rate and the average inventory value held."""
    if exclude_prefix:
        cv = cv[~cv.unique_id.str.split("|").str[1].str.startswith(exclude_prefix)]
    pol = safety_stock_policy(cv, model, sku, service_level)
    served_all = demand_all = stock_value = 0.0
    for uid, d in cv.sort_values("ds").groupby("unique_id"):
        d = d.drop_duplicates("ds")
        L = int(pol.loc[uid, "lead_time_weeks"])
        ss = pol.loc[uid, "safety_stock_units"]
        f, y = d[model].to_numpy(), d.y.to_numpy()
        stock = f[: L + 1].sum() + ss
        pipeline = [0.0] * L
        held = 0.0
        for t in range(len(y)):
            stock += pipeline.pop(0)
            sale = min(stock, y[t])
            served_all += sale
            stock -= sale
            held += stock
            ahead = f[t + 1: t + 2 + L]
            target = (ahead.mean() if len(ahead) else f[t]) * (L + 1) + ss
            pipeline.append(max(0.0, target - stock - sum(pipeline)))
        demand_all += y.sum()
        stock_value += held / len(y) * pol.loc[uid, "unit_cost_inr"]
    return {"model": model, "service_level_target": service_level,
            "fill_rate_pct": round(float(100 * served_all / demand_all), 2),
            "avg_inventory_value_lakh": round(float(stock_value / 1e5), 1)}


def inventory_at_matched_fill(cv, sku, models, target_fill: float = 98.0,
                              grid=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)) -> pd.DataFrame:
    """For each model, the least inventory that achieves `target_fill` % (fair comparison of forecasts)."""
    rows = []
    for m in models:
        sims = [simulate_inventory(cv, m, sku, sl) for sl in grid]
        ok = [x for x in sims if x["fill_rate_pct"] >= target_fill]
        best = min(ok, key=lambda x: x["avg_inventory_value_lakh"]) if ok else max(sims, key=lambda x: x["fill_rate_pct"])
        rows.append(best)
    return pd.DataFrame(rows).set_index("model")


# ------------------------------------------------------------------ final forecast
def forecast_next_quarter(df: pd.DataFrame, cal_future: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """Fit on all history and forecast the next quarter with 80/95% conformal intervals (LightGBM)."""
    mlf = ml_model()
    mlf.fit(df, static_features=[], prediction_intervals=__import__("mlforecast.utils", fromlist=["PredictionIntervals"])
            .PredictionIntervals(n_windows=3, h=h))
    future = mlf.make_future_dataframe(h)
    future = future.merge(cal_future, on="ds", how="left").fillna(0)
    future["on_promotion"] = 0
    fc = mlf.predict(h, X_df=future, level=[80, 95])
    num = fc.columns.drop(["unique_id", "ds"])
    fc[num] = fc[num].clip(lower=0)
    return fc
