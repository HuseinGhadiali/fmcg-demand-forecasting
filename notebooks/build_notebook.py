import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Weekly Demand Forecasting & Inventory Policy for an FMCG Distributor\n"
       "3 warehouses (Mumbai, Pune, Nagpur) × 40 SKUs × 4 years of weekly sell-in · statistical vs machine-learning forecasters · "
       "rolling-origin backtests · safety stock & inventory simulation\n\n> Synthetic data: `src/generate_data.py` (seed 31)."),
    code("import sys, warnings\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')\n"
         "import pandas as pd\nfrom IPython.display import Image, display\nfrom forecasting import *\nimport charts\n"
         "sales, sku, cal = load()\nsales.head()"),
    md("## 1. Clean the target: censored demand\nIn stock-out weeks, *sales* understate *demand*. Training on raw sales would teach the model "
       "that supply failures are normal. Stock-out weeks are re-estimated from neighbouring in-stock, non-promo weeks."),
    code("sales = correct_censored_demand(sales)\nprint(f\"{(sales.demand_correction > 0).sum()} stock-out weeks corrected, "
         "{sales.demand_correction.sum():,.0f} units of hidden demand recovered "
         "({sales.demand_correction.sum() / sales.units_sold.sum():.1%} of sales)\")\ndf = to_long(sales, cal)\ndf.head()"),
    md("## 2. ABC-XYZ segmentation\nA = top 80% of inventory value, B = next 15%, C = last 5%. X/Y/Z = weekly coefficient of variation ≤0.25 / ≤0.5 / >0.5."),
    code("classes = abc_xyz(sales, sku)\ncharts.abc_xyz_chart(classes)\ndisplay(classes['class'].value_counts().to_frame())\n"
         "display(Image('../images/abc_xyz.png'))"),
    md("## 3. Backtest: 8 forecasters, 4 rolling origins × 13-week horizon\n"
       "| Family | Models |\n|---|---|\n| Baseline | Seasonal naive (same week last year) |\n"
       "| Statistical (statsforecast) | AutoETS, AutoTheta, MSTL, Croston, TSB (last two for intermittent demand) |\n"
       "| Machine learning (mlforecast) | One **global LightGBM** across all 120 series, with lags 1-4/13/52, rolling stats, calendar, "
       "promo flag and festival-calendar features, log target |\n| Ensemble | median(LightGBM, AutoETS, MSTL) |\n\n"
       "AutoARIMA with a 52-week season was tried and dropped: about 75 s per series in backtesting, impractical at 120 series."),
    code("cv = backtest(df)\noverall = scores(cv)\nfestive = festival_scores(cv, cal)\n"
         "display(overall.merge(festive, on=['group', 'model'], suffixes=('', '_diwali_weeks')).sort_values('wape_pct'))"),
    code("charts.wape_chart(overall, festive)\ncharts.diwali_shift(cv, cal)\ndisplay(Image('../images/model_wape.png'))\n"
         "display(Image('../images/diwali_shift.png'))"),
    md("### Best model by ABC-XYZ class"),
    code("best_model_per_class(cv, classes)"),
    md("### Intermittent festive SKUs\nGift boxes, diyas and rangoli sell almost only in Diwali week. No time-series model handles them well; "
       "they should be planned as **events** (pre-booked quantities from last year's festival sell-through), not forecast weekly."),
    code("d = cv.merge(classes[['unique_id', 'sku']])\nscores(d[d.sku.str.startswith('FST')]).sort_values('wape_pct')"),
    md("## 4. From forecast to inventory\nSafety stock = z × σ(forecast error) × √lead-time. A weekly order-up-to policy is simulated over 2025 for each model "
       "across service-level targets. The fair comparison is **how much stock each forecast needs to reach the same fill rate**."),
    code("grid = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)\n"
         "curves = {m: pd.DataFrame([simulate_inventory(cv, m, sku, sl) for sl in grid]) for m in ['SeasonalNaive', 'AutoETS', 'LightGBM']}\n"
         "charts.inventory_tradeoff(curves)\nmatched = inventory_at_matched_fill(cv, sku, ['SeasonalNaive', 'AutoETS', 'MSTL', 'LightGBM', 'Ensemble'], 99.5, grid)\n"
         "display(matched)\ndisplay(Image('../images/inventory_tradeoff.png'))"),
    code("saving = matched.loc['SeasonalNaive', 'avg_inventory_value_lakh'] - matched.loc['LightGBM', 'avg_inventory_value_lakh']\n"
         "print(f'Working capital released at the same 99.5% fill rate: Rs {saving:.0f} lakh '\n"
         "      f'({saving / matched.loc[\"SeasonalNaive\", \"avg_inventory_value_lakh\"]:.0%} less stock)')"),
    md("## 5. Next-quarter forecast (Q1 2026) with prediction intervals\nTrained on all history. 80/95% conformal intervals come from 3 backtest windows. "
       "The festival calendar (Holi 4-Mar, Eid 20-Mar-2026) is known in advance and fed as future features."),
    code("fc = forecast_next_quarter(df, cal.rename(columns={'week': 'ds'}))\nfc.to_csv('../reports/forecast_q1_2026.csv', index=False)\n"
         "charts.next_quarter(df, fc)\ndisplay(fc.head())\ndisplay(Image('../images/next_quarter_forecast.png'))"),
    code("policy = safety_stock_policy(cv, 'LightGBM', sku, 0.95)\npolicy.to_csv('../reports/safety_stock_policy.csv')\n"
         "policy.sort_values('safety_stock_value_inr', ascending=False).head(10)"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "demand_forecasting.ipynb")
