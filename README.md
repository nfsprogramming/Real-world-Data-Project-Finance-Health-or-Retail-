# Retail Sales Analysis & Prediction

A Python data science project demonstrating end-to-end retail sales analysis with **target leakage prevention**, **time-aware evaluation**, and **honest performance reporting** using the Sample Superstore dataset.

## Overview

This project performs two tasks:

1. **Business Analysis** — Aggregates sales by category, region, customer segment, and time period to surface actionable retail insights.
2. **Sales Estimation** — Trains a Random Forest regressor using only features available at transaction time (Quantity, Discount, temporal features) to estimate per-transaction sales, with results compared against a mean-prediction baseline.

The project prioritises methodological correctness over headline metrics: Profit is excluded from features to prevent circular target leakage, and train/test splitting is chronological rather than random to prevent future-data contamination.

## Key Features

| Feature | Detail |
|---|---|
| **Leakage prevention** | `Profit` excluded — it is derived from `Sales` and cannot be known before prediction |
| **Time-aware split** | Chronological 80/20 split (Jan 2023 – Aug 2024 train; Aug–Dec 2024 test) |
| **Baseline comparison** | Mean-prediction `DummyRegressor` establishes the minimum performance bar |
| **Honest reporting** | Model underperforms baseline (R² -0.19); documented and explained, not hidden |
| **Business insights** | Category, region, segment, and discount-impact analysis |
| **Visualisations** | 5 publication-ready charts saved to `outputs/` |
| **Reproducibility** | Fixed `random_state=42`, `pathlib` paths, logged every step |

## Results (Verified — Run 2026-11-05)

### Business Insights

| Metric | Value |
|---|---|
| Top category by revenue | Office Supplies — \$861,319.80 (34.4 %) |
| Top region by revenue | Central — \$646,970.20 (25.8 %) |
| Best sales period | July 2024 — \$130,337.02 |
| Total profit | \$440,037.05 |
| Overall profit margin | 17.73 % |
| Discount–Sales correlation | −0.039 (negligible) |

### Model Performance (Held-out test set, 200 transactions)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean Baseline | \$1,264.53 | \$1,457.67 | −0.0020 |
| Random Forest | \$1,321.81 | \$1,585.47 | −0.1854 |

**Interpretation:** The available features (Quantity, Discount, Year, Month, DayOfWeek, Quarter) do not carry sufficient predictive signal for transaction-level sales. This is an honest, realistic result that demonstrates why leakage prevention and time-aware evaluation matter: without them, apparent performance would be artificially inflated.

## Tech Stack

- **Python** 3.10+
- **pandas** ≥ 2.0 — data loading, cleaning, aggregation
- **NumPy** ≥ 1.24 — numerical operations
- **scikit-learn** ≥ 1.3 — `RandomForestRegressor`, `DummyRegressor`, metrics
- **Matplotlib** ≥ 3.7 — chart rendering
- **Seaborn** ≥ 0.12 — chart styling
- **Joblib** ≥ 1.3 — model serialisation

## Prerequisites

- Python 3.10 or later
- pip

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nfsprogramming/Real-world-Data-Project-Finance-Health-or-Retail-.git
cd Real-world-Data-Project-Finance-Health-or-Retail-

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python retail_analysis.py
```

The script runs end-to-end and prints structured logs to stdout. Expected duration: ~5 seconds.

**What it does, in order:**

1. Loads `data/SampleSuperstore.csv` and validates required columns
2. Runs a data quality audit (shape, missing values, duplicates, date range)
3. Cleans duplicates and engineers temporal features (Year, Month, DayOfWeek, Quarter)
4. Computes business insights (category, region, segment, profitability, discount correlation)
5. Saves 5 visualisation charts to `outputs/`
6. Builds a leakage-free feature matrix (excludes Profit)
7. Performs a chronological 80/20 train/test split
8. Trains a `DummyRegressor` baseline and a `RandomForestRegressor`
9. Evaluates both models (MAE, RMSE, R²) and logs comparison
10. Saves the trained model to `models/retail_sales_model.joblib`
11. Saves `outputs/model_metrics.csv` and `outputs/business_insights.csv`

## Project Structure

```
retail_sales_project/
├── data/
│   └── SampleSuperstore.csv      # Input dataset (1,000 rows, 8 columns)
├── models/
│   └── retail_sales_model.joblib # Serialised Random Forest model
├── outputs/
│   ├── monthly_sales_trend.png   # Chronological monthly sales line chart
│   ├── business_analysis.png     # 4-panel business dashboard
│   ├── sales_by_category.png     # Bar chart — sales by category
│   ├── profit_by_region.png      # Bar chart — profit by region
│   ├── actual_vs_predicted.png   # Scatter — actual vs predicted sales
│   ├── model_metrics.csv         # MAE, RMSE, R² for both models
│   └── business_insights.csv     # Key business insight values
├── retail_analysis.py            # Main analysis script (entry point)
├── Project_Report.md             # Detailed methodology and findings report
├── requirements.txt              # Python dependency list
├── LICENSE                       # MIT License
└── .gitignore
```

## Visualisations

**Monthly Sales Trend** (`outputs/monthly_sales_trend.png`)  
Chronological year-month aggregation showing the sales trajectory from Jan 2023 to Dec 2024.

**Business Analysis Dashboard** (`outputs/business_analysis.png`)  
Four-panel chart: Sales by Category, Profit by Region, Discount vs Sales, Quantity vs Sales.

**Actual vs Predicted Sales** (`outputs/actual_vs_predicted.png`)  
Scatter plot of model predictions against ground truth on the held-out test period, with a perfect-prediction reference line.

## Dataset

**File:** `data/SampleSuperstore.csv`

| Column | Type | Description |
|---|---|---|
| Order Date | date | Transaction date (2023-01-02 to 2024-12-30) |
| Category | string | Product category (Technology, Furniture, Office Supplies) |
| Region | string | Sales region (East, West, South, Central) |
| Customer Segment | string | Customer type (Consumer, Corporate, Home Office) |
| Quantity | int | Units ordered |
| Discount | float | Discount fraction (0.0–1.0) |
| Sales | float | **Prediction target** — transaction revenue in USD |
| Profit | float | Derived metric — excluded from model features |

**Size:** 1,000 rows · 8 columns · 0 missing values · 0 duplicates

## Limitations

- **Feature signal:** Quantity, Discount, and temporal features alone are weak predictors of per-transaction sales; categorical features and external variables would likely improve performance.
- **Dataset size:** 1,000 transactions over ~2 years limits generalisability.
- **No hyperparameter tuning:** Default Random Forest parameters used.
- **Sales estimation, not forecasting:** The model predicts transaction-level sales given known order features; it does not forecast future sales periods.
- **No cross-validation:** A single chronological split is used; rolling-window CV would give a more robust estimate.

## Future Improvements

- Encode categorical features (Category, Region, Customer Segment) with `ColumnTransformer` + `OneHotEncoder` (the code already imports these)
- Add lag features and rolling-window statistics for the target variable
- Implement time-series cross-validation (e.g., `sklearn.model_selection.TimeSeriesSplit`)
- Hyperparameter tuning with `RandomizedSearchCV`
- Try gradient boosting models (XGBoost, LightGBM)
- Add SHAP explainability for feature importance

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Commit your changes: `git commit -m "Add my change"`
4. Push and open a pull request

Please keep all claims traceable to the actual data or code.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Data Science Portfolio Project — nfsprogramming*
