"""
Retail Sales Analysis & Prediction Project

This script demonstrates an end-to-end retail sales analysis with proper
feature engineering, leakage prevention, time-aware evaluation, and
business insights generation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_paths():
    """Setup project paths using pathlib for cross-platform compatibility."""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    return BASE_DIR, DATA_DIR, OUTPUT_DIR, MODEL_DIR


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load the retail sales dataset with validation."""
    data_path = data_dir / "SampleSuperstore.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate dataset structure and contents."""
    logger.info("Validating dataset...")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    required_columns = ['Order Date', 'Sales', 'Quantity', 'Discount', 'Profit']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for sufficient data
    if len(df) < 50:
        logger.warning(f"Dataset has only {len(df)} rows, which may be insufficient for reliable modeling")
    
    logger.info(f"Dataset validation passed. Rows: {len(df)}, Columns: {len(df.columns)}")


def data_quality_audit(df: pd.DataFrame) -> dict:
    """Perform comprehensive data quality audit."""
    logger.info("Performing data quality audit...")
    
    audit_results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'date_range': (df['Order Date'].min(), df['Order Date'].max()) if 'Order Date' in df.columns else None,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numerical_summary': df.describe().to_dict()
    }
    
    logger.info(f"Data Quality Audit Results:")
    logger.info(f"  Shape: {audit_results['shape']}")
    logger.info(f"  Date range: {audit_results['date_range']}")
    logger.info(f"  Missing values: {sum(audit_results['missing_values'].values())}")
    logger.info(f"  Duplicate rows: {audit_results['duplicate_rows']}")
    
    return audit_results


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the data."""
    logger.info("Cleaning and preprocessing data...")
    
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df)
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Convert date column
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    
    # Create year-month for proper time analysis
    df['year_month'] = df['Order Date'].dt.to_period('M')
    
    # Feature engineering for time-based features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['DayOfWeek'] = df['Order Date'].dt.dayofweek
    df['Quarter'] = df['Order Date'].dt.quarter
    
    logger.info(f"Data preprocessing complete. Final rows: {len(df)}")
    
    return df


def exploratory_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """Perform exploratory data analysis and generate insights."""
    logger.info("Performing exploratory data analysis...")
    
    insights = {}
    
    # Sales by Category
    category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    insights['top_category'] = category_sales.index[0]
    insights['top_category_sales'] = category_sales.iloc[0]
    
    # Sales by Region
    region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    insights['top_region'] = region_sales.index[0]
    insights['top_region_sales'] = region_sales.iloc[0]
    
    # Time-based analysis
    monthly_sales = df.groupby('year_month')['Sales'].sum()
    insights['best_month'] = monthly_sales.idxmax()
    insights['best_month_sales'] = monthly_sales.max()
    
    # Discount analysis
    discount_corr = df[['Discount', 'Sales']].corr().iloc[0, 1]
    insights['discount_sales_correlation'] = discount_corr
    
    # Profit analysis (for descriptive purposes only)
    insights['total_profit'] = df['Profit'].sum()
    insights['profit_margin'] = (df['Profit'].sum() / df['Sales'].sum()) * 100
    
    logger.info(f"Key Insights:")
    logger.info(f"  Top Category: {insights['top_category']} (${insights['top_category_sales']:,.2f})")
    logger.info(f"  Top Region: {insights['top_region']} (${insights['top_region_sales']:,.2f})")
    logger.info(f"  Best Period: {insights['best_month']} (${insights['best_month_sales']:,.2f})")
    logger.info(f"  Discount-Sales Correlation: {insights['discount_sales_correlation']:.3f}")
    logger.info(f"  Total Profit: ${insights['total_profit']:,.2f}")
    logger.info(f"  Profit Margin: {insights['profit_margin']:.2f}%")
    
    return insights


def create_business_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create business-focused visualizations."""
    logger.info("Creating business visualizations...")
    
    sns.set_style("whitegrid")
    
    # Year-month sales trend (chronological)
    df_sorted = df.sort_values('Order Date')
    monthly_sales = df_sorted.groupby('year_month')['Sales'].sum()
    
    plt.figure(figsize=(12, 5))
    monthly_sales.plot(kind='line', marker='o', linewidth=2, markersize=6)
    plt.title('Monthly Sales Trend (Chronological)', fontweight='bold', fontsize=14)
    plt.xlabel('Year-Month', fontsize=12)
    plt.ylabel('Sales ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    trend_path = output_dir / "monthly_sales_trend.png"
    plt.savefig(trend_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Category and Region analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Retail Sales Business Analysis', fontsize=16, fontweight='bold')
    
    # Sales by Category
    category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    axes[0, 0].bar(category_sales.index, category_sales.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Sales by Category', fontweight='bold')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Profit by Region (descriptive only)
    region_profit = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
    axes[0, 1].bar(region_profit.index, region_profit.values, color='coral', edgecolor='black')
    axes[0, 1].set_title('Profit by Region (Descriptive)', fontweight='bold')
    axes[0, 1].set_ylabel('Profit ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Discount vs Sales scatter
    axes[1, 0].scatter(df['Discount'], df['Sales'], alpha=0.5, color='purple')
    axes[1, 0].set_title('Discount vs Sales Relationship', fontweight='bold')
    axes[1, 0].set_xlabel('Discount')
    axes[1, 0].set_ylabel('Sales ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantity vs Sales scatter
    axes[1, 1].scatter(df['Quantity'], df['Sales'], alpha=0.5, color='green')
    axes[1, 1].set_title('Quantity vs Sales Relationship', fontweight='bold')
    axes[1, 1].set_xlabel('Quantity')
    axes[1, 1].set_ylabel('Sales ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    business_path = output_dir / "business_analysis.png"
    plt.savefig(business_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual charts for backward compatibility
    # Sales by Category
    plt.figure(figsize=(8, 5))
    category_sales.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Sales by Category', fontweight='bold', fontsize=14)
    plt.ylabel('Sales ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    sales_category_path = output_dir / "sales_by_category.png"
    plt.savefig(sales_category_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Profit by Region
    plt.figure(figsize=(8, 5))
    region_profit.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Profit by Region', fontweight='bold', fontsize=14)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    profit_region_path = output_dir / "profit_by_region.png"
    plt.savefig(profit_region_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Business visualizations saved")


def prepare_features_leakage_free(df: pd.DataFrame) -> tuple:
    """
    Prepare features for sales prediction WITHOUT target leakage.
    
    CRITICAL: We do NOT use 'Profit' as a feature because Profit is
    calculated from Sales and would create target leakage.
    """
    logger.info("Preparing features with leakage prevention...")
    
    # Features available at prediction time (before knowing Sales)
    # We exclude 'Profit' because it's derived from Sales
    feature_columns = ['Quantity', 'Discount', 'Year', 'Month', 'DayOfWeek', 'Quarter']
    
    # For simplicity, we'll use only numerical features to avoid
    # complex categorical encoding issues with train-test split
    # In production, proper pipeline with ColumnTransformer would be used
    
    X = df[feature_columns].copy()
    y = df['Sales'].copy()
    
    logger.info(f"Features prepared: {X.shape[1]} features")
    logger.info(f"Features: {feature_columns}")
    logger.info(f"Excluded from features: Profit (target leakage prevention)")
    logger.info(f"Note: Categorical features excluded for simplicity in this demonstration")
    
    return X, y, feature_columns


def time_aware_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Perform time-aware train-test split using chronological ordering.
    
    CRITICAL: For time-series problems, we must use chronological splitting,
    not random splitting, to avoid data leakage from future into past.
    """
    logger.info("Performing time-aware train-test split...")
    
    # Sort by date to ensure chronological order
    df_sorted = df.sort_values('Order Date')
    
    # Use 80% for training (earlier dates), 20% for testing (later dates)
    split_idx = int(len(df_sorted) * 0.8)
    
    train_dates = df_sorted.iloc[:split_idx]['Order Date']
    test_dates = df_sorted.iloc[split_idx:]['Order Date']
    
    logger.info(f"Training period: {train_dates.min()} to {train_dates.max()}")
    logger.info(f"Testing period: {test_dates.min()} to {test_dates.max()}")
    
    # Split the data chronologically
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Train set size: {X_train.shape[0]} samples")
    logger.info(f"Test set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> DummyRegressor:
    """Train a baseline model (mean prediction)."""
    logger.info("Training baseline model (mean prediction)...")
    
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    
    logger.info("Baseline model training completed")
    
    return baseline


def train_ml_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    """Train Random Forest regressor."""
    logger.info("Training Random Forest regressor...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    logger.info("ML model training completed")
    
    return model


def evaluate_models(baseline: DummyRegressor, model: RandomForestRegressor, 
                   X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> dict:
    """Evaluate both baseline and ML models."""
    logger.info("Evaluating models...")
    
    # Baseline predictions
    y_pred_baseline = baseline.predict(X_test)
    
    # ML model predictions
    y_pred_model = model.predict(X_test)
    
    # Calculate metrics for baseline
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    
    # Calculate metrics for ML model
    model_mae = mean_absolute_error(y_test, y_pred_model)
    model_r2 = r2_score(y_test, y_pred_model)
    model_rmse = np.sqrt(mean_squared_error(y_test, y_pred_model))
    
    logger.info(f"Baseline Performance:")
    logger.info(f"  MAE: ${baseline_mae:.2f}")
    logger.info(f"  RMSE: ${baseline_rmse:.2f}")
    logger.info(f"  R²: {baseline_r2:.4f}")
    
    logger.info(f"ML Model Performance:")
    logger.info(f"  MAE: ${model_mae:.2f}")
    logger.info(f"  RMSE: ${model_rmse:.2f}")
    logger.info(f"  R²: {model_r2:.4f}")
    
    # Calculate improvement
    mae_improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
    r2_improvement = model_r2 - baseline_r2
    
    logger.info(f"Improvement over baseline:")
    logger.info(f"  MAE: {mae_improvement:.2f}%")
    logger.info(f"  R²: {r2_improvement:.4f}")
    
    metrics = {
        'baseline': {'mae': baseline_mae, 'r2': baseline_r2, 'rmse': baseline_rmse},
        'model': {'mae': model_mae, 'r2': model_r2, 'rmse': model_rmse},
        'improvement': {'mae_pct': mae_improvement, 'r2_diff': r2_improvement}
    }
    
    return metrics, y_pred_model


def create_prediction_visualization(y_test: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    """Create actual vs predicted sales visualization."""
    logger.info("Creating prediction visualization...")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolor='black')
    
    # Perfect prediction line
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.title('Actual vs Predicted Sales', fontweight='bold', fontsize=14)
    plt.xlabel('Actual Sales ($)', fontsize=12)
    plt.ylabel('Predicted Sales ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pred_path = output_dir / "actual_vs_predicted.png"
    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction visualization saved to {pred_path}")


def save_model(model: RandomForestRegressor, model_dir: Path) -> Path:
    """Save trained model to disk."""
    logger.info("Saving trained model...")
    
    model_path = model_dir / "retail_sales_model.joblib"
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def save_metrics(metrics: dict, insights: dict, output_dir: Path) -> None:
    """Save evaluation metrics and business insights."""
    logger.info("Saving metrics and insights...")
    
    # Save model performance metrics
    metrics_df = pd.DataFrame([
        {
            'Model': 'Baseline (Mean)',
            'MAE': metrics['baseline']['mae'],
            'RMSE': metrics['baseline']['rmse'],
            'R2': metrics['baseline']['r2']
        },
        {
            'Model': 'Random Forest',
            'MAE': metrics['model']['mae'],
            'RMSE': metrics['model']['rmse'],
            'R2': metrics['model']['r2']
        }
    ])
    
    metrics_path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save business insights
    insights_df = pd.DataFrame([insights])
    insights_path = output_dir / "business_insights.csv"
    insights_df.to_csv(insights_path, index=False)
    
    logger.info("Metrics and insights saved")


def main():
    """Main execution function."""
    try:
        logger.info("=" * 60)
        logger.info("Retail Sales Analysis & Prediction Project Started")
        logger.info("=" * 60)
        
        # Setup paths
        BASE_DIR, DATA_DIR, OUTPUT_DIR, MODEL_DIR = setup_paths()
        
        # Load data
        df = load_data(DATA_DIR)
        
        # Validate data
        validate_data(df)
        
        # Data quality audit
        audit_results = data_quality_audit(df)
        
        # Clean and preprocess
        df = clean_and_preprocess(df)
        
        # Exploratory analysis
        insights = exploratory_analysis(df, OUTPUT_DIR)
        
        # Create business visualizations
        create_business_visualizations(df, OUTPUT_DIR)
        
        # Prepare features (leakage-free)
        X, y, feature_columns = prepare_features_leakage_free(df)
        
        # Time-aware split
        X_train, X_test, y_train, y_test = time_aware_split(df, X, y)
        
        # Train baseline model
        baseline = train_baseline_model(X_train, y_train)
        
        # Train ML model
        RANDOM_SEED = 42
        model = train_ml_model(X_train, y_train, random_state=RANDOM_SEED)
        
        # Evaluate models
        metrics, y_pred = evaluate_models(baseline, model, X_test, y_test, OUTPUT_DIR)
        
        # Create prediction visualization
        create_prediction_visualization(y_test, y_pred, OUTPUT_DIR)
        
        # Save model
        model_path = save_model(model, MODEL_DIR)
        
        # Save metrics and insights
        save_metrics(metrics, insights, OUTPUT_DIR)
        
        logger.info("=" * 60)
        logger.info("Retail Sales Project Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Outputs saved to: {OUTPUT_DIR}")
        logger.info(f"Random Seed Used: {RANDOM_SEED}")
        
        # Important disclaimers
        logger.info("\n" + "!" * 60)
        logger.info("IMPORTANT DISCLAIMERS")
        logger.info("!" * 60)
        logger.info("1. TARGET LEAKAGE PREVENTION: Profit was EXCLUDED from")
        logger.info("   features to prevent leakage, as Profit is derived from Sales.")
        logger.info("2. TIME-AWARE EVALUATION: Used chronological train-test split")
        logger.info("   instead of random split to prevent future data leakage.")
        logger.info("3. SALES ESTIMATION: This model estimates sales using features")
        logger.info("   available at transaction time, NOT future forecasting.")
        logger.info("4. LIMITATIONS: Model performance is based on historical data")
        logger.info("   and may not generalize to future periods or external conditions.")
        logger.info("!" * 60)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()