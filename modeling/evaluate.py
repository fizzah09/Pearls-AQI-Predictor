import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from pathlib import Path

log = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'n_samples': len(y_true)
    }
    return metrics
def print_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"  RMSE (Root Mean Squared Error):  {metrics['rmse']:.4f}")
    print(f"  MAE  (Mean Absolute Error):      {metrics['mae']:.4f}")
    print(f"  R²   (R-squared Score):          {metrics['r2']:.4f}")
    print(f"  MAPE (Mean Abs Percentage Error): {metrics['mape']:.2f}%")
    print(f"  Samples:                          {metrics['n_samples']}")
    print(f"{'='*60}\n")


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "evaluation"
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive model evaluation on train/val/test sets.
    
    Args:
        model: Trained model with predict() method
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        output_dir: Directory to save evaluation plots
    
    Returns:
        Dictionary with metrics for each dataset split
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log.info("Evaluating model on all splits...")
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train.values, y_train_pred)
    val_metrics = calculate_metrics(y_val.values, y_val_pred)
    test_metrics = calculate_metrics(y_test.values, y_test_pred)
    
    print_metrics(train_metrics, "TRAINING SET PERFORMANCE")
    print_metrics(val_metrics, "VALIDATION SET PERFORMANCE")
    print_metrics(test_metrics, "TEST SET PERFORMANCE")
    
    _plot_predictions_vs_actual(y_test.values, y_test_pred, output_dir, "test")
    _plot_residuals(y_test.values, y_test_pred, output_dir)
    _plot_metrics_comparison(train_metrics, val_metrics, test_metrics, output_dir)
    
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(f"{output_dir}/metrics_summary.csv")
    log.info(" Metrics saved to: %s/metrics_summary.csv", output_dir)
    
    return results


def _plot_predictions_vs_actual(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    output_dir: str,
    dataset_name: str = "test"
):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual AQI', fontsize=12)
    plt.ylabel('Predicted AQI', fontsize=12)
    plt.title(f'Predicted vs Actual AQI ({dataset_name.upper()} Set)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_vs_actual_{dataset_name}.png", dpi=150)
    plt.close()
    
    log.info(" Saved plot: predictions_vs_actual_%s.png", dataset_name)


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Plot residual distribution."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted AQI', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_analysis.png", dpi=150)
    plt.close()
    
    log.info(" Saved plot: residuals_analysis.png")


def _plot_metrics_comparison(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    output_dir: str
):
    """Plot comparison of metrics across datasets."""
    metrics_names = ['rmse', 'mae', 'r2']
    train_vals = [train_metrics[m] for m in metrics_names]
    val_vals = [val_metrics[m] for m in metrics_names]
    test_vals = [test_metrics[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, train_vals, width, label='Train', alpha=0.8)
    ax.bar(x, val_vals, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_vals, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['RMSE', 'MAE', 'R²'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=150)
    plt.close()
    
    log.info(" Saved plot: metrics_comparison.png")


def plot_feature_importance(
    importance_df: pd.DataFrame, 
    output_dir: str,
    top_n: int = 20
):
    """
    Plot top N most important features.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_dir: Directory to save plot
        top_n: Number of top features to display
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values, alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close()
    
    log.info(" Saved plot: feature_importance.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    np.random.seed(42)
    y_true = np.array([1, 2, 3, 4, 5, 2, 3, 4, 3, 2])
    y_pred = y_true + np.random.normal(0, 0.5, len(y_true))
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "TEST METRICS")
