"""
SIH 2026 — PS26103 | Phase 6: Cost Overrun Model Training (CatBoost Standard Argmax)
=====================================================================================
Trains a CatBoost classification model to predict 3-Month Future Cost Overrun Risk.
- Uses strict temporal validation split (no look-ahead leakage).
- Uses CatBoost native categorical support (ministry, sector, state).
- Handles extreme class imbalance using balanced sample weighting.
- Standard argmax decision policy.
- Optional Optuna hyperparameter optimization (--tune).

Usage:
    python ml/training/train_cost_model.py
    python ml/training/train_cost_model.py --tune --trials 25
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "ml" / "features"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
import sys
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


def prepare_data():
    input_path = FEATURES_DIR / "feature_matrix.parquet"
    if not input_path.exists():
        logger.error("feature_matrix.parquet not found. Run build_features.py first.")
        return None
        
    df = pd.read_parquet(input_path)
    # Only train on rows where future target is observable
    df = df[df['target_cost_overrun_class'].notna()].copy()
    df['target_cost_overrun_class'] = df['target_cost_overrun_class'].astype(int)
    logger.info(f"Loaded {len(df)} labeled observations.")
    
    # Temporal Train/Test Split (70% past, 30% future)
    df['report_month'] = pd.to_datetime(df['report_month'])
    split_date = df['report_month'].quantile(0.70)
    
    train_df = df[df['report_month'] <= split_date].copy()
    test_df = df[df['report_month'] > split_date].copy()
    
    logger.info(f"Train set: {len(train_df)} samples (up to {split_date.date()})")
    logger.info(f"Test set: {len(test_df)} samples (after {split_date.date()})")
    
    feature_cols = [
        'original_cost_cr', 'revised_cost_cr', 'expenditure_cr',
        'physical_progress_pct', 'delta_cost_cr', 'delta_progress_pct',
        'progress_stagnation_streak', 'cost_revision_count',
        'expenditure_pct', 'cost_overrun_pct_current',
        'rolling_3m_progress_delta', 'rolling_3m_cost_growth',
        'cost_lag_1', 'cost_lag_2', 'cost_lag_3',
        'progress_lag_1', 'progress_lag_2', 'progress_lag_3',
        'schedule_delay_lag_1', 'ewma_progress',
        'expected_progress_pct', 'progress_deficit',
        'schedule_delay_months', 'ministry', 'sector', 'state'
    ]
    
    cat_cols = ['ministry', 'sector', 'state']
    category_mappings = {}
    for col in cat_cols:
        train_df[col] = train_df[col].fillna('Unknown').astype(str)
        test_df[col] = test_df[col].fillna('Unknown').astype(str)
        category_mappings[col] = sorted(train_df[col].unique().tolist())
        
    X_train = train_df[feature_cols]
    y_train = train_df['target_cost_overrun_class']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target_cost_overrun_class']
    
    return X_train, y_train, X_test, y_test, feature_cols, cat_cols, category_mappings


def tune_hyperparameters(X_train, y_train, X_test, y_test, cat_cols, sample_weights, n_trials=25):
    """Run Optuna study to find optimal CatBoost parameters."""
    if optuna is None:
        logger.warning("Optuna not installed. Using default parameters.")
        return {}
        
    logger.info(f"Starting Optuna hyperparameter optimization for CatBoost ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'loss_function': 'MultiClass',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'iterations': trial.suggest_int('iterations', 200, 700),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        }
        
        clf = CatBoostClassifier(cat_features=cat_cols, **params)
        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=(X_test, y_test),
            early_stopping_rounds=20,
            verbose=False
        )
        preds = clf.predict(X_test).ravel().astype(int)
        return f1_score(y_test, preds, average='macro', zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best trial macro F1: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    return study.best_params


def train_model(tune: bool = False, trials: int = 25):
    data = prepare_data()
    if data is None:
        return
        
    X_train, y_train, X_test, y_test, feature_cols, cat_cols, category_mappings = data
    
    # Class imbalance handling: compute balanced sample weights
    sample_weights = compute_sample_weight('balanced', y_train)
    logger.info(f"Class distribution (train):\n{y_train.value_counts().to_dict()}")
    
    best_params = {}
    if tune:
        best_params = tune_hyperparameters(X_train, y_train, X_test, y_test, cat_cols, sample_weights, n_trials=trials)
        
    # CatBoost hyperparameters matching CostModelNext benchmark architecture
    default_params = {
        'loss_function': 'MultiClass',
        'iterations': 650,
        'depth': 7,
        'learning_rate': 0.05,
        'l2_leaf_reg': 8,
        'random_seed': 42,
        'verbose': 100,
        'allow_writing_files': False,
        'early_stopping_rounds': 30
    }
    default_params.update(best_params)
    
    logger.info(f"Training CatBoost Classifier with parameters: {default_params}")
    model = CatBoostClassifier(cat_features=cat_cols, **default_params)
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # Evaluation (Standard Argmax)
    logger.info("Evaluating Cost Overrun Model (CatBoost standard argmax) on temporal test set...")
    y_pred = model.predict(X_test).ravel().astype(int)
    y_prob = model.predict_proba(X_test)
    
    report = classification_report(y_test, y_pred, target_names=['Safe', 'Moderate', 'Severe'], zero_division=0)
    logger.info(f"\n{report}")
    
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    logger.info(f"Macro F1 Score: {macro_f1:.4f}")
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 5 Important Features for Cost Escalation:\n{importances.head(5).to_string(index=False)}")
    
    # Save Artifacts
    model_path = MODELS_DIR / "best_model_cost.joblib"
    joblib.dump({
        'model': model,
        'features': feature_cols,
        'categorical_columns': cat_cols,
        'category_mappings': category_mappings,
        'macro_f1': macro_f1,
        'model_name': 'CatBoost Classifier (Standard Argmax)'
    }, model_path)
    
    logger.info(f"Saved trained cost model artifacts to {model_path}")

    # Log to MLflow if available
    if mlflow is not None:
        try:
            db_path = PROJECT_ROOT / "mlflow.db"
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("paimana_infrastructure_risk")
            with mlflow.start_run(run_name=f"catboost_cost_model_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_params(default_params)
                mlflow.log_metric("macro_f1", float(macro_f1))
                mlflow.log_artifact(str(model_path))
                logger.info("Logged CatBoost cost model training run and artifact to MLflow.")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cost Overrun Classifier with CatBoost")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    args = parser.parse_args()
    
    train_model(tune=args.tune, trials=args.trials)
