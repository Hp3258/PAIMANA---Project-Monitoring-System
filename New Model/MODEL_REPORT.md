# PAIMANA — New CatBoost Cost Overrun Model Report

This directory contains the production CatBoost model artifact, evaluation datasets, accuracy benchmarks, and portfolio scoring outputs integrated into PAIMANA.

---

## 1. Executive Summary & Accuracy Scoreboard

- **Model Type:** CatBoost Classifier (`MultiClass`, Standard Argmax Decision Policy)
- **Task:** 3-Month Future Cost Overrun Risk Classification
  - **Class 0 (Safe):** Cost growth $\le 5\%$
  - **Class 1 (Moderate):** Cost growth $5\% - 25\%$
  - **Class 2 (Severe):** Cost growth $> 25\%$
- **Validation Protocol:** Strict Temporal Split (Trained on data up to `2026-02-01`, evaluated on future holdout `2026-02-01` to `2026-08-01`)
- **Total Test Samples:** 1,570 observations

### Head-to-Head: New CatBoost vs Original XGBoost

| Metric | Original PAIMANA (XGBoost) | New Model (CatBoost) | Improvement |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 95.0% | **97.0%** | **+2.0%** |
| **Macro F1 Score** | 0.4826 | **0.5267** | **+9.1%** |
| **Weighted F1 Score** | 0.9600 | **0.9700** | **+1.0%** |
| **Severe Precision ($>25\%$)** | **25.0%** | **58.0%** | **+132% (2.3× higher)** |
| **Severe Recall ($>25\%$)** | 29.0% | **33.3%** | **+14.8%** |
| **Severe F1-Score** | **0.27** | **0.42** | **+55.6%** |

---

## 2. Detailed Classification Report (Holdout Test Set)

```
              precision    recall  f1-score   support

        Safe       0.98      0.99      0.98      1530
    Moderate       0.19      0.16      0.17        19
      Severe       0.58      0.33      0.42        21

    accuracy                           0.97      1570
   macro avg       0.58      0.49      0.53      1570
weighted avg       0.97      0.97      0.97      1570
```

### Confusion Matrix (Test Set: 1,570 rows)

| Actual \ Predicted | Predicted Safe | Predicted Moderate | Predicted Severe | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Safe** | **1,512** | 13 | 5 | 1,530 |
| **Actual Moderate** | 16 | **3** | 0 | 19 |
| **Actual Severe** | 14 | 0 | **7** | 21 |

- **Severe Overruns:** Only 5 false alarms out of 1,530 safe projects $\to$ **58% Precision**.

---

## 3. Top Feature Importances

| Rank | Feature | Importance (%) | Domain Meaning |
| :---: | :--- | :---: | :--- |
| 1 | `state` | **9.19%** | State-level land acquisition and execution environment |
| 2 | `sector` | **7.92%** | Domain complexity (e.g. Railways, Petroleum, Water) |
| 3 | `schedule_delay_months` | **7.48%** | Prior timeline slippage (strong precursor of cost escalation) |
| 4 | `expenditure_pct` | **7.40%** | Cumulative expenditure vs sanctioned budget |
| 5 | `original_cost_cr` | **7.27%** | Base financial sanction scale |
| 6 | `delta_cost_cr` | **5.76%** | Most recent month-over-month revision delta |
| 7 | `ewma_progress` | **5.36%** | Momentum of physical work |
| 8 | `progress_deficit` | **4.97%** | Gap between planned duration elapsed and physical progress |
| 9 | `cost_revision_count` | **4.64%** | Cumulative number of times the budget has been revised |
| 10 | `cost_lag_1` | **4.45%** | Base cost level from prior month snapshot |

---

## 4. File Inventory in this Directory

| File | Description |
| :--- | :--- |
| `catboost_model.joblib` | Production CatBoost model artifact with categoricals and feature lists |
| `catboost_benchmark_model.joblib` | High-capacity benchmark artifact (0.5969 macro F1 / 67% severe precision) |
| `metrics_summary.json` | Complete machine-readable accuracy, F1, and hyperparameter specifications |
| `classification_report.txt` | Formatted per-class precision, recall, and support |
| `confusion_matrix.csv` | Exact count matrix of actual vs predicted classes |
| `feature_importance.csv` | Ranked table of all 26 feature importances |
| `test_predictions.csv` | Project-by-project predictions with probabilities ($P_{\text{Safe}}, P_{\text{Mod}}, P_{\text{Sev}}$) |
| `latest_portfolio_risk_scores.csv` | All 754 projects scored with composite risk scores and SHAP explanations |
| `benchmark_21k_test_predictions.csv` | Predictions on the large 21,070-sample matched benchmark test split |

---

## 5. How to Load and Use in Code

```python
import joblib
import pandas as pd

# Load artifact
artifact = joblib.load("D:/Sapatee/PAIMANA/New Model/catboost_model.joblib")
model = artifact["model"]
features = artifact["features"]

# Predict on new feature DataFrame X
probabilities = model.predict_proba(X[features])
predictions = model.predict(X[features]).ravel()
```
