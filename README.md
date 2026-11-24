# PAIMANA: AI-Powered Predictive Analytics & Early Warning System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg)](https://react.dev/)
[![CatBoost](https://img.shields.io/badge/ML-CatBoost%20%7C%20LightGBM-yellow.svg)](https://catboost.ai/)
[![XAI](https://img.shields.io/badge/Explainability-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)

> **Organization:** Ministry of Statistics & Programme Implementation (MoSPI), Government of India  
> **Problem Statement ID:** SIH 2026 — PS26103  
> **Repository:** [Hp3258/PAIMANA---Project-Monitoring-System](https://github.com/Hp3258/PAIMANA---Project-Monitoring-System)

---

## 📌 Executive Summary

**PAIMANA** (**P**redictive **A**nalytics and **I**ntelligence for **M**onitoring of **A**dvanced **N**ational **A**ssets) transforms India's central infrastructure monitoring from **reactive retrospective auditing** into a **proactive 3-month early warning radar**.

The system ingests monthly MoSPI Flash Reports across **17 ministries**, processes multi-year longitudinal panels (86,000+ observation records across 5,000+ infrastructure projects), trains machine learning ensembles (**CatBoost** for cost escalation and **LightGBM** for schedule slippage), fuses them with deterministic safety rules into a composite **0–100 project risk score**, and delivers **TreeSHAP** plain-language root-cause explanations on an executive React dashboard.

---

## 🌟 Key Features

* **3-Month Forward Forecast Horizon ($T \to T+3$):** Predicts cost overrun severity and schedule delay state three months before official cabinet/budget revisions.
* **Champion CatBoost Cost Model:** Native target statistics and symmetric (oblivious) trees deliver **97.0% overall accuracy** and **58.3% precision on severe overruns** (+133% relative improvement over baseline models).
* **High-Performance Schedule Delay Model:** LightGBM leaf-wise gradient boosting achieves **93.57% test accuracy** and **0.9266 Macro-F1**.
* **Zero Temporal Leakage:** Strict chronological out-of-time train/test boundary (trained on pre-February 2026 records, tested strictly on holdout future snapshots).
* **Hybrid Risk Engine (0–100 Scale):** Blends multi-class ML severity probabilities with 5 deterministic domain rules serving as an auditable safety floor.
* **Explainable AI (TreeSHAP):** Decomposes every high/medium risk prediction into directional mathematical feature attributions rendered in plain English.
* **Full-Stack Application:** Asynchronous FastAPI backend (port 8000) with JWT authentication coupled with a modern React 18 + Vite + TailwindCSS dashboard (port 3000).

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                            │
│  MoSPI PDF Flash Reports (Table 6) ──> Table Extraction & Cleaning    │
│  Hierarchical UID Resolution ──> Clean Longitudinal Panel (Parquet)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FEATURE & TARGET ENGINEERING                       │
│  • 26 Features: Temporal Lags (t-1..3), Stagnation Streaks, Deficits   │
│  • T+3 Forward Targets: 3-Month Cost Growth & Absolute Delay Category  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│     COST OVERRUN PREDICTOR        │ │      SCHEDULE DELAY PREDICTOR     │
│   Algorithm: CatBoost (Argmax)    │ │       Algorithm: LightGBM         │
│   Target: Safe / Mod / Severe     │ │   Target: None / Minor / Severe   │
└─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘
                  │                                     │
                  └─────────────────┬───────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  HYBRID RISK & EXPLAINABILITY ENGINE                   │
│   ML Risk Score: 0.45*Cost + 0.35*Delay + 0.10*Stagnation + 0.10*Rev    │
│   Safety Floor: max(ML Risk Score, Deterministic Rules Score)           │
│   Explainability: TreeSHAP Attribution (Top Drivers & Diagnosis)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       FULL-STACK APPLICATION                           │
│   Backend: FastAPI REST API + SQLite (paimana.db) + JWT Auth           │
│   Frontend: React 18 + Vite + TailwindCSS + Recharts Interactive UI    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Feature Engineering

### Dataset Overview
* **Source:** Monthly MoSPI IPMD Flash Reports (Central Sector Projects costing $\ge$ ₹150 Cr).
* **Multi-Year Repository:** 86,333 raw observations across 5,008 unique projects.
* **Cleaned Longitudinal Panel:** 13,330 monthly observations across 17 ministries.
* **Active Operational Portfolio:** 754 projects valued at **₹21.36 Lakh Crore**.

### 26 Input Features
| Group | Features | Description |
| :--- | :--- | :--- |
| **Financial State** | `original_cost_cr`, `revised_cost_cr`, `expenditure_cr` | Budget scale, revised ceiling, and cumulative funds spent |
| **Physical Status** | `physical_progress_pct`, `schedule_delay_months`, `progress_stagnation_streak` | Cumulative progress %, delay in months, and consecutive stalled months |
| **Financial Ratios** | `expenditure_pct`, `cost_overrun_pct_current`, `delta_cost_cr`, `delta_progress_pct` | Utilization ratio, current overrun %, month-over-month differentials |
| **Progress Benchmarks** | `expected_progress_pct`, `progress_deficit` | Linear progress benchmark based on elapsed time vs. shortfall |
| **Temporal Memory** | `cost_lag_1..3`, `progress_lag_1..3`, `schedule_delay_lag_1` | Multi-step historical observations ($t-1, t-2, t-3$) |
| **Rolling & Momentum**| `rolling_3m_progress_delta`, `rolling_3m_cost_growth`, `ewma_progress`, `cost_revision_count` | 3-month velocity, cost additions, EWMA progress ($\alpha=0.5$), upward revision count |
| **Categorical Context**| `ministry`, `sector`, `state` | Institutional ownership, engineering complexity, and geographical execution context |

---

## 📈 Model Performance & Benchmarking

Evaluated on a **strict chronological holdout test partition of 1,570 observations** (March–May 2026):

### 1. Cost Overrun Model (CatBoost Classifier)
* **Overall Test Accuracy:** **97.0%**
* **Macro-F1 Score:** **0.5267**
* **Severe Overrun Precision ($>25\%$):** **58.3%**
* **Macro ROC-AUC:** **0.6997**

#### Confusion Matrix (Holdout Test Set: 1,570 rows)
| Actual \ Predicted | Predicted Safe | Predicted Moderate | Predicted Severe | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Safe (0)** | **1,512** | 13 | 5 | 1,530 |
| **Actual Moderate (1)** | 16 | **3** | 0 | 19 |
| **Actual Severe (2)** | 14 | 0 | **7** | 21 |

> **Operational Significance:** Out of 1,530 safe projects, only 5 produced false alarms. When an alert fires, administrators can trust it with **58.3% precision**, mitigating audit fatigue.

### 2. Schedule Delay Model (LightGBM Classifier)
* **Overall Test Accuracy:** **93.57%**
* **Macro-F1 Score:** **0.9266**
* **Macro ROC-AUC:** **0.9831**
* **Class-Wise F1:** No Delay: **0.9395** | Minor Delay: **0.8790** | Severe Delay: **0.9613**

### 3. Empirical Multi-Model Comparison
| Model Architecture | Accuracy | Macro-F1 | Severe Precision | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline LightGBM** | 96.5% | 0.4350 | 25.0% | ❌ Rejected (3/4 alerts were false alarms) |
| **Random Forest** | 95.8% | 0.4412 | 28.1% | ❌ Rejected (Poor generalization on unseen periods) |
| **XGBoost (Class-Weighted)** | 92.4% | 0.4780 | 31.2% | ❌ Rejected (Excess false positives on safe projects) |
| **CatBoost Standard Argmax** | **97.0%** | **0.5267** | **58.3%** | 🏆 **Champion Production Model** |

---

## ⚙️ Hybrid Risk Scoring Formula

The portfolio risk score (0 to 100) integrates predictive ML probabilities with physical project indicators:

$$\text{ML Risk} = 0.45 \cdot (\hat{P}_{\text{cost}} \cdot 100) + 0.35 \cdot (\hat{P}_{\text{delay}} \cdot 100) + 0.10 \cdot (\text{Stagnation} \cdot 100) + 0.10 \cdot (\text{Revision} \cdot 100)$$

$$\text{Final Risk Score} = \max(\text{ML Risk Score}, \text{Rule Score})$$

### Risk Bands
* 🟢 **Low Risk (0 – 40):** Normal baseline project execution.
* 🟡 **Medium Risk (41 – 70):** Early signs of stagnation or moderate overrun probability.
* 🔴 **High Risk (71 – 100):** Immediate audit required; high probability of acute budget or schedule breach.

### Deterministic Safety Rules
1. **Cost Tripled:** Revised cost $\ge 3.0 \times$ original sanction (+35 pts).
2. **Prolonged Stagnation:** Work stagnated for $\ge 6$ consecutive months (+30 pts) or $\ge 3$ months (+15 pts).
3. **Premature Budget Depletion:** Expenditure $\ge 90\%$ while progress $< 75\%$ (+25 pts).
4. **Multiple Revisions with Minimal Progress:** Cost revised $\ge 2$ times with progress $< 5\%$ (+20 pts).
5. **Severe Timeline Slippage:** Schedule delay $\ge 36$ months (+20 pts).

---

## 🚀 Quickstart Guide

### Prerequisites
* **Python 3.11+**
* **Node.js 18+** & npm
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/Hp3258/PAIMANA---Project-Monitoring-System.git
cd PAIMANA---Project-Monitoring-System
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install lightgbm catboost shap pyarrow

# Run FastAPI backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
* **API Documentation:** `http://localhost:8000/docs`

### 3. Frontend Setup
In a new terminal:
```bash
cd frontend
npm install
npm run dev
```
* **Web Dashboard:** `http://localhost:3000`

### 4. Default Login Credentials
| Role | Email | Password |
| :--- | :--- | :--- |
| **Administrator** | `admin@mospi.gov.in` | `Admin@123` |
| **Analyst** | `analyst@mospi.gov.in` | `Analyst@123` |

---

## 📂 Project Directory Structure

```
PAIMANA/
├── backend/                  # FastAPI web server and database services
│   ├── app/
│   │   ├── main.py           # Application entrypoint & CORS middleware
│   │   ├── routers/          # REST routes (projects, alerts, dashboard, auth)
│   │   ├── models/           # SQLAlchemy ORM schemas
│   │   └── services/         # Risk scoring service & database queries
│   └── core/                 # JWT authentication & app configurations
├── frontend/                 # React 18 + Vite + TailwindCSS application
│   ├── src/
│   │   ├── pages/            # Overview, Projects, Detail, Alerts, Admin
│   │   ├── components/       # Metric cards, SHAP bar charts, S-curves
│   │   └── api/              # Axios API client integrations
│   └── vite.config.ts        # Vite configuration & backend proxy
├── ml/                       # Machine learning pipeline
│   ├── features/             # Feature matrix generation (build_features.py)
│   ├── training/             # CatBoost & LightGBM training + scoring scripts
│   ├── models/               # Serialized model artifacts (.joblib)
│   └── eda_outputs/          # Diagnostic plots & scoring outputs
├── data/                     # Data processing pipeline
│   ├── master/               # Parquet longitudinal panels
│   └── scripts/              # PDF parsing & dataset cleaning routines
├── paimana.db                # SQLite database with 754 projects & SHAP records
├── start_all.bat             # One-click Windows launch script
└── README.md                 # Project documentation
```

---

## 👥 Authors & Acknowledgments

* **Harish Ravsaheb Pawar** ([@Hp3258](https://github.com/Hp3258))
* **Rohan Abasaheb Sapate** ([@rohansapate21](https://github.com/rohansapate21))

Built as part of **Smart India Hackathon (SIH 2026)** for Problem Statement ID **26103** under the mentorship of the Ministry of Statistics & Programme Implementation (MoSPI).

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
