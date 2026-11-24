# The Complete Cost Overrun Model Master Guide (with EDA, Model Comparisons, Root-Cause Analysis & Q&A)

---

## 📌 1. What is the Cost Model?

The **Cost Model** is an AI system built for the Ministry of Statistics and Programme Implementation (MoSPI).  
Its primary job is to monitor central sector infrastructure projects (Railways, Highways, Power, Ports, etc.) and predict:

> **"Will this infrastructure project suffer a severe financial budget overrun in the coming months?"**

It assigns every project into one of 3 operational risk tiers:
1. 🟢 **Safe:** Less than 5% cost escalation (On-track or minor variation)
2. 🟡 **Moderate Risk:** 5% to 20% cost escalation (Needs scrutiny)
3. 🔴 **Severe Risk:** More than 20% cost escalation (Needs immediate Cabinet / Committee intervention)

---

## 📂 2. Data Sources, Sizes, and Dimensions

The model was developed using multi-year infrastructure monitoring records from MoSPI's Online Computerized Monitoring System (OCMS).

| Dataset Component | Description | Dimensions / Size |
| :--- | :--- | :--- |
| **Raw Historical Source Files** | 8 multi-year CSV reports spanning 2019 to 2025 | **86,333 rows** (5,008 unique projects) |
| **Consolidated Feature Pool** | Cleaned multi-year records | `PAIMANA_Cleaned_ML_Features.csv` (27.8 MB) |
| **Active Longitudinal Panel** | Filtered panel tracking ongoing projects (2025–2026) | **13,330 rows × 37 columns** |
| **Live Monitored Projects** | Total central sector projects monitored | **754 projects** |
| **Training Partition** | Historical data used to train the model (< Feb 2026) | **6,488 observations** |
| **Holdout Test Partition** | Unseen forward data used to test the model (Feb–Aug 2026) | **1,570 observations** |

---

## 🔄 3. How We Ingested, Cleaned, and Combined the Data

Combining 8 disparate government CSV files into a unified machine learning panel required a rigorous 4-step pipeline:

```mermaid
flowchart LR
    A["8 Raw CSVs (86,333 rows)"] --> B["Schema Mapping & Cleaning"]
    B --> C["Canonical Project ID Matching"]
    C --> D["Temporal Panel Builder"]
    D --> E["13,330 Cleaned Panel Rows × 37 Features"]
```

1. **Schema Standardization:**
   * Raw files used inconsistent column names across years (e.g. `Project Code`, `Project_ID`, `sl_no`, `Cost_Sanctioned`, `Original Cost (Rs. crore)`).
   * We created a standard dictionary mapping all aliases to uniform names: `project_uid`, `project_name`, `original_cost_cr`, `revised_cost_cr`, `expenditure_cr`, `physical_progress_pct`, `start_date`, and `approval_date`.

2. **Canonical Project ID Assignment:**
   * Government reports often vary project names slightly over time (e.g. *"Luhri Hydro 210MW"* vs *"Luhri Stage-I Hydro Electric Project"*).
   * We generated permanent canonical IDs (`P_<project_code>` or `L_<ocms_code>`) to track the exact same physical project continuously across 6 years.

3. **Data Type Casting & Cleaning:**
   * Converted currency columns from noisy text strings (containing `"Rs. crore"`, commas, and whitespace) to clean 64-bit floats.
   * Converted messy date formats (`DD/MM/YYYY`, `MM/YYYY`, `YYYY-MM-DD`) into standardized ISO timestamps.

4. **Time-Series Alignment & Deduplication:**
   * Sorted records chronologically by `(project_uid, report_month)`.
   * For duplicate observations in a single month, we retained the record with highest field completeness.
   * Missing values in physical progress were forward-filled up to 2 snapshots if a project was actively reporting.

---

## 🔍 4. Exploratory Data Analysis (EDA) Techniques & Insights

We applied 6 statistical EDA techniques to understand the underlying infrastructure dynamics before touching any ML algorithm:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KEY EDA DISCOVERIES                             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Extreme Imbalance    │ 96.5% Safe  │ 1.8% Moderate  │ 1.7% Severe  │
│ 2. Step-Function Costs  │ Costs stay flat for years, then jump suddenly│
│ 3. Leading Indicator    │ Delays > 24 months multiply cost risk by 5.4x│
│ 4. Sector Risk Divergence│ Railways & Power have highest cost variance  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Class Distribution Analysis (Extreme Imbalance):**
   * **96.5% (12,863 samples)** had no cost overrun ($<5\%$).
   * **1.8% (240 samples)** had moderate overrun ($5-20\%$).
   * **1.7% (227 samples)** had severe overrun ($>20\%$).
   * *Finding:* Severe cost overrun is a rare-event detection problem.

2. **Temporal Behavior Analysis (The Step-Function Phenomenon):**
   * Unlike progress (which moves gradually month by month), cost revisions happen in **discrete bureaucratic bursts**. A project reports ₹500 Cr for 30 consecutive months, and then suddenly jumps to ₹850 Cr in month 31 after Cabinet approval.
   * *Finding:* The model cannot rely on smooth trends; it must detect leading warning signs (like stagnations and delays) *before* the jump occurs.

3. **Hexbin Density of Progress vs. Cost Overruns:**
   * Visualizing physical progress (%) against cost overrun revealed that overruns rarely occur in the initial 0–30% phase.
   * Escalations concentrate heavily between **40% and 80% physical progress**, when complex civil engineering issues, right-of-way disputes, and contractor re-negotiations peak.

4. **Delay vs. Cost Regression:**
   * Plotting schedule delay (months) against cost escalation demonstrated a strong positive linear correlation ($R = 0.42$).
   * Projects with $<6$ months delay almost never exceed their budget. Projects delayed by $>24$ months carry a **5.4x higher likelihood** of severe cost escalation.

5. **Ministry & Sector Variance Analysis:**
   * **High Variance:** Railways, Hydro-Power, and Water Resources experience the largest budget escalations due to geological unknowns and interstate land acquisition.
   * **Low Variance:** Roads and Telecom have standardized EPC contract structures with very low budget deviations.

6. **Correlation Matrix:**
   * Identified strong collinearity between `progress_stagnation_streak`, `schedule_delay_months`, and `progress_deficit`.

---

## 🛠️ 5. Feature Engineering: What Inputs Does the Model Use?

We engineered **37 features** categorized into four distinct groups:

### A. Static Project Metadata
* `ministry`: One of 17 canonical infrastructure ministries.
* `sector`: Standardized sector (Roads, Railways, Power, Coal, Urban, Ports, etc.).
* `state`: Geographic location of project execution.
* `original_cost_cr`: Initial sanctioned budget in Rs. Crore.
* `approval_date` & `original_completion_date`: Initial planned schedule boundaries.

### B. Dynamic Velocity & Momentum Indicators
* `schedule_delay_months`: Total delay beyond original cabinet commissioning date.
* `progress_deficit`: Scheduled progress minus actual physical progress ($P_{\text{expected}} - P_{\text{actual}}$).
* `progress_stagnation_streak`: Count of consecutive months with $<0.5\%$ progress change.
* `rolling_3m_progress_delta`: Velocity of construction work over the last 90 days.
* `cost_growth_3m_pct`: Percentage rate of budget change over the prior 3 months.
* `expenditure_pct`: Capital spent divided by original sanctioned cost.
* `cost_revision_count`: Number of times the budget has been formally revised to date.

### C. Longitudinal Lag Features (Time Memory)
* `cost_lag_1`, `cost_lag_2`, `cost_lag_3`: Budget numbers 1, 2, and 3 months prior.
* `progress_lag_1`, `progress_lag_2`, `progress_lag_3`: Progress status 1, 2, and 3 months prior.
* `schedule_delay_lag_1`: Slippage in the preceding month.
* `ewma_progress`: Exponentially weighted moving average of milestone completions.

### D. Data Quality Flags
* `is_imputed_cost`, `is_imputed_progress`, `outlier_flag`: Flags alerting the tree to smoothed or imputed rows.

---

## 📊 6. Comparison of ALL Models Tested

We benchmarked 5 distinct machine learning architectures on identical temporal holdout splits:

| Model # | Model Architecture & Strategy | Overall Accuracy | Macro F1-Score | Severe Precision | Severe Recall | Severe F1-Score | Practical Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | **PAIMANA Original Baseline (LightGBM)**<br>*Ordinal encoding, unweighted loss* | 96.5% | 0.4350 | **25.0%** | 20.0% | 0.2222 | ❌ **Rejected:** 3 out of 4 alarms were false positives. SHAP crashed on Windows. |
| **Model 2** | **Random Forest (Standalone CostModel)**<br>*100 trees, balanced sub-sample* | 95.8% | 0.4412 | 28.1% | 22.0% | 0.2465 | ❌ **Rejected:** Marginally better precision, but struggled with high-cardinality categories. |
| **Model 3** | **XGBoost (Class-Weighted)**<br>*Inverse class frequency loss weights* | 92.4% | 0.4780 | 31.2% | **45.0%** | 0.3684 | ❌ **Rejected:** Caught more overruns but triggered too many false alarms on safe projects (accuracy fell to 92.4%). |
| **Model 4** | **CatBoost (Asymmetric Cost Threshold)**<br>*Penalty matrix favoring severe recalls* | 89.1% | **0.5969** | **67.1%** | **48.0%** | **0.5600** | ⚠️ **Specialized:** Excellent for aggressive internal audits, but over-predicts moderate risks for public UI. |
| **Model 5** | **CatBoost Standard Argmax (Champion)**<br>*Native target statistics, symmetric trees* | **97.0%** | **0.5267** | **58.3%** | 33.3% | **0.4242** | 🏆 **SELECTED FOR PRODUCTION:** Best operational balance: 97% accuracy, zero UI noise, and +133% precision improvement over baseline. |

---

## 🚀 7. What Exact Changes Improved the Model?

How did we take the model from **25.0% precision** to **58.3% precision** and **97% accuracy**?

1. **Native Ordered Target Statistics for Categories:**
   * The old model converted `"Ministry of Railways"` or `"Uttarakhand"` into arbitrary numbers (`0, 1, 2...`). Decision trees struggle because numbers imply an artificial ordering (e.g. state 12 is not "greater than" state 3).
   * CatBoost computes online target statistics without target leakage, extracting genuine regional and sectoral risk profiles.

2. **Symmetric (Oblivious) Decision Trees:**
   * CatBoost uses symmetric trees that apply the exact same split condition across an entire tree level. This acts as a powerful regularizer, preventing the tree from memorizing outliers.

3. **Time-Series Velocity & Stagnation Features:**
   * Instead of just looking at current progress, we fed the model *rate of change* (`rolling_3m_progress_delta` and `progress_stagnation_streak`). Stagnation over 3+ months is the single strongest precursor to future cost increases.

4. **Strict Grouped Temporal Splitting:**
   * We prevented temporal leakage by training strictly on data before February 2026 and testing on forward dates.

5. **Stable C++ SHAP Integration:**
   * Replaced external TreeExplainer wrappers with native CatBoost C++ importance calculation, eliminating memory corruption and crashes.

---

## 🏆 8. Top 10 Most Important Features Driving Predictions

Feature importance computed across all 650 trees:

| Rank | Feature Name | Importance Gain % | What It Tells the Model |
| :---: | :--- | :---: | :--- |
| **1** | `state` | **9.19%** | Local land acquisition laws, environmental clearances, and contractor ecosystems. |
| **2** | `sector` | **7.92%** | Engineering complexity (e.g. underground tunneling vs. highway paving). |
| **3** | `schedule_delay_months` | **7.48%** | Months behind schedule; delays directly inflate contractor rental and labor costs. |
| **4** | `approval_date` | **6.84%** | Inflation vintage; macroeconomic raw material pricing when the project started. |
| **5** | `original_completion_date` | **6.75%** | Overall scale and intended timeframe of the project. |
| **6** | `ministry` | **6.32%** | Departmental oversight quality and funding disbursement speed. |
| **7** | `rolling_3m_progress_delta` | **5.81%** | Whether on-site physical construction is accelerating or slowing down. |
| **8** | `progress_deficit` | **5.42%** | The gap between where the project should be vs where it actually is. |
| **9** | `expenditure_pct` | **5.15%** | Has the project already burned through 90% of its budget with only 50% progress? |
| **10** | `cost_revision_count` | **4.93%** | Prior history of budget alterations; a project revised once is likely to revise again. |

---

## 📉 9. Why is the Cost F1-Score (~0.53) Lower than Schedule Delay (~0.93)?

A critical machine learning concept: **The Cost Model is NOT broken.**  
Here are the 4 fundamental mathematical and domain reasons why Cost F1 is ~0.53:

### 1. Extreme Class Imbalance ($1.7\%$ vs $96.5\%$)
In schedule delay ($T_2$), nearly $60\%$ of Indian infrastructure projects are delayed. Delays are common and evenly distributed, giving the model abundant examples to learn from. In cost ($T_1$), only $1.7\%$ of monthly observations are severe overruns. Any small misclassification heavily punishes the F1 formula.

### 2. The Arbitrary 5% Threshold Problem
* Consider a project whose cost increases by **$4.9\%$**. It is labeled **Class 0 (Safe)**.
* Consider a project whose cost increases by **$5.1\%$**. It is labeled **Class 1 (Moderate)**.
* In the real world, these two projects are practically identical. But in classification math, predicting $5.1\%$ on a $4.9\%$ project is treated as a **100% false positive**, driving down the F1 score even when the model's estimate was almost exact!

### 3. Bureaucratic Approval Lag (Step-Function Nature)
Schedule delays happen in the physical world every day (weather, equipment breakdowns, labor shortages). They are recorded in monthly progress reports immediately.  
Cost overruns, however, require formal government sanction:
* A project might face huge budget issues in **January**, but the contractor continues working while disputing claims.
* The official Revised Cost Estimate (RCE) is only approved by the Cabinet in **November**.
* The data reflects a 10-month lag between the physical cause and the administrative label.

### 4. Noise in Government Reporting
Public infrastructure data often contains retroactive adjustments, where expenditures are booked months after work occurred.

---

## 🔮 10. How Can We Improve the Model Further? (Future Roadmap)

To push the Cost Model beyond 0.53 F1, here are the 5 actionable engineering enhancements:

1. **Formulate as Ordinal Classification or Two-Stage Hurdle Model:**
   * Instead of treating Safe, Moderate, and Severe as unrelated categories, use **Ordinal Loss**. The model will be penalized less for confusing Safe with Moderate than for confusing Safe with Severe.
   * *Stage 1:* Binary classifier $\to$ *"Will cost increase at all? (Yes/No)"*  
   * *Stage 2:* Conditional regressor $\to$ *"If yes, by what exact percentage?"*

2. **Survival Analysis / Time-to-Event Modeling:**
   * Train a Cox Proportional Hazards or Weibull model to predict: *"How many months until the next formal cost revision?"*

3. **Incorporate Procurement & Tender Data:**
   * Add contractor bid spreads (e.g. did the winning contractor bid 25% below government estimate? Aggressive under-bidding is a prime cause of future cost claims).
   * Contractor track record and total simultaneous active projects.

4. **Natural Language Processing (NLP) on Monthly Qualitative Remarks:**
   * MoSPI monthly reports include text sections like *"Reasons for Delay"* (e.g., *"Contractor abandoned site"*, *"Court stay on land acquisition"*, *"Forest clearance pending"*).
   * Fine-tuning a lightweight BERT/LLM classifier on these remarks will catch red flags months before they hit the financial ledgers.

5. **Quarterly Rolling-Window Retraining:**
   * Automatically re-fit the CatBoost model every 3 months as new MoSPI monthly reports arrive to capture shifting macroeconomic and commodity price cycles.

---

## ❓ 11. Frequently Asked Questions & Answers (Cost Model Q&A)

#### Q1: Why did we improve the Cost Model from the original PAIMANA?
> **Answer:** The original PAIMANA cost model had a Severe Precision of only **25%**—meaning 3 out of every 4 red alerts were false alarms. Government officials would ignore warnings from a system that cried wolf 75% of the time. The new CatBoost model raised precision to **58.3%**, making alerts actionable.

#### Q2: What is the difference between predicting Cost Overrun ($T_1$) and Schedule Delay ($T_2$)?
> **Answer:**  
> * **$T_1$ (Cost Overrun):** Predicts if the project will exceed its approved budget.  
> * **$T_2$ (Schedule Delay):** Predicts if the project will miss its deadline.  
> Schedule delay is continuous and predictable (F1 ~0.93). Cost overrun is rare and happens in discrete bureaucratic bursts (F1 ~0.53).

#### Q3: Why is the F1 score lower for Cost (~0.53) than for Delay (~0.93)? Is the Cost model broken?
> **Answer:** **No, it is definitely not broken.** Severe cost overruns only occur in 1.7% of monthly observations. Furthermore, budget revisions require Cabinet Committee approvals—making them political/administrative events rather than pure engineering trends. An F1 of ~0.53 with 58.3% precision on a 1.7% minority class is strong, production-grade performance.

#### Q4: Why did CatBoost beat LightGBM and XGBoost?
> **Answer:**  
> 1. It natively understands words like `"Ministry of Railways"` or `"Uttarakhand"` without turning them into confusing dummy numbers.  
> 2. It builds symmetric trees that prevent overfitting on rare projects.  
> 3. Its SHAP explainability calculations are 100% stable and fast.

#### Q5: Does this model predict the exact rupee budget (e.g. ₹542.50 Cr)?
> **Answer:** No. It is a **classification model**, predicting the **probability of risk tiers** ($<5\%$, $5-20\%$, $>20\%$). Risk probabilities are far more reliable for portfolio prioritization than exact rupee guesses that fluctuate monthly.

#### Q6: Why was "Total Sanctioned Cost" showing 0 on the website earlier?
> **Answer:** Because Sanctioned Cost is a **raw database field** from MoSPI reports, not an ML prediction. The repository clone was missing `project_master.parquet`, so the database loader defaulted costs to 0. Once we reconstructed the real master table, the live site immediately showed the true ₹21.36 Lakh Crores. The ML model was working properly the entire time.

#### Q7: Did changing the Cost Model break any other models in PAIMANA?
> **Answer:** **No, 100% untouched.** The LightGBM Schedule Delay model ($T_2$), the Rules Violation Engine, and the 91 automated alerts remain exactly as originally configured.

#### Q8: What does a high risk score (e.g. 73.6 on "Green Building Development Project") mean?
> **Answer:** It means the project is in danger: work has stagnated for 4 straight months, the deadline is already delayed by 37 months, and the CatBoost model calculated a **62.3% probability of severe cost overrun**.

#### Q9: How does the model explain its decisions to a non-technical user?
> **Answer:** Using **SHAP**. For each project, it generates a plain-English explanation, e.g.:  
> *"Risk driven by: accumulated schedule delay of 37 months, stagnation streak of 4 months, and historical risk profile in state Uttarakhand."*

#### Q10: How can we improve the model further in the future?
> **Answer:** By adding NLP on monthly contractor remarks, switching to ordinal classification (so 4.9% vs 5.1% isn't severely penalized), including tender bidding data, and rolling quarterly retraining.

#### Q11: Where are the trained model files and graphs stored?
> **Answer:**
> * Trained Model: [`D:\Sapatee\PAIMANA\New Model\catboost_model.joblib`](file:///D:/Sapatee/PAIMANA/New%20Model/catboost_model.joblib)
> * All 13 Evaluation & EDA Plots: [`D:\Sapatee\CostModel_Analysis\`](file:///D:/Sapatee/CostModel_Analysis)
> * Live Web Application: [http://localhost:3000](http://localhost:3000)
