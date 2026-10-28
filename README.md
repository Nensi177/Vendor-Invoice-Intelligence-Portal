# 📦 Vendor Invoice Intelligence Portal
**Freight Cost Prediction & Invoice Risk Flagging**

An internal analytics and machine learning system for enterprise supply chain operations. The portal provides automated freight charge estimation and early-warning invoice risk assessment to help procurement and finance teams prevent financial leakage, identify accounting mismatches, and prioritize manual invoice reviews.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture & Directory Structure](#system-architecture--directory-structure)
- [Database Schema & Data Sources](#database-schema--data-sources)
- [Models & Methodological Framework](#models--methodological-framework)
  - [Freight Cost Prediction Model](#freight-cost-prediction-model)
  - [Invoice Risk Flagging Framework](#invoice-risk-flagging-framework)
- [Feature Status & Implementation Matrix](#feature-status--implementation-matrix)
- [Setup & Operational Runbook](#setup--operational-runbook)
- [Known Limitations & Future Improvements](#known-limitations--future-improvements)

---

<a id="project-overview"></a>
## 📋 1. Project Overview

In accounts payable and supply chain operations, manual verification of vendor invoices across thousands of line items is labor-intensive and error-prone. This system provides two complementary capabilities:

1. **Freight Cost Prediction**: Estimates expected freight charges from invoice spend using a verified **Linear Regression** rate card to detect abnormal logistics surcharges.
2. **Invoice Risk Flagging**: Triages invoices for manual review by combining deterministic 3-way accounting checks (billed vs. received dollars and quantities) with a **Random Forest** risk pattern classifier.

> [!NOTE]
> The freight model predicts expected charges based on a commercial rate card baseline. The risk classifier prioritizes anomalies for human audit triage and does not detect confirmed legal fraud.

<img width="1914" height="934" alt="Vendor Invoice Intelligence Portal — Executive Dashboard" src="https://github.com/user-attachments/assets/861ee8c5-c08b-46a6-9914-96a9471d2d64" />

---

<a id="system-architecture--directory-structure"></a>
## 🏗️ 2. System Architecture & Directory Structure

```
Machine Learning Project/
├── app.py                                  # Interactive Streamlit portal
├── README.md                               # Project documentation & operational runbook
├── data/inventory.db                       # SQLite database (5 operational tables)
├── freight_cost_prediction/                # Freight regression pipeline & corrected RMSE evaluation
├── invoice_flagging/                       # Leakage-free Random Forest pipeline & 5-fold CV tuning
├── inference/                              # Validated standalone inference modules (freight & risk)
├── models/                                 # Serialized model artifacts (Linear Regression & Pipeline)
├── notebooks/                              # Research & evaluation notebooks (exploratory & final audit)
└── tests/                                  # Comprehensive automated test suite (32 tests)
```

---

<a id="database-schema--data-sources"></a>
## 🗄️ 3. Database Schema & Data Sources

The primary data store is [`data/inventory.db`](data/inventory.db), an SQLite database containing 5 operational tables:

| Table | Total Rows | Role in System | Key Columns Used |
|---|---|---|---|
| `vendor_invoice` | 5,543 | Header-level invoice records | `PONumber` (PK, 1:1), `Dollars`, `Quantity`, `Freight`, `Approval`, `InvoiceDate`, `PODate` |
| `purchases` | 2,372,474 | Line-item warehouse receipts | `PONumber` (FK, 1:M), `Quantity`, `Dollars`, `Brand`, `PODate`, `ReceivingDate` |
| `purchase_prices` | 12,261 | Catalog price references | *Unused in current ML pipelines* |
| `begin_inventory` / `end_inventory` | ~431k | Periodic store inventory levels | *Unused in current ML pipelines* |

---

<a id="models--methodological-framework"></a>
## ⚙️ 4. Models & Methodological Framework

<a id="freight-cost-prediction-model"></a>
### 4.1 Freight Cost Prediction Model
* **Algorithm**: Ordinary Least Squares (OLS) Linear Regression (`models/predict_freight_model.pkl`).
* **Input Feature**: `Dollars` (Invoice spend) $\rightarrow$ **Target**: `Freight` (Continuous shipping charge).
* **Fitted Linear Equation**:
  $$\text{Expected Freight} = 5.008021 + 0.005015 \times \text{Dollars}$$
* **Rate Card Interpretation**: Captures a standard logistics tariff of a \$5.00 base charge plus approximately 0.50% (50 cents per \$100) of invoice value.
* **Cryptographic SHA-256 Hash**: `2fab0379bde8a5647b0ab72451334fda14d99aef29d3a0fd873b8fae4e705ab3` (verified 100% frozen).

#### Holdout Test Performance (80/20 Split, N=1,109)
* **Mean Absolute Error (MAE)**: **\$24.11** | **Root Mean Squared Error (RMSE)**: **\$124.72** | **$R^2$**: **96.99%**
* **Weighted Absolute Percentage Error (WAPE)**: **8.20%** across total freight spend ($\sum |y - \hat{y}| / \sum y$).
* **MAPE Metric Policy (190.72%)**: 84 holdout invoices have actual freight under \$1.00 (down to \$0.03). With an intercept of \$5.008, a \$0.05 shipment yields an individual error of 9,920%, distorting unweighted arithmetic MAPE. Volume-weighted **WAPE (8.20%)** represents true institutional financial exposure.

#### Residual Analysis & Heteroskedasticity
| Spend Tier | Invoices | Mean Actual | Mean Pred | MAE | Residual Behavior |
|---|---|---|---|---|---|
| **Low (< \$500)** | 189 (17.0%) | \$1.52 | \$6.14 | \$4.99 | Systematically overpredicted (97.4% of cases) due to \$5.01 baseline intercept. |
| **Medium (\$500 – \$5,000)** | 381 (34.4%) | \$9.67 | \$14.43 | \$5.20 | Stable linear tracking; minor overprediction (-\$4.76 mean residual, WAPE 53.8%). |
| **High (≥ \$5,000)** | 539 (48.6%) | \$597.86 | \$595.98 | \$44.17 | Unbiased mean; residual variance expands (RMSE \$178.81; max outlier +\$3,680). |

<img width="1911" height="940" alt="Freight Cost Prediction UI & Estimation Breakdown" src="https://github.com/user-attachments/assets/3d9e5b46-3879-4cc0-b7cd-a5fa8696c3d1" />

---

<a id="invoice-risk-flagging-framework"></a>
### 4.2 Invoice Risk Flagging Framework
The invoice risk pipeline decouples deterministic accounting verification from statistical machine learning classification:

1. **Deterministic 3-Way Match Layer (Reconciliation Engine)**:
   * **Quantity Discrepancy**: `|invoice_quantity - total_item_quantity| > 0` units.
   * **Dollar Discrepancy**: `|invoice_dollars - total_item_dollars| > $5.00`.
   * *Scope*: Verifies explicit mathematical errors; operational reliability depends on warehouse dock scan quality and tolerance thresholds.

2. **Statistical Pattern Classification Layer (Pipeline Model)**:
   * **Architecture**: `sklearn.pipeline.Pipeline` encapsulating `StandardScaler` and `RandomForestClassifier` (200 trees, min_samples_split=5).
   * **Fold Isolation**: `StandardScaler` is fitted strictly within each cross-validation training fold, preventing validation fold leakage.
   * **Features (5 variables)**: `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars`.
   * **Target Label Generation (Synthetic Ground Truth)**:
     $$\text{flag\_invoice} = 1 \iff |\text{invoice\_dollars} - \text{total\_item\_dollars}| > \$5.00 \quad \lor \quad \text{avg\_receiving\_delay} > 10.0\text{ days}$$
     *Heuristic Nature*: Ground truth labels reflect business rules rather than confirmed legal fraud or audit dispute records. The model serves as an operational review triage filter.

<img width="1918" height="780" alt="Invoice Risk Flagging" src="https://github.com/user-attachments/assets/6761ca6f-ada9-4ccd-9f6c-3a6e614da0de" />

#### Multi-Partition Holdout Evaluation Comparison
| Partition Strategy | Samples (Train / Test) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Key Evaluation Insight |
|---|---|---|---|---|---|---|---|---|
| **Historical Baseline (Seed 42)** | 4,434 / 1,109 | **88.82%** | **95.14%** | **71.35%** | **0.8155** | **0.8721** | **0.8775** | Baseline holdout (TN=711, FP=14, FN=110, TP=274). |
| **Fresh Stratified (Seed 2024)** | 4,434 / 1,109 | **89.90%** | **97.08%** | **71.89%** | **0.8261** | **0.8715** | **0.8735** | Completely untouched split; confirms high precision stability (TN=731, FP=8). |
| **Vendor-Grouped Holdout** | 4,388 / 1,155 | **87.97%** | **91.86%** | **70.21%** | **0.7959** | **0.8611** | **0.8588** | Evaluates **26 unseen vendors**; proves model does not memorize vendor IDs. |

#### Root Cause of False Negatives (Omission of Receiving Delay)
* 27.84% of database flags (515/1,850) are triggered solely by delivery delays (`avg_receiving_delay > 10` days) with zero dollar discrepancy.
* Because `avg_receiving_delay` was omitted from the 5-feature schema, the model misses 98 of 99 delay-only holdout cases, explaining **89.09% of all model false negatives** (98/110 FNs).
* In contrast, on invoices with actual dollar discrepancies, recall reaches **95.79%** (274 of 286 caught).

#### Notebook Leakage Audit Trail
* [`notebooks/Invoice Flagging.ipynb`](notebooks/Invoice%20Flagging.ipynb): Preserved as an immutable historical research audit trail documenting pre-split feature selection, test-set pruning, and pre-CV scaling.
* [`notebooks/Invoice_Flagging_Final_Evaluation.ipynb`](notebooks/Invoice_Flagging_Final_Evaluation.ipynb): Dedicated audit notebook executing a leakage-free workflow with pre-split isolation, pipeline CV, and frozen holdout evaluation.

---

<a id="feature-status--implementation-matrix"></a>
## 📊 5. Feature Status & Implementation Matrix

| Capability / Feature | Category | Implementation Status | Description |
|---|---|---|---|
| Linear Regression Freight Inference | Freight | **Implemented & Tested** | Predicts expected freight from `Dollars`. Rejects negative/null values (13 tests). |
| Freight Input Preservation | Freight | **Implemented & Tested** | Preserves non-modeled inputs (such as `Quantity`) through inference outputs. |
| Corrected Evaluation RMSE | Freight | **Implemented & Tested** | Replaced legacy MSE reporting bug with true $\sqrt{\text{MSE}}$ across evaluation scripts. |
| 3-Way Accounting Matching | Risk Flagging | **Implemented & Tested** | Deterministic dollar (> \$5.00) and quantity variance checks before ML scoring. |
| Random Forest Pipeline Inference | Risk Flagging | **Implemented & Tested** | Encapsulated sklearn Pipeline (`StandardScaler` + `RandomForest`) with CV isolation. |
| Streamlit Interactive Portal | UI / Dashboard | **Implemented & Tested** | Presentation-ready dashboard with robust input forms and decision rationale cards. |
| Delivery Delay Feature at Inference | Risk Flagging | **Planned Roadmap** | Add dock receipt timestamps to inference schema to capture delay-only flags. |
| ERP / WMS Direct Ingestion | Core System | **Planned Roadmap** | Ingestion hooks for EDI 810 invoices and enterprise payment registers. |

---

<a id="setup--operational-runbook"></a>
## 🚀 6. Setup & Operational Runbook

### Environment & Automated Testing
```powershell
# Run the complete test suite (32 unit & consistency tests)
python tests/run_all_tests.py
```
*Output: `Ran 32 tests in 2.5s. OK (Failures: 0, Errors: 0)`.*

### Standalone Inference Smoke Tests
```powershell
# 1. Freight Cost Prediction
python inference/predict_freight.py

# 2. Invoice Risk Flagging
python inference/predict_invoice_flag.py
```

### Launch Streamlit Portal
```powershell
streamlit run app.py
```
*Access the local web dashboard at `http://localhost:8501`.*

---

<a id="known-limitations--future-improvements"></a>
## ⚠️ 7. Known Limitations & Future Improvements

1. **Synthetic Heuristic Labels**: Ground truth is generated via programmatic rules rather than verified accounting dispute logs. *Future Improvement*: Train on historical enterprise accounts payable dispute outcomes.
2. **Freight Intercept Overprediction**: The \$5.008 intercept overpredicts low-spend invoices (< \$500). *Future Improvement*: Implement a tiered piecewise rate card when authorized by logistics business owners.
3. **Receiving Delay Data Collection**: Delivery delays drive 27.8% of synthetic risk flags but are evaluated outside the ML vector. *Future Improvement*: Add purchase order and receiving dates to the review form to automate SLA tracking.
