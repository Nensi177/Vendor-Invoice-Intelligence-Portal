# Vendor Invoice Intelligence Portal: Freight Forecasting & Invoice Risk Flagging

An internal analytics and machine learning system for enterprise supply chain operations. The portal provides automated freight charge estimation and early-warning invoice risk assessment to help procurement and finance teams prevent financial leakage, identify accounting mismatches, and prioritize manual invoice reviews.

---

## 1. Project Objective

In accounts payable and supply chain operations, manual verification of vendor invoices across thousands of line items is labor-intensive and error-prone. This system provides two capabilities:

1. **Freight Cost Prediction**: Estimates expected freight charges from invoice dollar totals using the verified **Linear Regression** model, establishing a baseline rate to detect excessive logistics surcharges.
2. **Invoice Risk Flagging**: Evaluates vendor invoices for manual review using a combination of deterministic 3-way accounting checks (comparing billed dollars and quantities against warehouse receipts) and statistical risk pattern classification.

> [!NOTE]
> The freight model predicts expected freight based on invoice spend under the baseline commercial rate card. It does not, by itself, prove legal fraud or contractual breach. Similarly, the risk classifier flags anomalous patterns for human review and must not be described as detecting confirmed fraud.

---

## 2. System Architecture & Directory Structure

```
Machine Learning Project/
├── app.py                                  # Interactive Streamlit dashboard
├── README.md                               # Project documentation & operational runbook
├── data/
│   └── inventory.db                        # SQLite database (5 tables)
├── freight_cost_prediction/                # Freight regression pipeline
│   ├── __init__.py                         # Package anchor
│   ├── data_preprocessing.py               # Data loading (inventory.db), feature extraction, split
│   ├── model_evaluation.py                 # Regressor definitions (LinearRegression) & corrected RMSE metrics
│   └── train.py                            # Training orchestrator saving to models/
├── invoice_flagging/                       # Invoice risk classification pipeline
│   ├── __init__.py                         # Package anchor
│   ├── data_preprocessing.py               # SQL aggregation CTE, risk labeling, StandardScaler
│   ├── modeling_evaluation.py              # RandomForestClassifier (GridSearchCV, 5-fold CV)
│   └── train.py                            # Training orchestrator saving classifier & scaler
├── inference/                              # Standalone prediction modules
│   ├── __init__.py                         # Package anchor
│   ├── predict_freight.py                  # Validated inference wrapper for freight model
│   └── predict_invoice_flag.py             # Validated inference wrapper for risk classifier
├── models/                                 # Serialized model artifacts
│   ├── predict_freight_model.pkl           # LinearRegression artifact (Dollars -> Freight)
│   ├── predict_flag_invoice.pkl            # RandomForestClassifier artifact
│   └── scaler.pkl                          # Fitted StandardScaler (5 features)
├── notebooks/                              # Research & exploratory notebooks
│   ├── Predicting Freight Cost.ipynb       # Freight analysis & Linear Regression prototyping
│   └── Invoice Flagging.ipynb              # Exploratory analysis for invoice risk flagging
└── tests/                                  # Automated test suite (29 tests)
    ├── __init__.py                         # Package anchor
    ├── test_predict_freight.py             # 13 tests: loading, validation, batch, preservation
    ├── test_predict_invoice_flag.py        # 12 tests: loading, features, accounting checks, isolation
    ├── test_pipeline_consistency.py        # 4 tests: paths, true RMSE calculation, custom scalers
    └── run_all_tests.py                    # Unified test discovery and execution script
```

---

## 3. Database Schema & Join Mechanics

The primary data store is [`data/inventory.db`](data/inventory.db), an SQLite database containing 5 tables:

| Table | Total Rows | Role in System | Key Columns Used |
|---|---|---|---|
| `vendor_invoice` | 5,543 | Header-level invoice records | `PONumber` (PK, 1:1), `Dollars`, `Quantity`, `Freight`, `Approval`, `InvoiceDate`, `PODate`, `PayDate` |
| `purchases` | 2,372,474 | Line-item warehouse receipts | `PONumber` (FK, 1:M), `Quantity`, `Dollars`, `Brand`, `PODate`, `ReceivingDate` |
| `purchase_prices` | 12,261 | Catalog price references | *Unused in current pipelines* |
| `begin_inventory` | 206,529 | Historical store inventory | *Unused in current pipelines* |
| `end_inventory` | 224,489 | Period-end store inventory | *Unused in current pipelines* |

### Join Cardinality & Pre-Aggregation
* **Purchase Order Uniqueness**: Every purchase order in `vendor_invoice` has a unique `PONumber` (5,543 distinct invoices).
* **Line-Item Aggregation**: `purchases` contains an average of 428 line items per purchase order. To avoid multi-million-row cartesian multiplication, line items are aggregated at the `PONumber` level inside a CTE before executing a `LEFT JOIN`:
  ```sql
  WITH purchase_agg AS (
      SELECT
          p.PONumber,
          COUNT(DISTINCT p.Brand) AS total_brands,
          SUM(p.Quantity) AS total_item_quantity,
          SUM(p.Dollars) AS total_item_dollars,
          AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
      FROM purchases p
      GROUP BY p.PONumber
  )
  SELECT vi.*, pa.*
  FROM vendor_invoice vi
  LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber;
  ```

---

## 4. Models & Methodological Framework

### 4.1 Freight Cost Prediction Model
* **Algorithm**: Ordinary Least Squares Linear Regression (`sklearn.linear_model.LinearRegression`).
* **Input Feature**: `Dollars` (Invoice dollar amount).
* **Target**: `Freight` (Continuous dollar cost).
* **Artifact**: [`models/predict_freight_model.pkl`](models/predict_freight_model.pkl).
* **Fitted Linear Equation**:
  $$\text{Expected Freight} = 5.008021 + 0.005015 \times \text{Dollars}$$
* **Rate Card Interpretation**: The model captures a commercial logistics formula of a \$5.00 base charge plus approximately 0.50% (50 cents per \$100) of invoice value.
* **Verified Test Metrics (80/20 Holdout Split, N=1,109)**:
  * **Mean Absolute Error (MAE)**: **\$24.11**
  * **Mean Squared Error (MSE)**: **\$15,555.32**
  * **True Root Mean Squared Error (RMSE)**: $\sqrt{15555.32} = \mathbf{\$124.72}$
  * **Coefficient of Determination ($R^2$)**: **96.99%**
  * **Weighted Absolute Percentage Error (WAPE)**: **8.20%** across total freight spend.

### 4.2 Invoice Risk Flagging Framework
The invoice risk pipeline decouples deterministic accounting verification from statistical classification:

1. **Deterministic 3-Way Match Layer**:
   * Evaluates physical quantity discrepancy: $\Delta_{\text{Qty}} = |\text{invoice\_quantity} - \text{total\_item\_quantity}|$.
   * Evaluates dollar mismatch: $\Delta_{\text{Dollars}} = |\text{invoice\_dollars} - \text{total\_item\_dollars}| > \$5.00$.
2. **Statistical Pattern Classification Layer**:
   * **Algorithm**: `RandomForestClassifier` (300 estimators, Gini criterion).
   * **Input Features** (scaled via `StandardScaler`):
     1. `invoice_quantity`
     2. `invoice_dollars`
     3. `Freight`
     4. `total_item_quantity`
     5. `total_item_dollars`
   * **Artifacts**: [`models/predict_flag_invoice.pkl`](models/predict_flag_invoice.pkl) and [`models/scaler.pkl`](models/scaler.pkl).
   * **Output**: Binary prediction (`Predicted_Flag`), continuous risk probability (`Risk_Score`), and structured human-readable review reasons.

---

## 5. Feature Status & Implementation Matrix

| Capability / Feature | Category | Implementation Status | Description |
|---|---|---|---|
| Linear Regression Freight Inference | Freight | **Implemented & Tested** | Predicts expected freight from `Dollars`. Rejects negative/null values. Verified with 13 unit tests. |
| Freight Input Preservation | Freight | **Implemented & Tested** | Preserves original inputs (such as `Quantity`) when passed alongside `Dollars`. |
| Corrected Evaluation RMSE | Freight | **Implemented & Tested** | Replaced raw MSE reporting bug with true $\sqrt{\text{MSE}}$ in evaluation scripts. |
| 3-Way Accounting Matching | Risk Flagging | **Implemented & Tested** | Detects billed vs. received dollar discrepancies ($> \$5$) and quantity variances. |
| Random Forest Risk Inference | Risk Flagging | **Implemented & Tested** | Scales 5 input features and outputs `Predicted_Flag` with continuous `Risk_Score`. |
| Dynamic Path Resolution | System Core | **Implemented & Tested** | Replaced hardcoded `/Users/DELL/...` paths with `pathlib.Path` relative to project root. |
| Streamlit Interactive Portal | UI / Dashboard | **Implemented & Tested** | Provides web forms for both modules with defensive error handling and clear reason breakdowns. |
| Delivery Delay Feature at Inference | Risk Flagging | **Implemented but Not Validated** | `avg_receiving_delay` is present in SQL aggregations but not collected on the UI form. |
| Delegation of Authority (> $250k) | Governance | **Implemented but Not Validated** | Database reflects executive sign-off (`Frank Delahunt`) on orders $\ge \$250,000$; rule codified in framework. |
| Automated ERP / WMS Ingestion | System Core | **Planned but Not Implemented** | Direct webhook/API ingestion of incoming EDI 810 invoices. |
| Anomaly Isolation Forest | Risk Flagging | **Planned but Not Implemented** | Unsupervised multi-attribute outlier scoring on vendor baselines. |

---

## 6. Setup & Operational Runbook

### Environment Requirements
* Python 3.10+ (tested on Python 3.13)
* Dependencies: `scikit-learn`, `pandas`, `numpy`, `joblib`, `streamlit`

### Running Automated Tests
Execute the complete test suite (29 tests across all modules):
```powershell
python tests/run_all_tests.py
```

### Running Standalone Inference Scripts

#### 1. Freight Cost Prediction
```powershell
python inference/predict_freight.py
```
*Expected Output*:
```
Freight Cost Predictions:
   Dollars  Predicted_Freight
0    18500               98.0
1     9000               50.0
2     3000               20.0
3      200                6.0
```

#### 2. Invoice Risk Flagging
```powershell
python inference/predict_invoice_flag.py
```
*Expected Output*:
```
Invoice Flagging Predictions:
   invoice_dollars  total_item_dollars  Predicted_Flag  Risk_Score  Accounting_Discrepancy_Detected   Review_Recommendation
0           352.95             2476.00               1      0.9850                             True  Manual Review Required
1         18500.00            18500.00               0      0.0320                            False     Standard Processing
```

### Launching the Streamlit Web Application
```powershell
streamlit run app.py
```
Once launched, navigate to `http://localhost:8501` to access the interactive portal.

---

## 7. Known Limitations & Future Improvements

1. **Synthetic Heuristic Labels**:
   * The risk classifier was trained on heuristic pseudo-labels (`|invoice_dollars - total_item_dollars| > 5` and `avg_receiving_delay > 10`), not verified audit failure records.
   * *Future Improvement*: Ingest historical accounts payable audit logs or vendor credit adjustments to supervise the model on genuine commercial disputes.
2. **Freight Single-Variable Baseline**:
   * While the Linear Regression model achieves a high $R^2$ (96.99%) across the macro spend distribution, it uses an additive intercept (\$5.01) that overpredicts small invoices ($< \$500$) on a percentage basis.
   * *Future Improvement*: Evaluate piecewise rate cards or shipment-tier segmented models when authorized by logistics business owners.
3. **Delivery Delay Collection**:
   * Receiving delay accounts for 27.8% of synthetic risk flags, but receiving dates are not captured in the current Streamlit front-end.
   * *Future Improvement*: Add purchase order date and warehouse receiving date fields to the review form to enable lead-time SLA checks.

