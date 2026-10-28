import streamlit as st
import pandas as pd
import numpy as np

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# ============================================================
# Page Configuration & Professional Styling
# ============================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for restrained executive dark theme
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Header Card */
    .portal-header {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .portal-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f9fafb;
        margin: 0;
    }
    .portal-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
    }
    .portal-description {
        font-size: 0.875rem;
        color: #d1d5db;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Metric & Information Cards */
    .card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .card-caption {
        font-size: 0.775rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    /* Notice and Callout Containers */
    .notice-box {
        background-color: #0f172a;
        border-left: 3px solid #38bdf8;
        padding: 0.85rem 1.15rem;
        border-radius: 4px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    .disclaimer-text {
        font-size: 0.775rem;
        color: #64748b;
        line-height: 1.4;
        margin-top: 1rem;
    }
    
    /* Status Pills */
    .status-badge-flagged {
        background-color: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    .status-badge-clean {
        background-color: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar Configuration & Project Navigation
# ============================================================

st.sidebar.markdown("### System Navigation")

selected_module = st.sidebar.radio(
    "Select Analytics Module",
    [
        "Freight Cost Prediction",
        "Invoice Risk Assessment"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Architecture")
st.sidebar.caption(
    "**Core Database:** `data/inventory.db`\n\n"
    "**Freight Model:** Linear Regression (OLS)\n"
    "• Input: `Dollars`\n"
    "• Target: `Freight`\n"
    "• Benchmark: R² = 96.99%, MAE = \\$24.11, RMSE = \\$124.72\n\n"
    "**Risk Classifier:** Random Forest Pipeline (200 Trees)\n"
    "• Preprocessing: `StandardScaler` inside Pipeline\n"
    "• Objective: Accounting & Pattern Triage\n"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Vendor Invoice Intelligence Portal**\n"
    "Enterprise Decision Support System\n"
    "Academic Demonstration Environment"
)


# ============================================================
# Main Header
# ============================================================

st.markdown("""
<div class="portal-header">
    <div class="portal-title">Vendor Invoice Intelligence Portal</div>
    <div class="portal-subtitle">Freight Cost Prediction & Invoice Risk Assessment</div>
    <div class="portal-description">
        An enterprise analytics platform designed to support procurement and accounts payable teams. 
        The system applies statistical regression to forecast expected freight charges and combines 
        deterministic three-way matching with machine learning to identify invoices warranting manual review.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Module 1: Freight Cost Prediction
# ============================================================

if selected_module == "Freight Cost Prediction":

    st.markdown("### Freight Cost Prediction")
    st.markdown(
        "Estimate expected freight costs from invoice spend based on historical commercial tariffs. "
        "The model uses Ordinary Least Squares Linear Regression to establish a quantitative baseline "
        "for budgeting and logistics variance analysis."
    )

    col_form, col_context = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("#### Invoice Input Parameters")
        
        with st.form("freight_form"):
            dollars = st.number_input(
                "Invoice Dollar Total ($)",
                min_value=1.0,
                max_value=10_000_000.0,
                value=18500.0,
                step=100.0,
                help="Total billed dollar amount on the vendor invoice."
            )

            quantity = st.number_input(
                "Invoice Quantity (Units)",
                min_value=1,
                max_value=10_000_000,
                value=1200,
                step=10,
                help="Total unit quantity on the invoice. Preserved as an operational record."
            )

            submit_freight = st.form_submit_button("Calculate Expected Freight", use_container_width=True)

        if submit_freight:
            try:
                input_payload = {
                    "Quantity": [quantity],
                    "Dollars": [dollars]
                }

                result_df = predict_freight_cost(input_payload)
                predicted_val = float(result_df["Predicted_Freight"].iloc[0])
                effective_rate = (predicted_val / dollars) * 100 if dollars > 0 else 0.0

                st.markdown("#### Prediction Output")
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Estimated Freight Cost</div>
                        <div class="card-value">${predicted_val:,.2f}</div>
                        <div class="card-caption">Baseline rate projection</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m_col2:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Effective Freight Rate</div>
                        <div class="card-value">{effective_rate:.2f}%</div>
                        <div class="card-caption">Percentage of invoice spend</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="notice-box">
                    <strong>Model Evaluation & Methodology:</strong><br>
                    The saved Ordinary Least Squares (OLS) Linear Regression model fits the contract tariff equation:
                    <br><code>Expected Freight = \\$5.0080 + 0.005015 × Dollars</code> (~0.50% freight surcharge with a \\$5.01 base charge).
                    On the held-out test split, this baseline achieves an MAE of <strong>\\$24.11</strong> and a true RMSE of <strong>\\$124.72</strong> (R² = 96.99%).
                </div>
                """, unsafe_allow_html=True)

            except Exception as exc:
                st.error("Unable to calculate freight estimation. Please check input parameters.")

    with col_context:
        st.markdown("#### Model Specification & Context")
        st.markdown("""
        The freight cost model uses **Ordinary Least Squares (OLS) Linear Regression** trained on historical vendor invoices.

        * **Mathematical Form:** $\\hat{y} = \\beta_0 + \\beta_1 X_{\\text{Dollars}}$
        * **Input Feature:** `Dollars` (Total invoice dollar spend)
        * **Target:** `Freight` (Continuous freight charge in USD)
        * **Feature Role:** Invoice spend provides a robust predictor for baseline freight tariffs across bulk vendor shipments.

        <div class="disclaimer-text">
            <strong>Auditing Disclosure:</strong> Expected freight estimates provide an analytical baseline for budgeting and variance detection. An estimate does not, by itself, determine whether a vendor's billed freight is legally fraudulent or contractually non-compliant.
        </div>
        """, unsafe_allow_html=True)

    # Workflow Architecture Section
    st.markdown("---")
    with st.expander("System Workflow: How Freight Estimation Works", expanded=False):
        st.markdown("""
        ```
        [User Inputs: Dollars ($), Quantity]
                      │
                      ▼
        [Validation: Numeric, Finite, Non-Negative]
                      │
                      ▼
        [Linear Regression: 0.005015 × Dollars + 5.008]
                      │
                      ▼
        [Output: Estimated Freight Cost ($), Rate %]
        ```
        1. **Input Normalization**: Ingests invoice dollars and quantity while verifying finite bounds.
        2. **Feature Isolation**: Extracts the `Dollars` feature for the trained scikit-learn regressor.
        3. **Tariff Projection**: Projects standard freight charge based on historical rate card coefficients.
        """)


# ============================================================
# Module 2: Invoice Risk Assessment
# ============================================================

else:

    st.markdown("### Invoice Risk Assessment")
    st.markdown(
        "Evaluate incoming vendor invoices for review prioritization. The system pairs deterministic "
        "three-way accounting reconciliation (comparing billed totals with warehouse receipts) with an "
        "ensemble machine learning classifier to identify anomalies before payment approval."
    )

    col_form, col_context = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("#### Invoice & Receiving Parameters")

        with st.form("invoice_risk_form"):
            st.caption("**1. Billed Invoice Header**")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                inv_dollars = st.number_input(
                    "Billed Invoice Dollars ($)",
                    min_value=1.0,
                    max_value=10_000_000.0,
                    value=352.95,
                    step=10.0,
                    help="Total dollar amount requested on the vendor's invoice."
                )
            with col_b2:
                inv_quantity = st.number_input(
                    "Billed Invoice Quantity (Units)",
                    min_value=1,
                    max_value=10_000_000,
                    value=50,
                    step=1,
                    help="Total number of units billed on the vendor's invoice."
                )

            freight_cost = st.number_input(
                "Billed Freight Cost ($)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=1.73,
                step=1.0,
                help="Freight charges billed on this invoice."
            )

            st.caption("**2. Warehouse Receipts (Goods Received)**")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rec_dollars = st.number_input(
                    "Received Item Dollars ($)",
                    min_value=1.0,
                    max_value=10_000_000.0,
                    value=2476.0,
                    step=10.0,
                    help="Total dollar value of items officially logged as received by the warehouse."
                )
            with col_r2:
                rec_quantity = st.number_input(
                    "Received Item Quantity (Units)",
                    min_value=1,
                    max_value=10_000_000,
                    value=162,
                    step=1,
                    help="Total physical units logged as received at receiving docks."
                )

            submit_risk = st.form_submit_button("Assess Invoice Risk", use_container_width=True)

        if submit_risk:
            try:
                input_payload = {
                    "invoice_quantity": [inv_quantity],
                    "invoice_dollars": [inv_dollars],
                    "Freight": [freight_cost],
                    "total_item_quantity": [rec_quantity],
                    "total_item_dollars": [rec_dollars]
                }

                result_df = predict_invoice_flag(input_payload)

                is_flagged = bool(result_df["Predicted_Flag"].iloc[0])
                recommendation = str(result_df["Review_Recommendation"].iloc[0])
                risk_score = float(result_df["Risk_Score"].iloc[0])
                reasons = result_df["Review_Reasons"].iloc[0]
                acct_mismatch = bool(result_df["Accounting_Discrepancy_Detected"].iloc[0])
                dollar_diff = abs(inv_dollars - rec_dollars)

                st.markdown("#### Assessment Findings")

                if is_flagged or acct_mismatch:
                    st.markdown("""
                    <div class="status-badge-flagged">
                        Action: Flagged for Manual Review
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("**Recommendation:** Hold payment authorization. Route to accounts payable for discrepancy reconciliation.")
                else:
                    st.markdown("""
                    <div class="status-badge-clean">
                        Action: Not Flagged by Current Model — Standard Processing
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("**Recommendation:** Invoice matches receiving thresholds and exhibits standard pattern baselines.")

                # Metrics display
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Three-Way Dollar Variance</div>
                        <div class="card-value">${dollar_diff:,.2f}</div>
                        <div class="card-caption">Billed vs. Received difference</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m2:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Classifier Pattern Score</div>
                        <div class="card-value">{risk_score * 100:.1f}%</div>
                        <div class="card-caption">Random Forest risk index</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("**Identified Factors & Evidence:**")
                for r in reasons:
                    clean_r = str(r).replace("$", r"\$")
                    st.markdown(f"• {clean_r}")

            except Exception as exc:
                st.error("Unable to evaluate invoice risk. Please check input parameters.")

    with col_context:
        st.markdown("#### Methodological Framework")
        st.markdown("""
        The risk assessment module applies a two-tier evaluation framework:

        1. **Deterministic Accounting Reconciliation:**
           * Physical Quantity Discrepancy: `|Billed Quantity - Received Quantity| > 0` (units mismatch).
           * Dollar Amount Variance: `|Billed Dollars - Received Dollars| > $5.00` (discrepancy exceeding tolerance).
        2. **Supervised Classification Triage:**
           * **Algorithm:** Random Forest Classifier Pipeline (200 estimators)
           * **Standardized Features:** Billed Quantity, Billed Dollars, Freight, Received Quantity, Received Dollars.
           * **Training Target:** Rule-derived binary flag identifying significant variances.

        <div class="disclaimer-text">
            <strong>Methodology Disclosure:</strong> Training labels for this classifier were derived from predefined synthetic business rules rather than independently audited fraud or legal dispute records. Consequently, predictions represent priority triage recommendations and do not establish verified fraud or legal misconduct.
        </div>
        """, unsafe_allow_html=True)

    # Workflow Architecture Section
    st.markdown("---")
    with st.expander("System Workflow: How Invoice Risk Triage Works", expanded=False):
        st.markdown("""
        ```
        [Invoice Billed Data + Warehouse Receipts]
                           │
                           ▼
        [Step 1: 3-Way Match Check (|Invoice$ - Item$| > $5)]
                           │
                           ▼
        [Step 2: Feature Standardization via StandardScaler]
                           │
                           ▼
        [Step 3: Random Forest Classifier Prediction]
                           │
                           ▼
        [Output: Flagged for Review vs. Standard Processing]
        ```
        1. **Matching Filter**: Evaluates explicit arithmetic variances between invoice bills and dock receipts.
        2. **Preprocessing**: Transforms the 5-feature vector using the training partition's scaler.
        3. **Decision Support**: Assigns priority routing for accounts payable review teams.
        """)
