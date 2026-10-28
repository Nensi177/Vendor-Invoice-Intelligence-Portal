import streamlit as st
import pandas as pd
import numpy as np

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# Header Section
# ============================================================

st.markdown("""
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning and accounting checks to:

- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Identify accounting discrepancies and reduce manual workload**
""")

st.divider()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**

- 📉 Improved cost forecasting
- 🧾 Reduced invoice anomalies & errors
- ⚙️ Faster finance operations
""")


# ============================================================
# Freight Cost Prediction
# ============================================================

if selected_model == "Freight Cost Prediction":

    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**

    Predict freight cost for a vendor invoice using
    **Invoice Dollars** (with Quantity recorded) to support
    budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "📦 Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button(
            "🔮 Predict Freight Cost"
        )

    if submit_freight:
        try:
            input_data = {
                "Quantity": [quantity],
                "Dollars": [dollars]
            }

            result_df = predict_freight_cost(input_data)
            prediction = result_df["Predicted_Freight"].iloc[0]

            st.success("Freight cost estimated successfully.")

            st.metric(
                label="📊 Estimated Freight Cost",
                value=f"${prediction:,.2f}"
            )
        except Exception as e:
            st.error(f"Error predicting freight cost: {e}")


# ============================================================
# Invoice Flag Prediction
# ============================================================

else:

    st.subheader("🧾 Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**

    Evaluate whether a vendor invoice should be
    **held for manual review** based on accounting
    discrepancies and abnormal risk patterns.
    """)

    with st.form("invoice_flag_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button(
            "🧠 Evaluate Invoice Risk"
        )

    if submit_flag:
        try:
            input_data = {
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars]
            }

            result_df = predict_invoice_flag(input_data)

            is_flagged = bool(result_df["Predicted_Flag"].iloc[0])
            recommendation = result_df["Review_Recommendation"].iloc[0]
            risk_score = result_df["Risk_Score"].iloc[0]
            reasons = result_df["Review_Reasons"].iloc[0]
            acct_discrepancy = bool(result_df["Accounting_Discrepancy_Detected"].iloc[0])

            if is_flagged or acct_discrepancy:
                st.warning(
                    f"⚠️ **Review Recommendation: {recommendation}**"
                )
            else:
                st.success(
                    f"✅ **Review Recommendation: {recommendation}**"
                )

            # Display metrics breakdown
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    label="🧠 Classifier Risk Score",
                    value=f"{risk_score * 100:.1f}%"
                )
            with col_m2:
                dollar_diff = abs(invoice_dollars - total_item_dollars)
                st.metric(
                    label="🧾 3-Way Dollar Variance",
                    value=f"${dollar_diff:,.2f}"
                )

            st.markdown("#### 📋 Assessment Details & Evidence")
            for r in reasons:
                st.write(f"- {r}")

        except Exception as e:
            st.error(f"Error evaluating invoice risk: {e}")
