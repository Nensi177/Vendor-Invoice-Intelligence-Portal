from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "predict_flag_invoice.pkl"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

REQUIRED_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]


def load_artifacts(model_path=None, scaler_path=None):
    """
    Load trained classifier and standard scaler from disk.
    Defaults to models/predict_flag_invoice.pkl and models/scaler.pkl.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if scaler_path is None:
        scaler_path = DEFAULT_SCALER_PATH

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier model artifact not found at: {model_path.resolve()}\n"
            "Please ensure 'predict_flag_invoice.pkl' exists in the 'models/' directory."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found at: {scaler_path.resolve()}\n"
            "Please ensure 'scaler.pkl' exists in the 'models/' directory."
        )

    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load classifier model from '{model_path.resolve()}': {e}"
        ) from e

    try:
        with open(scaler_path, "rb") as f:
            scaler = joblib.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load scaler from '{scaler_path.resolve()}': {e}"
        ) from e

    return model, scaler


def predict_invoice_flag(input_data, model_path=None, scaler_path=None):
    """
    Evaluate vendor invoice risk for manual review.

    Separates deterministic accounting discrepancy checks from statistical
    model predictions.

    Parameters
    ----------
    input_data : dict or pd.DataFrame
        Input data containing the required 5 invoice features.
    model_path : str or Path, optional
        Custom path to classifier model artifact.
    scaler_path : str or Path, optional
        Custom path to scaler artifact.

    Returns
    -------
    pd.DataFrame
        DataFrame with input columns, predicted flags, probabilities,
        accounting checks, and human-readable review recommendations.
    """
    model, scaler = load_artifacts(model_path, scaler_path)

    if isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    elif isinstance(input_data, dict):
        normalized_data = {}
        for k, v in input_data.items():
            if np.isscalar(v) or isinstance(v, (int, float, str)):
                normalized_data[k] = [v]
            else:
                normalized_data[k] = list(v)
        input_df = pd.DataFrame(normalized_data)
    else:
        raise TypeError(
            f"input_data must be a dict or pandas DataFrame, got {type(input_data).__name__}."
        )

    if input_df.empty:
        raise ValueError("Input data cannot be empty.")

    # Validate presence of required features
    missing_features = [f for f in REQUIRED_FEATURES if f not in input_df.columns]
    if missing_features:
        raise ValueError(
            f"Input data is missing required feature(s): {missing_features}. "
            f"Required features: {REQUIRED_FEATURES}. "
            f"Available columns: {list(input_df.columns)}"
        )

    # Validate numeric types, non-null, finite, and non-negative
    for feat in REQUIRED_FEATURES:
        col_series = pd.to_numeric(input_df[feat], errors="coerce")
        if col_series.isna().any():
            raise ValueError(
                f"Invalid input: Feature '{feat}' contains null or non-numeric values."
            )
        if np.isinf(col_series).any():
            raise ValueError(
                f"Invalid input: Feature '{feat}' contains infinite values."
            )
        if (col_series < 0).any():
            raise ValueError(
                f"Invalid input: Feature '{feat}' contains negative values. "
                "Invoice quantities, dollar amounts, and freight must be non-negative."
            )

    # Strictly extract features in exact training order (prevents target leakage)
    X = input_df[REQUIRED_FEATURES].copy()

    # Apply preprocessing and model prediction
    X_scaled = scaler.transform(X)
    raw_predictions = model.predict(X_scaled).astype(int)

    # Extract risk probability if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        probabilities = raw_predictions.astype(float)

    input_df["Predicted_Flag"] = raw_predictions
    input_df["Risk_Score"] = np.round(probabilities, 4)

    # Deterministic accounting discrepancy checks
    dollar_diff = (input_df["invoice_dollars"] - input_df["total_item_dollars"]).abs()
    qty_diff = (input_df["invoice_quantity"] - input_df["total_item_quantity"]).abs()

    has_accounting_discrepancy = dollar_diff > 5.0
    has_quantity_discrepancy = qty_diff > 0

    input_df["Accounting_Discrepancy_Detected"] = has_accounting_discrepancy

    # Build human-readable recommendations and reasons (never claiming confirmed fraud)
    recommendations = []
    reasons_list = []

    for i in range(len(input_df)):
        reasons = []
        is_model_flagged = raw_predictions[i] == 1
        is_acct_flagged = has_accounting_discrepancy.iloc[i]
        is_qty_flagged = has_quantity_discrepancy.iloc[i]

        if is_acct_flagged:
            diff_val = dollar_diff.iloc[i]
            inv_d = input_df['invoice_dollars'].iloc[i]
            item_d = input_df['total_item_dollars'].iloc[i]
            reasons.append(
                f"Accounting Discrepancy: Billed dollars (${inv_d:,.2f}) differs from item total (${item_d:,.2f}) by ${diff_val:,.2f}"
            )

        if is_qty_flagged:
            q_val = qty_diff.iloc[i]
            inv_q = input_df['invoice_quantity'].iloc[i]
            item_q = input_df['total_item_quantity'].iloc[i]
            reasons.append(
                f"Quantity Variance: Billed quantity ({inv_q:,.0f}) differs from received item quantity ({item_q:,.0f}) by {q_val:,.0f} units"
            )

        if is_model_flagged:
            reasons.append(
                f"Pattern Alert: Classifier flagged statistical risk pattern (Risk Score: {probabilities[i]*100:.1f}%)"
            )

        if is_acct_flagged or is_model_flagged:
            recommendation = "Manual Review Required"
        else:
            recommendation = "Standard Processing"
            reasons.append("No significant accounting discrepancies or high-risk patterns detected")

        recommendations.append(recommendation)
        reasons_list.append(reasons)

    input_df["Review_Recommendation"] = recommendations
    input_df["Review_Reasons"] = reasons_list

    return input_df


if __name__ == "__main__":
    sample_invoices = {
        "invoice_quantity": [50, 1200],
        "invoice_dollars": [352.95, 18500.0],
        "Freight": [1.73, 98.0],
        "total_item_quantity": [162, 1200],
        "total_item_dollars": [2476.0, 18500.0]
    }

    results = predict_invoice_flag(sample_invoices)
    print("Invoice Flagging Predictions:")
    display_cols = [
        "invoice_dollars", "total_item_dollars",
        "Predicted_Flag", "Risk_Score",
        "Accounting_Discrepancy_Detected", "Review_Recommendation"
    ]
    print(results[display_cols])
    print("\nDetailed Reasons for Invoice 0:")
    for r in results["Review_Reasons"].iloc[0]:
        print(f"  - {r}")
    print("\nDetailed Reasons for Invoice 1:")
    for r in results["Review_Reasons"].iloc[1]:
        print(f"  - {r}")
