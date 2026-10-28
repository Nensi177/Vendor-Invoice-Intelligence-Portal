from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "predict_freight_model.pkl"


def load_model(model_path=None):
    """
    Load trained freight cost prediction model from disk.
    Defaults to models/predict_freight_model.pkl relative to the project root.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Freight model artifact not found at: {model_path.resolve()}\n"
            "Please ensure 'predict_freight_model.pkl' exists in the 'models/' directory."
        )

    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load freight model from '{model_path.resolve()}': {e}"
        ) from e

    return model


def predict_freight_cost(input_data, model_path=None):
    """
    Predict freight cost for vendor invoices using invoice Dollars.

    Parameters
    ----------
    input_data : dict or pd.DataFrame
        Input data containing the required 'Dollars' column.
    model_path : str or Path, optional
        Custom path to the saved model artifact.

    Returns
    -------
    pd.DataFrame
        DataFrame containing input columns and 'Predicted_Freight'.
    """
    model = load_model(model_path)

    if isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    elif isinstance(input_data, dict):
        # Normalize scalar values to lists for single-record support
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

    if "Dollars" not in input_df.columns:
        raise ValueError(
            "Input data is missing required column: 'Dollars'. "
            f"Available columns: {list(input_df.columns)}"
        )

    # Validate that Dollars is numeric, non-null, finite, and non-negative
    numeric_dollars = pd.to_numeric(input_df["Dollars"], errors="coerce")
    if numeric_dollars.isna().any():
        raise ValueError(
            "Invalid input: 'Dollars' contains null or non-numeric values."
        )

    if np.isinf(numeric_dollars).any():
        raise ValueError(
            "Invalid input: 'Dollars' contains infinite values."
        )

    if (numeric_dollars < 0).any():
        raise ValueError(
            "Invalid input: 'Dollars' contains negative values. "
            "Invoice dollar amounts must be non-negative."
        )

    # Predict freight using the Dollars feature
    feature_df = input_df[["Dollars"]]
    raw_predictions = model.predict(feature_df)

    input_df["Predicted_Freight"] = np.round(raw_predictions)

    return input_df


if __name__ == "__main__":
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)
    print("Freight Cost Predictions:")
    print(prediction)
