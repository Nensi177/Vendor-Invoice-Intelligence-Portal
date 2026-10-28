import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from freight_cost_prediction.data_preprocessing import (
        load_vendor_invoice_data,
        prepare_features,
        split_data
    )
    from freight_cost_prediction.model_evaluation import (
        train_linear_regression,
        train_decision_tree,
        train_random_forest,
        evaluate_model
    )
except ImportError:
    from data_preprocessing import (
        load_vendor_invoice_data,
        prepare_features,
        split_data
    )
    from model_evaluation import (
        train_linear_regression,
        train_decision_tree,
        train_random_forest,
        evaluate_model
    )


def main():
    db_path = PROJECT_ROOT / "data" / "inventory.db"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at '{db_path.resolve()}'. "
            "Please ensure inventory.db is placed inside the data/ directory."
        )

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = []
    results.append(
        evaluate_model(
            lr_model, X_test, y_test, "Linear Regression"
        )
    )
    results.append(
        evaluate_model(
            dt_model, X_test, y_test, "Decision Tree Regression"
        )
    )
    results.append(
        evaluate_model(
            rf_model, X_test, y_test, "Random Forest Regression"
        )
    )

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path.resolve()}")


if __name__ == "__main__":
    main()
