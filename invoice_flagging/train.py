import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from invoice_flagging.data_preprocessing import (
        load_invoice_data,
        split_data,
        scale_features,
        apply_labels
    )
    from invoice_flagging.modeling_evaluation import (
        train_random_forest,
        evaluate_classifier
    )
except ImportError:
    from data_preprocessing import (
        load_invoice_data,
        split_data,
        scale_features,
        apply_labels
    )
    from modeling_evaluation import (
        train_random_forest,
        evaluate_classifier
    )


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "scaler.pkl"
    model_path = model_dir / "predict_flag_invoice.pkl"

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(
        df, FEATURES, TARGET
    )

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, scaler_path
    )

    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    joblib.dump(
        grid_search.best_estimator_,
        str(model_path)
    )
    print(f"\nModel saved to: {model_path.resolve()}")
    print(f"Scaler saved to: {scaler_path.resolve()}")


if __name__ == "__main__":
    main()
