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

    # Prepare data: 80/20 holdout split on raw features
    X_train, X_test, y_train, y_test = split_data(
        df, FEATURES, TARGET
    )

    # Train Pipeline (StandardScaler + RandomForest) via GridSearchCV
    # StandardScaler is fitted strictly within each cross-validation fold
    grid_search = train_random_forest(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    print("\n=======================================================")
    print("Cross-Validation GridSearch Completed")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print("=======================================================")

    # Final evaluation on the held-out test set (unseen during CV tuning)
    evaluate_classifier(
        best_pipeline,
        X_test,
        y_test,
        "Random Forest Classifier Pipeline"
    )

    # Save best pipeline model artifact
    joblib.dump(
        best_pipeline,
        str(model_path)
    )

    # Also save the fitted scaler step to preserve artifact compatibility
    if hasattr(best_pipeline, "named_steps") and "scaler" in best_pipeline.named_steps:
        joblib.dump(
            best_pipeline.named_steps["scaler"],
            str(scaler_path)
        )

    print(f"\nModel pipeline saved to: {model_path.resolve()}")
    print(f"Scaler saved to: {scaler_path.resolve()}")


if __name__ == "__main__":
    main()
