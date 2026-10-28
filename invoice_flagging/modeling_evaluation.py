from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    f1_score
)


def build_invoice_pipeline():
    """
    Construct an sklearn Pipeline with StandardScaler and RandomForestClassifier.
    Ensures preprocessing is encapsulated to prevent cross-validation fold leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            random_state=42
        ))
    ])


def train_random_forest(X_train, y_train, param_grid=None, cv=5, n_jobs=1):
    """
    Train RandomForestClassifier through an sklearn Pipeline inside GridSearchCV.
    StandardScaler is fitted strictly within each cross-validation training fold.
    """
    pipeline = build_invoice_pipeline()

    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 6],
            "classifier__min_samples_split": [2, 5],
        }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)

    return {
        "accuracy": accuracy,
        "classification_report": report
    }