import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

from freight_cost_prediction.data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from freight_cost_prediction.model_evaluation import evaluate_model
from invoice_flagging.data_preprocessing import load_invoice_data, scale_features
from invoice_flagging.modeling_evaluation import build_invoice_pipeline, train_random_forest


class TestPipelineConsistency(unittest.TestCase):

    def test_database_resolution_freight(self):
        """Verify load_vendor_invoice_data resolves the database from project root."""
        df = load_vendor_invoice_data()
        self.assertFalse(df.empty)
        self.assertIn("Dollars", df.columns)
        self.assertIn("Freight", df.columns)

    def test_database_resolution_invoice(self):
        """Verify load_invoice_data resolves the database from project root."""
        df = load_invoice_data()
        self.assertFalse(df.empty)
        self.assertIn("invoice_dollars", df.columns)
        self.assertIn("total_item_dollars", df.columns)

    def test_freight_rmse_evaluation_fix(self):
        """Verify that evaluate_model calculates true RMSE (sqrt of MSE), not MSE."""
        # Simple synthetic test to verify formula
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 34.0])  # diffs: 2, -2, 4 -> sq diffs: 4, 4, 16 -> MSE = 8.0, RMSE = sqrt(8.0) ~ 2.8284

        class DummyModel:
            def predict(self, X):
                return y_pred

        metrics = evaluate_model(DummyModel(), X_test=None, y_test=y_true, model_name="TestDummy")

        self.assertAlmostEqual(metrics["mse"], 8.0, places=4)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(8.0), places=4)
        self.assertNotEqual(metrics["rmse"], metrics["mse"], "RMSE must not equal MSE unless error is 0 or 1")

    def test_scale_features_respects_custom_path(self):
        """Verify that scale_features saves scaler to the provided scaler_path parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_scaler_path = Path(tmpdir) / "custom_subdir" / "my_scaler.pkl"
            X_train = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [10.0, 20.0, 30.0]})
            X_test = pd.DataFrame({"col1": [1.5], "col2": [15.0]})

            scale_features(X_train, X_test, custom_scaler_path)

            self.assertTrue(custom_scaler_path.exists(), "Scaler was not saved to the provided scaler_path!")

    def test_invoice_pipeline_structure(self):
        """Verify build_invoice_pipeline encapsulates StandardScaler and RandomForestClassifier."""
        pipe = build_invoice_pipeline()
        self.assertIsInstance(pipe, Pipeline, "Expected sklearn.pipeline.Pipeline instance")
        self.assertIn("scaler", pipe.named_steps, "Pipeline must contain 'scaler' step")
        self.assertIn("classifier", pipe.named_steps, "Pipeline must contain 'classifier' step")
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["classifier"], RandomForestClassifier)

    def test_invoice_gridsearch_cv_isolation(self):
        """Verify train_random_forest fits pipeline inside CV with prefixed parameter names."""
        # Small synthetic dataset
        X = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        })
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        custom_grid = {
            "classifier__n_estimators": [10],
            "classifier__max_depth": [3]
        }
        gs = train_random_forest(X, y, param_grid=custom_grid, cv=2, n_jobs=1)
        best_pipe = gs.best_estimator_

        self.assertIsInstance(best_pipe, Pipeline)
        self.assertTrue(hasattr(best_pipe.named_steps["scaler"], "mean_"))
        self.assertTrue(hasattr(best_pipe.named_steps["classifier"], "estimators_"))

    def test_saved_invoice_artifact_is_pipeline(self):
        """Verify the saved invoice model artifact is a valid Pipeline and scaler artifact exists."""
        model_path = Path(__file__).resolve().parent.parent / "models" / "predict_flag_invoice.pkl"
        scaler_path = Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"

        self.assertTrue(model_path.exists(), f"Model artifact missing at {model_path}")
        self.assertTrue(scaler_path.exists(), f"Scaler artifact missing at {scaler_path}")

        model = joblib.load(str(model_path))
        scaler = joblib.load(str(scaler_path))

        self.assertIsInstance(model, Pipeline, "Saved model artifact must be an sklearn Pipeline")
        self.assertIn("scaler", model.named_steps)
        self.assertIn("classifier", model.named_steps)
        self.assertIsInstance(scaler, StandardScaler, "Saved scaler artifact must be StandardScaler")


if __name__ == "__main__":
    unittest.main()

