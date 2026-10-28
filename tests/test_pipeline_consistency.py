import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from freight_cost_prediction.data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from freight_cost_prediction.model_evaluation import evaluate_model
from invoice_flagging.data_preprocessing import load_invoice_data, scale_features


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


if __name__ == "__main__":
    unittest.main()

