import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from inference.predict_invoice_flag import (
    predict_invoice_flag,
    load_artifacts,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    REQUIRED_FEATURES
)


class TestPredictInvoiceFlag(unittest.TestCase):

    def setUp(self):
        self.clean_invoice = {
            "invoice_quantity": 1200,
            "invoice_dollars": 18500.0,
            "Freight": 98.0,
            "total_item_quantity": 1200,
            "total_item_dollars": 18500.0
        }
        self.discrepant_invoice = {
            "invoice_quantity": 50,
            "invoice_dollars": 352.95,
            "Freight": 1.73,
            "total_item_quantity": 162,
            "total_item_dollars": 2476.0
        }

    def test_artifacts_loading(self):
        """Verify model and scaler artifacts exist and load properly."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model file missing at {DEFAULT_MODEL_PATH}")
        self.assertTrue(DEFAULT_SCALER_PATH.exists(), f"Scaler file missing at {DEFAULT_SCALER_PATH}")
        model, scaler = load_artifacts()
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(len(scaler.mean_), len(REQUIRED_FEATURES))

    def test_clean_invoice_prediction(self):
        """Verify that a clean invoice receives standard processing."""
        result = predict_invoice_flag(self.clean_invoice)
        self.assertEqual(len(result), 1)
        self.assertIn("Predicted_Flag", result.columns)
        self.assertIn("Risk_Score", result.columns)
        self.assertIn("Review_Recommendation", result.columns)
        self.assertIn("Accounting_Discrepancy_Detected", result.columns)

        self.assertFalse(result["Accounting_Discrepancy_Detected"].iloc[0])
        self.assertEqual(result["Review_Recommendation"].iloc[0], "Standard Processing")

    def test_discrepant_invoice_prediction(self):
        """Verify that an invoice with a large dollar mismatch is flagged for manual review."""
        result = predict_invoice_flag(self.discrepant_invoice)
        self.assertEqual(len(result), 1)
        self.assertTrue(result["Accounting_Discrepancy_Detected"].iloc[0])
        self.assertEqual(result["Review_Recommendation"].iloc[0], "Manual Review Required")
        reasons = result["Review_Reasons"].iloc[0]
        self.assertTrue(any("Accounting Discrepancy" in r for r in reasons))

    def test_multiple_invoices_batch(self):
        """Verify batch prediction with multiple invoices in a single dictionary."""
        batch = {
            "invoice_quantity": [1200, 50],
            "invoice_dollars": [18500.0, 352.95],
            "Freight": [98.0, 1.73],
            "total_item_quantity": [1200, 162],
            "total_item_dollars": [18500.0, 2476.0]
        }
        result = predict_invoice_flag(batch)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["Review_Recommendation"].tolist(), ["Standard Processing", "Manual Review Required"])

    def test_dataframe_input_support(self):
        """Verify pd.DataFrame inputs work seamlessly."""
        df_input = pd.DataFrame([self.clean_invoice, self.discrepant_invoice])
        result = predict_invoice_flag(df_input)
        self.assertEqual(len(result), 2)

    def test_missing_required_features(self):
        """Verify ValueError is raised when required features are missing."""
        incomplete_data = self.clean_invoice.copy()
        del incomplete_data["total_item_dollars"]
        with self.assertRaises(ValueError) as ctx:
            predict_invoice_flag(incomplete_data)
        self.assertIn("missing required feature", str(ctx.exception))
        self.assertIn("total_item_dollars", str(ctx.exception))

    def test_negative_values_rejected(self):
        """Verify negative values in features are rejected."""
        negative_data = self.clean_invoice.copy()
        negative_data["invoice_dollars"] = -500.0
        with self.assertRaises(ValueError) as ctx:
            predict_invoice_flag(negative_data)
        self.assertIn("negative values", str(ctx.exception))

    def test_non_numeric_values_rejected(self):
        """Verify string/non-numeric values are rejected."""
        bad_data = self.clean_invoice.copy()
        bad_data["Freight"] = "free_shipping"
        with self.assertRaises(ValueError) as ctx:
            predict_invoice_flag(bad_data)
        self.assertIn("null or non-numeric", str(ctx.exception))

    def test_infinite_values_rejected(self):
        """Verify infinite values are rejected."""
        inf_data = self.clean_invoice.copy()
        inf_data["total_item_quantity"] = np.inf
        with self.assertRaises(ValueError) as ctx:
            predict_invoice_flag(inf_data)
        self.assertIn("infinite values", str(ctx.exception))

    def test_target_isolation_no_leakage(self):
        """Verify that extra columns (such as target or labels) do not enter feature matrix."""
        polluted_data = self.clean_invoice.copy()
        polluted_data["flag_invoice"] = 1
        polluted_data["extra_column"] = "ignore_me"
        result = predict_invoice_flag(polluted_data)
        self.assertIn("flag_invoice", result.columns)
        self.assertIn("extra_column", result.columns)
        # Should still predict standard processing for clean invoice features
        self.assertEqual(result["Review_Recommendation"].iloc[0], "Standard Processing")

    def test_missing_model_file_handling(self):
        """Verify clear FileNotFoundError when model path does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            predict_invoice_flag(self.clean_invoice, model_path="models/missing_model.pkl")
        self.assertIn("Classifier model artifact not found", str(ctx.exception))

    def test_missing_scaler_file_handling(self):
        """Verify clear FileNotFoundError when scaler path does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            predict_invoice_flag(self.clean_invoice, scaler_path="models/missing_scaler.pkl")
        self.assertIn("Scaler artifact not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

