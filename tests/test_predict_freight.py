import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from inference.predict_freight import predict_freight_cost, load_model, DEFAULT_MODEL_PATH


class TestPredictFreight(unittest.TestCase):

    def test_freight_model_loading_and_architecture(self):
        """Verify the freight model loads, is LinearRegression, and preserves coefficients."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model file missing at {DEFAULT_MODEL_PATH}")
        model = load_model()
        self.assertIsInstance(model, LinearRegression, "Freight model must be LinearRegression")
        self.assertEqual(list(model.feature_names_in_), ["Dollars"])
        self.assertAlmostEqual(model.coef_[0], 0.00501517, places=5)
        self.assertAlmostEqual(model.intercept_, 5.00802058, places=5)

    def test_four_sample_inference_values(self):
        """Verify the four documented sample inputs produce identical expected outputs."""
        sample_data = {
            "Dollars": [18500, 9000, 3000, 200]
        }
        result = predict_freight_cost(sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Dollars", result.columns)
        self.assertIn("Predicted_Freight", result.columns)

        expected_freight = [98.0, 50.0, 20.0, 6.0]
        actual_freight = result["Predicted_Freight"].tolist()
        self.assertEqual(actual_freight, expected_freight)

    def test_single_record_inputs(self):
        """Verify single record prediction works with scalar and list values."""
        res_scalar = predict_freight_cost({"Dollars": 18500})
        self.assertEqual(len(res_scalar), 1)
        self.assertEqual(res_scalar["Predicted_Freight"].iloc[0], 98.0)

        res_list = predict_freight_cost({"Dollars": [9000]})
        self.assertEqual(len(res_list), 1)
        self.assertEqual(res_list["Predicted_Freight"].iloc[0], 50.0)

    def test_input_preservation_with_extra_columns(self):
        """Verify original input columns (e.g. Quantity from app.py) are preserved."""
        input_data = {
            "Quantity": [1200, 500],
            "Dollars": [18500, 9000]
        }
        result = predict_freight_cost(input_data)
        self.assertIn("Quantity", result.columns)
        self.assertIn("Dollars", result.columns)
        self.assertIn("Predicted_Freight", result.columns)
        self.assertEqual(result["Quantity"].tolist(), [1200, 500])
        self.assertEqual(result["Dollars"].tolist(), [18500, 9000])

    def test_dataframe_input_support(self):
        """Verify pd.DataFrame inputs work seamlessly."""
        df_input = pd.DataFrame({"Dollars": [3000.0, 200.0]})
        result = predict_freight_cost(df_input)
        self.assertEqual(result["Predicted_Freight"].tolist(), [20.0, 6.0])

    def test_missing_dollars_column(self):
        """Verify error when required 'Dollars' column is absent."""
        with self.assertRaises(ValueError) as ctx:
            predict_freight_cost({"Quantity": [100]})
        self.assertIn("missing required column: 'Dollars'", str(ctx.exception))

    def test_empty_input(self):
        """Verify error on empty inputs."""
        with self.assertRaises(ValueError) as ctx:
            predict_freight_cost({"Dollars": []})
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_negative_values_rejected(self):
        """Verify negative dollar amounts are rejected."""
        with self.assertRaises(ValueError) as ctx:
            predict_freight_cost({"Dollars": [-50.0]})
        self.assertIn("negative values", str(ctx.exception))

    def test_non_numeric_values_rejected(self):
        """Verify non-numeric values are rejected."""
        with self.assertRaises(ValueError) as ctx:
            predict_freight_cost({"Dollars": ["invalid_amount"]})
        self.assertIn("null or non-numeric", str(ctx.exception))

    def test_null_values_rejected(self):
        """Verify null/NaN values are rejected."""
        with self.assertRaises(ValueError) as ctx:
            predict_freight_cost({"Dollars": [None, 500.0]})
        self.assertIn("null or non-numeric", str(ctx.exception))

    def test_infinite_values_rejected(self):
        """Verify infinite values are rejected."""
        with self.assertRaises(ValueError) as ctx:
            predict_freight_cost({"Dollars": [np.inf]})
        self.assertIn("infinite values", str(ctx.exception))

    def test_invalid_type_rejected(self):
        """Verify non-dict, non-DataFrame inputs are rejected."""
        with self.assertRaises(TypeError):
            predict_freight_cost([18500, 9000])

    def test_missing_model_file_handling(self):
        """Verify clear FileNotFoundError when model path does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            predict_freight_cost({"Dollars": [100]}, model_path="models/non_existent_model.pkl")
        self.assertIn("Freight model artifact not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

