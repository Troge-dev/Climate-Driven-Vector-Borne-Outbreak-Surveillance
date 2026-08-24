"""
Integration Test: End-to-End Surveillance Pipeline Execution on Synthetic Dummy Dataset
Verifies that the complete pipeline (GIS W matrix, multi-table aggregation, distributed lags,
zero-leakage thresholding, ML benchmark tournament) executes cleanly on data/dummy_test_city/
without requiring external Kaggle datasets or credentials.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import pandas as pd
import numpy as np
from src.generate_dummy_data import get_project_root
from src.validation_runner import run_comprehensive_validation

class TestDummyDataPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = get_project_root()
        cls.dummy_raw_dir = cls.base_dir / "data" / "dummy_test_city"
        cls.dummy_proc_dir = cls.base_dir / "data" / "processed_dummy_test"
        cls.pilot_city_code = "PH990001000"

    def test_dummy_city_raw_files_exist(self):
        """Verifies that all 6 required raw CCHAIN schema-compliant CSV tables exist in dummy dataset."""
        required_files = [
            "location.csv",
            "brgy_geography.csv",
            "disease_lgu_disaggregated_totals.csv",
            "climate_atmosphere.csv",
            "google_open_buildings.csv",
            "worldpop_population.csv"
        ]
        for fname in required_files:
            file_path = self.dummy_raw_dir / fname
            self.assertTrue(file_path.exists(), f"Missing required dummy data file: {file_path}")
            self.assertGreater(file_path.stat().st_size, 0, f"Dummy file is empty: {file_path}")

    def test_end_to_end_dummy_pipeline_execution(self):
        """Executes full 5-stage validation pipeline on synthetic data and verifies outputs."""
        # Run validation pipeline
        run_comprehensive_validation(
            raw_dir=self.dummy_raw_dir,
            processed_dir=self.dummy_proc_dir,
            pilot_city_code=self.pilot_city_code,
            pilot_city_name="Synthetic Test City"
        )

        ready_csv = self.dummy_proc_dir / "synthetic_validation_surveillance_ready.csv"
        bench_csv = self.dummy_proc_dir / "synthetic_model_benchmarks.csv"

        self.assertTrue(ready_csv.exists(), f"Missing processed matrix: {ready_csv}")
        self.assertTrue(bench_csv.exists(), f"Missing benchmark matrix: {bench_csv}")

        # Verify engineered dataset properties
        df_ready = pd.read_csv(ready_csv)
        self.assertGreater(len(df_ready), 0, "Engineered synthetic matrix is empty")
        self.assertEqual(df_ready["adm3_pcode"].iloc[0], self.pilot_city_code)
        
        # Check target leakage protection
        self.assertTrue((df_ready["brgy_p75_threshold"] >= 5.0).all(), "Found threshold below min floor 5.0")
        self.assertTrue(set(df_ready["is_outbreak"].unique()).issubset({0, 1}), "Invalid binary target values")

        # Verify model benchmark results
        df_bench = pd.read_csv(bench_csv)
        self.assertGreater(len(df_bench), 0, "Benchmark table is empty")
        self.assertIn("ROC-AUC", df_bench.columns)
        self.assertIn("PR-AUC", df_bench.columns)
        self.assertIn("Recall (Sensitivity)", df_bench.columns)

        for _, row in df_bench.iterrows():
            self.assertFalse(np.isnan(row["ROC-AUC"]), f"NaN ROC-AUC in {row['Model']}")
            self.assertGreater(row["ROC-AUC"], 0.50, f"ROC-AUC below random guessing in {row['Model']}")
            self.assertFalse(np.isnan(row["PR-AUC"]), f"NaN PR-AUC in {row['Model']}")

if __name__ == "__main__":
    unittest.main()
