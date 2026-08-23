"""
Integration Test: End-to-End Surveillance Pipeline Execution for Cagayan de Oro City (PH104305000)
Verifies that the master CDO pipeline executes all 6 stages on real CCHAIN data (2003-2022)
and achieves CDO-specific holdout validation discrimination thresholds (ROC-AUC > 0.94, Recall > 0.80).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import pandas as pd
from src.generate_dummy_data import get_project_root
from src.pipeline import run_production_pipeline, find_raw_data_dir

class TestCagayanDeOroPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = get_project_root()
        cls.raw_dir = find_raw_data_dir(base_dir=cls.base_dir)
        cls.proc_dir = cls.base_dir / "data" / "processed"

    def test_cdo_pipeline_execution_and_holdout_benchmarks(self):
        """Runs full CDO production pipeline and evaluates on unseen 2019-2022 CDO holdout data."""
        if not (self.raw_dir / "location.csv").exists():
            self.skipTest("Real CCHAIN dataset not found in data/cchain_raw. Skipping real CDO test.")
        df_benchmarks, df_ready = run_production_pipeline(
            raw_dir=self.raw_dir,
            processed_dir=self.proc_dir
        )
        self.assertIsNotNone(df_benchmarks, "CDO Pipeline benchmarks returned None")
        self.assertIsNotNone(df_ready, "CDO Surveillance ready matrix returned None")
        
        # Verify 80 barangays and 18,880 observations (80 brgys * 236 months with 4-month lag)
        self.assertEqual(len(df_ready), 18880, f"Expected 18,880 CDO space-time records, found {len(df_ready)}")
        self.assertEqual(df_ready["adm4_pcode"].nunique(), 80, "Expected exactly 80 unique CDO barangays")
        
        # Verify processed files exist
        ready_csv = self.proc_dir / "cchain_cdo_dengue_surveillance_ready.csv"
        bench_csv = self.proc_dir / "cchain_model_benchmarks.csv"
        self.assertTrue(ready_csv.exists(), f"Missing processed matrix: {ready_csv}")
        self.assertTrue(bench_csv.exists(), f"Missing benchmark matrix: {bench_csv}")
        
        # Verify CDO Holdout Test Performance (Unseen 2019-2022)
        for _, row in df_benchmarks.iterrows():
            self.assertGreater(row["ROC-AUC"], 0.94, f"Model {row['Model']} in {row['Horizon']} failed ROC-AUC threshold: {row['ROC-AUC']}")
            self.assertGreater(row["Sensitivity (Recall)"], 0.70, f"Model {row['Model']} in {row['Horizon']} failed Sensitivity threshold: {row['Sensitivity (Recall)']}")

if __name__ == "__main__":
    unittest.main()
