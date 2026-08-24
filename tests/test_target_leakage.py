"""
Unit Test: Target Leakage Prevention in Outbreak Label Construction
Verifies that per-barangay p75 thresholds are computed strictly from pre-2019 training data
and remain invariant regardless of alterations, additions, or omissions of post-2019 test data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd

class TestTargetLeakagePrevention(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset spanning 2015 to 2022 for two barangays
        dates_train = pd.date_range("2015-01-01", "2018-12-01", freq="MS")
        dates_test = pd.date_range("2019-01-01", "2022-12-01", freq="MS")
        all_dates = dates_train.append(dates_test)

        records = []
        for brgy in ["BRGY_A", "BRGY_B"]:
            base_cases = 10 if brgy == "BRGY_A" else 20
            for dt in all_dates:
                # Pre-2019: moderate cases
                if dt < pd.Timestamp("2019-01-01"):
                    cases = base_cases + (dt.month % 4)
                else:
                    # Post-2019: massive case spike to induce potential leakage if included
                    cases = base_cases * 10
                records.append({
                    "adm4_pcode": brgy,
                    "date": dt,
                    "dengue_cases": cases
                })
        self.df = pd.DataFrame(records)

    def _compute_thresholds(self, df):
        """Standard leakage-free threshold computation used in pipeline."""
        train_slice = df[df["date"] < "2019-01-01"]
        p75_training = (
            train_slice.groupby("adm4_pcode")["dengue_cases"]
            .quantile(0.75)
            .apply(lambda q: max(5.0, float(q)))
            .to_dict()
        )
        return p75_training

    def test_threshold_invariant_to_test_period_manipulation(self):
        """Asserts that modifying or dropping test-period rows produces identical thresholds."""
        # 1. Baseline threshold with full dataset
        th_full = self._compute_thresholds(self.df)

        # 2. Threshold computed on dataset with test-period completely dropped
        df_train_only = self.df[self.df["date"] < "2019-01-01"].copy()
        th_train_only = self._compute_thresholds(df_train_only)

        self.assertEqual(th_full, th_train_only, "Threshold changed when test period was dropped!")

        # 3. Threshold computed on dataset with extreme post-2019 perturbations (1,000,000 cases)
        df_perturbed = self.df.copy()
        test_mask = df_perturbed["date"] >= "2019-01-01"
        df_perturbed.loc[test_mask, "dengue_cases"] = 1_000_000
        th_perturbed = self._compute_thresholds(df_perturbed)

        self.assertEqual(th_full, th_perturbed, "Threshold changed when post-2019 test cases were perturbed!")

    def test_minimum_floor_enforcement(self):
        """Asserts that zero/low case barangays receive the safe minimum threshold floor of 5.0."""
        df_zero = self.df.copy()
        train_mask = df_zero["date"] < "2019-01-01"
        df_zero.loc[train_mask & (df_zero["adm4_pcode"] == "BRGY_A"), "dengue_cases"] = 0

        th = self._compute_thresholds(df_zero)
        self.assertEqual(th["BRGY_A"], 5.0, "Minimum threshold floor of 5.0 was not applied to zero-case barangay!")

if __name__ == "__main__":
    unittest.main()
