"""
Unit Tests: Spatial Contiguity Weights Matrix (W) Construction for Cagayan de Oro City
Verifies row-stochasticity, symmetry, non-negativity, and topological adjacency for CDO's 80 Barangays.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd
import shapely.wkt
from src.generate_dummy_data import get_project_root
from src.pipeline import find_raw_data_dir

class TestSpatialMatrixProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = get_project_root()
        cls.dummy_dir = cls.base_dir / "data" / "dummy_test_city"
        cls.raw_dir = find_raw_data_dir(base_dir=cls.base_dir)
        cls.cdo_pcode = "PH104305000"
        cls.dummy_pcode = "PH990001000"

    def _verify_spatial_matrix(self, raw_data_dir: Path, target_city_code: str):
        df_loc = pd.read_csv(raw_data_dir / "location.csv")
        target_pcodes = sorted(df_loc[df_loc["adm3_pcode"] == target_city_code]["adm4_pcode"].unique().tolist())
        num_brgys = len(target_pcodes)
        self.assertGreater(num_brgys, 0, "No barangays found for city code")

        df_geo = pd.read_csv(raw_data_dir / "brgy_geography.csv")
        city_geo = df_geo[df_geo["adm4_pcode"].isin(target_pcodes)].drop_duplicates(subset=["adm4_pcode"]).copy()
        city_geo["poly"] = city_geo["geometry"].apply(shapely.wkt.loads)
        pcode_to_poly = dict(zip(city_geo["adm4_pcode"], city_geo["poly"]))

        for pcode, poly in pcode_to_poly.items():
            self.assertTrue(poly.is_valid, f"Invalid WKT polygon for {pcode}")
            self.assertFalse(poly.is_empty, f"Empty polygon for {pcode}")

        adj_matrix = np.zeros((num_brgys, num_brgys), dtype=float)
        for i, pcode_i in enumerate(target_pcodes):
            poly_i = pcode_to_poly.get(pcode_i)
            for j, pcode_j in enumerate(target_pcodes):
                if i != j:
                    poly_j = pcode_to_poly.get(pcode_j)
                    if poly_i and poly_j and (poly_i.touches(poly_j) or poly_i.intersects(poly_j)):
                        adj_matrix[i, j] = 1.0

        # Symmetry & Diagonal Checks
        self.assertTrue(np.allclose(adj_matrix, adj_matrix.T), "Adjacency matrix is not symmetric!")
        self.assertTrue(np.all(np.diag(adj_matrix) == 0), "Diagonal has self-loops!")

        # Row stochasticity
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        connected_mask = (row_sums > 0).flatten()
        row_sums_safe = row_sums.copy()
        row_sums_safe[row_sums_safe == 0] = 1.0
        W = adj_matrix / row_sums_safe
        self.assertTrue(np.allclose(W[connected_mask].sum(axis=1), 1.0), "Connected rows do not sum to 1.0!")
        return adj_matrix, W

    def test_dummy_city_spatial_matrix(self):
        """Unconditionally verifies spatial matrix topology on committed synthetic dummy city."""
        self._verify_spatial_matrix(self.dummy_dir, self.dummy_pcode)

    def test_cdo_barangay_wkt_validity_and_geometry(self):
        """Verifies that all 80 Cagayan de Oro barangays possess valid boundary polygons (if raw data present)."""
        if not (self.raw_dir / "location.csv").exists():
            self.skipTest("Real CCHAIN dataset not found in data/cchain_raw. Skipping real CDO test.")
        df_loc = pd.read_csv(self.raw_dir / "location.csv")
        cdo_pcodes = df_loc[df_loc["adm3_pcode"] == self.cdo_pcode]["adm4_pcode"].unique().tolist()
        self.assertEqual(len(cdo_pcodes), 80, f"Expected 80 CDO barangays, found {len(cdo_pcodes)}")

        df_geo = pd.read_csv(self.raw_dir / "brgy_geography.csv")
        cdo_geo = df_geo[df_geo["adm4_pcode"].isin(cdo_pcodes)]
        self.assertEqual(len(cdo_geo), 80, f"Expected 80 geometry records for CDO, found {len(cdo_geo)}")

        for _, row in cdo_geo.iterrows():
            wkt_str = row["geometry"]
            poly = shapely.wkt.loads(wkt_str)
            self.assertTrue(poly.is_valid, f"Invalid WKT polygon detected for {row['adm4_pcode']}")
            self.assertFalse(poly.is_empty, f"Empty polygon detected for {row['adm4_pcode']}")

    def test_cdo_spatial_weights_matrix_properties(self):
        """Verifies symmetry, absence of self-loops, neighbor density, and row-stochasticity for CDO."""
        if not (self.raw_dir / "location.csv").exists():
            self.skipTest("Real CCHAIN dataset not found in data/cchain_raw. Skipping real CDO test.")
        adj_matrix, W = self._verify_spatial_matrix(self.raw_dir, self.cdo_pcode)
        self.assertEqual(int(adj_matrix.sum()), 428, f"Expected 428 spatial edges in CDO, got {int(adj_matrix.sum())}")

if __name__ == "__main__":
    unittest.main()
