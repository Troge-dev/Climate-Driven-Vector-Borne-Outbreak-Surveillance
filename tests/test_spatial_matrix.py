"""
Unit Tests: Spatial Contiguity Weights Matrix (W) Construction for Cagayan de Oro City
Verifies row-stochasticity, symmetry, non-negativity, and topological adjacency for CDO's 80 Barangays.
"""

import unittest
import numpy as np
import pandas as pd
import shapely.wkt
from pathlib import Path
from src.generate_dummy_data import get_project_root

class TestCagayanDeOroSpatialMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = get_project_root()
        cls.raw_dir = cls.base_dir / "data" / "cchain_raw"
        cls.cdo_pcode = "PH104305000"

    def test_cdo_barangay_wkt_validity_and_geometry(self):
        """Verifies that all 80 Cagayan de Oro barangays possess valid, non-empty WKT boundary polygons."""
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
        df_loc = pd.read_csv(self.raw_dir / "location.csv")
        target_pcodes = sorted(df_loc[df_loc["adm3_pcode"] == self.cdo_pcode]["adm4_pcode"].unique().tolist())
        num_brgys = len(target_pcodes)

        df_geo = pd.read_csv(self.raw_dir / "brgy_geography.csv")
        cdo_geo = df_geo[df_geo["adm4_pcode"].isin(target_pcodes)].drop_duplicates(subset=["adm4_pcode"]).copy()
        cdo_geo["poly"] = cdo_geo["geometry"].apply(shapely.wkt.loads)
        pcode_to_poly = dict(zip(cdo_geo["adm4_pcode"], cdo_geo["poly"]))

        adj_matrix = np.zeros((num_brgys, num_brgys), dtype=float)
        for i, pcode_i in enumerate(target_pcodes):
            poly_i = pcode_to_poly.get(pcode_i)
            for j, pcode_j in enumerate(target_pcodes):
                if i != j:
                    poly_j = pcode_to_poly.get(pcode_j)
                    if poly_i.touches(poly_j) or poly_i.intersects(poly_j):
                        adj_matrix[i, j] = 1.0

        # Check 1: Adjacency symmetry (A == A.T)
        self.assertTrue(np.allclose(adj_matrix, adj_matrix.T), "CDO Adjacency matrix is not symmetric!")

        # Check 2: No self-loops (diag == 0)
        self.assertTrue(np.all(np.diag(adj_matrix) == 0), "Diagonal of CDO adjacency matrix has self-loops!")

        # Check 3: Connected neighbors (CDO has 428 neighbor connections, avg 5.35 per barangay)
        self.assertEqual(int(adj_matrix.sum()), 428, f"Expected 428 spatial edges in CDO, got {int(adj_matrix.sum())}")

        # Check 4: Row-normalization (Row-stochastic property for all connected barangays)
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        connected_mask = (row_sums > 0).flatten()
        row_sums_safe = row_sums.copy()
        row_sums_safe[row_sums_safe == 0] = 1.0
        W = adj_matrix / row_sums_safe

        # All 79 connected barangays in CDO must sum to 1.0 exactly
        self.assertTrue(np.allclose(W[connected_mask].sum(axis=1), 1.0), "Connected rows in spatial weights matrix W do not sum to 1.0!")
        # The 1 isolated enclave/detached barangay (Bugo) has row sum == 0.0
        self.assertEqual((~connected_mask).sum(), 1, "Expected exactly 1 detached enclave in raw geography")

if __name__ == "__main__":
    unittest.main()
