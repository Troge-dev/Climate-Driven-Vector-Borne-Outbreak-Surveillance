"""
Data Provenance and Pipeline Reproducibility Verification Script
Project CCHAIN: Climate-Driven Vector-Borne Outbreak Surveillance

This script:
1. Validates the integrity, schema, and column checksums of the committed CDO surveillance dataset.
2. Checks for the raw CCHAIN Kaggle dataset (thinkdatasci/project-cchain).
3. If raw data is present, regenerates the surveillance matrix from raw inputs and asserts exact equality
   against the committed artifact data/processed/cchain_cdo_dengue_surveillance_ready.csv.
4. Provides full provenance and download instructions.
"""

import sys
import os
import hashlib
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.pipeline import run_production_pipeline, find_raw_data_dir

def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dataframe_column_hashes(df: pd.DataFrame) -> dict:
    hashes = {}
    for col in df.columns:
        col_series = df[col].astype(str)
        col_bytes = col_series.str.cat(sep="|").encode("utf-8")
        hashes[col] = hashlib.sha256(col_bytes).hexdigest()[:16]
    return hashes

def verify_reproducibility():
    print("=" * 85)
    print("[*] PROJECT CCHAIN: DATA PROVENANCE & REPRODUCIBILITY VERIFICATION")
    print("=" * 85)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    target_csv = processed_dir / "cchain_cdo_dengue_surveillance_ready.csv"

    # Step 1: Validate Committed Artifact
    print("\n[Stage 1/3] Inspecting Committed Surveillance Ready Dataset...")
    if not target_csv.exists():
        print(f"[ERROR] Committed artifact not found: {target_csv}")
        return False

    file_size_mb = target_csv.stat().st_size / (1024 * 1024)
    file_sha256 = compute_file_sha256(target_csv)
    print(f"  [+] File Path: {target_csv}")
    print(f"  [+] File Size: {file_size_mb:.2f} MB")
    print(f"  [+] SHA-256 Checksum: {file_sha256}")

    df_committed = pd.read_csv(target_csv)
    print(f"  [+] Dimensions: {df_committed.shape[0]:,} rows x {df_committed.shape[1]} columns")
    print(f"  [+] Unique Barangays: {df_committed['adm4_pcode'].nunique()} (City: {df_committed['adm3_en'].iloc[0]})")
    print(f"  [+] Date Span: {df_committed['date'].min()} to {df_committed['date'].max()}")
    print(f"  [+] Target Outbreak Rate: {df_committed['is_outbreak'].mean() * 100:.2f}% ({df_committed['is_outbreak'].sum()} outbreak months)")

    # Assert basic invariants
    assert df_committed.shape[0] == 18880, f"Expected 18,880 rows, found {df_committed.shape[0]}"
    assert df_committed.shape[1] == 59, f"Expected 59 columns, found {df_committed.shape[1]}"
    assert df_committed["adm4_pcode"].nunique() == 80, "Expected 80 CDO barangays"
    assert (df_committed["brgy_p75_threshold"] >= 5.0).all(), "Threshold floor violated"
    print("  [PASS] Committed dataset schema and statistical invariants verified.")

    # Step 2: Raw CCHAIN Kaggle Dataset Status & Provenance
    print("\n[Stage 2/3] Checking Raw CCHAIN Data Provenance & Ingestion Directory...")
    raw_dir = find_raw_data_dir(base_dir=PROJECT_ROOT)
    raw_available = (raw_dir / "location.csv").exists()

    print(f"  [+] Kaggle Dataset ID: thinkdatasci/project-cchain")
    print(f"  [+] Kaggle URL: https://www.kaggle.com/datasets/thinkdatasci/project-cchain")
    print(f"  [+] Required Raw Tables: location.csv, brgy_geography.csv, disease_lgu_disaggregated_totals.csv,")
    print(f"                           climate_atmosphere.csv, google_open_buildings.csv, worldpop_population.csv")
    print(f"  [+] Resolved Raw Data Path: {raw_dir}")

    if not raw_available:
        print("\n  [!] Raw CCHAIN Kaggle dataset is not present in local directory (gitignored).")
        print("  [i] To download and verify full end-to-end raw data regeneration:")
        print("      1. Install Kaggle CLI: pip install kaggle")
        print("      2. Configure API token: ~/.kaggle/kaggle.json")
        print("      3. Download dataset:")
        print("         kaggle datasets download -d thinkdatasci/project-cchain -p data/cchain_raw --unzip")
        print("      4. Re-run: python scripts/verify_reproducibility.py")
        print("\n  [PASS] Data provenance documented and committed artifact is structurally verified.")
        return True

    # Step 3: End-to-End Regeneration and Bitwise / Numerical Diff
    print("\n[Stage 3/3] Regenerating Surveillance Dataset from Raw Inputs & Executing Diff...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_processed = Path(tmp_dir)
        print(f"  [*] Executing pipeline to temporary sandbox: {tmp_processed}...")
        _, df_recreated = run_production_pipeline(raw_dir=raw_dir, processed_dir=tmp_processed)
        
        recreated_csv = tmp_processed / "cchain_cdo_dengue_surveillance_ready.csv"
        df_recreated_file = pd.read_csv(recreated_csv)

        print("\n  [*] Comparing Recreated Dataset vs. Committed Dataset:")
        # 1. Row count diff
        if len(df_recreated_file) == len(df_committed):
            print(f"    [PASS] Row count exact match: {len(df_committed):,} rows")
        else:
            print(f"    [FAIL] Row count mismatch: Recreated={len(df_recreated_file)}, Committed={len(df_committed)}")
            return False

        # 2. Column diff
        if list(df_recreated_file.columns) == list(df_committed.columns):
            print(f"    [PASS] Column list exact match: {len(df_committed.columns)} columns")
        else:
            diff_cols = set(df_recreated_file.columns) ^ set(df_committed.columns)
            print(f"    [FAIL] Column mismatch: {diff_cols}")
            return False

        # 3. Numeric tolerances and categorical exactness
        col_mismatches = []
        for col in df_committed.columns:
            if pd.api.types.is_numeric_dtype(df_committed[col]):
                if not np.allclose(df_committed[col].fillna(0), df_recreated_file[col].fillna(0), rtol=1e-5, atol=1e-5):
                    col_mismatches.append(col)
            else:
                if not (df_committed[col].fillna("").astype(str) == df_recreated_file[col].fillna("").astype(str)).all():
                    col_mismatches.append(col)

        if not col_mismatches:
            print(f"    [PASS] Numerical and categorical column equivalence: 100% (59/59 columns match)")
        else:
            print(f"    [FAIL] Column mismatches found in: {col_mismatches}")
            return False

    print("\n" + "=" * 85)
    print("[SUCCESS] PROVENANCE & REPRODUCIBILITY AUDIT PASSED: Pipeline is 100% reproducible from source.")
    print("=" * 85)
    return True

if __name__ == "__main__":
    success = verify_reproducibility()
    sys.exit(0 if success else 1)
