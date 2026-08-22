import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ------------------------------------------------------------------------------
# 1. SETUP DIRECTORIES & CONFIGURATION
# ------------------------------------------------------------------------------
BASE_DIR = Path(".")
RAW_DATA_DIR = BASE_DIR / "data" / "cchain_raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

PILOT_CITY_CODE = "PH104305000"  # Cagayan de Oro City
PILOT_CITY_NAME = "Cagayan de Oro City"
TARGET_DISEASE = "DENGUE FEVER"

print(f"[*] Processing official CCHAIN Kaggle Dataset for: {PILOT_CITY_NAME} ({PILOT_CITY_CODE})")
print(f"[*] Target Disease: {TARGET_DISEASE}")

# ------------------------------------------------------------------------------
# 2. LOAD REAL GEOGRAPHY & BARANGAY BOUNDARIES
# ------------------------------------------------------------------------------
print("\n[1/5] Loading location.csv...")
df_loc = pd.read_csv(RAW_DATA_DIR / "location.csv")
cdo_brgys = df_loc[df_loc["adm3_pcode"] == PILOT_CITY_CODE][
    ["adm1_en", "adm2_en", "adm3_pcode", "adm3_en", "adm4_pcode", "adm4_en", "brgy_total_area"]
].drop_duplicates().reset_index(drop=True)

target_pcodes = cdo_brgys["adm4_pcode"].unique()
print(f"[+] Identified {len(target_pcodes)} barangays in {PILOT_CITY_NAME}.")

# ------------------------------------------------------------------------------
# 3. LOAD & AGGREGATE REAL HEALTH RECORDS (DENGUE)
# ------------------------------------------------------------------------------
print("\n[2/5] Loading disease_lgu_disaggregated_totals.csv...")
df_lgu = pd.read_csv(RAW_DATA_DIR / "disease_lgu_disaggregated_totals.csv")

# Filter for CDO, target disease, and valid barangay codes
df_dengue = df_lgu[
    (df_lgu["adm3_pcode"] == PILOT_CITY_CODE) &
    (df_lgu["disease_common_name"] == TARGET_DISEASE) &
    (df_lgu["adm4_pcode"].isin(target_pcodes))
].copy()

# Standardize date to Year-Month (Monthly Grain)
df_dengue["date"] = pd.to_datetime(df_dengue["date"]).dt.to_period("M").dt.to_timestamp()

# Aggregate over age groups and sexes
df_health_agg = df_dengue.groupby(["adm4_pcode", "date"], as_index=False).agg(
    dengue_cases=("case_total", "sum"),
    dengue_deaths=("death_total", "sum")
)
print(f"[+] Processed {len(df_health_agg)} barangay-month dengue records.")

# ------------------------------------------------------------------------------
# 4. LOAD & AGGREGATE REAL CLIMATE ATMOSPHERE DATA
# ------------------------------------------------------------------------------
print("\n[3/5] Loading climate_atmosphere.csv (ERA5)...")
df_clim = pd.read_csv(RAW_DATA_DIR / "climate_atmosphere.csv")
df_clim = df_clim[df_clim["adm4_pcode"].isin(target_pcodes)].copy()

df_clim["date"] = pd.to_datetime(df_clim["date"]).dt.to_period("M").dt.to_timestamp()

# Aggregate daily climate data to monthly statistics
df_clim_agg = df_clim.groupby(["adm4_pcode", "date"], as_index=False).agg(
    pr_monthly_total_mm=("pr", "sum"),
    tave_monthly_mean_c=("tave", "mean"),
    tmin_monthly_mean_c=("tmin", "mean"),
    tmax_monthly_mean_c=("tmax", "mean"),
    heat_index_monthly_mean_c=("heat_index", "mean"),
    heat_index_monthly_max_c=("heat_index", "max"),
    rh_monthly_mean_pct=("rh", "mean"),
    wind_speed_monthly_mean=("wind_speed", "mean"),
    solar_rad_monthly_mean=("solar_rad", "mean")
)
print(f"[+] Processed {len(df_clim_agg)} barangay-month climate records.")

# ------------------------------------------------------------------------------
# 5. LOAD REAL BUILT-ENVIRONMENT & POPULATION COVARIATES
# ------------------------------------------------------------------------------
print("\n[4/5] Loading google_open_buildings.csv and worldpop_population.csv...")
df_bldgs = pd.read_csv(RAW_DATA_DIR / "google_open_buildings.csv")
df_bldgs_cdo = df_bldgs[df_bldgs["adm4_pcode"].isin(target_pcodes)][
    ["adm4_pcode", "google_bldgs_count", "google_bldgs_density", "google_bldgs_pct_built_up_area", "google_bldgs_area_mean"]
].drop_duplicates(subset=["adm4_pcode"])

df_pop = pd.read_csv(RAW_DATA_DIR / "worldpop_population.csv")
# Take the most recent population density survey per barangay
df_pop_cdo = df_pop[df_pop["adm4_pcode"].isin(target_pcodes)].sort_values(by="date").groupby("adm4_pcode").last().reset_index()[
    ["adm4_pcode", "pop_count_total", "pop_density_mean"]
]

df_static = df_bldgs_cdo.merge(df_pop_cdo, on="adm4_pcode", how="left")

# ------------------------------------------------------------------------------
# 6. SPACE-TIME GRID ALIGNMENT & RELATIONAL MERGE
# ------------------------------------------------------------------------------
print("\n[5/5] Aligning full Space-Time grid & Engineering Biological Lags...")

# Generate time range corresponding to available climate records (2003-2022)
min_date = df_clim_agg["date"].min()
max_date = df_clim_agg["date"].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq="MS")

grid_idx = pd.MultiIndex.from_product([target_pcodes, all_dates], names=["adm4_pcode", "date"]).to_frame().reset_index(drop=True)

# Merge everything
df_merged = grid_idx.merge(cdo_brgys, on="adm4_pcode", how="left")
df_merged = df_merged.merge(df_clim_agg, on=["adm4_pcode", "date"], how="left")
df_merged = df_merged.merge(df_health_agg, on=["adm4_pcode", "date"], how="left")
df_merged = df_merged.merge(df_static, on="adm4_pcode", how="left")

# Clean target NaNs (unreported months = 0 recorded cases)
df_merged["dengue_cases"] = df_merged["dengue_cases"].fillna(0).astype(int)
df_merged["dengue_deaths"] = df_merged["dengue_deaths"].fillna(0).astype(int)

# Sort chronologically per barangay
df_merged = df_merged.sort_values(by=["adm4_pcode", "date"]).reset_index(drop=True)

# ------------------------------------------------------------------------------
# 7. FEATURE ENGINEERING: BIOLOGICAL LAGS & OUTBREAK LABELS
# ------------------------------------------------------------------------------
# Non-linear weather lags (1-month, 2-month, 3-month)
for lag in [1, 2, 3]:
    df_merged[f"pr_total_mm_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["pr_monthly_total_mm"].shift(lag)
    df_merged[f"heat_index_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["heat_index_monthly_mean_c"].shift(lag)
    df_merged[f"tave_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["tave_monthly_mean_c"].shift(lag)

# 3-month cumulative rainfall moving average
df_merged["pr_rolling_3m_avg"] = df_merged.groupby("adm4_pcode")["pr_monthly_total_mm"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Outbreak Definition: 1 if monthly cases exceed the barangay's historical 75th percentile (and at least 5 cases)
df_merged["brgy_p75_threshold"] = df_merged.groupby("adm4_pcode")["dengue_cases"].transform(
    lambda x: max(5, x.quantile(0.75))
)
df_merged["is_outbreak"] = (df_merged["dengue_cases"] >= df_merged["brgy_p75_threshold"]).astype(int)

# Drop initial lag NaNs
df_final = df_merged.dropna(subset=["pr_total_mm_lag_3m"]).copy()

# Save real processed matrix
output_csv_path = PROCESSED_DATA_DIR / "cchain_cdo_dengue_surveillance_ready.csv"
df_final.to_csv(output_csv_path, index=False)

print(f"[SUCCESS] Real CCHAIN surveillance dataset successfully exported to: {output_csv_path}")
print(f"[i] Final Matrix Dimensions: {df_final.shape[0]} rows x {df_final.shape[1]} columns")

# ------------------------------------------------------------------------------
# 8. TRAIN & EVALUATE BASELINE SURVEILLANCE MODEL
# ------------------------------------------------------------------------------
print("\n[*] Training Baseline Surveillance Classifier on Real CCHAIN Records...")

feature_cols = [
    "pr_monthly_total_mm", "tave_monthly_mean_c", "heat_index_monthly_mean_c",
    "heat_index_monthly_max_c", "rh_monthly_mean_pct", "wind_speed_monthly_mean",
    "pr_total_mm_lag_1m", "pr_total_mm_lag_2m", "pr_total_mm_lag_3m",
    "heat_index_mean_lag_1m", "heat_index_mean_lag_2m", "heat_index_mean_lag_3m",
    "tave_mean_lag_1m", "pr_rolling_3m_avg",
    "google_bldgs_density", "google_bldgs_pct_built_up_area", "pop_density_mean"
]

X = df_final[feature_cols].fillna(0)
y = df_final["is_outbreak"]

# Train/Test Split by temporal cutoff (Train: 2003-2018, Test: 2019-2022)
train_mask = df_final["date"] < "2019-01-01"
test_mask = df_final["date"] >= "2019-01-01"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

rf = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

if len(np.unique(y_test)) > 1:
    roc_score = roc_auc_score(y_test, y_prob)
    print(f"[SUCCESS] Real Data Outbreak Prediction ROC-AUC Score: {roc_score:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 5 Most Predictive Climate-Health Indicators (Real Data):")
for feat, imp in importances.head(5).items():
    print(f"  - {feat}: {imp:.4f}")
