"""
Project CCHAIN: Climate-Driven Vector-Borne Outbreak Surveillance Engine
Complete Advanced Spatial-Temporal Pipeline (Phases 1-4)

Key Capabilities:
1. Zero Future Climate Leakage (30-Day and 60-Day Lead Forecast Horizons)
2. Spatial Adjacency Contiguity Graph & Neighborhood Spillover (W * Y)
3. Urban-Climate Physical Interactions & Deterministic Demographics
4. Multi-Model Tournament (LogReg, Random Forest, LightGBM, XGBoost) & F2-Score Optimization
"""

from pathlib import Path
import numpy as np
import pandas as pd
import shapely.wkt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)
import lightgbm as lgb
import xgboost as xgb

# ------------------------------------------------------------------------------
# 1. SETUP DIRECTORIES & CONFIGURATION
# ------------------------------------------------------------------------------
BASE_DIR = Path(".")
RAW_DATA_DIR = BASE_DIR / "data" / "cchain_raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

PILOT_CITY_CODE = "PH104305000"  # Cagayan de Oro City
PILOT_CITY_NAME = "Cagayan de Oro City"
TARGET_DISEASE = "DENGUE FEVER"


def build_spatial_weights(raw_data_dir: Path, target_pcodes: list):
    """
    Parses barangay polygon boundaries and builds a row-standardized
    spatial contiguity matrix W (80x80) and coastal indicators.
    """
    df_geo = pd.read_csv(raw_data_dir / "brgy_geography.csv")
    cdo_geo = df_geo[df_geo["adm4_pcode"].isin(target_pcodes)].drop_duplicates(subset=["adm4_pcode"]).copy()

    cdo_geo["poly"] = cdo_geo["geometry"].apply(shapely.wkt.loads)
    pcode_to_poly = dict(zip(cdo_geo["adm4_pcode"], cdo_geo["poly"]))

    num_brgys = len(target_pcodes)
    adj_matrix = np.zeros((num_brgys, num_brgys), dtype=float)

    for i, pcode_i in enumerate(target_pcodes):
        poly_i = pcode_to_poly.get(pcode_i)
        if poly_i is None:
            continue
        for j, pcode_j in enumerate(target_pcodes):
            if i != j:
                poly_j = pcode_to_poly.get(pcode_j)
                if poly_j is not None and (poly_i.touches(poly_j) or poly_i.intersects(poly_j)):
                    adj_matrix[i, j] = 1.0

    # Row-normalize spatial weights matrix W
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_spatial = adj_matrix / row_sums

    coastal_map = dict(zip(cdo_geo["adm4_pcode"], cdo_geo["brgy_is_coastal"].astype(int)))
    return W_spatial, adj_matrix, coastal_map


def load_and_preprocess_dataset(raw_data_dir: Path, processed_data_dir: Path):
    """
    Aggregates multi-modal data streams (ERA5-Land, WorldPop, Google Buildings, DOH Surveillance)
    and computes biological lags, physical interactions, and spatial spillovers.
    """
    print("\n[1/5] Loading location boundaries & spatial contiguity...")
    df_loc = pd.read_csv(raw_data_dir / "location.csv")
    cdo_brgys = df_loc[df_loc["adm3_pcode"] == PILOT_CITY_CODE][
        ["adm1_en", "adm2_en", "adm3_pcode", "adm3_en", "adm4_pcode", "adm4_en", "brgy_total_area"]
    ].drop_duplicates(subset=["adm4_pcode"]).reset_index(drop=True)

    target_pcodes = sorted(cdo_brgys["adm4_pcode"].unique().tolist())
    W_spatial, adj_matrix, coastal_map = build_spatial_weights(raw_data_dir, target_pcodes)
    cdo_brgys["brgy_is_coastal"] = cdo_brgys["adm4_pcode"].map(coastal_map).fillna(0).astype(int)

    print(f"[+] Spatial Contiguity Matrix W: {W_spatial.shape[0]}x{W_spatial.shape[1]} ({int(adj_matrix.sum())} edges)")

    print("\n[2/5] Loading and aggregating epidemiological records...")
    df_lgu = pd.read_csv(raw_data_dir / "disease_lgu_disaggregated_totals.csv")
    df_dengue = df_lgu[
        (df_lgu["adm3_pcode"] == PILOT_CITY_CODE) &
        (df_lgu["disease_common_name"] == TARGET_DISEASE) &
        (df_lgu["adm4_pcode"].isin(target_pcodes))
    ].copy()

    df_dengue["date"] = pd.to_datetime(df_dengue["date"]).dt.to_period("M").dt.to_timestamp()
    df_health_agg = df_dengue.groupby(["adm4_pcode", "date"], as_index=False).agg(
        dengue_cases=("case_total", "sum"),
        dengue_deaths=("death_total", "sum")
    )

    print("\n[3/5] Loading and aggregating ERA5 climate atmosphere data...")
    df_clim = pd.read_csv(raw_data_dir / "climate_atmosphere.csv")
    df_clim = df_clim[df_clim["adm4_pcode"].isin(target_pcodes)].copy()
    df_clim["date"] = pd.to_datetime(df_clim["date"]).dt.to_period("M").dt.to_timestamp()

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

    print("\n[4/5] Loading urban buildings & WorldPop demographics...")
    df_bldgs = pd.read_csv(raw_data_dir / "google_open_buildings.csv")
    df_bldgs_cdo = df_bldgs[df_bldgs["adm4_pcode"].isin(target_pcodes)][
        ["adm4_pcode", "google_bldgs_count", "google_bldgs_density", "google_bldgs_pct_built_up_area", "google_bldgs_area_mean"]
    ].drop_duplicates(subset=["adm4_pcode"])

    df_pop = pd.read_csv(raw_data_dir / "worldpop_population.csv")
    df_pop_cdo = df_pop[df_pop["adm4_pcode"].isin(target_pcodes)].sort_values(by="date").groupby("adm4_pcode").last().reset_index()[
        ["adm4_pcode", "pop_count_total"]
    ]
    df_static = df_bldgs_cdo.merge(df_pop_cdo, on="adm4_pcode", how="left")

    print("\n[5/5] Performing space-time alignment & feature engineering...")
    min_date = df_clim_agg["date"].min()
    max_date = df_clim_agg["date"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="MS")

    grid_idx = pd.MultiIndex.from_product([target_pcodes, all_dates], names=["adm4_pcode", "date"]).to_frame().reset_index(drop=True)

    df_merged = grid_idx.merge(cdo_brgys, on="adm4_pcode", how="left")
    df_merged = df_merged.merge(df_clim_agg, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(df_health_agg, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(df_static, on="adm4_pcode", how="left")

    # Clean target NaNs
    df_merged["dengue_cases"] = df_merged["dengue_cases"].fillna(0).astype(int)
    df_merged["dengue_deaths"] = df_merged["dengue_deaths"].fillna(0).astype(int)

    # Deterministic population density imputation
    df_merged["pop_density_imputed"] = df_merged["pop_count_total"] / df_merged["brgy_total_area"]

    # Dynamic 75th percentile outbreak baseline
    df_merged["brgy_p75_threshold"] = df_merged.groupby("adm4_pcode")["dengue_cases"].transform(
        lambda x: max(5.0, float(x.quantile(0.75)))
    )
    df_merged["is_outbreak"] = (df_merged["dengue_cases"] >= df_merged["brgy_p75_threshold"]).astype(int)

    df_merged = df_merged.sort_values(by=["adm4_pcode", "date"]).reset_index(drop=True)

    # Meteorological lags
    for lag in [1, 2, 3, 4]:
        df_merged[f"pr_total_mm_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["pr_monthly_total_mm"].shift(lag)
        df_merged[f"heat_index_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["heat_index_monthly_mean_c"].shift(lag)
        df_merged[f"tave_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["tave_monthly_mean_c"].shift(lag)
        df_merged[f"rh_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["rh_monthly_mean_pct"].shift(lag)

    # Lagged 3-month rolling averages
    df_merged["pr_rolling_3m_lag1m"] = (
        df_merged["pr_total_mm_lag_1m"] + df_merged["pr_total_mm_lag_2m"] + df_merged["pr_total_mm_lag_3m"]
    ) / 3.0

    df_merged["pr_rolling_3m_lag2m"] = (
        df_merged["pr_total_mm_lag_2m"] + df_merged["pr_total_mm_lag_3m"] + df_merged["pr_total_mm_lag_4m"]
    ) / 3.0

    # Physical urban-climate interactions
    df_merged["runoff_risk_lag1m"] = df_merged["google_bldgs_pct_built_up_area"] * df_merged["pr_total_mm_lag_1m"]
    df_merged["runoff_risk_lag2m"] = df_merged["google_bldgs_pct_built_up_area"] * df_merged["pr_total_mm_lag_2m"]
    df_merged["urban_heat_trap_lag1m"] = df_merged["google_bldgs_density"] * df_merged["heat_index_mean_lag_1m"]
    df_merged["urban_heat_trap_lag2m"] = df_merged["google_bldgs_density"] * df_merged["heat_index_mean_lag_2m"]
    df_merged["host_exposure_index"] = df_merged["pop_density_imputed"] * df_merged["google_bldgs_pct_built_up_area"]

    # Autoregressive case lags
    for lag in [1, 2]:
        df_merged[f"dengue_cases_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["dengue_cases"].shift(lag)
        df_merged[f"is_outbreak_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["is_outbreak"].shift(lag)

    # Spatial contiguity spillover (W * Y)
    cases_pivot = df_merged.pivot(index="date", columns="adm4_pcode", values="dengue_cases")[target_pcodes]
    outbreak_pivot = df_merged.pivot(index="date", columns="adm4_pcode", values="is_outbreak")[target_pcodes]

    spatial_cases = pd.DataFrame(np.dot(cases_pivot.values, W_spatial.T), index=cases_pivot.index, columns=target_pcodes)
    spatial_outbreak = pd.DataFrame(np.dot(outbreak_pivot.values, W_spatial.T), index=outbreak_pivot.index, columns=target_pcodes)

    spatial_cases_long = spatial_cases.reset_index().melt(id_vars="date", var_name="adm4_pcode", value_name="spatial_cases_current")
    spatial_outbreak_long = spatial_outbreak.reset_index().melt(id_vars="date", var_name="adm4_pcode", value_name="spatial_outbreak_current")

    df_merged = df_merged.merge(spatial_cases_long, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(spatial_outbreak_long, on=["adm4_pcode", "date"], how="left")

    df_merged = df_merged.sort_values(by=["adm4_pcode", "date"]).reset_index(drop=True)
    df_merged["spatial_lag_cases_1m"] = df_merged.groupby("adm4_pcode")["spatial_cases_current"].shift(1)
    df_merged["spatial_lag_cases_2m"] = df_merged.groupby("adm4_pcode")["spatial_cases_current"].shift(2)
    df_merged["spatial_lag_outbreak_1m"] = df_merged.groupby("adm4_pcode")["spatial_outbreak_current"].shift(1)
    df_merged["spatial_lag_outbreak_2m"] = df_merged.groupby("adm4_pcode")["spatial_outbreak_current"].shift(2)

    df_merged = df_merged.drop(columns=["spatial_cases_current", "spatial_outbreak_current"])
    df_final = df_merged.dropna(subset=["pr_total_mm_lag_4m"]).copy()

    output_csv_path = processed_data_dir / "cchain_cdo_dengue_surveillance_ready.csv"
    df_final.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] Exported processed matrix to: {output_csv_path} ({df_final.shape[0]} rows x {df_final.shape[1]} cols)")
    return df_final


# ------------------------------------------------------------------------------
# 2. FEATURE PARTITIONS
# ------------------------------------------------------------------------------
FEATURES_30D_HORIZON = [
    "pr_total_mm_lag_1m", "pr_total_mm_lag_2m", "pr_total_mm_lag_3m", "pr_rolling_3m_lag1m",
    "heat_index_mean_lag_1m", "heat_index_mean_lag_2m", "heat_index_mean_lag_3m",
    "tave_mean_lag_1m", "tave_mean_lag_2m", "tave_mean_lag_3m",
    "rh_mean_lag_1m", "rh_mean_lag_2m",
    "google_bldgs_count", "google_bldgs_density", "google_bldgs_pct_built_up_area", "google_bldgs_area_mean",
    "brgy_total_area", "brgy_is_coastal", "pop_density_imputed",
    "runoff_risk_lag1m", "runoff_risk_lag2m", "urban_heat_trap_lag1m", "host_exposure_index",
    "dengue_cases_lag_1m", "is_outbreak_lag_1m",
    "spatial_lag_cases_1m", "spatial_lag_outbreak_1m"
]

FEATURES_60D_HORIZON = [
    "pr_total_mm_lag_2m", "pr_total_mm_lag_3m", "pr_total_mm_lag_4m", "pr_rolling_3m_lag2m",
    "heat_index_mean_lag_2m", "heat_index_mean_lag_3m", "heat_index_mean_lag_4m",
    "tave_mean_lag_2m", "tave_mean_lag_3m", "tave_mean_lag_4m",
    "rh_mean_lag_2m", "rh_mean_lag_3m",
    "google_bldgs_count", "google_bldgs_density", "google_bldgs_pct_built_up_area", "google_bldgs_area_mean",
    "brgy_total_area", "brgy_is_coastal", "pop_density_imputed",
    "runoff_risk_lag2m", "urban_heat_trap_lag2m", "host_exposure_index",
    "dengue_cases_lag_2m", "is_outbreak_lag_2m",
    "spatial_lag_cases_2m", "spatial_lag_outbreak_2m"
]


def optimize_fbeta_threshold(y_true, y_probs, beta=2.0):
    """Finds optimal classification threshold maximizing F-beta score."""
    best_th = 0.5
    best_f = 0.0
    for th in np.linspace(0.05, 0.95, 91):
        preds = (y_probs >= th).astype(int)
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if score > best_f:
            best_f = score
            best_th = th
    return best_th, best_f


def evaluate_models_for_horizon(df, train_mask, test_mask, horizon_name, feature_list, pos_weight):
    """Trains and benchmarks 4 algorithms with F2-score threshold calibration."""
    print("\n" + "-" * 80)
    print(f"[*] RUNNING BENCHMARK FOR: {horizon_name.upper()} ({len(feature_list)} features)")
    print("-" * 80)

    X_train = df.loc[train_mask, feature_list].fillna(0)
    X_test = df.loc[test_mask, feature_list].fillna(0)
    y_train = df.loc[train_mask, "is_outbreak"].values
    y_test = df.loc[test_mask, "is_outbreak"].values

    models = {
        "Logistic Regression (L2 Balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
        "LightGBM Classifier": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.03, scale_pos_weight=pos_weight,
            random_state=42, verbose=-1
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03, scale_pos_weight=pos_weight,
            random_state=42, eval_metric="logloss"
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        if hasattr(model, "predict_proba"):
            probs_test = model.predict_proba(X_test)[:, 1]
            probs_train = model.predict_proba(X_train)[:, 1]
        else:
            probs_test = model.predict(X_test)
            probs_train = model.predict(X_train)

        opt_th, _ = optimize_fbeta_threshold(y_train, probs_train, beta=2.0)
        preds_opt = (probs_test >= opt_th).astype(int)

        roc_auc = roc_auc_score(y_test, probs_test)
        pr_auc = average_precision_score(y_test, probs_test)
        f2_opt = fbeta_score(y_test, preds_opt, beta=2.0, zero_division=0)
        f1_opt = f1_score(y_test, preds_opt, zero_division=0)
        recall_opt = recall_score(y_test, preds_opt, zero_division=0)
        precision_opt = precision_score(y_test, preds_opt, zero_division=0)
        brier = brier_score_loss(y_test, probs_test)

        results.append({
            "Horizon": horizon_name,
            "Model": name,
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Optimal Thresh": round(opt_th, 2),
            "F2-Score (Opt)": round(f2_opt, 4),
            "F1-Score (Opt)": round(f1_opt, 4),
            "Recall (Sensitivity)": round(recall_opt, 4),
            "Precision": round(precision_opt, 4),
            "Brier Score": round(brier, 4)
        })

    df_res = pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False)
    print(df_res.to_string(index=False))

    best_model_name = df_res.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[+] Top 8 Most Predictive Features for {best_model_name} ({horizon_name}):")
    if hasattr(best_model, "feature_importances_"):
        imps = pd.Series(best_model.feature_importances_, index=feature_list).sort_values(ascending=False)
        for rank, (f, imp) in enumerate(imps.head(8).items(), 1):
            print(f"    {rank}. {f:<30} (Weight: {imp:.4f})")

    return df_res, trained_models


def main():
    print("=" * 80)
    print(f"[*] CCHAIN ADVANCED SPATIAL-TEMPORAL SURVEILLANCE PIPELINE")
    print(f"[*] Pilot City: {PILOT_CITY_NAME} ({PILOT_CITY_CODE}) | Target Disease: {TARGET_DISEASE}")
    print("=" * 80)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    df_final = load_and_preprocess_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    train_mask = df_final["date"] < "2019-01-01"
    test_mask = df_final["date"] >= "2019-01-01"
    y_train = df_final.loc[train_mask, "is_outbreak"].values
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    print(f"\n[i] Temporal Split -> Train: {train_mask.sum()} samples | Test: {test_mask.sum()} samples")
    print(f"[i] Class Imbalance Scale Weight: {pos_weight:.2f}")

    results_30d, _ = evaluate_models_for_horizon(df_final, train_mask, test_mask, "30-Day Early Warning (T+1)", FEATURES_30D_HORIZON, pos_weight)
    results_60d, _ = evaluate_models_for_horizon(df_final, train_mask, test_mask, "60-Day Early Warning (T+2)", FEATURES_60D_HORIZON, pos_weight)

    all_benchmarks = pd.concat([results_30d, results_60d], ignore_index=True)
    benchmark_csv_path = PROCESSED_DATA_DIR / "cchain_model_benchmarks.csv"
    all_benchmarks.to_csv(benchmark_csv_path, index=False)
    print(f"\n[SUCCESS] Exported benchmark results table to: {benchmark_csv_path}")
    print("\n" + "=" * 80)
    print("[*] PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
