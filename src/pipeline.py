"""
Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine
Production Pipeline (Phases 1-4)
- Zero Future Climate Leakage (30-Day and 60-Day Lead Forecasts)
- Spatial Adjacency Contiguity Matrix (W * Y)
- Urban-Climate Physical Interactions & Deterministic Demographics
- Multi-Model Benchmarking (Logistic Regression, Random Forest, LightGBM, XGBoost) & F2-Score Optimization
"""

import os
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
    brier_score_loss,
    accuracy_score,
    confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def find_raw_data_dir(raw_dir: Path = None, base_dir: Path = None) -> Path:
    if base_dir is None:
        base_dir = get_project_root()
    if raw_dir and (Path(raw_dir) / "location.csv").exists():
        return Path(raw_dir)
    candidates = [
        raw_dir,
        base_dir / "data" / "cchain_raw",
        base_dir.parent.parent / "datasets" / "cchain_raw",
        base_dir.parent / "datasets" / "cchain_raw",
        base_dir.parent.parent / "cchain_raw",
        base_dir.parent / "cchain_raw",
        Path("../../datasets/cchain_raw"),
        Path("../datasets/cchain_raw"),
        Path("datasets/cchain_raw"),
        Path("data/cchain_raw"),
    ]
    for c in candidates:
        if c and (Path(c) / "location.csv").exists():
            return Path(c)
    return base_dir.parent.parent / "datasets" / "cchain_raw"

def run_production_pipeline(raw_dir: Path = None, processed_dir: Path = None):
    BASE_DIR = get_project_root()
    RAW_DATA_DIR = find_raw_data_dir(raw_dir, BASE_DIR)
    PROCESSED_DATA_DIR = Path(processed_dir) if processed_dir else BASE_DIR / "data" / "processed"
    DOCS_DIR = BASE_DIR / "docs"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    PILOT_CITY_CODE = "PH104305000"  # Cagayan de Oro City
    PILOT_CITY_NAME = "Cagayan de Oro City"
    TARGET_DISEASE = "DENGUE FEVER"

    print("=" * 85)
    print(f"[*] CCHAIN ADVANCED SPATIAL-TEMPORAL SURVEILLANCE PIPELINE")
    print(f"[*] Pilot City: {PILOT_CITY_NAME} ({PILOT_CITY_CODE}) | Target Disease: {TARGET_DISEASE}")
    print(f"[*] Project Root: {BASE_DIR.resolve()}")
    print("=" * 85)

    # 1. LOAD GEOGRAPHY & BUILD SPATIAL CONTIGUITY MATRIX (W)
    print("\n[1/6] Loading location.csv & brgy_geography.csv (Building Spatial Contiguity Matrix W)...")
    df_loc = pd.read_csv(RAW_DATA_DIR / "location.csv")
    cdo_brgys = df_loc[df_loc["adm3_pcode"] == PILOT_CITY_CODE][
        ["adm1_en", "adm2_en", "adm3_pcode", "adm3_en", "adm4_pcode", "adm4_en", "brgy_total_area"]
    ].drop_duplicates(subset=["adm4_pcode"]).reset_index(drop=True)

    target_pcodes = sorted(cdo_brgys["adm4_pcode"].unique().tolist())
    num_brgys = len(target_pcodes)
    print(f"[+] Identified {num_brgys} unique barangays in {PILOT_CITY_NAME}.")

    df_geo = pd.read_csv(RAW_DATA_DIR / "brgy_geography.csv")
    cdo_geo = df_geo[df_geo["adm4_pcode"].isin(target_pcodes)].drop_duplicates(subset=["adm4_pcode"]).copy()

    cdo_geo["poly"] = cdo_geo["geometry"].apply(shapely.wkt.loads)
    pcode_to_poly = dict(zip(cdo_geo["adm4_pcode"], cdo_geo["poly"]))

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

    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_spatial = adj_matrix / row_sums

    print(f"[+] Spatial Contiguity Matrix W constructed: {W_spatial.shape[0]}x{W_spatial.shape[1]}")
    print(f"[+] Total spatial neighbor edges: {int(adj_matrix.sum())} (Avg: {adj_matrix.sum()/num_brgys:.2f} neighbors/barangay)")

    coastal_map = dict(zip(cdo_geo["adm4_pcode"], cdo_geo["brgy_is_coastal"].astype(int)))
    cdo_brgys["brgy_is_coastal"] = cdo_brgys["adm4_pcode"].map(coastal_map).fillna(0).astype(int)

    # 2. LOAD & AGGREGATE HEALTH RECORDS (DENGUE)
    print("\n[2/6] Loading disease_lgu_disaggregated_totals.csv...")
    df_lgu = pd.read_csv(RAW_DATA_DIR / "disease_lgu_disaggregated_totals.csv")
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
    print(f"[+] Aggregated {len(df_health_agg)} barangay-month dengue surveillance records.")

    # 3. LOAD & AGGREGATE CLIMATE ATMOSPHERE DATA
    print("\n[3/6] Loading climate_atmosphere.csv (ERA5-Land)...")
    df_clim = pd.read_csv(RAW_DATA_DIR / "climate_atmosphere.csv")
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
    print(f"[+] Aggregated {len(df_clim_agg)} barangay-month atmospheric climate records.")

    # 4. LOAD BUILT-ENVIRONMENT & POPULATION COVARIATES
    print("\n[4/6] Loading google_open_buildings.csv and worldpop_population.csv...")
    df_bldgs = pd.read_csv(RAW_DATA_DIR / "google_open_buildings.csv")
    df_bldgs_cdo = df_bldgs[df_bldgs["adm4_pcode"].isin(target_pcodes)][
        ["adm4_pcode", "google_bldgs_count", "google_bldgs_density", "google_bldgs_pct_built_up_area", "google_bldgs_area_mean"]
    ].drop_duplicates(subset=["adm4_pcode"])

    df_pop = pd.read_csv(RAW_DATA_DIR / "worldpop_population.csv")
    df_pop_cdo = df_pop[df_pop["adm4_pcode"].isin(target_pcodes)].sort_values(by="date").groupby("adm4_pcode").last().reset_index()[
        ["adm4_pcode", "pop_count_total"]
    ]

    df_static = df_bldgs_cdo.merge(df_pop_cdo, on="adm4_pcode", how="left")

    # 5. SPACE-TIME GRID ALIGNMENT & FEATURE ENGINEERING
    print("\n[5/6] Aligning Full Space-Time Grid & Engineering Advanced Physical-Spatial Features...")
    min_date = df_clim_agg["date"].min()
    max_date = df_clim_agg["date"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="MS")

    grid_idx = pd.MultiIndex.from_product([target_pcodes, all_dates], names=["adm4_pcode", "date"]).to_frame().reset_index(drop=True)

    df_merged = grid_idx.merge(cdo_brgys, on="adm4_pcode", how="left")
    df_merged = df_merged.merge(df_clim_agg, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(df_health_agg, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(df_static, on="adm4_pcode", how="left")

    df_merged["dengue_cases"] = df_merged["dengue_cases"].fillna(0).astype(int)
    df_merged["dengue_deaths"] = df_merged["dengue_deaths"].fillna(0).astype(int)
    df_merged["pop_density_imputed"] = df_merged["pop_count_total"] / df_merged["brgy_total_area"]

    # Outbreak Ground Truth Definition: 75th percentile per barangay strictly computed on pre-2019 training data
    # (prevents future test-period target leakage while enforcing min floor of 5 cases)
    train_slice = df_merged[df_merged["date"] < "2019-01-01"]
    p75_training_thresholds = (
        train_slice.groupby("adm4_pcode")["dengue_cases"]
        .quantile(0.75)
        .apply(lambda q: max(5.0, float(q)))
        .to_dict()
    )
    df_merged["brgy_p75_threshold"] = df_merged["adm4_pcode"].map(p75_training_thresholds).fillna(5.0)
    df_merged["is_outbreak"] = (df_merged["dengue_cases"] >= df_merged["brgy_p75_threshold"]).astype(int)

    df_merged = df_merged.sort_values(by=["adm4_pcode", "date"]).reset_index(drop=True)

    # Meteorological Lags (1, 2, 3, 4 months)
    for lag in [1, 2, 3, 4]:
        df_merged[f"pr_total_mm_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["pr_monthly_total_mm"].shift(lag)
        df_merged[f"heat_index_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["heat_index_monthly_mean_c"].shift(lag)
        df_merged[f"tave_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["tave_monthly_mean_c"].shift(lag)
        df_merged[f"rh_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["rh_monthly_mean_pct"].shift(lag)

    # 3-month rolling averages (lagged to prevent leakage)
    df_merged["pr_rolling_3m_lag1m"] = (
        df_merged["pr_total_mm_lag_1m"] + df_merged["pr_total_mm_lag_2m"] + df_merged["pr_total_mm_lag_3m"]
    ) / 3.0

    df_merged["pr_rolling_3m_lag2m"] = (
        df_merged["pr_total_mm_lag_2m"] + df_merged["pr_total_mm_lag_3m"] + df_merged["pr_total_mm_lag_4m"]
    ) / 3.0

    # Urban-Climate Physical Interaction Features
    df_merged["runoff_risk_lag1m"] = df_merged["google_bldgs_pct_built_up_area"] * df_merged["pr_total_mm_lag_1m"]
    df_merged["runoff_risk_lag2m"] = df_merged["google_bldgs_pct_built_up_area"] * df_merged["pr_total_mm_lag_2m"]
    df_merged["urban_heat_trap_lag1m"] = df_merged["google_bldgs_density"] * df_merged["heat_index_mean_lag_1m"]
    df_merged["urban_heat_trap_lag2m"] = df_merged["google_bldgs_density"] * df_merged["heat_index_mean_lag_2m"]
    df_merged["host_exposure_index"] = df_merged["pop_density_imputed"] * df_merged["google_bldgs_pct_built_up_area"]

    # Autoregressive Case Lags
    for lag in [1, 2]:
        df_merged[f"dengue_cases_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["dengue_cases"].shift(lag)
        df_merged[f"is_outbreak_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["is_outbreak"].shift(lag)

    # Spatial Contiguity Autoregressive Lags (W * Y)
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

    output_csv_path = PROCESSED_DATA_DIR / "cchain_cdo_dengue_surveillance_ready.csv"
    df_final.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] Exported enhanced surveillance dataset to: {output_csv_path}")
    print(f"[i] Dimensions: {df_final.shape[0]} rows x {df_final.shape[1]} columns")

    # 6. MULTI-MODEL TOURNAMENT & F2-SCORE CALIBRATION
    print("\n" + "=" * 85)
    print("[*] MULTI-MODEL BENCHMARKING & OPERATIONAL LEAD-TIME EVALUATION")
    print("=" * 85)

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

    train_mask = df_final["date"] < "2019-01-01"
    test_mask = df_final["date"] >= "2019-01-01"

    y_train = df_final.loc[train_mask, "is_outbreak"].values
    y_test = df_final.loc[test_mask, "is_outbreak"].values
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

    def optimize_fbeta_threshold(y_true, y_probs, beta=2.0):
        best_th = 0.50
        best_f = 0.0
        for th in np.linspace(0.05, 0.95, 91):
            preds = (y_probs >= th).astype(int)
            score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
            if score > best_f:
                best_f = score
                best_th = th
        return best_th, best_f

    def evaluate_models_for_horizon(horizon_name, feature_list):
        print("\n" + "-" * 85)
        print(f"[*] RUNNING BENCHMARK FOR: {horizon_name.upper()} ({len(feature_list)} features)")
        print("-" * 85)

        X_train = df_final.loc[train_mask, feature_list].fillna(0)
        X_test = df_final.loc[test_mask, feature_list].fillna(0)

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.03, scale_pos_weight=pos_weight,
                random_state=42, verbose=-1
            ),
            "XGBoost": xgb.XGBClassifier(
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
            acc = accuracy_score(y_test, preds_opt)
            rec = recall_score(y_test, preds_opt, zero_division=0)
            prec = precision_score(y_test, preds_opt, zero_division=0)
            f1_opt = f1_score(y_test, preds_opt, zero_division=0)
            f2_opt = fbeta_score(y_test, preds_opt, beta=2.0, zero_division=0)
            brier = brier_score_loss(y_test, probs_test)

            results.append({
                "Horizon": horizon_name,
                "Model": name,
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Accuracy": round(acc, 4),
                "Optimal Thresh": round(opt_th, 2),
                "Sensitivity (Recall)": round(rec, 4),
                "Precision": round(prec, 4),
                "F1-Score": round(f1_opt, 4),
                "F2-Score (Opt)": round(f2_opt, 4),
                "Brier Score": round(brier, 4)
            })

        df_res = pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False)
        print(df_res.to_string(index=False))
        return df_res, trained_models

    res_30d, _ = evaluate_models_for_horizon("30-Day Lead (T+1)", FEATURES_30D_HORIZON)
    res_60d, _ = evaluate_models_for_horizon("60-Day Lead (T+2)", FEATURES_60D_HORIZON)

    all_benchmarks = pd.concat([res_30d, res_60d], ignore_index=True)
    benchmark_csv_path = PROCESSED_DATA_DIR / "cchain_model_benchmarks.csv"
    all_benchmarks.to_csv(benchmark_csv_path, index=False)
    print(f"\n[SUCCESS] Exported benchmark results table to: {benchmark_csv_path}")

    print("\n" + "=" * 85)
    print("[*] SURVEILLANCE PIPELINE EXECUTION COMPLETE")
    print("=" * 85)
    return all_benchmarks, df_final

if __name__ == "__main__":
    run_production_pipeline()
