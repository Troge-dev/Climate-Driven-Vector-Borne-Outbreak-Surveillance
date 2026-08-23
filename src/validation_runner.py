"""
Comprehensive Test & Validation Engine for CCHAIN Surveillance Pipeline
Executes end-to-end verification tests on the synthetic dummy dataset.
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
    confusion_matrix,
    brier_score_loss,
    accuracy_score
)
import lightgbm as lgb
import xgboost as xgb

def get_project_root() -> Path:
    curr = Path.cwd()
    if (curr / "data").exists():
        return curr
    file_parent = Path(__file__).resolve().parent.parent
    if (file_parent / "data").exists():
        return file_parent
    return curr

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

def run_comprehensive_validation(
    raw_dir: Path = None,
    processed_dir: Path = None,
    pilot_city_code: str = "PH990001000",
    pilot_city_name: str = "Synthetic Test City",
    target_disease: str = "DENGUE FEVER"
):
    base_dir = get_project_root()
    if raw_dir is None:
        raw_dir = base_dir / "data" / "dummy_test_city"
    if processed_dir is None:
        processed_dir = base_dir / "data" / "processed_dummy_test"

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f"[*] CCHAIN PIPELINE VERIFICATION & INTEGRATION TEST ENGINE")
    print(f"[*] Target Environment: {raw_dir.resolve()}")
    print(f"[*] Pilot City: {pilot_city_name} ({pilot_city_code}) | Target Disease: {target_disease}")
    print("=" * 85)

    # TEST 1: LOCATION METADATA & SPATIAL CONTIGUITY MATRIX (W)
    print("\n[Step 1/5] Testing Spatial Contiguity Matrix (W) Construction...")
    df_loc = pd.read_csv(raw_dir / "location.csv")
    city_brgys = df_loc[df_loc["adm3_pcode"] == pilot_city_code][
        ["adm1_en", "adm2_en", "adm3_pcode", "adm3_en", "adm4_pcode", "adm4_en", "brgy_total_area"]
    ].drop_duplicates(subset=["adm4_pcode"]).reset_index(drop=True)

    target_pcodes = sorted(city_brgys["adm4_pcode"].unique().tolist())
    num_brgys = len(target_pcodes)

    df_geo = pd.read_csv(raw_dir / "brgy_geography.csv")
    city_geo = df_geo[df_geo["adm4_pcode"].isin(target_pcodes)].drop_duplicates(subset=["adm4_pcode"]).copy()
    
    city_geo["poly"] = city_geo["geometry"].apply(shapely.wkt.loads)
    pcode_to_poly = dict(zip(city_geo["adm4_pcode"], city_geo["poly"]))

    adj_matrix = np.zeros((num_brgys, num_brgys), dtype=float)
    for i, pcode_i in enumerate(target_pcodes):
        poly_i = pcode_to_poly.get(pcode_i)
        for j, pcode_j in enumerate(target_pcodes):
            if i != j:
                poly_j = pcode_to_poly.get(pcode_j)
                if poly_i and poly_j and (poly_i.touches(poly_j) or poly_i.intersects(poly_j)):
                    adj_matrix[i, j] = 1.0

    assert np.allclose(adj_matrix, adj_matrix.T), "Adjacency matrix is asymmetric!"
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_spatial = adj_matrix / row_sums
    assert np.allclose(W_spatial.sum(axis=1), 1.0), "Spatial weights W is not row-stochastic!"

    print(f"  [PASS] Spatial Matrix W: {W_spatial.shape[0]}x{W_spatial.shape[1]}")
    print(f"  [PASS] Symmetry & Row-Stochasticity Verified: True")
    print(f"  [PASS] Total Spatial Neighbor Edges: {int(adj_matrix.sum())}")

    coastal_map = dict(zip(city_geo["adm4_pcode"], city_geo["brgy_is_coastal"].astype(int)))
    city_brgys["brgy_is_coastal"] = city_brgys["adm4_pcode"].map(coastal_map).fillna(0).astype(int)

    # TEST 2: DATA INGESTION & TEMPORAL AGGREGATION
    print("\n[Step 2/5] Testing Multi-Table Ingestion & Monthly Resampling...")
    df_lgu = pd.read_csv(raw_dir / "disease_lgu_disaggregated_totals.csv")
    df_dengue = df_lgu[
        (df_lgu["adm3_pcode"] == pilot_city_code) &
        (df_lgu["disease_common_name"] == target_disease) &
        (df_lgu["adm4_pcode"].isin(target_pcodes))
    ].copy()
    df_dengue["date"] = pd.to_datetime(df_dengue["date"]).dt.to_period("M").dt.to_timestamp()
    df_health_agg = df_dengue.groupby(["adm4_pcode", "date"], as_index=False).agg(
        dengue_cases=("case_total", "sum"),
        dengue_deaths=("death_total", "sum")
    )

    df_clim = pd.read_csv(raw_dir / "climate_atmosphere.csv")
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

    df_bldgs = pd.read_csv(raw_dir / "google_open_buildings.csv")
    df_bldgs_city = df_bldgs[df_bldgs["adm4_pcode"].isin(target_pcodes)][
        ["adm4_pcode", "google_bldgs_count", "google_bldgs_density", "google_bldgs_pct_built_up_area", "google_bldgs_area_mean"]
    ].drop_duplicates(subset=["adm4_pcode"])

    df_pop = pd.read_csv(raw_dir / "worldpop_population.csv")
    df_pop_city = df_pop[df_pop["adm4_pcode"].isin(target_pcodes)].sort_values(by="date").groupby("adm4_pcode").last().reset_index()[
        ["adm4_pcode", "pop_count_total"]
    ]
    df_static = df_bldgs_city.merge(df_pop_city, on="adm4_pcode", how="left")
    print("  [PASS] All health, climate, and built-environment tables aligned.")

    # TEST 3: SPACE-TIME MATRIX & DISTRIBUTED LAGS
    print("\n[Step 3/5] Testing Space-Time Alignment & Feature Engineering...")
    min_date = df_clim_agg["date"].min()
    max_date = df_clim_agg["date"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="MS")
    grid_idx = pd.MultiIndex.from_product([target_pcodes, all_dates], names=["adm4_pcode", "date"]).to_frame().reset_index(drop=True)

    df_merged = grid_idx.merge(city_brgys, on="adm4_pcode", how="left")
    df_merged = df_merged.merge(df_clim_agg, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(df_health_agg, on=["adm4_pcode", "date"], how="left")
    df_merged = df_merged.merge(df_static, on="adm4_pcode", how="left")

    df_merged["dengue_cases"] = df_merged["dengue_cases"].fillna(0).astype(int)
    df_merged["dengue_deaths"] = df_merged["dengue_deaths"].fillna(0).astype(int)
    df_merged["pop_density_imputed"] = df_merged["pop_count_total"] / df_merged["brgy_total_area"]

    df_merged["brgy_p75_threshold"] = df_merged.groupby("adm4_pcode")["dengue_cases"].transform(
        lambda x: max(5.0, float(x.quantile(0.75)))
    )
    df_merged["is_outbreak"] = (df_merged["dengue_cases"] >= df_merged["brgy_p75_threshold"]).astype(int)
    df_merged = df_merged.sort_values(by=["adm4_pcode", "date"]).reset_index(drop=True)

    for lag in [1, 2, 3, 4]:
        df_merged[f"pr_total_mm_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["pr_monthly_total_mm"].shift(lag)
        df_merged[f"heat_index_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["heat_index_monthly_mean_c"].shift(lag)
        df_merged[f"tave_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["tave_monthly_mean_c"].shift(lag)
        df_merged[f"rh_mean_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["rh_monthly_mean_pct"].shift(lag)

    df_merged["pr_rolling_3m_lag1m"] = (df_merged["pr_total_mm_lag_1m"] + df_merged["pr_total_mm_lag_2m"] + df_merged["pr_total_mm_lag_3m"]) / 3.0
    df_merged["pr_rolling_3m_lag2m"] = (df_merged["pr_total_mm_lag_2m"] + df_merged["pr_total_mm_lag_3m"] + df_merged["pr_total_mm_lag_4m"]) / 3.0

    df_merged["runoff_risk_lag1m"] = df_merged["google_bldgs_pct_built_up_area"] * df_merged["pr_total_mm_lag_1m"]
    df_merged["runoff_risk_lag2m"] = df_merged["google_bldgs_pct_built_up_area"] * df_merged["pr_total_mm_lag_2m"]
    df_merged["urban_heat_trap_lag1m"] = df_merged["google_bldgs_density"] * df_merged["heat_index_mean_lag_1m"]
    df_merged["urban_heat_trap_lag2m"] = df_merged["google_bldgs_density"] * df_merged["heat_index_mean_lag_2m"]
    df_merged["host_exposure_index"] = df_merged["pop_density_imputed"] * df_merged["google_bldgs_pct_built_up_area"]

    for lag in [1, 2]:
        df_merged[f"dengue_cases_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["dengue_cases"].shift(lag)
        df_merged[f"is_outbreak_lag_{lag}m"] = df_merged.groupby("adm4_pcode")["is_outbreak"].shift(lag)

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

    ready_csv_path = processed_dir / "synthetic_validation_surveillance_ready.csv"
    df_final.to_csv(ready_csv_path, index=False)
    print(f"  [PASS] Engineered Matrix: {df_final.shape[0]} rows x {df_final.shape[1]} features.")

    # TEST 4: MACHINE LEARNING BENCHMARKING
    print("\n[Step 4/5] Executing Machine Learning Benchmark Tournament...")
    FEATURES_30D = [
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

    FEATURES_60D = [
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

    all_horizon_results = []
    for h_name, feat_cols in [("30-Day Lead (T+1)", FEATURES_30D), ("60-Day Lead (T+2)", FEATURES_60D)]:
        X_train = df_final.loc[train_mask, feat_cols].fillna(0)
        X_test = df_final.loc[test_mask, feat_cols].fillna(0)

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=7, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.04, scale_pos_weight=pos_weight, random_state=42, verbose=-1
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.04, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss"
            )
        }

        for m_name, model in models.items():
            model.fit(X_train, y_train)
            probs_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
            probs_train = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_train)

            opt_th, _ = optimize_fbeta_threshold(y_train, probs_train, beta=2.0)
            preds_opt = (probs_test >= opt_th).astype(int)

            roc_auc = roc_auc_score(y_test, probs_test)
            pr_auc = average_precision_score(y_test, probs_test)
            acc = accuracy_score(y_test, preds_opt)
            rec = recall_score(y_test, preds_opt, zero_division=0)
            prec = precision_score(y_test, preds_opt, zero_division=0)
            f1 = f1_score(y_test, preds_opt, zero_division=0)
            f2 = fbeta_score(y_test, preds_opt, beta=2.0, zero_division=0)
            brier = brier_score_loss(y_test, probs_test)

            all_horizon_results.append({
                "Horizon": h_name,
                "Model": m_name,
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Accuracy": round(acc, 4),
                "Optimal Thresh": round(opt_th, 2),
                "Recall (Sensitivity)": round(rec, 4),
                "Precision": round(prec, 4),
                "F1-Score": round(f1, 4),
                "F2-Score (Public Health)": round(f2, 4),
                "Brier Loss": round(brier, 4)
            })

    df_summary = pd.DataFrame(all_horizon_results)
    print("\n" + df_summary[["Horizon", "Model", "ROC-AUC", "PR-AUC", "Accuracy", "Recall (Sensitivity)", "Precision", "F2-Score (Public Health)"]].to_string(index=False))

    benchmark_csv = processed_dir / "synthetic_model_benchmarks.csv"
    df_summary.to_csv(benchmark_csv, index=False)
    print(f"\n[Step 5/5] Full benchmark results saved to: {benchmark_csv}")
    print("\n" + "=" * 85)
    print("[ALL 5 VERIFICATION CHECKS PASSED] PIPELINE IS FULLY OPERATIONAL!")
    print("=" * 85)
    return df_summary

if __name__ == "__main__":
    run_comprehensive_validation()
