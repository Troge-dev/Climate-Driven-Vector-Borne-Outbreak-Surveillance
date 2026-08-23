"""
Randomized Stress-Testing & Counterfactual Sensitivity Engine
Evaluates machine learning model behavior, stability, and biological dose-response
under randomized, extreme, adversarial, and counterfactual synthetic inputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def run_stress_testing(
    processed_dir: Path = None,
    random_seed: int = 42
):
    np.random.seed(random_seed)
    base_dir = get_project_root()
    if processed_dir is None:
        processed_dir = base_dir / "data" / "processed_dummy_test"
    processed_dir = Path(processed_dir)
    data_file = processed_dir / "synthetic_validation_surveillance_ready.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Missing {data_file}. Run validation_runner.py first.")

    df = pd.read_csv(data_file)
    
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

    train_mask = df["date"] < "2019-01-01"
    X_train = df.loc[train_mask, FEATURES_30D].fillna(0)
    y_train = df.loc[train_mask, "is_outbreak"].values
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

    print("[*] Training baseline models on synthetic training history...")
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

    for name, m in models.items():
        m.fit(X_train, y_train)

    # 1. COUNTERFACTUAL SCENARIOS
    print("\n" + "=" * 80)
    print(">> EXPERIMENT 1: Controlled Counterfactual Stress-Testing Scenarios")
    print("=" * 80)
    
    baseline_median = X_train.median()

    scenarios = {
        "1. Baseline Normal (Typical Dry Month)": baseline_median.copy(),
        "2. Hyper-Epidemic Storm (Lagged Deluge + Heat Spike + Dense Slum)": baseline_median.copy(),
        "3. Prolonged Extreme Drought (Zero Rain, High Heat, Low Mosquito Breeding)": baseline_median.copy(),
        "4. High-Altitude Cold Shock (Temperature < 18C, Suppressed EIP)": baseline_median.copy(),
        "5. Severe Contagion Spillover (Surrounding Barangays in Outbreak W*Y=1.0)": baseline_median.copy(),
        "6. Rural Vegetative Sparse Zone (2% Built-up, Low Pop Density)": baseline_median.copy(),
        "7. Flash Urban Flooding with Moderate Temp": baseline_median.copy()
    }

    s2 = scenarios["2. Hyper-Epidemic Storm (Lagged Deluge + Heat Spike + Dense Slum)"]
    s2["pr_total_mm_lag_1m"] = 380.0
    s2["pr_total_mm_lag_2m"] = 450.0
    s2["pr_total_mm_lag_3m"] = 320.0
    s2["pr_rolling_3m_lag1m"] = 383.33
    s2["heat_index_mean_lag_1m"] = 39.5
    s2["heat_index_mean_lag_2m"] = 41.2
    s2["heat_index_mean_lag_3m"] = 38.0
    s2["google_bldgs_pct_built_up_area"] = 85.0
    s2["google_bldgs_density"] = 0.0035
    s2["pop_density_imputed"] = 28000.0
    s2["runoff_risk_lag1m"] = 85.0 * 380.0
    s2["runoff_risk_lag2m"] = 85.0 * 450.0
    s2["urban_heat_trap_lag1m"] = 0.0035 * 39.5
    s2["host_exposure_index"] = 28000.0 * 85.0
    s2["spatial_lag_cases_1m"] = 18.0
    s2["spatial_lag_outbreak_1m"] = 1.0

    s3 = scenarios["3. Prolonged Extreme Drought (Zero Rain, High Heat, Low Mosquito Breeding)"]
    s3["pr_total_mm_lag_1m"] = 2.0
    s3["pr_total_mm_lag_2m"] = 0.5
    s3["pr_total_mm_lag_3m"] = 0.0
    s3["pr_rolling_3m_lag1m"] = 0.83
    s3["heat_index_mean_lag_1m"] = 38.5
    s3["heat_index_mean_lag_2m"] = 40.0
    s3["rh_mean_lag_1m"] = 48.0
    s3["rh_mean_lag_2m"] = 45.0
    s3["runoff_risk_lag1m"] = s3["google_bldgs_pct_built_up_area"] * 2.0
    s3["runoff_risk_lag2m"] = s3["google_bldgs_pct_built_up_area"] * 0.5

    s4 = scenarios["4. High-Altitude Cold Shock (Temperature < 18C, Suppressed EIP)"]
    s4["tave_mean_lag_1m"] = 16.5
    s4["tave_mean_lag_2m"] = 15.8
    s4["tave_mean_lag_3m"] = 16.0
    s4["heat_index_mean_lag_1m"] = 16.5
    s4["heat_index_mean_lag_2m"] = 15.8

    s5 = scenarios["5. Severe Contagion Spillover (Surrounding Barangays in Outbreak W*Y=1.0)"]
    s5["spatial_lag_cases_1m"] = 45.0
    s5["spatial_lag_outbreak_1m"] = 1.0
    s5["dengue_cases_lag_1m"] = 12.0
    s5["is_outbreak_lag_1m"] = 1.0

    s6 = scenarios["6. Rural Vegetative Sparse Zone (2% Built-up, Low Pop Density)"]
    s6["google_bldgs_pct_built_up_area"] = 2.5
    s6["google_bldgs_density"] = 0.00005
    s6["pop_density_imputed"] = 250.0
    s6["host_exposure_index"] = 2.5 * 250.0
    s6["runoff_risk_lag1m"] = 2.5 * s6["pr_total_mm_lag_1m"]

    s7 = scenarios["7. Flash Urban Flooding with Moderate Temp"]
    s7["pr_total_mm_lag_1m"] = 420.0
    s7["pr_total_mm_lag_2m"] = 390.0
    s7["runoff_risk_lag1m"] = s7["google_bldgs_pct_built_up_area"] * 420.0
    s7["runoff_risk_lag2m"] = s7["google_bldgs_pct_built_up_area"] * 390.0

    df_scenarios = pd.DataFrame(scenarios).T[FEATURES_30D]
    
    scenario_results = []
    for sc_name, row in df_scenarios.iterrows():
        input_df = pd.DataFrame([row], columns=FEATURES_30D)
        res_row = {"Scenario": sc_name}
        for m_name, model in models.items():
            prob = model.predict_proba(input_df)[0, 1]
            res_row[m_name] = round(float(prob), 4)
        scenario_results.append(res_row)

    df_sc_res = pd.DataFrame(scenario_results)
    print(df_sc_res.to_string(index=False))

    # 2. MONTE CARLO PROBING
    print("\n" + "=" * 80)
    print(">> EXPERIMENT 2: Monte Carlo Random Feature State-Space Probing (N=2,000)")
    print("=" * 80)

    N_RANDOM = 2000
    feat_mins = X_train.min()
    feat_maxs = X_train.max()
    random_matrix = np.random.uniform(feat_mins.values, feat_maxs.values, size=(N_RANDOM, len(FEATURES_30D)))
    df_random = pd.DataFrame(random_matrix, columns=FEATURES_30D)

    mc_preds = {}
    for m_name, model in models.items():
        probs = model.predict_proba(df_random)[:, 1]
        mc_preds[m_name] = probs

    df_mc_summary = pd.DataFrame({
        "Model": list(models.keys()),
        "Mean Prob": [round(float(np.mean(mc_preds[m])), 4) for m in models],
        "Std Dev": [round(float(np.std(mc_preds[m])), 4) for m in models],
        "Min Prob": [round(float(np.min(mc_preds[m])), 4) for m in models],
        "25th %": [round(float(np.percentile(mc_preds[m], 25)), 4) for m in models],
        "Median %": [round(float(np.median(mc_preds[m])), 4) for m in models],
        "75th %": [round(float(np.percentile(mc_preds[m], 75)), 4) for m in models],
        "Max Prob": [round(float(np.max(mc_preds[m])), 4) for m in models],
        "Alert Rate (P >= 0.50)": [f"{np.mean(mc_preds[m] >= 0.50)*100:.1f}%" for m in models]
    })
    print(df_mc_summary.to_string(index=False))

    # 3. GAUSSIAN NOISE PERTURBATION
    print("\n" + "=" * 80)
    print(">> EXPERIMENT 3: Gaussian Noise Perturbation & Stability Audit")
    print("=" * 80)

    test_mask = df["date"] >= "2019-01-01"
    df_test_orig = df.loc[test_mask, FEATURES_30D].fillna(0).copy()

    noise_levels = [0.05, 0.10, 0.25, 0.50]
    stability_records = []

    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, size=df_test_orig.shape)
        df_perturbed = pd.DataFrame(
            np.maximum(0, df_test_orig.values * (1.0 + noise)),
            columns=FEATURES_30D
        )

        for m_name, model in models.items():
            base_probs = model.predict_proba(df_test_orig)[:, 1]
            pert_probs = model.predict_proba(df_perturbed)[:, 1]

            mae_shift = np.mean(np.abs(pert_probs - base_probs))
            max_shift = np.max(np.abs(pert_probs - base_probs))
            pred_flip_rate = np.mean((base_probs >= 0.5) != (pert_probs >= 0.5))

            stability_records.append({
                "Noise Level": f"+/-{int(sigma*100)}%",
                "Model": m_name,
                "Mean Prob Shift (MAE)": round(float(mae_shift), 4),
                "Max Prob Shift": round(float(max_shift), 4),
                "Alert Classification Flip Rate": f"{pred_flip_rate*100:.2f}%"
            })

    df_stability = pd.DataFrame(stability_records)
    print(df_stability.to_string(index=False))

    # 4. MARGINAL DOSE-RESPONSE SWEEPS
    print("\n" + "=" * 80)
    print(">> EXPERIMENT 4: Marginal Biological Dose-Response Sweeps")
    print("=" * 80)

    rain_range = np.linspace(0, 600, 13)
    rain_responses = []
    for r in rain_range:
        vec = baseline_median.copy()
        vec["pr_total_mm_lag_2m"] = r
        vec["runoff_risk_lag2m"] = vec["google_bldgs_pct_built_up_area"] * r
        input_data = pd.DataFrame([vec[FEATURES_30D]])
        res_pt = {"Rain_Lag2m_mm": round(r, 1)}
        for m_name, m in models.items():
            res_pt[m_name] = round(float(m.predict_proba(input_data)[0, 1]), 4)
        rain_responses.append(res_pt)
    df_rain_sweep = pd.DataFrame(rain_responses)

    heat_range = np.linspace(22, 44, 12)
    heat_responses = []
    for h in heat_range:
        vec = baseline_median.copy()
        vec["heat_index_mean_lag_2m"] = h
        input_data = pd.DataFrame([vec[FEATURES_30D]])
        res_pt = {"HeatIndex_Lag2m_C": round(h, 1)}
        for m_name, m in models.items():
            res_pt[m_name] = round(float(m.predict_proba(input_data)[0, 1]), 4)
        heat_responses.append(res_pt)
    df_heat_sweep = pd.DataFrame(heat_responses)

    df_sc_res.to_csv(processed_dir / "stress_test_scenarios.csv", index=False)
    df_mc_summary.to_csv(processed_dir / "stress_test_monte_carlo.csv", index=False)
    df_stability.to_csv(processed_dir / "stress_test_stability.csv", index=False)
    df_rain_sweep.to_csv(processed_dir / "dose_response_rainfall.csv", index=False)
    df_heat_sweep.to_csv(processed_dir / "dose_response_heat.csv", index=False)
    print(f"\n[SUCCESS] Exported all stress test tables to: {processed_dir}")

    return df_sc_res, df_mc_summary, df_stability, df_rain_sweep, df_heat_sweep

if __name__ == "__main__":
    run_stress_testing()
