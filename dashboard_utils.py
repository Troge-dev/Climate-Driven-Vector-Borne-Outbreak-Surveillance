"""
Project CCHAIN: Climate-Driven Vector-Borne Outbreak Surveillance Engine
Dashboard Helper Utilities & Model Inference Cache
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import shapely.wkt
import shapely.geometry
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score
import lightgbm as lgb
import xgboost as xgb

BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path(".")
RAW_DATA_DIR = BASE_DIR / "data" / "cchain_raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

PILOT_CITY_CODE = "PH104305000"

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


@st.cache_data
def load_surveillance_data():
    """Loads the pre-engineered analysis-ready surveillance dataset."""
    csv_path = PROCESSED_DATA_DIR / "cchain_cdo_dengue_surveillance_ready.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["adm4_pcode", "date"]).reset_index(drop=True)
    return df


@st.cache_data
def load_geojson_and_centroids():
    """Builds GeoJSON polygons and centroid coordinates for all 80 barangays."""
    df_geo = pd.read_csv(RAW_DATA_DIR / "brgy_geography.csv")
    df_loc = pd.read_csv(RAW_DATA_DIR / "location.csv")

    cdo_loc = df_loc[df_loc["adm3_pcode"] == PILOT_CITY_CODE][
        ["adm4_pcode", "adm4_en"]
    ].drop_duplicates(subset=["adm4_pcode"])

    cdo_geo = df_geo[df_geo["adm4_pcode"].isin(cdo_loc["adm4_pcode"])].drop_duplicates(subset=["adm4_pcode"]).merge(
        cdo_loc, on="adm4_pcode"
    )

    cdo_geo["poly"] = cdo_geo["geometry"].apply(shapely.wkt.loads)
    cdo_geo["lat"] = cdo_geo["poly"].apply(lambda p: p.centroid.y)
    cdo_geo["lon"] = cdo_geo["poly"].apply(lambda p: p.centroid.x)

    geojson_features = []
    for _, row in cdo_geo.iterrows():
        geojson_features.append({
            "type": "Feature",
            "id": row["adm4_pcode"],
            "properties": {
                "adm4_pcode": row["adm4_pcode"],
                "adm4_en": row["adm4_en"],
                "brgy_total_area": row["brgy_total_area"],
                "brgy_is_coastal": int(row.get("brgy_is_coastal", 0))
            },
            "geometry": shapely.geometry.mapping(row["poly"])
        })

    geojson_dict = {"type": "FeatureCollection", "features": geojson_features}
    df_centroids = cdo_geo[["adm4_pcode", "adm4_en", "lat", "lon", "brgy_total_area", "brgy_is_coastal"]].copy()
    return geojson_dict, df_centroids


@st.cache_data
def load_health_infrastructure():
    """Loads DOH and OSM health facility indicators per barangay."""
    df_doh = pd.read_csv(RAW_DATA_DIR / "geoportal_doh_poi_health.csv")
    df_osm = pd.read_csv(RAW_DATA_DIR / "osm_poi_health.csv")

    doh_agg = df_doh.groupby("adm4_pcode", as_index=False).agg(
        doh_pois_count=("doh_pois_count", "last"),
        doh_brgy_health_stations=("doh_brgy_health_station_count", "last"),
        doh_hospitals=("doh_hospital_count", "last"),
        doh_nearest_hospital_km=("doh_hospital_nearest", lambda x: round(x.iloc[-1] / 1000, 2) if pd.notnull(x.iloc[-1]) else 0)
    )

    osm_agg = df_osm.groupby("adm4_pcode", as_index=False).agg(
        osm_clinics=("clinic_count", "last"),
        osm_pharmacies=("pharmacy_count", "last")
    )

    df_health = doh_agg.merge(osm_agg, on="adm4_pcode", how="left").fillna(0)
    return df_health


@st.cache_data
def load_benchmarks():
    """Loads model benchmark results table."""
    csv_path = PROCESSED_DATA_DIR / "cchain_model_benchmarks.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def optimize_fbeta_threshold(y_true, y_probs, beta=2.0):
    best_th = 0.5
    best_f = 0.0
    for th in np.linspace(0.05, 0.95, 91):
        preds = (y_probs >= th).astype(int)
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if score > best_f:
            best_f = score
            best_th = th
    return best_th, best_f


@st.cache_resource
def train_cached_models():
    """Trains and caches the tournament models across 30-day and 60-day horizons."""
    df = load_surveillance_data()
    train_mask = df["date"] < "2019-01-01"
    y_train = df.loc[train_mask, "is_outbreak"].values
    pos_weight = float((len(y_train) - sum(y_train)) / sum(y_train))

    models_dict = {}

    for horizon_key, features in [("30d", FEATURES_30D_HORIZON), ("60d", FEATURES_60D_HORIZON)]:
        X_train = df.loc[train_mask, features].fillna(0)

        clf_pool = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
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

        horizon_trained = {}
        for m_name, model in clf_pool.items():
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                train_probs = model.predict_proba(X_train)[:, 1]
            else:
                train_probs = model.predict(X_train)

            opt_th, best_f2 = optimize_fbeta_threshold(y_train, train_probs, beta=2.0)
            horizon_trained[m_name] = {
                "model": model,
                "opt_thresh": round(opt_th, 2),
                "features": features
            }
        models_dict[horizon_key] = horizon_trained

    return models_dict


def compute_lgu_triage_matrix(df_slice, probs, opt_thresh):
    """
    Computes priority ranking and prescriptive intervention tiers.
    Tier 1 (Red, >= 70% or >= opt_thresh + 0.1): Urgent Vector Elimination (48h larviciding, ULV fogging)
    Tier 2 (Yellow, 40% - 69%): Active Case Finding & NS1 Rapid Diagnostics
    Tier 3 (Green, < 40%): Routine Monitoring & Community Clean-up
    """
    df_res = df_slice.copy()
    df_res["predicted_outbreak_prob"] = probs
    df_res["risk_pct"] = (probs * 100).round(1)

    t1_bound = max(0.70, opt_thresh)
    t2_bound = min(0.40, opt_thresh)

    conditions = [
        (df_res["predicted_outbreak_prob"] >= t1_bound),
        (df_res["predicted_outbreak_prob"] >= t2_bound)
    ]
    tiers = ["Tier 1: Immediate Vector Control", "Tier 2: Enhanced Clinical Surveillance"]
    actions = [
        "Targeted Larviciding + ULV Space Spraying within 48h; Hospital triage bed reserve.",
        "Pre-position NS1 diagnostic kits in Barangay Health Station; Daily fever monitoring.",
        "Routine IEC campaigns; Standard source-reduction cleanups."
    ]

    df_res["Action_Tier"] = np.select(conditions, tiers, default="Tier 3: Routine Monitoring")
    df_res["Recommended_Intervention"] = np.select(conditions, actions[:2], default=actions[2])

    df_ranked = df_res.sort_values(by="predicted_outbreak_prob", ascending=False).reset_index(drop=True)
    df_ranked["Priority_Rank"] = df_ranked.index + 1

    return df_ranked
