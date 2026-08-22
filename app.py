"""
Project CCHAIN: Climate-Driven Vector-Borne Outbreak Surveillance Engine
Interactive Streamlit Dashboard & LGU Decision Support System
"""

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import dashboard_utils as utils

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Project CCHAIN | Outbreak Surveillance Engine",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .tier-1-badge {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .tier-2-badge {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .tier-3-badge {
        background-color: #dcfce7;
        color: #166534;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. DATA INGESTION & CACHING
# ------------------------------------------------------------------------------
with st.spinner("Initializing CCHAIN Surveillance Engine & Loading Spatio-Temporal Data..."):
    df_data = utils.load_surveillance_data()
    geojson_data, df_centroids = utils.load_geojson_and_centroids()
    df_health = utils.load_health_infrastructure()
    models_dict = utils.train_cached_models()
    df_benchmarks = utils.load_benchmarks()

# Merge health infrastructure and centroid coordinates into master dataframe slice
pcode_to_centroid = df_centroids.set_index("adm4_pcode").to_dict(orient="index")
pcode_to_health = df_health.set_index("adm4_pcode").to_dict(orient="index")


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & FILTERS
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/mosquito.png", width=64)
st.sidebar.title("Surveillance Controls")

horizon_option = st.sidebar.radio(
    "🎯 Early Warning Horizon",
    options=["30-Day Lead (T+1)", "60-Day Lead (T+2)"],
    index=0,
    help="Select the prediction horizon. 30-Day uses climate up to T-1; 60-Day uses climate up to T-2."
)
horizon_key = "30d" if "30-Day" in horizon_option else "60d"

model_option = st.sidebar.selectbox(
    "🤖 Classifier Model",
    options=["Logistic Regression", "LightGBM", "XGBoost", "Random Forest"],
    index=0,
    help="Select the machine learning algorithm calibrated for F2-Score."
)

current_model_info = models_dict[horizon_key][model_option]
active_model = current_model_info["model"]
active_features = current_model_info["features"]
active_thresh = current_model_info["opt_thresh"]

st.sidebar.markdown(f"**Calibrated $F_2$ Threshold**: `{active_thresh:.2f}`")

all_dates = sorted(df_data["date"].unique())
date_strings = [pd.to_datetime(d).strftime("%Y-%m") for d in all_dates]

selected_date_str = st.sidebar.select_slider(
    "📅 Surveillance Month",
    options=date_strings,
    value=date_strings[-12] if len(date_strings) >= 12 else date_strings[0],
    help="Slide across historical months to inspect retrospective surveillance predictions."
)
selected_date = pd.to_datetime(selected_date_str)

st.sidebar.markdown("---")
st.sidebar.subheader("Layer Overlays")
show_facilities = st.sidebar.checkbox("🏥 Show Health Facilities", value=True)
show_coastal = st.sidebar.checkbox("🌊 Highlight Coastal Barangays", value=False)


# ------------------------------------------------------------------------------
# 4. SLICE & INFERENCE PREPARATION
# ------------------------------------------------------------------------------
df_month = df_data[df_data["date"] == selected_date].copy().reset_index(drop=True)

# Add centroids and health details
df_month["lat"] = df_month["adm4_pcode"].map(lambda x: pcode_to_centroid.get(x, {}).get("lat", 8.455))
df_month["lon"] = df_month["adm4_pcode"].map(lambda x: pcode_to_centroid.get(x, {}).get("lon", 124.637))
df_month["doh_hospitals"] = df_month["adm4_pcode"].map(lambda x: pcode_to_health.get(x, {}).get("doh_hospitals", 0))
df_month["doh_brgy_health_stations"] = df_month["adm4_pcode"].map(lambda x: pcode_to_health.get(x, {}).get("doh_brgy_health_stations", 0))
df_month["doh_nearest_hospital_km"] = df_month["adm4_pcode"].map(lambda x: pcode_to_health.get(x, {}).get("doh_nearest_hospital_km", 0.0))

# Inference
X_slice = df_month[active_features].fillna(0)
if hasattr(active_model, "predict_proba"):
    slice_probs = active_model.predict_proba(X_slice)[:, 1]
else:
    slice_probs = active_model.predict(X_slice)

df_month["predicted_prob"] = slice_probs
df_month["risk_pct"] = (slice_probs * 100).round(1)
df_month["predicted_outbreak"] = (slice_probs >= active_thresh).astype(int)

df_triage = utils.compute_lgu_triage_matrix(df_month, slice_probs, active_thresh)


# ------------------------------------------------------------------------------
# 5. HEADER SECTION
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🦟 Project CCHAIN: Outbreak Surveillance Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Climate-Driven Vector-Borne Early Warning & LGU Decision Support System | '
    '<b>Pilot City: Cagayan de Oro (80 Barangays)</b></div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Geospatial Early Warning Map",
    "📈 Barangay Deep Dive & Diagnostics",
    "🧪 'What-If' Climate Simulator",
    "🚨 LGU Prescriptive Action Matrix",
    "🏆 Model Tournament & Diagnostics"
])


# ------------------------------------------------------------------------------
# TAB 1: GEOSPATIAL EARLY WARNING MAP
# ------------------------------------------------------------------------------
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    total_cases = int(df_month["dengue_cases"].sum())
    outbreak_count = int(df_month["predicted_outbreak"].sum())
    actual_outbreak_count = int(df_month["is_outbreak"].sum())
    mean_risk = float(df_month["predicted_prob"].mean() * 100)
    top_brgy = df_month.loc[df_month["predicted_prob"].idxmax()]["adm4_en"]
    top_risk = float(df_month["predicted_prob"].max() * 100)

    col1.metric("Selected Month", selected_date_str)
    col2.metric("Recorded CDO Cases", f"{total_cases:,}")
    col3.metric("Predicted Outbreak Alerts", f"{outbreak_count} / 80", delta=f"Actual: {actual_outbreak_count}")
    col4.metric("Average City Risk", f"{mean_risk:.1f}%")
    col5.metric("Highest Risk Barangay", f"{top_brgy} ({top_risk:.1f}%)")

    st.markdown("---")

    # Interactive Plotly Mapbox Choropleth
    fig_map = px.choropleth_mapbox(
        df_month,
        geojson=geojson_data,
        locations="adm4_pcode",
        color="predicted_prob",
        color_continuous_scale="RdYlGn_r",
        range_color=(0.0, 1.0),
        mapbox_style="carto-positron",
        zoom=10.5,
        center={"lat": 8.4558, "lon": 124.6371},
        opacity=0.75,
        hover_name="adm4_en",
        hover_data={
            "adm4_pcode": False,
            "predicted_prob": ":.2f",
            "risk_pct": ":.1f%",
            "dengue_cases": True,
            "pop_count_total": ":,.0f",
            "brgy_is_coastal": True,
            "doh_nearest_hospital_km": ":.2f km"
        },
        labels={
            "predicted_prob": "Outbreak Probability",
            "dengue_cases": "Recorded Cases",
            "pop_count_total": "Population",
            "brgy_is_coastal": "Coastal",
            "doh_nearest_hospital_km": "Hospital Dist"
        }
    )

    # Optional health facility overlay
    if show_facilities:
        df_with_hosp = df_month[df_month["doh_hospitals"] > 0]
        if not df_with_hosp.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=df_with_hosp["lat"],
                lon=df_with_hosp["lon"],
                mode="markers+text",
                marker=go.scattermapbox.Marker(size=12, color="#7c3aed", symbol="hospital"),
                text=df_with_hosp["adm4_en"],
                textposition="top right",
                name="DOH Hospitals / Main Centers",
                hoverinfo="text"
            ))

    fig_map.update_layout(
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        height=620,
        coloraxis_colorbar=dict(
            title="Outbreak Risk",
            ticksuffix="",
            len=0.75,
            x=0.01,
            y=0.5
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Outbreak Alerts Summary Table
    df_alerts = df_month[df_month["predicted_outbreak"] == 1][
        ["adm4_en", "risk_pct", "dengue_cases", "pop_count_total", "pop_density_imputed", "doh_nearest_hospital_km"]
    ].sort_values(by="risk_pct", ascending=False)

    if not df_alerts.empty:
        st.subheader(f"🚨 Outbreak Alert Barangays for {selected_date_str} ({len(df_alerts)} Identified)")
        st.dataframe(
            df_alerts.rename(columns={
                "adm4_en": "Barangay",
                "risk_pct": "Predicted Risk (%)",
                "dengue_cases": "Monthly Cases",
                "pop_count_total": "Population",
                "pop_density_imputed": "Host Density (pop/km²)",
                "doh_nearest_hospital_km": "Nearest Hospital (km)"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success(f"No barangays exceed the outbreak alert threshold ({active_thresh:.2f}) for {selected_date_str}.")


# ------------------------------------------------------------------------------
# TAB 2: BARANGAY DEEP DIVE & DIAGNOSTICS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🔍 Barangay-Specific Surveillance & Climate Trajectory")

    brgy_list = sorted(df_data["adm4_en"].unique())
    selected_brgy_name = st.selectbox("Select Barangay to Inspect", options=brgy_list, index=brgy_list.index("Carmen") if "Carmen" in brgy_list else 0)

    df_b = df_data[df_data["adm4_en"] == selected_brgy_name].sort_values(by="date").copy()
    b_info = df_b.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Land Area", f"{b_info['brgy_total_area']:.2f} km²")
    c2.metric("Building Footprints", f"{int(b_info['google_bldgs_count']):,}")
    c3.metric("Built-Up Area", f"{b_info['google_bldgs_pct_built_up_area']*100:.1f}%")
    c4.metric("Total 20-Year Cases", f"{int(df_b['dengue_cases'].sum()):,}")

    st.markdown("---")

    # Dual-Axis Time Series: Cases vs Lagged Rainfall & Heat Index
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    fig_ts.add_trace(
        go.Bar(
            x=df_b["date"],
            y=df_b["dengue_cases"],
            name="Dengue Cases (Actual)",
            marker_color="#ef4444",
            opacity=0.65
        ),
        secondary_y=False
    )

    fig_ts.add_trace(
        go.Scatter(
            x=df_b["date"],
            y=df_b["pr_total_mm_lag_1m"],
            name="Precipitation Lag 1m (mm)",
            line=dict(color="#3b82f6", width=2)
        ),
        secondary_y=True
    )

    fig_ts.add_trace(
        go.Scatter(
            x=df_b["date"],
            y=df_b["heat_index_mean_lag_2m"],
            name="Heat Index Lag 2m (°C)",
            line=dict(color="#f97316", width=2, dash="dot")
        ),
        secondary_y=True
    )

    fig_ts.update_layout(
        title=f"Epidemiological-Meteorological Time Series for {selected_brgy_name} (2003–2022)",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin={"l": 40, "r": 40, "t": 60, "b": 40}
    )
    fig_ts.update_yaxes(title_text="Monthly Dengue Cases", secondary_y=False)
    fig_ts.update_yaxes(title_text="Lagged Climate Indices", secondary_y=True)

    st.plotly_chart(fig_ts, use_container_width=True)

    # Local Feature Importances
    st.subheader(f"Key Model Predictors ({model_option} - {horizon_option})")
    if hasattr(active_model, "feature_importances_"):
        imps = pd.Series(active_model.feature_importances_, index=active_features).sort_values(ascending=True).tail(10)
        fig_imp = px.bar(
            x=imps.values,
            y=imps.index,
            orientation="h",
            labels={"x": "Relative Importance", "y": "Feature"},
            title=f"Top 10 Global Features for {model_option}",
            color=imps.values,
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    elif hasattr(active_model, "named_steps") and "clf" in active_model.named_steps:
        coefs = pd.Series(active_model.named_steps["clf"].coef_[0], index=active_features).sort_values(ascending=True).tail(10)
        fig_imp = px.bar(
            x=coefs.values,
            y=coefs.index,
            orientation="h",
            labels={"x": "L2 Logistic Coefficient (Log Odds)", "y": "Feature"},
            title=f"Top 10 Coefficients for {model_option}",
            color=coefs.values,
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: CLIMATE "WHAT-IF" SCENARIO SIMULATOR
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🧪 Climate Anomaly Scenario Studio")
    st.markdown(
        "Simulate how projected extreme weather shifts (e.g. intensified monsoon rainfall or heat waves) "
        "propagate through urban built-environment exposure and spatial adjacency to alter outbreak risk across all 80 barangays."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        pr_delta_pct = st.slider("🌧️ Precipitation Anomaly (%)", min_value=-50, max_value=80, value=25, step=5)
    with s_col2:
        temp_delta_c = st.slider("🌡️ Thermal / Heat Index Increase (°C)", min_value=-2.0, max_value=4.0, value=1.5, step=0.5)

    # Perform Simulation
    df_sim = df_month.copy()
    pr_factor = 1.0 + (pr_delta_pct / 100.0)

    for col in df_sim.columns:
        if "pr_total_mm" in col or "pr_rolling" in col:
            df_sim[col] = df_sim[col] * pr_factor
        if "heat_index" in col or "tave" in col:
            df_sim[col] = df_sim[col] + temp_delta_c
        if "runoff_risk" in col:
            df_sim[col] = df_sim[col] * pr_factor
        if "urban_heat_trap" in col:
            df_sim[col] = df_sim[col] + (df_sim["google_bldgs_density"] * temp_delta_c)

    X_sim = df_sim[active_features].fillna(0)
    if hasattr(active_model, "predict_proba"):
        sim_probs = active_model.predict_proba(X_sim)[:, 1]
    else:
        sim_probs = active_model.predict(X_sim)

    df_sim["simulated_prob"] = sim_probs
    df_sim["sim_risk_pct"] = (sim_probs * 100).round(1)
    df_sim["risk_diff_pct"] = (df_sim["sim_risk_pct"] - df_month["risk_pct"]).round(1)
    df_sim["sim_outbreak"] = (sim_probs >= active_thresh).astype(int)

    base_outbreaks = int(df_month["predicted_outbreak"].sum())
    sim_outbreaks = int(df_sim["sim_outbreak"].sum())
    net_outbreak_shift = sim_outbreaks - base_outbreaks
    avg_risk_shift = float(df_sim["sim_risk_pct"].mean() - df_month["risk_pct"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Outbreaks", f"{base_outbreaks} / 80")
    k2.metric("Simulated Outbreaks", f"{sim_outbreaks} / 80", delta=f"{net_outbreak_shift:+d} Barangays")
    k3.metric("Baseline Avg Risk", f"{df_month['risk_pct'].mean():.1f}%")
    k4.metric("Simulated Avg Risk", f"{df_sim['sim_risk_pct'].mean():.1f}%", delta=f"{avg_risk_shift:+.1f}%")

    st.markdown("---")

    # Map of Risk Shift (Difference)
    fig_sim_map = px.choropleth_mapbox(
        df_sim,
        geojson=geojson_data,
        locations="adm4_pcode",
        color="risk_diff_pct",
        color_continuous_scale="Turbo",
        mapbox_style="carto-positron",
        zoom=10.5,
        center={"lat": 8.4558, "lon": 124.6371},
        opacity=0.75,
        hover_name="adm4_en",
        hover_data={
            "adm4_pcode": False,
            "risk_diff_pct": ":+.1f%",
            "sim_risk_pct": ":.1f%",
            "risk_pct": ":.1f%"
        },
        labels={
            "risk_diff_pct": "Risk Delta (%)",
            "sim_risk_pct": "Simulated Risk",
            "risk_pct": "Baseline Risk"
        },
        title=f"Spatial Shift in Outbreak Risk (Precip: {pr_delta_pct:+d}%, Temp: {temp_delta_c:+.1f}°C)"
    )

    fig_sim_map.update_layout(margin={"r": 0, "t": 35, "l": 0, "b": 0}, height=550)
    st.plotly_chart(fig_sim_map, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: LGU PRESCRIPTIVE ACTION MATRIX
# ------------------------------------------------------------------------------
with tab4:
    st.subheader(f"🚨 LGU Resource Allocation & Action Matrix ({selected_date_str})")
    st.markdown(
        "Automated operational triage ranking each barangay into targeted public health action tiers "
        "for the City Health Office (CHO), vector control units, and hospital administrators."
    )

    tier1_count = len(df_triage[df_triage["Action_Tier"].str.contains("Tier 1")])
    tier2_count = len(df_triage[df_triage["Action_Tier"].str.contains("Tier 2")])
    tier3_count = len(df_triage[df_triage["Action_Tier"].str.contains("Tier 3")])

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Tier 1: Immediate Vector Control", f"{tier1_count} Barangays")
    m2.metric("🟡 Tier 2: Enhanced Clinical Surveillance", f"{tier2_count} Barangays")
    m3.metric("🟢 Tier 3: Routine Monitoring", f"{tier3_count} Barangays")

    st.markdown("---")

    display_triage = df_triage[[
        "Priority_Rank", "adm4_en", "Action_Tier", "risk_pct", "pop_count_total",
        "doh_nearest_hospital_km", "Recommended_Intervention"
    ]].copy()

    display_triage.columns = [
        "Rank", "Barangay", "Action Tier", "Predicted Risk (%)", "Population",
        "Nearest Hospital (km)", "Recommended Action"
    ]

    st.dataframe(
        display_triage,
        use_container_width=True,
        hide_index=True
    )

    csv_export = display_triage.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download LGU Action Plan CSV",
        data=csv_export,
        file_name=f"cchain_cdo_lgu_action_plan_{selected_date_str}.csv",
        mime="text/csv"
    )


# ------------------------------------------------------------------------------
# TAB 5: MODEL TOURNAMENT & DIAGNOSTICS
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("🏆 Multi-Model Tournament & Benchmark Diagnostics")
    st.markdown(
        "Rigorous cross-comparison across all 4 machine learning algorithms on strict temporal holdout data (2019–2022)."
    )

    if not df_benchmarks.empty:
        st.dataframe(df_benchmarks, use_container_width=True, hide_index=True)

        fig_bench = px.bar(
            df_benchmarks,
            x="Model",
            y=["ROC-AUC", "PR-AUC", "F2-Score (Opt)"],
            barmode="group",
            facet_col="Horizon",
            title="Comparison of Primary Discrimination & Sensitivity Metrics",
            height=450
        )
        st.plotly_chart(fig_bench, use_container_width=True)
    else:
        st.info("Benchmark table available in data/processed/cchain_model_benchmarks.csv.")

    st.markdown(r"""
    ### 🔬 Spatio-Temporal Methodology Highlights:
    * **Zero Future Climate Leakage**: Predictions at month $T$ strictly leverage meteorological lags $\le T-1$ (for 30-Day Lead) and $\le T-2$ (for 60-Day Lead).
    * **Spatial Autoregressive Contiguity ($W \times Y$)**: 80x80 row-standardized adjacency matrix capturing pathogen contagion across contiguous borders.
    * **$F_2$-Score Threshold Optimization**: Prioritizes **Recall (Sensitivity $\ge 91.3\%$)** over precision to minimize false negatives in public health early warnings.
    """)
