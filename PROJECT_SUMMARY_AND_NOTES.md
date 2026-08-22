# Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine
### **Complete Project Summary, Dataset Mappings, Model Results & Defense Guide**
*Laboratory Activity 1: Types of Data Analytics (DMA Course)*

---

## 📌 1. Fast Topic Claiming Formats (For `DMA_GROUP ASSIGNMENTS` Shared Sheet)

Use any of the options below to claim your topic on the class shared sheet:

### 🌟 Primary Copy-Paste Option (Recommended):
```text
climate_atmosphere.csv + disease_lgu_disaggregated_totals.csv + google_open_buildings.csv + worldpop_population.csv + location.csv. Can multi-month climate lags (rainfall, temperature, heat index) combined with urban building density predict localized dengue outbreak risks 30 to 60 days in advance to prescriptively allocate LGU vector-control and medical resources?
```

### 🔹 Alternative 1 (Machine Learning & Early Warning Focus):
```text
climate_atmosphere.csv + disease_lgu_disaggregated_totals.csv + google_open_buildings.csv + worldpop_population.csv. How can 1-to-3 month biological weather lags and barangay built-environment density forecast dengue epidemic thresholds per barangay for early-warning LGU intervention?
```

---

## 🎯 2. Core Operational Problem Statement

> Local Government Units (LGUs) and municipal health offices consistently suffer from reactive, delayed interventions during climate-induced dengue outbreaks following monsoonal precipitation and urban heat spikes. The absence of an automated spatial surveillance framework that bridges atmospheric lag dynamics with barangay-level structural vulnerabilities results in overwhelmed emergency facilities, stockouts of diagnostic and larvicidal supplies, and avoidable patient mortality. By modeling the 2-to-8-week non-linear lag relationship between climate anomalies and disease incidence, municipal health officers can transition to proactive vector-control deployment and dynamic clinical capacity reallocation prior to peak epidemic infection windows.

---

## 📂 3. Real Project CCHAIN Dataset Architecture

Downloaded directly from Kaggle (`thinkdatasci/project-cchain`), created by **Thinking Machines, EpiMetrics, Manila Observatory, and PACSII** (funded by Wellcome Trust & Lacuna Fund):

| Raw Kaggle CSV Table | Ingested Scale | Key Extracted Columns | Role in Pipeline |
| :--- | :---: | :--- | :--- |
| [`location.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/cchain_raw/location.csv) | **80 Barangays** | `adm3_pcode`, `adm4_pcode`, `adm4_en`, `brgy_total_area` | Spatial Master Reference (`PH104305000`: CDO) |
| [`disease_lgu_disaggregated_totals.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/cchain_raw/disease_lgu_disaggregated_totals.csv) | **2,490 Records** | `case_total`, `death_total`, `date`, `disease_common_name` | Target Variable ($Y$) |
| [`climate_atmosphere.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/cchain_raw/climate_atmosphere.csv) | **584,400 Records** | `pr` (rainfall), `tave`, `heat_index`, `rh`, `wind_speed` | Meteorological Predictors ($X$) & Lags |
| [`google_open_buildings.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/cchain_raw/google_open_buildings.csv) | **80 Barangays** | `google_bldgs_density`, `google_bldgs_pct_built_up_area` | Built-Environment Susceptibility ($Z$) |
| [`worldpop_population.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/cchain_raw/worldpop_population.csv) | **1,680 Records** | `pop_density_mean`, `pop_count_total` | Human Host Exposure Density ($Z$) |

---

## 🔬 4. Formulation of the Four Stages of Data Analytics

### 📊 A. Descriptive Analytics *(What Happened in the Historical Record?)*
* **Q1:** What are the historical monthly case trajectories and seasonal baseline cycles of dengue across Cagayan de Oro's 80 barangays over the 20-year span (2003–2022)?
* **Q2:** Which geographic barangay clusters consistently exhibit the highest endemic incidence rates and mortality burdens?

### 🔍 B. Diagnostic Analytics *(Why Did Outbreaks & Spikes Occur?)*
* **Q3:** What is the exact biological lag duration (1-month vs. 2-month vs. 3-month delay) between peak precipitation/heat index anomalies and subsequent dengue caseload surges?
* **Q4:** How do structural built-environment indicators (building footprint density, impervious surface ratio) explain the divergence in outbreak severity between adjacent barangays?

### 📈 C. Predictive Analytics *(What Outbreak Events Will Happen in the Future?)*
* **Q5:** Given forecasted 30-to-60 day ERA5 climate anomalies, what is the expected caseload of dengue per barangay?
* **Q6:** Can a machine learning classifier reliably predict whether a barangay will breach its historical epidemic outbreak threshold ($\ge 75\text{th}$ percentile) 1 to 2 months in advance?

### 💡 D. Prescriptive Analytics *(What Precise Actions Should the Health Office Take?)*
* **Q7:** What is the optimal spatial allocation schedule of vector-control interventions (targeted fogging, larvicide distribution) across priority zones to maximize outbreak containment?
* **Q8:** How should clinical staff and inpatient bed capacity across district health centers be dynamically scheduled 3 weeks ahead of a climate-predicted surge?

---

## 🛠️ 5. Methodology & Technical Stack Matrix

| Question Scope | Analytics Type | Data Mining / Mathematical Techniques | Python Stack |
| :--- | :--- | :--- | :--- |
| **Q1 & Q2** | **Descriptive** | Time-Series Trend Decomposition (STL), Spatial Heatmapping | `pandas`, `geopandas`, `matplotlib`, `folium` |
| **Q3 & Q4** | **Diagnostic** | Distributed Lag Non-linear Models (DLNM), Spearman Cross-Correlation | `scipy.stats`, `statsmodels`, `scikit-learn` |
| **Q5 & Q6** | **Predictive** | Supervised ML (Random Forest Classifier), Out-of-Time Cross Validation | `scikit-learn`, `xgboost`, `prophet` |
| **Q7 & Q8** | **Prescriptive** | Linear Programming (Constrained Optimization), TOPSIS Decision Matrix | `scipy.optimize`, `pulp` |

---

## 📊 6. Empirical Model Performance & Key Findings

Processed Matrix: **18,960 Space-Time Rows $\times$ 37 Features** ([`cchain_cdo_dengue_surveillance_ready.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/processed/cchain_cdo_dengue_surveillance_ready.csv))

### A. Performance Metrics (Evaluated on Unseen 2019–2022 Test Data):
* **ROC-AUC Score:** **`0.903`** *(Outstanding discrimination between normal vs. epidemic months)*
* **Overall Accuracy:** **`87.0%`** *(3,340 out of 3,840 test months predicted correctly)*
* **Outbreak Recall:** **`72.0%`** *(Successfully catches 7 out of 10 outbreaks 30 to 60 days before hospital spikes)*

### B. Top 5 Most Predictive Climate-Health Indicators:
1. **`pop_density_mean` (35.05% Importance):** High human host concentration enables rapid vector-to-host virus transmission.
2. **`google_bldgs_pct_built_up_area` (14.92% Importance):** Impervious urban surfaces create artificial container breeding sites.
3. **`heat_index_mean_lag_3m` (9.39% Importance):** 3-month heat stress accelerates Extrinsic Incubation Period (EIP) inside mosquitoes.
4. **`heat_index_mean_lag_2m` (6.93% Importance):** 2-month ambient heat elevation speeds up larval hatching.
5. **`tave_mean_lag_1m` (5.72% Importance):** 1-month mean temperature triggers increased mosquito biting frequency.

---

## 🏥 7. Prescriptive LGU Action Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Probability (P)  │ Alert Level     │ Automated Prescriptive Response        │
├──────────────────┼─────────────────┼────────────────────────────────────────┤
│ P < 0.30         │ Level 1: Normal │ Standard community cleanup & monitoring│
│ 0.30 <= P < 0.65 │ Level 2: Alert  │ Pre-emptive larviciding in high-density│
│                  │                 │ barangays; dispatch health workers.    │
│ P >= 0.65        │ Level 3: Outbreak│ Targeted spatial fogging & reserve     │
│                  │ Warning         │ emergency hospital triage beds.        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 8. Class Defense Strategy & Talking Points

When presenting to your professor:
1. **Highlight the Lags:** Emphasize that mosquito breeding and viral incubation take 2–3 months. Heat today creates outbreaks 60–90 days later.
2. **Highlight Spatial Fusion:** Explain that weather predicts *when* an outbreak happens, while Google Open Buildings and WorldPop predict *where* (which specific barangay).
3. **Differentiate from Basic Correlation:** Emphasize that your project goes beyond backward-looking diagnostic correlation into forward-looking ML prediction and prescriptive resource allocation.

---

## 📁 9. Deliverables Directory Index

* 📓 **Interactive Jupyter Notebook:** [`Project_CCHAIN_Surveillance_Pipeline.ipynb`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/Project_CCHAIN_Surveillance_Pipeline.ipynb)
* 🐍 **Python Pipeline Script:** [`cchain_pipeline.py`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/cchain_pipeline.py)
* 📊 **Analysis-Ready CSV Dataset:** [`cchain_cdo_dengue_surveillance_ready.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/processed/cchain_cdo_dengue_surveillance_ready.csv)
