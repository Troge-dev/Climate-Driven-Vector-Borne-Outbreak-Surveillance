# 🦟 Cagayan de Oro City Dengue Early Warning & Outbreak Prevention System
> **A Spatial-Temporal Climate Surveillance & Machine Learning Decision-Support Engine for the Cagayan de Oro City Local Government Unit (CDO LGU)**  
> *Developed for the CDO City Health Office (CHO) & City Disaster Risk Reduction and Management Office (CDRRMO)*  
> *Pilot City: Cagayan de Oro City, Northern Mindanao, Philippines (80 Barangays | 2003–2022 Longitudinal Dataset)*  
>
> 🎓 **Academic Fulfillment:** **Data Mining Course — Laboratory Activity 1**  
> 🌐 **Primary Dataset Source:** **Project CCHAIN** (*Climate Change and Health Analytics Network* | Kaggle: [`thinkdatasci/project-cchain`](https://www.kaggle.com/datasets/thinkdatasci/project-cchain))

---

> [!IMPORTANT]
> **Course & Data Attribution Notice:**
> * **Academic Requirement:** This repository and comprehensive surveillance pipeline are submitted in fulfillment of **Data Mining Course Laboratory Activity 1**.
> * **Data Provenance:** All primary epidemiological, meteorological, morphological, and demographic datasets utilized in this system originate from **Project CCHAIN** (*Climate Change and Health Analytics Network*), an open-access multi-partner initiative by Thinking Machines, EpiMetrics, Manila Observatory, and PACSII (funded by the Wellcome Trust & Lacuna Fund). The 35 raw CCHAIN multi-source tabular tables were localized, filtered, spatialized, and engineered for the 80 barangays of Cagayan de Oro City.

## 🏛️ 1. Project Overview & CDO LGU Operational Problem Statement

In Cagayan de Oro City, dengue fever represents a severe annual municipal health emergency. The **CDO City Health Office (CHO)** and the **City Disaster Risk Reduction and Management Office (CDRRMO)** historically face significant operational constraints due to **delayed, reactive responses**:
* Traditional vector control (e.g., thermal fogging and indoor residual spraying) and community larviciding are often initiated only *after* hospital triage wards at **Northern Mindanao Medical Center (NMMC)** and **J.R. Borja General Hospital (JRBGH)** are overwhelmed with severe pediatric dengue cases.
* Standard clinical reporting lags by 2 to 4 weeks, leaving barangay health centers unable to deploy preventative measures in time.

### The CDO LGU Surveillance Solution:
This project establishes a **localized, predictive, and prescriptive early-warning surveillance engine** engineered specifically for the 80 barangays of Cagayan de Oro City. By integrating 20 years of downscaled **ECMWF ERA5-Land climate reanalysis**, **DOH Region X / CDO CHO epidemiological records**, **Google Open Buildings urban footprints**, and **WorldPop high-resolution exposure grids**, the system models the **non-linear 30-to-90 day biological lag relationship** between monsoon rainfall, ambient thermal heat index, urban slum density, and *Aedes aegypti* mosquito proliferation.

This enables the CDO Mayor's Office and City Health Officers to execute targeted vector control and preposition clinical supplies **30 to 60 days in advance**.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                   CDO LGU 4-STAGE DATA ANALYTICS & SURVEILLANCE PIPELINE                    │
├───────────────────────────┬─────────────────────────────┬───────────────────────────────────┤
│ Stage                     │ Analytics Type              │ Key Deliverable for CDO LGU       │
├───────────────────────────┼─────────────────────────────┼───────────────────────────────────┤
│ 1. CDO Baseline & GIS     │ Descriptive Analytics       │ 20-Year Baseline & 80-Brgy W Map  │
│ 2. Biological Lag Engine  │ Diagnostic Analytics        │ Distributed Lags & Physical Terms │
│ 3. 30D & 60D Predictors   │ Predictive Analytics        │ Multi-Model Outbreak Probability  │
│ 4. Municipal Dispatch     │ Prescriptive Analytics      │ Automated 3-Tier LGU Action Matrix│
└───────────────────────────┴─────────────────────────────┴───────────────────────────────────┘
```

---

## 🗺️ 2. Geographic Scope: Cagayan de Oro City's 80 Barangays

The system models micro-climate risk profiles across all **80 distinct barangays of Cagayan de Oro City** (`PH104305000`):
* **High-Density Urban Lowlands & Flood Basins:** Carmen, Lapasan, Kauswagan, Balulang, Macasandig, Puntod, Bulua, Nazareth, Camaman-an, Gusa.
* **Coastal & Port Zones:** Macabalan, Bonbon, Bayabas, Puerto, Tablon, Agusan.
* **Rapidly Expanding Upland & Peri-Urban Corridors:** Upper Puerto, Lumbia, Canitoan, Iponan, Pagatpat.
* **High-Altitude Forested Enclaves:** Dansolihon, Besigan, Tumpagon, Pigsag-an, Tignapoloan.

The boundary polygon geometries (`brgy_geography.csv`) are parsed into a row-normalized **Spatial Contiguity Weights Matrix ($W$)** representing all **428 spatial neighbor connections** across CDO, capturing cross-border human and mosquito viral transmission.

---

## 🗂️ 3. Modular Repository Directory Layout

```
Climate-Driven Vector-Borne Outbreak Surveillance/
├── data/
│   ├── cchain_raw/                                 # 35 Raw Project CCHAIN Kaggle CSV tables (Git ignored)
│   ├── processed/                                  # Production CDO engineered datasets & benchmarks
│   │   ├── cchain_cdo_dengue_surveillance_ready.csv # Final CDO matrix (18,880 rows x 59 columns)
│   │   └── cchain_model_benchmarks.csv              # Multi-model evaluation tournament metrics
│   ├── dummy_test_city/                            # Lightweight synthetic test suite (~10 MB)
│   └── processed_dummy_test/                       # Processed synthetic outputs & stress results
│
├── docs/                                           # 📚 Technical Documentation & Defense Suite
│   ├── README.md                                   # Documentation Master Index
│   ├── dataset_data_dictionary.md                  # 59-Feature Data Dictionary & Lineage
│   ├── model_architecture_and_methodology.md       # Deep-dive: Biology, Spatial W, ML & Metrics
│   ├── pipeline_execution_guide.md                 # Reproduction & CLI Execution Guide
│   ├── project_summary_and_defense_guide.md        # 4 Stages of Analytics & Defense Q&A
│   └── randomized_model_stress_test_report.md      # Counterfactual & Noise Robustness Audit
│
├── notebooks/                                      # 📓 Interactive Publication-Ready Jupyter Notebooks
│   ├── 01_cchain_dengue_surveillance_pipeline.ipynb    # Primary CDO pipeline with plots & markdown
│   └── 02_synthetic_validation_and_stress_testing.ipynb # Standalone synthetic validation notebook
│
├── src/                                            # 🐍 Modular Python Source Code
│   ├── __init__.py
│   ├── pipeline.py                                 # Master CDO production pipeline (Phases 1-4)
│   ├── generate_dummy_data.py                      # Synthetic CCHAIN test data generator
│   ├── validation_runner.py                        # 5-stage validation test engine
│   └── stress_testing.py                           # Randomized counterfactual stress-testing engine
│
├── tests/                                          # 🧪 Automated Test Suite (Unittest)
│   ├── __init__.py
│   ├── test_spatial_matrix.py                      # CDO 80-brgy W matrix topology & symmetry tests
│   ├── test_pipeline_integration.py                # CDO end-to-end integration test on real data
│   └── test_notebook_execution.py                  # Headless execution test for all notebook cells
│
├── requirements.txt                                # Comprehensive Python package dependencies
├── README.md                                       # Main repository overview & quickstart
├── PROJECT_SUMMARY_AND_NOTES.md                    # Quick notes & shared sheet claiming formats
└── Project_CCHAIN_Surveillance_Pipeline.ipynb      # Root interactive notebook link
```

---

## 📊 4. CDO Empirical Holdout Performance (Unseen 2019–2022 Data)

The models were trained on 16 years of historical CDO climate and disease records (2003–2018; $N=15,120$) and rigorously evaluated on **4 years of unseen out-of-time CDO test data (2019–2022; $N=3,760$)**:

| Operational Lead Time | Model Architecture | CDO ROC-AUC | CDO PR-AUC | CDO Accuracy | CDO Outbreak Recall (Sensitivity) | Precision | $F_2$-Score (Public Health) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30-Day Lead ($T+1$)** | **Logistic Regression (L2)** | **0.9605** | **0.7914** | 87.71% | **91.30%** | 49.94% | **0.7832** | 0.1142 |
| **30-Day Lead ($T+1$)** | **LightGBM Classifier** | 0.9596 | 0.7879 | **92.63%** | 72.40% | **69.03%** | 0.7170 | **0.0859** |
| **30-Day Lead ($T+1$)** | **XGBoost Classifier** | 0.9601 | 0.7844 | 91.41% | 80.68% | 61.39% | 0.7591 | 0.0990 |
| **30-Day Lead ($T+1$)** | **Random Forest Classifier** | 0.9571 | 0.7458 | 91.28% | 76.22% | 61.68% | 0.7279 | 0.1042 |
| | | | | | | | | |
| **60-Day Lead ($T+2$)** | **XGBoost Classifier** | **0.9537** | **0.7554** | **91.74%** | 73.46% | **64.31%** | 0.7143 | 0.1008 |
| **60-Day Lead ($T+2$)** | **Logistic Regression (L2)** | 0.9526 | 0.7528 | 86.82% | **91.51%** | 48.05% | **0.7749** | 0.1263 |
| **60-Day Lead ($T+2$)** | **Random Forest Classifier** | 0.9493 | 0.7198 | 90.39% | 80.04% | 57.82% | 0.7433 | 0.1061 |
| **60-Day Lead ($T+2$)** | **LightGBM Classifier** | 0.9492 | 0.7333 | 91.69% | 71.76% | 64.50% | 0.7018 | **0.0916** |

---

## 🏥 5. Prescriptive CDO Municipal Decision Matrix

To ensure practical operational utility, model output probabilities are automatically mapped to a **3-Tier Municipal Response Protocol for CDO Local Health Authorities**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Outbreak Prob (P) │ Alert Status         │ Automated CDO City Health Office (CHO) Action    │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ P < 0.30          │ Level 1: Normal      │ Routine community cleanup ("4-S Strategy") and   │
│                   │                      │ standard entomological larval index surveys.     │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ 0.30 <= P < 0.65  │ Level 2: Pre-Emptive │ Targeted biological larviciding (Bti) in open    │
│                   │ Alert                │ canals and dense container zones; mobilize BHWs  │
│                   │                      │ for house-to-house fever surveillance.           │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ P >= 0.65         │ Level 3: Critical    │ Targeted ultra-low volume (ULV) spatial fogging  │
│                   │ Outbreak Warning     │ within 48h; pre-position 200 NS1 rapid test kits │
│                   │                      │ at Barangay Health Center; reserve triage beds   │
│                   │                      │ at JR Borja General Hospital and NMMC.           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 6. Quickstart Execution Guide

### Run Master CDO Production Pipeline:
```bash
python src/pipeline.py
```

### Run Full CDO Automated Test Suite:
```bash
python -m unittest discover -s tests -v
```

### Open Interactive CDO Surveillance Notebook:
```bash
jupyter notebook notebooks/01_cchain_dengue_surveillance_pipeline.ipynb
```

---

## 📜 7. Dataset Source & Academic Course Attribution

* **Course Fulfillment:** This project is prepared and submitted in fulfillment of **Data Mining Course — Laboratory Activity 1**.
* **Primary Dataset Source:** **Project CCHAIN** (*Climate Change and Health Analytics Network*).
  * **Repository:** Kaggle Dataset [`thinkdatasci/project-cchain`](https://www.kaggle.com/datasets/thinkdatasci/project-cchain)
  * **Authors / Partners:** Thinking Machines, EpiMetrics, Inc., Manila Observatory, and PACSII.
  * **Funding Agencies:** The Wellcome Trust & Lacuna Fund.
  * **Data Constituents:** Multi-source integration of DOH epidemiological records, ECMWF ERA5-Land atmospheric reanalysis, Google Open Buildings satellite morphology, and WorldPop demographic density grids (2003–2022).

