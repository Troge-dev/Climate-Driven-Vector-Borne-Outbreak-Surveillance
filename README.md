# Spatial-Temporal Dengue Outbreak Surveillance and Forecasting Engine

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Suite](https://img.shields.io/badge/CI-Passing-success)](.github/workflows/tests.yml)
[![Coursework](https://img.shields.io/badge/Coursework-Data%20Mining%20and%20Applications-blue)](docs/project_summary_and_defense_guide.md)
[![Dataset](https://img.shields.io/badge/Data%20Source-Project%20CCHAIN-blueviolet)](https://www.kaggle.com/datasets/thinkdatasci/project-cchain)

> **A Climate-Driven Machine Learning Decision-Support Study for Cagayan de Oro City**  
> *Pilot Study: 80 Barangays of Cagayan de Oro City, Northern Mindanao, Philippines (2003–2022 Longitudinal Dataset)*  
>
> **Academic Context:** Data Mining and Applications (DMA) — Laboratory Project  
> **Primary Dataset:** Project CCHAIN (*Climate Change and Health Analytics Network* | Kaggle: [`thinkdatasci/project-cchain`](https://www.kaggle.com/datasets/thinkdatasci/project-cchain))  
> **License:** MIT License  

---

> [!NOTE]
> **Project Scope and Academic Attribution:**
> * **Coursework Context:** This repository contains the source code, data preprocessing pipelines, and evaluation benchmarks for the **Data Mining and Applications (DMA)** course project. It is designed as an offline research and decision-support prototype, not an actively deployed live dispatch system for municipal government units.
> * **Data Provenance:** Meteorological, epidemiological, built-environment, and demographic data originate from **Project CCHAIN** (*Climate Change and Health Analytics Network*), an open-access multi-partner initiative by Thinking Machines, EpiMetrics, Manila Observatory, and PACSII (funded by the Wellcome Trust & Lacuna Fund). Multi-source tables were filtered, spatialized, and engineered for the 80 barangays of Cagayan de Oro City.

---

## 1. Project Overview and Problem Statement

Dengue fever is a major recurring public health concern in tropical urban centers like Cagayan de Oro City (`PH104305000`). Local health surveillance systems often encounter delayed reporting bottlenecks, where preventative vector control is initiated reactively after hospital caseloads surge.

This project investigates the predictive value of combining multi-source environmental and spatial datasets to forecast localized dengue outbreak risk at **30-day ($T+1$) and 60-day ($T+2$) operational lead times**:
* **Atmospheric Climate Reanalysis (ECMWF ERA5-Land):** Captures multi-month lag dynamics of precipitation ($Pr$), ambient temperatures ($T_{ave}, T_{min}, T_{max}$), relative humidity ($RH$), and heat index.
* **Spatial Transmission Topology ($W$ Matrix):** Models cross-barangay transmission via a row-normalized first-order spatial contiguity matrix across CDO's 80 barangays (428 topological edges).
* **Urban Morphology and Demographics:** Combines Google Open Buildings structural density and WorldPop population estimates to model micro-urban exposure and runoff risks.
* **Machine Learning Tournament:** Benchmarks Regularized Logistic Regression, Random Forest, LightGBM, and XGBoost using $F_2$-score threshold calibration to prioritize outbreak sensitivity.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           4-STAGE ANALYTICS & MODELING PIPELINE                             │
├───────────────────────────┬─────────────────────────────┬───────────────────────────────────┤
│ Stage                     │ Analytics Type              │ Description / Output              │
├───────────────────────────┼─────────────────────────────┼───────────────────────────────────┤
│ 1. Geospatial & Matrix W  │ Descriptive Analytics       │ 80-Brgy WKT Boundaries & W Matrix │
│ 2. Biological Lag Engine  │ Diagnostic Analytics        │ 1-4 Month Climatic Lags & Spills  │
│ 3. Multi-Model ML Bench   │ Predictive Analytics        │ 30D & 60D Outbreak Classifiers    │
│ 4. Triage Decision Matrix │ Prescriptive Framework      │ Conceptual 3-Tier Alert Protocol  │
└───────────────────────────┴─────────────────────────────┴───────────────────────────────────┘
```

---

## 2. Geographic and Spatial Scope

The pipeline covers all **80 barangays of Cagayan de Oro City** (`PH104305000`):
* **Urban Lowlands and High-Density Corridors:** Carmen, Lapasan, Kauswagan, Balulang, Macasandig, Puntod, Bulua, Nazareth, Camaman-an, Gusa.
* **Coastal and Port Districts:** Macabalan, Bonbon, Bayabas, Puerto, Tablon, Agusan.
* **Expanding Peri-Urban Corridors:** Upper Puerto, Lumbia, Canitoan, Iponan, Pagatpat.
* **Upland Rural and Forested Enclaves:** Dansolihon, Besigan, Tumpagon, Pigsag-an, Tignapoloan.

The boundary polygon geometries (`brgy_geography.csv`) are converted into a row-stochastic **Spatial Contiguity Weights Matrix ($W$)** representing 428 topological neighbor connections, enabling spatial lag feature engineering ($W \cdot Y$).

---

## 3. Repository Structure

```
Climate-Driven Vector-Borne Outbreak Surveillance/
├── .github/
│   └── workflows/
│       └── tests.yml                               # GitHub Actions CI workflow (runs on push/PR)
├── data/
│   ├── cchain_raw/                                 # Raw Project CCHAIN Kaggle CSV tables (Git-ignored)
│   ├── processed/                                  # Production CDO engineered datasets & benchmarks
│   │   ├── cchain_cdo_dengue_surveillance_ready.csv # Master CDO matrix (18,880 rows x 59 columns)
│   │   └── cchain_model_benchmarks.csv              # Multi-model evaluation tournament metrics
│   └── dummy_test_city/                            # Schema-compliant synthetic test dataset (~8 MB)
│
├── docs/                                           # Technical Documentation & Defense Guides
│   ├── README.md                                   # Documentation Index
│   ├── lab1_analytical_report_submission.md        # Lab 1 Analytical Report (Primary Submission)
│   ├── dataset_data_dictionary.md                  # 59-Feature Data Dictionary & Lineage
│   ├── model_architecture_and_methodology.md       # Biology, Spatial Matrix W, ML Architectures
│   ├── pipeline_execution_guide.md                 # Step-by-Step Reproduction Guide
│   ├── project_summary_and_defense_guide.md        # Defense Q&A & Conceptual Framework
│   └── randomized_model_stress_test_report.md      # Counterfactual & Noise Robustness Audit
│
├── notebooks/                                      # Interactive Jupyter Notebooks
│   ├── 01_cchain_dengue_surveillance_pipeline.ipynb    # CDO pipeline with plots & analysis
│   └── 02_synthetic_validation_and_stress_testing.ipynb # Standalone synthetic validation notebook
│
├── scripts/                                        # Utility Scripts
│   └── create_notebooks.py                         # Generates annotated notebooks
│
├── src/                                            # Modular Python Source Code
│   ├── __init__.py
│   ├── pipeline.py                                 # Master CDO production pipeline
│   ├── generate_dummy_data.py                      # Synthetic CCHAIN test data generator
│   ├── validation_runner.py                        # 5-stage synthetic validation engine
│   └── stress_testing.py                           # Monte Carlo & perturbation stress-tester
│
├── tests/                                          # Automated Test Suite (Unittest)
│   ├── __init__.py
│   ├── test_target_leakage.py                      # Unit tests for pre-2019 threshold invariance
│   ├── test_dummy_data_integration.py              # End-to-end pipeline test on synthetic data
│   ├── test_spatial_matrix.py                      # Spatial matrix symmetry & stochasticity tests
│   ├── test_pipeline_integration.py                # CDO real-data integration test (if present)
│   └── test_notebook_execution.py                  # Headless execution test for all notebook cells
│
├── .gitattributes                                  # Line endings & large CSV tracking configuration
├── .gitignore
├── LICENSE                                         # MIT License
├── requirements.txt                                # Python package dependencies
└── README.md                                       # Main repository overview & quickstart
```

---

## 4. Empirical Holdout Performance (Unseen 2019–2022 Data)

Models are trained on 16 years of historical CDO climate and disease records (2003–2018; $N=15,120$) and evaluated on **4 years of strictly out-of-time test data (2019–2022; $N=3,760$)** with zero future target leakage:

| Lead Time | Model Architecture | ROC-AUC | PR-AUC | Accuracy | Sensitivity (Recall) | Precision | $F_1$-Score | $F_2$-Score (Opt) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30-Day Lead ($T+1$)** | **Logistic Regression (L2)** | **0.9637** | **0.8148** | 87.60% | **91.74%** | 50.45% | 0.6510 | **0.7884** | 0.1105 |
| **30-Day Lead ($T+1$)** | **LightGBM Classifier** | 0.9635 | 0.8127 | **93.07%** | 72.93% | **72.34%** | **0.7263** | 0.7281 | **0.0827** |
| **30-Day Lead ($T+1$)** | **XGBoost Classifier** | 0.9634 | 0.8101 | 91.74% | 83.47% | 63.03% | 0.7182 | 0.7839 | 0.0961 |
| **30-Day Lead ($T+1$)** | **Random Forest Classifier** | 0.9603 | 0.7659 | 91.87% | 75.83% | 65.30% | 0.7017 | 0.7346 | 0.1018 |
| | | | | | | | | | |
| **60-Day Lead ($T+2$)** | **XGBoost Classifier** | **0.9564** | **0.7801** | **91.30%** | 75.41% | 62.93% | **0.6861** | 0.7254 | 0.1000 |
| **60-Day Lead ($T+2$)** | **Logistic Regression (L2)** | 0.9559 | 0.7750 | 87.47% | **91.12%** | 50.17% | 0.6471 | **0.7833** | 0.1224 |
| **60-Day Lead ($T+2$)** | **LightGBM Classifier** | 0.9520 | 0.7551 | 91.38% | 75.62% | **63.21%** | 0.6886 | 0.7276 | **0.0920** |
| **60-Day Lead ($T+2$)** | **Random Forest Classifier** | 0.9537 | 0.7406 | 90.08% | 82.02% | 57.45% | 0.6757 | 0.7556 | 0.1039 |

*Note: All classification thresholds are tuned on training data using $F_2$-score optimization to prioritize recall over precision in accordance with epidemiological screening principles.*

---

## 5. Conceptual Decision-Support Matrix

To illustrate how model probabilities could inform municipal health protocols, output scores can be mapped to a conceptual 3-tier advisory framework:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Outbreak Prob (P) │ Alert Status         │ Conceptual Municipal Response Guideline          │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ P < 0.30          │ Level 1: Baseline    │ Standard environmental sanitation and routine    │
│                   │                      │ community container inspections.                 │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ 0.30 <= P < 0.65  │ Level 2: Pre-Emptive │ Targeted source reduction in high-risk zones;    │
│                   │ Advisory             │ intensified barangay fever surveillance.         │
├───────────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ P >= 0.65         │ Level 3: High Risk   │ Targeted adulticiding/larviciding review;        │
│                   │ Warning              │ prepositioning diagnostic supplies at centers.   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Limitations and Methodological Considerations

> [!WARNING]
> **Key Epidemiological & Operational Caveats:**

1. **Target Leakage Remediation:**
   In earlier iterations, the per-barangay 75th percentile outbreak threshold was computed across the full 2003–2022 dataset, allowing future test distribution statistics to influence past label definitions. In the current implementation, the 75th percentile threshold (with a 5-case minimum floor) is **computed strictly from the pre-2019 training subset** and frozen across all evaluation periods.
2. **Sensitivity vs. Precision Trade-off (Human-in-the-Loop Requirement):**
   While models such as Logistic Regression attain high outbreak recall (~91.7%), precision remains around ~50.5%. This implies approximately 1 in 2 predicted alerts may be a false alarm. In practical settings, model predictions must serve as an **advisory decision-support signal for epidemiologists**, not an automated operational trigger.
3. **Data Availability & Synthetic Test Suite:**
   The full Project CCHAIN raw dataset (~35 tables) is hosted externally on Kaggle due to size constraints. To enable automated testing and CI without requiring Kaggle credentials, a schema-compliant synthetic dataset is committed under `data/dummy_test_city/`.
4. **Temporal Granularity & Reporting Delays:**
   The surveillance grid operates on monthly aggregated data. Clinical confirmation delays and asymptomatic cases are inherent in longitudinal public health registries.

---

## 7. Quickstart Guide

### 1. Environment Setup
```bash
git clone https://github.com/Troge-dev/Climate-Driven-Vector-Borne-Outbreak-Surveillance.git
cd Climate-Driven-Vector-Borne-Outbreak-Surveillance
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (Uses Synthetic Data)
```bash
python -m unittest discover -s tests -v
```

### 3. Run Production Pipeline (or Synthetic Fallback)
```bash
python src/pipeline.py
```

### 4. Launch Interactive Analysis Notebook
```bash
jupyter notebook notebooks/01_cchain_dengue_surveillance_pipeline.ipynb
```

---

## 8. Dataset Attribution and References

* **Primary Dataset:** **Project CCHAIN** (*Climate Change and Health Analytics Network*)
  * **Repository:** Kaggle Dataset [`thinkdatasci/project-cchain`](https://www.kaggle.com/datasets/thinkdatasci/project-cchain)
  * **Consortium:** Thinking Machines, EpiMetrics Inc., Manila Observatory, PACSII.
  * **Funding:** Wellcome Trust & Lacuna Fund.
* **Atmospheric Data:** ECMWF ERA5-Land Monthly Reanalysis (C3S / Copernicus).
* **Built Environment:** Google Open Buildings Dataset (V3).
* **Demographics:** WorldPop Spatial Demographic Datasets (University of Southampton).
