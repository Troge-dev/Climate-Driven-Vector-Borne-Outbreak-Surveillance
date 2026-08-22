# 📚 Climate-Driven Vector-Borne Dengue Outbreak Surveillance Documentation Portal

Welcome to the comprehensive documentation suite for the **Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine (Project CCHAIN)**.

This repository provides an end-to-end data analytics, machine learning, and decision-support pipeline for forecasting localized dengue epidemics across 80 barangays in Cagayan de Oro City, Philippines (2003–2022).

---

## 📑 Documentation Directory Index

```
docs/
├── README.md                              <-- (You are here) Master Documentation Index
│
├── model_architecture_and_methodology.md  <-- Complete technical & biological guide:
│                                              - Vector biology & Extrinsic Incubation Period (EIP)
│                                              - Spatial Contiguity Matrix W (Queen adjacency)
│                                              - Spatial autoregressive lags (W * Y)
│                                              - Multi-model tournament (LogReg, RF, LightGBM, XGBoost)
│                                              - F2-utility threshold optimization
│                                              - 30-Day and 60-Day zero-leakage early warning
│                                              - Prescriptive LGU 3-tier alert protocols
│
├── dataset_data_dictionary.md             <-- 59-Feature Data Dictionary & Lineage:
│                                              - ECMWF ERA5-Land reanalysis
│                                              - Google Open Buildings satellite morphology
│                                              - WorldPop demographic exposure grids
│                                              - DOH/CHO epidemiological case registries
│                                              - Physical interaction indices & lag definitions
│
├── project_summary_and_defense_guide.md   <-- Project Summary & Class Defense Guide:
│                                              - Fast topic claiming formats for class sheet
│                                              - Core operational problem statement
│                                              - The 4 Stages of Data Analytics (Descriptive, Diagnostic,
│                                                Predictive, Prescriptive)
│                                              - Oral defense talking points and expected Q&A
│
└── pipeline_execution_guide.md            <-- Step-by-Step Reproduction Guide:
                                               - Virtual environment setup & dependencies
                                               - Automated pipeline execution instructions
                                               - Interactive Jupyter Notebook walkthrough
                                               - Verification and troubleshooting
```

---

## 🚀 Quick Navigation Links

| Document | Purpose | Key Content |
| :--- | :--- | :--- |
| 📘 [**Model Architecture & Methodology**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/model_architecture_and_methodology.md) | In-depth technical breakdown | Mathematical formulation of spatial weights $W$, biological lags, ML algorithms, $F_2$-optimization, empirical results. |
| 📗 [**Dataset Data Dictionary**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/dataset_data_dictionary.md) | Complete attribute guide | Exact column names, physical units, formulas, missing value handling, and source lineage. |
| 📙 [**Project Summary & Defense Guide**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/project_summary_and_defense_guide.md) | High-level summary & defense | 4 stages of data analytics, topic claiming strings, expected defense questions and answers. |
| 📕 [**Pipeline Execution Guide**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/pipeline_execution_guide.md) | User manual & reproduction | CLI commands, notebook execution, environment setup, and verification. |

---

## 📊 Core Performance Snapshot (Unseen 2019–2022 Test Data)

```
========================================================================================================================
Operational Horizon        Best Model               ROC-AUC    PR-AUC     Recall (Sensitivity)    Precision    Brier Loss
========================================================================================================================
30-Day Early Warning (T+1) Logistic Regression (L2) 0.9605     0.7914     91.30%                  49.94%       0.1142
30-Day Early Warning (T+1) LightGBM Classifier      0.9596     0.7879     72.40%                  69.03%       0.0859
30-Day Early Warning (T+1) XGBoost Classifier       0.9601     0.7844     80.68%                  61.39%       0.0990
------------------------------------------------------------------------------------------------------------------------
60-Day Early Warning (T+2) XGBoost Classifier       0.9537     0.7554     73.46%                  64.31%       0.1008
60-Day Early Warning (T+2) Logistic Regression (L2) 0.9526     0.7528     91.51%                  48.05%       0.1263
60-Day Early Warning (T+2) LightGBM Classifier      0.9492     0.7333     71.76%                  64.50%       0.0916
========================================================================================================================
```
