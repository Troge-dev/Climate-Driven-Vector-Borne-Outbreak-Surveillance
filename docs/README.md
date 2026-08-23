# 📚 Cagayan de Oro City Dengue Early Warning Documentation Portal
> **Comprehensive Technical Documentation Suite for the CDO LGU Dengue Outbreak Surveillance Engine**  
> *Developed for the Cagayan de Oro City Health Office (CHO) & City Disaster Risk Reduction and Management Office (CDRRMO)*  
>
> 🎓 **Academic Fulfillment:** **Data Mining Course — Laboratory Activity 1**  
> 🌐 **Primary Dataset Source:** **Project CCHAIN** (*Climate Change and Health Analytics Network* | Kaggle: [`thinkdatasci/project-cchain`](https://www.kaggle.com/datasets/thinkdatasci/project-cchain))

---

## 🏛️ Welcome to the CDO LGU Surveillance Portal

This portal hosts the in-depth documentation, mathematical modeling specifications, data dictionaries, execution guides, and oral defense references for the **Cagayan de Oro City Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine**.

The system provides 30-day and 60-day operational early warnings across all **80 barangays of Cagayan de Oro City** by synthesizing 20 years of downscaled meteorological reanalysis (ECMWF ERA5-Land), local DOH/CHO epidemiological registers, Google Open Buildings satellite morphology, and WorldPop population exposure counts.

---

## 📑 Documentation Directory Index

```
docs/
├── README.md                              <-- (You are here) Master Documentation Index
│
├── model_architecture_and_methodology.md  <-- CDO-Centric Technical & Biological Guide:
│                                              - Local vector biology & Extrinsic Incubation Period (EIP)
│                                              - CDO 80-Barangay Spatial Contiguity Matrix W (428 edges)
│                                              - Spatial autoregressive lags (W * Y)
│                                              - Multi-model tournament (LogReg, RF, LightGBM, XGBoost)
│                                              - F2-utility threshold optimization for public health
│                                              - 30-Day and 60-Day zero-leakage early warning horizons
│                                              - CDO LGU 3-tier municipal prescriptive response matrix
│
├── dataset_data_dictionary.md             <-- 59-Feature Data Dictionary & Lineage:
│                                              - ECMWF ERA5-Land reanalysis for CDO coordinates
│                                              - Google Open Buildings morphology in CDO
│                                              - WorldPop demographic exposure grids (2003-2022)
│                                              - DOH Region X / CDO CHO case & death totals
│                                              - Physical interaction terms (Runoff, Heat Trap, Exposure)
│
├── project_summary_and_defense_guide.md   <-- CDO LGU Project Summary & Defense Guide:
│                                              - Fast topic claiming formats for 3rd Year class sheet
│                                              - Core operational problem statement for CDO City
│                                              - The 4 Stages of Data Analytics (Descriptive, Diagnostic,
│                                                Predictive, Prescriptive) tailored for CDO LGU
│                                              - Oral defense talking points and expected panel Q&A
│
├── pipeline_execution_guide.md            <-- Step-by-Step Reproduction Guide:
│                                              - Virtual environment setup & dependencies
│                                              - Automated CDO pipeline execution instructions
│                                              - Interactive CDO Jupyter Notebook walkthrough
│                                              - CDO-centric automated test suite execution
│
└── randomized_model_stress_test_report.md <-- Robustness & Counterfactual Audit:
                                               - Monte Carlo state-space probing (N=2,000)
                                               - CDO monsoon surge & drought scenarios
                                               - Noise jitter & sensor stability analysis
```

---

## 🚀 Quick Navigation Links

| Document | Purpose | Key Content |
| :--- | :--- | :--- |
| 📘 [**Model Architecture & Methodology**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/model_architecture_and_methodology.md) | In-depth technical breakdown | Spatial contiguity $W$, non-linear lag biology, 4 ML classifiers, $F_2$-optimization, empirical CDO metrics. |
| 📗 [**Dataset Data Dictionary**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/dataset_data_dictionary.md) | Complete attribute guide | 59 features, formulas, physical units, missing value imputation, and CDO spatial granularity. |
| 📙 [**Project Summary & Defense Guide**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/project_summary_and_defense_guide.md) | High-level summary & defense | 4 stages of data analytics, CDO LGU problem statements, expected panel questions and model answers. |
| 📕 [**Pipeline Execution Guide**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/pipeline_execution_guide.md) | User manual & reproduction | CLI commands, interactive notebook instructions, and CDO test runner execution. |
| 📓 [**Stress Testing & Robustness Report**](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/randomized_model_stress_test_report.md) | Sensitivity & scenario audit | Extreme storm surges, prolonged droughts, cold shocks, and Gaussian sensor perturbation tests. |

---

## 📊 CDO Holdout Performance Snapshot (Unseen 2019–2022 Data)

```
========================================================================================================================
Operational Horizon        Best Model               CDO ROC-AUC CDO PR-AUC  CDO Recall (Sensitivity) Precision  Brier Loss
========================================================================================================================
30-Day Early Warning (T+1) Logistic Regression (L2) 0.9605      0.7914      91.30%                   49.94%     0.1142
30-Day Early Warning (T+1) LightGBM Classifier      0.9596      0.7879      72.40%                   69.03%     0.0859
30-Day Early Warning (T+1) XGBoost Classifier       0.9601      0.7844      80.68%                   61.39%     0.0990
30-Day Early Warning (T+1) Random Forest Classifier 0.9571      0.7458      76.22%                   61.68%     0.1042
------------------------------------------------------------------------------------------------------------------------
60-Day Early Warning (T+2) XGBoost Classifier       0.9537      0.7554      73.46%                   64.31%     0.1008
60-Day Early Warning (T+2) Logistic Regression (L2) 0.9526      0.7528      91.51%                   48.05%     0.1263
60-Day Early Warning (T+2) LightGBM Classifier      0.9492      0.7333      71.76%                   64.50%     0.0916
60-Day Early Warning (T+2) Random Forest Classifier 0.9493      0.7198      80.04%                   57.82%     0.1061
========================================================================================================================
```
