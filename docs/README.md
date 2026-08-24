# Cagayan de Oro City Dengue Early Warning Documentation Portal

> **Comprehensive Technical Documentation Suite for the CDO Dengue Outbreak Surveillance Engine**  
>
> **Academic Context:** Data Mining and Applications (DMA) — Laboratory Project  
> **Primary Dataset Source:** Project CCHAIN (*Climate Change and Health Analytics Network* | Kaggle: [`thinkdatasci/project-cchain`](https://www.kaggle.com/datasets/thinkdatasci/project-cchain))  

---

## 1. Overview of Documentation Suite

This portal hosts the in-depth documentation, mathematical modeling specifications, data dictionaries, execution guides, and oral defense references for the **Cagayan de Oro City Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine**.

The system evaluates 30-day and 60-day operational early warnings across all **80 barangays of Cagayan de Oro City** by synthesizing 20 years of downscaled meteorological reanalysis (ECMWF ERA5-Land), local DOH/CHO epidemiological registers, Google Open Buildings satellite morphology, and WorldPop population exposure counts.

---

## 2. Documentation Directory Index

```
docs/
├── README.md                              <-- Master Documentation Index
│
├── lab1_analytical_report_submission.md   <-- Lab 1 Analytical Report (Primary Submission):
│                                              - Topic claiming text for shared class sheet
│                                              - 2-3 sentence core scenario & problem statement
│                                              - 8 analytical questions covering the 4 types of analytics
│                                              - Simple-English methodologies & key findings
│                                              - 10-minute presentation slide outline (LAB 1 - PPT)
│                                              - 300-500 word contribution essay (LAB 1 - CONTRIBUTION)
│
├── model_architecture_and_methodology.md  <-- Technical and Biological Guide:
│                                              - Vector biology & Extrinsic Incubation Period (EIP)
│                                              - CDO 80-Barangay Spatial Contiguity Matrix W (428 edges)
│                                              - Spatial autoregressive lags (W * Y)
│                                              - Multi-model tournament (LogReg, RF, LightGBM, XGBoost)
│                                              - F2-utility threshold optimization for public health
│                                              - 30-Day and 60-Day zero-leakage early warning horizons
│                                              - 3-tier municipal prescriptive response framework
│
├── dataset_data_dictionary.md             <-- 59-Feature Data Dictionary & Lineage:
│                                              - ECMWF ERA5-Land reanalysis for CDO coordinates
│                                              - Google Open Buildings morphology in CDO
│                                              - WorldPop demographic exposure grids (2003-2022)
│                                              - DOH Region X / CDO case & death totals
│                                              - Physical interaction terms (Runoff, Heat Trap, Exposure)
│
├── project_summary_and_defense_guide.md   <-- Project Summary & Defense Guide:
│                                              - Fast topic claiming formats for coursework submission
│                                              - Core operational problem statement for CDO
│                                              - The 4 Stages of Data Analytics (Descriptive, Diagnostic,
│                                                Predictive, Prescriptive) tailored for CDO
│                                                - Oral defense talking points and expected panel Q&A
│
├── pipeline_execution_guide.md            <-- Step-by-Step Reproduction Guide:
│                                              - Virtual environment setup & dependencies
│                                              - Automated pipeline execution instructions
│                                              - Interactive Jupyter Notebook walkthrough
│                                              - Automated test suite execution
│
└── randomized_model_stress_test_report.md <-- Robustness & Counterfactual Audit:
                                               - Monte Carlo state-space probing (N=2,000)
                                               - Monsoon surge & drought scenarios
                                               - Noise jitter & sensor stability analysis
```

---

## 3. Quick Navigation Links

| Document | Purpose | Key Content |
| :--- | :--- | :--- |
| [**Lab 1 Analytical Report**](lab1_analytical_report_submission.md) | Primary Coursework Submission | 4 stages of analytics, simple English Q&A, slide outline, and individual essay. |
| [**Model Architecture & Methodology**](model_architecture_and_methodology.md) | In-depth technical breakdown | Spatial contiguity $W$, non-linear lag biology, 4 ML classifiers, $F_2$-optimization, empirical holdout metrics. |
| [**Dataset Data Dictionary**](dataset_data_dictionary.md) | Complete attribute guide | 59 features, formulas, physical units, missing value imputation, and spatial granularity. |
| [**Project Summary & Defense Guide**](project_summary_and_defense_guide.md) | High-level summary & defense | 4 stages of data analytics, problem statements, expected panel questions, and model answers. |
| [**Pipeline Execution Guide**](pipeline_execution_guide.md) | User manual & reproduction | CLI commands, interactive notebook instructions, and test runner execution. |
| [**Stress Testing & Robustness Report**](randomized_model_stress_test_report.md) | Sensitivity & scenario audit | Extreme storm surges, prolonged droughts, cold shocks, and Gaussian sensor perturbation tests. |

---

## 4. Empirical Holdout Performance Snapshot (Unseen 2019–2022 Data)

```
========================================================================================================================
Operational Horizon        Best Model               ROC-AUC     PR-AUC      Recall (Sensitivity) Precision  Brier Loss
========================================================================================================================
30-Day Early Warning (T+1) Logistic Regression (L2) 0.9637      0.8148      91.74%                   50.45%     0.1105
30-Day Early Warning (T+1) LightGBM Classifier      0.9635      0.8127      72.93%                   72.34%     0.0827
30-Day Early Warning (T+1) XGBoost Classifier       0.9634      0.8101      83.47%                   63.03%     0.0961
30-Day Early Warning (T+1) Random Forest Classifier 0.9603      0.7659      75.83%                   65.30%     0.1018
------------------------------------------------------------------------------------------------------------------------
60-Day Early Warning (T+2) XGBoost Classifier       0.9564      0.7801      75.41%                   62.93%     0.1000
60-Day Early Warning (T+2) Logistic Regression (L2) 0.9559      0.7750      91.12%                   50.17%     0.1224
60-Day Early Warning (T+2) LightGBM Classifier      0.9520      0.7551      75.62%                   63.21%     0.0920
60-Day Early Warning (T+2) Random Forest Classifier 0.9537      0.7406      82.02%                   57.45%     0.1039
========================================================================================================================
```
