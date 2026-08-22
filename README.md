# 🦟 Project CCHAIN: Climate-Driven Vector-Borne Outbreak Surveillance Engine
> **A 4-Stage Data Analytics & Machine Learning Pipeline for Dengue Outbreak Forecasting & LGU Resource Optimization**  
> *Pilot Implementation: Cagayan de Oro City, Philippines (2003–2022)*

---

## 📖 1. Project Overview & Objectives

In tropical urban centers, Local Government Units (LGUs) often rely on reactive responses to climate-induced vector-borne disease surges. This project builds a localized early-warning surveillance engine using the official open-source **Project CCHAIN** dataset (developed by Thinking Machines, EpiMetrics, Manila Observatory, and PACSII).

By linking 20 years of **ERA5 atmospheric climate reanalysis**, **DOH/LGU epidemiological case registries**, **Google Open Buildings footprints**, and **WorldPop high-resolution demographics**, this repository implements all **Four Types of Data Analytics**:

1. **Descriptive:** Historical monthly seasonality and geographic high-risk barangay clustering.
2. **Diagnostic:** Multi-month biological lag dynamics (1-to-3 month thermal/rainfall lags driving *Aedes aegypti* vector breeding and viral incubation).
3. **Predictive:** Supervised machine learning classification (Random Forest) predicting outbreak risks 30 to 60 days ahead (**ROC-AUC = 0.903, Accuracy = 87.0%**).
4. **Prescriptive:** An automated decision matrix for LGUs to optimize spatial larviciding, targeted fogging, and hospital triage bed allocation.

---

## 🗂️ 2. Repository Architecture & Directory Layout

```
DMA/
├── .gitignore                               # Git exclusion rules (ignores raw data & artifacts)
├── README.md                                # Repository documentation & execution guide
├── requirements.txt                         # Python package dependencies
│
├── Project_CCHAIN_Surveillance_Pipeline.ipynb # Interactive end-to-end Jupyter Notebook
├── cchain_pipeline.py                       # Automated end-to-end Python pipeline script
│
├── generate_proposal_pdf.py                 # Script generating the 3-page Lab 1 Proposal PDF
├── generate_summary_pdf.py                  # Script generating the Model Evaluation Summary PDF
├── generate_notebook.py                     # Script generating the notebook template
│
├── cchain_proposal_framework.pdf            # 3-Page Publication-ready Lab 1 Analytical Proposal
├── cchain_model_results_summary.pdf         # 2-Page Executive Model Results & Evaluation Report
├── PROJECT_SUMMARY_AND_NOTES.md             # Complete project summary & defense talking points
│
└── data/
    ├── cchain_raw/                          # Raw Project CCHAIN Kaggle CSV tables (Git ignored)
    │   ├── location.csv                     # Spatial boundary definitions (80 CDO barangays)
    │   ├── disease_lgu_disaggregated_totals.csv # DOH/LGU monthly dengue records
    │   ├── climate_atmosphere.csv           # ECMWF ERA5 meteorological vectors
    │   ├── google_open_buildings.csv        # Urban building density & footprint metrics
    │   └── worldpop_population.csv          # High-resolution demographic exposure metrics
    │
    └── processed/                           # Engineered analysis-ready outputs (Tracked)
        └── cchain_cdo_dengue_surveillance_ready.csv # Final merged matrix (18,960 rows x 37 cols)
```

---

## 📦 3. Data Sources & Schema Mapping

The pipeline relies on the official **Project CCHAIN** dataset on Kaggle (`thinkdatasci/project-cchain`):

| Dataset Table | Scope (CDO) | Primary Extracted Features | Role in Pipeline |
| :--- | :---: | :--- | :--- |
| `location.csv` | 80 Barangays | `adm3_pcode`, `adm4_pcode`, `adm4_en`, `brgy_total_area` | Spatial Master Index |
| `disease_lgu_disaggregated_totals.csv` | 2,490 Rows | `case_total`, `death_total`, `date`, `disease_common_name` | Target Variable ($Y$) |
| `climate_atmosphere.csv` | 584,400 Rows | `pr` (precipitation), `tave`, `heat_index`, `rh`, `wind_speed` | Dynamic Weather Features & Lags ($X$) |
| `google_open_buildings.csv` | 80 Barangays | `google_bldgs_density`, `google_bldgs_pct_built_up_area` | Built-Environment Susceptibility ($Z$) |
| `worldpop_population.csv` | 1,680 Rows | `pop_density_mean`, `pop_count_total` | Human Host Density ($Z$) |

---

## ⚙️ 4. Prerequisites & Installation

### A. Clone Repository & Setup Environment
```bash
git clone https://github.com/your-username/project-cchain-surveillance.git
cd project-cchain-surveillance

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### B. Setup Kaggle API Credentials (For Downloading Raw Data)
Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

---

## 🚀 5. Quickstart: How to Run the Pipeline

### Option 1: Automated Full Pipeline (CLI Execution)
Runs data ingestion, space-time alignment, lag feature engineering, model training, and exports the clean dataset:
```bash
python cchain_pipeline.py
```

### Option 2: Interactive Jupyter Notebook
Open and run cells step-by-step to inspect data tables, lag graphs, and classification metrics:
```bash
jupyter notebook Project_CCHAIN_Surveillance_Pipeline.ipynb
```

### Option 3: Recompile Analytical PDF Deliverables
```bash
python generate_proposal_pdf.py   # Produces cchain_proposal_framework.pdf
python generate_summary_pdf.py    # Produces cchain_model_results_summary.pdf
```

---

## 📊 6. Empirical Results & Performance Evaluation

The model was evaluated using an **out-of-time temporal validation split** (Train: 2003–2018 | Test: 2019–2022 across all 80 barangays):

```
Outbreak Prediction Performance (Evaluated on 2019-2022 Test Period):

• ROC-AUC Score:      0.903 (90.3%)  [Outstanding Discriminative Power]
• Overall Accuracy:   87.0%          [3,340 of 3,840 Monthly Evaluations Correct]
• Outbreak Recall:     72.0%          [Catches 72% of Imminent Outbreaks in Advance]
```

### Top 5 Predictive Climate-Health Features:
1. **`pop_density_mean` (35.05%)** — High human host concentration accelerating vector biting contact.
2. **`google_bldgs_pct_built_up_area` (14.92%)** — Impervious surfaces creating artificial water collection containers.
3. **`heat_index_mean_lag_3m` (9.39%)** — 3-month heat stress accelerating viral incubation (EIP).
4. **`heat_index_mean_lag_2m` (6.93%)** — 2-month ambient temperature accelerating larval development.
5. **`tave_mean_lag_1m` (5.72%)** — 1-month ambient temperature driving adult mosquito biting activity.

---

## 🏥 7. Prescriptive LGU Decision Support Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Predicted Probability (P) │ Risk Level  │ Prescriptive Action Trigger       │
├───────────────────────────┼─────────────┼───────────────────────────────────┤
│ P < 0.30                  │ Level 1     │ Routine community cleanup &       │
│                           │ (Normal)    │ baseline larval index monitoring. │
├───────────────────────────┼─────────────┼───────────────────────────────────┤
│ 0.30 <= P < 0.65          │ Level 2     │ Pre-emptive chemical larviciding  │
│                           │ (Alert)     │ & deployment of health workers.   │
├───────────────────────────┼─────────────┼───────────────────────────────────┤
│ P >= 0.65                 │ Level 3     │ Immediate targeted spatial        │
│                           │ (Outbreak)  │ fogging & hospital surge buffer.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 8. Key Deliverables & Artifacts

* 📄 **Analytical Proposal PDF:** [`cchain_proposal_framework.pdf`](cchain_proposal_framework.pdf)
* 📄 **Model Evaluation Summary PDF:** [`cchain_model_results_summary.pdf`](cchain_model_results_summary.pdf)
* 📓 **Interactive Notebook:** [`Project_CCHAIN_Surveillance_Pipeline.ipynb`](Project_CCHAIN_Surveillance_Pipeline.ipynb)
* 🐍 **End-to-End Pipeline:** [`cchain_pipeline.py`](cchain_pipeline.py)
* 📊 **Analysis-Ready Dataset:** [`data/processed/cchain_cdo_dengue_surveillance_ready.csv`](data/processed/cchain_cdo_dengue_surveillance_ready.csv)

---

## 👥 9. Authors & Course Metadata
* **Course:** Data Mining & Analytics (DMA)
* **Activity:** Laboratory Activity 1 — *Types of Data Analytics*
* **Target Region:** City of Cagayan de Oro, Region X, Philippines
* **Dataset Attribution:** Thinking Machines, EpiMetrics, Manila Observatory, PACSII, Wellcome Trust & Lacuna Fund.
