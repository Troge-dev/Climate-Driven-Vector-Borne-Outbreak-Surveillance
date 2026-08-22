# Climate-Driven Vector-Borne Dengue Outbreak Surveillance: Pipeline Execution, Data Ingestion & Reproduction Guide

> **Repository**: Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine  
> **Target OS**: Windows, Linux, macOS  
> **Python Version**: Python 3.9+ (Python 3.10 / 3.11 recommended)  

---

## 1. Environment Setup & Dependency Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Troge-dev/Climate-Driven-Vector-Borne-Outbreak-Surveillance.git
cd Climate-Driven-Vector-Borne-Outbreak-Surveillance
```

### Step 2: Create & Activate Virtual Environment
* **On Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### Core Package Stack (`requirements.txt`):
* `numpy`, `pandas` (Vectorized numerical and tabular data processing)
* `shapely`, `geopandas` (Geospatial WKT polygon parsing & topological contiguity calculation)
* `scikit-learn` (Regularized logistic regression, random forests, scalers, classification metrics)
* `lightgbm`, `xgboost` (High-performance gradient boosting tree algorithms)
* `reportlab` (PDF generation engine)
* `jupyter`, `notebook` (Interactive exploratory analytics environment)

---

## 2. Dataset Ingestion & Directory Structure

The pipeline expects raw CSV tables in `data/cchain_raw/`.

```
Climate-Driven Vector-Borne Outbreak Surveillance/
├── data/
│   ├── cchain_raw/                             # Raw CSVs from Kaggle
│   │   ├── location.csv                        # Spatial boundary definitions (80 CDO barangays)
│   │   ├── disease_lgu_disaggregated_totals.csv# DOH/LGU monthly dengue records
│   │   ├── climate_atmosphere.csv              # ECMWF ERA5 meteorological vectors
│   │   ├── google_open_buildings.csv           # Urban building density & footprint metrics
│   │   ├── worldpop_population.csv             # High-resolution demographic exposure metrics
│   │   └── brgy_geography.csv                  # WKT geometry polygon boundaries
│   │
│   └── processed/                              # Analysis-ready outputs
│       ├── cchain_cdo_dengue_surveillance_ready.csv  # Merged space-time dataset
│       └── cchain_model_benchmarks.csv               # Model tournament evaluation metrics
│
├── docs/                                       # Comprehensive Documentation
│   ├── README.md                               # Documentation Index
│   ├── model_architecture_and_methodology.md   # Complete Model & Mathematical Guide
│   ├── dataset_data_dictionary.md              # 59-Feature Data Dictionary
│   ├── project_summary_and_defense_guide.md    # Defense Talking Points & Q&A
│   └── pipeline_execution_guide.md             # This Execution Guide
│
├── cchain_pipeline.py                          # Automated End-to-End Pipeline Script
└── Project_CCHAIN_Surveillance_Pipeline.ipynb  # Interactive Jupyter Notebook
```

---

## 3. Running the End-to-End Pipeline

Execute the master Python pipeline from the root directory:

```bash
python cchain_pipeline.py
```

### Pipeline Execution Phases:
1. **[1/6] Geography & Contiguity Matrix**: Parses `location.csv` and `brgy_geography.csv`, identifying 80 barangays in CDO, constructing an $80 \times 80$ row-normalized Queen contiguity spatial weights matrix $\mathbf{W}$ with 428 spatial edges.
2. **[2/6] Health Record Aggregation**: Filters `disease_lgu_disaggregated_totals.csv` for Dengue Fever in CDO, aggregating monthly cases and mortality per barangay.
3. **[3/6] Climate Reanalysis Processing**: Aggregates ECMWF ERA5 daily reanalysis into monthly totals, means, heat indices, and relative humidity.
4. **[4/6] Built Environment & Demographics**: Ingests Google Open Buildings satellite morphology and WorldPop population grids.
5. **[5/6] Space-Time Grid Alignment & Feature Engineering**: Creates full space-time grid ($80 \times 236$ months), computes 1m–4m meteorological lags, rolling averages, physical interaction indices, autoregressive case lags, and spatial autoregressive spillovers ($\mathbf{W} \cdot \mathbf{Y}$). Exports `cchain_cdo_dengue_surveillance_ready.csv`.
6. **[6/6] Multi-Model Tournament & Benchmarking**: Evaluates Logistic Regression, Random Forest, LightGBM, and XGBoost across 30-Day and 60-Day lead horizons using strict temporal holdout validation (Train: 2003–2018, Test: 2019–2022). Calibrates optimal $F_2$-score thresholds and exports `cchain_model_benchmarks.csv`.

---

## 4. Interactive Jupyter Notebook Walkthrough

To inspect data tables, visualize spatial heatmaps, plot correlation lag charts, and run interactive model predictions:

```bash
jupyter notebook Project_CCHAIN_Surveillance_Pipeline.ipynb
```

The notebook contains step-by-step code blocks covering:
* Data exploratory analysis (EDA) and seasonality decomposition.
* Spatial neighbor visualization.
* Lag cross-correlation heatmaps between climate variables and dengue incidence.
* Model training, ROC curve plotting, Precision-Recall curve generation, and feature importance rankings.
* Prescriptive scenario simulation and LGU alert dispatch generation.

---

## 5. Output Verification & Sanity Checks

After running the pipeline, verify that the following files exist and match expected dimensions:

| Output Artifact | Expected Path | Validation Check |
| :--- | :--- | :--- |
| **Merged Panel Dataset** | `data/processed/cchain_cdo_dengue_surveillance_ready.csv` | 18,880 rows, 59 columns, 0 missing values in lag features |
| **Benchmark CSV Table** | `data/processed/cchain_model_benchmarks.csv` | 8 rows (4 models $\times$ 2 horizons), ROC-AUC $> 0.94$, PR-AUC $> 0.71$ |

---

## 6. Troubleshooting Common Issues

### Issue 1: Missing Geometry Dependencies
* **Symptom**: `ImportError: cannot import name 'shapely'` or geometry parsing error.
* **Resolution**: Run `pip install --upgrade shapely geopandas` to ensure pre-compiled C binaries are installed.

### Issue 2: Memory Footprint during Grid Creation
* **Symptom**: Memory warning when merging large space-time grids.
* **Resolution**: The pipeline uses vectorized pandas merging with explicit categorical/date keys. Ensure at least 2 GB of free RAM.

### Issue 3: Class Weight Setting in LightGBM/XGBoost
* **Symptom**: `LightGBMError: scale_pos_weight parameter error`.
* **Resolution**: The pipeline dynamically computes `pos_weight = (N_neg) / N_pos` ($\approx 19.17$) directly from the training split before training.
