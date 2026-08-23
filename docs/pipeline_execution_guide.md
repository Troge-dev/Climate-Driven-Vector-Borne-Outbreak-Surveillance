# Project CCHAIN Surveillance Pipeline Execution Guide

This guide provides step-by-step instructions to execute, reproduce, and validate the **Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine**.

---

## 🛠️ 1. Environment Setup

Ensure Python 3.9+ is installed. Then set up your virtual environment and dependencies:

```bash
# Clone the repository
git clone https://github.com/Troge-dev/Climate-Driven-Vector-Borne-Outbreak-Surveillance.git
cd Climate-Driven-Vector-Borne-Outbreak-Surveillance

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 📂 2. Directory Architecture

```
Climate-Driven Vector-Borne Outbreak Surveillance/
├── data/
│   ├── cchain_raw/                                 # Raw Kaggle CSV tables
│   ├── processed/                                  # Production CDO engineered datasets
│   ├── dummy_test_city/                            # Synthetic validation test suite
│   └── processed_dummy_test/                       # Processed synthetic outputs
├── docs/                                           # Technical documentation suite
├── notebooks/                                      # Interactive Jupyter Notebooks
│   ├── 01_cchain_dengue_surveillance_pipeline.ipynb
│   └── 02_synthetic_validation_and_stress_testing.ipynb
├── src/                                            # Modular Python source code
│   ├── pipeline.py                                 # Master production pipeline
│   ├── generate_dummy_data.py                      # Synthetic dataset generator
│   ├── validation_runner.py                        # Validation suite runner
│   └── stress_testing.py                           # Stress-testing & counterfactual engine
└── tests/                                          # Automated unit & integration tests
    ├── test_spatial_matrix.py
    ├── test_pipeline_integration.py
    └── test_notebook_execution.py
```

---

## 🚀 3. Execution Workflows

### Option A: Run Full Production Pipeline (Cagayan de Oro Dataset)
Executes spatial matrix construction, multi-month lag engineering, multi-model tournament, and threshold optimization:
```bash
python src/pipeline.py
```
* **Output 1:** [`data/processed/cchain_cdo_dengue_surveillance_ready.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/processed/cchain_cdo_dengue_surveillance_ready.csv)
* **Output 2:** [`data/processed/cchain_model_benchmarks.csv`](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/data/processed/cchain_model_benchmarks.csv)

---

### Option B: Run Synthetic Validation & Stress-Testing Suite
```bash
# 1. Generate synthetic test city tables
python src/generate_dummy_data.py

# 2. Run automated validation runner
python src/validation_runner.py

# 3. Run randomized counterfactual & noise stress tests
python src/stress_testing.py
```

---

### Option C: Run Full Automated Test Suite
```bash
python -m unittest discover -s tests
```
Runs 3 comprehensive test suites:
1. `test_spatial_matrix.py`: Verifies $W$ matrix row-stochasticity, adjacency symmetry, and geometry validity.
2. `test_pipeline_integration.py`: Verifies end-to-end pipeline execution and holdout ROC-AUC metrics.
3. `test_notebook_execution.py`: Verifies headless execution of all code cells in both notebooks.

---

### Option D: Interactive Jupyter Notebooks
Launch Jupyter Notebook to inspect step-by-step visual analytics:
```bash
jupyter notebook notebooks/01_cchain_dengue_surveillance_pipeline.ipynb
```
Features included in the notebook:
* Dual-axis 20-year monthly precipitation vs. dengue caseload wave charts.
* Cross-correlation lag structure bar charts ($t-0$ to $t-6$ months).
* Multi-model ROC Curves and Precision-Recall Curves (PR-AUC).
* Top Gini feature importance rankings.
* Automated real-time LGU prescriptive dispatch schedule.
