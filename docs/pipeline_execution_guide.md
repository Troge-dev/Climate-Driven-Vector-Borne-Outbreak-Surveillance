# Project CCHAIN Surveillance Pipeline Execution Guide

This guide provides step-by-step instructions to execute, reproduce, and validate the **Climate-Driven Vector-Borne Dengue Outbreak Surveillance Engine**.

---

## 1. Environment Setup

Ensure Python 3.10+ is installed. Then set up your virtual environment and dependencies:

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

## 2. Directory Architecture

```
Climate-Driven Vector-Borne Outbreak Surveillance/
├── data/
│   ├── cchain_raw/                                 # Raw Kaggle CSV tables (gitignored)
│   ├── processed/                                  # Production CDO engineered datasets
│   └── dummy_test_city/                            # Synthetic validation test suite
├── docs/                                           # Technical documentation suite
├── notebooks/                                      # Interactive Jupyter Notebooks
│   ├── 01_cchain_dengue_surveillance_pipeline.ipynb
│   └── 02_synthetic_validation_and_stress_testing.ipynb
├── scripts/                                        # Notebook generator scripts
│   └── create_notebooks.py
├── src/                                            # Modular Python source code
│   ├── pipeline.py                                 # Master production pipeline
│   ├── generate_dummy_data.py                      # Synthetic dataset generator
│   ├── validation_runner.py                        # Validation suite runner
│   └── stress_testing.py                           # Stress-testing & counterfactual engine
└── tests/                                          # Automated unit & integration tests
    ├── test_target_leakage.py
    ├── test_dummy_data_integration.py
    ├── test_spatial_matrix.py
    ├── test_pipeline_integration.py
    └── test_notebook_execution.py
```

---

## 3. Execution Workflows

### Option A: Run Full Production Pipeline
Executes spatial matrix construction, multi-month lag engineering, multi-model tournament, and threshold optimization (falls back to synthetic data if raw CCHAIN tables are not present):
```bash
python src/pipeline.py
```
* **Output 1:** [`data/processed/cchain_cdo_dengue_surveillance_ready.csv`](../data/processed/cchain_cdo_dengue_surveillance_ready.csv)
* **Output 2:** [`data/processed/cchain_model_benchmarks.csv`](../data/processed/cchain_model_benchmarks.csv)

---

### Option B: Run Full Automated Test Suite (Uses Synthetic Data)
```bash
python -m unittest discover -s tests -v
```
Runs 5 comprehensive test suites:
1. `test_target_leakage.py`: Verifies temporal threshold invariance against post-2019 test data.
2. `test_dummy_data_integration.py`: Verifies end-to-end pipeline execution on synthetic data.
3. `test_spatial_matrix.py`: Verifies $W$ matrix row-stochasticity, adjacency symmetry, and geometry validity.
4. `test_pipeline_integration.py`: Verifies CDO real-data execution and holdout ROC-AUC metrics (if data present).
5. `test_notebook_execution.py`: Verifies headless execution of all code cells in both notebooks.

---

### Option C: Interactive Jupyter Notebooks
Launch Jupyter Notebook to inspect step-by-step visual analytics:
```bash
jupyter notebook notebooks/01_cchain_dengue_surveillance_pipeline.ipynb
```
Features included in the notebook:
* Dual-axis 20-year monthly precipitation vs. dengue caseload wave charts.
* Cross-correlation lag structure bar charts ($t-0$ to $t-6$ months).
* Multi-model ROC Curves and Precision-Recall Curves (PR-AUC).
* Top Gini feature importance rankings.
* Automated prescriptive decision matrix mapping.
