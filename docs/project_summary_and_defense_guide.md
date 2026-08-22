# Climate-Driven Vector-Borne Dengue Outbreak Surveillance: Summary, Analytics Stages & Oral Defense Guide

> **Course Context**: Data Mining & Analytics (DMA) — Laboratory Activity 1: Types of Data Analytics  
> **Topic Title**: Climate-Driven Vector-Borne Dengue Outbreak Surveillance & LGU Decision Support Engine  
> **Pilot LGU**: Cagayan de Oro City, Northern Mindanao (PSA Code: `PH104305000`)  
> **Data Repository**: Project CCHAIN (Thinking Machines, EpiMetrics, Manila Observatory, PACSII, Wellcome Trust, Lacuna Fund)  

---

## 📌 1. Fast Topic Claiming Formats (For Class Shared Sheet)

### 🌟 Primary Format (Comprehensive):
```text
climate_atmosphere.csv + disease_lgu_disaggregated_totals.csv + google_open_buildings.csv + worldpop_population.csv + location.csv. Can multi-month climate lags (rainfall, temperature, heat index) combined with urban building density predict localized dengue outbreak risks 30 to 60 days in advance to prescriptively allocate LGU vector-control and medical resources?
```

### 🔹 Alternative Format (Machine Learning & Early Warning Focus):
```text
climate_atmosphere.csv + disease_lgu_disaggregated_totals.csv + google_open_buildings.csv + worldpop_population.csv. How can 1-to-3 month biological weather lags and barangay built-environment density forecast dengue epidemic thresholds per barangay for early-warning LGU intervention?
```

---

## 🎯 2. Operational Problem Statement

Local Government Units (LGUs) and municipal health offices consistently suffer from **delayed, reactive interventions** during climate-induced dengue epidemics following monsoonal precipitation and urban heat spikes. 

Because clinical case reporting occurs only after symptomatic hospital presentations, vector-control teams are dispatched when community transmission has already peaked. The absence of an automated spatial surveillance framework that bridges atmospheric lag dynamics with barangay-level structural vulnerabilities results in:
* Overwhelmed emergency and pediatric hospital beds
* Acute stockouts of diagnostic kits, intravenous fluids, and larvicide
* Uncontrolled vector dispersion into adjacent contiguous barangays
* Avoidable patient morbidity and mortality

By modeling the **2-to-8-week non-linear biological lag relationship** between climate anomalies, satellite-observed urban morphology, and disease incidence, municipal health officers can transition to **proactive early warning (30–60 days in advance)** and **prescriptive vector-control allocation**.

---

## 🔬 3. The Four Stages of Data Analytics

```mermaid
graph LR
    A["1. Descriptive Analytics<br/><b>What Happened?</b><br/>Historical cycles, seasonality,<br/>spatial case clustering"] 
    --> B["2. Diagnostic Analytics<br/><b>Why Did It Happen?</b><br/>Thermal EIP acceleration,<br/>impervious runoff traps, lags"]
    --> C["3. Predictive Analytics<br/><b>What Will Happen?</b><br/>30-60 day early warning,<br/>F2-optimized ML classifiers"]
    --> D["4. Prescriptive Analytics<br/><b>What Should We Do?</b><br/>3-tier alert protocols,<br/>targeted fogging & bed reserves"]
```

### 📊 A. Descriptive Analytics *(What happened in the historical record?)*
* **Research Questions**:
  * **Q1**: What are the historical monthly case trajectories and seasonal baseline cycles of dengue across Cagayan de Oro's 80 barangays over 2003–2022?
  * **Q2**: Which geographic clusters consistently exhibit the highest endemic incidence and mortality burden?
* **Methods**: Time-series decomposition, seasonal baseline indexing, geospatial GIS choropleth mapping.
* **Key Findings**: Cases follow a distinct seasonal surge peaking between July and October (monsoon season), with urban centers (e.g., Carmen, Lapasan, Balulang, Kauswagan) accounting for over 45% of total historical case volume.

### 🔍 B. Diagnostic Analytics *(Why did outbreaks and spikes occur?)*
* **Research Questions**:
  * **Q3**: What is the exact biological lag duration between peak precipitation/heat index anomalies and subsequent dengue caseload surges?
  * **Q4**: How do structural built-environment indicators (building footprint density, impervious surface percentage) explain the divergence in outbreak severity between adjacent barangays?
* **Methods**: Cross-correlation lag analysis, Spearman rank correlation, feature interaction modeling.
* **Key Findings**: 
  * Heat index at 2-month and 3-month lags shows strong positive correlation with case surges by accelerating the Extrinsic Incubation Period (EIP).
  * Building density interacts multiplicatively with rainfall: high impervious cover prevents natural infiltration, creating stagnant pools in drainage gutters and artificial containers.

### 📈 C. Predictive Analytics *(What outbreak events will happen in the future?)*
* **Research Questions**:
  * **Q5**: Can machine learning models predict whether a barangay will breach its historical epidemic outbreak threshold ($\ge 75\text{th}$ percentile) 30 to 60 days in advance without future climate leakage?
  * **Q6**: Which model architecture provides the best balance of discriminative power (ROC-AUC / PR-AUC) and epidemiological sensitivity (Recall)?
* **Methods**: Multi-model tournament (Logistic Regression, Random Forest, LightGBM, XGBoost), temporal holdout validation (2003–2018 Train, 2019–2022 Test), $F_2$-score threshold optimization.
* **Key Findings**: 
  * Models achieve **0.957–0.960 ROC-AUC** and **0.784–0.791 PR-AUC** on the 30-day horizon, and **0.949–0.953 ROC-AUC** on the 60-day horizon.
  * $F_2$-threshold optimization captures **72.4% to 91.5% of all imminent outbreaks** on unseen test data.

### 💡 D. Prescriptive Analytics *(What precise actions should the health office take?)*
* **Research Questions**:
  * **Q7**: What is the optimal spatial allocation schedule of vector-control interventions (targeted fogging, chemical larviciding) across priority zones to maximize outbreak containment?
  * **Q8**: How should clinical staff and inpatient bed capacity across district health centers be dynamically scheduled 3 weeks ahead of a predicted surge?
* **Methods**: Constrained resource optimization, risk-stratified 3-tier decision matrix.
* **Key Findings**: Translating model probabilities into discrete alert levels enables pre-emptive intervention (Level 1: Routine sanitation, Level 2: Chemical larviciding in high-risk zones, Level 3: Targeted thermal fogging and emergency bed reservations).

---

## 📊 4. Empirical Performance & Key Metrics Summary

Evaluated across **3,840 unseen holdout space-time records (2019–2022)**:

| Metric | 30-Day Horizon ($T+1$) | 60-Day Horizon ($T+2$) | Public Health Implication |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.9605** | **0.9537** | Near-perfect discrimination between epidemic and normal months. |
| **PR-AUC** | **0.7914** | **0.7554** | High precision-recall area despite 19:1 class imbalance. |
| **Outbreak Recall** | **91.30%** (LogReg) / **80.68%** (XGB) | **91.51%** (LogReg) / **73.46%** (XGB) | Successfully flags 8 to 9 out of every 10 impending outbreaks. |
| **Precision** | **69.03%** (LightGBM) / **61.39%** (XGB) | **64.50%** (LightGBM) / **64.31%** (XGB) | 6 to 7 out of 10 dispatched alerts correspond to genuine surges. |
| **Brier Loss** | **0.0859** | **0.0916** | Well-calibrated continuous probability outputs. |

---

## 🛡️ 5. Oral Defense Strategy & Expected Q&A

### Q1: *"Why do you use lagged weather data instead of current weather?"*
> **Answer**: 
> "Vector biology operates with inherent biological time delays. Mosquito eggs take 7–10 days to hatch and develop into adults. Once an adult female takes an infected blood meal, the virus requires 5–14 days of Extrinsic Incubation (EIP) inside the mosquito before transmission can occur. After biting a human, the intrinsic incubation period in humans takes another 4–10 days before symptoms prompt a clinical visit. 
> Therefore, rainfall and heat anomalies today cause hospital case surges **1 to 2 months later**. Using current-month weather would cause data leakage and eliminate operational lead time. Our model uses strictly past weather ($T-1$, $T-2$, $T-3$, $T-4$) to forecast future outbreaks ($T+1$, $T+2$)."

### Q2: *"Why is building density and population included alongside weather?"*
> **Answer**: 
> "Weather determines *when* the biological conditions are favorable for mosquito breeding and viral replication across the whole city. Satellite building density (Google Open Buildings) and human population (WorldPop) determine *where* those conditions will translate into human infection. Dense urban barangays have high concentrations of artificial water containers and high human contact rates, amplifying transmission compared to rural barangays receiving the same rainfall."

### Q3: *"What is the Spatial Contiguity Matrix ($W$), and why is it important?"*
> **Answer**: 
> "Dengue is not confined within administrative boundaries. Infected mosquitoes and human commuters move between adjacent barangays. We constructed an $80 \times 80$ Queen contiguity spatial weights matrix ($W$) connecting neighboring barangays (428 spatial edges). Multiplying $W$ by historical case vectors ($W \cdot Y$) creates spatial autoregressive features that capture incoming transmission fronts from neighboring hotspots."

### Q4: *"Why did you optimize for $F_2$-Score instead of overall accuracy or standard $F_1$?"*
> **Answer**: 
> "Outbreak surveillance has an extreme asymmetric cost structure. A **False Negative** (failing to predict an outbreak) leads to overwhelmed emergency rooms, stockouts of IV fluids, and preventable deaths. A **False Positive** merely results in precautionary larviciding or community sanitation. The $F_2$-score weights Recall twice as heavily as Precision ($\beta = 2.0$), allowing the model to capture over 91% of outbreaks while maintaining practical precision."

### Q5: *"How do you prevent temporal data leakage in your model evaluation?"*
> **Answer**: 
> "We use a strict temporal holdout split. The models are trained exclusively on historical data from **2003 through 2018 (16 years)** and evaluated on the subsequent **2019 through 2022 test set (4 years)**. We never use random k-fold shuffling across time, and for the 30-day and 60-day horizons, all predictor features are strictly restricted to data available at $T-1$ and $T-2$ respectively."

---

## 📁 6. Documentation & Deliverables Index

* 📘 [Model Architecture & Methodology](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/model_architecture_and_methodology.md)
* 📗 [Dataset Data Dictionary](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/dataset_data_dictionary.md)
* 📙 [Pipeline Execution Guide](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/docs/pipeline_execution_guide.md)
* 🐍 [Python Pipeline Source](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Climate-Driven%20Vector-Borne%20Outbreak%20Surveillance/cchain_pipeline.py)
