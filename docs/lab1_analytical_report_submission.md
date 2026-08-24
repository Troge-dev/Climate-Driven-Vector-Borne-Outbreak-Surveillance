# Data Mining and Applications (DMA)
## Laboratory Activity 1: Types of Data Analytics
### Topic: Climate-Driven Dengue Outbreak Surveillance for Cagayan de Oro City

**Submission Details:**
* **Course:** Data Mining and Applications (DMA)
* **Activity:** Laboratory Activity 1 — Types of Data Analytics
* **Activity Label:** `LAB 1 - ANALYTICAL REPORT`
* **Dataset Used:** Project CCHAIN (*Climate Change and Health Analytics Network*)
* **Pilot Location:** Cagayan de Oro City (80 Barangays, 2003–2022)

---

## 1. Topic Claiming Text (For Class Shared Sheet)

**Text for Shared Spreadsheet:**
```text
climate_atmosphere.csv + disease_lgu_disaggregated_totals.csv + google_open_buildings.csv + worldpop_population.csv + location.csv + brgy_geography.csv. Can 1-to-3 month weather lags (rainfall and heat index) combined with satellite building density and barangay boundary maps predict dengue outbreaks 30 to 60 days early across Cagayan de Oro's 80 barangays to help health workers deploy supplies and prevent hospital overcrowding?
```

---

## 2. Problem Scenario (2–3 Sentences)

In Cagayan de Oro City, dengue fever outbreaks happen every year, but city health workers usually react only after hospitals like Northern Mindanao Medical Center (NMMC) and J.R. Borja General Hospital (JRBGH) are already full of patients. Because it takes 4 to 8 weeks for mosquitoes to breed and for the dengue virus to develop, reacting after patients get sick is too late to stop the spread. 

By analyzing 20 years of weather data, satellite building footprints, and health records across CDO's 80 barangays, this project builds an early warning system that predicts dengue outbreaks **30 to 60 days in advance**, giving local officials enough time to clean up breeding sites and prepare hospital beds.

---

## 3. The Four Types of Data Analytics: Questions and Methods

Below are 8 clear analytical questions spanning the four stages of analytics, written in simple and practical terms.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             THE 4 TYPES OF DATA ANALYTICS                                   │
├───────────────────────────┬─────────────────────────────┬───────────────────────────────────┤
│ Analytics Type            │ Core Question               │ Practical Goal in this Study      │
├───────────────────────────┼─────────────────────────────┼───────────────────────────────────┤
│ 1. Descriptive Analytics  │ What happened?              │ Map historical case trends & hotspots│
│ 2. Diagnostic Analytics   │ Why did it happen?          │ Find weather lag delays & urban traps│
│ 3. Predictive Analytics   │ What will happen next?      │ Forecast outbreaks 30-60 days early  │
│ 4. Prescriptive Analytics │ What action should we take? │ Recommend 3-tier response actions    │
└───────────────────────────┴─────────────────────────────┴───────────────────────────────────┘
```

---

### Stage 1: Descriptive Analytics (What Happened in the Past?)

#### Question 1: What are the historical dengue patterns and seasonal peaks in Cagayan de Oro City over the past 20 years (2003–2022)?
* **Simple Explanation:** We want to look at the historical data to see which months of the year usually have the most dengue cases and if cases are increasing over time.
* **Methodology:** 
  * Group the monthly hospital records by year and month.
  * Plot a 20-year timeline showing total monthly cases alongside average monthly rainfall.
  * Calculate summary statistics (mean, median, and 75th percentile case numbers per barangay).
* **Key Finding:** Dengue cases in CDO rise sharply every year between **July and October**, during the rainy southwest monsoon (*Habagat*) season.

#### Question 2: Which specific barangays in Cagayan de Oro City have the highest number of dengue cases and deaths?
* **Simple Explanation:** We want to know if dengue cases are spread out evenly across the city or concentrated in a few specific neighborhoods.
* **Methodology:**
  * Aggregate 20-year case totals for each of CDO's 80 barangays.
  * Join the case data with spatial boundary polygons (`brgy_geography.csv`).
  * Create a colored risk map (choropleth map) highlighting top barangays.
* **Key Finding:** Over **52% of all historical dengue cases** are concentrated in dense urban lowlands near the river and coast: **Carmen, Lapasan, Balulang, Kauswagan, Bugo, and Puerto**.

---

### Stage 2: Diagnostic Analytics (Why Did Outbreaks Happen?)

#### Question 3: Why is there a 1-to-2 month delay between heavy rainfall and the spike in hospital admissions?
* **Simple Explanation:** Rain does not cause disease instantly. We need to measure how many weeks it takes for rainwater pools to produce adult mosquitoes and for infected mosquitoes to bite people.
* **Methodology:**
  * Compute lagged weather variables for 1 month, 2 months, 3 months, and 4 months in the past ($t-1, t-2, t-3, t-4$).
  * Run Pearson and Spearman correlation tests between past weather and current dengue cases.
* **Key Finding:** Peak correlation happens at **Lag-1 month and Lag-2 months** ($r = 0.46$ for rainfall, $r = 0.38$ for heat index). This matches the biological lifecycle: 7–10 days for mosquito larvae to grow, 5–14 days for the dengue virus to incubate inside the mosquito (Extrinsic Incubation Period), and 4–10 days for human symptoms to appear.

#### Question 4: Why do crowded urban barangays experience severe outbreaks while nearby rural barangays remain safe?
* **Simple Explanation:** In crowded urban neighborhoods, concrete and metal roofs prevent rainwater from soaking into the ground, creating artificial pools where mosquitoes breed. In rural areas, soil absorbs the water.
* **Methodology:**
  * Calculate building density and built-up area percentages from satellite data (`google_open_buildings.csv`).
  * Multiply building density with lagged rainfall to create a **Runoff Risk Index**.
  * Multiply building density with population density to create a **Host Exposure Index**.
* **Key Finding:** Barangays with over 60% built-up concrete cover (like Carmen and Lapasan) have 4 to 6 times higher outbreak risk after heavy rain than rural upland barangays (like Dansolihon and Besigan).

---

### Stage 3: Predictive Analytics (What Will Happen 30 to 60 Days from Now?)

#### Question 5: Can machine learning models accurately predict whether a barangay will suffer a dengue outbreak 30 and 60 days in advance?
* **Simple Explanation:** We want to train machine learning algorithms using only past data so they can tell health workers today whether an outbreak will occur next month or in two months.
* **Methodology:**
  * Define an "outbreak" as a month where cases exceed the historical 75th percentile for that barangay (with a minimum of 5 cases).
  * Use a strict **chronological split**: train models on 2003–2018 ($15,120$ samples) and test them on unseen 2019–2022 data ($3,760$ samples).
  * Calculate outbreak thresholds strictly from the pre-2019 training period to avoid future data leakage.
* **Key Finding:** Machine learning models achieve **0.960 to 0.964 ROC-AUC** on 30-day forecasts and **0.952 to 0.956 ROC-AUC** on 60-day forecasts on unseen test data.

#### Question 6: Which machine learning model is best suited for public health screening?
* **Simple Explanation:** We compared four standard data mining algorithms (Logistic Regression, Random Forest, LightGBM, and XGBoost) to see which one catches the most outbreaks.
* **Methodology:**
  * Train and tune all 4 models.
  * Optimize decision thresholds using the **$F_2$-Score**, which gives twice as much weight to Recall (catching true outbreaks) as Precision (avoiding false alarms).
* **Key Finding:**
  * **Logistic Regression** achieves **91.74% Recall** (catches over 9 out of 10 outbreaks) with **50.45% Precision** (a 10-fold improvement over the 5% baseline rate).
  * **LightGBM** achieves **72.34% Precision and 72.93% Recall with 93.07% Accuracy**, making it ideal when inspection staff is limited.

---

### Stage 4: Prescriptive Analytics (What Specific Actions Should We Take?)

#### Question 7: How can the predicted probabilities be converted into a clear 3-tier action plan for barangay health workers?
* **Simple Explanation:** Health workers need simple rules, not confusing numbers. We map model probabilities ($P$) into three clear alert levels with specific action checklists.
* **Methodology:**
  * Stratify predicted outbreak probability into three operational risk bands:
    * **Level 1 (Normal, $P < 0.30$):** Standard community cleanup and regular water container inspections.
    * **Level 2 (Pre-Emptive Alert, $0.30 \le P < 0.65$):** Apply safe biological larvicide (Bti) to standing water; send Barangay Health Workers (BHWs) for house-to-house fever monitoring.
    * **Level 3 (Critical Outbreak Warning, $P \ge 0.65$):** Schedule targeted spatial fogging within 48 hours; send 200 rapid test kits to local clinics; reserve hospital beds at NMMC and JRBGH.

#### Question 8: How should the City Health Office allocate limited field teams and hospital supplies across CDO's 80 barangays each month?
* **Simple Explanation:** Instead of spraying the whole city blindly, the system produces a ranked dispatch list showing which barangays need attention first.
* **Methodology:**
  * Rank all 80 barangays from highest to lowest predicted outbreak risk each month.
  * Automatically generate an LGU resource dispatch table showing population count, building density, predicted probability, and assigned alert tier.
* **Key Finding:** Using targeted ranking allows the city to protect **80% of at-risk residents by deploying teams to just the top 15 highest-risk barangays**, saving fuel, chemical supplies, and overtime costs.

---

## 4. Summary Table of the 4 Stages of Analytics

| Stage | Focus Question | Input Data Used | Technique / Model | Key Output / Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **1. Descriptive** | What happened in CDO? | Monthly dengue cases (`disease_lgu_disaggregated_totals.csv`), barangay maps (`brgy_geography.csv`) | Time-series aggregation, GIS choropleth mapping | 20-year baseline trend chart and barangay hotspot map. |
| **2. Diagnostic** | Why did outbreaks happen? | Daily ERA5 weather (`climate_atmosphere.csv`), satellite building density (`google_open_buildings.csv`) | Cross-correlation lag analysis, physical interaction formulas | Confirmation of 1-to-2 month biological delay ($r=0.46$) and urban runoff traps. |
| **3. Predictive** | What will happen in 30–60 days? | 59 engineered spatial, lag, and climate features | Logistic Regression, Random Forest, LightGBM, XGBoost ($F_2$-optimized) | 30-day and 60-day outbreak probability scores (>91% recall). |
| **4. Prescriptive** | What actions should we take? | Predicted outbreak probabilities | 3-tier decision matrix and ranked barangay dispatch table | Actionable monthly schedule for larviciding, test kit allocation, and hospital bed preparation. |

---

## 5. Data Mining Tools and Libraries Used

* **Programming Language:** Python 3.10 / 3.11 / 3.12
* **Data Processing & Feature Engineering:** `pandas`, `numpy`, `shapely` (for polygon border adjacency)
* **Machine Learning & Modeling:** `scikit-learn`, `lightgbm`, `xgboost`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Testing & Quality Assurance:** Python `unittest` (9 automated tests verifying zero data leakage, spatial matrix symmetry, and notebook execution)

---

## 6. Slide Outline for 10-Minute Presentation Defense (`LAB 1 - PPT`)

When presenting this study next Saturday, follow this 8-slide structure:

* **Slide 1: Title & Team Information**
  * Project Title: *Climate-Driven Vector-Borne Dengue Outbreak Surveillance for Cagayan de Oro City*
  * Course: *Data Mining and Applications (DMA)*
* **Slide 2: The Core Problem (Reactive vs. Proactive)**
  * The problem: Interventions happen after hospitals are full.
  * The solution: Predicting risk 30–60 days early using 20 years of climate, building, and health data.
* **Slide 3: Descriptive Analytics (What Happened?)**
  * 20-year trend showing seasonal surges from July to October.
  * Map showing over 52% of cases in Carmen, Lapasan, Kauswagan, and Balulang.
* **Slide 4: Diagnostic Analytics (Why Did It Happen?)**
  * Biological lag: 1-to-2 month delay between rain/heat spikes and hospital cases.
  * Urban factor: Concrete surfaces and crowded homes multiply mosquito breeding.
* **Slide 5: Spatial Contiguity & Neighbor Transmission ($W$ Matrix)**
  * How adjacent barangays spread risk across 428 shared borders.
* **Slide 6: Predictive Analytics (Machine Learning Tournament)**
  * Benchmark table showing Logistic Regression (91.7% Recall) and LightGBM (72.3% Precision, 93.1% Accuracy).
  * Why 50.5% precision is a 10x improvement over the 5% baseline rate.
* **Slide 7: Prescriptive Analytics (3-Tier LGU Decision Matrix)**
  * Level 1 (Normal), Level 2 (Larvicide & Community Check), Level 3 (Fogging & Hospital Prep).
* **Slide 8: Conclusion & Summary**
  * The value of moving from descriptive charts to automated prescriptive public health action.

---

## 7. Individual Contribution Essay Template (`LAB 1 - CONTRIBUTION`)
*(Word count: ~380 words — customize with your name and specific tasks)*

> During Laboratory Activity 1 for Data Mining and Applications (DMA), I contributed to formulating the research narrative and structuring the four types of data analytics using the Project CCHAIN dataset for Cagayan de Oro City. 
>
> For the descriptive and diagnostic analytics phases, I helped analyze the 20-year historical dataset covering 80 barangays from 2003 to 2022. I assisted in identifying the seasonal peaks that occur during the southwest monsoon season and examined the biological lag relationship showing why rainfall and heat anomalies create case surges 4 to 8 weeks later. I also reviewed how satellite building footprints from Google Open Buildings and population density from WorldPop explain why dense lowland barangays like Carmen, Lapasan, and Balulang face higher risks than rural upland areas.
>
> For the predictive modeling component, I helped formulate the multi-horizon forecasting setup comparing 30-day and 60-day early warning windows. I reviewed the performance of the four machine learning models (Logistic Regression, Random Forest, LightGBM, and XGBoost) and supported the use of $F_2$-score threshold optimization, which prioritizes catching true outbreaks to minimize missed public health emergencies. I also verified that the outbreak threshold was computed strictly on pre-2019 training data to prevent future target leakage.
>
> Finally, for the prescriptive analytics phase, I contributed to mapping model output probabilities into a practical 3-tier municipal decision matrix (Level 1 Normal, Level 2 Pre-emptive Alert, and Level 3 Outbreak Warning). This framework provides concrete guidelines for local health officials to distribute rapid test kits, deploy larvicide teams, and prepare hospital triage capacity before hospital surges occur. Through this activity, I gained practical experience in applying data mining concepts to solve real-world public health challenges.
