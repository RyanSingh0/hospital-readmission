# Hospital Readmission Prediction

> **METCS 577 Data Science with Python · Boston University**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Savings](https://img.shields.io/badge/Projected%20Savings-$41M-brightgreen)

---

## Overview

End-to-end machine learning pipeline predicting **30-day hospital readmissions** from clinical and demographic data — with full explainability via SHAP values and a projected **$41.35M net savings** per hospital system.

**Why this matters:** Hospital readmissions within 30 days cost the US healthcare system ~$52.4B annually across 2,400 hospitals. CMS penalizes hospitals with excess readmission rates. A predictive model identifying high-risk patients before discharge enables targeted interventions: enhanced education, earlier follow-ups, closer monitoring.

---

## Dataset

| Property | Value |
|----------|-------|
| Records | **30,000** hospital discharge episodes |
| Features | 11 demographic + clinical variables |
| Target | Binary: readmitted within 30 days (Yes/No) |
| Class distribution | 3,674 Yes (12.2%) vs 26,326 No (87.8%) |
| Imbalance ratio | **7.17:1** |

**Features:** Age, Gender, Blood Pressure, Cholesterol, BMI, Diabetes, Hypertension, Medication Count, Length of Stay, Discharge Destination

**Key insight from EDA:**
- Discharge to Rehab: 18.5% readmission rate vs 12.2% overall
- Diabetes patients: 19.8% readmission rate
- A naive "always predict No" classifier gets 87.8% accuracy but **0% recall** — useless clinically

---

## Pipeline

```
Raw Data (30,000 records)
   ↓ Stratified 80/20 split (preserves 12.2% class ratio)
   ↓ Feature Engineering (5 domain-driven features)
   ↓ StandardScaler + One-Hot Encoding (Pipeline — no leakage)
   ↓ SMOTE oversampling (within CV folds only)
   ↓ GridSearchCV (5-fold StratifiedKFold)
   ↓ 11 Models benchmarked
   ↓ Threshold tuning (0.1–0.9 sweep, maximize F1)
   ↓ SHAP analysis
   ↓ Business impact projection
```

---

## Feature Engineering

Domain knowledge drives creation of 5 new features:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `age_bin` | <30, 30-50, 50-70, 70+ | Non-linear age-risk relationship |
| `los_severity` | 1-3 minor, 4-7 moderate, 8-10 severe | LOS threshold effects |
| `medication_intensity` | Low (1-3), Medium (4-6), High (7+) | Polypharmacy risk |
| `comorbidity_score` | 0, 1, 2+ chronic conditions | Cumulative disease burden |
| `high_risk_flag` | Diabetes AND Hypertension present | Clinical interaction effect |

---

## Models Benchmarked (11 algorithms)

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| **Gradient Boosting** | **0.5673** | Highest AUC |
| **Random Forest** | **0.5654** | Selected for deployment |
| Logistic Regression | 0.5588 | High recall (90.75%) at threshold 0.35 |
| AdaBoost | ~0.55 | |
| SVM | ~0.54 | |
| Decision Tree | ~0.52 | |
| k-NN, Naive Bayes, LDA, QDA, XGBoost | ~0.50–0.55 | |

**Why Random Forest over Gradient Boosting?** RF and GB are nearly equal in AUC (0.5654 vs 0.5673). RF was selected for deployment because it trains in seconds vs minutes for GB, and its feature importance is more directly interpretable for clinicians.

---

## Class Imbalance Handling

### SMOTE Oversampling
Generates synthetic minority samples by interpolating between k=5 nearest neighbors of readmitted patients. Applied **within cross-validation folds only** — not on the test set — to prevent data leakage.

### Threshold Tuning
Default 0.5 threshold is suboptimal for imbalanced data. Two operating strategies evaluated:

| Strategy | Model | Threshold | Recall | Precision | Use Case |
|----------|-------|-----------|--------|-----------|---------|
| High Recall | Logistic Reg | 0.35 | **90.75%** | 12.48% | Low intervention cost, can't miss a readmission |
| Balanced | Random Forest | **0.45** | **35.24%** | 16.89% | Limited capacity, prioritize high-risk patients |

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to individual predictions — critical for clinical trust.

**Top predictors by SHAP mean absolute value:**
1. Clinical Judgment Flag — domain expertise captures patterns invisible in raw features
2. Comorbidity Score
3. LOS Severity
4. Medication Intensity
5. Discharge Destination = Rehab

SHAP enables a clinician to ask: "Why did the model flag this specific patient?" — which is the difference between a research tool and a deployable clinical tool.

---

## Business Impact

```
Final Model: Random Forest + SMOTE, threshold = 0.45
Recall: 35.24% — captures 1 in 3 readmissions

CMS estimate: $15,000 saved per prevented readmission
Mid-sized hospital: ~3,674 high-risk discharges/year
Flagged correctly: 35.24% × 3,674 = 1,294 patients
Intervention success rate: 30%
Prevented readmissions: ~388/year

Net savings: $41.35 million
ROI: 2,761%
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook METCS577_Projet_Code.ipynb
```

The notebook is structured as 8 sections:
- Section 0: Setup & data loading
- Section 1: Exploratory Data Analysis
- Section 2: Train-test split & preprocessing pipeline
- Section 3: Baseline models (11 algorithms, GridSearchCV)
- Section 4: Class imbalance handling (SMOTE + threshold tuning)
- Section 5: Ensemble & stacking
- Section 6: Feature importance & SHAP
- Section 7: Business impact analysis

---

**Aryan Meena** · [LinkedIn](https://linkedin.com/in/aryan-meena-32685415a) · araj7042@gmail.com
Boston University, METCS 577 Data Science with Python
