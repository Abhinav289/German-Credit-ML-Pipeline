# Credit Risk ML Project — Master Context File
**Project:** German Credit Risk — Binary Classification  
**Stack:** Python, Pandas, Plotly, Scikit-learn, XGBoost, SHAP, Streamlit, MLflow  
**Goal:** End-to-end ML project for resume + GitHub, deployed as Streamlit app  

---

## Project Structure
credit-risk-project/
├── data/
│   ├── raw/                    # german.data — original, never touch
│   └── processed/              # cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb            # ✅ Done
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── explainability.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── models/
│   └── best_model.pkl
├── logs/
├── mlruns/
├── requirements.txt
├── .gitignore
└── README.md

---

## Dataset
- **Name:** German Credit Data (UCI)
- **File:** german.data — comma-separated, no header row
- **Rows:** 1,000
- **Features:** 20 input + 1 target
- **Target:** 1 = Good (700), 2 = Bad (300) — 70/30 split
- **Missing values:** None
- **Problem type:** Binary classification

### Column Names (in order):
```python
columns = [
    'checking_account', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment_since',
    'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_installments',
    'housing', 'existing_credits', 'job', 'liable_people',
    'telephone', 'foreign_worker', 'target'
]
```

### Feature Types:
- **Continuous:** duration, credit_amount, age
- **Ordinal numeric:** installment_rate, residence_since, 
  existing_credits, liable_people
- **Categorical (13):** all remaining features

---

## ML Design Patterns Applied
1. **Rebalancing** — SMOTE for 70/30 class imbalance
2. **Reframing** — probability calibration
3. **Checkpoints** — XGBoost early stopping
4. **Explainable Predictions** — SHAP
5. **Repeatable Splitting** — stratified train-test split

---

## ✅ Day 1 — EDA Complete

### Key Findings:

#### Continuous Features:
| Feature | Finding | Action Day 2 |
|---|---|---|
| duration | Right skewed, bad borrowers take longer loans (45–72 months) | Keep, use in interaction |
| credit_amount | 72 outliers (7.2%), long right tail, Amount ≠ Ability to Repay | Log transform |
| age | Bad peaks 24–25, Good peaks 26–27, younger = more default | Keep |

#### Categorical Features:
| Feature | Finding | Action Day 2 |
|---|---|---|
| checking_account | 66.8% absent/negative — most predictive feature | Keep |
| credit_history | critical/other lowest DR (17.1%) — counterintuitive | Keep |
| savings_status | <100 DM most common — low financial buffer | Keep |
| foreign_worker | 96.3% yes — near zero variance + fairness flag | Handle carefully |
| own_telephone | 3% gap — no discriminating power | Drop |
| job | 6.5% gap — categories too broad, signal diluted | Regroup |

#### Multivariate:
- credit_amount × duration correlation: 0.62 — multicollinearity
- All numeric correlations with target < 0.25 — not linearly separable
- <0 checking + no credits interaction = 76.92% default rate

### Default Borrower Profile:
1. checking_status < 0 or 0<=X<200
2. credit_history: all paid / no credits
3. other_parties: co-applicant
4. savings_status: <100 or <500 DM
5. property_magnitude: no known property
6. duration >= 45 months
7. age ~ 25–30

### Hypothesis Testing Results:
**Mann-Whitney U (numeric):**
- ✅ Significant: duration, credit_amount, age, installment_rate
- ❌ Not significant: residence_since, existing_credits, num_dependents

**Chi-Square (categorical):**
- ✅ Significant: checking_account, credit_history, savings_status,
  purpose, employment, personal_status, other_parties,
  property_magnitude, other_payment_plans, housing, foreign_worker
- ❌ Not significant: job, own_telephone

### Confirmed Drop Candidates:
```python
drop_candidates = [
    'residence_since',    # correlation = 0.00, p > 0.05
    'num_dependents',     # correlation = -0.00, p > 0.05
    'existing_credits',   # low variance, p > 0.05
    'own_telephone',      # 3% bivariate gap, p > 0.05
    'job'                 # 6.5% gap, categories too broad — regroup
]
```

---

## ⏳ Day 2 — Feature Engineering (Next)

### Action Plan:
| Action | Reason |
|---|---|
| Drop confirmed features | Not statistically significant |
| Log transform credit_amount | Right skew + 7.2% outliers |
| Engineer credit_amount/duration | Multicollinearity 0.62, monthly burden proxy |
| Engineer checking × credit_history | Interaction = 76.92% default rate |
| Engineer financial_stress_score | Combine checking_account + savings_status |
| Regroup job into 2 bins | stable vs unstable — recover diluted signal |
| Encode categoricals | Ordinal encoding for ordinal, WOE for high cardinality |
| Stratified train-test split | Preserve 70/30 class balance |
| SMOTE on training set only | Never on test set — leakage rule |
| Mutual Information scoring | Validate engineered features |
| Fairness audit | personal_status, foreign_worker |

---

## ⏳ Day 3 — Modeling
- Models: Logistic Regression (baseline) → Random Forest → XGBoost
- Validation: Stratified K-Fold CV
- Metrics: ROC-AUC, F1, Precision, Recall, Calibration
- Threshold optimization — justify beyond 0.5
- Calibration: reliability diagram + CalibratedClassifierCV

## ⏳ Day 4 — MLOps
- MLflow experiment tracking
- Pipeline: sklearn Pipeline object
- Evidently AI for data drift

## ⏳ Day 5 — Explainability
- SHAP values for feature importance
- SHAP waterfall for individual predictions
- Connect to default borrower profile from Day 1

## ⏳ Day 6 — Streamlit App
- Every input field has ⓘ tooltip
- Plain English explanation per field
- Trust by design philosophy
- Show SHAP explanation per prediction

## ⏳ Day 7 — GitHub + README Polish
- Export key plots to reports/figures/
- Write final README with default borrower profile
- Document ML design patterns used
- Clean notebook outputs

---

## Requirements.txt (current)
pandas
numpy
plotly
scipy
statsmodels
scikit-learn
nbformat
jupyter
ipykernel