# Customer Churn Prediction & Analysis

**Author:** Saiteja Chedadeepu | [GitHub](https://github.com/Chedadeepu)

End-to-end churn analysis on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Covers data cleaning, feature engineering, exploratory analysis, and two classification models (Logistic Regression + Random Forest) with full business interpretation.

> **Quick start:** A sample dataset is already included — just install requirements and run. Replace with the real Kaggle dataset for full results.

---

## Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Accuracy** | **81.22%** | 81.29% |
| **AUC** | **0.7927** ✅ | 0.7816 |
| **5-Fold CV** | **80.96%** | 81.25% |

**Best model:** Logistic Regression (highest AUC)

---

## Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Month-to-month customers churn at 19× the rate** of two-year contracts | Contract type EDA |
| 2 | **Short-tenure customers (0-1 yr)** are at highest churn risk | Tenure group analysis |
| 3 | **Higher monthly charges** correlate with increased churn probability | Correlation heatmap |
| 4 | Customers with **0 support services** churn significantly more | Feature importance |
| 5 | **~30% of test customers** fall in medium-to-high risk tier | Risk segmentation |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data cleaning | Python, Pandas, NumPy |
| EDA | Matplotlib, Seaborn, SciPy |
| ML Pipeline | Scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler, Pipeline) |
| Evaluation | ROC/AUC, Confusion Matrix, Classification Report, 5-Fold CV |
| Output | CSV scores, PNG charts, TXT report |

---

## Project Structure

```
customer-churn-prediction/
├── churn_analysis.py              ← Main script 
├── requirements.txt
├── README.md
└── telco_churn.csv            ← Sample data included OR download from Kaggle
└── outputs/
    ├── charts/
    │   ├── 01_churn_by_category.png    ← Churn rate by contract, internet, payment, tenure
    │   ├── 02_charges_tenure.png       ← Charge & tenure distributions by churn status
    │   ├── 03_correlation_heatmap.png  ← Numeric feature correlations
    │   ├── 04_model_evaluation.png     ← Confusion matrix, ROC curves, feature importance
    │   ├── 05_classification_report.png ← Precision/Recall/F1 heatmap
    │   └── 06_risk_segments.png        ← High/Medium/Low risk tier breakdown
    ├── churn_scores.csv               ← Predicted churn probability per customer
    └── churn_report.txt               ← Full business analysis report
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Chedadeepu/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run with included sample data (instant)
python churn_analysis.py

# 3b. OR download real dataset from Kaggle for full results:
#     https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#     Save as: data/telco_churn.csv → then run python churn_analysis.py
```

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `tenure_group` | Binned tenure into 4 groups (0-1yr, 1-2yr, 2-4yr, 4-6yr) |
| `avg_monthly_spend` | TotalCharges / tenure — true monthly burn rate |
| `has_support` | Count of support add-ons (OnlineSecurity + TechSupport + OnlineBackup) |
| `is_high_value` | Binary flag: MonthlyCharges above median |

---

## Business Recommendations

| Priority | Action | Target |
|----------|--------|--------|
| 1️⃣ | Offer incentive to switch Month-to-month → Annual | Top 500 high-risk M2M customers |
| 2️⃣ | 90-day onboarding with CSM check-ins at day 30, 60, 90 | tenure < 12 months |
| 3️⃣ | Free 90-day trial of support bundle (OnlineSecurity + TechSupport) | 0 add-ons customers |
| 4️⃣ | Monthly at-risk scoring → push high-risk list to CRM | Churn probability > 60% |

---

## Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Target:** `Churn` (Yes/No) — 26.5% churn rate
- **License:** Open Database License (ODbL)
