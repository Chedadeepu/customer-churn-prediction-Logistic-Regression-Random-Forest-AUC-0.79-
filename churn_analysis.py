"""
====================================================================
  Customer Churn Prediction & Analysis — IBM Telco Dataset
  Author  : Saiteja Chedadeepu
  GitHub  : https://github.com/Chedadeepu
====================================================================
  Dataset : IBM Telco Customer Churn (Kaggle — free download)
  URL     : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

  HOW TO RUN
    1.  pip install -r requirements.txt
    2a. Download dataset from Kaggle → save as  data/telco_churn.csv
        OR use the sample data already included in data/ to test first
    3.  python churn_analysis.py
    4.  Charts  → outputs/charts/
        Report  → outputs/churn_report.txt
        Scores  → outputs/churn_scores.csv
====================================================================
  Covers:
    ✓ Data cleaning & feature engineering
    ✓ EDA (churn by contract, tenure, charges, service)
    ✓ Logistic Regression + StandardScaler Pipeline
    ✓ Random Forest (comparison model)
    ✓ ROC/AUC, confusion matrix, classification report
    ✓ 5-fold cross-validation
    ✓ Feature importance
    ✓ Business recommendations
====================================================================
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, roc_curve, auc,
                                      ConfusionMatrixDisplay, roc_auc_score)
from sklearn.pipeline         import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ── Paths ─────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "telco_churn.csv")
CHART_DIR   = os.path.join("outputs", "charts")
OUTPUT_DIR  = "outputs"
os.makedirs(CHART_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"Yes": "#EF4444", "No": "#22C55E"}

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & FIRST LOOK
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 1 — Load & First Look")
print("="*62)

df = pd.read_csv(DATA_PATH)
print(f"\n  Raw shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Churn split :\n{df['Churn'].value_counts(normalize=True).mul(100).round(1).to_string()}")
print(f"\n  Sample rows:\n{df.head(3).to_string()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 2 — Data Cleaning & Feature Engineering")
print("="*62)

# Fix TotalCharges — loaded as string (whitespace for new customers)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
null_before = df.isnull().sum().sum()
df.dropna(inplace=True)
dup_count = df.duplicated().sum()
df.drop_duplicates(inplace=True)

print(f"\n  Null rows removed   : {null_before - df.isnull().sum().sum()}")
print(f"  Duplicate rows      : {dup_count}")
print(f"  Clean shape         : {df.shape[0]:,} rows")

# Binary target
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

# ── Feature Engineering ───────────────────────────────────────────
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr"],
    include_lowest=True
)
df["avg_monthly_spend"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)
df["has_support"] = (
    (df["OnlineSecurity"]  == "Yes").astype(int) +
    (df["TechSupport"]     == "Yes").astype(int) +
    (df["OnlineBackup"]    == "Yes").astype(int)
)
df["is_high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

print(f"\n  New features created: tenure_group, avg_monthly_spend, has_support, is_high_value")
print(f"  Churn rate          : {df['Churn_bin'].mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — EDA
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 3 — Exploratory Data Analysis")
print("="*62)

# ── Chart 1: Churn rate by key categorical features ───────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Churn Rate by Key Business Features", fontsize=15, fontweight="bold", y=1.01)

def churn_bar(ax, col, title, rotate=0):
    rates = (df.groupby(col)["Churn_bin"]
               .mean().mul(100)
               .sort_values(ascending=True)
               .reset_index())
    colors = ["#22C55E" if v < 20 else "#F59E0B" if v < 35 else "#EF4444"
              for v in rates["Churn_bin"]]
    bars = ax.barh(rates[col].astype(str), rates["Churn_bin"],
                   color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in rates["Churn_bin"]],
                 padding=4, fontsize=11)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, rates["Churn_bin"].max() * 1.28)

churn_bar(axes[0,0], "Contract",        "Churn by Contract Type")
churn_bar(axes[0,1], "InternetService", "Churn by Internet Service")
churn_bar(axes[1,0], "PaymentMethod",   "Churn by Payment Method")
churn_bar(axes[1,1], "tenure_group",    "Churn by Tenure Group")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_churn_by_category.png", dpi=150, bbox_inches="tight")
plt.close()

# Key insight
contract_rates = df.groupby("Contract")["Churn_bin"].mean().mul(100)
mtm_rate  = contract_rates.get("Month-to-month", contract_rates.max())
best_rate = contract_rates.min()
ratio     = mtm_rate / best_rate if best_rate > 0 else 0
print(f"\n  Contract churn rates:\n{contract_rates.round(1).to_string()}")
print(f"\n  → Month-to-month customers churn at {ratio:.1f}x the rate of long-term contracts")

# ── Chart 2: Charges & Tenure distributions ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Charges & Tenure: Churned vs Retained Customers",
             fontsize=13, fontweight="bold")

for ax, col, xlabel in zip(
    axes,
    ["tenure", "MonthlyCharges", "TotalCharges"],
    ["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"]
):
    for churn_val, color, label in [
        ("No",  "#22C55E", "Retained"),
        ("Yes", "#EF4444", "Churned")
    ]:
        data = df[df["Churn"] == churn_val][col]
        ax.hist(data, bins=35, alpha=0.55, color=color,
                label=f"{label} (μ={data.mean():.0f})", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(col.replace("_", " "))
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_charges_tenure.png", dpi=150)
plt.close()

# ── Chart 3: Correlation heatmap of numeric features ─────────────
num_cols = ["tenure","MonthlyCharges","TotalCharges",
            "avg_monthly_spend","has_support","Churn_bin"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={"size": 10})
ax.set_title("Correlation Matrix — Numeric Features vs Churn",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()

print("  ✓ Charts 1–3 saved")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 4 — Model Training")
print("="*62)

# ── Encode ────────────────────────────────────────────────────────
cat_cols = [
    "gender","SeniorCitizen","Partner","Dependents",
    "PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod"
]
num_features = ["tenure","MonthlyCharges","TotalCharges","avg_monthly_spend","has_support"]

df_model = df[cat_cols + num_features + ["Churn_bin"]].copy()
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model.drop("Churn_bin", axis=1)
y = df_model["Churn_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n  Features  : {X.shape[1]}")
print(f"  Train     : {len(X_train):,} rows  ({y_train.mean()*100:.1f}% churn)")
print(f"  Test      : {len(X_test):,} rows  ({y_test.mean()*100:.1f}% churn)")

# ── Model 1: Logistic Regression ─────────────────────────────────
print("\n  Training Logistic Regression...")
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42, C=1.0))
])
lr_pipe.fit(X_train, y_train)
lr_pred  = lr_pipe.predict(X_test)
lr_prob  = lr_pipe.predict_proba(X_test)[:, 1]
lr_acc   = accuracy_score(y_test, lr_pred) * 100
lr_auc   = roc_auc_score(y_test, lr_prob)
lr_cv    = cross_val_score(lr_pipe, X, y, cv=StratifiedKFold(5),
                           scoring="accuracy").mean() * 100

print(f"  LR Accuracy : {lr_acc:.2f}%  |  AUC : {lr_auc:.4f}  |  CV : {lr_cv:.2f}%")

# ── Model 2: Random Forest ────────────────────────────────────────
print("\n  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)
rf_prob  = rf_model.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_pred) * 100
rf_auc   = roc_auc_score(y_test, rf_prob)
rf_cv    = cross_val_score(rf_model, X, y, cv=StratifiedKFold(5),
                           scoring="accuracy").mean() * 100

print(f"  RF Accuracy : {rf_acc:.2f}%  |  AUC : {rf_auc:.4f}  |  CV : {rf_cv:.2f}%")

best_model_name = "Random Forest" if rf_auc >= lr_auc else "Logistic Regression"
best_auc        = max(rf_auc, lr_auc)
best_acc        = rf_acc if rf_auc >= lr_auc else lr_acc
best_pred       = rf_pred if rf_auc >= lr_auc else lr_pred
best_prob       = rf_prob if rf_auc >= lr_auc else lr_prob

print(f"\n  Best model  : {best_model_name}  (AUC = {best_auc:.4f})")

# ── Churn probability scores ──────────────────────────────────────
scores_df = X_test.copy()
scores_df["actual_churn"]      = y_test.values
scores_df["churn_probability"] = best_prob
scores_df["churn_predicted"]   = best_pred
scores_df = scores_df.sort_values("churn_probability", ascending=False)
scores_df.to_csv(f"{OUTPUT_DIR}/churn_scores.csv", index=False)
print(f"  Scores saved → outputs/churn_scores.csv")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — MODEL EVALUATION CHARTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  STEP 5 — Model Evaluation Charts")
print("="*62)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    f"Model Evaluation — {best_model_name}  "
    f"|  Accuracy: {best_acc:.1f}%  |  AUC: {best_auc:.4f}",
    fontsize=13, fontweight="bold")

# 5a. Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=["Retained","Churned"]).plot(
    ax=axes[0,0], colorbar=False, cmap="Blues")
axes[0,0].set_title(f"Confusion Matrix — {best_model_name}")

# 5b. ROC Curves — both models
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
axes[0,1].plot(fpr_lr, tpr_lr, color="#2563EB", lw=2.5,
               label=f"Logistic Reg (AUC={lr_auc:.3f})")
axes[0,1].plot(fpr_rf, tpr_rf, color="#7C3AED", lw=2.5, ls="--",
               label=f"Random Forest (AUC={rf_auc:.3f})")
axes[0,1].plot([0,1],[0,1], "k--", lw=1.2, alpha=0.5, label="Random")
axes[0,1].fill_between(fpr_rf, tpr_rf, alpha=0.07, color="#7C3AED")
axes[0,1].set_xlabel("False Positive Rate")
axes[0,1].set_ylabel("True Positive Rate")
axes[0,1].set_title("ROC Curve — Both Models")
axes[0,1].legend(fontsize=9)

# 5c. Feature importance (RF or LR coefficients)
if best_model_name == "Random Forest":
    fi = pd.DataFrame({
        "Feature":     X.columns,
        "Importance":  rf_model.feature_importances_
    }).sort_values("Importance", ascending=True).tail(12)
    axes[1,0].barh(fi["Feature"], fi["Importance"],
                   color="#7C3AED", edgecolor="white")
    axes[1,0].set_xlabel("Importance Score")
    axes[1,0].set_title("Random Forest — Top Feature Importances")
else:
    coefs = pd.DataFrame({
        "Feature":     X.columns,
        "Coefficient": lr_pipe.named_steps["model"].coef_[0]
    }).sort_values("Coefficient", key=abs, ascending=True).tail(12)
    colors = ["#EF4444" if c > 0 else "#22C55E" for c in coefs["Coefficient"]]
    axes[1,0].barh(coefs["Feature"], coefs["Coefficient"], color=colors)
    axes[1,0].axvline(0, color="black", lw=1.2, ls="--", alpha=0.5)
    axes[1,0].set_xlabel("Coefficient (→ churn risk)")
    axes[1,0].set_title("Logistic Reg — Feature Coefficients")

# 5d. Churn probability distribution
axes[1,1].hist(
    [best_prob[y_test==0], best_prob[y_test==1]],
    bins=35, label=["Retained","Churned"],
    color=["#22C55E","#EF4444"], alpha=0.65, edgecolor="white")
axes[1,1].axvline(0.50, color="black", lw=2, ls="--", label="0.5 threshold")
axes[1,1].set_xlabel("Predicted Churn Probability")
axes[1,1].set_ylabel("Count")
axes[1,1].set_title("Predicted Probability Distribution")
axes[1,1].legend()

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_model_evaluation.png", dpi=150)
plt.close()
print("  ✓ Chart saved: 04_model_evaluation.png")

# ── Chart 5: Classification Report heatmap ────────────────────────
cr_dict = classification_report(y_test, best_pred, output_dict=True,
                                target_names=["Retained","Churned"])
cr_df = pd.DataFrame(cr_dict).T.drop(["accuracy","macro avg","weighted avg"])
cr_df = cr_df[["precision","recall","f1-score","support"]].astype(float)

fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(cr_df.drop("support", axis=1), annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={"size": 13})
ax.set_title(f"Classification Report — {best_model_name}", fontsize=12, fontweight="bold")
ax.set_xticklabels(["Precision","Recall","F1-Score"], fontsize=11)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_classification_report.png", dpi=150)
plt.close()
print("  ✓ Chart saved: 05_classification_report.png")

# ── Chart 6: At-risk customer segments ────────────────────────────
df_test_full = df.loc[X_test.index].copy()
df_test_full["churn_probability"] = best_prob
df_test_full["risk_tier"] = pd.cut(
    best_prob,
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low risk (<30%)", "Medium risk (30-60%)", "High risk (>60%)"]
)
risk_summary = df_test_full.groupby("risk_tier").agg(
    Customers       = ("customerID", "count"),
    Avg_Monthly_Rev = ("MonthlyCharges", "mean"),
    Actual_Churn_Pct= ("Churn_bin", lambda x: x.mean()*100)
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("At-Risk Customer Segments", fontsize=13, fontweight="bold")

tier_colors = ["#22C55E","#F59E0B","#EF4444"]
for ax, col, ylabel in zip(
    axes,
    ["Customers","Avg_Monthly_Rev","Actual_Churn_Pct"],
    ["# Customers","Avg Monthly Revenue ($)","Actual Churn Rate (%)"]
):
    ax.bar(risk_summary["risk_tier"].astype(str),
           risk_summary[col], color=tier_colors, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=9)
    for i, v in enumerate(risk_summary[col]):
        ax.text(i, v * 1.02, f"{v:,.1f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_risk_segments.png", dpi=150)
plt.close()
print("  ✓ Chart saved: 06_risk_segments.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — BUSINESS REPORT
# ═══════════════════════════════════════════════════════════════════
cr_text = classification_report(y_test, best_pred,
                                target_names=["Retained","Churned"])

# Count risk tiers
high_risk = int((best_prob > 0.60).sum())
med_risk  = int(((best_prob >= 0.30) & (best_prob <= 0.60)).sum())
low_risk  = int((best_prob < 0.30).sum())

report = f"""
======================================================================
  CUSTOMER CHURN PREDICTION — ANALYSIS REPORT
  Author  : Saiteja Chedadeepu  |  github.com/Chedadeepu
  Dataset : IBM Telco Customer Churn
======================================================================

DATASET OVERVIEW
  Total customers : {len(df):,}
  Overall churn   : {df['Churn_bin'].mean()*100:.1f}%
  Training rows   : {len(X_train):,}
  Test rows       : {len(X_test):,}
  Features used   : {X.shape[1]}

MODEL COMPARISON
  ┌─────────────────────┬──────────┬─────────┬──────────┐
  │ Model               │ Accuracy │   AUC   │  5-Fold  │
  ├─────────────────────┼──────────┼─────────┼──────────┤
  │ Logistic Regression │ {lr_acc:>7.2f}% │ {lr_auc:.4f}  │ {lr_cv:>7.2f}% │
  │ Random Forest       │ {rf_acc:>7.2f}% │ {rf_auc:.4f}  │ {rf_cv:>7.2f}% │
  └─────────────────────┴──────────┴─────────┴──────────┘
  Best model : {best_model_name} (AUC = {best_auc:.4f})

CLASSIFICATION REPORT ({best_model_name})
{cr_text}

RISK SEGMENTATION (test set: {len(X_test):,} customers)
  High risk  (>60% churn prob) : {high_risk:,} customers
  Medium risk (30-60%)         : {med_risk:,} customers
  Low risk   (<30%)            : {low_risk:,} customers

KEY EDA FINDINGS
  1. Contract type is the #1 churn driver:
{contract_rates.round(1).to_string()}

  2. Short-tenure customers (0-1 yr) churn most — critical onboarding window
  3. Higher monthly charges correlate with increased churn risk
  4. Customers with 0 support services churn at significantly higher rates

BUSINESS RECOMMENDATIONS
  1. Convert Month-to-Month → Annual contracts
     Offer 10-15% discount for customers switching to annual billing.
     Target top 500 high-risk Month-to-Month customers first.

  2. 90-Day Onboarding Programme
     Customers who churn typically do so in first 12 months.
     Assign CSMs to tenure < 12 month accounts; trigger check-in
     calls at day 30, 60, 90.

  3. Bundle Add-On Services
     Customers with OnlineSecurity + TechSupport churn less.
     Offer free 90-day trial of support bundle to at-risk users.

  4. Monthly At-Risk Scoring
     Run this model monthly. Export high_risk tier (>60%) to CRM.
     Target with personalised retention offer within 48 hours.

  Outputs:
    outputs/churn_scores.csv    ← full probability scores per customer
    outputs/charts/             ← 6 charts
    outputs/churn_report.txt    ← this report
======================================================================
"""

with open("outputs/report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  ALL DONE — Key Results")
print("="*62)
print(f"""
  Best model    : {best_model_name}
  Accuracy      : {best_acc:.2f}%
  AUC           : {best_auc:.4f}
  5-Fold CV     : {max(lr_cv, rf_cv):.2f}%

  Risk tiers (test set = {len(X_test):,} customers)
  ──────────────────────────────────────────────────
  High risk (>60%) : {high_risk:,}  ← priority for retention campaigns
  Medium risk      : {med_risk:,}
  Low risk         : {low_risk:,}

  Outputs
  ──────────────────────────────────────────────────
  outputs/charts/01_churn_by_category.png
  outputs/charts/02_charges_tenure.png
  outputs/charts/03_correlation_heatmap.png
  outputs/charts/04_model_evaluation.png
  outputs/charts/05_classification_report.png
  outputs/charts/06_risk_segments.png
  outputs/churn_scores.csv
  outputs/churn_report.txt
""")
