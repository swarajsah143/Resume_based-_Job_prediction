"""
Data Preprocessing & Model Training Pipeline
=============================================
Datasets:
  1. AI_Resume_Screening.csv      (1K)   -> Resume Score Prediction (regression)
  2. jobs.csv                     (2.5K) -> Job skill database (reference)
  3. resumes.csv                  (10K)  -> Job Role Prediction (classification)
  4. resume_dataset_200k_enhanced (200K) -> Hire Prediction (binary classification)
  5. postings.csv                 (3.3M) -> Sampled for enrichment

Models trained:
  A. Job Role Predictor      — predicts best job role from skills/experience
  B. Hire Predictor          — predicts hire probability from resume features
  C. Resume Score Predictor  — predicts AI score (0-100) for a resume
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = "/home/swaraj-sah/Desktop/DataSet"
PROJECT_DIR = "/home/swaraj-sah/Desktop/idp/Resume_based-_Job_prediction"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
CLEAN_DATA_DIR = os.path.join(PROJECT_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def safe_parse_list(val):
    """Safely parse string representations of lists."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [s.strip() for s in str(val).split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Load & Inspect Raw Data
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 1: Loading Raw Datasets")

df_screening = pd.read_csv(os.path.join(DATA_DIR, "AI_Resume_Screening.csv"))
df_jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
df_resumes = pd.read_csv(os.path.join(DATA_DIR, "resumes.csv"))
df_enhanced = pd.read_csv(os.path.join(DATA_DIR, "resume_dataset_200k_enhanced.csv"))

print(f"AI_Resume_Screening : {df_screening.shape}")
print(f"jobs                : {df_jobs.shape}")
print(f"resumes             : {df_resumes.shape}")
print(f"resume_200k_enhanced: {df_enhanced.shape}")

# Skip postings.csv (3.3M rows) — too large for in-memory processing,
# and the other datasets cover job matching needs well.

for name, df in [("AI_Resume_Screening", df_screening), ("jobs", df_jobs),
                 ("resumes", df_resumes), ("resume_200k_enhanced", df_enhanced)]:
    print(f"\n--- {name} ---")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}" if df.isnull().sum().any() else "  Nulls: None")
    print(f"  Dtypes: {dict(df.dtypes)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Preprocess AI_Resume_Screening (Resume Score + Hire Prediction)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 2: Preprocessing AI_Resume_Screening.csv")

df1 = df_screening.copy()

# Clean column names
df1.columns = df1.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

# Parse skills into lists
df1["skills_list"] = df1["Skills"].apply(safe_parse_list)
df1["skill_count"] = df1["skills_list"].apply(len)

# Encode categorical columns
le_education = LabelEncoder()
df1["education_encoded"] = le_education.fit_transform(df1["Education"].fillna("Unknown"))

le_role = LabelEncoder()
df1["role_encoded"] = le_role.fit_transform(df1["Job_Role"].fillna("Unknown"))

le_cert = LabelEncoder()
df1["cert_encoded"] = le_cert.fit_transform(df1["Certifications"].fillna("None"))

# Binary target for hire prediction
df1["hire_label"] = (df1["Recruiter_Decision"] == "Hire").astype(int)

# Extract individual skill features via MultiLabelBinarizer
all_skills_1 = df1["skills_list"].apply(lambda x: [s.strip().lower() for s in x])
mlb_screen = MultiLabelBinarizer()
skill_features_1 = pd.DataFrame(
    mlb_screen.fit_transform(all_skills_1),
    columns=[f"skill_{s}" for s in mlb_screen.classes_],
    index=df1.index
)

df1_features = pd.concat([
    df1[["Experience_Years", "skill_count", "education_encoded", "cert_encoded", "Projects_Count"]],
    skill_features_1
], axis=1)

# Handle missing values
df1_features = df1_features.fillna(0)

print(f"Cleaned shape: {df1_features.shape}")
print(f"Hire distribution:\n{df1['hire_label'].value_counts().to_string()}")
print(f"Score stats:\n{df1['AI_Score_0-100'].describe().to_string()}")
print(f"Skill features: {len(mlb_screen.classes_)} unique skills")

# Save cleaned data
df1_clean = pd.concat([df1_features, df1[["role_encoded", "hire_label", "AI_Score_0-100", "Job_Role"]]], axis=1)
df1_clean.to_csv(os.path.join(CLEAN_DATA_DIR, "screening_cleaned.csv"), index=False)
print("Saved: data/screening_cleaned.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Preprocess resumes.csv (Job Role Prediction)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 3: Preprocessing resumes.csv")

df2 = df_resumes.copy()

# Parse skills
df2["skills_list"] = df2["skills"].apply(safe_parse_list)
df2["skill_count"] = df2["skills_list"].apply(len)

# Encode targets and features
le_role2 = LabelEncoder()
df2["role_encoded"] = le_role2.fit_transform(df2["role"])

le_seniority2 = LabelEncoder()
df2["seniority_encoded"] = le_seniority2.fit_transform(df2["seniority"])

le_education2 = LabelEncoder()
df2["education_encoded"] = le_education2.fit_transform(df2["education"])

le_industry2 = LabelEncoder()
df2["industry_encoded"] = le_industry2.fit_transform(df2["industry"])

# Binarize skills
all_skills_2 = df2["skills_list"].apply(lambda x: [s.strip().lower() for s in x])
mlb_resume = MultiLabelBinarizer()
skill_features_2 = pd.DataFrame(
    mlb_resume.fit_transform(all_skills_2),
    columns=[f"skill_{s}" for s in mlb_resume.classes_],
    index=df2.index
)

df2_features = pd.concat([
    df2[["years_experience", "skill_count", "seniority_encoded", "education_encoded", "industry_encoded"]],
    skill_features_2
], axis=1)

df2_features = df2_features.fillna(0)

print(f"Cleaned shape: {df2_features.shape}")
print(f"Roles ({len(le_role2.classes_)}): {list(le_role2.classes_)}")
print(f"Skill features: {len(mlb_resume.classes_)} unique skills")

df2_clean = pd.concat([df2_features, df2[["role_encoded", "role"]]], axis=1)
df2_clean.to_csv(os.path.join(CLEAN_DATA_DIR, "resumes_cleaned.csv"), index=False)
print("Saved: data/resumes_cleaned.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Preprocess resume_dataset_200k_enhanced (Hire Prediction)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 4: Preprocessing resume_dataset_200k_enhanced.csv")

df3 = df_enhanced.copy()

# Encode categoricals
le_edu3 = LabelEncoder()
df3["education_encoded"] = le_edu3.fit_transform(df3["education_level"])

le_tier3 = LabelEncoder()
df3["tier_encoded"] = le_tier3.fit_transform(df3["university_tier"])

le_company3 = LabelEncoder()
df3["company_encoded"] = le_company3.fit_transform(df3["company_type"])

# Feature set
feature_cols_3 = [
    "age", "education_encoded", "tier_encoded", "cgpa",
    "internships", "projects", "programming_languages",
    "certifications", "experience_years", "hackathons",
    "research_papers", "skills_score", "soft_skills_score",
    "resume_length_words", "company_encoded"
]

df3_features = df3[feature_cols_3].fillna(0)

# Check class balance
print(f"Cleaned shape: {df3_features.shape}")
print(f"Hire distribution:\n{df3['hired'].value_counts().to_string()}")
print(f"\nFeature stats:")
print(df3_features.describe().round(2).to_string())

# Detect and report outliers
for col in ["cgpa", "experience_years", "age"]:
    q1, q3 = df3_features[col].quantile(0.25), df3_features[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df3_features[col] < q1 - 1.5 * iqr) | (df3_features[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col}: {outliers} outliers ({outliers/len(df3_features)*100:.1f}%)")

# Cap outliers for experience_years and age
for col in ["experience_years", "age"]:
    q1, q99 = df3_features[col].quantile(0.01), df3_features[col].quantile(0.99)
    df3_features[col] = df3_features[col].clip(q1, q99)

df3_clean = pd.concat([df3_features, df3[["hired"]]], axis=1)
df3_clean.to_csv(os.path.join(CLEAN_DATA_DIR, "enhanced_cleaned.csv"), index=False)
print("Saved: data/enhanced_cleaned.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Preprocess jobs.csv (Job Skill Reference Database)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 5: Preprocessing jobs.csv")

df4 = df_jobs.copy()
df4["must_have_skills"] = df4["must_have_skills"].apply(safe_parse_list)
df4["nice_to_have_skills"] = df4["nice_to_have_skills"].apply(safe_parse_list)
df4["all_skills"] = df4["must_have_skills"] + df4["nice_to_have_skills"]
df4["skill_count"] = df4["all_skills"].apply(len)

le_seniority4 = LabelEncoder()
df4["seniority_encoded"] = le_seniority4.fit_transform(df4["seniority"])

le_industry4 = LabelEncoder()
df4["industry_encoded"] = le_industry4.fit_transform(df4["industry"])

# Build a job-title -> required skills mapping for the app
job_skill_map = df4.groupby("job_title")["must_have_skills"].apply(
    lambda x: list(set(s.lower() for skills in x for s in skills))
).to_dict()

print(f"Unique job titles: {df4['job_title'].nunique()}")
print(f"Unique industries: {df4['industry'].nunique()}")
print(f"Sample job-skill mapping:")
for title, skills in list(job_skill_map.items())[:5]:
    print(f"  {title}: {skills[:6]}...")

joblib.dump(job_skill_map, os.path.join(MODEL_DIR, "job_skill_map.joblib"))
df4.to_csv(os.path.join(CLEAN_DATA_DIR, "jobs_cleaned.csv"), index=False)
print("Saved: data/jobs_cleaned.csv, models/job_skill_map.joblib")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Train Model A — Job Role Predictor (resumes.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 6: Training Model A — Job Role Predictor")

X_role = df2_features
y_role = df2["role_encoded"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_role, y_role, test_size=0.2, random_state=42, stratify=y_role
)

# Scale features
scaler_role = StandardScaler()
X_train_r_scaled = scaler_role.fit_transform(X_train_r)
X_test_r_scaled = scaler_role.transform(X_test_r)

# Try multiple models
models_role = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
}

best_role_model = None
best_role_score = 0

for name, model in models_role.items():
    model.fit(X_train_r_scaled, y_train_r)
    y_pred = model.predict(X_test_r_scaled)
    acc = accuracy_score(y_test_r, y_pred)
    f1 = f1_score(y_test_r, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X_train_r_scaled, y_train_r, cv=5, scoring="accuracy")
    print(f"  {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print(f"    CV Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    if f1 > best_role_score:
        best_role_score = f1
        best_role_model = (name, model)

print(f"\n  Best model: {best_role_model[0]} (F1={best_role_score:.4f})")

# Save
joblib.dump(best_role_model[1], os.path.join(MODEL_DIR, "role_predictor.joblib"))
joblib.dump(scaler_role, os.path.join(MODEL_DIR, "role_scaler.joblib"))
joblib.dump(le_role2, os.path.join(MODEL_DIR, "role_label_encoder.joblib"))
joblib.dump(mlb_resume, os.path.join(MODEL_DIR, "role_skill_binarizer.joblib"))
joblib.dump(le_seniority2, os.path.join(MODEL_DIR, "role_seniority_encoder.joblib"))
joblib.dump(le_education2, os.path.join(MODEL_DIR, "role_education_encoder.joblib"))
joblib.dump(le_industry2, os.path.join(MODEL_DIR, "role_industry_encoder.joblib"))
joblib.dump(list(X_role.columns), os.path.join(MODEL_DIR, "role_feature_names.joblib"))

# Print classification report for best model
y_pred_best = best_role_model[1].predict(X_test_r_scaled)
print(f"\nClassification Report ({best_role_model[0]}):")
print(classification_report(y_test_r, y_pred_best, target_names=le_role2.classes_))


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Train Model B — Hire Predictor (resume_dataset_200k_enhanced)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 7: Training Model B — Hire Predictor")

X_hire = df3_features
y_hire = df3["hired"]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_hire, y_hire, test_size=0.2, random_state=42, stratify=y_hire
)

scaler_hire = StandardScaler()
X_train_h_scaled = scaler_hire.fit_transform(X_train_h)
X_test_h_scaled = scaler_hire.transform(X_test_h)

models_hire = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
}

best_hire_model = None
best_hire_score = 0

for name, model in models_hire.items():
    model.fit(X_train_h_scaled, y_train_h)
    y_pred = model.predict(X_test_h_scaled)
    y_prob = model.predict_proba(X_test_h_scaled)[:, 1]
    acc = accuracy_score(y_test_h, y_pred)
    f1 = f1_score(y_test_h, y_pred)
    auc = roc_auc_score(y_test_h, y_prob)
    print(f"  {name}:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    AUC-ROC:  {auc:.4f}")

    if auc > best_hire_score:
        best_hire_score = auc
        best_hire_model = (name, model)

print(f"\n  Best model: {best_hire_model[0]} (AUC={best_hire_score:.4f})")

# Feature importance for best model
if hasattr(best_hire_model[1], "feature_importances_"):
    importances = pd.Series(best_hire_model[1].feature_importances_, index=feature_cols_3)
    print(f"\n  Top 10 Feature Importances:")
    for feat, imp in importances.sort_values(ascending=False).head(10).items():
        print(f"    {feat}: {imp:.4f}")

# Save
joblib.dump(best_hire_model[1], os.path.join(MODEL_DIR, "hire_predictor.joblib"))
joblib.dump(scaler_hire, os.path.join(MODEL_DIR, "hire_scaler.joblib"))
joblib.dump(le_edu3, os.path.join(MODEL_DIR, "hire_education_encoder.joblib"))
joblib.dump(le_tier3, os.path.join(MODEL_DIR, "hire_tier_encoder.joblib"))
joblib.dump(le_company3, os.path.join(MODEL_DIR, "hire_company_encoder.joblib"))
joblib.dump(feature_cols_3, os.path.join(MODEL_DIR, "hire_feature_names.joblib"))

# Confusion matrix
y_pred_best_h = best_hire_model[1].predict(X_test_h_scaled)
cm = confusion_matrix(y_test_h, y_pred_best_h)
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
print(f"    FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")
print(f"\n{classification_report(y_test_h, y_pred_best_h, target_names=['Reject', 'Hire'])}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8: Train Model C — Resume Score Predictor (AI_Resume_Screening)
# ═══════════════════════════════════════════════════════════════════════════════
print_section("STEP 8: Training Model C — Resume Score Predictor")

X_score = df1_features
y_score = df1["AI_Score_0-100"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_score, y_score, test_size=0.2, random_state=42
)

scaler_score = StandardScaler()
X_train_s_scaled = scaler_score.fit_transform(X_train_s)
X_test_s_scaled = scaler_score.transform(X_test_s)

models_score = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
}

best_score_model = None
best_score_r2 = -999

for name, model in models_score.items():
    model.fit(X_train_s_scaled, y_train_s)
    y_pred = model.predict(X_test_s_scaled)
    mae = mean_absolute_error(y_test_s, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_s, y_pred))
    r2 = r2_score(y_test_s, y_pred)
    print(f"  {name}:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R²:   {r2:.4f}")

    if r2 > best_score_r2:
        best_score_r2 = r2
        best_score_model = (name, model)

print(f"\n  Best model: {best_score_model[0]} (R²={best_score_r2:.4f})")

# Feature importance
if hasattr(best_score_model[1], "feature_importances_"):
    importances_s = pd.Series(best_score_model[1].feature_importances_, index=X_score.columns)
    print(f"\n  Top 10 Feature Importances:")
    for feat, imp in importances_s.sort_values(ascending=False).head(10).items():
        print(f"    {feat}: {imp:.4f}")

# Save
joblib.dump(best_score_model[1], os.path.join(MODEL_DIR, "score_predictor.joblib"))
joblib.dump(scaler_score, os.path.join(MODEL_DIR, "score_scaler.joblib"))
joblib.dump(le_education, os.path.join(MODEL_DIR, "score_education_encoder.joblib"))
joblib.dump(le_cert, os.path.join(MODEL_DIR, "score_cert_encoder.joblib"))
joblib.dump(mlb_screen, os.path.join(MODEL_DIR, "score_skill_binarizer.joblib"))
joblib.dump(list(X_score.columns), os.path.join(MODEL_DIR, "score_feature_names.joblib"))


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 9: Summary
# ═══════════════════════════════════════════════════════════════════════════════
print_section("PIPELINE COMPLETE — Summary")

print("Cleaned Data:")
for f in sorted(os.listdir(CLEAN_DATA_DIR)):
    size = os.path.getsize(os.path.join(CLEAN_DATA_DIR, f)) / 1024
    print(f"  data/{f}  ({size:.1f} KB)")

print("\nTrained Models:")
for f in sorted(os.listdir(MODEL_DIR)):
    size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
    print(f"  models/{f}  ({size:.1f} KB)")

print(f"""
Model Performance Summary:
  A. Job Role Predictor      — F1 (weighted): {best_role_score:.4f}
  B. Hire Predictor          — AUC-ROC:       {best_hire_score:.4f}
  C. Resume Score Predictor  — R²:            {best_score_r2:.4f}

Next steps:
  - Integrate models into app.py via the saved .joblib files
  - Use role_predictor for ML-based job suggestions
  - Use hire_predictor for hire probability scoring
  - Use score_predictor for AI-based resume scoring
""")
