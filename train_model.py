"""
Resume-Based Job Role Prediction — Model Training Pipeline
===========================================================
Uses resumes.csv (10K rows, 24 roles) + AI_Resume_Screening.csv (1K rows, 4 roles).
Prevents data leakage by stripping role names from text fields.
Features: TF-IDF on skills + experience bullets, plus structured numeric features.
Compares LightGBM, XGBoost, Random Forest; tunes the best; exports for production.
"""

import os
import re
import ast
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/swaraj-sah/Desktop/DataSet")
OUTPUT_DIR = Path(__file__).parent / "model_artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ─── 1. Load & Preprocess ────────────────────────────────────────────────────

def parse_list_field(val):
    """Parse stringified list like "['a','b','c']" into actual list."""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in str(val).split(",")]


def clean_text(text):
    """Basic text cleaning."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9\s/+#.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def strip_role_from_text(text, role):
    """Remove the role name from text to prevent data leakage."""
    if pd.isna(text) or pd.isna(role):
        return text
    # Remove exact role name and common variants
    role_lower = str(role).lower()
    text_lower = str(text).lower()
    # Remove role name from text (case-insensitive)
    text_cleaned = re.sub(re.escape(role_lower), "", text_lower)
    # Also remove common shortened forms
    for word in role_lower.split():
        if len(word) > 3:  # Only strip meaningful words, not "of", "and", etc.
            text_cleaned = re.sub(r'\b' + re.escape(word) + r'\b', '', text_cleaned)
    return re.sub(r'\s+', ' ', text_cleaned).strip()


def load_and_preprocess():
    """Load both datasets, clean, and merge into unified format."""

    # --- Dataset 1: resumes.csv (10K rows, 24 roles) ---
    df1 = pd.read_csv(DATA_DIR / "resumes.csv")
    print(f"Loaded resumes.csv: {df1.shape[0]} rows, {df1['role'].nunique()} roles")

    df1["skills_list"] = df1["skills"].apply(parse_list_field)
    df1["experience_list"] = df1["experience_bullets"].apply(parse_list_field)
    df1["skills_text"] = df1["skills_list"].apply(lambda x: " ".join(x).lower())
    df1["experience_text"] = df1["experience_list"].apply(lambda x: " ".join(x).lower())

    # CRITICAL: Strip role name from summary to prevent leakage
    df1["summary_clean"] = df1.apply(
        lambda r: strip_role_from_text(r["summary"], r["role"]), axis=1
    )

    df1["num_skills"] = df1["skills_list"].apply(len)
    df1["years_exp"] = pd.to_numeric(df1["years_experience"], errors="coerce").fillna(0)

    seniority_map = {"Junior": 0, "Mid": 1, "Senior": 2}
    education_map = {
        "High School": 0, "BA": 1, "BSc": 2, "BEng": 3,
        "MA": 4, "MSc": 5, "MBA": 6, "PhD": 7,
    }
    df1["seniority_num"] = df1["seniority"].map(seniority_map).fillna(0)
    df1["education_num"] = df1["education"].map(education_map).fillna(1)

    # Combined text: skills + sanitized summary + experience (NO raw role leak)
    df1["combined_text"] = (
        df1["skills_text"] + " " +
        df1["summary_clean"] + " " +
        df1["experience_text"]
    )
    df1["target"] = df1["role"]

    cols = ["combined_text", "skills_text", "num_skills", "years_exp",
            "seniority_num", "education_num", "target"]
    df1_clean = df1[cols].copy()

    # --- Dataset 2: AI_Resume_Screening.csv (1K rows, 4 roles) ---
    df2 = pd.read_csv(DATA_DIR / "AI_Resume_Screening.csv")
    print(f"Loaded AI_Resume_Screening.csv: {df2.shape[0]} rows, {df2['Job Role'].nunique()} roles")

    df2["skills_text"] = df2["Skills"].apply(clean_text)
    df2["certs_text"] = df2["Certifications"].fillna("").apply(clean_text)
    df2["combined_text"] = df2["skills_text"] + " " + df2["certs_text"]
    df2["num_skills"] = df2["Skills"].apply(lambda x: len(str(x).split(",")))
    df2["years_exp"] = pd.to_numeric(df2["Experience (Years)"], errors="coerce").fillna(0)

    edu_map2 = {"B.Sc": 2, "B.Tech": 3, "M.Tech": 5, "MBA": 6, "Ph.D": 7, "M.Sc": 5}
    df2["education_num"] = df2["Education"].map(edu_map2).fillna(2)
    df2["seniority_num"] = df2["years_exp"].apply(
        lambda y: 0 if y <= 2 else (1 if y <= 5 else 2)
    )
    df2["target"] = df2["Job Role"]

    df2_clean = df2[cols].copy()

    # --- Merge ---
    df = pd.concat([df1_clean, df2_clean], ignore_index=True)

    # Drop duplicates
    n_before = len(df)
    df.drop_duplicates(subset=["combined_text", "target"], inplace=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} duplicate rows")

    # Verify no leakage: check that role name doesn't appear in combined_text
    leakage_count = 0
    for _, row in df.sample(min(200, len(df)), random_state=42).iterrows():
        role_words = str(row["target"]).lower().split()
        text = row["combined_text"].lower()
        # Check if ALL words of role appear together
        if all(w in text for w in role_words) and len(role_words) > 1:
            leakage_count += 1
    print(f"Leakage spot-check: {leakage_count}/200 samples (should be ~0)")

    print(f"\nFinal dataset: {len(df)} rows, {df['target'].nunique()} roles")
    print(f"Class distribution:\n{df['target'].value_counts().to_string()}\n")

    return df


# ─── 2. Feature Engineering ──────────────────────────────────────────────────

def build_features(df):
    """Build TF-IDF + structured features."""

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["target"])

    # TF-IDF on combined text (skills + sanitized summary + experience)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(df["combined_text"])

    # TF-IDF on skills only (separate, strong signal)
    tfidf_skills = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_skills_tfidf = tfidf_skills.fit_transform(df["skills_text"])

    # Structured numeric features
    numeric_features = df[["years_exp", "num_skills", "seniority_num", "education_num"]].values
    X_numeric = csr_matrix(numeric_features)

    # Combine all
    X = hstack([X_tfidf, X_skills_tfidf, X_numeric])

    print(f"Feature matrix: {X.shape}")
    print(f"  Combined TF-IDF: {X_tfidf.shape[1]}, Skills TF-IDF: {X_skills_tfidf.shape[1]}, Numeric: 4")

    return X, y, label_enc, tfidf, tfidf_skills


# ─── 3. Train & Compare ──────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate and return metrics dict."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        avg_conf = float(np.mean(np.max(model.predict_proba(X_test), axis=1)))
    else:
        avg_conf = 0.0

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1-score:    {f1:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Avg Conf:    {avg_conf:.4f}")

    return {"name": model_name, "model": model,
            "accuracy": acc, "f1": f1, "precision": prec,
            "recall": rec, "avg_confidence": avg_conf}


def train_models(X_train, X_test, y_train, y_test):
    """Train LightGBM, XGBoost, and Random Forest."""
    results = []

    # 1. LightGBM
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.05,
        num_leaves=63, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    results.append(evaluate_model(lgb_model, X_test, y_test, "LightGBM"))

    # 2. XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss", verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))

    # 3. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=30,
        min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))

    return results


# ─── 4. Hyperparameter Tuning ────────────────────────────────────────────────

def tune_best_model(best_result, X_train, y_train, X_test, y_test):
    """Fine-tune the best model with cross-validation."""
    name = best_result["name"]
    print(f"\n{'='*60}")
    print(f"  Tuning {name} with cross-validation...")
    print(f"{'='*60}")

    configs = {
        "LightGBM": [
            {"cls": lgb.LGBMClassifier, "params": dict(
                n_estimators=800, max_depth=15, num_leaves=127, learning_rate=0.03,
                min_child_samples=5, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=0.1, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)},
            {"cls": lgb.LGBMClassifier, "params": dict(
                n_estimators=1000, max_depth=12, num_leaves=63, learning_rate=0.02,
                min_child_samples=8, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.05, reg_lambda=0.05, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)},
        ],
        "XGBoost": [
            {"cls": xgb.XGBClassifier, "params": dict(
                n_estimators=800, max_depth=12, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="mlogloss", verbosity=0)},
            {"cls": xgb.XGBClassifier, "params": dict(
                n_estimators=1000, max_depth=10, learning_rate=0.02,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.05, reg_lambda=0.5,
                random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="mlogloss", verbosity=0)},
        ],
        "Random Forest": [
            {"cls": RandomForestClassifier, "params": dict(
                n_estimators=500, max_depth=35, min_samples_split=3, min_samples_leaf=1,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)},
            {"cls": RandomForestClassifier, "params": dict(
                n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)},
        ],
    }

    best_score = 0
    best_tuned = None
    for cfg in configs.get(name, []):
        model = cfg["cls"](**cfg["params"])
        cv = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_weighted", n_jobs=-1)
        mean_cv = cv.mean()
        print(f"  CV F1={mean_cv:.4f} (std={cv.std():.4f})")
        if mean_cv > best_score:
            best_score = mean_cv
            best_tuned = model

    if best_tuned is None:
        return best_result

    best_tuned.fit(X_train, y_train)
    return evaluate_model(best_tuned, X_test, y_test, f"{name} (Tuned)")


# ─── 5. Overfitting Check ────────────────────────────────────────────────────

def check_overfitting(model, X_train, y_train, X_test, y_test, name):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc

    print(f"\n--- Overfitting Check: {name} ---")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Gap:            {gap:.4f}", end="")
    if gap > 0.10:
        print("  [WARNING: overfitting]")
    elif gap > 0.05:
        print("  [CAUTION: slight]")
    else:
        print("  [OK]")
    return gap


# ─── 6. Save & Inference ─────────────────────────────────────────────────────

SENIORITY_MAP = {"Junior": 0, "Mid": 1, "Senior": 2}
EDUCATION_MAP = {
    "High School": 0, "BA": 1, "BSc": 2, "B.Sc": 2, "BEng": 3, "B.Tech": 3,
    "MA": 4, "MSc": 5, "M.Sc": 5, "M.Tech": 5, "MBA": 6, "PhD": 7, "Ph.D": 7,
}


def save_model(model, label_enc, tfidf, tfidf_skills):
    """Save all artifacts needed for inference."""
    artifacts = {
        "model": model,
        "label_encoder": label_enc,
        "tfidf_combined": tfidf,
        "tfidf_skills": tfidf_skills,
        "seniority_map": SENIORITY_MAP,
        "education_map": EDUCATION_MAP,
        "classes": list(label_enc.classes_),
    }
    out_path = OUTPUT_DIR / "job_prediction_model.joblib"
    joblib.dump(artifacts, out_path, compress=3)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nModel saved to {out_path} ({size_mb:.1f} MB)")
    print(f"Classes ({len(label_enc.classes_)}): {list(label_enc.classes_)}")
    return out_path


def predict_role(resume_text, skills_text="", years_exp=0, seniority="Mid", education="BSc"):
    """
    Predict job role from resume text. Used by app.py at inference time.
    Returns list of dicts with 'role' and 'confidence' keys, sorted desc.
    """
    artifacts = joblib.load(OUTPUT_DIR / "job_prediction_model.joblib")
    model = artifacts["model"]
    label_enc = artifacts["label_encoder"]
    tfidf = artifacts["tfidf_combined"]
    tfidf_skills = artifacts["tfidf_skills"]

    combined = clean_text(resume_text + " " + skills_text)
    skills_clean = clean_text(skills_text)

    X_tfidf = tfidf.transform([combined])
    X_skills = tfidf_skills.transform([skills_clean])

    sen_num = artifacts["seniority_map"].get(seniority, 1)
    edu_num = artifacts["education_map"].get(education, 2)
    num_skills = len([s for s in skills_text.split(",") if s.strip()]) if skills_text else 0
    X_numeric = csr_matrix([[years_exp, num_skills, sen_num, edu_num]])

    X = hstack([X_tfidf, X_skills, X_numeric])
    proba = model.predict_proba(X)[0]
    top_indices = np.argsort(proba)[::-1][:5]

    return [
        {"role": label_enc.classes_[i], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top_indices
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Resume Job Role Prediction — Training Pipeline")
    print("=" * 60)

    # 1. Load & preprocess (with leakage prevention)
    df = load_and_preprocess()

    # 2. Feature engineering
    X, y, label_enc, tfidf, tfidf_skills = build_features(df)

    # 3. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 4. Train & compare models
    results = train_models(X_train, X_test, y_train, y_test)

    # 5. Select best
    best = max(results, key=lambda r: r["f1"])
    print(f"\n{'*'*60}")
    print(f"  BEST MODEL: {best['name']} (F1={best['f1']:.4f})")
    print(f"{'*'*60}")

    # 6. Tune best model
    tuned = tune_best_model(best, X_train, y_train, X_test, y_test)

    # Pick winner
    if tuned["f1"] >= best["f1"]:
        final_model, final_name = tuned["model"], tuned["name"]
    else:
        final_model, final_name = best["model"], best["name"]

    # 7. Overfitting check
    check_overfitting(final_model, X_train, y_train, X_test, y_test, final_name)

    # 8. Classification report
    y_pred = final_model.predict(X_test)
    print(f"\n{'='*60}")
    print(f"  Classification Report — {final_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

    # 9. Save model
    save_model(final_model, label_enc, tfidf, tfidf_skills)

    # 10. Sample predictions
    print(f"\n{'='*60}")
    print("  Sample Predictions")
    print(f"{'='*60}")
    tests = [
        ("Python, Django, Flask, REST APIs, SQL, PostgreSQL, Docker, Git",
         "Built scalable web applications and REST APIs"),
        ("TensorFlow, PyTorch, NLP, Deep Learning, Python, Machine Learning",
         "Developed neural network models for text classification"),
        ("Excel, Tableau, Power BI, SQL, Data Visualization, Pandas",
         "Created dashboards and analyzed business metrics"),
        ("React, JavaScript, HTML, CSS, TypeScript, Next.js, Tailwind",
         "Built responsive web interfaces and single page applications"),
        ("Ethical Hacking, Cybersecurity, Linux, Penetration Testing, SIEM",
         "Performed vulnerability assessments and security audits"),
    ]
    for skills, desc in tests:
        preds = predict_role(desc, skills_text=skills)
        top = preds[0]
        print(f"  Skills: {skills[:60]}...")
        print(f"  => {top['role']} ({top['confidence']}%)")
        print()

    print("Done! Model ready for production use.")


if __name__ == "__main__":
    main()
