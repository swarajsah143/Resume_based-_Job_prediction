"""
Data Preprocessing Pipeline for Resume-Based Job Prediction
Cleans, normalizes, and merges 4 datasets for ML model training.
"""

import os
import re
import ast
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/swaraj-sah/Desktop/DataSet"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Normalization Maps ──────────────────────────────────────────────────────

SKILL_SYNONYMS = {
    "js": "javascript", "py": "python", "react.js": "react", "reactjs": "react",
    "node": "node.js", "nodejs": "node.js", "tf": "tensorflow", "sklearn": "scikit-learn",
    "sk-learn": "scikit-learn", "postgres": "postgresql", "mongo": "mongodb",
    "k8s": "kubernetes", "gcloud": "gcp", "google cloud": "gcp",
    "amazon web services": "aws", "microsoft azure": "azure",
    "dl": "deep learning", "ml": "machine learning", "ai": "artificial intelligence",
    "c sharp": "c#", "csharp": "c#", "golang": "go", "r lang": "r",
    "vue.js": "vue", "vuejs": "vue", "angular.js": "angular", "angularjs": "angular",
    "express.js": "express", "expressjs": "express", "next": "next.js",
    "nuxt": "nuxt.js", "sveltejs": "svelte", "sass/scss": "sass",
    "ci cd": "ci/cd", "cicd": "ci/cd", "continuous integration": "ci/cd",
    "ms excel": "excel", "microsoft excel": "excel",
    "power bi": "powerbi", "ms power bi": "powerbi",
    "ethical hacking": "cybersecurity", "pen testing": "penetration testing",
    "nlp": "natural language processing",
}

EDUCATION_MAP = {
    "phd": "PhD", "ph.d": "PhD", "ph.d.": "PhD", "doctorate": "PhD",
    "master": "Masters", "masters": "Masters", "m.tech": "Masters", "mtech": "Masters",
    "m.s": "Masters", "m.s.": "Masters", "m.sc": "Masters", "msc": "Masters",
    "mba": "Masters", "m.e": "Masters", "me": "Masters", "mca": "Masters",
    "m.com": "Masters",
    "bachelor": "Bachelors", "bachelors": "Bachelors", "b.tech": "Bachelors",
    "btech": "Bachelors", "b.e": "Bachelors", "b.sc": "Bachelors", "bsc": "Bachelors",
    "bca": "Bachelors", "b.com": "Bachelors", "b.a": "Bachelors", "ba": "Bachelors",
    "diploma": "Diploma", "associate": "Diploma",
    "high school": "HighSchool", "12th": "HighSchool", "hsc": "HighSchool",
}

EDUCATION_RANK = {"Unknown": 0, "HighSchool": 1, "Diploma": 2, "Bachelors": 3, "Masters": 4, "PhD": 5}

SENIORITY_MAP = {
    "entry": "Entry", "entry-level": "Entry", "entry level": "Entry",
    "junior": "Junior", "jr": "Junior",
    "mid": "Mid", "mid-level": "Mid", "mid level": "Mid", "associate": "Mid",
    "senior": "Senior", "sr": "Senior", "lead": "Senior",
    "executive": "Executive", "director": "Executive", "vp": "Executive",
    "c-level": "Executive", "principal": "Executive",
}

SENIORITY_RANK = {"Unknown": 0, "Entry": 1, "Junior": 2, "Mid": 3, "Senior": 4, "Executive": 5}


# ─── Utility Functions ───────────────────────────────────────────────────────

def safe_parse_list(val):
    """Parse a string representation of a Python list into an actual list."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or not isinstance(val, str) or val.strip() == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        return [s.strip() for s in val.split(",") if s.strip()]


def normalize_skill(skill):
    """Normalize a single skill string."""
    s = skill.lower().strip().rstrip(".,;:")
    s = re.sub(r"[^\w\s./#+-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return SKILL_SYNONYMS.get(s, s)


def normalize_skills(skill_list):
    """Normalize a list of skills, removing duplicates."""
    normalized = []
    seen = set()
    for s in skill_list:
        ns = normalize_skill(s)
        if ns and ns not in seen:
            seen.add(ns)
            normalized.append(ns)
    return normalized


def normalize_education(edu_str):
    """Map education string to canonical level."""
    if pd.isna(edu_str):
        return "Unknown"
    edu_lower = edu_str.lower().strip()
    for key, val in EDUCATION_MAP.items():
        if key in edu_lower:
            return val
    return "Unknown"


def normalize_seniority(sen_str):
    """Map seniority string to canonical level."""
    if pd.isna(sen_str):
        return "Unknown"
    sen_lower = sen_str.lower().strip()
    for key, val in SENIORITY_MAP.items():
        if key in sen_lower:
            return val
    return "Unknown"


def clean_column_names(df):
    """Standardize column names: lowercase, underscores, no special chars."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[(\)$]", "", regex=True)
        .str.replace(r"[\s-]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def normalize_salary_annual(row):
    """Normalize salary to annual based on pay_period."""
    salary = row.get("med_salary")
    if pd.isna(salary):
        salary = np.nanmean([row.get("min_salary", np.nan), row.get("max_salary", np.nan)])
    if pd.isna(salary):
        return np.nan
    period = str(row.get("pay_period", "")).lower()
    if "hour" in period:
        return salary * 2080
    elif "month" in period:
        return salary * 12
    elif "week" in period:
        return salary * 52
    return salary  # assume yearly


# ─── Step 1: Load Raw Data ───────────────────────────────────────────────────

def load_datasets():
    print("=" * 70)
    print("STEP 1: Loading Raw Datasets")
    print("=" * 70)

    screening = pd.read_csv(os.path.join(DATA_DIR, "AI_Resume_Screening.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    resumes = pd.read_csv(os.path.join(DATA_DIR, "resumes.csv"))

    postings_cols = [
        "job_id", "company_name", "title", "description", "max_salary", "min_salary",
        "med_salary", "pay_period", "location", "formatted_work_type",
        "formatted_experience_level", "skills_desc", "remote_allowed",
    ]
    postings = pd.read_csv(os.path.join(DATA_DIR, "postings.csv"), usecols=postings_cols)

    for name, df in [("screening", screening), ("jobs", jobs), ("resumes", resumes), ("postings", postings)]:
        print(f"  {name:12s} → {df.shape[0]:>7,} rows × {df.shape[1]:>2} cols | nulls: {df.isnull().sum().sum():,}")

    return screening, jobs, resumes, postings


# ─── Step 2: Clean AI_Resume_Screening ────────────────────────────────────────

def clean_screening(df):
    print("\n" + "=" * 70)
    print("STEP 2: Cleaning AI_Resume_Screening (1,000 rows)")
    print("=" * 70)

    df = clean_column_names(df.copy())

    # Parse and normalize skills
    df["skills_raw"] = df["skills"].apply(safe_parse_list)
    df["skills_clean"] = df["skills_raw"].apply(normalize_skills)
    df["skill_count"] = df["skills_clean"].apply(len)

    # Normalize education
    df["education_level"] = df["education"].apply(normalize_education)
    df["education_rank"] = df["education_level"].map(EDUCATION_RANK).fillna(0).astype(int)

    # Certifications
    df["certifications"] = df["certifications"].fillna("None")
    df["has_certification"] = (df["certifications"] != "None").astype(int)

    # Normalize job role
    df["job_role_clean"] = df["job_role"].str.strip().str.title()

    # Binary label for recruiter decision
    df["hire_label"] = (df["recruiter_decision"].str.lower() == "hire").astype(int)

    # Select and rename final columns
    result = df[[
        "resume_id", "name", "skills_clean", "skill_count",
        "experience_years", "education_level", "education_rank",
        "certifications", "has_certification",
        "job_role_clean", "recruiter_decision", "hire_label",
        "salary_expectation", "projects_count", "ai_score_0_100",
    ]].copy()
    result.rename(columns={
        "job_role_clean": "job_role",
        "ai_score_0_100": "ai_score",
        "salary_expectation": "salary_expected",
    }, inplace=True)

    print(f"  Cleaned shape: {result.shape}")
    print(f"  Unique roles: {result['job_role'].nunique()}")
    print(f"  Hire/Reject: {result['hire_label'].value_counts().to_dict()}")
    print(f"  Education dist: {result['education_level'].value_counts().to_dict()}")

    return result


# ─── Step 3: Clean Resumes ────────────────────────────────────────────────────

def clean_resumes(df):
    print("\n" + "=" * 70)
    print("STEP 3: Cleaning Resumes (10,000 rows)")
    print("=" * 70)

    df = clean_column_names(df.copy())

    # Parse and normalize skills
    df["skills_raw"] = df["skills"].apply(safe_parse_list)
    df["skills_clean"] = df["skills_raw"].apply(normalize_skills)
    df["skill_count"] = df["skills_clean"].apply(len)

    # Normalize education
    df["education_level"] = df["education"].apply(normalize_education)
    df["education_rank"] = df["education_level"].map(EDUCATION_RANK).fillna(0).astype(int)

    # Normalize seniority
    df["seniority_clean"] = df["seniority"].apply(normalize_seniority)
    df["seniority_rank"] = df["seniority_clean"].map(SENIORITY_RANK).fillna(0).astype(int)

    # Normalize role and industry
    df["role_clean"] = df["role"].str.strip().str.title()
    df["industry_clean"] = df["industry"].str.strip().str.title()

    # Text features from summary and bullets
    df["summary_word_count"] = df["summary"].fillna("").apply(lambda x: len(x.split()))
    bullets = df["experience_bullets"].apply(safe_parse_list)
    df["bullet_count"] = bullets.apply(len)
    df["has_quantified_results"] = bullets.apply(
        lambda bl: int(any(bool(re.search(r"\d+%?", b)) for b in bl))
    )

    result = df[[
        "resume_id", "role_clean", "seniority_clean", "seniority_rank",
        "years_experience", "industry_clean", "education_level", "education_rank",
        "skills_clean", "skill_count",
        "summary_word_count", "bullet_count", "has_quantified_results",
    ]].copy()
    result.rename(columns={
        "role_clean": "role",
        "seniority_clean": "seniority",
        "industry_clean": "industry",
    }, inplace=True)

    print(f"  Cleaned shape: {result.shape}")
    print(f"  Unique roles: {result['role'].nunique()}")
    print(f"  Seniority dist: {result['seniority'].value_counts().to_dict()}")

    return result


# ─── Step 4: Clean Jobs ──────────────────────────────────────────────────────

def clean_jobs(df):
    print("\n" + "=" * 70)
    print("STEP 4: Cleaning Jobs (2,500 rows)")
    print("=" * 70)

    df = clean_column_names(df.copy())

    # Parse skills
    df["must_have_list"] = df["must_have_skills"].apply(safe_parse_list).apply(normalize_skills)
    df["nice_to_have_list"] = df["nice_to_have_skills"].apply(safe_parse_list).apply(normalize_skills)
    df["all_skills"] = df.apply(
        lambda r: list(set(r["must_have_list"] + r["nice_to_have_list"])), axis=1
    )
    df["must_have_count"] = df["must_have_list"].apply(len)
    df["nice_to_have_count"] = df["nice_to_have_list"].apply(len)
    df["total_skill_count"] = df["all_skills"].apply(len)

    # Normalize seniority and industry
    df["seniority_clean"] = df["seniority"].apply(normalize_seniority)
    df["seniority_rank"] = df["seniority_clean"].map(SENIORITY_RANK).fillna(0).astype(int)
    df["industry_clean"] = df["industry"].str.strip().str.title()
    df["job_title_clean"] = df["job_title"].str.strip().str.title()

    # Text features
    df["description_word_count"] = df["description"].fillna("").apply(lambda x: len(x.split()))

    result = df[[
        "job_id", "job_title_clean", "seniority_clean", "seniority_rank",
        "industry_clean", "must_have_list", "nice_to_have_list", "all_skills",
        "must_have_count", "nice_to_have_count", "total_skill_count",
        "description", "description_word_count",
    ]].copy()
    result.rename(columns={
        "job_title_clean": "job_title",
        "seniority_clean": "seniority",
        "industry_clean": "industry",
    }, inplace=True)

    print(f"  Cleaned shape: {result.shape}")
    print(f"  Unique titles: {result['job_title'].nunique()}")
    print(f"  Industry dist: {result['industry'].value_counts().head(5).to_dict()}")

    return result


# ─── Step 5: Clean Postings ──────────────────────────────────────────────────

def clean_postings(df):
    print("\n" + "=" * 70)
    print("STEP 5: Cleaning Postings (123,849 rows)")
    print("=" * 70)

    df = clean_column_names(df.copy())

    # Drop rows with no title AND no description
    before = len(df)
    df = df.dropna(subset=["title", "description"], how="all")
    print(f"  Dropped {before - len(df)} rows with no title/description")

    # Normalize title
    df["title_clean"] = df["title"].fillna("Unknown").str.strip().str.title()

    # Experience level
    df["experience_level"] = df["formatted_experience_level"].apply(normalize_seniority)

    # Work type
    df["work_type"] = df["formatted_work_type"].fillna("Unknown").str.strip().str.title()

    # Remote flag
    df["is_remote"] = df["remote_allowed"].fillna(0).astype(int)

    # Salary normalization to annual
    df["annual_salary"] = df.apply(normalize_salary_annual, axis=1)
    df["has_salary"] = df["annual_salary"].notna().astype(int)

    # Parse skills from skills_desc (semicolon or comma separated text)
    def parse_posting_skills(desc):
        if pd.isna(desc) or not isinstance(desc, str):
            return []
        skills = re.split(r"[;,\n]", desc)
        return normalize_skills([s.strip() for s in skills if s.strip()])

    df["skills_list"] = df["skills_desc"].apply(parse_posting_skills)
    df["skill_count"] = df["skills_list"].apply(len)
    df["has_skills"] = (df["skill_count"] > 0).astype(int)

    # Location: extract state
    df["location_clean"] = df["location"].fillna("Unknown").str.strip()

    result = df[[
        "job_id", "title_clean", "experience_level", "work_type", "is_remote",
        "location_clean", "annual_salary", "has_salary",
        "skills_list", "skill_count", "has_skills",
    ]].copy()
    result.rename(columns={
        "title_clean": "title",
        "location_clean": "location",
    }, inplace=True)

    print(f"  Cleaned shape: {result.shape}")
    print(f"  With salary: {result['has_salary'].sum():,} ({result['has_salary'].mean()*100:.1f}%)")
    print(f"  With skills: {result['has_skills'].sum():,} ({result['has_skills'].mean()*100:.1f}%)")
    print(f"  Remote: {result['is_remote'].sum():,} ({result['is_remote'].mean()*100:.1f}%)")

    # --- Aggregated title summary for enrichment ---
    salary_data = result[result["has_salary"] == 1]
    title_summary = salary_data.groupby("title").agg(
        median_salary=("annual_salary", "median"),
        posting_count=("job_id", "count"),
    ).reset_index()
    title_summary = title_summary[title_summary["posting_count"] >= 3]

    return result, title_summary


# ─── Step 6: Build Unified Skill Vocabulary ──────────────────────────────────

def build_skill_vocabulary(screening, resumes, jobs, postings):
    print("\n" + "=" * 70)
    print("STEP 6: Building Unified Skill Vocabulary")
    print("=" * 70)

    all_skills = set()
    for df_skills in [screening["skills_clean"], resumes["skills_clean"],
                      jobs["all_skills"], postings["skills_list"]]:
        for skill_list in df_skills:
            all_skills.update(skill_list)

    # Remove empty/very short noise
    all_skills = sorted(s for s in all_skills if len(s) >= 2)
    print(f"  Total unique skills across all datasets: {len(all_skills)}")

    # Fit unified MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=all_skills)
    mlb.fit([all_skills])

    # Save vocab and binarizer
    pd.DataFrame({"skill": all_skills}).to_csv(
        os.path.join(CLEAN_DIR, "skill_vocabulary.csv"), index=False
    )
    joblib.dump(mlb, os.path.join(MODEL_DIR, "unified_skill_binarizer.joblib"))
    print(f"  Saved skill_vocabulary.csv and unified_skill_binarizer.joblib")

    return mlb, all_skills


# ─── Step 7: Encode and Save Final Datasets ──────────────────────────────────

def encode_and_save(screening, resumes, jobs, postings, title_summary, mlb):
    print("\n" + "=" * 70)
    print("STEP 7: Encoding Features & Saving Clean Data")
    print("=" * 70)

    encoders = {}

    # --- Screening ---
    le_screen_role = LabelEncoder()
    screening["job_role_encoded"] = le_screen_role.fit_transform(screening["job_role"])
    encoders["screening_role_encoder"] = le_screen_role

    screen_skill_matrix = mlb.transform(screening["skills_clean"])
    screen_skill_df = pd.DataFrame(screen_skill_matrix, columns=mlb.classes_, index=screening.index)

    screening_final = pd.concat([
        screening.drop(columns=["skills_clean"]),
        screen_skill_df,
    ], axis=1)
    screening_final.to_csv(os.path.join(CLEAN_DIR, "screening_cleaned.csv"), index=False)
    print(f"  screening_cleaned.csv → {screening_final.shape}")

    # --- Resumes ---
    le_role = LabelEncoder()
    le_industry = LabelEncoder()
    resumes["role_encoded"] = le_role.fit_transform(resumes["role"])
    resumes["industry_encoded"] = le_industry.fit_transform(resumes["industry"])
    encoders["resume_role_encoder"] = le_role
    encoders["resume_industry_encoder"] = le_industry

    resume_skill_matrix = mlb.transform(resumes["skills_clean"])
    resume_skill_df = pd.DataFrame(resume_skill_matrix, columns=mlb.classes_, index=resumes.index)

    resumes_final = pd.concat([
        resumes.drop(columns=["skills_clean"]),
        resume_skill_df,
    ], axis=1)
    resumes_final.to_csv(os.path.join(CLEAN_DIR, "resumes_cleaned.csv"), index=False)
    print(f"  resumes_cleaned.csv  → {resumes_final.shape}")

    # --- Jobs ---
    le_job_title = LabelEncoder()
    le_job_industry = LabelEncoder()
    jobs["job_title_encoded"] = le_job_title.fit_transform(jobs["job_title"])
    jobs["industry_encoded"] = le_job_industry.fit_transform(jobs["industry"])
    encoders["job_title_encoder"] = le_job_title
    encoders["job_industry_encoder"] = le_job_industry

    job_skill_matrix = mlb.transform(jobs["all_skills"])
    job_skill_df = pd.DataFrame(job_skill_matrix, columns=mlb.classes_, index=jobs.index)

    # Build job_skill_map {job_title: [skills]}
    job_skill_map = {}
    for _, row in jobs.iterrows():
        title = row["job_title"]
        if title not in job_skill_map:
            job_skill_map[title] = set()
        job_skill_map[title].update(row["all_skills"])
    job_skill_map = {k: sorted(v) for k, v in job_skill_map.items()}
    joblib.dump(job_skill_map, os.path.join(MODEL_DIR, "job_skill_map.joblib"))

    jobs_final = pd.concat([
        jobs.drop(columns=["must_have_list", "nice_to_have_list", "all_skills", "description"]),
        job_skill_df,
    ], axis=1)
    jobs_final.to_csv(os.path.join(CLEAN_DIR, "jobs_cleaned.csv"), index=False)
    print(f"  jobs_cleaned.csv     → {jobs_final.shape}")

    # --- Postings ---
    postings_save = postings.drop(columns=["skills_list"]).copy()
    postings_save.to_csv(os.path.join(CLEAN_DIR, "postings_cleaned.csv"), index=False)
    print(f"  postings_cleaned.csv → {postings_save.shape}")

    title_summary.to_csv(os.path.join(CLEAN_DIR, "postings_title_summary.csv"), index=False)
    print(f"  postings_title_summary.csv → {title_summary.shape}")

    # Save all encoders
    for name, enc in encoders.items():
        joblib.dump(enc, os.path.join(MODEL_DIR, f"{name}.joblib"))
    print(f"  Saved {len(encoders)} label encoders to models/")

    return screening_final, resumes_final, jobs_final


# ─── Step 8: Build Resume-Job Matching Pairs ─────────────────────────────────

def build_matching_pairs(resumes, jobs_raw):
    print("\n" + "=" * 70)
    print("STEP 8: Building Resume-Job Matching Pairs")
    print("=" * 70)

    pairs = []
    jobs_by_industry = jobs_raw.groupby("industry")

    for _, resume in resumes.iterrows():
        r_skills = set(resume["skills_clean"])
        r_industry = resume["industry"]
        r_seniority = resume["seniority"]
        r_exp = resume["years_experience"]

        # Match with same-industry jobs
        if r_industry in jobs_by_industry.groups:
            industry_jobs = jobs_by_industry.get_group(r_industry)
        else:
            industry_jobs = pd.DataFrame()

        # Also sample up to 3 cross-industry jobs
        other_jobs = jobs_raw[jobs_raw["industry"] != r_industry]
        if len(other_jobs) > 3:
            other_sample = other_jobs.sample(3, random_state=42)
        else:
            other_sample = other_jobs

        candidate_jobs = pd.concat([industry_jobs, other_sample]).drop_duplicates(subset=["job_id"])

        for _, job in candidate_jobs.iterrows():
            j_must = set(job["must_have_list"])
            j_nice = set(job["nice_to_have_list"])

            must_overlap = len(r_skills & j_must)
            nice_overlap = len(r_skills & j_nice)
            must_count = max(len(j_must), 1)

            skill_match_ratio = must_overlap / must_count
            nice_match_ratio = nice_overlap / max(len(j_nice), 1)

            seniority_match = int(r_seniority == job["seniority"])
            industry_match = int(r_industry == job["industry"])

            # Composite match score (0-100)
            match_score = round(
                skill_match_ratio * 50 +
                nice_match_ratio * 15 +
                seniority_match * 20 +
                industry_match * 15,
                2
            )

            pairs.append({
                "resume_id": resume["resume_id"],
                "job_id": job["job_id"],
                "resume_role": resume["role"],
                "job_title": job["job_title"],
                "must_have_overlap": must_overlap,
                "must_have_total": len(j_must),
                "skill_match_ratio": round(skill_match_ratio, 4),
                "nice_to_have_overlap": nice_overlap,
                "nice_match_ratio": round(nice_match_ratio, 4),
                "seniority_match": seniority_match,
                "industry_match": industry_match,
                "experience_years": r_exp,
                "job_seniority_rank": SENIORITY_RANK.get(job["seniority"], 0),
                "match_score": match_score,
                "is_good_match": int(match_score >= 40),
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(os.path.join(CLEAN_DIR, "resume_job_pairs.csv"), index=False)
    print(f"  Generated {len(pairs_df):,} resume-job pairs")
    print(f"  Good matches (score >= 40): {pairs_df['is_good_match'].sum():,} ({pairs_df['is_good_match'].mean()*100:.1f}%)")
    print(f"  Match score stats: mean={pairs_df['match_score'].mean():.1f}, "
          f"median={pairs_df['match_score'].median():.1f}, max={pairs_df['match_score'].max():.1f}")

    return pairs_df


# ─── Step 9: Save Metadata ──────────────────────────────────────────────────

def save_metadata(screening, resumes, jobs, postings, pairs, skill_vocab):
    print("\n" + "=" * 70)
    print("STEP 9: Saving Preprocessing Metadata")
    print("=" * 70)

    metadata = {
        "datasets": {
            "screening": {"rows": len(screening), "cols": screening.shape[1]},
            "resumes": {"rows": len(resumes), "cols": resumes.shape[1]},
            "jobs": {"rows": len(jobs), "cols": jobs.shape[1]},
            "postings": {"rows": len(postings), "cols": postings.shape[1]},
            "resume_job_pairs": {"rows": len(pairs), "cols": pairs.shape[1]},
        },
        "skill_vocabulary_size": len(skill_vocab),
        "unique_roles_screening": int(screening["job_role"].nunique()) if "job_role" in screening.columns else 0,
        "unique_roles_resumes": int(resumes["role"].nunique()) if "role" in resumes.columns else 0,
        "unique_job_titles": int(jobs["job_title"].nunique()) if "job_title" in jobs.columns else 0,
    }

    with open(os.path.join(CLEAN_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to data/preprocessing_metadata.json")
    print(f"\n  Skill vocabulary: {metadata['skill_vocabulary_size']} unique skills")
    print(f"  Screening roles: {metadata['unique_roles_screening']}")
    print(f"  Resume roles: {metadata['unique_roles_resumes']}")
    print(f"  Job titles: {metadata['unique_job_titles']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "#" * 70)
    print("#  RESUME-BASED JOB PREDICTION — DATA PREPROCESSING PIPELINE")
    print("#" * 70 + "\n")

    # Load
    screening_raw, jobs_raw, resumes_raw, postings_raw = load_datasets()

    # Clean each dataset
    screening = clean_screening(screening_raw)
    resumes = clean_resumes(resumes_raw)
    jobs = clean_jobs(jobs_raw)
    postings, title_summary = clean_postings(postings_raw)

    # Unified skill vocab
    mlb, skill_vocab = build_skill_vocabulary(screening, resumes, jobs, postings)

    # Encode and save
    screening_enc, resumes_enc, jobs_enc = encode_and_save(
        screening, resumes, jobs, postings, title_summary, mlb
    )

    # Build matching pairs
    pairs = build_matching_pairs(resumes, jobs)

    # Save metadata
    save_metadata(screening, resumes, jobs, postings, pairs, skill_vocab)

    print("\n" + "#" * 70)
    print("#  PREPROCESSING COMPLETE")
    print("#" * 70)
    print(f"\n  Output directory: {CLEAN_DIR}")
    print(f"  Models directory: {MODEL_DIR}")
    print("\n  Files created:")
    for f in sorted(os.listdir(CLEAN_DIR)):
        size = os.path.getsize(os.path.join(CLEAN_DIR, f)) / 1024
        print(f"    data/{f:40s} {size:>8.1f} KB")
    for f in sorted(os.listdir(MODEL_DIR)):
        size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
        print(f"    models/{f:38s} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
