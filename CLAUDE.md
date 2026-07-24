# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ResumeAI — a Flask web app that analyzes uploaded resumes (PDF/DOCX), extracts skills, scores resume strength, suggests matching job roles, detects skill gaps, and runs keyword-scored mock interviews. It has user accounts (local email/password + OTP, plus Google/GitHub OAuth), stores data in MongoDB Atlas, and is deployed as a Vercel serverless function.

## Commands

```bash
# Setup (Python 3.10+)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run locally — both entrypoints auto-find a free port in 5002–5052
python app.py          # primary; __main__ block guarded by `not IS_VERCEL`
python run.py          # alt entrypoint; supports --host / --port / --debug

# One-time data migration (SQLite resumeai.db -> MongoDB), rarely needed
python migrate_to_mongo.py
```

There is **no test suite, linter, or formatter configured** — no pytest, no CI that runs the app. Do not assume `pytest`/`make`/`npm` targets exist. `test_resume.pdf` / `test_resume.txt` are sample fixtures for manual upload testing, not automated tests.

## Architecture — the big picture

**`app.py` (~1370 lines) is the entire backend.** Config, DB models, all resume-analysis logic, and every route live in this one file. `run.py` and `api/index.py` are thin wrappers that import `app` from it.

**Runtime analysis is 100% rule-based, NOT machine learning.** This is the most important thing to understand before touching analysis behavior:
- All resume analysis runs off hardcoded Python dicts in `app.py`: `SKILL_DATABASE`, `JOB_ROLES`, `JOB_OPENINGS`, `INTERVIEW_QUESTIONS`, `RESUME_TIPS`, plus `EDUCATION_KEYWORDS` / `EXPERIENCE_KEYWORDS`. Skill extraction is keyword matching; role suggestion is `matched_required_skills / total_required` confidence; interview scoring is keyword + answer-length heuristics (`evaluate_answers`).
- The trained models in `models/*.joblib` and the training scripts (`train_model.py`, `preprocess_and_train.py`, `preprocess_data.py`) are a **separate offline ML pipeline that is NOT wired into the app** — `app.py` never imports `joblib` or loads any model. Editing the training scripts or model files has zero effect on what the running app does. To change app behavior, edit the dicts/functions in `app.py`.

**Vercel serverless deployment model:**
- `vercel.json` routes `/static/*` to static files and everything else to `api/index.py`, which just does `from app import app`.
- Behavior branches on the `IS_VERCEL` env flag (`VERCEL=1`): uploads go to `/tmp/uploads` (read-only FS elsewhere), and logging drops to INFO.
- MongoDB is treated as fragile across warm/cold container flips: `_connect_mongodb()` disconnects-then-reconnects with tiny pool + aggressive timeouts, and `@app.before_request _ensure_db()` re-pings and reconnects on every request, returning 503 for DB-dependent routes when down (while `/`, `/login`, `/register`, `/api/health` still serve).

**Data layer:** MongoEngine documents — `User`, `AnalysisHistory`, `OtpVerification`. The project was migrated off SQLite; `resumeai.db` and `migrate_to_mongo.py` are migration leftovers, not the live store.

**Auth:** session-based (`session['user_id']`). Local signup issues a 6-digit OTP emailed via Flask-Mail (5-min expiry, 60-sec resend cooldown) and only creates the `User` after `/api/verify-otp`. Google/GitHub OAuth via Authlib; `_oauth_configured()` gates each provider on whether real credentials are set (placeholder `your-...` values count as unconfigured). `/auth/check` reports which providers are live.

**Frontend:** server-rendered Jinja templates (`templates/index.html`, `dashboard.html`, `login.html`, `register.html`, `verify_otp.html`) with `static/js/app.js` and CSS. Not a JS-framework SPA. Analysis endpoints (`/upload`, `/job-openings`, `/mock-interview`, `/evaluate-interview`, `/skill-gap`) return JSON consumed by `app.js`.

**`/upload` request flow:** save file → extract text (`PyPDF2` for PDF, `python-docx` for DOCX) → `extract_skills` / `extract_education` / `extract_experience` → `calculate_resume_strength` (skills 40 / education 20 / experience 25 / quality keywords 15 = 100) → `suggest_job_roles` (top 5) → `get_resume_tips` → persist `AnalysisHistory` if logged in → delete the file → return JSON.

## Environment & conventions

- Config comes entirely from env vars (loaded via `python-dotenv` from `.env` locally, or the Vercel dashboard): `MONGODB_URI`, `SECRET_KEY`, `MAIL_SERVER`/`MAIL_PORT`/`MAIL_USERNAME`/`MAIL_PASSWORD`, `GOOGLE_CLIENT_ID`/`_SECRET`, `GITHUB_CLIENT_ID`/`_SECRET`. Missing `MONGODB_URI` falls back to `mongodb://localhost:27017/resumeai`. `.env` / `.env.local` are gitignored but present in the working tree and contain real secrets — never commit or echo them.
- **`Run-Process.md` is the accurate, current setup guide** (Mongo Atlas, Gmail app password, OAuth). **`README.md` is stale** — it describes an earlier version with no auth/MongoDB and the wrong project structure; trust the code and `Run-Process.md` over it.
- `.github/workflows/static.yml` publishes the whole repo to GitHub Pages as *static* content. It does not and cannot run this Flask app — treat it as unrelated to the real (Vercel) deployment.
