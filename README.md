# ResumeAI - Smart Resume Analyzer & Job Suggestion System

A web application that analyzes resumes to extract skills, suggest matching job roles, detect skill gaps, and conduct mock interviews with scoring.

## Features

- **Resume Upload** - Upload PDF or DOCX resumes (up to 16MB)
- **Skill Extraction** - Automatically identifies technical skills across 8 categories (programming, web dev, data science, databases, cloud, mobile, tools, soft skills)
- **Resume Strength Meter** - Scores resumes out of 100 based on skills, education, experience, and quality keywords
- **Job Role Matching** - Suggests top 5 job roles from 12 available roles with confidence scores
- **Skill Gap Detection** - Shows missing skills for any target role
- **Job Openings** - Displays curated job listings for each suggested role
- **Mock Interviews** - 5 role-specific questions per role with keyword-based evaluation
- **Performance Scoring** - Detailed breakdown with strengths, weaknesses, and improvement tips

## Workflow Diagram

```
+-------------------+
|   User uploads    |
|  resume (PDF/DOCX)|
+--------+----------+
         |
         v
+--------+----------+
|   Text Extraction  |
|  (PyPDF2 / docx)   |
+--------+----------+
         |
         v
+--------+----------+
|   Resume Analysis  |
|                    |
|  +-- Extract Skills (keyword matching across 8 categories)
|  +-- Extract Education (pattern matching)
|  +-- Extract Experience (pattern matching)
|  +-- Calculate Strength Score (0-100)
|  +-- Generate Resume Tips
+--------+----------+
         |
         v
+--------+----------+
|  Job Role Matching |
|                    |
|  Skills matched    |
|  against 12 roles  |
|  -> Top 5 returned |
|  with confidence % |
+--------+----------+
         |
         v
+--------+-----------+--------+
|                     |        |
v                     v        v
+-----------+  +------+----+  +--------+--------+
| Skill Gap |  |    Job     |  |     Mock        |
| Detector  |  |  Openings  |  |   Interview     |
|           |  |  (curated) |  | (5 questions)   |
+-----------+  +------------+  +--------+--------+
                                        |
                                        v
                               +--------+--------+
                               |   Answer        |
                               |   Evaluation    |
                               |                 |
                               | - Keyword match |
                               | - Length score  |
                               | - Coherence     |
                               +--------+--------+
                                        |
                                        v
                               +--------+--------+
                               | Performance     |
                               | Report          |
                               |                 |
                               | - Overall score |
                               | - Strengths     |
                               | - Weaknesses    |
                               | - Improvement   |
                               |   tips          |
                               +-----------------+
```

## Tech Stack

| Layer    | Technology          |
|----------|---------------------|
| Backend  | Flask (Python)      |
| Frontend | HTML, CSS, JS       |
| PDF Parse| PyPDF2              |
| DOCX Parse| python-docx        |
| CORS     | Flask-CORS          |

## Project Structure

```
Resume_based-_Job_prediction/
|-- app.py                  # Flask backend (routes + analysis logic)
|-- requirements.txt        # Python dependencies
|-- templates/
|   +-- index.html          # Single-page frontend
|-- static/
|   |-- css/style.css       # Styles
|   +-- js/app.js           # Frontend logic & API calls
+-- uploads/                # Temp directory for uploaded resumes (auto-cleaned)
```

## API Endpoints

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/`                   | Serve the web UI                     |
| POST   | `/upload`             | Upload & analyze a resume            |
| POST   | `/job-openings`       | Get job listings for a role          |
| POST   | `/mock-interview`     | Get interview questions for a role   |
| POST   | `/evaluate-interview` | Submit answers and get scored        |
| POST   | `/skill-gap`          | Get missing skills for a target role |

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Resume_based-_Job_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The app starts on `http://127.0.0.1:5002` (auto-finds an available port if 5002 is busy).

## Supported Job Roles

| #  | Role                      |
|----|---------------------------|
| 1  | Frontend Developer        |
| 2  | Backend Developer         |
| 3  | Full Stack Developer      |
| 4  | Data Scientist            |
| 5  | Data Analyst              |
| 6  | Machine Learning Engineer |
| 7  | DevOps Engineer           |
| 8  | Mobile App Developer      |
| 9  | Cloud Engineer            |
| 10 | Cybersecurity Analyst     |
| 11 | UI/UX Designer            |
| 12 | Software Engineer         |

## Resume Strength Scoring

| Component         | Max Points |
|-------------------|------------|
| Skills (count)    | 40         |
| Education         | 20         |
| Experience        | 25         |
| Quality keywords  | 15         |
| **Total**         | **100**    |

## License

This project is open source.
