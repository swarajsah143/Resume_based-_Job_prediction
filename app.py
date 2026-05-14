"""
Resume-Based Job Suggestion System — Flask Backend
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, TypeVar

from dotenv import load_dotenv
load_dotenv()

_T = TypeVar("_T")


def _take(items: list[_T], n: int) -> list[_T]:
    """Return the first n items from a list."""
    result: list[_T] = []
    for i, item in enumerate(items):
        if i >= n:
            break
        result.append(item)
    return result

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'resumeai-dev-secret-key-change-in-production')
CORS(app)

# ─── Database Configuration ──────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'resumeai.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

db = SQLAlchemy(app)

# ─── OAuth Configuration ─────────────────────────────────────────────────────

oauth = OAuth(app)

oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

oauth.register(
    name='github',
    client_id=os.environ.get('GITHUB_CLIENT_ID'),
    client_secret=os.environ.get('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)


# ─── User Model ──────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fullname = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=True)
    auth_provider = db.Column(db.String(20), default='local')
    profile_picture = db.Column(db.String(512), nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    analyses = db.relationship('AnalysisHistory', backref='user', lazy=True, order_by='AnalysisHistory.created_at.desc()')


class AnalysisHistory(db.Model):
    __tablename__ = 'analysis_history'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_role = db.Column(db.String(150), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    resume_score = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


with app.app_context():
    db.create_all()

# ─── Skill & Job Databases ────────────────────────────────────────────────────

SKILL_DATABASE = {
    "programming": ["python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "php", "swift", "kotlin", "typescript", "scala", "r", "matlab", "perl", "dart", "lua"],
    "web_development": ["html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "bootstrap", "tailwind", "jquery", "next.js", "nuxt.js", "svelte", "webpack", "sass", "less"],
    "data_science": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "matplotlib", "seaborn", "jupyter", "spark", "hadoop", "tableau", "power bi", "data analysis", "machine learning", "deep learning", "nlp", "computer vision", "statistics"],
    "databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "firebase", "oracle", "sqlite", "dynamodb", "cassandra", "elasticsearch", "neo4j"],
    "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "devops", "linux", "nginx", "apache", "heroku", "vercel", "netlify"],
    "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin", "xamarin", "ionic"],
    "tools": ["git", "github", "gitlab", "jira", "confluence", "slack", "figma", "photoshop", "vs code", "postman", "swagger"],
    "soft_skills": ["leadership", "communication", "teamwork", "problem solving", "critical thinking", "time management", "project management", "agile", "scrum"],
}

EDUCATION_KEYWORDS = [
    "b.tech", "btech", "b.e", "bachelor", "master", "m.tech", "mtech", "m.s", "m.sc", "mba",
    "ph.d", "phd", "diploma", "associate", "bsc", "b.sc", "bca", "mca", "b.com", "m.com",
    "engineering", "computer science", "information technology", "electronics",
    "mechanical", "electrical", "civil", "chemical", "biotechnology",
    "university", "college", "institute", "school", "iit", "nit", "iiit", "bits",
    "cgpa", "gpa", "percentage", "10th", "12th", "ssc", "hsc", "cbse", "icse",
]

EXPERIENCE_KEYWORDS = [
    "intern", "internship", "experience", "worked", "developed", "built", "created",
    "managed", "led", "designed", "implemented", "deployed", "maintained",
    "software engineer", "developer", "analyst", "consultant", "associate",
    "junior", "senior", "lead", "manager", "director", "fresher", "trainee",
    "full-time", "part-time", "contract", "freelance",
    "years", "months", "present", "current",
]

JOB_ROLES = {
    "frontend_developer": {
        "title": "Frontend Developer",
        "required_skills": ["html", "css", "javascript", "react", "angular", "vue", "typescript", "bootstrap", "sass", "webpack", "next.js", "tailwind", "figma"],
        "keywords": ["frontend", "front-end", "ui", "ux", "web development", "responsive", "react", "angular", "vue"],
    },
    "backend_developer": {
        "title": "Backend Developer",
        "required_skills": ["python", "java", "node.js", "express", "django", "flask", "sql", "mongodb", "postgresql", "redis", "docker", "api", "rest"],
        "keywords": ["backend", "back-end", "server", "api", "database", "microservices", "node.js", "django", "flask"],
    },
    "full_stack_developer": {
        "title": "Full Stack Developer",
        "required_skills": ["html", "css", "javascript", "react", "node.js", "python", "sql", "mongodb", "git", "docker", "api", "express", "django"],
        "keywords": ["full stack", "fullstack", "full-stack", "mern", "mean", "web"],
    },
    "data_scientist": {
        "title": "Data Scientist",
        "required_skills": ["python", "r", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "sql", "statistics", "machine learning", "deep learning", "data analysis", "jupyter", "tableau"],
        "keywords": ["data science", "machine learning", "deep learning", "ai", "artificial intelligence", "analytics", "statistics", "ml"],
    },
    "data_analyst": {
        "title": "Data Analyst",
        "required_skills": ["python", "sql", "excel", "tableau", "power bi", "pandas", "statistics", "data analysis", "r", "matplotlib"],
        "keywords": ["data analyst", "analytics", "business intelligence", "bi", "reporting", "visualization", "dashboard"],
    },
    "ml_engineer": {
        "title": "Machine Learning Engineer",
        "required_skills": ["python", "tensorflow", "pytorch", "scikit-learn", "docker", "kubernetes", "sql", "machine learning", "deep learning", "nlp", "computer vision", "mlops"],
        "keywords": ["machine learning", "ml engineer", "deep learning", "neural network", "model", "training", "inference"],
    },
    "devops_engineer": {
        "title": "DevOps Engineer",
        "required_skills": ["linux", "docker", "kubernetes", "jenkins", "aws", "azure", "gcp", "terraform", "ci/cd", "git", "python", "bash", "nginx", "monitoring"],
        "keywords": ["devops", "infrastructure", "deployment", "automation", "cloud", "ci/cd", "containers"],
    },
    "mobile_developer": {
        "title": "Mobile App Developer",
        "required_skills": ["android", "ios", "react native", "flutter", "kotlin", "swift", "dart", "java", "firebase"],
        "keywords": ["mobile", "android", "ios", "app development", "react native", "flutter"],
    },
    "cloud_engineer": {
        "title": "Cloud Engineer",
        "required_skills": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux", "networking", "security", "python", "ci/cd"],
        "keywords": ["cloud", "aws", "azure", "gcp", "infrastructure", "saas", "paas", "iaas"],
    },
    "cybersecurity_analyst": {
        "title": "Cybersecurity Analyst",
        "required_skills": ["networking", "linux", "python", "security", "penetration testing", "firewall", "encryption", "siem", "vulnerability assessment"],
        "keywords": ["cybersecurity", "security", "ethical hacking", "penetration", "vulnerability", "InfoSec"],
    },
    "ui_ux_designer": {
        "title": "UI/UX Designer",
        "required_skills": ["figma", "photoshop", "sketch", "wireframing", "prototyping", "user research", "html", "css", "design thinking"],
        "keywords": ["ui", "ux", "design", "user interface", "user experience", "wireframe", "prototype", "figma"],
    },
    "software_engineer": {
        "title": "Software Engineer",
        "required_skills": ["python", "java", "c++", "git", "sql", "data structures", "algorithms", "oop", "design patterns", "testing"],
        "keywords": ["software engineer", "sde", "software development", "programming", "coding", "algorithms"],
    },
}

JOB_OPENINGS = {
    "Frontend Developer": [
        {"title": "Frontend Developer", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹12-25 LPA", "link": "https://careers.google.com"},
        {"title": "React Developer", "company": "Microsoft", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹10-22 LPA", "link": "https://careers.microsoft.com"},
        {"title": "UI Developer", "company": "Flipkart", "location": "Bangalore, India", "type": "Full-time", "salary": "₹8-18 LPA", "link": "https://www.flipkartcareers.com"},
        {"title": "Frontend Engineer", "company": "Swiggy", "location": "Remote", "type": "Full-time", "salary": "₹10-20 LPA", "link": "https://careers.swiggy.com"},
    ],
    "Backend Developer": [
        {"title": "Backend Developer", "company": "Amazon", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹15-30 LPA", "link": "https://www.amazon.jobs"},
        {"title": "Node.js Developer", "company": "Zomato", "location": "Gurugram, India", "type": "Full-time", "salary": "₹10-20 LPA", "link": "https://www.zomato.com/careers"},
        {"title": "Python Backend Engineer", "company": "Razorpay", "location": "Bangalore, India", "type": "Full-time", "salary": "₹12-25 LPA", "link": "https://razorpay.com/careers"},
        {"title": "Java Developer", "company": "TCS", "location": "Multiple Locations", "type": "Full-time", "salary": "₹5-12 LPA", "link": "https://ibegin.tcs.com"},
    ],
    "Full Stack Developer": [
        {"title": "Full Stack Developer", "company": "Infosys", "location": "Pune, India", "type": "Full-time", "salary": "₹6-15 LPA", "link": "https://www.infosys.com/careers"},
        {"title": "MERN Stack Developer", "company": "Paytm", "location": "Noida, India", "type": "Full-time", "salary": "₹8-18 LPA", "link": "https://paytm.com/careers"},
        {"title": "Full Stack Engineer", "company": "Atlassian", "location": "Bangalore, India", "type": "Full-time", "salary": "₹18-35 LPA", "link": "https://www.atlassian.com/company/careers"},
        {"title": "Web Developer", "company": "Wipro", "location": "Chennai, India", "type": "Full-time", "salary": "₹4-10 LPA", "link": "https://careers.wipro.com"},
    ],
    "Data Scientist": [
        {"title": "Data Scientist", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹18-40 LPA", "link": "https://careers.google.com"},
        {"title": "ML Researcher", "company": "Meta", "location": "Remote", "type": "Full-time", "salary": "₹20-45 LPA", "link": "https://www.metacareers.com"},
        {"title": "Data Scientist", "company": "Flipkart", "location": "Bangalore, India", "type": "Full-time", "salary": "₹15-30 LPA", "link": "https://www.flipkartcareers.com"},
        {"title": "AI/ML Engineer", "company": "Samsung R&D", "location": "Noida, India", "type": "Full-time", "salary": "₹12-28 LPA", "link": "https://www.samsung.com/in/careers"},
    ],
    "Data Analyst": [
        {"title": "Data Analyst", "company": "Deloitte", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹6-14 LPA", "link": "https://www2.deloitte.com/careers"},
        {"title": "Business Analyst", "company": "Accenture", "location": "Mumbai, India", "type": "Full-time", "salary": "₹5-12 LPA", "link": "https://www.accenture.com/careers"},
        {"title": "Analytics Consultant", "company": "EY", "location": "Gurugram, India", "type": "Full-time", "salary": "₹7-15 LPA", "link": "https://www.ey.com/careers"},
        {"title": "Data Analyst", "company": "PhonePe", "location": "Bangalore, India", "type": "Full-time", "salary": "₹8-16 LPA", "link": "https://www.phonepe.com/careers"},
    ],
    "Machine Learning Engineer": [
        {"title": "ML Engineer", "company": "NVIDIA", "location": "Pune, India", "type": "Full-time", "salary": "₹20-45 LPA", "link": "https://www.nvidia.com/careers"},
        {"title": "Deep Learning Engineer", "company": "Intel", "location": "Bangalore, India", "type": "Full-time", "salary": "₹15-35 LPA", "link": "https://www.intel.com/careers"},
        {"title": "AI Engineer", "company": "Microsoft", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹18-40 LPA", "link": "https://careers.microsoft.com"},
        {"title": "ML Platform Engineer", "company": "Uber", "location": "Remote", "type": "Full-time", "salary": "₹22-48 LPA", "link": "https://www.uber.com/careers"},
    ],
    "DevOps Engineer": [
        {"title": "DevOps Engineer", "company": "Amazon", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹12-28 LPA", "link": "https://www.amazon.jobs"},
        {"title": "Cloud DevOps Engineer", "company": "HCL", "location": "Noida, India", "type": "Full-time", "salary": "₹6-15 LPA", "link": "https://www.hcltech.com/careers"},
        {"title": "Site Reliability Engineer", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹20-40 LPA", "link": "https://careers.google.com"},
        {"title": "Platform Engineer", "company": "Thoughtworks", "location": "Pune, India", "type": "Full-time", "salary": "₹10-22 LPA", "link": "https://www.thoughtworks.com/careers"},
    ],
    "Mobile App Developer": [
        {"title": "Android Developer", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹12-28 LPA", "link": "https://careers.google.com"},
        {"title": "iOS Developer", "company": "Apple", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹15-35 LPA", "link": "https://www.apple.com/careers"},
        {"title": "Flutter Developer", "company": "Meesho", "location": "Bangalore, India", "type": "Full-time", "salary": "₹8-18 LPA", "link": "https://meesho.io/careers"},
        {"title": "React Native Developer", "company": "CRED", "location": "Bangalore, India", "type": "Full-time", "salary": "₹12-25 LPA", "link": "https://careers.cred.club"},
    ],
    "Cloud Engineer": [
        {"title": "Cloud Architect", "company": "AWS", "location": "Mumbai, India", "type": "Full-time", "salary": "₹18-40 LPA", "link": "https://www.amazon.jobs"},
        {"title": "Azure Cloud Engineer", "company": "Microsoft", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹14-30 LPA", "link": "https://careers.microsoft.com"},
        {"title": "GCP Engineer", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹16-35 LPA", "link": "https://careers.google.com"},
        {"title": "Cloud Solutions Engineer", "company": "IBM", "location": "Pune, India", "type": "Full-time", "salary": "₹10-22 LPA", "link": "https://www.ibm.com/careers"},
    ],
    "Cybersecurity Analyst": [
        {"title": "Security Analyst", "company": "Cisco", "location": "Bangalore, India", "type": "Full-time", "salary": "₹10-22 LPA", "link": "https://jobs.cisco.com"},
        {"title": "Cybersecurity Engineer", "company": "Palo Alto Networks", "location": "Mumbai, India", "type": "Full-time", "salary": "₹12-28 LPA", "link": "https://www.paloaltonetworks.com/company/careers"},
        {"title": "InfoSec Analyst", "company": "Deloitte", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹8-18 LPA", "link": "https://www2.deloitte.com/careers"},
        {"title": "Ethical Hacker", "company": "TCS", "location": "Multiple Locations", "type": "Full-time", "salary": "₹5-12 LPA", "link": "https://ibegin.tcs.com"},
    ],
    "UI/UX Designer": [
        {"title": "UI/UX Designer", "company": "Swiggy", "location": "Bangalore, India", "type": "Full-time", "salary": "₹8-20 LPA", "link": "https://careers.swiggy.com"},
        {"title": "Product Designer", "company": "Razorpay", "location": "Bangalore, India", "type": "Full-time", "salary": "₹10-25 LPA", "link": "https://razorpay.com/careers"},
        {"title": "Visual Designer", "company": "Flipkart", "location": "Bangalore, India", "type": "Full-time", "salary": "₹8-18 LPA", "link": "https://www.flipkartcareers.com"},
        {"title": "UX Researcher", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹15-30 LPA", "link": "https://careers.google.com"},
    ],
    "Software Engineer": [
        {"title": "Software Engineer", "company": "Google", "location": "Bangalore, India", "type": "Full-time", "salary": "₹18-40 LPA", "link": "https://careers.google.com"},
        {"title": "SDE-1", "company": "Amazon", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹15-30 LPA", "link": "https://www.amazon.jobs"},
        {"title": "Software Developer", "company": "Adobe", "location": "Noida, India", "type": "Full-time", "salary": "₹12-28 LPA", "link": "https://www.adobe.com/careers"},
        {"title": "Associate Software Engineer", "company": "Salesforce", "location": "Hyderabad, India", "type": "Full-time", "salary": "₹10-22 LPA", "link": "https://salesforce.wd1.myworkdayjobs.com"},
    ],
}

INTERVIEW_QUESTIONS = {
    "Frontend Developer": [
        {"question": "Explain the difference between CSS Flexbox and Grid. When would you use each?", "keywords": ["flexbox", "grid", "layout", "one-dimensional", "two-dimensional", "container", "responsive"]},
        {"question": "What is the Virtual DOM in React, and how does it improve performance?", "keywords": ["virtual dom", "reconciliation", "diffing", "performance", "render", "real dom", "update"]},
        {"question": "How do you optimize a website's loading speed?", "keywords": ["lazy loading", "minification", "compression", "caching", "cdn", "image optimization", "code splitting", "bundle"]},
        {"question": "Explain the concept of responsive design and how you implement it.", "keywords": ["media queries", "breakpoints", "viewport", "flexible", "mobile-first", "fluid", "percentage", "rem"]},
        {"question": "What are closures in JavaScript? Give an example of their practical use.", "keywords": ["closure", "scope", "function", "variable", "lexical", "inner function", "outer", "encapsulation"]},
    ],
    "Backend Developer": [
        {"question": "Explain the difference between SQL and NoSQL databases. When would you choose one over the other?", "keywords": ["sql", "nosql", "relational", "schema", "scalability", "structured", "unstructured", "document", "table"]},
        {"question": "What is RESTful API design? What are the key principles?", "keywords": ["rest", "stateless", "http methods", "get", "post", "put", "delete", "resource", "endpoint", "status codes"]},
        {"question": "How do you handle authentication and authorization in a web application?", "keywords": ["jwt", "token", "session", "oauth", "bcrypt", "hash", "middleware", "role", "permission"]},
        {"question": "Explain the concept of middleware in web frameworks.", "keywords": ["middleware", "request", "response", "pipeline", "chain", "processing", "filter", "interceptor"]},
        {"question": "What strategies do you use for handling errors and exceptions in production?", "keywords": ["try", "catch", "logging", "monitoring", "error handling", "graceful", "fallback", "status code"]},
    ],
    "Full Stack Developer": [
        {"question": "How would you design the architecture for a full-stack e-commerce application?", "keywords": ["frontend", "backend", "database", "api", "authentication", "payment", "architecture", "microservices", "monolith"]},
        {"question": "Explain the concept of state management in modern web applications.", "keywords": ["state", "redux", "context", "vuex", "store", "global", "local", "props", "component"]},
        {"question": "How do you ensure smooth communication between frontend and backend?", "keywords": ["api", "rest", "graphql", "cors", "json", "fetch", "axios", "async", "promise"]},
        {"question": "What is your approach to testing across the full stack?", "keywords": ["unit test", "integration", "e2e", "jest", "mocha", "selenium", "coverage", "mock", "assertion"]},
        {"question": "Describe how you would implement real-time features in a web application.", "keywords": ["websocket", "socket.io", "real-time", "push", "pub/sub", "event", "polling", "sse"]},
    ],
    "Data Scientist": [
        {"question": "Explain the bias-variance tradeoff and how it affects model performance.", "keywords": ["bias", "variance", "overfitting", "underfitting", "generalization", "complexity", "tradeoff", "regularization"]},
        {"question": "What is the difference between supervised and unsupervised learning?", "keywords": ["supervised", "unsupervised", "labeled", "unlabeled", "classification", "regression", "clustering", "training"]},
        {"question": "How would you handle missing data in a dataset?", "keywords": ["imputation", "mean", "median", "drop", "interpolation", "knn", "missing", "null", "fillna"]},
        {"question": "Explain cross-validation and why it's important.", "keywords": ["cross-validation", "k-fold", "train", "test", "split", "validation", "overfitting", "generalization"]},
        {"question": "What evaluation metrics would you use for a classification problem?", "keywords": ["accuracy", "precision", "recall", "f1", "auc", "roc", "confusion matrix", "specificity"]},
    ],
    "Data Analyst": [
        {"question": "How would you approach analyzing a large, messy dataset?", "keywords": ["cleaning", "exploration", "eda", "missing", "outlier", "visualization", "summary", "statistics", "pattern"]},
        {"question": "Explain the difference between correlation and causation.", "keywords": ["correlation", "causation", "relationship", "confounding", "variable", "statistical", "experiment"]},
        {"question": "What tools and techniques do you use for data visualization?", "keywords": ["tableau", "power bi", "matplotlib", "chart", "dashboard", "visualization", "graph", "insight"]},
        {"question": "How do you communicate analytical findings to non-technical stakeholders?", "keywords": ["storytelling", "visualization", "simplify", "business", "insight", "actionable", "presentation"]},
        {"question": "Describe your process for creating a data-driven report.", "keywords": ["data", "analysis", "report", "kpi", "metric", "insight", "recommendation", "summary"]},
    ],
    "Machine Learning Engineer": [
        {"question": "How would you deploy a machine learning model to production?", "keywords": ["deployment", "api", "docker", "flask", "serving", "pipeline", "monitoring", "scalability", "mlops"]},
        {"question": "Explain the difference between batch processing and real-time inference.", "keywords": ["batch", "real-time", "streaming", "latency", "throughput", "pipeline", "inference", "processing"]},
        {"question": "How do you handle class imbalance in a classification problem?", "keywords": ["oversampling", "undersampling", "smote", "class weight", "imbalance", "threshold", "precision", "recall"]},
        {"question": "What is transfer learning, and when would you use it?", "keywords": ["transfer learning", "pre-trained", "fine-tune", "domain", "feature extraction", "model", "weights"]},
        {"question": "Describe the steps involved in building an end-to-end ML pipeline.", "keywords": ["data collection", "preprocessing", "feature engineering", "training", "evaluation", "deployment", "monitoring", "pipeline"]},
    ],
    "DevOps Engineer": [
        {"question": "Explain the CI/CD pipeline and its components.", "keywords": ["ci", "cd", "build", "test", "deploy", "automation", "jenkins", "pipeline", "integration", "delivery"]},
        {"question": "What is containerization, and how does Docker work?", "keywords": ["container", "docker", "image", "dockerfile", "registry", "isolation", "lightweight", "microservices"]},
        {"question": "How would you set up monitoring and alerting for a production system?", "keywords": ["monitoring", "alerting", "prometheus", "grafana", "logs", "metrics", "uptime", "dashboard"]},
        {"question": "Explain Infrastructure as Code (IaC) and its benefits.", "keywords": ["iac", "terraform", "ansible", "cloudformation", "automation", "reproducible", "version control", "infrastructure"]},
        {"question": "How do you handle secrets management in a DevOps workflow?", "keywords": ["secrets", "vault", "environment", "encryption", "key", "config", "secure", "credential"]},
    ],
    "Mobile App Developer": [
        {"question": "What is the difference between native and cross-platform mobile development?", "keywords": ["native", "cross-platform", "react native", "flutter", "performance", "platform-specific", "code sharing"]},
        {"question": "How do you handle state management in a mobile application?", "keywords": ["state", "provider", "bloc", "redux", "getx", "architecture", "lifecycle", "data flow"]},
        {"question": "Explain how you would optimize a mobile app's performance.", "keywords": ["performance", "lazy loading", "caching", "memory", "battery", "network", "optimization", "profiling"]},
        {"question": "What is your approach to handling offline functionality in mobile apps?", "keywords": ["offline", "cache", "sync", "local storage", "sqlite", "queue", "connectivity"]},
        {"question": "How do you ensure a smooth user experience across different screen sizes?", "keywords": ["responsive", "adaptive", "screen size", "orientation", "layout", "constraints", "scalable"]},
    ],
    "Cloud Engineer": [
        {"question": "Explain the differences between IaaS, PaaS, and SaaS.", "keywords": ["iaas", "paas", "saas", "infrastructure", "platform", "software", "service", "cloud", "model"]},
        {"question": "How would you design a highly available and scalable cloud architecture?", "keywords": ["high availability", "scalability", "load balancer", "auto scaling", "redundancy", "multi-az", "failover"]},
        {"question": "What are the key security considerations in cloud environments?", "keywords": ["security", "iam", "encryption", "firewall", "vpc", "compliance", "access control", "audit"]},
        {"question": "Explain the concept of serverless computing and its use cases.", "keywords": ["serverless", "lambda", "functions", "event-driven", "scalable", "pay-per-use", "microservices"]},
        {"question": "How do you manage costs in a cloud environment?", "keywords": ["cost", "optimization", "reserved", "spot", "right-sizing", "monitoring", "budget", "tagging"]},
    ],
    "Cybersecurity Analyst": [
        {"question": "Explain the OWASP Top 10 web application security risks.", "keywords": ["owasp", "injection", "xss", "authentication", "security", "vulnerability", "risk", "web"]},
        {"question": "How would you respond to a security incident?", "keywords": ["incident response", "containment", "eradication", "recovery", "analysis", "forensics", "communication"]},
        {"question": "What is the difference between symmetric and asymmetric encryption?", "keywords": ["symmetric", "asymmetric", "key", "public", "private", "aes", "rsa", "encryption"]},
        {"question": "Explain the concept of zero trust security.", "keywords": ["zero trust", "verify", "least privilege", "micro-segmentation", "authentication", "never trust"]},
        {"question": "How do you perform a vulnerability assessment?", "keywords": ["vulnerability", "scan", "nmap", "nessus", "penetration", "risk", "assessment", "remediation"]},
    ],
    "UI/UX Designer": [
        {"question": "Walk us through your design process from research to final deliverable.", "keywords": ["research", "wireframe", "prototype", "user", "testing", "iteration", "design", "feedback"]},
        {"question": "How do you conduct user research and incorporate findings into design?", "keywords": ["user research", "interview", "survey", "persona", "journey", "insight", "empathy", "data"]},
        {"question": "Explain the principles of visual hierarchy in UI design.", "keywords": ["hierarchy", "contrast", "size", "color", "spacing", "alignment", "typography", "focal point"]},
        {"question": "How do you ensure accessibility in your designs?", "keywords": ["accessibility", "wcag", "contrast", "screen reader", "alt text", "keyboard", "aria", "inclusive"]},
        {"question": "What is your approach to creating a design system?", "keywords": ["design system", "component", "token", "consistency", "reusable", "documentation", "style guide"]},
    ],
    "Software Engineer": [
        {"question": "Explain the SOLID principles of object-oriented design.", "keywords": ["solid", "single responsibility", "open closed", "liskov", "interface segregation", "dependency inversion", "oop"]},
        {"question": "What data structure would you use for implementing a cache with O(1) operations?", "keywords": ["hashmap", "linked list", "lru", "cache", "o(1)", "dictionary", "doubly linked"]},
        {"question": "How do you approach debugging a complex issue in production?", "keywords": ["logging", "monitoring", "reproduce", "debugging", "stack trace", "breakpoint", "root cause"]},
        {"question": "Explain the concept of design patterns and name a few you've used.", "keywords": ["design pattern", "singleton", "factory", "observer", "strategy", "mvc", "pattern", "reusable"]},
        {"question": "How would you design a URL shortener service?", "keywords": ["hash", "database", "redirect", "collision", "base62", "scalability", "api", "cache"]},
    ],
}

RESUME_TIPS = {
    "add_projects": {
        "title": "Add Personal Projects",
        "description": "Include 2-3 relevant projects with descriptions of technologies used and outcomes achieved. This shows practical application of your skills.",
        "icon": "💡"
    },
    "add_certifications": {
        "title": "Get Certified",
        "description": "Add industry-recognized certifications like AWS, Google Cloud, or relevant technology certifications to boost credibility.",
        "icon": "🏅"
    },
    "add_keywords": {
        "title": "Optimize with Keywords",
        "description": "Include industry-specific keywords and buzzwords that ATS systems look for. Match job descriptions closely.",
        "icon": "🔑"
    },
    "quantify_achievements": {
        "title": "Quantify Achievements",
        "description": "Use numbers and metrics to describe your achievements (e.g., 'Improved performance by 40%' or 'Managed a team of 5').",
        "icon": "📊"
    },
    "improve_formatting": {
        "title": "Improve Formatting",
        "description": "Use clean formatting with consistent fonts, bullet points, and clear section headers. Keep it to 1-2 pages.",
        "icon": "📝"
    },
    "add_summary": {
        "title": "Add Professional Summary",
        "description": "Include a 2-3 line professional summary at the top highlighting your key strengths and career goals.",
        "icon": "📋"
    },
    "add_github": {
        "title": "Add GitHub/Portfolio Links",
        "description": "Include links to your GitHub profile, portfolio website, or LinkedIn. Show your work beyond the resume.",
        "icon": "🔗"
    },
    "tailor_resume": {
        "title": "Tailor for Each Role",
        "description": "Customize your resume for each job application. Highlight the most relevant skills and experiences.",
        "icon": "🎯"
    },
}


# ─── Helper Functions ──────────────────────────────────────────────────────────

def extract_text_from_pdf(filepath):
    """Extract text from PDF file."""
    try:
        import PyPDF2
        text = ""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_docx(filepath):
    """Extract text from DOCX file."""
    try:
        import docx
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception:
        return ""


def extract_skills(text):
    """Extract skills from resume text using keyword matching."""
    import re
    text_lower = text.lower()
    found_skills = []
    for category, skills in SKILL_DATABASE.items():
        for skill in skills:
            # Use word boundaries for exact matching
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.append({
                    "name": skill.title() if len(skill) > 3 else skill.upper(),
                    "category": category.replace("_", " ").title()
                })
    # Deduplicate
    seen = set()
    unique_skills = []
    for s in found_skills:
        if s["name"].lower() not in seen:
            seen.add(s["name"].lower())
            unique_skills.append(s)
    return unique_skills


def extract_education(text: str) -> list[str]:
    """Extract education info from resume text."""
    lines = text.split('\n')
    education_lines: list[str] = []

    for keyword in EDUCATION_KEYWORDS:
        for i, line in enumerate(lines):
            if keyword in line.lower() and line.strip():
                context = line.strip()
                if len(context) > 10 and context not in education_lines:
                    education_lines.append(context)

    return _take(education_lines, 6)  # Return top 6 entries


def extract_experience(text: str) -> list[str]:
    """Extract experience info from resume text."""
    lines = text.split('\n')
    experience_lines: list[str] = []

    for keyword in EXPERIENCE_KEYWORDS:
        for i, line in enumerate(lines):
            if keyword in line.lower() and line.strip():
                context = line.strip()
                if len(context) > 15 and context not in experience_lines:
                    experience_lines.append(context)

    return _take(experience_lines, 8)  # Return top 8 entries


def calculate_resume_strength(skills, education, experience, text):
    """Calculate resume strength score."""
    score = 0
    max_score = 100

    # Skills (40 points)
    skill_count = len(skills)
    if skill_count >= 10:
        score += 40
    elif skill_count >= 7:
        score += 30
    elif skill_count >= 4:
        score += 20
    elif skill_count >= 1:
        score += 10

    # Education (20 points)
    edu_count = len(education)
    if edu_count >= 3:
        score += 20
    elif edu_count >= 2:
        score += 15
    elif edu_count >= 1:
        score += 10

    # Experience (25 points)
    exp_count = len(experience)
    if exp_count >= 5:
        score += 25
    elif exp_count >= 3:
        score += 18
    elif exp_count >= 1:
        score += 10

    # Keywords and quality (15 points)
    text_lower = text.lower()
    quality_keywords = ["project", "certification", "award", "achievement",
                        "github", "portfolio", "linkedin", "volunteer", "publication"]
    found_quality = sum(1 for k in quality_keywords if k in text_lower)
    score += min(found_quality * 3, 15)

    return min(score, max_score)


def suggest_job_roles(skills: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Suggest job roles based on extracted skills."""
    skill_names: list[str] = [s["name"].lower() for s in skills]
    role_scores: list[dict[str, Any]] = []

    for role_id, role_info in JOB_ROLES.items():
        required = role_info["required_skills"]
        matched = [s for s in required if s in skill_names]
        if len(matched) > 0:
            confidence = round((len(matched) / len(required)) * 100, 1)
            confidence = min(confidence, 98)  # Cap at 98%
            role_scores.append({
                "id": role_id,
                "title": role_info["title"],
                "confidence": confidence,
                "matched_skills": [s.title() if len(s) > 3 else s.upper() for s in matched],
                "total_required": len(required),
            })

    role_scores.sort(key=lambda x: x["confidence"], reverse=True)
    return _take(role_scores, 5)


def get_skill_gap(skills, role_title):
    """Get missing skills for a specific role."""
    skill_names = [s["name"].lower() for s in skills]
    for role_id, role_info in JOB_ROLES.items():
        if role_info["title"] == role_title:
            required = role_info["required_skills"]
            missing = [s.title() if len(s) > 3 else s.upper() for s in required
                       if not any(s in sn for sn in skill_names)]
            have = [s.title() if len(s) > 3 else s.upper() for s in required
                    if any(s in sn for sn in skill_names)]
            return {"missing": missing, "have": have, "total_required": len(required)}
    return {"missing": [], "have": [], "total_required": 0}


def evaluate_answers(role_title: str, answers: list[str]) -> dict[str, Any]:
    """Evaluate mock interview answers."""
    questions = INTERVIEW_QUESTIONS.get(role_title, [])
    total_score: float = 0
    evaluations: list[dict[str, Any]] = []
    strengths: list[str] = []
    weaknesses: list[str] = []

    for i, q in enumerate(questions):
        if i < len(answers):
            answer = answers[i].lower()
            answer_words = answer.split()
            # Keyword matching
            keywords_found = [k for k in q["keywords"] if k in answer]
            keyword_score = min((len(keywords_found) / len(q["keywords"])) * 60, 60)
            # Length-based score
            length_score = min(len(answer_words) * 1.5, 25)
            # Coherence bonus
            coherence_bonus = 15 if len(answer_words) > 20 else (10 if len(answer_words) > 10 else 5)

            question_score = round(keyword_score + length_score + coherence_bonus, 1)
            question_score = min(question_score, 100)
            total_score += question_score

            status = "Excellent" if question_score >= 75 else ("Good" if question_score >= 50 else ("Needs Improvement" if question_score >= 25 else "Poor"))

            evaluations.append({
                "question": q["question"],
                "score": question_score,
                "status": status,
                "keywords_matched": len(keywords_found),
                "total_keywords": len(q["keywords"]),
            })

            if question_score >= 70:
                strengths.append(f"Strong answer for: {q['question'][:60]}...")
            elif question_score < 40:
                weaknesses.append(f"Improve answer for: {q['question'][:60]}...")

    avg_score = round(total_score / max(len(questions), 1), 1)

    improvement_tips: list[str] = []
    if avg_score < 50:
        improvement_tips.extend([
            "Study core concepts related to the role",
            "Practice answering with specific examples and technical details",
            "Include relevant keywords and terminology in your answers",
        ])
    if avg_score < 70:
        improvement_tips.extend([
            "Provide more detailed and structured responses",
            "Use the STAR method (Situation, Task, Action, Result) for behavioral questions",
        ])
    improvement_tips.extend([
        "Practice coding problems related to the role on LeetCode/HackerRank",
        "Stay updated with latest industry trends and technologies",
        "Review common interview patterns for this role",
    ])

    # Deduplicate tips while preserving order
    unique_tips: list[str] = []
    seen_tips: set[str] = set()
    for tip in improvement_tips:
        if tip not in seen_tips:
            seen_tips.add(tip)
            unique_tips.append(tip)

    return {
        "overall_score": avg_score,
        "evaluations": evaluations,
        "strengths": _take(strengths, 4),
        "weaknesses": _take(weaknesses, 4),
        "improvement_tips": _take(unique_tips, 6),
    }


def get_resume_tips(skills: list[dict[str, str]], education: list[str], experience: list[str], resume_strength: int) -> list[dict[str, str]]:
    """Generate personalized resume improvement tips."""
    tips: list[dict[str, str]] = []
    text_check = " ".join([s["name"] for s in skills]).lower()

    if len(skills) < 5:
        tips.append(RESUME_TIPS["add_keywords"])
    if len(experience) < 2:
        tips.append(RESUME_TIPS["add_projects"])
    if "certification" not in text_check and "certified" not in text_check:
        tips.append(RESUME_TIPS["add_certifications"])
    if resume_strength < 60:
        tips.append(RESUME_TIPS["quantify_achievements"])
    if resume_strength < 40:
        tips.append(RESUME_TIPS["improve_formatting"])
    tips.append(RESUME_TIPS["add_summary"])
    tips.append(RESUME_TIPS["add_github"])
    tips.append(RESUME_TIPS["tailor_resume"])

    return _take(tips, 6)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    user = None
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
    return render_template('index.html', user=user)


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    user = db.session.get(User, session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login_page'))
    history = AnalysisHistory.query.filter_by(user_id=user.id).order_by(AnalysisHistory.created_at.desc()).all()
    return render_template('dashboard.html', user=user, history=history)


@app.route('/api/history')
def api_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    records = AnalysisHistory.query.filter_by(user_id=session['user_id']).order_by(AnalysisHistory.created_at.desc()).all()
    return jsonify({"history": [
        {
            "id": r.id,
            "filename": r.filename,
            "predicted_role": r.predicted_role,
            "confidence": r.confidence,
            "resume_score": r.resume_score,
            "created_at": r.created_at.strftime('%d %b %Y, %I:%M %p'),
        } for r in records
    ]})


@app.route('/login', methods=['GET'])
def login_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/register', methods=['GET'])
def register_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('register.html')


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    fullname = (data.get('fullname') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    confirm = data.get('confirm_password') or ''

    if not fullname or len(fullname) < 2:
        return jsonify({"error": "Full name must be at least 2 characters"}), 400
    if not email or '@' not in email:
        return jsonify({"error": "Please enter a valid email address"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    user = User(
        fullname=fullname,
        email=email,
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Account created successfully!", "user": {"id": user.id, "fullname": user.fullname}}), 201


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()

    if not user or not user.password_hash or not check_password_hash(user.password_hash, password):
        if user and not user.password_hash:
            provider = user.auth_provider or 'social'
            return jsonify({"error": f"This account uses {provider.title()} login. Please sign in with {provider.title()}."}), 401
        return jsonify({"error": "Invalid email or password"}), 401

    user.last_login = datetime.now(timezone.utc)
    db.session.commit()

    session.clear()
    session['user_id'] = user.id
    session['user_name'] = user.fullname

    return jsonify({"message": "Login successful!", "redirect": "/dashboard", "user": {"id": user.id, "fullname": user.fullname}}), 200


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


# ─── OAuth Helpers ────────────────────────────────────────────────────────────

def _find_or_create_oauth_user(email: str, name: str, picture: str | None, provider: str) -> User:
    user = User.query.filter_by(email=email).first()
    if user:
        if picture and not user.profile_picture:
            user.profile_picture = picture
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()
    else:
        user = User(
            fullname=name,
            email=email,
            password_hash=None,
            auth_provider=provider,
            profile_picture=picture,
            last_login=datetime.now(timezone.utc),
        )
        db.session.add(user)
        db.session.commit()
    return user


def _login_user(user: User) -> None:
    session.clear()
    session['user_id'] = user.id
    session['user_name'] = user.fullname


# ─── Google OAuth Routes ──────────────────────────────────────────────────────

@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get('userinfo')
        if not userinfo:
            userinfo = oauth.google.get('https://openidconnect.googleapis.com/v1/userinfo').json()

        email = userinfo['email'].lower()
        name = userinfo.get('name', email.split('@')[0])
        picture = userinfo.get('picture')

        user = _find_or_create_oauth_user(email, name, picture, 'google')
        _login_user(user)
        return redirect(url_for('dashboard'))
    except Exception:
        return redirect(url_for('login_page', error='oauth_failed'))


# ─── GitHub OAuth Routes ─────────────────────────────────────────────────────

@app.route('/auth/github')
def github_login():
    redirect_uri = url_for('github_callback', _external=True)
    return oauth.github.authorize_redirect(redirect_uri)


@app.route('/auth/github/callback')
def github_callback():
    try:
        token = oauth.github.authorize_access_token()
        resp = oauth.github.get('user', token=token)
        profile = resp.json()

        email = profile.get('email')
        if not email:
            emails_resp = oauth.github.get('user/emails', token=token)
            emails = emails_resp.json()
            primary = next((e for e in emails if e.get('primary') and e.get('verified')), None)
            email = primary['email'] if primary else emails[0]['email']

        email = email.lower()
        name = profile.get('name') or profile.get('login', email.split('@')[0])
        picture = profile.get('avatar_url')

        user = _find_or_create_oauth_user(email, name, picture, 'github')
        _login_user(user)
        return redirect(url_for('dashboard'))
    except Exception:
        return redirect(url_for('login_page', error='oauth_failed'))


@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ['pdf', 'docx']:
        return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Extract text
    if ext == 'pdf':
        text = extract_text_from_pdf(filepath)
    else:
        text = extract_text_from_docx(filepath)

    if not text.strip():
        return jsonify({"error": "Could not extract text from the file. Please upload a text-based PDF/DOCX."}), 400

    # Analyze
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)
    resume_strength = calculate_resume_strength(skills, education, experience, text)
    suggested_roles = suggest_job_roles(skills)
    tips = get_resume_tips(skills, education, experience, resume_strength)

    # Save analysis history for logged-in users
    if 'user_id' in session and suggested_roles:
        top_role = suggested_roles[0]
        record = AnalysisHistory(
            user_id=session['user_id'],
            filename=file.filename,
            predicted_role=top_role['title'],
            confidence=top_role['confidence'],
            resume_score=resume_strength,
        )
        db.session.add(record)
        db.session.commit()

    # Clean up uploaded file
    try:
        os.remove(filepath)
    except:
        pass

    return jsonify({
        "filename": file.filename,
        "skills": skills,
        "education": education,
        "experience": experience,
        "resume_strength": resume_strength,
        "suggested_roles": suggested_roles,
        "tips": tips,
    })


@app.route('/job-openings', methods=['POST'])
def job_openings():
    data = request.get_json()
    role_title = data.get("role_title", "")
    openings = JOB_OPENINGS.get(role_title, [])
    return jsonify({"openings": openings})


@app.route('/mock-interview', methods=['POST'])
def mock_interview():
    data = request.get_json()
    role_title = data.get("role_title", "")
    questions = INTERVIEW_QUESTIONS.get(role_title, [])
    return jsonify({
        "role": role_title,
        "questions": [q["question"] for q in questions],
    })


@app.route('/evaluate-interview', methods=['POST'])
def evaluate_interview():
    data = request.get_json()
    role_title = data.get("role_title", "")
    answers = data.get("answers", [])
    result = evaluate_answers(role_title, answers)
    return jsonify(result)


@app.route('/skill-gap', methods=['POST'])
def skill_gap():
    data = request.get_json()
    skills = data.get("skills", [])
    role_title = data.get("role_title", "")
    gap = get_skill_gap(skills, role_title)
    return jsonify(gap)


def find_available_port(start_port: int = 5002, end_port: int = 5100) -> int:
    """Find a free port between start_port and end_port."""
    import socket

    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue

    raise RuntimeError(f"No free port found between {start_port} and {end_port}")


if __name__ == '__main__':
    default_port = int(os.environ.get('PORT', 5002))
    port = find_available_port(default_port, default_port + 50)
    print(f"Starting Flask app on http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
