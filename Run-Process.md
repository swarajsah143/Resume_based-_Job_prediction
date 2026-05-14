# Run Process - Resume-Based Job Prediction System

## Prerequisites

- Python 3.10 or higher
- Git
- MongoDB Atlas account (free tier works)
- Gmail account (for OTP email verification)
- Google OAuth credentials (optional, for Google login)
- GitHub OAuth credentials (optional, for GitHub login)

---

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Resume_based-_Job_prediction
```

---

## Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

- Linux/Mac:
  ```bash
  source venv/bin/activate
  ```
- Windows:
  ```bash
  venv\Scripts\activate
  ```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Set Up MongoDB Atlas

1. Go to https://cloud.mongodb.com and sign up / log in.
2. Create a free M0 cluster (choose the region closest to you, e.g., Mumbai `ap-south-1` for India).
3. Go to **Database Access** and create a database user with a username and password.
4. Go to **Network Access** and add your IP address (or `0.0.0.0/0` for development).
5. Go to **Database** > **Connect** > **Drivers** and copy the connection string. It will look like:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

---

## Step 5: Set Up Gmail App Password (for OTP emails)

1. Go to https://myaccount.google.com/apppasswords
2. Sign in to your Google account.
3. Select **Mail** as the app and generate an App Password.
4. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`).

---

## Step 6: Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add the following content (replace placeholder values with your actual credentials):

```env
# Google OAuth (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth (optional)
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Email (Gmail SMTP)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-gmail-app-password

# MongoDB Atlas
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/resumeai?retryWrites=true&w=majority
```

**Important:**
- In `MONGODB_URI`, replace `<username>` and `<password>` with your Atlas database user credentials (remove the angle brackets).
- Add `/resumeai` before the `?` in the URI to specify the database name.
- Use the Gmail **App Password**, not your regular Gmail password.

---

## Step 7: Run the Application

```bash
python app.py
```

You should see:

```
Starting Flask app on http://127.0.0.1:5002
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5002
```

---

## Step 8: Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5002
```

---

## Available Features

| Feature | URL |
|---|---|
| Home Page | http://127.0.0.1:5002/ |
| Login | http://127.0.0.1:5002/login |
| Register | http://127.0.0.1:5002/register |
| Dashboard | http://127.0.0.1:5002/dashboard |
| Upload Resume | Via dashboard (PDF or DOCX) |
| OAuth Status Check | http://127.0.0.1:5002/auth/check |

---

## OAuth Setup (Optional)

### Google OAuth

1. Go to https://console.cloud.google.com
2. Create a new project or select an existing one.
3. Go to **APIs & Services** > **Credentials** > **Create Credentials** > **OAuth 2.0 Client ID**.
4. Set application type to **Web application**.
5. Add `http://127.0.0.1:5002/auth/google/callback` as an Authorized Redirect URI.
6. Copy the Client ID and Client Secret into your `.env` file.

### GitHub OAuth

1. Go to https://github.com/settings/developers
2. Click **New OAuth App**.
3. Set Homepage URL to `http://127.0.0.1:5002`.
4. Set Authorization Callback URL to `http://127.0.0.1:5002/auth/github/callback`.
5. Copy the Client ID and Client Secret into your `.env` file.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated virtual environment |
| MongoDB authentication failed | Check username/password in `MONGODB_URI`. Remove `< >` brackets around the password |
| Port 5002 already in use | The app auto-finds a free port. Or kill the process: `lsof -ti:5002 \| xargs kill -9` |
| OTP email not received | Verify `MAIL_USERNAME` and `MAIL_PASSWORD` in `.env`. Ensure you're using a Gmail App Password |
| OAuth not working | Check `/auth/check` endpoint to verify credentials are loaded. Ensure callback URLs match exactly |

---

## Project Structure

```
Resume_based-_Job_prediction/
|-- app.py                  # Main Flask application
|-- run.py                  # Alternative entry point
|-- migrate_to_mongo.py     # One-time SQLite to MongoDB migration script
|-- requirements.txt        # Python dependencies
|-- .env                    # Environment variables (not in git)
|-- .gitignore              # Git ignore rules
|-- templates/              # HTML templates
|-- static/                 # CSS, JS, images
|-- uploads/                # Temporary resume uploads (auto-created)
|-- data/                   # Training data
|-- models/                 # ML model files
|-- model_artifacts/        # Saved model artifacts
|-- train_model.py          # Model training script
|-- preprocess_data.py      # Data preprocessing
|-- preprocess_and_train.py # Combined preprocessing and training
```
