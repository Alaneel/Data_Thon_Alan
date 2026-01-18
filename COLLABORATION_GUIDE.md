# 🤝 Team Collaboration Guide - SDS Datathon 2026

Welcome to the **Company Intelligence Dashboard** project! This guide is designed for team members who might be new to using GitHub and Python in a collaborative environment.

Follow these steps to get the project running on your local machine and start contributing.

---

## 🛠️ 1. Prerequisites (Do this first)

Before you start, make sure you have these installed:

1.  **Python** (Version 3.9 or higher): [Download Python](https://www.python.org/downloads/)
2.  **VS Code** (Recommended Editor): [Download VS Code](https://code.visualstudio.com/)
3.  **Git**: [Download Git](https://git-scm.com/downloads)

> **👉 Tip:** If you are on Mac, you likely already have Python and Git. Open "Terminal" and type `python3 --version` and `git --version` to check.

---

## 🚀 2. Getting the Project (One-time Setup)

1.  **Open Terminal** (Mac) or **Command Prompt/PowerShell** (Windows).
2.  **Navigate to where you want the project stored** (e.g., Desktop):
    ```bash
    cd Desktop
    ```
3.  **Clone the repository** (Download the code):

    ```bash
    git clone https://github.com/YOUR_USERNAME/Datathon.git
    ```

    _(Replace with the actual repository URL we are using)_

4.  **Enter the project folder**:

    ```bash
    cd Datathon
    ```

5.  **Set up the Virtual Environment** (This keeps our project libraries separate from your system):

    **Mac/Linux:**

    ```bash
    python3 -m venv datathon_env
    source datathon_env/bin/activate
    ```

    **Windows:**

    ```bash
    python -m venv datathon_env
    datathon_env\Scripts\activate
    ```

6.  **Install Dependencies** (Download the required libraries):
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ 3. How to Run the Dashboard

Every time you want to work on the project:

1.  Open your Terminal/Command Prompt.
2.  Navigate to the folder (`cd Desktop/Datathon`).
3.  **Activate the environment** (Important!):
    - Mac: `source datathon_env/bin/activate`
    - Win: `datathon_env\Scripts\activate`
4.  **Run the App**:

    ```bash
    # If you have an API key for the LLM features:
    export GEMINI_API_KEY='your-api-key-here'  # On Windows use: set GEMINI_API_KEY='...'
    streamlit run app.py
    ```

    _If you don't have a key, just run `streamlit run app.py`. AI features will show a warning but the charts will still work._

5.  The dashboard should open automatically in your browser at `http://localhost:8501`.

---

## 📊 4. Dashboard Pages Overview

The dashboard has **6 main pages**:

| Page                      | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| **📊 Overview**           | Key metrics, cluster distribution charts, revenue vs employees scatter plot |
| **💰 Lead Scoring**       | B2B lead prioritization with priority/hot/warm/cold tiers                   |
| **🔍 Company Explorer**   | Search companies, view details, generate AI insights                        |
| **📈 Cluster Analysis**   | Compare cluster profiles, AI-generated business personas                    |
| **⚠️ Risk Detection**     | Shell company detection, data quality issues, risk investigation            |
| **⚖️ Company Comparison** | Side-by-side competitive analysis with AI                                   |

---

## 🔄 5. Simple Git Workflow (How to Collaborate)

Think of Git like a game save system + Dropbox.

### Step A: Before you start working ("Pull")

Always get the latest changes from your teammates before you start editing.

```bash
git pull
```

### Step B: Make your changes

Edit files in VS Code. Save them.

### Step C: Save your work (Add & Commit)

1.  **Tell Git which files to track**:

    ```bash
    git add .
    ```

    _(The `.` means "all changed files")_

2.  **Save the snapshot (Commit)**:
    ```bash
    git commit -m "Describe what you did briefly"
    ```
    _Example: `git commit -m "Fixed typo in title"` or `git commit -m "Added new chart"`_

### Step D: Share with the team (Push)

Upload your changes to GitHub.

```bash
git push
```

---

## 📂 6. Project Structure Map

| File                               | Description                                                           |
| ---------------------------------- | --------------------------------------------------------------------- |
| `app.py`                           | **Main Dashboard File** - All 6 pages and UI code                     |
| `enhanced_analysis.py`             | Analysis pipeline - generates clustering, lead scores, and risk flags |
| `llm_insights.py`                  | AI/LLM connection using Google Gemini                                 |
| `company_segmentation_results.csv` | Processed data used by the dashboard                                  |
| `champions_group_data.csv`         | Raw dataset (8,559 companies)                                         |
| `requirements.txt`                 | Python dependencies                                                   |

### Regenerating Analysis Results

If you change the analysis logic or need fresh results:

```bash
python enhanced_analysis.py
```

This will regenerate `company_segmentation_results.csv` with updated:

- Cluster assignments
- Lead scores (0-100)
- Risk flags
- Industry benchmarks

---

## 🆘 7. Troubleshooting Cheat Sheet

| Problem                           | Solution                                                                        |
| :-------------------------------- | :------------------------------------------------------------------------------ |
| **"Module not found: streamlit"** | You forgot to activate the environment. Run `source datathon_env/bin/activate`. |
| **"Git conflict"**                | Two people edited the same line. Ask for help before proceeding!                |
| **"Permission denied"**           | Make sure you are logged into GitHub. You might need a Personal Access Token.   |
| **App keeps crashing**            | Check the Terminal for error messages. Send the error to the group chat.        |
| **AI features not working**       | You need to set `GEMINI_API_KEY`. Charts will still work without it.            |
| **Data looks outdated**           | Run `python enhanced_analysis.py` to regenerate results.                        |

---

## 🔑 8. Getting a Gemini API Key (Optional)

For AI features to work, you need a free Google Gemini API key:

1. Go to [Google AI Studio](https://aistudio.google.com/api-keys)
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the key and set it before running the app:
   ```bash
   export GEMINI_API_KEY='your-key-here'
   ```

> **Note:** The dashboard works fine without an API key - you just won't have AI-generated insights.

---

## 📞 Need Help?

If you're stuck, drop a message in the team group chat with:

1. What you were trying to do
2. The error message (screenshot or copy-paste)
3. Which file you were working on
