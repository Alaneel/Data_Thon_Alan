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

## 🔄 4. Simple Git Workflow (How to Collaborate)

Think of Git like a game save system + Dropbox.

### Step A: Before you start working (Update your "Pull")

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

## 🆘 Troubleshooting Cheat Sheet

| Problem                           | Solution                                                                                           |
| :-------------------------------- | :------------------------------------------------------------------------------------------------- |
| **"Module not found: streamlit"** | You probably forgot to activate the environment. Run `source datathon_env/bin/activate`.           |
| **"Git conflict"**                | This happens if two people edit the same line. Ask for help before proceeding!                     |
| **"Permission denied"**           | Make sure you are logged into GitHub. You might need to set up a Personal Access Token or SSH key. |
| **App keeps crashing**            | Check the Terminal for error messages. Send the error to the group chat.                           |

---

## 📂 Project Structure Map

- `app.py`: **Main Dashboard File**. Most of the code for the UI is here.
- `enhanced_analysis.py`: Run this if you change the raw data or logic; it generates the results CSV.
- `llm_insights.py`: Handles the AI/LLM connection.
- `company_segmentation_results.csv`: The processed data used by the dashboard.
