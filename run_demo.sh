#!/bin/bash

# ========================================================
# 🚀 SDS Datathon 2026 - Instant Demo Runner
# This script sets up everything and runs the AI Dashboard.
# ========================================================

echo "========================================================"
echo "   🏢 SDS Datathon 2026 - Company Intelligence Demo"
echo "========================================================"

# 1. Environment Setup
if [ ! -d "datathon_env" ]; then
    echo "\n📦 Creating Python virtual environment..."
    python3 -m venv datathon_env
fi

echo "\n🔌 Activating environment..."
source datathon_env/bin/activate

# 2. Dependencies
echo "\n📦 Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️ Error installing dependencies. Please check requirements.txt"
    exit 1
fi
echo "✅ Dependencies ready."

# 3. API Key Check
echo "\n🔑 Checking for Google Gemini API Key..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "   (Required for AI features like Persona Generation & Risk Analysis)"
    read -p "   👉 Enter your Gemini API Key (or press Enter to skip): " input_key
    if [ ! -z "$input_key" ]; then
        export GEMINI_API_KEY=$input_key
    else
        echo "   ⚠️ Continuing without AI features (Mock Mode)."
    fi
else
    echo "✅ Key found in environment."
fi

# 4. Run Analysis Pipeline (Optional but recommended)
echo "\n🔄 Checking for processed data..."
if [ ! -f "data/company_segmentation_results.csv" ]; then
    echo "   Data not found. Running Analysis Pipeline (takes ~10s)..."
    python3 src/enhanced_analysis.py
else
    read -p "   Process data again? [y/N]: " run_analysis
    if [[ "$run_analysis" =~ ^[Yy]$ ]]; then
        python3 src/enhanced_analysis.py
    else
        echo "   Skipping analysis. Using existing data."
    fi
fi

# 5. Launch Dashboard
echo "\n🚀 Launching Dashboard..."
echo "   (Press Ctrl+C to stop)"
echo "========================================================\n"

streamlit run src/app.py
