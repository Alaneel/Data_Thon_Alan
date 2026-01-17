# 🏢 SDS Datathon 2026 - AI-Driven Company Intelligence

An AI-powered company segmentation and intelligence dashboard for the Champions Group dataset.

## 🎯 Project Overview

This project analyzes B2B company data to provide actionable business intelligence through:

- **K-Means Clustering**: Segments 8,559 companies into 4 distinct market clusters
- **Anomaly Detection**: Identifies 423 potential data anomalies using Isolation Forest
- **LLM-Powered Insights**: Generates business personas, risk assessments, and competitive analysis using Google Gemini
- **Interactive Dashboard**: Streamlit-based visualization for exploring insights

## 📊 Key Results

| Metric              | Value      |
| ------------------- | ---------- |
| Total Companies     | 8,559      |
| Clusters Identified | 4          |
| Silhouette Score    | 0.4801     |
| Anomalies Detected  | 423 (4.9%) |

### Cluster Profiles

- **Cluster 0**: Medium Subsidiaries - Service-focused entities with balanced revenue/employee ratios
- **Cluster 1**: Medium High-Revenue Parents - Asset-light holding structures
- **Cluster 2**: Small Branches - Compliance nodes with minimal operations
- **Cluster 3**: Lean HoldCos - Zero-employee high-revenue parent entities

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /Users/alanwang/Desktop/Datathon
python -m venv datathon_env
source datathon_env/bin/activate
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
export GEMINI_API_KEY='your-api-key'  # Required for AI features
streamlit run app.py
```

### 3. Open in Browser

Navigate to http://localhost:8501

## 📁 Project Structure

```
Datathon/
├── app.py                          # Streamlit dashboard
├── llm_insights.py                 # LLM integration module
├── company_intelligence_analysis.ipynb  # Full analysis notebook
├── company_segmentation_results.csv     # Processed results
├── champions_group_data.csv        # Raw dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Features

### Dashboard Pages

1. **📊 Overview** - Key metrics, cluster distribution, scatter plots
2. **🔍 Company Explorer** - Search and filter companies, view details
3. **📈 Cluster Analysis** - Compare cluster profiles with AI personas
4. **⚠️ Anomaly Detection** - Investigate flagged companies with AI
5. **⚖️ Company Comparison** - Side-by-side competitive analysis

### AI Capabilities

- **Cluster Personas**: Auto-generated business personas for each segment
- **Anomaly Explanation**: Risk assessment for flagged entities
- **Competitive Analysis**: AI-powered company comparisons

## 📋 Requirements

- Python 3.9+
- Google Gemini API Key (for AI features)
- See `requirements.txt` for full dependencies

## 👥 Team

SDS Datathon 2026 Submission

## 📜 License

MIT License
