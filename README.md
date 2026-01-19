# 🏢 SDS Datathon 2026 - AI-Driven Company Intelligence

An AI-powered company segmentation and intelligence dashboard for the Champions Group dataset.

## 🎯 Project Overview

This project analyzes B2B company data to provide actionable business intelligence through:

- **Multi-Dimensional Clustering**: Segments 8,559 companies into 4 distinct market clusters using K-Means
- **B2B Lead Scoring**: Calculates 0-100 lead scores based on revenue potential, decision-making power, and data quality
- **Industry Benchmarking**: Compares companies against their industry peers
- **Anomaly Detection**: Identifies high-risk entities using Isolation Forest algorithm
- **Risk Assessment**: Detects shell companies, data quality issues, and orphan subsidiaries
- **LLM-Powered Insights**: Generates business personas, risk assessments, and competitive analysis using Google Gemini

## 📊 Key Results

| Metric                     | Value  |
| -------------------------- | ------ |
| Total Companies            | 8,559  |
| Clusters Identified        | 4      |
| Silhouette Score           | 0.4801 |
| Industry Sectors Analyzed  | 70+    |
| Priority Leads (Score ≥70) | 2,500+ |
| High-Risk Entities         | 420+   |

### Dynamic Cluster Naming

Clusters are no longer static "Cluster 0/1". They are now dynamically named using a Tiered System ranking Revenue and Entity Structure:

- **Tier 1**: Top Revenue Segment
- **Tier 2-3**: Mid-Market
- **Tier 4+**: Small/Micro Segment
- **Structure**: HQ, Subsidiary, Branch, or Independent

**Example Result:** `Tier 1 HQ` vs `Tier 4 Branch`.

### Cluster Profiles

| Cluster       | Profile                     | Characteristics                                                |
| ------------- | --------------------------- | -------------------------------------------------------------- |
| **Cluster 0** | Medium Subsidiaries         | Service-focused entities with balanced revenue/employee ratios |
| **Cluster 1** | Medium High-Revenue Parents | Asset-light holding structures                                 |
| **Cluster 2** | Small Branches              | Compliance nodes with minimal operations                       |
| **Cluster 3** | Lean HoldCos                | Zero-employee high-revenue parent entities                     |

### Lead Score Distribution

| Tier        | Score Range | Description                                 |
| ----------- | ----------- | ------------------------------------------- |
| 🔥 Priority | 70-100      | High-value prospects for immediate outreach |
| 🌡️ Hot      | 50-69       | Strong potential, worth nurturing           |
| ☁️ Warm     | 30-49       | Moderate interest, requires qualification   |
| ❄️ Cold     | 0-29        | Low priority, minimal engagement            |

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /Users/alanwang/Desktop/Datathon
python3 -m venv datathon_env
source datathon_env/bin/activate
pip install -r requirements.txt
```

### 2. Run Enhanced Analysis (Optional)

If you need to regenerate analysis results:

```bash
python enhanced_analysis.py
```

### 3. Run the Dashboard

```bash
export GEMINI_API_KEY='your-api-key'  # Required for AI features
streamlit run app.py
```

### 4. Open in Browser

Navigate to http://localhost:8501

## 📁 Project Structure

```
Datathon/
├── app.py                              # Streamlit dashboard (6 pages)
├── enhanced_analysis.py                # Full analysis pipeline
├── llm_insights.py                     # LLM integration module
├── company_intelligence_analysis.ipynb # Jupyter notebook analysis
├── company_segmentation_results.csv    # Processed results with scores
├── champions_group_data.csv            # Raw dataset (8,559 companies)
├── champions_group_data_desc.csv       # Data dictionary
├── requirements.txt                    # Python dependencies
├── COLLABORATION_GUIDE.md              # Team onboarding guide
└── README.md                           # This file
```

## 🛠️ Dashboard Features

### 📊 Overview

Key metrics, cluster distribution pie charts, and interactive revenue vs employees scatter plots.

### 💰 Lead Scoring

B2B lead prioritization with tier breakdown, score distribution, and top prospects table.

**Lead Score Calculation:**

- Revenue Potential: 40%
- Decision-Making Power: 25%
- Productivity (Rev/Employee): 20%
- Data Quality: 15%

- Data Quality: 15%

### ⚔️ Battle Report (New)

Generate instant, AI-driven sales reports for any company:

- **Verdict**: Immediate "Go/No-Go" call.
- **Action**: Specific CRM next steps.
- **Risk**: Hidden red flags analysis.

### 🔍 Company Explorer

Search and filter companies, view detailed profiles, and generate AI-powered insights.

### 📈 Cluster Analysis

Compare cluster profiles with visual charts and AI-generated business personas.

### ⚠️ Risk Detection

Identify high-risk entities including:

- **Shell Companies**: High revenue, zero employees
- **Data Quality Issues**: Missing critical fields
- **Orphan Subsidiaries**: Subsidiaries without valid parent links

### ⚖️ Company Comparison

Side-by-side competitive analysis with AI-powered comparisons.

## 🤖 AI Capabilities

Powered by **Google Gemini**, the dashboard provides:

| Feature                   | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| **Cluster Personas**      | Auto-generated business personas for each market segment  |
| **Anomaly Investigation** | Risk assessment and explanation for flagged entities      |
| **Competitive Analysis**  | AI-powered company comparisons with strategic insights    |
| **⚔️ Battle Reports**     | Actionable sales playbooks with Verdict, Reason, & Action |

## 📋 Requirements

- Python 3.9+
- Google Gemini API Key (for AI features)
- ~50MB disk space for data files

### Key Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.0.0
scikit-learn>=1.3.0
google-generativeai>=0.3.0
```

See `requirements.txt` for full dependencies.

## 🏆 Competition Highlights

This project addresses all requirements of the SDS Datathon 2026:

- ✅ **Identify and group companies** with similar characteristics
- ✅ **Understand key differences** between companies within and across groups
- ✅ **Highlight notable patterns**, strengths, risks, and anomalies
- ✅ **Demonstrate commercial value** through Lead Scoring and Risk Assessment
- ✅ **BONUS: Generate interpretable explanations** using LLM integration

## 👥 Team

SDS Datathon 2026 Submission

## 📜 License

MIT License
