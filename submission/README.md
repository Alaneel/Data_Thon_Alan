# 🏢 SDS Datathon 2026 - AI-Driven Company Intelligence

An AI-powered company segmentation and intelligence dashboard for the Champions Group dataset.

## 🎯 Project Overview

> **🏆 Quick Start for Judges / Reviewers**
>
> **Mac/Linux**: Run this single command in your terminal:
>
> ```bash
> sh run_demo.sh
> ```
>
> _This script will automatically set up the environment, install dependencies, process the data, and launch the dashboard._

This project analyzes B2B company data to provide actionable business intelligence through:

- **Multi-Dimensional Clustering**: Segments 8,559 companies into 4 distinct market clusters using K-Means
- **B2B Lead Scoring**: Calculates 0-100 lead scores based on revenue potential, decision-making power, and data quality
- **Industry Benchmarking**: Compares companies against their industry peers
- **Anomaly Detection**: Identifies high-risk entities using Isolation Forest algorithm
- **Risk Assessment**: Detects shell companies, data quality issues, and orphan subsidiaries
- **LLM-Powered Insights**: Generates business personas, risk assessments, and competitive analysis using Google Gemini

## 📊 Key Results

| Metric                        | Value |
| ----------------------------- | ----- |
| Total Companies               | 8,559 |
| Clusters Identified           | 5     |
| Silhouette Score              | 0.34  |
| Industry Sectors Analyzed     | 70+   |
| Priority Leads (Score ≥70)    | 3     |
| High-Risk Entities (2+ Flags) | 156   |
| Shell Companies Detected      | 3,063 |
| Statistical Anomalies         | 428   |

### Dynamic Cluster Naming

Clusters are no longer static "Cluster 0/1". They are now dynamically named using a Tiered System ranking Revenue and Entity Structure. The 5 clusters map to 5 business tiers (Tiers 2 and 3 are often grouped for simplicity):

- **Tier 1**: Global HQ (Billion-dollar strategic hubs)
- **Tier 2**: Large Subsidiaries
- **Tier 3**: Mid-Market Subsidiaries
- **Tier 4**: Local HQ (Independent SMB Owners)
- **Tier 5**: Small Branches

**Example Result:** `Tier 1 Global HQ` vs `Tier 4 Local HQ`.

### Cluster Profiles

| Cluster    | Profile          | Characteristics                                |
| :--------- | :--------------- | :--------------------------------------------- |
| **Tier 1** | Global HQ        | Top revenue giants, strategic decision centers |
| **Tier 2** | Large Subsidiary | Major operational arms of global entities      |
| **Tier 3** | Mid-Market Subs  | Growing subsidiaries of larger groups          |
| **Tier 4** | Local HQ         | Independent parent entities, "SMB Owners"      |
| **Tier 5** | Small Branch     | Local offices, minimal strategic autonomy      |

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
# Navigate to project directory
python3 -m venv datathon_env
source datathon_env/bin/activate
pip install -r requirements.txt
```

### 2. Run Enhanced Analysis (Optional)

If you need to regenerate analysis results:

```bash
python src/enhanced_analysis.py
```

### 3. Run the Dashboard

```bash
export GEMINI_API_KEY='your-api-key'  # Required for AI features
streamlit run src/app.py
```

### 4. Open in Browser

Navigate to http://localhost:8501

## 📁 Project Structure

```
Datathon/
├── src/
│   ├── app.py                          # Streamlit dashboard (6 pages)
│   ├── enhanced_analysis.py            # Full analysis pipeline
│   └── llm_insights.py                 # LLM integration module
├── notebooks/
│   ├── company_intelligence_analysis.ipynb # Main Analysis
│   └── methodology_justification.ipynb     # Technical Appendix (Parameter Validation)
├── data/
│   ├── company_segmentation_results.csv # Processed results with scores
│   ├── champions_group_data.csv        # Raw dataset (8,559 companies)
│   └── champions_group_data_desc.csv   # Data dictionary
├── models/                             # Saved models and artifacts
├── requirements.txt                    # Python dependencies
├── COLLABORATION_GUIDE.md              # Team onboarding guide
└── README.md                           # This file
```

## 🛠️ Dashboard Features

### 📊 Overview

A command center for high-level ecosystem metrics.

- **KPIs**: Total Companies, Avg Revenue/Employees, Anomaly Count.
- **Interactive Charts**:
  - **Cluster Distribution**: 3D-style pie chart of market segments.
  - **Regional Heatmap**: Bar chart of companies by region.
  - **Revenue vs Employees**: Log-scale scatter plot to visualize efficiency and outliers.

### 💰 Lead Scoring

B2B lead prioritization with tier breakdown, score distribution, and top prospects table.

**Lead Score Calculation:**

- Revenue Potential: 40%
- Decision-Making Power: 25%
- Productivity (Rev/Employee): 20%
- Data Quality: 15%

- Data Quality: 15%

### 📋 Due Diligence Report (AI-Powered)

Generate comprehensive 1-page due diligence summaries for any company:

- **Executive Summary**: 2-sentence AI overview with verdict
- **Financial Health**: Revenue/Productivity vs industry benchmarks, A-F grade
- **Risk Assessment**: Overall risk level, red flags, mitigating factors
- **Hierarchy Analysis**: Decision-maker status, parent/subsidiary relationships
- **Recommendation**: Verdict (Proceed/Caution/Reject), action, timeline

**Commercial Value**: Traditional due diligence takes 2-3 hours → AI-powered: **30 seconds**

### 🚀 New Company Simulator

Simulate a new market entrant or prospect to evaluate potential fit and sales strategy before engagement.

### 🔬 Technical Validation & Advanced Analytics (New!)

To ensure enterprise-grade reliability, we implemented rigorous statistical validation:

- **Clustering Stability**: Validated using Bootstrap Analysis with an **ARI of 0.94** (Excellent Stability).
- **ML Validation**: Gradient Boosting classifier validates rule-based Lead Scoring with **93.11% accuracy**.
- **Hypothesis Testing**: T-test confirms "Priority" leads have significantly higher unit efficiency ($p < 0.003$).
- **Explainable AI (SHAP)**: Feature importance analysis confirms Revenue and Entity Score as key drivers.
- **Anomaly Case Studies**: Detailed profiling of "Ghost Giants" (Shell Companies) and "Lean Unicorns" (High Efficiency).

- **Interactive Inputs**: Input Company Name, Revenue, Employees, Industry, Entity Type, IT Spend, Market Value.
- **Real-Time Scoring**: Instantly calculates Lead Score (0-100) and Tier.
- **Cluster Prediction**: Predicts which tier (Global HQ, Local HQ, etc.) the company belongs to.
- **AI Report**: Generates actionable insights for sales and risk teams.

### 🌏 Market Entry Advisor (AI-Powered)

Plan expansion into new markets with data-driven recommendations:

- **Hybrid Input**: Natural language queries OR structured filters
- **Market Analysis**: Company counts, median revenue, avg Lead Score by country
- **Top Prospects**: Exportable list of highest-value targets per market
- **AI Strategy**: LLM-generated market entry recommendations

**Example Query**: _"Find fintech companies in Singapore with revenue over $5M"_

**Commercial Value**: Market research firms charge $20K-50K → Our platform: **instant & free**

### 🔍 Company Explorer

Search and filter the entire database with instant insights.

- **Intelligent Search**: Real-time filtering by company name.
- **Detailed Profiles**: View Revenue, Employees, Age, Market Value, IT Spend, and Decision-maker status.

### 📈 Cluster Analysis

Compare market segments to understand strategic positioning.

- **Visual Comparison**: Interactive bar charts comparing Median Revenue and Employees across clusters.
- **AI Personas**: Generate deep-dive "Cluster Personas" describing the typical characteristics and strategic needs of each segment.

### ⚠️ Risk Detection

Identify high-risk entities for due diligence.

- **Metrics**:
  - **Shell Company Risk**: High revenue (>100k) with missing/zero employees.
  - **Statistical Anomalies**: Outliers detected by Isolation Forest.
  - **High-Risk Entities**: Companies with multiple risk flags.
- **AI Investigation**: "Investigate with AI" feature explains _why_ a specific company was flagged as an anomaly compared to its peers.

### ⚖️ Company Comparison

Head-to-head competitive analysis.

- **Side-by-Side View**: Compare metrics (Revenue, Employees, Industry) of any two companies.
- **AI Competitive Intel**: Generates a strategic comparison highlighting "Leader vs Challenger" dynamics.

## 🤖 AI Capabilities

Powered by **Google Gemini**, the dashboard provides:

| Feature                   | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| **Cluster Personas**      | Auto-generated business personas for each market segment  |
| **Anomaly Investigation** | Risk assessment and explanation for flagged entities      |
| **Competitive Analysis**  | AI-powered company comparisons with strategic insights    |
| **⚔️ Action Reports**     | Actionable sales playbooks with Verdict, Reason, & Action |

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
- ✅ **Rigorous Technical Validation**: See `notebooks/methodology_justification.ipynb` for data-driven justification of all parameters (k=5, 5\% anomaly threshold, etc.).

## 👥 Team

Team Minions

## 📜 License

MIT License
