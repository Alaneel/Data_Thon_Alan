"""
LLM Insights Module for Company Intelligence Analysis
SDS Datathon 2026

This module provides AI-powered insight generation using Google Gemini API.
It offers three main capabilities:
- Cluster persona generation
- Anomaly explanation  
- Company comparison analysis

Usage:
    from llm_insights import CompanyInsightGenerator
    llm = CompanyInsightGenerator(api_key='your-key')
    insight = llm.generate_cluster_insight(cluster_id, profile)

Requires: GEMINI_API_KEY environment variable or pass api_key directly.
"""

import os
import google.generativeai as genai
from typing import Dict, List, Any
import pandas as pd
import json


class CompanyInsightGenerator:
    """
    Generates intelligent insights for company analysis using Large Language Models.
    Defaults to Google Gemini API.
    """
    
    def __init__(self, api_key: str = None, model_name: str = 'gemini-3-flash-preview'):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.enabled = False
        
        if not self.api_key:
            print("⚠️ Warning: GEMINI_API_KEY not found. Intelligent features will run in mock mode.")
            print("To enable: import os; os.environ['GEMINI_API_KEY'] = 'your_key'")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                self.enabled = True
                print(f"✅ LLM Insight Generator initialized with {model_name}")
            except Exception as e:
                print(f"❌ Failed to initialize LLM: {e}")

    def generate_cluster_insight(self, cluster_id: int, profile: Dict[str, Any], key_features: List[str] = None) -> str:
        """Generates a business persona and strategic analysis for a company cluster."""
        if not self.enabled:
            return f"Cluster {cluster_id} Analysis (Mock): This cluster contains {profile.get('Size', 'N/A')} companies. Enable LLM for detailed persona."
        
        prompt = f"""
        Act as a Senior Business Strategy Consultant. Analyze the following data profile for a group of companies (Cluster {cluster_id}).
        
        DATA PROFILE:
        - Size: {profile.get('Size', 'N/A')} companies ({profile.get('Percentage', 'N/A')} of total)
        - Median Revenue: ${profile.get('Median_Revenue_USD', 0):,.2f}
        - Median Employees: {profile.get('Median_Employees', 0):,.0f}
        - Top Region: {profile.get('Top_Region', 'N/A')}
        - Top Industry: {profile.get('Top_Industry', 'N/A')}
        - Primary Entity Type: {profile.get('Top_Entity_Type', 'N/A')}
        
        TASK:
        1. create a short, professional "Persona Name" for this cluster (e.g., "Asian Tech SMBs", "Global Enterprise HQs").
        2. Write a 2-sentence "Executive Summary" defining their key operational characteristics.
        3. Identify 3 "Strategic Needs" these companies likely have (e.g., cloud migration, credit lines, export compliance).
        
        OUTPUT FORMAT:
        **Persona:** [Name]
        **Summary:** [Text]
        **Strategic Needs:**
        - [Need 1]
        - [Need 2]
        - [Need 3]
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating insight: {e}"

    def explain_anomaly(self, company_row: pd.Series, cluster_avg: pd.Series) -> str:
        """Explains why a specific company is flagged as an anomaly compared to its cluster peers."""
        if not self.enabled:
            return "Anomaly Explanation (Mock): This company's metrics deviate significantly from the cluster average."

        prompt = f"""
        Act as a Risk Assessment Analyst. A company has been flagged as an 'Anomaly' by our Isolation Forest algorithm.
        Explain WHY based on the comparison below.
        
        COMPANY DATA:
        - Name/ID: {company_row.get('DUNS Number ', 'Unknown')}
        - Revenue: ${company_row.get('Revenue_USD_Clean', 0):,.2f}
        - Employees: {company_row.get('Employees_Total_Clean', 0):,.0f}
        - Region: {company_row.get('Region', 'N/A')}
        
        CLUSTER AVERAGE ({company_row.get('Cluster_Name', 'Unknown')}):
        - Avg Revenue: ${cluster_avg.get('Revenue_USD_Clean', 0):,.2f}
        - Avg Employees: {cluster_avg.get('Employees_Total_Clean', 0):,.0f}
        
        TASK:
        1. Identify the specific metric(s) that look unusual (too high? too low? mismatch?).
        2. Assess if this looks like a "High Performance" outlier (good) or a "Data Error/Financial Risk" outlier (bad).
        3. Provide a one-sentence recommendation for the analyst.
        
        Keep it concise (approx 100 words).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error explaining anomaly: {e}"

    def compare_companies(self, company_a: pd.Series, company_b: pd.Series) -> str:
        """Compares two companies and suggests competitive insights."""
        if not self.enabled:
            return "Company Comparison (Mock): Enable LLM to see detailed competitive analysis."

        prompt = f"""
        Compare these two companies for a sales strategy report.
        
        Company A:
        - Employees: {company_a.get('Employees_Total_Clean', 0)}
        - Revenue: ${company_a.get('Revenue_USD_Clean', 0):,.2f}
        - Region: {company_a.get('Region', 'N/A')}
        - Industry: {company_a.get('SIC Description', 'N/A')}
        
        Company B:
        - Employees: {company_b.get('Employees_Total_Clean', 0)}
        - Revenue: ${company_b.get('Revenue_USD_Clean', 0):,.2f}
        - Region: {company_b.get('Region', 'N/A')}
        - Industry: {company_b.get('SIC Description', 'N/A')}
        
        TASK:
        1. Highlight the biggest difference between them.
        2. Suggest which one is the "Leader" and which is the "Challenger" (or if they are peers).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error comparing companies: {e}"

    def generate_action_report(self, company_row: pd.Series) -> Dict[str, str]:
        """
        Generates a 'Battle Report' (Verdict, Reason, Risk, Action) for a company.
        Returns a dictionary.
        """
        if not self.enabled:
            return {
                "Method": "LLM-Based (Mock)",
                "Verdict": "🤖 AI VERDICT (Simulated)",
                "Reason": "Enable LLM to generate natural language analysis.",
                "Risk": "AI would identify risks here.",
                "Action": "Enable LLM for specific recommendations."
            }

        prompt = f'''
        Act as a Senior Sales & Risk Analyst. Analyze this company for our B2B Sales Team.
        
        COMPANY DATA:
        - Name: {company_row.get('DUNS Number ', 'Unknown')}
        - Cluster: {company_row.get('Cluster_Name', 'Unknown')}
        - Revenue: ${company_row.get('Revenue_USD_Clean', 0):,.0f}
        - Employees: {company_row.get('Employees_Total_Clean', 0):,.0f}
        - Lead Score: {company_row.get('Lead_Score', 0)} ({company_row.get('Lead_Tier', 'Unknown')})
        - Risk Flags: {company_row.get('Risk_Flags', 0)}
        - Details: Anomaly={company_row.get('Anomaly_Label', 'N/A')}, Shell={company_row.get('Risk_Shell', False)}
        
        OUTPUT JSON FORMAT:
        {{
            "Verdict": "Short powerful phrase (e.g., 'Prime Acquisition Target' or 'Stay Away')",
            "Reason": "One sharp sentence explaining why, highlighting nuances.",
            "Risk": "Assessment of risk, detecting subtleties (e.g., 'Revenue seems too high for team size').",
            "Action": "Specific recommendation (e.g., 'Send VP of Sales', 'Request Financial Audit')."
        }}
        
        Return ONLY VALID JSON.
        '''
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Clean potential markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            return {"Method": "LLM-Based", "Error": str(e)}
