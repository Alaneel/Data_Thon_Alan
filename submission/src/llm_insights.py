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
import time
import re


class CompanyInsightGenerator:
    """
    Generates intelligent insights for company analysis using Large Language Models.
    Defaults to Google Gemini API.
    """
    
    def __init__(self, api_key: str = None, model_name: str = 'gemini-3-flash-preview', fallback_model: str = 'gemini-2.5-flash-lite', status_callback=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.enabled = False
        self.status_callback = status_callback
        
        if not self.api_key:
            print("⚠️ Warning: GEMINI_API_KEY not found. Intelligent features will run in mock mode.")
            print("To enable: import os; os.environ['GEMINI_API_KEY'] = 'your_key'")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                self.fallback_model = genai.GenerativeModel(fallback_model)
                self.enabled = True
                print(f"✅ LLM Insight Generator initialized with Primary: {model_name}, Fallback: {fallback_model}")
            except Exception as e:
                print(f"❌ Failed to initialize LLM: {e}")

    def _safe_get_text(self, response) -> str:
        """Safely extracts text from response, handling empty/blocked cases."""
        try:
            # Check for candidates
            if not response.candidates:
                return "Error: No response candidates returned."
            
            candidate = response.candidates[0]
            
            # Check if safety filters were triggered (often finish_reason=3 or 4)
            # 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            if candidate.finish_reason > 2: 
                return f"Error: Request blocked (Reason Code: {candidate.finish_reason})"
            
            # Check for content parts
            if not candidate.content.parts:
                return "Error: Empty response content."
                
            return response.text
        except Exception as e:
            return f"Error parsing response: {e}"
            
    def _escape_markdown(self, text: str) -> str:
        """Escapes dollar signs to prevent Streamlit from interpreting them as LaTeX."""
        if not isinstance(text, str): return text
        return text.replace('$', '\\$')
        
    def _notify(self, message):
        """Helper to send status updates if callback exists."""
        if self.status_callback:
            self.status_callback(message)
        print(message)

    def _generate_with_retry(self, prompt, retries=3, backoff_factor=2):
        """Generates content with retry logic for rate limits + Model Fallback."""
        current_model = self.model
        using_fallback = False
        
        for i in range(retries + 1):
            try:
                return current_model.generate_content(prompt)
            except Exception as e:
                error_str = str(e)
                # Check for 429 / Quota Exceeded
                if "429" in error_str or "Quota exceeded" in error_str:
                    
                    # Try switching to fallback model if not already
                    if not using_fallback and i < retries:
                        self._notify(f"⚠️ Primary model quota exceeded. Switching to Fallback Model...")
                        current_model = self.fallback_model
                        using_fallback = True
                        time.sleep(1) # Small buffer
                        continue
                    
                    if i < retries:
                        # Smart wait: parse "retry in X seconds"
                        wait_match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                        if wait_match:
                            sleep_time = float(wait_match.group(1)) + 1 # Buffer
                            # Cap wait to avoid hanging too long (e.g., 60s)
                            sleep_time = min(sleep_time, 60)
                        else:
                            sleep_time = (i + 1) * backoff_factor
                            
                        self._notify(f"⏳ Rate limit hit. Waiting {sleep_time:.0f}s before retry {i+1}...")
                        time.sleep(sleep_time)
                        continue
                        self._notify(f"⏳ Rate limit hit. Waiting {sleep_time:.0f}s before retry {i+1}...")
                        time.sleep(sleep_time)
                        continue
                # If it's a 429 but we are out of retries, we still want to suppress it and return mock
                if "429" in error_str or "Quota exceeded" in error_str:
                     self._notify(f"⚠️ Quota exceeded and retries exhausted.")
                     break
                
                raise e # Re-raise if not quota error
        
        # If we exhausted retries for rate limit or encountered an unrecoverable 429
        self._notify(f"❌ Rate limit persisted. Returning fallback response.")
        
        # Create a Mock Response object structure that mimics Gemini response
        class MockCandidate:
            class Content:
                parts = ["Mock content"]
            content = Content()
            finish_reason = 1
            
        class MockResponse:
            candidates = [MockCandidate()]
            text = "Error: Rate Limit Exceeded. (Mock: This company shows strong fundamentals but requires further due diligence.)"
            
        return MockResponse()

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
        - Avg Age: {profile.get('Avg_Age', 'N/A')}
        
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
            response = self._generate_with_retry(prompt)
            return self._escape_markdown(self._safe_get_text(response))
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
        - Age: {company_row.get('Company_Age', 'N/A')} years
        - IT Spend: ${company_row.get('IT_Spend_Clean', 0):,.0f}
        - Market Value: ${company_row.get('Market_Value_Clean', 0):,.0f}
        
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
            response = self._generate_with_retry(prompt)
            return self._escape_markdown(self._safe_get_text(response))
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
            response = self._generate_with_retry(prompt)
            return self._escape_markdown(self._safe_get_text(response))
        except Exception as e:
            return f"Error comparing companies: {e}"

    def generate_action_report(self, company_row: pd.Series) -> Dict[str, str]:
        """
        Generates a 'Action Report' (Verdict, Reason, Risk, Action) for a company.
        Returns a dictionary.
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
        - Market Value: ${company_row.get('Market_Value_Clean', 0):,.0f}
        - IT Spend: ${company_row.get('IT_Spend_Clean', 0):,.0f}
        - Age: {company_row.get('Company_Age', 'N/A')} years
        - Domestic Ultimate: {company_row.get('Is_Domestic_Ultimate_Clean', 0)}
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
            response = self._generate_with_retry(prompt)
            text = self._safe_get_text(response).strip()
            
            # Check for error prefix from safe_get_text
            if text.startswith("Error"):
                 return {"Method": "LLM-Based", "Error": text}

            # Clean potential markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            # Escape values for markdown
            return {k: self._escape_markdown(v) for k, v in data.items()}
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return {"Method": "LLM-Based", "Error": str(e)}

    def generate_due_diligence_report(self, company_row: pd.Series, industry_stats: Dict = None, related_entities: List[str] = None) -> Dict[str, Any]:
        """
        Generates a comprehensive 1-page Due Diligence Summary for risk analysts and M&A teams.
        
        Returns a structured dictionary with:
        - executive_summary: 2-sentence overview
        - company_profile: Basic company information
        - financial_health: Revenue, productivity vs industry benchmarks
        - risk_assessment: Overall risk level with detailed flags
        - hierarchy_analysis: Decision-maker status, parent/subsidiary info
        - recommendation: Verdict, action, timeline
        """
        # Build context data
        revenue = company_row.get('Revenue_USD_Clean', 0)
        employees = company_row.get('Employees_Total_Clean', 0)
        productivity = revenue / employees if employees > 0 else 0
        
        # Industry comparison
        industry_median_rev = industry_stats.get('median_revenue', 1000000) if industry_stats else 1000000
        industry_median_prod = industry_stats.get('median_productivity', 50000) if industry_stats else 50000
        
        rev_vs_industry = ((revenue / industry_median_rev) - 1) * 100 if industry_median_rev > 0 else 0
        prod_vs_industry = ((productivity / industry_median_prod) - 1) * 100 if industry_median_prod > 0 else 0
        
        # Risk flags
        risk_flags = []
        if company_row.get('Risk_Shell', False):
            risk_flags.append("Shell Company Risk (high revenue, zero employees)")
        if company_row.get('Risk_DataQuality', False):
            risk_flags.append("Data Quality Issue (<50% completeness)")
        if company_row.get('Risk_OrphanSub', False):
            risk_flags.append("Orphan Subsidiary (no parent linkage)")
        if company_row.get('Anomaly_Label') == 'Anomaly':
            risk_flags.append("Statistical Anomaly (Isolation Forest)")
        
        risk_count = company_row.get('Risk_Flags', len(risk_flags))
        
        if not self.enabled:
            return {
                "executive_summary": f"Mock DD Report for {company_row.get('Company Sites', 'Unknown')}. Enable LLM for full analysis.",
                "company_profile": {
                    "name": company_row.get('Company Sites', 'Unknown'),
                    "industry": company_row.get('SIC Description', 'N/A'),
                    "region": company_row.get('Region', 'N/A'),
                    "country": company_row.get('Country', 'N/A'),
                    "age": f"{company_row.get('Company_Age', 'N/A')} years",
                    "entity_type": company_row.get('Entity Type', 'N/A')
                },
                "financial_health": {
                    "revenue": f"${revenue:,.0f}",
                    "employees": f"{employees:,.0f}",
                    "productivity": f"${productivity:,.0f}/employee",
                    "revenue_vs_industry": f"{rev_vs_industry:+.0f}%",
                    "productivity_vs_industry": f"{prod_vs_industry:+.0f}%",
                    "health_grade": "B" if rev_vs_industry > 0 else "C"
                },
                "risk_assessment": {
                    "overall_risk": "Medium" if risk_count > 0 else "Low",
                    "risk_score": risk_count,
                    "flags": risk_flags if risk_flags else ["No risk flags detected"],
                    "explanation": "Enable LLM for detailed risk analysis."
                },
                "hierarchy_analysis": {
                    "is_decision_maker": company_row.get('Is_Domestic_Ultimate_Clean', 0) == 1,
                    "entity_type": company_row.get('Entity Type', 'N/A'),
                    "has_parent": bool(company_row.get('Has_Parent', 0)),
                    "related_entities": related_entities or []
                },
                "recommendation": {
                    "verdict": "Review Required",
                    "action": "Enable LLM for specific recommendations",
                    "timeline": "N/A",
                    "confidence": "Low (Mock Mode)"
                }
            }

        prompt = f'''
        Act as a Senior Due Diligence Analyst at a Major Financial Institution.
        Generate a comprehensive Due Diligence Summary for the following company.
        
        === COMPANY DATA ===
        Name: {company_row.get('Company Sites', 'Unknown')}
        Industry: {company_row.get('SIC Description', 'N/A')}
        Region/Country: {company_row.get('Region', 'N/A')} / {company_row.get('Country', 'N/A')}
        Entity Type: {company_row.get('Entity Type', 'N/A')}
        Company Age: {company_row.get('Company_Age', 'N/A')} years
        
        === FINANCIAL METRICS ===
        Revenue: ${revenue:,.0f}
        Employees: {employees:,.0f}
        Productivity: ${productivity:,.0f} per employee
        Market Value: ${company_row.get('Market_Value_Clean', 0):,.0f}
        IT Spend: ${company_row.get('IT_Spend_Clean', 0):,.0f}
        
        === INDUSTRY COMPARISON ===
        Revenue vs Industry Median: {rev_vs_industry:+.1f}%
        Productivity vs Industry: {prod_vs_industry:+.1f}%
        
        === RISK SIGNALS ===
        Lead Score: {company_row.get('Lead_Score', 0)}/100 ({company_row.get('Lead_Tier', 'Unknown')})
        Total Risk Flags: {risk_count}
        Specific Flags: {', '.join(risk_flags) if risk_flags else 'None detected'}
        Is Domestic Ultimate (Decision Maker): {company_row.get('Is_Domestic_Ultimate_Clean', 0) == 1}
        
        === TASK ===
        Generate a structured JSON report with the following format:
        {{
            "executive_summary": "2-sentence overview: What is this company and should we engage?",
            "financial_health": {{
                "health_grade": "A/B/C/D/F based on overall financial position",
                "key_strength": "One sentence about their financial strength",
                "key_concern": "One sentence about financial concerns or 'None identified'"
            }},
            "risk_assessment": {{
                "overall_risk": "Low/Medium/High/Critical",
                "explanation": "2-3 sentences explaining the risk profile",
                "red_flags": ["List of specific concerns"],
                "mitigating_factors": ["List of positive factors that reduce risk"]
            }},
            "recommendation": {{
                "verdict": "PROCEED / PROCEED WITH CAUTION / REVIEW REQUIRED / DO NOT ENGAGE",
                "action": "Specific next step (e.g., 'Schedule call with CFO', 'Request audited financials')",
                "timeline": "Immediate / This Week / This Quarter / Long-term",
                "confidence": "High/Medium/Low"
            }}
        }}
        
        Return ONLY valid JSON. Be specific and data-driven in your analysis.
        '''
        
        try:
            response = self._generate_with_retry(prompt)
            text = self._safe_get_text(response).strip()
            
            if text.startswith("Error"):
                return {"error": text}
            
            # Clean potential markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            llm_output = json.loads(text.strip())
            
            # Merge LLM output with pre-computed data
            return {
                "executive_summary": self._escape_markdown(llm_output.get("executive_summary", "")),
                "company_profile": {
                    "name": company_row.get('Company Sites', 'Unknown'),
                    "industry": company_row.get('SIC Description', 'N/A'),
                    "region": company_row.get('Region', 'N/A'),
                    "country": company_row.get('Country', 'N/A'),
                    "age": f"{company_row.get('Company_Age', 'N/A')} years",
                    "entity_type": company_row.get('Entity Type', 'N/A')
                },
                "financial_health": {
                    "revenue": f"${revenue:,.0f}",
                    "employees": f"{employees:,.0f}",
                    "productivity": f"${productivity:,.0f}/employee",
                    "revenue_vs_industry": f"{rev_vs_industry:+.0f}%",
                    "productivity_vs_industry": f"{prod_vs_industry:+.0f}%",
                    "health_grade": llm_output.get("financial_health", {}).get("health_grade", "N/A"),
                    "key_strength": self._escape_markdown(llm_output.get("financial_health", {}).get("key_strength", "")),
                    "key_concern": self._escape_markdown(llm_output.get("financial_health", {}).get("key_concern", ""))
                },
                "risk_assessment": {
                    "overall_risk": llm_output.get("risk_assessment", {}).get("overall_risk", "Unknown"),
                    "risk_score": risk_count,
                    "flags": risk_flags if risk_flags else ["No risk flags detected"],
                    "explanation": self._escape_markdown(llm_output.get("risk_assessment", {}).get("explanation", "")),
                    "red_flags": llm_output.get("risk_assessment", {}).get("red_flags", []),
                    "mitigating_factors": llm_output.get("risk_assessment", {}).get("mitigating_factors", [])
                },
                "hierarchy_analysis": {
                    "is_decision_maker": company_row.get('Is_Domestic_Ultimate_Clean', 0) == 1,
                    "entity_type": company_row.get('Entity Type', 'N/A'),
                    "has_parent": bool(company_row.get('Has_Parent', 0)),
                    "related_entities": related_entities or []
                },
                "recommendation": {
                    "verdict": llm_output.get("recommendation", {}).get("verdict", "Review Required"),
                    "action": self._escape_markdown(llm_output.get("recommendation", {}).get("action", "")),
                    "timeline": llm_output.get("recommendation", {}).get("timeline", "N/A"),
                    "confidence": llm_output.get("recommendation", {}).get("confidence", "Medium")
                }
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            return {"error": f"Failed to parse LLM response: {e}"}
        except Exception as e:
            print(f"❌ DD Report Error: {e}")
            return {"error": str(e)}

    def parse_market_query(self, user_query: str) -> Dict[str, Any]:
        """
        Parses a natural language market entry query into structured filters.
        
        Example: "我想找新加坡的金融科技公司，收入超过500万美元"
        Returns: {"region": "Singapore", "industry": "finance", "min_revenue": 5000000}
        """
        if not self.enabled:
            return {
                "parsed": False,
                "error": "LLM not enabled. Use form filters instead.",
                "filters": {}
            }
        
        prompt = f'''
        You are a query parser for a B2B company database.
        
        USER QUERY: "{user_query}"
        
        CONTEXT:
        - The database MAINLY contains companies in "Asia".
        - Industries are stored as SIC Descriptions (e.g., "Mfg electronic components", "Whol computers", "Prepackaged software").
        - User might say "SaaS", you should map to ["software", "computer", "information"].
        - User might say "Fintech", you should map to ["finance", "bank", "investment", "insurance"].
        
        TASK:
        Extract filters. Return JSON only.
        
        {{
            "region": "Asia" (default to Asia if user asks for Asian countries, otherwise null),
            "country": "Specific country name (e.g. China, Singapore). Ignore trailing spaces.",
            "industry_keywords": ["keyword1", "keyword2", "keyword3"],  # Return 3-5 keywords to match partial SIC strings
            "min_revenue": number or null,
            "max_revenue": number or null,
            "min_employees": number or null,
            "max_employees": number or null,
            "target_tiers": [1,2,3,4,5] or subset,
            "exclude_high_risk": true/false,
            "limit": number of results (default 50)
        }}
        
        Return ONLY valid JSON. Use null for unspecified fields.
        '''
        
        try:
            response = self._generate_with_retry(prompt)
            text = self._safe_get_text(response).strip()
            
            if text.startswith("Error"):
                return {"parsed": False, "error": text, "filters": {}}
            
            # Clean markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            filters = json.loads(text.strip())
            return {"parsed": True, "filters": filters}
            
        except Exception as e:
            return {"parsed": False, "error": str(e), "filters": {}}

    def generate_market_entry_advice(self, market_stats: Dict, user_context: str = "") -> str:
        """
        Generates strategic market entry advice based on analyzed market data.
        """
        if not self.enabled:
            return "Enable LLM for AI-powered market entry strategy recommendations."
        
        prompt = f'''
        Act as a Senior Market Entry Strategist for B2B sales expansion.
        
        CONTEXT: {user_context}
        
        MARKET ANALYSIS DATA:
        {json.dumps(market_stats, indent=2, default=str)}
        
        TASK:
        Write a concise strategic recommendation (150 words max) covering:
        1. Which market to prioritize first and why
        2. Key opportunities in this market
        3. Potential challenges to watch
        4. Recommended first action
        
        Be specific and data-driven. Reference actual numbers from the data.
        '''
        
        try:
            response = self._generate_with_retry(prompt)
            return self._escape_markdown(self._safe_get_text(response))
        except Exception as e:
            return f"Error generating advice: {e}"
