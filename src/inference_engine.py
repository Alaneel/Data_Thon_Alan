import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore")

class CompanyEvaluator:
    """
    Evaluates new company data using pre-trained models.
    Aligned with Enhanced Analysis Pipeline v2.
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all models and stats from disk"""
        try:
            print(f"📂 Loading models from {self.model_dir}...")
            
            # Preprocessing
            self.scaler_impute = joblib.load(f"{self.model_dir}/scaler_impute.joblib")
            self.knn_imputer = joblib.load(f"{self.model_dir}/knn_imputer.joblib")
            
            # Clustering
            self.scaler_cluster = joblib.load(f"{self.model_dir}/scaler_cluster.joblib")
            self.kmeans = joblib.load(f"{self.model_dir}/kmeans.joblib")
            
            # Anomaly
            self.iso = joblib.load(f"{self.model_dir}/iso_forest.joblib")
            
            self.loaded = True
            print("✅ Models loaded successfully.")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.loaded = False

    def predict(self, company_data):
        """
        Predict Cluster, Score, and Risks for a single company dictionary.
        """
        if not self.loaded:
            return {"Error": "Models not loaded"}

        # --- 1. Clean & Parse Inputs ---
        rev_raw = self._clean_num(company_data.get("Revenue (USD)"))
        emp_raw = self._clean_num(company_data.get("Employees Total"))
        
        # New Inputs
        year_found = self._clean_num(company_data.get("Year Found"))
        is_domestic = 1 if company_data.get("Is Domestic Ultimate") else 0
        market_value = self._clean_num(company_data.get("Market Value (USD)"))
        it_spend = self._clean_num(company_data.get("IT Spend"))
        
        # Entity Map (Updated to Parent=3)
        entity_type = company_data.get("Entity Type", "Branch")
        entity_map = {'Parent': 3, 'Headquarters': 3, 'Subsidiary': 2, 'Branch': 1, 'Single Location': 3}
        entity_ord = entity_map.get(entity_type, 1)
        
        # --- 2. Imputation (Must match Pipeline) ---
        # Pipeline: Log Transform -> Scale -> Impute
        
        # Handle 0/NaN for log input
        rev_log_input = np.log1p(rev_raw) if (rev_raw and rev_raw > 0) else np.nan
        emp_log_input = np.log1p(emp_raw) if (emp_raw and emp_raw > 0) else np.nan
        
        # Vector: [Log_Rev, Log_Emp, Entity_Ord]
        input_vector = np.array([[rev_log_input, emp_log_input, entity_ord]])
        
        # Scale & Impute
        input_scaled = self.scaler_impute.transform(input_vector)
        input_imputed_scaled = self.knn_imputer.transform(input_scaled)
        input_imputed = self.scaler_impute.inverse_transform(input_imputed_scaled)
        
        # Retrieve Imputed Log Values
        log_rev = input_imputed[0][0]
        log_emp = input_imputed[0][1]
        
        # Convert back to real values for display/calculation
        rev_clean = np.expm1(log_rev)
        emp_clean = np.expm1(log_emp)
        
        # --- 3. Feature Engineering ---
        # Age
        current_year = 2026
        if pd.isna(year_found) or year_found == 0:
            age = 15 # Default median
        else:
            age = max(0, min(200, current_year - year_found))
            
        # RPE
        rpe = rev_clean / emp_clean if emp_clean > 0 else 0
        
        # IT Spend Per Emp
        if pd.isna(it_spend): it_spend = 0
        it_spend_per_emp = it_spend / emp_clean if emp_clean > 0 else 0
        
        # Has Parent
        has_parent = 1 if company_data.get("Parent Company") else 0
        
        # --- 4. Clustering ---
        # Features must match pipeline EXACTLY:
        # ['Log_Revenue', 'Log_Employees', 'Entity_Score', 'Has_Parent', 'Revenue_Per_Employee', 'Company_Age', 'Is_Domestic_Ultimate_Clean']
        
        cluster_vec = np.array([[
            log_rev, 
            log_emp, 
            float(entity_ord), 
            float(has_parent), 
            rpe, 
            float(age), 
            float(is_domestic)
        ]])
        
        # Handle Edge case NaNs
        cluster_vec = np.nan_to_num(cluster_vec)
        
        cluster_scaled = self.scaler_cluster.transform(cluster_vec)
        cluster_id = self.kmeans.predict(cluster_scaled)[0]
        
        # Dynamic Naming Logic (Replicated from Pipeline)
        # Parent=3, Sub=2, Branch=1
        # Logic: Tier based on cluster ID (approx) is hard to know without the map, 
        # so we rely on the loaded names if available, or simple fallback
        if hasattr(self, 'cluster_names_map') and cluster_id in self.cluster_names_map:
            cluster_name = self.cluster_names_map[cluster_id]
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        # --- 5. Anomaly Detection ---
        # Uses same scaled vector
        anomaly = self.iso.predict(cluster_scaled)[0]
        anomaly_label = "Anomaly" if anomaly == -1 else "Normal"
        
        # --- 6. Lead Scoring v2 ---
        completeness = self._calc_completeness(company_data, rev_raw, emp_raw)
        
        lead_score = self._calc_lead_score_v2(
            rev_clean, entity_ord, rpe, completeness, 
            is_domestic, market_value, it_spend_per_emp, age
        )
        
        if lead_score >= 75: lead_tier = "Priority" # Updated bin
        elif lead_score >= 50: lead_tier = "Hot"
        elif lead_score >= 30: lead_tier = "Warm"
        else: lead_tier = "Cold"
        
        return {
            "Cluster": cluster_name,
            "Lead_Score": lead_score,
            "Lead_Tier": lead_tier,
            "Anomaly": anomaly_label,
            "Revenue_Clean": rev_clean,
            "Employees_Clean": emp_clean,
            "Company_Age": age
        }

    def _clean_num(self, val):
        if val is None: return np.nan
        if isinstance(val, (int, float)): return val
        try:
            return float(str(val).replace(',', '').strip())
        except:
            return np.nan

    def _calc_completeness(self, data, rev, emp):
        # 7 fields: Rev, Emp, SIC, Entity, Region, Country, Year
        score = 0
        score += 1 if rev and rev > 0 else 0
        score += 1 if emp and emp > 0 else 0
        score += 1 if data.get("SIC Code") else 0
        score += 1 if data.get("Entity Type") else 0
        score += 1 if data.get("Region") else 0
        score += 1 if data.get("Country") else 0
        score += 1 if data.get("Year Found") else 0
        return score / 7.0

    def _calc_lead_score_v2(self, revenue, entity_score, rpe, completeness, is_domestic, market_val, it_spend_emp, age):
        score = 0
        # 1. Revenue (35)
        if revenue >= 1e8: score += 35
        elif revenue >= 1e7: score += 25
        elif revenue >= 1e6: score += 15
        elif revenue >= 1e5: score += 5
        
        # 2. Power (20)
        if is_domestic == 1: score += 15
        else: score += entity_score * 3
        
        # 3. Tech/Efficiency (20)
        if rpe >= 5e5: score += 10
        if it_spend_emp > 1000: score += 10
        elif it_spend_emp > 0: score += 5
        
        # 4. Market Value (15)
        if market_val and market_val > 0: score += 15
        
        # 5. Stability (10)
        if 3 <= age <= 10: score += 10
        elif age > 10: score += 5
        
        # Penalty
        if completeness < 0.5: score *= 0.8
        
        return min(100, max(0, score))

if __name__ == "__main__":
    evaluator = CompanyEvaluator()
    test_case = {
        "Name": "Local Kingpin",
        "Revenue (USD)": 15000000, 
        "Employees Total": 50, 
        "Entity Type": "Parent",
        "Year Found": 2010,
        "Is Domestic Ultimate": True
    }
    print(evaluator.predict(test_case))