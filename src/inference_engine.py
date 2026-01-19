import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.base import BaseEstimator, TransformerMixin

class CompanyEvaluator:
    """
    Evaluates new company data using pre-trained models.
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
            self.iso = joblib.load(f"{self.model_dir}/isolation_forest.joblib")
            
            # Stats (Global)
            with open(f"{self.model_dir}/global_stats.json", "r") as f:
                self.global_stats = json.load(f)
                
            self.cluster_names_map = {int(k): v for k, v in self.global_stats["cluster_names_map"].items()}
            self.rpe_median = self.global_stats["rpe_median"]
            self.entity_map = self.global_stats["entity_map"]
            
            # Stats (Industry)
            self.industry_stats = pd.read_csv(f"{self.model_dir}/industry_stats.csv")
            # Convert to dict for faster lookup
            self.ind_lookup = self.industry_stats.set_index('SIC_2Digit').to_dict('index')
            
            self.loaded = True
            print("✅ Models loaded successfully.")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.loaded = False

    def predict(self, company_data):
        """
        Predict Cluster, Score, and Risks for a single company dictionary.
        
        Input:
        {
            "Revenue (USD)": 1000000,
            "Employees Total": 50,
            "Entity Type": "Headquarters",
            "SIC Code": "7371",
            "Parent Company": "Foo Corp" (optional)
        }
        """
        if not self.loaded:
            return {"Error": "Models not loaded"}

        # 1. Clean & Parse Basic Inputs
        rev_raw = self._clean_num(company_data.get("Revenue (USD)"))
        emp_raw = self._clean_num(company_data.get("Employees Total"))
        
        # 2. Imputation (if needed)
        # We need to construct the exact input vector the imputer expects: [Revenue, Employees, Entity_Ord] (scaled)
        entity_type = company_data.get("Entity Type", "Branch")
        entity_ord = self.entity_map.get(entity_type, 1)
        
        # Prepare input vector (using 0/NaN logic same as training)
        # Note: KNN Imputer expects 2D array.
        # Training used: [Revenue, Employees, Entity_Ord] -> Scaled -> Imputed
        
        # Wait, in enhanced_analysis.py:
        # impute_df = pd.DataFrame({'Revenue': ..., 'Employees': ..., 'Entity_Ord': ...})
        # impute_scaled = scaler_impute.fit_transform(impute_df)
        # impute_filled_scaled = knn_imputer.fit_transform(impute_scaled)
        
        # So for a single row, we transform it similarly.
        # If value is 0 or None, treat as NaN for imputation
        rev_input = np.nan if (pd.isna(rev_raw) or rev_raw == 0) else rev_raw
        emp_input = np.nan if (pd.isna(emp_raw) or emp_raw == 0) else emp_raw
        
        input_vector = np.array([[rev_input, emp_input, entity_ord]])
        
        # Scale
        # Note: If NaNs are present, StandardScaler handles them (ignores for mean/std calc but we use transform here)
        # Actually standard scaler propagates NaNs. 
        # KNN Imputer handles the NaNs.
        
        input_scaled = self.scaler_impute.transform(input_vector)
        input_imputed_scaled = self.knn_imputer.transform(input_scaled)
        
        # Inverse Scale back to original units
        input_imputed = self.scaler_impute.inverse_transform(input_imputed_scaled)
        
        rev_clean = input_imputed[0][0]
        emp_clean = input_imputed[0][1]
        
        # 3. Feature Engineering
        # Revenue Per Employee
        rpe = rev_clean / emp_clean if emp_clean > 0 else self.rpe_median
        
        # Entity Score
        entity_score = entity_ord # Same mapping
        
        # Has Parent
        has_parent = 1 if company_data.get("Parent Company") else 0
        
        # Log Transforms
        log_rev = np.log1p(rev_clean)
        log_emp = np.log1p(emp_clean)
        
        # 4. Clustering
        # Features: ['Log_Revenue', 'Log_Employees', 'Entity_Score', 'Has_Parent', 'Revenue_Per_Employee']
        cluster_vec = np.array([[log_rev, log_emp, float(entity_score), float(has_parent), rpe]])
        
        # Handle Inf/NaN (Edge case fallback)
        cluster_vec = np.nan_to_num(cluster_vec)
        
        cluster_scaled = self.scaler_cluster.transform(cluster_vec)
        cluster_id = self.kmeans.predict(cluster_scaled)[0]
        cluster_name = self.cluster_names_map.get(cluster_id, f"Cluster {cluster_id}")
        
        # 5. Anomaly Detection
        # Features: ['Log_Revenue', 'Log_Employees', 'Revenue_Per_Employee']
        iso_vec = np.array([[log_rev, log_emp, rpe]])
        iso_vec = np.nan_to_num(iso_vec) # Safety
        anomaly = self.iso.predict(iso_vec)[0]
        anomaly_label = "Anomaly" if anomaly == -1 else "Normal"
        
        # 6. Lead Scoring
        # Need Data Completeness (Rough estimation for single dict)
        # Fields: Revenue, Employees, SIC, Entity, Region, Country
        completeness_score = 0
        completeness_score += 1 if rev_raw and rev_raw > 0 else 0
        completeness_score += 1 if emp_raw and emp_raw > 0 else 0
        completeness_score += 1 if company_data.get("SIC Code") else 0
        completeness_score += 1 if company_data.get("Entity Type") else 1 # Input usually has it or defaults
        completeness_score += 1 if company_data.get("Region") else 0
        completeness_score += 1 if company_data.get("Country") else 0
        completeness = completeness_score / 6.0
        
        lead_score = self._calc_lead_score(rev_clean, entity_score, rpe, completeness)
        
        if lead_score >= 70: lead_tier = "Priority"
        elif lead_score >= 50: lead_tier = "Hot"
        elif lead_score >= 30: lead_tier = "Warm"
        else: lead_tier = "Cold"
        
        return {
            "Cluster": cluster_name,
            "Lead_Score": lead_score,
            "Lead_Tier": lead_tier,
            "Anomaly": anomaly_label,
            "Revenue_Clean": rev_clean,
            "Employees_Clean": emp_clean
        }

    def _clean_num(self, val):
        if val is None: return np.nan
        if isinstance(val, (int, float)): return val
        try:
            return float(str(val).replace(',', '').strip())
        except:
            return np.nan

    def _calc_lead_score(self, revenue, entity_score, rpe, completeness):
        score = 0
        # Revenue
        if revenue >= 10_000_000: score += 40
        elif revenue >= 1_000_000: score += 30
        elif revenue >= 100_000: score += 20
        elif revenue >= 10_000: score += 10
        else: score += 5
        
        # Entity
        score += entity_score * 6.25
        
        # RPE
        if rpe >= 500_000: score += 20
        elif rpe >= 100_000: score += 15
        elif rpe >= 50_000: score += 10
        else: score += 5
        
        # Data
        score += completeness * 15
        
        return min(100, max(0, score))

if __name__ == "__main__":
    # Simple CLI Test
    evaluator = CompanyEvaluator()
    
    # Test Cases
    test_companies = [
        {
            "Name": "Big Corp HQ",
            "Revenue (USD)": 50000000, 
            "Employees Total": 200, 
            "Entity Type": "Headquarters",
            "Parent Company": None
        },
        {
            "Name": "Tiny Shop",
            "Revenue (USD)": 5000, 
            "Employees Total": 2, 
            "Entity Type": "Single Location"
        },
        {
            "Name": "Suspicious Shell",
            "Revenue (USD)": 10000000, 
            "Employees Total": 0,  # Should be imputed or flagged
            "Entity Type": "Subsidiary"
        }
    ]
    
    print("\n🧐 Testing Inference Engine...")
    for comp in test_companies:
        result = evaluator.predict(comp)
        print(f"\n🏢 {comp['Name']}")
        print(f"   Input: Rev=${comp.get('Revenue (USD)')}, Emp={comp.get('Employees Total')}")
        print(f"   Clean: Rev=${result['Revenue_Clean']:,.0f}, Emp={result['Employees_Clean']:.1f}")
        print(f"   Result: {result['Cluster']} | Score: {result['Lead_Score']:.1f} ({result['Lead_Tier']}) | {result['Anomaly']}")
