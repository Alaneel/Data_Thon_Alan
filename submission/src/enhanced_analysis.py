"""
Company Intelligence Analysis
SDS Datathon 2026 - Final Production Version

Updates:
- Integrated 'Company Age', 'Market Value', 'IT Spend', 'Domestic Ultimate'
- Fixed Clustering K=5 (Tier 1-5 Logic)
- Updated Entity Score Mapping (Parent=3)
- Enhanced Lead Scoring Model (v2)
- Full Contact Info Export
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore')

# Project Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

print("=" * 60)
print("🚀 COMPANY INTELLIGENCE PIPELINE (FINAL VERSION)")
print("   SDS Datathon 2026")
print("=" * 60)

# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
print("\n[1/7] Loading & Cleaning Data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'champions_group_data.csv'), on_bad_lines='skip')
print(f"   Loaded {len(df):,} rows.")

# Helpers
def clean_numeric(col_name):
    """Convert to numeric, keeping NaN as NaN"""
    if col_name not in df.columns: return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col_name].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

def clean_boolean(val):
    """Clean Yes/No/1/0"""
    s = str(val).lower()
    if s in ['yes', 'true', '1']: return 1
    return 0

# A. Basic Numeric Cleaning
revenue_raw = clean_numeric('Revenue (USD)')
employees_raw = clean_numeric('Employees Total')

df['Is_Revenue_Missing'] = revenue_raw.isna() | (revenue_raw == 0)
df['Is_Employees_Missing'] = employees_raw.isna() | (employees_raw == 0)

# B. KNN Imputation (Log Transformed)
print("   Running KNN Imputation...")
impute_df = pd.DataFrame({
    'Revenue': np.log1p(revenue_raw.replace(0, np.nan)),
    'Employees': np.log1p(employees_raw.replace(0, np.nan))
})

# Add Entity for context in imputation
entity_map_impute = {'Headquarters': 4, 'Parent': 3, 'Single Location': 3, 'Subsidiary': 2, 'Branch': 1}
impute_df['Entity_Ord'] = df['Entity Type'].map(entity_map_impute).fillna(1)

scaler_impute = StandardScaler()
impute_scaled = scaler_impute.fit_transform(impute_df)
knn_imputer = KNNImputer(n_neighbors=5)
impute_filled = knn_imputer.fit_transform(impute_scaled)
impute_final = scaler_impute.inverse_transform(impute_filled)

# Restore real values
df['Revenue_USD_Clean'] = np.expm1(impute_final[:, 0])
df['Employees_Total_Clean'] = np.expm1(impute_final[:, 1])

# Clean other metrics
df['Employees_Site_Clean'] = clean_numeric('Employees Single Site').fillna(0)
df['Corporate_Family_Size'] = clean_numeric('Corporate Family Members').fillna(0)

# ============================================================
# 2. FEATURE ENGINEERING (The "Brain")
# ============================================================
print("\n[2/7] Engineering Advanced Features...")

# 1. Hierarchy Power (Domestic Ultimate)
if 'Is Domestic Ultimate' in df.columns:
    df['Is_Domestic_Ultimate_Clean'] = df['Is Domestic Ultimate'].apply(clean_boolean)
else:
    df['Is_Domestic_Ultimate_Clean'] = 0

# 2. Company Age
current_year = 2026
df['Year_Found_Clean'] = pd.to_numeric(df['Year Found'], errors='coerce')
df['Company_Age'] = (current_year - df['Year_Found_Clean']).clip(0, 200)
df['Company_Age'] = df['Company_Age'].fillna(df['Company_Age'].median())

# 3. Market Value
df['Market_Value_Clean'] = clean_numeric('Market Value (USD)').fillna(0)

# 4. IT Spend & Tech Maturity
df['IT_Spend_Clean'] = clean_numeric('IT Spend').fillna(0)
df['IT_Spend_Per_Emp'] = df['IT_Spend_Clean'] / df['Employees_Total_Clean'].replace(0, np.nan)
df['IT_Spend_Per_Emp'] = df['IT_Spend_Per_Emp'].fillna(0)

# 5. Entity Score (Updated Mapping: Parent=3)
entity_map = {'Parent': 3, 'Subsidiary': 2, 'Branch': 1}
df['Entity_Score'] = df['Entity Type'].map(entity_map).fillna(1)

# 6. Structure & Efficiency
df['Has_Parent'] = df['Parent Company'].notna().astype(int)
df['SIC_2Digit'] = df['SIC Code'].astype(str).str[:2]
df['Revenue_Per_Employee'] = df['Revenue_USD_Clean'] / df['Employees_Total_Clean'].replace(0, np.nan)
df['Revenue_Per_Employee'] = df['Revenue_Per_Employee'].fillna(df['Revenue_Per_Employee'].median())

# 7. Log Transforms
df['Log_Revenue'] = np.log1p(df['Revenue_USD_Clean'])
df['Log_Employees'] = np.log1p(df['Employees_Total_Clean'])

# 8. Data Completeness
cols_to_check = ['Is_Revenue_Missing', 'Is_Employees_Missing', 'SIC Code', 'Entity Type', 'Region', 'Country', 'Year Found']
df['Data_Completeness'] = (
    (~df['Is_Revenue_Missing']).astype(int) + 
    (~df['Is_Employees_Missing']).astype(int) + 
    df['SIC Code'].notna().astype(int) + 
    df['Entity Type'].notna().astype(int) + 
    df['Region'].notna().astype(int) + 
    df['Country'].notna().astype(int) +
    df['Year Found'].notna().astype(int)
) / 7

print("   Added: Age, Market Value, IT Spend, Domestic Ultimate, New Entity Score.")

# ============================================================
# 3. INDUSTRY BENCHMARKING
# ============================================================
print("\n[3/7] Calculating Benchmarks...")
industry_stats = df.groupby('SIC_2Digit').agg({
    'Revenue_USD_Clean': 'median',
    'Employees_Total_Clean': 'median'
}).reset_index()
industry_stats.columns = ['SIC_2Digit', 'Ind_Rev_Med', 'Ind_Emp_Med']

df = df.merge(industry_stats, on='SIC_2Digit', how='left')

df['Revenue_vs_Industry'] = np.where(
    df['Ind_Rev_Med'] > 0,
    (df['Revenue_USD_Clean'] / df['Ind_Rev_Med'] - 1) * 100, 0
).clip(-100, 500)

df['Employees_vs_Industry'] = np.where(
    df['Ind_Emp_Med'] > 0,
    (df['Employees_Total_Clean'] / df['Ind_Emp_Med'] - 1) * 100, 0
).clip(-100, 500)

# ============================================================
# 4. CLUSTERING (Fixed k=5)
# ============================================================
print("\n[4/7] Segmentation (Tier 1-5)...")

cluster_features = [
    'Log_Revenue', 'Log_Employees', 'Entity_Score', 'Has_Parent', 
    'Revenue_Per_Employee', 'Company_Age', 'Is_Domestic_Ultimate_Clean'
]
X_cluster = df[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(df[cluster_features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Force k=5 as determined in analysis
best_k = 5
print(f"   Applying K-Means with k={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Naming Logic (Parent=3)
def get_cluster_names(df_data):
    profiles = df_data.groupby('Cluster').agg({
        'Revenue_USD_Clean': 'median',
        'Entity_Score': lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()
    }).reset_index()
    profiles['Rank'] = profiles['Revenue_USD_Clean'].rank(ascending=False, method='min').astype(int)
    
    names_map = {}
    for _, row in profiles.iterrows():
        tier = f"Tier {int(row['Rank'])}"
        score = row['Entity_Score']
        
        # New Logic: Parent(3), Sub(2), Branch(1) + SMB check
        if row['Rank'] == 1: structure = "Global HQ"
        elif row['Rank'] >= 4 and score >= 2.8: structure = "Local HQ"
        elif score >= 2.8: structure = "Parent HQ"
        elif score >= 1.8: structure = "Subsidiary"
        else: structure = "Branch"
        
        names_map[row['Cluster']] = f"{tier} {structure}"
    return names_map

cluster_map = get_cluster_names(df)
df['Cluster_Name'] = df['Cluster'].map(cluster_map)

# Print Distribution
print("   Cluster Distribution:")
vc = df['Cluster_Name'].value_counts()
for name, count in vc.items():
    print(f"      {name:<25}: {count:,}")

# ============================================================
# 5. LEAD SCORING (Enhanced)
# ============================================================
print("\n[5/7] Scoring Leads...")

def calculate_lead_score_v2(row):
    score = 0
    # 1. Revenue (35)
    rev = row['Revenue_USD_Clean']
    if rev >= 1e8: score += 35
    elif rev >= 1e7: score += 25
    elif rev >= 1e6: score += 15
    elif rev >= 1e5: score += 5
    
    # 2. Power (20) - Domestic Ultimate Bonus
    if row['Is_Domestic_Ultimate_Clean'] == 1: score += 15
    else: score += row['Entity_Score'] * 3
    
    # 3. Tech/Efficiency (20)
    if row['Revenue_Per_Employee'] >= 5e5: score += 10
    if row['IT_Spend_Per_Emp'] > 1000: score += 10
    elif row['IT_Spend_Per_Emp'] > 0: score += 5
    
    # 4. Market Value (15)
    if row['Market_Value_Clean'] > 0: score += 15
    
    # 5. Stability (10)
    age = row['Company_Age']
    if 3 <= age <= 10: score += 10
    elif age > 10: score += 5
    
    # Penalty
    if row['Data_Completeness'] < 0.5: score *= 0.8
    
    return min(100, max(0, score))

df['Lead_Score'] = df.apply(calculate_lead_score_v2, axis=1)
df['Lead_Tier'] = pd.cut(df['Lead_Score'], bins=[0, 30, 50, 75, 100], labels=['Cold', 'Warm', 'Hot', 'Priority'])

print(f"   Priority Leads: {(df['Lead_Tier']=='Priority').sum():,}")

# ============================================================
# 6. RISK DETECTION (Fixed Logic)
# ============================================================
print("\n[6/7] Detecting Risks & Anomalies...")

# 1. Shell Company Risk
# Logic: Revenue is very high (>100,000), but the original data shows the number of employees as empty or 0 (Is_Employees_Missing=True).
df['Risk_Shell'] = (df['Revenue_USD_Clean'] > 100_000) & (df['Is_Employees_Missing'] == 1)

# 2. Data Quality Risk
# Logic: Many key fields are missing
df['Risk_DataQuality'] = df['Data_Completeness'] < 0.5

# 3. Orphan Subsidiary Risk
# Logic: The type is subsidiary, but there is no parent company link.
df['Risk_OrphanSub'] = (df['Entity Type'] == 'Subsidiary') & (df['Has_Parent'] == 0)

# 4. Statistical Anomalies (Isolation Forest)
# This is an unsupervised algorithm that forces the identification of the 5% of the most "strange" data.
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_Label'] = iso.fit_predict(X_scaled) # -1=Anomaly, 1=Normal
df['Anomaly_Label'] = df['Anomaly_Label'].map({1: 'Normal', -1: 'Anomaly'})
df['Anomaly_Score'] = iso.decision_function(X_scaled)

# Combined Risk Flags
# If any risk is triggered, the Flag will be greater than 0.
risk_anomaly = (df['Anomaly_Label'] == 'Anomaly').astype(int)
df['Risk_Flags'] = (df['Risk_Shell'].astype(int) + 
                    df['Risk_DataQuality'].astype(int) + 
                    df['Risk_OrphanSub'].astype(int) + 
                    risk_anomaly)

print(f"   Risk Analysis:")
print(f"      Shell Companies Detected: {df['Risk_Shell'].sum():,}")
print(f"      Anomalies Detected: {risk_anomaly.sum():,}")

# ============================================================
# 7. EXPORT & SAVE (Full Stack)
# ============================================================
print("\n[7/7] Saving Artifacts...")
os.makedirs(MODELS_DIR, exist_ok=True)

# Save Models
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_cluster.joblib'))
joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans.joblib'))
joblib.dump(iso, os.path.join(MODELS_DIR, 'iso_forest.joblib'))
joblib.dump(scaler_impute, os.path.join(MODELS_DIR, 'scaler_impute.joblib'))
joblib.dump(knn_imputer, os.path.join(MODELS_DIR, 'knn_imputer.joblib'))
print("   Models saved.")

# Export CSV 
output_cols = [
    # Identity
    'DUNS Number ', 'Company Sites', 'Website', 
    'Address Line 1', 'City', 'State', 'Region', 'Country', 'Phone Number', 
    
    # Firmographics
    'SIC Code', 'SIC Description', 'Year Found', 'Company_Age',
    
    # Hierarchy
    'Entity Type', 'Entity_Score', 'Parent Company', 
    'Is Domestic Ultimate', 'Is_Domestic_Ultimate_Clean',
    
    # Metrics
    'Revenue (USD)', 'Revenue_USD_Clean', 
    'Employees Total', 'Employees_Total_Clean',
    'Market_Value_Clean', 'IT_Spend_Clean',
    
    # Analysis
    'Cluster', 'Cluster_Name', 
    'Lead_Score', 'Lead_Tier',
    'Revenue_vs_Industry',
    
    # Risk
    'Risk_Flags', 'Risk_Shell', 'Anomaly_Label'
]



valid_cols = [c for c in output_cols if c in df.columns]
df[valid_cols].to_csv(os.path.join(DATA_DIR, 'company_segmentation_results.csv'), index=False)

print(f"✅ Pipeline Complete. Results saved to {DATA_DIR}/company_segmentation_results.csv")