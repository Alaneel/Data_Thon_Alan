"""
Enhanced Company Intelligence Analysis
SDS Datathon 2026 - Competitive Version

This script performs:
1. Multi-dimensional clustering (Revenue, Employees, Entity Type, Industry)
2. B2B Lead Score calculation (0-100)
3. Industry benchmarking
4. Risk signal detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 ENHANCED COMPANY INTELLIGENCE ANALYSIS")
print("   SDS Datathon 2026 - Competitive Version")
print("=" * 60)

# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
print("\n📂 Loading data...")
df = pd.read_csv('champions_group_data.csv')
print(f"   Loaded {len(df):,} companies")

# Clean numeric columns - distinguish between true zeros and missing values
def clean_numeric(col):
    """Convert to numeric, keeping NaN as NaN (not filling with 0)"""
    return pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

# First, get raw cleaned values (with NaN for missing)
revenue_raw = clean_numeric('Revenue (USD)')
employees_raw = clean_numeric('Employees Total')

# Create missing value indicator columns (before filling)
# Treat 0 as missing for the purpose of imputation (assuming 0 revenue/employees is unlikely for active companies)
df['Is_Revenue_Missing'] = revenue_raw.isna() | (revenue_raw == 0)
df['Is_Employees_Missing'] = employees_raw.isna() | (employees_raw == 0)

# Prepare for KNN Imputation
print("   Performing KNN Imputation for missing values...")
impute_df = pd.DataFrame({
    'Revenue': revenue_raw.replace(0, np.nan),
    'Employees': employees_raw.replace(0, np.nan)
})

# Add Entity Type as ordinal feature to help imputation
entity_map = {'Headquarters': 4, 'Single Location': 3, 'Subsidiary': 2, 'Branch': 1}
impute_df['Entity_Ord'] = df['Entity Type'].map(entity_map).fillna(1)

# Scale before KNN
scaler_impute = StandardScaler()
impute_scaled = scaler_impute.fit_transform(impute_df)

# Impute (using 5 neighbors)
knn_imputer = KNNImputer(n_neighbors=5)
impute_filled_scaled = knn_imputer.fit_transform(impute_scaled)

# Inverse transform to get original scale
impute_filled = scaler_impute.inverse_transform(impute_filled_scaled)

# Assign back to dataframe
df['Revenue_USD_Clean'] = impute_filled[:, 0]
df['Employees_Total_Clean'] = impute_filled[:, 1]
df['Employees_Site_Clean'] = clean_numeric('Employees Single Site').fillna(0)
df['Corporate_Family_Size'] = clean_numeric('Corporate Family Members').fillna(0)

print(f"   Revenue missing/zero: {df['Is_Revenue_Missing'].sum():,} ({df['Is_Revenue_Missing'].mean()*100:.1f}%)")
print(f"   Employees missing/zero: {df['Is_Employees_Missing'].sum():,} ({df['Is_Employees_Missing'].mean()*100:.1f}%)")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n🔧 Engineering features...")

# Entity Type Score (decision-making power)
entity_map = {
    'Headquarters': 4,
    'Single Location': 3,
    'Subsidiary': 2,
    'Branch': 1
}
df['Entity_Score'] = df['Entity Type'].map(entity_map).fillna(1)

# Has Parent Company
df['Has_Parent'] = df['Parent Company'].notna().astype(int)

# Industry Sector (2-digit SIC)
df['SIC_2Digit'] = df['SIC Code'].astype(str).str[:2]

# Revenue per Employee (productivity indicator)
# Handle potential division by zero if KNN somehow left 0s (unlikely but safe) or if imputation resulted in very small numbers
df['Revenue_Per_Employee'] = df['Revenue_USD_Clean'] / df['Employees_Total_Clean'].replace(0, np.nan)

# Fill any remaining NaNs (e.g. from 0 employees) with median
rpe_median = df['Revenue_Per_Employee'].median()
df['Revenue_Per_Employee'] = df['Revenue_Per_Employee'].fillna(rpe_median)

# Log-transformed features for better clustering
# Use small offset for zero values to avoid log(0) issues
df['Log_Revenue'] = np.log1p(df['Revenue_USD_Clean'])
df['Log_Employees'] = np.log1p(df['Employees_Total_Clean'])

# Data Completeness Score - now considers zero values as incomplete
important_fields = ['Revenue (USD)', 'Employees Total', 'SIC Code', 'Entity Type', 'Region', 'Country']
# A field is considered complete if it's not null AND not zero (for numeric fields)
df['Data_Completeness'] = (
    (~df['Is_Revenue_Missing']).astype(int) + 
    (~df['Is_Employees_Missing']).astype(int) + 
    df['SIC Code'].notna().astype(int) + 
    df['Entity Type'].notna().astype(int) + 
    df['Region'].notna().astype(int) + 
    df['Country'].notna().astype(int)
) / 6

print(f"   Created {6} new features")

# ============================================================
# 3. INDUSTRY BENCHMARKING
# ============================================================
print("\n📊 Calculating industry benchmarks...")

# Group by industry sector
industry_stats = df.groupby('SIC_2Digit').agg({
    'Revenue_USD_Clean': ['median', 'mean', 'std', 'count'],
    'Employees_Total_Clean': ['median', 'mean', 'std']
}).reset_index()

industry_stats.columns = ['SIC_2Digit', 'Ind_Revenue_Median', 'Ind_Revenue_Mean', 'Ind_Revenue_Std', 'Ind_Count',
                          'Ind_Employees_Median', 'Ind_Employees_Mean', 'Ind_Employees_Std']

df = df.merge(industry_stats, on='SIC_2Digit', how='left')

# Calculate percentiles within industry
def industry_percentile(row, col):
    if pd.isna(row['Ind_Revenue_Median']):
        return 50  # Default to median if no industry data
    industry_data = df[df['SIC_2Digit'] == row['SIC_2Digit']][col]
    if len(industry_data) == 0:
        return 50
    return (industry_data < row[col]).sum() / len(industry_data) * 100

# Simplified: Compare to industry median
df['Revenue_vs_Industry'] = np.where(
    df['Ind_Revenue_Median'] > 0,
    (df['Revenue_USD_Clean'] / df['Ind_Revenue_Median'] - 1) * 100,
    0
).clip(-100, 500)

df['Employees_vs_Industry'] = np.where(
    df['Ind_Employees_Median'] > 0,
    (df['Employees_Total_Clean'] / df['Ind_Employees_Median'] - 1) * 100,
    0
).clip(-100, 500)

print(f"   Benchmarked against {len(industry_stats)} industry sectors")

# ============================================================
# 4. MULTI-DIMENSIONAL CLUSTERING
# ============================================================
print("\n🎯 Performing multi-dimensional clustering...")

# Prepare clustering features
cluster_features = ['Log_Revenue', 'Log_Employees', 'Entity_Score', 'Has_Parent', 'Revenue_Per_Employee']
X_cluster = df[cluster_features].copy()

# Handle infinities and NaNs
X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan)
X_cluster = X_cluster.fillna(X_cluster.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find optimal k using silhouette
best_k = 4
best_score = -1
for k in range(3, 8):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_temp)
    if score > best_score:
        best_score = score
        best_k = k

print(f"   Optimal clusters: {best_k} (Silhouette: {best_score:.4f})")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Generate cluster names based on characteristics
cluster_profiles = df.groupby('Cluster').agg({
    'Revenue_USD_Clean': 'median',
    'Employees_Total_Clean': 'median',
    'Entity_Score': 'mean',
    'Has_Parent': 'mean'
}).round(2)

def get_cluster_context(df_data, k):
    """
    Dynamically names clusters using MODE (Most Frequent) for Entity Type
    to avoid averaging dilution.
    """
    # 1. Calculate Centroids & Dominant Type
    # 使用 lambda x: x.mode()[0] 来提取众数（该组中最常见的类型）
    profiles = df_data.groupby('Cluster').agg({
        'Revenue_USD_Clean': 'median',
        'Employees_Total_Clean': 'median',
        'Entity_Score': lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean() 
    }).reset_index()
    
    # 2. Assign Tiers based on Revenue Rank (1 = Highest Revenue)
    profiles['Rank'] = profiles['Revenue_USD_Clean'].rank(ascending=False, method='min').astype(int)
    
    # 3. Generate Dynamic Names
    def generate_name(row):
        # Tier Part
        tier = f"Tier {row['Rank']}"
        
        # Structure Part - 现在这是众数，是整数 (4, 3, 2, 1)
        score = row['Entity_Score']
        
        if score >= 4:       # 主要是总部
            structure = "HQ"
        elif score >= 3:     # 主要是独立地点
            structure = "Indep/Strategic" 
        elif score >= 2:     # 主要是子公司
            structure = "Subsidiary"
        else:                # 主要是分支机构
            structure = "Branch"
            
        return f"{tier} {structure}"

    profiles['Cluster_Name'] = profiles.apply(generate_name, axis=1)
    
    return profiles.set_index('Cluster')['Cluster_Name'].to_dict()

cluster_names_map = get_cluster_context(df, best_k)
df['Cluster_Name'] = df['Cluster'].map(cluster_names_map)

print(f"   Cluster distribution (Dynamic Naming):")
cluster_counts = df['Cluster'].value_counts()
for cluster_id in sorted(df['Cluster'].unique()):
    name = cluster_names_map[cluster_id]
    count = cluster_counts[cluster_id]
    # Get median revenue for context in print output
    med_rev = df[df['Cluster'] == cluster_id]['Revenue_USD_Clean'].median()
    print(f"      {name:<25} (ID {cluster_id}): {count:,} companies | Median Rev: ${med_rev:,.0f}")

print(f"   Cluster distribution:")


# ============================================================
# 5. B2B LEAD SCORE CALCULATION
# ============================================================
print("\n💰 Calculating B2B Lead Scores...")

def calculate_lead_score(row):
    """
    B2B Lead Score (0-100) based on:
    - Revenue potential (40%)
    - Decision-making power (25%)
    - Growth indicators (20%)
    - Data quality (15%)
    """
    score = 0
    
    # 1. Revenue Potential (0-40 points)
    revenue = row['Revenue_USD_Clean']
    if revenue >= 10_000_000:
        score += 40
    elif revenue >= 1_000_000:
        score += 30
    elif revenue >= 100_000:
        score += 20
    elif revenue >= 10_000:
        score += 10
    else:
        score += 5
    
    # 2. Decision-Making Power (0-25 points)
    entity_score = row['Entity_Score']
    score += entity_score * 6.25  # 4 * 6.25 = 25 max
    
    # 3. Growth Indicators (0-20 points)
    # Higher revenue-per-employee = more efficient/growth potential
    rpe = row['Revenue_Per_Employee']
    if rpe >= 500_000:
        score += 20
    elif rpe >= 100_000:
        score += 15
    elif rpe >= 50_000:
        score += 10
    else:
        score += 5
    
    # 4. Data Quality (0-15 points)
    score += row['Data_Completeness'] * 15
    
    return min(100, max(0, score))

df['Lead_Score'] = df.apply(calculate_lead_score, axis=1)

# Lead Score tiers
df['Lead_Tier'] = pd.cut(
    df['Lead_Score'],
    bins=[0, 30, 50, 70, 100],
    labels=['Cold', 'Warm', 'Hot', 'Priority']
)

print(f"   Lead Score distribution:")
print(f"      Priority (70-100): {(df['Lead_Score'] >= 70).sum():,}")
print(f"      Hot (50-70): {((df['Lead_Score'] >= 50) & (df['Lead_Score'] < 70)).sum():,}")
print(f"      Warm (30-50): {((df['Lead_Score'] >= 30) & (df['Lead_Score'] < 50)).sum():,}")
print(f"      Cold (0-30): {(df['Lead_Score'] < 30).sum():,}")

# ============================================================
# 6. RISK SIGNAL DETECTION
# ============================================================
print("\n⚠️ Detecting risk signals...")

# Risk 1: Shell company risk (high revenue, zero employees)
df['Risk_Shell'] = (df['Revenue_USD_Clean'] > 100000) & (df['Employees_Total_Clean'] == 0)

# Risk 2: Data inconsistency (missing critical fields)
df['Risk_DataQuality'] = df['Data_Completeness'] < 0.5

# Risk 3: Orphan subsidiary (Subsidiary type but no parent linkage)
df['Risk_OrphanSub'] = (df['Entity Type'] == 'Subsidiary') & (df['Has_Parent'] == 0)

# Isolation Forest for statistical anomalies
iso_features = ['Log_Revenue', 'Log_Employees', 'Revenue_Per_Employee']
X_iso = df[iso_features].replace([np.inf, -np.inf], np.nan)
# Use median instead of 0 for missing values (0 would skew anomaly detection)
X_iso = X_iso.fillna(X_iso.median())
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso.fit_predict(X_iso)
df['Anomaly_Label'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Combined risk score
df['Risk_Flags'] = df['Risk_Shell'].astype(int) + df['Risk_DataQuality'].astype(int) + df['Risk_OrphanSub'].astype(int) + (df['Anomaly'] == -1).astype(int)

print(f"   Risk signals detected:")
print(f"      Shell company risk: {df['Risk_Shell'].sum():,}")
print(f"      Data quality risk: {df['Risk_DataQuality'].sum():,}")
print(f"      Orphan subsidiary: {df['Risk_OrphanSub'].sum():,}")
print(f"      Statistical anomaly: {(df['Anomaly'] == -1).sum():,}")

# ============================================================
# 7. SAVE MODELS & ARTIFACTS
# ============================================================
print("\n💾 Saving models & artifacts...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# 1. Preprocessing Objects
joblib.dump(scaler_impute, f"{model_dir}/scaler_impute.joblib")
joblib.dump(knn_imputer, f"{model_dir}/knn_imputer.joblib")
print(f"   Saved imputer objects to {model_dir}/")

# 2. Industry Stats
industry_stats.to_csv(f"{model_dir}/industry_stats.csv", index=False)
print(f"   Saved industry stats to {model_dir}/industry_stats.csv")

# 3. Global Stats (Medians, Defaults)
global_stats = {
    "rpe_median": float(rpe_median),
    "cluster_names_map": {int(k): v for k, v in cluster_names_map.items()},
    "best_k": int(best_k),
    "entity_map": entity_map
}
with open(f"{model_dir}/global_stats.json", "w") as f:
    json.dump(global_stats, f, indent=4)
print(f"   Saved global stats to {model_dir}/global_stats.json")

# 4. Clustering Models
joblib.dump(scaler, f"{model_dir}/scaler_cluster.joblib")
joblib.dump(kmeans, f"{model_dir}/kmeans.joblib")
print(f"   Saved clustering models to {model_dir}/")

# 5. Isolation Forest
joblib.dump(iso, f"{model_dir}/isolation_forest.joblib")
print(f"   Saved anomaly detection model to {model_dir}/")


# ============================================================
# 8. SAVE ENHANCED RESULTS
# ============================================================
print("\n💾 Saving enhanced results...")

output_cols = [
    'DUNS Number ', 'Company Sites', 'Country', 'Region', 'Entity Type',
    'SIC Code', 'SIC Description', 'Employees_Total_Clean', 'Revenue_USD_Clean',
    'Is_Revenue_Missing', 'Is_Employees_Missing',  # New: missing value indicators
    'Entity_Score', 'Has_Parent', 'Revenue_Per_Employee', 'Data_Completeness',
    'Revenue_vs_Industry', 'Employees_vs_Industry',
    'Cluster', 'Cluster_Name',
    'Lead_Score', 'Lead_Tier',
    'Anomaly_Label', 'Risk_Shell', 'Risk_DataQuality', 'Risk_OrphanSub', 'Risk_Flags'
]

df_output = df[[c for c in output_cols if c in df.columns]]
df_output.to_csv('company_segmentation_results.csv', index=False)

print(f"   Saved {len(df_output):,} companies to company_segmentation_results.csv")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("📈 ENHANCED ANALYSIS SUMMARY")
print("=" * 60)
print(f"   Total Companies: {len(df):,}")
print(f"   Clusters: {best_k} (Silhouette: {best_score:.4f})")
print(f"   Industry Sectors: {len(industry_stats)}")
print(f"   Priority Leads: {(df['Lead_Score'] >= 70).sum():,}")
print(f"   High-Risk Entities: {(df['Risk_Flags'] >= 2).sum():,}")
print("=" * 60)
print("✅ Enhanced analysis complete!")
