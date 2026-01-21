import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading data...")
df = pd.read_csv('data/company_segmentation_results.csv')

# Re-engineer features
df['Log_Revenue'] = np.log1p(df['Revenue_USD_Clean'])
df['Log_Employees'] = np.log1p(df['Employees_Total_Clean'])
df['Revenue_Per_Employee'] = df['Revenue_USD_Clean'] / df['Employees_Total_Clean'].replace(0, 1)

features = ['Log_Revenue', 'Log_Employees', 'Entity_Score', 
            'Revenue_Per_Employee', 'Company_Age', 'Is_Domestic_Ultimate_Clean']

# Standardize features for Clustering
X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== 1. Clustering Stability Analysis (Bootstrap) ===")
n_bootstraps = 20
k = 5
aris = []

# Base clustering
kmeans_base = KMeans(n_clusters=k, random_state=42, n_init=10)
base_labels = kmeans_base.fit_predict(X_scaled)

print(f"Running {n_bootstraps} bootstrap iterations...")
for i in range(n_bootstraps):
    # Resample indices
    indices = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
    X_boot = X_scaled[indices]
    
    # Cluster bootstrap sample
    kmeans_boot = KMeans(n_clusters=k, random_state=i, n_init=5)
    _ = kmeans_boot.fit(X_boot) # Fit on bootstrap
    
    # Predict labels for original data using bootstrap centers
    # This aligns the bootstrap clusters to original space
    matched_labels = kmeans_boot.predict(X_scaled)
    
    # Calculate ARI between Base Labels and Bootstrap-derived Labels
    score = adjusted_rand_score(base_labels, matched_labels)
    aris.append(score)

mean_ari = np.mean(aris)
print(f"Mean Adjusted Rand Index (Stability): {mean_ari:.4f}")

# Plot Stability
plt.figure(figsize=(8, 5))
sns.histplot(aris, bins=10, color='#4A90A4', kde=True)
plt.axvline(mean_ari, color='red', linestyle='--', label=f'Mean ARI: {mean_ari:.2f}')
plt.title(f'Clustering Stability (Mean ARI: {mean_ari:.2f})', fontsize=12, fontweight='bold')
plt.xlabel('Adjusted Rand Index')
plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/clustering_stability.png', dpi=150)
print("Saved: clustering_stability.png")


print("\n=== 2. Permutation Feature Importance (Cluster Predictors) ===")
# We interpret clusters by training a classifier to predict Cluster ID
# and checking which features are most important.
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clust_labels = kmeans_base.labels_
X_train, X_test, y_train, y_test = train_test_split(X, clust_labels, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Cluster Prediction Accuracy: {acc:.2%}")

# Permutation Importance
perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

# Plot
plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[sorted_idx], perm_importance.importances_mean[sorted_idx], color='#7B68EE')
plt.title('Feature Importance for Cluster Definition', fontsize=12, fontweight='bold')
plt.xlabel('Permutation Importance (Mean Accuracy Decrease)')
plt.tight_layout()
plt.savefig('reports/figures/cluster_feature_importance.png', dpi=150)
print("Saved: cluster_feature_importance.png")
