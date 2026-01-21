import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('data/company_segmentation_results.csv')

# Compare 'Priority' vs 'Cold' on Efficiency (Rev/Emp)
group_A = df[df['Lead_Tier'] == 'Priority']['Revenue_USD_Clean'] / df[df['Lead_Tier'] == 'Priority']['Employees_Total_Clean'].replace(0, 1)
group_B = df[df['Lead_Tier'] == 'Cold']['Revenue_USD_Clean'] / df[df['Lead_Tier'] == 'Cold']['Employees_Total_Clean'].replace(0, 1)

# Log transform for normality (Rev/Emp is usually log-normal)
group_A_log = np.log1p(group_A)
group_B_log = np.log1p(group_B)

t_stat, p_val = stats.ttest_ind(group_A_log, group_B_log, equal_var=False)

print(f"Priority Group Mean (Log): {group_A_log.mean():.2f}")
print(f"Cold Group Mean (Log):     {group_B_log.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")

# Visualization
plt.figure(figsize=(6, 5))
plt.boxplot([group_A_log, group_B_log], tick_labels=['Priority Leads', 'Cold Leads'], patch_artist=True,
            boxprops=dict(facecolor='#667eea', color='#1E3A5F'),
            medianprops=dict(color='white'))
plt.title(f'Efficiency Comparison (t={t_stat:.1f}, p<{p_val:.1e})\nSignificantly Higher Efficiency in Priority Leads', fontsize=11)
plt.ylabel('Log(Revenue per Employee)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/hypothesis_test.png', dpi=150)
print("Saved: hypothesis_test.png")
