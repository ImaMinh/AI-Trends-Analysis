import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

df_remote = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')[['salary_usd', 'remote_ratio']]

print(df_remote.isna().sum())
df_remote['remote_ratio']
print(df_remote['remote_ratio'].unique())

df_remote['work_mode'] = df_remote['remote_ratio'].map({
    0:   'On-site',
    50:  'Hybrid',
    100: 'Remote'
})

df_stats = df_remote.groupby(['remote_ratio'])['salary_usd'].agg(['mean', 'median', lambda x: x.quantile(.75)-x.quantile(.25)]) \
            .rename(columns={'<lambda_0>':'IQR'})\
            .reset_index(drop=False)

print(df_stats)

remote_ratios = [0, 50, 100]
data = [
    df_remote.loc[df_remote['remote_ratio'] == t, 'salary_usd'] 
    for t in remote_ratios
] # list comprehension

# Run Kruskal-Willis test:
H, p_value = stats.kruskal(*data)
print(f"H = {H: .3f}, p = {p_value: .3f}", "\n")

if p_value < 0.05:
    print("⇒ Significant difference between at least two groups")
else:
    print("⇒ No significant difference detected")
    
# Plotting the box-plot:
plt.figure()
sns.boxplot(
    x='work_mode',
    y='salary_usd',
    data=df_remote,
    order=['On-site', 'Hybrid', 'Remote'],
)
plt.title("Salary Distribution by Work Mode")
plt.xlabel("Work Mode")
plt.ylabel("Salary (USD)")
plt.ylim(0, df_remote['salary_usd'].quantile(0.99))  # zoom in, drop top 1% to reduce huge outliers
plt.show()