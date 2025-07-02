import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

data = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')[['company_location', 'salary_usd']]

print(data.sample(10))

df_stats = data.groupby(['company_location'])['salary_usd'].agg(['mean', 'median', max])
#print(df_stats)

locations = data['company_location'].unique()

print(data['company_location'].value_counts().__len__())

fig, axes = plt.subplots(2, 10, sharex=True, sharey=True) # dùng một trục y chung cho tất cả các subplot
axes = axes.flatten()

plt.ylabel('Distribution')

for ax, t in zip(axes, locations):
    vals = data.loc[data['company_location']==t, 'salary_usd']
    ax.hist(vals, bins=5, edgecolor='black', alpha=0.5)
    ax.set_title(t)
    #ax.set_xticklabels(list(np.arange(90000, 300000, 3)),rotation = 45)
    ax.set_xlabel("Salary")

plt.title("Salary Distributions by Countries", weight="bold", loc='center', y=2.3)
plt.tight_layout()
plt.show()

# --- Statistical ---
groups = [data.loc[data['company_location']==c,'salary_usd'].reset_index(drop=True) for c in locations] #type: ignore
H, p = stats.kruskal(*groups)
print(f"H={H}, p={p:.3e}")


# 2. Compute median and bootstrap CI
def bootstrap_ci(x, stat=np.median, B=2000, alpha=0.05):
    boot_stats = np.array([
        stat(np.random.choice(x, size=len(x), replace=True))
        for _ in range(B)
    ])
    lower, upper = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])
    return lower, upper

summary = (
    data
      .groupby('company_location')['salary_usd']
      .agg(median='median', count='size')
)

cis = (
    data
      .groupby('company_location')['salary_usd']
      .apply(lambda x: pd.Series(bootstrap_ci(x), index=['ci_low','ci_high']))
)

summary = summary.join(cis).sort_values('median', ascending=False).reset_index(drop=False)
print(summary)


plt.figure()
plt.bar(summary['company_location'],summary['median'], color='purple', alpha=0.2, edgecolor='black')
plt.xticks(rotation=45)
plt.show()

