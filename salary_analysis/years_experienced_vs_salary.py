import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')

print(df.shape, '\n')
print(df.head)
print(df.tail)
print(df.sample(10))

salary = df['salary_usd']
years_experience = df['years_experience']

print(years_experience.isna().sum())


plt.figure(figsize=(6,4))
sns.regplot(x='years_experience', y='salary_usd', data=df,
            scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
plt.xlabel('Years of Experience')
plt.ylabel('Salary (USD)')
plt.title('Salary vs. Years of Experience')

# ——— 2) Compute “nice” bin edges ———
exp_step = 4
sal_step = 10000

# experience: from 0 up to next multiple of exp_step
max_exp = df['years_experience'].max()
exp_bins = np.arange(0, np.ceil(max_exp/exp_step)*exp_step + exp_step, exp_step)

# salary: from 0 up to next multiple of sal_step
max_sal = df['salary_usd'].max()
sal_bins = np.arange(0, np.ceil(max_sal/sal_step)*sal_step + sal_step, sal_step)

# ——— 3) Bin the data ———
df['exp_group'] = pd.cut(df['years_experience'], bins=exp_bins, right=False)
df['sal_group'] = pd.cut(df['salary_usd'],       bins=sal_bins, right=False)

# ——— 4) Pivot to get counts ———
heatmap_data = (
    df
    .groupby(['sal_group','exp_group'])
    .size()
    .unstack(fill_value=0)
)

# ——— 5) Plot ———
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    cbar_kws={'label': 'Count'},
    linewidths=0.5
)
plt.title("Heatmap of Salary vs. Years of Experience")
plt.xlabel("Years of Experience (bins of 4 years)")
plt.ylabel("Salary (bins of $10 000)")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# --- Compute Numerical Correlation Coeffiecient: ---

# Since our distribution is Monotonic, computing the Spearman rho is sufficient. This can be done using corr method of pandas.
rho = df['years_experience'].corr(df['salary_usd'])

print(f"ρ = {rho:.2f}")

if (rho >= 0.7):
    print("Strong positive correlation")
elif (rho >= 0.3):
    print("Moderate positive correlation (noticeable trend, but with scatter)")
elif (rho >= 0.1):
    print("Weak positive correlation (little association, lots of noise)")
elif( rho > -0.1):
    print("Negligible correlation (no meaningful linear relationship)")
elif (rho > -0.3):
    print("Weak negative correlation (slight inverse tendency)")
elif (rho > -0.7):
    print("Moderate negative correlation (clear inverse trend, but not perfect)")
else:
    print("Strong negative correlation")

plt.show()