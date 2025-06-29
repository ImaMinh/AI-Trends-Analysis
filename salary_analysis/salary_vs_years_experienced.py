import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')

print(df.shape, '\n')
print(df.head)
print(df.tail)
print(df.sample(10))

salary = df['salary_usd']
years_experience = df['years_experience']

print(years_experience.isna().sum())

plt.scatter(salary, years_experience, alpha = 0.2, color='purple')
plt.grid(True)
plt.title('salary vs years of experience')
plt.xlabel('salary_usd')
plt.ylabel('years_experience')


plt.figure(figsize=(6,4))
sns.regplot(x='years_experience', y='salary_usd', data=df,
            scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
plt.xlabel('Years of Experience')
plt.ylabel('Salary (USD)')
plt.title('Salary vs. Years of Experience')
plt.show()


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