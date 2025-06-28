import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')
df_comp_size = df[['salary_usd','company_size']]
print(df_comp_size)

print(df_comp_size.shape)
print(df_comp_size['company_size'].unique())

print(df_comp_size.isnull().sum())

df_comp_size = df_comp_size.groupby('company_size')

df_stats = df_comp_size['salary_usd'].agg(['count', 'mean', 'median']).reset_index()

print("\n", df_stats)

# Plotting:

company_size = df_stats['company_size']
mean = df_stats['mean']
median = df_stats['median']

x = np.arange(len(company_size))

fig, ax1 = plt.subplots() # plt.subplot vs plt.subplots???

width = 0.32
ax1.set_xticks(x)
ax1.set_xticklabels([f"{size}" for size in company_size], weight = 'bold')

ax1.bar(x - width/2, mean, width, color = 'purple', alpha = 0.5, ec = 'black', label = "mean")
ax1.bar(x + width/2, median, width, color = 'blue', alpha = 0.5, ec = 'black', label = "median")

ax1.legend()

# Plotting Salary Distribution Among Cp Size:

fig2, ax2 = plt.subplots()
ax2.set_xticks(x)
ax2.set_xticklabels(company_size, weight='bold')

# Displaying The Plots:
plt.show()
