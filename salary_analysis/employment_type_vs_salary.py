import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')

df = df[['salary_usd','employment_type']]

# print(df.shape, df.columns, df.head(), df.tail(), df.sample(10), df['employment_type'].unique())
# print(df.isna().sum())

# Renaming the Employment Types:
df['employment_type'] = df['employment_type'].replace({'CT': 'Contract', 'FL': 'Freelance', 'PT': 'Part-Time', 'FT': 'Full-Time'})
#print(df['employment_type'].unique())

# Grouping
df_groups = df.groupby(['employment_type'])

# Counting the Percentages of each Employment Type:

counts = df['employment_type'].value_counts()
total = df['employment_type'].count()

series_counts_values = counts.values
series_employment_labels = counts.index

print(series_counts_values)

percentages = [(count/total)*100 for count in counts]
print(percentages)

plt.figure()

colors = ("lightcoral","grey","indianred", "beige")


plt.pie(
    list(series_counts_values), 
    labels = list(series_employment_labels), 
    autopct='%.2f%%', startangle=90, 
    wedgeprops={'edgecolor':'black'}, textprops={'weight': 'bold'}, colors=colors)
plt.title('Data Set Employment Type Percentages', weight='bold')


# Calculating Salary Stats for each Employment Type:
emp_stats = df.groupby('employment_type')['salary_usd'] \
            .agg(['count','median','mean', lambda x: x.quantile(.75)-x.quantile(.25)]) \
            .reset_index(drop=False) \
            .rename(columns={'<lambda_0>':'IQR', 'mean': 'mean_salary', 'median': 'median_salary'}) \
            .sort_values('median_salary',ascending=False)
              
print(emp_stats)

plt.figure()
plt.bar(list(np.arange(len(emp_stats['employment_type']))), emp_stats['mean_salary'], label='mean salary', ec='black', width=0.32)
plt.legend()
plt.xticks(list(np.arange(len(emp_stats['employment_type']))), labels=[f"{emp_type}" for emp_type in emp_stats['employment_type']])

# --- Running One Way ANOVA: ---

# Checking Data Validity:
# Decide on an order for the categories:
types = ['Full-Time','Freelance','Contract','Part-Time']

# --- 1) Grouped boxplot ---
fig, ax = plt.subplots(figsize=(8, 6))
# build one list of arrays, in order, for each employment type
data = [df.loc[df['employment_type']==t, 'salary_usd'].values for t in types]
print(data)

bp = ax.boxplot(data, # type: ignore
                label=types,
                vert=False,      # horizontal boxes
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black'),
                medianprops=dict(color='red', linewidth=2))

ax.set_xlabel("Salary (USD)", weight="bold")
ax.set_title("Salary by Employment Type (Boxplot)", weight="bold")
plt.tight_layout()

# --- 2) Histograms per type ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, t in zip(axes, types):
    vals = df.loc[df['employment_type']==t, 'salary_usd']
    ax.hist(vals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(t)
    ax.set_xlabel("Salary")
    ax.set_ylabel("Frequency")

plt.suptitle("Salary Distributions by Employment Type", weight="bold", y=1.02)
plt.tight_layout()
plt.show()  
