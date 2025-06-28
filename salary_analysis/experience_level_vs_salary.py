# Questions to Answer:
# - Are senior roles really pay better?
# - Find how years of experience affects pay
#   • Is there a strong correlation between years of experience and salary
#   • Are there any anomalies (high pay at low experience?)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading the file
df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')
df = df[['job_title', 'salary_usd', 'experience_level', 'years_experience']]


# Are senior roles really pay better?
# Spot anomalies (high pay at low experience)


# --- Are Senior Roles Get Paid Better? ---

group_stats = df.groupby('experience_level') # grouping data by experience levels
#print(group_stats.groups)

aggregate_stats = dict() # Can't access group stats like a normal dict

for key, sub_groups in group_stats:
    #print(sub_groups)
    aggregate_stats[key] = sub_groups['salary_usd'].agg(['count', 'mean', 'median'])

aggregate_stats = pd.DataFrame(aggregate_stats)

levels = aggregate_stats.columns
means = aggregate_stats.loc['mean']
medians = aggregate_stats.loc['median']

# Positions on the x-axis
x = np.arange(len(levels))
width = 0.35  # width of the bars

# Create the bar chart
fig, ax = plt.subplots()  # if no parameter -> subplot return axes of (1, 1) meaning 1 row and 1 column 
ax.bar(x - width/2, means, width, label='Mean')
ax.bar(x + width/2, medians, width, label='Median')

# Labeling
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_xlabel('Experience Level')
ax.set_ylabel('Salary (USD)')
ax.set_title('Mean vs Median Salary by Experience Level')
ax.legend()

plt.tight_layout()
#plt.show()


# --- Spotting Anomalies <EN get paid better than EX e.g.> ---
keys = list(group_stats.groups.keys())
experience_level_salaries = dict().fromkeys(keys, [])

for key in keys:
    experience_level_salaries[key] = list(group_stats.get_group(key)['salary_usd'])

# Using dict comprehension to create a dictionary of Series.
# Since our list of salaries are not at the same length, by using Series, Pandas will fill out the ragged lists with NaN.

# dict comprehension:
experience_level_salaries = {experience_level: pd.Series(salaries) for experience_level, salaries in experience_level_salaries.items()}
experience_level_salaries = pd.DataFrame(experience_level_salaries)

fig, ax = plt.subplots(figsize=(8,6))

levels = experience_level_salaries.columns
for i, lvl in enumerate(levels):
    # grab the non-NaN salaries
    ys = experience_level_salaries[lvl].dropna().values
    
    # place them all at x = i (or add a tiny jitter so points don't
    # perfectly overlap)
    xs = np.random.normal(i, 0.05, size=len(ys))
    
    ax.scatter(xs, list(ys), alpha=0.4, s=10)

# tidy up the x-axis
ax.set_xticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.set_xlabel("Experience Level")
ax.set_ylabel("Salary (USD)")
ax.set_title("Salary Distribution by Experience Level (scatter)")

plt.tight_layout()
plt.show()