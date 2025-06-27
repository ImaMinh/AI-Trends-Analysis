# Questions to Answer:
# • Are senior roles really pay better?
# - Find how years of experience affects pay
# • Is there a strong correlation between years of experience and salary
# • Are there any anomalies (high pay at low experience?)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../ai_job_dataset_cleaned.csv')

df = df[['job_title', 'salary_usd', 'experience_level', 'years_experience']]


# Are senior roles really pay better?

group_stats = df.groupby('experience_level') # grouping data by experience levels
#print(group_stats.groups)

aggregate_stats = dict() # Can't access group stats like a normal dict

for key, sub_groups in group_stats:
    aggregate_stats[key] = sub_groups['salary_usd'].agg(['count', 'mean', 'median'])

aggregate_stats = pd.DataFrame(aggregate_stats)

levels = aggregate_stats.columns
means = aggregate_stats.loc['mean']
medians = aggregate_stats.loc['median']

# Positions on the x-axis
x = np.arange(len(levels))
width = 0.35  # width of the bars

# Create the bar chart
fig, ax = plt.subplots()
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
plt.show()