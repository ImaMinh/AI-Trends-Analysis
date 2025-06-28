import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')
df_comp_size = df[['salary_usd','company_size']]
print(df_comp_size)

print(df_comp_size.shape)
print(df_comp_size['company_size'].unique())

print(df_comp_size.isnull().sum())

# Renaming
df_comp_size['company_size'] = df_comp_size['company_size'].replace({'L': 'Large', 'M': 'Medium', 'S': 'Small'})

# Grouping

df_comp_size = df_comp_size.groupby('company_size')

df_stats = df_comp_size['salary_usd'].agg(['count', 'mean', 'median']).reset_index()

print("\n", df_stats)

# Plotting:

company_size = df_stats['company_size']

mean = df_stats['mean']
median = df_stats['median']

x = np.arange(len(company_size))
print(x)

fig, ax1 = plt.subplots() # plt.subplot vs plt.subplots???

width = 0.32
ax1.set_xticks(x)
ax1.set_xticklabels([f"{size}" for size in company_size], weight = 'bold')
ax1.set_title("Salary Mean and Median by Company Size", weight = 'bold')

ax1.bar(x - width/2, mean, width, color = 'purple', alpha = 0.5, ec = 'black', label = "mean")
ax1.bar(x + width/2, median, width, color = 'blue', alpha = 0.5, ec = 'black', label = "median")

ax1.legend()

# Plotting Salary Distribution Among Cp Size:

fig2, ax2 = plt.subplots()

ax2.set_xticks(x)
ax2.set_xticklabels(company_size, weight='bold')

ax2.set_ylabel('Salary in USD', weight = 'bold')

ax2.set_title("Salary Distribution by Company Size", weight = 'bold')


rng = np.random.default_rng()

for index, size in enumerate(company_size):
    salary = df_comp_size.get_group(size)['salary_usd'].dropna().reset_index(drop=True)
    #print(salary)
    xs = np.random.normal(index, 0.05, size=len(salary))  
    # Tại sao ở đây lại là index chứ không phải x?? Nãy để x thì bị lỗi mà: 
    # ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (4998,) and arg 1 with shape (3,).
    # Trả lời: tại vì x là một mảng 3 phần tử, index ở đây là một vị trí individual trên trục x --> do x là một mảng 3 phần tử nên khi gán x với mảng size salary nó bị lỗi mismatch như trên.
    ax2.scatter(xs, list(salary), alpha=0.5, s=4)
    plt.tight_layout()
    
# Displaying The Plots:
plt.show()
