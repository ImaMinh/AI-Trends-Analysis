import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')

df = df[['salary_usd','employment_type']]

print(df.shape, "\n", df.columns, "\n", df.head(), "\n", df.tail(), "\n", df.sample(10), "\n", df['employment_type'].unique())
print(df.isna().sum())

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

# ==================
# ==== Plotting ====
# ==================


# ----- Pie Chart ------

plt.figure()

colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']

plt.pie(
    list(series_counts_values), 
    labels = list(series_employment_labels), 
    autopct='%.2f%%', startangle=90, 
    wedgeprops={'edgecolor':'none'}, textprops={'weight': 'bold'}, colors=colors)
plt.title('Data Set Employment Type Percentages', weight='bold')    


# ================================================================
# ====== Calculating Salary Stats for each Employment Type =======
# ================================================================

# ---- calculating stats ----
emp_stats = df.groupby('employment_type')['salary_usd'] \
            .agg(['count','median','mean', lambda x: x.quantile(.75)-x.quantile(.25)]) \
            .reset_index(drop=False) \
            .rename(columns={'<lambda_0>':'IQR', 'mean': 'mean_salary', 'median': 'median_salary'}) \
            .sort_values('median_salary',ascending=False)
              
print(emp_stats)

x = np.arange(len(emp_stats['employment_type']))
width = 0.2  # width of the bars

# ---- Create the bar chart ----
fig, ax = plt.subplots()  # if no parameter -> subplot return axes of (1, 1) meaning 1 row and 1 column 
ax.bar(x - width/2, emp_stats['mean_salary'], width, label='Mean', edgecolor = 'maroon', color='brown')
ax.bar(x + width/2, emp_stats['median_salary'], width, label='Median', edgecolor = 'darkolivegreen', color='olivedrab')

# Labeling
labels = [
    f"{et}\nMean={mean:.0f}\nMedian={med:.0f}"
    for et, mean, med in zip(
        emp_stats['employment_type'],
        emp_stats['mean_salary'],
        emp_stats['median_salary'],
    )
]

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
ax.set_xlabel('Employment Type')
ax.set_ylabel('Salary (USD)')
ax.set_title('Mean vs Median Salary by Employment Type', weight='bold')
ax.legend()

plt.tight_layout()

# Checking Data Validity:
# Decide on an order for the categories and Un-grouping:
types = ['Full-Time','Freelance','Contract','Part-Time']
data = [df.loc[df['employment_type']==t, 'salary_usd'].values for t in types]

print(data)

# -------------------------
# ---- Grouped boxplot ----
# -------------------------
# Đọc kỹ lại đoạn này
# fig, ax = plt.subplots(figsize=(8, 6))
# # build one list of arrays, in order, for each employment type
# print(data)

# bplots = ax.boxplot(data, # type: ignore
#                 label=types,
#                 vert=False,      # horizontal boxes
#                 patch_artist=True,
#                 boxprops=dict(facecolor='lightblue', edgecolor='black'),
#                 medianprops=dict(color='red', linewidth=3),
#                 flierprops=dict(markerfacecolor='gray', alpha=0.5))

# for patch, color in zip(bplots['boxes'], colors):
#     patch.set_facecolor(color)

# ax.set_xlabel("Salary (USD)", weight="bold")
# ax.set_yticklabels(types)
# ax.set_title("Salary by Employment Type (Boxplot)", weight="bold")
# plt.tight_layout()


plt.figure(figsize=(8,4))
sns.boxplot(
    y='employment_type',
    x='salary_usd',
    data=df,
    order=types,            # enforces your exact order
    palette='Set2',
    flierprops={'markerfacecolor':'gray', 'alpha':0.5},
    medianprops={'color':'red'}
)
plt.xlabel('Salary (USD)')
plt.ylabel('Employment Type')
plt.title('Salary by Employment Type', weight='bold')
plt.show()

# --- Histograms per type ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, t in zip(axes, types):
    vals = df.loc[df['employment_type']==t, 'salary_usd']
    ax.hist(vals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(t, weight='bold')
    ax.set_xlabel("Salary")
    ax.set_ylabel("Distribution")

plt.suptitle("Salary Distributions by Employment Type", weight="bold")
plt.tight_layout()


"""
Đọc phân tích trong Notes <Phân tích Variance giữa các Employment Type> 
"""
# ==============================
# ==== Statistical Analysis ====
# ==============================

# --- Sử dụng Kruskal-Wallis để phân tích độ lớn variance giữa các Nhóm emp_type và tính episilon bình phương:

stat, p_value = stats.kruskal(*data) # <--- Group unpacking

# calculate epsilon squared:
H = stat
k = len(data)
n = sum(len(group) for group in data) # number of data points across all groups (total number of data points)

epsilon_squared = (H - k + 1) / (n - k)

print(
    "H: ", stat, "\n"
    "p-value: ", p_value, "\n"
    "epsilon-squared: ", epsilon_squared
)

if(p_value > 0.05):
    print('H0 can be true <-> No significant variance')
else:
    print('H0 might not be true <-> Might have significant variance')
if(epsilon_squared < 0.06):
    print("Effect size is small")
elif(0.06 <= epsilon_squared <= 0.14):
    print("Effect size is medium")
elif(epsilon_squared > 0.14):
    print("Effect size is large")
    
plt.figtext(0.5, -0.05, f"H = {H:.2f}, p = {p_value:.4f}, ε² = {epsilon_squared:.3f}", ha='center', fontsize=10)

# Show the plots:
plt.show()
