import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from matplotlib.cm import get_cmap # get color map method from matplot

# =========================================
# ========= Read and Prepare Data =========
# =========================================

data = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')[['company_location', 'salary_usd']]

print(">>> Null Values: \n", data.isnull().sum())
print(">>> Sample: \n",data.sample(10))

df_stats = data.groupby(['company_location'])['salary_usd'].agg(['mean', 'median', max])
print(">>> Initial Statistics: \n", df_stats.sort_values(by='median', ascending=False))

locations = data['company_location'].unique()
print(">>> Unique Locations: \n", locations) 

middle_index = len(locations) // 2

print(data['company_location'].value_counts().__len__())

# ====================================================================================
# ======= Inspect how salary distribution is across countries before comparing =======
# ====================================================================================

# ----- Plotting all the Salary Distribution Here -----

# Parameters for plotting:
salary_ticks = np.linspace(30000, 300000, 4, dtype=int)

# --- First Batch of Countries ---

fig, axes = plt.subplots(2, 5)
axes = axes.flatten() # Convert the 2-D array into a flat 1-D array (n dim,). Example: shape = (3, 7) -> axes.flatten() will convert the array into shape (3x7,) = (21,)

fig.suptitle('Salary Distribution by Countries', weight='bold')
fig.supylabel('Counts', weight='bold')
fig.supxlabel('Salary in USD', weight = 'bold')

for ax, t in zip(axes, locations[:middle_index]):
    # plot the histogram
    vals = data.loc[data['company_location']==t, 'salary_usd']
    ax.hist(vals, bins=20, edgecolor='black', alpha=0.5)
    
    # setting the ticks and titles
    ax.set_xticks(salary_ticks)
    ax.set_xticklabels([f"{salary}USD" for salary in salary_ticks], rotation=30, fontsize=7, weight='bold')
    
    ax.set_title(t, weight='bold')
    ax.set_ylabel("Counts", fontsize=7)

ncols = 5
for ax in axes[ncols : 2*ncols]:
    ax.set_xlabel("Salary", fontsize=7, weight='bold')

plt.subplots_adjust(hspace=0.356, wspace=0.303, left=0.068, right=0.967)

# --- Second Batch of Countries ---
fig, axes = plt.subplots(2, 5, sharex=False, sharey=True) 
axes = axes.flatten()

fig.suptitle('Salary Distribution by Countries', weight='bold')
fig.supylabel('Counts', weight='bold')
fig.supxlabel('Salary in USD', weight = 'bold')

fig.suptitle('Salary Distribution by Countries', weight='bold')
fig.supylabel('Counts', weight='bold')
fig.supxlabel('Salary in USD', weight = 'bold')

for ax, t in zip(axes, locations[middle_index:]):
    vals = data.loc[data['company_location']==t, 'salary_usd']
    ax.hist(vals, bins=20, edgecolor='black', alpha=0.5)
   
    # setting the ticks and titles
    ax.set_xticks(salary_ticks)
    ax.set_xticklabels([f"{salary}USD" for salary in salary_ticks], rotation=30, fontsize=7, weight='bold')
    
    ax.set_title(t, weight='bold')
    ax.set_ylabel("Counts", fontsize=7)

ncols = 5
for ax in axes[ncols : 2*ncols]:
    ax.set_xlabel("Salary", fontsize=7, weight='bold')

plt.subplots_adjust(hspace=0.356, wspace=0.303, left=0.068, right=0.967)

plt.show()

# =====================================================================================================
# ===== Use Kruskal Wallis to evaluate if there are significant difference between each countries =====
# =====================================================================================================

groups = [data.loc[data['company_location']==c,'salary_usd'].reset_index(drop=True) for c in locations] #type: ignore
H, p = stats.kruskal(*groups)
print(f"================\n>>>> H Statistics ={H} \n>>>> p-value = {p:.3e}\n================")

# ==============================================================================================
# ====== Using Bootstrap to Evaluate Sampling Distribution Median and Compare the Medians ======
# ==============================================================================================

def bootstrap_ci(x, stat=np.median, number_of_repeats=2000, alpha=0.05):
    boot_stats = np.array([
        stat(np.random.choice(x, size=len(x), replace=True)) for _ in range(number_of_repeats) #bootstrap sampling B number of times.
    ])
    lower, upper = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)]) # calculate the lower/upper CI intervals.
    return lower, upper

summary = (
    data
      .groupby('company_location')['salary_usd']
      .agg(median='median', count='size')
)

cis = (
    data
      .groupby('company_location')['salary_usd']
      .apply(lambda x: pd.Series(bootstrap_ci(x), index=['ci_low', 'ci_high'])).reset_index(drop=False)
)

# Pivot to wide form:
cis_wide = (
    cis
    .pivot(index='company_location', columns='level_1', values='salary_usd')
    .rename(columns={'ci_low':'ci_low', 'ci_high':'ci_high'})
    .reset_index()
)

print(cis)

# 2) Merge with summary (which already has company_location, median, count)
summary = summary.merge(cis_wide, on='company_location')

# 3) (Optional) sort & reset index
summary = summary.sort_values('median', ascending=False).reset_index(drop=True)

print(summary)



# ====================================
# ==== Plotting the Distribution =====
# ====================================

# 1) Data
countries = summary['company_location']
medians   = summary['median'].values
ci_lows   = summary['ci_low'].values
ci_highs  = summary['ci_high'].values

print(
    countries, "\n\n",
    medians, "\n\n",
    ci_lows, "\n\n",
    ci_highs
)

y_err_low = medians - ci_lows
y_err_high = ci_highs - medians

y_err = [y_err_low, y_err_high]

# 2) Compute bar positions and width
n = len(countries)
width = 0.6   # try 0.6 or 0.5 for narrower bars
x = np.arange(n)

# 3) Color map based on median magnitude
denom = medians.max() - medians.min()
scaled_data = (medians - medians.min()) / denom
cmap   = get_cmap('viridis')
colors = []
for decimal in scaled_data:
    colors.append(cmap(decimal))

# 4) Plot bars
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('gainsboro')
bars = ax.bar(x, medians, width=width, color=colors, edgecolor='black')

# 5) Add error bars
ax.errorbar(
    x, medians,
    yerr= y_err,
    fmt='none',       # no connecting line
    ecolor='teal',   # errorbar color
    capsize=4,
    linewidth=1
)

# 6) Add median markers
ax.scatter(
    x, medians,
    marker='D',       # diamond marker
    color='white',
    edgecolor='black',
    zorder=5,          # draw on top
)

for x_pos, median in zip(x, medians):
    ax.text(x_pos + 0.1, median + 3000, f'{median:.0f}', fontsize = 8.5)

# 7) Ticks & labels
ax.set_xticks(x)
ax.set_xticklabels(countries, rotation=45, ha='right')
ax.set_xlim(-0.5, n-0.5)  # pad 0.5 on each side
ax.set_ylabel("Median Salary (USD)")
ax.set_title("Median AI Job Salary by Country with 95% Bootstrap CI", weight='bold')

plt.tight_layout()
plt.show()