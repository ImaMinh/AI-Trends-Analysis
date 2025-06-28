import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./ai_job_dataset.csv")

# 1 --- Initial Schema and Sanity Checks ---: 
# * Quick structural checks:
#      - Look at: df.shape, df.columns, df.dtypes
#      - Peek at df.head(), df.tail() to validate that the content looks sane (no shifted rows, garbage characters, etc.)
shape = df.shape
columns = df.columns
dtypes = df.dtypes

# casting general object dtypes into generic python dtype
object_columns = df.select_dtypes(include=object) # select columns within df that contains dtype of objects
for column in object_columns.columns:
    dtypes[column] = type(df[column][1])

print(
    ">>> Initial Structural Checks on Dataset: \n",
    "1. Shape: ", shape, "\n",
    "2. Columns: ", columns, "\n"
    "3. Data types: \n", dtypes, "\n\n"
)

print(
    ">>> Head Peek: \n", df.head(), "\n\n",
    ">>> Tail Peek: \n", df.tail(), "\n\n",
    ">>> Sample: \n", df.sample(10), "\n\n"
)

# 2 --- Check for Missing Values ---:
def check_for_missing_values(df: pd.DataFrame)->None:
    missing_counts = [(df[col].isna()).sum() for col in df]
    missing_counts = pd.DataFrame([missing_counts], columns = df.columns, index=(['duplicates'])) # additional index column, add list braces to be syntax-correct
    print(missing_counts.T, "\n\n")

check_for_missing_values(df)

# Tasks for tommorow: 
# 1. Show percentages
# 3. Combine counts and percents
# 4. Handle large schema gracefully

# --- Data Type and Schema Enforcement:
# * Cast each column to its intended type: int, float, category, datetime
# * Fail fast (raise an error), if a value cannot be coerced

#3. --- Check for Duplicates ---

# check for general duplicated rows:
def check_general_duplicates(df: pd.DataFrame)->None:
    duplicates = df[df.duplicated(keep=False)]
    if duplicates.empty:
        print(">>> No duplicated rows\n\n")
    else:
        print(">>> Duplicated rows: \n", duplicates, "\n\n")

def checking_for_unique_job_titles(df: pd.DataFrame) -> None:
    
    unique_roles = df['job_title'].nunique()
    
    print(">>> Number of unique roles: ", unique_roles)
    
    job_roles = []
    for job in df['job_title']:
        if job not in job_roles:
            job_roles.append(job)
    
    if(unique_roles == len(job_roles)):
        print(">>> The job roles collected are exact and as follows: \n", job_roles, "\n\n")
    else:
        print("*** Error in finding duplicate job roles ***\n\n")

def checking_for_duplicates_job_ids(df: pd.DataFrame) -> None:
    duplicated_ids = df[df.duplicated(['job_id'])]
    if duplicated_ids.empty:
        print(">>> No duplicated job IDs\n\n")
    else:
        print(">>> Duplicated IDs: \n", duplicated_ids)

check_general_duplicates(df)
checking_for_unique_job_titles(df)
checking_for_duplicates_job_ids(df)

# --- Checking for Outliers ---

print(">>> General Numeric Stats: \n", df.describe(), "\n")

salary = df['salary_usd']

# Univariate Salary Analysis
print(">>> Standard salary deviation: ", salary.std())

def plot_hist(salary: pd.Series) -> None:
    #plot.figure()  --> bỏ đi tại vì nó tạo thêm một plot mới. Một cái này và cái figure khác khi mình gọi plt.subplots bên dưới -> 2 figures
    fig, axes = plt.subplots(figsize=(12,8))

    # Histogram on the first subplot
    counts, bins, _ = axes.hist(
        salary, 100, color="purple", edgecolor="black", alpha=0.1, label="salary distribution"
    )
    
    axes.set_xlabel("Salary", weight="bold")
    axes.set_ylabel("Occurrences", weight="bold")
    axes.legend()
    axes.set_title("Salary Distribution", weight="bold")

    # Plot line connecting bin midpoints
    bin_midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    axes.plot(
        bin_midpoints,
        counts,
        marker="o",
        linestyle="-",
        mfc="blue",
        color="black",
        ms=5,
    )
    
    axes.set_xticks(bin_midpoints[::2])
    axes.set_xticklabels(
        [f"{m}" for m in bin_midpoints[::2]],  # or just `bin_midpoints`
        rotation=45,
        ha="right",# align right so they don’t overlap
    )
    axes.tick_params(axis = 'x', labelsize=8)

    # Calculate and print skewness and kurtosis
    skew = salary.skew()
    kurtosis = salary.kurtosis()
    print(
        ">>> Skewness of Distribution: ", skew, "\n", 
        ">>> Kurtosis of Distribution: ", kurtosis
    )
    
    if(kurtosis > 3.0): # type: ignore
        print('--> Leptokurtic')
    elif (kurtosis == 3.0):
        print('--> Mesokurtic')
    else:
        print('--> Platykurtic')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
def plot_box(salary: pd.Series)->None:
    Q1 = np.percentile(salary, 25)
    Q2 = np.percentile(salary, 50)
    Q3 = np.percentile(salary, 75)
    IQR = Q3 - Q1
    
    # Boxplot
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    ax.boxplot(salary, vert=False)
    ax.set_xlabel("Salary Distribution (USD)", weight="bold")
    ax.set_xticks([Q1, Q2, Q3])
    ax.set_xticklabels([f'Q{index + 1}: {m} USD' for index, m in enumerate([Q1, Q2, Q3])], rotation=45, ha='right', weight='bold')
    ax.set_yticks([])
    
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
     
     
    # Plotting upper and lower Tukey Fence
    lower_fence = Q1 - 1.5*IQR
    upper_fence = Q3 + 1.5*IQR
    
    ax.plot(lower_fence, 1, marker = 'x', mfc = 'red', ms = 10, mew=4, mec='red')
    ax.text(float(lower_fence), 1.03, 'lower tukey fence', weight='bold')
    ax.plot(upper_fence, 1, marker = 'x', mfc = 'blue', ms = 10, mew=4, mec='blue')
    ax.text(float(upper_fence), 1.03, 'upper tukey fence', weight='bold')
    
    plt.title('Box Plot of Salary Distribution', weight='bold')

    ax.plot(Q1, 1, ms = 30, mfc = 'yellow')
    plt.tight_layout()
    
def flag_zScore_outliers(salary: pd.Series, upper_fence) -> None:
    print(">>> Examining Potential Outliers w/ Value > Upper IQR fence\n\n")
    potential_outlying_salary = df[df["salary_usd"] > upper_fence].reset_index(drop=True)
    print(">>> Potential Outlying Salary: ",potential_outlying_salary, "\n\n")
    
    plt.figure()
    plt.title("Potential Outlying Salary Distribution", weight='bold')
    plt.hist(potential_outlying_salary['salary_usd'], ec='black', bins=10, label='outlying salary distribution')
    
    print(">>> Examine Experience Level: ",potential_outlying_salary['experience_level'].unique(), "\n\n",
          ">>> Examine Job Roles: \n", potential_outlying_salary['job_title'].unique())
    
    
Q1 = np.percentile(salary, 25)
Q3 = np.percentile(salary, 75)

IQR = Q3 - Q1
lower_fence = Q1 - 1.5*IQR
upper_fence = Q3 + 1.5*IQR
    
plot_hist(salary)
plot_box(salary)
flag_zScore_outliers(salary, upper_fence)

plt.show()

# --- Dropping redundant columns ---
to_drop = [
  'job_id',                # just an identifier
  'salary_currency',       # already have salary_usd
  'employee_residence',    # not used in any of the 4 analyses
  'posting_date',          # only needed if we do time-series or “time_to_deadline”
  'application_deadline',  # ditto
  'job_description_length',# unrelated
  'benefits_score',        # not in our objectives
  'company_name',          # too granular—you're grouping by location/size
  'education_required',    # outside our scope
  'industry'               # unrelated
]
df_clean = df.drop(columns=to_drop)
df_clean.to_csv('ai_job_dataset_cleaned.csv', index=False, sep=',')