import pandas as pd
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
for columns in object_columns.columns:
    dtypes[columns] = type(df[columns][1])

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
    missing_counts = pd.DataFrame([missing_counts], columns = df.columns) # additional index column
    print(missing_counts.T, "\n\n")

check_for_missing_values(df)

# Tasks for tommorow: 
# 1. Show percentages
# 3. Combine counts and percents
# 4. Handle large schema gracefully

# --- Data Type and Schema Enforcement:
# * Cast each column to its intended type: int, float, category, datetime
# * Fail fast (raise an error), if a value cannot be coerced

# if df.isnull().values.any():
#     print("Rows with any missing value:\n", df[df.isnull().any(axis=1)])



# --- Check for Duplicate Rows ---
# dups = df[df.duplicated(keep=False)]
# print("Total duplicated rows:", dups.shape[0], "\n")
# if not dups.empty:
#     print(dups)

def checking_for_unique_job_titles(df: pd.DataFrame) -> None:
    
    unique_roles = df['job_title'].nunique()
    
    print(">>> Number of unique roles: ", unique_roles)
    
    job_roles = []
    for job in df['job_title']:
        if job not in job_roles:
            job_roles.append(job)
    
    if(unique_roles == len(job_roles)):
        print(">>> The job roles collected are exact and as follows: \n", job_roles)
    else:
        print("*** Error in finding duplicate job roles ***")

def checking_for_duplicates_job_ids(df: pd.DataFrame) -> None:
    duplicated_ids = df[df.duplicated(['job_id'])]
    if duplicated_ids.empty:
        print(">>> *** There are no duplicated job IDs")
    else:
        print(">>> Duplicated IDs: \n", duplicated_ids)

checking_for_unique_job_titles(df)
checking_for_duplicates_job_ids(df)

# --- Checking for Outliers ---


# --- Dropping Redundant columns ---
# cols = [col for col in df.columns if col not in ['job_title','salary_usd','experience_level','employment_type','remote_ratio','company_size','company_location']]
# df = df.drop(cols, axis=1)

# print("Cleaned Dataset:\n", df)

