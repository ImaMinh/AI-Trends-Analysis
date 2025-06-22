import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./ai_job_dataset.csv")

# --- Check for Missing Values ---
null_counts = df.isnull().sum()
print("Missing values per column:\n", null_counts, "\n")

if df.isnull().values.any():
    print("Rows with any missing value:\n", df[df.isnull().any(axis=1)])

# --- Check for Duplicate Rows ---
dups = df[df.duplicated(keep=False)]
print("Total duplicated rows:", dups.shape[0], "\n")
if not dups.empty:
    print(dups)

# --- Checking for Outliers ---


# --- Dropping Redundant columns ---

cols = [col for col in df.columns if col not in ['job_title','salary_usd','experience_level','employment_type','remote_ratio','company_size','company_location']]
df = df.drop(cols, axis=1)

print("Cleaned Dataset:\n", df)