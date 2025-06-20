import pandas as pd

df = pd.read_csv("./ai_job_dataset.csv")

# --- Check for Missing Values ---
null_counts = df.isnull().sum()
print("Missing values per column:\n", null_counts, "\n")

if df.isnull().values.any():
    print("Rows with any missing value:\n", df[df.isnull().any(axis=1)])

# --- Check for Duplicate Rows ---
dups = df[df.duplicated(keep=False)]
print("Total duplicated rows:", dups.shape[0])
if not dups.empty:
    print(dups)


