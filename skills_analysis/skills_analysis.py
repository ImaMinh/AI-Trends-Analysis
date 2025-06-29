import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')

skill_set = df['required_skills'].explode().reset_index(drop=True)

# Mỗi vấn đề sẽ có một cách giải quyết riêng, ví dụ bây giờ muốn tách string skills ra thành individual skills -> cần một kiểu giải quyết khác 
# --> phối hợp những gì mình đã biết, hiểu rõ vấn đề, và tách lớp vấn đề ra từng bước giải quyết nhỏ khác nhau.

temp = []

for row in skill_set.iloc:
    skills = row.split(',')
    temp.append(skills)

print(len(temp))

df_skill = pd.DataFrame(temp)

print(
    "Shape: ", df_skill.shape,
    "Columns: \n", df_skill.columns, "\n",
    "Dtypes: \n", df_skill.dtypes, "\n",
    "Sample: \n", df_skill.sample(100)
)

# Cleaning and Normalize Raw Job Table: 
print("Missing Values per Columns: ")
for col in df_skill.columns:
    print(col, ":", df_skill[col].isnull().sum(), "missing values")

temp = pd.Series()

for col in df_skill.columns:
    print(df_skill[col].unique(), "\n")
    temp = pd.concat([temp, pd.Series(df_skill[col].unique())])
    
print(temp.unique())

temp = temp.str.strip()

print("\n\n", temp.unique())

temp = pd.Series(temp.unique())

def check_duplicate_skills(temp: pd.Series):
    temp = temp.str.lower()
    dups = temp[temp.duplicated(keep=False)]
    print("dups = ", dups)
    if(dups.empty):
        print(">>> No Duplicated Skill Title")
        return True
    else:
        print(">>> There are Duplicates, Check Again")
        raise KeyError("Duplicates Please Check Again\n")

unique_roles = pd.Series

if(check_duplicate_skills(temp)):
    unique_skills = list(temp)
    print(">>> Unique Skills: \n", np.array(unique_skills).T)    

# --> No duplicated skill titles, only need to strip blank space
# --> Process df_skill:

for col in df_skill.columns:
    df_skill[col] = df_skill[col].str.strip()

occurence = dict.fromkeys(unique_skills, 0)

print(occurence)

def count_occurence(temp_dictionary: dict) -> dict:
    for col in df_skill.columns:
        for key in temp_dictionary:
            temp_dictionary[key] += (df_skill[col] == key).sum()
            
    return temp_dictionary

occ = count_occurence(occurence)

print("\n\n Occurences: \n", occurence)

skills_df = (
    pd.Series(occ)
      .drop(index=[None])           # drop the None entry
      .sort_values(ascending=False) # highest → lowest
      .rename_axis('skill')
      .reset_index(name='count')
)

top = skills_df  # or skills_df.iloc[:10] for just the top 10
plt.figure(figsize=(8,6))
plt.barh(top['skill'], top['count'], ec='black', color='purple', alpha = 0.5)
plt.gca().invert_yaxis()   # largest at top
plt.xlabel('Number of Job Postings', weight = 'bold')
plt.yticks(weight = 'bold')
plt.title('Required Skills — Ranked by Frequency')
plt.tight_layout()
plt.show()

    




