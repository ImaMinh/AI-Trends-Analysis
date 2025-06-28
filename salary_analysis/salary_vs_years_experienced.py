import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/handu/Codes/Personal/Data Analysis Project/AI_Trend Analysis/ai_job_dataset_cleaned.csv')

print(df.shape, '\n')
print(df.head)
print(df.tail)
print(df.sample(10))

print(df['employment_type'].unique())

salary = df['salary_usd']
years_experience = df['years_experience']

plt.scatter(salary, years_experience, alpha = 0.2, color='purple')
plt.grid(True)
plt.title('salary vs years of experience')
plt.xlabel('salary_usd')
plt.ylabel('years_experience')
plt.show()

