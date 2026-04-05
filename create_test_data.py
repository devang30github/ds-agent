import pandas as pd
import numpy as np

np.random.seed(42)
n = 150

age        = np.random.randint(22, 60, n)
salary     = np.random.randint(30000, 120000, n)
department = np.random.choice(["eng", "hr", "sales", "marketing"], n)
tenure     = np.random.randint(1, 15, n)

# Churn logic — older low salary employees churn more
churn = ((salary < 50000) & (age > 40)).astype(int)
churn += (department == "hr").astype(int)
churn  = (churn + np.random.randint(0, 2, n)) % 2

# Add some missing values
age_col        = age.astype(float)
salary_col     = salary.astype(float)
dept_col       = department.astype(object)

age_col[np.random.choice(n, 10, replace=False)]    = np.nan
salary_col[np.random.choice(n, 5, replace=False)]  = np.nan
dept_col[np.random.choice(n, 8, replace=False)]    = np.nan

df = pd.DataFrame({
    "age":        age_col,
    "salary":     salary_col,
    "department": dept_col,
    "tenure":     tenure,
    "churn":      churn,
})

df.to_csv("uploads/test.csv", index=False)
print(f"Created uploads/test.csv — {len(df)} rows, {df.shape[1]} cols")
print(df.head())
print(f"\nChurn distribution:\n{df['churn'].value_counts()}")