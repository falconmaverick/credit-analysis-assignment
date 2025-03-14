import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Ensure 'application_train.csv' is in the working directory)
df = pd.read_csv("application_train.csv")

# Problem 2: Understanding the overview of data
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# Checking missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\nMissing Values:")
print(missing_values)

# Visualizing class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='TARGET', data=df, palette='coolwarm')
plt.title('Distribution of Loan Repayment (0: Repaid, 1: Default)')
plt.show()

# Problem 3: Setting research questions
questions = [
    "Which features are most correlated with default risk?",
    "How do income levels affect loan defaults?",
    "What impact does employment status have on credit risk?"
]
print("Research Questions:")
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

# Problem 4: Data Exploration
plt.figure(figsize=(8,6))
sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df)
plt.title('Income Distribution by Loan Repayment Status')
plt.show()

# Heatmap of top correlated features
correlation_matrix = df.corr()
top_corr_features = correlation_matrix['TARGET'].abs().sort_values(ascending=False)[1:11]
plt.figure(figsize=(10,8))
sns.heatmap(df[top_corr_features.index].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Top Features')
plt.show()

# Conclusion:
print("\nInsights:")
print("1. Income levels seem to affect default risk.")
print("2. Certain features have a strong correlation with loan repayment.")
print("3. Further investigation is required for employment and family status.")
