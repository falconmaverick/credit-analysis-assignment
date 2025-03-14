# Credit Information Analysis

## Overview

#This project focuses on analyzing credit risk using the **Home Credit Default Risk** competition dataset from Kaggle. The goal is to explore real-world credit data and develop insights before applying machine learning techniques.

## Objectives

#- Analyze real-world credit information data
#- Identify key issues and challenges in the dataset
#- Conduct exploratory data analysis (EDA) to uncover patterns

## Understanding the Competition
#The **Home Credit Default Risk** competition aims to help Home Credit, a financial services company, improve their risk assessment for credit applicants. The competition requires participants to develop predictive models that estimate the likelihood of a customer defaulting on a loan. By improving credit scoring models, companies can reduce financial risk and offer better financial services.

### What is Home Credit?
#->Home Credit is a financial institution that provides loans to individuals who may not have a sufficient credit history. The company seeks to enhance its credit risk assessment by leveraging data science and machine learning techniques.

### Expected Outcome of the Competition
#->Participants are expected to build predictive models that estimate the likelihood of loan default. These models should help Home Credit make better lending decisions, reducing financial losses and improving customer financial inclusion.

### Benefits of Predicting Credit Risk

#- **Better Risk Management:** More accurate credit scoring models reduce loan defaults.
#- **Improved Financial Inclusion:** More people gain access to credit based on reliable risk assessments.
#- **Operational Efficiency:** Automating risk assessments streamlines loan approval processes.

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
