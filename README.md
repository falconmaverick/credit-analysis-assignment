# Credit Analysis Assignment

## Overview

This project analyzes credit risk using the Kaggle **Home Credit Default Risk** dataset. The goal is to explore data, identify patterns, and prepare it for machine learning modeling. The analysis involves exploratory data analysis (EDA), visualization, and insights extraction.

## Dataset

- **Source:** Kaggle - Home Credit Default Risk
- **Primary File Used:** `application_train.csv`
- **Column Descriptions:** `HomeCredit_columns_description.csv`

## Analysis Process

### 1. Understanding the Competition

- **Company Overview:** Home Credit is a financial company that provides loans to customers with little or no credit history.
- **Prediction Task:** The goal is to predict whether an applicant will have difficulty repaying a loan.
- **Business Impact:** The prediction helps reduce financial risks for lenders and optimize credit approval processes.

### 2. Data Inspection

- Used `.head()`, `.info()`, `.describe()` for basic understanding.
- Identified missing values.
- Visualized the class distribution of loan repayment status.

### 3. Research Questions

- How does income level affect loan repayment?
- What features have the highest correlation with default risk?
- Do employment length and contract type influence credit risk?

### 4. Data Exploration & Visualization

- Distribution plots and histograms for income levels.
- Correlation heatmaps for feature selection.
- Scatter plots to understand relationships between key variables.
- Boxplots to compare features across loan repayment statuses.

### 5. Insights & Conclusion

- Certain features (e.g., **income level, contract type, employment status**) show strong correlations with loan defaults.
- Data cleaning and feature engineering will be critical for building an accurate predictive model.

## Technologies Used

- Python 3.7
- Pandas, NumPy, Matplotlib, Seaborn
- Missingno (for missing value analysis)
- Jupyter Notebook
