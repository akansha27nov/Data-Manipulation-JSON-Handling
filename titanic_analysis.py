"""
Titanic Data Analysis and JSON Export
Author: [Your Name]
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""
 
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
 
 # ======================================================
# Step 1: Set up paths
# =======================================================
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_data.json"
 
# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)
 
print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")

# ======================================================
# Step 2: Importing and Exploring the Data
# ======================================================

df = pd.read_csv(CSV_FILE)
print(f"Titanic data loaded successfully! Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(df.info())
print(f"\nFirst few rows:")
print(df.head())

# ======================================================
# Step 3: Calculate descriptive statistics
# ======================================================
# Select numeric columns only
numeric_columns = df.select_dtypes(include=np.number)
print("Numerical columns:", numeric_columns)
# Calculate statistics (.mean, median, std)
for col in numeric_columns.columns:
    col_mean = numeric_columns[col].mean()
    col_median = numeric_columns[col].median()
    col_std = numeric_columns[col].std()
    print(f"{col} - Mean: {col_mean:.2f}, Median: {col_median: .2f}, std: {col_std:.2f}")
    
# ======================================================
# Step 4: Identifying Missing Values
# ======================================================
missing_values = {}

for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_values[col] = {
        "count": missing_count,
        "percentage": missing_percentage
    }
    print(f"{col} - Missing Count: {missing_count}, Missing Percentage: {missing_percentage:.2f}%")
 
print(missing_values)
# Identify which columns have the most missing data
column_with_most_missing = max(missing_values, key=lambda x: missing_values[x]["count"])
# column_with_most_missing = max(missing_values.values())
print("Column with the most missing values:", column_with_most_missing)

# ======================================================
# Step 5: Feature Engineering
# ======================================================
# Create a copy of the dataframe for feature engineering
df_features = df.copy()
# Feature 1: Family Size
df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch'] + 1
print(df_features[['SibSp', 'Parch', 'FamilySize']].head(10))
