"""
Titanic Data Analysis and JSON Export
Author: 
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""
 
from nbformat import write
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
 
print("Missing values dictionary:", missing_values)
# Identify which columns have the most missing data
# using lambda to get the key with the maximum count of missing values since 
# it is a dictionary of dictionaries, we need to access the "count" key for  each column to find the maximum.
column_with_most_missing = max(missing_values, key=lambda x: missing_values[x]["count"]) 
print("Column with the most missing values:", column_with_most_missing)

# using pandas built-in functions to find the column with the most missing values
column_with_most_missing_value = df.isnull().sum().idxmax()
print("Column with the most missing values:", column_with_most_missing_value)

# ======================================================
# Step 5: Feature Engineering
# ======================================================
# Create a copy of the dataframe for feature engineering
df_features = df.copy()
# Feature 1: Family Size
df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch'] + 1
print(df_features[['SibSp', 'Parch', 'FamilySize']].head(10))

# Feature 2: Is Alone
df_features['IsAlone'] = (df_features['FamilySize'] == 1) # using astype(int) to convert boolean to integer (1 for True, 0 for False)
print("First 10 rows with family size and alone status: ")
print(df_features[['FamilySize', 'IsAlone']].head(10))

# Feature 3: Age Groups

custom_bins = [0, 18, 30, 50, 80]
custom_labels = ['Child', 'Teenager', 'Adult', 'Senior'] # label should be len(bins) - 1
df_features['AgeGroup'] = pd.cut(df_features['Age'], bins=custom_bins, labels=custom_labels)

# Add 'Unknown' category for NaN values in AgeGroup
df_features['AgeGroup'] = df_features['AgeGroup'].cat.add_categories('Unknown').fillna('Unknown')

print("Age groups assigned:")
print(df_features[['Age', 'AgeGroup']].head(10)) # this still returns NaN for Age column, but AgeGroup will have 'Unknown' for those rows

# Analyze feature differences between survivors and non-survivors
print("\n" + "="*50)
print("FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED")
print("="*50)

print("\nFamily Size by Survival:")
family_survival = df_features.groupby('Survived')['FamilySize'].agg(['mean', 'median', 'std'])
print(family_survival)

# Analyze Survival rate by IsAlone
print("Survival rate by IsAlone:")
alone_survival = df_features.groupby('IsAlone')['Survived'].agg(['mean', 'median', 'std'])
print(alone_survival)

# Analyze Survival rate by AgeGroup
print("Survival rate by AgeGroup:")
agegroup_survival = df_features.groupby('AgeGroup')['Survived'].agg(['mean', 'median', 'std'])
print(agegroup_survival)

# Statistical test: Do these features help differentiate?
print("\n" + "="*50)
print("FEATURE DIFFERENTIATION ANALYSIS")
print("="*50)
 
survived = df_features[df_features['Survived'] == 1]
not_survived = df_features[df_features['Survived'] == 0]
 
print("\nFamily Size:")
print(f"  Survived mean: {survived['FamilySize'].mean():.2f}")
print(f"  Not Survived mean: {not_survived['FamilySize'].mean():.2f}")
print(f"  Difference: {abs(survived['FamilySize'].mean() - not_survived['FamilySize'].mean()):.2f}")

print("\nIs Alone:")
print(f"  Survived mean: {survived['IsAlone'].mean():.2f}")
print(f"  Not Survived mean: {not_survived['IsAlone'].mean():.2f}")
print(f"  Difference: {abs(survived['IsAlone'].mean() - not_survived['IsAlone'].mean()):.2f}")

print("\nAge Group:")
survival_by_agegroup = df_features.groupby('AgeGroup', observed=True)['Survived'].mean()
print(survival_by_agegroup)

print("Interpretation of results: ")
print("  - Family Size: is not the most most powerful indicator of survival. The difference is very small (0.06)")
print("  - Is Alone: Those who were not alone had a higher survival rate.")
print("  - Age Group: children had the highest survival rates then adults. Teenagers, Seniors, and Unknown had lower survival rates. This suggests that age played a significant role in survival, with children being prioritized for rescue.")

# ======================================================
# Step 6: Creating a Data Export Class
# ======================================================
# Create a Python class to structure and export data to JSON format.

class Passenger:
    def __init__(self, passenger_id, name, age, sex, survived, pclass, 
                 fare, embarked=None, family_size=None, is_alone=None, title=None):
        # Tip: Use pd.notna() to check if a value is not null/NaN
        # Tip: Convert to appropriate types (int, float, str)
        self.passenger_id = int(passenger_id) if pd.notna(passenger_id) else None
        self.name = str(name) if pd.notna(name) else "Unknown"
        self.age = float(age) if pd.notna(age) else None
        self.sex = str(sex) if pd.notna(sex) else "Unknown"
        self.survived = int(survived) if pd.notna(survived) else None
        self.pclass = int(pclass) if pd.notna(pclass) else None
        self.fare = float(fare) if pd.notna(fare) else 0.0
        self.embarked = str(embarked) if pd.notna(embarked) else None
        self.family_size = int(family_size) if pd.notna(family_size) else None
        self.is_alone = bool(is_alone) if pd.notna(is_alone) else None
        self.title = str(title) if pd.notna(title) else "Unknown"
        
    def to_dict(self):
        """Convert passenger to dictionary for JSON serialization."""
        return self.__dict__  # This will automatically convert all attributes to a dictionary
    
class TitanicDataset:
    """Represents the entire Titanic dataset with methods for JSON export."""
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.passengers = []  # Will store Passenger objects
        self._create_passengers()
    
    def _create_passengers(self):
        """Create Passenger objects from dataframe."""
        # Tip: Use row.get('ColumnName', default_value) to safely get values
        for _, row in self.dataframe.iterrows():
            passenger = Passenger(
                passenger_id=row.get('PassengerId'),
                name=row.get('Name'),
                age=row.get('Age'),
                sex=row.get('Sex'),
                survived=row.get('Survived'),
                pclass=row.get('Pclass'),
                fare=row.get('Fare'),
                embarked=row.get('Embarked'),
                family_size=row.get('FamilySize'),
                is_alone=row.get('IsAlone'),
                title=row.get('Title')
            )
            self.passengers.append(passenger)
    
    def to_json(self, filename='titanic_data.json'):
        """Export dataset to JSON file."""
        # create a dictionary with metadata and passenger data
        data = {
            'metadata': {
                'dataset_name': 'Titanic Passenger Dataset',
                'export_date': datetime.now().isoformat(),
                'total_passengers': len(self.passengers),
                'survival_rate': float(self.dataframe['Survived'].mean()) # convert to float for JSON serialization
            },
            'passengers': [p.to_dict() for p in self.passengers]
        }
        # write to JSON file with indentation for readability
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {filename}")
        return data
    
    def get_summary_stats(self):
        """Get summary statistics."""
        # Calculate and return summary statistics
        survived_count = sum(1 for p in self.passengers if p.survived == 1)
        did_not_survive_count = sum(1 for p in self.passengers if p.survived == 0)
        total_passengers = len(self.passengers)
        average_age = float(np.mean([p.age for p in self.passengers if p.age is not None])) # use float() to ensure JSON serializable
        average_fare = float(np.mean([p.fare for p in self.passengers if p.fare is not None]))
        
        return {
            'total_passengers': total_passengers,
            'survived': survived_count,
            'did_not_survive': did_not_survive_count,
            'average_age': average_age,
            'average_fare': average_fare
        }
        
# Create dataset object and export
# Check if df_features exists and is not empty
if 'df_features' in locals() and not df_features.empty:
    # Create a TitanicDataset object
    dataset = TitanicDataset(df_features)
    # Print basic information about the dataset
    print(f"The dataset contains: {len(dataset.passengers)} passengers")
    
    stats = dataset.get_summary_stats()
    print("********* Summary Statistics ************")
    for key, value in stats.items():
        print(f"{key.replace('_',' ').title()}: {value}")     
    
    dataset.to_json('titanic_data.json')

