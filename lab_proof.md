# Titanic Data Analysis
This repo processes Titanic passenger dataset. After some feature engineering it then exports the file to a validated JSON format.

## What it has?
- `Passenger` Class: to structure the data
- `TitanicDataset` Class: manages the collection, statistical analysis, and JSON serialization.
- `Validation Script`: to test and validate the JSON file generated

## Insights
- Engineered `FamilySize` (calculated from SibSp and Parch) and `IsAlone` (a boolean flag indicating whether a passenger traveled without family members).
- Survival Differentiation: These features reveal that social support was a key variable; by quantifying family presence, the model can better identify how solo travelers often faced different survival outcomes compared to those traveling with family members.
- Encapsulation using classes keeps it clean maintainable

## Summary Statistics
The pipeline automatically generates key metrics, including:
- Total Passenger count.
- Survival rates.
- Missing value counts (e.g., Age and Fare data gaps).

## Proof of working solution
The proof of working solution is attached in the screenshots. Please take a look. Below is the copy paste from terminal.
````
Project setup complete!
Data directory: data
CSV file location: data/titanic.csv
Titanic data loaded successfully! Shape: (891, 12)

Columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    str    
 4   Sex          891 non-null    str    
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    str    
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    str    
 11  Embarked     889 non-null    str    
dtypes: float64(2), int64(5), str(5)
memory usage: 83.7 KB
None

First few rows:
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
Numerical columns:      PassengerId  Survived  Pclass   Age  SibSp  Parch     Fare
0              1         0       3  22.0      1      0   7.2500
1              2         1       1  38.0      1      0  71.2833
2              3         1       3  26.0      0      0   7.9250
3              4         1       1  35.0      1      0  53.1000
4              5         0       3  35.0      0      0   8.0500
..           ...       ...     ...   ...    ...    ...      ...
886          887         0       2  27.0      0      0  13.0000
887          888         1       1  19.0      0      0  30.0000
888          889         0       3   NaN      1      2  23.4500
889          890         1       1  26.0      0      0  30.0000
890          891         0       3  32.0      0      0   7.7500

[891 rows x 7 columns]
PassengerId - Mean: 446.00, Median:  446.00, std: 257.35
Survived - Mean: 0.38, Median:  0.00, std: 0.49
Pclass - Mean: 2.31, Median:  3.00, std: 0.84
Age - Mean: 29.70, Median:  28.00, std: 14.53
SibSp - Mean: 0.52, Median:  0.00, std: 1.10
Parch - Mean: 0.38, Median:  0.00, std: 0.81
Fare - Mean: 32.20, Median:  14.45, std: 49.69
PassengerId - Missing Count: 0, Missing Percentage: 0.00%
Survived - Missing Count: 0, Missing Percentage: 0.00%
Pclass - Missing Count: 0, Missing Percentage: 0.00%
Name - Missing Count: 0, Missing Percentage: 0.00%
Sex - Missing Count: 0, Missing Percentage: 0.00%
Age - Missing Count: 177, Missing Percentage: 19.87%
SibSp - Missing Count: 0, Missing Percentage: 0.00%
Parch - Missing Count: 0, Missing Percentage: 0.00%
Ticket - Missing Count: 0, Missing Percentage: 0.00%
Fare - Missing Count: 0, Missing Percentage: 0.00%
Cabin - Missing Count: 687, Missing Percentage: 77.10%
Embarked - Missing Count: 2, Missing Percentage: 0.22%
Missing values dictionary: {'PassengerId': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Survived': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Pclass': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Name': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Sex': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Age': {'count': np.int64(177), 'percentage': np.float64(19.865319865319865)}, 'SibSp': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Parch': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Ticket': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Fare': {'count': np.int64(0), 'percentage': np.float64(0.0)}, 'Cabin': {'count': np.int64(687), 'percentage': np.float64(77.10437710437711)}, 'Embarked': {'count': np.int64(2), 'percentage': np.float64(0.22446689113355783)}}
Column with the most missing values: Cabin
Column with the most missing values: Cabin
   SibSp  Parch  FamilySize
0      1      0           2
1      1      0           2
2      0      0           1
3      1      0           2
4      0      0           1
5      0      0           1
6      0      0           1
7      3      1           5
8      0      2           3
9      1      0           2
First 10 rows with family size and alone status: 
   FamilySize  IsAlone
0           2    False
1           2    False
2           1     True
3           2    False
4           1     True
5           1     True
6           1     True
7           5    False
8           3    False
9           2    False
Age groups assigned:
    Age  AgeGroup
0  22.0  Teenager
1  38.0     Adult
2  26.0  Teenager
3  35.0     Adult
4  35.0     Adult
5   NaN   Unknown
6  54.0    Senior
7   2.0     Child
8  27.0  Teenager
9  14.0     Child

==================================================
FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED
==================================================

Family Size by Survival:
              mean  median       std
Survived                            
0         1.883424     1.0  1.830669
1         1.938596     2.0  1.186076
Survival rate by IsAlone:
             mean  median       std
IsAlone                            
False    0.505650     1.0  0.500676
True     0.303538     0.0  0.460214
Survival rate by AgeGroup:
              mean  median       std
AgeGroup                            
Child     0.503597     1.0  0.501795
Teenager  0.355556     0.0  0.479570
Adult     0.423237     0.0  0.495100
Senior    0.343750     0.0  0.478714
Unknown   0.293785     0.0  0.456787

==================================================
FEATURE DIFFERENTIATION ANALYSIS
==================================================

Family Size:
  Survived mean: 1.94
  Not Survived mean: 1.88
  Difference: 0.06

Is Alone:
  Survived mean: 0.48
  Not Survived mean: 0.68
  Difference: 0.20

Age Group:
AgeGroup
Child       0.503597
Teenager    0.355556
Adult       0.423237
Senior      0.343750
Unknown     0.293785
Name: Survived, dtype: float64
Interpretation of results: 
  - Family Size: is not the most most powerful indicator of survival. The difference is very small (0.06)
  - Is Alone: Those who were not alone had a higher survival rate.
  - Age Group: children had the highest survival rates then adults. Teenagers, Seniors, and Unknown had lower survival rates. This suggests that age played a significant role in survival, with children being prioritized for rescue.
The dataset contains: 891 passengers
********* Summary Statistics ************
Total Passengers: 891
Survived: 342
Did Not Survive: 549
Average Age: 29.7
Average Fare: 32.2
Missing Values: {'age': 177, 'fare': 0}
Data exported to titanic_data.json

==================================================
RUNNING VALIDATION
==================================================
**** JSON successfully loaded *****
Expected keys are found in JSON file
Total passenger count: 891
Sample passenger data with ID: 1
JSON is valid.
Summary stats found: ['total_passengers', 'survived', 'did_not_survive', 'average_age', 'average_fare', 'missing_values']
Missing age count: 177

==================================================
VALIDATION COMPLETE
==================================================
```