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

