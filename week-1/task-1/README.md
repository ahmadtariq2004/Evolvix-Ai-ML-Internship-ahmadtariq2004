# Week 1 Task 1 - Exploratory Data Analysis (EDA) Project

**Author:** Muhammad Ahmad
**Dataset:** Automobile Fuel Economy (`cars.csv`)

## Project Overview
This project is part of the EvolviX AI/ML Internship. It focuses on performing Exploratory Data Analysis (EDA) on a dataset containing specifications of 261 cars to understand the factors affecting fuel economy (MPG).

## Steps Performed
1. **Data Loading & Inspection:** Loaded the raw dataset and inspected its structure, identifying missing values and incorrect formatting.
2. **Data Cleaning:** 
   - Stripped leading/trailing spaces from column names and text values.
   - Converted numerical columns to their correct data types.
   - Imputed missing values in `cubicinches` and `weightlbs` using the column medians.
   - Standardized region labels in the `brand` column (US, Europe, Japan).
3. **Summary Statistics:** Generated descriptive statistics to understand the basic distribution of all variables.
4. **Data Visualization:** Created 5 insightful visualizations using `matplotlib` and `seaborn`:
   - Histogram showing the distribution of Fuel Economy (MPG).
   - Boxplot comparing Fuel Economy by Number of Cylinders.
   - Boxplot comparing Fuel Economy by Region of Origin.
   - Scatterplot with a regression line showing the relationship between Vehicle Weight and Fuel Economy.
   - Line plot showing the trend of Average Fuel Economy over the model years.

## Key Insights
Based on the visual and statistical exploration, here are the top 3 insights:
1. **Weight negatively impacts fuel economy:** There is a strong negative correlation between a car's weight and its MPG. Heavier cars consume significantly more fuel.
2. **Region of origin matters:** Japanese cars generally offer the highest fuel economy (averaging ~30.2 MPG), followed by European cars (~27.5 MPG), while US cars have the lowest average (~19.6 MPG).
3. **Efficiency improved over time:** The average fuel economy consistently increased from 1971 to 1983, likely reflecting changing industry standards and a shift towards lighter, smaller-engine vehicles.
