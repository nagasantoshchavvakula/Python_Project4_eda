#-------------------------------------
#EDA Script
#-------------------------------------
import numpy as np
import pandas as pd

# Loads/reads the dataset
df = pd.read_csv(r'data\student_habits_performance.csv')  

#--------------------------------------
#Initial Inspection:
#--------------------------------------

print("Initial Inspection:")
print("\nThe Shape of data:",df.shape)  # Displays the shape of the DataFrame
print("\nFew Rows of data:",df.head())  # Displays the first few rows of the DataFrame
print("\nColumn names:", df.columns.tolist())  # Displays the column names of the DataFrame
print("\nSummary of data types:", df.info())  # Displays the summary of data types in the DataFrame
print("\nSummary of data:", df.describe())  # Displays the summary statistics of the DataFrame

#--------------------------------------
#Data Cleaning and Preprocessing:
#--------------------------------------
print("\nData Cleaning and Preprocessing:")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in each column:\n", missing_values)
missing_values_in_columns=missing_values[missing_values > 0].index.tolist()
print("\nMissing values present in columns:", missing_values_in_columns)

#checking duplicate rows 
duplicate_rows = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_rows)
duplicated_rows_list = df[df.duplicated()].index.tolist()
print("\nIndices of duplicate rows:", duplicated_rows_list)













