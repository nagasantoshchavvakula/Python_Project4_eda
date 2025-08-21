"""
Step 2: Exploratory Data Analysis (EDA) 
This script works with the Student Performance dataset by loading the data, 
performing inspection and cleaning as required, and creating visualizations 
for univariate, bivariate, and multivariate analysis.
"""

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

#--------------------------------------------
#Univariate Analysis - Plotting Histograms
#--------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
# Set the aesthetic style of the plots
sns.set_style("whitegrid")
# Plotting the distribution of each numerical column
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns for Univariate Analysis:", numerical_columns)


# 1.Plotting histograms for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30, edgecolor="black")
    plt.title(f'Distribution of {column}',fontsize=11)
    plt.suptitle('Univariate Analysis', fontsize=16, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'outputPlotsUnivariate/histogramPlots/histogram_{column}.png')  # Save the plot
    plt.show()

# 2.Plotting histograms for all numerical columns in a single figure
df[numerical_columns].hist(bins=20, figsize=(12, 5), edgecolor="black")
plt.suptitle("Distribution of all", fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig("outputPlotsUnivariate/histogramPlots/overall_combined_plot.png")
plt.show()















