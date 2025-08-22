"""
Step 2: Exploratory Data Analysis (EDA) 
This script works with the Student Performance dataset by loading the data,
and creating visualizations for univariate analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#--------------------------------------------
#Univariate Analysis - Plotting Histograms
#--------------------------------------------


# Loads/reads the dataset
df = pd.read_csv(r'data\student_habits_performance.csv')  

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

print('\nNumerical Columns:',df.select_dtypes(include=[np.number]))
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

#--------------------------------------------
#Univariate Analysis - Plotting Bar charts
#--------------------------------------------
print('\nCategorial Columns:',df.select_dtypes(include=['object']))
# Categorial features for univariate analysis
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns for Univariate Analysis:", categorical_columns)

# Automatically remove unique identifiers (high-cardinality columns)
unique_identifiers = [col for col in categorical_columns if df[col].nunique() == len(df)]

# Drop them from categorical list
for col in unique_identifiers:
    categorical_columns.remove(col)

print("\nRemoved unique identifier columns:", unique_identifiers)
print("\nFinal categorical columns for univariate analysis:", categorical_columns)

# 1.Plotting bar charts for each categorical column
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    ax=sns.countplot(data=df, x=column, palette='viridis', width=0.4)

    # Adding value labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
        
    plt.title(f'Count of {column}', fontsize=11)
    plt.suptitle('Univariate Analysis', fontsize=16, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(f'outputPlotsUnivariate/barPlots/bar_{column}.png')  # Save the plot
    plt.show()