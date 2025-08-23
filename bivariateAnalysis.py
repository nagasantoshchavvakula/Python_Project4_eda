"""
Step 2: Exploratory Data Analysis (EDA) 
This script works with the Student Performance dataset by loading the data,
and creating visualizations  for bivariate analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Loads/reads the dataset
df = pd.read_csv(r'data\student_habits_performance.csv') 


#--------------------------------------------
#Bivariate Analysis - Plotting Scatter Plots    
#--------------------------------------------

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
print('\nNumerical Columns:',df.select_dtypes(include=[np.number]))
# Numerical features for bivariate analysis
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns for Bivariate Analysis:", numerical_columns)

# 1.Plotting scatter plots for each pair of numerical columns
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[numerical_columns[i]], y=df[numerical_columns[j]], alpha=0.6)
        plt.title(f'Scatter Plot of {numerical_columns[i]} vs {numerical_columns[j]}', fontsize=11)
        plt.suptitle('Bivariate Analysis', fontsize=16, fontweight='bold')
        plt.xlabel(numerical_columns[i])
        plt.ylabel(numerical_columns[j])
        plt.tight_layout()
        plt.savefig(f'outputPlotsBivariate/scatterPlots/scatter_{numerical_columns[i]}_vs_{numerical_columns[j]}.png')  # Save the plot
        plt.show()

# 2.Plotting scatter plots for all numerical columns in a single pairplot
sns.pairplot(df[numerical_columns], diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle("Pairplot of Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputPlotsBivariate/scatterPlots/overall_pairplot.png")  # Save the plot
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap of Numerical Features", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("outputPlotsBivariate/scatterPlots/correlation_heatmap.png")  # Save the plot
plt.show()

#--------------------------------------------
#Bivariate Analysis - Plotting Box Plots
#--------------------------------------------

print('\nCategorial Columns:',df.select_dtypes(include=['object']))
# Categorial features for bivariate analysis
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns for Bivariate Analysis:", categorical_columns)
# Automatically remove unique identifiers (high-cardinality columns)
unique_identifiers = [col for col in categorical_columns if df[col].nunique() == len(df)]
# Drop them from categorical list
for col in unique_identifiers:
    categorical_columns.remove(col)
print("\nRemoved unique identifier columns:", unique_identifiers)
print("\nFinal categorical columns for bivariate analysis:", categorical_columns)

# Plotting box plots for each pair of categorical and numerical columns
for cat_col in categorical_columns:
    for num_col in numerical_columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df[cat_col], y=df[num_col], palette='Set3')
        plt.title(f'Box Plot of {num_col} by {cat_col}', fontsize=11)
        plt.suptitle('Bivariate Analysis', fontsize=16, fontweight='bold')
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'outputPlotsBivariate/boxPlots/boxplot_{num_col}_by_{cat_col}.png')  # Save the plot
        plt.show()
        plt.close() # Close the plot to free up memory

# Plotting box plots for all numerical columns in a single figure
for cat_col in categorical_columns:
    plt.figure(figsize=(15, 10))
    df_melted = df.melt(id_vars=cat_col, value_vars=numerical_columns, var_name='Numerical Feature', value_name='Value')
    sns.boxplot(x='Numerical Feature', y='Value', hue=cat_col, data=df_melted, palette='Set3')
    plt.title(f'Box Plots of Numerical Features by {cat_col}', fontsize=14, fontweight='bold')
    plt.suptitle('Bivariate Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Numerical Features')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'outputPlotsBivariate/boxPlots/overall_boxplot_by_{cat_col}.png')  # Save the plot
    plt.show()
    plt.close() # Close the plot to free up memory

        