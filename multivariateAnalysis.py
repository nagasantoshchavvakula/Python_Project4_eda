"""
Step 2: Exploratory Data Analysis (EDA) 
This script performs multivariate analysis on a dataset containing student habits and performance.
It includes loading the data, and creating visualizations for multivariate analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Loads/reads the dataset
df = pd.read_csv(r'data\student_habits_performance.csv') 

#--------------------------------------------------------------------------------
#Multivariate Analysis - relationship between three or more multiple variables
#--------------------------------------------------------------------------------

# Set the aesthetic style of the plots
sns.set_style("whitegrid") 

print('\nNumerical Columns:',df.select_dtypes(include=[np.number]))
# Numerical features for multivariate analysis
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns for Multivariate Analysis:", numerical_columns)
print('\nCategorial Columns:',df.select_dtypes(include=['object']))
# Categorial features for multivariate analysis
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns for Multivariate Analysis:", categorical_columns)

# 1. Pairplot with hue for categorical variable
if categorical_columns:
    for cat_col in categorical_columns:
        sns.pairplot(df, hue=cat_col, diag_kind='kde', plot_kws={'alpha':0.6})
        plt.suptitle(f"Pairplot of Numerical Features colored by {cat_col}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"outputPlotsMultivariate/pairPlots/pairplot_hue_{cat_col}.png")  # Save the plot
        plt.show()
else:
    print("No categorical columns available for hue in pairplot.")
# 2. 3D Scatter Plot for three numerical variables
if len(numerical_columns) >= 3:
    fig = plt.figure(figsize=(10, 8))
    plotX = fig.add_subplot(111, projection='3d')
    plotX.scatter(df[numerical_columns[0]], df[numerical_columns[1]], df[numerical_columns[2]], c='b', marker='o', alpha=0.6)
    plotX.set_xlabel(numerical_columns[0])
    plotX.set_ylabel(numerical_columns[1])
    plotX.set_zlabel(numerical_columns[2])
    plt.title(f'3D Scatter Plot of {numerical_columns[0]}, {numerical_columns[1]}, and {numerical_columns[2]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputPlotsMultivariate/3D_scatterPlots/3D_scatter_{numerical_columns[0]}_{numerical_columns[1]}_{numerical_columns[2]}.png')  # Save the plot
    plt.show()
else:
    print("Not enough numerical columns for 3D scatter plot.")

# Relationship between two numerical variables and one categorical variable using box plots
if len(numerical_columns) >= 2 and categorical_columns:
    for cat_col in categorical_columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=cat_col, y=numerical_columns[0], hue=numerical_columns[1], data=df)
        plt.title(f'Box Plot of {numerical_columns[0]} by {cat_col} and {numerical_columns[1]}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'outputPlotsMultivariate/boxPlots/boxplot_{numerical_columns[0]}_by_{cat_col}_and_{numerical_columns[1]}.png')  # Save the plot
        plt.show()
else:
    print("Not enough numerical columns or no categorical columns for box plots.")

# Relationship between two categorical variables and one numerical variable using violin plots
if len(categorical_columns) >= 2 and numerical_columns:
    for i in range(len(categorical_columns)):
        for j in range(i + 1, len(categorical_columns)):
            plt.figure(figsize=(12, 8))
            sns.violinplot(x=categorical_columns[i], y=numerical_columns[0], hue=categorical_columns[j], data=df, split=True)
            plt.title(f'Violin Plot of {numerical_columns[0]} by {categorical_columns[i]} and {categorical_columns[j]}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'outputPlotsMultivariate/violinPlots/violinplot_{numerical_columns[0]}_by_{categorical_columns[i]}_and_{categorical_columns[j]}.png')  # Save the plot
            plt.show()
    else:
        print("Not enough categorical columns or no numerical columns for violin plots.")


#--------------------------------------------------------------------------------
# Heatmap for correlation between numerical variables
#--------------------------------------------------------------------------------
print('\nNumerical Columns:',df.select_dtypes(include=[np.number]))
# Numerical columns for correlation heatmap
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical Columns for Bivariate Analysis:", numerical_columns)

plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap of Numerical Features", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("outputPlotsMultivariate/correlationHeatMap/correlation_heatmap.png")  # Save the plot
plt.show()

#--------------------------------------------------------------------------------
#  grouped bar charts to compare mean scores of multiple numerical categories
#--------------------------------------------------------------------------------
if categorical_columns and len(numerical_columns) >= 2:
    for cat_col in categorical_columns:
        mean_values = df.groupby(cat_col)[numerical_columns].mean().reset_index()
        mean_values_melted = mean_values.melt(id_vars=cat_col, value_vars=numerical_columns, var_name='Subject', value_name='Mean Score')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=cat_col, y='Mean Score', hue='Subject', data=mean_values_melted)
        plt.title(f'Mean Scores of Numerical Categories by {cat_col}', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'outputPlotsMultivariate/groupedBarCharts/grouped_bar_chart_mean_scores_by_{cat_col}.png')  # Save the plot
        plt.show()
else:
    print("Not enough categorical or numerical columns for grouped bar charts.")





#--------------------------------------------------------------------------------------------
# End of Multivariate Analysis
#--------------------------------------------------------------------------------------------
    
print("\nEnd of Multivariate Analysis")