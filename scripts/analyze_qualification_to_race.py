import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Load Data
races = pd.read_csv('data/races.csv')
results = pd.read_csv('data/results.csv')
constructors = pd.read_csv('data/constructors.csv')

# Merge data to include constructor names
results = results.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')

# Filter out races where the position is NaN (e.g., Did Not Finish)
results = results.dropna(subset=['positionOrder'])

# Convert positions and grid to numeric types
results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')

# Group data by constructor
constructor_names = results['name'].unique()

# Initialize a dictionary to store results
analysis_results = {
   'constructor': [],
   'spearman_corr': [],
   'p_value': []
}

# Analyze the relationship between qualification position and race result for each constructor
for constructor in constructor_names:
   constructor_data = results[results['name'] == constructor]

   # Calculate Spearman correlation
   corr, p_value = spearmanr(constructor_data['grid'], constructor_data['positionOrder'])

   # Store the results
   analysis_results['constructor'].append(constructor)
   analysis_results['spearman_corr'].append(corr)
   analysis_results['p_value'].append(p_value)

# Convert results to DataFrame
results_df = pd.DataFrame(analysis_results)
results_df = results_df.sort_values(by='spearman_corr', ascending=False)

# Save Results
results_df.to_csv('results/constructor_qualification_to_race_correlation.csv', index=False)

# Visualize the correlation
plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='spearman_corr', y='constructor', palette='viridis')
plt.title('Spearman Correlation of Qualification Position to Race Result by Constructor')
plt.xlabel('Spearman Correlation Coefficient')
plt.ylabel('Constructor')
plt.show()

# Print significant correlations (p < 0.05)
significant_results = results_df[results_df['p_value'] < 0.05]
print("Significant Correlations (p < 0.05):")
print(significant_results)