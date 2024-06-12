import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data.dta import parameters

# Load the resampled dataset
df = pd.read_csv('../data/resampled_data_with_results.csv')

# Heatmap for correlation between parameters and final score
plt.figure(figsize=(15, 10))
corr = df[parameters + ['Final Score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
