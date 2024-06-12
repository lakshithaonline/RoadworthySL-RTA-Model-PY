import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data.dta import parameters

# Load the resampled dataset
df = pd.read_csv('../data/resampled_data_with_results.csv')

# Histogram for each parameter
fig, axes = plt.subplots(4, 4, figsize=(20, 15))
axes = axes.flatten()
for i, param in enumerate(parameters):
    sns.histplot(df[param], kde=True, ax=axes[i])
    axes[i].set_title(f'{param} Score Distribution')
plt.tight_layout()
plt.show()

# Box plot for each parameter
plt.figure(figsize=(20, 10))
sns.boxplot(data=df[parameters])
plt.title('Parameter Score Distributions')
plt.xticks(rotation=45)
plt.show()

# Heatmap for correlation between parameters and final score
plt.figure(figsize=(15, 10))
corr = df[parameters + ['Final Score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot for final score vs parameter scores
fig, axes = plt.subplots(4, 4, figsize=(20, 15))
axes = axes.flatten()
for i, param in enumerate(parameters):
    sns.scatterplot(x=df[param], y=df['Final Score'], hue=df['Outcome'], ax=axes[i])
    axes[i].set_title(f'Final Score vs {param} Score')
plt.tight_layout()
plt.show()

# Pie chart for outcome distribution
outcome_counts = df['Outcome'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Outcome Distribution')
plt.show()

# Stacked bar chart for final score ranges and outcomes
bins = [i for i in range(0, 110, 10)]
df['Final Score Range'] = pd.cut(df['Final Score'], bins)
score_range_outcome = df.groupby(['Final Score Range', 'Outcome']).size().unstack()
score_range_outcome.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Final Score Ranges and Outcomes')
plt.xlabel('Final Score Range')
plt.ylabel('Count')
plt.show()
