import joblib
import pandas as pd
from scipy.stats import chi2_contingency

model_lr = joblib.load('../models/LogisticRegression_model2.joblib')
model_rf = joblib.load('../models/RandomForest_model.joblib')

test_data = pd.DataFrame({
    'Tyres': [2, 3, 4, 5, 6],
    'Brakes': [1, 2, 1, 2, 3],
    'Suspension': [5, 4, 6, 7, 8],
    'Body and Chassis': [8, 7, 9, 6, 5],
    'Lights': [3, 2, 4, 5, 1],
    'Glazing': [6, 5, 7, 8, 6],
    'Wipers': [4, 3, 5, 4, 6],
    'Doors': [2, 4, 3, 2, 5],
    'Seat Belts': [7, 8, 6, 7, 5],
    'Airbags': [6, 7, 5, 6, 8],
    'Speedometer': [5, 4, 3, 6, 5],
    'Exhaust System': [7, 6, 8, 7, 6],
    'Fuel System': [4, 5, 3, 4, 6],
    'Final Score': [70, 60, 80, 50, 90]
})

true_labels = [1, 0, 1, 0, 1]

predictions_lr = model_lr.predict(test_data)
predictions_rf = model_rf.predict(test_data)

contingency_table = pd.crosstab(predictions_lr, predictions_rf, rownames=['Logistic Regression'],
                                colnames=['Random Forest'])

print("Contingency Table:")
print(contingency_table)

chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

alpha = 0.05  # Significance level
if p_value < alpha:
    print(
        "\nThe result is statistically significant. There is evidence to suggest a significant difference in predictions between the models.")
else:
    print(
        "\nThe result is not statistically significant. There is no evidence to suggest a significant difference in predictions between the models.")
