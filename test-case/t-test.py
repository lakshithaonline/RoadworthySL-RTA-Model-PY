import joblib
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.utils import shuffle

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

test_data, true_labels = shuffle(test_data, pd.Series(true_labels), random_state=42)

loo = LeaveOneOut()

cv_scores_rf = cross_val_score(model_rf, test_data, true_labels, cv=loo, scoring='accuracy')

baseline_performance = 0.5

t_stat, p_value = ttest_1samp(cv_scores_rf, baseline_performance)

print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("The performance of the Random Forest model is significantly better than the baseline.")
else:
    print("The performance of the Random Forest model is not significantly different from the baseline.")
