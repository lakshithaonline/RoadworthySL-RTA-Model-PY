import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    'Final Score': [76, 60, 80, 50, 90]
})

true_labels = [1, 0, 1, 0, 1]

predictions_rf = model_rf.predict(test_data)

accuracy_rf = accuracy_score(true_labels, predictions_rf)
precision_rf = precision_score(true_labels, predictions_rf)
recall_rf = recall_score(true_labels, predictions_rf)
f1_rf = f1_score(true_labels, predictions_rf)

print(f"Initial Model Performance:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
