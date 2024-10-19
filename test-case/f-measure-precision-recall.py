import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

model = joblib.load('../models/RandomForest_model.joblib')

test_data = pd.DataFrame([
    {'Tyres': 2, 'Brakes': 3, 'Suspension': 9, 'Body and Chassis': 8, 'Lights': 2,
     'Glazing': 8, 'Wipers': 7, 'Doors': 8, 'Seat Belts': 9, 'Airbags': 8,
     'Speedometer': 7, 'Exhaust System': 8, 'Fuel System': 7, 'Final Score': 70.86},
])

true_labels = [1]  # Example: 0 for fail, 1 for pass

predicted_labels = model.predict(test_data)

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
