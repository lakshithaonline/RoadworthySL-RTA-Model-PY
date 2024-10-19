import joblib
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model = joblib.load('../models/RandomForest_model.joblib')

test_data = pd.DataFrame([
    {'Tyres': 2, 'Brakes': 3, 'Suspension': 9, 'Body and Chassis': 8, 'Lights': 2,
     'Glazing': 8, 'Wipers': 7, 'Doors': 8, 'Seat Belts': 9, 'Airbags': 8,
     'Speedometer': 7, 'Exhaust System': 8, 'Fuel System': 7, 'Final Score': 70.86},
    {'Tyres': 6, 'Brakes': 7, 'Suspension': 8, 'Body and Chassis': 6, 'Lights': 7,
     'Glazing': 7, 'Wipers': 8, 'Doors': 6, 'Seat Belts': 6, 'Airbags': 7,
     'Speedometer': 7, 'Exhaust System': 7, 'Fuel System': 8, 'Final Score': 80.00},
])

true_labels = [0, 1]

probs = model.predict_proba(test_data)

fpr, tpr, thresholds = roc_curve(true_labels, probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
