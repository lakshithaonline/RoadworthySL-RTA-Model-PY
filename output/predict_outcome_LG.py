import joblib
import pandas as pd


model = joblib.load('../models/LogisticRegression_model2.joblib')

# Example test data (replace with actual data if available)
random_values = {
    'Tyres': 8,
    'Brakes': 7,
    'Suspension': 9,
    'Body and Chassis': 8,
    'Lights': 7,
    'Glazing': 8,
    'Wipers': 7,
    'Doors': 8,
    'Seat Belts': 9,
    'Airbags': 8,
    'Speedometer': 7,
    'Exhaust System': 8,
    'Fuel System': 7,
    'Final Score': 77.17
}

test_data = pd.DataFrame([random_values])

# Predict the outcome
outcome = model.predict(test_data)
# Predict the outcome and get the probabilities
probs = model.predict_proba(test_data)
confidence_pass = probs[0][0]
confidence_fail = probs[0][1]


# Print the predicted outcome
print("Predicted outcome:", outcome[0])
print("Confidence:", max(confidence_pass, confidence_fail))