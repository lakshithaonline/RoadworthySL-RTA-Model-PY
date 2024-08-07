import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('models/RandomForest_model.joblib')

# List of parameter names
parameters = [
    'Tyres', 'Brakes', 'Suspension', 'Body and Chassis', 'Lights',
    'Glazing', 'Wipers', 'Doors', 'Seat Belts', 'Airbags', 'Speedometer',
    'Exhaust System', 'Fuel System'
]

# Define the weights for each critical level
WEIGHTS = {
    'High': 10,
    'Medium': 6,
    'Low': 4
}

# Define the criticality ranking for each parameter
PARAMETER_CRITICALITY = {
    'Tyres': 'High',
    'Brakes': 'High',
    'Suspension': 'Medium',
    'Body and Chassis': 'Medium',
    'Lights': 'High',
    'Glazing': 'Medium',
    'Wipers': 'Low',
    'Doors': 'Low',
    'Seat Belts': 'Medium',
    'Airbags': 'High',
    'Speedometer': 'Low',
    'Exhaust System': 'Medium',
    'Fuel System': 'High'
}


def calculate_final_score(parameter_scores):
    weighted_scores = [score * WEIGHTS[PARAMETER_CRITICALITY[param]] for param, score in parameter_scores.items()]
    total_weighted_score = sum(weighted_scores)
    max_possible_weighted_score = sum([10 * WEIGHTS[PARAMETER_CRITICALITY[param]] for param in parameter_scores])
    final_percentage_score = (total_weighted_score / max_possible_weighted_score) * 100
    return final_percentage_score


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the form data
            data = {param: float(request.form[param.replace(' ', '_')]) for param in parameters}

            # Calculate the final score
            final_score = calculate_final_score(data)
            data['Final Score'] = final_score

            # Modify the critical concern check to create a list of dictionaries
            high_critical_concern = [
                {'parameter': param, 'score': data[param], 'severity': PARAMETER_CRITICALITY[param]}
                for param in parameters if data[param] < 5]

            # Convert to DataFrame
            test_data = pd.DataFrame([data])

            # Predict the outcome and get the probabilities
            outcome = model.predict(test_data)[0]
            probs = model.predict_proba(test_data)[0]
            confidence_pass = probs[0]
            confidence_fail = probs[1]

            return render_template('index.html',
                                   outcome=outcome,
                                   confidence=max(confidence_pass, confidence_fail),
                                   data=data,
                                   final_score=final_score,
                                   high_critical_concern=high_critical_concern)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Ensure all parameters are provided
        if not all(param in data for param in parameters):
            return jsonify({'error': 'Missing parameters'}), 400

        # Calculate the final score
        final_score = calculate_final_score(data)
        data['Final Score'] = final_score

        # Check for high critical concerns
        high_critical_concern = [{'parameter': param, 'score': data[param], 'severity': PARAMETER_CRITICALITY[param]}
                                 for param in parameters if data[param] < 5]

        # Convert to DataFrame
        test_data = pd.DataFrame([data])

        # Predict the outcome and get the probabilities
        outcome = model.predict(test_data)[0]
        probs = model.predict_proba(test_data)[0]
        confidence_pass = probs[0]
        confidence_fail = probs[1]

        response = {
            'outcome': int(outcome),  # Convert int64 to int
            'confidence': float(max(confidence_pass, confidence_fail)),  # Convert float64 to float
            'data': {k: float(v) if isinstance(v, np.float64) else v for k, v in data.items()},  # Convert float64 to float
            'final_score': float(final_score),  # Convert float64 to float
            'high_critical_concern': high_critical_concern
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
