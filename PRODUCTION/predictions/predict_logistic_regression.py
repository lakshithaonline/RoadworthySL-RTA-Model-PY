import logging

import joblib
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def create_test_data():
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
    return pd.DataFrame([random_values])


def predict_outcome(model, test_data):
    try:
        outcome = model.predict(test_data)
        probs = model.predict_proba(test_data)

        confidence_pass = probs[0][0]
        confidence_fail = probs[0][1]

        return outcome[0], confidence_pass, confidence_fail
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise


def main():
    model_path = '../../PRODUCTION/models/logisticR.joblib'
    model = load_model(model_path)

    test_data = create_test_data()

    predicted_outcome, confidence_pass, confidence_fail = predict_outcome(model, test_data)

    logging.info(f"Predicted outcome: {predicted_outcome}")
    logging.info(f"Confidence (Pass): {confidence_pass:.4f}")
    logging.info(f"Confidence (Fail): {confidence_fail:.4f}")


if __name__ == "__main__":
    main()
