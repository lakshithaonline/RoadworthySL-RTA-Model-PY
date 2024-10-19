import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(data):
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Encode categorical 'Outcome' values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logging.info("Data preprocessed successfully.")
    return X, y_encoded


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Data split into training and testing sets: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")


def save_model(model, model_path):
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}.")


def main():
    file_path = '../../../data/preprocessed_data3.csv'

    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, '../../models/RandomF.joblib')


if __name__ == "__main__":
    main()
