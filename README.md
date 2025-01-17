# Vehicle Inspection WOF

This project automates vehicle inspection tasks using machine learning.

## Project Structure

The project is organized into the following directories:

**charts (Optional):** This directory may contain charts or visualizations generated by the project.

**data**

* `dta.py`: This script generates the dataset used for training the models.
* `preprocessed_data.csv`: This CSV file stores the preprocessed data version 01.
* `preprocessed_data3.csv`: This CSV file stores the preprocessed data version 02.
* `resampled_data_with_results.csv` This CSV file stores the raw data version generated by ``dta.py``.

**models**

* `LogisticRegression_model2.joblib`: Serialized trained Logistic Regression model version 02.
* `LogisticRegression_model3.joblib`: Serialized trained Logistic Regression model version 03.
* `RandomForest_model.joblib`: Serialized trained Random Forest model version 01. 

**output**

* `finalScore_calculator.py`: calculator that used to get final score according to the formula **Windows form App**.
* `predict_outcome_LG.py`: Predicts outcomes using the Logistic Regression model.
* `predict_outome_RF.py`: Predicts outcomes using the Random Forest model.

**preprocess**

* `preprocess.py`: Preprocesses data for use in the models.

**venv (Likely):** Virtual environment isolating project dependencies (remove from version control).

``LogisticRegression.py``: contains code for training and using the Logistic Regression model.

``RandomForest.py``:  contains code for training and using the Random Forest model.

## External Libraries

List the specific libraries used in the project, e.g., scikit-learn, pandas. Refer to their websites for installation instructions.

## Getting Started

**1. Set Up a Virtual Environment (Recommended):**

   Isolate project dependencies using tools like `venv` .

**2. Install Dependencies:**

   Activate the virtual environment and install required libraries using `pip` (refer to `requirements.txt` if it exists, or examine project code).

**3. Run Data Generation Script:**

   Execute `dta.py` to generate the dataset.

**4. Preprocess the Data:**

   Run `preprocess.py` to preprocess the generated data.

**5. Train the Models:**

   Train Logistic Regression and Random Forest models using `LogisticRegression.py` and `RandomForest.py`, respectively.

**6. Use Output Scripts:**

   Utilize scripts in the `output` directory to calculate final scores and make predictions using the trained models.

## Additional Notes

* This structure is based on the provided information. Refer to the code for specific usage instructions.
* Add comments to your code for better readability and maintainability.
* Consider including a license file (e.g., MIT License) if you plan to share the project publicly.
