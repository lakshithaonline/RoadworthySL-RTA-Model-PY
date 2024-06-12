import random
import csv
from imblearn.over_sampling import SMOTE
import pandas as pd

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
    weighted_scores = [score * WEIGHTS[critical_level] for score, critical_level in parameter_scores]
    total_weighted_score = sum(weighted_scores)
    max_possible_weighted_score = sum([10 * WEIGHTS[critical_level] for _, critical_level in parameter_scores])
    final_percentage_score = (total_weighted_score / max_possible_weighted_score) * 100
    return final_percentage_score


def determine_outcome(final_score, parameter_scores):
    # Check if any high critical level parameter has a score below 4
    high_critical_params_below_4 = sum(1 for score, critical_level in parameter_scores
                                       if critical_level == 'High' and score < 4)

    # Fail if the final score is below 70 or if there are two high critical level parameters below 4
    if final_score < 70 or high_critical_params_below_4 >= 2:
        return 'Failed'
    else:
        return 'Pass'


# def determine_outcome(final_score):
#     # Fail if the final score is below 60
#     if final_score < 70:
#         return 'Failed'
#     else:
#         return 'Pass'

# Generate random data
random_data = []
for _ in range(500):
    data_row = {}
    parameter_scores = []
    for param in parameters:
        score = random.randint(1, 10)
        critical_level = PARAMETER_CRITICALITY[param]
        parameter_scores.append((score, critical_level))
        data_row[param] = score
    final_score = calculate_final_score(parameter_scores)
    data_row['Final Score'] = final_score
    data_row['Outcome'] = determine_outcome(final_score, parameter_scores)
    random_data.append(data_row)

# Convert to DataFrame
df = pd.DataFrame(random_data)

# Separate features and target
X = df[parameters]
y = df['Outcome']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
resampled_df = pd.DataFrame(X_resampled, columns=parameters)
resampled_df['Outcome'] = y_resampled

# Calculate Final Score for resampled data
resampled_data = resampled_df.to_dict('records')
for data_row in resampled_data:
    parameter_scores = [(data_row[param], PARAMETER_CRITICALITY[param]) for param in parameters]
    final_score = calculate_final_score(parameter_scores)
    data_row['Final Score'] = final_score
    data_row['Outcome'] = determine_outcome(final_score, parameter_scores)

# Export to CSV
with open('resampled_data_with_results.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=parameters + ['Final Score', 'Outcome'])
    writer.writeheader()
    for data_row in resampled_data:
        writer.writerow(data_row)

# Print some outputs for verification
for i, data_row in enumerate(resampled_data[:10]):
    print(f"Data set {i + 1}: {data_row['Outcome']} (Final Score: {data_row['Final Score']:.2f})")

print("Resampled data with results has been exported to 'resampled_data_with_results.csv'")
