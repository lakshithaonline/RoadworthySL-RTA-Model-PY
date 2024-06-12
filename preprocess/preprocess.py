import pandas as pd


def preprocess_csv(input_file, output_file):
    data = pd.read_csv(input_file)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Save the preprocessed data
    data.to_csv(output_file, index=False)


# Example usage
if __name__ == "__main__":
    preprocess_csv('../data/resampled_data_with_results.csv', '../data/preprocessed_data.csv')
