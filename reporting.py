import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import metrics
from diagnostics import model_predictions

# Function for reporting


def score_model(test_data_path, matrix_output_path):
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    # Reading data
    logging.info(f"Reading in data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    logging.debug(f"DataFrame head: {test_df.head()}")
    predictions = model_predictions(test_df)
    logging.debug(f"Predictions: {predictions}")
    actual_y = test_df['exited']
    logging.debug(f"Actual y values: {actual_y}")

    # Generate confusion matrix
    logging.info("Generating confusion matrix.")
    cf_matrix = metrics.confusion_matrix(actual_y, predictions)
    logging.debug(f"Confusion matrix: {cf_matrix}")

    # Generate confusion matrix image
    sns.heatmap(cf_matrix, annot=True)

    # Write confusion matrix image
    logging.info(f"Saving confusion matrix to: {matrix_output_path}")
    plt.savefig(matrix_output_path)


def main():
    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = Path(config['test_data_path']) / 'testdata.csv'
    matrix_output_path = Path(
        config['output_model_path']) / 'confusionmatrix.png'

    score_model(test_data_path, matrix_output_path)


if __name__ == '__main__':
    main()
