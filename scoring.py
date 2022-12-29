import pickle
import logging
import json
import pandas as pd
from sklearn import metrics
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = Path(config['test_data_path']) / 'testdata.csv'
model_path = Path(config['output_model_path']) / 'trainedmodel.pkl'
score_path = Path(config['output_model_path']) / 'latestscore.txt'

# Function for model scoring


def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    # reading the test data
    logging.info(f"Reading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    test_X = test_data[['lastmonth_activity', 'lastyear_activity',
                        'number_of_employees']].values.reshape(-1, 3)
    test_y = test_data['exited'].values.reshape(-1, 1)

    # reading model
    logging.info(f"Reading test data from: {model_path}")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # score model
    logging.info(f"Scoring model.")
    predicted = model.predict(test_X)
    logging.debug(f"Predicted y values: {predicted}")
    logging.debug(f"Actual y values: {test_y}")
    f1score = metrics.f1_score(predicted, test_y)
    logging.debug(f"f1 Score: {f1score}")

    # write out latest score
    logging.debug(f"Writing latest score to: {score_path}")
    with open(score_path, 'w') as score_file:
        score_file.write(f'{f1score}\n')

    return f1score


if __name__ == '__main__':
    score_model()
