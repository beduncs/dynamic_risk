import json
import logging
import pickle
import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.DEBUG)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path']) / 'finaldata.csv'
model_path = Path(config['output_model_path']) / 'trainedmodel.pkl'


# Function for training the model
def train_model():
    logging.info("Initializing the model.")

    # gather the data
    logging.info(f"Reading the data from: {dataset_csv_path}.")
    training_data = pd.read_csv(dataset_csv_path)
    logging.debug(f"Data head: {training_data.head()}")
    training_X = training_data[['lastmonth_activity', 'lastyear_activity',
                                'number_of_employees']].values.reshape(-1, 3)
    training_y = training_data['exited'].values.reshape(-1, 1)

    # use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)

    # fit the logistic regression to your data
    logging.info("Fitting the model.")
    training_start = time.time()
    model = lr.fit(training_X, training_y)
    training_end = time.time()
    logging.debug(f"Time to fit: {training_end - training_start}")

    # write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Saving the model to: {model_path}")
    with open(model_path, 'wb') as filehandler:
        pickle.dump(model, filehandler)


if __name__ == '__main__':
    train_model()
