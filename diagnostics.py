
import pandas as pd
import numpy as np
import timeit
import os
import json
from pathlib import Path
import logging
import pickle
import subprocess

logging.basicConfig(level=logging.DEBUG)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = Path(config['prod_deployment_path'])
test_data_path = Path(config['test_data_path']) / 'testdata.csv'
model_path = Path(config['output_model_path']) / 'trainedmodel.pkl'

# Function to get model predictions


def model_predictions(dataset: pd.DataFrame):
    # read the deployed model and a test dataset, calculate predictions
    deployed_model_path = prod_deployment_path / 'trainedmodel.pkl'
    logging.info(f"Reading model from: {deployed_model_path}")
    with open(deployed_model_path, 'rb') as file:
        model = pickle.load(file)

    # transform input data
    logging.info(f"Transforming input data.")
    logging.debug(f"Input dataset head: {dataset.head()}")
    data_X = dataset[['lastmonth_activity', 'lastyear_activity',
                      'number_of_employees']].values.reshape(-1, 3)

    # form predictions
    logging.info("Using model for inference.")
    predictions = model.predict(data_X)
    logging.debug(f"Predictions: {predictions}")

    return predictions

# Function to get summary statistics


def dataframe_summary(dataset: pd.DataFrame):
    # calculate summary statistics
    logging.info("Calculating summary statistics.")
    numeric_data = dataset[['lastmonth_activity', 'lastyear_activity',
                            'number_of_employees']]
    means = numeric_data.mean().to_list()
    logging.debug(f"Mean statistics: {means}")

    medians = numeric_data.median().to_list()
    logging.debug(f"Median statistics: {medians}")

    stddevs = numeric_data.std().to_list()
    logging.debug(f"Standard deviation statistics: {stddevs}")

    stats_list = [*means, *medians, *stddevs]
    logging.debug(f"Statistics: {stats_list} ")

    return stats_list

# Function to checking missing values


def dataframe_missing(dataset: pd.DataFrame):
    # Check for missing data and calculate percent missing
    logging.info("Calculating missing percent.")

    # find count of na values
    missing_count = dataset.isna().sum()
    logging.debug(f"Missing count: {missing_count}")

    # calculate missing percent for each entry
    miss_per_list = []
    for index_col, missing_num in missing_count.items():
        miss_percent = missing_num / len(dataset[index_col])
        miss_per_list.append(miss_percent)
    logging.debug(f"Missing percent list: {miss_per_list}")
    return miss_per_list

# Function to get timings


def execution_time():
    # calculate timing of training.py and ingestion.py
    logging.info("Calculating execution time.")

    # Define scripts to time
    scripts_to_time = ['ingestion.py', 'training.py']
    timing_list = []

    # Iterate over scripts to time
    for script in scripts_to_time:
        logging.debug(f"Timing {script}.")
        starttime = timeit.default_timer()
        os.system(f'python3 {script}')
        timing = timeit.default_timer() - starttime
        logging.debug(f"{script} executed in {timing} seconds.")
        timing_list.append(timing)
    logging.debug(f"Timing list: {timing_list}")

    return timing_list

# Function to check dependencies


def outdated_packages_list():
    # get a list of outaged packages in use.
    logging.info("Calling pip to check outdated dependecies.")
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    logging.debug(f"Outdated dependencies: {outdated}")

    return outdated


if __name__ == '__main__':
    dataset = pd.read_csv(test_data_path)
    model_predictions(dataset)
    dataframe_summary(dataset)
    dataframe_missing(dataset)
    execution_time()
    outdated_packages_list()
