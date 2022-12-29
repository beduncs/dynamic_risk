from flask import Flask, session, jsonify, request, make_response
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging

from pathlib import Path
from diagnostics import model_predictions, dataframe_summary, dataframe_missing, execution_time, outdated_packages_list
from scoring import score_model


logging.basicConfig(level=logging.DEBUG)

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path']) / 'finaldata.csv'

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # Get data path from user
    filename = request.args.get('filename')
    filename = Path(filename)

    logging.info(f"Reading data from: {filename}")
    # Check if the provided path exists
    if filename.exists():
        data_df = pd.read_csv(filename)
        logging.debug(f"Data head: {data_df.head()}")
    else:
        raise ValueError("Empty data provided.")

    # Infer with model
    predictions = model_predictions(data_df)

    return make_response(jsonify(predictions), 200)

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    # check the score of the deployed model
    f1score = score_model()

    return make_response(jsonify(f1score), 200)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def sumstats():
    # Read data for test
    logging.info(f'Reading data from: {dataset_csv_path}')
    data_df = pd.read_csv(dataset_csv_path)
    logging.debug(f'DataFrame head: {data_df.head()}')

    # check means, medians, and modes for each column
    stats_list = dataframe_summary(data_df)

    return make_response(jsonify(stats_list), 200)

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    # Read data for test
    logging.info(f'Reading data from: {dataset_csv_path}')
    data_df = pd.read_csv(dataset_csv_path)
    logging.debug(f'DataFrame head: {data_df.head()}')

    # Calculate missing percent per column
    miss_percents = dataframe_missing(data_df)

    # Calculate execution time
    timings = execution_time()

    # Find outdated packages
    outdated_packages = outdated_packages_list()

    return make_response(jsonify([miss_percents, timings, outdated_packages]), 200)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
