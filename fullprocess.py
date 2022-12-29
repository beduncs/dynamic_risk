import logging
import json
import ast
from pathlib import Path

from scoring import score_model
from ingestion import main as ingestion
from training import main as training
from deployment import main as deployment
from apicalls import main as apicalls
from reporting import main as reporting

logging.basicConfig(level=logging.INFO)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = Path(config['prod_deployment_path'])
input_folder_path = Path(config['input_folder_path'])
output_folder_path = Path(config['output_folder_path'])
output_model_path = Path(config['output_model_path'])

model_path = prod_deployment_path / 'trainedmodel.pkl'
ingested_record_file = prod_deployment_path / 'ingestedfiles.txt'
prod_score_path = prod_deployment_path / 'latestscore.txt'
score_path = output_model_path / 'latestscore.txt'
ingested_data = output_folder_path / 'finaldata.csv'

# Check and read new data
# first, read ingestedfiles.txt
logging.info(f"Reading file at: {ingested_record_file}")
with open(ingested_record_file, "r") as record_file:
    raw_file = record_file.read()
    current_file_paths = raw_file.split("\n")
logging.debug(f"Current files {current_file_paths}")

logging.info(f"Finding files in {input_folder_path}")
new_file_paths = list(input_folder_path.glob('*.csv'))
logging.debug(f"Found files: {new_file_paths}")

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
differences = set(current_file_paths) - set(new_file_paths)
logging.debug(f"File path list difference: {differences}")

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(differences) > 0:
    logging.info(f"New files found: {differences}")
    logging.info(f"Ingesting the new data.")
    ingestion()
else:
    logging.info(f"No new files, stopping execution.")
    exit()

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
logging.info(f"Reading latest model score from {prod_score_path}")
with open(prod_score_path, "r") as score_file:
    latest_score = ast.literal_eval(score_file.read())
logging.info(f"Latest score: {latest_score}")

# Gather new ingested data
f1score = score_model(model_path, ingested_data, score_path)
logging.info(f"New F1 score: {f1score}")

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if f1score < latest_score:
    logging.info(f"Model drift has been indicated.")
    logging.info(f"Re-training a model on the new data.")
    training()
else:
    logging.info(f"No model drift indicated.")
    exit()

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
logging.info(f"Re-deploying model and metadata.")
deployment()

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
logging.info(f"Calling diagnositic end points of the server.")
apicalls()

logging.info(f"Calling diagnositic end points of the server.")
apicalls()
reporting()
