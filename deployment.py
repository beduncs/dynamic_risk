import json
import shutil
import logging
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = Path(config['output_model_path']) / 'trainedmodel.pkl'
score_path = Path(config['output_model_path']) / 'latestscore.txt'
ingest_record_path = Path(config['output_folder_path']) / 'ingestedfiles.txt'
prod_deployment_path = Path(config['prod_deployment_path'])
deployed_model_path = prod_deployment_path / model_path.name
deployed_score_path = prod_deployment_path / score_path.name
deployed_ingest_path = prod_deployment_path / ingest_record_path.name

# function for deployment


def migrate_deployment_files():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    logging.info("Copying deployment files.")

    # copying model
    logging.debug(f"Copying model from {model_path} to {deployed_model_path}")
    shutil.copy(model_path, deployed_model_path)

    # copying score record
    logging.debug(
        f"Copying score record from {score_path} to {deployed_score_path}")
    shutil.copy(score_path, deployed_score_path)

    # copying ingest record
    logging.debug(
        f"Copying model from {ingest_record_path} to {deployed_ingest_path}")
    shutil.copy(ingest_record_path, deployed_ingest_path)


if __name__ == '__main__':
    migrate_deployment_files()
