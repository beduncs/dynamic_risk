import requests
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = Path(config['test_data_path']) / 'testdata.csv'
api_return_path = Path(config['output_model_path']) / 'apireturns.txt'

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"
PORT = 8000


# Call each API endpoint and store the responses
logging.info("Making requests to each end point.")

response1 = requests.post(
    f'{URL}:{PORT}/prediction?filename={test_data_path}').content
response2 = requests.get(f'{URL}:{PORT}/scoring').content
response3 = requests.get(f'{URL}:{PORT}/summarystats').content
response4 = requests.get(f'{URL}:{PORT}/diagnostics').content

# combine all API responses
responses = [response1, response2, response3, response4]
logging.debug(f"Respones: {responses}")

# write the responses to your workspace
logging.info(f"Writing api return respones to: {api_return_path}")
with open(api_return_path, 'w') as api_output:
    api_output.write(f'{responses}\n')
