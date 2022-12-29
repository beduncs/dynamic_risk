import json
import logging
from pathlib import Path
import pandas as pd

# Function for data ingestion


def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # check for datasets, compile them together, and write to an output file

    # Gathering csv paths
    logging.info('Gathering files for merge.')
    file_paths = list(input_folder_path.glob('*.csv'))
    logging.debug(f'Found paths: {list(file_paths)}')

    # Iterating reads of files
    logging.info('Reading and merging DataFrames.')
    file_list = []
    for file_path in file_paths:
        logging.debug(f'Reading file from: {file_path}')
        file_df = pd.read_csv(file_path)
        file_list.append(file_df)

    # Concatenating df
    combined_df = pd.concat(file_list)
    logging.debug(f'Combined df shape: {combined_df.shape}')
    logging.debug(f'Combined df: {combined_df.head()}')

    # Dropping duplicates
    final_df = combined_df.drop_duplicates()
    logging.debug(f'Final df shape: {final_df.shape}')
    logging.debug(f'Final df: {final_df.head()}')

    # Writing the final df
    output_df_path = output_folder_path / 'finaldata.csv'
    logging.info(f'Writing final df to: {str(output_df_path)}')
    final_df.to_csv(output_df_path)

    # Saving record of filenames
    output_record_path = output_folder_path / 'ingestedfiles.txt'
    logging.info(f'Saving record to: {str(output_record_path)}')
    with open(output_record_path, 'w') as output_record:
        for file_path in file_paths:
            output_record.write(f'{file_path}\n')


def main():

    # Load config.json and get input and output paths
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = Path(config['input_folder_path'])
    output_folder_path = Path(config['output_folder_path'])

    merge_multiple_dataframe(input_folder_path, output_folder_path)


if __name__ == '__main__':
    main()
