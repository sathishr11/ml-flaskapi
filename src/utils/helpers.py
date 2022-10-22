import yaml
import os
import json
import pandas as pd
import argparse
import logging
from src.constants import CONFIG_FILE_PATH
from src.constants import PARAMS_FILE_PATH
from ensure import ensure_annotations
from pathlib import Path
from box import ConfigBox


class Helper():
    def __init__(self):
        self.config_path = None
        self.params_path = None

    def load_args(self)-> None:
        args = argparse.ArgumentParser()
        args.add_argument("--config", "-c", default=CONFIG_FILE_PATH)
        args.add_argument("--params", "-p", default=PARAMS_FILE_PATH)
        parsed_args = args.parse_args()
        self.config_path = parsed_args.config
        self.params_path = parsed_args.params
        
    @staticmethod    
    @ensure_annotations
    def read_yaml(path_to_yaml: Path)-> ConfigBox:
        """Read yaml file

        Args:
            path_to_yaml (str): Path of the yaml file

        Returns:
            dict: Contents of yaml file as dictionary
        """
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)

        return ConfigBox(content)

    @staticmethod
    @ensure_annotations
    def create_directories(dirs: list, logger: logging.Logger):
        """Create a directory if it does not exist

        Args:
            dirs (list): List of directories to be created
        """
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f'Directory {dir_path} created')

    @staticmethod
    @ensure_annotations
    def save_dataframe(data: pd.DataFrame , data_path: Path, logger: logging.Logger, index: bool =False):
        """Save the dataframe to a csv file

        Args:
            data (pandas dataframe): dataframe that needs to be saved
            data_path (string): directory where the dataframe needs to be saved
            index (bool, optional): index parameter for to_csv. Defaults to False.
        """
        data.to_csv(data_path, index=index)
        logger.info(f'Data saved to {data_path}')

    @staticmethod
    @ensure_annotations
    def save_evaluation_reports(reports: dict, reports_path: Path):
        """Save the evaluation reports to a csv file

        Args:
            reports (dict): dictionary of evaluation reports
            reports_path (string): directory where the evaluation reports needs to be saved
        """
        with open(reports_path, 'w') as f:
            # for key, value in reports.items():
            #     f.write(f'{key}: {value}\n')
            json.dump(reports, f, indent=4)
        print(f'Evaluation reports saved to {reports_path}')
