import logging
from src.utils import Helper
from src.utils import logger
import os
import sys
import pandas as pd
from src.entity import TrainTestSplitConfig
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from pathlib import Path

class TrainTestSplit():
    def __init__(self, config: TrainTestSplitConfig, logger: logging.Logger=logger):
        self.logger = logger
        self.helper = Helper()
        self.config = config

    def split_data(self):
        try:
            data = pd.read_csv(self.config.raw_file_path)
            train, test = train_test_split(
                data, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )

            train_file_path = Path(
                os.path.join(
                    self.config.train_data_dir,
                    self.config.train_file
                )
            )
            test_file_path = Path(
                os.path.join(
                    self.config.train_data_dir,
                    self.config.test_file
                )
            )

            self.helper.create_directories([self.config.train_data_dir], self.logger)

            for data, data_path in (train, train_file_path), (test, test_file_path):
                self.helper.save_dataframe(data, data_path, self.logger, index=False)


        except Exception as e:
            self.logger.error(CustomException(e, sys))
            raise CustomException(e, sys) from e