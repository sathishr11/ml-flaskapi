from src.utils import Helper
import os
import sys
import pandas as pd
import logging
from src.utils import logger
from src.entity import DataTransformationConfig
from src.exception import CustomException
from pathlib import Path


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        file_path: Path,
        logger: logging.Logger = logger,
    ):
        self.logger = logger
        self.helper = Helper()
        self.config = config
        self.file_path = file_path
        self.helper.create_directories([self.config.transformed_data_dir], self.logger)

    def transform_data(self):
        try:

            data = pd.read_csv(self.file_path)
            integer_features = self.config.integer_features
            float_features = self.config.float_features
            # convert the datatypes of the integer features to int
            data[integer_features] = data[integer_features].astype(int)
            # convert the datatypes of the float features to float
            data[float_features] = data[float_features].astype(float)

            # Apply map
            for column in self.config.map_function:
                data[column] = data[column].replace(self.config.map_function[column])
            # save data set to local
            # create the directory if it does not exist
            transformed_file_path = Path(
                os.path.join(
                    self.config.transformed_data_dir, os.path.basename(self.file_path)
                )
            )
            self.helper.save_dataframe(
                data, transformed_file_path, self.logger, index=False
            )

        except Exception as e:
            self.logger.error(CustomException(e, sys))
            raise CustomException(e, sys) from e
