import logging
from src.utils import Helper
import pandas as pd
import os
from pathlib import Path
from src.utils import logger
from src.exception import CustomException
from src.entity import DataIngestionConfig
import sys


class DataExtraction:
    def __init__(
        self, config: DataIngestionConfig, logger: logging.Logger = logger
    ) -> None:
        self.helper = Helper()
        self.config = config
        self.logger = logger

    def _combine_multiple_regions(self) -> pd.DataFrame:
        try:
            df_region1 = pd.read_csv(self.config.data_source, nrows=123)
            df_region2 = pd.read_csv(self.config.data_source, skiprows=124)
            region_name1 = df_region1.columns[0]
            region_name2 = df_region2.columns[0]
            # Convert multi index to columns
            df_region1.reset_index(inplace=True)
            df_region2.reset_index(inplace=True)

            # Take column names from first row.
            # Some column contained spaces hence stripping the leading and trailing whitespace
            df_region1.columns = df_region1.iloc[0].str.strip()
            df_region2.columns = df_region2.iloc[0].str.strip()

            # assign the region name for each dataset
            # using insert function so that the order is maintained
            df_region1.insert(0, "Region", region_name1)
            df_region2.insert(0, "Region", region_name2)

            # Take values only from the second row for each row and concat them to single dataframe
            data = pd.concat([df_region1[1:], df_region2[1:]], ignore_index=True)
            # Remove the leading and trailing white spaces in classess
            data["Classes"] = data["Classes"].str.strip()
            # Drop the single row with missing value
            data.dropna(inplace=True)
            data.reset_index(inplace=True, drop=True)

        except Exception as e:
            self.logger.error(CustomException(e, sys))
            raise CustomException(e, sys) from e
        return data

    def extract_data(self) -> None:
        try:
            df = self._combine_multiple_regions()
            # save data set to local
            # create the directory if it does not exist
            self.helper.create_directories([self.config.raw_file_dir], self.logger)
            raw_local_file_path = Path(
                os.path.join(self.config.raw_file_dir, self.config.raw_file_name)
            )
            self.helper.save_dataframe(
                df, raw_local_file_path, self.logger, index=False
            )
        except Exception as e:
            self.logger.error(CustomException(e, sys))
            raise CustomException(e, sys) from e
