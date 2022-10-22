from src.config.configuration import ConfigurationManager
from src.utils import logger
from src.exception import CustomException
from src.components import DataTransformation
import sys
import os
from pathlib import Path

STAGE_NAME = "Data Transformation Stage"


def main():
    config = ConfigurationManager()
    train_test_split_config = config.get_train_split_config()
    data_transformation_config = config.get_data_transformation_config()
    for file in [train_test_split_config.train_file, train_test_split_config.test_file]:
        data_transformation = DataTransformation(
            config=data_transformation_config,
            file_path=Path(os.path.join(train_test_split_config.train_data_dir, file)),
            logger=logger,
        )
        data_transformation.transform_data()


if __name__ == "__main__":
    try:
        logger.info(f"\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys) from e
