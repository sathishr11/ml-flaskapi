from src.config.configuration import ConfigurationManager
from src.utils import logger
from src.exception import CustomException
from src.components import TrainTestSplit
import sys

STAGE_NAME = "Train test split Stage"


def main():
    config = ConfigurationManager()
    train_test_split_config = config.get_train_split_config()
    train_test_split = TrainTestSplit(config=train_test_split_config, logger=logger)
    train_test_split.split_data()


if __name__ == "__main__":
    try:
        logger.info(f"\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys) from e
