from src.config.configuration import ConfigurationManager
from src.utils import logger
from src.exception import CustomException
from src.components import DataExtraction
import sys

STAGE_NAME = "Data Ingestion Stage"

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataExtraction(config=data_ingestion_config, logger=logger)
    data_ingestion.extract_data()
    # data_ingestion.download_file()
    # data_ingestion.unzip_and_clean()


if __name__ == "__main__":
    try:
        logger.info(f"\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys) from e
