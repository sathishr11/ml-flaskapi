from src.config.configuration import ConfigurationManager
from src.utils import logger
from src.exception import CustomException
from src.components.regression_model_training import TrainModel
import sys

STAGE_NAME = "Training Stage for regression"

def main():
    config = ConfigurationManager()
    training_config = config.get_training_config()
    regression_params = config.get_regression_params()
    training_model = TrainModel(
        config=training_config,
        params=regression_params,
        logger=logger
    )
    training_model.train_model()

if __name__ == "__main__":
    try:
        logger.info(f"\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys) from e
